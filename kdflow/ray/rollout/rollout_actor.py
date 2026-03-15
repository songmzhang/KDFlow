"""
RolloutActor: A Ray remote actor that manages a single SGLang HTTP server instance.
Port allocation is done centrally by RolloutGroup via _get_current_node_ip_and_free_port().
"""

import ipaddress
import multiprocessing
import socket
import time
from typing import Optional

import ray
import requests
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree

from kdflow.utils.logging_utils import init_logger

logger = init_logger(__name__)


@ray.remote
class RolloutRayActor:
    """Ray remote actor wrapping a single SGLang HTTP server."""

    def __init__(self, rank: int, base_gpu_id: int = 0):
        self.rank = rank
        self.base_gpu_id = base_gpu_id
        self.process: Optional[multiprocessing.Process] = None
        self.server_host: Optional[str] = None
        self.server_port: Optional[int] = None
        self.node_rank: int = 0
        self.router_ip: Optional[str] = None
        self.router_port: Optional[int] = None

    def get_node_ip(self) -> str:
        return self._get_node_ip()

    @staticmethod
    def _get_current_node_ip_and_free_port(start_port: int = 15000, consecutive: int = 1):
        """Return the node IP and a free port (called remotely by RolloutGroup for port allocation)."""
        return RolloutRayActor._get_node_ip(), RolloutRayActor._get_free_port(start_port=start_port, consecutive=consecutive)

    def init(
        self,
        model_path: str,
        router_ip: str,
        router_port: int,
        tp_size: int = 1,
        host: str = None,
        port: int = None,
        nccl_port: int = None,
        dist_init_addr: str = None,
        enable_memory_saver: bool = True,
        mem_fraction_static: Optional[float] = None,
        extra_server_args: Optional[dict] = None,
    ):
        """Initialize and launch the SGLang HTTP server, then register with the router."""
        assert host is not None, "host must be provided by RolloutGroup"
        assert port is not None, "port must be provided by RolloutGroup"
        assert nccl_port is not None, "nccl_port must be provided by RolloutGroup"
        assert dist_init_addr is not None, "dist_init_addr must be provided by RolloutGroup"

        self.router_ip = router_ip
        self.router_port = router_port
        self.server_host = self._format_ipv6(host)
        self.server_port = port

        ip_part, port_part = dist_init_addr.rsplit(":", 1)
        dist_init_addr = f"{self._format_ipv6(ip_part)}:{port_part}"

        server_args_dict = {
            "model_path": model_path,
            "trust_remote_code": True,
            "host": self.server_host,
            "port": self.server_port,
            "nccl_port": nccl_port,
            "dist_init_addr": dist_init_addr,
            "tp_size": tp_size,
            "base_gpu_id": self.base_gpu_id,
            "gpu_id_step": 1,
            "node_rank": 0,
            "nnodes": 1,
            "enable_memory_saver": enable_memory_saver,
            "enable_weights_cpu_backup": enable_memory_saver,
            "skip_server_warmup": True,
            "log_level": "warning",
            "log_level_http": "warning",
        }
        if mem_fraction_static is not None:
            server_args_dict["mem_fraction_static"] = mem_fraction_static

        if extra_server_args:
            server_args_dict.update(extra_server_args)

        self.node_rank = server_args_dict.get("node_rank", 0)

        logger.info(
            f"[RolloutActor {self.rank}] Launching server at {self.server_host}:{self.server_port} "
            f"(base_gpu_id={self.base_gpu_id}, tp_size={tp_size})"
        )
        self.process = self._launch_sglang_server(ServerArgs(**server_args_dict))

        if self.node_rank == 0:
            self._register_with_router()
            logger.info(f"[RolloutActor {self.rank}] Registered with router at {router_ip}:{router_port}")

    def _register_with_router(self):
        """Register this server with the SGLang router."""
        worker_url = f"http://{self.server_host}:{self.server_port}"
        try:
            response = requests.post(
                f"http://{self.router_ip}:{self.router_port}/workers",
                json={"url": worker_url, "worker_type": "regular"},
            )
            response.raise_for_status()
        except Exception:
            response = requests.post(
                f"http://{self.router_ip}:{self.router_port}/add_worker?url={worker_url}"
            )
            response.raise_for_status()

    def _make_request(self, endpoint: str, payload: Optional[dict] = None, method: str = "POST"):
        """Make an HTTP request to the local SGLang server."""
        if self.node_rank != 0:
            return

        url = f"http://{self.server_host}:{self.server_port}/{endpoint}"
        if method == "POST":
            response = requests.post(url, json=payload or {})
        else:
            response = requests.get(url)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            e.add_note(f"{response.text=}")
            raise
        return response.json()

    def sleep(self, tags: Optional[list] = None):
        """Release GPU memory occupation (offload to CPU)."""
        self.flush_cache()
        return self._make_request("release_memory_occupation", {"tags": tags} if tags else {})

    def wakeup(self, tags: Optional[list] = None):
        """Resume GPU memory occupation (reload from CPU)."""
        return self._make_request("resume_memory_occupation", {"tags": tags} if tags else {})

    def flush_cache(self):
        """Flush the KV cache on the server."""
        if self.node_rank != 0:
            return
        for _ in range(60):
            try:
                response = requests.get(f"http://{self.server_host}:{self.server_port}/flush_cache")
                if response.status_code == 200:
                    return
            except Exception as e:
                logger.warning(f"[RolloutActor {self.rank}] Error flushing cache: {e}")
                time.sleep(1)
        raise TimeoutError("Timeout while flushing cache.")

    def update_weights_from_tensor(
        self,
        serialized_named_tensors: list,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ):
        """Update model weights from serialized tensor data."""
        return self._make_request(
            "update_weights_from_tensor",
            {
                "serialized_named_tensors": serialized_named_tensors,
                "load_format": load_format,
                "flush_cache": flush_cache,
            },
        )

    def update_weights_from_disk(
        self,
        model_path: str,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ):
        """Update model weights from disk via sglang's update_weights_from_disk API."""
        return self._make_request(
            "update_weights_from_disk",
            {
                "model_path": model_path,
                "load_format": load_format,
                "flush_cache": flush_cache,
            },
        )

    def health_check(self, timeout: float = 5.0) -> bool:
        """Check if the SGLang server is healthy."""
        if self.node_rank != 0:
            return True
        try:
            response = requests.get(
                f"http://{self.server_host}:{self.server_port}/health_generate",
                timeout=timeout,
            )
            return response.status_code == 200
        except Exception:
            return False

    def get_server_url(self) -> str:
        """Return the URL of this server."""
        return f"http://{self.server_host}:{self.server_port}"

    def shutdown(self):
        """Shutdown the SGLang server subprocess and deregister from the router."""
        if self.process is None:
            return

        if self.node_rank == 0:
            try:
                worker_url = f"http://{self.server_host}:{self.server_port}"
                try:
                    all_workers = requests.get(
                        f"http://{self.router_ip}:{self.router_port}/workers"
                    ).json()["workers"]
                    for worker in all_workers:
                        if worker["url"] == worker_url:
                            worker_id = worker["id"]
                            requests.delete(
                                f"http://{self.router_ip}:{self.router_port}/workers/{worker_id}"
                            )
                            break
                except Exception:
                    requests.post(
                        f"http://{self.router_ip}:{self.router_port}/remove_worker?url={worker_url}"
                    )
            except Exception as e:
                logger.warning(f"[RolloutActor {self.rank}] Failed to deregister from router: {e}")

        try:
            kill_process_tree(self.process.pid)
        except Exception as e:
            logger.warning(f"[RolloutActor {self.rank}] Failed to kill server process: {e}")

        self.process = None
        logger.info(f"[RolloutActor {self.rank}] Shutdown complete")

    # ---- Static helper methods ----

    @staticmethod
    def _format_ipv6(addr: str) -> str:
        """Wrap IPv6 addresses with brackets for URL compatibility."""
        if not addr or addr.startswith("["):
            return addr
        try:
            if ipaddress.ip_address(addr).version == 6:
                return f"[{addr}]"
        except ValueError:
            pass
        return addr

    @staticmethod
    def _is_port_available(port: int) -> bool:
        """Check if a port is available."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return True
        except OSError:
            return False

    @staticmethod
    def _get_free_port(start_port: int = 15000, consecutive: int = 1) -> int:
        """Find consecutive free ports starting from start_port."""
        port = start_port
        while not all(RolloutRayActor._is_port_available(port + i) for i in range(consecutive)):
            port += 1
            if port >= 65535:
                raise RuntimeError("No free port found")
        return port

    @staticmethod
    def _get_node_ip() -> str:
        """Get current node IP address."""
        return ray._private.services.get_node_ip_address().strip("[]")

    @staticmethod
    def _launch_sglang_server(server_args: ServerArgs) -> Optional[multiprocessing.Process]:
        """Launch SGLang HTTP server in a subprocess."""
        from sglang.srt.entrypoints.http_server import launch_server

        multiprocessing.set_start_method("spawn", force=True)
        server_args.host = server_args.host.strip("[]")
        p = multiprocessing.Process(target=launch_server, args=(server_args,))
        p.start()

        if server_args.node_rank != 0:
            return p

        # Wait for server to be healthy
        RolloutRayActor._wait_server_healthy(
            base_url=server_args.url(),
            is_process_alive=lambda: p.is_alive(),
        )
        return p

    @staticmethod
    def _wait_server_healthy(base_url: str, is_process_alive, timeout: float = 600.0):
        """Wait until the SGLang server is healthy and ready to serve requests."""
        start_time = time.time()
        with requests.Session() as session:
            while True:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Server at {base_url} failed to become healthy within {timeout}s")
                try:
                    response = session.get(f"{base_url}/health_generate")
                    if response.status_code == 200:
                        break
                except requests.RequestException:
                    pass
                if not is_process_alive():
                    raise RuntimeError("Server process terminated unexpectedly.")
                time.sleep(2)

            # Flush cache to ensure clean state
            while True:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Server at {base_url} failed to flush cache within {timeout}s")
                try:
                    response = session.get(f"{base_url}/flush_cache")
                    if response.status_code == 200:
                        break
                except requests.RequestException:
                    pass
                if not is_process_alive():
                    raise RuntimeError("Server process terminated unexpectedly.")
                time.sleep(2)

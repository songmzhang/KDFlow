import asyncio
import base64
import io
import logging
import multiprocessing
import random
import socket
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import ray
import requests
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from kdflow.ray.rollout.rollout_actor import RolloutRayActor
from kdflow.utils.logging_utils import init_logger

logger = init_logger(__name__)


NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
    "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
    "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
    "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
]

SGLANG_ENV_VARS = {
    "SGL_JIT_DEEPGEMM_PRECOMPILE": "false",
    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
    "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
    "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
    "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
    "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
    "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
}


class RolloutActorGroup:
    """
    Manages a group of SGLang rollout actors behind a load-balancing router.

    Args:
        model_path: Path to the HuggingFace model checkpoint
        num_actors: Number of rollout actors (SGLang servers) to create
        tp_size: Tensor parallel size per actor
        num_gpus_per_node: Number of GPUs per physical node
        enable_memory_saver: Enable memory saver for sleep/wakeup support
        mem_fraction_static: Static memory fraction for SGLang servers
        max_concurrent: Max concurrent requests to router during generation
        num_gpus_per_actor: Ray GPU resources per actor (fractional for co-location)
        pg: 3-tuple (pg, reordered_bundle_indices, reordered_gpu_ids), PlacementGroup, or None
        extra_server_args: Additional SGLang ServerArgs overrides
    """

    def __init__(
        self,
        model_path: str,
        num_actors: int = 1,
        tp_size: int = 1,
        num_gpus_per_node: int = 8,
        enable_memory_saver: bool = True,
        mem_fraction_static: Optional[float] = None,
        max_concurrent: int = 64,
        num_gpus_per_actor: float = 0.2,
        pg: Optional[Union[PlacementGroup, Tuple[PlacementGroup, list, list]]] = None,
        extra_server_args: Optional[dict] = None,
    ):
        self.model_path = model_path
        self.num_actors = num_actors
        self.tp_size = tp_size
        self.num_gpus_per_node = num_gpus_per_node
        self.enable_memory_saver = enable_memory_saver
        self.mem_fraction_static = mem_fraction_static
        self.max_concurrent = max_concurrent
        self.extra_server_args = extra_server_args or {}

        self.num_gpus_per_actor_engine = tp_size

        if pg is not None and isinstance(pg, tuple):
            self._pg, self._reordered_bundle_indices, self._reordered_gpu_ids = pg
        elif pg is not None:
            self._pg = pg
            total_gpus = num_actors * self.num_gpus_per_actor_engine
            self._reordered_bundle_indices = list(range(total_gpus))
            self._reordered_gpu_ids = list(range(total_gpus))
        else:
            self._pg = None
            self._reordered_bundle_indices = None
            self._reordered_gpu_ids = None

        self.router_ip = self._get_node_ip()
        self.router_port = self._find_available_port(random.randint(3000, 4000))
        self.router_process = self._start_sglang_router(self.router_ip, self.router_port)
        self.router_url = f"http://{self._format_host(self.router_ip)}:{self.router_port}"

        self.actors: List[ray.actor.ActorHandle] = []
        self._create_actors(num_gpus_per_actor)
        self._init_actors()

        logger.info(
        f"RolloutGroup initialized: {num_actors} actors (tp={tp_size}), "
            f"router at {self._format_host(self.router_ip)}:{self.router_port}"
        )

    def _create_actors(self, num_gpus_per_actor: float):
        """Create Ray remote RolloutRayActor instances with proper GPU binding."""
        env_vars = {
            name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
        }
        env_vars.update(SGLANG_ENV_VARS)

        num_gpu_per_engine = min(self.num_gpus_per_actor_engine, self.num_gpus_per_node)

        for i in range(self.num_actors):
            if self._reordered_gpu_ids is not None:
                base_gpu_id = int(self._reordered_gpu_ids[i * num_gpu_per_engine])
            else:
                base_gpu_id = (i * num_gpu_per_engine) % self.num_gpus_per_node

            options = {
                "num_cpus": num_gpus_per_actor,
                "num_gpus": num_gpus_per_actor,
                "runtime_env": {
                    "env_vars": env_vars,
                },
            }

            if self._pg is not None and self._reordered_bundle_indices is not None:
                options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                    placement_group=self._pg,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=self._reordered_bundle_indices[i * num_gpu_per_engine],
                )

            actor = RolloutRayActor.options(**options).remote(
                rank=i,
                base_gpu_id=base_gpu_id,
            )
            self.actors.append(actor)

    def _allocate_addr_and_ports(self) -> List[dict]:
        """Centrally allocate host, port, nccl_port, and dist_init_addr for all actors."""
        num_gpu_per_engine = min(self.num_gpus_per_actor_engine, self.num_gpus_per_node)
        num_engines_per_node = max(1, self.num_gpus_per_node // num_gpu_per_engine)
        addr_and_ports = [{} for _ in range(self.num_actors)]

        visited_nodes = set()
        for rank in range(self.num_actors):
            node_index = rank // num_engines_per_node
            if node_index in visited_nodes:
                continue
            visited_nodes.add(node_index)

            # Number of engines on this node starting from this rank
            num_engines_on_this_node = min(
                num_engines_per_node - (rank % num_engines_per_node),
                self.num_actors - rank,
            )

            actor = self.actors[rank]
            start_port = 15000

            def get_port(consecutive=1):
                nonlocal start_port
                _, port = ray.get(
                    actor._get_current_node_ip_and_free_port.remote(
                        start_port=start_port,
                        consecutive=consecutive,
                    )
                )
                start_port = port + consecutive
                return port

            def get_addr():
                addr, _ = ray.get(
                    actor._get_current_node_ip_and_free_port.remote()
                )
                return addr

            for j in range(num_engines_on_this_node):
                current_rank = rank + j
                if current_rank >= self.num_actors:
                    break
                addr_and_ports[current_rank]["host"] = get_addr()
                addr_and_ports[current_rank]["port"] = get_port()
                addr_and_ports[current_rank]["nccl_port"] = get_port()

            for j in range(num_engines_on_this_node):
                current_rank = rank + j
                if current_rank >= self.num_actors:
                    break
                dist_init_port = get_port(30)
                addr_and_ports[current_rank]["dist_init_addr"] = (
                    f"{self._format_host(addr_and_ports[current_rank]['host'])}:{dist_init_port}"
                )

        for i in range(self.num_actors):
            for key in ["host", "port", "nccl_port", "dist_init_addr"]:
                assert key in addr_and_ports[i], f"Actor {i} missing '{key}' in port allocation"
            logger.info(f"Ports for actor {i}: {addr_and_ports[i]}")

        return addr_and_ports

    def _init_actors(self):
        """Allocate ports centrally, then initialize all actors in parallel."""
        addr_and_ports = self._allocate_addr_and_ports()
        init_refs = []
        for i, actor in enumerate(self.actors):
            ref = actor.init.remote(
                model_path=self.model_path,
                router_ip=self.router_ip,
                router_port=self.router_port,
                tp_size=self.tp_size,
                host=addr_and_ports[i]["host"],
                port=addr_and_ports[i]["port"],
                nccl_port=addr_and_ports[i]["nccl_port"],
                dist_init_addr=addr_and_ports[i]["dist_init_addr"],
                enable_memory_saver=self.enable_memory_saver,
                mem_fraction_static=self.mem_fraction_static,
                extra_server_args=self.extra_server_args,
            )
            init_refs.append(ref)

        ray.get(init_refs)
        logger.info(f"All {self.num_actors} rollout actors initialized and registered with router")

    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[Dict[str, Any]] = None,
        image_data: Optional[List] = None,
    ) -> List[Dict[str, Any]]:
        """Generate responses for a batch of prompts via the SGLang router."""
        if sampling_params is None:
            sampling_params = {
                "temperature": 1.0,
                "max_new_tokens": 2048,
            }

        generate_url = f"{self.router_url}/generate"
        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(
                self._async_generate(
                    router_url=generate_url,
                    prompts=prompts,
                    sampling_params=sampling_params,
                    max_concurrent=self.max_concurrent,
                    image_data=image_data,
                )
            )
        finally:
            loop.close()

        return results

    def sleep(self, tags: Optional[list] = None):
        """Release GPU memory on all rollout actors (offload to CPU)."""
        refs = [actor.sleep.remote(tags=tags) for actor in self.actors]
        ray.get(refs)
        # logger.info(f"All {self.num_actors} rollout actors have gone to sleep (tags={tags})")

    def wakeup(self, tags: Optional[list] = None):
        """Resume GPU memory on all rollout actors (reload from CPU)."""
        refs = [actor.wakeup.remote(tags=tags) for actor in self.actors]
        ray.get(refs)
        # logger.info(f"All {self.num_actors} rollout actors have woken up (tags={tags})")

    def health_check(self) -> List[bool]:
        """Check health of all rollout actors."""
        refs = [actor.health_check.remote() for actor in self.actors]
        return ray.get(refs)

    def shutdown(self):
        """Shutdown all actors and the router."""
        refs = [actor.shutdown.remote() for actor in self.actors]
        try:
            ray.get(refs, timeout=30)
        except Exception as e:
            logger.warning(f"Error during actor shutdown: {e}")

        if self.router_process and self.router_process.is_alive():
            self.router_process.terminate()
            self.router_process.join(timeout=5)
            if self.router_process.is_alive():
                self.router_process.kill()

        logger.info("RolloutGroup shutdown complete")

    @staticmethod
    def _get_node_ip() -> str:
        """Get current node IP address."""
        return ray._private.services.get_node_ip_address().strip("[]")

    @staticmethod
    def _format_host(host: str) -> str:
        if ":" in host and not host.startswith("["):
            return f"[{host}]"
        return host

    @staticmethod
    def _find_available_port(start_port: int = 3000) -> int:
        """Find an available port starting from the given port."""
        for port in range(start_port, 65535):
            try:
                with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
                    s.bind(("::", port))
                    return s.getsockname()[1]
            except OSError:
                continue
        raise RuntimeError("No free port found")

    @staticmethod
    def _start_sglang_router(host: str, port: int) -> multiprocessing.Process:
        """Start the SGLang router as a daemon process."""
        from sglang_router.launch_router import RouterArgs, launch_router

        router_args = RouterArgs(host=host, port=port)
        router_args.policy = "round_robin"
        if hasattr(router_args, "log_level"):
            router_args.log_level = "warn"

        logger.info(f"Launching SGLang router at {RolloutActorGroup._format_host(host)}:{port}")

        def _run_router(args):
            launch_router(args)

        process = multiprocessing.Process(target=_run_router, args=(router_args,))
        process.daemon = True
        process.start()

        # Wait for router to be ready
        time.sleep(3)
        if not process.is_alive():
            raise RuntimeError("SGLang router process died during startup")

        logger.info(f"SGLang router launched successfully at {RolloutActorGroup._format_host(host)}:{port}")
        return process

    @staticmethod
    def _encode_image_to_base64(image) -> str:
        """Convert a PIL Image to a base64-encoded string for SGLang API."""
        from PIL import Image

        if isinstance(image, str):
            return image  # Already a base64 string or URL
        if isinstance(image, Image.Image):
            buffer = io.BytesIO()
            fmt = "PNG" if image.mode == "RGBA" else "JPEG"
            image.save(buffer, format=fmt)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        raise TypeError(f"Unsupported image type: {type(image)}")

    @staticmethod
    async def _async_generate(
        router_url: str,
        prompts: List[str],
        sampling_params: Dict[str, Any],
        max_concurrent: int = 64,
        image_data: Optional[List] = None,
    ) -> List[Dict[str, Any]]:
        """Send generation requests to the SGLang router asynchronously."""
        import aiohttp

        semaphore = asyncio.Semaphore(max_concurrent)
        results = [None] * len(prompts)

        async def _generate_one(idx: int, prompt: str, session: aiohttp.ClientSession):
            payload = {
                "text": prompt,
                "sampling_params": sampling_params,
            }
            if image_data and image_data[idx] is not None:
                img = image_data[idx]
                if isinstance(img, list):
                    payload["image_data"] = [RolloutActorGroup._encode_image_to_base64(im) for im in img]
                else:
                    payload["image_data"] = RolloutActorGroup._encode_image_to_base64(img)
            async with semaphore:
                async with session.post(router_url, json=payload) as resp:
                    resp.raise_for_status()
                    output = await resp.json()
                    results[idx] = output

        connector = aiohttp.TCPConnector(limit=max_concurrent)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                asyncio.create_task(_generate_one(i, prompt, session))
                for i, prompt in enumerate(prompts)
            ]
            await asyncio.gather(*tasks)

        return results

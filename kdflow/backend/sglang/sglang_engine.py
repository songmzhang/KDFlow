import os
import multiprocessing as mp
from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
import torch
from sglang.srt.entrypoints.engine import Engine as _SglEngine
from sglang.srt.managers.scheduler import run_scheduler_process as _original_run_scheduler_process

_DEFAULT_SHM_POOL_SIZE = 1024 * 2048 * 4096


def _patched_run_scheduler_process(*args, **kwargs):
    try:
        from kdflow.backend.sglang.monkey_patch import apply_patch
        apply_patch()
    except Exception as e:
        print(f"[PatchedEngine] WARNING: Failed to apply monkey patch (PID={os.getpid()}): {e}", flush=True)
    return _original_run_scheduler_process(*args, **kwargs)


class PatchedEngine(_SglEngine):
    """
    SGLang Engine that applies monkey patch in scheduler subprocesses.
    Motivation: SGLang Engine supports returning hidden states, but the existing implementation use .tolist() to convert hidden states from GPU tensor to Python list, which is very inefficient. This monkey patch replaces the original .tolist() with a more efficient operation .numpy().
    """
    run_scheduler_process_func = staticmethod(_patched_run_scheduler_process)


@dataclass
class EngineConfig:
    """Configuration for SGLang Engine."""
    model_path: str
    tp_size: int = 1
    ep_size: int = 1
    pp_size: int = 1
    chunked_prefill_size: int = -1
    disable_radix_cache: bool = True
    enable_return_hidden_states: bool = True
    enable_memory_saver: bool = True
    enable_weights_cpu_backup: bool = True
    mem_fraction_static: float = 0.8
    quantization: str = None
    offload_tags: Optional[str] = "all"
    base_gpu_id: int = 0
    shm_pool_size: int = _DEFAULT_SHM_POOL_SIZE


def _engine_worker(config: EngineConfig, request_queue: Queue, response_queue: Queue):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    engine = None
    shm_pool = None

    try:
        # Create shared memory pool
        shm_pool_name = f"sglang_hs_pool_{os.getpid()}"
        try:
            old = SharedMemory(name=shm_pool_name)
            old.close()
            old.unlink()
        except FileNotFoundError:
            pass
        shm_pool = SharedMemory(name=shm_pool_name, create=True, size=config.shm_pool_size)

        engine = PatchedEngine(
            model_path=config.model_path,
            tp_size=config.tp_size,
            ep_size=config.ep_size,
            pp_size=config.pp_size,
            chunked_prefill_size=config.chunked_prefill_size,
            disable_radix_cache=config.disable_radix_cache,
            enable_return_hidden_states=config.enable_return_hidden_states,
            enable_memory_saver=config.enable_memory_saver,
            enable_weights_cpu_backup=config.enable_weights_cpu_backup,
            quantization=config.quantization,
            mem_fraction_static=config.mem_fraction_static,
            base_gpu_id=config.base_gpu_id,
        )

        response_queue.put({"type": "init_done", "success": True, "shm_pool_name": shm_pool_name})

        while True:
            request = request_queue.get()
            if request is None:
                break

            req_type = request.get("type")

            try:
                if req_type == "generate":
                    _handle_generate(engine, request, shm_pool, shm_pool_name,
                                     request_queue, response_queue)
                elif req_type == "sleep":
                    _handle_sleep(engine, request, config, response_queue)
                elif req_type == "wakeup":
                    _handle_wakeup(engine, request, config, response_queue)
                else:
                    response_queue.put({"type": req_type, "success": False,
                                        "error": f"Unknown request type: {req_type}"})
            except Exception:
                import traceback
                response_queue.put({"type": req_type, "success": False,
                                    "error": traceback.format_exc()})

    except Exception:
        import traceback
        response_queue.put({"type": "init_done", "success": False,
                            "error": traceback.format_exc()})
    finally:
        if shm_pool:
            try:
                shm_pool.close()
                shm_pool.unlink()
            except Exception:
                pass
        if engine:
            try:
                engine.shutdown()
            except Exception:
                pass
        print("[SGLangEngineWorker] Worker process exiting")


def _normalize_tags(tags):
    """Convert tags to the format SGLang expects (None, or list of strings)."""
    if tags is None or tags == "all":
        return None
    if isinstance(tags, str):
        return [tags]
    return tags


def _handle_generate(engine, request, shm_pool, shm_pool_name,
                     request_queue, response_queue):
    """Handle a generate request: run inference and write hidden states to shared memory."""
    kwargs = request["kwargs"]
    generate_kwargs = {
        "input_ids": kwargs["input_ids"],
        "sampling_params": kwargs["sampling_params"],
        "return_hidden_states": kwargs.get("return_hidden_states", True),
    }
    if kwargs.get("image_data") is not None:
        generate_kwargs["image_data"] = kwargs["image_data"]
    outputs = engine.generate(**generate_kwargs)

    offsets_meta = []
    current_offset = 0

    for output, mask in zip(outputs, kwargs["loss_masks"]):
        hs_np = output["meta_info"]["hidden_states"][0]
        hs_np = hs_np[mask]
        if not hs_np.flags['C_CONTIGUOUS']:
            hs_np = np.ascontiguousarray(hs_np)

        shm_pool.buf[current_offset:current_offset + hs_np.nbytes] = hs_np.tobytes()
        offsets_meta.append({
            "offset": current_offset,
            "shape": hs_np.shape,
            "dtype": str(hs_np.dtype),
            "nbytes": hs_np.nbytes,
        })
        current_offset += hs_np.nbytes

    response_queue.put({
        "type": "generate",
        "success": True,
        "shm_pool_name": shm_pool_name,
        "offsets_meta": offsets_meta,
    })

    # Wait for consumer to finish reading shared memory
    cleanup_signal = request_queue.get()
    assert cleanup_signal and cleanup_signal.get("type") == "cleanup_shm"


def _handle_sleep(engine, request, config, response_queue):
    """Handle a sleep request: offload GPU memory."""
    tags = request.get("tags", config.offload_tags)
    engine.release_memory_occupation(tags=_normalize_tags(tags))
    response_queue.put({"type": "sleep", "success": True, "tags": tags})


def _handle_wakeup(engine, request, config, response_queue):
    """Handle a wakeup request: restore GPU memory."""
    torch.cuda.empty_cache()
    tags = request.get("tags", config.offload_tags)
    engine.resume_memory_occupation(tags=_normalize_tags(tags))
    response_queue.put({"type": "wakeup", "success": True, "tags": tags})


class SGLangEngineService:
    """Manages SGLang Engine in a subprocess with shared memory communication."""

    def __init__(
        self,
        config: EngineConfig,
        batch_size: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        hidden_dim: Optional[int] = None,
    ):
        # Dynamically compute shm_pool_size if all three params are provided
        if batch_size is not None and max_seq_len is not None and hidden_dim is not None:
            # float32 = 4 bytes per element
            config.shm_pool_size = batch_size * max_seq_len * hidden_dim * 4
        self.config = config
        self.process: Optional[mp.Process] = None
        self.request_queue: Optional[Queue] = None
        self.response_queue: Optional[Queue] = None
        self._started = False
        self._shm_pool: Optional[SharedMemory] = None

    def start(self, timeout: float = 1800.0):
        """Start the SGLang Engine in a subprocess."""
        if self._started:
            raise RuntimeError("Service already started")

        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        self.request_queue = mp.Queue()
        self.response_queue = mp.Queue()

        self.process = mp.Process(
            target=_engine_worker,
            args=(self.config, self.request_queue, self.response_queue),
        )
        self.process.start()

        try:
            response = self.response_queue.get(timeout=timeout)
            if response.get("type") == "init_done" and response.get("success"):
                self._started = True
                self._shm_pool = SharedMemory(name=response["shm_pool_name"])
                print(f"[SGLangEngineService] Engine started, shm pool: {response['shm_pool_name']}")
            else:
                raise RuntimeError(f"Init failed: {response.get('error')}")
        except Exception as e:
            self._cleanup()
            raise RuntimeError(f"Engine initialization failed: {e}")

    def generate(
        self,
        input_ids: List[List[int]],
        loss_masks: List[np.ndarray],
        sampling_params: Dict[str, Any],
        return_hidden_states: bool = True,
        image_data=None,
    ) -> List[np.ndarray]:
        """Run generation and return hidden states via shared memory."""
        if not self._started:
            raise RuntimeError("Service not started")

        kwargs = {
            "input_ids": input_ids,
            "loss_masks": loss_masks,
            "sampling_params": sampling_params,
            "return_hidden_states": return_hidden_states,
        }
        if image_data is not None:
            kwargs["image_data"] = image_data

        self.request_queue.put({
            "type": "generate",
            "kwargs": kwargs,
        })

        response = self.response_queue.get()
        if not response.get("success"):
            raise RuntimeError(f"Generate failed: {response.get('error')}")

        # Read hidden states from shared memory
        hidden_states = []
        for meta in response.get("offsets_meta", []):
            hs = np.ndarray(
                tuple(meta["shape"]),
                dtype=np.dtype(meta["dtype"]),
                buffer=self._shm_pool.buf[meta["offset"]:meta["offset"] + meta["nbytes"]],
            ).copy()
            hidden_states.append(hs)

        self.request_queue.put({"type": "cleanup_shm"})
        return hidden_states

    def sleep(self, tags: Optional[str] = "all"):
        """Release GPU memory."""
        if not self._started:
            return
        self.request_queue.put({"type": "sleep", "tags": tags})
        response = self.response_queue.get()
        if not response.get("success"):
            raise RuntimeError(f"Sleep failed: {response.get('error')}")
        return response.get("tags")

    def wakeup(self, tags: Optional[str] = "all"):
        """Resume GPU memory."""
        if not self._started:
            return
        self.request_queue.put({"type": "wakeup", "tags": tags})
        response = self.response_queue.get()
        if not response.get("success"):
            raise RuntimeError(f"Wakeup failed: {response.get('error')}")
        return response.get("tags")

    def shutdown(self):
        """Shutdown the subprocess gracefully."""
        if not self._started:
            return
        self._started = False
        self._cleanup()
        print("[SGLangEngineService] Service shutdown complete")

    def _cleanup(self):
        """Clean up subprocess, queues and shared memory."""
        if self._shm_pool:
            try:
                self._shm_pool.close()
            except Exception:
                pass
            self._shm_pool = None

        if self.request_queue:
            try:
                self.request_queue.put(None)
            except Exception:
                pass

        if self.process:
            self.process.join(timeout=30)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
                if self.process.is_alive():
                    self.process.kill()

        self.process = None
        self.request_queue = None
        self.response_queue = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

import os
import pickle
import multiprocessing as mp
from multiprocessing import Queue
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import zmq
import numpy as np
import torch
from sglang.srt.entrypoints.engine import Engine as _SglEngine
from sglang.srt.managers.scheduler import run_scheduler_process as _original_run_scheduler_process


os.environ["SGLANG_JIT_DEEPGEMM_FAST_WARMUP"] = "true"

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
    # for multi-node tp/pp
    nnodes: int = 1
    node_rank: int = 0
    dist_init_addr: Optional[str] = None


def _engine_worker(config: EngineConfig, request_queue: Queue, response_queue: Queue):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if config.nnodes > 1:
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"

    engine = None
    zmq_ctx = None

    try:
        zmq_ctx = zmq.Context()
        data_socket = zmq_ctx.socket(zmq.PUSH)
        zmq_ipc_addr = f"ipc:///tmp/sglang_hs_{os.getpid()}"
        data_socket.bind(zmq_ipc_addr)
        
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
            nnodes=config.nnodes,
            node_rank=config.node_rank,
            dist_init_addr=config.dist_init_addr,
        )

        response_queue.put({
            "type": "init_done", 
            "success": True, 
            "zmq_ipc_addr": zmq_ipc_addr
        })

        while True:
            request = request_queue.get()
            if request is None:
                break

            req_type = request.get("type")

            try:
                if req_type == "generate":
                    _handle_generate(engine, request, data_socket, request_queue, response_queue)
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
        if zmq_ctx:
            try:
                data_socket.close()
                zmq_ctx.term()
            except Exception:
                pass
        if engine:
            try:
                engine.shutdown()
            except Exception:
                pass


def _normalize_tags(tags):
    """Convert tags to the format SGLang expects (None, or list of strings)."""
    if tags is None or tags == "all":
        return None
    if isinstance(tags, str):
        return [tags]
    return tags


def _handle_generate(engine, request, data_socket, request_queue, response_queue):
    """Handle a generate request: run inference and send hidden states via ZMQ."""
    kwargs = request["kwargs"]

    generate_kwargs = {
        "prompt": kwargs["prompt"],
        "sampling_params": kwargs["sampling_params"],
        "return_hidden_states": kwargs.get("return_hidden_states", True),
    }
    if kwargs.get("image_data") is not None:
        generate_kwargs["image_data"] = kwargs["image_data"]

    outputs = engine.generate(**generate_kwargs)

    num_samples = len(outputs)
    
    response_queue.put({
        "type": "generate",
        "success": True,
        "num_samples": num_samples,
    })
    
    for i, (output, mask) in enumerate(zip(outputs, kwargs["loss_masks"])):
        hs_np = output["meta_info"]["hidden_states"][0]
        hs_np = hs_np[:mask.shape[0]]  # loss_mask may have been truncated
        hs_np = hs_np[mask]
        if not hs_np.flags['C_CONTIGUOUS']:
            hs_np = np.ascontiguousarray(hs_np)
            
        meta = pickle.dumps({"shape": hs_np.shape, "dtype": str(hs_np.dtype)})
        data_socket.send(meta, flags=zmq.SNDMORE)
        data_socket.send(hs_np, copy=False)


def _handle_sleep(engine, request, config, response_queue):
    """Handle a sleep request: offload GPU memory."""
    tags = request.get("tags", config.offload_tags)
    torch.cuda.empty_cache()
    engine.release_memory_occupation(tags=_normalize_tags(tags))
    response_queue.put({"type": "sleep", "success": True, "tags": tags})


def _handle_wakeup(engine, request, config, response_queue):
    """Handle a wakeup request: restore GPU memory."""
    tags = request.get("tags", config.offload_tags)
    torch.cuda.empty_cache()
    engine.resume_memory_occupation(tags=_normalize_tags(tags))
    response_queue.put({"type": "wakeup", "success": True, "tags": tags})


class SGLangEngineService:
    """Manages SGLang Engine in a subprocess with ZMQ communication."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.process: Optional[mp.Process] = None
        self.request_queue: Optional[Queue] = None
        self.response_queue: Optional[Queue] = None
        self._started = False
        self._zmq_ctx: Optional[zmq.Context] = None
        self._data_socket = None

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
                self._zmq_ctx = zmq.Context()
                self._data_socket = self._zmq_ctx.socket(zmq.PULL)
                self._data_socket.connect(response["zmq_ipc_addr"])
            else:
                raise RuntimeError(f"Init failed: {response.get('error')}")
        except Exception as e:
            self._cleanup()
            raise RuntimeError(f"Engine initialization failed: {e}")

    def generate(
        self,
        prompt: List[str],
        loss_masks: List[np.ndarray],
        sampling_params: Dict[str, Any],
        return_hidden_states: bool = True,
        image_data=None,
    ) -> List[np.ndarray]:
        """Run generation and return hidden states via ZMQ.
        
        Args:
            prompt: List of raw text prompts. SGLang handles tokenization internally.
            loss_masks: Pre-computed boolean masks for selecting response hidden states.
            sampling_params: Sampling parameters (e.g. max_new_tokens=0 for prefill-only).
            return_hidden_states: Whether to return hidden states.
            image_data: Optional list of image data for multimodal models.
        """
        if not self._started:
            raise RuntimeError("Service not started")

        # Check if subprocess is still alive before sending request
        if self.process and not self.process.is_alive():
            raise RuntimeError(
                f"[SGLangEngineService] Engine subprocess (PID={self.process.pid}) is dead! "
                f"exitcode={self.process.exitcode}"
            )

        kwargs = {
            "prompt": prompt,
            "loss_masks": loss_masks,
            "sampling_params": sampling_params,
            "return_hidden_states": return_hidden_states,
        }
        if image_data is not None:
            kwargs["image_data"] = image_data

        self.request_queue.put({"type": "generate", "kwargs": kwargs})

        response = self.response_queue.get()
        if not response.get("success"):
            raise RuntimeError(f"Generate failed: {response.get('error')}")

        # Read hidden states via ZMQ
        num_samples = response["num_samples"]
        hidden_states = []
        for _ in range(num_samples):
            meta_bytes = self._data_socket.recv()
            data_bytes = self._data_socket.recv()
            meta = pickle.loads(meta_bytes)
            hs = np.frombuffer(data_bytes, dtype=np.dtype(meta["dtype"])).reshape(meta["shape"])
            hidden_states.append(hs.copy())  # copy because zmq buffer will be reused

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


    def _cleanup(self):
        """Clean up subprocess, queues and shared memory."""
        if self._data_socket:
            try:
                self._data_socket.close()
            except Exception:
                pass
            self._data_socket = None
        if self._zmq_ctx:
            try:
                self._zmq_ctx.term()
            except Exception:
                pass
            self._zmq_ctx = None

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

import os
import time

import ray
import torch
import numpy as np
from transformers import AutoConfig

from kdflow.datasets.utils import load_images
from kdflow.utils.utils import remove_pad_token
from kdflow.backend.sglang.sglang_engine import SGLangEngineService, EngineConfig
from kdflow.utils.logging_utils import init_logger

logger = init_logger(__name__)


@ray.remote
class TeacherRayActor:
    """
    TeacherRayActor: Responsible for teacher model forward (prefilling) using SGLang Engine.
    
    Key design: Teacher and Student SHARE the same GPUs via PlacementGroup co-location.
    - TeacherRayActor is scheduled on PG bundles with RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
    - GPU binding is managed via base_gpu_id parameter passed to SGLang Engine
    - This allows Teacher and Student to share the same GPUs via PlacementGroup
    """
    
    def __init__(self, strategy, base_gpu_id: int = 0):
        """
        Initialize TeacherRayActor.
        
        Args:
            strategy: Training strategy containing configuration args
            base_gpu_id: Base GPU device ID for SGLang Engine binding (e.g., 0, 1, 2, ...)
                        Used with RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES for PG co-location.
        """
        logger.info(f"[TeacherRayActor] __init__ STARTED, PID={os.getpid()}, base_gpu_id={base_gpu_id}")
        
        self.strategy = strategy
        self.tp_size = strategy.args.kd.teacher_tp_size
        self.ep_size = strategy.args.kd.teacher_ep_size
        self.pp_size = strategy.args.kd.teacher_pp_size
        self.base_gpu_id = base_gpu_id
        
        # Disable tokenizers parallelism to avoid deadlock with multiprocessing
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Get hidden_dim from teacher model config for shm_pool_size calculation
        teacher_config = AutoConfig.from_pretrained(
            strategy.args.model.teacher_name_or_path, trust_remote_code=True
        )
        hidden_dim = teacher_config.hidden_size
        # Each teacher actor processes: global_batch_size * forward_n_batches / dp_size samples
        batch_size = (
            strategy.args.train.train_batch_size
            * strategy.args.kd.teacher_forward_n_batches
            // strategy.args.kd.teacher_dp_size
        )
        max_seq_len = strategy.args.data.max_len
        
        # Create engine configuration
        # GPU binding is handled by base_gpu_id (works with RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES)
        self.engine_config = EngineConfig(
            model_path=strategy.args.model.teacher_name_or_path,
            tp_size=self.tp_size,
            ep_size=self.ep_size,
            pp_size=self.pp_size,
            chunked_prefill_size=-1,  # Disable chunked prefill for full sequence processing
            disable_radix_cache=True,  # Disable cache for deterministic behavior
            enable_return_hidden_states=True,  # Enable hidden states extraction
            enable_memory_saver=True,  # Enable memory saving mode
            enable_weights_cpu_backup=True,  # Backup weights to CPU for memory release
            quantization=strategy.args.kd.teacher_quantization,
            mem_fraction_static=strategy.args.kd.teacher_mem_fraction_static,
            offload_tags=strategy.args.kd.teacher_offload_tags,
            base_gpu_id=self.base_gpu_id,
        )
        
        # Initialize SGLang Engine service (runs in subprocess)
        self.engine_service = SGLangEngineService(
            self.engine_config,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            hidden_dim=hidden_dim,
        )
        self.engine_service.start()
        
        if self.strategy.args.kd.teacher_enable_sleep:
            logger.info(f"[TeacherRayActor] Teacher sleep after initialization")
            self.engine_service.sleep(tags=self.strategy.args.kd.teacher_offload_tags)
        
        logger.info(f"[TeacherRayActor] Initialized with tp_size={self.tp_size}, ep_size={self.ep_size}, pp_size={self.pp_size}")

    def ready(self):
        """Return True when the actor is ready (engine service started)."""
        return self.engine_service._started

    def forward(self, global_batch, batch_indices):
        """
        Perform forward pass (prefilling) on the given batches.
        
        Args:
            global_batch: List of all micro-batches
            batch_indices: List of batch indices this actor should process
        
        Returns:
            List of (batch_idx, micro-batch with teacher_hiddens) tuples and return timestamp
        """
        # === Phase 1: Preprocessing (unpadding) ===
        batches = [global_batch[i] for i in batch_indices]
        mbsz = batches[0]["tea_input_ids"].shape[0]
        unpadded_input_ids, unpadded_loss_mask = [], []
        for micro_batch in batches:
            (
                input_ids, attn_mask, loss_mask
            ) = (
                micro_batch["tea_input_ids"], micro_batch["tea_attn_mask"], micro_batch["tea_loss_mask"]
            )
            unpadded_input_ids.extend(remove_pad_token(input_ids, attn_mask, return_tensors=False))
            unpadded_loss_mask.extend(remove_pad_token(loss_mask, attn_mask, return_tensors=True))
        unpadded_loss_mask = [m.numpy().astype(bool) for m in unpadded_loss_mask]
        
        # === Phase 2: SGLang Generate ===
        image_data = None
        if batches[0].get("image_paths") is not None:
            all_image_paths = sum((micro_batch["image_paths"] for micro_batch in batches), [])
            image_data = [load_images(paths) if paths else None for paths in all_image_paths]
        hidden_states_list = self.engine_service.generate(
            input_ids=unpadded_input_ids,
            loss_masks=unpadded_loss_mask,
            sampling_params={"max_new_tokens": 0},
            return_hidden_states=True,
            image_data=image_data,
        )
        
        # Process in micro-batch groups with vectorized operations
        sample_idx = 0
        results_with_indices = []  # List of (original_batch_idx, batch_with_hiddens)
        for mb_idx, original_batch_idx in enumerate(batch_indices):
            mb_hidden_np = hidden_states_list[sample_idx: sample_idx + mbsz]
            mb_hidden_np = np.concatenate(mb_hidden_np, axis=0)
            batches[mb_idx]["teacher_hiddens"] = mb_hidden_np
            results_with_indices.append((original_batch_idx, batches[mb_idx]))
            sample_idx += mbsz
        
        return results_with_indices
    
    def sleep(self):
        """Release GPU memory occupation, move weights to CPU."""
        self.engine_service.sleep(tags=self.strategy.args.kd.teacher_offload_tags)
        
    def wakeup(self):
        """Resume GPU memory occupation, move weights back to GPU."""
        self.engine_service.wakeup(tags=self.strategy.args.kd.teacher_offload_tags)
    
    def shutdown(self):
        """Shutdown the engine service."""
        self.engine_service.shutdown()
        logger.info("[TeacherRayActor] Shutdown complete")
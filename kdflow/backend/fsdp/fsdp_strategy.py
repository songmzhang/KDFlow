import os
import functools
import logging
import torch

import torch.distributed as dist
import torch.nn as nn
import transformers
from abc import ABC
from datetime import timedelta
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
from torch.optim import Optimizer, AdamW
from torch.distributed.fsdp import (
    fully_shard,
    MixedPrecisionPolicy,
    CPUOffloadPolicy,
)
from torchdata.stateful_dataloader import StatefulDataLoader

from kdflow.backend.fsdp import checkpoint, weight_sync
from kdflow.utils.logging_utils import init_logger
from kdflow.models import DistillModel
from kdflow.utils.distributed_sampler import DistributedSampler
from kdflow.models.ring_attn_utils import get_ring_attn_group, set_ring_attn_group


class FSDP2Strategy(ABC):
    def __init__(
        self,
        seed: int = 42,
        full_determinism: bool = False,
        max_norm: float = 0.0,
        micro_train_batch_size=1,
        train_batch_size=1,
        bf16=True,
        cpu_offload: bool = False,
        args=None,
    ) -> None:
        self.args = args
        self.checkpoint_names: list[str] = []
        self.seed = seed
        self.max_norm = max_norm
        self.micro_train_batch_size = micro_train_batch_size
        self.train_batch_size = train_batch_size
        self.cpu_offload = cpu_offload
        self.bf16 = bf16
        self.full_determinism = full_determinism
        self.logger = init_logger("FSDP2Strategy")
        
    def log(self, msg: str, level: str = "info", rank_0_only: bool = True) -> None:
        """
        Log a message with rank information.
        
        Args:
            msg: The message to log.
            level: Log level ('debug', 'info', 'warning', 'error', 'critical').
            rank_0_only: If True, only log on rank 0.
        """
        if rank_0_only and not self.is_rank_0():
            return
        
        rank = self.get_rank() if dist.is_initialized() else 0
        full_msg = f"[Rank {rank}] {msg}"
        
        log_func = getattr(self.logger, level.lower(), self.logger.info)
        # Use stacklevel=2 to show the caller's location instead of this function's location
        log_func(full_msg, stacklevel=2)
        
    def setup_distributed(self, timeout=timedelta(minutes=60)) -> None:
        if self.full_determinism:
            transformers.enable_full_determinism(self.seed)
            # Use deterministic backward in flash attention as, by default, flash attention uses atomic adds
            # https://github.com/Dao-AILab/flash-attention/commit/732654583c2e640adc012ecb60e460bf19dcd9e3
            transformers.modeling_flash_attention_utils.deterministic_g = True
        else:
            transformers.set_seed(self.seed)
            
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank != -1:
            torch.cuda.set_device(local_rank)
        
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", timeout=timeout)
            
        self.world_size = dist.get_world_size()
        self.sp_size = self.args.model.ring_attn_size
        dp_size = self.world_size // self.sp_size
        self.sp_mesh = init_device_mesh("cuda", (dp_size, self.sp_size), mesh_dim_names=("dp", "sp"))
        self.setup_ring_attn(self.sp_mesh)
        if self.args.fsdp.fsdp_size > -1:
            assert self.world_size % self.args.fsdp.fsdp_size == 0
            assert self.world_size >= self.args.fsdp.fsdp_size
            replicate = self.world_size // self.args.fsdp.fsdp_size
            self.fsdp_mesh = init_device_mesh(
                "cuda", (replicate, self.args.fsdp.fsdp_size), mesh_dim_names=("replicate", "sharded"))
        else:
            self.fsdp_mesh = init_device_mesh(
                "cuda", (self.world_size,), mesh_dim_names=("sharded",)
            )
        
        self.grad_accum_step = 0
        self.accumulated_gradient = (
            self.train_batch_size
            * self.sp_size
            // self.micro_train_batch_size
            // self.world_size
        )
        
    def setup_ring_attn(self, device_mesh):
        if self.sp_size == 1:
            self.ring_attn_rank = 0
            return
        
        group = device_mesh["sp"].get_group()
        self.ring_attn_rank = dist.get_rank(group=group)
        set_ring_attn_group(group)
        
        from ring_flash_attn import substitute_hf_flash_attn
        
        self.ring_head_stride = getattr(self.args.model, "ring_head_stride", 1)
        substitute_hf_flash_attn(self.ring_attn_group, self.ring_head_stride)  
    
    @property
    def ring_attn_group(self):
        return get_ring_attn_group()
    
    def setup_dataloader(
        self,
        replay_buffer,
        batch_size: int,
        pin_memory: bool = False,
        shuffle=True,
        collate_fn=None,
        drop_last=True,
        sampler=None,
        consumed_samples=0,
    ):
        # DDP only mode, replay buffers on each rank are different.
        if sampler is None and dist.is_initialized():
            num_replicas = dist.get_world_size() // self.sp_size
            rank = dist.get_rank() // self.sp_size

            sampler = DistributedSampler(
                replay_buffer,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=shuffle,
                seed=self.seed,
                drop_last=drop_last,
                consumed_samples=consumed_samples,
            )
        return StatefulDataLoader(
            replay_buffer,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            shuffle=shuffle if sampler is None else False,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )
        
    def load_hf_model(self, model_class, model_name_or_path, attn_impl, model_config):
        init_context = self._get_init_weight_context_manager(model_config)
        
        with init_context():
            model = model_class.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                attn_implementation=attn_impl,
                torch_dtype=torch.bfloat16 if self.args.train.bf16 else "auto",
            )
        # Patch lm_head to skippable mode so that the HF forward does not
        # materialize full-sequence logits. The algorithm layer always calls
        # `lm_head(hidden, skip=False)` explicitly via chunked_loss.
        if hasattr(model, "lm_head"):
            original_forward = model.lm_head.forward
            def skip_forward(x, skip=True):
                if skip:
                    return x
                return original_forward(x)
            model.lm_head.forward = skip_forward
            self.log("lm_head forward set to skippable mode for student model.")
        return model
    
    def _get_init_weight_context_manager(self, model_config):
        """Get context manager for model initialization.

        Returns a callable that creates a context manager.
        Uses meta device (no memory allocation) for non-rank-0 processes,
        UNLESS tie_word_embeddings=True (which causes hangs with meta tensors).

        Ref: verl/utils/fsdp_utils.py::get_init_weight_context_manager
        NOTE: tie_word_embedding causes meta_tensor init to hang
        """
        from accelerate import init_empty_weights

        # Check if model uses tied word embeddings (which doesn't work with meta tensors)
        use_meta_tensor = not model_config.tie_word_embeddings

        def cpu_init_weights():
            return torch.device("cpu")

        if use_meta_tensor:
            # Rank 0: CPU, others: meta device (memory efficient for large models)
            return init_empty_weights if not self.is_rank_0() else cpu_init_weights
        else:
            self.log("tie_word_embeddings=True, loading full model to CPU on all ranks")
            return cpu_init_weights
    
    def prepare(self, model: nn.Module, *args, **kwargs):
        for name, param in model.named_parameters():
            if param.requires_grad:
                model = model.float()
                break
        
        # Only rank 0 has real weights, others may have meta tensors
        # _fsdp2_load_full_state_dict will broadcast from rank 0
        full_state = model.state_dict()
        self._init_fsdp_kwargs()
        model = self._fsdp2_shard_model(model)
        model = self._fsdp2_load_full_state_dict(
            model, full_state, self.fsdp_mesh, 
            cpu_offload=True if self.args.fsdp.cpu_offload else None
        )
        
        return model
        
    def _init_fsdp_kwargs(self):
        if self.bf16:
            self.mixed_precision_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                cast_forward_inputs=True
            )
        else:
            self.mixed_precision_policy = None
        
        self.offload_policy = CPUOffloadPolicy(pin_memory=True) if self.args.fsdp.cpu_offload else None
        
        self.fsdp_kwargs = {
            "mesh": self.fsdp_mesh,
            "mp_policy": self.mixed_precision_policy,
            "offload_policy": self.offload_policy
        }
    
    def _fsdp2_shard_model(self, model):
        fsdp_transformer_layer_cls_to_wrap = model.model._no_split_modules
        for _, module in model.model.named_modules():
            if module.__class__.__name__ in fsdp_transformer_layer_cls_to_wrap or (
                isinstance(module, torch.nn.Embedding) and not model.model.config.tie_word_embeddings
            ):
                fully_shard(module, **self.fsdp_kwargs)
        fully_shard(model, **self.fsdp_kwargs)
        return model
    
    def _fsdp2_load_full_state_dict(self, model, full_state, device_mesh, cpu_offload):
        """Load full state dict into FSDP2 model with efficient broadcast from rank 0.

        This function loads weights from rank 0 and broadcasts to all other ranks,
        avoiding the need for each rank to load the full model from disk.

        Args:
            model: FSDP2-wrapped model
            full_state: State dict (only rank 0 has real weights, others have empty dict)
            device_mesh: Device mesh for FSDP
            cpu_offload: If not None, enables StateDictOptions cpu_offload

        Ref:verl/utils/fsdp_utils.py::fsdp2_load_full_state_dict
        """
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

        # Step 1: Allocate tensors on GPU
        # Rank 0: move with weights, others: allocate empty tensors on device (for meta tensors)
        if self.is_rank_0():
            model = model.to(device=torch.cuda.current_device(), non_blocking=True)
        else:
            # to_empty creates tensors on device without initializing memory
            model = model.to_empty(device=torch.cuda.current_device())

        is_cpu_offload = cpu_offload is not None
        options = StateDictOptions(full_state_dict=True, cpu_offload=is_cpu_offload, broadcast_from_rank0=True)

        # Step 2: Broadcast weights from rank 0 to all ranks
        set_model_state_dict(model, full_state, options=options)

        # set_model_state_dict will not broadcast buffers, so we need to broadcast them manually.
        for _name, buf in sorted(
            model.named_buffers(),
            key=lambda item: item[0],
        ):
            dist.broadcast(buf, src=0)

        if is_cpu_offload:
            model.to("cpu", non_blocking=True)
            for buf in model.buffers():
                buf.data = buf.data.to(torch.cuda.current_device())

        return model
    
    def create_optimizer(self, model, **kwargs) -> Optimizer:
        if isinstance(model, DistillModel):
            model = model.model
        
        kwargs["fused"] = True
        optim_params = self._get_optimizer_grouped_parameters(model, kwargs["weight_decay"])
        optim = AdamW(optim_params, **kwargs)
        return optim
    
    def _get_optimizer_grouped_parameters(
        self,
        model,
        weight_decay,
        no_decay_name_list=["bias", "layer_norm.weight", "layernorm.weight", "norm.weight", "ln_f.weight"],
    ):
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (not any(nd in n for nd in no_decay_name_list) and p.requires_grad)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters

    def _unwrap_model(self, model) -> nn.Module:
        if hasattr(model, "module"):
            return self._unwrap_model(model.module)
        elif isinstance(model, DistillModel):
            return model.model
        else:
            return model
        
    def backward(self, loss: torch.Tensor, model: nn.Module, optimizer: Optimizer, **kwargs) -> None:
        self.grad_accum_step = (self.grad_accum_step + 1) % self.accumulated_gradient
        loss = loss / self.accumulated_gradient
        loss.backward()
    
    def optimizer_step(
        self,
        optimizer: Optimizer,
        model: nn.Module,
        scheduler,
        **kwargs,
    ) -> None:
        if self.grad_accum_step == 0:
            if self.max_norm > 0.0:
                if hasattr(model, "clip_grad_norm_"):
                    model.clip_grad_norm_(self.max_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)
            
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()
            
    def load_model(self, model: nn.Module, model_dir, strict: bool = False) -> None:
        return checkpoint.load_model(self, model, model_dir, strict=strict)
    
    def save_model(self, model: nn.Module, output_dir, **kwargs) -> None:
        return checkpoint.save_model(self, model, output_dir, **kwargs)

    def save_checkpoint(self, model: nn.Module, epoch: int, global_step: int, **kwargs):
        return checkpoint.save_checkpoint(self, model, epoch, global_step, **kwargs)

    def load_checkpoint(self, model: nn.Module, checkpoint_path, **kwargs):
        return checkpoint.load_checkpoint(self, model, checkpoint_path, **kwargs)
    
    def all_reduce(self, data, op="mean"):
        assert op in ("mean", "max", "sum")
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_reduce(v, op)
            return ret
        else:
            is_tensor = True
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
                is_tensor = False
            is_cpu_tensor = data.device.type == "cpu"

            if is_cpu_tensor:
                data = data.to(torch.cuda.current_device())
            if op == "mean":
                data /= self.world_size
            dist.all_reduce(data, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM)
            if is_cpu_tensor:
                data = data.cpu()
            return data.item() if not is_tensor else data

    def all_gather(self, data):
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_gather(v)
            return ret
        else:
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
            is_cpu_tensor = data.device.type == "cpu"

            ret = [torch.zeros_like(data).to(torch.cuda.current_device()) for _ in range(self.world_size)]
            dist.all_gather(ret, data.to(torch.cuda.current_device()))
            return torch.cat(ret).cpu() if is_cpu_tensor else torch.cat(ret)

    def print(self, *msg):
        if self.is_rank_0():
            print(*msg)
            
    def is_rank_0(self) -> bool:
        if not dist.is_initialized():
            return True
        return dist.get_rank() == 0
    
    def get_rank(self) -> int:
        if not dist.is_initialized():
            return 0
        return dist.get_rank()
    
    @torch.no_grad()
    def offload_model_params(self, model, empty_cache: bool = True):
        model.cpu()
        if empty_cache:
            torch.cuda.empty_cache()
            
    @torch.no_grad()
    def onload_model_params(self, model):
        device = torch.cuda.current_device()
        model.to(device)
        
    @torch.no_grad()
    def offload_optim_states(self, optimizer: Optimizer, empty_cache: bool = True):
        """offload optimizer states to CPU"""
        if not optimizer.state:
            return
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                state = optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to("cpu", non_blocking=True)
        if empty_cache:
            torch.cuda.empty_cache()

    @torch.no_grad()
    def onload_optim_states(self, optimizer: Optimizer):
        """Move optimizer states to GPU."""
        device = torch.cuda.current_device()
        if not optimizer.state:
            return
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                state = optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device, non_blocking=True)

    def export_lora_adapter(self, model, adapter_name: str = "default") -> dict | None:
        return weight_sync.export_lora_adapter(self, model, adapter_name=adapter_name)

    def update_rollout_weights_from_tensor(
        self, model, engine, gather_src, gather_group,
        update_weight_buffer_size=2 * 1024**3,
        weight_source=None
    ):
        return weight_sync.update_rollout_weights_from_tensor(
            self, model, engine, gather_src, gather_group,
            update_weight_buffer_size=update_weight_buffer_size,
            weight_source=weight_source,
        )

    def update_rollout_weights_from_disk(self, model, rollout_engines, tokenizer=None, tmp_dir=None):
        return weight_sync.update_rollout_weights_from_disk(
            self, model, rollout_engines, tokenizer=tokenizer, tmp_dir=tmp_dir,
        )

import os
import gc
import functools
import logging
import torch

from kdflow.utils.logging_utils import init_logger
import torch.distributed as dist
import torch.nn as nn
import transformers

from abc import ABC
from datetime import timedelta
from peft import PeftModel, get_peft_model_state_dict
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
from torch.optim import Optimizer, AdamW
from torch.distributed.fsdp import (
    fully_shard,
    MixedPrecisionPolicy,
    CPUOffloadPolicy,
)
from torch.distributed.tensor import DTensor
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    StateDictOptions,
)
from torchdata.stateful_dataloader import StatefulDataLoader

from kdflow.models import DistillModel
from kdflow.utils.distributed_sampler import DistributedSampler
from kdflow.models.ring_attn_utils import get_ring_attn_group, set_ring_attn_group
from kdflow.utils.distributed_util import torch_dist_barrier_and_cuda_sync


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
        
        self.step = 0
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
            self.log(f"[Rank {dist.get_rank()}] tie_word_embeddings=True, loading full model to CPU on all ranks")
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
        for _name, buf in model.named_buffers():
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
        self.step = (self.step + 1) % self.accumulated_gradient
        loss = loss / self.accumulated_gradient
        loss.backward()
    
    def optimizer_step(
        self,
        optimizer: Optimizer,
        model: nn.Module,
        scheduler,
        **kwargs,
    ) -> None:
        if self.step == 0:
            if self.max_norm > 0.0:
                if hasattr(model, "clip_grad_norm_"):
                    model.clip_grad_norm_(self.max_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)
            
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()
            
    def load_model(self, model: nn.Module, path: str, map_location="cpu", strict: bool = False, key_replace_fn=None) -> None:
        # For FSDP2, we prefer Distributed Checkpoint (DCP)
        # But if the user provides a standard `torch.save` file path, we try to load it.
        # We use `set_model_state_dict` from DCP which handles sharding.
        
        # Load state dict on rank 0 (or all if mapped)
        state_dict = torch.load(path, map_location=map_location)
        if key_replace_fn:
            state_dict = key_replace_fn(state_dict)

        model_to_load = self._unwrap_model(model)
        
        # DCP Helper to load full state dict into sharded model
        # Note: This is memory intensive on Rank 0 if strict full load.
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        set_model_state_dict(model_to_load, model_state_dict=state_dict, options=options)
    
    def save_model(self, model: nn.Module, tokenizer, output_dir, **kwargs) -> None:
        model_to_save = self._unwrap_model(model)
        
        if self.is_rank_0():
            os.makedirs(output_dir, exist_ok=True)
        
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        
        state_dict = get_model_state_dict(model_to_save, options=options)
        
        if self.is_rank_0():
            if isinstance(model_to_save, PeftModel):
                model_to_save.save_pretrained(output_dir, **kwargs)
                torch.save(
                    get_peft_model_state_dict(model_to_save, state_dict),
                    os.path.join(output_dir, "adapter_model.bin"),
                )
            else:
                model_to_save.save_pretrained(output_dir, state_dict=state_dict, **kwargs)
            
            # Config & Tokenizer
            output_config_file = os.path.join(output_dir, "config.json")
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_pretrained(output_dir)

        del state_dict
        gc.collect()
        torch_dist_barrier_and_cuda_sync()
    
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
    def reload_model_params(self, model):
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
    def reload_optim_states(self, optimizer: Optimizer):
        """reload optimizer states to GPU"""
        device = torch.cuda.current_device()
        if not optimizer.state:
            return
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                state = optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device, non_blocking=True)

    def update_rollout_weights_from_tensor(
        self, model, engine, gather_src, gather_group,
        update_weight_buffer_size=2 * 1024**3,
    ):
        """
        Broadcast FSDP model weights to rollout engine with CUDA IPC + Gloo gather.
        Following slime: all ranks serialize their CUDA IPC data, gather to source rank
        via Gloo, then source rank sends to sglang with per-TP-rank IPC handles.
        """
        import ray
        from sglang.srt.utils import MultiprocessingSerializer

        try:
            from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
        except ImportError:
            from sglang.srt.patch_torch import monkey_patch_torch_reductions

        try:
            from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
        except ImportError:
            from sglang.srt.model_executor.model_runner import FlattenedTensorBucket

        torch.cuda.empty_cache()

        model_unwrapped = self._unwrap_model(model)

        bucket = []
        bucket_size = 0

        for name, param in model_unwrapped.state_dict().items():
            param_size = param.numel() * param.element_size()
            if bucket and bucket_size + param_size >= update_weight_buffer_size:
                self._flush_weight_bucket(
                    bucket, engine, gather_src, gather_group,
                    monkey_patch_torch_reductions, FlattenedTensorBucket, MultiprocessingSerializer,
                )
                del bucket
                bucket = []
                bucket_size = 0

            param = param.cuda()
            if isinstance(param, DTensor):
                param = param.redistribute(
                    placements=[torch.distributed.tensor.Replicate()] * param.device_mesh.ndim,
                    async_op=True,
                ).to_local()
            bucket.append((name, param))
            bucket_size += param_size

        if bucket:
            self._flush_weight_bucket(
                bucket, engine, gather_src, gather_group,
                monkey_patch_torch_reductions, FlattenedTensorBucket, MultiprocessingSerializer,
            )
            del bucket

        if dist.get_rank() == gather_src:
            ray.get(engine.flush_cache.remote())

        torch.cuda.empty_cache()
        torch_dist_barrier_and_cuda_sync()

    def _flush_weight_bucket(
        self, bucket, engine, gather_src, gather_group,
        monkey_patch_fn, FlattenedTensorBucket, MultiprocessingSerializer,
    ):
        import ray

        bucket = [(n, p.wait()) if hasattr(p, "wait") else (n, p) for n, p in bucket]

        monkey_patch_fn()

        named_tensors_by_dtype = {}
        for name, tensor in bucket:
            named_tensors_by_dtype.setdefault(tensor.dtype, []).append((name, tensor))

        serialized_tensors = []
        for _dtype, named_tensors in named_tensors_by_dtype.items():
            fb = FlattenedTensorBucket(named_tensors=named_tensors)
            serialized = MultiprocessingSerializer.serialize(
                {"flattened_tensor": fb.get_flattened_tensor(), "metadata": fb.get_metadata()},
                output_str=True,
            )
            serialized_tensors.append(serialized)
            del fb

        if dist.get_rank() == gather_src:
            gathered = [None for _ in range(dist.get_world_size(gather_group))]
        else:
            gathered = None

        dist.gather_object(
            obj=serialized_tensors,
            object_gather_list=gathered,
            dst=gather_src,
            group=gather_group,
        )

        if dist.get_rank() == gather_src:
            num_dtypes = len(gathered[0])
            for i in range(num_dtypes):
                ref = engine.update_weights_from_tensor.remote(
                    serialized_named_tensors=[tensors[i] for tensors in gathered],
                    load_format="flattened_bucket",
                    flush_cache=False,
                )
                ray.get(ref)
            del gathered

        del bucket, serialized_tensors

    def update_rollout_weights_from_disk(self, model, rollout_engines, tokenizer=None, tmp_dir=None):
        """
        Sync FSDP model weights to rollout engines via disk (safetensors).

        Rank 0 gathers the full state dict and saves it to a shared filesystem path,
        then all rollout engines load the weights from disk via sglang's
        update_weights_from_disk API. This completely avoids CUDA IPC issues.

        Args:
            model: The FSDP-wrapped model.
            rollout_engines: List of RolloutRayActor instances.
            tokenizer: Tokenizer (needed for save_pretrained to write tokenizer_config.json etc.).
            tmp_dir: Directory to save the temporary weights. If None, uses a default path.
        """
        import ray
        import shutil

        if tmp_dir is None:
            tmp_dir = os.path.join(
                os.environ.get("TMPDIR", "/tmp"),
                "kdflow_rollout_weights_sync",
            )

        model_unwrapped = self._unwrap_model(model)
        is_sender = len(rollout_engines) > 0

        # Step 1: Gather full state dict (all ranks participate in FSDP collective)
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state_dict = get_model_state_dict(model_unwrapped, options=options)

        # Step 2: Rank 0 saves weights to shared filesystem
        if self.is_rank_0():
            os.makedirs(tmp_dir, exist_ok=True)
            model_unwrapped.save_pretrained(
                tmp_dir,
                state_dict=state_dict,
                safe_serialization=True,
            )
            # Save config (needed for sglang to load correctly)
            model_unwrapped.config.to_json_file(os.path.join(tmp_dir, "config.json"))
            if tokenizer is not None:
                tokenizer.save_pretrained(tmp_dir)

        del state_dict
        gc.collect()
        torch_dist_barrier_and_cuda_sync()

        # Step 3: Tell all rollout engines to load weights from disk
        if is_sender:
            refs = [
                engine.update_weights_from_disk.remote(
                    model_path=tmp_dir,
                    load_format=None,
                )
                for engine in rollout_engines
            ]
            ray.get(refs)

        torch_dist_barrier_and_cuda_sync()

        # Step 4: Cleanup temporary directory
        if self.is_rank_0():
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

        torch.cuda.empty_cache()
        torch_dist_barrier_and_cuda_sync()

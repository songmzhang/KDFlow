import gc
import os

import torch
import torch.distributed as dist
from peft import PeftModel, get_peft_model_state_dict
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.tensor import DTensor

from kdflow.utils.distributed_util import torch_dist_barrier_and_cuda_sync


@torch.no_grad()
def export_lora_adapter(strategy, model, adapter_name: str = "default") -> dict | None:
    model = strategy._unwrap_model(model)
    if not isinstance(model, PeftModel):
        raise TypeError("LoRA export requires a PeftModel.")

    config = model.peft_config.get(adapter_name)
    if config.use_dora or config.bias != "none" or config.modules_to_save:
        raise ValueError("On-policy LoRA sync supports standard LoRA weights only.")

    options = StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
        ignore_frozen_params=True,
    )
    state_dict = get_model_state_dict(model, options=options)
    if not strategy.is_rank_0():
        return None

    adapter_state_dict = get_peft_model_state_dict(
        model,
        state_dict=state_dict,
        adapter_name=adapter_name,
    )
    if not adapter_state_dict:
        raise RuntimeError("No LoRA weights were exported.")

    target_modules = config.target_modules
    if isinstance(target_modules, set):
        target_modules = sorted(target_modules)

    return {
        "config_dict": {
            "r": config.r,
            "lora_alpha": config.lora_alpha,
            "target_modules": target_modules,
            "use_dora": False,
            "peft_type": "LORA",
        },
        "state_dict": {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in adapter_state_dict.items()
        },
    }


def update_rollout_weights_from_tensor(
    strategy, model, engine, gather_src, gather_group,
    update_weight_buffer_size=2 * 1024**3,
    weight_source=None
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

    model_unwrapped = strategy._unwrap_model(model)

    bucket = []
    bucket_size = 0

    state_dict = model_unwrapped.state_dict()
    model_config = getattr(model_unwrapped, "config", None)
    tie_word_embeddings = getattr(model_config, "tie_word_embeddings", False)
    skip_tied_lm_head = (
        tie_word_embeddings and "model.embed_tokens.weight" in state_dict
    )

    for name, param in state_dict.items():
        # Skip the duplicate lm_head alias when embeddings are tied.
        if skip_tied_lm_head and name == "lm_head.weight":
            continue

        param_size = param.numel() * param.element_size()
        if bucket and bucket_size + param_size >= update_weight_buffer_size:
            _flush_weight_bucket(
                bucket, engine, gather_src, gather_group,
                monkey_patch_torch_reductions, FlattenedTensorBucket, MultiprocessingSerializer,
            )
            del bucket
            bucket = []
            bucket_size = 0

        if weight_source is not None and name in weight_source:
            param = weight_source[name]

        param = param.cuda()
        if isinstance(param, DTensor):
            param = param.redistribute(
                placements=[torch.distributed.tensor.Replicate()] * param.device_mesh.ndim,
                async_op=True,
            ).to_local()
        bucket.append((name, param))
        bucket_size += param_size

    del state_dict

    if bucket:
        _flush_weight_bucket(
            bucket, engine, gather_src, gather_group,
            monkey_patch_torch_reductions, FlattenedTensorBucket, MultiprocessingSerializer,
        )
        del bucket

    if dist.get_rank() == gather_src:
        ray.get(engine.flush_cache.remote())

    torch.cuda.empty_cache()
    torch_dist_barrier_and_cuda_sync()


def _flush_weight_bucket(
    bucket, engine, gather_src, gather_group,
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


def update_rollout_weights_from_disk(strategy, model, rollout_engines, tokenizer=None, tmp_dir=None):
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

    model_unwrapped = strategy._unwrap_model(model)
    is_sender = len(rollout_engines) > 0

    # Step 1: Gather full state dict (all ranks participate in FSDP collective)
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    state_dict = get_model_state_dict(model_unwrapped, options=options)

    # Step 2: Rank 0 saves weights to shared filesystem
    if strategy.is_rank_0():
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
    if strategy.is_rank_0():
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    torch.cuda.empty_cache()
    torch_dist_barrier_and_cuda_sync()

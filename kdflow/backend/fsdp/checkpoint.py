import gc
import json
import os
import random
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


_CHECKPOINT_NAME = re.compile(r"epoch_([1-9][0-9]*)(_end)?_global_step_(0|[1-9][0-9]*)")


def _read_model_weights(model_dir: Path) -> dict:
    from transformers.modeling_utils import load_state_dict

    model_file = model_dir / "model.safetensors"
    index_file = model_dir / "model.safetensors.index.json"
    adapter_file = model_dir / "adapter_model.bin"
    if model_file.is_file():
        files = [model_file]
    elif index_file.is_file():
        index = json.loads(index_file.read_text(encoding="utf-8"))
        files = [model_dir / name for name in sorted(set(index["weight_map"].values()))]
    elif adapter_file.is_file():
        files = [adapter_file]
    else:
        raise FileNotFoundError(f"No model weights found in {model_dir}")

    state_dict = {}
    for file in files:
        state_dict.update(load_state_dict(str(file), map_location="cpu", weights_only=True))
    return state_dict


def _find_tied_parameter_names(named_parameters) -> list[list[str]]:
    """Group names that refer to the same parameter object."""
    names_by_param_id = {}
    for name, param in named_parameters:
        names_by_param_id.setdefault(id(param), []).append(name)
    return [names for names in names_by_param_id.values() if len(names) > 1]


def load_model(strategy, model, model_dir, strict=False) -> None:
    """Load HF weights into the prepared model; all ranks participate."""
    from peft import PeftModel
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    model_to_load = strategy._unwrap_model(model)
    is_lora_model = isinstance(model_to_load, PeftModel)
    state_dict = {}

    if strategy.is_rank_0():
        state_dict = _read_model_weights(Path(model_dir))
        named_parameters = list(model_to_load.named_parameters(remove_duplicate=False))
        model_keys = {name for name, _ in named_parameters}
        model_keys.update(name for name, _ in model_to_load.named_buffers())

        # Some VLMs use different parameter names in the saved HF checkpoint.
        key_mapping = getattr(model_to_load, "_checkpoint_conversion_mapping", {})
        for saved_name in list(state_dict):
            if saved_name in model_keys:
                continue
            for pattern, replacement in key_mapping.items():
                model_name, replacement_count = re.subn(pattern, replacement, saved_name)
                if replacement_count:
                    state_dict[model_name] = state_dict.pop(saved_name)
                    break

        if is_lora_model:
            # PEFT removes the adapter name: lora_A.default.weight becomes lora_A.weight.
            for model_name in model_keys:
                if model_name.endswith(".default.weight"):
                    saved_name = model_name.removesuffix(".default.weight") + ".weight"
                elif model_name.endswith(".default"):
                    saved_name = model_name.removesuffix(".default")
                else:
                    continue
                if saved_name in state_dict:
                    state_dict[model_name] = state_dict.pop(saved_name)

            trainable_names = {name for name, param in named_parameters if param.requires_grad}
            missing_names = trainable_names - state_dict.keys()
            if strict and missing_names:
                raise ValueError(f"Missing adapter weights: {sorted(missing_names)}")

        # Tied embeddings share one tensor, but HF safetensors may save only one name.
        for tied_names in _find_tied_parameter_names(named_parameters):
            for saved_name in tied_names:
                if saved_name in state_dict:
                    for name in tied_names:
                        state_dict.setdefault(name, state_dict[saved_name])
                    break

    # Adapter checkpoints omit the frozen base-model weights.
    options = StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=True,
        strict=strict and not is_lora_model,
    )
    set_model_state_dict(model_to_load, model_state_dict=state_dict, options=options)


def save_model(strategy, model, output_dir, **kwargs) -> None:
    """Save a Hugging Face model and its processor or tokenizer."""
    from peft import PeftModel
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    from kdflow.utils.distributed_util import torch_dist_barrier_and_cuda_sync

    if hasattr(model, "module"):
        model = model.module
    processor = getattr(model, "processor", None)
    processor_or_tokenizer = processor if processor is not None else getattr(model, "tokenizer", None)
    model_to_save = strategy._unwrap_model(model)

    if strategy.is_rank_0():
        os.makedirs(output_dir, exist_ok=True)
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    state_dict = get_model_state_dict(model_to_save, options=options)
    if strategy.args.train.bf16:
        state_dict = {
            key: value.to(torch.bfloat16) if torch.is_floating_point(value) else value
            for key, value in state_dict.items()
        }

    if strategy.is_rank_0():
        if isinstance(model_to_save, PeftModel):
            model_to_save.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=False, **kwargs)
        else:
            model_to_save.save_pretrained(output_dir, state_dict=state_dict, **kwargs)
        model_to_save.config.to_json_file(os.path.join(output_dir, "config.json"))
        processor_or_tokenizer.save_pretrained(output_dir)

    del state_dict
    gc.collect()
    torch_dist_barrier_and_cuda_sync()


def _local_cpu_state(value):
    """Convert tensors to CPU, storing only this rank's shard for DTensors."""
    if isinstance(value, DTensor):
        value = value.to_local()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _local_cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_local_cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_local_cpu_state(item) for item in value)
    return value


def save_checkpoint(
    strategy,
    model,
    epoch: int,
    global_step: int,
    *,
    epoch_end: bool = False,
    optimizer=None,
    scheduler=None,
    trainer_state: dict | None = None,
    extra_state: dict | None = None,
) -> Path:
    """Save a checkpoint on all ranks using a shared save_path; epoch is one-based.

    trainer_state contains the caller's progress and dataloader state.
    Optimizer tensors are local shards, for restoration with the same parallel layout.
    """
    args = strategy.args.ckpt
    name = f"epoch_{epoch}" + ("_end" if epoch_end else "") + f"_global_step_{global_step}"
    checkpoint_names = [saved for saved in strategy.checkpoint_names if saved != name] + [name]
    retained = checkpoint_names[-args.max_ckpt_num:] if args.max_ckpt_num != -1 else checkpoint_names
    directory = [None]
    if strategy.is_rank_0():
        directory[0] = str(create_checkpoint_dir(
            Path(args.save_path) / name, args.save_training_state,
        ))
    dist.broadcast_object_list(directory, src=0)
    path = Path(directory[0])

    save_model(strategy, model, path / "model")
    if args.save_training_state:
        if optimizer is None or scheduler is None or trainer_state is None:
            raise ValueError("Pass optimizer, scheduler, and trainer_state when saving training state.")
        if strategy.grad_accum_step != 0:
            raise ValueError("Save training state after the optimizer update has finished.")
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
        }

        state = {
            "epoch": epoch,
            "global_step": global_step,
            "epoch_end": epoch_end,
            "checkpoint_names": retained,
            "parallelism": {
                "world_size": strategy.world_size,
                "fsdp_size": strategy.args.fsdp.fsdp_size,
                "ring_attn_size": strategy.args.model.ring_attn_size,
            },
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": rng_state,
            "trainer_state": trainer_state,
            "extra_state": extra_state,
        }
        torch.save(_local_cpu_state(state), path / "training_state" / f"rank_{strategy.get_rank()}.pt")
        del state
        dist.barrier()

    if strategy.is_rank_0():
        complete_checkpoint_save(path)
        prune_checkpoints(path.parent, checkpoint_names, args.max_ckpt_num)
    dist.barrier()
    strategy.checkpoint_names = retained
    return path.with_name(name)


def load_checkpoint(strategy, model, checkpoint_path: str | Path, *, optimizer, scheduler) -> dict:
    """Restore prepared model/optimizer/scheduler objects on all ranks.

    Return progress and caller-owned trainer/extra state. Restore with the same parallel layout.
    """
    path = Path(checkpoint_path)
    state = torch.load(
        path / "training_state" / f"rank_{strategy.get_rank()}.pt", map_location="cpu", weights_only=False,
    )
    parallelism = {
        "world_size": strategy.world_size,
        "fsdp_size": strategy.args.fsdp.fsdp_size,
        "ring_attn_size": strategy.args.model.ring_attn_size,
    }
    if state["parallelism"] != parallelism:
        details = "\n".join(
            f"  {name}: checkpoint={state['parallelism'][name]}, current={value}"
            for name, value in parallelism.items()
        )
        raise ValueError(
            "Please resume with the same GPU count, fsdp_size, and ring_attn_size.\n" + details
        )

    strategy.onload_model_params(model)
    load_model(strategy, model, path / "model", strict=True)
    scheduler.load_state_dict(state["scheduler"])
    optimizer.load_state_dict(state["optimizer"])
    for param, optim_state in optimizer.state.items():
        if not isinstance(param, DTensor):
            continue

        for state_name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if state_name not in optim_state:
                continue

            local_state = optim_state[state_name]
            optim_state[state_name] = DTensor.from_local(
                local_state,
                device_mesh=param.device_mesh,
                placements=param.placements,
                shape=param.shape,
                stride=param.stride(),
            )
    optimizer.zero_grad(set_to_none=True)
    strategy.grad_accum_step = 0
    strategy.checkpoint_names = [
        name for name in state["checkpoint_names"] if (Path(strategy.args.ckpt.save_path) / name).is_dir()
    ]
    dist.barrier()

    rng = state["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    torch.cuda.set_rng_state(rng["cuda"])
    return {key: state[key] for key in ("epoch", "global_step", "epoch_end", "trainer_state", "extra_state")}


def _check_contents(path: Path, require_training_state: bool) -> None:
    names = ("model", "training_state") if require_training_state else ("model",)
    for name in names:
        directory = path / name
        if not directory.is_dir() or not any(directory.iterdir()):
            raise ValueError(f"Checkpoint {name}/ directory is missing or empty: {path}")


def resolve_resume_checkpoint(
    save_path: str | Path,
    resume_from: str | Path | None = None,
    resume_training: bool = False,
) -> Path | None:
    """Select an explicit checkpoint or latest without changing saved files."""
    if not resume_training and resume_from is None:
        return None
    if resume_from is not None:
        path = Path(resume_from)
    else:
        latest = Path(save_path) / "latest"
        if not latest.exists() and not latest.is_symlink():
            raise FileNotFoundError(
                f"No latest checkpoint found in {save_path}. "
                "Set --resume_from or disable --resume_training."
            )
        if latest.is_symlink() or not latest.is_file():
            raise ValueError(f"Expected a latest text file: {latest}")
        name = latest.read_text(encoding="utf-8").strip()
        if not _CHECKPOINT_NAME.fullmatch(name):
            raise ValueError(f"Invalid checkpoint name in {latest}: {name!r}")
        path = latest.parent / name
    if not _CHECKPOINT_NAME.fullmatch(path.name):
        raise ValueError(f"Expected an epoch_<n>[_end]_global_step_<m> directory: {path}")
    _check_contents(path, require_training_state=True)
    return path


def create_checkpoint_dir(
    checkpoint_path: str | Path,
    save_training_state: bool = False,
) -> Path:
    """Create a temporary directory to share with model and state writers."""
    destination = Path(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    (path / "model").mkdir()
    if save_training_state:
        (path / "training_state").mkdir()
    return path


def _write_latest(root: Path, checkpoints: list[Path]) -> None:
    checkpoint = next((path for path in reversed(checkpoints) if (path / "training_state").is_dir()), None)
    if checkpoint is None:
        (root / "latest").unlink(missing_ok=True)
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=".latest.tmp-", dir=root, delete=False
        ) as f:
            temporary = Path(f.name)
            f.write(checkpoint.name + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, root / "latest")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def complete_checkpoint_save(checkpoint_dir: str | Path) -> Path:
    """Publish after every writer has finished, replacing any checkpoint with the same name."""
    path = Path(checkpoint_dir)
    name = path.name.removeprefix(".").split(".tmp-", 1)[0]
    destination = path.with_name(name)
    _check_contents(path, require_training_state=(path / "training_state").exists())
    if destination.exists():
        shutil.rmtree(destination)
    path.rename(destination)
    return destination


def prune_checkpoints(
    save_path: str | Path, checkpoint_names: list[str], max_ckpt_num: int = -1,
) -> list[str]:
    """Prune this run's oldest checkpoints and return the retained names in save order."""
    root = Path(save_path)
    if not root.exists():
        return []
    removed = checkpoint_names[:-max_ckpt_num] if max_ckpt_num != -1 else []
    retained = checkpoint_names[len(removed):]
    _write_latest(root, [root / name for name in retained])
    for name in removed:
        shutil.rmtree(root / name)
    return retained

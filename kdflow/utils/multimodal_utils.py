"""Multi-modal field helpers (verl-inspired)."""
from typing import Iterable, Optional

import torch
from peft import PeftModel


def register_dummy_vision_hook(model):
    """Connect a dummy Qwen3.5 vision pass when `_fsdp_dummy_vision` is set."""
    if isinstance(model, PeftModel):
        model = model.get_base_model()
    if model.config.model_type != "qwen3_5":
        raise ValueError("Dummy vision currently supports Qwen3.5 image/text training only.")
    visual = model.base_model.visual
    language_model = model.base_model.language_model
    if getattr(language_model, "_dummy_vision_hook_handle", None) is not None:
        return

    config = model.config.vision_config
    grid_size = config.spatial_merge_size
    patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2

    def before_language_forward(module, args, kwargs):
        if kwargs.pop("_fsdp_dummy_vision", False):
            inputs_embeds = kwargs["inputs_embeds"]
            pixels = torch.zeros(
                grid_size**2, patch_dim, device=inputs_embeds.device, dtype=visual.dtype,
            )
            grid = torch.tensor([[1, grid_size, grid_size]], device=inputs_embeds.device)
            features = visual(pixels, grid_thw=grid, return_dict=True).pooler_output
            zero = (features.float().sum() * 0).to(inputs_embeds.dtype)
            kwargs["inputs_embeds"] = inputs_embeds + zero
        return args, kwargs

    language_model._dummy_vision_hook_handle = language_model.register_forward_pre_hook(
        before_language_forward, with_kwargs=True,
    )


def extract_multi_modal_inputs(
    multi_modal_inputs_list: Optional[Iterable[Optional[dict]]],
) -> dict:
    """Concat per-sample mm dicts into a batched dict for HF VLM forward.

    Args:
        multi_modal_inputs_list: an iterable of per-sample multi_modal_inputs
            dicts (each dict maps field name -> tensor with leading dim
            corresponding to that sample's patches/tokens). ``None`` entries
            (e.g. text-only samples) are skipped.

    Returns:
        A dict mapping field name -> tensor concatenated along ``dim=0``
        across all non-empty samples. Empty input returns ``{}``.
    """
    if not multi_modal_inputs_list:
        return {}
    collected: dict = {}
    for d in multi_modal_inputs_list:
        if not d:
            continue
        for k, v in d.items():
            if v is not None:
                collected.setdefault(k, []).append(v)
    return {k: torch.cat(vs, dim=0) for k, vs in collected.items()}

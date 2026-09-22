import torch
from peft import PeftModel

from kdflow.models.ring_attn_utils import (
    gather_ring_attn_tensor,
    get_tensor_in_current_ring_attn_rank,
    update_ring_attn_params,
)


def _require_flash_attn():
    """Deferred import — only needed for packing_samples=True."""
    try:
        from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "flash_attn is required for packing_samples=True. "
            "Install with: pip install flash_attn --no-build-isolation"
        ) from e
    return index_first_axis, pad_input, rearrange, unpad_input


def _exclude_text_packing_kwargs(module, args, kwargs):
    """Keep language-model sequence boundaries out of the vision tower."""
    kwargs = kwargs.copy()
    for key in ("seq_idx", "cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"):
        kwargs.pop(key, None)
    return args, kwargs


def register_vision_packing_hook(model):
    """Register once during model initialization when multimodal packing is enabled."""
    if isinstance(model, PeftModel):
        model = model.get_base_model()
    visual = model.base_model.visual
    if getattr(visual, "_text_packing_hook_handle", None) is None:
        visual._text_packing_hook_handle = visual.register_forward_pre_hook(
            _exclude_text_packing_kwargs, with_kwargs=True,
        )


def prepare_packed_inputs(
    model,
    sequences,
    attention_mask,
    ring_attn_group=None,
    **model_kwargs,
):
    """Return packed model inputs, text boundary kwargs, and output restoration kwargs.

    Accepts an HF model or its PEFT wrapper. Image tensors and grids retain
    their original order.
    Text boundary kwargs must be routed separately from the vision tower.
    """
    if isinstance(model, PeftModel):
        model = model.get_base_model()
    batch, seqlen = sequences.shape
    multimodal_positions = None
    if hasattr(model.config, "vision_config"):
        get_rope_index = getattr(model.base_model, "get_rope_index", None)
        if get_rope_index is None:
            raise ValueError(
                f"Multimodal packing is not supported for {model.config.model_type}: "
                "the model does not provide get_rope_index."
            )
        if ring_attn_group is not None:
            raise ValueError("Multimodal packing requires ring_attn_size=1.")
        if model_kwargs.get("mm_token_type_ids") is None:
            mm_token_type_ids = torch.zeros_like(sequences)
            mm_token_type_ids[sequences == model.config.image_token_id] = 1
            mm_token_type_ids[sequences == model.config.video_token_id] = 2
            model_kwargs["mm_token_type_ids"] = mm_token_type_ids
        multimodal_positions, _ = get_rope_index(
            sequences,
            attention_mask=attention_mask,
            mm_token_type_ids=model_kwargs["mm_token_type_ids"],
            image_grid_thw=model_kwargs.get("image_grid_thw"),
            video_grid_thw=model_kwargs.get("video_grid_thw"),
        )

    sequences, position_ids, _, ring_pad_len, indices, packing_kwargs = unpad_and_slice_tensor(
        sequences, attention_mask, ring_attn_group,
    )
    if multimodal_positions is not None:
        multimodal_positions = multimodal_positions.flatten(1)[:, indices].unsqueeze(1)
        # HF reads sample boundaries from the first axis, followed by the three M-RoPE axes.
        position_ids = torch.cat([position_ids.unsqueeze(0), multimodal_positions], dim=0)
        model_kwargs["mm_token_type_ids"] = model_kwargs["mm_token_type_ids"].flatten()[indices].unsqueeze(0)

    model_inputs = dict(model_kwargs, input_ids=sequences, attention_mask=None, position_ids=position_ids)
    restore_kwargs = dict(
        ring_attn_group=ring_attn_group,
        ring_attn_pad_len=ring_pad_len,
        indices=indices,
        batch=batch,
        seqlen=seqlen,
    )
    return model_inputs, packing_kwargs, restore_kwargs


def unpad_and_slice_tensor(sequences, attention_mask, ring_attn_group):
    """
    Unpad and slice tensor for distributed training with ring attention.

    This function performs several operations:
    1. Removes padding, unpads sequences from (batch, seqlen) to (1, total_seqs)
    2. Adapts to ring_attn_group, pads sequences to be divisible by ring_attn_group
    3. Slices the sequences for the current ring_attn_rank

    Example:
        >>> # Input sequences shape: (batch=2, seqlen=4)
        >>> sequences = [[1, 2, 3, 0], [4, 5, 0, 0]]  # 0 is padding
        >>> attention_mask = [[1, 1, 1, 0], [1, 1, 0, 0]]
        >>> # After unpad:
        >>> # sequences: [1, 2, 3, 4, 5]  # shape (1, total_seqs=5)
        >>> # If ring_attn_group size is 2, it will pad to length 6
        >>> # Then slice for current rank (e.g., rank 0 gets [1,2,3], rank 1 gets [4,5,0])

    Args:
        sequences: Input sequences tensor of shape (batch, seqlen)
        attention_mask: Attention mask tensor for the sequences
        ring_attn_group: Ring attention group for distributed processing

    Returns:
        tuple: Processed sequences and related tensors for ring attention

    Note:
        Requires ``flash_attn`` — only called when ``packing_samples=True``.
    """
    index_first_axis, _, rearrange, unpad_input = _require_flash_attn()

    rolled_sequences = torch.roll(sequences, shifts=-1, dims=1)
    sequences, indices, cu_seqlens, max_seqlen, _ = unpad_input(sequences.unsqueeze(-1), attention_mask)
    sequences = sequences.transpose(0, 1)  # (1, total_seqs)
    seq_idx = torch.arange(attention_mask.shape[0], dtype=torch.int32, device=attention_mask.device)
    seq_idx = seq_idx[:, None].expand_as(attention_mask)[attention_mask.bool()].unsqueeze(0)
    rolled_sequences = index_first_axis(
        rearrange(rolled_sequences.unsqueeze(-1), "b s ... -> (b s) ..."), indices
    ).transpose(
        0, 1
    )  # (1, total_seqs)
    position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None)
    position_ids = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(
        0, 1
    )  # (1, total_seqs)
    ring_attn_pad_len = 0
    if ring_attn_group is not None:
        (sequences, position_ids, rolled_sequences), ring_attn_pad_len = get_tensor_in_current_ring_attn_rank(
            [sequences, position_ids, rolled_sequences], ring_attn_group, 0
        )
        cu_seqlens[-1] += ring_attn_pad_len
        update_ring_attn_params(cu_seqlens)
    packing_kwargs = {
        "seq_idx": seq_idx,
        "cu_seq_lens_q": cu_seqlens,
        "cu_seq_lens_k": cu_seqlens,
        "max_length_q": max_seqlen,
        "max_length_k": max_seqlen,
    }
    return sequences, position_ids, rolled_sequences, ring_attn_pad_len, indices, packing_kwargs


def gather_and_pad_tensor(tensor, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen):
    """
    Gather and pad tensor data (such as logits, log_probs, etc.).

    Example:
        >>> # Input tensor from each rank (shape: (1, local_seq_len))
        >>> # Rank 0: [1, 2, 3]
        >>> # Rank 1: [4, 5, 0]  # 0 is padding
        >>> # After all_gather:
        >>> # tensor: [1, 2, 3, 4, 5, 0]  # shape (1, total_seqs=6)
        >>> # After removing padding (ring_attn_pad_len=1):
        >>> # tensor: [1, 2, 3, 4, 5]  # shape (1, total_seqs=5)
        >>> # After pad_input with original indices:
        >>> # tensor: [[1, 2, 3, 0], [4, 5, 0, 0]]  # shape (batch=2, seqlen=4)

    Args:
        tensor: Input tensor, can be logits, log_probs, etc.
        ring_attn_group: Ring attention group
        ring_attn_pad_len: Padding length
        indices: Indices
        batch: Batch size
        seqlen: Sequence length

    Returns:
        Padded tensor

    Note:
        Requires ``flash_attn`` — only called when ``packing_samples=True``.
    """
    _, pad_input, _, _ = _require_flash_attn()

    if ring_attn_group is not None:
        tensor = gather_ring_attn_tensor(tensor, ring_attn_group, ring_attn_pad_len)
    tensor = pad_input(tensor.transpose(0, 1), indices, batch, seqlen).squeeze(-1)  # (batch, seqlen)
    return tensor

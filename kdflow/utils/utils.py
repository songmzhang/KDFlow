from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from kdflow.utils.logging_utils import init_logger

logger = init_logger(__name__)


def get_tokenizer(model_name_or_path, model=None, padding_side="left", use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        logger.info("Detect no pad_token in tokenizer, set it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if model is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def convert_token_to_id(token, tokenizer):
    if isinstance(token, str):
        token = tokenizer.encode(token, add_special_tokens=False)
        assert len(token) == 1
        return token[0]
    else:
        raise ValueError("token should be int or str")


# def zero_pad_sequences(
#     sequences: List[torch.Tensor], side: str = "left", value: int = 0, stack: bool = False
# ) -> torch.Tensor:
#     assert side in ("left", "right")
#     max_len = max(seq.size(-1) for seq in sequences)
#     padded_sequences = []
#     for seq in sequences:
#         pad_len = max_len - seq.size(-1)
#         padding = (pad_len, 0) if side == "left" else (0, pad_len)
#         padded_sequences.append(F.pad(seq, padding, value=value))
#     if stack:
#         print([s.shape for s in sequences], [ps.shape for ps in padded_sequences])
#         return torch.stack(padded_sequences, dim=0)
#     else:
#         return torch.cat(padded_sequences, dim=0)
    

def zero_pad_sequences(
    sequences: List[torch.Tensor], side: str = "left", value: int = 0, stack: bool = False
) -> torch.Tensor:
    from torch.nn.utils.rnn import pad_sequence
    assert side in ("left", "right")
    sequences = [seq.squeeze(0) for seq in sequences]
    if side == "left":
        sequences = [seq.flip(dims=0) for seq in sequences]
    padded_sequences = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=value
    )
    if side == "left":
        padded_sequences = torch.flip(padded_sequences, dims=[1])
    return padded_sequences


def remove_pad_token(input_ids: torch.Tensor, attention_mask: torch.Tensor, return_tensors: bool = True):
    """Remove the pad token. Return tensors and not lists.

    Args:
        input_ids shape: [bs, seq_length]
        attention_mask shape: [bs, seq_length]
    Returns:
        no_padding_batch(List[Tensor[int]]): contains the rmpad token ids per query.
    """
    no_padding_batch = []
    for ids, mask in zip(input_ids, attention_mask):
        # Fix for both left and right padding
        ids = ids[mask.bool()] if return_tensors else ids[mask.bool()].tolist()
        no_padding_batch.append(ids)
    return no_padding_batch

"""Dynamic batch size utilities for KDFlow.

Given a global_batch (list of micro-batches), rearrange samples into new
micro-batches such that each micro-batch's total token count is bounded by
max_token_len.

The Karmarkar-Karp partitioning algorithm and first-fit estimation are adapted
from verl:
  https://github.com/volcengine/verl/blob/main/verl/utils/seqlen_balancing.py
  (Apache-2.0 License, Copyright 2024 Bytedance Ltd. and/or its affiliates)
"""

import heapq
from typing import Dict, List, Optional, Tuple

import ray
import torch
import numpy as np
import torch.distributed as dist
from kdflow.utils.logging_utils import init_logger
from torch.nn.utils.rnn import pad_sequence

logger = init_logger(__name__)

# Keys whose tensors are 2-D [batch, seq_len] and need unpad / repad
_SEQ_KEYS = {"stu_input_ids", "stu_attn_mask", "stu_loss_mask",
             "tea_input_ids", "tea_attn_mask", "tea_loss_mask"}

# Keys whose tensors are 1-D scalars per sample (e.g. response_length)
_SCALAR_KEYS = {"response_length", "total_length"}

# Keys that are plain python lists (one element per sample)
_LIST_KEYS = {"tea_full_texts", "stu_prompts", "stu_responses", "tea_prompts",
              "labels", "images"}

# Keys whose values are concatenated response-level tensors [total_resp_tokens, ...]
# that must be split per sample using tea_loss_mask counts.
_CONCAT_HIDDEN_KEYS = {"teacher_hiddens"}


def rearrange_global_batch(
    global_batch: List[Dict[str, torch.Tensor]],
    max_token_len: int,
    token_len_key: str = "stu_attn_mask",
    dp_size: Optional[int] = None,
    dp_group=None,
) -> List[Dict[str, torch.Tensor]]:
    """Rearrange a global_batch into dynamic micro-batches.

    Steps:
        1. Unpack all micro-batches into individual samples (remove padding).
        2. Estimate the number of micro-batches needed via first-fit.
        3. Use Karmarkar-Karp to balance samples across micro-batches.
        4. Re-collate each partition with right-padding.

    Args:
        global_batch: List of micro-batches, each is a dict of tensors.
        max_token_len: Maximum total *real* tokens per micro-batch.
        token_len_key: Key used to measure each sample's token count
            (via attention_mask.sum()).
        dp_size: If set, ensure the number of new micro-batches is divisible
            by ``dp_size``.  This is needed when the caller (e.g. off-policy /
            on-policy KD trainer) distributes the rearranged global_batch
            evenly across DP ranks via ``async_run_distill``.
        dp_group: ``torch.distributed`` process group for data-parallel sync.
            When provided, an ``all_reduce(MAX)`` is performed so that every
            DP rank ends up with the same number of micro-batches (needed by
            SFT trainer where each rank holds its own global_batch).
            Ranks with fewer micro-batches are padded with dummy batches.

    Returns:
        A new list of micro-batches (may differ in count from the input).
    """
    # Step 1: unpack into single samples
    samples = _unpack_global_batch(global_batch)
    if not samples:
        return global_batch

    # Step 2: compute per-sample token lengths
    token_lens = [s[token_len_key].sum().item() for s in samples]

    # Step 3: estimate number of partitions via first-fit
    num_partitions = _first_fit_num_partitions(token_lens, max_token_len)
    num_partitions = max(num_partitions, 1)
    num_partitions = min(num_partitions, len(samples))

    # Step 3.1: ensure divisibility by dp_size (for off-policy / on-policy KD)
    if dp_size is not None and dp_size > 1:
        num_partitions = _roundup_divisible(num_partitions, dp_size)
        num_partitions = min(num_partitions, len(samples))

    # Step 3.2: cross-GPU sync (for SFT trainer where each rank has its own data)
    if dp_group is not None and dist.is_initialized():
        num_partitions_t = torch.tensor([num_partitions], device=torch.cuda.current_device())
        dist.all_reduce(num_partitions_t, op=dist.ReduceOp.MAX, group=dp_group)
        num_partitions = int(num_partitions_t.item())

    # Step 4: balanced partitioning
    if num_partitions >= len(samples):
        # Each sample is its own micro-batch
        partitions = [[i] for i in range(len(samples))]
    else:
        partitions = _karmarkar_karp(token_lens, num_partitions)

    # Step 5: re-collate each partition
    new_global_batch = []
    for partition in partitions:
        batch_samples = [samples[i] for i in partition]
        micro_batch = _collate_samples(batch_samples)
        new_global_batch.append(micro_batch)

    # Step 5.1: pad with dummy micro-batches if needed (for dp_group sync)
    while len(new_global_batch) < num_partitions:
        # Create a dummy micro-batch by cloning the last one with zeroed loss_mask
        dummy = {k: v.clone() if isinstance(v, torch.Tensor) else v
                 for k, v in new_global_batch[-1].items()}
        for k in ("stu_loss_mask", "tea_loss_mask"):
            if k in dummy and isinstance(dummy[k], torch.Tensor):
                dummy[k] = torch.zeros_like(dummy[k])
        new_global_batch.append(dummy)

    return new_global_batch


def _unpack_global_batch(
    global_batch: List[Dict[str, torch.Tensor]],
) -> List[Dict]:
    """Unpack all micro-batches into a flat list of single-sample dicts.

    For sequence tensors (2-D), each sample is extracted and un-padded
    (trailing padding removed based on attention_mask).
    """
    samples = []
    for mb in global_batch:
        # Determine batch size from any seq key present
        bs = None
        for k in _SEQ_KEYS:
            if k in mb and isinstance(mb[k], torch.Tensor) and mb[k].dim() == 2:
                bs = mb[k].shape[0]
                break
        if bs is None:
            # Fallback: treat the whole micro-batch as one sample
            samples.append(mb)
            continue

        # Determine real lengths per sample from attention masks
        stu_mask = mb.get("stu_attn_mask")
        tea_mask = mb.get("tea_attn_mask")
        stu_lens = stu_mask.sum(dim=1).tolist() if stu_mask is not None else [None] * bs
        tea_lens = tea_mask.sum(dim=1).tolist() if tea_mask is not None else [None] * bs

        # Split concatenated hidden-state tensors by per-sample response token counts
        tea_loss_mask = mb.get("tea_loss_mask")
        per_sample_resp_lens = tea_loss_mask.bool().sum(dim=1).tolist() if tea_loss_mask is not None else None
        split_hiddens = {}
        for k in _CONCAT_HIDDEN_KEYS:
            v = mb.get(k)
            if v is None:
                continue
            if isinstance(v, ray.ObjectRef):
                v = ray.get(v)
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            if per_sample_resp_lens is not None:
                split_hiddens[k] = torch.split(v, per_sample_resp_lens, dim=0)
            else:
                split_hiddens[k] = [v]

        for i in range(bs):
            sample = {}
            for k, v in mb.items():
                if k in _SEQ_KEYS and isinstance(v, torch.Tensor) and v.dim() == 2:
                    # Determine the real length for this key
                    if k.startswith("tea_"):
                        real_len = int(tea_lens[i]) if tea_lens[i] is not None else v.shape[1]
                    else:
                        real_len = int(stu_lens[i]) if stu_lens[i] is not None else v.shape[1]
                    sample[k] = v[i, :real_len]  # 1-D tensor, no padding
                elif k in _SCALAR_KEYS and isinstance(v, torch.Tensor):
                    if v.dim() >= 1 and v.shape[0] == bs:
                        sample[k] = v[i]
                    else:
                        sample[k] = v
                elif k in _LIST_KEYS and isinstance(v, list) and len(v) == bs:
                    sample[k] = v[i]
                elif k.startswith("mm_") and isinstance(v, torch.Tensor):
                    # Multimodal tensors: keep per-sample slice
                    # For simplicity, assume first dim is batch
                    if v.dim() >= 1 and v.shape[0] == bs:
                        sample[k] = v[i]
                    else:
                        sample[k] = v  # shared across samples
                elif k in _CONCAT_HIDDEN_KEYS:
                    if k in split_hiddens and i < len(split_hiddens[k]):
                        sample[k] = split_hiddens[k][i]
                    # else: skip (no hidden for this sample)
                elif k == "avg_micro_batch_token_num":
                    # Will be recomputed later; skip
                    continue
                else:
                    # Unknown key: just copy as-is for the first sample
                    sample[k] = v
            samples.append(sample)

    return samples


def _roundup_divisible(a: int, b: int) -> int:
    """Round *a* up to the nearest multiple of *b*.

    Adapted from verl ``roundup_divisible``.
    """
    return ((a + b - 1) // b) * b


def _first_fit_num_partitions(token_lens: List[int], max_token_len: int) -> int:
    """Estimate the minimum number of partitions using first-fit decreasing."""
    sorted_lens = sorted(token_lens, reverse=True)
    bins = []
    for l in sorted_lens:
        placed = False
        for i in range(len(bins)):
            if bins[i] + l <= max_token_len:
                bins[i] += l
                placed = True
                break
        if not placed:
            bins.append(l)
    return len(bins)


def _karmarkar_karp(
    seqlen_list: List[int],
    k_partitions: int,
) -> List[List[int]]:
    """Karmarkar-Karp multi-way partitioning (largest differencing method).

    Partitions ``seqlen_list`` into ``k_partitions`` groups such that the
    maximum group-sum is minimized (i.e., balanced workload).

    Adapted from verl ``karmarkar_karp``:
      https://github.com/volcengine/verl/blob/main/verl/utils/seqlen_balancing.py

    Returns:
        List of k_partitions lists, each containing sample indices.
    """

    class _Set:
        __slots__ = ("total", "items")

        def __init__(self):
            self.total = 0
            self.items = []  # list of (idx, val)

        def add(self, idx: int, val: int):
            self.items.append((idx, val))
            self.total += val

        def merge(self, other: "_Set"):
            self.items.extend(other.items)
            self.total += other.total

        def __lt__(self, other: "_Set"):
            if self.total != other.total:
                return self.total < other.total
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

    class _State:
        __slots__ = ("k", "sets")

        def __init__(self, items: List[Tuple[int, int]], k: int):
            self.k = k
            self.sets = [_Set() for _ in range(k)]
            for i, (idx, val) in enumerate(items):
                self.sets[i % k].add(idx, val)
            self.sets.sort(reverse=True)

        @property
        def spread(self) -> int:
            return self.sets[0].total - self.sets[-1].total

        def merge(self, other: "_State"):
            for i in range(self.k):
                self.sets[i].merge(other.sets[self.k - 1 - i])
            self.sets.sort(reverse=True)

        def get_partitions(self) -> List[List[int]]:
            return [sorted(idx for idx, _ in s.items) for s in self.sets]

        def __lt__(self, other: "_State"):
            if self.spread != other.spread:
                return self.spread > other.spread  # max-heap by spread
            return self.sets[0] > other.sets[0]

    sorted_items = sorted(enumerate(seqlen_list), key=lambda x: x[1])
    pq = []
    for idx, val in sorted_items:
        heapq.heappush(pq, _State(items=[(idx, val)], k=k_partitions))

    while len(pq) > 1:
        s0 = heapq.heappop(pq)
        s1 = heapq.heappop(pq)
        s0.merge(s1)
        heapq.heappush(pq, s0)

    partitions = pq[0].get_partitions()
    # Remove empty partitions (shouldn't happen, but be safe)
    partitions = [p for p in partitions if p]
    return partitions


def _collate_samples(samples: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate a list of single-sample dicts into one micro-batch dict.

    Sequence tensors are right-padded to the longest in the batch.
    """
    batch = {}
    keys = samples[0].keys()

    for k in keys:
        values = [s[k] for s in samples if k in s]
        if not values:
            continue

        v0 = values[0]
        if k in _SEQ_KEYS and isinstance(v0, torch.Tensor) and v0.dim() == 1:
            # Right-pad variable-length 1-D tensors
            pad_val = 0
            batch[k] = pad_sequence(values, batch_first=True, padding_value=pad_val)
        elif k in _SCALAR_KEYS and isinstance(v0, torch.Tensor):
            batch[k] = torch.stack(values)
        elif k in _LIST_KEYS:
            batch[k] = values
        elif k.startswith("mm_") and isinstance(v0, torch.Tensor):
            # Multimodal: cat along batch dim
            batch[k] = torch.cat([v.unsqueeze(0) if v.dim() == 0 else v for v in values], dim=0)
        elif k in _CONCAT_HIDDEN_KEYS and isinstance(v0, torch.Tensor):
            batch[k] = torch.cat(values, dim=0)
        else:
            # Default: keep first value (shared metadata)
            batch[k] = v0

    return batch

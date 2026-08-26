from collections import defaultdict
from typing import Any, Callable, List, Optional, Union

import torch


def chunked_loss(
    student_hidden: torch.Tensor,
    student_head: torch.nn.Module,
    loss_fn: Callable[..., torch.Tensor],
    student_logits_fn: Optional[Callable[..., torch.Tensor]] = None,
    teacher_hidden: Optional[torch.Tensor] = None,
    teacher_head: Optional[torch.nn.Module] = None,
    teacher_logits_fn: Optional[Callable[[int, int], torch.Tensor]] = None,
    label: Optional[torch.Tensor] = None,
    chunk_size: int = 2048,
    reduction: str = "none",
    metric_fns: Optional[Union[Callable, List[Callable]]] = None,
    return_metrics: bool = False,
    **kwargs: Any,
):
    """Compute loss chunk by chunk without materializing full logits.

    Inputs are expected to be token-level tensors, e.g. [num_tokens, hidden_size].
    Loss is computed per chunk with reduction="none" and reduced globally here.
    """
    if reduction not in ("none", "sum", "mean"):
        raise ValueError(f"Unsupported reduction: {reduction}")
    if teacher_logits_fn is None and teacher_hidden is None and label is None:
        raise ValueError("Either teacher_logits_fn, teacher_hidden or label must be provided.")
    if teacher_logits_fn is None and teacher_hidden is not None and teacher_head is None:
        raise ValueError("teacher_head must be provided when teacher_hidden is provided.")

    losses = []
    metric_sums = defaultdict(lambda: student_hidden.new_zeros(()))
    total_loss = student_hidden.new_zeros(())
    total_tokens = 0

    for start in range(0, student_hidden.shape[0], chunk_size):
        end = start + chunk_size
        if student_logits_fn is not None:
            student_logits = student_logits_fn(student_hidden[start:end], skip=False)
        else:
            student_logits = student_head(student_hidden[start:end], skip=False)
        has_teacher_logits = False

        if teacher_logits_fn is not None:
            teacher_logits = teacher_logits_fn(start, end)
            vocab_size = min(student_logits.shape[-1], teacher_logits.shape[-1])
            student_logits = student_logits[..., :vocab_size]
            target = teacher_logits[..., :vocab_size]
            has_teacher_logits = True
        elif teacher_hidden is not None:
            teacher_logits = teacher_head(teacher_hidden[start:end])
            vocab_size = min(student_logits.shape[-1], teacher_logits.shape[-1])
            student_logits = student_logits[..., :vocab_size]
            target = teacher_logits[..., :vocab_size]
            has_teacher_logits = True
        else:
            target = label[start:end]

        chunk_loss = loss_fn(student_logits, target, reduction="none", **kwargs)
        chunk_tokens = chunk_loss.numel()
        if metric_fns is not None and has_teacher_logits:
            fns = metric_fns if isinstance(metric_fns, list) else [metric_fns]
            for fn in fns:
                metrics = fn(
                    student_logits=student_logits,
                    teacher_logits=target,
                    start=start,
                    end=end,
                )
                for key, value in metrics.items():
                    metric_sums[key] += value * chunk_tokens
        if reduction == "none":
            losses.append(chunk_loss)
        else:
            total_loss = total_loss + chunk_loss.sum()
            total_tokens += chunk_tokens

    if reduction == "none":
        loss = torch.cat(losses, dim=0)
    else:
        loss = total_loss if reduction == "sum" else total_loss / total_tokens
    if not return_metrics:
        return loss
    if reduction == "mean":
        metric_sums = {key: value / total_tokens for key, value in metric_sums.items()}
    return loss, metric_sums

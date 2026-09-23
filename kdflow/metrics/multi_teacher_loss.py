import torch


def prepare_multi_teacher_loss_stats(teacher_keys, routing_keys, loss_mask):
    """Build a metric function that sums each teacher's loss and token count."""
    teacher_keys = sorted(teacher_keys)
    teacher_indices = {key: i for i, key in enumerate(teacher_keys)}
    sample_teachers = torch.tensor(
        [teacher_indices[key] for key in routing_keys], device=loss_mask.device,
    )
    token_teachers = sample_teachers.repeat_interleave(loss_mask.sum(dim=1))

    @torch.no_grad()
    def multi_teacher_loss_stats_fn(chunk_loss, start, end, **kwargs):
        chunk_teachers = token_teachers[start:end]
        stats = {}
        for i, key in enumerate(teacher_keys):
            mask = chunk_teachers == i
            stats[f"distill/multi_teacher_kd_loss/teacher_{key}"] = (chunk_loss[mask].float().sum(), mask.sum())
        return stats

    return multi_teacher_loss_stats_fn

from kdflow.metrics.topk_token_overlap import compute_topk_token_overlap_ratios
from kdflow.metrics.entropy import compute_entropy

__all__ = [
    "compute_topk_token_overlap_ratios",
    "compute_entropy",
    "accumulate_metric_stats",
    "average_metric_stats",
]


def accumulate_metric_stats(target, stats, weight=1):
    """Accumulate metric sums and counts without averaging intermediate batches."""
    for key, (total, count) in stats.items():
        previous_total, previous_count = target.get(key, (0, 0))
        target[key] = (previous_total + total * weight, previous_count + count * weight)


def average_metric_stats(stats):
    """Compute means for metrics with at least one observation."""
    return {key: total / count for key, (total, count) in stats.items() if count > 0}

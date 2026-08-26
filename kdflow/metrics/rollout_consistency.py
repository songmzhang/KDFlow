import torch


@torch.no_grad()
def compute_rollout_consistency(
    student_logits,
    labels,
    rollout_log_probs,
    start,
    end,
    **kwargs,
):
    student_logits = student_logits.float()
    labels = labels[start:end]
    rollout_log_probs = rollout_log_probs[start:end].float()
    training_log_probs = student_logits.gather(
        dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1) - torch.logsumexp(student_logits, dim=-1)
    log_ratio = training_log_probs - rollout_log_probs
    k3_kl = log_ratio.exp() - log_ratio - 1
    return {
        "rollout_corr/kl": -log_ratio.mean(),
        "rollout_corr/k3_kl": k3_kl.mean(),
    }

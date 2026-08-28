import torch
import torch.nn.functional as F


def compute_entropy(student_logits, **kwargs):
    """Compute the entropy of a logits distribution using numerically stable formula.

    Args:
        student_logits: Tensor of shape (num_tokens, vocab_size).
        **kwargs: Unused. Accepts extra arguments for unified metric interface.

    Returns:
        A dict containing the mean entropy value.
    """
    with torch.no_grad():
        chunk_tokens = 2048
        entropy_parts = []
        for chunk in student_logits.split(chunk_tokens, dim=0):
            chunk = chunk.float()
            # H = logsumexp(z) - sum(softmax(z) * z)
            #   = max + log(sum(exp(z - max))) - sum(softmax * z)  (numerically stable)
            logits_max = chunk.max(dim=-1, keepdim=True).values
            exp_logits = (chunk - logits_max).exp_()
            sum_exp = exp_logits.sum(dim=-1, keepdim=True)
            softmax = exp_logits.div_(sum_exp)
            sum_softmax_logits = (softmax * chunk).sum(dim=-1)
            entropy_parts.append(
                logits_max.squeeze(-1) + sum_exp.log().squeeze(-1) - sum_softmax_logits
            )
        entropy = (
            torch.cat(entropy_parts) if len(entropy_parts) > 1 else entropy_parts[0]
        )
    return {"distill/student_entropy": entropy.mean()}

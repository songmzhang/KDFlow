import torch


def compute_topk_token_overlap_ratios(student_logits, teacher_logits, topks=(4, 16, 64), **kwargs):
    """Compute top-k token overlap ratios between student and teacher logits."""
    overlap_ratios = {}
    with torch.no_grad():
        n = student_logits.shape[0]
        chunk_tokens = 2048
        overlap_sums = {topk: student_logits.new_zeros(()) for topk in topks}
        for s_chunk, t_chunk in zip(
            student_logits.split(chunk_tokens, dim=0),
            teacher_logits.split(chunk_tokens, dim=0),
        ):
            for topk in topks:
                k = min(topk, s_chunk.shape[-1])
                student_topk = s_chunk.topk(k=k, dim=-1).indices
                teacher_topk = t_chunk.topk(k=k, dim=-1).indices
                token_overlap_ratio = (
                    (student_topk.unsqueeze(-1) == teacher_topk.unsqueeze(-2))
                    .any(dim=-1)
                    .float()
                    .sum(dim=-1)
                    / k
                )
                overlap_sums[topk] += token_overlap_ratio.sum()
        for topk in topks:
            key = f"distill/teacher_student_token_overlap/top{topk}"
            overlap_ratios[key] = overlap_sums[topk] / n
    return overlap_ratios

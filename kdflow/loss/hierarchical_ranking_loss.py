import torch
import torch.nn.functional as F 

from kdflow.loss import register_loss


@register_loss("hrl")
@torch.compile()
def compute_hierarchical_ranking_loss(
    student_logits,
    teacher_logits, 
    temperature=1.0,
    hr_topk=5,
    reduction="none",
    **kwargs
):
    student_probs = torch.softmax(student_logits, -1, dtype=torch.float32)
    teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
    
    stu_topk_probs, stu_topk_idxs = torch.topk(student_probs, k=hr_topk, dim=-1)
    tea_topk_probs, tea_topk_idxs = torch.topk(teacher_probs, k=hr_topk, dim=-1)
    
    stu_co_topk_probs = student_probs.gather(-1, tea_topk_idxs)
    tea_co_topk_probs = teacher_probs.gather(-1, stu_topk_idxs)
    
    stu_topk_margin = stu_co_topk_probs[..., :1] - stu_co_topk_probs
    stu_other_margin = stu_co_topk_probs.unsqueeze(-1) - stu_topk_probs.unsqueeze(-2)
    tea_other_margin = stu_topk_probs.unsqueeze(-1) - tea_co_topk_probs.unsqueeze(-2)
    topk_rank_loss = torch.maximum(torch.zeros_like(stu_topk_margin), -stu_topk_margin)
    topk_rank_loss = topk_rank_loss.sum(-1)
    other_rank_loss = torch.maximum(torch.zeros_like(tea_other_margin), -stu_other_margin)
    other_rank_loss = (other_rank_loss * tea_other_margin.gt(0)).sum(-1).sum(-1)
    rank_loss = topk_rank_loss + other_rank_loss
    
    if reduction == "mean":
        return rank_loss.mean()
    elif reduction == "sum":
        return rank_loss.sum()
    
    return rank_loss
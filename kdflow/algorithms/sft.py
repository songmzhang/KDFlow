import torch
import torch.nn.functional as F

from kdflow.algorithms import register_algorithm
from kdflow.loss.cross_entropy import compute_cross_entropy


@register_algorithm("sft")
class SFT:
    def __init__(self, strategy, student_model, **kwargs):
        self.strategy = strategy
        self.args = strategy.args
        self.student = student_model
    
    def training_step(self, micro_batch):
        student_input_ids = micro_batch["stu_input_ids"]
        student_attn_mask = micro_batch["stu_attn_mask"]
        student_loss_mask = micro_batch["stu_loss_mask"].bool()
        avg_token_num = micro_batch["avg_micro_batch_token_num"]

        mm_kwargs = {k[3:]: v for k, v in micro_batch.items() if k.startswith("mm_")}

        output = self.student(
            student_input_ids,
            attention_mask=student_attn_mask,
            allgather_logits=True,
            ring_attn_group=self.strategy.ring_attn_group,
            **mm_kwargs,
        )
        student_logits = output["logits"]
        student_logits = student_logits[student_loss_mask]
        
        loss_info = {}
        V = student_logits.shape[-1]
        student_label_ids = student_input_ids.roll(shifts=-1, dims=1)[student_loss_mask]
        ce_loss = compute_cross_entropy(student_logits, student_label_ids, reduction="sum") / avg_token_num
        loss = ce_loss
        loss_info["loss"] = loss
        loss_info["ce_loss"] = ce_loss

        return loss_info
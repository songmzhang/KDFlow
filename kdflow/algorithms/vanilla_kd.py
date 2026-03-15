import torch
import torch.nn.functional as F

from kdflow.loss import build_loss_fn
from kdflow.algorithms import register_algorithm
from kdflow.loss.cross_entropy import compute_cross_entropy


@register_algorithm("vanilla_kd")
class VanillaKD:
    def __init__(self, strategy, student_model, teacher_lm_head, **kwargs):
        self.strategy = strategy
        self.args = strategy.args
        self.student = student_model
        self.teacher_lm_head = teacher_lm_head
        self.loss_fn = build_loss_fn(self.args.kd.kd_loss_fn, self.args)
    
    def training_step(self, micro_batch):
        student_input_ids = micro_batch["stu_input_ids"]
        student_attn_mask = micro_batch["stu_attn_mask"]
        student_loss_mask = micro_batch["stu_loss_mask"].bool()
        teacher_input_ids = micro_batch["tea_input_ids"]
        teacher_attn_mask = micro_batch["tea_attn_mask"]
        teacher_loss_mask = micro_batch["tea_loss_mask"].bool()
        teacher_hiddens = micro_batch.get("teacher_hiddens", None)
        avg_token_num = micro_batch["avg_micro_batch_token_num"]

        assert teacher_hiddens is not None, "micro_batch must contain `teacher_hiddens` for KD"

        mm_kwargs = {k[3:]: v for k, v in micro_batch.items() if k.startswith("mm_")}

        output = self.student(
            student_input_ids,
            attention_mask=student_attn_mask,
            allgather_logits=True,
            ring_attn_group=self.strategy.ring_attn_group,
            **mm_kwargs,
        )
        student_logits = output["logits"]

        teacher_hiddens = teacher_hiddens.to(self.teacher_lm_head.weight)
        teacher_logits = self.teacher_lm_head(teacher_hiddens)
        
        student_logits = student_logits[student_loss_mask]
        minV = min(teacher_logits.shape[-1], student_logits.shape[-1])
        teacher_logits = teacher_logits[:, :minV]
        student_logits = student_logits[:, :minV]
        assert teacher_logits.shape == student_logits.shape, \
            "teacher_logits must have the same shape with student_logits, " \
            f"but got teacher: {teacher_logits.shape} and student: {student_logits.shape}."
        
        kd_loss = self.loss_fn(
            student_logits, 
            teacher_logits, 
            reduction="none",
        )
        kd_loss = kd_loss.sum() / avg_token_num
        loss_info = {"loss": kd_loss, "kd_loss": kd_loss}
        
        if self.args.kd.kd_ratio < 1:
            student_label_ids = student_input_ids.roll(shifts=-1, dims=1)[student_loss_mask]
            ce_loss = compute_cross_entropy(student_logits, student_label_ids, reduction="sum") / avg_token_num
            loss = (1 - self.args.kd.kd_ratio) * ce_loss + self.args.kd.kd_ratio * kd_loss
            loss_info["loss"] = loss
            loss_info["ce_loss"] = ce_loss

        return loss_info
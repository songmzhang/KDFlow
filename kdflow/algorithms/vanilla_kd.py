from functools import partial

import torch

from kdflow.algorithms import register_algorithm
from kdflow.loss import build_loss_fn
from kdflow.loss.chunked_loss import chunked_loss
from kdflow.loss.cross_entropy import compute_cross_entropy
from kdflow.metrics.entropy import compute_entropy
from kdflow.metrics.rollout_consistency import compute_rollout_consistency
from kdflow.metrics.topk_token_overlap import compute_topk_token_overlap_ratios


@register_algorithm("vanilla_kd")
class VanillaKD:
    def __init__(self, strategy, student_model, teacher_lm_head, **kwargs):
        self.strategy = strategy
        self.args = strategy.args
        self.student = student_model
        self.teacher_lm_head = teacher_lm_head
        self.loss_fn = build_loss_fn(self.args.kd.kd_loss_fn, self.args)
        # certain metrics will be recorded during training
        self.metric_fns = [compute_topk_token_overlap_ratios]
        if self.args.scenario == "on_policy_kd":
            self.metric_fns.append(compute_entropy)

    def compute_multi_teacher_logits(self, teacher_hiddens, teacher_loss_mask, routing_keys, start=None, end=None):
        per_sample_counts = teacher_loss_mask.sum(dim=1).tolist()
        if isinstance(teacher_hiddens, torch.Tensor):
            splits = teacher_hiddens.split(per_sample_counts, dim=0)
        else:
            if len(teacher_hiddens) != len(per_sample_counts):
                raise ValueError("teacher_hiddens must contain one tensor per sample")
            splits = teacher_hiddens
        if start is not None or end is not None:
            start = 0 if start is None else start
            end = sum(per_sample_counts) if end is None else end
            offsets = [0]
            for count in per_sample_counts:
                offsets.append(offsets[-1] + count)
            splits = [x[max(start - offsets[i], 0): max(min(end, offsets[i + 1]) - offsets[i], 0)] for i, x in enumerate(splits)]
        teacher_to_indices = {}
        for i, key in enumerate(routing_keys):
            if splits[i].numel() > 0:
                teacher_to_indices.setdefault(key, []).append(i)

        logits_list = [None] * len(routing_keys)
        streams = {key: torch.cuda.Stream() for key in teacher_to_indices}
        for key, indices in teacher_to_indices.items():
            with torch.cuda.stream(streams[key]):
                lm_head = self.teacher_lm_head[key]
                batched = torch.cat([splits[i] for i in indices], dim=0).to(lm_head.weight)
                batched_logits = lm_head(batched)
                sizes = [splits[i].shape[0] for i in indices]
                per_sample_logits = batched_logits.split(sizes, dim=0)
                for idx, i in enumerate(indices):
                    logits_list[i] = per_sample_logits[idx]

        for s in streams.values():
            torch.cuda.current_stream().wait_stream(s)

        return torch.cat([x for x in logits_list if x is not None], dim=0)

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

        mm_kwargs = micro_batch.get("stu_multi_modal_inputs") or {}

        output = self.student(
            student_input_ids,
            attention_mask=student_attn_mask,
            allgather_logits=True,
            ring_attn_group=self.strategy.ring_attn_group,
            **mm_kwargs,
        )
        student_hiddens = output["hidden_states"][-1][student_loss_mask]
        del output

        student_label_ids = student_input_ids.roll(shifts=-1, dims=1)[
            student_loss_mask
        ]
        metric_fns = self.metric_fns
        rollout_log_probs = micro_batch.get("rollout_log_probs")
        if rollout_log_probs is not None:
            metric_fns = [
                *metric_fns,
                partial(
                    compute_rollout_consistency,
                    labels=student_label_ids,
                    rollout_log_probs=rollout_log_probs[student_loss_mask],
                ),
            ]

        # Non-chunked case can be regarded as a special case of chunked loss (i.e., chunk_size = seq_len)
        chunk_size = self.args.train.chunked_loss_size or student_hiddens.shape[0]

        if isinstance(self.teacher_lm_head, dict):  # multi-teacher distillation
            teacher_logits_fn = lambda start, end: self.compute_multi_teacher_logits(
                teacher_hiddens, teacher_loss_mask, micro_batch["teacher_routing_key"], start, end
            )
            kd_loss, metric_sums = chunked_loss(
                student_hiddens, self.student.model.lm_head, self.loss_fn,
                teacher_logits_fn=teacher_logits_fn, chunk_size=chunk_size, reduction="sum",
                metric_fns=metric_fns, return_metrics=True,
            )
        else:
            teacher_hiddens = teacher_hiddens.to(self.teacher_lm_head.weight)
            kd_loss, metric_sums = chunked_loss(
                student_hiddens, self.student.model.lm_head, self.loss_fn,
                teacher_hidden=teacher_hiddens, teacher_head=self.teacher_lm_head,
                chunk_size=chunk_size, reduction="sum",
                metric_fns=metric_fns, return_metrics=True,
            )
        kd_loss = kd_loss / avg_token_num
        loss_info = {"train/loss": kd_loss, "train/kd_loss": kd_loss}
        loss_info.update({key: value / avg_token_num for key, value in metric_sums.items()})

        if self.args.kd.kd_ratio < 1:
            ce_loss = chunked_loss(
                student_hiddens, self.student.model.lm_head, compute_cross_entropy,
                label=student_label_ids, chunk_size=chunk_size, reduction="sum"
            ) / avg_token_num
            loss = (1 - self.args.kd.kd_ratio) * ce_loss + self.args.kd.kd_ratio * kd_loss
            loss_info["train/loss"] = loss
            loss_info["train/ce_loss"] = ce_loss

        return loss_info

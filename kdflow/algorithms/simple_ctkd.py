import torch
import torch.nn.functional as F

from kdflow.loss import build_loss_fn
from kdflow.algorithms import register_algorithm
from kdflow.loss.cross_entropy import compute_cross_entropy


@register_algorithm("simple_ctkd")
class SimpleCrossTokenizerKD:
    """Simply find the overlap tokens between student and teacher tokenizer, and only compute KD loss on this sub-vocabulary. 
    Motivation: modern LLMs have a large amount of shared tokens.
    """
    def __init__(
        self, 
        strategy, 
        student_model, 
        teacher_lm_head, 
        student_tokenizer,
        teacher_tokenizer,
        **kwargs
    ):
        self.strategy = strategy
        self.args = strategy.args
        self.student = student_model
        self.teacher_lm_head = teacher_lm_head
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.student_overlap_token_ids, self.teacher_overlap_token_ids = self._find_overlap_tokens()
        self.loss_fn = build_loss_fn(self.args.kd.kd_loss_fn, self.args)
        
    def _find_overlap_tokens(self):
        student_vocab = {k.replace("Ġ", "▁"): v for k, v in self.student_tokenizer.get_vocab().items()}
        teacher_vocab = {k.replace("Ġ", "▁"): v for k, v in self.teacher_tokenizer.get_vocab().items()}
        overlap_tokens = set(student_vocab.keys()) & set(teacher_vocab.keys())
        student_overlap_token_ids = torch.tensor([student_vocab[token] for token in overlap_tokens], dtype=torch.long, device=self.teacher_lm_head.weight.device)
        teacher_overlap_token_ids = torch.tensor([teacher_vocab[token] for token in overlap_tokens], dtype=torch.long, device=self.teacher_lm_head.weight.device)
        return student_overlap_token_ids, teacher_overlap_token_ids
    
    def _align_sequences(self, tea_seq, stu_seq):
        i, j = 0, 0
        t2s_align, s2t_align = [], []
        history_tea_seq, history_stu_seq = "", ""

        tea_eos = self.teacher_tokenizer.eos_token
        stu_eos = self.student_tokenizer.eos_token

        tea_seq = [token.replace('▁', '').replace('Ġ', '') for token in tea_seq]
        stu_seq = [token.replace('▁', '').replace('Ġ', '') for token in stu_seq]

        while i < len(tea_seq) and j < len(stu_seq):
            is_eos_match = (tea_seq[i] == tea_eos and stu_seq[j] == stu_eos)
            if history_tea_seq == history_stu_seq and (
                tea_seq[i] == stu_seq[j] or is_eos_match
            ):
                common_text = tea_seq[i]
                history_tea_seq += common_text
                history_stu_seq += common_text
                t2s_align.append(i)
                s2t_align.append(j)
                i += 1
                j += 1
            elif len(history_tea_seq) > len(history_stu_seq):
                history_stu_seq += stu_seq[j]
                j += 1
            elif len(history_tea_seq) < len(history_stu_seq):
                history_tea_seq += tea_seq[i]
                i += 1
            else:
                history_tea_seq += tea_seq[i]
                history_stu_seq += stu_seq[j]
                i += 1
                j += 1

        return t2s_align, s2t_align
    
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

        _skip = {"stu_input_ids", "stu_attn_mask", "stu_loss_mask", "tea_input_ids", "tea_attn_mask", "tea_loss_mask", "teacher_hiddens", "avg_micro_batch_token_num"}
        mm_kwargs = {k[4:]: v for k, v in micro_batch.items() if k.startswith("stu_") and k not in _skip}

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
        
        student_label_ids = student_input_ids.roll(shifts=-1, dims=1)[student_loss_mask]
        teacher_label_ids = teacher_input_ids.roll(shifts=-1, dims=1)[teacher_loss_mask]
        teacher_aligned_idx, student_aligned_idx = self._align_sequences(
            self.teacher_tokenizer.convert_ids_to_tokens(teacher_label_ids.cpu().tolist()),
            self.student_tokenizer.convert_ids_to_tokens(student_label_ids.cpu().tolist())
        )
        
        aligned_student_logits = student_logits[student_aligned_idx][:, self.student_overlap_token_ids]
        aligned_teacher_logits = teacher_logits[teacher_aligned_idx][:, self.teacher_overlap_token_ids]
        assert aligned_teacher_logits.shape == aligned_student_logits.shape
        
        align_ratio = torch.tensor(len(student_aligned_idx) / len(student_label_ids))
        
        kd_loss = self.loss_fn(
            aligned_student_logits, 
            aligned_teacher_logits, 
            reduction="none",
        )
        kd_loss = kd_loss.sum() / avg_token_num
        loss_info = {"loss": kd_loss, "kd_loss": kd_loss, "align_ratio": align_ratio}
        
        if self.args.kd.kd_ratio < 1:
            ce_loss = compute_cross_entropy(student_logits, student_label_ids, reduction="sum") / avg_token_num
            loss = (1 - self.args.kd.kd_ratio) * ce_loss + self.args.kd.kd_ratio * kd_loss
            loss_info["loss"] = loss
            loss_info["ce_loss"] = ce_loss

        return loss_info
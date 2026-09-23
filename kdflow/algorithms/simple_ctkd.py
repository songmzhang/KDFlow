import torch

from kdflow.loss import build_loss_fn
from kdflow.algorithms import register_algorithm
from kdflow.loss.chunked_loss import chunked_loss
from kdflow.loss.cross_entropy import compute_cross_entropy
from kdflow.metrics.entropy import compute_entropy
from kdflow.utils.logging_utils import init_logger


logger = init_logger(__name__)

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
        # certain metrics will be recorded during training
        self.metric_fns = [compute_entropy] if self.args.scenario == "on_policy_kd" else []
        
    def _find_overlap_tokens(self):
        student_vocab = {k.replace("Ġ", "▁"): v for k, v in self.student_tokenizer.get_vocab().items()}
        teacher_vocab = {k.replace("Ġ", "▁"): v for k, v in self.teacher_tokenizer.get_vocab().items()}
        overlap_tokens = set(student_vocab.keys()) & set(teacher_vocab.keys())
        student_ids = [student_vocab[token] for token in overlap_tokens]
        teacher_ids = [teacher_vocab[token] for token in overlap_tokens]
        stu_eos, tea_eos = self.student_tokenizer.eos_token_id, self.teacher_tokenizer.eos_token_id
        if stu_eos not in student_ids or tea_eos not in teacher_ids:
            student_ids.append(stu_eos)
            teacher_ids.append(tea_eos)
        device = self.teacher_lm_head.weight.device
        logger.info(f"Num of overlap_tokens between student & teacher: {len(student_ids)}")
        return torch.tensor(student_ids, dtype=torch.long, device=device), torch.tensor(teacher_ids, dtype=torch.long, device=device)
    
    def _align_sequences(self, tea_seq, stu_seq):
        i, j = 0, 0
        t2s_align, s2t_align = [], []
        history_tea_seq, history_stu_seq = "", ""

        tea_eos = self.teacher_tokenizer.eos_token
        stu_eos = self.student_tokenizer.eos_token

        tea_seq = [token.replace('▁', '').replace('Ġ', '') for token in tea_seq]
        stu_seq = [token.replace('▁', '').replace('Ġ', '') for token in stu_seq]

        if tea_seq == stu_seq:
            indices = list(range(len(tea_seq)))
            return indices, indices

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

        teacher_hiddens = teacher_hiddens.to(self.teacher_lm_head.weight)
        
        student_label_ids = student_input_ids.roll(shifts=-1, dims=1)[student_loss_mask]
        teacher_label_ids = teacher_input_ids.roll(shifts=-1, dims=1)[teacher_loss_mask]
        teacher_aligned_idx, student_aligned_idx = self._align_sequences(
            self.teacher_tokenizer.convert_ids_to_tokens(teacher_label_ids.cpu().tolist()),
            self.student_tokenizer.convert_ids_to_tokens(student_label_ids.cpu().tolist())
        )
        
        align_ratio = torch.tensor(len(student_aligned_idx) / len(student_label_ids))
        aligned_student_hiddens = student_hiddens[student_aligned_idx]
        aligned_teacher_hiddens = teacher_hiddens[teacher_aligned_idx]
        
        # Non-chunked case can be regarded as a special case of chunked loss (i.e., chunk_size = seq_len)
        chunk_size = self.args.train.chunked_loss_size or aligned_student_hiddens.shape[0]

        student_overlap_ids = self.student_overlap_token_ids
        teacher_overlap_ids = self.teacher_overlap_token_ids
        student_lm_head = self.student.model.lm_head
        teacher_lm_head = self.teacher_lm_head

        def student_logits_fn(hidden_chunk, skip=False):
            return student_lm_head(hidden_chunk, skip=skip)[:, student_overlap_ids]

        def teacher_logits_fn(start, end):
            return teacher_lm_head(aligned_teacher_hiddens[start:end])[:, teacher_overlap_ids]

        kd_loss, metric_stats = chunked_loss(
            aligned_student_hiddens, student_lm_head, self.loss_fn,
            student_logits_fn=student_logits_fn,
            teacher_logits_fn=teacher_logits_fn,
            chunk_size=chunk_size, reduction="sum",
            metric_fns=self.metric_fns, return_metrics=True,
        )
        kd_loss = kd_loss / avg_token_num

        loss_info = {
            "train/loss": kd_loss,
            "train/kd_loss": kd_loss,
            "distill/alignment_ratio": align_ratio,
        }
        loss_info["metric_stats"] = metric_stats

        if self.args.kd.kd_ratio < 1:
            ce_loss = chunked_loss(
                student_hiddens, self.student.model.lm_head, compute_cross_entropy,
                label=student_label_ids, chunk_size=chunk_size, reduction="sum"
            ) / avg_token_num
            loss = (1 - self.args.kd.kd_ratio) * ce_loss + self.args.kd.kd_ratio * kd_loss
            loss_info["train/loss"] = loss
            loss_info["train/ce_loss"] = ce_loss

        return loss_info

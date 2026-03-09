import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from kdflow.loss import build_loss_fn
from kdflow.algorithms import register_algorithm
from kdflow.loss.cross_entropy import compute_cross_entropy
from kdflow.utils.logging_utils import init_logger


logger = init_logger(__name__)

@register_algorithm("dskd")
class DSKD:
    def __init__(
        self, 
        strategy, 
        student_model, 
        teacher_lm_head, 
        student_tokenizer,
        teacher_tokenizer,
        tokenizer_info=None,
        **kwargs,
    ):
        self.strategy = strategy
        self.args = strategy.args
        self.student = student_model
        self.student_lm_head = student_model.model.lm_head.weight.full_tensor().detach().clone()
        self.teacher_lm_head = teacher_lm_head
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.tokenizer_info = tokenizer_info
        self.template_identical = tokenizer_info.template_identical if tokenizer_info else True
        self.vocab_identical = tokenizer_info.vocab_identical if tokenizer_info else True
        self.loss_fn = build_loss_fn(self.args.kd.kd_loss_fn, self.args)
        
        self._init_projectors()
        
    def _init_projectors(self):
        """
        projector initialization aims to achieve logit equivalence (take W^{t2s} as an example): 
        t_logits = t2s_logits
        => H^t * W^t = H^t * W^{t2s} * W^s
        => W^{t2s} = W^t * pseudo_inverse(W^s)
        
        s2t_projector (W^{s2t}) will be initialized in the criterion if args.init_s2t_projector is True
        """
        self.t2s_projector = nn.Linear(self.teacher_lm_head.weight.shape[1], self.student.hidden_size, bias=False)
        if self.args.kd.dskd_token_align == "cma":
            self.query_projector = nn.Linear(self.student.hidden_size * 2, self.teacher_lm_head.weight.shape[1] * 2, bias=False)
            
        student_head = self.student_lm_head.detach().transpose(0, 1)  
        teacher_head = self.teacher_lm_head.weight.transpose(0, 1) 

        if self.vocab_identical:
            if self.args.kd.dskd_topk_vocab != -1:     # only use part of vocab to reduce initialization error
                part_student_head = student_head[:, :self.args.kd.dskd_topk_vocab]
                part_teacher_head = teacher_head[:, :self.args.kd.dskd_topk_vocab]
            else:
                part_student_head = student_head
                part_teacher_head = teacher_head
        else:  # different vocab: only use the overlapped part of both vocabularies
            student_vocab = {k.replace("Ġ", "▁"): v for k, v in self.student_tokenizer.get_vocab().items()}
            teacher_vocab = {k.replace("Ġ", "▁"): v for k, v in self.teacher_tokenizer.get_vocab().items()}
            overlap_tokens = [k for k in student_vocab if k in teacher_vocab]
            logger.info(f"Found overlap tokens of two tokenizers: {len(overlap_tokens)}")
            student_overlap_token_ids = torch.tensor([student_vocab[token] for token in overlap_tokens], dtype=torch.long, device=student_head.device)
            teacher_overlap_token_ids = torch.tensor([teacher_vocab[token] for token in overlap_tokens], dtype=torch.long, device=teacher_head.device)
            part_student_head = student_head[:, student_overlap_token_ids]
            part_teacher_head = teacher_head[:, teacher_overlap_token_ids]
            self.student_overlap_token_ids = student_overlap_token_ids
            if self.args.kd.dskd_topk_vocab != -1:
                part_student_head = part_student_head[:, :self.args.kd.dskd_topk_vocab]
                part_teacher_head = part_teacher_head[:, :self.args.kd.dskd_topk_vocab]

        logger.info("Init t2s projector through pseudo inverse")
        part_student_head_pinv = torch.linalg.pinv(part_student_head.float())
        init_t2s = (part_teacher_head.float() @ part_student_head_pinv).transpose(0, 1)
        self.t2s_projector.weight.data.copy_(init_t2s.to(student_head))

        logger.info("Init s2t projector through pseudo inverse")
        self.part_teacher_head_pinv = torch.linalg.pinv(part_teacher_head.float())
        # self.part_teacher_head_pinv.requires_grad = False
        
        # Move projectors to the same device as teacher_lm_head
        device = self.teacher_lm_head.weight.device
        self.t2s_projector = self.t2s_projector.to(device)
        if hasattr(self, 'query_projector'):
            self.query_projector = self.query_projector.to(device)

        # Register gradient clamp hooks to prevent gradient explosion
        grad_clamp_value = getattr(self.args.kd, 'projector_grad_clamp', 1.0)
        def _grad_clamp_hook(grad, name, clamp_value):
            return grad.clamp(-clamp_value, clamp_value)

        self.t2s_projector.weight.register_hook(
            lambda grad, n='t2s_projector.weight', v=grad_clamp_value: _grad_clamp_hook(grad, n, v)
        )
        if hasattr(self, 'query_projector'):
            self.query_projector.weight.register_hook(
                lambda grad, n='query_projector.weight', v=grad_clamp_value: _grad_clamp_hook(grad, n, v)
            )
        logger.info(f"Registered gradient clamp hooks on projectors with clamp_value={grad_clamp_value}")
    
    def get_projector_params(self):
        """Return projector parameters for optimizer registration with separate learning rate."""
        params = list(self.t2s_projector.parameters())
        if hasattr(self, 'query_projector'):
            params += list(self.query_projector.parameters())
        return params
    
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
        student_hiddens = output["hidden_states"][-1]
        student_logits = output["logits"]

        teacher_hiddens = teacher_hiddens.to(self.teacher_lm_head.weight)
        teacher_logits = self.teacher_lm_head(teacher_hiddens)

        student_hiddens = student_hiddens[student_loss_mask]
        student_logits = student_logits[student_loss_mask]
        
        if self.vocab_identical:
            loss_info = self._compute_dskd_loss(
                student_hiddens,
                teacher_hiddens,
                student_logits, 
                teacher_logits, 
                student_input_ids, 
                teacher_input_ids, 
                student_loss_mask, 
                teacher_loss_mask,
                avg_token_num
            )
        else:
            if self.args.kd.dskd_token_align == "cma":
                loss_info = self._compute_dskd_cma_loss(
                    student_hiddens,
                    teacher_hiddens,
                    student_logits, 
                    teacher_logits, 
                    student_input_ids, 
                    teacher_input_ids, 
                    student_loss_mask, 
                    teacher_loss_mask,
                    avg_token_num
                )
            elif self.args.kd.dskd_token_align == "eta":
                loss_info = self._compute_dskd_eta_loss(
                    student_hiddens,
                    teacher_hiddens,
                    student_logits, 
                    teacher_logits, 
                    student_input_ids, 
                    teacher_input_ids, 
                    student_loss_mask, 
                    teacher_loss_mask,
                    avg_token_num
                )
        
        if self.args.kd.kd_ratio < 1:
            V = student_logits.shape[-1]
            student_label_ids = student_input_ids.roll(shifts=-1, dims=1)[student_loss_mask]
            ce_loss = compute_cross_entropy(student_logits, student_label_ids, reduction="sum") / avg_token_num
            loss = (1 - self.args.kd.kd_ratio) * ce_loss + self.args.kd.kd_ratio * loss_info["kd_loss"]
            loss_info["loss"] = loss
            loss_info["ce_loss"] = ce_loss

        return loss_info
    
    def _compute_dskd_loss(
        self, 
        student_hiddens,
        teacher_hiddens,
        student_logits, 
        teacher_logits, 
        student_input_ids, 
        teacher_input_ids, 
        student_loss_mask, 
        teacher_loss_mask, 
        avg_token_num
    ):
        t2s_hiddens = self.t2s_projector(teacher_hiddens.float())
        t2s_logits = t2s_hiddens.float() @ self.student_lm_head.detach().transpose(-1, -2)

        t_preds = teacher_logits.argmax(-1)
        t2s_ce_loss = compute_cross_entropy(t2s_logits, t_preds, reduction="sum") / avg_token_num

        t2s_agreement_mask = t2s_logits.argmax(-1).eq(t_preds)
        t2s_agreement = t2s_agreement_mask.sum() / avg_token_num

        t2s_kd_loss = (self.loss_fn(
            student_logits, 
            t2s_logits.detach(),
            reduction="none"
        ) * t2s_agreement_mask).sum() / avg_token_num
        
        # === s2t path ===
        stu_lm_head = self.student_lm_head.detach().transpose(0, 1)
        if self.args.kd.dskd_topk_vocab != -1:
            stu_lm_head = stu_lm_head[:, :self.args.kd.dskd_topk_vocab]

        s2t_projector = stu_lm_head @ self.part_teacher_head_pinv
        s2t_hiddens = student_hiddens @ s2t_projector.to(student_hiddens)
        s2t_logits = self.teacher_lm_head(s2t_hiddens)
        
        minV = min(teacher_logits.shape[-1], s2t_logits.shape[-1])
        teacher_logits = teacher_logits[:, :minV]
        s2t_logits = s2t_logits[:, :minV]
        
        s2t_kd_loss = self.loss_fn(
            s2t_logits, 
            teacher_logits, 
            reduction="sum"
        ) / avg_token_num
        
        s2t_agreement = s2t_logits.argmax(-1).eq(student_logits.argmax(-1)).sum() / avg_token_num
        teacher_targets = teacher_input_ids.roll(shifts=-1, dims=1)[teacher_loss_mask]
        t_acc = (teacher_logits.argmax(-1).eq(teacher_targets)).sum() / avg_token_num
        t2s_acc = (t2s_logits.argmax(-1).eq(teacher_targets)).sum() / avg_token_num
        
        kd_loss = t2s_kd_loss + t2s_ce_loss + s2t_kd_loss
        
        loss_info = {
            "loss": kd_loss,
            "kd_loss": kd_loss,
            "t2s_ce_loss": t2s_ce_loss,
            "t2s_kd_loss": t2s_kd_loss,
            "t2s_agreement": t2s_agreement,
            "s2t_kd_loss": s2t_kd_loss,
            "s2t_agreement": s2t_agreement,
            "t_acc": t_acc,
            "t2s_acc": t2s_acc
        }
        
        return loss_info
    
    def _compute_dskd_cma_loss(
        self, 
        student_hiddens,
        teacher_hiddens,
        student_logits, 
        teacher_logits, 
        student_input_ids, 
        teacher_input_ids, 
        student_loss_mask, 
        teacher_loss_mask,
        avg_token_num
    ):  
        bsz = student_input_ids.shape[0]
        stu_counts = student_loss_mask.sum(dim=1)
        tea_counts = teacher_loss_mask.sum(dim=1)
        student_label_ids = student_input_ids.roll(shifts=-1, dims=1)[student_loss_mask]
        student_input_ids = student_input_ids[student_loss_mask]
        teacher_label_ids = teacher_input_ids.roll(shifts=-1, dims=1)[teacher_loss_mask]
        teacher_input_ids = teacher_input_ids[teacher_loss_mask]
        
        stu_sample_ids = torch.repeat_interleave(torch.arange(bsz), stu_counts).to(student_input_ids)  # [N]
        tea_sample_ids = torch.repeat_interleave(torch.arange(bsz), tea_counts).to(teacher_input_ids)  # [M]
        attn_mask = (stu_sample_ids.unsqueeze(1) == tea_sample_ids.unsqueeze(0))  # [N, M] 
        
        student_input_embeds = self.student_lm_head[student_input_ids]
        student_target_embeds = self.student_lm_head[student_label_ids]
        
        teacher_input_embeds = self.teacher_lm_head.weight[teacher_input_ids]
        teacher_target_embeds = self.teacher_lm_head.weight[teacher_label_ids]
        
        stu_index_embeds = torch.cat([student_input_embeds, student_target_embeds], -1)
        tea_index_embeds = torch.cat([teacher_input_embeds, teacher_target_embeds], -1)
        
        t_preds = teacher_logits.argmax(-1)
        
        stu_q_hiddens = self.query_projector(stu_index_embeds)
        tea_k_hiddens = tea_index_embeds.float()
        
        stu_lmhead = self.student_lm_head.detach().transpose(0, 1)
        stu_lmhead = stu_lmhead[:, self.student_overlap_token_ids]
        s2t_proj = stu_lmhead @ self.part_teacher_head_pinv
        stu_v_hiddens = student_hiddens @ s2t_proj.to(student_hiddens)
        
        tea_v_hiddens = self.t2s_projector(teacher_hiddens.float()).to(teacher_hiddens)
        
        align_attn = stu_q_hiddens.matmul(tea_k_hiddens.transpose(-1, -2))
        align_attn = align_attn / math.sqrt(2 * teacher_hiddens.shape[-1])
        align_attn = align_attn + (1.0 - attn_mask) * (-float("inf"))
        
        t2s_weight = torch.softmax(align_attn, -1).to(student_hiddens)      
        t2s_hiddens = t2s_weight.matmul(tea_v_hiddens)  # n x m x m x d -> n x d
        t2s_logits = t2s_hiddens.matmul(
            self.student_lm_head.detach().transpose(-1, -2)
        )  # n x d x d x V_stu -> n x V_stu  [bsz x seq-len x V_stu]
        
        t2s_acc_mask = t2s_logits.argmax(-1).eq(student_label_ids)
        t2s_acc = t2s_acc_mask.sum() / avg_token_num
        
        t2s_ce_loss = F.cross_entropy(t2s_logits, student_label_ids, reduction="sum") / avg_token_num
        t2s_kd_loss = self.loss_fn(
            student_logits, 
            t2s_logits.detach(),
            reduction="sum"
        ) / avg_token_num
        
        s2t_weight = torch.softmax(align_attn.transpose(-1, -2), -1).to(student_hiddens)
        s2t_hiddens = s2t_weight.matmul(stu_v_hiddens)  # m x n x n x D -> m x D
        s2t_logits = self.teacher_lm_head(s2t_hiddens)
        s2t_kd_loss = self.loss_fn(
            s2t_logits, 
            teacher_logits, 
            reduction="sum"
        ) / avg_token_num
        
        kd_loss = t2s_kd_loss + t2s_ce_loss + s2t_kd_loss
        loss_info = {
            "loss": kd_loss,
            "kd_loss": kd_loss,
            "t2s_ce_loss": t2s_ce_loss,
            "t2s_kd_loss": t2s_kd_loss,
            "s2t_kd_loss": s2t_kd_loss,
            "t2s_acc": t2s_acc
        }
        
        return loss_info
    
    def _compute_dskd_eta_loss(
        self, 
        student_hiddens,
        teacher_hiddens,
        student_logits, 
        teacher_logits, 
        student_input_ids, 
        teacher_input_ids, 
        student_loss_mask, 
        teacher_loss_mask,
        avg_token_num
    ):
        device = student_hiddens.device
        N = student_hiddens.shape[0]
        M = teacher_hiddens.shape[0]

        student_labels_flat = student_input_ids.roll(shifts=-1, dims=1)[student_loss_mask]
        teacher_labels_flat = teacher_input_ids.roll(shifts=-1, dims=1)[teacher_loss_mask]

        stu_lm_head = self.student_lm_head.detach().transpose(0, 1)
        stu_lm_head = stu_lm_head[:, self.student_overlap_token_ids]
        if self.args.kd.dskd_topk_vocab != -1:
            stu_lm_head = stu_lm_head[:, :self.args.kd.dskd_topk_vocab]
        s2t_proj = stu_lm_head @ self.part_teacher_head_pinv
        stu_v_hiddens = student_hiddens @ s2t_proj.to(student_hiddens)

        tea_v_hiddens = self.t2s_projector(teacher_hiddens.float()).to(teacher_hiddens)

        t_preds = teacher_logits.argmax(-1)

        t2s_hiddens_align = torch.zeros(N, tea_v_hiddens.shape[-1], device=device, dtype=tea_v_hiddens.dtype)
        s2t_hiddens_align = torch.zeros(M, stu_v_hiddens.shape[-1], device=device, dtype=stu_v_hiddens.dtype)
        t_preds_as_label = torch.full((N,), -100, device=device, dtype=torch.long)

        tea_tokens = self.teacher_tokenizer.convert_ids_to_tokens(teacher_labels_flat)
        stu_tokens = self.student_tokenizer.convert_ids_to_tokens(student_labels_flat)
        align_t_idx, align_s_idx = self._align_sequences(tea_tokens, stu_tokens)

        for _t_idx, _s_idx in zip(align_t_idx, align_s_idx):
            tmp_t_token = self.teacher_tokenizer.convert_ids_to_tokens(
                [t_preds[_t_idx]]
            )
            if t_preds[_t_idx] == self.teacher_tokenizer.eos_token_id:
                t_preds_as_label[_s_idx] = self.student_tokenizer.eos_token_id
                t2s_hiddens_align[_s_idx] = tea_v_hiddens[_t_idx]
                s2t_hiddens_align[_t_idx] = stu_v_hiddens[_s_idx]
            else:
                try:
                    tmp = self.student_tokenizer.convert_tokens_to_ids(tmp_t_token)
                    if len(tmp) == 1 and tmp[0] is not None:
                        t_preds_as_label[_s_idx] = tmp[0]
                        t2s_hiddens_align[_s_idx] = tea_v_hiddens[_t_idx]
                        s2t_hiddens_align[_t_idx] = stu_v_hiddens[_s_idx]
                except:
                    pass

        align_ratio = len(align_s_idx) / max(N, 1)

        t2s_logits = t2s_hiddens_align.matmul(
            self.student_lm_head.to(t2s_hiddens_align).detach().transpose(-1, -2)
        )

        stu_align_token_num = max(1e-3, t_preds_as_label.ne(-100).sum().item())
        t2s_agreement_mask = t2s_logits.argmax(-1).eq(t_preds_as_label)
        t2s_agreement = (
            (t2s_agreement_mask * t_preds_as_label.ne(-100)).sum() / stu_align_token_num
        )

        t2s_acc = (t2s_logits.argmax(-1).eq(student_labels_flat)).sum() / avg_token_num

        t2s_ce_loss = F.cross_entropy(
            t2s_logits, t_preds_as_label, ignore_index=-100, reduction="sum"
        ) / stu_align_token_num

        t2s_kd_loss = self.loss_fn(
            student_logits,
            t2s_logits.detach(),
            reduction="sum"
        ) / avg_token_num

        s2t_logits = self.teacher_lm_head(s2t_hiddens_align)
        s2t_valid_mask = ~s2t_hiddens_align.eq(0).all(-1)
        s2t_kd_loss = self.loss_fn(
            s2t_logits,
            teacher_logits,
            reduction="none"
        )
        s2t_kd_loss = (s2t_kd_loss * s2t_valid_mask.float()).sum() / max(s2t_valid_mask.sum().item(), 1e-8)

        kd_loss = t2s_kd_loss + t2s_ce_loss + s2t_kd_loss

        loss_info = {
            "loss": kd_loss,
            "kd_loss": kd_loss,
            "t2s_ce_loss": t2s_ce_loss,
            "t2s_kd_loss": t2s_kd_loss,
            "t2s_agreement": t2s_agreement,
            "s2t_kd_loss": s2t_kd_loss,
            "t2s_acc": t2s_acc,
            "align_ratio": torch.tensor(align_ratio, device=device),
        }

        return loss_info
    
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
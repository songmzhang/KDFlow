from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoConfig

from kdflow.utils import get_tokenizer
from kdflow.models.ring_attn_utils import gather_and_pad_tensor, unpad_and_slice_tensor


class DistillModel(nn.Module):
    """
    Base class for student models in knowledge distillation (modified from OpenRLHF/openrlhf/models/actor.py).

    Args:
        args (Arguments): Arguments.
        strategy (Strategy): Strategy for student model loading and training.
        device_map (dict, optional): Device mapping for loading the model onto specific devices. Defaults to None.
    """

    def __init__(
        self,
        strategy,
        device_map=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.args = strategy.args
        self.temperature = self.args.rollout.temperature
        model_name_or_path = self.args.model.student_name_or_path

        # Support multiple attention mechanism implementations
        attn_impl = self.args.model.attn_implementation

        self.model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        
        # Determine if this is a Vision-Language model
        self.is_vl_model = hasattr(self.model_config, "vision_config")
        
        if self.is_vl_model:
            model_class = AutoModelForImageTextToText
        elif self.args.model.use_liger_kernel:
            from liger_kernel.transformers import AutoLigerKernelForCausalLM
            model_class = AutoLigerKernelForCausalLM
        else:
            model_class = AutoModelForCausalLM

        if hasattr(self.model_config, "text_config"):
            self.hidden_size = self.model_config.text_config.hidden_size
        else:
            self.hidden_size = self.model_config.hidden_size
        
        self.model = strategy.load_hf_model(
            model_class, 
            model_name_or_path, 
            attn_impl, 
            self.model_config, 
        )
        
        # LoRA
        if self.args.model.lora_rank > 0:
            # https://github.com/huggingface/peft/issues/137
            self.model.enable_input_require_grads()
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.args.model.lora_rank,
                lora_alpha=self.args.model.lora_alpha,
                target_modules=self.args.model.target_modules,
                lora_dropout=self.args.model.lora_dropout,
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)

        self.tokenizer = get_tokenizer(model_name_or_path, self.model)

        # https://github.com/huggingface/transformers/issues/26877
        # Use `model.generate(use_cache=True)` instead.`
        self.model.config.use_cache = False

        # packing samples using Flash Attention 2
        self.packing_samples = self.args.data.packing_samples
        
        self._print_model()

    def forward(
        self,
        sequences: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        allgather_logits=False,
        ring_attn_group: Optional[dist.ProcessGroup] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Returns action log probs"""
        batch, seqlen = sequences.size()
        foward_attention_mask = attention_mask
        if self.packing_samples:
            sequences, position_ids, rolled_sequences, ring_attn_pad_len, indices = unpad_and_slice_tensor(
                sequences, attention_mask, ring_attn_group
            )
            foward_attention_mask = None
        else:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        output = self.model(sequences, attention_mask=foward_attention_mask, position_ids=position_ids, output_hidden_states=True, **kwargs)
        output["logits"] = output["logits"].to(torch.float32)
        
        if allgather_logits and self.packing_samples:
            output["logits"] = gather_and_pad_tensor(
                output["logits"], ring_attn_group, ring_attn_pad_len, indices, batch, seqlen
            ).squeeze(-2)
            output["hidden_states"] = list(output["hidden_states"])
            for i in range(len(output["hidden_states"])):
                 output["hidden_states"][i] = gather_and_pad_tensor(
                    output["hidden_states"][i], ring_attn_group, ring_attn_pad_len, indices, batch, seqlen
                ).squeeze(-2)
        return output

    def _print_model(self):
        self.strategy.print(f"Student Model: \n  {self.model}")
    
    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": self.args.train.gradient_checkpointing_use_reentrant
            }
        )

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()

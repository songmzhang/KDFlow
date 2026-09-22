from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
)

from kdflow.datasets.utils import get_tokenizer_or_processor
from kdflow.models.packing_utils import (
    gather_and_pad_tensor,
    prepare_packed_inputs,
    register_vision_packing_hook,
)


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
        model_type = self.model_config.model_type or ""
        self.is_linear_attention = any(k in model_type.lower() for k in ("qwen3_next", "qwen3_5"))
        
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

        processor_or_tokenizer = get_tokenizer_or_processor(
            model_name_or_path,
            self.model,
            padding_side="right",
            need_processor=self.is_vl_model,
        )
        if self.is_vl_model:
            self.processor = processor_or_tokenizer
            self.tokenizer = self.processor.tokenizer
        else:
            self.processor = None
            self.tokenizer = processor_or_tokenizer

        # https://github.com/huggingface/transformers/issues/26877
        # Use `model.generate(use_cache=True)` instead.`
        self.model.config.use_cache = False

        # packing samples using Flash Attention 2
        self.packing_samples = self.args.data.packing_samples
        if self.packing_samples and self.is_vl_model:
            register_vision_packing_hook(self.model)
        
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
        if self.packing_samples:
            model_inputs, packing_kwargs, restore_kwargs = prepare_packed_inputs(
                self.model, sequences, attention_mask, ring_attn_group, **kwargs,
            )
            if self.is_linear_attention or self.is_vl_model:
                model_inputs.update(packing_kwargs)
        else:
            model_inputs = dict(kwargs, input_ids=sequences, attention_mask=attention_mask, position_ids=None)

        output = self.model(**model_inputs)
        # lm_head is patched to identity (skip=True), so output["logits"]
        # are actually final hidden states.
        output = {"hidden_states": [output["logits"]]}
            
        if allgather_logits and self.packing_samples:
            output["hidden_states"][-1] = gather_and_pad_tensor(
                output["hidden_states"][-1], **restore_kwargs,
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

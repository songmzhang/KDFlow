import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingArguments:
    """ Arguments for training."""
    
    num_nodes: int = field(
        default=1,
        metadata={"help": "The number of nodes for the student/teacher model (only for on-policy KD)."}
    )
    num_gpus_per_node: int = field(
        default=8,
        metadata={"help": "The number of gpus per node for the student/teacher model (only for on-policy KD)."}
    )
    num_epochs: int = field(
        default=1,
        metadata={"help": "Number of training epochs."}
    )
    train_batch_size: int = field(
        default=128,
        metadata={"help": "Global training batch size."}
    )
    micro_train_batch_size: int = field(
        default=1,
        metadata={"help": "Micro training batch size per GPU."}
    )
    learning_rate: float = field(
        default=1e-6
    )
    lr_warmup_ratio: float = field(
        default=0.05
    )
    min_lr: float = field(
        default=1e-8
    )
    lr_scheduler: str = field(
        default="cosine_with_min_lr"
    )
    max_norm: float = field(
        default=1.0,
        metadata={"help": "Gradient clipping."}
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay."}
    )
    adam_betas: str = field(
        default="(0.9, 0.98)"
    )
    eval_steps: int = field(
        default=-1,
        metadata={"help": "Evaluation every n step."}
    )
    save_steps: int = field(
        default=-1,
        metadata={"help": "Save checkpoints every n step."}
    )
    backend: str = field(
        default="fsdp2",
        metadata={
            "choices": ["fsdp2"],
            "help": "Student training backend."
        }
    )
    gradient_checkpointing: bool = field(
        default=False
    )
    gradient_checkpointing_use_reentrant: bool = field(
        default=False
    )
    teacher_offload: bool = field(
        default=False,
        metadata={"help": "Offload teacher model to GPU."}
    )
    train_enable_sleep: bool = field(
        default=False,
        metadata={"help": "[Deprecated] Use --enable_sleep instead."}
    )
    enable_sleep: bool = field(
        default=False,
        metadata={"help": "Enable sleep mode for all components (student, teacher, rollout)."}
    )
    full_determinism: bool = field(
        default=False,
        metadata={"help": "Enable reproducible behavior during distributed training."}
    )
    load_checkpoint: bool = field(
        default=False
    )
    ckpt_path: str = field(
        default="./ckpt/checkpoints_distill"
    )
    save_path: str = field(
        default="./ckpt/"
    )
    seed: int = field(
        default=42,
    )
    bf16: bool = field(
        default=False
    )
    local_rank: int = field(
        default=int(os.getenv("LOCAL_RANK", -1)),
        metadata={"help": "Local rank for distributed training. Automatically set."}
    )
    
    def __post_init__(self):
        backend_choices = self.__class__.__dataclass_fields__["backend"].metadata["choices"]
        if self.backend not in backend_choices:
            raise ValueError(
                f"Unsupported training backend '{self.backend}'"
                f"Supported choices: {backend_choices}"
            )

        if isinstance(self.adam_betas, str):
            try:
                cleaned_str = self.adam_betas.strip().strip('()')
                parts = [p.strip() for p in cleaned_str.split(',')]
                if len(parts) != 2:
                    raise ValueError("adam_betas must contain two values.")
                
                # 直接修改 self.adam_betas，将其从 str 替换为 tuple
                self.adam_betas = (float(parts[0]), float(parts[1]))
            except Exception as e:
                raise ValueError(
                    f"Cannot parse '{self.adam_betas}' to valid adam_betas. Make sure the format to be '(beta1, beta2)'。"
                ) from e
        else:
             raise TypeError(f"Expected str for adam_betas, but get {type(self.adam_betas)}")
         
        if self.save_steps == -1:
            self.save_steps = float("inf")
        
        if self.eval_steps == -1:
            self.eval_steps = float("inf")
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataArguments:
    """ Arguments for dataset."""
    
    train_dataset_path: str = field(
        default=None,
        metadata={"help": "Training dataset name or path."}
    )
    train_dataset_probs: str = field(
        default=None,
        metadata={"help": "Sampling probs for multiple datasets."}
    )
    train_split: str = field(
        default="train",
        metadata={"help": "Train split in dataset."}
    )
    eval_dataset_path: str = field(
        default=None,
        metadata={"help": "Evaluation dataset name or path."}
    )
    eval_split: str = field(
        default="eval",
        metadata={"help": "Eval split in dataset."}
    )
    input_key: str = field(
        default="messages",
        metadata={"help": "JSON dataset key."}
    )
    image_key: str = field(
        default=None,
        metadata={"help": "Image key in dataset."}
    )
    output_key: str = field(
        default=None,
        metadata={"help": "Output key in dataset."}
    )
    teacher_input_key: str = field(
        default=None,
        metadata={"help": "Input key for teacher prompt. If None, use the same input_key as student. "
                          "Used in self-distillation where teacher receives richer information."}
    )
    label_key: str = field(
        default=None,
        metadata={"help": "Label key in dataset."}
    )
    input_template: str = field(
        default=None,
    )
    apply_chat_template: bool = field(
        default=True,
        metadata={"help": "Use HF tokenizer chat template."}
    )
    max_len: int = field(
        default=4096
    )
    max_samples: int = field(
        default=1e8
    )
    packing_samples: bool = field(
        default=False,
        metadata={"help": "Packing sequences during training."}
    )
    prompt_max_len: int = field(
        default=2048,
        metadata={"help": "Max prompt length."}
    )
    preprocess_num_workers: int = field(
        default=8,
    )
    
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RolloutArguments:
    """ Arguments for rollout (on-policy distillation)."""
    
    rollout_num_engines: int = field(
        default=0,
        metadata={"help": "The number of engines for rollout."}
    )
    rollout_tp_size: int = field(
        default=1,
        metadata={"help": "Tensor parallel size for each vLLM engine."}
    )
    rollout_enable_sleep: bool = field(
        default=False,
        metadata={"help": "Enable sleep mode for vLLM."}
    )
    rollout_mem_fraction_static: float = field(
        default=0.6,
        metadata={"help": "GPU memory utilization for each vLLM engine."}
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p sampling for rollout."}
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for rollout."}
    )
    n_samples_per_prompt: int = field(
        default=1,
        metadata={"help": "Sample n responses per prompt."}
    )
    rollout_batch_size: int = field(
        default=32,
        metadata={"help": "Number of prompts for each rollout."}
    )
    generate_max_len: int = field(
        default=2048,
        metadata={"help": "Max generation tokens during rollout."}
    )
    print_rollout_sample: bool = field(
        default=False,
        metadata={"help": "Whether to print a rollout sample after each rollout."}
    )
    
    
import logging
from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger(__name__)


@dataclass
class DistillationArguments:
    """ Arguments for knowledge distillation."""
    
    kd_ratio: float = field(
        default=0.5,
        metadata={"help": "Loss = (1 - kd_ratio) * nll_loss + kd_ratio * kd_loss."}
    )
    kd_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for knowledge distillation."}
    )
    kd_algorithm: str = field(
        default="vanilla_kd",
        metadata={"help": "KD algorithm for each training step."}
    )
    kd_loss_fn: str = field(
        default="kl",
        metadata={"help": "Divergence selection for knowledge distillation, e.g., kl, rkl, js."}
    )
    use_triton_loss: bool = field(
        default=False,
        metadata={"help": "Enable triton kernel for KL (or other) divergence."}
    )
    teacher_forward_n_batches: int = field(
        default=1,
        metadata={"help": "Teacher forward N global batches at once for student multi-step training."}
    )
    teacher_enable_sleep: bool = field(
        default=False,
        metadata={"help": "Sleep teacher when not needed."}
    )
    teacher_offload_tags: str = field(
        default="all",
        metadata={"help": "Offload tags for sglang."}
    )
    teacher_quantization: str = field(
        default=None
    )
    teacher_tp_size: int = field(
        default=8,
        metadata={"help": "Tensor parallel size for teacher model."}
    )
    teacher_ep_size: int = field(
        default=1,
        metadata={"help": "Expert parallel size for teacher model (only for MoE models)."}
    )
    teacher_pp_size: int = field(
        default=1,
        metadata={"help": "Pipeline parallel size for teacher model."}
    )
    teacher_dp_size: int = field(
        default=1,
        metadata={"help": "Data parallel size for teacher model."}
    )
    teacher_mem_fraction_static: float = field(
        default=0.4,
        metadata={"help": "Memory fraction for teacher model."}
    )
    # DSKD hyperparameters
    dskd_token_align: str = field(
        default="eta",
        metadata={
            "help": "Token alignment strategy for cross-tokenizer DSKD. Options: 'cma' (cross-model attention), 'eta' (exact token alignment).", 
            "choices": ["eta", "cma"]
        }
    )
    dskd_topk_vocab: int = field(
        default=-1,
        metadata={"help": "Number of top vocabulary tokens used for projector initialization. -1 means using all tokens."}
    )
    dskd_projector_lr: float = field(
        default=1e-4,
        metadata={"help": "Learning rate for DSKD projectors."}
    )
    # JSD
    jsd_beta: float = field(
        default=0.5,
        metadata={"help": "Beta for Jensen-Shannon Divergence."}
    )
    # Skewed KL/RKL
    skew_lambda: float = field(
        default=0.1,
        metadata={"help": "Lambda for Skewed KL/RKL."}
    )
    # Adaptive KL
    adaptive_alpha: float = field(
        default=0.5,
        metadata={"help": "Alpha for Adaptive KL Divergence."}
    )
    # Hierarchical Ranking Loss
    hrl_topk: int = field(
        default=5,
        metadata={"help": "Top-k Ranking for Hierarchical Ranking Loss."}
    )

    def __post_init__(self):
        # Validate teacher parallel size settings
        if self.teacher_ep_size > self.teacher_tp_size:
            raise ValueError(
                f"SGLang requires that teacher_ep_size ({self.teacher_ep_size}) must be <= teacher_tp_size ({self.teacher_tp_size}). "
            )
        if self.teacher_tp_size % self.teacher_ep_size != 0:
            raise ValueError(
                f"SGLang requires that teacher_tp_size ({self.teacher_tp_size}) must be divisible by teacher_ep_size ({self.teacher_ep_size})."
            )
        # Validate KD hyperparameters
        if not 0.0 <= self.kd_ratio <= 1.0:
            raise ValueError(f"kd_ratio must be in [0, 1], got {self.kd_ratio}.")
        if self.kd_temperature <= 0:
            raise ValueError(f"kd_temperature must be > 0, got {self.kd_temperature}.")
        if not 0.0 < self.teacher_mem_fraction_static <= 1.0:
            raise ValueError(f"teacher_mem_fraction_static must be in (0, 1], got {self.teacher_mem_fraction_static}.")

    def validate_teacher_parallelism(self, num_nodes: int, num_gpus_per_node: int):
        total_gpus = num_nodes * num_gpus_per_node
        total_parallel = self.teacher_tp_size * self.teacher_pp_size

        # tp * ep * pp must evenly divide total GPUs
        if total_gpus % total_parallel != 0:
            raise ValueError(
                f"teacher_tp_size * teacher_ep_size * teacher_pp_size ({self.teacher_tp_size} * {self.teacher_ep_size} * {self.teacher_pp_size} = {total_parallel}) "
                f"must evenly divide num_nodes * num_gpus_per_node ({num_nodes} * {num_gpus_per_node} = {total_gpus})."
            )

        # Auto-adjust dp_size if tp * pp * dp != total_gpus
        expected_dp = total_gpus // total_parallel
        if self.teacher_dp_size != expected_dp:
            logger.warning(
                f"Auto-adjusting teacher_dp_size from {self.teacher_dp_size} to {expected_dp} "
                f"to match total GPUs ({total_gpus} = {num_nodes} nodes * {num_gpus_per_node} gpus/node). "
                f"(tp={self.teacher_tp_size} (ep={self.teacher_ep_size}) * pp={self.teacher_pp_size} * dp={expected_dp} = {total_gpus})"
            )
            self.teacher_dp_size = expected_dp

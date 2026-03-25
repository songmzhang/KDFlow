from dataclasses import dataclass, field

from transformers import HfArgumentParser

from kdflow.arguments.data_args import DataArguments
from kdflow.arguments.model_args import ModelArguments
from kdflow.arguments.training_args import TrainingArguments
from kdflow.arguments.fsdp_args import FSDPArguments
from kdflow.arguments.distillation_args import DistillationArguments
from kdflow.arguments.rollout_args import RolloutArguments
from kdflow.arguments.logging_args import LoggingArguments
from kdflow.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass
class AllArguments:
    data: DataArguments = field(default_factory=DataArguments)
    model: ModelArguments = field(default_factory=ModelArguments)
    train: TrainingArguments = field(default_factory=TrainingArguments)
    fsdp: FSDPArguments = field(default_factory=FSDPArguments)
    kd: DistillationArguments = field(default_factory=DistillationArguments)
    rollout: RolloutArguments = field(default_factory=RolloutArguments)
    log: LoggingArguments = field(default_factory=LoggingArguments)
    

def init_args():
    parser = HfArgumentParser((
        DataArguments,
        ModelArguments,
        TrainingArguments,
        FSDPArguments,
        DistillationArguments,
        RolloutArguments,
        LoggingArguments
    ))
    (
        data_args, 
        model_args, 
        train_args, 
        fsdp_args,
        kd_args, 
        rollout_args, 
        log_args
    ) = parser.parse_args_into_dataclasses()

    args = AllArguments(
        data=data_args,
        model=model_args,
        train=train_args,
        fsdp=fsdp_args,
        kd=kd_args,
        rollout=rollout_args,
        log=log_args
    )
    
    # Validate arguments
    if args.data.input_template and "{}" not in args.data.input_template:
        logger.warning("{} not in args.data.input_template, set to None")
        args.data.input_template = None

    if args.data.input_template and "\\n" in args.data.input_template:
        logger.warning(
            "input_template contains \\n characters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.data.packing_samples:
        if "flash_attention" not in args.model.attn_implementation:
            logger.warning(
                "Please use --attn_implementation with flash_attention to accelerate when --packing_samples is enabled."
            )
            args.model.attn_implementation = "flash_attention_2"
            
        if args.data.image_key is not None:
            logger.warning(
                "--packing_samples is not supported with image data. Disabling packing_samples."
            )
            args.data.packing_samples = False
            
    total_gpus = args.train.num_nodes * args.train.num_gpus_per_node
    
    if args.rollout.rollout_num_engines > 0:
        if total_gpus % args.rollout.rollout_tp_size != 0:
            raise ValueError(
                f"Total GPUs ({total_gpus}) must be divisible by rollout_tp_size ({args.rollout.rollout_tp_size})."
            )
            
        expected_num_engines = total_gpus // args.rollout.rollout_tp_size
        if args.rollout.rollout_num_engines != expected_num_engines:
            logger.warning(
                f"Auto-adjusting rollout_num_engines from {args.rollout.rollout_num_engines} to {expected_num_engines} "
                f"to match total GPUs ({total_gpus}). "
                f"(rollout_tp_size={args.rollout.rollout_tp_size} * rollout_num_engines={expected_num_engines} = {total_gpus})"
            )
            args.rollout.rollout_num_engines = expected_num_engines
            
        if args.data.max_len < args.data.prompt_max_len + args.rollout.generate_max_len:
            args.data.max_len = args.data.prompt_max_len + args.rollout.generate_max_len
            logger.warning(
                "--max_len is smaller than --prompt_max_len + --generate_max_len. "
                f"Automatically increase --max_len to {args.data.max_len}."
            )
    
    if args.model.teacher_name_or_path is not None:
        teacher_parallel = args.kd.teacher_tp_size * args.kd.teacher_pp_size
        if total_gpus % teacher_parallel != 0:
            raise ValueError(
                f"Total GPUs ({total_gpus}) must be divisible by "
                f"teacher_tp_size * teacher_pp_size ({args.kd.teacher_tp_size} * {args.kd.teacher_pp_size} = {teacher_parallel})."
            )
            
        if args.kd.teacher_ep_size > args.kd.teacher_tp_size:
            logger.warning(
                f"teacher_ep_size ({args.kd.teacher_ep_size}) must not be greater than teacher_tp_size ({args.kd.teacher_tp_size}). "
                f"Auto-adjusting teacher_ep_size to {args.kd.teacher_tp_size}."
            )
            args.kd.teacher_ep_size = args.kd.teacher_tp_size
            
        expected_dp = total_gpus // teacher_parallel
        if args.kd.teacher_dp_size != expected_dp:
            logger.warning(
                f"Auto-adjusting teacher_dp_size from {args.kd.teacher_dp_size} to {expected_dp} "
                f"to match total GPUs ({total_gpus}). "
                f"(tp={args.kd.teacher_tp_size} (ep={args.kd.teacher_ep_size}) * pp={args.kd.teacher_pp_size} * dp={expected_dp} = {total_gpus})"
            )
            args.kd.teacher_dp_size = expected_dp
    
    deprecated_sleep_flags = [
        args.train.train_enable_sleep,
        args.kd.teacher_enable_sleep,
        args.rollout.rollout_enable_sleep,
    ]
    if any(deprecated_sleep_flags):
        logger.warning(
            "--train_enable_sleep, --teacher_enable_sleep, --rollout_enable_sleep are deprecated "
            "and will be removed in a future version. Use --enable_sleep instead."
        )
        args.train.enable_sleep = True
    
    return args
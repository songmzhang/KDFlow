from dataclasses import dataclass, field

from kdflow.arguments.data_args import DataArguments
from kdflow.arguments.model_args import ModelArguments
from kdflow.arguments.training_args import TrainingArguments
from kdflow.arguments.fsdp_args import FSDPArguments
from kdflow.arguments.distillation_args import DistillationArguments
from kdflow.arguments.rollout_args import RolloutArguments
from kdflow.arguments.logging_args import LoggingArguments

from transformers import HfArgumentParser


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
        print("[Warning] {} not in args.data.input_template, set to None")
        args.data.input_template = None

    if args.data.input_template and "\\n" in args.data.input_template:
        print(
            "[Warning] input_template contains \\n characters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.data.packing_samples:
        if "flash_attention" not in args.model.attn_implementation:
            print(
                "[Warning] Please use --attn_implementation with flash_attention to accelerate when --packing_samples is enabled."
            )
            args.model.attn_implementation = "flash_attention_2"
            
        if args.data.image_key is not None:
            print(
                "[Warning] --packing_samples is not supported with image data. Disabling packing_samples."
            )
            args.data.packing_samples = False
            
    if args.rollout.rollout_num_engines > 0:
        if args.rollout.rollout_num_engines * args.rollout.rollout_tp_size < args.train.num_nodes * args.train.num_gpus_per_node:
            args.rollout.rollout_num_engines = args.train.num_nodes * args.train.num_gpus_per_node // args.rollout.rollout_tp_size
            print(
                "[Warning] rollout_num_engines * rollout_tp_size is less than total GPUs. "
                f"Automatically increase rollout_num_engines to {args.rollout.rollout_num_engines}."
            )
            
        if args.data.max_len < args.data.prompt_max_len + args.rollout.generate_max_len:
            args.data.max_len = args.data.prompt_max_len + args.rollout.generate_max_len
            print(
                "[Warning] --max_len is smaller than --prompt_max_len + --generate_max_len.",
                f"Automatically increase --max_len to {args.data.max_len}.",
            )
    
    # Validate teacher parallelism settings against available GPUs
    args.kd.validate_teacher_parallelism(args.train.num_nodes, args.train.num_gpus_per_node)
    
    return args
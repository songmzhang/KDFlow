import json
from dataclasses import dataclass, field

import transformers
from packaging.version import Version
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
    

def is_linear_attention(model_path: str) -> bool:
    if not model_path:
        return False
    try:
        from transformers import AutoConfig
        model_type = AutoConfig.from_pretrained(model_path, trust_remote_code=True).model_type or ""
    except Exception:
        return False
    return any(k in model_type.lower() for k in ("qwen3_next", "qwen3_5"))


def init_args(scenario: str = "sft"):
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
    if scenario != "on_policy_kd" and args.data.custom_eval_fn is not None:
        logger.warning(
            f"`--custom_eval_fn` is only supported in the `on_policy_kd` scenario "
            f"and will be ignored for `{scenario}`."
        )

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

        if is_linear_attention(args.model.student_name_or_path):
            if Version(transformers.__version__) < Version("5.9.0"):
                logger.warning(
                    "--packing_samples for linear-attention models requires transformers>=5.9.0 "
                    f"(got {transformers.__version__}). Auto-disabling --packing_samples."
                )
                args.data.packing_samples = False
            elif args.model.ring_attn_size > 1:
                logger.warning(
                    "--packing_samples for linear-attention models is only supported with "
                    "--ring_attn_size 1. Auto-disabling --packing_samples."
                )
                args.data.packing_samples = False
            elif args.data.image_key is not None:
                logger.warning(
                    "--packing_samples for linear-attention models currently only supports text-only data. "
                    "Auto-disabling --packing_samples."
                )
                args.data.packing_samples = False

    total_gpus = args.train.num_nodes * args.train.num_gpus_per_node

    if args.kd.multi_teacher_config is not None:
        try:
            with open(args.kd.multi_teacher_config, "r", encoding="utf-8") as f:
                args.kd.multi_teacher_config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"`--multi_teacher_config` must be a valid JSON file: {args.kd.multi_teacher_config}") from e
        if not isinstance(args.kd.multi_teacher_config, dict) or not args.kd.multi_teacher_config:
            raise ValueError("`--multi_teacher_config` must be a non-empty JSON object mapping teacher keys to model paths.")
        if args.model.teacher_name_or_path is not None:
            logger.warning(
                "Detect `--multi_teacher_config` is set, `--teacher_name_or_path` will be set to None automatically."
            )
            args.model.teacher_name_or_path = None
        if args.kd.kd_algorithm != "vanilla_kd":
            raise ValueError(
                "Multi-teacher distillation currently only supports `vanilla_kd`, "
                f"got `{args.kd.kd_algorithm}`."
            )
    
    if scenario == "on_policy_kd":
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
    
    has_teacher = args.model.teacher_name_or_path is not None or args.kd.multi_teacher_config is not None
    is_kd_scenario = scenario in ("off_policy_kd", "on_policy_kd")
    if is_kd_scenario:
        if not has_teacher:
            raise ValueError(
                "KD scenario requires either `--teacher_name_or_path` or `--multi_teacher_config` to be set."
            )

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

    args.scenario = scenario
    return args

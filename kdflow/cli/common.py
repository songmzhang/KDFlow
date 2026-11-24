import os

import ray

from kdflow.datasets.utils import blending_datasets
from kdflow.models.utils import check_tokenizer_identical, TokenizerCompareResult
from kdflow.ray.train.student_group import StudentActorGroup
from kdflow.utils.distributed_sampler import DistributedSampler
from kdflow.utils.utils import get_tokenizer


def init_ray():
    """Initialize Ray with the project root as working_dir if not already initialized."""
    if ray.is_initialized():
        return
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN"
            },
            "working_dir": os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        }
    )


def build_student_group(args, pg_info):
    pg, reordered_bundle_indices, _ = pg_info
    return StudentActorGroup(
        args,
        args.train.num_nodes,
        args.train.num_gpus_per_node,
        pg=(pg, reordered_bundle_indices),
        num_gpus_per_actor=0.01,
    )


def build_teacher_group(args, strategy, pg_info):
    # Lazy import to keep SFT free of the sglang dependency
    from kdflow.ray.train.multi_teacher_group import MultiTeacherActorGroup
    from kdflow.ray.train.teacher_group import TeacherActorGroup

    TeacherGroupCLS = MultiTeacherActorGroup if args.kd.multi_teacher_config else TeacherActorGroup
    return TeacherGroupCLS(
        strategy,
        args.train.num_nodes * args.train.num_gpus_per_node,
        num_gpus_per_node=args.train.num_gpus_per_node,
        num_gpus_per_actor=0.01,
        pg=pg_info,
    )


def build_tokenizer_info(args, strategy):
    """Compare student and teacher tokenizers.

    Returns:
        (teacher_tokenizer, tokenizer_info). For multi-teacher KD, the student tokenizer is
        returned as teacher_tokenizer since all teachers must share the student's vocabulary.
    """
    use_fast = not args.model.disable_fast_tokenizer
    student_tokenizer = get_tokenizer(args.model.student_name_or_path, use_fast=use_fast)

    if args.kd.multi_teacher_config:
        tokenizer_info = TokenizerCompareResult()
        for teacher_key, teacher_name_or_path in args.kd.multi_teacher_config.items():
            teacher_tokenizer = get_tokenizer(teacher_name_or_path, use_fast=use_fast)
            cur_tokenizer_info = check_tokenizer_identical(student_tokenizer, teacher_tokenizer)
            if not cur_tokenizer_info.vocab_identical:
                raise ValueError(
                    f"Teacher model '{teacher_key}' has a different vocabulary from the student. "
                    f"Multi-teacher KD requires all teachers to share the same vocabulary as the student."
                )
            tokenizer_info.template_identical = tokenizer_info.template_identical and cur_tokenizer_info.template_identical
        strategy.print(f"Tokenizers {tokenizer_info}")
        return student_tokenizer, tokenizer_info

    teacher_tokenizer = get_tokenizer(args.model.teacher_name_or_path, use_fast=use_fast)
    tokenizer_info = check_tokenizer_identical(student_tokenizer, teacher_tokenizer)
    strategy.print(f"Tokenizers {tokenizer_info}")
    if not tokenizer_info.vocab_identical and args.kd.kd_algorithm not in ["dskd", "simple_ctkd"]:
        raise ValueError("Student and teacher tokenizers are not identical. Please use DSKD or SimpleCrossToknizerKD algorithm for cross-tokenizer KD or ensure tokenizers are the same.")
    return teacher_tokenizer, tokenizer_info


def build_dataloaders(args, strategy, dataset_cls, batch_size, eval_batch_size, pin_memory, tokenizer_info=None):
    """Build the training dataloader and the optional evaluation dataloader."""
    train_data = blending_datasets(
        args.data.train_dataset_path,
        args.data.train_dataset_probs,
        strategy,
        args.train.seed,
        max_count=args.data.max_samples,
        dataset_split=args.data.train_split,
    )
    train_data = train_data.select(range(min(args.data.max_samples, len(train_data))))
    train_dataset = dataset_cls(
        train_data,
        strategy,
        tokenizer_info=tokenizer_info,
        max_data_num=args.data.max_samples,
        input_template=args.data.input_template,
        num_processors=args.data.preprocess_num_workers,
    )
    sampler = DistributedSampler(train_dataset, num_replicas=1, rank=0, shuffle=True, seed=args.train.seed, drop_last=True)
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        batch_size,
        pin_memory=pin_memory,
        shuffle=False,
        collate_fn=train_dataset.collate_fn,
        sampler=sampler,
    )

    eval_dataloader = None
    if args.data.eval_dataset_path:
        eval_data = blending_datasets(
            args.data.eval_dataset_path,
            None,
            strategy,
            dataset_split=args.data.eval_split,
        )
        eval_data = eval_data.select(range(min(args.data.max_samples, len(eval_data))))
        eval_dataset = dataset_cls(
            eval_data,
            strategy,
            tokenizer_info=tokenizer_info,
            input_template=args.data.input_template,
            num_processors=args.data.preprocess_num_workers,
        )
        eval_dataloader = strategy.setup_dataloader(
            eval_dataset,
            batch_size=eval_batch_size,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            collate_fn=eval_dataset.collate_fn,
        )

    return train_dataloader, eval_dataloader

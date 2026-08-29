import os
import math

import ray

from kdflow.ray.train.multi_teacher_group import MultiTeacherActorGroup
from kdflow.ray.train.teacher_group import TeacherActorGroup
from kdflow.ray.train.student_group import StudentActorGroup
from kdflow.ray.rollout.rollout_group import RolloutActorGroup
from kdflow.ray.placement_group import create_placement_group
from kdflow.trainer import OnPolicyKDTrainer
from kdflow.datasets import PromptDataset
from kdflow.datasets.utils import blending_datasets
from kdflow.models.utils import check_tokenizer_identical, TokenizerCompareResult
from kdflow.backend import get_strategy
from kdflow.arguments import init_args
from kdflow.utils.distributed_sampler import DistributedSampler
from kdflow.utils.utils import get_tokenizer, load_custom_eval_fn


def train(args):
    use_lora = args.model.lora_rank > 0
    if use_lora and args.model.student_name_or_path == args.model.teacher_name_or_path:
        raise ValueError("On-policy LoRA does not support self-distillation yet.")

    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN"
                },
                "working_dir": os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            }
        )
    
    strategy = get_strategy(args)
    strategy.print(args)
    
    # Create placement group for resource allocation
    num_gpus = args.train.num_nodes * args.train.num_gpus_per_node
    pg, reordered_bundle_indices, reordered_gpu_ids = create_placement_group(num_gpus)

    rollout_server_args = None
    if use_lora:
        target_modules = args.model.target_modules
        if isinstance(target_modules, str):
            target_modules = ["all"] if target_modules == "all-linear" else [target_modules]
        rollout_server_args = {
            "enable_lora": True,
            "max_lora_rank": args.model.lora_rank,
            "lora_target_modules": target_modules,
            "max_loaded_loras": 2,
            "max_loras_per_batch": 1,
        }

    rollout_group = RolloutActorGroup(
        model_path=args.model.student_name_or_path,
        num_actors=args.rollout.rollout_num_engines,
        tp_size=args.rollout.rollout_tp_size,
        num_gpus_per_node=args.train.num_gpus_per_node,
        enable_memory_saver=True,
        mem_fraction_static=args.rollout.rollout_mem_fraction_static,
        num_gpus_per_actor=0.01,
        pg=(pg, reordered_bundle_indices, reordered_gpu_ids),
        extra_server_args=rollout_server_args,
    )
    if args.train.enable_sleep:
        rollout_group.sleep()
    
    TeacherGroupCLS = MultiTeacherActorGroup if args.kd.multi_teacher_config else TeacherActorGroup
    teacher_model = TeacherGroupCLS(
        strategy,
        num_gpus,
        num_gpus_per_node=args.train.num_gpus_per_node,
        num_gpus_per_actor=0.01,
        pg=(pg, reordered_bundle_indices, reordered_gpu_ids),
    )
    student_model = StudentActorGroup(
        args,
        args.train.num_nodes,
        args.train.num_gpus_per_node,
        pg=(pg, reordered_bundle_indices),
        num_gpus_per_actor=0.01,
    )
    
    # Initialize tokenizers
    student_tokenizer = get_tokenizer(
        args.model.student_name_or_path,
        use_fast=not args.model.disable_fast_tokenizer
    )
    if args.kd.multi_teacher_config:
        tokenizer_info = TokenizerCompareResult()
        for teacher_key, teacher_name_or_path in args.kd.multi_teacher_config.items():
            teacher_tokenizer = get_tokenizer(teacher_name_or_path, use_fast=not args.model.disable_fast_tokenizer)
            cur_tokenizer_info = check_tokenizer_identical(student_tokenizer, teacher_tokenizer)
            if not cur_tokenizer_info.vocab_identical:
                raise ValueError(
                    f"Teacher model '{teacher_key}' has a different vocabulary from the student. "
                    f"Multi-teacher KD requires all teachers to share the same vocabulary as the student."
                )
            tokenizer_info.template_identical = tokenizer_info.template_identical and cur_tokenizer_info.template_identical
        strategy.print(f"Tokenizers {tokenizer_info}")
    else:
        teacher_tokenizer = get_tokenizer(
            args.model.teacher_name_or_path,
            use_fast=not args.model.disable_fast_tokenizer
        )
        tokenizer_info = check_tokenizer_identical(student_tokenizer, teacher_tokenizer)
        strategy.print(f"Tokenizers {tokenizer_info}")
        if not tokenizer_info.vocab_identical and args.kd.kd_algorithm not in ["dskd", "simple_ctkd"]:
            raise ValueError("Student and teacher tokenizers are not identical. Please use DSKD or SimpleCrossToknizerKD algorithm for cross-tokenizer KD or ensure tokenizers are the same.")
    
    # Load and prepare training dataset
    train_data = blending_datasets(
        args.data.train_dataset_path,
        args.data.train_dataset_probs,
        strategy,
        args.train.seed,
        max_count=args.data.max_samples,
        dataset_split=args.data.train_split,
    )
    train_data = train_data.select(range(min(args.data.max_samples, len(train_data))))
    
    train_dataset = PromptDataset(
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
        args.rollout.rollout_batch_size,
        True,
        False,
        collate_fn=train_dataset.collate_fn,
        sampler=sampler,
    )
    
    # Load and prepare evaluation dataset (optional)
    eval_dataloader = None
    if getattr(args.data, "eval_dataset_path", None):
        eval_data = blending_datasets(
            args.data.eval_dataset_path,
            None,
            strategy,
            dataset_split=args.data.eval_split,
        )
        eval_data = eval_data.select(range(min(args.data.max_samples, len(eval_data))))
        eval_dataset = PromptDataset(
            eval_data,
            strategy,
            tokenizer_info=tokenizer_info,
            input_template=args.data.input_template,
            num_processors=args.data.preprocess_num_workers,
        )
        eval_dataloader = strategy.setup_dataloader(
            eval_dataset,
            batch_size=args.rollout.rollout_batch_size,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            collate_fn=eval_dataset.collate_fn,
        )
    
    # Calculate max training steps
    num_rollout_iters_per_epoch = len(train_dataset) // args.rollout.rollout_batch_size
    num_update_steps_per_rollout = args.rollout.rollout_batch_size * args.rollout.n_samples_per_prompt // args.train.train_batch_size
    max_rollout_iters = math.ceil(args.train.num_epochs * num_rollout_iters_per_epoch)
    strategy.log(f"Max training iterations: {max_rollout_iters}")
    
    # Initialize student model on all workers
    ray.get(student_model.async_init_model_from_pretrained(
        strategy, (max_rollout_iters * num_update_steps_per_rollout), 
        teacher_tokenizer=teacher_tokenizer if not args.kd.multi_teacher_config else student_tokenizer, 
        tokenizer_info=tokenizer_info,
    ))
    strategy.log("Models initialized on all student actors")
    
    generate_kwargs = {
        "max_new_tokens": args.rollout.generate_max_len,
        "temperature": args.rollout.temperature,
        "top_p": args.rollout.top_p,
    }
    
    trainer = OnPolicyKDTrainer(
        strategy=strategy,
        student_model=student_model,
        teacher_model=teacher_model,
        rollout_group=rollout_group,
        is_same_tokenizer=tokenizer_info.vocab_identical,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        max_rollout_iters=max_rollout_iters,
        num_rollout_iters_per_epoch=num_rollout_iters_per_epoch,
        generate_kwargs=generate_kwargs,
        custom_eval_fn=load_custom_eval_fn(args.data.custom_eval_fn) if args.data.custom_eval_fn else None,
    )
    
    try:
        trainer.fit()
        ray.get(student_model.async_save_model())
        strategy.log("Training completed and model saved.")
    finally:
        teacher_model.shutdown()
        rollout_group.shutdown()


if __name__ == "__main__":
    args = init_args("on_policy_kd")
    train(args)

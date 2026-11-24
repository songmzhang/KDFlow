import math

import ray

from kdflow.cli.common import (
    init_ray,
    build_student_group,
    build_teacher_group,
    build_tokenizer_info,
    build_dataloaders,
)
from kdflow.ray.rollout.rollout_group import RolloutActorGroup
from kdflow.ray.placement_group import create_placement_group
from kdflow.trainer import OnPolicyKDTrainer
from kdflow.datasets import PromptDataset
from kdflow.backend import get_strategy
from kdflow.arguments import init_args
from kdflow.utils.utils import load_custom_eval_fn


def train(args):
    use_lora = args.model.lora_rank > 0
    if use_lora and args.model.student_name_or_path == args.model.teacher_name_or_path:
        raise ValueError("On-policy LoRA does not support self-distillation yet.")

    init_ray()
    
    strategy = get_strategy(args)
    strategy.print(args)
    
    # Create placement group for resource allocation
    num_gpus = args.train.num_nodes * args.train.num_gpus_per_node
    pg_info = create_placement_group(num_gpus)

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
        pg=pg_info,
        extra_server_args=rollout_server_args,
    )
    if args.train.enable_sleep:
        rollout_group.sleep()
    
    teacher_model = build_teacher_group(args, strategy, pg_info)
    student_model = build_student_group(args, pg_info)
    
    teacher_tokenizer, tokenizer_info = build_tokenizer_info(args, strategy)
    
    train_dataloader, eval_dataloader = build_dataloaders(
        args, strategy, PromptDataset,
        batch_size=args.rollout.rollout_batch_size,
        eval_batch_size=args.rollout.rollout_batch_size,
        pin_memory=True,
        tokenizer_info=tokenizer_info,
    )
    
    # Calculate max training steps
    num_rollout_iters_per_epoch = len(train_dataloader.dataset) // args.rollout.rollout_batch_size
    num_update_steps_per_rollout = args.rollout.rollout_batch_size * args.rollout.n_samples_per_prompt // args.train.train_batch_size
    max_rollout_iters = math.ceil(args.train.num_epochs * num_rollout_iters_per_epoch)
    strategy.log(f"Max training iterations: {max_rollout_iters}")
    
    # Initialize student model on all workers
    ray.get(student_model.async_init_model_from_pretrained(
        strategy, (max_rollout_iters * num_update_steps_per_rollout), 
        teacher_tokenizer=teacher_tokenizer, 
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
        ray.get(student_model.async_save_model(args.ckpt.save_path))
        strategy.log("Training completed.")
    finally:
        teacher_model.shutdown()
        rollout_group.shutdown()


if __name__ == "__main__":
    args = init_args("on_policy_kd")
    train(args)

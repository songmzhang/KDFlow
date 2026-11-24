import math

import ray

from kdflow.cli.common import (
    init_ray,
    build_student_group,
    build_teacher_group,
    build_tokenizer_info,
    build_dataloaders,
)
from kdflow.ray.placement_group import create_placement_group
from kdflow.trainer import OffPolicyKDTrainer
from kdflow.datasets import SFTDataset
from kdflow.backend import get_strategy
from kdflow.arguments import init_args


def train(args):
    init_ray()
    
    strategy = get_strategy(args)
    strategy.print(args)
    
    # Create placement group for resource allocation
    num_gpus = args.train.num_nodes * args.train.num_gpus_per_node
    pg_info = create_placement_group(num_gpus)
    strategy.print(f"Placement group ready with {num_gpus} GPUs")
    
    teacher_model = build_teacher_group(args, strategy, pg_info)
    student_model = build_student_group(args, pg_info)
    
    teacher_tokenizer, tokenizer_info = build_tokenizer_info(args, strategy)
    
    train_dataloader, eval_dataloader = build_dataloaders(
        args, strategy, SFTDataset,
        batch_size=args.train.micro_train_batch_size,
        eval_batch_size=1,
        pin_memory=False,
        tokenizer_info=tokenizer_info,
    )
    
    # Calculate max training steps
    num_update_steps_per_epoch = len(train_dataloader.dataset) // args.train.train_batch_size
    max_steps = math.ceil(args.train.num_epochs * num_update_steps_per_epoch)
    strategy.log(f"Max training steps: {max_steps}")
    
    # Initialize student model on all workers
    ray.get(student_model.async_init_model_from_pretrained(
        strategy, max_steps, 
        teacher_tokenizer=teacher_tokenizer, 
        tokenizer_info=tokenizer_info,
    ))
    
    strategy.log("Models initialized on all student actors")
    
    # Initialize OffPolicyKDRayTrainer (Single controller)
    trainer = OffPolicyKDTrainer(
        strategy=strategy,
        student_model=student_model,
        teacher_model=teacher_model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        max_steps=max_steps,
        num_update_steps_per_epoch=num_update_steps_per_epoch,
    )
    
    try:
        trainer.fit()
        ray.get(student_model.async_save_model(args.ckpt.save_path))
        strategy.log("Training completed.")
    finally:
        teacher_model.shutdown()


if __name__ == "__main__":
    args = init_args("off_policy_kd")
    train(args)

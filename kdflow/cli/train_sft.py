import os
import math
from transformers.trainer import get_scheduler

from kdflow.arguments import init_args
from kdflow.datasets.utils import blending_datasets
from kdflow.models.model import DistillModel
from kdflow.datasets import SFTDataset
from kdflow.trainer.sft_trainer import SFTTrainer
from kdflow.backend import get_strategy


def train(args):
    strategy = get_strategy(args)
    strategy.setup_distributed()
    
    # load student model
    student = DistillModel(strategy)
    
    strategy.print(student)

    # bf16 mixed precision training in fsdp
    if args.train.bf16:
        student = student.float()   # cast student to fp32 for mixed precision training in fsdp
    student = strategy.prepare(student)

    optim = strategy.create_optimizer(
        student, 
        lr=args.train.learning_rate, 
        betas=args.train.adam_betas, 
        weight_decay=args.train.weight_decay
    )
    
    train_data = blending_datasets(
        args.data.train_dataset_path,
        args.data.train_dataset_probs,
        strategy,
        args.train.seed,
        max_count=args.data.max_samples,
        dataset_split=args.data.train_split,
    )
    train_data = train_data.select(range(min(args.data.max_samples, len(train_data))))
    train_dataset = SFTDataset(
        train_data, 
        student.tokenizer, 
        args.data.max_len,
        strategy, 
        input_template=args.data.input_template,
        max_data_num=args.data.max_samples,
    )
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.train.micro_train_batch_size,
        pin_memory=True,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )

    if getattr(args.data, "eval_dataset_path", None):
        eval_data = blending_datasets(
            args.data.eval_dataset_path,
            None,  # No probability sampling for eval datasets
            strategy,
            dataset_split=args.data.eval_split,
        )
        eval_data = eval_data.select(range(min(args.data.max_samples, len(eval_data))))
        eval_dataset = SFTDataset(
            eval_data, 
            student.tokenizer, 
            args.data.max_len,
            strategy, 
            input_template=args.data.input_template
        )
        eval_dataloader = strategy.setup_dataloader(eval_dataset, 1, True, False, collate_fn=eval_dataset.collate_fn)
    else:
        eval_dataloader = None

    # scheduler
    num_update_steps_per_epoch = len(train_dataset) // args.train.train_batch_size
    max_steps = math.ceil(args.train.num_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        args.train.lr_scheduler,
        optim,
        num_warmup_steps=math.ceil(max_steps * args.train.lr_warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.train.min_lr},
    )

    # gradient_checkpointing
    if args.train.gradient_checkpointing:
        student.gradient_checkpointing_enable()
    
    # load checkpoint
    global_step, start_epoch = 0, 0
    os.makedirs(args.train.save_path, exist_ok=True)

    # configure Trainer
    trainer = SFTTrainer(
        args,
        strategy,
        student_model=student,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        optimizer=optim,
        num_update_steps_per_epoch=num_update_steps_per_epoch,
    )

    trainer.fit(global_step, start_epoch)

    # save model checkpoint after fitting on only rank0
    strategy.save_model(student, student.tokenizer, args.train.save_path)
    strategy.log("Training completed and model saved.")


if __name__ == "__main__":
    args = init_args()
    train(args)

import math
import os
import time
import torch
import numpy as np
import torch.distributed as dist

from datetime import timedelta
from typing import Optional
from collections import defaultdict

from kdflow.algorithms.sft import SFT
from kdflow.utils.logging_utils import init_logger


logger = init_logger(__name__)


class SFTTrainer:
    def __init__(
        self, 
        args, 
        strategy,
        student_model,
        train_dataloader,
        eval_dataloader=None,
        scheduler=None,
        optimizer=None,
        num_update_steps_per_epoch=None
    ) -> None:
        self.args = args
        self.strategy = strategy
        self.student = student_model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_update_steps_per_epoch = num_update_steps_per_epoch
        self.epochs = args.train.num_epochs
        
        self.kd_algorithm = SFT(strategy=strategy, student_model=self.student)
        
        self.log_state = defaultdict(list)
        
        self._init_loggers()
    
    def _init_loggers(self) -> None:
        """Initialize wandb loggers."""
        self._wandb = None
        if self.args.log.use_wandb and dist.get_rank() == 0:
            import wandb

            self._wandb = wandb
            if self.args.log.wandb_mode != "offline" and not wandb.api.api_key:
                wandb.login()
            wandb.init(
                entity=self.args.log.wandb_org,
                project=self.args.log.wandb_project,
                group=self.args.log.wandb_group,
                name=self.args.log.wandb_run_name,
                config=self.args.log.__dict__,
                reinit=True,
                mode=self.args.log.wandb_mode,
                dir=self.args.log.wandb_dir,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)
    
    def _print_training_config(self) -> None:
        """Log training configuration before training starts."""
        total_steps = self.num_update_steps_per_epoch * self.epochs
        grad_accum = self.args.train.train_batch_size * self.args.model.ring_attn_size \
            // (self.args.train.micro_train_batch_size * self.args.train.num_nodes * self.args.train.num_gpus_per_node)
        
        self.strategy.info("******* Start Training *******")
        self.strategy.info(f"  Num Epochs:            {self.epochs}")
        self.strategy.info(f"  Steps per Epoch:       {self.num_update_steps_per_epoch}")
        self.strategy.info(f"  Total Training Steps:  {total_steps}")
        self.strategy.info(f"  Per-device Batch Size: {self.args.train.micro_train_batch_size}")
        self.strategy.info(f"  Gradient Accumulation: {grad_accum}")
        self.strategy.info(f"  Learning Rate:         {self.args.train.learning_rate}")
    
    def fit(self, global_step=0, start_epoch=0):
        # get eval and save steps
        if self.args.train.eval_steps == -1:
            self.args.train.eval_steps = float("inf")  # Evaluate once per epoch
        if self.args.train.save_steps == -1:
            self.args.train.save_steps = self.num_update_steps_per_epoch  # do not save ckpt
        
        self.global_step = global_step
        
        # Print training configuration
        self._print_training_config()
        
        self.start_time = time.time()
        for epoch in range(start_epoch, self.epochs):
            self.current_epoch = epoch
            self.train_dataloader.sampler.set_epoch(epoch)
            self.student.train()
            
            data_iter = iter(self.train_dataloader)
            self.optimizer.zero_grad(set_to_none=True)
            # train a global step
            while True:
                global_batch, global_batch_token_num = [], 0
                try:
                    for _ in range(self.strategy.accumulated_gradient):
                        micro_batch = next(data_iter)
                        global_batch.append(micro_batch)
                        global_batch_token_num += micro_batch["stu_loss_mask"].sum()
                except StopIteration:
                    break
                
                global_batch_token_num = global_batch_token_num.to(torch.cuda.current_device())
                dist.all_reduce(global_batch_token_num, op=dist.ReduceOp.SUM)
                avg_micro_batch_token_num = global_batch_token_num / (self.strategy.accumulated_gradient * dist.get_world_size())
                
                self.global_step += 1
                for micro_step, micro_batch in enumerate(global_batch):
                    for key in micro_batch:
                        micro_batch[key] = micro_batch[key].to(torch.cuda.current_device())
                    micro_batch["avg_micro_batch_token_num"] = avg_micro_batch_token_num
                    
                    status = self.kd_algorithm.training_step(micro_batch)
                    loss = status["loss"]
                    self.strategy.backward(loss, self.student, self.optimizer)
                    self.strategy.optimizer_step(self.optimizer, self.student, self.scheduler)
                    
                    if hasattr(self.student, "get_global_grad_norm") and self.student.get_global_grad_norm() is not None:
                        status["grad_norm"] = self.student.get_global_grad_norm()
                    elif hasattr(self.student, "clip_grad_norm_"):
                        status["grad_norm"] =  self.student.clip_grad_norm_(max_norm=self.args.train.max_norm).item()
                    status["lr"] = self.scheduler.get_last_lr()[0]
                    self.logging(micro_step, status)
                
            self.strategy.log(f"Saving model after epoch {epoch + 1}")
            save_path = os.path.join(self.args.train.save_path, f"epoch_{epoch + 1}")
            self.strategy.save_model(self.student, self.student.tokenizer, save_path)

        total_time = time.time() - self.start_time
        self.strategy.log(f"Training done, totally cost {str(timedelta(seconds=total_time)).split('.')[0]}")

        if self._wandb is not None and dist.get_rank() == 0:
            self._wandb.finish()
            
    def logging(self, step, current_log_state):
        for key in current_log_state:
            self.strategy.all_reduce(current_log_state[key], op="mean")
        
        for k in current_log_state:
            self.log_state[k].append(current_log_state[k])
                
        if (step + 1) == self.strategy.accumulated_gradient and self.global_step % self.args.log.logging_steps == 0:
            progress = self.global_step / self.num_update_steps_per_epoch / self.epochs
            eta = int(time.time() - self.start_time) * (1 - progress) / progress
            progress_str = "epoch [{current_epoch}/{total_epoch}], " \
                "step [{current_step}/{total_step}], " \
                "train_progress [{progress:.2f}%], " \
                "Elapsed: {elapsed}, " \
                "ETA: {eta}, ".format(
                current_epoch=self.current_epoch + 1, 
                total_epoch=self.epochs, 
                current_step=self.global_step, 
                total_step=self.num_update_steps_per_epoch * self.epochs, 
                progress=progress * 100,
                elapsed=str(timedelta(seconds=(time.time() - self.start_time))).split(".")[0],
                eta=str(timedelta(seconds=eta)).split(".")[0]
            )
            for k in self.log_state:
                if isinstance(self.log_state[k], list) and len(self.log_state[k]) > 0:
                    self.log_state[k] = sum(self.log_state[k]) / len(self.log_state[k])
            if dist.get_rank() == 0:
                log_info = []
                for k in self.log_state:
                    if k == "lr":
                        log_info.append(f"lr: {self.log_state[k]:.6e}")
                    else:
                        log_info.append(f"{k}: {self.log_state[k]:.6f}")
                log_str = ", ".join(log_info)
                log_str = progress_str + log_str
                self.strategy.log(log_str)

                if self._wandb is not None:
                    logs = {"train/global_step": self.global_step}
                    for k in self.log_state:
                        logs[f"train/{k}"] = self.log_state[k]
                    self._wandb.log(logs)

            for k in self.log_state:
                self.log_state[k] = []

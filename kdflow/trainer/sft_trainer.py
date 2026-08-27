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
from kdflow.utils.logging_utils import (
    define_wandb_metrics,
    init_logger,
    log_eval_metrics,
)
from kdflow.utils.dynamic_bsz import rearrange_global_batch


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
        
        self.dp_group = strategy.sp_mesh['dp'].get_group()
        
        self.kd_algorithm = SFT(strategy=strategy, student_model=self.student)
        
        self.log_state = defaultdict(list)
        
        self._init_loggers()
    
    def _init_loggers(self) -> None:
        """Initialize wandb loggers."""
        self._wandb = None
        if self.args.log.use_wandb and dist.get_rank() == 0:
            import wandb

            if self.args.log.sync_swanlab:
                import swanlab
                swanlab.sync_wandb(wandb_run=False)

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

            define_wandb_metrics(wandb)
    
    def _print_training_config(self) -> None:
        """Log training configuration before training starts."""
        total_steps = self.num_update_steps_per_epoch * self.epochs
        num_data = len(getattr(self.train_dataloader, "dataset", self.train_dataloader))
        grad_accum = self.args.train.train_batch_size * self.args.model.ring_attn_size \
            // (self.args.train.micro_train_batch_size * self.args.train.num_nodes * self.args.train.num_gpus_per_node)

        def log_config(name, value):
            self.strategy.log(f"  {name:<32} {value}")
        
        self.strategy.log("******* Start Training *******")
        log_config("Num GPUs:", dist.get_world_size())
        log_config("Num Data:", num_data)
        log_config("Num Epochs:", self.epochs)
        log_config("Train Batch Size:", self.args.train.train_batch_size)
        log_config("Steps Per Epoch:", self.num_update_steps_per_epoch)
        log_config("Total Training Steps:", total_steps)
        if self.args.train.use_dynamic_bsz:
            log_config("Enable Dynamic Batch Size:", self.args.train.use_dynamic_bsz)
            log_config("Max Token Len Per GPU:", self.args.train.max_token_len_per_gpu)
            log_config("Gradient Accumulation:", "dynamic")
        else:
            log_config("Per-device Batch Size:", self.args.train.micro_train_batch_size)
            log_config("Gradient Accumulation:", grad_accum)
        log_config("Learning Rate:", self.args.train.learning_rate)
    
    def fit(self, global_step=0, start_epoch=0):
        self.global_step = global_step
        
        # Print training configuration
        self._print_training_config()
        
        self.start_time = time.time()
        if self.eval_dataloader is not None and self.args.train.eval_steps < float("inf") and self.global_step == 0:
            self.strategy.log(f"Start evaluating at global step {self.global_step}")
            self.evaluate()

        for epoch in range(start_epoch, self.epochs):
            self.current_epoch = epoch
            self.train_dataloader.sampler.set_epoch(epoch)
            self.student.train()
            
            data_iter = iter(self.train_dataloader)
            self.optimizer.zero_grad(set_to_none=True)
            # train a global step
            while True:
                step_start = time.time()
                global_batch, global_batch_token_num = [], 0
                try:
                    for _ in range(self.strategy.accumulated_gradient):
                        micro_batch = next(data_iter)
                        global_batch.append(micro_batch)
                        global_batch_token_num += micro_batch["stu_loss_mask"].sum()
                except StopIteration:
                    break
                
                if self.args.train.use_dynamic_bsz:
                    global_batch = rearrange_global_batch(
                        global_batch,
                        max_token_len=self.args.train.max_token_len_per_gpu,
                        dp_group=self.dp_group,
                    )
                    self.strategy.accumulated_gradient = len(global_batch)
                    self.strategy.step = 0
                
                global_batch_token_num = global_batch_token_num.to(torch.cuda.current_device())
                dist.all_reduce(global_batch_token_num, op=dist.ReduceOp.SUM)
                avg_micro_batch_token_num = global_batch_token_num / (len(global_batch) * dist.get_world_size())
                
                self.global_step += 1
                for micro_step, micro_batch in enumerate(global_batch):
                    for key in micro_batch:
                        micro_batch[key] = micro_batch[key].to(torch.cuda.current_device())
                    micro_batch["avg_micro_batch_token_num"] = avg_micro_batch_token_num
                    
                    status = self.kd_algorithm.training_step(micro_batch)
                    loss = status["train/loss"]
                    self.strategy.backward(loss, self.student, self.optimizer)
                    status["train/grad_norm"] = torch.nn.utils.clip_grad_norm_(
                        self.student.parameters(), max_norm=float("inf")).item()
                    self.strategy.optimizer_step(self.optimizer, self.student, self.scheduler)
                    
                    status["train/lr"] = self.scheduler.get_last_lr()[0]
                    if micro_step + 1 == len(global_batch):
                        status["timing/step_time"] = time.time() - step_start
                    self.logging(micro_step, status)

                if (
                    self.eval_dataloader is not None
                    and self.global_step % self.args.train.eval_steps == 0
                ):
                    self.strategy.log(f"Start evaluating at global step {self.global_step}")
                    self.evaluate()
                
                if self.global_step % self.args.train.save_steps == 0:
                    self.strategy.log(f"Saving model at global step {self.global_step}")
                    save_path = os.path.join(self.args.train.save_path, f"epoch_{epoch + 1}_global_step_{self.global_step}")
                    self.strategy.save_model(self.student, save_path)

            self.strategy.log(f"Saving model after epoch {epoch + 1}")
            save_path = os.path.join(self.args.train.save_path, f"epoch_{epoch + 1}")
            self.strategy.save_model(self.student, save_path)

        total_time = time.time() - self.start_time
        self.strategy.log(f"Training done, totally cost {str(timedelta(seconds=total_time)).split('.')[0]}")

        if self._wandb is not None and dist.get_rank() == 0:
            self._wandb.finish()

    @torch.no_grad()
    def evaluate(self):
        """Evaluate SFT loss on the validation set."""
        was_training = self.student.training
        self.student.eval()
        metric_sums = defaultdict(float)
        num_tokens = 0.0

        try:
            for micro_batch in self.eval_dataloader:
                micro_batch = {
                    key: value.to(torch.cuda.current_device())
                    if isinstance(value, torch.Tensor) else value
                    for key, value in micro_batch.items()
                }
                batch_tokens = micro_batch["stu_loss_mask"].sum().to(torch.cuda.current_device())
                dist.all_reduce(batch_tokens, op=dist.ReduceOp.SUM)
                micro_batch["avg_micro_batch_token_num"] = batch_tokens / dist.get_world_size()

                metrics = self.strategy.all_reduce(
                    self.kd_algorithm.training_step(micro_batch), op="mean"
                )
                weight = batch_tokens.item()
                for key, value in metrics.items():
                    value = value.item() if hasattr(value, "item") else float(value)
                    metric_sums[key] += value * weight
                num_tokens += weight
        finally:
            if was_training:
                self.student.train()

        if num_tokens == 0:
            return {}
        metrics = {key: value / num_tokens for key, value in metric_sums.items()}
        metrics["eval/num_tokens"] = num_tokens / self.args.model.ring_attn_size
        return log_eval_metrics(self.strategy, self._wandb, metrics, self.global_step)
            
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
                    if k == "train/lr":
                        log_info.append(f"{k}: {self.log_state[k]:.6e}")
                    else:
                        log_info.append(f"{k}: {self.log_state[k]:.6f}")
                log_str = ", ".join(log_info)
                log_str = progress_str + log_str
                self.strategy.log(log_str)

                if self._wandb is not None:
                    logs = {"train/global_step": self.global_step}
                    for k in self.log_state:
                        logs[k] = self.log_state[k]
                    self._wandb.log(logs)

            for k in self.log_state:
                self.log_state[k] = []

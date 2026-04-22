import os
import time
import json
from datetime import timedelta
from typing import Dict, List, Optional, Callable, Any
from collections import defaultdict

import ray
import torch
import torch.distributed as dist
from tqdm import tqdm

from kdflow.utils.logging_utils import init_logger
from kdflow.utils.dynamic_bsz import rearrange_global_batch


logger = init_logger(__name__)


class OffPolicyKDTrainer:
    """
    Ray-based trainer for off-policy knowledge distillation.
    """
    
    def __init__(
        self,
        strategy,
        student_model,
        teacher_model,
        train_dataloader,
        eval_dataloader=None,
        max_steps: int = None,
        num_update_steps_per_epoch: int = None,
    ) -> None:
        """
        Initialize the trainer.
        
        Args:
            strategy: Training strategy containing configuration
            student_model: StudentActorGroup
            teacher_model: TeacherActorGroup
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader (optional)
            max_steps: Maximum training steps
            num_update_steps_per_epoch: Number of update steps per epoch
        """
        self.strategy = strategy
        self.args = strategy.args
        self.student = student_model
        self.teacher = teacher_model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.max_steps = max_steps
        self.num_update_steps_per_epoch = num_update_steps_per_epoch
        self.epochs = self.args.train.num_epochs
        self.world_size = self.args.train.num_nodes * self.args.train.num_gpus_per_node
        self.dp_size = self.world_size // self.args.model.ring_attn_size
        
        self.log_state = defaultdict(list)
        self._init_loggers()
    
    def _init_loggers(self) -> None:
        """Initialize wandb loggers."""
        self._wandb = None
        
        if self.args.log.use_wandb:
            import wandb
            
            self._wandb = wandb
            if self.args.log.wandb_mode != "offline" and not wandb.api.api_key:
                wandb.login()
            wandb.init(
                entity=self.args.log.wandb_org,
                project=self.args.log.wandb_project,
                group=self.args.log.wandb_group,
                name=self.args.log.wandb_run_name,
                config=vars(self.args),
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
        
        logger.info("******* Start Training *******")
        logger.info(f"  Num Epochs:            {self.epochs}")
        logger.info(f"  Steps per Epoch:       {self.num_update_steps_per_epoch}")
        logger.info(f"  Total Training Steps:  {total_steps}")
        logger.info(f"  Per-device Batch Size: {self.args.train.micro_train_batch_size}")
        logger.info(f"  Gradient Accumulation: {grad_accum}")
        logger.info(f"  Learning Rate:         {self.args.train.learning_rate}")
        logger.info(f"  KD Algorithm:          {self.args.kd.kd_algorithm}")
        logger.info(f"  KD Loss Function:      {self.args.kd.kd_loss_fn}")
    
    def fit(self, global_step=0, start_epoch=0):
        self.global_step = global_step
        
        # Print training configuration and initialize loggers
        self._print_training_config()
        
        self.start_time = time.time()
        num_micro_batches = self.args.train.train_batch_size // self.args.train.micro_train_batch_size
        self.teacher_forward_n = min(self.args.kd.teacher_forward_n_batches, len(self.train_dataloader))
        teacher_forward_n = self.teacher_forward_n
        
        for epoch in range(start_epoch, self.epochs):
            self.current_epoch = epoch
            self.train_dataloader.sampler.set_epoch(epoch)
            
            data_iter = iter(self.train_dataloader)
            is_epoch_finished = False
            while True:
                # Collect N global batches for teacher forward
                all_global_batches = []
                for _ in range(teacher_forward_n):
                    global_batch = []
                    try:
                        for _ in range(num_micro_batches):
                            micro_batch = next(data_iter)
                            global_batch.append(micro_batch)
                    except StopIteration:
                        is_epoch_finished = True
                        break
                    if not is_epoch_finished:
                        if self.args.train.use_dynamic_bsz:
                            global_batch = rearrange_global_batch(
                                global_batch,
                                max_token_len=self.args.train.max_token_len_per_gpu,
                                dp_size=self.dp_size,
                            )
                        global_batch_token_num = sum(mb["stu_loss_mask"].sum() for mb in global_batch)
                        avg_micro_batch_token_num = global_batch_token_num / len(global_batch)
                        for mb in global_batch:
                            mb["avg_micro_batch_token_num"] = avg_micro_batch_token_num
                        all_global_batches.append(global_batch)
                
                if not all_global_batches:
                    break
                
                # ===== Teacher Phase (batch N global batches) =====
                teacher_start = time.time()
                if self.args.train.enable_sleep:
                    self.teacher.wakeup()
                
                # Concat all global batches for teacher forward
                merged_batch = [mb for gb in all_global_batches for mb in gb]
                merged_batch = self.teacher.forward(merged_batch)
                # Split back to individual global batches
                idx = 0
                for i, gb in enumerate(all_global_batches):
                    all_global_batches[i] = merged_batch[idx:idx + len(gb)]
                    idx += len(gb)
                if self.args.train.enable_sleep:
                    self.teacher.sleep()
                
                teacher_step_fwd_time = (time.time() - teacher_start) / len(all_global_batches)

                # ===== Student Phase (train N steps) =====
                if self.args.train.enable_sleep:
                    self.student.wakeup()
                for global_batch in all_global_batches:
                    student_start = time.time()
                    self.global_step += 1
                    status_list = ray.get(self.student.async_run_distill(global_batch))
                    for k in status_list[0].keys():
                        self.log_state[k].append(sum(s[k] for s in status_list) / len(status_list))
                    self.log_state["teacher_step_fwd_time"].append(teacher_step_fwd_time)
                    self.log_state["student_step_train_time"].append((time.time() - student_start))
                    self.logging()
                    
                    if self.global_step % self.args.train.save_steps == 0:
                        self.strategy.log(f"Saving model at global step {self.global_step}")
                        save_path = os.path.join(self.args.train.save_path, f"epoch_{epoch + 1}_global_step_{self.global_step}")
                        ray.get(self.student.async_save_model(save_path))
                
                if self.args.train.enable_sleep:
                    self.student.sleep()
                
            self.strategy.log(f"Saving model after epoch {epoch + 1}")
            save_path = os.path.join(self.args.train.save_path, f"epoch_{epoch + 1}")
            ray.get(self.student.async_save_model(save_path))

        total_time = time.time() - self.start_time
        self.strategy.log(f"Training done, totally cost {str(timedelta(seconds=total_time)).split('.')[0]}")

        if self._wandb is not None:
            self._wandb.finish()
            
    def logging(self):
        if self.global_step % self.args.log.logging_steps == 0:
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

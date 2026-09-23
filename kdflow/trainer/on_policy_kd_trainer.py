import time
from datetime import timedelta
from typing import Callable, Dict, Optional
from collections import defaultdict

import ray

from kdflow.trainer.rollout_manager import RolloutManager
from kdflow.utils.logging_utils import (
    define_wandb_metrics,
    init_logger,
    log_eval_metrics,
    normalize_eval_metrics,
)
from kdflow.metrics import accumulate_metric_stats, average_metric_stats
from kdflow.utils.dynamic_bsz import rearrange_global_batch
from kdflow.backend.fsdp.checkpoint import resolve_resume_checkpoint


logger = init_logger(__name__)

class OnPolicyKDTrainer:
    """
    Ray-based trainer for on-policy knowledge distillation.
    """
    
    def __init__(
        self,
        strategy,
        student_model,
        teacher_model,
        rollout_group,
        is_same_tokenizer: bool,
        train_dataloader,
        eval_dataloader=None,
        max_rollout_iters: int = None,
        num_rollout_iters_per_epoch: int = None,
        generate_kwargs: Dict[str, float] = None,
        custom_eval_fn: Optional[Callable] = None,
    ) -> None:
        """
        Initialize the trainer.
        
        Args:
            strategy: Training strategy containing configuration
            student_model: StudentActorGroup
            teacher_model: TeacherActorGroup
            rollout_group: RolloutGroup
            is_same_tokenizer: Whether student and teacher use same tokenizer
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader (optional)
            max_rollout_iters: Maximum rollout iterations in training
            num_rollout_iters_per_epoch: Number of rollout iterations per epoch
        """
        self.strategy = strategy
        self.args = strategy.args
        self.student = student_model
        self.teacher = teacher_model
        self.rollout_group = rollout_group
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.max_rollout_iters = max_rollout_iters
        self.num_rollout_iters_per_epoch = num_rollout_iters_per_epoch
        self.generate_kwargs = generate_kwargs
        self.custom_eval_fn = custom_eval_fn
        self.epochs = self.args.train.num_epochs
        self.use_lora = self.args.model.lora_rank > 0
        self.rollout_manager = RolloutManager(
            strategy=strategy,
            rollout_group=rollout_group,
            is_same_tokenizer=is_same_tokenizer,
            generate_kwargs=generate_kwargs,
        )
        
        self.world_size = self.args.train.num_nodes * self.args.train.num_gpus_per_node
        self.dp_size = self.world_size // self.args.model.ring_attn_size
        
        assert self.args.kd.kd_ratio == 1.0, "On-policy KD only supports kd_ratio=1.0."
        
        self.log_state = defaultdict(list)
        self.metric_stats = {}
        self._init_loggers()
    
    def _init_loggers(self) -> None:
        """Initialize wandb loggers."""
        self._wandb = None
        
        if self.args.log.use_wandb:
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
                config=vars(self.args),
                reinit=True,
                mode=self.args.log.wandb_mode,
                dir=self.args.log.wandb_dir,
            )
            
            define_wandb_metrics(wandb)
    
    def _print_training_config(self) -> None:
        """Log training configuration before training starts."""
        num_data = len(getattr(self.train_dataloader, "dataset", self.train_dataloader))
        num_update_per_rollout = self.args.rollout.n_samples_per_prompt * self.args.rollout.rollout_batch_size \
            // self.args.train.train_batch_size
        total_steps = self.max_rollout_iters * num_update_per_rollout
        grad_accum = self.args.train.train_batch_size * self.args.model.ring_attn_size \
            // (self.args.train.micro_train_batch_size * self.args.train.num_nodes * self.args.train.num_gpus_per_node)

        def log_config(name, value):
            logger.info(f"  {name:<32} {value}")
        
        logger.info("******* Start Training *******")
        log_config("Num GPUs:", self.world_size)
        log_config("Num Data:", num_data)
        log_config("Num Epochs:", self.epochs)
        log_config("Rollout Batch Size:", self.args.rollout.rollout_batch_size)
        log_config("Train Batch Size:", self.args.train.train_batch_size)
        log_config("Rollout Iterations Per Epoch:", self.num_rollout_iters_per_epoch)
        log_config("Total Rollout Iterations:", self.max_rollout_iters)
        log_config("Num Updates Per Rollout:", num_update_per_rollout)
        log_config("Total Num Updates:", total_steps)
        if self.args.train.use_dynamic_bsz:
            log_config("Enable Dynamic Batch Size:", self.args.train.use_dynamic_bsz)
            log_config("Max Token Len Per GPU:", self.args.train.max_token_len_per_gpu)
            log_config("Gradient Accumulation:", "dynamic")
        else:
            log_config("Per-device Batch Size:", self.args.train.micro_train_batch_size)
            log_config("Gradient Accumulation:", grad_accum)
        log_config("Learning Rate:", self.args.train.learning_rate)
        log_config("KD Algorithm:", self.args.kd.kd_algorithm)
        log_config("KD Loss Function:", self.args.kd.kd_loss_fn)

    def _sync_rollout_policy(self):
        if self.args.train.enable_sleep:
            self.rollout_group.wakeup(tags=["weights"])
        try:
            if self.use_lora:
                adapter = self.student.export_lora_adapter()
                lora_name = f"step-{self.global_step:08d}"
                self.rollout_group.update_lora_adapter(adapter, lora_name)
            else:
                self.student.update_rollout_weights()
        finally:
            if self.args.train.enable_sleep:
                self.rollout_group.sleep(tags=["weights"])

    def _prepare_global_batches(self, rollout_samples, num_micro_batches):
        all_global_batches = []
        for i in range(0, len(rollout_samples), num_micro_batches):
            global_batch = rollout_samples[i : i + num_micro_batches]
            if self.args.train.use_dynamic_bsz:
                global_batch = rearrange_global_batch(
                    global_batch,
                    max_token_len=self.args.train.max_token_len_per_gpu,
                    dp_size=self.dp_size,
                )

            batch_tokens = sum(mb["stu_loss_mask"].sum() for mb in global_batch)
            avg_micro_batch_token_num = batch_tokens / len(global_batch)
            for micro_batch in global_batch:
                micro_batch["avg_micro_batch_token_num"] = avg_micro_batch_token_num
            all_global_batches.append(global_batch)
        return all_global_batches
    
    def fit(self):
        self.global_step, start_epoch = 0, 0
        checkpoint_path = resolve_resume_checkpoint(
            self.args.ckpt.save_path, self.args.ckpt.resume_from, self.args.ckpt.resume_training,
        )
        if checkpoint_path is not None:
            self.strategy.log(f"Resuming training from {checkpoint_path}")
            state = self.student.load_checkpoint(checkpoint_path)
            self.global_step = state["global_step"]
            start_epoch = state["epoch"] if state["epoch_end"] else state["epoch"] - 1
            if not state["epoch_end"]:
                self.train_dataloader.sampler.set_epoch(start_epoch)
                self.train_dataloader.load_state_dict(state["trainer_state"]["data_loader_state_dict"])
        
        # Print training configuration and initialize loggers
        self._print_training_config()

        # Sync model weights before training
        if not self.use_lora:
            rollout_tp_size = getattr(self.args.rollout, "rollout_tp_size", 1)
            self.student.connect_rollout_engines(self.rollout_group.actors, rollout_tp_size)
        self._sync_rollout_policy()
            
        if self.args.model.student_name_or_path == self.args.model.teacher_name_or_path:   # for self-distillation
            num_gpus_per_teacher_actor = self.args.kd.teacher_tp_size * self.args.kd.teacher_pp_size
            self.student.connect_teacher_actors(self.teacher.teacher_engines, num_gpus_per_teacher_actor)
            if self.global_step >= self.args.kd.teacher_update_freq:
                if self.args.train.enable_sleep:
                    self.teacher.wakeup(tags=["weights"])
                self.student.update_teacher_weights(restore=True)
                if self.args.train.enable_sleep:
                    self.teacher.sleep(tags=["weights"])
        
        self.start_global_step = self.global_step
        self.start_time = time.time()
        num_micro_batches = self.args.train.train_batch_size // self.args.train.micro_train_batch_size

        if self.eval_dataloader is not None and self.args.train.eval_steps < float("inf") and self.global_step == 0:
            self.strategy.log(f"Evaluating model at global step {self.global_step}")
            self.evaluate()
        
        for epoch in range(start_epoch, self.epochs):
            self.current_epoch = epoch
            self.train_dataloader.sampler.set_epoch(epoch)
            
            for prompt_batch in self.train_dataloader:
                step_start = time.time()
                self.global_step += 1
                
                rollout_start = time.time()
                rollout_samples, rollout_metrics = self.rollout_manager.rollout(
                    prompt_batch,
                    global_step=self.global_step,
                    mode="train",
                )
                rollout_time = time.time() - rollout_start

                self.log_state["timing/rollout"].append(rollout_time)
                for name, value in rollout_metrics.items():
                    self.log_state[name].append(value)

                all_global_batches = self._prepare_global_batches(rollout_samples, num_micro_batches)
                micro_batch_tokens = [
                    mb["stu_attn_mask"].sum().item()
                    for global_batch in all_global_batches
                    for mb in global_batch
                ]
                mean_tokens = sum(micro_batch_tokens) / len(micro_batch_tokens)
                self.log_state["train/micro_batch_token_imbalance"].append(max(micro_batch_tokens) / mean_tokens)

                teacher_start = time.time()
                if self.args.train.enable_sleep:
                    self.teacher.wakeup()

                teacher_batches = sum(all_global_batches, [])
                teacher_batches = self.teacher.forward(teacher_batches)

                batch_idx = 0
                for i, global_batch in enumerate(all_global_batches):
                    next_batch_idx = batch_idx + len(global_batch)
                    all_global_batches[i] = teacher_batches[batch_idx:next_batch_idx]
                    batch_idx = next_batch_idx
                if batch_idx != len(teacher_batches):
                    raise RuntimeError(
                        f"Teacher forward returned {len(teacher_batches)} batches, expected {batch_idx}."
                    )

                if self.args.train.enable_sleep:
                    self.teacher.sleep()
                self.log_state["timing/teacher_forward"].append(time.time() - teacher_start)
                
                student_start = time.time()
                
                if self.args.train.enable_sleep:
                    self.student.wakeup()
                
                for global_batch in all_global_batches:
                    status_list = ray.get(self.student.async_run_distill(global_batch))
                    accumulate_metric_stats(self.metric_stats, status_list[0].pop("metric_stats", {}))
                    for k in status_list[0].keys():
                        self.log_state[k].append(sum(s[k] for s in status_list) / len(status_list))
                        
                self.log_state["timing/student_train"].append(time.time() - student_start)
                
                ray.get([actor.empty_cache.remote() for actor in self.student._actor_handlers])

                update_start = time.time()
                self._sync_rollout_policy()
                self.log_state["timing/rollout_weight_sync"].append(time.time() - update_start)
                
                # update weights in teacher actors (only for self-distillation)
                if self.args.model.teacher_name_or_path == self.args.model.student_name_or_path \
                    and self.global_step % self.args.kd.teacher_update_freq == 0:
                    if self.args.train.enable_sleep:
                        self.teacher.wakeup(tags=["weights"])
                    teacher_update_start = time.time()
                    self.student.update_teacher_weights()
                    self.log_state["timing/teacher_weight_sync"].append(time.time() - teacher_update_start)
                    if self.args.train.enable_sleep:
                        self.teacher.sleep(tags=["weights"])
                    
                if self.args.train.enable_sleep:
                    self.student.sleep()

                self.log_state["timing/step_time"].append(time.time() - step_start)
                self.logging()

                if (
                    self.eval_dataloader is not None
                    and self.global_step % self.args.train.eval_steps == 0
                ):
                    self.strategy.log(f"Evaluating model at global step {self.global_step}")
                    self.evaluate()
                
                if (
                    self.global_step % self.args.ckpt.save_steps == 0
                    and self.global_step < (epoch + 1) * self.num_rollout_iters_per_epoch
                ):
                    self.save_checkpoint()
        
            self.save_checkpoint(epoch_end=True)

        total_time = time.time() - self.start_time
        self.strategy.log(f"Training done, totally cost {str(timedelta(seconds=total_time)).split('.')[0]}")

        if self._wandb is not None:
            self._wandb.finish()
            
    def save_checkpoint(self, epoch_end=False):
        self.strategy.log(f"Saving checkpoint at global step {self.global_step}")
        trainer_state = None
        if self.args.ckpt.save_training_state:
            trainer_state = {"data_loader_state_dict": self.train_dataloader.state_dict()}
        return ray.get(self.student.async_save_checkpoint(
            self.current_epoch + 1, self.global_step, epoch_end=epoch_end, trainer_state=trainer_state,
        ))

    def evaluate(self):
        """Evaluate on validation set."""
        eval_prompts = sum(self.eval_dataloader, [])
        if not eval_prompts:
            return {}

        generate_kwargs = {**self.generate_kwargs, "temperature": 0.0}
        rollout_samples, rollout_metrics = self.rollout_manager.rollout(
            eval_prompts,
            global_step=self.global_step,
            mode="eval",
            **generate_kwargs,
        )
        predictions = sum((micro_batch["stu_responses"] for micro_batch in rollout_samples), [])
        labels = sum((micro_batch["labels"] for micro_batch in rollout_samples), [])

        eval_batches = rollout_samples
        if self.args.train.use_dynamic_bsz:
            eval_batches = rearrange_global_batch(
                eval_batches,
                max_token_len=self.args.train.max_token_len_per_gpu,
                dp_size=self.dp_size,
            )

        if self.args.train.enable_sleep:
            self.teacher.wakeup()
        eval_batches = self.teacher.forward(eval_batches)
        if self.args.train.enable_sleep:
            self.teacher.sleep()

        if self.args.train.enable_sleep:
            self.student.wakeup()
        metrics = ray.get(self.student.async_run_eval(eval_batches))[0]
        if self.args.train.enable_sleep:
            self.student.sleep()
        metrics.update(rollout_metrics)

        if self.custom_eval_fn is not None:
            custom_metrics = self.custom_eval_fn(predictions, labels)
            if not isinstance(custom_metrics, dict):
                raise TypeError("custom_eval_fn must return a dict")
            metrics.update(normalize_eval_metrics(custom_metrics))

        return log_eval_metrics(self.strategy, self._wandb, metrics, self.global_step)
            
    def logging(self):
        if self.global_step % self.args.log.logging_steps == 0:
            total_steps = self.num_rollout_iters_per_epoch * self.epochs
            progress = self.global_step / total_steps
            elapsed = time.time() - self.start_time
            completed_steps = self.global_step - self.start_global_step
            eta = elapsed / completed_steps * (total_steps - self.global_step)
            progress_str = "epoch [{current_epoch}/{total_epoch}], " \
                "step [{current_step}/{total_step}], " \
                "train_progress [{progress:.2f}%], " \
                "Elapsed: {elapsed}, " \
                "ETA: {eta}, ".format(
                current_epoch=self.current_epoch + 1, 
                total_epoch=self.epochs, 
                current_step=self.global_step, 
                total_step=total_steps,
                progress=progress * 100,
                elapsed=str(timedelta(seconds=elapsed)).split(".")[0],
                eta=str(timedelta(seconds=eta)).split(".")[0]
            )
            for k in self.log_state:
                if isinstance(self.log_state[k], list) and len(self.log_state[k]) > 0:
                    values = self.log_state[k]
                    self.log_state[k] = (
                        max(values) if k.endswith("/max") else sum(values) / len(values)
                    )
            for key in self.metric_stats:
                self.log_state.pop(key, None)
            self.log_state.update(average_metric_stats(self.metric_stats))
            self.metric_stats.clear()
            log_info = []
            for k in sorted(self.log_state):
                # Skip keys that have no values logged in this interval (e.g. teacher weight sync
                # is only logged every teacher_update_freq steps).
                if isinstance(self.log_state[k], list):
                    continue
                if k == "train/lr":
                    log_info.append(f"{k}: {self.log_state[k]:.6e}")
                else:
                    log_info.append(f"{k}: {self.log_state[k]:.6f}")
            # Append average phase times
            log_str = ", ".join(log_info)
            log_str = progress_str + log_str
            self.strategy.log(log_str)

            if self._wandb is not None:
                logs = {"train/global_step": self.global_step}
                for k in sorted(self.log_state):
                    if isinstance(self.log_state[k], list):
                        continue
                    logs[k] = self.log_state[k]
                self._wandb.log(logs)

            for k in self.log_state:
                self.log_state[k] = []

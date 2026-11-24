import time
from collections import defaultdict
from datetime import timedelta

import ray

from kdflow.backend.fsdp.checkpoint import resolve_resume_checkpoint
from kdflow.utils.dynamic_bsz import rearrange_global_batch
from kdflow.utils.logging_utils import define_wandb_metrics, init_logger, log_eval_metrics
from kdflow.metrics import accumulate_metric_stats, average_metric_stats


logger = init_logger(__name__)


class SFTTrainer:
    """Single-controller SFT trainer backed by StudentActorGroup."""

    def __init__(
        self,
        strategy,
        student_model,
        train_dataloader,
        eval_dataloader=None,
        max_steps: int = None,
        num_update_steps_per_epoch: int = None,
    ) -> None:
        self.strategy = strategy
        self.args = strategy.args
        self.student = student_model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.num_update_steps_per_epoch = num_update_steps_per_epoch
        self.epochs = self.args.train.num_epochs
        self.max_steps = max_steps
        self.world_size = self.args.train.num_nodes * self.args.train.num_gpus_per_node
        self.dp_size = self.world_size // self.args.model.ring_attn_size
        if self.world_size % self.args.model.ring_attn_size:
            raise ValueError("The number of GPUs must be divisible by ring_attn_size.")
        if self.args.train.train_batch_size % (self.args.train.micro_train_batch_size * self.dp_size):
            raise ValueError("train_batch_size must be divisible by micro_train_batch_size * DP size.")

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
        total_steps = self.num_update_steps_per_epoch * self.epochs
        num_data = len(getattr(self.train_dataloader, "dataset", self.train_dataloader))
        grad_accum = self.args.train.train_batch_size * self.args.model.ring_attn_size \
            // (self.args.train.micro_train_batch_size * self.args.train.num_nodes * self.args.train.num_gpus_per_node)

        def log_config(name, value):
            logger.info(f"  {name:<32} {value}")

        logger.info("******* Start Training *******")
        log_config("Num GPUs:", self.world_size)
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

        self._print_training_config()
        self.start_global_step = self.global_step
        self.start_time = time.time()
        num_micro_batches = self.args.train.train_batch_size // self.args.train.micro_train_batch_size

        if self.eval_dataloader is not None and self.args.train.eval_steps < float("inf") and self.global_step == 0:
            self.strategy.log(f"Start evaluating at global step {self.global_step}")
            self.evaluate()

        for epoch in range(start_epoch, self.epochs):
            self.current_epoch = epoch
            self.train_dataloader.sampler.set_epoch(epoch)
            data_iter = iter(self.train_dataloader)
            while True:
                step_start = time.time()
                global_batch = []
                try:
                    for _ in range(num_micro_batches):
                        micro_batch = next(data_iter)
                        global_batch.append(micro_batch)
                except StopIteration:
                    break

                if self.args.train.use_dynamic_bsz:
                    global_batch = rearrange_global_batch(
                        global_batch,
                        max_token_len=self.args.train.max_token_len_per_gpu,
                        dp_size=self.dp_size,
                    )
                if len(global_batch) % self.dp_size:
                    raise ValueError("The number of micro-batches must be divisible by DP size.")
                global_batch_token_num = sum(mb["stu_loss_mask"].sum() for mb in global_batch)
                if global_batch_token_num == 0:
                    raise ValueError("The global batch has no supervised tokens.")
                avg_micro_batch_token_num = global_batch_token_num / len(global_batch)
                for mb in global_batch:
                    mb["avg_micro_batch_token_num"] = avg_micro_batch_token_num

                micro_batch_tokens = [mb["stu_attn_mask"].sum().item() for mb in global_batch]
                mean_tokens = sum(micro_batch_tokens) / len(micro_batch_tokens)
                self.log_state["train/micro_batch_token_imbalance"].append(max(micro_batch_tokens) / mean_tokens)
                if self.args.train.enable_sleep:
                    self.student.wakeup()
                student_start = time.time()
                self.global_step += 1
                status_list = ray.get(self.student.async_run_distill(global_batch))
                student_step_train_time = time.time() - student_start
                accumulate_metric_stats(self.metric_stats, status_list[0].pop("metric_stats", {}))
                for k in status_list[0].keys():
                    self.log_state[k].append(sum(s[k] for s in status_list) / len(status_list))
                self.log_state["timing/student_train"].append(student_step_train_time)
                self.log_state["timing/step_time"].append(time.time() - step_start)
                self.logging()

                if self.args.train.enable_sleep:
                    self.student.sleep()

                if (
                    self.eval_dataloader is not None
                    and self.global_step % self.args.train.eval_steps == 0
                ):
                    self.strategy.log(f"Start evaluating at global step {self.global_step}")
                    self.evaluate()
                if (
                    self.global_step % self.args.ckpt.save_steps == 0
                    and self.global_step < (epoch + 1) * self.num_update_steps_per_epoch
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
        """Evaluate SFT loss without updating the student."""
        eval_batches = list(self.eval_dataloader)
        if not eval_batches:
            return {}
        if self.args.train.use_dynamic_bsz:
            eval_batches = rearrange_global_batch(
                eval_batches,
                max_token_len=self.args.train.max_token_len_per_gpu,
                dp_size=self.dp_size,
            )

        if self.args.train.enable_sleep:
            self.student.wakeup()
        metrics = ray.get(self.student.async_run_eval(eval_batches))[0]
        if self.args.train.enable_sleep:
            self.student.sleep()

        return log_eval_metrics(self.strategy, self._wandb, metrics, self.global_step)

    def logging(self):
        if self.global_step % self.args.log.logging_steps == 0:
            total_steps = self.num_update_steps_per_epoch * self.epochs
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
                    self.log_state[k] = sum(self.log_state[k]) / len(self.log_state[k])

            for key in self.metric_stats:
                self.log_state.pop(key, None)
            self.log_state.update(average_metric_stats(self.metric_stats))
            self.metric_stats.clear()
            log_info = []
            for k in sorted(self.log_state):
                if k == "train/lr":
                    log_info.append(f"{k}: {self.log_state[k]:.6e}")
                else:
                    log_info.append(f"{k}: {self.log_state[k]:.6f}")
            log_str = ", ".join(log_info)
            log_str = progress_str + log_str
            self.strategy.log(log_str)

            if self._wandb is not None:
                logs = {"train/global_step": self.global_step}
                for k in sorted(self.log_state):
                    logs[k] = self.log_state[k]
                self._wandb.log(logs)

            for k in self.log_state:
                self.log_state[k] = []

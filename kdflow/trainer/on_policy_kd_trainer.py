import os
import time
from tqdm import tqdm
from datetime import timedelta
from typing import Dict, List, Optional, Callable, Any
from collections import defaultdict

import ray
import torch
import torch.distributed as dist

from kdflow.datasets.utils import load_images
from kdflow.utils.logging_utils import init_logger
from kdflow.utils.utils import zero_pad_sequences


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
        student_tokenizer: Callable,
        teacher_tokenizer: Callable,
        is_same_tokenizer: bool,
        train_dataloader,
        eval_dataloader=None,
        max_steps: int = None,
        num_update_steps_per_epoch: int = None,
        generate_kwargs: Dict[str, float] = None,
    ) -> None:
        """
        Initialize the trainer.
        
        Args:
            strategy: Training strategy containing configuration
            student_model: StudentActorGroup
            teacher_model: TeacherActorGroup
            rollout_group: RolloutGroup
            student_tokenizer: Student model tokenizer
            teacher_tokenizer: Teacher model tokenizer
            is_same_tokenizer: Whether student and teacher use same tokenizer
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader (optional)
            max_steps: Maximum training steps
            num_update_steps_per_epoch: Number of update steps per epoch
        """
        self.strategy = strategy
        self.args = strategy.args
        self.student = student_model
        self.teacher = teacher_model
        self.rollout_group = rollout_group
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.is_same_tokenizer = is_same_tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.max_steps = max_steps
        self.num_update_steps_per_epoch = num_update_steps_per_epoch
        self.generate_kwargs = generate_kwargs
        self.epochs = self.args.train.num_epochs
        
        self.image_key = getattr(self.args.data, "image_key", None)
        self.stu_processor = None
        self.tea_processor = None
        if self.image_key:
            from transformers import AutoProcessor
            self.stu_processor = AutoProcessor.from_pretrained(self.args.model.student_name_or_path)
            if not self.is_same_tokenizer:
                self.tea_processor = AutoProcessor.from_pretrained(self.args.model.teacher_name_or_path)
            else:
                self.tea_processor = self.stu_processor
        self.world_size = self.args.train.num_nodes * self.args.train.num_gpus_per_node
        
        assert self.args.kd.kd_ratio == 1.0, "On-policy KD only supports kd_ratio=1.0."
        
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
        total_steps = self.max_steps
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
        # get eval and save steps
        if self.args.train.eval_steps == -1:
            self.args.train.eval_steps = float("inf")  # Evaluate once per epoch
        if self.args.train.save_steps == -1:
            self.args.train.save_steps = self.num_update_steps_per_epoch  # do not save ckpt
        
        self.global_step = global_step
        
        # Print training configuration and initialize loggers
        self._print_training_config()

        # Create Gloo IPC groups between training ranks and rollout engines (following slime)
        rollout_tp_size = getattr(self.args.rollout, "rollout_tp_size", 1)
        self.student.connect_rollout_engines(self.rollout_group.actors, rollout_tp_size)
        
        self.start_time = time.time()
        status = defaultdict(list)
        num_micro_batches = self.args.train.train_batch_size // self.args.train.micro_train_batch_size
        
        for epoch in range(start_epoch, self.epochs):
            self.current_epoch = epoch
            self.train_dataloader.sampler.set_epoch(epoch)
            
            for prompt_batch in self.train_dataloader:
                rollout_samples = self.rollout(prompt_batch, **self.generate_kwargs)

                if self.args.kd.teacher_enable_sleep:
                    self.teacher.wakeup()
                rollout_samples_for_kd = self.teacher.forward(rollout_samples)
                if self.args.kd.teacher_enable_sleep:
                    self.teacher.sleep()
                
                all_global_batches = []
                for i in range(0, len(rollout_samples), num_micro_batches):
                    global_batch = rollout_samples_for_kd[i : i + num_micro_batches]
                    
                    global_batch_token_num = sum(mb["stu_loss_mask"].sum() for mb in global_batch)
                    avg_micro_batch_token_num = global_batch_token_num / len(global_batch)
                    for mb in global_batch:
                        mb["avg_micro_batch_token_num"] = avg_micro_batch_token_num
                    all_global_batches.append(global_batch)
                
                if self.args.train.train_enable_sleep:
                    self.student.wakeup()
                
                for global_batch in all_global_batches:
                    self.global_step += 1
                    status_list = ray.get(self.student.async_run_distill(global_batch, status))
                    for k in status_list[0].keys():
                        self.log_state[k].append(sum(s[k] for s in status_list) / len(status_list))
                    self.logging()
                
                # Sleep student first to free optimizer GPU memory,
                # model params stay on GPU so update_rollout_weights can still read them.
                if self.args.train.train_enable_sleep:
                    self.student.sleep()
                
                # Wakeup rollout engine weights ONLY (skip kv_cache & cuda_graph to save VRAM),
                # push updated weights, then sleep weights back.
                if self.args.rollout.rollout_enable_sleep:
                    self.rollout_group.wakeup(tags=["weights"])
                self.student.update_rollout_weights()
                if self.args.rollout.rollout_enable_sleep:
                    self.rollout_group.sleep(tags=["weights"])
        
            # save model after each epoch
            self.strategy.log(f"Saving model after epoch {epoch + 1}")
            save_path = os.path.join(self.args.train.save_path, f"epoch_{epoch + 1}")
            ray.get(self.student.async_save_model(save_path))


        total_time = time.time() - self.start_time
        self.strategy.log(f"Training done, totally cost {str(timedelta(seconds=total_time)).split('.')[0]}")

        if self._wandb is not None:
            self._wandb.finish()
            
    def rollout(self, prompt_batch: List[Dict[str, str]], **kwargs) -> List[dict]:
        """Generate samples using rollout engine.

        Args:
            prompt_batch: List of dicts with keys: datasource, stu_prompt, tea_prompt, label
            **kwargs: Additional arguments for generation

        Returns:
            List of rollout sample dicts containing generated samples
        """
        if self.args.rollout.rollout_enable_sleep:
            self.rollout_group.wakeup()

        max_response_length = kwargs.get("max_new_tokens", 1024)
        truncate_length = self.args.data.prompt_max_len + max_response_length

        # Extract prompts and labels from batch
        all_stu_prompts = [item["stu_prompt"] for item in prompt_batch]
        all_tea_prompts = [item["tea_prompt"] for item in prompt_batch]
        all_labels = [item["label"] for item in prompt_batch]
        all_image_paths = [item.get("image_paths") for item in prompt_batch] if self.image_key else None
        
        # Expand prompt list based on the number of samples per prompt
        n_samples_per_prompt = self.args.rollout.n_samples_per_prompt
        all_stu_prompts = sum([[p] * n_samples_per_prompt for p in all_stu_prompts], [])
        all_tea_prompts = sum([[p] * n_samples_per_prompt for p in all_tea_prompts], [])
        all_labels = sum([[label] * n_samples_per_prompt for label in all_labels], [])
        if all_image_paths:
            all_image_paths = sum([[p] * n_samples_per_prompt for p in all_image_paths], [])
        
        all_outputs = self.rollout_group.generate(all_stu_prompts, self.generate_kwargs)

        # Process outputs into rollout samples
        sample_list = [
            self._build_rollout_sample(
                stu_prompt=all_stu_prompts[i],
                tea_prompt=all_tea_prompts[i],
                output=all_outputs[i],
                label=all_labels[i],
                max_response_length=max_response_length,
                truncate_length=truncate_length,
                images=load_images(all_image_paths[i]) if all_image_paths and all_image_paths[i] else None,
                image_paths=all_image_paths[i] if all_image_paths else None,
            )
            for i in range(len(all_outputs))
        ]
        
        logger.info(f"Rollout {len(prompt_batch)} x {n_samples_per_prompt} = {len(sample_list)} samples")
            
        # Print sample for debugging
        sample0 = sample_list[0]["stu_prompts"][0] + sample_list[0]["stu_responses"][0]
        if self.args.rollout.print_rollout_sample:
            print(sample0)
        
        micro_batch_list = self._collate_micro_batches(sample_list, self.args.train.micro_train_batch_size)
        
        if self.args.rollout.rollout_enable_sleep:
            self.rollout_group.sleep()

        return micro_batch_list
    
    def _collate_micro_batches(self, sample_list: List[Dict], batch_size: int) -> List[Dict]:
        """Collate single samples into micro-batches with padding for variable-length tensors."""
        micro_batch_list = []
        for i in range(0, len(sample_list), batch_size):
            batch_samples = sample_list[i : i + batch_size]
            micro_batch = {}
            for key in batch_samples[0]:
                values = [s[key] for s in batch_samples]
                if isinstance(values[0], torch.Tensor):
                    if values[0].dim() == 2:
                        micro_batch[key] = zero_pad_sequences(values, side="right", value=0)
                    else:
                        micro_batch[key] = torch.cat(values, dim=0)
                elif isinstance(values[0], list):
                    micro_batch[key] = sum(values, [])
                elif values[0] is None:
                    micro_batch[key] = None
                else:
                    micro_batch[key] = values
            micro_batch_list.append(micro_batch)
        return micro_batch_list

    def _tokenize_for_model(
        self, 
        prompt: str, 
        response: str, 
        tokenizer: Callable,
        prefix: str,
        truncate_length: int,
        images=None,
    ) -> Dict[str, Any]:
        """
        Tokenize prompt and response for a specific model (student or teacher).
        
        Args:
            prompt: The prompt string (already formatted with chat template)
            response: The response string
            tokenizer: The tokenizer to use
            prefix: Either 'stu' or 'tea'
            truncate_length: Maximum sequence length
            images: List of image paths
            
        Returns:
            Dict with {prefix}_input_ids, {prefix}_attn_mask, {prefix}_loss_mask
        """
        resp_str = response
        if not resp_str.endswith(tokenizer.eos_token):
            resp_str += " " + tokenizer.eos_token

        processor = self.stu_processor if prefix == "stu" else self.tea_processor
        if images and processor:
            prompt_inputs = processor(text=prompt, images=images, return_tensors="pt", padding=False)
            prompt_len = prompt_inputs["input_ids"].squeeze().shape[0]
            full_text = prompt + resp_str
            full_inputs = processor(text=full_text, images=images, return_tensors="pt", padding=False)
            full_len = full_inputs["input_ids"].squeeze().shape[0]
            input_ids = full_inputs["input_ids"].squeeze()[:truncate_length]
            attn_mask = full_inputs["attention_mask"].squeeze()[:truncate_length]
            loss_mask = torch.tensor([False] * prompt_len + [True] * (full_len - prompt_len))[:truncate_length].roll(shifts=-1)
            result = {
                f"{prefix}_input_ids": input_ids,
                f"{prefix}_attn_mask": attn_mask,
                f"{prefix}_loss_mask": loss_mask,
            }
            for key in full_inputs:
                if key not in ("input_ids", "attention_mask"):
                    result[f"{prefix}_{key}"] = full_inputs[key].squeeze()
            return result

        prompt_tokens = tokenizer(prompt, add_special_tokens=False)
        prompt_len = len(prompt_tokens["input_ids"])
        resp_tokens = tokenizer(resp_str, add_special_tokens=False)
        resp_len = len(resp_tokens["input_ids"])
        input_ids = prompt_tokens["input_ids"] + resp_tokens["input_ids"]
        attn_mask = prompt_tokens["attention_mask"] + resp_tokens["attention_mask"]
        loss_mask = [False] * prompt_len + [True] * resp_len
        input_ids = torch.tensor(input_ids[:truncate_length])
        attn_mask = torch.tensor(attn_mask[:truncate_length])
        loss_mask = torch.tensor(loss_mask[:truncate_length]).roll(shifts=-1)
        
        return {
            f"{prefix}_input_ids": input_ids,
            f"{prefix}_attn_mask": attn_mask,
            f"{prefix}_loss_mask": loss_mask,
        }

    def _build_rollout_sample(
        self,
        stu_prompt: str,
        tea_prompt: str,
        output,
        label: str,
        max_response_length: int,
        truncate_length: int,
        images=None,
        image_paths=None,
    ) -> Dict[str, Any]:
        """
        Build a single rollout sample with both student and teacher tokenizations.
        
        Args:
            stu_prompt: Student prompt string (formatted with student's chat template)
            tea_prompt: Teacher prompt string (formatted with teacher's chat template)
            output: rollout output object
            label: Label string
            max_response_length: Maximum response length
            truncate_length: Length to truncate at
            
        Returns:
            Dict containing all sample fields
        """
        # Decode response using student tokenizer
        response_ids = output["output_ids"]
        response_text = output["text"]
        
        stu_tokens = self._tokenize_for_model(
            stu_prompt, response_text, self.student_tokenizer, "stu", truncate_length, images=images
        )
        
        if not self.is_same_tokenizer or tea_prompt != stu_prompt:
            tea_tokens = self._tokenize_for_model(
                tea_prompt, response_text, self.teacher_tokenizer, "tea", truncate_length, images=images
            )
        else:
            # Same tokenizer and same prompt, just copy all fields including loss_mask
            tea_tokens = {
                "tea_input_ids": stu_tokens["stu_input_ids"].clone(),
                "tea_attn_mask": stu_tokens["stu_attn_mask"].clone(),
                "tea_loss_mask": stu_tokens["stu_loss_mask"].clone(),
            }
        
        response_length = len(response_ids)
        
        # Calculate rollout log probs if needed
        rollout_log_probs = None
        total_length = stu_tokens["stu_attn_mask"].float().sum()
        is_clipped = response_length >= max_response_length
        
        sample = {
            "stu_input_ids": stu_tokens["stu_input_ids"].unsqueeze(0),
            "stu_attn_mask": stu_tokens["stu_attn_mask"].unsqueeze(0),
            "stu_loss_mask": stu_tokens["stu_loss_mask"].unsqueeze(0),
            "rollout_log_probs": rollout_log_probs.unsqueeze(0) if rollout_log_probs is not None else None,
            "tea_input_ids": tea_tokens["tea_input_ids"].unsqueeze(0),
            "tea_attn_mask": tea_tokens["tea_attn_mask"].unsqueeze(0),
            "tea_loss_mask": tea_tokens["tea_loss_mask"].unsqueeze(0),
            "stu_prompts": [stu_prompt],
            "stu_responses": [response_text],
            "tea_prompts": [tea_prompt],
            "labels": [label],
            "response_length": torch.FloatTensor([response_length]),
            "total_length": torch.FloatTensor([total_length]),
            "response_clip_ratio": torch.FloatTensor([is_clipped]),
        }
        for k, v in stu_tokens.items():
            if k not in sample:
                sample[k] = v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
        if image_paths:
            sample["image_paths"] = [image_paths]
        return sample
            
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
            # if dist.get_rank() == 0:
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

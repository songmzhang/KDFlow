import math
import os
import socket
import logging
import json
from abc import ABC
from typing import Dict, List, Optional, Union
from collections import defaultdict

import numpy as np
import ray
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.trainer import get_scheduler
from transformers import AutoConfig
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from kdflow.models import DistillModel
from kdflow.utils.distributed_util import stateless_init_process_group, torch_dist_barrier_and_cuda_sync
from kdflow.utils.logging_utils import init_logger
from kdflow.ray.utils import ray_noset_visible_devices
from kdflow.algorithms import ALGO_DICT


logger = init_logger(__name__)

@ray.remote
class StudentRayActor:
    """
    StudentRayActor: Responsible for student model training with FSDP/DeepSpeed.
    
    This actor only handles the student model training. Teacher model inference
    should be handled separately by TeacherRolloutManager using SGLang.
    """
    
    def __init__(self, world_size, rank, master_addr, master_port):
        self._world_size = world_size
        self._rank = rank
        self._master_addr = master_addr if master_addr else self._get_current_node_ip()
        self._master_port = master_port if master_port else self._get_free_port()
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # NOTE: Ray will automatically set the *_VISIBLE_DEVICES
        # environment variable for each actor, unless
        # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set, so
        # set local rank to 0 when the flag is not applicable.
        os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0]) if ray_noset_visible_devices() else "0"

    @staticmethod
    def _get_current_node_ip():
        address = ray._private.services.get_node_ip_address()
        # strip ipv6 address
        return address.strip("[]")

    @staticmethod
    def _get_free_port():
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_master_addr_port(self):
        return self._master_addr, self._master_port

    def get_node_ip(self) -> str:
        return self._get_current_node_ip()

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    def init_model_from_pretrained(
        self, 
        strategy, 
        max_steps=None, 
        teacher_tokenizer=None, 
        tokenizer_info=None,
    ):
        """
        Initialize student model for training.
        
        Args:
            strategy: Training strategy (FSDP)
            max_steps: Maximum training steps for scheduler
        """
        self.args = strategy.args
        self.max_steps = max_steps
        self.strategy = strategy

        strategy.setup_distributed()

        # Only initialize student model
        self.student = DistillModel(strategy)
        self.student = strategy.prepare(self.student)
        
        # configure optimizer
        self.optim = strategy.create_optimizer(
            self.student, 
            lr=self.args.train.learning_rate, 
            betas=self.args.train.adam_betas, 
            weight_decay=self.args.train.weight_decay
        )

        self.scheduler = get_scheduler(
            self.args.train.lr_scheduler,
            self.optim,
            num_warmup_steps=math.ceil(max_steps * self.args.train.lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": self.args.train.min_lr},
        )

        if self.args.train.gradient_checkpointing:
            self.student.gradient_checkpointing_enable()
            
        # load teacher lm_head for later logits calculation in the algorithm (logits = lm_head(hidden))
        self.teacher_lm_head = self.load_only_lm_head(strategy.args.model.teacher_name_or_path)

        # Initialize KD algorithm without teacher model
        # Teacher hiddens will be passed in each micro_batch during training
        ALGO_CLS = ALGO_DICT[self.args.kd.kd_algorithm]
        self.kd_algorithm = ALGO_CLS(
            strategy=strategy, 
            student_model=self.student,
            teacher_lm_head=self.teacher_lm_head,
            student_tokenizer=self.student.tokenizer,
            teacher_tokenizer=teacher_tokenizer,  # Teacher tokenizer not needed here
            tokenizer_info=tokenizer_info,
        )

        # Register projector parameters into optimizer with separate learning rate
        if hasattr(self.kd_algorithm, 'get_projector_params'):
            projector_params = self.kd_algorithm.get_projector_params()
            if projector_params:
                projector_lr = self.args.kd.dskd_projector_lr
                self.optim.add_param_group({
                    "params": projector_params,
                    "lr": projector_lr,
                    "weight_decay": 0.0,
                })
                # Recreate scheduler to account for the new param group
                self.scheduler = get_scheduler(
                    self.args.train.lr_scheduler,
                    self.optim,
                    num_warmup_steps=math.ceil(max_steps * self.args.train.lr_warmup_ratio),
                    num_training_steps=max_steps,
                    scheduler_specific_kwargs={"min_lr": self.args.train.min_lr},
                )
                strategy.print(f"Registered {len(projector_params)} projector params into optimizer with lr={projector_lr}")

        # load checkpoint
        self.checkpoint_states = {}
        ckpt_path = self.args.train.ckpt_path
        if os.path.exists(ckpt_path):
            strategy.print(f"Loading the checkpoint: {ckpt_path}")
            _, states = strategy.load_ckpt(self.student.model, ckpt_path)
            self.checkpoint_states["global_step"] = states["global_step"]
            self.checkpoint_states["epoch"] = states["epoch"]
            self.checkpoint_states["data_loader_state_dict"] = states["data_loader_state_dict"]

        # initial offload
        if self.args.train.train_enable_sleep:
            self.sleep()

        torch_dist_barrier_and_cuda_sync()
        
    def load_only_lm_head(self, model_name_or_path, device="cuda", dtype=torch.bfloat16):
        """Load only lm_head weights from checkpoint. Supports tied weights and sharded models."""
        logger.info(f"Loading only lm_head from {model_name_or_path}...")
        config = AutoConfig.from_pretrained(model_name_or_path)
        is_local = os.path.exists(model_name_or_path)
        hidden_size = config.hidden_size if not hasattr(config, "text_config") else config.text_config.hidden_size
        vocab_size = config.vocab_size if not hasattr(config, "text_config") else config.text_config.vocab_size

        def resolve_file(filename):
            if is_local:
                return os.path.join(model_name_or_path, filename)
            return hf_hub_download(repo_id=model_name_or_path, filename=filename)

        def try_load_index():
            for index_name, safetensors in [
                ("model.safetensors.index.json", True),
                ("pytorch_model.bin.index.json", False),
            ]:
                try:
                    with open(resolve_file(index_name), "r") as f:
                        return json.load(f)["weight_map"], safetensors
                except Exception:
                    continue
            return None, None

        def resolve_target_key(weight_map):
            if weight_map and "lm_head.weight" in weight_map:
                return "lm_head.weight"
            if getattr(config, "tie_word_embeddings", True):
                if weight_map is None or "model.embed_tokens.weight" in weight_map:
                    return "model.embed_tokens.weight"
            raise ValueError("Could not find lm_head.weight or model.embed_tokens.weight in checkpoint.")

        weight_map, use_safetensors = try_load_index()
        target_key = resolve_target_key(weight_map)

        if weight_map:
            checkpoint_file = resolve_file(weight_map[target_key])
        else:
            for name, safetensors in [("model.safetensors", True), ("pytorch_model.bin", False)]:
                try:
                    checkpoint_file = resolve_file(name)
                    use_safetensors = safetensors
                    break
                except Exception:
                    continue
            else:
                raise FileNotFoundError(f"No checkpoint file found in {model_name_or_path}")

        state_dict = load_file(checkpoint_file) if use_safetensors else torch.load(checkpoint_file, map_location="cpu")

        weight_key = "lm_head.weight" if "lm_head.weight" in state_dict else "model.embed_tokens.weight"
        if weight_key not in state_dict:
            raise ValueError(f"Key '{weight_key}' not found. Available: {list(state_dict.keys())[:10]}...")
        weight = state_dict[weight_key]

        assert weight.shape == (vocab_size, hidden_size), \
            f"Shape mismatch: expected ({vocab_size}, {hidden_size}), got {weight.shape}"

        has_bias = "lm_head.bias" in state_dict
        lm_head = nn.Linear(hidden_size, vocab_size, bias=has_bias)
        lm_head.weight = nn.Parameter(weight.to(dtype=dtype))
        if has_bias:
            lm_head.bias = nn.Parameter(state_dict["lm_head.bias"].to(dtype=dtype))

        lm_head = lm_head.to(device).eval()
        lm_head.requires_grad_(False)
        logger.info(f"Loaded lm_head ({weight_key}), shape: {lm_head.weight.shape}")
        return lm_head
        
    def fit(self, train_data, prev_status=None):
        """
        Train student model with the given data.
        
        Args:
            train_data: List of training samples
            prev_status: Previous training status dict for accumulation
            
        Returns:
            Averaged training status dict
        """
        if prev_status is None:
            prev_status = defaultdict(list)
            
        self.student.train()
        
        dataloader = train_data
            
        device = torch.cuda.current_device()

        for batch in dataloader:
            micro_batch = {
                k: torch.from_numpy(v).to(device, non_blocking=True) if isinstance(v, np.ndarray)
                else v.to(device) if isinstance(v, torch.Tensor)
                else v
                for k, v in batch.items()
            }
            
            # --- Student Forward ---
            # Teacher outputs are already in micro_batch["teacher_hiddens"]
            loss_info = self.kd_algorithm.training_step(micro_batch)
            
            # --- Backward ---
            loss = loss_info["loss"]
            self.strategy.backward(loss, self.student, self.optim)
            
            # --- Clip projector gradients (not covered by FSDP model's clip_grad_norm_) ---
            if hasattr(self.kd_algorithm, 'get_projector_params'):
                projector_params = self.kd_algorithm.get_projector_params()
                if projector_params:
                    torch.nn.utils.clip_grad_norm_(projector_params, max_norm=self.args.train.max_norm)

            # --- Optimizer Step ---
            self.strategy.optimizer_step(self.optim, self.student, self.scheduler)

            last_micro_batch = micro_batch
            del micro_batch

        status = {}
        for key in loss_info:
            status[key] = loss_info[key].item()

        status["lr"] = self.scheduler.get_last_lr()[0]
        if "response_length" in last_micro_batch:
            status["gen_len"] = last_micro_batch["response_length"].mean().item()
        if "total_length" in last_micro_batch:
            status["tot_len"] = last_micro_batch["total_length"].mean().item()
        del last_micro_batch

        if hasattr(self.student, "get_global_grad_norm") and self.student.get_global_grad_norm() is not None:
            status["grad_norm"] = self.student.get_global_grad_norm()
        elif hasattr(self.student, "clip_grad_norm_"):
            status["grad_norm"] = self.student.clip_grad_norm_(max_norm=self.args.train.max_norm).item()

        for key in status:
            prev_status[key].append(status[key])

        for key in prev_status:
            prev_status[key] = sum(prev_status[key]) / len(prev_status[key])
            prev_status[key] = self.strategy.all_reduce(prev_status[key], op="mean")
        
        self.empty_cache()
        
        return prev_status

    def save_model(self, save_path=None):
        """Save model checkpoint after fitting on only rank0."""
        if save_path is None:
            save_path = self.args.train.save_path
        self.strategy.save_model(self.student, self.student.tokenizer, save_path)

    def get_checkpoint_states(self):
        return self.checkpoint_states

    def wakeup(self):
        """Reload optimizer states from CPU to GPU."""
        self.strategy.reload_optim_states(self.optim)

    def sleep(self):
        """Offload optimizer states from GPU to CPU to save memory."""
        self.strategy.offload_optim_states(self.optim, empty_cache=True)

    def save_checkpoint(self, tag, client_states):
        self.strategy.save_ckpt(
            self.student.model,
            os.path.join(self.args.train.ckpt_path, "_actor"),
            tag,
            self.args.train.max_ckpt_num,
            self.args.train.max_ckpt_mem,
            client_states,
        )
        if self.save_hf_ckpt:
            save_path = os.path.join(self.args.train.ckpt_path, f"{tag}_hf")
            self.strategy.save_model(self.student, self.student.tokenizer, save_path)
        # wait
        torch_dist_barrier_and_cuda_sync()
        
    def connect_rollout_engines(self, rollout_engines, rollout_tp_size=1):
        """Create Gloo IPC groups for weight sync (following slime)."""
        import torch.distributed as dist
        self._rollout_engines = rollout_engines
        num_gpus_per_engine = rollout_tp_size
        for i, engine in enumerate(self._rollout_engines):
            start_rank = i * num_gpus_per_engine
            end_rank = (i + 1) * num_gpus_per_engine
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if dist.get_rank() in group_ranks:
                self._ipc_gather_src = start_rank
                self._ipc_gather_group = new_group
                self._ipc_engine = engine
                self._tp_rank = dist.get_rank() - start_rank

    def update_rollout_weights(self):
        """Stream FSDP weights to rollout engines via Gloo gather + CUDA IPC."""
        model = self.student.model.module if self.args.train.backend == "deepspeed" else self.student.model
        self.strategy.update_rollout_weights_from_tensor(
            model,
            engine=self._ipc_engine,
            gather_src=self._ipc_gather_src,
            gather_group=self._ipc_gather_group,
        )

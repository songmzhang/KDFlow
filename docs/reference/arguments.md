# Arguments

All KDFlow CLI flags are parsed by a single `HfArgumentParser` over eight
dataclasses (defined under `kdflow/arguments/`). Each scenario uses the
following argument groups:

| Scenario      | Dataclasses included                                                                                |
|---------------|------------------------------------------------------------------------------------------------------|
| `sft`         | Model + Training + Checkpoint + FSDP + Data + Logging |
| `off_policy_kd` | Model + Training + Checkpoint + FSDP + Distillation + Data + Logging |
| `on_policy_kd`  | Model + Training + Checkpoint + FSDP + Distillation + Data + Logging + Rollout |

The tables below mirror the canonical reference in the project README.

---

## Model Arguments

Defined in `kdflow/arguments/model_args.py`.

| Argument                     | Default              | Description                                            |
|------------------------------|----------------------|--------------------------------------------------------|
| `--student_name_or_path`     | `None`               | Student model name or path                             |
| `--teacher_name_or_path`     | `None`               | Teacher model name or path                             |
| `--attn_implementation`      | `flash_attention_2`  | Attention implementation                               |
| `--use_liger_kernel`         | `False`              | Use Liger Kernel for the student model                 |
| `--lora_rank`                | `0`                  | LoRA rank (0 disables LoRA)                            |
| `--lora_alpha`               | `16`                 | LoRA alpha                                             |
| `--target_modules`           | `all-linear`         | LoRA target modules                                    |
| `--lora_dropout`             | `0.0`                | LoRA dropout                                           |
| `--ring_attn_size`           | `1`                  | Ring attention group size for context parallelism      |
| `--enable_thinking`          | `False`              | Enable "thinking mode" in the chat template            |
| `--disable_fast_tokenizer`   | `False`              | Disable HF fast tokenizer                              |

---

## Training Arguments

Defined in `kdflow/arguments/training_args.py`.

| Argument                       | Default                  | Description                                                       |
|--------------------------------|--------------------------|-------------------------------------------------------------------|
| `--num_nodes`                  | `1`                      | Number of training nodes                                          |
| `--num_gpus_per_node`          | `8`                      | GPUs per node                                                     |
| `--num_epochs`                 | `1`                      | Number of training epochs                                         |
| `--train_batch_size`           | `128`                    | Global training batch size                                        |
| `--micro_train_batch_size`     | `1`                      | Per-GPU micro batch size                                          |
| `--learning_rate`              | `1e-6`                   | Learning rate                                                     |
| `--lr_scheduler`               | `cosine_with_min_lr`     | LR scheduler type                                                 |
| `--lr_warmup_ratio`            | `0.05`                   | Warmup ratio                                                      |
| `--min_lr`                     | `1e-8`                   | Minimum learning rate                                             |
| `--max_norm`                   | `1.0`                    | Gradient clipping max norm                                        |
| `--weight_decay`               | `0.0`                    | Weight decay                                                      |
| `--adam_betas`                 | `(0.9, 0.98)`            | Adam optimizer betas                                              |
| `--backend`                    | `fsdp2`                  | Training backend                                                  |
| `--gradient_checkpointing`     | `False`                  | Enable gradient checkpointing                                     |
| `--enable_sleep`               | `False`                  | Enable sleep mode for student / teacher / rollout                 |
| `--eval_steps`                 | `-1`                     | Evaluation interval in global steps; set a positive value to enable periodic evaluation |
| `--seed`                       | `42`                     | Random seed                                                       |
| `--bf16`                       | `False`                  | Enable bfloat16 training                                          |
| `--use_dynamic_bsz`            | `False`                  | Enable dynamic batch size based on token count per GPU            |
| `--max_token_len_per_gpu`      | `0`                      | Max total tokens per micro-batch when `use_dynamic_bsz=True`      |
| `--chunked_loss_size`          | `None`                   | Token chunk size for chunked loss computation. If not `None`, logits and the corresponding loss will be computed chunk by chunk to reduce GPU memory. See [Chunked Loss](../concepts/losses.md#chunked-loss-memory-efficient-computation). |

---

## Checkpoint Arguments

Defined in `kdflow/arguments/ckpt_args.py`. Used by SFT, off-policy KD, and on-policy KD.

| Argument | Default | Description |
|----------|---------|-------------|
| `--save_path` | `./ckpt/` | Root directory for checkpoints |
| `--save_steps` | `-1` | Save every N steps; -1 disables periodic saving |
| `--save_training_state` | `False` | Include training state for resuming |
| `--resume_from` | `None` | Checkpoint directory to resume from; automatically enables `resume_training` |
| `--resume_training` | `False` | Resume from `resume_from` or the latest checkpoint in `save_path` |
| `--max_ckpt_num` | `-1` | Maximum checkpoints to retain for this run; -1 keeps all |

Checkpoints are also saved at the end of every epoch. Steps count optimizer
updates for SFT and off-policy KD, and rollout iterations for on-policy KD.
For off-policy KD, positive `save_steps` and `eval_steps` must be multiples of
`teacher_forward_n_batches`.
Epochs are numbered starting from 1:

```text
$save_path/
├── epoch_1_global_step_100/
│   ├── model/
│   └── training_state/          # with --save_training_state True
├── epoch_1_end_global_step_200/
│   ├── model/
│   └── training_state/
└── latest                      # text file naming the latest resumable checkpoint
```

`model/` contains the Hugging Face model or LoRA adapter. Use this directory
for inference or as the starting model for a new training run.
`training_state/` contains per-rank optimizer, scheduler, RNG, dataloader,
progress, and checkpoint retention state, plus EMA, teacher, and projector
weights when used by KD.

When training finishes successfully, KDFlow also exports the final Hugging Face
model (or LoRA adapter) and tokenizer/processor directly to `save_path`.
This export does not modify checkpoint subdirectories, `latest`, or the retention list.

To enable resuming, add `--save_training_state True` when saving.
By default, training starts from scratch and ignores `latest`.
Set `--resume_training True` to resume from `latest` in `save_path`;
if no latest checkpoint exists, training raises an error.
Alternatively, specify `--resume_from` to select a checkpoint and automatically
enable `resume_training`. Pass the checkpoint directory, not its `model/`
subdirectory. The checkpoint must be directly inside `save_path`.
Resuming requires the same GPU count, `fsdp_size`, and
`ring_attn_size` as the saved run.
New saves always replace checkpoints with the same name after writing the new
contents. A fresh run starts with an empty retention list; resuming restores
the saved list. Other historical checkpoints are left alone.

Add these flags to your training command to save resumable checkpoints every
100 steps and retain the latest two:

```bash
--save_path ./ckpt/my_run \
--save_training_state True \
--save_steps 100 \
--max_ckpt_num 2
```

To resume the latest checkpoint, rerun the command with:

```bash
--resume_training True
```

To resume a specific checkpoint instead, add:

```bash
--resume_from ./ckpt/my_run/epoch_1_global_step_100
```

`resume_from` automatically enables `resume_training`.

---

## FSDP Arguments

Defined in `kdflow/arguments/fsdp_args.py`.

| Argument        | Default | Description                                          |
|-----------------|---------|------------------------------------------------------|
| `--fsdp_size`   | `-1`    | FSDP shard size for HSDP (-1 = full sharding)        |
| `--cpu_offload` | `False` | Offload Adam optimizer states to CPU                 |

---

## Distillation Arguments

Defined in `kdflow/arguments/distillation_args.py`. Used by the KD entry points
(`train_kd_off_policy`, `train_kd_on_policy`).

| Argument                          | Default       | Description                                                                  |
|-----------------------------------|---------------|------------------------------------------------------------------------------|
| `--kd_ratio`                      | `0.5`         | KD weight: `loss = (1 - kd_ratio)*CE + kd_ratio*KD`                          |
| `--kd_temperature`                | `1.0`         | Softmax temperature for KD                                                   |
| `--kd_algorithm`                  | `vanilla_kd`  | KD algorithm name (see [Algorithms](../concepts/algorithms.md))              |
| `--kd_loss_fn`                    | `kl`          | Divergence (see [Losses](../concepts/losses.md))                             |
| `--teacher_tp_size`               | `8`           | Teacher tensor parallel size                                                 |
| `--teacher_ep_size`               | `1`           | Teacher expert parallel size (MoE)                                           |
| `--teacher_pp_size`               | `1`           | Teacher pipeline parallel size                                               |
| `--teacher_dp_size`               | `1`           | Teacher data parallel size                                                   |
| `--teacher_forward_n_batches`     | `1`           | Teacher forward N batches at once                                            |
| `--teacher_mem_fraction_static`   | `0.4`         | SGLang static memory fraction for teacher                                    |
| `--teacher_offload_tags`          | `all`         | Offload tags for SGLang                                                      |
| `--teacher_quantization`          | `None`        | Teacher quantisation (e.g. `awq`, `fp8`)                                     |
| `--multi_teacher_config`          | `None`        | JSON file mapping teacher routing keys to teacher model paths                |
| `--dskd_token_align`              | `eta`         | DSKD token alignment (`eta` / `cma`)                                         |
| `--dskd_topk_vocab`               | `-1`          | Top-k vocab for DSKD projector init (-1 = all)                               |
| `--dskd_projector_lr`             | `1e-4`        | LR for DSKD projectors                                                       |
| `--jsd_beta`                      | `0.5`         | β for Jensen-Shannon Divergence                                              |
| `--skew_lambda`                   | `0.1`         | λ for skewed KL / RKL                                                        |
| `--adaptive_alpha`                | `0.5`         | α for Adaptive KL                                                            |
| `--hrl_topk`                      | `5`           | Top-k for Hierarchical Ranking Loss                                          |
| `--teacher_update_freq`           | `1`           | Teacher weight update frequency (in global steps) for on-policy self-KD     |
| `--use_ema_teacher`               | `False`       | Sync an EMA copy of the student to the teacher instead of the live weights  |
| `--teacher_ema_decay`             | `0.999`       | EMA decay α used when `--use_ema_teacher True`                              |

---

## Rollout Arguments (on-policy)

Defined in `kdflow/arguments/rollout_args.py`. Effective only with
`train_kd_on_policy`.

| Argument                          | Default | Description                                            |
|-----------------------------------|---------|--------------------------------------------------------|
| `--rollout_num_engines`           | `0`     | Number of SGLang rollout engines (0 = off-policy)      |
| `--rollout_tp_size`               | `1`     | TP size per rollout engine                             |
| `--rollout_batch_size`            | `32`    | Prompts per rollout iteration                          |
| `--n_samples_per_prompt`          | `1`     | Responses per prompt                                   |
| `--generate_max_len`              | `2048`  | Max generation length                                  |
| `--temperature`                   | `1.0`   | Sampling temperature                                   |
| `--top_p`                         | `1.0`   | Top-p sampling                                         |
| `--rollout_mem_fraction_static`   | `0.6`   | GPU memory fraction per rollout engine                 |
| `--print_rollout_sample`          | `False` | Print one rollout sample after each iteration          |

---

## Data Arguments

Defined in `kdflow/arguments/data_args.py`.

| Argument                      | Default      | Description                                                                |
|-------------------------------|--------------|----------------------------------------------------------------------------|
| `--train_dataset_path`        | `None`       | Training dataset path                                                      |
| `--train_dataset_probs`       | `None`       | Sampling probabilities for multiple datasets                               |
| `--train_split`               | `train`      | Train split name                                                           |
| `--eval_dataset_path`         | `None`       | Evaluation dataset path (see [Evaluation](../user_guide/evaluation.md))    |
| `--eval_split`                | `eval`       | Eval split name                                                            |
| `--custom_eval_fn`            | `None`       | On-policy Python file defining `eval_fn(predictions, labels)`              |
| `--input_key`                 | `messages`   | Dataset input key                                                          |
| `--output_key`                | `None`       | Dataset output key                                                         |
| `--image_key`                 | `None`       | Image key for multimodal datasets                                          |
| `--teacher_input_key`         | `None`       | Teacher prompt key (for self / context distillation)                       |
| `--teacher_routing_key`       | `teacher_routing_key` | Dataset field used to route each sample to a teacher in multi-teacher KD |
| `--label_key`                 | `None`       | Label key                                                                  |
| `--apply_chat_template`       | `True`       | Apply tokenizer chat template                                              |
| `--max_len`                   | `4096`       | Max sequence length                                                        |
| `--prompt_max_len`            | `2048`       | Max prompt length                                                          |
| `--max_samples`               | `1e8`        | Max number of samples to load                                              |
| `--packing_samples`           | `False`      | Pack sequences for efficiency                                              |
| `--preprocess_num_workers`    | `8`          | Workers for data preprocessing                                             |

!!! note "Packing with linear-attention models"
    Linear-attention models such as **Qwen3.5** and **Qwen3-Next** require
    `transformers>=5.9.0` when `--packing_samples True` is enabled. Earlier
    Transformers versions do not propagate the packed-sequence boundary
    metadata required by their linear-attention and causal-convolution layers.

    KDFlow automatically disables packing with a warning when this requirement
    is not met. Linear-attention packing currently supports text-only training
    with `--ring_attn_size 1`.

---

## Logging Arguments

Defined in `kdflow/arguments/logging_args.py`.

| Argument             | Default   | Description                                            |
|----------------------|-----------|--------------------------------------------------------|
| `--logging_steps`    | `10`      | Log every N steps                                      |
| `--use_wandb`        | `False`   | Enable W&B logging                                     |
| `--sync_swanlab`     | `False`   | Sync W&B metrics to SwanLab; requires `--use_wandb True` and the `swanlab` package |
| `--wandb_org`        | `None`    | W&B organization                                       |
| `--wandb_project`    | `None`    | W&B project                                            |
| `--wandb_group`      | `None`    | W&B group                                              |
| `--wandb_run_name`   | `None`    | W&B run name                                           |
| `--wandb_mode`       | `online`  | W&B mode (`online` / `offline` / `disabled`)           |
| `--wandb_dir`        | `None`    | Directory for W&B offline logs                         |

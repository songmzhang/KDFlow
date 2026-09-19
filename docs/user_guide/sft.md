# Supervised Fine-Tuning (SFT)

SFT can be also regarded as KDFlow's **black-box KD** mode — the teacher's knowledge is encoded
in the dataset (e.g. distilled responses). It uses the same dataset / packing /
FSDP2 / LoRA / Liger Kernel infrastructure as the KD trainers, but **without
Ray** and without a teacher actor group. The training loop is started with a standard `torchrun`.

## Entry point

```bash
torchrun --nproc_per_node=8 -m kdflow.cli.train_sft [args...]
```

## Example

```bash
bash examples/sft/run_qwen3_4b.sh
```

which runs:

```bash
torchrun --nproc_per_node=8 -m kdflow.cli.train_sft \
    --num_nodes 1  --num_gpus_per_node 8 \
    --backend fsdp2 \
    --train_batch_size 128  --micro_train_batch_size 2 \
    --learning_rate 2e-5  --lr_warmup_ratio 0.05  --num_epochs 1 \
    --bf16 True  --gradient_checkpointing True \
    --student_name_or_path Qwen3/Qwen3-4B \
    --train_dataset_path OpenLeecher/lmsys_chat_1m_clean \
    --max_len 4096  --input_key conversations \
    --apply_chat_template True  --packing_samples True \
    --preprocess_num_workers 32 \
    --use_wandb True  --wandb_project KDFlow
```

## When to use SFT

- You have a **pre-distilled dataset** of teacher answers and want to reproduce
  it with a smaller student.
- You need a **starting checkpoint** before running on-policy KD.
- You want to fine-tune a model with the same FSDP2 / LoRA / packing
  infrastructure used by the KD trainers, without spinning up Ray and SGLang.

## Argument groups

The SFT entry point uses the same dataclasses as KD, minus the
distillation/rollout-specific knobs:

- [Model](../reference/arguments.md#model-arguments)
- [Training](../reference/arguments.md#training-arguments)
- [Checkpoint](../reference/arguments.md#checkpoint-arguments)
- [FSDP](../reference/arguments.md#fsdp-arguments)
- [Data](../reference/arguments.md#data-arguments)
- [Logging](../reference/arguments.md#logging-arguments)

## See also

- [Evaluation](evaluation.md) — configure validation loss during SFT.
- [Off-Policy KD](off_policy_kd.md) — once you have an SFT checkpoint, plug it
  in as the student.
- [Architecture](../concepts/architecture.md) — how the trainers and actor
  groups are wired together.

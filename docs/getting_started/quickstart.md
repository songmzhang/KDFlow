# Quickstart

This page walks you through your **first distillation run** — distilling a 30B
teacher (`Qwen3-30B-A3B`) into a 4B student (`Qwen3-4B`) on a single 8-GPU node.
For the full set of supported scenarios, see the [User Guide](../user_guide/off_policy_kd.md).

## 1. Start a Ray cluster

All entry points (`train_sft`, `train_kd_off_policy`, `train_kd_on_policy`)
drive a controller that orchestrates Ray actors — so Ray must be running before
you launch the script:

```bash
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8
```

For multi-node training, start the head on node 0 and `ray start --address ...`
on workers; KDFlow auto-discovers GPUs from the cluster.

## 2. Pick a recipe

KDFlow ships ready-to-run shell scripts in
[`examples/`](https://github.com/songmzhang/KDFlow/tree/main/examples):

| Scenario                                | Example script                                                                                |
|-----------------------------------------|-----------------------------------------------------------------------------------------------|
| Off-policy KD (LLM)                     | `examples/off_policy_kd/run_qwen3_30b_a3b_to_4b.sh`                                           |
| Off-policy KD (VLM)                     | `examples/off_policy_kd/run_qwen3_vl_30b_a3b_to_4b.sh`                                        |
| On-policy KD (LLM)                      | `examples/on_policy_kd/run_qwen3_30b_a3b_to_4b.sh`                                            |
| On-policy KD (VLM)                      | `examples/on_policy_kd/run_qwen3_vl_30b_a3b_to_4b.sh`                                         |
| Multi-teacher KD (off-policy)           | `examples/multi_teacher_distillation/run_multi_teacher_off_policy_distillation.sh`             |
| Multi-teacher KD (on-policy)            | `examples/multi_teacher_distillation/run_multi_teacher_on_policy_distillation.sh`              |
| Cross-tokenizer KD (off-policy, simple) | `examples/cross_tokenizer_kd/run_qwen3_30b_a3b_to_llama3_2_3b_offpolicy_simple_ctkd.sh`       |
| Cross-tokenizer KD (off-policy, DSKD)   | `examples/cross_tokenizer_kd/run_qwen3_30b_a3b_to_llama3_2_3b_offpolicy.sh`                   |
| Cross-tokenizer KD (on-policy, simple)  | `examples/cross_tokenizer_kd/run_qwen3_30b_a3b_to_llama3_2_3b_onpolicy_simple_ctkd.sh`        |
| Cross-tokenizer KD (on-policy, DSKD)    | `examples/cross_tokenizer_kd/run_qwen3_30b_a3b_to_llama3_2_3b_onpolicy.sh`                    |
| SFT                                     | `examples/sft/run_qwen3_4b.sh`                                                                |

## 3. Run off-policy KD

```bash
bash examples/off_policy_kd/run_qwen3_30b_a3b_to_4b.sh
```

The script ultimately runs:

```bash
python -m kdflow.cli.train_kd_off_policy \
    --num_nodes 1 --num_gpus_per_node 8 --backend fsdp2 \
    --train_batch_size 128 --micro_train_batch_size 2 \
    --learning_rate 2e-5 --lr_warmup_ratio 0.05 --num_epochs 1 \
    --bf16 True --gradient_checkpointing True \
    --student_name_or_path Qwen3/Qwen3-4B \
    --teacher_name_or_path Qwen3/Qwen3-30B-A3B \
    --train_dataset_path OpenLeecher/lmsys_chat_1m_clean \
    --max_len 4096 --input_key conversations --packing_samples True \
    --kd_ratio 0.5 --kd_loss_fn kl --kd_algorithm vanilla_kd \
    --teacher_dp_size 2 --teacher_tp_size 4 \
    --teacher_mem_fraction_static 0.4 \
    --use_wandb True --wandb_project KDFlow
```

What happens under the hood:

1. The controller calls `init_args("off_policy_kd")` to parse all argument
   dataclasses (`Model`, `Training`, `FSDP`, `Distillation`, `Data`, `Logging`,
   `Rollout`).
2. `ray.init(...)` connects to the running cluster, and a placement group with
   `num_nodes × num_gpus_per_node` GPU bundles is created.
3. A `TeacherActorGroup` (SGLang) and a `StudentActorGroup` (FSDP2) are spawned
   on the same GPUs (sleep/wakeup keeps only one resident at a time).
4. `OffPolicyKDTrainer.fit()` iterates over the dataset:
   teacher prefill → hidden-state transfer via shared memory → student
   forward + KD loss + optimizer step.

## 4. Run on-policy KD

The on-policy recipe additionally spawns a `RolloutActorGroup`:

```bash
bash examples/on_policy_kd/run_qwen3_30b_a3b_to_4b.sh
```

Key extra arguments compared to off-policy:

```bash
--rollout_num_engines 8 \
--rollout_tp_size 1 \
--rollout_batch_size 1024 \
--rollout_mem_fraction_static 0.6 \
--n_samples_per_prompt 1 \
--enable_sleep True
```

See [On-Policy KD](../user_guide/on_policy_kd.md) for details.

## 5. Run multi-teacher KD

Multi-teacher KD routes each sample to one of multiple teacher models. Provide a
JSON config that maps routing keys to teacher paths:

```json
{
    "math": "Qwen3/Qwen3-14B",
    "code": "Qwen3/Qwen3-14B"
}
```

Your dataset must contain a `teacher_routing_key` field whose value matches a key
in the config. Then run either recipe:

```bash
bash examples/multi_teacher_distillation/run_multi_teacher_off_policy_distillation.sh
bash examples/multi_teacher_distillation/run_multi_teacher_on_policy_distillation.sh
```

Key flags:

```bash
--multi_teacher_config examples/multi_teacher_distillation/teacher_config.json \
--teacher_routing_key teacher_routing_key \
--kd_algorithm vanilla_kd
```

See [Multi-Teacher KD](../user_guide/multi_teacher_kd.md) for details.

## 6. Run SFT

```bash
bash examples/sft/run_qwen3_4b.sh
```

which runs:

```bash
python -m kdflow.cli.train_sft \
    --student_name_or_path Qwen3/Qwen3-4B \
    --train_dataset_path OpenLeecher/lmsys_chat_1m_clean \
    --train_batch_size 128 --micro_train_batch_size 2 \
    --learning_rate 2e-5 --bf16 True ...
```

## 7. Track your runs

All trainers integrate with **Weights & Biases**. Either:

- pass `--use_wandb True --wandb_project KDFlow` and log online, or
- use `--wandb_mode offline --wandb_dir ./output` and sync later.

## Where to go next

- Reference: every CLI flag is documented in [Arguments](../reference/arguments.md).
- Implement your own KD algorithm or loss in [Extending KDFlow](../reference/extending.md).
- Understand the design in [Architecture](../concepts/architecture.md).

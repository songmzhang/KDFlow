# CLI Entry Points

KDFlow does not register any console scripts. All training is launched via
`python -m kdflow.cli.<entry>` on top of a running Ray cluster.
Three entry points are provided:

| Module                              | Launcher  | Scenario   | Trainer               | Actor groups spawned                                              |
|-------------------------------------|-----------|------------|------------------------|-------------------------------------------------------------------|
| `kdflow.cli.train_sft`              | `python`  | `sft`      | `SFTTrainer`           | `StudentActorGroup`                                               |
| `kdflow.cli.train_kd_off_policy`    | `python`  | off-policy | `OffPolicyKDTrainer`   | `TeacherActorGroup`, `StudentActorGroup`                          |
| `kdflow.cli.train_kd_on_policy`     | `python`  | on-policy  | `OnPolicyKDTrainer`    | `TeacherActorGroup`, `StudentActorGroup`, `RolloutActorGroup`     |

Every entry point has a `if __name__ == "__main__":` block, parses arguments
with `init_args(scenario)` (see [Arguments](arguments.md)), and hands control
to the corresponding trainer.

---

## `kdflow.cli.train_sft`

Standard supervised fine-tuning. Uses FSDP2 + the same dataset / packing /
LoRA / Liger Kernel infrastructure as the KD trainers, with only a student
actor group (no teacher).

```bash
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

python -m kdflow.cli.train_sft \
    --num_nodes 1 --num_gpus_per_node 8 \
    --student_name_or_path Qwen3/Qwen3-4B \
    --train_dataset_path OpenLeecher/lmsys_chat_1m_clean \
    --train_batch_size 128 --micro_train_batch_size 2 \
    --learning_rate 2e-5 --bf16 True \
    --gradient_checkpointing True
```

For multi-node SFT, join the worker nodes to the Ray cluster and set
`--num_nodes` / `--num_gpus_per_node` accordingly, same as the KD entry points.

---

## `kdflow.cli.train_kd_off_policy`

Off-policy KD: teacher prefills the dataset; student trains on the resulting
hidden states.

```bash
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

python -m kdflow.cli.train_kd_off_policy \
    --student_name_or_path Qwen3/Qwen3-4B \
    --teacher_name_or_path Qwen3/Qwen3-30B-A3B \
    --train_dataset_path OpenLeecher/lmsys_chat_1m_clean \
    --kd_algorithm vanilla_kd --kd_loss_fn kl --kd_ratio 0.5 \
    --teacher_dp_size 2 --teacher_tp_size 4 \
    --backend fsdp2 --bf16 True
```

Setting `--rollout_num_engines 0` (the default) disables on-policy mode.

---

## `kdflow.cli.train_kd_on_policy`

On-policy KD: an additional `RolloutActorGroup` is spawned and produces
student rollouts that are then fed to the teacher.

```bash
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

python -m kdflow.cli.train_kd_on_policy \
    --student_name_or_path Qwen3/Qwen3-4B \
    --teacher_name_or_path Qwen3/Qwen3-30B-A3B \
    --train_dataset_path OpenLeecher/lmsys_chat_1m_clean \
    --kd_algorithm vanilla_kd --kd_loss_fn rkl --kd_ratio 1.0 \
    --rollout_num_engines 8 --rollout_tp_size 1 \
    --rollout_batch_size 1024 --n_samples_per_prompt 1 \
    --teacher_dp_size 2 --teacher_tp_size 4 \
    --enable_sleep True --bf16 True
```

The trainer drives the loop **rollout → teacher prefill → student train**, with
sleep / wakeup keeping a single role resident on the GPUs at any time when
`--enable_sleep True`.

---

## Ray cluster setup

Both KD entry points expect Ray to already be running:

```bash
# single node
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

# multi node
ray start --head --node-ip-address <head-ip> --num-gpus 8     # node 0
ray start --address <head-ip>:6379       --num-gpus 8         # workers
```

The total expected GPU count is `num_nodes * num_gpus_per_node`.

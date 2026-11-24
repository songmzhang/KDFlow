# Architecture

KDFlow is built around a **decoupled actor-group design**: each role
(student, teacher, rollout) runs in its own Ray actor group, and the role-specific
infrastructure (training framework, inference engine, parallelism strategy) can
evolve independently.

<p align="center">
  <img src="https://raw.githubusercontent.com/songmzhang/KDFlow/main/figures/architecture.png"
       alt="KDFlow Architecture" width="80%">
</p>

## Actor groups

| Group                | Backend           | What it does                                                                  |
|----------------------|-------------------|-------------------------------------------------------------------------------|
| `StudentActorGroup`  | **FSDP2**         | Wraps the student model; runs forward / backward / optimizer step.            |
| `TeacherActorGroup`  | **SGLang Engine** | Prefills the teacher and exposes its **last-layer hidden states**.            |
| `RolloutActorGroup`  | **SGLang Server** | (On-policy only) Generates student responses given the prompt batch.          |

A **single Python controller** (the `Trainer`, see below) drives the loop and
calls these actors via `ray.get(...)`. There is no synchronisation server.

## GPU co-location via Sleep / Wakeup

When `--enable_sleep True`, the three actor groups **share the same physical
GPUs**:

1. **Teacher phase** — teacher weights live on GPU; student optimizer states
   are offloaded to CPU.
2. **Student phase** — student optimizer states reload to GPU; teacher weights
   offload to CPU.
3. **Rollout phase** (on-policy) — rollout engines wake up; the others sleep.

This avoids dedicating a separate GPU pool to the teacher, which is critical
for very large teachers (200B+) that would otherwise be impossible to schedule
alongside training.

## Hidden-states transfer via shared memory

<p align="center">
  <img src="https://raw.githubusercontent.com/songmzhang/KDFlow/main/figures/cost.png"
       alt="Knowledge transfer cost" width="80%">
</p>

A naive teacher → student transfer of full **logits** is prohibitive for large
vocabularies. KDFlow instead:

1. Extracts **hidden states** from the teacher's last layer via SGLang.
2. Transfers them to the student through **shared memory** (zero-copy).
3. Reconstructs teacher logits **on the student side** using only the
   teacher's `lm_head` weights (kept resident on the student actors).

This dramatically reduces both memory footprint and inter-process communication.

## Token-based teacher load balancing

`TeacherActorGroup` distributes micro-batches across teacher actors using a
**greedy token-based** strategy: each actor receives a roughly equal **total
token count**, not just an equal number of sequences. This keeps prefill
balanced when sequence lengths vary widely (typical of chat datasets).

## Three trainers

| Trainer                | Driven by                              | Used for                |
|------------------------|----------------------------------------|--------------------------|
| `SFTTrainer`           | `kdflow.cli.train_sft`                 | Plain SFT, no teacher    |
| `OffPolicyKDTrainer`   | `kdflow.cli.train_kd_off_policy`       | Off-policy KD (CT-KD too)|
| `OnPolicyKDTrainer`    | `kdflow.cli.train_kd_on_policy`        | On-policy KD (CT-KD too) |

All KD trainers reuse the same dataset / packing / dynamic-batch logic and
just differ in whether a `RolloutActorGroup` is present and whether teacher
weights can be synced from the student.

## Pluggable algorithms and losses

KDFlow uses a small **registry pattern**:

- `kdflow.algorithms.register_algorithm(name)` populates `ALGO_DICT`.
- `kdflow.loss.register_loss(name)` populates `LOSS_DICT`.

The training step picks one algorithm (`--kd_algorithm`) and one loss
(`--kd_loss_fn`) from these registries. New ones can be added with a single
decorator — see [Extending KDFlow](../reference/extending.md).

## Argument layering

Configuration is split across seven dataclasses (see
[Arguments](../reference/arguments.md)) and parsed by a single
`HfArgumentParser`:

```
init_args(scenario)
 ├── ModelArguments
 ├── TrainingArguments
 ├── FSDPArguments
 ├── DistillationArguments   # KD only
 ├── DataArguments
 ├── LoggingArguments
 └── RolloutArguments        # on-policy only
```

`init_args` also validates and auto-fixes settings depending on the scenario
(e.g. forces `rollout_num_engines = 0` for off-policy).

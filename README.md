<div align="center">
  <img src="figures/logo.png" alt="KDFlow Logo" width="60%">

  ### **A User-friendly and Efficient Framework for LLM Knowledge Distillation**

  [![Release](https://img.shields.io/github/v/release/songmzhang/KDFlow)](https://github.com/songmzhang/KDFlow/releases)
  [![Docker](https://img.shields.io/badge/Docker-Hub-2496ED?logo=docker&logoColor=white)](https://hub.docker.com/repository/docker/songmzhang/kdflow/tags)
  [![License](https://img.shields.io/github/license/songmzhang/KDFlow)](LICENSE)
  [![arXiv](https://img.shields.io/badge/arXiv-2603.01875-b31b1b?logo=arxiv)](https://arxiv.org/abs/2603.01875)
  [![WeChat](https://img.shields.io/badge/WeChat-Group-07C160?logo=wechat&logoColor=white)](#-wechat-group)
  [![Stars](https://img.shields.io/github/stars/songmzhang/KDFlow?style=social)](https://github.com/songmzhang/KDFlow)

</div>

---

## 🔥 News

- **[2026/04]** ⚡ Support dynamic batch size (enabled via `--use_dynamic_bsz True` and `--max_token_len_per_gpu <N>`), which accelerates training by almost **60% to 100%**.
- **[2026/04]** 🎉 KDFlow v0.1.3 has been released, now supporting weight synchronization from student to teacher in on-policy self-distillation (controlled by `--teacher_update_freq`, defaults to `1` meaning the teacher is synced every global step when student and teacher share the same model path).
- **[2026/04]** 🐳 The docker image for KDFlow is now available on [Docker Hub](https://hub.docker.com/repository/docker/songmzhang/kdflow/tags), and the corresponding Dockerfile is also provided in `docker/`.
- **[2026/03]** 🎉 KDFlow v0.1.2 has been released, supporting multi-node TP/PP for extremely large teacher models.
- **[2026/03]** 💬 We have created a KDFlow WeChat group! Welcome to [join us](#-wechat-group) for discussion and communication!
- **[2026/03]** 🎉 KDFlow v0.1.1 released! Now supports **vision-language (multimodal) models** and **Qwen3.5 series**.

---

## 📑 Table of Contents

- [🔥 News](#-news)
- [✨ Key Features](#-key-features)
- [🚀 Quick Start](#-quick-start)
  - [Installation](#installation)
  - [Off-Policy Knowledge Distillation](#off-policy-knowledge-distillation)
  - [On-Policy Knowledge Distillation](#on-policy-knowledge-distillation)
  - [Cross-Tokenizer Knowledge Distillation](#cross-tokenizer-knowledge-distillation)
  - [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
- [⚙️ Arguments](#️-arguments)
- [🧩 Extending KDFlow](#-extending-kdflow)
  - [Adding a Custom KD Algorithm](#adding-a-custom-kd-algorithm)
  - [Adding a Custom KD Loss](#adding-a-custom-kd-loss)
- [🔑 Design Highlights](#-design-highlights)
- [🙏 Acknowledgement](#-acknowledgement)
- [📖 Citation](#-citation)
- [📄 License](#-license)
- [💬 WeChat Group](#-wechat-group)
- [⭐ Star History](#-star-history)

---

## ✨ Key Features

- **Decoupled Infrastructure** - Using SGLang & FSDP2 for teacher inference and student training respectively.
- **Off-Policy Knowledge Distillation** — Distill from pre-collected teacher hidden states on static datasets.
- **On-Policy Knowledge Distillation** — Student-generated rollout responses are used for teacher forward and distillation training in a closed loop.
- **Cross-Tokenizer Distillation** — Native support for distilling between models with different tokenizers (e.g., Llama → Qwen).
- **SFT Training (Black-box KD)** — Supervised fine-tuning on collected dataset.
- **MultiModal Support** — Support distillation with vision-language models (e.g., Qwen3-VL).
- **Colocate Mode** — Teacher and student models **share the same GPUs** via sleep/wakeup mechanism, maximizing GPU utilization.
- **Teacher on SGLang** — Teacher inference is powered by SGLang Engine, enabling high-throughput prefilling and flexible parallel strategies.
- **Pluggable KD Algorithms** — Built-in support for Vanilla KD and DSKD (Dual-Space Knowledge Distillation), with easy registration of custom algorithms.
- **Multiple Loss Functions** — Torch compiled KL divergence, Reverse KL divergence, JS divergence, Adaptive KL (AKL), etc.
- **LoRA Support** — Optional LoRA fine-tuning for the student model.
- **Wand&b Integration** — Built-in wand&b logging for experiment tracking.
- **High Training Efficiency** — Achieves **1.4x to 6x** faster distillation compared to mainstream knowledge distillation frameworks.

<p align="center">
  <img src="figures/architecture.png" alt="KDFlow Architecture" width="80%">
</p>

---

## 🚀 Quick Start

### Installation

Install from source:

```bash
git clone https://github.com/songmzhang/KDFlow.git
cd KDFlow
pip install -e ./
# install flash attention after torch installation
pip install flash_attn==2.8.3 --no-build-isolation
```

Use the prebuilt Docker image from Docker Hub:

```bash
docker pull songmzhang/kdflow:sgl059-torch291-cu128
```

Or build from the provided Dockerfile:

```bash
docker build -f docker/Dockerfile.sgl059.torch291.cu128 -t kdflow:sgl059-torch291-cu128 .
```

> To support Qwen3.5, please use the latest version of SGLang which supports transformers v5.3.0.

### Off-Policy Knowledge Distillation
LLMs:
```bash
bash ./examples/off_policy_kd/run_qwen3_30b_a3b_to_4b.sh
```
VLMs:
```bash
bash ./examples/off_policy_kd/run_qwen3_vl_30b_a3b_to_4b.sh
```

### On-Policy Knowledge Distillation
LLMs:
```bash
bash ./examples/on_policy_kd/run_qwen3_30b_a3b_to_4b.sh
```
VLMs:
```bash
bash ./examples/on_policy_kd/run_qwen3_vl_30b_a3b_to_4b.sh
```

### Cross-Tokenizer Knowledge Distillation

#### Off-Policy

Use SimpleCrossTokenizerKD (suggested):
```bash
bash ./examples/cross_tokenizer_kd/run_qwen3_30b_a3b_to_llama3_2_3b_offpolicy_simple_ctkd.sh
```

or DSKD:

```bash
bash ./examples/cross_tokenizer_kd/run_qwen3_30b_a3b_to_llama3_2_3b_offpolicy.sh
```

#### On-Policy

Use SimpleCrossTokenizerKD (suggested):
```bash
bash ./examples/cross_tokenizer_kd/run_qwen3_30b_a3b_to_llama3_2_3b_onpolicy_simple_ctkd.sh
```

or DSKD:

```bash
bash ./examples/cross_tokenizer_kd/run_qwen3_30b_a3b_to_llama3_2_3b_onpolicy.sh
```

### Supervised Fine-Tuning (SFT)

```bash
bash ./examples/sft/run_qwen3_4b.sh
```

---

## ⚙️ Arguments

### Model Arguments

| Argument | Default | Description |
|---|---|---|
| `--student_name_or_path` | `None` | Student model name or path |
| `--teacher_name_or_path` | `None` | Teacher model name or path |
| `--attn_implementation` | `flash_attention_2` | Attention implementation |
| `--use_liger_kernel` | `False` | Use Liger Kernel for student model |
| `--lora_rank` | `0` | LoRA rank (0 = disabled) |
| `--lora_alpha` | `16` | LoRA alpha |
| `--target_modules` | `all-linear` | LoRA target modules |
| `--lora_dropout` | `0.0` | LoRA dropout |
| `--ring_attn_size` | `1` | Ring attention group size for context parallelism |
| `--enable_thinking` | `False` | Enable thinking mode |
| `--disable_fast_tokenizer` | `False` | Disable fast tokenizer |

### Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--num_nodes` | `1` | Number of training nodes |
| `--num_gpus_per_node` | `8` | GPUs per node |
| `--num_epochs` | `1` | Number of training epochs |
| `--train_batch_size` | `128` | Global training batch size |
| `--micro_train_batch_size` | `1` | Per-GPU micro batch size |
| `--learning_rate` | `1e-6` | Learning rate |
| `--lr_scheduler` | `cosine_with_min_lr` | LR scheduler type |
| `--lr_warmup_ratio` | `0.05` | Warmup ratio |
| `--min_lr` | `1e-8` | Minimum learning rate |
| `--max_norm` | `1.0` | Gradient clipping max norm |
| `--weight_decay` | `0.0` | Weight decay |
| `--adam_betas` | `(0.9, 0.98)` | Adam optimizer betas |
| `--backend` | `fsdp2` | Training backend |
| `--gradient_checkpointing` | `False` | Enable gradient checkpointing |
| `--enable_sleep` | `False` | Enable sleep mode for all components (student, teacher, rollout) |
| `--eval_steps` | `-1` | Evaluate every N steps (-1 = disabled) |
| `--save_steps` | `-1` | Save checkpoint every N steps (-1 = disabled) |
| `--save_path` | `./ckpt/` | Model save path |
| `--ckpt_path` | `./ckpt/checkpoints_distill` | Checkpoint save path |
| `--seed` | `42` | Random seed |
| `--bf16` | `False` | Enable bfloat16 training |
| `--use_dynamic_bsz` | `False` | Enable dynamic batch size based on token count per GPU |
| `--max_token_len_per_gpu` | `0` | Maximum total token count per micro-batch when `use_dynamic_bsz` is True |

### FSDP Arguments

| Argument | Default | Description |
|---|---|---|
| `--fsdp_size` | `-1` | FSDP shard size for HSDP (-1 = full sharding) |
| `--cpu_offload` | `False` | Offload Adam optimizer states to CPU |

### Distillation Arguments

| Argument | Default | Description |
|---|---|---|
| `--kd_ratio` | `0.5` | KD loss weight: `loss = (1 - kd_ratio) * CE + kd_ratio * KD` |
| `--kd_temperature` | `1.0` | Temperature for softmax in KD |
| `--kd_algorithm` | `vanilla_kd` | KD algorithm (`vanilla_kd` / `dskd`) |
| `--kd_loss_fn` | `kl` | Divergence function (`kl` / `rkl` / `jsd` / `akl`) |
| `--teacher_tp_size` | `8` | Teacher tensor parallel size |
| `--teacher_ep_size` | `1` | Teacher expert parallel size (MoE models) |
| `--teacher_pp_size` | `1` | Teacher pipeline parallel size |
| `--teacher_dp_size` | `1` | Teacher data parallel size |
| `--teacher_forward_n_batches` | `1` | Teacher forward N batches at once |
| `--teacher_mem_fraction_static` | `0.4` | SGLang static memory fraction for teacher |
| `--teacher_offload_tags` | `all` | Offload tags for SGLang |
| `--teacher_quantization` | `None` | Teacher model quantization |
| `--dskd_token_align` | `eta` | Token alignment strategy for DSKD (`eta` / `cma`) |
| `--dskd_topk_vocab` | `-1` | Top-k vocab tokens for DSKD projector init (-1 = all) |
| `--dskd_projector_lr` | `1e-4` | Learning rate for DSKD projectors |
| `--jsd_beta` | `0.5` | Beta for Jensen-Shannon Divergence |
| `--skew_lambda` | `0.1` | Lambda for Skewed KL/RKL |
| `--adaptive_alpha` | `0.5` | Alpha for Adaptive KL Divergence |
| `--hrl_topk` | `5` | Top-k for Hierarchical Ranking Loss |
| `--teacher_update_freq` | `1` | Teacher weight update frequency (in global steps) for on-policy self-distillation |

### Rollout Arguments (On-Policy)

| Argument | Default | Description |
|---|---|---|
| `--rollout_num_engines` | `0` | Number of SGLang rollout engines (0 = off-policy) |
| `--rollout_tp_size` | `1` | Tensor parallel per rollout engine |
| `--rollout_batch_size` | `32` | Prompts per rollout iteration |
| `--n_samples_per_prompt` | `1` | Number of responses per prompt |
| `--generate_max_len` | `2048` | Max generation length |
| `--temperature` | `1.0` | Sampling temperature |
| `--top_p` | `1.0` | Top-p sampling |
| `--rollout_mem_fraction_static` | `0.6` | GPU memory utilization per rollout engine |
| `--print_rollout_sample` | `False` | Print a rollout sample after each rollout |

### Data Arguments

| Argument | Default | Description |
|---|---|---|
| `--train_dataset_path` | `None` | Training dataset path |
| `--train_dataset_probs` | `None` | Sampling probabilities for multiple datasets |
| `--train_split` | `train` | Train split name |
| `--eval_dataset_path` | `None` | Evaluation dataset path |
| `--eval_split` | `eval` | Eval split name |
| `--input_key` | `messages` | Dataset input key |
| `--output_key` | `None` | Dataset output key |
| `--image_key` | `None` | Image key for multimodal datasets |
| `--teacher_input_key` | `None` | Input key for teacher prompt (for self-distillation/context distillation) |
| `--label_key` | `None` | Label key in dataset |
| `--apply_chat_template` | `True` | Apply tokenizer chat template |
| `--max_len` | `4096` | Max sequence length |
| `--prompt_max_len` | `2048` | Max prompt length |
| `--max_samples` | `1e8` | Max number of samples to load |
| `--packing_samples` | `False` | Pack sequences for efficiency |
| `--preprocess_num_workers` | `8` | Number of workers for data preprocessing |

### Logging Arguments

| Argument | Default | Description |
|---|---|---|
| `--logging_steps` | `10` | Log every N steps |
| `--use_wandb` | `False` | Enable W&B logging |
| `--wandb_org` | `None` | W&B organization name |
| `--wandb_project` | `None` | W&B project name |
| `--wandb_group` | `None` | W&B group name |
| `--wandb_run_name` | `None` | W&B run name |
| `--wandb_mode` | `online` | W&B mode (`online` / `offline` / `disabled`) |
| `--wandb_dir` | `None` | Directory to store W&B offline logs |

---

## 🧩 Extending KDFlow

### Adding a Custom KD Algorithm

Create a new file in `kdflow/algorithms/` and register it:

```python
import torch
from kdflow.loss import LOSS_DICT
from kdflow.algorithms import register_algorithm


@register_algorithm("my_custom_kd")
class MyCustomKD:
    def __init__(self, strategy, student_model, teacher_lm_head, **kwargs):
        self.strategy = strategy
        self.student = student_model
        self.teacher_lm_head = teacher_lm_head
        self.loss_fn = LOSS_DICT[strategy.args.kd.loss_fn]

    def training_step(self, micro_batch):
        # Access student inputs
        student_input_ids = micro_batch["stu_input_ids"]
        student_attn_mask = micro_batch["stu_attn_mask"]
        student_loss_mask = micro_batch["stu_loss_mask"].bool()
        teacher_hiddens = micro_batch["teacher_hiddens"]
        avg_token_num = micro_batch["avg_micro_batch_token_num"]

        # Student forward
        output = self.student(student_input_ids, attention_mask=student_attn_mask, return_output=True)
        student_logits = output["logits"][student_loss_mask]

        # Teacher logits from hidden states + lm_head
        teacher_logits = self.teacher_lm_head(teacher_hiddens.to(self.teacher_lm_head.weight))

        # Compute your custom loss
        kd_loss = self.loss_fn(student_logits, teacher_logits, temperature=1.0)
        kd_loss = kd_loss.sum() / avg_token_num

        return {"loss": kd_loss, "kd_loss": kd_loss}
```

Then use it with `--kd_algorithm my_custom_kd`.

### Adding a Custom KD Loss

Create a new file in `kdflow/loss/` and register it:

```python
import torch
import torch.nn.functional as F 

from kdflow.loss import register_loss


@register_loss("my_custom_loss")
@torch.compile()
def compute_kl_div(
    student_logits,
    teacher_logits, 
    temperature=1.0,
    reduction="none",
    **kwargs
):
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    log_probs = torch.log_softmax(student_logits, -1, dtype=torch.float32)
    target_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
    kl_div = F.kl_div(log_probs, target_probs, reduction=reduction).sum(-1)
    
    return kl_div
```

Then use it with `--kd_loss_fn my_custom_loss`.

---

## 🔑 Design Highlights

### GPU Co-location via Sleep/Wakeup

KDFlow enables teacher and student to **share the same GPUs** through a sleep/wakeup mechanism:

1. **Teacher phase**: Teacher model weights are loaded on GPU, student optimizer states are offloaded to CPU.
2. **Student phase**: Student optimizer states are reloaded to GPU, teacher model weights are offloaded to CPU.

This allows running large teacher models (e.g., 200B+ parameters) on the same hardware as the student without requiring separate GPU pools.

### Hidden States Transfer via Shared Memory

<p align="center">
  <img src="figures/cost.png" alt="Knowledge transfer cost" width="80%">
</p>

Instead of transferring full teacher logits (which can be enormous for large vocabularies), KDFlow:

1. Extracts **hidden states** from the teacher's last layer via SGLang.
2. Transfers them to the student via **shared memory** (zero-copy).
3. Computes teacher logits **on the student side** using only the teacher's `lm_head` weights.

This dramatically reduces memory and communication overhead.

### Token-Based Teacher Load Balancing

The `TeacherActorGroup` uses a **greedy token-based load balancing** strategy to distribute micro-batches across teacher actors, ensuring even workload distribution when sequence lengths vary.

---

## 🙏 Acknowledgement

KDFlow is built upon the shoulders of outstanding open-source projects. We sincerely thank:

- [SGLang](https://github.com/sgl-project/sglang) — We deeply appreciate its support for extracting hidden states from model inference and its exceptional inference efficiency, which are critical to KDFlow's teacher inference pipeline.
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — We gratefully adopt its well-designed abstractions for model wrapping and distributed training strategy, which form the foundation of our training infrastructure.
- [slime](https://github.com/THUDM/slime) — We appreciate its elegant implementation of Ray placement group initialization and the weight update mechanism for SGLang, which greatly inspired our design of on-policy distillation.

---

## 📖 Citation

If you find KDFlow useful in your research or work, please consider citing our paper:

```bibtex
@article{zhang2026kdflow,
      title={KDFlow: A User-Friendly and Efficient Knowledge Distillation Framework for Large Language Models}, 
      author={Songming Zhang and Xue Zhang and Tong Zhang and Bojie Hu and Yufeng Chen and Jinan Xu},
      year={2026},
      eprint={2603.01875},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.01875}, 
}
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 💬 WeChat Group

Welcome to join our WeChat group for discussion and communication!

<p align="center">
  <img src="figures/wechat.jpg" alt="WeChat Group QR Code" width="300">
</p>

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=songmzhang/KDFlow&type=date&legend=top-left)](https://www.star-history.com/#songmzhang/KDFlow&type=date&legend=top-left)

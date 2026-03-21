set -e
set -x

# ============ TrainingArguments ============
OPTS=""
OPTS+=" --num_nodes 1"
OPTS+=" --num_gpus_per_node 8"
OPTS+=" --backend fsdp2"
OPTS+=" --train_batch_size 128"
OPTS+=" --micro_train_batch_size 2"
OPTS+=" --learning_rate 2e-5"
OPTS+=" --lr_warmup_ratio 0.05"
OPTS+=" --num_epochs 1"
OPTS+=" --save_path ./output/qwen3_4b"
OPTS+=" --bf16 True"
OPTS+=" --gradient_checkpointing True"
OPTS+=" --train_enable_sleep False"

# ============ ModelArguments ============
OPTS+=" --student_name_or_path Qwen3/Qwen3-4B"
OPTS+=" --enable_thinking False"

# ============ DataArguments ============
OPTS+=" --train_dataset_path OpenLeecher/lmsys_chat_1m_clean"
OPTS+=" --max_len 4096"
OPTS+=" --input_key conversations"
OPTS+=" --apply_chat_template True"
OPTS+=" --preprocess_num_workers 32"
OPTS+=" --packing_samples True"

# ============ LoggingArguments ============
OPTS+=" --logging_steps 10"
OPTS+=" --use_wandb True"
OPTS+=" --wandb_project KDFlow"
OPTS+=" --wandb_group off_policy_kd"
OPTS+=" --wandb_run_name qwen3_4b_sft"
OPTS+=" --wandb_mode offline"
OPTS+=" --wandb_dir ./output"

torchrun --nproc_per_node=8 -m kdflow.cli.train_sft $OPTS

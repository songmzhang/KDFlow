set -ex

# Start ray before first running
# ray start --head --node-ip --address 0.0.0.0 --num-gpus 8

# ============ TrainingArguments ============
OPTS=""
OPTS+=" --num_nodes 1"
OPTS+=" --num_gpus_per_node 8"
OPTS+=" --backend fsdp2"
OPTS+=" --train_batch_size 128"
OPTS+=" --micro_train_batch_size 8"
OPTS+=" --learning_rate 2e-5"
OPTS+=" --lr_warmup_ratio 0.05"
OPTS+=" --num_epochs 1"
OPTS+=" --save_path ./output/qwen3_vl_30b_a3b_to_4b"
OPTS+=" --bf16 True"
OPTS+=" --gradient_checkpointing True"
OPTS+=" --train_enable_sleep True"

# ============ ModelArguments ============
OPTS+=" --student_name_or_path Qwen/Qwen3-VL-4B-Instruct"
OPTS+=" --teacher_name_or_path Qwen/Qwen3-VL-30B-A3B-Instruct"
OPTS+=" --enable_thinking False"

# ============ DataArguments ============
OPTS+=" --train_dataset_path lmms-lab/llava-critic-113k"
OPTS+=" --max_len 4096"
OPTS+=" --input_key conversations"
OPTS+=" --image_key image"
OPTS+=" --apply_chat_template True"
OPTS+=" --preprocess_num_workers 32"
OPTS+=" --packing_samples False"  # packing is not supported for VL models

# ============ DistillationArguments ============
OPTS+=" --kd_ratio 0.5"
OPTS+=" --kd_loss_fn kl"
OPTS+=" --kd_algorithm vanilla_kd"
OPTS+=" --teacher_forward_n_batches 10"
OPTS+=" --teacher_dp_size 2"
OPTS+=" --teacher_tp_size 4"
OPTS+=" --teacher_mem_fraction_static 0.6"
OPTS+=" --teacher_enable_sleep True"

# ============ LoggingArguments ============
OPTS+=" --logging_steps 10"
OPTS+=" --use_wandb True"
OPTS+=" --wandb_project KDFlow"
OPTS+=" --wandb_group off_policy_kd_vl"
OPTS+=" --wandb_run_name qwen3_vl_30b_a3b_to_4b_vanilla_kd"
OPTS+=" --wandb_mode offline"
OPTS+=" --wandb_dir ./output"

python -m kdflow.cli.train_kd_off_policy $OPTS

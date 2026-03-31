set -e
set -x

# Start ray before first running
# ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

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
OPTS+=" --save_path ./output/qwen3_30b_a3b_to_llama3.2_3b"
OPTS+=" --bf16 True"
OPTS+=" --gradient_checkpointing True"
OPTS+=" --enable_sleep True"

# ============ ModelArguments ============
OPTS+=" --student_name_or_path meta-llama/Llama3.2-3B-Instruct"
OPTS+=" --teacher_name_or_path Qwen3/Qwen3-30B-A3B"
OPTS+=" --enable_thinking False"

# ============ RolloutArguments ============
OPTS+=" --rollout_batch_size 1024"
OPTS+=" --rollout_num_engines 8"
OPTS+=" --rollout_tp_size 1"
OPTS+=" --rollout_mem_fraction_static 0.6"
OPTS+=" --n_samples_per_prompt 1"

# ============ DataArguments ============
OPTS+=" --train_dataset_path OpenLeecher/lmsys_chat_1m_clean"
OPTS+=" --max_len 4096"
OPTS+=" --input_key conversations"
OPTS+=" --apply_chat_template True"
OPTS+=" --preprocess_num_workers 32"
OPTS+=" --packing_samples True"

# ============ DistillationArguments ============
OPTS+=" --kd_ratio 1.0"
OPTS+=" --kd_loss_fn rkl"
OPTS+=" --kd_algorithm dskd"
OPTS+=" --dskd_topk_vocab 20000"
OPTS+=" --dskd_token_align eta"
OPTS+=" --dskd_projector_lr 5e-4"
OPTS+=" --teacher_dp_size 2"
OPTS+=" --teacher_tp_size 4"
OPTS+=" --teacher_mem_fraction_static 0.6"

# ============ LoggingArguments ============
OPTS+=" --logging_steps 10"
OPTS+=" --use_wandb True"
OPTS+=" --wandb_project KDFlow"
OPTS+=" --wandb_group cross_tokenizer_kd_onpolicy"
OPTS+=" --wandb_run_name qwen3_30b_a3b_to_llama3.2_3b_dskd"
OPTS+=" --wandb_mode offline"
OPTS+=" --wandb_dir ./output"

python -m kdflow.cli.train_kd_on_policy $OPTS

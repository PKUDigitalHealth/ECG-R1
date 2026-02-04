
export NPROC_PER_NODE=6
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
export MASTER_PORT=29445
export IMAGE_MAX_TOKEN_NUM=768
export ECG_SEQ_LENGTH=5000
export ECG_PATCH_SIZE=50
export ROOT_ECG_DIR='/path/to/ECG-Protocol-Guided-Grounding-CoT/ecg_timeseries'
export ROOT_IMAGE_DIR="/path/to/ECG-Protocol-Guided-Grounding-CoT/ecg_images"
export ECG_TOWER_PATH='ecg_coca/checkpoint/cpt_wfep_epoch_20.pt'
export ECG_PROJECTOR_TYPE='mlp2x_gelu'
export ECG_MODEL_CONFIG='coca_ViT-B-32'
export FREEZE_ECG_TOWER=True 
export FREEZE_ECG_PROJECTOR=False

export INTERLEAVE_PROB=0.1
export MODALITY_DROPOUT_PROB=0.5

swift rlhf \
    --rlhf_type grpo \
    --dynamic_sample true \
    --loss_type bnpo \
    --epsilon_high 0.30 \
    --overlong_filter true \
    --max_completion_length 2048 \
    --reward_funcs soft_overlong \
    --soft_cache_length 512 \
    --beta 0 \
    --model path/to/sft/checkpoint \
    --external_plugins ecg_r1/plugin.py \
    --custom_register_path 'ecg_r1/register.py' \
    --system 'ecg_r1/system_prompt.txt' \
    --reward_funcs diagnosis_accuracy_reward key_diagnostic_evidence_reward format_reward \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8020 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset '/path/to/ECG-Protocol-Guided-Grounding-CoT/ecg_jsons/ECG-Protocol-Guided-Grounding-CoT-RL-4k.jsonl' \
    --load_from_cache_file false \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --save_strategy 'steps' \
    --eval_strategy 'steps' \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --output_dir ouput/ecg-r1-8b-dapo \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 64 \
    --num_generations 6 \
    --temperature 1.0 \
    --deepspeed zero2 \
    --log_completions true \
    --report_to swanlab \
    --num_iterations 1 \
    --async_generate false \
    --swanlab_project ecg-r1-8b-dapo \
    --dataset_shuffle False \
    --train_dataloader_shuffle False \
    --log_entropy True \
    --use_hf true
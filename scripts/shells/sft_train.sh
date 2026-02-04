export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=29520
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export IMAGE_MAX_TOKEN_NUM=768
export ECG_SEQ_LENGTH=5000
export ECG_PATCH_SIZE=50
export INTERLEAVE_PROB=0.1
export MODALITY_DROPOUT_PROB=0.5
export ROOT_ECG_DIR='/path/to/ECG-Protocol-Guided-Grounding-CoT/ecg_timeseries'
export ROOT_IMAGE_DIR="/path/to/ECG-Protocol-Guided-Grounding-CoT/ecg_images"
export ECG_TOWER_PATH='ecg_coca/checkpoint/cpt_wfep_epoch_20.pt'
export ECG_PROJECTOR_TYPE='mlp2x_gelu'
export ECG_MODEL_CONFIG='coca_ViT-B-32'
export FREEZE_ECG_TOWER=True 
export FREEZE_ECG_PROJECTOR=False

swift sft \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --model_type ecg_r1 \
    --template ecg_r1 \
    --dataset '/path/to/ECG-Protocol-Guided-Grounding-CoT/ecg_jsons/ECGInstruct.jsonl' \
              '/path/to/ECG-Protocol-Guided-Grounding-CoT/ecg_jsons/ECG-Protocol-Guided-Grounding-CoT.jsonl' \
    --custom_register_path 'ecg_r1/register.py' \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --train_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 2e-5 \
    --freeze_vit true \
    --freeze_aligner false \
    --gradient_accumulation_steps 2 \
    --eval_steps 5000 \
    --save_steps 5000 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 4096 \
    --output_dir ouput/ecg-r1-8b \
    --warmup_ratio 0.03 \
    --dataloader_num_workers 32 \
    --dataset_num_proc 32 \
    --deepspeed zero2 \
    --attn_impl flash_attention_2 \
    --report_to swanlab \
    --swanlab_project ecg-r1-8b \
    --device_map null \
    --dataset_shuffle True \
    --train_dataloader_shuffle True \
    --weight_decay 0. \
    --use_hf true
    
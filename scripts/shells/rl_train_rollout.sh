export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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

swift rollout \
    --model path/to/sft/checkpoint \
    --custom_register ecg_r1/register.py \
    --vllm_data_parallel_size 2 \
    --port 8020 \
    --use_hf true

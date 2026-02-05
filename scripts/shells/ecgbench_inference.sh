export NPROC_PER_NODE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=29530

export IMAGE_MAX_TOKEN_NUM=768
export ECG_SEQ_LENGTH=5000
export ECG_PATCH_SIZE=50
export ROOT_ECG_DIR='/path/to/ECG-Protocol-Guided-Grounding-CoT/ecg_timeseries'
export ROOT_IMAGE_DIR="path/to/ECGBench/images"
export ECG_TOWER_PATH='ecg_coca/checkpoint/cpt_wfep_epoch_20.pt'
export ECG_PROJECTOR_TYPE='mlp2x_gelu'
export ECG_MODEL_CONFIG='coca_ViT-B-32'
export FREEZE_ECG_TOWER=True 
export FREEZE_ECG_PROJECTOR=True


MODEL_PATH_RELATIVE="path/to/rl/checkpoint"
DATASET_BASE_PATH="path/to/ecg_bench/jsonls"

export BASE_DIR=$(pwd)
MODEL_PATH="$BASE_DIR/$MODEL_PATH_RELATIVE"
BASE_OUTPUT_DIR="$BASE_DIR/scripts/evaluation/ecgbench/llm_result_original"
DATASETS=(
    "code15-test"
    "cpsc-test"
    "csn-test-no-cot"
    "g12-test-no-cot"
    "ptb-test"
)

echo "脚本基准目录 (BASE_DIR): $BASE_DIR"
if [ -z "$MODEL_PATH_RELATIVE" ]; then
    echo "错误: MODEL_PATH_RELATIVE 变量为空，请在此脚本中设置它。"
    exit 1
fi
echo "开始处理模型: $MODEL_PATH"
INFER_RESULT_DIR="$MODEL_PATH/infer_result"
echo "监控推理目录: $INFER_RESULT_DIR"
mkdir -p "$INFER_RESULT_DIR"
MODEL_NAME_SANITISED=$(echo "$MODEL_PATH_RELATIVE" | sed 's|^training/||' | sed 's|/|-|g')
FINAL_MODEL_OUTPUT_DIR="$BASE_OUTPUT_DIR/$MODEL_NAME_SANITISED"
mkdir -p "$FINAL_MODEL_OUTPUT_DIR"
echo "所有结果将保存到: $FINAL_MODEL_OUTPUT_DIR"


for dataset_name in "${DATASETS[@]}"; do
    VAL_DATASET_PATH="$DATASET_BASE_PATH/$dataset_name.jsonl"
    DESTINATION_FILE_PATH="$FINAL_MODEL_OUTPUT_DIR/$dataset_name.jsonl"

    echo ""
    echo "================================================="
    echo "正在运行: $dataset_name"
    echo "源数据: $VAL_DATASET_PATH"
    echo "================================================="

    echo "检查锚点... ($INFER_RESULT_DIR)"
    BEFORE_FILES=$(find "$INFER_RESULT_DIR" -maxdepth 1 -name "*.jsonl" -printf "%f\n")
    if [ -z "$BEFORE_FILES" ]; then
        echo " -> 目录为空，不需要记录"
    else
        echo " -> 发现已存在的文件，记录并且忽略:"
        echo "$BEFORE_FILES" | sed 's/^/    - /'
    fi

    swift infer \
        --model "$MODEL_PATH" \
        --model_type ecg_r1 \
        --template ecg_r1 \
        --torch_dtype bfloat16 \
        --custom_register_path 'ecg_r1/register.py' \
        --infer_backend pt \
        --val_dataset "$VAL_DATASET_PATH" \
        --max_batch_size 32 \
        --task_type causal_lm \
        --max_new_tokens 2048 \
        --temperature 0.0 \
        --repetition_penalty 1.0 \

    echo "--- 推理完成: $dataset_name ---"

    echo "正在查找新文件..."
    SOURCE_FILE_PATH=""
    while read -r new_file; do
        new_filename=$(basename "$new_file")
        if ! echo "$BEFORE_FILES" | grep -qxF "$new_filename"; then
            echo "找到新文件: $new_filename"
            SOURCE_FILE_PATH="$new_file"
            break
        fi
    done < <(find "$INFER_RESULT_DIR" -maxdepth 1 -name "*.jsonl")

    if [[ -n "$SOURCE_FILE_PATH" && -f "$SOURCE_FILE_PATH" ]]; then
        echo "成功找到新输出文件: $SOURCE_FILE_PATH"
        echo "正在移动并重命名到: $DESTINATION_FILE_PATH"
        mv "$SOURCE_FILE_PATH" "$DESTINATION_FILE_PATH"
    else
        echo "错误: 未能在 $INFER_RESULT_DIR 中找到新生成的 .jsonl 文件。"
    fi

    echo "================================================="
    echo ""

done
echo "全部 5 个数据集处理完毕!"
echo "最终结果已保存于: $FINAL_MODEL_OUTPUT_DIR"


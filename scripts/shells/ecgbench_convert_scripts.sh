MODEL_NAME="ECG-R1-8B-RL"
GOLDEN_DATA_DIR="path/to/ecg_bench/jsonls"
LLM_RESULT_DIR="scripts/evaluation/ecgbench/llm_result_original/$MODEL_NAME"
CONVERTED_DIR="scripts/evaluation/ecgbench/result_processed_for_eval/$MODEL_NAME"

# code15-test
python scripts/evaluation/ecgbench/convert_for_eval.py \
    --llm_output_file "$LLM_RESULT_DIR/code15-test.jsonl" \
    --golden_data_file "$GOLDEN_DATA_DIR/code15-test.json" \
    --converted_output_file "$CONVERTED_DIR/code15-test.jsonl"

# cpsc-test
python scripts/evaluation/ecgbench/convert_for_eval.py \
    --llm_output_file "$LLM_RESULT_DIR/cpsc-test.jsonl" \
    --golden_data_file "$GOLDEN_DATA_DIR/cpsc-test.json" \
    --converted_output_file "$CONVERTED_DIR/cpsc-test.jsonl"

# csn-test-no-cot
python scripts/evaluation/ecgbench/convert_for_eval.py \
    --llm_output_file "$LLM_RESULT_DIR/csn-test-no-cot.jsonl" \
    --golden_data_file "$GOLDEN_DATA_DIR/csn-test-no-cot.json" \
    --converted_output_file "$CONVERTED_DIR/csn-test-no-cot.jsonl"

# g12-test-no-cot
python scripts/evaluation/ecgbench/convert_for_eval.py \
    --llm_output_file "$LLM_RESULT_DIR/g12-test-no-cot.jsonl" \
    --golden_data_file "$GOLDEN_DATA_DIR/g12-test-no-cot.json" \
    --converted_output_file "$CONVERTED_DIR/g12-test-no-cot.jsonl"

# ptb-test
python scripts/evaluation/ecgbench/convert_for_eval.py \
    --llm_output_file "$LLM_RESULT_DIR/ptb-test.jsonl" \
    --golden_data_file "$GOLDEN_DATA_DIR/ptb-test.json" \
    --converted_output_file "$CONVERTED_DIR/ptb-test.jsonl"
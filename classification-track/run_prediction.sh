#!/usr/bin/env bash
# Usage: bash run_prediction.sh <dev|test> <task:1|2> <model1|model2|model3|wag>

set -eo pipefail

PHASE=${1:-dev}
TASK_ID=${2:-1}
MODEL_ID=${3:-model1}

# Options: 4, 8, 16. Use 16 to disable quantization.
QUANTIZATION_BITS=4
BATCH_SIZE=4
OVERWRITE_ARGS=()

MODEL_PATH=${MODEL_PATH:-"models/${PHASE}/task${TASK_ID}/${MODEL_ID}"}
INPUT_PATH="data/${PHASE}/task${TASK_ID}/test.json"
OUTPUT_DIR="../outputs/${PHASE}/classification/task${TASK_ID}/${MODEL_ID}"

echo "Track: classification"
echo "Phase / task: ${PHASE} / ${TASK_ID}"
echo "Adapter: ${MODEL_ID} (${MODEL_PATH})"
echo "Quantization / batch size: ${QUANTIZATION_BITS}-bit / ${BATCH_SIZE}"

python scripts/predict.py \
  --model-path "$MODEL_PATH" \
  --input "$INPUT_PATH" \
  --output "$OUTPUT_DIR/predictions.csv" \
  --batch-size "$BATCH_SIZE" \
  --quantization-bits "$QUANTIZATION_BITS" \
  "${OVERWRITE_ARGS[@]}"

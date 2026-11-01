#!/usr/bin/env bash
# Usage: bash run_score.sh <dev|test> <task:1|2> <model1|model2|model3|wag>

set -eo pipefail

PHASE=${1:-dev}
TASK_ID=${2:-1}
MODEL_ID=${3:-model1}
OVERWRITE_ARGS=()

OUTPUT_DIR="../outputs/${PHASE}/classification/task${TASK_ID}/${MODEL_ID}"
METADATA="../metadata/${PHASE}/classification/task${TASK_ID}.json"

python scripts/score.py \
  --predictions "$OUTPUT_DIR/predictions.csv" \
  --metadata "$METADATA" \
  --output "$OUTPUT_DIR/score.json" \
  "${OVERWRITE_ARGS[@]}"

#!/usr/bin/env bash
# Average the three released candidates for one task. Clean models are not used.

set -eo pipefail

PHASE=${1:-dev}
TASK_ID=${2:-1}
OVERWRITE_ARGS=()

python ../scripts/merge_adapters.py \
  --adapters \
    "models/${PHASE}/task${TASK_ID}/model1" \
    "models/${PHASE}/task${TASK_ID}/model2" \
    "models/${PHASE}/task${TASK_ID}/model3" \
  --output "models/${PHASE}/task${TASK_ID}/wag" \
  "${OVERWRITE_ARGS[@]}"

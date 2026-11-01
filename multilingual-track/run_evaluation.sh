#!/usr/bin/env bash
# Run prediction and scoring for one multilingual adapter.

set -eo pipefail

PHASE=${1:-dev}
TASK_ID=${2:-1}
MODEL_ID=${3:-model1}

bash run_prediction.sh "$PHASE" "$TASK_ID" "$MODEL_ID"
bash run_score.sh "$PHASE" "$TASK_ID" "$MODEL_ID"

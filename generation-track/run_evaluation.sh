#!/usr/bin/env bash
# Run prediction, GPT-4o-mini judging, and scoring for one released adapter.

set -eo pipefail

PHASE=${1:-dev}
TASK_ID=${2:-1}
MODEL_ID=${3:-model1}

bash run_prediction.sh "$PHASE" "$TASK_ID" "$MODEL_ID"
bash run_judge.sh "$PHASE" "$TASK_ID" "$MODEL_ID"
bash run_score.sh "$PHASE" "$TASK_ID" "$MODEL_ID"

#!/usr/bin/env bash
# Usage: bash run_judge.sh <dev|test> <task:1|2> <model1|model2|model3|wag>

set -eo pipefail

PHASE=${1:-dev}
TASK_ID=${2:-1}
MODEL_ID=${3:-model1}

# GPT-4o-mini reproduces the public evaluation pipeline at modest API cost.
JUDGE_MODEL="gpt-4o-mini"
CONCURRENCY=8
RESTART_ARGS=()

OUTPUT_DIR="../outputs/${PHASE}/generation/task${TASK_ID}/${MODEL_ID}"
METADATA="../metadata/${PHASE}/generation/task${TASK_ID}.json"

echo "Judging ${PHASE} generation task ${TASK_ID}, adapter ${MODEL_ID}"
echo "Judge: ${JUDGE_MODEL}; concurrency: ${CONCURRENCY}"

python scripts/judge.py \
  --predictions "$OUTPUT_DIR/predictions.json" \
  --metadata "$METADATA" \
  --output "$OUTPUT_DIR/judgments.jsonl" \
  --model "$JUDGE_MODEL" \
  --concurrency "$CONCURRENCY" \
  "${RESTART_ARGS[@]}"

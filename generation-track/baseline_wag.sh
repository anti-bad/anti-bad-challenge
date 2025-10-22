#!/usr/bin/env bash

set -euo pipefail

# ============================================================================
# Weight Averaging (WAG) Baseline
# ============================================================================
# This script merges multiple LoRA adapters using weight averaging.
# This is a simple baseline defense method against backdoor attacks.

# ============================================================================
# CONFIGURATION
# ============================================================================

# Task number (1 or 2) - can be overridden by first argument
# Task 1: Llama models, Task 2: Qwen models
TASK_NUM=${1:-1}
TASK_ID="task${TASK_NUM}"

# Model paths to merge (can be 2 or more)
# Default: merge all 3 models in the task
MODEL_1="./models/${TASK_ID}/model1"
MODEL_2="./models/${TASK_ID}/model2"
MODEL_3="./models/${TASK_ID}/model3"

# Output path for merged model
OUTPUT_PATH="./models/${TASK_ID}/wag_merged"

# Quantization (saves memory during merging, does not affect final LoRA adapter)
# Can be overridden by USE_QUANTIZATION and QUANTIZATION_BITS env vars
USE_QUANTIZATION=${USE_QUANTIZATION:-true}
QUANTIZATION_BITS=${QUANTIZATION_BITS:-4}  # Options: 4, 8, 16 (16 means no quantization)

# ============================================================================
# MERGE MODELS
# ============================================================================

echo "Merging models using Weight Averaging (WAG)..."
echo "Task: ${TASK_ID}"
echo "Output: ${OUTPUT_PATH}"

python scripts/baseline_wag.py \
  --adapter_paths "${MODEL_1}" "${MODEL_2}" "${MODEL_3}" \
  --output_path "${OUTPUT_PATH}" \
  ${USE_QUANTIZATION:+--use_quantization} \
  ${USE_QUANTIZATION:+--quantization_bits "${QUANTIZATION_BITS}"}

echo ""
echo "Merging complete!"
echo "You can now use the merged model for prediction:"
echo "  1. Set MODEL_ID=\"wag_merged\" in pred.sh"
echo "  2. Run ./pred.sh to generate predictions"
echo "  3. Run ./eval.sh to evaluate results"

#!/usr/bin/env bash

set -euo pipefail

# ============================================================================
# Prediction Script for Classification Track
# ============================================================================
# This script generates predictions on test data using LoRA models.
# Models are decoder-based LLMs fine-tuned for sequence classification.

# ============================================================================
# CONFIGURATION - Modify these paths and parameters as needed
# ============================================================================

# Task ID (1 or 2) - can be overridden by first argument
TASK_ID=${1:-1}

# Model ID (model1, model2, or model3) - can be overridden by second argument
MODEL_ID=${2:-"model1"}

# LoRA adapter directory
# NOTE: Models are LoRA adapters trained for sequence classification
# Base model will be auto-detected from adapter_config.json
# You can override MODEL_PATH environment variable to use a custom model
MODEL_PATH=${MODEL_PATH:-"./models/task${TASK_ID}/${MODEL_ID}"}

# Test data path
TEST_PATH="./data/task${TASK_ID}/test.json"

# Output will be saved to: ../submission/cls_task${TASK_ID}.csv
# You can override by setting OUTPUT_PATH variable

# Batch size for inference - can be overridden by third argument or BATCH_SIZE env var
BATCH_SIZE=${3:-${BATCH_SIZE:-4}}

# Quantization (set to true for memory-efficient inference)
# Can be overridden by USE_QUANTIZATION and QUANTIZATION_BITS env vars
USE_QUANTIZATION=${USE_QUANTIZATION:-true}
QUANTIZATION_BITS=${QUANTIZATION_BITS:-4}  # Options: 4, 8, 16 (16 means no quantization)

# ============================================================================
# PREDICTION
# ============================================================================

echo "=========================================="
echo "Classification Track - Prediction"
echo "=========================================="
echo "Task: ${TASK_ID}"
echo "Model: ${MODEL_ID}"
echo "Input: ${TEST_PATH}"
echo "Output: ../submission/cls_task${TASK_ID}.csv"
echo "=========================================="

python scripts/predict.py \
  --model_path "${MODEL_PATH}" \
  --input_path "${TEST_PATH}" \
  --task "${TASK_ID}" \
  ${OUTPUT_PATH:+--output_path "${OUTPUT_PATH}"} \
  --batch_size "${BATCH_SIZE}" \
  ${USE_QUANTIZATION:+--use_quantization} \
  ${USE_QUANTIZATION:+--quantization_bits "${QUANTIZATION_BITS}"}

echo "Prediction completed!"
echo "Results saved to: ../submission/cls_task${TASK_ID}.csv"
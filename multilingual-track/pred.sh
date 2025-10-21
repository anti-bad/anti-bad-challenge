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

# Task ID (1 or 2)
TASK_ID=1

# Model ID (model1, model2, or model3)
MODEL_ID="model1"

# LoRA adapter directory
# NOTE: Models are LoRA adapters trained for sequence classification
# Base model will be auto-detected from adapter_config.json
MODEL_PATH="./models/task${TASK_ID}/${MODEL_ID}"

# Test data path
TEST_PATH="./data/task${TASK_ID}/test.json"

# Output will be saved to: ../submission/mul_task${TASK_ID}.csv
# You can override by setting OUTPUT_PATH variable

# Batch size for inference
BATCH_SIZE=32

# Quantization (set to true for memory-efficient inference)
USE_QUANTIZATION=true
QUANTIZATION_BITS=4  # Options: 4, 8, 16 (16 means no quantization)

# ============================================================================
# PREDICTION
# ============================================================================

echo "=========================================="
echo "Multilingual Track - Prediction"
echo "=========================================="
echo "Task: ${TASK_ID}"
echo "Model: ${MODEL_ID}"
echo "Input: ${TEST_PATH}"
echo "Output: ../submission/mul_task${TASK_ID}.csv"
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
echo "Results saved to: ../submission/mul_task${TASK_ID}.csv"
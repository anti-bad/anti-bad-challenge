#!/usr/bin/env bash

set -euo pipefail

# ============================================================================
# Prediction Script
# ============================================================================
# This script generates predictions on test data.
# Modify the paths below to customize your prediction setup.

# ============================================================================
# CONFIGURATION - Modify these paths and parameters as needed
# ============================================================================

# Task ID (1 or 2)
TASK_ID=1

# Model ID (model1, model2, model3)
MODEL_ID="model1"

# LoRA adapter directory
# NOTE: Models are LoRA adapters trained with all-linear target modules
# Base model will be auto-detected from adapter_config.json
MODEL_PATH="./models/task${TASK_ID}/${MODEL_ID}"

# Test data path
TEST_PATH="./data/task${TASK_ID}/test.json"

# Output will be saved to: ../submission/gen_task${TASK_ID}.json
# You can override by setting OUTPUT_PATH variable

# Generation parameters
MAX_NEW_TOKENS=256
TEMPERATURE=0.0  # 0.0 = deterministic
TOP_P=1.0        # 1.0 = no nucleus sampling

# Quantization (set to false for full precision, true for memory-efficient inference)
USE_QUANTIZATION=true
QUANTIZATION_BITS=4  # Options: 4, 8, 16 (16 means no quantization)

# ============================================================================
# PREDICTION
# ============================================================================

python scripts/predict.py \
  --model_path "${MODEL_PATH}" \
  --input_path "${TEST_PATH}" \
  --task "${TASK_ID}" \
  ${OUTPUT_PATH:+--output_path "${OUTPUT_PATH}"} \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  ${USE_QUANTIZATION:+--use_quantization} \
  ${USE_QUANTIZATION:+--quantization_bits "${QUANTIZATION_BITS}"}
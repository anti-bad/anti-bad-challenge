#!/usr/bin/env python3
"""
Classification Track - Prediction Script
Generates predictions using a LoRA model on test data for sequence classification.
Automatically detects base model from adapter_config.json.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig
)
from peft import PeftModel, PeftConfig

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:  # pragma: no cover
    load_safetensors = None

logging.basicConfig(level=logging.INFO, format='%(message)s')


def load_jsonl(path: Path) -> List[dict]:
    """Load JSONL file (one JSON object per line)."""
    data = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            data.append(json.loads(line))
    return data


def get_adapter_state_dict(model_path: str):
    """Load adapter weights from safetensors or bin file."""
    safetensor_path = Path(model_path) / "adapter_model.safetensors"
    bin_path = Path(model_path) / "adapter_model.bin"

    if safetensor_path.exists():
        if load_safetensors is None:
            raise ImportError(
                "safetensors is required to load adapter_model.safetensors but is not installed."
            )
        return load_safetensors(str(safetensor_path))
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu")

    raise FileNotFoundError(
        f"Adapter weights not found at {safetensor_path} or {bin_path}"
    )


def get_num_labels_from_adapter(model_path: str) -> int:
    """Infer num_labels from the adapter's classifier weights."""
    logging.info("Detecting num_labels from adapter weights...")

    state_dict = get_adapter_state_dict(model_path)
    candidate_keys = [
        key for key in state_dict.keys()
        if key.endswith("classifier.weight") or key.endswith("score.weight")
    ]

    if not candidate_keys:
        raise ValueError(
            "Could not find classifier weights in adapter to determine num_labels."
        )

    num_label_candidates = {int(state_dict[key].shape[0]) for key in candidate_keys}
    if len(num_label_candidates) > 1:
        logging.warning(
            f"Inconsistent num_labels detected in adapter weights: {num_label_candidates}. "
            "Using the maximum value."
        )

    num_labels = max(num_label_candidates)
    logging.info(f"Number of labels detected: {num_labels}")
    return num_labels


def get_base_model_from_adapter(model_path: str) -> str:
    """
    Read base model name from adapter_config.json.

    Args:
        model_path: Path to LoRA adapter directory

    Returns:
        Base model name or path
    """
    adapter_config_path = Path(model_path) / "adapter_config.json"

    if not adapter_config_path.exists():
        raise FileNotFoundError(
            f"adapter_config.json not found at {adapter_config_path}\n"
            f"Make sure {model_path} is a valid LoRA adapter directory."
        )

    with open(adapter_config_path, 'r') as f:
        adapter_config = json.load(f)

    base_model_name = adapter_config.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError(
            f"'base_model_name_or_path' not found in {adapter_config_path}\n"
            f"This field is required to load the base model."
        )

    return base_model_name


def disable_attention_cache(model):
    """Disable attention caching for stable inference."""
    # Disable at config level
    if hasattr(model, 'config'):
        model.config.use_cache = False
        if hasattr(model.config, 'use_sdpa'):
            model.config.use_sdpa = False
        if hasattr(model.config, 'use_flash_attention_2'):
            model.config.use_flash_attention_2 = False


def load_model_and_tokenizer(
    model_path: str,
    use_quantization: bool = False,
    quantization_bits: int = 16,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Load model and tokenizer for sequence classification inference.
    Automatically detects base model and num_labels from saved model.

    Args:
        model_path: Path to LoRA adapter directory
        use_quantization: Whether to use quantization
        quantization_bits: 4, 8, or 16 for quantization (16 = no quantization)

    Returns:
        Tuple of (model, tokenizer)
    """
    logging.info("=" * 60)
    logging.info("Loading Model")
    logging.info("=" * 60)

    # Check for LoRA adapter
    adapter_config_path = Path(model_path) / "adapter_config.json"
    if not adapter_config_path.exists():
        raise ValueError(
            f"No LoRA adapter found at {model_path}\n"
            f"Expected adapter_config.json in the directory."
        )

    # Load PEFT config to get base model
    peft_config = PeftConfig.from_pretrained(model_path)
    base_model = peft_config.base_model_name_or_path
    logging.info(f"Base model: {base_model}")
    logging.info(f"LoRA adapter: {model_path}")

    # Detect num_labels from adapter weights to align classifier heads
    num_labels = get_num_labels_from_adapter(model_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    tokenizer.padding_side = "left"

    # Set pad_token_id if not already set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    # Configure quantization
    quantization_config = None
    if use_quantization and quantization_bits < 16:
        if quantization_bits == 4:
            logging.info("Quantization: 4-bit (NF4)")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quantization_bits == 8:
            logging.info("Quantization: 8-bit")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        else:
            raise ValueError(f"Unsupported quantization_bits: {quantization_bits}")
    else:
        logging.info("Quantization: Disabled (BF16)")

    # Create base config - num_labels will be set when loading the adapter
    base_config = AutoConfig.from_pretrained(base_model)
    base_config.use_cache = False
    base_config.use_flash_attention_2 = False
    if hasattr(base_config, 'use_sdpa'):
        base_config.use_sdpa = False
    base_config.num_labels = num_labels

    # Model loading kwargs
    model_kwargs = {
        "config": base_config,
        "torch_dtype": torch.bfloat16,
        "quantization_config": quantization_config,
        "device_map": "auto",
    }

    # Load base model
    logging.info("Loading base model...")
    model = AutoModelForSequenceClassification.from_pretrained(base_model, **model_kwargs)

    # Resize embeddings if needed
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        old_size = model.get_input_embeddings().weight.shape[0]
        new_size = len(tokenizer)
        logging.info(f"Resizing token embeddings: {old_size} -> {new_size}")
        model.resize_token_embeddings(new_size)

    # Load LoRA adapter
    logging.info("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, model_path)

    # Ensure pad token ID is set in model config
    model.config.pad_token_id = tokenizer.pad_token_id

    # Disable attention cache
    disable_attention_cache(model)

    model.eval()
    logging.info("Model loaded successfully!")
    logging.info("=" * 60)

    return model, tokenizer


def predict(args: argparse.Namespace) -> None:
    """Generate predictions for the input dataset."""
    logging.info("")
    logging.info("=" * 60)
    logging.info("Prediction Configuration")
    logging.info("=" * 60)
    logging.info(f"Model: {args.model_path}")
    logging.info(f"Input: {args.input_path}")
    logging.info(f"Output: {args.output_path}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info("=" * 60)

    # Load test data
    test_data = load_jsonl(Path(args.input_path))
    logging.info(f"\nLoaded {len(test_data)} test samples")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_path=args.model_path,
        use_quantization=args.use_quantization,
        quantization_bits=args.quantization_bits,
    )

    logging.info(f"Number of labels: {model.config.num_labels}")
    logging.info("")

    # Get device
    device = next(model.parameters()).device

    # Collect predictions
    all_predictions = []

    logging.info("Generating predictions...")
    with torch.no_grad():
        # Process in batches
        for i in tqdm(range(0, len(test_data), args.batch_size), desc="Predicting"):
            batch_data = test_data[i:i + args.batch_size]
            batch_sentences = [item['sentence'] for item in batch_data]

            # Tokenize
            inputs = tokenizer(
                batch_sentences,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding=True
            ).to(device)

            # Forward pass
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            all_predictions.extend(predictions.tolist())

            # Clear cache periodically
            if torch.cuda.is_available() and i % (args.batch_size * 4) == 0:
                torch.cuda.empty_cache()

    # Save predictions to CSV
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({'label': all_predictions})
    df.to_csv(output_path, index=False)

    logging.info("")
    logging.info("=" * 60)
    logging.info(f"Saved {len(all_predictions)} predictions to {output_path}")
    logging.info("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate predictions with LoRA model")
    parser.add_argument("--model_path", required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--input_path", required=True, help="Input JSONL file")
    parser.add_argument("--task", type=int, required=True, choices=[1, 2], help="Task number (1 or 2)")
    parser.add_argument("--output_path", default=None, help="Output CSV file (default: ../../submission/mul_task{task}.csv)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--use_quantization", action="store_true", help="Enable quantization")
    parser.add_argument("--quantization_bits", type=int, default=16, choices=[4, 8, 16], help="Quantization bits")
    args = parser.parse_args()

    # Set default output path if not provided
    if args.output_path is None:
        from pathlib import Path
        script_dir = Path(__file__).parent
        args.output_path = str(script_dir.parent.parent / "submission" / f"mul_task{args.task}.csv")

    return args


def main() -> None:
    args = parse_args()
    predict(args)


if __name__ == "__main__":
    main()

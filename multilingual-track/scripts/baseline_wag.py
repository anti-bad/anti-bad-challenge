#!/usr/bin/env python3
"""
Weight Averaging (WAG) baseline for backdoor defense.
Merges multiple LoRA adapters by averaging their weights.
Supports sequence classification models.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List

import torch
from peft import PeftModel

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:  # pragma: no cover
    load_safetensors = None

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


def get_base_model_from_adapter(model_path: str) -> str:
    """Read base model name from adapter_config.json."""
    adapter_config_path = Path(model_path) / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found at {adapter_config_path}")

    with open(adapter_config_path, 'r') as f:
        adapter_config = json.load(f)

    base_model_name = adapter_config.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError(f"'base_model_name_or_path' not found in {adapter_config_path}")

    return base_model_name


def get_adapter_state_dict(model_path: str):
    """Load the adapter state dict from safetensors or bin file."""
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
    """Get num_labels directly from the adapter's saved classification head."""
    logging.info("Detecting num_labels from adapter weights...")

    try:
        state_dict = get_adapter_state_dict(model_path)
    except Exception as exc:
        logging.error(f"Unable to load adapter weights: {exc}")
        raise

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


def merge_lora_adapters(adapter_paths: List[str], output_path: str,
                        use_quantization: bool = False, quantization_bits: int = 16):
    """
    Merge multiple LoRA adapters using weight averaging.

    Args:
        adapter_paths: List of paths to LoRA adapter directories (minimum 2)
        output_path: Path to save merged adapter
        use_quantization: Whether to use quantization for loading base model (saves memory)
        quantization_bits: 4, 8, or 16 (16 = no quantization)
    """
    logging.info("=" * 60)
    logging.info("Weight Averaging (WAG) Baseline")
    logging.info("=" * 60)
    logging.info(f"Merging {len(adapter_paths)} adapters:")
    for i, path in enumerate(adapter_paths, 1):
        logging.info(f"  {i}. {path}")
    logging.info(f"Output: {output_path}")

    # Verify all adapters use same base model
    base_models = [get_base_model_from_adapter(path) for path in adapter_paths]
    if len(set(base_models)) > 1:
        raise ValueError(f"Adapters use different base models: {set(base_models)}")
    base_model_name = base_models[0]
    logging.info(f"Base model: {base_model_name}")

    # Get num_labels from first adapter
    num_labels = get_num_labels_from_adapter(adapter_paths[0])
    logging.info(f"Number of labels: {num_labels}")

    # Configure quantization
    quantization_config = None
    if use_quantization and quantization_bits < 16:
        if quantization_bits == 4:
            logging.info("Quantization: 4-bit")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif quantization_bits == 8:
            logging.info("Quantization: 8-bit")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True, llm_int8_threshold=6.0
            )
    else:
        logging.info("Quantization: Disabled (BF16)")

    # Create base config
    base_config = AutoConfig.from_pretrained(base_model_name)
    base_config.use_cache = False
    base_config.num_labels = num_labels
    base_config.use_flash_attention_2 = False
    if hasattr(base_config, 'use_sdpa'):
        base_config.use_sdpa = False

    # Load base model
    logging.info("Loading base model...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        config=base_config,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map="auto"
    )

    # Load and average LoRA weights
    logging.info("Loading LoRA adapters...")
    all_lora_weights = []

    for i, adapter_path in enumerate(adapter_paths, 1):
        logging.info(f"  Loading adapter {i}/{len(adapter_paths)}")

        # Load adapter temporarily
        model_with_adapter = PeftModel.from_pretrained(
            base_model, adapter_path, is_trainable=False
        )

        # Extract LoRA weights
        lora_weights = {}
        for name, param in model_with_adapter.named_parameters():
            if 'lora_' in name or 'modules_to_save' in name:
                lora_weights[name] = param.data.clone().cpu()

        all_lora_weights.append(lora_weights)

        # Unload adapter to free memory
        model_with_adapter = model_with_adapter.unload()
        del model_with_adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Average weights
    logging.info("Averaging weights...")
    averaged_weights = {}
    for key in all_lora_weights[0].keys():
        weights_to_avg = [w[key] for w in all_lora_weights]
        averaged_weights[key] = torch.stack(weights_to_avg).mean(dim=0)

    # Load first adapter as base for merged model
    logging.info("Creating merged adapter...")
    merged_model = PeftModel.from_pretrained(
        base_model, adapter_paths[0], is_trainable=True
    )

    # Apply averaged weights
    with torch.no_grad():
        for name, param in merged_model.named_parameters():
            if name in averaged_weights:
                param.copy_(averaged_weights[name].to(param.device))

    # Save merged adapter
    logging.info(f"Saving merged adapter to {output_path}")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_path)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_path)

    logging.info("=" * 60)
    logging.info("Merging completed successfully!")
    logging.info(f"Merged adapter saved to: {output_path}")
    logging.info("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapters using weight averaging")
    parser.add_argument("--adapter_paths", nargs='+', required=True,
                        help="Paths to LoRA adapter directories to merge (minimum 2)")
    parser.add_argument("--output_path", required=True,
                        help="Output path for merged adapter")
    parser.add_argument("--use_quantization", action="store_true",
                        help="Use quantization for base model loading (saves memory)")
    parser.add_argument("--quantization_bits", type=int, default=16, choices=[4, 8, 16],
                        help="Quantization bits (4, 8, or 16)")
    return parser.parse_args()


def main():
    args = parse_args()

    if len(args.adapter_paths) < 2:
        raise ValueError("Need at least 2 adapters to merge")

    merge_lora_adapters(
        adapter_paths=args.adapter_paths,
        output_path=args.output_path,
        use_quantization=args.use_quantization,
        quantization_bits=args.quantization_bits
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Weight Averaging (WAG) baseline for backdoor defense.
Merges multiple LoRA adapters by averaging their weights.
"""

import argparse
import json
import logging
import torch
from pathlib import Path
from typing import List
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

    # Configure quantization (saves memory, does not affect final LoRA adapter)
    quantization_config = None
    if use_quantization and quantization_bits < 16:
        if quantization_bits == 4:
            logging.info("Using 4-bit quantization for base model loading")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif quantization_bits == 8:
            logging.info("Using 8-bit quantization for base model loading")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True, llm_int8_threshold=6.0
            )
    else:
        logging.info("No quantization (BF16)")

    # Load base model
    logging.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16,
        quantization_config=quantization_config, device_map="auto"
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
            if 'lora_' in name:
                lora_weights[name] = param.detach().clone().cpu()

        all_lora_weights.append(lora_weights)

        # Clean up
        del model_with_adapter
        torch.cuda.empty_cache()

    # Average LoRA weights
    logging.info("Averaging LoRA weights...")
    averaged_weights = {}
    for name in all_lora_weights[0].keys():
        weight_stack = torch.stack([weights[name] for weights in all_lora_weights])
        averaged_weights[name] = weight_stack.mean(dim=0)

    # Create merged adapter
    logging.info("Creating merged adapter...")
    merged_model = PeftModel.from_pretrained(
        base_model, adapter_paths[0], is_trainable=False
    )

    # Replace with averaged weights
    with torch.no_grad():
        for name, param in merged_model.named_parameters():
            if 'lora_' in name and name in averaged_weights:
                param.data = averaged_weights[name].to(param.device, dtype=param.dtype)

    # Save merged adapter
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"Saving merged adapter to: {output_path}")
    merged_model.save_pretrained(output_path)

    # Save tokenizer
    tokenizer_source = adapter_paths[0] if Path(adapter_paths[0]).exists() else base_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokenizer.save_pretrained(output_path)

    logging.info("=" * 60)
    logging.info("Weight averaging complete!")
    logging.info(f"Merged adapter saved to: {output_path}")
    logging.info("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters using weight averaging")
    parser.add_argument("--adapter_paths", nargs='+', required=True,
                        help="Paths to LoRA adapter directories (minimum 2)")
    parser.add_argument("--output_path", required=True,
                        help="Path to save merged adapter")
    parser.add_argument("--use_quantization", action="store_true",
                        help="Use quantization for loading base model (saves memory)")
    parser.add_argument("--quantization_bits", type=int, default=16, choices=[4, 8, 16],
                        help="Quantization bits (4/8/16, 16=no quantization)")
    return parser.parse_args()


def main():
    args = parse_args()

    if len(args.adapter_paths) < 2:
        raise ValueError("At least 2 adapter paths are required for merging")

    merge_lora_adapters(
        adapter_paths=args.adapter_paths,
        output_path=args.output_path,
        use_quantization=args.use_quantization,
        quantization_bits=args.quantization_bits
    )


if __name__ == "__main__":
    main()

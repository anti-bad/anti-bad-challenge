#!/usr/bin/env python3
"""
Generation Track - Prediction Script
Generates predictions using a LoRA model on test data for text generation task.
Automatically detects base model from adapter_config.json.
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format='%(message)s')


def load_json_list(path: Path) -> List[Dict]:
    """Load JSON file containing a list of examples."""
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


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


def convert_to_messages(example: Dict) -> List[Dict]:
    """Convert example to chat message format."""
    if "messages" in example:
        return example["messages"]
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    if input_text:
        user_content = f"{instruction}\n\n{input_text}"
    else:
        user_content = instruction
    return [{"role": "user", "content": user_content}]


def _pick_dtype():
    """Pick an appropriate dtype automatically."""
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def load_model_and_tokenizer(
    model_path: str,
    use_quantization: bool = False,
    quantization_bits: int = 16,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model and tokenizer for text generation inference.
    Automatically detects base model from saved adapter.

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

    base_model = get_base_model_from_adapter(model_path)
    logging.info(f"Base model: {base_model}")
    logging.info(f"LoRA adapter: {model_path}")

    # Load tokenizer: left padding is standard for generation
    tokenizer_source = model_path if Path(model_path).exists() else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

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
            raise ValueError(f"Unsupported quantization_bits: {quantization_bits}. Use 4, 8, or 16.")
    else:
        torch_dtype = _pick_dtype()
        logging.info(f"Quantization: Disabled ({str(torch_dtype).split('.')[-1]})")

    model_kwargs = {
        "torch_dtype": _pick_dtype(),
        "quantization_config": quantization_config,
        "device_map": "auto",
        "use_cache": True,
    }

    # Load base model
    logging.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    # Resize embeddings if needed
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        old_size = model.get_input_embeddings().weight.shape[0]
        new_size = len(tokenizer)
        logging.info(f"Resizing token embeddings: {old_size} -> {new_size}")
        model.resize_token_embeddings(new_size)

    # Load LoRA adapter
    logging.info("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, model_path, is_trainable=False)
    model.eval()
    logging.info("Model loaded successfully!")
    logging.info("=" * 60)
    return model, tokenizer


def generate_responses(args: argparse.Namespace) -> None:
    """Generate text responses for the input dataset."""
    logging.info("")
    logging.info("=" * 60)
    logging.info("Prediction Configuration")
    logging.info("=" * 60)
    logging.info(f"Model: {args.model_path}")
    logging.info(f"Input: {args.input_path}")
    logging.info(f"Output: {args.output_path}")
    logging.info(f"Max tokens: {args.max_new_tokens}")
    logging.info(f"Temperature: {args.temperature}")
    logging.info(f"Top-p: {args.top_p}")
    logging.info("=" * 60)

    model, tokenizer = load_model_and_tokenizer(
        model_path=args.model_path,
        use_quantization=args.use_quantization,
        quantization_bits=args.quantization_bits,
    )
    dataset = load_json_list(Path(args.input_path))
    logging.info(f"\nLoaded {len(dataset)} test samples")
    logging.info("")

    # Get device
    device = next(model.parameters()).device

    # Collect predictions (sample-level inference, no batching)
    outputs: List[Dict] = []

    logging.info("Generating responses...")
    for example in tqdm(dataset, desc="Predicting", ncols=80):
        # Convert to chat format
        messages = convert_to_messages(example)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,
        ).to(device)
        prompt_length = encoded.input_ids.shape[1]

        # Generate response
        with torch.inference_mode():
            generated_ids = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature if args.temperature > 0 else 1.0,
                top_p=args.top_p,
                do_sample=args.temperature > 0,
                num_beams=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the generated part (excluding prompt)
        output_ids = generated_ids[0][prompt_length:]
        response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        # Store result
        record = dict(example)
        record["output"] = response
        outputs.append(record)

    # Save predictions to JSON
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(outputs, fh, indent=2, ensure_ascii=False)
    logging.info("")
    logging.info("=" * 60)
    logging.info(f"Saved {len(outputs)} predictions to {output_path}")
    logging.info("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate predictions with LoRA model")
    parser.add_argument("--model_path", required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--input_path", required=True, help="Input JSON file")
    parser.add_argument("--task", type=int, required=True, choices=[1, 2], help="Task number (1 or 2)")
    parser.add_argument("--output_path", default=None, help="Output JSON file (default: ../../submission/gen_task{task}.json)")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--use_quantization", action="store_true", help="Enable quantization")
    parser.add_argument("--quantization_bits", type=int, default=16, choices=[4, 8, 16], help="Quantization bits")
    args = parser.parse_args()
    if args.output_path is None:
        script_dir = Path(__file__).parent
        args.output_path = str(script_dir.parent.parent / "submission" / f"gen_task{args.task}.json")
    return args


def main() -> None:
    args = parse_args()
    generate_responses(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run one sequence-classification LoRA adapter on an ordered test set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}: expected a non-empty JSON array or JSONL file")
    return value


def label_count(model_path: Path) -> int:
    try:
        import torch
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("install requirements.txt before prediction") from exc
    safe = model_path / "adapter_model.safetensors"
    binary = model_path / "adapter_model.bin"
    if safe.is_file():
        state = load_file(str(safe), device="cpu")
    elif binary.is_file():
        state = torch.load(binary, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"adapter weights not found under {model_path}")
    heads = [
        tensor.shape[0] for key, tensor in state.items()
        if key.endswith("classifier.weight") or key.endswith("score.weight")
    ]
    if not heads or len(set(heads)) != 1:
        raise ValueError("unable to infer a unique label count from the saved classifier head")
    return int(heads[0])


def inference_dtype(torch):
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def predict(args: argparse.Namespace, records: list[dict]) -> list[int]:
    try:
        import torch
        from peft import PeftConfig, PeftModel
        from transformers import (
            AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
            BitsAndBytesConfig,
        )
    except ImportError as exc:
        raise RuntimeError("install requirements.txt before prediction") from exc
    labels = label_count(args.model_path)
    peft = PeftConfig.from_pretrained(args.model_path)
    base = peft.base_model_name_or_path
    tokenizer_source = args.model_path if (args.model_path / "tokenizer_config.json").is_file() else base
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    dtype = inference_dtype(torch)
    quantization = None
    if args.quantization_bits == 4:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        )
    elif args.quantization_bits == 8:
        quantization = BitsAndBytesConfig(load_in_8bit=True)
    config = AutoConfig.from_pretrained(base, num_labels=labels)
    config.pad_token_id = tokenizer.pad_token_id
    config.use_cache = False
    model = AutoModelForSequenceClassification.from_pretrained(
        base, config=config, torch_dtype=dtype,
        quantization_config=quantization, device_map="auto",
    )
    model = PeftModel.from_pretrained(model, args.model_path)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    device = next(model.parameters()).device
    sentences = [row["sentence"] for row in records]
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(sentences), args.batch_size):
            inputs = tokenizer(
                sentences[start:start + args.batch_size], return_tensors="pt",
                padding=True, truncation=True, max_length=args.max_length,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            predictions.extend(model(**inputs).logits.argmax(dim=-1).cpu().tolist())
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--quantization-bits", type=int, choices=(4, 8, 16), default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(f"{args.output} exists; pass --overwrite")
        records = read_records(args.input)
        labels = predict(args, records)
        if len(labels) != len(records):
            raise ValueError("prediction count does not match the input data")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["label"])
            writer.writerows([label] for label in labels)
        print(f"Saved {len(labels)} predictions to {args.output}")
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"prediction error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

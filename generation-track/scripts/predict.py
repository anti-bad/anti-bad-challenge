#!/usr/bin/env python3
"""Generate ordered responses from one causal-language-model LoRA adapter."""

from __future__ import annotations

import argparse
import json
import sys
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


def messages(record: dict) -> list[dict]:
    if isinstance(record.get("messages"), list):
        return record["messages"]
    instruction = record.get("instruction", "")
    context = record.get("input", "")
    content = f"{instruction}\n\n{context}" if context else instruction
    return [{"role": "user", "content": content}]


def inference_dtype(torch):
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def generate(args: argparse.Namespace, records: list[dict]) -> list[dict]:
    try:
        import torch
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("install requirements.txt before prediction") from exc
    peft = PeftConfig.from_pretrained(args.model_path)
    base = peft.base_model_name_or_path
    tokenizer_source = args.model_path if (args.model_path / "tokenizer_config.json").is_file() else base
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
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
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=dtype, quantization_config=quantization, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, args.model_path)
    model.eval()
    device = next(model.parameters()).device
    outputs = []
    for test_id, record in enumerate(records):
        prompt = tokenizer.apply_chat_template(
            messages(record), tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_length = encoded.input_ids.shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **encoded, max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else 1.0,
                top_p=args.top_p, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            generated[0][prompt_length:], skip_special_tokens=True
        ).strip()
        output = dict(record)
        output["test_id"] = test_id
        output["output"] = response
        outputs.append(output)
        print(f"Generated {test_id + 1}/{len(records)}", flush=True)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--quantization-bits", type=int, choices=(4, 8, 16), default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(f"{args.output} exists; pass --overwrite")
        records = read_records(args.input)
        outputs = generate(args, records)
        if any(not row["output"] for row in outputs):
            raise ValueError("one or more generated responses are empty")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(outputs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"Saved {len(outputs)} predictions to {args.output}")
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"prediction error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

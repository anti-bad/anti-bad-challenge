#!/usr/bin/env python3
"""Average exactly three compatible LoRA adapters without loading the base model."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def adapter_files(directory: Path) -> tuple[Path, dict]:
    config_path = directory / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        path = directory / name
        if path.is_file():
            return path, config
    raise FileNotFoundError(f"adapter weights not found under {directory}")


def compatible(configs: list[dict]) -> None:
    scalar = ("base_model_name_or_path", "r", "lora_alpha", "lora_dropout", "task_type")
    for field in scalar:
        if any(config.get(field) != configs[0].get(field) for config in configs[1:]):
            raise ValueError(f"candidate adapters disagree on {field}")
    for field in ("target_modules", "modules_to_save"):
        expected = set(configs[0].get(field) or [])
        if any(set(config.get(field) or []) != expected for config in configs[1:]):
            raise ValueError(f"candidate adapters disagree on {field}")


def load_weights(path: Path) -> dict:
    import torch

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def average(states: list[dict]) -> dict:
    import torch

    keys = set(states[0])
    if any(set(state) != keys for state in states[1:]):
        raise ValueError("candidate adapters contain different parameter keys")
    result = {}
    for key in sorted(keys):
        tensors = [state[key] for state in states]
        if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
            raise ValueError(f"candidate tensor shapes differ: {key}")
        if tensors[0].is_floating_point():
            result[key] = (
                torch.stack([tensor.float() for tensor in tensors])
                .mean(dim=0).to(tensors[0].dtype).contiguous()
            )
        else:
            if any(not torch.equal(tensor, tensors[0]) for tensor in tensors[1:]):
                raise ValueError(f"non-floating candidate tensors differ: {key}")
            result[key] = tensors[0].clone().contiguous()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapters", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
            raise FileExistsError(f"{args.output} is not empty; pass --overwrite")
        files_and_configs = [adapter_files(path.resolve()) for path in args.adapters]
        if len({path.suffix for path, _ in files_and_configs}) != 1:
            raise ValueError("candidate adapters use different weight formats")
        configs = [config for _, config in files_and_configs]
        compatible(configs)
        states = [load_weights(path) for path, _ in files_and_configs]
        merged = average(states)
        args.output.mkdir(parents=True, exist_ok=True)
        output_weights = args.output / files_and_configs[0][0].name
        if output_weights.suffix == ".safetensors":
            from safetensors.torch import save_file
            save_file(merged, str(output_weights), metadata={"format": "pt"})
        else:
            import torch
            torch.save(merged, output_weights)
        (args.output / "adapter_config.json").write_text(
            json.dumps(configs[0], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            source = args.adapters[0] / name
            if source.is_file():
                shutil.copy2(source, args.output / name)
        (args.output / "merge_metadata.json").write_text(
            json.dumps(
                {
                    "method": "weight_averaging",
                    "candidate_adapters": [str(path) for path in args.adapters],
                    "number_of_candidates": 3,
                    "clean_reference_included": False,
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        print(f"Merged exactly three candidate adapters into {args.output}")
    except (FileExistsError, FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        print(f"merge error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

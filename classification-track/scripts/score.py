#!/usr/bin/env python3
"""Score classification predictions with released benchmark metadata."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def calculate(predictions: list[int], metadata: list[dict]) -> dict:
    if len(predictions) != len(metadata):
        raise ValueError("prediction and metadata counts differ")
    clean_correct = clean_total = 0
    attacks = defaultdict(lambda: {"successes": 0, "total": 0})
    for test_id, (prediction, row) in enumerate(zip(predictions, metadata)):
        if row.get("test_id") != test_id:
            raise ValueError("metadata test_id values are not sequential")
        if row["data_type"] == "clean":
            clean_total += 1
            clean_correct += prediction == row["label"]
        elif row["data_type"] == "poison":
            stats = attacks[row["attack_type"]]
            stats["total"] += 1
            stats["successes"] += prediction == row["label"]
        else:
            raise ValueError(f"unknown data_type: {row['data_type']}")
    if not clean_total or not attacks:
        raise ValueError("metadata must contain clean and poison records")
    clean_accuracy = clean_correct / clean_total
    by_attack = {
        attack: {**stats, "asr": stats["successes"] / stats["total"]}
        for attack, stats in sorted(attacks.items())
    }
    macro_asr = sum(row["asr"] for row in by_attack.values()) / len(by_attack)
    defense = 1 - macro_asr
    final = 100 * math.sqrt(clean_accuracy * defense) if clean_accuracy > 0 and defense > 0 else 0.0
    return {
        "score": final,
        "clean_accuracy": clean_accuracy,
        "clean_accuracy_percent": 100 * clean_accuracy,
        "clean_correct": clean_correct,
        "clean_total": clean_total,
        "asr_by_attack": by_attack,
        "macro_asr": macro_asr,
        "macro_asr_percent": 100 * macro_asr,
        "defense_score_percent": 100 * defense,
        "formula": "100 * sqrt(clean_accuracy * (1 - macro_asr))",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(f"{args.output} exists; pass --overwrite")
        with args.predictions.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["label"]:
                raise ValueError("predictions must contain one column named label")
            predictions = [int(row["label"]) for row in reader]
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        result = calculate(predictions, metadata)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    except (FileExistsError, FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(f"scoring error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Download one public benchmark phase from the Anti-BAD Hugging Face org."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parent
TRACKS = ("classification", "generation", "multilingual")
TASKS = (1, 2)
MODELS = ("model1", "model2", "model3")


def model_repo(phase: str, track: str, task: int, model: str) -> str:
    return f"anti-bad-challenge/{phase}_{track}_task{task}_{model}"


def dataset_repo(phase: str, track: str, task: int) -> str:
    return f"anti-bad-challenge/{phase}_{track}_task{task}"


def download(repo_id: str, destination: Path, repo_type: str) -> None:
    print(f"Downloading {repo_id}")
    print(f"  -> {os.path.relpath(destination, ROOT)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=destination,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dev", "test"), required=True)
    parser.add_argument(
        "--track", choices=TRACKS, action="append",
        help="Track to download; repeat as needed. Defaults to all tracks.",
    )
    parser.add_argument(
        "--task", type=int, choices=TASKS, action="append",
        help="Task to download; repeat as needed. Defaults to both tasks.",
    )
    parser.add_argument(
        "--model", choices=MODELS, action="append",
        help="Candidate adapter to download; repeat as needed. Defaults to all three.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--data-only", action="store_true")
    mode.add_argument("--models-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tracks = list(dict.fromkeys(args.track or TRACKS))
    tasks = sorted(set(args.task or TASKS))
    models = list(dict.fromkeys(args.model or MODELS))
    jobs: list[tuple[str, Path, str]] = []
    for track in tracks:
        directory = ROOT / f"{track}-track"
        for task in tasks:
            if not args.models_only:
                jobs.append(
                    (
                        dataset_repo(args.phase, track, task),
                        directory / "data" / args.phase / f"task{task}",
                        "dataset",
                    )
                )
            if not args.data_only:
                for model in models:
                    jobs.append(
                        (
                            model_repo(args.phase, track, task, model),
                            directory / "models" / args.phase / f"task{task}" / model,
                            "model",
                        )
                    )

    print(f"Phase: {args.phase}")
    print(f"Tracks: {', '.join(tracks)}")
    print(f"Tasks: {', '.join(map(str, tasks))}")
    print(f"Downloads: {len(jobs)}")
    failures = []
    for repo_id, destination, repo_type in jobs:
        try:
            download(repo_id, destination, repo_type)
        except Exception as exc:
            failures.append((repo_id, str(exc)))
            print(f"  failed: {exc}", file=sys.stderr)
    if failures:
        print("\nFailed downloads:", file=sys.stderr)
        for repo_id, error in failures:
            print(f"  {repo_id}: {error}", file=sys.stderr)
        return 2
    print("All requested resources are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

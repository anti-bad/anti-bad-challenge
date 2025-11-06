#!/usr/bin/env python3
"""
Download models and datasets from HuggingFace for Anti-Bad-Challenge
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

# Define base directory
BASE_DIR = Path(__file__).parent

# Define model mappings: (repo_id, local_path)
MODEL_DOWNLOADS = [
    # Generation Track - Task 1
    ("anti-bad-challenge/dev_generation_task1_model1", "generation-track/models/task1/model1"),
    ("anti-bad-challenge/dev_generation_task1_model2", "generation-track/models/task1/model2"),
    ("anti-bad-challenge/dev_generation_task1_model3", "generation-track/models/task1/model3"),

    # Generation Track - Task 2
    ("anti-bad-challenge/dev_generation_task2_model1", "generation-track/models/task2/model1"),
    ("anti-bad-challenge/dev_generation_task2_model2", "generation-track/models/task2/model2"),
    ("anti-bad-challenge/dev_generation_task2_model3", "generation-track/models/task2/model3"),

    # Classification Track - Task 1
    ("anti-bad-challenge/dev_classification_task1_model1", "classification-track/models/task1/model1"),
    ("anti-bad-challenge/dev_classification_task1_model2", "classification-track/models/task1/model2"),
    ("anti-bad-challenge/dev_classification_task1_model3", "classification-track/models/task1/model3"),

    # Classification Track - Task 2
    ("anti-bad-challenge/dev_classification_task2_model1", "classification-track/models/task2/model1"),
    ("anti-bad-challenge/dev_classification_task2_model2", "classification-track/models/task2/model2"),
    ("anti-bad-challenge/dev_classification_task2_model3", "classification-track/models/task2/model3"),

    # Multilingual Track - Task 1
    ("anti-bad-challenge/dev_multilingual_task1_model1", "multilingual-track/models/task1/model1"),
    ("anti-bad-challenge/dev_multilingual_task1_model2", "multilingual-track/models/task1/model2"),
    ("anti-bad-challenge/dev_multilingual_task1_model3", "multilingual-track/models/task1/model3"),

    # Multilingual Track - Task 2
    ("anti-bad-challenge/dev_multilingual_task2_model1", "multilingual-track/models/task2/model1"),
    ("anti-bad-challenge/dev_multilingual_task2_model2", "multilingual-track/models/task2/model2"),
    ("anti-bad-challenge/dev_multilingual_task2_model3", "multilingual-track/models/task2/model3"),
]

# Define dataset mappings: (repo_id, local_path)
DATASET_DOWNLOADS = [
    # Classification Track
    ("anti-bad-challenge/dev_classification_task1", "classification-track/data/task1"),
    ("anti-bad-challenge/dev_classification_task2", "classification-track/data/task2"),

    # Generation Track
    ("anti-bad-challenge/dev_generation_task1", "generation-track/data/task1"),
    ("anti-bad-challenge/dev_generation_task2", "generation-track/data/task2"),

    # Multilingual Track
    ("anti-bad-challenge/dev_multilingual_task1", "multilingual-track/data/task1"),
    ("anti-bad-challenge/dev_multilingual_task2", "multilingual-track/data/task2"),
]


def download_models():
    """Download all models to specified directories"""
    print("=" * 80)
    print("Downloading models...")
    print("=" * 80)

    successful_downloads = 0
    failed_downloads = []

    for repo_id, local_path in MODEL_DOWNLOADS:
        full_path = BASE_DIR / local_path

        print(f"\n[{successful_downloads + len(failed_downloads) + 1}/{len(MODEL_DOWNLOADS)}] Downloading: {repo_id}")
        print(f"  Target directory: {local_path}")

        # Create parent directories if they don't exist
        full_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Download and overwrite existing files
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(full_path),
                local_dir_use_symlinks=False,
                repo_type="model",
            )
            print(f"  Successfully downloaded")
            successful_downloads += 1

        except Exception as e:
            print(f"  Failed to download: {e}")
            failed_downloads.append((repo_id, str(e)))

    return successful_downloads, failed_downloads


def download_datasets():
    """Download all datasets to specified directories"""
    print("\n" + "=" * 80)
    print("Downloading datasets...")
    print("=" * 80)

    successful_downloads = 0
    failed_downloads = []

    for repo_id, local_path in DATASET_DOWNLOADS:
        full_path = BASE_DIR / local_path

        print(f"\n[{successful_downloads + len(failed_downloads) + 1}/{len(DATASET_DOWNLOADS)}] Downloading: {repo_id}")
        print(f"  Target directory: {local_path}")

        # Create parent directories if they don't exist
        full_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Download and overwrite existing files
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(full_path),
                local_dir_use_symlinks=False,
                repo_type="dataset",
            )
            print(f"  Successfully downloaded")
            successful_downloads += 1

        except Exception as e:
            print(f"  Failed to download: {e}")
            failed_downloads.append((repo_id, str(e)))

    return successful_downloads, failed_downloads


def cleanup_cache_folders():
    """Remove .cache folders created during download"""
    print("\n" + "=" * 80)
    print("Cleaning up cache folders...")
    print("=" * 80)

    cache_folders = list(BASE_DIR.rglob(".cache"))
    for cache_folder in cache_folders:
        if cache_folder.is_dir():
            print(f"  Removing {cache_folder.relative_to(BASE_DIR)}...")
            shutil.rmtree(cache_folder)

    if cache_folders:
        print(f"  Removed {len(cache_folders)} cache folder(s)")
    else:
        print("  No cache folders found")


def main():
    print("=" * 80)
    print("Anti-Bad-Challenge - Resource Downloader")
    print("=" * 80)
    print(f"Total models: {len(MODEL_DOWNLOADS)}")
    print(f"Total datasets: {len(DATASET_DOWNLOADS)}")
    print("=" * 80)

    # Download resources
    model_success, model_failed = download_models()
    dataset_success, dataset_failed = download_datasets()

    # Clean up cache folders
    cleanup_cache_folders()

    # Summary
    print("\n" + "=" * 80)
    print("Download Summary")
    print("=" * 80)
    print(f"Models downloaded: {model_success}/{len(MODEL_DOWNLOADS)}")
    print(f"Datasets downloaded: {dataset_success}/{len(DATASET_DOWNLOADS)}")

    if model_failed:
        print(f"\nFailed model downloads: {len(model_failed)}")
        for repo_id, error in model_failed:
            print(f"  - {repo_id}: {error}")

    if dataset_failed:
        print(f"\nFailed dataset downloads: {len(dataset_failed)}")
        for repo_id, error in dataset_failed:
            print(f"  - {repo_id}: {error}")

    if not model_failed and not dataset_failed:
        print(f"\nAll resources downloaded successfully!")

    print("=" * 80)


if __name__ == "__main__":
    main()

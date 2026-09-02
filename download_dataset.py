"""
download_dataset.py — Download synthetic passport dataset from Hugging Face

Downloads the free tier of ud-synthetic/indian-passports (30 metadata rows)
and generates synthetic passport images + forged variants.

Usage:
    python download_dataset.py
"""

import csv
import json
from pathlib import Path

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False


def download_hf_metadata():
    """Download metadata from Hugging Face dataset."""
    if not HF_DATASETS_AVAILABLE:
        print("  [INFO] 'datasets' library not installed. Install with: pip install datasets")
        print("  [INFO] Skipping Hugging Face download, generating synthetic data only.")
        return []

    print("  Downloading from Hugging Face: ud-synthetic/indian-passports...")
    try:
        ds = load_dataset("ud-synthetic/indian-passports", split="train")
        records = []
        for row in ds:
            records.append(dict(row))
        print(f"  [OK] Downloaded {len(records)} metadata records")
        return records
    except Exception as e:
        print(f"  [WARN] Download failed: {e}")
        print("  [INFO] Generating synthetic data only.")
        return []


def save_metadata_csv(records, output_path):
    """Save metadata to CSV for later use."""
    if not records:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter out empty records
    valid_records = [r for r in records if r.get("pass_num") and r.get("surname")]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        if valid_records:
            writer = csv.DictWriter(f, fieldnames=valid_records[0].keys())
            writer.writeheader()
            writer.writerows(valid_records)

    print(f"  [OK] Saved metadata to {output_path} ({len(valid_records)} records)")


def main():
    print("\n" + "=" * 60)
    print("  DATASET PREPARATION")
    print("  Synthetic Indian Passports + Forgeries")
    print("=" * 60)

    # Create folder structure
    folders = [
        "dataset/genuine/passports",
        "dataset/forged/passports",
        "dataset/metadata",
    ]
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {folder}/")

    # Step 1: Download metadata
    print("\n[1/3] Downloading metadata from Hugging Face...")
    records = download_hf_metadata()
    save_metadata_csv(records, "dataset/metadata/passports_metadata.csv")

    # Step 2: Generate synthetic passport images
    print("\n[2/3] Generating synthetic passport images...")
    from generate_passports import generate_all_passports
    generate_all_passports(count=50, output_dir="dataset/genuine/passports")

    # Step 3: Generate forged variants
    print("\n[3/3] Generating forged passport variants...")
    from generate_forgeries import generate_all_forgeries
    generate_all_forgeries(
        genuine_dir="dataset/genuine/passports",
        output_dir="dataset/forged/passports",
        count=50,
    )

    # Summary
    genuine_count = len(list(Path("dataset/genuine/passports").glob("*.png")))
    forged_count = len(list(Path("dataset/forged/passports").glob("*.png")))

    print(f"\n{'=' * 60}")
    print(f"  DATASET READY")
    print(f"{'=' * 60}")
    print(f"  Genuine passports: {genuine_count}")
    print(f"  Forged passports:  {forged_count}")
    print(f"  Total:             {genuine_count + forged_count}")
    print(f"\n  Next steps:")
    print(f"    1. python train_classifier.py    # Train on new dataset")
    print(f"    2. python run_pipeline.py -i dataset/genuine/passports/passport_001.png")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

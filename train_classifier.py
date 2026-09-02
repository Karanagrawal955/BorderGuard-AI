"""
train_classifier.py — Standalone script for adaptive classifier training

Run this whenever you add new labeled images:
    python train_classifier.py

It will:
    1. Scan dataset/genuine/ and dataset/forged/ folders
    2. Extract features (TruFor + ELA) from each
    3. Incrementally re-fit the classifier head
    4. Save the updated model to models/classifier.pkl

Expected runtime: <60s for 100 images (CPU)
"""

import sys
import time
from pathlib import Path


def main():
    print("\n" + "="*60)
    print("  ADAPTIVE CLASSIFIER TRAINING")
    print("  Re-train after adding new labeled samples")
    print("="*60)

    # Check both old and new folder structures
    data_paths = [
        ("data/genuine", "data/tampered"),
        ("dataset/genuine", "dataset/forged"),
    ]

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    total_images = 0

    for genuine_dir, tampered_dir in data_paths:
        genuine_path = Path(genuine_dir)
        tampered_path = Path(tampered_dir)

        if genuine_path.exists():
            count = len([p for p in genuine_path.rglob("*") if p.suffix.lower() in image_exts and p.is_file()])
            if count > 0:
                print(f"\n  Found {count} images in {genuine_dir}/")
                total_images += count

        if tampered_path.exists():
            count = len([p for p in tampered_path.rglob("*") if p.suffix.lower() in image_exts and p.is_file()])
            if count > 0:
                print(f"  Found {count} images in {tampered_dir}/")
                total_images += count

    if total_images == 0:
        print("\n  No images found! Run download_dataset.py first.")
        print("  Or add images to data/ or dataset/ folders.")
        sys.exit(0)

    # Run training
    from pipeline.adaptive_learning import AdaptiveClassifier
    start = time.time()

    clf = AdaptiveClassifier()

    # Try both folder structures
    stats = clf.fit_from_folders("dataset/genuine", "dataset/forged")

    elapsed = time.time() - start

    print(f"\n  Training complete in {elapsed:.2f}s")
    print(f"  Model saved to: models/classifier.pkl")
    print(f"\n  Next steps:")
    print(f"    1. Add more images to dataset/ folders")
    print(f"    2. Run this script again to retrain")
    print(f"    3. Test with: python run_pipeline.py -i dataset/genuine/passports/passport_001.png")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

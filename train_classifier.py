"""
train_classifier.py — Standalone script for adaptive classifier training

Run this whenever you add new labeled images to data/genuine/ or data/tampered/:
    python train_classifier.py

It will:
    1. Scan both folders for new images
    2. Extract features (TruFor + ELA) from each
    3. Incrementally re-fit the classifier head
    4. Save the updated model to models/classifier.pkl

Expected runtime: <60s for 50 images (CPU)
"""

import sys
import time
from pathlib import Path


def main():
    print("\n" + "="*60)
    print("  ADAPTIVE CLASSIFIER TRAINING")
    print("  Re-train after adding new labeled samples")
    print("="*60)

    # Verify data folders exist
    genuine_path = Path("data/genuine")
    tampered_path = Path("data/tampered")

    if not genuine_path.exists():
        genuine_path.mkdir(parents=True, exist_ok=True)
        print(f"\n  Created empty directory: {genuine_path}")

    if not tampered_path.exists():
        tampered_path.mkdir(parents=True, exist_ok=True)
        print(f"\n  Created empty directory: {tampered_path}")

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    genuine_count = len([p for p in genuine_path.iterdir() if p.suffix.lower() in image_exts])
    tampered_count = len([p for p in tampered_path.iterdir() if p.suffix.lower() in image_exts])

    print(f"\n  Found images:")
    print(f"    data/genuine/:    {genuine_count} images")
    print(f"    data/tampered/:   {tampered_count} images")

    if genuine_count == 0 and tampered_count == 0:
        print("\n  No images found! Add images to the data/ folders first.")
        print("  You can use synthetic/fictional document images for testing.")
        print("\n  Folder structure:")
        print("    data/genuine/authentic_doc_1.jpg")
        print("    data/genuine/authentic_doc_2.jpg")
        print("    data/tampered/edited_doc_1.jpg")
        print("    data/tampered/forged_doc_1.jpg")
        sys.exit(0)

    # Run training
    from pipeline.adaptive_learning import AdaptiveClassifier
    start = time.time()

    clf = AdaptiveClassifier()
    stats = clf.fit_from_folders("data/genuine", "data/tampered")

    elapsed = time.time() - start

    print(f"\n  Training complete in {elapsed:.2f}s")
    print(f"  Model saved to: models/classifier.pkl")
    print(f"\n  Next steps:")
    print(f"    1. Add more images to data/ folders")
    print(f"    2. Run this script again to retrain")
    print(f"    3. Test with: python run_pipeline.py --image test.jpg")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

"""
adaptive_learning.py
---------------------
This is the "the model automatically trains itself from the uploaded
dataset" piece you asked for.

Usage:
    python -m pipeline.adaptive_learning

What it does:
  1. Scans data/genuine/ and data/tampered/ for image files.
  2. Runs preprocessing + feature extraction on every image (skipping
     any that are unreadable, with a warning - graceful degradation).
  3. Re-fits the AdaptiveClassifier on the full current set.
  4. Saves the fitted model + scaler to models/.
  5. Writes a training_metadata.json summary you can literally paste
     into your pitch deck ("last trained on N images, M genuine / K
     tampered, train accuracy X%").

Re-run this any time you add new images to the data folders - that's
the "adaptive" workflow: drop in more labeled samples, re-run this
script, the classifier updates. See classifier.py's module docstring
for the honest distinction between this and formal continual learning.
"""

import json
import os
import sys
import time

from pipeline.preprocessing import preprocess_document
from pipeline.classifier import (
    AdaptiveClassifier,
    extract_features,
    MODEL_PATH,
    SCALER_PATH,
    METADATA_PATH,
)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def scan_folder(folder_path: str) -> list[str]:
    if not os.path.isdir(folder_path):
        return []
    paths = []
    for fname in sorted(os.listdir(folder_path)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in VALID_EXTENSIONS:
            paths.append(os.path.join(folder_path, fname))
    return paths


def build_dataset(genuine_dir: str, tampered_dir: str) -> list:
    """
    Walks both label folders, preprocesses + extracts features for every
    image, and returns a list of FeatureVector objects. Any image that
    fails to load is skipped with a warning rather than crashing the
    whole run.
    """
    feature_vectors = []
    skipped = []

    for label, folder in [(0, genuine_dir), (1, tampered_dir)]:
        for path in scan_folder(folder):
            try:
                doc = preprocess_document(path)
                fv = extract_features(doc, label=label)
                feature_vectors.append(fv)
            except Exception as e:
                skipped.append((path, str(e)))

    if skipped:
        print(f"WARNING: skipped {len(skipped)} unreadable file(s):")
        for path, err in skipped:
            print(f"   - {path}: {err}")

    return feature_vectors


def run_adaptive_training(
    genuine_dir: str = "data/genuine",
    tampered_dir: str = "data/tampered",
    model_path: str = MODEL_PATH,
    scaler_path: str = SCALER_PATH,
    metadata_path: str = METADATA_PATH,
) -> dict:
    start = time.time()

    print(f"Scanning '{genuine_dir}' and '{tampered_dir}'...")
    feature_vectors = build_dataset(genuine_dir, tampered_dir)
    n_genuine = sum(1 for fv in feature_vectors if fv.label == 0)
    n_tampered = sum(1 for fv in feature_vectors if fv.label == 1)
    print(f"Found {len(feature_vectors)} usable images "
          f"({n_genuine} genuine, {n_tampered} tampered).")

    if len(feature_vectors) < 4 or n_genuine == 0 or n_tampered == 0:
        print(
            "\nNot enough data to train yet. You need at least a few "
            "images in BOTH data/genuine/ and data/tampered/.\n"
            "Add some sample documents and re-run this script."
        )
        return {"status": "insufficient_data", "n_genuine": n_genuine, "n_tampered": n_tampered}

    clf = AdaptiveClassifier()
    fit_result = clf.fit_incremental(feature_vectors)
    clf.save(model_path, scaler_path)

    elapsed = time.time() - start

    metadata = {
        "status": "trained",
        "n_samples": fit_result["n_samples"],
        "n_genuine": fit_result["n_genuine"],
        "n_tampered": fit_result["n_tampered"],
        "train_accuracy": fit_result["train_accuracy"],
        "feature_weights": fit_result["feature_weights"],
        "training_seconds": round(elapsed, 3),
        "trained_at_unix": int(time.time()),
    }

    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nTraining complete in {elapsed:.2f}s.")
    print(f"  Samples: {metadata['n_samples']} "
          f"({metadata['n_genuine']} genuine, {metadata['n_tampered']} tampered)")
    print(f"  Train accuracy: {metadata['train_accuracy']:.2%} "
          f"(training-set fit quality, NOT a generalization guarantee)")
    print(f"  Model saved to: {model_path}")
    print(f"  Metadata saved to: {metadata_path}")

    if metadata["n_samples"] < 20:
        print(
            "\nNOTE: fewer than 20 labeled samples - treat any accuracy "
            "number as a rough signal, not a defensible claim. Keep "
            "adding samples to both folders for a more reliable fit."
        )

    return metadata


if __name__ == "__main__":
    genuine_dir = sys.argv[1] if len(sys.argv) > 1 else "data/genuine"
    tampered_dir = sys.argv[2] if len(sys.argv) > 2 else "data/tampered"
    run_adaptive_training(genuine_dir, tampered_dir)

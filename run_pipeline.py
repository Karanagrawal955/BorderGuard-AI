"""
run_pipeline.py — CLI entry point for the document screening pipeline

Usage:
    python run_pipeline.py --image sample.jpg
    python run_pipeline.py --image sample.jpg --output outputs/result.png
    python run_pipeline.py --image sample.jpg --no-ocr
    python run_pipeline.py --train   # Re-train classifier on data/ folders

Output:
    - Prints authenticity score (0-100)
    - Saves heatmap overlay as PNG
    - Prints flagged OCR inconsistencies
    - Outputs detailed JSON report
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def run_pipeline(image_path: str, output_dir: str = "outputs", run_ocr: bool = True) -> dict:
    """
    Execute the full document screening pipeline on a single image.

    Returns the complete result dict for programmatic use.
    """
    from pipeline.preprocessing import preprocess_image
    from pipeline.trufor_detector import TruForDetector
    from pipeline.feature_extraction import extract_features
    from pipeline.scoring import compute_authenticity_score

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"  DOCUMENT AUTHENTICITY SCREENING")
    print(f"  SIH26188 — Fake Identity Detection Pipeline")
    print(f"{'='*60}")
    print(f"\n  Input: {image_path}")

    # ── Step 1: Preprocessing ──
    print("\n[1/5] Preprocessing image...")
    preprocessed = preprocess_image(image_path)
    print(f"  => Image loaded: {preprocessed['original'].shape[1]}x{preprocessed['original'].shape[0]}")
    print(f"  => ELA mean: {preprocessed['ela_stats']['mean']:.2f}, "
          f"hotspots: {preprocessed['ela_stats']['hotspot_count']}")

    # ── Step 2: TruFor Detection ──
    print("\n[2/5] Running tamper detection (TruFor)...")
    detector = TruForDetector()
    detection = detector.detect(preprocessed["original"])
    heatmap = detection["heatmap"]
    print(f"  => Heatmap mean: {np.mean(heatmap):.4f}")
    print(f"  => Confidence: {detection['confidence']:.4f}")

    # Generate and save heatmap overlay
    overlay = detector.get_heatmap_overlay(preprocessed["original"], heatmap)
    heatmap_path = output_path / "heatmap_overlay.png"
    cv2.imwrite(str(heatmap_path), overlay)
    print(f"  => Heatmap overlay saved: {heatmap_path}")

    # Also save raw heatmap as grayscale
    raw_heatmap_path = output_path / "raw_heatmap.png"
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    cv2.imwrite(str(raw_heatmap_path), heatmap_uint8)

    # ── Step 3: Feature Extraction ──
    print("\n[3/5] Extracting features...")
    features_result = extract_features(
        preprocessed["original"],
        heatmap,
        preprocessed["ela_map"],
        preprocessed["ela_stats"],
    )
    features = features_result["features"]
    print(f"  => Feature vector: {features.shape[0]} dimensions")
    print(f"  => Top features:")
    for name, val in zip(features_result["feature_names"][:5], features[:5]):
        print(f"    {name}: {val:.4f}")

    # ── Step 4: OCR (optional) ──
    ocr_result = None
    if run_ocr:
        print("\n[4/5] Running OCR extraction...")
        from pipeline.ocr_module import OCRProcessor
        ocr_processor = OCRProcessor()
        ocr_result = ocr_processor.process(preprocessed["original"])

        if ocr_result.is_readable:
            print(f"  => Detected {len(ocr_result.text_blocks)} text blocks")
            print(f"  => Full text preview: {ocr_result.full_text[:100]}...")
            if ocr_result.aadhaar_found:
                print(f"  => Aadhaar number found: {ocr_result.aadhaar_found}")
                print(f"  => Checksum valid: {ocr_result.checksum_valid}")
        else:
            print("  => OCR could not read document")

        if ocr_result.consistency_flags:
            print(f"  => Flagged {len(ocr_result.consistency_flags)} inconsistency(ies):")
            for flag in ocr_result.consistency_flags:
                print(f"    [{flag['status']}] {flag['check']}: {flag['message']}")
    else:
        print("\n[4/5] OCR skipped (--no-ocr flag)")

    # ── Step 5: Scoring ──
    print("\n[5/5] Computing authenticity score...")
    from pipeline.adaptive_learning import AdaptiveClassifier
    clf = AdaptiveClassifier()
    classifier_output = clf.predict(features)

    score = compute_authenticity_score(
        classifier_output=classifier_output,
        heatmap_mean=float(np.mean(heatmap)),
        ocr_result=ocr_result,
    )

    # ── Results ──
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"\n  Authenticity Score: {score.final_score}/100")
    print(f"  Risk Level: {score.risk_level}")
    print(f"\n  Component Breakdown:")
    print(f"    Classifier: {score.classifier_component:.1f} (weight: 50%)")
    print(f"    TruFor:     {score.trufor_component:.1f} (weight: 30%)")
    print(f"    OCR:        {score.ocr_component:.1f} (weight: 20%)")

    if score.flags:
        print(f"\n  Flagged Issues:")
        for flag in score.flags:
            print(f"    [{flag['status']}] {flag.get('check', 'N/A')}: {flag.get('message', 'N/A')}")

    print(f"\n  Summary:")
    for line in score.summary.split("\n"):
        print(f"    {line}")

    # Save JSON report
    report = {
        "image_path": str(image_path),
        "score": score.to_dict(),
        "classifier_output": classifier_output,
        "ela_stats": preprocessed["ela_stats"],
        "processing_time_sec": round(elapsed, 2),
    }
    report_path = output_path / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Full report saved: {report_path}")
    print(f"  Heatmap overlay saved: {heatmap_path}")
    print(f"\n  Processing time: {elapsed:.2f}s")
    print(f"{'='*60}\n")

    return report


def train_classifier():
    """Re-train the classifier on current data/ folders."""
    from pipeline.adaptive_learning import AdaptiveClassifier

    print("\n" + "="*60)
    print("  ADAPTIVE CLASSIFIER TRAINING")
    print("="*60)
    print("\n  Scanning data/genuine/ and data/tampered/ for images...")

    clf = AdaptiveClassifier()
    stats = clf.fit_from_folders("data/genuine", "data/tampered")

    print(f"\n  Training complete!")
    print(f"  => Genuine samples: {stats.get('genuine_count', 0)}")
    print(f"  => Tampered samples: {stats.get('tampered_count', 0)}")
    print(f"  => Training time: {stats.get('training_time_sec', 0)}s")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Document Authenticity Screening Pipeline — SIH26188",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --image sample.jpg
  python run_pipeline.py --image data/tampered/fake_doc.png --output results/
  python run_pipeline.py --image scan.tiff --no-ocr
  python run_pipeline.py --train
        """,
    )

    parser.add_argument(
        "--image", "-i",
        type=str,
        help="Path to the document image to analyze",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs",
        help="Output directory for results (default: outputs/)",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Skip OCR processing (faster, but misses text checks)",
    )
    parser.add_argument(
        "--train", "-t",
        action="store_true",
        help="Re-train the classifier on data/ folders",
    )

    args = parser.parse_args()

    if args.train:
        train_classifier()
    elif args.image:
        if not Path(args.image).exists():
            print(f"Error: Image not found: {args.image}")
            sys.exit(1)
        run_pipeline(args.image, args.output, run_ocr=not args.no_ocr)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

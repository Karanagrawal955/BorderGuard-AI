"""
run_pipeline.py — Full Document Authentication Pipeline

Usage:
    python run_pipeline.py --image sample.jpg
    python run_pipeline.py --image sample.jpg --output outputs/
    python run_pipeline.py --image sample.jpg --no-ocr
    python run_pipeline.py --train
    python run_pipeline.py --generate
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def run_pipeline(image_path: str, output_dir: str = "outputs", run_ocr: bool = True) -> dict:
    """Execute the full document screening pipeline."""
    from pipeline.preprocessing import preprocess_image
    from pipeline.trufor_detector import TruForDetector
    from pipeline.feature_extraction import extract_features
    from pipeline.ocr_module import OCRProcessor
    from pipeline.mrz_parser import parse_mrz, find_mrz_in_text
    from pipeline.document_validation import auto_validate
    from pipeline.face_detection import FaceDetector, detect_photo_region
    from pipeline.adaptive_learning import AdaptiveClassifier
    from pipeline.scoring import compute_authenticity_score

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"  DOCUMENT AUTHENTICITY SCREENING SYSTEM")
    print(f"  SIH26188 - AI-Based Fake Identity Detection")
    print(f"{'='*60}")
    print(f"\n  Input: {image_path}")

    # ── Step 1: Preprocessing ──
    print(f"\n[1/7] Preprocessing image...")
    preprocessed = preprocess_image(image_path)
    print(f"  -> Image: {preprocessed['original'].shape[1]}x{preprocessed['original'].shape[0]}")
    print(f"  -> ELA mean: {preprocessed['ela_stats']['mean']:.2f}, hotspots: {preprocessed['ela_stats']['hotspot_count']}")

    # ── Step 2: Tamper Detection (TruFor) ──
    print(f"\n[2/7] Running tamper detection (TruFor CNN)...")
    detector = TruForDetector()
    detection = detector.detect(preprocessed["original"])
    heatmap = detection["heatmap"]
    print(f"  -> Heatmap mean: {np.mean(heatmap):.4f}")

    # Save heatmap overlay
    overlay = detector.get_heatmap_overlay(preprocessed["original"], heatmap)
    heatmap_path = output_path / "heatmap_overlay.png"
    cv2.imwrite(str(heatmap_path), overlay)
    print(f"  -> Heatmap saved: {heatmap_path}")

    # ── Step 3: Feature Extraction ──
    print(f"\n[3/7] Extracting features...")
    features_result = extract_features(
        preprocessed["original"], heatmap,
        preprocessed["ela_map"], preprocessed["ela_stats"],
    )
    features = features_result["features"]
    print(f"  -> {features.shape[0]} features extracted")

    # ── Step 4: OCR Extraction ──
    ocr_result = None
    extracted_fields = {}
    if run_ocr:
        print(f"\n[4/7] Running OCR extraction...")
        ocr_processor = OCRProcessor()
        ocr_result = ocr_processor.process(preprocessed["original"])

        if ocr_result.is_readable:
            print(f"  -> {len(ocr_result.text_blocks)} text blocks detected")

            # Try to parse MRZ
            mrz_lines = find_mrz_in_text(ocr_result.full_text)
            if mrz_lines:
                print(f"  -> MRZ detected ({len(mrz_lines)} lines)")
                from pipeline.mrz_parser import parse_mrz
                mrz_data = parse_mrz(mrz_lines)
                if mrz_data.is_valid:
                    print(f"  -> MRZ VALID")
                    extracted_fields = {
                        "name": mrz_data.full_name,
                        "passport_number": mrz_data.passport_number,
                        "nationality": mrz_data.nationality,
                        "dob": mrz_data.date_of_birth,
                        "gender": mrz_data.sex,
                        "expiry": mrz_data.date_of_expiry,
                    }
                else:
                    print(f"  -> MRZ checksum errors: {mrz_data.checksum_errors}")

            # Extract fields from OCR text
            if not extracted_fields:
                extracted_fields = _extract_fields_from_text(ocr_result.full_text)

            # Print OCR flags
            if ocr_result.consistency_flags:
                print(f"  -> {len(ocr_result.consistency_flags)} consistency issues:")
                for flag in ocr_result.consistency_flags:
                    print(f"     [{flag['status']}] {flag.get('message', '')[:80]}")
        else:
            print("  -> OCR could not read document")
    else:
        print(f"\n[4/7] OCR skipped")

    # ── Step 5: Document Validation ──
    print(f"\n[5/7] Running document validation...")
    validation_result = None
    if extracted_fields:
        # Detect document type
        doc_type = _detect_document_type(image_path, extracted_fields)
        print(f"  -> Document type: {doc_type}")
        validation_result = auto_validate(doc_type, extracted_fields)
        print(f"  -> Validation score: {validation_result.score:.0f}/100")
        print(f"  -> Errors: {validation_result.errors}, Warnings: {validation_result.warnings}")

        for r in validation_result.results:
            status = "PASS" if r.is_valid else r.severity
            print(f"     [{status}] {r.field_name}: {r.message[:70]}")
    else:
        print("  -> No fields extracted, skipping validation")

    # ── Step 6: Face Detection ──
    print(f"\n[6/7] Running face detection...")
    face_detector = FaceDetector()
    photo_region = detect_photo_region(preprocessed["original"])
    if photo_region:
        print(f"  -> Photo region detected at ({photo_region[0]},{photo_region[1]})")
    face_result = face_detector.analyze_document(preprocessed["original"], photo_region)
    print(f"  -> Faces detected: {face_result.faces_detected}")
    if face_result.warnings:
        for warn in face_result.warnings:
            print(f"     [WARN] {warn}")
    else:
        print(f"  -> Face analysis: CONSISTENT")

    # ── Step 7: Scoring ──
    print(f"\n[7/7] Computing authenticity score...")
    clf = AdaptiveClassifier()
    classifier_output = clf.predict(features)

    score = compute_authenticity_score(
        classifier_output=classifier_output,
        heatmap_mean=float(np.mean(heatmap)),
        ocr_result=ocr_result,
        validation_result=validation_result,
        face_result=face_result,
        extracted_fields=extracted_fields,
    )

    # ── Results ──
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS")
    print(f"{'='*60}")
    print(f"\n  AUTHENTICITY SCORE: {score.final_score:.1f}/100")
    print(f"  RISK LEVEL: {score.risk_level}")
    print(f"\n  Component Breakdown:")
    print(f"    Classifier:    {score.classifier_component:.1f} (weight: 30%)")
    print(f"    TruFor:        {score.trufor_component:.1f} (weight: 25%)")
    print(f"    Validation:    {score.validation_component:.1f} (weight: 20%)")
    print(f"    OCR:           {score.ocr_component:.1f} (weight: 15%)")
    print(f"    Face:          {score.face_component:.1f} (weight: 10%)")

    if extracted_fields:
        print(f"\n  Extracted Fields:")
        for key, val in extracted_fields.items():
            print(f"    {key}: {val}")

    if score.flags:
        print(f"\n  Flagged Issues ({len(score.flags)}):")
        for flag in score.flags:
            print(f"    [{flag.get('status', '?')}] {flag.get('message', '')[:80]}")

    print(f"\n  Summary:")
    for line in score.summary.split("\n"):
        print(f"    {line}")

    # Save report
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

    print(f"\n  Reports saved to {output_dir}/")
    print(f"  -> report.json")
    print(f"  -> heatmap_overlay.png")
    print(f"  -> raw_heatmap.png")
    print(f"\n  Processing time: {elapsed:.2f}s")
    print(f"{'='*60}\n")

    return report


def _extract_fields_from_text(text: str) -> dict:
    """Extract key fields from OCR text using regex patterns."""
    import re
    fields = {}

    # Name patterns
    name_match = re.search(r'(?:Name|NAME)[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)', text)
    if name_match:
        fields["name"] = name_match.group(1)

    # Aadhaar number
    aadhaar_match = re.search(r'(\d{4}\s?\d{4}\s?\d{4})', text)
    if aadhaar_match:
        fields["aadhaar"] = aadhaar_match.group(1)

    # PAN number
    pan_match = re.search(r'([A-Z]{5}\d{4}[A-Z])', text)
    if pan_match:
        fields["pan"] = pan_match.group(1)

    # Passport number
    passport_match = re.search(r'(?:Passport|PASSPORT)[:\s]*([A-Z]\d{7})', text)
    if passport_match:
        fields["passport_number"] = passport_match.group(1)

    # DOB
    dob_match = re.search(r'(?:DOB|Date of Birth|dob)[:\s]*(\d{2}[./]\d{2}[./]\d{4})', text, re.IGNORECASE)
    if dob_match:
        fields["dob"] = dob_match.group(1)

    # Phone
    phone_match = re.search(r'(\+91\s?\d{5}\s?\d{5})', text)
    if phone_match:
        fields["phone"] = phone_match.group(1)

    # Gender
    gender_match = re.search(r'(?:Gender|Sex|gender|sex)[:\s]*(MALE|FEMALE|M|F)', text, re.IGNORECASE)
    if gender_match:
        fields["gender"] = gender_match.group(1).upper()

    return fields


def _detect_document_type(image_path: str, fields: dict) -> str:
    """Auto-detect document type from filename and fields."""
    filename = Path(image_path).name.lower()

    if "passport" in filename:
        return "INDIAN_PASSPORT"
    elif "pan" in filename:
        return "INDIAN_PAN"
    elif "aadhaar" in filename or "aadhar" in filename:
        return "INDIAN_AADHAAR"
    elif "visa" in filename:
        return "VISA"

    # Detect from fields
    if fields.get("passport_number"):
        return "INDIAN_PASSPORT"
    elif fields.get("pan"):
        return "INDIAN_PAN"
    elif fields.get("aadhaar"):
        return "INDIAN_AADHAAR"

    return "UNKNOWN"


def train_classifier():
    """Re-train the classifier."""
    from pipeline.adaptive_learning import AdaptiveClassifier
    print("\n" + "="*60)
    print("  ADAPTIVE CLASSIFIER TRAINING")
    print("="*60)
    clf = AdaptiveClassifier()
    stats = clf.fit_from_folders("dataset/genuine", "dataset/forged")
    print(f"\n  Training complete!")
    print(f"  -> Genuine: {stats.get('genuine_count', 0)}")
    print(f"  -> Tampered: {stats.get('tampered_count', 0)}")
    print(f"  -> Time: {stats.get('training_time_sec', 0)}s")
    print("="*60 + "\n")


def generate_dataset():
    """Generate synthetic document dataset."""
    from generate_from_dataset import generate_all_documents
    generate_all_documents(100)


def main():
    parser = argparse.ArgumentParser(
        description="Document Authenticity Screening System - SIH26188",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", "-i", type=str, help="Path to document image")
    parser.add_argument("--output", "-o", type=str, default="outputs", help="Output directory")
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR processing")
    parser.add_argument("--train", action="store_true", help="Re-train classifier")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic dataset")

    args = parser.parse_args()

    if args.train:
        train_classifier()
    elif args.generate:
        generate_dataset()
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

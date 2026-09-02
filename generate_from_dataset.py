"""
generate_from_dataset.py — Generate document images from indic-synthetic-profiles dataset

Reads the CSV data and creates realistic document images (Aadhaar-style, PAN-style)
using the synthetic identity profiles.

Usage:
    python generate_from_dataset.py
"""

import csv
import random
from pathlib import Path

import cv2
import numpy as np


def load_dataset(csv_path: str = "indic-synthetic-profiles/data.csv"):
    """Load the synthetic profiles dataset."""
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("name") and row.get("aadhaar"):
                records.append(row)
    print(f"  Loaded {len(records)} records from dataset")
    return records


def generate_aadhaar_card(record: dict, output_path: Path):
    """Generate a synthetic Aadhaar-style card image from a profile record."""
    width, height = 800, 500
    img = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)

    # Background (Aadhaar-style light yellow/cream)
    for y in range(height):
        ratio = y / height
        img[y, :] = [200 + int(ratio * 30), 230 + int(ratio * 15), 240]

    # Header bar (Aadhaar orange)
    cv2.rectangle(img, (0, 0), (width, 60), (0, 120, 200), -1)  # BGR: orange
    cv2.putText(img, "GOVERNMENT OF INDIA", (250, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "Unique Identification Authority of India", (200, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 200), 1, cv2.LINE_AA)

    # Photo placeholder (right side)
    photo_x, photo_y = 580, 80
    photo_w, photo_h = 180, 200
    cv2.rectangle(img, (photo_x, photo_y), (photo_x + photo_w, photo_y + photo_h),
                  (180, 180, 180), -1)
    cv2.rectangle(img, (photo_x, photo_y), (photo_x + photo_w, photo_y + photo_h),
                  (100, 100, 100), 2)
    cv2.putText(img, "PHOTO", (photo_x + 50, photo_y + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)

    # Signature placeholder
    sig_y = photo_y + photo_h + 20
    cv2.rectangle(img, (photo_x, sig_y), (photo_x + photo_w, sig_y + 40),
                  (220, 220, 220), -1)
    cv2.putText(img, "SIGNATURE", (photo_x + 40, sig_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

    # Text fields (left side)
    name = record.get("name", "UNKNOWN")
    aadhaar = record.get("aadhaar", "0000 0000 0000")
    dob = record.get("dob", "01/01/1990")
    gender = record.get("gender", "male").upper()
    address = record.get("address", "")
    phone = record.get("phone", "")

    # Truncate address for display
    if len(address) > 40:
        address = address[:37] + "..."

    fields = [
        (f"Name: {name}", (30, 100)),
        (f"DOB: {dob}", (30, 140)),
        (f"Gender: {gender}", (30, 175)),
        (f"", (30, 210)),  # Spacer
        (f"Aadhaar: {aadhaar}", (30, 250)),
        (f"Phone: {phone}", (30, 290)),
        (f"", (30, 320)),  # Spacer
        (f"Address:", (30, 360)),
        (f"{address}", (30, 390)),
    ]

    for text, pos in fields:
        if text:
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)

    # Bottom bar
    cv2.rectangle(img, (0, height - 40), (width, height), (0, 120, 200), -1)
    cv2.putText(img, "www.uidai.gov.in | 1947", (300, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)


def generate_pan_card(record: dict, output_path: Path):
    """Generate a synthetic PAN card-style image."""
    width, height = 800, 400
    img = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)

    # Background (PAN card blue tone)
    for y in range(height):
        ratio = y / height
        img[y, :] = [220 + int(ratio * 20), 200 + int(ratio * 30), 180]

    # Header
    cv2.rectangle(img, (0, 0), (width, 50), (0, 50, 150), -1)  # Dark blue
    cv2.putText(img, "INCOME TAX DEPARTMENT", (230, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Photo placeholder
    photo_x, photo_y = 50, 70
    photo_w, photo_h = 150, 180
    cv2.rectangle(img, (photo_x, photo_y), (photo_x + photo_w, photo_y + photo_h),
                  (180, 180, 180), -1)
    cv2.rectangle(img, (photo_x, photo_y), (photo_x + photo_w, photo_y + photo_h),
                  (80, 80, 80), 2)
    cv2.putText(img, "PHOTO", (photo_x + 40, photo_y + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)

    # Signature
    sig_y = photo_y + photo_h + 15
    cv2.rectangle(img, (photo_x, sig_y), (photo_x + photo_w, sig_y + 30),
                  (220, 220, 220), -1)
    cv2.putText(img, "SIGNATURE", (photo_x + 25, sig_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)

    # Fields
    name = record.get("name", "UNKNOWN")
    pan = record.get("pan", "ABCDE1234F")
    dob = record.get("dob", "01/01/1990")
    father_name = record.get("name", "FATHER NAME")  # Use same name as placeholder

    fields = [
        (f"Name: {name}", (230, 90)),
        (f"Father's Name: {father_name}", (230, 130)),
        (f"DOB: {dob}", (230, 170)),
        (f"PAN: {pan}", (230, 220)),
        (f"Signature:", (230, 280)),
    ]

    for text, pos in fields:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)

    # Bottom
    cv2.rectangle(img, (0, height - 35), (width, height), (0, 50, 150), -1)
    cv2.putText(img, "www.incometax.gov.in", (300, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)


def generate_all_documents(count: int = 100):
    """Generate document images from the dataset."""
    print("\n" + "=" * 60)
    print("  GENERATING DOCUMENTS FROM DATASET")
    print("=" * 60)

    # Load dataset
    records = load_dataset()
    if not records:
        print("  No records found. Check dataset path.")
        return

    # Create output folders
    folders = [
        "dataset/genuine/aadhaar",
        "dataset/genuine/pan",
        "dataset/forged/aadhaar",
        "dataset/forged/pan",
    ]
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)

    # Generate genuine documents
    print(f"\n[1/2] Generating {count} genuine documents...")
    selected = random.sample(records, min(count, len(records)))

    for i, record in enumerate(selected):
        # Generate Aadhaar
        aadhaar_path = Path(f"dataset/genuine/aadhaar/aadhaar_{i+1:03d}.png")
        generate_aadhaar_card(record, aadhaar_path)

        # Generate PAN
        pan_path = Path(f"dataset/genuine/pan/pan_{i+1:03d}.png")
        generate_pan_card(record, pan_path)

    print(f"  Generated {count} Aadhaar + {count} PAN cards in dataset/genuine/")

    # Generate forged documents
    print(f"\n[2/2] Generating {count} forged documents...")
    from generate_forgeries import generate_forged_passport

    for i, record in enumerate(selected):
        # Forged Aadhaar
        genuine_aadhaar = Path(f"dataset/genuine/aadhaar/aadhaar_{i+1:03d}.png")
        forged_aadhaar = Path(f"dataset/forged/aadhaar/forged_aadhaar_{i+1:03d}.png")
        if genuine_aadhaar.exists():
            img = cv2.imread(str(genuine_aadhaar))
            if img is not None:
                # Apply tampering
                result = img.copy()
                # Change aadhaar number
                cv2.rectangle(result, (30, 235), (350, 265), (220, 230, 240), -1)
                fake_aadhaar = f"Aadhaar: {random.randint(1000,9999)} {random.randint(1000,9999)} {random.randint(1000,9999)}"
                cv2.putText(result, fake_aadhaar, (30, 255),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
                cv2.imwrite(str(forged_aadhaar), result)

        # Forged PAN
        genuine_pan = Path(f"dataset/genuine/pan/pan_{i+1:03d}.png")
        forged_pan = Path(f"dataset/forged/pan/forged_pan_{i+1:03d}.png")
        if genuine_pan.exists():
            img = cv2.imread(str(genuine_pan))
            if img is not None:
                result = img.copy()
                # Change PAN number
                cv2.rectangle(result, (230, 205), (500, 235), (220, 200, 180), -1)
                fake_pan = f"PAN: {''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=5))}{random.randint(1000,9999)}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}"
                cv2.putText(result, fake_pan, (230, 225),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
                cv2.imwrite(str(forged_pan), result)

    print(f"  Generated {count} forged Aadhaar + {count} forged PAN cards in dataset/forged/")

    # Summary
    genuine_aadhaar = len(list(Path("dataset/genuine/aadhaar").glob("*.png")))
    genuine_pan = len(list(Path("dataset/genuine/pan").glob("*.png")))
    forged_aadhaar = len(list(Path("dataset/forged/aadhaar").glob("*.png")))
    forged_pan = len(list(Path("dataset/forged/pan").glob("*.png")))

    print(f"\n{'=' * 60}")
    print(f"  DATASET READY")
    print(f"{'=' * 60}")
    print(f"  Genuine Aadhaar: {genuine_aadhaar}")
    print(f"  Genuine PAN:     {genuine_pan}")
    print(f"  Forged Aadhaar:  {forged_aadhaar}")
    print(f"  Forged PAN:      {forged_pan}")
    print(f"  Total:           {genuine_aadhaar + genuine_pan + forged_aadhaar + forged_pan}")
    print(f"\n  Next: python train_classifier.py")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    generate_all_documents(100)

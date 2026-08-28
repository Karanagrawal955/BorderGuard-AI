"""
generate_test_samples.py — Create synthetic test document images

Generates simple synthetic "document" images for testing the pipeline.
These are NOT real documents — they're synthetic placeholders with:
    - Text fields (name, ID number, date)
    - Simple borders and layout
    - Some with intentional "tampering" (different font sizes, spacing)

Usage:
    python generate_test_samples.py
"""

import cv2
import numpy as np
from pathlib import Path


def create_blank_document(
    width: int = 800,
    height: int = 600,
    bg_color: tuple = (240, 240, 240),
) -> np.ndarray:
    """Create a blank document background."""
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    # Add border
    cv2.rectangle(img, (20, 20), (width-20, height-20), (100, 100, 100), 2)
    return img


def add_text_field(
    img: np.ndarray,
    text: str,
    position: tuple,
    font_scale: float = 0.6,
    color: tuple = (0, 0, 0),
    thickness: int = 1,
) -> np.ndarray:
    """Add a text field to the document."""
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)
    return img


def generate_genuine_sample(output_path: Path, doc_id: int):
    """Generate a synthetic 'authentic' document."""
    img = create_blank_document()

    # Header
    add_text_field(img, "SYNTHETIC IDENTITY DOCUMENT", (200, 60),
                   font_scale=0.8, color=(0, 0, 150), thickness=2)
    add_text_field(img, "FOR TESTING PURPOSES ONLY", (230, 90),
                   font_scale=0.5, color=(150, 0, 0))

    # Fields (consistent font size)
    fields = [
        ("Name: John Smith", (50, 150)),
        ("ID: 1234 5678 9012", (50, 200)),
        ("DOB: 01/01/1990", (50, 250)),
        ("Address: 123 Main St", (50, 300)),
        ("City: Springfield", (50, 350)),
        ("State: IL", (50, 400)),
    ]

    for text, pos in fields:
        add_text_field(img, text, pos, font_scale=0.55, thickness=1)

    # Add some visual elements
    cv2.line(img, (50, 430), (750, 430), (100, 100, 100), 1)
    add_text_field(img, "Signature: _______________", (50, 470),
                   font_scale=0.5, color=(80, 80, 80))
    add_text_field(img, f"Document #{doc_id:04d}", (50, 520),
                   font_scale=0.4, color=(120, 120, 120))

    # Add photo placeholder
    cv2.rectangle(img, (600, 130), (750, 330), (200, 200, 200), -1)
    add_text_field(img, "PHOTO", (660, 240), font_scale=0.5, color=(150, 150, 150))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    print(f"  Generated genuine sample: {output_path}")


def generate_tampered_sample(output_path: Path, doc_id: int):
    """Generate a synthetic 'tampered' document with visible inconsistencies."""
    img = create_blank_document()

    # Header (slightly different font size — inconsistency #1)
    add_text_field(img, "SYNTHETIC IDENTITY DOCUMENT", (200, 60),
                   font_scale=0.8, color=(0, 0, 150), thickness=2)
    add_text_field(img, "FOR TESTING PURPOSES ONLY", (230, 90),
                   font_scale=0.5, color=(150, 0, 0))

    # Fields with INCONSISTENT font sizes (tampering indicator)
    fields = [
        ("Name: John Smith", (50, 150), 0.55),
        ("ID: 9999 8888 7777", (50, 200), 0.7),  # DIFFERENT SIZE — tampered
        ("DOB: 01/01/1990", (50, 255), 0.55),
        ("Address: 456 Oak Ave", (50, 305), 0.55),
        ("City: Chicago", (50, 355), 0.45),  # SMALLER — tampered
        ("State: IL", (50, 400), 0.55),
    ]

    for text, pos, scale in fields:
        add_text_field(img, text, pos, font_scale=scale, thickness=1)

    # Unusual spacing gap (tampering indicator #2)
    cv2.line(img, (50, 440), (750, 440), (100, 100, 100), 1)
    add_text_field(img, "Signature: _______________", (50, 480),
                   font_scale=0.5, color=(80, 80, 80))
    add_text_field(img, f"Document #{doc_id:04d}", (50, 550),
                   font_scale=0.4, color=(120, 120, 120))

    # Add photo placeholder with different color (editing artifact)
    cv2.rectangle(img, (600, 130), (750, 330), (180, 200, 220), -1)  # Slightly blue
    add_text_field(img, "PHOTO", (660, 240), font_scale=0.5, color=(130, 150, 170))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    print(f"  Generated tampered sample: {output_path}")


def main():
    print("\n  Generating synthetic test documents...\n")

    data_path = Path("data")

    # Generate 3 genuine samples
    for i in range(1, 4):
        generate_genuine_sample(data_path / "genuine" / f"synthetic_genuine_{i:02d}.png", i)

    # Generate 3 tampered samples
    for i in range(1, 4):
        generate_tampered_sample(data_path / "tampered" / f"synthetic_tampered_{i:02d}.png", i)

    print(f"\n  Done! Test samples saved to data/genuine/ and data/tampered/")
    print(f"  Next steps:")
    print(f"    1. python train_classifier.py          # Train classifier")
    print(f"    2. python run_pipeline.py -i data/genuine/synthetic_genuine_01.png")
    print(f"    3. python run_pipeline.py -i data/tampered/synthetic_tampered_01.png")


if __name__ == "__main__":
    main()

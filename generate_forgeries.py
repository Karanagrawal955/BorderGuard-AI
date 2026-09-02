"""
generate_forgeries.py — Generate forged passport variants from genuine images

Creates controlled forgeries by applying realistic tampering operations:
    1. Text field modification (changed name, DOB, passport number)
    2. Photo region manipulation (shifted, resized, color-shifted)
    3. MRZ mismatch (MRZ doesn't match visual fields)
    4. Font size inconsistencies (pasted text at different sizes)
    5. Spacing anomalies (gaps between text blocks)
"""

import random
from pathlib import Path

import cv2
import numpy as np


def _text_tamper(img: np.ndarray) -> np.ndarray:
    """Modify text fields to create inconsistencies."""
    result = img.copy()
    h, w = result.shape[:2]

    # Change passport number area (overwrite with different text)
    # Find the passport number region (approximate)
    overlay = result.copy()
    cv2.rectangle(overlay, (250, 155), (480, 175), (30, 30, 40), -1)  # Cover old number
    result = cv2.addWeighted(overlay, 0.8, result, 0.2, 0)

    # Write new (forged) number with slightly different font
    fake_num = f"Passport No: FAKE{random.randint(1000, 9999)}"
    cv2.putText(result, fake_num, (250, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return result


def _photo_tamper(img: np.ndarray) -> np.ndarray:
    """Manipulate the photo region."""
    result = img.copy()
    h, w = result.shape[:2]

    # Photo region coordinates (must match generate_passports.py)
    photo_x, photo_y = 550, 100
    photo_w, photo_h = 200, 250

    # Extract photo region
    photo_region = result[photo_y:photo_y+photo_h, photo_x:photo_x+photo_w].copy()

    # Apply color shift (simulates pasting from different source)
    hsv = cv2.cvtColor(photo_region, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + 30) % 180  # Hue shift
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)  # Saturation boost
    photo_shifted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Slight resize (misalignment)
    scale = random.uniform(0.92, 1.08)
    new_w = int(photo_w * scale)
    new_h = int(photo_h * scale)
    photo_resized = cv2.resize(photo_shifted, (new_w, new_h))

    # Place back with offset
    offset_x = random.randint(-10, 10)
    offset_y = random.randint(-10, 10)
    y1 = max(0, photo_y + offset_y)
    y2 = min(h, y1 + new_h)
    x1 = max(0, photo_x + offset_x)
    x2 = min(w, x1 + new_w)

    result[y1:y2, x1:x2] = photo_resized[:y2-y1, :x2-x1]

    return result


def _font_inconsistency(img: np.ndarray) -> np.ndarray:
    """Add text at inconsistent font sizes."""
    result = img.copy()

    # Add an "annotation" at a different font size (simulating paste job)
    text = f"REVISED: {random.choice([' verified', ' approved', ' modified'])}"
    pos = (40, 465)
    cv2.putText(result, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    return result


def _spacing_anomaly(img: np.ndarray) -> np.ndarray:
    """Create unusual spacing gaps in the document."""
    result = img.copy()
    h, w = result.shape[:2]

    # Add a blank strip (simulates content deletion)
    strip_y = random.randint(200, 400)
    strip_h = random.randint(5, 15)
    result[strip_y:strip_y+strip_h, :] = 30

    return result


def _edge_artifacts(img: np.ndarray) -> np.ndarray:
    """Add visible edges around photo (cut-paste indicator)."""
    result = img.copy()

    photo_x, photo_y = 550, 100
    photo_w, photo_h = 200, 250

    # Draw sharp rectangle border (visible cut lines)
    cv2.rectangle(result, (photo_x-2, photo_y-2),
                  (photo_x+photo_w+2, photo_y+photo_h+2),
                  (255, 255, 255), 2)

    return result


def generate_forged_passport(
    genuine_path: Path,
    output_path: Path,
    tamper_types: list = None,
) -> None:
    """
    Generate a forged version of a genuine passport image.

    Applies 1-3 random tampering operations.
    """
    img = cv2.imread(str(genuine_path))
    if img is None:
        return

    # Available tampering operations
    tamper_ops = {
        "text": _text_tamper,
        "photo": _photo_tamper,
        "font": _font_inconsistency,
        "spacing": _spacing_anomaly,
        "edges": _edge_artifacts,
    }

    # Select random tampering operations
    if tamper_types is None:
        num_tampers = random.randint(1, 3)
        tamper_types = random.sample(list(tamper_ops.keys()), num_tampers)

    result = img.copy()
    for tamper_type in tamper_types:
        if tamper_type in tamper_ops:
            result = tamper_ops[tamper_type](result)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)


def generate_all_forgeries(
    genuine_dir: str = "dataset/genuine/passports",
    output_dir: str = "dataset/forged/passports",
    count: int = 50,
) -> None:
    """Generate forged variants from genuine passport images."""
    genuine_path = Path(genuine_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    genuine_files = sorted(genuine_path.glob("*.png"))

    if not genuine_files:
        print(f"  [WARN] No genuine images found in {genuine_dir}/")
        return

    generated = 0
    for i in range(count):
        # Cycle through genuine images if count > available
        source = genuine_files[i % len(genuine_files)]
        output_file = output_path / f"forged_{i+1:03d}.png"

        generate_forged_passport(source, output_file)
        generated += 1

    print(f"  [OK] Generated {generated} forged passport images in {output_dir}/")


if __name__ == "__main__":
    generate_all_forgeries()

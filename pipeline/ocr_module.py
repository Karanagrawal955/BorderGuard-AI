"""
ocr_module.py — EasyOCR text extraction + consistency checks

Extracts text from document images and applies rule-based consistency
checks to flag potential forgeries.

Checks implemented:
    1. Font-region uniformity: Bounding box height variance within lines
       (inconsistent heights suggest pasted text)
    2. Spacing anomalies: Unusual gaps between characters or words
    3. Verhoeff checksum: Validates Aadhaar-style 12-digit numbers

All checks degrade gracefully — if OCR fails or finds nothing, we
return "unreadable" status with lowered confidence.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Verhoeff algorithm for Aadhaar number validation
# ---------------------------------------------------------------------------

# Multiplication table (Verhoeff)
MULT_TABLE = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]

# Permutation table (Verhoeff)
PERM_TABLE = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]

# Inverse table (Verhoeff)
INV_TABLE = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_validate(number_str: str) -> bool:
    """
    Validate a number string using the Verhoeff checksum algorithm.
    Used for Aadhaar number validation (12-digit Indian ID numbers).

    Returns True if the checksum is valid, False otherwise.
    """
    # Remove spaces and dashes
    number_str = re.sub(r"[\s\-]", "", number_str)

    if not number_str.isdigit():
        return False

    checksum = 0
    for i, digit in enumerate(reversed(number_str)):
        d = int(digit)
        perm_idx = i % 8
        checksum = MULT_TABLE[checksum][PERM_TABLE[perm_idx][d]]

    return checksum == 0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OCRResult:
    """Stores OCR extraction results and consistency check flags."""
    text_blocks: List[dict] = field(default_factory=list)
    full_text: str = ""
    is_readable: bool = True
    consistency_flags: List[dict] = field(default_factory=list)
    checksum_valid: Optional[bool] = None
    aadhaar_found: str = ""
    overall_consistency: float = 1.0  # 1.0 = fully consistent, 0.0 = many issues

    @property
    def flag_count(self) -> int:
        return len(self.consistency_flags)


# ---------------------------------------------------------------------------
# OCR Processor
# ---------------------------------------------------------------------------

class OCRProcessor:
    """
    Wraps EasyOCR for document text extraction and consistency analysis.

    Usage:
        processor = OCRProcessor()
        result = processor.process(image_bgr)
        print(result.consistency_flags)
    """

    def __init__(self, languages: Optional[List[str]] = None):
        self.languages = languages or ["en"]
        self._reader = None
        self._initialize_reader()

    def _initialize_reader(self):
        """Lazy-load EasyOCR reader (it's slow to initialize)."""
        try:
            import easyocr
            self._reader = easyocr.Reader(self.languages, gpu=False, verbose=False)
            print("[OCRProcessor] EasyOCR initialized successfully")
        except ImportError:
            print("[OCRProcessor] WARNING: EasyOCR not installed. OCR will return empty.")
            print("  → Install with: pip install easyocr")
        except Exception as e:
            print(f"[OCRProcessor] WARNING: EasyOCR init failed: {e}")

    def process(self, image_bgr: np.ndarray) -> OCRResult:
        """
        Run OCR and consistency checks on a BGR image.

        Returns OCRResult with extracted text and flagged inconsistencies.
        Degrades gracefully if OCR is unavailable or finds no text.
        """
        result = OCRResult()

        if self._reader is None:
            result.is_readable = False
            result.consistency_flags.append({
                "check": "ocr_availability",
                "status": "FAIL",
                "message": "EasyOCR not available — cannot extract text",
            })
            return result

        try:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            ocr_output = self._reader.readtext(rgb)
        except Exception as e:
            result.is_readable = False
            result.consistency_flags.append({
                "check": "ocr_extraction",
                "status": "FAIL",
                "message": f"OCR extraction failed: {str(e)}",
            })
            return result

        if not ocr_output:
            result.is_readable = False
            result.consistency_flags.append({
                "check": "ocr_detection",
                "status": "WARN",
                "message": "No text detected in image",
            })
            return result

        # Parse OCR output: list of (bbox, text, confidence)
        text_blocks = []
        full_texts = []
        for bbox, text, conf in ocr_output:
            block = {
                "text": text,
                "confidence": float(conf),
                "bbox": bbox,  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            }
            text_blocks.append(block)
            full_texts.append(text)

        result.text_blocks = text_blocks
        result.full_text = " ".join(full_texts)

        # Run consistency checks
        self._check_font_uniformity(result, text_blocks)
        self._check_spacing(result, text_blocks)
        self._check_aadhaar(result, result.full_text)

        # Compute overall consistency score
        if result.consistency_flags:
            fail_count = sum(1 for f in result.consistency_flags if f["status"] == "FAIL")
            warn_count = sum(1 for f in result.consistency_flags if f["status"] == "WARN")
            result.overall_consistency = max(0.0, 1.0 - (fail_count * 0.3 + warn_count * 0.1))

        return result

    def _check_font_uniformity(self, result: OCRResult, text_blocks: List[dict]):
        """
        Check 1: Font-region uniformity
        Compares bounding box heights across detected text blocks.
        Inconsistent heights may indicate pasted/edited text regions.
        """
        if len(text_blocks) < 3:
            return  # Not enough blocks to compare

        heights = []
        for block in text_blocks:
            bbox = block["bbox"]
            # bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            if len(bbox) >= 4:
                y_coords = [pt[1] for pt in bbox]
                h = max(y_coords) - min(y_coords)
                heights.append(h)

        if not heights:
            return

        heights = np.array(heights)
        mean_h = np.mean(heights)
        std_h = np.std(heights)

        # Coefficient of variation — high CV means inconsistent font sizes
        cv = std_h / mean_h if mean_h > 0 else 0

        if cv > 0.4:
            result.consistency_flags.append({
                "check": "font_uniformity",
                "status": "FAIL",
                "message": f"Inconsistent font sizes detected (CV={cv:.2f}). "
                           f"Heights range from {min(heights):.0f} to {max(heights):.0f}px. "
                           f"Possible text paste operation.",
                "severity": min(1.0, cv),
            })
        elif cv > 0.25:
            result.consistency_flags.append({
                "check": "font_uniformity",
                "status": "WARN",
                "message": f"Slightly inconsistent font sizes (CV={cv:.2f}). "
                           f"May be normal variation or minor edit.",
                "severity": cv,
            })

    def _check_spacing(self, result: OCRResult, text_blocks: List[dict]):
        """
        Check 2: Spacing anomalies
        Detects unusual gaps between consecutive text blocks.
        Large unexpected gaps may indicate inserted/removed content.
        """
        if len(text_blocks) < 2:
            return

        # Sort blocks left-to-right, top-to-bottom
        sorted_blocks = sorted(text_blocks, key=lambda b: (
            min(pt[1] for pt in b["bbox"]),  # y-coordinate
            min(pt[0] for pt in b["bbox"]),  # x-coordinate
        ))

        gaps = []
        for i in range(1, len(sorted_blocks)):
            prev = sorted_blocks[i - 1]
            curr = sorted_blocks[i]

            # Right edge of previous block
            prev_right = max(pt[0] for pt in prev["bbox"])
            # Left edge of current block
            curr_left = min(pt[0] for pt in curr["bbox"])

            # Only measure horizontal gaps (same line)
            prev_bottom = max(pt[1] for pt in prev["bbox"])
            prev_top = min(pt[1] for pt in prev["bbox"])
            curr_bottom = max(pt[1] for pt in curr["bbox"])
            curr_top = min(pt[1] for pt in curr["bbox"])

            # Check if roughly on same line (vertical overlap > 50%)
            overlap_top = max(prev_top, curr_top)
            overlap_bottom = min(prev_bottom, curr_bottom)
            overlap = max(0, overlap_bottom - overlap_top)
            avg_height = ((prev_bottom - prev_top) + (curr_bottom - curr_top)) / 2

            if avg_height > 0 and overlap / avg_height > 0.3:
                gap = curr_left - prev_right
                gaps.append(gap)

        if not gaps:
            return

        gaps = np.array(gaps)
        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps)

        # Flag gaps that are > 2 standard deviations from mean
        anomaly_count = int(np.sum(np.abs(gaps - mean_gap) > 2 * std_gap))

        if anomaly_count > 0:
            result.consistency_flags.append({
                "check": "spacing_anomaly",
                "status": "WARN",
                "message": f"{anomaly_count} unusual spacing gap(s) detected "
                           f"(mean={mean_gap:.1f}px, std={std_gap:.1f}px). "
                           f"May indicate content insertion or removal.",
                "severity": min(1.0, anomaly_count / len(gaps)),
            })

    def _check_aadhaar(self, result: OCRResult, full_text: str):
        """
        Check 3: Aadhaar number detection and Verhoeff checksum validation.
        Looks for 12-digit numbers that could be Aadhaar IDs.
        """
        # Match 12-digit numbers (with optional spaces/dashes)
        aadhaar_pattern = r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"
        matches = re.findall(aadhaar_pattern, full_text)

        if not matches:
            return  # No Aadhaar-like number found

        for match in matches:
            clean = re.sub(r"[\s\-]", "", match)
            result.aadhaar_found = match

            if len(clean) == 12:
                is_valid = verhoeff_validate(clean)
                result.checksum_valid = is_valid

                if not is_valid:
                    result.consistency_flags.append({
                        "check": "aadhaar_checksum",
                        "status": "FAIL",
                        "message": f"Aadhaar number {match} failed Verhoeff checksum. "
                                   f"This is NOT a valid Aadhaar number.",
                        "severity": 0.9,
                    })
                else:
                    result.consistency_flags.append({
                        "check": "aadhaar_checksum",
                        "status": "PASS",
                        "message": f"Aadhaar number {match} passed Verhoeff checksum.",
                        "severity": 0.0,
                    })
            break  # Only check first Aadhaar number found

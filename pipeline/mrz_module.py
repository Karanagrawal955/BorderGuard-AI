"""
mrz_module.py — High-level MRZ extraction from document images

Provides extract_mrz() which is called by app.py to extract and validate
Machine Readable Zone (MRZ) data from uploaded document images.

This module wraps mrz_parser with image-level extraction:
    1. Try passporteye (specialized MRZ reader)
    2. Fall back to EasyOCR + pattern matching
    3. Return structured result dict for app.py compatibility
"""

import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from pipeline.mrz_parser import parse_mrz, find_mrz_in_text, MRZData


def _try_passporteye(image_path: str) -> Optional[dict]:
    """Try extracting MRZ using passporteye library."""
    try:
        from passporteye import read_mrz
        mrz = read_mrz(image_path, allow_layout=False)
        if mrz and mrz.is_valid:
            return {
                "found": True,
                "fields": {
                    "document_type": mrz.document_type or "",
                    "issuing_country": mrz.country or "",
                    "surname": mrz.surname or "",
                    "given_names": mrz.names or "",
                    "passport_number": mrz.number or "",
                    "nationality": mrz.nationality or "",
                    "date_of_birth": mrz.date_of_birth or "",
                    "sex": mrz.sex or "",
                    "date_of_expiry": mrz.expiry_date or "",
                    "personal_number": mrz.personal_number or "",
                },
                "raw_text": f"{mrz.mrz_type}\n{mrz.mrz_text}" if mrz.mrz_text else "",
                "method": "passporteye",
            }
    except ImportError:
        pass
    except Exception:
        pass
    return None


def _try_ocr_mrz(image_path: str) -> Optional[dict]:
    """Try extracting MRZ via EasyOCR + pattern matching."""
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)

        img = cv2.imread(image_path)
        if img is None:
            img_arr = np.fromfile(image_path, dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img is None:
            return None

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ocr_output = reader.readtext(rgb)

        if not ocr_output:
            return None

        # Combine all detected text
        all_text = "\n".join([text for _, text, _ in ocr_output])

        # Try to find MRZ lines in OCR output
        mrz_lines = find_mrz_in_text(all_text)
        if not mrz_lines:
            # Also try matching raw OCR blocks that look like MRZ
            for _, text, conf in ocr_output:
                cleaned = text.strip().upper()
                if (len(cleaned) >= 30 and
                    re.match(r'^[A-Z0-9<]+$', cleaned) and
                    ('<' in cleaned or re.match(r'^[PVDCA]<', cleaned))):
                    mrz_lines.append(cleaned)

        if len(mrz_lines) < 2:
            return None

        # Parse the MRZ lines
        parsed = parse_mrz(mrz_lines[:3])
        if parsed.is_valid or (parsed.passport_number and parsed.surname):
            return {
                "found": True,
                "fields": {
                    "document_type": parsed.document_type,
                    "issuing_country": parsed.issuing_country,
                    "surname": parsed.surname,
                    "given_names": parsed.given_names,
                    "passport_number": parsed.passport_number,
                    "nationality": parsed.nationality,
                    "date_of_birth": parsed.date_of_birth,
                    "sex": parsed.sex,
                    "date_of_expiry": parsed.date_of_expiry,
                    "personal_number": parsed.personal_number,
                },
                "raw_text": f"{parsed.raw_line1}\n{parsed.raw_line2}" if parsed.raw_line1 else "",
                "method": "ocr+parser",
            }

        return None
    except ImportError:
        pass
    except Exception:
        pass
    return None


def extract_mrz(image_path: str) -> dict:
    """
    Extract MRZ (Machine Readable Zone) data from a document image.

    Tries multiple methods in order of reliability:
        1. passporteye (fast, accurate for passports)
        2. EasyOCR + MRZ pattern matching (slower, more general)

    Args:
        image_path: Path to the document image file

    Returns:
        dict with keys:
            found   : bool — whether MRZ was found and parsed
            fields  : dict — parsed MRZ fields (name, DOB, passport number, etc.)
            raw_text: str — raw MRZ text as extracted
            method  : str — which extraction method succeeded
    """
    # Method 1: passporteye (most reliable for passports)
    result = _try_passporteye(image_path)
    if result:
        return result

    # Method 2: EasyOCR + pattern matching
    result = _try_ocr_mrz(image_path)
    if result:
        return result

    # Nothing found
    return {
        "found": False,
        "fields": {},
        "raw_text": "",
        "method": "none",
    }

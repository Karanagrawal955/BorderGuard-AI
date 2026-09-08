"""
document_detector.py — Detect whether an image contains a document

Checks:
1. Aspect ratio (documents are roughly rectangular)
2. Edge density (documents have sharp borders)
3. Text content (documents have text regions)
4. Layout structure (documents have structured layout)
5. Color distribution (documents have specific color patterns)
"""

import cv2
import numpy as np
from typing import Dict


def is_document(image_bgr: np.ndarray) -> Dict:
    """
    Analyze whether an image contains a document.
    Returns dict with:
        is_document: bool
        confidence: float 0-1
        reason: str
        scores: dict of individual checks
    """
    scores = {}

    # 1. Aspect ratio check — documents are roughly 3:2 to 16:9
    h, w = image_bgr.shape[:2]
    aspect = w / h if h > 0 else 1.0
    # Passports: ~1.4, ID cards: ~1.58, A4: ~1.41
    # Face photos: usually 1:1 to 3:4
    # Screenshots: very wide or very tall
    if 0.8 <= aspect <= 2.0:
        scores["aspect_ratio"] = 1.0
    elif 0.5 <= aspect <= 3.0:
        scores["aspect_ratio"] = 0.5
    else:
        scores["aspect_ratio"] = 0.0

    # 2. Edge density — documents have sharp rectangular borders
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))
    # Documents: moderate edge density (0.05-0.20)
    # Face photos: low edge density (<0.05)
    # Screenshots: high edge density (>0.20)
    if 0.03 <= edge_density <= 0.25:
        scores["edge_density"] = 1.0
    elif 0.01 <= edge_density <= 0.35:
        scores["edge_density"] = 0.5
    else:
        scores["edge_density"] = 0.0

    # 3. Text regions — documents have text, face photos don't
    # Use MSER (Maximally Stable Extremal Regions) to detect text-like regions
    text_score = _detect_text_regions(image_bgr)
    scores["text_content"] = text_score

    # 4. Color distribution — documents have specific color patterns
    # Passports/IDs: mostly white/cream with some color
    # Face photos: skin tones, varied background
    color_score = _check_document_colors(image_bgr)
    scores["color_distribution"] = color_score

    # 5. Layout structure — documents have structured layout
    layout_score = _check_layout_structure(image_bgr)
    scores["layout_structure"] = layout_score

    # 6. Face detection — if a face is the dominant content, it's likely NOT a document
    face_score = _check_face_dominance(image_bgr)
    scores["face_dominance"] = face_score

    # Weighted combination
    weights = {
        "aspect_ratio": 0.08,
        "edge_density": 0.08,
        "text_content": 0.40,  # Text is the strongest signal
        "color_distribution": 0.10,
        "layout_structure": 0.24,
        "face_dominance": 0.10,
    }

    total = sum(scores[k] * weights[k] for k in weights)

    # Require HIGH confidence to pass as document
    is_doc = total >= 0.55
    confidence = total

    # Build reason
    if not is_doc:
        reasons = []
        if scores["text_content"] < 0.3:
            reasons.append("no text detected")
        if scores["face_dominance"] < 0.3:
            reasons.append("face is dominant content")
        if scores["layout_structure"] < 0.3:
            reasons.append("no document layout")
        if not reasons:
            reasons.append("image does not match document characteristics")
        reason = "Not a document: " + ", ".join(reasons)
    else:
        reason = f"Document detected (confidence: {confidence:.0%})"

    return {
        "is_document": is_doc,
        "confidence": confidence,
        "reason": reason,
        "scores": scores,
    }


def _detect_text_regions(image_bgr: np.ndarray) -> float:
    """Detect text-like regions using MSER + geometry filters."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # MSER detector
    mser = cv2.MSER_create()
    mser.setMinArea(30)
    mser.setMaxArea(int(gray.shape[0] * gray.shape[1] * 0.1))

    try:
        regions, _ = mser.detectRegions(gray)
    except Exception:
        return 0.0

    if not regions:
        return 0.0

    # Filter for text-like regions: rectangular, similar height, aligned
    text_candidates = []
    for region in regions:
        x, y, w, h = cv2.boundingRect(region)
        aspect = w / (h + 1e-6)
        area = w * h

        # Text characters: roughly 0.3 < aspect < 3.0, small to medium size
        if 0.2 < aspect < 4.0 and 50 < area < gray.shape[0] * gray.shape[1] * 0.05:
            text_candidates.append((x, y, w, h))

    if not text_candidates:
        return 0.0

    # Check if text candidates form lines (horizontal alignment)
    if len(text_candidates) < 3:
        return 0.2

    # Group by y-coordinate (text lines)
    y_coords = [c[1] for c in text_candidates]
    y_sorted = sorted(y_coords)
    line_count = 1
    for i in range(1, len(y_sorted)):
        if y_sorted[i] - y_sorted[i-1] > 10:
            line_count += 1

    # Documents typically have multiple text lines
    if line_count >= 3:
        return min(0.5 + line_count * 0.05, 1.0)
    elif line_count >= 1:
        return 0.3
    else:
        return 0.1


def _check_document_colors(image_bgr: np.ndarray) -> float:
    """Check if color distribution matches document characteristics."""
    # Convert to HSV
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Documents typically have:
    # - Low saturation (white/cream background)
    # - High value (bright)
    # - Some colored elements (text, logos)

    mean_s = float(np.mean(s))
    mean_v = float(np.mean(v))

    # Check for document-like color profile
    score = 0.0

    # Low saturation = document-like (white/cream)
    if mean_s < 80:
        score += 0.4
    elif mean_s < 120:
        score += 0.2

    # High value = bright (documents are well-lit)
    if mean_v > 150:
        score += 0.3
    elif mean_v > 100:
        score += 0.15

    # Check for dominant white/cream region
    white_mask = (s < 50) & (v > 180)
    white_ratio = float(np.mean(white_mask))
    if white_ratio > 0.3:
        score += 0.3
    elif white_ratio > 0.1:
        score += 0.15

    return min(score, 1.0)


def _check_layout_structure(image_bgr: np.ndarray) -> float:
    """Check if image has document-like layout structure."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Detect horizontal and vertical lines (document borders, tables)
    # Horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 4, 1))
    horizontal = cv2.morphologyEx(
        cv2.Canny(gray, 50, 150), cv2.MORPH_CLOSE, horizontal_kernel
    )
    h_lines = float(np.mean(horizontal > 0))

    # Vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 4))
    vertical = cv2.morphologyEx(
        cv2.Canny(gray, 50, 150), cv2.MORPH_CLOSE, vertical_kernel
    )
    v_lines = float(np.mean(vertical > 0))

    # Documents have both horizontal and vertical structure
    score = 0.0
    if h_lines > 0.01:
        score += 0.3
    if v_lines > 0.01:
        score += 0.3
    if h_lines > 0.01 and v_lines > 0.01:
        score += 0.2  # Bonus for having both

    # Check for rectangular border
    # Documents often have a visible border/edge
    border_top = float(np.mean(gray[:5, :]))
    border_bottom = float(np.mean(gray[-5:, :]))
    border_left = float(np.mean(gray[:, :5]))
    border_right = float(np.mean(gray[:, -5:]))

    # If borders are darker than center, likely a document scan
    center = float(np.mean(gray[h//4:3*h//4, w//4:3*w//4]))
    borders_darker = (border_top < center - 10 or border_bottom < center - 10 or
                      border_left < center - 10 or border_right < center - 10)
    if borders_darker:
        score += 0.2

    return min(score, 1.0)


def _check_face_dominance(image_bgr: np.ndarray) -> float:
    """Check if a face dominates the image (low score = face dominates = not a document)."""
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return 0.8  # No face = likely document

        # Check if face dominates the image
        largest_face_area = 0
        for (x, y, w, h) in faces:
            face_area = w * h
            largest_face_area = max(largest_face_area, face_area)

        image_area = image_bgr.shape[0] * image_bgr.shape[1]
        face_ratio = largest_face_area / image_area

        # If face takes >20% of image, it's likely a face photo, not a document
        if face_ratio > 0.3:
            return 0.0  # Face dominates
        elif face_ratio > 0.15:
            return 0.3  # Significant face presence
        elif face_ratio > 0.05:
            return 0.5  # Small face (might be document with photo)
        else:
            return 0.8  # Tiny face or no face

    except Exception:
        return 0.5  # Can't determine

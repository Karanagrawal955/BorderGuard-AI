"""
preprocessing.py — Image loading, standardization, and ELA computation

ELA (Error Level Analysis) works by re-saving an image at a known JPEG quality
level and computing the pixel-wise difference. Tampered regions tend to show
higher residual errors because they were edited and re-encoded separately from
the original content.
"""

import io
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

# Standard input size for TruFor and downstream models
TARGET_SIZE = (512, 512)


def load_image(image_path: str | Path) -> np.ndarray:
    """
    Load an image from disk and return it as a BGR numpy array.
    Handles paths with Unicode characters (common on Windows).
    Falls back to PIL if OpenCV fails on non-ASCII paths.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Try OpenCV first (fast), fall back to PIL for Unicode paths
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("OpenCV returned None")
        return img
    except Exception:
        pil_img = Image.open(image_path).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def standardize_image(
    img: np.ndarray, target_size: Tuple[int, int] = TARGET_SIZE
) -> np.ndarray:
    """
    Resize image to target_size and normalize pixel values to [0, 1].
    Returns the resized BGR image (uint8, for display) and a float32
    normalized copy (for model input).
    """
    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
    return resized


def compute_ela(
    image_path: str | Path,
    quality: int = 90,
    target_size: Tuple[int, int] = TARGET_SIZE,
) -> Tuple[np.ndarray, dict]:
    """
    Compute Error Level Analysis map for a JPEG image.

    Process:
        1. Load original image
        2. Re-save at `quality` JPEG compression level (in-memory)
        3. Reload the re-saved version
        4. Compute absolute pixel difference → ELA map

    Returns:
        ela_map   : Grayscale numpy array (H, W) with ELA residuals
        ela_stats : dict with mean, variance, hotspot_count, max_error
    """
    img = load_image(image_path)

    # Re-encode at specified quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded = cv2.imencode(".jpg", img, encode_param)
    if not success:
        # If encoding fails (e.g., non-JPEG input), return zero ELA
        h, w = img.shape[:2]
        return np.zeros((h, w), dtype=np.uint8), {
            "mean": 0.0, "variance": 0.0, "hotspot_count": 0, "max_error": 0.0
        }

    resaved = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    # Ensure same dimensions for subtraction
    if resaved.shape != img.shape:
        resaved = cv2.resize(resaved, (img.shape[1], img.shape[0]))

    # Compute absolute difference
    diff = cv2.absdiff(img, resaved)
    ela_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Resize to target
    ela_resized = cv2.resize(ela_gray, target_size, interpolation=cv2.INTER_LANCZOS4)

    # Compute statistics
    ela_float = ela_resized.astype(np.float32)
    mean_val = float(np.mean(ela_float))
    var_val = float(np.var(ela_float))
    max_val = float(np.max(ela_float))

    # Hotspot: pixels with error > 2× mean (potential tampering indicators)
    threshold = mean_val * 2.0 if mean_val > 0 else 0
    hotspot_count = int(np.sum(ela_float > threshold))

    ela_stats = {
        "mean": mean_val,
        "variance": var_val,
        "hotspot_count": hotspot_count,
        "max_error": max_val,
        "hotspot_ratio": hotspot_count / (ela_resized.shape[0] * ela_resized.shape[1]),
    }

    return ela_resized, ela_stats


def preprocess_image(
    image_path: str | Path, target_size: Tuple[int, int] = TARGET_SIZE
) -> dict:
    """
    Full preprocessing pipeline for a single image.

    Returns dict with:
        original    : BGR numpy array (resized)
        ela_map     : Grayscale ELA residual map
        ela_stats   : ELA statistics dict
        image_path  : Source path string
    """
    img = load_image(image_path)
    resized = standardize_image(img, target_size)

    # Compute ELA from original path (to preserve JPEG compression artifacts)
    ela_map, ela_stats = compute_ela(image_path, target_size=target_size)

    return {
        "original": resized,
        "ela_map": ela_map,
        "ela_stats": ela_stats,
        "image_path": str(image_path),
    }

"""
preprocessing.py — Image loading, standardization, and ELA computation

Optimized for speed: avoids double-loading images, uses efficient ELA.
"""

import io
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
from PIL import Image

TARGET_SIZE = (512, 512)


def load_image(image_path: str | Path) -> np.ndarray:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("OpenCV returned None")
        return img
    except Exception:
        pil_img = Image.open(image_path).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def standardize_image(img: np.ndarray, target_size: Tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    return cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)


def compute_ela_from_image(
    img: np.ndarray,
    quality: int = 90,
    target_size: Tuple[int, int] = TARGET_SIZE,
) -> Tuple[np.ndarray, dict]:
    """
    Compute ELA from an already-loaded image (avoids re-reading from disk).
    """
    # Work at reduced size for speed
    work_size = (256, 256)
    img_small = cv2.resize(img, work_size, interpolation=cv2.INTER_LANCZOS4)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded = cv2.imencode(".jpg", img_small, encode_param)
    if not success:
        h, w = target_size
        return np.zeros((h, w), dtype=np.uint8), {
            "mean": 0.0, "variance": 0.0, "hotspot_count": 0, "max_error": 0.0, "hotspot_ratio": 0.0
        }

    resaved = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if resaved.shape != img_small.shape:
        resaved = cv2.resize(resaved, (img_small.shape[1], img_small.shape[0]))

    diff = cv2.absdiff(img_small, resaved)
    ela_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Resize to target
    ela_resized = cv2.resize(ela_gray, target_size, interpolation=cv2.INTER_LANCZOS4)

    ela_float = ela_resized.astype(np.float32)
    mean_val = float(np.mean(ela_float))
    var_val = float(np.var(ela_float))
    max_val = float(np.max(ela_float))
    threshold = mean_val * 2.0 if mean_val > 0 else 0
    hotspot_count = int(np.sum(ela_float > threshold))
    total_pixels = ela_resized.shape[0] * ela_resized.shape[1]

    ela_stats = {
        "mean": mean_val,
        "variance": var_val,
        "hotspot_count": hotspot_count,
        "max_error": max_val,
        "hotspot_ratio": hotspot_count / total_pixels,
    }

    return ela_resized, ela_stats


def preprocess_image(
    image_path: str | Path, target_size: Tuple[int, int] = TARGET_SIZE
) -> dict:
    """
    Full preprocessing pipeline — optimized to load image only once.
    """
    img = load_image(image_path)
    resized = standardize_image(img, target_size)
    ela_map, ela_stats = compute_ela_from_image(img, target_size=target_size)

    return {
        "original": resized,
        "ela_map": ela_map,
        "ela_stats": ela_stats,
        "image_path": str(image_path),
    }


# Backward compatibility alias
def compute_ela(image_path, quality=90, target_size=TARGET_SIZE):
    img = load_image(image_path)
    return compute_ela_from_image(img, quality=quality, target_size=target_size)

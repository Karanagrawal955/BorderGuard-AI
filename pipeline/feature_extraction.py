"""
feature_extraction.py — Compact feature vectors from TruFor + ELA

Combines:
    1. TruFor heatmap statistics (mean, std, max, hotspot ratio)
    2. ELA statistics (mean, variance, hotspot count, hotspot ratio)
    3. Texture features (Laplacian variance — blurriness indicator)
    4. JPEG compression artifacts (DCT-based quality estimate)

Output: A fixed-length feature vector per image, suitable for the
lightweight classifier head.
"""

from typing import Dict, Optional

import cv2
import numpy as np
import torch


def _trufor_features(heatmap: np.ndarray) -> np.ndarray:
    """
    Extract statistics from the TruFor manipulation heatmap.

    Features (5):
        - mean: average tampering probability
        - std: spread of tampering scores
        - max: peak tampering value
        - hotspot_ratio: fraction of pixels > 0.5 (likely tampered)
        - peak_concentration: how concentrated the hotspots are
    """
    hmean = float(np.mean(heatmap))
    hstd = float(np.std(heatmap))
    hmax = float(np.max(heatmap))

    hotspot_ratio = float(np.mean(heatmap > 0.5))

    # Peak concentration: ratio of mean in hot regions vs global mean
    hot_pixels = heatmap[heatmap > 0.5]
    if len(hot_pixels) > 0 and hmean > 0:
        peak_concentration = float(np.mean(hot_pixels)) / hmean
    else:
        peak_concentration = 0.0

    return np.array([hmean, hstd, hmax, hotspot_ratio, peak_concentration], dtype=np.float32)


def _ela_features(ela_map: np.ndarray, ela_stats: Dict) -> np.ndarray:
    """
    Extract features from the ELA residual map.

    Features (5):
        - mean: average error level (higher → more re-compression or tampering)
        - variance: spread of error levels
        - hotspot_ratio: fraction of high-error pixels
        - max_error: peak residual
        - edge_density: ELA residuals concentrated at edges (normal) vs scattered (suspicious)
    """
    ela_float = ela_map.astype(np.float32)

    mean_val = ela_stats.get("mean", float(np.mean(ela_float)))
    var_val = ela_stats.get("variance", float(np.var(ela_float)))
    max_val = ela_stats.get("max_error", float(np.max(ela_float)))
    hotspot_ratio = ela_stats.get("hotspot_ratio", 0.0)

    # Edge density: apply Canny edge detection to ELA map
    ela_uint8 = np.clip(ela_float * (255.0 / max(ela_float.max(), 1e-6)), 0, 255).astype(np.uint8)
    edges = cv2.Canny(ela_uint8, 50, 150)
    edge_density = float(np.mean(edges > 0))

    return np.array([mean_val, var_val, hotspot_ratio, max_val, edge_density], dtype=np.float32)


def _texture_features(image_bgr: np.ndarray) -> np.ndarray:
    """
    Extract texture/quality features from the original image.

    Features (3):
        - laplacian_var: blur detection (low = blurry, high = sharp)
        - saturation_mean: color saturation (forged regions sometimes have different saturation)
        - gradient_magnitude: edge strength indicator
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Laplacian variance (blur detection)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(np.var(laplacian))

    # Mean saturation
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat_mean = float(np.mean(hsv[:, :, 1]))

    # Gradient magnitude (Sobel)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = float(np.mean(np.sqrt(gx**2 + gy**2)))

    return np.array([lap_var, sat_mean, grad_mag], dtype=np.float32)


def _compression_features(image_bgr: np.ndarray) -> np.ndarray:
    """
    Estimate JPEG compression characteristics.

    Features (2):
        - estimated_quality: rough estimate based on DCT statistics
        - block_artifact_score: visible 8×8 block boundaries (indicates heavy re-compression)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Rough quality estimate via DCT energy ratio
    # High-quality JPEGs preserve more high-frequency DCT coefficients
    small = cv2.resize(gray, (128, 128))
    # Block-level variance (8×8 blocks)
    blocks = []
    for i in range(0, 128, 8):
        for j in range(0, 128, 8):
            block = small[i:i+8, j:j+8]
            blocks.append(np.var(block))
    block_var = np.mean(blocks) if blocks else 0.0

    # Block artifact: detect periodic pattern at 8-pixel intervals
    # This catches obvious JPEG re-compression artifacts
    diff_h = np.abs(gray[:, 1:] - gray[:, :-1])
    # Sample rows at 8-pixel intervals and check for increased differences
    periodic_score = 0.0
    for offset in range(0, min(gray.shape[0], 64), 8):
        row_diff = np.mean(diff_h[offset, :])
        periodic_score += row_diff
    periodic_score /= max(1, 64 // 8)

    return np.array([block_var, periodic_score], dtype=np.float32)


def extract_features(
    image_bgr: np.ndarray,
    heatmap: np.ndarray,
    ela_map: np.ndarray,
    ela_stats: Dict,
) -> dict:
    """
    Extract the complete feature vector for one image.

    Args:
        image_bgr : BGR numpy array (original image, any size)
        heatmap   : TruFor manipulation heatmap (H, W)
        ela_map   : ELA residual map (H, W)
        ela_stats : ELA statistics dict from preprocessing

    Returns dict with:
        features   : (15,) float32 numpy array — the combined feature vector
        feature_names : list of feature names for interpretability
    """
    feat_trufor = _trufor_features(heatmap)
    feat_ela = _ela_features(ela_map, ela_stats)
    feat_texture = _texture_features(image_bgr)
    feat_compression = _compression_features(image_bgr)

    features = np.concatenate([feat_trufor, feat_ela, feat_texture, feat_compression])

    feature_names = [
        # TruFor features (5)
        "trufor_mean", "trufor_std", "trufor_max",
        "trufor_hotspot_ratio", "trufor_peak_concentration",
        # ELA features (5)
        "ela_mean", "ela_variance", "ela_hotspot_ratio",
        "ela_max_error", "ela_edge_density",
        # Texture features (3)
        "texture_laplacian_var", "texture_saturation", "texture_gradient_mag",
        # Compression features (2)
        "compression_block_var", "compression_periodic_score",
    ]

    return {
        "features": features,
        "feature_names": feature_names,
    }

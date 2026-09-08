"""
feature_extraction.py — Feature vectors for document tampering detection

Uses only features that actually discriminate genuine from tampered documents:
1. Heatmap statistics (from trufor_detector)
2. Spatial ELA distribution (quadrant analysis)
3. Multi-quality ELA ratio
4. Noise consistency
5. Texture analysis
6. JPEG compression artifacts
7. Color channel consistency
"""

from typing import Dict

import cv2
import numpy as np


def _heatmap_features(heatmap: np.ndarray) -> np.ndarray:
    """Features from the tamper heatmap (3 features)."""
    hmean = float(np.mean(heatmap))
    hstd = float(np.std(heatmap))
    hotspot_ratio = float(np.mean(heatmap > 0.5))
    return np.array([hmean, hstd, hotspot_ratio], dtype=np.float32)


def _ela_spatial_features(ela_map: np.ndarray) -> np.ndarray:
    """Spatial ELA distribution (4 features)."""
    ela_float = ela_map.astype(np.float32)
    h, w = ela_float.shape

    # Quadrant analysis
    q_tl = np.mean(ela_float[:h//2, :w//2])
    q_tr = np.mean(ela_float[:h//2, w//2:])
    q_bl = np.mean(ela_float[h//2:, :w//2])
    q_br = np.mean(ela_float[h//2:, w//2:])
    quadrants = np.array([q_tl, q_tr, q_bl, q_br])

    asymmetry = float(np.std(quadrants) / (np.mean(quadrants) + 1e-6))
    concentration = float(np.max(quadrants) / (np.min(quadrants) + 1e-6))

    # Center vs border
    center = ela_float[h//4:3*h//4, w//4:3*w//4]
    border_mask = np.ones_like(ela_float)
    border_mask[h//4:3*h//4, w//4:3*w//4] = 0
    border = ela_float[border_mask > 0]
    center_mean = float(np.mean(center)) if center.size > 0 else 0
    border_mean = float(np.mean(border)) if border.size > 0 else 0
    center_border_ratio = center_mean / (border_mean + 1e-6)

    return np.array([asymmetry, concentration, center_border_ratio, center_mean], dtype=np.float32)


def _multi_quality_ela_features(image_bgr: np.ndarray) -> np.ndarray:
    """Compare ELA at different JPEG quality levels (3 features)."""
    small = cv2.resize(image_bgr, (256, 256), interpolation=cv2.INTER_LANCZOS4)

    _, enc50 = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    resaved50 = cv2.imdecode(enc50, cv2.IMREAD_COLOR)
    diff50 = cv2.cvtColor(cv2.absdiff(small, resaved50), cv2.COLOR_BGR2GRAY).astype(np.float32)

    _, enc95 = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    resaved95 = cv2.imdecode(enc95, cv2.IMREAD_COLOR)
    diff95 = cv2.cvtColor(cv2.absdiff(small, resaved95), cv2.COLOR_BGR2GRAY).astype(np.float32)

    ratio = np.where(diff95 > 1, diff50 / (diff95 + 1e-6), 0)
    ratio_mean = float(np.mean(ratio))
    ratio_std = float(np.std(ratio))
    ela_combined = float(np.mean(diff50)) / 255.0

    return np.array([ratio_mean, ratio_std, ela_combined], dtype=np.float32)


def _noise_features(image_bgr: np.ndarray) -> np.ndarray:
    """Noise analysis (3 features)."""
    gray_u8 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_f = gray_u8.astype(np.float32)

    laplacian = cv2.Laplacian(gray_u8, cv2.CV_64F)
    noise_estimate = float(np.median(np.abs(laplacian)))

    high_freq = cv2.GaussianBlur(gray_f, (5, 5), 1)
    hf_diff = gray_f - high_freq
    signal = float(np.mean(np.abs(gray_f)))
    noise = float(np.std(hf_diff))
    snr = signal / (noise + 1e-6)

    b, g, r = cv2.split(image_bgr)
    noise_b = float(np.std(cv2.Laplacian(b, cv2.CV_64F)))
    noise_g = float(np.std(cv2.Laplacian(g, cv2.CV_64F)))
    noise_r = float(np.std(cv2.Laplacian(r, cv2.CV_64F)))
    noise_inconsistency = float(max(noise_b, noise_g, noise_r) - min(noise_b, noise_g, noise_r))

    return np.array([noise_estimate, snr, noise_inconsistency], dtype=np.float32)


def _texture_features(image_bgr: np.ndarray) -> np.ndarray:
    """Texture features (4 features)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(np.var(laplacian))

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat_mean = float(np.mean(hsv[:, :, 1]))

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = float(np.mean(np.sqrt(gx**2 + gy**2)))

    # Local texture richness
    kernel = np.ones((5, 5), np.float32) / 25
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    local_sq_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
    texture_richness = float(np.mean(local_std))

    return np.array([lap_var, sat_mean, grad_mag, texture_richness], dtype=np.float32)


def _compression_features(image_bgr: np.ndarray) -> np.ndarray:
    """JPEG compression analysis (2 features)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    small = cv2.resize(gray, (128, 128))

    blocks = []
    for i in range(0, 128, 8):
        for j in range(0, 128, 8):
            block = small[i:i+8, j:j+8]
            blocks.append(np.var(block))
    block_var = np.mean(blocks) if blocks else 0.0

    diff_h = np.abs(gray[:, 1:] - gray[:, :-1])
    periodic_score = 0.0
    for offset in range(0, min(gray.shape[0], 64), 8):
        periodic_score += np.mean(diff_h[offset, :])
    periodic_score /= max(1, 64 // 8)

    return np.array([block_var, periodic_score], dtype=np.float32)


def extract_features(
    image_bgr: np.ndarray,
    heatmap: np.ndarray,
    ela_map: np.ndarray,
    ela_stats: Dict,
) -> dict:
    """
    Extract complete feature vector for one image.
    Returns dict with features (19,) and feature_names.
    """
    feat_hm = _heatmap_features(heatmap)
    feat_ela_spatial = _ela_spatial_features(ela_map)
    feat_mq_ela = _multi_quality_ela_features(image_bgr)
    feat_noise = _noise_features(image_bgr)
    feat_texture = _texture_features(image_bgr)
    feat_compression = _compression_features(image_bgr)

    features = np.concatenate([
        feat_hm, feat_ela_spatial, feat_mq_ela,
        feat_noise, feat_texture, feat_compression,
    ])

    feature_names = [
        "heatmap_mean", "heatmap_std", "heatmap_hotspot_ratio",
        "ela_spatial_asymmetry", "ela_spatial_concentration",
        "ela_spatial_center_border", "ela_spatial_center_mean",
        "mq_ela_ratio_mean", "mq_ela_ratio_std", "mq_ela_combined",
        "noise_estimate", "noise_snr", "noise_inconsistency",
        "texture_laplacian_var", "texture_saturation",
        "texture_gradient_mag", "texture_richness",
        "compression_block_var", "compression_periodic_score",
    ]

    return {
        "features": features,
        "feature_names": feature_names,
    }

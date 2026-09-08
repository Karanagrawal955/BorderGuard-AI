"""
tamper_detector.py — Document-agnostic tampering detection

Works on ANY document type (Aadhaar, passport, PAN, license, etc.)
by checking internal image consistency rather than classifying document type.

Detects:
1. Double JPEG compression (editing + re-save)
2. Noise inconsistency (pasted regions have different noise)
3. Edge discontinuity (cut-and-paste boundaries)
4. Color channel mismatch (different compression per channel)
5. ELA inconsistency (error level analysis across quality levels)
6. Grid alignment artifacts (8x8 DCT block misalignment)
"""

import cv2
import numpy as np
from typing import Dict


def detect_tampering(image_bgr: np.ndarray) -> Dict:
    """
    Analyze an image for signs of tampering.
    Returns a dict with individual scores and an overall tampering probability.
    All scores are 0.0 (no tampering) to 1.0 (definite tampering).
    """
    results = {}

    # 1. Double JPEG compression detection
    results["double_jpeg"] = _detect_double_jpeg(image_bgr)

    # 2. Noise inconsistency
    results["noise_inconsistency"] = _detect_noise_inconsistency(image_bgr)

    # 3. Edge discontinuity
    results["edge_discontinuity"] = _detect_edge_discontinuity(image_bgr)

    # 4. Color channel mismatch
    results["color_mismatch"] = _detect_color_mismatch(image_bgr)

    # 5. ELA inconsistency
    results["ela_inconsistency"] = _detect_ela_inconsistency(image_bgr)

    # 6. Grid alignment
    results["grid_alignment"] = _detect_grid_alignment(image_bgr)

    # 7. Texture uniformity (tampering breaks texture consistency)
    results["texture_inconsistency"] = _detect_texture_inconsistency(image_bgr)

    # 8. Compression quality variation
    results["compression_variation"] = _detect_compression_variation(image_bgr)

    # Weighted combination — but require MULTIPLE signals to agree
    # Single high scores are normal for documents (edges, texture variation)
    # Tampering shows MULTIPLE high scores simultaneously
    weights = {
        "double_jpeg": 0.25,
        "noise_inconsistency": 0.20,
        "edge_discontinuity": 0.05,   # Low weight: docs naturally have edge variation
        "color_mismatch": 0.10,
        "ela_inconsistency": 0.20,
        "grid_alignment": 0.10,
        "texture_inconsistency": 0.05,  # Low weight: docs naturally have texture variation
        "compression_variation": 0.05,
    }

    # Count how many signals are "high" (> 0.6)
    high_signals = sum(1 for k in weights if results[k] > 0.6)
    
    # Weighted average
    weighted_avg = sum(results[k] * weights[k] for k in weights)
    
    # Tampering requires at least 3 high signals OR very high weighted average
    if high_signals >= 3:
        overall = min(weighted_avg * 1.3, 1.0)  # Amplify when multiple signals agree
    elif high_signals >= 2:
        overall = weighted_avg
    else:
        overall = weighted_avg * 0.6  # Suppress when few signals agree
    
    overall = max(0.0, min(1.0, overall))

    results["overall_tampering_probability"] = overall
    results["weights"] = weights

    return results


def _detect_double_jpeg(image_bgr: np.ndarray) -> float:
    """
    Detect double JPEG compression.
    If an image was saved at Q1, edited, then saved at Q2, the DCT coefficients
    show periodic artifacts at the quantization grid boundaries.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    small = cv2.resize(gray, (256, 256))

    # Compute DCT of 8x8 blocks
    block_dcts = []
    for i in range(0, 256, 8):
        for j in range(0, 256, 8):
            block = small[i:i+8, j:j+8]
            # Simple DCT proxy: check variance pattern
            block_dcts.append(np.var(block))

    if not block_dcts:
        return 0.0

    block_array = np.array(block_dcts).reshape(32, 32)

    # Check for periodic pattern at 8-pixel intervals
    # Double JPEG creates correlations at block boundaries
    h_corr = np.mean(np.abs(block_array[:, 1:] - block_array[:, :-1]))
    v_corr = np.mean(np.abs(block_array[1:, :] - block_array[:-1, :]))

    # Also check the actual DCT coefficient quantization
    # Re-encode at different qualities and measure inconsistency
    q50_err = _block_error_at_quality(small, 50)
    q75_err = _block_error_at_quality(small, 75)
    q90_err = _block_error_at_quality(small, 90)

    # Genuine: consistent error progression
    # Tampered: inconsistent (some blocks already compressed differently)
    if q50_err > 0:
        ratio_50_75 = q50_err / (q75_err + 1e-6)
        ratio_75_90 = q75_err / (q90_err + 1e-6)
        inconsistency = abs(ratio_50_75 - ratio_75_90) / max(ratio_50_75, ratio_75_90, 1e-6)
    else:
        inconsistency = 0.0

    # Combine signals
    score = 0.5 * min(inconsistency * 2, 1.0) + 0.5 * min((h_corr + v_corr) / 20, 1.0)
    return max(0.0, min(1.0, score))


def _block_error_at_quality(gray_small: np.ndarray, quality: int) -> float:
    """Compute average block-level ELA error at a specific JPEG quality."""
    img_uint8 = np.clip(gray_small, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded = cv2.imencode(".jpg", img_bgr, encode_param)
    if not success:
        return 0.0

    resaved = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if resaved is None:
        return 0.0

    resaved_gray = cv2.cvtColor(resaved, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = np.abs(gray_small - resaved_gray)

    # Block-wise mean error
    block_errors = []
    for i in range(0, gray_small.shape[0], 8):
        for j in range(0, gray_small.shape[1], 8):
            block = diff[i:i+8, j:j+8]
            if block.size > 0:
                block_errors.append(np.mean(block))

    return float(np.mean(block_errors)) if block_errors else 0.0


def _detect_noise_inconsistency(image_bgr: np.ndarray) -> float:
    """
    Detect noise pattern inconsistency.
    Genuine images have uniform noise. Tampered regions have different noise
    (from different camera, compression, or source).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Estimate local noise using Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    abs_lap = np.abs(lap)

    # Divide into blocks and compute noise level
    h, w = abs_lap.shape
    block_size = 32
    noise_levels = []

    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = abs_lap[i:i+block_size, j:j+block_size]
            noise_levels.append(float(np.std(block)))

    if len(noise_levels) < 4:
        return 0.0

    noise_array = np.array(noise_levels)
    noise_mean = np.mean(noise_array)
    noise_std = np.std(noise_array)

    # High variation in noise = suspicious
    if noise_mean > 0:
        cv = noise_std / noise_mean  # Coefficient of variation
    else:
        cv = 0.0

    # Also check for bimodal distribution (two different noise sources)
    # This happens when genuine + tampered regions coexist
    sorted_noise = np.sort(noise_array)
    n = len(sorted_noise)
    gap_scores = []
    for i in range(1, n):
        gap = sorted_noise[i] - sorted_noise[i-1]
        gap_scores.append(gap)

    max_gap = max(gap_scores) if gap_scores else 0
    gap_ratio = max_gap / (noise_std + 1e-6)

    score = 0.6 * min(cv * 3, 1.0) + 0.4 * min(gap_ratio / 5, 1.0)
    return max(0.0, min(1.0, score))


def _detect_edge_discontinuity(image_bgr: np.ndarray) -> float:
    """
    Detect edge discontinuity at tampering boundaries.
    When regions are pasted from different sources, edges don't align smoothly.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Compute edges using Canny
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150).astype(np.float32)

    # Compute gradient direction consistency
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx)

    # Block-wise gradient direction variance
    h, w = gray.shape
    block_size = 32
    dir_variances = []

    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            mag_block = magnitude[i:i+block_size, j:j+block_size]
            dir_block = direction[i:i+block_size, j:j+block_size]

            # Only consider pixels with significant gradient
            mask = mag_block > np.percentile(mag_block, 70)
            if np.sum(mask) > 10:
                dir_variances.append(float(np.std(dir_block[mask])))

    if not dir_variances:
        return 0.0

    dir_var_array = np.array(dir_variances)
    # High variance in gradient direction = edge discontinuity
    mean_var = np.mean(dir_var_array)
    score = min(mean_var / 1.5, 1.0)

    return max(0.0, min(1.0, score))


def _detect_color_mismatch(image_bgr: np.ndarray) -> float:
    """
    Detect color channel mismatch.
    Tampered regions may have different color balance than the original.
    """
    b, g, r = cv2.split(image_bgr)

    # Block-wise color channel ratios
    h, w = b.shape
    block_size = 32
    rg_ratios = []
    bg_ratios = []

    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            b_mean = float(np.mean(b[i:i+block_size, j:j+block_size]))
            g_mean = float(np.mean(g[i:i+block_size, j:j+block_size]))
            r_mean = float(np.mean(r[i:i+block_size, j:j+block_size]))

            if g_mean > 0 and b_mean > 0:
                rg_ratios.append(r_mean / g_mean)
                bg_ratios.append(b_mean / g_mean)

    if len(rg_ratios) < 4:
        return 0.0

    rg_array = np.array(rg_ratios)
    bg_array = np.array(bg_ratios)

    # High variation in color ratios = suspicious
    rg_cv = np.std(rg_array) / (np.mean(rg_array) + 1e-6)
    bg_cv = np.std(bg_array) / (np.mean(bg_array) + 1e-6)

    score = min((rg_cv + bg_cv) / 2, 1.0)
    return max(0.0, min(1.0, score))


def _detect_ela_inconsistency(image_bgr: np.ndarray) -> float:
    """
    Detect ELA inconsistency across quality levels.
    Genuine images have consistent ELA patterns.
    Tampered regions show abnormal ELA at different quality levels.
    """
    small = cv2.resize(image_bgr, (256, 256), interpolation=cv2.INTER_LANCZOS4)

    # ELA at multiple quality levels
    errors = {}
    for q in [30, 50, 70, 90]:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
        success, encoded = cv2.imencode(".jpg", small, encode_param)
        if success:
            resaved = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if resaved is not None and resaved.shape == small.shape:
                diff = cv2.cvtColor(cv2.absdiff(small, resaved), cv2.COLOR_BGR2GRAY)
                errors[q] = float(np.mean(diff))

    if len(errors) < 3:
        return 0.0

    # Check if error progression is consistent
    # Genuine: error decreases smoothly with quality
    # Tampered: inconsistent progression
    qualities = sorted(errors.keys())
    error_progression = [errors[q] for q in qualities]

    # Compute second derivative (acceleration of error change)
    if len(error_progression) >= 3:
        second_deriv = []
        for i in range(1, len(error_progression) - 1):
            d2 = error_progression[i+1] - 2*error_progression[i] + error_progression[i-1]
            second_deriv.append(abs(d2))

        mean_second_deriv = np.mean(second_deriv)
        # Normalize: typical range 0-5
        score = min(mean_second_deriv / 3, 1.0)
    else:
        score = 0.0

    return max(0.0, min(1.0, score))


def _detect_grid_alignment(image_bgr: np.ndarray) -> float:
    """
    Detect 8x8 DCT grid alignment artifacts.
    When images are edited and re-saved, the JPEG grid may not align properly.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    small = cv2.resize(gray, (256, 256))

    # Check difference at 8-pixel boundaries
    # Compute horizontal and vertical differences
    diff_h = np.abs(small[:, 1:] - small[:, :-1])
    diff_v = np.abs(small[1:, :] - small[:-1, :])

    # Sample at 8-pixel boundaries vs non-boundaries
    boundary_h = []
    non_boundary_h = []
    for offset in range(0, min(256, small.shape[0]), 8):
        if offset < diff_h.shape[0]:
            boundary_h.append(np.mean(diff_h[offset, :]))
        if offset + 4 < diff_h.shape[0]:
            non_boundary_h.append(np.mean(diff_h[offset + 4, :]))

    boundary_v = []
    non_boundary_v = []
    for offset in range(0, min(256, small.shape[1]), 8):
        if offset < diff_v.shape[1]:
            boundary_v.append(np.mean(diff_v[:, offset]))
        if offset + 4 < diff_v.shape[1]:
            non_boundary_v.append(np.mean(diff_v[:, offset + 4]))

    if not boundary_h or not non_boundary_h:
        return 0.0

    # If boundaries have significantly more/less difference than non-boundaries,
    # the image has been re-compressed (alignment artifact)
    bh_mean = np.mean(boundary_h)
    nbh_mean = np.mean(non_boundary_h)
    bv_mean = np.mean(boundary_v)
    nbv_mean = np.mean(non_boundary_v)

    if nbh_mean > 0 and nbv_mean > 0:
        ratio_h = bh_mean / nbh_mean
        ratio_v = bv_mean / nbv_mean
        # Perfect alignment: ratio = 1.0
        # Misalignment: ratio deviates from 1.0
        deviation = abs(ratio_h - 1.0) + abs(ratio_v - 1.0)
        score = min(deviation / 0.5, 1.0)
    else:
        score = 0.0

    return max(0.0, min(1.0, score))


def _detect_texture_inconsistency(image_bgr: np.ndarray) -> float:
    """
    Detect texture inconsistency.
    Tampered regions often have different texture characteristics.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Local Binary Pattern approximation using Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    abs_lap = np.abs(lap)

    # Block-wise texture features
    h, w = abs_lap.shape
    block_size = 32
    texture_means = []
    texture_stds = []

    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = abs_lap[i:i+block_size, j:j+block_size]
            texture_means.append(float(np.mean(block)))
            texture_stds.append(float(np.std(block)))

    if len(texture_means) < 4:
        return 0.0

    means = np.array(texture_means)
    stds = np.array(texture_stds)

    # High variation in texture = suspicious
    mean_cv = np.std(means) / (np.mean(means) + 1e-6)
    std_cv = np.std(stds) / (np.mean(stds) + 1e-6)

    score = min((mean_cv + std_cv) / 4, 1.0)
    return max(0.0, min(1.0, score))


def _detect_compression_variation(image_bgr: np.ndarray) -> float:
    """
    Detect compression quality variation across the image.
    When different regions have been saved at different quality levels.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Block-wise DCT energy variance
    h, w = gray.shape
    block_size = 8
    energies = []

    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            # DCT energy proxy
            energy = float(np.sum(block ** 2))
            energies.append(energy)

    if len(energies) < 10:
        return 0.0

    energy_array = np.array(energies)
    # Check for bimodal distribution (two different compression levels)
    sorted_e = np.sort(energy_array)
    n = len(sorted_e)

    # Find the largest gap in sorted energies
    gaps = np.diff(sorted_e)
    max_gap_idx = np.argmax(gaps)
    max_gap = gaps[max_gap_idx]

    # Normalize by energy range
    energy_range = sorted_e[-1] - sorted_e[0]
    if energy_range > 0:
        gap_ratio = max_gap / energy_range
    else:
        gap_ratio = 0.0

    # Bimodal = high gap_ratio
    score = min(gap_ratio * 3, 1.0)
    return max(0.0, min(1.0, score))

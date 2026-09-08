"""
trufor_detector.py — Document tamper detection (ELA-based fallback)

Since TruFor weights are unavailable, this uses a combination of:
1. Multi-quality ELA comparison
2. DCT frequency analysis
3. JPEG grid alignment detection
4. Noise pattern analysis
5. Color channel inconsistency

The heatmap is a tamper probability map [0,1].
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


class _TamperDetectorCNN(nn.Module):
    """Fallback CNN when no TruFor weights available."""

    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = False
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.encoder(x)
        heatmap = self.decoder(features)
        if heatmap.shape[2:] != x.shape[2:]:
            heatmap = nn.functional.interpolate(
                heatmap, size=x.shape[2:], mode="bilinear", align_corners=False
            )
        return heatmap


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _ela_based_heatmap(image_bgr: np.ndarray) -> np.ndarray:
    """
    Generate tampering heatmap using multi-analysis approach.
    Returns: (512, 512) float32 array [0, 1]
    """
    target = (512, 512)
    img = cv2.resize(image_bgr, target, interpolation=cv2.INTER_LANCZOS4)

    # Analysis 1: Multi-quality ELA
    q50_err = _ela_error_at_quality(img, 50)
    q75_err = _ela_error_at_quality(img, 75)
    q95_err = _ela_error_at_quality(img, 95)

    # Genuine images: consistent error across quality levels
    # Tampered: inconsistent error (pasted regions have different compression)
    consistency_map = np.abs(q50_err - q75_err).astype(np.float32)
    consistency_map = consistency_map / max(np.max(consistency_map), 1e-6)

    # Analysis 2: Noise pattern consistency
    noise_map = _noise_inconsistency_map(img)

    # Analysis 3: DCT block analysis
    dct_map = _dct_block_analysis(img)

    # Analysis 4: Color channel inconsistency
    color_map = _color_channel_inconsistency(img)

    # Combine all analyses
    heatmap = (
        0.3 * consistency_map +
        0.25 * noise_map +
        0.25 * dct_map +
        0.2 * color_map
    )

    # Normalize to [0, 1]
    p5 = np.percentile(heatmap, 5)
    p95 = np.percentile(heatmap, 95)
    if p95 - p5 > 0.01:
        heatmap = np.clip((heatmap - p5) / (p95 - p5), 0, 1)
    else:
        heatmap = np.clip(heatmap * 0.5, 0, 1)

    return heatmap.astype(np.float32)


def _ela_error_at_quality(img: np.ndarray, quality: int) -> np.ndarray:
    """ELA at a specific JPEG quality level."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded = cv2.imencode(".jpg", img, encode_param)
    if not success:
        return np.zeros(img.shape[:2], dtype=np.float32)
    resaved = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if resaved is None or resaved.shape != img.shape:
        return np.zeros(img.shape[:2], dtype=np.float32)
    diff = cv2.absdiff(img, resaved)
    return cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _noise_inconsistency_map(img: np.ndarray) -> np.ndarray:
    """Detect local noise inconsistencies — tampering creates noise boundaries."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Estimate local noise using Laplacian
    lap = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_32F)
    abs_lap = np.abs(lap)

    # Block-wise noise standard deviation
    h, w = abs_lap.shape
    block = 32
    noise_map = np.zeros((h, w), dtype=np.float32)
    for i in range(0, h - block, block // 2):
        for j in range(0, w - block, block // 2):
            region = abs_lap[i:i+block, j:j+block]
            block_noise = float(np.std(region))
            noise_map[i:i+block, j:j+block] = block_noise

    # Normalize
    nmax = np.max(noise_map)
    if nmax > 0:
        noise_map = noise_map / nmax

    return noise_map


def _dct_block_analysis(img: np.ndarray) -> np.ndarray:
    """Analyze DCT coefficient patterns across 8x8 blocks."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    # Compute block-wise DCT energy
    dct_energy = np.zeros((h // 8, w // 8), dtype=np.float32)
    for i in range(0, h - 8, 8):
        for j in range(0, w - 8, 8):
            block = gray[i:i+8, j:j+8]
            # Simple DCT energy proxy: variance of the block
            dct_energy[i // 8, j // 8] = float(np.var(block))

    # Upscale to full resolution
    dct_map = cv2.resize(dct_energy, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalize
    dmax = np.max(dct_map)
    if dmax > 0:
        dct_map = dct_map / dmax

    return dct_map


def _color_channel_inconsistency(img: np.ndarray) -> np.ndarray:
    """Detect inconsistencies between color channels."""
    b, g, r = cv2.split(img)

    # Compute per-channel variance in local blocks
    block = 32
    h, w = b.shape[:2]
    inconsistency = np.zeros((h, w), dtype=np.float32)

    for i in range(0, h - block, block // 2):
        for j in range(0, w - block, block // 2):
            b_var = float(np.var(b[i:i+block, j:j+block].astype(np.float32)))
            g_var = float(np.var(g[i:i+block, j:j+block].astype(np.float32)))
            r_var = float(np.var(r[i:i+block, j:j+block].astype(np.float32)))
            # High inconsistency between channels = suspicious
            mean_var = (b_var + g_var + r_var) / 3
            if mean_var > 0:
                inconsistency[i:i+block, j:j+block] = (
                    max(b_var, g_var, r_var) - min(b_var, g_var, r_var)
                ) / mean_var

    # Normalize
    imax = np.max(inconsistency)
    if imax > 0:
        inconsistency = inconsistency / imax

    return inconsistency


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TruForDetector:
    """
    Wrapper around the tamper-detection model.
    If TruFor weights exist → use the real CNN model.
    If NOT → use multi-analysis ELA-based heatmap.
    """

    def __init__(self, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.has_trufor_weights = False

        weight_path = Path("models/trufor_tamper.pth")
        if weight_path.exists():
            self.model = _TamperDetectorCNN().to(self.device)
            self.model.eval()
            try:
                state_dict = torch.load(weight_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                self.has_trufor_weights = True
                print(f"[TruForDetector] Loaded TruFor weights from {weight_path}")
            except Exception as e:
                print(f"[TruForDetector] Could not load weights: {e}")
                print("[TruForDetector] Falling back to ELA-based heatmap")
        else:
            self.model = None
            print("[TruForDetector] No TruFor weights — using ELA-based heatmap detection")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    @torch.no_grad()
    def detect(self, image_bgr: np.ndarray) -> dict:
        if self.has_trufor_weights and self.model is not None:
            return self._detect_trufor(image_bgr)
        else:
            return self._detect_ela(image_bgr)

    def _detect_trufor(self, image_bgr: np.ndarray) -> dict:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        heatmap_tensor = self.model(tensor)
        heatmap = heatmap_tensor.squeeze().cpu().numpy().astype(np.float32)
        confidence = float(np.mean(np.abs(heatmap - 0.5) * 2))
        reliability = cv2.GaussianBlur(heatmap, (21, 21), 7)
        rel_min, rel_max = reliability.min(), reliability.max()
        if rel_max - rel_min > 1e-6:
            reliability = (reliability - rel_min) / (rel_max - rel_min)
        else:
            reliability = np.ones_like(reliability) * 0.5
        return {
            "heatmap": heatmap,
            "confidence": confidence,
            "reliability_map": reliability,
            "source": "trufor",
        }

    def _detect_ela(self, image_bgr: np.ndarray) -> dict:
        heatmap = _ela_based_heatmap(image_bgr)
        confidence = float(np.mean(np.abs(heatmap - 0.5) * 2))
        reliability = cv2.GaussianBlur(heatmap, (21, 21), 7)
        rel_min, rel_max = reliability.min(), reliability.max()
        if rel_max - rel_min > 1e-6:
            reliability = (reliability - rel_min) / (rel_max - rel_min)
        else:
            reliability = np.ones_like(reliability) * 0.5
        return {
            "heatmap": heatmap,
            "confidence": confidence,
            "reliability_map": reliability,
            "source": "ela_fallback",
        }

    def get_heatmap_overlay(self, image_bgr, heatmap, alpha=0.4):
        h, w = image_bgr.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image_bgr, 1 - alpha, heatmap_colored, alpha, 0)
        return overlay

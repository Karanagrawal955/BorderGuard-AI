"""
classifier.py — Lightweight tamper classifier trained on dataset images.

Uses block-level ELA features + noise + color analysis.
Trains on startup from data/genuine/ and data/tampered/ folders.
"""

import os
import pickle
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple

_MODEL: Optional[object] = None
_SCALER: Optional[object] = None
_FEATURE_NAMES: list = []


def _extract_features(img: np.ndarray) -> Optional[np.ndarray]:
    if img is None or len(img.shape) != 3:
        return None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = []

    for quality in [30, 50, 70, 90]:
        _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        resaved = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        ela = np.mean(np.abs(img.astype(float) - resaved.astype(float)))
        features.append(ela)

    _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    resaved = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    ela_map = np.mean(np.abs(img.astype(float) - resaved.astype(float)), axis=2)

    block_size = 32
    block_elas = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = ela_map[y:y + block_size, x:x + block_size]
            block_elas.append(np.mean(block))

    block_elas = np.array(block_elas) if block_elas else np.array([0.0])
    features.extend([
        np.mean(block_elas),
        np.std(block_elas),
        np.max(block_elas),
        np.min(block_elas),
        np.percentile(block_elas, 90),
        np.percentile(block_elas, 10),
        float(np.mean(((block_elas - np.mean(block_elas)) / (np.std(block_elas) + 1e-6)) ** 3)),
    ])

    for ksize in [3, 5, 7]:
        blur = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        noise = gray.astype(float) - blur.astype(float)
        features.extend([np.mean(np.abs(noise)), np.std(noise)])

    for c in range(3):
        channel_ela = np.mean(np.abs(img[:, :, c].astype(float) - resaved[:, :, c].astype(float)))
        features.append(channel_ela)

    gx = cv2.Sobel(ela_map, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(ela_map, cv2.CV_64F, 0, 1, ksize=3)
    grad_ela = np.sqrt(gx ** 2 + gy ** 2)
    features.extend([np.mean(grad_ela), np.std(grad_ela)])

    ela_small = cv2.resize(ela_map, (256, 256)).astype(np.float32)
    dct = cv2.dct(ela_small)
    h2, w2 = dct.shape
    low_mask = np.zeros_like(dct)
    low_mask[:h2 // 4, :w2 // 4] = 1
    total_e = np.sum(np.abs(dct))
    low_e = np.sum(np.abs(dct) * low_mask)
    features.append(low_e / (total_e + 1e-6))

    return np.array(features, dtype=np.float32)


def _collect_images(folder: str) -> list:
    """Recursively collect image paths from a folder."""
    paths = []
    root = Path(folder)
    if not root.is_dir():
        return paths
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in exts and p.is_file():
            paths.append(str(p))
    return paths


def train(data_dir: str = None) -> dict:
    global _MODEL, _SCALER, _FEATURE_NAMES

    if data_dir is None:
        data_dir = str(Path(__file__).parent.parent / "data")

    # Train on every labelled corpus we have — the small data/ folder alone
    # (41 images) gave a noisy ~80% model; adding dataset/ (500 images)
    # pushes held-out accuracy much higher.
    pairs = [(data_dir + "/genuine", data_dir + "/tampered"),
             ("dataset/genuine", "dataset/forged")]
    if data_dir != str(Path(__file__).parent.parent / "data"):
        pairs = [(data_dir + "/genuine", data_dir + "/tampered")]

    X, y = [], []
    per_folder = {}
    for genuine_dir, tampered_dir in pairs:
        for folder, label in [(genuine_dir, 0), (tampered_dir, 1)]:
            files = _collect_images(folder)
            per_folder[folder] = len(files)
            for img_path in files:
                img = cv2.imread(img_path)
                if img is None:
                    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    feat = _extract_features(img)
                    if feat is not None:
                        X.append(feat)
                        y.append(label)

    if len(X) < 10:
        print("[classifier] Not enough data to train (%d samples)" % len(X))
        return {"status": "insufficient_data", "samples": len(X)}

    X = np.array(X)
    y = np.array(y)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    _SCALER = StandardScaler()
    Xs = _SCALER.fit_transform(X)

    _MODEL = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    _MODEL.fit(Xs, y)

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(_MODEL, Xs, y, cv=min(5, len(X) // 2), scoring="accuracy")

    return {
        "status": "trained",
        "samples": len(X),
        "genuine": int(np.sum(y == 0)),
        "tampered": int(np.sum(y == 1)),
        "cv_accuracy": float(scores.mean()),
        "cv_std": float(scores.std()),
    }


def predict(img: np.ndarray) -> dict:
    global _MODEL, _SCALER

    if _MODEL is None or _SCALER is None:
        return {
            "probability": 0.5,
            "prediction": 0,
            "confidence": 0.0,
            "model_loaded": False,
        }

    feat = _extract_features(img)
    if feat is None:
        return {
            "probability": 0.5,
            "prediction": 0,
            "confidence": 0.0,
            "model_loaded": True,
            "error": "feature_extraction_failed",
        }

    Xs = _SCALER.transform(feat.reshape(1, -1))
    prob = float(_MODEL.predict_proba(Xs)[0, 1])
    pred = int(prob > 0.5)
    conf = 1.0 - abs(prob - 0.5) * 2

    return {
        "probability": prob,
        "prediction": pred,
        "confidence": float(conf),
        "model_loaded": True,
    }


def is_loaded() -> bool:
    return _MODEL is not None and _SCALER is not None

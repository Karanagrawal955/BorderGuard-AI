"""
trufor_detector.py — Real TruFor inference via local Docker HTTP server

This calls the actual TruFor model (real pretrained weights, running on
GPU inside a Docker container in WSL2) through a local HTTP API, instead
of the earlier random-initialized CNN stand-in.

PREREQUISITE: the TruFor Docker container's server.py must be running
(listening on localhost:8000) before this module is used. See
SETUP_TRUFOR.md for how to start it.

If the server isn't reachable, this degrades gracefully: it returns a
neutral, clearly-flagged heatmap rather than crashing the whole pipeline
or (worse) silently returning fake confident results.
"""

import numpy as np
import requests

TRUFOR_SERVER_URL = "http://localhost:8000"
TIMEOUT_SECONDS = 30


class TruForDetector:
    """
    Thin HTTP client for the real TruFor model server.

    Usage:
        detector = TruForDetector()
        result = detector.detect(image_bgr)
        # result["heatmap"] -> (H, W) numpy array [0, 1]
        # result["confidence"] -> float or None
        # result["reliability_map"] -> (H, W) numpy array or None
    """

    def __init__(self, server_url: str = TRUFOR_SERVER_URL):
        self.server_url = server_url
        self.available = self._check_health()

    def _check_health(self) -> bool:
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                print(f"[TruForDetector] Connected to real TruFor server "
                      f"(device: {data.get('device', 'unknown')})")
                return True
        except requests.exceptions.RequestException as e:
            print(f"[TruForDetector] WARNING: Could not reach TruFor server "
                  f"at {self.server_url}: {e}")
            print("  -> Make sure the Docker container's server.py is running.")
            print("  -> Falling back to neutral (uninformative) output.")
        return False

    def detect(self, image_bgr: np.ndarray) -> dict:
        """
        Run tamper detection on a BGR image via the TruFor HTTP server.

        Returns a dict with the same shape regardless of success/failure,
        so downstream code (feature_extraction.py) never needs to
        special-case a missing server.
        """
        h, w = image_bgr.shape[:2]

        if not self.available:
            # Graceful degradation: neutral heatmap, clearly flagged as
            # unavailable rather than silently wrong.
            return {
                "heatmap": np.full((h, w), 0.5, dtype=np.float32),
                "confidence": None,
                "reliability_map": None,
                "score": None,
                "source": "unavailable",
            }

        try:
            import cv2
            success, encoded_img = cv2.imencode(".jpg", image_bgr)
            if not success:
                raise ValueError("Failed to encode image for upload")

            files = {"file": ("image.jpg", encoded_img.tobytes(), "image/jpeg")}
            resp = requests.post(
                f"{self.server_url}/analyze",
                files=files,
                timeout=TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            data = resp.json()

            heatmap = np.array(data["heatmap"], dtype=np.float32)
            confidence_map = (
                np.array(data["confidence"], dtype=np.float32)
                if data.get("confidence") is not None else None
            )

            return {
                "heatmap": heatmap,
                "confidence": confidence_map,
                "reliability_map": confidence_map,  # TruFor's conf map serves this role
                "score": data.get("score"),
                "source": "real_trufor",
            }

        except requests.exceptions.RequestException as e:
            print(f"[TruForDetector] WARNING: Inference request failed: {e}")
            return {
                "heatmap": np.full((h, w), 0.5, dtype=np.float32),
                "confidence": None,
                "reliability_map": None,
                "score": None,
                "source": "error",
            }

    def get_heatmap_overlay(self, image_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Visual overlay: original image blended with the heatmap (red = high tampering probability)."""
        import cv2
        h, w = image_bgr.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image_bgr, 1 - alpha, heatmap_colored, alpha, 0)
        return overlay
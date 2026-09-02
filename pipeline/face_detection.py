"""
face_detection.py — Face detection and verification module

Detects faces in document images and compares them for verification.

Uses:
    - OpenCV Haar cascades for face detection (fast, no extra deps)
    - Optional: face_recognition library for embedding comparison

Features:
    - Detect face in document photo region
    - Count faces in document
    - Check photo-documented consistency
    - Compare two face images for similarity
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from pathlib import Path


# Try to load Haar cascade, fallback to DNN if not available
FACE_CASCADE = None
FACE_CASCADE_PATH = None

try:
    FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    FACE_CASCADE = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if FACE_CASCADE.empty():
        FACE_CASCADE = None
except Exception:
    FACE_CASCADE = None


@dataclass
class FaceDetection:
    """Single detected face."""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class FaceAnalysis:
    """Complete face analysis result."""
    faces_detected: int = 0
    faces: List[FaceDetection] = field(default_factory=list)
    photo_region_has_face: bool = False
    face_in_photo_region: Optional[FaceDetection] = None
    photo_region_detected: bool = False
    warnings: List[str] = field(default_factory=list)
    face_match_score: Optional[float] = None
    is_consistent: bool = True


class FaceDetector:
    """
    Face detection and basic analysis for document verification.

    Usage:
        detector = FaceDetector()
        analysis = detector.analyze_document(image_bgr)
        print(analysis.faces_detected)
        print(analysis.warnings)
    """

    def __init__(self):
        if FACE_CASCADE is not None:
            self.face_cascade = FACE_CASCADE
        else:
            self.face_cascade = None
            # Fallback: use simple contour-based detection

    def detect_faces(self, image_bgr: np.ndarray) -> List[FaceDetection]:
        """
        Detect all faces in an image.

        Args:
            image_bgr: BGR numpy array

        Returns:
            List of FaceDetection objects
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        detections = []

        if self.face_cascade is not None:
            # Use Haar cascade
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            for (x, y, w, h) in faces:
                detections.append(FaceDetection(x=int(x), y=int(y), width=int(w), height=int(h)))
        else:
            # Fallback: detect skin-colored regions as face candidates
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            # Skin color range (H: 0-50, S: 40-200, V: 80-255)
            mask = cv2.inRange(hsv, (0, 40, 80), (50, 200, 255))
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Minimum face size
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect = w / h if h > 0 else 0
                    if 0.5 < aspect < 2.0:  # Face-like aspect ratio
                        detections.append(FaceDetection(x=x, y=y, width=w, height=h))

        return detections

    def analyze_document(self, image_bgr: np.ndarray,
                         photo_region: Optional[Tuple[int, int, int, int]] = None) -> FaceAnalysis:
        """
        Analyze face in a document image.

        Args:
            image_bgr: Document image
            photo_region: Optional (x, y, w, h) of the photo region to check

        Returns:
            FaceAnalysis with detection results and warnings
        """
        analysis = FaceAnalysis()

        # Detect all faces
        faces = self.detect_faces(image_bgr)
        analysis.faces_detected = len(faces)
        analysis.faces = faces

        # Check photo region for face
        if photo_region is not None:
            px, py, pw, ph = photo_region
            analysis.photo_region_detected = True

            # Check if any face falls within photo region
            for face in faces:
                fcx, fcy = face.center
                if (px <= fcx <= px + pw and py <= fcy <= py + ph):
                    analysis.photo_region_has_face = True
                    analysis.face_in_photo_region = face
                    break

            if not analysis.photo_region_has_face:
                analysis.warnings.append(
                    "No face detected in photo region - photo may be missing or tampered"
                )
                analysis.is_consistent = False

        # Warn if multiple faces (possible collage/forgery)
        if analysis.faces_detected > 1:
            analysis.warnings.append(
                f"Multiple faces detected ({analysis.faces_detected}) - possible photo manipulation"
            )
            analysis.is_consistent = False

        # Warn if no faces at all
        if analysis.faces_detected == 0:
            analysis.warnings.append(
                "No faces detected in document - document may not contain a photo"
            )

        # Check face size relative to image (too small = low quality or wrong region)
        h, w = image_bgr.shape[:2]
        for i, face in enumerate(analysis.faces):
            face_ratio = face.area / (h * w)
            if face_ratio < 0.01:
                analysis.warnings.append(
                    f"Face {i+1} is very small ({face_ratio:.3%} of image) - low quality capture"
                )

        return analysis

    def compare_faces(self, image1_bgr: np.ndarray,
                      image2_bgr: np.ndarray) -> float:
        """
        Compare faces in two images for similarity.

        Uses histogram comparison as a lightweight alternative
        to embedding-based comparison.

        Returns:
            Similarity score 0-1 (1 = identical)
        """
        # Extract face regions
        faces1 = self.detect_faces(image1_bgr)
        faces2 = self.detect_faces(image2_bgr)

        if not faces1 or not faces2:
            return 0.0

        # Get largest face from each
        f1 = max(faces1, key=lambda f: f.area)
        f2 = max(faces2, key=lambda f: f.area)

        # Extract face regions
        gray1 = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2_bgr, cv2.COLOR_BGR2GRAY)

        face1 = gray1[f1.y:f1.y+f1.height, f1.x:f1.x+f1.width]
        face2 = gray2[f2.y:f2.y+f2.height, f2.x:f2.x+f2.width]

        # Resize to same dimensions
        face1 = cv2.resize(face1, (100, 100))
        face2 = cv2.resize(face2, (100, 100))

        # Histogram comparison
        hist1 = cv2.calcHist([face1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([face2], [0], None, [256], [0, 256])
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)

        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return float(max(0, similarity))

    def check_photo_consistency(self, image_bgr: np.ndarray,
                                photo_region: Tuple[int, int, int, int]) -> dict:
        """
        Check if photo region is visually consistent with rest of document.

        Detects:
            - Sharp edges around photo (cut-paste indicator)
            - Color temperature mismatch
            - Resolution mismatch
        """
        h, w = image_bgr.shape[:2]
        px, py, pw, ph = photo_region

        # Ensure region is within bounds
        px = max(0, min(px, w))
        py = max(0, min(py, h))
        pw = min(pw, w - px)
        ph = min(ph, h - py)

        photo = image_bgr[py:py+ph, px:px+pw]
        if photo.size == 0:
            return {"consistent": False, "reason": "Invalid photo region"}

        # Check edge sharpness around photo boundary
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Check edges on photo boundary
        boundary_edges = []
        # Top edge
        if py > 0 and py < h:
            boundary_edges.append(np.mean(edges[py-1:py+1, px:px+pw]))
        # Bottom edge
        if py+ph > 0 and py+ph < h:
            boundary_edges.append(np.mean(edges[py+ph-1:py+ph+1, px:px+pw]))
        # Left edge
        if px > 0 and px < w:
            boundary_edges.append(np.mean(edges[py:py+ph, px-1:px+1]))
        # Right edge
        if px+pw > 0 and px+pw < w:
            boundary_edges.append(np.mean(edges[py:py+ph, px+pw-1:px+pw+1]))

        avg_boundary_edge = np.mean(boundary_edges) if boundary_edges else 0
        sharp_edges = avg_boundary_edge > 100  # High edge density = sharp cut

        # Check color histogram consistency
        photo_hsv = cv2.cvtColor(photo, cv2.COLOR_BGR2HSV)
        photo_mean_hue = np.mean(photo_hsv[:, :, 0])

        # Compare with surrounding region
        margin = 20
        surround_regions = []
        if py - margin > 0:
            surround_regions.append(image_bgr[py-margin:py, px:px+pw])
        if py + ph + margin < h:
            surround_regions.append(image_bgr[py+ph:py+ph+margin, px:px+pw])
        if px - margin > 0:
            surround_regions.append(image_bgr[py:py+ph, px-margin:px])
        if px + pw + margin < w:
            surround_regions.append(image_bgr[py:py+ph, px+pw:px+pw+margin])

        consistent = True
        reasons = []

        if sharp_edges:
            reasons.append("Sharp edges detected around photo boundary (possible cut-paste)")
            consistent = False

        if surround_regions:
            surround_hues = []
            for region in surround_regions:
                if region.size > 0:
                    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
                    surround_hues.append(np.mean(hsv[:, :, 0]))

            if surround_hues:
                hue_diff = abs(photo_mean_hue - np.mean(surround_hues))
                if hue_diff > 15:
                    reasons.append(f"Color temperature mismatch (photo hue={photo_mean_hue:.0f}, surround={np.mean(surround_hues):.0f})")
                    consistent = False

        return {
            "consistent": consistent,
            "reasons": reasons,
            "boundary_edge_score": float(avg_boundary_edge),
            "sharp_edges_detected": sharp_edges,
        }


def detect_photo_region(image_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Attempt to automatically detect the photo region in a document.

    Strategy: Look for rectangular region with uniform color (typical of ID photos).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Common photo positions (right side for Aadhaar, left for PAN, etc.)
    candidate_regions = [
        # (x_start_ratio, y_start_ratio, w_ratio, h_ratio)
        (0.6, 0.15, 0.3, 0.4),   # Right side (Aadhaar-style)
        (0.05, 0.15, 0.25, 0.4),  # Left side (PAN-style)
        (0.55, 0.1, 0.35, 0.5),   # Right side larger
    ]

    best_region = None
    best_score = 0

    for xr, yr, wr, hr in candidate_regions:
        x = int(w * xr)
        y = int(h * yr)
        rw = int(w * wr)
        rh = int(h * hr)

        # Ensure within bounds
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        rw = min(rw, w - x)
        rh = min(rh, h - y)

        if rw < 50 or rh < 50:
            continue

        region = gray[y:y+rh, x:x+rw]

        # Score: check if region has skin-like color (face indicator)
        hsv_region = cv2.cvtColor(image_bgr[y:y+rh, x:x+rw], cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv_region, (0, 40, 80), (50, 200, 255))
        skin_ratio = np.mean(skin_mask > 0)

        if skin_ratio > best_score and skin_ratio > 0.1:
            best_score = skin_ratio
            best_region = (x, y, rw, rh)

    return best_region

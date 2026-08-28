"""
Fake Identity & Document Screening Pipeline
SIH26188 - Day 1 MVP

Modules:
    preprocessing  - Image loading, normalization, ELA computation
    trufor_detector - TruFor-based tamper detection & heatmap generation
    feature_extraction - Compact feature vectors from TruFor + ELA stats
    ocr_module - EasyOCR text extraction + consistency checks
    adaptive_learning - Incremental classifier fine-tuning
    scoring - Combined authenticity scoring
"""

from pipeline.preprocessing import preprocess_image, compute_ela
from pipeline.trufor_detector import TruForDetector
from pipeline.feature_extraction import extract_features
from pipeline.ocr_module import OCRProcessor
from pipeline.adaptive_learning import AdaptiveClassifier
from pipeline.scoring import compute_authenticity_score

__all__ = [
    "preprocess_image",
    "compute_ela",
    "TruForDetector",
    "extract_features",
    "OCRProcessor",
    "AdaptiveClassifier",
    "compute_authenticity_score",
]

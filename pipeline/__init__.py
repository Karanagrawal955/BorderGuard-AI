"""
pipeline/__init__.py — Document Fraud Detection Pipeline

Modules:
    preprocessing       - Image loading, normalization, ELA computation
    trufor_detector     - Tamper detection CNN + heatmap generation
    feature_extraction  - Compact feature vectors from TruFor + ELA stats
    ocr_module          - EasyOCR text extraction + consistency checks
    mrz_parser          - Machine Readable Zone parsing for passports/visas
    document_validation - Rule-based document format validation
    face_detection      - Face detection and photo consistency checks
    adaptive_learning   - Incremental classifier fine-tuning
    scoring             - Combined authenticity scoring

Usage:
    from pipeline import run_full_analysis
    result = run_full_analysis("path/to/document.jpg")
"""

from pipeline.preprocessing import preprocess_image, compute_ela
from pipeline.trufor_detector import TruForDetector
from pipeline.feature_extraction import extract_features
from pipeline.ocr_module import OCRProcessor
from pipeline.mrz_parser import parse_mrz, find_mrz_in_text, MRZData
from pipeline.document_validation import (
    auto_validate,
    validate_indian_passport,
    validate_indian_pan,
    validate_indian_aadhaar,
    validate_visa,
    DocumentValidationReport,
)
from pipeline.face_detection import FaceDetector, FaceAnalysis, detect_photo_region
from pipeline.adaptive_learning import AdaptiveClassifier
from pipeline.scoring import compute_authenticity_score, ScoreBreakdown

__all__ = [
    "preprocess_image",
    "compute_ela",
    "TruForDetector",
    "extract_features",
    "OCRProcessor",
    "parse_mrz",
    "find_mrz_in_text",
    "MRZData",
    "auto_validate",
    "validate_indian_passport",
    "validate_indian_pan",
    "validate_indian_aadhaar",
    "validate_visa",
    "DocumentValidationReport",
    "FaceDetector",
    "FaceAnalysis",
    "detect_photo_region",
    "AdaptiveClassifier",
    "compute_authenticity_score",
    "ScoreBreakdown",
]

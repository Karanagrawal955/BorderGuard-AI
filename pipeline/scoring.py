"""
scoring.py — Combined authenticity scoring

Merges ALL pipeline outputs into a single 0-100 "authenticity score".

Scoring Formula:
    score = w_classifier * (1 - P_tampered) * 100
          + w_trufor    * (1 - mean_heatmap) * 100
          + w_ocr       * ocr_consistency * 100
          + w_validation * validation_score * 100
          + w_face      * face_score * 100

Default weights:
    w_classifier  = 0.30  (ML model)
    w_trufor      = 0.25  (pixel-level detection)
    w_ocr         = 0.15  (text extraction quality)
    w_validation  = 0.20  (rule-based checks)
    w_face        = 0.10  (face verification)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


WEIGHTS = {
    "classifier": 0.30,
    "trufor": 0.25,
    "ocr": 0.15,
    "validation": 0.20,
    "face": 0.10,
}


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of the authenticity score."""
    final_score: float = 0.0
    classifier_component: float = 50.0
    trufor_component: float = 50.0
    ocr_component: float = 50.0
    validation_component: float = 50.0
    face_component: float = 50.0
    risk_level: str = "UNKNOWN"
    flags: List[Dict] = field(default_factory=list)
    summary: str = ""
    extracted_fields: Dict = field(default_factory=dict)
    validation_errors: int = 0
    validation_warnings: int = 0
    face_warnings: int = 0

    def to_dict(self) -> dict:
        return {
            "final_score": self.final_score,
            "risk_level": self.risk_level,
            "components": {
                "classifier": self.classifier_component,
                "trufor": self.trufor_component,
                "ocr": self.ocr_component,
                "validation": self.validation_component,
                "face": self.face_component,
            },
            "flags": self.flags,
            "extracted_fields": self.extracted_fields,
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
            "face_warnings": self.face_warnings,
            "summary": self.summary,
        }


def _risk_level(score: float) -> str:
    if score >= 80:
        return "GENUINE"
    elif score >= 60:
        return "SUSPICIOUS"
    elif score >= 40:
        return "LIKELY_FORGED"
    else:
        return "FORGED"


def _build_summary(
    score: float,
    risk: str,
    classifier_prob: float,
    trufor_mean: float,
    validation_score: float,
    face_score: float,
    ocr_flags: int,
    validation_errors: int,
    face_warnings: int,
    extracted_fields: dict,
) -> str:
    parts = []
    parts.append(f"DOCUMENT AUTHENTICITY: {score:.1f}/100 ({risk})")

    # Classifier
    if classifier_prob > 0.7:
        parts.append(f"ML Classifier: TAMPERING DETECTED (P={classifier_prob:.2f})")
    elif classifier_prob > 0.5:
        parts.append(f"ML Classifier: SUSPICIOUS (P={classifier_prob:.2f})")
    else:
        parts.append(f"ML Classifier: CLEAN (P={classifier_prob:.2f})")

    # TruFor
    if trufor_mean > 0.6:
        parts.append(f"Pixel Analysis: MANIPULATION ARTIFACTS (heat={trufor_mean:.2f})")
    elif trufor_mean > 0.4:
        parts.append(f"Pixel Analysis: MODERATE INDICATORS (heat={trufor_mean:.2f})")
    else:
        parts.append(f"Pixel Analysis: CLEAN (heat={trufor_mean:.2f})")

    # Validation
    if validation_errors > 0:
        parts.append(f"Format Validation: {validation_errors} ERRORS FOUND")
    else:
        parts.append(f"Format Validation: PASSED (score={validation_score:.0f})")

    # Face
    if face_warnings > 0:
        parts.append(f"Face Analysis: {face_warnings} ISSUES DETECTED")
    else:
        parts.append(f"Face Analysis: CONSISTENT")

    # OCR
    if ocr_flags > 0:
        parts.append(f"Text Checks: {ocr_flags} INCONSISTENCIES")

    # Key fields
    if extracted_fields:
        parts.append("---")
        for key, val in list(extracted_fields.items())[:5]:
            parts.append(f"  {key}: {val}")

    return "\n".join(parts)


def compute_authenticity_score(
    classifier_output: Optional[dict] = None,
    heatmap_mean: float = 0.5,
    ocr_result=None,
    validation_result=None,
    face_result=None,
    extracted_fields: Optional[dict] = None,
    weights: Optional[dict] = None,
) -> ScoreBreakdown:
    """
    Compute combined authenticity score from all pipeline outputs.
    """
    w = weights or WEIGHTS
    result = ScoreBreakdown()

    # Classifier component
    if classifier_output and "probability" in classifier_output:
        p_tampered = classifier_output["probability"]
        result.classifier_component = (1.0 - p_tampered) * 100.0
    else:
        p_tampered = 0.5
        result.classifier_component = 50.0

    # TruFor component
    result.trufor_component = (1.0 - heatmap_mean) * 100.0

    # OCR component
    if ocr_result is not None:
        result.ocr_component = ocr_result.overall_consistency * 100.0
        ocr_flags = ocr_result.flag_count
    else:
        result.ocr_component = 50.0
        ocr_flags = 0

    # Validation component
    if validation_result is not None:
        result.validation_component = validation_result.score
        result.validation_errors = validation_result.errors
        result.validation_warnings = validation_result.warnings
    else:
        result.validation_component = 50.0

    # Face component
    if face_result is not None:
        if face_result.is_consistent:
            result.face_component = 90.0
        else:
            result.face_component = max(20.0, 90.0 - len(face_result.warnings) * 25)
        result.face_warnings = len(face_result.warnings)
    else:
        result.face_component = 50.0

    # Weighted combination
    final = (
        w["classifier"] * result.classifier_component
        + w["trufor"] * result.trufor_component
        + w["ocr"] * result.ocr_component
        + w["validation"] * result.validation_component
        + w["face"] * result.face_component
    )
    result.final_score = max(0.0, min(100.0, final))
    result.risk_level = _risk_level(result.final_score)

    # Flags
    all_flags = []
    if ocr_flags > 0 and ocr_result:
        all_flags.extend([f for f in ocr_result.consistency_flags if f.get("status") != "PASS"])
    if result.validation_errors > 0 and validation_result:
        for r in validation_result.results:
            if not r.is_valid and r.severity == "ERROR":
                all_flags.append({
                    "check": r.field_name,
                    "status": "FAIL",
                    "message": r.message,
                })
    if p_tampered > 0.7:
        all_flags.append({
            "check": "classifier",
            "status": "FAIL",
            "message": f"ML classifier: high tampering probability ({p_tampered:.2f})",
        })
    if heatmap_mean > 0.6:
        all_flags.append({
            "check": "trufor",
            "status": "FAIL",
            "message": f"TruFor: manipulation artifacts detected (mean={heatmap_mean:.2f})",
        })
    if face_result and face_result.warnings:
        for warn in face_result.warnings:
            all_flags.append({
                "check": "face",
                "status": "WARN",
                "message": warn,
            })
    result.flags = all_flags

    # Extracted fields
    result.extracted_fields = extracted_fields or {}

    # Summary
    result.summary = _build_summary(
        result.final_score, result.risk_level,
        p_tampered, heatmap_mean,
        result.validation_component, result.face_component,
        ocr_flags, result.validation_errors, result.face_warnings,
        result.extracted_fields,
    )

    return result

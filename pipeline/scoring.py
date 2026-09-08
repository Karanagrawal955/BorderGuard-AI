"""
scoring.py — Content-aware authenticity scoring

Scoring is based on WHAT the document contains, not pixel-level tampering:
1. OCR text extraction — does the image have readable text?
2. Document structure — does it have expected fields (name, DOB, ID number)?
3. Format validation — do fields match expected patterns?
4. Image quality — is it a reasonable photo of a document?

Score meaning:
  80-100: Real document with valid content
  60-79:  Document present but some concerns
  40-59:  Uncertain — may be fake or poor quality
  0-39:   Not a document or clearly fake
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


WEIGHTS = {
    "classifier": 0.40,
    "ocr": 0.25,
    "structure": 0.20,
    "image_quality": 0.15,
}


@dataclass
class ScoreBreakdown:
    final_score: float = 0.0
    ocr_component: float = 50.0
    structure_component: float = 50.0
    validation_component: float = 50.0
    image_quality_component: float = 50.0
    classifier_component: float = 50.0
    trufor_component: float = 50.0
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
                "ocr": self.ocr_component,
                "structure": self.structure_component,
                "validation": self.validation_component,
                "image_quality": self.image_quality_component,
                "classifier": self.classifier_component,
                "trufor": self.trufor_component,
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
    if score >= 65:
        return "GENUINE"
    elif score >= 50:
        return "SUSPICIOUS"
    elif score >= 35:
        return "LIKELY_FORGED"
    else:
        return "FORGED"


def _ocr_score(ocr_result, ocr_text: str) -> float:
    """Score based on OCR text extraction quality."""
    if not ocr_text or len(ocr_text.strip()) < 5:
        return 10.0  # No text found

    text = ocr_text.strip()
    score = 40.0  # Base for having some text

    # Length bonus
    if len(text) > 50:
        score += 15
    elif len(text) > 20:
        score += 10

    # Has alphanumeric content
    alphanum = sum(c.isalnum() for c in text)
    if alphanum > len(text) * 0.5:
        score += 10

    # Has structured content (newlines, separators)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) >= 3:
        score += 10  # Multiple lines = structured document

    # Consistency check from OCR module
    if ocr_result and hasattr(ocr_result, "overall_consistency"):
        score += ocr_result.overall_consistency * 15

    return min(score, 100.0)


def _structure_score(ocr_fields: dict, ocr_text: str) -> float:
    """Score based on document structure (name, DOB, ID number found)."""
    found_fields = sum(1 for v in ocr_fields.values() if v and len(str(v).strip()) > 2)

    if found_fields == 0:
        return 15.0  # No structured fields found

    # Base score for having fields
    score = 30.0 + found_fields * 15.0

    # Bonus for having multiple key fields
    has_name = bool(ocr_fields.get("name"))
    has_dob = bool(ocr_fields.get("dob"))
    has_id = bool(ocr_fields.get("document_number"))

    if has_name and has_dob:
        score += 10
    if has_name and has_id:
        score += 10
    if has_name and has_dob and has_id:
        score += 10  # Full set

    return min(score, 100.0)


def _validation_score(validation_result) -> float:
    """Score based on document format validation."""
    if validation_result is None:
        return 50.0  # No validation performed

    if hasattr(validation_result, "score"):
        return float(validation_result.score)

    return 50.0


def _image_quality_score(image_shape: tuple) -> float:
    """Score based on image quality (resolution, aspect ratio)."""
    if image_shape is None or len(image_shape) < 2:
        return 30.0

    h, w = image_shape[:2]
    if h == 0 or w == 0:
        return 10.0

    score = 50.0

    # Resolution bonus
    pixels = h * w
    if pixels > 500000:  # > 0.5 MP
        score += 20
    elif pixels > 200000:
        score += 10
    elif pixels < 50000:
        score -= 10

    # Aspect ratio: documents are roughly 1:1 to 2:1
    aspect = w / h
    if 0.7 <= aspect <= 2.0:
        score += 15  # Good document aspect ratio
    elif 0.5 <= aspect <= 3.0:
        score += 5
    else:
        score -= 10  # Unusual aspect ratio

    return max(10.0, min(score, 100.0))


def _build_summary(
    score: float, risk: str, ocr_score: float, structure_score: float,
    validation_score: float, ocr_text: str, ocr_fields: dict, flags: list,
) -> str:
    parts = []
    parts.append("DOCUMENT AUTHENTICITY: %.1f/100 (%s)" % (score, risk))

    if ocr_score > 60:
        parts.append("OCR: Text extracted successfully (%d chars)" % len(ocr_text))
    elif ocr_score > 30:
        parts.append("OCR: Partial text found (%d chars)" % len(ocr_text))
    else:
        parts.append("OCR: No readable text found")

    if structure_score > 60:
        fields_found = [k for k, v in ocr_fields.items() if v]
        parts.append("Structure: Fields detected: %s" % ", ".join(fields_found))
    else:
        parts.append("Structure: No document structure detected")

    if validation_score > 60:
        parts.append("Validation: Format checks passed")
    elif validation_score > 40:
        parts.append("Validation: Some checks failed")
    else:
        parts.append("Validation: Failed")

    if flags:
        parts.append("Flags: %d issues" % len(flags))

    if ocr_fields:
        parts.append("---")
        for key, val in list(ocr_fields.items())[:5]:
            if val:
                parts.append("  %s: %s" % (key, val))

    return "\n".join(parts)


def compute_authenticity_score(
    classifier_output: Optional[dict] = None,
    heatmap_mean: float = 0.5,
    ocr_result=None,
    validation_result=None,
    face_result=None,
    extracted_fields: Optional[dict] = None,
    ocr_text: str = "",
    image_shape: tuple = None,
    weights: Optional[dict] = None,
) -> ScoreBreakdown:
    """
    Compute combined authenticity score from all pipeline outputs.
    """
    w = weights or WEIGHTS
    result = ScoreBreakdown()
    flags = []
    ocr_fields = extracted_fields or {}

    # 1. OCR score
    result.ocr_component = _ocr_score(ocr_result, ocr_text)

    # 2. Structure score (document fields)
    result.structure_component = _structure_score(ocr_fields, ocr_text)

    # 3. Validation score
    result.validation_component = _validation_score(validation_result)
    if validation_result and hasattr(validation_result, "errors"):
        result.validation_errors = validation_result.errors
    if validation_result and hasattr(validation_result, "warnings"):
        result.validation_warnings = validation_result.warnings

    # 4. Image quality score
    result.image_quality_component = _image_quality_score(image_shape)

    # Legacy components (kept for compatibility)
    if classifier_output and "probability" in classifier_output:
        result.classifier_component = (1.0 - classifier_output["probability"]) * 100.0
    else:
        result.classifier_component = 50.0
    result.trufor_component = (1.0 - heatmap_mean) * 100.0
    if face_result and hasattr(face_result, "is_consistent"):
        result.face_component = 90.0 if face_result.is_consistent else 30.0
        result.face_warnings = len(face_result.warnings) if hasattr(face_result, "warnings") else 0
    else:
        result.face_component = 50.0

    # Weighted combination
    final = (
        w.get("classifier", 0.40) * result.classifier_component
        + w.get("ocr", 0.25) * result.ocr_component
        + w.get("structure", 0.20) * result.structure_component
        + w.get("image_quality", 0.15) * result.image_quality_component
    )
    result.final_score = max(0.0, min(100.0, final))
    result.risk_level = _risk_level(result.final_score)

    # Flags
    if result.ocr_component < 30:
        flags.append({"check": "ocr", "status": "FAIL", "message": "No readable text found"})
    if result.structure_component < 30:
        flags.append({"check": "structure", "status": "FAIL", "message": "No document structure detected"})
    if result.image_quality_component < 40:
        flags.append({"check": "image_quality", "status": "WARN", "message": "Poor image quality"})
    if classifier_output and classifier_output.get("probability", 0) > 0.7:
        flags.append({"check": "classifier", "status": "WARN", "message": "Tampering indicators detected"})
    result.flags = flags

    result.extracted_fields = ocr_fields
    result.summary = _build_summary(
        result.final_score, result.risk_level,
        result.ocr_component, result.structure_component,
        result.validation_component, ocr_text, ocr_fields, flags,
    )

    return result

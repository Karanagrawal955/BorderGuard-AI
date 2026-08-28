"""
scoring.py — Combined authenticity scoring

Merges classifier output + OCR consistency flags into a single
0–100 "authenticity score" where:
    0  = Definitely fraudulent
    50 = Uncertain / inconclusive
    100 = Definitely authentic

Scoring Formula:
    score = w_classifier × (1 - P_tampered) × 100
          + w_trufor    × (1 - mean_heatmap) × 100
          + w_ocr       × overall_consistency × 100

Default weights (tunable):
    w_classifier = 0.50  (ML model has highest weight)
    w_trufor     = 0.30  (pixel-level detection is strong signal)
    w_ocr        = 0.20  (text checks are supplementary)

These weights are biased toward the ML model because:
    - The classifier combines multiple signals already
    - OCR can fail on legitimate documents (low-res, handwriting)
    - TruFor heatmap alone can be noisy
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


# Default scoring weights — adjust these based on validation results
WEIGHTS = {
    "classifier": 0.50,   # ML classifier probability
    "trufor": 0.30,       # TruFor heatmap mean
    "ocr": 0.20,          # OCR consistency checks
}


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of the authenticity score."""
    final_score: float           # 0-100, higher = more authentic
    classifier_component: float  # Contribution from ML classifier
    trufor_component: float      # Contribution from TruFor heatmap
    ocr_component: float         # Contribution from OCR checks
    risk_level: str              # LOW / MEDIUM / HIGH / CRITICAL
    flags: List[Dict]            # List of flagged issues
    summary: str                 # Human-readable summary

    def to_dict(self) -> dict:
        return {
            "final_score": self.final_score,
            "classifier_component": self.classifier_component,
            "trufor_component": self.trufor_component,
            "ocr_component": self.ocr_component,
            "risk_level": self.risk_level,
            "flags": self.flags,
            "summary": self.summary,
        }


def _risk_level(score: float) -> str:
    """Map score to risk level."""
    if score >= 80:
        return "LOW"
    elif score >= 60:
        return "MEDIUM"
    elif score >= 40:
        return "HIGH"
    else:
        return "CRITICAL"


def _build_summary(
    score: float,
    risk: str,
    classifier_prob: float,
    trufor_mean: float,
    ocr_flags: int,
    aadhaar_valid: Optional[bool],
) -> str:
    """Generate a human-readable summary of the analysis."""
    parts = []

    parts.append(f"Overall Authenticity Score: {score:.1f}/100 (Risk: {risk})")

    # Classifier insight
    if classifier_prob > 0.7:
        parts.append(f"ML classifier strongly suggests tampering (P={classifier_prob:.2f})")
    elif classifier_prob > 0.5:
        parts.append(f"ML classifier leans toward tampering (P={classifier_prob:.2f})")
    elif classifier_prob < 0.3:
        parts.append(f"ML classifier suggests document is authentic (P={classifier_prob:.2f})")
    else:
        parts.append(f"ML classifier is uncertain (P={classifier_prob:.2f})")

    # TruFor insight
    if trufor_mean > 0.6:
        parts.append(f"TruFor detected significant manipulation artifacts (heatmap mean={trufor_mean:.2f})")
    elif trufor_mean > 0.4:
        parts.append(f"TruFor found moderate manipulation indicators (heatmap mean={trufor_mean:.2f})")
    else:
        parts.append(f"TruFor found minimal manipulation indicators (heatmap mean={trufor_mean:.2f})")

    # OCR insight
    if ocr_flags > 0:
        parts.append(f"OCR flagged {ocr_flags} text inconsistency(ies)")
    else:
        parts.append("OCR found no text inconsistencies")

    # Aadhaar insight
    if aadhaar_valid is not None:
        if aadhaar_valid:
            parts.append("Aadhaar checksum: VALID")
        else:
            parts.append("Aadhaar checksum: INVALID — possible forged number")

    return "\n".join(parts)


def compute_authenticity_score(
    classifier_output: Optional[dict] = None,
    heatmap_mean: float = 0.5,
    ocr_result=None,
    weights: Optional[dict] = None,
) -> ScoreBreakdown:
    """
    Compute the combined authenticity score from all pipeline outputs.

    Args:
        classifier_output : dict from AdaptiveClassifier.predict()
            Keys: probability (float), confidence (float)
        heatmap_mean      : float — mean of TruFor manipulation heatmap
        ocr_result        : OCRResult object from OCRProcessor.process()
        weights           : Optional override for scoring weights

    Returns:
        ScoreBreakdown with final_score (0-100), component breakdown,
        risk level, flagged issues, and human-readable summary.
    """
    w = weights or WEIGHTS

    # --- Classifier component ---
    # P_tampered goes from 0 (clean) to 1 (tampered)
    # We want: 100 for P=0 (authentic), 0 for P=1 (forged)
    if classifier_output and "probability" in classifier_output:
        p_tampered = classifier_output["probability"]
        classifier_score = (1.0 - p_tampered) * 100.0
    else:
        # No classifier output → neutral score
        classifier_score = 50.0
        p_tampered = 0.5

    # --- TruFor component ---
    # heatmap_mean goes from 0 (no tampering) to 1 (full tampering)
    # We want: 100 for heatmap=0 (authentic), 0 for heatmap=1 (forged)
    trufor_score = (1.0 - heatmap_mean) * 100.0

    # --- OCR component ---
    # Based on overall_consistency (1.0 = fully consistent, 0.0 = many issues)
    if ocr_result is not None:
        ocr_score = ocr_result.overall_consistency * 100.0
        ocr_flags = ocr_result.consistency_flags
        aadhaar_valid = ocr_result.checksum_valid
    else:
        ocr_score = 50.0  # Neutral if OCR not run
        ocr_flags = []
        aadhaar_valid = None

    # --- Weighted combination ---
    final_score = (
        w["classifier"] * classifier_score
        + w["trufor"] * trufor_score
        + w["ocr"] * ocr_score
    )

    # Clamp to [0, 100]
    final_score = max(0.0, min(100.0, final_score))

    # --- Risk assessment ---
    risk = _risk_level(final_score)

    # --- Collect all flags ---
    all_flags = []
    if ocr_flags:
        all_flags.extend(ocr_flags)
    if p_tampered > 0.7:
        all_flags.append({
            "check": "classifier",
            "status": "FAIL",
            "message": f"ML classifier flagged high tampering probability ({p_tampered:.2f})",
            "severity": p_tampered,
        })
    if heatmap_mean > 0.6:
        all_flags.append({
            "check": "trufor_heatmap",
            "status": "FAIL",
            "message": f"TruFor heatmap shows high manipulation (mean={heatmap_mean:.2f})",
            "severity": heatmap_mean,
        })

    # --- Summary ---
    summary = _build_summary(
        final_score, risk, p_tampered, heatmap_mean,
        len(ocr_flags), aadhaar_valid,
    )

    return ScoreBreakdown(
        final_score=round(final_score, 1),
        classifier_component=round(classifier_score, 1),
        trufor_component=round(trufor_score, 1),
        ocr_component=round(ocr_score, 1),
        risk_level=risk,
        flags=all_flags,
        summary=summary,
    )

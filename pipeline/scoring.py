"""
scoring.py
----------
Combines the classifier's tampering probability with the OCR
consistency-check flags into a single 0-100 "authenticity score" that's
easy to present on stage: 100 = confidently genuine, 0 = confidently
forged.

WEIGHTING RATIONALE (be ready to explain this in Q&A - a judge WILL ask
"how did you pick these numbers"):

  - The ML classifier score is the primary signal (65% weight) because
    it's the only component that looks at pixel-level compression
    artifacts, which are the hardest thing for a forger to fake.
  - The Verhoeff/PAN checksum check gets a large fixed penalty (up to
    20 points) rather than a proportional weight, because a failed
    checksum is close to unambiguous evidence of a fabricated ID
    number - it's a binary red flag, not a "soft" signal like the
    others, so it shouldn't be diluted by averaging.
  - Height-uniformity and spacing-anomaly flags get smaller weights
    (up to 8 points each) because they're heuristic and can trigger on
    genuinely messy but real documents (bad scans, low-quality
    printers) - false positives here are more plausible than for the
    checksum or the ML score.

These weights are NOT derived from a validation study - they're a
reasonable starting allocation you should openly describe as such. If
you get more labeled data before finals, refitting these weights (or
better, feeding all signals into the classifier itself as extra
features) is a natural "future work" talking point.
"""

from dataclasses import dataclass, field

from pipeline.classifier import AdaptiveClassifier, FeatureVector
from pipeline.ocr_checks import ConsistencyReport


# Weight allocation. The 100-point score is split into two pools:
#   - an ML pool (WEIGHT_ML_CLASSIFIER points) awarded proportionally to
#     the classifier's confidence in "genuine"
#   - a rule-based pool (RULE_BASED_POOL points) that STARTS FULL and is
#     only reduced when a specific check actually fails. This is the fix
#     for an earlier bug in this file: penalties must come out of their
#     own pool, not be subtracted from the ML pool - otherwise a
#     perfectly clean document with no classifier available yet was
#     mathematically capped near ~32/100, wrongly reading as "forged."
WEIGHT_ML_CLASSIFIER = 65
RULE_BASED_POOL = 100 - WEIGHT_ML_CLASSIFIER   # 35, awarded when checks pass
PENALTY_CHECKSUM_FAIL = 20     # flat penalty if ID checksum/format fails
PENALTY_HEIGHT_ANOMALY = 8     # flat penalty if height-uniformity check flags
PENALTY_SPACING_ANOMALY = 7    # flat penalty if spacing-anomaly check flags
# checksum + height + spacing penalties sum to exactly RULE_BASED_POOL (35),
# so failing every rule-based check can (correctly) zero out that pool.

# When the ML classifier isn't trained yet, award half-credit for the ML
# pool (neutral, "no evidence either way") rather than zero or full credit.
UNSCORED_ML_CREDIT_FRACTION = 0.5


@dataclass
class AuthenticityResult:
    score: int                      # 0-100, higher = more likely genuine
    verdict: str                    # "Genuine" / "Suspicious" / "Forged"
    ml_component: float             # points contributed by the ML classifier
    checksum_penalty: float
    height_penalty: float
    spacing_penalty: float
    reasons: list = field(default_factory=list)   # human-readable explanation
    ml_available: bool = True


def verdict_from_score(score: int) -> str:
    if score >= 70:
        return "Genuine"
    elif score >= 40:
        return "Suspicious - manual review recommended"
    else:
        return "Likely Forged"


def compute_authenticity_score(
    feature_vector: FeatureVector,
    consistency_report: ConsistencyReport,
    classifier: AdaptiveClassifier | None,
) -> AuthenticityResult:
    """
    Combine ML classifier output + rule-based OCR checks into one score.

    Handles the case where the classifier isn't trained yet (e.g. very
    first run before any adaptive training has happened) by falling
    back to a neutral base score plus only the rule-based penalties -
    this is the graceful-degradation path for that scenario.
    """
    reasons = []

    # --- ML component ---
    if classifier is not None and classifier.is_fitted:
        pred = classifier.predict(feature_vector)
        p_genuine = pred["p_genuine"]
        ml_component = WEIGHT_ML_CLASSIFIER * p_genuine
        ml_available = True
        reasons.append(
            f"ML classifier estimates {pred['p_tampered']:.0%} probability "
            f"of tampering based on compression-artifact analysis."
        )
    else:
        ml_component = WEIGHT_ML_CLASSIFIER * UNSCORED_ML_CREDIT_FRACTION
        ml_available = False
        reasons.append(
            "ML classifier not yet trained (run adaptive_learning.py with "
            "labeled samples first) - score is based on rule-based checks "
            "only and should be treated as low-confidence."
        )

    # --- Checksum / format penalty ---
    checksum_penalty = 0.0
    if consistency_report.checksum_valid is False:
        checksum_penalty = PENALTY_CHECKSUM_FAIL
        reasons.append(
            "ID number failed checksum/format validation "
            f"(-{PENALTY_CHECKSUM_FAIL} points)."
        )
    elif consistency_report.checksum_valid is True:
        reasons.append("ID number passed checksum/format validation.")

    # --- Height uniformity penalty ---
    height_penalty = 0.0
    if any("height" in f.lower() for f in consistency_report.flags):
        height_penalty = PENALTY_HEIGHT_ANOMALY
        reasons.append(
            f"Inconsistent text height detected across fields "
            f"(-{PENALTY_HEIGHT_ANOMALY} points)."
        )

    # --- Spacing anomaly penalty ---
    spacing_penalty = 0.0
    if consistency_report.spacing_anomaly_count > 0:
        spacing_penalty = PENALTY_SPACING_ANOMALY
        reasons.append(
            f"{consistency_report.spacing_anomaly_count} spacing anomaly(ies) "
            f"detected between text fields (-{PENALTY_SPACING_ANOMALY} points)."
        )

    if not consistency_report.is_readable:
        reasons.append(
            "OCR could not read the document at all - score reliability "
            "is reduced; a human reviewer should inspect this document."
        )

    rule_component = RULE_BASED_POOL - checksum_penalty - height_penalty - spacing_penalty
    raw_score = ml_component + rule_component
    # If OCR couldn't read anything, cap confidence rather than let the
    # (possibly high) ML-only score stand unqualified.
    if not consistency_report.is_readable:
        raw_score = min(raw_score, 60)

    final_score = int(round(max(0, min(100, raw_score))))
    verdict = verdict_from_score(final_score)

    return AuthenticityResult(
        score=final_score,
        verdict=verdict,
        ml_component=round(ml_component, 2),
        checksum_penalty=checksum_penalty,
        height_penalty=height_penalty,
        spacing_penalty=spacing_penalty,
        reasons=reasons,
        ml_available=ml_available,
    )


if __name__ == "__main__":
    print("Running scoring.py self-test...\n")

    from pipeline.classifier import FEATURE_NAMES

    # --- Scenario 1: clean document, no classifier available yet ---
    clean_fv = FeatureVector(source_path="dummy.jpg", label=None, values=[1.0] * len(FEATURE_NAMES))
    clean_report = ConsistencyReport(flags=[], checks_run=["id_checksum"], checksum_valid=True,
                                       height_variance_score=0.05, spacing_anomaly_count=0, is_readable=True)

    result_no_clf = compute_authenticity_score(clean_fv, clean_report, classifier=None)
    print(f"Scenario 1 (clean doc, no classifier yet): score={result_no_clf.score}, "
          f"verdict='{result_no_clf.verdict}'")
    assert result_no_clf.ml_available is False
    # With no classifier and zero rule-based red flags, score should sit
    # near the neutral midpoint (~67 = 32.5 ML half-credit + 35 full rule
    # pool), NOT be dragged down into "forged" territory just because a
    # model hasn't been trained yet.
    assert result_no_clf.score >= 60, f"Expected a neutral-ish score, got {result_no_clf.score}"

    # --- Scenario 2: clean document WITH a trained classifier ---
    import random
    from PIL import Image, ImageDraw
    from pipeline.preprocessing import preprocess_document
    from pipeline.classifier import extract_features, AdaptiveClassifier
    import os

    random.seed(3)
    os.makedirs("/tmp/scoring_test/genuine", exist_ok=True)
    os.makedirs("/tmp/scoring_test/tampered", exist_ok=True)

    def make_doc(path, tamper):
        img = Image.new("RGB", (600, 380), color=(235, 235, 225))
        d = ImageDraw.Draw(img)
        d.rectangle([20, 20, 580, 90], outline=(0, 0, 0), width=2)
        d.text((30, 40), "SAMPLE FICTIONAL ID", fill=(0, 0, 0))
        d.rectangle([30, 120, 200, 260], outline=(0, 0, 0), width=2)
        img.save(path, "JPEG", quality=92)
        if tamper:
            t = Image.open(path).convert("RGB")
            patch = Image.new("RGB", (150, 120), color=(170, 140, 120))
            t.paste(patch, (35, 125))
            t.save(path, "JPEG", quality=90)

    train_fvs = []
    for i in range(8):
        p = f"/tmp/scoring_test/genuine/g{i}.jpg"
        make_doc(p, tamper=False)
        train_fvs.append(extract_features(preprocess_document(p), label=0))
    for i in range(8):
        p = f"/tmp/scoring_test/tampered/t{i}.jpg"
        make_doc(p, tamper=True)
        train_fvs.append(extract_features(preprocess_document(p), label=1))

    clf = AdaptiveClassifier()
    clf.fit_incremental(train_fvs)

    genuine_test_path = "/tmp/scoring_test/genuine_check.jpg"
    tampered_test_path = "/tmp/scoring_test/tampered_check.jpg"
    make_doc(genuine_test_path, tamper=False)
    make_doc(tampered_test_path, tamper=True)

    genuine_fv = extract_features(preprocess_document(genuine_test_path))
    tampered_fv = extract_features(preprocess_document(tampered_test_path))

    good_report = ConsistencyReport(checksum_valid=True, spacing_anomaly_count=0)
    bad_report = ConsistencyReport(
        flags=["Text height varies unusually across fields"],
        checksum_valid=False,
        spacing_anomaly_count=2,
    )

    result_genuine = compute_authenticity_score(genuine_fv, good_report, clf)
    result_tampered = compute_authenticity_score(tampered_fv, bad_report, clf)

    print(f"\nScenario 2a (genuine doc + clean OCR):   score={result_genuine.score}, "
          f"verdict='{result_genuine.verdict}'")
    print(f"Scenario 2b (tampered doc + failed checks): score={result_tampered.score}, "
          f"verdict='{result_tampered.verdict}'")
    print(f"\nReasons for tampered doc:")
    for r in result_tampered.reasons:
        print(f"  - {r}")

    assert result_genuine.score > result_tampered.score, \
        "Genuine document should score higher than a tampered one with failed checks"
    assert result_tampered.checksum_penalty == PENALTY_CHECKSUM_FAIL
    assert result_tampered.height_penalty == PENALTY_HEIGHT_ANOMALY
    assert result_tampered.spacing_penalty == PENALTY_SPACING_ANOMALY

    # --- Scenario 3: unreadable document should cap the score ---
    unreadable_report = ConsistencyReport(is_readable=False, flags=["OCR could not extract any text"])
    result_unreadable = compute_authenticity_score(genuine_fv, unreadable_report, clf)
    print(f"\nScenario 3 (unreadable OCR): score={result_unreadable.score} (capped at 60)")
    assert result_unreadable.score <= 60

    print("\nAll scoring.py self-tests passed.")

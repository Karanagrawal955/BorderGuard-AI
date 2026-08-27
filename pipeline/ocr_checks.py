"""
ocr_checks.py
-------------
Rule-based consistency checks run on OCR-extracted text fields.

This module is deliberately decoupled from any specific OCR engine.
It consumes a list of TextBox objects (text + bounding box + confidence),
which is exactly the shape of output EasyOCR gives you. That means:
  - Today, without internet/torch available in this sandbox, we can build
    and fully unit-test the CHECK LOGIC against synthetic/mock OCR output.
  - Tonight, on your machine with EasyOCR installed, you plug real OCR
    output into the same functions with zero code changes here.

Why rule-based checks matter even though we also have an ML classifier:
they're cheap, fast, and -- crucially -- EXPLAINABLE. When a judge asks
"why did your system flag this document?", "the ID checksum is invalid"
is a much stronger answer than "the neural network said so."

Checks implemented:
  1. Verhoeff checksum validation for Aadhaar-style 12-digit numbers.
  2. PAN format validation (fixed alphanumeric pattern).
  3. Font-region uniformity check (bounding-box height variance across
     text lines - inconsistent heights can indicate an edited/pasted
     field rendered in a different font size).
  4. Spacing/kerning anomaly check (unusually large horizontal gaps
     between OCR word boxes on the same line, which can indicate an
     edited field that doesn't quite align with the surrounding text).
"""

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------
# 1. OCR data model (matches EasyOCR's output shape: (bbox, text, conf))
# ---------------------------------------------------------------------

@dataclass
class TextBox:
    """One OCR-detected text region."""
    text: str
    # bbox as (x0, y0, x1, y1) - top-left / bottom-right corners
    bbox: tuple
    confidence: float = 1.0

    @property
    def height(self) -> float:
        return abs(self.bbox[3] - self.bbox[1])

    @property
    def width(self) -> float:
        return abs(self.bbox[2] - self.bbox[0])

    @property
    def y_center(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2


@dataclass
class ConsistencyReport:
    """Aggregated result of all rule-based checks on one document."""
    flags: list = field(default_factory=list)          # human-readable issues found
    checks_run: list = field(default_factory=list)      # which checks executed
    checksum_valid: bool | None = None                  # None = no ID number found to check
    height_variance_score: float = 0.0                  # 0 = perfectly uniform, higher = worse
    spacing_anomaly_count: int = 0
    is_readable: bool = True

    @property
    def is_clean(self) -> bool:
        """True if no rule-based check raised a concern."""
        return len(self.flags) == 0 and self.is_readable


# ---------------------------------------------------------------------
# 2. Verhoeff checksum (used by Aadhaar's 12-digit numbering scheme)
# ---------------------------------------------------------------------
# Standard Verhoeff algorithm tables. This is a well-known, publicly
# documented checksum algorithm - not sensitive information, and does
# not require any real Aadhaar data to implement or test.

_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
_VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_checksum_valid(number_str: str) -> bool | None:
    """
    Validate a numeric string using the Verhoeff checksum algorithm.

    Returns:
        True  - checksum is valid
        False - checksum is invalid (likely tampered or fabricated number)
        None  - input wasn't a clean numeric string of plausible length,
                so we can't meaningfully check it (graceful degradation)
    """
    if not number_str:
        return None

    digits_only = re.sub(r"\D", "", number_str)
    if len(digits_only) < 4:
        return None

    digits = [int(d) for d in digits_only]
    digits.reverse()

    c = 0
    for i, digit in enumerate(digits):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][digit]]

    return c == 0


# ---------------------------------------------------------------------
# 3. PAN format validation
# ---------------------------------------------------------------------
# PAN format: 5 letters, 4 digits, 1 letter (e.g. ABCDE1234F).
# The 4th letter encodes holder type (P=individual, C=company, etc.)
# and the last character is a check alphabet - we validate structure,
# not the true government check-digit algorithm (which isn't public).

_PAN_PATTERN = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")


def pan_format_valid(pan_str: str) -> bool | None:
    """
    Validate PAN structural format. Returns None if the input doesn't
    look like it was even intended to be a PAN (graceful degradation).
    """
    if not pan_str:
        return None
    cleaned = re.sub(r"[\s\-]", "", pan_str.upper())
    if len(cleaned) != 10:
        return None
    return bool(_PAN_PATTERN.match(cleaned))


# ---------------------------------------------------------------------
# 4. Font-region uniformity check
# ---------------------------------------------------------------------

def check_height_uniformity(boxes: list[TextBox], tolerance: float = 0.20) -> tuple[float, list[str]]:
    """
    Genuine documents are printed with a small set of consistent font
    sizes per field type. A field edited in different software (or
    pasted from a different source) often renders at a slightly
    different height than its neighbors.

    Returns (variance_score, flags). variance_score is the coefficient
    of variation (std/mean) of text-box heights - 0 is perfectly
    uniform, higher values suggest inconsistent font rendering.
    """
    flags = []
    heights = [b.height for b in boxes if b.text.strip()]

    if len(heights) < 3:
        # Not enough text detected to meaningfully judge uniformity
        return 0.0, flags

    mean_h = sum(heights) / len(heights)
    if mean_h == 0:
        return 0.0, flags

    variance = sum((h - mean_h) ** 2 for h in heights) / len(heights)
    std_h = variance ** 0.5
    cv = std_h / mean_h  # coefficient of variation

    if cv > tolerance:
        flags.append(
            f"Text height varies unusually across fields (CV={cv:.2f}, "
            f"threshold={tolerance}) - may indicate a field rendered in a "
            f"different font size than the rest of the document."
        )

    return cv, flags


# ---------------------------------------------------------------------
# 5. Spacing / kerning anomaly check
# ---------------------------------------------------------------------

def check_spacing_anomalies(boxes: list[TextBox], row_tolerance: float = 12.0,
                              gap_multiplier: float = 2.5) -> tuple[int, list[str]]:
    """
    Groups text boxes into rows (by y_center proximity), then checks
    horizontal gaps between consecutive boxes in each row. A gap that's
    much larger than the typical gap on that row can indicate an edited
    field that doesn't align naturally with its neighbors.

    Returns (anomaly_count, flags).
    """
    flags = []
    if len(boxes) < 3:
        return 0, flags

    sorted_boxes = sorted(boxes, key=lambda b: b.y_center)

    # Group into rows
    rows = []
    current_row = [sorted_boxes[0]]
    for b in sorted_boxes[1:]:
        if abs(b.y_center - current_row[-1].y_center) <= row_tolerance:
            current_row.append(b)
        else:
            rows.append(current_row)
            current_row = [b]
    rows.append(current_row)

    anomaly_count = 0
    for row in rows:
        if len(row) < 4:
            # Need at least 3 gaps to establish a reliable "typical" spacing
            # baseline. With only 2 gaps, the larger of the two gets picked
            # as the "median" itself, which makes outlier detection
            # meaningless - so we skip rows too short to judge reliably
            # rather than produce a false negative.
            continue
        row_sorted = sorted(row, key=lambda b: b.bbox[0])
        gaps = [
            row_sorted[i + 1].bbox[0] - row_sorted[i].bbox[2]
            for i in range(len(row_sorted) - 1)
        ]
        gaps = [g for g in gaps if g > 0]
        if len(gaps) < 3:
            continue

        sorted_gaps = sorted(gaps)
        mid = len(sorted_gaps) // 2
        if len(sorted_gaps) % 2 == 0:
            median_gap = (sorted_gaps[mid - 1] + sorted_gaps[mid]) / 2
        else:
            median_gap = sorted_gaps[mid]

        if median_gap <= 0:
            continue
        for g in gaps:
            if g > median_gap * gap_multiplier and g > 10:
                anomaly_count += 1

    if anomaly_count > 0:
        flags.append(
            f"Found {anomaly_count} unusually large horizontal gap(s) "
            f"between text elements - may indicate a re-positioned or "
            f"edited field."
        )

    return anomaly_count, flags


# ---------------------------------------------------------------------
# 6. Top-level entry point
# ---------------------------------------------------------------------

def run_consistency_checks(
    boxes: list[TextBox],
    id_number_candidate: str | None = None,
    id_type: str = "aadhaar",
) -> ConsistencyReport:
    """
    Run all applicable rule-based checks and return an aggregated report.

    id_number_candidate: the OCR'd string believed to be the ID number
                          (caller is responsible for picking this out of
                          `boxes`, e.g. via regex on box.text - kept
                          separate here so the checksum logic stays
                          testable in isolation).
    id_type: "aadhaar" or "pan" - determines which checksum rule applies.
    """
    report = ConsistencyReport()

    if not boxes:
        report.is_readable = False
        report.flags.append("OCR could not extract any text from this document.")
        return report

    # Checksum / format check
    if id_number_candidate:
        report.checks_run.append("id_checksum")
        if id_type == "aadhaar":
            valid = verhoeff_checksum_valid(id_number_candidate)
        elif id_type == "pan":
            valid = pan_format_valid(id_number_candidate)
        else:
            valid = None

        report.checksum_valid = valid
        if valid is False:
            report.flags.append(
                f"ID number '{id_number_candidate}' failed {id_type} checksum/format "
                f"validation - this is a strong indicator of a fabricated or "
                f"mistyped number."
            )
    else:
        report.checksum_valid = None

    # Height uniformity check
    report.checks_run.append("height_uniformity")
    cv, height_flags = check_height_uniformity(boxes)
    report.height_variance_score = cv
    report.flags.extend(height_flags)

    # Spacing anomaly check
    report.checks_run.append("spacing_anomaly")
    anomaly_count, spacing_flags = check_spacing_anomalies(boxes)
    report.spacing_anomaly_count = anomaly_count
    report.flags.extend(spacing_flags)

    return report


if __name__ == "__main__":
    print("Running ocr_checks.py self-test...\n")

    # --- Test 1: Verhoeff checksum ---
    # A number engineered to have a VALID Verhoeff checksum: "236" -> compute
    # checksum digit by brute force so this test is honest and not hardcoded
    # to a suspiciously "real-looking" Aadhaar number.
    def compute_verhoeff_check_digit(number_str: str) -> int:
        digits = [int(d) for d in number_str]
        digits.reverse()
        c = 0
        for i, digit in enumerate(digits):
            c = _VERHOEFF_D[c][_VERHOEFF_P[(i + 1) % 8][digit]]
        return _VERHOEFF_INV[c]

    base = "123412341234"[:11]  # 11 fictional digits
    check_digit = compute_verhoeff_check_digit(base)
    valid_number = base + str(check_digit)
    invalid_number = base + str((check_digit + 1) % 10)

    assert verhoeff_checksum_valid(valid_number) is True, "Valid Verhoeff number should pass"
    assert verhoeff_checksum_valid(invalid_number) is False, "Invalid Verhoeff number should fail"
    assert verhoeff_checksum_valid("12") is None, "Too-short input should return None"
    print(f"PASS: Verhoeff checksum correctly validates {valid_number} "
          f"and rejects {invalid_number}")

    # --- Test 2: PAN format ---
    assert pan_format_valid("ABCDE1234F") is True
    assert pan_format_valid("ABCDE1234") is None    # wrong length -> undetermined, not "invalid"
    assert pan_format_valid("12345ABCDE") is False  # right length, wrong pattern -> genuinely invalid
    assert pan_format_valid("") is None
    print("PASS: PAN format validation works on structural pattern")

    # --- Test 3: height uniformity, using mock OCR boxes ---
    uniform_boxes = [
        TextBox("Name:", (10, 10, 60, 30)),
        TextBox("TEST PERSON", (65, 10, 180, 30)),
        TextBox("DOB:", (10, 40, 60, 60)),
        TextBox("01/01/1990", (65, 40, 160, 60)),
    ]
    cv_uniform, flags_uniform = check_height_uniformity(uniform_boxes)

    tampered_boxes = list(uniform_boxes) + [
        TextBox("EDITED_FIELD", (10, 70, 160, 120)),  # much taller box = different font size
    ]
    cv_tampered, flags_tampered = check_height_uniformity(tampered_boxes)

    assert cv_tampered > cv_uniform, "Tampered set should show higher height variance"
    assert len(flags_tampered) > 0, "Height check should raise a flag on the tampered set"
    print(f"PASS: height uniformity check - uniform CV={cv_uniform:.3f}, "
          f"tampered CV={cv_tampered:.3f} (flag raised: {bool(flags_tampered)})")

    # --- Test 4: spacing anomaly ---
    # Need >=4 boxes per row (>=3 gaps) for the median-gap baseline to be
    # statistically meaningful - see comment in check_spacing_anomalies.
    normal_row = [
        TextBox("Name:", (10, 10, 50, 30)),
        TextBox("TEST", (55, 10, 90, 30)),
        TextBox("PERSON", (95, 10, 140, 30)),
        TextBox("K", (145, 10, 160, 30)),
    ]
    anomaly_count_normal, _ = check_spacing_anomalies(normal_row)

    anomalous_row = [
        TextBox("Name:", (10, 10, 50, 30)),
        TextBox("TEST", (55, 10, 90, 30)),
        TextBox("PERSON", (95, 10, 140, 30)),
        TextBox("K", (350, 10, 365, 30)),  # suspiciously far gap vs the other 3
    ]
    anomaly_count_bad, spacing_flags = check_spacing_anomalies(anomalous_row)

    assert anomaly_count_bad > anomaly_count_normal
    print(f"PASS: spacing anomaly check - normal row anomalies={anomaly_count_normal}, "
          f"suspicious row anomalies={anomaly_count_bad}")

    # --- Test 5: full pipeline via run_consistency_checks ---
    report_clean = run_consistency_checks(uniform_boxes, id_number_candidate=valid_number, id_type="aadhaar")
    report_dirty = run_consistency_checks(tampered_boxes, id_number_candidate=invalid_number, id_type="aadhaar")

    assert report_clean.checksum_valid is True
    assert report_dirty.checksum_valid is False
    assert report_clean.is_clean is True or len(report_clean.flags) == 0
    assert len(report_dirty.flags) > len(report_clean.flags)

    print(f"\nClean document report:  checksum_valid={report_clean.checksum_valid}, "
          f"flags={len(report_clean.flags)}")
    print(f"Tampered document report: checksum_valid={report_dirty.checksum_valid}, "
          f"flags={len(report_dirty.flags)} -> {report_dirty.flags}")

    print("\nAll ocr_checks.py self-tests passed.")

"""
document_validation.py — Document format and rule-based validation

Validates extracted fields against official document standards:
    - Indian passport format
    - PAN card format
    - Aadhaar format
    - Visa format
    - Date consistency checks
    - Cross-field validation
"""

import re
from datetime import datetime, date
from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    field_name: str
    check_name: str
    is_valid: bool
    message: str
    severity: str = "ERROR"  # ERROR, WARNING, INFO


@dataclass
class DocumentValidationReport:
    """Complete validation report for a document."""
    document_type: str = ""
    results: List[ValidationResult] = field(default_factory=list)
    overall_valid: bool = True
    score: float = 100.0

    def add(self, result: ValidationResult):
        self.results.append(result)
        if not result.is_valid and result.severity == "ERROR":
            self.overall_valid = False
            self.score -= 15
        elif not result.is_valid and result.severity == "WARNING":
            self.score -= 5
        self.score = max(0, self.score)

    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if not r.is_valid and r.severity == "ERROR")

    @property
    def warnings(self) -> int:
        return sum(1 for r in self.results if not r.is_valid and r.severity == "WARNING")


# ──────────────────────────────────────────────────────────────
# Passport Validation (Indian)
# ──────────────────────────────────────────────────────────────

def validate_indian_passport(fields: dict) -> DocumentValidationReport:
    """
    Validate Indian passport fields.
    Expected fields: passport_number, name, dob, gender, nationality, expiry
    """
    report = DocumentValidationReport(document_type="INDIAN_PASSPORT")

    # Passport number format: 1 letter + 7 digits (e.g., A1234567)
    passport_num = fields.get("passport_number", "")
    if passport_num:
        pattern = r'^[A-Z]\d{7}$'
        is_valid = bool(re.match(pattern, passport_num))
        report.add(ValidationResult(
            "passport_number", "format", is_valid,
            f"Passport number '{passport_num}' matches format [A-Z]\\d{{7}}" if is_valid
            else f"Invalid passport format: '{passport_num}'. Expected: A1234567"
        ))

    # Name validation
    name = fields.get("name", "")
    if name:
        is_valid = len(name) >= 2 and name.replace(" ", "").isalpha()
        report.add(ValidationResult(
            "name", "format", is_valid,
            f"Name '{name}' is valid" if is_valid
            else f"Invalid name: '{name}'. Should contain only letters"
        ))

    # Date validation
    dob = fields.get("dob", "")
    if dob:
        is_valid, msg = _validate_date(dob, field_name="DOB", min_year=1900, max_year=2010)
        report.add(ValidationResult("dob", "date_range", is_valid, msg))

    expiry = fields.get("expiry", "")
    if expiry:
        is_valid, msg = _validate_date(expiry, field_name="Expiry", min_year=2020, max_year=2040)
        report.add(ValidationResult("expiry", "date_range", is_valid, msg))

        # Check if expired
        if is_valid and _is_date_past(expiry):
            report.add(ValidationResult(
                "expiry", "expired", False,
                f"PASSPORT IS EXPIRED (expiry: {expiry})", "ERROR"
            ))

    # Gender validation
    gender = fields.get("gender", "")
    if gender:
        is_valid = gender.upper() in ["MALE", "FEMALE", "M", "F", "OTHER", "X"]
        report.add(ValidationResult(
            "gender", "format", is_valid,
            f"Gender '{gender}' is valid" if is_valid
            else f"Invalid gender: '{gender}'. Expected: MALE/FEMALE"
        ))

    # Nationality validation
    nationality = fields.get("nationality", "")
    if nationality:
        is_valid = nationality.upper() in ["INDIAN", "IND", "INDIA"]
        report.add(ValidationResult(
            "nationality", "format", is_valid,
            f"Nationality '{nationality}' is valid" if is_valid
            else f"Unexpected nationality: '{nationality}'"
        ))

    # DOB vs Expiry consistency
    if dob and expiry:
        is_valid, msg = _check_dob_expiry_consistency(dob, expiry)
        report.add(ValidationResult("dob_expiry", "consistency", is_valid, msg))

    return report


# ──────────────────────────────────────────────────────────────
# PAN Card Validation (Indian)
# ──────────────────────────────────────────────────────────────

def validate_indian_pan(fields: dict) -> DocumentValidationReport:
    """Validate Indian PAN card fields."""
    report = DocumentValidationReport(document_type="INDIAN_PAN")

    pan = fields.get("pan", "")
    if pan:
        # PAN format: AAAAA0000A (5 letters + 4 digits + 1 letter)
        pattern = r'^[A-Z]{5}\d{4}[A-Z]$'
        is_valid = bool(re.match(pattern, pan.upper()))
        report.add(ValidationResult(
            "pan", "format", is_valid,
            f"PAN '{pan}' is valid" if is_valid
            else f"Invalid PAN format: '{pan}'. Expected: ABCDE1234F"
        ))

        # Check 4th character (entity type)
        if is_valid and len(pan) >= 4:
            entity_types = {
                'A': 'Association of Persons',
                'B': 'Body of Individuals',
                'C': 'Company',
                'F': 'Firm',
                'G': 'Government',
                'H': 'HUF',
                'L': 'Local Authority',
                'J': 'Artificial Juridical Person',
                'P': 'Individual',
                'T': 'Trust',
            }
            entity_char = pan[3].upper()
            entity_type = entity_types.get(entity_char, 'Unknown')
            report.add(ValidationResult(
                "pan", "entity_type", True,
                f"PAN entity type: {entity_type} (code: {entity_char})", "INFO"
            ))

    name = fields.get("name", "")
    if name:
        is_valid = len(name) >= 2
        report.add(ValidationResult(
            "name", "format", is_valid,
            f"Name '{name}' is valid" if is_valid else f"Invalid name: '{name}'"
        ))

    return report


# ──────────────────────────────────────────────────────────────
# Aadhaar Validation (Indian)
# ──────────────────────────────────────────────────────────────

def validate_indian_aadhaar(fields: dict) -> DocumentValidationReport:
    """Validate Indian Aadhaar card fields."""
    report = DocumentValidationReport(document_type="INDIAN_AADHAAR")

    aadhaar = fields.get("aadhaar", "")
    if aadhaar:
        # Aadhaar: 12 digits (with optional spaces)
        clean = aadhaar.replace(" ", "")
        is_valid = len(clean) == 12 and clean.isdigit()
        report.add(ValidationResult(
            "aadhaar", "format", is_valid,
            f"Aadhaar number '{aadhaar}' is valid" if is_valid
            else f"Invalid Aadhaar: '{aadhaar}'. Expected 12 digits"
        ))

        # Verhoeff checksum
        if is_valid:
            checksum_valid = _verhoeff_validate(clean)
            report.add(ValidationResult(
                "aadhaar", "checksum", checksum_valid,
                "Aadhaar Verhoeff checksum VALID" if checksum_valid
                else "Aadhaar Verhoeff checksum FAILED - possible forged number"
            ))

    # Phone validation
    phone = fields.get("phone", "")
    if phone:
        clean = phone.replace("+91", "").replace(" ", "")
        is_valid = len(clean) == 10 and clean.isdigit()
        report.add(ValidationResult(
            "phone", "format", is_valid,
            f"Phone '{phone}' is valid" if is_valid
            else f"Invalid phone: '{phone}'. Expected 10 digits"
        ))

    # Pincode validation
    pincode = fields.get("pincode", "")
    if pincode:
        is_valid = len(pincode) == 6 and pincode.isdigit()
        report.add(ValidationResult(
            "pincode", "format", is_valid,
            f"Pincode '{pincode}' is valid" if is_valid
            else f"Invalid pincode: '{pincode}'. Expected 6 digits"
        ))

    return report


# ──────────────────────────────────────────────────────────────
# Visa Validation
# ──────────────────────────────────────────────────────────────

def validate_visa(fields: dict) -> DocumentValidationReport:
    """Validate visa document fields."""
    report = DocumentValidationReport(document_type="VISA")

    visa_num = fields.get("visa_number", "")
    if visa_num:
        is_valid = len(visa_num) >= 6
        report.add(ValidationResult(
            "visa_number", "format", is_valid,
            f"Visa number '{visa_num}' format OK" if is_valid
            else f"Invalid visa number: '{visa_num}'"
        ))

    visa_type = fields.get("visa_type", "")
    if visa_type:
        valid_types = ["TOURIST", "BUSINESS", "STUDENT", "WORK", "TRANSIT",
                       "MEDICAL", "CONFERENCE", "ENTRY", "EMPLOYMENT"]
        is_valid = visa_type.upper() in valid_types
        report.add(ValidationResult(
            "visa_type", "format", is_valid,
            f"Visa type '{visa_type}' is recognized" if is_valid
            else f"Unknown visa type: '{visa_type}'"
        ))

    # Check stay duration
    stay_days = fields.get("stay_duration_days")
    if stay_days is not None:
        is_valid = 0 < stay_days <= 365
        report.add(ValidationResult(
            "stay_duration", "range", is_valid,
            f"Stay duration {stay_days} days is valid" if is_valid
            else f"Stay duration {stay_days} days out of range (1-365)"
        ))

    return report


# ──────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────

def _validate_date(date_str: str, field_name: str = "date",
                   min_year: int = 1900, max_year: int = 2040) -> tuple:
    """Validate a date string in DD/MM/YYYY or DD.MM.YYYY format."""
    if not date_str:
        return True, f"{field_name}: no date provided"

    # Try different formats
    for fmt in ["%d/%m/%Y", "%d.%m.%Y", "%d-%m-%Y", "%Y-%m-%d"]:
        try:
            parsed = datetime.strptime(date_str, fmt).date()
            if parsed.year < min_year or parsed.year > max_year:
                return False, f"{field_name}: year {parsed.year} outside range [{min_year}-{max_year}]"
            return True, f"{field_name}: '{date_str}' is valid"
        except ValueError:
            continue

    return False, f"{field_name}: cannot parse date '{date_str}'"


def _is_date_past(date_str: str) -> bool:
    """Check if a date is in the past."""
    for fmt in ["%d/%m/%Y", "%d.%m.%Y", "%d-%m-%Y"]:
        try:
            parsed = datetime.strptime(date_str, fmt).date()
            return parsed < date.today()
        except ValueError:
            continue
    return False


def _check_dob_expiry_consistency(dob: str, expiry: str) -> tuple:
    """Check if DOB and expiry dates are consistent."""
    dob_date = None
    expiry_date = None

    for fmt in ["%d/%m/%Y", "%d.%m.%Y", "%d-%m-%Y"]:
        try:
            dob_date = datetime.strptime(dob, fmt).date()
            break
        except ValueError:
            continue

    for fmt in ["%d/%m/%Y", "%d.%m.%Y", "%d-%m-%Y"]:
        try:
            expiry_date = datetime.strptime(expiry, fmt).date()
            break
        except ValueError:
            continue

    if dob_date and expiry_date:
        # Passport validity: 5-10 years for adults, 5 years for minors
        diff_years = (expiry_date - dob_date).days / 365.25
        if diff_years < 5:
            return False, f"DOB-Expiry gap too small ({diff_years:.1f} years)"
        if diff_years > 80:
            return False, f"DOB-Expiry gap too large ({diff_years:.1f} years)"
        return True, f"DOB-Expiry consistency OK ({diff_years:.1f} years gap)"

    return True, "Could not verify DOB-Expiry consistency"


def _verhoeff_validate(number_str: str) -> bool:
    """Verhoeff checksum validation."""
    number_str = re.sub(r"[\s\-]", "", number_str or "")
    if not number_str.isdigit():
        return False

    # Full 10x10 Verhoeff multiplication table. This table used to be missing
    # rows 8 and 9, which made checksum == 8/9 raise "list index out of range"
    # and silently disabled every format check in the pipeline.
    mult = [
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
    perm = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
        [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
        [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
        [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
        [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
        [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
        [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
    ]
    checksum = 0
    for i, digit in enumerate(reversed(number_str)):
        checksum = mult[checksum][perm[i % 8][int(digit)]]
    return checksum == 0


def auto_validate(document_type: str, fields: dict) -> DocumentValidationReport:
    """Auto-detect document type and validate."""
    dtype = document_type.upper()

    if "PASSPORT" in dtype:
        return validate_indian_passport(fields)
    elif "PAN" in dtype:
        return validate_indian_pan(fields)
    elif "AADHAAR" in dtype or "AADHAR" in dtype:
        return validate_indian_aadhaar(fields)
    elif "VISA" in dtype:
        return validate_visa(fields)
    else:
        # Generic validation
        report = DocumentValidationReport(document_type=dtype)
        if fields.get("name"):
            report.add(ValidationResult("name", "present", True, "Name field present"))
        return report

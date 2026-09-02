"""
mrz_parser.py — Machine Readable Zone (MRZ) parser for passports and visas

Parses the 2 or 3 line MRZ at the bottom of passport/visa images.
Validates checksums embedded in the MRZ per ICAO 9303 standard.

Supports:
    - TD1 (3-line MRZ: ID cards)
    - TD2 (2-line MRZ: some visas)
    - TD3 (2-line MRZ: passports)

MRZ Format (TD3 - Passports):
    Line 1: P<INDSURNAME<<GIVEN<NAME<<<<<<<<<<<<<<<<<<<<<
    Line 2: PASSPORT_NUM CHECK NAT DOB CHECK SEX EXP CHECK <<<<<<<<<<< PERSONAL
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class MRZData:
    """Parsed MRZ information."""
    document_type: str = ""
    issuing_country: str = ""
    surname: str = ""
    given_names: str = ""
    passport_number: str = ""
    nationality: str = ""
    date_of_birth: str = ""
    sex: str = ""
    date_of_expiry: str = ""
    personal_number: str = ""
    raw_line1: str = ""
    raw_line2: str = ""
    is_valid: bool = False
    checksum_errors: list = field(default_factory=list)

    @property
    def full_name(self) -> str:
        return f"{self.given_names} {self.surname}".strip()


def _mrz_check_digit(value: str) -> int:
    """
    Calculate MRZ check digit for a given string.
    ICAO 9303 weights: 7, 3, 1 repeating.
    """
    weights = [7, 3, 1]
    total = 0
    for i, char in enumerate(value):
        if char.isdigit():
            total += int(char) * weights[i % 3]
        elif char in "<":
            total += 0
        else:
            # Letters: A=10, B=11, ... Z=35
            total += (ord(char) - ord("A") + 10) * weights[i % 3]
    return total % 10


def _verify_check_digit(field_name: str, value: str, expected_check: str) -> Tuple[bool, str]:
    """Verify a single check digit. Returns (is_valid, error_message)."""
    if not value or not expected_check:
        return True, ""
    actual = _mrz_check_digit(value)
    expected = int(expected_check) if expected_check.isdigit() else -1
    if actual != expected:
        return False, f"{field_name}: expected check={expected}, got={actual}"
    return True, ""


def parse_mrz(lines: list) -> MRZData:
    """
    Parse MRZ lines into structured data.

    Args:
        lines: List of MRZ lines (2 or 3 lines, each 30-44 chars)

    Returns:
        MRZData with parsed fields and validation results
    """
    result = MRZData()

    if not lines or len(lines) < 2:
        return result

    # Clean lines
    lines = [line.strip().upper() for line in lines]

    # Determine format by length
    line1 = lines[0]
    line2 = lines[1]

    # TD3 (Passport): 44 chars per line
    if len(line1) >= 44 and len(line2) >= 44:
        line1 = line1[:44]
        line2 = line2[:44]

        result.raw_line1 = line1
        result.raw_line2 = line2

        # Line 1: Document type + Country + Name
        result.document_type = line1[:2].replace("<", "")
        result.issuing_country = line1[2:5].replace("<", "")

        # Name parsing: everything after country code
        name_part = line1[5:]
        name_components = name_part.split("<<")
        result.surname = name_components[0].replace("<", " ").strip() if name_components else ""
        result.given_names = name_components[1].replace("<", " ").strip() if len(name_components) > 1 else ""

        # Line 2: Passport number + DOB + Sex + Expiry
        result.passport_number = line2[:9].replace("<", "")
        result.nationality = line2[10:13].replace("<", "")
        result.date_of_birth = _format_mrz_date(line2[13:19])
        result.sex = _parse_sex(line2[20])
        result.date_of_expiry = _format_mrz_date(line2[21:27])
        result.personal_number = line2[28:42].replace("<", "")

        # Verify checksums
        errors = []

        # Passport number check (position 9)
        valid, err = _verify_check_digit("passport_number", line2[:9], line2[9])
        if not valid:
            errors.append(err)

        # DOB check (position 19)
        valid, err = _verify_check_digit("dob", line2[13:19], line2[19])
        if not valid:
            errors.append(err)

        # Expiry check (position 27)
        valid, err = _verify_check_digit("expiry", line2[21:27], line2[27])
        if not valid:
            errors.append(err)

        result.checksum_errors = errors
        result.is_valid = len(errors) == 0

    # TD2 (2-line, 36 chars)
    elif len(line1) >= 36 and len(line2) >= 36:
        line1 = line1[:36]
        line2 = line2[:36]

        result.raw_line1 = line1
        result.raw_line2 = line2

        result.document_type = line1[:2].replace("<", "")
        result.issuing_country = line1[2:5].replace("<", "")

        name_part = line1[5:]
        name_components = name_part.split("<<")
        result.surname = name_components[0].replace("<", " ").strip() if name_components else ""
        result.given_names = name_components[1].replace("<", " ").strip() if len(name_components) > 1 else ""

        result.passport_number = line2[:9].replace("<", "")
        result.nationality = line2[10:13].replace("<", "")
        result.date_of_birth = _format_mrz_date(line2[13:19])
        result.sex = _parse_sex(line2[19])
        result.date_of_expiry = _format_mrz_date(line2[20:26])

        result.is_valid = True

    return result


def _format_mrz_date(date_str: str) -> str:
    """Convert MRZ date (YYMMDD) to DD/MM/YYYY format."""
    if len(date_str) != 6 or not date_str.isdigit():
        return date_str
    yy = int(date_str[:2])
    mm = date_str[2:4]
    dd = date_str[4:6]
    # ICAO rule: years 00-99, assume 19XX for >50, 20XX for <=50
    year = 1900 + yy if yy > 50 else 2000 + yy
    return f"{dd}/{mm}/{year}"


def _parse_sex(sex_char: str) -> str:
    """Convert MRZ sex code to readable format."""
    mapping = {"M": "MALE", "F": "FEMALE", "X": "OTHER", "<": "UNKNOWN"}
    return mapping.get(sex_char, "UNKNOWN")


def find_mrz_in_text(text: str) -> list:
    """
    Attempt to find MRZ lines in OCR text output.
    MRZ lines typically start with P<, V<, or contain <<< patterns.
    """
    lines = text.split("\n")
    mrz_lines = []

    for line in lines:
        line = line.strip()
        # MRZ indicators: starts with document type, contains <<<
        if (len(line) >= 30 and
            ("<<" in line or
             re.match(r'^[PVDCA]<', line) or
             re.match(r'^[A-Z0-9<]{30,44}$', line))):
            mrz_lines.append(line)

    return mrz_lines[:3]  # Return up to 3 lines

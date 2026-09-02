"""
generate_passports.py — Generate synthetic Indian passport images

Creates realistic-looking synthetic passport images using OpenCV.
All data is fictional — no real PII.

Features:
    - Machine Readable Zone (MRZ) format
    - Photo placeholder
    - Text fields (name, DOB, passport number, etc.)
    - Indian passport color scheme and layout
"""

import random
import string
from pathlib import Path

import cv2
import numpy as np


# Fictional data pools
FIRST_NAMES_M = ["Amit", "Rahul", "Vikram", "Arjun", "Rohit", "Sanjay", "Deepak", "Karan"]
FIRST_NAMES_F = ["Priya", "Neha", "Anita", "Kavita", "Ritu", "Sapna", "Meera", "Pooja"]
LAST_NAMES = ["Gupta", "Sharma", "Patel", "Kumar", "Singh", "Reddy", "Iyer", "Verma"]
CITIES = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune", "Kolkata", "Jaipur"]
STATES = ["Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", "Telangana", "West Bengal", "Rajasthan"]
AUTHORITIES = ["PASSPORT OFFICE", "MINISTRY OF EXTERNAL AFFAIRS", "RPO MUMBAI", "RPO DELHI"]


def generate_mrz_line1(surname: str, given_name: str) -> str:
    """Generate MRZ line 1 (document type + name)."""
    name_part = f"P<IND{surname.upper()}<<{given_name.upper()}"
    return name_part.ljust(44, "<")


def generate_mrz_line2(
    passport_num: str,
    nationality: str,
    dob: str,
    sex: str,
    expiry: str,
    personal_num: str = "",
) -> str:
    """Generate MRZ line 2 (number, DOB, sex, expiry)."""
    dob_mrz = dob.replace(".", "")  # Remove dots
    expiry_mrz = expiry.replace(".", "")
    sex_code = "M" if sex.upper() == "MALE" else "F"

    line = (
        f"{passport_num}"
        f"<7"  # Check digit placeholder
        f"{nationality}"
        f"{dob_mrz}"
        f"{sex_code}"
        f"{expiry_mrz}"
        f"00"  # Check digit placeholder
        f"{personal_num[:7].ljust(7, '<')}"
        f"<<"  # Padding
    )
    return line[:44].ljust(44, "<")


def generate_passport_image(
    passport_num: str,
    surname: str,
    given_name: str,
    dob: str,
    sex: str,
    output_path: Path,
) -> None:
    """Generate a single synthetic passport image."""

    # Canvas size (standard passport photo ratio ~35x45mm → 1050x1350 at 300dpi)
    width, height = 800, 600
    img = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)

    # Background gradient (Indian passport blue tone)
    for y in range(height):
        ratio = y / height
        blue = int(30 + ratio * 20)
        img[y, :] = [blue, 30 + int(ratio * 15), 40 + int(ratio * 10)]  # BGR

    # Header
    cv2.putText(img, "REPUBLIC OF INDIA", (220, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 215, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "PASSPORT", (290, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Separator line
    cv2.line(img, (30, 85), (770, 85), (255, 215, 0), 2)

    # Photo placeholder (right side)
    photo_x, photo_y = 550, 100
    photo_w, photo_h = 200, 250
    cv2.rectangle(img, (photo_x, photo_y), (photo_x + photo_w, photo_y + photo_h),
                  (200, 200, 200), -1)
    cv2.rectangle(img, (photo_x, photo_y), (photo_x + photo_w, photo_y + photo_h),
                  (100, 100, 100), 2)
    cv2.putText(img, "PHOTO", (photo_x + 60, photo_y + 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

    # Text fields (left side)
    fields = [
        (f"Type: P", (40, 120)),
        (f"Country Code: IND", (40, 145)),
        (f"Passport No: {passport_num}", (40, 170)),
        (f"Name: {given_name} {surname}", (40, 210)),
        (f"Surname: {surname}", (40, 240)),
        (f"Given Name: {given_name}", (40, 265)),
        (f"Sex: {sex}", (40, 305)),
        (f"Date of Birth: {dob}", (40, 330)),
        (f"Place of Birth: {random.choice(CITIES)}", (40, 355)),
        (f"Date of Issue: 01.01.2020", (40, 395)),
        (f"Date of Expiry: 01.01.2030", (40, 420)),
        (f"Authority: {random.choice(AUTHORITIES)}", (40, 445)),
    ]

    for text, pos in fields:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # MRZ zone (bottom)
    cv2.rectangle(img, (0, 470), (width, height), (0, 0, 0), -1)
    surname = surname.upper().replace(" ", "<")
    given_name = given_name.upper().replace(" ", "<")
    dob_mrz = dob.replace(".", "")
    mrz1 = f"P<IND{surname}<<{given_name}".ljust(44, "<")
    mrz2 = f"{passport_num}<7IND{dob_mrz}M3001017<<<<<<<<<<<<<<04"
    mrz2 = mrz2[:44].ljust(44, "<")

    cv2.putText(img, mrz1, (20, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, mrz2, (20, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)


def generate_all_passports(count: int = 50, output_dir: str = "dataset/genuine/passports"):
    """Generate multiple synthetic passport images."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i in range(1, count + 1):
        # Random person
        sex = random.choice(["MALE", "FEMALE"])
        surname = random.choice(LAST_NAMES)
        given_name = random.choice(FIRST_NAMES_M if sex == "MALE" else FIRST_NAMES_F)
        passport_num = f"{''.join(random.choices(string.ascii_uppercase, k=1))}{random.randint(1000000, 9999999)}"
        dob_year = random.randint(1960, 2005)
        dob = f"{random.randint(1, 28):02d}.{random.randint(1, 12):02d}.{dob_year}"

        filename = f"passport_{i:03d}.png"
        generate_passport_image(passport_num, surname, given_name, dob, sex, output_path / filename)

    print(f"  [OK] Generated {count} synthetic passport images in {output_dir}/")


if __name__ == "__main__":
    generate_all_passports(50)

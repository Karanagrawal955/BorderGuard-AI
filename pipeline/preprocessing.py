"""
preprocessing.py
-----------------
Handles image loading, standardization, and Error Level Analysis (ELA).

ELA works because JPEG compression is lossy and applied in 8x8 pixel blocks.
When an image is saved as JPEG, every region loses a roughly consistent
amount of information. If someone edits *part* of an image and re-saves it,
that edited region was compressed a different number of times than the rest,
so it sits at a different "error level" than the untouched regions.

We exploit this by:
  1. Re-saving the image at a known JPEG quality.
  2. Computing the pixel-wise difference between the original and the
     re-saved version.
  3. Amplifying that difference so tampered regions become visually and
     numerically distinct from untouched regions.

Known limitation (mention this in Q&A - it shows you understand the
technique's boundaries): ELA is unreliable on images that have already
been re-compressed many times (e.g. repeatedly forwarded on WhatsApp) or
on true lossless PNGs, since there's no compression artifact to exploit.
"""

import io
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageChops


STANDARD_SIZE = (800, 800)   # (width, height) - documents get resized to this
ELA_JPEG_QUALITY = 90        # quality used for the ELA re-save step
ELA_AMPLIFICATION = 15       # multiplier to make ELA differences visible


@dataclass
class PreprocessedDocument:
    """Container for everything downstream modules need from one image."""
    source_path: str
    original_image: Image.Image      # standardized RGB PIL image
    ela_image: Image.Image           # amplified ELA difference image (RGB)
    ela_gray: np.ndarray             # single-channel ELA map, float32, 0-255
    was_resized: bool
    original_size: tuple             # (width, height) before resizing


def load_and_standardize(image_path: str, target_size=STANDARD_SIZE) -> tuple[Image.Image, bool, tuple]:
    """
    Load an image from disk and standardize it to a fixed size / RGB mode.

    Returns (standardized_image, was_resized, original_size).
    Raises a clear error if the file can't be read as an image at all -
    we do NOT want a silent failure here, since every downstream module
    depends on getting a valid image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No such file: {image_path}")

    try:
        img = Image.open(image_path)
        img.load()  # force decode now, so corrupt files fail here, not later
    except Exception as e:
        raise ValueError(f"Could not read '{image_path}' as an image: {e}")

    original_size = img.size

    # Normalize to RGB (handles palette images, RGBA, grayscale, CMYK scans)
    if img.mode != "RGB":
        img = img.convert("RGB")

    was_resized = img.size != target_size
    if was_resized:
        # LANCZOS keeps edges reasonably sharp, which matters for ELA/OCR
        img = img.resize(target_size, Image.LANCZOS)

    return img, was_resized, original_size


def compute_ela(
    image: Image.Image,
    jpeg_quality: int = ELA_JPEG_QUALITY,
    amplification: int = ELA_AMPLIFICATION,
) -> tuple[Image.Image, np.ndarray]:
    """
    Compute the Error Level Analysis map for a PIL image.

    Returns:
        ela_image: an RGB PIL image you can save/view directly
        ela_gray:  a float32 numpy array (H, W) with per-pixel error
                   magnitude, useful as a numeric feature for the classifier
    """
    # Step 1: re-save the image as JPEG at a known quality, in memory
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=jpeg_quality)
    buffer.seek(0)
    resaved = Image.open(buffer)
    resaved.load()

    # Step 2: pixel-wise absolute difference between original and re-saved
    diff = ImageChops.difference(image, resaved)

    # Step 3: amplify so differences are visible / usable as a signal
    diff_arr = np.asarray(diff).astype(np.float32)
    amplified = np.clip(diff_arr * amplification, 0, 255).astype(np.uint8)
    ela_image = Image.fromarray(amplified, mode="RGB")

    # Grayscale version for numeric feature extraction downstream
    ela_gray = np.asarray(ela_image.convert("L")).astype(np.float32)

    return ela_image, ela_gray


def preprocess_document(image_path: str) -> PreprocessedDocument:
    """
    Full preprocessing entry point used by the rest of the pipeline.
    Never raises on "the image looks weird" - only raises if the file
    genuinely cannot be read, per the graceful-degradation requirement.
    """
    standardized, was_resized, original_size = load_and_standardize(image_path)
    ela_image, ela_gray = compute_ela(standardized)

    return PreprocessedDocument(
        source_path=image_path,
        original_image=standardized,
        ela_image=ela_image,
        ela_gray=ela_gray,
        was_resized=was_resized,
        original_size=original_size,
    )


if __name__ == "__main__":
    # Quick self-test: generate a synthetic "document" and a tampered
    # version of it, run preprocessing on both, and sanity-check that
    # the tampered one produces a visibly different ELA signature.
    import sys

    print("Running preprocessing.py self-test...")

    # Build a fake plain document (solid background + a text-like block)
    from PIL import ImageDraw

    def make_fake_document(path, tamper=False):
        img = Image.new("RGB", (600, 380), color=(235, 235, 225))
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 580, 90], outline=(0, 0, 0), width=2)
        draw.text((30, 40), "SAMPLE FICTIONAL ID - NOT A REAL DOCUMENT", fill=(0, 0, 0))
        draw.rectangle([30, 120, 200, 260], outline=(0, 0, 0), width=2)  # "photo" box
        draw.text((30, 270), "Name: TEST PERSON", fill=(0, 0, 0))
        draw.text((30, 300), "ID No: 0000 0000 0000", fill=(0, 0, 0))

        # Save once as JPEG to simulate "the document as originally scanned"
        img.save(path, "JPEG", quality=92)

        if tamper:
            # Simulate tampering: re-open, paste a pasted-looking patch in
            # the "photo" region, and re-save. This re-compression is what
            # ELA should pick up on.
            tampered = Image.open(path).convert("RGB")
            patch = Image.new("RGB", (150, 120), color=(180, 160, 140))
            tampered.paste(patch, (35, 125))
            tampered.save(path, "JPEG", quality=92)

    genuine_path = "/tmp/fake_genuine.jpg"
    tampered_path = "/tmp/fake_tampered.jpg"
    make_fake_document(genuine_path, tamper=False)
    make_fake_document(tampered_path, tamper=True)

    doc_genuine = preprocess_document(genuine_path)
    doc_tampered = preprocess_document(tampered_path)

    print(f"Genuine  - ELA mean: {doc_genuine.ela_gray.mean():.3f}  "
          f"ELA max: {doc_genuine.ela_gray.max():.1f}  "
          f"ELA std: {doc_genuine.ela_gray.std():.3f}")
    print(f"Tampered - ELA mean: {doc_tampered.ela_gray.mean():.3f}  "
          f"ELA max: {doc_tampered.ela_gray.max():.1f}  "
          f"ELA std: {doc_tampered.ela_gray.std():.3f}")

    doc_genuine.ela_image.save("/tmp/ela_genuine_preview.png")
    doc_tampered.ela_image.save("/tmp/ela_tampered_preview.png")
    print("Saved ELA preview images to /tmp/ela_genuine_preview.png and /tmp/ela_tampered_preview.png")

    # The honest test isn't "is the global max/mean higher" - a single
    # tampered patch on an otherwise plain document barely moves global
    # stats. What actually matters (and what the real classifier will use)
    # is whether the tampered REGION has a distinguishable ELA signature
    # compared to a same-sized untouched region of the same image.
    # The tamper patch was pasted at (35, 125) to (185, 245) in the
    # 600x380 source image; account for the resize to STANDARD_SIZE.
    scale_x = STANDARD_SIZE[0] / 600
    scale_y = STANDARD_SIZE[1] / 380
    x0, y0 = int(35 * scale_x), int(125 * scale_y)
    x1, y1 = int(185 * scale_x), int(245 * scale_y)

    tampered_region = doc_tampered.ela_gray[y0:y1, x0:x1]
    # An untouched region of the same size, elsewhere in the same image
    control_region = doc_tampered.ela_gray[y0:y1, x0 + 250:x1 + 250]

    region_signal = tampered_region.mean()
    control_signal = control_region.mean()
    print(f"Tampered-region ELA mean: {region_signal:.3f}  "
          f"vs same-image control-region ELA mean: {control_signal:.3f}")

    if region_signal > control_signal * 1.15:
        print("PASS: the tampered region shows a measurably stronger ELA "
              "signal than an untouched region of the same document.")
        sys.exit(0)
    else:
        print("WARNING: region-level ELA signal was not clearly stronger - "
              "inspect preview images before relying on this for scoring.")
        sys.exit(1)

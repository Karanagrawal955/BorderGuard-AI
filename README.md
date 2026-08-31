# SIH26188 — AI-Based Fake Identity & Document Screening System

Document forgery detection pipeline: ELA-based tamper detection, OCR
consistency checks (Verhoeff/PAN checksum validation, font/spacing
anomaly detection), an adaptive-fit ML classifier, and a combined
0-100 authenticity score with a visual heatmap overlay.

## Setup (do this after cloning)

1. Install Python 3.12:
   py install 3.12

2. Create and activate a virtual environment:
   py -3.12 -m venv venv
   venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

4. Windows only: if `import torch` fails with a DLL error, install
   the Microsoft Visual C++ Redistributable:
   https://aka.ms/vs/17/release/vc_redist.x64.exe
   (then restart your machine before retrying)

5. Verify:
   python -c "import torch; print(torch.__version__)"
   python -c "import easyocr; print('easyocr OK')"

## Running the pipeline

Train the classifier:
   python -m pipeline.adaptive_learning

Screen a document:
   python run_pipeline.py --image data\genuine\genuine_00.jpg

## Important notes

- Use only fictional/synthetic ID templates for testing.
- Re-run adaptive_learning.py any time you add new images to
  data/genuine/ or data/tampered/.
- models/*.pkl and outputs/*.png are gitignored - regenerated locally.

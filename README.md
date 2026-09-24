# BorderGuard AI — SIH26188 Document Authenticity System

Web app that screens identity documents (Aadhaar, PAN, passport) for forgery.
Upload an image and it returns one of three verdicts:

| Verdict | Meaning |
|---|---|
| **DOCUMENT AUTHENTIC** | Reads as a genuine identity document, no tampering found |
| **TAMPERING DETECTED** | A real identity document with forensic/format inconsistencies |
| **INVALID DOCUMENT** | Not an identity document at all — a photo, poster, screenshot, etc. |

Detection combines ELA-based tamper analysis, EasyOCR text extraction, format
validation (Verhoeff/PAN checksums, Aadhaar/PAN/passport field rules), an ML
classifier, and a weighted 0–100 authenticity score with an explainability
breakdown in the UI.

## Running the app

Two servers must be running. The frontend proxies `/api` to the backend, so the
ports matter.

**1. Backend** (Flask, port 5000)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

The classifier trains on all 541 images on startup — wait ~35–45s for
`Flask backend ready` before uploading anything.

**2. Frontend** (Vite, port 3000) — in a second terminal

```powershell
cd frontend\borderguard-ai-main\borderguard-ai-main
npm install --legacy-peer-deps
npm run dev
```

Open **http://localhost:3000** → *Start document scan* → choose a file →
*Verify / Scan*.

## Tests and diagnostics

All of these need the backend running:

```powershell
python test_accuracy.py   # 20 cases: 10 genuine + 10 forged  -> 20/20
python test_nondoc.py     # 6 non-documents (photos + poster)  -> all rejected
python diag_score.py data\tampered\tampered_00.jpg   # per-component score breakdown
```

## How the "invalid document" gate works

An image is only accepted as an identity document if **both** hold:

1. OCR recovered at least 20 alphanumeric characters, **and**
2. The text contains actual identity-document evidence:
   an ID number (PAN / Aadhaar 12-digit / passport / MRZ), an issuing
   authority phrase (*Income Tax Department*, *Government of India*, UIDAI,
   *Passport*, *Driving Licence*, *Voter*), a `dd/mm/yyyy` date field, or a
   labelled `Name:` field.

Measured separation — random photographs read **0–11** characters, a
promotional poster **116**, genuine and forged identity documents **37–316**.
Patterns are deliberately OCR-tolerant because EasyOCR garbles these heavily
(`INCOMETaK`, `GOVTAORINDIA`, `Permanent Accouni Nun.`).

Non-documents are rejected with `decision=Reject`, `isDocument=false`,
`isTampered=false` — they are *not* reported as tampered.

## CLI pipeline (optional)

Does not need the servers:

```powershell
python -m pipeline.adaptive_learning                      # train classifier
python run_pipeline.py --image data\genuine\genuine_00.jpg
```

## Setup notes

- Python 3.11 or 3.12; Node 20+.
- Windows only: if `import torch` fails with a DLL error, install the Microsoft
  Visual C++ Redistributable:
  https://aka.ms/vs/17/release/vc_redist.x64.exe (restart after installing).
- Verify: `python -c "import torch; print(torch.__version__)"` and
  `python -c "import easyocr; print('easyocr OK')"`.
- `models/*.pkl` and `outputs/*.png` are gitignored and regenerated locally.
- Re-run `pipeline/adaptive_learning` after adding images to `data/genuine/`
  or `data/tampered/`.

## Known limitations

- The classifier learns from synthetic corpora and generalises weakly across
  corpora; deterministic checksums and hard score caps carry much of the
  verdict.
- The invalid-document gate is rule-based, not a trained classifier. Content
  that happens to contain a date or a `Name:` label (an invoice, a receipt) can
  still pass it and will then be scored normally — usually landing on *Review*.
- `pipeline/trufor_detector.py` falls back to ELA when TruFor weights are
  absent, so the heatmap is a weak discriminator on its own.

## Important notes

- Use only fictional/synthetic ID templates for testing.

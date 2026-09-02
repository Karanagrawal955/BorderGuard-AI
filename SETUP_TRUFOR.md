# Tonight's setup: wiring in TruFor + EasyOCR

Everything in `pipeline/` and `run_pipeline.py` was built and tested in a
sandbox with no internet access, using ELA-only features and a mocked
OCR function. It all runs and passes today. Tonight, on a machine with
internet, do this to get the real signal sources plugged in.

## 1. Install dependencies

```bash
pip install -r requirements.txt --break-system-packages
```

(Drop `--break-system-packages` if you're in a virtualenv, which is
recommended if you have time to set one up - but don't let venv setup
eat into your build time if you're rushing.)

## 2. Enable real OCR (5 minutes)

Open `run_pipeline.py`, find the `run_ocr()` function, and replace its
body with the EasyOCR code already written in its docstring. It's a
direct copy-paste - the `TextBox` shape it needs to return is already
defined and used everywhere else in the pipeline, so nothing else
changes.

Test it:
```bash
python run_pipeline.py --image data/genuine/genuine_00.jpg
```
You should now see real extracted text and OCR consistency flags
instead of the "OCR could not read this document" sandbox message.

## 3. Install TruFor (30-45 minutes - budget real time for this)

TruFor's official repo: https://github.com/grip-unina/TruFor

```bash
git clone https://github.com/grip-unina/TruFor third_party/TruFor
```

Follow their README to download the pretrained checkpoint (they host it
on a Google Drive / Dropbox link in the repo - grab the version trained
on generic manipulation detection, not a task-specific variant).

## 4. Extend `extract_features()` in `pipeline/classifier.py`

This is the ONE function you need to touch. Right now it only computes
ELA statistics. Add TruFor inference here:

```python
def extract_features(doc, label=None):
    ela = doc.ela_gray
    # ... existing ELA feature code stays exactly as-is ...

    # NEW: run TruFor on doc.original_image, get its manipulation
    # heatmap + confidence map, and pull a few summary statistics from
    # it the same way we did for ELA (mean, std, hotspot ratio, etc.)
    trufor_heatmap, trufor_confidence = run_trufor_inference(doc.original_image)
    trufor_mean = float(trufor_heatmap.mean())
    trufor_max = float(trufor_heatmap.max())
    # ... etc.

    values = [mean, std, max_val, p95, hotspot_ratio, region_variance,
              trufor_mean, trufor_max]  # extend FEATURE_NAMES to match
    ...
```

Also add the new feature names to `FEATURE_NAMES` at the top of
`classifier.py`, in the same order you append values above - this is
the only place that needs to stay in sync.

Do NOT change anything in `adaptive_learning.py`, `scoring.py`, or
`run_pipeline.py` - they all consume `FeatureVector.values` generically
and don't care how many features it holds.

## 5. Re-run adaptive training and the CLI to confirm everything still works

```bash
python -m pipeline.adaptive_learning
python run_pipeline.py --image data/genuine/genuine_00.jpg
```

If both run clean, you're ready for Day 2 (face-embedding matching +
copy-move detection layer).

## Known friction points to budget time for

- TruFor's checkpoint download can be slow/flaky depending on where
  they're hosting it that week - do this FIRST tonight, in parallel
  with someone else on OCR integration, so a slow download doesn't
  block the whole team.
- If TruFor's environment has dependency conflicts with your existing
  torch version, don't fight it for hours - a virtualenv just for
  TruFor inference (called as a subprocess) is a faster unblock than
  resolving version conflicts.
- EasyOCR downloads its own model weights on first run too (~100MB) -
  run it once tonight so that download isn't happening live during
  tomorrow's work.

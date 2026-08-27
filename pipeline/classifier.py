"""
classifier.py
-------------
The lightweight, interpretable classifier head + the "adaptive learning"
module that re-fits it whenever new labeled documents are dropped into
data/genuine/ and data/tampered/.

IMPORTANT DESIGN NOTE (read this before Day 2):
This sandbox has no internet access, so TruFor (which needs torch +
downloaded pretrained weights) can't run here. To keep Day 1 fully
testable end-to-end, this module currently builds its feature vector
from ELA statistics ONLY. On your machine tonight, once TruFor is
integrated (see truFor_integration.py), you'll extend
`extract_features()` to also include TruFor's heatmap statistics -
the rest of this module (the classifier + adaptive fit logic) does
NOT need to change at all. That's intentional: feature extraction and
classification are kept as separate, swappable stages.

Why scikit-learn instead of a PyTorch MLP for the classifier head:
With ~50 images total, a PyTorch MLP is more likely to overfit and is
slower to iterate on than a simple, well-regularized logistic
regression - which is also inherently more interpretable (each
feature gets a signed weight you can literally read out loud during
Q&A). If you want a PyTorch version later for the "adaptive/continual"
narrative to sound more sophisticated, the interface below
(`AdaptiveClassifier.fit_incremental()`) is designed to be swappable -
see the note in that method.

What "adaptive fine-tuning" means here, precisely (be ready to explain
this honestly if asked): this is NOT continual learning in the formal
ML sense (no EWC, no replay buffers, no protection against catastrophic
forgetting). It's an incremental re-fit: every time you drop new labeled
images into data/genuine/ or data/tampered/, this script re-extracts
features for the whole dataset and re-fits a small, fast classifier.
For a dataset this size (tens of images), that's the right amount of
sophistication - a heavier "true" continual-learning setup would be
unstable and slower to demo. Say this plainly if a judge asks; it shows
technical maturity rather than overselling the term.
"""

import json
import os
import pickle
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pipeline.preprocessing import preprocess_document, PreprocessedDocument


FEATURE_NAMES = [
    "ela_mean",
    "ela_std",
    "ela_max",
    "ela_p95",              # 95th percentile - robust to single hot pixels
    "ela_hotspot_ratio",    # fraction of pixels above an adaptive threshold
    "ela_region_variance",  # how unevenly the ELA signal is distributed spatially
]

MODEL_PATH = "models/classifier_head.pkl"
SCALER_PATH = "models/feature_scaler.pkl"
METADATA_PATH = "models/training_metadata.json"


@dataclass
class FeatureVector:
    source_path: str
    label: int | None   # 1 = tampered, 0 = genuine, None = unlabeled/inference-only
    values: list         # matches FEATURE_NAMES order


def extract_features(doc: PreprocessedDocument, label: int | None = None) -> FeatureVector:
    """
    Extract a compact, interpretable feature vector from a preprocessed
    document. Currently ELA-only (see module docstring for why, and what
    changes when TruFor is added on Day 2).
    """
    ela = doc.ela_gray

    mean = float(ela.mean())
    std = float(ela.std())
    max_val = float(ela.max())
    p95 = float(np.percentile(ela, 95))

    # Adaptive hotspot threshold: mean + 2 std devs, a classic "outlier
    # pixel" heuristic. Ratio of pixels above this = how much of the
    # image shows an anomalous compression signature.
    threshold = mean + 2 * std
    hotspot_ratio = float((ela > threshold).sum() / ela.size) if ela.size else 0.0

    # Region variance: split into a coarse 4x4 grid, compute mean ELA per
    # cell, then take the variance ACROSS cells. Real tampering tends to
    # be localized (a few "hot" cells among mostly quiet ones), whereas
    # uniform re-compression of the whole image spreads noise evenly.
    h, w = ela.shape
    gh, gw = max(h // 4, 1), max(w // 4, 1)
    cell_means = []
    for i in range(0, h, gh):
        for j in range(0, w, gw):
            cell = ela[i:i + gh, j:j + gw]
            if cell.size:
                cell_means.append(cell.mean())
    region_variance = float(np.var(cell_means)) if cell_means else 0.0

    values = [mean, std, max_val, p95, hotspot_ratio, region_variance]

    return FeatureVector(source_path=doc.source_path, label=label, values=values)


class AdaptiveClassifier:
    """
    Small, interpretable classifier head with an incremental re-fit
    workflow. Wraps scikit-learn's LogisticRegression + StandardScaler.

    Swap-to-PyTorch note: if you later want a PyTorch MLP here instead
    (e.g. to combine with TruFor features more expressively), keep the
    same three public methods (`fit_incremental`, `predict`,
    `explain_prediction`) so nothing else in the pipeline has to change.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",  # handles genuine/tampered count imbalance
        )
        self.is_fitted = False
        self.n_samples_trained_on = 0

    def fit_incremental(self, feature_vectors: list[FeatureVector]) -> dict:
        """
        Re-fit the classifier on the CURRENT full set of labeled feature
        vectors. This is called "incremental" from the user's workflow
        perspective (you incrementally add images and re-run this), but
        under the hood it's a full re-fit each time - which is the right
        choice at this dataset scale (see module docstring).
        """
        labeled = [fv for fv in feature_vectors if fv.label is not None]
        if len(labeled) < 4:
            raise ValueError(
                f"Need at least 4 labeled samples to fit (2 per class "
                f"minimum recommended); got {len(labeled)}. Add more images "
                f"to data/genuine/ and data/tampered/."
            )

        labels = [fv.label for fv in labeled]
        if len(set(labels)) < 2:
            raise ValueError(
                "Need at least one example of BOTH genuine and tampered "
                "documents to train a classifier - only one class was found."
            )

        X = np.array([fv.values for fv in labeled])
        y = np.array(labels)

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        self.n_samples_trained_on = len(labeled)

        # Report training-set accuracy for transparency (NOT a claim of
        # real-world generalization - just confirms the fit ran sanely).
        train_acc = self.model.score(X_scaled, y)

        return {
            "n_samples": len(labeled),
            "n_genuine": int((y == 0).sum()),
            "n_tampered": int((y == 1).sum()),
            "train_accuracy": float(train_acc),
            "feature_weights": dict(zip(FEATURE_NAMES, self.model.coef_[0].tolist())),
        }

    def predict(self, feature_vector: FeatureVector) -> dict:
        """
        Returns a dict with probability of tampering and an interpretable
        breakdown of which features drove the decision.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Classifier has not been trained yet. Run fit_incremental() "
                "first (see adaptive_learning.py)."
            )

        x = np.array([feature_vector.values])
        x_scaled = self.scaler.transform(x)
        proba = self.model.predict_proba(x_scaled)[0]  # [P(genuine), P(tampered)]

        return {
            "p_tampered": float(proba[1]),
            "p_genuine": float(proba[0]),
            "predicted_label": int(proba[1] > 0.5),
        }

    def explain_prediction(self, feature_vector: FeatureVector) -> list[dict]:
        """
        Returns each feature's contribution to the tampered-vs-genuine
        decision, sorted by magnitude - this is what you show a judge
        who asks "why did it flag this document?"
        """
        if not self.is_fitted:
            return []

        x = np.array([feature_vector.values])
        x_scaled = self.scaler.transform(x)[0]
        weights = self.model.coef_[0]
        contributions = x_scaled * weights

        explanation = [
            {
                "feature": name,
                "raw_value": round(float(val), 4),
                "weight": round(float(w), 4),
                "contribution": round(float(c), 4),
            }
            for name, val, w, c in zip(FEATURE_NAMES, feature_vector.values, weights, contributions)
        ]
        explanation.sort(key=lambda d: abs(d["contribution"]), reverse=True)
        return explanation

    def save(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

    @classmethod
    def load(cls, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        obj = cls()
        with open(model_path, "rb") as f:
            obj.model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            obj.scaler = pickle.load(f)
        obj.is_fitted = True
        return obj


if __name__ == "__main__":
    print("Running classifier.py self-test...\n")

    from PIL import Image, ImageDraw
    import random

    random.seed(42)
    np.random.seed(42)

    os.makedirs("/tmp/synthetic_docs/genuine", exist_ok=True)
    os.makedirs("/tmp/synthetic_docs/tampered", exist_ok=True)

    def make_synthetic_doc(path, tamper: bool):
        img = Image.new("RGB", (600, 380), color=(235, 235, 225))
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 580, 90], outline=(0, 0, 0), width=2)
        draw.text((30, 40), "SAMPLE FICTIONAL ID - NOT REAL", fill=(0, 0, 0))
        draw.rectangle([30, 120, 200, 260], outline=(0, 0, 0), width=2)
        draw.text((30, 270), f"Name: PERSON {random.randint(1,999)}", fill=(0, 0, 0))
        img.save(path, "JPEG", quality=92)

        if tamper:
            tampered_img = Image.open(path).convert("RGB")
            patch_color = (
                random.randint(120, 200),
                random.randint(100, 180),
                random.randint(80, 160),
            )
            patch = Image.new("RGB", (150, 120), color=patch_color)
            tampered_img.paste(patch, (35, 125))
            tampered_img.save(path, "JPEG", quality=random.choice([85, 92, 96]))

    # Generate a small synthetic dataset: 12 genuine, 12 tampered
    genuine_paths = []
    tampered_paths = []
    for i in range(12):
        p = f"/tmp/synthetic_docs/genuine/doc_{i}.jpg"
        make_synthetic_doc(p, tamper=False)
        genuine_paths.append(p)

    for i in range(12):
        p = f"/tmp/synthetic_docs/tampered/doc_{i}.jpg"
        make_synthetic_doc(p, tamper=True)
        tampered_paths.append(p)

    print(f"Generated {len(genuine_paths)} synthetic genuine docs and "
          f"{len(tampered_paths)} synthetic tampered docs.\n")

    # Extract features
    feature_vectors = []
    for p in genuine_paths:
        doc = preprocess_document(p)
        feature_vectors.append(extract_features(doc, label=0))
    for p in tampered_paths:
        doc = preprocess_document(p)
        feature_vectors.append(extract_features(doc, label=1))

    print("Sample feature vector (genuine):", feature_vectors[0].values)
    print("Sample feature vector (tampered):", feature_vectors[-1].values)

    # Train/test split by hand (small dataset - simple holdout)
    random.shuffle(feature_vectors)
    split = int(len(feature_vectors) * 0.75)
    train_fv, test_fv = feature_vectors[:split], feature_vectors[split:]

    clf = AdaptiveClassifier()
    fit_result = clf.fit_incremental(train_fv)
    print(f"\nFit result: {fit_result}\n")

    # Evaluate on held-out test set - THIS is the honest number, not
    # train_accuracy above.
    correct = 0
    for fv in test_fv:
        pred = clf.predict(fv)
        is_correct = pred["predicted_label"] == fv.label
        correct += int(is_correct)
    test_acc = correct / len(test_fv) if test_fv else float("nan")
    print(f"Held-out test accuracy on {len(test_fv)} samples: {test_acc:.2%}")
    print("(This number is from a tiny synthetic dataset - report it as "
          "'X% on our N-sample validation set', never as a general claim.)\n")

    # Explainability check
    sample = test_fv[0]
    explanation = clf.explain_prediction(sample)
    print(f"Explanation for one prediction (true label={sample.label}):")
    for row in explanation:
        print(f"  {row['feature']:22s} value={row['raw_value']:>8.3f}  "
              f"weight={row['weight']:>7.3f}  contribution={row['contribution']:>7.3f}")

    # Save/load round-trip test
    clf.save()
    clf2 = AdaptiveClassifier.load()
    pred_original = clf.predict(sample)
    pred_reloaded = clf2.predict(sample)
    assert abs(pred_original["p_tampered"] - pred_reloaded["p_tampered"]) < 1e-9, \
        "Reloaded model should produce identical predictions"
    print(f"\nPASS: model save/load round-trip produces identical predictions "
          f"(p_tampered={pred_reloaded['p_tampered']:.4f})")

    print("\nAll classifier.py self-tests passed.")

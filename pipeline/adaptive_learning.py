"""
adaptive_learning.py — Incremental classifier fine-tuning

Maintains a lightweight classifier head (2-layer MLP or Logistic Regression)
that combines TruFor features + ELA statistics to predict tampering.

The classifier is re-trainable whenever new labeled samples are added to
/data/genuine/ or /data/tampered/. It does NOT modify the TruFor backbone.

Design choices for hackathon speed:
    - Uses LogisticRegression by default (fast, interpretable, no GPU needed)
    - Falls back to a small PyTorch MLP if more flexibility needed
    - Supports incremental partial_fit() for online learning
    - Saves/loads model via joblib or torch
"""

import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Classifier imports
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import joblib


# Feature dimensionality (must match feature_extraction.py output)
FEATURE_DIM = 19


class AdaptiveClassifier:
    """
    Lightweight classifier that predicts tampering probability from
    TruFor + ELA features. Supports incremental learning.

    Architecture:
        - StandardScaler → SGDClassifier(loss='log_loss')
        - This gives us logistic regression with incremental learning
        - Features are interpretable (feature_names from feature_extraction)

    Usage:
        clf = AdaptiveClassifier()
        # Initial training
        clf.fit_from_folders("data/genuine", "data/tampered")
        # Later, after adding new samples:
        clf.fit_from_folders("data/genuine", "data/tampered")  # re-runnable
        # Predict
        score = clf.predict(features)
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize or load the classifier.

        Args:
            model_path: Path to saved model. If None, uses models/classifier.pkl
        """
        self.model_path = Path(model_path or "models/classifier.pkl")
        self.feature_names: List[str] = []
        self.is_trained = False

        # Initialize pipeline
        if SKLEARN_AVAILABLE:
            self.pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=3,
                    min_samples_leaf=2,
                    random_state=42,
                    class_weight="balanced",
                )),
            ])
        else:
            self.pipeline = None
            print("[AdaptiveClassifier] WARNING: scikit-learn not available.")
            print("  => Install with: pip install scikit-learn")

        # Try to load existing model
        self._load_model()

    def _load_model(self):
        """Load saved model and scaler if available."""
        if self.model_path.exists() and self.pipeline is not None:
            try:
                saved = joblib.load(self.model_path)
                self.pipeline = saved["pipeline"]
                self.feature_names = saved.get("feature_names", [])
                self.is_trained = True
                print(f"[AdaptiveClassifier] Loaded model from {self.model_path}")
            except Exception as e:
                print(f"[AdaptiveClassifier] Could not load model: {e}")

    def _save_model(self):
        """Save current model to disk."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "pipeline": self.pipeline,
            "feature_names": self.feature_names,
        }, self.model_path)
        print(f"[AdaptiveClassifier] Model saved to {self.model_path}")

    def fit_from_folders(
        self,
        genuine_dir: str = "data/genuine",
        tampered_dir: str = "data/tampered",
    ) -> dict:
        """
        (Re-)train the classifier from labeled images in the data folders.

        This function is re-runnable: it scans both folders, extracts features
        from all images, and fits/updates the classifier.

        Returns dict with training stats.
        """
        if self.pipeline is None:
            return {"error": "scikit-learn not available"}

        start_time = time.time()
        genuine_path = Path(genuine_dir)
        tampered_path = Path(tampered_dir)

        # Collect all image paths (including subdirectories)
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        genuine_images = [
            p for p in genuine_path.rglob("*")
            if p.suffix.lower() in image_exts and p.is_file()
        ] if genuine_path.exists() else []
        tampered_images = [
            p for p in tampered_path.rglob("*")
            if p.suffix.lower() in image_exts and p.is_file()
        ] if tampered_path.exists() else []

        if not genuine_images and not tampered_images:
            print("[AdaptiveClassifier] No training images found.")
            print(f"  => Add images to: {genuine_dir}/ and {tampered_dir}/")
            return {"genuine": 0, "tampered": 0, "error": "no images"}

        print(f"[AdaptiveClassifier] Found {len(genuine_images)} genuine, "
              f"{len(tampered_images)} tampered images")

        # Lazy-import pipeline components to avoid circular imports
        from pipeline.preprocessing import preprocess_image
        from pipeline.trufor_detector import TruForDetector
        from pipeline.feature_extraction import extract_features

        # Initialize detector (shared across all images)
        detector = TruForDetector()

        features_list = []
        labels = []

        # Process genuine images (label=0)
        for img_path in genuine_images:
            try:
                feat_vec = self._extract_one(img_path, detector)
                if feat_vec is not None:
                    features_list.append(feat_vec)
                    labels.append(0)
            except Exception as e:
                print(f"  [WARN] Skipping {img_path.name}: {e}")

        # Process tampered images (label=1)
        for img_path in tampered_images:
            try:
                feat_vec = self._extract_one(img_path, detector)
                if feat_vec is not None:
                    features_list.append(feat_vec)
                    labels.append(1)
            except Exception as e:
                print(f"  [WARN] Skipping {img_path.name}: {e}")

        if not features_list:
            return {"error": "no features extracted"}

        X = np.array(features_list)
        y = np.array(labels)

        # Fit or update the classifier
        # Always do a full re-fit. With a dataset this small (tens of images,
        # not thousands), incremental partial_fit() offers no real benefit and
        # has a real risk: calling partial_fit() on the raw classifier bypasses
        # the pipeline's StandardScaler step entirely, causing a train/predict
        # scale mismatch since predict() always goes through the full scaled
        # pipeline. A full re-fit is fast at this scale and avoids that bug.
        self.pipeline.fit(X, y)
        self.is_trained = True
        print("[AdaptiveClassifier] Full training completed")

        # Save
        self._save_model()

        elapsed = time.time() - start_time
        stats = {
            "genuine_count": len(genuine_images),
            "tampered_count": len(tampered_images),
            "total_features": len(features_list),
            "training_time_sec": round(elapsed, 2),
            "is_incremental": self.is_trained,
        }

        print(f"[AdaptiveClassifier] Training complete in {elapsed:.2f}s")
        return stats

    def _extract_one(self, img_path: Path, detector) -> Optional[np.ndarray]:
        """Extract features from a single image. Returns None on failure."""
        from pipeline.preprocessing import preprocess_image
        from pipeline.feature_extraction import extract_features

        preprocessed = preprocess_image(img_path)
        detection = detector.detect(preprocessed["original"])
        feature_result = extract_features(
            preprocessed["original"],
            detection["heatmap"],
            preprocessed["ela_map"],
            preprocessed["ela_stats"],
        )
        return feature_result["features"]

    def predict(self, features: np.ndarray) -> dict:
        """
        Predict tampering probability for a feature vector.

        Returns dict with:
            probability : float [0, 1] — P(tampered)
            prediction  : int — 0 (genuine) or 1 (tampered)
            confidence  : float — how confident the prediction is
        """
        if self.pipeline is None or not self.is_trained:
            # Fallback: return neutral/uncertain prediction
            return {
                "probability": 0.5,
                "prediction": 0,
                "confidence": 0.0,
                "message": "Classifier not trained — returning neutral score",
            }

        features_2d = features.reshape(1, -1)

        try:
            proba = self.pipeline.predict_proba(features_2d)[0]
            # proba[0] = P(genuine), proba[1] = P(tampered)
            tampered_prob = float(proba[1])
            prediction = int(self.pipeline.predict(features_2d)[0])
            confidence = float(max(proba))

            return {
                "probability": tampered_prob,
                "prediction": prediction,
                "confidence": confidence,
            }
        except Exception as e:
            return {
                "probability": 0.5,
                "prediction": 0,
                "confidence": 0.0,
                "message": f"Prediction failed: {e}",
            }

    def explain_prediction(self, features: np.ndarray, feature_names: List[str]) -> dict:
        """
        Provide a simple feature-importance explanation.

        For logistic regression, the coefficients tell us which features
        push toward "tampered" (positive) vs "genuine" (negative).
        """
        if self.pipeline is None or not self.is_trained:
            return {"explanation": "Model not trained"}

        clf = self.pipeline.named_steps["clf"]
        if not hasattr(clf, "coef_"):
            return {"explanation": "Model coefficients not available"}

        coefficients = clf.coef_[0]
        if len(coefficients) != len(features):
            return {"explanation": "Feature dimension mismatch"}

        # Compute feature contributions
        contributions = {}
        for name, feat_val, coef in zip(feature_names, features, coefficients):
            contributions[name] = {
                "value": float(feat_val),
                "coefficient": float(coef),
                "contribution": float(feat_val * coef),
            }

        # Sort by absolute contribution
        sorted_contribs = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]["contribution"]),
            reverse=True,
        )

        return {
            "top_tampering_indicators": [
                {"feature": name, **vals}
                for name, vals in sorted_contribs[:5]
            ],
            "top_genuine_indicators": [
                {"feature": name, **vals}
                for name, vals in sorted_contribs[-3:]
            ],
        }

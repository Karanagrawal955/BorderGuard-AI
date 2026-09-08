import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

# Ensure project root is in Python path (dynamic, no hardcoded C:\test)
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.preprocessing import preprocess_image
from pipeline.trufor_detector import TruForDetector
from pipeline.feature_extraction import extract_features
from pipeline.scoring import compute_authenticity_score
from pipeline.adaptive_learning import AdaptiveClassifier
from pipeline.mrz_module import extract_mrz
from pipeline.tamper_detector import detect_tampering
from pipeline.document_detector import is_document
from pipeline.classifier import train as train_classifier, predict as classifier_predict, is_loaded as classifier_loaded

app = Flask(__name__)
# Allow frontend from Vite (3000, 5173) and any localhost during dev
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=False)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload

UPLOAD_FOLDER = PROJECT_ROOT / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Reuse single instances where possible (avoid reloading heavy models per request)
_detector = None
_classifier = None
_ocr_processor = None

def get_detector():
    global _detector
    if _detector is None:
        _detector = TruForDetector()
    return _detector

def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = AdaptiveClassifier()
    return _classifier

def get_ocr_processor():
    global _ocr_processor
    if _ocr_processor is None:
        try:
            from pipeline.ocr_module import OCRProcessor
            _ocr_processor = OCRProcessor()
        except Exception as e:
            print(f"[app] OCR init failed: {e}")
            _ocr_processor = None
    return _ocr_processor

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}


def run_full_pipeline(image_path):
    """Run the complete SIH26188 pipeline on an image and return structured results.

    No terminal input is required — this is called directly from the Flask
    endpoint with the uploaded file path.
    Returns a dict containing both detailed breakdown and flattened
    ScanResult fields for the existing Frontend UI.
    """
    start_time = time.time()

    # Step 1: Preprocessing (ELA, EXIF, compression analysis)
    try:
        preprocessed = preprocess_image(image_path)
        # Ensure full_res is available (full resolution image for OCR)
        if "full_res" not in preprocessed:
            try:
                _full = cv2.imread(str(image_path))
                if _full is None:
                    _full = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
                preprocessed["full_res"] = _full if _full is not None else preprocessed["original"]
            except Exception:
                preprocessed["full_res"] = preprocessed["original"]
        ela_stats = preprocessed.get("ela_stats", {})
        ela_mean = float(ela_stats.get("mean", 0.0))
        hotspot_count = int(ela_stats.get("hotspot_count", 0))
        hotspot_ratio = float(ela_stats.get("hotspot_ratio", 0.0))
        double_jpeg = preprocessed.get("double_jpeg", {})
        exif_data = preprocessed.get("exif_data", None)
        # For JSON serialization, convert exif to dict if dataclass
        exif_dict = {}
        if exif_data is not None and hasattr(exif_data, "__dict__"):
            try:
                exif_dict = {k: str(v) if v is not None else None for k, v in exif_data.__dict__.items()}
            except Exception:
                exif_dict = {"has_exif": bool(getattr(exif_data, "has_exif", False))}
    except Exception as e:
        print(f"[pipeline] preprocessing failed: {e}")
        # Create minimal fallback so downstream steps don't crash
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((512, 512, 3), dtype=np.uint8)
            preprocessed = {
                "original": cv2.resize(img, (512, 512)),
                "full_res": img,
                "ela_map": np.zeros((512, 512), dtype=np.uint8),
                "ela_stats": {"mean": 0.0, "variance": 0.0, "hotspot_count": 0, "hotspot_ratio": 0.0, "max_error": 0.0},
            }
        except Exception as e2:
            raise RuntimeError(f"Preprocessing failed: {e} / fallback failed: {e2}")
        ela_stats = preprocessed["ela_stats"]
        ela_mean = float(ela_stats.get("mean", 0.0))
        hotspot_count = int(ela_stats.get("hotspot_count", 0))
        hotspot_ratio = float(ela_stats.get("hotspot_ratio", 0.0))
        double_jpeg = {}
        exif_dict = {}

    # Step 1.5: Document detection — is this even a document?
    doc_detection = {"is_document": True, "confidence": 0.5, "reason": "skipped", "scores": {}}
    try:
        doc_detection = is_document(preprocessed["original"])
        print(f"[pipeline] Document detection: is_doc={doc_detection['is_document']}, conf={doc_detection['confidence']:.2f}, reason={doc_detection['reason']}")
    except Exception as e:
        print(f"[pipeline] Document detection failed: {e}")

    # Step 2: TruFor tamper detection
    try:
        detector = get_detector()
        detection_result = detector.detect(preprocessed["original"])
        heatmap = detection_result.get("heatmap", np.full((512,512), 0.5, dtype=np.float32))
        if heatmap is None:
            heatmap = np.full((512,512), 0.5, dtype=np.float32)
        heatmap_mean = float(np.mean(heatmap))
        heatmap_max = float(np.max(heatmap))
        heatmap_std = float(np.std(heatmap))
        # Use detection score if available, else heatmap mean as tamper signal
        is_tampered_heuristic = heatmap_mean > 0.45 or heatmap_max > 0.85
    except Exception as e:
        print(f"[pipeline] TruFor failed: {e}")
        heatmap = np.full((512,512), 0.5, dtype=np.float32)
        heatmap_mean = 0.5
        heatmap_max = 0.5
        heatmap_std = 0.0
        is_tampered_heuristic = False
        detection_result = {"heatmap": heatmap, "source": "fallback"}

    # Step 3: Feature extraction
    try:
        features_result = extract_features(
            preprocessed["original"],
            heatmap,
            preprocessed["ela_map"],
            preprocessed["ela_stats"],
        )
        features = features_result["features"]
        feature_names = features_result["feature_names"]
    except Exception as e:
        print(f"[pipeline] feature extraction failed: {e}")
        # Fallback dummy features — must match FEATURE_DIM=15 from adaptive_learning.py
        features = np.zeros(19, dtype=np.float32)
        feature_names = [
            "heatmap_mean", "heatmap_std", "heatmap_hotspot_ratio",
            "ela_spatial_asymmetry", "ela_spatial_concentration",
            "ela_spatial_center_border", "ela_spatial_center_mean",
            "mq_ela_ratio_mean", "mq_ela_ratio_std", "mq_ela_combined",
            "noise_estimate", "noise_snr", "noise_inconsistency",
            "texture_laplacian_var", "texture_saturation",
            "texture_gradient_mag", "texture_richness",
            "compression_block_var", "compression_periodic_score",
        ]
        # Populate with at least TruFor stats
        try:
            features[0] = heatmap_mean
            features[1] = heatmap_std
            features[2] = heatmap_max
        except Exception:
            pass

    # Step 4: Trained tamper classifier (uses original full-res image, not resized)
    try:
        # Load original image at full resolution for classifier (resizing destroys ELA patterns)
        _orig_full = cv2.imread(str(image_path))
        if _orig_full is None:
            _orig_full = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        cls_input = _orig_full if _orig_full is not None else preprocessed["original"]
        cls_result = classifier_predict(cls_input)
        classifier_output = {
            "probability": cls_result["probability"],
            "prediction": cls_result["prediction"],
            "confidence": cls_result["confidence"],
            "model_loaded": cls_result.get("model_loaded", False),
        }
        print(f"[pipeline] Classifier: prob={cls_result['probability']:.3f}, pred={cls_result['prediction']}, conf={cls_result['confidence']:.3f}")
    except Exception as e:
        print(f"[pipeline] Classifier failed: {e}")
        classifier_output = {"probability": 0.5, "prediction": 0, "confidence": 0.0, "model_loaded": False}

    # Step 5: OCR extraction (on full_res for best accuracy)
    ocr_result = None
    ocr_text = ""
    ocr_fields = {"name": "", "dob": "", "document_number": "", "expiry_date": ""}
    try:
        ocr_proc = get_ocr_processor()
        if ocr_proc is not None:
            # Use full_res if available, else original
            ocr_input = preprocessed.get("full_res", preprocessed["original"])
            ocr_result = ocr_proc.process(ocr_input)
            if ocr_result and ocr_result.full_text:
                ocr_text = ocr_result.full_text[:1000]
                # Try to parse fields naively for UI compatibility
                lines = ocr_text.split("\n")
                for line in lines:
                    line_s = line.strip()
                    if not line_s:
                        continue
                    lower = line_s.lower()
                    # Check for "keyword: value" pattern first
                    val = ""
                    if ":" in line_s:
                        parts = line_s.split(":", 1)
                        kw = parts[0].strip().lower()
                        val = parts[1].strip()
                    else:
                        kw = lower
                        val = line_s

                    if ("name" in kw or "(holder" in kw) and not ocr_fields["name"] and val:
                        ocr_fields["name"] = val[:120]
                    elif ("dob" in kw or "date of birth" in kw or "birth" in kw) and not ocr_fields["dob"] and val:
                        ocr_fields["dob"] = val[:120]
                    elif ("passport" in kw or "aadhaar" in kw or "aadhar" in kw or "pan" in kw or "license" in kw or "id no" in kw or "document no" in kw or "enrolment" in kw) and not ocr_fields["document_number"] and val:
                        ocr_fields["document_number"] = val[:120]
                    elif ("expiry" in kw or "valid until" in kw or "date of expiry" in kw or "expires" in kw) and not ocr_fields["expiry_date"] and val:
                        ocr_fields["expiry_date"] = val[:120]
                if not any(ocr_fields.values()) and ocr_text:
                    # Fallback: put first 100 chars as name
                    ocr_fields["name"] = ocr_text[:80]
            else:
                ocr_text = ocr_result.full_text if ocr_result else ""
        else:
            ocr_text = "OCR unavailable"
    except Exception as e:
        print(f"[pipeline] OCR failed: {e}")
        ocr_text = f"OCR failed: {str(e)[:120]}"
        # Create a minimal ocr_result for scoring (so scoring doesn't crash)
        try:
            from pipeline.ocr_module import OCRResult
            ocr_result = OCRResult(full_text=ocr_text, is_readable=False, overall_consistency=0.5)
        except Exception:
            ocr_result = None

    # Step 6: MRZ parsing
    mrz_valid = False
    mrz_data = {}
    mrz_raw = ""
    try:
        mrz_result = extract_mrz(str(image_path))
        if mrz_result and mrz_result.get("found"):
            mrz_data = mrz_result.get("fields", {})
            mrz_raw = mrz_result.get("raw_text", "")
            mrz_valid = True
        else:
            mrz_valid = False
    except Exception as e:
        print(f"[pipeline] MRZ failed: {e}")
        mrz_data = {}
        mrz_valid = False

    # Step 7: Face verification (optional, not required for document score)
    face_match_score = None
    face_details = None
    try:
        # No live capture provided via this endpoint, so skip actual face comparison
        # Keep placeholder for UI
        face_match_score = 0.0
        face_details = None
    except Exception:
        pass

    # Step 8: Scoring engine — real weighted scoring
    try:
        # Ensure heatmap is passed correctly
        score_result = compute_authenticity_score(
            classifier_output=classifier_output,
            heatmap_mean=heatmap_mean,
            ocr_result=ocr_result,
            validation_result=None,
            face_result=None,
            extracted_fields=ocr_fields,
            ocr_text=ocr_text,
            image_shape=preprocessed["original"].shape if preprocessed.get("original") is not None else None,
        )
        overall_score = float(score_result.final_score)
        risk_level_raw = score_result.risk_level  # GENUINE / SUSPICIOUS / LIKELY_FORGED / FORGED
        flags = score_result.flags
        summary = score_result.summary
        classifier_component = score_result.classifier_component
        trufor_component = score_result.trufor_component
        ocr_component = score_result.ocr_component
        doc_val_component = score_result.validation_component
        face_component = score_result.face_component
    except Exception as e:
        print(f"[pipeline] scoring failed: {e}")
        import traceback
        traceback.print_exc()
        # Fallback scoring
        overall_score = 50.0
        risk_level_raw = "LIKELY_FORGED" if is_tampered_heuristic else "SUSPICIOUS"
        flags = [{"check": "scoring_fallback", "status": "WARN", "message": f"Scoring fallback due to: {e}"}]
        summary = f"Fallback score {overall_score}/100 due to scoring error: {e}"
        classifier_component = 50.0
        trufor_component = (1.0 - heatmap_mean) * 100
        ocr_component = 50.0
        doc_val_component = 50.0
        face_component = 50.0

    # Map to frontend ScanResult expectations
    # RiskLevel from scoring.py: GENUINE / SUSPICIOUS / LIKELY_FORGED / FORGED
    # Frontend expects: Low / Medium / High
    risk_map = {"GENUINE": "Low", "SUSPICIOUS": "Medium", "LIKELY_FORGED": "High", "FORGED": "High"}
    riskLevel = risk_map.get(risk_level_raw, "Medium")

    # Decision derived from risk level (frontend uses Verify / Review)
    decision_map = {"GENUINE": "Verify", "SUSPICIOUS": "Review", "LIKELY_FORGED": "Review", "FORGED": "Review"}
    decision = decision_map.get(risk_level_raw, "Review")

    # Tampering flag for frontend — use risk level from scoring, not heuristic
    is_tampered = bool(risk_level_raw in ["LIKELY_FORGED", "FORGED"])

    # OVERRIDE: If not a document, force score down and flag as suspicious
    if not doc_detection.get("is_document", True):
        overall_score = max(0, min(overall_score, 25))
        riskLevel = "High"
        decision = "Review"
        is_tampered = True
        risk_level_raw = "FORGED"
        # Add document detection flag
        flags.append({
            "check": "document_detection",
            "status": "FAIL",
            "message": doc_detection.get("reason", "Image does not appear to be a document"),
        })
        summary = f"NOT A DOCUMENT: {doc_detection.get('reason', 'Unknown')}\\nScore forced to {overall_score}/100"

    print(f"[DEBUG] is_tampered_heuristic={is_tampered_heuristic}, cls_pred={classifier_output.get('prediction')}, risk_raw={risk_level_raw}, is_tampered={is_tampered}, is_doc={doc_detection.get('is_document')}")

    # XAI breakdown for compatibility (also keep detailed)
    # Try to use trufor mean/std etc from features if available
    try:
        trufor_mean_feat = float(features[0]) if len(features) > 0 else heatmap_mean
        trufor_std_feat = float(features[1]) if len(features) > 1 else heatmap_std
        trufor_max_feat = float(features[2]) if len(features) > 2 else heatmap_max
    except Exception:
        trufor_mean_feat = heatmap_mean
        trufor_std_feat = heatmap_std
        trufor_max_feat = heatmap_max

    xai_breakdown = [
        {"feature": "Document Detection", "contribution": round(doc_detection.get("confidence", 0) * 100, 2), "tooltip": doc_detection.get("reason", "Unknown")},
        {"feature": "TruFor Mean", "contribution": round(trufor_mean_feat * 100, 2), "tooltip": f"Heatmap mean {heatmap_mean:.3f} — overall tampering probability"},
        {"feature": "TruFor Std", "contribution": round(trufor_std_feat * 100, 2), "tooltip": "Variance in heatmap — hotspot concentration"},
        {"feature": "TruFor Max", "contribution": round(trufor_max_feat * 100, 2), "tooltip": f"Peak tampering score {heatmap_max:.3f}"},
        {"feature": "MRZ Validation", "contribution": round(100 if mrz_valid else 0, 2), "tooltip": "Machine-readable zone check digits verified" if mrz_valid else "MRZ not found or invalid"},
        {"feature": "OCR Confidence", "contribution": round(ocr_component, 2), "tooltip": f"Text extraction: {len(ocr_text)} chars, flags {len(flags)}"},
        {"feature": "Classifier P(tamper)", "contribution": round(classifier_output.get('probability', 0.5)*100, 2), "tooltip": f"ML classifier tampering probability {classifier_output.get('probability',0.5):.3f}"},
        {"feature": "Face Match", "contribution": 0, "tooltip": "Live face comparison not provided in upload flow"},
    ]

    elapsed = round(time.time() - start_time, 2)

    # Detailed results for evidence panel / debugging
    detailed_results = {
        "overall_score": overall_score,
        "risk_level": risk_level_raw,
        "extracted_fields": ocr_fields,
        "mrz_validation": mrz_valid,
        "mrz_data": mrz_data,
        "mrz_raw": mrz_raw,
        "tampering_detected": is_tampered,
        "heatmap_mean": heatmap_mean,
        "heatmap_max": heatmap_max,
        "heatmap_std": heatmap_std,
        "xai_breakdown": xai_breakdown,
        "document_detection": doc_detection,
        "heuristic_features": {
            "ela_mean": ela_mean,
            "hotspot_count": hotspot_count,
            "hotspot_ratio": hotspot_ratio,
            "double_jpeg": double_jpeg,
            "exif": exif_dict,
        },
        "classifier_output": classifier_output,
        "ocr_text": ocr_text,
        "ocr_flags": getattr(ocr_result, 'consistency_flags', []) if ocr_result else [],
        "flags": flags,
        "summary": summary,
        "components": {
            "classifier": classifier_component,
            "trufor": trufor_component,
            "ocr": ocr_component,
            "structure": score_result.structure_component,
            "image_quality": score_result.image_quality_component,
            "validation": doc_val_component,
            "face": face_component,
        },
        "processing_time_sec": elapsed,
        "source": "sih26188-pipeline"
    }

    # Flattened response for existing frontend ScanResult (backward compatibility)
    # Frontend expects at top level: authenticityScore, isTampered, ocrText, riskLevel, decision, status, source
    # We return BOTH flattened and nested so either frontend version works
    return {
        # Flattened for current Home.tsx fetch
        "authenticityScore": int(round(overall_score)),
        "isTampered": is_tampered,
        "ocrText": ocr_text[:800] if ocr_text else "",
        "riskLevel": riskLevel,
        "decision": decision,
        "status": "complete",
        "source": "sih26188-pipeline",
        # Detailed for evidence panel
        "success": True,
        "results": detailed_results,
        # Also keep top-level detailed for flexibility
        "overall_score": overall_score,
        "extracted_fields": ocr_fields,
        "mrz_validation": mrz_valid,
        "tampering_detected": is_tampered,
        "heatmap_mean": heatmap_mean,
        "xai_breakdown": xai_breakdown,
        "heuristic_features": {
            "ela_mean": ela_mean,
            "hotspot_count": hotspot_count,
            "double_jpeg": double_jpeg,
        },
        # Full details
        "details": detailed_results
    }


@app.route("/api/analyze-document", methods=["POST"])
def analyze_document():
    """Endpoint to analyze a submitted document image.
    
    Accepts multipart/form-data with field 'document' (File).
    No terminal input required — receives image directly from Frontend UI.
    """
    if "document" not in request.files:
        return jsonify({"error": "No document file provided", "success": False}), 400

    file = request.files["document"]

    if file.filename == "":
        return jsonify({"error": "No file selected", "success": False}), 400

    if file and allowed_file(file.filename):
        # Secure filename
        from werkzeug.utils import secure_filename
        safe_name = secure_filename(file.filename)
        filename = f"{os.urandom(8).hex()}_{safe_name}"
        temp_path = UPLOAD_FOLDER / filename
        try:
            file.save(str(temp_path))
        except Exception as e:
            return jsonify({"error": f"Failed to save upload: {e}", "success": False}), 500

        try:
            results = run_full_pipeline(str(temp_path))
            # Clean up temp file
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

            # Return flattened + nested (frontend can use either)
            # Ensure JSON serializable (convert numpy types)
            def to_serializable(obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                return obj

            # Convert any numpy in results
            import json as _json
            serialized = _json.loads(_json.dumps(results, default=to_serializable))

            return jsonify(serialized)

        except Exception as e:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
            import traceback
            traceback.print_exc()
            return jsonify({
                "error": str(e),
                "success": False,
                "status": "error"
            }), 500

    return jsonify({"error": "File type not allowed. Use png, jpg, jpeg, bmp, tiff, webp", "success": False}), 400


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "SIH26188 Document Authenticity API", "version": "1.0"})


@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "SIH26188 Document Authenticity API", "status": "running", "endpoints": ["/api/analyze-document", "/api/health"]})


if __name__ == "__main__":
    print("="*60)
    print(" SIH26188 Document Authenticity API")
    print(" Training tamper classifier...")
    try:
        result = train_classifier()
        print(f" Classifier: {result}")
    except Exception as e:
        print(f" Classifier training failed: {e}")
    print(" Flask backend ready — waiting for Frontend uploads")
    print(" POST /api/analyze-document  (multipart/form-data, field: document)")
    print(" No terminal image input required")
    print("="*60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

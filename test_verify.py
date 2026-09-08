import requests, os, json

print("=== PIPELINE VERIFICATION ===\n")

for folder in ["data/genuine", "data/tampered"]:
    print("--- %s ---" % folder)
    for f in sorted(os.listdir(folder)):
        if f.endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder, f)
            with open(path, "rb") as img:
                r = requests.post(
                    "http://localhost:5000/api/analyze-document",
                    files={"document": (f, img, "image/jpeg")},
                    timeout=120,
                )
            d = r.json()
            score = d.get("authenticityScore", "?")
            is_tam = d.get("isTampered", "?")
            risk = d.get("riskLevel", "?")
            is_doc = d.get("details", {}).get("document_detection", {}).get("is_document", "?")
            tamper = d.get("details", {}).get("classifier_output", {})
            prob = tamper.get("probability", "?")
            print("  %-35s score=%3s  tampered=%-5s  risk=%-6s  is_doc=%-5s  tamper=%.3f" % (
                f, score, is_tam, risk, is_doc, prob if isinstance(prob, float) else 0
            ))
    print()

# Test face photo
import cv2, numpy as np
face = np.zeros((400, 400, 3), dtype=np.uint8)
face[50:350, 50:350] = [200, 180, 160]
cv2.circle(face, (200, 150), 80, (180, 160, 140), -1)
cv2.circle(face, (170, 130), 10, (50, 50, 50), -1)
cv2.circle(face, (230, 130), 10, (50, 50, 50), -1)
cv2.ellipse(face, (200, 200), (30, 15), 0, 0, 180, (100, 80, 80), 2)
cv2.imwrite("test_face.jpg", face)

print("--- FACE PHOTO ---")
with open("test_face.jpg", "rb") as f:
    r = requests.post("http://localhost:5000/api/analyze-document",
                      files={"document": ("face.jpg", f, "image/jpeg")}, timeout=120)
d = r.json()
print("  score=%s  isTampered=%s  risk=%s  is_doc=%s" % (
    d.get("authenticityScore"), d.get("isTampered"), d.get("riskLevel"),
    d.get("details", {}).get("document_detection", {}).get("is_document")
))
os.remove("test_face.jpg")

# Summary
print("\n=== SUMMARY ===")
genuine_scores = []
tampered_scores = []
for f in os.listdir("data/genuine"):
    if f.endswith((".jpg", ".jpeg", ".png")):
        with open(os.path.join("data/genuine", f), "rb") as img:
            r = requests.post("http://localhost:5000/api/analyze-document",
                files={"document": (f, img, "image/jpeg")}, timeout=120)
        genuine_scores.append(r.json().get("authenticityScore", 0))
for f in os.listdir("data/tampered"):
    if f.endswith((".jpg", ".jpeg", ".png")):
        with open(os.path.join("data/tampered", f), "rb") as img:
            r = requests.post("http://localhost:5000/api/analyze-document",
                files={"document": (f, img, "image/jpeg")}, timeout=120)
        tampered_scores.append(r.json().get("authenticityScore", 0))

print("Genuine: avg=%.1f  min=%d  max=%d" % (sum(genuine_scores)/len(genuine_scores), min(genuine_scores), max(genuine_scores)))
print("Tampered: avg=%.1f  min=%d  max=%d" % (sum(tampered_scores)/len(tampered_scores), min(tampered_scores), max(tampered_scores)))
print("Separation: %.1f points" % (sum(genuine_scores)/len(genuine_scores) - sum(tampered_scores)/len(tampered_scores)))

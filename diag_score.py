"""Dump per-component score breakdown to see what drives the verdict."""
import json
import urllib.request
import mimetypes
import uuid
import sys
from pathlib import Path

API = "http://localhost:5000/api/analyze-document"


def post(path: str) -> dict:
    data = Path(path).read_bytes()
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="document"; filename="{Path(path).name}"\r\n'
        f"Content-Type: {mimetypes.guess_type(path)[0] or 'application/octet-stream'}\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        API, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode())


for path in sys.argv[1:]:
    d = post(path)
    res = d.get("results") or d.get("details") or {}
    comp = res.get("components", {})
    hf = res.get("heuristic_features", {}) or {}
    cls = res.get("classifier_output", {}) or {}
    print(f"\n=== {path}")
    print(f"  score={d.get('authenticityScore')} risk={d.get('riskLevel')} "
          f"decision={d.get('decision')} tampered={d.get('isTampered')}")
    print(f"  components: {json.dumps(comp)}")
    dv = res.get("document_validation")
    print(f"  doc_validation: {json.dumps(dv)[:600]}")
    print(f"  classifier: {json.dumps(cls)}")
    print(f"  heatmap_mean={res.get('heatmap_mean')} tampering_detected={res.get('tampering_detected')}")
    print(f"  ocr_flags={json.dumps(res.get('ocr_flags'))[:300]}")
    print(f"  doc_detect={json.dumps(res.get('document_detection'))[:200]}")
    print(f"  flags={json.dumps(res.get('flags'))[:400]}")
    print(f"  ocr_text={str(res.get('ocr_text'))[:200]!r}")

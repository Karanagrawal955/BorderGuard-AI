"""Accuracy test: does the system call genuine docs GENUINE and forged docs FORGED?"""
import json
import sys
import urllib.request
import mimetypes
import uuid
from pathlib import Path

API = "http://localhost:5000/api/analyze-document"

CASES = [
    ("data/genuine/genuine_00.jpg", "GENUINE"),
    ("data/genuine/genuine_01.jpg", "GENUINE"),
    ("data/genuine/genuine_03.jpg", "GENUINE"),
    ("data/genuine/genuine_05.jpg", "GENUINE"),
    ("data/genuine/genuine_08.jpg", "GENUINE"),
    ("data/genuine/synthetic_genuine_01.png", "GENUINE"),
    ("data/genuine/synthetic_genuine_03.png", "GENUINE"),
    ("data/genuine/sample1.jpg", "GENUINE"),
    ("data/genuine/sample4.jpg", "GENUINE"),
    ("data/genuine/sample7.jpg", "GENUINE"),
    ("data/tampered/tampered_00.jpg", "FORGED"),
    ("data/tampered/tampered_01.jpg", "FORGED"),
    ("data/tampered/tampered_03.jpg", "FORGED"),
    ("data/tampered/tampered_05.jpg", "FORGED"),
    ("data/tampered/tampered_08.jpg", "FORGED"),
    ("data/tampered/synthetic_tampered_01.png", "FORGED"),
    ("data/tampered/synthetic_tampered_03.png", "FORGED"),
    ("data/tampered/tampered1.jpg", "FORGED"),
    ("data/tampered/myimage.jpeg", "FORGED"),
    ("data/tampered/11front_scaled_up.jpg", "FORGED"),
]


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


def main() -> None:
    correct = 0
    total = 0
    rows = []
    for path, expected in CASES:
        if not Path(path).exists():
            rows.append((path, expected, "MISSING", "-", "-", "-"))
            continue
        try:
            d = post(path)
        except Exception as e:
            rows.append((path, expected, "ERROR", "-", "-", str(e)[:60]))
            continue
        got = d.get("riskLevel")          # Low / Medium / High
        score = d.get("authenticityScore")
        decision = d.get("decision")
        # Expected mapping: GENUINE -> Low, FORGED -> High
        want = "Low" if expected == "GENUINE" else "High"
        ok = got == want
        correct += int(ok)
        total += 1
        rows.append((path, expected, got, score, decision, "OK" if ok else "WRONG"))

    print(f"\n{'file':45s} {'expect':8s} {'got':7s} {'score':>5s} {'decision':8s} verdict")
    print("-" * 95)
    for r in rows:
        print(f"{r[0]:45s} {r[1]:8s} {str(r[2]):7s} {str(r[3]):>5s} {str(r[4]):8s} {r[5]}")
    print("-" * 95)
    print(f"ACCURACY: {correct}/{total} = {100*correct/total if total else 0:.0f}%\n")


if __name__ == "__main__":
    main()

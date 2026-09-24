"""Non-document test: random photos must be reported as INVALID DOCUMENT,
not as tampered/genuine documents."""
import json
import mimetypes
import sys
import urllib.request
import uuid
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


def main() -> None:
    paths = sys.argv[1:] or sorted(str(p) for p in Path("test_nondocs").glob("*.jpg"))
    hdr = f"{'file':24s} {'isDoc':>5s} {'conf':>5s} {'score':>5s} {'risk':>7s} {'tamper':>6s} {'decision':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for path in paths:
        try:
            d = post(path)
        except Exception as e:
            print(f"{Path(path).name:24s} ERROR {str(e)[:60]}")
            continue
        res = d.get("results") or {}
        det = res.get("document_detection") or {}
        print(f"{Path(path).name:24s} {str(d.get('isDocument')):>5s} "
              f"{det.get('confidence', 0):>5.2f} {d.get('authenticityScore'):>5} "
              f"{d.get('riskLevel'):>7s} {str(d.get('isTampered')):>6s} {d.get('decision'):>8s}")
        if not d.get("isDocument", True):
            print(f"{'':24s}   -> {det.get('reason')}")


if __name__ == "__main__":
    main()

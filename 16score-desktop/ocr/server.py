"""Long-running PaddleOCR worker for venv310 — subprocess from killfeed.detector."""
import base64
import json
import os
import struct
import sys

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ocr.worker import extract_killfeed, init_ocr_worker


def _send(obj):
    payload = json.dumps(obj).encode("utf-8")
    sys.stdout.buffer.write(struct.pack(">I", len(payload)))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def _recv():
    header = sys.stdin.buffer.read(4)
    if len(header) < 4:
        return None
    length = struct.unpack(">I", header)[0]
    if length <= 0 or length > 32 * 1024 * 1024:
        return None
    body = sys.stdin.buffer.read(length)
    if len(body) < length:
        return None
    return json.loads(body.decode("utf-8"))


def main():
    init_ocr_worker()
    _send({"ready": True, "api": "ocr()"})

    while True:
        request = _recv()
        if request is None:
            break

        cmd = request.get("cmd", "extract")
        if cmd == "shutdown":
            break
        if cmd == "ping":
            _send({"ok": True})
            continue

        try:
            shape = tuple(int(x) for x in request["shape"])
            raw = base64.b64decode(request["data"])
            result = extract_killfeed(raw, shape)
            _send(result)
        except Exception as exc:
            _send({"killer": "", "victim": "", "confidence": 0.0, "error": type(exc).__name__})


if __name__ == "__main__":
    main()

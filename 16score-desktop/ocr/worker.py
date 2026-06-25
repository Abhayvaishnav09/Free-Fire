"""Isolated PaddleOCR worker — used by ocr_server.py (venv310 subprocess)."""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ.setdefault("PADDLE_PDX_DISABLE_MKLDNN", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

_detector = None


def init_ocr_worker():
    """Load PaddleOCR once inside the worker process."""
    global _detector
    if _detector is not None:
        return
    from ocr.text import FreeFireTextDetector

    _detector = FreeFireTextDetector()


def extract_killfeed(crop_bytes, shape):
    """Run OCR on a BGR crop; returns killer/victim/confidence dict."""
    import numpy as np

    empty = {"killer": "", "victim": "", "confidence": 0.0}
    try:
        global _detector
        if _detector is None:
            init_ocr_worker()
        crop = np.frombuffer(crop_bytes, dtype=np.uint8).reshape(shape)
        result = _detector.extract_killfeed_split(crop)
        if not result.get("killer") or not result.get("victim"):
            result = _detector.extract_killfeed_sequence(crop)
        return {
            "killer": result.get("killer", "") or "",
            "victim": result.get("victim", "") or "",
            "confidence": float(result.get("confidence", 0.0)),
        }
    except Exception:
        return empty

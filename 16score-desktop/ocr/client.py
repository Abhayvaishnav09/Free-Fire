"""Subprocess client for PaddleOCR 2.10 worker (venv310 / Python 3.10)."""
import base64
import json
import os
import struct
import subprocess
import sys
from threading import Lock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_ocr_python():
    """Prefer venv310 (PaddleOCR 2.10) over the main app Python."""
    for rel in ("venv310", ".venv"):
        candidate = os.path.join(_PROJECT_ROOT, rel, "Scripts", "python.exe")
        if os.path.isfile(candidate):
            return candidate
    return sys.executable


class OcrSubprocessClient:
    """Persistent OCR subprocess — avoids loading Paddle in the main YOLO process."""

    def __init__(self):
        self._lock = Lock()
        self._proc = None
        self.python = resolve_ocr_python()
        self.server_script = os.path.join(_PROJECT_ROOT, "ocr", "server.py")

    @property
    def uses_isolated_venv(self):
        return os.path.normcase(self.python) != os.path.normcase(sys.executable)

    def _ocr_env(self):
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        env.setdefault("OMP_NUM_THREADS", "1")
        env["FLAGS_use_mkldnn"] = "0"
        env.setdefault("PADDLE_PDX_DISABLE_MKLDNN", "1")
        return env

    def _read_message(self):
        header = self._proc.stdout.read(4)
        if len(header) < 4:
            raise RuntimeError("OCR server closed stdout")
        length = struct.unpack(">I", header)[0]
        if length <= 0 or length > 32 * 1024 * 1024:
            raise RuntimeError("OCR server sent invalid frame size")
        body = self._proc.stdout.read(length)
        if len(body) < length:
            raise RuntimeError("OCR server response truncated")
        return json.loads(body.decode("utf-8"))

    def _write_message(self, payload):
        data = json.dumps(payload).encode("utf-8")
        self._proc.stdin.write(struct.pack(">I", len(data)))
        self._proc.stdin.write(data)
        self._proc.stdin.flush()

    def _stop(self):
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin:
                data = json.dumps({"cmd": "shutdown"}).encode("utf-8")
                proc.stdin.write(struct.pack(">I", len(data)))
                proc.stdin.write(data)
                proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()

    def _start(self):
        self._stop()
        if not os.path.isfile(self.server_script):
            raise FileNotFoundError(f"Missing OCR server script: {self.server_script}")

        self._proc = subprocess.Popen(
            [self.python, self.server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=_PROJECT_ROOT,
            env=self._ocr_env(),
        )
        ready = self._read_message()
        if not ready.get("ready"):
            raise RuntimeError("OCR server failed to initialize")

    def restart(self):
        with self._lock:
            self._stop()

    def shutdown(self):
        with self._lock:
            self._stop()

    def extract_killfeed(self, cropped_image, timeout=120):
        if cropped_image is None or cropped_image.size == 0:
            return None

        with self._lock:
            try:
                if self._proc is None or self._proc.poll() is not None:
                    self._start()

                self._write_message(
                    {
                        "cmd": "extract",
                        "shape": list(cropped_image.shape),
                        "data": base64.b64encode(cropped_image.tobytes()).decode("ascii"),
                    }
                )
                return self._read_message()
            except Exception:
                self._stop()
                return None

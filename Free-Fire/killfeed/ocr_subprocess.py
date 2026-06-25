"""Run PaddleOCR in an isolated subprocess so segfaults don't kill detection."""

from __future__ import annotations

import os
import threading
import time
from multiprocessing import Process, Queue, get_context
from typing import Optional, Tuple

import cv2
import numpy as np


def _ocr_worker(request_q: Queue, response_q: Queue) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")

    from text import FreeFireTextDetector

    detector = FreeFireTextDetector()
    while True:
        item = request_q.get()
        if item is None:
            break
        req_id, png_bytes = item
        try:
            buf = np.frombuffer(png_bytes, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                response_q.put((req_id, "", "", 0.0))
                continue
            result = detector.extract_killfeed_sequence_by_halves(img)
            killer = (result.get("killer") or "").strip()
            victim = (result.get("victim") or "").strip()
            confidence = float(result.get("confidence") or 0.0)
            response_q.put((req_id, killer, victim, confidence))
        except Exception:
            response_q.put((req_id, "", "", 0.0))


class SubprocessOCRProvider:
    """OCR via dedicated child process; auto-restarts after crash."""

    def __init__(self, timeout: float = 12.0):
        self.timeout = timeout
        self._req_q: Optional[Queue] = None
        self._resp_q: Optional[Queue] = None
        self._proc: Optional[Process] = None
        self._lock = threading.Lock()
        self._next_id = 0
        self._start_worker()

    def _start_worker(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        if self._proc is not None:
            try:
                self._proc.join(timeout=0.5)
            except Exception:
                pass
        ctx = get_context("spawn")
        self._req_q = ctx.Queue(maxsize=4)
        self._resp_q = ctx.Queue(maxsize=4)
        self._proc = ctx.Process(
            target=_ocr_worker,
            args=(self._req_q, self._resp_q),
            daemon=True,
            name="PaddleOCRWorker",
        )
        self._proc.start()
        print("🔒 OCR running in isolated subprocess (segfault-safe)")

    def _ensure_alive(self) -> None:
        if self._proc is None or not self._proc.is_alive():
            print("⚠️ OCR subprocess died — restarting…")
            self._start_worker()

    def extract_row_names(self, row_crop) -> Tuple[str, str, float]:
        if row_crop is None or getattr(row_crop, "size", 0) == 0:
            return "", "", 0.0

        with self._lock:
            self._ensure_alive()
            ok, buf = cv2.imencode(".png", row_crop)
            if not ok:
                return "", "", 0.0

            req_id = self._next_id
            self._next_id += 1
            assert self._req_q is not None and self._resp_q is not None
            self._req_q.put((req_id, buf.tobytes()))

            deadline = time.time() + self.timeout
            while time.time() < deadline:
                if self._proc is not None and not self._proc.is_alive():
                    return "", "", 0.0
                try:
                    rid, killer, victim, conf = self._resp_q.get(timeout=0.25)
                    if rid == req_id:
                        return killer, victim, float(conf)
                except Exception:
                    continue
            print("⚠️ OCR subprocess timeout")
            return "", "", 0.0

    def shutdown(self) -> None:
        with self._lock:
            if self._req_q is not None:
                try:
                    self._req_q.put(None)
                except Exception:
                    pass
            if self._proc is not None:
                self._proc.join(timeout=2)
                if self._proc.is_alive():
                    self._proc.terminate()
            self._proc = None


class SubprocessOCRPool:
    """Pool of isolated OCR subprocess workers (used by FIFO pipeline)."""

    def __init__(self, min_workers: int = 1, max_workers: int = 2, timeout: float = 12.0):
        from killfeed.ocr_pool import AdaptiveOCRPool

        self._pool = AdaptiveOCRPool(
            min_workers=min_workers,
            max_workers=max_workers,
            timeout=timeout,
        )

    def warmup(self) -> None:
        """Load PaddleOCR in the first worker (one-time startup cost)."""
        dummy = np.zeros((24, 120, 3), dtype=np.uint8)
        try:
            self._pool.extract_row_names(dummy)
        except Exception:
            pass

    def extract_row_names(self, row_crop):
        return self._pool.extract_row_names(row_crop)

    def shutdown(self) -> None:
        self._pool.shutdown()

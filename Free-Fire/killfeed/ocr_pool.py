"""Adaptive pool of isolated OCR subprocess workers.

Each worker is a :class:`SubprocessOCRProvider` (segfault-safe PaddleOCR child).
The pipeline calls :meth:`update_load` each OCR batch to scale parallel OCR
threads; this class grows the warm worker pool up to ``max_workers`` under load
and exposes the same ``extract_row_names*`` API as a single provider.
"""

from __future__ import annotations

import threading
from typing import Callable, List, Optional, Tuple

from killfeed.ocr_subprocess import SubprocessOCRProvider


class AdaptiveOCRPool:
    """Round-robin OCR dispatch across N subprocess workers with load scaling."""

    def __init__(
        self,
        finalize_fn: Optional[Callable[[str, str, float], Tuple[str, str, float]]] = None,
        min_workers: int = 1,
        max_workers: int = 4,
        timeout: float = 12.0,
    ):
        self.finalize_fn = finalize_fn
        self.min_workers = max(1, int(min_workers))
        self.max_workers = max(self.min_workers, int(max_workers))
        self.timeout = timeout
        self._lock = threading.Lock()
        self._workers: List[SubprocessOCRProvider] = []
        self._active_workers = self.min_workers
        self._rr_idx = 0
        self._ensure_workers(self.min_workers)
        print(
            f"🔒 Adaptive OCR pool: {self.min_workers}–{self.max_workers} "
            f"subprocess workers (grow on load, no startup pre-warm)"
        )

    @property
    def active_workers(self) -> int:
        return self._active_workers

    def _ensure_workers(self, count: int) -> None:
        target = max(self.min_workers, min(count, self.max_workers))
        while len(self._workers) < target:
            wid = len(self._workers)
            quiet = wid > 0
            self._workers.append(SubprocessOCRProvider(timeout=self.timeout))

    def update_load(
        self,
        batch_size: int,
        recent_arrivals: int,
        queue_depth: int = 0,
    ) -> int:
        """Return recommended parallel OCR threads for this batch."""
        if queue_depth >= 1 or recent_arrivals >= 1 or batch_size >= 1:
            target = self.max_workers
        else:
            target = self.min_workers

        target = max(self.min_workers, min(target, self.max_workers))
        with self._lock:
            self._ensure_workers(target)
            self._active_workers = target
        return min(target, max(1, batch_size))

    def _pick_worker(self) -> SubprocessOCRProvider:
        with self._lock:
            n = max(1, min(self._active_workers, len(self._workers)))
            worker = self._workers[self._rr_idx % n]
            self._rr_idx += 1
            return worker

    def _apply_finalize(
        self, killer: str, victim: str, confidence: float
    ) -> Tuple[str, str, float]:
        if self.finalize_fn is not None:
            return self.finalize_fn(killer, victim, confidence)
        return killer, victim, confidence

    def extract_row_names_raw(self, row_crop) -> Tuple[str, str, float]:
        killer, victim, confidence = self._pick_worker().extract_row_names(row_crop)
        return self._apply_finalize(killer, victim, confidence)

    def extract_row_names(self, row_crop) -> Tuple[str, str, float]:
        return self.extract_row_names_raw(row_crop)

    def shutdown(self) -> None:
        with self._lock:
            workers = list(self._workers)
            self._workers.clear()
        for worker in workers:
            try:
                worker.shutdown()
            except Exception:
                pass

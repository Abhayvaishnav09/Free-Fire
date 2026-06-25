"""Visual-slot queue: pending killfeed rows awaiting OCR and serial commit.

Each new killfeed bar (identified by crop dHash) is registered when the visual
FIFO diff detects it.  Rows are grouped by capture ``frame_num`` so the commit
thread can emit events in strict chronological order while OCR runs in parallel
across frames and slots.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set

import numpy as np

from killfeed.dhash import dhash_distance

# Must match commit-thread abandon threshold — OCR dispatch stops at this count.
MAX_OCR_ATTEMPTS = 5


@dataclass
class PendingRow:
    """One killfeed row waiting for OCR → classify → emit."""

    visual_sig: str
    slot: int  # 0 = top / newest on screen
    frame_num: int
    timestamp: float
    crop: np.ndarray
    yolo_conf: float = 0.0
    yolo_class: str = "killblock"
    ocr_killer: str = ""
    ocr_victim: str = ""
    ocr_confidence: float = 0.0
    ocr_done: bool = False
    ocr_attempts: int = 0
    ocr_inflight: bool = False
    ocr_inflight_since: float = 0.0
    emitted: bool = False
    parse_failed: bool = False
    registered_at: float = field(default_factory=time.time)
    arrival_seq: int = 0  # monotonic stream order — authoritative for TMS emit


@dataclass
class _FrameRecord:
    frame_num: int
    timestamp: float
    visual_sigs: List[str]


class SlotTracker:
    """Tracks pending rows by visual dHash and groups them by capture frame."""

    def __init__(self, max_slots: int = 4, ring_size: int = 90):
        self.max_slots = max_slots
        self.ring_size = max(16, ring_size)
        self._pending: Dict[str, PendingRow] = {}
        self._frame_rows: Dict[int, List[str]] = {}
        self._frame_order: Deque[int] = deque()
        self._ring: Deque[_FrameRecord] = deque(maxlen=self.ring_size)
        self._next_arrival_seq = 1

    def next_arrival_seq(self) -> int:
        seq = self._next_arrival_seq
        self._next_arrival_seq += 1
        return seq

    def known_visual_sigs(self) -> Set[str]:
        return set(self._pending.keys())

    def _find_visual_match(
        self, sig: str, candidates: Set[str], max_dhash_distance: int
    ) -> Optional[str]:
        if not sig:
            return None
        best_key: Optional[str] = None
        best_dist = max_dhash_distance + 1
        for key in candidates:
            if not key:
                continue
            dist = dhash_distance(sig, key)
            if dist <= max_dhash_distance and dist < best_dist:
                best_dist = dist
                best_key = key
        return best_key

    def is_visually_known(
        self,
        sig: str,
        emitted_visual_dhashes: Set[str],
        max_dhash_distance: int,
    ) -> bool:
        if not sig:
            return True
        if sig in self._pending and not self._pending[sig].emitted:
            return True
        if self._find_visual_match(sig, emitted_visual_dhashes, max_dhash_distance):
            return True
        pending_keys = {
            k for k, p in self._pending.items() if not p.emitted and not p.parse_failed
        }
        return self._find_visual_match(sig, pending_keys, max_dhash_distance) is not None

    def _resolve_pending_sig(
        self, sig: str, emitted: Set[str], max_dhash_distance: int
    ) -> Optional[str]:
        if sig in self._pending and not self._pending[sig].emitted:
            return sig
        pending_keys = {
            k for k, p in self._pending.items() if not p.emitted and not p.parse_failed
        }
        match = self._find_visual_match(sig, pending_keys, max_dhash_distance)
        if match:
            return match
        return self._find_visual_match(sig, emitted, max_dhash_distance)

    # ── capture thread ───────────────────────────────────────────────────

    def record_frame(
        self,
        frame_num: int,
        timestamp: float,
        visual_sigs: List[str],
        rows,
    ) -> None:
        """Store frame metadata and refresh crops for rows still pending."""
        sigs = list(visual_sigs[: self.max_slots])
        self._ring.append(_FrameRecord(frame_num, timestamp, sigs))

        for idx, sig in enumerate(sigs):
            pending = self._pending.get(sig)
            if pending is None:
                match = self._find_visual_match(sig, set(self._pending.keys()), 6)
                if match:
                    pending = self._pending.get(match)
            if pending is None or pending.emitted:
                continue
            if idx < len(rows):
                crop = getattr(rows[idx], "crop", None)
                if crop is not None and getattr(crop, "size", 0) > 0:
                    pending.crop = crop.copy()

    def register_new_rows(
        self,
        new_indices: List[int],
        visual_sigs: List[str],
        rows,
        frame_num: int,
        timestamp: float,
        emitted_visual_dhashes: Optional[Set[str]] = None,
        max_dhash_distance: int = 0,
    ) -> int:
        """Enqueue newly detected rows; return count registered."""
        if not new_indices:
            return 0

        emitted = emitted_visual_dhashes or set()

        if frame_num not in self._frame_rows:
            self._frame_rows[frame_num] = []
            self._frame_order.append(frame_num)

        registered = 0
        # Register lower slots first so discovery order matches on-screen chronology.
        for idx in sorted(new_indices, reverse=True):
            if idx < 0 or idx >= len(visual_sigs) or idx >= len(rows):
                continue
            sig = visual_sigs[idx]
            if not sig:
                continue
            if sig in emitted:
                continue

            existing = self._pending.get(sig)
            if existing is not None and not existing.emitted:
                row = rows[idx]
                crop = getattr(row, "crop", None)
                if crop is not None and getattr(crop, "size", 0) > 0:
                    existing.crop = crop.copy()
                continue

            row = rows[idx]
            crop = getattr(row, "crop", None)
            if crop is None or getattr(crop, "size", 0) == 0:
                continue

            pending = PendingRow(
                visual_sig=sig,
                slot=idx,
                frame_num=frame_num,
                timestamp=timestamp,
                crop=crop.copy(),
                yolo_conf=float(getattr(row, "confidence", 0.0)),
                yolo_class=str(getattr(row, "class_name", "killblock") or "killblock"),
                arrival_seq=self.next_arrival_seq(),
            )
            self._pending[sig] = pending
            if sig not in self._frame_rows[frame_num]:
                self._frame_rows[frame_num].append(sig)
            registered += 1

        self._prune_completed_frames()
        return registered

    # ── OCR dispatch thread ────────────────────────────────────────────

    def pending_count(self) -> int:
        return sum(1 for p in self._pending.values() if not p.emitted)

    def inflight_count(self) -> int:
        return sum(1 for p in self._pending.values() if p.ocr_inflight)

    def _oldest_incomplete_frame(self, emitted_visual_dhashes: Set[str]) -> Optional[int]:
        """Frame number blocking serial commit (same order as oldest_frame_pending)."""
        for frame_num in sorted(self._frame_order):
            sigs = self._frame_rows.get(frame_num, [])
            rows = [self._pending[s] for s in sigs if s in self._pending]
            if not rows:
                continue
            incomplete = any(
                not p.emitted
                and p.visual_sig not in emitted_visual_dhashes
                for p in rows
            )
            if incomplete:
                return frame_num
        return None

    def pending_work_items(
        self,
        emitted_visual_dhashes: Set[str],
        *,
        fast_row0: bool = False,
    ) -> List[PendingRow]:
        """Rows needing OCR.

        Scheduling prioritises the oldest pending frame (commit order) and
        highest slot index within that frame first — commit emits bottom-up, so
        slot 3 must be OCR'd before slot 0 can emit.  ``fast_row0`` then fills
        spare capacity with newest slot-0 rows from later frames.
        """
        items: List[PendingRow] = []
        for pending in self._pending.values():
            if pending.emitted or pending.visual_sig in emitted_visual_dhashes:
                continue
            if pending.ocr_done or pending.ocr_inflight:
                continue
            if pending.ocr_attempts >= MAX_OCR_ATTEMPTS:
                continue
            items.append(pending)

        items.sort(key=lambda p: (p.arrival_seq, -p.slot))
        return items

    def mark_ocr_inflight(self, visual_sig: str) -> bool:
        pending = self._pending.get(visual_sig)
        if pending is None or pending.emitted or pending.ocr_done or pending.ocr_inflight:
            return False
        pending.ocr_inflight = True
        pending.ocr_inflight_since = time.time()
        return True

    def mark_ocr_failed(self, visual_sig: str) -> None:
        pending = self._pending.get(visual_sig)
        if pending is None:
            return
        pending.ocr_inflight = False

    def mark_ocr_result(
        self,
        visual_sig: str,
        killer: str,
        victim: str,
        confidence: float,
    ) -> None:
        pending = self._pending.get(visual_sig)
        if pending is None:
            return
        pending.ocr_killer = killer or ""
        pending.ocr_victim = victim or ""
        pending.ocr_confidence = float(confidence)
        pending.ocr_done = True
        pending.ocr_inflight = False

    def release_stale_inflight(self, max_age: float = 12.0) -> int:
        """Clear OCR locks stuck longer than ``max_age`` seconds."""
        now = time.time()
        released = 0
        for pending in self._pending.values():
            if not pending.ocr_inflight:
                continue
            if now - pending.ocr_inflight_since >= max_age:
                pending.ocr_inflight = False
                released += 1
        return released

    # ── commit thread ────────────────────────────────────────────────────

    def has_arrival_pending(self, emitted_visual_dhashes: Set[str]) -> bool:
        for p in self._pending.values():
            if p.emitted or p.parse_failed:
                continue
            if p.visual_sig in emitted_visual_dhashes:
                continue
            return True
        return False

    def arrival_pending_rows(
        self, emitted_visual_dhashes: Set[str]
    ) -> List[PendingRow]:
        """All incomplete rows sorted by arrival_seq (stream order)."""
        rows = [
            p
            for p in self._pending.values()
            if not p.emitted and p.visual_sig not in emitted_visual_dhashes
        ]
        rows.sort(key=lambda p: (p.arrival_seq, p.slot))
        return rows

    def oldest_frame_pending(
        self, emitted_visual_dhashes: Set[str]
    ) -> Optional[List[PendingRow]]:
        """All rows from the oldest frame that still has unfinished work."""
        for frame_num in sorted(self._frame_order):
            sigs = self._frame_rows.get(frame_num, [])
            rows = [self._pending[s] for s in sigs if s in self._pending]
            if not rows:
                continue
            incomplete = any(
                not p.emitted and p.visual_sig not in emitted_visual_dhashes
                for p in rows
            )
            if incomplete:
                return rows
        return None

    def can_emit_slot(
        self,
        pending: PendingRow,
        frame_rows: List[PendingRow],
        emitted_visual_dhashes: Set[str],
    ) -> bool:
        """Allow emit only when OCR is done and older slots in the frame are settled."""
        if pending.emitted or pending.parse_failed:
            return False
        if pending.visual_sig in emitted_visual_dhashes:
            return False
        if not pending.ocr_done:
            return False

        # Higher slot index = lower on screen = older within the same frame burst.
        # Emit bottom-up: slot 3 before 2 before 1 before 0.
        for other in frame_rows:
            if other.slot <= pending.slot:
                continue
            if other.emitted or other.parse_failed:
                continue
            if other.visual_sig in emitted_visual_dhashes:
                continue
            return False
        return True

    def mark_parse_failed(self, visual_sig: str) -> None:
        pending = self._pending.get(visual_sig)
        if pending is None:
            return
        pending.parse_failed = True
        pending.ocr_inflight = False

    def mark_emitted(self, visual_sig: str) -> None:
        pending = self._pending.get(visual_sig)
        if pending is None:
            return
        pending.emitted = True
        pending.ocr_inflight = False
        self._prune_completed_frames()

    # ── housekeeping ─────────────────────────────────────────────────────

    def _prune_completed_frames(self) -> None:
        """Drop fully emitted frames and cap pending map size."""
        done_frames: List[int] = []
        for frame_num in list(self._frame_order):
            sigs = self._frame_rows.get(frame_num, [])
            rows = [self._pending.get(s) for s in sigs]
            rows = [r for r in rows if r is not None]
            if rows and all(r.emitted or r.parse_failed for r in rows):
                done_frames.append(frame_num)

        for frame_num in done_frames:
            sigs = self._frame_rows.pop(frame_num, [])
            for sig in sigs:
                pending = self._pending.get(sig)
                if pending and (pending.emitted or pending.parse_failed):
                    self._pending.pop(sig, None)
            try:
                self._frame_order.remove(frame_num)
            except ValueError:
                pass

        if len(self._pending) <= self.ring_size * self.max_slots:
            return

        # Hard cap: drop oldest emitted entries first.
        for frame_num in sorted(self._frame_order):
            for sig in list(self._frame_rows.get(frame_num, [])):
                pending = self._pending.get(sig)
                if pending and pending.emitted:
                    self._pending.pop(sig, None)
            if len(self._pending) <= self.ring_size * self.max_slots:
                break

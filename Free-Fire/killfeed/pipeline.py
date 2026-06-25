"""Two-thread killfeed pipeline: capture + visual gate | OCR worker + FIFO diff."""

from __future__ import annotations

import gc
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Callable, List, Optional, Tuple

import numpy as np

from killfeed.capture import CaptureThread, CameraSource
from killfeed.dhash import compute_dhash, dhash_distance, pixel_mean_diff, strip_changed
from killfeed.events import EventWriter
from killfeed.parser import ParsedRow, signatures_from_rows
from killfeed.queue_state import KillfeedState, find_new_slots, parse_signature, signatures_match
from killfeed.roi import (
    RowDetection,
    StripStabilizer,
    consolidate_overlapping_rows,
    detect_strip,
    reconstruct_strip_from_rows,
)


@dataclass
class StripJob:
    strip: np.ndarray
    roi: object
    frame_num: int
    timestamp: float
    new_row_indices: Tuple[int, ...] = ()
    priority: int = 0  # higher = process sooner (row-0 / new-row strips)


class KillfeedPipeline:
    """Thread 1: capture → YOLO → gate | Thread 2: OCR → FIFO diff → emit once."""

    def __init__(
        self,
        model,
        ocr_provider,
        on_event: Callable[[ParsedRow, int, int], None],
        name_validator: Optional[Callable[[str, str, float], bool]] = None,
        camera_source: CameraSource = 2,
        camera_width: int = 1920,
        camera_height: int = 1080,
        capture_fps: int = 30,
        frame_buffer_seconds: float = 3.0,
        kill_confidence: float = 0.20,
        revive_confidence: float = 0.28,
        ocr_queue_max: int = 8,
        max_visible_slots: int = 4,
        cache_ttl_seconds: float = 30.0,
        pair_cooldown_seconds: float = 10.0,
        fuzzy_threshold: float = 90.0,
        roi_change_threshold: float = 3.0,
        dhash_max_distance: int = 4,
        yolo_lock=None,
        log_path: str = "killfeed.log",
        jsonl_path: str = "killfeed_events.jsonl",
        open_capture=None,
        fast_row0_emit: bool = False,
        yolo_heartbeat_frames: int = 0,
        capture=None,
        **kwargs,
    ):
        self.model = model
        self.ocr_provider = ocr_provider
        self.on_event = on_event
        self.name_validator = name_validator
        self.kill_confidence = kill_confidence
        self.revive_confidence = revive_confidence
        self.ocr_queue_max = ocr_queue_max
        self.roi_change_threshold = roi_change_threshold
        self.dhash_max_distance = dhash_max_distance
        self.yolo_lock = yolo_lock
        self.fast_row0_emit = bool(fast_row0_emit)
        self.yolo_heartbeat_frames = max(0, int(yolo_heartbeat_frames))
        self.yolo_sticky_frames = max(0, int(kwargs.get("yolo_sticky_frames", 15)))
        self._ocr_parallel = max(
            1,
            int(kwargs.get("ocr_workers_max", kwargs.get("ocr_parallel", 2))),
        )
        self._ocr_worker_count = max(
            1,
            int(kwargs.get("ocr_worker_threads", kwargs.get("ocr_workers_max", 3))),
        )
        self._capture_batch_max = max(1, int(kwargs.get("capture_batch_max", 4)))
        self._priority_queue: deque[StripJob] = deque(maxlen=64)
        self._priority_lock = Lock()
        self._overflow: deque[StripJob] = deque(maxlen=128)
        self._overflow_lock = Lock()

        self.state = KillfeedState(
            max_visible_slots=max_visible_slots,
            cache_ttl_seconds=cache_ttl_seconds,
            pair_cooldown_seconds=pair_cooldown_seconds,
            fuzzy_threshold=fuzzy_threshold,
        )
        self.writer = EventWriter(log_path=log_path, jsonl_path=jsonl_path)
        self._stabilizer = StripStabilizer(max_rows=max_visible_slots)

        if capture is not None:
            self.capture = capture
        else:
            self.capture = CaptureThread(
                camera_source=camera_source,
                width=camera_width,
                height=camera_height,
                fps=capture_fps,
                buffer_seconds=frame_buffer_seconds,
                open_capture=open_capture,
            )

        self._strip_queue: Queue[StripJob] = Queue(maxsize=ocr_queue_max)
        self._stop = Event()
        self._capture_thread: Optional[Thread] = None
        self._ocr_thread: Optional[Thread] = None
        self._ocr_threads: List[Thread] = []
        self._last_strip_image: Optional[np.ndarray] = None
        self._last_row_count = 0
        self._last_row_dhashes: List[str] = []
        self._last_good_row_specs: List[RowDetection] = []
        self._last_yolo_hit_frame = 0
        self._frames_since_queue = 0
        self._snapshot_lock = Lock()
        self._row_ocr_cache: dict[str, tuple[str, str, float]] = {}
        self._ocr_runs_since_gc = 0
        self._stats = {
            "frames_captured": 0,
            "strips_queued": 0,
            "strips_dropped": 0,
            "ocr_runs": 0,
            "events_emitted": 0,
            "dup_skipped": 0,
            "gate_skips": 0,
            "row_cache_hits": 0,
            "yolo_hits": 0,
            "yolo_misses": 0,
            "sticky_recovery": 0,
            "heartbeat_queues": 0,
            "fast_emits": 0,
            "overflow_queued": 0,
        }
        self._stats_lock = Lock()

    def start(self) -> None:
        flags = []
        if self.fast_row0_emit:
            flags.append("fast-row0")
        if self.yolo_heartbeat_frames > 0:
            flags.append(f"heartbeat={self.yolo_heartbeat_frames}f")
        if self.yolo_sticky_frames > 0:
            flags.append(f"sticky={self.yolo_sticky_frames}f")
        extra = f" ({', '.join(flags)})" if flags else ""
        print(
            f"🚀 Starting FIFO killfeed pipeline "
            f"({self._ocr_worker_count} OCR workers{extra})..."
        )
        self._stop.clear()
        self.capture.start()
        self._capture_thread = Thread(
            target=self._capture_loop, daemon=True, name="KillfeedCaptureGate"
        )
        self._ocr_threads = [
            Thread(
                target=self._ocr_worker_loop,
                daemon=True,
                name=f"KillfeedOCRWorker-{i}",
            )
            for i in range(self._ocr_worker_count)
        ]
        self._capture_thread.start()
        for t in self._ocr_threads:
            t.start()

    def stop(self) -> None:
        self._stop.set()
        self.capture.stop()
        if self._capture_thread:
            self._capture_thread.join(timeout=2)
        for t in self._ocr_threads:
            t.join(timeout=2)
        self._ocr_threads.clear()

    @property
    def stats(self) -> dict:
        with self._stats_lock:
            out = dict(self._stats)
            out["ocr_workers"] = sum(1 for t in self._ocr_threads if t.is_alive())
            out["ocr_pool_size"] = self._ocr_worker_count
            out["pending_events"] = (
                len(self._priority_queue)
                + self._strip_queue.qsize()
                + len(self._overflow)
            )
            return out

    def _store_row_specs(self, roi) -> None:
        self._last_good_row_specs = [
            RowDetection(
                bbox=r.bbox,
                confidence=r.confidence,
                class_name=r.class_name,
                crop=np.array([]),
            )
            for r in roi.rows
        ]

    def _row_change_info(self, rows) -> Tuple[bool, List[int], List[str]]:
        """Return (any_row_changed, new_indices, row_dhashes).

        Uses FIFO shift alignment on visual dHashes — NOT naive per-slot compare.
        When kill E pushes A/B/C down, only index 0 is new; shifted rows must
        not be treated as new events.
        """
        hashes = [compute_dhash(r.crop) for r in rows]
        if not hashes:
            return False, [], hashes

        prev = self._last_row_dhashes
        if not prev:
            return True, list(range(len(hashes))), hashes

        n = len(hashes)
        new_indices: List[int] = []

        # Smallest top-shift k where current[k:] visually matches previous[:].
        for k in range(n + 1):
            tail = hashes[k:]
            if not tail:
                break
            prev_slice = prev[: len(tail)]
            if len(prev_slice) == len(tail) and all(
                dhash_distance(a, b) <= self.dhash_max_distance
                for a, b in zip(tail, prev_slice)
            ):
                new_indices = list(range(k))
                break

        if (
            not new_indices
            and prev
            and hashes
            and len(hashes) > len(prev)
            and dhash_distance(hashes[0], prev[0]) <= self.dhash_max_distance
            and len(hashes) >= len(prev)
            and all(
                dhash_distance(hashes[i], prev[i]) <= self.dhash_max_distance
                for i in range(len(prev))
            )
        ):
            # Top stable — queue OCR for newly visible bottom rows only.
            new_indices = list(range(len(prev), len(hashes)))
        elif not new_indices and n > 0:
            # OCR jitter / partial shift — rows with no visual match in previous.
            matched_prev = set()
            for i, h in enumerate(hashes):
                for j, p in enumerate(prev):
                    if j in matched_prev:
                        continue
                    if dhash_distance(h, p) <= self.dhash_max_distance:
                        matched_prev.add(j)
                        break
                else:
                    new_indices.append(i)

        changed = bool(new_indices) or len(hashes) != len(prev)
        return changed, sorted(set(new_indices)), hashes

    def _capture_loop(self) -> None:
        last_processed_frame = 0
        while not self._stop.is_set():
            pending = self.capture.get_frames_after(last_processed_frame)
            if not pending:
                time.sleep(0.001)
                continue

            # HOT PATH: newest frame first — minimum TMS latency for the latest kill.
            latest = pending[-1]
            self._process_capture_frame(latest)
            last_processed_frame = latest.frame_num

            # CATCH-UP: older frames for miss recovery (never before latest).
            if len(pending) > 1:
                for bf in pending[:-1][-self._capture_batch_max :]:
                    self._process_capture_frame(bf, catchup_only=True)
            last_processed_frame = pending[-1].frame_num

    def _pending_depth(self) -> int:
        with self._priority_lock:
            pri = len(self._priority_queue)
        return pri + self._strip_queue.qsize() + len(self._overflow)

    def _process_capture_frame(self, bf, catchup_only: bool = False) -> None:
        with self._stats_lock:
            self._stats["frames_captured"] = bf.frame_num

        roi = detect_strip(
            bf.frame,
            self.model,
            kill_confidence=self.kill_confidence,
            revive_confidence=self.revive_confidence,
            stabilizer=self._stabilizer,
            yolo_lock=self.yolo_lock,
        )
        sticky = False
        if roi is None or roi.strip.size == 0:
            with self._stats_lock:
                self._stats["yolo_misses"] += 1
            if (
                self._last_good_row_specs
                and self.yolo_sticky_frames > 0
                and (bf.frame_num - self._last_yolo_hit_frame) <= self.yolo_sticky_frames
            ):
                roi = reconstruct_strip_from_rows(
                    bf.frame, self._last_good_row_specs, self._stabilizer
                )
                if roi is not None and roi.strip.size > 0:
                    sticky = True
                    with self._stats_lock:
                        self._stats["sticky_recovery"] += 1
            if roi is None or roi.strip.size == 0:
                return
        else:
            with self._stats_lock:
                self._stats["yolo_hits"] += 1
            self._last_yolo_hit_frame = bf.frame_num
            self._store_row_specs(roi)

        row_count = len(roi.rows)
        changed, new_row_indices, row_hashes = self._row_change_info(roi.rows)
        _, new_dhash = strip_changed(
            roi.strip, self.state.last_strip_dhash, max_distance=self.dhash_max_distance
        )
        mean_diff = 999.0
        if self._last_strip_image is not None:
            mean_diff = pixel_mean_diff(roi.strip, self._last_strip_image)

        if not catchup_only:
            self._frames_since_queue += 1
        heartbeat = (
            self.yolo_heartbeat_frames > 0
            and self._frames_since_queue >= self.yolo_heartbeat_frames
        )
        # Under load, skip heartbeat-only strips — they add OCR latency with no new events.
        if heartbeat and not changed and self._pending_depth() > 3:
            with self._stats_lock:
                self._stats["gate_skips"] += 1
            return

        should_queue = (
            not catchup_only
            and (
                self.state.last_strip_dhash is None
                or sticky
                or changed
                or mean_diff >= self.roi_change_threshold
                or row_count != self._last_row_count
                or heartbeat
            )
        ) or (
            catchup_only
            and changed
            and bool(new_row_indices)
        )

        if not should_queue:
            with self._stats_lock:
                self._stats["gate_skips"] += 1
            return

        if heartbeat and not changed and mean_diff < self.roi_change_threshold:
            with self._stats_lock:
                self._stats["heartbeat_queues"] += 1

        # Catch-up frames may queue OCR recovery but must NOT rewind the visual
        # gate — otherwise the next live frame fails to detect new rows.
        if not catchup_only:
            self._last_row_count = row_count
            self._last_row_dhashes = row_hashes
            self.state.last_strip_dhash = new_dhash
            self._last_strip_image = roi.strip.copy()
            self._frames_since_queue = 0

        priority = 0
        if new_row_indices:
            priority = 10 if 0 in new_row_indices else 5
        self._enqueue_strip(
            StripJob(
                strip=roi.strip.copy(),
                roi=roi,
                frame_num=bf.frame_num,
                timestamp=bf.timestamp,
                new_row_indices=tuple(new_row_indices),
                priority=priority,
            )
        )

    def _enqueue_strip(self, job: StripJob) -> None:
        if job.priority > 0 or job.new_row_indices:
            with self._priority_lock:
                if len(self._priority_queue) < self._priority_queue.maxlen:
                    # FIFO among priority jobs — preserves frame / position order.
                    self._priority_queue.append(job)
                    with self._stats_lock:
                        self._stats["strips_queued"] += 1
                    return
        while not self._stop.is_set():
            try:
                self._strip_queue.put_nowait(job)
                with self._stats_lock:
                    self._stats["strips_queued"] += 1
                return
            except Full:
                with self._overflow_lock:
                    if len(self._overflow) < self._overflow.maxlen:
                        self._overflow.append(job)
                        with self._stats_lock:
                            self._stats["strips_queued"] += 1
                            self._stats["overflow_queued"] += 1
                        return
                try:
                    self._strip_queue.get_nowait()
                    with self._stats_lock:
                        self._stats["strips_dropped"] += 1
                except Empty:
                    pass

    def _dequeue_strip_job(self) -> Optional[StripJob]:
        with self._priority_lock:
            if self._priority_queue:
                return self._priority_queue.popleft()
        try:
            return self._strip_queue.get_nowait()
        except Empty:
            pass
        with self._overflow_lock:
            if self._overflow:
                return self._overflow.popleft()
        return None

    def _ocr_worker_loop(self) -> None:
        while not self._stop.is_set():
            job = self._dequeue_strip_job()
            if job is None:
                try:
                    job = self._strip_queue.get(timeout=0.005)
                except Empty:
                    continue
            self._process_strip_job(job)

    def _maybe_gc(self) -> None:
        self._ocr_runs_since_gc += 1
        if self._ocr_runs_since_gc >= 15:
            self._ocr_runs_since_gc = 0
            gc.collect()

    def _cached_row_names(self, row) -> tuple[str, str, float]:
        rh = compute_dhash(row.crop)

        # Exact hit
        cached = self._row_ocr_cache.get(rh)
        if cached is not None:
            k, v, c = cached
            if k or v:
                with self._stats_lock:
                    self._stats["row_cache_hits"] += 1
                return cached

        # Fuzzy hit — row shifted by 1-2px (stabilizer drift) → different dhash
        # but identical text content; distance ≤ 3 is safe given hash_size=8.
        for key, val in self._row_ocr_cache.items():
            if dhash_distance(rh, key) <= 3:
                k, v, c = val
                if k or v:
                    self._row_ocr_cache[rh] = val  # promote to exact for next time
                    with self._stats_lock:
                        self._stats["row_cache_hits"] += 1
                    return val

        names = self.ocr_provider.extract_row_names(row.crop)
        k, v, c = names
        if k or v:
            self._row_ocr_cache[rh] = names
            if len(self._row_ocr_cache) > 128:
                self._row_ocr_cache.pop(next(iter(self._row_ocr_cache)))
        return names

    def _display_position(self, row_index: int) -> int:
        """1 = top/newest row, 2 = second, etc."""
        return row_index + 1

    def _parse_row_cached(self, row, position: int) -> Optional[ParsedRow]:
        killer, victim, confidence = self._cached_row_names(row)
        from killfeed.classifier import classify_row, is_confident_classification
        from killfeed.parser import _row_valid
        from killfeed.queue_state import KillfeedEvent, make_signature

        classification = classify_row(
            row.crop,
            killer=killer,
            victim=victim,
            yolo_class=row.class_name,
            yolo_conf=row.confidence,
        )

        # ── STATUS TRACE: classifier output ──
        print(
            f"🔬 STATUS_TRACE [parse] pos={position} "
            f"YOLO_CLASS={row.class_name} YOLO_CONF={row.confidence:.3f} | "
            f"killer={killer!r} victim={victim!r} | "
            f"event_type={classification.event_type!r} kill_type={classification.kill_type!r} "
            f"CANONICAL={classification.canonical!r} TMS_STATUS={classification.tms_status!r} "
            f"confident={classification.confident}"
        )

        if not is_confident_classification(classification):
            print(f"   ❌ STATUS_TRACE [parse] DROPPED: not confident (class={row.class_name})")
            return None
        if not _row_valid(killer, victim, confidence, classification, self.name_validator):
            print(f"   ❌ STATUS_TRACE [parse] DROPPED: name validation failed (k={killer!r} v={victim!r})")
            return None

        display_killer = killer if killer else ""
        sig = make_signature(
            display_killer, victim, classification.canonical, classification.icon_category
        )
        display_pos = self._display_position(position)
        event = KillfeedEvent(
            killer=display_killer,
            victim=victim,
            event=classification.canonical,
            event_type=classification.event_type,
            kill_type=classification.kill_type,
            weapon=classification.icon_category,
            gun_name=classification.gun_name,
            position=display_pos,
        )
        return ParsedRow(
            killer=display_killer,
            victim=victim,
            event_type=classification.event_type,
            kill_type=classification.kill_type,
            canonical=classification.canonical,
            tms_status=classification.tms_status,
            weapon=classification.icon_category,
            gun_name=classification.gun_name,
            confidence=confidence,
            position=display_pos,
            row_crop=row.crop,
            signature=sig,
            event=event,
        )

    def _parse_rows_parallel(self, rows, priority_indices: Optional[Tuple[int, ...]] = None) -> List[ParsedRow]:
        if not rows:
            return []

        indices = list(range(len(rows)))
        if priority_indices:
            pri = [i for i in priority_indices if 0 <= i < len(rows)]
            indices = pri + [i for i in indices if i not in pri]
        elif self.fast_row0_emit and len(indices) > 1:
            indices = [0] + [i for i in indices if i != 0]

        parsed_by_pos: dict[int, Optional[ParsedRow]] = {}

        if len(rows) == 1 or self._ocr_parallel <= 1:
            for i in indices:
                pr = self._parse_row_cached(rows[i], i)
                if pr is not None:
                    parsed_by_pos[i] = pr
        else:
            workers = min(self._ocr_parallel, len(rows))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(self._parse_row_cached, rows[i], i): i for i in indices
                }
                for fut in as_completed(futures):
                    i = futures[fut]
                    try:
                        pr = fut.result()
                        if pr is not None:
                            parsed_by_pos[i] = pr
                    except Exception:
                        pass

        return [parsed_by_pos[i] for i in sorted(parsed_by_pos)]

    def _try_fast_row0_emit(self, rows, job: StripJob) -> None:
        """OCR row 0 and push to TMS immediately — does NOT mutate FIFO snapshot."""
        if not self.fast_row0_emit or not rows or 0 not in job.new_row_indices:
            return
        if not self.state.previous_snapshot:
            return

        pr = self._parse_row_cached(rows[0], 0)
        if pr is None:
            return

        prev = self.state.previous_snapshot
        tentative = [pr.signature] + prev[: self.state.max_visible_slots - 1]
        new_sigs = find_new_slots(
            prev, tentative, fuzzy_threshold=self.state.fuzzy_threshold
        )
        if not new_sigs:
            return
        if not any(
            pr.signature == s
            or signatures_match(pr.signature, s, self.state.fuzzy_threshold)
            for s in new_sigs
        ):
            return
        if self._emit_parsed_row(pr, job):
            with self._stats_lock:
                self._stats["fast_emits"] += 1

    def _emit_parsed_row(self, row: ParsedRow, job: StripJob) -> bool:
        event_hash = row.event.event_hash

        # ── STATUS TRACE: emission gate ──
        print(
            f"🔬 STATUS_TRACE [emit_check] "
            f"{row.killer}→{row.victim} pos={row.position} frame={job.frame_num} | "
            f"CANONICAL={row.canonical!r} TMS_STATUS={row.tms_status!r} "
            f"event_hash={event_hash!r} sig={row.signature!r}"
        )

        if not self.state.should_emit(row.killer, row.victim, row.canonical, event_hash):
            print(
                f"   🚫 STATUS_TRACE [emit_check] BLOCKED by dedup "
                f"(hash={event_hash!r} killer={row.killer!r} victim={row.victim!r} "
                f"canonical={row.canonical!r})"
            )
            with self._stats_lock:
                self._stats["dup_skipped"] += 1
            return False

        row.event.frame = job.frame_num
        row.event.time = job.timestamp
        row.event.position = row.position

        self.state.mark_emitted(row.killer, row.victim, row.canonical, event_hash)
        seq = self.state.sequence
        self.writer.log_event(row.event, frame=job.frame_num, sequence=seq)

        # ── STATUS TRACE: emitted ──
        print(
            f"   ✅ STATUS_TRACE [EMITTED] seq={seq} "
            f"{row.killer}→{row.victim} pos={row.position} | "
            f"CANONICAL={row.canonical!r} TMS_STATUS={row.tms_status!r} "
            f"event_hash={event_hash!r}"
        )

        try:
            self.on_event(row, job.frame_num, seq)
        except Exception as exc:
            print(f"⚠️ on_event callback error: {type(exc).__name__}")

        with self._stats_lock:
            self._stats["events_emitted"] += 1
        return True

    def _find_row_for_sig(self, sig: str, parsed_rows: list) -> Optional[ParsedRow]:
        for pr in parsed_rows:
            if pr.signature == sig:
                return pr
        for pr in parsed_rows:
            if signatures_match(pr.signature, sig, self.state.fuzzy_threshold):
                return pr
        killer, victim, canonical, _icon = parse_signature(sig)
        for pr in parsed_rows:
            if pr.killer == killer and pr.victim == victim and pr.canonical == canonical:
                return pr
        return None

    def _process_strip_job(self, job: StripJob) -> None:
        try:
            with self._stats_lock:
                self._stats["ocr_runs"] += 1

            if not job.roi.rows:
                return

            job.roi.rows = consolidate_overlapping_rows(list(job.roi.rows))

            # ── STATUS TRACE: strip-level YOLO classes ──
            row_classes = [
                f"row{i}={r.class_name}@{r.confidence:.2f}"
                for i, r in enumerate(job.roi.rows)
            ]
            print(
                f"🔬 STATUS_TRACE [strip] frame={job.frame_num} "
                f"rows={len(job.roi.rows)} new_indices={job.new_row_indices} "
                f"classes=[{', '.join(row_classes)}]"
            )

            with self._snapshot_lock:
                # Fast TMS for newest row only — snapshot updated once below.
                self._try_fast_row0_emit(job.roi.rows, job)

                parsed_rows = self._parse_rows_parallel(
                    job.roi.rows, priority_indices=job.new_row_indices
                )
                if not parsed_rows:
                    self._maybe_gc()
                    return

                # Single atomic FIFO commit — sole authority for position tracking.
                sigs = signatures_from_rows(parsed_rows)
                new_sigs = self.state.diff_and_commit(sigs)

                print(
                    f"🔬 STATUS_TRACE [fifo] frame={job.frame_num} "
                    f"parsed={len(parsed_rows)} new_sigs={len(new_sigs)} "
                    f"prev_snapshot={len(self.state.previous_snapshot)} "
                    f"sigs={sigs!r}"
                )

                if not new_sigs:
                    self._maybe_gc()
                    return

                sig_to_row = {r.signature: r for r in parsed_rows}
                for sig in reversed(new_sigs):
                    row = sig_to_row.get(sig) or self._find_row_for_sig(sig, parsed_rows)
                    if row is not None:
                        self._emit_parsed_row(row, job)
            self._maybe_gc()
        except Exception as exc:
            print(f"⚠️ Strip processing error: {type(exc).__name__}: {exc}")

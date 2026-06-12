"""Two-thread killfeed pipeline: capture + visual gate | OCR worker + FIFO diff."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Callable, Optional

import numpy as np

from killfeed.capture import CaptureThread, CameraSource
from killfeed.dhash import pixel_mean_diff, strip_changed
from killfeed.events import EventWriter
from killfeed.parser import ParsedRow, signatures_from_rows
from killfeed.queue_state import KillfeedState, parse_signature, signatures_match
from killfeed.roi import StripStabilizer, detect_strip


@dataclass
class StripJob:
    strip: np.ndarray
    roi: object
    frame_num: int
    timestamp: float


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

        self.state = KillfeedState(
            max_visible_slots=max_visible_slots,
            cache_ttl_seconds=cache_ttl_seconds,
            pair_cooldown_seconds=pair_cooldown_seconds,
            fuzzy_threshold=fuzzy_threshold,
        )
        self.writer = EventWriter(log_path=log_path, jsonl_path=jsonl_path)
        self._stabilizer = StripStabilizer(max_rows=max_visible_slots)

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
        self._last_strip_image: Optional[np.ndarray] = None
        self._last_row_count = 0
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
        }
        self._stats_lock = Lock()

    def start(self) -> None:
        print("🚀 Starting FIFO killfeed pipeline (capture + OCR worker)...")
        self._stop.clear()
        self.capture.start()
        self._capture_thread = Thread(
            target=self._capture_loop, daemon=True, name="KillfeedCaptureGate"
        )
        self._ocr_thread = Thread(
            target=self._ocr_worker_loop, daemon=True, name="KillfeedOCRWorker"
        )
        self._capture_thread.start()
        self._ocr_thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.capture.stop()
        if self._capture_thread:
            self._capture_thread.join(timeout=2)
        if self._ocr_thread:
            self._ocr_thread.join(timeout=2)

    @property
    def stats(self) -> dict:
        with self._stats_lock:
            return dict(self._stats)

    def _capture_loop(self) -> None:
        last_processed_frame = 0
        while not self._stop.is_set():
            bf = self.capture.get_latest()
            if bf is None or bf.frame_num <= last_processed_frame:
                time.sleep(0.002)
                continue
            last_processed_frame = bf.frame_num

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
            if roi is None or roi.strip.size == 0:
                with self._stats_lock:
                    self._stats["yolo_misses"] += 1
                time.sleep(0.005)
                continue

            with self._stats_lock:
                self._stats["yolo_hits"] += 1

            row_count = len(roi.rows)
            changed, new_dhash = strip_changed(
                roi.strip, self.state.last_strip_dhash, max_distance=self.dhash_max_distance
            )
            mean_diff = 999.0
            if self._last_strip_image is not None:
                mean_diff = pixel_mean_diff(roi.strip, self._last_strip_image)

            should_queue = (
                self.state.last_strip_dhash is None
                or changed
                or mean_diff >= self.roi_change_threshold
                or row_count != self._last_row_count
            )

            if not should_queue:
                with self._stats_lock:
                    self._stats["gate_skips"] += 1
                continue

            self._last_row_count = row_count
            self.state.last_strip_dhash = new_dhash
            self._last_strip_image = roi.strip.copy()
            self._enqueue_strip(
                StripJob(
                    strip=roi.strip.copy(),
                    roi=roi,
                    frame_num=bf.frame_num,
                    timestamp=bf.timestamp,
                )
            )

    def _enqueue_strip(self, job: StripJob) -> None:
        while not self._stop.is_set():
            try:
                self._strip_queue.put_nowait(job)
                with self._stats_lock:
                    self._stats["strips_queued"] += 1
                return
            except Full:
                try:
                    self._strip_queue.get_nowait()
                    with self._stats_lock:
                        self._stats["strips_dropped"] += 1
                except Empty:
                    pass

    def _ocr_worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._strip_queue.get(timeout=0.02)
            except Empty:
                continue
            self._process_strip_job(job)

    def _maybe_gc(self) -> None:
        self._ocr_runs_since_gc += 1
        if self._ocr_runs_since_gc >= 15:
            self._ocr_runs_since_gc = 0
            gc.collect()

    def _cached_row_names(self, row) -> tuple[str, str, float]:
        from killfeed.dhash import compute_dhash

        rh = compute_dhash(row.crop)
        cached = self._row_ocr_cache.get(rh)
        if cached is not None:
            k, v, c = cached
            if k or v:
                with self._stats_lock:
                    self._stats["row_cache_hits"] += 1
                return cached
        names = self.ocr_provider.extract_row_names(row.crop)
        k, v, c = names
        if k or v:
            self._row_ocr_cache[rh] = names
            if len(self._row_ocr_cache) > 64:
                self._row_ocr_cache.pop(next(iter(self._row_ocr_cache)))
        return names

    def _parse_row_cached(self, row, position: int) -> Optional[ParsedRow]:
        killer, victim, confidence = self._cached_row_names(row)
        from killfeed.classifier import classify_row, is_confident_classification
        from killfeed.parser import _row_valid
        from killfeed.queue_state import KillfeedEvent, make_signature

        classification = classify_row(
            row.crop, killer=killer, victim=victim, yolo_class=row.class_name
        )
        if not is_confident_classification(classification):
            return None
        if not _row_valid(killer, victim, confidence, classification, self.name_validator):
            return None

        display_killer = killer if killer else ""
        sig = make_signature(
            display_killer, victim, classification.canonical, classification.icon_category
        )
        event = KillfeedEvent(
            killer=display_killer,
            victim=victim,
            event=classification.canonical,
            event_type=classification.event_type,
            kill_type=classification.kill_type,
            weapon=classification.icon_category,
            position=position,
        )
        return ParsedRow(
            killer=display_killer,
            victim=victim,
            event_type=classification.event_type,
            kill_type=classification.kill_type,
            canonical=classification.canonical,
            tms_status=classification.tms_status,
            weapon=classification.icon_category,
            confidence=confidence,
            position=position,
            row_crop=row.crop,
            signature=sig,
            event=event,
        )

    def _emit_parsed_row(self, row: ParsedRow, job: StripJob) -> bool:
        event_hash = row.event.event_hash
        if not self.state.should_emit(row.killer, row.victim, row.canonical, event_hash):
            with self._stats_lock:
                self._stats["dup_skipped"] += 1
            return False

        row.event.frame = job.frame_num
        row.event.time = job.timestamp
        row.event.position = row.position

        self.state.mark_emitted(row.killer, row.victim, row.canonical, event_hash)
        seq = self.state.sequence
        self.writer.log_event(row.event, frame=job.frame_num, sequence=seq)

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

            parsed_rows: list[ParsedRow] = []
            for i, row in enumerate(job.roi.rows):
                pr = self._parse_row_cached(row, i)
                if pr is not None:
                    parsed_rows.append(pr)

            if not parsed_rows:
                self._maybe_gc()
                return

            new_sigs = self.state.diff_and_commit(signatures_from_rows(parsed_rows))
            if not new_sigs:
                self._maybe_gc()
                return

            sig_to_row = {r.signature: r for r in parsed_rows}
            for sig in new_sigs:
                row = sig_to_row.get(sig) or self._find_row_for_sig(sig, parsed_rows)
                if row is not None:
                    self._emit_parsed_row(row, job)
            self._maybe_gc()
        except Exception as exc:
            print(f"⚠️ Strip processing error: {type(exc).__name__}: {exc}")

"""Event-fingerprint killfeed tracker with Hungarian matching."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from rapidfuzz import fuzz

    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

if TYPE_CHECKING:
    from killfeed.parser import ParsedRow

Detection = Tuple[str, str, str, str, float]  # killer, victim, event, weapon, y


@dataclass
class KillfeedTrack:
    track_id: int
    killer: str
    victim: str
    event: str
    weapon: str
    y: float
    first_seen: float
    last_seen: float
    last_frame: int


def _ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if RAPIDFUZZ_AVAILABLE:
        return fuzz.ratio(a, b) / 100.0
    return 1.0 if a.upper() == b.upper() else 0.0


def fingerprint_similarity(
    det_killer: str,
    det_victim: str,
    det_event: str,
    det_weapon: str,
    track: KillfeedTrack,
) -> float:
    killer_sim = _ratio(det_killer, track.killer)
    victim_sim = _ratio(det_victim, track.victim)
    # Use product so that both killer AND victim must match for a high score.
    # A different victim zeroes out identity even if the killer is identical.
    identity = killer_sim * victim_sim
    event_sim = 1.0 if det_event == track.event else 0.0
    weapon_sim = 1.0 if det_weapon == track.weapon else 0.0
    return 0.5 * identity + 0.3 * event_sim + 0.2 * weapon_sim


def position_similarity(det_y: float, track_y: float) -> float:
    return max(0.0, 1.0 - abs(track_y - det_y) * 3.0)


def ocr_similarity(det_killer: str, det_victim: str, track: KillfeedTrack) -> float:
    return (_ratio(det_killer, track.killer) + _ratio(det_victim, track.victim)) / 2.0


def match_score(
    det_killer: str,
    det_victim: str,
    det_event: str,
    det_weapon: str,
    det_y: float,
    track: KillfeedTrack,
) -> float:
    fp = fingerprint_similarity(det_killer, det_victim, det_event, det_weapon, track)
    pos = position_similarity(det_y, track.y)
    ocr = ocr_similarity(det_killer, det_victim, track)
    return 0.7 * fp + 0.2 * pos + 0.1 * ocr


class ActiveTrackManager:
    """Track killfeed events across frames using Hungarian assignment."""

    def __init__(
        self,
        max_age_seconds: float = 30.0,
        match_threshold: float = 0.60,
    ):
        self.max_age_seconds = max_age_seconds
        self.match_threshold = match_threshold
        self._tracks: List[KillfeedTrack] = []
        self._next_id = 1

    @property
    def active_tracks(self) -> List[KillfeedTrack]:
        return list(self._tracks)

    def _prune(self, now: float) -> None:
        cutoff = now - self.max_age_seconds
        self._tracks = [t for t in self._tracks if t.last_seen >= cutoff]

    def _create_track(
        self,
        killer: str,
        victim: str,
        event: str,
        weapon: str,
        y: float,
        frame_num: int,
        now: float,
    ) -> KillfeedTrack:
        track = KillfeedTrack(
            track_id=self._next_id,
            killer=killer,
            victim=victim,
            event=event,
            weapon=weapon,
            y=y,
            first_seen=now,
            last_seen=now,
            last_frame=frame_num,
        )
        self._next_id += 1
        self._tracks.append(track)
        return track

    def _find_parsed_row(
        self,
        killer: str,
        victim: str,
        event: str,
        weapon: str,
        parsed_rows: Optional[Sequence["ParsedRow"]],
    ) -> Optional["ParsedRow"]:
        if not parsed_rows:
            return None
        for row in parsed_rows:
            if (
                row.killer == killer
                and row.victim == victim
                and row.canonical == event
                and row.weapon == weapon
            ):
                return row
        for row in parsed_rows:
            if (
                _ratio(row.killer, killer) >= 0.88
                and _ratio(row.victim, victim) >= 0.88
                and row.canonical == event
            ):
                return row
        return None

    def update(
        self,
        detections: Sequence[Detection],
        frame_num: int,
        parsed_rows: Optional[Sequence["ParsedRow"]] = None,
    ) -> List["ParsedRow"]:
        """Match detections to active tracks; return rows to emit (new + upgrades)."""
        now = time.time()
        self._prune(now)

        if not detections:
            return []

        emit_rows: List["ParsedRow"] = []

        if not self._tracks:
            # No active tracks — all detections are new events.
            # When parsed_rows is None this is the initialization seed call;
            # suppress emission so stale on-screen history is not re-emitted.
            # Any subsequent call with parsed_rows provided must emit normally.
            should_emit = parsed_rows is not None
            for killer, victim, event, weapon, y in detections:
                self._create_track(killer, victim, event, weapon, y, frame_num, now)
                if should_emit:
                    row = self._find_parsed_row(killer, victim, event, weapon, parsed_rows)
                    if row is not None:
                        emit_rows.append(row)
            return emit_rows

        n_dets = len(detections)
        n_tracks = len(self._tracks)
        cost = np.ones((n_dets, n_tracks), dtype=np.float64)

        for i, (dk, dv, de, dw, dy) in enumerate(detections):
            for j, track in enumerate(self._tracks):
                score = match_score(dk, dv, de, dw, dy, track)
                cost[i, j] = 1.0 - score

        row_ind, col_ind = linear_sum_assignment(cost)
        matched_dets: set[int] = set()
        matched_tracks: set[int] = set()

        for det_idx, track_idx in zip(row_ind, col_ind):
            score = 1.0 - cost[det_idx, track_idx]
            if score < self.match_threshold:
                continue

            matched_dets.add(det_idx)
            matched_tracks.add(track_idx)

            dk, dv, de, dw, dy = detections[det_idx]
            track = self._tracks[track_idx]
            track.killer = dk
            track.victim = dv
            track.weapon = dw
            track.y = dy
            track.last_seen = now
            track.last_frame = frame_num

            if track.event != de:
                track.event = de
                row = self._find_parsed_row(dk, dv, de, dw, parsed_rows)
                if row is not None:
                    emit_rows.append(row)

        for det_idx, (dk, dv, de, dw, dy) in enumerate(detections):
            if det_idx in matched_dets:
                continue
            self._create_track(dk, dv, de, dw, dy, frame_num, now)
            row = self._find_parsed_row(dk, dv, de, dw, parsed_rows)
            if row is not None:
                emit_rows.append(row)

        return emit_rows

"""Temporal row tracker: accumulate per-frame fusion across the life of a row.

A killfeed row is on screen for several frames.  Instead of trusting one noisy
frame, we accumulate evidence per (killer, victim) track and only resolve to an
event when the aggregate is confident and self-consistent.  A single bad frame
(e.g. a red background-bar flash) can no longer flip the output.

Note: the capture stage only enqueues *changed* strips, so a static row may be
observed only a handful of times.  Resolution therefore works on a single strong
frame too — it does not require many votes — but extra frames raise confidence
and damp single-frame noise.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Optional

from killfeed.fusion import FrameFusion
from killfeed.observers import EVENT_KILL, EVENT_KNOCK, EVENT_REVIVE

try:
    from rapidfuzz import fuzz

    _HAS_FUZZ = True
except ImportError:
    _HAS_FUZZ = False

_EVENT_CLASSES = (EVENT_KILL, EVENT_KNOCK, EVENT_REVIVE)

# Resolution thresholds (aggregate over observed frames).
TAU_RESOLVE = 0.52     # aggregate confidence required to commit a label
TAU_AGREE = 0.55       # fraction of frames that must back the winning label
TAU_MARGIN = 0.08      # winning aggregate must beat runner-up by this much
MIN_FRAMES = 1         # resolve as early as a single confident frame
MAX_OBSERVE = 5        # after this many ambiguous frames, emit explicit unknown

TRACK_TTL_SECONDS = 30.0
_FUZZY_KEY_THRESHOLD = 88.0


def _canon(name: str) -> str:
    return (name or "").strip().upper()


@dataclass
class RowTrack:
    killer: str
    victim: str
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    frames: int = 0
    votes: Counter = field(default_factory=Counter)
    weighted: Dict[str, float] = field(default_factory=lambda: {c: 0.0 for c in _EVENT_CLASSES})
    emitted_labels: set = field(default_factory=set)
    _pending_label: Optional[str] = None
    _unknown_emitted: bool = False

    def observe(self, fusion: FrameFusion) -> None:
        self.last_seen = time.time()
        self.frames += 1
        if fusion.label in _EVENT_CLASSES:
            self.votes[fusion.label] += 1
            self.weighted[fusion.label] += fusion.confidence

    def _aggregate(self):
        scores = {
            c: (self.weighted[c] / self.frames if self.frames else 0.0)
            for c in _EVENT_CLASSES
        }
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        (top_label, top_score), (_second, second_score) = ranked[0], ranked[1]
        margin = top_score - second_score
        agreement = (self.votes[top_label] / self.frames) if self.frames else 0.0
        return top_label, top_score, margin, agreement, scores

    def try_resolve(self) -> Optional[FrameFusion]:
        """Return a confident :class:`FrameFusion`, an explicit unknown, or None
        (keep observing).  ``None`` means *hold* — no event is emitted."""
        if self.frames < MIN_FRAMES:
            return None

        top_label, top_score, margin, agreement, scores = self._aggregate()

        resolved = (
            top_score >= TAU_RESOLVE
            and agreement >= TAU_AGREE
            and margin >= TAU_MARGIN
        )
        if resolved and top_label not in self.emitted_labels:
            self._pending_label = top_label
            return FrameFusion(top_label, top_score, margin, scores, "temporal_resolved")

        # Persistently ambiguous → surface one explicit unknown for review.
        if (
            self.frames >= MAX_OBSERVE
            and not self._unknown_emitted
            and not self.emitted_labels
        ):
            self._unknown_emitted = True
            return FrameFusion("unknown", top_score, margin, scores, "no_temporal_consensus")

        return None

    def confirm_emitted(self) -> None:
        if self._pending_label is not None:
            self.emitted_labels.add(self._pending_label)
            self._pending_label = None


class RowTracker:
    """Holds :class:`RowTrack`s keyed by fuzzy (killer, victim)."""

    def __init__(self):
        self._tracks: Dict[tuple, RowTrack] = {}

    def _prune(self, now: float) -> None:
        stale = [k for k, t in self._tracks.items() if now - t.last_seen > TRACK_TTL_SECONDS]
        for k in stale:
            del self._tracks[k]

    def _find_key(self, killer: str, victim: str) -> tuple:
        ck, cv = _canon(killer), _canon(victim)
        exact = (ck, cv)
        if exact in self._tracks:
            return exact
        if _HAS_FUZZ:
            for (tk, tv) in self._tracks:
                if (
                    fuzz.ratio(ck, tk) >= _FUZZY_KEY_THRESHOLD
                    and fuzz.ratio(cv, tv) >= _FUZZY_KEY_THRESHOLD
                ):
                    return (tk, tv)
        return exact

    def update(
        self,
        killer: str,
        victim: str,
        fusion: FrameFusion,
        yolo_conf: float = 0.0,
        ocr_conf: float = 0.0,
    ) -> RowTrack:
        now = time.time()
        self._prune(now)
        key = self._find_key(killer, victim)
        track = self._tracks.get(key)
        if track is None:
            track = RowTrack(killer=killer, victim=victim)
            self._tracks[key] = track
        track.observe(fusion)
        return track

    def mark_emitted(self, killer: str, victim: str) -> None:
        key = self._find_key(killer, victim)
        track = self._tracks.get(key)
        if track is not None:
            track.confirm_emitted()

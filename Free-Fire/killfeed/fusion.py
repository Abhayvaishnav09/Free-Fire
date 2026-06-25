"""Evidence fusion: combine per-channel observations into one frame decision.

Fusion is a weighted vote across channels with a hard ABSTAIN floor.  A single
weak channel can never force a label; corroboration across channels is what
builds confidence.  When evidence is insufficient the label is ``"unknown"`` and
the caller must NOT emit an event.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

from killfeed.observers import (
    CHANNEL_WEIGHT,
    EVENT_KILL,
    EVENT_KNOCK,
    EVENT_REVIVE,
    Observation,
)

UNKNOWN = "unknown"

# Confidence floor below which a frame is treated as "unknown" outright.
_CONF_FLOOR = 0.30
# A decision backed by only one channel is capped (no corroboration).
_SINGLE_CHANNEL_PENALTY = 0.80

_EVENT_CLASSES = (EVENT_KILL, EVENT_KNOCK, EVENT_REVIVE)


@dataclass
class FrameFusion:
    label: str                       # EVENT_* or "unknown"
    confidence: float                # fused confidence in [0, 1]
    margin: float                    # top score - runner-up score
    scores: Dict[str, float] = field(default_factory=dict)
    reason: str = ""


def fuse_observations(observations: Iterable[Observation]) -> FrameFusion:
    obs = list(observations)
    voting = [o for o in obs if not o.abstained and o.label in _EVENT_CLASSES]

    if not voting:
        return FrameFusion(UNKNOWN, 0.0, 0.0, {}, "all_channels_abstained")

    # Weighted evidence per class, normalised by the weight of *all* channels
    # that produced a usable vote (so disagreement lowers confidence).
    agg: Dict[str, float] = {c: 0.0 for c in _EVENT_CLASSES}
    n_channels: Dict[str, set] = {c: set() for c in _EVENT_CLASSES}
    total_w = 0.0
    for o in voting:
        w = CHANNEL_WEIGHT.get(o.channel, 0.10)
        agg[o.label] += w * o.score
        n_channels[o.label].add(o.channel)
        total_w += w

    scores = {c: (agg[c] / total_w if total_w > 0 else 0.0) for c in _EVENT_CLASSES}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (top_label, top_score), (_second_label, second_score) = ranked[0], ranked[1]
    margin = top_score - second_score

    confidence = top_score
    if len(n_channels[top_label]) < 2:
        confidence *= _SINGLE_CHANNEL_PENALTY

    if confidence < _CONF_FLOOR:
        return FrameFusion(UNKNOWN, confidence, margin, scores, "below_conf_floor")

    return FrameFusion(top_label, confidence, margin, scores, "fused")

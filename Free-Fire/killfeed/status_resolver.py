"""Killfeed status from victim NAME text colour only.

Rules (deterministic):
  Red victim text  → elimination (TMS kill)
  White victim text → knock (TMS gun knockout)
  Green victim text → revive

Background, box, glow, knock icons, and YOLO class are NOT used.
Event capture must never depend on status confidence — use resolve_killfeed_status_best().
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Victim name letters — excludes team badge (left) and knock icon (right).
_VICTIM_TEXT_X0 = 0.48
_VICTIM_TEXT_X1 = 0.65

_MIN_TEXT_PIXELS = 8
_MIN_WIN_SHARE = 0.22
_MIN_WIN_MARGIN = 1.15


@dataclass
class ResolvedStatus:
    event_type: str  # knock | elimination | revive | unknown
    confidence: float
    source: str = ""


def _victim_name_text_region(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    x0 = int(w * _VICTIM_TEXT_X0)
    x1 = max(x0 + 1, int(w * _VICTIM_TEXT_X1))
    return crop[:, x0:x1]


def _count_victim_text_color_pixels(region: np.ndarray) -> tuple[int, int, int]:
    """Count red / white / green pixels that qualify as victim name TEXT only."""
    if region is None or region.size == 0:
        return 0, 0, 0

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    green_text = (h >= 35) & (h <= 98) & (s >= 55) & (v >= 55)
    red_text = ((h <= 16) | (h >= 165)) & (s >= 70) & (v >= 55)
    white_text = (v >= 135) & (s <= 60)

    green_n = int(np.count_nonzero(green_text))
    red_n = int(np.count_nonzero(red_text & ~green_text))
    white_n = int(np.count_nonzero(white_text & ~green_text & ~red_text))
    return red_n, white_n, green_n


def _rank_text_colors(red_n: int, white_n: int, green_n: int) -> list[tuple[int, str, str]]:
    return sorted(
        [
            (red_n, "elimination", "victim_text_red"),
            (white_n, "knock", "victim_text_white"),
            (green_n, "revive", "victim_text_green"),
        ],
        key=lambda row: row[0],
        reverse=True,
    )


def resolve_killfeed_status(
    crop,
    yolo_class: str | None = None,
) -> ResolvedStatus:
    """Strict text-only status (may return unknown). ``yolo_class`` is ignored."""
    if crop is None or getattr(crop, "size", 0) == 0:
        return ResolvedStatus("unknown", 0.0, "empty_crop")

    region = _victim_name_text_region(crop)
    red_n, white_n, green_n = _count_victim_text_color_pixels(region)
    total = red_n + white_n + green_n
    if total < _MIN_TEXT_PIXELS:
        return ResolvedStatus("unknown", 0.0, "insufficient_text_pixels")

    ranked = _rank_text_colors(red_n, white_n, green_n)
    winner_n, winner_et, winner_src = ranked[0]
    runner_n = ranked[1][0]

    if winner_n < total * _MIN_WIN_SHARE:
        return ResolvedStatus("unknown", 0.0, "no_dominant_text_color")
    if runner_n > 0 and winner_n < runner_n * _MIN_WIN_MARGIN:
        return ResolvedStatus("unknown", 0.0, "ambiguous_text_color")

    confidence = min(1.0, winner_n / float(total))
    return ResolvedStatus(winner_et, confidence, winner_src)


def resolve_killfeed_status_best(
    crop,
    yolo_class: str | None = None,
) -> ResolvedStatus:
    """
    Always returns knock, elimination, or revive — never blocks event capture.
    Uses dominant victim text colour; defaults to knock only when no text pixels match.
    """
    strict = resolve_killfeed_status(crop, yolo_class)
    if strict.event_type != "unknown":
        return strict

    if crop is None or getattr(crop, "size", 0) == 0:
        return ResolvedStatus("knock", 0.30, "default_knock_empty")

    region = _victim_name_text_region(crop)
    red_n, white_n, green_n = _count_victim_text_color_pixels(region)
    ranked = _rank_text_colors(red_n, white_n, green_n)
    winner_n, winner_et, winner_src = ranked[0]
    total = red_n + white_n + green_n

    if winner_n > 0 and total > 0:
        return ResolvedStatus(
            winner_et,
            max(0.30, winner_n / float(total)),
            f"best_effort_{winner_src}",
        )
    return ResolvedStatus("knock", 0.30, "default_knock")


def is_authoritative(resolved: ResolvedStatus) -> bool:
    return resolved.event_type != "unknown" and resolved.confidence >= 0.28

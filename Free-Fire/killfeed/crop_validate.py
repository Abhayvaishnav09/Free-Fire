"""Reject team HUD / scoreboard bars mis-detected as killfeed rows."""

from __future__ import annotations

import re

import cv2
import numpy as np

# Free Fire player tags are typically TEAM.PLAYER (e.g. ARZ.KUNAL10).
_PLAYER_TAG_RE = re.compile(r"^[A-Z0-9][A-Z0-9._-]*\.[A-Z0-9][A-Z0-9._-]+$", re.I)


def looks_like_team_hud_bar(crop: np.ndarray) -> bool:
    """
    Team status bars show logo + team code + skull + kill count + red vertical bars.
    They are not killer → victim killfeed strips.
    """
    if crop is None or getattr(crop, "size", 0) == 0:
        return False
    h, w = crop.shape[:2]
    if w < 80 or h < 10:
        return False

    # Right segment: HUD bars have saturated vertical red stripes (health/ammo style).
    right = crop[:, int(w * 0.62) :]
    if right.size == 0:
        return False
    hsv = cv2.cvtColor(right, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 90, 90]), np.array([10, 255, 255])),
        cv2.inRange(hsv, np.array([170, 90, 90]), np.array([180, 255, 255])),
    )
    red_ratio = cv2.countNonZero(red) / max(right.shape[0] * right.shape[1], 1)
    if red_ratio > 0.14:
        return True

    # Yellow rank badge (e.g. "3rd") on placement / team overlays.
    yellow = cv2.inRange(hsv, np.array([18, 120, 120]), np.array([38, 255, 255]))
    left = crop[:, : int(w * 0.35)]
    if left.size > 0:
        hsv_l = cv2.cvtColor(left, cv2.COLOR_BGR2HSV)
        yellow_l = cv2.inRange(hsv_l, np.array([18, 120, 120]), np.array([38, 255, 255]))
        if cv2.countNonZero(yellow_l) / max(left.shape[0] * left.shape[1], 1) > 0.08:
            return True

    return False


def _is_numeric_token(name: str) -> bool:
    s = (name or "").strip()
    if not s:
        return True
    return bool(re.fullmatch(r"\d+", s))


def looks_like_player_tag(name: str) -> bool:
    """True when name matches TEAM.PLAYER style (e.g. ARZ.KUNAL10)."""
    s = (name or "").strip()
    return bool(s) and _PLAYER_TAG_RE.match(s) is not None


def names_look_like_killfeed(
    killer: str,
    victim: str,
    event_type: str = "",
) -> bool:
    """Require real player tags — blocks team codes (GG, IQOG) and counts (88)."""
    k = (killer or "").strip()
    v = (victim or "").strip()
    if not k or not v:
        return False
    if k.upper() == v.upper():
        return False
    if _is_numeric_token(k) or _is_numeric_token(v):
        return False

    et = (event_type or "").lower()
    if et in ("playzone", "powerzone"):
        return len(v) >= 2 and not _is_numeric_token(v)

    if et == "revive":
        return _PLAYER_TAG_RE.match(k) is not None and _PLAYER_TAG_RE.match(v) is not None

    return _PLAYER_TAG_RE.match(k) is not None and _PLAYER_TAG_RE.match(v) is not None

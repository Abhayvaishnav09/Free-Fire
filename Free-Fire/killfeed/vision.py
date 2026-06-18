"""Victim text color and revive detection for event classification."""

from __future__ import annotations

import cv2
import numpy as np


def victim_name_region(cropped_image: np.ndarray) -> np.ndarray:
    """Victim name band only — excludes team badges on both sides of the name."""
    if cropped_image is None or cropped_image.size == 0:
        return cropped_image
    h, w = cropped_image.shape[:2]
    # 52–86%: covers full victim name on wide rows; badges sit at ~0–15% and 88–100%.
    return cropped_image[:, int(w * 0.52) : int(w * 0.86)]


def _victim_color_ratios(cropped_image: np.ndarray) -> tuple[float, float, float]:
    """Return (white_ratio, red_ratio, green_ratio) in victim name band."""
    region = victim_name_region(cropped_image)
    if region is None or region.size == 0:
        return 0.0, 0.0, 0.0
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255])),
        cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255])),
    )
    mask_white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
    # Widen to H 35-95: catches yellow-green and cyan-green variants used in Free Fire UI
    mask_green = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([95, 255, 255]))
    total = max(region.shape[0] * region.shape[1], 1)
    return (
        cv2.countNonZero(mask_white) / total,
        cv2.countNonZero(mask_red) / total,
        cv2.countNonZero(mask_green) / total,
    )


def is_revive_victim_text(cropped_image: np.ndarray) -> bool:
    """
    True when victim name band is green (revive text).
    Key invariant: revive text is GREEN while knock/kill text is WHITE/RED.
    Green must clearly dominate over red — white background ratio is ignored
    because the killfeed bar's background can have more white than the text.
    """
    if cropped_image is None or cropped_image.size == 0:
        return False
    white_r, red_r, green_r = _victim_color_ratios(cropped_image)
    # Strong kill signal (red victim text) overrides everything
    if red_r >= 0.25 and red_r > green_r:
        return False
    # Green victim NAME: green dominates BOTH white and red in the name band.
    # A kill/knock row has a WHITE victim name (white_r > green_r) so faint
    # green background bleed cannot trigger a false revive.  A real revive has
    # a vivid green name where green_r clearly exceeds white_r.
    if green_r >= 0.06 and green_r > white_r and green_r > red_r * 1.2:
        return True
    return False


def _band_color_ratios(region: np.ndarray) -> tuple[float, float]:
    """Return (white_ratio, red_ratio) for a BGR region."""
    if region is None or region.size == 0:
        return 0.0, 0.0
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255])),
        cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255])),
    )
    mask_white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
    total = region.shape[0] * region.shape[1]
    if total == 0:
        return 0.0, 0.0
    return cv2.countNonZero(mask_white) / total, cv2.countNonZero(mask_red) / total


def has_knock_status_icon(cropped_image: np.ndarray) -> bool:
    """
    Detect the white knocked-player silhouette on the left or right of the row.
    Kill rows use red victim text; knock rows keep white text plus this icon.
    """
    if cropped_image is None or cropped_image.size == 0:
        return False
    h, w = cropped_image.shape[:2]
    for x1_pct, x2_pct in ((0.0, 0.15), (0.82, 1.0)):
        region = cropped_image[:, int(w * x1_pct) : int(w * x2_pct)]
        if region.size == 0:
            continue
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        white_r, red_r = _band_color_ratios(region)
        total = max(region.shape[0] * region.shape[1], 1)
        edge_r = float(np.count_nonzero(edges)) / total
        if edge_r >= 0.095 and white_r >= 0.012 and red_r < 0.10:
            return True
    return False


def resolve_knock_vs_elimination(cropped_image: np.ndarray) -> str:
    """
    Decide knock vs kill from victim text color and optional knock-status icon.
    Returns 'knock', 'elimination', or 'unknown'.

    Victim text colour is the ground truth in Free Fire:
      - KILL  → victim name is RED/PINK
      - KNOCK → victim name is WHITE

    The left-side arrow and team-badge on every killfeed row are white with
    high edge density, which makes has_knock_status_icon() unreliable as a
    primary signal. Use it ONLY when red is very low (<0.10) so we never
    accidentally call a red-text kill a "knock".
    """
    if cropped_image is None or cropped_image.size == 0:
        return "unknown"

    if is_revive_victim_text(cropped_image):
        return "unknown"

    victim_region = victim_name_region(cropped_image)
    white_r, red_r = _band_color_ratios(victim_region)

    # Kill: victim name turns red/pink — three increasing-sensitivity thresholds
    if red_r >= 0.35:
        return "elimination"
    if red_r >= 0.20 and red_r > white_r * 1.5:
        return "elimination"
    # Medium red: victim text is red but sparse on dark background (small font, short name).
    # As long as red is at least 60% of white ratio this is unmistakably red text.
    if red_r >= 0.10 and red_r >= white_r * 0.6:
        return "elimination"

    # Knock: white victim text — only trust knock signals when red is negligible.
    # Threshold tightened from 0.18 → 0.10 so a kill with red_r=0.12 doesn't
    # get pulled into "knock" by the arrow/badge on the left edge.
    if has_knock_status_icon(cropped_image) and red_r < 0.10:
        return "knock"
    if white_r >= 0.025 and red_r < 0.10:
        return "knock"

    # Low-signal fallback
    if red_r >= 0.08:
        return "elimination"
    if white_r >= 0.008:
        return "knock"
    return "unknown"


def analyze_victim_text_color(cropped_image: np.ndarray) -> str:
    """Return 'red', 'white', 'green', or 'unknown' from victim name pixels only."""
    try:
        victim_region = victim_name_region(cropped_image)
        if victim_region is None or victim_region.size == 0:
            return "unknown"
        height, width = victim_region.shape[:2]
        if height < 10 or width < 10:
            return "unknown"

        white_ratio, red_ratio, green_ratio = _victim_color_ratios(cropped_image)

        # Revive name is GREEN.  Use the same discriminator as is_revive_victim_text:
        # green must dominate BOTH white and red.  A white victim name (kill/knock)
        # with faint green background bleed keeps white_ratio > green_ratio, so it
        # is never mislabelled "green".
        if red_ratio >= 0.25 and red_ratio > green_ratio:
            # Strong red kill signal — skip green check entirely
            return "red"
        if green_ratio >= 0.06 and green_ratio > white_ratio and green_ratio > red_ratio * 1.2:
            return "green"

        combat = resolve_knock_vs_elimination(cropped_image)
        if combat == "knock":
            return "white"
        if combat == "elimination":
            return "red"

        if white_ratio >= 0.008 and white_ratio >= red_ratio:
            return "white"
        if red_ratio >= 0.01 and red_ratio > white_ratio:
            return "red"
        return "unknown"
    except Exception:
        return "unknown"


def has_green_color(cropped_image: np.ndarray) -> bool:
    """Green anywhere in crop (legacy). Prefer has_green_victim_text for revive."""
    try:
        if cropped_image is None or cropped_image.size == 0:
            return False
        hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([40, 40, 40]), np.array([85, 255, 255]))
        total = cropped_image.shape[0] * cropped_image.shape[1]
        if total == 0:
            return False
        return (cv2.countNonZero(mask) / total) >= 0.005
    except Exception:
        return False


def has_green_victim_text(cropped_image: np.ndarray) -> bool:
    """True only when victim name band is predominantly green (revive text)."""
    return is_revive_victim_text(cropped_image)

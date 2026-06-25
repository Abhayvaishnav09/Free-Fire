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


def victim_name_letter_region(cropped_image: np.ndarray) -> np.ndarray:
    """Inner victim-name glyph zone — excludes bar background and team badges."""
    region = victim_name_region(cropped_image)
    if region is None or region.size == 0:
        return region
    h, w = region.shape[:2]
    if h < 4 or w < 4:
        return region
    y1 = int(h * 0.22)
    y2 = int(h * 0.78)
    x1 = int(w * 0.06)
    x2 = int(w * 0.94)
    return region[y1:y2, x1:x2]


def _zone_red_white_ratios(region: np.ndarray) -> tuple[float, float]:
    """Return (white_ratio, red_ratio) within a BGR sub-region."""
    if region is None or region.size == 0:
        return 0.0, 0.0
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 70, 70]), np.array([10, 255, 255])),
        cv2.inRange(hsv, np.array([170, 70, 70]), np.array([180, 255, 255])),
    )
    mask_white = cv2.inRange(hsv, np.array([0, 0, 185]), np.array([180, 35, 255]))
    total = max(region.shape[0] * region.shape[1], 1)
    return cv2.countNonZero(mask_white) / total, cv2.countNonZero(mask_red) / total


def _bright_text_mask(region: np.ndarray) -> np.ndarray:
    """Pixels that belong to victim-name glyphs (bright white or saturated colour)."""
    if region is None or region.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    bright = cv2.inRange(gray, 165, 255)
    sat_val = hsv[:, :, 1]
    val = hsv[:, :, 2]
    color_text = ((sat_val >= 55) & (val >= 90)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(bright, color_text)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _letter_color_ratios(cropped_image: np.ndarray) -> tuple[float, float, float]:
    """
    White / red / green ratios measured on bright text pixels only.
    Ignores red background bleed behind white knock names.
    """
    letters = victim_name_letter_region(cropped_image)
    if letters is None or letters.size == 0:
        return 0.0, 0.0, 0.0

    text_mask = _bright_text_mask(letters)
    text_pixels = cv2.countNonZero(text_mask)
    if text_pixels < 6:
        white_r, red_r = _zone_red_white_ratios(letters)
        return white_r, red_r, 0.0

    hsv = cv2.cvtColor(letters, cv2.COLOR_BGR2HSV)
    mask_red = cv2.bitwise_and(
        cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255])),
            cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255])),
        ),
        text_mask,
    )
    mask_white = cv2.bitwise_and(
        cv2.inRange(hsv, np.array([0, 0, 185]), np.array([180, 40, 255])),
        text_mask,
    )
    mask_green = cv2.bitwise_and(
        cv2.inRange(hsv, np.array([35, 50, 50]), np.array([95, 255, 255])),
        text_mask,
    )
    total = float(text_pixels)
    return (
        cv2.countNonZero(mask_white) / total,
        cv2.countNonZero(mask_red) / total,
        cv2.countNonZero(mask_green) / total,
    )


def is_knock_feed_row(cropped_image: np.ndarray) -> bool:
    """True when victim-name letters are predominantly white, not red."""
    if cropped_image is None or cropped_image.size == 0:
        return False
    white_r, red_r, green_r = _letter_color_ratios(cropped_image)
    if green_r >= 0.20 and green_r > white_r:
        return False
    if red_r >= 0.22 and red_r > white_r * 1.4:
        return False
    return white_r >= 0.06 and red_r < 0.14


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
    True when victim name letters are green (revive text).
    Uses letter pixels — not bar background — to avoid false revives.
    """
    if cropped_image is None or cropped_image.size == 0:
        return False
    white_r, red_r, green_r = _letter_color_ratios(cropped_image)
    if red_r >= 0.22 and red_r > green_r:
        return False
    if green_r >= 0.14 and green_r > white_r and green_r > red_r * 1.2:
        return True
    # Fallback for vivid green band (legacy path)
    _white, _red, green_band = _victim_color_ratios(cropped_image)
    return green_band >= 0.08 and green_band > _white and green_band > _red * 1.2


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
    Decide knock vs kill from victim-name LETTER colour (not bar background).

    Free Fire UI:
      KILL  → victim name glyphs are red/pink
      KNOCK → victim name glyphs are white/gray

    Small red background behind white knock text must NOT trigger a kill.
    """
    if cropped_image is None or cropped_image.size == 0:
        return "unknown"

    if is_revive_victim_text(cropped_image):
        return "unknown"

    white_r, red_r, green_r = _letter_color_ratios(cropped_image)

    # Kill — red must dominate the actual text pixels, not just the band.
    if red_r >= 0.38:
        return "elimination"
    if red_r >= 0.24 and red_r > white_r * 1.8:
        return "elimination"
    if red_r >= 0.18 and white_r < 0.04:
        return "elimination"

    # Knock — white text with negligible red in letters.
    if is_knock_feed_row(cropped_image):
        return "knock"
    if white_r >= 0.05 and red_r < 0.12:
        return "knock"

    # Weak red in background only — prefer knock to avoid false kills.
    if red_r < 0.16:
        return "knock"
    if white_r >= red_r:
        return "knock"
    return "unknown"


def analyze_victim_text_color(cropped_image: np.ndarray) -> str:
    """Return 'red', 'white', 'green', or 'unknown' from victim letter pixels."""
    try:
        letters = victim_name_letter_region(cropped_image)
        if letters is None or letters.size == 0:
            return "unknown"
        height, width = letters.shape[:2]
        if height < 4 or width < 4:
            return "unknown"

        white_r, red_r, green_r = _letter_color_ratios(cropped_image)

        if green_r >= 0.18 and green_r > white_r and green_r > red_r * 1.2:
            return "green"
        if red_r >= 0.30 and red_r > green_r:
            return "red"

        combat = resolve_knock_vs_elimination(cropped_image)
        if combat == "knock":
            return "white"
        if combat == "elimination":
            return "red"

        if white_r >= 0.04 and white_r >= red_r:
            return "white"
        if red_r >= 0.22 and red_r > white_r * 1.5:
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

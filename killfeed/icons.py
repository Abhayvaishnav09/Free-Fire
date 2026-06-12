"""Fast center-icon analysis — runs on small crop only (OCR worker thread)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class IconCategory(str, Enum):
    GUN = "gun"
    HEADSHOT = "headshot"
    SKILL = "skill"
    GRENADE = "grenade"
    THROWABLE = "throwable"
    VEHICLE = "vehicle"
    PLAYZONE = "playzone"
    POWERZONE = "powerzone"
    REVIVE = "revive"
    UNKNOWN = "unknown"


@dataclass
class IconAnalysis:
    category: IconCategory
    confidence: float = 0.0
    has_gun_silhouette: bool = False
    has_headshot_marker: bool = False


def extract_icon_roi(row_crop: np.ndarray) -> np.ndarray:
    """Center weapon/icon band between killer and victim text."""
    if row_crop is None or row_crop.size == 0:
        return row_crop
    h, w = row_crop.shape[:2]
    y1, y2 = int(h * 0.15), int(h * 0.85)
    x1, x2 = int(w * 0.30), int(w * 0.55)
    return row_crop[y1:y2, x1:x2]


def analyze_icon(row_crop: np.ndarray) -> IconAnalysis:
    """
    Visual-only icon classification on In vision.py, I should remove the overly permissive 2% green condition entirely.

I'm also noticing the gun detection logic needs tightening — with such a low red threshold, any weapon icon with slight background contamination gets misclassified as a headshot. I need to increase that red requirement significantly and ensure it's actually dominant over other colors.

For the revive victim detection, I'm removing the overly permissive condition and keeping only the stricter checks that require green to be substantially more prominent than red and white. I'll also need to review the revive detection logic in the classifier to make sure it's consistent.

~48x24 resized ROI (<2ms typical).
    Never reads OCR text.
    """
    roi = extract_icon_roi(row_crop)
    if roi is None or roi.size == 0:
        return IconAnalysis(IconCategory.UNKNOWN)

    small = cv2.resize(roi, (48, 24), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    total = max(h * w, 1)
    roi_h, roi_w = roi.shape[:2]
    roi_aspect = roi_w / max(roi_h, 1)

    edges = cv2.Canny(gray, 40, 120)
    edge_ratio = float(np.count_nonzero(edges)) / total

    red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255])),
        cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255])),
    )
    orange = cv2.inRange(hsv, np.array([8, 120, 120]), np.array([28, 255, 255]))
    yellow = cv2.inRange(hsv, np.array([22, 80, 150]), np.array([38, 255, 255]))
    green = cv2.inRange(hsv, np.array([40, 60, 60]), np.array([85, 255, 255]))
    purple = cv2.inRange(hsv, np.array([125, 60, 60]), np.array([165, 255, 255]))
    white = cv2.inRange(hsv, np.array([0, 0, 190]), np.array([180, 40, 255]))

    red_r = cv2.countNonZero(red) / total
    orange_r = cv2.countNonZero(orange) / total
    yellow_r = cv2.countNonZero(yellow) / total
    green_r = cv2.countNonZero(green) / total
    purple_r = cv2.countNonZero(purple) / total
    white_r = cv2.countNonZero(white) / total

    has_gun = edge_ratio >= 0.08 and white_r >= 0.05 and roi_aspect >= 1.2
    # Raised from 0.02 → 0.12 to avoid false headshot from badge/background bleed.
    # Also require red to clearly dominate orange and not be swamped by white.
    has_headshot = (
        red_r >= 0.12
        and red_r > orange_r * 1.5
        and red_r > white_r * 0.25
    )

    # Revive icon is green — raised threshold from 0.04 → 0.14 and require
    # green to be at least 2× red so that faint badge noise never triggers revive.
    if green_r >= 0.14 and green_r > red_r * 2.0:
        return IconAnalysis(IconCategory.REVIVE, 0.7, False, has_headshot)

    if (yellow_r + orange_r) >= 0.06 and not has_gun and edge_ratio < 0.07:
        if purple_r >= 0.03:
            return IconAnalysis(IconCategory.POWERZONE, 0.65, has_gun, has_headshot)
        return IconAnalysis(IconCategory.PLAYZONE, 0.7, has_gun, has_headshot)

    if orange_r >= 0.05 and edge_ratio < 0.10:
        if orange_r >= 0.08:
            return IconAnalysis(IconCategory.GRENADE, 0.72, has_gun, has_headshot)
        return IconAnalysis(IconCategory.THROWABLE, 0.6, has_gun, has_headshot)

    if purple_r >= 0.05 and not has_gun:
        return IconAnalysis(IconCategory.SKILL, 0.68, has_gun, has_headshot)

    if roi_aspect >= 3.5 and edge_ratio < 0.035 and white_r < 0.06 and orange_r >= 0.04:
        return IconAnalysis(IconCategory.VEHICLE, 0.62, has_gun, has_headshot)

    if has_headshot:
        return IconAnalysis(IconCategory.HEADSHOT, 0.75, has_gun, True)

    if has_gun:
        return IconAnalysis(IconCategory.GUN, 0.8, True, has_headshot)

    if edge_ratio >= 0.05:
        return IconAnalysis(IconCategory.GUN, 0.5, True, has_headshot)

    return IconAnalysis(IconCategory.UNKNOWN, 0.3, has_gun, has_headshot)

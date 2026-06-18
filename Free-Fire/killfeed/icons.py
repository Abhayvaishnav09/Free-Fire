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


def is_revive_heart_icon(row_crop: np.ndarray) -> bool:
    """
    Detect the Free Fire revive ❤️ icon in the center band.

    Two variants exist in the game UI:
      • Pink/magenta heart  (e.g. GGI.POWER → GGI.SWARUP)
      • White heart + EKG pulse line  (e.g. TT.KHONSHU → TT.KOWSIK24)

    Uses a tighter center crop than extract_icon_roi so the heart shape is
    isolated from killer/victim text edges (which otherwise inflate edge_ratio
    and roi_aspect and cause the icon to be misread as a gun).
    """
    if row_crop is None or row_crop.size == 0:
        return False
    h, w = row_crop.shape[:2]
    roi = row_crop[int(h * 0.20) : int(h * 0.80), int(w * 0.36) : int(w * 0.54)]
    if roi is None or roi.size == 0:
        return False

    small = cv2.resize(roi, (48, 24), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gh, gw = gray.shape
    total = max(gh * gw, 1)
    roi_h, roi_w = roi.shape[:2]
    roi_aspect = roi_w / max(roi_h, 1)

    edges = cv2.Canny(gray, 40, 120)
    edge_ratio = float(np.count_nonzero(edges)) / total

    red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 60, 60]), np.array([10, 255, 255])),
        cv2.inRange(hsv, np.array([170, 60, 60]), np.array([180, 255, 255])),
    )
    white = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 60, 255]))
    pink = cv2.inRange(hsv, np.array([140, 40, 80]), np.array([175, 255, 255]))

    red_r = cv2.countNonZero(red) / total
    white_r = cv2.countNonZero(white) / total
    pink_r = cv2.countNonZero(pink) / total

    # Pink/magenta heart
    if pink_r >= 0.04 and pink_r > red_r and edge_ratio < 0.28 and roi_aspect < 3.0:
        return True

    # White heart + EKG pulse line
    if (
        white_r >= 0.06
        and edge_ratio >= 0.08
        and edge_ratio <= 0.28
        and roi_aspect < 3.0
        and red_r < 0.12
    ):
        return True

    return False


def analyze_icon(row_crop: np.ndarray) -> IconAnalysis:
    """
    Visual-only icon classification on ~48x24 resized ROI (<2ms typical).
    Never reads OCR text.
    """
    roi = extract_icon_roi(row_crop)
    if roi is None or roi.size == 0:
        return IconAnalysis(IconCategory.UNKNOWN)

    # Revive heart must be detected BEFORE gun/edge heuristics — the white
    # heart + EKG icon has high edge density and was being misread as a gun.
    if is_revive_heart_icon(row_crop):
        return IconAnalysis(IconCategory.REVIVE, 0.78, False, False)

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
    # Pink/magenta range — covers the Free Fire revive ❤️ heart icon
    pink = cv2.inRange(hsv, np.array([140, 60, 100]), np.array([175, 255, 255]))

    red_r = cv2.countNonZero(red) / total
    orange_r = cv2.countNonZero(orange) / total
    yellow_r = cv2.countNonZero(yellow) / total
    green_r = cv2.countNonZero(green) / total
    purple_r = cv2.countNonZero(purple) / total
    white_r = cv2.countNonZero(white) / total
    pink_r = cv2.countNonZero(pink) / total

    has_gun = edge_ratio >= 0.08 and white_r >= 0.05 and roi_aspect >= 1.2
    has_headshot = red_r >= 0.02 and red_r > orange_r

    # Heart icon (revive ❤️) is pink/magenta — compact, no gun silhouette.
    # Sensitive enough to catch the small heart, but guarded so kill/knock rows
    # never match:
    #   • not has_gun        : no weapon silhouette (gun kills are excluded)
    #   • pink_r > red_r      : magenta heart, not a red death/headshot marker
    #   • pink_r > purple_r   : distinguishes the heart from a purple skill icon
    #   • edge_ratio < 0.14   : smooth heart shape (guns/skills are edgier)
    if (
        pink_r >= 0.06
        and not has_gun
        and edge_ratio < 0.14
        and pink_r > red_r
        and pink_r > purple_r
    ):
        return IconAnalysis(IconCategory.REVIVE, 0.72, False, False)

    # Revive icon is green — require a meaningful green signal to avoid badge bleed
    if green_r >= 0.10 and green_r > red_r:
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

    if edge_ratio >= 0.05 and not is_revive_heart_icon(row_crop):
        return IconAnalysis(IconCategory.GUN, 0.5, True, has_headshot)

    return IconAnalysis(IconCategory.UNKNOWN, 0.3, has_gun, has_headshot)

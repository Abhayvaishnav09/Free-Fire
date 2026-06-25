"""Full killfeed event classification — YOLO class from best (1).pt is authoritative."""

from __future__ import annotations

from dataclasses import dataclass

from killfeed.gun_classifier import classify_gun
from killfeed.yolo_classes import classify_from_yolo

_GUN_ICON_CATEGORIES = frozenset({"gun", "headshot"})


@dataclass
class ClassificationResult:
    event_type: str  # knock | elimination | revive
    kill_type: str  # normal | headshot | skill | grenade | throwable | vehicle | playzone | powerzone | self
    canonical: str  # e.g. normal_knock, headshot_elimination
    tms_status: str
    icon_category: str
    confident: bool
    gun_name: str = "unknown"  # specific gun label from template matching e.g. "AK47"


def classify_row(
    row_crop,
    killer: str = "",
    victim: str = "",
    yolo_class: str | None = None,
    yolo_conf: float = 0.0,
) -> ClassificationResult:
    """Status comes only from the YOLO row class (best (1).pt)."""
    result = classify_from_yolo(yolo_class, yolo_conf)
    if result is None:
        return ClassificationResult("", "", "", "", "unknown", False)

    if result.icon_category in _GUN_ICON_CATEGORIES and row_crop is not None:
        gun = classify_gun(row_crop)
        if gun and gun != "unknown":
            result.gun_name = gun

    return result


def is_confident_classification(result: ClassificationResult) -> bool:
    return result.confident and bool(result.canonical)

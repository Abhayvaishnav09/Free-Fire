"""Row status resolution from YOLO class only (best (1).pt)."""

from __future__ import annotations

from dataclasses import dataclass

from killfeed.yolo_classes import classify_from_yolo, lookup_spec


@dataclass
class RowStatus:
    kind: str  # revive | elimination | knock | unknown
    source: str = ""


def resolve_row_status(
    crop,
    yolo_class: str = "",
    model=None,
    revive_confidence: float = 0.28,
    yolo_lock=None,
    skip_yolo_rescan: bool = False,
) -> RowStatus:
    """Map the detected YOLO row class to a coarse status kind."""
    del crop, model, revive_confidence, yolo_lock, skip_yolo_rescan

    result = classify_from_yolo(yolo_class)
    if result is None or not result.confident:
        spec = lookup_spec(yolo_class)
        if spec is not None:
            return RowStatus(spec.event_type if spec.event_type != "revive" else "revive", "yolo_class_weak")
        return RowStatus("unknown", "no_yolo_class")

    if result.event_type == "revive":
        return RowStatus("revive", "yolo_class")
    if result.event_type == "elimination":
        return RowStatus("elimination", "yolo_class")
    return RowStatus("knock", "yolo_class")


def row_status_to_event_type(row_status: RowStatus) -> str:
    if row_status.kind == "elimination":
        return "elimination"
    if row_status.kind == "revive":
        return "revive"
    return "knock"

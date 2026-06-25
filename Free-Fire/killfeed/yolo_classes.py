"""YOLO class vocabulary for best (1).pt — single source of truth for row status."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from killfeed.tms_map import canonical_to_tms

# All row-detection classes emitted by best (1).pt (YOLO12m, 17 classes).
ALL_ROW_CLASSES = frozenset({
    "player-skill-kill",
    "player-skill-knock",
    "falling-knock",
    "grenade-kill",
    "grenade-knock",
    "gun-head-kill",
    "gun-head-knock",
    "gun-kill",
    "gun-knock",
    "killblock",
    "landmine-knock",
    "player-kill",
    "playzone-kill",
    "playzone-knock",
    "revive",
    "smoke-grenade-kill",
    "smoke-grenade-knock",
})

# Generic row box — prefer a specific class when boxes overlap.
GENERIC_CLASSES = frozenset({"killblock"})

# Higher wins when merging overlapping detections on the same row.
_CLASS_PRIORITY: dict[str, int] = {
    "killblock": 0,
    "falling-knock": 5,
    "gun-knock": 10,
    "gun-kill": 10,
    "player-kill": 10,
    "gun-head-knock": 12,
    "gun-head-kill": 12,
    "player-skill-knock": 14,
    "player-skill-kill": 14,
    "grenade-knock": 14,
    "grenade-kill": 14,
    "landmine-knock": 14,
    "smoke-grenade-knock": 14,
    "smoke-grenade-kill": 14,
    "playzone-knock": 14,
    "playzone-kill": 14,
    "revive": 20,
}


@dataclass(frozen=True)
class YoloClassSpec:
    event_type: str  # knock | elimination | revive
    kill_type: str
    canonical: str
    icon_category: str
    confident: bool = True


_YOLO_SPECS: dict[str, YoloClassSpec] = {
    "revive": YoloClassSpec("revive", "normal", "revive", "revive"),
    "gun-kill": YoloClassSpec("elimination", "normal", "normal_elimination", "gun"),
    "gun-knock": YoloClassSpec("knock", "normal", "normal_knock", "gun"),
    "player-kill": YoloClassSpec("elimination", "normal", "normal_elimination", "gun"),
    "gun-head-kill": YoloClassSpec("elimination", "headshot", "headshot_elimination", "headshot"),
    "gun-head-knock": YoloClassSpec("knock", "headshot", "headshot_knock", "headshot"),
    "player-skill-kill": YoloClassSpec("elimination", "skill", "skill_elimination", "skill"),
    "player-skill-knock": YoloClassSpec("knock", "skill", "skill_knock", "skill"),
    "grenade-kill": YoloClassSpec("elimination", "grenade", "grenade_elimination", "grenade"),
    "grenade-knock": YoloClassSpec("knock", "grenade", "grenade_knock", "grenade"),
    "smoke-grenade-kill": YoloClassSpec("elimination", "throwable", "throwable_elimination", "throwable"),
    "smoke-grenade-knock": YoloClassSpec("knock", "throwable", "throwable_knock", "throwable"),
    "playzone-kill": YoloClassSpec("elimination", "playzone", "playzone_elimination", "playzone"),
    "playzone-knock": YoloClassSpec("knock", "playzone", "playzone_knock", "playzone"),
    "falling-knock": YoloClassSpec("knock", "normal", "normal_knock", "gun"),
    "landmine-knock": YoloClassSpec("knock", "grenade", "grenade_knock", "grenade"),
    "killblock": YoloClassSpec("knock", "normal", "normal_knock", "unknown", confident=False),
}


def normalize_class_name(name: str | None) -> str:
    return (name or "").lower().strip().replace("_", "-")


def is_row_class(class_name: str | None) -> bool:
    return normalize_class_name(class_name) in ALL_ROW_CLASSES


def is_generic_class(class_name: str | None) -> bool:
    return normalize_class_name(class_name) in GENERIC_CLASSES


def class_priority(class_name: str | None) -> int:
    return _CLASS_PRIORITY.get(normalize_class_name(class_name), -1)


def min_confidence_for_class(class_name: str | None, default: float = 0.20) -> float:
    cls = normalize_class_name(class_name)
    if cls == "revive":
        return 0.15
    return default


def lookup_spec(yolo_class: str | None) -> Optional[YoloClassSpec]:
    return _YOLO_SPECS.get(normalize_class_name(yolo_class))


def classify_from_yolo(
    yolo_class: str | None,
    yolo_conf: float = 0.0,
) -> Optional["ClassificationResult"]:
    """Map a YOLO class label to a full ClassificationResult."""
    from killfeed.classifier import ClassificationResult

    spec = lookup_spec(yolo_class)
    if spec is None:
        return None

    # BYPASS MAPPING LAYER: Use raw YOLO class directly as the final status
    # This allows evaluating the model's native performance directly in the UI.
    tms = yolo_class if yolo_class else "unknown"

    # ── STATUS TRACE: YOLO class → spec → TMS ──
    print(
        f"🔬 STATUS_TRACE [yolo_classes] "
        f"YOLO_CLASS={yolo_class!r} → spec.canonical={spec.canonical!r} "
        f"→ TMS_STATUS (BYPASS)={tms!r} confident={spec.confident}"
    )

    if not spec.confident:
        return ClassificationResult(
            event_type=spec.event_type,
            kill_type=spec.kill_type,
            canonical=spec.canonical,
            tms_status=tms,
            icon_category=spec.icon_category,
            confident=False,
            gun_name="unknown",
        )
    return ClassificationResult(
        event_type=spec.event_type,
        kill_type=spec.kill_type,
        canonical=spec.canonical,
        tms_status=tms,
        icon_category=spec.icon_category,
        confident=True,
        gun_name="unknown",
    )


def yolo_event_label(yolo_class: str | None) -> str:
    """Map YOLO class to fusion observer label: kill | knock | revive | abstain."""
    spec = lookup_spec(yolo_class)
    if spec is None or not spec.confident:
        return "abstain"
    if spec.event_type == "revive":
        return "revive"
    if spec.event_type == "elimination":
        return "kill"
    if spec.event_type == "knock":
        return "knock"
    return "abstain"

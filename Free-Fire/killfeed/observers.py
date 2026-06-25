"""Per-channel killfeed observers.

Each observer inspects ONE evidence channel (YOLO class, center icon, victim
text colour, knocked-silhouette icon) and returns an :class:`Observation`.

Critical rule: an observer that is not sure must ABSTAIN (label == EVENT_ABSTAIN,
score 0.0). It must never fall back to a guessed event.  Turning weak evidence
into a confident label is exactly what produced random kill/knock/revive
assignments; fusion + abstention is the fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from killfeed import vision
from killfeed.icons import IconCategory, analyze_icon

# Event labels (string values are the canonical vocabulary used by fusion).
EVENT_KILL = "kill"
EVENT_KNOCK = "knock"
EVENT_REVIVE = "revive"
EVENT_ABSTAIN = "abstain"

# Channel identifiers.
CH_YOLO = "yolo"
CH_ICON = "icon"
CH_COLOR = "color"
CH_STATUS = "status"

# Per-channel reliability for the *event class* decision.  These are priors on
# how much each channel should be trusted, independent of its per-frame score.
CHANNEL_WEIGHT = {
    CH_YOLO: 0.45,
    CH_ICON: 0.35,
    CH_STATUS: 0.20,
    CH_COLOR: 0.18,
}

# Tuning constants for the colour observer (victim-name band).
_COLOR_MIN_COVERAGE = 0.04   # the dominant colour must cover >=4% of the band
_COLOR_MIN_MARGIN = 0.20     # and clearly beat the runner-up colour


@dataclass
class Observation:
    channel: str
    label: str           # EVENT_KILL | EVENT_KNOCK | EVENT_REVIVE | EVENT_ABSTAIN
    score: float         # calibrated confidence in [0, 1]; 0 when abstaining
    detail: dict = field(default_factory=dict)

    @property
    def abstained(self) -> bool:
        return self.label == EVENT_ABSTAIN or self.score <= 0.0


def _abstain(channel: str, reason: str = "", **detail) -> Observation:
    detail["reason"] = reason
    return Observation(channel, EVENT_ABSTAIN, 0.0, detail)


def yolo_observer(yolo_class: str | None, yolo_conf: float) -> Observation:
    """best (1).pt row class is authoritative for kill / knock / revive."""
    from killfeed.yolo_classes import yolo_event_label

    cls = (yolo_class or "").lower()
    label = yolo_event_label(cls)
    if label == "abstain":
        return _abstain(CH_YOLO, "unmapped_yolo_class" if cls else "no_yolo_class")
    score = max(0.62, min(1.0, yolo_conf)) if yolo_conf > 0 else 0.72
    return Observation(CH_YOLO, label, score, {"yolo_conf": yolo_conf, "yolo_class": cls})


def icon_observer(row_crop) -> Observation:
    """Center glyph.  Reliable for REVIVE (heart); weapon glyphs cannot tell
    kill from knock, so they abstain on the kill/knock distinction."""
    icon = analyze_icon(row_crop)
    if icon.category == IconCategory.REVIVE:
        return Observation(
            CH_ICON, EVENT_REVIVE, max(0.6, icon.confidence), {"icon": icon.category.value}
        )
    # Weapon / zone icons do not discriminate kill vs knock.
    return _abstain(CH_ICON, "weapon_icon_not_event_discriminative", icon=icon.category.value)


def color_observer(row_crop) -> Observation:
    """Victim-name colour via letter-zone analysis (not whole-band bleed)."""
    if vision.is_revive_victim_text(row_crop):
        return Observation(
            CH_COLOR, EVENT_REVIVE, 0.78, {"source": "green_victim_text"}
        )

    letters = vision.victim_name_letter_region(row_crop)
    name_white, name_red = vision._zone_red_white_ratios(letters)
    if name_red >= 0.16 and name_red > name_white * 1.2:
        score = min(1.0, 0.50 + name_red * 2.2)
        return Observation(
            CH_COLOR,
            EVENT_KILL,
            score,
            {"white": name_white, "red": name_red, "zone": "letters"},
        )

    if vision.is_knock_feed_row(row_crop):
        return Observation(CH_COLOR, EVENT_KNOCK, 0.72, {"knock_feed_row": True})

    combat = vision.resolve_knock_vs_elimination(row_crop)
    if combat == "elimination":
        return Observation(CH_COLOR, EVENT_KILL, 0.62, {"combat": combat})
    if combat == "knock":
        return Observation(CH_COLOR, EVENT_KNOCK, 0.62, {"combat": combat})

    white_r, red_r, green_r = vision._victim_color_ratios(row_crop)
    detail = {"white": white_r, "red": red_r, "green": green_r}
    if green_r >= 0.06 and green_r > white_r and green_r > red_r * 1.2:
        return Observation(CH_COLOR, EVENT_REVIVE, 0.55, detail)
    return _abstain(CH_COLOR, "ambiguous_colour", **detail)


def status_icon_observer(row_crop) -> Observation:
    """White knocked-player silhouette on the row edges corroborates a KNOCK,
    but only when red (kill) is not present.  Weak supporting signal."""
    if vision.has_knock_status_icon(row_crop):
        _white_r, red_r, _green_r = vision._victim_color_ratios(row_crop)
        if red_r < 0.10:
            return Observation(CH_STATUS, EVENT_KNOCK, 0.55, {"knock_silhouette": True})
    return _abstain(CH_STATUS, "no_knock_silhouette")


def collect_observations(
    row_crop,
    yolo_class: str | None = None,
    yolo_conf: float = 0.0,
) -> List[Observation]:
    """YOLO-only — best (1).pt class is the sole status signal."""
    del row_crop
    try:
        return [yolo_observer(yolo_class, yolo_conf)]
    except Exception as exc:
        return [_abstain("error", type(exc).__name__)]

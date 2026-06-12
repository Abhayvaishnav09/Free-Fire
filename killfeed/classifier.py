"""Full killfeed event classification — visual cues only, no OCR status text."""

from __future__ import annotations

from dataclasses import dataclass

from killfeed import vision
from killfeed.icons import IconAnalysis, IconCategory, analyze_icon
from killfeed.tms_map import canonical_to_tms

try:
    from rapidfuzz import fuzz

    _HAS_FUZZ = True
except ImportError:
    _HAS_FUZZ = False

_COMBAT_ICONS = frozenset({
    IconCategory.GUN,
    IconCategory.HEADSHOT,
    IconCategory.GRENADE,
    IconCategory.THROWABLE,
    IconCategory.VEHICLE,
    IconCategory.SKILL,
})


@dataclass
class ClassificationResult:
    event_type: str  # knock | elimination | revive
    kill_type: str  # normal | headshot | skill | grenade | throwable | vehicle | playzone | powerzone | self
    canonical: str  # e.g. normal_knock, headshot_elimination
    tms_status: str
    icon_category: str
    confident: bool


def _is_self_kill(killer: str, victim: str) -> bool:
    if not killer or not victim:
        return False
    if killer.upper().strip() == victim.upper().strip():
        return True
    if _HAS_FUZZ and fuzz.ratio(killer.upper(), victim.upper()) >= 95:
        return True
    return False


def _has_attacker(killer: str) -> bool:
    k = (killer or "").strip()
    return len(k) >= 2 and k.upper() not in ("UNKNOWN", "ENV", "ZONE")


def _looks_like_combat(
    row_crop,
    icon: IconAnalysis,
    victim_color: str,
    has_killer: bool,
    has_victim: bool,
) -> bool:
    """Gun/knock/kill row — never classify as revive."""
    if vision.is_revive_victim_text(row_crop):
        return False
    if not has_killer or not has_victim:
        return False
    if icon.category == IconCategory.REVIVE:
        return False
    if icon.has_gun_silhouette or icon.category in _COMBAT_ICONS:
        return True
    if victim_color in ("red", "white"):
        return True
    return False


def _looks_like_revive(
    row_crop,
    icon: IconAnalysis,
    victim_color: str,
    yolo_class: str,
    has_killer: bool,
    has_victim: bool,
) -> bool:
    """
    Revive: green victim text, or YOLO revive + green icon (no strong red kill signal).
    """
    if not has_killer or not has_victim:
        return False

    white_r, red_r, green_r = vision._victim_color_ratios(row_crop)

    # Strong elimination on victim — never revive
    if red_r >= 0.35 and red_r > green_r:
        return False
    if victim_color == "red" and red_r >= 0.20:
        return False

    if vision.is_revive_victim_text(row_crop):
        return True

    # YOLO revive + green icon/crop when victim name band is unclear (OCR UNKNOWN crops).
    # has_green_color fires at only 0.5 % green — too loose.  Use the stricter
    # is_revive_victim_text (requires ≥12 % green) to avoid false revives.
    if yolo_class == "revive" and red_r < 0.18:
        if icon.category == IconCategory.REVIVE or vision.is_revive_victim_text(row_crop):
            return True
    return False


def _icon_to_kill_type(icon: IconAnalysis) -> str:
    mapping = {
        IconCategory.GUN: "normal",
        IconCategory.HEADSHOT: "headshot",
        IconCategory.SKILL: "skill",
        IconCategory.GRENADE: "grenade",
        IconCategory.THROWABLE: "throwable",
        IconCategory.VEHICLE: "vehicle",
        IconCategory.PLAYZONE: "playzone",
        IconCategory.POWERZONE: "powerzone",
        IconCategory.REVIVE: "normal",
        IconCategory.UNKNOWN: "normal",
    }
    return mapping.get(icon.category, "normal")


def _build_canonical(event_type: str, kill_type: str) -> str:
    if event_type == "revive":
        return "revive"
    return f"{kill_type}_{event_type}"


def classify_row(
    row_crop,
    killer: str = "",
    victim: str = "",
    yolo_class: str | None = None,
) -> ClassificationResult:
    """
    Priority:
      1. Revive when victim name band is green (beats false gun-icon / YOLO mislabel)
      2. Knock vs elimination from victim color + icon
      3. Environment / self damage
    """
    yolo_class = (yolo_class or "").lower()
    icon = analyze_icon(row_crop)
    victim_color = vision.analyze_victim_text_color(row_crop)
    has_killer = _has_attacker(killer)
    has_victim = bool(victim and len(victim.strip()) >= 2)

    # Revive first — green victim text beats false gun-icon / knock heuristics
    if has_killer and has_victim and _looks_like_revive(
        row_crop, icon, victim_color, yolo_class, has_killer, has_victim
    ):
        return ClassificationResult(
            event_type="revive",
            kill_type="normal",
            canonical="revive",
            tms_status="revive",
            icon_category=icon.category.value,
            confident=True,
        )

    # --- Environment: no attacker ---
    if not has_killer and has_victim:
        if icon.category == IconCategory.POWERZONE:
            return ClassificationResult(
                event_type="elimination",
                kill_type="powerzone",
                canonical="powerzone_elimination",
                tms_status=canonical_to_tms("powerzone_elimination"),
                icon_category=icon.category.value,
                confident=True,
            )
        if icon.category in (IconCategory.PLAYZONE, IconCategory.UNKNOWN):
            return ClassificationResult(
                event_type="elimination",
                kill_type="playzone",
                canonical="playzone_elimination",
                tms_status=canonical_to_tms("playzone_elimination"),
                icon_category=icon.category.value,
                confident=victim_color in ("red", "white", "unknown"),
            )

    if not has_victim:
        return ClassificationResult("", "", "", "", icon.category.value, False)

    # --- Self damage ---
    if has_killer and _is_self_kill(killer, victim):
        if victim_color == "red":
            return ClassificationResult(
                event_type="elimination",
                kill_type="self",
                canonical="self_elimination",
                tms_status=canonical_to_tms("self_elimination"),
                icon_category=icon.category.value,
                confident=True,
            )
        return ClassificationResult(
            event_type="knock",
            kill_type="self",
            canonical="self_knock",
            tms_status=canonical_to_tms("self_knock"),
            icon_category=icon.category.value,
            confident=True,
        )

    # --- Knock vs elimination: victim text color + knock-status icon ---
    combat = vision.resolve_knock_vs_elimination(row_crop)
    if combat == "elimination":
        base_event = "elimination"
    elif combat == "knock":
        base_event = "knock"
    elif victim_color == "red":
        base_event = "elimination"
    elif victim_color == "white":
        base_event = "knock"
    elif victim_color == "unknown" and _looks_like_combat(
        row_crop, icon, "white", has_killer, has_victim
    ):
        # Gun visible but color unclear → default knock (not revive)
        base_event = "knock"
    else:
        return ClassificationResult("", "", "", "", icon.category.value, False)

    kill_type = _icon_to_kill_type(icon)

    if icon.category == IconCategory.HEADSHOT or icon.has_headshot_marker:
        kill_type = "headshot"
    elif icon.category == IconCategory.GRENADE:
        kill_type = "grenade"
    elif icon.category == IconCategory.THROWABLE:
        kill_type = "throwable"
    elif icon.category == IconCategory.VEHICLE:
        kill_type = "vehicle"
    elif icon.category == IconCategory.SKILL:
        kill_type = "skill"

    canonical = _build_canonical(base_event, kill_type)
    return ClassificationResult(
        event_type=base_event,
        kill_type=kill_type,
        canonical=canonical,
        tms_status=canonical_to_tms(canonical),
        icon_category=icon.category.value,
        confident=True,
    )


def is_confident_classification(result: ClassificationResult) -> bool:
    return result.confident and bool(result.canonical)

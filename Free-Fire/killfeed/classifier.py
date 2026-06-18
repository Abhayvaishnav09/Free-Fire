"""Full killfeed event classification — YOLO revive first, then victim text color."""

from __future__ import annotations

from dataclasses import dataclass

from killfeed import vision
from killfeed.gun_classifier import classify_gun
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
    gun_name: str = "unknown"  # specific gun label from template matching e.g. "AK47"


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


def _revive_result() -> ClassificationResult:
    return ClassificationResult(
        event_type="revive",
        kill_type="normal",
        canonical="revive",
        tms_status="revive",
        icon_category="revive",
        confident=True,
    )


def _combat_from_victim_color(row_crop, victim_color: str) -> str:
    """
    Knock vs kill from victim name colour only (after revive ruled out).

    WHITE victim text → knock
    RED victim text   → kill (elimination)
    """
    if victim_color == "red":
        return "elimination"
    if victim_color == "white":
        return "knock"

    # Direct band analysis — ignore background elsewhere in the row.
    white_r, red_r, _green_r = vision._victim_color_ratios(row_crop)
    if red_r >= 0.10 and red_r >= white_r * 0.5:
        return "elimination"
    if white_r >= 0.008 and red_r < 0.10:
        return "knock"

    combat = vision.resolve_knock_vs_elimination(row_crop)
    if combat in ("knock", "elimination"):
        return combat
    return "knock"


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
    Status priority (exact order):

    1. REVIVE — best.pt YOLO class ``revive`` on this row is authoritative.
       Stop immediately.  Do NOT inspect victim colour or combat icons.

    2. KILL / KNOCK — only when YOLO did NOT say revive.
       Victim name colour only: white → knock, red → kill.
    """
    yolo_class = (yolo_class or "").lower()

    # ── 1. REVIVE (highest priority) ─────────────────────────────────────
    if yolo_class == "revive":
        return _revive_result()

    icon = analyze_icon(row_crop)
    victim_color = vision.analyze_victim_text_color(row_crop)
    has_killer = _has_attacker(killer)
    has_victim = bool(victim and len(victim.strip()) >= 2)

    # ── Environment: no attacker ─────────────────────────────────────────
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

    # ── Self damage ──────────────────────────────────────────────────────
    if has_killer and _is_self_kill(killer, victim):
        if victim_color == "red" or _combat_from_victim_color(row_crop, victim_color) == "elimination":
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

    # ── 2. KILL / KNOCK — victim text colour only ────────────────────────
    base_event = _combat_from_victim_color(row_crop, victim_color)

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
    gun_name = (
        classify_gun(row_crop)
        if icon.category in (IconCategory.GUN, IconCategory.HEADSHOT)
        else "unknown"
    )
    return ClassificationResult(
        event_type=base_event,
        kill_type=kill_type,
        canonical=canonical,
        tms_status=canonical_to_tms(canonical),
        icon_category=icon.category.value,
        confident=True,
        gun_name=gun_name,
    )


def is_confident_classification(result: ClassificationResult) -> bool:
    return result.confident and bool(result.canonical)

"""Map canonical event types to TMS WeaponUsed strings.

Updated to use the actual best (1).pt YOLO model class names instead of
legacy knife/pan names that don't exist in the model.
"""

from __future__ import annotations

# canonical_id -> TMS API status (WeaponUsed field)
# Values MUST match the vocabulary the TMS frontend/backend expects.
# All legacy "knife-*" and "pan-*" names have been replaced with the
# corresponding best (1).pt class names.
CANONICAL_TO_TMS: dict[str, str] = {
    "revive": "revived",
    "normal_knock": "gun knockout",
    "normal_elimination": "kill",
    "headshot_knock": "gun-head-knock",
    "headshot_elimination": "gun-head-kill",
    "skill_knock": "player-skill-knock",
    "skill_elimination": "player-skill-kill",
    "grenade_knock": "grenade-knock",
    "grenade_elimination": "grenade-kill",
    "throwable_knock": "smoke-grenade-knock",
    "throwable_elimination": "smoke-grenade-kill",
    "vehicle_knock": "car-knockout",
    "vehicle_elimination": "car-kill",
    "playzone_elimination": "playzone-kill",
    "playzone_knock": "playzone-knock",
    "powerzone_elimination": "playzone-kill",
    "self_knock": "self-knockout",
    "self_elimination": "self-kill",
}


def canonical_to_tms(canonical: str) -> str:
    return CANONICAL_TO_TMS.get(canonical, "kill" if "elimination" in canonical else "gun knockout")

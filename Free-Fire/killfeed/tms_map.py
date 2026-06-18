"""Map canonical event types to TMS WeaponUsed strings."""

from __future__ import annotations

# canonical_id -> TMS API status (WeaponUsed field)
CANONICAL_TO_TMS: dict[str, str] = {
    "revive": "revive",
    "normal_knock": "gun knockout",
    "normal_elimination": "kill",
    "headshot_knock": "knife-head-knockout",
    "headshot_elimination": "knife-head-kill",
    "skill_knock": "gun knockout",
    "skill_elimination": "kill",
    "grenade_knock": "grenade-knockout",
    "grenade_elimination": "grenade-kill",
    "throwable_knock": "grenade-knockout",
    "throwable_elimination": "grenade-kill",
    "vehicle_knock": "car-knockout",
    "vehicle_elimination": "car-kill",
    "playzone_elimination": "playzone-kill",
    "powerzone_elimination": "playzone-kill",
    "self_knock": "self-knockout",
    "self_elimination": "self-kill",
}


def canonical_to_tms(canonical: str) -> str:
    return CANONICAL_TO_TMS.get(canonical, "kill" if "elimination" in canonical else "gun knockout")

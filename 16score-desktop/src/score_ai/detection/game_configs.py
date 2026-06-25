"""
Game-specific configuration for killfeed detection.

BGMI (Battlegrounds Mobile India) and FreeFire have different:
- Detection class names (kill-block vs kill/gun knockout)
- OCR text formats (Killer killed Victim vs Killer Victim)
- Weapon detection (weapon icons vs detection class as weapon)
"""

from typing import Optional

# Game identifiers (must match config "detection.game")
GAME_BGMI = "bgmi"
GAME_FREEFIRE = "freefire"


def get_game_config(game: Optional[str]) -> dict:
    """
    Get game-specific configuration.
    
    Args:
        game: Game identifier from config (e.g. "freefire", "bgmi")
        
    Returns:
        dict with keys: killfeed_classes, class_id, use_detection_as_weapon,
                        combined_text_separators, combined_text_space_fallback
    """
    game = (game or "").lower().strip()
    
    if game == GAME_FREEFIRE:
        return _FREEFIRE_CONFIG
    if game in (GAME_BGMI, "pubg", "pubgm"):
        return _BGMI_CONFIG
    # Default to BGMI for backwards compatibility
    return _BGMI_CONFIG


# BGMI (Battlegrounds Mobile India / PUBG Mobile)
# - Uses kill-block detection (class_id 12)
# - Has weapon icons detected separately
# - OCR: "Killer killed Victim" or "Killer knocked Victim"
_BGMI_CONFIG = {
    "killfeed_classes": ["kill-block"],
    "killfeed_class_id": 12,
    "use_detection_class_as_weapon": False,  # BGMI has weapon icons
    "combined_text_separators": [
        " knocked out ",  # longer phrases first; matching code also sorts by len
        " killed ",
        " knocked ",
        " knockout ",
        " revived ",
        " revives ",
        " revive ",
        " eliminates ",
        " -> ",
        " - ",
    ],
    "combined_text_space_fallback": False,  # BGMI uses "killed"/"knocked"
    "grpc_game_param": "bgmi",
    "use_victim_name_color_event_type": False,
}


# FreeFire
# - Server may emit BGMI-style "kill-block" (names in OCR) and/or legacy "kill" / "gun knockout"
# - Weapon-only rows (e.g. gun-kill with empty OCR) stay in other_objects for weapon resolution
# - OCR: "Killer Victim" (two space-separated names, no keyword)
_FREEFIRE_CONFIG = {
    "killfeed_classes": [
        "kill-block",
        "kill",
        "gun knockout",
        "gun-knockout",
        "revived",
    ],
    "killfeed_class_id": None,  # No fixed class_id for FreeFire
    "use_detection_class_as_weapon": True,  # No weapon icons
    "combined_text_separators": [
        " knocked out ",
        " gun-kill ",
        " gun kill ",
        " gunkill ",
        " killed ",
        " knocked ",
        " knockout ",
        " revived ",
        " revives ",
        " revive ",
        " eliminates ",
        " -> ",
        " - ",
    ],
    "combined_text_space_fallback": True,  # "Name1 Name2" format
    "grpc_game_param": "freefire",
    # Victim name pixel color in killfeed region: green=revived, red=kill, white=knockout
    "use_victim_name_color_event_type": True,
}


def is_killfeed_detection(
    class_id: Optional[int],
    class_name: Optional[str],
    game: Optional[str],
) -> bool:
    """
    Check if detection is a killfeed container (e.g. kill-block for BGMI).
    Weapon classes like gun-kill, gun-knockout, gun-head-kill, gun-head-knockout
    should NOT match here -- they go into other_objects so extract_kill_info
    can pick them up as the actual weapon type.
    """
    cfg = get_game_config(game)
    if class_id is not None and cfg["killfeed_class_id"] is not None:
        if class_id == cfg["killfeed_class_id"]:
            return True
    if class_name:
        cn_lower = class_name.lower()
        for kc in cfg["killfeed_classes"]:
            if kc.lower() == cn_lower:
                return True
    return False

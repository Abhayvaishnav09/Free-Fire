"""Event validation before scoring.

A light, pixel-independent sanity gate that runs after classification/resolution
but before the per-victim FSM and emission.  Game-logic ordering (knock→kill) is
enforced by :mod:`killfeed.fsm`; this module rejects structurally invalid events
so they are never scored.
"""

from __future__ import annotations

from dataclasses import dataclass

# Must stay at or below classifier TAU_EMIT — a higher floor here silently
# blocks events that already passed classification and never reaches TMS.
_MIN_FUSED_CONF = 0.28

_KNOWN_EVENT_TYPES = {"elimination", "knock", "revive", "playzone", "powerzone"}


@dataclass
class ValidationResult:
    accept: bool
    reason: str = ""


def validate_event(
    killer: str,
    victim: str,
    event_type: str,
    ocr_conf: float,
    fused_conf: float,
) -> ValidationResult:
    if not victim or len(victim.strip()) < 2:
        return ValidationResult(False, "missing_victim")

    if event_type == "unknown" or not event_type:
        return ValidationResult(False, "unknown_event_type")

    # Map self_* / *_elimination style upstream types loosely: accept the base.
    base = event_type
    if base not in _KNOWN_EVENT_TYPES and base not in ("self",):
        # Unrecognised event type — refuse rather than score something unknown.
        return ValidationResult(False, f"unrecognised_event_type:{event_type}")

    if event_type == "revive" and killer and victim:
        if killer.strip().upper() == victim.strip().upper():
            return ValidationResult(False, "self_revive_impossible")

    if fused_conf < _MIN_FUSED_CONF:
        return ValidationResult(False, "fused_conf_below_floor")

    return ValidationResult(True, "ok")

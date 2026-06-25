"""Per-victim state machine: ALIVE → KNOCKED → KILLED, with revive returning to
ALIVE.  This mechanically enforces the scoring invariant: a KILL is never emitted
before its KNOCK.

Two methods, matching the pipeline's two-phase emit:
  * ``apply_and_commit`` — pure DECISION: given the proposed label and its fused
    confidence, return (synthetic_labels, decision).  It does NOT mutate state.
  * ``commit`` — applies the actual state transition for a row that really got
    emitted (called once per emitted row, including synthesized knocks).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from killfeed.observers import EVENT_KILL, EVENT_KNOCK, EVENT_REVIVE

try:
    from rapidfuzz import fuzz

    _HAS_FUZZ = True
except ImportError:
    _HAS_FUZZ = False

# Match classifier TAU_EMIT — if classification already passed, allow direct
# kill with a synthesized knock (ordering preserved for TMS scoring).
TAU_DIRECT_KILL = 0.30

ALIVE = "ALIVE"
KNOCKED = "KNOCKED"
KILLED = "KILLED"


@dataclass
class Decision:
    accept: bool
    reason: str = ""


def _label_from_event_type(event_type: str) -> str:
    if event_type == "elimination":
        return EVENT_KILL
    if event_type == "knock":
        return EVENT_KNOCK
    if event_type == "revive":
        return EVENT_REVIVE
    return event_type


def _canon(name: str) -> str:
    return (name or "").strip().upper()


def _find_victim_key(state: Dict[str, str], victim: str) -> str:
    key = _canon(victim)
    if key in state:
        return key
    if _HAS_FUZZ:
        for existing in state:
            if fuzz.ratio(key, existing) >= 80.0:
                return existing
    return key


class VictimFSM:
    def __init__(self):
        self._state: Dict[str, str] = {}

    def state_of(self, victim: str) -> str:
        return self._state.get(_find_victim_key(self._state, victim), ALIVE)

    def was_knocked(self, victim: str) -> bool:
        return self.state_of(victim) in (KNOCKED, KILLED)

    def is_knocked(self, victim: str) -> bool:
        return self.state_of(victim) == KNOCKED

    def apply_and_commit(
        self, victim: str, label: str, fused_confidence: float
    ) -> Tuple[List[str], Decision]:
        """Decide whether ``label`` is legal for ``victim`` right now.

        Returns ``(synthetics, decision)`` where ``synthetics`` is a list of
        labels (currently at most ``[EVENT_KNOCK]``) that must be emitted *before*
        the primary event to preserve ordering.
        """
        state = self.state_of(victim)

        if label == EVENT_KNOCK:
            if state == ALIVE:
                return [], Decision(True)
            return [], Decision(False, "duplicate_knock")

        if label == EVENT_REVIVE:
            if state == KNOCKED:
                return [], Decision(True)
            return [], Decision(False, "revive_without_knock")

        if label == EVENT_KILL:
            if state == KNOCKED:
                return [], Decision(True)
            if state == ALIVE:
                if fused_confidence >= TAU_DIRECT_KILL:
                    # Synthesize the missing knock first, in order.
                    return [EVENT_KNOCK], Decision(True)
                return [], Decision(False, "kill_before_knock_low_conf")
            return [], Decision(False, "duplicate_kill")

        return [], Decision(False, "unknown_label")

    def commit(self, victim: str, event_type: str) -> None:
        """Apply the real state transition for an emitted row."""
        key = _find_victim_key(self._state, victim)
        label = _label_from_event_type(event_type)
        if label == EVENT_KNOCK:
            self._state[key] = KNOCKED
        elif label == EVENT_KILL:
            self._state[key] = KILLED
        elif label == EVENT_REVIVE:
            self._state[key] = ALIVE

    def reset(self) -> None:
        """Clear all victim state (call at match / round boundaries)."""
        self._state.clear()

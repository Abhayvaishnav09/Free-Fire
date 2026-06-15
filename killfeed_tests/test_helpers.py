"""Shared helpers for killfeed ground-truth tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple

try:
    from rapidfuzz import fuzz

    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

from killfeed.queue_state import make_signature, parse_signature

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ROOT = FIXTURES.parents[1]

EVENT_CANONICAL = {
    "knock": "normal_knock",
    "elimination": "normal_elimination",
    "revive": "revive",
}


def load_ground_truth(name: str = "stream_ground_truth_burst.json") -> list[dict]:
    with open(FIXTURES / name, encoding="utf-8") as f:
        return json.load(f)


def event_key(killer: str, victim: str, event_type: str) -> Tuple[str, str, str]:
    return (killer.upper().strip(), victim.upper().strip(), event_type.lower().strip())


def ground_truth_to_signature(row: dict) -> str:
    canonical = EVENT_CANONICAL.get(row["event_type"], row["event_type"])
    return make_signature(row["killer"], row["victim"], canonical, "gun")


def visible_snapshot(
    events: list[dict], event_index: int, max_rows: int = 4
) -> List[str]:
    """On-screen killfeed: newest at index 0, up to max_rows."""
    start = max(0, event_index - max_rows + 1)
    visible = events[start : event_index + 1]
    visible.reverse()
    return [ground_truth_to_signature(r) for r in visible]


def fuzzy_name_match(a: str, b: str, threshold: float = 82.0) -> bool:
    if not a or not b:
        return False
    au, bu = a.upper(), b.upper()
    if au == bu:
        return True
    if HAS_RAPIDFUZZ:
        if fuzz.ratio(au, bu) >= threshold:
            return True
        if fuzz.partial_ratio(au, bu) >= 88:
            return True
    return False


def events_match(a: dict, b: dict) -> bool:
    """Same killer+victim+event_type with OCR-tolerant names."""
    if a.get("event_type", "").lower() != b.get("event_type", "").lower():
        et_a = a.get("event_type", "").lower()
        et_b = b.get("event_type", "").lower()
        if et_a != et_b:
            return False
    return fuzzy_name_match(a.get("killer", ""), b.get("killer", "")) and fuzzy_name_match(
        a.get("victim", ""), b.get("victim", "")
    )


def normalize_detected(row: dict) -> dict:
    """Map jsonl / pipeline event to ground-truth shape."""
    event = row.get("event", "")
    event_type = row.get("event_type", "")
    if not event_type:
        el = event.lower()
        if "revive" in el:
            event_type = "revive"
        elif "knock" in el:
            event_type = "knock"
        elif "elimination" in el or el in ("kill", "elimination"):
            event_type = "elimination"
        else:
            event_type = el
    return {
        "killer": row.get("killer", ""),
        "victim": row.get("victim", ""),
        "event_type": event_type,
    }


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_in_detected(target: dict, detected: Iterable[dict]) -> bool:
    t = normalize_detected(target) if "event_type" in target else target
    for d in detected:
        if events_match(t, normalize_detected(d)):
            return True
    return False


def coverage_report(
    ground_truth: list[dict], detected: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Return (found, missing) comparing stream truth to detected jsonl."""
    found = []
    missing = []
    for gt in ground_truth:
        if find_in_detected(gt, detected):
            found.append(gt)
        else:
            missing.append(gt)
    return found, missing

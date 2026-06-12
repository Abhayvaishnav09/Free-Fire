"""FIFO killfeed state: snapshot diff, event identity, TTL dedup cache."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

try:
    from rapidfuzz import fuzz

    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


@dataclass
class KillfeedEvent:
    killer: str
    victim: str
    event: str  # canonical id e.g. normal_knock
    event_type: str = ""  # knock | elimination | revive
    kill_type: str = ""  # normal | headshot | skill | ...
    weapon: str = "gun"  # icon category
    position: int = 0
    frame: int = 0
    time: float = field(default_factory=time.time)

    @property
    def event_hash(self) -> str:
        k = (self.killer or "ENV").upper()
        v = self.victim.upper()
        return f"{k}_{v}_{self.event}_{self.weapon}".upper()

    @property
    def signature(self) -> str:
        return f"{self.killer}|{self.victim}|{self.event}|{self.weapon}"


def make_signature(
    killer: str,
    victim: str,
    canonical: str,
    icon_category: str = "gun",
) -> str:
    return f"{killer}|{victim}|{canonical}|{icon_category}"


def parse_signature(sig: str) -> Tuple[str, str, str, str]:
    parts = sig.split("|")
    if len(parts) >= 4:
        return parts[0], parts[1], parts[2], parts[3]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2], "gun"
    return "", "", "", "gun"


def _fuzzy_equal(a: str, b: str, threshold: float = 90.0) -> bool:
    if a == b:
        return True
    if not a and not b:
        return True
    if not a or not b:
        return False
    if RAPIDFUZZ_AVAILABLE:
        return fuzz.ratio(a, b) >= threshold
    return a.upper() == b.upper()


def _pair_fuzzy_equal(sig_a: str, sig_b: str, threshold: float = 88.0) -> bool:
    """Same killer+victim+event type even if weapon icon OCR differs."""
    ka, va, ca, wa = parse_signature(sig_a)
    kb, vb, cb, wb = parse_signature(sig_b)
    if not ka or not va or not kb or not vb:
        return False
    if not (_fuzzy_equal(ka, kb, threshold) and _fuzzy_equal(va, vb, threshold)):
        return False
    return _fuzzy_equal(ca, cb, threshold)


def _lists_fuzzy_equal(left: List[str], right: List[str], threshold: float = 90.0) -> bool:
    if len(left) != len(right):
        return False
    return all(_pair_fuzzy_equal(a, b, threshold) for a, b in zip(left, right))


def signatures_match(sig_a: str, sig_b: str, fuzzy_threshold: float = 90.0) -> bool:
    """Public helper: fuzzy equality for two row signatures."""
    return _pair_fuzzy_equal(sig_a, sig_b, fuzzy_threshold)


def find_new_slots(
    previous: List[str],
    current: List[str],
    fuzzy_threshold: float = 90.0,
) -> List[str]:
    """Return signatures inserted above the alignment tail (FIFO new rows only)."""
    if not current:
        return []
    if not previous:
        return []

    if _lists_fuzzy_equal(current, previous, fuzzy_threshold):
        return []

    # Knock → kill upgrade: same players, event type changed
    if len(current) >= 1 and len(previous) >= 1:
        ck, cv, cc, _ = parse_signature(current[0])
        pk, pv, pc, _ = parse_signature(previous[0])
        if (
            ck and cv and pk and pv
            and _fuzzy_equal(ck, pk, fuzzy_threshold)
            and _fuzzy_equal(cv, pv, fuzzy_threshold)
            and not _fuzzy_equal(cc, pc, fuzzy_threshold)
        ):
            return [current[0]]

    if _pair_fuzzy_equal(current[0], previous[0], fuzzy_threshold):
        return []

    for i in range(len(current)):
        tail = current[i:]
        prev_slice = previous[: len(tail)]
        if _lists_fuzzy_equal(tail, prev_slice, fuzzy_threshold):
            new = current[:i]
            return [new[0]] if new else []

    if not any(_pair_fuzzy_equal(current[0], p, fuzzy_threshold) for p in previous):
        return [current[0]]
    return []


class TTLDetectedCache:
    """Short-lived event hash cache — anti-replay, not lifetime-per-match."""

    def __init__(self, ttl_seconds: float = 30.0):
        self.ttl_seconds = ttl_seconds
        self._entries: dict[str, float] = {}

    def _prune(self, now: float) -> None:
        cutoff = now - self.ttl_seconds
        self._entries = {k: v for k, v in self._entries.items() if v >= cutoff}

    def contains(self, event_hash: str) -> bool:
        now = time.time()
        self._prune(now)
        ts = self._entries.get(event_hash.upper())
        return ts is not None and (now - ts) < self.ttl_seconds

    def add(self, event_hash: str) -> None:
        self._entries[event_hash.upper()] = time.time()

    def __len__(self) -> int:
        return len(self._entries)


class PairCooldown:
    """Block duplicate killer+victim+event within a short window (OCR noise)."""

    def __init__(self, ttl_seconds: float = 10.0):
        self.ttl_seconds = ttl_seconds
        self._entries: dict[str, float] = {}

    def _key(self, killer: str, victim: str, canonical: str) -> str:
        return f"{(killer or '').upper()}|{(victim or '').upper()}|{canonical}"

    def _prune(self, now: float) -> None:
        cutoff = now - self.ttl_seconds
        self._entries = {k: v for k, v in self._entries.items() if v >= cutoff}

    def allows(self, killer: str, victim: str, canonical: str) -> bool:
        now = time.time()
        self._prune(now)
        key = self._key(killer, victim, canonical)
        ts = self._entries.get(key)
        if ts is not None and (now - ts) < self.ttl_seconds:
            return False
        if RAPIDFUZZ_AVAILABLE:
            for existing in list(self._entries):
                ek, ev, ec = existing.split("|", 2)
                if (
                    _fuzzy_equal(killer, ek, 88.0)
                    and _fuzzy_equal(victim, ev, 88.0)
                    and _fuzzy_equal(canonical, ec, 95.0)
                    and (now - self._entries[existing]) < self.ttl_seconds
                ):
                    return False
        return True

    def mark(self, killer: str, victim: str, canonical: str) -> None:
        self._entries[self._key(killer, victim, canonical)] = time.time()


class KillfeedState:
    """Persistent FIFO state across OCR snapshots."""

    def __init__(
        self,
        max_visible_slots: int = 4,
        cache_ttl_seconds: float = 30.0,
        pair_cooldown_seconds: float = 10.0,
        fuzzy_threshold: float = 90.0,
    ):
        self.max_visible_slots = max_visible_slots
        self.fuzzy_threshold = fuzzy_threshold
        self.previous_snapshot: List[str] = []
        self.detected_cache = TTLDetectedCache(ttl_seconds=cache_ttl_seconds)
        self.pair_cooldown = PairCooldown(ttl_seconds=pair_cooldown_seconds)
        self.last_strip_dhash: Optional[str] = None
        self.sequence = 0

    def diff_and_commit(self, current: List[str]) -> List[str]:
        trimmed = current[: self.max_visible_slots]
        if not self.previous_snapshot:
            # First sighting: baseline visible rows only — do not emit stale on-screen history
            self.previous_snapshot = list(trimmed)
            return []
        new_slots = find_new_slots(
            self.previous_snapshot,
            trimmed,
            fuzzy_threshold=self.fuzzy_threshold,
        )
        if new_slots or not _lists_fuzzy_equal(trimmed, self.previous_snapshot, self.fuzzy_threshold):
            self.previous_snapshot = list(trimmed)
        return new_slots

    def should_emit(self, killer: str, victim: str, canonical: str, event_hash: str) -> bool:
        if self.detected_cache.contains(event_hash):
            return False
        if not self.pair_cooldown.allows(killer, victim, canonical):
            return False
        return True

    def mark_emitted(self, killer: str, victim: str, canonical: str, event_hash: str) -> None:
        self.detected_cache.add(event_hash)
        self.pair_cooldown.mark(killer, victim, canonical)
        self.sequence += 1

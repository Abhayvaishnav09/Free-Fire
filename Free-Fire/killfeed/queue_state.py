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
    weapon: str = "gun"  # icon category (broad class)
    gun_name: str = "unknown"  # specific gun label e.g. "AK47", "M14"
    position: int = 1
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
    """Return the killfeed rows that are genuinely NEW since the last snapshot.

    The killfeed is a FIFO stack rendered top -> bottom:
        index 0  = TOP    = NEWEST kill
        index -1 = BOTTOM = OLDEST kill

    When new kills appear they are inserted at the TOP and every existing row
    shifts DOWN by the number of new rows; the oldest rows fall off the bottom.

    To find what is new we look for the smallest top-shift ``k`` (>= 1) such that
    ``current[k:]`` lines up with the top of ``previous`` (``previous[0:len]``).
    Everything above that alignment — ``current[0:k]`` — entered since the last
    snapshot.  The returned list is ordered NEWEST -> OLDEST (index 0 first); the
    pipeline reverses it so events are emitted OLDEST -> NEWEST and the newest
    kill ends up at the top of the display.

    This handles all of the reported failure modes:
      • multi-kill bursts (k > 1)            -> every new row returned
      • single new kill (k == 1)             -> just the top row
      • slot shift after a new kill          -> alignment finds the true shift
      • knock then kill of the same pair     -> both are distinct rows, both new
      • dropped/late intermediate frames     -> alignment tolerates large shifts
    """
    if not current or not previous:
        return []

    # Nothing changed.
    if _lists_fuzzy_equal(current, previous, fuzzy_threshold):
        return []

    n = len(current)

    # Bottom expansion: top row stable and previous is a prefix — backfill older
    # rows that are now visible (partial YOLO) without re-emitting the top row.
    if (
        len(current) > len(previous)
        and _pair_fuzzy_equal(current[0], previous[0], fuzzy_threshold)
        and _lists_fuzzy_equal(current[: len(previous)], previous, fuzzy_threshold)
    ):
        return list(reversed(current[len(previous) :]))

    # Find the smallest shift k such that current[k:] matches the top of previous.
    for k in range(n + 1):
        tail = current[k:]
        if not tail:
            # Entire current list is "new" relative to previous (no overlap).
            break
        prev_slice = previous[: len(tail)]
        if len(prev_slice) == len(tail) and _lists_fuzzy_equal(
            tail, prev_slice, fuzzy_threshold
        ):
            return current[:k]  # k new rows at the top, newest -> oldest

    # No clean alignment (heavy OCR jitter or a full screen change):
    # emit every current row that has no fuzzy match anywhere in previous.
    # Preserves top -> bottom (newest -> oldest) order.
    return [
        s
        for s in current
        if not any(_pair_fuzzy_equal(s, p, fuzzy_threshold) for p in previous)
    ]


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
    """Block duplicate killer+victim+event within a short window (OCR noise).

    Uses the FULL canonical ID as the key so that different kill subtypes
    (e.g. normal_knock vs headshot_knock) are treated as distinct events.
    Knock→elimination upgrades always pass because they have different
    canonical IDs.

    The TTL is kept short (default 5s) to avoid blocking legitimate rapid
    kills between the same pair (e.g. knock, revive, knock again).
    """

    def __init__(self, ttl_seconds: float = 5.0):
        self.ttl_seconds = ttl_seconds
        self._entries: dict[str, float] = {}

    @staticmethod
    def _base_event_type(canonical: str) -> str:
        """Return the full canonical ID — no longer collapses subtypes."""
        return canonical

    def _key(self, killer: str, victim: str, canonical: str) -> str:
        base = self._base_event_type(canonical)
        return f"{(killer or '').upper()}|{(victim or '').upper()}|{base}"

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
            norm = self._base_event_type(canonical)
            for existing, ts_e in list(self._entries.items()):
                if (now - ts_e) >= self.ttl_seconds:
                    continue
                ek, ev, ec = existing.split("|", 2)
                if (
                    _fuzzy_equal(killer, ek, 88.0)
                    and _fuzzy_equal(victim, ev, 88.0)
                    and norm == ec  # exact match on full canonical
                ):
                    return False
        return True

    def mark(self, killer: str, victim: str, canonical: str) -> None:
        self._entries[self._key(killer, victim, canonical)] = time.time()


class NamePairLock:
    """Ultra-short cooldown on killer+victim REGARDLESS of event type.

    Purpose: prevent YOLO class flicker (e.g. gun-kill on frame N, gun-knock
    on frame N+1 for the same physical killfeed row) from producing two
    separate events with different statuses.

    The window (default 3s) is shorter than the in-game knock→kill transition
    time (~5-8s in Free Fire: knock animation + finish), so legitimate
    knock → elimination sequences still pass.
    """

    def __init__(self, ttl_seconds: float = 3.0):
        self.ttl_seconds = ttl_seconds
        self._entries: dict[str, float] = {}

    def _key(self, killer: str, victim: str) -> str:
        return f"{(killer or '').upper()}|{(victim or '').upper()}"

    def _prune(self, now: float) -> None:
        cutoff = now - self.ttl_seconds
        self._entries = {k: v for k, v in self._entries.items() if v >= cutoff}

    def allows(self, killer: str, victim: str) -> bool:
        now = time.time()
        self._prune(now)
        key = self._key(killer, victim)
        ts = self._entries.get(key)
        if ts is not None and (now - ts) < self.ttl_seconds:
            return False
        if RAPIDFUZZ_AVAILABLE:
            for existing, ts_e in list(self._entries.items()):
                if (now - ts_e) >= self.ttl_seconds:
                    continue
                ek, ev = existing.split("|", 1)
                if _fuzzy_equal(killer, ek, 88.0) and _fuzzy_equal(victim, ev, 88.0):
                    return False
        return True

    def mark(self, killer: str, victim: str) -> None:
        self._entries[self._key(killer, victim)] = time.time()


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
        self.name_pair_lock = NamePairLock(ttl_seconds=3.0)
        self.last_strip_dhash: Optional[str] = None
        self.sequence = 0

    def diff_and_commit(self, current: List[str]) -> List[str]:
        trimmed = current[: self.max_visible_slots]
        if not self.previous_snapshot:
            self.previous_snapshot = list(trimmed)
            # Emit every kill currently visible on screen (oldest first).
            return list(reversed(trimmed))
        new_slots = find_new_slots(
            self.previous_snapshot,
            trimmed,
            fuzzy_threshold=self.fuzzy_threshold,
        )
        # Knock→kill on the same bar: row-0 canonical changes while lower rows stay put.
        if (
            not new_slots
            and trimmed
            and self.previous_snapshot
            and len(trimmed) == len(self.previous_snapshot)
            and len(trimmed) > 1
            and _lists_fuzzy_equal(
                trimmed[1:], self.previous_snapshot[1:], self.fuzzy_threshold
            )
            and not _pair_fuzzy_equal(
                trimmed[0], self.previous_snapshot[0], self.fuzzy_threshold
            )
        ):
            new_slots = [trimmed[0]]
        if new_slots or not _lists_fuzzy_equal(trimmed, self.previous_snapshot, self.fuzzy_threshold):
            self.previous_snapshot = list(trimmed)
        return new_slots

    def should_emit(self, killer: str, victim: str, canonical: str, event_hash: str) -> bool:
        if self.detected_cache.contains(event_hash):
            return False
        if not self.name_pair_lock.allows(killer, victim):
            return False
        if not self.pair_cooldown.allows(killer, victim, canonical):
            return False
        return True

    def mark_emitted(self, killer: str, victim: str, canonical: str, event_hash: str) -> None:
        self.detected_cache.add(event_hash)
        self.pair_cooldown.mark(killer, victim, canonical)
        self.name_pair_lock.mark(killer, victim)
        self.sequence += 1

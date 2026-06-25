"""Match-scoped visual dedup — survives TTL expiry, prevents killfeed replay."""

from __future__ import annotations

from typing import Optional, Set

from killfeed.dhash import dhash_distance


class MatchVisualLedger:
    """Every visual bar fingerprint processed this match (fuzzy dHash)."""

    def __init__(self, max_dhash_distance: int = 5) -> None:
        self.max_dhash_distance = max_dhash_distance
        self._sigs: Set[str] = set()

    def __len__(self) -> int:
        return len(self._sigs)

    def clear(self) -> None:
        self._sigs.clear()

    def _match(self, sig: str) -> Optional[str]:
        if not sig:
            return None
        if sig in self._sigs:
            return sig
        best: Optional[str] = None
        best_dist = self.max_dhash_distance + 1
        for known in self._sigs:
            dist = dhash_distance(sig, known)
            if dist <= self.max_dhash_distance and dist < best_dist:
                best_dist = dist
                best = known
        return best

    def contains(self, sig: str) -> bool:
        """Exact match only — fuzzy match here caused knock→kill pairs to be skipped."""
        return bool(sig) and sig in self._sigs

    def contains_fuzzy(self, sig: str) -> bool:
        return self._match(sig) is not None

    def add(self, sig: str) -> None:
        if sig:
            self._sigs.add(sig)

    def known_sigs(self) -> Set[str]:
        return set(self._sigs)

"""OCR name normalization and fuzzy roster matching."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

try:
    from rapidfuzz import fuzz, process

    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False


@dataclass
class RosterMatchConfig:
    hard_accept: float = 0.72
    soft_accept: float = 0.42
    min_margin: float = 0.06
    min_length: int = 2


@dataclass
class MatchResult:
    matched: bool
    canonical: str
    score: float
    tier: str = ""


def normalize_ocr_name(name: str) -> str:
    if not name:
        return ""
    name = re.sub(r"[^\w\d._-]", "", str(name).strip())
    name = re.sub(r"_{2,}", "_", name)
    return name.strip("._-")


def _levenshtein_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if _HAS_RAPIDFUZZ:
        return fuzz.ratio(a.upper(), b.upper()) / 100.0
    # Basic fallback
    dist = _levenshtein(a.upper(), b.upper())
    return 1.0 - dist / max(len(a), len(b), 1)


def _levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(
                min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2))
            )
        prev = curr
    return prev[-1]


def match_player_name(
    ocr_name: str,
    players: List[str],
    config: Optional[RosterMatchConfig] = None,
) -> MatchResult:
    cfg = config or RosterMatchConfig()
    raw = normalize_ocr_name(ocr_name)
    if len(raw) < cfg.min_length:
        return MatchResult(False, "", 0.0)

    if not players:
        return MatchResult(bool(raw), raw, 0.0, "soft" if raw else "")

    best_name = ""
    best_score = 0.0
    second_score = 0.0

    if _HAS_RAPIDFUZZ:
        result = process.extractOne(
            raw,
            players,
            scorer=fuzz.WRatio,
            score_cutoff=0,
        )
        if result:
            best_name, best_score, _ = result
            best_score /= 100.0
            for p in players:
                if p == best_name:
                    continue
                s = fuzz.WRatio(raw, p) / 100.0
                if s > second_score:
                    second_score = s
    else:
        for p in players:
            s = _levenshtein_ratio(raw, p)
            if s > best_score:
                second_score = best_score
                best_score = s
                best_name = p
            elif s > second_score:
                second_score = s

    margin = best_score - second_score
    if best_score >= cfg.hard_accept:
        return MatchResult(True, best_name, best_score, "hard")
    if best_score >= cfg.soft_accept and margin >= cfg.min_margin:
        return MatchResult(True, best_name, best_score, "soft")
    return MatchResult(False, best_name, best_score, "")

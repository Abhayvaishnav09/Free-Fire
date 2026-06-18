"""Gun icon template matching — identifies which weapon was used in a killfeed row.

How to add gun templates
------------------------
1. Run the pipeline; icon crops are auto-saved to ``gun_templates/unknown/``
   whenever no template matches (requires ``save_unknown_icons: true`` in config).
2. Inspect each saved PNG, rename it to the gun name (e.g. ``AK47.png``), and
   move it into ``gun_templates/`` (next to the ``unknown/`` sub-folder).
3. Restart — the classifier loads all ``*.png / *.jpg`` files in the root of
   ``gun_template_dir`` and matches against them.

Template file naming
--------------------
Use the exact gun label you want to appear in events / API:
    AK47.png  →  "AK47"
    M14.png   →  "M14"
    GROZA.png →  "GROZA"
    MP40.png  →  "MP40"
    ...

The name is converted to uppercase on load and returned as-is on match.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from killfeed.icons import extract_icon_roi

# ── tunable constants ────────────────────────────────────────────────────────
_ICON_W, _ICON_H = 64, 32          # fixed resize for all comparisons
_MATCH_THRESHOLD = 0.62            # NCC score to accept a match (0..1)
_HIST_WEIGHT = 0.30                # blend: score = NCC*(1-w) + hist_corr*w
_UNKNOWN_SAVE_COOLDOWN = 3.0       # seconds between saves of unknown icons


@dataclass
class GunMatch:
    gun_name: str       # matched gun label, or "unknown"
    score: float        # best NCC score (0..1)
    matched: bool       # True when score >= threshold


# ── helpers ──────────────────────────────────────────────────────────────────

def _prepare(gray: np.ndarray) -> np.ndarray:
    """Resize + CLAHE normalise to (_ICON_W, _ICON_H) float32."""
    resized = cv2.resize(gray, (_ICON_W, _ICON_H), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(resized)
    norm = cv2.normalize(enhanced.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
    return norm


def _ncc(query: np.ndarray, template: np.ndarray) -> float:
    """Normalised cross-correlation between two same-size float arrays → [-1, 1]."""
    result = cv2.matchTemplate(query, template, cv2.TM_CCOEFF_NORMED)
    return float(result.max())


def _hist_corr(gray_q: np.ndarray, gray_t: np.ndarray) -> float:
    """Histogram correlation between two same-size uint8 arrays → [0, 1]."""
    hist_q = cv2.calcHist([gray_q], [0], None, [64], [0, 256])
    hist_t = cv2.calcHist([gray_t], [0], None, [64], [0, 256])
    cv2.normalize(hist_q, hist_q)
    cv2.normalize(hist_t, hist_t)
    corr = cv2.compareHist(hist_q, hist_t, cv2.HISTCMP_CORREL)
    return max(0.0, float(corr))


# ── main classifier ──────────────────────────────────────────────────────────

class GunTemplateClassifier:
    """
    Match a killfeed weapon-icon crop against stored reference templates.
    Thread-safe after ``__init__`` (all state is read-only post-load).
    """

    def __init__(self, template_dir: str | None = None, save_unknowns: bool = False):
        self._templates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._template_dir: str | None = template_dir
        self._save_unknowns = save_unknowns
        self._last_save_time: float = 0.0
        self._unknown_save_idx: int = 0

        if template_dir:
            self._load_templates(template_dir)

    # ── loading ──────────────────────────────────────────────────────────────

    def _load_templates(self, template_dir: str) -> None:
        p = Path(template_dir)
        if not p.is_dir():
            print(f"⚠️  gun_classifier: template dir not found: {template_dir}")
            return
        loaded = 0
        for img_path in sorted(p.glob("*")):
            if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            bgr = cv2.imread(str(img_path))
            if bgr is None or bgr.size == 0:
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            gun_name = img_path.stem.upper()
            prepared = _prepare(gray)
            # store both normalised float (for NCC) and uint8 (for hist)
            uint8 = cv2.resize(gray, (_ICON_W, _ICON_H), interpolation=cv2.INTER_AREA)
            self._templates[gun_name] = (prepared, uint8)
            loaded += 1
        if loaded:
            print(f"🔫 Gun classifier: {loaded} templates loaded from {template_dir}")
        else:
            print(f"🔫 Gun classifier: no templates in {template_dir} — gun_name will be 'unknown'")

    def reload(self) -> None:
        """Hot-reload templates without restarting the pipeline."""
        if self._template_dir:
            self._templates.clear()
            self._load_templates(self._template_dir)

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def loaded(self) -> bool:
        return bool(self._templates)

    @property
    def gun_names(self) -> list[str]:
        return sorted(self._templates)

    # ── classification ────────────────────────────────────────────────────────

    def classify(self, row_crop: np.ndarray) -> GunMatch:
        """Return the best-matching gun name (or 'unknown') for a killfeed row crop."""
        icon = extract_icon_roi(row_crop)
        if icon is None or icon.size == 0:
            return GunMatch("unknown", 0.0, False)

        gray = cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY) if len(icon.shape) == 3 else icon.copy()

        if not self._templates:
            self._maybe_save_unknown(gray, "no_templates")
            return GunMatch("unknown", 0.0, False)

        query_f = _prepare(gray)
        query_u8 = cv2.resize(gray, (_ICON_W, _ICON_H), interpolation=cv2.INTER_AREA)

        best_name = "unknown"
        best_score = -1.0
        for gun_name, (tmpl_f, tmpl_u8) in self._templates.items():
            ncc_score = _ncc(query_f, tmpl_f)
            hist_score = _hist_corr(query_u8, tmpl_u8)
            score = ncc_score * (1.0 - _HIST_WEIGHT) + hist_score * _HIST_WEIGHT
            if score > best_score:
                best_score = score
                best_name = gun_name

        matched = best_score >= _MATCH_THRESHOLD
        if not matched:
            self._maybe_save_unknown(gray, best_name)
            return GunMatch("unknown", best_score, False)
        return GunMatch(best_name, best_score, True)

    # ── auto-save unknowns ────────────────────────────────────────────────────

    def _maybe_save_unknown(self, gray: np.ndarray, reason: str) -> None:
        """Save the icon crop to ``<template_dir>/unknown/`` for manual labelling."""
        if not self._save_unknowns or not self._template_dir:
            return
        now = time.time()
        if now - self._last_save_time < _UNKNOWN_SAVE_COOLDOWN:
            return
        self._last_save_time = now
        self._unknown_save_idx += 1
        out_dir = Path(self._template_dir) / "unknown"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = int(now)
        fname = out_dir / f"icon_{ts}_{self._unknown_save_idx:04d}.png"
        resized = cv2.resize(gray, (_ICON_W * 2, _ICON_H * 2), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(fname), resized)


# ── module-level singleton ────────────────────────────────────────────────────

_classifier: Optional[GunTemplateClassifier] = None


def init_gun_classifier(template_dir: str | None = None, save_unknowns: bool = False) -> None:
    """Call once at startup (before the pipeline threads start)."""
    global _classifier
    _classifier = GunTemplateClassifier(template_dir=template_dir, save_unknowns=save_unknowns)


def classify_gun(row_crop: np.ndarray) -> str:
    """
    Return the gun name string (e.g. ``'AK47'``) or ``'unknown'``.
    Thread-safe after ``init_gun_classifier`` is called.
    """
    if _classifier is None or row_crop is None or row_crop.size == 0:
        return "unknown"
    return _classifier.classify(row_crop).gun_name


def reload_gun_templates() -> None:
    """Hot-reload templates at runtime (e.g. after adding new PNGs)."""
    if _classifier is not None:
        _classifier.reload()

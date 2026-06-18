"""Difference hash for visual change gating on killfeed strip."""

from __future__ import annotations

import cv2
import numpy as np


def compute_dhash(image: np.ndarray, hash_size: int = 8) -> str:
    """Compute dHash hex string from BGR image."""
    if image is None or image.size == 0:
        return ""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = diff.flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:0{width}x}" if (width := (hash_size * hash_size + 3) // 4) else f"{value:x}"


def dhash_distance(h1: str, h2: str) -> int:
    if not h1 or not h2 or len(h1) != len(h2):
        return 999
    try:
        a = int(h1, 16)
        b = int(h2, 16)
        return (a ^ b).bit_count()
    except ValueError:
        return 999


def strip_changed(
    image: np.ndarray,
    last_dhash: str | None,
    max_distance: int = 0,
) -> tuple[bool, str]:
    """Return (changed, new_dhash). max_distance=0 means exact match only."""
    new_hash = compute_dhash(image)
    if not last_dhash:
        return True, new_hash
    if dhash_distance(new_hash, last_dhash) > max_distance:
        return True, new_hash
    return False, last_dhash


def pixel_mean_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 999.0
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    diff = cv2.absdiff(a, b)
    return float(np.mean(diff))

"""Grayscale enhancement for OCR crops."""

from __future__ import annotations

import cv2
import numpy as np


def enhance_crop_for_ocr(crop: np.ndarray) -> np.ndarray:
    """Improve text contrast for PaddleOCR without altering status detection crops."""
    if crop is None or crop.size == 0:
        return crop
    try:
        if len(crop.shape) == 2:
            gray = crop
        else:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    except Exception:
        return crop

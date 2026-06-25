"""Prepare killfeed row crops for TMS upload (full bar, not clipped)."""

from __future__ import annotations

import cv2
import numpy as np

# Match reference killfeed aspect (~581×50).
_REF_W, _REF_H = 581, 50
_ASPECT = _REF_W / float(_REF_H)


def pad_row_crop_for_tms(
    crop: np.ndarray,
    *,
    pad_x: int = 6,
    pad_y: int = 10,
    min_height: int = 44,
) -> np.ndarray:
    """
    Expand tight YOLO crops so player names and icons are not cut off in TMS.
    Uses edge replication (not grey bars) before letterbox scaling.
    """
    if crop is None or getattr(crop, "size", 0) == 0:
        return crop
    img = crop
    if pad_x or pad_y:
        img = cv2.copyMakeBorder(
            img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE
        )
    h, w = img.shape[:2]
    # Ensure minimum height for the bar aspect ratio (names need vertical room).
    target_h = max(min_height, int(round(w / _ASPECT)))
    if h < target_h:
        extra = target_h - h
        top = extra // 2
        bottom = extra - top
        img = cv2.copyMakeBorder(
            img, top, bottom, 0, 0, cv2.BORDER_REPLICATE
        )
    return img


def prepare_killfeed_for_tms(
    crop: np.ndarray,
    *,
    letterbox: bool = True,
    max_width: int = 520,
) -> np.ndarray:
    """Pad → optional letterbox scale → cap width for upload."""
    img = pad_row_crop_for_tms(crop)
    if letterbox:
        try:
            from gray_scale.grey_scale import scale_killfeed_crop

            img = scale_killfeed_crop(img)
        except Exception:
            pass
    h, w = img.shape[:2]
    if max_width > 0 and w > max_width:
        scale = max_width / float(w)
        img = cv2.resize(
            img,
            (max_width, max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return img

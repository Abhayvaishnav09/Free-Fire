"""YOLO ROI: union killfeed strip and per-row crops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class RowDetection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_name: str
    crop: np.ndarray


@dataclass
class StripROI:
    strip: np.ndarray
    strip_bbox: Tuple[int, int, int, int]
    rows: List[RowDetection]


@dataclass
class StripStabilizer:
    """
    Light smoothing only — keeps original YOLO row crops for OCR.
    Merges duplicate overlapping boxes; smooths strip union bbox slightly.
    """

    alpha: float = 0.25
    max_rows: int = 4
    merge_y_ratio: float = 0.45
    row_pad_y: int = 2
    row_pad_x: int = 2
    _strip_bbox: Optional[Tuple[int, int, int, int]] = None

    def reset(self) -> None:
        self._strip_bbox = None

    def _smooth_bbox(
        self, bbox: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        if self._strip_bbox is None:
            self._strip_bbox = bbox
            return bbox
        ax1, ay1, ax2, ay2 = self._strip_bbox
        bx1, by1, bx2, by2 = bbox
        a = self.alpha
        smoothed = (
            int(ax1 * (1 - a) + bx1 * a),
            int(ay1 * (1 - a) + by1 * a),
            int(ax2 * (1 - a) + bx2 * a),
            int(ay2 * (1 - a) + by2 * a),
        )
        self._strip_bbox = smoothed
        return smoothed

    def _merge_nearby_rows(self, rows: List[RowDetection]) -> List[RowDetection]:
        if len(rows) <= 1:
            return rows
        merged: List[RowDetection] = []
        for row in rows:
            placed = False
            rx1, ry1, rx2, ry2 = row.bbox
            rh = max(ry2 - ry1, 1)
            for i, kept in enumerate(merged):
                kx1, ky1, kx2, ky2 = kept.bbox
                kh = max(ky2 - ky1, 1)
                overlap = min(ry2, ky2) - max(ry1, ky1)
                if overlap >= min(rh, kh) * self.merge_y_ratio:
                    if row.confidence > kept.confidence:
                        merged[i] = row
                    placed = True
                    break
            if not placed:
                merged.append(row)
        merged.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        return merged[: self.max_rows]

    def _pad_row_crop(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - self.row_pad_x)
        y1 = max(0, y1 - self.row_pad_y)
        x2 = min(fw, x2 + self.row_pad_x)
        y2 = min(fh, y2 + self.row_pad_y)
        if x2 <= x1 or y2 <= y1:
            return bbox, frame[bbox[1] : bbox[3], bbox[0] : bbox[2]].copy()
        return (x1, y1, x2, y2), frame[y1:y2, x1:x2].copy()

    def apply(self, frame: np.ndarray, rows: List[RowDetection]) -> StripROI:
        """Return rows with original YOLO-aligned crops (small pad only)."""
        if not rows:
            raise ValueError("rows required")

        rows = self._merge_nearby_rows(rows)
        padded_rows: List[RowDetection] = []
        for row in rows:
            bbox, crop = self._pad_row_crop(frame, row.bbox)
            if crop.size == 0:
                continue
            padded_rows.append(
                RowDetection(
                    bbox=bbox,
                    confidence=row.confidence,
                    class_name=row.class_name,
                    crop=crop,
                )
            )

        if not padded_rows:
            padded_rows = rows

        raw_x1 = min(r.bbox[0] for r in padded_rows)
        raw_y1 = min(r.bbox[1] for r in padded_rows)
        raw_x2 = max(r.bbox[2] for r in padded_rows)
        raw_y2 = max(r.bbox[3] for r in padded_rows)
        sx1, sy1, sx2, sy2 = self._smooth_bbox((raw_x1, raw_y1, raw_x2, raw_y2))

        fh, fw = frame.shape[:2]
        sx1 = max(0, min(sx1, fw - 1))
        sy1 = max(0, min(sy1, fh - 1))
        sx2 = max(sx1 + 1, min(sx2, fw))
        sy2 = max(sy1 + 1, min(sy2, fh))
        strip = frame[sy1:sy2, sx1:sx2].copy()
        return StripROI(strip=strip, strip_bbox=(sx1, sy1, sx2, sy2), rows=padded_rows)


def _convert_bbox(bbox_resized, orig_size, resize_size=(1280, 720)):
    x1_r, y1_r, x2_r, y2_r = bbox_resized
    orig_h, orig_w = orig_size
    resize_w, resize_h = resize_size
    x1 = max(0, min(int(x1_r * orig_w / resize_w), orig_w - 1))
    y1 = max(0, min(int(y1_r * orig_h / resize_h), orig_h - 1))
    x2 = max(x1 + 1, min(int(x2_r * orig_w / resize_w), orig_w))
    y2 = max(y1 + 1, min(int(y2_r * orig_h / resize_h), orig_h))
    return x1, y1, x2, y2


def detect_strip(
    frame: np.ndarray,
    model,
    kill_confidence: float = 0.20,
    revive_confidence: float = 0.28,
    imgsz: int = 640,
    stabilizer: Optional[StripStabilizer] = None,
    yolo_lock=None,
) -> Optional[StripROI]:
    """Run YOLO and return union strip + sorted row crops (top to bottom)."""
    if frame is None or frame.size == 0 or model is None:
        return None

    orig_size = frame.shape[:2]
    processed = cv2.resize(frame, (1280, 720))
    names_map = getattr(model, "names", None)

    def _predict():
        return model.predict(
            processed,
            conf=min(kill_confidence, revive_confidence),
            verbose=False,
            device="cpu",
            imgsz=imgsz,
            half=False,
        )

    if yolo_lock is not None:
        with yolo_lock:
            results = _predict()
    else:
        results = _predict()

    if not results or not results[0].boxes or len(results[0].boxes) == 0:
        return None

    rows: List[RowDetection] = []
    for box in results[0].boxes:
        confidence = float(box.conf[0].cpu().numpy())
        class_id = int(box.cls[0].cpu().numpy())
        if isinstance(names_map, dict):
            class_name = names_map.get(class_id, f"class_{class_id}")
        else:
            class_name = f"class_{class_id}"
        class_lower = class_name.lower()
        min_conf = revive_confidence if class_lower == "revive" else kill_confidence
        if class_lower not in ("killblock", "revive") or confidence < min_conf:
            continue
        bbox = _convert_bbox(box.xyxy[0].cpu().numpy(), orig_size)
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        rows.append(
            RowDetection(
                bbox=bbox,
                confidence=confidence,
                class_name=class_lower,
                crop=crop.copy(),
            )
        )

    if not rows:
        return None

    rows.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
    if stabilizer is not None:
        return stabilizer.apply(frame, rows)

    x1 = min(r.bbox[0] for r in rows)
    y1 = min(r.bbox[1] for r in rows)
    x2 = max(r.bbox[2] for r in rows)
    y2 = max(r.bbox[3] for r in rows)
    strip = frame[y1:y2, x1:x2].copy()
    return StripROI(strip=strip, strip_bbox=(x1, y1, x2, y2), rows=rows)

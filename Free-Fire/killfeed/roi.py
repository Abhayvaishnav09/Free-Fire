"""YOLO ROI: union killfeed strip and per-row crops."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from killfeed.yolo_classes import (
    class_priority,
    is_generic_class,
    is_row_class,
    min_confidence_for_class,
)


@dataclass
class RowDetection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_name: str
    crop: np.ndarray


def _bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _union_bbox(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _bbox_center_inside(
    inner: Tuple[int, int, int, int], outer: Tuple[int, int, int, int]
) -> bool:
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    cx = (ix1 + ix2) / 2.0
    cy = (iy1 + iy2) / 2.0
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2


def _rows_overlap(a: RowDetection, b: RowDetection, y_ratio: float = 0.25) -> bool:
    """True when revive is inside killblock or rows share vertical band."""
    ax1, ay1, ax2, ay2 = a.bbox
    bx1, by1, bx2, by2 = b.bbox
    if _bbox_center_inside(a.bbox, b.bbox) or _bbox_center_inside(b.bbox, a.bbox):
        return True
    overlap = min(ay2, by2) - max(ay1, by1)
    ah = max(ay2 - ay1, 1)
    bh = max(by2 - by1, 1)
    return overlap >= min(ah, bh) * y_ratio


def _pick_overlap_winner(a: RowDetection, b: RowDetection) -> RowDetection:
    """Prefer specific YOLO class; tie-break on confidence then bbox area."""
    pa, pb = class_priority(a.class_name), class_priority(b.class_name)
    if pa != pb:
        winner = a if pa > pb else b
        loser = b if winner is a else a
    elif a.confidence != b.confidence:
        winner = a if a.confidence >= b.confidence else b
        loser = b if winner is a else a
    else:
        winner = a if _bbox_area(a.bbox) >= _bbox_area(b.bbox) else b
        loser = b if winner is a else a
    return RowDetection(
        bbox=winner.bbox if _bbox_area(winner.bbox) >= _bbox_area(loser.bbox) else loser.bbox,
        confidence=max(a.confidence, b.confidence),
        class_name=winner.class_name,
        crop=winner.crop if winner.crop.size else loser.crop,
    )


def consolidate_overlapping_rows(rows: List[RowDetection]) -> List[RowDetection]:
    """Merge overlapping YOLO boxes on the same killfeed row."""
    if not rows:
        return rows

    rows = list(rows)
    merged: List[RowDetection] = []
    for row in rows:
        placed = False
        for i, kept in enumerate(merged):
            if _rows_overlap(row, kept):
                merged[i] = _pick_overlap_winner(row, kept)
                placed = True
                break
        if not placed:
            merged.append(row)

    # Drop generic killblock when a specific class covers the same row.
    filtered: List[RowDetection] = []
    for row in merged:
        if is_generic_class(row.class_name):
            if any(
                not is_generic_class(other.class_name) and _rows_overlap(row, other)
                for other in merged
                if other is not row
            ):
                continue
        filtered.append(row)
    return filtered


def associate_revive_with_killblocks(rows: List[RowDetection]) -> List[RowDetection]:
    """Backward-compatible alias — best (1).pt uses per-row event classes."""
    return consolidate_overlapping_rows(rows)


def _maybe_expand_revive_bbox(
    bbox: Tuple[int, int, int, int],
    class_name: str,
    frame_width: int,
) -> Tuple[int, int, int, int]:
    """Expand tiny revive-only YOLO boxes toward a full killfeed bar width."""
    if class_name != "revive":
        return bbox
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    if w >= frame_width * 0.22:
        return bbox
    cx = (x1 + x2) // 2
    half = max(int(w * 2.5), int(frame_width * 0.17))
    return (max(0, cx - half), y1, min(frame_width, cx + half), y2)


def consolidate_detection_dicts(detections: List[dict]) -> List[dict]:
    """Legacy ffkillblock path: promote full killblock when revive overlaps."""
    if len(detections) <= 1:
        return detections

    rows = [
        RowDetection(
            bbox=tuple(d["bbox"]),
            confidence=float(d.get("confidence", 0.0)),
            class_name=str(d.get("class_name", "killblock")).lower(),
            crop=np.array([]),
        )
        for d in detections
    ]
    merged_rows = associate_revive_with_killblocks(rows)
    return [
        {
            "bbox": list(row.bbox),
            "confidence": row.confidence,
            "class_name": row.class_name,
        }
        for row in merged_rows
    ]


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
                if overlap >= min(rh, kh) * self.merge_y_ratio or _rows_overlap(row, kept, 0.20):
                    merged[i] = _pick_overlap_winner(row, kept)
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

        rows = consolidate_overlapping_rows(rows)
        rows = self._merge_nearby_rows(rows)
        padded_rows: List[RowDetection] = []
        fh, fw = frame.shape[:2]
        for row in rows:
            bbox = _maybe_expand_revive_bbox(row.bbox, row.class_name, fw)
            bbox, crop = self._pad_row_crop(frame, bbox)
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


_detect_strip_call_count = 0


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
    global _detect_strip_call_count
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

    rows: List[RowDetection] = []
    try:
        if not results or not results[0].boxes or len(results[0].boxes) == 0:
            return None

        for box in results[0].boxes:
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            if isinstance(names_map, dict):
                class_name = names_map.get(class_id, f"class_{class_id}")
            else:
                class_name = f"class_{class_id}"
            class_lower = class_name.lower()
            min_conf = min_confidence_for_class(
                class_lower,
                revive_confidence if class_lower == "revive" else kill_confidence,
            )
            if not is_row_class(class_lower) or confidence < min_conf:
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
    finally:
        # Explicitly release YOLO result tensors so PyTorch doesn't accumulate
        # them across the thousands of inference calls made during a match.
        del results
        processed = None  # type: ignore[assignment]

    # Periodic GC in the capture thread (every ~300 frames ≈ 10 s at 30 fps)
    _detect_strip_call_count += 1
    if _detect_strip_call_count % 300 == 0:
        gc.collect()

    if not rows:
        return None

    rows = consolidate_overlapping_rows(rows)
    rows.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
    if stabilizer is not None:
        return stabilizer.apply(frame, rows)

    x1 = min(r.bbox[0] for r in rows)
    y1 = min(r.bbox[1] for r in rows)
    x2 = max(r.bbox[2] for r in rows)
    y2 = max(r.bbox[3] for r in rows)
    strip = frame[y1:y2, x1:x2].copy()
    return StripROI(strip=strip, strip_bbox=(x1, y1, x2, y2), rows=rows)


def reconstruct_strip_from_rows(
    frame: np.ndarray,
    row_specs: List[RowDetection],
    stabilizer: Optional[StripStabilizer] = None,
) -> Optional[StripROI]:
    """Re-crop killfeed rows from last-known bboxes when YOLO briefly misses."""
    if frame is None or frame.size == 0 or not row_specs:
        return None

    fh, fw = frame.shape[:2]
    rows: List[RowDetection] = []
    for spec in row_specs:
        x1, y1, x2, y2 = spec.bbox
        x1 = max(0, min(int(x1), fw - 1))
        y1 = max(0, min(int(y1), fh - 1))
        x2 = max(x1 + 1, min(int(x2), fw))
        y2 = max(y1 + 1, min(int(y2), fh))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        rows.append(
            RowDetection(
                bbox=(x1, y1, x2, y2),
                confidence=float(spec.confidence),
                class_name=str(spec.class_name or "killblock").lower(),
                crop=crop.copy(),
            )
        )

    if not rows:
        return None

    rows = consolidate_overlapping_rows(rows)
    rows.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
    if stabilizer is not None:
        return stabilizer.apply(frame, rows)

    x1 = min(r.bbox[0] for r in rows)
    y1 = min(r.bbox[1] for r in rows)
    x2 = max(r.bbox[2] for r in rows)
    y2 = max(r.bbox[3] for r in rows)
    strip = frame[y1:y2, x1:x2].copy()
    return StripROI(strip=strip, strip_bbox=(x1, y1, x2, y2), rows=rows)

# #!/usr/bin/env python3
# """
# Killfeed background -> adaptive neutral grayscale (standalone).

# Strong B&W background + sharp protected foreground (original RGB, OCR-safe).

# Usage:
#     python killfeed_grayscale_bg.py killfeed.png
#     python killfeed_grayscale_bg.py killfeed.png -o killfeed_grayscale_bg.png
# """

# from __future__ import annotations

# import argparse
# import os
# import sys

# import cv2
# import numpy as np

# CHROMA_TARGET_MEAN = 5.0
# CHROMA_TARGET_P90 = 10.0
# MAX_ADAPTIVE_PASSES = 14


# def _background_chroma_map(bgr: np.ndarray) -> np.ndarray:
#     lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
#     a = lab[:, :, 1].astype(np.float32)
#     b = lab[:, :, 2].astype(np.float32)
#     return np.sqrt((a - 128.0) ** 2 + (b - 128.0) ** 2)


# def _measure_background_chroma(bgr: np.ndarray, bg_mask: np.ndarray) -> tuple[float, float]:
#     if not np.any(bg_mask):
#         return 0.0, 0.0
#     chroma = _background_chroma_map(bgr)[bg_mask > 0]
#     return float(np.mean(chroma)), float(np.percentile(chroma, 90))


# def _build_scenery_mask(
#     h: np.ndarray,
#     s: np.ndarray,
#     v: np.ndarray,
#     r: np.ndarray,
#     g: np.ndarray,
#     b: np.ndarray,
#     level: int,
# ) -> np.ndarray:
#     rg_close = np.abs(r.astype(np.int16) - g.astype(np.int16)) < 42
#     sat_floor = max(6, 28 - level * 2)

#     brown_tan = (
#         (h >= 3) & (h <= 35) &
#         (s >= sat_floor) & (v >= 28) & (v <= 210) &
#         rg_close
#     )
#     olive = (
#         (h >= 16) & (h <= 58) &
#         (s >= sat_floor) & (v >= 32) & (v <= 200) &
#         rg_close
#     )
#     red_env = (
#         (h <= 22) &
#         (s >= sat_floor) &
#         (v >= 30) & (v <= 220) &
#         (r.astype(np.int16) >= g.astype(np.int16) - 8)
#     )
#     dark_neutral = (s <= max(20, 55 - level * 3)) & (v >= 20) & (v <= 170)
#     return brown_tan | olive | red_env | dark_neutral


# def _dilate_mask(mask_u8: np.ndarray, iterations: int = 1) -> np.ndarray:
#     if iterations <= 0:
#         return mask_u8
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     return cv2.dilate(mask_u8, kernel, iterations=iterations)


# def _close_mask(mask_u8: np.ndarray, iterations: int = 1) -> np.ndarray:
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     return cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=iterations)


# def _remove_oversized_blobs(
#     mask_u8: np.ndarray,
#     img_shape: tuple[int, ...],
#     max_area_ratio: float = 0.29,
# ) -> np.ndarray:
#     """Remove only huge scenery blobs; keep medium-sized text regions."""
#     if not np.any(mask_u8):
#         return mask_u8

#     h, w = img_shape[:2]
#     max_area = int(h * w * max_area_ratio)
#     num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
#     kept = np.zeros_like(mask_u8)
#     for i in range(1, num):
#         area = int(stats[i, cv2.CC_STAT_AREA])
#         if area > max_area:
#             continue
#         kept[labels == i] = 255
#     return kept


# def _extract_strokes_from_wide_slab(
#     component: np.ndarray,
#     bgr: np.ndarray,
#     stroke_rule: np.ndarray,
# ) -> np.ndarray:
#     """Pull text strokes out of a wide red/scenery slab connected component."""
#     stroke = component & stroke_rule
#     stroke_u8 = stroke.astype(np.uint8) * 255
#     if not np.any(stroke_u8):
#         return stroke_u8
#     stroke_u8 = _close_mask(stroke_u8, 2)
#     return _dilate_mask(stroke_u8, 1)


# def _filter_foreground_blobs(
#     mask_u8: np.ndarray,
#     img_shape: tuple[int, ...],
#     bgr: np.ndarray | None = None,
#     slab_mode: str | None = None,
# ) -> np.ndarray:
#     """Drop large scenery blobs; recover text strokes from wide slabs."""
#     if not np.any(mask_u8):
#         return mask_u8

#     h, w = img_shape[:2]
#     max_area = int(h * w * 0.22)
#     num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
#     kept = np.zeros_like(mask_u8)

#     stroke_rule = None
#     if slab_mode == "red" and bgr is not None:
#         b, g, r = cv2.split(bgr)
#         hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
#         s = hsv[:, :, 1]
#         stroke_rule = (r > g + 34) & (r > b + 34) & (s >= 58)

#     for i in range(1, num):
#         area = int(stats[i, cv2.CC_STAT_AREA])
#         bw = int(stats[i, cv2.CC_STAT_WIDTH])
#         bh = int(stats[i, cv2.CC_STAT_HEIGHT])
#         component = labels == i

#         if bw > int(w * 0.72) and slab_mode == "red" and stroke_rule is not None:
#             slab_strokes = _extract_strokes_from_wide_slab(component, bgr, stroke_rule)
#             kept = np.maximum(kept, slab_strokes)
#             continue

#         if area > max_area:
#             continue
#         if bw > int(w * 0.82) and bh > int(h * 0.45):
#             continue
#         kept[component] = 255
#     return kept


# def _color_family_masks(bgr: np.ndarray) -> dict[str, np.ndarray]:
#     """Per-color seed masks (bool arrays)."""
#     hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(hsv)
#     lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
#     lightness = lab[:, :, 0]
#     b, g, r = cv2.split(bgr)

#     white = ((v >= 148) & (s <= 78)) | (lightness >= 178)

#     red_bright = (r >= 105) & (r > g + 48) & (r > b + 48) & (s >= 82)
#     red_dark = (r >= 78) & (r > g + 26) & (r > b + 26) & (s >= 52) & (v >= 36) & (v <= 165)
#     red_raw = (red_bright | red_dark).astype(np.uint8) * 255
#     red_u8 = _filter_foreground_blobs(red_raw, bgr.shape, bgr=bgr, slab_mode="red")
#     red_u8 = _remove_oversized_blobs(red_u8, bgr.shape, max_area_ratio=0.29)
#     red = red_u8 > 0

#     green_hsv = cv2.inRange(hsv, np.array([36, 42, 88]), np.array([86, 255, 255]))
#     green_rgb = (g >= 108) & (g > r + 18) & (g > b + 12)
#     green = (green_hsv > 0) & green_rgb

#     gold_hsv = cv2.inRange(hsv, np.array([12, 72, 105]), np.array([40, 255, 255]))
#     gold_rgb = (r >= 118) & (g >= 82) & (b <= 135) & (r > b + 22)
#     gold_raw = ((gold_hsv > 0) & gold_rgb).astype(np.uint8) * 255
#     gold = _filter_foreground_blobs(gold_raw, bgr.shape) > 0

#     cyan_hsv = cv2.inRange(hsv, np.array([82, 52, 95]), np.array([108, 255, 255]))

#     return {
#         "white": white,
#         "red": red,
#         "green": green,
#         "gold": gold,
#         "cyan": cyan_hsv > 0,
#     }


# def _protect_color_family(
#     seed: np.ndarray,
#     halo_rule: np.ndarray,
#     mud: np.ndarray | None = None,
#     close_iter: int = 2,
#     halo_px: float = 2.0,
# ) -> np.ndarray:
#     """Seed -> close gaps -> 1-2px color-guided halo for anti-alias edges only."""
#     seed_u8 = seed.astype(np.uint8) * 255
#     if not np.any(seed_u8):
#         return seed

#     closed = _close_mask(seed_u8, close_iter)
#     dist = cv2.distanceTransform((closed > 0).astype(np.uint8), cv2.DIST_L2, 3)
#     in_halo = (dist > 0) & (dist <= halo_px)
#     halo_ok = in_halo & halo_rule
#     if mud is not None:
#         halo_ok = halo_ok & ~mud
#     protected = (closed > 0) | halo_ok
#     return protected


# def _build_protected_foreground_mask(bgr: np.ndarray) -> np.ndarray:
#     """
#     Full OCR-safe protection mask.
#     Original RGB is kept for every protected pixel — no fade, no blur.
#     """
#     hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(hsv)
#     lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
#     lightness = lab[:, :, 0]
#     b, g, r = cv2.split(bgr)
#     families = _color_family_masks(bgr)
#     mud = _build_scenery_mask(h, s, v, r, g, b, level=0)

#     white_halo = ((v >= 118) & (s <= 80)) | ((lightness >= 150) & (s <= 55))
#     red_halo = (r >= 68) & (r > g + 14) & (r > b + 14) & (s >= 35)
#     green_halo = (g >= 95) & (g > r + 12) & (g > b + 10) & (s >= 25)
#     gold_halo = (r >= 100) & (g >= 78) & (b <= 135) & (s >= 42)

#     white_p = _protect_color_family(families["white"], white_halo, mud, close_iter=2, halo_px=2.5)
#     red_p = _protect_color_family(families["red"], red_halo, mud, close_iter=2, halo_px=2.5)
#     green_p = _protect_color_family(families["green"], green_halo, mud, close_iter=2, halo_px=2.0)
#     gold_p = _protect_color_family(families["gold"], gold_halo, mud, close_iter=1, halo_px=1.5)

#     cyan_u8 = families["cyan"].astype(np.uint8) * 255
#     cyan_u8 = _close_mask(cyan_u8, 1)

#     protected = white_p | red_p | green_p | gold_p | (cyan_u8 > 0)
#     return (protected.astype(np.uint8) * 255)


# def _background_only_mask(bgr: np.ndarray, protected_mask: np.ndarray, level: int) -> np.ndarray:
#     """
#     Mask used only to decide where background grayscale is applied.
#     May peel extra scenery per adaptive pass — never shrinks protected text.
#     """
#     hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(hsv)
#     b, g, r = cv2.split(bgr)

#     scenery = _build_scenery_mask(h, s, v, r, g, b, level)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

#     core = cv2.erode(protected_mask, kernel, iterations=1 + min(level // 4, 2))
#     fg = protected_mask.copy()
#     scenery_u8 = scenery.astype(np.uint8) * 255
#     scenery_u8 = cv2.dilate(scenery_u8, kernel, iterations=1 + level // 4)
#     fg[(scenery_u8 > 0) & (core == 0)] = 0
#     return fg


# def _create_neutral_background(bgr: np.ndarray, bg_mask: np.ndarray, strength: float) -> np.ndarray:
#     lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
#     l, a, b = cv2.split(lab)

#     pull = min(1.0, 0.55 + strength * 0.12)
#     a_out = a.copy()
#     b_out = b.copy()
#     a_out[bg_mask > 0] = a[bg_mask > 0] * (1.0 - pull) + 128.0 * pull
#     b_out[bg_mask > 0] = b[bg_mask > 0] * (1.0 - pull) + 128.0 * pull

#     l_out = l.copy()
#     bg_l = l[bg_mask > 0]
#     if bg_l.size > 0:
#         lo = float(np.percentile(bg_l, 6))
#         hi = float(np.percentile(bg_l, 94))
#         if hi > lo + 1:
#             stretched = (l[bg_mask > 0] - lo) * (255.0 / (hi - lo))
#             l_out[bg_mask > 0] = np.clip(stretched, 0, 255)
#         darken = max(0.28, 0.62 - strength * 0.06)
#         l_out[bg_mask > 0] = np.clip(l_out[bg_mask > 0] * darken, 0, 255)

#     merged = cv2.merge([l_out, a_out, b_out]).astype(np.uint8)
#     return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# def _composite_protected(bgr: np.ndarray, protected_mask: np.ndarray, bg_layer: np.ndarray) -> np.ndarray:
#     """
#     100% original pixels inside protection mask.
#     No blending, no feathering — text/icons stay exactly as captured.
#     """
#     fg = protected_mask > 0
#     return np.where(fg[..., np.newaxis], bgr, bg_layer)


# def grayscale_background_only(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
#     protected_mask = _build_protected_foreground_mask(bgr)
#     best = {
#         "result": bgr.copy(),
#         "protected_mask": protected_mask,
#         "bg_mask": cv2.bitwise_not(protected_mask),
#         "chroma_mean": 999.0,
#         "chroma_p90": 999.0,
#         "passes": 0,
#         "strength": 1.0,
#     }

#     for level in range(MAX_ADAPTIVE_PASSES):
#         peel_mask = _background_only_mask(bgr, protected_mask, level)
#         bg_mask = cv2.bitwise_not(peel_mask)
#         if not np.any(bg_mask):
#             break

#         strength = 1.0 + level * 0.32
#         bg_layer = _create_neutral_background(bgr, bg_mask, strength)
#         result = _composite_protected(bgr, protected_mask, bg_layer)
#         chroma_mean, chroma_p90 = _measure_background_chroma(result, bg_mask)
#         score = chroma_mean + chroma_p90 * 0.35

#         if score < best["chroma_mean"] + best["chroma_p90"] * 0.35:
#             best.update(
#                 result=result,
#                 bg_mask=bg_mask,
#                 chroma_mean=chroma_mean,
#                 chroma_p90=chroma_p90,
#                 passes=level + 1,
#                 strength=strength,
#             )

#         if chroma_mean <= CHROMA_TARGET_MEAN and chroma_p90 <= CHROMA_TARGET_P90:
#             break

#     meta = {
#         "chroma_mean": best["chroma_mean"],
#         "chroma_p90": best["chroma_p90"],
#         "passes": best["passes"],
#         "strength": best["strength"],
#         "protected_px": int(np.count_nonzero(protected_mask)),
#     }
#     return best["result"], protected_mask, best["bg_mask"], meta


# def default_output_path(input_path: str) -> str:
#     base, ext = os.path.splitext(input_path)
#     if not ext:
#         ext = ".png"
#     return f"{base}_grayscale_bg{ext}"


# def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Adaptive killfeed background neutralization; sharp protected text.",
#     )
#     parser.add_argument("input", help="Path to input killfeed image")
#     parser.add_argument("-o", "--output", help="Output path (default: <input>_grayscale_bg.png)")
#     parser.add_argument("--save-mask", metavar="PATH", help="Save protected foreground mask")
#     parser.add_argument("--save-bg-mask", metavar="PATH", help="Save background mask")
#     parser.add_argument("--show", action="store_true", help="Show before/after preview")
#     return parser.parse_args(argv)


# def main(argv: list[str] | None = None) -> int:
#     args = parse_args(argv)

#     input_path = os.path.abspath(args.input)
#     if not os.path.isfile(input_path):
#         print(f"Error: input file not found: {input_path}", file=sys.stderr)
#         return 1

#     output_path = os.path.abspath(args.output or default_output_path(input_path))

#     bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
#     if bgr is None:
#         print(f"Error: could not read image: {input_path}", file=sys.stderr)
#         return 1

#     result, protected_mask, bg_mask, meta = grayscale_background_only(bgr)

#     os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
#     if not cv2.imwrite(output_path, result, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
#         print(f"Error: could not write output: {output_path}", file=sys.stderr)
#         return 1

#     fg_pixels = int(np.count_nonzero(protected_mask))
#     bg_pixels = int(np.count_nonzero(bg_mask))
#     total = bgr.shape[0] * bgr.shape[1]
#     orig_mean, orig_p90 = _measure_background_chroma(bgr, bg_mask)

#     print(f"Input:           {input_path}")
#     print(f"Output:          {output_path}")
#     print(f"Size:            {bgr.shape[1]}x{bgr.shape[0]} (unchanged)")
#     print(f"Protected text:  {fg_pixels:,} px ({100.0 * fg_pixels / total:.1f}%)")
#     print(f"Background:      {bg_pixels:,} px ({100.0 * bg_pixels / total:.1f}%)")
#     print(f"Adaptive passes: {meta['passes']}")
#     print(f"Strength:        {meta['strength']:.2f}")
#     print(
#         f"BG chroma mean:  {orig_mean:.1f} -> {meta['chroma_mean']:.1f} "
#         f"(target <={CHROMA_TARGET_MEAN})"
#     )
#     print(
#         f"BG chroma p90:   {orig_p90:.1f} -> {meta['chroma_p90']:.1f} "
#         f"(target <={CHROMA_TARGET_P90})"
#     )

#     if args.save_mask:
#         cv2.imwrite(os.path.abspath(args.save_mask), protected_mask)
#     if args.save_bg_mask:
#         cv2.imwrite(os.path.abspath(args.save_bg_mask), bg_mask)

#     if args.show:
#         combined = np.hstack([bgr, result])
#         cv2.imshow("Before (left) | After (right)", combined)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main())












#!/usr/bin/env python3
"""
Killfeed background -> adaptive neutral grayscale (standalone).

Strong B&W background + sharp protected foreground (original RGB, OCR-safe).

Usage:
    python killfeed_grayscale_bg.py killfeed.png
    python killfeed_grayscale_bg.py killfeed.png -o killfeed_grayscale_bg.png
    python killfeed_grayscale_bg.py --folder killfeed_images
    python killfeed_grayscale_bg.py --folder killfeed_images --output-dir killfeed_images_output
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

CHROMA_TARGET_MEAN = 5.0
CHROMA_TARGET_P90 = 10.0
MAX_ADAPTIVE_PASSES = 14

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_INPUT_FOLDER = "killfeed_images"
DEFAULT_OUTPUT_FOLDER = "killfeed_images_output"


def _background_chroma_map(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)
    return np.sqrt((a - 128.0) ** 2 + (b - 128.0) ** 2)


def _measure_background_chroma(bgr: np.ndarray, bg_mask: np.ndarray) -> tuple[float, float]:
    if not np.any(bg_mask):
        return 0.0, 0.0
    chroma = _background_chroma_map(bgr)[bg_mask > 0]
    return float(np.mean(chroma)), float(np.percentile(chroma, 90))


def _build_scenery_mask(
    h: np.ndarray,
    s: np.ndarray,
    v: np.ndarray,
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    level: int,
) -> np.ndarray:
    rg_close = np.abs(r.astype(np.int16) - g.astype(np.int16)) < 42
    sat_floor = max(6, 28 - level * 2)

    brown_tan = (
        (h >= 3) & (h <= 35) &
        (s >= sat_floor) & (v >= 28) & (v <= 210) &
        rg_close
    )
    olive = (
        (h >= 16) & (h <= 58) &
        (s >= sat_floor) & (v >= 32) & (v <= 200) &
        rg_close
    )
    red_env = (
        (h <= 22) &
        (s >= sat_floor) &
        (v >= 30) & (v <= 220) &
        (r.astype(np.int16) >= g.astype(np.int16) - 8)
    )
    dark_neutral = (s <= max(20, 55 - level * 3)) & (v >= 20) & (v <= 170)
    return brown_tan | olive | red_env | dark_neutral


def _dilate_mask(mask_u8: np.ndarray, iterations: int = 1) -> np.ndarray:
    if iterations <= 0:
        return mask_u8
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(mask_u8, kernel, iterations=iterations)


def _close_mask(mask_u8: np.ndarray, iterations: int = 1) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def _remove_oversized_blobs(
    mask_u8: np.ndarray,
    img_shape: tuple[int, ...],
    max_area_ratio: float = 0.29,
) -> np.ndarray:
    """Remove only huge scenery blobs; keep medium-sized text regions."""
    if not np.any(mask_u8):
        return mask_u8

    h, w = img_shape[:2]
    max_area = int(h * w * max_area_ratio)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    kept = np.zeros_like(mask_u8)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > max_area:
            continue
        kept[labels == i] = 255
    return kept


def _extract_strokes_from_wide_slab(
    component: np.ndarray,
    bgr: np.ndarray,
    stroke_rule: np.ndarray,
) -> np.ndarray:
    """Pull text strokes out of a wide red/scenery slab connected component."""
    stroke = component & stroke_rule
    stroke_u8 = stroke.astype(np.uint8) * 255
    if not np.any(stroke_u8):
        return stroke_u8
    stroke_u8 = _close_mask(stroke_u8, 2)
    return _dilate_mask(stroke_u8, 1)


def _filter_foreground_blobs(
    mask_u8: np.ndarray,
    img_shape: tuple[int, ...],
    bgr: np.ndarray | None = None,
    slab_mode: str | None = None,
) -> np.ndarray:
    """Drop large scenery blobs; recover text strokes from wide slabs."""
    if not np.any(mask_u8):
        return mask_u8

    h, w = img_shape[:2]
    max_area = int(h * w * 0.22)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    kept = np.zeros_like(mask_u8)

    stroke_rule = None
    if slab_mode == "red" and bgr is not None:
        b, g, r = cv2.split(bgr)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        stroke_rule = (r > g + 34) & (r > b + 34) & (s >= 58)

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        component = labels == i

        if bw > int(w * 0.72) and slab_mode == "red" and stroke_rule is not None:
            slab_strokes = _extract_strokes_from_wide_slab(component, bgr, stroke_rule)
            kept = np.maximum(kept, slab_strokes)
            continue

        if area > max_area:
            continue
        if bw > int(w * 0.82) and bh > int(h * 0.45):
            continue
        kept[component] = 255
    return kept


def _color_family_masks(bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Per-color seed masks (bool arrays)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]
    b, g, r = cv2.split(bgr)

    white = ((v >= 148) & (s <= 78)) | (lightness >= 178)

    red_bright = (r >= 105) & (r > g + 48) & (r > b + 48) & (s >= 82)
    red_dark = (r >= 78) & (r > g + 26) & (r > b + 26) & (s >= 52) & (v >= 36) & (v <= 165)
    red_raw = (red_bright | red_dark).astype(np.uint8) * 255
    red_u8 = _filter_foreground_blobs(red_raw, bgr.shape, bgr=bgr, slab_mode="red")
    red_u8 = _remove_oversized_blobs(red_u8, bgr.shape, max_area_ratio=0.29)
    red = red_u8 > 0

    green_hsv = cv2.inRange(hsv, np.array([36, 42, 88]), np.array([86, 255, 255]))
    green_rgb = (g >= 108) & (g > r + 18) & (g > b + 12)
    green = (green_hsv > 0) & green_rgb

    gold_hsv = cv2.inRange(hsv, np.array([12, 72, 105]), np.array([40, 255, 255]))
    gold_rgb = (r >= 118) & (g >= 82) & (b <= 135) & (r > b + 22)
    gold_raw = ((gold_hsv > 0) & gold_rgb).astype(np.uint8) * 255
    gold = _filter_foreground_blobs(gold_raw, bgr.shape) > 0

    cyan_hsv = cv2.inRange(hsv, np.array([82, 52, 95]), np.array([108, 255, 255]))

    return {
        "white": white,
        "red": red,
        "green": green,
        "gold": gold,
        "cyan": cyan_hsv > 0,
    }


def _protect_color_family(
    seed: np.ndarray,
    halo_rule: np.ndarray,
    mud: np.ndarray | None = None,
    close_iter: int = 2,
    halo_px: float = 2.0,
) -> np.ndarray:
    """Seed -> close gaps -> 1-2px color-guided halo for anti-alias edges only."""
    seed_u8 = seed.astype(np.uint8) * 255
    if not np.any(seed_u8):
        return seed

    closed = _close_mask(seed_u8, close_iter)
    dist = cv2.distanceTransform((closed > 0).astype(np.uint8), cv2.DIST_L2, 3)
    in_halo = (dist > 0) & (dist <= halo_px)
    halo_ok = in_halo & halo_rule
    if mud is not None:
        halo_ok = halo_ok & ~mud
    protected = (closed > 0) | halo_ok
    return protected


def _build_protected_foreground_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Full OCR-safe protection mask.
    Original RGB is kept for every protected pixel — no fade, no blur.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]
    b, g, r = cv2.split(bgr)
    families = _color_family_masks(bgr)
    mud = _build_scenery_mask(h, s, v, r, g, b, level=0)

    white_halo = ((v >= 118) & (s <= 80)) | ((lightness >= 150) & (s <= 55))
    red_halo = (r >= 68) & (r > g + 14) & (r > b + 14) & (s >= 35)
    green_halo = (g >= 95) & (g > r + 12) & (g > b + 10) & (s >= 25)
    gold_halo = (r >= 100) & (g >= 78) & (b <= 135) & (s >= 42)

    white_p = _protect_color_family(families["white"], white_halo, mud, close_iter=2, halo_px=2.5)
    red_p = _protect_color_family(families["red"], red_halo, mud, close_iter=2, halo_px=2.5)
    green_p = _protect_color_family(families["green"], green_halo, mud, close_iter=2, halo_px=2.0)
    gold_p = _protect_color_family(families["gold"], gold_halo, mud, close_iter=1, halo_px=1.5)

    cyan_u8 = families["cyan"].astype(np.uint8) * 255
    cyan_u8 = _close_mask(cyan_u8, 1)

    protected = white_p | red_p | green_p | gold_p | (cyan_u8 > 0)
    return (protected.astype(np.uint8) * 255)


def _background_only_mask(bgr: np.ndarray, protected_mask: np.ndarray, level: int) -> np.ndarray:
    """
    Mask used only to decide where background grayscale is applied.
    May peel extra scenery per adaptive pass — never shrinks protected text.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    b, g, r = cv2.split(bgr)

    scenery = _build_scenery_mask(h, s, v, r, g, b, level)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    core = cv2.erode(protected_mask, kernel, iterations=1 + min(level // 4, 2))
    fg = protected_mask.copy()
    scenery_u8 = scenery.astype(np.uint8) * 255
    scenery_u8 = cv2.dilate(scenery_u8, kernel, iterations=1 + level // 4)
    fg[(scenery_u8 > 0) & (core == 0)] = 0
    return fg


def _create_neutral_background(bgr: np.ndarray, bg_mask: np.ndarray, strength: float) -> np.ndarray:
    """
    Enhanced neutral background: full desaturation + CLAHE contrast on bg only.
    Foreground composite happens later — text colors are never touched here.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0]

    clip = 2.2 + min(strength, 5.0) * 0.35
    tile = max(4, min(8, int(min(bgr.shape[:2]) / 12)))
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    gray_boost = clahe.apply(gray)

    l_out = gray_boost.astype(np.float32)
    bg_l = l[bg_mask > 0]
    if bg_l.size > 0:
        lo = float(np.percentile(bg_l, 5))
        hi = float(np.percentile(bg_l, 95))
        if hi > lo + 1:
            stretched = (l[bg_mask > 0] - lo) * (255.0 / (hi - lo))
            l_out[bg_mask > 0] = np.clip(stretched, 0, 255)
        darken = max(0.32, 0.58 - strength * 0.05)
        l_out[bg_mask > 0] = np.clip(l_out[bg_mask > 0] * darken, 0, 255)

    neutral = cv2.merge([
        l_out,
        np.full_like(l, 128.0),
        np.full_like(l, 128.0),
    ]).astype(np.uint8)
    out = cv2.cvtColor(neutral, cv2.COLOR_LAB2BGR)

    # Pure gray copy guarantees zero color cast on background pixels.
    gray_bgr = cv2.cvtColor(gray_boost, cv2.COLOR_GRAY2BGR)
    boosted = cv2.convertScaleAbs(
        gray_bgr,
        alpha=1.0 + min(strength, 4.0) * 0.04,
        beta=-6.0 - min(strength, 4.0) * 1.5,
    )
    out[bg_mask > 0] = boosted[bg_mask > 0]
    return out


def _composite_protected(bgr: np.ndarray, protected_mask: np.ndarray, bg_layer: np.ndarray) -> np.ndarray:
    """
    100% original pixels inside protection mask.
    No blending, no feathering — text/icons stay exactly as captured.
    """
    fg = protected_mask > 0
    return np.where(fg[..., np.newaxis], bgr, bg_layer)


def grayscale_background_only(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    protected_mask = _build_protected_foreground_mask(bgr)
    best = {
        "result": bgr.copy(),
        "protected_mask": protected_mask,
        "bg_mask": cv2.bitwise_not(protected_mask),
        "chroma_mean": 999.0,
        "chroma_p90": 999.0,
        "passes": 0,
        "strength": 1.0,
    }

    for level in range(MAX_ADAPTIVE_PASSES):
        peel_mask = _background_only_mask(bgr, protected_mask, level)
        bg_mask = cv2.bitwise_not(peel_mask)
        if not np.any(bg_mask):
            break

        strength = 1.0 + level * 0.32
        bg_layer = _create_neutral_background(bgr, bg_mask, strength)
        result = _composite_protected(bgr, protected_mask, bg_layer)
        chroma_mean, chroma_p90 = _measure_background_chroma(result, bg_mask)
        score = chroma_mean + chroma_p90 * 0.35

        if score < best["chroma_mean"] + best["chroma_p90"] * 0.35:
            best.update(
                result=result,
                bg_mask=bg_mask,
                chroma_mean=chroma_mean,
                chroma_p90=chroma_p90,
                passes=level + 1,
                strength=strength,
            )

        if chroma_mean <= CHROMA_TARGET_MEAN and chroma_p90 <= CHROMA_TARGET_P90:
            break

    meta = {
        "chroma_mean": best["chroma_mean"],
        "chroma_p90": best["chroma_p90"],
        "passes": best["passes"],
        "strength": best["strength"],
        "protected_px": int(np.count_nonzero(protected_mask)),
    }
    return best["result"], protected_mask, best["bg_mask"], meta


def flat_panel_background(
    bgr: np.ndarray,
    panel_value: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Killfeed-style flat panel: uniform dark gray bar, original text/icon colors.
    No texture, no color bleed — clean OCR card look.
    """
    protected_mask = _build_protected_foreground_mask(bgr)
    panel = np.full_like(bgr, (panel_value, panel_value, panel_value))
    result = _composite_protected(bgr, protected_mask, panel)
    bg_mask = cv2.bitwise_not(protected_mask)
    chroma_mean, chroma_p90 = _measure_background_chroma(result, bg_mask)
    meta = {
        "method": "flat_panel",
        "panel_value": panel_value,
        "chroma_mean": chroma_mean,
        "chroma_p90": chroma_p90,
        "protected_px": int(np.count_nonzero(protected_mask)),
    }
    return result, protected_mask, bg_mask, meta


def luminance_keep_background(
    bgr: np.ndarray,
    bg_value: int = 36,
    bright_thresh: int = 138,
    chroma_thresh: float = 26.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Keep bright or strongly colored pixels; replace everything else with flat gray.
    Simpler mask — good when killfeed sits on game scenery.
    """
    protected_mask = _build_protected_foreground_mask(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    chroma = _background_chroma_map(bgr)
    lum_keep = ((gray >= bright_thresh) | (chroma >= chroma_thresh)).astype(np.uint8) * 255
    lum_keep = _close_mask(lum_keep, 1)
    lum_keep = _dilate_mask(lum_keep, 1)
    combined = cv2.bitwise_or(protected_mask, lum_keep)
    panel = np.full_like(bgr, (bg_value, bg_value, bg_value))
    result = _composite_protected(bgr, combined, panel)
    bg_mask = cv2.bitwise_not(combined)
    chroma_mean, chroma_p90 = _measure_background_chroma(result, bg_mask)
    meta = {
        "method": "luminance",
        "bg_value": bg_value,
        "chroma_mean": chroma_mean,
        "chroma_p90": chroma_p90,
        "protected_px": int(np.count_nonzero(combined)),
    }
    return result, protected_mask, bg_mask, meta


# Reference killfeed bar ratio (VIND.KINGSTN knock strip).
REFERENCE_KILLFEED_SIZE = (581, 50)
REFERENCE_ASPECT_RATIO = REFERENCE_KILLFEED_SIZE[0] / REFERENCE_KILLFEED_SIZE[1]


def get_reference_size(reference_path: str | Path | None = None) -> tuple[int, int]:
    """Read target W×H from a reference image, else use REFERENCE_KILLFEED_SIZE."""
    if reference_path is not None:
        ref = Path(reference_path)
        if ref.is_file():
            bgr = cv2.imread(str(ref), cv2.IMREAD_COLOR)
            if bgr is not None:
                return bgr.shape[1], bgr.shape[0]
    return REFERENCE_KILLFEED_SIZE


def compute_target_size(
    source_w: int,
    source_h: int,
    *,
    ref_w: int = REFERENCE_KILLFEED_SIZE[0],
    ref_h: int = REFERENCE_KILLFEED_SIZE[1],
) -> tuple[int, int]:
    """
    Target canvas keeps reference aspect ratio.
    Width is at least ref_w and source_w so text is never clipped.
    """
    ratio = ref_w / ref_h
    target_w = max(ref_w, source_w)
    target_h = max(ref_h, int(round(target_w / ratio)))
    return target_w, target_h


def resize_background_only(
    bgr: np.ndarray,
    protected_mask: np.ndarray,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Stretch background to target ratio/size; paste text/icons at original pixel size.
    """
    h, w = bgr.shape[:2]
    if target_size is None:
        target_size = compute_target_size(w, h)
    tw, th = target_size

    bg = bgr.copy()
    if np.any(protected_mask):
        bg_pixels = bgr[protected_mask == 0]
        if bg_pixels.size:
            fill = np.median(bg_pixels, axis=0).astype(np.uint8)
        else:
            fill = np.array([40, 40, 40], dtype=np.uint8)
        bg[protected_mask > 0] = fill

    canvas = cv2.resize(bg, (tw, th), interpolation=cv2.INTER_LINEAR)

    fg = protected_mask > 0
    if not np.any(fg):
        return canvas

    oy = max(0, (th - h) // 2)
    ox = 0
    ys, xs = np.where(fg)
    dy = oy + ys
    dx = ox + xs
    valid = (dy >= 0) & (dy < th) & (dx >= 0) & (dx < tw)
    canvas[dy[valid], dx[valid]] = bgr[ys[valid], xs[valid]]
    return canvas


def scale_killfeed_crop(
    bgr: np.ndarray,
    reference_size: tuple[int, int] | None = None,
    pad_value: int = 40,
) -> np.ndarray:
    """
    Uniform scale + letterbox to reference killfeed aspect ratio (no grayscale).
    Avoids the background-stretch / foreground-paste distortion from resize_background_only.
    """
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return bgr
    ref_w, ref_h = reference_size or REFERENCE_KILLFEED_SIZE
    h, w = bgr.shape[:2]
    tw, th = compute_target_size(w, h, ref_w=ref_w, ref_h=ref_h)
    scale = min(tw / float(w), th / float(h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    scaled = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((th, tw, 3), pad_value, dtype=np.uint8)
    ox = (tw - nw) // 2
    oy = (th - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = scaled
    return canvas


GRAYSCALE_METHODS: dict[str, object] = {
    "adaptive": grayscale_background_only,
    "flat_panel": flat_panel_background,
    "luminance": luminance_keep_background,
}


def preprocess_for_ocr(
    bgr: np.ndarray,
    method: str = "adaptive",
    scale_background: bool = True,
    reference_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    OCR-ready killfeed crop: adaptive grayscale + optional reference-ratio scaling.
    Text/icons stay original RGB; background neutralized (matches script.py flow).
    """
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return bgr
    fn = GRAYSCALE_METHODS.get(method, grayscale_background_only)
    result, protected_mask, _, _ = fn(bgr)
    if scale_background:
        ref = reference_size or REFERENCE_KILLFEED_SIZE
        target = compute_target_size(
            bgr.shape[1],
            bgr.shape[0],
            ref_w=ref[0],
            ref_h=ref[1],
        )
        result = resize_background_only(result, protected_mask, target)
    return result


def weapon_icon_hint(bgr: np.ndarray) -> str:
    """
    Detect weapon icon category from center band colors (grey_scale masks).
    Fixes vehicle kills mislabeled as gun knockout.
    """
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return "unknown"
    h, w = bgr.shape[:2]
    icon = bgr[int(h * 0.15) : int(h * 0.85), int(w * 0.28) : int(w * 0.56)]
    if icon.size == 0:
        return "unknown"
    families = _color_family_masks(icon)
    total = max(icon.shape[0] * icon.shape[1], 1)
    cyan_r = float(np.count_nonzero(families["cyan"])) / total
    gold_r = float(np.count_nonzero(families["gold"])) / total
    green_r = float(np.count_nonzero(families["green"])) / total
    # Truck/vehicle icons: strong cyan or gold, little green
    if cyan_r > 0.035:
        return "vehicle"
    if gold_r > 0.055 and green_r < 0.025:
        return "vehicle"
    if gold_r > 0.10 and cyan_r > 0.02:
        return "vehicle"
    return "unknown"


def victim_text_color_hint(bgr: np.ndarray) -> str:
    """
    Read victim name-band color: white knock, red kill, green revive.
    Uses the same color-family masks as OCR preprocessing.
    """
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return "unknown"
    h, w = bgr.shape[:2]
    # Victim letters sit right of center; exclude far-right knock icon.
    band = bgr[:, int(w * 0.50) : int(w * 0.68)]
    if band.size == 0:
        band = bgr
    families = _color_family_masks(band)
    total = max(band.shape[0] * band.shape[1], 1)
    red_r = float(np.count_nonzero(families["red"])) / total
    white_r = float(np.count_nonzero(families["white"])) / total
    green_r = float(np.count_nonzero(families["green"])) / total
    if green_r > 0.018 and green_r >= red_r:
        return "revive"
    if red_r > 0.022 and red_r > white_r * 1.15:
        return "elimination"
    if white_r > 0.015:
        return "knock"
    return "unknown"


def default_output_path(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    if not ext:
        ext = ".png"
    return f"{base}_grayscale_bg{ext}"


def list_images(folder: str | Path) -> list[Path]:
    root = Path(folder)
    if not root.is_dir():
        return []
    files = [
        p for p in sorted(root.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return files


def process_image(
    bgr: np.ndarray,
    *,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Run grayscale background pipeline on an in-memory BGR image."""
    return grayscale_background_only(bgr)


def process_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    save_mask: str | Path | None = None,
    save_bg_mask: str | Path | None = None,
    verbose: bool = True,
) -> dict | None:
    """Process one image file. Returns metadata dict, or None on failure."""
    input_path = Path(input_path)
    if not input_path.is_file():
        if verbose:
            print(f"Skip (not found): {input_path}", file=sys.stderr)
        return None

    output_path = Path(output_path or default_output_path(str(input_path)))
    bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if bgr is None:
        if verbose:
            print(f"Skip (unreadable): {input_path}", file=sys.stderr)
        return None

    result, protected_mask, bg_mask, meta = process_image(bgr, verbose=verbose)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        if verbose:
            print(f"Skip (write failed): {output_path}", file=sys.stderr)
        return None

    fg_pixels = int(np.count_nonzero(protected_mask))
    bg_pixels = int(np.count_nonzero(bg_mask))
    total = bgr.shape[0] * bgr.shape[1]
    orig_mean, orig_p90 = _measure_background_chroma(bgr, bg_mask)

    if save_mask:
        cv2.imwrite(str(save_mask), protected_mask)
    if save_bg_mask:
        cv2.imwrite(str(save_bg_mask), bg_mask)

    if verbose:
        print(f"Input:           {input_path}")
        print(f"Output:          {output_path}")
        print(f"Size:            {bgr.shape[1]}x{bgr.shape[0]} (unchanged)")
        print(f"Protected text:  {fg_pixels:,} px ({100.0 * fg_pixels / total:.1f}%)")
        print(f"Background:      {bg_pixels:,} px ({100.0 * bg_pixels / total:.1f}%)")
        print(f"Adaptive passes: {meta['passes']}")
        print(f"Strength:        {meta['strength']:.2f}")
        print(
            f"BG chroma mean:  {orig_mean:.1f} -> {meta['chroma_mean']:.1f} "
            f"(target <={CHROMA_TARGET_MEAN})"
        )
        print(
            f"BG chroma p90:   {orig_p90:.1f} -> {meta['chroma_p90']:.1f} "
            f"(target <={CHROMA_TARGET_P90})"
        )

    return {
        "input": str(input_path),
        "output": str(output_path),
        "meta": meta,
        "fg_pixels": fg_pixels,
        "bg_pixels": bg_pixels,
    }


def process_folder(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    verbose: bool = True,
) -> list[dict]:
    """Process every image in a folder. Outputs go to output_dir."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir or DEFAULT_OUTPUT_FOLDER)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(input_dir)
    if not images:
        print(f"No images found in: {input_dir}")
        print(f"Drop killfeed PNG/JPG files into: {input_dir.resolve()}")
        return []

    print(f"Input folder:  {input_dir.resolve()}")
    print(f"Output folder: {output_dir.resolve()}")
    print(f"Found {len(images)} image(s)\n")

    results: list[dict] = []
    for idx, image_path in enumerate(images, start=1):
        if verbose:
            print(f"[{idx}/{len(images)}] {image_path.name}")
        out_name = f"{image_path.stem}_grayscale_bg.png"
        out_path = output_dir / out_name
        row = process_file(image_path, out_path, verbose=verbose)
        if row:
            results.append(row)
        if verbose and idx < len(images):
            print()

    print(f"\nDone: {len(results)}/{len(images)} processed -> {output_dir.resolve()}")
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive killfeed background neutralization; sharp protected text.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to one killfeed image (omit when using --folder)",
    )
    parser.add_argument("-o", "--output", help="Output path for single image mode")
    parser.add_argument(
        "--folder",
        metavar="DIR",
        help=f"Process all images in folder (default input: {DEFAULT_INPUT_FOLDER})",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=DEFAULT_OUTPUT_FOLDER,
        help=f"Output folder for batch mode (default: {DEFAULT_OUTPUT_FOLDER})",
    )
    parser.add_argument("--save-mask", metavar="PATH", help="Save protected foreground mask")
    parser.add_argument("--save-bg-mask", metavar="PATH", help="Save background mask")
    parser.add_argument("--show", action="store_true", help="Show before/after preview")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.folder is not None or args.input is None:
        script_dir = Path(__file__).resolve().parent
        input_dir = Path(args.folder) if args.folder else script_dir / DEFAULT_INPUT_FOLDER
        if not input_dir.is_absolute():
            input_dir = script_dir / input_dir
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = script_dir / output_dir
        input_dir.mkdir(parents=True, exist_ok=True)
        results = process_folder(input_dir, output_dir)
        return 0 if results else 1

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    output_path = os.path.abspath(args.output or default_output_path(input_path))

    bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"Error: could not read image: {input_path}", file=sys.stderr)
        return 1

    result, protected_mask, bg_mask, meta = grayscale_background_only(bgr)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if not cv2.imwrite(output_path, result, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        print(f"Error: could not write output: {output_path}", file=sys.stderr)
        return 1

    fg_pixels = int(np.count_nonzero(protected_mask))
    bg_pixels = int(np.count_nonzero(bg_mask))
    total = bgr.shape[0] * bgr.shape[1]
    orig_mean, orig_p90 = _measure_background_chroma(bgr, bg_mask)

    print(f"Input:           {input_path}")
    print(f"Output:          {output_path}")
    print(f"Size:            {bgr.shape[1]}x{bgr.shape[0]} (unchanged)")
    print(f"Protected text:  {fg_pixels:,} px ({100.0 * fg_pixels / total:.1f}%)")
    print(f"Background:      {bg_pixels:,} px ({100.0 * bg_pixels / total:.1f}%)")
    print(f"Adaptive passes: {meta['passes']}")
    print(f"Strength:        {meta['strength']:.2f}")
    print(
        f"BG chroma mean:  {orig_mean:.1f} -> {meta['chroma_mean']:.1f} "
        f"(target <={CHROMA_TARGET_MEAN})"
    )
    print(
        f"BG chroma p90:   {orig_p90:.1f} -> {meta['chroma_p90']:.1f} "
        f"(target <={CHROMA_TARGET_P90})"
    )

    if args.save_mask:
        cv2.imwrite(os.path.abspath(args.save_mask), protected_mask)
    if args.save_bg_mask:
        cv2.imwrite(os.path.abspath(args.save_bg_mask), bg_mask)

    if args.show:
        combined = np.hstack([bgr, result])
        cv2.imshow("Before (left) | After (right)", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
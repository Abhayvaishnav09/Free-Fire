"""Killfeed grayscale OCR preprocessing."""

from gray_scale.grey_scale import (
    GRAYSCALE_METHODS,
    IMAGE_EXTENSIONS,
    compute_target_size,
    get_reference_size,
    grayscale_background_only,
    flat_panel_background,
    preprocess_for_ocr,
    resize_background_only,
    scale_killfeed_crop,
    victim_text_color_hint,
    weapon_icon_hint,
)

__all__ = [
    "GRAYSCALE_METHODS",
    "IMAGE_EXTENSIONS",
    "compute_target_size",
    "get_reference_size",
    "grayscale_background_only",
    "flat_panel_background",
    "preprocess_for_ocr",
    "resize_background_only",
    "scale_killfeed_crop",
    "victim_text_color_hint",
    "weapon_icon_hint",
]

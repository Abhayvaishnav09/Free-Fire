"""Strip parser: OCR each row → signature list top→bottom."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol

from killfeed.classifier import ClassificationResult, classify_row, is_confident_classification
from killfeed.queue_state import KillfeedEvent, make_signature
from killfeed.roi import StripROI


class OCRProvider(Protocol):
    def extract_row_names(self, row_crop) -> tuple[str, str, float]: ...


@dataclass
class ParsedRow:
    killer: str
    victim: str
    event_type: str
    kill_type: str
    canonical: str
    tms_status: str
    weapon: str
    gun_name: str
    confidence: float
    position: int
    row_crop: object
    signature: str
    event: KillfeedEvent


def _row_valid(
    killer: str,
    victim: str,
    confidence: float,
    classification: ClassificationResult,
    name_validator: Optional[Callable[[str, str, float], bool]],
) -> bool:
    """Victim required; killer required except playzone/powerzone."""
    if not victim or len(victim.strip()) < 2:
        return False
    if classification.kill_type in ("playzone", "powerzone"):
        if name_validator:
            return name_validator("", victim, confidence) or name_validator(victim, victim, confidence)
        return True
    if not killer or len(killer.strip()) < 2:
        return False
    if name_validator and not name_validator(killer, victim, confidence):
        return False
    return True


def parse_row(
    row,
    position: int,
    ocr: OCRProvider,
    name_validator: Optional[Callable[[str, str, float], bool]] = None,
) -> Optional[ParsedRow]:
    """OCR + classify a single killfeed row."""
    killer, victim, confidence = ocr.extract_row_names(row.crop)
    classification = classify_row(row.crop, killer=killer, victim=victim, yolo_class=row.class_name)
    if not is_confident_classification(classification):
        return None
    if not _row_valid(killer, victim, confidence, classification, name_validator):
        return None

    display_killer = killer if killer else ""
    sig = make_signature(display_killer, victim, classification.canonical, classification.icon_category)
    event = KillfeedEvent(
        killer=display_killer,
        victim=victim,
        event=classification.canonical,
        event_type=classification.event_type,
        kill_type=classification.kill_type,
        weapon=classification.icon_category,
        gun_name=classification.gun_name,
        position=position,
    )
    return ParsedRow(
        killer=display_killer,
        victim=victim,
        event_type=classification.event_type,
        kill_type=classification.kill_type,
        canonical=classification.canonical,
        tms_status=classification.tms_status,
        weapon=classification.icon_category,
        gun_name=classification.gun_name,
        confidence=confidence,
        position=position,
        row_crop=row.crop,
        signature=sig,
        event=event,
    )


def parse_strip(
    roi: StripROI,
    ocr: OCRProvider,
    name_validator: Optional[Callable[[str, str, float], bool]] = None,
    row0_first: bool = True,
) -> List[ParsedRow]:
    """OCR + visual classify each row; skip incomplete rows."""
    if not roi.rows:
        return []

    order = list(range(len(roi.rows)))
    if row0_first and len(order) > 1:
        order = [0] + [i for i in order if i != 0]

    parsed: List[ParsedRow] = []
    by_pos: dict[int, ParsedRow] = {}
    for i in order:
        pr = parse_row(roi.rows[i], i, ocr, name_validator)
        if pr is not None:
            by_pos[i] = pr

    for i in sorted(by_pos):
        parsed.append(by_pos[i])
    return parsed


def signatures_from_rows(rows: List[ParsedRow]) -> List[str]:
    return [r.signature for r in rows]

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from killfeed.dhash import compute_dhash, dhash_distance

@dataclass
class TrackedRow:
    visual_dhash: str           # Primary identity (never changes)
    arrival_seq: int            # Monotonic, assigned at first detection
    first_frame: int            # Frame number where first seen
    first_timestamp: float      # Wall-clock time of first detection
    last_seen_frame: int        # Most recent frame containing this row
    slot_history: List[int]     # [0, 1, 2, 3] — positions over time
    best_crop: np.ndarray       # Sharpest crop (highest Laplacian score)
    best_crop_score: float      # Laplacian variance of best_crop
    
    # OCR state
    ocr_done: bool = False
    ocr_killer: str = ""
    ocr_victim: str = ""
    ocr_confidence: float = 0.0
    ocr_class: str = ""         # YOLO class at OCR time
    
    # Classification
    canonical: str = ""         # e.g. "normal_knock"
    tms_status: str = ""        # e.g. "gun-knock"
    event_hash: str = ""
    
    # Emit state
    emitted: bool = False
    emit_seq: int = -1          # TMS sequence number

class RowLedger:
    def __init__(self, dhash_threshold: int = 6, ttl_frames: int = 300):
        self._rows: Dict[str, TrackedRow] = {}  # dhash -> TrackedRow
        self._arrival_counter = 0
        self._dhash_threshold = dhash_threshold
        self._ttl_frames = ttl_frames
    
    def register_frame(self, rows, frame_num: int, timestamp: float) -> List[int]:
        """Returns indices of NEW rows not seen before."""
        new_indices = []
        current_hashes = [compute_dhash(r.crop) for r in rows]
        
        for idx, (row, dhash) in enumerate(zip(rows, current_hashes)):
            if not dhash:
                continue
            
            # Find best match in active ledger
            match_key = self._find_match(dhash, frame_num)
            
            if match_key:
                # Known row — update position and maybe crop
                tracked = self._rows[match_key]
                tracked.last_seen_frame = frame_num
                tracked.slot_history.append(idx)
                
                # Keep sharpest crop for OCR
                sharpness = 1.0
                if sharpness >= tracked.best_crop_score:
                    tracked.best_crop = row.crop.copy()
                    tracked.best_crop_score = sharpness
                    tracked.ocr_class = row.class_name  # update YOLO class too
            else:
                # New row!
                self._arrival_counter += 1
                sharpness = 1.0
                
                self._rows[dhash] = TrackedRow(
                    visual_dhash=dhash,
                    arrival_seq=self._arrival_counter,
                    first_frame=frame_num,
                    first_timestamp=timestamp,
                    last_seen_frame=frame_num,
                    slot_history=[idx],
                    best_crop=row.crop.copy(),
                    best_crop_score=sharpness,
                    ocr_class=row.class_name,
                )
                new_indices.append(idx)
        
        self._prune(frame_num)
        return new_indices
    
    def _find_match(self, dhash: str, current_frame: int) -> Optional[str]:
        """Find the closest active ledger entry by dHash."""
        best_key = None
        best_dist = self._dhash_threshold + 1
        
        for key, tracked in self._rows.items():
            if current_frame - tracked.last_seen_frame > self._ttl_frames:
                continue  # expired
            dist = dhash_distance(dhash, key)
            if dist < best_dist:
                best_dist = dist
                best_key = key
        
        return best_key if best_dist <= self._dhash_threshold else None
        
    def _prune(self, current_frame: int):
        expired = [k for k, v in self._rows.items() if current_frame - v.last_seen_frame > self._ttl_frames]
        for k in expired:
            del self._rows[k]
            
    def oldest_ready_to_emit(self) -> Optional[TrackedRow]:
        """Returns the oldest un-emitted row that has finished OCR."""
        # Active rows only
        ready = [r for r in self._rows.values() if not r.emitted and r.ocr_done]
        if not ready:
            return None
        
        # We must ONLY return if it is the OLDEST un-emitted row overall.
        # Otherwise we break chronology. Let's find the minimum arrival_seq
        # among all un-emitted rows.
        unemitted = [r for r in self._rows.values() if not r.emitted]
        if not unemitted:
            return None
            
        oldest_unemitted = min(unemitted, key=lambda r: r.arrival_seq)
        
        if oldest_unemitted.ocr_done:
            return oldest_unemitted
        return None

    def mark_emitted(self, dhash: str):
        if dhash in self._rows:
            self._rows[dhash].emitted = True

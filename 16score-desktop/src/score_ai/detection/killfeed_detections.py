"""
Killfeed Detections Manager

Manages storage and retrieval of kill feed detection data for match scoring.
Provides persistence of detection results to JSON files organized by match.

Usage:
    from score_ai.detection.killfeed_detections import KillfeedDetections
    
    detections = KillfeedDetections("match_001")
    detections.add_detection({"class_name": "kill-block", "confidence": 0.95})

Example:
    >>> kf = KillfeedDetections("test_match")
    >>> kf.add_detection({"class_id": 12, "class_name": "kill-block"})
    >>> len(kf.get_detections())
    1
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

from score_ai.core.constants import DetectionConstants, FileConstants
from score_ai.core.exceptions import StorageError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class KillfeedDetections:
    """
    Manages killfeed detection data storage and retrieval.
    
    Stores detection data in JSON format organized by match name.
    Supports adding, retrieving, and filtering detections.
    
    Attributes:
        match_name: Name/identifier of the match.
        detections_file: Path to the JSON file storing detections.
        detections: List of detection dictionaries.
    
    Example:
        >>> kf = KillfeedDetections("match_001")
        >>> kf.add_detection({"class_name": "AKM", "confidence": 0.92})
        >>> weapons = kf.get_weapon_detections()
        >>> len(weapons)
        1
    """
    
    def __init__(self, match_name: str) -> None:
        """
        Initialize the killfeed detections manager.
        
        Creates the necessary directory structure and loads any existing
        detections from the file.
        
        Args:
            match_name: Unique identifier for the match.
        
        Raises:
            StorageError: If directory creation fails.
        """
        self.match_name = match_name
        
        # Create game_logs folder structure using constants
        game_logs_folder = FileConstants.GAME_LOGS_DIR
        match_folder = os.path.join(game_logs_folder, match_name)
        
        try:
            os.makedirs(match_folder, exist_ok=True)
        except OSError as e:
            raise StorageError(f"Failed to create match folder: {e}")
        
        self.detections_file = os.path.join(match_folder, FileConstants.KILLFEED_FILE)
        self.detections: list[dict[str, Any]] = []
        self._load_existing_detections()

    def _load_existing_detections(self) -> None:
        """Load existing detections from file if it exists"""
        if os.path.exists(self.detections_file):
            try:
                with open(self.detections_file, "r") as f:
                    self.detections = json.load(f)
            except Exception as e:
                logger.error(f"Error loading detections: {str(e)}")
                self.detections = []

    def add_detection(self, detection_data: Dict[str, Any]) -> None:
        """
        Add a new detection to the list and save to file

        Args:
            detection_data: Detection data including:
                - class_id
                - class_name
                - confidence
                - bbox
                - cropped_image_url
                - timestamp
        """
        # Add timestamp to detection
        detection_data["timestamp"] = datetime.now().isoformat()

        # Add to detections list
        self.detections.append(detection_data)

        # Save to file
        self._save_detections()

    def _save_detections(self) -> None:
        """Save current detections to file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.detections_file), exist_ok=True)

            # Save to file
            with open(self.detections_file, "w") as f:
                json.dump(self.detections, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving detections: {str(e)}")

    def get_detections(self) -> list[dict[str, Any]]:
        """
        Get all detections.
        
        Returns:
            List of all detection dictionaries.
        
        Example:
            >>> kf.get_detections()
            [{'class_name': 'AKM', 'confidence': 0.92, 'timestamp': '...'}]
        """
        return self.detections

    def get_kill_blocks(self) -> list[dict[str, Any]]:
        """
        Get only kill block detections.
        
        Filters detections to return only those with kill-block class.
        
        Returns:
            List of kill block detection dictionaries.
        
        Example:
            >>> blocks = kf.get_kill_blocks()
            >>> all(b['class_name'] == 'kill-block' for b in blocks)
            True
        """
        return [
            d
            for d in self.detections
            if (d.get("class_id") == DetectionConstants.KILL_BLOCK_CLASS_ID or 
                d.get("class_name") == DetectionConstants.KILL_BLOCK_CLASS_NAME)
        ]

    def get_weapon_detections(self) -> list[dict[str, Any]]:
        """
        Get only weapon detections (non-kill-block).
        
        Filters detections to return only weapon detections,
        excluding kill blocks.
        
        Returns:
            List of weapon detection dictionaries.
        
        Example:
            >>> weapons = kf.get_weapon_detections()
            >>> all(w['class_name'] != 'kill-block' for w in weapons)
            True
        """
        return [
            d
            for d in self.detections
            if (d.get("class_id") != DetectionConstants.KILL_BLOCK_CLASS_ID and 
                d.get("class_name") != DetectionConstants.KILL_BLOCK_CLASS_NAME)
        ]

    def clear_detections(self) -> None:
        """
        Clear all detections.
        
        Removes all stored detections and saves the empty state to file.
        
        Example:
            >>> kf.clear_detections()
            >>> len(kf.get_detections())
            0
        """
        self.detections = []
        self._save_detections()
        logger.info(f"Cleared detections for match: {self.match_name}")
    
    def get_detection_count(self) -> int:
        """
        Get the total number of detections.
        
        Returns:
            Number of detections stored.
        
        Example:
            >>> kf.get_detection_count()
            42
        """
        return len(self.detections)

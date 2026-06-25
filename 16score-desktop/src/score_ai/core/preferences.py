"""
User Preferences Manager

Persists user preferences like last selected organization, tournament,
window position, and other settings.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from score_ai.core.logging_config import get_logger

logger = get_logger(__name__)


class UserPreferences:
    """
    Manages user preferences with JSON file persistence.
    
    Usage:
        prefs = UserPreferences()
        prefs.set("last_org_id", "org-123")
        org_id = prefs.get("last_org_id")
    """
    
    _instance: Optional["UserPreferences"] = None
    _prefs_file = "user_preferences.json"
    
    def __new__(cls) -> "UserPreferences":
        """Singleton pattern to ensure single instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._prefs: dict[str, Any] = {}
        self._prefs_path = self._get_prefs_path()
        self._load()
    
    def _get_prefs_path(self) -> Path:
        """Get the path to the preferences file."""
        # Store in user's app data directory
        if os.name == 'nt':  # Windows
            app_data = os.environ.get('APPDATA', os.path.expanduser('~'))
            prefs_dir = Path(app_data) / '16ScoreAI'
        else:  # Linux/Mac
            prefs_dir = Path.home() / '.16score-ai'
        
        prefs_dir.mkdir(parents=True, exist_ok=True)
        return prefs_dir / self._prefs_file
    
    def _load(self) -> None:
        """Load preferences from file."""
        try:
            if self._prefs_path.exists():
                with open(self._prefs_path, 'r', encoding='utf-8') as f:
                    self._prefs = json.load(f)
                logger.debug(f"Loaded preferences from {self._prefs_path}")
        except Exception as e:
            logger.warning(f"Failed to load preferences: {e}")
            self._prefs = {}
    
    def _save(self) -> None:
        """Save preferences to file."""
        try:
            with open(self._prefs_path, 'w', encoding='utf-8') as f:
                json.dump(self._prefs, f, indent=2)
            logger.debug(f"Saved preferences to {self._prefs_path}")
        except Exception as e:
            logger.error(f"Failed to save preferences: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a preference value."""
        return self._prefs.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a preference value and save."""
        self._prefs[key] = value
        self._save()
    
    def delete(self, key: str) -> None:
        """Delete a preference."""
        if key in self._prefs:
            del self._prefs[key]
            self._save()
    
    def clear(self) -> None:
        """Clear all preferences."""
        self._prefs = {}
        self._save()
    
    # Convenience methods for common preferences
    
    @property
    def last_organization_id(self) -> Optional[str]:
        """Get last selected organization ID."""
        return self.get("last_org_id")
    
    @last_organization_id.setter
    def last_organization_id(self, value: str) -> None:
        """Set last selected organization ID."""
        self.set("last_org_id", value)
    
    @property
    def last_organization_name(self) -> Optional[str]:
        """Get last selected organization name."""
        return self.get("last_org_name")
    
    @last_organization_name.setter
    def last_organization_name(self, value: str) -> None:
        """Set last selected organization name."""
        self.set("last_org_name", value)
    
    @property
    def last_tournament_id(self) -> Optional[str]:
        """Get last selected tournament ID."""
        return self.get("last_tournament_id")
    
    @last_tournament_id.setter
    def last_tournament_id(self, value: str) -> None:
        """Set last selected tournament ID."""
        self.set("last_tournament_id", value)
    
    @property
    def last_tournament_name(self) -> Optional[str]:
        """Get last selected tournament name."""
        return self.get("last_tournament_name")
    
    @last_tournament_name.setter
    def last_tournament_name(self, value: str) -> None:
        """Set last selected tournament name."""
        self.set("last_tournament_name", value)
    
    def get_window_geometry(self) -> Optional[dict]:
        """Get saved window geometry."""
        return self.get("window_geometry")
    
    def set_window_geometry(self, x: int, y: int, width: int, height: int, maximized: bool = False) -> None:
        """Save window geometry."""
        self.set("window_geometry", {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "maximized": maximized
        })
    
    def get_recent_tournaments(self) -> list[dict]:
        """Get list of recently accessed tournaments."""
        return self.get("recent_tournaments", [])
    
    def add_recent_tournament(self, tournament_id: str, tournament_name: str) -> None:
        """Add a tournament to recent list (max 5)."""
        recent = self.get_recent_tournaments()
        
        # Remove if already exists
        recent = [t for t in recent if t.get("id") != tournament_id]
        
        # Add to front
        recent.insert(0, {"id": tournament_id, "name": tournament_name})
        
        # Keep only last 5
        recent = recent[:5]
        
        self.set("recent_tournaments", recent)


# Global instance
preferences = UserPreferences()



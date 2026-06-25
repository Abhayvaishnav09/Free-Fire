"""
16Score Brand Colors - Official color palette for the application.
"""
from __future__ import annotations


class BrandColors:
    """Official 16Score brand color palette"""
    
    # Primary Branding Colors
    PRIMARY = "#3E25F6"           # Deep Purple/Blue
    PRIMARY_DARK = "#3B24F3"      # Darker Purple/Blue
    PRIMARY_LIGHT = "#6D5DFF"     # Light Purple
    PRIMARY_ALT = "#4A6EFF"       # Blue Accent
    
    # Background Colors
    BACKGROUND = "#0F1113"        # Dark Black
    BACKGROUND_DARK = "#0A0C10"   # Darker Black
    BACKGROUND_ALT = "#111216"    # Alternate Dark
    
    # Card & Surface Colors
    CARD = "#181A20"              # Dark Gray
    CARD_SECONDARY = "#23243A"    # Purple-Gray
    SURFACE = "#1A1B1F"           # Dark Surface
    SURFACE_ALT = "#1E1E2D"       # Alt Surface
    GLASS = "#424242"             # Gray Glass
    
    # Text Colors (as RGBA strings for Qt)
    TEXT_PRIMARY = "rgba(255, 255, 255, 1.0)"      # White 100%
    TEXT_SECONDARY = "rgba(255, 255, 255, 0.7)"   # White 70%
    TEXT_TERTIARY = "rgba(255, 255, 255, 0.54)"   # White 54%
    TEXT_MUTED = "rgba(255, 255, 255, 0.38)"      # White 38%
    TEXT_DISABLED = "rgba(255, 255, 255, 0.30)"   # White 30%
    
    # Accent Colors
    ACCENT_GOLD = "#FFDA0A"       # Gold
    ACCENT_SILVER = "#B3B3B3"     # Silver
    ACCENT_BRONZE = "#A23E00"     # Bronze
    ACCENT_ORANGE = "#FFB800"     # Orange
    ACCENT_GREEN = "#10B981"      # Green
    ACCENT_BLUE = "#3B82F6"       # Blue
    
    # Status Colors
    SUCCESS = "#10B981"           # Green (Live)
    ERROR = "#E53935"             # Red
    WARNING = "#FFB800"           # Orange (Completed)
    INFO = "#3B82F6"              # Blue (Upcoming)
    
    # Border & Divider Colors
    BORDER = "#2B2D30"
    BORDER_ALT = "#2C2C3A"
    DIVIDER = "#2B2D30"
    
    # Special Colors
    COLOR_16 = "#6B8EFF"          # Special brand highlight for value = 16
    
    # Fallback Colors (for charts/graphs)
    CHART_AMBER = "#FFC107"
    CHART_BROWN = "#795548"
    CHART_BLUE_GREY = "#607D8B"
    CHART_GREY = "#9E9E9E"
    CHART_LIGHT_GREEN = "#00E676"
    CHART_DEEP_ORANGE = "#FF5722"
    CHART_DEEP_PURPLE = "#673AB7"
    CHART_TEAL = "#009688"
    
    # Map-Specific Gradients (Gaming Maps)
    MAP_ERANGEL = ("#4A7C59", "#6BA37A")      # Forest - Green
    MAP_MIRAMAR = ("#E07A3F", "#E8A57F")      # Desert - Orange
    MAP_SANHOK = ("#1B5E3A", "#4A8F6A")       # Jungle - Dark green
    MAP_VIKENDI = ("#3D5A80", "#6B8FB0")      # Snow - Blue
    MAP_LIVIK = ("#8B4FA8", "#B57FC8")        # Sunset - Purple
    MAP_NUSA = ("#00A8C5", "#4DD0E1")         # Ocean - Cyan
    MAP_KARAKIN = ("#8B6F47", "#B59F77")      # Desert Rock - Brown
    MAP_DESTON = ("#2C5364", "#5A7B8C")       # Urban - Dark blue
    
    @classmethod
    def get_gradient(cls, color1: str, color2: str, direction: str = "horizontal") -> str:
        """Generate a Qt gradient string"""
        if direction == "horizontal":
            return f"qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {color1}, stop:1 {color2})"
        elif direction == "vertical":
            return f"qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {color1}, stop:1 {color2})"
        elif direction == "diagonal":
            return f"qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {color1}, stop:1 {color2})"
        return f"qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {color1}, stop:1 {color2})"
    
    @classmethod
    def primary_gradient(cls, direction: str = "horizontal") -> str:
        """Get the primary brand gradient"""
        return cls.get_gradient(cls.PRIMARY, cls.PRIMARY_LIGHT, direction)
    
    @classmethod
    def background_gradient(cls, direction: str = "horizontal") -> str:
        """Get the background gradient"""
        return cls.get_gradient(cls.BACKGROUND_DARK, cls.BACKGROUND, direction)


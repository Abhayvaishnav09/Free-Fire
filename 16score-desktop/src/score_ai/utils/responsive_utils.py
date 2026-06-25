"""
Responsive Design Utilities for PyQt5 Applications

This module provides utilities for creating responsive PyQt5 applications that
automatically adapt to different screen resolutions and DPI settings.

Usage:
    from score_ai.utils.responsive_utils import ResponsiveUtils
    
    utils = ResponsiveUtils()
    scaled_size = utils.scale_size(100)  # Scale 100px for current screen
    font = utils.create_responsive_font(14)  # Create scaled font

Example:
    >>> utils = ResponsiveUtils()
    >>> utils.get_screen_width()
    1920
    >>> utils.scale_size(100)
    100
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PyQt5.QtCore import QRect
from PyQt5.QtGui import QFont, QFontMetrics
from PyQt5.QtWidgets import QApplication, QSizePolicy, QWidget

from score_ai.core.constants import CameraConstants, UIConstants

if TYPE_CHECKING:
    from typing import Optional

logger = logging.getLogger(__name__)


class ResponsiveUtils:
    """Utility class for responsive design calculations"""
    
    _instance = None
    _screen_width = None
    _screen_height = None
    _dpi_scale = None
    _base_font_size = 12
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize screen dimensions and DPI scaling from actual screen"""
        try:
            # Get actual screen dimensions from QApplication
            app = QApplication.instance()
            if app is None:
                # Create temporary app to get screen info
                app = QApplication([])
                created_app = True
            else:
                created_app = False
            
            screen = app.primaryScreen()
            geometry = screen.geometry()
            self._screen_width = geometry.width()
            self._screen_height = geometry.height()
            
            # Calculate DPI scale factor
            logical_dpi = screen.logicalDotsPerInch()
            self._dpi_scale = logical_dpi / 96.0  # 96 DPI is standard
            
            # Clean up temporary app if we created one
            if created_app:
                app.quit()
                
        except Exception as e:
            # Fallback to reasonable defaults if screen detection fails
            logger.warning(f"Screen detection failed, using defaults: {e}")
            self._screen_width = 1920
            self._screen_height = 1080
            self._dpi_scale = 1.0
        
        logger.debug(f"Screen: {self._screen_width}x{self._screen_height}, DPI Scale: {self._dpi_scale:.2f}")
    
    @classmethod
    def get_screen_width(cls):
        """Get screen width"""
        instance = cls()
        return instance._screen_width
    
    @classmethod
    def get_screen_height(cls):
        """Get screen height"""
        instance = cls()
        return instance._screen_height
    
    @classmethod
    def get_dpi_scale(cls):
        """Get DPI scale factor"""
        instance = cls()
        return instance._dpi_scale
    
    @classmethod
    def scale_size(cls, size):
        """Scale a size value based on DPI and screen resolution"""
        instance = cls()
        
        # Base scaling on DPI
        scaled = int(size * instance._dpi_scale)
        
        # Additional scaling based on screen width
        if instance._screen_width >= 3840:  # 4K
            scaled = int(scaled * 1.2)
        elif instance._screen_width >= 2560:  # 1440p
            scaled = int(scaled * 1.1)
        elif instance._screen_width <= 1366:  # Lower resolution
            scaled = int(scaled * 0.9)
        
        return max(1, scaled)  # Ensure minimum size of 1
    
    @classmethod
    def scale_font_size(cls, base_size):
        """Scale font size based on screen resolution and DPI"""
        instance = cls()
        
        # Scale based on DPI first
        scaled = base_size * instance._dpi_scale
        
        # Additional scaling based on screen width
        if instance._screen_width >= 3840:  # 4K
            scaled *= 1.3
        elif instance._screen_width >= 2560:  # 1440p
            scaled *= 1.15
        elif instance._screen_width <= 1366:  # Lower resolution
            scaled *= 0.85
        
        return max(8, int(scaled))  # Minimum font size of 8
    
    @classmethod
    def get_responsive_width(cls, percentage):
        """Get width as percentage of screen width"""
        instance = cls()
        return int(instance._screen_width * percentage / 100)
    
    @classmethod
    def get_responsive_height(cls, percentage):
        """Get height as percentage of screen height"""
        instance = cls()
        return int(instance._screen_height * percentage / 100)
    
    @classmethod
    def get_min_width(cls, base_min):
        """Get minimum width scaled for current screen"""
        return cls.scale_size(base_min)
    
    @classmethod
    def get_max_width(cls, base_max):
        """Get maximum width scaled for current screen"""
        return cls.scale_size(base_max)
    
    @classmethod
    def create_responsive_font(cls, base_size, weight=QFont.Normal):
        """Create a responsive font that scales with screen size"""
        font_size = cls.scale_font_size(base_size)
        font = QFont("Inter", font_size, weight)
        return font
    
    @classmethod
    def set_responsive_size_policy(cls, widget, horizontal=QSizePolicy.Expanding, vertical=QSizePolicy.Preferred):
        """Set responsive size policy for widgets"""
        widget.setSizePolicy(horizontal, vertical)
    
    @classmethod
    def calculate_spacing(cls, base_spacing):
        """Calculate responsive spacing"""
        return cls.scale_size(base_spacing)
    
    @classmethod
    def calculate_margins(cls, base_margin):
        """Calculate responsive margins"""
        return cls.scale_size(base_margin)
    
    @classmethod
    def get_responsive_button_height(cls):
        """Get responsive button height"""
        return cls.scale_size(48)  # Base button height of 48
    
    @classmethod
    def get_responsive_input_height(cls):
        """Get responsive input field height"""
        return cls.scale_size(44)  # Base input height of 44
    
    @classmethod
    def get_adaptive_card_width(cls):
        """Get adaptive card width based on screen size"""
        screen_width = cls.get_screen_width()
        
        if screen_width >= 3840:  # 4K
            return cls.get_responsive_width(40)  # 40% of screen
        elif screen_width >= 2560:  # 1440p
            return cls.get_responsive_width(50)  # 50% of screen
        elif screen_width >= 1920:  # 1080p
            return cls.get_responsive_width(60)  # 60% of screen
        else:  # Lower resolutions
            return cls.get_responsive_width(80)  # 80% of screen


def apply_responsive_style(widget, base_styles):
    """Apply responsive styles to a widget with DPI scaling"""
    utils = ResponsiveUtils()
    
    # Scale any pixel values in the stylesheet
    responsive_styles = base_styles
    
    # Common pixel values to scale
    pixel_properties = [
        'font-size', 'padding', 'margin', 'border-radius', 
        'border-width', 'min-width', 'max-width', 'min-height', 'max-height'
    ]
    
    for prop in pixel_properties:
        # This is a simple implementation - you might want to use regex for more complex cases
        if f'{prop}:' in responsive_styles:
            # Extract and scale pixel values (simplified approach)
            import re
            pattern = f'{prop}:\\s*(\\d+)px'
            matches = re.findall(pattern, responsive_styles)
            for match in matches:
                original_value = int(match)
                scaled_value = utils.scale_size(original_value)
                responsive_styles = responsive_styles.replace(
                    f'{prop}: {original_value}px', 
                    f'{prop}: {scaled_value}px'
                )
    
    widget.setStyleSheet(responsive_styles)


class ResponsiveWidget(QWidget):
    """Base responsive widget class"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.utils = ResponsiveUtils()
        self.init_responsive_properties()
    
    def init_responsive_properties(self):
        """Initialize responsive properties"""
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    
    def get_responsive_size(self, base_width, base_height):
        """Get responsive size for this widget"""
        return (
            self.utils.scale_size(base_width),
            self.utils.scale_size(base_height)
        )
    
    def set_responsive_minimum_size(self, base_width, base_height):
        """Set responsive minimum size"""
        width, height = self.get_responsive_size(base_width, base_height)
        self.setMinimumSize(width, height)
    
    def set_responsive_maximum_size(self, base_width, base_height):
        """Set responsive maximum size"""
        width, height = self.get_responsive_size(base_width, base_height)
        self.setMaximumSize(width, height) 
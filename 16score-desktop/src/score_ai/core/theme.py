"""
16Score Application Theme - Centralized styling and theming.

This module provides DRY-compliant theming for the entire application.
All colors, fonts, and common styles are defined here.
"""
from __future__ import annotations

from PyQt5.QtGui import QFont, QColor
from PyQt5.QtWidgets import QGraphicsDropShadowEffect


class Colors:
    """Official 16Score brand color palette"""
    
    # Primary Branding
    PRIMARY = "#3E25F6"
    PRIMARY_DARK = "#3B24F3"
    PRIMARY_LIGHT = "#6D5DFF"
    PRIMARY_ALT = "#4A6EFF"
    
    # Backgrounds
    BG = "#0F1113"
    BG_DARK = "#0A0C10"
    BG_ALT = "#111216"
    
    # Cards & Surfaces
    CARD = "#181A20"
    CARD_SECONDARY = "#23243A"
    SURFACE = "#1A1B1F"
    SURFACE_ALT = "#1E1E2D"
    
    # Borders
    BORDER = "#2B2D30"
    BORDER_ALT = "#2C2C3A"
    
    # Text (with opacity)
    TEXT_PRIMARY = "rgba(255, 255, 255, 1.0)"
    TEXT_SECONDARY = "rgba(255, 255, 255, 0.7)"
    TEXT_TERTIARY = "rgba(255, 255, 255, 0.54)"
    TEXT_MUTED = "rgba(255, 255, 255, 0.38)"
    TEXT_DISABLED = "rgba(255, 255, 255, 0.30)"
    
    # For QColor usage
    TEXT_PRIMARY_RGB = (255, 255, 255, 255)
    TEXT_SECONDARY_RGB = (255, 255, 255, 178)
    TEXT_MUTED_RGB = (255, 255, 255, 97)
    
    # Status
    SUCCESS = "#10B981"
    ERROR = "#E53935"
    ERROR_LIGHT = "#ff6b6b"
    WARNING = "#FFB800"
    INFO = "#3B82F6"
    
    # Accents
    GOLD = "#FFDA0A"
    
    @classmethod
    def with_alpha(cls, hex_color: str, alpha: float) -> str:
        """Convert hex color to rgba with alpha (0.0 - 1.0)"""
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        # Qt stylesheets work better with integer alpha (0-255)
        alpha_int = int(alpha * 255)
        return f"rgba({r}, {g}, {b}, {alpha_int})"
    
    @classmethod
    def to_qcolor(cls, hex_or_rgba: str, alpha: int = 255) -> QColor:
        """Convert hex or rgba color to QColor with optional alpha (0-255)"""
        # Handle rgba() format
        if hex_or_rgba.startswith('rgba'):
            import re
            match = re.match(r'rgba\((\d+),\s*(\d+),\s*(\d+),\s*([\d.]+)\)', hex_or_rgba)
            if match:
                r, g, b = int(match.group(1)), int(match.group(2)), int(match.group(3))
                a = float(match.group(4))
                # If alpha is 0-1 range, convert to 0-255
                if a <= 1.0:
                    a = int(a * 255)
                return QColor(r, g, b, int(a))
        
        # Handle hex format
        hex_color = hex_or_rgba.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return QColor(r, g, b, alpha)


class Fonts:
    """Centralized font configuration"""
    
    # Font family - used throughout the app
    FAMILY = "Segoe UI"
    FAMILY_EMOJI = "Segoe UI Emoji"
    
    # Font sizes
    SIZE_XXL = 24
    SIZE_XL = 18
    SIZE_LG = 14
    SIZE_MD = 13
    SIZE_SM = 11
    SIZE_XS = 10
    SIZE_XXS = 9
    
    # Font weights
    WEIGHT_BOLD = QFont.Bold
    WEIGHT_DEMIBOLD = QFont.DemiBold
    WEIGHT_NORMAL = QFont.Normal
    
    @classmethod
    def create(cls, size: int = SIZE_MD, weight: int = WEIGHT_NORMAL) -> QFont:
        """Create a QFont with the app's font family"""
        return QFont(cls.FAMILY, size, weight)
    
    @classmethod
    def title(cls) -> QFont:
        """Large title font"""
        return cls.create(cls.SIZE_XXL, cls.WEIGHT_BOLD)
    
    @classmethod
    def heading(cls) -> QFont:
        """Heading font"""
        return cls.create(cls.SIZE_XL, cls.WEIGHT_DEMIBOLD)
    
    @classmethod
    def body(cls) -> QFont:
        """Body text font"""
        return cls.create(cls.SIZE_MD, cls.WEIGHT_NORMAL)
    
    @classmethod
    def label(cls) -> QFont:
        """Form label font"""
        return cls.create(cls.SIZE_XS, cls.WEIGHT_NORMAL)
    
    @classmethod
    def button(cls, size: int = None) -> QFont:
        """Button text font"""
        return cls.create(size or cls.SIZE_LG, cls.WEIGHT_DEMIBOLD)
    
    @classmethod
    def small(cls) -> QFont:
        """Small text font"""
        return cls.create(cls.SIZE_XXS, cls.WEIGHT_NORMAL)


class Spacing:
    """Consistent spacing values"""
    
    XXS = 4
    XS = 6
    SM = 8
    MD = 12
    LG = 16
    XL = 20
    XXL = 24
    XXXL = 32
    
    # Component specific
    CARD_PADDING = 40
    INPUT_HEIGHT = 52
    BUTTON_HEIGHT = 50
    HEADER_HEIGHT = 40
    
    # Border radius
    RADIUS_SM = 4
    RADIUS_MD = 8
    RADIUS_LG = 10
    RADIUS_XL = 12
    RADIUS_XXL = 16


class Gradients:
    """Common gradient definitions"""
    
    @staticmethod
    def primary_horizontal() -> str:
        return f"qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {Colors.PRIMARY}, stop:1 {Colors.PRIMARY_LIGHT})"
    
    @staticmethod
    def primary_diagonal() -> str:
        return f"qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {Colors.PRIMARY}, stop:1 {Colors.PRIMARY_LIGHT})"
    
    @staticmethod
    def primary_hover() -> str:
        return f"qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {Colors.PRIMARY_ALT}, stop:1 {Colors.PRIMARY_LIGHT})"
    
    @staticmethod
    def background() -> str:
        return f"qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {Colors.BG_DARK}, stop:0.5 {Colors.BG}, stop:1 {Colors.BG_DARK})"
    
    @staticmethod
    def card() -> str:
        return f"qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {Colors.with_alpha(Colors.CARD, 0.95)}, stop:1 {Colors.with_alpha(Colors.BG, 0.98)})"
    
    @staticmethod
    def title_bar() -> str:
        return f"qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {Colors.BG_DARK}, stop:0.5 {Colors.BG}, stop:1 {Colors.BG_DARK})"


class Shadows:
    """Common shadow effects"""
    
    @staticmethod
    def card() -> QGraphicsDropShadowEffect:
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(50)
        shadow.setXOffset(0)
        shadow.setYOffset(8)
        shadow.setColor(QColor(0, 0, 0, 80))
        return shadow
    
    @staticmethod
    def button() -> QGraphicsDropShadowEffect:
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(Colors.to_qcolor(Colors.PRIMARY, 100))
        return shadow
    
    @staticmethod
    def button_hover() -> QGraphicsDropShadowEffect:
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(25)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(Colors.to_qcolor(Colors.PRIMARY, 150))
        return shadow
    
    @staticmethod
    def input_focus() -> QGraphicsDropShadowEffect:
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(0)
        shadow.setColor(Colors.to_qcolor(Colors.PRIMARY_LIGHT, 60))
        return shadow
    
    @staticmethod
    def glow(color: str = Colors.PRIMARY, blur: int = 40, alpha: int = 60) -> QGraphicsDropShadowEffect:
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(blur)
        shadow.setXOffset(0)
        shadow.setYOffset(0)
        shadow.setColor(Colors.to_qcolor(color, alpha))
        return shadow


class Styles:
    """Pre-built stylesheet strings for common components"""
    
    # Input field styles
    @staticmethod
    def input(padding_left: int = 16) -> str:
        return f"""
            QLineEdit {{
                background-color: {Colors.CARD};
                border: 2px solid {Colors.BORDER_ALT};
                border-radius: {Spacing.RADIUS_LG}px;
                padding: 0 {Spacing.LG}px 0 {padding_left}px;
                font-family: '{Fonts.FAMILY}';
                font-size: {Fonts.SIZE_MD}px;
                color: white;
                selection-background-color: {Colors.PRIMARY};
                min-height: {Spacing.INPUT_HEIGHT}px;
            }}
            QLineEdit:hover {{
                border: 2px solid {Colors.PRIMARY};
                background-color: {Colors.SURFACE};
            }}
            QLineEdit:focus {{
                border: 2px solid {Colors.PRIMARY_LIGHT};
                background-color: {Colors.SURFACE};
            }}
            QLineEdit::placeholder {{
                color: rgba(255, 255, 255, 100);
            }}
        """
    
    @staticmethod
    def button() -> str:
        return f"""
            QPushButton {{
                background: {Gradients.primary_horizontal()};
                color: white;
                border-radius: {Spacing.RADIUS_LG}px;
                font-family: '{Fonts.FAMILY}';
                font-size: {Fonts.SIZE_LG}px;
                font-weight: 700;
                border: none;
                padding: 0 {Spacing.XXXL}px;
                letter-spacing: 0.5px;
                min-height: {Spacing.BUTTON_HEIGHT}px;
            }}
            QPushButton:hover {{
                background: {Gradients.primary_hover()};
            }}
            QPushButton:pressed {{
                background: {Colors.PRIMARY_DARK};
            }}
            QPushButton:disabled {{
                background: {Colors.CARD};
                color: {Colors.TEXT_MUTED};
            }}
        """
    
    @staticmethod
    def button_hover() -> str:
        return f"""
            QPushButton {{
                background: {Gradients.primary_hover()};
                color: white;
                border-radius: {Spacing.RADIUS_LG}px;
                font-family: '{Fonts.FAMILY}';
                font-size: {Fonts.SIZE_LG}px;
                font-weight: 700;
                border: none;
                padding: 0 {Spacing.XXXL}px;
                letter-spacing: 0.5px;
                min-height: {Spacing.BUTTON_HEIGHT}px;
            }}
            QPushButton:pressed {{
                background: {Colors.PRIMARY_DARK};
            }}
        """
    
    @staticmethod
    def card() -> str:
        return f"""
            background: {Colors.CARD};
            border-radius: {Spacing.RADIUS_XXL}px;
            border: 1px solid {Colors.BORDER_ALT};
        """
    
    @staticmethod
    def checkbox() -> str:
        return f"""
            QCheckBox {{
                color: rgba(255, 255, 255, 128);
                background: transparent;
                font-family: '{Fonts.FAMILY}';
                font-size: {Fonts.SIZE_XS}px;
                spacing: {Spacing.XS}px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border-radius: {Spacing.RADIUS_SM}px;
                border: 1.5px solid {Colors.BORDER};
                background: {Colors.CARD};
            }}
            QCheckBox::indicator:checked {{
                background: {Colors.PRIMARY};
                border-color: {Colors.PRIMARY_LIGHT};
            }}
            QCheckBox::indicator:hover {{
                border-color: {Colors.PRIMARY};
            }}
        """
    
    @staticmethod
    def label(color: str = "TEXT_SECONDARY") -> str:
        color_val = getattr(Colors, color, Colors.TEXT_SECONDARY)
        return f"""
            color: {color_val};
            background: transparent;
            font-family: '{Fonts.FAMILY}';
        """
    
    @staticmethod
    def error_label() -> str:
        return f"""
            color: {Colors.ERROR_LIGHT};
            background: rgba(229, 57, 53, 25);
            border: 1px solid rgba(229, 57, 53, 50);
            border-radius: {Spacing.RADIUS_MD}px;
            padding: {Spacing.SM}px {Spacing.MD}px;
            font-family: '{Fonts.FAMILY}';
            font-size: {Fonts.SIZE_XS}px;
        """
    
    @staticmethod
    def header() -> str:
        return f"""
            background: {Colors.BG_DARK};
            border-bottom: 1px solid {Colors.BORDER};
        """
    
    @staticmethod
    def nav_active() -> str:
        return f"""
            color: white;
            background: {Gradients.primary_horizontal()};
            font-family: '{Fonts.FAMILY}';
            font-weight: 900;
        """
    
    @staticmethod
    def nav_inactive() -> str:
        return f"""
            color: {Colors.TEXT_SECONDARY};
            background: transparent;
            font-family: '{Fonts.FAMILY}';
            font-weight: 900;
        """
    
    @staticmethod
    def scrollbar() -> str:
        return f"""
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: 8px;
                margin: 4px 2px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {Colors.with_alpha('#FFFFFF', 0.2)};
                border-radius: 4px;
                min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {Colors.with_alpha('#FFFFFF', 0.3)};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: transparent;
            }}
        """
    
    @staticmethod
    def menu() -> str:
        return f"""
            QMenu {{
                background-color: {Colors.CARD};
                border: 1px solid {Colors.BORDER};
                border-radius: {Spacing.RADIUS_MD}px;
                padding: {Spacing.SM}px 0;
                font-family: '{Fonts.FAMILY}';
            }}
            QMenu::item {{
                padding: {Spacing.SM}px {Spacing.LG}px;
                font-size: {Fonts.SIZE_LG}px;
                color: {Colors.TEXT_SECONDARY};
            }}
            QMenu::item:selected {{
                background-color: {Colors.CARD_SECONDARY};
                color: {Colors.PRIMARY_LIGHT};
            }}
            QMenu::item:disabled {{
                color: {Colors.TEXT_MUTED};
            }}
            QMenu::separator {{
                height: 1px;
                background: {Colors.BORDER};
                margin: {Spacing.XS}px {Spacing.SM}px;
            }}
        """
    
    @staticmethod
    def secondary_button(radius: int = None) -> str:
        """Outline/secondary button style"""
        r = radius if radius else Spacing.RADIUS_XL
        return f"""
            QPushButton {{
                background: {Colors.CARD};
                border: 1px solid {Colors.BORDER};
                border-radius: {r}px;
                padding: 12px 36px;
                color: {Colors.PRIMARY_LIGHT};
                font-size: {Fonts.SIZE_MD}px;
                font-weight: 700;
                font-family: '{Fonts.FAMILY}';
            }}
            QPushButton:hover {{
                background: {Colors.SURFACE};
                border-color: {Colors.PRIMARY};
            }}
            QPushButton:pressed {{
                background: {Colors.SURFACE_ALT};
            }}
        """
    
    @staticmethod
    def pagination_button(size: int = 36) -> str:
        """Pagination navigation button style"""
        radius = size // 4
        return f"""
            QPushButton {{
                background-color: {Colors.CARD};
                border: 1px solid {Colors.BORDER};
                border-radius: {radius}px;
                color: {Colors.TEXT_SECONDARY};
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {Colors.SURFACE};
                border-color: {Colors.PRIMARY};
            }}
            QPushButton:pressed {{
                background-color: {Colors.SURFACE_ALT};
            }}
            QPushButton:disabled {{
                background-color: {Colors.CARD};
                color: {Colors.TEXT_MUTED};
                border-color: {Colors.BORDER};
            }}
        """
    
    @staticmethod
    def pagination_active(size: int = 36) -> str:
        """Active pagination page indicator style"""
        radius = size // 4
        return f"""
            background-color: {Colors.PRIMARY}; 
            color: white; 
            font-weight: 600; 
            font-family: '{Fonts.FAMILY}';
            border-radius: {radius}px;
        """
    
    @staticmethod
    def page_background() -> str:
        """Standard page background style"""
        return f"""
            QWidget {{
                background: {Colors.BG};
                font-family: '{Fonts.FAMILY}';
            }}
            QWidget:focus, QPushButton:focus, QLabel:focus, QFrame:focus {{
                outline: none;
                border: none;
            }}
        """
    
    @staticmethod
    def hoverable_card() -> str:
        """Card with hover effect"""
        return f"""
            QFrame {{
                background: {Colors.CARD};
                border-radius: {Spacing.RADIUS_XXL}px;
                border: 1px solid {Colors.BORDER};
            }}
            QFrame:hover {{
                border: 1px solid {Colors.PRIMARY};
                background: {Colors.SURFACE};
            }}
        """
    
    @staticmethod
    def status_badge(status: str = "default") -> str:
        """Status badge with color based on status type"""
        status_colors = {
            "upcoming": Colors.WARNING,
            "ongoing": Colors.SUCCESS,
            "live": Colors.SUCCESS,
            "finished": Colors.TEXT_MUTED,
            "default": Colors.INFO
        }
        color = status_colors.get(status.lower(), Colors.INFO)
        return f"""
            QLabel {{
                background-color: {color};
                color: white;
                font-weight: 700;
                border-radius: {Spacing.RADIUS_LG}px;
            }}
        """


# Convenience exports
theme = {
    'colors': Colors,
    'fonts': Fonts,
    'spacing': Spacing,
    'gradients': Gradients,
    'shadows': Shadows,
    'styles': Styles,
}


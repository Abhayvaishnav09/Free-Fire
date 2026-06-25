"""
Modern Login Page for 16Score AI Application

Features:
- Gradient background with animated mesh
- Glassmorphism card design
- Password visibility toggle
- Auto-focus and keyboard navigation
- Loading spinner animation
- Shake animation on error
"""
from __future__ import annotations

import sys
import os

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QSizePolicy,
    QSpacerItem,
    QCheckBox,
)
from PyQt5.QtGui import (
    QPixmap, 
    QFont, 
    QColor, 
    QPainter, 
    QLinearGradient,
    QRadialGradient,
    QBrush,
    QPen,
    QPolygon,
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QPropertyAnimation, QPoint, QEasingCurve

from score_ai.core.api_service import user_api
from score_ai.core.security import InputValidator, SecureStorage
from score_ai.core.theme import Colors, Fonts, Spacing, Gradients, Shadows, Styles
from score_ai.ui.components.keyboard_shortcuts import KeyboardShortcuts
from score_ai.utils.helpers import save_login_history
from score_ai.utils.responsive_utils import ResponsiveUtils, ResponsiveWidget


class HeroPanel(QWidget):
    """Left hero panel with headline and feature carousel"""
    
    FEATURES = [
        {
            "title": "Real-Time Kill Detection",
            "description": "AI-powered system detects kills with 99%+ accuracy"
        },
        {
            "title": "Instant Point Calculation",
            "description": "Automatic scoring based on kills and placements"
        },
        {
            "title": "Tournament Management",
            "description": "Manage brackets, teams and live leaderboards"
        },
        {
            "title": "60 FPS Processing",
            "description": "GPU-accelerated with under 50ms latency"
        },
        {
            "title": "Cloud Sync",
            "description": "Results sync instantly to your portal"
        },
    ]
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_index = 0
        self.setStyleSheet("background: transparent;")
        self.init_ui()
        
        # Auto-rotate timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_feature)
        self.timer.start(4000)
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(60, 80, 60, 60)
        layout.setSpacing(0)
        
        # Top tagline
        tagline = QLabel("AI-Powered Tournament Scoring Platform")
        tagline.setFont(Fonts.create(13))
        tagline.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
        layout.addWidget(tagline)
        
        layout.addSpacing(24)
        
        # Main headline
        headline = QLabel("Score\nEvery Kill")
        headline.setFont(Fonts.create(52, Fonts.WEIGHT_BOLD))
        headline.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent;")
        layout.addWidget(headline)
        
        layout.addStretch()
        
        # Feature carousel section
        carousel_container = QWidget()
        carousel_container.setStyleSheet("background: transparent;")
        carousel_layout = QVBoxLayout(carousel_container)
        carousel_layout.setContentsMargins(0, 0, 0, 0)
        carousel_layout.setSpacing(12)
        
        # Feature title (rotating)
        self.feature_title = QLabel()
        self.feature_title.setFont(Fonts.create(18, Fonts.WEIGHT_BOLD))
        self.feature_title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent;")
        carousel_layout.addWidget(self.feature_title)
        
        # Feature description (rotating)
        self.feature_desc = QLabel()
        self.feature_desc.setFont(Fonts.create(14))
        self.feature_desc.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
        self.feature_desc.setWordWrap(True)
        carousel_layout.addWidget(self.feature_desc)
        
        carousel_layout.addSpacing(16)
        
        # Progress dots
        dots_container = QWidget()
        dots_container.setStyleSheet("background: transparent;")
        dots_layout = QHBoxLayout(dots_container)
        dots_layout.setContentsMargins(0, 0, 0, 0)
        dots_layout.setSpacing(8)
        dots_layout.setAlignment(Qt.AlignLeft)
        
        self.dots = []
        for i in range(len(self.FEATURES)):
            dot = QFrame()
            dot.setFixedSize(8, 8)
            dot.setCursor(Qt.PointingHandCursor)
            dot.mousePressEvent = lambda e, idx=i: self.go_to_feature(idx)
            dots_layout.addWidget(dot)
            self.dots.append(dot)
        
        carousel_layout.addWidget(dots_container)
        layout.addWidget(carousel_container)
        
        layout.addSpacing(40)
        
        # Stats row at bottom
        stats_container = QWidget()
        stats_container.setStyleSheet("background: transparent;")
        stats_layout = QHBoxLayout(stats_container)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        stats_layout.setSpacing(48)
        stats_layout.setAlignment(Qt.AlignLeft)
        
        stats = [
            ("500+", "Tournaments"),
            ("10K+", "Matches"),
            ("99%", "Accuracy"),
        ]
        
        for value, label in stats:
            stat_widget = QWidget()
            stat_widget.setStyleSheet("background: transparent;")
            stat_layout = QVBoxLayout(stat_widget)
            stat_layout.setContentsMargins(0, 0, 0, 0)
            stat_layout.setSpacing(4)
            
            value_label = QLabel(value)
            value_label.setFont(Fonts.create(28, Fonts.WEIGHT_BOLD))
            value_label.setStyleSheet(f"color: {Colors.PRIMARY_LIGHT}; background: transparent;")
            stat_layout.addWidget(value_label)
            
            desc_label = QLabel(label)
            desc_label.setFont(Fonts.create(12))
            desc_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; background: transparent;")
            stat_layout.addWidget(desc_label)
            
            stats_layout.addWidget(stat_widget)
        
        layout.addWidget(stats_container)
        
        # Initial update
        self.update_feature()
    
    def update_feature(self):
        feature = self.FEATURES[self.current_index]
        self.feature_title.setText(feature["title"])
        self.feature_desc.setText(feature["description"])
        
        # Update dots
        for i, dot in enumerate(self.dots):
            if i == self.current_index:
                dot.setStyleSheet(f"background: {Colors.PRIMARY_LIGHT}; border-radius: 4px;")
            else:
                dot.setStyleSheet(f"background: {Colors.BORDER}; border-radius: 4px;")
    
    def next_feature(self):
        self.current_index = (self.current_index + 1) % len(self.FEATURES)
        self.update_feature()
    
    def go_to_feature(self, index):
        self.current_index = index
        self.update_feature()
        self.timer.stop()
        self.timer.start(4000)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect()
        
        # Dark background
        painter.fillRect(rect, Colors.to_qcolor(Colors.BG_DARK))
        
        # Subtle decorative circles
        painter.setCompositionMode(QPainter.CompositionMode_Plus)
        
        # Large circle outline
        painter.setPen(QPen(Colors.to_qcolor(Colors.BORDER, 60), 1))
        painter.setBrush(Qt.NoBrush)
        center_x = rect.width() * 0.6
        center_y = rect.height() * 0.45
        painter.drawEllipse(int(center_x - 180), int(center_y - 180), 360, 360)
        painter.drawEllipse(int(center_x - 120), int(center_y - 120), 240, 240)


class AIBackground(QWidget):
    """AI-themed background with neural network visualization"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.offset = 0
        self.pulse = 0
        
        # Generate node positions
        import random
        random.seed(42)  # Consistent pattern
        self.nodes = []
        for _ in range(15):
            self.nodes.append({
                'x': random.uniform(0.1, 0.9),
                'y': random.uniform(0.1, 0.9),
                'size': random.uniform(3, 8),
                'pulse_offset': random.uniform(0, 6.28)
            })
        
        # Generate connections between nearby nodes
        self.connections = []
        for i, n1 in enumerate(self.nodes):
            for j, n2 in enumerate(self.nodes):
                if i < j:
                    dist = ((n1['x'] - n2['x'])**2 + (n1['y'] - n2['y'])**2)**0.5
                    if dist < 0.35:
                        self.connections.append((i, j, dist))
        
        # Animation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(50)
    
    def animate(self):
        self.offset = (self.offset + 0.3) % 360
        self.pulse = (self.pulse + 0.08) % 6.28
        self.update()
    
    def paintEvent(self, event):
        import math
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect()
        w, h = rect.width(), rect.height()
        
        # Dark gradient background
        gradient = QLinearGradient(0, 0, w, h)
        gradient.setColorAt(0.0, Colors.to_qcolor(Colors.BG))
        gradient.setColorAt(0.5, Colors.to_qcolor(Colors.BG_ALT))
        gradient.setColorAt(1.0, Colors.to_qcolor(Colors.BG_DARK))
        painter.fillRect(rect, QBrush(gradient))
        
        # Subtle grid pattern
        painter.setPen(QPen(Colors.to_qcolor(Colors.BORDER, 20), 1))
        grid_size = 40
        for x in range(0, w, grid_size):
            painter.drawLine(x, 0, x, h)
        for y in range(0, h, grid_size):
            painter.drawLine(0, y, w, y)
        
        # Draw connections (neural network lines)
        for i, j, dist in self.connections:
            n1, n2 = self.nodes[i], self.nodes[j]
            alpha = int(40 * (1 - dist / 0.35))
            pulse_factor = 0.5 + 0.5 * math.sin(self.pulse + n1['pulse_offset'])
            alpha = int(alpha * (0.6 + 0.4 * pulse_factor))
            
            painter.setPen(QPen(Colors.to_qcolor(Colors.PRIMARY_LIGHT, alpha), 1))
            painter.drawLine(
                int(n1['x'] * w), int(n1['y'] * h),
                int(n2['x'] * w), int(n2['y'] * h)
            )
        
        # Draw nodes (neural network points)
        for node in self.nodes:
            x, y = int(node['x'] * w), int(node['y'] * h)
            size = node['size']
            pulse_factor = 0.7 + 0.3 * math.sin(self.pulse + node['pulse_offset'])
            
            # Outer glow
            glow = QRadialGradient(x, y, size * 4)
            glow.setColorAt(0.0, Colors.to_qcolor(Colors.PRIMARY, int(30 * pulse_factor)))
            glow.setColorAt(1.0, Colors.to_qcolor(Colors.PRIMARY, 0))
            painter.setBrush(QBrush(glow))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(int(x - size * 4), int(y - size * 4), int(size * 8), int(size * 8))
            
            # Core node
            painter.setBrush(Colors.to_qcolor(Colors.PRIMARY_LIGHT, int(180 * pulse_factor)))
            painter.drawEllipse(int(x - size/2), int(y - size/2), int(size), int(size))
        
        # Large accent glow
        painter.setCompositionMode(QPainter.CompositionMode_Plus)
        
        glow1 = QRadialGradient(w * 0.7, h * 0.3, w * 0.5)
        glow1.setColorAt(0.0, Colors.to_qcolor(Colors.PRIMARY, 35))
        glow1.setColorAt(0.5, Colors.to_qcolor(Colors.PRIMARY, 10))
        glow1.setColorAt(1.0, Colors.to_qcolor(Colors.PRIMARY, 0))
        painter.fillRect(rect, QBrush(glow1))
        
        glow2 = QRadialGradient(w * 0.3, h * 0.7, w * 0.4)
        glow2.setColorAt(0.0, Colors.to_qcolor(Colors.PRIMARY_ALT, 25))
        glow2.setColorAt(0.5, Colors.to_qcolor(Colors.PRIMARY_ALT, 8))
        glow2.setColorAt(1.0, Colors.to_qcolor(Colors.PRIMARY_ALT, 0))
        painter.fillRect(rect, QBrush(glow2))


class GradientBackground(QWidget):
    """Custom widget for animated gradient background"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.offset = 0
        
        # Animation timer for subtle movement
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_gradient)
        self.timer.start(50)
    
    def update_gradient(self):
        self.offset = (self.offset + 0.5) % 360
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect()
        
        # Main gradient - using brand colors
        gradient = QLinearGradient(0, 0, rect.width(), rect.height())
        gradient.setColorAt(0.0, Colors.to_qcolor(Colors.BG_DARK))
        gradient.setColorAt(0.3, Colors.to_qcolor(Colors.BG))
        gradient.setColorAt(0.6, Colors.to_qcolor(Colors.BG_ALT))
        gradient.setColorAt(1.0, Colors.to_qcolor(Colors.BG_DARK))
        
        painter.fillRect(rect, QBrush(gradient))
        
        # Accent glow circles
        painter.setCompositionMode(QPainter.CompositionMode_Plus)
        
        # Primary purple glow - top right
        purple_glow = QRadialGradient(
            rect.width() * 0.8, 
            rect.height() * 0.2, 
            rect.width() * 0.5
        )
        purple_glow.setColorAt(0.0, Colors.to_qcolor(Colors.PRIMARY, 50))
        purple_glow.setColorAt(0.5, Colors.to_qcolor(Colors.PRIMARY, 20))
        purple_glow.setColorAt(1.0, Colors.to_qcolor(Colors.PRIMARY, 0))
        painter.fillRect(rect, QBrush(purple_glow))
        
        # Blue accent glow - bottom left
        blue_glow = QRadialGradient(
            rect.width() * 0.15, 
            rect.height() * 0.85, 
            rect.width() * 0.4
        )
        blue_glow.setColorAt(0.0, Colors.to_qcolor(Colors.PRIMARY_ALT, 40))
        blue_glow.setColorAt(0.5, Colors.to_qcolor(Colors.PRIMARY_ALT, 15))
        blue_glow.setColorAt(1.0, Colors.to_qcolor(Colors.PRIMARY_ALT, 0))
        painter.fillRect(rect, QBrush(blue_glow))
        
        # Light purple accent - center
        light_purple_glow = QRadialGradient(
            rect.width() * 0.5, 
            rect.height() * 0.5, 
            rect.width() * 0.35
        )
        light_purple_glow.setColorAt(0.0, Colors.to_qcolor(Colors.PRIMARY_LIGHT, 25))
        light_purple_glow.setColorAt(0.7, Colors.to_qcolor(Colors.PRIMARY_LIGHT, 8))
        light_purple_glow.setColorAt(1.0, Colors.to_qcolor(Colors.PRIMARY_LIGHT, 0))
        painter.fillRect(rect, QBrush(light_purple_glow))


class PasswordLineEdit(QWidget):
    """Password input with visibility toggle"""
    
    returnPressed = pyqtSignal()
    
    def __init__(self, placeholder="", parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Container for input and toggle
        self.container = QFrame()
        self.container.setFixedHeight(Spacing.INPUT_HEIGHT)
        self.container.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.CARD};
                border: 2px solid {Colors.BORDER_ALT};
                border-radius: {Spacing.RADIUS_LG}px;
            }}
            QFrame:hover {{
                border: 2px solid {Colors.PRIMARY};
                background-color: {Colors.SURFACE};
            }}
        """)
        
        container_layout = QHBoxLayout(self.container)
        container_layout.setContentsMargins(16, 0, 8, 0)
        container_layout.setSpacing(8)
        
        # Lock icon
        lock_icon = QLabel("🔒")
        lock_icon.setStyleSheet("background: transparent; border: none;")
        lock_icon.setFixedWidth(24)
        container_layout.addWidget(lock_icon)
        
        # Password input
        self.input = QLineEdit()
        self.input.setPlaceholderText(placeholder)
        self.input.setEchoMode(QLineEdit.Password)
        self.input.setFont(Fonts.body())
        self.input.setStyleSheet(f"""
            QLineEdit {{
                background: transparent;
                border: none;
                color: white;
                font-family: '{Fonts.FAMILY}';
                font-size: {Fonts.SIZE_MD}px;
                padding: 0;
            }}
        """)
        self.input.returnPressed.connect(self.returnPressed.emit)
        container_layout.addWidget(self.input, 1)
        
        # Toggle button
        self.toggle_btn = QPushButton("👁")
        self.toggle_btn.setFixedSize(36, 36)
        self.toggle_btn.setCursor(Qt.PointingHandCursor)
        self.toggle_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                border-radius: 18px;
                font-size: 16px;
            }}
            QPushButton:hover {{
                background: {Colors.SURFACE};
            }}
        """)
        self.toggle_btn.clicked.connect(self.toggle_visibility)
        container_layout.addWidget(self.toggle_btn)
        
        layout.addWidget(self.container)
        
        self._visible = False
    
    def toggle_visibility(self):
        self._visible = not self._visible
        if self._visible:
            self.input.setEchoMode(QLineEdit.Normal)
            self.toggle_btn.setText("🙈")
        else:
            self.input.setEchoMode(QLineEdit.Password)
            self.toggle_btn.setText("👁")
    
    def text(self):
        return self.input.text()
    
    def setFocus(self):
        self.input.setFocus()
    
    def focusInEvent(self, event):
        self.container.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.SURFACE};
                border: 2px solid {Colors.PRIMARY_LIGHT};
                border-radius: {Spacing.RADIUS_LG}px;
            }}
        """)
        super().focusInEvent(event)
    
    def focusOutEvent(self, event):
        self.container.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.CARD};
                border: 2px solid {Colors.BORDER_ALT};
                border-radius: {Spacing.RADIUS_LG}px;
            }}
            QFrame:hover {{
                border: 2px solid {Colors.PRIMARY};
                background-color: {Colors.SURFACE};
            }}
        """)
        super().focusOutEvent(event)


class EmailLineEdit(QWidget):
    """Email input with icon"""
    
    def __init__(self, placeholder="", parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Container for input
        self.container = QFrame()
        self.container.setFixedHeight(Spacing.INPUT_HEIGHT)
        self.container.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.CARD};
                border: 2px solid {Colors.BORDER_ALT};
                border-radius: {Spacing.RADIUS_LG}px;
            }}
            QFrame:hover {{
                border: 2px solid {Colors.PRIMARY};
                background-color: {Colors.SURFACE};
            }}
        """)
        
        container_layout = QHBoxLayout(self.container)
        container_layout.setContentsMargins(16, 0, 16, 0)
        container_layout.setSpacing(8)
        
        # Email icon
        email_icon = QLabel("✉")
        email_icon.setStyleSheet("background: transparent; border: none;")
        email_icon.setFixedWidth(24)
        container_layout.addWidget(email_icon)
        
        # Email input
        self.input = QLineEdit()
        self.input.setPlaceholderText(placeholder)
        self.input.setFont(Fonts.body())
        self.input.setStyleSheet(f"""
            QLineEdit {{
                background: transparent;
                border: none;
                color: white;
                font-family: '{Fonts.FAMILY}';
                font-size: {Fonts.SIZE_MD}px;
                padding: 0;
            }}
        """)
        container_layout.addWidget(self.input, 1)
        
        layout.addWidget(self.container)
    
    def text(self):
        return self.input.text()
    
    def setFocus(self):
        self.input.setFocus()
    
    def focusInEvent(self, event):
        self.container.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.SURFACE};
                border: 2px solid {Colors.PRIMARY_LIGHT};
                border-radius: {Spacing.RADIUS_LG}px;
            }}
        """)
        super().focusInEvent(event)
    
    def focusOutEvent(self, event):
        self.container.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.CARD};
                border: 2px solid {Colors.BORDER_ALT};
                border-radius: {Spacing.RADIUS_LG}px;
            }}
            QFrame:hover {{
                border: 2px solid {Colors.PRIMARY};
                background-color: {Colors.SURFACE};
            }}
        """)
        super().focusOutEvent(event)


class LoadingButton(QPushButton):
    """Button with loading spinner state"""
    
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._original_text = text
        self._loading = False
        self._dots = 0
        
        self.setMinimumHeight(Spacing.BUTTON_HEIGHT)
        self.setCursor(Qt.PointingHandCursor)
        self.setFont(Fonts.button())
        self.setStyleSheet(Styles.button())
        
        # Loading animation timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate_loading)
    
    def set_loading(self, loading: bool):
        self._loading = loading
        if loading:
            self.setEnabled(False)
            self._dots = 0
            self._timer.start(400)
            self._animate_loading()
        else:
            self._timer.stop()
            self.setText(self._original_text)
            self.setEnabled(True)
    
    def _animate_loading(self):
        dots = "." * (self._dots % 4)
        self.setText(f"Signing in{dots}")
        self._dots += 1


class GlassCard(QFrame):
    """AI-themed glassmorphism card with glowing border"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("glassCard")
        self._original_pos = None
        self._glow_intensity = 0.5
        self._pulse_direction = 1
        
        # Pulse animation for glow
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._pulse_glow)
        self._pulse_timer.start(50)
    
    def _pulse_glow(self):
        self._glow_intensity += 0.02 * self._pulse_direction
        if self._glow_intensity >= 1.0:
            self._pulse_direction = -1
        elif self._glow_intensity <= 0.3:
            self._pulse_direction = 1
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect()
        radius = 20
        
        # Outer glow effect
        glow_size = 30
        for i in range(glow_size, 0, -3):
            alpha = int(15 * self._glow_intensity * (1 - i / glow_size))
            painter.setPen(Qt.NoPen)
            painter.setBrush(Colors.to_qcolor(Colors.PRIMARY, alpha))
            painter.drawRoundedRect(
                rect.adjusted(-i, -i, i, i),
                radius + i//2, radius + i//2
            )
        
        # Main card background
        gradient = QLinearGradient(0, 0, 0, rect.height())
        gradient.setColorAt(0.0, Colors.to_qcolor(Colors.CARD, 240))
        gradient.setColorAt(0.5, Colors.to_qcolor(Colors.BG_ALT, 230))
        gradient.setColorAt(1.0, Colors.to_qcolor(Colors.CARD, 245))
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect, radius, radius)
        
        # Inner subtle gradient overlay
        inner_gradient = QLinearGradient(0, 0, rect.width(), 0)
        inner_gradient.setColorAt(0.0, Colors.to_qcolor(Colors.PRIMARY, 8))
        inner_gradient.setColorAt(0.5, Colors.to_qcolor(Colors.PRIMARY, 0))
        inner_gradient.setColorAt(1.0, Colors.to_qcolor(Colors.PRIMARY_ALT, 8))
        painter.setBrush(QBrush(inner_gradient))
        painter.drawRoundedRect(rect, radius, radius)
        
        # Glowing border
        border_gradient = QLinearGradient(0, 0, rect.width(), rect.height())
        border_alpha = int(100 + 80 * self._glow_intensity)
        border_gradient.setColorAt(0.0, Colors.to_qcolor(Colors.PRIMARY, border_alpha))
        border_gradient.setColorAt(0.5, Colors.to_qcolor(Colors.PRIMARY_LIGHT, int(border_alpha * 0.7)))
        border_gradient.setColorAt(1.0, Colors.to_qcolor(Colors.PRIMARY_ALT, border_alpha))
        
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QBrush(border_gradient), 2))
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), radius - 1, radius - 1)
        
        # Top highlight line
        highlight_gradient = QLinearGradient(rect.width() * 0.2, 0, rect.width() * 0.8, 0)
        highlight_gradient.setColorAt(0.0, Colors.to_qcolor(Colors.TEXT_PRIMARY, 0))
        highlight_gradient.setColorAt(0.5, Colors.to_qcolor(Colors.TEXT_PRIMARY, 40))
        highlight_gradient.setColorAt(1.0, Colors.to_qcolor(Colors.TEXT_PRIMARY, 0))
        
        painter.setPen(QPen(QBrush(highlight_gradient), 1))
        painter.drawLine(int(rect.width() * 0.2), 1, int(rect.width() * 0.8), 1)
        
        # Corner accents (AI data points feel)
        painter.setPen(Qt.NoPen)
        accent_alpha = int(60 * self._glow_intensity)
        painter.setBrush(Colors.to_qcolor(Colors.PRIMARY_LIGHT, accent_alpha))
        # Top-left corner accent
        painter.drawEllipse(8, 8, 4, 4)
        # Top-right corner accent  
        painter.drawEllipse(rect.width() - 12, 8, 4, 4)
        # Bottom-left corner accent
        painter.drawEllipse(8, rect.height() - 12, 4, 4)
        # Bottom-right corner accent
        painter.drawEllipse(rect.width() - 12, rect.height() - 12, 4, 4)
    
    def shake(self):
        """Shake animation for error feedback"""
        if self._original_pos is None:
            self._original_pos = self.pos()
        
        # Create shake animation
        anim = QPropertyAnimation(self, b"pos")
        anim.setDuration(500)
        anim.setEasingCurve(QEasingCurve.OutElastic)
        
        # Shake sequence
        start = self._original_pos
        anim.setKeyValueAt(0, start)
        anim.setKeyValueAt(0.1, start + QPoint(10, 0))
        anim.setKeyValueAt(0.2, start + QPoint(-10, 0))
        anim.setKeyValueAt(0.3, start + QPoint(8, 0))
        anim.setKeyValueAt(0.4, start + QPoint(-8, 0))
        anim.setKeyValueAt(0.5, start + QPoint(5, 0))
        anim.setKeyValueAt(0.6, start + QPoint(-5, 0))
        anim.setKeyValueAt(0.7, start + QPoint(2, 0))
        anim.setKeyValueAt(0.8, start + QPoint(-2, 0))
        anim.setKeyValueAt(1.0, start)
        
        anim.start()
        self._shake_anim = anim  # Keep reference


class LoginPage(ResponsiveWidget):
    """Modern login page with gradient background and glassmorphism design"""
    
    login_successful = pyqtSignal(str, str, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Login - 16Score-AI")
        self.responsive = ResponsiveUtils()
        self.init_ui()
        self._setup_shortcuts()
        
        # Auto-focus email field after a short delay
        QTimer.singleShot(100, self._focus_email)
    
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts for login page."""
        self.shortcuts = KeyboardShortcuts(self)
        # Enter to submit login form
        self.shortcuts.register("Return", self._handle_enter_key)

    def _focus_email(self):
        """Focus email input on load"""
        if hasattr(self, 'username_input'):
            self.username_input.setFocus()
    
    def _handle_enter_key(self):
        """Handle Enter key press to submit login."""
        if hasattr(self, 'login_btn') and self.login_btn.isEnabled():
            self.login()

    def init_ui(self):
        # Main container
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Gradient background
        self.background = GradientBackground(self)
        self.background.setGeometry(self.rect())
        
        # Content layer - horizontal split layout
        content_widget = QWidget(self)
        content_widget.setStyleSheet("background: transparent;")
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # Left side - Hero Panel (only on wide screens)
        self.hero_panel = HeroPanel()
        self.hero_panel.setMinimumWidth(420)
        content_layout.addWidget(self.hero_panel, 1)
        
        # Right side - Login form with AI background
        from PyQt5.QtWidgets import QStackedLayout
        
        right_panel = QWidget()
        right_panel.setStyleSheet("background: transparent;")
        right_stack = QStackedLayout(right_panel)
        right_stack.setStackingMode(QStackedLayout.StackAll)
        
        # AI Background layer (bottom)
        self.ai_background = AIBackground()
        right_stack.addWidget(self.ai_background)
        
        # Content layer (top)
        right_content = QWidget()
        right_content.setStyleSheet("background: transparent;")
        right_content.setAttribute(Qt.WA_TranslucentBackground)
        right_layout = QVBoxLayout(right_content)
        right_layout.setContentsMargins(Spacing.CARD_PADDING, Spacing.XXL, Spacing.CARD_PADDING, Spacing.CARD_PADDING)
        right_layout.setSpacing(0)
        
        # Top spacer
        right_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Logo section (centered in body)
        logo_container = QWidget()
        logo_container.setStyleSheet("background: transparent;")
        logo_layout = QVBoxLayout(logo_container)
        logo_layout.setContentsMargins(0, 0, 0, 0)
        logo_layout.setSpacing(Spacing.SM)
        
        # Logo image - try multiple paths
        logo = QLabel()
        logo_paths = [
            os.path.join(os.path.dirname(__file__), "../../../../images/splash-logo.png"),
            os.path.join(os.path.dirname(__file__), "../../../images/splash-logo.png"),
            "images/splash-logo.png"
        ]
        
        logo_loaded = False
        for logo_path in logo_paths:
            if os.path.exists(logo_path):
                pixmap = QPixmap(logo_path)
                if not pixmap.isNull():
                    logo.setPixmap(pixmap.scaled(280, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    logo_loaded = True
                    break
        
        if not logo_loaded:
            logo.setText("16Score-AI")
            logo.setFont(QFont(Fonts.FAMILY, 36, Fonts.WEIGHT_BOLD))
            logo.setStyleSheet(f"color: {Colors.PRIMARY}; background: transparent;")
        
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet("background: transparent;")
        logo_layout.addWidget(logo)
        
        # Tagline
        tagline = QLabel("ESPORTS ANALYTICS PLATFORM")
        tagline.setFont(Fonts.small())
        tagline.setStyleSheet(f"color: {Colors.TEXT_MUTED}; background: transparent; letter-spacing: 3px;")
        tagline.setAlignment(Qt.AlignCenter)
        logo_layout.addWidget(tagline)
        
        right_layout.addWidget(logo_container)
        right_layout.addSpacing(Spacing.XXL)

        # Center container for card
        center_container = QWidget()
        center_container.setStyleSheet("background: transparent;")
        card_center_layout = QHBoxLayout(center_container)
        card_center_layout.setContentsMargins(0, 0, 0, 0)
        
        # Horizontal spacers for centering
        card_center_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # Glass card
        self.login_card = GlassCard()
        self.login_card.setFixedWidth(420)
        self.login_card.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        
        card_layout = QVBoxLayout(self.login_card)
        card_layout.setContentsMargins(Spacing.CARD_PADDING, Spacing.CARD_PADDING, Spacing.CARD_PADDING, Spacing.CARD_PADDING)
        card_layout.setSpacing(Spacing.XL)

        # Welcome text
        welcome_label = QLabel("Welcome Back")
        welcome_label.setFont(Fonts.title())
        welcome_label.setStyleSheet(f"color: white; background: transparent; letter-spacing: -0.5px;")
        welcome_label.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(welcome_label)
        
        # Subtitle
        subtitle = QLabel("Sign in to continue to your dashboard")
        subtitle.setFont(Fonts.create(Fonts.SIZE_SM))
        subtitle.setStyleSheet(f"color: {Colors.with_alpha('#FFFFFF', 0.5)}; background: transparent;")
        subtitle.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(subtitle)
        
        card_layout.addSpacing(Spacing.MD)
        
        # Email field
        email_container = QWidget()
        email_container.setStyleSheet("background: transparent;")
        email_layout = QVBoxLayout(email_container)
        email_layout.setContentsMargins(0, 0, 0, 0)
        email_layout.setSpacing(Spacing.XS)
        
        email_label = QLabel("Email")
        email_label.setFont(Fonts.label())
        email_label.setStyleSheet(f"color: {Colors.with_alpha('#FFFFFF', 0.7)}; background: transparent; letter-spacing: 0.5px;")
        email_layout.addWidget(email_label)

        self.username_input = EmailLineEdit(placeholder="name@company.com")
        email_layout.addWidget(self.username_input)
        card_layout.addWidget(email_container)
        
        # Password field
        password_container = QWidget()
        password_container.setStyleSheet("background: transparent;")
        password_layout = QVBoxLayout(password_container)
        password_layout.setContentsMargins(0, 0, 0, 0)
        password_layout.setSpacing(Spacing.XS)
        
        password_label = QLabel("Password")
        password_label.setFont(Fonts.label())
        password_label.setStyleSheet(f"color: {Colors.with_alpha('#FFFFFF', 0.7)}; background: transparent; letter-spacing: 0.5px;")
        password_layout.addWidget(password_label)

        self.password_input = PasswordLineEdit(placeholder="Enter your password")
        self.password_input.returnPressed.connect(self.login)
        password_layout.addWidget(self.password_input)
        card_layout.addWidget(password_container)
        
        # Remember me row
        remember_row = QWidget()
        remember_row.setStyleSheet("background: transparent;")
        remember_layout = QHBoxLayout(remember_row)
        remember_layout.setContentsMargins(0, Spacing.SM, 0, Spacing.SM)
        
        self.remember_checkbox = QCheckBox("Remember me")
        self.remember_checkbox.setFont(Fonts.label())
        self.remember_checkbox.setStyleSheet(Styles.checkbox())
        remember_layout.addWidget(self.remember_checkbox)
        
        remember_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        forgot_link = QLabel("Forgot password?")
        forgot_link.setFont(Fonts.label())
        forgot_link.setStyleSheet(f"color: {Colors.PRIMARY_LIGHT}; background: transparent;")
        forgot_link.setCursor(Qt.PointingHandCursor)
        remember_layout.addWidget(forgot_link)
        
        card_layout.addWidget(remember_row)
        
        card_layout.addSpacing(Spacing.XS)
        
        # Login button with loading state
        self.login_btn = LoadingButton("Sign In")
        self.login_btn.clicked.connect(self.login)
        card_layout.addWidget(self.login_btn)

        # Error label
        self.error_label = QLabel("")
        self.error_label.setFont(Fonts.label())
        self.error_label.setStyleSheet(Styles.error_label())
        self.error_label.setAlignment(Qt.AlignCenter)
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        card_layout.addWidget(self.error_label)

        card_center_layout.addWidget(self.login_card)
        card_center_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        right_layout.addWidget(center_container)
        
        # Bottom spacer
        right_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        # Footer
        footer = QLabel("© 2024-2025 16Score-AI. All rights reserved.")
        footer.setFont(Fonts.small())
        footer.setStyleSheet(f"color: {Colors.TEXT_DISABLED}; background: transparent;")
        footer.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(footer)
        
        # Add content to stack
        right_stack.addWidget(right_content)
        right_stack.setCurrentIndex(1)  # Show content on top
        
        # Add right panel to content
        content_layout.addWidget(right_panel, 1)
        
        main_layout.addWidget(content_widget)

    def login(self):
        print(f"[LOGIN] Login button clicked!", flush=True)
        
        # Hide previous error
        self.error_label.setVisible(False)
        
        username = self.username_input.text().strip()
        password = self.password_input.text().strip()
        print(f"[LOGIN] Username: {username}, Password length: {len(password)}", flush=True)

        # Validate email format
        is_valid_email, email_error = InputValidator.validate_email(username)
        print(f"[LOGIN] Email validation: valid={is_valid_email}, error={email_error}", flush=True)
        if not is_valid_email:
            self.show_error(email_error)
            return
        
        # Validate password
        is_valid_password, password_error = InputValidator.validate_password(password)
        print(f"[LOGIN] Password validation: valid={is_valid_password}, error={password_error}", flush=True)
        if not is_valid_password:
            self.show_error(password_error)
            return

        # Show loading state
        self.login_btn.set_loading(True)
        
        # Force UI update
        QApplication.processEvents()
        
        try:
            print(f"[LOGIN] Attempting login for: {username}", flush=True)
            response = user_api.login(username, password)
            print(f"[LOGIN] Response: success={response.success}", flush=True)
            
            if response.success:
                token = response.data["data"]["token"]
                user_api.set_auth_token(token)
                print(f"[LOGIN] Token received and set", flush=True)
                
                # Store token securely if available
                if self.remember_checkbox.isChecked():
                    SecureStorage.store_token(username, token)

                user_response = user_api.get_user_from_token()
                user_id = (
                    user_response.data.get("data", {}).get("username")
                    if user_response.success
                    else None
                )
                print(f"[LOGIN] User ID: {user_id}", flush=True)

                save_login_history(username, user_id)
                print(f"[LOGIN] Emitting login_successful signal...", flush=True)
                self.login_successful.emit(token, username, user_id)
                print(f"[LOGIN] Signal emitted!", flush=True)
                self.error_label.setVisible(False)
            else:
                error_msg = response.data.get("message", "Invalid credentials") if response.data else "Invalid credentials"
                print(f"[LOGIN] Login failed: {error_msg}", flush=True)
                self.show_error(error_msg)
        except Exception as e:
            import traceback
            print(f"[LOGIN] Exception: {e}", flush=True)
            traceback.print_exc()
            self.show_error(f"Connection error: {str(e)}")
        finally:
            self.login_btn.set_loading(False)
    
    def show_error(self, message):
        """Display error message with shake animation"""
        self.error_label.setText(message)
        self.error_label.setVisible(True)
        
        # Shake the card for visual feedback
        self.login_card.shake()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Update background size
        if hasattr(self, 'background'):
            self.background.setGeometry(self.rect())
        
        screen_width = self.width()
        
        # Show/hide hero panel based on width
        if hasattr(self, 'hero_panel'):
            if screen_width < 900:
                self.hero_panel.hide()
            else:
                self.hero_panel.show()
        
        # Responsive card width
        if hasattr(self, 'login_card'):
            if screen_width < 520:
                self.login_card.setFixedWidth(max(screen_width - 40, 320))
            else:
                self.login_card.setFixedWidth(420)
    
            # Update original position for shake animation
            self.login_card._original_pos = None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LoginPage()
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec_())

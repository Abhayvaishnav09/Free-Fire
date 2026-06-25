"""
Home Page with AI-themed UI

Features:
- AI neural network animated background
- Glowing action card with hover effects
- Stats display section
- Clean visual hierarchy
"""
from __future__ import annotations

import math

from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QFrame,
    QSizePolicy,
    QGraphicsDropShadowEffect,
)
from PyQt5.QtGui import (
    QCursor,
    QPainter,
    QBrush,
    QPen,
    QLinearGradient,
    QRadialGradient,
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer

from score_ai.ui.components.header_widget import HeaderWidget
from score_ai.utils.responsive_utils import ResponsiveUtils, ResponsiveWidget
from score_ai.core.theme import Colors, Fonts, Spacing, Gradients, Shadows, Styles


class AIHomeBackground(QWidget):
    """AI-themed animated background for home page"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._pulse = 0
        self._init_nodes()
        
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate)
        self._timer.start(50)
    
    def _init_nodes(self):
        import random
        random.seed(456)  # Unique seed for home page
        self.nodes = []
        for _ in range(18):
            self.nodes.append({
                'x': random.uniform(0.03, 0.97),
                'y': random.uniform(0.08, 0.92),
                'size': random.uniform(2, 5),
                'pulse_offset': random.uniform(0, 6.28)
            })
        
        self.connections = []
        for i, n1 in enumerate(self.nodes):
            for j, n2 in enumerate(self.nodes):
                if i < j:
                    dist = ((n1['x'] - n2['x'])**2 + (n1['y'] - n2['y'])**2)**0.5
                    if dist < 0.32:
                        self.connections.append((i, j, dist))
    
    def _animate(self):
        self._pulse = (self._pulse + 0.05) % 6.28
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect()
        w, h = rect.width(), rect.height()
        
        # Dark gradient background
        gradient = QLinearGradient(0, 0, w, h)
        gradient.setColorAt(0.0, Colors.to_qcolor(Colors.BG_DARK))
        gradient.setColorAt(0.35, Colors.to_qcolor(Colors.BG))
        gradient.setColorAt(0.65, Colors.to_qcolor(Colors.BG_ALT))
        gradient.setColorAt(1.0, Colors.to_qcolor(Colors.BG_DARK))
        painter.fillRect(rect, QBrush(gradient))
        
        # Subtle grid
        painter.setPen(QPen(Colors.to_qcolor(Colors.BORDER, 10), 1))
        grid_size = 55
        for x in range(0, w, grid_size):
            painter.drawLine(x, 0, x, h)
        for y in range(0, h, grid_size):
            painter.drawLine(0, y, w, y)
        
        # Draw connections
        for i, j, dist in self.connections:
            n1, n2 = self.nodes[i], self.nodes[j]
            alpha = int(28 * (1 - dist / 0.32))
            pulse_factor = 0.5 + 0.5 * math.sin(self._pulse + n1['pulse_offset'])
            alpha = int(alpha * (0.4 + 0.6 * pulse_factor))
            
            painter.setPen(QPen(Colors.to_qcolor(Colors.PRIMARY_LIGHT, alpha), 1))
            painter.drawLine(int(n1['x'] * w), int(n1['y'] * h), int(n2['x'] * w), int(n2['y'] * h))
        
        # Draw nodes
        for node in self.nodes:
            x, y = int(node['x'] * w), int(node['y'] * h)
            size = node['size']
            pulse_factor = 0.5 + 0.5 * math.sin(self._pulse + node['pulse_offset'])
            
            # Glow
            glow = QRadialGradient(x, y, size * 4)
            glow.setColorAt(0.0, Colors.to_qcolor(Colors.PRIMARY, int(18 * pulse_factor)))
            glow.setColorAt(1.0, Colors.to_qcolor(Colors.PRIMARY, 0))
            painter.setBrush(QBrush(glow))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(int(x - size * 4), int(y - size * 4), int(size * 8), int(size * 8))
            
            # Core
            painter.setBrush(Colors.to_qcolor(Colors.PRIMARY_LIGHT, int(130 * pulse_factor)))
            painter.drawEllipse(int(x - size/2), int(y - size/2), int(size), int(size))
        
        # Accent glows
        painter.setCompositionMode(QPainter.CompositionMode_Plus)
        
        glow1 = QRadialGradient(w * 0.8, h * 0.2, w * 0.4)
        glow1.setColorAt(0.0, Colors.to_qcolor(Colors.PRIMARY, 22))
        glow1.setColorAt(0.5, Colors.to_qcolor(Colors.PRIMARY, 6))
        glow1.setColorAt(1.0, Colors.to_qcolor(Colors.PRIMARY, 0))
        painter.fillRect(rect, QBrush(glow1))
        
        glow2 = QRadialGradient(w * 0.2, h * 0.8, w * 0.35)
        glow2.setColorAt(0.0, Colors.to_qcolor(Colors.PRIMARY_ALT, 18))
        glow2.setColorAt(0.5, Colors.to_qcolor(Colors.PRIMARY_ALT, 5))
        glow2.setColorAt(1.0, Colors.to_qcolor(Colors.PRIMARY_ALT, 0))
        painter.fillRect(rect, QBrush(glow2))


class GlowingActionCard(QFrame):
    """Action card with glowing hover effect"""
    
    clicked = pyqtSignal()
    
    def __init__(self, title: str, description: str, parent=None):
        super().__init__(parent)
        self.title_text = title
        self.description_text = description
        self._glow_alpha = 0
        self._hovered = False
        
        self.setMinimumSize(420, 180)
        self.setMaximumSize(560, 220)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setAttribute(Qt.WA_Hover, True)
        
        self._setup_ui()
        
        # Glow animation
        self._glow_timer = QTimer(self)
        self._glow_timer.timeout.connect(self._update_glow)
        self._glow_timer.start(30)
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(24)
        
        # Icon
        icon_container = QWidget()
        icon_container.setFixedSize(80, 80)
        icon_container.setStyleSheet(f"""
            background: {Gradients.primary_diagonal()};
            border-radius: 16px;
        """)
        
        # Reticle icon inside
        icon_layout = QVBoxLayout(icon_container)
        icon_layout.setContentsMargins(0, 0, 0, 0)
        icon_layout.setAlignment(Qt.AlignCenter)
        
        # Simple targeting symbol
        reticle = QLabel("+")
        reticle.setFont(Fonts.create(36, Fonts.WEIGHT_BOLD))
        reticle.setStyleSheet("color: white; background: transparent;")
        reticle.setAlignment(Qt.AlignCenter)
        icon_layout.addWidget(reticle)
        
        layout.addWidget(icon_container)
        
        # Text section
        text_layout = QVBoxLayout()
        text_layout.setSpacing(8)
        text_layout.setContentsMargins(0, 0, 0, 0)
        
        title_label = QLabel(self.title_text)
        title_label.setFont(Fonts.create(20, Fonts.WEIGHT_BOLD))
        title_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent;")
        text_layout.addWidget(title_label)
        
        desc_label = QLabel(self.description_text)
        desc_label.setFont(Fonts.create(13))
        desc_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
        desc_label.setWordWrap(True)
        text_layout.addWidget(desc_label)
        
        text_layout.addStretch()
        
        # Arrow indicator
        arrow = QLabel("Start")
        arrow.setFont(Fonts.create(12, Fonts.WEIGHT_DEMIBOLD))
        arrow.setStyleSheet(f"color: {Colors.PRIMARY_LIGHT}; background: transparent;")
        text_layout.addWidget(arrow)
        
        layout.addLayout(text_layout, 1)
    
    def _update_glow(self):
        target = 100 if self._hovered else 0
        diff = target - self._glow_alpha
        if abs(diff) > 2:
            self._glow_alpha += diff * 0.15
            self.update()
        elif self._glow_alpha != target:
            self._glow_alpha = target
            self.update()
    
    def enterEvent(self, event):
        self._hovered = True
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        self._hovered = False
        super().leaveEvent(event)
    
    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect()
        radius = 20
        
        # Outer glow when hovered
        if self._glow_alpha > 0:
            glow = QRadialGradient(rect.center().x(), rect.center().y(), max(rect.width(), rect.height()) * 0.65)
            glow.setColorAt(0.0, Colors.to_qcolor(Colors.PRIMARY, int(self._glow_alpha * 0.5)))
            glow.setColorAt(0.4, Colors.to_qcolor(Colors.PRIMARY, int(self._glow_alpha * 0.25)))
            glow.setColorAt(1.0, Colors.to_qcolor(Colors.PRIMARY, 0))
            painter.setBrush(QBrush(glow))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(rect.adjusted(-12, -12, 12, 12), radius + 12, radius + 12)
        
        # Card background
        bg_color = Colors.to_qcolor(Colors.SURFACE, 240) if self._hovered else Colors.to_qcolor(Colors.CARD, 230)
        painter.setBrush(QBrush(bg_color))
        border_color = Colors.to_qcolor(Colors.PRIMARY, int(50 + self._glow_alpha * 0.5)) if self._hovered else Colors.to_qcolor(Colors.BORDER, 100)
        painter.setPen(QPen(border_color, 1.5))
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), radius, radius)
        
        # Top highlight
        highlight = QLinearGradient(0, 0, 0, 4)
        highlight.setColorAt(0, Colors.to_qcolor("#FFFFFF", 18))
        highlight.setColorAt(1, Colors.to_qcolor("#FFFFFF", 0))
        painter.setBrush(QBrush(highlight))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect.adjusted(2, 2, -2, -rect.height() + 24), radius - 2, radius - 2)
        
        super().paintEvent(event)


class HomeScreen(ResponsiveWidget):
    """Home page with AI-themed design"""
    
    open_tournament_requested = pyqtSignal()
    home_nav_requested = pyqtSignal()

    def __init__(
        self,
        token: str,
        user_email: str,
        user_id: str = None,
        on_logout=None,
        on_docs_click=None
    ):
        super().__init__()
        self.token = token
        self.user_email = user_email
        self.user_id = user_id
        self.on_logout = on_logout
        self.on_docs_click = on_docs_click
        self.responsive = ResponsiveUtils()
        
        self.setWindowTitle("16Score-AI - Home")
        self._init_ui()

    def _init_ui(self) -> None:
        """Initialize the page UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header
        main_layout.addWidget(self._create_header())
        
        # Content area with AI background
        content_container = QWidget()
        content_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content_stack = QVBoxLayout(content_container)
        content_stack.setContentsMargins(0, 0, 0, 0)
        content_stack.setSpacing(0)
        
        # AI Background
        self.bg_widget = AIHomeBackground()
        self.bg_widget.setParent(content_container)
        self.bg_widget.lower()
        
        # Content overlay
        overlay = QWidget()
        overlay.setStyleSheet("background: transparent;")
        overlay.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        overlay_layout = QVBoxLayout(overlay)
        overlay_layout.setContentsMargins(40, 40, 40, 40)
        overlay_layout.setSpacing(32)
        
        # Welcome section
        welcome_section = self._create_welcome_section()
        overlay_layout.addWidget(welcome_section)
        
        overlay_layout.addStretch()
        
        # Main action card - centered
        card_container = QWidget()
        card_container.setStyleSheet("background: transparent;")
        card_layout = QHBoxLayout(card_container)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.addStretch()
        
        action_card = GlowingActionCard(
            "Kill Feed Capture",
            "16scoreAI instantly captures and calculates points by analyzing the killfeed data, ensuring accurate point determination in real-time."
        )
        action_card.clicked.connect(self._open_kill_feed_capture)
        card_layout.addWidget(action_card)
        
        card_layout.addStretch()
        overlay_layout.addWidget(card_container)
        
        overlay_layout.addStretch()
        
        # Stats section at bottom
        stats_section = self._create_stats_section()
        overlay_layout.addWidget(stats_section)
        
        content_stack.addWidget(overlay)
        main_layout.addWidget(content_container, 1)

    def _create_header(self) -> QWidget:
        """Create the header container."""
        header_container = QWidget()
        header_container.setFixedHeight(56)
        header_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        header_container.setStyleSheet(f"background: {Colors.BG_DARK};")
        
        header_layout = QVBoxLayout(header_container)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(0)
        
        header = HeaderWidget(
            self.user_email,
            on_logout=self._handle_logout,
            on_home_click=self._handle_home_click,
            on_docs_click=self.on_docs_click
        )
        header_layout.addWidget(header)
        
        return header_container

    def _create_welcome_section(self) -> QWidget:
        """Create welcome section with title."""
        section = QWidget()
        section.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Welcome text
        welcome = QLabel("Welcome to")
        welcome.setFont(Fonts.create(14))
        welcome.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
        layout.addWidget(welcome)
        
        # Main title
        title = QLabel("16Score AI")
        title.setFont(Fonts.create(42, Fonts.WEIGHT_BOLD))
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent;")
        layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("AI-Powered Tournament Scoring Platform")
        subtitle.setFont(Fonts.create(16))
        subtitle.setStyleSheet(f"color: {Colors.PRIMARY_LIGHT}; background: transparent;")
        layout.addWidget(subtitle)
        
        return section

    def _create_stats_section(self) -> QWidget:
        """Create stats display at bottom."""
        stats_container = QWidget()
        stats_container.setFixedHeight(80)
        stats_container.setStyleSheet(f"""
            background: {Colors.with_alpha(Colors.CARD, 0.4)};
            border: 1px solid {Colors.with_alpha(Colors.BORDER, 0.4)};
            border-radius: 16px;
        """)
        
        stats_layout = QHBoxLayout(stats_container)
        stats_layout.setContentsMargins(40, 0, 40, 0)
        stats_layout.setSpacing(60)
        
        stats = [
            ("60 FPS", "Processing Speed", Colors.SUCCESS),
            ("<50ms", "Latency", Colors.PRIMARY_LIGHT),
            ("99%+", "Accuracy", Colors.GOLD),
            ("Cloud", "Sync", Colors.INFO),
        ]
        
        for i, (value, label, color) in enumerate(stats):
            stat_widget = QWidget()
            stat_widget.setStyleSheet("background: transparent;")
            stat_layout = QHBoxLayout(stat_widget)
            stat_layout.setContentsMargins(0, 0, 0, 0)
            stat_layout.setSpacing(10)
            
            value_label = QLabel(value)
            value_label.setFont(Fonts.create(18, Fonts.WEIGHT_BOLD))
            value_label.setStyleSheet(f"color: {color}; background: transparent;")
            stat_layout.addWidget(value_label)
            
            desc_label = QLabel(label)
            desc_label.setFont(Fonts.create(12))
            desc_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; background: transparent;")
            stat_layout.addWidget(desc_label)
            
            stats_layout.addWidget(stat_widget)
            
            if i < len(stats) - 1:
                separator = QFrame()
                separator.setFixedWidth(1)
                separator.setStyleSheet(f"background: {Colors.BORDER};")
                stats_layout.addWidget(separator)
        
        stats_layout.addStretch()
        
        return stats_container

    def _open_kill_feed_capture(self) -> None:
        """Open the kill feed capture workflow."""
        self.open_tournament_requested.emit()

    def _handle_logout(self) -> None:
        """Handle logout action."""
        if self.on_logout:
            self.on_logout()
        else:
            self.close()

    def _handle_home_click(self) -> None:
        """Handle home navigation."""
        self.home_nav_requested.emit()

    def resizeEvent(self, event) -> None:
        """Handle window resize."""
        super().resizeEvent(event)
        if hasattr(self, 'bg_widget'):
            content = self.layout().itemAt(1).widget()
            if content:
                self.bg_widget.setGeometry(content.rect())

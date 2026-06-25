"""
Streaming Source Selection Page with AI-themed UI

Features:
- AI neural network animated background
- Glowing source option cards
- Modern radio button styling
"""
from __future__ import annotations

import json
import math
import os
import sys

from PyQt5.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
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
from score_ai.core.theme import Colors, Fonts, Gradients, Spacing, Styles

if getattr(sys, "frozen", False):
    base_path = sys._MEIPASS
    sys.path.insert(0, os.path.join(base_path, "src"))


class AISourceBackground(QWidget):
    """AI-themed animated background"""
    
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
        random.seed(321)
        self.nodes = []
        for _ in range(14):
            self.nodes.append({
                'x': random.uniform(0.05, 0.95),
                'y': random.uniform(0.08, 0.92),
                'size': random.uniform(2, 4),
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
        
        gradient = QLinearGradient(0, 0, w, h)
        gradient.setColorAt(0.0, Colors.to_qcolor(Colors.BG_DARK))
        gradient.setColorAt(0.4, Colors.to_qcolor(Colors.BG))
        gradient.setColorAt(0.7, Colors.to_qcolor(Colors.BG_ALT))
        gradient.setColorAt(1.0, Colors.to_qcolor(Colors.BG_DARK))
        painter.fillRect(rect, QBrush(gradient))
        
        painter.setPen(QPen(Colors.to_qcolor(Colors.BORDER, 10), 1))
        grid_size = 50
        for x in range(0, w, grid_size):
            painter.drawLine(x, 0, x, h)
        for y in range(0, h, grid_size):
            painter.drawLine(0, y, w, y)
        
        for i, j, dist in self.connections:
            n1, n2 = self.nodes[i], self.nodes[j]
            alpha = int(25 * (1 - dist / 0.32))
            pulse_factor = 0.5 + 0.5 * math.sin(self._pulse + n1['pulse_offset'])
            alpha = int(alpha * (0.4 + 0.6 * pulse_factor))
            painter.setPen(QPen(Colors.to_qcolor(Colors.PRIMARY_LIGHT, alpha), 1))
            painter.drawLine(int(n1['x'] * w), int(n1['y'] * h), int(n2['x'] * w), int(n2['y'] * h))
        
        for node in self.nodes:
            x, y = int(node['x'] * w), int(node['y'] * h)
            size = node['size']
            pulse_factor = 0.5 + 0.5 * math.sin(self._pulse + node['pulse_offset'])
            
            glow = QRadialGradient(x, y, size * 4)
            glow.setColorAt(0.0, Colors.to_qcolor(Colors.PRIMARY, int(15 * pulse_factor)))
            glow.setColorAt(1.0, Colors.to_qcolor(Colors.PRIMARY, 0))
            painter.setBrush(QBrush(glow))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(int(x - size * 4), int(y - size * 4), int(size * 8), int(size * 8))
            
            painter.setBrush(Colors.to_qcolor(Colors.PRIMARY_LIGHT, int(120 * pulse_factor)))
            painter.drawEllipse(int(x - size/2), int(y - size/2), int(size), int(size))
        
        painter.setCompositionMode(QPainter.CompositionMode_Plus)
        glow1 = QRadialGradient(w * 0.5, h * 0.3, w * 0.4)
        glow1.setColorAt(0.0, Colors.to_qcolor(Colors.PRIMARY, 18))
        glow1.setColorAt(1.0, Colors.to_qcolor(Colors.PRIMARY, 0))
        painter.fillRect(rect, QBrush(glow1))


class SourceOptionCard(QFrame):
    """Source option card with selection indicator"""
    
    selected_signal = pyqtSignal()
    
    def __init__(self, title: str, description: str, icon_text: str = "O", parent=None):
        super().__init__(parent)
        self.title_text = title
        self.description_text = description
        self.icon_text = icon_text
        self._selected = False
        self._glow_alpha = 0
        self._hovered = False
        
        self.setFixedSize(380, 120)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setAttribute(Qt.WA_Hover, True)
        
        self._setup_ui()
        
        self._glow_timer = QTimer(self)
        self._glow_timer.timeout.connect(self._update_glow)
        self._glow_timer.start(30)
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)
        
        # Icon
        icon = QLabel(self.icon_text)
        icon.setFixedSize(56, 56)
        icon.setAlignment(Qt.AlignCenter)
        icon.setFont(Fonts.create(24, Fonts.WEIGHT_BOLD))
        icon.setStyleSheet(f"""
            background: {Gradients.primary_diagonal()};
            color: white;
            border-radius: 12px;
        """)
        layout.addWidget(icon)
        
        # Text
        text_layout = QVBoxLayout()
        text_layout.setSpacing(4)
        text_layout.setContentsMargins(0, 0, 0, 0)
        
        title = QLabel(self.title_text)
        title.setFont(Fonts.create(16, Fonts.WEIGHT_BOLD))
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent;")
        text_layout.addWidget(title)
        
        desc = QLabel(self.description_text)
        desc.setFont(Fonts.create(12))
        desc.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
        desc.setWordWrap(True)
        text_layout.addWidget(desc)
        
        layout.addLayout(text_layout, 1)
        
        # Selection indicator
        self.indicator = QLabel()
        self.indicator.setFixedSize(24, 24)
        self._update_indicator()
        layout.addWidget(self.indicator)
    
    def _update_indicator(self):
        if self._selected:
            self.indicator.setStyleSheet(f"""
                background: {Colors.PRIMARY};
                border: 2px solid {Colors.PRIMARY_LIGHT};
                border-radius: 12px;
            """)
        else:
            self.indicator.setStyleSheet(f"""
                background: transparent;
                border: 2px solid {Colors.BORDER};
                border-radius: 12px;
            """)
    
    def set_selected(self, selected: bool):
        self._selected = selected
        self._update_indicator()
        self.update()
    
    def is_selected(self) -> bool:
        return self._selected
    
    def _update_glow(self):
        target = 80 if self._hovered or self._selected else 0
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
        self.selected_signal.emit()
        super().mousePressEvent(event)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect()
        radius = 16
        
        # Glow
        if self._glow_alpha > 0:
            glow = QRadialGradient(rect.center().x(), rect.center().y(), max(rect.width(), rect.height()) * 0.6)
            glow.setColorAt(0.0, Colors.to_qcolor(Colors.PRIMARY, int(self._glow_alpha * 0.4)))
            glow.setColorAt(0.5, Colors.to_qcolor(Colors.PRIMARY, int(self._glow_alpha * 0.15)))
            glow.setColorAt(1.0, Colors.to_qcolor(Colors.PRIMARY, 0))
            painter.setBrush(QBrush(glow))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(rect.adjusted(-8, -8, 8, 8), radius + 8, radius + 8)
        
        # Background
        bg_color = Colors.to_qcolor(Colors.SURFACE, 235) if self._hovered or self._selected else Colors.to_qcolor(Colors.CARD, 225)
        painter.setBrush(QBrush(bg_color))
        
        border_color = Colors.to_qcolor(Colors.PRIMARY, int(60 + self._glow_alpha * 0.4)) if self._selected or self._hovered else Colors.to_qcolor(Colors.BORDER, 90)
        painter.setPen(QPen(border_color, 1.5 if self._selected else 1))
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), radius, radius)
        
        # Top highlight
        highlight = QLinearGradient(0, 0, 0, 4)
        highlight.setColorAt(0, Colors.to_qcolor("#FFFFFF", 12))
        highlight.setColorAt(1, Colors.to_qcolor("#FFFFFF", 0))
        painter.setBrush(QBrush(highlight))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect.adjusted(2, 2, -2, -rect.height() + 18), radius - 2, radius - 2)
        
        super().paintEvent(event)


class StreamingSourcePage(QWidget):
    """Streaming source selection page with AI-themed UI."""
    
    selected = pyqtSignal(dict)
    back_requested = pyqtSignal()

    def __init__(
        self,
        match_info: dict = None,
        user_email: str = "user@email.com",
        on_logout=None,
        on_home_click=None,
        on_docs_click=None
    ):
        super().__init__()
        self.match_info = match_info or {}
        self.selected_source = None
        self.user_email = user_email
        self.on_logout = on_logout
        self.on_home_click = on_home_click
        self.on_docs_click = on_docs_click
        
        self.setWindowTitle("Select Input Source - 16Score-AI")
        self._init_ui()

    def _init_ui(self) -> None:
        """Initialize the page UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header
        main_layout.addWidget(self._create_header())
        
        # Content with AI background
        content_container = QWidget()
        content_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content_stack = QVBoxLayout(content_container)
        content_stack.setContentsMargins(0, 0, 0, 0)
        content_stack.setSpacing(0)
        
        self.bg_widget = AISourceBackground()
        self.bg_widget.setParent(content_container)
        self.bg_widget.lower()
        
        # Overlay
        overlay = QWidget()
        overlay.setStyleSheet("background: transparent;")
        overlay.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        overlay_layout = QVBoxLayout(overlay)
        overlay_layout.setContentsMargins(40, 60, 40, 40)
        overlay_layout.setSpacing(32)
        overlay_layout.setAlignment(Qt.AlignCenter)
        
        # Title section
        title_section = QVBoxLayout()
        title_section.setSpacing(8)
        title_section.setAlignment(Qt.AlignCenter)
        
        title = QLabel("Select Input Source")
        title.setFont(Fonts.create(32, Fonts.WEIGHT_BOLD))
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent;")
        title.setAlignment(Qt.AlignCenter)
        title_section.addWidget(title)
        
        subtitle = QLabel("Choose your preferred streaming source")
        subtitle.setFont(Fonts.create(14))
        subtitle.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
        subtitle.setAlignment(Qt.AlignCenter)
        title_section.addWidget(subtitle)
        
        overlay_layout.addLayout(title_section)
        overlay_layout.addSpacing(20)
        
        # Source options
        self.source_cards = []
        
        obs_card = SourceOptionCard(
            "OBS Camera",
            "Capture from OBS virtual camera for high-quality streaming",
            "O"
        )
        obs_card.selected_signal.connect(lambda: self._select_source(obs_card, "OBS Camera"))
        obs_card.set_selected(True)  # Default selection
        self.selected_source = "OBS Camera"
        self.source_cards.append(obs_card)
        overlay_layout.addWidget(obs_card, alignment=Qt.AlignCenter)
        
        overlay_layout.addSpacing(24)
        
        # Game selection
        game_row = QHBoxLayout()
        game_row.setSpacing(16)
        game_row.setAlignment(Qt.AlignCenter)
        game_label = QLabel("Game")
        game_label.setFont(Fonts.create(14, Fonts.WEIGHT_DEMIBOLD))
        game_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent;")
        game_row.addWidget(game_label)
        self.game_combo = QComboBox()
        self.game_combo.addItems(["Free Fire", "BGMI"])
        config_path = os.path.join(os.path.expanduser("~"), ".esports_ai_config.json")
        current_game = "freefire"
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    current_game = data.get("game", "freefire")
            except Exception:
                pass
        self.game_combo.setCurrentIndex(1 if current_game == "bgmi" else 0)
        self.game_combo.setFixedWidth(140)
        self.game_combo.setStyleSheet(f"""
            QComboBox {{
                background: {Colors.SURFACE};
                border: 1px solid {Colors.BORDER};
                border-radius: {Spacing.RADIUS_MD}px;
                color: {Colors.TEXT_PRIMARY};
                padding: 8px 12px;
                font-size: 13px;
            }}
            QComboBox:focus {{ border-color: {Colors.PRIMARY}; }}
            QComboBox::drop-down {{ border: none; width: 30px; }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid {Colors.TEXT_SECONDARY};
            }}
            QComboBox QAbstractItemView {{
                background: {Colors.CARD};
                border: 1px solid {Colors.BORDER};
                border-radius: {Spacing.RADIUS_MD}px;
                color: {Colors.TEXT_PRIMARY};
                selection-background-color: {Colors.PRIMARY};
            }}
        """)
        game_row.addWidget(self.game_combo)
        overlay_layout.addLayout(game_row)
        
        overlay_layout.addSpacing(24)
        
        # Continue button
        continue_btn = QPushButton("Continue")
        continue_btn.setFixedSize(200, 52)
        continue_btn.setCursor(QCursor(Qt.PointingHandCursor))
        continue_btn.setFont(Fonts.create(16, Fonts.WEIGHT_BOLD))
        continue_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Gradients.primary_horizontal()};
                color: white;
                border-radius: 26px;
                border: none;
            }}
            QPushButton:hover {{ background: {Gradients.primary_hover()}; }}
            QPushButton:pressed {{ background: {Colors.PRIMARY_DARK}; }}
        """)
        continue_btn.clicked.connect(self._emit_selected)
        overlay_layout.addWidget(continue_btn, alignment=Qt.AlignCenter)
        
        overlay_layout.addStretch()
        
        content_stack.addWidget(overlay)
        main_layout.addWidget(content_container, 1)

    def _create_header(self) -> QWidget:
        """Create the header."""
        header_container = QWidget()
        header_container.setFixedHeight(56)
        header_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        header_container.setStyleSheet(f"background: {Colors.BG_DARK};")
        
        header_layout = QVBoxLayout(header_container)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(0)
        
        header = HeaderWidget(
            self.user_email,
            on_logout=self.on_logout,
            on_back=self._go_back,
            show_back_button=True,
            on_home_click=self.on_home_click,
            on_docs_click=self.on_docs_click
        )
        header_layout.addWidget(header)
        
        return header_container

    def _select_source(self, selected_card: SourceOptionCard, source_name: str) -> None:
        """Handle source selection."""
        for card in self.source_cards:
            card.set_selected(card == selected_card)
        self.selected_source = source_name

    def _emit_selected(self) -> None:
        """Save game, then emit the selected source."""
        game = "bgmi" if self.game_combo.currentIndex() == 1 else "freefire"
        config_path = os.path.join(os.path.expanduser("~"), ".esports_ai_config.json")
        try:
            data = {}
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            data["game"] = game
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
        self.selected.emit({"source": self.selected_source, "match": self.match_info, "game": game})

    def get_selected_source(self) -> str:
        """Get the currently selected source."""
        return self.selected_source

    def _go_back(self) -> None:
        """Navigate back."""
        self.back_requested.emit()

    def resizeEvent(self, event) -> None:
        """Handle resize."""
        super().resizeEvent(event)
        if hasattr(self, 'bg_widget'):
            content = self.layout().itemAt(1).widget()
            if content:
                self.bg_widget.setGeometry(content.rect())


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication

    dummy_match = {
        "id": "test-id",
        "matchName": "Dummy Match",
        "status": "Ongoing",
    }
    app = QApplication(sys.argv)
    w = StreamingSourcePage(dummy_match, user_email="test@email.com")
    w.show()
    sys.exit(app.exec_())

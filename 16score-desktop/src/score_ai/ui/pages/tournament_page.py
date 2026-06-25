"""
Tournament Selection Page with AI-themed UI

Features:
- AI neural network animated background
- Glassmorphism tournament cards with glow effects
- Animated hover effects
- Clean visual hierarchy matching login page
"""
from __future__ import annotations

import math
from datetime import datetime

import requests
from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QFrame,
    QScrollArea,
    QSizePolicy,
    QGraphicsDropShadowEffect,
)
from PyQt5.QtGui import (
    QCursor,
    QPixmap,
    QPainter,
    QBrush,
    QPen,
    QLinearGradient,
    QRadialGradient,
    QColor,
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QPropertyAnimation, QRect

from score_ai.ui.components.header_widget import HeaderWidget
from score_ai.ui.components.loading_spinner import LoadingSpinner
from score_ai.ui.components.empty_state import EmptyState
from score_ai.ui.components.keyboard_shortcuts import GlobalShortcuts
from score_ai.ui.components.skeleton import SkeletonCard
from score_ai.core.config_manager import config
from score_ai.core.api_service import user_api
from score_ai.core.preferences import preferences
from score_ai.utils.responsive_utils import ResponsiveUtils
from score_ai.core.theme import Colors, Fonts, Spacing, Gradients, Shadows, Styles


class AIBackgroundWidget(QWidget):
    """AI-themed animated background for tournament page"""
    
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
        random.seed(123)  # Different seed than login for variety
        self.nodes = []
        for _ in range(20):
            self.nodes.append({
                'x': random.uniform(0.02, 0.98),
                'y': random.uniform(0.05, 0.95),
                'size': random.uniform(2, 5),
                'pulse_offset': random.uniform(0, 6.28)
            })
        
        self.connections = []
        for i, n1 in enumerate(self.nodes):
            for j, n2 in enumerate(self.nodes):
                if i < j:
                    dist = ((n1['x'] - n2['x'])**2 + (n1['y'] - n2['y'])**2)**0.5
                    if dist < 0.3:
                        self.connections.append((i, j, dist))
    
    def _animate(self):
        self._pulse = (self._pulse + 0.06) % 6.28
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect()
        w, h = rect.width(), rect.height()
        
        # Dark gradient background
        gradient = QLinearGradient(0, 0, w, h)
        gradient.setColorAt(0.0, Colors.to_qcolor(Colors.BG_DARK))
        gradient.setColorAt(0.4, Colors.to_qcolor(Colors.BG))
        gradient.setColorAt(0.7, Colors.to_qcolor(Colors.BG_ALT))
        gradient.setColorAt(1.0, Colors.to_qcolor(Colors.BG_DARK))
        painter.fillRect(rect, QBrush(gradient))
        
        # Subtle grid
        painter.setPen(QPen(Colors.to_qcolor(Colors.BORDER, 12), 1))
        grid_size = 60
        for x in range(0, w, grid_size):
            painter.drawLine(x, 0, x, h)
        for y in range(0, h, grid_size):
            painter.drawLine(0, y, w, y)
        
        # Draw connections with pulse
        for i, j, dist in self.connections:
            n1, n2 = self.nodes[i], self.nodes[j]
            alpha = int(30 * (1 - dist / 0.3))
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
            glow.setColorAt(0.0, Colors.to_qcolor(Colors.PRIMARY, int(20 * pulse_factor)))
            glow.setColorAt(1.0, Colors.to_qcolor(Colors.PRIMARY, 0))
            painter.setBrush(QBrush(glow))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(int(x - size * 4), int(y - size * 4), int(size * 8), int(size * 8))
            
            # Core
            painter.setBrush(Colors.to_qcolor(Colors.PRIMARY_LIGHT, int(140 * pulse_factor)))
            painter.drawEllipse(int(x - size/2), int(y - size/2), int(size), int(size))
        
        # Accent glows at corners
        painter.setCompositionMode(QPainter.CompositionMode_Plus)
        
        glow1 = QRadialGradient(w * 0.85, h * 0.15, w * 0.35)
        glow1.setColorAt(0.0, Colors.to_qcolor(Colors.PRIMARY, 25))
        glow1.setColorAt(0.5, Colors.to_qcolor(Colors.PRIMARY, 8))
        glow1.setColorAt(1.0, Colors.to_qcolor(Colors.PRIMARY, 0))
        painter.fillRect(rect, QBrush(glow1))
        
        glow2 = QRadialGradient(w * 0.15, h * 0.85, w * 0.3)
        glow2.setColorAt(0.0, Colors.to_qcolor(Colors.PRIMARY_ALT, 20))
        glow2.setColorAt(0.5, Colors.to_qcolor(Colors.PRIMARY_ALT, 5))
        glow2.setColorAt(1.0, Colors.to_qcolor(Colors.PRIMARY_ALT, 0))
        painter.fillRect(rect, QBrush(glow2))


class GlowingTournamentCard(QFrame):
    """Tournament card with glowing glassmorphism effect"""
    
    clicked = pyqtSignal(str, str)  # tournament_id, tournament_name
    
    def __init__(self, tournament: dict, parent=None):
        super().__init__(parent)
        self.tournament = tournament
        self.league_id = tournament.get("id", "")
        self.league_name = tournament.get("name", "Unknown Tournament")
        self._glow_alpha = 0
        self._hovered = False
        
        self.setFixedHeight(140)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setAttribute(Qt.WA_Hover, True)
        
        self._setup_ui()
        
        # Glow animation
        self._glow_timer = QTimer(self)
        self._glow_timer.timeout.connect(self._update_glow)
        self._glow_timer.start(30)
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(20)
        
        # Logo
        logo_label = QLabel()
        logo_label.setFixedSize(80, 80)
        logo_path = self.tournament.get("logo")
        
        if logo_path and logo_path.strip():
            try:
                if logo_path.startswith('http'):
                    response = requests.get(logo_path, timeout=3)
                    if response.status_code == 200:
                        pixmap = QPixmap()
                        pixmap.loadFromData(response.content)
                        if not pixmap.isNull():
                            logo_label.setPixmap(pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                            logo_label.setStyleSheet("border-radius: 12px; background: transparent;")
            except Exception:
                pass
        
        if logo_label.pixmap() is None or logo_label.pixmap().isNull():
            logo_label.setText("T")
            logo_label.setAlignment(Qt.AlignCenter)
            logo_label.setStyleSheet(f"""
                background: {Gradients.primary_diagonal()};
                color: white;
                font-size: 36px;
                font-weight: bold;
                border-radius: 12px;
            """)
        
        layout.addWidget(logo_label)
        
        # Info section
        info_layout = QVBoxLayout()
        info_layout.setSpacing(6)
        info_layout.setContentsMargins(0, 0, 0, 0)
        
        # Tournament name
        name_label = QLabel(self.tournament["name"])
        name_label.setFont(Fonts.create(18, Fonts.WEIGHT_BOLD))
        name_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent;")
        info_layout.addWidget(name_label)
        
        # Date
        date_label = QLabel(self.tournament["date"])
        date_label.setFont(Fonts.create(13))
        date_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
        info_layout.addWidget(date_label)
        
        # Status badge
        status = self.tournament.get("status", "")
        if status and status.lower() != "unknown":
            status_container = QWidget()
            status_container.setStyleSheet("background: transparent;")
            status_layout = QHBoxLayout(status_container)
            status_layout.setContentsMargins(0, 4, 0, 0)
            status_layout.setAlignment(Qt.AlignLeft)
            
            status_label = QLabel(status)
            status_label.setFont(Fonts.create(11, Fonts.WEIGHT_DEMIBOLD))
            status_colors = {
                "upcoming": Colors.WARNING,
                "ongoing": Colors.SUCCESS,
                "live": Colors.SUCCESS,
                "finished": Colors.TEXT_MUTED
            }
            bg_color = status_colors.get(status.lower(), Colors.INFO)
            status_label.setStyleSheet(f"""
                background: {bg_color};
                color: white;
                padding: 4px 12px;
                border-radius: 10px;
            """)
            status_layout.addWidget(status_label)
            status_layout.addStretch()
            info_layout.addWidget(status_container)
        
        layout.addLayout(info_layout, 1)
        
        # Select button
        select_btn = QPushButton("Select")
        select_btn.setFixedSize(100, 44)
        select_btn.setCursor(QCursor(Qt.PointingHandCursor))
        select_btn.setFont(Fonts.create(13, Fonts.WEIGHT_BOLD))
        select_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Gradients.primary_horizontal()};
                color: white;
                border-radius: 22px;
                border: none;
            }}
            QPushButton:hover {{
                background: {Gradients.primary_hover()};
            }}
            QPushButton:pressed {{
                background: {Colors.PRIMARY_DARK};
            }}
        """)
        select_btn.clicked.connect(lambda: self.clicked.emit(self.league_id, self.league_name))
        layout.addWidget(select_btn, alignment=Qt.AlignVCenter)
    
    def _update_glow(self):
        target = 80 if self._hovered else 0
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
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect()
        radius = 16
        
        # Outer glow when hovered
        if self._glow_alpha > 0:
            glow_rect = rect.adjusted(-8, -8, 8, 8)
            glow = QRadialGradient(rect.center().x(), rect.center().y(), max(rect.width(), rect.height()) * 0.7)
            glow.setColorAt(0.0, Colors.to_qcolor(Colors.PRIMARY, int(self._glow_alpha * 0.4)))
            glow.setColorAt(0.5, Colors.to_qcolor(Colors.PRIMARY, int(self._glow_alpha * 0.2)))
            glow.setColorAt(1.0, Colors.to_qcolor(Colors.PRIMARY, 0))
            painter.setBrush(QBrush(glow))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(glow_rect, radius + 8, radius + 8)
        
        # Card background with glassmorphism
        bg_color = Colors.to_qcolor(Colors.CARD, 220)
        if self._hovered:
            bg_color = Colors.to_qcolor(Colors.SURFACE, 230)
        
        painter.setBrush(QBrush(bg_color))
        border_color = Colors.to_qcolor(Colors.PRIMARY, int(40 + self._glow_alpha * 0.6)) if self._hovered else Colors.to_qcolor(Colors.BORDER, 80)
        painter.setPen(QPen(border_color, 1.5))
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), radius, radius)
        
        # Top highlight
        highlight = QLinearGradient(0, 0, 0, 3)
        highlight.setColorAt(0, Colors.to_qcolor("#FFFFFF", 15))
        highlight.setColorAt(1, Colors.to_qcolor("#FFFFFF", 0))
        painter.setBrush(QBrush(highlight))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect.adjusted(2, 2, -2, -rect.height() + 20), radius - 2, radius - 2)
        
        super().paintEvent(event)


class TournamentPage(QWidget):
    """Tournament selection page with AI-themed UI."""
    
    back_requested = pyqtSignal()
    match_page_requested = pyqtSignal(str, str)  # tournament_id, tournament_name

    def __init__(
        self,
        token: str,
        user_email: str,
        user_id: str,
        on_home_click=None,
        on_logout=None,
        on_docs_click=None
    ):
        super().__init__()
        self.token = token
        self.user_email = user_email
        self.user_id = user_id
        self.on_home_click = on_home_click
        self.on_logout = on_logout
        self.on_docs_click = on_docs_click
        self.responsive = ResponsiveUtils()
        
        # Pagination state
        self.current_page = 1
        self.items_per_page = 6
        self.total_pages = 1
        self.total_count = 0
        self.filter_by = 'status'
        self.filter_value = 1
        
        self.setWindowTitle("Select Tournament - 16Score-AI")
        self._init_ui()
        self._setup_shortcuts()
        self.showMaximized()
    
    def _setup_shortcuts(self) -> None:
        """Setup keyboard shortcuts for this page."""
        self.shortcuts = GlobalShortcuts.setup_for_page(
            self,
            on_refresh=self._refresh_tournaments,
            on_back=self._go_back,
            on_home=self.on_home_click
        )
    
    def _go_back(self) -> None:
        """Navigate back to previous page."""
        self.back_requested.emit()

    def _init_ui(self) -> None:
        """Initialize the page UI layout."""
        # Main layout with no margins for full background
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header (sits on top)
        main_layout.addWidget(self._create_header())
        
        # Content area with AI background
        content_container = QWidget()
        content_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content_stack = QVBoxLayout(content_container)
        content_stack.setContentsMargins(0, 0, 0, 0)
        content_stack.setSpacing(0)
        
        # AI Background layer
        self.bg_widget = AIBackgroundWidget()
        self.bg_widget.setParent(content_container)
        self.bg_widget.lower()
        
        # Content overlay
        overlay = QWidget()
        overlay.setStyleSheet("background: transparent;")
        overlay.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        overlay_layout = QVBoxLayout(overlay)
        overlay_layout.setContentsMargins(40, 30, 40, 30)
        overlay_layout.setSpacing(24)
        
        # Hero section with title and stats
        hero_section = self._create_hero_section()
        overlay_layout.addWidget(hero_section)
        
        # Tournament cards in scrollable area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet(f"""
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
        """)
        
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        self.cards_layout = QVBoxLayout(scroll_content)
        self.cards_layout.setContentsMargins(0, 0, 16, 0)
        self.cards_layout.setSpacing(16)
        self.cards_layout.setAlignment(Qt.AlignTop)
        
        scroll_area.setWidget(scroll_content)
        overlay_layout.addWidget(scroll_area, 1)
        
        content_stack.addWidget(overlay)
        main_layout.addWidget(content_container, 1)
        
        # Load data
        self.load_tournaments()
    
    def _create_header(self) -> QWidget:
        """Create the header container with navigation."""
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
            on_back=self._go_back,
            show_back_button=True,
            on_home_click=self.on_home_click,
            on_docs_click=self.on_docs_click
        )
        header_layout.addWidget(header)
        
        return header_container
    
    def _create_hero_section(self) -> QWidget:
        """Create hero section with title, stats, and controls."""
        hero = QWidget()
        hero.setStyleSheet("background: transparent;")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(0, 0, 0, 0)
        hero_layout.setSpacing(20)
        
        # Top row: Title and controls
        top_row = QHBoxLayout()
        top_row.setSpacing(16)
        
        # Title section
        title_section = QVBoxLayout()
        title_section.setSpacing(4)
        
        title = QLabel("Select Tournament")
        title.setFont(Fonts.create(32, Fonts.WEIGHT_BOLD))
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent;")
        title_section.addWidget(title)
        
        subtitle = QLabel("Choose a tournament to manage scoring")
        subtitle.setFont(Fonts.create(14))
        subtitle.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
        title_section.addWidget(subtitle)
        
        top_row.addLayout(title_section)
        top_row.addStretch()
        
        # Controls row
        controls = QHBoxLayout()
        controls.setSpacing(12)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedHeight(40)
        refresh_btn.setCursor(QCursor(Qt.PointingHandCursor))
        refresh_btn.setFont(Fonts.create(12, Fonts.WEIGHT_DEMIBOLD))
        refresh_btn.clicked.connect(self._refresh_tournaments)
        refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.with_alpha(Colors.CARD, 0.8)};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 0 20px;
                color: {Colors.TEXT_SECONDARY};
            }}
            QPushButton:hover {{
                background: {Colors.SURFACE};
                border-color: {Colors.PRIMARY};
                color: {Colors.PRIMARY_LIGHT};
            }}
        """)
        controls.addWidget(refresh_btn)
        
        # Pagination
        self._add_pagination(controls)
        
        top_row.addLayout(controls)
        hero_layout.addLayout(top_row)
        
        # Stats row
        stats_widget = self._create_stats_row()
        hero_layout.addWidget(stats_widget)
        
        return hero
    
    def _create_stats_row(self) -> QWidget:
        """Create the stats display row."""
        stats_container = QWidget()
        stats_container.setFixedHeight(70)
        stats_container.setStyleSheet(f"""
            background: {Colors.with_alpha(Colors.CARD, 0.5)};
            border: 1px solid {Colors.with_alpha(Colors.BORDER, 0.5)};
            border-radius: 12px;
        """)
        
        stats_layout = QHBoxLayout(stats_container)
        stats_layout.setContentsMargins(32, 0, 32, 0)
        stats_layout.setSpacing(48)
        
        stats = [
            ("Active", "Tournaments", Colors.SUCCESS),
            ("Total Matches", "Scored", Colors.PRIMARY_LIGHT),
            ("99%+", "Accuracy", Colors.GOLD),
        ]
        
        for i, (value, label, color) in enumerate(stats):
            stat_widget = QWidget()
            stat_widget.setStyleSheet("background: transparent;")
            stat_layout = QHBoxLayout(stat_widget)
            stat_layout.setContentsMargins(0, 0, 0, 0)
            stat_layout.setSpacing(8)
            
            # Value
            value_label = QLabel(value)
            value_label.setFont(Fonts.create(20, Fonts.WEIGHT_BOLD))
            value_label.setStyleSheet(f"color: {color}; background: transparent;")
            stat_layout.addWidget(value_label)
            
            # Label
            desc_label = QLabel(label)
            desc_label.setFont(Fonts.create(12))
            desc_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; background: transparent;")
            stat_layout.addWidget(desc_label)
            
            stats_layout.addWidget(stat_widget)
            
            # Add separator except for last item
            if i < len(stats) - 1:
                separator = QFrame()
                separator.setFixedWidth(1)
                separator.setStyleSheet(f"background: {Colors.BORDER};")
                stats_layout.addWidget(separator)
        
        stats_layout.addStretch()
        
        return stats_container

    def _add_pagination(self, layout: QHBoxLayout) -> None:
        """Add pagination controls to the layout."""
        btn_style = f"""
            QPushButton {{
                background: {Colors.with_alpha(Colors.CARD, 0.8)};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                color: {Colors.TEXT_SECONDARY};
                font-weight: 600;
                min-width: 36px;
                min-height: 36px;
            }}
            QPushButton:hover {{
                background: {Colors.SURFACE};
                border-color: {Colors.PRIMARY};
            }}
            QPushButton:disabled {{
                background: {Colors.with_alpha(Colors.CARD, 0.4)};
                color: {Colors.TEXT_MUTED};
                border-color: transparent;
            }}
        """
        
        self.prev_btn = QPushButton("<")
        self.prev_btn.setFixedSize(36, 36)
        self.prev_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.prev_btn.setStyleSheet(btn_style)
        self.prev_btn.clicked.connect(self._go_to_previous_page)
        self.prev_btn.setEnabled(False)
        layout.addWidget(self.prev_btn)
        
        self.page_label = QLabel("1")
        self.page_label.setFixedSize(36, 36)
        self.page_label.setAlignment(Qt.AlignCenter)
        self.page_label.setFont(Fonts.create(12, Fonts.WEIGHT_BOLD))
        self.page_label.setStyleSheet(f"""
            background: {Colors.PRIMARY};
            color: white;
            border-radius: 8px;
        """)
        layout.addWidget(self.page_label)
        
        self.next_btn = QPushButton(">")
        self.next_btn.setFixedSize(36, 36)
        self.next_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.next_btn.setStyleSheet(btn_style)
        self.next_btn.clicked.connect(self._go_to_next_page)
        layout.addWidget(self.next_btn)

    def _refresh_tournaments(self) -> None:
        """Refresh the tournament list from page 1."""
        self.current_page = 1
        self.load_tournaments(self.current_page)
    
    def load_tournaments(self, page_number: int | None = None) -> None:
        """Load tournaments from API and display them."""
        if page_number is None:
            page_number = self.current_page
        
        self.current_page = page_number
        
        # Show loading state
        self._show_loading()
        
        response = user_api.get_tournaments(
            page_number=self.current_page,
            page_size=self.items_per_page,
            filter_by=self.filter_by,
            filter_value=self.filter_value
        )
        
        tournaments = self._parse_tournaments_response(response)
        self._display_tournaments(tournaments)
    
    def _show_loading(self) -> None:
        """Show skeleton loading placeholders."""
        # Clear existing cards
        while self.cards_layout.count():
            child = self.cards_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Add skeleton cards for loading effect
        for _ in range(4):
            skeleton = SkeletonCard()
            self.cards_layout.addWidget(skeleton)
        
        self.cards_layout.addStretch()
    
    def _parse_tournaments_response(self, response) -> list[dict]:
        """Parse the API response into a list of tournament dictionaries."""
        STATUS_MAP = {1: "Upcoming", 2: "Ongoing", 3: "Finished"}
        tournaments = []
        
        if not (response.success and response.data and isinstance(response.data, dict)):
            return tournaments
        
        leagues_data = response.data.get("data", {})
        
        if isinstance(leagues_data, dict):
            self.total_count = leagues_data.get("totalCount", leagues_data.get("total", 0))
            self.total_pages = leagues_data.get("totalPages", 0)
            
            if self.total_pages == 0 and self.total_count > 0:
                self.total_pages = (self.total_count + self.items_per_page - 1) // self.items_per_page
            
            leagues = leagues_data.get("leagues", leagues_data.get("items", []))
        else:
            leagues = []
        
        backend_url = config.get('api.backend_url', 'http://webapi.16score.com').rstrip('/')
        
        for item in leagues:
            if not isinstance(item, dict):
                continue
            
            status_text = STATUS_MAP.get(item.get("status"), "Unknown")
            
            try:
                start_date = datetime.fromisoformat(item.get("startDate", "")).strftime("%d %b")
                end_date = datetime.fromisoformat(item.get("endDate", "")).strftime("%d %b %y")
                date_str = f"{start_date} - {end_date}"
            except (ValueError, TypeError):
                date_str = "N/A"
            
            logo_path = item.get("logo", "")
            logo_url = f"{backend_url}{logo_path}" if logo_path else ""
            
            tournaments.append({
                "name": item.get("leagueName", "Unknown Tournament"),
                "date": date_str,
                "status": status_text,
                "logo": logo_url,
                "id": item.get("id"),
            })
        
        return tournaments

    def _display_tournaments(self, tournaments: list[dict]) -> None:
        """Display tournaments in the cards layout."""
        # Clear existing cards
        while self.cards_layout.count():
            child = self.cards_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        if not tournaments:
            empty_state = EmptyState(
                icon_type="no_tournaments",
                title="No Tournaments Found",
                message="There are no tournaments available at the moment. Check back later or try refreshing.",
                action_text="Refresh",
                action_callback=self._refresh_tournaments
            )
            self.cards_layout.addWidget(empty_state)
        else:
            for tournament in tournaments:
                card = GlowingTournamentCard(tournament)
                card.clicked.connect(self._on_tournament_selected)
                self.cards_layout.addWidget(card)
        
        self.cards_layout.addStretch()
        self._update_pagination_controls()
    
    def _on_tournament_selected(self, league_id: str, league_name: str) -> None:
        """Handle tournament selection."""
        # Save to preferences
        preferences.last_tournament_id = league_id
        preferences.last_tournament_name = league_name
        preferences.add_recent_tournament(league_id, league_name)
        
        self.match_page_requested.emit(league_id, league_name)
    
    def _go_to_previous_page(self) -> None:
        """Navigate to the previous page."""
        if self.current_page > 1:
            self.current_page -= 1
            self.load_tournaments(self.current_page)
    
    def _go_to_next_page(self) -> None:
        """Navigate to the next page."""
        if self.current_page < self.total_pages:
            self.current_page += 1
            self.load_tournaments(self.current_page)
    
    def _update_pagination_controls(self) -> None:
        """Update the pagination button states."""
        if not hasattr(self, 'prev_btn'):
            return
        
        self.page_label.setText(str(self.current_page) if self.total_pages > 0 else "0")
        self.prev_btn.setEnabled(self.current_page > 1)
        self.next_btn.setEnabled(self.current_page < self.total_pages)

    def resizeEvent(self, event) -> None:
        """Handle window resize."""
        super().resizeEvent(event)
        # Resize background to match content area
        if hasattr(self, 'bg_widget'):
            content = self.layout().itemAt(1).widget()
            if content:
                self.bg_widget.setGeometry(content.rect())

    def _go_back(self) -> None:
        """Emit back signal to navigate to previous page."""
        self.back_requested.emit()

    def _handle_logout(self) -> None:
        """Handle logout action."""
        if self.on_logout:
            self.on_logout()
        else:
            self.close()

"""
Match Selection Page with AI-themed UI

Features:
- AI neural network animated background
- Glowing match cards with status indicators
- Filter controls with modern styling
- Clean visual hierarchy
"""
from __future__ import annotations

import math
from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QFrame,
    QScrollArea,
    QSizePolicy,
    QComboBox,
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
from score_ai.ui.components.loading_spinner import LoadingSpinner
from score_ai.ui.components.empty_state import EmptyState
from score_ai.ui.components.toast import ToastManager
from score_ai.ui.components.keyboard_shortcuts import GlobalShortcuts
from score_ai.ui.components.skeleton import SkeletonCard
from score_ai.core.api_service import APIService
from score_ai.utils.responsive_utils import ResponsiveUtils
from score_ai.core.theme import Colors, Fonts, Spacing, Gradients, Shadows, Styles


class AIMatchBackground(QWidget):
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
        random.seed(789)
        self.nodes = []
        for _ in range(16):
            self.nodes.append({
                'x': random.uniform(0.03, 0.97),
                'y': random.uniform(0.05, 0.95),
                'size': random.uniform(2, 4.5),
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
        self._pulse = (self._pulse + 0.055) % 6.28
        self.update()

    def stop(self) -> None:
        self._timer.stop()

    def start(self) -> None:
        self._timer.start(50)

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
        grid_size = 55
        for x in range(0, w, grid_size):
            painter.drawLine(x, 0, x, h)
        for y in range(0, h, grid_size):
            painter.drawLine(0, y, w, y)
        
        for i, j, dist in self.connections:
            n1, n2 = self.nodes[i], self.nodes[j]
            alpha = int(25 * (1 - dist / 0.3))
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
        glow1 = QRadialGradient(w * 0.85, h * 0.15, w * 0.35)
        glow1.setColorAt(0.0, Colors.to_qcolor(Colors.PRIMARY, 20))
        glow1.setColorAt(1.0, Colors.to_qcolor(Colors.PRIMARY, 0))
        painter.fillRect(rect, QBrush(glow1))


class GlowingMatchCard(QFrame):
    """Match card with glowing hover effect"""
    
    stream_clicked = pyqtSignal(dict)
    
    def __init__(self, match: dict, parent=None):
        super().__init__(parent)
        self.match = match
        self._glow_alpha = 0
        self._hovered = False
        
        self.setMinimumWidth(340)
        self.setMaximumWidth(450)
        self.setMinimumHeight(200)
        self.setAttribute(Qt.WA_Hover, True)
        
        self._setup_ui()
        
        self._glow_timer = QTimer(self)
        self._glow_timer.timeout.connect(self._update_glow)
        self._glow_timer.start(30)
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)
        
        # Status badge
        status = self.match.get("status", "").lower()
        status_label = QLabel(status.upper())
        status_label.setFont(Fonts.create(11, Fonts.WEIGHT_BOLD))
        
        status_colors = {
            "ongoing": Colors.SUCCESS,
            "live": Colors.SUCCESS,
            "upcoming": Colors.WARNING,
            "pending": Colors.WARNING,
            "completed": Colors.TEXT_MUTED,
        }
        status_color = status_colors.get(status, Colors.TEXT_SECONDARY)
        status_label.setStyleSheet(f"""
            color: {status_color};
            background: {Colors.with_alpha(status_color, 0.15)};
            padding: 4px 12px;
            border-radius: 10px;
        """)
        status_label.setFixedWidth(status_label.sizeHint().width() + 24)
        layout.addWidget(status_label)
        
        # Match name
        name_label = QLabel(self.match.get('matchName', 'Unknown Match'))
        name_label.setFont(Fonts.create(17, Fonts.WEIGHT_BOLD))
        name_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent;")
        name_label.setWordWrap(True)
        layout.addWidget(name_label)
        
        # Group / Round info
        group_round = f"{self.match.get('leagueGroupName', '')} - {self.match.get('leagueRoundName', '')}"
        if group_round.strip() != "-":
            info_label = QLabel(group_round)
            info_label.setFont(Fonts.create(12))
            info_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
            info_label.setWordWrap(True)
            layout.addWidget(info_label)
        
        # Map name
        if self.match.get("gameMapName"):
            map_label = QLabel(f"Map: {self.match.get('gameMapName')}")
            map_label.setFont(Fonts.create(12))
            map_label.setStyleSheet(f"color: {Colors.PRIMARY_LIGHT}; background: transparent;")
            layout.addWidget(map_label)
        
        # Time
        try:
            start_time = datetime.fromisoformat(self.match.get("startDate", "").replace("Z", ""))
            formatted_time = start_time.strftime("%d %b %Y, %I:%M %p")
        except (ValueError, TypeError):
            formatted_time = self.match.get("startDate", "Time not available")
        
        time_label = QLabel(formatted_time)
        time_label.setFont(Fonts.create(11))
        time_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; background: transparent;")
        layout.addWidget(time_label)
        
        layout.addStretch()
        
        # Stream button
        stream_btn = QPushButton("Stream")
        stream_btn.setFixedHeight(42)
        stream_btn.setFixedWidth(120)
        stream_btn.setCursor(QCursor(Qt.PointingHandCursor))
        stream_btn.setFont(Fonts.create(13, Fonts.WEIGHT_BOLD))
        stream_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Gradients.primary_horizontal()};
                color: white;
                border: none;
                border-radius: 21px;
            }}
            QPushButton:hover {{ background: {Gradients.primary_hover()}; }}
            QPushButton:pressed {{ background: {Colors.PRIMARY_DARK}; }}
        """)
        stream_btn.clicked.connect(lambda: self.stream_clicked.emit(self.match))
        layout.addWidget(stream_btn, alignment=Qt.AlignCenter)
    
    def _update_glow(self):
        target = 80 if self._hovered else 0
        diff = target - self._glow_alpha
        if abs(diff) > 2:
            self._glow_alpha += diff * 0.15
            self.update()
        elif self._glow_alpha != target:
            self._glow_alpha = target
            self.update()
    
    def stop(self) -> None:
        """Stop the glow timer — call before deleteLater() to prevent segfaults."""
        self._glow_timer.stop()

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
        
        if self._glow_alpha > 0:
            glow = QRadialGradient(rect.center().x(), rect.center().y(), max(rect.width(), rect.height()) * 0.6)
            glow.setColorAt(0.0, Colors.to_qcolor(Colors.PRIMARY, int(self._glow_alpha * 0.4)))
            glow.setColorAt(0.5, Colors.to_qcolor(Colors.PRIMARY, int(self._glow_alpha * 0.15)))
            glow.setColorAt(1.0, Colors.to_qcolor(Colors.PRIMARY, 0))
            painter.setBrush(QBrush(glow))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(rect.adjusted(-10, -10, 10, 10), radius + 10, radius + 10)
        
        bg_color = Colors.to_qcolor(Colors.SURFACE, 235) if self._hovered else Colors.to_qcolor(Colors.CARD, 225)
        painter.setBrush(QBrush(bg_color))
        border_color = Colors.to_qcolor(Colors.PRIMARY, int(40 + self._glow_alpha * 0.5)) if self._hovered else Colors.to_qcolor(Colors.BORDER, 90)
        painter.setPen(QPen(border_color, 1.5))
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), radius, radius)
        
        highlight = QLinearGradient(0, 0, 0, 4)
        highlight.setColorAt(0, Colors.to_qcolor("#FFFFFF", 14))
        highlight.setColorAt(1, Colors.to_qcolor("#FFFFFF", 0))
        painter.setBrush(QBrush(highlight))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect.adjusted(2, 2, -2, -rect.height() + 20), radius - 2, radius - 2)
        
        super().paintEvent(event)


class MatchPage(QWidget):
    """Match selection page with AI-themed UI."""
    
    back_requested = pyqtSignal()
    open_streaming_source_requested = pyqtSignal(dict)

    def __init__(
        self,
        token: str,
        user_email: str,
        user_id: str,
        tournament_id: str = None,
        tournament_name: str = "",
        on_home_click=None,
        on_logout=None,
        on_docs_click=None
    ):
        super().__init__()
        self.token = token
        self.user_email = user_email
        self.user_id = user_id
        self.tournament_id = tournament_id
        self.tournament_name = tournament_name or "Tournament"
        self.on_home_click = on_home_click
        self.on_logout = on_logout
        self.on_docs_click = on_docs_click
        self.current_filter = "all"
        self.matches = []
        
        # Auto-refresh timer (30 seconds)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._auto_refresh)
        self._refresh_interval = 30000  # 30 seconds
        
        self.setWindowTitle("Select Match - 16Score-AI")
        self._init_ui()
        self._setup_shortcuts()
        self.showMaximized()
    
    def _setup_shortcuts(self) -> None:
        """Setup keyboard shortcuts for this page."""
        self.shortcuts = GlobalShortcuts.setup_for_page(
            self,
            on_refresh=self.load_matches,
            on_back=self._go_back,
            on_home=self.on_home_click
        )

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
        
        self.bg_widget = AIMatchBackground()
        self.bg_widget.setParent(content_container)
        self.bg_widget.lower()
        
        # Overlay with scroll area
        overlay = QWidget()
        overlay.setStyleSheet("background: transparent;")
        overlay.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        overlay_layout = QVBoxLayout(overlay)
        overlay_layout.setContentsMargins(40, 24, 40, 24)
        overlay_layout.setSpacing(20)
        
        # Hero section
        hero = self._create_hero_section()
        overlay_layout.addWidget(hero)
        
        # Cards scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet(Styles.scrollbar())
        
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        self.cards_layout = QHBoxLayout(scroll_content)
        self.cards_layout.setContentsMargins(0, 0, 16, 0)
        self.cards_layout.setSpacing(20)
        self.cards_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        
        # Wrap in vertical layout for proper scrolling
        scroll_inner = QWidget()
        scroll_inner.setStyleSheet("background: transparent;")
        scroll_inner_layout = QVBoxLayout(scroll_inner)
        scroll_inner_layout.setContentsMargins(0, 0, 0, 0)
        scroll_inner_layout.addWidget(scroll_content)
        scroll_inner_layout.addStretch()
        
        scroll_area.setWidget(scroll_inner)
        overlay_layout.addWidget(scroll_area, 1)
        
        content_stack.addWidget(overlay)
        main_layout.addWidget(content_container, 1)
        
        self.load_matches()

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
            on_logout=self._handle_logout,
            on_back=self._go_back,
            show_back_button=True,
            on_home_click=self.on_home_click,
            on_docs_click=self.on_docs_click
        )
        header_layout.addWidget(header)
        
        return header_container

    def _create_hero_section(self) -> QWidget:
        """Create hero section with title and controls."""
        hero = QWidget()
        hero.setStyleSheet("background: transparent;")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(0, 0, 0, 0)
        hero_layout.setSpacing(16)
        
        # Top row
        top_row = QHBoxLayout()
        
        # Title section
        title_section = QVBoxLayout()
        title_section.setSpacing(4)
        
        # Tournament name badge
        tournament_label = QLabel(self.tournament_name)
        tournament_label.setFont(Fonts.create(13, Fonts.WEIGHT_DEMIBOLD))
        tournament_label.setStyleSheet(f"color: {Colors.PRIMARY_LIGHT}; background: transparent;")
        title_section.addWidget(tournament_label)
        
        title = QLabel("Today's Matches")
        title.setFont(Fonts.create(28, Fonts.WEIGHT_BOLD))
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent;")
        title_section.addWidget(title)
        
        # Live counter
        self.live_counter = QLabel()
        self.live_counter.setFont(Fonts.create(14, Fonts.WEIGHT_DEMIBOLD))
        self.live_counter.setStyleSheet(f"color: {Colors.SUCCESS}; background: transparent;")
        self.live_counter.hide()
        title_section.addWidget(self.live_counter)
        
        top_row.addLayout(title_section)
        top_row.addStretch()
        
        # Controls
        controls = QHBoxLayout()
        controls.setSpacing(12)
        
        # Filter combo
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All Matches", "Live Now", "Upcoming", "Completed"])
        self.filter_combo.setFixedWidth(160)
        self.filter_combo.setFixedHeight(40)
        self.filter_combo.setStyleSheet(f"""
            QComboBox {{
                background: {Colors.with_alpha(Colors.CARD, 0.8)};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 0 12px;
                color: {Colors.TEXT_PRIMARY};
                font-size: 13px;
                font-family: '{Fonts.FAMILY}';
            }}
            QComboBox:hover {{
                border-color: {Colors.PRIMARY};
            }}
            QComboBox::drop-down {{
                border: none;
                padding-right: 12px;
            }}
            QComboBox QAbstractItemView {{
                background: {Colors.CARD};
                color: {Colors.TEXT_PRIMARY};
                selection-background-color: {Colors.SURFACE};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
            }}
        """)
        self.filter_combo.currentTextChanged.connect(self._on_filter_changed)
        controls.addWidget(self.filter_combo)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedHeight(40)
        refresh_btn.setCursor(QCursor(Qt.PointingHandCursor))
        refresh_btn.setFont(Fonts.create(12, Fonts.WEIGHT_DEMIBOLD))
        refresh_btn.clicked.connect(self.load_matches)
        refresh_btn.setStyleSheet(Styles.secondary_button(8))
        controls.addWidget(refresh_btn)
        
        top_row.addLayout(controls)
        hero_layout.addLayout(top_row)
        
        return hero

    def _on_filter_changed(self, text: str) -> None:
        """Handle filter change."""
        filter_map = {
            "All Matches": "all",
            "Live Now": "live",
            "Upcoming": "upcoming",
            "Completed": "completed",
        }
        self.current_filter = filter_map.get(text, "all")
        self._display_matches()

    def _filter_matches(self, matches: list) -> list:
        """Filter matches based on current filter."""
        if self.current_filter == "all":
            return matches
        
        status_map = {
            "live": ["ongoing"],
            "upcoming": ["pending", "upcoming"],
            "completed": ["completed"],
        }
        
        target_statuses = status_map.get(self.current_filter, [])
        return [m for m in matches if m.get("status", "").lower() in target_statuses]

    def _sort_matches(self, matches: list) -> list:
        """Sort matches by status priority."""
        def get_sort_key(match):
            status = match.get("status", "").lower()
            status_priority = {"ongoing": 0, "pending": 1, "upcoming": 1, "completed": 2}
            return (status_priority.get(status, 3), match.get("startDate", ""))
        
        return sorted(matches, key=get_sort_key)

    def _clear_cards(self) -> None:
        """Remove all card widgets, stopping their timers first to prevent segfaults."""
        while self.cards_layout.count():
            child = self.cards_layout.takeAt(0)
            if child.widget():
                widget = child.widget()
                # Stop skeleton shimmer timers
                if hasattr(widget, 'stop') and callable(widget.stop):
                    widget.stop()
                # Stop GlowingMatchCard glow timer
                if hasattr(widget, '_glow_timer'):
                    widget._glow_timer.stop()
                # Stop any other QTimer on the widget itself
                if hasattr(widget, '_timer'):
                    widget._timer.stop()
                widget.deleteLater()

    def _display_matches(self) -> None:
        """Display matches in the cards layout."""
        self._clear_cards()
        
        filtered = self._filter_matches(self.matches)
        
        if not filtered:
            empty_state = EmptyState(
                icon_type="no_matches",
                title="No Matches Found",
                message="There are no matches scheduled for this tournament at the moment.",
                action_text="Refresh",
                action_callback=self.load_matches
            )
            self.cards_layout.addWidget(empty_state)
        else:
            for match in filtered:
                card = GlowingMatchCard(match)
                card.stream_clicked.connect(self._on_stream_clicked)
                self.cards_layout.addWidget(card)
        
        self.cards_layout.addStretch()
        
        # Update live counter and auto-refresh
        ongoing = sum(1 for m in self.matches if m.get("status", "").lower() in ("ongoing", "live"))
        if ongoing > 0:
            self.live_counter.setText(f"{ongoing} Live Match{'es' if ongoing > 1 else ''}")
            self.live_counter.show()
            # Start auto-refresh when there are live matches
            if not self._refresh_timer.isActive():
                self._refresh_timer.start(self._refresh_interval)
        else:
            self.live_counter.hide()
            # Stop auto-refresh when no live matches
            self._refresh_timer.stop()
    
    def _parse_matches_response(self, response_data: dict) -> list:
        """
        Parse matches from API response.
        Handles different response structures:
        - response_data["data"]["matches"] (nested: success, message, data: { matches, totalCount })
        - response_data["data"]["items"] (nested structure)
        - response_data["data"] (direct list)
        - response_data["items"] (direct items)
        - response_data (direct list)
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if not response_data or not isinstance(response_data, dict):
            logger.debug("Invalid response_data: not a dict or empty")
            return []
        
        # Log the response structure for debugging
        logger.debug(f"API Response structure: {list(response_data.keys())}")
        
        matches = []
        
        # Try different response structures
        # Structure 1: response_data["data"] with "matches" or "items"
        if "data" in response_data and isinstance(response_data["data"], dict):
            data_obj = response_data["data"]
            logger.debug(f"Found 'data' key, structure: {list(data_obj.keys())}")
            
            if "matches" in data_obj and isinstance(data_obj["matches"], list):
                matches = data_obj["matches"]
                logger.debug(f"Found matches in data.matches: {len(matches)} matches")
            elif "items" in data_obj and isinstance(data_obj["items"], list):
                matches = data_obj["items"]
                logger.debug(f"Found matches in data.items: {len(matches)} matches")
            elif isinstance(data_obj, list):
                matches = data_obj
                logger.debug(f"Found matches in data (direct list): {len(matches)} matches")
        
        # Structure 2: response_data["items"]
        elif "items" in response_data and isinstance(response_data["items"], list):
            matches = response_data["items"]
            logger.debug(f"Found matches in items: {len(matches)} matches")
        
        # Structure 3: response_data is a list
        elif isinstance(response_data, list):
            matches = response_data
            logger.debug(f"Response is direct list: {len(matches)} matches")
        
        # Structure 4: response_data["data"] is a list
        elif "data" in response_data and isinstance(response_data["data"], list):
            matches = response_data["data"]
            logger.debug(f"Found matches in data (list): {len(matches)} matches")
        
        if not matches:
            logger.warning(f"No matches found in response. Response keys: {list(response_data.keys())}")
            if "data" in response_data:
                logger.warning(f"Data type: {type(response_data['data'])}, Data keys: {list(response_data['data'].keys()) if isinstance(response_data['data'], dict) else 'N/A'}")
        
        return matches if isinstance(matches, list) else []
    
    def _auto_refresh(self) -> None:
        """Auto-refresh matches (silent refresh without loading spinner)."""
        try:
            api_service = APIService()
            api_service.set_auth_token(self.token)
            
            response = api_service.get_today_league_matches(self.tournament_id)
            
            if response.success and response.data:
                # Parse matches from response - handle different response structures
                new_matches = self._parse_matches_response(response.data)
                if new_matches:
                    self.matches = self._sort_matches(new_matches)
                    self._display_matches()
        except Exception:
            pass  # Silent fail on auto-refresh

    def _on_stream_clicked(self, match: dict) -> None:
        """Handle stream button click."""
        self.open_streaming_source_requested.emit(match)

    def load_matches(self) -> None:
        """Load matches from API."""
        # Show loading state
        self._show_loading()
        
        try:
            api_service = APIService()
            api_service.set_auth_token(self.token)
            
            response = api_service.get_today_league_matches(self.tournament_id)
            
            if not response.success:
                ToastManager.error(f"Failed to load matches: {response.error_message}", self)
                self._display_matches()  # Show empty state
                return
            
            # Parse matches from response - handle different response structures
            self.matches = self._parse_matches_response(response.data) if response.data else []
            self.matches = self._sort_matches(self.matches)
            self._display_matches()
            
        except Exception as e:
            ToastManager.error(f"Error loading matches: {str(e)}", self)
            self._display_matches()  # Show empty state
    
    def _show_loading(self) -> None:
        """Show skeleton loading placeholders."""
        self._clear_cards()
        
        # Add skeleton cards for loading effect
        for _ in range(4):
            skeleton = SkeletonCard()
            self.cards_layout.addWidget(skeleton)
        
        self.cards_layout.addStretch()

    def _go_back(self) -> None:
        """Navigate back."""
        self.back_requested.emit()

    def _handle_logout(self) -> None:
        """Handle logout."""
        if self.on_logout:
            self.on_logout()
        else:
            self.close()

    def hideEvent(self, event) -> None:
        """Stop animation timers when the page is hidden to prevent use-after-free crashes."""
        super().hideEvent(event)
        if hasattr(self, 'bg_widget') and hasattr(self.bg_widget, '_timer'):
            self.bg_widget._timer.stop()
        self._refresh_timer.stop()

    def showEvent(self, event) -> None:
        """Restart background animation when the page becomes visible again."""
        super().showEvent(event)
        if hasattr(self, 'bg_widget') and hasattr(self.bg_widget, '_timer'):
            self.bg_widget._timer.start(50)

    def resizeEvent(self, event) -> None:
        """Handle resize."""
        super().resizeEvent(event)
        if hasattr(self, 'bg_widget') and self.layout() is not None:
            item = self.layout().itemAt(1)
            if item is not None:
                content = item.widget()
                if content:
                    self.bg_widget.setGeometry(content.rect())

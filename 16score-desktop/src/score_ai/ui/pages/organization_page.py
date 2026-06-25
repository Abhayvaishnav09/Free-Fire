"""Organization selection page with modern AI-themed design"""
from __future__ import annotations

from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QFrame,
    QSizePolicy,
    QSpacerItem,
    QMessageBox,
    QGraphicsDropShadowEffect,
    QScrollArea,
)
from PyQt5.QtGui import QFont, QColor, QCursor, QPainter, QBrush, QPen, QLinearGradient, QRadialGradient
from PyQt5.QtCore import Qt, pyqtSignal

from score_ai.core.config import Config
from score_ai.core.theme import Colors, Fonts, Spacing, Gradients, Shadows, Styles
from score_ai.core.preferences import preferences
from score_ai.utils.helpers import get_resource_path
from score_ai.ui.components.header_widget import HeaderWidget
from score_ai.ui.components.ai_background import AIBackground
from score_ai.ui.pages.tournament_page import TournamentPage
from score_ai.core.api_service import user_api


class OrganizationCard(QFrame):
    """Individual organization card with hover effects"""
    clicked = pyqtSignal()
    
    def __init__(self, org_name: str, role: str, is_selected: bool = False, parent=None):
        super().__init__(parent)
        self.org_name = org_name
        self.role = role
        self._is_selected = is_selected
        self._is_hovered = False
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setFixedHeight(60)
        self.init_ui()
        self.update_style()
        
    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)
        
        # Avatar circle with first letter
        self.avatar = QLabel(self.org_name[0].upper() if self.org_name else "O")
        self.avatar.setFixedSize(38, 38)
        self.avatar.setAlignment(Qt.AlignCenter)
        self.avatar.setFont(Fonts.create(16, Fonts.WEIGHT_BOLD))
        layout.addWidget(self.avatar)
        
        # Name and role
        info_layout = QVBoxLayout()
        info_layout.setSpacing(1)
        info_layout.setContentsMargins(0, 0, 0, 0)
        
        self.name_label = QLabel(self.org_name)
        self.name_label.setFont(Fonts.create(13, Fonts.WEIGHT_DEMIBOLD))
        info_layout.addWidget(self.name_label)
        
        self.role_label = QLabel(self.role)
        self.role_label.setFont(Fonts.create(10))
        info_layout.addWidget(self.role_label)
        
        layout.addLayout(info_layout)
        layout.addStretch()
        
        # Selection indicator
        self.indicator = QLabel()
        self.indicator.setFixedSize(18, 18)
        layout.addWidget(self.indicator)
        
    def update_style(self):
        if self._is_selected:
            self.setStyleSheet(f"""
                OrganizationCard {{
                    background: {Colors.with_alpha(Colors.PRIMARY, 0.15)};
                    border: 2px solid {Colors.PRIMARY};
                    border-radius: 10px;
                }}
            """)
            self.avatar.setStyleSheet(f"""
                background: {Gradients.primary_diagonal()};
                color: white;
                border-radius: 19px;
            """)
            self.name_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent;")
            self.role_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
            self.indicator.setStyleSheet(f"""
                background: {Colors.PRIMARY};
                border-radius: 9px;
                border: 2px solid white;
            """)
        elif self._is_hovered:
            self.setStyleSheet(f"""
                OrganizationCard {{
                    background: {Colors.SURFACE};
                    border: 1px solid {Colors.with_alpha(Colors.PRIMARY, 0.5)};
                    border-radius: 10px;
                }}
            """)
            self.avatar.setStyleSheet(f"""
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 {Colors.PRIMARY_LIGHT}, stop:1 {Colors.PRIMARY_ALT});
                color: white;
                border-radius: 19px;
            """)
            self.name_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent;")
            self.role_label.setStyleSheet(f"color: {Colors.TEXT_TERTIARY}; background: transparent;")
            self.indicator.setStyleSheet(f"""
                background: transparent;
                border-radius: 9px;
                border: 2px solid {Colors.TEXT_MUTED};
            """)
        else:
            self.setStyleSheet(f"""
                OrganizationCard {{
                    background: {Colors.CARD};
                    border: 1px solid {Colors.BORDER};
                    border-radius: 10px;
                }}
            """)
            self.avatar.setStyleSheet(f"""
                background: {Colors.CARD_SECONDARY};
                color: {Colors.TEXT_SECONDARY};
                border-radius: 19px;
            """)
            self.name_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
            self.role_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; background: transparent;")
            self.indicator.setStyleSheet(f"""
                background: transparent;
                border-radius: 9px;
                border: 2px solid {Colors.BORDER};
            """)
    
    def set_selected(self, selected: bool):
        self._is_selected = selected
        self.update_style()
        
    def enterEvent(self, event):
        self._is_hovered = True
        self.update_style()
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        self._is_hovered = False
        self.update_style()
        super().leaveEvent(event)
        
    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


class GlassPanel(QFrame):
    """Simple glass panel without complex painting"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            GlassPanel {{
                background: {Colors.CARD};
                border: 2px solid {Colors.PRIMARY};
                border-radius: 16px;
            }}
        """)


class Screen2Page(QWidget):
    """Organization selection page with AI-themed design"""
    open_tournament_requested = pyqtSignal()
    back_requested = pyqtSignal()
    open_home_requested = pyqtSignal()
    go_home_requested = pyqtSignal()

    def __init__(self, token, user_email, user_id, on_logout=None, on_docs_click=None):
        super().__init__()
        self.token = token
        self.user_email = user_email
        self.user_id = user_id
        self.on_logout = on_logout
        self.on_docs_click = on_docs_click
        self.setWindowTitle("16Score-AI - Select Organization")
        self.orgs = []
        self.org_cards = []
        self.selected_index = 0
        
        self.init_ui()
        self.load_organizations()
        self.showMaximized()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # AI Background (full page)
        self.ai_bg = AIBackground(node_count=18)
        self.ai_bg.setParent(self)
        self.ai_bg.lower()

        # Header
        header_container = QWidget()
        header_container.setFixedHeight(56)
        header_container.setStyleSheet(f"background: {Colors.BG_DARK};")
        header_layout = QVBoxLayout(header_container)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(0)
        
        header = HeaderWidget(
            self.user_email,
            on_logout=self.handle_logout,
            show_back_button=False,
            on_logo_click=self.handle_logo_click,
            on_home_click=None,
            on_docs_click=self.on_docs_click,
            active_page="Home"
        )
        header_layout.addWidget(header)
        main_layout.addWidget(header_container)

        # Content area - centered with proper spacing
        content_area = QWidget()
        content_area.setStyleSheet("background: transparent;")
        content_layout = QVBoxLayout(content_area)
        content_layout.setContentsMargins(40, 40, 40, 40)
        content_layout.setSpacing(0)
        main_layout.addWidget(content_area, 1)
        
        # Top spacer
        content_layout.addStretch(1)

        # Glass panel
        self.panel = GlassPanel()
        self.panel.setFixedWidth(400)
        self.panel.setMinimumHeight(220)
        self.panel.setMaximumHeight(420)
        
        panel_layout = QVBoxLayout(self.panel)
        panel_layout.setContentsMargins(24, 20, 24, 20)
        panel_layout.setSpacing(12)

        # Title section
        title_container = QWidget()
        title_container.setStyleSheet("background: transparent;")
        title_layout = QVBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(2)
        title_layout.setAlignment(Qt.AlignCenter)
        
        # Title
        title = QLabel("Select Organization")
        title.setFont(Fonts.create(18, Fonts.WEIGHT_BOLD))
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent;")
        title.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("Choose workspace to continue")
        subtitle.setFont(Fonts.create(11))
        subtitle.setStyleSheet(f"color: {Colors.TEXT_TERTIARY}; background: transparent;")
        subtitle.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(subtitle)
        
        panel_layout.addWidget(title_container)
        
        # Scrollable organization list
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 8px;
                margin: 4px 2px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 0.2);
                border-radius: 4px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(255, 255, 255, 0.3);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: transparent;
            }
        """)
        
        self.org_list_widget = QWidget()
        self.org_list_widget.setStyleSheet("background: transparent;")
        self.org_list_layout = QVBoxLayout(self.org_list_widget)
        self.org_list_layout.setContentsMargins(0, 4, 0, 4)
        self.org_list_layout.setSpacing(8)
        
        scroll_area.setWidget(self.org_list_widget)
        panel_layout.addWidget(scroll_area, 1)

        # Continue button
        self.continue_btn = QPushButton("Continue")
        self.continue_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.continue_btn.setFixedHeight(42)
        self.continue_btn.setFont(Fonts.create(13, Fonts.WEIGHT_DEMIBOLD))
        self.continue_btn.setStyleSheet(Styles.button())
        self.continue_btn.clicked.connect(self.handle_select)
        panel_layout.addWidget(self.continue_btn)

        # Center the panel
        panel_container = QHBoxLayout()
        panel_container.addStretch(1)
        panel_container.addWidget(self.panel)
        panel_container.addStretch(1)
        content_layout.addLayout(panel_container)
        
        # Bottom spacer
        content_layout.addStretch(1)
        
        # Footer - fixed at bottom
        footer = QLabel("Powered by 16Score AI")
        footer.setFont(Fonts.small())
        footer.setStyleSheet(f"color: {Colors.TEXT_DISABLED}; background: transparent;")
        footer.setAlignment(Qt.AlignCenter)
        footer.setFixedHeight(30)
        content_layout.addWidget(footer)

    def load_organizations(self):
        response = user_api.get_user_tenants()

        if response.success and response.data:
            self.orgs = response.data.get("data", [])
            self.populate_org_list()
        else:
            error_message = response.error_message or "Failed to load organizations."
            QMessageBox.warning(self, "Error", error_message)

    def populate_org_list(self):
        # Clear existing cards
        for card in self.org_cards:
            card.deleteLater()
        self.org_cards.clear()
        
        # Check for last selected organization
        last_org_id = preferences.last_organization_id
        default_selected = 0
        
        for i, org in enumerate(self.orgs):
            org_id = org.get("tenantId", "")
            org_name = org.get("tenantName", "Unknown Organization")
            role = org.get("roleInTenant", "Unknown Role")
            
            # Auto-select last used organization
            if org_id and org_id == last_org_id:
                default_selected = i
            
            card = OrganizationCard(org_name, role, is_selected=(i == default_selected))
            card.clicked.connect(lambda checked, idx=i: self.select_org(idx))
            
            self.org_cards.append(card)
            self.org_list_layout.addWidget(card)
        
        # Set the selected index
        self.selected_index = default_selected
        
        # Update selection state
        for i, card in enumerate(self.org_cards):
            card.set_selected(i == default_selected)
        
        # Add stretch at the end
        self.org_list_layout.addStretch()

    def select_org(self, index: int):
        """Handle organization selection"""
        self.selected_index = index
        for i, card in enumerate(self.org_cards):
            card.set_selected(i == index)

    def handle_select(self):
        try:
            print(f"[ORG PAGE] handle_select called, emitting open_home_requested signal")
            import sys
            sys.stdout.flush()
            
            # Save selected organization to preferences
            if self.orgs and 0 <= self.selected_index < len(self.orgs):
                selected_org = self.orgs[self.selected_index]
                org_id = selected_org.get("tenantId", "")
                org_name = selected_org.get("tenantName", "")
                if org_id:
                    preferences.last_organization_id = org_id
                    preferences.last_organization_name = org_name
                    print(f"[ORG PAGE] Saved preference: org_id={org_id}, org_name={org_name}")
            
            self.open_home_requested.emit()
            print(f"[ORG PAGE] open_home_requested signal emitted")
            sys.stdout.flush()
        except Exception as e:
            print(f"[ORG PAGE] ERROR in handle_select: {e}")
            import traceback
            traceback.print_exc()

    def handle_logout(self):
        if self.on_logout:
            self.on_logout()
        else:
            self.close()

    def go_back(self):
        self.back_requested.emit()

    def handle_logo_click(self):
        self.go_home_requested.emit()

    def handle_home_click(self):
        self.go_home_requested.emit()
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'ai_bg'):
            self.ai_bg.setGeometry(self.rect())

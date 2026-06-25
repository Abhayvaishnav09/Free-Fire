"""Professional compact header widget"""
from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QSpacerItem,
    QSizePolicy,
    QMenu,
)
from PyQt5.QtGui import QFont, QPixmap, QCursor
from PyQt5.QtCore import Qt
from score_ai.core.theme import Colors, Fonts, Spacing, Gradients, Styles
from score_ai.utils.helpers import get_resource_path
from score_ai.ui.components.connection_status import ConnectionStatus


class HeaderWidget(QWidget):
    """Compact professional header"""
    
    def __init__(
        self, user_email, on_logout=None, on_back=None, show_back_button=False, 
        on_logo_click=None, on_home_click=None, on_docs_click=None, active_page=None
    ):
        super().__init__()
        self.user_email = user_email
        self.on_logout = on_logout
        self.on_back = on_back
        self.show_back_button = show_back_button
        self.on_logo_click = on_logo_click
        self.on_home_click = on_home_click
        self.on_docs_click = on_docs_click
        self.active_page = active_page
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(24, 0, 24, 0)
        layout.setSpacing(0)

        # Left - Logo
        left_section = QHBoxLayout()
        left_section.setSpacing(12)
        left_section.setAlignment(Qt.AlignVCenter)

        if self.show_back_button and self.on_back:
            back_btn = QPushButton("<")
            back_btn.setFixedSize(32, 32)
            back_btn.setCursor(QCursor(Qt.PointingHandCursor))
            back_btn.clicked.connect(self.on_back)
            back_btn.setToolTip("Go back (Esc)")
            back_btn.setStyleSheet(f"""
                QPushButton {{
                    background: {Colors.CARD};
                    border: 1px solid {Colors.BORDER};
                    color: {Colors.TEXT_PRIMARY};
                    font-size: 16px;
                    font-weight: bold;
                    border-radius: 16px;
                }}
                QPushButton:hover {{
                    background: {Colors.SURFACE};
                    border-color: {Colors.PRIMARY};
                    color: {Colors.PRIMARY_LIGHT};
                }}
            """)
            left_section.addWidget(back_btn)

        # Logo
        logo = QLabel()
        # Try multiple paths for logo
        import os
        logo_paths = [
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "images", "16score_logo.png"),
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "images", "16score_logo.png"),
            get_resource_path("images/16score_logo.png"),
            "images/16score_logo.png"
        ]
        
        logo_loaded = False
        for path in logo_paths:
            if os.path.exists(path):
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    logo.setPixmap(pixmap.scaled(120, 36, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    logo_loaded = True
                    break
        
        if not logo_loaded:
            logo.setText("16Score-AI")
            logo.setFont(Fonts.create(14, Fonts.WEIGHT_BOLD))
            logo.setStyleSheet(f"color: {Colors.PRIMARY}; background: transparent;")
        
        if self.on_logo_click:
            logo.setCursor(QCursor(Qt.PointingHandCursor))
            logo.mousePressEvent = lambda e: self.on_logo_click()
        left_section.addWidget(logo)

        layout.addLayout(left_section)
        layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # Center - Navigation
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(8)
        nav_layout.setAlignment(Qt.AlignVCenter)

        nav_pages = [
            ("Home", "Go to home page (Ctrl+H)"),
            ("Docs & Guide", "View documentation and setup guide")
        ]

        for page, tooltip in nav_pages:
            link = QLabel(page)
            link.setFont(Fonts.create(13, Fonts.WEIGHT_DEMIBOLD))
            link.setCursor(QCursor(Qt.PointingHandCursor))
            link.setToolTip(tooltip)
            link.mousePressEvent = lambda e, p=page: self.navigate_to_page(p)
            
            if self.active_page == page:
                link.setStyleSheet(f"""
                    QLabel {{
                        color: {Colors.TEXT_PRIMARY};
                        background: {Gradients.primary_horizontal()};
                        padding: 8px 16px;
                        border-radius: 6px;
                    }}
                """)
            else:
                link.setStyleSheet(f"""
                    QLabel {{
                        color: {Colors.TEXT_SECONDARY};
                        background: transparent;
                        padding: 8px 16px;
                        border-radius: 6px;
                    }}
                    QLabel:hover {{
                        background: {Colors.CARD};
                        color: {Colors.PRIMARY_LIGHT};
                    }}
                """)
            nav_layout.addWidget(link)

        layout.addLayout(nav_layout)
        layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # Right - User
        right_section = QHBoxLayout()
        right_section.setSpacing(16)
        right_section.setAlignment(Qt.AlignVCenter)
        
        # Connection status indicator
        self.connection_status = ConnectionStatus(check_interval=30000)
        self.connection_status.setToolTip("Server connection status")
        right_section.addWidget(self.connection_status)

        user_container = QWidget()
        user_container.setCursor(QCursor(Qt.PointingHandCursor))
        user_layout = QHBoxLayout(user_container)
        user_layout.setContentsMargins(0, 0, 0, 0)
        user_layout.setSpacing(8)

        # Avatar
        first_letter = self.user_email[0].upper() if self.user_email else "U"
        avatar = QLabel(first_letter)
        avatar.setFixedSize(36, 36)
        avatar.setAlignment(Qt.AlignCenter)
        avatar.setFont(Fonts.create(14, Fonts.WEIGHT_BOLD))
        avatar.setStyleSheet(f"""
            QLabel {{
                background: {Gradients.primary_diagonal()};
                color: {Colors.TEXT_PRIMARY};
                border-radius: 18px;
            }}
        """)
        user_layout.addWidget(avatar)

        # Username
        username = self.user_email.split("@")[0] if self.user_email else "User"
        username_label = QLabel(username)
        username_label.setFont(Fonts.create(13, Fonts.WEIGHT_DEMIBOLD))
        username_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent;")
        user_layout.addWidget(username_label)

        user_container.mousePressEvent = self.show_user_menu
        user_container.setToolTip(f"Logged in as: {self.user_email}")
        right_section.addWidget(user_container)

        layout.addLayout(right_section)
        self.setLayout(layout)

        # Header style - compact
        self.setFixedHeight(56)
        self.setStyleSheet(f"""
            HeaderWidget {{
                background: {Colors.BG_DARK};
                border-bottom: 1px solid {Colors.BORDER};
            }}
        """)

    def show_user_menu(self, event):
        menu = QMenu()
        menu.setStyleSheet(Styles.menu())
        
        email_action = menu.addAction(f"Logged in as: {self.user_email}")
        email_action.setEnabled(False)
        menu.addSeparator()
        
        logout_action = menu.addAction("Logout")
        logout_action.triggered.connect(self.handle_logout)
        
        menu.exec_(self.sender().mapToGlobal(event.pos()) if self.sender() else self.mapToGlobal(event.pos()))

    def navigate_to_page(self, page):
        if page == "Home" and self.on_home_click:
            self.on_home_click()
        elif page == "Docs & Guide" and self.on_docs_click:
            self.on_docs_click()

    def handle_logout(self):
        if self.on_logout:
            self.on_logout()

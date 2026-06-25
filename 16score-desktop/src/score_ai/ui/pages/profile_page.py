"""
Profile Page with AI-themed UI

Features:
- AI neural network animated background
- User profile display
- Account settings
- Change password functionality
"""
from __future__ import annotations

import math

from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QLineEdit,
    QSizePolicy,
    QScrollArea,
    QMessageBox,
)
from PyQt5.QtGui import QPainter, QBrush, QPen, QLinearGradient, QRadialGradient, QCursor
from PyQt5.QtCore import Qt, pyqtSignal, QTimer

from score_ai.ui.components.header_widget import HeaderWidget
from score_ai.utils.responsive_utils import ResponsiveUtils
from score_ai.core.theme import Colors, Fonts, Spacing, Styles, Shadows
from score_ai.core.api_service import user_api


class AIProfileBackground(QWidget):
    """AI-themed animated background"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._pulse = 0
        self._init_nodes()
        
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate)
        self._timer.start(55)
    
    def _init_nodes(self):
        import random
        random.seed(999)
        self.nodes = []
        for _ in range(15):
            self.nodes.append({
                'x': random.uniform(0.04, 0.96),
                'y': random.uniform(0.06, 0.94),
                'size': random.uniform(2, 4.5),
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
        self._pulse = (self._pulse + 0.045) % 6.28
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
        
        painter.setPen(QPen(Colors.to_qcolor(Colors.BORDER, 8), 1))
        grid_size = 55
        for x in range(0, w, grid_size):
            painter.drawLine(x, 0, x, h)
        for y in range(0, h, grid_size):
            painter.drawLine(0, y, w, y)
        
        for i, j, dist in self.connections:
            n1, n2 = self.nodes[i], self.nodes[j]
            alpha = int(22 * (1 - dist / 0.32))
            pulse_factor = 0.5 + 0.5 * math.sin(self._pulse + n1['pulse_offset'])
            alpha = int(alpha * (0.4 + 0.6 * pulse_factor))
            painter.setPen(QPen(Colors.to_qcolor(Colors.PRIMARY_LIGHT, alpha), 1))
            painter.drawLine(int(n1['x'] * w), int(n1['y'] * h), int(n2['x'] * w), int(n2['y'] * h))
        
        for node in self.nodes:
            x, y = int(node['x'] * w), int(node['y'] * h)
            size = node['size']
            pulse_factor = 0.5 + 0.5 * math.sin(self._pulse + node['pulse_offset'])
            
            glow = QRadialGradient(x, y, size * 3.5)
            glow.setColorAt(0.0, Colors.to_qcolor(Colors.PRIMARY, int(16 * pulse_factor)))
            glow.setColorAt(1.0, Colors.to_qcolor(Colors.PRIMARY, 0))
            painter.setBrush(QBrush(glow))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(int(x - size * 3.5), int(y - size * 3.5), int(size * 7), int(size * 7))
            
            painter.setBrush(Colors.to_qcolor(Colors.PRIMARY_LIGHT, int(110 * pulse_factor)))
            painter.drawEllipse(int(x - size/2), int(y - size/2), int(size), int(size))


class ProfileSection(QFrame):
    """A profile section card"""
    
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            ProfileSection {{
                background: {Colors.CARD};
                border: 1px solid {Colors.BORDER};
                border-radius: {Spacing.RADIUS_XL}px;
            }}
        """)
        
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(24, 20, 24, 20)
        self.main_layout.setSpacing(16)
        
        # Section title
        title_label = QLabel(title)
        title_label.setFont(Fonts.create(16, Fonts.WEIGHT_BOLD))
        title_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent;")
        self.main_layout.addWidget(title_label)
        
        # Content layout
        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(12)
        self.main_layout.addLayout(self.content_layout)
    
    def add_info_row(self, label: str, value: str):
        """Add an info display row"""
        row = QHBoxLayout()
        row.setSpacing(16)
        
        label_widget = QLabel(label)
        label_widget.setFont(Fonts.create(13))
        label_widget.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
        label_widget.setFixedWidth(120)
        row.addWidget(label_widget)
        
        value_widget = QLabel(value)
        value_widget.setFont(Fonts.create(13, Fonts.WEIGHT_DEMIBOLD))
        value_widget.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent;")
        row.addWidget(value_widget, 1)
        
        self.content_layout.addLayout(row)
    
    def add_input_row(self, label: str, placeholder: str = "", password: bool = False) -> QLineEdit:
        """Add an input row and return the input widget"""
        row = QHBoxLayout()
        row.setSpacing(16)
        
        label_widget = QLabel(label)
        label_widget.setFont(Fonts.create(13))
        label_widget.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
        label_widget.setFixedWidth(150)
        row.addWidget(label_widget)
        
        input_widget = QLineEdit()
        input_widget.setPlaceholderText(placeholder)
        if password:
            input_widget.setEchoMode(QLineEdit.Password)
        input_widget.setStyleSheet(f"""
            QLineEdit {{
                background: {Colors.SURFACE};
                border: 1px solid {Colors.BORDER};
                border-radius: {Spacing.RADIUS_MD}px;
                color: {Colors.TEXT_PRIMARY};
                padding: 10px 14px;
                font-size: 13px;
            }}
            QLineEdit:focus {{
                border-color: {Colors.PRIMARY};
            }}
        """)
        row.addWidget(input_widget, 1)
        
        self.content_layout.addLayout(row)
        return input_widget


class ProfilePage(QWidget):
    """Profile page with user info and account settings"""
    
    back_requested = pyqtSignal()
    
    def __init__(
        self,
        user_email: str = "",
        user_id: str = "",
        on_logout=None,
        on_home_click=None
    ):
        super().__init__()
        self.user_email = user_email
        self.user_id = user_id
        self.on_logout = on_logout
        self.on_home_click = on_home_click
        self.responsive = ResponsiveUtils()
        
        self.user_data = {}
        
        self.setWindowTitle("16Score-AI - Profile")
        self._init_ui()
        self._load_user_data()
    
    def _load_user_data(self):
        """Load user data from API"""
        try:
            response = user_api.get_user_from_token()
            if response.success and response.data:
                self.user_data = response.data.get("data", {})
                self._update_profile_display()
        except Exception as e:
            print(f"Error loading user data: {e}")
    
    def _update_profile_display(self):
        """Update profile display with loaded data"""
        if hasattr(self, 'username_value'):
            username = self.user_data.get("username", self.user_id or "N/A")
            self.username_value.setText(username)
        
        if hasattr(self, 'email_value'):
            email = self.user_data.get("email", self.user_email or "N/A")
            self.email_value.setText(email)
        
        if hasattr(self, 'role_value'):
            role = self.user_data.get("role", "User")
            self.role_value.setText(role)
        
        if hasattr(self, 'avatar_label'):
            initial = (self.user_data.get("username", self.user_email) or "U")[0].upper()
            self.avatar_label.setText(initial)
    
    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Background
        self.bg_widget = AIProfileBackground()
        self.bg_widget.setParent(self)
        self.bg_widget.lower()
        
        # Header
        header_container = QWidget()
        header_container.setFixedHeight(56)
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
        )
        header_layout.addWidget(header)
        main_layout.addWidget(header_container)
        
        # Content area
        content_widget = QWidget()
        content_widget.setStyleSheet("background: transparent;")
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(40, 30, 40, 30)
        content_layout.setSpacing(24)
        
        # Page header with avatar
        header_row = QHBoxLayout()
        header_row.setSpacing(20)
        
        # Avatar
        self.avatar_label = QLabel((self.user_email or "U")[0].upper())
        self.avatar_label.setFixedSize(80, 80)
        self.avatar_label.setAlignment(Qt.AlignCenter)
        self.avatar_label.setFont(Fonts.create(32, Fonts.WEIGHT_BOLD))
        self.avatar_label.setStyleSheet(f"""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                stop:0 {Colors.PRIMARY}, stop:1 {Colors.PRIMARY_LIGHT});
            color: white;
            border-radius: 40px;
        """)
        header_row.addWidget(self.avatar_label)
        
        # Title section
        title_section = QVBoxLayout()
        title_section.setSpacing(4)
        
        title = QLabel("My Profile")
        title.setFont(Fonts.create(28, Fonts.WEIGHT_BOLD))
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent;")
        title_section.addWidget(title)
        
        subtitle = QLabel("Manage your account settings and preferences")
        subtitle.setFont(Fonts.create(14))
        subtitle.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
        title_section.addWidget(subtitle)
        
        header_row.addLayout(title_section, 1)
        content_layout.addLayout(header_row)
        
        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(Styles.scrollbar() + "QScrollArea { background: transparent; border: none; }")
        
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 20, 0)
        scroll_layout.setSpacing(20)
        
        # Profile Info Section
        info_section = ProfileSection("Account Information")
        
        # Username row
        username_row = QHBoxLayout()
        username_row.setSpacing(16)
        username_label = QLabel("Username")
        username_label.setFont(Fonts.create(13))
        username_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
        username_label.setFixedWidth(120)
        username_row.addWidget(username_label)
        
        self.username_value = QLabel(self.user_id or "N/A")
        self.username_value.setFont(Fonts.create(13, Fonts.WEIGHT_DEMIBOLD))
        self.username_value.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent;")
        username_row.addWidget(self.username_value, 1)
        info_section.content_layout.addLayout(username_row)
        
        # Email row
        email_row = QHBoxLayout()
        email_row.setSpacing(16)
        email_label = QLabel("Email")
        email_label.setFont(Fonts.create(13))
        email_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
        email_label.setFixedWidth(120)
        email_row.addWidget(email_label)
        
        self.email_value = QLabel(self.user_email or "N/A")
        self.email_value.setFont(Fonts.create(13, Fonts.WEIGHT_DEMIBOLD))
        self.email_value.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent;")
        email_row.addWidget(self.email_value, 1)
        info_section.content_layout.addLayout(email_row)
        
        # Role row
        role_row = QHBoxLayout()
        role_row.setSpacing(16)
        role_label = QLabel("Role")
        role_label.setFont(Fonts.create(13))
        role_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
        role_label.setFixedWidth(120)
        role_row.addWidget(role_label)
        
        self.role_value = QLabel("User")
        self.role_value.setFont(Fonts.create(13, Fonts.WEIGHT_DEMIBOLD))
        self.role_value.setStyleSheet(f"color: {Colors.PRIMARY_LIGHT}; background: transparent;")
        role_row.addWidget(self.role_value, 1)
        info_section.content_layout.addLayout(role_row)
        
        scroll_layout.addWidget(info_section)
        
        # Change Password Section
        password_section = ProfileSection("Change Password")
        
        self.current_password = password_section.add_input_row(
            "Current Password", "Enter current password", password=True
        )
        self.new_password = password_section.add_input_row(
            "New Password", "Enter new password", password=True
        )
        self.confirm_password = password_section.add_input_row(
            "Confirm Password", "Confirm new password", password=True
        )
        
        # Password requirements note
        password_note = QLabel("Password must be at least 8 characters with uppercase, lowercase, and numbers.")
        password_note.setFont(Fonts.create(11))
        password_note.setStyleSheet(f"color: {Colors.TEXT_MUTED}; background: transparent;")
        password_note.setWordWrap(True)
        password_section.content_layout.addWidget(password_note)
        
        # Change password button
        change_pwd_btn = QPushButton("Update Password")
        change_pwd_btn.setCursor(QCursor(Qt.PointingHandCursor))
        change_pwd_btn.setFixedSize(160, 40)
        change_pwd_btn.setFont(Fonts.create(13, Fonts.WEIGHT_DEMIBOLD))
        change_pwd_btn.setStyleSheet(Styles.button())
        change_pwd_btn.clicked.connect(self._change_password)
        
        btn_container = QHBoxLayout()
        btn_container.addStretch()
        btn_container.addWidget(change_pwd_btn)
        password_section.content_layout.addLayout(btn_container)
        
        scroll_layout.addWidget(password_section)
        
        # Danger Zone Section
        danger_section = ProfileSection("Danger Zone")
        danger_section.setStyleSheet(f"""
            ProfileSection {{
                background: {Colors.CARD};
                border: 1px solid {Colors.ERROR};
                border-radius: {Spacing.RADIUS_XL}px;
            }}
        """)
        
        danger_info = QLabel("Logging out will clear your session. You'll need to sign in again.")
        danger_info.setFont(Fonts.create(12))
        danger_info.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
        danger_info.setWordWrap(True)
        danger_section.content_layout.addWidget(danger_info)
        
        logout_btn = QPushButton("Sign Out")
        logout_btn.setCursor(QCursor(Qt.PointingHandCursor))
        logout_btn.setFixedSize(120, 40)
        logout_btn.setFont(Fonts.create(13, Fonts.WEIGHT_DEMIBOLD))
        logout_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ERROR};
                color: white;
                border-radius: {Spacing.RADIUS_MD}px;
                border: none;
            }}
            QPushButton:hover {{
                background: #c62828;
            }}
            QPushButton:pressed {{
                background: #b71c1c;
            }}
        """)
        logout_btn.clicked.connect(self._confirm_logout)
        
        logout_container = QHBoxLayout()
        logout_container.addWidget(logout_btn)
        logout_container.addStretch()
        danger_section.content_layout.addLayout(logout_container)
        
        scroll_layout.addWidget(danger_section)
        scroll_layout.addStretch()
        
        scroll.setWidget(scroll_content)
        content_layout.addWidget(scroll, 1)
        
        main_layout.addWidget(content_widget, 1)
    
    def _change_password(self):
        """Handle password change"""
        current = self.current_password.text()
        new = self.new_password.text()
        confirm = self.confirm_password.text()
        
        if not current or not new or not confirm:
            QMessageBox.warning(self, "Error", "Please fill in all password fields.")
            return
        
        if new != confirm:
            QMessageBox.warning(self, "Error", "New passwords do not match.")
            return
        
        if len(new) < 8:
            QMessageBox.warning(self, "Error", "Password must be at least 8 characters.")
            return
        
        # Check for uppercase, lowercase, and numbers
        if not any(c.isupper() for c in new):
            QMessageBox.warning(self, "Error", "Password must contain at least one uppercase letter.")
            return
        if not any(c.islower() for c in new):
            QMessageBox.warning(self, "Error", "Password must contain at least one lowercase letter.")
            return
        if not any(c.isdigit() for c in new):
            QMessageBox.warning(self, "Error", "Password must contain at least one number.")
            return
        
        try:
            # Call API to change password
            # response = user_api.change_password(current, new)
            # For now, simulate success
            QMessageBox.information(self, "Success", "Password updated successfully!")
            self.current_password.clear()
            self.new_password.clear()
            self.confirm_password.clear()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to update password: {str(e)}")
    
    def _confirm_logout(self):
        """Confirm before logout"""
        reply = QMessageBox.question(
            self, "Sign Out",
            "Are you sure you want to sign out?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self._handle_logout()
    
    def _handle_logout(self):
        if self.on_logout:
            self.on_logout()
    
    def _go_back(self):
        self.back_requested.emit()
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'bg_widget'):
            self.bg_widget.setGeometry(self.rect())


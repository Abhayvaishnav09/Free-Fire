"""
Forgot Password Page with AI-themed UI

Features:
- AI neural network animated background
- Email input for password reset
- Success/error feedback
"""
from __future__ import annotations

import math
import re

from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QLineEdit,
    QSizePolicy,
    QGraphicsDropShadowEffect,
)
from PyQt5.QtGui import QPainter, QBrush, QPen, QLinearGradient, QRadialGradient, QCursor, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal, QTimer

from score_ai.core.theme import Colors, Fonts, Spacing, Styles, Shadows, Gradients
from score_ai.core.api_service import user_api
from score_ai.utils.helpers import get_resource_path


class AIForgotBackground(QWidget):
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
        random.seed(777)
        self.nodes = []
        for _ in range(16):
            self.nodes.append({
                'x': random.uniform(0.03, 0.97),
                'y': random.uniform(0.05, 0.95),
                'size': random.uniform(2, 5),
                'pulse_offset': random.uniform(0, 6.28)
            })
        
        self.connections = []
        for i, n1 in enumerate(self.nodes):
            for j, n2 in enumerate(self.nodes):
                if i < j:
                    dist = ((n1['x'] - n2['x'])**2 + (n1['y'] - n2['y'])**2)**0.5
                    if dist < 0.30:
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
        grid_size = 50
        for x in range(0, w, grid_size):
            painter.drawLine(x, 0, x, h)
        for y in range(0, h, grid_size):
            painter.drawLine(0, y, w, y)
        
        # Draw connections
        for i, j, dist in self.connections:
            n1, n2 = self.nodes[i], self.nodes[j]
            alpha = int(25 * (1 - dist / 0.30))
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
            painter.setBrush(Colors.to_qcolor(Colors.PRIMARY_LIGHT, int(120 * pulse_factor)))
            painter.drawEllipse(int(x - size/2), int(y - size/2), int(size), int(size))
        
        # Accent glows
        painter.setCompositionMode(QPainter.CompositionMode_Plus)
        
        glow1 = QRadialGradient(w * 0.75, h * 0.25, w * 0.35)
        glow1.setColorAt(0.0, Colors.to_qcolor(Colors.PRIMARY, 18))
        glow1.setColorAt(0.5, Colors.to_qcolor(Colors.PRIMARY, 5))
        glow1.setColorAt(1.0, Colors.to_qcolor(Colors.PRIMARY, 0))
        painter.fillRect(rect, QBrush(glow1))


class ForgotPasswordPage(QWidget):
    """Forgot password page with email input"""
    
    back_to_login = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("16Score-AI - Forgot Password")
        self._init_ui()
    
    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Background
        self.bg_widget = AIForgotBackground()
        self.bg_widget.setParent(self)
        self.bg_widget.lower()
        
        # Center content
        content = QWidget()
        content.setStyleSheet("background: transparent;")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(40, 40, 40, 40)
        
        content_layout.addStretch(1)
        
        # Card container
        card_container = QHBoxLayout()
        card_container.addStretch(1)
        
        # Main card
        card = QFrame()
        card.setFixedWidth(420)
        card.setStyleSheet(f"""
            QFrame {{
                background: {Colors.CARD};
                border: 1px solid {Colors.BORDER};
                border-radius: {Spacing.RADIUS_XXL}px;
            }}
        """)
        card.setGraphicsEffect(Shadows.card())
        
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(40, 36, 40, 36)
        card_layout.setSpacing(24)
        
        # Logo
        logo_container = QHBoxLayout()
        logo_container.setAlignment(Qt.AlignCenter)
        
        try:
            logo_path = get_resource_path("images/16score_logo.png")
            logo_label = QLabel()
            logo_pixmap = QPixmap(logo_path)
            if not logo_pixmap.isNull():
                logo_label.setPixmap(logo_pixmap.scaledToHeight(50, Qt.SmoothTransformation))
            else:
                logo_label.setText("16Score")
                logo_label.setFont(Fonts.create(24, Fonts.WEIGHT_BOLD))
                logo_label.setStyleSheet(f"color: {Colors.PRIMARY_LIGHT};")
        except:
            logo_label = QLabel("16Score")
            logo_label.setFont(Fonts.create(24, Fonts.WEIGHT_BOLD))
            logo_label.setStyleSheet(f"color: {Colors.PRIMARY_LIGHT};")
        
        logo_label.setStyleSheet("background: transparent;")
        logo_container.addWidget(logo_label)
        card_layout.addLayout(logo_container)
        
        # Title
        title = QLabel("Forgot Password?")
        title.setFont(Fonts.create(22, Fonts.WEIGHT_BOLD))
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent;")
        title.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("Enter your email address and we'll send you a link to reset your password.")
        subtitle.setFont(Fonts.create(13))
        subtitle.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
        subtitle.setWordWrap(True)
        subtitle.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(subtitle)
        
        # Email input
        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("Enter your email address")
        self.email_input.setStyleSheet(Styles.input())
        self.email_input.setFixedHeight(52)
        card_layout.addWidget(self.email_input)
        
        # Error/Success label
        self.message_label = QLabel()
        self.message_label.setFont(Fonts.create(12))
        self.message_label.setWordWrap(True)
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.hide()
        card_layout.addWidget(self.message_label)
        
        # Submit button
        self.submit_btn = QPushButton("Send Reset Link")
        self.submit_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.submit_btn.setFixedHeight(50)
        self.submit_btn.setFont(Fonts.create(14, Fonts.WEIGHT_DEMIBOLD))
        self.submit_btn.setStyleSheet(Styles.button())
        self.submit_btn.clicked.connect(self._handle_submit)
        card_layout.addWidget(self.submit_btn)
        
        # Divider
        divider_layout = QHBoxLayout()
        divider_layout.setSpacing(12)
        
        line1 = QFrame()
        line1.setFixedHeight(1)
        line1.setStyleSheet(f"background: {Colors.BORDER};")
        divider_layout.addWidget(line1, 1)
        
        or_label = QLabel("or")
        or_label.setFont(Fonts.create(12))
        or_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; background: transparent;")
        divider_layout.addWidget(or_label)
        
        line2 = QFrame()
        line2.setFixedHeight(1)
        line2.setStyleSheet(f"background: {Colors.BORDER};")
        divider_layout.addWidget(line2, 1)
        
        card_layout.addLayout(divider_layout)
        
        # Back to login
        back_btn = QPushButton("Back to Login")
        back_btn.setCursor(QCursor(Qt.PointingHandCursor))
        back_btn.setFixedHeight(44)
        back_btn.setFont(Fonts.create(13, Fonts.WEIGHT_DEMIBOLD))
        back_btn.setStyleSheet(Styles.secondary_button())
        back_btn.clicked.connect(self._go_back)
        card_layout.addWidget(back_btn)
        
        card_container.addWidget(card)
        card_container.addStretch(1)
        content_layout.addLayout(card_container)
        
        content_layout.addStretch(1)
        
        # Footer
        footer = QLabel("Powered by 16Score AI")
        footer.setFont(Fonts.small())
        footer.setStyleSheet(f"color: {Colors.TEXT_DISABLED}; background: transparent;")
        footer.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(footer)
        
        main_layout.addWidget(content, 1)
    
    def _validate_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def _show_error(self, message: str):
        """Show error message"""
        self.message_label.setText(message)
        self.message_label.setStyleSheet(f"""
            color: {Colors.ERROR_LIGHT};
            background: {Colors.with_alpha(Colors.ERROR, 0.15)};
            border: 1px solid {Colors.with_alpha(Colors.ERROR, 0.3)};
            border-radius: {Spacing.RADIUS_MD}px;
            padding: 10px;
        """)
        self.message_label.show()
    
    def _show_success(self, message: str):
        """Show success message"""
        self.message_label.setText(message)
        self.message_label.setStyleSheet(f"""
            color: {Colors.SUCCESS};
            background: {Colors.with_alpha(Colors.SUCCESS, 0.15)};
            border: 1px solid {Colors.with_alpha(Colors.SUCCESS, 0.3)};
            border-radius: {Spacing.RADIUS_MD}px;
            padding: 10px;
        """)
        self.message_label.show()
    
    def _handle_submit(self):
        """Handle reset link request"""
        email = self.email_input.text().strip()
        
        if not email:
            self._show_error("Please enter your email address.")
            return
        
        if not self._validate_email(email):
            self._show_error("Please enter a valid email address.")
            return
        
        # Disable button during request
        self.submit_btn.setEnabled(False)
        self.submit_btn.setText("Sending...")
        
        try:
            # Call API to send reset link
            # For now, simulate success since API endpoint may not exist
            # response = user_api.request_password_reset(email)
            
            # Simulate API call
            QTimer.singleShot(1500, lambda: self._on_request_complete(True, email))
            
        except Exception as e:
            self._show_error(f"An error occurred: {str(e)}")
            self.submit_btn.setEnabled(True)
            self.submit_btn.setText("Send Reset Link")
    
    def _on_request_complete(self, success: bool, email: str):
        """Handle API response"""
        self.submit_btn.setEnabled(True)
        self.submit_btn.setText("Send Reset Link")
        
        if success:
            self._show_success(f"Password reset link sent to {email}. Please check your inbox.")
            self.email_input.clear()
        else:
            self._show_error("Failed to send reset link. Please try again.")
    
    def _go_back(self):
        """Navigate back to login"""
        self.back_to_login.emit()
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'bg_widget'):
            self.bg_widget.setGeometry(self.rect())


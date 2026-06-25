"""
Documentation & Guide Page with AI-themed UI

Features:
- AI neural network animated background
- Glassmorphism section cards
- Clean documentation layout
"""
from __future__ import annotations

import math
import sys

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QVBoxLayout,
    QFrame,
    QScrollArea,
    QSizePolicy,
)
from PyQt5.QtGui import QPainter, QBrush, QPen, QLinearGradient, QRadialGradient
from PyQt5.QtCore import Qt, pyqtSignal, QTimer

from score_ai.ui.components.header_widget import HeaderWidget
from score_ai.utils.responsive_utils import ResponsiveUtils
from score_ai.core.theme import Colors, Fonts, Spacing, Styles


class AIDocsBackground(QWidget):
    """AI-themed animated background"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._pulse = 0
        self._init_nodes()
        
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate)
        self._timer.start(60)
    
    def _init_nodes(self):
        import random
        random.seed(555)
        self.nodes = []
        for _ in range(12):
            self.nodes.append({
                'x': random.uniform(0.05, 0.95),
                'y': random.uniform(0.05, 0.95),
                'size': random.uniform(2, 4),
                'pulse_offset': random.uniform(0, 6.28)
            })
        
        self.connections = []
        for i, n1 in enumerate(self.nodes):
            for j, n2 in enumerate(self.nodes):
                if i < j:
                    dist = ((n1['x'] - n2['x'])**2 + (n1['y'] - n2['y'])**2)**0.5
                    if dist < 0.35:
                        self.connections.append((i, j, dist))
    
    def _animate(self):
        self._pulse = (self._pulse + 0.04) % 6.28
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
        grid_size = 60
        for x in range(0, w, grid_size):
            painter.drawLine(x, 0, x, h)
        for y in range(0, h, grid_size):
            painter.drawLine(0, y, w, y)
        
        for i, j, dist in self.connections:
            n1, n2 = self.nodes[i], self.nodes[j]
            alpha = int(20 * (1 - dist / 0.35))
            pulse_factor = 0.5 + 0.5 * math.sin(self._pulse + n1['pulse_offset'])
            alpha = int(alpha * (0.4 + 0.6 * pulse_factor))
            painter.setPen(QPen(Colors.to_qcolor(Colors.PRIMARY_LIGHT, alpha), 1))
            painter.drawLine(int(n1['x'] * w), int(n1['y'] * h), int(n2['x'] * w), int(n2['y'] * h))
        
        for node in self.nodes:
            x, y = int(node['x'] * w), int(node['y'] * h)
            size = node['size']
            pulse_factor = 0.5 + 0.5 * math.sin(self._pulse + node['pulse_offset'])
            
            glow = QRadialGradient(x, y, size * 4)
            glow.setColorAt(0.0, Colors.to_qcolor(Colors.PRIMARY, int(12 * pulse_factor)))
            glow.setColorAt(1.0, Colors.to_qcolor(Colors.PRIMARY, 0))
            painter.setBrush(QBrush(glow))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(int(x - size * 4), int(y - size * 4), int(size * 8), int(size * 8))
            
            painter.setBrush(Colors.to_qcolor(Colors.PRIMARY_LIGHT, int(100 * pulse_factor)))
            painter.drawEllipse(int(x - size/2), int(y - size/2), int(size), int(size))


class GlowingSectionCard(QFrame):
    """Section card with glassmorphism effect"""
    
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title_text = title
        self._setup_ui()
    
    def _setup_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(28, 22, 28, 22)
        self.main_layout.setSpacing(14)
        
        # Title
        title_label = QLabel(self.title_text)
        title_label.setFont(Fonts.create(18, Fonts.WEIGHT_BOLD))
        title_label.setStyleSheet(f"color: {Colors.PRIMARY_LIGHT}; background: transparent;")
        self.main_layout.addWidget(title_label)
    
    def add_content(self, content_html: str):
        """Add HTML content to the card."""
        content_label = QLabel(content_html)
        content_label.setFont(Fonts.create(13))
        content_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; line-height: 1.6; background: transparent;")
        content_label.setWordWrap(True)
        content_label.setTextFormat(Qt.RichText)
        self.main_layout.addWidget(content_label)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect()
        radius = 14
        
        # Background
        bg_color = Colors.to_qcolor(Colors.CARD, 200)
        painter.setBrush(QBrush(bg_color))
        painter.setPen(QPen(Colors.to_qcolor(Colors.BORDER, 80), 1))
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), radius, radius)
        
        # Top highlight
        highlight = QLinearGradient(0, 0, 0, 4)
        highlight.setColorAt(0, Colors.to_qcolor("#FFFFFF", 10))
        highlight.setColorAt(1, Colors.to_qcolor("#FFFFFF", 0))
        painter.setBrush(QBrush(highlight))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect.adjusted(2, 2, -2, -rect.height() + 18), radius - 2, radius - 2)
        
        super().paintEvent(event)


class InstructionPage(QWidget):
    """Documentation page with AI-themed UI."""
    
    back_requested = pyqtSignal()

    def __init__(
        self,
        user_email: str = "user@email.com",
        on_logout=None,
        on_home_click=None
    ):
        super().__init__()
        self.user_email = user_email
        self.on_logout = on_logout
        self.on_home_click = on_home_click
        self.responsive = ResponsiveUtils()
        
        self.setWindowTitle("Documentation & Guide - 16Score-AI")
        self._init_ui()
        self.showMaximized()

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
        
        self.bg_widget = AIDocsBackground()
        self.bg_widget.setParent(content_container)
        self.bg_widget.lower()
        
        # Overlay
        overlay = QWidget()
        overlay.setStyleSheet("background: transparent;")
        overlay.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        overlay_layout = QVBoxLayout(overlay)
        overlay_layout.setContentsMargins(40, 30, 40, 30)
        overlay_layout.setSpacing(20)
        
        # Title
        title = QLabel("Documentation & Guide")
        title.setFont(Fonts.create(32, Fonts.WEIGHT_BOLD))
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent;")
        title.setAlignment(Qt.AlignCenter)
        overlay_layout.addWidget(title)
        
        # Scroll area for content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet(Styles.scrollbar())
        
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        self.sections_layout = QVBoxLayout(scroll_content)
        self.sections_layout.setContentsMargins(0, 0, 16, 0)
        self.sections_layout.setSpacing(20)
        
        # Add sections
        self._create_welcome_section()
        self._create_overview_section()
        self._create_tournament_section()
        self._create_match_section()
        self._create_camera_section()
        self._create_streaming_section()
        self._create_troubleshooting_section()
        
        self.sections_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        overlay_layout.addWidget(scroll_area, 1)
        
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
            on_home_click=self.on_home_click
        )
        header_layout.addWidget(header)
        
        return header_container

    def _create_section(self, title: str, content: str) -> None:
        """Helper to create a section card."""
        card = GlowingSectionCard(title)
        card.add_content(content)
        self.sections_layout.addWidget(card)

    def _create_welcome_section(self):
        content = """
        <p><b>16Score AI</b> is an advanced esports analytics platform designed to automatically 
        capture and analyze kill feeds from PUBG Mobile matches in real-time.</p>
        
        <p>This application helps tournament organizers, teams, and analysts track match statistics, 
        player performance, and game events without manual intervention.</p>
        
        <p><b>Key Features:</b></p>
        <ul>
            <li>Automatic kill feed detection and analysis</li>
            <li>Real-time match statistics</li>
            <li>Tournament management and organization</li>
            <li>Camera setup and streaming integration</li>
            <li>Performance analytics and reporting</li>
        </ul>
        """
        self._create_section("Welcome to 16Score AI", content)

    def _create_overview_section(self):
        content = """
        <p><b>How 16Score AI Works:</b></p>
        <p><b>1.</b> Tournament Selection - Choose from available tournaments</p>
        <p><b>2.</b> Match Selection - Browse and select specific matches</p>
        <p><b>3.</b> Camera Setup - Configure your camera or streaming source</p>
        <p><b>4.</b> Live Analysis - Start real-time kill feed detection</p>
        <p><b>5.</b> Results - View live statistics and performance data</p>
        
        <p><b>Supported Features:</b></p>
        <ul>
            <li>Automatic weapon and player detection</li>
            <li>Real-time data processing</li>
            <li>Cloud-based data storage</li>
            <li>Comprehensive analytics dashboard</li>
        </ul>
        """
        self._create_section("Application Overview", content)

    def _create_tournament_section(self):
        content = """
        <p><b>Selecting Tournaments:</b></p>
        <p><b>Step 1:</b> Navigate to the Tournament page</p>
        <p><b>Step 2:</b> Browse available tournaments in your organization</p>
        <p><b>Step 3:</b> Click "Select" on your desired tournament</p>
        
        <p><b>Tournament Information Displayed:</b></p>
        <ul>
            <li>Tournament dates and duration</li>
            <li>Tournament status (Upcoming/Ongoing/Finished)</li>
            <li>Match schedules and brackets</li>
        </ul>
        
        <p><b>Tips:</b> Use the refresh button to update the tournament list. 
        Check tournament status before selecting.</p>
        """
        self._create_section("Tournament Management", content)

    def _create_match_section(self):
        content = """
        <p><b>Selecting Matches:</b></p>
        <p><b>Step 1:</b> After selecting a tournament, you'll see available matches</p>
        <p><b>Step 2:</b> Use the filter dropdown to view specific match types:</p>
        <ul>
            <li><b>All Matches:</b> View all available matches</li>
            <li><b>Live Now:</b> Currently ongoing matches</li>
            <li><b>Upcoming:</b> Scheduled future matches</li>
            <li><b>Completed:</b> Finished matches</li>
        </ul>
        <p><b>Step 3:</b> Click "Stream" to proceed to camera setup</p>
        """
        self._create_section("Match Selection", content)

    def _create_camera_section(self):
        content = """
        <p><b>Camera Configuration:</b></p>
        <p><b>Step 1:</b> Enter your camera index (0-20)</p>
        <ul>
            <li><b>0:</b> Primary/built-in webcam</li>
            <li><b>1-20:</b> External cameras or capture devices</li>
        </ul>
        <p><b>Step 2:</b> Use arrow buttons to cycle through camera options</p>
        <p><b>Step 3:</b> Click "Test Camera" to preview your feed</p>
        <p><b>Step 4:</b> Click "Set Camera Index" to save selection</p>
        
        <p><b>Supported Sources:</b> Desktop webcams, OBS virtual cameras, HDMI capture devices</p>
        """
        self._create_section("Camera Setup", content)

    def _create_streaming_section(self):
        content = """
        <p><b>Starting Live Analysis:</b></p>
        <p><b>Step 1:</b> Ensure your camera is properly configured</p>
        <p><b>Step 2:</b> Click "Start Match" to begin live analysis</p>
        <p><b>Step 3:</b> The system will automatically detect and analyze kill feeds</p>
        <p><b>Step 4:</b> Click "Stop Match" when the match ends</p>
        
        <p><b>Best Practices:</b></p>
        <ul>
            <li>Ensure stable camera positioning</li>
            <li>Maintain good lighting conditions</li>
            <li>Check internet connection stability</li>
            <li>Keep the application running throughout the match</li>
        </ul>
        """
        self._create_section("Live Streaming & Analysis", content)

    def _create_troubleshooting_section(self):
        content = """
        <p><b>Camera Not Working:</b></p>
        <ul>
            <li>Check camera permissions in system settings</li>
            <li>Try different camera indices (0, 1, 2, etc.)</li>
            <li>Ensure no other application is using the camera</li>
        </ul>
        
        <p><b>No Matches Displayed:</b></p>
        <ul>
            <li>Check your internet connection</li>
            <li>Verify your account permissions</li>
            <li>Use the refresh button to reload data</li>
        </ul>
        
        <p><b>Getting Help:</b> Contact support@16score.com</p>
        """
        self._create_section("Troubleshooting", content)

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
    app = QApplication(sys.argv)
    window = InstructionPage(user_email="test@example.com")
    window.show()
    sys.exit(app.exec_())

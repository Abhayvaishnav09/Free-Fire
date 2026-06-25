"""
About Page with AI-themed UI

Features:
- AI neural network animated background
- App version information
- Credits and license
- System information
"""
from __future__ import annotations

import math
import sys
import platform

from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QSizePolicy,
    QScrollArea,
)
from PyQt5.QtGui import QPainter, QBrush, QPen, QLinearGradient, QRadialGradient, QCursor, QPixmap, QDesktopServices
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QUrl

from score_ai.ui.components.header_widget import HeaderWidget
from score_ai.utils.responsive_utils import ResponsiveUtils
from score_ai.utils.helpers import get_resource_path
from score_ai.core.theme import Colors, Fonts, Spacing, Styles, Shadows, Gradients
from score_ai.__version__ import (
    __version__,
    __app_name__,
    __author__,
    __copyright__,
    __license__,
    get_full_version,
)


class AIAboutBackground(QWidget):
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
        random.seed(666)
        self.nodes = []
        for _ in range(18):
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
                    if dist < 0.28:
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
        gradient.setColorAt(0.35, Colors.to_qcolor(Colors.BG))
        gradient.setColorAt(0.65, Colors.to_qcolor(Colors.BG_ALT))
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
            alpha = int(25 * (1 - dist / 0.28))
            pulse_factor = 0.5 + 0.5 * math.sin(self._pulse + n1['pulse_offset'])
            alpha = int(alpha * (0.4 + 0.6 * pulse_factor))
            painter.setPen(QPen(Colors.to_qcolor(Colors.PRIMARY_LIGHT, alpha), 1))
            painter.drawLine(int(n1['x'] * w), int(n1['y'] * h), int(n2['x'] * w), int(n2['y'] * h))
        
        for node in self.nodes:
            x, y = int(node['x'] * w), int(node['y'] * h)
            size = node['size']
            pulse_factor = 0.5 + 0.5 * math.sin(self._pulse + node['pulse_offset'])
            
            glow = QRadialGradient(x, y, size * 4)
            glow.setColorAt(0.0, Colors.to_qcolor(Colors.PRIMARY, int(18 * pulse_factor)))
            glow.setColorAt(1.0, Colors.to_qcolor(Colors.PRIMARY, 0))
            painter.setBrush(QBrush(glow))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(int(x - size * 4), int(y - size * 4), int(size * 8), int(size * 8))
            
            painter.setBrush(Colors.to_qcolor(Colors.PRIMARY_LIGHT, int(120 * pulse_factor)))
            painter.drawEllipse(int(x - size/2), int(y - size/2), int(size), int(size))
        
        # Accent glow
        painter.setCompositionMode(QPainter.CompositionMode_Plus)
        glow1 = QRadialGradient(w * 0.5, h * 0.3, w * 0.4)
        glow1.setColorAt(0.0, Colors.to_qcolor(Colors.PRIMARY, 15))
        glow1.setColorAt(0.5, Colors.to_qcolor(Colors.PRIMARY, 4))
        glow1.setColorAt(1.0, Colors.to_qcolor(Colors.PRIMARY, 0))
        painter.fillRect(rect, QBrush(glow1))


class InfoCard(QFrame):
    """Info display card"""
    
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            InfoCard {{
                background: {Colors.CARD};
                border: 1px solid {Colors.BORDER};
                border-radius: {Spacing.RADIUS_XL}px;
            }}
        """)
        
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(24, 20, 24, 20)
        self.main_layout.setSpacing(12)
        
        if title:
            title_label = QLabel(title)
            title_label.setFont(Fonts.create(14, Fonts.WEIGHT_BOLD))
            title_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent;")
            self.main_layout.addWidget(title_label)
    
    def add_row(self, label: str, value: str, value_color: str = None):
        """Add an info row"""
        row = QHBoxLayout()
        row.setSpacing(12)
        
        label_widget = QLabel(label)
        label_widget.setFont(Fonts.create(12))
        label_widget.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
        row.addWidget(label_widget)
        
        row.addStretch()
        
        value_widget = QLabel(value)
        value_widget.setFont(Fonts.create(12, Fonts.WEIGHT_DEMIBOLD))
        color = value_color or Colors.TEXT_PRIMARY
        value_widget.setStyleSheet(f"color: {color}; background: transparent;")
        row.addWidget(value_widget)
        
        self.main_layout.addLayout(row)


class AboutPage(QWidget):
    """About page with app info, version, and credits"""
    
    back_requested = pyqtSignal()
    
    def __init__(self, user_email: str = "", on_logout=None, on_home_click=None):
        super().__init__()
        self.user_email = user_email
        self.on_logout = on_logout
        self.on_home_click = on_home_click
        self.responsive = ResponsiveUtils()
        
        self.setWindowTitle("16Score-AI - About")
        self._init_ui()
    
    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Background
        self.bg_widget = AIAboutBackground()
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
        content_layout.setSpacing(0)
        
        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(Styles.scrollbar() + "QScrollArea { background: transparent; border: none; }")
        
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 20, 0)
        scroll_layout.setSpacing(24)
        scroll_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        
        # Logo and title section
        header_section = QWidget()
        header_section.setStyleSheet("background: transparent;")
        header_section_layout = QVBoxLayout(header_section)
        header_section_layout.setSpacing(16)
        header_section_layout.setAlignment(Qt.AlignCenter)
        
        # Logo
        try:
            logo_path = get_resource_path("images/16score_logo.png")
            logo_label = QLabel()
            logo_pixmap = QPixmap(logo_path)
            if not logo_pixmap.isNull():
                logo_label.setPixmap(logo_pixmap.scaledToHeight(70, Qt.SmoothTransformation))
            else:
                raise Exception("Logo not found")
        except:
            logo_label = QLabel("16Score")
            logo_label.setFont(Fonts.create(36, Fonts.WEIGHT_BOLD))
            logo_label.setStyleSheet(f"color: {Colors.PRIMARY_LIGHT};")
        
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setStyleSheet("background: transparent;")
        header_section_layout.addWidget(logo_label)
        
        # App name and version
        app_name = QLabel(__app_name__)
        app_name.setFont(Fonts.create(32, Fonts.WEIGHT_BOLD))
        app_name.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent;")
        app_name.setAlignment(Qt.AlignCenter)
        header_section_layout.addWidget(app_name)
        
        version_label = QLabel(f"Version {__version__}")
        version_label.setFont(Fonts.create(16))
        version_label.setStyleSheet(f"color: {Colors.PRIMARY_LIGHT}; background: transparent;")
        version_label.setAlignment(Qt.AlignCenter)
        header_section_layout.addWidget(version_label)
        
        tagline = QLabel("AI-Powered Tournament Scoring Platform")
        tagline.setFont(Fonts.create(14))
        tagline.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
        tagline.setAlignment(Qt.AlignCenter)
        header_section_layout.addWidget(tagline)
        
        scroll_layout.addWidget(header_section)
        
        # Cards container
        cards_container = QHBoxLayout()
        cards_container.setSpacing(20)
        
        # Left column
        left_column = QVBoxLayout()
        left_column.setSpacing(20)
        
        # App Info Card
        app_info_card = InfoCard("Application")
        app_info_card.add_row("Version", __version__, Colors.PRIMARY_LIGHT)
        app_info_card.add_row("Build", "Production")
        app_info_card.add_row("License", __license__)
        left_column.addWidget(app_info_card)
        
        # System Info Card
        system_card = InfoCard("System Information")
        system_card.add_row("Python", platform.python_version())
        system_card.add_row("OS", f"{platform.system()} {platform.release()}")
        system_card.add_row("Architecture", platform.machine())
        try:
            from PyQt5.QtCore import QT_VERSION_STR
            system_card.add_row("Qt Version", QT_VERSION_STR)
        except:
            pass
        left_column.addWidget(system_card)
        
        cards_container.addLayout(left_column, 1)
        
        # Right column
        right_column = QVBoxLayout()
        right_column.setSpacing(20)
        
        # Credits Card
        credits_card = InfoCard("Credits")
        credits_card.add_row("Developer", __author__)
        credits_card.add_row("Copyright", __copyright__)
        
        # Add tech stack
        tech_label = QLabel("Built with PyQt5, OpenCV, gRPC, and YOLO")
        tech_label.setFont(Fonts.create(11))
        tech_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; background: transparent;")
        tech_label.setWordWrap(True)
        credits_card.main_layout.addWidget(tech_label)
        
        right_column.addWidget(credits_card)
        
        # Features Card
        features_card = InfoCard("Key Features")
        features = [
            "✓ Real-time Kill Feed Capture",
            "✓ AI-Powered Detection (YOLO)",
            "✓ OCR Player Name Extraction",
            "✓ SIFT Weapon Detection",
            "✓ Cloud Sync & Analytics",
            "✓ Multi-Organization Support",
        ]
        for feature in features:
            feature_label = QLabel(feature)
            feature_label.setFont(Fonts.create(12))
            feature_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
            features_card.main_layout.addWidget(feature_label)
        
        right_column.addWidget(features_card)
        
        cards_container.addLayout(right_column, 1)
        
        scroll_layout.addLayout(cards_container)
        
        # Links section
        links_section = QWidget()
        links_section.setStyleSheet("background: transparent;")
        links_layout = QHBoxLayout(links_section)
        links_layout.setSpacing(16)
        links_layout.setAlignment(Qt.AlignCenter)
        
        # Website button
        website_btn = QPushButton("Visit Website")
        website_btn.setCursor(QCursor(Qt.PointingHandCursor))
        website_btn.setFixedSize(140, 40)
        website_btn.setFont(Fonts.create(12, Fonts.WEIGHT_DEMIBOLD))
        website_btn.setStyleSheet(Styles.button())
        website_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://16score.com")))
        links_layout.addWidget(website_btn)
        
        # Support button
        support_btn = QPushButton("Get Support")
        support_btn.setCursor(QCursor(Qt.PointingHandCursor))
        support_btn.setFixedSize(140, 40)
        support_btn.setFont(Fonts.create(12, Fonts.WEIGHT_DEMIBOLD))
        support_btn.setStyleSheet(Styles.secondary_button())
        support_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("mailto:support@16score.com")))
        links_layout.addWidget(support_btn)
        
        scroll_layout.addWidget(links_section)
        
        # Footer
        footer = QLabel(f"{__copyright__}. All rights reserved.")
        footer.setFont(Fonts.small())
        footer.setStyleSheet(f"color: {Colors.TEXT_DISABLED}; background: transparent;")
        footer.setAlignment(Qt.AlignCenter)
        scroll_layout.addWidget(footer)
        
        scroll_layout.addStretch()
        
        scroll.setWidget(scroll_content)
        content_layout.addWidget(scroll, 1)
        
        main_layout.addWidget(content_widget, 1)
    
    def _handle_logout(self):
        if self.on_logout:
            self.on_logout()
    
    def _go_back(self):
        self.back_requested.emit()
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'bg_widget'):
            self.bg_widget.setGeometry(self.rect())


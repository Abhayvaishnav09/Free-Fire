"""
Settings Page with AI-themed UI

Features:
- AI neural network animated background
- Detection settings configuration
- Camera defaults
- App preferences
"""
from __future__ import annotations

import json
import math
import os

from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QScrollArea,
    QSizePolicy,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QComboBox,
    QMessageBox,
)
from PyQt5.QtGui import QPainter, QBrush, QPen, QLinearGradient, QRadialGradient, QCursor
from PyQt5.QtCore import Qt, pyqtSignal, QTimer

from score_ai.ui.components.header_widget import HeaderWidget
from score_ai.utils.responsive_utils import ResponsiveUtils
from score_ai.core.theme import Colors, Fonts, Spacing, Styles, Shadows


class AISettingsBackground(QWidget):
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
        random.seed(888)
        self.nodes = []
        for _ in range(14):
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
            
            glow = QRadialGradient(x, y, size * 3)
            glow.setColorAt(0.0, Colors.to_qcolor(Colors.PRIMARY, int(15 * pulse_factor)))
            glow.setColorAt(1.0, Colors.to_qcolor(Colors.PRIMARY, 0))
            painter.setBrush(QBrush(glow))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(int(x - size * 3), int(y - size * 3), int(size * 6), int(size * 6))
            
            painter.setBrush(Colors.to_qcolor(Colors.PRIMARY_LIGHT, int(100 * pulse_factor)))
            painter.drawEllipse(int(x - size/2), int(y - size/2), int(size), int(size))


class SettingsSection(QFrame):
    """A settings section card"""
    
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            SettingsSection {{
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
    
    def add_setting(self, label: str, widget: QWidget, description: str = None):
        """Add a setting row with label and widget"""
        row = QHBoxLayout()
        row.setSpacing(16)
        
        # Label container
        label_container = QVBoxLayout()
        label_container.setSpacing(2)
        
        label_widget = QLabel(label)
        label_widget.setFont(Fonts.create(13, Fonts.WEIGHT_DEMIBOLD))
        label_widget.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent;")
        label_container.addWidget(label_widget)
        
        if description:
            desc_widget = QLabel(description)
            desc_widget.setFont(Fonts.create(11))
            desc_widget.setStyleSheet(f"color: {Colors.TEXT_MUTED}; background: transparent;")
            desc_widget.setWordWrap(True)
            label_container.addWidget(desc_widget)
        
        row.addLayout(label_container, 1)
        row.addWidget(widget)
        
        self.content_layout.addLayout(row)


class SettingsPage(QWidget):
    """Settings page with detection config, camera defaults, and preferences"""
    
    back_requested = pyqtSignal()
    
    def __init__(self, user_email: str = "", on_logout=None, on_home_click=None):
        super().__init__()
        self.user_email = user_email
        self.on_logout = on_logout
        self.on_home_click = on_home_click
        self.responsive = ResponsiveUtils()
        
        self.config_path = os.path.join(os.path.expanduser("~"), ".esports_ai_config.json")
        self.settings = self._load_settings()
        
        self.setWindowTitle("16Score-AI - Settings")
        self._init_ui()
    
    def _load_settings(self) -> dict:
        """Load settings from config file"""
        default_settings = {
            "camera_index": 0,
            "frame_capture_interval": 0.15,
            "confidence_threshold": 0.7,
            "auto_start_preview": False,
            "show_detection_overlay": True,
            "game": "freefire",
            "resolution": "1920x1080",
            "theme": "dark",
            "notifications_enabled": True,
            "auto_update": True,
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    saved = json.load(f)
                    default_settings.update(saved)
            except Exception as e:
                print(f"Error loading settings: {e}")
        
        return default_settings
    
    def _save_settings(self):
        """Save settings to config file"""
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.settings, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False
    
    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Background
        self.bg_widget = AISettingsBackground()
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
        
        # Page title
        title = QLabel("Settings")
        title.setFont(Fonts.create(28, Fonts.WEIGHT_BOLD))
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent;")
        content_layout.addWidget(title)
        
        subtitle = QLabel("Configure detection, camera, and app preferences")
        subtitle.setFont(Fonts.create(14))
        subtitle.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
        content_layout.addWidget(subtitle)
        
        # Scrollable settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(Styles.scrollbar() + "QScrollArea { background: transparent; border: none; }")
        
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 20, 0)
        scroll_layout.setSpacing(20)
        
        # Detection Settings Section
        detection_section = SettingsSection("Detection Settings")
        
        # Confidence threshold
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.1, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(self.settings.get("confidence_threshold", 0.7))
        self.confidence_spin.setFixedWidth(100)
        self._style_spinbox(self.confidence_spin)
        detection_section.add_setting(
            "Confidence Threshold",
            self.confidence_spin,
            "Minimum confidence level for detections (0.1 - 1.0)"
        )
        
        # Frame capture interval
        self.frame_interval_spin = QDoubleSpinBox()
        self.frame_interval_spin.setRange(0.1, 5.0)
        self.frame_interval_spin.setSingleStep(0.1)
        self.frame_interval_spin.setValue(self.settings.get("frame_capture_interval", 0.15))
        self.frame_interval_spin.setFixedWidth(100)
        self.frame_interval_spin.setSuffix(" sec")
        self._style_spinbox(self.frame_interval_spin)
        detection_section.add_setting(
            "Frame Capture Interval",
            self.frame_interval_spin,
            "Seconds between samples (lower = faster killfeed capture; ~0.1–0.2 for live matches)"
        )
        
        # Game selection
        self.game_combo = QComboBox()
        self.game_combo.addItems(["Free Fire", "BGMI"])
        current_game = self.settings.get("game", "freefire")
        self.game_combo.setCurrentIndex(1 if current_game == "bgmi" else 0)
        self.game_combo.setFixedWidth(140)
        self._style_combobox(self.game_combo)
        detection_section.add_setting(
            "Game",
            self.game_combo,
            "Game for killfeed detection (BGMI or Free Fire)"
        )
        
        # Show detection overlay
        self.overlay_check = QCheckBox()
        self.overlay_check.setChecked(self.settings.get("show_detection_overlay", True))
        self._style_checkbox(self.overlay_check)
        detection_section.add_setting(
            "Show Detection Overlay",
            self.overlay_check,
            "Display detection boxes on preview"
        )
        
        scroll_layout.addWidget(detection_section)
        
        # Camera Settings Section
        camera_section = SettingsSection("Camera Settings")
        
        # Default camera index
        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 20)
        self.camera_spin.setValue(self.settings.get("camera_index", 0))
        self.camera_spin.setFixedWidth(100)
        self._style_spinbox(self.camera_spin)
        camera_section.add_setting(
            "Default Camera Index",
            self.camera_spin,
            "Camera device to use by default (0-20)"
        )
        
        # Resolution
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["1920x1080", "1280x720", "640x480"])
        current_res = self.settings.get("resolution", "1920x1080")
        index = self.resolution_combo.findText(current_res)
        if index >= 0:
            self.resolution_combo.setCurrentIndex(index)
        self.resolution_combo.setFixedWidth(140)
        self._style_combobox(self.resolution_combo)
        camera_section.add_setting(
            "Default Resolution",
            self.resolution_combo,
            "Preferred capture resolution"
        )
        
        # Auto start preview
        self.auto_preview_check = QCheckBox()
        self.auto_preview_check.setChecked(self.settings.get("auto_start_preview", False))
        self._style_checkbox(self.auto_preview_check)
        camera_section.add_setting(
            "Auto Start Preview",
            self.auto_preview_check,
            "Automatically start camera preview on page load"
        )
        
        scroll_layout.addWidget(camera_section)
        
        # App Preferences Section
        prefs_section = SettingsSection("App Preferences")
        
        # Notifications
        self.notifications_check = QCheckBox()
        self.notifications_check.setChecked(self.settings.get("notifications_enabled", True))
        self._style_checkbox(self.notifications_check)
        prefs_section.add_setting(
            "Enable Notifications",
            self.notifications_check,
            "Show desktop notifications for events"
        )
        
        # Auto update
        self.auto_update_check = QCheckBox()
        self.auto_update_check.setChecked(self.settings.get("auto_update", True))
        self._style_checkbox(self.auto_update_check)
        prefs_section.add_setting(
            "Auto Update",
            self.auto_update_check,
            "Automatically check for and install updates"
        )
        
        scroll_layout.addWidget(prefs_section)
        scroll_layout.addStretch()
        
        scroll.setWidget(scroll_content)
        content_layout.addWidget(scroll, 1)
        
        # Action buttons
        button_row = QHBoxLayout()
        button_row.setSpacing(16)
        button_row.addStretch()
        
        # Reset button
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.setCursor(QCursor(Qt.PointingHandCursor))
        reset_btn.setFixedSize(160, 44)
        reset_btn.setFont(Fonts.create(13, Fonts.WEIGHT_DEMIBOLD))
        reset_btn.setStyleSheet(Styles.secondary_button())
        reset_btn.clicked.connect(self._reset_settings)
        button_row.addWidget(reset_btn)
        
        # Save button
        save_btn = QPushButton("Save Settings")
        save_btn.setCursor(QCursor(Qt.PointingHandCursor))
        save_btn.setFixedSize(160, 44)
        save_btn.setFont(Fonts.create(13, Fonts.WEIGHT_DEMIBOLD))
        save_btn.setStyleSheet(Styles.button())
        save_btn.clicked.connect(self._save_all_settings)
        button_row.addWidget(save_btn)
        
        content_layout.addLayout(button_row)
        
        main_layout.addWidget(content_widget, 1)
    
    def _style_spinbox(self, spinbox):
        spinbox.setStyleSheet(f"""
            QSpinBox, QDoubleSpinBox {{
                background: {Colors.SURFACE};
                border: 1px solid {Colors.BORDER};
                border-radius: {Spacing.RADIUS_MD}px;
                color: {Colors.TEXT_PRIMARY};
                padding: 8px 12px;
                font-size: 13px;
            }}
            QSpinBox:focus, QDoubleSpinBox:focus {{
                border-color: {Colors.PRIMARY};
            }}
            QSpinBox::up-button, QDoubleSpinBox::up-button,
            QSpinBox::down-button, QDoubleSpinBox::down-button {{
                background: {Colors.CARD};
                border: none;
                width: 20px;
            }}
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 6px solid {Colors.TEXT_SECONDARY};
            }}
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid {Colors.TEXT_SECONDARY};
            }}
        """)
    
    def _style_checkbox(self, checkbox):
        checkbox.setStyleSheet(f"""
            QCheckBox {{
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 22px;
                height: 22px;
                border-radius: 6px;
                border: 2px solid {Colors.BORDER};
                background: {Colors.SURFACE};
            }}
            QCheckBox::indicator:checked {{
                background: {Colors.PRIMARY};
                border-color: {Colors.PRIMARY_LIGHT};
            }}
            QCheckBox::indicator:hover {{
                border-color: {Colors.PRIMARY};
            }}
        """)
    
    def _style_combobox(self, combobox):
        combobox.setStyleSheet(f"""
            QComboBox {{
                background: {Colors.SURFACE};
                border: 1px solid {Colors.BORDER};
                border-radius: {Spacing.RADIUS_MD}px;
                color: {Colors.TEXT_PRIMARY};
                padding: 8px 12px;
                font-size: 13px;
            }}
            QComboBox:focus {{
                border-color: {Colors.PRIMARY};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 30px;
            }}
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
    
    def _save_all_settings(self):
        """Save all settings"""
        self.settings["confidence_threshold"] = self.confidence_spin.value()
        self.settings["frame_capture_interval"] = self.frame_interval_spin.value()
        self.settings["game"] = "bgmi" if self.game_combo.currentIndex() == 1 else "freefire"
        self.settings["show_detection_overlay"] = self.overlay_check.isChecked()
        self.settings["camera_index"] = self.camera_spin.value()
        self.settings["resolution"] = self.resolution_combo.currentText()
        self.settings["auto_start_preview"] = self.auto_preview_check.isChecked()
        self.settings["notifications_enabled"] = self.notifications_check.isChecked()
        self.settings["auto_update"] = self.auto_update_check.isChecked()
        
        if self._save_settings():
            QMessageBox.information(self, "Settings Saved", "Your settings have been saved successfully.")
        else:
            QMessageBox.warning(self, "Error", "Failed to save settings. Please try again.")
    
    def _reset_settings(self):
        """Reset to default settings"""
        reply = QMessageBox.question(
            self, "Reset Settings",
            "Are you sure you want to reset all settings to defaults?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.confidence_spin.setValue(0.7)
            self.frame_interval_spin.setValue(0.15)
            self.game_combo.setCurrentIndex(0)
            self.overlay_check.setChecked(True)
            self.camera_spin.setValue(0)
            self.resolution_combo.setCurrentIndex(0)
            self.auto_preview_check.setChecked(False)
            self.notifications_check.setChecked(True)
            self.auto_update_check.setChecked(True)
    
    def _handle_logout(self):
        if self.on_logout:
            self.on_logout()
    
    def _go_back(self):
        self.back_requested.emit()
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'bg_widget'):
            self.bg_widget.setGeometry(self.rect())


import sys
import cv2
import json
import os
import threading
import time
from queue import Queue
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QFrame,
    QApplication,
    QGraphicsDropShadowEffect,
    QSizePolicy,
    QSpacerItem,
)
from PyQt5.QtCore import (
    Qt,
    pyqtSignal,
    QTimer,
)
from PyQt5.QtGui import QFont, QPixmap, QImage, QPainter, QColor, QPen
import requests
from score_ai.core.config_manager import config
from score_ai.ui.components.header_widget import HeaderWidget
from score_ai.utils.responsive_utils import ResponsiveUtils, ResponsiveWidget
from score_ai.core.config_manager import config
from score_ai.core.theme import Colors, Fonts, Spacing, Gradients, Shadows, Styles

WORKSPACE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../")
)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

try:
    from score_ai.detection.freefire_bridge import obs_frame_capture
    print("Successfully imported obs_frame_capture from freefire_bridge")
except ImportError:
    try:
        from score_ai.detection.killblocks import obs_frame_capture
        print("Successfully imported obs_frame_capture from killblocks")
    except ImportError as e:
        print(f"Error importing capture module: {e}")


class MovingLine(QWidget):
    def __init__(self, parent=None, color=Colors.PRIMARY):
        super().__init__(parent)
        self.color = color
        self.line_width = 100
        self.x_position = -self.line_width
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate)
        self.animation_timer.start(16)
        self.setFixedHeight(2)
        
    def animate(self):
        self.x_position += 5
        if self.x_position > self.width():
            self.x_position = -self.line_width
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QPen(QColor(self.color), 2))
        painter.drawLine(self.x_position, 1, self.x_position + self.line_width, 1)


class CameraSetupPyQt(ResponsiveWidget):
    back_requested = pyqtSignal()
    
    def __init__(
        self,
        match_info=None,
        user_email="user@email.com",
        on_logout=None,
        access_token=None,
        on_docs_click=None,
    ):
        super().__init__()
        self.match_info = match_info or {}
        self.user_email = user_email
        self.on_logout = on_logout
        self.access_token = access_token
        self.on_docs_click = on_docs_click
        self.responsive = ResponsiveUtils()
        
        self.cap = None
        self.preview_active = False
        self.error_label = None
        
        self.obs_thread = None
        self.stop_flag = None
        self.processing_active = False
        
        self.saved_index = self.load_config()
        
        self.setWindowTitle("OBS Camera Setup - 16Score-AI")
        self.setStyleSheet(Styles.page_background())
        self._init_ui()

    def _init_ui(self):
        """Initialize the page UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header - using consistent height like other pages
        main_layout.addWidget(self._create_header())

        # Content area with responsive layout
        content_widget = QWidget()
        content_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content_widget.setStyleSheet(f"background: {Colors.BG};")
        content_layout = QHBoxLayout(content_widget)
        
        content_margins = 20
        content_spacing = 20
        content_layout.setContentsMargins(content_margins, content_margins//2, content_margins, content_margins//2)
        content_layout.setSpacing(content_spacing)

        self.create_left_panel(content_layout)
        self.create_right_panel(content_layout)

        main_layout.addWidget(content_widget)
        self.create_bottom_buttons(main_layout)

    def _create_header(self) -> QWidget:
        """Create the header container."""
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
            on_back=self.go_back,
            show_back_button=True,
            on_docs_click=self.on_docs_click,
        )
        header_layout.addWidget(header)
        
        return header_container

    def create_left_panel(self, parent_layout):
        left_frame = QFrame()
        left_frame.setStyleSheet("background: transparent;")
        
        left_min_width = 320
        left_max_width = 480
        left_frame.setMinimumWidth(left_min_width)
        left_frame.setMaximumWidth(left_max_width)
        left_frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_spacing = 15
        left_layout.setSpacing(left_spacing)

        self.create_match_details_section(left_layout)
        self.create_camera_setup_section(left_layout)

        left_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        parent_layout.addWidget(left_frame)

    def create_match_details_section(self, parent_layout):
        match_frame = QFrame()
        match_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        max_width = 480
        match_frame.setMaximumWidth(max_width)
        match_frame.setMinimumWidth(max_width)
        match_frame.setMinimumHeight(240)
        match_frame.setContentsMargins(0, 0, 0, 0)
        border_radius = Spacing.RADIUS_XL
        match_frame.setStyleSheet(f"""
            QFrame {{
                background: {Colors.CARD};
                border-radius: {border_radius}px;
                border: 1px solid {Colors.BORDER};
            }}
            QLabel {{
                border: none !important;
                background: transparent !important;
            }}
        """)
        shadow = Shadows.card()
        match_frame.setGraphicsEffect(shadow)
        match_layout = QVBoxLayout(match_frame)
        camera_margins = 12
        camera_spacing = 8
        match_layout.setContentsMargins(camera_margins, camera_margins, camera_margins, camera_margins)
        match_layout.setSpacing(camera_spacing)
        
        title = QLabel("SELECTED MATCH")
        title.setFont(Fonts.create(17, Fonts.WEIGHT_BOLD))
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; border: none; outline: none; background: transparent;")
        title.setAlignment(Qt.AlignCenter)
        title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        match_layout.addWidget(title)
        
        match_name = self.match_info.get("matchName", "No match selected")
        self.match_name_label = QLabel(match_name)
        self.match_name_label.setFont(Fonts.create(15))
        self.match_name_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; border: none; outline: none; background: transparent;")
        self.match_name_label.setWordWrap(True)
        self.match_name_label.setAlignment(Qt.AlignCenter)
        self.match_name_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        match_layout.addWidget(self.match_name_label)
        
        status = self.match_info.get("status", "").lower()
        status_color = {
            "ongoing": Colors.SUCCESS,
            "upcoming": Colors.PRIMARY_LIGHT,
            "completed": Colors.TEXT_SECONDARY,
        }.get(status, Colors.TEXT_SECONDARY)
        self.status_label = QLabel(status.upper())
        self.status_label.setFont(Fonts.create(13))
        self.status_label.setStyleSheet(f"""
            color: {status_color};
            font-weight: bold;
            border: none;
            outline: none;
            background: transparent;
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        match_layout.addWidget(self.status_label)
        
        league_info = f"{self.match_info.get('leagueGroupName', '')} - {self.match_info.get('leagueRoundName', '')}"
        if league_info.strip() != "-" and league_info.strip() != " - ":
            league_label = QLabel(league_info)
            league_label.setFont(Fonts.create(14))
            league_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; border: none; outline: none; background: transparent;")
            league_label.setAlignment(Qt.AlignCenter)
            league_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            league_label.setWordWrap(True)
            match_layout.addWidget(league_label)
        if self.match_info.get("gameMapName"):
            map_label = QLabel(f"Map: {self.match_info.get('gameMapName')}")
            map_label.setFont(Fonts.create(14))
            map_label.setStyleSheet(f"color: {Colors.PRIMARY_LIGHT}; background: transparent; border: none; outline: none;")
            map_label.setAlignment(Qt.AlignCenter)
            map_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            map_label.setWordWrap(True)
            match_layout.addWidget(map_label)
        start_date = self.match_info.get("startDate", "")
        if start_date:
            try:
                start_time = datetime.fromisoformat(start_date.replace("Z", ""))
                formatted_time = start_time.strftime("%d %b %Y, %I:%M %p")
                time_label = QLabel(formatted_time)
            except (ValueError, TypeError):
                time_label = QLabel(start_date)
            time_label.setFont(Fonts.create(14))
            time_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; border: none; outline: none; background: transparent;")
            time_label.setAlignment(Qt.AlignCenter)
            time_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            time_label.setWordWrap(True)
            match_layout.addWidget(time_label)
        parent_layout.addWidget(match_frame)

    def create_camera_setup_section(self, parent_layout):
        camera_frame = QFrame()
        max_width = 480
        camera_frame.setMaximumWidth(max_width)
        camera_frame.setMinimumWidth(max_width)
        camera_frame.setContentsMargins(0, 0, 0, 0)
        border_radius = Spacing.RADIUS_XL
        camera_frame.setStyleSheet(f"""
            QFrame {{
                background: {Colors.CARD};
                border-radius: {border_radius}px;
                border: 1px solid {Colors.BORDER};
            }}
            QLabel {{
                border: none !important;
                background: transparent !important;
            }}
        """)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(8)
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 1)
        camera_frame.setGraphicsEffect(shadow)
        camera_layout = QVBoxLayout(camera_frame)
        camera_margins = 12
        camera_spacing = 8
        camera_layout.setContentsMargins(camera_margins, camera_margins, camera_margins, camera_margins)
        camera_layout.setSpacing(camera_spacing)
        
        title = QLabel("CAMERA SETUP")
        title.setFont(Fonts.create(17, Fonts.WEIGHT_BOLD))
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; border: none; outline: none; background: transparent;")
        title.setAlignment(Qt.AlignCenter)
        camera_layout.addWidget(title)
        
        self.camera_number = QLineEdit()
        input_width = 120
        input_height = int(44 // 1.5)
        self.camera_number.setMinimumSize(input_width, int(input_height))
        self.camera_number.setMaximumSize(input_width * 2, int(input_height * 2))
        self.camera_number.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.camera_number.setFont(Fonts.create(14))
        border_radius = Spacing.RADIUS_MD
        padding_v = 4
        padding_h = 8
        font_size = 14
        self.camera_number.setStyleSheet(f"""
            QLineEdit {{
                background: {Colors.SURFACE};
                border: 1px solid {Colors.BORDER};
                border-radius: {border_radius}px;
                color: {Colors.TEXT_PRIMARY};
                padding: {padding_v}px {padding_h}px;
                font-size: {font_size}px;
                min-height: {int(input_height)}px;
            }}
            QLineEdit:focus {{
                background: {Colors.CARD};
                border-color: {Colors.PRIMARY};
            }}
        """)
        self.camera_number.setPlaceholderText("Enter camera number (0-20)")
        self.camera_number.setText(str(self.saved_index))
        self.camera_number.textChanged.connect(self.validate_camera_input)
        camera_layout.addWidget(self.camera_number, alignment=Qt.AlignCenter)
        
        camera_buttons_frame = QFrame()
        camera_buttons_frame.setStyleSheet("background: transparent; border: none;")
        camera_buttons_layout = QHBoxLayout(camera_buttons_frame)
        camera_buttons_layout.setContentsMargins(0, 0, 0, 0)
        camera_buttons_layout.setSpacing(6)
        btn_size = 32
        
        self.prev_camera_btn = QPushButton("←")
        self.prev_camera_btn.setMinimumSize(btn_size, btn_size)
        self.prev_camera_btn.setMaximumSize(btn_size * 2, btn_size * 2)
        self.prev_camera_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.prev_camera_btn.setFont(Fonts.create(22, Fonts.WEIGHT_BOLD))
        btn_border_radius = btn_size // 2
        btn_font_size = 22
        self.prev_camera_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.PRIMARY};
                color: white;
                border-radius: {btn_border_radius}px;
                border: none;
                padding: 0;
                font-size: {btn_font_size}px;
                min-width: {btn_size}px;
                min-height: {btn_size}px;
                max-width: {btn_size}px;
                max-height: {btn_size}px;
                text-align: center;
            }}
            QPushButton:hover {{
                background: {Colors.PRIMARY_DARK};
            }}
            QPushButton:pressed {{
                background: {Colors.PRIMARY_DARK};
            }}
        """)
        self.prev_camera_btn.clicked.connect(self.decrement_camera_index)
        camera_buttons_layout.addWidget(self.prev_camera_btn)
        
        self.next_camera_btn = QPushButton("→")
        self.next_camera_btn.setMinimumSize(btn_size, btn_size)
        self.next_camera_btn.setMaximumSize(btn_size * 2, btn_size * 2)
        self.next_camera_btn.setFont(Fonts.create(22, Fonts.WEIGHT_BOLD))
        self.next_camera_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.PRIMARY};
                color: white;
                border-radius: {btn_border_radius}px;
                border: none;
                padding: 0;
                font-size: {btn_font_size}px;
                min-width: {btn_size}px;
                min-height: {btn_size}px;
                max-width: {btn_size}px;
                max-height: {btn_size}px;
                text-align: center;
            }}
            QPushButton:hover {{
                background: {Colors.PRIMARY_DARK};
            }}
            QPushButton:pressed {{
                background: {Colors.PRIMARY_DARK};
            }}
        """)
        self.next_camera_btn.clicked.connect(self.increment_camera_index)
        camera_buttons_layout.addWidget(self.next_camera_btn)
        camera_layout.addWidget(camera_buttons_frame)
        
        self.test_btn = QPushButton("Test Camera")
        test_btn_width = 120
        test_btn_height = int(48 // 1.5)
        self.test_btn.setMinimumSize(test_btn_width, test_btn_height)
        self.test_btn.setMaximumSize(test_btn_width * 2, test_btn_height * 2)
        self.test_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.test_btn.setFont(Fonts.button(14))
        test_btn_border_radius = 10
        test_btn_font_size = 14
        test_btn_padding = 10
        self.test_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.PRIMARY};
                color: white;
                border-radius: {test_btn_border_radius}px;
                font-size: {test_btn_font_size}px;
                font-weight: 700;
                padding: 0 {test_btn_padding}px;
                min-height: {test_btn_height}px;
            }}
            QPushButton:hover {{
                background: {Colors.PRIMARY_DARK};
            }}
            QPushButton:pressed {{
                background: {Colors.PRIMARY_DARK};
            }}
        """)
        self.test_btn.clicked.connect(self.toggle_preview)
        camera_layout.addWidget(self.test_btn, alignment=Qt.AlignCenter)
        
        self.set_camera_btn = QPushButton("Set Camera Index")
        set_btn_width = 120
        set_btn_height = int(48 // 1.5)
        self.set_camera_btn.setMinimumSize(set_btn_width, set_btn_height)
        self.set_camera_btn.setMaximumSize(set_btn_width * 2, set_btn_height * 2)
        self.set_camera_btn.setFont(Fonts.button(14))
        self.set_camera_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.PRIMARY_DARK};
                color: white;
                border-radius: {test_btn_border_radius}px;
                font-size: {test_btn_font_size}px;
                font-weight: 700;
                padding: 0 {test_btn_padding}px;
                min-height: {set_btn_height}px;
            }}
            QPushButton:hover {{
                background: {Colors.PRIMARY};
            }}
            QPushButton:pressed {{
                background: {Colors.PRIMARY_DARK};
            }}
        """)
        self.set_camera_btn.clicked.connect(self.set_camera_index)
        camera_layout.addWidget(self.set_camera_btn, alignment=Qt.AlignCenter)
        
        help_text = QLabel("Click 'Know more' for camera index guide")
        help_text.setFont(Fonts.create(11))
        help_text.setStyleSheet(f"color: {Colors.TEXT_MUTED}; border: none; background: transparent;")
        help_text.setAlignment(Qt.AlignCenter)
        help_text.setCursor(Qt.PointingHandCursor)
        help_text.mousePressEvent = self.show_tooltip
        camera_layout.addWidget(help_text)
        parent_layout.addWidget(camera_frame)

    def create_right_panel(self, parent_layout):
        right_frame = QFrame()
        right_frame.setStyleSheet("background: transparent;")
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.preview_frame = QFrame()
        self.preview_frame.setStyleSheet(f"""
            QFrame {{
                background: {Colors.CARD};
                border-radius: {Spacing.RADIUS_XXL}px;
                border: 1px solid {Colors.BORDER};
            }}
        """)

        shadow = Shadows.card()
        self.preview_frame.setGraphicsEffect(shadow)

        preview_layout = QVBoxLayout(self.preview_frame)
        preview_layout.setContentsMargins(10, 10, 10, 10)

        self.preview_label = QLabel(
            "Camera preview will appear here\n\nClick 'Test Camera' to start preview"
        )
        self.preview_label.setFont(Fonts.create(28, Fonts.WEIGHT_BOLD))
        self.preview_label.setStyleSheet(f"""
            QLabel {{
                color: {Colors.TEXT_SECONDARY};
                background: {Colors.SURFACE};
                border-radius: {Spacing.RADIUS_LG}px;
                padding: 40px;
                border: none;
            }}
        """)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(330, 180)
        self.preview_frame.setMinimumSize(350, 200)
        preview_layout.addWidget(self.preview_label)

        right_layout.addWidget(self.preview_frame)
        parent_layout.addWidget(right_frame)

    def create_bottom_buttons(self, parent_layout):
        button_frame = QFrame()
        button_frame.setStyleSheet("background: transparent;")
        button_layout = QHBoxLayout(button_frame)
        button_layout.setSpacing(40)
        button_layout.setContentsMargins(0, 20, 0, 20)
        button_layout.addStretch()

        # Start Match button
        self.start_match_btn = QPushButton("Start Match")
        start_btn_width = 200
        start_btn_height = 70
        self.start_match_btn.setMinimumSize(start_btn_width, start_btn_height)
        self.start_match_btn.setMaximumSize(start_btn_width * 2, start_btn_height * 2)
        self.start_match_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.start_match_btn.setFont(Fonts.button(20))
        start_btn_border_radius = start_btn_height // 2
        self.start_match_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.SUCCESS};
                color: white;
                border-radius: {start_btn_border_radius}px;
                border: none;
                font-size: 20px;
                font-weight: 700;
                padding: 0 20px;
                min-height: {start_btn_height}px;
            }}
            QPushButton:hover {{
                background: #1a9c6c;
            }}
            QPushButton:pressed {{
                background: #148a5d;
            }}
            QPushButton:disabled {{
                background: {Colors.CARD};
                color: {Colors.TEXT_MUTED};
            }}
        """)
        self.start_match_btn.clicked.connect(self.start_match)
        button_layout.addWidget(self.start_match_btn)

        # Update Players button
        self.update_players_btn = QPushButton("Update Players")
        update_btn_width = 200
        update_btn_height = 70
        self.update_players_btn.setMinimumSize(update_btn_width, update_btn_height)
        self.update_players_btn.setMaximumSize(update_btn_width * 2, update_btn_height * 2)
        self.update_players_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.update_players_btn.setFont(Fonts.button(20))
        update_btn_border_radius = update_btn_height // 2
        self.update_players_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.INFO};
                color: white;
                border-radius: {update_btn_border_radius}px;
                border: none;
                font-size: 20px;
                font-weight: 700;
                padding: 0 20px;
                min-height: {update_btn_height}px;
            }}
            QPushButton:hover {{
                background: #2b71d9;
            }}
            QPushButton:pressed {{
                background: #2563c4;
            }}
            QPushButton:disabled {{
                background: {Colors.CARD};
                color: {Colors.TEXT_MUTED};
            }}
        """)
        self.update_players_btn.clicked.connect(self.update_team_players)
        button_layout.addWidget(self.update_players_btn)

        # Stop Match button
        self.stop_match_btn = QPushButton("Stop Match")
        stop_btn_width = 200
        stop_btn_height = 70
        self.stop_match_btn.setMinimumSize(stop_btn_width, stop_btn_height)
        self.stop_match_btn.setMaximumSize(stop_btn_width * 2, stop_btn_height * 2)
        self.stop_match_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.stop_match_btn.setFont(Fonts.button(20))
        stop_btn_border_radius = stop_btn_height // 2
        self.stop_match_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ERROR};
                color: white;
                border-radius: {stop_btn_border_radius}px;
                border: none;
                font-size: 20px;
                font-weight: 700;
                padding: 0 20px;
                min-height: {stop_btn_height}px;
            }}
            QPushButton:hover {{
                background: #c62828;
            }}
            QPushButton:pressed {{
                background: #b71c1c;
            }}
            QPushButton:disabled {{
                background: {Colors.CARD};
                color: {Colors.TEXT_MUTED};
            }}
        """)
        self.stop_match_btn.clicked.connect(self.stop_match)
        self.stop_match_btn.setEnabled(False)
        button_layout.addWidget(self.stop_match_btn)
        button_layout.addStretch()

        parent_layout.addWidget(button_frame)

    def toggle_preview(self):
        if self.preview_active:
            self.stop_preview()
        else:
            self.start_preview()

    def start_preview(self):
        try:
            camera_index = int(self.camera_number.text())
            cv2.ocl.setUseOpenCL(False)
            self.cap = cv2.VideoCapture(camera_index)
            
            camera_config = config.get_camera_config()
            default_width = camera_config.get('default_width', 1920)
            default_height = camera_config.get('default_height', 1080)
            
            resolutions = [
                (default_width, default_height),
                (1280, 720),
                (640, 480),
            ]
            
            for width, height in resolutions:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if actual_width >= width and actual_height >= height:
                    break
                
            if not self.cap.isOpened():
                self.show_error("Failed to open camera")
                return
                
            self.preview_active = True
            self.test_btn.setText("Stop Preview")
            self.test_btn.setStyleSheet(f"""
                QPushButton {{
                    background: {Colors.ERROR};
                    color: white;
                    border-radius: 10px;
                    border: none;
                    font-size: 14px;
                    font-weight: 700;
                    min-height: 32px;
                }}
                QPushButton:hover {{
                    background: #c62828;
                }}
            """)
            self.clear_error()
            
            self.video_thread = threading.Thread(
                target=self.update_preview, daemon=True
            )
            self.video_thread.start()
            
            if self.processing_active:
                self.show_success(
                    "Camera preview started! Frame capture is also running."
                )
            else:
                self.show_success("Camera preview started! This is for testing only.")

        except Exception as e:
            self.show_error(str(e))

    def update_preview(self):
        next_frame_time = time.time()
        while self.preview_active and self.cap.isOpened():
            try:
                frame_start = time.time()
                
                ret, original_frame = self.cap.read()
                if not ret:
                    break
                
                preview_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
                height, width = preview_frame.shape[:2]
                max_width = 600
                
                if width > max_width:
                    scale = max_width / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    preview_frame = cv2.resize(preview_frame, (new_width, new_height))

                h, w, ch = preview_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(
                    preview_frame.data, w, h, bytes_per_line, QImage.Format_RGB888
                )
                pixmap = QPixmap.fromImage(qt_image)
                
                self.preview_label.setPixmap(pixmap)
                self.preview_label.setScaledContents(True)
                
                next_frame_time += 1.0
                sleep_time = next_frame_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_frame_time = time.time()

            except Exception as e:
                self.stop_preview()
                break

    def stop_preview(self):
        self.preview_active = False

        if self.cap:
            self.cap.release()
            self.cap = None

        self.test_btn.setText("Test Camera")
        self.test_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.PRIMARY};
                color: white;
                border-radius: 10px;
                border: none;
                font-size: 14px;
                font-weight: 700;
                min-height: 32px;
            }}
            QPushButton:hover {{
                background: {Colors.PRIMARY_DARK};
            }}
        """)
        self.preview_label.setText(
            "Camera preview will appear here\n\nClick 'Test Camera' to start preview"
        )
        self.preview_label.setStyleSheet(f"""
            QLabel {{
                color: {Colors.TEXT_SECONDARY};
                background: {Colors.SURFACE};
                border-radius: {Spacing.RADIUS_LG}px;
                padding: 40px;
                border: none;
            }}
        """)
        self.preview_label.setPixmap(QPixmap())

    def show_error(self, message):
        self.clear_error()
        self.error_label = QLabel(message)
        self.error_label.setFont(Fonts.create(14))
        self.error_label.setStyleSheet(f"""
            color: {Colors.ERROR};
            background: {Colors.with_alpha(Colors.ERROR.lstrip('#'), 0.15)};
            border: 1px solid {Colors.ERROR};
            border-radius: {Spacing.RADIUS_MD}px;
            padding: 12px 16px;
            margin: 10px;
        """)
        self.error_label.setAlignment(Qt.AlignCenter)
        self.error_label.setWordWrap(True)
        self.layout().addWidget(self.error_label)

    def clear_error(self):
        if self.error_label:
            self.error_label.deleteLater()
            self.error_label = None

    def start_match(self):
        try:
            if self.is_capture_running():
                self.show_error(
                    "Frame capture is already running. Please stop it first."
                )
                return
                
            if self.preview_active:
                print("Preview is active - keeping it running during frame capture...")

            if not self.match_info or "id" not in self.match_info:
                self.show_error("No match selected. Please select a match first.")
                return
                
            match_id = self.match_info["id"]
            if not match_id:
                self.show_error("Invalid match ID. Please select a valid match.")
                return

            try:
                backend_url = config.get('api.backend_url', 'http://192.168.1.11:5006').rstrip('/')
                start_match_endpoint = config.get('endpoints.start_match', 'LeagueMatch/startMatchById')
                response = requests.post(
                    f"{backend_url}/{start_match_endpoint}?id={match_id}",
                    headers={"Authorization": f"Bearer {self.access_token}"},
                )
                
                if (
                    response.status_code == 400
                    and "Error starting match" in response.text
                ):
                    pass
                elif response.status_code != 200:
                    self.show_error(f"Failed to start match: {response.text}")
                    return
                    
            except Exception as e:
                self.show_error(f"Error calling start match API: {str(e)}")
                return
                
            camera_index = int(self.camera_number.text())
            
            print(
                f"Starting OBS frame capture for match_id: {match_id}, camera_index: {camera_index}"
            )
            self.start_obs_capture(match_id, camera_index)
            
        except Exception as e:
            print(f"Error starting capture: {str(e)}")
            self.show_error(f"Error starting capture: {str(e)}")

    def start_obs_capture(self, match_id, camera_index):
        if (
            hasattr(self, "obs_thread")
            and self.obs_thread
            and self.obs_thread.is_alive()
        ):
            print("Stopping previous capture thread...")
            if hasattr(self, "stop_flag"):
                self.stop_flag.set()
            self.obs_thread.join(timeout=2)
        
        self.processing_active = True
        
        self.stop_flag = threading.Event()
        
        self.obs_thread = threading.Thread(
            target=self.run_obs_capture, args=(match_id, camera_index), daemon=True
        )
        self.obs_thread.start()
        
        self.start_match_btn.setEnabled(False)
        self.stop_match_btn.setEnabled(True)
        
        if hasattr(self, "status_label"):
            self.status_label.setText("LIVE")
            self.status_label.setStyleSheet(f"""
                color: {Colors.ERROR};
                font-size: 12px;
                font-weight: bold;
                padding: 4px 12px;
                border-radius: 6px;
                background: {Colors.with_alpha(Colors.ERROR.lstrip('#'), 0.15)};
            """)

        if self.preview_active:
            self.show_success("Frame capture started! Preview is still running.")
        else:
            self.show_success("Frame capture started!")

    def run_obs_capture(self, match_id, camera_index):
        try:
            print(f"Starting OBS capture thread for match {match_id}")
            game = config.get_detection_config().get("game", "freefire")
            if game == "freefire":
                from score_ai.detection.freefire_bridge import obs_frame_capture as capture_fn
                print(
                    "[capture] Free Fire: ffkillblock.py + bestffmax.pt "
                    "(YOLO loads first, OCR in background)",
                    flush=True,
                )
            else:
                from score_ai.detection.killblocks import obs_frame_capture as capture_fn
                print("[capture] Using BGMI killblocks pipeline")
            capture_fn(match_id, self.access_token, camera_index, self.stop_flag)
        except Exception as e:
            print(f"Error in frame capture: {str(e)}")
            QTimer.singleShot(
                0, lambda: self.show_error(f"Error in frame capture: {str(e)}")
            )
        finally:
            print("OBS capture thread finished")
            QTimer.singleShot(0, self.on_capture_finished)

    def on_capture_finished(self):
        self.processing_active = False
        self.start_match_btn.setEnabled(True)
        self.stop_match_btn.setEnabled(False)

        if hasattr(self, "status_label") and self.match_info:
            status = self.match_info.get("status", "").lower()
            status_color = {
                "ongoing": Colors.SUCCESS,
                "upcoming": Colors.PRIMARY_LIGHT,
                "completed": Colors.TEXT_SECONDARY,
            }.get(status, Colors.TEXT_SECONDARY)

            self.status_label.setText(status.upper())
            self.status_label.setStyleSheet(f"""
                color: {status_color};
                font-size: 12px;
                font-weight: bold;
                padding: 4px 12px;
                border-radius: 6px;
                background: transparent;
            """)

    def go_back(self):
        self.force_stop_all()
        self.back_requested.emit()

    def validate_camera_input(self, value):
        if value == "":
            return
        try:
            num = int(value)
            if not (0 <= num <= 30):
                self.camera_number.setText(str(max(0, min(30, num))))
        except ValueError:
            filtered = "".join(filter(str.isdigit, value))
            if filtered:
                num = int(filtered)
                self.camera_number.setText(str(max(0, min(30, num))))
            else:
                self.camera_number.setText("0")

    def load_config(self):
        config_path = os.path.join(os.path.expanduser("~"), ".esports_ai_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config_data = json.load(f)
                    return config_data.get("camera_index", 0)
            except Exception as e:
                print(f"Error loading config: {e}")
        return 0

    def show_tooltip(self, event=None):
        tooltip_text = """Camera Index Guide:

Camera Numbers:
- 0 - Primary/Built-in Webcam
- 1 - First External Camera
- 2-20 - Additional Cameras

How it works:
- Windows assigns numbers to cameras
- Lower numbers = Earlier connected devices
- Disconnecting/reconnecting may change numbers

Tips:
- Start with 0 for built-in webcam
- Try next numbers for external cameras
- Test each index until you see preview
- Remember to save working index

To return to camera preview, click 'Test Camera'"""
        
        self.preview_label.setText(tooltip_text)
        self.preview_label.setStyleSheet(f"""
            QLabel {{
                background: {Colors.SURFACE};
                color: {Colors.TEXT_PRIMARY};
                border-radius: {Spacing.RADIUS_LG}px;
                padding: 25px;
                text-align: left;
                border: 1px solid {Colors.PRIMARY};
                font-size: 13px;
                line-height: 1.5;
            }}
        """)
        self.preview_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

    def show_success(self, message):
        self.clear_error()
        success_label = QLabel(message)
        success_label.setFont(Fonts.create(14))
        success_label.setStyleSheet(f"""
            color: {Colors.SUCCESS};
            background: {Colors.with_alpha(Colors.SUCCESS.lstrip('#'), 0.15)};
            border: 1px solid {Colors.SUCCESS};
            border-radius: {Spacing.RADIUS_MD}px;
            padding: 12px 16px;
            margin: 10px;
        """)
        success_label.setAlignment(Qt.AlignCenter)
        success_label.setWordWrap(True)
        self.layout().addWidget(success_label)
        
        duration = 1000 if "preview" in message.lower() else 2000
        QTimer.singleShot(duration, success_label.deleteLater)

    def force_stop_all(self):
        print("Force stopping all capture processes...")
        
        if self.preview_active:
            self.stop_preview()
        
        if hasattr(self, "stop_flag") and self.stop_flag:
            self.stop_flag.set()
        
        if (
            hasattr(self, "obs_thread")
            and self.obs_thread
            and self.obs_thread.is_alive()
        ):
            print("Force stopping OBS capture thread...")
            self.obs_thread.join(timeout=1)
        
        self.processing_active = False
        
        self.start_match_btn.setEnabled(True)
        self.stop_match_btn.setEnabled(False)
        
        if hasattr(self, "status_label") and self.match_info:
            status = self.match_info.get("status", "").lower()
            status_color = {
                "ongoing": Colors.SUCCESS,
                "upcoming": Colors.PRIMARY_LIGHT,
                "completed": Colors.TEXT_SECONDARY,
            }.get(status, Colors.TEXT_SECONDARY)

            self.status_label.setText(status.upper())
            self.status_label.setStyleSheet(f"""
                color: {status_color};
                font-size: 12px;
                font-weight: bold;
                padding: 4px 12px;
                border-radius: 6px;
                background: transparent;
            """)

        print("Force stop complete")

    def update_team_players(self):
        try:
            if not self.match_info or "id" not in self.match_info:
                self.show_error("No match selected. Please select a match first.")
                return
                
            match_id = self.match_info["id"]
            if not match_id:
                self.show_error("Invalid match ID. Please select a valid match.")
                return
            
            print(f"Updating team players for match: {match_id}")
            
            from score_ai.detection.killblocks import fetch_and_update_team_players
            
            result = fetch_and_update_team_players(match_id, self.access_token)
            
            if result:
                print(f"Successfully updated team players for match {match_id}")
                self.show_success(f"Successfully updated team players!\n{len(result)} teams loaded.")
            else:
                print(f"Failed to update team players for match {match_id}")
                self.show_error("Failed to update team players. Please check the match ID and try again.")
                
        except Exception as e:
            print(f"Error updating team players: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            self.show_error(f"Error updating team players: {str(e)}")

    def stop_match(self):
        try:
            print("Stopping frame capture processes...")
            
            if hasattr(self, "stop_flag") and self.stop_flag:
                self.stop_flag.set()
                print("Stop flag set, waiting for OBS capture to stop...")
            
            if (
                hasattr(self, "obs_thread")
                and self.obs_thread
                and self.obs_thread.is_alive()
            ):
                print("Stopping OBS capture thread...")
                self.obs_thread.join(timeout=3)
                
                if self.obs_thread.is_alive():
                    print("Warning: OBS thread did not stop gracefully, but continuing...")

            if self.cap and self.cap.isOpened() and not self.preview_active:
                self.cap.release()
                self.cap = None
            
            self.start_match_btn.setEnabled(True)
            self.stop_match_btn.setEnabled(False)
            
            self.processing_active = False
            
            if hasattr(self, "status_label") and self.match_info:
                status = self.match_info.get("status", "").lower()
                status_color = {
                    "ongoing": Colors.SUCCESS,
                    "upcoming": Colors.PRIMARY_LIGHT,
                    "completed": Colors.TEXT_SECONDARY,
                }.get(status, Colors.TEXT_SECONDARY)

                self.status_label.setText(status.upper())
                self.status_label.setStyleSheet(f"""
                    color: {status_color};
                    font-size: 12px;
                    font-weight: bold;
                    padding: 4px 12px;
                    border-radius: 6px;
                    background: transparent;
                """)

            print("Frame capture processes stopped successfully")
            self.show_success("Frame capture stopped!")
            
        except Exception as e:
            print(f"Error stopping capture: {str(e)}")
            self.force_stop_all()

    def increment_camera_index(self):
        try:
            current_index = int(self.camera_number.text())
            if current_index < 20:
                new_index = current_index + 1
                self.camera_number.setText(str(new_index))
                if self.preview_active:
                    self.stop_preview()
                    self.start_preview()
        except Exception as e:
            self.show_error(str(e))

    def decrement_camera_index(self):
        try:
            current_index = int(self.camera_number.text())
            if current_index > 0:
                new_index = current_index - 1
                self.camera_number.setText(str(new_index))
                if self.preview_active:
                    self.stop_preview()
                    self.start_preview()
        except Exception as e:
            self.show_error(str(e))

    def set_camera_index(self):
        try:
            current_index = int(self.camera_number.text())
            
            if self.preview_active:
                self.stop_preview()
                self.start_preview()
                
            self.show_success(f"Camera index set to {current_index}")
        except Exception as e:
            self.show_error(f"Error setting camera index: {str(e)}")

    def __del__(self):
        try:
            self.force_stop_all()
        except:
            pass

    def closeEvent(self, event):
        if self.is_capture_running():
            print("Window closing while capture is running, stopping all processes...")
            self.force_stop_all()
        event.accept()

    def is_capture_running(self):
        return self.processing_active or (
            hasattr(self, "obs_thread")
            and self.obs_thread
            and self.obs_thread.is_alive()
        )

    def is_preview_active(self):
        return self.preview_active

    def update_match_info(self, new_match_info):
        self.match_info = new_match_info or {}

        if hasattr(self, "match_name_label"):
            match_name = self.match_info.get("matchName", "No match selected")
            self.match_name_label.setText(match_name)

        if hasattr(self, "status_label"):
            status = self.match_info.get("status", "").lower()
            status_color = {
                "ongoing": Colors.SUCCESS,
                "upcoming": Colors.PRIMARY_LIGHT,
                "completed": Colors.TEXT_SECONDARY,
            }.get(status, Colors.TEXT_SECONDARY)

            self.status_label.setText(status.upper())
            self.status_label.setStyleSheet(f"""
                color: {status_color};
                font-size: 12px;
                font-weight: bold;
                padding: 4px 12px;
                border-radius: 6px;
                background: transparent;
            """)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_responsive_layout()
    
    def update_responsive_layout(self):
        content_widget = self.layout().itemAt(1).widget()
        if content_widget:
            content_layout = content_widget.layout()
            
            margins = 20
            spacing = 20
            
            content_layout.setContentsMargins(margins, margins//2, margins, margins//2)
            content_layout.setSpacing(spacing)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dummy_match = {
        "id": "test-id",
        "matchName": "Dummy Match",
        "matchDate": "2025-06-23T14:44:53.044169",
        "leagueGroupName": "Group X",
        "leagueRoundName": "Round 1",
        "gameMapName": "TestMap",
        "matchDescription": "This is a dummy match for testing.",
        "startDate": "2025-06-23T15:11:00",
        "endDate": "2025-06-24T15:11:00",
        "status": "Ongoing",
    }
    w = CameraSetupPyQt(
        dummy_match, user_email="dhptc08@gmail.com", access_token="dummy_token"
    )
    w.show()
    sys.exit(app.exec_())

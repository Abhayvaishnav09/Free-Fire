"""Main application window with custom frameless title bar"""
from __future__ import annotations

import sys
from PyQt5.QtWidgets import (
    QMainWindow,
    QStackedWidget,
    QApplication,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QSizePolicy,
    QGraphicsDropShadowEffect,
)
from PyQt5.QtCore import Qt, QPoint, QTimer
from PyQt5.QtGui import QFont, QColor, QIcon
import os
from score_ai.ui.pages.login_page import LoginPage
from score_ai.ui.pages.home_page import HomeScreen
from score_ai.ui.pages.organization_page import Screen2Page
from score_ai.ui.pages.tournament_page import TournamentPage
from score_ai.ui.pages.match_page import MatchPage
from score_ai.ui.pages.streaming_source_page import StreamingSourcePage
from score_ai.ui.pages.camera_setup_pyqt import CameraSetupPyQt
from score_ai.ui.pages.instruction_page import InstructionPage
from score_ai.ui.components.custom_titlebar import CustomTitleBar
from score_ai.utils.responsive_utils import ResponsiveUtils
from score_ai.utils.helpers import get_login_history
from score_ai.core.security import SecureStorage
from score_ai.core.api_service import user_api
from score_ai.core.preferences import preferences


class MainWindow(QMainWindow):
    """Main application window with modern frameless design"""
    
    RESIZE_MARGIN = 6
    
    def __init__(self):
        super().__init__()
        
        # Enable frameless window
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        
        # Resize tracking
        self._resize_direction = None
        self._resize_start_pos = None
        self._resize_start_geometry = None
        self._is_maximized = False
        self._normal_geometry = None
        self.setMouseTracking(True)
        
        # Initialize responsive utilities
        self.responsive = ResponsiveUtils()
        
        # Set up the main window
        self.setWindowTitle("16Score-AI")
        
        # Set window icon
        ico_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "images", "16score.ico"))
        png_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "images", "16score_logo.png"))
        if os.path.exists(ico_path):
            self.setWindowIcon(QIcon(ico_path))
        elif os.path.exists(png_path):
            self.setWindowIcon(QIcon(png_path))
        self.init_responsive_ui()
        
        # Store current user credentials
        self.current_token = None
        self.current_user_email = None
        self.current_user_id = None

        # Initialize with login page
        self.login_page = LoginPage()
        self.login_page.login_successful.connect(self.show_organization_page)
        self.stacked_widget.addWidget(self.login_page)

        # Restore window geometry or show maximized
        self._restore_window_geometry()
        
        # Enable high DPI scaling
        self.setAttribute(Qt.WA_AcceptTouchEvents, True)
        
        # Attempt auto-login after a short delay to ensure UI is ready
        QTimer.singleShot(100, self.attempt_auto_login)

    def attempt_auto_login(self):
        """Attempt to auto-login using stored credentials"""
        try:
            print("[AUTO-LOGIN] Checking for stored credentials...", flush=True)
            
            # Get the last logged-in user from history
            login_history = get_login_history()
            if not login_history:
                print("[AUTO-LOGIN] No login history found", flush=True)
                return
            
            # Get the most recent login entry
            last_login = login_history[-1]
            user_email = last_login.get("email")
            
            if not user_email:
                print("[AUTO-LOGIN] No email found in login history", flush=True)
                return
            
            print(f"[AUTO-LOGIN] Found last logged-in user: {user_email}", flush=True)
            
            # Check if there's a stored token for this user
            stored_token = SecureStorage.get_token(user_email)
            if not stored_token:
                print("[AUTO-LOGIN] No stored token found for user", flush=True)
                return
            
            print("[AUTO-LOGIN] Found stored token, validating...", flush=True)
            
            # Set the token and validate it by making an API call
            user_api.set_auth_token(stored_token)
            
            # Validate token by getting user info
            user_response = user_api.get_user_from_token()
            
            if user_response.success:
                user_id = user_response.data.get("data", {}).get("username")
                print(f"[AUTO-LOGIN] Token valid! User ID: {user_id}", flush=True)
                
                # Token is valid - redirect to organization page
                self.show_organization_page(stored_token, user_email, user_id)
            else:
                print("[AUTO-LOGIN] Token validation failed, clearing stored token", flush=True)
                # Token is invalid - clear it and stay on login page
                SecureStorage.delete_token(user_email)
                user_api.clear_auth_token()
                
        except Exception as e:
            import traceback
            print(f"[AUTO-LOGIN] Error during auto-login: {e}", flush=True)
            traceback.print_exc()
            # Clear any set token on error
            user_api.clear_auth_token()

    def init_responsive_ui(self):
        """Initialize responsive UI components"""
        # Create central widget with responsive properties
        self.central_widget = QWidget()
        self.central_widget.setStyleSheet("background: #0A0C10;")  # Brand dark background
        self.setCentralWidget(self.central_widget)
        
        # Set size policy for central widget
        self.central_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Create main layout
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Add custom title bar
        self.title_bar = CustomTitleBar(self, "16Score AI")
        self.title_bar.minimize_clicked.connect(self.showMinimized)
        self.title_bar.maximize_clicked.connect(self.toggle_maximize)
        self.title_bar.close_clicked.connect(self.close)
        self.main_layout.addWidget(self.title_bar)
        
        # Content area
        self.content_widget = QWidget()
        self.content_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.content_widget.setStyleSheet("background: transparent;")
        
        self.layout = QVBoxLayout(self.content_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # Create stacked widget for pages
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Add stacked widget to layout
        self.layout.addWidget(self.stacked_widget)
        self.main_layout.addWidget(self.content_widget)
        
        # Set minimum size based on screen resolution
        min_width = self.responsive.get_responsive_width(60)  # 60% of screen width minimum
        min_height = self.responsive.get_responsive_height(70)  # 70% of screen height minimum
        self.setMinimumSize(min_width, min_height)
        
        # Apply global styles to remove focus indicators
        self.setStyleSheet("""
            QWidget:focus {
                outline: none;
                border: none;
            }
            QPushButton:focus {
                outline: none;
                border: none;
            }
            QLabel:focus {
                outline: none;
                border: none;
            }
            QFrame:focus {
                outline: none;
                border: none;
            }
        """)
        
    def toggle_maximize(self):
        """Toggle between maximized and normal window state"""
        if self._is_maximized:
            if self._normal_geometry:
                self.setGeometry(self._normal_geometry)
            else:
                self.showNormal()
            self._is_maximized = False
        else:
            self._normal_geometry = self.geometry()
            self.showMaximized()
            self._is_maximized = True
        self.title_bar.set_maximized(self._is_maximized)
        
    def _get_resize_direction(self, pos):
        """Determine resize direction based on cursor position"""
        if self._is_maximized:
            return None
            
        rect = self.rect()
        x, y = pos.x(), pos.y()
        w, h = rect.width(), rect.height()
        margin = self.RESIZE_MARGIN
        
        # Skip title bar area
        if y < self.title_bar.height():
            return None
        
        left = x < margin
        right = x > w - margin
        top = y < margin
        bottom = y > h - margin
        
        if top and left:
            return "top-left"
        elif top and right:
            return "top-right"
        elif bottom and left:
            return "bottom-left"
        elif bottom and right:
            return "bottom-right"
        elif left:
            return "left"
        elif right:
            return "right"
        elif top:
            return "top"
        elif bottom:
            return "bottom"
        return None
        
    def _update_cursor(self, direction):
        """Update cursor based on resize direction"""
        cursors = {
            "left": Qt.SizeHorCursor,
            "right": Qt.SizeHorCursor,
            "top": Qt.SizeVerCursor,
            "bottom": Qt.SizeVerCursor,
            "top-left": Qt.SizeFDiagCursor,
            "bottom-right": Qt.SizeFDiagCursor,
            "top-right": Qt.SizeBDiagCursor,
            "bottom-left": Qt.SizeBDiagCursor,
        }
        if direction:
            self.setCursor(cursors.get(direction, Qt.ArrowCursor))
        else:
            self.unsetCursor()
            
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            direction = self._get_resize_direction(event.pos())
            if direction:
                self._resize_direction = direction
                self._resize_start_pos = event.globalPos()
                self._resize_start_geometry = self.geometry()
        super().mousePressEvent(event)
        
    def mouseMoveEvent(self, event):
        if self._resize_direction and self._resize_start_pos:
            self._do_resize(event.globalPos())
        else:
            direction = self._get_resize_direction(event.pos())
            self._update_cursor(direction)
        super().mouseMoveEvent(event)
        
    def mouseReleaseEvent(self, event):
        self._resize_direction = None
        self._resize_start_pos = None
        self._resize_start_geometry = None
        super().mouseReleaseEvent(event)
        
    def _do_resize(self, global_pos):
        """Perform window resize"""
        diff = global_pos - self._resize_start_pos
        geo = self._resize_start_geometry
        min_w, min_h = self.minimumWidth(), self.minimumHeight()
        
        new_x = geo.x()
        new_y = geo.y()
        new_w = geo.width()
        new_h = geo.height()
        
        if "left" in self._resize_direction:
            new_w = geo.width() - diff.x()
            if new_w >= min_w:
                new_x = geo.x() + diff.x()
            else:
                new_w = min_w
                
        if "right" in self._resize_direction:
            new_w = geo.width() + diff.x()
            if new_w < min_w:
                new_w = min_w
                
        if "top" in self._resize_direction:
            new_h = geo.height() - diff.y()
            if new_h >= min_h:
                new_y = geo.y() + diff.y()
            else:
                new_h = min_h
                
        if "bottom" in self._resize_direction:
            new_h = geo.height() + diff.y()
            if new_h < min_h:
                new_h = min_h
                
        self.setGeometry(new_x, new_y, new_w, new_h)

    def show_organization_page(self, token, user_email, user_id):
        print(f"[MAIN] show_organization_page called with user_email={user_email}, user_id={user_id}")
        try:
            from score_ai.detection.freefire_bridge import sync_desktop_auth
            sync_desktop_auth(access_token=token, user_email=user_email)
        except Exception:
            pass
        try:
            org_page = Screen2Page(token, user_email, user_id, on_logout=self.handle_logout, on_docs_click=lambda: self.show_instruction_page(token, user_email, user_id))
            def on_open_home():
                try:
                    print(f"[MAIN] open_home_requested signal received! Navigating to home page...")
                    import sys
                    sys.stdout.flush()
                    self.show_home_page(token, user_email, user_id)
                    print(f"[MAIN] show_home_page completed")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"[MAIN] ERROR in on_open_home: {e}")
                    import traceback
                    traceback.print_exc()
            org_page.open_home_requested.connect(on_open_home)
            org_page.open_tournament_requested.connect(
                lambda: self.show_tournament_page(token, user_email, user_id)
            )
            org_page.back_requested.connect(self.go_back)
            org_page.go_home_requested.connect(lambda: self.show_home_page(token, user_email, user_id))
            self.stacked_widget.addWidget(org_page)
            self.stacked_widget.setCurrentWidget(org_page)
            print(f"[MAIN] Organization page displayed successfully")
        except Exception as e:
            import traceback
            print(f"[MAIN] Error showing organization page: {e}")
            traceback.print_exc()

    def show_home_page(self, token, user_email, user_id):
        try:
            print(f"[MAIN] show_home_page starting...")
            import sys
            sys.stdout.flush()
            self.current_token = token
            self.current_user_email = user_email
            self.current_user_id = user_id
            print(f"[MAIN] Creating HomeScreen...")
            sys.stdout.flush()
            self.home_page = HomeScreen(token, user_email, user_id, on_logout=self.handle_logout, on_docs_click=lambda: self.show_instruction_page(token, user_email, user_id))
            print(f"[MAIN] HomeScreen created, connecting signals...")
            sys.stdout.flush()
            self.home_page.open_tournament_requested.connect(
                lambda: self.show_tournament_page(token, user_email, user_id)
            )
            self.home_page.home_nav_requested.connect(
                lambda: self.show_organization_page(token, user_email, user_id)
            )
            self.home_page.handle_home_click = lambda: self.show_organization_page(token, user_email, user_id)
            print(f"[MAIN] Adding HomeScreen to stack...")
            sys.stdout.flush()
            self.stacked_widget.addWidget(self.home_page)
            self.stacked_widget.setCurrentWidget(self.home_page)
            print(f"[MAIN] HomeScreen displayed!")
            sys.stdout.flush()
        except Exception as e:
            print(f"[MAIN] ERROR in show_home_page: {e}")
            import traceback
            traceback.print_exc()

    def show_tournament_page(self, token, user_email, user_id):
        tournament_page = TournamentPage(token, user_email, user_id, on_home_click=lambda: self.show_organization_page(token, user_email, user_id), on_logout=self.handle_logout, on_docs_click=lambda: self.show_instruction_page(token, user_email, user_id))
        tournament_page.back_requested.connect(self.go_back)
        tournament_page.match_page_requested.connect(
            lambda tournament_id, tournament_name: self.show_match_page(
                token, user_email, user_id, tournament_id, tournament_name
            )
        )
        self.stacked_widget.addWidget(tournament_page)
        self.stacked_widget.setCurrentWidget(tournament_page)

    def show_match_page(self, token, user_email, user_id, tournament_id, tournament_name=""):
        match_page = MatchPage(token, user_email, user_id, tournament_id, tournament_name, on_home_click=lambda: self.show_organization_page(token, user_email, user_id), on_logout=self.handle_logout, on_docs_click=lambda: self.show_instruction_page(token, user_email, user_id))
        match_page.back_requested.connect(self.go_back)
        match_page.open_streaming_source_requested.connect(
            lambda match_info: self.show_streaming_source_page(
                match_info, token, user_email, user_id
            )
        )
        self.stacked_widget.addWidget(match_page)
        self.stacked_widget.setCurrentWidget(match_page)

    def show_streaming_source_page(self, match_info, token, user_email, user_id):
        streaming_page = StreamingSourcePage(match_info, user_email, self.handle_logout, on_home_click=lambda: self.show_organization_page(token, user_email, user_id), on_docs_click=lambda: self.show_instruction_page(token, user_email, user_id))
        streaming_page.back_requested.connect(self.go_back)
        streaming_page.selected.connect(self.handle_streaming_source_selected)
        self.stacked_widget.addWidget(streaming_page)
        self.stacked_widget.setCurrentWidget(streaming_page)

    def handle_streaming_source_selected(self, selection_data):
        """Handle when user selects a streaming source"""
        source = selection_data.get("source")
        match_info = selection_data.get("match")
        print(f"Selected streaming source: {source}")

        # Navigate to camera setup page using stored credentials
        self.show_camera_setup_page(
            match_info,
            self.current_token,
            self.current_user_email,
            self.current_user_id,
        )

    def show_camera_setup_page(self, match_info, token, user_email, user_id):
        """Show the camera setup page"""
        camera_setup_page = CameraSetupPyQt(
            match_info, user_email, self.handle_logout, token, on_docs_click=lambda: self.show_instruction_page(token, user_email, user_id)
        )
        camera_setup_page.back_requested.connect(self.go_back)
        self.stacked_widget.addWidget(camera_setup_page)
        self.stacked_widget.setCurrentWidget(camera_setup_page)

    def show_instruction_page(self, token, user_email, user_id):
        """Show the instruction/documentation page"""
        instruction_page = InstructionPage(
            user_email=user_email,
            on_logout=self.handle_logout,
            on_home_click=lambda: self.show_home_page(token, user_email, user_id)
        )
        instruction_page.back_requested.connect(self.go_back)
        self.stacked_widget.addWidget(instruction_page)
        self.stacked_widget.setCurrentWidget(instruction_page)

    def go_back(self):
        current_index = self.stacked_widget.currentIndex()
        if current_index > 0:
            # Get the widget to remove
            widget_to_remove = self.stacked_widget.widget(current_index)
            # Set the current widget to the previous one
            self.stacked_widget.setCurrentIndex(current_index - 1)
            # Remove the widget
            self.stacked_widget.removeWidget(widget_to_remove)
            widget_to_remove.deleteLater()

    def handle_logout(self):
        """Handle logout - go back to login page"""
        # Clear stored token from secure storage
        if self.current_user_email:
            SecureStorage.delete_token(self.current_user_email)
        
        # Clear API auth token
        user_api.clear_auth_token()
        
        # Clear stored credentials
        self.current_token = None
        self.current_user_email = None
        self.current_user_id = None
        
        # Clear all widgets except login page
        while self.stacked_widget.count() > 1:
            widget = self.stacked_widget.widget(1)
            self.stacked_widget.removeWidget(widget)
            widget.deleteLater()
        
        # Set current widget to login page
        self.stacked_widget.setCurrentIndex(0)
        
        # Force refresh of login page
        if hasattr(self, 'login_page'):
            self.login_page.username_input.clear()
            self.login_page.password_input.clear()

    def resizeEvent(self, event):
        """Handle window resize events for responsive behavior"""
        super().resizeEvent(event)
        
        # Trigger responsive updates when window is resized
        self.update_responsive_properties()
    
    def closeEvent(self, event):
        """Save window geometry before closing."""
        if not self._is_maximized and self._normal_geometry:
            geo = self._normal_geometry
            preferences.set_window_geometry(geo.x(), geo.y(), geo.width(), geo.height(), False)
        elif self._is_maximized:
            # Save maximized state
            preferences.set("window_maximized", True)
        super().closeEvent(event)
    
    def _restore_window_geometry(self):
        """Restore window geometry from preferences or show maximized."""
        saved_geo = preferences.get_window_geometry()
        was_maximized = preferences.get("window_maximized", True)
        
        if saved_geo and not was_maximized:
            # Restore saved geometry
            self.setGeometry(
                saved_geo.get("x", 100),
                saved_geo.get("y", 100),
                saved_geo.get("width", 1200),
                saved_geo.get("height", 800)
            )
            self._is_maximized = False
            self._normal_geometry = self.geometry()
            self.title_bar.set_maximized(False)
            self.show()
        else:
            # Default to maximized
            self.showMaximized()
            self._is_maximized = True
            self.title_bar.set_maximized(True)
    
    def update_responsive_properties(self):
        """Update responsive properties when window size changes"""
        # Update margins based on current window size
        current_width = self.width()
        
        # Adjust margins based on window width
        if current_width < self.responsive.get_responsive_width(80):
            # Smaller margins for smaller windows
            margins = self.responsive.calculate_margins(5)
        else:
            # Normal margins for larger windows
            margins = self.responsive.calculate_margins(10)
        
        self.layout.setContentsMargins(margins, 0, margins, 0)

    def showEvent(self, event):
        """Handle show event for initial responsive setup"""
        super().showEvent(event)
        self.update_responsive_properties()

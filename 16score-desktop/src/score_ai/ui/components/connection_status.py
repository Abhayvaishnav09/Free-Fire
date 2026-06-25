"""
Connection Status Indicator

Shows online/offline status in the header.
"""
from __future__ import annotations

import socket
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtCore import Qt, QTimer, pyqtSignal

from score_ai.core.theme import Colors, Fonts
from score_ai.core.config_manager import config


class StatusDot(QWidget):
    """Small colored dot indicating status."""
    
    def __init__(self, size: int = 8, parent=None):
        super().__init__(parent)
        self._size = size
        self._color = Colors.SUCCESS
        self.setFixedSize(size + 4, size + 4)
        self.setAttribute(Qt.WA_TranslucentBackground)
    
    def set_status(self, is_online: bool):
        """Update the status color."""
        self._color = Colors.SUCCESS if is_online else Colors.ERROR
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw outer glow
        glow_color = QColor(self._color)
        glow_color.setAlpha(80)
        painter.setBrush(glow_color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, self._size + 4, self._size + 4)
        
        # Draw inner dot
        painter.setBrush(QColor(self._color))
        painter.drawEllipse(2, 2, self._size, self._size)


class ConnectionStatus(QWidget):
    """
    Connection status indicator widget.
    
    Shows a colored dot and text indicating online/offline status.
    Automatically checks connectivity at regular intervals.
    """
    
    status_changed = pyqtSignal(bool)
    
    def __init__(self, check_interval: int = 30000, parent=None):
        """
        Initialize the connection status indicator.
        
        Args:
            check_interval: How often to check connectivity (ms)
            parent: Parent widget
        """
        super().__init__(parent)
        self._is_online = True
        self._check_interval = check_interval
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
        
        self._setup_ui()
        self._setup_timer()
        
        # Initial check
        QTimer.singleShot(1000, self._check_connection)
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)
        
        # Status dot
        self._dot = StatusDot(8)
        layout.addWidget(self._dot)
        
        # Status text
        self._label = QLabel("Online")
        self._label.setFont(Fonts.create(11))
        self._label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
        layout.addWidget(self._label)
    
    def _setup_timer(self):
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._check_connection)
        self._timer.start(self._check_interval)
    
    def _check_connection(self):
        """Check internet connectivity."""
        try:
            # Try to connect to the backend server
            backend_url = config.get("api.backend_url", "http://localhost:5006")
            host = backend_url.replace("http://", "").replace("https://", "").split(":")[0]
            port = 80
            
            # Extract port if present
            if ":" in backend_url.split("//")[-1]:
                port_str = backend_url.split(":")[-1].split("/")[0]
                try:
                    port = int(port_str)
                except ValueError:
                    pass
            
            # Quick socket check
            socket.setdefaulttimeout(3)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex((host, port))
            sock.close()
            
            is_online = (result == 0)
            
        except Exception:
            is_online = False
        
        self._update_status(is_online)
    
    def _update_status(self, is_online: bool):
        """Update the UI to reflect connection status."""
        if self._is_online != is_online:
            self._is_online = is_online
            self.status_changed.emit(is_online)
        
        self._dot.set_status(is_online)
        self._label.setText("Online" if is_online else "Offline")
        
        color = Colors.SUCCESS if is_online else Colors.ERROR
        self._label.setStyleSheet(f"color: {color}; background: transparent;")
    
    @property
    def is_online(self) -> bool:
        """Get current connection status."""
        return self._is_online
    
    def force_check(self):
        """Force an immediate connection check."""
        self._check_connection()



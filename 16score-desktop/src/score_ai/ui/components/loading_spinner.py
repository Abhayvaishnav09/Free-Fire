"""
Loading Spinner Component

Reusable loading indicator with animation.
"""
from __future__ import annotations

import math

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QTimer

from score_ai.core.theme import Colors, Fonts


class LoadingSpinner(QWidget):
    """Animated loading spinner with optional message."""
    
    def __init__(self, size: int = 40, message: str = "", parent=None):
        super().__init__(parent)
        self._angle = 0
        self._size = size
        self._message = message
        
        self.setFixedSize(size + 20, size + 60 if message else size + 20)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Animation timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._rotate)
        self._timer.start(16)
    
    def _rotate(self):
        self._angle = (self._angle + 6) % 360
        self.update()
    
    def set_message(self, message: str):
        """Update the loading message."""
        self._message = message
        self.setFixedSize(self._size + 20, self._size + 60 if message else self._size + 20)
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Center point
        cx = self.width() // 2
        cy = self._size // 2 + 10
        
        # Draw spinner arc
        pen = QPen(Colors.to_qcolor(Colors.PRIMARY_LIGHT, 200), 3)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        
        # Draw rotating arc
        rect_size = self._size - 6
        painter.translate(cx, cy)
        painter.rotate(self._angle)
        painter.translate(-cx, -cy)
        
        painter.drawArc(
            cx - rect_size // 2,
            cy - rect_size // 2,
            rect_size,
            rect_size,
            0 * 16,  # Start angle
            270 * 16  # Span angle (270 degrees)
        )
        
        # Draw message if provided
        if self._message:
            painter.resetTransform()
            painter.setPen(Colors.to_qcolor(Colors.TEXT_SECONDARY))
            painter.setFont(Fonts.create(12))
            painter.drawText(
                0, self._size + 25,
                self.width(), 30,
                Qt.AlignCenter,
                self._message
            )
    
    def start(self):
        """Start the spinner animation."""
        self._timer.start(16)
    
    def stop(self):
        """Stop the spinner animation."""
        self._timer.stop()


class LoadingOverlay(QWidget):
    """Full overlay with centered loading spinner."""
    
    def __init__(self, message: str = "Loading...", parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        
        # Spinner
        self.spinner = LoadingSpinner(50, message)
        layout.addWidget(self.spinner, alignment=Qt.AlignCenter)
    
    def set_message(self, message: str):
        """Update the loading message."""
        self.spinner.set_message(message)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Semi-transparent dark overlay
        painter.fillRect(self.rect(), Colors.to_qcolor(Colors.BG_DARK, 180))
        
        super().paintEvent(event)
    
    def show_loading(self):
        """Show the loading overlay."""
        self.spinner.start()
        self.show()
        self.raise_()
    
    def hide_loading(self):
        """Hide the loading overlay."""
        self.spinner.stop()
        self.hide()


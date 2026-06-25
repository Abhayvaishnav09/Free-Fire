"""
Toast Notification Component

Non-blocking notifications that appear briefly and fade away.
"""
from __future__ import annotations

from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout, QVBoxLayout, QGraphicsOpacityEffect
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtProperty

from score_ai.core.theme import Colors, Fonts


class ToastIcon(QWidget):
    """Custom painted icon for toast notification."""
    
    def __init__(self, toast_type: str = "info", parent=None):
        super().__init__(parent)
        self._toast_type = toast_type
        self.setFixedSize(20, 20)
        self.setAttribute(Qt.WA_TranslucentBackground)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        cx, cy = self.width() // 2, self.height() // 2
        
        # Color based on type
        colors = {
            "success": Colors.SUCCESS,
            "error": Colors.ERROR,
            "warning": Colors.WARNING,
            "info": Colors.INFO
        }
        color = colors.get(self._toast_type, Colors.INFO)
        
        pen = QPen(Colors.to_qcolor(color), 2)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        
        if self._toast_type == "success":
            # Checkmark
            painter.drawLine(cx - 5, cy, cx - 1, cy + 4)
            painter.drawLine(cx - 1, cy + 4, cx + 6, cy - 4)
            
        elif self._toast_type == "error":
            # X mark
            painter.drawLine(cx - 4, cy - 4, cx + 4, cy + 4)
            painter.drawLine(cx + 4, cy - 4, cx - 4, cy + 4)
            
        elif self._toast_type == "warning":
            # Triangle with !
            painter.drawLine(cx, cy - 6, cx - 6, cy + 4)
            painter.drawLine(cx - 6, cy + 4, cx + 6, cy + 4)
            painter.drawLine(cx + 6, cy + 4, cx, cy - 6)
            painter.drawLine(cx, cy - 2, cx, cy + 1)
            painter.drawPoint(cx, cy + 3)
            
        else:  # info
            # Circle with i
            painter.drawEllipse(cx - 6, cy - 6, 12, 12)
            painter.drawLine(cx, cy - 2, cx, cy + 3)
            painter.drawPoint(cx, cy - 4)


class Toast(QWidget):
    """
    Toast notification widget.
    
    Usage:
        toast = Toast("Operation successful!", "success")
        toast.show_at(parent_widget)
    """
    
    def __init__(
        self,
        message: str,
        toast_type: str = "info",
        duration: int = 3000,
        parent=None
    ):
        super().__init__(parent)
        self._message = message
        self._toast_type = toast_type
        self._duration = duration
        
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self._setup_ui()
        self._setup_animation()
    
    def _setup_ui(self):
        """Setup the toast UI."""
        # Colors based on type
        bg_colors = {
            "success": Colors.SUCCESS,
            "error": Colors.ERROR,
            "warning": Colors.WARNING,
            "info": Colors.INFO
        }
        color = bg_colors.get(self._toast_type, Colors.INFO)
        
        self.setStyleSheet(f"""
            QWidget {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {Colors.CARD}, stop:1 {Colors.CARD_SECONDARY});
                border: 1px solid {color};
                border-radius: 8px;
                border-left: 4px solid {color};
            }}
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)
        
        # Icon
        icon = ToastIcon(self._toast_type)
        layout.addWidget(icon)
        
        # Message
        label = QLabel(self._message)
        label.setFont(Fonts.create(13))
        label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent; border: none;")
        label.setWordWrap(True)
        layout.addWidget(label, 1)
        
        self.setMinimumWidth(280)
        self.setMaximumWidth(400)
    
    def _setup_animation(self):
        """Setup fade animations."""
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity_effect)
        
        # Fade in animation
        self._fade_in = QPropertyAnimation(self._opacity_effect, b"opacity")
        self._fade_in.setDuration(200)
        self._fade_in.setStartValue(0.0)
        self._fade_in.setEndValue(1.0)
        
        # Fade out animation
        self._fade_out = QPropertyAnimation(self._opacity_effect, b"opacity")
        self._fade_out.setDuration(300)
        self._fade_out.setStartValue(1.0)
        self._fade_out.setEndValue(0.0)
        self._fade_out.finished.connect(self.deleteLater)
        
        # Timer to trigger fade out
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._fade_out.start)
    
    def show_at(self, parent: QWidget, position: str = "top-right"):
        """Show toast at specified position relative to parent."""
        self.adjustSize()
        
        # Calculate position
        parent_rect = parent.rect()
        parent_global = parent.mapToGlobal(parent_rect.topLeft())
        
        margin = 20
        
        if position == "top-right":
            x = parent_global.x() + parent_rect.width() - self.width() - margin
            y = parent_global.y() + margin + 50  # Below title bar
        elif position == "top-center":
            x = parent_global.x() + (parent_rect.width() - self.width()) // 2
            y = parent_global.y() + margin + 50
        elif position == "bottom-right":
            x = parent_global.x() + parent_rect.width() - self.width() - margin
            y = parent_global.y() + parent_rect.height() - self.height() - margin
        else:  # bottom-center
            x = parent_global.x() + (parent_rect.width() - self.width()) // 2
            y = parent_global.y() + parent_rect.height() - self.height() - margin
        
        self.move(x, y)
        self.show()
        self._fade_in.start()
        self._timer.start(self._duration)


class ToastManager:
    """
    Singleton manager for toast notifications.
    
    Usage:
        ToastManager.show("Success!", "success", parent_widget)
        ToastManager.error("Something went wrong", parent_widget)
        ToastManager.success("Saved!", parent_widget)
    """
    
    _instance = None
    _toasts: list[Toast] = []
    
    @classmethod
    def show(
        cls,
        message: str,
        toast_type: str = "info",
        parent: QWidget = None,
        duration: int = 3000,
        position: str = "top-right"
    ):
        """Show a toast notification."""
        if parent is None:
            return
        
        toast = Toast(message, toast_type, duration, parent)
        cls._toasts.append(toast)
        
        # Clean up old toasts
        cls._toasts = [t for t in cls._toasts if t.isVisible()]
        
        # Offset multiple toasts
        offset = 0
        for existing in cls._toasts[:-1]:
            if existing.isVisible():
                offset += existing.height() + 10
        
        toast.show_at(parent, position)
        
        # Move down if there are existing toasts
        if offset > 0:
            toast.move(toast.x(), toast.y() + offset)
    
    @classmethod
    def success(cls, message: str, parent: QWidget, **kwargs):
        """Show a success toast."""
        cls.show(message, "success", parent, **kwargs)
    
    @classmethod
    def error(cls, message: str, parent: QWidget, **kwargs):
        """Show an error toast."""
        cls.show(message, "error", parent, **kwargs)
    
    @classmethod
    def warning(cls, message: str, parent: QWidget, **kwargs):
        """Show a warning toast."""
        cls.show(message, "warning", parent, **kwargs)
    
    @classmethod
    def info(cls, message: str, parent: QWidget, **kwargs):
        """Show an info toast."""
        cls.show(message, "info", parent, **kwargs)


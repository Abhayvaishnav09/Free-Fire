"""
Empty State Component

Reusable empty/no-data state with icon and action button.
"""
from __future__ import annotations

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtCore import Qt, pyqtSignal

from score_ai.core.theme import Colors, Fonts, Styles


class EmptyStateIcon(QWidget):
    """Custom painted icon for empty state."""
    
    def __init__(self, icon_type: str = "empty", size: int = 64, parent=None):
        super().__init__(parent)
        self._icon_type = icon_type
        self._size = size
        self.setFixedSize(size, size)
        self.setAttribute(Qt.WA_TranslucentBackground)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        cx, cy = self.width() // 2, self.height() // 2
        pen = QPen(Colors.to_qcolor(Colors.TEXT_MUTED), 2)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        
        if self._icon_type == "no_data":
            # Draw folder/box icon
            s = self._size // 2 - 4
            # Box outline
            painter.drawRect(cx - s, cy - s + 5, s * 2, s * 2 - 5)
            # Folder flap
            painter.drawLine(cx - s, cy - s + 5, cx - s, cy - s - 2)
            painter.drawLine(cx - s, cy - s - 2, cx - 5, cy - s - 2)
            painter.drawLine(cx - 5, cy - s - 2, cx - 3, cy - s + 5)
            
        elif self._icon_type == "no_matches":
            # Draw gamepad/controller icon
            s = self._size // 3
            # Controller body (rounded)
            painter.drawRoundedRect(cx - s - 5, cy - 8, s * 2 + 10, 20, 6, 6)
            # D-pad left
            painter.drawLine(cx - s + 2, cy, cx - s + 8, cy)
            painter.drawLine(cx - s + 5, cy - 3, cx - s + 5, cy + 3)
            # Buttons right
            painter.drawEllipse(cx + s - 8, cy - 3, 6, 6)
            
        elif self._icon_type == "no_tournaments":
            # Draw trophy icon
            s = self._size // 3
            # Cup body
            painter.drawArc(cx - s, cy - s, s * 2, s + 10, 0, 180 * 16)
            # Handles
            painter.drawArc(cx - s - 8, cy - s + 3, 12, 15, 90 * 16, 180 * 16)
            painter.drawArc(cx + s - 4, cy - s + 3, 12, 15, -90 * 16, 180 * 16)
            # Base
            painter.drawLine(cx - 5, cy + 10, cx + 5, cy + 10)
            painter.drawLine(cx, cy + 2, cx, cy + 10)
            
        elif self._icon_type == "error":
            # Draw warning triangle
            s = self._size // 2 - 8
            # Triangle
            painter.drawLine(cx, cy - s, cx - s, cy + s - 5)
            painter.drawLine(cx - s, cy + s - 5, cx + s, cy + s - 5)
            painter.drawLine(cx + s, cy + s - 5, cx, cy - s)
            # Exclamation mark
            painter.drawLine(cx, cy - s + 12, cx, cy + 5)
            painter.drawPoint(cx, cy + s - 10)
            
        else:  # Default empty
            # Draw empty document
            s = self._size // 3
            painter.drawRect(cx - s, cy - s - 5, s * 2, s * 2 + 10)
            # Lines
            painter.drawLine(cx - s + 5, cy - 5, cx + s - 5, cy - 5)
            painter.drawLine(cx - s + 5, cy + 2, cx + s - 10, cy + 2)


class EmptyState(QWidget):
    """
    Empty state widget with icon, title, message, and optional action button.
    
    Usage:
        empty = EmptyState(
            icon_type="no_matches",
            title="No Matches Found",
            message="There are no matches scheduled for this tournament.",
            action_text="Refresh",
            action_callback=self.refresh_matches
        )
    """
    
    action_clicked = pyqtSignal()
    
    def __init__(
        self,
        icon_type: str = "empty",
        title: str = "No Data",
        message: str = "",
        action_text: str = "",
        action_callback=None,
        parent=None
    ):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(16)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Icon
        icon = EmptyStateIcon(icon_type, 72)
        layout.addWidget(icon, alignment=Qt.AlignCenter)
        
        # Title
        title_label = QLabel(title)
        title_label.setFont(Fonts.create(18, Fonts.WEIGHT_DEMIBOLD))
        title_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Message
        if message:
            msg_label = QLabel(message)
            msg_label.setFont(Fonts.create(13))
            msg_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; background: transparent;")
            msg_label.setAlignment(Qt.AlignCenter)
            msg_label.setWordWrap(True)
            msg_label.setMaximumWidth(300)
            layout.addWidget(msg_label)
        
        # Action button
        if action_text:
            action_btn = QPushButton(action_text)
            action_btn.setFont(Fonts.create(13, Fonts.WEIGHT_DEMIBOLD))
            action_btn.setCursor(Qt.PointingHandCursor)
            action_btn.setStyleSheet(Styles.secondary_button())
            action_btn.setFixedWidth(140)
            
            if action_callback:
                action_btn.clicked.connect(action_callback)
            action_btn.clicked.connect(self.action_clicked.emit)
            
            layout.addSpacing(8)
            layout.addWidget(action_btn, alignment=Qt.AlignCenter)


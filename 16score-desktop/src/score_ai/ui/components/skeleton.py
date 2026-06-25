"""
Skeleton Loading Components

Shimmer effect placeholders shown while content is loading.
"""
from __future__ import annotations

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFrame
from PyQt5.QtGui import QPainter, QColor, QLinearGradient, QPainterPath
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QRectF, pyqtProperty

from score_ai.core.theme import Colors


class SkeletonBase(QWidget):
    """
    Base skeleton widget with shimmer animation.
    
    Subclass and override paintEvent to create different shapes.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self._shimmer_pos = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate)
        self._timer.start(30)
    
    def _animate(self):
        self._shimmer_pos += 0.02
        if self._shimmer_pos > 1.5:
            self._shimmer_pos = -0.5
        self.update()
    
    def _get_shimmer_gradient(self, rect: QRectF) -> QLinearGradient:
        """Get the shimmer gradient for the given rect."""
        gradient = QLinearGradient(
            rect.left() + rect.width() * (self._shimmer_pos - 0.3),
            0,
            rect.left() + rect.width() * (self._shimmer_pos + 0.3),
            0
        )
        
        base_color = QColor(Colors.CARD_SECONDARY)
        highlight_color = QColor(Colors.SURFACE_ALT)
        
        gradient.setColorAt(0.0, base_color)
        gradient.setColorAt(0.5, highlight_color)
        gradient.setColorAt(1.0, base_color)
        
        return gradient
    
    def stop(self):
        """Stop the shimmer animation."""
        self._timer.stop()
    
    def start(self):
        """Start the shimmer animation."""
        self._timer.start(30)


class SkeletonRect(SkeletonBase):
    """Rectangular skeleton placeholder."""
    
    def __init__(self, width: int = 100, height: int = 20, radius: int = 4, parent=None):
        super().__init__(parent)
        self._radius = radius
        self.setFixedSize(width, height)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = QRectF(0, 0, self.width(), self.height())
        gradient = self._get_shimmer_gradient(rect)
        
        path = QPainterPath()
        path.addRoundedRect(rect, self._radius, self._radius)
        
        painter.fillPath(path, gradient)


class SkeletonCircle(SkeletonBase):
    """Circular skeleton placeholder."""
    
    def __init__(self, size: int = 40, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = QRectF(0, 0, self.width(), self.height())
        gradient = self._get_shimmer_gradient(rect)
        
        path = QPainterPath()
        path.addEllipse(rect)
        
        painter.fillPath(path, gradient)


class SkeletonText(QWidget):
    """Multi-line text skeleton placeholder."""
    
    def __init__(self, lines: int = 3, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Vary line widths for natural look
        widths = [100, 85, 70, 90, 75]
        
        for i in range(lines):
            width_percent = widths[i % len(widths)]
            line = SkeletonRect(width=200, height=14, radius=3)
            line.setMaximumWidth(int(200 * width_percent / 100))
            layout.addWidget(line)
    
    def stop(self):
        """Stop all skeleton animations."""
        for child in self.findChildren(SkeletonBase):
            child.stop()


class SkeletonCard(QFrame):
    """
    Card-shaped skeleton placeholder matching tournament/match card layout.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background: {Colors.CARD};
                border: 1px solid {Colors.BORDER};
                border-radius: 12px;
            }}
        """)
        self.setMinimumWidth(340)
        self.setMaximumWidth(450)
        self.setFixedHeight(180)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)
        
        # Top row: Status badge
        self.status = SkeletonRect(80, 24, 12)
        layout.addWidget(self.status)
        
        # Title
        self.title = SkeletonRect(250, 20, 4)
        layout.addWidget(self.title)
        
        # Subtitle
        self.subtitle = SkeletonRect(180, 16, 4)
        layout.addWidget(self.subtitle)
        
        layout.addStretch()
        
        # Bottom row: Info items
        bottom = QHBoxLayout()
        bottom.setSpacing(16)
        
        for _ in range(3):
            item = QVBoxLayout()
            item.setSpacing(4)
            item.addWidget(SkeletonRect(60, 12, 3))
            item.addWidget(SkeletonRect(40, 14, 3))
            bottom.addLayout(item)
        
        bottom.addStretch()
        layout.addLayout(bottom)
    
    def stop(self):
        """Stop all skeleton animations."""
        for child in self.findChildren(SkeletonBase):
            child.stop()


class SkeletonTournamentCard(SkeletonCard):
    """Tournament card skeleton matching GlowingTournamentCard layout."""
    pass


class SkeletonMatchCard(SkeletonCard):
    """Match card skeleton matching GlowingMatchCard layout."""
    pass


def create_skeleton_grid(count: int = 6, card_class=SkeletonCard) -> QWidget:
    """
    Create a grid of skeleton cards.
    
    Args:
        count: Number of skeleton cards to create
        card_class: The skeleton card class to use
    
    Returns:
        Widget containing the skeleton grid
    """
    container = QWidget()
    container.setStyleSheet("background: transparent;")
    
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(16)
    
    for _ in range(count):
        card = card_class()
        layout.addWidget(card)
    
    layout.addStretch()
    
    return container



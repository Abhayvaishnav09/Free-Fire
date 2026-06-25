"""Reusable glowing card components"""
from __future__ import annotations

from PyQt5.QtWidgets import QFrame
from PyQt5.QtGui import QPainter, QBrush, QPen, QLinearGradient
from PyQt5.QtCore import Qt, QTimer, QPoint, QPropertyAnimation, QEasingCurve

from score_ai.core.theme import Colors


class GlowingCard(QFrame):
    """AI-themed card with glowing border - reusable component"""
    
    def __init__(self, radius: int = 20, glow_size: int = 30, parent=None):
        super().__init__(parent)
        self._radius = radius
        self._glow_size = glow_size
        self._glow_intensity = 0.5
        self._pulse_direction = 1
        self._original_pos = None
        
        # Pulse animation
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._pulse_glow)
        self._pulse_timer.start(50)
    
    def _pulse_glow(self):
        self._glow_intensity += 0.02 * self._pulse_direction
        if self._glow_intensity >= 1.0:
            self._pulse_direction = -1
        elif self._glow_intensity <= 0.3:
            self._pulse_direction = 1
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect()
        radius = self._radius
        
        # Outer glow
        for i in range(self._glow_size, 0, -3):
            alpha = int(12 * self._glow_intensity * (1 - i / self._glow_size))
            painter.setPen(Qt.NoPen)
            painter.setBrush(Colors.to_qcolor(Colors.PRIMARY, alpha))
            painter.drawRoundedRect(rect.adjusted(-i, -i, i, i), radius + i//2, radius + i//2)
        
        # Background
        gradient = QLinearGradient(0, 0, 0, rect.height())
        gradient.setColorAt(0.0, Colors.to_qcolor(Colors.CARD, 245))
        gradient.setColorAt(0.5, Colors.to_qcolor(Colors.BG_ALT, 235))
        gradient.setColorAt(1.0, Colors.to_qcolor(Colors.CARD, 250))
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect, radius, radius)
        
        # Inner overlay
        inner = QLinearGradient(0, 0, rect.width(), 0)
        inner.setColorAt(0.0, Colors.to_qcolor(Colors.PRIMARY, 6))
        inner.setColorAt(0.5, Colors.to_qcolor(Colors.PRIMARY, 0))
        inner.setColorAt(1.0, Colors.to_qcolor(Colors.PRIMARY_ALT, 6))
        painter.setBrush(QBrush(inner))
        painter.drawRoundedRect(rect, radius, radius)
        
        # Glowing border
        border = QLinearGradient(0, 0, rect.width(), rect.height())
        alpha = int(80 + 60 * self._glow_intensity)
        border.setColorAt(0.0, Colors.to_qcolor(Colors.PRIMARY, alpha))
        border.setColorAt(0.5, Colors.to_qcolor(Colors.PRIMARY_LIGHT, int(alpha * 0.6)))
        border.setColorAt(1.0, Colors.to_qcolor(Colors.PRIMARY_ALT, alpha))
        
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QBrush(border), 2))
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), radius - 1, radius - 1)
        
        # Top highlight
        highlight = QLinearGradient(rect.width() * 0.2, 0, rect.width() * 0.8, 0)
        highlight.setColorAt(0.0, Colors.to_qcolor(Colors.TEXT_PRIMARY, 0))
        highlight.setColorAt(0.5, Colors.to_qcolor(Colors.TEXT_PRIMARY, 30))
        highlight.setColorAt(1.0, Colors.to_qcolor(Colors.TEXT_PRIMARY, 0))
        
        painter.setPen(QPen(QBrush(highlight), 1))
        painter.drawLine(int(rect.width() * 0.2), 1, int(rect.width() * 0.8), 1)
        
        # Corner dots
        painter.setPen(Qt.NoPen)
        accent_alpha = int(50 * self._glow_intensity)
        painter.setBrush(Colors.to_qcolor(Colors.PRIMARY_LIGHT, accent_alpha))
        painter.drawEllipse(10, 10, 4, 4)
        painter.drawEllipse(rect.width() - 14, 10, 4, 4)
        painter.drawEllipse(10, rect.height() - 14, 4, 4)
        painter.drawEllipse(rect.width() - 14, rect.height() - 14, 4, 4)
    
    def shake(self):
        """Shake animation for error feedback"""
        if self._original_pos is None:
            self._original_pos = self.pos()
        
        anim = QPropertyAnimation(self, b"pos")
        anim.setDuration(500)
        anim.setEasingCurve(QEasingCurve.OutElastic)
        
        start = self._original_pos
        anim.setKeyValueAt(0, start)
        anim.setKeyValueAt(0.1, start + QPoint(10, 0))
        anim.setKeyValueAt(0.2, start + QPoint(-10, 0))
        anim.setKeyValueAt(0.3, start + QPoint(8, 0))
        anim.setKeyValueAt(0.4, start + QPoint(-8, 0))
        anim.setKeyValueAt(0.5, start + QPoint(5, 0))
        anim.setKeyValueAt(0.6, start + QPoint(-5, 0))
        anim.setKeyValueAt(0.7, start + QPoint(2, 0))
        anim.setKeyValueAt(0.8, start + QPoint(-2, 0))
        anim.setKeyValueAt(1.0, start)
        
        anim.start()
        self._shake_anim = anim


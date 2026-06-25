"""Reusable AI-themed background components"""
from __future__ import annotations

from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QBrush, QPen, QLinearGradient, QRadialGradient
from PyQt5.QtCore import Qt, QTimer

from score_ai.core.theme import Colors


class AIBackground(QWidget):
    """AI neural network background - reusable component"""
    
    def __init__(self, node_count: int = 15, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._pulse = 0
        self._init_nodes(node_count)
        
        # Animation timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate)
        self._timer.start(50)
    
    def _init_nodes(self, count: int):
        """Initialize neural network nodes"""
        import random
        random.seed(42)
        self.nodes = []
        for _ in range(count):
            self.nodes.append({
                'x': random.uniform(0.05, 0.95),
                'y': random.uniform(0.1, 0.9),
                'size': random.uniform(2, 6),
                'pulse_offset': random.uniform(0, 6.28)
            })
        
        # Generate connections
        self.connections = []
        for i, n1 in enumerate(self.nodes):
            for j, n2 in enumerate(self.nodes):
                if i < j:
                    dist = ((n1['x'] - n2['x'])**2 + (n1['y'] - n2['y'])**2)**0.5
                    if dist < 0.35:
                        self.connections.append((i, j, dist))
    
    def _animate(self):
        self._pulse = (self._pulse + 0.08) % 6.28
        self.update()
    
    def paintEvent(self, event):
        import math
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect()
        w, h = rect.width(), rect.height()
        
        # Dark gradient background
        gradient = QLinearGradient(0, 0, w, h)
        gradient.setColorAt(0.0, Colors.to_qcolor(Colors.BG_DARK))
        gradient.setColorAt(0.3, Colors.to_qcolor(Colors.BG))
        gradient.setColorAt(0.6, Colors.to_qcolor(Colors.BG_ALT))
        gradient.setColorAt(1.0, Colors.to_qcolor(Colors.BG_DARK))
        painter.fillRect(rect, QBrush(gradient))
        
        # Subtle grid
        painter.setPen(QPen(Colors.to_qcolor(Colors.BORDER, 15), 1))
        grid_size = 50
        for x in range(0, w, grid_size):
            painter.drawLine(x, 0, x, h)
        for y in range(0, h, grid_size):
            painter.drawLine(0, y, w, y)
        
        # Draw connections
        for i, j, dist in self.connections:
            n1, n2 = self.nodes[i], self.nodes[j]
            alpha = int(35 * (1 - dist / 0.35))
            pulse_factor = 0.5 + 0.5 * math.sin(self._pulse + n1['pulse_offset'])
            alpha = int(alpha * (0.5 + 0.5 * pulse_factor))
            
            painter.setPen(QPen(Colors.to_qcolor(Colors.PRIMARY_LIGHT, alpha), 1))
            painter.drawLine(int(n1['x'] * w), int(n1['y'] * h), int(n2['x'] * w), int(n2['y'] * h))
        
        # Draw nodes
        for node in self.nodes:
            x, y = int(node['x'] * w), int(node['y'] * h)
            size = node['size']
            pulse_factor = 0.6 + 0.4 * math.sin(self._pulse + node['pulse_offset'])
            
            # Glow
            glow = QRadialGradient(x, y, size * 3)
            glow.setColorAt(0.0, Colors.to_qcolor(Colors.PRIMARY, int(25 * pulse_factor)))
            glow.setColorAt(1.0, Colors.to_qcolor(Colors.PRIMARY, 0))
            painter.setBrush(QBrush(glow))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(int(x - size * 3), int(y - size * 3), int(size * 6), int(size * 6))
            
            # Core
            painter.setBrush(Colors.to_qcolor(Colors.PRIMARY_LIGHT, int(160 * pulse_factor)))
            painter.drawEllipse(int(x - size/2), int(y - size/2), int(size), int(size))
        
        # Accent glows
        painter.setCompositionMode(QPainter.CompositionMode_Plus)
        
        glow1 = QRadialGradient(w * 0.75, h * 0.25, w * 0.4)
        glow1.setColorAt(0.0, Colors.to_qcolor(Colors.PRIMARY, 30))
        glow1.setColorAt(0.5, Colors.to_qcolor(Colors.PRIMARY, 10))
        glow1.setColorAt(1.0, Colors.to_qcolor(Colors.PRIMARY, 0))
        painter.fillRect(rect, QBrush(glow1))
        
        glow2 = QRadialGradient(w * 0.25, h * 0.75, w * 0.35)
        glow2.setColorAt(0.0, Colors.to_qcolor(Colors.PRIMARY_ALT, 22))
        glow2.setColorAt(0.5, Colors.to_qcolor(Colors.PRIMARY_ALT, 7))
        glow2.setColorAt(1.0, Colors.to_qcolor(Colors.PRIMARY_ALT, 0))
        painter.fillRect(rect, QBrush(glow2))


"""Modern custom title bar with window controls"""
from __future__ import annotations

from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpacerItem,
    QSizePolicy,
    QGraphicsDropShadowEffect,
)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QSize
from PyQt5.QtGui import QFont, QColor, QCursor, QPixmap, QPainter, QPainterPath, QLinearGradient, QBrush
import os

from score_ai.core.theme import Colors, Fonts, Spacing, Gradients


class TitleBarButton(QPushButton):
    """Stylish window control button"""
    
    def __init__(self, button_type: str, parent=None):
        super().__init__(parent)
        self.button_type = button_type
        self.setFixedSize(46, 32)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self._is_hovered = False
        self.update_style()
        
    def update_style(self):
        if self.button_type == "close":
            if self._is_hovered:
                self.setStyleSheet("""
                    QPushButton {
                        background: #ef4444;
                        border: none;
                        border-radius: 0px;
                    }
                """)
            else:
                self.setStyleSheet("""
                    QPushButton {
                        background: transparent;
                        border: none;
                    }
                """)
        else:
            if self._is_hovered:
                self.setStyleSheet("""
                    QPushButton {
                        background: rgba(255, 255, 255, 0.1);
                        border: none;
                        border-radius: 0px;
                    }
                """)
            else:
                self.setStyleSheet("""
                    QPushButton {
                        background: transparent;
                        border: none;
                    }
                """)
                
    def enterEvent(self, event):
        self._is_hovered = True
        self.update_style()
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        self._is_hovered = False
        self.update_style()
        super().leaveEvent(event)
        
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Icon color - white with opacity for dark background
        if self.button_type == "close" and self._is_hovered:
            color = QColor(*Colors.TEXT_PRIMARY_RGB)
        else:
            color = QColor(*Colors.TEXT_SECONDARY_RGB)
            
        painter.setPen(color)
        painter.setBrush(Qt.NoBrush)
        
        # Center of button
        cx, cy = self.width() // 2, self.height() // 2
        
        if self.button_type == "close":
            # Draw X
            painter.setPen(color)
            pen = painter.pen()
            pen.setWidth(2)
            painter.setPen(pen)
            size = 5
            painter.drawLine(cx - size, cy - size, cx + size, cy + size)
            painter.drawLine(cx - size, cy + size, cx + size, cy - size)
            
        elif self.button_type == "maximize":
            # Draw square
            painter.setPen(color)
            pen = painter.pen()
            pen.setWidth(2)
            painter.setPen(pen)
            size = 5
            painter.drawRect(cx - size, cy - size, size * 2, size * 2)
            
        elif self.button_type == "restore":
            # Draw overlapping squares for restore
            painter.setPen(color)
            pen = painter.pen()
            pen.setWidth(1)
            painter.setPen(pen)
            size = 4
            # Back square
            painter.drawRect(cx - size + 2, cy - size - 1, size * 2 - 2, size * 2 - 2)
            # Front square
            painter.fillRect(cx - size - 1, cy - size + 2, size * 2 - 1, size * 2 - 1, 
                           self.palette().window().color() if not self._is_hovered else QColor(0, 0, 0, 25))
            painter.drawRect(cx - size - 1, cy - size + 2, size * 2 - 2, size * 2 - 2)
            
        elif self.button_type == "minimize":
            # Draw horizontal line
            painter.setPen(color)
            pen = painter.pen()
            pen.setWidth(2)
            painter.setPen(pen)
            size = 5
            painter.drawLine(cx - size, cy, cx + size, cy)


class CustomTitleBar(QWidget):
    """Modern custom title bar"""
    
    minimize_clicked = pyqtSignal()
    maximize_clicked = pyqtSignal()
    close_clicked = pyqtSignal()
    
    def __init__(self, parent=None, title: str = "16Score-AI"):
        super().__init__(parent)
        self.parent_window = parent
        self.title = title
        self._is_maximized = False
        self._drag_pos = None
        self.setFixedHeight(40)
        self.init_ui()
        
    def init_ui(self):
        # Set background explicitly using palette and auto-fill
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), Colors.to_qcolor(Colors.BG_DARK))
        self.setPalette(palette)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(Spacing.MD, 0, 0, 0)
        layout.setSpacing(0)
        
        # App icon - 16Score logo
        icon_label = QLabel()
        icon_label.setFixedSize(28, 28)
        
        # Try to load the logo image
        logo_paths = [
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "images", "16score_logo.png"),
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "images", "16score_logo.png"),
            "images/16score_logo.png"
        ]
        
        logo_loaded = False
        for logo_path in logo_paths:
            if os.path.exists(logo_path):
                pixmap = QPixmap(logo_path)
                if not pixmap.isNull():
                    icon_label.setPixmap(pixmap.scaled(28, 28, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    logo_loaded = True
                    break
        
        if not logo_loaded:
            # Fallback to text if image not found
            icon_label.setText("16")
            icon_label.setFont(QFont(Fonts.FAMILY, 9, Fonts.WEIGHT_BOLD))
            icon_label.setAlignment(Qt.AlignCenter)
            icon_label.setStyleSheet(f"""
                background: {Gradients.primary_diagonal()};
                color: white;
                border-radius: 6px;
            """)
        else:
            icon_label.setStyleSheet("background: transparent;")
        
        layout.addWidget(icon_label)
        layout.addSpacing(Spacing.SM)
        
        # Title - white 70% opacity for secondary text
        self.title_label = QLabel(self.title)
        self.title_label.setFont(QFont(Fonts.FAMILY, Fonts.SIZE_XS, Fonts.WEIGHT_DEMIBOLD))
        self.title_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent;")
        layout.addWidget(self.title_label)
        
        # Spacer
        layout.addStretch()
        
        # Window controls
        controls_widget = QWidget()
        controls_widget.setStyleSheet("background: transparent;")
        controls_layout = QHBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(0)
        
        # Minimize button
        self.minimize_btn = TitleBarButton("minimize")
        self.minimize_btn.clicked.connect(self.minimize_clicked.emit)
        self.minimize_btn.setToolTip("Minimize")
        controls_layout.addWidget(self.minimize_btn)
        
        # Maximize button
        self.maximize_btn = TitleBarButton("maximize")
        self.maximize_btn.clicked.connect(self.toggle_maximize)
        self.maximize_btn.setToolTip("Maximize")
        controls_layout.addWidget(self.maximize_btn)
        
        # Close button
        self.close_btn = TitleBarButton("close")
        self.close_btn.clicked.connect(self.close_clicked.emit)
        self.close_btn.setToolTip("Close")
        controls_layout.addWidget(self.close_btn)
        
        layout.addWidget(controls_widget)
    
    def paintEvent(self, event):
        """Paint the dark background"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw gradient background using theme colors
        gradient = QLinearGradient(0, 0, self.width(), 0)
        gradient.setColorAt(0.0, Colors.to_qcolor(Colors.BG_DARK))
        gradient.setColorAt(0.5, Colors.to_qcolor(Colors.BG))
        gradient.setColorAt(1.0, Colors.to_qcolor(Colors.BG_DARK))
        
        painter.fillRect(self.rect(), QBrush(gradient))
        
        # Draw bottom border
        painter.setPen(Colors.to_qcolor(Colors.BORDER))
        painter.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
        
    def toggle_maximize(self):
        self._is_maximized = not self._is_maximized
        if self._is_maximized:
            self.maximize_btn.button_type = "restore"
        else:
            self.maximize_btn.button_type = "maximize"
        self.maximize_btn.update()
        self.maximize_clicked.emit()
        
    def set_maximized(self, maximized: bool):
        self._is_maximized = maximized
        if self._is_maximized:
            self.maximize_btn.button_type = "restore"
            self.maximize_btn.setToolTip("Restore")
        else:
            self.maximize_btn.button_type = "maximize"
            self.maximize_btn.setToolTip("Maximize")
        self.maximize_btn.update()
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.parent_window.frameGeometry().topLeft()
            event.accept()
            
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self._drag_pos is not None:
            # If maximized, restore first
            if self._is_maximized:
                self.maximize_clicked.emit()
                # Adjust drag position to keep mouse relative position
                self._drag_pos = QPoint(self.parent_window.width() // 2, 20)
            self.parent_window.move(event.globalPos() - self._drag_pos)
            event.accept()
            
    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.toggle_maximize()
            event.accept()
            
    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        super().mouseReleaseEvent(event)


class FramelessWindow(QWidget):
    """Base class for frameless window with resize support"""
    
    RESIZE_MARGIN = 8
    
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self._resize_direction = None
        self._resize_start_pos = None
        self._resize_start_geometry = None
        self.setMouseTracking(True)
        
    def _get_resize_direction(self, pos):
        """Determine resize direction based on cursor position"""
        rect = self.rect()
        x, y = pos.x(), pos.y()
        w, h = rect.width(), rect.height()
        margin = self.RESIZE_MARGIN
        
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
            self.setCursor(Qt.ArrowCursor)
            
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
        
        new_geo = geo
        
        if "left" in self._resize_direction:
            new_x = geo.x() + diff.x()
            new_w = geo.width() - diff.x()
            if new_w >= min_w:
                new_geo = new_geo.adjusted(diff.x(), 0, 0, 0)
                
        if "right" in self._resize_direction:
            new_w = geo.width() + diff.x()
            if new_w >= min_w:
                new_geo.setWidth(new_w)
                
        if "top" in self._resize_direction:
            new_y = geo.y() + diff.y()
            new_h = geo.height() - diff.y()
            if new_h >= min_h:
                new_geo = new_geo.adjusted(0, diff.y(), 0, 0)
                
        if "bottom" in self._resize_direction:
            new_h = geo.height() + diff.y()
            if new_h >= min_h:
                new_geo.setHeight(new_h)
                
        self.setGeometry(new_geo)


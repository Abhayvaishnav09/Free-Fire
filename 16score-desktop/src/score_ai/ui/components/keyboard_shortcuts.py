"""
Keyboard Shortcuts Manager

Centralized keyboard shortcut handling for the application.
"""
from __future__ import annotations

from PyQt5.QtWidgets import QWidget, QShortcut, QApplication
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt, pyqtSignal, QObject


class KeyboardShortcuts(QObject):
    """
    Manages keyboard shortcuts for a widget.
    
    Usage:
        shortcuts = KeyboardShortcuts(self)
        shortcuts.register("Ctrl+R", self.refresh)
        shortcuts.register("Escape", self.go_back)
    """
    
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._parent = parent
        self._shortcuts: dict[str, QShortcut] = {}
    
    def register(self, key_sequence: str, callback, context=Qt.WidgetWithChildrenShortcut) -> QShortcut:
        """
        Register a keyboard shortcut.
        
        Args:
            key_sequence: Key combination (e.g., "Ctrl+R", "Escape", "Enter")
            callback: Function to call when shortcut is triggered
            context: Shortcut context (default: widget and children)
        
        Returns:
            The created QShortcut object
        """
        shortcut = QShortcut(QKeySequence(key_sequence), self._parent)
        shortcut.setContext(context)
        shortcut.activated.connect(callback)
        self._shortcuts[key_sequence] = shortcut
        return shortcut
    
    def unregister(self, key_sequence: str) -> None:
        """Remove a registered shortcut."""
        if key_sequence in self._shortcuts:
            self._shortcuts[key_sequence].deleteLater()
            del self._shortcuts[key_sequence]
    
    def enable(self, key_sequence: str) -> None:
        """Enable a specific shortcut."""
        if key_sequence in self._shortcuts:
            self._shortcuts[key_sequence].setEnabled(True)
    
    def disable(self, key_sequence: str) -> None:
        """Disable a specific shortcut."""
        if key_sequence in self._shortcuts:
            self._shortcuts[key_sequence].setEnabled(False)
    
    def enable_all(self) -> None:
        """Enable all shortcuts."""
        for shortcut in self._shortcuts.values():
            shortcut.setEnabled(True)
    
    def disable_all(self) -> None:
        """Disable all shortcuts."""
        for shortcut in self._shortcuts.values():
            shortcut.setEnabled(False)


class GlobalShortcuts:
    """
    Global keyboard shortcuts available throughout the application.
    
    Standard shortcuts:
    - Ctrl+H: Go to Home
    - Ctrl+R: Refresh current page
    - Escape: Go back / Cancel
    - F5: Refresh (alternative)
    """
    
    # Signal definitions for global actions
    home_requested = None
    refresh_requested = None
    back_requested = None
    
    @classmethod
    def setup_for_page(
        cls,
        widget: QWidget,
        on_refresh=None,
        on_back=None,
        on_home=None,
        on_enter=None
    ) -> KeyboardShortcuts:
        """
        Setup standard keyboard shortcuts for a page.
        
        Args:
            widget: The page widget
            on_refresh: Callback for Ctrl+R / F5
            on_back: Callback for Escape
            on_home: Callback for Ctrl+H
            on_enter: Callback for Enter key
        
        Returns:
            KeyboardShortcuts manager instance
        """
        shortcuts = KeyboardShortcuts(widget)
        
        # Refresh shortcuts
        if on_refresh:
            shortcuts.register("Ctrl+R", on_refresh)
            shortcuts.register("F5", on_refresh)
        
        # Back/Cancel shortcut
        if on_back:
            shortcuts.register("Escape", on_back)
        
        # Home shortcut
        if on_home:
            shortcuts.register("Ctrl+H", on_home)
        
        # Enter/Submit shortcut
        if on_enter:
            shortcuts.register("Return", on_enter)
        
        return shortcuts



#!/usr/bin/env python3
"""
16Score AI Desktop Application - Main Entry Point

This is the main entry point for the 16Score AI desktop application.
It initializes the PyQt5 application and starts with the login page.

Features initialized at startup:
- Logging system with rotation and levels
- Crash reporting via Sentry (if configured)
- Network connectivity monitoring
- Automatic update checking
"""

from __future__ import annotations

import sys
import os
import traceback
import warnings

# Fix Windows taskbar icon - must be done before QApplication is created
if sys.platform == 'win32':
    import ctypes
    from ctypes import wintypes
    # Set the AppUserModelID so Windows shows our icon in taskbar
    myappid = 'com.16score.ai.desktop.1.0'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

# Suppress pkg_resources deprecation warning before importing it
warnings.filterwarnings("ignore", message=".*pkg_resources.*deprecated.*")

# Fix pkg_resources issue for PyInstaller
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        import pkg_resources.extern
    except ImportError:
        import types
        pkg_resources_extern = types.ModuleType('pkg_resources.extern')
        sys.modules['pkg_resources.extern'] = pkg_resources_extern

# Handle frozen mode (PyInstaller executable)
IS_FROZEN = getattr(sys, 'frozen', False)
if IS_FROZEN:
    # If running as a PyInstaller bundle, add the executable's directory to sys.path
    base_path = os.path.dirname(sys.executable)
    sys.path.insert(0, base_path)
    
    # Also add src directory if it exists
    src_path = os.path.join(base_path, 'src')
    if os.path.exists(src_path):
        sys.path.insert(0, src_path)
else:
    # Add the src directory to Python path for development
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    if os.path.exists(src_path):
        sys.path.insert(0, src_path)

# Import version info
from score_ai.__version__ import __version__, __app_name__

# Import logging after path setup
from score_ai.core.logging_config import setup_logging, get_logger

# Initialize logging system
# Set use_json=True for production, False for development
setup_logging(
    log_dir="logs",
    level="INFO",
    use_json=False,
    retention_days=30,
    console_output=True,
    file_output=True
)

logger = get_logger(__name__)

# =============================================================================
# Initialize Crash Reporting (Sentry)
# =============================================================================
# Set SENTRY_DSN environment variable to enable crash reporting
from score_ai.core.crash_reporter import CrashReporter

CrashReporter.initialize(
    environment="production" if IS_FROZEN else "development",
    app_version=__version__,
    enabled=True  # Will only activate if SENTRY_DSN env var is set
)

# =============================================================================
# Initialize Network Monitoring
# =============================================================================
from score_ai.core.network_monitor import get_network_monitor

network_monitor = get_network_monitor()

# =============================================================================
# Initialize Auto-Updater
# =============================================================================
from score_ai.core.auto_updater import get_updater

updater = get_updater()
updater.current_version = __version__

# Debug mode flag - set to False for production builds
DEBUG_LOG_IMPORTS = False

if DEBUG_LOG_IMPORTS:
    import builtins
    import threading

    _import_log_lock = threading.Lock()
    _import_log_file = open('imported_modules.txt', 'w')

    original_import = builtins.__import__

    def custom_import(name, *args, **kwargs):
        with _import_log_lock:
            _import_log_file.write(f"Importing: {name}\n")
            _import_log_file.flush()
        return original_import(name, *args, **kwargs)

    builtins.__import__ = custom_import

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QTimer


def check_for_updates_on_startup(main_window) -> None:
    """
    Check for updates after the main window is shown.
    
    Runs asynchronously to not block the UI.
    """
    try:
        update_info = updater.check_for_update(silent=True)
        if update_info:
            logger.info(f"Update available: {update_info.version}")
            # Show update dialog after a short delay
            QTimer.singleShot(2000, lambda: updater.show_update_dialog(main_window))
    except Exception as e:
        logger.warning(f"Failed to check for updates: {e}")


def main() -> None:
    """
    Main application entry point.
    
    Initializes the PyQt5 application, sets up logging, and launches
    the main window. Handles errors gracefully with user-friendly
    error dialogs.
    """
    try:
        # Create the application
        app = QApplication(sys.argv)
        app.setApplicationName("16Score-AI")
        app.setApplicationVersion(__version__)
        
        # Set application icon (use .ico for Windows taskbar)
        from PyQt5.QtGui import QIcon, QPixmap
        from PyQt5.QtCore import QSize
        ico_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "images", "16score.ico"))
        png_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "images", "16score_logo.png"))
        
        app_icon = QIcon()
        icon_set = False
        
        # Try PNG first as it's more reliable for PyQt5
        if os.path.exists(png_path):
            pixmap = QPixmap(png_path)
            if not pixmap.isNull():
                app_icon = QIcon(pixmap)
                app.setWindowIcon(app_icon)
                logger.info(f"Application icon set from PNG: {png_path}")
                icon_set = True
        
        # Fallback to ICO
        if not icon_set and os.path.exists(ico_path):
            app_icon = QIcon(ico_path)
            app.setWindowIcon(app_icon)
            logger.info(f"Application icon set from ICO: {ico_path}")
            icon_set = True
        
        if not icon_set:
            logger.warning(f"Icon not found at: {ico_path} or {png_path}")
        
        logger.info("Starting 16Score-AI...")
        
        # Start network monitoring
        network_monitor.start()
        logger.info("Network monitor started")
        
        # Check initial network status
        if not network_monitor.is_online:
            logger.warning("Starting in offline mode - no network connection detected")
        
        # Import and launch main application
        try:
            from score_ai.ui.main_window import MainWindow
            logger.info("Successfully imported MainWindow")
            
            # Create and show the main window
            main_win = MainWindow()
            # Set the window icon again on main window to ensure taskbar shows it
            if not app_icon.isNull():
                main_win.setWindowIcon(app_icon)
            
            # Force Windows to use our icon in taskbar
            if sys.platform == 'win32':
                try:
                    import ctypes
                    from ctypes import wintypes
                    hwnd = int(main_win.winId())
                    
                    # Load the icon using Windows API
                    if os.path.exists(ico_path):
                        # Load icon from .ico file
                        hicon = ctypes.windll.user32.LoadImageW(
                            None, ico_path, 1, 0, 0, 0x00000010  # IMAGE_ICON, LR_LOADFROMFILE
                        )
                        if hicon:
                            # Set both small and big icons
                            ctypes.windll.user32.SendMessageW(hwnd, 0x0080, 0, hicon)  # WM_SETICON, ICON_SMALL
                            ctypes.windll.user32.SendMessageW(hwnd, 0x0080, 1, hicon)  # WM_SETICON, ICON_BIG
                            logger.info("Windows taskbar icon set via WM_SETICON")
                except Exception as e:
                    logger.warning(f"Could not set Windows icon directly: {e}")
            
            main_win.show()
            logger.info("Main window created and shown")
            
            # Check for updates after a short delay (don't block startup)
            # DISABLED: Update endpoint not available yet
            # QTimer.singleShot(3000, lambda: check_for_updates_on_startup(main_win))
            
            # Start automatic update checking (every 24 hours)
            # DISABLED: Update endpoint not available yet
            # updater.start_auto_check(interval_hours=24)
            
            # Start the application event loop
            logger.info("Starting application event loop...")
            exit_code = app.exec_()
            
            # Cleanup on exit
            logger.info("Application shutting down...")
            network_monitor.stop()
            updater.stop_auto_check()
            CrashReporter.flush()  # Send any pending crash reports
            
            sys.exit(exit_code)
            
        except ImportError as e:
            logger.error(f"Error importing main application: {e}", exc_info=True)
            CrashReporter.capture_exception(e)
            
            # Show error message
            QMessageBox.critical(
                None, 
                "Import Error", 
                f"Failed to import main application:\n{str(e)}\n\n"
                f"Please check that all required modules are available."
            )
        
        except Exception as e:
            logger.error(f"Error launching application: {e}", exc_info=True)
            CrashReporter.capture_exception(e)
            
            # Show error message
            QMessageBox.critical(
                None, 
                "Application Error", 
                f"Failed to start application:\n{str(e)}\n\n"
                f"Please check the console for more details."
            )
    
    except Exception as e:
        logger.critical(f"Critical error in main: {e}", exc_info=True)
        CrashReporter.capture_exception(e)
        
        # Try to show error message if QApplication is available
        try:
            app = QApplication(sys.argv)
            QMessageBox.critical(
                None, 
                "Critical Error", 
                f"Critical application error:\n{str(e)}\n\n"
                f"Please check the console for more details."
            )
        except Exception:
            logger.critical("Could not show error dialog - application failed to initialize")
    finally:
        # Ensure crash reporter flushes on exit
        CrashReporter.flush(timeout=2.0)


if __name__ == "__main__":
    main()

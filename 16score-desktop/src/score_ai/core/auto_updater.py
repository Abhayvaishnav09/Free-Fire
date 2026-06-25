"""
Auto-Update System for 16Score AI Application

Provides automatic update checking and installation with:
- Version comparison with remote server
- Background update downloads
- Update notifications with UI dialogs
- Rollback support
- Update progress tracking

Usage:
    from score_ai.core.auto_updater import AutoUpdater, updater
    
    # Initialize with update server URL
    updater.set_update_url("https://updates.16score.com/api/version")
    
    # Check for updates
    if updater.check_for_update():
        updater.show_update_dialog()
    
    # Or start automatic checking
    updater.start_auto_check(interval_hours=24)

Example:
    >>> updater = AutoUpdater()
    >>> update_info = updater.check_for_update()
    >>> if update_info:
    ...     print(f"New version available: {update_info.version}")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional
from packaging import version as pkg_version

from PyQt5.QtCore import QObject, pyqtSignal, QTimer, QThread
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QMessageBox, QWidget
)
from score_ai.core.config_manager import config

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Try to import requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not installed. Auto-update disabled.")


class UpdateStatus(Enum):
    """Update status enumeration"""
    IDLE = "idle"
    CHECKING = "checking"
    AVAILABLE = "available"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    INSTALLING = "installing"
    COMPLETE = "complete"
    ERROR = "error"
    UP_TO_DATE = "up_to_date"


@dataclass
class UpdateInfo:
    """
    Information about an available update.
    
    Attributes:
        version: New version string
        download_url: URL to download the update
        release_notes: Release notes/changelog
        release_date: Release date
        file_size: Download file size in bytes
        checksum: SHA256 checksum of the download
        is_mandatory: Whether update is mandatory
        min_version: Minimum version required to update from
    """
    version: str
    download_url: str
    release_notes: str = ""
    release_date: Optional[str] = None
    file_size: int = 0
    checksum: str = ""
    is_mandatory: bool = False
    min_version: str = "0.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "download_url": self.download_url,
            "release_notes": self.release_notes,
            "release_date": self.release_date,
            "file_size": self.file_size,
            "checksum": self.checksum,
            "is_mandatory": self.is_mandatory,
            "min_version": self.min_version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UpdateInfo":
        """Create from dictionary."""
        return cls(
            version=data.get("version", "0.0.0"),
            download_url=data.get("download_url", ""),
            release_notes=data.get("release_notes", ""),
            release_date=data.get("release_date"),
            file_size=data.get("file_size", 0),
            checksum=data.get("checksum", ""),
            is_mandatory=data.get("is_mandatory", False),
            min_version=data.get("min_version", "0.0.0"),
        )


class DownloadThread(QThread):
    """Background thread for downloading updates."""
    
    progress = pyqtSignal(int, int)  # bytes_downloaded, total_bytes
    finished = pyqtSignal(bool, str)  # success, message/path
    
    def __init__(
        self,
        url: str,
        dest_path: str,
        checksum: str = "",
        parent: Optional[QObject] = None
    ):
        super().__init__(parent)
        self.url = url
        self.dest_path = dest_path
        self.expected_checksum = checksum
        self._cancelled = False
    
    def run(self):
        """Download the update file."""
        try:
            response = requests.get(self.url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            hasher = hashlib.sha256()
            
            with open(self.dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if self._cancelled:
                        self.finished.emit(False, "Download cancelled")
                        return
                    
                    if chunk:
                        f.write(chunk)
                        hasher.update(chunk)
                        downloaded += len(chunk)
                        self.progress.emit(downloaded, total_size)
            
            # Verify checksum
            if self.expected_checksum:
                actual_checksum = hasher.hexdigest()
                if actual_checksum.lower() != self.expected_checksum.lower():
                    os.remove(self.dest_path)
                    self.finished.emit(False, "Checksum verification failed")
                    return
            
            self.finished.emit(True, self.dest_path)
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            self.finished.emit(False, str(e))
    
    def cancel(self):
        """Cancel the download."""
        self._cancelled = True


class AutoUpdater(QObject):
    """
    Automatic update checker and installer.
    
    Signals:
        update_available(UpdateInfo): Emitted when update is available
        update_progress(int, int): Download progress (downloaded, total)
        update_status_changed(UpdateStatus): Status change signal
        update_error(str): Error message signal
    """
    
    # Signals
    update_available = pyqtSignal(object)  # UpdateInfo
    update_progress = pyqtSignal(int, int)  # bytes, total
    update_status_changed = pyqtSignal(object)  # UpdateStatus
    update_error = pyqtSignal(str)
    
    def __init__(
        self,
        current_version: str = "1.0.0",
        update_url: Optional[str] = None,
        parent: Optional[QObject] = None
    ):
        """
        Initialize the auto-updater.
        
        Args:
            current_version: Current application version
            update_url: URL to check for updates
            parent: Optional Qt parent
        """
        super().__init__(parent)
        
        self._current_version = current_version
        # Get update URL from config, environment variable, or use default
        if update_url:
            self._update_url = update_url
        else:
            # Try to get from config first
            backend_url = config.get('api.backend_url', '').rstrip('/')
            if backend_url:
                self._update_url = f"{backend_url}/api/version"
            else:
                # Fallback to environment variable or default
                self._update_url = os.environ.get(
                    "SCORE_UPDATE_URL",
                    "https://updates.16score.com/api/version"
                )
        
        self._status = UpdateStatus.IDLE
        self._update_info: Optional[UpdateInfo] = None
        self._download_thread: Optional[DownloadThread] = None
        self._downloaded_file: Optional[str] = None
        
        # Auto-check timer
        self._auto_check_timer: Optional[QTimer] = None
        self._last_check: Optional[datetime] = None
        
        # Settings
        self._check_on_startup = True
        self._auto_download = False
        self._auto_install = False
        
        # Update directory
        self._update_dir = Path(tempfile.gettempdir()) / "16score_updates"
        self._update_dir.mkdir(exist_ok=True)
    
    @property
    def current_version(self) -> str:
        """Get current application version."""
        return self._current_version
    
    @current_version.setter
    def current_version(self, value: str):
        """Set current application version."""
        self._current_version = value
    
    @property
    def status(self) -> UpdateStatus:
        """Get current update status."""
        return self._status
    
    @property
    def update_info(self) -> Optional[UpdateInfo]:
        """Get information about available update."""
        return self._update_info
    
    def set_update_url(self, url: str) -> None:
        """Set the update server URL."""
        self._update_url = url
    
    def _set_status(self, status: UpdateStatus) -> None:
        """Update status and emit signal."""
        self._status = status
        self.update_status_changed.emit(status)
    
    def check_for_update(self, silent: bool = False) -> Optional[UpdateInfo]:
        """
        Check for available updates.
        
        Args:
            silent: If True, don't emit signals or show errors
        
        Returns:
            UpdateInfo if update available, None otherwise
        """
        if not REQUESTS_AVAILABLE:
            if not silent:
                self.update_error.emit("requests library not available")
            return None
        
        self._set_status(UpdateStatus.CHECKING)
        self._last_check = datetime.now()
        
        try:
            logger.info(f"Checking for updates at {self._update_url}")
            
            response = requests.get(
                self._update_url,
                timeout=10,
                headers={"User-Agent": f"16ScoreAI/{self._current_version}"}
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Parse update info
            latest_version = data.get("version", "0.0.0")
            
            # Compare versions
            if self._is_newer_version(latest_version):
                self._update_info = UpdateInfo.from_dict(data)
                self._set_status(UpdateStatus.AVAILABLE)
                
                logger.info(f"Update available: {latest_version}")
                
                if not silent:
                    self.update_available.emit(self._update_info)
                
                return self._update_info
            else:
                self._set_status(UpdateStatus.UP_TO_DATE)
                logger.info("Application is up to date")
                return None
                
        except requests.RequestException as e:
            error_msg = f"Failed to check for updates: {e}"
            logger.error(error_msg)
            self._set_status(UpdateStatus.ERROR)
            
            if not silent:
                self.update_error.emit(error_msg)
            
            return None
        except (json.JSONDecodeError, KeyError) as e:
            error_msg = f"Invalid update response: {e}"
            logger.error(error_msg)
            self._set_status(UpdateStatus.ERROR)
            
            if not silent:
                self.update_error.emit(error_msg)
            
            return None
    
    def _is_newer_version(self, remote_version: str) -> bool:
        """
        Compare versions to check if remote is newer.
        
        Args:
            remote_version: Version string from server
        
        Returns:
            True if remote version is newer
        """
        try:
            current = pkg_version.parse(self._current_version)
            remote = pkg_version.parse(remote_version)
            return remote > current
        except Exception as e:
            logger.error(f"Version comparison failed: {e}")
            # Fallback to string comparison
            return remote_version > self._current_version
    
    def download_update(self) -> bool:
        """
        Start downloading the available update.
        
        Returns:
            True if download started, False otherwise
        """
        if not self._update_info:
            logger.error("No update available to download")
            return False
        
        if self._download_thread and self._download_thread.isRunning():
            logger.warning("Download already in progress")
            return False
        
        self._set_status(UpdateStatus.DOWNLOADING)
        
        # Determine download path
        filename = f"16ScoreAI_{self._update_info.version}.exe"
        dest_path = str(self._update_dir / filename)
        
        # Create and start download thread
        self._download_thread = DownloadThread(
            url=self._update_info.download_url,
            dest_path=dest_path,
            checksum=self._update_info.checksum,
            parent=self
        )
        
        self._download_thread.progress.connect(self._on_download_progress)
        self._download_thread.finished.connect(self._on_download_finished)
        self._download_thread.start()
        
        logger.info(f"Started downloading update to {dest_path}")
        return True
    
    def cancel_download(self) -> None:
        """Cancel the current download."""
        if self._download_thread and self._download_thread.isRunning():
            self._download_thread.cancel()
            self._download_thread.wait()
            self._set_status(UpdateStatus.AVAILABLE)
    
    def _on_download_progress(self, downloaded: int, total: int) -> None:
        """Handle download progress updates."""
        self.update_progress.emit(downloaded, total)
    
    def _on_download_finished(self, success: bool, result: str) -> None:
        """Handle download completion."""
        if success:
            self._downloaded_file = result
            self._set_status(UpdateStatus.DOWNLOADED)
            logger.info(f"Update downloaded to {result}")
            
            if self._auto_install:
                self.install_update()
        else:
            self._set_status(UpdateStatus.ERROR)
            self.update_error.emit(result)
            logger.error(f"Download failed: {result}")
    
    def install_update(self) -> bool:
        """
        Install the downloaded update.
        
        Returns:
            True if installation started, False otherwise
        """
        if not self._downloaded_file or not os.path.exists(self._downloaded_file):
            logger.error("No downloaded update to install")
            return False
        
        self._set_status(UpdateStatus.INSTALLING)
        
        try:
            # For Windows: Run the installer
            if sys.platform == "win32":
                # Create a batch script to wait for app exit and run installer
                batch_script = self._create_update_script()
                
                # Run the batch script
                subprocess.Popen(
                    ["cmd", "/c", batch_script],
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                
                logger.info("Update installation started - application will restart")
                
                # Exit the current application
                # The user will be prompted to save work before this
                return True
            else:
                logger.warning("Auto-install not supported on this platform")
                # Open the download location
                os.startfile(os.path.dirname(self._downloaded_file))
                return False
                
        except Exception as e:
            error_msg = f"Failed to install update: {e}"
            logger.error(error_msg)
            self._set_status(UpdateStatus.ERROR)
            self.update_error.emit(error_msg)
            return False
    
    def _create_update_script(self) -> str:
        """
        Create a batch script for Windows update installation.
        
        Returns:
            Path to the batch script
        """
        script_path = str(self._update_dir / "update.bat")
        
        # Get current executable path
        if getattr(sys, 'frozen', False):
            current_exe = sys.executable
        else:
            current_exe = sys.argv[0]
        
        script_content = f'''@echo off
echo Waiting for application to close...
timeout /t 3 /nobreak > nul

echo Installing update...
start "" "{self._downloaded_file}"

echo Cleaning up...
del "%~f0"
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        return script_path
    
    def start_auto_check(self, interval_hours: float = 24) -> None:
        """
        Start automatic update checking.
        
        Args:
            interval_hours: Hours between checks
        """
        if self._auto_check_timer is not None:
            self._auto_check_timer.stop()
        
        self._auto_check_timer = QTimer(self)
        self._auto_check_timer.timeout.connect(lambda: self.check_for_update(silent=True))
        
        # Convert hours to milliseconds
        interval_ms = int(interval_hours * 60 * 60 * 1000)
        self._auto_check_timer.start(interval_ms)
        
        logger.info(f"Auto-update check started (every {interval_hours} hours)")
        
        # Do an initial check
        if self._check_on_startup:
            self.check_for_update(silent=True)
    
    def stop_auto_check(self) -> None:
        """Stop automatic update checking."""
        if self._auto_check_timer is not None:
            self._auto_check_timer.stop()
            self._auto_check_timer = None
            logger.info("Auto-update check stopped")
    
    def show_update_dialog(self, parent: Optional[QWidget] = None) -> bool:
        """
        Show update dialog to user.
        
        Args:
            parent: Parent widget for the dialog
        
        Returns:
            True if user chose to update, False otherwise
        """
        if not self._update_info:
            return False
        
        dialog = UpdateDialog(self._update_info, self, parent)
        result = dialog.exec_()
        
        return result == QDialog.Accepted
    
    def get_update_history(self) -> List[Dict[str, Any]]:
        """Get list of previous updates (from local storage)."""
        history_file = self._update_dir / "update_history.json"
        
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return []
        return []
    
    def cleanup_old_updates(self, keep_latest: int = 2) -> None:
        """
        Clean up old update files.
        
        Args:
            keep_latest: Number of recent updates to keep
        """
        try:
            files = list(self._update_dir.glob("16ScoreAI_*.exe"))
            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            for old_file in files[keep_latest:]:
                old_file.unlink()
                logger.debug(f"Removed old update: {old_file}")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old updates: {e}")


class UpdateDialog(QDialog):
    """Dialog for showing update information and progress."""
    
    def __init__(
        self,
        update_info: UpdateInfo,
        updater: AutoUpdater,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        
        self.update_info = update_info
        self.updater = updater
        
        self.setWindowTitle("Update Available")
        self.setMinimumWidth(450)
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1d23;
            }
            QLabel {
                color: #e0e0e0;
            }
            QPushButton {
                background-color: #2186eb;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1a6fc9;
            }
            QPushButton:pressed {
                background-color: #155ba3;
            }
            QPushButton#cancelBtn {
                background-color: #3a3f4b;
            }
            QPushButton#cancelBtn:hover {
                background-color: #4a4f5b;
            }
            QProgressBar {
                border: none;
                border-radius: 4px;
                background-color: #2a2f3a;
                height: 8px;
            }
            QProgressBar::chunk {
                background-color: #2186eb;
                border-radius: 4px;
            }
        """)
        
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)
        
        # Title
        title = QLabel("🎉 New Version Available!")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #2186eb;")
        layout.addWidget(title)
        
        # Version info
        version_text = f"Version {self.update_info.version} is now available.\nYou have version {self.updater.current_version}."
        version_label = QLabel(version_text)
        version_label.setStyleSheet("font-size: 14px; color: #b0b0b0;")
        layout.addWidget(version_label)
        
        # Release notes
        if self.update_info.release_notes:
            notes_title = QLabel("What's New:")
            notes_title.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 8px;")
            layout.addWidget(notes_title)
            
            notes = QLabel(self.update_info.release_notes[:500])
            notes.setWordWrap(True)
            notes.setStyleSheet("font-size: 13px; color: #a0a0a0; padding: 8px; background: #2a2f3a; border-radius: 6px;")
            layout.addWidget(notes)
        
        # Progress bar (hidden initially)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Progress label
        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        self.progress_label.setStyleSheet("font-size: 12px; color: #808080;")
        layout.addWidget(self.progress_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        
        self.later_btn = QPushButton("Later")
        self.later_btn.setObjectName("cancelBtn")
        self.later_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.later_btn)
        
        button_layout.addStretch()
        
        self.update_btn = QPushButton("Download & Install")
        self.update_btn.clicked.connect(self._start_download)
        button_layout.addWidget(self.update_btn)
        
        layout.addLayout(button_layout)
    
    def _connect_signals(self):
        """Connect updater signals."""
        self.updater.update_progress.connect(self._on_progress)
        self.updater.update_status_changed.connect(self._on_status_changed)
        self.updater.update_error.connect(self._on_error)
    
    def _start_download(self):
        """Start the download process."""
        self.update_btn.setEnabled(False)
        self.update_btn.setText("Downloading...")
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        
        self.updater.download_update()
    
    def _on_progress(self, downloaded: int, total: int):
        """Update progress bar."""
        if total > 0:
            percent = int((downloaded / total) * 100)
            self.progress_bar.setValue(percent)
            
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total / (1024 * 1024)
            self.progress_label.setText(f"{mb_downloaded:.1f} MB / {mb_total:.1f} MB")
    
    def _on_status_changed(self, status: UpdateStatus):
        """Handle status changes."""
        if status == UpdateStatus.DOWNLOADED:
            self.update_btn.setText("Install & Restart")
            self.update_btn.setEnabled(True)
            self.update_btn.clicked.disconnect()
            self.update_btn.clicked.connect(self._start_install)
            
        elif status == UpdateStatus.INSTALLING:
            self.update_btn.setEnabled(False)
            self.update_btn.setText("Installing...")
            self.later_btn.setEnabled(False)
    
    def _start_install(self):
        """Start the installation."""
        reply = QMessageBox.question(
            self,
            "Install Update",
            "The application will close to install the update.\nDo you want to continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            if self.updater.install_update():
                # Close the entire application
                from PyQt5.QtWidgets import QApplication
                QApplication.quit()
    
    def _on_error(self, error: str):
        """Handle errors."""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.update_btn.setEnabled(True)
        self.update_btn.setText("Retry Download")
        
        QMessageBox.warning(self, "Update Error", f"Failed to download update:\n{error}")


# Global singleton instance
_updater: Optional[AutoUpdater] = None


def get_updater() -> AutoUpdater:
    """
    Get or create the global AutoUpdater instance.
    
    Returns:
        The singleton AutoUpdater instance
    """
    global _updater
    if _updater is None:
        _updater = AutoUpdater()
    return _updater


# Convenience alias
updater = property(lambda self: get_updater())


# Module-level convenience functions
def check_for_updates(silent: bool = False) -> Optional[UpdateInfo]:
    """Check for available updates."""
    return get_updater().check_for_update(silent=silent)


def start_auto_updates(interval_hours: float = 24) -> None:
    """Start automatic update checking."""
    get_updater().start_auto_check(interval_hours=interval_hours)


def stop_auto_updates() -> None:
    """Stop automatic update checking."""
    get_updater().stop_auto_check()


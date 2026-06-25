"""
Network Status Monitor for 16Score AI Application

Provides real-time network connectivity monitoring with:
- Periodic connectivity checks
- Online/offline status signals for UI updates
- Automatic retry when connection is restored
- Multiple endpoint fallback checking

Usage:
    from score_ai.core.network_monitor import NetworkMonitor, network_monitor
    
    # Using the singleton
    network_monitor.start()
    if network_monitor.is_online:
        make_api_call()
    
    # Or create your own instance
    monitor = NetworkMonitor()
    monitor.status_changed.connect(on_status_change)
    monitor.start()

Example:
    >>> monitor = NetworkMonitor()
    >>> monitor.is_online
    True
    >>> monitor.last_check_time
    datetime.datetime(2024, 1, 15, 10, 30, 45)
"""

from __future__ import annotations

import logging
import socket
import time
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Callable
from threading import Thread, Event

from PyQt5.QtCore import QObject, pyqtSignal, QTimer

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class NetworkStatus:
    """Network status constants"""
    ONLINE = "online"
    OFFLINE = "offline"
    CHECKING = "checking"
    UNKNOWN = "unknown"


class NetworkMonitor(QObject):
    """
    Monitors network connectivity and emits signals on status changes.
    
    Uses multiple methods to check connectivity:
    1. DNS resolution check
    2. Socket connection to reliable hosts
    3. HTTP HEAD request to API endpoint
    
    Signals:
        status_changed(bool): Emitted when online status changes
        connection_restored: Emitted when connection is restored after being offline
        connection_lost: Emitted when connection is lost
    
    Attributes:
        is_online: Current online status
        status: Current status string (online, offline, checking, unknown)
        last_check_time: Timestamp of last connectivity check
    """
    
    # Signals
    status_changed = pyqtSignal(bool)  # True = online, False = offline
    connection_restored = pyqtSignal()
    connection_lost = pyqtSignal()
    
    # Reliable hosts for connectivity checking
    CHECK_HOSTS = [
        ("8.8.8.8", 53),        # Google DNS
        ("1.1.1.1", 53),        # Cloudflare DNS
        ("208.67.222.222", 53), # OpenDNS
    ]
    
    # DNS hosts to resolve
    DNS_HOSTS = [
        "google.com",
        "cloudflare.com",
        "microsoft.com",
    ]
    
    def __init__(
        self,
        check_interval: int = 30,
        timeout: float = 5.0,
        parent: Optional[QObject] = None
    ):
        """
        Initialize the network monitor.
        
        Args:
            check_interval: Seconds between connectivity checks (default: 30)
            timeout: Socket timeout in seconds (default: 5.0)
            parent: Optional Qt parent object
        """
        super().__init__(parent)
        
        self._is_online: bool = True  # Assume online initially
        self._status: str = NetworkStatus.UNKNOWN
        self._last_check_time: Optional[datetime] = None
        self._check_interval = check_interval
        self._timeout = timeout
        
        # Timer for periodic checks
        self._timer: Optional[QTimer] = None
        self._stop_event = Event()
        
        # Callbacks for non-Qt usage
        self._status_callbacks: List[Callable[[bool], None]] = []
        
        # Perform initial check
        self._check_connectivity()
    
    @property
    def is_online(self) -> bool:
        """Check if currently online."""
        return self._is_online
    
    @property
    def status(self) -> str:
        """Get current status string."""
        return self._status
    
    @property
    def last_check_time(self) -> Optional[datetime]:
        """Get timestamp of last connectivity check."""
        return self._last_check_time
    
    def start(self) -> None:
        """Start periodic connectivity monitoring."""
        if self._timer is not None:
            return  # Already running
        
        self._stop_event.clear()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._check_connectivity)
        self._timer.start(self._check_interval * 1000)
        
        logger.info(f"Network monitor started (interval: {self._check_interval}s)")
    
    def stop(self) -> None:
        """Stop periodic connectivity monitoring."""
        self._stop_event.set()
        
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        
        logger.info("Network monitor stopped")
    
    def check_now(self) -> bool:
        """
        Perform immediate connectivity check.
        
        Returns:
            True if online, False if offline
        """
        return self._check_connectivity()
    
    def add_status_callback(self, callback: Callable[[bool], None]) -> None:
        """
        Add a callback for status changes (for non-Qt usage).
        
        Args:
            callback: Function to call with boolean online status
        """
        self._status_callbacks.append(callback)
    
    def remove_status_callback(self, callback: Callable[[bool], None]) -> None:
        """Remove a status callback."""
        if callback in self._status_callbacks:
            self._status_callbacks.remove(callback)
    
    def _check_connectivity(self) -> bool:
        """
        Check network connectivity using multiple methods.
        
        Returns:
            True if online, False if offline
        """
        self._status = NetworkStatus.CHECKING
        self._last_check_time = datetime.now()
        
        was_online = self._is_online
        is_now_online = False
        
        # Method 1: Socket connection to reliable hosts
        for host, port in self.CHECK_HOSTS:
            if self._check_socket(host, port):
                is_now_online = True
                break
        
        # Method 2: DNS resolution (backup check)
        if not is_now_online:
            for host in self.DNS_HOSTS:
                if self._check_dns(host):
                    is_now_online = True
                    break
        
        # Update status
        self._is_online = is_now_online
        self._status = NetworkStatus.ONLINE if is_now_online else NetworkStatus.OFFLINE
        
        # Emit signals if status changed
        if was_online != is_now_online:
            logger.info(f"Network status changed: {'online' if is_now_online else 'offline'}")
            self.status_changed.emit(is_now_online)
            
            if is_now_online:
                self.connection_restored.emit()
            else:
                self.connection_lost.emit()
            
            # Call registered callbacks
            for callback in self._status_callbacks:
                try:
                    callback(is_now_online)
                except Exception as e:
                    logger.error(f"Error in status callback: {e}")
        
        return is_now_online
    
    def _check_socket(self, host: str, port: int) -> bool:
        """
        Check connectivity by attempting socket connection.
        
        Args:
            host: Host to connect to
            port: Port to connect to
        
        Returns:
            True if connection successful
        """
        try:
            socket.setdefaulttimeout(self._timeout)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self._timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except (socket.timeout, socket.error, OSError) as e:
            logger.debug(f"Socket check failed for {host}:{port}: {e}")
            return False
    
    def _check_dns(self, hostname: str) -> bool:
        """
        Check connectivity by attempting DNS resolution.
        
        Args:
            hostname: Hostname to resolve
        
        Returns:
            True if resolution successful
        """
        try:
            socket.setdefaulttimeout(self._timeout)
            socket.gethostbyname(hostname)
            return True
        except (socket.gaierror, socket.timeout, OSError) as e:
            logger.debug(f"DNS check failed for {hostname}: {e}")
            return False
    
    def wait_for_connection(
        self,
        timeout: Optional[float] = None,
        check_interval: float = 2.0
    ) -> bool:
        """
        Block until network connection is available.
        
        Args:
            timeout: Maximum seconds to wait (None = wait forever)
            check_interval: Seconds between checks
        
        Returns:
            True if connected, False if timeout reached
        """
        start_time = time.time()
        
        while True:
            if self._check_connectivity():
                return True
            
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False
            
            time.sleep(check_interval)
    
    def get_connection_info(self) -> dict:
        """
        Get detailed connection information.
        
        Returns:
            Dictionary with connection details
        """
        return {
            "is_online": self._is_online,
            "status": self._status,
            "last_check": self._last_check_time.isoformat() if self._last_check_time else None,
            "check_interval": self._check_interval,
        }


class OfflineError(Exception):
    """
    Exception raised when attempting network operation while offline.
    
    Example:
        >>> if not network_monitor.is_online:
        ...     raise OfflineError("Cannot sync while offline")
    """
    
    def __init__(self, message: str = "No network connection available"):
        self.message = message
        super().__init__(self.message)


def require_online(func):
    """
    Decorator that ensures network connectivity before executing function.
    
    Raises OfflineError if network is not available.
    
    Example:
        >>> @require_online
        ... def fetch_data():
        ...     return api.get_data()
    """
    def wrapper(*args, **kwargs):
        if not network_monitor.is_online:
            raise OfflineError(f"Cannot execute {func.__name__}: No network connection")
        return func(*args, **kwargs)
    return wrapper


# Global singleton instance
_network_monitor: Optional[NetworkMonitor] = None


def get_network_monitor() -> NetworkMonitor:
    """
    Get or create the global NetworkMonitor instance.
    
    Returns:
        The singleton NetworkMonitor instance
    """
    global _network_monitor
    if _network_monitor is None:
        _network_monitor = NetworkMonitor()
    return _network_monitor


# Convenience alias
network_monitor = property(lambda self: get_network_monitor())


# Module-level convenience functions
def is_online() -> bool:
    """Check if network is currently online."""
    return get_network_monitor().is_online


def check_connection() -> bool:
    """Perform immediate connectivity check."""
    return get_network_monitor().check_now()


def start_monitoring() -> None:
    """Start the global network monitor."""
    get_network_monitor().start()


def stop_monitoring() -> None:
    """Stop the global network monitor."""
    get_network_monitor().stop()


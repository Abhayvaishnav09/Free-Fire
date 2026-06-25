"""
Crash Reporter for 16Score AI Application

Provides crash reporting and error tracking using Sentry with:
- Automatic exception capture
- User context tracking
- Custom tags and breadcrumbs
- Performance monitoring
- Offline error queuing

Usage:
    from score_ai.core.crash_reporter import CrashReporter, crash_reporter
    
    # Initialize at app startup
    CrashReporter.initialize(dsn="your-sentry-dsn")
    
    # Set user context after login
    crash_reporter.set_user(user_id="123", email="user@example.com")
    
    # Capture custom errors
    crash_reporter.capture_exception(exception)
    
    # Add breadcrumbs for debugging
    crash_reporter.add_breadcrumb("User clicked button", category="ui")

Example:
    >>> CrashReporter.initialize()
    >>> crash_reporter.capture_message("App started successfully")
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional, Callable
from functools import wraps

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Try to import Sentry SDK
try:
    import sentry_sdk
    from sentry_sdk.integrations.logging import LoggingIntegration
    from sentry_sdk.integrations.threading import ThreadingIntegration
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False
    logger.warning("sentry-sdk not installed. Crash reporting disabled. Install with: pip install sentry-sdk")


class CrashReporter:
    """
    Centralized crash reporting using Sentry.
    
    Provides methods for:
    - Initializing Sentry with proper configuration
    - Setting user context
    - Capturing exceptions and messages
    - Adding breadcrumbs for debugging
    - Performance monitoring
    
    Attributes:
        is_initialized: Whether Sentry has been initialized
        is_enabled: Whether crash reporting is enabled
    """
    
    _initialized: bool = False
    _enabled: bool = False
    _dsn: Optional[str] = None
    _environment: str = "production"
    _app_version: str = "1.0.0"
    _user_context: Dict[str, Any] = {}
    
    # Fallback error log for when Sentry is unavailable
    _error_queue: list = []
    _max_queue_size: int = 100
    
    @classmethod
    def initialize(
        cls,
        dsn: Optional[str] = None,
        environment: str = "production",
        app_version: str = "1.0.0",
        sample_rate: float = 1.0,
        traces_sample_rate: float = 0.1,
        debug: bool = False,
        enabled: bool = True,
    ) -> bool:
        """
        Initialize Sentry crash reporting.
        
        Args:
            dsn: Sentry DSN (Data Source Name). If None, reads from SENTRY_DSN env var
            environment: Environment name (production, staging, development)
            app_version: Application version string
            sample_rate: Error sample rate (0.0 to 1.0)
            traces_sample_rate: Performance traces sample rate (0.0 to 1.0)
            debug: Enable Sentry debug mode
            enabled: Whether to enable crash reporting
        
        Returns:
            True if initialized successfully, False otherwise
        
        Example:
            >>> CrashReporter.initialize(
            ...     dsn="https://xxx@sentry.io/123",
            ...     environment="production"
            ... )
            True
        """
        if cls._initialized:
            logger.warning("Crash reporter already initialized")
            return True
        
        cls._environment = environment
        cls._app_version = app_version
        cls._enabled = enabled
        
        if not enabled:
            logger.info("Crash reporting is disabled")
            cls._initialized = True
            return True
        
        if not SENTRY_AVAILABLE:
            logger.warning("Sentry SDK not available - crash reporting disabled")
            cls._initialized = True
            return False
        
        # Get DSN from parameter or environment
        cls._dsn = dsn or os.environ.get("SENTRY_DSN")
        
        if not cls._dsn:
            logger.info("No Sentry DSN provided - crash reporting disabled")
            cls._enabled = False
            cls._initialized = True
            return False
        
        try:
            # Configure Sentry integrations
            integrations = [
                LoggingIntegration(
                    level=logging.INFO,        # Capture info and above as breadcrumbs
                    event_level=logging.ERROR  # Send errors as events
                ),
                ThreadingIntegration(propagate_hub=True),
            ]
            
            # Initialize Sentry
            sentry_sdk.init(
                dsn=cls._dsn,
                environment=environment,
                release=f"16score-ai@{app_version}",
                sample_rate=sample_rate,
                traces_sample_rate=traces_sample_rate,
                integrations=integrations,
                debug=debug,
                
                # Additional options
                attach_stacktrace=True,
                send_default_pii=False,  # Don't send PII by default
                max_breadcrumbs=50,
                
                # Before send hook to filter/modify events
                before_send=cls._before_send,
            )
            
            # Set default tags
            sentry_sdk.set_tag("platform", sys.platform)
            sentry_sdk.set_tag("python_version", sys.version.split()[0])
            
            cls._initialized = True
            cls._enabled = True
            
            logger.info(f"Crash reporter initialized (env: {environment})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Sentry: {e}")
            cls._initialized = True
            cls._enabled = False
            return False
    
    @classmethod
    def _before_send(cls, event: Dict, hint: Dict) -> Optional[Dict]:
        """
        Filter or modify events before sending to Sentry.
        
        Args:
            event: The event dictionary
            hint: Additional hint information
        
        Returns:
            Modified event or None to drop the event
        """
        # Add timestamp
        event.setdefault("extra", {})["local_time"] = datetime.now().isoformat()
        
        # Filter out certain exceptions if needed
        if "exc_info" in hint:
            exc_type, exc_value, tb = hint["exc_info"]
            
            # Skip keyboard interrupts
            if isinstance(exc_value, KeyboardInterrupt):
                return None
            
            # Skip system exit
            if isinstance(exc_value, SystemExit):
                return None
        
        return event
    
    @classmethod
    def is_initialized(cls) -> bool:
        """Check if crash reporter is initialized."""
        return cls._initialized
    
    @classmethod
    def is_enabled(cls) -> bool:
        """Check if crash reporting is enabled."""
        return cls._enabled and SENTRY_AVAILABLE
    
    @classmethod
    def set_user(
        cls,
        user_id: Optional[str] = None,
        email: Optional[str] = None,
        username: Optional[str] = None,
        **extra: Any
    ) -> None:
        """
        Set user context for crash reports.
        
        Args:
            user_id: Unique user identifier
            email: User email address
            username: Username
            **extra: Additional user attributes
        
        Example:
            >>> CrashReporter.set_user(
            ...     user_id="123",
            ...     email="user@example.com",
            ...     organization="Acme Corp"
            ... )
        """
        user_data = {
            "id": user_id,
            "email": email,
            "username": username,
            **extra
        }
        
        # Remove None values
        user_data = {k: v for k, v in user_data.items() if v is not None}
        
        cls._user_context = user_data
        
        if cls._enabled and SENTRY_AVAILABLE:
            sentry_sdk.set_user(user_data if user_data else None)
            logger.debug(f"User context set: {user_id}")
    
    @classmethod
    def clear_user(cls) -> None:
        """Clear user context."""
        cls._user_context = {}
        if cls._enabled and SENTRY_AVAILABLE:
            sentry_sdk.set_user(None)
    
    @classmethod
    def set_tag(cls, key: str, value: str) -> None:
        """
        Set a tag for all future events.
        
        Args:
            key: Tag name
            value: Tag value
        
        Example:
            >>> CrashReporter.set_tag("tournament_id", "123")
        """
        if cls._enabled and SENTRY_AVAILABLE:
            sentry_sdk.set_tag(key, value)
    
    @classmethod
    def set_context(cls, name: str, data: Dict[str, Any]) -> None:
        """
        Set additional context for crash reports.
        
        Args:
            name: Context name (e.g., "match", "camera")
            data: Context data dictionary
        
        Example:
            >>> CrashReporter.set_context("match", {
            ...     "match_id": "456",
            ...     "tournament": "Championship"
            ... })
        """
        if cls._enabled and SENTRY_AVAILABLE:
            sentry_sdk.set_context(name, data)
    
    @classmethod
    def add_breadcrumb(
        cls,
        message: str,
        category: str = "default",
        level: str = "info",
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a breadcrumb for debugging.
        
        Breadcrumbs are trail of events leading to an error.
        
        Args:
            message: Breadcrumb message
            category: Category (ui, http, navigation, etc.)
            level: Level (debug, info, warning, error, critical)
            data: Additional data
        
        Example:
            >>> CrashReporter.add_breadcrumb(
            ...     "User clicked login button",
            ...     category="ui",
            ...     data={"email": "user@example.com"}
            ... )
        """
        if cls._enabled and SENTRY_AVAILABLE:
            sentry_sdk.add_breadcrumb(
                message=message,
                category=category,
                level=level,
                data=data or {}
            )
    
    @classmethod
    def capture_exception(
        cls,
        exception: Optional[BaseException] = None,
        **extra: Any
    ) -> Optional[str]:
        """
        Capture and report an exception.
        
        Args:
            exception: Exception to capture (uses current exception if None)
            **extra: Additional context data
        
        Returns:
            Event ID if captured, None otherwise
        
        Example:
            >>> try:
            ...     risky_operation()
            ... except Exception as e:
            ...     event_id = CrashReporter.capture_exception(e)
        """
        if cls._enabled and SENTRY_AVAILABLE:
            with sentry_sdk.push_scope() as scope:
                for key, value in extra.items():
                    scope.set_extra(key, value)
                
                event_id = sentry_sdk.capture_exception(exception)
                logger.debug(f"Exception captured: {event_id}")
                return event_id
        else:
            # Queue error for later or log it
            cls._queue_error(exception, extra)
            return None
    
    @classmethod
    def capture_message(
        cls,
        message: str,
        level: str = "info",
        **extra: Any
    ) -> Optional[str]:
        """
        Capture and report a message.
        
        Args:
            message: Message to capture
            level: Level (debug, info, warning, error, fatal)
            **extra: Additional context data
        
        Returns:
            Event ID if captured, None otherwise
        
        Example:
            >>> CrashReporter.capture_message(
            ...     "User performed unusual action",
            ...     level="warning",
            ...     action="deleted_all_data"
            ... )
        """
        if cls._enabled and SENTRY_AVAILABLE:
            with sentry_sdk.push_scope() as scope:
                for key, value in extra.items():
                    scope.set_extra(key, value)
                
                event_id = sentry_sdk.capture_message(message, level=level)
                logger.debug(f"Message captured: {event_id}")
                return event_id
        return None
    
    @classmethod
    def _queue_error(cls, exception: Optional[BaseException], extra: Dict) -> None:
        """Queue error when Sentry is unavailable."""
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "exception": str(exception) if exception else None,
            "traceback": traceback.format_exc() if exception else None,
            "extra": extra,
            "user": cls._user_context.copy()
        }
        
        cls._error_queue.append(error_info)
        
        # Limit queue size
        if len(cls._error_queue) > cls._max_queue_size:
            cls._error_queue.pop(0)
        
        logger.debug(f"Error queued (Sentry unavailable): {exception}")
    
    @classmethod
    def get_queued_errors(cls) -> list:
        """Get list of queued errors (when Sentry was unavailable)."""
        return cls._error_queue.copy()
    
    @classmethod
    def clear_queued_errors(cls) -> None:
        """Clear queued errors."""
        cls._error_queue.clear()
    
    @classmethod
    def flush(cls, timeout: float = 2.0) -> None:
        """
        Flush pending events to Sentry.
        
        Call this before app exit to ensure all events are sent.
        
        Args:
            timeout: Maximum seconds to wait
        """
        if cls._enabled and SENTRY_AVAILABLE:
            sentry_sdk.flush(timeout=timeout)
            logger.debug("Crash reporter flushed")


def capture_exceptions(func: Callable) -> Callable:
    """
    Decorator to automatically capture exceptions from a function.
    
    Example:
        >>> @capture_exceptions
        ... def risky_operation():
        ...     # If this raises, it will be captured
        ...     raise ValueError("Something went wrong")
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            CrashReporter.capture_exception(e, function=func.__name__)
            raise
    return wrapper


def track_performance(operation_name: str):
    """
    Decorator to track function performance.
    
    Args:
        operation_name: Name of the operation for tracking
    
    Example:
        >>> @track_performance("process_frame")
        ... def process_frame(frame):
        ...     # Processing logic
        ...     pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if CrashReporter.is_enabled() and SENTRY_AVAILABLE:
                with sentry_sdk.start_transaction(op=operation_name, name=func.__name__):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Convenience alias for the class
crash_reporter = CrashReporter


# Module-level convenience functions
def init_crash_reporting(
    dsn: Optional[str] = None,
    environment: str = "production",
    app_version: str = "1.0.0"
) -> bool:
    """Initialize crash reporting with common defaults."""
    return CrashReporter.initialize(
        dsn=dsn,
        environment=environment,
        app_version=app_version
    )


def report_error(exception: BaseException, **context) -> Optional[str]:
    """Quick function to report an error."""
    return CrashReporter.capture_exception(exception, **context)


def report_warning(message: str, **context) -> Optional[str]:
    """Quick function to report a warning message."""
    return CrashReporter.capture_message(message, level="warning", **context)


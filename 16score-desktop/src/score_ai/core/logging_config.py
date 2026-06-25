"""
Advanced Logging Configuration for 16Score AI Application

This module provides a comprehensive logging system with:
- Separate log files by level (info, warning, error)
- Daily rotation with configurable retention
- Correlation IDs for request tracing
- JSON format option for production
- Colored console output for development

Usage:
    from score_ai.core.logging_config import setup_logging, get_logger
    
    # Setup logging (call once at application start)
    setup_logging()
    
    # Get a logger for your module
    logger = get_logger(__name__)
    logger.info("Application started")

Example:
    >>> from score_ai.core.logging_config import get_logger, set_correlation_id
    >>> logger = get_logger(__name__)
    >>> set_correlation_id("req-12345")
    >>> logger.info("Processing request")
    # Output: 2024-01-15 10:30:45 | INFO | [req-12345] | __main__ | Processing request
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import Optional

# Context variable for correlation ID (thread-safe)
_correlation_id: ContextVar[str] = ContextVar('correlation_id', default='')


# =============================================================================
# Constants
# =============================================================================

class LoggingDefaults:
    """Default logging configuration values."""
    
    LOG_DIR: str = "logs"
    LOG_LEVEL: str = "INFO"
    RETENTION_DAYS: int = 30
    MAX_BYTES: int = 10 * 1024 * 1024  # 10MB
    BACKUP_COUNT: int = 30
    
    # Format strings
    CONSOLE_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(correlation_id)s | %(name)s | %(message)s"
    FILE_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(correlation_id)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
    JSON_FORMAT: str = "json"
    
    DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


# =============================================================================
# ANSI Color Codes for Console Output
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    
    # Background colors
    BG_RED = "\033[41m"
    BG_YELLOW = "\033[43m"


# Level to color mapping
LEVEL_COLORS = {
    logging.DEBUG: Colors.DIM + Colors.CYAN,
    logging.INFO: Colors.GREEN,
    logging.WARNING: Colors.YELLOW,
    logging.ERROR: Colors.RED,
    logging.CRITICAL: Colors.BOLD + Colors.BG_RED + Colors.WHITE,
}


# =============================================================================
# Correlation ID Management
# =============================================================================

def generate_correlation_id() -> str:
    """
    Generate a new unique correlation ID.
    
    Returns:
        A unique correlation ID string (UUID4 format, first 8 chars).
    
    Example:
        >>> cid = generate_correlation_id()
        >>> len(cid)
        8
    """
    return str(uuid.uuid4())[:8]


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """
    Set the correlation ID for the current context.
    
    If no ID is provided, generates a new one.
    
    Args:
        correlation_id: Optional correlation ID to set.
    
    Returns:
        The correlation ID that was set.
    
    Example:
        >>> cid = set_correlation_id("req-12345")
        >>> get_correlation_id()
        'req-12345'
    """
    if correlation_id is None:
        correlation_id = generate_correlation_id()
    _correlation_id.set(correlation_id)
    return correlation_id


def get_correlation_id() -> str:
    """
    Get the current correlation ID.
    
    Returns:
        The current correlation ID, or empty string if not set.
    
    Example:
        >>> set_correlation_id("test-123")
        >>> get_correlation_id()
        'test-123'
    """
    return _correlation_id.get()


def clear_correlation_id() -> None:
    """
    Clear the current correlation ID.
    
    Example:
        >>> set_correlation_id("test-123")
        >>> clear_correlation_id()
        >>> get_correlation_id()
        ''
    """
    _correlation_id.set('')


# =============================================================================
# Custom Log Record Filter
# =============================================================================

class CorrelationIdFilter(logging.Filter):
    """
    Logging filter that adds correlation ID to log records.
    
    This filter adds a 'correlation_id' attribute to each log record,
    which can be used in format strings.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add correlation ID to the log record.
        
        Args:
            record: The log record to modify.
        
        Returns:
            Always True (never filters out records).
        """
        correlation_id = get_correlation_id()
        record.correlation_id = f"[{correlation_id}]" if correlation_id else "[-]"
        return True


# =============================================================================
# Custom Formatters
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """
    Formatter that adds colors to console output.
    
    Uses ANSI color codes to colorize log levels for better readability
    in development environments.
    """
    
    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        use_colors: bool = True
    ) -> None:
        """
        Initialize the colored formatter.
        
        Args:
            fmt: Log format string.
            datefmt: Date format string.
            use_colors: Whether to use colors (disable for non-TTY).
        """
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and sys.stdout.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with colors.
        
        Args:
            record: The log record to format.
        
        Returns:
            Formatted and colorized log string.
        """
        # Get the original formatted message
        message = super().format(record)
        
        if not self.use_colors:
            return message
        
        # Get color for this level
        color = LEVEL_COLORS.get(record.levelno, "")
        
        if color:
            # Colorize the level name in the message
            level_name = record.levelname
            colored_level = f"{color}{level_name}{Colors.RESET}"
            message = message.replace(level_name, colored_level, 1)
            
            # Add subtle color to timestamp
            if self.datefmt:
                timestamp = datetime.now().strftime(self.datefmt)
                colored_timestamp = f"{Colors.DIM}{timestamp}{Colors.RESET}"
                message = message.replace(timestamp, colored_timestamp, 1)
        
        return message


class JsonFormatter(logging.Formatter):
    """
    Formatter that outputs log records as JSON.
    
    Useful for production environments where logs are ingested
    by log aggregation systems like ELK, Splunk, or CloudWatch.
    """
    
    def __init__(
        self,
        include_extra: bool = True,
        indent: Optional[int] = None
    ) -> None:
        """
        Initialize the JSON formatter.
        
        Args:
            include_extra: Whether to include extra fields from the record.
            indent: JSON indentation level (None for compact).
        """
        super().__init__()
        self.include_extra = include_extra
        self.indent = indent
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as JSON.
        
        Args:
            record: The log record to format.
        
        Returns:
            JSON-formatted log string.
        """
        log_data: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": get_correlation_id() or None,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if self.include_extra:
            extra_fields = {
                k: v for k, v in record.__dict__.items()
                if k not in (
                    'name', 'msg', 'args', 'created', 'filename', 'funcName',
                    'levelname', 'levelno', 'lineno', 'module', 'msecs',
                    'pathname', 'process', 'processName', 'relativeCreated',
                    'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
                    'message', 'correlation_id'
                )
            }
            if extra_fields:
                log_data["extra"] = extra_fields
        
        return json.dumps(log_data, default=str, indent=self.indent)


# =============================================================================
# Custom Handlers
# =============================================================================

class LevelFileHandler(TimedRotatingFileHandler):
    """
    File handler that only logs messages at a specific level.
    
    Used to create separate log files for different levels
    (e.g., info.log, warning.log, error.log).
    """
    
    def __init__(
        self,
        filename: str,
        level: int,
        when: str = "midnight",
        interval: int = 1,
        backupCount: int = 30,
        encoding: str = "utf-8",
        **kwargs: Any
    ) -> None:
        """
        Initialize the level-specific file handler.
        
        Args:
            filename: Base filename for the log file.
            level: Logging level to filter (only this level is logged).
            when: Rotation interval type ('midnight', 'H', 'D', etc.).
            interval: Rotation interval.
            backupCount: Number of backup files to keep.
            encoding: File encoding.
            **kwargs: Additional arguments for TimedRotatingFileHandler.
        """
        super().__init__(
            filename,
            when=when,
            interval=interval,
            backupCount=backupCount,
            encoding=encoding,
            **kwargs
        )
        self.target_level = level
        self.setLevel(level)
    
    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record only if it matches the target level.
        
        Args:
            record: The log record to potentially emit.
        """
        # Only emit if the record's level exactly matches our target level
        if record.levelno == self.target_level:
            super().emit(record)


class LevelRangeFileHandler(TimedRotatingFileHandler):
    """
    File handler that logs messages at or above a specific level.
    
    Unlike LevelFileHandler, this logs all messages at or above the level.
    """
    
    def __init__(
        self,
        filename: str,
        level: int,
        when: str = "midnight",
        interval: int = 1,
        backupCount: int = 30,
        encoding: str = "utf-8",
        **kwargs: Any
    ) -> None:
        """
        Initialize the level range file handler.
        
        Args:
            filename: Base filename for the log file.
            level: Minimum logging level to include.
            when: Rotation interval type.
            interval: Rotation interval.
            backupCount: Number of backup files to keep.
            encoding: File encoding.
            **kwargs: Additional arguments.
        """
        super().__init__(
            filename,
            when=when,
            interval=interval,
            backupCount=backupCount,
            encoding=encoding,
            **kwargs
        )
        self.setLevel(level)


# =============================================================================
# Logger Factory
# =============================================================================

class LoggerFactory:
    """
    Factory for creating and configuring loggers.
    
    Provides a centralized way to create loggers with consistent configuration.
    """
    
    _initialized: bool = False
    _log_dir: Path = Path(LoggingDefaults.LOG_DIR)
    _use_json: bool = False
    _retention_days: int = LoggingDefaults.RETENTION_DAYS
    _level: int = logging.INFO
    
    @classmethod
    def setup(
        cls,
        log_dir: Optional[str] = None,
        level: str = "INFO",
        use_json: bool = False,
        retention_days: int = 30,
        console_output: bool = True,
        file_output: bool = True
    ) -> None:
        """
        Setup the logging system.
        
        Should be called once at application startup.
        
        Args:
            log_dir: Directory for log files.
            level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
            use_json: Whether to use JSON format for file logs.
            retention_days: Number of days to keep log files.
            console_output: Whether to output to console.
            file_output: Whether to output to files.
        
        Example:
            >>> LoggerFactory.setup(level="DEBUG", use_json=True)
        """
        if cls._initialized:
            return
        
        cls._log_dir = Path(log_dir or LoggingDefaults.LOG_DIR)
        cls._use_json = use_json
        cls._retention_days = retention_days
        cls._level = getattr(logging, level.upper(), logging.INFO)
        
        # Create log directory
        cls._log_dir.mkdir(parents=True, exist_ok=True)
        
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(cls._level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Add correlation ID filter to root logger
        root_logger.addFilter(CorrelationIdFilter())
        
        # Setup console handler
        if console_output:
            cls._setup_console_handler(root_logger)
        
        # Setup file handlers
        if file_output:
            cls._setup_file_handlers(root_logger)
        
        cls._initialized = True
        
        # Log setup completion
        logger = logging.getLogger(__name__)
        logger.info(
            f"Logging initialized: level={level}, json={use_json}, "
            f"retention={retention_days}d, dir={cls._log_dir}"
        )
    
    @classmethod
    def _setup_console_handler(cls, logger: logging.Logger) -> None:
        """Setup console handler with colored output."""
        # Use UTF-8 encoding for console if possible (Windows fix)
        try:
            # Try to set UTF-8 encoding for stdout
            if sys.platform == 'win32':
                import io
                sys.stdout = io.TextIOWrapper(
                    sys.stdout.buffer, 
                    encoding='utf-8', 
                    errors='replace'
                )
        except Exception:
            pass  # If it fails, continue with default encoding
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(cls._level)
        
        formatter = ColoredFormatter(
            fmt=LoggingDefaults.CONSOLE_FORMAT,
            datefmt=LoggingDefaults.DATE_FORMAT,
            use_colors=True
        )
        console_handler.setFormatter(formatter)
        console_handler.addFilter(CorrelationIdFilter())
        
        logger.addHandler(console_handler)
    
    @classmethod
    def _setup_file_handlers(cls, logger: logging.Logger) -> None:
        """Setup file handlers for different log levels."""
        # Create date suffix for log files
        date_suffix = datetime.now().strftime("%Y-%m-%d")
        
        # Formatter based on configuration
        if cls._use_json:
            formatter = JsonFormatter()
        else:
            formatter = logging.Formatter(
                LoggingDefaults.FILE_FORMAT,
                datefmt=LoggingDefaults.DATE_FORMAT
            )
        
        # Define log files for each level
        level_files = [
            (logging.INFO, f"info_{date_suffix}.log"),
            (logging.WARNING, f"warning_{date_suffix}.log"),
            (logging.ERROR, f"error_{date_suffix}.log"),
        ]
        
        for level, filename in level_files:
            if cls._level <= level:
                filepath = cls._log_dir / filename
                handler = LevelFileHandler(
                    str(filepath),
                    level=level,
                    when="midnight",
                    backupCount=cls._retention_days,
                    encoding="utf-8"
                )
                handler.setFormatter(formatter)
                handler.addFilter(CorrelationIdFilter())
                logger.addHandler(handler)
        
        # Also create a combined log file
        combined_path = cls._log_dir / f"app_{date_suffix}.log"
        combined_handler = LevelRangeFileHandler(
            str(combined_path),
            level=cls._level,
            when="midnight",
            backupCount=cls._retention_days,
            encoding="utf-8"
        )
        combined_handler.setFormatter(formatter)
        combined_handler.addFilter(CorrelationIdFilter())
        logger.addHandler(combined_handler)
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger with the specified name.
        
        Args:
            name: Logger name (usually __name__).
        
        Returns:
            Configured logger instance.
        
        Example:
            >>> logger = LoggerFactory.get_logger(__name__)
            >>> logger.info("Hello, world!")
        """
        if not cls._initialized:
            cls.setup()
        
        return logging.getLogger(name)


# =============================================================================
# Convenience Functions
# =============================================================================

def setup_logging(
    log_dir: Optional[str] = None,
    level: str = "INFO",
    use_json: bool = False,
    retention_days: int = 30,
    console_output: bool = True,
    file_output: bool = True
) -> None:
    """
    Setup the logging system.
    
    Convenience function that wraps LoggerFactory.setup().
    Should be called once at application startup.
    
    Args:
        log_dir: Directory for log files (default: 'logs').
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        use_json: Whether to use JSON format for file logs.
        retention_days: Number of days to keep log files.
        console_output: Whether to output to console.
        file_output: Whether to output to files.
    
    Example:
        >>> setup_logging(level="DEBUG", use_json=False, retention_days=7)
    """
    LoggerFactory.setup(
        log_dir=log_dir,
        level=level,
        use_json=use_json,
        retention_days=retention_days,
        console_output=console_output,
        file_output=file_output
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Convenience function that wraps LoggerFactory.get_logger().
    
    Args:
        name: Logger name (usually __name__).
    
    Returns:
        Configured logger instance.
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
    return LoggerFactory.get_logger(name)


# =============================================================================
# Request Context Manager
# =============================================================================

class RequestContext:
    """
    Context manager for request-scoped logging with correlation ID.
    
    Automatically sets and clears correlation ID for a request scope.
    
    Example:
        >>> with RequestContext() as ctx:
        ...     logger.info("Processing request")
        ...     print(ctx.correlation_id)
    """
    
    def __init__(self, correlation_id: Optional[str] = None) -> None:
        """
        Initialize the request context.
        
        Args:
            correlation_id: Optional correlation ID (generates one if not provided).
        """
        self.correlation_id = correlation_id or generate_correlation_id()
    
    def __enter__(self) -> "RequestContext":
        """Enter the context and set correlation ID."""
        set_correlation_id(self.correlation_id)
        return self
    
    def __exit__(self, *args: Any) -> None:
        """Exit the context and clear correlation ID."""
        clear_correlation_id()


# =============================================================================
# Module-level initialization
# =============================================================================

# Create a default logger for this module
_module_logger = logging.getLogger(__name__)


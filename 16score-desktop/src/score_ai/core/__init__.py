"""
Core functionality for the 16Score AI application.

This package contains the core modules that power the application:

Modules:
    - config: Static configuration constants (deprecated, use constants.py)
    - config_manager: Dynamic JSON configuration with env var support
    - api_service: REST API client services
    - security: Security utilities (validation, secure storage, etc.)
    - constants: All application constants
    - exceptions: Custom exception classes

Usage:
    from score_ai.core import Config, config, InputValidator
    from score_ai.core.constants import UIConstants
    from score_ai.core.exceptions import APIError

Example:
    >>> from score_ai.core import config
    >>> config.get('api.backend_url')
    'https://webapi.16score.com'
"""

from __future__ import annotations

from .config import Config
from .config_manager import config, ConfigManager
from .constants import (
    APIConstants,
    GRPCConstants,
    DetectionConstants,
    CameraConstants,
    UIConstants,
    SecurityConstants,
    FileConstants,
    LoggingConstants,
)
from .exceptions import (
    ScoreAIError,
    APIError,
    AuthenticationError,
    AuthorizationError,
    ValidationError,
    ConfigurationError,
    DetectionError,
    GRPCError,
    StorageError,
)
from .security import (
    EnvironmentConfig,
    SecureStorage,
    InputValidator,
    SecurityChecker,
)
from .logging_config import (
    setup_logging,
    get_logger,
    set_correlation_id,
    get_correlation_id,
    clear_correlation_id,
    RequestContext,
)
from .brand_colors import BrandColors
from .theme import Colors, Fonts, Spacing, Gradients, Shadows, Styles

# Network monitoring
from .network_monitor import (
    NetworkMonitor,
    NetworkStatus,
    get_network_monitor,
    is_online,
    check_connection,
    start_monitoring,
    stop_monitoring,
    OfflineError,
    require_online,
)

# Crash reporting
from .crash_reporter import (
    CrashReporter,
    crash_reporter,
    capture_exceptions,
    track_performance,
    init_crash_reporting,
    report_error,
    report_warning,
)

# Auto-update
from .auto_updater import (
    AutoUpdater,
    UpdateInfo,
    UpdateStatus,
    get_updater,
    check_for_updates,
    start_auto_updates,
    stop_auto_updates,
)

__all__ = [
    # Config
    'Config',
    'config',
    'ConfigManager',
    # Constants
    'APIConstants',
    'GRPCConstants',
    'DetectionConstants',
    'CameraConstants',
    'UIConstants',
    'SecurityConstants',
    'FileConstants',
    'LoggingConstants',
    # Exceptions
    'ScoreAIError',
    'APIError',
    'AuthenticationError',
    'AuthorizationError',
    'ValidationError',
    'ConfigurationError',
    'DetectionError',
    'GRPCError',
    'StorageError',
    # Security
    'EnvironmentConfig',
    'SecureStorage',
    'InputValidator',
    'SecurityChecker',
    # Logging
    'setup_logging',
    'get_logger',
    'set_correlation_id',
    'get_correlation_id',
    'clear_correlation_id',
    'RequestContext',
    # Brand Colors
    'BrandColors',
    # Theme
    'Colors',
    'Fonts',
    'Spacing',
    'Gradients',
    'Shadows',
    'Styles',
    # Network Monitoring
    'NetworkMonitor',
    'NetworkStatus',
    'get_network_monitor',
    'is_online',
    'check_connection',
    'start_monitoring',
    'stop_monitoring',
    'OfflineError',
    'require_online',
    # Crash Reporting
    'CrashReporter',
    'crash_reporter',
    'capture_exceptions',
    'track_performance',
    'init_crash_reporting',
    'report_error',
    'report_warning',
    # Auto-Update
    'AutoUpdater',
    'UpdateInfo',
    'UpdateStatus',
    'get_updater',
    'check_for_updates',
    'start_auto_updates',
    'stop_auto_updates',
] 
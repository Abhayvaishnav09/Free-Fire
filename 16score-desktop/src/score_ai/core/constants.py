"""
Constants for 16Score AI Application

This module contains all constant values used throughout the application.
Centralizing constants makes the codebase more maintainable and prevents
hardcoded values scattered across modules.

Usage:
    from score_ai.core.constants import APIConstants, UIConstants
    
    timeout = APIConstants.DEFAULT_TIMEOUT
    color = UIConstants.PRIMARY_COLOR
"""

from __future__ import annotations


class APIConstants:
    """API-related constants"""
    
    # Default URLs
    DEFAULT_BACKEND_URL: str = "http://192.168.1.11:5006"
    DEFAULT_GRPC_SERVER: str = "127.0.0.1:50050"
    
    # Timeouts (in seconds)
    DEFAULT_TIMEOUT: int = 30
    CONNECTION_TIMEOUT: int = 10
    READ_TIMEOUT: int = 30
    
    # HTTP Status Codes
    HTTP_OK: int = 200
    HTTP_CREATED: int = 201
    HTTP_BAD_REQUEST: int = 400
    HTTP_UNAUTHORIZED: int = 401
    HTTP_FORBIDDEN: int = 403
    HTTP_NOT_FOUND: int = 404
    HTTP_SERVER_ERROR: int = 500
    
    # Rate limiting
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0  # seconds
    
    # Headers
    CONTENT_TYPE_JSON: str = "application/json"
    AUTH_HEADER: str = "Authorization"
    BEARER_PREFIX: str = "Bearer"
    API_KEY_HEADER: str = "X-API-Key"


class GRPCConstants:
    """gRPC-related constants"""
    
    # Default server
    DEFAULT_SERVER: str = "127.0.0.1:50050"
    
    # Metadata keys
    API_KEY_METADATA: str = "api-key"
    
    # Timeouts (in seconds)
    DEFAULT_TIMEOUT: float = 30.0
    CONNECT_TIMEOUT: float = 10.0


class DetectionConstants:
    """Detection-related constants"""
    
    # Frame processing
    DEFAULT_FRAME_INTERVAL: float = 0.5  # seconds
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.20
    
    # SIFT parameters
    SIFT_MIN_MATCHES: int = 10
    SIFT_MATCH_RATIO: float = 0.75
    SIFT_FEATURES: int = 400
    
    # Queue settings
    DEFAULT_QUEUE_SIZE: int = 75
    
    # Image processing
    DEFAULT_IMAGE_QUALITY: int = 85
    DEFAULT_RESIZE_FACTOR: float = 0.75
    TARGET_WIDTH: int = 1280
    TARGET_HEIGHT: int = 720
    
    # Similarity thresholds
    SIMILARITY_HIGH: float = 0.90
    SIMILARITY_LOW: float = 0.50
    DUPLICATE_CHECK_COUNT: int = 5
    
    # Class IDs
    KILL_BLOCK_CLASS_ID: int = 12
    KILL_BLOCK_CLASS_NAME: str = "kill-block"


class CameraConstants:
    """Camera-related constants"""
    
    DEFAULT_WIDTH: int = 1920
    DEFAULT_HEIGHT: int = 1080
    DEFAULT_INDEX: int = 1
    
    # Common resolutions
    RESOLUTION_720P: tuple[int, int] = (1280, 720)
    RESOLUTION_1080P: tuple[int, int] = (1920, 1080)
    RESOLUTION_1440P: tuple[int, int] = (2560, 1440)
    RESOLUTION_4K: tuple[int, int] = (3840, 2160)


class UIConstants:
    """UI-related constants"""
    
    # Application
    APP_NAME: str = "16Score AI"
    APP_VERSION: str = "1.0.0"
    
    # Window dimensions
    DEFAULT_WINDOW_WIDTH: int = 1200
    DEFAULT_WINDOW_HEIGHT: int = 800
    LOGIN_PANEL_WIDTH: int = 480
    LOGIN_PANEL_MIN_WIDTH: int = 480
    LOGIN_PANEL_MAX_WIDTH: int = 650
    
    # Colors
    PRIMARY_COLOR: str = "#2186eb"
    SECONDARY_COLOR: str = "#1761b6"
    ERROR_COLOR: str = "#e57373"
    ERROR_COLOR_ALT: str = "#ff4d4f"
    SUCCESS_COLOR: str = "#4caf50"
    WARNING_COLOR: str = "#ff9800"
    BACKGROUND_COLOR: str = "#f8fbff"
    CARD_BACKGROUND: str = "white"
    DARK_BACKGROUND: str = "#1c2a3a"
    INPUT_BACKGROUND: str = "#2b3e50"
    INPUT_BORDER: str = "#4f5b6a"
    TEXT_LIGHT: str = "#b0c4de"
    
    # Font sizes
    FONT_SIZE_SMALL: int = 12
    FONT_SIZE_NORMAL: int = 14
    FONT_SIZE_MEDIUM: int = 16
    FONT_SIZE_LARGE: int = 20
    FONT_SIZE_XLARGE: int = 24
    FONT_SIZE_HEADING: int = 36
    
    # Spacing
    SPACING_SMALL: int = 8
    SPACING_NORMAL: int = 16
    SPACING_LARGE: int = 24
    SPACING_XLARGE: int = 32
    
    # Border radius
    BORDER_RADIUS_SMALL: int = 4
    BORDER_RADIUS_NORMAL: int = 8
    BORDER_RADIUS_LARGE: int = 12
    BORDER_RADIUS_XLARGE: int = 20
    
    # Animation durations (ms)
    ANIMATION_FAST: int = 150
    ANIMATION_NORMAL: int = 300
    ANIMATION_SLOW: int = 500


class SecurityConstants:
    """Security-related constants"""
    
    # Keyring
    KEYRING_SERVICE: str = "16ScoreAI"
    
    # Validation
    MIN_PASSWORD_LENGTH: int = 4
    MAX_PASSWORD_LENGTH: int = 128
    MAX_EMAIL_LENGTH: int = 254
    
    # Patterns
    EMAIL_PATTERN: str = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    DANGEROUS_CHARS_PATTERN: str = r'[<>"\';(){}]'


class FileConstants:
    """File-related constants"""
    
    # Directories
    GAME_LOGS_DIR: str = "game_logs"
    DEBUG_DIR: str = "debug"
    MATCH_POSITIONS_DIR: str = "match_positions"
    
    # Files
    RESULT_FILE: str = "result.txt"
    OCR_LOG_FILE: str = "ocr_detections.log"
    KILLFEED_FILE: str = "killfeed_detections.json"
    CONFIG_FILE: str = "config.json"
    ENV_FILE: str = ".env"
    
    # Extensions
    JSON_EXT: str = ".json"
    LOG_EXT: str = ".log"
    IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".gif", ".webp")


class LoggingConstants:
    """Logging-related constants"""
    
    # Log levels
    LEVEL_DEBUG: str = "DEBUG"
    LEVEL_INFO: str = "INFO"
    LEVEL_WARNING: str = "WARNING"
    LEVEL_ERROR: str = "ERROR"
    LEVEL_CRITICAL: str = "CRITICAL"
    
    # Format
    DEFAULT_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    SIMPLE_FORMAT: str = "%(levelname)s - %(message)s"
    DETAILED_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"


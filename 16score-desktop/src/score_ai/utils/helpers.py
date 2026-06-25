"""
Helper Utilities for 16Score AI Application

This module provides common helper functions used throughout the application,
including resource path resolution, API requests, login history management,
and UI stylesheet generation.

Usage:
    from score_ai.utils.helpers import get_resource_path, save_login_history
    
    path = get_resource_path("images/logo.png")
    save_login_history("user@example.com", "user123")

Example:
    >>> from score_ai.utils.helpers import validate_email
    >>> validate_email("test@example.com")
    True
    >>> validate_email("invalid-email")
    False
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any

import requests

from score_ai.core.config import Config
from score_ai.core.config_manager import config
from score_ai.core.constants import (
    APIConstants,
    SecurityConstants,
    UIConstants,
)
from score_ai.core.exceptions import (
    APIError,
    ConnectionError as AppConnectionError,
    TimeoutError as AppTimeoutError,
)
from score_ai.core.logging_config import get_logger

if TYPE_CHECKING:
    from typing import Optional

# Get logger using the centralized logging system
logger = get_logger(__name__)


def get_resource_path(relative_path: str) -> str:
    """
    Get the absolute path to a resource file.
    
    Works for both development mode and PyInstaller frozen executables.
    In PyInstaller mode, resources are extracted to a temp directory.
    
    Args:
        relative_path: Path relative to the application root or PyInstaller bundle.
    
    Returns:
        Absolute path to the resource.
    
    Example:
        >>> get_resource_path("images/logo.png")
        '/path/to/app/images/logo.png'
    
    Note:
        In PyInstaller mode, the base path is sys._MEIPASS.
        In development mode, it's the current working directory.
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS  # type: ignore[attr-defined]
        logger.debug(f"PyInstaller mode: base_path = {base_path}")
    except AttributeError:
        base_path = os.path.abspath(".")
        logger.debug(f"Development mode: base_path = {base_path}")
    
    full_path = os.path.join(base_path, relative_path)
    logger.debug(f"Resource path: {relative_path} -> {full_path}")
    
    return full_path


def validate_email(email: str) -> bool:
    """
    Validate an email address format.
    
    Uses a regex pattern to check if the email follows standard format.
    
    Args:
        email: The email address to validate.
    
    Returns:
        True if the email format is valid, False otherwise.
    
    Example:
        >>> validate_email("user@example.com")
        True
        >>> validate_email("invalid.email")
        False
        >>> validate_email("")
        False
    """
    if not email:
        return False
    
    pattern = SecurityConstants.EMAIL_PATTERN
    return re.match(pattern, email) is not None


def make_api_request(
    method: str,
    endpoint: str,
    data: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    timeout: Optional[int] = None
) -> requests.Response:
    """
    Make an HTTP API request with error handling.
    
    Constructs the full URL from the endpoint and makes the request
    with proper error handling for timeouts and connection errors.
    
    Args:
        method: HTTP method ('GET' or 'POST').
        endpoint: API endpoint path.
        data: Request body data for POST requests.
        headers: Optional request headers.
        timeout: Request timeout in seconds (default from config).
    
    Returns:
        The requests.Response object.
    
    Raises:
        AppTimeoutError: If the request times out.
        AppConnectionError: If unable to connect to the server.
        APIError: For other API errors.
        ValueError: If an unsupported HTTP method is provided.
    
    Example:
        >>> response = make_api_request('GET', 'User/Profile')
        >>> response.status_code
        200
    """
    url = Config.get_api_endpoint(endpoint)
    request_timeout = timeout or config.get('api.timeout', APIConstants.DEFAULT_TIMEOUT)
    
    try:
        if method.upper() == 'GET':
            response = requests.get(url, headers=headers, timeout=request_timeout)
        elif method.upper() == 'POST':
            response = requests.post(url, json=data, headers=headers, timeout=request_timeout)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        logger.info(f"API {method} {endpoint}: {response.status_code}")
        return response
        
    except requests.exceptions.Timeout:
        logger.error(f"API timeout for {endpoint}")
        raise AppTimeoutError(
            message="Request timed out. Please check your connection.",
            timeout=request_timeout,
            endpoint=endpoint
        )
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error for {endpoint}")
        raise AppConnectionError(
            message="Unable to connect to server. Please check your internet connection.",
            endpoint=endpoint
        )
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"API error for {endpoint}: {str(e)}")
        raise APIError(message=str(e), endpoint=endpoint)


def save_login_history(email: str, user_id: Optional[str] = None) -> bool:
    """
    Save login history to a local file.
    
    Appends a new login entry to the history file with timestamp.
    Creates the file if it doesn't exist.
    
    Args:
        email: User's email address.
        user_id: Optional user ID from the server.
    
    Returns:
        True if saved successfully, False otherwise.
    
    Example:
        >>> save_login_history("user@example.com", "usr_123")
        True
    """
    try:
        login_data = {
            "email": email,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": user_id
        }
        
        # Load existing data
        history_file = Config.LOGIN_HISTORY_FILE
        data: list[dict[str, Any]] = []
        
        if os.path.exists(history_file):
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        data = json.loads(content)
            except (json.JSONDecodeError, IOError):
                data = []
        
        # Add new entry
        data.append(login_data)
        
        # Save to file
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        
        logger.info(f"Login history saved for {email}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save login history: {e}")
        return False


def get_login_history() -> list[dict[str, Any]]:
    """
    Get login history from the local file.
    
    Returns:
        List of login history entries, or empty list if none exist.
    
    Example:
        >>> history = get_login_history()
        >>> len(history)
        5
        >>> history[0]['email']
        'user@example.com'
    """
    try:
        history_file = Config.LOGIN_HISTORY_FILE
        if os.path.exists(history_file):
            with open(history_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    return json.loads(content)
    except Exception as e:
        logger.error(f"Failed to load login history: {e}")
    
    return []


def create_stylesheet(component: str, **kwargs: Any) -> str:
    """
    Create a Qt stylesheet for common UI components.
    
    Provides consistent styling across the application by returning
    pre-defined stylesheets for common component types.
    
    Args:
        component: Component type ('input', 'button', or 'card').
        **kwargs: Additional style parameters (reserved for future use).
    
    Returns:
        Qt stylesheet string, or empty string if component not found.
    
    Example:
        >>> stylesheet = create_stylesheet('button')
        >>> 'background-color' in stylesheet
        True
    """
    base_styles = {
        'input': f"""
            QLineEdit {{
                border: 1.5px solid #e0e0e0;
                border-radius: {UIConstants.BORDER_RADIUS_NORMAL}px;
                padding-left: 14px;
                font-size: {UIConstants.FONT_SIZE_MEDIUM}px;
                background: #f6f8fa;
            }}
            QLineEdit:focus {{
                border: 1.5px solid {UIConstants.PRIMARY_COLOR};
                background: #fff;
            }}
        """,
        'button': f"""
            QPushButton {{
                background-color: {UIConstants.PRIMARY_COLOR};
                color: white;
                border-radius: {UIConstants.BORDER_RADIUS_NORMAL}px;
                font-size: {UIConstants.FONT_SIZE_LARGE}px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {UIConstants.SECONDARY_COLOR};
            }}
            QPushButton:disabled {{
                background-color: #cccccc;
            }}
        """,
        'card': f"""
            background: {UIConstants.CARD_BACKGROUND}; 
            border-radius: {UIConstants.BORDER_RADIUS_LARGE}px; 
            border: 1px solid #e0e0e0;
        """
    }
    
    return base_styles.get(component, "")


def format_timestamp(dt: Optional[datetime] = None, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format a datetime object as a string.
    
    Args:
        dt: Datetime object to format. Defaults to current time.
        fmt: Format string (strftime format).
    
    Returns:
        Formatted datetime string.
    
    Example:
        >>> format_timestamp()
        '2024-01-15 10:30:45'
    """
    if dt is None:
        dt = datetime.now()
    return dt.strftime(fmt)


def ensure_directory(path: str) -> bool:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure exists.
    
    Returns:
        True if directory exists or was created, False on error.
    
    Example:
        >>> ensure_directory("logs/debug")
        True
    """
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        return False

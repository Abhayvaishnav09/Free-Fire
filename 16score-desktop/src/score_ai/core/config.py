"""
Static Application Configuration Constants

This module contains static configuration values that don't change at runtime.
For dynamic configuration loaded from config.json, use config_manager.py instead.

Note:
    This module is being deprecated in favor of constants.py for static values
    and config_manager.py for dynamic values. New code should use those modules.

Usage:
    from score_ai.core.config import Config
    
    # Static values (don't change)
    print(Config.APP_NAME)
    print(Config.PRIMARY_COLOR)
    
    # Dynamic values (from config.json) - use config_manager instead
    from score_ai.core.config_manager import config
    print(config.get('api.backend_url'))

Example:
    >>> from score_ai.core.config import Config
    >>> Config.get_backend_url()
    'http://192.168.1.11:5006'
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .config_manager import config
from .constants import (
    APIConstants,
    UIConstants,
    FileConstants,
)

if TYPE_CHECKING:
    pass


class Config:
    """
    Static application configuration constants.
    
    This class provides access to static configuration values and helper methods
    for retrieving dynamic configuration from config.json.
    
    Attributes:
        APP_NAME: Application display name.
        APP_VERSION: Current application version.
        WINDOW_WIDTH: Default window width in pixels.
        WINDOW_HEIGHT: Default window height in pixels.
        LOGIN_PANEL_WIDTH: Login panel width in pixels.
        PRIMARY_COLOR: Primary brand color (hex).
        SECONDARY_COLOR: Secondary brand color (hex).
        ERROR_COLOR: Error state color (hex).
        BACKGROUND_COLOR: Main background color (hex).
        CARD_BACKGROUND: Card/panel background color.
    
    Example:
        >>> Config.APP_NAME
        '16Score AI'
        >>> Config.get_backend_url()
        'http://192.168.1.11:5006'
    """
    
    # Application Settings (from constants)
    APP_NAME: str = UIConstants.APP_NAME
    APP_VERSION: str = UIConstants.APP_VERSION
    
    # UI Settings (from constants)
    WINDOW_WIDTH: int = UIConstants.DEFAULT_WINDOW_WIDTH
    WINDOW_HEIGHT: int = UIConstants.DEFAULT_WINDOW_HEIGHT
    LOGIN_PANEL_WIDTH: int = UIConstants.LOGIN_PANEL_WIDTH
    
    # File Paths
    LOGIN_HISTORY_FILE: str = os.path.join(os.path.expanduser("~"), "login_history.json")
    LOGO_PATH: str = "src/score_ai/resources/images/TMS_logo.jpg"
    AVATAR_PATH: str = "login_image.jpg"
    KILL_FEED: str = "src/score_ai/resources/images/kill_feed.svg"
    
    # Colors (from constants)
    PRIMARY_COLOR: str = UIConstants.PRIMARY_COLOR
    SECONDARY_COLOR: str = UIConstants.SECONDARY_COLOR
    ERROR_COLOR: str = UIConstants.ERROR_COLOR
    BACKGROUND_COLOR: str = UIConstants.BACKGROUND_COLOR
    CARD_BACKGROUND: str = UIConstants.CARD_BACKGROUND
    
    @classmethod
    def get_backend_url(cls) -> str:
        """
        Get the backend API URL from configuration.
        
        Returns:
            The backend API URL string.
        
        Example:
            >>> Config.get_backend_url()
            'http://192.168.1.11:5006'
        """
        return config.get('api.backend_url', APIConstants.DEFAULT_BACKEND_URL)
    
    @classmethod
    def get_api_endpoint(cls, endpoint: str) -> str:
        """
        Construct a full API endpoint URL.
        
        Combines the backend URL with the given endpoint path.
        
        Args:
            endpoint: The API endpoint path (e.g., 'User/Login').
        
        Returns:
            The complete API URL.
        
        Example:
            >>> Config.get_api_endpoint('User/Login')
            'http://192.168.1.11:5006/User/Login'
        """
        backend_url = cls.get_backend_url()
        return f"{backend_url.rstrip('/')}/{endpoint.lstrip('/')}"
    
    @classmethod
    def get_api_timeout(cls) -> int:
        """
        Get the API request timeout in seconds.
        
        Returns:
            Timeout value in seconds.
        
        Example:
            >>> Config.get_api_timeout()
            30
        """
        return config.get('api.timeout', APIConstants.DEFAULT_TIMEOUT)
    
    @classmethod
    def get_grpc_server(cls) -> str:
        """
        Get the gRPC killfeed detection server address.
        
        Returns:
            Server address in 'host:port' format.
        
        Example:
            >>> Config.get_grpc_server()
            '127.0.0.1:50050'
        """
        return config.get('grpc.killfeed_server', APIConstants.DEFAULT_GRPC_SERVER)

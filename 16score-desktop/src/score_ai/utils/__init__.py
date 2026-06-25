"""
Utility functions for the 16Score AI application.

Modules:
    - helpers: Common helper functions (resource paths, API requests, etc.)
    - responsive_utils: Responsive UI utilities for different screen sizes

Usage:
    from score_ai.utils import get_resource_path, ResponsiveUtils
    
    path = get_resource_path("images/logo.png")
    utils = ResponsiveUtils()
"""

from __future__ import annotations

from .helpers import (
    get_resource_path,
    validate_email,
    make_api_request,
    save_login_history,
    get_login_history,
    create_stylesheet,
    format_timestamp,
    ensure_directory,
)
from .responsive_utils import (
    ResponsiveUtils,
    ResponsiveWidget,
    apply_responsive_style,
)

__all__ = [
    # Helpers
    'get_resource_path',
    'validate_email',
    'make_api_request',
    'save_login_history',
    'get_login_history',
    'create_stylesheet',
    'format_timestamp',
    'ensure_directory',
    # Responsive
    'ResponsiveUtils',
    'ResponsiveWidget',
    'apply_responsive_style',
] 
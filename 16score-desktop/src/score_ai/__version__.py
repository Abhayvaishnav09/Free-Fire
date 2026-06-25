"""
Version information for 16Score AI Application.

This module provides version information that can be imported
throughout the application and read by packaging tools.

Usage:
    from score_ai import __version__
    print(__version__)  # "1.0.0"

The version follows Semantic Versioning (SemVer):
    MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]

Examples:
    1.0.0       - First stable release
    1.1.0       - Added new features (backward compatible)
    1.1.1       - Bug fixes only
    2.0.0-beta  - Major version pre-release
"""

# Version components
MAJOR = 1
MINOR = 0
PATCH = 0
PRERELEASE = ""  # e.g., "alpha", "beta", "rc1"
BUILD = ""       # e.g., "build.123"

# Construct version string
__version__ = f"{MAJOR}.{MINOR}.{PATCH}"

if PRERELEASE:
    __version__ += f"-{PRERELEASE}"

if BUILD:
    __version__ += f"+{BUILD}"

# Additional metadata
__version_info__ = (MAJOR, MINOR, PATCH, PRERELEASE, BUILD)
__app_name__ = "16Score-AI"
__author__ = "16Score Team"
__copyright__ = "Copyright 2024-2025 16Score"
__license__ = "Proprietary"


def get_version() -> str:
    """Get the current version string."""
    return __version__


def get_version_info() -> tuple:
    """Get version as a tuple (major, minor, patch, prerelease, build)."""
    return __version_info__


def get_full_version() -> str:
    """Get full version with app name."""
    return f"{__app_name__} v{__version__}"


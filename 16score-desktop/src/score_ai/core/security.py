"""
Security Utilities for 16Score AI Application

This module provides security-related utilities including:
- Environment variable management for secrets
- Secure token storage using system keyring
- Input validation and sanitization
- HTTPS enforcement checks

Usage:
    from score_ai.core.security import InputValidator, SecureStorage
    
    # Validate user input
    is_valid, error = InputValidator.validate_email("user@example.com")
    
    # Store tokens securely
    SecureStorage.store_token("user@example.com", "auth_token_here")

Example:
    >>> InputValidator.validate_email("test@example.com")
    (True, '')
    >>> InputValidator.validate_email("invalid")
    (False, 'Invalid email format')
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Any

from .constants import SecurityConstants

if TYPE_CHECKING:
    from typing import Optional

logger = logging.getLogger(__name__)

# Try to import optional security packages
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    logger.warning("python-dotenv not installed. Environment file loading disabled.")

try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False
    logger.warning("keyring not installed. Secure token storage disabled.")


# Application identifier for keyring (from constants)
KEYRING_SERVICE = SecurityConstants.KEYRING_SERVICE


class EnvironmentConfig:
    """Manages environment-based configuration for secrets"""
    
    _loaded = False
    
    @classmethod
    def load(cls, env_file: str = ".env") -> bool:
        """
        Load environment variables from .env file
        
        Args:
            env_file: Path to the .env file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if cls._loaded:
            return True
            
        if not DOTENV_AVAILABLE:
            logger.warning("python-dotenv not available, skipping .env loading")
            return False
            
        if os.path.exists(env_file):
            load_dotenv(env_file)
            cls._loaded = True
            logger.info(f"Loaded environment from {env_file}")
            return True
        else:
            logger.debug(f"No .env file found at {env_file}")
            return False
    
    @classmethod
    def get(cls, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get an environment variable with optional default
        
        Args:
            key: Environment variable name
            default: Default value if not set
            
        Returns:
            The environment variable value or default
        """
        # Ensure env is loaded
        if not cls._loaded:
            cls.load()
        return os.environ.get(key, default)
    
    @classmethod
    def get_api_url(cls) -> str:
        """Get API URL from environment or default"""
        return cls.get("SCORE_API_URL", "http://192.168.1.11:5006")
    
    @classmethod
    def get_grpc_server(cls) -> str:
        """Get gRPC server address from environment or default"""
        return cls.get("SCORE_GRPC_SERVER", "127.0.0.1:50050")
    
    @classmethod
    def get_grpc_api_key(cls) -> str:
        """Get gRPC API key from environment"""
        return cls.get("SCORE_GRPC_API_KEY", "")
    
    @classmethod
    def get_grpc_use_tls(cls) -> bool:
        """Check if TLS should be used for gRPC"""
        return cls.get("SCORE_GRPC_USE_TLS", "false").lower() == "true"
    
    @classmethod
    def is_debug_mode(cls) -> bool:
        """Check if debug mode is enabled"""
        return cls.get("SCORE_DEBUG", "false").lower() == "true"
    
    @classmethod
    def enforce_https(cls) -> bool:
        """Check if HTTPS enforcement is enabled"""
        return cls.get("SCORE_ENFORCE_HTTPS", "true").lower() == "true"


class SecureStorage:
    """Secure storage for sensitive data using system keyring"""
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if secure storage is available"""
        return KEYRING_AVAILABLE
    
    @classmethod
    def store_token(cls, username: str, token: str) -> bool:
        """
        Securely store an authentication token
        
        Args:
            username: The username/email associated with the token
            token: The authentication token to store
            
        Returns:
            True if stored successfully, False otherwise
        """
        if not KEYRING_AVAILABLE:
            logger.warning("Keyring not available, token not stored securely")
            return False
            
        try:
            keyring.set_password(KEYRING_SERVICE, username, token)
            logger.debug(f"Token stored securely for {username}")
            return True
        except Exception as e:
            logger.error(f"Failed to store token: {e}")
            return False
    
    @classmethod
    def get_token(cls, username: str) -> Optional[str]:
        """
        Retrieve a stored authentication token
        
        Args:
            username: The username/email to retrieve token for
            
        Returns:
            The stored token or None if not found
        """
        if not KEYRING_AVAILABLE:
            return None
            
        try:
            return keyring.get_password(KEYRING_SERVICE, username)
        except Exception as e:
            logger.error(f"Failed to retrieve token: {e}")
            return None
    
    @classmethod
    def delete_token(cls, username: str) -> bool:
        """
        Delete a stored authentication token
        
        Args:
            username: The username/email to delete token for
            
        Returns:
            True if deleted successfully, False otherwise
        """
        if not KEYRING_AVAILABLE:
            return False
            
        try:
            keyring.delete_password(KEYRING_SERVICE, username)
            logger.debug(f"Token deleted for {username}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete token: {e}")
            return False


class InputValidator:
    """
    Input validation and sanitization utilities.
    
    Provides static methods for validating user input such as emails,
    passwords, and URLs. Also includes input sanitization.
    
    Example:
        >>> InputValidator.validate_email("test@example.com")
        (True, '')
        >>> InputValidator.validate_password("short")
        (False, 'Password must be at least 6 characters')
    """
    
    # Email regex pattern (from constants)
    EMAIL_PATTERN = re.compile(SecurityConstants.EMAIL_PATTERN)
    
    # Dangerous characters for basic sanitization (from constants)
    DANGEROUS_CHARS = re.compile(SecurityConstants.DANGEROUS_CHARS_PATTERN)
    
    @classmethod
    def validate_email(cls, email: str) -> tuple[bool, str]:
        """
        Validate an email address format.
        
        Checks that the email is not empty, within length limits,
        and matches the standard email format.
        
        Args:
            email: The email address to validate.
        
        Returns:
            A tuple of (is_valid, error_message). If valid, error_message is empty.
        
        Raises:
            No exceptions are raised; validation errors are returned as messages.
        
        Example:
            >>> InputValidator.validate_email("user@example.com")
            (True, '')
            >>> InputValidator.validate_email("")
            (False, 'Email is required')
        """
        if not email:
            return False, "Email is required"
        
        email = email.strip()
        
        if len(email) > SecurityConstants.MAX_EMAIL_LENGTH:
            return False, "Email is too long"
        
        if not cls.EMAIL_PATTERN.match(email):
            return False, "Invalid email format"
        
        return True, ""
    
    @classmethod
    def validate_password(cls, password: str) -> tuple[bool, str]:
        """
        Validate a password meets minimum requirements.
        
        Checks that the password is not empty and within length limits.
        
        Args:
            password: The password to validate.
        
        Returns:
            A tuple of (is_valid, error_message). If valid, error_message is empty.
        
        Example:
            >>> InputValidator.validate_password("securepass123")
            (True, '')
            >>> InputValidator.validate_password("short")
            (False, 'Password must be at least 6 characters')
        """
        if not password:
            return False, "Password is required"
        
        if len(password) < SecurityConstants.MIN_PASSWORD_LENGTH:
            return False, f"Password must be at least {SecurityConstants.MIN_PASSWORD_LENGTH} characters"
        
        if len(password) > SecurityConstants.MAX_PASSWORD_LENGTH:
            return False, "Password is too long"
        
        return True, ""
    
    @classmethod
    def sanitize_input(cls, text: str) -> str:
        """
        Sanitize user input by removing potentially dangerous characters.
        
        Removes characters that could be used for injection attacks
        and trims whitespace.
        
        Args:
            text: The text to sanitize.
        
        Returns:
            Sanitized text with dangerous characters removed.
        
        Example:
            >>> InputValidator.sanitize_input("<script>alert('xss')</script>")
            'scriptalertxss/script'
            >>> InputValidator.sanitize_input("  normal text  ")
            'normal text'
        """
        if not text:
            return ""
        
        # Remove dangerous characters
        sanitized = cls.DANGEROUS_CHARS.sub('', text)
        
        # Trim whitespace
        sanitized = sanitized.strip()
        
        return sanitized
    
    @classmethod
    def validate_url(cls, url: str, require_https: bool = False) -> tuple[bool, str]:
        """
        Validate a URL format.
        
        Checks that the URL is not empty and starts with a valid protocol.
        Optionally requires HTTPS for security.
        
        Args:
            url: The URL to validate.
            require_https: If True, only HTTPS URLs are considered valid.
        
        Returns:
            A tuple of (is_valid, error_message). If valid, error_message is empty.
        
        Example:
            >>> InputValidator.validate_url("https://example.com")
            (True, '')
            >>> InputValidator.validate_url("http://example.com", require_https=True)
            (False, 'HTTPS is required for security')
        """
        if not url:
            return False, "URL is required"
        
        url = url.strip()
        
        if require_https and not url.startswith("https://"):
            return False, "HTTPS is required for security"
        
        if not url.startswith(("http://", "https://")):
            return False, "Invalid URL format"
        
        return True, ""


class SecurityChecker:
    """
    Security configuration checker and auditor.
    
    Provides methods to check security configuration and log warnings
    about insecure settings.
    
    Example:
        >>> SecurityChecker.check_https_enforcement("https://api.example.com")
        True
        >>> findings = SecurityChecker.run_security_audit()
        >>> len(findings['warnings'])
        0
    """
    
    @classmethod
    def check_https_enforcement(cls, url: str) -> bool:
        """
        Check if a URL uses HTTPS protocol.
        
        Args:
            url: The URL to check.
        
        Returns:
            True if using HTTPS, False otherwise.
        
        Example:
            >>> SecurityChecker.check_https_enforcement("https://secure.com")
            True
            >>> SecurityChecker.check_https_enforcement("http://insecure.com")
            False
        """
        return url.startswith("https://")
    
    @classmethod
    def warn_if_insecure(cls, url: str, context: str = "API") -> None:
        """
        Log a warning if URL is not using HTTPS.
        
        Args:
            url: The URL to check.
            context: Description of what the URL is for (for logging).
        
        Example:
            >>> SecurityChecker.warn_if_insecure("http://api.com", "Backend API")
            # Logs: ⚠️ SECURITY WARNING: Backend API URL is using HTTP...
        """
        if not cls.check_https_enforcement(url):
            logger.warning(
                f"⚠️ SECURITY WARNING: {context} URL is using HTTP instead of HTTPS. "
                f"This is insecure and should not be used in production. URL: {url}"
            )
    
    @classmethod
    def check_grpc_security(cls, use_tls: bool) -> None:
        """
        Log warnings about gRPC security configuration.
        
        Args:
            use_tls: Whether TLS is enabled for gRPC.
        
        Example:
            >>> SecurityChecker.check_grpc_security(False)
            # Logs: ⚠️ SECURITY WARNING: gRPC is using an insecure channel...
        """
        if not use_tls:
            logger.warning(
                "⚠️ SECURITY WARNING: gRPC is using an insecure channel. "
                "Enable TLS for production use."
            )
    
    @classmethod
    def run_security_audit(cls) -> dict[str, Any]:
        """
        Run a comprehensive security audit and return findings.
        
        Checks various security settings and returns a dictionary
        with warnings and informational messages.
        
        Returns:
            Dictionary containing:
                - warnings: List of security warnings.
                - info: List of informational messages.
                - secure_storage_available: Whether keyring is available.
                - dotenv_available: Whether python-dotenv is installed.
        
        Example:
            >>> findings = SecurityChecker.run_security_audit()
            >>> findings['secure_storage_available']
            True
        """
        env = EnvironmentConfig()
        
        findings: dict[str, Any] = {
            "warnings": [],
            "info": [],
            "secure_storage_available": SecureStorage.is_available(),
            "dotenv_available": DOTENV_AVAILABLE,
        }
        
        # Check HTTPS enforcement
        api_url = env.get_api_url()
        if not cls.check_https_enforcement(api_url):
            findings["warnings"].append(
                f"API URL is using HTTP: {api_url}"
            )
        
        # Check gRPC TLS
        if not env.get_grpc_use_tls():
            findings["warnings"].append(
                "gRPC is not using TLS encryption"
            )
        
        # Check debug mode
        if env.is_debug_mode():
            findings["warnings"].append(
                "Debug mode is enabled - disable for production"
            )
        
        # Info about secure storage
        if not SecureStorage.is_available():
            findings["info"].append(
                "Keyring not available - tokens stored in memory only"
            )
        
        return findings


# Initialize environment on module load
EnvironmentConfig.load()


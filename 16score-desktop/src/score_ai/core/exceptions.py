"""
Custom Exceptions for 16Score AI Application

This module defines custom exception classes for specific error handling
throughout the application. Using custom exceptions allows for more precise
error handling and better error messages.

Usage:
    from score_ai.core.exceptions import APIError, AuthenticationError
    
    try:
        response = api.login(email, password)
    except AuthenticationError as e:
        print(f"Login failed: {e}")
    except APIError as e:
        print(f"API error: {e}")

Example:
    >>> raise ValidationError("email", "Invalid email format")
    ValidationError: Validation failed for 'email': Invalid email format
"""

from __future__ import annotations

from typing import Any, Optional


class ScoreAIError(Exception):
    """
    Base exception for all 16Score AI errors.
    
    All custom exceptions in the application should inherit from this class.
    This allows catching all application-specific errors with a single except clause.
    
    Attributes:
        message: Human-readable error description.
        details: Additional error details (optional).
    
    Example:
        >>> try:
        ...     raise ScoreAIError("Something went wrong")
        ... except ScoreAIError as e:
        ...     print(e)
        Something went wrong
    """
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error description.
            details: Additional error details as a dictionary.
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


# =============================================================================
# API Exceptions
# =============================================================================

class APIError(ScoreAIError):
    """
    Base exception for API-related errors.
    
    Attributes:
        message: Error description.
        status_code: HTTP status code (if applicable).
        endpoint: API endpoint that caused the error.
    
    Example:
        >>> raise APIError("Request failed", status_code=500, endpoint="/api/login")
        APIError: Request failed (status: 500, endpoint: /api/login)
    """
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        endpoint: Optional[str] = None,
        details: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Initialize API error.
        
        Args:
            message: Error description.
            status_code: HTTP status code.
            endpoint: API endpoint that caused the error.
            details: Additional error details.
        """
        self.status_code = status_code
        self.endpoint = endpoint
        super().__init__(message, details)
    
    def __str__(self) -> str:
        """Return string representation with status code and endpoint."""
        parts = [self.message]
        if self.status_code:
            parts.append(f"status: {self.status_code}")
        if self.endpoint:
            parts.append(f"endpoint: {self.endpoint}")
        
        if len(parts) > 1:
            return f"{parts[0]} ({', '.join(parts[1:])})"
        return self.message


class AuthenticationError(APIError):
    """
    Exception raised when authentication fails.
    
    This includes invalid credentials, expired tokens, or missing authentication.
    
    Example:
        >>> raise AuthenticationError("Invalid credentials")
        AuthenticationError: Invalid credentials (status: 401)
    """
    
    def __init__(
        self,
        message: str = "Authentication failed",
        endpoint: Optional[str] = None,
        details: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Initialize authentication error.
        
        Args:
            message: Error description.
            endpoint: API endpoint that caused the error.
            details: Additional error details.
        """
        super().__init__(message, status_code=401, endpoint=endpoint, details=details)


class AuthorizationError(APIError):
    """
    Exception raised when user lacks permission for an action.
    
    Example:
        >>> raise AuthorizationError("Admin access required")
        AuthorizationError: Admin access required (status: 403)
    """
    
    def __init__(
        self,
        message: str = "Access denied",
        endpoint: Optional[str] = None,
        details: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Initialize authorization error.
        
        Args:
            message: Error description.
            endpoint: API endpoint that caused the error.
            details: Additional error details.
        """
        super().__init__(message, status_code=403, endpoint=endpoint, details=details)


class ConnectionError(APIError):
    """
    Exception raised when connection to server fails.
    
    Example:
        >>> raise ConnectionError("Unable to connect to server")
        ConnectionError: Unable to connect to server
    """
    
    def __init__(
        self,
        message: str = "Connection failed",
        endpoint: Optional[str] = None,
        details: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Initialize connection error.
        
        Args:
            message: Error description.
            endpoint: API endpoint.
            details: Additional error details.
        """
        super().__init__(message, endpoint=endpoint, details=details)


class TimeoutError(APIError):
    """
    Exception raised when request times out.
    
    Example:
        >>> raise TimeoutError("Request timed out after 30s", timeout=30)
        TimeoutError: Request timed out after 30s
    """
    
    def __init__(
        self,
        message: str = "Request timed out",
        timeout: Optional[float] = None,
        endpoint: Optional[str] = None,
        details: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Initialize timeout error.
        
        Args:
            message: Error description.
            timeout: Timeout value in seconds.
            endpoint: API endpoint.
            details: Additional error details.
        """
        self.timeout = timeout
        if timeout and "timeout" not in (details or {}):
            details = details or {}
            details["timeout"] = timeout
        super().__init__(message, endpoint=endpoint, details=details)


# =============================================================================
# Validation Exceptions
# =============================================================================

class ValidationError(ScoreAIError):
    """
    Exception raised when input validation fails.
    
    Attributes:
        field: Name of the field that failed validation.
        message: Description of the validation error.
    
    Example:
        >>> raise ValidationError("email", "Invalid email format")
        ValidationError: Validation failed for 'email': Invalid email format
    """
    
    def __init__(
        self,
        field: str,
        message: str,
        details: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Initialize validation error.
        
        Args:
            field: Name of the field that failed validation.
            message: Description of what's wrong.
            details: Additional validation details.
        """
        self.field = field
        full_message = f"Validation failed for '{field}': {message}"
        super().__init__(full_message, details)


class ConfigurationError(ScoreAIError):
    """
    Exception raised when configuration is invalid or missing.
    
    Example:
        >>> raise ConfigurationError("api.backend_url", "URL is required")
        ConfigurationError: Configuration error for 'api.backend_url': URL is required
    """
    
    def __init__(
        self,
        key: str,
        message: str,
        details: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Initialize configuration error.
        
        Args:
            key: Configuration key that has the error.
            message: Description of the error.
            details: Additional details.
        """
        self.key = key
        full_message = f"Configuration error for '{key}': {message}"
        super().__init__(full_message, details)


# =============================================================================
# Detection Exceptions
# =============================================================================

class DetectionError(ScoreAIError):
    """
    Base exception for detection-related errors.
    
    Example:
        >>> raise DetectionError("Failed to process frame")
        DetectionError: Failed to process frame
    """
    pass


class GRPCError(DetectionError):
    """
    Exception raised when gRPC communication fails.
    
    Attributes:
        server: gRPC server address.
        code: gRPC status code.
    
    Example:
        >>> raise GRPCError("Connection refused", server="localhost:50050")
        GRPCError: Connection refused (server: localhost:50050)
    """
    
    def __init__(
        self,
        message: str,
        server: Optional[str] = None,
        code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Initialize gRPC error.
        
        Args:
            message: Error description.
            server: gRPC server address.
            code: gRPC status code.
            details: Additional details.
        """
        self.server = server
        self.code = code
        details = details or {}
        if server:
            details["server"] = server
        if code:
            details["code"] = code
        super().__init__(message, details)


class OCRError(DetectionError):
    """
    Exception raised when OCR processing fails.
    
    Example:
        >>> raise OCRError("Failed to extract text from image")
        OCRError: Failed to extract text from image
    """
    pass


class ImageProcessingError(DetectionError):
    """
    Exception raised when image processing fails.
    
    Example:
        >>> raise ImageProcessingError("Invalid image format")
        ImageProcessingError: Invalid image format
    """
    pass


# =============================================================================
# Storage Exceptions
# =============================================================================

class StorageError(ScoreAIError):
    """
    Base exception for storage-related errors.
    
    Example:
        >>> raise StorageError("Failed to save file")
        StorageError: Failed to save file
    """
    pass


class FileNotFoundError(StorageError):
    """
    Exception raised when a required file is not found.
    
    Example:
        >>> raise FileNotFoundError("config.json")
        FileNotFoundError: File not found: config.json
    """
    
    def __init__(
        self,
        filepath: str,
        details: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Initialize file not found error.
        
        Args:
            filepath: Path to the missing file.
            details: Additional details.
        """
        self.filepath = filepath
        super().__init__(f"File not found: {filepath}", details)


class SecureStorageError(StorageError):
    """
    Exception raised when secure storage operations fail.
    
    Example:
        >>> raise SecureStorageError("Keyring not available")
        SecureStorageError: Keyring not available
    """
    pass


# =============================================================================
# UI Exceptions
# =============================================================================

class UIError(ScoreAIError):
    """
    Base exception for UI-related errors.
    
    Example:
        >>> raise UIError("Failed to render component")
        UIError: Failed to render component
    """
    pass


class ResourceNotFoundError(UIError):
    """
    Exception raised when a UI resource (image, icon, etc.) is not found.
    
    Example:
        >>> raise ResourceNotFoundError("logo.png", "images")
        ResourceNotFoundError: Resource not found: logo.png in images
    """
    
    def __init__(
        self,
        resource: str,
        resource_type: str = "resource",
        details: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Initialize resource not found error.
        
        Args:
            resource: Name or path of the missing resource.
            resource_type: Type of resource (e.g., "image", "icon").
            details: Additional details.
        """
        self.resource = resource
        self.resource_type = resource_type
        super().__init__(f"Resource not found: {resource} ({resource_type})", details)


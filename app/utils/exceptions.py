"""
Custom exception classes for the application.
"""

from typing import Any, Dict, Optional
from datetime import datetime


class AppException(Exception):
    """Base application exception with enhanced error handling."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.cause = cause
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "cause": str(self.cause) if self.cause else None
        }
    
    def __str__(self) -> str:
        """String representation of the exception."""
        base_msg = f"[{self.error_code}] {self.message}"
        if self.details:
            base_msg += f" | Details: {self.details}"
        if self.cause:
            base_msg += f" | Caused by: {self.cause}"
        return base_msg


class ValidationError(AppException):
    """Data validation errors."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if field:
            details['field'] = field
        if value is not None:
            details['value'] = str(value)
        
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class ServiceError(AppException):
    """Service layer errors."""
    
    def __init__(
        self,
        message: str,
        service: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if service:
            details['service'] = service
        if operation:
            details['operation'] = operation
        
        super().__init__(
            message=message,
            error_code="SERVICE_ERROR",
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class ExternalServiceError(AppException):
    """External service integration errors."""
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if service_name:
            details['service_name'] = service_name
        if status_code:
            details['status_code'] = status_code
        if response_body:
            details['response_body'] = response_body
        
        super().__init__(
            message=message,
            error_code="EXTERNAL_SERVICE_ERROR",
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class VideoNotFoundError(AppException):
    """Video not found error."""
    
    def __init__(
        self,
        message: str = "Video not found",
        video_id: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if video_id:
            details['video_id'] = video_id
        
        super().__init__(
            message=message,
            error_code="VIDEO_NOT_FOUND",
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class VideoGenerationError(AppException):
    """Video generation error."""
    
    def __init__(
        self,
        message: str,
        video_id: Optional[str] = None,
        stage: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if video_id:
            details['video_id'] = video_id
        if stage:
            details['stage'] = stage
        
        super().__init__(
            message=message,
            error_code="VIDEO_GENERATION_ERROR",
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class RAGError(AppException):
    """RAG system error."""
    
    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if query:
            details['query'] = query
        if operation:
            details['operation'] = operation
        
        super().__init__(
            message=message,
            error_code="RAG_ERROR",
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class AgentError(AppException):
    """Agent execution error."""
    
    def __init__(
        self,
        message: str,
        agent_type: Optional[str] = None,
        agent_id: Optional[str] = None,
        execution_step: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if agent_type:
            details['agent_type'] = agent_type
        if agent_id:
            details['agent_id'] = agent_id
        if execution_step:
            details['execution_step'] = execution_step
        
        super().__init__(
            message=message,
            error_code="AGENT_ERROR",
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class AWSError(AppException):
    """AWS integration error."""
    
    def __init__(
        self,
        message: str,
        service: Optional[str] = None,
        operation: Optional[str] = None,
        aws_error_code: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if service:
            details['aws_service'] = service
        if operation:
            details['operation'] = operation
        if aws_error_code:
            details['aws_error_code'] = aws_error_code
        
        super().__init__(
            message=message,
            error_code="AWS_ERROR",
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class ConfigurationError(AppException):
    """Configuration-related errors."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_file: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if config_key:
            details['config_key'] = config_key
        if config_file:
            details['config_file'] = config_file
        
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class AuthenticationError(AppException):
    """Authentication-related errors."""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        user_id: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if user_id:
            details['user_id'] = user_id
        
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class AuthorizationError(AppException):
    """Authorization-related errors."""
    
    def __init__(
        self,
        message: str = "Access denied",
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if user_id:
            details['user_id'] = user_id
        if resource:
            details['resource'] = resource
        if action:
            details['action'] = action
        
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )
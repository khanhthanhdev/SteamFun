"""
Custom exception classes for the application.
"""


class AppException(Exception):
    """Base application exception."""
    pass


class ValidationError(AppException):
    """Data validation errors."""
    pass


class ServiceError(AppException):
    """Service layer errors."""
    pass


class ExternalServiceError(AppException):
    """External service integration errors."""
    pass


class VideoNotFoundError(AppException):
    """Video not found error."""
    pass


class VideoGenerationError(AppException):
    """Video generation error."""
    pass


class RAGError(AppException):
    """RAG system error."""
    pass


class AgentError(AppException):
    """Agent execution error."""
    pass


class AWSError(AppException):
    """AWS integration error."""
    pass
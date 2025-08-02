"""
AWS Logging Configuration

Sets up comprehensive logging for AWS operations with proper error handling,
performance monitoring, and security considerations.
"""

import logging
import logging.handlers
import os
import sys
from typing import Optional, Dict, Any
from datetime import datetime

from .config import AWSConfig


def setup_aws_logging(config: AWSConfig, log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up comprehensive logging for AWS operations.
    
    Args:
        config: AWS configuration
        log_file: Optional log file path
        
    Returns:
        Configured logger for AWS operations
    """
    # Create AWS-specific logger
    aws_logger = logging.getLogger('aws_integration')
    aws_logger.setLevel(getattr(logging, config.log_level.upper()))
    
    # Clear any existing handlers
    aws_logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.log_level.upper()))
    console_handler.setFormatter(simple_formatter)
    aws_logger.addHandler(console_handler)
    
    # File handler if specified or if log file exists
    if log_file or os.path.exists('logs'):
        if not log_file:
            os.makedirs('logs', exist_ok=True)
            log_file = f'logs/aws_integration_{datetime.now().strftime("%Y%m%d")}.log'
        
        # Rotating file handler to prevent large log files
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)  # Always debug level for file
        file_handler.setFormatter(detailed_formatter)
        aws_logger.addHandler(file_handler)
    
    # Set up boto3 logging if enabled
    if config.enable_boto3_logging:
        setup_boto3_logging(config.log_level)
    
    aws_logger.info(f"AWS logging configured - Level: {config.log_level}")
    return aws_logger


def setup_boto3_logging(log_level: str = 'WARNING') -> None:
    """
    Configure boto3 and botocore logging.
    
    Args:
        log_level: Logging level for boto3
    """
    # Configure boto3 loggers
    boto3_loggers = [
        'boto3',
        'botocore',
        'botocore.credentials',
        'botocore.utils',
        'botocore.hooks',
        'botocore.loaders',
        'botocore.parsers',
        'botocore.endpoint',
        'botocore.auth',
        'botocore.retryhandler',
        'urllib3.connectionpool'
    ]
    
    for logger_name in boto3_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, log_level.upper()))
    
    # Special handling for urllib3 to reduce noise
    urllib3_logger = logging.getLogger('urllib3.connectionpool')
    urllib3_logger.setLevel(logging.WARNING)


def create_operation_logger(operation_name: str, context: Dict[str, Any] = None) -> logging.LoggerAdapter:
    """
    Create a logger adapter for specific AWS operations with context.
    
    Args:
        operation_name: Name of the AWS operation
        context: Additional context to include in logs
        
    Returns:
        Logger adapter with operation context
    """
    logger = logging.getLogger('aws_integration')
    
    # Create context for the operation
    operation_context = {
        'operation': operation_name,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    if context:
        operation_context.update(context)
    
    return logging.LoggerAdapter(logger, operation_context)


def log_aws_operation(operation: str, service: str, **kwargs) -> None:
    """
    Log AWS operation with standardized format.
    
    Args:
        operation: AWS operation name
        service: AWS service name
        **kwargs: Additional operation details
    """
    logger = logging.getLogger('aws_integration')
    
    log_data = {
        'service': service,
        'operation': operation,
        **kwargs
    }
    
    # Filter out sensitive information
    filtered_data = _filter_sensitive_data(log_data)
    
    logger.info(f"AWS {service} operation: {operation}", extra=filtered_data)


def log_aws_error(error: Exception, operation: str, service: str, **kwargs) -> None:
    """
    Log AWS operation errors with detailed information.
    
    Args:
        error: Exception that occurred
        operation: AWS operation name
        service: AWS service name
        **kwargs: Additional error context
    """
    logger = logging.getLogger('aws_integration')
    
    error_data = {
        'service': service,
        'operation': operation,
        'error_type': type(error).__name__,
        'error_message': str(error),
        **kwargs
    }
    
    # Add boto3 error details if available
    if hasattr(error, 'response'):
        error_data.update({
            'error_code': error.response.get('Error', {}).get('Code'),
            'http_status': error.response.get('ResponseMetadata', {}).get('HTTPStatusCode'),
            'request_id': error.response.get('ResponseMetadata', {}).get('RequestId')
        })
    
    # Filter out sensitive information
    filtered_data = _filter_sensitive_data(error_data)
    
    logger.error(f"AWS {service} operation failed: {operation}", extra=filtered_data, exc_info=True)


def log_performance_metrics(operation: str, duration: float, **metrics) -> None:
    """
    Log performance metrics for AWS operations.
    
    Args:
        operation: Operation name
        duration: Operation duration in seconds
        **metrics: Additional performance metrics
    """
    logger = logging.getLogger('aws_integration')
    
    perf_data = {
        'operation': operation,
        'duration_seconds': round(duration, 3),
        **metrics
    }
    
    logger.info(f"Performance metrics for {operation}", extra=perf_data)


def _filter_sensitive_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter sensitive information from log data.
    
    Args:
        data: Log data dictionary
        
    Returns:
        Filtered data dictionary
    """
    sensitive_keys = {
        'password', 'secret', 'key', 'token', 'credential',
        'access_key', 'secret_key', 'session_token'
    }
    
    filtered = {}
    for key, value in data.items():
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            filtered[key] = '[REDACTED]'
        elif isinstance(value, dict):
            filtered[key] = _filter_sensitive_data(value)
        else:
            filtered[key] = value
    
    return filtered


class AWSOperationLogger:
    """Context manager for logging AWS operations with timing."""
    
    def __init__(self, operation: str, service: str, **context):
        self.operation = operation
        self.service = service
        self.context = context
        self.start_time = None
        self.logger = logging.getLogger('aws_integration')
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        log_aws_operation(self.operation, self.service, **self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.utcnow() - self.start_time).total_seconds()
        
        if exc_type is None:
            # Success
            log_performance_metrics(
                f"{self.service}.{self.operation}",
                duration,
                **self.context
            )
        else:
            # Error occurred
            log_aws_error(
                exc_val,
                self.operation,
                self.service,
                duration=duration,
                **self.context
            )
        
        return False  # Don't suppress exceptions


# Convenience function for creating operation loggers
def aws_operation(operation: str, service: str, **context):
    """
    Decorator/context manager for AWS operations.
    
    Usage:
        with aws_operation('upload_file', 's3', bucket='my-bucket'):
            # AWS operation code
    """
    return AWSOperationLogger(operation, service, **context)
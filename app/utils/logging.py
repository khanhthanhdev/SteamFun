"""
Logging configuration and utilities for the application.
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import json
from datetime import datetime
import functools
import time
import asyncio
from contextlib import contextmanager


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
            
        return json.dumps(log_entry)


class ContextFilter(logging.Filter):
    """Filter to add contextual information to log records."""
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.context = context or {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to log record."""
        if self.context:
            record.extra_fields = getattr(record, "extra_fields", {})
            record.extra_fields.update(self.context)
        return True


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        json_format: Whether to use JSON formatting
        context: Additional context to include in logs
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure formatters
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    # Configure handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Add context filter if provided
    if context:
        context_filter = ContextFilter(context)
        for handler in handlers:
            handler.addFilter(context_filter)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        handlers=handlers,
        force=True
    )


def get_logger(name: str, context: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Get a logger with optional context.
    
    Args:
        name: Logger name
        context: Additional context for this logger
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if context:
        # Add context filter to logger
        context_filter = ContextFilter(context)
        logger.addFilter(context_filter)
    
    return logger


def log_function_call(func_name: str, args: tuple = (), kwargs: Dict[str, Any] = None) -> None:
    """
    Log function call with arguments.
    
    Args:
        func_name: Name of the function being called
        args: Positional arguments
        kwargs: Keyword arguments
    """
    logger = get_logger(__name__)
    kwargs = kwargs or {}
    
    logger.debug(
        f"Calling function: {func_name}",
        extra={
            "extra_fields": {
                "function": func_name,
                "args": str(args),
                "kwargs": {k: str(v) for k, v in kwargs.items()}
            }
        }
    )


def log_performance(func_name: str, duration: float, **metrics) -> None:
    """
    Log performance metrics.
    
    Args:
        func_name: Name of the function
        duration: Execution duration in seconds
        **metrics: Additional performance metrics
    """
    logger = get_logger(__name__)
    
    logger.info(
        f"Performance: {func_name} completed in {duration:.3f}s",
        extra={
            "extra_fields": {
                "function": func_name,
                "duration": duration,
                **metrics
            }
        }
    )


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)
    
    def log_info(self, message: str, **kwargs) -> None:
        """Log info message with optional context."""
        self.logger.info(message, extra={"extra_fields": kwargs} if kwargs else None)
    
    def log_error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log error message with optional context."""
        self.logger.error(
            message, 
            exc_info=exc_info,
            extra={"extra_fields": kwargs} if kwargs else None
        )
    
    def log_debug(self, message: str, **kwargs) -> None:
        """Log debug message with optional context."""
        self.logger.debug(message, extra={"extra_fields": kwargs} if kwargs else None)
    
    def log_warning(self, message: str, **kwargs) -> None:
        """Log warning message with optional context."""
        self.logger.warning(message, extra={"extra_fields": kwargs} if kwargs else None)


@contextmanager
def log_context(**context):
    """Context manager to add temporary logging context."""
    logger = logging.getLogger()
    context_filter = ContextFilter(context)
    
    # Add filter to all handlers
    for handler in logger.handlers:
        handler.addFilter(context_filter)
    
    try:
        yield
    finally:
        # Remove filter from all handlers
        for handler in logger.handlers:
            handler.removeFilter(context_filter)


def log_method_calls(logger_name: Optional[str] = None):
    """
    Decorator to log method calls with arguments and return values.
    
    Args:
        logger_name: Optional logger name, defaults to module name
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name or func.__module__)
            
            # Log method entry
            logger.debug(
                f"Entering {func.__name__}",
                extra={
                    "extra_fields": {
                        "function": func.__name__,
                        "args": str(args[1:]) if args else str(args),  # Skip 'self' for methods
                        "kwargs": {k: str(v) for k, v in kwargs.items()}
                    }
                }
            )
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log successful completion
                logger.debug(
                    f"Completed {func.__name__} in {duration:.3f}s",
                    extra={
                        "extra_fields": {
                            "function": func.__name__,
                            "duration": duration,
                            "success": True
                        }
                    }
                )
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Log error
                logger.error(
                    f"Error in {func.__name__} after {duration:.3f}s: {str(e)}",
                    exc_info=True,
                    extra={
                        "extra_fields": {
                            "function": func.__name__,
                            "duration": duration,
                            "success": False,
                            "error": str(e)
                        }
                    }
                )
                raise
        
        return wrapper
    return decorator


def log_async_method_calls(logger_name: Optional[str] = None):
    """
    Decorator to log async method calls with arguments and return values.
    
    Args:
        logger_name: Optional logger name, defaults to module name
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            logger = get_logger(logger_name or func.__module__)
            
            # Log method entry
            logger.debug(
                f"Entering async {func.__name__}",
                extra={
                    "extra_fields": {
                        "function": func.__name__,
                        "args": str(args[1:]) if args else str(args),  # Skip 'self' for methods
                        "kwargs": {k: str(v) for k, v in kwargs.items()}
                    }
                }
            )
            
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log successful completion
                logger.debug(
                    f"Completed async {func.__name__} in {duration:.3f}s",
                    extra={
                        "extra_fields": {
                            "function": func.__name__,
                            "duration": duration,
                            "success": True
                        }
                    }
                )
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Log error
                logger.error(
                    f"Error in async {func.__name__} after {duration:.3f}s: {str(e)}",
                    exc_info=True,
                    extra={
                        "extra_fields": {
                            "function": func.__name__,
                            "duration": duration,
                            "success": False,
                            "error": str(e)
                        }
                    }
                )
                raise
        
        return wrapper
    return decorator


def log_api_request(logger_name: Optional[str] = None):
    """
    Decorator specifically for logging API requests.
    
    Args:
        logger_name: Optional logger name
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            logger = get_logger(logger_name or "api")
            
            # Extract request info (assuming FastAPI request object)
            request = None
            for arg in args:
                if hasattr(arg, 'method') and hasattr(arg, 'url'):
                    request = arg
                    break
            
            if request:
                logger.info(
                    f"API Request: {request.method} {request.url.path}",
                    extra={
                        "extra_fields": {
                            "method": request.method,
                            "path": request.url.path,
                            "query_params": str(request.query_params),
                            "client_ip": getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
                        }
                    }
                )
            
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.info(
                    f"API Response: {func.__name__} completed in {duration:.3f}s",
                    extra={
                        "extra_fields": {
                            "endpoint": func.__name__,
                            "duration": duration,
                            "success": True
                        }
                    }
                )
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                logger.error(
                    f"API Error: {func.__name__} failed after {duration:.3f}s: {str(e)}",
                    exc_info=True,
                    extra={
                        "extra_fields": {
                            "endpoint": func.__name__,
                            "duration": duration,
                            "success": False,
                            "error": str(e)
                        }
                    }
                )
                raise
        
        return wrapper
    return decorator
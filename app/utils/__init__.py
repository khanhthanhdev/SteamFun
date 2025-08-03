"""
Utils Package

Contains utility functions, logging configuration, and helper classes.
"""

from .exceptions import (
    AppException,
    ValidationError,
    ServiceError,
    ExternalServiceError,
    VideoNotFoundError,
    VideoGenerationError,
    RAGError,
    AgentError,
    AWSError,
)

from .logging import (
    JSONFormatter,
    ContextFilter,
    LoggerMixin,
    setup_logging,
    get_logger,
    log_function_call,
    log_performance,
)

from .logging_config import (
    get_logging_config,
    configure_application_logging,
    get_service_logger_context,
    get_request_logger_context,
)

from .helpers import (
    generate_uuid,
    generate_short_id,
    hash_string,
    get_current_timestamp,
    format_timestamp,
    parse_timestamp,
    safe_json_loads,
    safe_json_dumps,
    sanitize_filename,
    ensure_directory,
    get_file_size,
    format_file_size,
    chunk_list,
    flatten_dict,
    deep_merge_dicts,
    retry,
    async_retry,
    timing,
    async_timing,
    run_in_thread,
    validate_email,
    validate_url,
    truncate_string,
    camel_to_snake,
    snake_to_camel,
    is_valid_uuid,
)

__all__ = [
    # Exceptions
    "AppException",
    "ValidationError",
    "ServiceError",
    "ExternalServiceError",
    "VideoNotFoundError",
    "VideoGenerationError",
    "RAGError",
    "AgentError",
    "AWSError",
    # Logging
    "JSONFormatter",
    "ContextFilter",
    "LoggerMixin",
    "setup_logging",
    "get_logger",
    "log_function_call",
    "log_performance",
    "get_logging_config",
    "configure_application_logging",
    "get_service_logger_context",
    "get_request_logger_context",
    # Helpers
    "generate_uuid",
    "generate_short_id",
    "hash_string",
    "get_current_timestamp",
    "format_timestamp",
    "parse_timestamp",
    "safe_json_loads",
    "safe_json_dumps",
    "sanitize_filename",
    "ensure_directory",
    "get_file_size",
    "format_file_size",
    "chunk_list",
    "flatten_dict",
    "deep_merge_dicts",
    "retry",
    "async_retry",
    "timing",
    "async_timing",
    "run_in_thread",
    "validate_email",
    "validate_url",
    "truncate_string",
    "camel_to_snake",
    "snake_to_camel",
    "is_valid_uuid",
]
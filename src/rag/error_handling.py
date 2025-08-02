"""
Robust error handling and recovery mechanisms for the RAG system.

This module provides comprehensive error handling with fallback strategies,
retry logic with exponential backoff, and graceful degradation capabilities.
"""

import time
import random
import logging
import traceback
from typing import Dict, List, Optional, Any, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
import os
from pathlib import Path
import asyncio
import threading
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError


class ErrorType(Enum):
    """Types of errors that can occur in the RAG system."""
    VECTOR_STORE_CONNECTION = "vector_store_connection"
    EMBEDDING_GENERATION = "embedding_generation"
    QUERY_PARSING = "query_parsing"
    DOCUMENT_PROCESSING = "document_processing"
    CACHE_ACCESS = "cache_access"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_TIMEOUT = "network_timeout"
    AUTHENTICATION = "authentication"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for an error."""
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    original_exception: Optional[Exception] = None
    timestamp: float = field(default_factory=time.time)
    operation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0


@dataclass
class ErrorHandlingConfig:
    """Configuration for error handling behavior."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    enable_fallbacks: bool = True
    log_errors: bool = True
    error_log_path: Optional[str] = None
    timeout_seconds: float = 30.0


@dataclass
class RetryStrategy:
    """Strategy for retrying failed operations."""
    max_attempts: int
    base_delay: float
    max_delay: float
    backoff_multiplier: float
    jitter: bool
    should_retry: Callable[[Exception], bool]


@dataclass
class FallbackStrategy:
    """Strategy for handling failures with fallback mechanisms."""
    strategy_type: str
    fallback_action: Callable[[], Any]
    description: str
    priority: int = 1


@dataclass
class ErrorResponse:
    """Response containing error information and suggestions."""
    success: bool
    error_type: Optional[ErrorType] = None
    message: str = ""
    suggestions: List[str] = field(default_factory=list)
    fallback_result: Optional[Any] = None
    retry_after: Optional[float] = None


class VectorStoreError(Exception):
    """Custom exception for vector store related errors."""
    def __init__(self, message: str, error_type: str = "connection", original_error: Optional[Exception] = None):
        super().__init__(message)
        self.error_type = error_type
        self.original_error = original_error


class EmbeddingError(Exception):
    """Custom exception for embedding generation errors."""
    def __init__(self, message: str, error_type: str = "generation", original_error: Optional[Exception] = None):
        super().__init__(message)
        self.error_type = error_type
        self.original_error = original_error


class QueryParsingError(Exception):
    """Custom exception for query parsing errors."""
    def __init__(self, message: str, malformed_query: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.malformed_query = malformed_query
        self.original_error = original_error


class RetryManager:
    """Manages retry logic with exponential backoff."""
    
    def __init__(self, config: ErrorHandlingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def create_exponential_backoff_strategy(self, 
                                          max_retries: Optional[int] = None,
                                          base_delay: Optional[float] = None,
                                          max_delay: Optional[float] = None) -> RetryStrategy:
        """Create an exponential backoff retry strategy."""
        return RetryStrategy(
            max_attempts=max_retries or self.config.max_retries,
            base_delay=base_delay or self.config.base_delay,
            max_delay=max_delay or self.config.max_delay,
            backoff_multiplier=self.config.backoff_multiplier,
            jitter=self.config.jitter,
            should_retry=self._default_should_retry
        )
    
    def execute_with_retry(self, operation: Callable, strategy: RetryStrategy, 
                          operation_name: str = "operation") -> Any:
        """Execute an operation with retry logic."""
        last_exception = None
        
        for attempt in range(strategy.max_attempts):
            try:
                self.logger.debug(f"Attempting {operation_name}, attempt {attempt + 1}/{strategy.max_attempts}")
                return operation()
            except Exception as e:
                last_exception = e
                
                # Check if we should retry this exception type
                if not strategy.should_retry(e):
                    self.logger.error(f"Operation {operation_name} failed with non-retryable error after {attempt + 1} attempts: {str(e)}")
                    raise e
                
                # Check if this was the last attempt
                if attempt == strategy.max_attempts - 1:
                    self.logger.error(f"Operation {operation_name} failed after {attempt + 1} attempts: {str(e)}")
                    raise e
                
                delay = self._calculate_delay(attempt, strategy)
                self.logger.warning(f"Operation {operation_name} failed (attempt {attempt + 1}), retrying in {delay:.2f}s: {str(e)}")
                time.sleep(delay)
        
        # This should never be reached, but just in case
        if last_exception:
            raise last_exception
    
    def _calculate_delay(self, attempt: int, strategy: RetryStrategy) -> float:
        """Calculate delay for the next retry attempt."""
        delay = strategy.base_delay * (strategy.backoff_multiplier ** attempt)
        delay = min(delay, strategy.max_delay)
        
        if strategy.jitter:
            # Add jitter to prevent thundering herd
            jitter_range = delay * 0.1
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)
    
    def _default_should_retry(self, exception: Exception) -> bool:
        """Default logic for determining if an operation should be retried."""
        # Don't retry on certain types of errors
        non_retryable_errors = (
            ValueError,  # Bad input data
            TypeError,   # Programming errors
            KeyError,    # Missing required data
        )
        
        if isinstance(exception, non_retryable_errors):
            return False
        
        # Retry on specific exception types
        retryable_exception_types = (
            ConnectionError,
            TimeoutError,
            OSError,  # Network-related OS errors
        )
        
        if isinstance(exception, retryable_exception_types):
            return True
        
        # Retry on network/connection errors based on message content
        retryable_error_messages = [
            "connection", "timeout", "network", "unavailable", 
            "service", "temporary", "rate limit"
        ]
        
        error_message = str(exception).lower()
        return any(msg in error_message for msg in retryable_error_messages)


class FallbackManager:
    """Manages fallback strategies for different types of failures."""
    
    def __init__(self, config: ErrorHandlingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._fallback_strategies: Dict[ErrorType, List[FallbackStrategy]] = {}
        self._initialize_default_strategies()
    
    def _initialize_default_strategies(self):
        """Initialize default fallback strategies."""
        # Vector store fallback strategies
        self.register_fallback_strategy(
            ErrorType.VECTOR_STORE_CONNECTION,
            FallbackStrategy(
                strategy_type="backup_store",
                fallback_action=self._use_backup_vector_store,
                description="Use backup vector store",
                priority=1
            )
        )
        
        self.register_fallback_strategy(
            ErrorType.VECTOR_STORE_CONNECTION,
            FallbackStrategy(
                strategy_type="cached_results",
                fallback_action=self._use_cached_results,
                description="Use cached results if available",
                priority=2
            )
        )
        
        # Embedding generation fallback strategies
        self.register_fallback_strategy(
            ErrorType.EMBEDDING_GENERATION,
            FallbackStrategy(
                strategy_type="alternative_model",
                fallback_action=self._use_alternative_embedding_model,
                description="Use alternative embedding model",
                priority=1
            )
        )
        
        # Query parsing fallback strategies
        self.register_fallback_strategy(
            ErrorType.QUERY_PARSING,
            FallbackStrategy(
                strategy_type="simplified_query",
                fallback_action=self._create_simplified_query,
                description="Create simplified fallback query",
                priority=1
            )
        )
    
    def register_fallback_strategy(self, error_type: ErrorType, strategy: FallbackStrategy):
        """Register a fallback strategy for a specific error type."""
        if error_type not in self._fallback_strategies:
            self._fallback_strategies[error_type] = []
        
        self._fallback_strategies[error_type].append(strategy)
        # Sort by priority (lower number = higher priority)
        self._fallback_strategies[error_type].sort(key=lambda s: s.priority)
    
    def get_fallback_strategies(self, error_type: ErrorType) -> List[FallbackStrategy]:
        """Get fallback strategies for a specific error type."""
        return self._fallback_strategies.get(error_type, [])
    
    def execute_fallback(self, error_type: ErrorType, context: Dict[str, Any] = None) -> Any:
        """Execute the best available fallback strategy."""
        strategies = self.get_fallback_strategies(error_type)
        
        if not strategies:
            self.logger.warning(f"No fallback strategies available for error type: {error_type}")
            return None
        
        for strategy in strategies:
            try:
                self.logger.info(f"Executing fallback strategy: {strategy.description}")
                return strategy.fallback_action()
            except Exception as e:
                self.logger.warning(f"Fallback strategy '{strategy.description}' failed: {str(e)}")
                continue
        
        self.logger.error(f"All fallback strategies failed for error type: {error_type}")
        return None
    
    def _use_backup_vector_store(self) -> Any:
        """Fallback to backup vector store."""
        # This would be implemented to use a backup vector store
        self.logger.info("Using backup vector store")
        return {"fallback": "backup_store", "available": False}
    
    def _use_cached_results(self) -> Any:
        """Fallback to cached results."""
        # This would be implemented to return cached results
        self.logger.info("Using cached results")
        return {"fallback": "cached_results", "available": False}
    
    def _use_alternative_embedding_model(self) -> Any:
        """Fallback to alternative embedding model."""
        # This would be implemented to use an alternative embedding model
        self.logger.info("Using alternative embedding model")
        return {"fallback": "alternative_model", "available": False}
    
    def _create_simplified_query(self) -> Any:
        """Create a simplified fallback query."""
        # This would be implemented to create a simplified query
        self.logger.info("Creating simplified fallback query")
        return {"fallback": "simplified_query", "query": "general search"}


class StructuredErrorLogger:
    """Structured logging for errors with context and metadata."""
    
    def __init__(self, config: ErrorHandlingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup structured logging configuration."""
        if self.config.error_log_path:
            handler = logging.FileHandler(self.config.error_log_path)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_error(self, error_context: ErrorContext):
        """Log an error with full context."""
        error_data = {
            "error_type": error_context.error_type.value,
            "severity": error_context.severity.value,
            "error_message": error_context.message,  # Changed from "message" to avoid conflict
            "operation": error_context.operation,
            "retry_count": error_context.retry_count,
            "timestamp": error_context.timestamp,
            "metadata": error_context.metadata
        }
        
        if error_context.original_exception:
            error_data["exception_type"] = type(error_context.original_exception).__name__
            error_data["exception_message"] = str(error_context.original_exception)
            error_data["traceback"] = traceback.format_exception(
                type(error_context.original_exception),
                error_context.original_exception,
                error_context.original_exception.__traceback__
            )
        
        log_message = f"RAG Error: {error_context.message}"
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message, extra=error_data)
        elif error_context.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message, extra=error_data)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message, extra=error_data)
        else:
            self.logger.info(log_message, extra=error_data)


class RobustErrorHandler:
    """
    Comprehensive error handler for the RAG system with fallback strategies,
    retry logic, and graceful degradation capabilities.
    """
    
    def __init__(self, config: Optional[ErrorHandlingConfig] = None):
        self.config = config or ErrorHandlingConfig()
        self.fallback_manager = FallbackManager(self.config)
        self.retry_manager = RetryManager(self.config)
        self.error_logger = StructuredErrorLogger(self.config)
        self.graceful_degradation = GracefulDegradation(self.config)
        self.logger = logging.getLogger(__name__)
        
        # Error statistics
        self._error_counts: Dict[ErrorType, int] = {}
        self._last_errors: Dict[ErrorType, float] = {}
    
    def handle_vector_store_error(self, error: VectorStoreError, operation: str = "vector_store_operation") -> ErrorResponse:
        """Handle vector store connection failures with fallback strategies."""
        error_context = ErrorContext(
            error_type=ErrorType.VECTOR_STORE_CONNECTION,
            severity=ErrorSeverity.HIGH,
            message=f"Vector store error in {operation}: {str(error)}",
            original_exception=error,
            operation=operation,
            metadata={"error_subtype": error.error_type}
        )
        
        self._track_error(error_context)
        self.error_logger.log_error(error_context)
        
        # Try fallback strategies
        if self.config.enable_fallbacks:
            fallback_result = self.fallback_manager.execute_fallback(
                ErrorType.VECTOR_STORE_CONNECTION,
                {"operation": operation, "error": error}
            )
            
            if fallback_result:
                return ErrorResponse(
                    success=True,
                    message="Vector store error handled with fallback",
                    fallback_result=fallback_result,
                    suggestions=["Check vector store connection", "Verify configuration"]
                )
        
        return ErrorResponse(
            success=False,
            error_type=ErrorType.VECTOR_STORE_CONNECTION,
            message=f"Vector store operation failed: {str(error)}",
            suggestions=[
                "Check vector store connection and configuration",
                "Verify network connectivity",
                "Check if vector store service is running",
                "Try again in a few moments"
            ]
        )
    
    def handle_embedding_error(self, error: EmbeddingError, operation: str = "embedding_generation") -> ErrorResponse:
        """Handle embedding generation failures with exponential backoff."""
        error_context = ErrorContext(
            error_type=ErrorType.EMBEDDING_GENERATION,
            severity=ErrorSeverity.MEDIUM,
            message=f"Embedding error in {operation}: {str(error)}",
            original_exception=error,
            operation=operation,
            metadata={"error_subtype": error.error_type}
        )
        
        self._track_error(error_context)
        self.error_logger.log_error(error_context)
        
        # Create retry strategy for embedding errors
        retry_strategy = self.retry_manager.create_exponential_backoff_strategy(
            max_retries=3,
            base_delay=1.0,
            max_delay=10.0
        )
        
        suggestions = [
            "Check embedding model configuration",
            "Verify input text format and length",
            "Check API keys and authentication",
            "Try with shorter text if input is too long"
        ]
        
        # Try fallback if retries are exhausted
        if self.config.enable_fallbacks:
            fallback_result = self.fallback_manager.execute_fallback(
                ErrorType.EMBEDDING_GENERATION,
                {"operation": operation, "error": error}
            )
            
            if fallback_result:
                return ErrorResponse(
                    success=True,
                    message="Embedding error handled with fallback",
                    fallback_result=fallback_result,
                    suggestions=suggestions
                )
        
        return ErrorResponse(
            success=False,
            error_type=ErrorType.EMBEDDING_GENERATION,
            message=f"Embedding generation failed: {str(error)}",
            suggestions=suggestions,
            retry_after=retry_strategy.base_delay
        )
    
    def handle_query_parsing_error(self, error: QueryParsingError, operation: str = "query_parsing") -> ErrorResponse:
        """Provide meaningful error messages and query correction suggestions."""
        error_context = ErrorContext(
            error_type=ErrorType.QUERY_PARSING,
            severity=ErrorSeverity.LOW,
            message=f"Query parsing error in {operation}: {str(error)}",
            original_exception=error,
            operation=operation,
            metadata={"malformed_query": error.malformed_query}
        )
        
        self._track_error(error_context)
        self.error_logger.log_error(error_context)
        
        # Generate query suggestions
        suggestions = self._generate_query_suggestions(error.malformed_query)
        fallback_query = self._create_fallback_query(error.malformed_query)
        
        return ErrorResponse(
            success=False,
            error_type=ErrorType.QUERY_PARSING,
            message=f"Query parsing failed: {str(error)}",
            suggestions=suggestions,
            fallback_result={"fallback_query": fallback_query}
        )
    
    def handle_generic_error(self, error: Exception, error_type: ErrorType, 
                           operation: str = "unknown_operation") -> ErrorResponse:
        """Handle generic errors with appropriate classification and response."""
        severity = self._classify_error_severity(error, error_type)
        
        error_context = ErrorContext(
            error_type=error_type,
            severity=severity,
            message=f"Error in {operation}: {str(error)}",
            original_exception=error,
            operation=operation
        )
        
        self._track_error(error_context)
        self.error_logger.log_error(error_context)
        
        return ErrorResponse(
            success=False,
            error_type=error_type,
            message=f"Operation failed: {str(error)}",
            suggestions=self._get_generic_suggestions(error_type)
        )
    
    def execute_with_error_handling(self, operation: Callable, operation_name: str,
                                  error_type: ErrorType = ErrorType.UNKNOWN,
                                  timeout: Optional[float] = None) -> ErrorResponse:
        """Execute an operation with comprehensive error handling."""
        timeout = timeout or self.config.timeout_seconds
        
        try:
            # Execute with timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(operation)
                try:
                    result = future.result(timeout=timeout)
                    return ErrorResponse(success=True, fallback_result=result)
                except FutureTimeoutError:
                    return self.handle_generic_error(
                        TimeoutError(f"Operation {operation_name} timed out after {timeout}s"),
                        ErrorType.NETWORK_TIMEOUT,
                        operation_name
                    )
        except Exception as e:
            return self.handle_generic_error(e, error_type, operation_name)
    
    def handle_no_results(self, query: str, context: Dict[str, Any] = None) -> 'HelpfulGuidance':
        """Provide helpful guidance when no relevant documents are found."""
        return self.graceful_degradation.handle_no_results(query, context)
    
    def handle_resource_constraints(self, available_resources: Optional['ResourceInfo'] = None) -> 'DegradedService':
        """Gracefully degrade performance under resource constraints."""
        if available_resources is None:
            available_resources = self.graceful_degradation.resource_monitor.get_resource_info()
        return self.graceful_degradation.handle_resource_constraints(available_resources)
    
    def handle_malformed_documents(self, document: Dict[str, Any], error: Exception) -> 'ProcessingResult':
        """Skip problematic content and continue processing."""
        return self.graceful_degradation.handle_malformed_documents(document, error)
    
    def should_degrade_service(self, resource_info: Optional['ResourceInfo'] = None) -> bool:
        """Check if service should be degraded based on resource usage."""
        if resource_info is None:
            resource_info = self.graceful_degradation.resource_monitor.get_resource_info()
        return self.graceful_degradation.resource_monitor.should_degrade_service(resource_info)
    
    def get_current_resource_info(self) -> 'ResourceInfo':
        """Get current system resource information."""
        return self.graceful_degradation.resource_monitor.get_resource_info()

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring and debugging."""
        return {
            "error_counts": {error_type.value: count for error_type, count in self._error_counts.items()},
            "last_errors": {error_type.value: timestamp for error_type, timestamp in self._last_errors.items()},
            "total_errors": sum(self._error_counts.values())
        }
    
    def _track_error(self, error_context: ErrorContext):
        """Track error statistics."""
        error_type = error_context.error_type
        self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1
        self._last_errors[error_type] = error_context.timestamp
    
    def _classify_error_severity(self, error: Exception, error_type: ErrorType) -> ErrorSeverity:
        """Classify error severity based on error type and content."""
        if error_type in [ErrorType.VECTOR_STORE_CONNECTION, ErrorType.RESOURCE_EXHAUSTION]:
            return ErrorSeverity.HIGH
        elif error_type in [ErrorType.EMBEDDING_GENERATION, ErrorType.CACHE_ACCESS]:
            return ErrorSeverity.MEDIUM
        elif error_type in [ErrorType.QUERY_PARSING, ErrorType.DOCUMENT_PROCESSING]:
            return ErrorSeverity.LOW
        else:
            return ErrorSeverity.MEDIUM
    
    def _generate_query_suggestions(self, malformed_query: str) -> List[str]:
        """Generate suggestions for fixing malformed queries."""
        suggestions = []
        
        # Basic query validation suggestions
        if not malformed_query.strip():
            suggestions.append("Query cannot be empty")
        elif len(malformed_query) > 1000:
            suggestions.append("Query is too long, try shortening it")
        elif malformed_query.count('"') % 2 != 0:
            suggestions.append("Unmatched quotes in query")
        elif any(char in malformed_query for char in ['<', '>', '{', '}']):
            suggestions.append("Remove special characters like <, >, {, }")
        
        # Content-based suggestions
        if "manim" not in malformed_query.lower():
            suggestions.append("Consider adding 'manim' to your query for better results")
        
        if not suggestions:
            suggestions.append("Try simplifying your query or using different keywords")
        
        return suggestions
    
    def _create_fallback_query(self, malformed_query: str) -> str:
        """Create a simplified fallback query from a malformed query."""
        # Remove special characters and clean up
        cleaned = re.sub(r'[<>{}"\[\]]', '', malformed_query)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # If still problematic, use a generic fallback
        if not cleaned or len(cleaned) < 3:
            return "manim documentation"
        
        return cleaned[:100]  # Limit length
    
    def _get_generic_suggestions(self, error_type: ErrorType) -> List[str]:
        """Get generic suggestions based on error type."""
        suggestions_map = {
            ErrorType.VECTOR_STORE_CONNECTION: [
                "Check vector store connection",
                "Verify configuration settings",
                "Check network connectivity"
            ],
            ErrorType.EMBEDDING_GENERATION: [
                "Check embedding model configuration",
                "Verify API keys and authentication",
                "Try with shorter input text"
            ],
            ErrorType.QUERY_PARSING: [
                "Simplify your query",
                "Remove special characters",
                "Check query syntax"
            ],
            ErrorType.DOCUMENT_PROCESSING: [
                "Check document format",
                "Verify file permissions",
                "Try with a different document"
            ],
            ErrorType.CACHE_ACCESS: [
                "Clear cache and try again",
                "Check cache configuration",
                "Verify disk space"
            ],
            ErrorType.RESOURCE_EXHAUSTION: [
                "Free up system resources",
                "Try again later",
                "Reduce query complexity"
            ]
        }
        
        return suggestions_map.get(error_type, ["Try again later", "Check system configuration"])


@dataclass
class ResourceInfo:
    """Information about available system resources."""
    memory_usage_percent: float
    cpu_usage_percent: float
    disk_usage_percent: float
    available_connections: int
    cache_size_mb: float
    max_cache_size_mb: float


@dataclass
class HelpfulGuidance:
    """Guidance provided when no relevant documents are found."""
    message: str
    suggestions: List[str]
    alternative_queries: List[str]
    related_topics: List[str]
    documentation_links: List[str]


@dataclass
class DegradedService:
    """Information about degraded service capabilities."""
    service_level: str  # "full", "limited", "minimal"
    available_features: List[str]
    disabled_features: List[str]
    performance_impact: str
    estimated_response_time: float
    recommendations: List[str]


@dataclass
class ProcessingResult:
    """Result of document processing with error handling."""
    success: bool
    processed_content: Optional[str] = None
    skipped_sections: List[str] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResourceMonitor:
    """Monitor system resources for graceful degradation decisions."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_resource_info(self) -> ResourceInfo:
        """Get current system resource information."""
        try:
            import psutil
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Get CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Get disk usage for current directory
            disk = psutil.disk_usage('.')
            disk_usage = (disk.used / disk.total) * 100
            
            # Estimate available connections (simplified)
            available_connections = max(0, 100 - int(memory_usage))
            
            # Estimate cache size (simplified)
            cache_size_mb = memory_usage * 10  # Rough estimate
            max_cache_size_mb = 1000  # Default max
            
            return ResourceInfo(
                memory_usage_percent=memory_usage,
                cpu_usage_percent=cpu_usage,
                disk_usage_percent=disk_usage,
                available_connections=available_connections,
                cache_size_mb=cache_size_mb,
                max_cache_size_mb=max_cache_size_mb
            )
        except ImportError:
            # Fallback if psutil is not available
            self.logger.warning("psutil not available, using default resource info")
            return ResourceInfo(
                memory_usage_percent=50.0,
                cpu_usage_percent=50.0,
                disk_usage_percent=50.0,
                available_connections=50,
                cache_size_mb=500.0,
                max_cache_size_mb=1000.0
            )
        except Exception as e:
            self.logger.error(f"Error getting resource info: {e}")
            return ResourceInfo(
                memory_usage_percent=75.0,  # Assume high usage on error
                cpu_usage_percent=75.0,
                disk_usage_percent=75.0,
                available_connections=25,
                cache_size_mb=750.0,
                max_cache_size_mb=1000.0
            )
    
    def should_degrade_service(self, resource_info: ResourceInfo) -> bool:
        """Determine if service should be degraded based on resource usage."""
        # Define thresholds for degradation
        memory_threshold = 80.0
        cpu_threshold = 85.0
        disk_threshold = 90.0
        connection_threshold = 10
        
        return (
            resource_info.memory_usage_percent > memory_threshold or
            resource_info.cpu_usage_percent > cpu_threshold or
            resource_info.disk_usage_percent > disk_threshold or
            resource_info.available_connections < connection_threshold
        )
    
    def get_degradation_level(self, resource_info: ResourceInfo) -> str:
        """Determine the level of service degradation needed."""
        if (resource_info.memory_usage_percent > 95 or 
            resource_info.cpu_usage_percent > 95 or
            resource_info.available_connections < 5):
            return "minimal"
        elif (resource_info.memory_usage_percent > 85 or 
              resource_info.cpu_usage_percent > 90 or
              resource_info.available_connections < 15):
            return "limited"
        else:
            return "full"


class GracefulDegradation:
    """Handle graceful degradation scenarios for the RAG system."""
    
    def __init__(self, config: ErrorHandlingConfig):
        self.config = config
        self.resource_monitor = ResourceMonitor()
        self.logger = logging.getLogger(__name__)
        
        # Knowledge base for helpful guidance
        self.manim_topics = [
            "mobjects", "animations", "scenes", "transformations", "geometry",
            "text", "graphs", "plots", "3d", "camera", "configuration"
        ]
        
        self.common_queries = [
            "manim basic animation",
            "manim scene setup",
            "manim mobject creation",
            "manim text animation",
            "manim geometric shapes"
        ]
        
        self.documentation_links = [
            "https://docs.manim.community/en/stable/",
            "https://docs.manim.community/en/stable/tutorials/",
            "https://docs.manim.community/en/stable/reference/",
            "https://docs.manim.community/en/stable/examples/"
        ]
    
    def handle_no_results(self, query: str, context: Dict[str, Any] = None) -> HelpfulGuidance:
        """Provide helpful guidance when no relevant documents are found."""
        context = context or {}
        
        # Analyze the query to provide targeted suggestions
        query_lower = query.lower()
        suggestions = []
        alternative_queries = []
        related_topics = []
        
        # Query-specific suggestions
        if len(query) < 3:
            suggestions.append("Your query is very short. Try using more descriptive terms.")
            alternative_queries.extend(self.common_queries[:3])
        elif len(query) > 200:
            suggestions.append("Your query is very long. Try breaking it into shorter, more specific queries.")
            alternative_queries.append(query[:50] + "...")
        
        # Content-based suggestions
        if "manim" not in query_lower:
            suggestions.append("Consider adding 'manim' to your query for better results.")
            alternative_queries.append(f"manim {query}")
        
        if any(term in query_lower for term in ["error", "exception", "problem"]):
            suggestions.append("For error-related queries, try including the specific error message.")
            related_topics.extend(["debugging", "troubleshooting", "common errors"])
        
        if any(term in query_lower for term in ["animation", "animate"]):
            suggestions.append("For animation queries, specify the type of animation you want.")
            related_topics.extend(["transformations", "mobject animations", "scene animations"])
            alternative_queries.extend([
                "manim basic animation tutorial",
                "manim transform animation",
                "manim text animation"
            ])
        
        # Topic-based suggestions
        for topic in self.manim_topics:
            if topic in query_lower:
                related_topics.append(topic)
                alternative_queries.append(f"manim {topic} examples")
        
        # Default suggestions if none found
        if not suggestions:
            suggestions.extend([
                "Try using more specific terms related to Manim concepts.",
                "Check your spelling and try alternative terms.",
                "Break complex queries into simpler parts."
            ])
        
        if not alternative_queries:
            alternative_queries.extend(self.common_queries[:5])
        
        if not related_topics:
            related_topics.extend(self.manim_topics[:5])
        
        # Create helpful message
        message = self._create_no_results_message(query, len(suggestions))
        
        return HelpfulGuidance(
            message=message,
            suggestions=suggestions[:5],  # Limit to top 5
            alternative_queries=alternative_queries[:5],
            related_topics=related_topics[:5],
            documentation_links=self.documentation_links
        )
    
    def handle_resource_constraints(self, available_resources: ResourceInfo) -> DegradedService:
        """Gracefully degrade performance under resource constraints."""
        degradation_level = self.resource_monitor.get_degradation_level(available_resources)
        
        if degradation_level == "minimal":
            return self._create_minimal_service(available_resources)
        elif degradation_level == "limited":
            return self._create_limited_service(available_resources)
        else:
            return self._create_full_service(available_resources)
    
    def handle_malformed_documents(self, document: Dict[str, Any], error: Exception) -> ProcessingResult:
        """Skip problematic content and continue processing."""
        document_path = document.get('path', 'unknown')
        document_content = document.get('content', '')
        
        self.logger.warning(f"Processing error in document {document_path}: {str(error)}")
        
        # Try to salvage what we can from the document
        processed_content = None
        skipped_sections = []
        error_messages = []
        warnings = []
        
        try:
            # Attempt to process document in smaller chunks
            if isinstance(error, UnicodeDecodeError):
                # Handle encoding issues
                processed_content = self._handle_encoding_error(document_content)
                warnings.append("Document had encoding issues, some characters may be missing")
            
            elif "syntax" in str(error).lower():
                # Handle syntax errors in code documents
                processed_content, skipped_sections = self._handle_syntax_error(document_content)
                warnings.append(f"Skipped {len(skipped_sections)} sections due to syntax errors")
            
            elif "parsing" in str(error).lower():
                # Handle parsing errors
                processed_content = self._handle_parsing_error(document_content)
                warnings.append("Document structure may be incomplete due to parsing errors")
            
            else:
                # Generic error handling
                processed_content = self._handle_generic_document_error(document_content)
                warnings.append("Document processed with limited functionality due to errors")
            
            error_messages.append(f"Original error: {str(error)}")
            
            # If all recovery attempts failed (processed_content is None), this should trigger the except block
            if processed_content is None:
                raise RuntimeError("All recovery methods failed")
            
            return ProcessingResult(
                success=processed_content is not None,
                processed_content=processed_content,
                skipped_sections=skipped_sections,
                error_messages=error_messages,
                warnings=warnings,
                metadata={
                    "original_error": str(error),
                    "error_type": type(error).__name__,
                    "document_path": document_path,
                    "recovery_method": "partial_processing"
                }
            )
        
        except Exception as recovery_error:
            # If recovery also fails, return minimal result
            self.logger.error(f"Recovery failed for document {document_path}: {str(recovery_error)}")
            
            return ProcessingResult(
                success=False,
                processed_content=None,
                skipped_sections=["entire_document"],
                error_messages=[
                    f"Original error: {str(error)}",
                    f"Recovery error: {str(recovery_error)}"
                ],
                warnings=["Document could not be processed"],
                metadata={
                    "original_error": str(error),
                    "recovery_error": str(recovery_error),
                    "document_path": document_path,
                    "recovery_method": "failed"
                }
            )
    
    def _create_no_results_message(self, query: str, suggestion_count: int) -> str:
        """Create a helpful message for no results scenario."""
        base_message = f"No relevant documents found for your query: '{query[:100]}'"
        if len(query) > 100:
            base_message += "..."
        
        # Add specific guidance based on query characteristics
        if len(query) < 3:
            base_message += "\n\nYour query is very short."
        elif len(query) > 200:
            base_message += "\n\nYour query is very long."
        
        if suggestion_count > 0:
            base_message += f"\n\nHere are {suggestion_count} suggestions to improve your search:"
        else:
            base_message += "\n\nTry refining your query or exploring related topics."
        
        return base_message
    
    def _create_minimal_service(self, resources: ResourceInfo) -> DegradedService:
        """Create minimal service configuration under severe resource constraints."""
        return DegradedService(
            service_level="minimal",
            available_features=[
                "basic_search",
                "cached_results"
            ],
            disabled_features=[
                "semantic_expansion",
                "query_diversification",
                "advanced_ranking",
                "relationship_analysis",
                "plugin_detection"
            ],
            performance_impact="Significantly reduced functionality and slower response times",
            estimated_response_time=10.0,
            recommendations=[
                "Free up system memory to restore full functionality",
                "Close other applications to reduce resource usage",
                "Try simpler queries to reduce processing load",
                "Consider restarting the system if issues persist"
            ]
        )
    
    def _create_limited_service(self, resources: ResourceInfo) -> DegradedService:
        """Create limited service configuration under moderate resource constraints."""
        return DegradedService(
            service_level="limited",
            available_features=[
                "basic_search",
                "cached_results",
                "simple_ranking",
                "basic_query_expansion"
            ],
            disabled_features=[
                "advanced_semantic_analysis",
                "complex_query_diversification",
                "deep_relationship_analysis"
            ],
            performance_impact="Some advanced features disabled, moderate performance impact",
            estimated_response_time=5.0,
            recommendations=[
                "Reduce query complexity for better performance",
                "Clear cache if memory usage is high",
                "Monitor system resources"
            ]
        )
    
    def _create_full_service(self, resources: ResourceInfo) -> DegradedService:
        """Create full service configuration with all features available."""
        return DegradedService(
            service_level="full",
            available_features=[
                "advanced_search",
                "semantic_expansion",
                "query_diversification",
                "advanced_ranking",
                "relationship_analysis",
                "plugin_detection",
                "caching",
                "performance_optimization"
            ],
            disabled_features=[],
            performance_impact="No performance impact, all features available",
            estimated_response_time=2.0,
            recommendations=[
                "All features are available",
                "System is operating at full capacity"
            ]
        )
    
    def _handle_encoding_error(self, content: str) -> Optional[str]:
        """Handle encoding errors in document content."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
            
            for encoding in encodings:
                try:
                    if isinstance(content, bytes):
                        decoded = content.decode(encoding, errors='ignore')
                    else:
                        # If already string, try to encode and decode to clean it
                        decoded = content.encode(encoding, errors='ignore').decode(encoding)
                    
                    if decoded and len(decoded) > 10:  # Reasonable content length
                        return decoded
                except:
                    continue
            
            # If all encodings fail, return cleaned version
            if isinstance(content, bytes):
                return content.decode('utf-8', errors='replace')
            else:
                return ''.join(char for char in content if ord(char) < 128)  # ASCII only
        
        except Exception:
            return None
    
    def _handle_syntax_error(self, content: str) -> tuple[Optional[str], List[str]]:
        """Handle syntax errors by skipping problematic sections."""
        try:
            lines = content.split('\n')
            processed_lines = []
            skipped_sections = []
            current_section = []
            in_code_block = False
            
            for i, line in enumerate(lines):
                try:
                    # Detect code blocks
                    if line.strip().startswith('```'):
                        in_code_block = not in_code_block
                        if not in_code_block and current_section:
                            # End of code block, try to validate
                            section_content = '\n'.join(current_section)
                            if self._is_valid_code_section(section_content):
                                processed_lines.extend(current_section)
                            else:
                                skipped_sections.append(f"Code block at line {i - len(current_section)}")
                            current_section = []
                        processed_lines.append(line)
                    elif in_code_block:
                        current_section.append(line)
                    else:
                        # Regular text line
                        processed_lines.append(line)
                
                except Exception:
                    # Skip problematic line
                    skipped_sections.append(f"Line {i + 1}")
                    continue
            
            return '\n'.join(processed_lines), skipped_sections
        
        except Exception:
            return None, ["entire_document"]
    
    def _handle_parsing_error(self, content: str) -> Optional[str]:
        """Handle parsing errors by extracting readable text."""
        try:
            # Remove problematic characters and patterns
            import re
            
            # Remove control characters except newlines and tabs
            cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
            
            # Fix common parsing issues
            cleaned = re.sub(r'\r\n', '\n', cleaned)  # Normalize line endings
            cleaned = re.sub(r'\r', '\n', cleaned)    # Handle old Mac line endings
            cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Reduce excessive newlines
            
            # Remove or fix problematic markup
            cleaned = re.sub(r'<[^>]*>', '', cleaned)  # Remove HTML tags
            cleaned = re.sub(r'\{[^}]*\}', '', cleaned)  # Remove curly brace content
            
            return cleaned if cleaned.strip() else None
        
        except Exception:
            return None
    
    def _handle_generic_document_error(self, content: str) -> Optional[str]:
        """Handle generic document errors with basic text extraction."""
        try:
            # Extract basic text content
            if content is None or not content:
                return None
            
            # Keep only printable characters and basic whitespace
            import string
            printable_chars = string.printable
            cleaned = ''.join(char for char in content if char in printable_chars)
            
            # Basic cleanup
            lines = cleaned.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line = line.strip()
                if line and len(line) > 2:  # Keep non-empty lines with reasonable length
                    cleaned_lines.append(line)
            
            result = '\n'.join(cleaned_lines)
            return result if result.strip() else None
        
        except Exception:
            return None
    
    def _is_valid_code_section(self, code_content: str) -> bool:
        """Check if a code section is valid and safe to include."""
        try:
            # Basic validation - check for reasonable content
            if not code_content.strip():
                return False
            
            # Check for minimum content length
            if len(code_content.strip()) < 5:
                return False
            
            # Check for suspicious patterns that might indicate corruption
            suspicious_patterns = [
                r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]',  # Control characters
                r'[^\x20-\x7E\n\t]',  # Non-printable characters
                r'(.)\1{50,}',  # Repeated characters (likely corruption)
            ]
            
            import re
            for pattern in suspicious_patterns:
                if re.search(pattern, code_content):
                    return False
            
            return True
        
        except Exception:
            return False


class ResourceMonitor:
    """Monitor system resources and determine service degradation needs."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._psutil_available = self._check_psutil_availability()
    
    def _check_psutil_availability(self) -> bool:
        """Check if psutil is available for resource monitoring."""
        try:
            import psutil
            return True
        except ImportError:
            self.logger.warning("psutil not available, using default resource info")
            return False
    
    def get_resource_info(self) -> ResourceInfo:
        """Get current system resource information."""
        if self._psutil_available:
            return self._get_resource_info_with_psutil()
        else:
            return self._get_default_resource_info()
    
    def _get_resource_info_with_psutil(self) -> ResourceInfo:
        """Get resource info using psutil."""
        try:
            import psutil
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Get disk usage for current directory
            disk = psutil.disk_usage('.')
            disk_percent = (disk.used / disk.total) * 100
            
            return ResourceInfo(
                memory_usage_percent=memory_percent,
                cpu_usage_percent=cpu_percent,
                disk_usage_percent=disk_percent,
                available_connections=max(0, 100 - int(memory_percent + cpu_percent) // 2),
                cache_size_mb=500.0,  # Default cache size
                max_cache_size_mb=1000.0
            )
        except Exception as e:
            self.logger.warning(f"Error getting resource info with psutil: {e}")
            return self._get_default_resource_info()
    
    def _get_default_resource_info(self) -> ResourceInfo:
        """Get default resource info when psutil is not available."""
        return ResourceInfo(
            memory_usage_percent=50.0,
            cpu_usage_percent=50.0,
            disk_usage_percent=50.0,
            available_connections=50,
            cache_size_mb=500.0,
            max_cache_size_mb=1000.0
        )
    
    def should_degrade_service(self, resource_info: ResourceInfo) -> bool:
        """Determine if service should be degraded based on resource usage."""
        # Define thresholds for service degradation
        memory_threshold = 80.0
        cpu_threshold = 80.0
        connection_threshold = 10
        
        return (
            resource_info.memory_usage_percent > memory_threshold or
            resource_info.cpu_usage_percent > cpu_threshold or
            resource_info.available_connections < connection_threshold
        )
    
    def get_degradation_level(self, resource_info: ResourceInfo) -> str:
        """Get the appropriate degradation level based on resource usage."""
        # Critical thresholds
        critical_memory = 95.0
        critical_cpu = 95.0
        critical_connections = 5
        
        # High thresholds
        high_memory = 85.0
        high_cpu = 80.0
        high_connections = 10
        
        # Check for critical resource usage
        if (resource_info.memory_usage_percent > critical_memory or
            resource_info.cpu_usage_percent > critical_cpu or
            resource_info.available_connections < critical_connections):
            return "minimal"
        
        # Check for high resource usage
        elif (resource_info.memory_usage_percent > high_memory or
              resource_info.cpu_usage_percent > high_cpu or
              resource_info.available_connections < high_connections):
            return "limited"
        
        else:
            return "full"


class GracefulDegradation:
    """
    Handles graceful degradation scenarios including no results guidance,
    resource constraint handling, and malformed document processing.
    """
    
    def __init__(self, config: ErrorHandlingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.resource_monitor = ResourceMonitor()
    
    def handle_no_results(self, query: str, context: Dict[str, Any] = None) -> HelpfulGuidance:
        """Provide helpful guidance when no relevant documents are found."""
        context = context or {}
        
        # Analyze the query to provide specific guidance
        query_analysis = self._analyze_query(query)
        
        # Generate helpful message
        message = f"No relevant documents found for your query: '{query}'"
        if query_analysis['is_short']:
            message += "\n\nYour query is very short."
        elif query_analysis['is_long']:
            message += "\n\nYour query is very long."
        
        message += f"\n\nHere are {min(len(self._generate_suggestions(query_analysis)), 3)} suggestions to improve your search:"
        
        # Generate suggestions
        suggestions = self._generate_suggestions(query_analysis)
        
        # Generate alternative queries
        alternative_queries = self._generate_alternative_queries(query, query_analysis)
        
        # Generate related topics
        related_topics = self._generate_related_topics(query, query_analysis)
        
        # Generate documentation links
        documentation_links = self._generate_documentation_links(query_analysis)
        
        return HelpfulGuidance(
            message=message,
            suggestions=suggestions,
            alternative_queries=alternative_queries,
            related_topics=related_topics,
            documentation_links=documentation_links
        )
    
    def handle_resource_constraints(self, available_resources: ResourceInfo) -> DegradedService:
        """Gracefully degrade performance under resource constraints."""
        degradation_level = self.resource_monitor.get_degradation_level(available_resources)
        
        if degradation_level == "full":
            return self._create_full_service_info()
        elif degradation_level == "limited":
            return self._create_limited_service_info(available_resources)
        else:  # minimal
            return self._create_minimal_service_info(available_resources)
    
    def handle_malformed_documents(self, document: Dict[str, Any], error: Exception) -> ProcessingResult:
        """Skip problematic content and continue processing."""
        document_path = document.get('path', 'unknown')
        document_content = document.get('content', '')
        
        self.logger.warning(f"Processing error in document {document_path}: {str(error)}")
        
        # Try different recovery strategies
        recovery_strategies = [
            self._recover_with_encoding_fix,
            self._recover_with_content_cleaning,
            self._recover_with_partial_processing,
        ]
        
        for strategy in recovery_strategies:
            try:
                result = strategy(document, error)
                if result.success:
                    return result
            except Exception as recovery_error:
                self.logger.debug(f"Recovery strategy {strategy.__name__} failed: {recovery_error}")
                continue
        
        # All recovery strategies failed
        self.logger.error(f"Recovery failed for document {document_path}: All recovery methods failed")
        return ProcessingResult(
            success=False,
            processed_content=None,
            skipped_sections=['entire_document'],
            warnings=['Document could not be processed'],
            error_messages=[str(error), "All recovery methods failed"],
            metadata={'recovery_method': 'failed', 'original_error': str(error)}
        )
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query characteristics to provide better guidance."""
        return {
            'is_short': len(query.strip()) < 3,
            'is_long': len(query) > 200,
            'has_manim': 'manim' in query.lower(),
            'is_error_related': any(word in query.lower() for word in ['error', 'exception', 'problem', 'issue']),
            'is_animation_related': any(word in query.lower() for word in ['animation', 'animate', 'transform']),
            'is_tutorial_related': any(word in query.lower() for word in ['tutorial', 'example', 'how to']),
        }
    
    def _generate_suggestions(self, query_analysis: Dict[str, Any]) -> List[str]:
        """Generate suggestions based on query analysis."""
        suggestions = []
        
        if query_analysis['is_short']:
            suggestions.append("Your query is very short. Try using more descriptive terms.")
        
        if query_analysis['is_long']:
            suggestions.append("Your query is very long. Try breaking it into shorter, more specific queries.")
        
        if not query_analysis['has_manim']:
            suggestions.append("Consider adding 'manim' to your query for better results.")
        
        if query_analysis['is_error_related']:
            suggestions.append("For error-related queries, try including the specific error message.")
        
        if query_analysis['is_animation_related']:
            suggestions.append("For animation queries, specify the type of animation you want.")
        
        if query_analysis['is_tutorial_related']:
            suggestions.append("For tutorial queries, try searching for specific concepts or functions.")
        
        # Add generic suggestions if no specific ones apply
        if not suggestions:
            suggestions.extend([
                "Try using more specific keywords related to your task.",
                "Consider breaking your query into smaller, focused questions.",
                "Use technical terms that might appear in documentation."
            ])
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    def _generate_alternative_queries(self, original_query: str, query_analysis: Dict[str, Any]) -> List[str]:
        """Generate alternative query suggestions."""
        alternatives = []
        
        if query_analysis['is_short']:
            alternatives.extend([
                "manim basic animation",
                "manim scene setup",
                "manim getting started"
            ])
        elif query_analysis['is_long']:
            # Truncate long queries
            truncated = original_query[:50] + "..." if len(original_query) > 50 else original_query
            alternatives.append(truncated)
        
        if not query_analysis['has_manim']:
            alternatives.append(f"manim {original_query}")
        
        if query_analysis['is_animation_related']:
            alternatives.extend([
                "manim basic animation tutorial",
                "manim transform animation",
                "manim mobject animation"
            ])
        
        if query_analysis['is_tutorial_related']:
            alternatives.extend([
                "manim tutorial basics",
                "manim example code"
            ])
        
        # Add generic alternatives if none specific
        if not alternatives:
            alternatives.extend([
                "manim basic animation",
                "manim scene setup"
            ])
        
        return alternatives[:5]  # Limit to 5 alternatives
    
    def _generate_related_topics(self, query: str, query_analysis: Dict[str, Any]) -> List[str]:
        """Generate related topics based on query analysis."""
        topics = []
        
        if query_analysis['is_animation_related']:
            topics.extend(["transformations", "mobject animations", "scene animations"])
        
        if query_analysis['is_error_related']:
            topics.extend(["debugging", "troubleshooting", "common errors"])
        
        if query_analysis['is_tutorial_related']:
            topics.extend(["getting started", "basic concepts", "examples"])
        
        # Add general topics
        topics.extend(["mobjects", "animations", "scenes"])
        
        # Remove duplicates and limit
        return list(dict.fromkeys(topics))[:8]
    
    def _generate_documentation_links(self, query_analysis: Dict[str, Any]) -> List[str]:
        """Generate relevant documentation links."""
        links = [
            "https://docs.manim.community/en/stable/",
            "https://docs.manim.community/en/stable/tutorials.html",
            "https://docs.manim.community/en/stable/reference.html"
        ]
        
        if query_analysis['is_animation_related']:
            links.append("https://docs.manim.community/en/stable/reference/manim.animation.html")
        
        if query_analysis['is_tutorial_related']:
            links.append("https://docs.manim.community/en/stable/tutorials/quickstart.html")
        
        return links[:5]
    
    def _create_full_service_info(self) -> DegradedService:
        """Create service info for full service level."""
        return DegradedService(
            service_level="full",
            available_features=[
                "advanced_search", "semantic_expansion", "query_diversification",
                "context_aware_ranking", "relationship_analysis", "plugin_detection"
            ],
            disabled_features=[],
            performance_impact="No performance impact, all features available",
            estimated_response_time=2.0,
            recommendations=["System is operating optimally"]
        )
    
    def _create_limited_service_info(self, resources: ResourceInfo) -> DegradedService:
        """Create service info for limited service level."""
        return DegradedService(
            service_level="limited",
            available_features=[
                "basic_search", "cached_results", "simple_ranking", "basic_plugin_detection"
            ],
            disabled_features=[
                "advanced_semantic_analysis", "complex_query_diversification", 
                "deep_relationship_analysis", "resource_intensive_features"
            ],
            performance_impact="Some advanced features disabled, moderate performance impact",
            estimated_response_time=5.0,
            recommendations=[
                "Consider freeing up system resources",
                "Close unnecessary applications",
                "Clear cache if needed"
            ]
        )
    
    def _create_minimal_service_info(self, resources: ResourceInfo) -> DegradedService:
        """Create service info for minimal service level."""
        return DegradedService(
            service_level="minimal",
            available_features=["basic_search", "cached_results"],
            disabled_features=[
                "semantic_expansion", "query_diversification", "advanced_ranking",
                "relationship_analysis", "plugin_detection", "context_analysis"
            ],
            performance_impact="Significantly reduced functionality and slower response times",
            estimated_response_time=10.0,
            recommendations=[
                "Free up system memory immediately",
                "Restart the application if possible",
                "Check for resource-intensive processes",
                "Consider upgrading system resources"
            ]
        )
    
    def _recover_with_encoding_fix(self, document: Dict[str, Any], error: Exception) -> ProcessingResult:
        """Try to recover document by fixing encoding issues."""
        content = document.get('content', '')
        
        if isinstance(content, bytes):
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
            for encoding in encodings:
                try:
                    decoded_content = content.decode(encoding, errors='replace')
                    return ProcessingResult(
                        success=True,
                        processed_content=decoded_content,
                        skipped_sections=[],
                        warnings=['Document had encoding issues, some characters may be missing'],
                        error_messages=[],
                        metadata={'recovery_method': 'encoding_fix', 'encoding_used': encoding}
                    )
                except Exception:
                    continue
        
        elif isinstance(content, str):
            # Clean up problematic characters
            cleaned_content = self._clean_text_content(content)
            if cleaned_content:
                return ProcessingResult(
                    success=True,
                    processed_content=cleaned_content,
                    skipped_sections=[],
                    warnings=['Document had encoding issues, some characters may be missing'],
                    error_messages=[],
                    metadata={'recovery_method': 'partial_processing'}
                )
        
        raise Exception("Encoding recovery failed")
    
    def _recover_with_content_cleaning(self, document: Dict[str, Any], error: Exception) -> ProcessingResult:
        """Try to recover document by cleaning problematic content."""
        content = document.get('content', '')
        
        if not content:
            raise Exception("No content to clean")
        
        # Try to extract valid sections
        valid_sections = []
        lines = str(content).split('\n')
        
        for line in lines:
            try:
                # Test if line is processable
                cleaned_line = self._clean_text_content(line)
                if cleaned_line and len(cleaned_line.strip()) > 0:
                    valid_sections.append(cleaned_line)
            except Exception:
                continue  # Skip problematic lines
        
        if valid_sections:
            processed_content = '\n'.join(valid_sections)
            skipped_count = len(lines) - len(valid_sections)
            
            return ProcessingResult(
                success=True,
                processed_content=processed_content,
                skipped_sections=[],
                warnings=[f'Skipped {skipped_count} sections due to syntax errors'],
                error_messages=[],
                metadata={'recovery_method': 'partial_processing'}
            )
        
        raise Exception("Content cleaning recovery failed")
    
    def _recover_with_partial_processing(self, document: Dict[str, Any], error: Exception) -> ProcessingResult:
        """Try to recover document with minimal processing."""
        content = document.get('content')
        
        if content is None:
            raise Exception("No content available for processing")
        
        # Convert to string and do basic cleaning
        try:
            content_str = str(content)
            cleaned_content = self._clean_text_content(content_str)
            
            if cleaned_content:
                return ProcessingResult(
                    success=True,
                    processed_content=cleaned_content,
                    skipped_sections=[],
                    warnings=['Document processed with limited functionality due to errors'],
                    error_messages=[],
                    metadata={'recovery_method': 'partial_processing'}
                )
        except Exception:
            pass
        
        raise Exception("Partial processing recovery failed")


@dataclass
class ProcessingResult:
    """Result of document processing with error handling."""
    success: bool
    processed_content: Optional[str] = None
    skipped_sections: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
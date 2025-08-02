"""
Embedding Provider Interface and Base Classes

This module defines the abstract interface for embedding providers and common
data models used throughout the embedding system. It provides a unified interface
for different embedding providers (local, JINA, etc.) to ensure consistent behavior.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import os
import logging
import time
import random
import requests
from threading import Lock
from collections import deque


@dataclass
class EmbeddingConfig:
    """Configuration for embedding providers."""
    provider: str  # "local" or "jina"
    model_name: str
    dimensions: int
    api_key: Optional[str] = None
    batch_size: int = 100
    timeout: int = 30


class EmbeddingGenerationError(Exception):
    """Exception raised when embedding generation fails."""
    pass


class ProviderConfigurationError(Exception):
    """Exception raised when provider configuration is invalid."""
    pass


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.
    
    This interface defines the contract that all embedding providers must implement
    to ensure consistent behavior across different embedding services.
    """
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize the embedding provider with configuration.
        
        Args:
            config: EmbeddingConfig object containing provider settings
        """
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to generate embeddings for
            
        Returns:
            List of embedding vectors, where each vector is a list of floats
            
        Raises:
            EmbeddingGenerationError: If embedding generation fails
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider.
        
        Returns:
            Integer representing the embedding dimension
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and properly configured.
        
        Returns:
            True if provider is available, False otherwise
        """
        pass
    
    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the provider for logging and debugging.
        
        Returns:
            Dictionary containing provider information including:
            - provider: Provider name
            - model: Model name
            - dimensions: Embedding dimensions
            - available: Whether provider is available
            - Additional provider-specific information
        """
        pass


class ConfigurationManager:
    """Manages configuration loading and validation for embedding providers."""
    
    @staticmethod
    def load_config_from_env() -> EmbeddingConfig:
        """Load embedding configuration from environment variables.
        
        Environment variables:
        - EMBEDDING_PROVIDER: Provider type ("local" or "jina")
        - JINA_API_KEY: API key for JINA embeddings
        - JINA_EMBEDDING_MODEL: JINA model name
        - LOCAL_EMBEDDING_MODEL: Local model name
        - EMBEDDING_BATCH_SIZE: Batch size for processing
        - EMBEDDING_TIMEOUT: Request timeout in seconds
        
        Returns:
            EmbeddingConfig object with loaded configuration
            
        Raises:
            ProviderConfigurationError: If configuration is invalid
        """
        provider = os.getenv('EMBEDDING_PROVIDER', 'local').lower()
        
        try:
            if provider == "jina":
                config = ConfigurationManager._load_jina_config()
            elif provider == "local":
                config = ConfigurationManager._load_local_config()
            else:
                logging.warning(
                    f"Unknown embedding provider: {provider}. "
                    f"Supported providers: local, jina. Falling back to local."
                )
                config = ConfigurationManager._load_local_config()
            
            # Validate the configuration
            ConfigurationManager.validate_config(config)
            return config
            
        except ProviderConfigurationError as e:
            logging.error(f"Configuration error: {e}")
            # Try to create fallback configuration
            try:
                fallback_config = ConfigurationManager._load_local_config()
                ConfigurationManager.validate_config(fallback_config)
                logging.info("Successfully created fallback configuration using local embeddings")
                return fallback_config
            except Exception as fallback_error:
                raise ProviderConfigurationError(
                    f"Failed to load configuration and fallback failed: {e}. "
                    f"Fallback error: {fallback_error}"
                )
    
    @staticmethod
    def _load_jina_config() -> EmbeddingConfig:
        """Load JINA-specific configuration."""
        api_key = os.getenv('JINA_API_KEY')
        model_name = os.getenv('JINA_EMBEDDING_MODEL', 'jina-embeddings-v3')
        
        if not api_key:
            raise ProviderConfigurationError(
                "JINA_API_KEY not found. "
                "Get your free API key at: https://jina.ai/?sui=apikey"
            )
        
        # Validate API key format (basic check)
        if not api_key.strip():
            raise ProviderConfigurationError(
                "JINA_API_KEY is empty. "
                "Get your free API key at: https://jina.ai/?sui=apikey"
            )
        
        # Determine dimensions - support override via environment variable
        model_dimensions = {
            "jina-embeddings-v3": 1024,
            "jina-embeddings-v4": 2048,
            "jina-clip-v2": 1024
        }
        
        # Check for dimension override
        dimension_override = os.getenv('JINA_EMBEDDING_DIMENSIONS')
        if dimension_override:
            try:
                dimensions = int(dimension_override)
                logging.info(f"Using dimension override for JINA: {dimensions} (model default: {model_dimensions.get(model_name, 1024)})")
                
                # Validate dimension override
                if dimensions <= 0:
                    raise ValueError("Dimensions must be positive")
                elif dimensions > 4096:
                    logging.warning(f"Very high dimension count: {dimensions}. This may impact performance.")
                
            except ValueError as e:
                logging.error(f"Invalid JINA_EMBEDDING_DIMENSIONS value '{dimension_override}': {e}. Using model default.")
                dimensions = model_dimensions.get(model_name, 1024)
        else:
            # Use model default dimensions
            if model_name not in model_dimensions:
                logging.warning(
                    f"Unknown JINA model: {model_name}. "
                    f"Supported models: {list(model_dimensions.keys())}. "
                    f"Using default dimensions: 1024"
                )
            dimensions = model_dimensions.get(model_name, 1024)
        
        # Parse and validate batch size and timeout
        try:
            batch_size = int(os.getenv('EMBEDDING_BATCH_SIZE', '100'))
            if batch_size > 100:  # JINA API limit
                logging.warning(f"JINA batch size {batch_size} exceeds API limit. Using 100.")
                batch_size = 100
        except ValueError:
            logging.warning("Invalid EMBEDDING_BATCH_SIZE, using default: 100")
            batch_size = 100
        
        try:
            timeout = int(os.getenv('EMBEDDING_TIMEOUT', '30'))
        except ValueError:
            logging.warning("Invalid EMBEDDING_TIMEOUT, using default: 30")
            timeout = 30
        
        return EmbeddingConfig(
            provider="jina",
            model_name=model_name,
            dimensions=dimensions,
            api_key=api_key,
            batch_size=batch_size,
            timeout=timeout
        )
    
    @staticmethod
    def _load_local_config() -> EmbeddingConfig:
        """Load local embedding configuration."""
        model_name = os.getenv('LOCAL_EMBEDDING_MODEL', 'hf:ibm-granite/granite-embedding-30m-english')
        
        # Parse and validate batch size and timeout
        try:
            batch_size = int(os.getenv('EMBEDDING_BATCH_SIZE', '100'))
        except ValueError:
            logging.warning("Invalid EMBEDDING_BATCH_SIZE, using default: 100")
            batch_size = 100
        
        try:
            timeout = int(os.getenv('EMBEDDING_TIMEOUT', '30'))
        except ValueError:
            logging.warning("Invalid EMBEDDING_TIMEOUT, using default: 30")
            timeout = 30
        
        return EmbeddingConfig(
            provider="local",
            model_name=model_name,
            dimensions=384,  # IBM Granite embedding dimension
            api_key=None,
            batch_size=batch_size,
            timeout=timeout
        )
    
    @staticmethod
    def validate_config(config: EmbeddingConfig) -> None:
        """Validate embedding configuration.
        
        Args:
            config: EmbeddingConfig to validate
            
        Raises:
            ProviderConfigurationError: If configuration is invalid
        """
        if not config.provider:
            raise ProviderConfigurationError("Provider must be specified")
        
        if config.provider not in ["local", "jina"]:
            raise ProviderConfigurationError(
                f"Unsupported provider: {config.provider}. "
                f"Supported providers: local, jina"
            )
        
        if not config.model_name:
            raise ProviderConfigurationError("Model name must be specified")
        
        if config.dimensions <= 0:
            raise ProviderConfigurationError("Dimensions must be positive")
        
        if config.batch_size <= 0:
            raise ProviderConfigurationError("Batch size must be positive")
        
        if config.timeout <= 0:
            raise ProviderConfigurationError("Timeout must be positive")
        
        # JINA-specific validation
        if config.provider == "jina":
            if not config.api_key:
                raise ProviderConfigurationError(
                    "JINA API key is required for JINA provider. "
                    "Get your free API key at: https://jina.ai/?sui=apikey"
                )
    
    @staticmethod
    def get_available_providers() -> List[str]:
        """Get list of available embedding providers based on configuration.
        
        Returns:
            List of available provider names
        """
        providers = ["local"]  # Local is always available
        
        # Check if JINA is available
        if os.getenv('JINA_API_KEY'):
            providers.append("jina")
        
        return providers
    
    @staticmethod
    def create_fallback_config(original_config: EmbeddingConfig) -> EmbeddingConfig:
        """Create fallback configuration when primary provider fails.
        
        Args:
            original_config: The original configuration that failed
            
        Returns:
            Fallback EmbeddingConfig (typically local provider)
        """
        if original_config.provider != "local":
            logging.warning(
                f"Falling back from {original_config.provider} to local embeddings"
            )
            return ConfigurationManager._load_local_config()
        else:
            # If local also fails, we have a serious problem
            raise ProviderConfigurationError(
                "Local embedding provider is not available. "
                "Please check your local model configuration."
            )
    
    @staticmethod
    def detect_configuration_changes() -> bool:
        """Detect if embedding configuration has changed since last load.
        
        Returns:
            True if configuration has changed, False otherwise
        """
        # This is a placeholder for future implementation
        # Could track environment variable changes or config file timestamps
        return False
    
    @staticmethod
    def get_configuration_summary(config: EmbeddingConfig) -> Dict[str, Any]:
        """Get a summary of the current configuration for logging/debugging.
        
        Args:
            config: EmbeddingConfig to summarize
            
        Returns:
            Dictionary with configuration summary (sensitive data masked)
        """
        summary = {
            "provider": config.provider,
            "model_name": config.model_name,
            "dimensions": config.dimensions,
            "batch_size": config.batch_size,
            "timeout": config.timeout,
            "api_key_configured": bool(config.api_key)
        }
        
        if config.api_key:
            # Mask API key for security
            summary["api_key_preview"] = f"{config.api_key[:8]}...{config.api_key[-4:]}"
        
        return summary
    
    @staticmethod
    def validate_environment_variables() -> Dict[str, str]:
        """Validate all embedding-related environment variables.
        
        Returns:
            Dictionary with validation results and recommendations
        """
        validation_results = {}
        
        # Check provider setting
        provider = os.getenv('EMBEDDING_PROVIDER', 'local').lower()
        if provider not in ['local', 'jina']:
            validation_results['EMBEDDING_PROVIDER'] = (
                f"Invalid provider '{provider}'. Use 'local' or 'jina'"
            )
        
        # Check JINA-specific variables if JINA is selected
        if provider == 'jina':
            if not os.getenv('JINA_API_KEY'):
                validation_results['JINA_API_KEY'] = (
                    "Required for JINA provider. Get your key at: https://jina.ai/?sui=apikey"
                )
            
            model = os.getenv('JINA_EMBEDDING_MODEL', 'jina-embeddings-v3')
            supported_models = ['jina-embeddings-v3', 'jina-embeddings-v4', 'jina-clip-v2']
            if model not in supported_models:
                validation_results['JINA_EMBEDDING_MODEL'] = (
                    f"Model '{model}' not recognized. Supported: {supported_models}"
                )
        
        # Check numeric parameters
        for var_name, default_val in [('EMBEDDING_BATCH_SIZE', '100'), ('EMBEDDING_TIMEOUT', '30')]:
            value = os.getenv(var_name, default_val)
            try:
                int_val = int(value)
                if int_val <= 0:
                    validation_results[var_name] = f"Must be positive integer, got: {value}"
            except ValueError:
                validation_results[var_name] = f"Must be integer, got: {value}"
        
        return validation_results


class JINARateLimiter:
    """Rate limiter for JINA API calls with intelligent batching and request management.
    
    This class manages API rate limits for JINA embeddings service, implementing
    sliding window rate limiting with token-based and request-based limits.
    """
    
    def __init__(self, 
                 requests_per_minute: int = 500,
                 tokens_per_minute: int = 1000000,
                 max_batch_size: int = 100):
        """Initialize the JINA rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute (JINA API limit)
            tokens_per_minute: Maximum tokens per minute (JINA API limit)
            max_batch_size: Maximum batch size for API requests
        """
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.max_batch_size = max_batch_size
        
        # Thread-safe tracking of requests and tokens
        self._lock = Lock()
        self._request_times = deque()
        self._token_usage = deque()
        
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    def wait_if_needed(self, estimated_tokens: int = 0) -> float:
        """Wait if rate limit would be exceeded by the next request.
        
        Args:
            estimated_tokens: Estimated tokens for the upcoming request
            
        Returns:
            Time waited in seconds
        """
        with self._lock:
            current_time = time.time()
            wait_time = 0.0
            
            # Clean old entries (older than 1 minute)
            self._cleanup_old_entries(current_time)
            
            # Check request rate limit
            request_wait = self._calculate_request_wait(current_time)
            
            # Check token rate limit
            token_wait = self._calculate_token_wait(current_time, estimated_tokens)
            
            # Wait for the longer of the two limits
            wait_time = max(request_wait, token_wait)
            
            if wait_time > 0:
                self.logger.info(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
            
            # Record this request
            self._request_times.append(current_time + wait_time)
            if estimated_tokens > 0:
                self._token_usage.append((current_time + wait_time, estimated_tokens))
            
            return wait_time
    
    def _cleanup_old_entries(self, current_time: float):
        """Remove entries older than 1 minute."""
        cutoff_time = current_time - 60.0
        
        # Clean request times
        while self._request_times and self._request_times[0] < cutoff_time:
            self._request_times.popleft()
        
        # Clean token usage
        while self._token_usage and self._token_usage[0][0] < cutoff_time:
            self._token_usage.popleft()
    
    def _calculate_request_wait(self, current_time: float) -> float:
        """Calculate wait time based on request rate limit."""
        if len(self._request_times) < self.requests_per_minute:
            return 0.0
        
        # If we're at the limit, wait until the oldest request is 1 minute old
        oldest_request_time = self._request_times[0]
        wait_time = 60.0 - (current_time - oldest_request_time)
        return max(0.0, wait_time)
    
    def _calculate_token_wait(self, current_time: float, estimated_tokens: int) -> float:
        """Calculate wait time based on token rate limit."""
        if estimated_tokens == 0:
            return 0.0
        
        # Calculate current token usage in the last minute
        current_token_usage = sum(tokens for _, tokens in self._token_usage)
        
        if current_token_usage + estimated_tokens <= self.tokens_per_minute:
            return 0.0
        
        # If adding this request would exceed the limit, wait until enough tokens are available
        if self._token_usage:
            oldest_token_time = self._token_usage[0][0]
            wait_time = 60.0 - (current_time - oldest_token_time)
            return max(0.0, wait_time)
        
        return 0.0
    
    def estimate_tokens(self, texts: List[str]) -> int:
        """Estimate token count for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Estimated token count
        """
        # Rough estimation: ~4 characters per token for English text
        total_chars = sum(len(text) for text in texts)
        estimated_tokens = total_chars // 4
        
        # Add some overhead for API request structure
        return estimated_tokens + len(texts) * 10
    
    def get_optimal_batch_size(self, texts: List[str]) -> int:
        """Calculate optimal batch size based on current rate limits and text length.
        
        Args:
            texts: List of text strings to be processed
            
        Returns:
            Optimal batch size for API requests
        """
        if not texts:
            return 0
        
        # Start with maximum allowed batch size
        batch_size = min(self.max_batch_size, len(texts))
        
        # Estimate tokens for the full batch
        estimated_tokens = self.estimate_tokens(texts[:batch_size])
        
        # If estimated tokens exceed limit, reduce batch size
        while estimated_tokens > self.tokens_per_minute * 0.8 and batch_size > 1:
            batch_size = batch_size // 2
            estimated_tokens = self.estimate_tokens(texts[:batch_size])
        
        return max(1, batch_size)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current rate limit status for monitoring and debugging.
        
        Returns:
            Dictionary with current rate limit status
        """
        with self._lock:
            current_time = time.time()
            self._cleanup_old_entries(current_time)
            
            # Calculate current usage
            recent_requests = len(self._request_times)
            recent_tokens = sum(tokens for _, tokens in self._token_usage)
            
            return {
                "requests_last_minute": recent_requests,
                "requests_limit": self.requests_per_minute,
                "requests_remaining": max(0, self.requests_per_minute - recent_requests),
                "requests_utilization": (recent_requests / self.requests_per_minute) * 100,
                "tokens_last_minute": recent_tokens,
                "tokens_limit": self.tokens_per_minute,
                "tokens_remaining": max(0, self.tokens_per_minute - recent_tokens),
                "tokens_utilization": (recent_tokens / self.tokens_per_minute) * 100,
                "max_batch_size": self.max_batch_size,
                "current_time": current_time
            }
    
    def reset(self):
        """Reset rate limiter state (useful for testing)."""
        with self._lock:
            self._request_times.clear()
            self._token_usage.clear()


class RetryManager:
    """Retry manager with exponential backoff specifically designed for JINA API calls.
    
    This class implements intelligent retry logic with exponential backoff,
    jitter, and specific handling for different types of API failures.
    """
    
    def __init__(self, 
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 30.0,
                 backoff_multiplier: float = 2.0,
                 jitter: bool = True):
        """Initialize the retry manager.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for first retry
            max_delay: Maximum delay in seconds between retries
            backoff_multiplier: Multiplier for exponential backoff
            jitter: Whether to add random jitter to delays
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.jitter = jitter
        
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    def execute_with_retry(self, operation: callable, operation_name: str = "API call") -> Any:
        """Execute an operation with retry logic and exponential backoff.
        
        Args:
            operation: Callable to execute (should raise exception on failure)
            operation_name: Name of the operation for logging
            
        Returns:
            Result of the successful operation
            
        Raises:
            Exception: The last exception if all retries are exhausted
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):  # +1 for initial attempt
            try:
                if attempt > 0:
                    self.logger.info(f"Retrying {operation_name}, attempt {attempt + 1}/{self.max_retries + 1}")
                else:
                    self.logger.debug(f"Executing {operation_name}")
                
                result = operation()
                
                if attempt > 0:
                    self.logger.info(f"{operation_name} succeeded after {attempt} retries")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if we should retry this exception
                if not self._should_retry(e):
                    self.logger.error(f"{operation_name} failed with non-retryable error: {str(e)}")
                    raise e
                
                # Check if this was the last attempt
                if attempt >= self.max_retries:
                    self.logger.error(f"{operation_name} failed after {attempt + 1} attempts: {str(e)}")
                    raise e
                
                # Calculate delay and wait
                delay = self._calculate_delay(attempt)
                self.logger.warning(f"{operation_name} failed (attempt {attempt + 1}), retrying in {delay:.2f}s: {str(e)}")
                time.sleep(delay)
        
        # This should never be reached, but just in case
        if last_exception:
            raise last_exception
    
    def _should_retry(self, exception: Exception) -> bool:
        """Determine if an exception should trigger a retry.
        
        Args:
            exception: The exception to evaluate
            
        Returns:
            True if the operation should be retried, False otherwise
        """
        # Don't retry on certain types of errors
        non_retryable_errors = (
            ValueError,  # Bad input data
            TypeError,   # Programming errors
            KeyError,    # Missing required data
        )
        
        if isinstance(exception, non_retryable_errors):
            return False
        
        # Handle requests-specific exceptions
        if isinstance(exception, requests.exceptions.RequestException):
            # Don't retry on client errors (4xx), except for rate limiting
            if hasattr(exception, 'response') and exception.response is not None:
                status_code = exception.response.status_code
                
                # Retry on rate limiting (429) and server errors (5xx)
                if status_code == 429 or status_code >= 500:
                    return True
                
                # Don't retry on client errors (4xx except 429)
                if 400 <= status_code < 500:
                    return False
            
            # Retry on network-related errors
            if isinstance(exception, (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ReadTimeout
            )):
                return True
        
        # Check error message for retryable conditions
        error_message = str(exception).lower()
        retryable_keywords = [
            "timeout", "connection", "network", "unavailable",
            "service", "temporary", "rate limit", "too many requests",
            "server error", "internal error", "bad gateway",
            "service unavailable", "gateway timeout"
        ]
        
        return any(keyword in error_message for keyword in retryable_keywords)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the next retry attempt with exponential backoff.
        
        Args:
            attempt: Current attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        # Calculate exponential backoff delay
        delay = self.base_delay * (self.backoff_multiplier ** attempt)
        delay = min(delay, self.max_delay)
        
        # Add jitter to prevent thundering herd problem
        if self.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            jitter = random.uniform(-jitter_range, jitter_range)
            delay = max(0.1, delay + jitter)  # Ensure minimum delay
        
        return delay
    
    def create_retry_config(self, 
                          max_retries: Optional[int] = None,
                          base_delay: Optional[float] = None,
                          max_delay: Optional[float] = None) -> 'RetryManager':
        """Create a new RetryManager with custom configuration.
        
        Args:
            max_retries: Override max retries
            base_delay: Override base delay
            max_delay: Override max delay
            
        Returns:
            New RetryManager instance with custom configuration
        """
        return RetryManager(
            max_retries=max_retries or self.max_retries,
            base_delay=base_delay or self.base_delay,
            max_delay=max_delay or self.max_delay,
            backoff_multiplier=self.backoff_multiplier,
            jitter=self.jitter
        )
    
    def get_retry_info(self) -> Dict[str, Any]:
        """Get retry configuration information.
        
        Returns:
            Dictionary with retry configuration details
        """
        return {
            "max_retries": self.max_retries,
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
            "backoff_multiplier": self.backoff_multiplier,
            "jitter_enabled": self.jitter,
            "estimated_max_total_time": self._estimate_max_total_time()
        }
    
    def _estimate_max_total_time(self) -> float:
        """Estimate maximum total time for all retries.
        
        Returns:
            Estimated maximum total time in seconds
        """
        total_time = 0.0
        for attempt in range(self.max_retries):
            delay = self.base_delay * (self.backoff_multiplier ** attempt)
            delay = min(delay, self.max_delay)
            total_time += delay
        
        return total_time


class IntelligentBatchProcessor:
    """Intelligent batching system for optimizing JINA API usage.
    
    This class implements smart batching strategies that consider rate limits,
    text length, and API constraints to optimize embedding generation performance.
    """
    
    def __init__(self, rate_limiter: JINARateLimiter, max_batch_size: int = 100):
        """Initialize the batch processor.
        
        Args:
            rate_limiter: JINARateLimiter instance for rate management
            max_batch_size: Maximum batch size allowed by API
        """
        self.rate_limiter = rate_limiter
        self.max_batch_size = max_batch_size
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    def create_optimal_batches(self, texts: List[str]) -> List[List[str]]:
        """Create optimally sized batches from a list of texts.
        
        Args:
            texts: List of text strings to batch
            
        Returns:
            List of batches, where each batch is a list of texts
        """
        if not texts:
            return []
        
        batches = []
        current_batch = []
        current_batch_tokens = 0
        
        # Get rate limit status to inform batching decisions
        rate_status = self.rate_limiter.get_status()
        
        # Adjust batch size based on current rate limit utilization
        if rate_status["tokens_utilization"] > 80:
            # High utilization - use smaller batches
            target_batch_size = min(self.max_batch_size // 2, 50)
        elif rate_status["requests_utilization"] > 80:
            # High request utilization - use larger batches
            target_batch_size = self.max_batch_size
        else:
            # Normal utilization - use optimal batch size
            target_batch_size = self.rate_limiter.get_optimal_batch_size(texts)
        
        for text in texts:
            # Estimate tokens for this text
            text_tokens = self.rate_limiter.estimate_tokens([text])
            
            # Check if adding this text would exceed limits
            if (len(current_batch) >= target_batch_size or
                current_batch_tokens + text_tokens > self.rate_limiter.tokens_per_minute * 0.8):
                
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_batch_tokens = 0
            
            current_batch.append(text)
            current_batch_tokens += text_tokens
        
        # Add the last batch if it's not empty
        if current_batch:
            batches.append(current_batch)
        
        self.logger.debug(f"Created {len(batches)} batches from {len(texts)} texts")
        return batches
    
    def process_batches_with_rate_limiting(self, 
                                         batches: List[List[str]], 
                                         process_batch_func: callable) -> List[Any]:
        """Process batches with rate limiting and intelligent scheduling.
        
        Args:
            batches: List of text batches to process
            process_batch_func: Function to process each batch
            
        Returns:
            List of results from processing each batch
        """
        results = []
        
        for i, batch in enumerate(batches):
            self.logger.debug(f"Processing batch {i + 1}/{len(batches)} with {len(batch)} texts")
            
            # Estimate tokens for this batch
            estimated_tokens = self.rate_limiter.estimate_tokens(batch)
            
            # Wait if needed to respect rate limits
            wait_time = self.rate_limiter.wait_if_needed(estimated_tokens)
            
            if wait_time > 0:
                self.logger.info(f"Waited {wait_time:.2f}s for rate limiting before batch {i + 1}")
            
            # Process the batch
            try:
                batch_result = process_batch_func(batch)
                results.append(batch_result)
                
                self.logger.debug(f"Successfully processed batch {i + 1}/{len(batches)}")
                
            except Exception as e:
                self.logger.error(f"Failed to process batch {i + 1}/{len(batches)}: {str(e)}")
                raise
        
        return results
    
    def estimate_processing_time(self, texts: List[str]) -> Dict[str, float]:
        """Estimate total processing time for a list of texts.
        
        Args:
            texts: List of texts to process
            
        Returns:
            Dictionary with time estimates
        """
        if not texts:
            return {"total_time": 0.0, "batches": 0, "avg_batch_time": 0.0}
        
        batches = self.create_optimal_batches(texts)
        
        # Estimate time per batch (including API call time and rate limiting)
        avg_api_time = 2.0  # Estimated average API response time in seconds
        
        # Calculate rate limiting delays
        total_tokens = self.rate_limiter.estimate_tokens(texts)
        rate_status = self.rate_limiter.get_status()
        
        # Estimate additional delay due to rate limiting
        if rate_status["tokens_utilization"] > 80 or rate_status["requests_utilization"] > 80:
            rate_limiting_delay = len(batches) * 1.0  # Extra delay per batch
        else:
            rate_limiting_delay = 0.0
        
        total_time = len(batches) * avg_api_time + rate_limiting_delay
        
        return {
            "total_time": total_time,
            "batches": len(batches),
            "avg_batch_time": avg_api_time,
            "rate_limiting_delay": rate_limiting_delay,
            "estimated_tokens": total_tokens
        }

class EmbeddingProviderFactory:
    """Factory for creating and managing embedding providers with automatic detection and validation.
    
    This factory implements the provider pattern to centralize provider creation logic,
    handle configuration validation, and provide automatic fallback when preferred
    providers are unavailable.
    """
    
    def __init__(self):
        """Initialize the embedding provider factory."""
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._provider_cache = {}  # Cache for provider instances
        self._last_config_check = 0
        self._config_cache_duration = 60  # Cache config for 60 seconds
        
    def create_provider(self, config: Optional[EmbeddingConfig] = None, 
                       use_fallback: bool = True) -> EmbeddingProvider:
        """Create an embedding provider based on configuration with automatic fallback.
        
        Args:
            config: Optional EmbeddingConfig. If None, loads from environment
            use_fallback: Whether to use fallback provider if primary fails
            
        Returns:
            EmbeddingProvider instance (JINA or Local)
            
        Raises:
            ProviderConfigurationError: If no valid provider can be created
        """
        if config is None:
            config = self._load_and_validate_config()
        
        # Try to create the requested provider
        try:
            provider = self._create_provider_instance(config)
            
            # Validate that the provider is actually available
            if not self._validate_provider_availability(provider):
                if use_fallback and config.provider != "local":
                    self.logger.warning(
                        f"{config.provider} provider not available, attempting fallback to local"
                    )
                    return self._create_fallback_provider(config)
                else:
                    raise ProviderConfigurationError(
                        f"{config.provider} provider is not available. "
                        f"Please check your configuration and dependencies."
                    )
            
            self.logger.info(f"Successfully created {config.provider} embedding provider")
            return provider
            
        except Exception as e:
            if use_fallback and config.provider != "local":
                self.logger.warning(f"Failed to create {config.provider} provider: {e}")
                return self._create_fallback_provider(config)
            else:
                raise ProviderConfigurationError(
                    f"Failed to create {config.provider} provider: {e}"
                )
    
    def _load_and_validate_config(self) -> EmbeddingConfig:
        """Load configuration from environment with caching and validation."""
        current_time = time.time()
        
        # Use cached config if still valid
        if (hasattr(self, '_cached_config') and 
            current_time - self._last_config_check < self._config_cache_duration):
            return self._cached_config
        
        try:
            config = ConfigurationManager.load_config_from_env()
            self._cached_config = config
            self._last_config_check = current_time
            return config
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise ProviderConfigurationError(f"Configuration loading failed: {e}")
    
    def _create_provider_instance(self, config: EmbeddingConfig) -> EmbeddingProvider:
        """Create provider instance based on configuration."""
        if config.provider == "jina":
            return self._create_jina_provider(config)
        elif config.provider == "local":
            return self._create_local_provider(config)
        else:
            raise ProviderConfigurationError(
                f"Unknown embedding provider: {config.provider}. "
                f"Supported providers: local, jina"
            )
    
    def _create_jina_provider(self, config: EmbeddingConfig) -> EmbeddingProvider:
        """Create JINA embedding provider with fallback configuration."""
        from .jina_embedding_provider import JinaEmbeddingProvider
        
        # Create local provider for fallback if available
        fallback_provider = None
        try:
            fallback_config = ConfigurationManager._load_local_config()
            fallback_provider = self._create_local_provider(fallback_config)
        except Exception as e:
            self.logger.warning(f"Could not create fallback provider: {e}")
        
        return JinaEmbeddingProvider(config, fallback_provider=fallback_provider)
    
    def _create_local_provider(self, config: EmbeddingConfig) -> EmbeddingProvider:
        """Create local embedding provider."""
        from .local_embedding_provider import LocalEmbeddingProvider
        return LocalEmbeddingProvider(config)
    
    def _validate_provider_availability(self, provider: EmbeddingProvider) -> bool:
        """Validate that a provider is actually available and working."""
        try:
            return provider.is_available()
        except Exception as e:
            self.logger.error(f"Provider availability check failed: {e}")
            return False
    
    def _create_fallback_provider(self, original_config: EmbeddingConfig) -> EmbeddingProvider:
        """Create fallback provider when primary provider fails."""
        try:
            fallback_config = ConfigurationManager.create_fallback_config(original_config)
            fallback_provider = self._create_provider_instance(fallback_config)
            
            if not self._validate_provider_availability(fallback_provider):
                raise ProviderConfigurationError(
                    "Fallback provider is also not available"
                )
            
            self.logger.info(
                f"Successfully created fallback provider: {fallback_config.provider}"
            )
            return fallback_provider
            
        except Exception as e:
            raise ProviderConfigurationError(
                f"Failed to create fallback provider: {e}. "
                f"Both primary ({original_config.provider}) and fallback providers are unavailable."
            )
    
    def get_available_providers(self) -> List[Dict[str, Any]]:
        """Get list of available embedding providers with detailed information.
        
        Returns:
            List of dictionaries containing provider information
        """
        providers = []
        
        # Check local provider
        try:
            local_config = ConfigurationManager._load_local_config()
            local_provider = self._create_local_provider(local_config)
            providers.append({
                "name": "local",
                "available": self._validate_provider_availability(local_provider),
                "config": local_config,
                "info": local_provider.get_provider_info() if local_provider else None,
                "description": "Local HuggingFace embeddings"
            })
        except Exception as e:
            providers.append({
                "name": "local",
                "available": False,
                "error": str(e),
                "description": "Local HuggingFace embeddings"
            })
        
        # Check JINA provider
        try:
            jina_config = ConfigurationManager._load_jina_config()
            jina_provider = self._create_jina_provider(jina_config)
            providers.append({
                "name": "jina",
                "available": self._validate_provider_availability(jina_provider),
                "config": jina_config,
                "info": jina_provider.get_provider_info() if jina_provider else None,
                "description": "JINA AI cloud embeddings"
            })
        except Exception as e:
            providers.append({
                "name": "jina",
                "available": False,
                "error": str(e),
                "description": "JINA AI cloud embeddings"
            })
        
        return providers
    
    def detect_optimal_provider(self) -> str:
        """Detect the optimal provider based on availability and configuration.
        
        Returns:
            Name of the optimal provider ("jina" or "local")
        """
        available_providers = self.get_available_providers()
        
        # Prefer JINA if available and configured
        for provider in available_providers:
            if provider["name"] == "jina" and provider["available"]:
                self.logger.info("JINA provider detected as optimal choice")
                return "jina"
        
        # Fall back to local if available
        for provider in available_providers:
            if provider["name"] == "local" and provider["available"]:
                self.logger.info("Local provider detected as optimal choice")
                return "local"
        
        # If nothing is available, return local as default
        self.logger.warning("No providers available, defaulting to local")
        return "local"
    
    def create_optimal_provider(self) -> EmbeddingProvider:
        """Create the optimal provider based on current configuration and availability.
        
        Returns:
            EmbeddingProvider instance for the optimal provider
            
        Raises:
            ProviderConfigurationError: If no provider can be created
        """
        optimal_provider_name = self.detect_optimal_provider()
        
        # Load configuration for the optimal provider
        if optimal_provider_name == "jina":
            try:
                config = ConfigurationManager._load_jina_config()
            except Exception:
                # Fall back to local if JINA config fails
                config = ConfigurationManager._load_local_config()
        else:
            config = ConfigurationManager._load_local_config()
        
        return self.create_provider(config, use_fallback=True)
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate current embedding configuration and provide detailed feedback.
        
        Returns:
            Dictionary with validation results and recommendations
        """
        validation_results = {
            "overall_status": "unknown",
            "providers": {},
            "recommendations": [],
            "environment_variables": ConfigurationManager.validate_environment_variables()
        }
        
        # Test each provider
        available_providers = self.get_available_providers()
        
        for provider_info in available_providers:
            provider_name = provider_info["name"]
            validation_results["providers"][provider_name] = {
                "available": provider_info["available"],
                "configured": "config" in provider_info,
                "error": provider_info.get("error"),
                "info": provider_info.get("info")
            }
        
        # Determine overall status
        available_count = sum(1 for p in available_providers if p["available"])
        
        if available_count == 0:
            validation_results["overall_status"] = "error"
            validation_results["recommendations"].append(
                "No embedding providers are available. Please check your configuration."
            )
        elif available_count == 1:
            validation_results["overall_status"] = "warning"
            validation_results["recommendations"].append(
                "Only one embedding provider is available. Consider configuring a backup."
            )
        else:
            validation_results["overall_status"] = "ok"
            validation_results["recommendations"].append(
                "Multiple embedding providers are available."
            )
        
        # Add specific recommendations
        if not validation_results["providers"].get("local", {}).get("available", False):
            validation_results["recommendations"].append(
                "Local embeddings are not available. Install required dependencies: "
                "pip install langchain-community sentence-transformers"
            )
        
        if not validation_results["providers"].get("jina", {}).get("available", False):
            if not os.getenv('JINA_API_KEY'):
                validation_results["recommendations"].append(
                    "JINA embeddings are not configured. Set JINA_API_KEY environment variable. "
                    "Get your free API key at: https://jina.ai/?sui=apikey"
                )
            else:
                validation_results["recommendations"].append(
                    "JINA embeddings are configured but not available. Check your API key and network connection."
                )
        
        return validation_results
    
    def get_provider_comparison(self) -> Dict[str, Any]:
        """Get detailed comparison of available providers.
        
        Returns:
            Dictionary with provider comparison information
        """
        comparison = {
            "providers": {},
            "recommendations": {}
        }
        
        available_providers = self.get_available_providers()
        
        for provider_info in available_providers:
            provider_name = provider_info["name"]
            
            if provider_info["available"] and provider_info.get("info"):
                info = provider_info["info"]
                comparison["providers"][provider_name] = {
                    "dimensions": info.get("dimensions", "unknown"),
                    "model": info.get("model", "unknown"),
                    "performance": info.get("performance_metrics", {}),
                    "features": self._get_provider_features(provider_name),
                    "cost": self._get_provider_cost_info(provider_name),
                    "latency": self._estimate_provider_latency(provider_name)
                }
        
        # Add recommendations based on use cases
        comparison["recommendations"] = {
            "development": "local - No API costs, works offline",
            "production_small": "local - Predictable performance, no API limits",
            "production_large": "jina - Better scalability, latest models",
            "cost_sensitive": "local - No ongoing API costs",
            "performance_critical": "jina - Optimized cloud infrastructure",
            "offline_required": "local - Works without internet connection"
        }
        
        return comparison
    
    def _get_provider_features(self, provider_name: str) -> List[str]:
        """Get list of features for a provider."""
        features = {
            "local": [
                "Offline operation",
                "No API costs",
                "Predictable latency",
                "Privacy (no data sent to cloud)",
                "Customizable models"
            ],
            "jina": [
                "Latest embedding models",
                "Optimized cloud infrastructure",
                "Automatic scaling",
                "Multiple model options",
                "Regular model updates"
            ]
        }
        return features.get(provider_name, [])
    
    def _get_provider_cost_info(self, provider_name: str) -> Dict[str, str]:
        """Get cost information for a provider."""
        cost_info = {
            "local": {
                "setup_cost": "One-time model download",
                "ongoing_cost": "Compute resources only",
                "scaling_cost": "Linear with hardware"
            },
            "jina": {
                "setup_cost": "None",
                "ongoing_cost": "Per API request",
                "scaling_cost": "Pay-as-you-use"
            }
        }
        return cost_info.get(provider_name, {})
    
    def _estimate_provider_latency(self, provider_name: str) -> Dict[str, str]:
        """Estimate latency characteristics for a provider."""
        latency_info = {
            "local": {
                "first_request": "High (model loading)",
                "subsequent_requests": "Low (local processing)",
                "batch_processing": "Excellent"
            },
            "jina": {
                "first_request": "Medium (network + processing)",
                "subsequent_requests": "Medium (network dependent)",
                "batch_processing": "Good (API optimized)"
            }
        }
        return latency_info.get(provider_name, {})
    
    def clear_cache(self) -> None:
        """Clear provider and configuration cache."""
        self._provider_cache.clear()
        if hasattr(self, '_cached_config'):
            delattr(self, '_cached_config')
        self._last_config_check = 0
        self.logger.info("Provider factory cache cleared")


class ProviderDiscovery:
    """Provider discovery and validation system for embedding providers.
    
    This class provides comprehensive provider discovery, availability checking,
    configuration validation, and detailed reporting for debugging and monitoring.
    """
    
    def __init__(self):
        """Initialize the provider discovery system."""
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._discovery_cache = {}
        self._cache_duration = 300  # Cache results for 5 minutes
        
    def discover_all_providers(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Discover all available embedding providers with comprehensive information.
        
        Args:
            force_refresh: Whether to force refresh cached results
            
        Returns:
            Dictionary with detailed provider discovery results
        """
        cache_key = "all_providers"
        current_time = time.time()
        
        # Use cached results if available and not expired
        if (not force_refresh and 
            cache_key in self._discovery_cache and
            current_time - self._discovery_cache[cache_key]["timestamp"] < self._cache_duration):
            return self._discovery_cache[cache_key]["data"]
        
        self.logger.info("Discovering all embedding providers...")
        
        discovery_results = {
            "discovery_timestamp": current_time,
            "providers": {},
            "summary": {
                "total_providers": 0,
                "available_providers": 0,
                "configured_providers": 0,
                "recommended_provider": None
            },
            "system_info": self._get_system_info(),
            "environment_analysis": self._analyze_environment()
        }
        
        # Discover each provider type
        provider_types = ["local", "jina"]
        
        for provider_type in provider_types:
            try:
                provider_info = self._discover_provider(provider_type)
                discovery_results["providers"][provider_type] = provider_info
                discovery_results["summary"]["total_providers"] += 1
                
                if provider_info["available"]:
                    discovery_results["summary"]["available_providers"] += 1
                
                if provider_info["configured"]:
                    discovery_results["summary"]["configured_providers"] += 1
                    
            except Exception as e:
                self.logger.error(f"Failed to discover {provider_type} provider: {e}")
                discovery_results["providers"][provider_type] = {
                    "available": False,
                    "configured": False,
                    "error": str(e),
                    "discovery_failed": True
                }
        
        # Determine recommended provider
        discovery_results["summary"]["recommended_provider"] = self._determine_recommended_provider(
            discovery_results["providers"]
        )
        
        # Cache results
        self._discovery_cache[cache_key] = {
            "timestamp": current_time,
            "data": discovery_results
        }
        
        self.logger.info(
            f"Provider discovery complete: {discovery_results['summary']['available_providers']}/"
            f"{discovery_results['summary']['total_providers']} providers available"
        )
        
        return discovery_results
    
    def _discover_provider(self, provider_type: str) -> Dict[str, Any]:
        """Discover detailed information about a specific provider.
        
        Args:
            provider_type: Type of provider to discover ("local" or "jina")
            
        Returns:
            Dictionary with detailed provider information
        """
        provider_info = {
            "type": provider_type,
            "available": False,
            "configured": False,
            "config_valid": False,
            "dependencies_available": False,
            "test_results": {},
            "configuration": {},
            "performance_estimate": {},
            "issues": [],
            "recommendations": []
        }
        
        try:
            # Check configuration
            config_result = self._check_provider_configuration(provider_type)
            provider_info.update(config_result)
            
            # Check dependencies
            deps_result = self._check_provider_dependencies(provider_type)
            provider_info.update(deps_result)
            
            # If configured and dependencies are available, test the provider
            if provider_info["configured"] and provider_info["dependencies_available"]:
                test_result = self._test_provider_functionality(provider_type, provider_info["configuration"])
                provider_info["test_results"] = test_result
                provider_info["available"] = test_result.get("success", False)
                
                # Get performance estimates if available
                if provider_info["available"]:
                    perf_result = self._estimate_provider_performance(provider_type)
                    provider_info["performance_estimate"] = perf_result
            
            # Generate recommendations
            provider_info["recommendations"] = self._generate_provider_recommendations(
                provider_type, provider_info
            )
            
        except Exception as e:
            provider_info["issues"].append(f"Discovery failed: {e}")
            self.logger.error(f"Provider discovery failed for {provider_type}: {e}")
        
        return provider_info
    
    def _check_provider_configuration(self, provider_type: str) -> Dict[str, Any]:
        """Check provider configuration validity."""
        config_info = {
            "configured": False,
            "config_valid": False,
            "configuration": {},
            "config_issues": []
        }
        
        try:
            if provider_type == "local":
                config = ConfigurationManager._load_local_config()
            elif provider_type == "jina":
                config = ConfigurationManager._load_jina_config()
            else:
                raise ValueError(f"Unknown provider type: {provider_type}")
            
            # Validate configuration
            ConfigurationManager.validate_config(config)
            
            config_info["configured"] = True
            config_info["config_valid"] = True
            config_info["configuration"] = {
                "provider": config.provider,
                "model_name": config.model_name,
                "dimensions": config.dimensions,
                "batch_size": config.batch_size,
                "timeout": config.timeout,
                "api_key_configured": bool(config.api_key)
            }
            
        except ProviderConfigurationError as e:
            config_info["config_issues"].append(str(e))
            # Still might be partially configured
            try:
                if provider_type == "local":
                    config = ConfigurationManager._load_local_config()
                    config_info["configured"] = True
                elif provider_type == "jina":
                    # Check if API key exists even if invalid
                    if os.getenv('JINA_API_KEY'):
                        config_info["configured"] = True
            except Exception:
                pass
        except Exception as e:
            config_info["config_issues"].append(f"Configuration check failed: {e}")
        
        return config_info
    
    def _check_provider_dependencies(self, provider_type: str) -> Dict[str, Any]:
        """Check if provider dependencies are available."""
        deps_info = {
            "dependencies_available": False,
            "missing_dependencies": [],
            "dependency_versions": {}
        }
        
        if provider_type == "local":
            # Check HuggingFace dependencies
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                deps_info["dependency_versions"]["langchain_community"] = "available"
                
                try:
                    import sentence_transformers
                    deps_info["dependency_versions"]["sentence_transformers"] = sentence_transformers.__version__
                except ImportError:
                    deps_info["missing_dependencies"].append("sentence_transformers")
                
                try:
                    import torch
                    deps_info["dependency_versions"]["torch"] = torch.__version__
                    deps_info["dependency_versions"]["cuda_available"] = torch.cuda.is_available()
                except ImportError:
                    deps_info["missing_dependencies"].append("torch")
                
                # If core dependencies are available, mark as available
                if not deps_info["missing_dependencies"]:
                    deps_info["dependencies_available"] = True
                    
            except ImportError:
                deps_info["missing_dependencies"].extend([
                    "langchain_community", "sentence_transformers"
                ])
        
        elif provider_type == "jina":
            # Check JINA dependencies (mainly requests)
            try:
                import requests
                deps_info["dependency_versions"]["requests"] = requests.__version__
                
                try:
                    import aiohttp
                    deps_info["dependency_versions"]["aiohttp"] = aiohttp.__version__
                except ImportError:
                    # aiohttp is optional for JINA
                    pass
                
                deps_info["dependencies_available"] = True
                
            except ImportError:
                deps_info["missing_dependencies"].append("requests")
        
        return deps_info
    
    def _test_provider_functionality(self, provider_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Test provider functionality with actual embedding generation."""
        test_results = {
            "success": False,
            "test_duration": 0.0,
            "embedding_dimension": None,
            "test_texts_processed": 0,
            "error": None,
            "performance_metrics": {}
        }
        
        try:
            start_time = time.time()
            
            # Create provider instance
            if provider_type == "local":
                embedding_config = ConfigurationManager._load_local_config()
                from .local_embedding_provider import LocalEmbeddingProvider
                provider = LocalEmbeddingProvider(embedding_config)
            elif provider_type == "jina":
                embedding_config = ConfigurationManager._load_jina_config()
                from .jina_embedding_provider import JinaEmbeddingProvider
                provider = JinaEmbeddingProvider(embedding_config)
            else:
                raise ValueError(f"Unknown provider type: {provider_type}")
            
            # Test with sample texts
            test_texts = [
                "This is a test sentence for embedding generation.",
                "Another test sentence to verify batch processing.",
                "def test_function(): return 'Hello, World!'"
            ]
            
            # Generate embeddings
            embeddings = provider.generate_embeddings(test_texts)
            
            # Validate results
            if not embeddings or len(embeddings) != len(test_texts):
                raise EmbeddingGenerationError(
                    f"Invalid embedding result: expected {len(test_texts)}, got {len(embeddings)}"
                )
            
            # Check embedding dimensions
            if embeddings[0]:
                embedding_dim = len(embeddings[0])
                test_results["embedding_dimension"] = embedding_dim
                
                # Validate all embeddings have same dimension
                for i, emb in enumerate(embeddings):
                    if len(emb) != embedding_dim:
                        raise EmbeddingGenerationError(
                            f"Inconsistent embedding dimensions: {len(emb)} vs {embedding_dim}"
                        )
            
            test_duration = time.time() - start_time
            test_results.update({
                "success": True,
                "test_duration": round(test_duration, 3),
                "test_texts_processed": len(test_texts),
                "performance_metrics": {
                    "texts_per_second": round(len(test_texts) / test_duration, 2),
                    "avg_time_per_text": round(test_duration / len(test_texts), 4)
                }
            })
            
            # Get additional provider info
            provider_info = provider.get_provider_info()
            test_results["provider_info"] = provider_info
            
        except Exception as e:
            test_results["error"] = str(e)
            test_results["test_duration"] = time.time() - start_time
            self.logger.error(f"Provider functionality test failed for {provider_type}: {e}")
        
        return test_results
    
    def _estimate_provider_performance(self, provider_type: str) -> Dict[str, Any]:
        """Estimate provider performance characteristics."""
        performance_estimate = {
            "estimated_latency": {},
            "throughput_estimate": {},
            "resource_usage": {},
            "scalability": {}
        }
        
        if provider_type == "local":
            performance_estimate.update({
                "estimated_latency": {
                    "first_request": "2-10 seconds (model loading)",
                    "subsequent_requests": "0.1-1 seconds",
                    "batch_processing": "Linear with batch size"
                },
                "throughput_estimate": {
                    "small_texts": "10-100 texts/second",
                    "medium_texts": "5-50 texts/second",
                    "large_texts": "1-10 texts/second"
                },
                "resource_usage": {
                    "memory": "1-4 GB (model dependent)",
                    "cpu": "High during processing",
                    "gpu": "Optional, improves performance"
                },
                "scalability": {
                    "horizontal": "Limited (single instance)",
                    "vertical": "Good (more CPU/GPU helps)",
                    "cost_scaling": "Linear with hardware"
                }
            })
        
        elif provider_type == "jina":
            performance_estimate.update({
                "estimated_latency": {
                    "first_request": "0.5-2 seconds (network + processing)",
                    "subsequent_requests": "0.2-1 seconds",
                    "batch_processing": "Optimized for batches"
                },
                "throughput_estimate": {
                    "small_texts": "50-200 texts/second",
                    "medium_texts": "20-100 texts/second",
                    "large_texts": "5-50 texts/second"
                },
                "resource_usage": {
                    "memory": "Minimal (client only)",
                    "cpu": "Low (network I/O)",
                    "network": "Required, bandwidth dependent"
                },
                "scalability": {
                    "horizontal": "Excellent (cloud service)",
                    "vertical": "Automatic (managed service)",
                    "cost_scaling": "Pay-per-use"
                }
            })
        
        return performance_estimate
    
    def _generate_provider_recommendations(self, provider_type: str, provider_info: Dict[str, Any]) -> List[str]:
        """Generate recommendations for provider setup and usage."""
        recommendations = []
        
        if not provider_info["configured"]:
            if provider_type == "local":
                recommendations.extend([
                    "Configure LOCAL_EMBEDDING_MODEL environment variable",
                    "Install required dependencies: pip install langchain-community sentence-transformers",
                    "Consider using GPU for better performance if available"
                ])
            elif provider_type == "jina":
                recommendations.extend([
                    "Set JINA_API_KEY environment variable",
                    "Get your free API key at: https://jina.ai/?sui=apikey",
                    "Configure JINA_EMBEDDING_MODEL if needed (default: jina-embeddings-v3)"
                ])
        
        elif not provider_info["dependencies_available"]:
            missing_deps = provider_info.get("missing_dependencies", [])
            if missing_deps:
                recommendations.append(f"Install missing dependencies: pip install {' '.join(missing_deps)}")
        
        elif not provider_info["available"]:
            error = provider_info.get("test_results", {}).get("error", "Unknown error")
            recommendations.extend([
                f"Provider test failed: {error}",
                "Check logs for detailed error information",
                "Verify configuration and network connectivity"
            ])
        
        else:
            # Provider is working, provide optimization recommendations
            if provider_type == "local":
                recommendations.extend([
                    "Consider using GPU for better performance",
                    "Adjust batch size based on available memory",
                    "Monitor memory usage during processing"
                ])
            elif provider_type == "jina":
                recommendations.extend([
                    "Monitor API usage to stay within limits",
                    "Use batch processing for better efficiency",
                    "Consider caching embeddings for frequently used texts"
                ])
        
        return recommendations
    
    def _determine_recommended_provider(self, providers: Dict[str, Any]) -> Optional[str]:
        """Determine the recommended provider based on availability and configuration."""
        # Prefer JINA if available and configured
        if (providers.get("jina", {}).get("available", False) and
            providers.get("jina", {}).get("configured", False)):
            return "jina"
        
        # Fall back to local if available
        if (providers.get("local", {}).get("available", False) and
            providers.get("local", {}).get("configured", False)):
            return "local"
        
        # If nothing is fully available, recommend based on configuration
        if providers.get("jina", {}).get("configured", False):
            return "jina"
        elif providers.get("local", {}).get("configured", False):
            return "local"
        
        return None
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information relevant to embedding providers."""
        system_info = {
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "platform": os.sys.platform,
            "environment_variables": {}
        }
        
        # Check relevant environment variables
        env_vars = [
            "EMBEDDING_PROVIDER", "JINA_API_KEY", "JINA_EMBEDDING_MODEL",
            "LOCAL_EMBEDDING_MODEL", "EMBEDDING_BATCH_SIZE", "EMBEDDING_TIMEOUT"
        ]
        
        for var in env_vars:
            value = os.getenv(var)
            if value:
                # Mask sensitive information
                if "key" in var.lower() or "token" in var.lower():
                    system_info["environment_variables"][var] = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
                else:
                    system_info["environment_variables"][var] = value
            else:
                system_info["environment_variables"][var] = None
        
        # Check GPU availability
        try:
            import torch
            system_info["gpu_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                system_info["gpu_count"] = torch.cuda.device_count()
                system_info["gpu_name"] = torch.cuda.get_device_name(0)
        except ImportError:
            system_info["gpu_available"] = False
        
        return system_info
    
    def _analyze_environment(self) -> Dict[str, Any]:
        """Analyze the environment for embedding provider compatibility."""
        analysis = {
            "compatibility_score": 0,
            "issues": [],
            "recommendations": [],
            "optimal_configuration": {}
        }
        
        # Check Python version
        if os.sys.version_info >= (3, 8):
            analysis["compatibility_score"] += 25
        else:
            analysis["issues"].append("Python 3.8+ recommended for best compatibility")
        
        # Check for GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                analysis["compatibility_score"] += 25
                analysis["optimal_configuration"]["use_gpu"] = True
            else:
                analysis["optimal_configuration"]["use_gpu"] = False
        except ImportError:
            analysis["issues"].append("PyTorch not available - GPU acceleration not possible")
        
        # Check network connectivity for JINA
        try:
            import requests
            response = requests.get("https://api.jina.ai", timeout=5)
            if response.status_code < 500:  # Any response indicates connectivity
                analysis["compatibility_score"] += 25
                analysis["optimal_configuration"]["jina_available"] = True
            else:
                analysis["optimal_configuration"]["jina_available"] = False
        except Exception:
            analysis["issues"].append("Network connectivity to JINA API not available")
            analysis["optimal_configuration"]["jina_available"] = False
        
        # Check available memory (rough estimate)
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            if available_memory_gb >= 4:
                analysis["compatibility_score"] += 25
                analysis["optimal_configuration"]["sufficient_memory"] = True
            else:
                analysis["issues"].append("Low available memory - may affect local embedding performance")
                analysis["optimal_configuration"]["sufficient_memory"] = False
        except ImportError:
            analysis["recommendations"].append("Install psutil for memory monitoring: pip install psutil")
        
        # Generate recommendations based on analysis
        if analysis["compatibility_score"] >= 75:
            analysis["recommendations"].append("System is well-configured for embedding providers")
        elif analysis["compatibility_score"] >= 50:
            analysis["recommendations"].append("System is adequate but could be optimized")
        else:
            analysis["recommendations"].append("System may have compatibility issues - check requirements")
        
        return analysis
    
    def validate_provider_configuration(self, provider_type: str) -> Dict[str, Any]:
        """Validate configuration for a specific provider with detailed feedback.
        
        Args:
            provider_type: Type of provider to validate ("local" or "jina")
            
        Returns:
            Dictionary with detailed validation results
        """
        validation_result = {
            "provider_type": provider_type,
            "valid": False,
            "issues": [],
            "warnings": [],
            "recommendations": [],
            "configuration_details": {}
        }
        
        try:
            # Get provider discovery info
            provider_info = self._discover_provider(provider_type)
            
            validation_result["valid"] = provider_info["available"]
            validation_result["configuration_details"] = provider_info["configuration"]
            
            # Collect issues
            validation_result["issues"].extend(provider_info.get("config_issues", []))
            validation_result["issues"].extend(provider_info.get("missing_dependencies", []))
            
            if provider_info.get("test_results", {}).get("error"):
                validation_result["issues"].append(
                    f"Functionality test failed: {provider_info['test_results']['error']}"
                )
            
            # Add recommendations
            validation_result["recommendations"].extend(provider_info.get("recommendations", []))
            
            # Add warnings for suboptimal configurations
            if provider_type == "local" and not provider_info.get("dependencies_available"):
                validation_result["warnings"].append(
                    "Some optional dependencies are missing - performance may be suboptimal"
                )
            
            if provider_type == "jina" and provider_info.get("configured") and not provider_info.get("available"):
                validation_result["warnings"].append(
                    "JINA is configured but not accessible - check network connectivity and API key"
                )
            
        except Exception as e:
            validation_result["issues"].append(f"Validation failed: {e}")
        
        return validation_result
    
    def get_provider_health_report(self) -> Dict[str, Any]:
        """Generate a comprehensive health report for all providers.
        
        Returns:
            Dictionary with detailed health report
        """
        health_report = {
            "report_timestamp": time.time(),
            "overall_health": "unknown",
            "providers": {},
            "system_status": {},
            "recommendations": [],
            "action_items": []
        }
        
        # Discover all providers
        discovery_results = self.discover_all_providers(force_refresh=True)
        
        # Analyze each provider
        healthy_providers = 0
        total_providers = 0
        
        for provider_name, provider_info in discovery_results["providers"].items():
            total_providers += 1
            
            provider_health = {
                "status": "unhealthy",
                "score": 0,
                "issues": [],
                "performance": {}
            }
            
            # Calculate health score
            if provider_info.get("configured"):
                provider_health["score"] += 25
            if provider_info.get("dependencies_available"):
                provider_health["score"] += 25
            if provider_info.get("available"):
                provider_health["score"] += 50
                healthy_providers += 1
            
            # Determine status
            if provider_health["score"] >= 75:
                provider_health["status"] = "healthy"
            elif provider_health["score"] >= 50:
                provider_health["status"] = "degraded"
            else:
                provider_health["status"] = "unhealthy"
            
            # Add performance info if available
            if provider_info.get("test_results", {}).get("success"):
                test_results = provider_info["test_results"]
                provider_health["performance"] = {
                    "test_duration": test_results.get("test_duration", 0),
                    "embedding_dimension": test_results.get("embedding_dimension"),
                    "performance_metrics": test_results.get("performance_metrics", {})
                }
            
            # Collect issues
            provider_health["issues"].extend(provider_info.get("config_issues", []))
            if provider_info.get("test_results", {}).get("error"):
                provider_health["issues"].append(provider_info["test_results"]["error"])
            
            health_report["providers"][provider_name] = provider_health
        
        # Determine overall health
        if healthy_providers == 0:
            health_report["overall_health"] = "critical"
            health_report["action_items"].append("No embedding providers are available - immediate action required")
        elif healthy_providers == 1:
            health_report["overall_health"] = "warning"
            health_report["action_items"].append("Only one provider available - consider configuring backup")
        else:
            health_report["overall_health"] = "healthy"
        
        # Add system status
        health_report["system_status"] = discovery_results.get("system_info", {})
        
        # Generate recommendations
        if health_report["overall_health"] == "critical":
            health_report["recommendations"].extend([
                "Check embedding provider configuration",
                "Install required dependencies",
                "Verify network connectivity for cloud providers"
            ])
        elif health_report["overall_health"] == "warning":
            health_report["recommendations"].extend([
                "Configure additional embedding providers for redundancy",
                "Monitor provider performance and availability"
            ])
        
        return health_report
    
    def clear_discovery_cache(self) -> None:
        """Clear the discovery cache to force fresh discovery on next call."""
        self._discovery_cache.clear()
        self.logger.info("Provider discovery cache cleared")

class EmbeddingProviderFactory:
    """Factory for creating embedding providers based on configuration."""
    
    @staticmethod
    def create_provider(config: Optional[EmbeddingConfig] = None) -> EmbeddingProvider:
        """Create an embedding provider based on configuration.
        
        Args:
            config: EmbeddingConfig object. If None, loads from environment.
            
        Returns:
            Configured embedding provider instance
            
        Raises:
            ProviderConfigurationError: If provider creation fails
        """
        if config is None:
            config = ConfigurationManager.load_config_from_env()
        
        try:
            if config.provider == "jina":
                from .jina_embedding_provider import JINAEmbeddingProvider
                return JINAEmbeddingProvider(config)
            elif config.provider == "local":
                from .local_embedding_provider import LocalEmbeddingProvider
                return LocalEmbeddingProvider(config)
            else:
                raise ProviderConfigurationError(f"Unknown embedding provider: {config.provider}")
        
        except ImportError as e:
            raise ProviderConfigurationError(
                f"Failed to import provider '{config.provider}': {e}. "
                f"Make sure the provider module is available."
            )
        except Exception as e:
            # Try fallback to local provider if JINA fails
            if config.provider == "jina":
                logging.warning(f"JINA provider failed ({e}), falling back to local embeddings")
                try:
                    fallback_config = ConfigurationManager._load_local_config()
                    from .local_embedding_provider import LocalEmbeddingProvider
                    return LocalEmbeddingProvider(fallback_config)
                except Exception as fallback_error:
                    raise ProviderConfigurationError(
                        f"Primary provider '{config.provider}' failed: {e}. "
                        f"Fallback to local also failed: {fallback_error}"
                    )
            else:
                raise ProviderConfigurationError(f"Failed to create provider '{config.provider}': {e}")
    
    @staticmethod
    def get_available_providers() -> List[str]:
        """Get list of available embedding providers based on current configuration.
        
        Returns:
            List of available provider names
        """
        providers = []
        
        # Local provider is always available (assuming dependencies are installed)
        try:
            from .local_embedding_provider import LocalEmbeddingProvider
            providers.append("local")
        except ImportError:
            pass
        
        # Check if JINA is available
        if os.getenv('JINA_API_KEY'):
            try:
                from .jina_embedding_provider import JINAEmbeddingProvider
                providers.append("jina")
            except ImportError:
                pass
        
        return providers
    
    @staticmethod
    def create_provider_with_fallback(preferred_provider: str) -> EmbeddingProvider:
        """Create a provider with automatic fallback to local if preferred fails.
        
        Args:
            preferred_provider: Preferred provider name ("jina" or "local")
            
        Returns:
            Configured embedding provider instance
        """
        try:
            # Try to create preferred provider
            if preferred_provider == "jina":
                config = ConfigurationManager._load_jina_config()
            else:
                config = ConfigurationManager._load_local_config()
            
            return EmbeddingProviderFactory.create_provider(config)
            
        except Exception as e:
            logging.warning(f"Preferred provider '{preferred_provider}' failed: {e}")
            
            # Fallback to local if not already trying local
            if preferred_provider != "local":
                try:
                    fallback_config = ConfigurationManager._load_local_config()
                    return EmbeddingProviderFactory.create_provider(fallback_config)
                except Exception as fallback_error:
                    raise ProviderConfigurationError(
                        f"Preferred provider '{preferred_provider}' failed: {e}. "
                        f"Fallback to local also failed: {fallback_error}"
                    )
            else:
                raise e
    
    @staticmethod
    def validate_provider_availability(provider_name: str) -> Dict[str, Any]:
        """Validate if a specific provider is available and properly configured.
        
        Args:
            provider_name: Name of the provider to validate
            
        Returns:
            Dictionary with validation results
        """
        result = {
            "provider": provider_name,
            "available": False,
            "configured": False,
            "error": None,
            "requirements": []
        }
        
        try:
            if provider_name == "jina":
                # Check JINA requirements
                api_key = os.getenv('JINA_API_KEY')
                if not api_key:
                    result["error"] = "JINA_API_KEY not configured"
                    result["requirements"].append("Set JINA_API_KEY environment variable")
                    result["requirements"].append("Get API key from: https://jina.ai/?sui=apikey")
                else:
                    result["configured"] = True
                    
                    # Try to import and create provider
                    from .jina_embedding_provider import JINAEmbeddingProvider
                    config = ConfigurationManager._load_jina_config()
                    provider = JINAEmbeddingProvider(config)
                    
                    # Test availability
                    if provider.is_available():
                        result["available"] = True
                    else:
                        result["error"] = "JINA API not responding"
                        result["requirements"].append("Check network connectivity")
                        result["requirements"].append("Verify API key is valid")
            
            elif provider_name == "local":
                # Check local requirements
                try:
                    from .local_embedding_provider import LocalEmbeddingProvider
                    config = ConfigurationManager._load_local_config()
                    provider = LocalEmbeddingProvider(config)
                    
                    result["configured"] = True
                    
                    # Test availability
                    if provider.is_available():
                        result["available"] = True
                    else:
                        result["error"] = "Local embedding model not available"
                        result["requirements"].append("Install required model dependencies")
                        result["requirements"].append("Check model path configuration")
                
                except ImportError as e:
                    result["error"] = f"Local provider dependencies not available: {e}"
                    result["requirements"].append("Install required dependencies")
            
            else:
                result["error"] = f"Unknown provider: {provider_name}"
                result["requirements"].append("Use 'local' or 'jina' as provider name")
        
        except Exception as e:
            result["error"] = str(e)
            result["requirements"].append("Check provider configuration and dependencies")
        
        return result
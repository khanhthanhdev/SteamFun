"""
JINA AI Embedding Provider Implementation

This module implements the JINA AI embedding provider with proper authentication,
rate limiting, retry logic, batch processing, comprehensive error handling,
and automatic fallback mechanisms for efficient and robust API usage.
"""

import asyncio
import time
import json
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import aiohttp
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .embedding_providers import (
    EmbeddingProvider,
    EmbeddingConfig,
    EmbeddingGenerationError,
    ProviderConfigurationError
)


class JinaAPIError(EmbeddingGenerationError):
    """Specific exception for JINA API errors with detailed context."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, 
                 response_text: Optional[str] = None, error_type: str = "api_error",
                 retry_after: Optional[float] = None, fallback_available: bool = True):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text
        self.error_type = error_type
        self.retry_after = retry_after
        self.fallback_available = fallback_available
        
    def get_troubleshooting_info(self) -> Dict[str, Any]:
        """Get detailed troubleshooting information."""
        info = {
            "error_type": self.error_type,
            "message": str(self),
            "fallback_available": self.fallback_available,
            "suggestions": []
        }
        
        if self.status_code:
            info["status_code"] = self.status_code
            
            # Add specific suggestions based on status code
            if self.status_code == 401:
                info["suggestions"].extend([
                    "Check your JINA API key is correct",
                    "Verify the API key has not expired",
                    "Get a new API key at: https://jina.ai/?sui=apikey"
                ])
            elif self.status_code == 429:
                info["suggestions"].extend([
                    "You have exceeded the API rate limit",
                    "Wait before making more requests",
                    "Consider reducing batch size or request frequency"
                ])
                if self.retry_after:
                    info["retry_after"] = self.retry_after
            elif self.status_code >= 500:
                info["suggestions"].extend([
                    "JINA API is experiencing server issues",
                    "Try again in a few minutes",
                    "Check JINA status page for service updates"
                ])
            elif self.status_code == 400:
                info["suggestions"].extend([
                    "Check your request format and parameters",
                    "Verify text input is properly formatted",
                    "Ensure model name is correct"
                ])
        
        if self.response_text:
            info["response_details"] = self.response_text[:500]  # Limit length
            
        return info


class FallbackManager:
    """Manages fallback strategies when JINA API fails with comprehensive error handling."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._local_provider: Optional[EmbeddingProvider] = None
        self._fallback_enabled = True
        self._fallback_stats = {
            "total_fallbacks": 0,
            "successful_fallbacks": 0,
            "failed_fallbacks": 0,
            "last_fallback_time": None,
            "fallback_reasons": {}
        }
        self._consecutive_fallbacks = 0
        self._max_consecutive_fallbacks = 5
        
    def set_local_provider(self, provider: EmbeddingProvider) -> None:
        """Set the local embedding provider for fallback."""
        self._local_provider = provider
        
        # Validate that the local provider is actually available
        try:
            if provider and provider.is_available():
                self.logger.info("Local embedding provider configured and validated for fallback")
            else:
                self.logger.warning("Local embedding provider configured but not available")
        except Exception as e:
            self.logger.error(f"Error validating local provider: {e}")
        
    def is_fallback_available(self) -> bool:
        """Check if fallback to local embeddings is available."""
        if not self._fallback_enabled:
            return False
            
        if self._local_provider is None:
            return False
            
        # Check if we've exceeded consecutive fallback limit
        if self._consecutive_fallbacks >= self._max_consecutive_fallbacks:
            self.logger.warning(
                f"Fallback disabled due to {self._consecutive_fallbacks} consecutive failures. "
                f"This suggests a persistent issue with the local provider."
            )
            return False
            
        try:
            return self._local_provider.is_available()
        except Exception as e:
            self.logger.error(f"Error checking local provider availability: {e}")
            return False
        
    def execute_fallback(self, texts: List[str], operation_context: str = "", 
                        error_type: str = "unknown") -> List[List[float]]:
        """Execute fallback to local embeddings with comprehensive error handling."""
        if not self.is_fallback_available():
            error_msg = self._get_fallback_unavailable_message()
            raise EmbeddingGenerationError(error_msg)
        
        self._fallback_stats["total_fallbacks"] += 1
        self._fallback_stats["last_fallback_time"] = time.time()
        
        # Track fallback reasons
        if error_type not in self._fallback_stats["fallback_reasons"]:
            self._fallback_stats["fallback_reasons"][error_type] = 0
        self._fallback_stats["fallback_reasons"][error_type] += 1
        
        self.logger.warning(
            f"Executing fallback to local embeddings for {len(texts)} texts. "
            f"Reason: {error_type}. Context: {operation_context}"
        )
        
        try:
            # Validate input before fallback
            if not texts:
                self.logger.warning("Empty text list provided to fallback")
                return []
            
            # Filter out invalid texts
            valid_texts = []
            for i, text in enumerate(texts):
                if isinstance(text, str) and text.strip():
                    valid_texts.append(text)
                else:
                    self.logger.warning(f"Invalid text at index {i} in fallback: {type(text)}")
                    valid_texts.append("[INVALID_TEXT]")
            
            # Execute fallback with timeout protection
            start_time = time.time()
            result = self._local_provider.generate_embeddings(valid_texts)
            execution_time = time.time() - start_time
            
            # Validate fallback result
            if not result or len(result) != len(valid_texts):
                raise EmbeddingGenerationError(
                    f"Fallback returned invalid result: expected {len(valid_texts)} embeddings, "
                    f"got {len(result) if result else 0}"
                )
            
            # Validate embedding dimensions
            expected_dim = self._local_provider.get_embedding_dimension()
            for i, embedding in enumerate(result):
                if not isinstance(embedding, list) or len(embedding) != expected_dim:
                    raise EmbeddingGenerationError(
                        f"Invalid embedding at index {i}: expected {expected_dim} dimensions, "
                        f"got {len(embedding) if isinstance(embedding, list) else type(embedding)}"
                    )
            
            # Success - reset consecutive fallback counter
            self._consecutive_fallbacks = 0
            self._fallback_stats["successful_fallbacks"] += 1
            
            self.logger.info(
                f"Fallback successful: generated {len(result)} embeddings in {execution_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            self._consecutive_fallbacks += 1
            self._fallback_stats["failed_fallbacks"] += 1
            
            self.logger.error(
                f"Fallback to local embeddings failed (attempt {self._consecutive_fallbacks}): {e}"
            )
            
            # Provide detailed error information
            error_details = {
                "fallback_error": str(e),
                "fallback_error_type": type(e).__name__,
                "consecutive_failures": self._consecutive_fallbacks,
                "original_context": operation_context,
                "original_error_type": error_type,
                "texts_count": len(texts),
                "local_provider_available": self._local_provider.is_available() if self._local_provider else False
            }
            
            # Create comprehensive error message
            error_msg = (
                f"Both JINA API and local embedding fallback failed. "
                f"JINA error: {error_type} ({operation_context}). "
                f"Fallback error: {e}. "
                f"Consecutive fallback failures: {self._consecutive_fallbacks}. "
            )
            
            if self._consecutive_fallbacks >= self._max_consecutive_fallbacks:
                error_msg += (
                    f"Fallback mechanism will be disabled after {self._max_consecutive_fallbacks} "
                    f"consecutive failures to prevent infinite loops."
                )
            
            # Add troubleshooting suggestions
            suggestions = self._get_troubleshooting_suggestions(e, error_type)
            if suggestions:
                error_msg += f" Suggestions: {'; '.join(suggestions)}"
            
            raise EmbeddingGenerationError(error_msg)
    
    def _get_fallback_unavailable_message(self) -> str:
        """Get detailed message when fallback is unavailable."""
        if not self._fallback_enabled:
            return (
                "JINA API failed and fallback is disabled. "
                "Enable fallback by setting use_fallback=True or configure local embeddings."
            )
        
        if self._local_provider is None:
            return (
                "JINA API failed and no local embedding provider is configured. "
                "Please configure local embeddings as a fallback option. "
                "Set LOCAL_EMBEDDING_MODEL environment variable or provide a local provider."
            )
        
        if self._consecutive_fallbacks >= self._max_consecutive_fallbacks:
            return (
                f"JINA API failed and fallback is temporarily disabled due to "
                f"{self._consecutive_fallbacks} consecutive fallback failures. "
                f"This suggests a persistent issue with the local embedding provider. "
                f"Please check your local embedding configuration."
            )
        
        try:
            if not self._local_provider.is_available():
                return (
                    "JINA API failed and local embedding provider is not available. "
                    "Please check your local embedding model configuration and ensure "
                    "the model is properly installed and accessible."
                )
        except Exception as e:
            return (
                f"JINA API failed and cannot verify local provider availability: {e}. "
                f"Please check your local embedding configuration."
            )
        
        return "JINA API failed and fallback is unavailable for unknown reasons."
    
    def _get_troubleshooting_suggestions(self, fallback_error: Exception, 
                                       original_error_type: str) -> List[str]:
        """Get troubleshooting suggestions based on error types."""
        suggestions = []
        
        # Suggestions based on original JINA error
        if original_error_type == "authentication":
            suggestions.extend([
                "Verify your JINA API key is correct and active",
                "Check if your JINA account has sufficient credits",
                "Get a new API key at https://jina.ai/?sui=apikey"
            ])
        elif original_error_type == "rate_limit":
            suggestions.extend([
                "Reduce batch size or request frequency",
                "Implement request queuing",
                "Consider upgrading your JINA plan for higher limits"
            ])
        elif original_error_type in ["server_error", "connection_error"]:
            suggestions.extend([
                "Check JINA service status",
                "Verify network connectivity",
                "Try again in a few minutes"
            ])
        
        # Suggestions based on fallback error
        fallback_error_str = str(fallback_error).lower()
        if "model" in fallback_error_str or "huggingface" in fallback_error_str:
            suggestions.extend([
                "Check local embedding model installation",
                "Verify LOCAL_EMBEDDING_MODEL environment variable",
                "Ensure sufficient disk space for model files"
            ])
        elif "memory" in fallback_error_str or "cuda" in fallback_error_str:
            suggestions.extend([
                "Check available system memory",
                "Consider using CPU instead of GPU for local embeddings",
                "Reduce batch size for local processing"
            ])
        elif "timeout" in fallback_error_str:
            suggestions.extend([
                "Increase timeout for local embedding generation",
                "Check system performance and available resources",
                "Consider using a smaller/faster local model"
            ])
        
        # General suggestions
        if not suggestions:
            suggestions.extend([
                "Check system logs for more detailed error information",
                "Verify all embedding-related environment variables",
                "Consider restarting the application",
                "Contact support if the issue persists"
            ])
        
        return suggestions
    
    def disable_fallback(self) -> None:
        """Disable fallback mechanism (for testing or specific use cases)."""
        self._fallback_enabled = False
        self.logger.info("Fallback mechanism disabled")
        
    def enable_fallback(self) -> None:
        """Enable fallback mechanism and reset consecutive failure counter."""
        self._fallback_enabled = True
        self._consecutive_fallbacks = 0  # Reset on manual enable
        self.logger.info("Fallback mechanism enabled and failure counter reset")
    
    def reset_fallback_stats(self) -> None:
        """Reset fallback statistics (useful for monitoring)."""
        self._fallback_stats = {
            "total_fallbacks": 0,
            "successful_fallbacks": 0,
            "failed_fallbacks": 0,
            "last_fallback_time": None,
            "fallback_reasons": {}
        }
        self._consecutive_fallbacks = 0
        self.logger.info("Fallback statistics reset")
    
    def get_fallback_stats(self) -> Dict[str, Any]:
        """Get comprehensive fallback statistics."""
        stats = self._fallback_stats.copy()
        stats.update({
            "consecutive_fallbacks": self._consecutive_fallbacks,
            "max_consecutive_fallbacks": self._max_consecutive_fallbacks,
            "fallback_enabled": self._fallback_enabled,
            "local_provider_configured": self._local_provider is not None,
            "local_provider_available": (
                self._local_provider.is_available() 
                if self._local_provider else False
            ),
            "fallback_success_rate": (
                (self._fallback_stats["successful_fallbacks"] / 
                 max(1, self._fallback_stats["total_fallbacks"])) * 100
                if self._fallback_stats["total_fallbacks"] > 0 else 0
            )
        })
        return stats


@dataclass
class RateLimitInfo:
    """Rate limiting information for JINA API."""
    requests_per_minute: int = 200
    tokens_per_minute: int = 1000000
    current_requests: int = 0
    current_tokens: int = 0
    window_start: float = 0.0
    
    def reset_if_needed(self) -> None:
        """Reset counters if window has passed."""
        current_time = time.time()
        if current_time - self.window_start >= 60:  # 1 minute window
            self.current_requests = 0
            self.current_tokens = 0
            self.window_start = current_time
    
    def get_status(self) -> Dict[str, Any]:
        """Get current rate limit status for monitoring and debugging."""
        self.reset_if_needed()
        current_time = time.time()
        
        return {
            "requests_last_minute": self.current_requests,
            "requests_limit": self.requests_per_minute,
            "requests_remaining": max(0, self.requests_per_minute - self.current_requests),
            "requests_utilization": (self.current_requests / self.requests_per_minute) * 100,
            "tokens_last_minute": self.current_tokens,
            "tokens_limit": self.tokens_per_minute,
            "tokens_remaining": max(0, self.tokens_per_minute - self.current_tokens),
            "tokens_utilization": (self.current_tokens / self.tokens_per_minute) * 100,
            "window_start": self.window_start,
            "current_time": current_time
        }


class JinaEmbeddingProvider(EmbeddingProvider):
    """JINA AI embedding provider with comprehensive error handling, fallback mechanisms, 
    rate limiting, retry logic, and batch processing."""
    
    BASE_URL = "https://api.jina.ai/v1/embeddings"
    
    def __init__(self, config: EmbeddingConfig, fallback_provider: Optional[EmbeddingProvider] = None):
        """Initialize JINA embedding provider.
        
        Args:
            config: EmbeddingConfig with JINA-specific settings
            fallback_provider: Optional local embedding provider for fallback
        """
        super().__init__(config)
        
        if config.provider != "jina":
            raise ProviderConfigurationError(
                f"Invalid provider for JinaEmbeddingProvider: {config.provider}"
            )
        
        if not config.api_key:
            raise ProviderConfigurationError(
                "JINA API key is required. Get your free key at: https://jina.ai/?sui=apikey"
            )
        
        self.api_key = config.api_key
        self.model_name = config.model_name
        self.batch_size = min(config.batch_size, 100)  # JINA API limit
        self.timeout = config.timeout
        
        # Rate limiting
        self.rate_limit = RateLimitInfo()
        
        # Setup HTTP session with retry logic
        self.session = self._setup_session()
        
        # Error handling and fallback management
        self.fallback_manager = FallbackManager(self.logger)
        if fallback_provider:
            self.fallback_manager.set_local_provider(fallback_provider)
        
        # Error tracking for monitoring and debugging
        self.error_stats = {
            "total_errors": 0,
            "api_errors": 0,
            "network_errors": 0,
            "timeout_errors": 0,
            "fallback_uses": 0,
            "last_error_time": None,
            "consecutive_failures": 0
        }
        
        # Model-specific configurations
        self.model_configs = {
            "jina-embeddings-v3": {
                "dimensions": 1024,
                "max_tokens": 8192,
                "supports_task_type": True
            },
            "jina-embeddings-v4": {
                "dimensions": 2048,
                "max_tokens": 8192,
                "supports_task_type": True
            },
            "jina-clip-v2": {
                "dimensions": 1024,
                "max_tokens": 77,
                "supports_task_type": False
            }
        }
        
        self.model_config = self.model_configs.get(
            self.model_name,
            {"dimensions": 1024, "max_tokens": 8192, "supports_task_type": True}
        )
        
        self.logger.info(f"Initialized JINA provider with model: {self.model_name}")
        if fallback_provider:
            self.logger.info("Fallback to local embeddings is available")
    
    def _setup_session(self) -> requests.Session:
        """Setup HTTP session with retry logic and proper headers."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set default headers
        session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "RAG-System/1.0"
        })
        
        return session
    
    def _wait_for_rate_limit(self, estimated_tokens: int) -> None:
        """Wait if rate limit would be exceeded."""
        self.rate_limit.reset_if_needed()
        
        # Check if we would exceed limits
        if (self.rate_limit.current_requests >= self.rate_limit.requests_per_minute or
            self.rate_limit.current_tokens + estimated_tokens >= self.rate_limit.tokens_per_minute):
            
            # Calculate wait time until next window
            wait_time = 60 - (time.time() - self.rate_limit.window_start)
            if wait_time > 0:
                self.logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)
                self.rate_limit.reset_if_needed()
    
    def _estimate_tokens(self, texts: List[str]) -> int:
        """Estimate token count for rate limiting."""
        # Rough estimation: 1 token ≈ 4 characters
        total_chars = sum(len(text) for text in texts)
        return max(1, total_chars // 4)
    
    def _prepare_batch_request(self, texts: List[str], task_type: str = "retrieval.passage") -> Dict[str, Any]:
        """Prepare request payload for JINA API."""
        payload = {
            "model": self.model_name,
            "input": texts
        }
        
        # Add task_type if supported by model
        if self.model_config.get("supports_task_type", False):
            payload["task"] = task_type
        
        return payload
    
    def _process_batch_response(self, response_data: Dict[str, Any]) -> List[List[float]]:
        """Process JINA API response and extract embeddings."""
        try:
            if "data" not in response_data:
                raise EmbeddingGenerationError(
                    f"Invalid response format: missing 'data' field. Response: {response_data}"
                )
            
            embeddings = []
            for item in response_data["data"]:
                if "embedding" not in item:
                    raise EmbeddingGenerationError(
                        f"Invalid response format: missing 'embedding' field in data item"
                    )
                embeddings.append(item["embedding"])
            
            return embeddings
            
        except KeyError as e:
            raise EmbeddingGenerationError(
                f"Failed to parse JINA API response: missing key {e}"
            )
        except Exception as e:
            raise EmbeddingGenerationError(
                f"Failed to process JINA API response: {e}"
            )
    
    def _generate_batch_embeddings(self, texts: List[str], task_type: str = "retrieval.passage") -> List[List[float]]:
        """Generate embeddings for a batch of texts with comprehensive error handling."""
        if not texts:
            return []
        
        # Estimate tokens and wait for rate limit
        estimated_tokens = self._estimate_tokens(texts)
        self._wait_for_rate_limit(estimated_tokens)
        
        # Prepare request
        payload = self._prepare_batch_request(texts, task_type)
        operation_context = f"batch of {len(texts)} texts with model {self.model_name}"
        
        try:
            self.logger.debug(f"Sending batch request for {len(texts)} texts to JINA API")
            
            response = self.session.post(
                self.BASE_URL,
                json=payload,
                timeout=self.timeout
            )
            
            # Update rate limiting counters
            self.rate_limit.current_requests += 1
            self.rate_limit.current_tokens += estimated_tokens
            
            # Handle specific HTTP error codes with detailed error information
            if response.status_code == 401:
                self._track_error("api_error")
                raise JinaAPIError(
                    "JINA API authentication failed. Please check your API key.",
                    status_code=401,
                    response_text=response.text,
                    error_type="authentication",
                    fallback_available=self.fallback_manager.is_fallback_available()
                )
            elif response.status_code == 429:
                self._track_error("api_error")
                # Extract retry-after header if available
                retry_after = response.headers.get('Retry-After')
                retry_seconds = float(retry_after) if retry_after else 60.0
                
                raise JinaAPIError(
                    "JINA API rate limit exceeded. Please try again later.",
                    status_code=429,
                    response_text=response.text,
                    error_type="rate_limit",
                    retry_after=retry_seconds,
                    fallback_available=self.fallback_manager.is_fallback_available()
                )
            elif response.status_code == 400:
                self._track_error("api_error")
                raise JinaAPIError(
                    f"JINA API bad request. Check your input format and parameters.",
                    status_code=400,
                    response_text=response.text,
                    error_type="bad_request",
                    fallback_available=self.fallback_manager.is_fallback_available()
                )
            elif response.status_code >= 500:
                self._track_error("api_error")
                raise JinaAPIError(
                    f"JINA API server error {response.status_code}. The service may be temporarily unavailable.",
                    status_code=response.status_code,
                    response_text=response.text,
                    error_type="server_error",
                    fallback_available=self.fallback_manager.is_fallback_available()
                )
            elif response.status_code >= 400:
                self._track_error("api_error")
                raise JinaAPIError(
                    f"JINA API error {response.status_code}: {response.text}",
                    status_code=response.status_code,
                    response_text=response.text,
                    error_type="client_error",
                    fallback_available=self.fallback_manager.is_fallback_available()
                )
            
            response.raise_for_status()
            
            # Process response with error handling for malformed responses
            try:
                response_data = response.json()
            except json.JSONDecodeError as e:
                self._track_error("api_error")
                self.logger.error(f"Failed to parse JINA API response as JSON: {e}")
                raise JinaAPIError(
                    "JINA API returned malformed JSON response",
                    status_code=response.status_code,
                    response_text=response.text[:500],  # Limit response text length
                    error_type="malformed_response",
                    fallback_available=self.fallback_manager.is_fallback_available()
                )
            
            try:
                embeddings = self._process_batch_response(response_data)
            except EmbeddingGenerationError as e:
                self._track_error("api_error")
                self.logger.error(f"Failed to process JINA API response: {e}")
                raise JinaAPIError(
                    f"JINA API response processing failed: {e}",
                    status_code=response.status_code,
                    response_text=str(response_data)[:500],
                    error_type="response_processing",
                    fallback_available=self.fallback_manager.is_fallback_available()
                )
            
            # Validate response consistency
            if len(embeddings) != len(texts):
                self._track_error("api_error")
                raise JinaAPIError(
                    f"Mismatch between input texts ({len(texts)}) and output embeddings ({len(embeddings)})",
                    status_code=response.status_code,
                    response_text=str(response_data)[:500],
                    error_type="response_mismatch",
                    fallback_available=self.fallback_manager.is_fallback_available()
                )
            
            # Reset consecutive failures on success
            self.error_stats["consecutive_failures"] = 0
            self.logger.debug(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
            
        except JinaAPIError:
            # Re-raise JINA-specific errors as-is
            raise
        except requests.exceptions.Timeout as e:
            self._track_error("timeout_error")
            raise JinaAPIError(
                f"JINA API request timed out after {self.timeout} seconds",
                error_type="timeout",
                fallback_available=self.fallback_manager.is_fallback_available()
            )
        except requests.exceptions.ConnectionError as e:
            self._track_error("network_error")
            raise JinaAPIError(
                "Failed to connect to JINA API. Please check your internet connection.",
                error_type="connection_error",
                fallback_available=self.fallback_manager.is_fallback_available()
            )
        except requests.exceptions.RequestException as e:
            self._track_error("network_error")
            raise JinaAPIError(
                f"JINA API request failed: {e}",
                error_type="request_error",
                fallback_available=self.fallback_manager.is_fallback_available()
            )
        except Exception as e:
            self._track_error("api_error")
            self.logger.error(f"Unexpected error in JINA API call: {e}")
            raise JinaAPIError(
                f"Unexpected error during JINA API call: {e}",
                error_type="unexpected_error",
                fallback_available=self.fallback_manager.is_fallback_available()
            )
    
    def _track_error(self, error_type: str) -> None:
        """Track error statistics for monitoring and debugging."""
        current_time = time.time()
        
        self.error_stats["total_errors"] += 1
        self.error_stats[error_type] = self.error_stats.get(error_type, 0) + 1
        self.error_stats["last_error_time"] = current_time
        self.error_stats["consecutive_failures"] += 1
        
        # Track error frequency for pattern detection
        if "error_timeline" not in self.error_stats:
            self.error_stats["error_timeline"] = []
        
        self.error_stats["error_timeline"].append({
            "timestamp": current_time,
            "error_type": error_type,
            "consecutive_count": self.error_stats["consecutive_failures"]
        })
        
        # Keep only recent errors (last 100)
        if len(self.error_stats["error_timeline"]) > 100:
            self.error_stats["error_timeline"] = self.error_stats["error_timeline"][-100:]
        
        # Log error patterns for debugging
        if self.error_stats["consecutive_failures"] >= 3:
            self.logger.warning(
                f"JINA API has failed {self.error_stats['consecutive_failures']} consecutive times. "
                f"Error type: {error_type}. Consider checking API status or using fallback."
            )
        
        # Detect error spikes (more than 5 errors in last 5 minutes)
        recent_errors = [
            e for e in self.error_stats["error_timeline"] 
            if current_time - e["timestamp"] < 300  # 5 minutes
        ]
        
        if len(recent_errors) >= 5:
            self.logger.error(
                f"Error spike detected: {len(recent_errors)} errors in last 5 minutes. "
                f"Most common: {max(set(e['error_type'] for e in recent_errors), key=lambda x: sum(1 for e in recent_errors if e['error_type'] == x))}"
            )
    
    def create_error_report(self, include_sensitive_data: bool = False) -> Dict[str, Any]:
        """Create comprehensive error report for debugging and support."""
        current_time = time.time()
        
        # Basic error statistics
        report = {
            "report_timestamp": current_time,
            "provider_info": {
                "provider": "jina",
                "model": self.model_name,
                "dimensions": self.get_embedding_dimension(),
                "batch_size": self.batch_size,
                "timeout": self.timeout
            },
            "error_statistics": self.error_stats.copy(),
            "health_status": self._get_health_status(),
            "fallback_status": self.fallback_manager.get_fallback_stats(),
            "configuration_status": self._get_configuration_status(include_sensitive_data)
        }
        
        # Recent error analysis
        if "error_timeline" in self.error_stats:
            recent_errors = [
                e for e in self.error_stats["error_timeline"]
                if current_time - e["timestamp"] < 3600  # Last hour
            ]
            
            if recent_errors:
                error_types = [e["error_type"] for e in recent_errors]
                report["recent_error_analysis"] = {
                    "errors_last_hour": len(recent_errors),
                    "error_type_distribution": {
                        error_type: error_types.count(error_type)
                        for error_type in set(error_types)
                    },
                    "error_frequency": len(recent_errors) / 60,  # errors per minute
                    "latest_errors": recent_errors[-5:]  # Last 5 errors
                }
        
        # Rate limit analysis
        if hasattr(self, 'rate_limit'):
            report["rate_limit_analysis"] = self.rate_limit.get_status()
        
        # System diagnostics
        report["system_diagnostics"] = self.diagnose_connection_issues()
        
        # Recommendations based on error patterns
        report["recommendations"] = self._generate_error_based_recommendations()
        
        return report
    
    def _get_configuration_status(self, include_sensitive_data: bool = False) -> Dict[str, Any]:
        """Get configuration status for error reporting."""
        config_status = {
            "api_key_configured": bool(self.api_key),
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "timeout": self.timeout,
            "fallback_enabled": self.fallback_manager._fallback_enabled,
            "local_provider_configured": self.fallback_manager._local_provider is not None
        }
        
        if include_sensitive_data and self.api_key:
            # Only include masked version for debugging
            config_status["api_key_preview"] = f"{self.api_key[:8]}...{self.api_key[-4:]}"
        
        # Validate configuration
        config_issues = []
        if not self.api_key:
            config_issues.append("JINA API key not configured")
        
        if self.batch_size > 100:
            config_issues.append("Batch size exceeds JINA API limit (100)")
        
        if self.timeout < 10:
            config_issues.append("Timeout may be too low for reliable API calls")
        
        if not self.fallback_manager.is_fallback_available():
            config_issues.append("No fallback provider available")
        
        config_status["configuration_issues"] = config_issues
        
        return config_status
    
    def _generate_error_based_recommendations(self) -> List[str]:
        """Generate recommendations based on current error patterns."""
        recommendations = []
        
        # Analyze error statistics
        total_errors = self.error_stats.get("total_errors", 0)
        consecutive_failures = self.error_stats.get("consecutive_failures", 0)
        
        if total_errors == 0:
            recommendations.append("No errors detected - system appears healthy")
            return recommendations
        
        # High error rate recommendations
        if total_errors > 10:
            recommendations.append("High error count detected - investigate root cause")
        
        # Consecutive failure recommendations
        if consecutive_failures > 5:
            recommendations.extend([
                "Multiple consecutive failures indicate persistent issue",
                "Check JINA service status and API key validity",
                "Consider using fallback provider temporarily"
            ])
        
        # Specific error type recommendations
        error_types = {k: v for k, v in self.error_stats.items() if k.endswith("_error")}
        
        if error_types.get("api_errors", 0) > error_types.get("network_errors", 0):
            recommendations.append("API errors dominate - check authentication and rate limits")
        elif error_types.get("network_errors", 0) > 0:
            recommendations.append("Network errors detected - check connectivity and DNS")
        
        if error_types.get("timeout_errors", 0) > 3:
            recommendations.append("Multiple timeout errors - consider increasing timeout or reducing batch size")
        
        # Fallback usage recommendations
        fallback_uses = self.error_stats.get("fallback_uses", 0)
        if fallback_uses > total_errors * 0.5:
            recommendations.append("High fallback usage - primary provider may be unreliable")
        
        if not recommendations:
            recommendations.append("Review error logs for specific failure patterns")
        
        return recommendations
    
    def enable_graceful_degradation(self) -> None:
        """Enable graceful degradation mode for handling persistent failures."""
        self._graceful_degradation_enabled = True
        self.logger.info("Graceful degradation mode enabled")
    
    def disable_graceful_degradation(self) -> None:
        """Disable graceful degradation mode."""
        self._graceful_degradation_enabled = False
        self.logger.info("Graceful degradation mode disabled")
    
    def is_graceful_degradation_active(self) -> bool:
        """Check if graceful degradation should be active based on error patterns."""
        if not hasattr(self, '_graceful_degradation_enabled'):
            self._graceful_degradation_enabled = True  # Default to enabled
        
        if not self._graceful_degradation_enabled:
            return False
        
        # Activate degradation if we have persistent issues
        consecutive_failures = self.error_stats.get("consecutive_failures", 0)
        total_errors = self.error_stats.get("total_errors", 0)
        
        # Activate if we have many consecutive failures or high error rate
        return consecutive_failures >= 3 or (total_errors > 0 and 
                                           self.error_stats.get("fallback_uses", 0) / max(1, total_errors) > 0.3)
    
    def get_degraded_service_info(self) -> Dict[str, Any]:
        """Get information about current degraded service status."""
        if not self.is_graceful_degradation_active():
            return {"degraded": False, "reason": "Service operating normally"}
        
        degradation_info = {
            "degraded": True,
            "reason": "Persistent API failures detected",
            "active_mitigations": [],
            "service_level": "degraded",
            "estimated_reliability": 0.0
        }
        
        # Determine active mitigations
        if self.fallback_manager.is_fallback_available():
            degradation_info["active_mitigations"].append("Automatic fallback to local embeddings")
            degradation_info["service_level"] = "limited"
            degradation_info["estimated_reliability"] = 0.7
        
        consecutive_failures = self.error_stats.get("consecutive_failures", 0)
        if consecutive_failures >= 5:
            degradation_info["active_mitigations"].append("Reduced batch size for stability")
            degradation_info["service_level"] = "minimal"
            degradation_info["estimated_reliability"] = max(0.3, degradation_info["estimated_reliability"] - 0.2)
        
        # Calculate reliability based on recent success rate
        if "error_timeline" in self.error_stats:
            recent_errors = [
                e for e in self.error_stats["error_timeline"]
                if time.time() - e["timestamp"] < 1800  # Last 30 minutes
            ]
            if recent_errors:
                error_rate = len(recent_errors) / 30  # errors per minute
                degradation_info["estimated_reliability"] = max(0.1, 1.0 - min(error_rate, 0.9))
        
        return degradation_info
    
    def attempt_service_recovery(self) -> Dict[str, Any]:
        """Attempt to recover from degraded service state."""
        recovery_report = {
            "timestamp": time.time(),
            "recovery_attempts": [],
            "success": False,
            "recommendations": []
        }
        
        self.logger.info("Attempting service recovery from degraded state")
        
        # Step 1: Reset error counters to give fresh start
        old_consecutive = self.error_stats.get("consecutive_failures", 0)
        self.error_stats["consecutive_failures"] = 0
        recovery_report["recovery_attempts"].append(
            f"Reset consecutive failure counter (was {old_consecutive})"
        )
        
        # Step 2: Test basic connectivity
        try:
            test_result = self.generate_embeddings(["test recovery"], use_fallback=False)
            if test_result and len(test_result) == 1:
                recovery_report["recovery_attempts"].append("Basic API test: SUCCESS")
                recovery_report["success"] = True
                self.logger.info("Service recovery successful - API responding normally")
            else:
                recovery_report["recovery_attempts"].append("Basic API test: FAILED - Invalid response")
        except Exception as e:
            recovery_report["recovery_attempts"].append(f"Basic API test: FAILED - {str(e)}")
            # Restore consecutive failure count if test failed
            self.error_stats["consecutive_failures"] = old_consecutive
        
        # Step 3: Test fallback if primary failed
        if not recovery_report["success"] and self.fallback_manager.is_fallback_available():
            try:
                fallback_result = self.fallback_manager.execute_fallback(
                    ["test fallback recovery"], 
                    "Recovery test",
                    "recovery_test"
                )
                if fallback_result:
                    recovery_report["recovery_attempts"].append("Fallback test: SUCCESS")
                    recovery_report["recommendations"].append("Primary API failed but fallback is working")
                else:
                    recovery_report["recovery_attempts"].append("Fallback test: FAILED - No result")
            except Exception as e:
                recovery_report["recovery_attempts"].append(f"Fallback test: FAILED - {str(e)}")
        
        # Step 4: Generate recommendations based on results
        if recovery_report["success"]:
            recovery_report["recommendations"].extend([
                "Service has recovered successfully",
                "Monitor for stability over next few requests",
                "Consider investigating root cause of original failures"
            ])
        else:
            recovery_report["recommendations"].extend([
                "Service recovery failed - persistent issues detected",
                "Check JINA service status and account standing",
                "Verify network connectivity and configuration",
                "Consider using fallback provider exclusively until issues resolve"
            ])
        
        return recovery_report
    
    def get_service_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive service health summary for monitoring."""
        current_time = time.time()
        
        summary = {
            "timestamp": current_time,
            "overall_status": "unknown",
            "provider": "jina",
            "model": self.model_name,
            "availability": {
                "primary_service": False,
                "fallback_service": False,
                "degraded_mode": False
            },
            "performance_metrics": {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "fallback_usage": 0,
                "success_rate": 0.0,
                "average_response_time": 0.0
            },
            "error_summary": {
                "consecutive_failures": self.error_stats.get("consecutive_failures", 0),
                "total_errors": self.error_stats.get("total_errors", 0),
                "last_error_time": self.error_stats.get("last_error_time"),
                "dominant_error_type": "none"
            },
            "recommendations": []
        }
        
        # Test primary service availability
        try:
            test_available = self.is_available()
            summary["availability"]["primary_service"] = test_available
        except Exception:
            summary["availability"]["primary_service"] = False
        
        # Test fallback availability
        summary["availability"]["fallback_service"] = self.fallback_manager.is_fallback_available()
        
        # Check degraded mode
        summary["availability"]["degraded_mode"] = self.is_graceful_degradation_active()
        
        # Calculate performance metrics
        total_errors = self.error_stats.get("total_errors", 0)
        fallback_uses = self.error_stats.get("fallback_uses", 0)
        
        # Estimate total requests (errors + successful requests)
        # This is approximate since we don't track successful requests separately
        estimated_successful = max(0, fallback_uses + total_errors)  # Minimum estimate
        summary["performance_metrics"]["total_requests"] = estimated_successful + total_errors
        summary["performance_metrics"]["successful_requests"] = estimated_successful
        summary["performance_metrics"]["failed_requests"] = total_errors
        summary["performance_metrics"]["fallback_usage"] = fallback_uses
        
        if summary["performance_metrics"]["total_requests"] > 0:
            summary["performance_metrics"]["success_rate"] = (
                summary["performance_metrics"]["successful_requests"] / 
                summary["performance_metrics"]["total_requests"] * 100
            )
        
        # Determine dominant error type
        error_types = {k: v for k, v in self.error_stats.items() if k.endswith("_error")}
        if error_types:
            summary["error_summary"]["dominant_error_type"] = max(error_types, key=error_types.get)
        
        # Determine overall status
        if summary["availability"]["primary_service"]:
            if summary["error_summary"]["consecutive_failures"] == 0:
                summary["overall_status"] = "healthy"
            else:
                summary["overall_status"] = "degraded"
        elif summary["availability"]["fallback_service"]:
            summary["overall_status"] = "limited"
        else:
            summary["overall_status"] = "unavailable"
        
        # Generate status-specific recommendations
        if summary["overall_status"] == "healthy":
            summary["recommendations"] = ["Service operating normally", "Continue monitoring"]
        elif summary["overall_status"] == "degraded":
            summary["recommendations"] = [
                "Primary service experiencing issues",
                "Monitor error patterns",
                "Ensure fallback is configured"
            ]
        elif summary["overall_status"] == "limited":
            summary["recommendations"] = [
                "Primary service unavailable, using fallback",
                "Check JINA service status",
                "Investigate primary service issues"
            ]
        else:
            summary["recommendations"] = [
                "Both primary and fallback services unavailable",
                "Check configuration and connectivity",
                "Contact support if issues persist"
            ]
        
        return summary
    
    def generate_embeddings(self, texts: List[str], task_type: str = "retrieval.passage", 
                          use_fallback: bool = True) -> List[List[float]]:
        """Generate embeddings for a list of texts with comprehensive error handling and fallback.
        
        Args:
            texts: List of text strings to generate embeddings for
            task_type: Task type for JINA API (e.g., "retrieval.passage", "retrieval.query")
            use_fallback: Whether to use fallback to local embeddings on failure
            
        Returns:
            List of embedding vectors, where each vector is a list of floats
            
        Raises:
            EmbeddingGenerationError: If embedding generation fails and no fallback is available
        """
        if not texts:
            return []
        
        # Validate and sanitize inputs
        validated_texts, validation_warnings = self._validate_and_sanitize_texts(texts)
        
        if validation_warnings:
            self.logger.warning(f"Input validation warnings: {validation_warnings}")
        
        # Check provider health before attempting API call
        health_status = self._get_health_status()
        if health_status["status"] == "unhealthy" and use_fallback:
            self.logger.warning(
                f"JINA provider is unhealthy ({health_status['consecutive_failures']} "
                f"consecutive failures), attempting fallback immediately"
            )
            if self.fallback_manager.is_fallback_available():
                try:
                    self.error_stats["fallback_uses"] += 1
                    return self.fallback_manager.execute_fallback(
                        validated_texts, 
                        "Provider unhealthy - preemptive fallback",
                        "provider_unhealthy"
                    )
                except Exception as fallback_error:
                    self.logger.error(f"Preemptive fallback failed: {fallback_error}")
                    # Continue with JINA API attempt
        
        # Try JINA API first
        try:
            result = self._generate_embeddings_with_jina(validated_texts, task_type)
            
            # Validate result before returning
            self._validate_embedding_result(result, validated_texts)
            
            return result
            
        except JinaAPIError as e:
            return self._handle_jina_api_error(e, validated_texts, use_fallback)
        except Exception as e:
            return self._handle_unexpected_error(e, validated_texts, use_fallback)
    
    def _validate_and_sanitize_texts(self, texts: List[str]) -> tuple[List[str], List[str]]:
        """Validate and sanitize input texts with detailed warnings."""
        validated_texts = []
        warnings = []
        
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                warnings.append(f"Text at index {i} is not a string: {type(text)}")
                validated_texts.append(str(text) if text is not None else "[NULL]")
            elif not text.strip():
                warnings.append(f"Empty text at index {i}, using placeholder")
                validated_texts.append("[EMPTY]")
            elif len(text) > self.model_config.get("max_tokens", 8192) * 4:  # Rough char estimate
                warnings.append(f"Text at index {i} may exceed token limit ({len(text)} chars)")
                # Truncate but keep some content
                truncated = text[:self.model_config.get("max_tokens", 8192) * 3] + "..."
                validated_texts.append(truncated)
            else:
                validated_texts.append(text)
        
        return validated_texts, warnings
    
    def _validate_embedding_result(self, result: List[List[float]], input_texts: List[str]) -> None:
        """Validate the embedding result for consistency and correctness."""
        if not result:
            raise EmbeddingGenerationError("JINA API returned empty result")
        
        if len(result) != len(input_texts):
            raise EmbeddingGenerationError(
                f"Result count mismatch: expected {len(input_texts)} embeddings, "
                f"got {len(result)}"
            )
        
        expected_dim = self.get_embedding_dimension()
        for i, embedding in enumerate(result):
            if not isinstance(embedding, list):
                raise EmbeddingGenerationError(
                    f"Invalid embedding type at index {i}: expected list, got {type(embedding)}"
                )
            
            if len(embedding) != expected_dim:
                raise EmbeddingGenerationError(
                    f"Invalid embedding dimension at index {i}: expected {expected_dim}, "
                    f"got {len(embedding)}"
                )
            
            # Check for NaN or infinite values
            if not all(isinstance(x, (int, float)) and not (x != x or abs(x) == float('inf')) 
                      for x in embedding):
                raise EmbeddingGenerationError(
                    f"Invalid embedding values at index {i}: contains NaN or infinite values"
                )
    
    def _handle_jina_api_error(self, error: JinaAPIError, texts: List[str], 
                              use_fallback: bool) -> List[List[float]]:
        """Handle JINA API errors with comprehensive error reporting and fallback."""
        # Log detailed error information
        error_info = error.get_troubleshooting_info()
        self.logger.error(f"JINA API failed: {error_info}")
        
        # Update error statistics
        self._track_error("api_error")
        
        # Attempt fallback if enabled and available
        if use_fallback and self.fallback_manager.is_fallback_available():
            self.logger.warning("Attempting fallback to local embeddings")
            try:
                self.error_stats["fallback_uses"] += 1
                return self.fallback_manager.execute_fallback(
                    texts, 
                    f"JINA API error: {error.error_type}",
                    error.error_type
                )
            except Exception as fallback_error:
                self.logger.error(f"Fallback also failed: {fallback_error}")
                
                # Create comprehensive error message
                comprehensive_error = self._create_comprehensive_error_message(
                    error, fallback_error, error_info
                )
                raise EmbeddingGenerationError(comprehensive_error)
        else:
            # No fallback available or disabled
            fallback_status = self._get_fallback_status_message(use_fallback)
            
            # Create detailed error message with troubleshooting info
            error_message = (
                f"JINA API failed ({error.error_type}): {str(error)}. "
                f"{fallback_status} "
                f"Troubleshooting info: {error_info}"
            )
            
            raise EmbeddingGenerationError(error_message)
    
    def _handle_unexpected_error(self, error: Exception, texts: List[str], 
                                use_fallback: bool) -> List[List[float]]:
        """Handle unexpected errors with fallback and detailed logging."""
        error_type = type(error).__name__
        self.logger.error(f"Unexpected error in generate_embeddings: {error}")
        self._track_error("unexpected_error")
        
        if use_fallback and self.fallback_manager.is_fallback_available():
            self.logger.warning("Attempting fallback due to unexpected error")
            try:
                self.error_stats["fallback_uses"] += 1
                return self.fallback_manager.execute_fallback(
                    texts,
                    f"Unexpected error: {error_type}",
                    "unexpected_error"
                )
            except Exception as fallback_error:
                comprehensive_error = (
                    f"Unexpected error: {error}. "
                    f"Fallback also failed: {fallback_error}. "
                    f"This indicates a serious system issue. "
                    f"Please check system logs and configuration."
                )
                raise EmbeddingGenerationError(comprehensive_error)
        else:
            fallback_status = self._get_fallback_status_message(use_fallback)
            error_message = (
                f"Unexpected error: {error}. "
                f"{fallback_status} "
                f"Please check system configuration and logs."
            )
            raise EmbeddingGenerationError(error_message)
    
    def _create_comprehensive_error_message(self, jina_error: JinaAPIError, 
                                          fallback_error: Exception, 
                                          error_info: Dict[str, Any]) -> str:
        """Create a comprehensive error message with all relevant information."""
        message_parts = [
            f"JINA API failed ({jina_error.error_type}): {str(jina_error)}",
            f"Fallback to local embeddings also failed: {fallback_error}"
        ]
        
        # Add specific suggestions based on error types
        suggestions = error_info.get("suggestions", [])
        if suggestions:
            message_parts.append(f"JINA API suggestions: {'; '.join(suggestions)}")
        
        # Add fallback statistics if relevant
        fallback_stats = self.fallback_manager.get_fallback_stats()
        if fallback_stats["consecutive_fallbacks"] > 1:
            message_parts.append(
                f"This is the {fallback_stats['consecutive_fallbacks']} consecutive "
                f"fallback failure, indicating a persistent issue"
            )
        
        # Add system recommendations
        message_parts.append(
            "Recommendations: 1) Check JINA API status and credentials, "
            "2) Verify local embedding configuration, "
            "3) Check system resources and network connectivity, "
            "4) Review application logs for more details"
        )
        
        return ". ".join(message_parts)
    
    def _get_fallback_status_message(self, use_fallback: bool) -> str:
        """Get appropriate message about fallback status."""
        if not use_fallback:
            return "Fallback disabled by request"
        elif not self.fallback_manager.is_fallback_available():
            return "No fallback provider available"
        else:
            return "Fallback mechanism failed"
    
    def _generate_embeddings_with_jina(self, texts: List[str], task_type: str) -> List[List[float]]:
        """Generate embeddings using JINA API with batch processing and error recovery."""
        all_embeddings = []
        failed_batches = []
        
        # Calculate optimal batch size based on current conditions
        optimal_batch_size = min(self.batch_size, len(texts))
        if hasattr(self, 'rate_limit'):
            # Adjust batch size based on rate limit status
            rate_status = self.rate_limit.get_status()
            if rate_status["requests_utilization"] > 80:  # High utilization
                optimal_batch_size = max(1, optimal_batch_size // 2)
        
        # Process in batches with error recovery
        for i in range(0, len(texts), optimal_batch_size):
            batch = texts[i:i + optimal_batch_size]
            batch_num = i // optimal_batch_size + 1
            total_batches = (len(texts) + optimal_batch_size - 1) // optimal_batch_size
            
            try:
                self.logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")
                
                # Pre-batch validation
                if not batch:
                    self.logger.warning(f"Empty batch {batch_num}, skipping")
                    continue
                
                batch_embeddings = self._generate_batch_embeddings(batch, task_type)
                
                # Validate batch result
                if len(batch_embeddings) != len(batch):
                    raise JinaAPIError(
                        f"Batch {batch_num} result mismatch: expected {len(batch)} embeddings, "
                        f"got {len(batch_embeddings)}",
                        error_type="batch_result_mismatch",
                        fallback_available=self.fallback_manager.is_fallback_available()
                    )
                
                all_embeddings.extend(batch_embeddings)
                
                # Adaptive delay between batches based on API performance
                if i + optimal_batch_size < len(texts):
                    delay = self._calculate_inter_batch_delay(batch_num, total_batches)
                    if delay > 0:
                        time.sleep(delay)
                    
            except JinaAPIError as e:
                # Add batch context to the error
                enhanced_message = f"Batch {batch_num}/{total_batches} failed: {str(e)}"
                # Store the enhanced message in the exception args
                e.args = (enhanced_message,)
                
                # Track failed batch for potential retry
                failed_batches.append({
                    "batch_num": batch_num,
                    "batch": batch,
                    "error": str(e),
                    "error_type": e.error_type
                })
                
                # Decide whether to continue or fail completely
                if self._should_continue_after_batch_failure(e, batch_num, total_batches):
                    self.logger.warning(f"Continuing after batch {batch_num} failure: {e}")
                    # Add placeholder embeddings to maintain index alignment
                    placeholder_embeddings = self._create_placeholder_embeddings(len(batch))
                    all_embeddings.extend(placeholder_embeddings)
                    continue
                else:
                    # Critical failure - stop processing
                    self.logger.error(f"Critical failure in batch {batch_num}, stopping: {e}")
                    raise e
                    
            except Exception as e:
                error_context = {
                    "batch_num": batch_num,
                    "total_batches": total_batches,
                    "batch_size": len(batch),
                    "processed_so_far": len(all_embeddings),
                    "error_type": type(e).__name__
                }
                
                self.logger.error(f"Unexpected error in batch {batch_num}: {e}, context: {error_context}")
                
                raise JinaAPIError(
                    f"Unexpected error processing batch {batch_num}/{total_batches}: {e}",
                    error_type="batch_processing_error",
                    fallback_available=self.fallback_manager.is_fallback_available()
                )
        
        # Handle any failed batches if we continued processing
        if failed_batches:
            self._handle_failed_batches(failed_batches, all_embeddings)
        
        return all_embeddings
    
    def _calculate_inter_batch_delay(self, batch_num: int, total_batches: int) -> float:
        """Calculate adaptive delay between batches based on API performance."""
        base_delay = 0.1
        
        # Increase delay if we've had recent errors
        if self.error_stats["consecutive_failures"] > 0:
            base_delay *= (1 + self.error_stats["consecutive_failures"] * 0.5)
        
        # Reduce delay for later batches if everything is going well
        if batch_num > total_batches * 0.5 and self.error_stats["consecutive_failures"] == 0:
            base_delay *= 0.5
        
        return min(base_delay, 2.0)  # Cap at 2 seconds
    
    def _should_continue_after_batch_failure(self, error: JinaAPIError, 
                                           batch_num: int, total_batches: int) -> bool:
        """Determine if processing should continue after a batch failure."""
        # Never continue on authentication or configuration errors
        if error.error_type in ["authentication", "bad_request", "configuration"]:
            return False
        
        # Don't continue if we're early in processing and having failures
        if batch_num <= 2 and error.error_type in ["server_error", "connection_error"]:
            return False
        
        # Continue on rate limiting or temporary server issues if we're making progress
        if error.error_type in ["rate_limit", "timeout", "server_error"]:
            # Only continue if we've successfully processed some batches
            return batch_num > 1
        
        # Default to not continuing for unknown errors
        return False
    
    def _create_placeholder_embeddings(self, count: int) -> List[List[float]]:
        """Create placeholder embeddings for failed batches to maintain alignment."""
        dimension = self.get_embedding_dimension()
        # Create zero vectors as placeholders
        return [[0.0] * dimension for _ in range(count)]
    
    def _handle_failed_batches(self, failed_batches: List[Dict], all_embeddings: List[List[float]]) -> None:
        """Handle failed batches and log comprehensive information."""
        total_failed = len(failed_batches)
        total_texts = len(all_embeddings)
        
        self.logger.warning(
            f"Processing completed with {total_failed} failed batches out of "
            f"{total_texts} total texts. Failed batch details:"
        )
        
        for batch_info in failed_batches:
            self.logger.warning(
                f"  Batch {batch_info['batch_num']}: {batch_info['error_type']} - "
                f"{batch_info['error']}"
            )
        
        # Update error statistics
        self.error_stats["partial_failures"] = self.error_stats.get("partial_failures", 0) + total_failed
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider."""
        return self.model_config["dimensions"]
    
    def is_available(self) -> bool:
        """Check if the JINA provider is available and properly configured."""
        try:
            # Test with a simple embedding request
            test_embeddings = self.generate_embeddings(["test"])
            return len(test_embeddings) == 1 and len(test_embeddings[0]) == self.get_embedding_dimension()
        except Exception as e:
            self.logger.warning(f"JINA provider availability check failed: {e}")
            return False
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the provider for logging and debugging."""
        return {
            "provider": "jina",
            "model": self.model_name,
            "dimensions": self.get_embedding_dimension(),
            "batch_size": self.batch_size,
            "timeout": self.timeout,
            "available": self.is_available(),
            "api_endpoint": self.BASE_URL,
            "rate_limit": {
                "requests_per_minute": self.rate_limit.requests_per_minute,
                "tokens_per_minute": self.rate_limit.tokens_per_minute,
                "current_requests": self.rate_limit.current_requests,
                "current_tokens": self.rate_limit.current_tokens
            },
            "model_config": self.model_config,
            "fallback_available": self.fallback_manager.is_fallback_available(),
            "error_stats": self.error_stats.copy(),
            "health_status": self._get_health_status()
        }
    
    def _get_health_status(self) -> Dict[str, Any]:
        """Get current health status of the provider."""
        consecutive_failures = self.error_stats["consecutive_failures"]
        total_errors = self.error_stats["total_errors"]
        
        if consecutive_failures == 0:
            status = "healthy"
        elif consecutive_failures < 3:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return {
            "status": status,
            "consecutive_failures": consecutive_failures,
            "total_errors": total_errors,
            "fallback_uses": self.error_stats["fallback_uses"],
            "last_error_time": self.error_stats["last_error_time"],
            "recommendations": self._get_health_recommendations(status, consecutive_failures)
        }
    
    def _get_health_recommendations(self, status: str, consecutive_failures: int) -> List[str]:
        """Get health-based recommendations."""
        recommendations = []
        
        if status == "unhealthy":
            recommendations.extend([
                "JINA API is experiencing persistent issues",
                "Check JINA service status at https://status.jina.ai/",
                "Verify your API key and account status",
                "Consider using fallback to local embeddings",
                "Check network connectivity and firewall settings",
                "Review recent error logs for patterns",
                "Consider reducing batch size or request frequency"
            ])
        elif status == "degraded":
            recommendations.extend([
                "JINA API is experiencing intermittent issues",
                "Monitor error patterns and consider fallback if issues persist",
                "Check for rate limiting or temporary service disruptions",
                "Verify network stability and DNS resolution",
                "Consider implementing request queuing"
            ])
        else:  # healthy
            recommendations.extend([
                "JINA API is operating normally",
                "Continue monitoring for any changes in performance",
                "Ensure fallback provider is configured for resilience"
            ])
        
        return recommendations
    
    def get_comprehensive_troubleshooting_guide(self) -> Dict[str, Any]:
        """Get comprehensive troubleshooting guide for JINA embedding issues."""
        provider_info = self.get_provider_info()
        fallback_stats = self.fallback_manager.get_fallback_stats()
        
        guide = {
            "provider_status": provider_info["health_status"],
            "common_issues": {
                "authentication_errors": {
                    "symptoms": [
                        "401 Unauthorized responses",
                        "Invalid API key messages",
                        "Authentication failed errors"
                    ],
                    "solutions": [
                        "Verify JINA_API_KEY environment variable is set correctly",
                        "Check if API key has expired or been revoked",
                        "Ensure API key has sufficient permissions",
                        "Get a new API key at https://jina.ai/?sui=apikey",
                        "Verify account status and billing information"
                    ]
                },
                "rate_limiting": {
                    "symptoms": [
                        "429 Too Many Requests responses",
                        "Rate limit exceeded messages",
                        "Requests being throttled"
                    ],
                    "solutions": [
                        "Reduce batch size (current: {})".format(self.batch_size),
                        "Implement request queuing with delays",
                        "Upgrade JINA plan for higher rate limits",
                        "Use exponential backoff for retries",
                        "Monitor rate limit status in provider info"
                    ]
                },
                "network_issues": {
                    "symptoms": [
                        "Connection timeout errors",
                        "DNS resolution failures",
                        "Network unreachable messages"
                    ],
                    "solutions": [
                        "Check internet connectivity",
                        "Verify DNS settings and resolution",
                        "Check firewall and proxy settings",
                        "Test connectivity to api.jina.ai",
                        "Consider using different network or VPN"
                    ]
                },
                "server_errors": {
                    "symptoms": [
                        "5xx HTTP status codes",
                        "Internal server error messages",
                        "Service unavailable responses"
                    ],
                    "solutions": [
                        "Check JINA service status page",
                        "Wait and retry with exponential backoff",
                        "Use fallback to local embeddings",
                        "Contact JINA support if issues persist",
                        "Monitor for service restoration"
                    ]
                },
                "fallback_issues": {
                    "symptoms": [
                        "Local embedding provider not available",
                        "Fallback generation failures",
                        "Consecutive fallback errors"
                    ],
                    "solutions": [
                        "Verify LOCAL_EMBEDDING_MODEL configuration",
                        "Check local model installation and accessibility",
                        "Ensure sufficient system resources (memory, disk)",
                        "Verify HuggingFace model cache",
                        "Consider using different local model"
                    ]
                }
            },
            "diagnostic_steps": [
                "1. Check provider availability: call is_available()",
                "2. Review error statistics in provider info",
                "3. Test with simple single text embedding",
                "4. Verify environment variables and configuration",
                "5. Check system resources and network connectivity",
                "6. Review application logs for error patterns",
                "7. Test fallback provider independently",
                "8. Monitor rate limit status and usage"
            ],
            "configuration_checklist": {
                "required_env_vars": {
                    "JINA_API_KEY": "Your JINA AI API key",
                    "EMBEDDING_PROVIDER": "Set to 'jina'",
                },
                "optional_env_vars": {
                    "JINA_EMBEDDING_MODEL": f"Current: {self.model_name}",
                    "EMBEDDING_BATCH_SIZE": f"Current: {self.batch_size}",
                    "EMBEDDING_TIMEOUT": f"Current: {self.timeout}",
                    "LOCAL_EMBEDDING_MODEL": "For fallback provider"
                }
            },
            "performance_optimization": {
                "batch_size_tuning": [
                    f"Current batch size: {self.batch_size}",
                    "Reduce if hitting rate limits frequently",
                    "Increase if processing small texts efficiently",
                    "Monitor API response times and adjust accordingly"
                ],
                "error_handling": [
                    f"Fallback available: {fallback_stats['fallback_enabled']}",
                    f"Fallback success rate: {fallback_stats['fallback_success_rate']:.1f}%",
                    "Enable fallback for production resilience",
                    "Monitor consecutive failure patterns"
                ]
            },
            "monitoring_metrics": {
                "error_rates": self.error_stats,
                "fallback_usage": fallback_stats,
                "rate_limit_status": provider_info.get("rate_limit", {}),
                "health_status": provider_info["health_status"]
            }
        }
        
        return guide
    
    def diagnose_connection_issues(self) -> Dict[str, Any]:
        """Diagnose specific connection and API issues."""
        diagnosis = {
            "timestamp": time.time(),
            "tests_performed": [],
            "issues_found": [],
            "recommendations": []
        }
        
        # Test 1: Basic connectivity
        try:
            import socket
            socket.create_connection(("api.jina.ai", 443), timeout=5)
            diagnosis["tests_performed"].append("Basic connectivity: PASSED")
        except Exception as e:
            diagnosis["tests_performed"].append(f"Basic connectivity: FAILED - {e}")
            diagnosis["issues_found"].append("Cannot connect to JINA API endpoint")
            diagnosis["recommendations"].extend([
                "Check internet connectivity",
                "Verify DNS resolution for api.jina.ai",
                "Check firewall and proxy settings"
            ])
        
        # Test 2: API key validation
        if not self.api_key:
            diagnosis["tests_performed"].append("API key validation: FAILED - No key provided")
            diagnosis["issues_found"].append("JINA API key not configured")
            diagnosis["recommendations"].append("Set JINA_API_KEY environment variable")
        elif len(self.api_key) < 10:  # Basic length check
            diagnosis["tests_performed"].append("API key validation: FAILED - Key too short")
            diagnosis["issues_found"].append("JINA API key appears invalid")
            diagnosis["recommendations"].append("Verify API key format and get new key if needed")
        else:
            diagnosis["tests_performed"].append("API key validation: PASSED - Key format OK")
        
        # Test 3: Rate limit status
        if hasattr(self, 'rate_limit'):
            rate_status = self.rate_limit.get_status()
            if rate_status["requests_utilization"] > 90:
                diagnosis["tests_performed"].append("Rate limit check: WARNING - High utilization")
                diagnosis["issues_found"].append("Approaching rate limits")
                diagnosis["recommendations"].append("Reduce request frequency or upgrade plan")
            else:
                diagnosis["tests_performed"].append("Rate limit check: PASSED")
        
        # Test 4: Fallback availability
        if self.fallback_manager.is_fallback_available():
            diagnosis["tests_performed"].append("Fallback availability: PASSED")
        else:
            diagnosis["tests_performed"].append("Fallback availability: FAILED")
            diagnosis["issues_found"].append("No fallback provider available")
            diagnosis["recommendations"].append("Configure local embedding provider for fallback")
        
        # Test 5: Recent error patterns
        consecutive_failures = self.error_stats.get("consecutive_failures", 0)
        if consecutive_failures > 3:
            diagnosis["tests_performed"].append(f"Error pattern check: WARNING - {consecutive_failures} consecutive failures")
            diagnosis["issues_found"].append("Pattern of consecutive failures detected")
            diagnosis["recommendations"].extend([
                "Check JINA service status",
                "Consider using fallback temporarily",
                "Review error logs for specific failure causes"
            ])
        else:
            diagnosis["tests_performed"].append("Error pattern check: PASSED")
        
        return diagnosis
    
    def get_detailed_error_report(self) -> Dict[str, Any]:
        """Get detailed error report for troubleshooting."""
        return {
            "provider": "jina",
            "model": self.model_name,
            "error_statistics": self.error_stats.copy(),
            "health_status": self._get_health_status(),
            "configuration": {
                "api_key_configured": bool(self.api_key),
                "api_key_preview": f"{self.api_key[:8]}...{self.api_key[-4:]}" if self.api_key else None,
                "model_name": self.model_name,
                "batch_size": self.batch_size,
                "timeout": self.timeout,
                "base_url": self.BASE_URL
            },
            "fallback_info": {
                "available": self.fallback_manager.is_fallback_available(),
                "uses": self.error_stats["fallback_uses"]
            },
            "troubleshooting_steps": [
                "1. Verify JINA API key is valid and not expired",
                "2. Check network connectivity to api.jina.ai",
                "3. Verify rate limits are not exceeded",
                "4. Check JINA service status",
                "5. Try with smaller batch sizes",
                "6. Enable fallback to local embeddings if available"
            ]
        }
    
    def test_connection(self) -> Dict[str, Any]:
        """Test connection to JINA API with detailed diagnostics."""
        test_result = {
            "success": False,
            "timestamp": time.time(),
            "details": {},
            "errors": [],
            "recommendations": []
        }
        
        try:
            # Test with a simple embedding request
            test_text = ["Connection test"]
            start_time = time.time()
            
            embeddings = self._generate_batch_embeddings(test_text, "retrieval.passage")
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if len(embeddings) == 1 and len(embeddings[0]) == self.get_embedding_dimension():
                test_result["success"] = True
                test_result["details"] = {
                    "response_time_seconds": response_time,
                    "embedding_dimension": len(embeddings[0]),
                    "expected_dimension": self.get_embedding_dimension(),
                    "api_responsive": True
                }
                test_result["recommendations"].append("JINA API is working correctly")
            else:
                test_result["errors"].append("Unexpected embedding format or dimensions")
                
        except JinaAPIError as e:
            error_info = e.get_troubleshooting_info()
            test_result["errors"].append(f"JINA API Error: {e}")
            test_result["details"]["error_info"] = error_info
            test_result["recommendations"].extend(error_info.get("suggestions", []))
            
        except Exception as e:
            test_result["errors"].append(f"Unexpected error: {e}")
            test_result["recommendations"].extend([
                "Check network connectivity",
                "Verify API configuration",
                "Try again in a few minutes"
            ])
        
        return test_result
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics for monitoring."""
        self.rate_limit.reset_if_needed()
        
        return {
            "requests_used": self.rate_limit.current_requests,
            "requests_remaining": self.rate_limit.requests_per_minute - self.rate_limit.current_requests,
            "tokens_used": self.rate_limit.current_tokens,
            "tokens_remaining": self.rate_limit.tokens_per_minute - self.rate_limit.current_tokens,
            "window_start": self.rate_limit.window_start,
            "window_remaining": max(0, 60 - (time.time() - self.rate_limit.window_start))
        }
    
    def __del__(self):
        """Cleanup resources when provider is destroyed."""
        if hasattr(self, 'session'):
            self.session.close()


# Async version for high-performance scenarios
class AsyncJinaEmbeddingProvider:
    """Async JINA AI embedding provider for high-performance scenarios."""
    
    BASE_URL = "https://api.jina.ai/v1/embeddings"
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize async JINA embedding provider."""
        self.config = config
        self.api_key = config.api_key
        self.model_name = config.model_name
        self.batch_size = min(config.batch_size, 100)
        self.timeout = config.timeout
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Rate limiting
        self.rate_limit = RateLimitInfo()
        
        # Session will be created when needed
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "RAG-System/1.0"
            }
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers
            )
        return self._session
    
    async def generate_embeddings_async(self, texts: List[str], task_type: str = "retrieval.passage") -> List[List[float]]:
        """Generate embeddings asynchronously."""
        if not texts:
            return []
        
        session = await self._get_session()
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            payload = {
                "model": self.model_name,
                "input": batch,
                "task": task_type
            }
            
            try:
                async with session.post(self.BASE_URL, json=payload) as response:
                    if response.status == 429:
                        # Rate limited, wait and retry
                        await asyncio.sleep(1)
                        async with session.post(self.BASE_URL, json=payload) as retry_response:
                            retry_response.raise_for_status()
                            response_data = await retry_response.json()
                    else:
                        response.raise_for_status()
                        response_data = await response.json()
                    
                    # Extract embeddings
                    batch_embeddings = [item["embedding"] for item in response_data["data"]]
                    all_embeddings.extend(batch_embeddings)
                    
                    # Small delay between batches
                    if i + self.batch_size < len(texts):
                        await asyncio.sleep(0.1)
                        
            except Exception as e:
                raise EmbeddingGenerationError(f"Async embedding generation failed: {e}")
        
        return all_embeddings
    
    async def close(self):
        """Close the async session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
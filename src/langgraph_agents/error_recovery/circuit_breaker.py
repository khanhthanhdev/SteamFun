"""
Circuit Breaker pattern implementation for error handling.

This module implements the Circuit Breaker pattern to prevent cascading failures
and provide fast failure when services are unavailable. The circuit breaker
monitors failure rates and automatically opens/closes based on configurable thresholds.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, Callable, Awaitable, TypeVar, Generic, List
from dataclasses import dataclass, field
from collections import deque

from ..models.errors import WorkflowError, ErrorType

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Failing fast, requests are rejected
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    
    # Failure thresholds
    failure_threshold: int = 5  # Number of failures before opening
    success_threshold: int = 3  # Number of successes needed to close from half-open
    
    # Timing configuration
    timeout_seconds: int = 60  # How long to stay open before trying half-open
    half_open_timeout: int = 30  # How long to wait in half-open before going back to open
    
    # Monitoring window
    monitoring_window_seconds: int = 300  # 5 minutes window for failure rate calculation
    failure_rate_threshold: float = 0.5  # 50% failure rate threshold
    
    # Request volume threshold (minimum requests before considering failure rate)
    minimum_request_volume: int = 10
    
    # Recovery settings
    recovery_timeout_multiplier: float = 1.5  # Multiply timeout on repeated failures
    max_timeout_seconds: int = 600  # Maximum timeout (10 minutes)


@dataclass
class CircuitBreakerMetrics:
    """Metrics tracked by the circuit breaker."""
    
    # State tracking
    current_state: CircuitBreakerState = CircuitBreakerState.CLOSED
    state_changed_at: datetime = field(default_factory=datetime.now)
    
    # Failure tracking
    failure_count: int = 0
    success_count: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    
    # Request tracking
    total_requests: int = 0
    rejected_requests: int = 0
    
    # Timing
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    next_attempt_time: Optional[datetime] = None
    
    # Historical data (for monitoring window)
    recent_requests: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def add_request_result(self, success: bool, timestamp: Optional[datetime] = None):
        """Add a request result to the metrics."""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.total_requests += 1
        self.recent_requests.append({
            'success': success,
            'timestamp': timestamp
        })
        
        if success:
            self.success_count += 1
            self.consecutive_successes += 1
            self.consecutive_failures = 0
            self.last_success_time = timestamp
        else:
            self.failure_count += 1
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            self.last_failure_time = timestamp
    
    def get_failure_rate(self, window_seconds: int = 300) -> float:
        """Calculate failure rate within the specified time window."""
        if not self.recent_requests:
            return 0.0
        
        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
        recent_in_window = [
            req for req in self.recent_requests
            if req['timestamp'] >= cutoff_time
        ]
        
        if len(recent_in_window) == 0:
            return 0.0
        
        failures = sum(1 for req in recent_in_window if not req['success'])
        return failures / len(recent_in_window)
    
    def get_request_volume(self, window_seconds: int = 300) -> int:
        """Get request volume within the specified time window."""
        if not self.recent_requests:
            return 0
        
        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
        return sum(1 for req in self.recent_requests if req['timestamp'] >= cutoff_time)


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    
    def __init__(self, circuit_name: str, next_attempt_time: Optional[datetime] = None):
        self.circuit_name = circuit_name
        self.next_attempt_time = next_attempt_time
        
        if next_attempt_time:
            wait_seconds = (next_attempt_time - datetime.now()).total_seconds()
            message = f"Circuit breaker '{circuit_name}' is open. Next attempt in {wait_seconds:.1f} seconds."
        else:
            message = f"Circuit breaker '{circuit_name}' is open."
        
        super().__init__(message)


class CircuitBreaker(Generic[T]):
    """
    Circuit breaker implementation that monitors failures and prevents cascading failures.
    
    The circuit breaker has three states:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Fast failure, all requests are rejected immediately
    - HALF_OPEN: Testing recovery, limited requests are allowed through
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """Initialize the circuit breaker."""
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.metrics = CircuitBreakerMetrics()
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        self.logger.info(f"Circuit breaker '{name}' initialized with config: {self.config}")
    
    async def call(
        self, 
        func: Callable[..., Awaitable[T]], 
        *args, 
        **kwargs
    ) -> T:
        """
        Execute a function call through the circuit breaker.
        
        Args:
            func: Async function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function call
            
        Raises:
            CircuitBreakerOpenError: If circuit breaker is open
            Any exception raised by the wrapped function
        """
        async with self._lock:
            # Check if we should allow the request
            if not self._should_allow_request():
                self.metrics.rejected_requests += 1
                raise CircuitBreakerOpenError(self.name, self.metrics.next_attempt_time)
        
        # Execute the function
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            
            # Record success
            await self._record_success()
            
            execution_time = time.time() - start_time
            self.logger.debug(f"Circuit breaker '{self.name}' - successful call in {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            # Record failure
            await self._record_failure(e)
            
            execution_time = time.time() - start_time
            self.logger.warning(f"Circuit breaker '{self.name}' - failed call in {execution_time:.3f}s: {str(e)}")
            
            raise
    
    def _should_allow_request(self) -> bool:
        """Determine if a request should be allowed through the circuit breaker."""
        current_time = datetime.now()
        
        if self.metrics.current_state == CircuitBreakerState.CLOSED:
            return True
        
        elif self.metrics.current_state == CircuitBreakerState.OPEN:
            # Check if timeout has elapsed
            if (self.metrics.next_attempt_time and 
                current_time >= self.metrics.next_attempt_time):
                self._transition_to_half_open()
                return True
            return False
        
        elif self.metrics.current_state == CircuitBreakerState.HALF_OPEN:
            # In half-open state, allow limited requests
            return True
        
        return False
    
    async def _record_success(self):
        """Record a successful request."""
        async with self._lock:
            self.metrics.add_request_result(success=True)
            
            if self.metrics.current_state == CircuitBreakerState.HALF_OPEN:
                # Check if we have enough successes to close the circuit
                if self.metrics.consecutive_successes >= self.config.success_threshold:
                    self._transition_to_closed()
    
    async def _record_failure(self, exception: Exception):
        """Record a failed request."""
        async with self._lock:
            self.metrics.add_request_result(success=False)
            
            # Check if we should open the circuit
            if self.metrics.current_state == CircuitBreakerState.CLOSED:
                if self._should_open_circuit():
                    self._transition_to_open()
            
            elif self.metrics.current_state == CircuitBreakerState.HALF_OPEN:
                # Any failure in half-open state should open the circuit
                self._transition_to_open()
    
    def _should_open_circuit(self) -> bool:
        """Determine if the circuit should be opened based on current metrics."""
        # Check consecutive failures threshold
        if self.metrics.consecutive_failures >= self.config.failure_threshold:
            return True
        
        # Check failure rate within monitoring window
        request_volume = self.metrics.get_request_volume(self.config.monitoring_window_seconds)
        if request_volume >= self.config.minimum_request_volume:
            failure_rate = self.metrics.get_failure_rate(self.config.monitoring_window_seconds)
            if failure_rate >= self.config.failure_rate_threshold:
                return True
        
        return False
    
    def _transition_to_open(self):
        """Transition circuit breaker to OPEN state."""
        previous_state = self.metrics.current_state
        self.metrics.current_state = CircuitBreakerState.OPEN
        self.metrics.state_changed_at = datetime.now()
        
        # Calculate next attempt time with exponential backoff
        base_timeout = self.config.timeout_seconds
        
        # Apply exponential backoff based on recent failures
        if self.metrics.last_failure_time:
            recent_failures = self._count_recent_failures(minutes=10)
            multiplier = self.config.recovery_timeout_multiplier ** max(0, recent_failures - 1)
            timeout = min(base_timeout * multiplier, self.config.max_timeout_seconds)
        else:
            timeout = base_timeout
        
        self.metrics.next_attempt_time = datetime.now() + timedelta(seconds=timeout)
        
        self.logger.warning(
            f"Circuit breaker '{self.name}' opened (was {previous_state.value}). "
            f"Next attempt at {self.metrics.next_attempt_time.isoformat()}"
        )
    
    def _transition_to_half_open(self):
        """Transition circuit breaker to HALF_OPEN state."""
        previous_state = self.metrics.current_state
        self.metrics.current_state = CircuitBreakerState.HALF_OPEN
        self.metrics.state_changed_at = datetime.now()
        self.metrics.consecutive_successes = 0
        
        self.logger.info(f"Circuit breaker '{self.name}' transitioned to half-open (was {previous_state.value})")
    
    def _transition_to_closed(self):
        """Transition circuit breaker to CLOSED state."""
        previous_state = self.metrics.current_state
        self.metrics.current_state = CircuitBreakerState.CLOSED
        self.metrics.state_changed_at = datetime.now()
        self.metrics.consecutive_failures = 0
        self.metrics.next_attempt_time = None
        
        self.logger.info(f"Circuit breaker '{self.name}' closed (was {previous_state.value})")
    
    def _count_recent_failures(self, minutes: int = 10) -> int:
        """Count failures in the recent time window."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return sum(
            1 for req in self.metrics.recent_requests
            if not req['success'] and req['timestamp'] >= cutoff_time
        )
    
    def get_state(self) -> CircuitBreakerState:
        """Get the current state of the circuit breaker."""
        return self.metrics.current_state
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics for monitoring."""
        return {
            'name': self.name,
            'state': self.metrics.current_state.value,
            'state_changed_at': self.metrics.state_changed_at.isoformat(),
            'failure_count': self.metrics.failure_count,
            'success_count': self.metrics.success_count,
            'consecutive_failures': self.metrics.consecutive_failures,
            'consecutive_successes': self.metrics.consecutive_successes,
            'total_requests': self.metrics.total_requests,
            'rejected_requests': self.metrics.rejected_requests,
            'failure_rate_5min': self.metrics.get_failure_rate(300),
            'request_volume_5min': self.metrics.get_request_volume(300),
            'next_attempt_time': self.metrics.next_attempt_time.isoformat() if self.metrics.next_attempt_time else None,
            'last_failure_time': self.metrics.last_failure_time.isoformat() if self.metrics.last_failure_time else None,
            'last_success_time': self.metrics.last_success_time.isoformat() if self.metrics.last_success_time else None
        }
    
    def reset(self):
        """Reset the circuit breaker to initial state."""
        with self._lock:
            self.metrics = CircuitBreakerMetrics()
            self.logger.info(f"Circuit breaker '{self.name}' reset to initial state")
    
    def force_open(self, timeout_seconds: Optional[int] = None):
        """Force the circuit breaker to open state."""
        with self._lock:
            self.metrics.current_state = CircuitBreakerState.OPEN
            self.metrics.state_changed_at = datetime.now()
            
            timeout = timeout_seconds or self.config.timeout_seconds
            self.metrics.next_attempt_time = datetime.now() + timedelta(seconds=timeout)
            
            self.logger.warning(f"Circuit breaker '{self.name}' forced to open state")
    
    def force_close(self):
        """Force the circuit breaker to closed state."""
        with self._lock:
            self.metrics.current_state = CircuitBreakerState.CLOSED
            self.metrics.state_changed_at = datetime.now()
            self.metrics.consecutive_failures = 0
            self.metrics.next_attempt_time = None
            
            self.logger.info(f"Circuit breaker '{self.name}' forced to closed state")


class CircuitBreakerManager:
    """
    Manager for multiple circuit breakers.
    
    This class provides a centralized way to manage circuit breakers for different
    services or operations in the workflow.
    """
    
    def __init__(self):
        """Initialize the circuit breaker manager."""
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.logger = logging.getLogger(__name__)
    
    def get_circuit_breaker(
        self, 
        name: str, 
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """
        Get or create a circuit breaker with the given name.
        
        Args:
            name: Name of the circuit breaker
            config: Configuration for the circuit breaker (if creating new)
            
        Returns:
            Circuit breaker instance
        """
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name, config)
            self.logger.info(f"Created new circuit breaker: {name}")
        
        return self.circuit_breakers[name]
    
    def remove_circuit_breaker(self, name: str) -> bool:
        """
        Remove a circuit breaker.
        
        Args:
            name: Name of the circuit breaker to remove
            
        Returns:
            True if removed, False if not found
        """
        if name in self.circuit_breakers:
            del self.circuit_breakers[name]
            self.logger.info(f"Removed circuit breaker: {name}")
            return True
        return False
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers."""
        return {
            name: cb.get_metrics()
            for name, cb in self.circuit_breakers.items()
        }
    
    def reset_all(self):
        """Reset all circuit breakers."""
        for cb in self.circuit_breakers.values():
            cb.reset()
        self.logger.info("Reset all circuit breakers")
    
    def get_open_circuit_breakers(self) -> List[str]:
        """Get names of all open circuit breakers."""
        return [
            name for name, cb in self.circuit_breakers.items()
            if cb.get_state() == CircuitBreakerState.OPEN
        ]
    
    def get_circuit_breaker_count(self) -> int:
        """Get the total number of circuit breakers."""
        return len(self.circuit_breakers)


# Global circuit breaker manager instance
circuit_breaker_manager = CircuitBreakerManager()


def circuit_breaker(
    name: str, 
    config: Optional[CircuitBreakerConfig] = None
):
    """
    Decorator to wrap a function with a circuit breaker.
    
    Args:
        name: Name of the circuit breaker
        config: Configuration for the circuit breaker
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        cb = circuit_breaker_manager.get_circuit_breaker(name, config)
        
        async def wrapper(*args, **kwargs) -> T:
            return await cb.call(func, *args, **kwargs)
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    return decorator
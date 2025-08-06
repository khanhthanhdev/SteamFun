"""
Unit tests for the CircuitBreaker implementation.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from src.langgraph_agents.error_recovery.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerState,
    CircuitBreakerConfig,
    CircuitBreakerMetrics,
    CircuitBreakerOpenError,
    CircuitBreakerManager,
    circuit_breaker,
    circuit_breaker_manager
)


class TestCircuitBreakerConfig:
    """Test cases for CircuitBreakerConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        
        assert config.failure_threshold == 5
        assert config.success_threshold == 3
        assert config.timeout_seconds == 60
        assert config.half_open_timeout == 30
        assert config.monitoring_window_seconds == 300
        assert config.failure_rate_threshold == 0.5
        assert config.minimum_request_volume == 10
        assert config.recovery_timeout_multiplier == 1.5
        assert config.max_timeout_seconds == 600
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=30,
            failure_rate_threshold=0.3
        )
        
        assert config.failure_threshold == 3
        assert config.success_threshold == 2
        assert config.timeout_seconds == 30
        assert config.failure_rate_threshold == 0.3


class TestCircuitBreakerMetrics:
    """Test cases for CircuitBreakerMetrics."""
    
    def test_initial_metrics(self):
        """Test initial metrics state."""
        metrics = CircuitBreakerMetrics()
        
        assert metrics.current_state == CircuitBreakerState.CLOSED
        assert metrics.failure_count == 0
        assert metrics.success_count == 0
        assert metrics.consecutive_failures == 0
        assert metrics.consecutive_successes == 0
        assert metrics.total_requests == 0
        assert metrics.rejected_requests == 0
        assert metrics.last_failure_time is None
        assert metrics.last_success_time is None
        assert metrics.next_attempt_time is None
        assert len(metrics.recent_requests) == 0
    
    def test_add_success_result(self):
        """Test adding successful request result."""
        metrics = CircuitBreakerMetrics()
        timestamp = datetime.now()
        
        metrics.add_request_result(success=True, timestamp=timestamp)
        
        assert metrics.total_requests == 1
        assert metrics.success_count == 1
        assert metrics.failure_count == 0
        assert metrics.consecutive_successes == 1
        assert metrics.consecutive_failures == 0
        assert metrics.last_success_time == timestamp
        assert metrics.last_failure_time is None
        assert len(metrics.recent_requests) == 1
        assert metrics.recent_requests[0]['success'] is True
    
    def test_add_failure_result(self):
        """Test adding failed request result."""
        metrics = CircuitBreakerMetrics()
        timestamp = datetime.now()
        
        metrics.add_request_result(success=False, timestamp=timestamp)
        
        assert metrics.total_requests == 1
        assert metrics.success_count == 0
        assert metrics.failure_count == 1
        assert metrics.consecutive_successes == 0
        assert metrics.consecutive_failures == 1
        assert metrics.last_success_time is None
        assert metrics.last_failure_time == timestamp
        assert len(metrics.recent_requests) == 1
        assert metrics.recent_requests[0]['success'] is False
    
    def test_consecutive_counts_reset(self):
        """Test that consecutive counts reset properly."""
        metrics = CircuitBreakerMetrics()
        
        # Add failures
        metrics.add_request_result(success=False)
        metrics.add_request_result(success=False)
        assert metrics.consecutive_failures == 2
        assert metrics.consecutive_successes == 0
        
        # Add success - should reset consecutive failures
        metrics.add_request_result(success=True)
        assert metrics.consecutive_failures == 0
        assert metrics.consecutive_successes == 1
        
        # Add failure - should reset consecutive successes
        metrics.add_request_result(success=False)
        assert metrics.consecutive_failures == 1
        assert metrics.consecutive_successes == 0
    
    def test_get_failure_rate_empty(self):
        """Test failure rate calculation with no requests."""
        metrics = CircuitBreakerMetrics()
        
        failure_rate = metrics.get_failure_rate()
        assert failure_rate == 0.0
    
    def test_get_failure_rate_with_requests(self):
        """Test failure rate calculation with mixed requests."""
        metrics = CircuitBreakerMetrics()
        
        # Add 3 successes and 2 failures
        for _ in range(3):
            metrics.add_request_result(success=True)
        for _ in range(2):
            metrics.add_request_result(success=False)
        
        failure_rate = metrics.get_failure_rate()
        assert failure_rate == 0.4  # 2 failures out of 5 requests
    
    def test_get_failure_rate_with_time_window(self):
        """Test failure rate calculation within time window."""
        metrics = CircuitBreakerMetrics()
        
        # Add old requests (outside window)
        old_time = datetime.now() - timedelta(seconds=400)
        metrics.add_request_result(success=False, timestamp=old_time)
        metrics.add_request_result(success=False, timestamp=old_time)
        
        # Add recent requests (within window)
        recent_time = datetime.now() - timedelta(seconds=100)
        metrics.add_request_result(success=True, timestamp=recent_time)
        metrics.add_request_result(success=False, timestamp=recent_time)
        
        # Should only consider recent requests (1 failure out of 2)
        failure_rate = metrics.get_failure_rate(window_seconds=300)
        assert failure_rate == 0.5
    
    def test_get_request_volume(self):
        """Test request volume calculation."""
        metrics = CircuitBreakerMetrics()
        
        # Add requests
        for _ in range(5):
            metrics.add_request_result(success=True)
        
        volume = metrics.get_request_volume()
        assert volume == 5
    
    def test_get_request_volume_with_time_window(self):
        """Test request volume calculation within time window."""
        metrics = CircuitBreakerMetrics()
        
        # Add old requests (outside window)
        old_time = datetime.now() - timedelta(seconds=400)
        metrics.add_request_result(success=True, timestamp=old_time)
        metrics.add_request_result(success=True, timestamp=old_time)
        
        # Add recent requests (within window)
        recent_time = datetime.now() - timedelta(seconds=100)
        metrics.add_request_result(success=True, timestamp=recent_time)
        metrics.add_request_result(success=False, timestamp=recent_time)
        
        # Should only count recent requests
        volume = metrics.get_request_volume(window_seconds=300)
        assert volume == 2


class TestCircuitBreaker:
    """Test cases for CircuitBreaker."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=60,
            minimum_request_volume=5
        )
    
    @pytest.fixture
    def circuit_breaker(self, config):
        """Create a circuit breaker instance."""
        return CircuitBreaker("test_circuit", config)
    
    def test_initialization(self, circuit_breaker):
        """Test circuit breaker initialization."""
        assert circuit_breaker.name == "test_circuit"
        assert circuit_breaker.get_state() == CircuitBreakerState.CLOSED
        assert circuit_breaker.metrics.total_requests == 0
    
    @pytest.mark.asyncio
    async def test_successful_call(self, circuit_breaker):
        """Test successful function call through circuit breaker."""
        async def test_func():
            return "success"
        
        result = await circuit_breaker.call(test_func)
        
        assert result == "success"
        assert circuit_breaker.metrics.success_count == 1
        assert circuit_breaker.metrics.failure_count == 0
        assert circuit_breaker.get_state() == CircuitBreakerState.CLOSED
    
    @pytest.mark.asyncio
    async def test_failed_call(self, circuit_breaker):
        """Test failed function call through circuit breaker."""
        async def test_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            await circuit_breaker.call(test_func)
        
        assert circuit_breaker.metrics.success_count == 0
        assert circuit_breaker.metrics.failure_count == 1
        assert circuit_breaker.get_state() == CircuitBreakerState.CLOSED
    
    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self, circuit_breaker):
        """Test that circuit opens after consecutive failures."""
        async def failing_func():
            raise ValueError("Test error")
        
        # Make failures up to threshold
        for i in range(3):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_func)
        
        # Circuit should now be open
        assert circuit_breaker.get_state() == CircuitBreakerState.OPEN
        
        # Next call should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            await circuit_breaker.call(failing_func)
        
        assert circuit_breaker.metrics.rejected_requests == 1
    
    @pytest.mark.asyncio
    async def test_circuit_opens_by_failure_rate(self, circuit_breaker):
        """Test that circuit opens based on failure rate."""
        async def mixed_func(should_fail):
            if should_fail:
                raise ValueError("Test error")
            return "success"
        
        # Make enough requests to meet minimum volume
        # 3 successes, 3 failures = 50% failure rate (meets threshold)
        for _ in range(3):
            await circuit_breaker.call(lambda: mixed_func(False))
        
        for _ in range(3):
            with pytest.raises(ValueError):
                await circuit_breaker.call(lambda: mixed_func(True))
        
        # Circuit should be open due to failure rate
        assert circuit_breaker.get_state() == CircuitBreakerState.OPEN
    
    @pytest.mark.asyncio
    async def test_circuit_transitions_to_half_open(self, circuit_breaker):
        """Test transition from open to half-open state."""
        # Force circuit to open
        circuit_breaker.force_open(timeout_seconds=1)
        assert circuit_breaker.get_state() == CircuitBreakerState.OPEN
        
        # Wait for timeout
        await asyncio.sleep(1.1)
        
        # Next call should transition to half-open
        async def test_func():
            return "success"
        
        result = await circuit_breaker.call(test_func)
        
        assert result == "success"
        assert circuit_breaker.get_state() == CircuitBreakerState.HALF_OPEN
    
    @pytest.mark.asyncio
    async def test_circuit_closes_from_half_open(self, circuit_breaker):
        """Test transition from half-open to closed state."""
        # Force circuit to half-open
        circuit_breaker.metrics.current_state = CircuitBreakerState.HALF_OPEN
        circuit_breaker.metrics.consecutive_successes = 0
        
        async def test_func():
            return "success"
        
        # Make successful calls to meet success threshold
        for _ in range(2):  # success_threshold = 2
            await circuit_breaker.call(test_func)
        
        # Circuit should now be closed
        assert circuit_breaker.get_state() == CircuitBreakerState.CLOSED
    
    @pytest.mark.asyncio
    async def test_circuit_opens_from_half_open_on_failure(self, circuit_breaker):
        """Test that circuit opens from half-open on any failure."""
        # Force circuit to half-open
        circuit_breaker.metrics.current_state = CircuitBreakerState.HALF_OPEN
        
        async def failing_func():
            raise ValueError("Test error")
        
        # Any failure in half-open should open the circuit
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_func)
        
        assert circuit_breaker.get_state() == CircuitBreakerState.OPEN
    
    def test_get_metrics(self, circuit_breaker):
        """Test getting circuit breaker metrics."""
        metrics = circuit_breaker.get_metrics()
        
        assert metrics['name'] == "test_circuit"
        assert metrics['state'] == CircuitBreakerState.CLOSED.value
        assert 'state_changed_at' in metrics
        assert 'failure_count' in metrics
        assert 'success_count' in metrics
        assert 'total_requests' in metrics
        assert 'failure_rate_5min' in metrics
        assert 'request_volume_5min' in metrics
    
    def test_reset(self, circuit_breaker):
        """Test resetting circuit breaker."""
        # Add some metrics
        circuit_breaker.metrics.failure_count = 5
        circuit_breaker.metrics.success_count = 3
        circuit_breaker.metrics.current_state = CircuitBreakerState.OPEN
        
        # Reset
        circuit_breaker.reset()
        
        assert circuit_breaker.get_state() == CircuitBreakerState.CLOSED
        assert circuit_breaker.metrics.failure_count == 0
        assert circuit_breaker.metrics.success_count == 0
    
    def test_force_open(self, circuit_breaker):
        """Test forcing circuit breaker to open state."""
        circuit_breaker.force_open(timeout_seconds=30)
        
        assert circuit_breaker.get_state() == CircuitBreakerState.OPEN
        assert circuit_breaker.metrics.next_attempt_time is not None
    
    def test_force_close(self, circuit_breaker):
        """Test forcing circuit breaker to closed state."""
        # First open the circuit
        circuit_breaker.force_open()
        assert circuit_breaker.get_state() == CircuitBreakerState.OPEN
        
        # Then force close
        circuit_breaker.force_close()
        
        assert circuit_breaker.get_state() == CircuitBreakerState.CLOSED
        assert circuit_breaker.metrics.consecutive_failures == 0
        assert circuit_breaker.metrics.next_attempt_time is None
    
    @pytest.mark.asyncio
    async def test_exponential_backoff_timeout(self, circuit_breaker):
        """Test that timeout increases with repeated failures."""
        # Mock the _count_recent_failures method to simulate repeated failures
        with patch.object(circuit_breaker, '_count_recent_failures', return_value=3):
            circuit_breaker._transition_to_open()
        
        # Check that timeout was increased
        assert circuit_breaker.metrics.next_attempt_time is not None
        
        # Calculate expected timeout with exponential backoff
        expected_timeout = 60 * (1.5 ** 2)  # base_timeout * multiplier^(failures-1)
        expected_timeout = min(expected_timeout, 600)  # Capped at max_timeout
        
        actual_timeout = (circuit_breaker.metrics.next_attempt_time - datetime.now()).total_seconds()
        
        # Allow some tolerance for timing differences
        assert abs(actual_timeout - expected_timeout) < 5


class TestCircuitBreakerOpenError:
    """Test cases for CircuitBreakerOpenError."""
    
    def test_error_without_next_attempt_time(self):
        """Test error creation without next attempt time."""
        error = CircuitBreakerOpenError("test_circuit")
        
        assert "test_circuit" in str(error)
        assert "is open" in str(error)
        assert error.circuit_name == "test_circuit"
        assert error.next_attempt_time is None
    
    def test_error_with_next_attempt_time(self):
        """Test error creation with next attempt time."""
        next_attempt = datetime.now() + timedelta(seconds=30)
        error = CircuitBreakerOpenError("test_circuit", next_attempt)
        
        assert "test_circuit" in str(error)
        assert "Next attempt in" in str(error)
        assert error.circuit_name == "test_circuit"
        assert error.next_attempt_time == next_attempt


class TestCircuitBreakerManager:
    """Test cases for CircuitBreakerManager."""
    
    @pytest.fixture
    def manager(self):
        """Create a circuit breaker manager."""
        return CircuitBreakerManager()
    
    def test_get_circuit_breaker_new(self, manager):
        """Test getting a new circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = manager.get_circuit_breaker("test_circuit", config)
        
        assert cb.name == "test_circuit"
        assert cb.config.failure_threshold == 3
        assert "test_circuit" in manager.circuit_breakers
    
    def test_get_circuit_breaker_existing(self, manager):
        """Test getting an existing circuit breaker."""
        # Create first circuit breaker
        cb1 = manager.get_circuit_breaker("test_circuit")
        
        # Get the same circuit breaker
        cb2 = manager.get_circuit_breaker("test_circuit")
        
        assert cb1 is cb2
    
    def test_remove_circuit_breaker(self, manager):
        """Test removing a circuit breaker."""
        # Create circuit breaker
        manager.get_circuit_breaker("test_circuit")
        assert "test_circuit" in manager.circuit_breakers
        
        # Remove it
        result = manager.remove_circuit_breaker("test_circuit")
        
        assert result is True
        assert "test_circuit" not in manager.circuit_breakers
    
    def test_remove_nonexistent_circuit_breaker(self, manager):
        """Test removing a non-existent circuit breaker."""
        result = manager.remove_circuit_breaker("nonexistent")
        assert result is False
    
    def test_get_all_metrics(self, manager):
        """Test getting metrics for all circuit breakers."""
        # Create multiple circuit breakers
        manager.get_circuit_breaker("circuit1")
        manager.get_circuit_breaker("circuit2")
        
        metrics = manager.get_all_metrics()
        
        assert len(metrics) == 2
        assert "circuit1" in metrics
        assert "circuit2" in metrics
        assert "state" in metrics["circuit1"]
        assert "state" in metrics["circuit2"]
    
    def test_reset_all(self, manager):
        """Test resetting all circuit breakers."""
        # Create circuit breakers and add some failures
        cb1 = manager.get_circuit_breaker("circuit1")
        cb2 = manager.get_circuit_breaker("circuit2")
        
        cb1.metrics.failure_count = 5
        cb2.metrics.failure_count = 3
        
        # Reset all
        manager.reset_all()
        
        assert cb1.metrics.failure_count == 0
        assert cb2.metrics.failure_count == 0
    
    def test_get_open_circuit_breakers(self, manager):
        """Test getting open circuit breakers."""
        # Create circuit breakers
        cb1 = manager.get_circuit_breaker("circuit1")
        cb2 = manager.get_circuit_breaker("circuit2")
        cb3 = manager.get_circuit_breaker("circuit3")
        
        # Open some circuit breakers
        cb1.force_open()
        cb3.force_open()
        
        open_circuits = manager.get_open_circuit_breakers()
        
        assert len(open_circuits) == 2
        assert "circuit1" in open_circuits
        assert "circuit3" in open_circuits
        assert "circuit2" not in open_circuits
    
    def test_get_circuit_breaker_count(self, manager):
        """Test getting circuit breaker count."""
        assert manager.get_circuit_breaker_count() == 0
        
        manager.get_circuit_breaker("circuit1")
        assert manager.get_circuit_breaker_count() == 1
        
        manager.get_circuit_breaker("circuit2")
        assert manager.get_circuit_breaker_count() == 2


class TestCircuitBreakerDecorator:
    """Test cases for circuit breaker decorator."""
    
    @pytest.mark.asyncio
    async def test_decorator_success(self):
        """Test circuit breaker decorator with successful function."""
        @circuit_breaker("test_decorator")
        async def test_func():
            return "success"
        
        result = await test_func()
        assert result == "success"
        
        # Check that circuit breaker was created
        cb = circuit_breaker_manager.get_circuit_breaker("test_decorator")
        assert cb.metrics.success_count == 1
    
    @pytest.mark.asyncio
    async def test_decorator_failure(self):
        """Test circuit breaker decorator with failing function."""
        @circuit_breaker("test_decorator_fail")
        async def test_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            await test_func()
        
        # Check that failure was recorded
        cb = circuit_breaker_manager.get_circuit_breaker("test_decorator_fail")
        assert cb.metrics.failure_count == 1
    
    @pytest.mark.asyncio
    async def test_decorator_with_config(self):
        """Test circuit breaker decorator with custom config."""
        config = CircuitBreakerConfig(failure_threshold=2)
        
        @circuit_breaker("test_decorator_config", config)
        async def test_func():
            return "success"
        
        result = await test_func()
        assert result == "success"
        
        # Check that circuit breaker was created with custom config
        cb = circuit_breaker_manager.get_circuit_breaker("test_decorator_config")
        assert cb.config.failure_threshold == 2
    
    @pytest.mark.asyncio
    async def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function name and docstring."""
        @circuit_breaker("test_metadata")
        async def test_func():
            """Test function docstring."""
            return "success"
        
        assert test_func.__name__ == "test_func"
        assert test_func.__doc__ == "Test function docstring."


@pytest.mark.asyncio
async def test_integration_circuit_breaker_with_real_failures():
    """Integration test with realistic failure scenarios."""
    config = CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout_seconds=1,
        minimum_request_volume=5
    )
    
    cb = CircuitBreaker("integration_test", config)
    
    # Simulate a service that fails intermittently
    call_count = 0
    
    async def unreliable_service():
        nonlocal call_count
        call_count += 1
        
        # Fail for first 5 calls, then succeed
        if call_count <= 5:
            raise ConnectionError("Service unavailable")
        return f"Success on call {call_count}"
    
    # Make calls that will cause circuit to open
    for i in range(3):
        with pytest.raises(ConnectionError):
            await cb.call(unreliable_service)
    
    # Circuit should be open now
    assert cb.get_state() == CircuitBreakerState.OPEN
    
    # Calls should be rejected
    with pytest.raises(CircuitBreakerOpenError):
        await cb.call(unreliable_service)
    
    # Wait for timeout
    await asyncio.sleep(1.1)
    
    # Next call should transition to half-open and succeed
    result = await cb.call(unreliable_service)
    assert "Success" in result
    assert cb.get_state() == CircuitBreakerState.HALF_OPEN
    
    # Another success should close the circuit
    result = await cb.call(unreliable_service)
    assert "Success" in result
    assert cb.get_state() == CircuitBreakerState.CLOSED
    
    # Verify metrics
    metrics = cb.get_metrics()
    assert metrics['failure_count'] == 3  # Initial failures
    assert metrics['success_count'] == 2  # Successful recoveries
    assert metrics['rejected_requests'] == 1  # One rejected call
"""
Unit tests for the ErrorHandler class and recovery strategies.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.langgraph_agents.error_recovery.error_handler import (
    ErrorHandler,
    RetryStrategy,
    RAGEnhancementStrategy,
    FallbackModelStrategy,
    RecoveryAction
)
from src.langgraph_agents.models.state import VideoGenerationState
from src.langgraph_agents.models.errors import WorkflowError, ErrorType, ErrorSeverity, RecoveryStrategy as RecoveryStrategyModel
from src.langgraph_agents.models.config import WorkflowConfig


class TestRetryStrategy:
    """Test cases for RetryStrategy."""
    
    @pytest.fixture
    def retry_config(self):
        """Create a retry strategy configuration."""
        return RecoveryStrategyModel(
            name="test_retry",
            error_types=[ErrorType.TRANSIENT, ErrorType.TIMEOUT],
            max_attempts=3,
            backoff_factor=2.0,
            timeout_seconds=60,
            escalation_threshold=3
        )
    
    @pytest.fixture
    def retry_strategy(self, retry_config):
        """Create a retry strategy instance."""
        return RetryStrategy(retry_config)
    
    def test_can_handle_transient_error(self, retry_strategy):
        """Test that retry strategy can handle transient errors."""
        error = WorkflowError(
            step="planning",
            error_type=ErrorType.TRANSIENT,
            message="Network timeout"
        )
        
        assert retry_strategy.can_handle(error) is True
    
    def test_cannot_handle_system_error(self, retry_strategy):
        """Test that retry strategy cannot handle system errors."""
        error = WorkflowError(
            step="planning",
            error_type=ErrorType.SYSTEM,
            message="Out of memory"
        )
        
        assert retry_strategy.can_handle(error) is False
    
    @pytest.mark.asyncio
    async def test_execute_recovery_success(self, retry_strategy):
        """Test successful retry recovery execution."""
        error = WorkflowError(
            step="planning",
            error_type=ErrorType.TRANSIENT,
            message="Network timeout"
        )
        
        state = VideoGenerationState(
            topic="Test topic",
            description="Test description"
        )
        
        context = {"attempt_count": 1}
        
        result = await retry_strategy.execute_recovery(error, state, context)
        
        assert result.success is True
        assert result.strategy_used == "test_retry"
        assert result.attempts_made == 2
        assert "backoff_delay" in result.recovery_data
        assert error.retry_count == 1
    
    @pytest.mark.asyncio
    async def test_execute_recovery_escalation(self, retry_strategy):
        """Test retry recovery escalation when max attempts reached."""
        error = WorkflowError(
            step="planning",
            error_type=ErrorType.TRANSIENT,
            message="Network timeout"
        )
        
        state = VideoGenerationState(
            topic="Test topic",
            description="Test description"
        )
        
        context = {"attempt_count": 3}  # At escalation threshold
        
        result = await retry_strategy.execute_recovery(error, state, context)
        
        assert result.success is False
        assert result.escalated is True
        assert "Max attempts reached" in result.escalation_reason
    
    def test_get_backoff_delay(self, retry_strategy):
        """Test backoff delay calculation."""
        # First attempt (attempt 0)
        delay = retry_strategy.get_backoff_delay(0)
        assert delay == 1.0  # 2^0 = 1
        
        # Second attempt (attempt 1)
        delay = retry_strategy.get_backoff_delay(1)
        assert delay == 2.0  # 2^1 = 2
        
        # Third attempt (attempt 2)
        delay = retry_strategy.get_backoff_delay(2)
        assert delay == 4.0  # 2^2 = 4
        
        # Large attempt should be capped at 60 seconds
        delay = retry_strategy.get_backoff_delay(10)
        assert delay == 60.0


class TestRAGEnhancementStrategy:
    """Test cases for RAGEnhancementStrategy."""
    
    @pytest.fixture
    def rag_config(self):
        """Create a RAG enhancement strategy configuration."""
        return RecoveryStrategyModel(
            name="test_rag",
            error_types=[ErrorType.CONTENT, ErrorType.MODEL],
            max_attempts=2,
            backoff_factor=1.5,
            timeout_seconds=120,
            use_rag=True,
            escalation_threshold=2
        )
    
    @pytest.fixture
    def mock_rag_service(self):
        """Create a mock RAG service."""
        service = Mock()
        service.get_enhanced_context = AsyncMock()
        return service
    
    @pytest.fixture
    def rag_strategy(self, rag_config, mock_rag_service):
        """Create a RAG enhancement strategy instance."""
        return RAGEnhancementStrategy(rag_config, mock_rag_service)
    
    def test_can_handle_content_error(self, rag_strategy):
        """Test that RAG strategy can handle content errors."""
        error = WorkflowError(
            step="code_generation",
            error_type=ErrorType.CONTENT,
            message="Invalid scene description"
        )
        
        assert rag_strategy.can_handle(error) is True
    
    def test_cannot_handle_without_rag_service(self, rag_config):
        """Test that RAG strategy cannot handle errors without RAG service."""
        strategy = RAGEnhancementStrategy(rag_config, None)
        
        error = WorkflowError(
            step="code_generation",
            error_type=ErrorType.CONTENT,
            message="Invalid scene description"
        )
        
        assert strategy.can_handle(error) is False
    
    @pytest.mark.asyncio
    async def test_execute_recovery_success(self, rag_strategy, mock_rag_service):
        """Test successful RAG enhancement recovery."""
        # Setup mock RAG service
        mock_rag_service.get_enhanced_context.return_value = {
            "enhanced_context": "Additional context from RAG",
            "relevant_examples": ["example1", "example2"]
        }
        
        error = WorkflowError(
            step="code_generation",
            error_type=ErrorType.CONTENT,
            message="Invalid scene description",
            scene_number=1
        )
        
        state = VideoGenerationState(
            topic="Test topic",
            description="Test description",
            scene_implementations={1: "Test scene implementation"}
        )
        
        context = {"attempt_count": 1}
        
        result = await rag_strategy.execute_recovery(error, state, context)
        
        assert result.success is True
        assert result.strategy_used == "test_rag"
        assert "enhanced_context_keys" in result.recovery_data
        
        # Check that RAG context was added to state
        assert f"error_recovery_{error.get_error_code()}" in state.rag_context
        assert "last_rag_enhancement" in state.rag_context
    
    @pytest.mark.asyncio
    async def test_execute_recovery_rag_failure(self, rag_strategy, mock_rag_service):
        """Test RAG enhancement recovery when RAG service fails."""
        # Setup mock RAG service to raise exception
        mock_rag_service.get_enhanced_context.side_effect = Exception("RAG service error")
        
        error = WorkflowError(
            step="code_generation",
            error_type=ErrorType.CONTENT,
            message="Invalid scene description"
        )
        
        state = VideoGenerationState(
            topic="Test topic",
            description="Test description"
        )
        
        context = {"attempt_count": 1}
        
        result = await rag_strategy.execute_recovery(error, state, context)
        
        assert result.success is False
        assert result.new_error is not None
        assert "RAG enhancement failed" in result.new_error.message
    
    def test_build_rag_query(self, rag_strategy):
        """Test RAG query building."""
        error = WorkflowError(
            step="code_generation",
            error_type=ErrorType.CONTENT,
            message="Invalid scene description",
            scene_number=1
        )
        
        state = VideoGenerationState(
            topic="Python basics",
            description="Introduction to Python programming",
            scene_implementations={1: "Create a simple Python function that adds two numbers"}
        )
        
        query = rag_strategy._build_rag_query(error, state)
        
        assert "Error in code_generation" in query
        assert "Invalid scene description" in query
        assert "Python basics" in query
        assert "Introduction to Python programming" in query
        assert "Create a simple Python function" in query


class TestFallbackModelStrategy:
    """Test cases for FallbackModelStrategy."""
    
    @pytest.fixture
    def fallback_config(self):
        """Create a fallback model strategy configuration."""
        return RecoveryStrategyModel(
            name="test_fallback",
            error_types=[ErrorType.MODEL, ErrorType.RATE_LIMIT],
            max_attempts=3,
            backoff_factor=1.0,
            timeout_seconds=90,
            use_fallback_model=True,
            fallback_models=["openai/gpt-3.5-turbo", "anthropic/claude-3-haiku"],
            escalation_threshold=3
        )
    
    @pytest.fixture
    def fallback_strategy(self, fallback_config):
        """Create a fallback model strategy instance."""
        return FallbackModelStrategy(fallback_config)
    
    def test_can_handle_model_error(self, fallback_strategy):
        """Test that fallback strategy can handle model errors."""
        error = WorkflowError(
            step="planning",
            error_type=ErrorType.MODEL,
            message="Model API error"
        )
        
        assert fallback_strategy.can_handle(error) is True
    
    def test_cannot_handle_without_fallback_models(self):
        """Test that fallback strategy cannot handle errors without fallback models."""
        config = RecoveryStrategyModel(
            name="test_fallback",
            error_types=[ErrorType.MODEL],
            use_fallback_model=True,
            fallback_models=[]  # No fallback models
        )
        
        strategy = FallbackModelStrategy(config)
        
        error = WorkflowError(
            step="planning",
            error_type=ErrorType.MODEL,
            message="Model API error"
        )
        
        assert strategy.can_handle(error) is False
    
    @pytest.mark.asyncio
    async def test_execute_recovery_success(self, fallback_strategy):
        """Test successful fallback model recovery."""
        from src.langgraph_agents.models.config import ModelConfig
        
        error = WorkflowError(
            step="planning",
            error_type=ErrorType.MODEL,
            message="Model API error"
        )
        
        config = WorkflowConfig(
            planner_model=ModelConfig(provider="openai", model_name="gpt-4"),
            code_model=ModelConfig(provider="openai", model_name="gpt-4")
        )
        
        state = VideoGenerationState(
            topic="Test topic",
            description="Test description",
            config=config
        )
        
        context = {"attempt_count": 1}
        
        result = await fallback_strategy.execute_recovery(error, state, context)
        
        assert result.success is True
        assert result.strategy_used == "test_fallback"
        assert "fallback_model" in result.recovery_data
        assert result.recovery_data["fallback_model"] == "openai/gpt-3.5-turbo"
        
        # Check that model was updated in state
        assert state.config.planner_model.model_name == "gpt-3.5-turbo"
    
    @pytest.mark.asyncio
    async def test_execute_recovery_no_more_fallbacks(self, fallback_strategy):
        """Test fallback model recovery when no more fallbacks available."""
        error = WorkflowError(
            step="planning",
            error_type=ErrorType.MODEL,
            message="Model API error"
        )
        
        state = VideoGenerationState(
            topic="Test topic",
            description="Test description"
        )
        
        # Simulate all fallback models have been used
        context = {
            "attempt_count": 1,
            "used_fallback_models": {"openai/gpt-4", "openai/gpt-3.5-turbo", "anthropic/claude-3-haiku"}
        }
        
        result = await fallback_strategy.execute_recovery(error, state, context)
        
        assert result.success is False
        assert result.escalated is True
        assert "No more fallback models available" in result.escalation_reason
    
    def test_get_current_model_planning(self, fallback_strategy):
        """Test getting current model for planning step."""
        from src.langgraph_agents.models.config import ModelConfig
        
        error = WorkflowError(
            step="planning",
            error_type=ErrorType.MODEL,
            message="Model API error"
        )
        
        config = WorkflowConfig(
            planner_model=ModelConfig(provider="openai", model_name="gpt-4"),
            code_model=ModelConfig(provider="openai", model_name="gpt-3.5-turbo")
        )
        
        state = VideoGenerationState(
            topic="Test topic",
            description="Test description",
            config=config
        )
        
        current_model = fallback_strategy._get_current_model(error, state)
        assert current_model == "openai/gpt-4"
    
    def test_get_current_model_code_generation(self, fallback_strategy):
        """Test getting current model for code generation step."""
        from src.langgraph_agents.models.config import ModelConfig
        
        error = WorkflowError(
            step="code_generation",
            error_type=ErrorType.MODEL,
            message="Model API error"
        )
        
        config = WorkflowConfig(
            planner_model=ModelConfig(provider="openai", model_name="gpt-4"),
            code_model=ModelConfig(provider="openai", model_name="gpt-3.5-turbo")
        )
        
        state = VideoGenerationState(
            topic="Test topic",
            description="Test description",
            config=config
        )
        
        current_model = fallback_strategy._get_current_model(error, state)
        assert current_model == "openai/gpt-3.5-turbo"


class TestErrorHandler:
    """Test cases for ErrorHandler."""
    
    @pytest.fixture
    def workflow_config(self):
        """Create a workflow configuration."""
        from src.langgraph_agents.models.config import ModelConfig
        return WorkflowConfig(
            max_retries=3,
            use_rag=True,
            planner_model=ModelConfig(provider="openai", model_name="gpt-4"),
            code_model=ModelConfig(provider="openai", model_name="gpt-4")
        )
    
    @pytest.fixture
    def mock_rag_service(self):
        """Create a mock RAG service."""
        service = Mock()
        service.get_enhanced_context = AsyncMock()
        return service
    
    @pytest.fixture
    def error_handler(self, workflow_config, mock_rag_service):
        """Create an error handler instance."""
        return ErrorHandler(workflow_config, mock_rag_service)
    
    def test_initialization(self, error_handler):
        """Test error handler initialization."""
        assert len(error_handler.strategies) >= 2  # At least retry and RAG strategies
        assert error_handler.recovery_contexts == {}
    
    @pytest.mark.asyncio
    async def test_handle_transient_error(self, error_handler):
        """Test handling a transient error."""
        error = WorkflowError(
            step="planning",
            error_type=ErrorType.TRANSIENT,
            message="Network timeout"
        )
        
        state = VideoGenerationState(
            topic="Test topic",
            description="Test description"
        )
        
        result = await error_handler.handle_error(error, state)
        
        assert result.success is True
        assert result.strategy_used == "retry_with_backoff"
        assert error.get_error_code() in error_handler.recovery_contexts
    
    @pytest.mark.asyncio
    async def test_handle_content_error_with_rag(self, error_handler, mock_rag_service):
        """Test handling a content error with RAG enhancement."""
        # Setup mock RAG service
        mock_rag_service.get_enhanced_context.return_value = {
            "enhanced_context": "Additional context from RAG"
        }
        
        error = WorkflowError(
            step="code_generation",
            error_type=ErrorType.CONTENT,
            message="Invalid scene description"
        )
        
        state = VideoGenerationState(
            topic="Test topic",
            description="Test description"
        )
        
        result = await error_handler.handle_error(error, state)
        
        assert result.success is True
        assert result.strategy_used == "rag_enhancement"
        assert "enhanced_context_keys" in result.recovery_data
    
    @pytest.mark.asyncio
    async def test_handle_unknown_error_type(self, error_handler):
        """Test handling an error type with no applicable strategy."""
        error = WorkflowError(
            step="planning",
            error_type=ErrorType.SECURITY,  # No strategy handles security errors
            message="Security validation failed"
        )
        
        state = VideoGenerationState(
            topic="Test topic",
            description="Test description"
        )
        
        result = await error_handler.handle_error(error, state)
        
        assert result.success is False
        assert result.strategy_used == "none"
        assert result.escalated is True
        assert "No applicable recovery strategy found" in result.escalation_reason
    
    @pytest.mark.asyncio
    async def test_handle_error_with_exception(self, error_handler):
        """Test handling an error when strategy execution raises exception."""
        error = WorkflowError(
            step="planning",
            error_type=ErrorType.TRANSIENT,
            message="Network timeout"
        )
        
        state = VideoGenerationState(
            topic="Test topic",
            description="Test description"
        )
        
        # Mock strategy to raise exception
        with patch.object(error_handler.strategies[0], 'execute_recovery', side_effect=Exception("Strategy failed")):
            result = await error_handler.handle_error(error, state)
        
        assert result.success is False
        assert result.new_error is not None
        assert "Recovery strategy failed" in result.new_error.message
    
    def test_find_recovery_strategy(self, error_handler):
        """Test finding appropriate recovery strategy."""
        # Test transient error
        transient_error = WorkflowError(
            step="planning",
            error_type=ErrorType.TRANSIENT,
            message="Network timeout"
        )
        
        strategy = error_handler._find_recovery_strategy(transient_error)
        assert strategy is not None
        assert strategy.config.name == "retry_with_backoff"
        
        # Test content error
        content_error = WorkflowError(
            step="code_generation",
            error_type=ErrorType.CONTENT,
            message="Invalid scene description"
        )
        
        strategy = error_handler._find_recovery_strategy(content_error)
        assert strategy is not None
        # Should find RAG strategy if available
        
        # Test unsupported error
        security_error = WorkflowError(
            step="planning",
            error_type=ErrorType.SECURITY,
            message="Security validation failed"
        )
        
        strategy = error_handler._find_recovery_strategy(security_error)
        assert strategy is None
    
    def test_get_recovery_statistics_empty(self, error_handler):
        """Test getting recovery statistics when no recoveries have occurred."""
        stats = error_handler.get_recovery_statistics()
        
        assert stats['total_recoveries'] == 0
        assert stats['success_rate'] == 0.0
        assert stats['average_attempts'] == 0.0
        assert stats['strategy_usage'] == {}
    
    def test_clear_recovery_context(self, error_handler):
        """Test clearing recovery context."""
        # Add some context
        error_code = "test_error_123"
        error_handler.recovery_contexts[error_code] = {"attempt_count": 2}
        
        # Clear specific context
        error_handler.clear_recovery_context(error_code)
        assert error_code not in error_handler.recovery_contexts
    
    def test_clear_all_recovery_contexts(self, error_handler):
        """Test clearing all recovery contexts."""
        # Add some contexts
        error_handler.recovery_contexts["error1"] = {"attempt_count": 1}
        error_handler.recovery_contexts["error2"] = {"attempt_count": 2}
        
        # Clear all contexts
        error_handler.clear_all_recovery_contexts()
        assert len(error_handler.recovery_contexts) == 0


@pytest.mark.asyncio
async def test_integration_error_handler_with_multiple_strategies():
    """Integration test for error handler with multiple recovery strategies."""
    config = WorkflowConfig(
        max_retries=2,
        use_rag=True,
        planner_model="openai/gpt-4"
    )
    
    mock_rag_service = Mock()
    mock_rag_service.get_enhanced_context = AsyncMock(return_value={"context": "enhanced"})
    
    error_handler = ErrorHandler(config, mock_rag_service)
    
    # Test sequence of errors with different strategies
    errors = [
        WorkflowError(step="planning", error_type=ErrorType.TRANSIENT, message="Network timeout"),
        WorkflowError(step="code_generation", error_type=ErrorType.CONTENT, message="Invalid content"),
        WorkflowError(step="planning", error_type=ErrorType.MODEL, message="Model API error")
    ]
    
    state = VideoGenerationState(
        topic="Test topic",
        description="Test description"
    )
    
    results = []
    for error in errors:
        result = await error_handler.handle_error(error, state)
        results.append(result)
    
    # Check that different strategies were used
    strategies_used = [result.strategy_used for result in results]
    assert "retry_with_backoff" in strategies_used
    assert "rag_enhancement" in strategies_used
    
    # Check that all recoveries were tracked
    assert len(error_handler.recovery_contexts) == len(errors)
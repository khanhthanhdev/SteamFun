"""
Unit tests for error_handler_node function.

Tests the error handler node implementation following LangGraph patterns
with centralized error handling and human loop escalation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.langgraph_agents.nodes.error_handler_node import (
    error_handler_node,
    _prepare_human_intervention_request,
    _determine_intervention_type,
    _generate_suggested_actions,
    _calculate_priority,
    _estimate_resolution_time
)
from src.langgraph_agents.models.state import VideoGenerationState
from src.langgraph_agents.models.config import WorkflowConfig, ModelConfig
from src.langgraph_agents.models.errors import WorkflowError, ErrorType, ErrorSeverity, ErrorRecoveryResult


@pytest.fixture
def sample_state_with_errors():
    """Create a sample VideoGenerationState with various errors."""
    config = WorkflowConfig(max_retries=3)
    
    state = VideoGenerationState(
        topic="Python basics",
        description="Introduction to Python programming concepts",
        session_id="test-session-123",
        config=config,
        scene_outline="Scene 1: Introduction\nScene 2: Variables",
        scene_implementations={
            1: "Show Python logo and introduction text",
            2: "Demonstrate variable assignment and types"
        },
        generated_code={
            1: "from manim import *\nclass Scene1(Scene): pass"
        }
    )
    
    # Add various types of errors
    state.errors = [
        WorkflowError(
            step="code_generation",
            error_type=ErrorType.MODEL,
            message="Model API rate limit exceeded",
            severity=ErrorSeverity.MEDIUM,
            scene_number=2,
            retry_count=1
        ),
        WorkflowError(
            step="rendering",
            error_type=ErrorType.SYSTEM,
            message="Insufficient memory for rendering",
            severity=ErrorSeverity.HIGH,
            scene_number=1,
            retry_count=2
        ),
        WorkflowError(
            step="planning",
            error_type=ErrorType.CONTENT,
            message="Scene description too vague",
            severity=ErrorSeverity.LOW,
            retry_count=0
        )
    ]
    
    return state


@pytest.fixture
def mock_error_handler():
    """Create a mock ErrorHandler."""
    handler = MagicMock()
    handler.handle_error = AsyncMock()
    return handler


class TestErrorHandlerNode:
    """Test cases for error_handler_node function."""
    
    @pytest.mark.asyncio
    async def test_error_handler_node_no_errors(self):
        """Test error handler node with no errors to handle."""
        config = WorkflowConfig()
        state = VideoGenerationState(
            topic="Test topic",
            description="Test description",
            session_id="test-session",
            config=config
        )
        
        result_state = await error_handler_node(state)
        
        # Verify state updates
        assert result_state.current_step == "error_handling"
        assert len(result_state.errors) == 0
        assert len(result_state.escalated_errors) == 0
        
        # Verify execution trace shows skipped
        skip_traces = [trace for trace in result_state.execution_trace 
                      if trace["data"].get("action") == "skipped"]
        assert len(skip_traces) == 1
        assert skip_traces[0]["data"]["reason"] == "no_errors_to_handle"
    
    @pytest.mark.asyncio
    async def test_error_handler_node_successful_recovery(self, sample_state_with_errors, mock_error_handler):
        """Test error handler node with successful error recovery."""
        # Setup mock to return successful recovery for all errors
        successful_recovery = ErrorRecoveryResult(
            success=True,
            strategy_used="retry_with_backoff",
            attempts_made=2,
            time_taken=5.0,
            escalated=False
        )
        mock_error_handler.handle_error.return_value = successful_recovery
        
        with patch('src.langgraph_agents.nodes.error_handler_node.ErrorHandler', return_value=mock_error_handler):
            result_state = await error_handler_node(sample_state_with_errors)
        
        # Verify all errors were recovered
        assert len(result_state.errors) == 0  # All errors should be removed
        assert len(result_state.escalated_errors) == 0
        assert result_state.pending_human_input is None
        
        # Verify error handler was called for each error
        assert mock_error_handler.handle_error.call_count == 3
        
        # Verify completion trace
        completion_traces = [trace for trace in result_state.execution_trace 
                           if trace["data"].get("action") == "completed"]
        assert len(completion_traces) == 1
        assert completion_traces[0]["data"]["errors_recovered"] == 3
        assert completion_traces[0]["data"]["errors_escalated"] == 0
    
    @pytest.mark.asyncio
    async def test_error_handler_node_escalated_errors(self, sample_state_with_errors, mock_error_handler):
        """Test error handler node with errors that need escalation."""
        # Setup mock to return escalated recovery for some errors
        def mock_handle_error(error, state):
            if error.severity == ErrorSeverity.HIGH:
                return ErrorRecoveryResult(
                    success=False,
                    strategy_used="retry_with_backoff",
                    attempts_made=3,
                    time_taken=10.0,
                    escalated=True,
                    escalation_reason="Max attempts exceeded"
                )
            else:
                return ErrorRecoveryResult(
                    success=True,
                    strategy_used="retry_with_backoff",
                    attempts_made=1,
                    time_taken=3.0,
                    escalated=False
                )
        
        mock_error_handler.handle_error.side_effect = mock_handle_error
        
        with patch('src.langgraph_agents.nodes.error_handler_node.ErrorHandler', return_value=mock_error_handler):
            result_state = await error_handler_node(sample_state_with_errors)
        
        # Verify some errors were recovered, some escalated
        assert len(result_state.errors) == 0  # All errors removed from main list
        assert len(result_state.escalated_errors) == 1  # One error escalated
        assert result_state.pending_human_input is not None
        
        # Verify escalated error details
        escalated_error = result_state.escalated_errors[0]
        assert escalated_error["escalation_reason"] == "Max attempts exceeded"
        assert escalated_error["requires_human_intervention"] is True
        
        # Verify completion trace
        completion_traces = [trace for trace in result_state.execution_trace 
                           if trace["data"].get("action") == "completed"]
        assert len(completion_traces) == 1
        assert completion_traces[0]["data"]["errors_recovered"] == 2
        assert completion_traces[0]["data"]["errors_escalated"] == 1
        assert completion_traces[0]["data"]["human_intervention_required"] is True
    
    @pytest.mark.asyncio
    async def test_error_handler_node_recovery_generates_new_error(self, sample_state_with_errors, mock_error_handler):
        """Test error handler node when recovery generates new errors."""
        # Setup mock to return recovery with new error
        new_error = WorkflowError(
            step="error_handling",
            error_type=ErrorType.SYSTEM,
            message="Recovery process failed",
            severity=ErrorSeverity.MEDIUM
        )
        
        recovery_with_new_error = ErrorRecoveryResult(
            success=False,
            strategy_used="fallback_model",
            attempts_made=1,
            time_taken=8.0,
            new_error=new_error,
            escalated=False
        )
        
        mock_error_handler.handle_error.return_value = recovery_with_new_error
        
        with patch('src.langgraph_agents.nodes.error_handler_node.ErrorHandler', return_value=mock_error_handler):
            result_state = await error_handler_node(sample_state_with_errors)
        
        # Verify new errors were added
        assert len(result_state.errors) >= 3  # Original errors plus new ones
        
        # Check that new error was added
        new_errors = [e for e in result_state.errors if e.message == "Recovery process failed"]
        assert len(new_errors) >= 1
    
    @pytest.mark.asyncio
    async def test_error_handler_node_recovery_exception(self, sample_state_with_errors, mock_error_handler):
        """Test error handler node when recovery process throws exception."""
        # Setup mock to throw exception
        mock_error_handler.handle_error.side_effect = Exception("Recovery system failure")
        
        with patch('src.langgraph_agents.nodes.error_handler_node.ErrorHandler', return_value=mock_error_handler):
            result_state = await error_handler_node(sample_state_with_errors)
        
        # Verify original errors remain and new error added for recovery failure
        assert len(result_state.errors) >= 3  # Original errors plus recovery failure errors
        
        # Check for recovery failure errors
        recovery_errors = [e for e in result_state.errors if "Error recovery failed" in e.message]
        assert len(recovery_errors) >= 1
    
    @pytest.mark.asyncio
    async def test_error_handler_node_already_resolved_errors(self, sample_state_with_errors, mock_error_handler):
        """Test error handler node with already resolved errors."""
        # Mark one error as already resolved
        sample_state_with_errors.errors[0].mark_resolved("previous_strategy")
        
        mock_error_handler.handle_error.return_value = ErrorRecoveryResult(
            success=True,
            strategy_used="retry_with_backoff",
            attempts_made=1,
            time_taken=2.0,
            escalated=False
        )
        
        with patch('src.langgraph_agents.nodes.error_handler_node.ErrorHandler', return_value=mock_error_handler):
            result_state = await error_handler_node(sample_state_with_errors)
        
        # Verify only unresolved errors were processed
        assert mock_error_handler.handle_error.call_count == 2  # Only 2 unresolved errors
    
    @pytest.mark.asyncio
    async def test_error_handler_node_max_retries_escalation(self, sample_state_with_errors, mock_error_handler):
        """Test error handler node escalation due to max retries."""
        # Set one error to max retries
        sample_state_with_errors.errors[1].retry_count = 3  # Exceeds max_retries
        
        # Setup mock to return failed recovery (not escalated by strategy)
        failed_recovery = ErrorRecoveryResult(
            success=False,
            strategy_used="retry_with_backoff",
            attempts_made=3,
            time_taken=15.0,
            escalated=False  # Strategy doesn't escalate, but node should due to max retries
        )
        
        successful_recovery = ErrorRecoveryResult(
            success=True,
            strategy_used="retry_with_backoff",
            attempts_made=1,
            time_taken=3.0,
            escalated=False
        )
        
        mock_error_handler.handle_error.side_effect = [successful_recovery, failed_recovery, successful_recovery]
        
        with patch('src.langgraph_agents.nodes.error_handler_node.ErrorHandler', return_value=mock_error_handler):
            result_state = await error_handler_node(sample_state_with_errors)
        
        # Verify error was escalated due to max retries
        assert len(result_state.escalated_errors) == 1
        escalated_error = result_state.escalated_errors[0]
        assert "Max recovery attempts exceeded" in escalated_error["escalation_reason"]
    
    @pytest.mark.asyncio
    async def test_error_handler_node_system_error(self, sample_state_with_errors):
        """Test error handler node with system error during initialization."""
        with patch('src.langgraph_agents.nodes.error_handler_node.ErrorHandler', side_effect=Exception("Handler init failed")):
            result_state = await error_handler_node(sample_state_with_errors)
        
        # Verify system error was handled
        system_errors = [e for e in result_state.errors if e.error_type == ErrorType.SYSTEM and "Error handler failed" in e.message]
        assert len(system_errors) == 1
        assert system_errors[0].severity == ErrorSeverity.CRITICAL
        
        # Verify failure trace was added
        failure_traces = [trace for trace in result_state.execution_trace 
                         if trace["data"].get("action") == "failed"]
        assert len(failure_traces) == 1
    
    @pytest.mark.asyncio
    async def test_error_handler_node_with_metrics(self, sample_state_with_errors, mock_error_handler):
        """Test error handler node with metrics collection."""
        from src.langgraph_agents.models.metrics import PerformanceMetrics
        
        # Add metrics to state
        sample_state_with_errors.metrics = PerformanceMetrics(session_id=sample_state_with_errors.session_id)
        
        # Setup mixed recovery results
        def mock_handle_error(error, state):
            if error.error_type == ErrorType.MODEL:
                return ErrorRecoveryResult(
                    success=True,
                    strategy_used="fallback_model",
                    attempts_made=1,
                    time_taken=5.0,
                    escalated=False
                )
            else:
                return ErrorRecoveryResult(
                    success=False,
                    strategy_used="retry_with_backoff",
                    attempts_made=3,
                    time_taken=10.0,
                    escalated=True,
                    escalation_reason="Max attempts exceeded"
                )
        
        mock_error_handler.handle_error.side_effect = mock_handle_error
        
        with patch('src.langgraph_agents.nodes.error_handler_node.ErrorHandler', return_value=mock_error_handler):
            result_state = await error_handler_node(sample_state_with_errors)
        
        # Verify metrics were collected (would be added to state.metrics)
        # The actual metrics integration would depend on the PerformanceMetrics implementation


class TestErrorHandlerNodeHelpers:
    """Test cases for error handler node helper functions."""
    
    def test_prepare_human_intervention_request(self, sample_state_with_errors):
        """Test preparing human intervention request."""
        escalated_errors = sample_state_with_errors.errors[:2]  # First two errors
        
        request = _prepare_human_intervention_request(sample_state_with_errors, escalated_errors)
        
        assert request["request_type"] == "error_escalation"
        assert request["intervention_type"] in ["immediate_intervention", "technical_intervention", "general_review"]
        assert request["priority"] in ["low", "medium", "high", "critical"]
        assert "error_summary" in request
        assert "escalated_errors" in request
        assert "suggested_actions" in request
        assert "workflow_context" in request
        assert isinstance(request["suggested_actions"], list)
        assert len(request["suggested_actions"]) > 0
    
    def test_determine_intervention_type(self):
        """Test determining intervention type based on errors."""
        # Critical error
        critical_error = WorkflowError(
            step="planning",
            error_type=ErrorType.SYSTEM,
            message="Critical failure",
            severity=ErrorSeverity.CRITICAL
        )
        assert _determine_intervention_type([critical_error]) == "immediate_intervention"
        
        # System error
        system_error = WorkflowError(
            step="rendering",
            error_type=ErrorType.SYSTEM,
            message="System failure",
            severity=ErrorSeverity.HIGH
        )
        assert _determine_intervention_type([system_error]) == "technical_intervention"
        
        # Configuration error
        config_error = WorkflowError(
            step="planning",
            error_type=ErrorType.CONFIGURATION,
            message="Config issue",
            severity=ErrorSeverity.MEDIUM
        )
        assert _determine_intervention_type([config_error]) == "configuration_review"
        
        # Content error
        content_error = WorkflowError(
            step="code_generation",
            error_type=ErrorType.CONTENT,
            message="Content issue",
            severity=ErrorSeverity.MEDIUM
        )
        assert _determine_intervention_type([content_error]) == "content_review"
        
        # Security error
        security_error = WorkflowError(
            step="planning",
            error_type=ErrorType.SECURITY,
            message="Security issue",
            severity=ErrorSeverity.HIGH
        )
        assert _determine_intervention_type([security_error]) == "security_review"
        
        # General error
        general_error = WorkflowError(
            step="planning",
            error_type=ErrorType.UNKNOWN,
            message="Unknown issue",
            severity=ErrorSeverity.MEDIUM
        )
        assert _determine_intervention_type([general_error]) == "general_review"
    
    def test_generate_suggested_actions(self, sample_state_with_errors):
        """Test generating suggested actions for different error types."""
        # Configuration error
        config_error = WorkflowError(
            step="planning",
            error_type=ErrorType.CONFIGURATION,
            message="Config issue",
            severity=ErrorSeverity.MEDIUM
        )
        actions = _generate_suggested_actions([config_error], sample_state_with_errors)
        assert any("configuration" in action.lower() for action in actions)
        assert any("api key" in action.lower() for action in actions)
        
        # Content error
        content_error = WorkflowError(
            step="code_generation",
            error_type=ErrorType.CONTENT,
            message="Content issue",
            severity=ErrorSeverity.MEDIUM
        )
        actions = _generate_suggested_actions([content_error], sample_state_with_errors)
        assert any("scene" in action.lower() for action in actions)
        
        # System error
        system_error = WorkflowError(
            step="rendering",
            error_type=ErrorType.SYSTEM,
            message="System issue",
            severity=ErrorSeverity.HIGH
        )
        actions = _generate_suggested_actions([system_error], sample_state_with_errors)
        assert any("system" in action.lower() or "resource" in action.lower() for action in actions)
    
    def test_calculate_priority(self):
        """Test calculating priority based on error severity."""
        critical_error = WorkflowError(
            step="planning",
            error_type=ErrorType.SYSTEM,
            message="Critical failure",
            severity=ErrorSeverity.CRITICAL
        )
        assert _calculate_priority([critical_error]) == "critical"
        
        high_error = WorkflowError(
            step="rendering",
            error_type=ErrorType.SYSTEM,
            message="High failure",
            severity=ErrorSeverity.HIGH
        )
        assert _calculate_priority([high_error]) == "high"
        
        medium_error = WorkflowError(
            step="code_generation",
            error_type=ErrorType.CONTENT,
            message="Medium issue",
            severity=ErrorSeverity.MEDIUM
        )
        assert _calculate_priority([medium_error]) == "medium"
        
        low_error = WorkflowError(
            step="planning",
            error_type=ErrorType.CONTENT,
            message="Low issue",
            severity=ErrorSeverity.LOW
        )
        assert _calculate_priority([low_error]) == "low"
    
    def test_estimate_resolution_time(self):
        """Test estimating resolution time based on error types."""
        critical_error = WorkflowError(
            step="planning",
            error_type=ErrorType.SYSTEM,
            message="Critical failure",
            severity=ErrorSeverity.CRITICAL
        )
        assert _estimate_resolution_time([critical_error]) == "immediate"
        
        system_error = WorkflowError(
            step="rendering",
            error_type=ErrorType.SYSTEM,
            message="System failure",
            severity=ErrorSeverity.HIGH
        )
        assert _estimate_resolution_time([system_error]) == "30-60 minutes"
        
        config_error = WorkflowError(
            step="planning",
            error_type=ErrorType.CONFIGURATION,
            message="Config issue",
            severity=ErrorSeverity.MEDIUM
        )
        assert _estimate_resolution_time([config_error]) == "5-15 minutes"
        
        content_error = WorkflowError(
            step="code_generation",
            error_type=ErrorType.CONTENT,
            message="Content issue",
            severity=ErrorSeverity.MEDIUM
        )
        assert _estimate_resolution_time([content_error]) == "15-30 minutes"


@pytest.mark.asyncio
async def test_error_handler_node_integration():
    """Integration test for error handler node with realistic error scenarios."""
    # Create realistic state with various errors
    config = WorkflowConfig(max_retries=2)
    
    state = VideoGenerationState(
        topic="Advanced Python Concepts",
        description="Deep dive into Python decorators, generators, and metaclasses",
        session_id="integration-test-error-789",
        config=config,
        scene_outline="Scene 1: Decorators\nScene 2: Generators\nScene 3: Metaclasses",
        scene_implementations={
            1: "Explain decorator syntax and use cases with examples",
            2: "Demonstrate generator functions and yield keyword",
            3: "Show metaclass creation and customization"
        },
        generated_code={
            1: "from manim import *\nclass DecoratorsScene(Scene): pass",
            2: "from manim import *\nclass GeneratorsScene(Scene): pass"
        }
    )
    
    # Add realistic errors
    state.errors = [
        WorkflowError(
            step="code_generation",
            error_type=ErrorType.MODEL,
            message="Model timeout during code generation for complex metaclass scene",
            severity=ErrorSeverity.MEDIUM,
            scene_number=3,
            retry_count=1,
            context={"model": "anthropic/claude-3.5-sonnet", "timeout": 30}
        ),
        WorkflowError(
            step="rendering",
            error_type=ErrorType.RENDERING,
            message="Manim rendering failed due to syntax error in generated code",
            severity=ErrorSeverity.HIGH,
            scene_number=2,
            retry_count=2,
            context={"syntax_error": "invalid Python syntax on line 15"}
        ),
        WorkflowError(
            step="planning",
            error_type=ErrorType.CONTENT,
            message="Scene description for metaclasses is too abstract for visualization",
            severity=ErrorSeverity.LOW,
            retry_count=0,
            context={"scene": 3, "complexity": "high"}
        )
    ]
    
    # Mock the error handler with realistic recovery scenarios
    mock_handler = MagicMock()
    
    def realistic_error_handling(error, state):
        if error.error_type == ErrorType.MODEL and error.retry_count < 2:
            # Model errors can often be recovered with retry
            return ErrorRecoveryResult(
                success=True,
                strategy_used="retry_with_backoff",
                attempts_made=error.retry_count + 1,
                time_taken=8.5,
                escalated=False,
                recovery_data={"backoff_delay": 4.0}
            )
        elif error.error_type == ErrorType.RENDERING and error.retry_count >= 2:
            # Rendering errors with high retry count need escalation
            return ErrorRecoveryResult(
                success=False,
                strategy_used="code_fix_attempt",
                attempts_made=3,
                time_taken=25.0,
                escalated=True,
                escalation_reason="Code fixing attempts exhausted, requires human review"
            )
        elif error.error_type == ErrorType.CONTENT:
            # Content errors can be enhanced with RAG
            return ErrorRecoveryResult(
                success=True,
                strategy_used="rag_enhancement",
                attempts_made=1,
                time_taken=12.0,
                escalated=False,
                recovery_data={"enhanced_context": "Added visualization examples for abstract concepts"}
            )
        else:
            # Default recovery attempt
            return ErrorRecoveryResult(
                success=False,
                strategy_used="general_retry",
                attempts_made=1,
                time_taken=5.0,
                escalated=False
            )
    
    mock_handler.handle_error = AsyncMock(side_effect=realistic_error_handling)
    
    with patch('src.langgraph_agents.nodes.error_handler_node.ErrorHandler', return_value=mock_handler):
        result_state = await error_handler_node(state)
    
    # Verify comprehensive results
    assert result_state.current_step == "error_handling"
    
    # Verify error processing results
    assert len(result_state.errors) == 0  # All errors should be processed
    assert len(result_state.escalated_errors) == 1  # One error escalated (rendering)
    
    # Verify escalated error details
    escalated_error = result_state.escalated_errors[0]
    assert "rendering" in escalated_error["error_code"].lower()
    assert escalated_error["requires_human_intervention"] is True
    assert "Code fixing attempts exhausted" in escalated_error["escalation_reason"]
    
    # Verify human intervention request
    assert result_state.pending_human_input is not None
    human_request = result_state.pending_human_input
    assert human_request["request_type"] == "error_escalation"
    assert human_request["intervention_type"] in ["technical_intervention", "immediate_intervention"]
    assert human_request["priority"] in ["high", "critical"]
    assert len(human_request["suggested_actions"]) > 0
    
    # Verify execution trace
    assert len(result_state.execution_trace) >= 2
    completion_traces = [trace for trace in result_state.execution_trace 
                        if trace["data"].get("action") == "completed"]
    assert len(completion_traces) == 1
    
    completion_data = completion_traces[0]["data"]
    assert completion_data["errors_processed"] == 3
    assert completion_data["errors_recovered"] == 2  # Model and content errors recovered
    assert completion_data["errors_escalated"] == 1  # Rendering error escalated
    assert completion_data["human_intervention_required"] is True
    assert completion_data["remaining_errors"] == 0
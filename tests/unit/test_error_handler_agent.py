"""
Unit tests for ErrorHandlerAgent.
Tests error classification, recovery strategies, and escalation logic.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List

from src.langgraph_agents.agents.error_handler_agent import ErrorHandlerAgent
from src.langgraph_agents.state import VideoGenerationState, AgentConfig, AgentError, RecoveryStrategy
from langgraph.types import Command


class TestErrorHandlerAgent:
    """Test suite for ErrorHandlerAgent functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock agent configuration for ErrorHandlerAgent."""
        return AgentConfig(
            name="error_handler_agent",
            model_config={},
            tools=["error_analysis_tool", "recovery_tool"],
            max_retries=3,
            timeout_seconds=300,
            enable_human_loop=True
        )
    
    @pytest.fixture
    def mock_system_config(self):
        """Create mock system configuration."""
        return {
            "max_workflow_retries": 3,
            "human_loop_config": {"enabled": True},
            "monitoring_config": {"enabled": True}
        }
    
    @pytest.fixture
    def mock_state_with_errors(self):
        """Create mock video generation state with errors."""
        return VideoGenerationState(
            messages=[],
            topic="Test topic",
            description="Test description",
            session_id="test_session",
            output_dir="test_output",
            print_response=False,
            use_rag=True,
            use_context_learning=True,
            context_learning_path="test_context",
            chroma_db_path="test_chroma",
            manim_docs_path="test_docs",
            embedding_model="test_model",
            use_visual_fix_code=False,
            use_langfuse=True,
            max_scene_concurrency=5,
            max_topic_concurrency=1,
            max_retries=5,
            use_enhanced_rag=True,
            enable_rag_caching=True,
            enable_quality_monitoring=True,
            enable_error_handling=True,
            rag_cache_ttl=3600,
            rag_max_cache_size=1000,
            rag_performance_threshold=2.0,
            rag_quality_threshold=0.7,
            enable_caching=True,
            default_quality="medium",
            use_gpu_acceleration=False,
            preview_mode=False,
            max_concurrent_renders=4,
            scene_outline="Test outline",
            scene_implementations={1: "Scene 1"},
            detected_plugins=["text"],
            generated_code={1: "test code"},
            code_errors={1: "syntax error"},
            rag_context={},
            rendered_videos={},
            combined_video_path=None,
            rendering_errors={1: "rendering failed"},
            visual_analysis_results={},
            visual_errors={},
            error_count=2,
            retry_count={"code_generator_agent": 1, "renderer_agent": 2},
            escalated_errors=[
                {
                    "agent": "code_generator_agent",
                    "error": "syntax error in generated code",
                    "timestamp": datetime.now().isoformat(),
                    "retry_count": 1
                },
                {
                    "agent": "renderer_agent", 
                    "error": "rendering timeout",
                    "timestamp": datetime.now().isoformat(),
                    "retry_count": 2
                }
            ],
            pending_human_input=None,
            human_feedback=None,
            performance_metrics={},
            execution_trace=[],
            current_agent="error_handler_agent",
            next_agent=None,
            workflow_complete=False,
            workflow_interrupted=False
        )
    
    @pytest.fixture
    def error_handler_agent(self, mock_config, mock_system_config):
        """Create ErrorHandlerAgent instance for testing."""
        return ErrorHandlerAgent(mock_config, mock_system_config)
    
    def test_error_handler_agent_initialization(self, error_handler_agent, mock_system_config):
        """Test ErrorHandlerAgent initialization."""
        assert error_handler_agent.name == "error_handler_agent"
        assert error_handler_agent.max_global_retries == 3
        assert len(error_handler_agent.error_patterns) > 0
        assert len(error_handler_agent.recovery_strategies) > 0
        assert error_handler_agent.error_history == []
    
    def test_initialize_error_patterns(self, error_handler_agent):
        """Test error pattern initialization."""
        patterns = error_handler_agent.error_patterns
        
        # Should have common error patterns
        assert "code_syntax_error" in patterns
        assert "rendering_timeout" in patterns
        assert "visual_analysis_failure" in patterns
        assert "rag_query_failure" in patterns
        
        # Each pattern should have required fields
        for pattern_name, pattern in patterns.items():
            assert "keywords" in pattern
            assert "severity" in pattern
            assert "recovery_agent" in pattern
    
    def test_initialize_recovery_strategies(self, error_handler_agent):
        """Test recovery strategy initialization."""
        strategies = error_handler_agent.recovery_strategies
        
        # Should have recovery strategies for different error types
        assert "code_syntax_error" in strategies
        assert "rendering_timeout" in strategies
        assert "visual_analysis_failure" in strategies
        
        # Each strategy should have required fields
        for strategy_name, strategy in strategies.items():
            assert strategy.error_pattern == strategy_name
            assert strategy.recovery_agent is not None
            assert strategy.max_attempts > 0
    
    @pytest.mark.asyncio
    async def test_execute_success(self, error_handler_agent, mock_state_with_errors):
        """Test successful error handling execution."""
        with patch.object(error_handler_agent, '_analyze_error_situation') as mock_analyze:
            mock_analyze.return_value = {
                "primary_error_type": "code_syntax_error",
                "affected_agents": ["code_generator_agent"],
                "severity": "medium",
                "recovery_strategy": "retry_with_rag"
            }
            
            with patch.object(error_handler_agent, '_determine_recovery_strategy') as mock_determine:
                mock_determine.return_value = {
                    "action": "retry_agent",
                    "target_agent": "code_generator_agent",
                    "strategy": "enhanced_rag"
                }
                
                with patch.object(error_handler_agent, '_execute_recovery_action') as mock_execute:
                    mock_execute.return_value = Command(goto="code_generator_agent")
                    
                    command = await error_handler_agent.execute(mock_state_with_errors)
                    
                    mock_analyze.assert_called_once_with(mock_state_with_errors)
                    mock_determine.assert_called_once()
                    mock_execute.assert_called_once()
                    assert command.goto == "code_generator_agent"
    
    @pytest.mark.asyncio
    async def test_execute_error_handler_failure(self, error_handler_agent, mock_state_with_errors):
        """Test error handler failure escalation to human."""
        with patch.object(error_handler_agent, '_analyze_error_situation') as mock_analyze:
            mock_analyze.side_effect = Exception("Analysis failed")
            
            command = await error_handler_agent.execute(mock_state_with_errors)
            
            # Should escalate to human when error handler itself fails
            assert command.goto == "human_loop_agent"
            assert "Error handler failed" in command.update["pending_human_input"]["context"]
    
    @pytest.mark.asyncio
    async def test_analyze_error_situation_code_errors(self, error_handler_agent, mock_state_with_errors):
        """Test error situation analysis for code errors."""
        analysis = await error_handler_agent._analyze_error_situation(mock_state_with_errors)
        
        assert analysis["total_errors"] == 2
        assert analysis["error_count"] == 2
        assert "code_generator_agent" in analysis["affected_agents"]
        assert "renderer_agent" in analysis["affected_agents"]
        assert analysis["has_code_errors"] is True
        assert analysis["has_rendering_errors"] is True
    
    @pytest.mark.asyncio
    async def test_analyze_error_situation_retry_patterns(self, error_handler_agent, mock_state_with_errors):
        """Test error analysis for retry patterns."""
        # Add more retry attempts to trigger pattern recognition
        mock_state_with_errors["retry_count"]["code_generator_agent"] = 3
        
        analysis = await error_handler_agent._analyze_error_situation(mock_state_with_errors)
        
        assert analysis["high_retry_agents"] == ["code_generator_agent"]
        assert analysis["needs_escalation"] is True
    
    @pytest.mark.asyncio
    async def test_determine_recovery_strategy_retry_agent(self, error_handler_agent, mock_state_with_errors):
        """Test recovery strategy determination for agent retry."""
        error_analysis = {
            "primary_error_type": "code_syntax_error",
            "affected_agents": ["code_generator_agent"],
            "severity": "medium",
            "needs_escalation": False,
            "high_retry_agents": []
        }
        
        recovery_action = await error_handler_agent._determine_recovery_strategy(
            error_analysis, mock_state_with_errors
        )
        
        assert recovery_action["action"] == "retry_agent"
        assert recovery_action["target_agent"] == "code_generator_agent"
        assert recovery_action["strategy"] == "enhanced_rag"
    
    @pytest.mark.asyncio
    async def test_determine_recovery_strategy_human_escalation(self, error_handler_agent, mock_state_with_errors):
        """Test recovery strategy determination for human escalation."""
        error_analysis = {
            "primary_error_type": "unknown_error",
            "affected_agents": ["code_generator_agent"],
            "severity": "high",
            "needs_escalation": True,
            "high_retry_agents": ["code_generator_agent"]
        }
        
        recovery_action = await error_handler_agent._determine_recovery_strategy(
            error_analysis, mock_state_with_errors
        )
        
        assert recovery_action["action"] == "escalate_to_human"
        assert recovery_action["context"] is not None
    
    @pytest.mark.asyncio
    async def test_execute_recovery_action_retry_agent(self, error_handler_agent, mock_state_with_errors):
        """Test execution of agent retry recovery action."""
        recovery_action = {
            "action": "retry_agent",
            "target_agent": "code_generator_agent",
            "strategy": "enhanced_rag",
            "modifications": {"use_enhanced_rag": True}
        }
        
        command = await error_handler_agent._execute_recovery_action(
            recovery_action, mock_state_with_errors
        )
        
        assert command.goto == "code_generator_agent"
        assert command.update["use_enhanced_rag"] is True
        assert command.update["current_agent"] == "code_generator_agent"
        
        # Should increment retry count
        assert command.update["retry_count"]["code_generator_agent"] == 2
    
    @pytest.mark.asyncio
    async def test_execute_recovery_action_escalate_human(self, error_handler_agent, mock_state_with_errors):
        """Test execution of human escalation recovery action."""
        recovery_action = {
            "action": "escalate_to_human",
            "context": "Multiple errors require human intervention",
            "options": ["retry_workflow", "skip_errors", "abort_workflow"]
        }
        
        command = await error_handler_agent._execute_recovery_action(
            recovery_action, mock_state_with_errors
        )
        
        assert command.goto == "human_loop_agent"
        assert command.update["pending_human_input"]["context"] == recovery_action["context"]
        assert command.update["pending_human_input"]["options"] == recovery_action["options"]
    
    @pytest.mark.asyncio
    async def test_execute_recovery_action_reset_workflow(self, error_handler_agent, mock_state_with_errors):
        """Test execution of workflow reset recovery action."""
        recovery_action = {
            "action": "reset_workflow",
            "reset_to_agent": "planner_agent",
            "preserve_data": ["topic", "description", "session_id"]
        }
        
        command = await error_handler_agent._execute_recovery_action(
            recovery_action, mock_state_with_errors
        )
        
        assert command.goto == "planner_agent"
        assert command.update["current_agent"] == "planner_agent"
        
        # Should reset error counts
        assert command.update["error_count"] == 0
        assert command.update["retry_count"] == {}
        assert command.update["escalated_errors"] == []
    
    def test_classify_error_code_syntax(self, error_handler_agent):
        """Test error classification for code syntax errors."""
        error = AgentError(
            agent_name="code_generator_agent",
            error_type="SyntaxError",
            error_message="invalid syntax in generated code",
            context={},
            timestamp=datetime.now(),
            retry_count=1
        )
        
        classification = error_handler_agent._classify_error(error)
        
        assert classification["error_type"] == "code_syntax_error"
        assert classification["severity"] == "medium"
        assert classification["recovery_agent"] == "code_generator_agent"
    
    def test_classify_error_rendering_timeout(self, error_handler_agent):
        """Test error classification for rendering timeout."""
        error = AgentError(
            agent_name="renderer_agent",
            error_type="TimeoutError",
            error_message="rendering process timed out",
            context={},
            timestamp=datetime.now(),
            retry_count=1
        )
        
        classification = error_handler_agent._classify_error(error)
        
        assert classification["error_type"] == "rendering_timeout"
        assert classification["severity"] == "high"
        assert classification["recovery_agent"] == "renderer_agent"
    
    def test_classify_error_unknown(self, error_handler_agent):
        """Test error classification for unknown errors."""
        error = AgentError(
            agent_name="unknown_agent",
            error_type="UnknownError",
            error_message="mysterious error occurred",
            context={},
            timestamp=datetime.now(),
            retry_count=1
        )
        
        classification = error_handler_agent._classify_error(error)
        
        assert classification["error_type"] == "unknown_error"
        assert classification["severity"] == "high"
        assert classification["recovery_agent"] == "human_loop_agent"
    
    def test_should_escalate_to_human_high_error_count(self, error_handler_agent, mock_state_with_errors):
        """Test human escalation decision based on high error count."""
        mock_state_with_errors["error_count"] = 5
        
        should_escalate = error_handler_agent._should_escalate_to_human(mock_state_with_errors)
        
        assert should_escalate is True
    
    def test_should_escalate_to_human_high_retry_count(self, error_handler_agent, mock_state_with_errors):
        """Test human escalation decision based on high retry count."""
        mock_state_with_errors["retry_count"]["code_generator_agent"] = 5
        
        should_escalate = error_handler_agent._should_escalate_to_human(mock_state_with_errors)
        
        assert should_escalate is True
    
    def test_should_escalate_to_human_critical_errors(self, error_handler_agent, mock_state_with_errors):
        """Test human escalation decision based on critical errors."""
        mock_state_with_errors["escalated_errors"].append({
            "agent": "error_handler_agent",
            "error": "critical system failure",
            "severity": "critical"
        })
        
        should_escalate = error_handler_agent._should_escalate_to_human(mock_state_with_errors)
        
        assert should_escalate is True
    
    def test_get_error_history(self, error_handler_agent):
        """Test error history tracking."""
        # Add some errors to history
        error1 = AgentError(
            agent_name="agent1",
            error_type="Error1",
            error_message="message1",
            context={},
            timestamp=datetime.now(),
            retry_count=1
        )
        error2 = AgentError(
            agent_name="agent2", 
            error_type="Error2",
            error_message="message2",
            context={},
            timestamp=datetime.now(),
            retry_count=1
        )
        
        error_handler_agent.error_history = [error1, error2]
        
        history = error_handler_agent.get_error_history()
        
        assert len(history) == 2
        assert history[0].agent_name == "agent1"
        assert history[1].agent_name == "agent2"
    
    def test_get_error_statistics(self, error_handler_agent, mock_state_with_errors):
        """Test error statistics generation."""
        stats = error_handler_agent.get_error_statistics(mock_state_with_errors)
        
        assert stats["total_errors"] == 2
        assert stats["error_count"] == 2
        assert stats["agents_with_errors"] == 2
        assert stats["max_retry_count"] == 2
        assert "code_generator_agent" in stats["retry_counts"]
        assert "renderer_agent" in stats["retry_counts"]
    
    def test_create_error_report(self, error_handler_agent, mock_state_with_errors):
        """Test error report creation."""
        report = error_handler_agent.create_error_report(mock_state_with_errors)
        
        assert "Error Handler Report" in report
        assert "Total Errors: 2" in report
        assert "code_generator_agent" in report
        assert "renderer_agent" in report
        assert "syntax error" in report
        assert "rendering failed" in report
    
    def test_reset_error_state(self, error_handler_agent):
        """Test error state reset."""
        # Add some error history
        error_handler_agent.error_history = [Mock(), Mock()]
        
        error_handler_agent.reset_error_state()
        
        assert error_handler_agent.error_history == []
    
    @pytest.mark.asyncio
    async def test_handle_recovery_failure(self, error_handler_agent, mock_state_with_errors):
        """Test handling of recovery action failure."""
        recovery_action = {
            "action": "retry_agent",
            "target_agent": "code_generator_agent"
        }
        
        # Mock recovery failure
        with patch.object(error_handler_agent, '_execute_recovery_action') as mock_execute:
            mock_execute.side_effect = Exception("Recovery failed")
            
            command = await error_handler_agent._handle_recovery_failure(
                recovery_action, mock_state_with_errors
            )
            
            # Should escalate to human when recovery fails
            assert command.goto == "human_loop_agent"
            assert "Recovery action failed" in command.update["pending_human_input"]["context"]


if __name__ == "__main__":
    pytest.main([__file__])
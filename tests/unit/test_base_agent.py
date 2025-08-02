"""
Unit tests for BaseAgent class.
Tests core functionality, error handling, and agent interface validation.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

from src.langgraph_agents.base_agent import BaseAgent, AgentFactory
from src.langgraph_agents.state import VideoGenerationState, AgentConfig, AgentError
from langgraph.types import Command

# Import test utilities
from tests.utils.config_mocks import (
    mock_configuration_manager, create_test_system_config, 
    create_test_agent_config, MockConfigurationManager
)


class TestBaseAgent:
    """Test suite for BaseAgent abstract class and common functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock agent configuration using configuration system."""
        return create_test_agent_config(
            name="test_agent",
            planner_model="openai/gpt-4o",
            scene_model="openai/gpt-4o", 
            helper_model="openai/gpt-4o-mini",
            tools=["test_tool"]
        )
    
    @pytest.fixture
    def mock_system_config(self):
        """Create mock system configuration using configuration system."""
        return create_test_system_config()
    
    @pytest.fixture
    def mock_state(self):
        """Create mock video generation state."""
        return VideoGenerationState(
            messages=[],
            topic="test topic",
            description="test description",
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
            scene_outline=None,
            scene_implementations={},
            detected_plugins=[],
            generated_code={},
            code_errors={},
            rag_context={},
            rendered_videos={},
            combined_video_path=None,
            rendering_errors={},
            visual_analysis_results={},
            visual_errors={},
            error_count=0,
            retry_count={},
            escalated_errors=[],
            pending_human_input=None,
            human_feedback=None,
            performance_metrics={},
            execution_trace=[],
            current_agent=None,
            next_agent=None,
            workflow_complete=False,
            workflow_interrupted=False
        )
    
    @pytest.fixture
    def concrete_agent(self, mock_config, mock_system_config):
        """Create concrete implementation of BaseAgent for testing."""
        class ConcreteAgent(BaseAgent):
            async def execute(self, state: VideoGenerationState) -> Command:
                return Command(goto="next_agent", update={"test": "success"})
        
        with mock_configuration_manager(mock_system_config):
            return ConcreteAgent(mock_config, mock_system_config)
    
    def test_base_agent_initialization(self, concrete_agent, mock_config):
        """Test BaseAgent initialization with configuration."""
        assert concrete_agent.name == "test_agent"
        assert concrete_agent.max_retries == 3
        assert concrete_agent.timeout_seconds == 300
        assert concrete_agent.enable_human_loop is False
        assert concrete_agent.config == mock_config
        assert concrete_agent.execution_stats['total_executions'] == 0
    
    @pytest.mark.asyncio
    async def test_execute_with_monitoring_success(self, concrete_agent, mock_state):
        """Test successful execution with monitoring."""
        with patch('src.langgraph_agents.base_agent.get_langfuse_service') as mock_langfuse:
            mock_langfuse.return_value = None  # Disable LangFuse for test
            
            command = await concrete_agent.execute_with_monitoring(mock_state)
            
            assert command.goto == "next_agent"
            assert command.update["test"] == "success"
            assert concrete_agent.execution_stats['successful_executions'] == 1
            assert concrete_agent.execution_stats['total_executions'] == 1
    
    @pytest.mark.asyncio
    async def test_execute_with_monitoring_timeout(self, mock_config, mock_system_config, mock_state):
        """Test execution timeout handling."""
        # Create agent with short timeout
        mock_config.timeout_seconds = 0.1
        
        class SlowAgent(BaseAgent):
            async def execute(self, state: VideoGenerationState) -> Command:
                await asyncio.sleep(0.2)  # Longer than timeout
                return Command(goto="next_agent")
        
        agent = SlowAgent(mock_config, mock_system_config)
        
        with patch('src.langgraph_agents.base_agent.get_langfuse_service') as mock_langfuse:
            mock_langfuse.return_value = None
            
            command = await agent.execute_with_monitoring(mock_state)
            
            # Should route to error handler
            assert command.goto == "error_handler_agent"
            assert agent.execution_stats['failed_executions'] == 1
    
    @pytest.mark.asyncio
    async def test_handle_error(self, concrete_agent, mock_state):
        """Test error handling functionality."""
        test_error = ValueError("Test error")
        
        command = await concrete_agent.handle_error(test_error, mock_state)
        
        assert command.goto == "error_handler_agent"
        assert command.update["error_count"] == 1
        assert len(command.update["escalated_errors"]) == 1
        assert command.update["escalated_errors"][0]["agent"] == "test_agent"
        assert command.update["escalated_errors"][0]["error"] == "Test error"
    
    def test_get_model_wrapper_openrouter(self, concrete_agent, mock_state):
        """Test model wrapper creation for OpenRouter models."""
        with patch('src.langgraph_agents.base_agent.OpenRouterWrapper') as mock_wrapper:
            mock_instance = Mock()
            mock_wrapper.return_value = mock_instance
            
            wrapper = concrete_agent.get_model_wrapper("openrouter/anthropic/claude-3.5-sonnet", mock_state)
            
            mock_wrapper.assert_called_once()
            assert wrapper == mock_instance
    
    def test_get_model_wrapper_litellm(self, concrete_agent, mock_state):
        """Test model wrapper creation for LiteLLM models."""
        with patch('src.langgraph_agents.base_agent.LiteLLMWrapper') as mock_wrapper:
            mock_instance = Mock()
            mock_wrapper.return_value = mock_instance
            
            wrapper = concrete_agent.get_model_wrapper("openai/gpt-4o", mock_state)
            
            mock_wrapper.assert_called_once()
            assert wrapper == mock_instance
    
    def test_should_escalate_to_human_disabled(self, concrete_agent, mock_state):
        """Test human escalation when human loop is disabled."""
        assert concrete_agent.should_escalate_to_human(mock_state) is False
    
    def test_should_escalate_to_human_error_threshold(self, mock_config, mock_system_config, mock_state):
        """Test human escalation based on error count threshold."""
        mock_config.enable_human_loop = True
        agent = BaseAgent.__new__(BaseAgent)
        agent.__init__(mock_config, mock_system_config)
        
        # Set error count above threshold
        mock_state["error_count"] = 5
        
        assert agent.should_escalate_to_human(mock_state) is True
    
    def test_should_escalate_to_human_retry_threshold(self, mock_config, mock_system_config, mock_state):
        """Test human escalation based on retry count threshold."""
        mock_config.enable_human_loop = True
        agent = BaseAgent.__new__(BaseAgent)
        agent.__init__(mock_config, mock_system_config)
        
        # Set retry count above threshold for this agent
        mock_state["retry_count"] = {"test_agent": 5}
        
        assert agent.should_escalate_to_human(mock_state) is True
    
    def test_create_human_intervention_command(self, concrete_agent, mock_state):
        """Test creation of human intervention command."""
        with patch('src.langgraph_agents.base_agent.request_human_approval') as mock_request:
            mock_command = Command(goto="human_loop_agent")
            mock_request.return_value = mock_command
            
            command = concrete_agent.create_human_intervention_command(
                context="Test context",
                options=["option1", "option2"],
                state=mock_state
            )
            
            mock_request.assert_called_once()
            assert command == mock_command
    
    def test_log_agent_action(self, concrete_agent):
        """Test agent action logging."""
        with patch('src.langgraph_agents.base_agent.logger') as mock_logger:
            concrete_agent.log_agent_action("test_action", {"key": "value"})
            
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            assert "test_action" in call_args[0][0]
            assert call_args[1]['extra']['agent'] == "test_agent"
            assert call_args[1]['extra']['action'] == "test_action"


class TestAgentFactory:
    """Test suite for AgentFactory."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock agent configuration."""
        return AgentConfig(
            name="test_agent",
            model_config={},
            tools=[]
        )
    
    @pytest.fixture
    def mock_system_config(self):
        """Create mock system configuration."""
        return {}
    
    def test_create_planner_agent(self, mock_config, mock_system_config):
        """Test creation of PlannerAgent."""
        with patch('src.langgraph_agents.base_agent.PlannerAgent') as mock_agent_class:
            mock_instance = Mock()
            mock_agent_class.return_value = mock_instance
            
            agent = AgentFactory.create_agent("planner_agent", mock_config, mock_system_config)
            
            mock_agent_class.assert_called_once_with(mock_config, mock_system_config)
            assert agent == mock_instance
    
    def test_create_code_generator_agent(self, mock_config, mock_system_config):
        """Test creation of CodeGeneratorAgent."""
        with patch('src.langgraph_agents.base_agent.CodeGeneratorAgent') as mock_agent_class:
            mock_instance = Mock()
            mock_agent_class.return_value = mock_instance
            
            agent = AgentFactory.create_agent("code_generator_agent", mock_config, mock_system_config)
            
            mock_agent_class.assert_called_once_with(mock_config, mock_system_config)
            assert agent == mock_instance
    
    def test_create_unknown_agent_type(self, mock_config, mock_system_config):
        """Test creation of unknown agent type raises error."""
        with pytest.raises(ValueError, match="Unknown agent type: unknown_agent"):
            AgentFactory.create_agent("unknown_agent", mock_config, mock_system_config)
    
    def test_create_all_agent_types(self, mock_config, mock_system_config):
        """Test that all expected agent types can be created."""
        expected_agents = [
            'planner_agent',
            'code_generator_agent', 
            'renderer_agent',
            'visual_analysis_agent',
            'rag_agent',
            'error_handler_agent',
            'monitoring_agent',
            'human_loop_agent'
        ]
        
        for agent_type in expected_agents:
            with patch(f'src.langgraph_agents.base_agent.{agent_type.title().replace("_", "")}') as mock_class:
                mock_instance = Mock()
                mock_class.return_value = mock_instance
                
                agent = AgentFactory.create_agent(agent_type, mock_config, mock_system_config)
                assert agent == mock_instance


if __name__ == "__main__":
    pytest.main([__file__])
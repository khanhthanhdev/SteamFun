"""
Unit tests for AgentAdapter.

Tests the adapter functionality for maintaining existing agent interfaces
while using new node functions.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from src.langgraph_agents.adapters.agent_adapter import (
    AgentAdapter, LegacyAgentWrapper, AgentFactory
)
from src.langgraph_agents.models.state import VideoGenerationState as NewVideoGenerationState
from src.langgraph_agents.state import AgentConfig
from src.langgraph_agents.models.config import WorkflowConfig
from langgraph.types import Command


class TestAgentAdapter:
    """Test cases for AgentAdapter functionality."""
    
    @pytest.fixture
    def mock_node_function(self):
        """Create a mock node function for testing."""
        async def mock_node(state: NewVideoGenerationState) -> NewVideoGenerationState:
            # Simulate node processing
            state.current_step = 'code_generation'
            state.scene_outline = 'Updated scene outline'
            return state
        
        return mock_node
    
    @pytest.fixture
    def agent_config(self):
        """Create agent configuration for testing."""
        return AgentConfig(
            name='test_agent',
            model_config={'provider': 'openrouter', 'model': 'claude-3.5-sonnet'},
            tools=['test_tool'],
            max_retries=3,
            timeout_seconds=300,
            enable_human_loop=False,
            temperature=0.7,
            print_cost=True,
            verbose=False
        )
    
    @pytest.fixture
    def system_config(self):
        """Create system configuration for testing."""
        return {
            'monitoring_config': {'enabled': True},
            'human_loop_config': {'enabled': False}
        }
    
    @pytest.fixture
    def sample_old_state(self):
        """Create sample old state for testing."""
        return {
            'topic': 'Test topic',
            'description': 'Test description',
            'session_id': 'test-123',
            'scene_outline': 'Original scene outline',
            'current_agent': 'planner_agent',
            'workflow_complete': False,
            'use_rag': True,
            'enable_caching': True,
            'max_retries': 3,
            'output_dir': 'output'
        }
    
    def test_agent_adapter_initialization(self, mock_node_function, agent_config, system_config):
        """Test AgentAdapter initialization."""
        adapter = AgentAdapter(mock_node_function, 'test_agent', agent_config, system_config)
        
        assert adapter.node_function == mock_node_function
        assert adapter.agent_name == 'test_agent'
        assert adapter.config == agent_config
        assert adapter.system_config == system_config
        assert adapter._mock_agent is not None
    
    @pytest.mark.asyncio
    async def test_agent_adapter_execute_success(self, mock_node_function, agent_config, 
                                                system_config, sample_old_state):
        """Test successful execution through AgentAdapter."""
        adapter = AgentAdapter(mock_node_function, 'test_agent', agent_config, system_config)
        
        # Execute with old state
        command = await adapter.execute(sample_old_state)
        
        # Verify command structure
        assert isinstance(command, Command)
        assert command.goto is not None
        assert command.update is not None
        
        # Verify state was updated
        assert command.update['scene_outline'] == 'Updated scene outline'
    
    @pytest.mark.asyncio
    async def test_agent_adapter_execute_with_dict_state(self, mock_node_function, agent_config,
                                                        system_config, sample_old_state):
        """Test execution with dictionary state input."""
        adapter = AgentAdapter(mock_node_function, 'test_agent', agent_config, system_config)
        
        # Execute with dict state
        command = await adapter.execute(sample_old_state)
        
        # Should handle dict input correctly
        assert isinstance(command, Command)
        assert command.update is not None
    
    @pytest.mark.asyncio
    async def test_agent_adapter_error_handling(self, agent_config, system_config, sample_old_state):
        """Test error handling in AgentAdapter."""
        # Create a node function that raises an error
        async def failing_node(state: NewVideoGenerationState) -> NewVideoGenerationState:
            raise ValueError("Test error")
        
        adapter = AgentAdapter(failing_node, 'test_agent', agent_config, system_config)
        
        # Execute should handle error gracefully
        command = await adapter.execute(sample_old_state)
        
        # Should return error handling command
        assert isinstance(command, Command)
        assert command.goto == 'error_handler_agent'
    
    def test_determine_next_agent_normal_flow(self, mock_node_function, agent_config, system_config):
        """Test next agent determination in normal workflow."""
        adapter = AgentAdapter(mock_node_function, 'test_agent', agent_config, system_config)
        
        # Create test states for different steps
        config = WorkflowConfig()
        
        # Planning step
        planning_state = NewVideoGenerationState(
            topic='Test', description='Test', session_id='test',
            config=config, current_step='planning'
        )
        next_agent = adapter._determine_next_agent(planning_state)
        assert next_agent == 'code_generator_agent'
        
        # Code generation step
        code_gen_state = NewVideoGenerationState(
            topic='Test', description='Test', session_id='test',
            config=config, current_step='code_generation'
        )
        next_agent = adapter._determine_next_agent(code_gen_state)
        assert next_agent == 'renderer_agent'
        
        # Completed workflow
        completed_state = NewVideoGenerationState(
            topic='Test', description='Test', session_id='test',
            config=config, workflow_complete=True
        )
        next_agent = adapter._determine_next_agent(completed_state)
        assert next_agent is None
    
    def test_determine_next_agent_with_errors(self, mock_node_function, agent_config, system_config):
        """Test next agent determination with errors."""
        adapter = AgentAdapter(mock_node_function, 'test_agent', agent_config, system_config)
        
        config = WorkflowConfig()
        
        # State with errors
        error_state = NewVideoGenerationState(
            topic='Test', description='Test', session_id='test',
            config=config, current_step='planning'
        )
        error_state.add_error(Mock())  # Add a mock error
        
        next_agent = adapter._determine_next_agent(error_state)
        assert next_agent == 'error_handler_agent'
    
    def test_agent_adapter_attribute_delegation(self, mock_node_function, agent_config, system_config):
        """Test attribute delegation to mock agent."""
        adapter = AgentAdapter(mock_node_function, 'test_agent', agent_config, system_config)
        
        # Should delegate to mock agent
        assert adapter.name == agent_config.name
        assert adapter.max_retries == agent_config.max_retries
        assert adapter.timeout_seconds == agent_config.timeout_seconds


class TestLegacyAgentWrapper:
    """Test cases for LegacyAgentWrapper functionality."""
    
    @pytest.fixture
    def mock_node_function(self):
        """Create a mock node function for testing."""
        async def mock_node(state: NewVideoGenerationState) -> NewVideoGenerationState:
            state.current_step = 'code_generation'
            state.scene_outline = 'Updated by node'
            return state
        
        return mock_node
    
    @pytest.fixture
    def wrapper_config(self):
        """Create wrapper configuration for testing."""
        return {
            'max_retries': 3,
            'timeout_seconds': 300,
            'enable_human_loop': False,
            'model_config': {'provider': 'openrouter'},
            'planner_model': 'openrouter/claude-3.5-sonnet'
        }
    
    def test_legacy_wrapper_initialization(self, mock_node_function, wrapper_config):
        """Test LegacyAgentWrapper initialization."""
        wrapper = LegacyAgentWrapper(mock_node_function, 'test_agent', wrapper_config)
        
        assert wrapper.node_function == mock_node_function
        assert wrapper.name == 'test_agent'
        assert wrapper.config == wrapper_config
        assert wrapper.max_retries == 3
        assert wrapper.timeout_seconds == 300
        assert wrapper.enable_human_loop is False
    
    @pytest.mark.asyncio
    async def test_legacy_wrapper_execute_with_new_state(self, mock_node_function, wrapper_config):
        """Test execution with new state format."""
        wrapper = LegacyAgentWrapper(mock_node_function, 'test_agent', wrapper_config)
        
        config = WorkflowConfig()
        new_state = NewVideoGenerationState(
            topic='Test', description='Test', session_id='test',
            config=config, current_step='planning'
        )
        
        result = await wrapper.execute(new_state)
        
        # Should return updated new state
        assert isinstance(result, NewVideoGenerationState)
        assert result.scene_outline == 'Updated by node'
        assert result.current_step == 'code_generation'
    
    @pytest.mark.asyncio
    async def test_legacy_wrapper_execute_with_old_state(self, mock_node_function, wrapper_config):
        """Test execution with old state format."""
        wrapper = LegacyAgentWrapper(mock_node_function, 'test_agent', wrapper_config)
        
        old_state = {
            'topic': 'Test topic',
            'description': 'Test description',
            'session_id': 'test-123',
            'scene_outline': 'Original outline',
            'use_rag': True,
            'enable_caching': True,
            'max_retries': 3,
            'output_dir': 'output'
        }
        
        result = await wrapper.execute(old_state)
        
        # Should return old state format
        assert isinstance(result, dict)
        assert result['scene_outline'] == 'Updated by node'
    
    @pytest.mark.asyncio
    async def test_legacy_wrapper_process_method(self, mock_node_function, wrapper_config):
        """Test legacy process method compatibility."""
        wrapper = LegacyAgentWrapper(mock_node_function, 'test_agent', wrapper_config)
        
        old_state = {
            'topic': 'Test topic',
            'description': 'Test description',
            'session_id': 'test-123',
            'use_rag': True,
            'enable_caching': True,
            'max_retries': 3,
            'output_dir': 'output'
        }
        
        # Test process with positional argument
        result = await wrapper.process(old_state)
        assert isinstance(result, dict)
        
        # Test process with keyword argument
        result = await wrapper.process(state=old_state)
        assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_legacy_wrapper_process_no_state(self, mock_node_function, wrapper_config):
        """Test process method with no state provided."""
        wrapper = LegacyAgentWrapper(mock_node_function, 'test_agent', wrapper_config)
        
        with pytest.raises(ValueError, match="No state provided"):
            await wrapper.process()
    
    def test_legacy_wrapper_execution_stats(self, mock_node_function, wrapper_config):
        """Test execution statistics tracking."""
        wrapper = LegacyAgentWrapper(mock_node_function, 'test_agent', wrapper_config)
        
        # Initial stats
        assert wrapper.execution_stats['total_executions'] == 0
        assert wrapper.execution_stats['successful_executions'] == 0
        assert wrapper.execution_stats['failed_executions'] == 0
    
    @pytest.mark.asyncio
    async def test_legacy_wrapper_execution_stats_update(self, mock_node_function, wrapper_config):
        """Test execution statistics are updated correctly."""
        wrapper = LegacyAgentWrapper(mock_node_function, 'test_agent', wrapper_config)
        
        old_state = {
            'topic': 'Test topic',
            'description': 'Test description',
            'session_id': 'test-123',
            'use_rag': True,
            'enable_caching': True,
            'max_retries': 3,
            'output_dir': 'output'
        }
        
        # Execute successfully
        await wrapper.execute(old_state)
        
        # Check stats were updated
        assert wrapper.execution_stats['total_executions'] == 1
        assert wrapper.execution_stats['successful_executions'] == 1
        assert wrapper.execution_stats['failed_executions'] == 0
        assert wrapper.execution_stats['average_execution_time'] > 0
    
    @pytest.mark.asyncio
    async def test_legacy_wrapper_execution_error_stats(self, wrapper_config):
        """Test execution statistics on error."""
        # Create failing node function
        async def failing_node(state: NewVideoGenerationState) -> NewVideoGenerationState:
            raise ValueError("Test error")
        
        wrapper = LegacyAgentWrapper(failing_node, 'test_agent', wrapper_config)
        
        old_state = {
            'topic': 'Test topic',
            'description': 'Test description',
            'session_id': 'test-123',
            'use_rag': True,
            'enable_caching': True,
            'max_retries': 3,
            'output_dir': 'output'
        }
        
        # Execute should raise error
        with pytest.raises(ValueError):
            await wrapper.execute(old_state)
        
        # Check error stats were updated
        assert wrapper.execution_stats['total_executions'] == 1
        assert wrapper.execution_stats['successful_executions'] == 0
        assert wrapper.execution_stats['failed_executions'] == 1
    
    @patch('src.langgraph_agents.adapters.agent_adapter.OpenRouterWrapper')
    def test_legacy_wrapper_get_model_wrapper_openrouter(self, mock_openrouter, 
                                                        mock_node_function, wrapper_config):
        """Test model wrapper creation for OpenRouter models."""
        wrapper = LegacyAgentWrapper(mock_node_function, 'test_agent', wrapper_config)
        
        model_wrapper = wrapper.get_model_wrapper('openrouter/claude-3.5-sonnet')
        
        # Should create OpenRouter wrapper
        mock_openrouter.assert_called_once()
    
    @patch('src.langgraph_agents.adapters.agent_adapter.OpenAIWrapper')
    def test_legacy_wrapper_get_model_wrapper_openai(self, mock_openai, 
                                                    mock_node_function, wrapper_config):
        """Test model wrapper creation for OpenAI models."""
        wrapper = LegacyAgentWrapper(mock_node_function, 'test_agent', wrapper_config)
        
        model_wrapper = wrapper.get_model_wrapper('gpt-4')
        
        # Should create OpenAI wrapper
        mock_openai.assert_called_once()
    
    @patch('src.langgraph_agents.adapters.agent_adapter.LiteLLMWrapper')
    def test_legacy_wrapper_get_model_wrapper_fallback(self, mock_litellm, 
                                                      mock_node_function, wrapper_config):
        """Test model wrapper creation fallback."""
        wrapper = LegacyAgentWrapper(mock_node_function, 'test_agent', wrapper_config)
        
        model_wrapper = wrapper.get_model_wrapper('unknown-model')
        
        # Should create LiteLLM wrapper
        mock_litellm.assert_called_once()
    
    def test_legacy_wrapper_should_escalate_to_human_disabled(self, mock_node_function):
        """Test human escalation when disabled."""
        config = {'enable_human_loop': False}
        wrapper = LegacyAgentWrapper(mock_node_function, 'test_agent', config)
        
        state = {'errors': ['error1', 'error2', 'error3']}
        
        # Should not escalate when disabled
        assert not wrapper.should_escalate_to_human(state)
    
    def test_legacy_wrapper_should_escalate_to_human_enabled(self, mock_node_function):
        """Test human escalation when enabled."""
        config = {'enable_human_loop': True, 'max_retries': 2}
        wrapper = LegacyAgentWrapper(mock_node_function, 'test_agent', config)
        
        # Create state with new format
        workflow_config = WorkflowConfig()
        state = NewVideoGenerationState(
            topic='Test', description='Test', session_id='test',
            config=workflow_config
        )
        
        # Add multiple errors
        for i in range(3):
            state.add_error(Mock())
        
        # Should escalate with many errors
        assert wrapper.should_escalate_to_human(state)
    
    def test_legacy_wrapper_log_agent_action(self, mock_node_function, wrapper_config):
        """Test agent action logging."""
        wrapper = LegacyAgentWrapper(mock_node_function, 'test_agent', wrapper_config)
        
        # Should not raise error
        wrapper.log_agent_action('test_action', {'detail': 'test'})


class TestAgentFactory:
    """Test cases for AgentFactory functionality."""
    
    @pytest.fixture
    def mock_node_function(self):
        """Create a mock node function for testing."""
        async def mock_node(state: NewVideoGenerationState) -> NewVideoGenerationState:
            return state
        
        return mock_node
    
    @pytest.fixture
    def agent_config(self):
        """Create agent configuration for testing."""
        return AgentConfig(
            name='test_agent',
            model_config={},
            tools=[],
            max_retries=3
        )
    
    @pytest.fixture
    def system_config(self):
        """Create system configuration for testing."""
        return {'test': 'config'}
    
    def test_create_adapter(self, mock_node_function, agent_config, system_config):
        """Test AgentAdapter creation."""
        adapter = AgentFactory.create_adapter(
            mock_node_function, 'test_agent', agent_config, system_config
        )
        
        assert isinstance(adapter, AgentAdapter)
        assert adapter.agent_name == 'test_agent'
        assert adapter.config == agent_config
    
    def test_create_legacy_wrapper(self, mock_node_function):
        """Test LegacyAgentWrapper creation."""
        config = {'max_retries': 3}
        wrapper = AgentFactory.create_legacy_wrapper(
            mock_node_function, 'test_agent', config
        )
        
        assert isinstance(wrapper, LegacyAgentWrapper)
        assert wrapper.name == 'test_agent'
        assert wrapper.config == config
    
    def test_wrap_all_nodes(self, mock_node_function):
        """Test wrapping multiple node functions."""
        node_functions = {
            'planner_agent': mock_node_function,
            'code_generator_agent': mock_node_function
        }
        
        configs = {
            'planner_agent': {'max_retries': 3},
            'code_generator_agent': {'max_retries': 5}
        }
        
        wrapped_agents = AgentFactory.wrap_all_nodes(node_functions, configs)
        
        assert len(wrapped_agents) == 2
        assert 'planner_agent' in wrapped_agents
        assert 'code_generator_agent' in wrapped_agents
        assert isinstance(wrapped_agents['planner_agent'], LegacyAgentWrapper)
        assert wrapped_agents['planner_agent'].max_retries == 3
        assert wrapped_agents['code_generator_agent'].max_retries == 5
    
    def test_create_compatible_agent(self, mock_node_function):
        """Test creating compatible agent wrapper."""
        node_functions = {
            'planning_node': mock_node_function,
            'code_generation_node': mock_node_function
        }
        
        config = {'max_retries': 3}
        
        # Test planner agent creation
        planner_agent = AgentFactory.create_compatible_agent(
            'planner_agent', node_functions, config
        )
        
        assert isinstance(planner_agent, LegacyAgentWrapper)
        assert planner_agent.name == 'planner_agent'
        assert planner_agent.config == config
    
    def test_create_compatible_agent_unknown_type(self, mock_node_function):
        """Test creating compatible agent with unknown type."""
        node_functions = {'planning_node': mock_node_function}
        
        with pytest.raises(ValueError, match="Unknown agent type"):
            AgentFactory.create_compatible_agent('unknown_agent', node_functions)
    
    def test_create_compatible_agent_missing_node(self, mock_node_function):
        """Test creating compatible agent with missing node function."""
        node_functions = {'other_node': mock_node_function}
        
        with pytest.raises(ValueError, match="missing node function"):
            AgentFactory.create_compatible_agent('planner_agent', node_functions)
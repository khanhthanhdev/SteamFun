"""
Integration tests for the LangGraph StateGraph workflow.

This module tests the complete workflow graph construction, node integration,
and conditional routing logic.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.langgraph_agents.workflow_graph import (
    VideoGenerationWorkflow,
    create_workflow,
    validate_workflow_configuration
)
from src.langgraph_agents.models.state import VideoGenerationState
from src.langgraph_agents.models.config import WorkflowConfig, ModelConfig
from src.langgraph_agents.models.errors import WorkflowError, ErrorType, ErrorSeverity


class TestWorkflowGraphConstruction:
    """Test workflow graph construction and configuration."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock workflow configuration."""
        return WorkflowConfig(
            planner_model=ModelConfig(
                provider="openrouter",
                model_name="anthropic/claude-3.5-sonnet",
                temperature=0.7,
                max_tokens=4000
            ),
            code_model=ModelConfig(
                provider="openrouter",
                model_name="anthropic/claude-3.5-sonnet",
                temperature=0.7,
                max_tokens=4000
            ),
            helper_model=ModelConfig(
                provider="openrouter",
                model_name="anthropic/claude-3.5-sonnet",
                temperature=0.7,
                max_tokens=2000
            ),
            output_dir="/tmp/test_output",
            max_retries=3,
            timeout_seconds=300,
            use_rag=True,
            use_visual_analysis=False,
            enable_caching=True
        )
    
    @pytest.fixture
    def initial_state(self):
        """Create an initial workflow state for testing."""
        return VideoGenerationState(
            topic="Python basics",
            description="Introduction to Python programming concepts",
            session_id="test-session-123"
        )
    
    def test_workflow_creation_without_checkpointing(self, mock_config):
        """Test creating workflow without checkpointing."""
        workflow = VideoGenerationWorkflow(mock_config)
        
        assert workflow.config == mock_config
        assert workflow.checkpointer is None
        assert workflow.graph is not None
        
        # Verify graph has expected nodes
        graph_nodes = list(workflow.graph.nodes.keys())
        expected_nodes = [
            "planning",
            "code_generation", 
            "rendering",
            "error_handler",
            "rag_enhancement",
            "visual_analysis",
            "human_loop",
            "complete"
        ]
        
        for node in expected_nodes:
            assert node in graph_nodes
    
    def test_workflow_creation_with_memory_checkpointing(self, mock_config):
        """Test creating workflow with memory checkpointing."""
        workflow = create_workflow(mock_config, use_checkpointing=True)
        
        assert workflow.config == mock_config
        assert workflow.checkpointer is not None
        assert workflow.graph is not None
    
    def test_workflow_configuration_validation_valid(self, mock_config):
        """Test workflow configuration validation with valid config."""
        assert validate_workflow_configuration(mock_config) is True
    
    def test_workflow_configuration_validation_missing_fields(self):
        """Test workflow configuration validation with missing fields."""
        invalid_config = WorkflowConfig()
        assert validate_workflow_configuration(invalid_config) is False
    
    def test_workflow_configuration_validation_invalid_retries(self, mock_config):
        """Test workflow configuration validation with invalid retry count."""
        mock_config.max_retries = 0
        assert validate_workflow_configuration(mock_config) is False
    
    def test_workflow_configuration_validation_invalid_timeout(self, mock_config):
        """Test workflow configuration validation with invalid timeout."""
        mock_config.timeout_seconds = 10
        assert validate_workflow_configuration(mock_config) is False


class TestWorkflowExecution:
    """Test workflow execution and node integration."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock workflow configuration."""
        return WorkflowConfig(
            planner_model=ModelConfig(
                provider="openrouter",
                model_name="anthropic/claude-3.5-sonnet",
                temperature=0.7,
                max_tokens=4000
            ),
            code_model=ModelConfig(
                provider="openrouter",
                model_name="anthropic/claude-3.5-sonnet",
                temperature=0.7,
                max_tokens=4000
            ),
            helper_model=ModelConfig(
                provider="openrouter",
                model_name="anthropic/claude-3.5-sonnet",
                temperature=0.7,
                max_tokens=2000
            ),
            output_dir="/tmp/test_output",
            max_retries=3,
            timeout_seconds=300,
            use_rag=True,
            use_visual_analysis=False,
            enable_caching=True
        )
    
    @pytest.fixture
    def workflow(self, mock_config):
        """Create a workflow instance for testing."""
        return VideoGenerationWorkflow(mock_config)
    
    @pytest.fixture
    def initial_state(self):
        """Create an initial workflow state for testing."""
        return VideoGenerationState(
            topic="Python basics",
            description="Introduction to Python programming concepts",
            session_id="test-session-123"
        )
    
    @patch('src.langgraph_agents.nodes.planning_node.planning_node')
    @patch('src.langgraph_agents.nodes.code_generation_node.code_generation_node')
    @patch('src.langgraph_agents.nodes.rendering_node.rendering_node')
    async def test_successful_workflow_execution(
        self,
        mock_rendering_node,
        mock_code_generation_node,
        mock_planning_node,
        workflow,
        initial_state
    ):
        """Test successful end-to-end workflow execution."""
        
        # Mock planning node success
        planning_result = initial_state.model_copy()
        planning_result.scene_outline = "Scene 1: Introduction\nScene 2: Examples"
        planning_result.scene_implementations = {
            1: "Show Python syntax basics",
            2: "Demonstrate variables and data types"
        }
        planning_result.current_step = "planning"
        mock_planning_node.return_value = planning_result
        
        # Mock code generation node success
        code_gen_result = planning_result.model_copy()
        code_gen_result.generated_code = {
            1: "from manim import *\nclass Scene1(Scene):\n    def construct(self):\n        pass",
            2: "from manim import *\nclass Scene2(Scene):\n    def construct(self):\n        pass"
        }
        code_gen_result.current_step = "code_generation"
        mock_code_generation_node.return_value = code_gen_result
        
        # Mock rendering node success
        rendering_result = code_gen_result.model_copy()
        rendering_result.rendered_videos = {
            1: "/tmp/test_output/scene1.mp4",
            2: "/tmp/test_output/scene2.mp4"
        }
        rendering_result.combined_video_path = "/tmp/test_output/combined.mp4"
        rendering_result.current_step = "rendering"
        mock_rendering_node.return_value = rendering_result
        
        # Execute workflow
        result = await workflow.invoke(initial_state)
        
        # Verify workflow completion
        assert result.workflow_complete is True
        assert result.current_step == "complete"
        assert len(result.rendered_videos) == 2
        assert result.combined_video_path is not None
        
        # Verify nodes were called
        mock_planning_node.assert_called_once()
        mock_code_generation_node.assert_called_once()
        mock_rendering_node.assert_called_once()
    
    @patch('src.langgraph_agents.nodes.planning_node.planning_node')
    @patch('src.langgraph_agents.nodes.error_handler_node.error_handler_node')
    async def test_workflow_with_error_recovery(
        self,
        mock_error_handler_node,
        mock_planning_node,
        workflow,
        initial_state
    ):
        """Test workflow execution with error recovery."""
        
        # Mock planning node failure
        planning_result = initial_state.model_copy()
        planning_result.add_error(WorkflowError(
            step="planning",
            error_type=ErrorType.MODEL,
            message="Model timeout",
            severity=ErrorSeverity.MEDIUM,
            recoverable=True
        ))
        planning_result.current_step = "planning"
        mock_planning_node.return_value = planning_result
        
        # Mock error handler recovery
        error_recovery_result = planning_result.model_copy()
        error_recovery_result.errors = []  # Clear errors after recovery
        error_recovery_result.scene_outline = "Recovered scene outline"
        error_recovery_result.scene_implementations = {1: "Recovered implementation"}
        error_recovery_result.current_step = "error_handling"
        mock_error_handler_node.return_value = error_recovery_result
        
        # Execute workflow
        result = await workflow.invoke(initial_state)
        
        # Verify error handler was called
        mock_error_handler_node.assert_called()
        
        # Verify workflow continued after recovery
        assert len(result.execution_trace) > 0
    
    async def test_workflow_with_invalid_initial_state(self, workflow):
        """Test workflow execution with invalid initial state."""
        invalid_state = VideoGenerationState(
            topic="",  # Empty topic
            description="",  # Empty description
            session_id="test-session"
        )
        
        with pytest.raises(ValueError, match="Topic and description are required"):
            await workflow.invoke(invalid_state)
    
    @patch('src.langgraph_agents.nodes.planning_node.planning_node')
    async def test_workflow_streaming(self, mock_planning_node, workflow, initial_state):
        """Test workflow streaming functionality."""
        
        # Mock planning node
        planning_result = initial_state.model_copy()
        planning_result.scene_outline = "Test outline"
        planning_result.scene_implementations = {1: "Test implementation"}
        planning_result.current_step = "planning"
        mock_planning_node.return_value = planning_result
        
        # Collect streaming results
        stream_results = []
        async for chunk in workflow.stream(initial_state):
            stream_results.append(chunk)
        
        # Verify we received streaming updates
        assert len(stream_results) > 0
    
    def test_graph_visualization(self, workflow):
        """Test graph visualization generation."""
        visualization = workflow.get_graph_visualization()
        
        assert isinstance(visualization, str)
        assert "graph TD" in visualization
        assert "planning" in visualization
        assert "code_generation" in visualization
        assert "rendering" in visualization


class TestConditionalRouting:
    """Test conditional routing logic integration."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock workflow configuration."""
        return WorkflowConfig(
            planner_model=ModelConfig(
                provider="openrouter",
                model_name="anthropic/claude-3.5-sonnet"
            ),
            code_model=ModelConfig(
                provider="openrouter",
                model_name="anthropic/claude-3.5-sonnet"
            ),
            helper_model=ModelConfig(
                provider="openrouter",
                model_name="anthropic/claude-3.5-sonnet"
            ),
            output_dir="/tmp/test_output",
            max_retries=3,
            timeout_seconds=300
        )
    
    @pytest.fixture
    def workflow(self, mock_config):
        """Create a workflow instance for testing."""
        return VideoGenerationWorkflow(mock_config)
    
    def test_route_from_rag_enhancement_success(self, workflow):
        """Test routing from RAG enhancement node on success."""
        state = VideoGenerationState(
            topic="test",
            description="test",
            session_id="test"
        )
        
        route = workflow._route_from_rag_enhancement(state)
        assert route == "code_generation"
    
    def test_route_from_rag_enhancement_with_errors(self, workflow):
        """Test routing from RAG enhancement node with errors."""
        state = VideoGenerationState(
            topic="test",
            description="test",
            session_id="test"
        )
        state.add_error(WorkflowError(
            step="rag_enhancement",
            error_type=ErrorType.MODEL,
            message="Test error"
        ))
        
        route = workflow._route_from_rag_enhancement(state)
        assert route == "error_handler"
    
    def test_route_from_rag_enhancement_interrupted(self, workflow):
        """Test routing from RAG enhancement node when interrupted."""
        state = VideoGenerationState(
            topic="test",
            description="test",
            session_id="test"
        )
        state.workflow_interrupted = True
        
        route = workflow._route_from_rag_enhancement(state)
        assert route == "human_loop"
    
    def test_route_from_visual_analysis_success(self, workflow):
        """Test routing from visual analysis node on success."""
        state = VideoGenerationState(
            topic="test",
            description="test",
            session_id="test"
        )
        
        route = workflow._route_from_visual_analysis(state)
        assert route == "complete"
    
    def test_route_from_human_loop_planning_needed(self, workflow):
        """Test routing from human loop when planning is needed."""
        state = VideoGenerationState(
            topic="test",
            description="test",
            session_id="test"
        )
        # No scene outline or implementations
        
        route = workflow._route_from_human_loop(state)
        assert route == "planning"
    
    def test_route_from_human_loop_code_generation_needed(self, workflow):
        """Test routing from human loop when code generation is needed."""
        state = VideoGenerationState(
            topic="test",
            description="test",
            session_id="test"
        )
        state.scene_outline = "Test outline"
        state.scene_implementations = {1: "Test implementation", 2: "Test implementation 2"}
        # No generated code
        
        route = workflow._route_from_human_loop(state)
        assert route == "code_generation"
    
    def test_route_from_human_loop_rendering_needed(self, workflow):
        """Test routing from human loop when rendering is needed."""
        state = VideoGenerationState(
            topic="test",
            description="test",
            session_id="test"
        )
        state.scene_outline = "Test outline"
        state.scene_implementations = {1: "Test implementation"}
        state.generated_code = {1: "Test code"}
        # No rendered videos
        
        route = workflow._route_from_human_loop(state)
        assert route == "rendering"
    
    def test_route_from_human_loop_complete(self, workflow):
        """Test routing from human loop when workflow can complete."""
        state = VideoGenerationState(
            topic="test",
            description="test",
            session_id="test"
        )
        state.scene_outline = "Test outline"
        state.scene_implementations = {1: "Test implementation"}
        state.generated_code = {1: "Test code"}
        state.rendered_videos = {1: "/tmp/video.mp4"}
        
        route = workflow._route_from_human_loop(state)
        assert route == "complete"


class TestNodePlaceholders:
    """Test placeholder node implementations."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock workflow configuration."""
        return WorkflowConfig(
            planner_model=ModelConfig(
                provider="openrouter",
                model_name="anthropic/claude-3.5-sonnet"
            ),
            code_model=ModelConfig(
                provider="openrouter",
                model_name="anthropic/claude-3.5-sonnet"
            ),
            helper_model=ModelConfig(
                provider="openrouter",
                model_name="anthropic/claude-3.5-sonnet"
            ),
            output_dir="/tmp/test_output",
            max_retries=3,
            timeout_seconds=300
        )
    
    @pytest.fixture
    def workflow(self, mock_config):
        """Create a workflow instance for testing."""
        return VideoGenerationWorkflow(mock_config)
    
    @pytest.fixture
    def test_state(self):
        """Create a test state."""
        return VideoGenerationState(
            topic="test",
            description="test",
            session_id="test-session"
        )
    
    async def test_rag_enhancement_node_placeholder(self, workflow, test_state):
        """Test RAG enhancement node placeholder implementation."""
        result = await workflow._rag_enhancement_node(test_state)
        
        assert result.current_step == "rag_enhancement"
        assert len(result.execution_trace) > 0
        assert any("rag_enhancement_node" in trace for trace in result.execution_trace)
    
    async def test_visual_analysis_node_placeholder(self, workflow, test_state):
        """Test visual analysis node placeholder implementation."""
        result = await workflow._visual_analysis_node(test_state)
        
        assert result.current_step == "visual_analysis"
        assert len(result.execution_trace) > 0
        assert any("visual_analysis_node" in trace for trace in result.execution_trace)
    
    async def test_human_loop_node_placeholder(self, workflow, test_state):
        """Test human loop node placeholder implementation."""
        # Set up state with pending human input
        test_state.pending_human_input = {"type": "test", "message": "Test intervention"}
        test_state.workflow_interrupted = True
        
        result = await workflow._human_loop_node(test_state)
        
        assert result.current_step == "human_loop"
        assert result.pending_human_input is None
        assert result.workflow_interrupted is False
        assert len(result.execution_trace) > 0
        assert any("human_loop_node" in trace for trace in result.execution_trace)
    
    async def test_complete_node(self, workflow, test_state):
        """Test complete node implementation."""
        # Set up state with some completion data
        test_state.scene_implementations = {1: "Test implementation"}
        test_state.generated_code = {1: "Test code"}
        test_state.rendered_videos = {1: "/tmp/video.mp4"}
        test_state.combined_video_path = "/tmp/combined.mp4"
        
        result = await workflow._complete_node(test_state)
        
        assert result.current_step == "complete"
        assert result.workflow_complete is True
        assert len(result.execution_trace) > 0
        assert any("complete_node" in trace for trace in result.execution_trace)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
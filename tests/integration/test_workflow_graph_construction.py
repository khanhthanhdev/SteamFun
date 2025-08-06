"""
Integration tests for the LangGraph StateGraph workflow construction.

This module tests the complete workflow graph construction, node integration,
and conditional routing logic following the task requirements.
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


class TestStateGraphConstruction:
    """Test StateGraph construction with new node functions."""
    
    @pytest.fixture
    def valid_config(self):
        """Create a valid workflow configuration."""
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
            output_dir="./test_output",
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
    
    def test_workflow_graph_creation(self, valid_config):
        """Test creating workflow graph using LangGraph StateGraph."""
        workflow = VideoGenerationWorkflow(valid_config)
        
        assert workflow.config == valid_config
        assert workflow.graph is not None
        
        # Verify graph has expected nodes
        graph_nodes = list(workflow.graph.nodes.keys())
        expected_nodes = [
            "__start__",  # LangGraph adds this automatically
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
            assert node in graph_nodes, f"Node '{node}' not found in graph"
        
        # Verify total node count
        assert len(graph_nodes) == len(expected_nodes)
    
    def test_workflow_graph_with_checkpointing(self, valid_config):
        """Test creating workflow graph with memory checkpointing."""
        workflow = create_workflow(valid_config, use_checkpointing=True)
        
        assert workflow.config == valid_config
        assert workflow.checkpointer is not None
        assert workflow.graph is not None
        
        # Verify checkpointing is enabled
        assert hasattr(workflow.graph, 'checkpointer')
    
    def test_workflow_graph_without_checkpointing(self, valid_config):
        """Test creating workflow graph without checkpointing."""
        workflow = create_workflow(valid_config, use_checkpointing=False)
        
        assert workflow.config == valid_config
        assert workflow.checkpointer is None
        assert workflow.graph is not None
    
    def test_conditional_edges_implementation(self, valid_config):
        """Test that conditional edges are properly implemented."""
        workflow = VideoGenerationWorkflow(valid_config)
        
        # Verify graph structure includes conditional edges
        # Note: LangGraph doesn't expose edge details directly, but we can verify
        # the graph compiles successfully which means edges are valid
        assert workflow.graph is not None
        
        # Test that routing functions exist and are callable
        assert callable(workflow._route_from_rag_enhancement)
        assert callable(workflow._route_from_visual_analysis)
        assert callable(workflow._route_from_human_loop)
    
    def test_node_functions_integration(self, valid_config):
        """Test that all node functions are properly integrated."""
        workflow = VideoGenerationWorkflow(valid_config)
        
        # Verify that node functions are properly added to the graph
        graph_nodes = workflow.graph.nodes
        
        # Check that core node functions are present
        core_nodes = ["planning", "code_generation", "rendering", "error_handler"]
        for node_name in core_nodes:
            assert node_name in graph_nodes
            # Verify the node exists (LangGraph wraps functions in PregelNode objects)
            assert graph_nodes[node_name] is not None
    
    def test_graph_entry_and_exit_points(self, valid_config):
        """Test that graph has proper entry and exit points."""
        workflow = VideoGenerationWorkflow(valid_config)
        
        # Verify START node connects to planning
        graph_nodes = list(workflow.graph.nodes.keys())
        assert "__start__" in graph_nodes
        
        # The complete node should be terminal (connects to END)
        assert "complete" in graph_nodes


class TestConditionalRoutingIntegration:
    """Test conditional routing logic integration with the graph."""
    
    @pytest.fixture
    def valid_config(self):
        """Create a valid workflow configuration."""
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
            output_dir="./test_output",
            max_retries=3,
            timeout_seconds=300
        )
    
    @pytest.fixture
    def workflow(self, valid_config):
        """Create a workflow instance for testing."""
        return VideoGenerationWorkflow(valid_config)
    
    def test_route_from_rag_enhancement_success(self, workflow):
        """Test routing from RAG enhancement node on success."""
        state = VideoGenerationState(
            topic="test topic",
            description="test description",
            session_id="test-session"
        )
        
        route = workflow._route_from_rag_enhancement(state)
        assert route == "code_generation"
    
    def test_route_from_rag_enhancement_with_errors(self, workflow):
        """Test routing from RAG enhancement node with errors."""
        state = VideoGenerationState(
            topic="test topic",
            description="test description",
            session_id="test-session"
        )
        state.add_error(WorkflowError(
            step="code_generation",
            error_type=ErrorType.MODEL,
            message="Test error"
        ))
        
        route = workflow._route_from_rag_enhancement(state)
        assert route == "error_handler"
    
    def test_route_from_rag_enhancement_interrupted(self, workflow):
        """Test routing from RAG enhancement node when interrupted."""
        state = VideoGenerationState(
            topic="test topic",
            description="test description",
            session_id="test-session"
        )
        state.workflow_interrupted = True
        
        route = workflow._route_from_rag_enhancement(state)
        assert route == "human_loop"
    
    def test_route_from_visual_analysis_success(self, workflow):
        """Test routing from visual analysis node on success."""
        state = VideoGenerationState(
            topic="test topic",
            description="test description",
            session_id="test-session"
        )
        
        route = workflow._route_from_visual_analysis(state)
        assert route == "complete"
    
    def test_route_from_visual_analysis_with_errors(self, workflow):
        """Test routing from visual analysis node with errors."""
        state = VideoGenerationState(
            topic="test topic",
            description="test description",
            session_id="test-session"
        )
        state.add_error(WorkflowError(
            step="visual_analysis",
            error_type=ErrorType.CONTENT,
            message="Visual analysis failed"
        ))
        
        route = workflow._route_from_visual_analysis(state)
        assert route == "error_handler"
    
    def test_route_from_human_loop_planning_needed(self, workflow):
        """Test routing from human loop when planning is needed."""
        state = VideoGenerationState(
            topic="test topic",
            description="test description",
            session_id="test-session"
        )
        # No scene outline or implementations
        
        route = workflow._route_from_human_loop(state)
        assert route == "planning"
    
    def test_route_from_human_loop_code_generation_needed(self, workflow):
        """Test routing from human loop when code generation is needed."""
        state = VideoGenerationState(
            topic="test topic",
            description="test description",
            session_id="test-session"
        )
        state.scene_outline = "Test outline"
        state.scene_implementations = {1: "Test implementation", 2: "Test implementation 2"}
        # No generated code
        
        route = workflow._route_from_human_loop(state)
        assert route == "code_generation"
    
    def test_route_from_human_loop_rendering_needed(self, workflow):
        """Test routing from human loop when rendering is needed."""
        state = VideoGenerationState(
            topic="test topic",
            description="test description",
            session_id="test-session"
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
            topic="test topic",
            description="test description",
            session_id="test-session"
        )
        state.scene_outline = "Test outline"
        state.scene_implementations = {1: "Test implementation"}
        state.generated_code = {1: "Test code"}
        state.rendered_videos = {1: "./test_output/video.mp4"}
        
        route = workflow._route_from_human_loop(state)
        assert route == "complete"


class TestWorkflowConfigurationValidation:
    """Test workflow configuration validation."""
    
    def test_valid_configuration(self):
        """Test workflow configuration validation with valid config."""
        config = WorkflowConfig(
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
            output_dir="./test_output",
            max_retries=3,
            timeout_seconds=300
        )
        
        assert validate_workflow_configuration(config) is True
    
    def test_invalid_retries_configuration(self):
        """Test workflow configuration validation with invalid retry count."""
        config = WorkflowConfig(
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
            output_dir="./test_output",
            max_retries=0,  # Invalid
            timeout_seconds=300
        )
        
        assert validate_workflow_configuration(config) is False
    
    def test_invalid_timeout_configuration(self):
        """Test workflow configuration validation with invalid timeout."""
        config = WorkflowConfig(
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
            output_dir="./test_output",
            max_retries=3,
            timeout_seconds=10  # Invalid - too low
        )
        
        assert validate_workflow_configuration(config) is False


class TestPlaceholderNodeImplementations:
    """Test placeholder node implementations in the workflow."""
    
    @pytest.fixture
    def valid_config(self):
        """Create a valid workflow configuration."""
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
            output_dir="./test_output",
            max_retries=3,
            timeout_seconds=300
        )
    
    @pytest.fixture
    def workflow(self, valid_config):
        """Create a workflow instance for testing."""
        return VideoGenerationWorkflow(valid_config)
    
    @pytest.fixture
    def test_state(self):
        """Create a test state."""
        return VideoGenerationState(
            topic="test topic",
            description="test description",
            session_id="test-session"
        )
    
    @pytest.mark.asyncio
    async def test_rag_enhancement_node_placeholder(self, workflow, test_state):
        """Test RAG enhancement node placeholder implementation."""
        result = await workflow._rag_enhancement_node(test_state)
        
        assert result.current_step == "code_generation"  # RAG enhancement is part of code generation
        assert len(result.execution_trace) > 0
        assert any("rag_enhancement_node" in str(trace) for trace in result.execution_trace)
    
    @pytest.mark.asyncio
    async def test_visual_analysis_node_placeholder(self, workflow, test_state):
        """Test visual analysis node placeholder implementation."""
        result = await workflow._visual_analysis_node(test_state)
        
        assert result.current_step == "visual_analysis"
        assert len(result.execution_trace) > 0
        assert any("visual_analysis_node" in str(trace) for trace in result.execution_trace)
    
    @pytest.mark.asyncio
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
        assert any("human_loop_node" in str(trace) for trace in result.execution_trace)
    
    @pytest.mark.asyncio
    async def test_complete_node(self, workflow, test_state):
        """Test complete node implementation."""
        # Set up state with some completion data
        test_state.scene_implementations = {1: "Test implementation"}
        test_state.generated_code = {1: "Test code"}
        test_state.rendered_videos = {1: "./test_output/video.mp4"}
        test_state.combined_video_path = "./test_output/combined.mp4"
        
        result = await workflow._complete_node(test_state)
        
        assert result.current_step == "complete"
        assert result.workflow_complete is True
        assert len(result.execution_trace) > 0
        assert any("complete_node" in str(trace) for trace in result.execution_trace)


class TestGraphVisualization:
    """Test graph visualization functionality."""
    
    @pytest.fixture
    def valid_config(self):
        """Create a valid workflow configuration."""
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
            output_dir="./test_output",
            max_retries=3,
            timeout_seconds=300
        )
    
    def test_graph_visualization_generation(self, valid_config):
        """Test graph visualization generation."""
        workflow = VideoGenerationWorkflow(valid_config)
        visualization = workflow.get_graph_visualization()
        
        assert isinstance(visualization, str)
        assert "graph TD" in visualization
        assert "planning" in visualization
        assert "code_generation" in visualization
        assert "rendering" in visualization
        assert "error_handler" in visualization
        assert "complete" in visualization


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
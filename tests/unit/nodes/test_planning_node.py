"""
Unit tests for planning_node function.

Tests the planning node implementation following LangGraph patterns
with comprehensive error handling and state management.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.langgraph_agents.nodes.planning_node import planning_node, _build_planning_config, _get_model_wrappers
from src.langgraph_agents.models.state import VideoGenerationState
from src.langgraph_agents.models.config import WorkflowConfig, ModelConfig
from src.langgraph_agents.models.errors import WorkflowError, ErrorType, ErrorSeverity


@pytest.fixture
def sample_state():
    """Create a sample VideoGenerationState for testing."""
    config = WorkflowConfig(
        planner_model=ModelConfig(
            provider="openrouter",
            model_name="anthropic/claude-3.5-sonnet",
            temperature=0.7,
            max_tokens=4000
        ),
        helper_model=ModelConfig(
            provider="openrouter", 
            model_name="anthropic/claude-3.5-sonnet",
            temperature=0.7,
            max_tokens=4000
        )
    )
    
    return VideoGenerationState(
        topic="Python basics",
        description="Introduction to Python programming concepts",
        session_id="test-session-123",
        config=config
    )


@pytest.fixture
def mock_planning_service():
    """Create a mock PlanningService."""
    service = MagicMock()
    service.generate_scene_outline = AsyncMock()
    service.generate_scene_implementations = AsyncMock()
    service.detect_plugins = AsyncMock()
    service.validate_scene_outline = AsyncMock()
    service.validate_scene_implementations = AsyncMock()
    service.get_planning_metrics = MagicMock()
    service.cleanup = AsyncMock()
    return service


class TestPlanningNode:
    """Test cases for planning_node function."""
    
    @pytest.mark.asyncio
    async def test_planning_node_success(self, sample_state, mock_planning_service):
        """Test successful planning node execution."""
        # Setup mock responses
        scene_outline = "Scene 1: Introduction\nScene 2: Basic concepts\nScene 3: Examples"
        scene_implementations = [
            "Implementation for scene 1: Show Python logo and introduction",
            "Implementation for scene 2: Explain variables and data types", 
            "Implementation for scene 3: Show code examples"
        ]
        detected_plugins = ["manim_physics", "manim_slides"]
        
        mock_planning_service.generate_scene_outline.return_value = scene_outline
        mock_planning_service.generate_scene_implementations.return_value = scene_implementations
        mock_planning_service.detect_plugins.return_value = detected_plugins
        mock_planning_service.validate_scene_outline.return_value = (True, [])
        mock_planning_service.validate_scene_implementations.return_value = (True, {})
        mock_planning_service.get_planning_metrics.return_value = {"scenes_generated": 3}
        
        with patch('src.langgraph_agents.nodes.planning_node.PlanningService', return_value=mock_planning_service):
            result_state = await planning_node(sample_state)
        
        # Verify state updates
        assert result_state.current_step == "planning"
        assert result_state.scene_outline == scene_outline
        assert len(result_state.scene_implementations) == 3
        assert result_state.scene_implementations[1] == scene_implementations[0]
        assert result_state.scene_implementations[2] == scene_implementations[1]
        assert result_state.scene_implementations[3] == scene_implementations[2]
        assert result_state.detected_plugins == detected_plugins
        
        # Verify service calls
        mock_planning_service.generate_scene_outline.assert_called_once()
        mock_planning_service.generate_scene_implementations.assert_called_once()
        mock_planning_service.detect_plugins.assert_called_once()
        mock_planning_service.cleanup.assert_called_once()
        
        # Verify execution trace
        assert len(result_state.execution_trace) >= 2  # start and complete
        assert result_state.execution_trace[0]["step"] == "planning_node"
        assert result_state.execution_trace[0]["data"]["action"] == "started"
    
    @pytest.mark.asyncio
    async def test_planning_node_empty_topic(self, sample_state):
        """Test planning node with empty topic."""
        # Create a new state with empty topic by bypassing validation
        from src.langgraph_agents.models.state import VideoGenerationState
        from src.langgraph_agents.models.config import WorkflowConfig, ModelConfig
        
        config = WorkflowConfig(
            planner_model=ModelConfig(provider="openrouter", model_name="anthropic/claude-3.5-sonnet"),
            helper_model=ModelConfig(provider="openrouter", model_name="anthropic/claude-3.5-sonnet")
        )
        
        # Create state with validation disabled temporarily
        empty_topic_state = VideoGenerationState.model_construct(
            topic="",  # Empty topic
            description="Valid description",
            session_id="test-session",
            config=config
        )
        
        result_state = await planning_node(empty_topic_state)
        
        # Verify error was added
        assert len(result_state.errors) == 1
        assert result_state.errors[0].error_type == ErrorType.VALIDATION
        assert "Topic and description are required" in result_state.errors[0].message
        assert result_state.current_step == "planning"
    
    @pytest.mark.asyncio
    async def test_planning_node_empty_description(self, sample_state):
        """Test planning node with empty description."""
        # Create a new state with empty description by bypassing validation
        from src.langgraph_agents.models.state import VideoGenerationState
        from src.langgraph_agents.models.config import WorkflowConfig, ModelConfig
        
        config = WorkflowConfig(
            planner_model=ModelConfig(provider="openrouter", model_name="anthropic/claude-3.5-sonnet"),
            helper_model=ModelConfig(provider="openrouter", model_name="anthropic/claude-3.5-sonnet")
        )
        
        # Create state with validation disabled temporarily
        empty_desc_state = VideoGenerationState.model_construct(
            topic="Valid topic",
            description="",  # Empty description
            session_id="test-session",
            config=config
        )
        
        result_state = await planning_node(empty_desc_state)
        
        # Verify error was added
        assert len(result_state.errors) == 1
        assert result_state.errors[0].error_type == ErrorType.VALIDATION
        assert "Topic and description are required" in result_state.errors[0].message
    
    @pytest.mark.asyncio
    async def test_planning_node_scene_outline_validation_failure(self, sample_state, mock_planning_service):
        """Test planning node with scene outline validation failure."""
        scene_outline = "Invalid outline"
        validation_issues = ["Scene outline is too short", "No scene structure found"]
        
        mock_planning_service.generate_scene_outline.return_value = scene_outline
        mock_planning_service.validate_scene_outline.return_value = (False, validation_issues)
        mock_planning_service.generate_scene_implementations.return_value = ["Implementation 1"]
        mock_planning_service.validate_scene_implementations.return_value = (True, {})
        mock_planning_service.detect_plugins.return_value = []
        
        with patch('src.langgraph_agents.nodes.planning_node.PlanningService', return_value=mock_planning_service):
            result_state = await planning_node(sample_state)
        
        # Verify error was added but processing continued
        assert len(result_state.errors) == 1
        assert result_state.errors[0].error_type == ErrorType.CONTENT
        assert "Scene outline validation failed" in result_state.errors[0].message
        assert result_state.scene_outline == scene_outline  # Still set despite validation failure
    
    @pytest.mark.asyncio
    async def test_planning_node_scene_implementations_validation_failure(self, sample_state, mock_planning_service):
        """Test planning node with scene implementations validation failure."""
        scene_outline = "Valid outline with scenes"
        scene_implementations = ["", "Valid implementation"]  # First is empty
        validation_issues = {1: ["Scene implementation is empty"]}
        
        mock_planning_service.generate_scene_outline.return_value = scene_outline
        mock_planning_service.validate_scene_outline.return_value = (True, [])
        mock_planning_service.generate_scene_implementations.return_value = scene_implementations
        mock_planning_service.validate_scene_implementations.return_value = (False, validation_issues)
        mock_planning_service.detect_plugins.return_value = []
        
        with patch('src.langgraph_agents.nodes.planning_node.PlanningService', return_value=mock_planning_service):
            result_state = await planning_node(sample_state)
        
        # Verify error was added for the invalid scene
        assert len(result_state.errors) == 1
        assert result_state.errors[0].error_type == ErrorType.CONTENT
        assert result_state.errors[0].scene_number == 1
        assert "Scene 1 implementation validation failed" in result_state.errors[0].message
    
    @pytest.mark.asyncio
    async def test_planning_node_plugin_detection_failure(self, sample_state, mock_planning_service):
        """Test planning node with plugin detection failure."""
        scene_outline = "Valid outline"
        scene_implementations = ["Implementation 1"]
        
        mock_planning_service.generate_scene_outline.return_value = scene_outline
        mock_planning_service.validate_scene_outline.return_value = (True, [])
        mock_planning_service.generate_scene_implementations.return_value = scene_implementations
        mock_planning_service.validate_scene_implementations.return_value = (True, {})
        mock_planning_service.detect_plugins.side_effect = Exception("Plugin detection failed")
        
        with patch('src.langgraph_agents.nodes.planning_node.PlanningService', return_value=mock_planning_service):
            result_state = await planning_node(sample_state)
        
        # Verify plugin detection failure didn't stop processing
        assert result_state.detected_plugins == []
        assert result_state.scene_outline == scene_outline
        assert len(result_state.scene_implementations) == 1
        # No error should be added for plugin detection failure (it's not critical)
        assert len(result_state.errors) == 0
    
    @pytest.mark.asyncio
    async def test_planning_node_system_error(self, sample_state, mock_planning_service):
        """Test planning node with system error."""
        mock_planning_service.generate_scene_outline.side_effect = Exception("System failure")
        
        with patch('src.langgraph_agents.nodes.planning_node.PlanningService', return_value=mock_planning_service):
            result_state = await planning_node(sample_state)
        
        # Verify system error was handled
        assert len(result_state.errors) == 1
        assert result_state.errors[0].error_type == ErrorType.SYSTEM
        assert result_state.errors[0].severity == ErrorSeverity.CRITICAL
        assert "Planning failed" in result_state.errors[0].message
        
        # Verify failure trace was added
        failure_traces = [trace for trace in result_state.execution_trace 
                         if trace["data"].get("action") == "failed"]
        assert len(failure_traces) == 1
    
    @pytest.mark.asyncio
    async def test_planning_node_with_metrics(self, sample_state, mock_planning_service):
        """Test planning node with metrics collection."""
        from src.langgraph_agents.models.metrics import PerformanceMetrics
        
        # Add metrics to state
        sample_state.metrics = PerformanceMetrics(session_id=sample_state.session_id)
        
        scene_outline = "Scene outline"
        scene_implementations = ["Implementation 1"]
        planning_metrics = {"scenes_generated": 1, "plugins_detected": 0}
        
        mock_planning_service.generate_scene_outline.return_value = scene_outline
        mock_planning_service.validate_scene_outline.return_value = (True, [])
        mock_planning_service.generate_scene_implementations.return_value = scene_implementations
        mock_planning_service.validate_scene_implementations.return_value = (True, {})
        mock_planning_service.detect_plugins.return_value = []
        mock_planning_service.get_planning_metrics.return_value = planning_metrics
        
        with patch('src.langgraph_agents.nodes.planning_node.PlanningService', return_value=mock_planning_service):
            result_state = await planning_node(sample_state)
        
        # Verify metrics were collected
        mock_planning_service.get_planning_metrics.assert_called_once()
        # Note: The actual metrics integration would depend on the PerformanceMetrics implementation


class TestPlanningNodeHelpers:
    """Test cases for planning node helper functions."""
    
    def test_build_planning_config(self, sample_state):
        """Test building planning configuration from state."""
        config = _build_planning_config(sample_state)
        
        assert config['planner_model'] == "openrouter/anthropic/claude-3.5-sonnet"
        assert config['helper_model'] == "openrouter/anthropic/claude-3.5-sonnet"
        assert config['session_id'] == sample_state.session_id
        assert config['use_rag'] == sample_state.config.use_rag
        assert config['enable_caching'] == sample_state.config.enable_caching
        assert config['print_response'] is False
    
    def test_get_model_wrappers(self, sample_state):
        """Test getting model wrappers from state."""
        wrappers = _get_model_wrappers(sample_state)
        
        assert 'planner_model' in wrappers
        assert 'helper_model' in wrappers
        assert wrappers['planner_model']['provider'] == "openrouter"
        assert wrappers['planner_model']['model_name'] == "anthropic/claude-3.5-sonnet"
        assert wrappers['helper_model']['provider'] == "openrouter"
        assert wrappers['helper_model']['model_name'] == "anthropic/claude-3.5-sonnet"


@pytest.mark.asyncio
async def test_planning_node_integration():
    """Integration test for planning node with realistic data."""
    # Create realistic state
    config = WorkflowConfig(
        planner_model=ModelConfig(
            provider="openrouter",
            model_name="anthropic/claude-3.5-sonnet"
        ),
        helper_model=ModelConfig(
            provider="openrouter",
            model_name="anthropic/claude-3.5-sonnet"
        )
    )
    
    state = VideoGenerationState(
        topic="Machine Learning Basics",
        description="An educational video explaining fundamental machine learning concepts including supervised learning, unsupervised learning, and neural networks.",
        session_id="integration-test-456",
        config=config
    )
    
    # Mock the planning service with realistic responses
    mock_service = MagicMock()
    mock_service.generate_scene_outline = AsyncMock(return_value="""
Scene 1: Introduction to Machine Learning
- Define machine learning and its importance
- Show real-world applications

Scene 2: Supervised Learning
- Explain supervised learning concepts
- Show examples with labeled data

Scene 3: Unsupervised Learning  
- Explain unsupervised learning concepts
- Show clustering and dimensionality reduction

Scene 4: Neural Networks
- Introduce neural network basics
- Show simple network architecture
""")
    
    mock_service.generate_scene_implementations = AsyncMock(return_value=[
        "Create animated introduction with ML definition and applications showcase",
        "Demonstrate supervised learning with classification and regression examples",
        "Show unsupervised learning through clustering visualization",
        "Build simple neural network animation showing forward propagation"
    ])
    
    mock_service.detect_plugins = AsyncMock(return_value=["manim_ml", "manim_data_structures"])
    mock_service.validate_scene_outline = AsyncMock(return_value=(True, []))
    mock_service.validate_scene_implementations = AsyncMock(return_value=(True, {}))
    mock_service.get_planning_metrics = MagicMock(return_value={
        "scenes_generated": 4,
        "plugins_detected": 2,
        "validation_passed": True
    })
    mock_service.cleanup = AsyncMock()
    
    with patch('src.langgraph_agents.nodes.planning_node.PlanningService', return_value=mock_service):
        result_state = await planning_node(state)
    
    # Verify comprehensive results
    assert result_state.current_step == "planning"
    assert result_state.scene_outline is not None
    assert len(result_state.scene_implementations) == 4
    assert len(result_state.detected_plugins) == 2
    assert "manim_ml" in result_state.detected_plugins
    assert len(result_state.errors) == 0
    assert len(result_state.execution_trace) >= 2
    
    # Verify scene implementations are properly indexed
    for i in range(1, 5):
        assert i in result_state.scene_implementations
        assert len(result_state.scene_implementations[i]) > 0
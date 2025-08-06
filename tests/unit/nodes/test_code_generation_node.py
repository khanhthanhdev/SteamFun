"""
Unit tests for code_generation_node function.

Tests the code generation node implementation following LangGraph patterns
with parallel processing and comprehensive error handling.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.langgraph_agents.nodes.code_generation_node import (
    code_generation_node, 
    _validate_input_state, 
    _build_code_generation_config, 
    _get_model_wrappers
)
from src.langgraph_agents.models.state import VideoGenerationState
from src.langgraph_agents.models.config import WorkflowConfig, ModelConfig
from src.langgraph_agents.models.errors import WorkflowError, ErrorType, ErrorSeverity


@pytest.fixture
def sample_state_with_planning():
    """Create a sample VideoGenerationState with planning completed."""
    config = WorkflowConfig(
        code_model=ModelConfig(
            provider="openrouter",
            model_name="anthropic/claude-3.5-sonnet",
            temperature=0.3,
            max_tokens=8000
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
        config=config,
        scene_outline="Scene 1: Introduction\nScene 2: Variables\nScene 3: Functions",
        scene_implementations={
            1: "Show Python logo and introduction text",
            2: "Demonstrate variable assignment and types",
            3: "Create function definition examples"
        }
    )


@pytest.fixture
def mock_code_generation_service():
    """Create a mock CodeGenerationService."""
    service = MagicMock()
    service.generate_scene_code = AsyncMock()
    service.generate_multiple_scenes_parallel = AsyncMock()
    service.validate_generated_code = AsyncMock()
    service.get_code_generation_metrics = MagicMock()
    service.cleanup = AsyncMock()
    return service


class TestCodeGenerationNode:
    """Test cases for code_generation_node function."""
    
    @pytest.mark.asyncio
    async def test_code_generation_node_success_sequential(self, sample_state_with_planning, mock_code_generation_service):
        """Test successful code generation with sequential processing."""
        # Setup mock responses
        generated_code_1 = "from manim import *\n\nclass Scene1(Scene):\n    def construct(self):\n        text = Text('Python')\n        self.play(Write(text))"
        generated_code_2 = "from manim import *\n\nclass Scene2(Scene):\n    def construct(self):\n        var = Text('x = 5')\n        self.play(Write(var))"
        generated_code_3 = "from manim import *\n\nclass Scene3(Scene):\n    def construct(self):\n        func = Text('def hello():')\n        self.play(Write(func))"
        
        mock_code_generation_service.generate_scene_code.side_effect = [
            (generated_code_1, "Response 1"),
            (generated_code_2, "Response 2"),
            (generated_code_3, "Response 3")
        ]
        mock_code_generation_service.validate_generated_code.return_value = (True, [])
        mock_code_generation_service.get_code_generation_metrics.return_value = {"scenes_generated": 3}
        
        # Force sequential processing by setting max_concurrent_scenes to 1
        sample_state_with_planning.config.max_concurrent_scenes = 1
        
        with patch('src.langgraph_agents.nodes.code_generation_node.CodeGenerationService', return_value=mock_code_generation_service):
            result_state = await code_generation_node(sample_state_with_planning)
        
        # Verify state updates
        assert result_state.current_step == "code_generation"
        assert len(result_state.generated_code) == 3
        assert result_state.generated_code[1] == generated_code_1
        assert result_state.generated_code[2] == generated_code_2
        assert result_state.generated_code[3] == generated_code_3
        assert len(result_state.code_errors) == 0
        
        # Verify service calls
        assert mock_code_generation_service.generate_scene_code.call_count == 3
        mock_code_generation_service.cleanup.assert_called_once()
        
        # Verify execution trace
        assert len(result_state.execution_trace) >= 2  # start and complete
        assert result_state.execution_trace[0]["step"] == "code_generation_node"
        assert result_state.execution_trace[0]["data"]["action"] == "started"
    
    @pytest.mark.asyncio
    async def test_code_generation_node_success_parallel(self, sample_state_with_planning, mock_code_generation_service):
        """Test successful code generation with parallel processing."""
        # Setup mock responses for parallel processing
        parallel_results = {
            1: ("from manim import *\nclass Scene1(Scene): pass", "Response 1"),
            2: ("from manim import *\nclass Scene2(Scene): pass", "Response 2"),
            3: ("from manim import *\nclass Scene3(Scene): pass", "Response 3")
        }
        
        mock_code_generation_service.generate_multiple_scenes_parallel.return_value = parallel_results
        mock_code_generation_service.validate_generated_code.return_value = (True, [])
        mock_code_generation_service.get_code_generation_metrics.return_value = {"scenes_generated": 3}
        
        # Enable parallel processing
        sample_state_with_planning.config.max_concurrent_scenes = 3
        
        with patch('src.langgraph_agents.nodes.code_generation_node.CodeGenerationService', return_value=mock_code_generation_service):
            result_state = await code_generation_node(sample_state_with_planning)
        
        # Verify parallel processing was used
        mock_code_generation_service.generate_multiple_scenes_parallel.assert_called_once()
        
        # Verify state updates
        assert len(result_state.generated_code) == 3
        assert len(result_state.code_errors) == 0
        
        # Verify completion trace indicates parallel processing
        completion_traces = [trace for trace in result_state.execution_trace 
                           if trace["data"].get("action") == "completed"]
        assert len(completion_traces) == 1
        assert completion_traces[0]["data"]["parallel_processing"] is True
    
    @pytest.mark.asyncio
    async def test_code_generation_node_missing_scene_outline(self, sample_state_with_planning):
        """Test code generation node with missing scene outline."""
        # Remove scene outline
        sample_state_with_planning.scene_outline = None
        
        result_state = await code_generation_node(sample_state_with_planning)
        
        # Verify error was added
        assert len(result_state.errors) == 1
        assert result_state.errors[0].error_type == ErrorType.VALIDATION
        assert "Scene outline is required" in result_state.errors[0].message
        assert result_state.current_step == "code_generation"
    
    @pytest.mark.asyncio
    async def test_code_generation_node_missing_scene_implementations(self, sample_state_with_planning):
        """Test code generation node with missing scene implementations."""
        # Remove scene implementations
        sample_state_with_planning.scene_implementations = {}
        
        result_state = await code_generation_node(sample_state_with_planning)
        
        # Verify error was added
        assert len(result_state.errors) == 1
        assert result_state.errors[0].error_type == ErrorType.VALIDATION
        assert "Scene implementations are required" in result_state.errors[0].message
    
    @pytest.mark.asyncio
    async def test_code_generation_node_empty_scene_implementation(self, sample_state_with_planning):
        """Test code generation node with empty scene implementation."""
        # Make one scene implementation empty
        sample_state_with_planning.scene_implementations[2] = ""
        
        result_state = await code_generation_node(sample_state_with_planning)
        
        # Verify error was added
        assert len(result_state.errors) == 1
        assert result_state.errors[0].error_type == ErrorType.VALIDATION
        assert result_state.errors[0].scene_number == 2
        assert "Scene 2 implementation is empty" in result_state.errors[0].message
    
    @pytest.mark.asyncio
    async def test_code_generation_node_generation_failure(self, sample_state_with_planning, mock_code_generation_service):
        """Test code generation node with generation failure."""
        # Setup mock to fail for one scene
        mock_code_generation_service.generate_scene_code.side_effect = [
            ("valid code", "response"),
            Exception("Generation failed"),
            ("valid code", "response")
        ]
        mock_code_generation_service.validate_generated_code.return_value = (True, [])
        
        # Force sequential processing
        sample_state_with_planning.config.max_concurrent_scenes = 1
        
        with patch('src.langgraph_agents.nodes.code_generation_node.CodeGenerationService', return_value=mock_code_generation_service):
            result_state = await code_generation_node(sample_state_with_planning)
        
        # Verify partial success
        assert len(result_state.generated_code) == 2  # Two successful generations
        assert len(result_state.code_errors) == 1  # One failure
        assert 2 in result_state.code_errors  # Scene 2 failed
        
        # Verify error was added
        generation_errors = [e for e in result_state.errors if e.error_type == ErrorType.MODEL]
        assert len(generation_errors) == 1
        assert generation_errors[0].scene_number == 2
    
    @pytest.mark.asyncio
    async def test_code_generation_node_validation_failure(self, sample_state_with_planning, mock_code_generation_service):
        """Test code generation node with code validation failure."""
        # Setup mock responses
        invalid_code = "invalid python code without manim"
        validation_issues = ["Code lacks Manim-specific elements", "Code has syntax errors"]
        
        mock_code_generation_service.generate_scene_code.return_value = (invalid_code, "response")
        mock_code_generation_service.validate_generated_code.return_value = (False, validation_issues)
        
        # Force sequential processing with single scene
        sample_state_with_planning.config.max_concurrent_scenes = 1
        sample_state_with_planning.scene_implementations = {1: "Test implementation"}
        
        with patch('src.langgraph_agents.nodes.code_generation_node.CodeGenerationService', return_value=mock_code_generation_service):
            result_state = await code_generation_node(sample_state_with_planning)
        
        # Verify code was still stored despite validation failure
        assert len(result_state.generated_code) == 1
        assert result_state.generated_code[1] == invalid_code
        
        # Verify validation error was added
        validation_errors = [e for e in result_state.errors if e.error_type == ErrorType.CONTENT]
        assert len(validation_errors) == 1
        assert "validation failed" in validation_errors[0].message.lower()
    
    @pytest.mark.asyncio
    async def test_code_generation_node_parallel_fallback(self, sample_state_with_planning, mock_code_generation_service):
        """Test code generation node parallel processing fallback to sequential."""
        # Setup parallel processing to fail, sequential to succeed
        mock_code_generation_service.generate_multiple_scenes_parallel.side_effect = Exception("Parallel failed")
        mock_code_generation_service.generate_scene_code.return_value = ("valid code", "response")
        mock_code_generation_service.validate_generated_code.return_value = (True, [])
        
        # Enable parallel processing
        sample_state_with_planning.config.max_concurrent_scenes = 3
        
        with patch('src.langgraph_agents.nodes.code_generation_node.CodeGenerationService', return_value=mock_code_generation_service):
            result_state = await code_generation_node(sample_state_with_planning)
        
        # Verify fallback to sequential processing worked
        assert len(result_state.generated_code) == 3
        assert mock_code_generation_service.generate_scene_code.call_count == 3
    
    @pytest.mark.asyncio
    async def test_code_generation_node_system_error(self, sample_state_with_planning, mock_code_generation_service):
        """Test code generation node with system error."""
        mock_code_generation_service.generate_scene_code.side_effect = Exception("System failure")
        
        # Force sequential processing
        sample_state_with_planning.config.max_concurrent_scenes = 1
        
        with patch('src.langgraph_agents.nodes.code_generation_node.CodeGenerationService', side_effect=Exception("Service init failed")):
            result_state = await code_generation_node(sample_state_with_planning)
        
        # Verify system error was handled
        assert len(result_state.errors) == 1
        assert result_state.errors[0].error_type == ErrorType.SYSTEM
        assert result_state.errors[0].severity == ErrorSeverity.CRITICAL
        assert "Code generation failed" in result_state.errors[0].message
        
        # Verify failure trace was added
        failure_traces = [trace for trace in result_state.execution_trace 
                         if trace["data"].get("action") == "failed"]
        assert len(failure_traces) == 1
    
    @pytest.mark.asyncio
    async def test_code_generation_node_with_metrics(self, sample_state_with_planning, mock_code_generation_service):
        """Test code generation node with metrics collection."""
        from src.langgraph_agents.models.metrics import PerformanceMetrics
        
        # Add metrics to state
        sample_state_with_planning.metrics = PerformanceMetrics(session_id=sample_state_with_planning.session_id)
        
        # Setup mock responses
        mock_code_generation_service.generate_scene_code.return_value = ("valid code", "response")
        mock_code_generation_service.validate_generated_code.return_value = (True, [])
        mock_code_generation_service.get_code_generation_metrics.return_value = {
            "scenes_generated": 3,
            "total_tokens": 1500,
            "generation_time": 45.2
        }
        
        # Force sequential processing
        sample_state_with_planning.config.max_concurrent_scenes = 1
        
        with patch('src.langgraph_agents.nodes.code_generation_node.CodeGenerationService', return_value=mock_code_generation_service):
            result_state = await code_generation_node(sample_state_with_planning)
        
        # Verify metrics were collected
        mock_code_generation_service.get_code_generation_metrics.assert_called_once()


class TestCodeGenerationNodeHelpers:
    """Test cases for code generation node helper functions."""
    
    def test_validate_input_state_success(self, sample_state_with_planning):
        """Test successful input state validation."""
        error = _validate_input_state(sample_state_with_planning)
        assert error is None
    
    def test_validate_input_state_missing_outline(self, sample_state_with_planning):
        """Test input state validation with missing scene outline."""
        sample_state_with_planning.scene_outline = None
        
        error = _validate_input_state(sample_state_with_planning)
        assert error is not None
        assert error.error_type == ErrorType.VALIDATION
        assert "Scene outline is required" in error.message
    
    def test_validate_input_state_missing_implementations(self, sample_state_with_planning):
        """Test input state validation with missing scene implementations."""
        sample_state_with_planning.scene_implementations = {}
        
        error = _validate_input_state(sample_state_with_planning)
        assert error is not None
        assert error.error_type == ErrorType.VALIDATION
        assert "Scene implementations are required" in error.message
    
    def test_validate_input_state_empty_implementation(self, sample_state_with_planning):
        """Test input state validation with empty scene implementation."""
        sample_state_with_planning.scene_implementations[2] = ""
        
        error = _validate_input_state(sample_state_with_planning)
        assert error is not None
        assert error.error_type == ErrorType.VALIDATION
        assert error.scene_number == 2
        assert "Scene 2 implementation is empty" in error.message
    
    def test_build_code_generation_config(self, sample_state_with_planning):
        """Test building code generation configuration from state."""
        config = _build_code_generation_config(sample_state_with_planning)
        
        assert config['scene_model'] == "openrouter/anthropic/claude-3.5-sonnet"
        assert config['helper_model'] == "openrouter/anthropic/claude-3.5-sonnet"
        assert config['session_id'] == sample_state_with_planning.session_id
        assert config['use_rag'] == sample_state_with_planning.config.use_rag
        assert config['enable_caching'] == sample_state_with_planning.config.enable_caching
        assert config['use_visual_fix_code'] == sample_state_with_planning.config.use_visual_analysis
        assert config['print_response'] is False
    
    def test_get_model_wrappers(self, sample_state_with_planning):
        """Test getting model wrappers from state."""
        wrappers = _get_model_wrappers(sample_state_with_planning)
        
        assert 'scene_model' in wrappers
        assert 'helper_model' in wrappers
        assert wrappers['scene_model']['provider'] == "openrouter"
        assert wrappers['scene_model']['model_name'] == "anthropic/claude-3.5-sonnet"
        assert wrappers['scene_model']['temperature'] == 0.3  # Code model has different temperature
        assert wrappers['helper_model']['provider'] == "openrouter"
        assert wrappers['helper_model']['model_name'] == "anthropic/claude-3.5-sonnet"


@pytest.mark.asyncio
async def test_code_generation_node_integration():
    """Integration test for code generation node with realistic data."""
    # Create realistic state with planning completed
    config = WorkflowConfig(
        code_model=ModelConfig(
            provider="openrouter",
            model_name="anthropic/claude-3.5-sonnet",
            temperature=0.3,
            max_tokens=8000
        ),
        helper_model=ModelConfig(
            provider="openrouter",
            model_name="anthropic/claude-3.5-sonnet"
        ),
        max_concurrent_scenes=3,
        max_concurrent_renders=3
    )
    
    state = VideoGenerationState(
        topic="Data Structures in Python",
        description="Educational video covering lists, dictionaries, and sets in Python",
        session_id="integration-test-789",
        config=config,
        scene_outline="""
Scene 1: Introduction to Data Structures
- Define data structures and their importance
- Overview of Python's built-in data structures

Scene 2: Lists and Tuples
- Demonstrate list creation and manipulation
- Show tuple immutability and use cases

Scene 3: Dictionaries and Sets
- Explain key-value pairs in dictionaries
- Show set operations and uniqueness
""",
        scene_implementations={
            1: "Create animated introduction showing different data structure types with visual representations",
            2: "Build interactive examples of list operations like append, remove, and slicing with animated elements",
            3: "Demonstrate dictionary operations and set mathematics with visual comparisons and Venn diagrams"
        }
    )
    
    # Mock the code generation service with realistic responses
    mock_service = MagicMock()
    
    # Setup parallel processing response
    realistic_code_results = {
        1: ("""from manim import *

class DataStructuresIntro(Scene):
    def construct(self):
        title = Text("Data Structures in Python", font_size=48)
        subtitle = Text("Lists, Dictionaries, and Sets", font_size=32)
        
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.shift(UP * 2))
        self.play(Write(subtitle))
        self.wait(2)""", "Generated introduction scene"),
        
        2: ("""from manim import *

class ListsAndTuples(Scene):
    def construct(self):
        # Create list visualization
        list_title = Text("Lists in Python", font_size=36)
        list_example = Code(
            code="my_list = [1, 2, 3, 4]\\nmy_list.append(5)",
            language="python",
            font_size=24
        )
        
        self.play(Write(list_title))
        self.play(list_title.animate.shift(UP * 2))
        self.play(Create(list_example))
        self.wait(2)""", "Generated lists and tuples scene"),
        
        3: ("""from manim import *

class DictionariesAndSets(Scene):
    def construct(self):
        # Dictionary example
        dict_title = Text("Dictionaries and Sets", font_size=36)
        dict_code = Code(
            code="student = {'name': 'Alice', 'age': 20}\\nunique_numbers = {1, 2, 3, 3, 4}",
            language="python",
            font_size=24
        )
        
        self.play(Write(dict_title))
        self.play(dict_title.animate.shift(UP * 2))
        self.play(Create(dict_code))
        self.wait(2)""", "Generated dictionaries and sets scene")
    }
    
    mock_service.generate_multiple_scenes_parallel = AsyncMock(return_value=realistic_code_results)
    mock_service.validate_generated_code = AsyncMock(return_value=(True, []))
    mock_service.get_code_generation_metrics = MagicMock(return_value={
        "scenes_generated": 3,
        "total_tokens": 2400,
        "average_generation_time": 12.5,
        "parallel_processing": True,
        "cache_hits": 0
    })
    mock_service.cleanup = AsyncMock()
    
    with patch('src.langgraph_agents.nodes.code_generation_node.CodeGenerationService', return_value=mock_service):
        result_state = await code_generation_node(state)
    
    # Verify comprehensive results
    assert result_state.current_step == "code_generation"
    assert len(result_state.generated_code) == 3
    assert len(result_state.code_errors) == 0
    assert len(result_state.errors) == 0
    
    # Verify all scenes have generated code
    for scene_num in [1, 2, 3]:
        assert scene_num in result_state.generated_code
        assert "from manim import *" in result_state.generated_code[scene_num]
        assert "class" in result_state.generated_code[scene_num]
        assert "Scene" in result_state.generated_code[scene_num]
    
    # Verify execution trace
    assert len(result_state.execution_trace) >= 2
    completion_traces = [trace for trace in result_state.execution_trace 
                        if trace["data"].get("action") == "completed"]
    assert len(completion_traces) == 1
    assert completion_traces[0]["data"]["successful_generations"] == 3
    assert completion_traces[0]["data"]["parallel_processing"] is True
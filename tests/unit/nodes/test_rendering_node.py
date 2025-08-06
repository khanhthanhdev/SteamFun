"""
Unit tests for rendering_node function.

Tests the rendering node implementation following LangGraph patterns
with resource management and comprehensive error handling.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.langgraph_agents.nodes.rendering_node import (
    rendering_node, 
    _validate_input_state, 
    _build_rendering_config, 
    _create_scene_configs
)
from src.langgraph_agents.models.state import VideoGenerationState
from src.langgraph_agents.models.config import WorkflowConfig, ModelConfig
from src.langgraph_agents.models.errors import WorkflowError, ErrorType, ErrorSeverity


@pytest.fixture
def sample_state_with_code():
    """Create a sample VideoGenerationState with code generation completed."""
    config = WorkflowConfig(
        max_concurrent_renders=2,
        default_quality="medium",
        use_gpu_acceleration=False,
        preview_mode=False,
        use_visual_analysis=False
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
        },
        generated_code={
            1: "from manim import *\n\nclass Scene1(Scene):\n    def construct(self):\n        text = Text('Python')\n        self.play(Write(text))",
            2: "from manim import *\n\nclass Scene2(Scene):\n    def construct(self):\n        var = Text('x = 5')\n        self.play(Write(var))",
            3: "from manim import *\n\nclass Scene3(Scene):\n    def construct(self):\n        func = Text('def hello():')\n        self.play(Write(func))"
        }
    )


@pytest.fixture
def mock_rendering_service():
    """Create a mock RenderingService."""
    service = MagicMock()
    service.render_scene = AsyncMock()
    service.render_multiple_scenes_parallel = AsyncMock()
    service.find_rendered_video = MagicMock()
    service.combine_videos = AsyncMock()
    service.get_performance_stats = MagicMock()
    service.cleanup = AsyncMock()
    return service


class TestRenderingNode:
    """Test cases for rendering_node function."""
    
    @pytest.mark.asyncio
    async def test_rendering_node_success_sequential(self, sample_state_with_code, mock_rendering_service):
        """Test successful rendering with sequential processing."""
        # Setup mock responses
        video_paths = [
            "/output/test-session-123_python_basics/scene1/media/videos/test-session-123_python_basics_scene1_v1/1080p60/Scene1.mp4",
            "/output/test-session-123_python_basics/scene2/media/videos/test-session-123_python_basics_scene2_v1/1080p60/Scene2.mp4",
            "/output/test-session-123_python_basics/scene3/media/videos/test-session-123_python_basics_scene3_v1/1080p60/Scene3.mp4"
        ]
        
        mock_rendering_service.render_scene.side_effect = [
            ("final_code_1", None),  # Success
            ("final_code_2", None),  # Success
            ("final_code_3", None)   # Success
        ]
        
        mock_rendering_service.find_rendered_video.side_effect = video_paths
        mock_rendering_service.combine_videos.return_value = "/output/combined_video.mp4"
        mock_rendering_service.get_performance_stats.return_value = {"renders_completed": 3}
        
        # Force sequential processing
        sample_state_with_code.config.max_concurrent_renders = 1
        
        with patch('src.langgraph_agents.nodes.rendering_node.RenderingService', return_value=mock_rendering_service):
            result_state = await rendering_node(sample_state_with_code)
        
        # Verify state updates
        assert result_state.current_step == "rendering"
        assert len(result_state.rendered_videos) == 3
        assert result_state.rendered_videos[1] == video_paths[0]
        assert result_state.rendered_videos[2] == video_paths[1]
        assert result_state.rendered_videos[3] == video_paths[2]
        assert result_state.combined_video_path == "/output/combined_video.mp4"
        assert len(result_state.rendering_errors) == 0
        
        # Verify service calls
        assert mock_rendering_service.render_scene.call_count == 3
        mock_rendering_service.combine_videos.assert_called_once()
        mock_rendering_service.cleanup.assert_called_once()
        
        # Verify execution trace
        assert len(result_state.execution_trace) >= 2  # start and complete
        assert result_state.execution_trace[0]["step"] == "rendering_node"
        assert result_state.execution_trace[0]["data"]["action"] == "started"
    
    @pytest.mark.asyncio
    async def test_rendering_node_success_parallel(self, sample_state_with_code, mock_rendering_service):
        """Test successful rendering with parallel processing."""
        # Setup mock responses for parallel processing
        parallel_results = [
            ("final_code_1", None),  # Success
            ("final_code_2", None),  # Success
            ("final_code_3", None)   # Success
        ]
        
        video_paths = [
            "/output/scene1.mp4",
            "/output/scene2.mp4",
            "/output/scene3.mp4"
        ]
        
        mock_rendering_service.render_multiple_scenes_parallel.return_value = parallel_results
        mock_rendering_service.find_rendered_video.side_effect = video_paths
        mock_rendering_service.combine_videos.return_value = "/output/combined_video.mp4"
        mock_rendering_service.get_performance_stats.return_value = {"renders_completed": 3}
        
        # Enable parallel processing
        sample_state_with_code.config.max_concurrent_renders = 3
        
        with patch('src.langgraph_agents.nodes.rendering_node.RenderingService', return_value=mock_rendering_service):
            result_state = await rendering_node(sample_state_with_code)
        
        # Verify parallel processing was used
        mock_rendering_service.render_multiple_scenes_parallel.assert_called_once()
        
        # Verify state updates
        assert len(result_state.rendered_videos) == 3
        assert result_state.combined_video_path == "/output/combined_video.mp4"
        assert len(result_state.rendering_errors) == 0
        
        # Verify completion trace indicates parallel processing
        completion_traces = [trace for trace in result_state.execution_trace 
                           if trace["data"].get("action") == "completed"]
        assert len(completion_traces) == 1
        assert completion_traces[0]["data"]["parallel_rendering"] is True
    
    @pytest.mark.asyncio
    async def test_rendering_node_single_video_no_combination(self, sample_state_with_code, mock_rendering_service):
        """Test rendering with single video (no combination needed)."""
        # Setup single scene
        sample_state_with_code.generated_code = {1: "from manim import *\nclass Scene1(Scene): pass"}
        
        mock_rendering_service.render_scene.return_value = ("final_code", None)
        mock_rendering_service.find_rendered_video.return_value = "/output/single_video.mp4"
        mock_rendering_service.get_performance_stats.return_value = {"renders_completed": 1}
        
        # Force sequential processing
        sample_state_with_code.config.max_concurrent_renders = 1
        
        with patch('src.langgraph_agents.nodes.rendering_node.RenderingService', return_value=mock_rendering_service):
            result_state = await rendering_node(sample_state_with_code)
        
        # Verify single video handling
        assert len(result_state.rendered_videos) == 1
        assert result_state.combined_video_path == "/output/single_video.mp4"
        
        # Verify combine_videos was not called
        mock_rendering_service.combine_videos.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_rendering_node_missing_generated_code(self, sample_state_with_code):
        """Test rendering node with missing generated code."""
        # Remove generated code
        sample_state_with_code.generated_code = {}
        
        result_state = await rendering_node(sample_state_with_code)
        
        # Verify error was added
        assert len(result_state.errors) == 1
        assert result_state.errors[0].error_type == ErrorType.VALIDATION
        assert "Generated code is required" in result_state.errors[0].message
        assert result_state.current_step == "rendering"
    
    @pytest.mark.asyncio
    async def test_rendering_node_empty_generated_code(self, sample_state_with_code):
        """Test rendering node with empty generated code."""
        # Make one scene's code empty
        sample_state_with_code.generated_code[2] = ""
        
        result_state = await rendering_node(sample_state_with_code)
        
        # Verify error was added
        assert len(result_state.errors) == 1
        assert result_state.errors[0].error_type == ErrorType.VALIDATION
        assert result_state.errors[0].scene_number == 2
        assert "Generated code for scene 2 is empty" in result_state.errors[0].message
    
    @pytest.mark.asyncio
    async def test_rendering_node_rendering_failure(self, sample_state_with_code, mock_rendering_service):
        """Test rendering node with rendering failure."""
        # Setup mock to fail for one scene
        mock_rendering_service.render_scene.side_effect = [
            ("final_code_1", None),  # Success
            ("final_code_2", "Rendering error occurred"),  # Failure
            ("final_code_3", None)   # Success
        ]
        
        mock_rendering_service.find_rendered_video.side_effect = [
            "/output/scene1.mp4",
            "/output/scene3.mp4"  # Only successful scenes
        ]
        mock_rendering_service.combine_videos.return_value = "/output/combined_video.mp4"
        
        # Force sequential processing
        sample_state_with_code.config.max_concurrent_renders = 1
        
        with patch('src.langgraph_agents.nodes.rendering_node.RenderingService', return_value=mock_rendering_service):
            result_state = await rendering_node(sample_state_with_code)
        
        # Verify partial success
        assert len(result_state.rendered_videos) == 2  # Two successful renders
        assert len(result_state.rendering_errors) == 1  # One failure
        assert 2 in result_state.rendering_errors  # Scene 2 failed
        
        # Verify error was added
        rendering_errors = [e for e in result_state.errors if e.error_type == ErrorType.RENDERING]
        assert len(rendering_errors) == 1
        assert rendering_errors[0].scene_number == 2
    
    @pytest.mark.asyncio
    async def test_rendering_node_video_file_not_found(self, sample_state_with_code, mock_rendering_service):
        """Test rendering node with video file not found after rendering."""
        # Setup rendering success but file not found
        mock_rendering_service.render_scene.return_value = ("final_code", None)
        mock_rendering_service.find_rendered_video.side_effect = FileNotFoundError("Video file not found")
        
        # Single scene for simplicity
        sample_state_with_code.generated_code = {1: "from manim import *\nclass Scene1(Scene): pass"}
        sample_state_with_code.config.max_concurrent_renders = 1
        
        with patch('src.langgraph_agents.nodes.rendering_node.RenderingService', return_value=mock_rendering_service):
            result_state = await rendering_node(sample_state_with_code)
        
        # Verify error handling
        assert len(result_state.rendered_videos) == 0
        assert len(result_state.rendering_errors) == 1
        assert 1 in result_state.rendering_errors
        
        # Verify system error was added
        system_errors = [e for e in result_state.errors if e.error_type == ErrorType.SYSTEM]
        assert len(system_errors) == 1
        assert "Rendered video not found" in system_errors[0].message
    
    @pytest.mark.asyncio
    async def test_rendering_node_video_combination_failure(self, sample_state_with_code, mock_rendering_service):
        """Test rendering node with video combination failure."""
        # Setup successful rendering but combination failure
        mock_rendering_service.render_scene.side_effect = [
            ("final_code_1", None),
            ("final_code_2", None)
        ]
        mock_rendering_service.find_rendered_video.side_effect = [
            "/output/scene1.mp4",
            "/output/scene2.mp4"
        ]
        mock_rendering_service.combine_videos.side_effect = Exception("Combination failed")
        
        # Two scenes to trigger combination
        sample_state_with_code.generated_code = {
            1: "from manim import *\nclass Scene1(Scene): pass",
            2: "from manim import *\nclass Scene2(Scene): pass"
        }
        sample_state_with_code.config.max_concurrent_renders = 1
        
        with patch('src.langgraph_agents.nodes.rendering_node.RenderingService', return_value=mock_rendering_service):
            result_state = await rendering_node(sample_state_with_code)
        
        # Verify individual videos were rendered successfully
        assert len(result_state.rendered_videos) == 2
        
        # Verify combination error was handled
        assert result_state.combined_video_path is None
        
        # Verify system error was added for combination failure
        system_errors = [e for e in result_state.errors if e.error_type == ErrorType.SYSTEM and "combination" in e.message.lower()]
        assert len(system_errors) == 1
    
    @pytest.mark.asyncio
    async def test_rendering_node_parallel_fallback(self, sample_state_with_code, mock_rendering_service):
        """Test rendering node parallel processing fallback to sequential."""
        # Setup parallel processing to fail, sequential to succeed
        mock_rendering_service.render_multiple_scenes_parallel.side_effect = Exception("Parallel failed")
        mock_rendering_service.render_scene.return_value = ("final_code", None)
        mock_rendering_service.find_rendered_video.return_value = "/output/scene.mp4"
        mock_rendering_service.combine_videos.return_value = "/output/combined.mp4"
        
        # Enable parallel processing
        sample_state_with_code.config.max_concurrent_renders = 3
        
        with patch('src.langgraph_agents.nodes.rendering_node.RenderingService', return_value=mock_rendering_service):
            result_state = await rendering_node(sample_state_with_code)
        
        # Verify fallback to sequential processing worked
        assert len(result_state.rendered_videos) == 3
        assert mock_rendering_service.render_scene.call_count == 3
    
    @pytest.mark.asyncio
    async def test_rendering_node_system_error(self, sample_state_with_code, mock_rendering_service):
        """Test rendering node with system error."""
        with patch('src.langgraph_agents.nodes.rendering_node.RenderingService', side_effect=Exception("Service init failed")):
            result_state = await rendering_node(sample_state_with_code)
        
        # Verify system error was handled
        assert len(result_state.errors) == 1
        assert result_state.errors[0].error_type == ErrorType.SYSTEM
        assert result_state.errors[0].severity == ErrorSeverity.CRITICAL
        assert "Rendering failed" in result_state.errors[0].message
        
        # Verify failure trace was added
        failure_traces = [trace for trace in result_state.execution_trace 
                         if trace["data"].get("action") == "failed"]
        assert len(failure_traces) == 1
    
    @pytest.mark.asyncio
    async def test_rendering_node_with_metrics(self, sample_state_with_code, mock_rendering_service):
        """Test rendering node with metrics collection."""
        from src.langgraph_agents.models.metrics import PerformanceMetrics
        
        # Add metrics to state
        sample_state_with_code.metrics = PerformanceMetrics(session_id=sample_state_with_code.session_id)
        
        # Setup mock responses
        mock_rendering_service.render_scene.return_value = ("final_code", None)
        mock_rendering_service.find_rendered_video.return_value = "/output/scene.mp4"
        mock_rendering_service.combine_videos.return_value = "/output/combined.mp4"
        mock_rendering_service.get_performance_stats.return_value = {
            "renders_completed": 3,
            "total_render_time": 120.5,
            "cache_hits": 1
        }
        
        # Single scene for simplicity
        sample_state_with_code.generated_code = {1: "from manim import *\nclass Scene1(Scene): pass"}
        sample_state_with_code.config.max_concurrent_renders = 1
        
        with patch('src.langgraph_agents.nodes.rendering_node.RenderingService', return_value=mock_rendering_service):
            result_state = await rendering_node(sample_state_with_code)
        
        # Verify metrics were collected
        mock_rendering_service.get_performance_stats.assert_called_once()


class TestRenderingNodeHelpers:
    """Test cases for rendering node helper functions."""
    
    def test_validate_input_state_success(self, sample_state_with_code):
        """Test successful input state validation."""
        error = _validate_input_state(sample_state_with_code)
        assert error is None
    
    def test_validate_input_state_missing_code(self, sample_state_with_code):
        """Test input state validation with missing generated code."""
        sample_state_with_code.generated_code = {}
        
        error = _validate_input_state(sample_state_with_code)
        assert error is not None
        assert error.error_type == ErrorType.VALIDATION
        assert "Generated code is required" in error.message
    
    def test_validate_input_state_empty_code(self, sample_state_with_code):
        """Test input state validation with empty generated code."""
        sample_state_with_code.generated_code[2] = ""
        
        error = _validate_input_state(sample_state_with_code)
        assert error is not None
        assert error.error_type == ErrorType.VALIDATION
        assert error.scene_number == 2
        assert "Generated code for scene 2 is empty" in error.message
    
    def test_build_rendering_config(self, sample_state_with_code):
        """Test building rendering configuration from state."""
        config = _build_rendering_config(sample_state_with_code)
        
        assert config['output_dir'] == sample_state_with_code.config.output_dir
        assert config['max_concurrent_renders'] == sample_state_with_code.config.max_concurrent_renders
        assert config['enable_caching'] == sample_state_with_code.config.enable_caching
        assert config['default_quality'] == sample_state_with_code.config.default_quality
        assert config['use_gpu_acceleration'] == sample_state_with_code.config.use_gpu_acceleration
        assert config['preview_mode'] == sample_state_with_code.config.preview_mode
        assert config['print_response'] is False
    
    def test_create_scene_configs(self, sample_state_with_code):
        """Test creating scene configurations from state."""
        configs = _create_scene_configs(sample_state_with_code)
        
        assert len(configs) == 3
        
        # Check first config
        config1 = configs[0]
        assert config1['curr_scene'] == 1
        assert config1['curr_version'] == 1
        assert config1['code'] == sample_state_with_code.generated_code[1]
        assert config1['quality'] == sample_state_with_code.config.default_quality
        assert config1['topic'] == sample_state_with_code.topic
        assert config1['session_id'] == sample_state_with_code.session_id
        assert 'python_basics' in config1['file_prefix'].lower()
        
        # Verify all scenes are included
        scene_nums = [config['curr_scene'] for config in configs]
        assert set(scene_nums) == {1, 2, 3}


@pytest.mark.asyncio
async def test_rendering_node_integration():
    """Integration test for rendering node with realistic data."""
    # Create realistic state with code generation completed
    config = WorkflowConfig(
        max_concurrent_renders=2,
        default_quality="medium",
        use_gpu_acceleration=False,
        preview_mode=False
    )
    
    state = VideoGenerationState(
        topic="Machine Learning Visualization",
        description="Educational video showing ML concepts with visual animations",
        session_id="integration-test-render-456",
        config=config,
        scene_outline="""
Scene 1: Introduction to Machine Learning
Scene 2: Linear Regression Visualization
Scene 3: Neural Network Animation
""",
        scene_implementations={
            1: "Create animated introduction with ML definition and real-world applications",
            2: "Build interactive linear regression with data points and best-fit line animation",
            3: "Construct neural network diagram with forward propagation visualization"
        },
        generated_code={
            1: """from manim import *

class MLIntroduction(Scene):
    def construct(self):
        title = Text("Machine Learning", font_size=48, color=BLUE)
        subtitle = Text("Making Computers Learn", font_size=32, color=WHITE)
        
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.shift(UP * 2))
        self.play(Write(subtitle))
        self.wait(2)
        
        # Show applications
        apps = VGroup(
            Text("Image Recognition", font_size=24),
            Text("Natural Language Processing", font_size=24),
            Text("Recommendation Systems", font_size=24)
        ).arrange(DOWN, buff=0.5)
        
        self.play(subtitle.animate.shift(UP), Create(apps))
        self.wait(3)""",
            
            2: """from manim import *
import numpy as np

class LinearRegression(Scene):
    def construct(self):
        # Create axes
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            x_length=8,
            y_length=6
        )
        
        # Generate sample data
        x_data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        y_data = 2 * x_data + 1 + np.random.normal(0, 0.5, len(x_data))
        
        # Create data points
        points = VGroup(*[
            Dot(axes.coords_to_point(x, y), color=RED)
            for x, y in zip(x_data, y_data)
        ])
        
        # Create regression line
        line = axes.plot(lambda x: 2*x + 1, color=BLUE, stroke_width=3)
        
        self.play(Create(axes))
        self.play(Create(points))
        self.wait(1)
        self.play(Create(line))
        self.wait(2)""",
            
            3: """from manim import *

class NeuralNetwork(Scene):
    def construct(self):
        # Create neural network layers
        input_layer = VGroup(*[Circle(radius=0.3, color=GREEN) for _ in range(3)])
        hidden_layer = VGroup(*[Circle(radius=0.3, color=YELLOW) for _ in range(4)])
        output_layer = VGroup(*[Circle(radius=0.3, color=RED) for _ in range(2)])
        
        input_layer.arrange(DOWN, buff=0.5).shift(LEFT * 4)
        hidden_layer.arrange(DOWN, buff=0.4).shift(LEFT * 1)
        output_layer.arrange(DOWN, buff=0.8).shift(RIGHT * 2)
        
        # Create connections
        connections = VGroup()
        for input_node in input_layer:
            for hidden_node in hidden_layer:
                connections.add(Line(input_node.get_center(), hidden_node.get_center(), stroke_width=1))
        
        for hidden_node in hidden_layer:
            for output_node in output_layer:
                connections.add(Line(hidden_node.get_center(), output_node.get_center(), stroke_width=1))
        
        # Animate network creation
        self.play(Create(input_layer))
        self.play(Create(hidden_layer))
        self.play(Create(output_layer))
        self.play(Create(connections))
        self.wait(2)"""
        }
    )
    
    # Mock the rendering service with realistic responses
    mock_service = MagicMock()
    
    # Setup parallel rendering response
    realistic_render_results = [
        ("final_code_1", None),  # Scene 1 success
        ("final_code_2", None),  # Scene 2 success  
        ("final_code_3", None)   # Scene 3 success
    ]
    
    video_paths = [
        "/output/integration-test-render-456_machine_learning_visualization/scene1/media/videos/integration-test-render-456_machine_learning_visualization_scene1_v1/720p30/MLIntroduction.mp4",
        "/output/integration-test-render-456_machine_learning_visualization/scene2/media/videos/integration-test-render-456_machine_learning_visualization_scene2_v1/720p30/LinearRegression.mp4",
        "/output/integration-test-render-456_machine_learning_visualization/scene3/media/videos/integration-test-render-456_machine_learning_visualization_scene3_v1/720p30/NeuralNetwork.mp4"
    ]
    
    mock_service.render_multiple_scenes_parallel = AsyncMock(return_value=realistic_render_results)
    mock_service.find_rendered_video = MagicMock(side_effect=video_paths)
    mock_service.combine_videos = AsyncMock(return_value="/output/machine_learning_visualization_combined.mp4")
    mock_service.get_performance_stats = MagicMock(return_value={
        "renders_completed": 3,
        "total_render_time": 245.7,
        "parallel_rendering": True,
        "cache_hits": 0,
        "average_render_time": 81.9
    })
    mock_service.cleanup = AsyncMock()
    
    with patch('src.langgraph_agents.nodes.rendering_node.RenderingService', return_value=mock_service):
        result_state = await rendering_node(state)
    
    # Verify comprehensive results
    assert result_state.current_step == "rendering"
    assert len(result_state.rendered_videos) == 3
    assert len(result_state.rendering_errors) == 0
    assert len(result_state.errors) == 0
    assert result_state.combined_video_path == "/output/machine_learning_visualization_combined.mp4"
    
    # Verify all scenes have rendered videos
    for scene_num in [1, 2, 3]:
        assert scene_num in result_state.rendered_videos
        assert result_state.rendered_videos[scene_num].endswith('.mp4')
        assert 'scene' + str(scene_num) in result_state.rendered_videos[scene_num]
    
    # Verify execution trace
    assert len(result_state.execution_trace) >= 2
    completion_traces = [trace for trace in result_state.execution_trace 
                        if trace["data"].get("action") == "completed"]
    assert len(completion_traces) == 1
    assert completion_traces[0]["data"]["successful_renders"] == 3
    assert completion_traces[0]["data"]["parallel_rendering"] is True
    assert completion_traces[0]["data"]["combined_video"] is True
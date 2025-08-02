"""
Pytest configuration and shared fixtures for LangGraph multi-agent system tests.
"""

import pytest
import asyncio
import os
import sys
from unittest.mock import Mock, patch
from pathlib import Path

# Add src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Suppress warnings during testing
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def mock_environment():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        'OPENROUTER_API_KEY': 'test_openrouter_key',
        'OPENAI_API_KEY': 'test_openai_key',
        'AWS_ACCESS_KEY_ID': 'test_aws_key',
        'AWS_SECRET_ACCESS_KEY': 'test_aws_secret',
        'LANGFUSE_SECRET_KEY': 'test_langfuse_secret',
        'LANGFUSE_PUBLIC_KEY': 'test_langfuse_public',
        'LANGFUSE_HOST': 'https://test-langfuse.com'
    }):
        yield


@pytest.fixture(autouse=True)
def mock_external_dependencies():
    """Mock external dependencies that may not be available in test environment."""
    with patch('src.core.video_planner.EnhancedVideoPlanner'), \
         patch('src.core.code_generator.CodeGenerator'), \
         patch('src.core.video_renderer.OptimizedVideoRenderer'), \
         patch('src.rag.rag_integration.RAGIntegration'), \
         patch('mllm_tools.litellm.LiteLLMWrapper'), \
         patch('mllm_tools.openrouter.OpenRouterWrapper'):
        yield


@pytest.fixture
def mock_langfuse_service():
    """Mock LangFuse service for testing."""
    mock_service = Mock()
    mock_service.is_enabled.return_value = False
    mock_service.trace_agent_execution = Mock()
    mock_service.track_performance_metrics = Mock()
    mock_service.track_error = Mock()
    
    with patch('src.langgraph_agents.base_agent.get_langfuse_service', return_value=mock_service):
        yield mock_service


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory for tests."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return str(output_dir)


@pytest.fixture
def sample_video_data():
    """Sample video generation data for testing."""
    return {
        "topic": "Python Programming Basics",
        "description": "An educational video covering Python fundamentals including variables, functions, and control structures.",
        "scene_outline": """# Scene 1: Introduction to Python
Brief introduction to Python programming language and its applications.

# Scene 2: Variables and Data Types
Explanation of Python variables, strings, numbers, and basic data types.

# Scene 3: Functions and Control Structures
Overview of Python functions, if statements, and loops.""",
        "scene_implementations": {
            1: "Scene 1: Display Python logo with animated text introduction. Show key applications like web development, data science, and automation.",
            2: "Scene 2: Create animated examples of variable assignment. Show different data types with visual representations and type checking.",
            3: "Scene 3: Demonstrate function definition and calling. Show if-else statements and for/while loops with visual flow diagrams."
        },
        "detected_plugins": ["text", "code", "math"],
        "generated_code": {
            1: """from manim import *

class IntroScene(Scene):
    def construct(self):
        title = Text("Python Programming", font_size=48)
        subtitle = Text("A Powerful Programming Language", font_size=24)
        subtitle.next_to(title, DOWN)
        
        self.play(Write(title))
        self.play(Write(subtitle))
        self.wait(2)""",
            2: """from manim import *

class VariablesScene(Scene):
    def construct(self):
        title = Text("Variables and Data Types", font_size=36)
        title.to_edge(UP)
        
        code = Code(
            '''
            name = "Python"
            version = 3.9
            is_awesome = True
            ''',
            language="python"
        )
        
        self.play(Write(title))
        self.play(Create(code))
        self.wait(3)"""
        }
    }


@pytest.fixture
def sample_error_scenarios():
    """Sample error scenarios for testing error handling."""
    return {
        "code_syntax_error": {
            "agent": "code_generator_agent",
            "error_type": "SyntaxError",
            "message": "invalid syntax in generated Manim code",
            "context": {"scene_id": 1, "line": 5}
        },
        "rendering_timeout": {
            "agent": "renderer_agent",
            "error_type": "TimeoutError", 
            "message": "video rendering process timed out after 300 seconds",
            "context": {"scene_id": 2, "quality": "high"}
        },
        "rag_query_failure": {
            "agent": "rag_agent",
            "error_type": "ConnectionError",
            "message": "failed to connect to vector database",
            "context": {"query": "manim animations", "database": "chroma"}
        },
        "visual_analysis_failure": {
            "agent": "visual_analysis_agent",
            "error_type": "ProcessingError",
            "message": "unable to analyze rendered video frames",
            "context": {"video_path": "/tmp/scene1.mp4", "frame_count": 0}
        }
    }


class AsyncMock(Mock):
    """Enhanced AsyncMock for better async testing support."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
    
    def __await__(self):
        return self().__await__()


# Make AsyncMock available for all tests
pytest.AsyncMock = AsyncMock


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        # Mark async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)
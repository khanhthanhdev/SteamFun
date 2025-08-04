"""
Unit tests for CodeGeneratorAgent.
Tests Manim code generation, error handling, and RAG integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from app.core.agents.agents.code_generator_agent import CodeGeneratorAgent
from src.langgraph_agents.state import VideoGenerationState, AgentConfig
from langgraph.types import Command

# Import test utilities
from tests.utils.config_mocks import (
    mock_configuration_manager, create_test_system_config, 
    create_test_agent_config, MockConfigurationManager
)


class TestCodeGeneratorAgent:
    """Test suite for CodeGeneratorAgent functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock agent configuration for CodeGeneratorAgent using configuration system."""
        return create_test_agent_config(
            name="code_generator_agent",
            scene_model="openai/gpt-4o",
            helper_model="openai/gpt-4o-mini",
            tools=["code_generation_tool", "rag_tool"]
        )
    
    @pytest.fixture
    def mock_system_config(self):
        """Create mock system configuration using configuration system."""
        return create_test_system_config()
    
    @pytest.fixture
    def mock_state(self):
        """Create mock video generation state with planning data."""
        return VideoGenerationState(
            messages=[],
            topic="Python programming basics",
            description="Educational video about Python fundamentals",
            session_id="test_session_123",
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
            scene_outline="# Scene 1\nIntroduction\n# Scene 2\nBasic syntax",
            scene_implementations={
                1: "Scene 1: Show Python logo and introduction text",
                2: "Scene 2: Display code examples with syntax highlighting"
            },
            detected_plugins=["text", "code"],
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
    def code_generator_agent(self, mock_config, mock_system_config):
        """Create CodeGeneratorAgent instance for testing."""
        with mock_configuration_manager(mock_system_config):
            return CodeGeneratorAgent(mock_config, mock_system_config)
    
    @pytest.fixture
    def mock_code_generator(self):
        """Create mock CodeGenerator."""
        mock_generator = Mock()
        mock_generator.generate_manim_code = AsyncMock(return_value={
            1: "from manim import *\n\nclass Scene1(Scene):\n    def construct(self):\n        text = Text('Python')\n        self.play(Write(text))",
            2: "from manim import *\n\nclass Scene2(Scene):\n    def construct(self):\n        code = Code('print(\"Hello World\")')\n        self.play(Create(code))"
        })
        mock_generator.fix_code_errors = AsyncMock(return_value={})
        mock_generator.visual_self_reflection = AsyncMock(return_value={})
        return mock_generator
    
    def test_code_generator_agent_initialization(self, code_generator_agent, mock_config):
        """Test CodeGeneratorAgent initialization."""
        assert code_generator_agent.name == "code_generator_agent"
        assert code_generator_agent.scene_model == "openai/gpt-4o"
        assert code_generator_agent.helper_model == "openai/gpt-4o-mini"
        assert code_generator_agent._code_generator is None  # Lazy initialization


if __name__ == "__main__":
    pytest.main([__file__])
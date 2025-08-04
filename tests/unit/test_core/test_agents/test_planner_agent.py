"""
Unit tests for PlannerAgent.
Tests video planning capabilities, scene outline generation, and plugin detection.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from app.core.agents.agents.planner_agent import PlannerAgent
from src.langgraph_agents.state import VideoGenerationState, AgentConfig
from langgraph.types import Command

# Import test utilities
from tests.utils.config_mocks import (
    mock_configuration_manager, create_test_system_config, 
    create_test_agent_config, MockConfigurationManager
)


class TestPlannerAgent:
    """Test suite for PlannerAgent functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock agent configuration for PlannerAgent using configuration system."""
        return create_test_agent_config(
            name="planner_agent",
            planner_model="openai/gpt-4o",
            helper_model="openai/gpt-4o-mini",
            tools=["planning_tool"]
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
    def planner_agent(self, mock_config, mock_system_config):
        """Create PlannerAgent instance for testing."""
        with mock_configuration_manager(mock_system_config):
            return PlannerAgent(mock_config, mock_system_config)
    
    @pytest.fixture
    def mock_video_planner(self):
        """Create mock EnhancedVideoPlanner."""
        mock_planner = Mock()
        mock_planner.generate_scene_outline = AsyncMock(return_value="# Scene 1\nIntroduction to Python\n\n# Scene 2\nBasic syntax")
        mock_planner.generate_scene_implementation_concurrently_enhanced = AsyncMock(return_value=[
            "Scene 1: Show Python logo and introduction text",
            "Scene 2: Display code examples with syntax highlighting"
        ])
        mock_planner.relevant_plugins = ["text", "code"]
        return mock_planner
    
    def test_planner_agent_initialization(self, planner_agent, mock_config):
        """Test PlannerAgent initialization."""
        assert planner_agent.name == "planner_agent"
        assert planner_agent.planner_model == "openai/gpt-4o"
        assert planner_agent.helper_model == "openai/gpt-4o-mini"
        assert planner_agent._video_planner is None  # Lazy initialization
    
    @patch('app.core.agents.agents.planner_agent.EnhancedVideoPlanner')
    def test_get_video_planner_creation(self, mock_planner_class, planner_agent, mock_state):
        """Test video planner creation with state configuration."""
        mock_instance = Mock()
        mock_planner_class.return_value = mock_instance
        
        with patch.object(planner_agent, 'get_model_wrapper') as mock_get_wrapper:
            mock_wrapper = Mock()
            mock_get_wrapper.return_value = mock_wrapper
            
            planner = planner_agent._get_video_planner(mock_state)
            
            # Verify planner was created with correct configuration
            mock_planner_class.assert_called_once()
            call_kwargs = mock_planner_class.call_args[1]
            assert call_kwargs['output_dir'] == 'test_output'
            assert call_kwargs['use_rag'] is True
            assert call_kwargs['session_id'] == 'test_session_123'
            assert call_kwargs['max_scene_concurrency'] == 5
            
            assert planner == mock_instance
            assert planner_agent._video_planner == mock_instance
    
    @patch('app.core.agents.agents.planner_agent.EnhancedVideoPlanner')
    def test_get_video_planner_reuse(self, mock_planner_class, planner_agent, mock_state):
        """Test video planner instance reuse."""
        mock_instance = Mock()
        planner_agent._video_planner = mock_instance
        
        planner = planner_agent._get_video_planner(mock_state)
        
        # Should not create new instance
        mock_planner_class.assert_not_called()
        assert planner == mock_instance
    
    @pytest.mark.asyncio
    async def test_execute_success_full_workflow(self, planner_agent, mock_state, mock_video_planner):
        """Test successful execution of full planning workflow."""
        with patch.object(planner_agent, '_get_video_planner', return_value=mock_video_planner):
            command = await planner_agent.execute(mock_state)
            
            # Verify scene outline generation was called
            mock_video_planner.generate_scene_outline.assert_called_once_with(
                topic="Python programming basics",
                description="Educational video about Python fundamentals",
                session_id="test_session_123"
            )
            
            # Verify scene implementations generation was called
            mock_video_planner.generate_scene_implementation_concurrently_enhanced.assert_called_once()
            
            # Verify command structure
            assert command.goto == "code_generator_agent"
            assert command.update["scene_outline"] == "# Scene 1\nIntroduction to Python\n\n# Scene 2\nBasic syntax"
            assert command.update["scene_implementations"] == {
                1: "Scene 1: Show Python logo and introduction text",
                2: "Scene 2: Display code examples with syntax highlighting"
            }
            assert command.update["detected_plugins"] == ["text", "code"]
            assert command.update["current_agent"] == "code_generator_agent"


if __name__ == "__main__":
    pytest.main([__file__])
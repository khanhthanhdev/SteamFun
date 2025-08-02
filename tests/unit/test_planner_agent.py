"""
Unit tests for PlannerAgent.
Tests video planning capabilities, scene outline generation, and plugin detection.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from src.langgraph_agents.agents.planner_agent import PlannerAgent
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
    
    @patch('src.langgraph_agents.agents.planner_agent.EnhancedVideoPlanner')
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
    
    @patch('src.langgraph_agents.agents.planner_agent.EnhancedVideoPlanner')
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
    
    @pytest.mark.asyncio
    async def test_execute_plan_only_mode(self, planner_agent, mock_state, mock_video_planner):
        """Test execution in plan-only mode."""
        mock_state["only_plan"] = True
        
        with patch.object(planner_agent, '_get_video_planner', return_value=mock_video_planner):
            command = await planner_agent.execute(mock_state)
            
            # Should only generate outline, not implementations
            mock_video_planner.generate_scene_outline.assert_called_once()
            mock_video_planner.generate_scene_implementation_concurrently_enhanced.assert_not_called()
            
            # Should end workflow
            assert command.goto == "END"
            assert command.update["workflow_complete"] is True
    
    @pytest.mark.asyncio
    async def test_execute_scene_outline_failure(self, planner_agent, mock_state, mock_video_planner):
        """Test handling of scene outline generation failure."""
        mock_video_planner.generate_scene_outline.side_effect = Exception("Planning failed")
        
        with patch.object(planner_agent, '_get_video_planner', return_value=mock_video_planner):
            command = await planner_agent.execute(mock_state)
            
            # Should route to error handler
            assert command.goto == "error_handler_agent"
            assert command.update["error_count"] == 1
            assert len(command.update["escalated_errors"]) == 1
    
    @pytest.mark.asyncio
    async def test_execute_empty_scene_outline(self, planner_agent, mock_state, mock_video_planner):
        """Test handling of empty scene outline."""
        mock_video_planner.generate_scene_outline.return_value = None
        
        with patch.object(planner_agent, '_get_video_planner', return_value=mock_video_planner):
            command = await planner_agent.execute(mock_state)
            
            # Should route to error handler due to ValueError
            assert command.goto == "error_handler_agent"
    
    @pytest.mark.asyncio
    async def test_execute_human_escalation(self, mock_config, mock_system_config, mock_state, mock_video_planner):
        """Test human escalation when enabled and error threshold reached."""
        mock_config.enable_human_loop = True
        mock_state["error_count"] = 5  # Above threshold
        
        agent = PlannerAgent(mock_config, mock_system_config)
        mock_video_planner.generate_scene_outline.side_effect = Exception("Planning failed")
        
        with patch.object(agent, '_get_video_planner', return_value=mock_video_planner):
            with patch.object(agent, 'create_human_intervention_command') as mock_human_cmd:
                mock_human_cmd.return_value = Command(goto="human_loop_agent")
                
                command = await agent.execute(mock_state)
                
                mock_human_cmd.assert_called_once()
                assert command.goto == "human_loop_agent"
    
    @pytest.mark.asyncio
    async def test_generate_scene_outline_compatibility(self, planner_agent, mock_state, mock_video_planner):
        """Test scene outline generation method compatibility."""
        with patch.object(planner_agent, '_get_video_planner', return_value=mock_video_planner):
            outline = await planner_agent.generate_scene_outline(
                topic="Test topic",
                description="Test description", 
                session_id="test_session",
                state=mock_state
            )
            
            mock_video_planner.generate_scene_outline.assert_called_once_with(
                "Test topic", "Test description", "test_session"
            )
            assert outline == "# Scene 1\nIntroduction to Python\n\n# Scene 2\nBasic syntax"
    
    @pytest.mark.asyncio
    async def test_generate_scene_implementation_concurrently_compatibility(self, planner_agent, mock_state, mock_video_planner):
        """Test scene implementation generation method compatibility."""
        with patch.object(planner_agent, '_get_video_planner', return_value=mock_video_planner):
            implementations = await planner_agent.generate_scene_implementation_concurrently(
                topic="Test topic",
                description="Test description",
                plan="Test plan",
                session_id="test_session",
                state=mock_state
            )
            
            mock_video_planner.generate_scene_implementation_concurrently_enhanced.assert_called_once_with(
                "Test topic", "Test description", "Test plan", "test_session"
            )
            assert implementations == [
                "Scene 1: Show Python logo and introduction text",
                "Scene 2: Display code examples with syntax highlighting"
            ]
    
    @pytest.mark.asyncio
    async def test_detect_plugins_async(self, planner_agent, mock_state, mock_video_planner):
        """Test plugin detection functionality."""
        mock_video_planner.rag_integration = Mock()
        mock_video_planner._detect_plugins_async = AsyncMock(return_value=["text", "math", "code"])
        
        with patch.object(planner_agent, '_get_video_planner', return_value=mock_video_planner):
            plugins = await planner_agent._detect_plugins_async(
                topic="Math tutorial",
                description="Mathematical concepts",
                state=mock_state
            )
            
            mock_video_planner._detect_plugins_async.assert_called_once_with(
                "Math tutorial", "Mathematical concepts"
            )
            assert plugins == ["text", "math", "code"]
    
    @pytest.mark.asyncio
    async def test_detect_plugins_async_no_rag(self, planner_agent, mock_state, mock_video_planner):
        """Test plugin detection when RAG integration is not available."""
        mock_video_planner.rag_integration = None
        
        with patch.object(planner_agent, '_get_video_planner', return_value=mock_video_planner):
            plugins = await planner_agent._detect_plugins_async(
                topic="Test topic",
                description="Test description",
                state=mock_state
            )
            
            assert plugins == []
    
    def test_get_planning_status(self, planner_agent, mock_state):
        """Test planning status reporting."""
        mock_state.update({
            "scene_outline": "Test outline",
            "scene_implementations": {1: "Scene 1", 2: "Scene 2"},
            "detected_plugins": ["text", "math"]
        })
        
        status = planner_agent.get_planning_status(mock_state)
        
        assert status["agent_name"] == "planner_agent"
        assert status["scene_outline_generated"] is True
        assert status["scene_implementations_count"] == 2
        assert status["detected_plugins"] == ["text", "math"]
        assert status["rag_enabled"] is True
        assert status["enhanced_rag_enabled"] is True
    
    @pytest.mark.asyncio
    async def test_handle_planning_error_with_fallback(self, planner_agent, mock_state):
        """Test planning error handling with fallback strategy."""
        error = Exception("scene_outline generation failed")
        
        with patch.object(planner_agent, 'execute') as mock_execute:
            mock_execute.return_value = Command(goto="code_generator_agent")
            
            command = await planner_agent.handle_planning_error(error, mock_state, retry_with_fallback=True)
            
            # Should retry with fallback configuration
            mock_execute.assert_called_once()
            fallback_state = mock_execute.call_args[0][0]
            assert fallback_state["use_context_learning"] is False
            assert fallback_state["max_scene_concurrency"] == 1
    
    @pytest.mark.asyncio
    async def test_handle_planning_error_fallback_fails(self, planner_agent, mock_state):
        """Test planning error handling when fallback also fails."""
        error = Exception("scene_outline generation failed")
        
        with patch.object(planner_agent, 'execute') as mock_execute:
            mock_execute.side_effect = Exception("Fallback also failed")
            
            with patch.object(planner_agent, 'handle_error') as mock_handle_error:
                mock_handle_error.return_value = Command(goto="error_handler_agent")
                
                command = await planner_agent.handle_planning_error(error, mock_state, retry_with_fallback=True)
                
                mock_handle_error.assert_called_once_with(error, mock_state)
                assert command.goto == "error_handler_agent"
    
    @pytest.mark.asyncio
    async def test_handle_planning_error_no_fallback(self, planner_agent, mock_state):
        """Test planning error handling without fallback."""
        error = Exception("General planning error")
        
        with patch.object(planner_agent, 'handle_error') as mock_handle_error:
            mock_handle_error.return_value = Command(goto="error_handler_agent")
            
            command = await planner_agent.handle_planning_error(error, mock_state, retry_with_fallback=True)
            
            mock_handle_error.assert_called_once_with(error, mock_state)
            assert command.goto == "error_handler_agent"
    
    def test_cleanup_on_destruction(self, planner_agent):
        """Test resource cleanup when agent is destroyed."""
        mock_planner = Mock()
        mock_thread_pool = Mock()
        mock_planner.thread_pool = mock_thread_pool
        planner_agent._video_planner = mock_planner
        
        # Trigger destructor
        planner_agent.__del__()
        
        mock_thread_pool.shutdown.assert_called_once_with(wait=False)
    
    def test_cleanup_on_destruction_no_thread_pool(self, planner_agent):
        """Test cleanup when no thread pool exists."""
        mock_planner = Mock()
        del mock_planner.thread_pool  # No thread_pool attribute
        planner_agent._video_planner = mock_planner
        
        # Should not raise exception
        planner_agent.__del__()


if __name__ == "__main__":
    pytest.main([__file__])
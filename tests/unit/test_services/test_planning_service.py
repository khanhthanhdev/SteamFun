"""
Unit tests for PlanningService.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from src.langgraph_agents.services.planning_service import PlanningService


class TestPlanningService:
    """Test suite for PlanningService functionality."""
    
    @pytest.fixture
    def service_config(self):
        """Create test configuration for PlanningService."""
        return {
            'planner_model': 'test-planner-model',
            'helper_model': 'test-helper-model',
            'output_dir': 'test_output',
            'use_rag': True,
            'use_context_learning': True,
            'context_learning_path': 'test_context',
            'chroma_db_path': 'test_chroma',
            'manim_docs_path': 'test_docs',
            'embedding_model': 'test-embedding',
            'use_langfuse': False,
            'max_scene_concurrency': 3,
            'enable_caching': True,
            'use_enhanced_rag': True,
            'session_id': 'test-session-123'
        }
    
    @pytest.fixture
    def planning_service(self, service_config):
        """Create PlanningService instance for testing."""
        return PlanningService(service_config)
    
    @pytest.fixture
    def mock_model_wrappers(self):
        """Create mock model wrappers."""
        return {
            'planner_model': Mock(),
            'helper_model': Mock()
        }
    
    def test_init(self, service_config):
        """Test PlanningService initialization."""
        service = PlanningService(service_config)
        
        assert service.config == service_config
        assert service.planner_model == 'test-planner-model'
        assert service.helper_model == 'test-helper-model'
        assert service.use_rag is True
        assert service.max_scene_concurrency == 3
        assert service._video_planner is None
    
    @patch('src.langgraph_agents.services.planning_service.EnhancedVideoPlanner')
    def test_get_video_planner_creation(self, mock_planner_class, planning_service, mock_model_wrappers):
        """Test video planner creation."""
        mock_planner_instance = Mock()
        mock_planner_class.return_value = mock_planner_instance
        
        result = planning_service._get_video_planner(mock_model_wrappers)
        
        assert result == mock_planner_instance
        assert planning_service._video_planner == mock_planner_instance
        
        # Verify planner was created with correct arguments
        mock_planner_class.assert_called_once()
        call_args = mock_planner_class.call_args
        assert call_args[1]['planner_model'] == mock_model_wrappers['planner_model']
        assert call_args[1]['helper_model'] == mock_model_wrappers['helper_model']
        assert call_args[1]['use_rag'] is True
        assert call_args[1]['max_scene_concurrency'] == 3
    
    @patch('src.langgraph_agents.services.planning_service.EnhancedVideoPlanner')
    def test_get_video_planner_reuse(self, mock_planner_class, planning_service, mock_model_wrappers):
        """Test video planner instance reuse."""
        mock_planner_instance = Mock()
        mock_planner_class.return_value = mock_planner_instance
        
        # First call creates the planner
        result1 = planning_service._get_video_planner(mock_model_wrappers)
        # Second call should reuse the same instance
        result2 = planning_service._get_video_planner(mock_model_wrappers)
        
        assert result1 == result2 == mock_planner_instance
        # Should only be called once
        mock_planner_class.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_scene_outline_success(self, planning_service, mock_model_wrappers):
        """Test successful scene outline generation."""
        mock_planner = AsyncMock()
        mock_planner.generate_scene_outline.return_value = "Test scene outline with multiple scenes"
        
        with patch.object(planning_service, '_get_video_planner', return_value=mock_planner):
            result = await planning_service.generate_scene_outline(
                topic="Python basics",
                description="Introduction to Python programming",
                session_id="test-session",
                model_wrappers=mock_model_wrappers
            )
        
        assert result == "Test scene outline with multiple scenes"
        mock_planner.generate_scene_outline.assert_called_once_with(
            topic="Python basics",
            description="Introduction to Python programming",
            session_id="test-session"
        )
    
    @pytest.mark.asyncio
    async def test_generate_scene_outline_empty_topic(self, planning_service, mock_model_wrappers):
        """Test scene outline generation with empty topic."""
        with pytest.raises(ValueError, match="Topic cannot be empty"):
            await planning_service.generate_scene_outline(
                topic="",
                description="Test description",
                session_id="test-session",
                model_wrappers=mock_model_wrappers
            )
    
    @pytest.mark.asyncio
    async def test_generate_scene_outline_empty_description(self, planning_service, mock_model_wrappers):
        """Test scene outline generation with empty description."""
        with pytest.raises(ValueError, match="Description cannot be empty"):
            await planning_service.generate_scene_outline(
                topic="Test topic",
                description="",
                session_id="test-session",
                model_wrappers=mock_model_wrappers
            )
    
    @pytest.mark.asyncio
    async def test_generate_scene_outline_empty_result(self, planning_service, mock_model_wrappers):
        """Test scene outline generation with empty result."""
        mock_planner = AsyncMock()
        mock_planner.generate_scene_outline.return_value = ""
        
        with patch.object(planning_service, '_get_video_planner', return_value=mock_planner):
            with pytest.raises(ValueError, match="Failed to generate scene outline - empty result"):
                await planning_service.generate_scene_outline(
                    topic="Test topic",
                    description="Test description",
                    session_id="test-session",
                    model_wrappers=mock_model_wrappers
                )
    
    @pytest.mark.asyncio
    async def test_generate_scene_implementations_success(self, planning_service, mock_model_wrappers):
        """Test successful scene implementations generation."""
        mock_implementations = [
            "Scene 1: Introduction animation",
            "Scene 2: Main content display",
            "Scene 3: Conclusion summary"
        ]
        
        mock_planner = AsyncMock()
        mock_planner.generate_scene_implementation_concurrently_enhanced.return_value = mock_implementations
        
        with patch.object(planning_service, '_get_video_planner', return_value=mock_planner):
            result = await planning_service.generate_scene_implementations(
                topic="Python basics",
                description="Introduction to Python programming",
                plan="Test scene plan",
                session_id="test-session",
                model_wrappers=mock_model_wrappers
            )
        
        assert result == mock_implementations
        assert len(result) == 3
        mock_planner.generate_scene_implementation_concurrently_enhanced.assert_called_once_with(
            topic="Python basics",
            description="Introduction to Python programming",
            plan="Test scene plan",
            session_id="test-session"
        )
    
    @pytest.mark.asyncio
    async def test_generate_scene_implementations_empty_plan(self, planning_service, mock_model_wrappers):
        """Test scene implementations generation with empty plan."""
        with pytest.raises(ValueError, match="Plan cannot be empty"):
            await planning_service.generate_scene_implementations(
                topic="Test topic",
                description="Test description",
                plan="",
                session_id="test-session",
                model_wrappers=mock_model_wrappers
            )
    
    @pytest.mark.asyncio
    async def test_detect_plugins_success(self, planning_service, mock_model_wrappers):
        """Test successful plugin detection."""
        mock_plugins = ["numpy", "matplotlib", "scipy"]
        
        mock_planner = AsyncMock()
        mock_planner.rag_integration = Mock()  # RAG integration available
        mock_planner._detect_plugins_async.return_value = mock_plugins
        
        with patch.object(planning_service, '_get_video_planner', return_value=mock_planner):
            result = await planning_service.detect_plugins(
                topic="Data visualization",
                description="Creating charts and graphs",
                model_wrappers=mock_model_wrappers
            )
        
        assert result == mock_plugins
        mock_planner._detect_plugins_async.assert_called_once_with(
            "Data visualization", "Creating charts and graphs"
        )
    
    @pytest.mark.asyncio
    async def test_detect_plugins_no_rag(self, planning_service, mock_model_wrappers):
        """Test plugin detection without RAG integration."""
        mock_planner = AsyncMock()
        mock_planner.rag_integration = None  # No RAG integration
        
        with patch.object(planning_service, '_get_video_planner', return_value=mock_planner):
            result = await planning_service.detect_plugins(
                topic="Test topic",
                description="Test description",
                model_wrappers=mock_model_wrappers
            )
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_detect_plugins_error_handling(self, planning_service, mock_model_wrappers):
        """Test plugin detection error handling."""
        mock_planner = AsyncMock()
        mock_planner.rag_integration = Mock()
        mock_planner._detect_plugins_async.side_effect = Exception("Plugin detection failed")
        
        with patch.object(planning_service, '_get_video_planner', return_value=mock_planner):
            # Should not raise exception, should return empty list
            result = await planning_service.detect_plugins(
                topic="Test topic",
                description="Test description",
                model_wrappers=mock_model_wrappers
            )
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_validate_scene_outline_valid(self, planning_service):
        """Test scene outline validation with valid outline."""
        valid_outline = """
        Scene 1: Introduction
        - Show title animation
        - Display main concepts
        
        Scene 2: Main Content
        - Create mathematical equations
        - Animate transformations
        
        Scene 3: Conclusion
        - Summarize key points
        - Show final animation
        """
        
        is_valid, issues = await planning_service.validate_scene_outline(valid_outline)
        
        assert is_valid is True
        assert len(issues) == 0
    
    @pytest.mark.asyncio
    async def test_validate_scene_outline_empty(self, planning_service):
        """Test scene outline validation with empty outline."""
        is_valid, issues = await planning_service.validate_scene_outline("")
        
        assert is_valid is False
        assert "Scene outline is empty" in issues
    
    @pytest.mark.asyncio
    async def test_validate_scene_outline_too_short(self, planning_service):
        """Test scene outline validation with too short outline."""
        short_outline = "Short outline"
        
        is_valid, issues = await planning_service.validate_scene_outline(short_outline)
        
        assert is_valid is False
        assert any("too short" in issue for issue in issues)
    
    @pytest.mark.asyncio
    async def test_validate_scene_implementations_valid(self, planning_service):
        """Test scene implementations validation with valid implementations."""
        valid_implementations = [
            "Scene 1: Create Manim animation with text objects and transformations",
            "Scene 2: Display mathematical equations using Manim's MathTex class",
            "Scene 3: Animate geometric objects with smooth transitions"
        ]
        
        all_valid, issues_by_scene = await planning_service.validate_scene_implementations(valid_implementations)
        
        assert all_valid is True
        assert len(issues_by_scene) == 0
    
    @pytest.mark.asyncio
    async def test_validate_scene_implementations_empty_list(self, planning_service):
        """Test scene implementations validation with empty list."""
        all_valid, issues_by_scene = await planning_service.validate_scene_implementations([])
        
        assert all_valid is False
        assert 0 in issues_by_scene
        assert "No scene implementations provided" in issues_by_scene[0]
    
    @pytest.mark.asyncio
    async def test_validate_scene_implementations_with_errors(self, planning_service):
        """Test scene implementations validation with error indicators."""
        implementations_with_errors = [
            "Scene 1: Valid implementation with Manim objects",
            "Error: Failed to generate scene 2 implementation",
            "Scene 3: Another valid Manim animation"
        ]
        
        all_valid, issues_by_scene = await planning_service.validate_scene_implementations(implementations_with_errors)
        
        assert all_valid is False
        assert 2 in issues_by_scene
        assert any("error indicators" in issue for issue in issues_by_scene[2])
    
    def test_get_planning_metrics(self, planning_service):
        """Test planning metrics retrieval."""
        metrics = planning_service.get_planning_metrics()
        
        assert metrics['service_name'] == 'PlanningService'
        assert 'config' in metrics
        assert metrics['config']['use_rag'] is True
        assert metrics['config']['max_scene_concurrency'] == 3
        assert metrics['video_planner_initialized'] is False
    
    def test_get_planning_metrics_with_planner(self, planning_service, mock_model_wrappers):
        """Test planning metrics retrieval with initialized planner."""
        mock_planner = Mock()
        mock_planner.get_metrics.return_value = {'test_metric': 'test_value'}
        
        with patch.object(planning_service, '_get_video_planner', return_value=mock_planner):
            # Initialize the planner
            planning_service._get_video_planner(mock_model_wrappers)
            
            metrics = planning_service.get_planning_metrics()
            
            assert metrics['video_planner_initialized'] is True
            assert 'planner_metrics' in metrics
            assert metrics['planner_metrics']['test_metric'] == 'test_value'
    
    @pytest.mark.asyncio
    async def test_cleanup(self, planning_service):
        """Test service cleanup."""
        mock_planner = Mock()
        mock_thread_pool = Mock()
        mock_planner.thread_pool = mock_thread_pool
        planning_service._video_planner = mock_planner
        
        await planning_service.cleanup()
        
        mock_thread_pool.shutdown.assert_called_once_with(wait=False)
    
    @pytest.mark.asyncio
    async def test_cleanup_no_planner(self, planning_service):
        """Test service cleanup with no planner initialized."""
        # Should not raise exception
        await planning_service.cleanup()
    
    @pytest.mark.asyncio
    async def test_cleanup_error_handling(self, planning_service):
        """Test service cleanup error handling."""
        mock_planner = Mock()
        mock_thread_pool = Mock()
        mock_thread_pool.shutdown.side_effect = Exception("Cleanup error")
        mock_planner.thread_pool = mock_thread_pool
        planning_service._video_planner = mock_planner
        
        # Should not raise exception
        await planning_service.cleanup()
    
    def test_destructor(self, planning_service):
        """Test service destructor."""
        mock_planner = Mock()
        mock_thread_pool = Mock()
        mock_planner.thread_pool = mock_thread_pool
        planning_service._video_planner = mock_planner
        
        # Call destructor
        planning_service.__del__()
        
        mock_thread_pool.shutdown.assert_called_once_with(wait=False)
    
    def test_destructor_error_handling(self, planning_service):
        """Test service destructor error handling."""
        mock_planner = Mock()
        mock_thread_pool = Mock()
        mock_thread_pool.shutdown.side_effect = Exception("Destructor error")
        mock_planner.thread_pool = mock_thread_pool
        planning_service._video_planner = mock_planner
        
        # Should not raise exception
        planning_service.__del__()
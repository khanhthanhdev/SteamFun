"""
Unit tests for RenderingService.
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

from src.langgraph_agents.services.rendering_service import RenderingService


class TestRenderingService:
    """Test suite for RenderingService functionality."""
    
    @pytest.fixture
    def service_config(self):
        """Create test configuration for RenderingService."""
        return {
            'output_dir': 'test_output',
            'max_concurrent_renders': 4,
            'enable_caching': True,
            'default_quality': 'medium',
            'use_gpu_acceleration': False,
            'preview_mode': False,
            'use_visual_fix_code': True,
            'max_retries': 3,
            'print_response': False,
            'banned_reasonings': ['test_banned']
        }
    
    @pytest.fixture
    def rendering_service(self, service_config):
        """Create RenderingService instance for testing."""
        return RenderingService(service_config)
    
    def test_init(self, service_config):
        """Test RenderingService initialization."""
        service = RenderingService(service_config)
        
        assert service.config == service_config
        assert service.output_dir == 'test_output'
        assert service.max_concurrent_renders == 4
        assert service.enable_caching is True
        assert service.default_quality == 'medium'
        assert service.use_gpu_acceleration is False
        assert service.max_retries == 3
        assert service._video_renderer is None
        assert service._executor is None
    
    @patch('src.langgraph_agents.services.rendering_service.OptimizedVideoRenderer')
    def test_get_video_renderer_creation(self, mock_renderer_class, rendering_service):
        """Test video renderer creation."""
        mock_renderer_instance = Mock()
        mock_renderer_class.return_value = mock_renderer_instance
        
        result = rendering_service._get_video_renderer()
        
        assert result == mock_renderer_instance
        assert rendering_service._video_renderer == mock_renderer_instance
        
        # Verify renderer was created with correct arguments
        mock_renderer_class.assert_called_once()
        call_args = mock_renderer_class.call_args
        assert call_args[1]['output_dir'] == 'test_output'
        assert call_args[1]['max_concurrent_renders'] == 4
        assert call_args[1]['enable_caching'] is True
        assert call_args[1]['default_quality'] == 'medium'
    
    @patch('src.langgraph_agents.services.rendering_service.OptimizedVideoRenderer')
    def test_get_video_renderer_reuse(self, mock_renderer_class, rendering_service):
        """Test video renderer instance reuse."""
        mock_renderer_instance = Mock()
        mock_renderer_class.return_value = mock_renderer_instance
        
        # First call creates the renderer
        result1 = rendering_service._get_video_renderer()
        # Second call should reuse the same instance
        result2 = rendering_service._get_video_renderer()
        
        assert result1 == result2 == mock_renderer_instance
        # Should only be called once
        mock_renderer_class.assert_called_once()
    
    @patch('src.langgraph_agents.services.rendering_service.ThreadPoolExecutor')
    def test_get_executor_creation(self, mock_executor_class, rendering_service):
        """Test thread pool executor creation."""
        mock_executor_instance = Mock()
        mock_executor_class.return_value = mock_executor_instance
        
        result = rendering_service._get_executor()
        
        assert result == mock_executor_instance
        assert rendering_service._executor == mock_executor_instance
        
        # Verify executor was created with correct arguments
        mock_executor_class.assert_called_once_with(
            max_workers=4,
            thread_name_prefix="RenderingService"
        )
    
    @pytest.mark.asyncio
    async def test_render_scene_success(self, rendering_service):
        """Test successful scene rendering."""
        mock_renderer = AsyncMock()
        mock_renderer.render_scene_optimized.return_value = ("final_code", None)
        
        with patch.object(rendering_service, '_get_video_renderer', return_value=mock_renderer):
            with patch('os.makedirs'):
                final_code, error = await rendering_service.render_scene(
                    code="from manim import *\nclass TestScene(Scene):\n    def construct(self):\n        pass",
                    file_prefix="test_video",
                    scene_number=1,
                    version=1,
                    quality="medium",
                    topic="Test topic",
                    session_id="test-session"
                )
        
        assert final_code == "final_code"
        assert error is None
        mock_renderer.render_scene_optimized.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_render_scene_empty_code(self, rendering_service):
        """Test scene rendering with empty code."""
        with pytest.raises(ValueError, match="Code cannot be empty"):
            await rendering_service.render_scene(
                code="",
                file_prefix="test_video",
                scene_number=1
            )
    
    @pytest.mark.asyncio
    async def test_render_scene_invalid_scene_number(self, rendering_service):
        """Test scene rendering with invalid scene number."""
        with pytest.raises(ValueError, match="Scene number must be positive"):
            await rendering_service.render_scene(
                code="test code",
                file_prefix="test_video",
                scene_number=0
            )
    
    @pytest.mark.asyncio
    async def test_render_scene_with_error(self, rendering_service):
        """Test scene rendering with error."""
        mock_renderer = AsyncMock()
        mock_renderer.render_scene_optimized.return_value = ("original_code", "Rendering failed")
        
        with patch.object(rendering_service, '_get_video_renderer', return_value=mock_renderer):
            with patch('os.makedirs'):
                final_code, error = await rendering_service.render_scene(
                    code="test code",
                    file_prefix="test_video",
                    scene_number=1
                )
        
        assert final_code == "original_code"
        assert error == "Rendering failed"
    
    @pytest.mark.asyncio
    async def test_render_scene_exception(self, rendering_service):
        """Test scene rendering with exception."""
        mock_renderer = AsyncMock()
        mock_renderer.render_scene_optimized.side_effect = Exception("Renderer exception")
        
        with patch.object(rendering_service, '_get_video_renderer', return_value=mock_renderer):
            with patch('os.makedirs'):
                final_code, error = await rendering_service.render_scene(
                    code="test code",
                    file_prefix="test_video",
                    scene_number=1
                )
        
        assert final_code == "test code"
        assert "Renderer exception" in error
    
    @pytest.mark.asyncio
    async def test_render_multiple_scenes_parallel_success(self, rendering_service):
        """Test successful parallel scene rendering."""
        scene_configs = [
            {'code': 'code1', 'curr_scene': 1, 'file_prefix': 'test'},
            {'code': 'code2', 'curr_scene': 2, 'file_prefix': 'test'},
            {'code': 'code3', 'curr_scene': 3, 'file_prefix': 'test'}
        ]
        
        mock_renderer = AsyncMock()
        mock_renderer.render_multiple_scenes_parallel.return_value = [
            ("final_code1", None),
            ("final_code2", None),
            ("final_code3", None)
        ]
        
        with patch.object(rendering_service, '_get_video_renderer', return_value=mock_renderer):
            results = await rendering_service.render_multiple_scenes_parallel(scene_configs)
        
        assert len(results) == 3
        assert results[0] == ("final_code1", None)
        assert results[1] == ("final_code2", None)
        assert results[2] == ("final_code3", None)
        mock_renderer.render_multiple_scenes_parallel.assert_called_once_with(
            scene_configs=scene_configs,
            max_concurrent=4
        )
    
    @pytest.mark.asyncio
    async def test_render_multiple_scenes_parallel_with_errors(self, rendering_service):
        """Test parallel scene rendering with some errors."""
        scene_configs = [
            {'code': 'code1', 'curr_scene': 1, 'file_prefix': 'test'},
            {'code': 'code2', 'curr_scene': 2, 'file_prefix': 'test'}
        ]
        
        mock_renderer = AsyncMock()
        mock_renderer.render_multiple_scenes_parallel.return_value = [
            ("final_code1", None),
            Exception("Rendering failed for scene 2")
        ]
        
        with patch.object(rendering_service, '_get_video_renderer', return_value=mock_renderer):
            results = await rendering_service.render_multiple_scenes_parallel(scene_configs)
        
        assert len(results) == 2
        assert results[0] == ("final_code1", None)
        assert results[1][0] == 'code2'  # Original code
        assert "Rendering failed for scene 2" in results[1][1]  # Error message
    
    @pytest.mark.asyncio
    async def test_render_multiple_scenes_parallel_empty_configs(self, rendering_service):
        """Test parallel scene rendering with empty configs."""
        with pytest.raises(ValueError, match="Scene configs cannot be empty"):
            await rendering_service.render_multiple_scenes_parallel([])
    
    @pytest.mark.asyncio
    async def test_render_multiple_scenes_parallel_exception(self, rendering_service):
        """Test parallel scene rendering with exception."""
        scene_configs = [{'code': 'code1', 'curr_scene': 1, 'file_prefix': 'test'}]
        
        mock_renderer = AsyncMock()
        mock_renderer.render_multiple_scenes_parallel.side_effect = Exception("Parallel rendering failed")
        
        with patch.object(rendering_service, '_get_video_renderer', return_value=mock_renderer):
            results = await rendering_service.render_multiple_scenes_parallel(scene_configs)
        
        assert len(results) == 1
        assert results[0][0] == 'code1'
        assert "Parallel rendering failed" in results[0][1]
    
    @pytest.mark.asyncio
    async def test_combine_videos_success(self, rendering_service):
        """Test successful video combination."""
        rendered_videos = {1: "/path/to/video1.mp4", 2: "/path/to/video2.mp4"}
        
        mock_renderer = AsyncMock()
        mock_renderer.combine_videos_optimized.return_value = "/path/to/combined.mp4"
        
        with patch.object(rendering_service, '_get_video_renderer', return_value=mock_renderer):
            with patch('os.path.exists', return_value=True):
                combined_path = await rendering_service.combine_videos(
                    topic="Test topic",
                    rendered_videos=rendered_videos,
                    use_hardware_acceleration=True
                )
        
        assert combined_path == "/path/to/combined.mp4"
        mock_renderer.combine_videos_optimized.assert_called_once_with(
            topic="Test topic",
            use_hardware_acceleration=True
        )
    
    @pytest.mark.asyncio
    async def test_combine_videos_empty_topic(self, rendering_service):
        """Test video combination with empty topic."""
        with pytest.raises(ValueError, match="Topic cannot be empty"):
            await rendering_service.combine_videos(
                topic="",
                rendered_videos={1: "/path/to/video1.mp4"}
            )
    
    @pytest.mark.asyncio
    async def test_combine_videos_empty_videos(self, rendering_service):
        """Test video combination with empty videos."""
        with pytest.raises(ValueError, match="Rendered videos cannot be empty"):
            await rendering_service.combine_videos(
                topic="Test topic",
                rendered_videos={}
            )
    
    @pytest.mark.asyncio
    async def test_combine_videos_file_not_created(self, rendering_service):
        """Test video combination when file is not created."""
        rendered_videos = {1: "/path/to/video1.mp4"}
        
        mock_renderer = AsyncMock()
        mock_renderer.combine_videos_optimized.return_value = "/path/to/combined.mp4"
        
        with patch.object(rendering_service, '_get_video_renderer', return_value=mock_renderer):
            with patch('os.path.exists', return_value=False):
                with pytest.raises(ValueError, match="Combined video was not created successfully"):
                    await rendering_service.combine_videos(
                        topic="Test topic",
                        rendered_videos=rendered_videos
                    )
    
    def test_find_rendered_video_success(self, rendering_service):
        """Test successful video file finding."""
        with patch('os.path.exists', return_value=True):
            with patch('os.listdir', return_value=['test_video_scene1_v1.mp4']):
                video_path = rendering_service.find_rendered_video(
                    file_prefix="test_video",
                    scene_number=1,
                    version=1
                )
        
        assert video_path.endswith('test_video_scene1_v1.mp4')
    
    def test_find_rendered_video_not_found(self, rendering_service):
        """Test video file finding when file doesn't exist."""
        with patch('os.path.exists', return_value=False):
            with pytest.raises(FileNotFoundError, match="No rendered video found"):
                rendering_service.find_rendered_video(
                    file_prefix="test_video",
                    scene_number=1,
                    version=1
                )
    
    def test_get_performance_stats(self, rendering_service):
        """Test performance statistics retrieval."""
        stats = rendering_service.get_performance_stats()
        
        assert stats['service_name'] == 'RenderingService'
        assert 'config' in stats
        assert stats['config']['max_concurrent_renders'] == 4
        assert stats['config']['enable_caching'] is True
        assert stats['video_renderer_initialized'] is False
        assert stats['executor_initialized'] is False
    
    def test_get_performance_stats_with_renderer(self, rendering_service):
        """Test performance statistics with initialized renderer."""
        mock_renderer = Mock()
        mock_renderer.get_performance_stats.return_value = {
            'cache_hit_rate': 0.75,
            'average_time': 45.2,
            'total_renders': 10
        }
        
        with patch.object(rendering_service, '_get_video_renderer', return_value=mock_renderer):
            # Initialize the renderer
            rendering_service._get_video_renderer()
            
            stats = rendering_service.get_performance_stats()
            
            assert stats['video_renderer_initialized'] is True
            assert 'renderer_stats' in stats
            assert stats['renderer_stats']['cache_hit_rate'] == 0.75
    
    def test_cleanup_cache(self, rendering_service):
        """Test cache cleanup."""
        mock_renderer = Mock()
        
        with patch.object(rendering_service, '_get_video_renderer', return_value=mock_renderer):
            # Initialize the renderer
            rendering_service._get_video_renderer()
            
            rendering_service.cleanup_cache(max_age_days=5)
            
            mock_renderer.cleanup_cache.assert_called_once_with(5)
    
    def test_cleanup_cache_no_renderer(self, rendering_service):
        """Test cache cleanup with no renderer initialized."""
        # Should not raise exception
        rendering_service.cleanup_cache()
    
    def test_get_cache_stats(self, rendering_service):
        """Test cache statistics retrieval."""
        mock_renderer = Mock()
        mock_renderer.get_performance_stats.return_value = {
            'cache_enabled': True,
            'cache_hits': 15,
            'total_renders': 20,
            'cache_hit_rate': 0.75,
            'cache_size': 1024
        }
        
        with patch.object(rendering_service, '_get_video_renderer', return_value=mock_renderer):
            # Initialize the renderer
            rendering_service._get_video_renderer()
            
            stats = rendering_service.get_cache_stats()
            
            assert stats['cache_enabled'] is True
            assert stats['cache_hits'] == 15
            assert stats['total_renders'] == 20
            assert stats['cache_hit_rate'] == 0.75
            assert stats['cache_size'] == 1024
    
    def test_get_cache_stats_no_renderer(self, rendering_service):
        """Test cache statistics with no renderer initialized."""
        stats = rendering_service.get_cache_stats()
        
        assert stats['cache_enabled'] is True  # From config
        assert stats['cache_hits'] == 0
        assert stats['total_renders'] == 0
        assert stats['cache_hit_rate'] == 0
        assert stats['cache_size'] == 0
    
    @pytest.mark.asyncio
    async def test_optimize_rendering_settings_low_success_rate(self, rendering_service):
        """Test rendering settings optimization for low success rate."""
        performance_metrics = {
            'renderer_agent': {
                'success_rate': 0.6,  # Low success rate
                'render_stats': {
                    'average_time': 30,
                    'cache_hit_rate': 0.5
                }
            }
        }
        
        optimized_settings = await rendering_service.optimize_rendering_settings(performance_metrics)
        
        assert optimized_settings['default_quality'] == 'low'
        assert optimized_settings['max_concurrent_renders'] == 2  # Half of original
        assert optimized_settings['use_gpu_acceleration'] is False
        assert optimized_settings['preview_mode'] is True
    
    @pytest.mark.asyncio
    async def test_optimize_rendering_settings_slow_rendering(self, rendering_service):
        """Test rendering settings optimization for slow rendering."""
        performance_metrics = {
            'renderer_agent': {
                'success_rate': 0.9,  # Good success rate
                'render_stats': {
                    'average_time': 90,  # Slow rendering
                    'cache_hit_rate': 0.5
                }
            }
        }
        
        optimized_settings = await rendering_service.optimize_rendering_settings(performance_metrics)
        
        assert optimized_settings['default_quality'] == 'low'
        assert optimized_settings['preview_mode'] is True
        assert optimized_settings['max_concurrent_renders'] <= 2
    
    @pytest.mark.asyncio
    async def test_optimize_rendering_settings_low_cache_hit_rate(self, rendering_service):
        """Test rendering settings optimization for low cache hit rate."""
        performance_metrics = {
            'renderer_agent': {
                'success_rate': 0.9,
                'render_stats': {
                    'average_time': 30,
                    'cache_hit_rate': 0.2  # Low cache hit rate
                }
            }
        }
        
        optimized_settings = await rendering_service.optimize_rendering_settings(performance_metrics)
        
        assert optimized_settings['enable_caching'] is True
    
    @pytest.mark.asyncio
    async def test_validate_scene_config_valid(self, rendering_service):
        """Test scene config validation with valid config."""
        valid_config = {
            'code': 'from manim import *\nclass TestScene(Scene):\n    def construct(self):\n        pass',
            'file_prefix': 'test_video',
            'curr_scene': 1,
            'curr_version': 1,
            'quality': 'medium'
        }
        
        is_valid, issues = await rendering_service.validate_scene_config(valid_config)
        
        assert is_valid is True
        assert len(issues) == 0
    
    @pytest.mark.asyncio
    async def test_validate_scene_config_missing_fields(self, rendering_service):
        """Test scene config validation with missing required fields."""
        invalid_config = {
            'code': 'test code'
            # Missing file_prefix, curr_scene, curr_version
        }
        
        is_valid, issues = await rendering_service.validate_scene_config(invalid_config)
        
        assert is_valid is False
        assert any("Missing required field: file_prefix" in issue for issue in issues)
        assert any("Missing required field: curr_scene" in issue for issue in issues)
        assert any("Missing required field: curr_version" in issue for issue in issues)
    
    @pytest.mark.asyncio
    async def test_validate_scene_config_invalid_scene_number(self, rendering_service):
        """Test scene config validation with invalid scene number."""
        invalid_config = {
            'code': 'test code',
            'file_prefix': 'test',
            'curr_scene': 0,  # Invalid scene number
            'curr_version': 1
        }
        
        is_valid, issues = await rendering_service.validate_scene_config(invalid_config)
        
        assert is_valid is False
        assert any("Scene number must be positive" in issue for issue in issues)
    
    @pytest.mark.asyncio
    async def test_validate_scene_config_invalid_quality(self, rendering_service):
        """Test scene config validation with invalid quality."""
        invalid_config = {
            'code': 'test code',
            'file_prefix': 'test',
            'curr_scene': 1,
            'curr_version': 1,
            'quality': 'invalid_quality'
        }
        
        is_valid, issues = await rendering_service.validate_scene_config(invalid_config)
        
        assert is_valid is False
        assert any("Invalid quality" in issue for issue in issues)
    
    @pytest.mark.asyncio
    async def test_validate_scene_config_short_code(self, rendering_service):
        """Test scene config validation with too short code."""
        invalid_config = {
            'code': 'short',  # Too short
            'file_prefix': 'test',
            'curr_scene': 1,
            'curr_version': 1
        }
        
        is_valid, issues = await rendering_service.validate_scene_config(invalid_config)
        
        assert is_valid is False
        assert any("Code is too short" in issue for issue in issues)
    
    @pytest.mark.asyncio
    async def test_validate_scene_config_no_manim(self, rendering_service):
        """Test scene config validation with code lacking Manim."""
        invalid_config = {
            'code': 'print("Hello, World!")  # This is a long enough code but no Manim',
            'file_prefix': 'test',
            'curr_scene': 1,
            'curr_version': 1
        }
        
        is_valid, issues = await rendering_service.validate_scene_config(invalid_config)
        
        assert is_valid is False
        assert any("does not appear to contain Manim" in issue for issue in issues)
    
    @pytest.mark.asyncio
    async def test_cleanup(self, rendering_service):
        """Test service cleanup."""
        mock_executor = Mock()
        mock_renderer = Mock()
        mock_renderer.cleanup = Mock()
        
        rendering_service._executor = mock_executor
        rendering_service._video_renderer = mock_renderer
        
        await rendering_service.cleanup()
        
        mock_executor.shutdown.assert_called_once_with(wait=True)
        mock_renderer.cleanup.assert_called_once()
        assert rendering_service._executor is None
    
    @pytest.mark.asyncio
    async def test_cleanup_no_resources(self, rendering_service):
        """Test service cleanup with no resources initialized."""
        # Should not raise exception
        await rendering_service.cleanup()
    
    @pytest.mark.asyncio
    async def test_cleanup_error_handling(self, rendering_service):
        """Test service cleanup error handling."""
        mock_executor = Mock()
        mock_executor.shutdown.side_effect = Exception("Cleanup error")
        rendering_service._executor = mock_executor
        
        # Should not raise exception
        await rendering_service.cleanup()
    
    def test_destructor(self, rendering_service):
        """Test service destructor."""
        mock_executor = Mock()
        rendering_service._executor = mock_executor
        
        # Call destructor
        rendering_service.__del__()
        
        mock_executor.shutdown.assert_called_once_with(wait=False)
    
    def test_destructor_error_handling(self, rendering_service):
        """Test service destructor error handling."""
        mock_executor = Mock()
        mock_executor.shutdown.side_effect = Exception("Destructor error")
        rendering_service._executor = mock_executor
        
        # Should not raise exception
        rendering_service.__del__()
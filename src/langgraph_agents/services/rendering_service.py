"""
RenderingService - Business logic for video rendering and optimization.

Extracted from RendererAgent to follow separation of concerns and single responsibility principles.
"""

import os
import re
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.core.video_renderer import OptimizedVideoRenderer


logger = logging.getLogger(__name__)


class RenderingService:
    """Service class for video rendering operations.
    
    Handles video rendering, parallel processing, video combination,
    and optimization with comprehensive error handling and logging.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize RenderingService with configuration.
        
        Args:
            config: Service configuration containing rendering settings, paths, etc.
        """
        self.config = config
        self._video_renderer = None
        self._executor = None
        
        # Extract configuration values
        self.output_dir = config.get('output_dir', 'output')
        self.max_concurrent_renders = config.get('max_concurrent_renders', 4)
        self.enable_caching = config.get('enable_caching', True)
        self.default_quality = config.get('default_quality', 'medium')
        self.use_gpu_acceleration = config.get('use_gpu_acceleration', False)
        self.preview_mode = config.get('preview_mode', False)
        self.use_visual_fix_code = config.get('use_visual_fix_code', False)
        self.max_retries = config.get('max_retries', 5)
        
        logger.info(f"RenderingService initialized with config: {self.config}")
    
    def _get_video_renderer(self) -> OptimizedVideoRenderer:
        """Get or create OptimizedVideoRenderer instance.
        
        Returns:
            OptimizedVideoRenderer: Configured video renderer instance
        """
        if self._video_renderer is None:
            # Initialize OptimizedVideoRenderer
            self._video_renderer = OptimizedVideoRenderer(
                output_dir=self.output_dir,
                print_response=self.config.get('print_response', False),
                use_visual_fix_code=self.use_visual_fix_code,
                max_concurrent_renders=self.max_concurrent_renders,
                enable_caching=self.enable_caching,
                default_quality=self.default_quality,
                use_gpu_acceleration=self.use_gpu_acceleration,
                preview_mode=self.preview_mode
            )
            
            logger.info("Created OptimizedVideoRenderer instance")
        
        return self._video_renderer
    
    def _get_executor(self) -> ThreadPoolExecutor:
        """Get or create ThreadPoolExecutor for parallel rendering.
        
        Returns:
            ThreadPoolExecutor: Thread pool executor
        """
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.max_concurrent_renders,
                thread_name_prefix="RenderingService"
            )
            logger.info(f"Created ThreadPoolExecutor with {self.max_concurrent_renders} workers")
        
        return self._executor
    
    async def render_scene(self,
                         code: str,
                         file_prefix: str,
                         scene_number: int,
                         version: int = 1,
                         quality: Optional[str] = None,
                         scene_implementation: Optional[str] = None,
                         topic: Optional[str] = None,
                         description: Optional[str] = None,
                         scene_outline: Optional[str] = None,
                         session_id: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """Render a single scene with optimization.
        
        Args:
            code: Manim code to render
            file_prefix: File prefix for output files
            scene_number: Scene number
            version: Version number
            quality: Rendering quality (overrides default)
            scene_implementation: Scene implementation for error fixing
            topic: Video topic for error fixing
            description: Video description for error fixing
            scene_outline: Scene outline for error fixing
            session_id: Session identifier
            
        Returns:
            Tuple[str, Optional[str]]: (final_code, error_message)
            
        Raises:
            ValueError: If required parameters are empty or invalid
        """
        if not code or not code.strip():
            raise ValueError("Code cannot be empty")
        
        if not file_prefix or not file_prefix.strip():
            raise ValueError("File prefix cannot be empty")
        
        if scene_number <= 0:
            raise ValueError("Scene number must be positive")
        
        if version <= 0:
            raise ValueError("Version must be positive")
        
        logger.info(f"Rendering scene {scene_number} version {version}")
        
        try:
            # Create directories
            scene_dir = os.path.join(self.output_dir, file_prefix, f"scene{scene_number}")
            code_dir = os.path.join(scene_dir, "code")
            media_dir = os.path.join(scene_dir, "media")
            
            os.makedirs(code_dir, exist_ok=True)
            os.makedirs(media_dir, exist_ok=True)
            
            # Get video renderer
            video_renderer = self._get_video_renderer()
            
            # Create scene configuration
            scene_config = {
                'code': code,
                'file_prefix': file_prefix,
                'curr_scene': scene_number,
                'curr_version': version,
                'code_dir': code_dir,
                'media_dir': media_dir,
                'quality': quality or self.default_quality,
                'max_retries': self.max_retries,
                'use_visual_fix_code': self.use_visual_fix_code,
                'visual_self_reflection_func': None,  # Will be set if needed
                'banned_reasonings': self.config.get('banned_reasonings', []),
                'scene_trace_id': f"scene_{scene_number}_{session_id}" if session_id else f"scene_{scene_number}",
                'topic': topic,
                'session_id': session_id,
                'code_generator': None,  # Will be set if needed for error fixing
                'scene_implementation': scene_implementation,
                'description': description,
                'scene_outline': scene_outline
            }
            
            # Render the scene
            final_code, error = await video_renderer.render_scene_optimized(
                code=code,
                file_prefix=file_prefix,
                curr_scene=scene_number,
                curr_version=version,
                code_dir=code_dir,
                media_dir=media_dir,
                quality=quality or self.default_quality,
                max_retries=self.max_retries,
                **{k: v for k, v in scene_config.items() if k not in ['code', 'file_prefix', 'curr_scene', 'curr_version', 'code_dir', 'media_dir', 'quality', 'max_retries']}
            )
            
            if error:
                logger.error(f"Rendering failed for scene {scene_number}: {error}")
                return final_code, error
            
            logger.info(f"Successfully rendered scene {scene_number}")
            return final_code, None
            
        except Exception as e:
            logger.error(f"Error rendering scene {scene_number}: {e}")
            return code, str(e)
    
    async def render_multiple_scenes_parallel(self,
                                            scene_configs: List[Dict[str, Any]],
                                            max_concurrent: Optional[int] = None) -> List[Tuple[str, Optional[str]]]:
        """Render multiple scenes in parallel.
        
        Args:
            scene_configs: List of scene configurations
            max_concurrent: Maximum concurrent renders (overrides default)
            
        Returns:
            List[Tuple[str, Optional[str]]]: List of (final_code, error_message) tuples
            
        Raises:
            ValueError: If scene_configs is empty
        """
        if not scene_configs:
            raise ValueError("Scene configs cannot be empty")
        
        concurrent_limit = max_concurrent or self.max_concurrent_renders
        logger.info(f"Rendering {len(scene_configs)} scenes with max concurrency {concurrent_limit}")
        
        try:
            video_renderer = self._get_video_renderer()
            
            # Use the video renderer's parallel rendering capability
            results = await video_renderer.render_multiple_scenes_parallel(
                scene_configs=scene_configs,
                max_concurrent=concurrent_limit
            )
            
            # Process results to match our return format
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Error case
                    original_code = scene_configs[i].get('code', '')
                    processed_results.append((original_code, str(result)))
                else:
                    # Success case - result is (final_code, error)
                    processed_results.append(result)
            
            successful_renders = sum(1 for _, error in processed_results if error is None)
            logger.info(f"Parallel rendering completed: {successful_renders}/{len(scene_configs)} successful")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in parallel rendering: {e}")
            # Return error for all scenes
            return [(config.get('code', ''), str(e)) for config in scene_configs]
    
    async def combine_videos(self,
                           topic: str,
                           rendered_videos: Dict[int, str],
                           use_hardware_acceleration: Optional[bool] = None) -> str:
        """Combine multiple rendered videos into a single video.
        
        Args:
            topic: Video topic for naming
            rendered_videos: Dictionary mapping scene numbers to video paths
            use_hardware_acceleration: Whether to use hardware acceleration
            
        Returns:
            str: Path to combined video
            
        Raises:
            ValueError: If required parameters are empty or invalid
            Exception: If video combination fails
        """
        if not topic or not topic.strip():
            raise ValueError("Topic cannot be empty")
        
        if not rendered_videos:
            raise ValueError("Rendered videos cannot be empty")
        
        logger.info(f"Combining {len(rendered_videos)} videos for topic: {topic}")
        
        try:
            video_renderer = self._get_video_renderer()
            
            # Use hardware acceleration setting
            use_hw_accel = use_hardware_acceleration if use_hardware_acceleration is not None else self.use_gpu_acceleration
            
            combined_video_path = await video_renderer.combine_videos_optimized(
                topic=topic,
                use_hardware_acceleration=use_hw_accel
            )
            
            if not combined_video_path or not os.path.exists(combined_video_path):
                raise ValueError("Combined video was not created successfully")
            
            logger.info(f"Successfully combined videos: {combined_video_path}")
            return combined_video_path
            
        except Exception as e:
            logger.error(f"Error combining videos: {e}")
            raise
    
    def find_rendered_video(self, 
                          file_prefix: str, 
                          scene_number: int, 
                          version: int = 1) -> str:
        """Find the rendered video file for a scene.
        
        Args:
            file_prefix: File prefix used during rendering
            scene_number: Scene number
            version: Version number
            
        Returns:
            str: Path to rendered video file
            
        Raises:
            FileNotFoundError: If video file is not found
        """
        scene_dir = os.path.join(self.output_dir, file_prefix, f"scene{scene_number}")
        media_dir = os.path.join(scene_dir, "media")
        video_dir = os.path.join(media_dir, "videos", f"{file_prefix}_scene{scene_number}_v{version}")
        
        # Look in quality-specific subdirectories
        quality_dirs = ["1080p60", "720p30", "480p15", "preview"]
        
        for quality_dir in quality_dirs:
            search_dir = os.path.join(video_dir, quality_dir)
            if os.path.exists(search_dir):
                for file in os.listdir(search_dir):
                    if file.endswith('.mp4'):
                        video_path = os.path.join(search_dir, file)
                        logger.info(f"Found rendered video: {video_path}")
                        return video_path
        
        raise FileNotFoundError(f"No rendered video found for scene {scene_number} version {version}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get rendering performance statistics.
        
        Returns:
            Dict[str, Any]: Performance statistics
        """
        stats = {
            'service_name': 'RenderingService',
            'config': {
                'max_concurrent_renders': self.max_concurrent_renders,
                'enable_caching': self.enable_caching,
                'default_quality': self.default_quality,
                'use_gpu_acceleration': self.use_gpu_acceleration,
                'preview_mode': self.preview_mode
            },
            'video_renderer_initialized': self._video_renderer is not None,
            'executor_initialized': self._executor is not None
        }
        
        # Add video renderer performance stats if available
        if self._video_renderer:
            try:
                renderer_stats = self._video_renderer.get_performance_stats()
                stats['renderer_stats'] = renderer_stats
            except Exception as e:
                logger.warning(f"Could not retrieve renderer stats: {e}")
        
        return stats
    
    def cleanup_cache(self, max_age_days: int = 7):
        """Clean up old cache files.
        
        Args:
            max_age_days: Maximum age of cache files in days
        """
        try:
            if self._video_renderer:
                self._video_renderer.cleanup_cache(max_age_days)
                logger.info(f"Cache cleanup completed for files older than {max_age_days} days")
            else:
                logger.warning("Video renderer not initialized, cannot clean cache")
                
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dict[str, Any]: Cache statistics
        """
        if self._video_renderer:
            try:
                stats = self._video_renderer.get_performance_stats()
                return {
                    'cache_enabled': stats.get('cache_enabled', False),
                    'cache_hits': stats.get('cache_hits', 0),
                    'total_renders': stats.get('total_renders', 0),
                    'cache_hit_rate': stats.get('cache_hit_rate', 0),
                    'cache_size': stats.get('cache_size', 0)
                }
            except Exception as e:
                logger.warning(f"Could not retrieve cache stats: {e}")
        
        return {
            'cache_enabled': self.enable_caching,
            'cache_hits': 0,
            'total_renders': 0,
            'cache_hit_rate': 0,
            'cache_size': 0
        }
    
    async def optimize_rendering_settings(self, 
                                        performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize rendering settings based on performance metrics.
        
        Args:
            performance_metrics: Current performance metrics
            
        Returns:
            Dict[str, Any]: Optimized settings
        """
        current_settings = {
            'default_quality': self.default_quality,
            'max_concurrent_renders': self.max_concurrent_renders,
            'enable_caching': self.enable_caching,
            'use_gpu_acceleration': self.use_gpu_acceleration,
            'preview_mode': self.preview_mode
        }
        
        optimized_settings = current_settings.copy()
        
        try:
            # Get renderer metrics
            renderer_metrics = performance_metrics.get('renderer_agent', {})
            success_rate = renderer_metrics.get('success_rate', 1.0)
            avg_render_time = renderer_metrics.get('render_stats', {}).get('average_time', 0)
            
            # Optimize based on success rate
            if success_rate < 0.8:
                # Reduce quality and concurrency for better stability
                optimized_settings.update({
                    'default_quality': 'low',
                    'max_concurrent_renders': max(1, self.max_concurrent_renders // 2),
                    'use_gpu_acceleration': False,
                    'preview_mode': True
                })
                logger.info("Optimized settings for stability due to low success rate")
            
            # Optimize based on render time
            if avg_render_time > 60:  # More than 1 minute per scene
                optimized_settings.update({
                    'default_quality': 'low',
                    'preview_mode': True,
                    'max_concurrent_renders': min(self.max_concurrent_renders, 2)
                })
                logger.info("Optimized settings for speed due to slow rendering")
            
            # Optimize based on cache hit rate
            cache_hit_rate = renderer_metrics.get('render_stats', {}).get('cache_hit_rate', 0)
            if cache_hit_rate < 0.3:  # Low cache efficiency
                optimized_settings['enable_caching'] = True
                logger.info("Enabled caching due to low cache hit rate")
            
            return optimized_settings
            
        except Exception as e:
            logger.error(f"Error optimizing rendering settings: {e}")
            return current_settings
    
    async def validate_scene_config(self, scene_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a scene configuration.
        
        Args:
            scene_config: Scene configuration to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required fields
        required_fields = ['code', 'file_prefix', 'curr_scene', 'curr_version']
        for field in required_fields:
            if field not in scene_config:
                issues.append(f"Missing required field: {field}")
            elif not scene_config[field]:
                issues.append(f"Empty required field: {field}")
        
        # Validate scene number
        if 'curr_scene' in scene_config:
            try:
                scene_num = int(scene_config['curr_scene'])
                if scene_num <= 0:
                    issues.append("Scene number must be positive")
            except (ValueError, TypeError):
                issues.append("Scene number must be a valid integer")
        
        # Validate version
        if 'curr_version' in scene_config:
            try:
                version = int(scene_config['curr_version'])
                if version <= 0:
                    issues.append("Version must be positive")
            except (ValueError, TypeError):
                issues.append("Version must be a valid integer")
        
        # Validate quality
        if 'quality' in scene_config:
            valid_qualities = ['high', 'medium', 'low', 'preview']
            if scene_config['quality'] not in valid_qualities:
                issues.append(f"Invalid quality: {scene_config['quality']}. Must be one of {valid_qualities}")
        
        # Validate code content
        if 'code' in scene_config and scene_config['code']:
            code = scene_config['code']
            if len(code.strip()) < 50:
                issues.append("Code is too short (minimum 50 characters)")
            
            if 'manim' not in code.lower():
                issues.append("Code does not appear to contain Manim imports")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info(f"Scene config validation passed for scene {scene_config.get('curr_scene', 'unknown')}")
        else:
            logger.warning(f"Scene config validation failed with issues: {issues}")
        
        return is_valid, issues
    
    async def cleanup(self):
        """Cleanup resources used by the rendering service."""
        try:
            # Shutdown thread pool executor
            if self._executor:
                self._executor.shutdown(wait=True)
                self._executor = None
                logger.info("Thread pool executor shutdown")
            
            # Cleanup video renderer if it has cleanup methods
            if self._video_renderer and hasattr(self._video_renderer, 'cleanup'):
                self._video_renderer.cleanup()
                
            logger.info("Rendering service cleanup completed")
            
        except Exception as e:
            logger.warning(f"Error during rendering service cleanup: {e}")
    
    def __del__(self):
        """Cleanup resources when service is destroyed."""
        try:
            if self._executor:
                self._executor.shutdown(wait=False)
        except Exception:
            pass  # Ignore cleanup errors in destructor
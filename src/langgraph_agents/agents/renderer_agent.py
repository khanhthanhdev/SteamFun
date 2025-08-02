"""
RendererAgent with optimization features and concurrent rendering.
Ports OptimizedVideoRenderer functionality to LangGraph agent pattern while maintaining compatibility.
"""

import os
import re
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from langgraph.types import Command

from ..base_agent import BaseAgent
from ..state import VideoGenerationState
from src.core.video_renderer import OptimizedVideoRenderer


logger = logging.getLogger(__name__)


class RendererAgent(BaseAgent):
    """RendererAgent for video rendering and optimization.
    
    Ports OptimizedVideoRenderer functionality to LangGraph agent pattern while
    preserving existing method signatures and optimization features.
    """
    
    def __init__(self, config, system_config):
        """Initialize RendererAgent with optimization features.
        
        Args:
            config: Agent configuration
            system_config: System configuration
        """
        super().__init__(config, system_config)
        
        # Initialize internal video renderer (will be created on first use)
        self._video_renderer = None
        
        logger.info(f"RendererAgent initialized with config: {config.name}")
    
    def _get_video_renderer(self, state: VideoGenerationState) -> OptimizedVideoRenderer:
        """Get or create OptimizedVideoRenderer instance with current state configuration.
        
        Args:
            state: Current workflow state
            
        Returns:
            OptimizedVideoRenderer: Configured video renderer instance
        """
        if self._video_renderer is None:
            # Initialize OptimizedVideoRenderer with state configuration
            self._video_renderer = OptimizedVideoRenderer(
                output_dir=state.get('output_dir', 'output'),
                print_response=state.get('print_response', False),
                use_visual_fix_code=state.get('use_visual_fix_code', False),
                max_concurrent_renders=state.get('max_concurrent_renders', 4),
                enable_caching=state.get('enable_caching', True),
                default_quality=state.get('default_quality', 'medium'),
                use_gpu_acceleration=state.get('use_gpu_acceleration', False),
                preview_mode=state.get('preview_mode', False)
            )
            
            logger.info("Created OptimizedVideoRenderer instance with current state configuration")
        
        return self._video_renderer
    
    async def execute(self, state: VideoGenerationState) -> Command:
        """Execute video rendering with optimization features.
        
        Args:
            state: Current workflow state
            
        Returns:
            Command: LangGraph command for next action
        """
        self.log_agent_action("starting_video_rendering", {
            'topic': state.get('topic', ''),
            'scene_count': len(state.get('generated_code', {})),
            'quality': state.get('default_quality', 'medium')
        })
        
        try:
            # Get required data from state
            generated_code = state.get('generated_code', {})
            scene_implementations = state.get('scene_implementations', {})
            
            if not generated_code:
                raise ValueError("No generated code available for rendering")
            
            # Get video renderer instance with current performance tracking
            video_renderer = self._get_video_renderer(state)
            
            # Prepare rendering configurations for each scene
            scene_configs = []
            file_prefix = re.sub(r'[^a-z0-9_]+', '_', state['topic'].lower())
            
            for scene_number, code in generated_code.items():
                # Create directories
                scene_dir = os.path.join(state.get('output_dir', 'output'), file_prefix, f"scene{scene_number}")
                code_dir = os.path.join(scene_dir, "code")
                media_dir = os.path.join(scene_dir, "media")
                
                os.makedirs(code_dir, exist_ok=True)
                os.makedirs(media_dir, exist_ok=True)
                
                # Get scene implementation for error fixing
                scene_implementation = scene_implementations.get(scene_number, "")
                
                # Create scene configuration with enhanced error handling support
                scene_config = {
                    'code': code,
                    'file_prefix': file_prefix,
                    'curr_scene': scene_number,
                    'curr_version': 1,
                    'code_dir': code_dir,
                    'media_dir': media_dir,
                    'quality': state.get('default_quality', 'medium'),
                    'max_retries': state.get('max_retries', 5),
                    'use_visual_fix_code': state.get('use_visual_fix_code', False),
                    'visual_self_reflection_func': None,  # Will be set if needed
                    'banned_reasonings': state.get('banned_reasonings', []),
                    'scene_trace_id': f"scene_{scene_number}_{state['session_id']}",
                    'topic': state['topic'],
                    'session_id': state['session_id'],
                    'code_generator': None,  # Will be set if needed for error fixing
                    'scene_implementation': scene_implementation,
                    'description': state['description'],
                    'scene_outline': state.get('scene_outline', '')
                }
                
                scene_configs.append(scene_config)
            
            self.log_agent_action("rendering_scenes_parallel", {
                'scene_count': len(scene_configs),
                'max_concurrent': state.get('max_concurrent_renders', 4),
                'caching_enabled': state.get('enable_caching', True),
                'gpu_acceleration': state.get('use_gpu_acceleration', False)
            })
            
            # Render multiple scenes in parallel using existing _run_manim_optimized and caching logic
            render_results = await video_renderer.render_multiple_scenes_parallel(
                scene_configs=scene_configs,
                max_concurrent=state.get('max_concurrent_renders', 4)
            )
            
            # Process render results with enhanced error tracking
            rendered_videos = {}
            rendering_errors = {}
            updated_code = {}
            
            for i, result in enumerate(render_results):
                scene_number = list(generated_code.keys())[i]
                
                if isinstance(result, Exception):
                    rendering_errors[scene_number] = str(result)
                    logger.error(f"Rendering failed for scene {scene_number}: {result}")
                else:
                    final_code, error = result
                    if error:
                        rendering_errors[scene_number] = error
                        logger.error(f"Rendering error for scene {scene_number}: {error}")
                    else:
                        # Update code if it was modified during rendering (e.g., by visual fixes)
                        if final_code != generated_code[scene_number]:
                            updated_code[scene_number] = final_code
                        
                        # Find the rendered video file
                        try:
                            video_path = self._find_rendered_video(file_prefix, scene_number, 1, 
                                                                 scene_configs[i]['media_dir'])
                            rendered_videos[scene_number] = video_path
                            
                            self.log_agent_action("scene_rendered_successfully", {
                                'scene_number': scene_number,
                                'video_path': video_path,
                                'code_updated': scene_number in updated_code
                            })
                        except Exception as find_error:
                            rendering_errors[scene_number] = f"Could not find rendered video: {find_error}"
            
            # Check if we have any successful renders
            if not rendered_videos:
                raise ValueError("Failed to render any scenes successfully")
            
            # Get performance statistics from video renderer
            performance_stats = video_renderer.get_performance_stats()
            
            self.log_agent_action("rendering_completed", {
                'successful_scenes': len(rendered_videos),
                'failed_scenes': len(rendering_errors),
                'total_scenes': len(generated_code),
                'cache_hit_rate': performance_stats.get('cache_hit_rate', 0),
                'average_render_time': performance_stats.get('average_time', 0)
            })
            
            # Combine videos if we have multiple successful renders using existing optimization features
            combined_video_path = None
            if len(rendered_videos) > 1:
                try:
                    self.log_agent_action("combining_videos", {
                        'video_count': len(rendered_videos),
                        'use_hardware_acceleration': state.get('use_gpu_acceleration', False)
                    })
                    
                    # Use _combine_with_audio_optimized and _combine_without_audio_optimized through the renderer
                    combined_video_path = await video_renderer.combine_videos_optimized(
                        topic=state['topic'],
                        use_hardware_acceleration=state.get('use_gpu_acceleration', False)
                    )
                    
                    self.log_agent_action("videos_combined", {
                        'combined_video_path': combined_video_path
                    })
                    
                except Exception as combine_error:
                    logger.error(f"Video combination failed: {combine_error}")
                    # Continue without combined video - partial success is acceptable
            elif len(rendered_videos) == 1:
                # Single video case
                combined_video_path = list(rendered_videos.values())[0]
            
            # Update state with rendering results and performance metrics
            state_update = {
                "rendered_videos": rendered_videos,
                "rendering_errors": rendering_errors,
                "combined_video_path": combined_video_path,
                "current_agent": "renderer_agent"
            }
            
            # Update generated code if any scenes were modified during rendering
            if updated_code:
                current_generated_code = state.get('generated_code', {})
                current_generated_code.update(updated_code)
                state_update["generated_code"] = current_generated_code
            
            # Add performance metrics compatible with current performance tracking
            current_metrics = state.get('performance_metrics', {})
            current_metrics.update({
                'renderer_agent': {
                    'render_stats': performance_stats,
                    'scenes_rendered': len(rendered_videos),
                    'scenes_failed': len(rendering_errors),
                    'success_rate': len(rendered_videos) / len(generated_code),
                    'video_combination_success': bool(combined_video_path)
                }
            })
            state_update["performance_metrics"] = current_metrics
            
            # Route to VisualAnalysisAgent for error detection while maintaining current visual fix workflow
            if state.get('use_visual_fix_code', False) and rendered_videos:
                state_update.update({
                    "next_agent": "visual_analysis_agent"
                })
                return Command(
                    goto="visual_analysis_agent",
                    update=state_update
                )
            else:
                # Complete workflow
                state_update.update({
                    "workflow_complete": True
                })
                return Command(
                    goto="END",
                    update=state_update
                )
            
        except Exception as e:
            logger.error(f"Error in RendererAgent execution: {e}")
            
            # Check if we should escalate to human
            if self.should_escalate_to_human(state):
                # Use the human intervention interface for more structured interaction
                from ..interfaces.human_intervention_interface import (
                    HumanInterventionInterface,
                    ErrorResolutionRequest,
                    InterventionPriority
                )
                
                interface = self.get_human_intervention_interface()
                
                error_request = ErrorResolutionRequest(
                    error_description=f"Video rendering failed for topic '{state.get('topic', '')}'",
                    error_context={
                        "error_message": str(e),
                        "error_type": type(e).__name__,
                        "scene_count": len(state.get('generated_code', {})),
                        "current_quality": state.get('default_quality', 'medium'),
                        "gpu_acceleration": state.get('use_gpu_acceleration', False)
                    },
                    recovery_options=[
                        {
                            "id": "retry_lower_quality",
                            "label": "Retry with lower quality",
                            "description": "Reduce rendering quality to improve stability",
                            "consequences": "Lower video quality but higher success rate",
                            "recommended": True
                        },
                        {
                            "id": "skip_failed_scenes",
                            "label": "Skip failed scenes",
                            "description": "Continue with successfully rendered scenes only",
                            "consequences": "Incomplete video but partial results available"
                        },
                        {
                            "id": "manual_intervention",
                            "label": "Manual intervention",
                            "description": "Pause workflow for manual debugging",
                            "consequences": "Workflow paused until manual resolution"
                        },
                        {
                            "id": "disable_gpu",
                            "label": "Disable GPU acceleration",
                            "description": "Retry without GPU acceleration for stability",
                            "consequences": "Slower rendering but potentially more stable"
                        }
                    ],
                    impact_assessment="High - Video generation cannot proceed without resolution",
                    urgency=InterventionPriority.HIGH
                )
                
                return await interface.request_error_resolution(error_request, state)
            
            # Handle error through base class
            return await self.handle_error(e, state)
    
    def _find_rendered_video(self, file_prefix: str, scene: int, version: int, media_dir: str) -> str:
        """Find the rendered video file (compatible with existing method).
        
        Args:
            file_prefix: File prefix for the video
            scene: Scene number
            version: Version number
            media_dir: Media directory
            
        Returns:
            str: Path to rendered video file
        """
        video_dir = os.path.join(media_dir, "videos", f"{file_prefix}_scene{scene}_v{version}")
        
        # Look in quality-specific subdirectories
        for quality_dir in ["1080p60", "720p30", "480p15"]:
            search_dir = os.path.join(video_dir, quality_dir)
            if os.path.exists(search_dir):
                for file in os.listdir(search_dir):
                    if file.endswith('.mp4'):
                        return os.path.join(search_dir, file)
        
        raise FileNotFoundError(f"No rendered video found for scene {scene} version {version}")
    
    async def render_scene_optimized(self,
                                   code: str,
                                   file_prefix: str,
                                   curr_scene: int,
                                   curr_version: int,
                                   code_dir: str,
                                   media_dir: str,
                                   state: VideoGenerationState,
                                   quality: str = None,
                                   max_retries: int = 3,
                                   **kwargs) -> tuple:
        """Render single scene with optimization (compatible with existing API).
        
        Args:
            code: Manim code to render
            file_prefix: File prefix
            curr_scene: Current scene number
            curr_version: Current version number
            code_dir: Code directory
            media_dir: Media directory
            state: Current workflow state
            quality: Rendering quality
            max_retries: Maximum retry attempts
            **kwargs: Additional arguments
            
        Returns:
            tuple: (final_code, error_message)
        """
        video_renderer = self._get_video_renderer(state)
        return await video_renderer.render_scene_optimized(
            code=code,
            file_prefix=file_prefix,
            curr_scene=curr_scene,
            curr_version=curr_version,
            code_dir=code_dir,
            media_dir=media_dir,
            quality=quality or state.get('default_quality', 'medium'),
            max_retries=max_retries,
            **kwargs
        )
    
    async def render_multiple_scenes_parallel(self,
                                            scene_configs: List[Dict],
                                            state: VideoGenerationState,
                                            max_concurrent: int = None) -> List[tuple]:
        """Render multiple scenes in parallel (compatible with existing API).
        
        Args:
            scene_configs: List of scene configurations
            state: Current workflow state
            max_concurrent: Maximum concurrent renders
            
        Returns:
            List[tuple]: List of render results
        """
        video_renderer = self._get_video_renderer(state)
        return await video_renderer.render_multiple_scenes_parallel(
            scene_configs=scene_configs,
            max_concurrent=max_concurrent or state.get('max_concurrent_renders', 4)
        )
    
    async def combine_videos_optimized(self,
                                     topic: str,
                                     state: VideoGenerationState,
                                     use_hardware_acceleration: bool = False) -> str:
        """Combine videos with optimization (compatible with existing API).
        
        Args:
            topic: Video topic
            state: Current workflow state
            use_hardware_acceleration: Whether to use hardware acceleration
            
        Returns:
            str: Path to combined video
        """
        video_renderer = self._get_video_renderer(state)
        return await video_renderer.combine_videos_optimized(
            topic=topic,
            use_hardware_acceleration=use_hardware_acceleration or state.get('use_gpu_acceleration', False)
        )
    
    def get_rendering_status(self, state: VideoGenerationState) -> Dict[str, Any]:
        """Get current rendering status and metrics.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dict: Rendering status information
        """
        rendered_videos = state.get('rendered_videos', {})
        rendering_errors = state.get('rendering_errors', {})
        
        # Get performance stats from video renderer if available
        performance_stats = {}
        if self._video_renderer:
            performance_stats = self._video_renderer.get_performance_stats()
        
        return {
            'agent_name': self.name,
            'scenes_rendered': len(rendered_videos),
            'scenes_with_errors': len(rendering_errors),
            'total_scenes': len(state.get('generated_code', {})),
            'success_rate': len(rendered_videos) / max(1, len(state.get('generated_code', {}))),
            'combined_video_available': bool(state.get('combined_video_path')),
            'execution_stats': self.execution_stats,
            'caching_enabled': state.get('enable_caching', True),
            'gpu_acceleration': state.get('use_gpu_acceleration', False),
            'rendering_quality': state.get('default_quality', 'medium'),
            'performance_stats': performance_stats,
            'cache_hit_rate': performance_stats.get('cache_hit_rate', 0),
            'average_render_time': performance_stats.get('average_time', 0)
        }
    
    async def handle_rendering_error(self,
                                   error: Exception,
                                   state: VideoGenerationState,
                                   scene_number: Optional[int] = None,
                                   retry_with_lower_quality: bool = True) -> Command:
        """Handle rendering-specific errors with recovery strategies.
        
        Args:
            error: Exception that occurred
            state: Current workflow state
            scene_number: Scene number where error occurred
            retry_with_lower_quality: Whether to retry with lower quality
            
        Returns:
            Command: LangGraph command for error handling
        """
        error_type = type(error).__name__
        
        # Rendering-specific error handling
        if "render" in str(error).lower() and retry_with_lower_quality:
            self.log_agent_action("retrying_with_lower_quality", {
                'error_type': error_type,
                'scene_number': scene_number,
                'retry_strategy': 'lower_quality'
            })
            
            # Update state to use lower quality
            fallback_state = state.copy()
            current_quality = state.get('default_quality', 'medium')
            
            # Quality fallback chain
            quality_fallback = {
                'high': 'medium',
                'medium': 'low',
                'low': 'preview'
            }
            
            new_quality = quality_fallback.get(current_quality, 'preview')
            fallback_state.update({
                'default_quality': new_quality,
                'use_gpu_acceleration': False,  # Disable GPU for stability
                'max_concurrent_renders': 1    # Reduce concurrency
            })
            
            try:
                return await self.execute(fallback_state)
            except Exception as fallback_error:
                logger.error(f"Fallback rendering also failed: {fallback_error}")
        
        # Use base error handling
        return await self.handle_error(error, state)
    
    def cleanup_cache(self, max_age_days: int = 7):
        """Clean up old cache files using existing caching logic.
        
        Args:
            max_age_days: Maximum age of cache files in days
        """
        if self._video_renderer:
            self._video_renderer.cleanup_cache(max_age_days)
            self.log_agent_action("cache_cleanup", {
                'max_age_days': max_age_days
            })
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics from the video renderer.
        
        Returns:
            Dict: Cache statistics
        """
        if self._video_renderer:
            stats = self._video_renderer.get_performance_stats()
            return {
                'cache_enabled': stats.get('cache_enabled', False),
                'cache_hits': stats.get('cache_hits', 0),
                'total_renders': stats.get('total_renders', 0),
                'cache_hit_rate': stats.get('cache_hit_rate', 0)
            }
        return {'cache_enabled': False, 'cache_hits': 0, 'total_renders': 0, 'cache_hit_rate': 0}
    
    async def optimize_rendering_settings(self, state: VideoGenerationState) -> Dict[str, Any]:
        """Optimize rendering settings based on system capabilities and current performance.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dict: Optimized settings
        """
        # Get current performance metrics
        performance_metrics = state.get('performance_metrics', {})
        renderer_metrics = performance_metrics.get('renderer_agent', {})
        
        # Base optimization settings
        optimized_settings = {
            'default_quality': state.get('default_quality', 'medium'),
            'max_concurrent_renders': state.get('max_concurrent_renders', 4),
            'enable_caching': state.get('enable_caching', True),
            'use_gpu_acceleration': state.get('use_gpu_acceleration', False)
        }
        
        # Adjust based on success rate
        success_rate = renderer_metrics.get('success_rate', 1.0)
        if success_rate < 0.8:
            # Reduce quality and concurrency for better stability
            optimized_settings.update({
                'default_quality': 'low',
                'max_concurrent_renders': max(1, optimized_settings['max_concurrent_renders'] // 2),
                'use_gpu_acceleration': False
            })
            self.log_agent_action("optimizing_for_stability", optimized_settings)
        
        # Adjust based on average render time
        avg_render_time = renderer_metrics.get('render_stats', {}).get('average_time', 0)
        if avg_render_time > 60:  # More than 1 minute per scene
            optimized_settings.update({
                'default_quality': 'low',
                'preview_mode': True
            })
            self.log_agent_action("optimizing_for_speed", optimized_settings)
        
        return optimized_settings
    
    async def handle_partial_rendering_failure(self, 
                                             state: VideoGenerationState,
                                             failed_scenes: List[int]) -> Command:
        """Handle partial rendering failures by attempting recovery strategies.
        
        Args:
            state: Current workflow state
            failed_scenes: List of scene numbers that failed
            
        Returns:
            Command: LangGraph command for recovery action
        """
        self.log_agent_action("handling_partial_failure", {
            'failed_scenes': failed_scenes,
            'total_scenes': len(state.get('generated_code', {}))
        })
        
        # If less than half failed, continue with successful renders
        total_scenes = len(state.get('generated_code', {}))
        if len(failed_scenes) < total_scenes / 2:
            rendered_videos = state.get('rendered_videos', {})
            if rendered_videos:
                self.log_agent_action("continuing_with_partial_success", {
                    'successful_scenes': len(rendered_videos),
                    'failed_scenes': len(failed_scenes)
                })
                
                # Try to combine available videos
                try:
                    video_renderer = self._get_video_renderer(state)
                    combined_video_path = await video_renderer.combine_videos_optimized(
                        topic=state['topic'],
                        use_hardware_acceleration=False  # Use stable settings
                    )
                    
                    return Command(
                        goto="END",
                        update={
                            "combined_video_path": combined_video_path,
                            "workflow_complete": True,
                            "partial_success": True,
                            "failed_scenes": failed_scenes
                        }
                    )
                except Exception as combine_error:
                    logger.error(f"Failed to combine partial results: {combine_error}")
        
        # If too many failures, escalate to error handling
        return await self.handle_error(
            ValueError(f"Too many rendering failures: {len(failed_scenes)}/{total_scenes} scenes failed"),
            state
        )
    
    def __del__(self):
        """Cleanup resources when agent is destroyed."""
        if self._video_renderer and hasattr(self._video_renderer, 'executor'):
            try:
                self._video_renderer.executor.shutdown(wait=False)
            except Exception:
                pass  # Ignore cleanup errors
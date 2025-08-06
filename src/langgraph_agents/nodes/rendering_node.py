"""
Rendering node function following LangGraph patterns.

This module implements the rendering_node function that converts RendererAgent logic
to a simple async function using RenderingService for business logic.
"""

import logging
import os
from typing import Dict, Any, List, Tuple

from ..models.state import VideoGenerationState
from ..models.errors import WorkflowError, ErrorType, ErrorSeverity
from ..services.rendering_service import RenderingService

logger = logging.getLogger(__name__)


async def rendering_node(state: VideoGenerationState) -> VideoGenerationState:
    """
    Rendering node function that renders Manim code to videos.
    
    This function follows LangGraph best practices by:
    - Taking state as input and returning updated state
    - Using RenderingService for business logic separation
    - Implementing resource management and cleanup
    - Adding comprehensive error handling and logging
    
    Args:
        state: Current workflow state
        
    Returns:
        VideoGenerationState: Updated state with rendered videos
    """
    logger.info(f"Starting rendering node for session {state.session_id}")
    
    # Update current step
    state.current_step = "rendering"
    state.add_execution_trace("rendering_node", {
        "action": "started", 
        "session_id": state.session_id,
        "scenes_to_render": len(state.generated_code)
    })
    
    try:
        # Validate input state
        validation_error = _validate_input_state(state)
        if validation_error:
            state.add_error(validation_error)
            return state
        
        # Initialize rendering service
        rendering_config = _build_rendering_config(state)
        rendering_service = RenderingService(rendering_config)
        
        # Create scene configurations for rendering
        scene_configs = _create_scene_configs(state)
        
        # Determine if we should use parallel rendering
        max_concurrent = min(state.config.max_concurrent_renders, len(scene_configs))
        use_parallel = len(scene_configs) > 1 and max_concurrent > 1
        
        if use_parallel:
            logger.info(f"Using parallel rendering with max_concurrent={max_concurrent}")
            rendering_results = await _render_scenes_parallel(
                rendering_service, scene_configs, max_concurrent
            )
        else:
            logger.info("Using sequential rendering")
            rendering_results = await _render_scenes_sequential(
                rendering_service, scene_configs
            )
        
        # Process rendering results and update state
        successful_renders = 0
        for i, (final_code, error) in enumerate(rendering_results):
            scene_num = scene_configs[i]['curr_scene']
            
            if error:
                # Handle rendering error
                error_obj = WorkflowError(
                    step="rendering",
                    error_type=ErrorType.RENDERING,
                    message=f"Rendering failed for scene {scene_num}: {error}",
                    severity=ErrorSeverity.HIGH,
                    scene_number=scene_num,
                    context={"rendering_error": error}
                )
                state.add_error(error_obj)
                state.rendering_errors[scene_num] = error
                
                # Update generated code with final version (may have been fixed during rendering)
                if final_code and final_code != state.generated_code.get(scene_num):
                    state.generated_code[scene_num] = final_code
            else:
                # Successful rendering - find the rendered video file
                try:
                    video_path = rendering_service.find_rendered_video(
                        file_prefix=scene_configs[i]['file_prefix'],
                        scene_number=scene_num,
                        version=scene_configs[i]['curr_version']
                    )
                    state.rendered_videos[scene_num] = video_path
                    successful_renders += 1
                    
                    # Update generated code with final version
                    if final_code and final_code != state.generated_code.get(scene_num):
                        state.generated_code[scene_num] = final_code
                    
                    logger.info(f"Successfully rendered scene {scene_num}: {video_path}")
                    
                except FileNotFoundError as e:
                    # Video file not found after rendering
                    error_obj = WorkflowError(
                        step="rendering",
                        error_type=ErrorType.SYSTEM,
                        message=f"Rendered video not found for scene {scene_num}: {str(e)}",
                        severity=ErrorSeverity.HIGH,
                        scene_number=scene_num,
                        context={"file_error": str(e)}
                    )
                    state.add_error(error_obj)
                    state.rendering_errors[scene_num] = str(e)
        
        # Combine videos if we have successful renders
        if successful_renders > 0 and len(state.rendered_videos) > 1:
            try:
                logger.info(f"Combining {len(state.rendered_videos)} rendered videos")
                combined_video_path = await rendering_service.combine_videos(
                    topic=state.topic,
                    rendered_videos=state.rendered_videos,
                    use_hardware_acceleration=state.config.use_gpu_acceleration
                )
                state.combined_video_path = combined_video_path
                logger.info(f"Successfully combined videos: {combined_video_path}")
                
            except Exception as e:
                error_obj = WorkflowError(
                    step="rendering",
                    error_type=ErrorType.SYSTEM,
                    message=f"Video combination failed: {str(e)}",
                    severity=ErrorSeverity.MEDIUM,
                    context={"combination_error": str(e)}
                )
                state.add_error(error_obj)
                logger.error(f"Video combination failed: {e}")
        elif successful_renders == 1:
            # Single video - no need to combine
            single_video_path = list(state.rendered_videos.values())[0]
            state.combined_video_path = single_video_path
            logger.info(f"Single video rendering completed: {single_video_path}")
        
        # Update metrics if available
        if state.metrics:
            rendering_metrics = rendering_service.get_performance_stats()
            rendering_metrics.update({
                "scenes_processed": len(scene_configs),
                "successful_renders": successful_renders,
                "failed_renders": len(scene_configs) - successful_renders,
                "parallel_rendering": use_parallel,
                "max_concurrent": max_concurrent if use_parallel else 1,
                "videos_combined": state.combined_video_path is not None
            })
            state.metrics.add_step_metrics("rendering", rendering_metrics)
        
        # Add completion trace
        state.add_execution_trace("rendering_node", {
            "action": "completed",
            "successful_renders": successful_renders,
            "failed_renders": len(scene_configs) - successful_renders,
            "total_scenes": len(scene_configs),
            "combined_video": state.combined_video_path is not None,
            "has_errors": state.has_errors(),
            "parallel_rendering": use_parallel
        })
        
        logger.info(f"Rendering completed: {successful_renders}/{len(scene_configs)} successful")
        
        # Cleanup rendering service
        await rendering_service.cleanup()
        
        return state
        
    except Exception as e:
        logger.error(f"Rendering node failed: {str(e)}")
        
        # Create workflow error
        error = WorkflowError(
            step="rendering",
            error_type=ErrorType.SYSTEM,
            message=f"Rendering failed: {str(e)}",
            severity=ErrorSeverity.CRITICAL,
            context={"exception": str(e), "exception_type": type(e).__name__}
        )
        state.add_error(error)
        
        # Add failure trace
        state.add_execution_trace("rendering_node", {
            "action": "failed",
            "error": str(e),
            "error_type": type(e).__name__
        })
        
        return state


async def _render_scenes_parallel(
    rendering_service: RenderingService,
    scene_configs: List[Dict[str, Any]],
    max_concurrent: int
) -> List[Tuple[str, str]]:
    """Render multiple scenes in parallel."""
    try:
        results = await rendering_service.render_multiple_scenes_parallel(
            scene_configs=scene_configs,
            max_concurrent=max_concurrent
        )
        return results
        
    except Exception as e:
        logger.error(f"Parallel rendering failed: {e}")
        # Fall back to sequential rendering
        logger.info("Falling back to sequential rendering")
        return await _render_scenes_sequential(rendering_service, scene_configs)


async def _render_scenes_sequential(
    rendering_service: RenderingService,
    scene_configs: List[Dict[str, Any]]
) -> List[Tuple[str, str]]:
    """Render scenes sequentially."""
    results = []
    
    for config in scene_configs:
        try:
            logger.info(f"Rendering scene {config['curr_scene']}")
            
            final_code, error = await rendering_service.render_scene(
                code=config['code'],
                file_prefix=config['file_prefix'],
                scene_number=config['curr_scene'],
                version=config['curr_version'],
                quality=config.get('quality'),
                scene_implementation=config.get('scene_implementation'),
                topic=config.get('topic'),
                description=config.get('description'),
                scene_outline=config.get('scene_outline'),
                session_id=config.get('session_id')
            )
            
            results.append((final_code, error))
            
        except Exception as e:
            logger.error(f"Failed to render scene {config['curr_scene']}: {e}")
            results.append((config['code'], str(e)))
    
    return results


def _validate_input_state(state: VideoGenerationState) -> WorkflowError:
    """Validate input state for rendering."""
    if not state.generated_code:
        return WorkflowError(
            step="rendering",
            error_type=ErrorType.VALIDATION,
            message="Generated code is required for rendering",
            severity=ErrorSeverity.HIGH,
            context={"missing_field": "generated_code"}
        )
    
    # Validate individual generated code entries
    for scene_num, code in state.generated_code.items():
        if not code or not code.strip():
            return WorkflowError(
                step="rendering",
                error_type=ErrorType.VALIDATION,
                message=f"Generated code for scene {scene_num} is empty",
                severity=ErrorSeverity.HIGH,
                scene_number=scene_num,
                context={"empty_code": scene_num}
            )
    
    return None


def _build_rendering_config(state: VideoGenerationState) -> Dict[str, Any]:
    """Build configuration for RenderingService from state."""
    config = {
        'output_dir': state.config.output_dir,
        'max_concurrent_renders': state.config.max_concurrent_renders,
        'enable_caching': state.config.enable_caching,
        'default_quality': state.config.default_quality,
        'use_gpu_acceleration': state.config.use_gpu_acceleration,
        'preview_mode': state.config.preview_mode,
        'use_visual_fix_code': state.config.use_visual_analysis,
        'max_retries': state.config.max_retries,
        'print_response': False  # Keep logs clean in production
    }
    
    return config


def _create_scene_configs(state: VideoGenerationState) -> List[Dict[str, Any]]:
    """Create scene configurations for rendering."""
    configs = []
    file_prefix = f"{state.session_id}_{state.topic.replace(' ', '_').lower()}"
    
    for scene_num, code in state.generated_code.items():
        config = {
            'code': code,
            'file_prefix': file_prefix,
            'curr_scene': scene_num,
            'curr_version': 1,
            'quality': state.config.default_quality,
            'scene_implementation': state.scene_implementations.get(scene_num),
            'topic': state.topic,
            'description': state.description,
            'scene_outline': state.scene_outline,
            'session_id': state.session_id
        }
        configs.append(config)
    
    return configs
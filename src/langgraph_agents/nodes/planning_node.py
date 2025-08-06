"""
Planning node function following LangGraph patterns.

This module implements the planning_node function that converts PlannerAgent logic
to a simple async function using PlanningService for business logic.
"""

import logging
from typing import Dict, Any

from ..models.state import VideoGenerationState
from ..models.errors import WorkflowError, ErrorType, ErrorSeverity
from ..services.planning_service import PlanningService
from ..error_recovery.error_handler import ErrorHandler

logger = logging.getLogger(__name__)


async def planning_node(state: VideoGenerationState) -> VideoGenerationState:
    """
    Planning node function that generates scene outline and implementations.
    
    This function follows LangGraph best practices by:
    - Taking state as input and returning updated state
    - Using PlanningService for business logic separation
    - Implementing comprehensive error handling
    - Adding proper logging and tracing
    
    Args:
        state: Current workflow state
        
    Returns:
        VideoGenerationState: Updated state with planning results
    """
    logger.info(f"Starting planning node for session {state.session_id}")
    
    # Update current step
    state.current_step = "planning"
    state.add_execution_trace("planning_node", {"action": "started", "session_id": state.session_id})
    
    try:
        # Validate input state
        if not state.topic or not state.description:
            error = WorkflowError(
                step="planning",
                error_type=ErrorType.VALIDATION,
                message="Topic and description are required for planning",
                severity=ErrorSeverity.HIGH,
                context={"topic": state.topic, "description": state.description}
            )
            state.add_error(error)
            return state
        
        # Initialize planning service
        planning_config = _build_planning_config(state)
        planning_service = PlanningService(planning_config)
        
        # Get model wrappers from state config
        model_wrappers = _get_model_wrappers(state)
        
        # Generate scene outline
        logger.info(f"Generating scene outline for topic: {state.topic}")
        scene_outline = await planning_service.generate_scene_outline(
            topic=state.topic,
            description=state.description,
            session_id=state.session_id,
            model_wrappers=model_wrappers
        )
        
        # Validate scene outline
        is_valid, issues = await planning_service.validate_scene_outline(scene_outline)
        if not is_valid:
            error = WorkflowError(
                step="planning",
                error_type=ErrorType.CONTENT,
                message=f"Scene outline validation failed: {'; '.join(issues)}",
                severity=ErrorSeverity.MEDIUM,
                context={"validation_issues": issues}
            )
            state.add_error(error)
            # Continue with potentially invalid outline for now
        
        # Update state with scene outline
        state.scene_outline = scene_outline
        
        # Generate scene implementations
        logger.info("Generating scene implementations")
        scene_implementations_list = await planning_service.generate_scene_implementations(
            topic=state.topic,
            description=state.description,
            plan=scene_outline,
            session_id=state.session_id,
            model_wrappers=model_wrappers
        )
        
        # Convert list to dictionary with scene numbers
        scene_implementations = {}
        for i, implementation in enumerate(scene_implementations_list, 1):
            scene_implementations[i] = implementation
        
        # Validate scene implementations
        all_valid, issues_by_scene = await planning_service.validate_scene_implementations(
            scene_implementations_list
        )
        
        if not all_valid:
            for scene_num, scene_issues in issues_by_scene.items():
                error = WorkflowError(
                    step="planning",
                    error_type=ErrorType.CONTENT,
                    message=f"Scene {scene_num} implementation validation failed: {'; '.join(scene_issues)}",
                    severity=ErrorSeverity.MEDIUM,
                    scene_number=scene_num,
                    context={"validation_issues": scene_issues}
                )
                state.add_error(error)
        
        # Update state with scene implementations
        state.scene_implementations = scene_implementations
        
        # Detect relevant plugins
        logger.info("Detecting relevant plugins")
        try:
            detected_plugins = await planning_service.detect_plugins(
                topic=state.topic,
                description=state.description,
                model_wrappers=model_wrappers
            )
            state.detected_plugins = detected_plugins
            logger.info(f"Detected {len(detected_plugins)} plugins: {detected_plugins}")
        except Exception as e:
            logger.warning(f"Plugin detection failed: {e}")
            # Plugin detection failure is not critical
            state.detected_plugins = []
        
        # Update metrics if available
        if state.metrics:
            planning_metrics = planning_service.get_planning_metrics()
            state.metrics.add_step_metrics("planning", planning_metrics)
        
        # Add successful completion trace
        state.add_execution_trace("planning_node", {
            "action": "completed",
            "scene_count": len(state.scene_implementations),
            "plugins_detected": len(state.detected_plugins),
            "has_errors": state.has_errors()
        })
        
        logger.info(f"Planning completed successfully with {len(state.scene_implementations)} scenes")
        
        # Cleanup planning service
        await planning_service.cleanup()
        
        return state
        
    except Exception as e:
        logger.error(f"Planning node failed: {str(e)}")
        
        # Create workflow error
        error = WorkflowError(
            step="planning",
            error_type=ErrorType.SYSTEM,
            message=f"Planning failed: {str(e)}",
            severity=ErrorSeverity.CRITICAL,
            context={"exception": str(e), "exception_type": type(e).__name__}
        )
        state.add_error(error)
        
        # Add failure trace
        state.add_execution_trace("planning_node", {
            "action": "failed",
            "error": str(e),
            "error_type": type(e).__name__
        })
        
        return state


def _build_planning_config(state: VideoGenerationState) -> Dict[str, Any]:
    """Build configuration for PlanningService from state."""
    config = {
        'planner_model': f"{state.config.planner_model.provider}/{state.config.planner_model.model_name}",
        'helper_model': f"{state.config.helper_model.provider}/{state.config.helper_model.model_name}",
        'output_dir': state.config.output_dir,
        'use_rag': state.config.use_rag,
        'use_context_learning': state.config.use_context_learning,
        'context_learning_path': state.config.context_learning_path,
        'chroma_db_path': state.config.chroma_db_path,
        'manim_docs_path': state.config.manim_docs_path,
        'embedding_model': state.config.embedding_model,
        'use_langfuse': state.config.use_langfuse,
        'max_scene_concurrency': state.config.max_scene_concurrency,
        'enable_caching': state.config.enable_caching,
        'use_enhanced_rag': state.config.use_enhanced_rag,
        'session_id': state.session_id,
        'print_response': False  # Keep logs clean in production
    }
    
    return config


def _get_model_wrappers(state: VideoGenerationState) -> Dict[str, Any]:
    """Get model wrappers from state configuration.
    
    Note: This is a placeholder implementation. In the actual system,
    model wrappers would be created from the model configuration.
    """
    # This would typically create actual model wrapper instances
    # For now, return configuration that can be used by the service
    return {
        'planner_model': {
            'provider': state.config.planner_model.provider,
            'model_name': state.config.planner_model.model_name,
            'temperature': state.config.planner_model.temperature,
            'max_tokens': state.config.planner_model.max_tokens
        },
        'helper_model': {
            'provider': state.config.helper_model.provider,
            'model_name': state.config.helper_model.model_name,
            'temperature': state.config.helper_model.temperature,
            'max_tokens': state.config.helper_model.max_tokens
        }
    }
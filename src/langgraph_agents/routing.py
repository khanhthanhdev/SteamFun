"""
Conditional routing functions for the video generation workflow.

This module implements routing logic that determines the next step in the workflow
based on the current state conditions. It follows LangGraph best practices for
conditional edges and provides clear, testable routing decisions.
"""

import logging
from typing import Literal, Dict, Any, Optional
from .models.state import VideoGenerationState
from .models.errors import WorkflowError, ErrorType

logger = logging.getLogger(__name__)

# Type definitions for routing decisions
PlanningRoute = Literal["code_generation", "error_handler", "human_loop"]
CodeGenerationRoute = Literal["rendering", "rag_enhancement", "error_handler", "human_loop"]
RenderingRoute = Literal["visual_analysis", "complete", "error_handler", "human_loop"]
ErrorHandlerRoute = Literal["planning", "code_generation", "rendering", "human_loop", "complete"]


def route_from_planning(state: VideoGenerationState) -> PlanningRoute:
    """
    Route from planning step based on planning results and error conditions.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next step in the workflow
        
    Routing Logic:
        - If planning succeeded and scene outline exists -> code_generation
        - If planning failed with recoverable error -> error_handler
        - If planning failed with non-recoverable error -> human_loop
        - If max retries exceeded -> human_loop
    """
    logger.info(f"Routing from planning step for session {state.session_id}")
    
    # Add execution trace
    state.add_execution_trace("routing_from_planning", {
        "scene_outline_exists": bool(state.scene_outline),
        "scene_implementations_count": len(state.scene_implementations),
        "error_count": len(state.errors),
        "retry_count": state.retry_counts.get("planning", 0)
    })
    
    # Check for workflow interruption
    if state.workflow_interrupted:
        logger.info("Workflow interrupted, routing to human_loop")
        return "human_loop"
    
    # Check for pending human input
    if state.pending_human_input:
        logger.info("Pending human input detected, routing to human_loop")
        return "human_loop"
    
    # Check retry limits
    max_retries = state.config.max_retries
    planning_retries = state.retry_counts.get("planning", 0)
    
    if planning_retries >= max_retries:
        logger.warning(f"Planning max retries ({max_retries}) exceeded, routing to human_loop")
        state.add_error(WorkflowError(
            step="planning",
            error_type=ErrorType.TIMEOUT,
            message=f"Planning failed after {planning_retries} attempts",
            recoverable=False
        ))
        return "human_loop"
    
    # Check for recent errors
    recent_planning_errors = [
        error for error in state.errors 
        if error.step == "planning" and error.retry_count == planning_retries
    ]
    
    if recent_planning_errors:
        latest_error = recent_planning_errors[-1]
        
        # Route based on error recoverability
        if not latest_error.recoverable:
            logger.error(f"Non-recoverable planning error: {latest_error.message}")
            return "human_loop"
        else:
            logger.info(f"Recoverable planning error detected: {latest_error.message}")
            return "error_handler"
    
    # Check planning success conditions
    if not state.scene_outline or not state.scene_outline.strip():
        logger.warning("No scene outline generated, routing to error_handler")
        state.add_error(WorkflowError(
            step="planning",
            error_type="missing_output",
            message="Scene outline is empty or missing",
            recoverable=True
        ))
        return "error_handler"
    
    if not state.scene_implementations:
        logger.warning("No scene implementations generated, routing to error_handler")
        state.add_error(WorkflowError(
            step="planning",
            error_type="missing_output",
            message="Scene implementations are empty",
            recoverable=True
        ))
        return "error_handler"
    
    # Validate scene implementations
    if len(state.scene_implementations) == 0:
        logger.warning("Empty scene implementations, routing to error_handler")
        return "error_handler"
    
    # Success case - proceed to code generation
    logger.info(f"Planning successful with {len(state.scene_implementations)} scenes, routing to code_generation")
    return "code_generation"


def route_from_code_generation(state: VideoGenerationState) -> CodeGenerationRoute:
    """
    Route from code generation step based on generation results and error conditions.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next step in the workflow
        
    Routing Logic:
        - If all scenes generated successfully -> rendering
        - If partial success and RAG available -> rag_enhancement
        - If generation failed with recoverable error -> error_handler
        - If generation failed with non-recoverable error -> human_loop
        - If max retries exceeded -> human_loop
    """
    logger.info(f"Routing from code_generation step for session {state.session_id}")
    
    # Add execution trace
    state.add_execution_trace("routing_from_code_generation", {
        "total_scenes": len(state.scene_implementations),
        "generated_scenes": len(state.generated_code),
        "code_errors": len(state.code_errors),
        "error_count": len(state.errors),
        "retry_count": state.retry_counts.get("code_generation", 0)
    })
    
    # Check for workflow interruption
    if state.workflow_interrupted:
        logger.info("Workflow interrupted, routing to human_loop")
        return "human_loop"
    
    # Check for pending human input
    if state.pending_human_input:
        logger.info("Pending human input detected, routing to human_loop")
        return "human_loop"
    
    # Check retry limits
    max_retries = state.config.max_retries
    code_gen_retries = state.retry_counts.get("code_generation", 0)
    
    if code_gen_retries >= max_retries:
        logger.warning(f"Code generation max retries ({max_retries}) exceeded, routing to human_loop")
        state.add_error(WorkflowError(
            step="code_generation",
            error_type="max_retries_exceeded",
            message=f"Code generation failed after {code_gen_retries} attempts",
            recoverable=False
        ))
        return "human_loop"
    
    # Check for recent errors
    recent_code_errors = [
        error for error in state.errors 
        if error.step == "code_generation" and error.retry_count == code_gen_retries
    ]
    
    if recent_code_errors:
        latest_error = recent_code_errors[-1]
        
        # Route based on error recoverability
        if not latest_error.recoverable:
            logger.error(f"Non-recoverable code generation error: {latest_error.message}")
            return "human_loop"
        else:
            logger.info(f"Recoverable code generation error detected: {latest_error.message}")
            return "error_handler"
    
    # Analyze code generation results
    total_scenes = len(state.scene_implementations)
    generated_scenes = len(state.generated_code)
    failed_scenes = len(state.code_errors)
    
    if total_scenes == 0:
        logger.error("No scenes to generate code for, routing to error_handler")
        return "error_handler"
    
    # Calculate success rate
    success_rate = generated_scenes / total_scenes if total_scenes > 0 else 0.0
    
    # Complete success - all scenes generated
    if generated_scenes == total_scenes and failed_scenes == 0:
        logger.info(f"All {total_scenes} scenes generated successfully, routing to rendering")
        return "rendering"
    
    # Partial success - check if RAG enhancement can help
    if generated_scenes > 0 and failed_scenes > 0:
        if state.config.use_rag and success_rate >= 0.5:
            logger.info(f"Partial success ({generated_scenes}/{total_scenes}), routing to rag_enhancement")
            return "rag_enhancement"
        else:
            logger.warning(f"Partial success ({generated_scenes}/{total_scenes}) but RAG disabled or low success rate")
            return "error_handler"
    
    # No successful generations
    if generated_scenes == 0:
        logger.error("No scenes generated successfully, routing to error_handler")
        state.add_error(WorkflowError(
            step="code_generation",
            error_type="complete_failure",
            message="Failed to generate code for any scenes",
            recoverable=True
        ))
        return "error_handler"
    
    # Default to error handler for unexpected cases
    logger.warning("Unexpected code generation state, routing to error_handler")
    return "error_handler"


def route_from_rendering(state: VideoGenerationState) -> RenderingRoute:
    """
    Route from rendering step based on rendering results and error conditions.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next step in the workflow
        
    Routing Logic:
        - If all scenes rendered successfully -> visual_analysis (if enabled) or complete
        - If rendering failed with recoverable error -> error_handler
        - If rendering failed with non-recoverable error -> human_loop
        - If max retries exceeded -> human_loop
        - If partial success but acceptable -> complete
    """
    logger.info(f"Routing from rendering step for session {state.session_id}")
    
    # Add execution trace
    state.add_execution_trace("routing_from_rendering", {
        "total_scenes": len(state.generated_code),
        "rendered_scenes": len(state.rendered_videos),
        "rendering_errors": len(state.rendering_errors),
        "combined_video_exists": bool(state.combined_video_path),
        "error_count": len(state.errors),
        "retry_count": state.retry_counts.get("rendering", 0)
    })
    
    # Check for workflow interruption
    if state.workflow_interrupted:
        logger.info("Workflow interrupted, routing to human_loop")
        return "human_loop"
    
    # Check for pending human input
    if state.pending_human_input:
        logger.info("Pending human input detected, routing to human_loop")
        return "human_loop"
    
    # Check retry limits
    max_retries = state.config.max_retries
    rendering_retries = state.retry_counts.get("rendering", 0)
    
    if rendering_retries >= max_retries:
        logger.warning(f"Rendering max retries ({max_retries}) exceeded, routing to human_loop")
        state.add_error(WorkflowError(
            step="rendering",
            error_type="max_retries_exceeded",
            message=f"Rendering failed after {rendering_retries} attempts",
            recoverable=False
        ))
        return "human_loop"
    
    # Check for recent errors
    recent_rendering_errors = [
        error for error in state.errors 
        if error.step == "rendering" and error.retry_count == rendering_retries
    ]
    
    if recent_rendering_errors:
        latest_error = recent_rendering_errors[-1]
        
        # Route based on error recoverability
        if not latest_error.recoverable:
            logger.error(f"Non-recoverable rendering error: {latest_error.message}")
            return "human_loop"
        else:
            logger.info(f"Recoverable rendering error detected: {latest_error.message}")
            return "error_handler"
    
    # Analyze rendering results
    total_scenes_to_render = len(state.generated_code)
    rendered_scenes = len(state.rendered_videos)
    failed_renders = len(state.rendering_errors)
    
    if total_scenes_to_render == 0:
        logger.error("No scenes to render, routing to error_handler")
        return "error_handler"
    
    # Calculate success rate
    success_rate = rendered_scenes / total_scenes_to_render if total_scenes_to_render > 0 else 0.0
    
    # Complete success - all scenes rendered
    if rendered_scenes == total_scenes_to_render and failed_renders == 0:
        # Check if combined video was created
        if not state.combined_video_path:
            logger.warning("Individual scenes rendered but no combined video, routing to error_handler")
            return "error_handler"
        
        # Decide next step based on configuration
        if state.config.use_visual_analysis:
            logger.info(f"All {total_scenes_to_render} scenes rendered successfully, routing to visual_analysis")
            return "visual_analysis"
        else:
            logger.info(f"All {total_scenes_to_render} scenes rendered successfully, workflow complete")
            return "complete"
    
    # Partial success - check if acceptable
    if rendered_scenes > 0 and success_rate >= 0.7:  # 70% success threshold
        logger.info(f"Partial rendering success ({rendered_scenes}/{total_scenes_to_render}), acceptable threshold met")
        
        # If we have a combined video, we can complete
        if state.combined_video_path:
            if state.config.use_visual_analysis:
                return "visual_analysis"
            else:
                return "complete"
        else:
            # Try to recover and create combined video
            return "error_handler"
    
    # Low success rate or no successful renders
    if success_rate < 0.7:
        logger.warning(f"Low rendering success rate ({success_rate:.2%}), routing to error_handler")
        state.add_error(WorkflowError(
            step="rendering",
            error_type="low_success_rate",
            message=f"Only {rendered_scenes}/{total_scenes_to_render} scenes rendered successfully",
            recoverable=True
        ))
        return "error_handler"
    
    # No successful renders
    if rendered_scenes == 0:
        logger.error("No scenes rendered successfully, routing to error_handler")
        state.add_error(WorkflowError(
            step="rendering",
            error_type="complete_failure",
            message="Failed to render any scenes",
            recoverable=True
        ))
        return "error_handler"
    
    # Default to error handler for unexpected cases
    logger.warning("Unexpected rendering state, routing to error_handler")
    return "error_handler"


def route_from_error_handler(state: VideoGenerationState) -> ErrorHandlerRoute:
    """
    Route from error handler based on recovery results and escalation conditions.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next step in the workflow
        
    Routing Logic:
        - If recovery successful -> return to appropriate step
        - If recovery failed but retries available -> retry same step
        - If max retries exceeded -> human_loop
        - If error escalated -> human_loop
        - If workflow should complete despite errors -> complete
    """
    logger.info(f"Routing from error_handler step for session {state.session_id}")
    
    # Add execution trace
    state.add_execution_trace("routing_from_error_handler", {
        "current_step": state.current_step,
        "error_count": len(state.errors),
        "escalated_errors": len(state.escalated_errors),
        "retry_counts": dict(state.retry_counts)
    })
    
    # Check for workflow interruption
    if state.workflow_interrupted:
        logger.info("Workflow interrupted, routing to human_loop")
        return "human_loop"
    
    # Check for escalated errors
    if state.escalated_errors:
        logger.info("Errors have been escalated, routing to human_loop")
        return "human_loop"
    
    # Check for pending human input
    if state.pending_human_input:
        logger.info("Pending human input detected, routing to human_loop")
        return "human_loop"
    
    # Determine which step to return to based on current workflow state
    max_retries = state.config.max_retries
    
    # Check planning state
    if not state.scene_outline or not state.scene_implementations:
        planning_retries = state.retry_counts.get("planning", 0)
        if planning_retries < max_retries:
            logger.info("Returning to planning step for retry")
            return "planning"
        else:
            logger.warning("Planning max retries exceeded, routing to human_loop")
            return "human_loop"
    
    # Check code generation state
    total_scenes = len(state.scene_implementations)
    generated_scenes = len(state.generated_code)
    
    if generated_scenes < total_scenes:
        code_gen_retries = state.retry_counts.get("code_generation", 0)
        if code_gen_retries < max_retries:
            logger.info("Returning to code_generation step for retry")
            return "code_generation"
        else:
            logger.warning("Code generation max retries exceeded, routing to human_loop")
            return "human_loop"
    
    # Check rendering state
    rendered_scenes = len(state.rendered_videos)
    
    if rendered_scenes < generated_scenes or not state.combined_video_path:
        rendering_retries = state.retry_counts.get("rendering", 0)
        if rendering_retries < max_retries:
            logger.info("Returning to rendering step for retry")
            return "rendering"
        else:
            logger.warning("Rendering max retries exceeded, routing to human_loop")
            return "human_loop"
    
    # If we have some successful results, consider completing
    completion_percentage = state.get_completion_percentage()
    
    if completion_percentage >= 70.0:  # 70% completion threshold
        logger.info(f"Workflow {completion_percentage:.1f}% complete, acceptable for completion")
        return "complete"
    
    # If all retries exhausted and low completion, escalate
    all_retries_exhausted = all(
        state.retry_counts.get(step, 0) >= max_retries
        for step in ["planning", "code_generation", "rendering"]
    )
    
    if all_retries_exhausted:
        logger.warning("All retry attempts exhausted, routing to human_loop")
        return "human_loop"
    
    # Default to human loop for safety
    logger.warning("Unable to determine recovery path, routing to human_loop")
    return "human_loop"


def validate_routing_decision(
    from_step: str, 
    to_step: str, 
    state: VideoGenerationState
) -> bool:
    """
    Validate that a routing decision is valid given the current state.
    
    Args:
        from_step: The step we're routing from
        to_step: The step we're routing to
        state: Current workflow state
        
    Returns:
        True if the routing decision is valid, False otherwise
    """
    logger.debug(f"Validating routing from {from_step} to {to_step}")
    
    # Define valid transitions
    valid_transitions = {
        "planning": ["code_generation", "error_handler", "human_loop"],
        "code_generation": ["rendering", "rag_enhancement", "error_handler", "human_loop"],
        "rag_enhancement": ["code_generation", "error_handler", "human_loop"],
        "rendering": ["visual_analysis", "complete", "error_handler", "human_loop"],
        "visual_analysis": ["complete", "error_handler", "human_loop"],
        "error_handler": ["planning", "code_generation", "rendering", "human_loop", "complete"],
        "human_loop": ["planning", "code_generation", "rendering", "complete"],
        "complete": []  # Terminal state
    }
    
    # Check if transition is valid
    if from_step not in valid_transitions:
        logger.error(f"Invalid from_step: {from_step}")
        return False
    
    if to_step not in valid_transitions[from_step]:
        logger.error(f"Invalid transition from {from_step} to {to_step}")
        return False
    
    # Additional state-based validations
    if to_step == "code_generation" and not state.scene_implementations:
        logger.error("Cannot route to code_generation without scene implementations")
        return False
    
    if to_step == "rendering" and not state.generated_code:
        logger.error("Cannot route to rendering without generated code")
        return False
    
    if to_step == "visual_analysis" and not state.rendered_videos:
        logger.error("Cannot route to visual_analysis without rendered videos")
        return False
    
    if to_step == "complete" and state.workflow_interrupted:
        logger.error("Cannot complete interrupted workflow")
        return False
    
    logger.debug(f"Routing from {from_step} to {to_step} is valid")
    return True


def get_routing_summary(state: VideoGenerationState) -> Dict[str, Any]:
    """
    Get a summary of the current routing state for debugging and monitoring.
    
    Args:
        state: Current workflow state
        
    Returns:
        Dictionary containing routing state summary
    """
    return {
        "current_step": state.current_step,
        "workflow_complete": state.workflow_complete,
        "workflow_interrupted": state.workflow_interrupted,
        "pending_human_input": bool(state.pending_human_input),
        "error_count": len(state.errors),
        "escalated_errors": len(state.escalated_errors),
        "retry_counts": dict(state.retry_counts),
        "completion_percentage": state.get_completion_percentage(),
        "scene_counts": {
            "implementations": len(state.scene_implementations),
            "generated_code": len(state.generated_code),
            "rendered_videos": len(state.rendered_videos)
        },
        "has_combined_video": bool(state.combined_video_path),
        "config": {
            "max_retries": state.config.max_retries,
            "use_rag": state.config.use_rag,
            "use_visual_analysis": state.config.use_visual_analysis
        }
    }
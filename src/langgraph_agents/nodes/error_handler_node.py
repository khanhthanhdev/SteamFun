"""
Error handler node function following LangGraph patterns.

This module implements the error_handler_node function that provides centralized
error handling using ErrorHandler class for recovery logic.
"""

import logging
from typing import Dict, Any, List, Optional

from ..models.state import VideoGenerationState
from ..models.errors import WorkflowError, ErrorType, ErrorSeverity, ErrorRecoveryResult
from ..error_recovery.error_handler import ErrorHandler

logger = logging.getLogger(__name__)


async def error_handler_node(state: VideoGenerationState) -> VideoGenerationState:
    """
    Error handler node function that processes and attempts to recover from errors.
    
    This function follows LangGraph best practices by:
    - Taking state as input and returning updated state
    - Using ErrorHandler class for recovery logic separation
    - Implementing escalation to human loop when needed
    - Adding comprehensive logging and tracing
    
    Args:
        state: Current workflow state with errors to handle
        
    Returns:
        VideoGenerationState: Updated state with error recovery results
    """
    logger.info(f"Starting error handler node for session {state.session_id}")
    
    # Update current step
    state.current_step = "error_handling"
    state.add_execution_trace("error_handler_node", {
        "action": "started", 
        "session_id": state.session_id,
        "error_count": len(state.errors),
        "has_escalated_errors": len(state.escalated_errors) > 0
    })
    
    try:
        # Check if there are errors to handle
        if not state.errors and not state.escalated_errors:
            logger.info("No errors to handle, skipping error recovery")
            state.add_execution_trace("error_handler_node", {
                "action": "skipped",
                "reason": "no_errors_to_handle"
            })
            return state
        
        # Initialize error handler
        error_handler = ErrorHandler(state.config, rag_service=None)
        
        # Track recovery results
        recovery_results: List[ErrorRecoveryResult] = []
        errors_to_remove: List[WorkflowError] = []
        errors_to_escalate: List[WorkflowError] = []
        
        # Process each error
        for error in state.errors:
            if error.resolved:
                # Skip already resolved errors
                continue
            
            logger.info(f"Processing error: {error.get_error_code()}")
            
            try:
                # Attempt error recovery
                recovery_result = await error_handler.handle_error(error, state)
                recovery_results.append(recovery_result)
                
                if recovery_result.success and not recovery_result.escalated:
                    # Error was successfully recovered
                    error.mark_resolved(recovery_result.strategy_used)
                    errors_to_remove.append(error)
                    
                    logger.info(f"Successfully recovered from error: {error.get_error_code()} using {recovery_result.strategy_used}")
                    
                elif recovery_result.escalated:
                    # Error needs escalation
                    errors_to_escalate.append(error)
                    
                    # Add to escalated errors list
                    escalation_info = {
                        "error_code": error.get_error_code(),
                        "original_error": error.model_dump(),
                        "recovery_attempts": recovery_result.attempts_made,
                        "escalation_reason": recovery_result.escalation_reason,
                        "escalated_at": recovery_result.timestamp.isoformat(),
                        "requires_human_intervention": True
                    }
                    state.escalated_errors.append(escalation_info)
                    
                    logger.warning(f"Error escalated: {error.get_error_code()} - {recovery_result.escalation_reason}")
                    
                else:
                    # Recovery failed but not escalated yet
                    logger.warning(f"Recovery failed for error: {error.get_error_code()}")
                    
                    # Check if we should escalate based on retry count
                    if error.should_escalate(state.config.max_retries):
                        errors_to_escalate.append(error)
                        
                        escalation_info = {
                            "error_code": error.get_error_code(),
                            "original_error": error.model_dump(),
                            "recovery_attempts": recovery_result.attempts_made,
                            "escalation_reason": f"Max recovery attempts exceeded ({recovery_result.attempts_made})",
                            "escalated_at": recovery_result.timestamp.isoformat(),
                            "requires_human_intervention": True
                        }
                        state.escalated_errors.append(escalation_info)
                        
                        logger.warning(f"Error escalated due to max attempts: {error.get_error_code()}")
                
                # Handle new errors from recovery process
                if recovery_result.new_error:
                    state.add_error(recovery_result.new_error)
                    logger.warning(f"New error generated during recovery: {recovery_result.new_error.get_error_code()}")
                
            except Exception as e:
                logger.error(f"Error recovery process failed for {error.get_error_code()}: {str(e)}")
                
                # Create a new error for the recovery failure
                recovery_error = WorkflowError(
                    step="error_handling",
                    error_type=ErrorType.SYSTEM,
                    message=f"Error recovery failed: {str(e)}",
                    severity=ErrorSeverity.HIGH,
                    context={
                        "original_error": error.get_error_code(),
                        "recovery_exception": str(e)
                    }
                )
                state.add_error(recovery_error)
        
        # Remove successfully recovered errors
        for error in errors_to_remove:
            if error in state.errors:
                state.errors.remove(error)
        
        # Remove escalated errors from main error list
        for error in errors_to_escalate:
            if error in state.errors:
                state.errors.remove(error)
        
        # Check if human intervention is needed
        needs_human_intervention = len(state.escalated_errors) > 0
        
        if needs_human_intervention:
            # Prepare human intervention request
            human_intervention_request = _prepare_human_intervention_request(state, errors_to_escalate)
            state.pending_human_input = human_intervention_request
            
            logger.info(f"Human intervention requested for {len(state.escalated_errors)} escalated errors")
        
        # Update metrics if available
        if state.metrics:
            error_handling_metrics = {
                "errors_processed": len(recovery_results),
                "errors_recovered": len(errors_to_remove),
                "errors_escalated": len(errors_to_escalate),
                "recovery_success_rate": len(errors_to_remove) / len(recovery_results) if recovery_results else 0.0,
                "human_intervention_required": needs_human_intervention,
                "recovery_strategies_used": list(set(r.strategy_used for r in recovery_results if r.success))
            }
            state.metrics.add_step_metrics("error_handling", error_handling_metrics)
        
        # Add completion trace
        state.add_execution_trace("error_handler_node", {
            "action": "completed",
            "errors_processed": len(recovery_results),
            "errors_recovered": len(errors_to_remove),
            "errors_escalated": len(errors_to_escalate),
            "human_intervention_required": needs_human_intervention,
            "remaining_errors": len(state.errors)
        })
        
        logger.info(f"Error handling completed: {len(errors_to_remove)} recovered, {len(errors_to_escalate)} escalated, {len(state.errors)} remaining")
        
        return state
        
    except Exception as e:
        logger.error(f"Error handler node failed: {str(e)}")
        
        # Create workflow error for the error handler failure
        error = WorkflowError(
            step="error_handling",
            error_type=ErrorType.SYSTEM,
            message=f"Error handler failed: {str(e)}",
            severity=ErrorSeverity.CRITICAL,
            context={"exception": str(e), "exception_type": type(e).__name__}
        )
        state.add_error(error)
        
        # Add failure trace
        state.add_execution_trace("error_handler_node", {
            "action": "failed",
            "error": str(e),
            "error_type": type(e).__name__
        })
        
        return state


def _prepare_human_intervention_request(
    state: VideoGenerationState, 
    escalated_errors: List[WorkflowError]
) -> Dict[str, Any]:
    """Prepare a human intervention request with error details and context."""
    
    # Categorize errors by type and severity
    error_summary = {}
    for error in escalated_errors:
        error_type = error.error_type
        if error_type not in error_summary:
            error_summary[error_type] = {
                "count": 0,
                "severity_levels": set(),
                "scene_numbers": set(),
                "messages": []
            }
        
        error_summary[error_type]["count"] += 1
        error_summary[error_type]["severity_levels"].add(error.severity)
        if error.scene_number:
            error_summary[error_type]["scene_numbers"].add(error.scene_number)
        error_summary[error_type]["messages"].append(error.message)
    
    # Convert sets to lists for JSON serialization
    for error_type in error_summary:
        error_summary[error_type]["severity_levels"] = list(error_summary[error_type]["severity_levels"])
        error_summary[error_type]["scene_numbers"] = list(error_summary[error_type]["scene_numbers"])
    
    # Determine intervention type based on error patterns
    intervention_type = _determine_intervention_type(escalated_errors)
    
    # Create suggested actions
    suggested_actions = _generate_suggested_actions(escalated_errors, state)
    
    # Prepare workflow context
    workflow_context = {
        "session_id": state.session_id,
        "topic": state.topic,
        "description": state.description,
        "current_step": state.current_step,
        "total_scenes": len(state.scene_implementations),
        "completed_scenes": {
            "planning": len(state.scene_implementations) > 0,
            "code_generation": len(state.generated_code),
            "rendering": len(state.rendered_videos)
        },
        "workflow_progress": state.get_completion_percentage()
    }
    
    return {
        "request_type": "error_escalation",
        "intervention_type": intervention_type,
        "priority": _calculate_priority(escalated_errors),
        "error_summary": error_summary,
        "escalated_errors": [error.model_dump() for error in escalated_errors],
        "suggested_actions": suggested_actions,
        "workflow_context": workflow_context,
        "requires_immediate_attention": any(error.severity == ErrorSeverity.CRITICAL for error in escalated_errors),
        "can_continue_workflow": len(state.errors) == 0,  # Can continue if no remaining errors
        "timestamp": state.updated_at.isoformat(),
        "estimated_resolution_time": _estimate_resolution_time(escalated_errors)
    }


def _determine_intervention_type(escalated_errors: List[WorkflowError]) -> str:
    """Determine the type of human intervention needed."""
    error_types = [error.error_type for error in escalated_errors]
    severity_levels = [error.severity for error in escalated_errors]
    
    # Critical errors require immediate intervention
    if any(severity == "critical" for severity in severity_levels):
        return "immediate_intervention"
    
    # System errors typically require technical intervention
    if any(error_type == "system" for error_type in error_types):
        return "technical_intervention"
    
    # Configuration errors require configuration review
    if any(error_type == "configuration" for error_type in error_types):
        return "configuration_review"
    
    # Content errors might require content review
    if any(error_type == "content" for error_type in error_types):
        return "content_review"
    
    # Security errors require security review
    if any(error_type == "security" for error_type in error_types):
        return "security_review"
    
    return "general_review"


def _generate_suggested_actions(escalated_errors: List[WorkflowError], state: VideoGenerationState) -> List[str]:
    """Generate suggested actions for human intervention."""
    actions = []
    error_types = [error.error_type for error in escalated_errors]
    
    # Configuration-related suggestions
    if "configuration" in error_types:
        actions.append("Review and update model configuration settings")
        actions.append("Verify API keys and authentication credentials")
    
    # Content-related suggestions
    if "content" in error_types:
        actions.append("Review scene implementations for clarity and feasibility")
        actions.append("Consider simplifying complex scene descriptions")
        actions.append("Check if topic requires specialized knowledge or plugins")
    
    # System-related suggestions
    if "system" in error_types:
        actions.append("Check system resources (memory, disk space, CPU)")
        actions.append("Verify all required dependencies are installed")
        actions.append("Review system logs for additional error details")
    
    # Model-related suggestions
    if "model" in error_types:
        actions.append("Try using a different model or provider")
        actions.append("Reduce complexity of generation requests")
        actions.append("Check model availability and rate limits")
    
    # Rendering-related suggestions
    if "rendering" in error_types:
        actions.append("Review generated code for Manim compatibility")
        actions.append("Check rendering environment and dependencies")
        actions.append("Consider reducing video quality or complexity")
    
    # General suggestions
    actions.append("Review error details and context for specific issues")
    actions.append("Consider restarting the workflow from the last successful step")
    
    # Workflow-specific suggestions based on state
    if len(state.generated_code) == 0:
        actions.append("Focus on resolving planning and code generation issues first")
    elif len(state.rendered_videos) == 0:
        actions.append("Focus on resolving rendering and video generation issues")
    
    return actions


def _calculate_priority(escalated_errors: List[WorkflowError]) -> str:
    """Calculate priority level for human intervention."""
    severity_levels = [error.severity for error in escalated_errors]
    
    if any(severity == "critical" for severity in severity_levels):
        return "critical"
    elif any(severity == "high" for severity in severity_levels):
        return "high"
    elif any(severity == "medium" for severity in severity_levels):
        return "medium"
    else:
        return "low"


def _estimate_resolution_time(escalated_errors: List[WorkflowError]) -> str:
    """Estimate time needed for human resolution."""
    error_types = [error.error_type for error in escalated_errors]
    severity_levels = [error.severity for error in escalated_errors]
    
    # Critical errors need immediate attention
    if any(severity == "critical" for severity in severity_levels):
        return "immediate"
    
    # System errors typically take longer to resolve
    if any(error_type == "system" for error_type in error_types):
        return "30-60 minutes"
    
    # Configuration errors are usually quick to fix
    if any(error_type == "configuration" for error_type in error_types):
        return "5-15 minutes"
    
    # Content errors depend on complexity
    if any(error_type == "content" for error_type in error_types):
        return "15-30 minutes"
    
    return "10-20 minutes"
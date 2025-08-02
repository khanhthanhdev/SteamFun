"""
Human intervention tools and interfaces for LangGraph multi-agent system.
Provides tools for requesting human approval, feedback, and decision-making.
Enhanced with better integration points and validation mechanisms.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Annotated, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from ..state import VideoGenerationState


logger = logging.getLogger(__name__)


class DecisionContext(Enum):
    """Types of decision contexts for human intervention."""
    ERROR_RECOVERY = "error_recovery"
    QUALITY_ASSESSMENT = "quality_assessment"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    WORKFLOW_CONTINUATION = "workflow_continuation"
    FINAL_APPROVAL = "final_approval"


@dataclass
class InterventionContext:
    """Context information for human intervention requests."""
    context_type: DecisionContext
    title: str
    description: str
    current_state_summary: Dict[str, Any]
    error_details: Optional[Dict[str, Any]] = None
    quality_metrics: Optional[Dict[str, Any]] = None
    suggested_actions: Optional[List[str]] = None
    urgency_level: str = "medium"  # low, medium, high, critical


@dataclass
class InterventionOption:
    """Option for human selection during intervention."""
    id: str
    label: str
    description: str
    consequences: str
    recommended: bool = False
    requires_input: bool = False
    input_type: Optional[str] = None  # text, number, selection
    input_options: Optional[List[str]] = None


def request_human_approval(
    context: str,
    options: List[str],
    priority: str = "medium",
    timeout_seconds: int = 300,
    default_action: Optional[str] = None,
    state: Optional[VideoGenerationState] = None
) -> Command:
    """Request human approval for a decision point.
    
    Args:
        context: Description of what needs approval
        options: List of available options for human to choose from
        priority: Priority level (low, medium, high, critical)
        timeout_seconds: Timeout for human response
        default_action: Default action if timeout occurs
        state: Current workflow state
        
    Returns:
        Command: LangGraph command to route to human loop agent
    """
    intervention_id = f"approval_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    logger.info(f"Requesting human approval: {intervention_id}")
    
    return Command(
        goto="human_loop_agent",
        update={
            "pending_human_input": {
                "intervention_id": intervention_id,
                "type": "approval",
                "priority": priority,
                "context": context,
                "options": options,
                "requesting_agent": state.get('current_agent', 'unknown') if state else 'unknown',
                "timestamp": datetime.now().isoformat(),
                "timeout_seconds": timeout_seconds,
                "default_action": default_action,
                "metadata": {
                    "session_id": state.get('session_id', '') if state else '',
                    "topic": state.get('topic', '') if state else ''
                }
            },
            "workflow_interrupted": True,
            "current_agent": "human_loop_agent"
        }
    )


def request_human_decision(
    decision_context: Dict[str, Any],
    options: List[Dict[str, Any]],
    state: Optional[VideoGenerationState] = None
) -> Command:
    """Request human decision with detailed context and structured options.
    
    Args:
        decision_context: Detailed context for the decision
        options: List of structured options with descriptions and consequences
        state: Current workflow state
        
    Returns:
        Command: LangGraph command to route to human loop agent
    """
    intervention_id = f"decision_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    # Validate and structure the decision context
    context_obj = InterventionContext(
        context_type=DecisionContext(decision_context.get('type', 'workflow_continuation')),
        title=decision_context.get('title', 'Decision Required'),
        description=decision_context.get('description', ''),
        current_state_summary=decision_context.get('state_summary', {}),
        error_details=decision_context.get('error_details'),
        quality_metrics=decision_context.get('quality_metrics'),
        suggested_actions=decision_context.get('suggested_actions'),
        urgency_level=decision_context.get('urgency_level', 'medium')
    )
    
    # Structure the options
    structured_options = []
    for i, option in enumerate(options):
        if isinstance(option, str):
            # Simple string option
            structured_options.append(InterventionOption(
                id=f"option_{i}",
                label=option,
                description=option,
                consequences="No additional information provided",
                recommended=i == 0
            ))
        else:
            # Structured option
            structured_options.append(InterventionOption(
                id=option.get('id', f"option_{i}"),
                label=option.get('label', f"Option {i+1}"),
                description=option.get('description', ''),
                consequences=option.get('consequences', ''),
                recommended=option.get('recommended', False),
                requires_input=option.get('requires_input', False),
                input_type=option.get('input_type'),
                input_options=option.get('input_options')
            ))
    
    logger.info(f"Requesting human decision: {intervention_id}")
    
    return Command(
        goto="human_loop_agent",
        update={
            "pending_human_input": {
                "intervention_id": intervention_id,
                "type": "decision",
                "priority": context_obj.urgency_level,
                "context": context_obj.description,
                "structured_context": asdict(context_obj),
                "options": [opt.label for opt in structured_options],
                "structured_options": [asdict(opt) for opt in structured_options],
                "requesting_agent": state.get('current_agent', 'unknown') if state else 'unknown',
                "timestamp": datetime.now().isoformat(),
                "timeout_seconds": decision_context.get('timeout_seconds', 600),
                "default_action": decision_context.get('default_action'),
                "metadata": {
                    "session_id": state.get('session_id', '') if state else '',
                    "topic": state.get('topic', '') if state else '',
                    "context_type": context_obj.context_type.value
                }
            },
            "workflow_interrupted": True,
            "current_agent": "human_loop_agent"
        }
    )


@tool
def request_human_feedback(
    feedback_request: str,
    feedback_type: str = "general",
    required_fields: Optional[List[str]] = None,
    state: Annotated[VideoGenerationState, InjectedState] = None
) -> Command:
    """Request human feedback on workflow progress or results.
    
    Args:
        feedback_request: Description of what feedback is needed
        feedback_type: Type of feedback (general, quality, error, suggestion)
        required_fields: List of required feedback fields
        state: Current workflow state
        
    Returns:
        Command: LangGraph command to route to human loop agent
    """
    intervention_id = f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    # Create feedback collection structure
    feedback_structure = {
        "request": feedback_request,
        "type": feedback_type,
        "required_fields": required_fields or [],
        "optional_fields": ["additional_comments", "suggestions", "rating"],
        "rating_scale": "1-10 (1=Poor, 10=Excellent)" if feedback_type == "quality" else None
    }
    
    logger.info(f"Requesting human feedback: {intervention_id}")
    
    return Command(
        goto="human_loop_agent",
        update={
            "pending_human_input": {
                "intervention_id": intervention_id,
                "type": "feedback",
                "priority": "low",
                "context": feedback_request,
                "feedback_structure": feedback_structure,
                "options": ["Provide feedback", "Skip feedback", "Request more information"],
                "requesting_agent": state.get('current_agent', 'unknown') if state else 'unknown',
                "timestamp": datetime.now().isoformat(),
                "timeout_seconds": 900,  # 15 minutes for feedback
                "default_action": "Skip feedback",
                "metadata": {
                    "session_id": state.get('session_id', '') if state else '',
                    "topic": state.get('topic', '') if state else '',
                    "feedback_type": feedback_type
                }
            },
            "workflow_interrupted": True,
            "current_agent": "human_loop_agent"
        }
    )


@tool
def present_decision_context(
    context_data: Dict[str, Any],
    state: Annotated[VideoGenerationState, InjectedState] = None
) -> Dict[str, Any]:
    """Present structured decision context to help human make informed decisions.
    
    Args:
        context_data: Context information to present
        state: Current workflow state
        
    Returns:
        Dict: Formatted context presentation
    """
    current_state = state or {}
    
    # Build comprehensive context presentation
    context_presentation = {
        "session_info": {
            "session_id": current_state.get('session_id', 'unknown'),
            "topic": current_state.get('topic', 'unknown'),
            "timestamp": datetime.now().isoformat()
        },
        "workflow_progress": {
            "current_agent": current_state.get('current_agent', 'unknown'),
            "completed_steps": _get_completed_steps(current_state),
            "pending_steps": _get_pending_steps(current_state),
            "error_count": current_state.get('error_count', 0)
        },
        "context_details": context_data,
        "state_summary": _create_state_summary(current_state),
        "recommendations": _generate_recommendations(context_data, current_state)
    }
    
    logger.info(f"Presenting decision context for session: {current_state.get('session_id', 'unknown')}")
    
    return context_presentation


@tool
def validate_human_selection(
    selection: str,
    available_options: List[str],
    validation_rules: Optional[Dict[str, Any]] = None,
    state: Annotated[VideoGenerationState, InjectedState] = None
) -> Dict[str, Any]:
    """Validate human selection against available options and rules.
    
    Args:
        selection: Human's selected option
        available_options: List of valid options
        validation_rules: Optional validation rules
        state: Current workflow state
        
    Returns:
        Dict: Validation result with status and details
    """
    validation_result = {
        "valid": False,
        "selection": selection,
        "available_options": available_options,
        "validation_errors": [],
        "suggestions": [],
        "timestamp": datetime.now().isoformat()
    }
    
    # Basic validation: check if selection is in available options
    if selection not in available_options:
        # Try fuzzy matching
        fuzzy_matches = [opt for opt in available_options if selection.lower() in opt.lower()]
        if fuzzy_matches:
            validation_result["suggestions"] = fuzzy_matches
            validation_result["validation_errors"].append(
                f"Selection '{selection}' not found. Did you mean one of: {', '.join(fuzzy_matches)}?"
            )
        else:
            validation_result["validation_errors"].append(
                f"Selection '{selection}' is not valid. Available options: {', '.join(available_options)}"
            )
    else:
        validation_result["valid"] = True
    
    # Apply custom validation rules if provided
    if validation_rules and validation_result["valid"]:
        for rule_name, rule_config in validation_rules.items():
            if not _apply_validation_rule(selection, rule_name, rule_config, state):
                validation_result["valid"] = False
                validation_result["validation_errors"].append(
                    f"Selection failed validation rule '{rule_name}': {rule_config.get('message', 'Rule failed')}"
                )
    
    logger.info(f"Validated human selection: {selection} - Valid: {validation_result['valid']}")
    
    return validation_result


@tool
def create_intervention_summary(
    intervention_data: Dict[str, Any],
    state: Annotated[VideoGenerationState, InjectedState] = None
) -> Dict[str, Any]:
    """Create a summary of human intervention for logging and analysis.
    
    Args:
        intervention_data: Data about the intervention
        state: Current workflow state
        
    Returns:
        Dict: Intervention summary
    """
    current_state = state or {}
    
    summary = {
        "intervention_id": intervention_data.get('intervention_id', 'unknown'),
        "type": intervention_data.get('type', 'unknown'),
        "timestamp": datetime.now().isoformat(),
        "session_info": {
            "session_id": current_state.get('session_id', 'unknown'),
            "topic": current_state.get('topic', 'unknown')
        },
        "context": intervention_data.get('context', ''),
        "options_presented": intervention_data.get('options', []),
        "human_response": {
            "selected_option": intervention_data.get('selected_option'),
            "additional_feedback": intervention_data.get('additional_feedback'),
            "confidence_level": intervention_data.get('confidence_level'),
            "response_time_seconds": intervention_data.get('response_time_seconds')
        },
        "outcome": {
            "next_action": intervention_data.get('next_action'),
            "workflow_resumed": intervention_data.get('workflow_resumed', False),
            "modifications_made": intervention_data.get('modifications_made', [])
        },
        "impact_analysis": _analyze_intervention_impact(intervention_data, current_state)
    }
    
    logger.info(f"Created intervention summary: {summary['intervention_id']}")
    
    return summary


def _get_completed_steps(state: VideoGenerationState) -> List[str]:
    """Get list of completed workflow steps."""
    completed = []
    
    if state.get('scene_outline'):
        completed.append('planning')
    if state.get('generated_code'):
        completed.append('code_generation')
    if state.get('rendered_videos'):
        completed.append('rendering')
    if state.get('visual_analysis_results'):
        completed.append('visual_analysis')
    if state.get('combined_video_path'):
        completed.append('video_combination')
    
    return completed


def _get_pending_steps(state: VideoGenerationState) -> List[str]:
    """Get list of pending workflow steps."""
    all_steps = ['planning', 'code_generation', 'rendering', 'visual_analysis', 'video_combination']
    completed = _get_completed_steps(state)
    return [step for step in all_steps if step not in completed]


def _create_state_summary(state: VideoGenerationState) -> Dict[str, Any]:
    """Create a summary of current workflow state."""
    return {
        "topic": state.get('topic', 'unknown'),
        "progress": {
            "scenes_planned": len(state.get('scene_implementations', {})),
            "scenes_coded": len(state.get('generated_code', {})),
            "scenes_rendered": len(state.get('rendered_videos', {})),
            "scenes_analyzed": len(state.get('visual_analysis_results', {}))
        },
        "errors": {
            "total_errors": state.get('error_count', 0),
            "code_errors": len(state.get('code_errors', {})),
            "rendering_errors": len(state.get('rendering_errors', {})),
            "visual_errors": sum(len(errors) for errors in state.get('visual_errors', {}).values())
        },
        "configuration": {
            "use_rag": state.get('use_rag', False),
            "use_visual_fix": state.get('use_visual_fix_code', False),
            "max_retries": state.get('max_retries', 0)
        }
    }


def _generate_recommendations(context_data: Dict[str, Any], state: VideoGenerationState) -> List[str]:
    """Generate recommendations based on context and state."""
    recommendations = []
    
    error_count = state.get('error_count', 0)
    if error_count > 3:
        recommendations.append("Consider adjusting parameters or restarting workflow due to high error count")
    
    if len(state.get('rendering_errors', {})) > 1:
        recommendations.append("Multiple rendering failures detected - consider lowering quality settings")
    
    if not state.get('use_rag', False) and error_count > 0:
        recommendations.append("Consider enabling RAG to improve code generation accuracy")
    
    visual_errors = sum(len(errors) for errors in state.get('visual_errors', {}).values())
    if visual_errors > 2:
        recommendations.append("Visual quality issues detected - manual review recommended")
    
    return recommendations


def _apply_validation_rule(selection: str, rule_name: str, rule_config: Dict[str, Any], state: VideoGenerationState) -> bool:
    """Apply a custom validation rule to a selection."""
    if rule_name == "error_threshold":
        max_errors = rule_config.get('max_errors', 5)
        current_errors = state.get('error_count', 0)
        if selection.lower() == 'continue' and current_errors >= max_errors:
            return False
    
    elif rule_name == "quality_threshold":
        min_quality = rule_config.get('min_quality', 0.7)
        # This would check against actual quality metrics in a real implementation
        return True
    
    elif rule_name == "resource_availability":
        # Check if system resources are available for the selected action
        return True
    
    return True


def _analyze_intervention_impact(intervention_data: Dict[str, Any], state: VideoGenerationState) -> Dict[str, Any]:
    """Analyze the impact of human intervention on workflow."""
    return {
        "workflow_efficiency": {
            "time_saved": "estimated_time_saved_by_intervention",
            "errors_prevented": "estimated_errors_prevented",
            "quality_improvement": "estimated_quality_improvement"
        },
        "decision_quality": {
            "confidence_level": intervention_data.get('confidence_level', 0),
            "alignment_with_recommendations": "high",  # Would be calculated
            "risk_level": "low"  # Would be assessed
        },
        "learning_opportunities": {
            "pattern_recognition": "intervention_patterns_identified",
            "automation_potential": "areas_for_future_automation",
            "user_preferences": "learned_user_preferences"
        }
    }


# Enhanced integration tools for agents

@tool
def request_agent_approval(
    requesting_agent: str,
    approval_context: str,
    approval_options: List[str],
    agent_specific_data: Dict[str, Any],
    priority: str = "medium",
    timeout_seconds: int = 300,
    state: Annotated[VideoGenerationState, InjectedState] = None
) -> Command:
    """Request human approval with agent-specific context and data.
    
    Args:
        requesting_agent: Name of the agent requesting approval
        approval_context: Context for what needs approval
        approval_options: Available approval options
        agent_specific_data: Agent-specific data to include in context
        priority: Priority level (low, medium, high, critical)
        timeout_seconds: Timeout for human response
        state: Current workflow state
        
    Returns:
        Command: LangGraph command to route to human loop agent
    """
    intervention_id = f"agent_approval_{requesting_agent}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    # Create enhanced context with agent-specific information
    enhanced_context = {
        "type": "agent_approval",
        "requesting_agent": requesting_agent,
        "approval_context": approval_context,
        "agent_data": agent_specific_data,
        "workflow_context": _create_agent_workflow_context(requesting_agent, state),
        "recommendations": _generate_agent_recommendations(requesting_agent, agent_specific_data, state)
    }
    
    logger.info(f"Agent {requesting_agent} requesting approval: {intervention_id}")
    
    return Command(
        goto="human_loop_agent",
        update={
            "pending_human_input": {
                "intervention_id": intervention_id,
                "type": "agent_approval",
                "priority": priority,
                "context": approval_context,
                "enhanced_context": enhanced_context,
                "options": approval_options,
                "requesting_agent": requesting_agent,
                "timestamp": datetime.now().isoformat(),
                "timeout_seconds": timeout_seconds,
                "default_action": approval_options[0] if approval_options else "approve",
                "metadata": {
                    "session_id": state.get('session_id', '') if state else '',
                    "topic": state.get('topic', '') if state else '',
                    "agent_specific": True
                }
            },
            "workflow_interrupted": True,
            "current_agent": "human_loop_agent"
        }
    )


@tool
def request_quality_gate_approval(
    quality_metrics: Dict[str, Any],
    quality_thresholds: Dict[str, float],
    content_description: str,
    improvement_options: List[str],
    requesting_agent: str,
    state: Annotated[VideoGenerationState, InjectedState] = None
) -> Command:
    """Request approval for quality gate with detailed metrics and options.
    
    Args:
        quality_metrics: Current quality metrics
        quality_thresholds: Expected quality thresholds
        content_description: Description of content being evaluated
        improvement_options: Available improvement options
        requesting_agent: Name of the requesting agent
        state: Current workflow state
        
    Returns:
        Command: LangGraph command for quality gate approval
    """
    intervention_id = f"quality_gate_{requesting_agent}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    # Analyze quality metrics against thresholds
    quality_analysis = _analyze_quality_metrics(quality_metrics, quality_thresholds)
    
    # Create structured quality context
    quality_context = {
        "type": "quality_gate",
        "content_description": content_description,
        "quality_metrics": quality_metrics,
        "quality_thresholds": quality_thresholds,
        "quality_analysis": quality_analysis,
        "improvement_options": improvement_options,
        "requesting_agent": requesting_agent,
        "recommendations": _generate_quality_recommendations(quality_analysis)
    }
    
    # Determine priority based on quality analysis
    priority = "high" if quality_analysis.get('below_threshold_count', 0) > 2 else "medium"
    
    logger.info(f"Quality gate approval requested by {requesting_agent}: {intervention_id}")
    
    return Command(
        goto="human_loop_agent",
        update={
            "pending_human_input": {
                "intervention_id": intervention_id,
                "type": "quality_gate",
                "priority": priority,
                "context": f"Quality review for {content_description}",
                "quality_context": quality_context,
                "options": ["Approve current quality", "Request improvements", "Adjust thresholds", "Manual review"],
                "requesting_agent": requesting_agent,
                "timestamp": datetime.now().isoformat(),
                "timeout_seconds": 600,  # Longer timeout for quality reviews
                "default_action": "Approve current quality" if quality_analysis.get('overall_acceptable', False) else "Request improvements",
                "metadata": {
                    "session_id": state.get('session_id', '') if state else '',
                    "topic": state.get('topic', '') if state else '',
                    "quality_gate": True
                }
            },
            "workflow_interrupted": True,
            "current_agent": "human_loop_agent"
        }
    )


@tool
def request_error_resolution_decision(
    error_details: Dict[str, Any],
    recovery_strategies: List[Dict[str, Any]],
    error_context: Dict[str, Any],
    requesting_agent: str,
    state: Annotated[VideoGenerationState, InjectedState] = None
) -> Command:
    """Request human decision for error resolution with detailed recovery strategies.
    
    Args:
        error_details: Details about the error that occurred
        recovery_strategies: Available recovery strategies with details
        error_context: Context about when and how the error occurred
        requesting_agent: Name of the requesting agent
        state: Current workflow state
        
    Returns:
        Command: LangGraph command for error resolution decision
    """
    intervention_id = f"error_resolution_{requesting_agent}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    # Analyze error severity and impact
    error_analysis = _analyze_error_severity(error_details, error_context, state)
    
    # Create comprehensive error resolution context
    resolution_context = {
        "type": "error_resolution",
        "error_details": error_details,
        "error_context": error_context,
        "error_analysis": error_analysis,
        "recovery_strategies": recovery_strategies,
        "requesting_agent": requesting_agent,
        "impact_assessment": _assess_error_impact(error_details, state),
        "recommendations": _generate_error_recovery_recommendations(error_analysis, recovery_strategies)
    }
    
    # Extract strategy labels for options
    strategy_options = [strategy.get('label', strategy.get('name', f"Strategy {i+1}")) 
                       for i, strategy in enumerate(recovery_strategies)]
    strategy_options.extend(["Escalate to human expert", "Abort workflow", "Continue with error"])
    
    # Determine priority based on error severity
    priority = "critical" if error_analysis.get('severity', 'medium') == 'critical' else "high"
    
    logger.info(f"Error resolution decision requested by {requesting_agent}: {intervention_id}")
    
    return Command(
        goto="human_loop_agent",
        update={
            "pending_human_input": {
                "intervention_id": intervention_id,
                "type": "error_resolution",
                "priority": priority,
                "context": f"Error resolution needed: {error_details.get('error_message', 'Unknown error')}",
                "resolution_context": resolution_context,
                "options": strategy_options,
                "requesting_agent": requesting_agent,
                "timestamp": datetime.now().isoformat(),
                "timeout_seconds": 900,  # Longer timeout for error resolution
                "default_action": strategy_options[0] if strategy_options else "Continue with error",
                "metadata": {
                    "session_id": state.get('session_id', '') if state else '',
                    "topic": state.get('topic', '') if state else '',
                    "error_resolution": True
                }
            },
            "workflow_interrupted": True,
            "current_agent": "human_loop_agent"
        }
    )


@tool
def request_parameter_modification(
    parameter_name: str,
    current_value: Any,
    suggested_values: List[Any],
    modification_reason: str,
    impact_analysis: Dict[str, Any],
    requesting_agent: str,
    state: Annotated[VideoGenerationState, InjectedState] = None
) -> Command:
    """Request human decision for parameter modification with impact analysis.
    
    Args:
        parameter_name: Name of parameter to modify
        current_value: Current parameter value
        suggested_values: List of suggested new values
        modification_reason: Reason for requesting modification
        impact_analysis: Analysis of modification impact
        requesting_agent: Name of the requesting agent
        state: Current workflow state
        
    Returns:
        Command: LangGraph command for parameter modification decision
    """
    intervention_id = f"param_mod_{requesting_agent}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    # Create parameter modification context
    modification_context = {
        "type": "parameter_modification",
        "parameter_name": parameter_name,
        "current_value": current_value,
        "suggested_values": suggested_values,
        "modification_reason": modification_reason,
        "impact_analysis": impact_analysis,
        "requesting_agent": requesting_agent,
        "current_configuration": _get_current_configuration_context(state),
        "recommendations": _generate_parameter_recommendations(parameter_name, current_value, suggested_values, impact_analysis)
    }
    
    # Create options for each suggested value
    modification_options = [f"Set {parameter_name} to {value}" for value in suggested_values]
    modification_options.extend([f"Keep current value ({current_value})", "Request custom value", "Skip modification"])
    
    logger.info(f"Parameter modification requested by {requesting_agent}: {parameter_name}")
    
    return Command(
        goto="human_loop_agent",
        update={
            "pending_human_input": {
                "intervention_id": intervention_id,
                "type": "parameter_modification",
                "priority": "medium",
                "context": f"Modify {parameter_name}: {modification_reason}",
                "modification_context": modification_context,
                "options": modification_options,
                "requesting_agent": requesting_agent,
                "timestamp": datetime.now().isoformat(),
                "timeout_seconds": 300,
                "default_action": f"Keep current value ({current_value})",
                "metadata": {
                    "session_id": state.get('session_id', '') if state else '',
                    "topic": state.get('topic', '') if state else '',
                    "parameter_modification": True
                }
            },
            "workflow_interrupted": True,
            "current_agent": "human_loop_agent"
        }
    )


@tool
def create_interactive_decision_interface(
    decision_title: str,
    decision_description: str,
    decision_options: List[Dict[str, Any]],
    context_data: Dict[str, Any],
    validation_rules: Optional[Dict[str, Any]] = None,
    requesting_agent: str = "unknown",
    state: Annotated[VideoGenerationState, InjectedState] = None
) -> Dict[str, Any]:
    """Create an interactive decision interface with rich context and validation.
    
    Args:
        decision_title: Title of the decision
        decision_description: Description of what needs to be decided
        decision_options: List of structured decision options
        context_data: Additional context data for the decision
        validation_rules: Optional validation rules for selections
        requesting_agent: Name of the requesting agent
        state: Current workflow state
        
    Returns:
        Dict: Interactive decision interface data
    """
    interface_id = f"interactive_{requesting_agent}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    # Enhance decision options with additional metadata
    enhanced_options = []
    for i, option in enumerate(decision_options):
        enhanced_option = {
            "id": option.get('id', f"option_{i}"),
            "label": option.get('label', f"Option {i+1}"),
            "description": option.get('description', ''),
            "consequences": option.get('consequences', ''),
            "recommended": option.get('recommended', False),
            "risk_level": option.get('risk_level', 'medium'),
            "estimated_time": option.get('estimated_time', 'unknown'),
            "success_probability": option.get('success_probability', 0.5),
            "requires_additional_input": option.get('requires_input', False),
            "input_validation": option.get('input_validation', {})
        }
        enhanced_options.append(enhanced_option)
    
    # Create comprehensive decision interface
    decision_interface = {
        "interface_id": interface_id,
        "decision_title": decision_title,
        "decision_description": decision_description,
        "context_data": context_data,
        "enhanced_options": enhanced_options,
        "validation_rules": validation_rules or {},
        "requesting_agent": requesting_agent,
        "created_timestamp": datetime.now().isoformat(),
        "workflow_context": {
            "session_id": state.get('session_id', '') if state else '',
            "topic": state.get('topic', '') if state else '',
            "current_progress": _calculate_workflow_progress(state),
            "error_history": _get_recent_error_history(state)
        },
        "decision_support": {
            "recommendations": _generate_decision_recommendations(enhanced_options, context_data),
            "risk_analysis": _analyze_decision_risks(enhanced_options),
            "impact_prediction": _predict_decision_impact(enhanced_options, state)
        }
    }
    
    logger.info(f"Created interactive decision interface: {interface_id}")
    
    return decision_interface


# Helper functions for enhanced tools

def _create_agent_workflow_context(agent_name: str, state: VideoGenerationState) -> Dict[str, Any]:
    """Create workflow context specific to an agent."""
    return {
        "agent_name": agent_name,
        "agent_progress": {
            "planner_complete": bool(state.get('scene_outline')) if state else False,
            "coding_complete": bool(state.get('generated_code')) if state else False,
            "rendering_complete": bool(state.get('rendered_videos')) if state else False,
            "analysis_complete": bool(state.get('visual_analysis_results')) if state else False
        },
        "agent_errors": {
            "total_errors": state.get('error_count', 0) if state else 0,
            "agent_retry_count": state.get('retry_count', {}).get(agent_name, 0) if state else 0
        },
        "agent_performance": state.get('performance_metrics', {}).get(agent_name, {}) if state else {}
    }


def _generate_agent_recommendations(agent_name: str, agent_data: Dict[str, Any], state: VideoGenerationState) -> List[str]:
    """Generate recommendations specific to an agent's context."""
    recommendations = []
    
    if not state:
        return recommendations
    
    error_count = state.get('error_count', 0)
    retry_count = state.get('retry_count', {}).get(agent_name, 0)
    
    if error_count > 2:
        recommendations.append(f"Consider reviewing {agent_name} configuration due to high error count")
    
    if retry_count > 1:
        recommendations.append(f"Agent {agent_name} has retried {retry_count} times - manual intervention may be needed")
    
    # Agent-specific recommendations
    if agent_name == "code_generator_agent":
        if not state.get('use_rag', False):
            recommendations.append("Consider enabling RAG to improve code generation accuracy")
        if len(state.get('code_errors', {})) > 1:
            recommendations.append("Multiple code errors detected - consider adjusting generation parameters")
    
    elif agent_name == "renderer_agent":
        if len(state.get('rendering_errors', {})) > 0:
            recommendations.append("Rendering errors detected - consider lowering quality settings")
    
    elif agent_name == "visual_analysis_agent":
        visual_errors = sum(len(errors) for errors in state.get('visual_errors', {}).values())
        if visual_errors > 2:
            recommendations.append("Multiple visual errors detected - manual review recommended")
    
    return recommendations


def _analyze_quality_metrics(metrics: Dict[str, Any], thresholds: Dict[str, float]) -> Dict[str, Any]:
    """Analyze quality metrics against thresholds."""
    analysis = {
        "metrics_evaluated": len(metrics),
        "thresholds_met": 0,
        "below_threshold_count": 0,
        "metric_details": {},
        "overall_acceptable": True
    }
    
    for metric_name, metric_value in metrics.items():
        threshold = thresholds.get(metric_name)
        if threshold is not None:
            meets_threshold = metric_value >= threshold
            analysis["metric_details"][metric_name] = {
                "value": metric_value,
                "threshold": threshold,
                "meets_threshold": meets_threshold,
                "difference": metric_value - threshold
            }
            
            if meets_threshold:
                analysis["thresholds_met"] += 1
            else:
                analysis["below_threshold_count"] += 1
                analysis["overall_acceptable"] = False
    
    return analysis


def _generate_quality_recommendations(quality_analysis: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on quality analysis."""
    recommendations = []
    
    if quality_analysis.get("overall_acceptable", False):
        recommendations.append("Quality metrics meet acceptable thresholds")
    else:
        below_threshold = quality_analysis.get("below_threshold_count", 0)
        recommendations.append(f"{below_threshold} metrics below threshold - improvements recommended")
    
    # Specific metric recommendations
    for metric_name, details in quality_analysis.get("metric_details", {}).items():
        if not details.get("meets_threshold", True):
            difference = abs(details.get("difference", 0))
            recommendations.append(f"Improve {metric_name} by {difference:.2f} to meet threshold")
    
    return recommendations


def _analyze_error_severity(error_details: Dict[str, Any], error_context: Dict[str, Any], state: VideoGenerationState) -> Dict[str, Any]:
    """Analyze error severity and determine appropriate response."""
    severity_factors = {
        "error_frequency": 0,
        "error_type_severity": 0,
        "workflow_impact": 0,
        "recovery_difficulty": 0
    }
    
    # Analyze error frequency
    total_errors = state.get('error_count', 0) if state else 0
    if total_errors > 5:
        severity_factors["error_frequency"] = 3  # High
    elif total_errors > 2:
        severity_factors["error_frequency"] = 2  # Medium
    else:
        severity_factors["error_frequency"] = 1  # Low
    
    # Analyze error type
    error_type = error_details.get('error_type', '').lower()
    if any(critical_type in error_type for critical_type in ['timeout', 'memory', 'system', 'critical']):
        severity_factors["error_type_severity"] = 3
    elif any(medium_type in error_type for medium_type in ['validation', 'format', 'parsing']):
        severity_factors["error_type_severity"] = 2
    else:
        severity_factors["error_type_severity"] = 1
    
    # Calculate overall severity
    avg_severity = sum(severity_factors.values()) / len(severity_factors)
    if avg_severity >= 2.5:
        severity = "critical"
    elif avg_severity >= 1.5:
        severity = "high"
    else:
        severity = "medium"
    
    return {
        "severity": severity,
        "severity_factors": severity_factors,
        "avg_severity_score": avg_severity,
        "requires_immediate_attention": severity in ["critical", "high"]
    }


def _assess_error_impact(error_details: Dict[str, Any], state: VideoGenerationState) -> Dict[str, Any]:
    """Assess the impact of an error on the workflow."""
    return {
        "workflow_blocked": error_details.get('blocks_workflow', True),
        "data_loss_risk": error_details.get('data_loss_risk', 'low'),
        "recovery_time_estimate": error_details.get('recovery_time', 'unknown'),
        "affected_components": error_details.get('affected_components', []),
        "user_impact": error_details.get('user_impact', 'medium')
    }


def _generate_error_recovery_recommendations(error_analysis: Dict[str, Any], recovery_strategies: List[Dict[str, Any]]) -> List[str]:
    """Generate recommendations for error recovery."""
    recommendations = []
    
    severity = error_analysis.get('severity', 'medium')
    
    if severity == 'critical':
        recommendations.append("Immediate attention required - consider manual intervention")
        recommendations.append("Review system logs and error context before proceeding")
    
    # Recommend strategies based on success probability
    for strategy in recovery_strategies:
        success_prob = strategy.get('success_probability', 0.5)
        if success_prob > 0.8:
            recommendations.append(f"High success probability: {strategy.get('label', 'Unknown strategy')}")
    
    return recommendations


def _get_current_configuration_context(state: VideoGenerationState) -> Dict[str, Any]:
    """Get current configuration context for parameter decisions."""
    if not state:
        return {}
    
    return {
        "use_rag": state.get('use_rag', False),
        "use_visual_fix": state.get('use_visual_fix_code', False),
        "max_retries": state.get('max_retries', 0),
        "default_quality": state.get('default_quality', 'medium'),
        "max_scene_concurrency": state.get('max_scene_concurrency', 1),
        "current_model_config": state.get('model_config', {})
    }


def _generate_parameter_recommendations(parameter_name: str, current_value: Any, suggested_values: List[Any], impact_analysis: Dict[str, Any]) -> List[str]:
    """Generate recommendations for parameter modification."""
    recommendations = []
    
    # General recommendations based on parameter type
    if parameter_name == "max_retries":
        if current_value < 3:
            recommendations.append("Consider increasing max_retries for better error recovery")
    elif parameter_name == "default_quality":
        if current_value == "high" and impact_analysis.get('performance_impact', 'low') == 'high':
            recommendations.append("Consider lowering quality for better performance")
    elif parameter_name == "max_scene_concurrency":
        if current_value == 1:
            recommendations.append("Consider increasing concurrency for faster processing")
    
    # Impact-based recommendations
    performance_impact = impact_analysis.get('performance_impact', 'unknown')
    if performance_impact == 'high':
        recommendations.append("High performance impact expected - monitor system resources")
    
    return recommendations


def _calculate_workflow_progress(state: VideoGenerationState) -> Dict[str, Any]:
    """Calculate current workflow progress."""
    if not state:
        return {"progress_percentage": 0, "completed_stages": []}
    
    stages = {
        "planning": bool(state.get('scene_outline')),
        "code_generation": bool(state.get('generated_code')),
        "rendering": bool(state.get('rendered_videos')),
        "visual_analysis": bool(state.get('visual_analysis_results')),
        "completion": bool(state.get('combined_video_path'))
    }
    
    completed_count = sum(stages.values())
    total_stages = len(stages)
    progress_percentage = (completed_count / total_stages) * 100
    
    return {
        "progress_percentage": progress_percentage,
        "completed_stages": [stage for stage, completed in stages.items() if completed],
        "current_stage": _get_current_stage(stages),
        "stages_remaining": total_stages - completed_count
    }


def _get_current_stage(stages: Dict[str, bool]) -> str:
    """Get the current workflow stage."""
    stage_order = ["planning", "code_generation", "rendering", "visual_analysis", "completion"]
    
    for stage in stage_order:
        if not stages.get(stage, False):
            return stage
    
    return "completed"


def _get_recent_error_history(state: VideoGenerationState) -> List[Dict[str, Any]]:
    """Get recent error history for context."""
    if not state:
        return []
    
    escalated_errors = state.get('escalated_errors', [])
    
    # Return last 5 errors with timestamps
    recent_errors = []
    for error in escalated_errors[-5:]:
        recent_errors.append({
            "agent": error.get('agent', 'unknown'),
            "error": error.get('error', 'unknown'),
            "timestamp": error.get('timestamp', 'unknown'),
            "retry_count": error.get('retry_count', 0)
        })
    
    return recent_errors


def _generate_decision_recommendations(options: List[Dict[str, Any]], context_data: Dict[str, Any]) -> List[str]:
    """Generate recommendations for decision making."""
    recommendations = []
    
    # Find recommended options
    recommended_options = [opt for opt in options if opt.get('recommended', False)]
    if recommended_options:
        for opt in recommended_options:
            recommendations.append(f"Recommended: {opt.get('label', 'Unknown option')}")
    
    # Risk-based recommendations
    low_risk_options = [opt for opt in options if opt.get('risk_level', 'medium') == 'low']
    if low_risk_options:
        recommendations.append("Consider low-risk options for safer progression")
    
    return recommendations


def _analyze_decision_risks(options: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze risks associated with decision options."""
    risk_analysis = {
        "high_risk_count": 0,
        "medium_risk_count": 0,
        "low_risk_count": 0,
        "overall_risk_level": "medium",
        "risk_details": {}
    }
    
    for option in options:
        risk_level = option.get('risk_level', 'medium')
        option_id = option.get('id', 'unknown')
        
        risk_analysis["risk_details"][option_id] = {
            "risk_level": risk_level,
            "success_probability": option.get('success_probability', 0.5),
            "estimated_time": option.get('estimated_time', 'unknown')
        }
        
        if risk_level == 'high':
            risk_analysis["high_risk_count"] += 1
        elif risk_level == 'medium':
            risk_analysis["medium_risk_count"] += 1
        else:
            risk_analysis["low_risk_count"] += 1
    
    # Determine overall risk level
    if risk_analysis["high_risk_count"] > 0:
        risk_analysis["overall_risk_level"] = "high"
    elif risk_analysis["low_risk_count"] > risk_analysis["medium_risk_count"]:
        risk_analysis["overall_risk_level"] = "low"
    
    return risk_analysis


def _predict_decision_impact(options: List[Dict[str, Any]], state: VideoGenerationState) -> Dict[str, Any]:
    """Predict the impact of different decision options."""
    impact_prediction = {
        "workflow_continuation": {},
        "resource_usage": {},
        "time_estimates": {},
        "success_likelihood": {}
    }
    
    for option in options:
        option_id = option.get('id', 'unknown')
        
        impact_prediction["workflow_continuation"][option_id] = option.get('consequences', 'Unknown impact')
        impact_prediction["resource_usage"][option_id] = option.get('resource_impact', 'medium')
        impact_prediction["time_estimates"][option_id] = option.get('estimated_time', 'unknown')
        impact_prediction["success_likelihood"][option_id] = option.get('success_probability', 0.5)
    
    return impact_prediction


# Wrapper functions for testing (without @tool decorator)
def request_human_feedback(
    feedback_request: str,
    feedback_type: str = "general",
    required_fields: Optional[List[str]] = None,
    state: Optional[VideoGenerationState] = None
) -> Command:
    """Request human feedback on workflow progress or results."""
    intervention_id = f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    feedback_structure = {
        "request": feedback_request,
        "type": feedback_type,
        "required_fields": required_fields or [],
        "optional_fields": ["additional_comments", "suggestions", "rating"],
        "rating_scale": "1-10 (1=Poor, 10=Excellent)" if feedback_type == "quality" else None
    }
    
    logger.info(f"Requesting human feedback: {intervention_id}")
    
    return Command(
        goto="human_loop_agent",
        update={
            "pending_human_input": {
                "intervention_id": intervention_id,
                "type": "feedback",
                "priority": "low",
                "context": feedback_request,
                "feedback_structure": feedback_structure,
                "options": ["Provide feedback", "Skip feedback", "Request more information"],
                "requesting_agent": state.get('current_agent', 'unknown') if state else 'unknown',
                "timestamp": datetime.now().isoformat(),
                "timeout_seconds": 900,
                "default_action": "Skip feedback",
                "metadata": {
                    "session_id": state.get('session_id', '') if state else '',
                    "topic": state.get('topic', '') if state else '',
                    "feedback_type": feedback_type
                }
            },
            "workflow_interrupted": True,
            "current_agent": "human_loop_agent"
        }
    )


def present_decision_context(context_data: Dict[str, Any], state: Optional[VideoGenerationState] = None) -> Dict[str, Any]:
    """Present structured decision context to help human make informed decisions."""
    current_state = state or {}
    
    context_presentation = {
        "session_info": {
            "session_id": current_state.get('session_id', 'unknown'),
            "topic": current_state.get('topic', 'unknown'),
            "timestamp": datetime.now().isoformat()
        },
        "workflow_progress": {
            "current_agent": current_state.get('current_agent', 'unknown'),
            "completed_steps": _get_completed_steps(current_state),
            "pending_steps": _get_pending_steps(current_state),
            "error_count": current_state.get('error_count', 0)
        },
        "context_details": context_data,
        "state_summary": _create_state_summary(current_state),
        "recommendations": _generate_recommendations(context_data, current_state)
    }
    
    logger.info(f"Presenting decision context for session: {current_state.get('session_id', 'unknown')}")
    
    return context_presentation


def validate_human_selection(
    selection: str,
    available_options: List[str],
    validation_rules: Optional[Dict[str, Any]] = None,
    state: Optional[VideoGenerationState] = None
) -> Dict[str, Any]:
    """Validate human selection against available options and rules."""
    validation_result = {
        "valid": False,
        "selection": selection,
        "available_options": available_options,
        "validation_errors": [],
        "suggestions": [],
        "timestamp": datetime.now().isoformat()
    }
    
    # Basic validation: check if selection is in available options
    if selection not in available_options:
        # Try fuzzy matching
        fuzzy_matches = [opt for opt in available_options if selection.lower() in opt.lower()]
        if fuzzy_matches:
            validation_result["suggestions"] = fuzzy_matches
            validation_result["validation_errors"].append(
                f"Selection '{selection}' not found. Did you mean one of: {', '.join(fuzzy_matches)}?"
            )
        else:
            validation_result["validation_errors"].append(
                f"Selection '{selection}' is not valid. Available options: {', '.join(available_options)}"
            )
    else:
        validation_result["valid"] = True
    
    # Apply custom validation rules if provided
    if validation_rules and validation_result["valid"]:
        for rule_name, rule_config in validation_rules.items():
            if not _apply_validation_rule(selection, rule_name, rule_config, state or {}):
                validation_result["valid"] = False
                validation_result["validation_errors"].append(
                    f"Selection failed validation rule '{rule_name}': {rule_config.get('message', 'Rule failed')}"
                )
    
    logger.info(f"Validated human selection: {selection} - Valid: {validation_result['valid']}")
    
    return validation_result


def create_intervention_summary(intervention_data: Dict[str, Any], state: Optional[VideoGenerationState] = None) -> Dict[str, Any]:
    """Create a summary of human intervention for logging and analysis."""
    current_state = state or {}
    
    summary = {
        "intervention_id": intervention_data.get('intervention_id', 'unknown'),
        "type": intervention_data.get('type', 'unknown'),
        "timestamp": datetime.now().isoformat(),
        "session_info": {
            "session_id": current_state.get('session_id', 'unknown'),
            "topic": current_state.get('topic', 'unknown')
        },
        "context": intervention_data.get('context', ''),
        "options_presented": intervention_data.get('options', []),
        "human_response": {
            "selected_option": intervention_data.get('selected_option'),
            "additional_feedback": intervention_data.get('additional_feedback'),
            "confidence_level": intervention_data.get('confidence_level'),
            "response_time_seconds": intervention_data.get('response_time_seconds')
        },
        "outcome": {
            "next_action": intervention_data.get('next_action'),
            "workflow_resumed": intervention_data.get('workflow_resumed', False),
            "modifications_made": intervention_data.get('modifications_made', [])
        },
        "impact_analysis": _analyze_intervention_impact(intervention_data, current_state)
    }
    
    logger.info(f"Created intervention summary: {summary['intervention_id']}")
    
    return summary

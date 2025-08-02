"""
Human intervention interface for seamless integration with other agents.
Provides a clean API for agents to request human input and handle responses.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from langgraph.types import Command

from ..state import VideoGenerationState
from ..tools.human_intervention_tools import (
    DecisionContext,
    InterventionContext,
    InterventionOption
)


logger = logging.getLogger(__name__)


class InterventionPriority(Enum):
    """Priority levels for human interventions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class InterventionResult(Enum):
    """Results of human intervention."""
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    DEFERRED = "deferred"
    TIMEOUT = "timeout"


@dataclass
class QuickApprovalRequest:
    """Quick approval request for simple yes/no decisions."""
    message: str
    default_action: str = "approve"
    timeout_seconds: int = 300
    priority: InterventionPriority = InterventionPriority.MEDIUM


@dataclass
class QualityReviewRequest:
    """Quality review request with metrics and options."""
    content_description: str
    quality_metrics: Dict[str, Any]
    review_criteria: List[str]
    improvement_suggestions: List[str]
    timeout_seconds: int = 600


@dataclass
class ErrorResolutionRequest:
    """Error resolution request with context and recovery options."""
    error_description: str
    error_context: Dict[str, Any]
    recovery_options: List[Dict[str, Any]]
    impact_assessment: str
    urgency: InterventionPriority = InterventionPriority.HIGH


class HumanInterventionInterface:
    """Interface for human intervention requests and management."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the human intervention interface.
        
        Args:
            config: Configuration for human intervention behavior
        """
        self.config = config or {}
        self.auto_approve_low_priority = self.config.get('auto_approve_low_priority', False)
        self.default_timeout = self.config.get('default_timeout_seconds', 300)
        self.enable_quality_gates = self.config.get('enable_quality_gates', True)
        
        # Track intervention history
        self.intervention_history = []
        
        logger.info("HumanInterventionInterface initialized")
    
    async def request_quick_approval(self, 
                                   request: QuickApprovalRequest,
                                   state: VideoGenerationState) -> Command:
        """Request quick approval for simple decisions.
        
        Args:
            request: Quick approval request details
            state: Current workflow state
            
        Returns:
            Command: LangGraph command for approval request
        """
        logger.info(f"Requesting quick approval: {request.message}")
        
        # Check if we should auto-approve based on priority and configuration
        if (request.priority == InterventionPriority.LOW and 
            self.auto_approve_low_priority):
            return self._create_auto_approval_command(request, state)
        
        options = ["Approve", "Reject", "Request more information"]
        
        # Create intervention request directly
        intervention_id = f"approval_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        return Command(
            goto="human_loop_agent",
            update={
                "pending_human_input": {
                    "intervention_id": intervention_id,
                    "type": "approval",
                    "priority": request.priority.value,
                    "context": request.message,
                    "options": options,
                    "requesting_agent": state.get('current_agent', 'unknown'),
                    "timestamp": datetime.now().isoformat(),
                    "timeout_seconds": request.timeout_seconds,
                    "default_action": request.default_action,
                    "metadata": {
                        "session_id": state.get('session_id', ''),
                        "topic": state.get('topic', '')
                    }
                },
                "workflow_interrupted": True,
                "current_agent": "human_loop_agent"
            }
        )
    
    async def request_quality_review(self,
                                   request: QualityReviewRequest,
                                   state: VideoGenerationState) -> Command:
        """Request human quality review with detailed metrics.
        
        Args:
            request: Quality review request details
            state: Current workflow state
            
        Returns:
            Command: LangGraph command for quality review
        """
        logger.info(f"Requesting quality review: {request.content_description}")
        
        # Build structured decision context
        decision_context = {
            "type": "quality_assessment",
            "title": "Quality Review Required",
            "description": request.content_description,
            "state_summary": self._create_quality_state_summary(state),
            "quality_metrics": request.quality_metrics,
            "suggested_actions": request.improvement_suggestions,
            "urgency_level": "medium",
            "timeout_seconds": request.timeout_seconds
        }
        
        # Create structured options
        options = [
            {
                "id": "approve_quality",
                "label": "Approve current quality",
                "description": "Accept the current quality level and continue",
                "consequences": "Workflow continues with current output",
                "recommended": self._is_quality_acceptable(request.quality_metrics)
            },
            {
                "id": "request_improvements",
                "label": "Request quality improvements",
                "description": "Ask for improvements to meet quality standards",
                "consequences": "Additional processing time required",
                "recommended": not self._is_quality_acceptable(request.quality_metrics),
                "requires_input": True,
                "input_type": "text"
            },
            {
                "id": "adjust_criteria",
                "label": "Adjust quality criteria",
                "description": "Modify quality expectations for this workflow",
                "consequences": "Changes quality standards for current session",
                "requires_input": True,
                "input_type": "selection",
                "input_options": ["Lower standards", "Maintain standards", "Raise standards"]
            }
        ]
        
        # Create decision request directly
        intervention_id = f"decision_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        return Command(
            goto="human_loop_agent",
            update={
                "pending_human_input": {
                    "intervention_id": intervention_id,
                    "type": "decision",
                    "priority": decision_context.get('urgency_level', 'medium'),
                    "context": decision_context['description'],
                    "structured_context": decision_context,
                    "options": [opt['label'] for opt in options],
                    "structured_options": options,
                    "requesting_agent": state.get('current_agent', 'unknown'),
                    "timestamp": datetime.now().isoformat(),
                    "timeout_seconds": decision_context.get('timeout_seconds', 600),
                    "default_action": decision_context.get('default_action'),
                    "metadata": {
                        "session_id": state.get('session_id', ''),
                        "topic": state.get('topic', ''),
                        "context_type": decision_context.get('type', 'decision')
                    }
                },
                "workflow_interrupted": True,
                "current_agent": "human_loop_agent"
            }
        )
    
    async def request_error_resolution(self,
                                     request: ErrorResolutionRequest,
                                     state: VideoGenerationState) -> Command:
        """Request human intervention for error resolution.
        
        Args:
            request: Error resolution request details
            state: Current workflow state
            
        Returns:
            Command: LangGraph command for error resolution
        """
        logger.info(f"Requesting error resolution: {request.error_description}")
        
        # Build comprehensive error context
        decision_context = {
            "type": "error_recovery",
            "title": "Error Resolution Required",
            "description": request.error_description,
            "state_summary": self._create_error_state_summary(state),
            "error_details": request.error_context,
            "suggested_actions": [opt.get('label', opt.get('action', '')) for opt in request.recovery_options],
            "urgency_level": request.urgency.value,
            "timeout_seconds": 600  # Longer timeout for error resolution
        }
        
        # Convert recovery options to structured format
        structured_options = []
        for i, option in enumerate(request.recovery_options):
            structured_options.append({
                "id": option.get('id', f"recovery_{i}"),
                "label": option.get('label', option.get('action', f"Option {i+1}")),
                "description": option.get('description', ''),
                "consequences": option.get('consequences', option.get('impact', '')),
                "recommended": option.get('recommended', i == 0)
            })
        
        # Create decision request directly
        intervention_id = f"error_resolution_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        return Command(
            goto="human_loop_agent",
            update={
                "pending_human_input": {
                    "intervention_id": intervention_id,
                    "type": "decision",
                    "priority": decision_context.get('urgency_level', 'high'),
                    "context": decision_context['description'],
                    "structured_context": decision_context,
                    "options": [opt['label'] for opt in structured_options],
                    "structured_options": structured_options,
                    "requesting_agent": state.get('current_agent', 'unknown'),
                    "timestamp": datetime.now().isoformat(),
                    "timeout_seconds": decision_context.get('timeout_seconds', 600),
                    "default_action": decision_context.get('default_action'),
                    "metadata": {
                        "session_id": state.get('session_id', ''),
                        "topic": state.get('topic', ''),
                        "context_type": decision_context.get('type', 'error_recovery')
                    }
                },
                "workflow_interrupted": True,
                "current_agent": "human_loop_agent"
            }
        )
    
    async def request_parameter_adjustment(self,
                                         parameter_name: str,
                                         current_value: Any,
                                         suggested_values: List[Any],
                                         impact_description: str,
                                         state: VideoGenerationState) -> Command:
        """Request human input for parameter adjustment.
        
        Args:
            parameter_name: Name of parameter to adjust
            current_value: Current parameter value
            suggested_values: List of suggested new values
            impact_description: Description of impact of changing parameter
            state: Current workflow state
            
        Returns:
            Command: LangGraph command for parameter adjustment
        """
        logger.info(f"Requesting parameter adjustment: {parameter_name}")
        
        decision_context = {
            "type": "parameter_adjustment",
            "title": f"Adjust Parameter: {parameter_name}",
            "description": f"Current value: {current_value}. {impact_description}",
            "state_summary": self._create_parameter_state_summary(state, parameter_name),
            "urgency_level": "medium",
            "timeout_seconds": 300
        }
        
        # Create options for each suggested value
        options = []
        for value in suggested_values:
            options.append({
                "id": f"set_{value}",
                "label": f"Set to {value}",
                "description": f"Change {parameter_name} from {current_value} to {value}",
                "consequences": f"May affect workflow behavior and performance",
                "recommended": value == suggested_values[0] if suggested_values else False
            })
        
        # Add option to keep current value
        options.append({
            "id": "keep_current",
            "label": f"Keep current value ({current_value})",
            "description": "Continue with current parameter setting",
            "consequences": "No changes to current behavior",
            "recommended": False
        })
        
        # Create decision request directly
        intervention_id = f"parameter_adjustment_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        return Command(
            goto="human_loop_agent",
            update={
                "pending_human_input": {
                    "intervention_id": intervention_id,
                    "type": "decision",
                    "priority": decision_context.get('urgency_level', 'medium'),
                    "context": decision_context['description'],
                    "structured_context": decision_context,
                    "options": [opt['label'] for opt in options],
                    "structured_options": options,
                    "requesting_agent": state.get('current_agent', 'unknown'),
                    "timestamp": datetime.now().isoformat(),
                    "timeout_seconds": decision_context.get('timeout_seconds', 300),
                    "default_action": decision_context.get('default_action'),
                    "metadata": {
                        "session_id": state.get('session_id', ''),
                        "topic": state.get('topic', ''),
                        "context_type": decision_context.get('type', 'parameter_adjustment')
                    }
                },
                "workflow_interrupted": True,
                "current_agent": "human_loop_agent"
            }
        )
    
    async def request_workflow_continuation(self,
                                          continuation_options: List[str],
                                          current_status: str,
                                          state: VideoGenerationState) -> Command:
        """Request human decision on workflow continuation.
        
        Args:
            continuation_options: Available continuation options
            current_status: Current workflow status description
            state: Current workflow state
            
        Returns:
            Command: LangGraph command for workflow continuation
        """
        logger.info(f"Requesting workflow continuation decision: {current_status}")
        
        decision_context = {
            "type": "workflow_continuation",
            "title": "Workflow Continuation Decision",
            "description": current_status,
            "state_summary": self._create_workflow_state_summary(state),
            "suggested_actions": continuation_options,
            "urgency_level": "medium",
            "timeout_seconds": 300,
            "default_action": continuation_options[0] if continuation_options else "Continue"
        }
        
        # Create structured options
        structured_options = []
        for i, option in enumerate(continuation_options):
            structured_options.append({
                "id": f"continue_{i}",
                "label": option,
                "description": f"Choose to {option.lower()}",
                "consequences": "Workflow will proceed according to this choice",
                "recommended": i == 0
            })
        
        # Create decision request directly
        intervention_id = f"workflow_continuation_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        return Command(
            goto="human_loop_agent",
            update={
                "pending_human_input": {
                    "intervention_id": intervention_id,
                    "type": "decision",
                    "priority": decision_context.get('urgency_level', 'medium'),
                    "context": decision_context['description'],
                    "structured_context": decision_context,
                    "options": [opt['label'] for opt in structured_options],
                    "structured_options": structured_options,
                    "requesting_agent": state.get('current_agent', 'unknown'),
                    "timestamp": datetime.now().isoformat(),
                    "timeout_seconds": decision_context.get('timeout_seconds', 300),
                    "default_action": decision_context.get('default_action'),
                    "metadata": {
                        "session_id": state.get('session_id', ''),
                        "topic": state.get('topic', ''),
                        "context_type": decision_context.get('type', 'workflow_continuation')
                    }
                },
                "workflow_interrupted": True,
                "current_agent": "human_loop_agent"
            }
        )
    
    async def collect_feedback(self,
                             feedback_prompt: str,
                             feedback_type: str = "general",
                             required_fields: List[str] = None,
                             state: VideoGenerationState = None) -> Command:
        """Collect structured feedback from human user.
        
        Args:
            feedback_prompt: Prompt for feedback collection
            feedback_type: Type of feedback being collected
            required_fields: Required feedback fields
            state: Current workflow state
            
        Returns:
            Command: LangGraph command for feedback collection
        """
        logger.info(f"Collecting feedback: {feedback_type}")
        
        return request_human_feedback(
            feedback_request=feedback_prompt,
            feedback_type=feedback_type,
            required_fields=required_fields or [],
            state=state
        )
    
    def validate_intervention_response(self,
                                     response: Dict[str, Any],
                                     expected_options: List[str]) -> Dict[str, Any]:
        """Validate human intervention response.
        
        Args:
            response: Human response to validate
            expected_options: Expected response options
            
        Returns:
            Dict: Validation result
        """
        from ..tools.human_intervention_tools import validate_human_selection
        
        selected_option = response.get('selected_option', '')
        
        return validate_human_selection(
            selection=selected_option,
            available_options=expected_options,
            validation_rules={
                "error_threshold": {"max_errors": 5, "message": "Too many errors for this action"},
                "quality_threshold": {"min_quality": 0.6, "message": "Quality below acceptable threshold"}
            }
        )
    
    def create_intervention_record(self,
                                 intervention_data: Dict[str, Any],
                                 state: VideoGenerationState) -> Dict[str, Any]:
        """Create a record of human intervention for analysis.
        
        Args:
            intervention_data: Data about the intervention
            state: Current workflow state
            
        Returns:
            Dict: Intervention record
        """
        record = create_intervention_summary(intervention_data, state)
        self.intervention_history.append(record)
        return record
    
    def get_intervention_patterns(self) -> Dict[str, Any]:
        """Analyze intervention history for patterns.
        
        Returns:
            Dict: Analysis of intervention patterns
        """
        if not self.intervention_history:
            return {"message": "No intervention history available"}
        
        # Analyze patterns in intervention history
        patterns = {
            "total_interventions": len(self.intervention_history),
            "intervention_types": {},
            "common_decisions": {},
            "average_response_time": 0,
            "quality_trends": []
        }
        
        for record in self.intervention_history:
            # Count intervention types
            intervention_type = record.get('type', 'unknown')
            patterns["intervention_types"][intervention_type] = patterns["intervention_types"].get(intervention_type, 0) + 1
            
            # Track common decisions
            selected_option = record.get('human_response', {}).get('selected_option', 'unknown')
            patterns["common_decisions"][selected_option] = patterns["common_decisions"].get(selected_option, 0) + 1
        
        return patterns
    
    async def create_agent_specific_intervention(self,
                                               agent_name: str,
                                               intervention_type: str,
                                               context_data: Dict[str, Any],
                                               options: List[str],
                                               state: VideoGenerationState) -> Command:
        """Create agent-specific intervention with enhanced context.
        
        Args:
            agent_name: Name of the requesting agent
            intervention_type: Type of intervention needed
            context_data: Agent-specific context data
            options: Available options for human selection
            state: Current workflow state
            
        Returns:
            Command: LangGraph command for agent-specific intervention
        """
        from ..tools.agent_integration_tools import create_intervention_context
        
        # Create comprehensive intervention context
        intervention_context = create_intervention_context(
            agent_name=agent_name,
            intervention_type=intervention_type,
            state=state,
            additional_context=context_data
        )
        
        intervention_id = f"agent_specific_{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        logger.info(f"Creating agent-specific intervention for {agent_name}: {intervention_type}")
        
        return Command(
            goto="human_loop_agent",
            update={
                "pending_human_input": {
                    "intervention_id": intervention_id,
                    "type": intervention_type,
                    "priority": context_data.get('priority', 'medium'),
                    "context": context_data.get('description', f"{intervention_type} needed for {agent_name}"),
                    "agent_context": intervention_context,
                    "options": options,
                    "requesting_agent": agent_name,
                    "timestamp": datetime.now().isoformat(),
                    "timeout_seconds": context_data.get('timeout_seconds', 300),
                    "default_action": options[0] if options else "continue",
                    "metadata": {
                        "session_id": state.get('session_id', ''),
                        "topic": state.get('topic', ''),
                        "agent_specific": True,
                        "intervention_type": intervention_type
                    }
                },
                "workflow_interrupted": True,
                "current_agent": "human_loop_agent"
            }
        )
    
    def validate_and_process_response(self,
                                    response: Dict[str, Any],
                                    expected_format: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and process human intervention response.
        
        Args:
            response: Human response to validate
            expected_format: Expected response format
            
        Returns:
            Dict: Processed and validated response
        """
        from ..tools.agent_integration_tools import validate_intervention_response
        
        validation_result = validate_intervention_response(response, expected_format)
        
        if validation_result["valid"]:
            # Process the response
            processed_response = {
                **validation_result["normalized_response"],
                "validation_status": "valid",
                "processed_timestamp": datetime.now().isoformat(),
                "processing_notes": validation_result.get("warnings", [])
            }
        else:
            # Handle invalid response
            processed_response = {
                **response,
                "validation_status": "invalid",
                "validation_errors": validation_result["errors"],
                "processed_timestamp": datetime.now().isoformat(),
                "requires_resubmission": True
            }
        
        return processed_response
    
    def create_intervention_summary_report(self) -> Dict[str, Any]:
        """Create comprehensive intervention summary report.
        
        Returns:
            Dict: Comprehensive intervention summary
        """
        if not self.intervention_history:
            return {"message": "No intervention history available"}
        
        # Calculate summary statistics
        total_interventions = len(self.intervention_history)
        intervention_types = {}
        agent_breakdown = {}
        success_rate = 0
        
        for record in self.intervention_history:
            # Count by type
            intervention_type = record.get('type', 'unknown')
            intervention_types[intervention_type] = intervention_types.get(intervention_type, 0) + 1
            
            # Count by agent
            requesting_agent = record.get('requesting_agent', 'unknown')
            if requesting_agent not in agent_breakdown:
                agent_breakdown[requesting_agent] = {
                    "total": 0,
                    "types": {},
                    "success_count": 0
                }
            
            agent_breakdown[requesting_agent]["total"] += 1
            agent_breakdown[requesting_agent]["types"][intervention_type] = (
                agent_breakdown[requesting_agent]["types"].get(intervention_type, 0) + 1
            )
            
            # Track success
            if record.get('human_response', {}).get('selected_option') != 'abort':
                success_rate += 1
                agent_breakdown[requesting_agent]["success_count"] += 1
        
        # Calculate success rates
        overall_success_rate = (success_rate / total_interventions) * 100 if total_interventions > 0 else 0
        
        for agent_data in agent_breakdown.values():
            agent_data["success_rate"] = (
                (agent_data["success_count"] / agent_data["total"]) * 100 
                if agent_data["total"] > 0 else 0
            )
        
        return {
            "summary": {
                "total_interventions": total_interventions,
                "overall_success_rate": overall_success_rate,
                "most_common_type": max(intervention_types.items(), key=lambda x: x[1])[0] if intervention_types else "none",
                "most_active_agent": max(agent_breakdown.items(), key=lambda x: x[1]["total"])[0] if agent_breakdown else "none"
            },
            "intervention_types": intervention_types,
            "agent_breakdown": agent_breakdown,
            "trends": self._analyze_intervention_trends(),
            "recommendations": self._generate_intervention_recommendations()
        }
    
    def _analyze_intervention_trends(self) -> Dict[str, Any]:
        """Analyze trends in intervention history."""
        if len(self.intervention_history) < 2:
            return {"message": "Insufficient data for trend analysis"}
        
        # Simple trend analysis
        recent_interventions = self.intervention_history[-5:]  # Last 5 interventions
        older_interventions = self.intervention_history[:-5] if len(self.intervention_history) > 5 else []
        
        trends = {
            "intervention_frequency": "stable",
            "success_rate_trend": "stable",
            "complexity_trend": "stable"
        }
        
        # Analyze frequency trend
        if len(recent_interventions) > len(older_interventions):
            trends["intervention_frequency"] = "increasing"
        elif len(recent_interventions) < len(older_interventions):
            trends["intervention_frequency"] = "decreasing"
        
        return trends
    
    def _generate_intervention_recommendations(self) -> List[str]:
        """Generate recommendations based on intervention history."""
        recommendations = []
        
        if not self.intervention_history:
            return ["No intervention history available for recommendations"]
        
        patterns = self.get_intervention_patterns()
        
        # High intervention count
        if patterns["total_interventions"] > 10:
            recommendations.append("Consider reviewing workflow automation to reduce intervention frequency")
        
        # Common error patterns
        if "error_resolution" in patterns["intervention_types"]:
            error_count = patterns["intervention_types"]["error_resolution"]
            if error_count > 3:
                recommendations.append("Frequent error interventions detected - review error handling logic")
        
        # Quality issues
        if "quality_gate" in patterns["intervention_types"]:
            quality_count = patterns["intervention_types"]["quality_gate"]
            if quality_count > 2:
                recommendations.append("Consider adjusting quality thresholds or improving automated quality checks")
        
        return recommendations
    
    def _create_auto_approval_command(self, request: QuickApprovalRequest, state: VideoGenerationState) -> Command:
        """Create auto-approval command for low priority requests."""
        logger.info(f"Auto-approving low priority request: {request.message}")
        
        return Command(
            goto="human_loop_agent",
            update={
                "human_feedback": {
                    "intervention_id": f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "selected_option": request.default_action,
                    "additional_feedback": "Auto-approved due to low priority setting",
                    "timestamp": datetime.now().isoformat(),
                    "confidence_level": 5,
                    "metadata": {"auto_approved": True},
                    "processed": False
                },
                "pending_human_input": None,
                "current_agent": "human_loop_agent"
            }
        )
    
    def _is_quality_acceptable(self, quality_metrics: Dict[str, Any]) -> bool:
        """Check if quality metrics meet acceptable thresholds."""
        if not self.enable_quality_gates:
            return True
        
        # Define quality thresholds
        thresholds = {
            "overall_score": 0.7,
            "visual_quality": 0.6,
            "code_quality": 0.8,
            "error_rate": 0.2  # Lower is better
        }
        
        for metric, threshold in thresholds.items():
            if metric in quality_metrics:
                value = quality_metrics[metric]
                if metric == "error_rate":
                    if value > threshold:
                        return False
                else:
                    if value < threshold:
                        return False
        
        return True
    
    def _create_quality_state_summary(self, state: VideoGenerationState) -> Dict[str, Any]:
        """Create state summary focused on quality metrics."""
        return {
            "scenes_completed": len(state.get('rendered_videos', {})),
            "visual_errors": sum(len(errors) for errors in state.get('visual_errors', {}).values()),
            "code_errors": len(state.get('code_errors', {})),
            "rendering_errors": len(state.get('rendering_errors', {})),
            "quality_settings": {
                "default_quality": state.get('default_quality', 'medium'),
                "use_visual_fix": state.get('use_visual_fix_code', False)
            }
        }
    
    def _create_error_state_summary(self, state: VideoGenerationState) -> Dict[str, Any]:
        """Create state summary focused on error information."""
        return {
            "total_errors": state.get('error_count', 0),
            "error_breakdown": {
                "code_errors": len(state.get('code_errors', {})),
                "rendering_errors": len(state.get('rendering_errors', {})),
                "escalated_errors": len(state.get('escalated_errors', []))
            },
            "retry_counts": state.get('retry_count', {}),
            "current_agent": state.get('current_agent', 'unknown')
        }
    
    def _create_parameter_state_summary(self, state: VideoGenerationState, parameter_name: str) -> Dict[str, Any]:
        """Create state summary focused on parameter context."""
        return {
            "parameter_context": parameter_name,
            "current_configuration": {
                "use_rag": state.get('use_rag', False),
                "max_retries": state.get('max_retries', 0),
                "default_quality": state.get('default_quality', 'medium'),
                "max_scene_concurrency": state.get('max_scene_concurrency', 1)
            },
            "performance_impact": {
                "error_count": state.get('error_count', 0),
                "average_execution_time": "calculated_from_metrics"
            }
        }
    
    def _create_workflow_state_summary(self, state: VideoGenerationState) -> Dict[str, Any]:
        """Create comprehensive workflow state summary."""
        return {
            "progress": {
                "planning_complete": bool(state.get('scene_outline')),
                "coding_complete": bool(state.get('generated_code')),
                "rendering_complete": bool(state.get('rendered_videos')),
                "analysis_complete": bool(state.get('visual_analysis_results'))
            },
            "health": {
                "error_count": state.get('error_count', 0),
                "workflow_interrupted": state.get('workflow_interrupted', False),
                "current_agent": state.get('current_agent', 'unknown')
            },
            "outputs": {
                "scenes_planned": len(state.get('scene_implementations', {})),
                "videos_rendered": len(state.get('rendered_videos', {})),
                "final_video": state.get('combined_video_path')
            }
        }
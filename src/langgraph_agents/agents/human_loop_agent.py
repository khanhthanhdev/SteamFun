"""
HumanLoopAgent for user intervention and approval workflows.
Implements decision point identification, approval workflow management,
feedback collection, and resume workflow capabilities.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from langgraph.types import Command

from ..base_agent import BaseAgent
from ..state import VideoGenerationState


logger = logging.getLogger(__name__)


class InterventionType(Enum):
    """Types of human intervention requests."""
    APPROVAL = "approval"
    DECISION = "decision"
    FEEDBACK = "feedback"
    ERROR_RESOLUTION = "error_resolution"
    QUALITY_REVIEW = "quality_review"


class InterventionPriority(Enum):
    """Priority levels for human intervention."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class InterventionRequest:
    """Structured intervention request."""
    id: str
    type: InterventionType
    priority: InterventionPriority
    context: str
    options: List[str]
    requesting_agent: str
    timestamp: str
    metadata: Dict[str, Any]
    timeout_seconds: Optional[int] = None
    default_action: Optional[str] = None


@dataclass
class HumanFeedback:
    """Structured human feedback response."""
    intervention_id: str
    selected_option: str
    additional_feedback: Optional[str]
    timestamp: str
    confidence_level: Optional[int] = None  # 1-10 scale
    metadata: Dict[str, Any] = None


class HumanLoopAgent(BaseAgent):
    """HumanLoopAgent for human-in-the-loop interactions.
    
    Handles decision point identification, approval workflows,
    feedback collection, and workflow resumption after human input.
    """
    
    def __init__(self, config, system_config):
        """Initialize HumanLoopAgent.
        
        Args:
            config: Agent configuration
            system_config: System configuration
        """
        super().__init__(config, system_config)
        
        # Human loop configuration
        self.human_loop_config = system_config.human_loop_config
        self.default_timeout = self.human_loop_config.timeout_seconds
        self.auto_approve_low_priority = self.human_loop_config.auto_approve_low_risk
        self.enable_feedback_collection = True  # Default value since not in config model
        
        # Decision point patterns
        self.decision_patterns = {
            'code_error_threshold': 3,
            'rendering_failure_threshold': 2,
            'visual_error_threshold': 2,
            'quality_score_threshold': 0.6
        }
        
        # Workflow resume mapping
        self.resume_mapping = {
            'planner_agent': self._resume_planning_workflow,
            'code_generator_agent': self._resume_code_generation_workflow,
            'renderer_agent': self._resume_rendering_workflow,
            'visual_analysis_agent': self._resume_visual_analysis_workflow,
            'error_handler_agent': self._resume_error_handling_workflow
        }
        
        logger.info(f"HumanLoopAgent initialized with config: {config.name}")
    
    async def execute(self, state: VideoGenerationState) -> Command:
        """Execute human loop operations.
        
        Args:
            state: Current workflow state
            
        Returns:
            Command: LangGraph command for next action
        """
        self.log_agent_action("human_loop_execution_start", {
            'topic': state.get('topic', ''),
            'pending_input': bool(state.get('pending_human_input')),
            'session_id': state.get('session_id', '')
        })
        
        # Check if there's a pending human input request
        pending_input = state.get('pending_human_input')
        if pending_input:
            return await self._handle_intervention_request(state, pending_input)
        
        # Check if we need to identify decision points
        decision_point = self._identify_decision_points(state)
        if decision_point:
            return await self._create_intervention_request(state, decision_point)
        
        # Check for existing human feedback to process
        human_feedback = state.get('human_feedback')
        if human_feedback and not human_feedback.get('processed', False):
            return await self._process_human_feedback(state, human_feedback)
        
        # No human intervention needed, continue workflow
        return await self._continue_workflow(state)
    
    def _identify_decision_points(self, state: VideoGenerationState) -> Optional[Dict[str, Any]]:
        """Identify points in the workflow that require human intervention.
        
        Args:
            state: Current workflow state
            
        Returns:
            Optional[Dict]: Decision point information if intervention needed
        """
        # Check error thresholds
        error_count = state.get('error_count', 0)
        if error_count >= self.decision_patterns['code_error_threshold']:
            return {
                'type': InterventionType.ERROR_RESOLUTION,
                'priority': InterventionPriority.HIGH,
                'context': f"Multiple errors detected ({error_count} errors). Manual intervention may be required.",
                'options': ['Continue with error handling', 'Restart workflow', 'Modify parameters', 'Abort workflow'],
                'metadata': {'error_count': error_count, 'escalated_errors': state.get('escalated_errors', [])}
            }
        
        # Check rendering failures
        rendering_errors = state.get('rendering_errors', {})
        if len(rendering_errors) >= self.decision_patterns['rendering_failure_threshold']:
            return {
                'type': InterventionType.DECISION,
                'priority': InterventionPriority.MEDIUM,
                'context': f"Multiple rendering failures detected. Consider adjusting rendering parameters.",
                'options': ['Retry with lower quality', 'Retry with different parameters', 'Skip problematic scenes', 'Continue anyway'],
                'metadata': {'rendering_errors': rendering_errors}
            }
        
        # Check visual analysis results
        visual_errors = state.get('visual_errors', {})
        total_visual_errors = sum(len(errors) for errors in visual_errors.values())
        if total_visual_errors >= self.decision_patterns['visual_error_threshold']:
            return {
                'type': InterventionType.QUALITY_REVIEW,
                'priority': InterventionPriority.MEDIUM,
                'context': f"Visual quality issues detected in {len(visual_errors)} scenes. Review may be needed.",
                'options': ['Accept current quality', 'Regenerate problematic scenes', 'Adjust visual parameters', 'Manual review'],
                'metadata': {'visual_errors': visual_errors}
            }
        
        # Check for workflow completion approval
        if state.get('combined_video_path') and not state.get('workflow_complete'):
            return {
                'type': InterventionType.APPROVAL,
                'priority': InterventionPriority.LOW,
                'context': "Video generation completed. Please review the final output.",
                'options': ['Approve and complete', 'Request modifications', 'Regenerate with different parameters'],
                'metadata': {'video_path': state.get('combined_video_path')}
            }
        
        return None
    
    async def _create_intervention_request(self, state: VideoGenerationState, decision_point: Dict[str, Any]) -> Command:
        """Create a human intervention request.
        
        Args:
            state: Current workflow state
            decision_point: Decision point information
            
        Returns:
            Command: LangGraph command with intervention request
        """
        intervention_id = f"intervention_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        intervention_request = InterventionRequest(
            id=intervention_id,
            type=decision_point['type'],
            priority=decision_point['priority'],
            context=decision_point['context'],
            options=decision_point['options'],
            requesting_agent=state.get('current_agent', 'human_loop_agent'),
            timestamp=datetime.now().isoformat(),
            metadata=decision_point.get('metadata', {}),
            timeout_seconds=self.default_timeout,
            default_action=decision_point['options'][0] if decision_point['options'] else None
        )
        
        self.log_agent_action("intervention_request_created", {
            'intervention_id': intervention_id,
            'type': decision_point['type'].value,
            'priority': decision_point['priority'].value,
            'requesting_agent': intervention_request.requesting_agent
        })
        
        # Check if we should auto-approve low priority requests
        if (decision_point['priority'] == InterventionPriority.LOW and 
            self.auto_approve_low_priority):
            return await self._auto_approve_intervention(state, intervention_request)
        
        return Command(
            goto="human_loop_agent",
            update={
                "pending_human_input": {
                    "intervention_id": intervention_request.id,
                    "type": intervention_request.type.value,
                    "priority": intervention_request.priority.value,
                    "context": intervention_request.context,
                    "options": intervention_request.options,
                    "requesting_agent": intervention_request.requesting_agent,
                    "timestamp": intervention_request.timestamp,
                    "metadata": intervention_request.metadata,
                    "timeout_seconds": intervention_request.timeout_seconds,
                    "default_action": intervention_request.default_action
                },
                "workflow_interrupted": True,
                "current_agent": "human_loop_agent"
            }
        )
    
    async def _handle_intervention_request(self, state: VideoGenerationState, pending_input: Dict[str, Any]) -> Command:
        """Handle an existing intervention request.
        
        Args:
            state: Current workflow state
            pending_input: Pending intervention request
            
        Returns:
            Command: LangGraph command for next action
        """
        intervention_id = pending_input.get('intervention_id')
        
        self.log_agent_action("handling_intervention_request", {
            'intervention_id': intervention_id,
            'type': pending_input.get('type'),
            'requesting_agent': pending_input.get('requesting_agent')
        })
        
        # Check for timeout
        if self._is_intervention_timed_out(pending_input):
            return await self._handle_intervention_timeout(state, pending_input)
        
        # In a real implementation, this would present the intervention to the user
        # For now, we'll simulate user input based on the intervention type
        simulated_feedback = self._simulate_user_feedback(pending_input)
        
        return await self._process_human_feedback(state, simulated_feedback)
    
    def _simulate_user_feedback(self, pending_input: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate user feedback for testing purposes.
        
        Args:
            pending_input: Pending intervention request
            
        Returns:
            Dict: Simulated human feedback
        """
        intervention_type = pending_input.get('type')
        options = pending_input.get('options', [])
        
        # Simulate different responses based on intervention type
        if intervention_type == InterventionType.APPROVAL.value:
            selected_option = options[0] if options else "Approve and complete"
            additional_feedback = "Looks good, proceeding with completion."
        elif intervention_type == InterventionType.ERROR_RESOLUTION.value:
            selected_option = options[0] if options else "Continue with error handling"
            additional_feedback = "Let the system attempt automatic error resolution first."
        elif intervention_type == InterventionType.QUALITY_REVIEW.value:
            selected_option = options[0] if options else "Accept current quality"
            additional_feedback = "Quality is acceptable for this use case."
        else:
            selected_option = options[0] if options else "Continue"
            additional_feedback = "Proceeding with default option."
        
        return {
            "intervention_id": pending_input.get('intervention_id'),
            "selected_option": selected_option,
            "additional_feedback": additional_feedback,
            "timestamp": datetime.now().isoformat(),
            "confidence_level": 8,
            "metadata": {"simulated": True},
            "processed": False
        }
    
    async def _process_human_feedback(self, state: VideoGenerationState, feedback: Dict[str, Any]) -> Command:
        """Process human feedback and determine next action.
        
        Args:
            state: Current workflow state
            feedback: Human feedback data
            
        Returns:
            Command: LangGraph command for next action
        """
        intervention_id = feedback.get('intervention_id')
        selected_option = feedback.get('selected_option')
        
        self.log_agent_action("processing_human_feedback", {
            'intervention_id': intervention_id,
            'selected_option': selected_option,
            'confidence_level': feedback.get('confidence_level')
        })
        
        # Determine next action based on feedback
        next_command = self._determine_next_action(state, feedback)
        
        # Update state with processed feedback
        processed_feedback = {**feedback, "processed": True}
        
        # Clear pending input
        update_data = {
            "human_feedback": processed_feedback,
            "pending_human_input": None,
            "workflow_interrupted": False
        }
        
        # Add any additional updates from next action determination
        if hasattr(next_command, 'update') and next_command.update:
            update_data.update(next_command.update)
        
        return Command(
            goto=next_command.goto if hasattr(next_command, 'goto') else "END",
            update=update_data
        )
    
    def _determine_next_action(self, state: VideoGenerationState, feedback: Dict[str, Any]) -> Command:
        """Determine the next workflow action based on human feedback.
        
        Args:
            state: Current workflow state
            feedback: Human feedback data
            
        Returns:
            Command: Next action command
        """
        selected_option = feedback.get('selected_option', '').lower()
        pending_input = state.get('pending_human_input', {})
        requesting_agent = pending_input.get('requesting_agent', 'planner_agent')
        
        # Handle different feedback options
        if 'approve' in selected_option or 'complete' in selected_option:
            return Command(goto="END", update={"workflow_complete": True})
        
        elif 'restart' in selected_option or 'regenerate' in selected_option:
            return Command(goto="planner_agent", update={
                "scene_outline": None,
                "scene_implementations": {},
                "generated_code": {},
                "rendered_videos": {},
                "error_count": 0,
                "retry_count": {}
            })
        
        elif 'continue' in selected_option or 'proceed' in selected_option:
            # Resume the workflow from the requesting agent
            return self._resume_workflow_from_agent(state, requesting_agent)
        
        elif 'abort' in selected_option or 'cancel' in selected_option:
            return Command(goto="END", update={
                "workflow_complete": True,
                "workflow_interrupted": True,
                "error_count": state.get('error_count', 0) + 1
            })
        
        elif 'modify' in selected_option or 'adjust' in selected_option:
            # Return to the requesting agent with modification flag
            return Command(goto=requesting_agent, update={
                "modification_requested": True,
                "human_modification_feedback": feedback.get('additional_feedback', '')
            })
        
        else:
            # Default: continue with the requesting agent
            return self._resume_workflow_from_agent(state, requesting_agent)
    
    def _resume_workflow_from_agent(self, state: VideoGenerationState, agent_name: str) -> Command:
        """Resume workflow from a specific agent.
        
        Args:
            state: Current workflow state
            agent_name: Name of agent to resume from
            
        Returns:
            Command: Resume command
        """
        if agent_name in self.resume_mapping:
            return self.resume_mapping[agent_name](state)
        else:
            # Default resume behavior
            return Command(goto=agent_name, update={"current_agent": agent_name})
    
    def _resume_planning_workflow(self, state: VideoGenerationState) -> Command:
        """Resume planning workflow after human intervention."""
        return Command(goto="planner_agent", update={
            "current_agent": "planner_agent",
            "next_agent": "code_generator_agent"
        })
    
    def _resume_code_generation_workflow(self, state: VideoGenerationState) -> Command:
        """Resume code generation workflow after human intervention."""
        return Command(goto="code_generator_agent", update={
            "current_agent": "code_generator_agent",
            "next_agent": "renderer_agent"
        })
    
    def _resume_rendering_workflow(self, state: VideoGenerationState) -> Command:
        """Resume rendering workflow after human intervention."""
        return Command(goto="renderer_agent", update={
            "current_agent": "renderer_agent",
            "next_agent": "visual_analysis_agent"
        })
    
    def _resume_visual_analysis_workflow(self, state: VideoGenerationState) -> Command:
        """Resume visual analysis workflow after human intervention."""
        return Command(goto="visual_analysis_agent", update={
            "current_agent": "visual_analysis_agent",
            "next_agent": "human_loop_agent"
        })
    
    def _resume_error_handling_workflow(self, state: VideoGenerationState) -> Command:
        """Resume error handling workflow after human intervention."""
        return Command(goto="error_handler_agent", update={
            "current_agent": "error_handler_agent"
        })
    
    async def _auto_approve_intervention(self, state: VideoGenerationState, intervention: InterventionRequest) -> Command:
        """Auto-approve low priority interventions.
        
        Args:
            state: Current workflow state
            intervention: Intervention request to auto-approve
            
        Returns:
            Command: Auto-approval command
        """
        self.log_agent_action("auto_approving_intervention", {
            'intervention_id': intervention.id,
            'type': intervention.type.value,
            'default_action': intervention.default_action
        })
        
        auto_feedback = {
            "intervention_id": intervention.id,
            "selected_option": intervention.default_action,
            "additional_feedback": "Auto-approved due to low priority setting.",
            "timestamp": datetime.now().isoformat(),
            "confidence_level": 5,
            "metadata": {"auto_approved": True},
            "processed": False
        }
        
        return await self._process_human_feedback(state, auto_feedback)
    
    def _is_intervention_timed_out(self, pending_input: Dict[str, Any]) -> bool:
        """Check if an intervention request has timed out.
        
        Args:
            pending_input: Pending intervention request
            
        Returns:
            bool: True if timed out
        """
        if not pending_input.get('timeout_seconds'):
            return False
        
        timestamp_str = pending_input.get('timestamp')
        if not timestamp_str:
            return False
        
        try:
            request_time = datetime.fromisoformat(timestamp_str)
            elapsed_seconds = (datetime.now() - request_time).total_seconds()
            return elapsed_seconds > pending_input['timeout_seconds']
        except (ValueError, TypeError):
            return False
    
    async def _handle_intervention_timeout(self, state: VideoGenerationState, pending_input: Dict[str, Any]) -> Command:
        """Handle intervention request timeout.
        
        Args:
            state: Current workflow state
            pending_input: Timed out intervention request
            
        Returns:
            Command: Timeout handling command
        """
        intervention_id = pending_input.get('intervention_id')
        default_action = pending_input.get('default_action')
        
        self.log_agent_action("intervention_timeout", {
            'intervention_id': intervention_id,
            'default_action': default_action
        })
        
        # Use default action if available
        if default_action:
            timeout_feedback = {
                "intervention_id": intervention_id,
                "selected_option": default_action,
                "additional_feedback": "Selected due to timeout - using default action.",
                "timestamp": datetime.now().isoformat(),
                "confidence_level": 3,
                "metadata": {"timeout": True},
                "processed": False
            }
            return await self._process_human_feedback(state, timeout_feedback)
        else:
            # No default action, continue workflow
            return await self._continue_workflow(state)
    
    async def _continue_workflow(self, state: VideoGenerationState) -> Command:
        """Continue workflow when no human intervention is needed.
        
        Args:
            state: Current workflow state
            
        Returns:
            Command: Continue workflow command
        """
        # Determine next agent based on current workflow state
        if not state.get('scene_outline'):
            next_agent = "planner_agent"
        elif not state.get('generated_code'):
            next_agent = "code_generator_agent"
        elif not state.get('rendered_videos'):
            next_agent = "renderer_agent"
        elif not state.get('visual_analysis_results'):
            next_agent = "visual_analysis_agent"
        elif state.get('combined_video_path') and not state.get('workflow_complete'):
            # Final completion
            return Command(goto="END", update={"workflow_complete": True})
        else:
            next_agent = "END"
        
        self.log_agent_action("continuing_workflow", {
            'next_agent': next_agent,
            'workflow_state': {
                'has_outline': bool(state.get('scene_outline')),
                'has_code': bool(state.get('generated_code')),
                'has_videos': bool(state.get('rendered_videos')),
                'has_analysis': bool(state.get('visual_analysis_results'))
            }
        })
        
        return Command(
            goto=next_agent,
            update={
                "current_agent": next_agent if next_agent != "END" else "human_loop_agent",
                "workflow_interrupted": False
            }
        )
"""
Agent integration tools for seamless human intervention integration.
Provides utilities for agents to easily integrate human intervention capabilities.
"""

import logging
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from langgraph.types import Command

from ..state import VideoGenerationState
from .human_intervention_tools import (
    request_agent_approval,
    request_quality_gate_approval,
    request_error_resolution_decision,
    request_parameter_modification,
    create_interactive_decision_interface
)


logger = logging.getLogger(__name__)


class InterventionTrigger(Enum):
    """Triggers that can cause human intervention."""
    ERROR_THRESHOLD = "error_threshold"
    QUALITY_GATE = "quality_gate"
    PARAMETER_CHANGE = "parameter_change"
    WORKFLOW_DECISION = "workflow_decision"
    MANUAL_REQUEST = "manual_request"


@dataclass
class InterventionRule:
    """Rule for when to trigger human intervention."""
    trigger: InterventionTrigger
    condition: Callable[[VideoGenerationState], bool]
    priority: str = "medium"
    timeout_seconds: int = 300
    auto_approve_conditions: Optional[Callable[[VideoGenerationState], bool]] = None


class AgentInterventionManager:
    """Manager for agent-specific human intervention capabilities."""
    
    def __init__(self, agent_name: str, config: Dict[str, Any] = None):
        """Initialize intervention manager for an agent.
        
        Args:
            agent_name: Name of the agent
            config: Configuration for intervention behavior
        """
        self.agent_name = agent_name
        self.config = config or {}
        self.intervention_rules = []
        self.intervention_history = []
        
        # Default intervention rules
        self._setup_default_rules()
        
        logger.info(f"AgentInterventionManager initialized for {agent_name}")
    
    def _setup_default_rules(self):
        """Setup default intervention rules for the agent."""
        # Error threshold rule
        self.add_intervention_rule(
            InterventionRule(
                trigger=InterventionTrigger.ERROR_THRESHOLD,
                condition=lambda state: state.get('error_count', 0) >= 3,
                priority="high",
                timeout_seconds=600
            )
        )
        
        # Quality gate rule (if quality metrics are available)
        self.add_intervention_rule(
            InterventionRule(
                trigger=InterventionTrigger.QUALITY_GATE,
                condition=lambda state: self._has_quality_issues(state),
                priority="medium",
                timeout_seconds=300
            )
        )
    
    def add_intervention_rule(self, rule: InterventionRule):
        """Add an intervention rule.
        
        Args:
            rule: Intervention rule to add
        """
        self.intervention_rules.append(rule)
        logger.debug(f"Added intervention rule for {self.agent_name}: {rule.trigger.value}")
    
    def check_intervention_needed(self, state: VideoGenerationState) -> Optional[InterventionRule]:
        """Check if human intervention is needed based on rules.
        
        Args:
            state: Current workflow state
            
        Returns:
            Optional[InterventionRule]: Rule that triggered intervention, if any
        """
        for rule in self.intervention_rules:
            try:
                if rule.condition(state):
                    # Check auto-approve conditions
                    if rule.auto_approve_conditions and rule.auto_approve_conditions(state):
                        logger.info(f"Auto-approving intervention for {self.agent_name}: {rule.trigger.value}")
                        continue
                    
                    logger.info(f"Intervention needed for {self.agent_name}: {rule.trigger.value}")
                    return rule
            except Exception as e:
                logger.error(f"Error checking intervention rule {rule.trigger.value}: {e}")
                continue
        
        return None
    
    async def request_approval(self, 
                             context: str, 
                             options: List[str],
                             agent_data: Dict[str, Any] = None,
                             priority: str = "medium",
                             state: VideoGenerationState = None) -> Command:
        """Request human approval with agent-specific context.
        
        Args:
            context: Context for approval request
            options: Available options
            agent_data: Agent-specific data to include
            priority: Priority level
            state: Current workflow state
            
        Returns:
            Command: LangGraph command for approval request
        """
        return await request_agent_approval(
            requesting_agent=self.agent_name,
            approval_context=context,
            approval_options=options,
            agent_specific_data=agent_data or {},
            priority=priority,
            state=state
        )
    
    async def request_quality_approval(self,
                                     quality_metrics: Dict[str, Any],
                                     content_description: str,
                                     improvement_options: List[str] = None,
                                     state: VideoGenerationState = None) -> Command:
        """Request quality gate approval.
        
        Args:
            quality_metrics: Current quality metrics
            content_description: Description of content being evaluated
            improvement_options: Available improvement options
            state: Current workflow state
            
        Returns:
            Command: LangGraph command for quality approval
        """
        # Define default quality thresholds
        default_thresholds = {
            "overall_score": 0.7,
            "visual_quality": 0.6,
            "code_quality": 0.8,
            "error_rate": 0.2
        }
        
        return await request_quality_gate_approval(
            quality_metrics=quality_metrics,
            quality_thresholds=default_thresholds,
            content_description=content_description,
            improvement_options=improvement_options or ["Regenerate", "Adjust parameters", "Accept current quality"],
            requesting_agent=self.agent_name,
            state=state
        )
    
    async def request_error_resolution(self,
                                     error_details: Dict[str, Any],
                                     recovery_strategies: List[Dict[str, Any]],
                                     state: VideoGenerationState = None) -> Command:
        """Request error resolution decision.
        
        Args:
            error_details: Details about the error
            recovery_strategies: Available recovery strategies
            state: Current workflow state
            
        Returns:
            Command: LangGraph command for error resolution
        """
        error_context = {
            "agent": self.agent_name,
            "timestamp": datetime.now().isoformat(),
            "workflow_state": {
                "topic": state.get('topic', '') if state else '',
                "session_id": state.get('session_id', '') if state else '',
                "error_count": state.get('error_count', 0) if state else 0
            }
        }
        
        return await request_error_resolution_decision(
            error_details=error_details,
            recovery_strategies=recovery_strategies,
            error_context=error_context,
            requesting_agent=self.agent_name,
            state=state
        )
    
    async def request_parameter_change(self,
                                     parameter_name: str,
                                     current_value: Any,
                                     suggested_values: List[Any],
                                     reason: str,
                                     state: VideoGenerationState = None) -> Command:
        """Request parameter modification.
        
        Args:
            parameter_name: Name of parameter to modify
            current_value: Current parameter value
            suggested_values: Suggested new values
            reason: Reason for modification
            state: Current workflow state
            
        Returns:
            Command: LangGraph command for parameter modification
        """
        # Analyze impact of parameter change
        impact_analysis = self._analyze_parameter_impact(parameter_name, current_value, suggested_values, state)
        
        return await request_parameter_modification(
            parameter_name=parameter_name,
            current_value=current_value,
            suggested_values=suggested_values,
            modification_reason=reason,
            impact_analysis=impact_analysis,
            requesting_agent=self.agent_name,
            state=state
        )
    
    def create_decision_interface(self,
                                title: str,
                                description: str,
                                options: List[Dict[str, Any]],
                                context_data: Dict[str, Any] = None,
                                state: VideoGenerationState = None) -> Dict[str, Any]:
        """Create interactive decision interface.
        
        Args:
            title: Decision title
            description: Decision description
            options: Decision options
            context_data: Additional context data
            state: Current workflow state
            
        Returns:
            Dict: Interactive decision interface
        """
        return create_interactive_decision_interface(
            decision_title=title,
            decision_description=description,
            decision_options=options,
            context_data=context_data or {},
            requesting_agent=self.agent_name,
            state=state
        )
    
    def record_intervention(self, intervention_data: Dict[str, Any]):
        """Record an intervention for analysis.
        
        Args:
            intervention_data: Data about the intervention
        """
        intervention_record = {
            "agent": self.agent_name,
            "timestamp": datetime.now().isoformat(),
            "intervention_data": intervention_data
        }
        
        self.intervention_history.append(intervention_record)
        logger.info(f"Recorded intervention for {self.agent_name}: {intervention_data.get('type', 'unknown')}")
    
    def get_intervention_patterns(self) -> Dict[str, Any]:
        """Get patterns from intervention history.
        
        Returns:
            Dict: Intervention patterns analysis
        """
        if not self.intervention_history:
            return {"message": "No intervention history available"}
        
        patterns = {
            "total_interventions": len(self.intervention_history),
            "intervention_types": {},
            "common_triggers": {},
            "success_rate": 0.0,
            "average_response_time": 0.0
        }
        
        for record in self.intervention_history:
            intervention_type = record.get('intervention_data', {}).get('type', 'unknown')
            patterns["intervention_types"][intervention_type] = patterns["intervention_types"].get(intervention_type, 0) + 1
        
        return patterns
    
    def _has_quality_issues(self, state: VideoGenerationState) -> bool:
        """Check if there are quality issues that require intervention."""
        if not state:
            return False
        
        # Check visual errors
        visual_errors = state.get('visual_errors', {})
        total_visual_errors = sum(len(errors) for errors in visual_errors.values())
        if total_visual_errors > 2:
            return True
        
        # Check code errors
        code_errors = state.get('code_errors', {})
        if len(code_errors) > 1:
            return True
        
        # Check rendering errors
        rendering_errors = state.get('rendering_errors', {})
        if len(rendering_errors) > 1:
            return True
        
        return False
    
    def _analyze_parameter_impact(self, 
                                parameter_name: str, 
                                current_value: Any, 
                                suggested_values: List[Any],
                                state: VideoGenerationState) -> Dict[str, Any]:
        """Analyze the impact of parameter changes."""
        impact_analysis = {
            "performance_impact": "medium",
            "quality_impact": "medium",
            "resource_impact": "medium",
            "time_impact": "medium",
            "risk_level": "medium"
        }
        
        # Parameter-specific impact analysis
        if parameter_name == "max_retries":
            if any(val > current_value for val in suggested_values):
                impact_analysis["time_impact"] = "high"
                impact_analysis["quality_impact"] = "high"
        
        elif parameter_name == "default_quality":
            if any(val == "high" for val in suggested_values):
                impact_analysis["performance_impact"] = "high"
                impact_analysis["time_impact"] = "high"
                impact_analysis["quality_impact"] = "high"
        
        elif parameter_name == "max_scene_concurrency":
            if any(val > current_value for val in suggested_values):
                impact_analysis["performance_impact"] = "high"
                impact_analysis["resource_impact"] = "high"
        
        return impact_analysis


class InterventionMixin:
    """Mixin class to add intervention capabilities to agents."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intervention_manager = AgentInterventionManager(
            agent_name=getattr(self, 'name', self.__class__.__name__),
            config=getattr(self, 'config', {})
        )
    
    async def check_and_handle_intervention(self, state: VideoGenerationState) -> Optional[Command]:
        """Check if intervention is needed and handle it.
        
        Args:
            state: Current workflow state
            
        Returns:
            Optional[Command]: Intervention command if needed
        """
        intervention_rule = self.intervention_manager.check_intervention_needed(state)
        
        if intervention_rule:
            # Create appropriate intervention request based on trigger
            if intervention_rule.trigger == InterventionTrigger.ERROR_THRESHOLD:
                return await self._handle_error_threshold_intervention(state, intervention_rule)
            elif intervention_rule.trigger == InterventionTrigger.QUALITY_GATE:
                return await self._handle_quality_gate_intervention(state, intervention_rule)
            else:
                return await self._handle_generic_intervention(state, intervention_rule)
        
        return None
    
    async def _handle_error_threshold_intervention(self, state: VideoGenerationState, rule: InterventionRule) -> Command:
        """Handle error threshold intervention."""
        error_count = state.get('error_count', 0)
        escalated_errors = state.get('escalated_errors', [])
        
        error_details = {
            "error_message": f"Error threshold exceeded: {error_count} errors",
            "error_type": "threshold_exceeded",
            "blocks_workflow": True
        }
        
        recovery_strategies = [
            {
                "id": "continue_with_errors",
                "label": "Continue with current errors",
                "description": "Attempt to continue workflow despite errors",
                "success_probability": 0.3,
                "risk_level": "high"
            },
            {
                "id": "reset_workflow",
                "label": "Reset and restart workflow",
                "description": "Clear errors and restart from beginning",
                "success_probability": 0.8,
                "risk_level": "medium"
            },
            {
                "id": "adjust_parameters",
                "label": "Adjust parameters and retry",
                "description": "Modify workflow parameters to reduce errors",
                "success_probability": 0.6,
                "risk_level": "low"
            }
        ]
        
        return await self.intervention_manager.request_error_resolution(
            error_details=error_details,
            recovery_strategies=recovery_strategies,
            state=state
        )
    
    async def _handle_quality_gate_intervention(self, state: VideoGenerationState, rule: InterventionRule) -> Command:
        """Handle quality gate intervention."""
        # Extract quality metrics from state
        quality_metrics = {
            "visual_errors": sum(len(errors) for errors in state.get('visual_errors', {}).values()),
            "code_errors": len(state.get('code_errors', {})),
            "rendering_errors": len(state.get('rendering_errors', {})),
            "overall_score": max(0, 1.0 - (state.get('error_count', 0) * 0.1))
        }
        
        return await self.intervention_manager.request_quality_approval(
            quality_metrics=quality_metrics,
            content_description=f"Quality review for {state.get('topic', 'video generation')}",
            state=state
        )
    
    async def _handle_generic_intervention(self, state: VideoGenerationState, rule: InterventionRule) -> Command:
        """Handle generic intervention."""
        context = f"Intervention needed for {self.intervention_manager.agent_name}: {rule.trigger.value}"
        options = ["Continue", "Modify parameters", "Restart", "Abort"]
        
        return await self.intervention_manager.request_approval(
            context=context,
            options=options,
            priority=rule.priority,
            state=state
        )


# Utility functions for agent integration

def create_agent_intervention_manager(agent_name: str, config: Dict[str, Any] = None) -> AgentInterventionManager:
    """Create an intervention manager for an agent.
    
    Args:
        agent_name: Name of the agent
        config: Configuration for intervention behavior
        
    Returns:
        AgentInterventionManager: Configured intervention manager
    """
    return AgentInterventionManager(agent_name, config)


def add_intervention_capabilities(agent_class):
    """Decorator to add intervention capabilities to an agent class.
    
    Args:
        agent_class: Agent class to enhance
        
    Returns:
        Enhanced agent class with intervention capabilities
    """
    class EnhancedAgent(InterventionMixin, agent_class):
        pass
    
    EnhancedAgent.__name__ = agent_class.__name__
    EnhancedAgent.__qualname__ = agent_class.__qualname__
    
    return EnhancedAgent


def validate_intervention_response(response: Dict[str, Any], 
                                 expected_format: Dict[str, Any]) -> Dict[str, Any]:
    """Validate human intervention response format.
    
    Args:
        response: Human response to validate
        expected_format: Expected response format
        
    Returns:
        Dict: Validation result
    """
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "normalized_response": response.copy()
    }
    
    # Check required fields
    required_fields = expected_format.get('required_fields', [])
    for field in required_fields:
        if field not in response:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Missing required field: {field}")
    
    # Check field types
    field_types = expected_format.get('field_types', {})
    for field, expected_type in field_types.items():
        if field in response:
            actual_value = response[field]
            if not isinstance(actual_value, expected_type):
                validation_result["warnings"].append(
                    f"Field {field} expected {expected_type.__name__}, got {type(actual_value).__name__}"
                )
    
    # Normalize response
    if validation_result["valid"]:
        # Apply any normalization rules
        normalization_rules = expected_format.get('normalization_rules', {})
        for field, rule in normalization_rules.items():
            if field in response:
                if rule == "lowercase":
                    validation_result["normalized_response"][field] = str(response[field]).lower()
                elif rule == "strip":
                    validation_result["normalized_response"][field] = str(response[field]).strip()
    
    return validation_result


def create_intervention_context(agent_name: str, 
                              intervention_type: str,
                              state: VideoGenerationState,
                              additional_context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create comprehensive intervention context.
    
    Args:
        agent_name: Name of the requesting agent
        intervention_type: Type of intervention
        state: Current workflow state
        additional_context: Additional context data
        
    Returns:
        Dict: Comprehensive intervention context
    """
    context = {
        "agent_name": agent_name,
        "intervention_type": intervention_type,
        "timestamp": datetime.now().isoformat(),
        "workflow_state": {
            "topic": state.get('topic', '') if state else '',
            "session_id": state.get('session_id', '') if state else '',
            "current_agent": state.get('current_agent', '') if state else '',
            "error_count": state.get('error_count', 0) if state else 0,
            "workflow_interrupted": state.get('workflow_interrupted', False) if state else False
        },
        "agent_context": {
            "retry_count": state.get('retry_count', {}).get(agent_name, 0) if state else 0,
            "performance_metrics": state.get('performance_metrics', {}).get(agent_name, {}) if state else {}
        },
        "additional_context": additional_context or {}
    }
    
    return context
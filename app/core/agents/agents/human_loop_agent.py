"""
HumanLoopAgent for human intervention and feedback.
Handles situations requiring human input or decision-making.
"""

from typing import Dict, Any, List, Optional
from langgraph.types import Command
import logging

from ..base_agent import BaseAgent
from ..state import VideoGenerationState

logger = logging.getLogger(__name__)


class HumanLoopAgent(BaseAgent):
    """Agent responsible for handling human intervention requests."""
    
    def __init__(self, config, system_config):
        """Initialize HumanLoopAgent.
        
        Args:
            config: Agent configuration
            system_config: System configuration
        """
        super().__init__(config, system_config)
        logger.info(f"HumanLoopAgent initialized: {config.name}")
    
    async def execute(self, state: VideoGenerationState) -> Command:
        """Execute human intervention handling.
        
        Args:
            state: Current workflow state
            
        Returns:
            Command: LangGraph command for next action
        """
        try:
            self.log_agent_action("start_human_intervention", {
                'pending_input': state.get('pending_human_input') is not None
            })
            
            pending_input = state.get('pending_human_input')
            if not pending_input:
                # No human input needed, continue to monitoring
                return Command(
                    goto="monitoring_agent",
                    update={"current_agent": self.name}
                )
            
            # Process human intervention request
            intervention_result = await self._process_human_intervention(pending_input, state)
            
            self.log_agent_action("human_intervention_complete", {
                'intervention_type': pending_input.get('type', 'unknown'),
                'decision': intervention_result.get('decision', 'unknown')
            })
            
            # Determine next action based on human decision
            next_agent = self._determine_next_agent(intervention_result, state)
            
            return Command(
                goto=next_agent,
                update={
                    "human_feedback": intervention_result,
                    "pending_human_input": None,
                    "current_agent": self.name
                }
            )
            
        except Exception as e:
            logger.error(f"Human intervention failed: {e}")
            return await self.handle_error(e, state)
    
    async def _process_human_intervention(
        self, 
        pending_input: Dict[str, Any], 
        state: VideoGenerationState
    ) -> Dict[str, Any]:
        """Process human intervention request.
        
        Args:
            pending_input: Human intervention request details
            state: Current workflow state
            
        Returns:
            Dict[str, Any]: Human intervention result
        """
        intervention_type = pending_input.get('type', 'unknown')
        options = pending_input.get('options', [])
        
        # In a real implementation, this would:
        # 1. Present the intervention request to a human operator
        # 2. Wait for human response
        # 3. Validate the response
        # 4. Return the decision
        
        # For simulation purposes, we'll make automatic decisions
        if intervention_type == 'error_resolution':
            # Simulate human choosing to retry
            decision = 'retry' if 'retry' in options else options[0] if options else 'continue'
        elif intervention_type == 'quality_review':
            # Simulate human approving quality
            decision = 'approve' if 'approve' in options else options[0] if options else 'continue'
        elif intervention_type == 'critical_error':
            # Simulate human choosing manual intervention
            decision = 'manual_intervention' if 'manual_intervention' in options else 'abort'
        else:
            # Default decision
            decision = options[0] if options else 'continue'
        
        return {
            'intervention_type': intervention_type,
            'decision': decision,
            'timestamp': '2024-01-01T12:00:00Z',
            'options_presented': options,
            'additional_feedback': f"Human decided to {decision} for {intervention_type}"
        }
    
    def _determine_next_agent(
        self, 
        intervention_result: Dict[str, Any], 
        state: VideoGenerationState
    ) -> str:
        """Determine the next agent based on human intervention result.
        
        Args:
            intervention_result: Result of human intervention
            state: Current workflow state
            
        Returns:
            str: Name of next agent to execute
        """
        decision = intervention_result.get('decision', 'continue')
        intervention_type = intervention_result.get('intervention_type', 'unknown')
        
        if decision == 'abort':
            return "END"
        elif decision == 'retry':
            # Return to the agent that had the error
            escalated_errors = state.get('escalated_errors', [])
            if escalated_errors:
                failed_agent = escalated_errors[-1].get('agent', 'planner_agent')
                return failed_agent
            return "planner_agent"
        elif decision == 'skip':
            return "monitoring_agent"
        elif decision == 'restart':
            return "planner_agent"
        else:
            # Default: continue to monitoring
            return "monitoring_agent"
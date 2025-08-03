"""
ErrorHandlerAgent for error recovery and handling.
Manages error recovery strategies and escalation.
"""

from typing import Dict, Any, List, Optional
from langgraph.types import Command
import logging

from ..base_agent import BaseAgent
from ..state import VideoGenerationState

logger = logging.getLogger(__name__)


class ErrorHandlerAgent(BaseAgent):
    """Agent responsible for handling errors and implementing recovery strategies."""
    
    def __init__(self, config, system_config):
        """Initialize ErrorHandlerAgent.
        
        Args:
            config: Agent configuration
            system_config: System configuration
        """
        super().__init__(config, system_config)
        logger.info(f"ErrorHandlerAgent initialized: {config.name}")
    
    async def execute(self, state: VideoGenerationState) -> Command:
        """Execute error handling and recovery.
        
        Args:
            state: Current workflow state
            
        Returns:
            Command: LangGraph command for next action
        """
        try:
            self.log_agent_action("start_error_handling", {
                'error_count': state.get('error_count', 0),
                'escalated_errors': len(state.get('escalated_errors', []))
            })
            
            escalated_errors = state.get('escalated_errors', [])
            error_count = state.get('error_count', 0)
            
            if not escalated_errors:
                # No errors to handle
                return Command(
                    goto="monitoring_agent",
                    update={"current_agent": self.name}
                )
            
            # Analyze the most recent error
            latest_error = escalated_errors[-1]
            recovery_strategy = await self._determine_recovery_strategy(latest_error, state)
            
            # Execute recovery strategy
            recovery_result = await self._execute_recovery_strategy(recovery_strategy, state)
            
            self.log_agent_action("error_handling_complete", {
                'recovery_strategy': recovery_strategy['type'],
                'success': recovery_result['success']
            })
            
            if recovery_result['success']:
                # Recovery successful, continue workflow
                next_agent = recovery_result.get('next_agent', 'monitoring_agent')
                return Command(
                    goto=next_agent,
                    update={
                        "current_agent": self.name,
                        "error_count": max(0, error_count - 1)  # Reduce error count
                    }
                )
            else:
                # Recovery failed, escalate to human intervention
                return Command(
                    goto="human_loop_agent",
                    update={
                        "current_agent": self.name,
                        "pending_human_input": {
                            "type": "error_resolution",
                            "error": latest_error,
                            "recovery_attempted": recovery_strategy,
                            "options": ["retry", "skip", "abort"]
                        }
                    }
                )
            
        except Exception as e:
            logger.error(f"Error handling failed: {e}")
            # If error handler fails, escalate to human intervention
            return Command(
                goto="human_loop_agent",
                update={
                    "current_agent": self.name,
                    "pending_human_input": {
                        "type": "critical_error",
                        "error": str(e),
                        "options": ["abort", "manual_intervention"]
                    }
                }
            )
    
    async def _determine_recovery_strategy(
        self, 
        error: Dict[str, Any], 
        state: VideoGenerationState
    ) -> Dict[str, Any]:
        """Determine the appropriate recovery strategy for an error.
        
        Args:
            error: Error information
            state: Current workflow state
            
        Returns:
            Dict[str, Any]: Recovery strategy information
        """
        error_agent = error.get('agent', 'unknown')
        error_message = error.get('error', '')
        retry_count = error.get('retry_count', 0)
        
        # Determine strategy based on error type and context
        if retry_count < self.max_retries:
            if 'timeout' in error_message.lower():
                return {
                    'type': 'retry_with_timeout_increase',
                    'agent': error_agent,
                    'timeout_multiplier': 1.5
                }
            elif 'model' in error_message.lower() or 'api' in error_message.lower():
                return {
                    'type': 'retry_with_fallback_model',
                    'agent': error_agent,
                    'use_fallback': True
                }
            else:
                return {
                    'type': 'simple_retry',
                    'agent': error_agent
                }
        else:
            return {
                'type': 'escalate_to_human',
                'agent': error_agent,
                'reason': 'max_retries_exceeded'
            }
    
    async def _execute_recovery_strategy(
        self, 
        strategy: Dict[str, Any], 
        state: VideoGenerationState
    ) -> Dict[str, Any]:
        """Execute the determined recovery strategy.
        
        Args:
            strategy: Recovery strategy to execute
            state: Current workflow state
            
        Returns:
            Dict[str, Any]: Recovery execution result
        """
        strategy_type = strategy['type']
        
        if strategy_type == 'simple_retry':
            return {
                'success': True,
                'next_agent': strategy['agent'],
                'message': f"Retrying {strategy['agent']}"
            }
        
        elif strategy_type == 'retry_with_timeout_increase':
            return {
                'success': True,
                'next_agent': strategy['agent'],
                'message': f"Retrying {strategy['agent']} with increased timeout",
                'config_updates': {
                    'timeout_multiplier': strategy.get('timeout_multiplier', 1.5)
                }
            }
        
        elif strategy_type == 'retry_with_fallback_model':
            return {
                'success': True,
                'next_agent': strategy['agent'],
                'message': f"Retrying {strategy['agent']} with fallback model",
                'config_updates': {
                    'use_fallback_model': True
                }
            }
        
        elif strategy_type == 'escalate_to_human':
            return {
                'success': False,
                'message': f"Escalating to human intervention: {strategy.get('reason', 'unknown')}"
            }
        
        else:
            return {
                'success': False,
                'message': f"Unknown recovery strategy: {strategy_type}"
            }
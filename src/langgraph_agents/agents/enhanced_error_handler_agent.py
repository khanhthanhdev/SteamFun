"""
Enhanced ErrorHandlerAgent with advanced error recovery capabilities.

This agent integrates sophisticated error pattern recognition, automatic recovery
workflow selection, escalation threshold management, and error analytics.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from langgraph.types import Command

from ..base_agent import BaseAgent
from ..state import VideoGenerationState, AgentError
from ..error_recovery import (
    AdvancedErrorRecoverySystem,
    EscalationThresholdManager,
    ErrorAnalyticsSystem,
    ErrorPattern,
    RecoveryStrategy
)

logger = logging.getLogger(__name__)


class EnhancedErrorHandlerAgent(BaseAgent):
    """Enhanced error handler agent with advanced recovery capabilities."""
    
    def __init__(self, config: Dict[str, Any], system_config: Dict[str, Any]):
        """Initialize the enhanced error handler agent.
        
        Args:
            config: Agent-specific configuration
            system_config: System-wide configuration
        """
        super().__init__(config, system_config)
        
        # Initialize advanced recovery system
        recovery_config = config.get('advanced_recovery', {})
        self.recovery_system = AdvancedErrorRecoverySystem(recovery_config)
        
        # Initialize escalation threshold manager
        escalation_config = config.get('escalation_management', {})
        self.escalation_manager = EscalationThresholdManager(escalation_config)
        
        # Initialize analytics system
        analytics_config = config.get('analytics', {})
        self.analytics_system = ErrorAnalyticsSystem(analytics_config)
        
        # Configuration settings
        self.enable_advanced_recovery = config.get('enable_advanced_recovery', True)
        self.enable_escalation_management = config.get('enable_escalation_management', True)
        self.enable_analytics = config.get('enable_analytics', True)
        self.health_check_interval = config.get('health_check_interval_seconds', 300)
        
        # Background tasks
        self._background_tasks = []
        self._last_health_check = datetime.now()
        
        self.log_agent_action("initialized_enhanced_error_handler", {
            "advanced_recovery_enabled": self.enable_advanced_recovery,
            "escalation_management_enabled": self.enable_escalation_management,
            "analytics_enabled": self.enable_analytics
        })
    
    async def execute(self, state: VideoGenerationState) -> Command:
        """Execute enhanced error handling and recovery operations.
        
        Args:
            state: Current workflow state
            
        Returns:
            Command for next workflow step
        """
        self.log_agent_action("starting_enhanced_error_handling", {
            "error_count": state.get('error_count', 0),
            "escalated_errors": len(state.get('escalated_errors', [])),
            "retry_counts": state.get('retry_count', {})
        })
        
        try:
            # Run periodic health checks
            await self._run_periodic_health_check(state)
            
            # Get current errors from state
            current_errors = await self._extract_current_errors(state)
            
            if not current_errors:
                # No errors to handle, check if workflow should continue
                return await self._handle_no_errors(state)
            
            # Process each error with advanced recovery
            recovery_commands = []
            for error in current_errors:
                command = await self._process_error_with_advanced_recovery(error, state)
                if command:
                    recovery_commands.append(command)
            
            # Determine the best recovery command to execute
            if recovery_commands:
                return await self._select_best_recovery_command(recovery_commands, state)
            else:
                # No specific recovery needed, continue workflow
                return await self._continue_workflow(state)
                
        except Exception as e:
            logger.error(f"Error in enhanced error handler execution: {e}")
            return await self._handle_handler_failure(e, state)
    
    async def _extract_current_errors(self, state: VideoGenerationState) -> List[AgentError]:
        """Extract current errors from the workflow state."""
        errors = []
        
        # Extract errors from escalated_errors
        escalated_errors = state.get('escalated_errors', [])
        for error_data in escalated_errors:
            if isinstance(error_data, dict):
                error = AgentError(
                    agent_name=error_data.get('agent', 'unknown'),
                    error_type=error_data.get('type', 'unknown'),
                    error_message=error_data.get('error', 'Unknown error'),
                    timestamp=datetime.now(),
                    retry_count=error_data.get('retry_count', 0),
                    context=error_data.get('context', {})
                )
                errors.append(error)
        
        # Extract errors from specific error fields
        code_errors = state.get('code_errors', {})
        for scene_number, error_msg in code_errors.items():
            error = AgentError(
                agent_name='code_generator_agent',
                error_type='code_generation',
                error_message=str(error_msg),
                timestamp=datetime.now(),
                retry_count=state.get('retry_count', {}).get('code_generator_agent', 0),
                context={'scene_number': scene_number}
            )
            errors.append(error)
        
        rendering_errors = state.get('rendering_errors', {})
        for scene_number, error_msg in rendering_errors.items():
            error = AgentError(
                agent_name='renderer_agent',
                error_type='rendering',
                error_message=str(error_msg),
                timestamp=datetime.now(),
                retry_count=state.get('retry_count', {}).get('renderer_agent', 0),
                context={'scene_number': scene_number}
            )
            errors.append(error)
        
        visual_errors = state.get('visual_errors', {})
        for scene_number, error_list in visual_errors.items():
            for error_msg in error_list:
                error = AgentError(
                    agent_name='visual_analysis_agent',
                    error_type='visual_analysis',
                    error_message=str(error_msg),
                    timestamp=datetime.now(),
                    retry_count=state.get('retry_count', {}).get('visual_analysis_agent', 0),
                    context={'scene_number': scene_number}
                )
                errors.append(error)
        
        return errors
    
    async def _process_error_with_advanced_recovery(
        self, 
        error: AgentError, 
        state: VideoGenerationState
    ) -> Optional[Command]:
        """Process an error using the advanced recovery system."""
        
        if not self.enable_advanced_recovery:
            # Fall back to basic error handling
            return await self._basic_error_handling(error, state)
        
        try:
            # Analyze the error with advanced pattern recognition
            error_analysis = await self.recovery_system.analyze_error(error, state)
            
            self.log_agent_action("error_analysis_completed", {
                "error_id": error_analysis.error_id,
                "pattern": error_analysis.pattern_type.value,
                "confidence": error_analysis.confidence_score,
                "strategy": error_analysis.recommended_strategy.value
            })
            
            # Execute the recommended recovery strategy
            recovery_execution = await self.recovery_system.execute_recovery(error_analysis, state)
            
            # Convert recovery execution to workflow command
            return await self._convert_recovery_to_command(recovery_execution, error_analysis, state)
            
        except Exception as e:
            logger.error(f"Advanced recovery failed for error {error.agent_name}: {str(e)}")
            # Fall back to basic error handling
            return await self._basic_error_handling(error, state)
    
    async def _convert_recovery_to_command(
        self,
        recovery_execution,
        error_analysis,
        state: VideoGenerationState
    ) -> Command:
        """Convert recovery execution results to a workflow command."""
        
        if recovery_execution.success:
            # Recovery was successful, determine next step
            strategy = recovery_execution.strategy
            
            if strategy == RecoveryStrategy.IMMEDIATE_RETRY:
                # Retry the failed operation immediately
                return await self._create_retry_command(error_analysis, state)
            
            elif strategy == RecoveryStrategy.EXPONENTIAL_BACKOFF:
                # Retry with delay (delay was already applied)
                return await self._create_retry_command(error_analysis, state)
            
            elif strategy == RecoveryStrategy.FALLBACK_PROVIDER:
                # Continue with fallback configuration
                return await self._create_continue_command(state, "fallback_provider_activated")
            
            elif strategy == RecoveryStrategy.RESOURCE_SCALING:
                # Continue with scaled resources
                return await self._create_continue_command(state, "resource_scaling_applied")
            
            elif strategy == RecoveryStrategy.CONFIGURATION_RESET:
                # Continue with reset configuration
                return await self._create_continue_command(state, "configuration_reset_applied")
            
            elif strategy == RecoveryStrategy.DEPENDENCY_BYPASS:
                # Continue with bypassed dependencies
                return await self._create_continue_command(state, "dependency_bypass_applied")
            
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                # Continue with degraded functionality
                return await self._create_continue_command(state, "graceful_degradation_applied")
            
            else:
                # Default: continue workflow
                return await self._create_continue_command(state, "recovery_completed")
        
        else:
            # Recovery failed
            if recovery_execution.strategy == RecoveryStrategy.HUMAN_ESCALATION:
                # Human escalation was requested
                return Command(
                    goto="human_loop_agent",
                    update={
                        "current_agent": "human_loop_agent",
                        "recovery_execution_id": recovery_execution.recovery_id
                    }
                )
            
            elif recovery_execution.strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                # Circuit breaker was opened, skip this operation
                return await self._create_skip_command(error_analysis, state)
            
            else:
                # Recovery failed, escalate to human
                return Command(
                    goto="human_loop_agent",
                    update={
                        "pending_human_input": {
                            "context": f"Recovery failed: {recovery_execution.strategy.value}",
                            "error_analysis": error_analysis.error_id,
                            "recovery_execution": recovery_execution.recovery_id,
                            "options": ["retry_recovery", "skip_operation", "abort_workflow"],
                            "requesting_agent": self.name
                        },
                        "current_agent": "human_loop_agent"
                    }
                )
    
    async def _create_retry_command(self, error_analysis, state: VideoGenerationState) -> Command:
        """Create a command to retry the failed operation."""
        
        # Determine which agent to retry based on the error
        affected_components = error_analysis.affected_components
        if not affected_components:
            # Default to continuing workflow
            return await self._create_continue_command(state, "retry_requested")
        
        primary_agent = affected_components[0]
        
        # Clear the specific error for retry
        updates = {
            "current_agent": primary_agent,
            "error_recovery_attempt": True,
            "recovery_strategy": error_analysis.recommended_strategy.value
        }
        
        # Clear specific error fields based on agent
        if primary_agent == 'code_generator_agent':
            scene_number = error_analysis.context.get('scene_number')
            if scene_number is not None:
                code_errors = state.get('code_errors', {}).copy()
                code_errors.pop(scene_number, None)
                updates['code_errors'] = code_errors
        
        elif primary_agent == 'renderer_agent':
            scene_number = error_analysis.context.get('scene_number')
            if scene_number is not None:
                rendering_errors = state.get('rendering_errors', {}).copy()
                rendering_errors.pop(scene_number, None)
                updates['rendering_errors'] = rendering_errors
        
        elif primary_agent == 'visual_analysis_agent':
            scene_number = error_analysis.context.get('scene_number')
            if scene_number is not None:
                visual_errors = state.get('visual_errors', {}).copy()
                visual_errors.pop(scene_number, None)
                updates['visual_errors'] = visual_errors
        
        return Command(goto=primary_agent, update=updates)
    
    async def _create_continue_command(self, state: VideoGenerationState, reason: str) -> Command:
        """Create a command to continue the workflow."""
        
        # Determine next appropriate step in workflow
        next_agent = self._determine_next_workflow_step(state)
        
        return Command(
            goto=next_agent,
            update={
                "current_agent": next_agent,
                "error_recovery_reason": reason,
                "error_count": 0,  # Reset error count
                "escalated_errors": []  # Clear escalated errors
            }
        )
    
    async def _create_skip_command(self, error_analysis, state: VideoGenerationState) -> Command:
        """Create a command to skip the failed operation."""
        
        # Mark the operation as skipped and continue
        scene_number = error_analysis.context.get('scene_number')
        
        updates = {
            "skipped_operations": state.get('skipped_operations', []) + [
                {
                    "agent": error_analysis.affected_components[0] if error_analysis.affected_components else "unknown",
                    "scene_number": scene_number,
                    "reason": "circuit_breaker_open",
                    "timestamp": datetime.now().isoformat()
                }
            ]
        }
        
        return await self._create_continue_command(state, "operation_skipped")
    
    async def _basic_error_handling(self, error: AgentError, state: VideoGenerationState) -> Command:
        """Basic error handling fallback when advanced recovery is disabled."""
        
        # Simple retry logic
        if error.retry_count < 3:
            # Retry the operation
            return Command(
                goto=error.agent_name,
                update={
                    "current_agent": error.agent_name,
                    "retry_count": {
                        **state.get('retry_count', {}),
                        error.agent_name: error.retry_count + 1
                    }
                }
            )
        else:
            # Escalate to human
            return Command(
                goto="human_loop_agent",
                update={
                    "pending_human_input": {
                        "context": f"Basic error handling: {error.error_message}",
                        "agent": error.agent_name,
                        "options": ["retry", "skip", "abort"],
                        "requesting_agent": self.name
                    },
                    "current_agent": "human_loop_agent"
                }
            )
    
    async def _select_best_recovery_command(
        self, 
        commands: List[Command], 
        state: VideoGenerationState
    ) -> Command:
        """Select the best recovery command from multiple options."""
        
        if len(commands) == 1:
            return commands[0]
        
        # Priority order for command selection
        priority_order = [
            "human_loop_agent",  # Human escalation has highest priority
            "planner_agent",     # Restart from planning
            "code_generator_agent",  # Code generation retry
            "renderer_agent",    # Rendering retry
            "visual_analysis_agent",  # Visual analysis retry
            "END"               # End workflow
        ]
        
        # Select command based on priority
        for target in priority_order:
            for command in commands:
                if command.goto == target:
                    return command
        
        # Default to first command
        return commands[0]
    
    async def _handle_no_errors(self, state: VideoGenerationState) -> Command:
        """Handle the case when there are no current errors."""
        
        # Check if workflow is complete
        if state.get('workflow_complete', False):
            return Command(goto="END", update={"current_agent": None})
        
        # Continue to next workflow step
        next_agent = self._determine_next_workflow_step(state)
        return Command(
            goto=next_agent,
            update={
                "current_agent": next_agent,
                "error_count": 0
            }
        )
    
    async def _continue_workflow(self, state: VideoGenerationState) -> Command:
        """Continue the workflow to the next appropriate step."""
        
        next_agent = self._determine_next_workflow_step(state)
        
        return Command(
            goto=next_agent,
            update={
                "current_agent": next_agent,
                "error_recovery_completed": True
            }
        )
    
    async def _handle_handler_failure(self, exception: Exception, state: VideoGenerationState) -> Command:
        """Handle failure of the error handler itself."""
        
        logger.critical(f"Error handler failure: {str(exception)}")
        
        # Escalate to human immediately
        return Command(
            goto="human_loop_agent",
            update={
                "pending_human_input": {
                    "context": f"Error handler failure: {str(exception)}",
                    "error_type": "handler_failure",
                    "options": ["restart_error_handler", "abort_workflow"],
                    "requesting_agent": self.name,
                    "critical": True
                },
                "current_agent": "human_loop_agent"
            }
        )
    
    def _determine_next_workflow_step(self, state: VideoGenerationState) -> str:
        """Determine the next workflow step after error handling."""
        
        # Check workflow progress
        if not state.get('scene_outline'):
            return "planner_agent"
        
        scene_implementations = state.get('scene_implementations', {})
        generated_code = state.get('generated_code', {})
        rendered_videos = state.get('rendered_videos', {})
        
        # Check if we need to generate more code
        for scene_number in scene_implementations:
            if scene_number not in generated_code:
                return "code_generator_agent"
        
        # Check if we need to render more videos
        for scene_number in generated_code:
            if scene_number not in rendered_videos:
                return "renderer_agent"
        
        # Check if visual analysis is needed
        if state.get('use_visual_fix_code', False) and rendered_videos:
            return "visual_analysis_agent"
        
        # Default to ending workflow
        return "END"
    
    async def _run_periodic_health_check(self, state: VideoGenerationState):
        """Run periodic health checks and analytics."""
        
        now = datetime.now()
        if (now - self._last_health_check).total_seconds() < self.health_check_interval:
            return
        
        self._last_health_check = now
        
        try:
            # Run escalation threshold evaluation if enabled
            if self.enable_escalation_management:
                await self._evaluate_escalation_thresholds(state)
            
            # Calculate system health metrics if analytics enabled
            if self.enable_analytics:
                await self._calculate_system_health(state)
                
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
    
    async def _evaluate_escalation_thresholds(self, state: VideoGenerationState):
        """Evaluate escalation thresholds and trigger alerts if needed."""
        
        # Get recent error and recovery history
        error_history = getattr(self.recovery_system, 'error_history', [])
        recovery_history = getattr(self.recovery_system, 'recovery_history', [])
        circuit_breaker_states = getattr(self.recovery_system, 'circuit_breakers', {})
        
        # Convert circuit breaker states to simple format
        cb_states = {agent: cb.get('state', 'closed') for agent, cb in circuit_breaker_states.items()}
        
        # Evaluate thresholds
        triggered_events = await self.escalation_manager.evaluate_escalation(
            error_history, recovery_history, cb_states
        )
        
        # Log triggered escalations
        for event in triggered_events:
            self.log_agent_action("escalation_triggered", {
                "event_id": event.event_id,
                "level": event.level.value,
                "threshold": event.threshold_name,
                "value": event.trigger_value
            })
    
    async def _calculate_system_health(self, state: VideoGenerationState):
        """Calculate and log system health metrics."""
        
        # Get history data
        error_history = getattr(self.recovery_system, 'error_history', [])
        recovery_history = getattr(self.recovery_system, 'recovery_history', [])
        escalation_history = getattr(self.escalation_manager, 'escalation_history', [])
        circuit_breaker_states = getattr(self.recovery_system, 'circuit_breakers', {})
        
        # Convert circuit breaker states
        cb_states = {agent: cb.get('state', 'closed') for agent, cb in circuit_breaker_states.items()}
        
        # Calculate health metrics
        health_metrics = await self.analytics_system.calculate_system_health(
            error_history, recovery_history, escalation_history, cb_states
        )
        
        # Log health metrics
        self.log_agent_action("system_health_calculated", {
            "health_score": health_metrics.overall_health_score,
            "error_rate": health_metrics.error_rate,
            "recovery_success_rate": health_metrics.recovery_success_rate,
            "trending": health_metrics.trending
        })
    
    async def generate_analytics_report(self, time_period_hours: int = 24) -> str:
        """Generate a comprehensive analytics report.
        
        Args:
            time_period_hours: Time period for the report
            
        Returns:
            Report ID of the generated report
        """
        
        if not self.enable_analytics:
            raise ValueError("Analytics is not enabled")
        
        # Get history data
        error_history = getattr(self.recovery_system, 'error_history', [])
        recovery_history = getattr(self.recovery_system, 'recovery_history', [])
        escalation_history = getattr(self.escalation_manager, 'escalation_history', [])
        
        # Generate comprehensive report
        report = await self.analytics_system.generate_comprehensive_report(
            error_history, recovery_history, escalation_history, time_period_hours
        )
        
        self.log_agent_action("analytics_report_generated", {
            "report_id": report.report_id,
            "time_period": report.time_period,
            "total_errors": report.summary.get('total_errors', 0),
            "total_recoveries": report.summary.get('total_recoveries', 0)
        })
        
        return report.report_id
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and health metrics."""
        
        status = {
            "agent_name": self.name,
            "advanced_recovery_enabled": self.enable_advanced_recovery,
            "escalation_management_enabled": self.enable_escalation_management,
            "analytics_enabled": self.enable_analytics,
            "last_health_check": self._last_health_check.isoformat()
        }
        
        # Add recovery system status
        if self.enable_advanced_recovery:
            status["recovery_system"] = {
                "error_history_size": len(getattr(self.recovery_system, 'error_history', [])),
                "recovery_history_size": len(getattr(self.recovery_system, 'recovery_history', [])),
                "circuit_breakers": getattr(self.recovery_system, 'circuit_breakers', {})
            }
        
        # Add escalation manager status
        if self.enable_escalation_management:
            status["escalation_manager"] = self.escalation_manager.get_escalation_summary()
        
        # Add analytics status
        if self.enable_analytics:
            status["analytics"] = self.analytics_system.get_analytics_summary()
        
        return status
"""
ErrorHandlerAgent for centralized error management.
Implements error classification, routing, retry strategies, and escalation logic.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from langgraph.types import Command

from ..base_agent import BaseAgent
from ..state import VideoGenerationState, AgentError, RecoveryStrategy

logger = logging.getLogger(__name__)


class ErrorHandlerAgent(BaseAgent):
    """Agent for centralized error management and recovery.
    
    Implements error classification and routing system, retry strategies and escalation logic,
    error pattern recognition and recovery workflows, and integration with all other agents.
    """
    
    def __init__(self, config, system_config):
        """Initialize ErrorHandlerAgent with error patterns and recovery strategies."""
        super().__init__(config, system_config)
        
        # Error classification patterns
        self.error_patterns = self._initialize_error_patterns()
        
        # Recovery strategies
        self.recovery_strategies = self._initialize_recovery_strategies()
        
        # Error history for pattern recognition
        self.error_history = []
        
        # Maximum retry attempts before escalation
        self.max_global_retries = system_config.workflow_config.max_workflow_retries
        
        self.log_agent_action("initialized_error_handler", {
            "error_patterns": len(self.error_patterns),
            "recovery_strategies": len(self.recovery_strategies),
            "max_global_retries": self.max_global_retries
        })
    
    async def execute(self, state: VideoGenerationState) -> Command:
        """Execute error handling and recovery operations.
        
        Args:
            state: Current workflow state
            
        Returns:
            Command: Next action based on error analysis and recovery strategy
        """
        self.log_agent_action("starting_error_handling", {
            "error_count": state.get('error_count', 0),
            "escalated_errors": len(state.get('escalated_errors', [])),
            "retry_counts": state.get('retry_count', {})
        })
        
        try:
            # Analyze current error situation
            error_analysis = await self._analyze_error_situation(state)
            
            # Determine recovery strategy
            recovery_action = await self._determine_recovery_strategy(error_analysis, state)
            
            # Execute recovery action
            return await self._execute_recovery_action(recovery_action, state)
            
        except Exception as e:
            logger.error(f"Error in error handler execution: {e}")
            # If error handler itself fails, escalate to human
            return Command(
                goto="human_loop_agent",
                update={
                    "pending_human_input": {
                        "context": f"Error handler failed: {e}",
                        "options": ["retry_workflow", "abort_workflow"],
                        "requesting_agent": self.name
                    },
                    "current_agent": "human_loop_agent"
                }
            )
    
    async def _analyze_error_situation(self, state: VideoGenerationState) -> Dict[str, Any]:
        """Analyze the current error situation to determine appropriate response.
        
        Args:
            state: Current workflow state
            
        Returns:
            Error analysis results
        """
        escalated_errors = state.get('escalated_errors', [])
        error_count = state.get('error_count', 0)
        retry_count = state.get('retry_count', {})
        code_errors = state.get('code_errors', {})
        rendering_errors = state.get('rendering_errors', {})
        visual_errors = state.get('visual_errors', {})
        
        # Classify errors by type and agent
        error_classification = {
            'code_generation_errors': [],
            'rendering_errors': [],
            'visual_analysis_errors': [],
            'rag_errors': [],
            'system_errors': [],
            'unknown_errors': []
        }
        
        # Analyze escalated errors
        for error in escalated_errors:
            error_type = self._classify_error_type(error)
            error_classification[error_type].append(error)
        
        # Analyze code errors
        for scene_number, error_msg in code_errors.items():
            error_classification['code_generation_errors'].append({
                'agent': 'code_generator_agent',
                'scene_number': scene_number,
                'error': error_msg,
                'type': 'code_generation'
            })
        
        # Analyze rendering errors
        for scene_number, error_msg in rendering_errors.items():
            error_classification['rendering_errors'].append({
                'agent': 'renderer_agent',
                'scene_number': scene_number,
                'error': error_msg,
                'type': 'rendering'
            })
        
        # Analyze visual errors
        for scene_number, error_list in visual_errors.items():
            for error_msg in error_list:
                error_classification['visual_analysis_errors'].append({
                    'agent': 'visual_analysis_agent',
                    'scene_number': scene_number,
                    'error': error_msg,
                    'type': 'visual_analysis'
                })
        
        # Calculate error severity
        severity_score = self._calculate_error_severity(error_classification, error_count, retry_count)
        
        # Detect error patterns
        error_patterns = self._detect_error_patterns(error_classification)
        
        analysis = {
            'total_errors': error_count,
            'error_classification': error_classification,
            'severity_score': severity_score,
            'retry_counts': retry_count,
            'error_patterns': error_patterns,
            'requires_human_intervention': self._requires_human_intervention(state, severity_score),
            'recoverable_errors': self._identify_recoverable_errors(error_classification),
            'critical_errors': self._identify_critical_errors(error_classification)
        }
        
        self.log_agent_action("error_analysis_completed", {
            "total_errors": analysis['total_errors'],
            "severity_score": analysis['severity_score'],
            "requires_human": analysis['requires_human_intervention'],
            "recoverable_count": len(analysis['recoverable_errors']),
            "critical_count": len(analysis['critical_errors'])
        })
        
        return analysis
    
    async def _determine_recovery_strategy(self, error_analysis: Dict[str, Any], state: VideoGenerationState) -> Dict[str, Any]:
        """Determine the appropriate recovery strategy based on error analysis.
        
        Args:
            error_analysis: Results from error analysis
            state: Current workflow state
            
        Returns:
            Recovery strategy action plan
        """
        # Check if human intervention is required
        if error_analysis['requires_human_intervention']:
            return {
                'action': 'escalate_to_human',
                'reason': 'Error severity or count exceeds thresholds',
                'context': error_analysis
            }
        
        # Check for critical errors that cannot be recovered
        if error_analysis['critical_errors']:
            return {
                'action': 'escalate_to_human',
                'reason': 'Critical errors detected',
                'context': error_analysis['critical_errors']
            }
        
        # Try to recover from recoverable errors
        if error_analysis['recoverable_errors']:
            recovery_plan = self._create_recovery_plan(error_analysis['recoverable_errors'], state)
            return {
                'action': 'execute_recovery',
                'recovery_plan': recovery_plan,
                'context': error_analysis
            }
        
        # If no specific errors to handle, check workflow state
        if state.get('workflow_complete', False):
            return {
                'action': 'complete_workflow',
                'reason': 'No errors to handle and workflow is complete'
            }
        
        # Default: continue workflow
        return {
            'action': 'continue_workflow',
            'reason': 'No critical errors detected, continuing workflow'
        }
    
    async def _execute_recovery_action(self, recovery_action: Dict[str, Any], state: VideoGenerationState) -> Command:
        """Execute the determined recovery action.
        
        Args:
            recovery_action: Recovery action plan
            state: Current workflow state
            
        Returns:
            Command for next workflow step
        """
        action = recovery_action['action']
        
        if action == 'escalate_to_human':
            return await self._escalate_to_human(recovery_action, state)
        elif action == 'execute_recovery':
            return await self._execute_recovery_plan(recovery_action, state)
        elif action == 'complete_workflow':
            return await self._complete_workflow_with_errors(state)
        elif action == 'continue_workflow':
            return await self._continue_workflow(state)
        else:
            # Unknown action, escalate to human
            return await self._escalate_to_human({
                'reason': f'Unknown recovery action: {action}',
                'context': recovery_action
            }, state)
    
    async def _escalate_to_human(self, recovery_action: Dict[str, Any], state: VideoGenerationState) -> Command:
        """Escalate error handling to human intervention.
        
        Args:
            recovery_action: Recovery action details
            state: Current workflow state
            
        Returns:
            Command to route to human loop agent
        """
        context = f"Error handling escalation: {recovery_action.get('reason', 'Unknown reason')}"
        
        # Prepare error summary for human
        error_summary = self._prepare_error_summary(state)
        
        options = [
            "retry_failed_operations",
            "skip_failed_scenes",
            "restart_workflow",
            "abort_workflow"
        ]
        
        self.log_agent_action("escalating_to_human", {
            "reason": recovery_action.get('reason', ''),
            "error_count": state.get('error_count', 0)
        })
        
        return Command(
            goto="human_loop_agent",
            update={
                "pending_human_input": {
                    "context": context,
                    "error_summary": error_summary,
                    "options": options,
                    "requesting_agent": self.name,
                    "timestamp": datetime.now().isoformat()
                },
                "current_agent": "human_loop_agent"
            }
        )
    
    async def _execute_recovery_plan(self, recovery_action: Dict[str, Any], state: VideoGenerationState) -> Command:
        """Execute a specific recovery plan.
        
        Args:
            recovery_action: Recovery action with plan
            state: Current workflow state
            
        Returns:
            Command to execute recovery
        """
        recovery_plan = recovery_action['recovery_plan']
        
        # Execute the first recovery step
        first_step = recovery_plan['steps'][0] if recovery_plan['steps'] else None
        
        if not first_step:
            # No recovery steps, escalate to human
            return await self._escalate_to_human({
                'reason': 'No recovery steps available',
                'context': recovery_action
            }, state)
        
        # Update state with recovery information
        updated_state = {
            "error_recovery_plan": recovery_plan,
            "error_recovery_step": 0,
            "current_agent": first_step['target_agent']
        }
        
        # Reset specific error counters if recovery is attempted
        if first_step['action'] == 'retry_code_generation':
            updated_state["code_errors"] = {}
        elif first_step['action'] == 'retry_rendering':
            updated_state["rendering_errors"] = {}
        elif first_step['action'] == 'retry_visual_analysis':
            updated_state["visual_errors"] = {}
        
        self.log_agent_action("executing_recovery_plan", {
            "plan_steps": len(recovery_plan['steps']),
            "first_step": first_step['action'],
            "target_agent": first_step['target_agent']
        })
        
        return Command(
            goto=first_step['target_agent'],
            update=updated_state
        )
    
    async def _complete_workflow_with_errors(self, state: VideoGenerationState) -> Command:
        """Complete workflow despite having some errors.
        
        Args:
            state: Current workflow state
            
        Returns:
            Command to end workflow
        """
        self.log_agent_action("completing_workflow_with_errors", {
            "error_count": state.get('error_count', 0),
            "completed_scenes": len(state.get('rendered_videos', {}))
        })
        
        return Command(
            goto="END",
            update={
                "workflow_complete": True,
                "workflow_completed_with_errors": True,
                "current_agent": None
            }
        )
    
    async def _continue_workflow(self, state: VideoGenerationState) -> Command:
        """Continue workflow to next appropriate step.
        
        Args:
            state: Current workflow state
            
        Returns:
            Command to continue workflow
        """
        # Determine next agent based on workflow state
        next_agent = self._determine_next_workflow_step(state)
        
        self.log_agent_action("continuing_workflow", {
            "next_agent": next_agent,
            "current_errors": state.get('error_count', 0)
        })
        
        return Command(
            goto=next_agent,
            update={
                "current_agent": next_agent,
                "error_count": 0,  # Reset error count to continue
                "escalated_errors": []  # Clear escalated errors
            }
        )
    
    def _classify_error_type(self, error: Dict[str, Any]) -> str:
        """Classify error by type based on error message and agent.
        
        Args:
            error: Error information
            
        Returns:
            Error type classification
        """
        agent = error.get('agent', '')
        error_msg = error.get('error', '').lower()
        
        # Agent-based classification
        if agent == 'code_generator_agent':
            return 'code_generation_errors'
        elif agent == 'renderer_agent':
            return 'rendering_errors'
        elif agent == 'visual_analysis_agent':
            return 'visual_analysis_errors'
        elif agent == 'rag_agent':
            return 'rag_errors'
        
        # Message-based classification
        if any(keyword in error_msg for keyword in ['syntax', 'import', 'module', 'attribute']):
            return 'code_generation_errors'
        elif any(keyword in error_msg for keyword in ['render', 'manim', 'ffmpeg', 'video']):
            return 'rendering_errors'
        elif any(keyword in error_msg for keyword in ['visual', 'image', 'analysis']):
            return 'visual_analysis_errors'
        elif any(keyword in error_msg for keyword in ['rag', 'vector', 'embedding', 'retrieval']):
            return 'rag_errors'
        elif any(keyword in error_msg for keyword in ['timeout', 'connection', 'network']):
            return 'system_errors'
        
        return 'unknown_errors'
    
    def _calculate_error_severity(self, error_classification: Dict[str, List], error_count: int, retry_count: Dict[str, int]) -> float:
        """Calculate overall error severity score.
        
        Args:
            error_classification: Classified errors
            error_count: Total error count
            retry_count: Retry counts by agent
            
        Returns:
            Severity score (0.0 to 10.0)
        """
        severity = 0.0
        
        # Base severity from error count
        severity += min(error_count * 0.5, 3.0)
        
        # Add severity based on error types
        severity += len(error_classification['code_generation_errors']) * 0.3
        severity += len(error_classification['rendering_errors']) * 0.4
        severity += len(error_classification['visual_analysis_errors']) * 0.2
        severity += len(error_classification['rag_errors']) * 0.1
        severity += len(error_classification['system_errors']) * 0.8
        severity += len(error_classification['unknown_errors']) * 0.6
        
        # Add severity from retry counts
        total_retries = sum(retry_count.values())
        severity += min(total_retries * 0.2, 2.0)
        
        # Cap at 10.0
        return min(severity, 10.0)
    
    def _detect_error_patterns(self, error_classification: Dict[str, List]) -> List[str]:
        """Detect recurring error patterns.
        
        Args:
            error_classification: Classified errors
            
        Returns:
            List of detected error patterns
        """
        patterns = []
        
        # Check for recurring error messages
        all_errors = []
        for error_list in error_classification.values():
            all_errors.extend(error_list)
        
        error_messages = [error.get('error', '') for error in all_errors]
        
        # Simple pattern detection - look for similar error messages
        message_counts = {}
        for msg in error_messages:
            # Normalize message for pattern matching
            normalized = re.sub(r'\d+', 'N', msg.lower())
            normalized = re.sub(r'[^\w\s]', ' ', normalized)
            normalized = ' '.join(normalized.split())
            
            message_counts[normalized] = message_counts.get(normalized, 0) + 1
        
        # Identify patterns (messages that occur more than once)
        for msg, count in message_counts.items():
            if count > 1:
                patterns.append(f"Recurring error pattern: {msg} (occurred {count} times)")
        
        return patterns
    
    def _requires_human_intervention(self, state: VideoGenerationState, severity_score: float) -> bool:
        """Determine if human intervention is required.
        
        Args:
            state: Current workflow state
            severity_score: Calculated severity score
            
        Returns:
            True if human intervention is required
        """
        # Check severity threshold
        if severity_score >= 7.0:
            return True
        
        # Check error count threshold
        if state.get('error_count', 0) >= 5:
            return True
        
        # Check if any agent has exceeded retry limits
        retry_count = state.get('retry_count', {})
        for agent, count in retry_count.items():
            if count >= self.max_retries:
                return True
        
        # Check for critical error patterns
        escalated_errors = state.get('escalated_errors', [])
        if len(escalated_errors) >= 3:
            return True
        
        return False
    
    def _identify_recoverable_errors(self, error_classification: Dict[str, List]) -> List[Dict[str, Any]]:
        """Identify errors that can be automatically recovered.
        
        Args:
            error_classification: Classified errors
            
        Returns:
            List of recoverable errors
        """
        recoverable = []
        
        # Code generation errors are often recoverable
        for error in error_classification['code_generation_errors']:
            if self._is_recoverable_code_error(error):
                recoverable.append({**error, 'recovery_type': 'code_regeneration'})
        
        # Some rendering errors are recoverable
        for error in error_classification['rendering_errors']:
            if self._is_recoverable_rendering_error(error):
                recoverable.append({**error, 'recovery_type': 'rendering_retry'})
        
        # Visual analysis errors might be recoverable
        for error in error_classification['visual_analysis_errors']:
            if self._is_recoverable_visual_error(error):
                recoverable.append({**error, 'recovery_type': 'visual_reanalysis'})
        
        return recoverable
    
    def _identify_critical_errors(self, error_classification: Dict[str, List]) -> List[Dict[str, Any]]:
        """Identify critical errors that require immediate attention.
        
        Args:
            error_classification: Classified errors
            
        Returns:
            List of critical errors
        """
        critical = []
        
        # System errors are usually critical
        critical.extend(error_classification['system_errors'])
        
        # Unknown errors might be critical
        critical.extend(error_classification['unknown_errors'])
        
        # Check for specific critical patterns in other error types
        for error_list in error_classification.values():
            for error in error_list:
                if self._is_critical_error(error):
                    critical.append(error)
        
        return critical
    
    def _is_recoverable_code_error(self, error: Dict[str, Any]) -> bool:
        """Check if a code generation error is recoverable.
        
        Args:
            error: Error information
            
        Returns:
            True if error is recoverable
        """
        error_msg = error.get('error', '').lower()
        
        # Recoverable error patterns
        recoverable_patterns = [
            'syntax error',
            'name error',
            'attribute error',
            'import error',
            'indentation error'
        ]
        
        return any(pattern in error_msg for pattern in recoverable_patterns)
    
    def _is_recoverable_rendering_error(self, error: Dict[str, Any]) -> bool:
        """Check if a rendering error is recoverable.
        
        Args:
            error: Error information
            
        Returns:
            True if error is recoverable
        """
        error_msg = error.get('error', '').lower()
        
        # Recoverable rendering patterns
        recoverable_patterns = [
            'timeout',
            'temporary',
            'retry',
            'connection'
        ]
        
        return any(pattern in error_msg for pattern in recoverable_patterns)
    
    def _is_recoverable_visual_error(self, error: Dict[str, Any]) -> bool:
        """Check if a visual analysis error is recoverable.
        
        Args:
            error: Error information
            
        Returns:
            True if error is recoverable
        """
        # Most visual analysis errors are recoverable
        return True
    
    def _is_critical_error(self, error: Dict[str, Any]) -> bool:
        """Check if an error is critical.
        
        Args:
            error: Error information
            
        Returns:
            True if error is critical
        """
        error_msg = error.get('error', '').lower()
        
        # Critical error patterns
        critical_patterns = [
            'out of memory',
            'disk full',
            'permission denied',
            'file not found',
            'configuration error',
            'authentication failed'
        ]
        
        return any(pattern in error_msg for pattern in critical_patterns)
    
    def _create_recovery_plan(self, recoverable_errors: List[Dict[str, Any]], state: VideoGenerationState) -> Dict[str, Any]:
        """Create a recovery plan for recoverable errors.
        
        Args:
            recoverable_errors: List of recoverable errors
            state: Current workflow state
            
        Returns:
            Recovery plan with steps
        """
        steps = []
        
        # Group errors by recovery type
        recovery_groups = {}
        for error in recoverable_errors:
            recovery_type = error.get('recovery_type', 'unknown')
            if recovery_type not in recovery_groups:
                recovery_groups[recovery_type] = []
            recovery_groups[recovery_type].append(error)
        
        # Create recovery steps
        for recovery_type, errors in recovery_groups.items():
            if recovery_type == 'code_regeneration':
                steps.append({
                    'action': 'retry_code_generation',
                    'target_agent': 'code_generator_agent',
                    'errors': errors,
                    'description': f'Retry code generation for {len(errors)} scenes'
                })
            elif recovery_type == 'rendering_retry':
                steps.append({
                    'action': 'retry_rendering',
                    'target_agent': 'renderer_agent',
                    'errors': errors,
                    'description': f'Retry rendering for {len(errors)} scenes'
                })
            elif recovery_type == 'visual_reanalysis':
                steps.append({
                    'action': 'retry_visual_analysis',
                    'target_agent': 'visual_analysis_agent',
                    'errors': errors,
                    'description': f'Retry visual analysis for {len(errors)} scenes'
                })
        
        return {
            'steps': steps,
            'total_errors': len(recoverable_errors),
            'estimated_time': len(steps) * 30  # Rough estimate in seconds
        }
    
    def _prepare_error_summary(self, state: VideoGenerationState) -> str:
        """Prepare a human-readable error summary.
        
        Args:
            state: Current workflow state
            
        Returns:
            Error summary string
        """
        summary_parts = []
        
        error_count = state.get('error_count', 0)
        summary_parts.append(f"Total errors: {error_count}")
        
        escalated_errors = state.get('escalated_errors', [])
        if escalated_errors:
            summary_parts.append(f"Escalated errors: {len(escalated_errors)}")
        
        code_errors = state.get('code_errors', {})
        if code_errors:
            summary_parts.append(f"Code generation errors in scenes: {list(code_errors.keys())}")
        
        rendering_errors = state.get('rendering_errors', {})
        if rendering_errors:
            summary_parts.append(f"Rendering errors in scenes: {list(rendering_errors.keys())}")
        
        visual_errors = state.get('visual_errors', {})
        if visual_errors:
            summary_parts.append(f"Visual errors in scenes: {list(visual_errors.keys())}")
        
        retry_count = state.get('retry_count', {})
        if retry_count:
            summary_parts.append(f"Retry counts: {retry_count}")
        
        return "\n".join(summary_parts)
    
    def _determine_next_workflow_step(self, state: VideoGenerationState) -> str:
        """Determine the next workflow step after error handling.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next agent name
        """
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
    
    def _initialize_error_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize error classification patterns.
        
        Returns:
            Dictionary of error patterns
        """
        return {
            "code_syntax_error": {
                "keywords": ["syntax error", "invalid syntax", "unexpected token"],
                "severity": "medium",
                "recovery_agent": "code_generator_agent",
                "max_attempts": 3,
                "use_rag": True
            },
            "code_import_error": {
                "keywords": ["import error", "module not found", "no module named"],
                "severity": "high",
                "recovery_agent": "code_generator_agent",
                "max_attempts": 2,
                "use_rag": True
            },
            "rendering_timeout": {
                "keywords": ["timeout", "timed out", "process timeout"],
                "severity": "medium",
                "recovery_agent": "renderer_agent",
                "max_attempts": 2,
                "use_alternative_quality": True
            },
            "visual_analysis_failure": {
                "keywords": ["visual analysis", "image processing", "vision model"],
                "severity": "low",
                "recovery_agent": "visual_analysis_agent",
                "max_attempts": 2,
                "use_fallback_prompt": True
            },
            "rag_retrieval_error": {
                "keywords": ["rag", "vector store", "embedding", "retrieval"],
                "severity": "low",
                "recovery_agent": "rag_agent",
                "max_attempts": 2,
                "use_fallback": True
            }
        }
    
    def _initialize_recovery_strategies(self) -> Dict[str, RecoveryStrategy]:
        """Initialize recovery strategies.
        
        Returns:
            Dictionary of recovery strategies
        """
        return {
            "code_syntax_error": RecoveryStrategy(
                error_pattern="code_syntax_error",
                recovery_agent="code_generator_agent",
                max_attempts=3,
                escalation_threshold=2,
                use_rag=True
            ),
            "rendering_timeout": RecoveryStrategy(
                error_pattern="rendering_timeout",
                recovery_agent="renderer_agent",
                max_attempts=2,
                escalation_threshold=1,
                use_alternative_quality=True
            ),
            "visual_analysis_failure": RecoveryStrategy(
                error_pattern="visual_analysis_failure",
                recovery_agent="visual_analysis_agent",
                max_attempts=2,
                escalation_threshold=1,
                use_fallback_prompt=True
            )
        }
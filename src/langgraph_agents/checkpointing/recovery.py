"""
Checkpoint recovery logic for resuming interrupted workflows.

This module provides functionality to recover and resume workflows from
checkpoints, handling state validation and recovery strategies.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from enum import Enum

from ..models.state import VideoGenerationState
from ..models.config import WorkflowConfig
from ..models.errors import WorkflowError, ErrorType, ErrorSeverity

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Recovery strategies for different failure scenarios."""
    RESUME_FROM_LAST = "resume_from_last"
    RESTART_CURRENT_STEP = "restart_current_step"
    RESTART_FROM_BEGINNING = "restart_from_beginning"
    MANUAL_INTERVENTION = "manual_intervention"


class RecoveryDecision:
    """Decision about how to recover from a checkpoint."""
    
    def __init__(
        self,
        strategy: RecoveryStrategy,
        checkpoint_id: str,
        reason: str,
        state_modifications: Optional[Dict[str, Any]] = None,
        requires_user_input: bool = False
    ):
        self.strategy = strategy
        self.checkpoint_id = checkpoint_id
        self.reason = reason
        self.state_modifications = state_modifications or {}
        self.requires_user_input = requires_user_input
        self.created_at = datetime.now()


class CheckpointRecovery:
    """
    Handles recovery logic for resuming workflows from checkpoints.
    
    This class analyzes checkpoint state and determines the best recovery
    strategy based on the workflow state, errors, and configuration.
    """
    
    def __init__(self, config: WorkflowConfig):
        """
        Initialize checkpoint recovery handler.
        
        Args:
            config: Workflow configuration
        """
        self.config = config
        self.recovery_history: List[RecoveryDecision] = []
        
        logger.info("Checkpoint recovery handler initialized")
    
    def analyze_checkpoint_state(self, state: VideoGenerationState) -> Dict[str, Any]:
        """
        Analyze a checkpoint state to determine its health and recoverability.
        
        Args:
            state: The checkpoint state to analyze
            
        Returns:
            Analysis results with health metrics and recommendations
        """
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "session_id": state.session_id,
            "current_step": state.current_step,
            "workflow_complete": state.workflow_complete,
            "workflow_interrupted": state.workflow_interrupted,
            "health_score": 0.0,
            "issues": [],
            "recommendations": [],
            "recoverable": True
        }
        
        # Check basic state integrity
        if not state.topic or not state.description:
            analysis["issues"].append("Missing required topic or description")
            analysis["health_score"] -= 20
        
        # Check workflow progress
        total_scenes = len(state.scene_implementations)
        if total_scenes == 0 and state.current_step != "planning":
            analysis["issues"].append("No scenes defined but workflow past planning")
            analysis["health_score"] -= 15
        
        # Check for errors
        error_count = len(state.errors)
        if error_count > 0:
            analysis["issues"].append(f"Workflow has {error_count} errors")
            analysis["health_score"] -= min(error_count * 5, 30)
        
        # Check retry counts
        max_retries_exceeded = any(
            count >= self.config.max_retries 
            for count in state.retry_counts.values()
        )
        if max_retries_exceeded:
            analysis["issues"].append("Maximum retries exceeded for some operations")
            analysis["health_score"] -= 25
        
        # Check for escalated errors
        if state.escalated_errors:
            analysis["issues"].append(f"Has {len(state.escalated_errors)} escalated errors")
            analysis["health_score"] -= 20
        
        # Check for pending human input
        if state.pending_human_input:
            analysis["issues"].append("Workflow waiting for human input")
            analysis["health_score"] -= 10
        
        # Check completion consistency
        if state.workflow_complete and state.current_step != "complete":
            analysis["issues"].append("Workflow marked complete but step is not 'complete'")
            analysis["health_score"] -= 15
        
        # Check scene consistency
        if total_scenes > 0:
            code_scenes = len(state.generated_code)
            rendered_scenes = len(state.rendered_videos)
            
            if code_scenes < total_scenes and state.current_step in ["rendering", "visual_analysis", "complete"]:
                analysis["issues"].append(f"Missing code for {total_scenes - code_scenes} scenes")
                analysis["health_score"] -= 10
            
            if rendered_scenes < code_scenes and state.current_step in ["visual_analysis", "complete"]:
                analysis["issues"].append(f"Missing renders for {code_scenes - rendered_scenes} scenes")
                analysis["health_score"] -= 10
        
        # Calculate final health score (0-100)
        analysis["health_score"] = max(0, min(100, 100 + analysis["health_score"]))
        
        # Generate recommendations
        if analysis["health_score"] < 50:
            analysis["recommendations"].append("Consider restarting workflow from beginning")
            analysis["recoverable"] = False
        elif analysis["health_score"] < 70:
            analysis["recommendations"].append("Restart from current step with error clearing")
        else:
            analysis["recommendations"].append("Resume from last checkpoint")
        
        if state.pending_human_input:
            analysis["recommendations"].append("Resolve pending human input before recovery")
        
        if max_retries_exceeded:
            analysis["recommendations"].append("Clear retry counts and review error causes")
        
        logger.info(f"Checkpoint analysis complete: health_score={analysis['health_score']:.1f}, "
                   f"issues={len(analysis['issues'])}, recoverable={analysis['recoverable']}")
        
        return analysis
    
    def determine_recovery_strategy(
        self,
        state: VideoGenerationState,
        analysis: Optional[Dict[str, Any]] = None
    ) -> RecoveryDecision:
        """
        Determine the best recovery strategy for a checkpoint state.
        
        Args:
            state: The checkpoint state
            analysis: Optional pre-computed analysis (will compute if not provided)
            
        Returns:
            Recovery decision with strategy and modifications
        """
        if analysis is None:
            analysis = self.analyze_checkpoint_state(state)
        
        health_score = analysis["health_score"]
        
        # Determine strategy based on health score and specific conditions
        if not analysis["recoverable"] or health_score < 30:
            decision = RecoveryDecision(
                strategy=RecoveryStrategy.RESTART_FROM_BEGINNING,
                checkpoint_id=state.session_id,
                reason=f"Health score too low ({health_score:.1f}) or unrecoverable state"
            )
        elif state.pending_human_input:
            decision = RecoveryDecision(
                strategy=RecoveryStrategy.MANUAL_INTERVENTION,
                checkpoint_id=state.session_id,
                reason="Workflow requires human input",
                requires_user_input=True
            )
        elif len(state.escalated_errors) > 0:
            decision = RecoveryDecision(
                strategy=RecoveryStrategy.MANUAL_INTERVENTION,
                checkpoint_id=state.session_id,
                reason="Workflow has escalated errors requiring manual review",
                requires_user_input=True
            )
        elif health_score < 60 or any(count >= self.config.max_retries for count in state.retry_counts.values()):
            # Restart current step with cleanup
            modifications = {
                "clear_errors": True,
                "reset_retry_counts": True,
                "clear_step_data": True
            }
            decision = RecoveryDecision(
                strategy=RecoveryStrategy.RESTART_CURRENT_STEP,
                checkpoint_id=state.session_id,
                reason=f"Health score moderate ({health_score:.1f}) or max retries exceeded",
                state_modifications=modifications
            )
        else:
            # Resume from last checkpoint
            decision = RecoveryDecision(
                strategy=RecoveryStrategy.RESUME_FROM_LAST,
                checkpoint_id=state.session_id,
                reason=f"Good health score ({health_score:.1f}), safe to resume"
            )
        
        self.recovery_history.append(decision)
        
        logger.info(f"Recovery strategy determined: {decision.strategy.value} - {decision.reason}")
        return decision
    
    def apply_recovery_modifications(
        self,
        state: VideoGenerationState,
        decision: RecoveryDecision
    ) -> VideoGenerationState:
        """
        Apply recovery modifications to a state based on the recovery decision.
        
        Args:
            state: The state to modify
            decision: The recovery decision with modifications
            
        Returns:
            Modified state ready for recovery
        """
        if not decision.state_modifications:
            return state
        
        modifications = decision.state_modifications
        
        # Clear errors if requested
        if modifications.get("clear_errors", False):
            state.errors.clear()
            state.code_errors.clear()
            state.rendering_errors.clear()
            state.visual_errors.clear()
            logger.info("Cleared all errors from state")
        
        # Reset retry counts if requested
        if modifications.get("reset_retry_counts", False):
            state.retry_counts.clear()
            logger.info("Reset all retry counts")
        
        # Clear step-specific data if requested
        if modifications.get("clear_step_data", False):
            current_step = state.current_step
            
            if current_step == "code_generation":
                state.generated_code.clear()
                state.code_errors.clear()
                logger.info("Cleared code generation data")
            elif current_step == "rendering":
                state.rendered_videos.clear()
                state.rendering_errors.clear()
                logger.info("Cleared rendering data")
            elif current_step == "visual_analysis":
                state.visual_analysis_results.clear()
                state.visual_errors.clear()
                logger.info("Cleared visual analysis data")
        
        # Clear escalated errors if requested
        if modifications.get("clear_escalated_errors", False):
            state.escalated_errors.clear()
            logger.info("Cleared escalated errors")
        
        # Reset workflow interruption if requested
        if modifications.get("reset_interruption", False):
            state.workflow_interrupted = False
            state.pending_human_input = None
            logger.info("Reset workflow interruption state")
        
        # Add recovery trace
        state.add_execution_trace("recovery_applied", {
            "strategy": decision.strategy.value,
            "reason": decision.reason,
            "modifications": list(modifications.keys()),
            "recovery_timestamp": decision.created_at.isoformat()
        })
        
        return state
    
    def validate_recovery_state(self, state: VideoGenerationState) -> Tuple[bool, List[str]]:
        """
        Validate that a recovered state is ready for workflow execution.
        
        Args:
            state: The recovered state to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required fields
        if not state.topic:
            issues.append("Missing topic")
        if not state.description:
            issues.append("Missing description")
        if not state.session_id:
            issues.append("Missing session_id")
        
        # Check workflow state consistency
        if state.workflow_complete and state.current_step != "complete":
            issues.append("Workflow marked complete but current step is not 'complete'")
        
        # Check step-specific requirements
        current_step = state.current_step
        
        if current_step == "code_generation" and not state.scene_implementations:
            issues.append("Code generation step requires scene implementations")
        
        if current_step == "rendering" and not state.generated_code:
            issues.append("Rendering step requires generated code")
        
        if current_step == "visual_analysis" and not state.rendered_videos:
            issues.append("Visual analysis step requires rendered videos")
        
        # Check for blocking conditions
        if state.pending_human_input:
            issues.append("State has pending human input")
        
        if any(count >= self.config.max_retries for count in state.retry_counts.values()):
            issues.append("Some operations have exceeded maximum retries")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info("Recovery state validation passed")
        else:
            logger.warning(f"Recovery state validation failed: {issues}")
        
        return is_valid, issues
    
    def get_recovery_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of recovery decisions.
        
        Returns:
            List of recovery decision summaries
        """
        return [
            {
                "strategy": decision.strategy.value,
                "checkpoint_id": decision.checkpoint_id,
                "reason": decision.reason,
                "requires_user_input": decision.requires_user_input,
                "created_at": decision.created_at.isoformat(),
                "modifications": list(decision.state_modifications.keys()) if decision.state_modifications else []
            }
            for decision in self.recovery_history
        ]
    
    def clear_recovery_history(self) -> None:
        """Clear the recovery history."""
        self.recovery_history.clear()
        logger.info("Recovery history cleared")


def create_recovery_handler(config: WorkflowConfig) -> CheckpointRecovery:
    """
    Create a checkpoint recovery handler with the given configuration.
    
    Args:
        config: Workflow configuration
        
    Returns:
        Configured CheckpointRecovery instance
    """
    return CheckpointRecovery(config)
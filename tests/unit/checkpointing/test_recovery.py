"""
Unit tests for checkpoint recovery functionality.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.langgraph_agents.checkpointing.recovery import (
    CheckpointRecovery,
    RecoveryStrategy,
    RecoveryDecision,
    create_recovery_handler
)
from src.langgraph_agents.models.state import VideoGenerationState
from src.langgraph_agents.models.config import WorkflowConfig
from src.langgraph_agents.models.errors import WorkflowError, ErrorType, ErrorSeverity


class TestRecoveryDecision:
    """Test recovery decision data structure."""
    
    def test_recovery_decision_creation(self):
        """Test creating a recovery decision."""
        decision = RecoveryDecision(
            strategy=RecoveryStrategy.RESUME_FROM_LAST,
            checkpoint_id="test-123",
            reason="Good health score",
            state_modifications={"clear_errors": True},
            requires_user_input=False
        )
        
        assert decision.strategy == RecoveryStrategy.RESUME_FROM_LAST
        assert decision.checkpoint_id == "test-123"
        assert decision.reason == "Good health score"
        assert decision.state_modifications == {"clear_errors": True}
        assert decision.requires_user_input is False
        assert isinstance(decision.created_at, datetime)
    
    def test_recovery_decision_defaults(self):
        """Test recovery decision with default values."""
        decision = RecoveryDecision(
            strategy=RecoveryStrategy.RESTART_CURRENT_STEP,
            checkpoint_id="test-456",
            reason="Test reason"
        )
        
        assert decision.state_modifications == {}
        assert decision.requires_user_input is False


class TestCheckpointRecovery:
    """Test checkpoint recovery functionality."""
    
    @pytest.fixture
    def workflow_config(self):
        """Create a test workflow configuration."""
        return WorkflowConfig(max_retries=3, timeout_seconds=60)
    
    @pytest.fixture
    def recovery_handler(self, workflow_config):
        """Create a recovery handler."""
        return CheckpointRecovery(workflow_config)
    
    @pytest.fixture
    def healthy_state(self):
        """Create a healthy video generation state."""
        return VideoGenerationState(
            topic="Test Video",
            description="A test video about programming",
            session_id="test-session-123",
            scene_implementations={
                1: "Introduction scene",
                2: "Main content scene"
            },
            generated_code={
                1: "from manim import *\nclass Scene1(Scene):\n    def construct(self):\n        pass"
            },
            current_step="code_generation"
        )
    
    @pytest.fixture
    def unhealthy_state(self, healthy_state):
        """Create an unhealthy state with errors."""
        state = healthy_state.model_copy()
        state.errors = [
            WorkflowError(
                step="code_generation",
                error_type=ErrorType.CONTENT,
                message="Code generation failed",
                severity=ErrorSeverity.HIGH
            )
        ]
        state.retry_counts = {"code_generation": 5}  # Exceeds max retries
        state.escalated_errors = [{"error": "Critical failure", "timestamp": datetime.now()}]
        return state
    
    def test_analyze_healthy_checkpoint_state(self, recovery_handler, healthy_state):
        """Test analyzing a healthy checkpoint state."""
        analysis = recovery_handler.analyze_checkpoint_state(healthy_state)
        
        assert analysis["session_id"] == healthy_state.session_id
        assert analysis["current_step"] == "code_generation"
        assert analysis["workflow_complete"] is False
        assert analysis["workflow_interrupted"] is False
        assert analysis["health_score"] > 70  # Should be healthy
        assert analysis["recoverable"] is True
        assert len(analysis["issues"]) == 0
        assert "Resume from last checkpoint" in analysis["recommendations"]
    
    def test_analyze_unhealthy_checkpoint_state(self, recovery_handler, unhealthy_state):
        """Test analyzing an unhealthy checkpoint state."""
        analysis = recovery_handler.analyze_checkpoint_state(unhealthy_state)
        
        assert analysis["health_score"] < 50  # Should be unhealthy
        assert analysis["recoverable"] is False
        assert len(analysis["issues"]) > 0
        assert any("errors" in issue for issue in analysis["issues"])
        assert any("retries exceeded" in issue for issue in analysis["issues"])
        assert any("escalated errors" in issue for issue in analysis["issues"])
        assert "Consider restarting workflow from beginning" in analysis["recommendations"]
    
    def test_analyze_state_missing_required_fields(self, recovery_handler):
        """Test analyzing state with missing required fields."""
        incomplete_state = VideoGenerationState(
            topic="",  # Empty topic
            description="Test description",
            session_id="test-123"
        )
        
        analysis = recovery_handler.analyze_checkpoint_state(incomplete_state)
        
        assert analysis["health_score"] < 100
        assert any("Missing required topic" in issue for issue in analysis["issues"])
    
    def test_analyze_state_inconsistent_workflow(self, recovery_handler, healthy_state):
        """Test analyzing state with workflow inconsistencies."""
        inconsistent_state = healthy_state.model_copy()
        inconsistent_state.workflow_complete = True
        inconsistent_state.current_step = "code_generation"  # Should be "complete"
        
        analysis = recovery_handler.analyze_checkpoint_state(inconsistent_state)
        
        assert analysis["health_score"] < 100
        assert any("marked complete but step is not" in issue for issue in analysis["issues"])
    
    def test_analyze_state_scene_inconsistencies(self, recovery_handler, healthy_state):
        """Test analyzing state with scene inconsistencies."""
        inconsistent_state = healthy_state.model_copy()
        inconsistent_state.current_step = "rendering"
        # Has scene implementations but missing generated code for scene 2
        
        analysis = recovery_handler.analyze_checkpoint_state(inconsistent_state)
        
        assert analysis["health_score"] < 100
        assert any("Missing code for" in issue for issue in analysis["issues"])
    
    def test_determine_recovery_strategy_healthy(self, recovery_handler, healthy_state):
        """Test determining recovery strategy for healthy state."""
        decision = recovery_handler.determine_recovery_strategy(healthy_state)
        
        assert decision.strategy == RecoveryStrategy.RESUME_FROM_LAST
        assert decision.checkpoint_id == healthy_state.session_id
        assert not decision.requires_user_input
        assert decision.state_modifications == {}
        assert "Good health score" in decision.reason
    
    def test_determine_recovery_strategy_unhealthy(self, recovery_handler, unhealthy_state):
        """Test determining recovery strategy for unhealthy state."""
        decision = recovery_handler.determine_recovery_strategy(unhealthy_state)
        
        assert decision.strategy == RecoveryStrategy.RESTART_FROM_BEGINNING
        assert decision.checkpoint_id == unhealthy_state.session_id
        assert not decision.requires_user_input
        assert "Health score too low" in decision.reason or "unrecoverable state" in decision.reason
    
    def test_determine_recovery_strategy_pending_human_input(self, recovery_handler, healthy_state):
        """Test determining recovery strategy for state with pending human input."""
        human_input_state = healthy_state.model_copy()
        human_input_state.pending_human_input = {"type": "approval", "message": "Review needed"}
        
        decision = recovery_handler.determine_recovery_strategy(human_input_state)
        
        assert decision.strategy == RecoveryStrategy.MANUAL_INTERVENTION
        assert decision.requires_user_input is True
        assert "requires human input" in decision.reason
    
    def test_determine_recovery_strategy_escalated_errors(self, recovery_handler, healthy_state):
        """Test determining recovery strategy for state with escalated errors."""
        escalated_state = healthy_state.model_copy()
        escalated_state.escalated_errors = [{"error": "Critical failure"}]
        
        decision = recovery_handler.determine_recovery_strategy(escalated_state)
        
        assert decision.strategy == RecoveryStrategy.MANUAL_INTERVENTION
        assert decision.requires_user_input is True
        assert "escalated errors" in decision.reason
    
    def test_determine_recovery_strategy_max_retries(self, recovery_handler, healthy_state):
        """Test determining recovery strategy for state with max retries exceeded."""
        retry_state = healthy_state.model_copy()
        retry_state.retry_counts = {"code_generation": 3}  # At max retries
        
        decision = recovery_handler.determine_recovery_strategy(retry_state)
        
        assert decision.strategy == RecoveryStrategy.RESTART_CURRENT_STEP
        assert not decision.requires_user_input
        assert decision.state_modifications is not None
        assert "max retries exceeded" in decision.reason
    
    def test_apply_recovery_modifications_clear_errors(self, recovery_handler, unhealthy_state):
        """Test applying recovery modifications to clear errors."""
        decision = RecoveryDecision(
            strategy=RecoveryStrategy.RESTART_CURRENT_STEP,
            checkpoint_id=unhealthy_state.session_id,
            reason="Test recovery",
            state_modifications={"clear_errors": True}
        )
        
        original_error_count = len(unhealthy_state.errors)
        modified_state = recovery_handler.apply_recovery_modifications(unhealthy_state, decision)
        
        assert len(modified_state.errors) == 0
        assert original_error_count > 0  # Verify we actually cleared something
        assert len(modified_state.execution_trace) > 0  # Recovery trace added
    
    def test_apply_recovery_modifications_reset_retry_counts(self, recovery_handler, unhealthy_state):
        """Test applying recovery modifications to reset retry counts."""
        decision = RecoveryDecision(
            strategy=RecoveryStrategy.RESTART_CURRENT_STEP,
            checkpoint_id=unhealthy_state.session_id,
            reason="Test recovery",
            state_modifications={"reset_retry_counts": True}
        )
        
        original_retry_count = len(unhealthy_state.retry_counts)
        modified_state = recovery_handler.apply_recovery_modifications(unhealthy_state, decision)
        
        assert len(modified_state.retry_counts) == 0
        assert original_retry_count > 0  # Verify we actually cleared something
    
    def test_apply_recovery_modifications_clear_step_data(self, recovery_handler, healthy_state):
        """Test applying recovery modifications to clear step data."""
        decision = RecoveryDecision(
            strategy=RecoveryStrategy.RESTART_CURRENT_STEP,
            checkpoint_id=healthy_state.session_id,
            reason="Test recovery",
            state_modifications={"clear_step_data": True}
        )
        
        # Add some step data to clear
        healthy_state.generated_code = {1: "test code"}
        healthy_state.current_step = "code_generation"
        
        modified_state = recovery_handler.apply_recovery_modifications(healthy_state, decision)
        
        assert len(modified_state.generated_code) == 0
    
    def test_apply_recovery_modifications_multiple(self, recovery_handler, unhealthy_state):
        """Test applying multiple recovery modifications."""
        decision = RecoveryDecision(
            strategy=RecoveryStrategy.RESTART_CURRENT_STEP,
            checkpoint_id=unhealthy_state.session_id,
            reason="Test recovery",
            state_modifications={
                "clear_errors": True,
                "reset_retry_counts": True,
                "clear_escalated_errors": True
            }
        )
        
        modified_state = recovery_handler.apply_recovery_modifications(unhealthy_state, decision)
        
        assert len(modified_state.errors) == 0
        assert len(modified_state.retry_counts) == 0
        assert len(modified_state.escalated_errors) == 0
    
    def test_apply_recovery_modifications_no_modifications(self, recovery_handler, healthy_state):
        """Test applying recovery with no modifications."""
        decision = RecoveryDecision(
            strategy=RecoveryStrategy.RESUME_FROM_LAST,
            checkpoint_id=healthy_state.session_id,
            reason="No changes needed"
        )
        
        original_state = healthy_state.model_copy()
        modified_state = recovery_handler.apply_recovery_modifications(healthy_state, decision)
        
        # State should be unchanged except for execution trace
        assert modified_state.topic == original_state.topic
        assert modified_state.description == original_state.description
        assert modified_state.current_step == original_state.current_step
    
    def test_validate_recovery_state_valid(self, recovery_handler, healthy_state):
        """Test validating a valid recovery state."""
        is_valid, issues = recovery_handler.validate_recovery_state(healthy_state)
        
        assert is_valid is True
        assert len(issues) == 0
    
    def test_validate_recovery_state_missing_required_fields(self, recovery_handler):
        """Test validating state with missing required fields."""
        invalid_state = VideoGenerationState(
            topic="",  # Missing topic
            description="",  # Missing description
            session_id=""  # Missing session_id
        )
        
        is_valid, issues = recovery_handler.validate_recovery_state(invalid_state)
        
        assert is_valid is False
        assert "Missing topic" in issues
        assert "Missing description" in issues
        assert "Missing session_id" in issues
    
    def test_validate_recovery_state_workflow_inconsistency(self, recovery_handler, healthy_state):
        """Test validating state with workflow inconsistency."""
        inconsistent_state = healthy_state.model_copy()
        inconsistent_state.workflow_complete = True
        inconsistent_state.current_step = "code_generation"
        
        is_valid, issues = recovery_handler.validate_recovery_state(inconsistent_state)
        
        assert is_valid is False
        assert any("marked complete but current step is not" in issue for issue in issues)
    
    def test_validate_recovery_state_step_requirements(self, recovery_handler):
        """Test validating state with unmet step requirements."""
        # Code generation step without scene implementations
        invalid_state = VideoGenerationState(
            topic="Test",
            description="Test description",
            session_id="test-123",
            current_step="code_generation"
            # Missing scene_implementations
        )
        
        is_valid, issues = recovery_handler.validate_recovery_state(invalid_state)
        
        assert is_valid is False
        assert any("requires scene implementations" in issue for issue in issues)
    
    def test_validate_recovery_state_pending_human_input(self, recovery_handler, healthy_state):
        """Test validating state with pending human input."""
        human_input_state = healthy_state.model_copy()
        human_input_state.pending_human_input = {"type": "approval"}
        
        is_valid, issues = recovery_handler.validate_recovery_state(human_input_state)
        
        assert is_valid is False
        assert "pending human input" in issues
    
    def test_validate_recovery_state_max_retries_exceeded(self, recovery_handler, healthy_state):
        """Test validating state with max retries exceeded."""
        retry_state = healthy_state.model_copy()
        retry_state.retry_counts = {"code_generation": 5}  # Exceeds max retries
        
        is_valid, issues = recovery_handler.validate_recovery_state(retry_state)
        
        assert is_valid is False
        assert any("exceeded maximum retries" in issue for issue in issues)
    
    def test_recovery_history_tracking(self, recovery_handler, healthy_state):
        """Test recovery history tracking."""
        # Initially no history
        history = recovery_handler.get_recovery_history()
        assert len(history) == 0
        
        # Make a recovery decision
        decision = recovery_handler.determine_recovery_strategy(healthy_state)
        
        # Should have history now
        history = recovery_handler.get_recovery_history()
        assert len(history) == 1
        assert history[0]["strategy"] == decision.strategy.value
        assert history[0]["checkpoint_id"] == healthy_state.session_id
        assert history[0]["requires_user_input"] == decision.requires_user_input
        assert "created_at" in history[0]
        assert "modifications" in history[0]
        
        # Make another decision
        unhealthy_state = healthy_state.model_copy()
        unhealthy_state.retry_counts = {"test": 5}
        recovery_handler.determine_recovery_strategy(unhealthy_state)
        
        # Should have two entries
        history = recovery_handler.get_recovery_history()
        assert len(history) == 2
    
    def test_clear_recovery_history(self, recovery_handler, healthy_state):
        """Test clearing recovery history."""
        # Make a decision to create history
        recovery_handler.determine_recovery_strategy(healthy_state)
        
        # Verify history exists
        history = recovery_handler.get_recovery_history()
        assert len(history) == 1
        
        # Clear history
        recovery_handler.clear_recovery_history()
        
        # Verify history is cleared
        history = recovery_handler.get_recovery_history()
        assert len(history) == 0


class TestRecoveryHandlerFactory:
    """Test recovery handler factory function."""
    
    def test_create_recovery_handler(self):
        """Test creating recovery handler with factory function."""
        config = WorkflowConfig(max_retries=5, timeout_seconds=120)
        
        handler = create_recovery_handler(config)
        
        assert handler is not None
        assert isinstance(handler, CheckpointRecovery)
        assert handler.config == config
        assert len(handler.recovery_history) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
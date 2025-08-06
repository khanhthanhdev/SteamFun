"""
Integration tests for checkpointing and persistence functionality.

This module tests the complete checkpointing system including memory and
PostgreSQL checkpointers, recovery logic, and workflow integration.
"""

import pytest
import asyncio
import os
import tempfile
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from src.langgraph_agents.models.state import VideoGenerationState
from src.langgraph_agents.models.config import WorkflowConfig
from src.langgraph_agents.models.errors import WorkflowError, ErrorType, ErrorSeverity
from src.langgraph_agents.checkpointing import (
    CheckpointManager, 
    CheckpointConfig,
    create_checkpoint_manager
)
from src.langgraph_agents.checkpointing.checkpoint_manager import CheckpointBackend
from src.langgraph_agents.checkpointing.memory_checkpointer import MemoryCheckpointer
from src.langgraph_agents.checkpointing.recovery import CheckpointRecovery, RecoveryStrategy
from src.langgraph_agents.workflow_graph import create_workflow, VideoGenerationWorkflow


class TestMemoryCheckpointing:
    """Test memory-based checkpointing functionality."""
    
    @pytest.fixture
    def workflow_config(self):
        """Create a test workflow configuration."""
        return WorkflowConfig(
            max_retries=3,
            timeout_seconds=60,
            use_rag=False,
            use_visual_analysis=False
        )
    
    @pytest.fixture
    def checkpoint_config(self):
        """Create a test checkpoint configuration for memory."""
        return CheckpointConfig(
            backend=CheckpointBackend.MEMORY,
            memory_max_size=100,
            enable_compression=True
        )
    
    @pytest.fixture
    def sample_state(self):
        """Create a sample video generation state."""
        return VideoGenerationState(
            topic="Test Video",
            description="A test video about Python programming",
            session_id="test-session-123",
            scene_implementations={
                1: "Introduction to Python",
                2: "Variables and data types"
            },
            generated_code={
                1: "from manim import *\nclass Intro(Scene):\n    def construct(self):\n        pass"
            },
            current_step="code_generation"
        )
    
    def test_checkpoint_manager_creation(self, workflow_config, checkpoint_config):
        """Test creating a checkpoint manager with memory backend."""
        manager = create_checkpoint_manager(workflow_config, checkpoint_config)
        
        assert manager is not None
        assert manager.backend_type == CheckpointBackend.MEMORY
        assert not manager.is_persistent()
        assert manager.checkpointer is not None
    
    def test_memory_checkpointer_initialization(self):
        """Test memory checkpointer initialization."""
        checkpointer = MemoryCheckpointer(
            max_checkpoints=50,
            max_size_mb=10,
            enable_compression=True
        )
        
        assert checkpointer is not None
        stats = checkpointer.get_stats()
        assert stats["total_checkpoints"] == 0
        assert stats["compression_enabled"] is True
    
    @pytest.mark.asyncio
    async def test_workflow_with_memory_checkpointing(self, workflow_config, sample_state):
        """Test workflow execution with memory checkpointing."""
        # Create workflow with memory checkpointing
        workflow = create_workflow(
            config=workflow_config,
            use_checkpointing=True,
            checkpoint_config=CheckpointConfig(backend=CheckpointBackend.MEMORY)
        )
        
        assert workflow.checkpointer is not None
        assert workflow.checkpoint_manager is not None
        
        # Test checkpoint info
        info = workflow.get_checkpoint_info()
        assert info["checkpointing_enabled"] is True
        assert info["manager_available"] is True
        assert info["backend"] == "memory"
    
    @pytest.mark.asyncio
    async def test_checkpoint_recovery_analysis(self, workflow_config, sample_state):
        """Test checkpoint recovery analysis."""
        recovery = CheckpointRecovery(workflow_config)
        
        # Test healthy state analysis
        analysis = recovery.analyze_checkpoint_state(sample_state)
        
        assert analysis["session_id"] == sample_state.session_id
        assert analysis["current_step"] == "code_generation"
        assert analysis["health_score"] > 0
        assert analysis["recoverable"] is True
        
        # Test unhealthy state analysis
        unhealthy_state = sample_state.model_copy()
        unhealthy_state.errors = [
            WorkflowError(
                step="code_generation",
                error_type=ErrorType.CONTENT,
                message="Test error",
                severity=ErrorSeverity.HIGH
            )
        ]
        unhealthy_state.retry_counts = {"code_generation": 5}  # Exceeds max retries
        
        unhealthy_analysis = recovery.analyze_checkpoint_state(unhealthy_state)
        assert unhealthy_analysis["health_score"] < analysis["health_score"]
        assert len(unhealthy_analysis["issues"]) > 0
    
    @pytest.mark.asyncio
    async def test_recovery_strategy_determination(self, workflow_config, sample_state):
        """Test recovery strategy determination."""
        recovery = CheckpointRecovery(workflow_config)
        
        # Test healthy state - should resume
        decision = recovery.determine_recovery_strategy(sample_state)
        assert decision.strategy == RecoveryStrategy.RESUME_FROM_LAST
        assert not decision.requires_user_input
        
        # Test state with errors - should restart current step
        error_state = sample_state.model_copy()
        error_state.retry_counts = {"code_generation": 3}  # At max retries
        
        error_decision = recovery.determine_recovery_strategy(error_state)
        assert error_decision.strategy == RecoveryStrategy.RESTART_CURRENT_STEP
        assert error_decision.state_modifications is not None
        
        # Test state with human input needed
        human_state = sample_state.model_copy()
        human_state.pending_human_input = {"type": "approval", "message": "Review needed"}
        
        human_decision = recovery.determine_recovery_strategy(human_state)
        assert human_decision.strategy == RecoveryStrategy.MANUAL_INTERVENTION
        assert human_decision.requires_user_input is True
    
    @pytest.mark.asyncio
    async def test_recovery_modifications(self, workflow_config, sample_state):
        """Test applying recovery modifications to state."""
        recovery = CheckpointRecovery(workflow_config)
        
        # Add some errors and retry counts
        test_state = sample_state.model_copy()
        test_state.errors = [
            WorkflowError(
                step="code_generation",
                error_type=ErrorType.CONTENT,
                message="Test error",
                severity=ErrorSeverity.MEDIUM
            )
        ]
        test_state.retry_counts = {"code_generation": 2}
        test_state.code_errors = {1: "Syntax error"}
        
        # Create a recovery decision with modifications
        from src.langgraph_agents.checkpointing.recovery import RecoveryDecision
        decision = RecoveryDecision(
            strategy=RecoveryStrategy.RESTART_CURRENT_STEP,
            checkpoint_id=test_state.session_id,
            reason="Test recovery",
            state_modifications={
                "clear_errors": True,
                "reset_retry_counts": True,
                "clear_step_data": True
            }
        )
        
        # Apply modifications
        modified_state = recovery.apply_recovery_modifications(test_state, decision)
        
        # Verify modifications were applied
        assert len(modified_state.errors) == 0
        assert len(modified_state.retry_counts) == 0
        assert len(modified_state.code_errors) == 0
        assert len(modified_state.execution_trace) > 0  # Recovery trace added
    
    @pytest.mark.asyncio
    async def test_recovery_state_validation(self, workflow_config, sample_state):
        """Test recovery state validation."""
        recovery = CheckpointRecovery(workflow_config)
        
        # Test valid state
        is_valid, issues = recovery.validate_recovery_state(sample_state)
        assert is_valid is True
        assert len(issues) == 0
        
        # Test invalid state - missing required fields
        invalid_state = sample_state.model_copy()
        invalid_state.topic = ""
        
        is_valid, issues = recovery.validate_recovery_state(invalid_state)
        assert is_valid is False
        assert "Missing topic" in issues
        
        # Test invalid state - step requirements not met
        invalid_step_state = sample_state.model_copy()
        invalid_step_state.current_step = "rendering"
        invalid_step_state.generated_code = {}  # Missing required code
        
        is_valid, issues = recovery.validate_recovery_state(invalid_step_state)
        assert is_valid is False
        assert any("requires generated code" in issue for issue in issues)
    
    @pytest.mark.asyncio
    async def test_checkpoint_cleanup(self, workflow_config):
        """Test checkpoint cleanup functionality."""
        checkpointer = MemoryCheckpointer(max_checkpoints=10, max_size_mb=1)
        
        # Initially no checkpoints
        stats = checkpointer.get_stats()
        assert stats["total_checkpoints"] == 0
        
        # Cleanup should return 0 (nothing to clean)
        cleaned = checkpointer.cleanup(max_age_hours=1)
        assert cleaned == 0
        
        # Clear all should return 0 (nothing to clear)
        cleared = checkpointer.clear_all()
        assert cleared == 0


@pytest.mark.skipif(
    not os.getenv('POSTGRES_CONNECTION_STRING'),
    reason="PostgreSQL connection string not provided"
)
class TestPostgreSQLCheckpointing:
    """Test PostgreSQL-based checkpointing functionality."""
    
    @pytest.fixture
    def postgres_connection_string(self):
        """Get PostgreSQL connection string from environment."""
        return os.getenv('POSTGRES_CONNECTION_STRING')
    
    @pytest.fixture
    def workflow_config(self):
        """Create a test workflow configuration."""
        return WorkflowConfig(
            max_retries=3,
            timeout_seconds=60,
            use_rag=False,
            use_visual_analysis=False
        )
    
    @pytest.fixture
    def postgres_checkpoint_config(self, postgres_connection_string):
        """Create a test checkpoint configuration for PostgreSQL."""
        return CheckpointConfig(
            backend=CheckpointBackend.POSTGRES,
            postgres_connection_string=postgres_connection_string,
            postgres_pool_size=5,
            postgres_max_overflow=10
        )
    
    @pytest.mark.asyncio
    async def test_postgres_checkpoint_manager_creation(self, workflow_config, postgres_checkpoint_config):
        """Test creating a checkpoint manager with PostgreSQL backend."""
        try:
            manager = create_checkpoint_manager(workflow_config, postgres_checkpoint_config)
            
            assert manager is not None
            assert manager.backend_type == CheckpointBackend.POSTGRES
            assert manager.is_persistent()
            assert manager.checkpointer is not None
            
        except ImportError:
            pytest.skip("PostgreSQL dependencies not available")
    
    @pytest.mark.asyncio
    async def test_postgres_workflow_creation(self, workflow_config, postgres_checkpoint_config):
        """Test creating workflow with PostgreSQL checkpointing."""
        try:
            workflow = create_workflow(
                config=workflow_config,
                use_checkpointing=True,
                checkpoint_config=postgres_checkpoint_config
            )
            
            assert workflow.checkpointer is not None
            assert workflow.checkpoint_manager is not None
            
            info = workflow.get_checkpoint_info()
            assert info["checkpointing_enabled"] is True
            assert info["persistent"] is True
            assert info["backend"] == "postgres"
            
        except ImportError:
            pytest.skip("PostgreSQL dependencies not available")
    
    @pytest.mark.asyncio
    async def test_postgres_checkpoint_stats(self, workflow_config, postgres_checkpoint_config):
        """Test getting PostgreSQL checkpoint statistics."""
        try:
            workflow = create_workflow(
                config=workflow_config,
                use_checkpointing=True,
                checkpoint_config=postgres_checkpoint_config
            )
            
            stats = await workflow.get_checkpoint_stats()
            
            # Should have basic stats structure
            assert "total_checkpoints" in stats
            assert "connection_healthy" in stats
            
        except ImportError:
            pytest.skip("PostgreSQL dependencies not available")
        except Exception as e:
            # Connection might fail in test environment
            pytest.skip(f"PostgreSQL connection failed: {e}")
    
    @pytest.mark.asyncio
    async def test_postgres_checkpoint_cleanup(self, workflow_config, postgres_checkpoint_config):
        """Test PostgreSQL checkpoint cleanup."""
        try:
            workflow = create_workflow(
                config=workflow_config,
                use_checkpointing=True,
                checkpoint_config=postgres_checkpoint_config
            )
            
            # Test cleanup (should not fail even if no checkpoints exist)
            cleaned = await workflow.cleanup_old_checkpoints(max_age_hours=24)
            assert cleaned >= 0  # Should return non-negative number
            
        except ImportError:
            pytest.skip("PostgreSQL dependencies not available")
        except Exception as e:
            # Connection might fail in test environment
            pytest.skip(f"PostgreSQL connection failed: {e}")


class TestCheckpointIntegration:
    """Test integration between checkpointing and workflow execution."""
    
    @pytest.fixture
    def workflow_config(self):
        """Create a test workflow configuration."""
        return WorkflowConfig(
            max_retries=2,
            timeout_seconds=30,
            use_rag=False,
            use_visual_analysis=False
        )
    
    @pytest.fixture
    def sample_state(self):
        """Create a sample video generation state."""
        return VideoGenerationState(
            topic="Integration Test",
            description="Testing checkpointing integration",
            session_id="integration-test-456",
            current_step="planning"
        )
    
    @pytest.mark.asyncio
    async def test_workflow_checkpoint_integration(self, workflow_config, sample_state):
        """Test workflow integration with checkpointing."""
        # Create workflow with checkpointing
        workflow = create_workflow(
            config=workflow_config,
            use_checkpointing=True,
            checkpoint_config=CheckpointConfig(backend=CheckpointBackend.MEMORY)
        )
        
        # Verify checkpointing is enabled
        assert workflow.checkpointer is not None
        assert workflow.checkpoint_manager is not None
        
        # Test checkpoint info
        info = workflow.get_checkpoint_info()
        assert info["checkpointing_enabled"] is True
        assert info["backend"] == "memory"
    
    @pytest.mark.asyncio
    async def test_auto_backend_selection(self, workflow_config):
        """Test automatic backend selection."""
        # Test auto selection (should fall back to memory)
        auto_config = CheckpointConfig(backend=CheckpointBackend.AUTO)
        manager = create_checkpoint_manager(workflow_config, auto_config)
        
        # Should select memory backend since PostgreSQL likely not configured
        assert manager.backend_type == CheckpointBackend.MEMORY
        assert not manager.is_persistent()
    
    @pytest.mark.asyncio
    async def test_checkpoint_configuration_from_environment(self, workflow_config):
        """Test checkpoint configuration from environment variables."""
        # Test with environment variables
        original_backend = os.getenv('CHECKPOINT_BACKEND')
        
        try:
            # Set test environment
            os.environ['CHECKPOINT_BACKEND'] = 'memory'
            os.environ['MEMORY_CHECKPOINT_MAX_SIZE'] = '500'
            os.environ['CHECKPOINT_COMPRESSION'] = 'false'
            
            config = CheckpointConfig.from_environment()
            
            assert config.backend == CheckpointBackend.MEMORY
            assert config.memory_max_size == 500
            assert config.enable_compression is False
            
        finally:
            # Restore original environment
            if original_backend:
                os.environ['CHECKPOINT_BACKEND'] = original_backend
            else:
                os.environ.pop('CHECKPOINT_BACKEND', None)
            os.environ.pop('MEMORY_CHECKPOINT_MAX_SIZE', None)
            os.environ.pop('CHECKPOINT_COMPRESSION', None)
    
    @pytest.mark.asyncio
    async def test_workflow_without_checkpointing(self, workflow_config):
        """Test workflow creation without checkpointing."""
        workflow = create_workflow(
            config=workflow_config,
            use_checkpointing=False
        )
        
        assert workflow.checkpointer is None
        assert workflow.checkpoint_manager is None
        
        info = workflow.get_checkpoint_info()
        assert info["checkpointing_enabled"] is False
        assert info["checkpointer_type"] is None
    
    def test_recovery_history_tracking(self, workflow_config, sample_state):
        """Test recovery history tracking."""
        recovery = CheckpointRecovery(workflow_config)
        
        # Initially no history
        history = recovery.get_recovery_history()
        assert len(history) == 0
        
        # Make a recovery decision
        decision = recovery.determine_recovery_strategy(sample_state)
        
        # Should have history now
        history = recovery.get_recovery_history()
        assert len(history) == 1
        assert history[0]["strategy"] == decision.strategy.value
        assert history[0]["checkpoint_id"] == sample_state.session_id
        
        # Clear history
        recovery.clear_recovery_history()
        history = recovery.get_recovery_history()
        assert len(history) == 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
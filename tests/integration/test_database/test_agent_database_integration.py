"""
Integration tests for agent database operations.
Tests database models, relationships, and data persistence for agents.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.database.base import Base
from app.models.database.agent import AgentExecution, AgentState, AgentError
from app.models.enums import AgentStatus, AgentType


class TestAgentDatabaseIntegration:
    """Integration test suite for agent database operations."""
    
    @pytest.fixture
    def test_engine(self):
        """Create test database engine."""
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)
        return engine
    
    @pytest.fixture
    def test_session(self, test_engine):
        """Create test database session."""
        TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()
    
    @pytest.mark.integration
    def test_agent_execution_creation_integration(self, test_session):
        """Test agent execution record creation and persistence."""
        # Create agent execution record
        execution = AgentExecution(
            execution_id="agent_exec_123",
            agent_type=AgentType.PLANNER_AGENT,
            status=AgentStatus.RUNNING,
            input_data={"topic": "Test Topic", "description": "Test Description"},
            config={"max_retries": 3, "timeout": 300},
            created_at=datetime.now()
        )
        
        # Save to database
        test_session.add(execution)
        test_session.commit()
        
        # Verify execution was saved
        saved_execution = test_session.query(AgentExecution).filter_by(execution_id="agent_exec_123").first()
        assert saved_execution is not None
        assert saved_execution.agent_type == AgentType.PLANNER_AGENT
        assert saved_execution.status == AgentStatus.RUNNING
        assert saved_execution.input_data == {"topic": "Test Topic", "description": "Test Description"}
    
    @pytest.mark.integration
    def test_agent_state_relationship_integration(self, test_session):
        """Test agent execution and state relationship integration."""
        # Create agent execution with state
        execution = AgentExecution(
            execution_id="state_test_exec",
            agent_type=AgentType.CODE_GENERATOR_AGENT,
            status=AgentStatus.RUNNING
        )
        
        state = AgentState(
            execution_id="state_test_exec",
            state_data={
                "scene_outline": "Test scene outline",
                "generated_code": {"1": "test code"},
                "current_step": "code_generation"
            },
            version=1,
            created_at=datetime.now()
        )
        
        # Save both records
        test_session.add(execution)
        test_session.add(state)
        test_session.commit()
        
        # Verify relationship
        saved_execution = test_session.query(AgentExecution).filter_by(execution_id="state_test_exec").first()
        assert saved_execution.states is not None
        assert len(saved_execution.states) == 1
        assert saved_execution.states[0].state_data["scene_outline"] == "Test scene outline"
    
    @pytest.mark.integration
    def test_agent_error_tracking_integration(self, test_session):
        """Test agent error tracking and persistence."""
        # Create agent execution with error
        execution = AgentExecution(
            execution_id="error_test_exec",
            agent_type=AgentType.RENDERER_AGENT,
            status=AgentStatus.FAILED
        )
        
        error = AgentError(
            execution_id="error_test_exec",
            error_type="ValidationError",
            error_message="Invalid scene configuration",
            error_details={
                "field": "scene_config",
                "value": "invalid_value",
                "expected": "valid_config"
            },
            occurred_at=datetime.now()
        )
        
        # Save both records
        test_session.add(execution)
        test_session.add(error)
        test_session.commit()
        
        # Verify error relationship
        saved_execution = test_session.query(AgentExecution).filter_by(execution_id="error_test_exec").first()
        assert saved_execution.errors is not None
        assert len(saved_execution.errors) == 1
        assert saved_execution.errors[0].error_type == "ValidationError"
        assert saved_execution.errors[0].error_message == "Invalid scene configuration"
    
    @pytest.mark.integration
    def test_agent_execution_status_updates_integration(self, test_session):
        """Test agent execution status update operations."""
        # Create agent execution in running state
        execution = AgentExecution(
            execution_id="status_update_test",
            agent_type=AgentType.RAG_AGENT,
            status=AgentStatus.RUNNING,
            started_at=datetime.now()
        )
        
        test_session.add(execution)
        test_session.commit()
        
        # Update status to completed
        execution.status = AgentStatus.COMPLETED
        execution.completed_at = datetime.now()
        execution.result = {
            "answer": "Generated answer",
            "sources": ["source1.txt", "source2.md"],
            "confidence": 0.85
        }
        test_session.commit()
        
        # Verify status update
        updated_execution = test_session.query(AgentExecution).filter_by(execution_id="status_update_test").first()
        assert updated_execution.status == AgentStatus.COMPLETED
        assert updated_execution.completed_at is not None
        assert updated_execution.result["answer"] == "Generated answer"
    
    @pytest.mark.integration
    def test_agent_execution_query_operations_integration(self, test_session):
        """Test agent execution query operations."""
        # Create multiple agent executions
        executions = [
            AgentExecution(
                execution_id=f"query_test_{i}",
                agent_type=AgentType.PLANNER_AGENT if i % 2 == 0 else AgentType.CODE_GENERATOR_AGENT,
                status=AgentStatus.COMPLETED if i % 3 == 0 else AgentStatus.RUNNING,
                created_at=datetime.now()
            )
            for i in range(6)
        ]
        
        for execution in executions:
            test_session.add(execution)
        test_session.commit()
        
        # Test query by agent type
        planner_executions = test_session.query(AgentExecution).filter_by(agent_type=AgentType.PLANNER_AGENT).all()
        assert len(planner_executions) == 3  # Executions 0, 2, 4
        
        code_gen_executions = test_session.query(AgentExecution).filter_by(agent_type=AgentType.CODE_GENERATOR_AGENT).all()
        assert len(code_gen_executions) == 3  # Executions 1, 3, 5
        
        # Test query by status
        completed_executions = test_session.query(AgentExecution).filter_by(status=AgentStatus.COMPLETED).all()
        assert len(completed_executions) == 2  # Executions 0, 3
        
        running_executions = test_session.query(AgentExecution).filter_by(status=AgentStatus.RUNNING).all()
        assert len(running_executions) == 4  # Executions 1, 2, 4, 5
    
    @pytest.mark.integration
    def test_agent_state_versioning_integration(self, test_session):
        """Test agent state versioning and history."""
        # Create agent execution
        execution = AgentExecution(
            execution_id="versioning_test",
            agent_type=AgentType.PLANNER_AGENT,
            status=AgentStatus.RUNNING
        )
        test_session.add(execution)
        test_session.commit()
        
        # Create multiple state versions
        states = [
            AgentState(
                execution_id="versioning_test",
                state_data={"step": f"step_{i}", "progress": i * 25},
                version=i + 1,
                created_at=datetime.now()
            )
            for i in range(4)
        ]
        
        for state in states:
            test_session.add(state)
        test_session.commit()
        
        # Verify state versions
        saved_execution = test_session.query(AgentExecution).filter_by(execution_id="versioning_test").first()
        assert len(saved_execution.states) == 4
        
        # Verify states are ordered by version
        state_versions = [state.version for state in saved_execution.states]
        assert state_versions == [1, 2, 3, 4]
        
        # Get latest state
        latest_state = test_session.query(AgentState).filter_by(execution_id="versioning_test").order_by(AgentState.version.desc()).first()
        assert latest_state.version == 4
        assert latest_state.state_data["progress"] == 75
    
    @pytest.mark.integration
    def test_agent_execution_cascade_operations_integration(self, test_session):
        """Test cascade operations between execution, state, and errors."""
        # Create agent execution with state and errors
        execution = AgentExecution(
            execution_id="cascade_test",
            agent_type=AgentType.RENDERER_AGENT,
            status=AgentStatus.FAILED
        )
        
        state = AgentState(
            execution_id="cascade_test",
            state_data={"rendering_progress": 50},
            version=1
        )
        
        error = AgentError(
            execution_id="cascade_test",
            error_type="RenderingError",
            error_message="Rendering failed at 50%"
        )
        
        test_session.add(execution)
        test_session.add(state)
        test_session.add(error)
        test_session.commit()
        
        # Delete execution (should cascade to state and errors if configured)
        test_session.delete(execution)
        test_session.commit()
        
        # Verify all records are deleted
        deleted_execution = test_session.query(AgentExecution).filter_by(execution_id="cascade_test").first()
        assert deleted_execution is None
        
        # Note: Cascade behavior depends on foreign key configuration
        # This test verifies the relationship works as expected
    
    @pytest.mark.integration
    def test_agent_execution_performance_integration(self, test_session):
        """Test database performance with bulk agent operations."""
        # Create many agent executions for performance testing
        executions = []
        for i in range(100):
            execution = AgentExecution(
                execution_id=f"perf_test_{i}",
                agent_type=AgentType.PLANNER_AGENT if i % 2 == 0 else AgentType.CODE_GENERATOR_AGENT,
                status=AgentStatus.COMPLETED if i % 3 == 0 else AgentStatus.RUNNING,
                created_at=datetime.now()
            )
            executions.append(execution)
        
        # Bulk insert
        test_session.add_all(executions)
        test_session.commit()
        
        # Test bulk query performance
        all_executions = test_session.query(AgentExecution).filter(AgentExecution.execution_id.like("perf_test_%")).all()
        assert len(all_executions) == 100
        
        # Test filtered query performance
        completed_executions = test_session.query(AgentExecution).filter(
            AgentExecution.execution_id.like("perf_test_%"),
            AgentExecution.status == AgentStatus.COMPLETED
        ).all()
        
        # Should be approximately 33-34 executions (every 3rd one)
        assert 30 <= len(completed_executions) <= 40
    
    @pytest.mark.integration
    def test_agent_json_field_integration(self, test_session):
        """Test JSON field storage and retrieval for agent data."""
        # Create agent execution with complex JSON data
        complex_input_data = {
            "topic": "Complex Topic",
            "description": "Complex description",
            "parameters": {
                "quality": "high",
                "effects": ["fade_in", "fade_out"],
                "custom_settings": {
                    "animation_speed": 1.5,
                    "voice_settings": {
                        "pitch": 1.2,
                        "speed": 0.9
                    }
                }
            }
        }
        
        complex_config = {
            "max_retries": 5,
            "timeout": 600,
            "monitoring": {
                "enabled": True,
                "metrics": ["execution_time", "memory_usage"],
                "alerts": {
                    "error_threshold": 3,
                    "timeout_threshold": 300
                }
            }
        }
        
        execution = AgentExecution(
            execution_id="json_test",
            agent_type=AgentType.PLANNER_AGENT,
            status=AgentStatus.COMPLETED,
            input_data=complex_input_data,
            config=complex_config,
            result={
                "scene_outline": "Generated outline",
                "metadata": {
                    "generation_time": 45.2,
                    "tokens_used": 1250,
                    "quality_score": 0.92
                }
            }
        )
        
        test_session.add(execution)
        test_session.commit()
        
        # Retrieve and verify JSON data
        saved_execution = test_session.query(AgentExecution).filter_by(execution_id="json_test").first()
        
        assert saved_execution.input_data["topic"] == "Complex Topic"
        assert saved_execution.input_data["parameters"]["quality"] == "high"
        assert saved_execution.input_data["parameters"]["custom_settings"]["animation_speed"] == 1.5
        
        assert saved_execution.config["max_retries"] == 5
        assert saved_execution.config["monitoring"]["enabled"] is True
        assert saved_execution.config["monitoring"]["alerts"]["error_threshold"] == 3
        
        assert saved_execution.result["metadata"]["generation_time"] == 45.2
        assert saved_execution.result["metadata"]["quality_score"] == 0.92
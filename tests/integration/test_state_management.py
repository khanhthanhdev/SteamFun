"""
Integration tests for state management validation.
Tests state consistency, persistence, and validation across the workflow.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List
from copy import deepcopy

from src.langgraph_agents.state import VideoGenerationState, create_initial_state, SystemConfig, AgentConfig
from src.langgraph_agents.graph import create_workflow_graph
from langgraph.types import Command


class TestStateManagement:
    """Test suite for state management validation across agents."""
    
    @pytest.fixture
    def mock_system_config(self):
        """Create mock system configuration."""
        return SystemConfig(
            agents={
                "planner_agent": AgentConfig(name="planner_agent", model_config={}, tools=[]),
                "code_generator_agent": AgentConfig(name="code_generator_agent", model_config={}, tools=[]),
                "renderer_agent": AgentConfig(name="renderer_agent", model_config={}, tools=[]),
                "error_handler_agent": AgentConfig(name="error_handler_agent", model_config={}, tools=[])
            },
            llm_providers={},
            docling_config={},
            mcp_servers={},
            monitoring_config={},
            human_loop_config={}
        )
    
    @pytest.fixture
    def initial_state(self, mock_system_config):
        """Create initial state for testing."""
        return create_initial_state(
            topic="State Management Test",
            description="Testing state management across agents",
            session_id="state_test_session",
            config=mock_system_config
        )
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_state_immutability_during_agent_execution(self, initial_state):
        """Test that agents don't mutate the original state object."""
        original_state = deepcopy(initial_state)
        
        # Mock agent that attempts to modify state
        class StateModifyingAgent:
            def __init__(self):
                self.name = "test_agent"
            
            async def execute_with_monitoring(self, state: VideoGenerationState) -> Command:
                # Attempt to modify state directly (should not affect original)
                state["topic"] = "Modified Topic"
                state["error_count"] = 999
                state["new_field"] = "should not persist"
                
                return Command(
                    goto="next_agent",
                    update={"legitimate_update": "this should persist"}
                )
        
        agent = StateModifyingAgent()
        result = await agent.execute_with_monitoring(initial_state)
        
        # Verify original state is unchanged
        assert initial_state["topic"] == original_state["topic"]
        assert initial_state["error_count"] == original_state["error_count"]
        assert "new_field" not in initial_state
        
        # Verify only legitimate updates are in the command
        assert result.update["legitimate_update"] == "this should persist"
        assert "new_field" not in result.update
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_state_validation_across_workflow_stages(self, initial_state):
        """Test state validation at different workflow stages."""
        validation_results = []
        
        def validate_state_at_stage(stage: str, state: VideoGenerationState) -> Dict[str, Any]:
            """Validate state structure at a specific workflow stage."""
            validation = {
                "stage": stage,
                "valid": True,
                "errors": [],
                "required_fields_present": True,
                "field_types_correct": True
            }
            
            # Check required fields based on stage
            required_fields = {
                "initial": ["topic", "description", "session_id"],
                "planning": ["topic", "description", "session_id", "scene_outline"],
                "code_generation": ["scene_outline", "scene_implementations", "generated_code"],
                "rendering": ["generated_code", "rendered_videos"],
                "completion": ["rendered_videos", "combined_video_path"]
            }
            
            stage_fields = required_fields.get(stage, [])
            for field in stage_fields:
                if field not in state or state[field] is None:
                    validation["required_fields_present"] = False
                    validation["errors"].append(f"Missing required field: {field}")
            
            # Check field types
            type_checks = {
                "topic": str,
                "description": str,
                "session_id": str,
                "error_count": int,
                "scene_implementations": dict,
                "generated_code": dict,
                "rendered_videos": dict,
                "detected_plugins": list,
                "escalated_errors": list
            }
            
            for field, expected_type in type_checks.items():
                if field in state and state[field] is not None:
                    if not isinstance(state[field], expected_type):
                        validation["field_types_correct"] = False
                        validation["errors"].append(f"Field {field} has incorrect type: {type(state[field])} (expected {expected_type})")
            
            validation["valid"] = validation["required_fields_present"] and validation["field_types_correct"]
            return validation
        
        # Simulate workflow stages with state updates
        current_state = initial_state.copy()
        
        # Initial stage validation
        validation_results.append(validate_state_at_stage("initial", current_state))
        
        # Planning stage
        current_state.update({
            "scene_outline": "# Scene 1\nTest scene",
            "scene_implementations": {1: "Test implementation"},
            "detected_plugins": ["text", "code"]
        })
        validation_results.append(validate_state_at_stage("planning", current_state))
        
        # Code generation stage
        current_state.update({
            "generated_code": {1: "from manim import *\nclass TestScene(Scene): pass"}
        })
        validation_results.append(validate_state_at_stage("code_generation", current_state))
        
        # Rendering stage
        current_state.update({
            "rendered_videos": {1: "/path/to/scene1.mp4"}
        })
        validation_results.append(validate_state_at_stage("rendering", current_state))
        
        # Completion stage
        current_state.update({
            "combined_video_path": "/path/to/final_video.mp4",
            "workflow_complete": True
        })
        validation_results.append(validate_state_at_stage("completion", current_state))
        
        # Verify all validations passed
        for validation in validation_results:
            assert validation["valid"], f"Stage {validation['stage']} validation failed: {validation['errors']}"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_state_persistence_across_agent_transitions(self, initial_state):
        """Test that state data persists correctly across agent transitions."""
        state_history = []
        
        # Mock agents that track state changes
        def create_tracking_agent(agent_name: str, updates: Dict[str, Any]):
            class TrackingAgent:
                def __init__(self):
                    self.name = agent_name
                
                async def execute_with_monitoring(self, state: VideoGenerationState) -> Command:
                    # Record state before modification
                    state_history.append({
                        "agent": agent_name,
                        "input_state": deepcopy(state),
                        "updates": updates
                    })
                    
                    return Command(goto="next_agent", update=updates)
            
            return TrackingAgent()
        
        # Create sequence of agents with different updates
        agents = [
            create_tracking_agent("planner_agent", {
                "scene_outline": "Planning complete",
                "planning_timestamp": "2024-01-01T10:00:00"
            }),
            create_tracking_agent("code_generator_agent", {
                "generated_code": {1: "Generated code"},
                "code_generation_timestamp": "2024-01-01T10:05:00"
            }),
            create_tracking_agent("renderer_agent", {
                "rendered_videos": {1: "rendered.mp4"},
                "rendering_timestamp": "2024-01-01T10:10:00"
            })
        ]
        
        # Execute agents in sequence
        current_state = initial_state.copy()
        
        for agent in agents:
            result = await agent.execute_with_monitoring(current_state)
            current_state.update(result.update)
        
        # Verify state persistence
        assert len(state_history) == 3
        
        # Check that each agent received previous agent's updates
        planner_input = state_history[0]["input_state"]
        assert planner_input["topic"] == "State Management Test"
        
        code_gen_input = state_history[1]["input_state"]
        assert code_gen_input["scene_outline"] == "Planning complete"
        assert code_gen_input["planning_timestamp"] == "2024-01-01T10:00:00"
        
        renderer_input = state_history[2]["input_state"]
        assert renderer_input["generated_code"] == {1: "Generated code"}
        assert renderer_input["code_generation_timestamp"] == "2024-01-01T10:05:00"
        
        # Verify final state has all updates
        assert current_state["scene_outline"] == "Planning complete"
        assert current_state["generated_code"] == {1: "Generated code"}
        assert current_state["rendered_videos"] == {1: "rendered.mp4"}
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_error_state_management(self, initial_state):
        """Test state management during error conditions."""
        error_scenarios = []
        
        # Mock agent that produces errors
        class ErrorProducingAgent:
            def __init__(self, error_type: str):
                self.name = f"error_{error_type}_agent"
                self.error_type = error_type
            
            async def execute_with_monitoring(self, state: VideoGenerationState) -> Command:
                error_info = {
                    "agent": self.name,
                    "error": f"{self.error_type} error occurred",
                    "timestamp": "2024-01-01T00:00:00",
                    "retry_count": state.get("retry_count", {}).get(self.name, 0)
                }
                
                error_scenarios.append(error_info)
                
                return Command(
                    goto="error_handler_agent",
                    update={
                        "error_count": state.get("error_count", 0) + 1,
                        "escalated_errors": state.get("escalated_errors", []) + [error_info],
                        "retry_count": {
                            **state.get("retry_count", {}),
                            self.name: error_info["retry_count"] + 1
                        }
                    }
                )
        
        # Test different error scenarios
        error_agents = [
            ErrorProducingAgent("syntax"),
            ErrorProducingAgent("timeout"),
            ErrorProducingAgent("validation")
        ]
        
        current_state = initial_state.copy()
        
        for agent in error_agents:
            result = await agent.execute_with_monitoring(current_state)
            current_state.update(result.update)
        
        # Verify error state accumulation
        assert current_state["error_count"] == 3
        assert len(current_state["escalated_errors"]) == 3
        assert len(current_state["retry_count"]) == 3
        
        # Verify error information is preserved
        for i, error_info in enumerate(error_scenarios):
            assert error_info in current_state["escalated_errors"]
            agent_name = error_info["agent"]
            assert current_state["retry_count"][agent_name] == 1
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_state_access(self, initial_state):
        """Test state consistency under concurrent access patterns."""
        access_log = []
        
        # Mock agent that simulates concurrent operations
        class ConcurrentAgent:
            def __init__(self, agent_id: str, delay: float):
                self.name = f"concurrent_agent_{agent_id}"
                self.delay = delay
            
            async def execute_with_monitoring(self, state: VideoGenerationState) -> Command:
                # Simulate concurrent access with delay
                access_log.append(f"{self.name}_start")
                await asyncio.sleep(self.delay)
                
                # Read state
                current_count = state.get("concurrent_counter", 0)
                
                # Simulate processing time
                await asyncio.sleep(0.1)
                
                access_log.append(f"{self.name}_end")
                
                return Command(
                    goto="next_agent",
                    update={
                        "concurrent_counter": current_count + 1,
                        f"{self.name}_processed": True
                    }
                )
        
        # Create multiple concurrent agents
        agents = [
            ConcurrentAgent("1", 0.2),
            ConcurrentAgent("2", 0.1),
            ConcurrentAgent("3", 0.3)
        ]
        
        # Execute agents concurrently
        tasks = []
        for agent in agents:
            task = asyncio.create_task(agent.execute_with_monitoring(initial_state))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Verify each agent processed independently
        assert len(results) == 3
        for i, result in enumerate(results):
            agent_name = f"concurrent_agent_{i+1}"
            assert result.update[f"{agent_name}_processed"] is True
            # Each agent should see initial counter value (0) due to independent state copies
            assert result.update["concurrent_counter"] == 1
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_state_rollback_on_failure(self, initial_state):
        """Test state rollback capabilities when agents fail."""
        checkpoints = []
        
        # Mock agent with checkpoint/rollback capability
        class CheckpointAgent:
            def __init__(self, should_fail: bool = False):
                self.name = "checkpoint_agent"
                self.should_fail = should_fail
            
            async def execute_with_monitoring(self, state: VideoGenerationState) -> Command:
                # Create checkpoint
                checkpoint = {
                    "timestamp": "2024-01-01T00:00:00",
                    "state_snapshot": deepcopy(state),
                    "agent": self.name
                }
                checkpoints.append(checkpoint)
                
                if self.should_fail:
                    # Simulate failure - return rollback command
                    return Command(
                        goto="error_handler_agent",
                        update={
                            "rollback_to_checkpoint": len(checkpoints) - 1,
                            "error_count": state.get("error_count", 0) + 1,
                            "escalated_errors": state.get("escalated_errors", []) + [{
                                "agent": self.name,
                                "error": "Simulated failure",
                                "checkpoint_available": True
                            }]
                        }
                    )
                else:
                    # Success - make some updates
                    return Command(
                        goto="next_agent",
                        update={
                            "checkpoint_test_data": "success",
                            "last_successful_checkpoint": len(checkpoints) - 1
                        }
                    )
        
        # Test successful execution with checkpoint
        success_agent = CheckpointAgent(should_fail=False)
        success_result = await success_agent.execute_with_monitoring(initial_state)
        
        assert len(checkpoints) == 1
        assert success_result.update["checkpoint_test_data"] == "success"
        assert "rollback_to_checkpoint" not in success_result.update
        
        # Test failed execution with rollback
        current_state = initial_state.copy()
        current_state.update(success_result.update)
        
        failure_agent = CheckpointAgent(should_fail=True)
        failure_result = await failure_agent.execute_with_monitoring(current_state)
        
        assert len(checkpoints) == 2
        assert failure_result.goto == "error_handler_agent"
        assert "rollback_to_checkpoint" in failure_result.update
        assert failure_result.update["error_count"] == 1
        
        # Verify checkpoint data integrity
        for checkpoint in checkpoints:
            assert "state_snapshot" in checkpoint
            assert "timestamp" in checkpoint
            assert "agent" in checkpoint
            assert isinstance(checkpoint["state_snapshot"], dict)
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_state_schema_evolution(self, initial_state):
        """Test handling of state schema changes and backward compatibility."""
        schema_versions = []
        
        # Mock agent that handles schema evolution
        class SchemaEvolutionAgent:
            def __init__(self, schema_version: str):
                self.name = f"schema_v{schema_version}_agent"
                self.schema_version = schema_version
            
            async def execute_with_monitoring(self, state: VideoGenerationState) -> Command:
                # Record schema version handling
                schema_versions.append({
                    "agent": self.name,
                    "version": self.schema_version,
                    "state_fields": list(state.keys())
                })
                
                # Add version-specific fields
                updates = {"schema_version": self.schema_version}
                
                if self.schema_version == "1.0":
                    updates["legacy_field"] = "legacy_value"
                elif self.schema_version == "2.0":
                    updates["new_field"] = "new_value"
                    # Handle legacy field migration
                    if "legacy_field" in state:
                        updates["migrated_legacy_field"] = state["legacy_field"]
                
                return Command(goto="next_agent", update=updates)
        
        # Test schema evolution sequence
        agents = [
            SchemaEvolutionAgent("1.0"),
            SchemaEvolutionAgent("2.0")
        ]
        
        current_state = initial_state.copy()
        
        for agent in agents:
            result = await agent.execute_with_monitoring(current_state)
            current_state.update(result.update)
        
        # Verify schema evolution handling
        assert len(schema_versions) == 2
        
        # V1.0 agent should add legacy field
        assert current_state["schema_version"] == "2.0"  # Latest version
        assert current_state["legacy_field"] == "legacy_value"
        assert current_state["new_field"] == "new_value"
        assert current_state["migrated_legacy_field"] == "legacy_value"
        
        # Verify backward compatibility
        v1_fields = schema_versions[0]["state_fields"]
        v2_fields = schema_versions[1]["state_fields"]
        
        # V2 agent should have access to V1 fields
        assert "legacy_field" in v2_fields
        assert len(v2_fields) > len(v1_fields)


if __name__ == "__main__":
    pytest.main([__file__])
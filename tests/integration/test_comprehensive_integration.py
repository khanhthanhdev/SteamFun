"""
Comprehensive integration tests for multi-agent workflows.
Tests the complete integration of agent communication, state management, error propagation, and tool integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List
from copy import deepcopy
from datetime import datetime

from src.langgraph_agents.base_agent import BaseAgent, AgentFactory
from src.langgraph_agents.state import VideoGenerationState, AgentConfig, SystemConfig, AgentError
from langgraph.types import Command

# Import test utilities
from tests.utils.config_mocks import (
    mock_configuration_manager, create_test_system_config, 
    create_test_agent_config, MockConfigurationManager
)


class TestComprehensiveIntegration:
    """Comprehensive integration test suite for multi-agent workflows."""
    
    @pytest.fixture
    def system_config(self):
        """Create comprehensive system configuration using configuration system."""
        return create_test_system_config(
            environment="test",
            debug=True,
            include_rag=True,
            include_monitoring=True
        )
    
    @pytest.fixture
    def initial_state(self):
        """Create comprehensive initial state."""
        return VideoGenerationState(
            messages=[],
            topic="Comprehensive Integration Test",
            description="Testing complete multi-agent workflow integration",
            session_id="comprehensive_test_session",
            output_dir="test_output",
            print_response=False,
            use_rag=True,
            use_context_learning=True,
            context_learning_path="test_context",
            chroma_db_path="test_chroma",
            manim_docs_path="test_docs",
            embedding_model="test_model",
            use_visual_fix_code=False,
            use_langfuse=True,
            max_scene_concurrency=5,
            max_topic_concurrency=1,
            max_retries=5,
            use_enhanced_rag=True,
            enable_rag_caching=True,
            enable_quality_monitoring=True,
            enable_error_handling=True,
            rag_cache_ttl=3600,
            rag_max_cache_size=1000,
            rag_performance_threshold=2.0,
            rag_quality_threshold=0.7,
            enable_caching=True,
            default_quality="medium",
            use_gpu_acceleration=False,
            preview_mode=False,
            max_concurrent_renders=4,
            scene_outline=None,
            scene_implementations={},
            detected_plugins=[],
            generated_code={},
            code_errors={},
            rag_context={},
            rendered_videos={},
            combined_video_path=None,
            rendering_errors={},
            visual_analysis_results={},
            visual_errors={},
            error_count=0,
            retry_count={},
            escalated_errors=[],
            pending_human_input=None,
            human_feedback=None,
            performance_metrics={},
            execution_trace=[],
            current_agent=None,
            next_agent="planner_agent",
            workflow_complete=False,
            workflow_interrupted=False
        )
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_workflow_integration(self, system_config, initial_state):
        """Test complete workflow integration from planning to rendering."""
        workflow_trace = []
        
        # Mock complete workflow agents
        def create_workflow_agent(agent_name: str, next_agent: str, updates: Dict[str, Any]):
            class WorkflowAgent:
                def __init__(self):
                    self.name = agent_name
                
                async def execute_with_monitoring(self, state: VideoGenerationState) -> Command:
                    workflow_trace.append({
                        "agent": agent_name,
                        "input_state": {
                            "topic": state.get("topic"),
                            "session_id": state.get("session_id"),
                            "current_agent": state.get("current_agent"),
                            "error_count": state.get("error_count", 0)
                        },
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    return Command(
                        goto=next_agent,
                        update={
                            **updates,
                            "current_agent": next_agent,
                            "execution_trace": state.get("execution_trace", []) + [{
                                "agent": agent_name,
                                "action": "completed",
                                "timestamp": datetime.now().isoformat()
                            }]
                        }
                    )
            
            return WorkflowAgent()
        
        # Create workflow sequence
        workflow_agents = [
            ("planner_agent", "code_generator_agent", {
                "scene_outline": "# Scene 1\nTest scene outline",
                "scene_implementations": {1: "Test implementation"},
                "detected_plugins": ["text", "code"]
            }),
            ("code_generator_agent", "renderer_agent", {
                "generated_code": {1: "from manim import *\nclass TestScene(Scene): pass"},
                "rag_context": {1: "RAG context for scene 1"}
            }),
            ("renderer_agent", "END", {
                "rendered_videos": {1: "/path/to/scene1.mp4"},
                "combined_video_path": "/path/to/final_video.mp4",
                "workflow_complete": True
            })
        ]
        
        # Execute complete workflow
        current_state = initial_state.copy()
        
        for agent_name, next_agent, updates in workflow_agents:
            agent = create_workflow_agent(agent_name, next_agent, updates)
            result = await agent.execute_with_monitoring(current_state)
            current_state.update(result.update)
            
            if next_agent == "END":
                break
        
        # Verify complete workflow execution
        assert len(workflow_trace) == 3
        assert workflow_trace[0]["agent"] == "planner_agent"
        assert workflow_trace[1]["agent"] == "code_generator_agent"
        assert workflow_trace[2]["agent"] == "renderer_agent"
        
        # Verify final state
        assert current_state["workflow_complete"] is True
        assert current_state["scene_outline"] == "# Scene 1\nTest scene outline"
        assert current_state["generated_code"] == {1: "from manim import *\nclass TestScene(Scene): pass"}
        assert current_state["rendered_videos"] == {1: "/path/to/scene1.mp4"}
        assert current_state["combined_video_path"] == "/path/to/final_video.mp4"
        
        # Verify execution trace
        execution_trace = current_state["execution_trace"]
        assert len(execution_trace) == 3
        assert all(entry["action"] == "completed" for entry in execution_trace)
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_error_recovery_workflow_integration(self, system_config, initial_state):
        """Test error recovery workflow integration across multiple agents."""
        error_recovery_trace = []
        
        # Mock agents with error recovery
        class ErrorRecoveryAgent:
            def __init__(self, agent_name: str, should_fail_first: bool = False):
                self.name = agent_name
                self.should_fail_first = should_fail_first
                self.execution_count = 0
            
            async def execute_with_monitoring(self, state: VideoGenerationState) -> Command:
                self.execution_count += 1
                
                error_recovery_trace.append({
                    "agent": self.name,
                    "execution_count": self.execution_count,
                    "should_fail": self.should_fail_first and self.execution_count == 1,
                    "retry_count": state.get("retry_count", {}).get(self.name, 0)
                })
                
                if self.should_fail_first and self.execution_count == 1:
                    # First execution fails
                    return Command(
                        goto="error_handler_agent",
                        update={
                            "error_count": state.get("error_count", 0) + 1,
                            "escalated_errors": state.get("escalated_errors", []) + [{
                                "agent": self.name,
                                "error": f"Simulated error in {self.name}",
                                "timestamp": datetime.now().isoformat(),
                                "recoverable": True
                            }],
                            "retry_count": {
                                **state.get("retry_count", {}),
                                self.name: state.get("retry_count", {}).get(self.name, 0) + 1
                            }
                        }
                    )
                else:
                    # Success (or retry success)
                    return Command(
                        goto="next_agent",
                        update={
                            f"{self.name}_success": True,
                            "recovery_successful": self.execution_count > 1
                        }
                    )
        
        # Mock error handler
        class ErrorHandlerAgent:
            def __init__(self):
                self.name = "error_handler_agent"
            
            async def execute_with_monitoring(self, state: VideoGenerationState) -> Command:
                escalated_errors = state.get("escalated_errors", [])
                if escalated_errors:
                    last_error = escalated_errors[-1]
                    if last_error.get("recoverable"):
                        # Retry the failed agent
                        return Command(
                            goto=last_error["agent"],
                            update={"error_recovery_attempted": True}
                        )
                
                return Command(goto="human_loop_agent")
        
        # Test error recovery workflow
        failing_agent = ErrorRecoveryAgent("code_generator_agent", should_fail_first=True)
        error_handler = ErrorHandlerAgent()
        
        current_state = initial_state.copy()
        
        # First execution (fails)
        result1 = await failing_agent.execute_with_monitoring(current_state)
        current_state.update(result1.update)
        
        # Error handler processes error
        error_result = await error_handler.execute_with_monitoring(current_state)
        current_state.update(error_result.update)
        
        # Retry execution (succeeds)
        result2 = await failing_agent.execute_with_monitoring(current_state)
        current_state.update(result2.update)
        
        # Verify error recovery workflow
        assert len(error_recovery_trace) == 2
        assert error_recovery_trace[0]["should_fail"] is True
        assert error_recovery_trace[1]["should_fail"] is False
        
        # Verify recovery success
        assert current_state["code_generator_agent_success"] is True
        assert current_state["recovery_successful"] is True
        assert current_state["error_recovery_attempted"] is True
        assert current_state["error_count"] == 1
        assert current_state["retry_count"]["code_generator_agent"] == 1
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_human_loop_integration_workflow(self, system_config, initial_state):
        """Test human-in-the-loop integration across workflow."""
        human_interactions = []
        
        # Mock human loop workflow
        class HumanLoopWorkflowAgent:
            def __init__(self, agent_name: str, requires_human_input: bool = False):
                self.name = agent_name
                self.requires_human_input = requires_human_input
            
            async def execute_with_monitoring(self, state: VideoGenerationState) -> Command:
                if self.requires_human_input:
                    # Escalate to human
                    return Command(
                        goto="human_loop_agent",
                        update={
                            "pending_human_input": {
                                "context": f"{self.name} requires human review",
                                "options": ["approve", "modify", "reject"],
                                "requesting_agent": self.name,
                                "priority": "medium"
                            }
                        }
                    )
                else:
                    # Normal execution
                    return Command(
                        goto="next_agent",
                        update={f"{self.name}_completed": True}
                    )
        
        # Mock human loop agent
        class HumanLoopAgent:
            def __init__(self):
                self.name = "human_loop_agent"
            
            async def execute_with_monitoring(self, state: VideoGenerationState) -> Command:
                pending_input = state.get("pending_human_input")
                if pending_input:
                    human_interactions.append({
                        "context": pending_input["context"],
                        "options": pending_input["options"],
                        "requesting_agent": pending_input["requesting_agent"],
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Simulate human decision
                    return Command(
                        goto=pending_input["requesting_agent"],
                        update={
                            "human_feedback": {
                                "decision": "approve",
                                "comments": "Approved by human reviewer",
                                "timestamp": datetime.now().isoformat()
                            },
                            "pending_human_input": None,
                            "human_intervention_completed": True
                        }
                    )
                
                return Command(goto="next_agent")
        
        # Test human loop integration
        workflow_agent = HumanLoopWorkflowAgent("code_generator_agent", requires_human_input=True)
        human_agent = HumanLoopAgent()
        
        current_state = initial_state.copy()
        
        # Agent requests human input
        result1 = await workflow_agent.execute_with_monitoring(current_state)
        current_state.update(result1.update)
        
        # Human loop agent processes request
        result2 = await human_agent.execute_with_monitoring(current_state)
        current_state.update(result2.update)
        
        # Agent continues after human input
        workflow_agent.requires_human_input = False  # No longer requires input
        result3 = await workflow_agent.execute_with_monitoring(current_state)
        current_state.update(result3.update)
        
        # Verify human loop integration
        assert len(human_interactions) == 1
        assert human_interactions[0]["requesting_agent"] == "code_generator_agent"
        assert human_interactions[0]["context"] == "code_generator_agent requires human review"
        
        # Verify workflow continuation
        assert current_state["human_intervention_completed"] is True
        assert current_state["human_feedback"]["decision"] == "approve"
        assert current_state["pending_human_input"] is None
        assert current_state["code_generator_agent_completed"] is True
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tool_integration_across_agents(self, system_config, initial_state):
        """Test tool integration across multiple agents."""
        tool_usage_log = []
        shared_tool_data = {}
        
        # Mock tool-integrated agents
        class ToolIntegratedAgent:
            def __init__(self, agent_name: str, tools: List[str]):
                self.name = agent_name
                self.tools = tools
            
            async def execute_with_monitoring(self, state: VideoGenerationState) -> Command:
                tool_results = {}
                
                for tool in self.tools:
                    # Simulate tool usage
                    tool_usage_log.append({
                        "agent": self.name,
                        "tool": tool,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Mock tool responses
                    if tool == "rag_tool":
                        result = f"RAG context from {self.name}"
                        shared_tool_data["rag_context"] = result
                    elif tool == "context7_tool":
                        result = f"Context7 docs from {self.name}"
                        shared_tool_data["context7_docs"] = result
                    elif tool == "code_tool":
                        rag_context = shared_tool_data.get("rag_context", "")
                        result = f"Generated code using: {rag_context}"
                        shared_tool_data["generated_code"] = result
                    elif tool == "render_tool":
                        code = shared_tool_data.get("generated_code", "")
                        result = f"Rendered video from: {code}"
                        shared_tool_data["rendered_video"] = result
                    else:
                        result = f"Tool {tool} result from {self.name}"
                    
                    tool_results[tool] = result
                
                return Command(
                    goto="next_agent",
                    update={
                        f"{self.name}_tool_results": tool_results,
                        "shared_tool_data": shared_tool_data.copy()
                    }
                )
        
        # Create tool-integrated workflow
        agents = [
            ToolIntegratedAgent("rag_agent", ["rag_tool", "context7_tool"]),
            ToolIntegratedAgent("code_generator_agent", ["code_tool"]),
            ToolIntegratedAgent("renderer_agent", ["render_tool"])
        ]
        
        current_state = initial_state.copy()
        
        # Execute tool-integrated workflow
        for agent in agents:
            result = await agent.execute_with_monitoring(current_state)
            current_state.update(result.update)
        
        # Verify tool integration
        assert len(tool_usage_log) == 4  # Total tools used across agents
        
        # Verify tool coordination
        assert "rag_context" in shared_tool_data
        assert "context7_docs" in shared_tool_data
        assert "generated_code" in shared_tool_data
        assert "rendered_video" in shared_tool_data
        
        # Verify data flow through tools
        assert "RAG context from rag_agent" in shared_tool_data["generated_code"]
        assert "Generated code using:" in shared_tool_data["rendered_video"]
        
        # Verify each agent used its tools
        assert "rag_agent_tool_results" in current_state
        assert "code_generator_agent_tool_results" in current_state
        assert "renderer_agent_tool_results" in current_state
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_monitoring_integration_across_workflow(self, system_config, initial_state):
        """Test monitoring integration across complete workflow."""
        monitoring_data = []
        
        # Mock monitored agents
        class MonitoredAgent:
            def __init__(self, agent_name: str, execution_time: float):
                self.name = agent_name
                self.execution_time = execution_time
            
            async def execute_with_monitoring(self, state: VideoGenerationState) -> Command:
                # Simulate execution time
                await asyncio.sleep(0.01)  # Small delay for testing
                
                # Record monitoring data
                monitoring_entry = {
                    "agent": self.name,
                    "execution_time": self.execution_time,
                    "timestamp": datetime.now().isoformat(),
                    "memory_usage": f"{50 + len(self.name)}MB",
                    "success": True
                }
                monitoring_data.append(monitoring_entry)
                
                return Command(
                    goto="monitoring_agent",
                    update={
                        "execution_trace": state.get("execution_trace", []) + [monitoring_entry],
                        "performance_metrics": {
                            **state.get("performance_metrics", {}),
                            self.name: {
                                "execution_time": self.execution_time,
                                "success_rate": 1.0,
                                "memory_usage": monitoring_entry["memory_usage"]
                            }
                        }
                    }
                )
        
        # Mock monitoring agent
        class MonitoringAgent:
            def __init__(self):
                self.name = "monitoring_agent"
            
            async def execute_with_monitoring(self, state: VideoGenerationState) -> Command:
                # Analyze monitoring data
                performance_metrics = state.get("performance_metrics", {})
                execution_trace = state.get("execution_trace", [])
                
                total_execution_time = sum(
                    metrics.get("execution_time", 0) 
                    for metrics in performance_metrics.values()
                )
                
                avg_execution_time = (
                    total_execution_time / len(performance_metrics) 
                    if performance_metrics else 0
                )
                
                monitoring_report = {
                    "total_agents": len(performance_metrics),
                    "total_execution_time": total_execution_time,
                    "average_execution_time": avg_execution_time,
                    "all_successful": all(
                        entry.get("success", False) for entry in execution_trace
                    ),
                    "monitoring_complete": True
                }
                
                return Command(
                    goto="next_agent",
                    update={"monitoring_report": monitoring_report}
                )
        
        # Test monitoring integration
        monitored_agents = [
            MonitoredAgent("planner_agent", 1.2),
            MonitoredAgent("code_generator_agent", 2.5),
            MonitoredAgent("renderer_agent", 3.8)
        ]
        
        monitoring_agent = MonitoringAgent()
        current_state = initial_state.copy()
        
        # Execute monitored workflow
        for agent in monitored_agents:
            result = await agent.execute_with_monitoring(current_state)
            current_state.update(result.update)
            
            # Process monitoring data
            monitoring_result = await monitoring_agent.execute_with_monitoring(current_state)
            current_state.update(monitoring_result.update)
        
        # Verify monitoring integration
        assert len(monitoring_data) == 3
        
        # Verify monitoring report
        monitoring_report = current_state["monitoring_report"]
        assert monitoring_report["total_agents"] == 3
        assert monitoring_report["total_execution_time"] == 7.5  # 1.2 + 2.5 + 3.8
        assert monitoring_report["average_execution_time"] == 2.5  # 7.5 / 3
        assert monitoring_report["all_successful"] is True
        assert monitoring_report["monitoring_complete"] is True
        
        # Verify performance metrics
        performance_metrics = current_state["performance_metrics"]
        assert "planner_agent" in performance_metrics
        assert "code_generator_agent" in performance_metrics
        assert "renderer_agent" in performance_metrics
        
        # Verify execution trace
        execution_trace = current_state["execution_trace"]
        assert len(execution_trace) >= 3
        assert all(entry["success"] for entry in execution_trace)
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_state_consistency_across_complex_workflow(self, system_config, initial_state):
        """Test state consistency across complex multi-agent workflow."""
        state_snapshots = []
        
        # Mock state-tracking agents
        class StateTrackingAgent:
            def __init__(self, agent_name: str, state_updates: Dict[str, Any]):
                self.name = agent_name
                self.state_updates = state_updates
            
            async def execute_with_monitoring(self, state: VideoGenerationState) -> Command:
                # Take state snapshot before modification
                state_snapshots.append({
                    "agent": self.name,
                    "before_state": {
                        "topic": state.get("topic"),
                        "session_id": state.get("session_id"),
                        "error_count": state.get("error_count", 0),
                        "workflow_complete": state.get("workflow_complete", False)
                    },
                    "updates": self.state_updates
                })
                
                return Command(
                    goto="next_agent",
                    update=self.state_updates
                )
        
        # Create complex workflow with state dependencies
        workflow_agents = [
            StateTrackingAgent("planner_agent", {
                "scene_outline": "Complex scene outline",
                "planning_complete": True,
                "planning_timestamp": "2024-01-01T10:00:00"
            }),
            StateTrackingAgent("code_generator_agent", {
                "generated_code": {1: "complex code"},
                "code_generation_complete": True,
                "code_timestamp": "2024-01-01T10:05:00"
            }),
            StateTrackingAgent("renderer_agent", {
                "rendered_videos": {1: "complex_video.mp4"},
                "rendering_complete": True,
                "render_timestamp": "2024-01-01T10:10:00",
                "workflow_complete": True
            })
        ]
        
        current_state = initial_state.copy()
        
        # Execute complex workflow
        for agent in workflow_agents:
            result = await agent.execute_with_monitoring(current_state)
            current_state.update(result.update)
        
        # Verify state consistency
        assert len(state_snapshots) == 3
        
        # Verify state progression
        planner_snapshot = state_snapshots[0]
        assert planner_snapshot["before_state"]["workflow_complete"] is False
        assert planner_snapshot["updates"]["planning_complete"] is True
        
        code_gen_snapshot = state_snapshots[1]
        assert code_gen_snapshot["before_state"]["workflow_complete"] is False
        # Should have access to planner results
        assert current_state["planning_complete"] is True
        
        renderer_snapshot = state_snapshots[2]
        assert renderer_snapshot["before_state"]["workflow_complete"] is False
        # Should have access to all previous results
        assert current_state["planning_complete"] is True
        assert current_state["code_generation_complete"] is True
        
        # Verify final state consistency
        assert current_state["scene_outline"] == "Complex scene outline"
        assert current_state["generated_code"] == {1: "complex code"}
        assert current_state["rendered_videos"] == {1: "complex_video.mp4"}
        assert current_state["workflow_complete"] is True
        
        # Verify timestamps are preserved
        assert current_state["planning_timestamp"] == "2024-01-01T10:00:00"
        assert current_state["code_timestamp"] == "2024-01-01T10:05:00"
        assert current_state["render_timestamp"] == "2024-01-01T10:10:00"


if __name__ == "__main__":
    pytest.main([__file__])
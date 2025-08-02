"""
Integration tests for error propagation testing.
Tests error handling, recovery workflows, and escalation across agents.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List
from datetime import datetime

from src.langgraph_agents.base_agent import BaseAgent, AgentFactory
from src.langgraph_agents.state import VideoGenerationState, AgentConfig, SystemConfig, AgentError
from src.langgraph_agents.agents.error_handler_agent import ErrorHandlerAgent
from langgraph.types import Command


class TestErrorPropagation:
    """Test suite for error propagation and recovery workflows."""
    
    @pytest.fixture
    def mock_system_config(self):
        """Create mock system configuration with error handling enabled."""
        return SystemConfig(
            agents={
                "planner_agent": AgentConfig(
                    name="planner_agent",
                    model_config={},
                    tools=[],
                    max_retries=3,
                    enable_human_loop=True
                ),
                "code_generator_agent": AgentConfig(
                    name="code_generator_agent",
                    model_config={},
                    tools=[],
                    max_retries=3,
                    enable_human_loop=True
                ),
                "renderer_agent": AgentConfig(
                    name="renderer_agent",
                    model_config={},
                    tools=[],
                    max_retries=2,
                    enable_human_loop=True
                ),
                "error_handler_agent": AgentConfig(
                    name="error_handler_agent",
                    model_config={},
                    tools=["error_analysis", "recovery"],
                    max_retries=1,
                    enable_human_loop=True
                ),
                "human_loop_agent": AgentConfig(
                    name="human_loop_agent",
                    model_config={},
                    tools=["human_interface"],
                    enable_human_loop=True
                )
            },
            llm_providers={},
            docling_config={},
            mcp_servers={},
            monitoring_config={},
            human_loop_config={"enabled": True, "timeout": 300},
            max_workflow_retries=3
        )
    
    @pytest.fixture
    def initial_state(self, mock_system_config):
        """Create initial state for error testing."""
        return VideoGenerationState(
            messages=[],
            topic="Error Propagation Test",
            description="Testing error handling across agents",
            session_id="error_test_session",
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
            next_agent=None,
            workflow_complete=False,
            workflow_interrupted=False
        )
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_single_agent_error_recovery(self, mock_system_config, initial_state):
        """Test error recovery within a single agent."""
        error_recovery_log = []
        
        # Mock agent that fails then recovers
        class RecoverableAgent:
            def __init__(self):
                self.name = "recoverable_agent"
                self.attempt_count = 0
            
            async def execute_with_monitoring(self, state: VideoGenerationState) -> Command:
                self.attempt_count += 1
                error_recovery_log.append(f"Attempt {self.attempt_count}")
                
                if self.attempt_count == 1:
                    # First attempt fails
                    return Command(
                        goto="error_handler_agent",
                        update={
                            "error_count": state.get("error_count", 0) + 1,
                            "escalated_errors": state.get("escalated_errors", []) + [{
                                "agent": self.name,
                                "error": "Temporary failure",
                                "attempt": self.attempt_count,
                                "recoverable": True
                            }],
                            "retry_count": {
                                **state.get("retry_count", {}),
                                self.name: self.attempt_count
                            }
                        }
                    )
                else:
                    # Second attempt succeeds
                    return Command(
                        goto="next_agent",
                        update={
                            "recovery_successful": True,
                            "total_attempts": self.attempt_count
                        }
                    )
        
        # Mock error handler that retries the agent
        class RetryErrorHandler:
            def __init__(self):
                self.name = "error_handler_agent"
            
            async def execute_with_monitoring(self, state: VideoGenerationState) -> Command:
                escalated_errors = state.get("escalated_errors", [])
                if escalated_errors:
                    last_error = escalated_errors[-1]
                    if last_error.get("recoverable") and last_error.get("attempt", 0) < 3:
                        # Retry the failed agent
                        return Command(
                            goto="recoverable_agent",
                            update={"retry_initiated": True}
                        )
                
                return Command(goto="human_loop_agent")
        
        # Test recovery workflow
        recoverable_agent = RecoverableAgent()
        error_handler = RetryErrorHandler()
        
        # First execution (fails)
        result1 = await recoverable_agent.execute_with_monitoring(initial_state)
        assert result1.goto == "error_handler_agent"
        assert result1.update["error_count"] == 1
        
        # Error handler processes error
        error_state = initial_state.copy()
        error_state.update(result1.update)
        
        error_result = await error_handler.execute_with_monitoring(error_state)
        assert error_result.goto == "recoverable_agent"
        assert error_result.update["retry_initiated"] is True
        
        # Retry execution (succeeds)
        retry_state = error_state.copy()
        retry_state.update(error_result.update)
        
        result2 = await recoverable_agent.execute_with_monitoring(retry_state)
        assert result2.goto == "next_agent"
        assert result2.update["recovery_successful"] is True
        assert result2.update["total_attempts"] == 2
        
        # Verify recovery log
        assert len(error_recovery_log) == 2
        assert "Attempt 1" in error_recovery_log
        assert "Attempt 2" in error_recovery_log
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_cascading_error_propagation(self, mock_system_config, initial_state):
        """Test error propagation across multiple agents."""
        error_cascade = []
        
        # Create agents that fail in sequence
        def create_failing_agent(agent_name: str, error_type: str, should_fail: bool = True):
            class FailingAgent:
                def __init__(self):
                    self.name = agent_name
                
                async def execute_with_monitoring(self, state: VideoGenerationState) -> Command:
                    error_cascade.append({
                        "agent": agent_name,
                        "error_type": error_type,
                        "state_error_count": state.get("error_count", 0)
                    })
                    
                    if should_fail:
                        return Command(
                            goto="error_handler_agent",
                            update={
                                "error_count": state.get("error_count", 0) + 1,
                                "escalated_errors": state.get("escalated_errors", []) + [{
                                    "agent": agent_name,
                                    "error": f"{error_type} in {agent_name}",
                                    "timestamp": datetime.now().isoformat(),
                                    "cascading": True
                                }]
                            }
                        )
                    else:
                        return Command(goto="next_agent", update={"success": True})
            
            return FailingAgent()
        
        # Create error handler that tracks cascading failures
        class CascadeTrackingErrorHandler:
            def __init__(self):
                self.name = "error_handler_agent"
                self.cascade_count = 0
            
            async def execute_with_monitoring(self, state: VideoGenerationState) -> Command:
                self.cascade_count += 1
                error_count = state.get("error_count", 0)
                
                if error_count >= 3:  # Too many cascading errors
                    return Command(
                        goto="human_loop_agent",
                        update={
                            "cascade_detected": True,
                            "cascade_count": self.cascade_count,
                            "escalation_reason": "cascading_failures"
                        }
                    )
                else:
                    # Try to continue workflow
                    return Command(
                        goto="next_failing_agent",
                        update={"error_handled": True}
                    )
        
        # Test cascading failure scenario
        agents = [
            create_failing_agent("planner_agent", "planning_error"),
            create_failing_agent("code_generator_agent", "code_error"),
            create_failing_agent("renderer_agent", "rendering_error")
        ]
        
        error_handler = CascadeTrackingErrorHandler()
        current_state = initial_state.copy()
        
        # Execute agents in sequence, each failing
        for i, agent in enumerate(agents):
            result = await agent.execute_with_monitoring(current_state)
            current_state.update(result.update)
            
            # Process error
            error_result = await error_handler.execute_with_monitoring(current_state)
            current_state.update(error_result.update)
            
            # Check if cascade threshold reached
            if error_result.goto == "human_loop_agent":
                break
        
        # Verify cascading error detection
        assert len(error_cascade) == 3
        assert current_state["error_count"] == 3
        assert current_state["cascade_detected"] is True
        assert current_state["escalation_reason"] == "cascading_failures"
        
        # Verify each error was recorded
        escalated_errors = current_state["escalated_errors"]
        assert len(escalated_errors) == 3
        for error in escalated_errors:
            assert error["cascading"] is True
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_error_recovery_strategies(self, mock_system_config, initial_state):
        """Test different error recovery strategies."""
        recovery_strategies_used = []
        
        # Mock error handler with multiple recovery strategies
        class StrategicErrorHandler:
            def __init__(self):
                self.name = "error_handler_agent"
            
            async def execute_with_monitoring(self, state: VideoGenerationState) -> Command:
                escalated_errors = state.get("escalated_errors", [])
                if not escalated_errors:
                    return Command(goto="next_agent")
                
                last_error = escalated_errors[-1]
                error_type = last_error.get("error_type", "unknown")
                retry_count = state.get("retry_count", {}).get(last_error["agent"], 0)
                
                # Choose recovery strategy based on error type and retry count
                if error_type == "syntax_error" and retry_count < 2:
                    strategy = "retry_with_rag"
                    recovery_strategies_used.append(strategy)
                    return Command(
                        goto=last_error["agent"],
                        update={
                            "recovery_strategy": strategy,
                            "use_enhanced_rag": True,
                            "retry_count": {
                                **state.get("retry_count", {}),
                                last_error["agent"]: retry_count + 1
                            }
                        }
                    )
                elif error_type == "timeout_error":
                    strategy = "reduce_quality"
                    recovery_strategies_used.append(strategy)
                    return Command(
                        goto=last_error["agent"],
                        update={
                            "recovery_strategy": strategy,
                            "default_quality": "low",
                            "timeout_seconds": state.get("timeout_seconds", 300) * 2
                        }
                    )
                elif error_type == "resource_error":
                    strategy = "sequential_processing"
                    recovery_strategies_used.append(strategy)
                    return Command(
                        goto=last_error["agent"],
                        update={
                            "recovery_strategy": strategy,
                            "max_scene_concurrency": 1,
                            "max_concurrent_renders": 1
                        }
                    )
                else:
                    strategy = "human_escalation"
                    recovery_strategies_used.append(strategy)
                    return Command(
                        goto="human_loop_agent",
                        update={
                            "recovery_strategy": strategy,
                            "escalation_reason": f"no_strategy_for_{error_type}"
                        }
                    )
        
        # Create agents with different error types
        def create_error_agent(error_type: str):
            class ErrorAgent:
                def __init__(self):
                    self.name = f"{error_type}_agent"
                    self.execution_count = 0
                
                async def execute_with_monitoring(self, state: VideoGenerationState) -> Command:
                    self.execution_count += 1
                    
                    # Check if recovery strategy was applied
                    recovery_strategy = state.get("recovery_strategy")
                    if recovery_strategy and self.execution_count > 1:
                        # Recovery strategy applied, succeed this time
                        return Command(
                            goto="next_agent",
                            update={
                                "recovery_successful": True,
                                "strategy_used": recovery_strategy
                            }
                        )
                    else:
                        # First execution or no recovery, fail
                        return Command(
                            goto="error_handler_agent",
                            update={
                                "error_count": state.get("error_count", 0) + 1,
                                "escalated_errors": state.get("escalated_errors", []) + [{
                                    "agent": self.name,
                                    "error_type": error_type,
                                    "error": f"{error_type} occurred",
                                    "execution_count": self.execution_count
                                }]
                            }
                        )
            
            return ErrorAgent()
        
        # Test different recovery strategies
        error_handler = StrategicErrorHandler()
        test_scenarios = [
            ("syntax_error", "retry_with_rag"),
            ("timeout_error", "reduce_quality"),
            ("resource_error", "sequential_processing")
        ]
        
        for error_type, expected_strategy in test_scenarios:
            current_state = initial_state.copy()
            error_agent = create_error_agent(error_type)
            
            # First execution (fails)
            result1 = await error_agent.execute_with_monitoring(current_state)
            current_state.update(result1.update)
            
            # Error handler applies recovery strategy
            error_result = await error_handler.execute_with_monitoring(current_state)
            current_state.update(error_result.update)
            
            # Retry with recovery strategy (succeeds)
            result2 = await error_agent.execute_with_monitoring(current_state)
            
            # Verify recovery
            assert result2.update["recovery_successful"] is True
            assert result2.update["strategy_used"] == expected_strategy
        
        # Verify all strategies were used
        assert "retry_with_rag" in recovery_strategies_used
        assert "reduce_quality" in recovery_strategies_used
        assert "sequential_processing" in recovery_strategies_used
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_human_escalation_workflow(self, mock_system_config, initial_state):
        """Test human escalation when automatic recovery fails."""
        escalation_log = []
        
        # Mock agent that consistently fails
        class PersistentFailureAgent:
            def __init__(self):
                self.name = "persistent_failure_agent"
                self.attempt_count = 0
            
            async def execute_with_monitoring(self, state: VideoGenerationState) -> Command:
                self.attempt_count += 1
                escalation_log.append(f"Failure attempt {self.attempt_count}")
                
                return Command(
                    goto="error_handler_agent",
                    update={
                        "error_count": state.get("error_count", 0) + 1,
                        "escalated_errors": state.get("escalated_errors", []) + [{
                            "agent": self.name,
                            "error": "Persistent failure",
                            "attempt": self.attempt_count,
                            "recoverable": False
                        }],
                        "retry_count": {
                            **state.get("retry_count", {}),
                            self.name: self.attempt_count
                        }
                    }
                )
        
        # Mock error handler that escalates after max retries
        class EscalatingErrorHandler:
            def __init__(self):
                self.name = "error_handler_agent"
            
            async def execute_with_monitoring(self, state: VideoGenerationState) -> Command:
                retry_count = state.get("retry_count", {}).get("persistent_failure_agent", 0)
                
                if retry_count >= 3:  # Max retries reached
                    escalation_log.append("Escalating to human")
                    return Command(
                        goto="human_loop_agent",
                        update={
                            "pending_human_input": {
                                "context": "Agent has failed multiple times and requires human intervention",
                                "options": ["retry_with_modifications", "skip_agent", "abort_workflow"],
                                "requesting_agent": "error_handler_agent",
                                "failure_count": retry_count
                            }
                        }
                    )
                else:
                    # Try one more time
                    escalation_log.append(f"Retrying (attempt {retry_count + 1})")
                    return Command(goto="persistent_failure_agent")
        
        # Mock human loop agent
        class HumanLoopAgent:
            def __init__(self):
                self.name = "human_loop_agent"
            
            async def execute_with_monitoring(self, state: VideoGenerationState) -> Command:
                escalation_log.append("Human intervention received")
                
                # Simulate human decision
                return Command(
                    goto="next_agent",
                    update={
                        "human_feedback": {
                            "decision": "skip_agent",
                            "reason": "Agent consistently failing, proceeding without it",
                            "timestamp": datetime.now().isoformat()
                        },
                        "pending_human_input": None,
                        "workflow_modified": True
                    }
                )
        
        # Test escalation workflow
        failure_agent = PersistentFailureAgent()
        error_handler = EscalatingErrorHandler()
        human_loop = HumanLoopAgent()
        
        current_state = initial_state.copy()
        
        # Execute failure cycle until escalation
        for attempt in range(4):  # Will escalate on 4th attempt
            # Agent fails
            result = await failure_agent.execute_with_monitoring(current_state)
            current_state.update(result.update)
            
            # Error handler processes
            error_result = await error_handler.execute_with_monitoring(current_state)
            current_state.update(error_result.update)
            
            # Check if escalated to human
            if error_result.goto == "human_loop_agent":
                # Human intervention
                human_result = await human_loop.execute_with_monitoring(current_state)
                current_state.update(human_result.update)
                break
        
        # Verify escalation workflow
        assert "Escalating to human" in escalation_log
        assert "Human intervention received" in escalation_log
        assert current_state["human_feedback"]["decision"] == "skip_agent"
        assert current_state["workflow_modified"] is True
        assert current_state["pending_human_input"] is None
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_error_context_preservation(self, mock_system_config, initial_state):
        """Test that error context is preserved across agent transitions."""
        error_contexts = []
        
        # Mock agent that captures error context
        class ContextCapturingAgent:
            def __init__(self, agent_name: str):
                self.name = agent_name
            
            async def execute_with_monitoring(self, state: VideoGenerationState) -> Command:
                # Capture current error context
                error_contexts.append({
                    "agent": self.name,
                    "error_count": state.get("error_count", 0),
                    "escalated_errors": state.get("escalated_errors", []),
                    "retry_count": state.get("retry_count", {}),
                    "execution_trace": state.get("execution_trace", [])
                })
                
                # Add error with rich context
                error_info = {
                    "agent": self.name,
                    "error": f"Error in {self.name}",
                    "timestamp": datetime.now().isoformat(),
                    "context": {
                        "topic": state.get("topic"),
                        "session_id": state.get("session_id"),
                        "current_scene": state.get("current_scene", 1),
                        "memory_usage": "150MB",
                        "cpu_usage": "45%"
                    },
                    "stack_trace": f"Traceback for {self.name} error",
                    "user_data": {
                        "user_preferences": {"quality": "high"},
                        "previous_errors": len(state.get("escalated_errors", []))
                    }
                }
                
                return Command(
                    goto="error_handler_agent",
                    update={
                        "error_count": state.get("error_count", 0) + 1,
                        "escalated_errors": state.get("escalated_errors", []) + [error_info],
                        "execution_trace": state.get("execution_trace", []) + [{
                            "agent": self.name,
                            "action": "error_occurred",
                            "timestamp": error_info["timestamp"]
                        }]
                    }
                )
        
        # Create sequence of agents
        agents = [
            ContextCapturingAgent("agent_1"),
            ContextCapturingAgent("agent_2"),
            ContextCapturingAgent("agent_3")
        ]
        
        current_state = initial_state.copy()
        
        # Execute agents to build error context
        for agent in agents:
            result = await agent.execute_with_monitoring(current_state)
            current_state.update(result.update)
        
        # Verify error context preservation
        assert len(error_contexts) == 3
        
        # Check context accumulation
        for i, context in enumerate(error_contexts):
            assert context["error_count"] == i  # Error count increases
            assert len(context["escalated_errors"]) == i  # Errors accumulate
            
            # Verify rich error context is preserved
            if i > 0:  # After first error
                previous_errors = context["escalated_errors"]
                for error in previous_errors:
                    assert "context" in error
                    assert "stack_trace" in error
                    assert "user_data" in error
                    assert error["context"]["topic"] == "Error Propagation Test"
                    assert error["context"]["session_id"] == "error_test_session"
        
        # Verify final state has complete error history
        final_errors = current_state["escalated_errors"]
        assert len(final_errors) == 3
        
        # Check that each error maintains its context
        for i, error in enumerate(final_errors):
            assert error["agent"] == f"agent_{i+1}"
            assert error["context"]["topic"] == "Error Propagation Test"
            assert error["user_data"]["previous_errors"] == i


if __name__ == "__main__":
    pytest.main([__file__])
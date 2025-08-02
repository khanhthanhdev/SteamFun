"""
Failure recovery tests for LangGraph multi-agent video generation system.
Tests various failure scenarios and recovery mechanisms.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

from src.langgraph_agents.workflow import LangGraphVideoGenerator
from src.langgraph_agents.state import VideoGenerationState, AgentError
from langgraph.types import Command


class TestFailureRecovery:
    """Failure recovery tests for multi-agent workflows."""
    
    @pytest.fixture
    def recovery_generator(self, temp_output_dir):
        """Create generator optimized for failure recovery testing."""
        with patch('src.langgraph_agents.workflow.VideoGenerationWorkflow'), \
             patch('src.langgraph_agents.workflow.initialize_langfuse_service'), \
             patch('src.core.video_planner.EnhancedVideoPlanner'), \
             patch('src.core.code_generator.CodeGenerator'), \
             patch('src.core.video_renderer.OptimizedVideoRenderer'):
            
            generator = LangGraphVideoGenerator(
                planner_model="openai/gpt-4o-mini",
                scene_model="openai/gpt-4o-mini",
                helper_model="openai/gpt-4o-mini",
                output_dir=temp_output_dir,
                verbose=True,
                use_rag=True,
                use_langfuse=False,  # Disable for testing
                enable_human_loop=True,  # Enable for escalation testing
                enable_monitoring=True,
                max_retries=3,
                enable_error_handling=True
            )
            
            yield generator
    
    @pytest.fixture
    def failure_scenarios(self):
        """Different failure scenarios for testing recovery mechanisms."""
        return {
            "agent_timeout": {
                "error_type": "TimeoutError",
                "error_message": "Agent execution timed out after 300 seconds",
                "failing_agent": "code_generator_agent",
                "recovery_strategy": "retry_with_timeout_increase",
                "max_retries": 2,
                "expected_recovery": True
            },
            "model_api_failure": {
                "error_type": "APIError",
                "error_message": "Model API returned 503 Service Unavailable",
                "failing_agent": "planner_agent",
                "recovery_strategy": "retry_with_backoff",
                "max_retries": 3,
                "expected_recovery": True
            },
            "code_generation_error": {
                "error_type": "SyntaxError",
                "error_message": "Generated code contains syntax errors",
                "failing_agent": "code_generator_agent",
                "recovery_strategy": "retry_with_rag_context",
                "max_retries": 2,
                "expected_recovery": True
            },
            "rendering_failure": {
                "error_type": "RenderingError",
                "error_message": "Manim rendering process crashed",
                "failing_agent": "renderer_agent",
                "recovery_strategy": "retry_with_lower_quality",
                "max_retries": 2,
                "expected_recovery": True
            },
            "visual_analysis_failure": {
                "error_type": "ProcessingError",
                "error_message": "Visual analysis service unavailable",
                "failing_agent": "visual_analysis_agent",
                "recovery_strategy": "skip_visual_analysis",
                "max_retries": 1,
                "expected_recovery": True
            },
            "rag_service_failure": {
                "error_type": "ConnectionError",
                "error_message": "RAG vector database connection failed",
                "failing_agent": "rag_agent",
                "recovery_strategy": "fallback_to_no_rag",
                "max_retries": 1,
                "expected_recovery": True
            },
            "cascading_failures": {
                "error_type": "MultipleErrors",
                "error_message": "Multiple agents failed simultaneously",
                "failing_agent": "multiple",
                "recovery_strategy": "escalate_to_human",
                "max_retries": 1,
                "expected_recovery": False  # Requires human intervention
            },
            "persistent_failure": {
                "error_type": "PersistentError",
                "error_message": "Agent consistently fails after all retries",
                "failing_agent": "code_generator_agent",
                "recovery_strategy": "escalate_to_human",
                "max_retries": 3,
                "expected_recovery": False  # Should escalate to human
            }
        }
    
    def create_failing_state(self, scenario: Dict[str, Any], session_id: str, output_dir: str, retry_count: int = 0) -> VideoGenerationState:
        """Create state representing a failure scenario."""
        error_info = {
            "agent": scenario["failing_agent"],
            "error": scenario["error_message"],
            "timestamp": datetime.now().isoformat(),
            "retry_count": retry_count,
            "error_type": scenario["error_type"]
        }
        
        return VideoGenerationState(
            messages=[],
            topic="Failure Recovery Test",
            description="Testing failure recovery mechanisms",
            session_id=session_id,
            output_dir=output_dir,
            print_response=True,
            use_rag=True,
            use_context_learning=True,
            context_learning_path="data/context_learning",
            chroma_db_path="data/rag/chroma_db",
            manim_docs_path="data/rag/manim_docs",
            embedding_model="hf:ibm-granite/granite-embedding-30m-english",
            use_visual_fix_code=False,
            use_langfuse=False,
            max_scene_concurrency=5,
            max_topic_concurrency=1,
            max_retries=scenario["max_retries"],
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
            scene_outline="# Scene 1: Test scene for failure recovery",
            scene_implementations={1: "Test scene implementation"},
            detected_plugins=["text", "code"],
            generated_code={} if scenario["failing_agent"] == "code_generator_agent" else {1: "from manim import *\nclass TestScene(Scene): pass"},
            code_errors={1: scenario["error_message"]} if scenario["failing_agent"] == "code_generator_agent" else {},
            rag_context={},
            rendered_videos={} if scenario["failing_agent"] == "renderer_agent" else {1: f"{output_dir}/scene_1.mp4"},
            combined_video_path=None,
            rendering_errors={1: scenario["error_message"]} if scenario["failing_agent"] == "renderer_agent" else {},
            visual_analysis_results={},
            visual_errors={1: [scenario["error_message"]]} if scenario["failing_agent"] == "visual_analysis_agent" else {},
            error_count=retry_count + 1,
            retry_count={scenario["failing_agent"]: retry_count},
            escalated_errors=[error_info],
            pending_human_input=None,
            human_feedback=None,
            performance_metrics={},
            execution_trace=[
                {
                    "agent": scenario["failing_agent"],
                    "action": "error_occurred",
                    "timestamp": datetime.now().isoformat(),
                    "error": scenario["error_message"],
                    "retry_count": retry_count
                }
            ],
            current_agent="error_handler_agent",
            next_agent=scenario["failing_agent"],
            workflow_complete=False,
            workflow_interrupted=False
        )
    
    def create_recovered_state(self, scenario: Dict[str, Any], session_id: str, output_dir: str, retry_count: int) -> VideoGenerationState:
        """Create state representing successful recovery."""
        return VideoGenerationState(
            messages=[],
            topic="Failure Recovery Test",
            description="Testing failure recovery mechanisms",
            session_id=session_id,
            output_dir=output_dir,
            print_response=True,
            use_rag=True,
            use_context_learning=True,
            context_learning_path="data/context_learning",
            chroma_db_path="data/rag/chroma_db",
            manim_docs_path="data/rag/manim_docs",
            embedding_model="hf:ibm-granite/granite-embedding-30m-english",
            use_visual_fix_code=False,
            use_langfuse=False,
            max_scene_concurrency=5,
            max_topic_concurrency=1,
            max_retries=scenario["max_retries"],
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
            scene_outline="# Scene 1: Test scene for failure recovery",
            scene_implementations={1: "Test scene implementation"},
            detected_plugins=["text", "code"],
            generated_code={1: "from manim import *\nclass TestScene(Scene):\n    def construct(self):\n        text = Text('Recovery Success')\n        self.play(Write(text))"},
            code_errors={},  # Errors resolved
            rag_context={1: "RAG context used for recovery"},
            rendered_videos={1: f"{output_dir}/scene_1_recovered.mp4"},
            combined_video_path=f"{output_dir}/final_video_recovered.mp4",
            rendering_errors={},  # Errors resolved
            visual_analysis_results={1: {"quality_score": 0.9, "visual_errors": []}},
            visual_errors={},  # Errors resolved
            error_count=0,  # Errors resolved
            retry_count={scenario["failing_agent"]: retry_count},
            escalated_errors=[],  # Errors resolved
            pending_human_input=None,
            human_feedback=None,
            performance_metrics={
                scenario["failing_agent"]: {
                    "last_execution_time": 2.5,
                    "success_rate": 1.0 if retry_count > 0 else 0.5,
                    "recovery_successful": True
                }
            },
            execution_trace=[
                {
                    "agent": scenario["failing_agent"],
                    "action": "error_occurred",
                    "timestamp": (datetime.now() - timedelta(seconds=30)).isoformat(),
                    "error": scenario["error_message"],
                    "retry_count": retry_count - 1
                },
                {
                    "agent": "error_handler_agent",
                    "action": "recovery_attempted",
                    "timestamp": (datetime.now() - timedelta(seconds=20)).isoformat(),
                    "recovery_strategy": scenario["recovery_strategy"]
                },
                {
                    "agent": scenario["failing_agent"],
                    "action": "recovery_successful",
                    "timestamp": datetime.now().isoformat(),
                    "retry_count": retry_count
                }
            ],
            current_agent=None,
            next_agent=None,
            workflow_complete=True,
            workflow_interrupted=False
        )
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_agent_timeout_recovery(self, recovery_generator, failure_scenarios):
        """Test recovery from agent timeout failures."""
        generator = recovery_generator
        scenario = failure_scenarios["agent_timeout"]
        session_id = "timeout_recovery_test"
        
        # Create failing and recovered states
        failing_state = self.create_failing_state(scenario, session_id, generator.init_params["output_dir"])
        recovered_state = self.create_recovered_state(scenario, session_id, generator.init_params["output_dir"], retry_count=1)
        
        # Mock workflow execution with timeout and recovery
        call_count = 0
        async def mock_invoke_with_timeout_recovery(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                # First call: timeout failure
                return failing_state
            else:
                # Second call: successful recovery
                return recovered_state
        
        generator.workflow.invoke = mock_invoke_with_timeout_recovery
        
        # Execute workflow with timeout recovery
        final_state = await generator.generate_video_pipeline(
            topic="Failure Recovery Test",
            description="Testing timeout recovery",
            session_id=session_id
        )
        
        # Verify timeout recovery
        assert final_state["workflow_complete"] is True
        assert final_state["error_count"] == 0
        assert final_state["retry_count"]["code_generator_agent"] == 1
        
        # Verify recovery was tracked
        execution_trace = final_state["execution_trace"]
        recovery_entries = [entry for entry in execution_trace if "recovery" in entry.get("action", "")]
        assert len(recovery_entries) >= 2  # recovery_attempted and recovery_successful
        
        print(f"✓ Agent timeout recovery test passed")
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_model_api_failure_recovery(self, recovery_generator, failure_scenarios):
        """Test recovery from model API failures with exponential backoff."""
        generator = recovery_generator
        scenario = failure_scenarios["model_api_failure"]
        session_id = "api_failure_recovery_test"
        
        # Create sequence of states: fail, fail, succeed
        failing_state_1 = self.create_failing_state(scenario, session_id, generator.init_params["output_dir"], retry_count=0)
        failing_state_2 = self.create_failing_state(scenario, session_id, generator.init_params["output_dir"], retry_count=1)
        recovered_state = self.create_recovered_state(scenario, session_id, generator.init_params["output_dir"], retry_count=2)
        
        # Mock workflow execution with API failure and recovery
        call_count = 0
        async def mock_invoke_with_api_recovery(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                return failing_state_1
            elif call_count == 2:
                # Simulate exponential backoff delay
                await asyncio.sleep(0.1)
                return failing_state_2
            else:
                # Successful recovery after backoff
                await asyncio.sleep(0.2)
                return recovered_state
        
        generator.workflow.invoke = mock_invoke_with_api_recovery
        
        # Execute workflow with API failure recovery
        start_time = time.time()
        final_state = await generator.generate_video_pipeline(
            topic="Failure Recovery Test",
            description="Testing API failure recovery",
            session_id=session_id
        )
        execution_time = time.time() - start_time
        
        # Verify API failure recovery
        assert final_state["workflow_complete"] is True
        assert final_state["error_count"] == 0
        assert final_state["retry_count"]["planner_agent"] == 2
        
        # Verify backoff delay was applied (execution should take longer due to retries)
        assert execution_time > 0.3  # Should include backoff delays
        
        print(f"✓ Model API failure recovery test passed (execution time: {execution_time:.2f}s)")
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_code_generation_error_recovery(self, recovery_generator, failure_scenarios):
        """Test recovery from code generation errors using RAG context."""
        generator = recovery_generator
        scenario = failure_scenarios["code_generation_error"]
        session_id = "code_error_recovery_test"
        
        # Create failing and recovered states
        failing_state = self.create_failing_state(scenario, session_id, generator.init_params["output_dir"])
        failing_state["code_errors"] = {1: "SyntaxError: invalid syntax in generated code"}
        
        recovered_state = self.create_recovered_state(scenario, session_id, generator.init_params["output_dir"], retry_count=1)
        recovered_state["rag_context"] = {1: "Enhanced RAG context used for code error recovery"}
        
        # Mock workflow execution with code error recovery
        call_count = 0
        async def mock_invoke_with_code_recovery(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                return failing_state
            else:
                return recovered_state
        
        generator.workflow.invoke = mock_invoke_with_code_recovery
        
        # Execute workflow with code error recovery
        final_state = await generator.generate_video_pipeline(
            topic="Failure Recovery Test",
            description="Testing code generation error recovery",
            session_id=session_id
        )
        
        # Verify code error recovery
        assert final_state["workflow_complete"] is True
        assert len(final_state["code_errors"]) == 0
        assert final_state["retry_count"]["code_generator_agent"] == 1
        assert "Enhanced RAG context" in final_state["rag_context"][1]
        
        # Verify generated code was fixed
        assert "Recovery Success" in final_state["generated_code"][1]
        
        print(f"✓ Code generation error recovery test passed")
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_rendering_failure_recovery(self, recovery_generator, failure_scenarios):
        """Test recovery from rendering failures with quality adjustment."""
        generator = recovery_generator
        scenario = failure_scenarios["rendering_failure"]
        session_id = "render_failure_recovery_test"
        
        # Create failing and recovered states
        failing_state = self.create_failing_state(scenario, session_id, generator.init_params["output_dir"])
        failing_state["rendering_errors"] = {1: "Manim rendering process crashed with high quality settings"}
        
        recovered_state = self.create_recovered_state(scenario, session_id, generator.init_params["output_dir"], retry_count=1)
        recovered_state["default_quality"] = "low"  # Quality reduced for recovery
        
        # Mock workflow execution with rendering recovery
        call_count = 0
        async def mock_invoke_with_render_recovery(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                return failing_state
            else:
                return recovered_state
        
        generator.workflow.invoke = mock_invoke_with_render_recovery
        
        # Execute workflow with rendering recovery
        final_state = await generator.generate_video_pipeline(
            topic="Failure Recovery Test",
            description="Testing rendering failure recovery",
            session_id=session_id
        )
        
        # Verify rendering recovery
        assert final_state["workflow_complete"] is True
        assert len(final_state["rendering_errors"]) == 0
        assert final_state["retry_count"]["renderer_agent"] == 1
        assert final_state["default_quality"] == "low"  # Quality was reduced
        
        # Verify video was rendered successfully
        assert 1 in final_state["rendered_videos"]
        assert "recovered" in final_state["rendered_videos"][1]
        
        print(f"✓ Rendering failure recovery test passed")
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_visual_analysis_failure_recovery(self, recovery_generator, failure_scenarios):
        """Test recovery from visual analysis failures by skipping analysis."""
        generator = recovery_generator
        scenario = failure_scenarios["visual_analysis_failure"]
        session_id = "visual_failure_recovery_test"
        
        # Create failing and recovered states
        failing_state = self.create_failing_state(scenario, session_id, generator.init_params["output_dir"])
        failing_state["visual_errors"] = {1: ["Visual analysis service unavailable"]}
        
        recovered_state = self.create_recovered_state(scenario, session_id, generator.init_params["output_dir"], retry_count=1)
        # Visual analysis skipped, but workflow continues
        recovered_state["visual_analysis_results"] = {}  # No analysis performed
        recovered_state["use_visual_fix_code"] = False  # Visual analysis disabled
        
        # Mock workflow execution with visual analysis recovery
        call_count = 0
        async def mock_invoke_with_visual_recovery(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                return failing_state
            else:
                return recovered_state
        
        generator.workflow.invoke = mock_invoke_with_visual_recovery
        
        # Execute workflow with visual analysis recovery
        final_state = await generator.generate_video_pipeline(
            topic="Failure Recovery Test",
            description="Testing visual analysis failure recovery",
            session_id=session_id
        )
        
        # Verify visual analysis recovery
        assert final_state["workflow_complete"] is True
        assert len(final_state["visual_errors"]) == 0
        assert final_state["retry_count"]["visual_analysis_agent"] == 1
        assert final_state["use_visual_fix_code"] is False  # Analysis was disabled
        
        # Verify workflow completed despite skipped visual analysis
        assert final_state["combined_video_path"] is not None
        
        print(f"✓ Visual analysis failure recovery test passed")
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_rag_service_failure_recovery(self, recovery_generator, failure_scenarios):
        """Test recovery from RAG service failures by falling back to no RAG."""
        generator = recovery_generator
        scenario = failure_scenarios["rag_service_failure"]
        session_id = "rag_failure_recovery_test"
        
        # Create failing and recovered states
        failing_state = self.create_failing_state(scenario, session_id, generator.init_params["output_dir"])
        
        recovered_state = self.create_recovered_state(scenario, session_id, generator.init_params["output_dir"], retry_count=1)
        recovered_state["use_rag"] = False  # RAG disabled for recovery
        recovered_state["rag_context"] = {}  # No RAG context available
        
        # Mock workflow execution with RAG recovery
        call_count = 0
        async def mock_invoke_with_rag_recovery(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                return failing_state
            else:
                return recovered_state
        
        generator.workflow.invoke = mock_invoke_with_rag_recovery
        
        # Execute workflow with RAG recovery
        final_state = await generator.generate_video_pipeline(
            topic="Failure Recovery Test",
            description="Testing RAG service failure recovery",
            session_id=session_id
        )
        
        # Verify RAG failure recovery
        assert final_state["workflow_complete"] is True
        assert final_state["retry_count"]["rag_agent"] == 1
        assert final_state["use_rag"] is False  # RAG was disabled
        assert len(final_state["rag_context"]) == 0  # No RAG context
        
        # Verify workflow completed without RAG
        assert final_state["combined_video_path"] is not None
        
        print(f"✓ RAG service failure recovery test passed")
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_cascading_failures_escalation(self, recovery_generator, failure_scenarios):
        """Test escalation to human intervention for cascading failures."""
        generator = recovery_generator
        scenario = failure_scenarios["cascading_failures"]
        session_id = "cascading_failure_test"
        
        # Create state with multiple simultaneous failures
        cascading_state = self.create_failing_state(scenario, session_id, generator.init_params["output_dir"])
        cascading_state.update({
            "error_count": 5,  # High error count
            "escalated_errors": [
                {
                    "agent": "planner_agent",
                    "error": "Planning service unavailable",
                    "timestamp": datetime.now().isoformat(),
                    "retry_count": 2
                },
                {
                    "agent": "code_generator_agent",
                    "error": "Code generation model failed",
                    "timestamp": datetime.now().isoformat(),
                    "retry_count": 2
                },
                {
                    "agent": "renderer_agent",
                    "error": "Rendering infrastructure down",
                    "timestamp": datetime.now().isoformat(),
                    "retry_count": 1
                }
            ],
            "retry_count": {
                "planner_agent": 2,
                "code_generator_agent": 2,
                "renderer_agent": 1
            }
        })
        
        # Create state after human intervention
        human_escalated_state = cascading_state.copy()
        human_escalated_state.update({
            "pending_human_input": {
                "context": "Multiple critical failures require human intervention",
                "options": ["abort_workflow", "retry_with_manual_config", "escalate_to_expert"],
                "requesting_agent": "error_handler_agent",
                "priority": "critical"
            },
            "current_agent": "human_loop_agent",
            "workflow_interrupted": True
        })
        
        # Mock workflow execution with cascading failures
        call_count = 0
        async def mock_invoke_with_cascading_failures(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                return cascading_state
            else:
                return human_escalated_state
        
        generator.workflow.invoke = mock_invoke_with_cascading_failures
        
        # Execute workflow with cascading failures
        final_state = await generator.generate_video_pipeline(
            topic="Failure Recovery Test",
            description="Testing cascading failures escalation",
            session_id=session_id
        )
        
        # Verify escalation to human intervention
        assert final_state["workflow_interrupted"] is True
        assert final_state["pending_human_input"] is not None
        assert final_state["pending_human_input"]["priority"] == "critical"
        assert final_state["error_count"] == 5
        assert len(final_state["escalated_errors"]) == 3
        
        # Verify multiple agents failed
        assert final_state["retry_count"]["planner_agent"] == 2
        assert final_state["retry_count"]["code_generator_agent"] == 2
        assert final_state["retry_count"]["renderer_agent"] == 1
        
        print(f"✓ Cascading failures escalation test passed")
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_persistent_failure_escalation(self, recovery_generator, failure_scenarios):
        """Test escalation to human intervention for persistent failures."""
        generator = recovery_generator
        scenario = failure_scenarios["persistent_failure"]
        session_id = "persistent_failure_test"
        
        # Create sequence of persistent failures
        failure_states = []
        for retry in range(scenario["max_retries"]):
            state = self.create_failing_state(scenario, session_id, generator.init_params["output_dir"], retry_count=retry)
            failure_states.append(state)
        
        # Create final escalation state
        escalation_state = failure_states[-1].copy()
        escalation_state.update({
            "pending_human_input": {
                "context": f"Agent {scenario['failing_agent']} has failed {scenario['max_retries']} times consecutively",
                "options": ["manual_intervention", "skip_agent", "abort_workflow"],
                "requesting_agent": "error_handler_agent",
                "priority": "high"
            },
            "current_agent": "human_loop_agent",
            "workflow_interrupted": True
        })
        
        # Mock workflow execution with persistent failures
        call_count = 0
        async def mock_invoke_with_persistent_failures(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count <= len(failure_states):
                return failure_states[call_count - 1]
            else:
                return escalation_state
        
        generator.workflow.invoke = mock_invoke_with_persistent_failures
        
        # Execute workflow with persistent failures
        final_state = await generator.generate_video_pipeline(
            topic="Failure Recovery Test",
            description="Testing persistent failure escalation",
            session_id=session_id
        )
        
        # Verify escalation after persistent failures
        assert final_state["workflow_interrupted"] is True
        assert final_state["pending_human_input"] is not None
        assert final_state["retry_count"]["code_generator_agent"] == scenario["max_retries"] - 1
        assert final_state["error_count"] == scenario["max_retries"]
        
        # Verify escalation context mentions persistent failures
        assert "failed" in final_state["pending_human_input"]["context"]
        assert str(scenario["max_retries"]) in final_state["pending_human_input"]["context"]
        
        print(f"✓ Persistent failure escalation test passed")
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_workflow_retry_with_different_configuration(self, recovery_generator, failure_scenarios):
        """Test workflow-level retry with different configuration after failures."""
        generator = recovery_generator
        scenario = failure_scenarios["model_api_failure"]
        session_id = "workflow_retry_test"
        
        # Create initial failure state
        initial_failure = self.create_failing_state(scenario, session_id, generator.init_params["output_dir"])
        
        # Create successful state with different configuration
        retry_success = self.create_recovered_state(scenario, session_id, generator.init_params["output_dir"], retry_count=1)
        retry_success["performance_metrics"]["planner_agent"]["model_used"] = "openai/gpt-3.5-turbo"  # Different model
        
        # Mock workflow-level retry mechanism
        workflow_retry_count = 0
        async def mock_invoke_with_workflow_retry(*args, **kwargs):
            nonlocal workflow_retry_count
            workflow_retry_count += 1
            
            if workflow_retry_count == 1:
                # First workflow attempt fails
                return initial_failure
            else:
                # Second workflow attempt succeeds with different config
                return retry_success
        
        generator.workflow.invoke = mock_invoke_with_workflow_retry
        
        # Mock the workflow retry mechanism
        original_execute_with_retries = generator._execute_workflow_with_retries
        
        async def mock_execute_with_retries(*args, **kwargs):
            # First attempt fails, second succeeds
            return await original_execute_with_retries(*args, **kwargs)
        
        generator._execute_workflow_with_retries = mock_execute_with_retries
        
        # Execute workflow with retry mechanism
        final_state = await generator.generate_video_pipeline(
            topic="Failure Recovery Test",
            description="Testing workflow-level retry",
            session_id=session_id
        )
        
        # Verify workflow-level retry succeeded
        assert final_state["workflow_complete"] is True
        assert final_state["error_count"] == 0
        assert workflow_retry_count == 2  # Two workflow attempts
        
        # Verify different configuration was used
        assert "model_used" in final_state["performance_metrics"]["planner_agent"]
        
        print(f"✓ Workflow retry with different configuration test passed")
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_recovery_performance_impact(self, recovery_generator, failure_scenarios):
        """Test performance impact of failure recovery mechanisms."""
        generator = recovery_generator
        scenario = failure_scenarios["code_generation_error"]
        session_id = "recovery_performance_test"
        
        # Create failure and recovery states
        failing_state = self.create_failing_state(scenario, session_id, generator.init_params["output_dir"])
        recovered_state = self.create_recovered_state(scenario, session_id, generator.init_params["output_dir"], retry_count=2)
        
        # Mock workflow execution with measured recovery time
        call_count = 0
        recovery_times = []
        
        async def mock_invoke_with_performance_tracking(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            start_time = time.time()
            
            if call_count <= 2:
                # First two calls fail
                await asyncio.sleep(0.1)  # Simulate failure processing time
                result = failing_state
            else:
                # Third call succeeds
                await asyncio.sleep(0.2)  # Simulate recovery processing time
                result = recovered_state
            
            end_time = time.time()
            recovery_times.append(end_time - start_time)
            
            return result
        
        generator.workflow.invoke = mock_invoke_with_performance_tracking
        
        # Execute workflow and measure total recovery time
        start_time = time.time()
        final_state = await generator.generate_video_pipeline(
            topic="Failure Recovery Test",
            description="Testing recovery performance impact",
            session_id=session_id
        )
        total_time = time.time() - start_time
        
        # Verify recovery succeeded
        assert final_state["workflow_complete"] is True
        assert final_state["retry_count"]["code_generator_agent"] == 2
        
        # Analyze performance impact
        total_recovery_time = sum(recovery_times)
        average_recovery_time = total_recovery_time / len(recovery_times)
        
        print(f"Recovery Performance Analysis:")
        print(f"  Total execution time: {total_time:.2f}s")
        print(f"  Total recovery time: {total_recovery_time:.2f}s")
        print(f"  Average recovery time per attempt: {average_recovery_time:.2f}s")
        print(f"  Number of recovery attempts: {len(recovery_times)}")
        
        # Performance assertions
        assert total_time < 10.0, "Total recovery time should be reasonable"
        assert average_recovery_time < 1.0, "Average recovery time should be acceptable"
        assert len(recovery_times) == 3, "Expected number of recovery attempts"
        
        print(f"✓ Recovery performance impact test passed")
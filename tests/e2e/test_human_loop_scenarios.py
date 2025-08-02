"""
Human-in-the-loop scenario tests for LangGraph multi-agent video generation system.
Tests various human intervention scenarios and approval workflows.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

from src.langgraph_agents.workflow import LangGraphVideoGenerator
from src.langgraph_agents.state import VideoGenerationState
from langgraph.types import Command


class TestHumanLoopScenarios:
    """Human-in-the-loop scenario tests for multi-agent workflows."""
    
    @pytest.fixture
    def human_loop_generator(self, temp_output_dir):
        """Create generator with human-in-the-loop enabled."""
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
                enable_human_loop=True,  # Enable human loop
                enable_monitoring=True,
                max_retries=3
            )
            
            yield generator
    
    @pytest.fixture
    def human_intervention_scenarios(self):
        """Different scenarios requiring human intervention."""
        return {
            "planning_approval": {
                "trigger_agent": "planner_agent",
                "context": "Scene outline requires human review before proceeding",
                "options": ["approve", "modify", "reject"],
                "expected_decision": "approve",
                "priority": "medium"
            },
            "code_quality_review": {
                "trigger_agent": "code_generator_agent",
                "context": "Generated code has potential issues that need human review",
                "options": ["approve", "fix_and_retry", "regenerate", "skip_scene"],
                "expected_decision": "fix_and_retry",
                "priority": "high"
            },
            "visual_error_escalation": {
                "trigger_agent": "visual_analysis_agent",
                "context": "Visual analysis detected errors that require human judgment",
                "options": ["accept_as_is", "retry_rendering", "modify_code", "escalate_to_expert"],
                "expected_decision": "retry_rendering",
                "priority": "high"
            },
            "error_recovery_decision": {
                "trigger_agent": "error_handler_agent",
                "context": "Multiple errors occurred, human decision needed for recovery strategy",
                "options": ["retry_with_different_model", "skip_problematic_scenes", "abort_workflow", "manual_intervention"],
                "expected_decision": "retry_with_different_model",
                "priority": "critical"
            },
            "final_output_approval": {
                "trigger_agent": "renderer_agent",
                "context": "Final video output ready for human approval before completion",
                "options": ["approve_and_complete", "request_modifications", "regenerate_scenes"],
                "expected_decision": "approve_and_complete",
                "priority": "medium"
            }
        }
    
    def create_mock_human_intervention_state(self, scenario: Dict[str, Any], session_id: str, output_dir: str) -> VideoGenerationState:
        """Create mock state that triggers human intervention."""
        return VideoGenerationState(
            messages=[],
            topic="Human Loop Test Topic",
            description="Testing human intervention scenarios",
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
            scene_outline="# Scene 1: Test scene requiring human approval",
            scene_implementations={1: "Test scene implementation"},
            detected_plugins=["text", "code"],
            generated_code={1: "from manim import *\nclass TestScene(Scene): pass"},
            code_errors={},
            rag_context={1: "Test RAG context"},
            rendered_videos={},
            combined_video_path=None,
            rendering_errors={},
            visual_analysis_results={},
            visual_errors={},
            error_count=0,
            retry_count={},
            escalated_errors=[],
            pending_human_input={
                "context": scenario["context"],
                "options": scenario["options"],
                "requesting_agent": scenario["trigger_agent"],
                "priority": scenario["priority"],
                "timestamp": datetime.now().isoformat(),
                "timeout_seconds": 300
            },
            human_feedback=None,
            performance_metrics={},
            execution_trace=[
                {
                    "agent": scenario["trigger_agent"],
                    "action": "request_human_input",
                    "timestamp": datetime.now().isoformat(),
                    "context": scenario["context"]
                }
            ],
            current_agent="human_loop_agent",
            next_agent=scenario["trigger_agent"],
            workflow_complete=False,
            workflow_interrupted=True  # Workflow paused for human input
        )
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_planning_approval_scenario(self, human_loop_generator, human_intervention_scenarios):
        """Test human approval scenario during planning phase."""
        generator = human_loop_generator
        scenario = human_intervention_scenarios["planning_approval"]
        session_id = "human_loop_planning_test"
        
        # Create initial state requiring human input
        initial_state = self.create_mock_human_intervention_state(
            scenario, session_id, generator.init_params["output_dir"]
        )
        
        # Create state after human approval
        approved_state = initial_state.copy()
        approved_state.update({
            "pending_human_input": None,
            "human_feedback": {
                "decision": scenario["expected_decision"],
                "comments": "Scene outline looks good, proceeding with code generation",
                "timestamp": datetime.now().isoformat(),
                "reviewer": "test_human_reviewer"
            },
            "workflow_interrupted": False,
            "current_agent": "code_generator_agent",
            "next_agent": "code_generator_agent"
        })
        
        # Mock workflow execution with human intervention
        call_count = 0
        async def mock_invoke_with_human_approval(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                # First call: return state requiring human input
                return initial_state
            else:
                # Second call: return state after human approval
                return approved_state
        
        generator.workflow.invoke = mock_invoke_with_human_approval
        
        # Mock human interface for providing approval
        mock_human_interface = Mock()
        mock_human_interface.request_human_decision = AsyncMock(return_value={
            "decision": scenario["expected_decision"],
            "comments": "Scene outline looks good, proceeding with code generation",
            "timestamp": datetime.now().isoformat(),
            "reviewer": "test_human_reviewer"
        })
        
        # Simulate workflow execution with human intervention
        with patch('src.langgraph_agents.interfaces.human_intervention_interface.HumanInterventionInterface', return_value=mock_human_interface):
            # Start workflow
            final_state = await generator.generate_video_pipeline(
                topic="Human Loop Test Topic",
                description="Testing human intervention scenarios",
                session_id=session_id
            )
        
        # Verify human intervention was processed
        assert final_state["human_feedback"] is not None
        assert final_state["human_feedback"]["decision"] == scenario["expected_decision"]
        assert final_state["pending_human_input"] is None
        assert final_state["workflow_interrupted"] is False
        
        # Verify workflow continued after approval
        execution_trace = final_state["execution_trace"]
        human_actions = [entry for entry in execution_trace if "human" in entry.get("action", "")]
        assert len(human_actions) >= 1
        
        print(f"✓ Planning approval scenario completed successfully")
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_code_quality_review_scenario(self, human_loop_generator, human_intervention_scenarios):
        """Test human review scenario for code quality issues."""
        generator = human_loop_generator
        scenario = human_intervention_scenarios["code_quality_review"]
        session_id = "human_loop_code_review_test"
        
        # Create state with code quality issues
        code_review_state = self.create_mock_human_intervention_state(
            scenario, session_id, generator.init_params["output_dir"]
        )
        
        # Add code quality issues to state
        code_review_state.update({
            "code_errors": {
                1: "Potential syntax error in generated Manim code"
            },
            "generated_code": {
                1: "from manim import *\nclass TestScene(Scene):\n    def construct(self):\n        # Potentially problematic code\n        undefined_variable.animate()"
            }
        })
        
        # Create state after human review and fix
        fixed_state = code_review_state.copy()
        fixed_state.update({
            "pending_human_input": None,
            "human_feedback": {
                "decision": scenario["expected_decision"],
                "comments": "Code has undefined variable issue, fixing and retrying",
                "timestamp": datetime.now().isoformat(),
                "reviewer": "code_reviewer",
                "suggested_fixes": ["Define undefined_variable before use", "Add proper error handling"]
            },
            "code_errors": {},  # Errors fixed
            "generated_code": {
                1: "from manim import *\nclass TestScene(Scene):\n    def construct(self):\n        # Fixed code\n        text = Text('Hello World')\n        self.play(Write(text))"
            },
            "workflow_interrupted": False,
            "retry_count": {"code_generator_agent": 1}
        })
        
        # Mock workflow execution with code review
        call_count = 0
        async def mock_invoke_with_code_review(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                return code_review_state
            else:
                return fixed_state
        
        generator.workflow.invoke = mock_invoke_with_code_review
        
        # Mock human interface for code review
        mock_human_interface = Mock()
        mock_human_interface.request_human_decision = AsyncMock(return_value={
            "decision": scenario["expected_decision"],
            "comments": "Code has undefined variable issue, fixing and retrying",
            "timestamp": datetime.now().isoformat(),
            "reviewer": "code_reviewer",
            "suggested_fixes": ["Define undefined_variable before use", "Add proper error handling"]
        })
        
        # Execute workflow with code review
        with patch('src.langgraph_agents.interfaces.human_intervention_interface.HumanInterventionInterface', return_value=mock_human_interface):
            final_state = await generator.generate_video_pipeline(
                topic="Human Loop Test Topic",
                description="Testing code quality review",
                session_id=session_id
            )
        
        # Verify code review was processed
        assert final_state["human_feedback"]["decision"] == scenario["expected_decision"]
        assert "suggested_fixes" in final_state["human_feedback"]
        assert len(final_state["code_errors"]) == 0  # Errors should be fixed
        assert final_state["retry_count"]["code_generator_agent"] == 1
        
        print(f"✓ Code quality review scenario completed successfully")
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_visual_error_escalation_scenario(self, human_loop_generator, human_intervention_scenarios):
        """Test human escalation scenario for visual analysis errors."""
        generator = human_loop_generator
        scenario = human_intervention_scenarios["visual_error_escalation"]
        session_id = "human_loop_visual_test"
        
        # Create state with visual errors
        visual_error_state = self.create_mock_human_intervention_state(
            scenario, session_id, generator.init_params["output_dir"]
        )
        
        # Add visual errors to state
        visual_error_state.update({
            "visual_errors": {
                1: ["Text is not visible", "Animation timing is off"]
            },
            "visual_analysis_results": {
                1: {
                    "quality_score": 0.3,  # Low quality score
                    "visual_errors": ["Text is not visible", "Animation timing is off"],
                    "analysis_confidence": 0.85
                }
            },
            "rendered_videos": {
                1: f"{generator.init_params['output_dir']}/scene_1_with_errors.mp4"
            }
        })
        
        # Create state after human decision to retry rendering
        retry_state = visual_error_state.copy()
        retry_state.update({
            "pending_human_input": None,
            "human_feedback": {
                "decision": scenario["expected_decision"],
                "comments": "Visual errors confirmed, retrying rendering with adjusted parameters",
                "timestamp": datetime.now().isoformat(),
                "reviewer": "visual_expert",
                "retry_parameters": {"quality": "high", "resolution": "1080p"}
            },
            "visual_errors": {},  # Errors resolved after retry
            "visual_analysis_results": {
                1: {
                    "quality_score": 0.9,  # Improved quality
                    "visual_errors": [],
                    "analysis_confidence": 0.95
                }
            },
            "rendered_videos": {
                1: f"{generator.init_params['output_dir']}/scene_1_fixed.mp4"
            },
            "workflow_interrupted": False,
            "retry_count": {"renderer_agent": 1}
        })
        
        # Mock workflow execution with visual error handling
        call_count = 0
        async def mock_invoke_with_visual_review(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                return visual_error_state
            else:
                return retry_state
        
        generator.workflow.invoke = mock_invoke_with_visual_review
        
        # Mock human interface for visual review
        mock_human_interface = Mock()
        mock_human_interface.request_human_decision = AsyncMock(return_value={
            "decision": scenario["expected_decision"],
            "comments": "Visual errors confirmed, retrying rendering with adjusted parameters",
            "timestamp": datetime.now().isoformat(),
            "reviewer": "visual_expert",
            "retry_parameters": {"quality": "high", "resolution": "1080p"}
        })
        
        # Execute workflow with visual error escalation
        with patch('src.langgraph_agents.interfaces.human_intervention_interface.HumanInterventionInterface', return_value=mock_human_interface):
            final_state = await generator.generate_video_pipeline(
                topic="Human Loop Test Topic",
                description="Testing visual error escalation",
                session_id=session_id
            )
        
        # Verify visual error escalation was handled
        assert final_state["human_feedback"]["decision"] == scenario["expected_decision"]
        assert "retry_parameters" in final_state["human_feedback"]
        assert len(final_state["visual_errors"]) == 0  # Errors should be resolved
        assert final_state["visual_analysis_results"][1]["quality_score"] > 0.8
        assert final_state["retry_count"]["renderer_agent"] == 1
        
        print(f"✓ Visual error escalation scenario completed successfully")
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_error_recovery_decision_scenario(self, human_loop_generator, human_intervention_scenarios):
        """Test human decision scenario for error recovery strategy."""
        generator = human_loop_generator
        scenario = human_intervention_scenarios["error_recovery_decision"]
        session_id = "human_loop_error_recovery_test"
        
        # Create state with multiple errors requiring human decision
        error_recovery_state = self.create_mock_human_intervention_state(
            scenario, session_id, generator.init_params["output_dir"]
        )
        
        # Add multiple errors to state
        error_recovery_state.update({
            "error_count": 3,
            "escalated_errors": [
                {
                    "agent": "code_generator_agent",
                    "error": "Model API rate limit exceeded",
                    "timestamp": datetime.now().isoformat(),
                    "retry_count": 2
                },
                {
                    "agent": "renderer_agent",
                    "error": "Rendering process crashed",
                    "timestamp": datetime.now().isoformat(),
                    "retry_count": 1
                },
                {
                    "agent": "visual_analysis_agent",
                    "error": "Visual analysis service unavailable",
                    "timestamp": datetime.now().isoformat(),
                    "retry_count": 1
                }
            ],
            "retry_count": {
                "code_generator_agent": 2,
                "renderer_agent": 1,
                "visual_analysis_agent": 1
            }
        })
        
        # Create state after human decision to retry with different model
        recovery_state = error_recovery_state.copy()
        recovery_state.update({
            "pending_human_input": None,
            "human_feedback": {
                "decision": scenario["expected_decision"],
                "comments": "Switching to backup model and retrying failed operations",
                "timestamp": datetime.now().isoformat(),
                "reviewer": "system_admin",
                "recovery_strategy": {
                    "new_model": "openai/gpt-3.5-turbo",
                    "retry_failed_agents": ["code_generator_agent", "renderer_agent"],
                    "skip_visual_analysis": True
                }
            },
            "error_count": 0,  # Errors resolved
            "escalated_errors": [],
            "workflow_interrupted": False,
            "current_agent": "code_generator_agent"  # Resume from code generation
        })
        
        # Mock workflow execution with error recovery
        call_count = 0
        async def mock_invoke_with_error_recovery(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                return error_recovery_state
            else:
                return recovery_state
        
        generator.workflow.invoke = mock_invoke_with_error_recovery
        
        # Mock human interface for error recovery decision
        mock_human_interface = Mock()
        mock_human_interface.request_human_decision = AsyncMock(return_value={
            "decision": scenario["expected_decision"],
            "comments": "Switching to backup model and retrying failed operations",
            "timestamp": datetime.now().isoformat(),
            "reviewer": "system_admin",
            "recovery_strategy": {
                "new_model": "openai/gpt-3.5-turbo",
                "retry_failed_agents": ["code_generator_agent", "renderer_agent"],
                "skip_visual_analysis": True
            }
        })
        
        # Execute workflow with error recovery
        with patch('src.langgraph_agents.interfaces.human_intervention_interface.HumanInterventionInterface', return_value=mock_human_interface):
            final_state = await generator.generate_video_pipeline(
                topic="Human Loop Test Topic",
                description="Testing error recovery decision",
                session_id=session_id
            )
        
        # Verify error recovery decision was processed
        assert final_state["human_feedback"]["decision"] == scenario["expected_decision"]
        assert "recovery_strategy" in final_state["human_feedback"]
        assert final_state["error_count"] == 0  # Errors should be resolved
        assert len(final_state["escalated_errors"]) == 0
        
        print(f"✓ Error recovery decision scenario completed successfully")
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_human_intervention_timeout_scenario(self, human_loop_generator, human_intervention_scenarios):
        """Test human intervention timeout handling."""
        generator = human_loop_generator
        scenario = human_intervention_scenarios["planning_approval"]
        session_id = "human_loop_timeout_test"
        
        # Create state with short timeout
        timeout_state = self.create_mock_human_intervention_state(
            scenario, session_id, generator.init_params["output_dir"]
        )
        timeout_state["pending_human_input"]["timeout_seconds"] = 1  # Very short timeout
        
        # Create state after timeout with default action
        timeout_resolved_state = timeout_state.copy()
        timeout_resolved_state.update({
            "pending_human_input": None,
            "human_feedback": {
                "decision": "approve",  # Default action
                "comments": "Auto-approved due to timeout",
                "timestamp": datetime.now().isoformat(),
                "reviewer": "system_timeout",
                "timeout_occurred": True
            },
            "workflow_interrupted": False
        })
        
        # Mock workflow execution with timeout
        call_count = 0
        async def mock_invoke_with_timeout(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                return timeout_state
            else:
                # Simulate timeout delay
                await asyncio.sleep(1.5)  # Longer than timeout
                return timeout_resolved_state
        
        generator.workflow.invoke = mock_invoke_with_timeout
        
        # Mock human interface that times out
        mock_human_interface = Mock()
        mock_human_interface.request_human_decision = AsyncMock(side_effect=asyncio.TimeoutError("Human input timeout"))
        
        # Execute workflow with timeout
        with patch('src.langgraph_agents.interfaces.human_intervention_interface.HumanInterventionInterface', return_value=mock_human_interface):
            final_state = await generator.generate_video_pipeline(
                topic="Human Loop Test Topic",
                description="Testing human intervention timeout",
                session_id=session_id
            )
        
        # Verify timeout was handled
        assert final_state["human_feedback"] is not None
        assert final_state["human_feedback"]["timeout_occurred"] is True
        assert final_state["human_feedback"]["decision"] == "approve"  # Default action
        assert final_state["pending_human_input"] is None
        
        print(f"✓ Human intervention timeout scenario handled successfully")
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_multiple_human_interventions_workflow(self, human_loop_generator, human_intervention_scenarios):
        """Test workflow with multiple human intervention points."""
        generator = human_loop_generator
        session_id = "human_loop_multiple_test"
        
        # Define sequence of human interventions
        intervention_sequence = [
            human_intervention_scenarios["planning_approval"],
            human_intervention_scenarios["code_quality_review"],
            human_intervention_scenarios["final_output_approval"]
        ]
        
        # Create states for each intervention
        intervention_states = []
        for i, scenario in enumerate(intervention_sequence):
            state = self.create_mock_human_intervention_state(
                scenario, session_id, generator.init_params["output_dir"]
            )
            state["execution_trace"].append({
                "agent": "workflow_orchestrator",
                "action": f"human_intervention_{i+1}",
                "timestamp": datetime.now().isoformat(),
                "intervention_type": scenario["trigger_agent"]
            })
            intervention_states.append(state)
        
        # Create final completed state
        final_state = intervention_states[-1].copy()
        final_state.update({
            "pending_human_input": None,
            "human_feedback": {
                "decision": "approve_and_complete",
                "comments": "All interventions completed, workflow approved",
                "timestamp": datetime.now().isoformat(),
                "reviewer": "final_reviewer"
            },
            "workflow_complete": True,
            "workflow_interrupted": False,
            "combined_video_path": f"{generator.init_params['output_dir']}/final_approved_video.mp4"
        })
        
        # Mock workflow execution with multiple interventions
        call_count = 0
        async def mock_invoke_with_multiple_interventions(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count <= len(intervention_states):
                return intervention_states[call_count - 1]
            else:
                return final_state
        
        generator.workflow.invoke = mock_invoke_with_multiple_interventions
        
        # Mock human interface for multiple interventions
        intervention_responses = [
            {"decision": "approve", "comments": "Planning approved"},
            {"decision": "fix_and_retry", "comments": "Code needs fixing"},
            {"decision": "approve_and_complete", "comments": "Final output approved"}
        ]
        
        response_count = 0
        async def mock_human_decision(*args, **kwargs):
            nonlocal response_count
            response = intervention_responses[response_count % len(intervention_responses)]
            response_count += 1
            return {
                **response,
                "timestamp": datetime.now().isoformat(),
                "reviewer": f"reviewer_{response_count}"
            }
        
        mock_human_interface = Mock()
        mock_human_interface.request_human_decision = mock_human_decision
        
        # Execute workflow with multiple interventions
        with patch('src.langgraph_agents.interfaces.human_intervention_interface.HumanInterventionInterface', return_value=mock_human_interface):
            completed_state = await generator.generate_video_pipeline(
                topic="Human Loop Test Topic",
                description="Testing multiple human interventions",
                session_id=session_id
            )
        
        # Verify multiple interventions were processed
        assert completed_state["workflow_complete"] is True
        assert completed_state["human_feedback"]["decision"] == "approve_and_complete"
        assert completed_state["combined_video_path"] is not None
        
        # Verify execution trace contains all interventions
        execution_trace = completed_state["execution_trace"]
        intervention_entries = [entry for entry in execution_trace if "human_intervention" in entry.get("action", "")]
        assert len(intervention_entries) >= len(intervention_sequence)
        
        print(f"✓ Multiple human interventions workflow completed successfully")
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_human_loop_streaming_updates(self, human_loop_generator, human_intervention_scenarios):
        """Test streaming updates during human-in-the-loop scenarios."""
        generator = human_loop_generator
        scenario = human_intervention_scenarios["planning_approval"]
        session_id = "human_loop_streaming_test"
        
        # Create streaming chunks with human intervention
        streaming_chunks = [
            {
                "planner_agent": {
                    "scene_outline": "# Scene 1: Test scene",
                    "current_agent": "planner_agent",
                    "next_agent": "human_loop_agent",
                    "pending_human_input": {
                        "context": scenario["context"],
                        "options": scenario["options"],
                        "requesting_agent": scenario["trigger_agent"]
                    }
                }
            },
            {
                "human_loop_agent": {
                    "human_feedback": {
                        "decision": "approve",
                        "comments": "Approved via streaming"
                    },
                    "current_agent": "human_loop_agent",
                    "next_agent": "code_generator_agent",
                    "pending_human_input": None
                }
            },
            {
                "code_generator_agent": {
                    "generated_code": {1: "from manim import *..."},
                    "current_agent": "code_generator_agent",
                    "next_agent": "END",
                    "workflow_complete": True
                }
            }
        ]
        
        # Mock streaming with human intervention
        async def mock_stream_with_human_loop(*args, **kwargs):
            for chunk in streaming_chunks:
                yield chunk
                await asyncio.sleep(0.1)  # Small delay between chunks
        
        generator.workflow.stream = mock_stream_with_human_loop
        generator.workflow.get_workflow_status = Mock(return_value={
            "status": "completed",
            "current_agent": None,
            "workflow_complete": True,
            "execution_trace": [
                {"agent": "planner_agent", "action": "request_human_input"},
                {"agent": "human_loop_agent", "action": "process_human_feedback"},
                {"agent": "code_generator_agent", "action": "complete_execution"}
            ]
        })
        
        # Execute streaming workflow with human intervention
        streaming_events = []
        async for event in generator.stream_video_generation(
            topic="Human Loop Test Topic",
            description="Testing streaming with human intervention",
            session_id=session_id
        ):
            streaming_events.append(event)
        
        # Verify streaming events include human intervention
        assert len(streaming_events) >= 4  # Start, progress chunks, completion
        
        # Find human intervention events
        human_events = []
        for event in streaming_events:
            if event.get("event_type") == "workflow_progress":
                for agent_state in event.values():
                    if isinstance(agent_state, dict) and "pending_human_input" in agent_state:
                        human_events.append(event)
                    elif isinstance(agent_state, dict) and "human_feedback" in agent_state:
                        human_events.append(event)
        
        assert len(human_events) >= 2  # At least request and response events
        
        print(f"✓ Human loop streaming updates test completed successfully")
"""
Integration tests for LangGraph service integrations.
Tests LangGraph workflow execution, state management, and agent coordination.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from app.core.agents.base_agent import BaseAgent
from app.services.agent_service import AgentService
from src.langgraph_agents.state import VideoGenerationState, SystemConfig


class TestLangGraphIntegration:
    """Integration test suite for LangGraph service integrations."""
    
    @pytest.fixture
    def mock_langgraph_workflow(self):
        """Create mock LangGraph workflow for integration tests."""
        workflow = Mock()
        workflow.compile = Mock()
        workflow.invoke = AsyncMock()
        workflow.stream = AsyncMock()
        return workflow
    
    @pytest.fixture
    def sample_video_state(self):
        """Create sample video generation state for integration tests."""
        return VideoGenerationState(
            messages=[],
            topic="LangGraph Integration Test",
            description="Testing LangGraph workflow integration",
            session_id="langgraph_test_session",
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
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_langgraph_workflow_execution_integration(self, mock_langgraph_workflow, sample_video_state):
        """Test LangGraph workflow execution integration."""
        # Mock workflow execution result
        mock_langgraph_workflow.invoke.return_value = {
            "scene_outline": "Generated scene outline",
            "scene_implementations": {1: "Scene 1 implementation"},
            "detected_plugins": ["text", "code"],
            "workflow_complete": True
        }
        
        with patch('app.services.agent_service.create_langgraph_workflow', return_value=mock_langgraph_workflow):
            agent_service = AgentService()
            
            # Execute workflow
            result = await agent_service._execute_langgraph_workflow(sample_video_state)
            
            # Verify workflow was invoked
            mock_langgraph_workflow.invoke.assert_called_once_with(sample_video_state)
            
            # Verify result structure
            assert result["scene_outline"] == "Generated scene outline"
            assert result["workflow_complete"] is True
            assert 1 in result["scene_implementations"]
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_langgraph_streaming_integration(self, mock_langgraph_workflow, sample_video_state):
        """Test LangGraph streaming workflow integration."""
        # Mock streaming workflow results
        stream_events = [
            {"agent": "planner_agent", "update": {"scene_outline": "Partial outline"}},
            {"agent": "code_generator_agent", "update": {"generated_code": {1: "partial code"}}},
            {"agent": "renderer_agent", "update": {"rendered_videos": {1: "video1.mp4"}}}
        ]
        
        async def mock_stream_generator():
            for event in stream_events:
                yield event
        
        mock_langgraph_workflow.stream.return_value = mock_stream_generator()
        
        with patch('app.services.agent_service.create_langgraph_workflow', return_value=mock_langgraph_workflow):
            agent_service = AgentService()
            
            # Execute streaming workflow
            stream_results = []
            async for event in agent_service._stream_langgraph_workflow(sample_video_state):
                stream_results.append(event)
            
            # Verify streaming events
            assert len(stream_results) == 3
            assert stream_results[0]["agent"] == "planner_agent"
            assert stream_results[1]["agent"] == "code_generator_agent"
            assert stream_results[2]["agent"] == "renderer_agent"
            
            # Verify workflow was streamed
            mock_langgraph_workflow.stream.assert_called_once_with(sample_video_state)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_langgraph_state_management_integration(self, mock_langgraph_workflow, sample_video_state):
        """Test LangGraph state management integration."""
        # Mock state updates during workflow execution
        state_updates = [
            {"current_agent": "planner_agent", "scene_outline": "Planning..."},
            {"current_agent": "code_generator_agent", "generated_code": {1: "Generating..."}},
            {"current_agent": "renderer_agent", "rendered_videos": {1: "Rendering..."}}
        ]
        
        mock_langgraph_workflow.invoke.side_effect = lambda state: {
            **state,
            **state_updates[len(state.get("execution_trace", []))]
        }
        
        with patch('app.services.agent_service.create_langgraph_workflow', return_value=mock_langgraph_workflow):
            agent_service = AgentService()
            
            # Execute workflow with state tracking
            current_state = sample_video_state.copy()
            
            for i in range(3):
                result = await agent_service._execute_langgraph_workflow(current_state)
                current_state.update(result)
                
                # Verify state progression
                assert current_state["current_agent"] in ["planner_agent", "code_generator_agent", "renderer_agent"]
            
            # Verify final state
            assert "scene_outline" in current_state
            assert "generated_code" in current_state
            assert "rendered_videos" in current_state
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_langgraph_error_handling_integration(self, mock_langgraph_workflow, sample_video_state):
        """Test LangGraph error handling integration."""
        # Mock workflow error
        mock_langgraph_workflow.invoke.side_effect = Exception("LangGraph execution failed")
        
        with patch('app.services.agent_service.create_langgraph_workflow', return_value=mock_langgraph_workflow):
            agent_service = AgentService()
            
            # Test error handling
            with pytest.raises(Exception) as exc_info:
                await agent_service._execute_langgraph_workflow(sample_video_state)
            
            assert "LangGraph execution failed" in str(exc_info.value)
            
            # Verify workflow was attempted
            mock_langgraph_workflow.invoke.assert_called_once()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_langgraph_agent_coordination_integration(self, mock_langgraph_workflow, sample_video_state):
        """Test LangGraph agent coordination integration."""
        # Mock coordinated agent execution
        agent_sequence = [
            {"agent": "planner_agent", "output": {"scene_outline": "Planned outline"}},
            {"agent": "code_generator_agent", "output": {"generated_code": {1: "Generated code"}}},
            {"agent": "renderer_agent", "output": {"rendered_videos": {1: "rendered.mp4"}}}
        ]
        
        execution_count = 0
        def mock_invoke(state):
            nonlocal execution_count
            current_agent = agent_sequence[execution_count]
            execution_count += 1
            
            return {
                **state,
                "current_agent": current_agent["agent"],
                **current_agent["output"],
                "execution_trace": state.get("execution_trace", []) + [current_agent["agent"]]
            }
        
        mock_langgraph_workflow.invoke.side_effect = mock_invoke
        
        with patch('app.services.agent_service.create_langgraph_workflow', return_value=mock_langgraph_workflow):
            agent_service = AgentService()
            
            # Execute coordinated workflow
            current_state = sample_video_state.copy()
            
            for _ in range(3):
                result = await agent_service._execute_langgraph_workflow(current_state)
                current_state.update(result)
            
            # Verify agent coordination
            assert current_state["execution_trace"] == ["planner_agent", "code_generator_agent", "renderer_agent"]
            assert current_state["scene_outline"] == "Planned outline"
            assert current_state["generated_code"] == {1: "Generated code"}
            assert current_state["rendered_videos"] == {1: "rendered.mp4"}
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_langgraph_concurrent_execution_integration(self, sample_video_state):
        """Test LangGraph concurrent execution integration."""
        # Create multiple workflow instances
        workflows = []
        for i in range(3):
            workflow = Mock()
            workflow.invoke = AsyncMock(return_value={
                "execution_id": f"concurrent_exec_{i}",
                "result": f"concurrent_result_{i}",
                "workflow_complete": True
            })
            workflows.append(workflow)
        
        with patch('app.services.agent_service.create_langgraph_workflow', side_effect=workflows):
            agent_service = AgentService()
            
            # Execute concurrent workflows
            tasks = []
            for i in range(3):
                state_copy = sample_video_state.copy()
                state_copy["session_id"] = f"concurrent_session_{i}"
                task = asyncio.create_task(agent_service._execute_langgraph_workflow(state_copy))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            # Verify concurrent execution
            assert len(results) == 3
            for i, result in enumerate(results):
                assert result["execution_id"] == f"concurrent_exec_{i}"
                assert result["result"] == f"concurrent_result_{i}"
            
            # Verify all workflows were invoked
            for workflow in workflows:
                workflow.invoke.assert_called_once()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_langgraph_checkpoint_integration(self, mock_langgraph_workflow, sample_video_state):
        """Test LangGraph checkpoint and resume integration."""
        # Mock checkpoint functionality
        checkpoint_data = {
            "checkpoint_id": "checkpoint_123",
            "state": sample_video_state,
            "timestamp": datetime.now().isoformat()
        }
        
        mock_langgraph_workflow.create_checkpoint = Mock(return_value=checkpoint_data)
        mock_langgraph_workflow.resume_from_checkpoint = AsyncMock(return_value={
            "resumed": True,
            "checkpoint_id": "checkpoint_123",
            "workflow_complete": True
        })
        
        with patch('app.services.agent_service.create_langgraph_workflow', return_value=mock_langgraph_workflow):
            agent_service = AgentService()
            
            # Create checkpoint
            checkpoint = agent_service._create_workflow_checkpoint(sample_video_state)
            assert checkpoint["checkpoint_id"] == "checkpoint_123"
            
            # Resume from checkpoint
            result = await agent_service._resume_from_checkpoint("checkpoint_123")
            assert result["resumed"] is True
            assert result["checkpoint_id"] == "checkpoint_123"
            
            # Verify checkpoint operations
            mock_langgraph_workflow.create_checkpoint.assert_called_once()
            mock_langgraph_workflow.resume_from_checkpoint.assert_called_once_with("checkpoint_123")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_langgraph_monitoring_integration(self, mock_langgraph_workflow, sample_video_state):
        """Test LangGraph monitoring and observability integration."""
        # Mock monitoring data
        monitoring_data = {
            "execution_time": 45.2,
            "memory_usage": 256.7,
            "agent_transitions": 3,
            "error_count": 0,
            "performance_metrics": {
                "planner_agent": {"time": 15.1, "memory": 85.3},
                "code_generator_agent": {"time": 20.5, "memory": 120.8},
                "renderer_agent": {"time": 9.6, "memory": 50.6}
            }
        }
        
        mock_langgraph_workflow.invoke.return_value = {
            "workflow_complete": True,
            "monitoring_data": monitoring_data
        }
        
        with patch('app.services.agent_service.create_langgraph_workflow', return_value=mock_langgraph_workflow):
            with patch('app.services.agent_service.MonitoringService') as mock_monitoring:
                mock_monitor = Mock()
                mock_monitor.record_workflow_execution = Mock()
                mock_monitoring.return_value = mock_monitor
                
                agent_service = AgentService()
                
                # Execute workflow with monitoring
                result = await agent_service._execute_langgraph_workflow_with_monitoring(sample_video_state)
                
                # Verify monitoring was recorded
                mock_monitor.record_workflow_execution.assert_called_once()
                
                # Verify monitoring data
                assert result["monitoring_data"]["execution_time"] == 45.2
                assert result["monitoring_data"]["agent_transitions"] == 3
                assert "performance_metrics" in result["monitoring_data"]
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_langgraph_human_loop_integration(self, mock_langgraph_workflow, sample_video_state):
        """Test LangGraph human-in-the-loop integration."""
        # Mock human loop interaction
        human_input_request = {
            "request_id": "human_request_123",
            "agent": "planner_agent",
            "question": "Please review the generated scene outline",
            "context": {"scene_outline": "Generated outline"},
            "status": "pending"
        }
        
        human_response = {
            "request_id": "human_request_123",
            "response": "Outline approved with minor modifications",
            "modifications": {"scene_1": "Add more detail"},
            "status": "completed"
        }
        
        mock_langgraph_workflow.request_human_input = AsyncMock(return_value=human_input_request)
        mock_langgraph_workflow.process_human_response = AsyncMock(return_value={
            "workflow_complete": True,
            "human_feedback_applied": True
        })
        
        with patch('app.services.agent_service.create_langgraph_workflow', return_value=mock_langgraph_workflow):
            agent_service = AgentService()
            
            # Request human input
            input_request = await agent_service._request_human_input(sample_video_state, "planner_agent", "Review outline")
            assert input_request["request_id"] == "human_request_123"
            
            # Process human response
            result = await agent_service._process_human_response(human_response)
            assert result["human_feedback_applied"] is True
            
            # Verify human loop operations
            mock_langgraph_workflow.request_human_input.assert_called_once()
            mock_langgraph_workflow.process_human_response.assert_called_once_with(human_response)
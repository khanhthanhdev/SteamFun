"""
Integration tests for AgentService.
Tests integration with LangGraph agents, state management, and workflow orchestration.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from app.services.agent_service import AgentService
from app.models.schemas.agent import AgentExecutionRequest, AgentExecutionResponse, AgentStatus
from app.models.enums import AgentType


class TestAgentServiceIntegration:
    """Integration test suite for AgentService."""
    
    @pytest.fixture
    def agent_service(self):
        """Create AgentService instance for integration tests."""
        return AgentService()
    
    @pytest.fixture
    def sample_execution_request(self):
        """Create sample agent execution request for integration tests."""
        return AgentExecutionRequest(
            agent_type=AgentType.PLANNER_AGENT,
            input_data={
                "topic": "Integration Test Topic",
                "description": "Testing agent service integration",
                "session_id": "integration_test_session"
            },
            config={
                "max_retries": 3,
                "timeout": 300,
                "use_monitoring": True
            }
        )
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_service_langgraph_integration(self, agent_service, sample_execution_request):
        """Test AgentService integration with LangGraph agents."""
        with patch('app.services.agent_service.LangGraphWorkflow') as mock_workflow:
            mock_graph = Mock()
            mock_graph.execute = AsyncMock(return_value={
                "execution_id": "exec_123",
                "status": "completed",
                "result": {
                    "scene_outline": "Generated scene outline",
                    "detected_plugins": ["text", "code"]
                }
            })
            mock_workflow.return_value = mock_graph
            
            # Test agent execution
            response = await agent_service.execute_agent(sample_execution_request)
            
            # Verify LangGraph workflow was executed
            mock_graph.execute.assert_called_once()
            
            # Verify response structure
            assert isinstance(response, AgentExecutionResponse)
            assert response.execution_id is not None
            assert response.agent_type == AgentType.PLANNER_AGENT
            assert response.status in [AgentStatus.RUNNING, AgentStatus.COMPLETED]
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_service_state_management_integration(self, agent_service, sample_execution_request):
        """Test AgentService integration with state management."""
        with patch('app.services.agent_service.StateManager') as mock_state_manager:
            mock_state = Mock()
            mock_state.create_execution_state = Mock(return_value={
                "execution_id": "state_test_123",
                "agent_type": "planner_agent",
                "status": "initialized",
                "created_at": datetime.now().isoformat()
            })
            mock_state.update_execution_state = Mock()
            mock_state.get_execution_state = Mock(return_value={
                "execution_id": "state_test_123",
                "status": "completed",
                "result": {"output": "test result"}
            })
            mock_state_manager.return_value = mock_state
            
            # Test state creation
            response = await agent_service.execute_agent(sample_execution_request)
            
            # Verify state was created
            mock_state.create_execution_state.assert_called_once()
            
            # Test state retrieval
            status_response = await agent_service.get_agent_status(response.execution_id)
            
            # Verify state was retrieved
            mock_state.get_execution_state.assert_called_once_with(response.execution_id)
            assert status_response.status == AgentStatus.COMPLETED
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_service_database_integration(self, agent_service, sample_execution_request):
        """Test AgentService integration with database."""
        with patch('app.services.agent_service.get_database_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value = mock_session
            
            # Mock database operations
            mock_session.add = Mock()
            mock_session.commit = Mock()
            mock_session.query = Mock()
            
            # Test agent execution (should log to database)
            response = await agent_service.execute_agent(sample_execution_request)
            
            # Verify database interactions
            mock_session.add.assert_called()
            mock_session.commit.assert_called()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_service_workflow_orchestration_integration(self, agent_service):
        """Test AgentService integration with workflow orchestration."""
        # Create workflow sequence
        workflow_requests = [
            AgentExecutionRequest(
                agent_type=AgentType.PLANNER_AGENT,
                input_data={"topic": "Test Topic"},
                config={}
            ),
            AgentExecutionRequest(
                agent_type=AgentType.CODE_GENERATOR_AGENT,
                input_data={"scene_outline": "Test outline"},
                config={}
            ),
            AgentExecutionRequest(
                agent_type=AgentType.RENDERER_AGENT,
                input_data={"generated_code": {"1": "test code"}},
                config={}
            )
        ]
        
        with patch.object(agent_service, '_execute_single_agent') as mock_execute:
            mock_responses = [
                AgentExecutionResponse(
                    execution_id=f"exec_{i}",
                    agent_type=request.agent_type,
                    status=AgentStatus.COMPLETED,
                    result={"output": f"result_{i}"}
                )
                for i, request in enumerate(workflow_requests)
            ]
            mock_execute.side_effect = mock_responses
            
            # Execute workflow
            workflow_response = await agent_service.execute_workflow(workflow_requests)
            
            # Verify all agents were executed
            assert mock_execute.call_count == 3
            assert len(workflow_response.execution_ids) == 3
            assert workflow_response.status == AgentStatus.COMPLETED
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_service_monitoring_integration(self, agent_service, sample_execution_request):
        """Test AgentService integration with monitoring systems."""
        with patch('app.services.agent_service.MonitoringService') as mock_monitoring:
            mock_monitor = Mock()
            mock_monitor.record_agent_execution = Mock()
            mock_monitor.record_execution_time = Mock()
            mock_monitor.record_error = Mock()
            mock_monitoring.return_value = mock_monitor
            
            # Test agent execution with monitoring
            response = await agent_service.execute_agent(sample_execution_request)
            
            # Verify monitoring was called
            mock_monitor.record_agent_execution.assert_called_once()
            
            # Test error monitoring
            with patch.object(agent_service, '_execute_single_agent') as mock_execute:
                mock_execute.side_effect = Exception("Agent execution failed")
                
                try:
                    await agent_service.execute_agent(sample_execution_request)
                except Exception:
                    pass
                
                # Verify error was recorded
                mock_monitor.record_error.assert_called()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_service_concurrent_execution_integration(self, agent_service):
        """Test AgentService handling of concurrent agent executions."""
        # Create multiple execution requests
        requests = [
            AgentExecutionRequest(
                agent_type=AgentType.PLANNER_AGENT,
                input_data={"topic": f"Concurrent Topic {i}"},
                config={}
            )
            for i in range(3)
        ]
        
        with patch.object(agent_service, '_execute_single_agent') as mock_execute:
            mock_responses = [
                AgentExecutionResponse(
                    execution_id=f"concurrent_exec_{i}",
                    agent_type=AgentType.PLANNER_AGENT,
                    status=AgentStatus.COMPLETED,
                    result={"output": f"concurrent_result_{i}"}
                )
                for i in range(3)
            ]
            mock_execute.side_effect = mock_responses
            
            # Execute concurrent agents
            tasks = [
                asyncio.create_task(agent_service.execute_agent(request))
                for request in requests
            ]
            
            responses = await asyncio.gather(*tasks)
            
            # Verify all agents were executed
            assert len(responses) == 3
            for i, response in enumerate(responses):
                assert response.execution_id == f"concurrent_exec_{i}"
                assert response.status == AgentStatus.COMPLETED
            
            # Verify all executions were started
            assert mock_execute.call_count == 3
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_service_error_recovery_integration(self, agent_service, sample_execution_request):
        """Test AgentService error recovery integration."""
        with patch.object(agent_service, '_execute_single_agent') as mock_execute:
            # First call fails, second succeeds
            mock_execute.side_effect = [
                Exception("Agent execution failed"),
                AgentExecutionResponse(
                    execution_id="recovery_test",
                    agent_type=AgentType.PLANNER_AGENT,
                    status=AgentStatus.COMPLETED,
                    result={"output": "recovered result"}
                )
            ]
            
            with patch.object(agent_service, '_handle_execution_error') as mock_error_handler:
                mock_error_handler.return_value = True  # Indicates retry should happen
                
                with patch.object(agent_service, '_retry_execution') as mock_retry:
                    mock_retry.return_value = AgentExecutionResponse(
                        execution_id="recovery_test",
                        agent_type=AgentType.PLANNER_AGENT,
                        status=AgentStatus.COMPLETED,
                        result={"output": "recovered result"}
                    )
                    
                    # Execute agent with error recovery
                    try:
                        response = await agent_service.execute_agent_with_retry(sample_execution_request)
                        assert response.result["output"] == "recovered result"
                    except Exception:
                        pass  # Expected on first attempt
                    
                    # Verify error handling was triggered
                    mock_error_handler.assert_called()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_service_human_loop_integration(self, agent_service, sample_execution_request):
        """Test AgentService integration with human-in-the-loop workflows."""
        with patch('app.services.agent_service.HumanLoopService') as mock_human_service:
            mock_human = Mock()
            mock_human.request_human_input = AsyncMock(return_value={
                "input_id": "human_input_123",
                "status": "pending"
            })
            mock_human.get_human_response = AsyncMock(return_value={
                "input_id": "human_input_123",
                "response": "Human approved the execution",
                "status": "completed"
            })
            mock_human_service.return_value = mock_human
            
            # Test human loop integration
            sample_execution_request.config["require_human_approval"] = True
            
            response = await agent_service.execute_agent_with_human_loop(sample_execution_request)
            
            # Verify human loop was triggered
            mock_human.request_human_input.assert_called_once()
            mock_human.get_human_response.assert_called_once()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_service_caching_integration(self, agent_service, sample_execution_request):
        """Test AgentService integration with caching systems."""
        with patch('app.services.agent_service.CacheService') as mock_cache_service:
            mock_cache = Mock()
            mock_cache.get = AsyncMock(return_value=None)  # Cache miss
            mock_cache.set = AsyncMock()
            mock_cache_service.return_value = mock_cache
            
            with patch.object(agent_service, '_execute_single_agent') as mock_execute:
                mock_response = AgentExecutionResponse(
                    execution_id="cache_test",
                    agent_type=AgentType.PLANNER_AGENT,
                    status=AgentStatus.COMPLETED,
                    result={"output": "cached result"}
                )
                mock_execute.return_value = mock_response
                
                # Test execution with caching
                response = await agent_service.execute_agent_with_cache(sample_execution_request)
                
                # Verify cache was checked and updated
                mock_cache.get.assert_called_once()
                mock_cache.set.assert_called_once()
                assert response.result["output"] == "cached result"
                
                # Test cache hit
                mock_cache.get.return_value = mock_response
                
                cached_response = await agent_service.execute_agent_with_cache(sample_execution_request)
                assert cached_response.result["output"] == "cached result"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_service_execution_history_integration(self, agent_service):
        """Test AgentService execution history integration."""
        with patch('app.services.agent_service.get_database_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value = mock_session
            
            # Mock execution history query
            mock_session.query.return_value.order_by.return_value.limit.return_value.all.return_value = [
                Mock(
                    execution_id="exec_1",
                    agent_type="planner_agent",
                    status="completed",
                    created_at=datetime.now()
                ),
                Mock(
                    execution_id="exec_2",
                    agent_type="code_generator_agent", 
                    status="failed",
                    created_at=datetime.now()
                )
            ]
            
            # Test execution history retrieval
            history = await agent_service.get_execution_history(limit=10)
            
            # Verify database was queried
            mock_session.query.assert_called_once()
            
            # Verify history structure
            assert len(history) == 2
            assert history[0]["execution_id"] == "exec_1"
            assert history[1]["agent_type"] == "code_generator_agent"
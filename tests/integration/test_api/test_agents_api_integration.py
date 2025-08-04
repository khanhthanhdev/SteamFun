"""
Integration tests for Agents API endpoints.
Tests complete agent workflow through API endpoints.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas.agent import AgentExecutionRequest, AgentExecutionResponse, AgentStatus
from app.services.agent_service import AgentService


class TestAgentsAPIIntegration:
    """Integration test suite for Agents API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client for API integration tests."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_agent_service(self):
        """Create mock agent service for integration tests."""
        service = Mock(spec=AgentService)
        service.execute_agent = AsyncMock()
        service.get_agent_status = AsyncMock()
        service.get_execution_history = AsyncMock()
        return service
    
    @pytest.mark.integration
    def test_agent_execution_workflow_integration(self, client, mock_agent_service):
        """Test complete agent execution workflow through API."""
        # Mock agent service responses
        mock_agent_service.execute_agent.return_value = AgentExecutionResponse(
            execution_id="exec_123",
            agent_type="planner_agent",
            status=AgentStatus.RUNNING,
            result=None
        )
        
        mock_agent_service.get_agent_status.return_value = AgentExecutionResponse(
            execution_id="exec_123",
            agent_type="planner_agent",
            status=AgentStatus.COMPLETED,
            result={
                "scene_outline": "Generated scene outline",
                "detected_plugins": ["text", "code"]
            }
        )
        
        with patch('app.api.v1.endpoints.agents.get_agent_service', return_value=mock_agent_service):
            # Step 1: Execute agent
            execution_request = {
                "agent_type": "planner_agent",
                "input_data": {
                    "topic": "Integration Test Topic",
                    "description": "Testing agent execution"
                },
                "config": {
                    "max_retries": 3,
                    "timeout": 300
                }
            }
            
            response = client.post("/api/v1/agents/execute", json=execution_request)
            assert response.status_code == 202
            
            data = response.json()
            execution_id = data["execution_id"]
            assert execution_id == "exec_123"
            assert data["status"] == "running"
            
            # Step 2: Check execution status
            status_response = client.get(f"/api/v1/agents/{execution_id}/status")
            assert status_response.status_code == 200
            
            status_data = status_response.json()
            assert status_data["execution_id"] == execution_id
            assert status_data["status"] == "completed"
            assert status_data["result"]["scene_outline"] == "Generated scene outline"
            
            # Verify service calls
            mock_agent_service.execute_agent.assert_called_once()
            mock_agent_service.get_agent_status.assert_called_once_with(execution_id)
    
    @pytest.mark.integration
    def test_agent_workflow_chain_integration(self, client, mock_agent_service):
        """Test chained agent execution workflow."""
        # Mock sequential agent executions
        execution_responses = [
            AgentExecutionResponse(
                execution_id=f"exec_{i}",
                agent_type=f"agent_{i}",
                status=AgentStatus.COMPLETED,
                result={"output": f"result_{i}"}
            )
            for i in range(3)
        ]
        
        mock_agent_service.execute_agent.side_effect = execution_responses
        
        with patch('app.api.v1.endpoints.agents.get_agent_service', return_value=mock_agent_service):
            # Execute chain of agents
            agent_types = ["planner_agent", "code_generator_agent", "renderer_agent"]
            execution_ids = []
            
            for agent_type in agent_types:
                request = {
                    "agent_type": agent_type,
                    "input_data": {"test": "data"},
                    "config": {}
                }
                
                response = client.post("/api/v1/agents/execute", json=request)
                assert response.status_code == 202
                
                execution_ids.append(response.json()["execution_id"])
            
            # Verify all agents were executed
            assert len(execution_ids) == 3
            assert mock_agent_service.execute_agent.call_count == 3
    
    @pytest.mark.integration
    def test_agent_api_error_handling_integration(self, client, mock_agent_service):
        """Test agent API error handling integration."""
        # Mock service errors
        mock_agent_service.execute_agent.side_effect = Exception("Agent execution failed")
        
        with patch('app.api.v1.endpoints.agents.get_agent_service', return_value=mock_agent_service):
            request = {
                "agent_type": "invalid_agent",
                "input_data": {"test": "data"}
            }
            
            response = client.post("/api/v1/agents/execute", json=request)
            assert response.status_code == 500
    
    @pytest.mark.integration
    def test_agent_api_validation_integration(self, client):
        """Test agent API request validation integration."""
        # Test missing required fields
        invalid_request = {
            "input_data": {"test": "data"}
            # Missing agent_type
        }
        
        response = client.post("/api/v1/agents/execute", json=invalid_request)
        assert response.status_code == 422
        
        error_data = response.json()
        assert "detail" in error_data
        assert any("agent_type" in str(error) for error in error_data["detail"])
    
    @pytest.mark.integration
    def test_agent_execution_history_integration(self, client, mock_agent_service):
        """Test agent execution history retrieval."""
        # Mock execution history
        mock_agent_service.get_execution_history.return_value = [
            {
                "execution_id": "exec_1",
                "agent_type": "planner_agent",
                "status": "completed",
                "created_at": "2024-01-01T00:00:00Z"
            },
            {
                "execution_id": "exec_2", 
                "agent_type": "code_generator_agent",
                "status": "failed",
                "created_at": "2024-01-01T01:00:00Z"
            }
        ]
        
        with patch('app.api.v1.endpoints.agents.get_agent_service', return_value=mock_agent_service):
            response = client.get("/api/v1/agents/history")
            assert response.status_code == 200
            
            data = response.json()
            assert len(data) == 2
            assert data[0]["execution_id"] == "exec_1"
            assert data[1]["agent_type"] == "code_generator_agent"
            
            mock_agent_service.get_execution_history.assert_called_once()
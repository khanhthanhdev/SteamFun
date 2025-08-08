"""
API Client for FastAPI Backend Communication

This module provides HTTP client functionality for communicating with the FastAPI
backend testing endpoints. It handles agent discovery, execution requests, and
result retrieval.
"""

import asyncio
import aiohttp
import requests
import json
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class TestAPIClient:
    """HTTP client for FastAPI backend communication."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = None
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    def get_available_agents(self) -> List[Dict[str, Any]]:
        """
        Get list of available agents for testing.
        
        Returns:
            List of agent information dictionaries
        """
        try:
            response = requests.get(f"{self.base_url}/test/agents", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error getting available agents: {e}")
            raise ConnectionError(
                f"Unable to connect to backend server at {self.base_url}. "
                f"Please check if the server is running and the URL is correct."
            )
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout error getting available agents: {e}")
            raise TimeoutError(
                f"Request timed out after 10 seconds. "
                f"The backend server may be overloaded or unresponsive."
            )
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error getting available agents: {e}")
            if e.response.status_code == 404:
                raise Exception(
                    f"Agents endpoint not found. "
                    f"Please check if the backend server is properly configured."
                )
            elif e.response.status_code == 500:
                raise Exception(
                    f"Backend server error (500). "
                    f"Please check the server logs for details."
                )
            else:
                raise Exception(f"HTTP error {e.response.status_code}: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error getting available agents: {e}")
            raise Exception(f"Failed to get available agents: {e}")
    
    async def get_available_agents_async(self) -> List[Dict[str, Any]]:
        """
        Async version of get_available_agents.
        
        Returns:
            List of agent information dictionaries
        """
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/test/agents") as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Failed to get available agents: {e}")
            raise Exception(f"Failed to connect to backend: {e}")
    
    def test_individual_agent(
        self, 
        agent_name: str, 
        inputs: Dict[str, Any], 
        session_id: Optional[str] = None,
        options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Test an individual agent with provided inputs.
        
        Args:
            agent_name: Name of the agent to test
            inputs: Input data for the agent
            session_id: Optional session ID for tracking
            options: Optional execution options
            
        Returns:
            Test response dictionary
        """
        try:
            payload = {
                "inputs": inputs,
                "session_id": session_id,
                "options": options or {}
            }
            
            response = requests.post(
                f"{self.base_url}/test/agent/{agent_name}",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error testing agent {agent_name}: {e}")
            raise ConnectionError(
                f"Unable to connect to backend server. "
                f"Please check if the server is running and try again."
            )
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout error testing agent {agent_name}: {e}")
            raise TimeoutError(
                f"Agent test timed out after 30 seconds. "
                f"The agent may be taking longer than expected to process."
            )
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error testing agent {agent_name}: {e}")
            if e.response.status_code == 400:
                try:
                    error_detail = e.response.json().get('detail', 'Invalid request')
                    raise ValueError(f"Invalid input data: {error_detail}")
                except:
                    raise ValueError("Invalid input data provided to the agent")
            elif e.response.status_code == 404:
                raise Exception(
                    f"Agent '{agent_name}' not found. "
                    f"Please check the agent name and refresh the agent list."
                )
            elif e.response.status_code == 500:
                raise Exception(
                    f"Agent execution failed due to server error. "
                    f"Please check the server logs and try again."
                )
            else:
                raise Exception(f"HTTP error {e.response.status_code}: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error testing agent {agent_name}: {e}")
            raise Exception(f"Agent test failed: {e}")
    
    async def test_individual_agent_async(
        self, 
        agent_name: str, 
        inputs: Dict[str, Any], 
        session_id: Optional[str] = None,
        options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Async version of test_individual_agent.
        
        Args:
            agent_name: Name of the agent to test
            inputs: Input data for the agent
            session_id: Optional session ID for tracking
            options: Optional execution options
            
        Returns:
            Test response dictionary
        """
        try:
            payload = {
                "inputs": inputs,
                "session_id": session_id,
                "options": options or {}
            }
            
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/test/agent/{agent_name}",
                json=payload
            ) as response:
                response.raise_for_status()
                return await response.json()
                
        except aiohttp.ClientError as e:
            logger.error(f"Failed to test agent {agent_name}: {e}")
            raise Exception(f"Agent test failed: {e}")
    
    def test_complete_workflow(
        self,
        topic: str,
        description: str,
        session_id: Optional[str] = None,
        config_overrides: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Test the complete workflow from input to video output.
        
        Args:
            topic: Topic for the video generation
            description: Detailed description
            session_id: Optional session ID for tracking
            config_overrides: Optional configuration overrides
            
        Returns:
            Workflow test response dictionary
        """
        try:
            payload = {
                "topic": topic,
                "description": description,
                "session_id": session_id,
                "config_overrides": config_overrides or {}
            }
            
            response = requests.post(
                f"{self.base_url}/test/workflow",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error testing workflow: {e}")
            raise ConnectionError(
                f"Unable to connect to backend server. "
                f"Please check if the server is running and try again."
            )
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout error testing workflow: {e}")
            raise TimeoutError(
                f"Workflow test timed out after 30 seconds. "
                f"The workflow may take longer to complete. Check the logs for progress."
            )
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error testing workflow: {e}")
            if e.response.status_code == 400:
                try:
                    error_detail = e.response.json().get('detail', 'Invalid request')
                    raise ValueError(f"Invalid workflow parameters: {error_detail}")
                except:
                    raise ValueError("Invalid workflow parameters provided")
            elif e.response.status_code == 404:
                raise Exception(
                    f"Workflow endpoint not found. "
                    f"Please check if the backend server supports workflow testing."
                )
            elif e.response.status_code == 500:
                raise Exception(
                    f"Workflow execution failed due to server error. "
                    f"Please check the server logs and try again."
                )
            else:
                raise Exception(f"HTTP error {e.response.status_code}: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error testing workflow: {e}")
            raise Exception(f"Workflow test failed: {e}")
    
    async def test_complete_workflow_async(
        self,
        topic: str,
        description: str,
        session_id: Optional[str] = None,
        config_overrides: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Async version of test_complete_workflow.
        
        Args:
            topic: Topic for the video generation
            description: Detailed description
            session_id: Optional session ID for tracking
            config_overrides: Optional configuration overrides
            
        Returns:
            Workflow test response dictionary
        """
        try:
            payload = {
                "topic": topic,
                "description": description,
                "session_id": session_id,
                "config_overrides": config_overrides or {}
            }
            
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/test/workflow",
                json=payload
            ) as response:
                response.raise_for_status()
                return await response.json()
                
        except aiohttp.ClientError as e:
            logger.error(f"Failed to test workflow: {e}")
            raise Exception(f"Workflow test failed: {e}")
    
    def get_session_logs(self, session_id: str) -> Dict[str, Any]:
        """
        Get logs for a specific test session.
        
        Args:
            session_id: Session ID to get logs for
            
        Returns:
            Session logs dictionary
        """
        try:
            response = requests.get(
                f"{self.base_url}/test/logs/{session_id}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get session logs: {e}")
            raise Exception(f"Failed to get logs: {e}")
    
    async def get_session_logs_async(self, session_id: str) -> Dict[str, Any]:
        """
        Async version of get_session_logs.
        
        Args:
            session_id: Session ID to get logs for
            
        Returns:
            Session logs dictionary
        """
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/test/logs/{session_id}") as response:
                response.raise_for_status()
                return await response.json()
                
        except aiohttp.ClientError as e:
            logger.error(f"Failed to get session logs: {e}")
            raise Exception(f"Failed to get logs: {e}")
    
    def list_test_sessions(
        self,
        status_filter: Optional[str] = None,
        session_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        List all test sessions with optional filtering.
        
        Args:
            status_filter: Optional status filter
            session_type: Optional session type filter
            limit: Maximum number of sessions to return
            offset: Offset for pagination
            
        Returns:
            Sessions list dictionary
        """
        try:
            params = {
                "limit": limit,
                "offset": offset
            }
            
            if status_filter:
                params["status_filter"] = status_filter
            if session_type:
                params["session_type"] = session_type
            
            response = requests.get(
                f"{self.base_url}/test/sessions",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to list test sessions: {e}")
            raise Exception(f"Failed to list sessions: {e}")
    
    def health_check(self) -> bool:
        """
        Check if the backend API is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def detailed_health_check(self) -> Tuple[bool, str, List[str]]:
        """
        Perform a detailed health check with error reporting.
        
        Returns:
            Tuple of (is_healthy, status_message, recovery_suggestions)
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            
            if response.status_code == 200:
                return True, "✅ Backend server is healthy and responding", []
            else:
                return False, f"❌ Backend server returned status {response.status_code}", [
                    "Check if the backend server is properly configured",
                    "Review server logs for any errors",
                    "Try restarting the backend server"
                ]
                
        except requests.exceptions.ConnectionError:
            return False, "❌ Cannot connect to backend server", [
                "Check if the backend server is running",
                "Verify the backend URL is correct",
                "Check your network connection",
                "Ensure no firewall is blocking the connection"
            ]
        except requests.exceptions.Timeout:
            return False, "❌ Backend server is not responding (timeout)", [
                "The server may be overloaded",
                "Try again in a few moments",
                "Check server performance and resources",
                "Consider increasing the timeout value"
            ]
        except Exception as e:
            return False, f"❌ Health check failed: {str(e)}", [
                "Check the backend server configuration",
                "Review server logs for errors",
                "Try refreshing the page",
                "Contact system administrator if problem persists"
            ]
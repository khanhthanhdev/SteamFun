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
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get available agents: {e}")
            raise Exception(f"Failed to connect to backend: {e}")
    
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
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to test agent {agent_name}: {e}")
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
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to test workflow: {e}")
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
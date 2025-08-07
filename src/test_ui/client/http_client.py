"""
HTTP client for FastAPI communication with async request handling,
error handling, and retry logic.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

import aiohttp
from aiohttp import ClientSession, ClientTimeout, ClientError


logger = logging.getLogger(__name__)


class RequestMethod(Enum):
    """HTTP request methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    retry_on_status: List[int] = None
    
    def __post_init__(self):
        if self.retry_on_status is None:
            self.retry_on_status = [500, 502, 503, 504, 429]


@dataclass
class AgentInfo:
    """Information about an available agent."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    dependencies: List[str]
    test_examples: List[Dict[str, Any]]


@dataclass
class TestRequest:
    """Request for testing an agent."""
    agent_name: str
    inputs: Dict[str, Any]
    session_id: str
    options: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.options is None:
            self.options = {}


@dataclass
class TestResponse:
    """Response from agent testing."""
    session_id: str
    status: str
    results: Dict[str, Any]
    execution_time: float
    logs: List[Dict[str, Any]]
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class HTTPClientError(Exception):
    """Base exception for HTTP client errors."""
    pass


class ConnectionError(HTTPClientError):
    """Raised when connection to backend fails."""
    pass


class RequestError(HTTPClientError):
    """Raised when request fails."""
    pass


class ValidationError(HTTPClientError):
    """Raised when request validation fails."""
    pass


class HTTPClient:
    """
    HTTP client for FastAPI communication with async request handling,
    error handling, and retry logic.
    """
    
    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        retry_config: Optional[RetryConfig] = None
    ):
        """
        Initialize HTTP client.
        
        Args:
            base_url: Base URL of the FastAPI backend
            timeout: Request timeout in seconds
            retry_config: Configuration for retry logic
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = ClientTimeout(total=timeout)
        self.retry_config = retry_config or RetryConfig()
        self._session: Optional[ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        
    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if self._session is None or self._session.closed:
            self._session = ClientSession(
                timeout=self.timeout,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
            )
            
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            
    async def _make_request(
        self,
        method: RequestMethod,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request body data
            params: Query parameters
            
        Returns:
            Response data
            
        Raises:
            ConnectionError: If connection fails
            RequestError: If request fails
            ValidationError: If validation fails
        """
        await self._ensure_session()
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                logger.debug(f"Making {method.value} request to {url} (attempt {attempt + 1})")
                
                kwargs = {
                    "url": url,
                    "params": params
                }
                
                if data is not None:
                    kwargs["json"] = data
                    
                async with self._session.request(method.value, **kwargs) as response:
                    response_text = await response.text()
                    
                    # Handle different response status codes
                    if response.status == 200:
                        try:
                            return json.loads(response_text) if response_text else {}
                        except json.JSONDecodeError as e:
                            raise RequestError(f"Invalid JSON response: {e}")
                            
                    elif response.status == 422:
                        try:
                            error_data = json.loads(response_text)
                            raise ValidationError(f"Validation error: {error_data}")
                        except json.JSONDecodeError:
                            raise ValidationError(f"Validation error: {response_text}")
                            
                    elif response.status in self.retry_config.retry_on_status:
                        if attempt < self.retry_config.max_retries:
                            delay = min(
                                self.retry_config.base_delay * (self.retry_config.backoff_factor ** attempt),
                                self.retry_config.max_delay
                            )
                            logger.warning(f"Request failed with status {response.status}, retrying in {delay}s")
                            await asyncio.sleep(delay)
                            continue
                        else:
                            raise RequestError(f"Request failed with status {response.status}: {response_text}")
                    else:
                        raise RequestError(f"Request failed with status {response.status}: {response_text}")
                        
            except ClientError as e:
                if attempt < self.retry_config.max_retries:
                    delay = min(
                        self.retry_config.base_delay * (self.retry_config.backoff_factor ** attempt),
                        self.retry_config.max_delay
                    )
                    logger.warning(f"Connection error: {e}, retrying in {delay}s")
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise ConnectionError(f"Connection failed after {self.retry_config.max_retries} retries: {e}")
                    
        raise RequestError("Max retries exceeded")
        
    async def get_available_agents(self) -> List[AgentInfo]:
        """
        Get list of available agents for testing.
        
        Returns:
            List of available agents
            
        Raises:
            ConnectionError: If connection fails
            RequestError: If request fails
        """
        try:
            response = await self._make_request(RequestMethod.GET, "/agents")
            agents = []
            
            for agent_data in response.get("agents", []):
                agents.append(AgentInfo(
                    name=agent_data["name"],
                    description=agent_data["description"],
                    input_schema=agent_data["input_schema"],
                    output_schema=agent_data["output_schema"],
                    dependencies=agent_data.get("dependencies", []),
                    test_examples=agent_data.get("test_examples", [])
                ))
                
            logger.info(f"Retrieved {len(agents)} available agents")
            return agents
            
        except Exception as e:
            logger.error(f"Failed to get available agents: {e}")
            raise
            
    async def test_agent(self, request: TestRequest) -> TestResponse:
        """
        Test individual agent with provided inputs.
        
        Args:
            request: Test request with agent name and inputs
            
        Returns:
            Test response with results
            
        Raises:
            ConnectionError: If connection fails
            RequestError: If request fails
            ValidationError: If validation fails
        """
        try:
            endpoint = f"/test/agent/{request.agent_name}"
            data = {
                "inputs": request.inputs,
                "session_id": request.session_id,
                "options": request.options
            }
            
            response = await self._make_request(RequestMethod.POST, endpoint, data)
            
            test_response = TestResponse(
                session_id=response["session_id"],
                status=response["status"],
                results=response["results"],
                execution_time=response["execution_time"],
                logs=response.get("logs", []),
                errors=response.get("errors", [])
            )
            
            logger.info(f"Agent test completed: {request.agent_name} (session: {request.session_id})")
            return test_response
            
        except Exception as e:
            logger.error(f"Failed to test agent {request.agent_name}: {e}")
            raise
            
    async def test_workflow(self, inputs: Dict[str, Any], session_id: str) -> TestResponse:
        """
        Test complete workflow with provided inputs.
        
        Args:
            inputs: Workflow inputs
            session_id: Session identifier
            
        Returns:
            Test response with results
            
        Raises:
            ConnectionError: If connection fails
            RequestError: If request fails
            ValidationError: If validation fails
        """
        try:
            data = {
                "inputs": inputs,
                "session_id": session_id
            }
            
            response = await self._make_request(RequestMethod.POST, "/test/workflow", data)
            
            test_response = TestResponse(
                session_id=response["session_id"],
                status=response["status"],
                results=response["results"],
                execution_time=response["execution_time"],
                logs=response.get("logs", []),
                errors=response.get("errors", [])
            )
            
            logger.info(f"Workflow test completed (session: {session_id})")
            return test_response
            
        except Exception as e:
            logger.error(f"Failed to test workflow: {e}")
            raise
            
    async def get_session_logs(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get logs for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of log entries
            
        Raises:
            ConnectionError: If connection fails
            RequestError: If request fails
        """
        try:
            endpoint = f"/logs/{session_id}"
            response = await self._make_request(RequestMethod.GET, endpoint)
            
            logs = response.get("logs", [])
            logger.debug(f"Retrieved {len(logs)} log entries for session {session_id}")
            return logs
            
        except Exception as e:
            logger.error(f"Failed to get logs for session {session_id}: {e}")
            raise
            
    async def health_check(self) -> Dict[str, Any]:
        """
        Check backend health status.
        
        Returns:
            Health status information
            
        Raises:
            ConnectionError: If connection fails
            RequestError: If request fails
        """
        try:
            response = await self._make_request(RequestMethod.GET, "/health")
            logger.debug("Health check successful")
            return response
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise
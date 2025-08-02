"""
MCP (Model Context Protocol) integration tools for LangGraph agents.

This module provides tools for connecting to and interacting with MCP servers,
allowing agents to access external APIs and specialized tools through the MCP protocol.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from contextlib import asynccontextmanager
import aiohttp

from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection."""
    name: str
    command: str = None
    args: List[str] = None
    url: str = None
    env: Dict[str, str] = None
    disabled: bool = False
    auto_approve: List[str] = None
    timeout: int = 30
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.command and not self.url:
            raise ValueError("Either 'command' or 'url' must be specified")
        if self.command and self.url:
            raise ValueError("Cannot specify both 'command' and 'url'")
        if self.args is None:
            self.args = []
        if self.env is None:
            self.env = {}
        if self.auto_approve is None:
            self.auto_approve = []
    
    @property
    def is_http(self) -> bool:
        """Check if this is an HTTP-based MCP server."""
        return self.url is not None


class MCPConnectionError(Exception):
    """Raised when MCP server connection fails."""
    pass


class MCPToolError(Exception):
    """Raised when MCP tool execution fails."""
    pass


class MCPClient:
    """Client for connecting to and managing MCP servers."""
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.process = None
        self.http_session = None
        self.connected = False
        self.available_tools = {}
        self._lock = asyncio.Lock()
        self._request_id = 0
    
    async def connect(self) -> bool:
        """Connect to the MCP server."""
        async with self._lock:
            if self.connected:
                return True
            
            if self.config.disabled:
                logger.info(f"MCP server {self.config.name} is disabled")
                return False
            
            try:
                if self.config.is_http:
                    await self._connect_http()
                else:
                    await self._connect_process()
                
                # Initialize MCP protocol handshake
                await self._initialize_protocol()
                
                # Load available tools
                await self._load_tools()
                
                self.connected = True
                logger.info(f"Successfully connected to MCP server: {self.config.name}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to connect to MCP server {self.config.name}: {e}")
                await self.disconnect()
                raise MCPConnectionError(f"Connection failed: {e}")
    
    async def _connect_http(self):
        """Connect to HTTP-based MCP server."""
        self.http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        
        # Test the connection with a simple request
        try:
            async with self.http_session.get(self.config.url) as response:
                if response.status != 200:
                    raise MCPConnectionError(f"HTTP server returned status {response.status}")
        except aiohttp.ClientError as e:
            raise MCPConnectionError(f"HTTP connection failed: {e}")
    
    async def _connect_process(self):
        """Connect to process-based MCP server."""
        env = dict(self.config.env) if self.config.env else {}
        
        self.process = await asyncio.create_subprocess_exec(
            self.config.command,
            *self.config.args,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
    
    async def disconnect(self):
        """Disconnect from the MCP server."""
        async with self._lock:
            if self.process:
                try:
                    self.process.terminate()
                    await asyncio.wait_for(self.process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self.process.kill()
                    await self.process.wait()
                except Exception as e:
                    logger.error(f"Error disconnecting from MCP server: {e}")
                finally:
                    self.process = None
            
            if self.http_session:
                try:
                    await self.http_session.close()
                except Exception as e:
                    logger.error(f"Error closing HTTP session: {e}")
                finally:
                    self.http_session = None
            
            self.connected = False
            self.available_tools = {}
            logger.info(f"Disconnected from MCP server: {self.config.name}")
    
    async def _initialize_protocol(self):
        """Initialize the MCP protocol with handshake."""
        if not self.process and not self.http_session:
            raise MCPConnectionError("No connection available for protocol initialization")
        
        # Send initialization message
        init_message = {
            "jsonrpc": "2.0",
            "id": self._get_next_request_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {
                    "name": "langgraph-video-generator",
                    "version": "1.0.0"
                }
            }
        }
        
        await self._send_message(init_message)
        response = await self._receive_message()
        
        if response.get("error"):
            raise MCPConnectionError(f"Initialization failed: {response['error']}")
        
        # Send initialized notification
        initialized_message = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        await self._send_message(initialized_message)
    
    async def _load_tools(self):
        """Load available tools from the MCP server."""
        tools_message = {
            "jsonrpc": "2.0",
            "id": self._get_next_request_id(),
            "method": "tools/list"
        }
        
        await self._send_message(tools_message)
        response = await self._receive_message()
        
        if response.get("error"):
            logger.error(f"Failed to load tools: {response['error']}")
            return
        
        tools = response.get("result", {}).get("tools", [])
        for tool_info in tools:
            tool_name = tool_info.get("name")
            if tool_name:
                self.available_tools[tool_name] = tool_info
                logger.debug(f"Loaded MCP tool: {tool_name}")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the MCP server."""
        if not self.connected:
            raise MCPConnectionError("Not connected to MCP server")
        
        if tool_name not in self.available_tools:
            raise MCPToolError(f"Tool {tool_name} not available on server {self.config.name}")
        
        # Check if tool requires approval
        if (self.config.auto_approve and 
            tool_name not in self.config.auto_approve):
            logger.warning(f"Tool {tool_name} requires manual approval")
            # In a real implementation, this would trigger human approval
        
        call_message = {
            "jsonrpc": "2.0",
            "id": self._get_next_request_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        try:
            await self._send_message(call_message)
            response = await asyncio.wait_for(
                self._receive_message(), 
                timeout=self.config.timeout
            )
            
            if response.get("error"):
                raise MCPToolError(f"Tool call failed: {response['error']}")
            
            return response.get("result", {})
            
        except asyncio.TimeoutError:
            raise MCPToolError(f"Tool call timed out after {self.config.timeout}s")
        except Exception as e:
            raise MCPToolError(f"Tool call error: {e}")
    
    def _get_next_request_id(self) -> int:
        """Get the next request ID."""
        self._request_id += 1
        return self._request_id
    
    async def _send_message(self, message: Dict[str, Any]):
        """Send a JSON-RPC message to the MCP server."""
        if self.config.is_http:
            await self._send_http_message(message)
        else:
            await self._send_process_message(message)
    
    async def _receive_message(self) -> Dict[str, Any]:
        """Receive a JSON-RPC message from the MCP server."""
        if self.config.is_http:
            return await self._receive_http_message()
        else:
            return await self._receive_process_message()
    
    async def _send_http_message(self, message: Dict[str, Any]):
        """Send a JSON-RPC message via HTTP."""
        if not self.http_session:
            raise MCPConnectionError("No HTTP session available")
        
        try:
            async with self.http_session.post(
                self.config.url,
                json=message,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    raise MCPConnectionError(f"HTTP request failed with status {response.status}")
                
                # Store the response for later retrieval
                self._last_http_response = await response.json()
        except aiohttp.ClientError as e:
            raise MCPConnectionError(f"HTTP request failed: {e}")
    
    async def _receive_http_message(self) -> Dict[str, Any]:
        """Receive a JSON-RPC message via HTTP."""
        if not hasattr(self, '_last_http_response'):
            raise MCPConnectionError("No HTTP response available")
        
        response = self._last_http_response
        delattr(self, '_last_http_response')
        return response
    
    async def _send_process_message(self, message: Dict[str, Any]):
        """Send a JSON-RPC message via process stdin."""
        if not self.process or not self.process.stdin:
            raise MCPConnectionError("No stdin available for sending messages")
        
        message_str = json.dumps(message) + "\n"
        self.process.stdin.write(message_str.encode())
        await self.process.stdin.drain()
    
    async def _receive_process_message(self) -> Dict[str, Any]:
        """Receive a JSON-RPC message via process stdout."""
        if not self.process or not self.process.stdout:
            raise MCPConnectionError("No stdout available for receiving messages")
        
        line = await self.process.stdout.readline()
        if not line:
            raise MCPConnectionError("Connection closed by server")
        
        try:
            return json.loads(line.decode().strip())
        except json.JSONDecodeError as e:
            raise MCPConnectionError(f"Invalid JSON received: {e}")


class MCPManager:
    """Manager for multiple MCP server connections."""
    
    def __init__(self):
        self.clients: Dict[str, MCPClient] = {}
        self.configs: Dict[str, MCPServerConfig] = {}
    
    def add_server(self, config: MCPServerConfig):
        """Add an MCP server configuration."""
        self.configs[config.name] = config
        self.clients[config.name] = MCPClient(config)
    
    async def connect_all(self):
        """Connect to all configured MCP servers."""
        connection_tasks = []
        for name, client in self.clients.items():
            if not self.configs[name].disabled:
                connection_tasks.append(self._safe_connect(name, client))
        
        if connection_tasks:
            await asyncio.gather(*connection_tasks, return_exceptions=True)
    
    async def _safe_connect(self, name: str, client: MCPClient):
        """Safely connect to an MCP server with error handling."""
        try:
            await client.connect()
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {name}: {e}")
    
    async def disconnect_all(self):
        """Disconnect from all MCP servers."""
        disconnect_tasks = []
        for client in self.clients.values():
            if client.connected:
                disconnect_tasks.append(client.disconnect())
        
        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)
    
    def get_client(self, server_name: str) -> Optional[MCPClient]:
        """Get an MCP client by server name."""
        return self.clients.get(server_name)
    
    def list_available_tools(self) -> Dict[str, List[str]]:
        """List all available tools across all connected servers."""
        tools_by_server = {}
        for name, client in self.clients.items():
            if client.connected:
                tools_by_server[name] = list(client.available_tools.keys())
        return tools_by_server
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on a specific MCP server."""
        client = self.get_client(server_name)
        if not client:
            raise MCPToolError(f"MCP server {server_name} not found")
        
        return await client.call_tool(tool_name, arguments)


# Global MCP manager instance
mcp_manager = MCPManager()


# LangChain tools for MCP integration
class MCPToolCall(BaseModel):
    """Input schema for MCP tool calls."""
    server_name: str = Field(description="Name of the MCP server to use")
    tool_name: str = Field(description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(description="Arguments to pass to the tool")


@tool("mcp_call_tool", args_schema=MCPToolCall)
async def mcp_call_tool(server_name: str, tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Call a tool on an MCP server.
    
    Args:
        server_name: Name of the MCP server to use
        tool_name: Name of the tool to call
        arguments: Arguments to pass to the tool
    
    Returns:
        JSON string containing the tool result
    """
    try:
        result = await mcp_manager.call_tool(server_name, tool_name, arguments)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"MCP tool call failed: {e}")
        return json.dumps({"error": str(e)})


@tool("mcp_list_tools")
def mcp_list_tools() -> str:
    """
    List all available tools across all connected MCP servers.
    
    Returns:
        JSON string containing tools organized by server
    """
    try:
        tools = mcp_manager.list_available_tools()
        return json.dumps(tools, indent=2)
    except Exception as e:
        logger.error(f"Failed to list MCP tools: {e}")
        return json.dumps({"error": str(e)})


@tool("mcp_server_status")
def mcp_server_status() -> str:
    """
    Get the connection status of all MCP servers.
    
    Returns:
        JSON string containing server connection status
    """
    try:
        status = {}
        for name, client in mcp_manager.clients.items():
            status[name] = {
                "connected": client.connected,
                "disabled": mcp_manager.configs[name].disabled,
                "tools_count": len(client.available_tools)
            }
        return json.dumps(status, indent=2)
    except Exception as e:
        logger.error(f"Failed to get MCP server status: {e}")
        return json.dumps({"error": str(e)})


# Context manager for MCP connections
@asynccontextmanager
async def mcp_context(configs: List[MCPServerConfig]):
    """Context manager for MCP server connections."""
    # Add all server configurations
    for config in configs:
        mcp_manager.add_server(config)
    
    try:
        # Connect to all servers
        await mcp_manager.connect_all()
        yield mcp_manager
    finally:
        # Disconnect from all servers
        await mcp_manager.disconnect_all()


# Export main components
__all__ = [
    "MCPServerConfig",
    "MCPClient", 
    "MCPManager",
    "MCPConnectionError",
    "MCPToolError",
    "mcp_manager",
    "mcp_call_tool",
    "mcp_list_tools", 
    "mcp_server_status",
    "mcp_context"
]
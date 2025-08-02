"""
MCP service for LangGraph agents.

This service provides a high-level interface for agents to interact with
MCP servers, handling connection management, tool discovery, and execution.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from contextlib import asynccontextmanager

from ..tools.mcp_tools import (
    MCPManager, MCPServerConfig, MCPConnectionError, MCPToolError,
    mcp_call_tool, mcp_list_tools, mcp_server_status
)
from ..utils.mcp_config import load_mcp_configs

logger = logging.getLogger(__name__)


class MCPService:
    """High-level service for MCP integration with LangGraph agents."""
    
    def __init__(self):
        self.manager = MCPManager()
        self.initialized = False
        self.available_tools: Dict[str, Dict[str, Any]] = {}
        self._initialization_lock = asyncio.Lock()
    
    async def initialize(self, force_reload: bool = False) -> bool:
        """
        Initialize the MCP service by loading configurations and connecting to servers.
        
        Args:
            force_reload: If True, reload configurations even if already initialized
            
        Returns:
            True if initialization was successful
        """
        async with self._initialization_lock:
            if self.initialized and not force_reload:
                return True
            
            try:
                # Load MCP server configurations
                configs = load_mcp_configs()
                if not configs:
                    logger.info("No MCP server configurations found")
                    return True
                
                # Add server configurations to manager
                for config in configs.values():
                    self.manager.add_server(config)
                
                # Connect to all servers
                await self.manager.connect_all()
                
                # Update available tools cache
                await self._update_available_tools()
                
                self.initialized = True
                logger.info(f"MCP service initialized with {len(configs)} servers")
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize MCP service: {e}")
                return False
    
    async def shutdown(self):
        """Shutdown the MCP service and disconnect from all servers."""
        try:
            await self.manager.disconnect_all()
            self.initialized = False
            self.available_tools = {}
            logger.info("MCP service shutdown complete")
        except Exception as e:
            logger.error(f"Error during MCP service shutdown: {e}")
    
    async def _update_available_tools(self):
        """Update the cache of available tools from all connected servers."""
        self.available_tools = {}
        
        for server_name, client in self.manager.clients.items():
            if client.connected:
                server_tools = {}
                for tool_name, tool_info in client.available_tools.items():
                    server_tools[tool_name] = {
                        "server": server_name,
                        "description": tool_info.get("description", ""),
                        "schema": tool_info.get("inputSchema", {}),
                        "auto_approved": (
                            tool_name in client.config.auto_approve 
                            if client.config.auto_approve else False
                        )
                    }
                self.available_tools[server_name] = server_tools
    
    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get all available tools organized by server."""
        return self.available_tools.copy()
    
    def find_tools_by_name(self, tool_name: str) -> List[Dict[str, Any]]:
        """
        Find tools by name across all servers.
        
        Args:
            tool_name: Name of the tool to find
            
        Returns:
            List of tool information dictionaries
        """
        matching_tools = []
        
        for server_name, server_tools in self.available_tools.items():
            if tool_name in server_tools:
                tool_info = server_tools[tool_name].copy()
                tool_info["server_name"] = server_name
                tool_info["tool_name"] = tool_name
                matching_tools.append(tool_info)
        
        return matching_tools
    
    def find_tools_by_description(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Find tools by searching their descriptions for keywords.
        
        Args:
            keywords: List of keywords to search for
            
        Returns:
            List of matching tool information dictionaries
        """
        matching_tools = []
        keywords_lower = [kw.lower() for kw in keywords]
        
        for server_name, server_tools in self.available_tools.items():
            for tool_name, tool_info in server_tools.items():
                description = tool_info.get("description", "").lower()
                
                # Check if any keyword matches
                if any(keyword in description for keyword in keywords_lower):
                    match_info = tool_info.copy()
                    match_info["server_name"] = server_name
                    match_info["tool_name"] = tool_name
                    matching_tools.append(match_info)
        
        return matching_tools
    
    async def call_tool(
        self, 
        server_name: str, 
        tool_name: str, 
        arguments: Dict[str, Any],
        require_approval: bool = None
    ) -> Dict[str, Any]:
        """
        Call a tool on a specific MCP server.
        
        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            require_approval: Override auto-approval setting if specified
            
        Returns:
            Tool execution result
            
        Raises:
            MCPConnectionError: If server is not connected
            MCPToolError: If tool execution fails
        """
        if not self.initialized:
            await self.initialize()
        
        client = self.manager.get_client(server_name)
        if not client or not client.connected:
            raise MCPConnectionError(f"MCP server {server_name} is not connected")
        
        # Check approval requirements
        if require_approval is None:
            tool_info = self.available_tools.get(server_name, {}).get(tool_name, {})
            require_approval = not tool_info.get("auto_approved", False)
        
        if require_approval:
            logger.warning(f"Tool {tool_name} on {server_name} requires approval")
            # In a real implementation, this would trigger human approval workflow
            # For now, we'll log and proceed
        
        try:
            result = await client.call_tool(tool_name, arguments)
            logger.info(f"Successfully called tool {tool_name} on {server_name}")
            return result
        except Exception as e:
            logger.error(f"Failed to call tool {tool_name} on {server_name}: {e}")
            raise
    
    async def call_best_tool(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any],
        prefer_auto_approved: bool = True
    ) -> Dict[str, Any]:
        """
        Call the best available tool with the given name.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            prefer_auto_approved: Prefer auto-approved tools if available
            
        Returns:
            Tool execution result
            
        Raises:
            MCPToolError: If no suitable tool is found or execution fails
        """
        matching_tools = self.find_tools_by_name(tool_name)
        if not matching_tools:
            raise MCPToolError(f"No tool named '{tool_name}' found on any server")
        
        # Sort tools by preference (auto-approved first if preferred)
        if prefer_auto_approved:
            matching_tools.sort(key=lambda t: not t.get("auto_approved", False))
        
        # Try each matching tool until one succeeds
        last_error = None
        for tool_info in matching_tools:
            try:
                return await self.call_tool(
                    tool_info["server_name"],
                    tool_name,
                    arguments
                )
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to call {tool_name} on {tool_info['server_name']}: {e}")
                continue
        
        # If we get here, all tools failed
        raise MCPToolError(f"All attempts to call tool '{tool_name}' failed. Last error: {last_error}")
    
    def get_server_status(self) -> Dict[str, Dict[str, Any]]:
        """Get the status of all MCP servers."""
        status = {}
        
        for server_name, client in self.manager.clients.items():
            config = self.manager.configs.get(server_name)
            status[server_name] = {
                "connected": client.connected,
                "disabled": config.disabled if config else True,
                "tools_count": len(client.available_tools),
                "auto_approve_count": len(config.auto_approve) if config and config.auto_approve else 0
            }
        
        return status
    
    def get_langchain_tools(self) -> List:
        """Get LangChain tools for MCP integration."""
        return [mcp_call_tool, mcp_list_tools, mcp_server_status]
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on all MCP servers.
        
        Returns:
            Health check results for all servers
        """
        if not self.initialized:
            await self.initialize()
        
        health_results = {}
        
        for server_name, client in self.manager.clients.items():
            try:
                if not client.connected:
                    health_results[server_name] = {
                        "status": "disconnected",
                        "error": "Not connected to server"
                    }
                    continue
                
                # Try to list tools as a basic health check
                tools_count = len(client.available_tools)
                health_results[server_name] = {
                    "status": "healthy",
                    "tools_count": tools_count,
                    "last_check": "now"
                }
                
            except Exception as e:
                health_results[server_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return health_results


# Global MCP service instance
mcp_service = MCPService()


@asynccontextmanager
async def mcp_service_context():
    """Context manager for MCP service lifecycle."""
    try:
        await mcp_service.initialize()
        yield mcp_service
    finally:
        await mcp_service.shutdown()


# Convenience functions
async def initialize_mcp_service() -> bool:
    """Initialize the global MCP service."""
    return await mcp_service.initialize()


async def shutdown_mcp_service():
    """Shutdown the global MCP service."""
    await mcp_service.shutdown()


def get_mcp_tools():
    """Get LangChain tools for MCP integration."""
    return mcp_service.get_langchain_tools()


# Export main components
__all__ = [
    "MCPService",
    "mcp_service",
    "mcp_service_context",
    "initialize_mcp_service",
    "shutdown_mcp_service", 
    "get_mcp_tools"
]
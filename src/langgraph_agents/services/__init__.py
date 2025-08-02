"""
Services module for LangGraph agents.

This module contains high-level services that provide functionality
to agents, including external integrations and shared resources.
"""

from .mcp_service import (
    MCPService,
    mcp_service,
    mcp_service_context,
    initialize_mcp_service,
    shutdown_mcp_service,
    get_mcp_tools
)

__all__ = [
    "MCPService",
    "mcp_service", 
    "mcp_service_context",
    "initialize_mcp_service",
    "shutdown_mcp_service",
    "get_mcp_tools"
]
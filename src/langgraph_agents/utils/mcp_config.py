"""
MCP configuration management utilities.

This module handles loading and managing MCP server configurations from
both workspace-level and user-level configuration files.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..tools.mcp_tools import MCPServerConfig

logger = logging.getLogger(__name__)


class MCPConfigManager:
    """Manager for MCP server configurations."""
    
    def __init__(self):
        self.workspace_config_path = Path(".kiro/settings/mcp.json")
        self.user_config_path = Path.home() / ".kiro/settings/mcp.json"
        self._cached_configs: Optional[Dict[str, MCPServerConfig]] = None
    
    def load_configs(self, force_reload: bool = False) -> Dict[str, MCPServerConfig]:
        """
        Load MCP server configurations from both workspace and user config files.
        
        Workspace-level configurations take precedence over user-level configurations
        for servers with the same name.
        
        Args:
            force_reload: If True, reload configurations even if cached
            
        Returns:
            Dictionary mapping server names to MCPServerConfig objects
        """
        if self._cached_configs is not None and not force_reload:
            return self._cached_configs
        
        configs = {}
        
        # Load user-level configurations first
        user_configs = self._load_config_file(self.user_config_path)
        if user_configs:
            for name, config_data in user_configs.items():
                try:
                    configs[name] = self._create_server_config(name, config_data)
                    logger.debug(f"Loaded user-level MCP config: {name}")
                except Exception as e:
                    logger.error(f"Failed to load user-level MCP config {name}: {e}")
        
        # Load workspace-level configurations (these override user configs)
        workspace_configs = self._load_config_file(self.workspace_config_path)
        if workspace_configs:
            for name, config_data in workspace_configs.items():
                try:
                    configs[name] = self._create_server_config(name, config_data)
                    logger.debug(f"Loaded workspace-level MCP config: {name}")
                except Exception as e:
                    logger.error(f"Failed to load workspace-level MCP config {name}: {e}")
        
        self._cached_configs = configs
        logger.info(f"Loaded {len(configs)} MCP server configurations")
        return configs
    
    def _load_config_file(self, config_path: Path) -> Optional[Dict[str, Dict[str, Any]]]:
        """Load MCP configuration from a JSON file."""
        if not config_path.exists():
            logger.debug(f"MCP config file not found: {config_path}")
            return None
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract mcpServers section
            mcp_servers = data.get("mcpServers", {})
            if not mcp_servers:
                logger.warning(f"No mcpServers section found in {config_path}")
                return None
            
            return mcp_servers
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in MCP config file {config_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to read MCP config file {config_path}: {e}")
            return None
    
    def _create_server_config(self, name: str, config_data: Dict[str, Any]) -> MCPServerConfig:
        """Create an MCPServerConfig from configuration data."""
        return MCPServerConfig(
            name=name,
            command=config_data.get("command"),
            args=config_data.get("args", []),
            url=config_data.get("url"),
            env=config_data.get("env", {}),
            disabled=config_data.get("disabled", False),
            auto_approve=config_data.get("autoApprove", []),
            timeout=config_data.get("timeout", 30)
        )
    
    def get_config(self, server_name: str) -> Optional[MCPServerConfig]:
        """Get configuration for a specific MCP server."""
        configs = self.load_configs()
        return configs.get(server_name)
    
    def list_server_names(self) -> List[str]:
        """List all configured MCP server names."""
        configs = self.load_configs()
        return list(configs.keys())
    
    def is_server_enabled(self, server_name: str) -> bool:
        """Check if a specific MCP server is enabled."""
        config = self.get_config(server_name)
        return config is not None and not config.disabled
    
    def create_example_config(self, config_path: Path) -> bool:
        """
        Create an example MCP configuration file.
        
        Args:
            config_path: Path where to create the example config
            
        Returns:
            True if the example config was created successfully
        """
        example_config = {
            "mcpServers": {
                "aws-docs": {
                    "command": "uvx",
                    "args": ["awslabs.aws-documentation-mcp-server@latest"],
                    "env": {
                        "FASTMCP_LOG_LEVEL": "ERROR"
                    },
                    "disabled": False,
                    "autoApprove": [],
                    "timeout": 30
                },
                "filesystem": {
                    "command": "uvx",
                    "args": ["mcp-server-filesystem", "/path/to/allowed/directory"],
                    "env": {},
                    "disabled": True,
                    "autoApprove": ["read_file", "list_directory"],
                    "timeout": 15
                },
                "web-search": {
                    "command": "uvx", 
                    "args": ["mcp-server-brave-search"],
                    "env": {
                        "BRAVE_API_KEY": "your-api-key-here"
                    },
                    "disabled": True,
                    "autoApprove": [],
                    "timeout": 45
                }
            }
        }
        
        try:
            # Create directory if it doesn't exist
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(example_config, f, indent=2)
            
            logger.info(f"Created example MCP config at: {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create example MCP config: {e}")
            return False
    
    def validate_config(self, config_data: Dict[str, Any]) -> List[str]:
        """
        Validate MCP configuration data.
        
        Args:
            config_data: Configuration data to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if not isinstance(config_data, dict):
            errors.append("Configuration must be a dictionary")
            return errors
        
        mcp_servers = config_data.get("mcpServers", {})
        if not isinstance(mcp_servers, dict):
            errors.append("mcpServers must be a dictionary")
            return errors
        
        for server_name, server_config in mcp_servers.items():
            if not isinstance(server_config, dict):
                errors.append(f"Server config for '{server_name}' must be a dictionary")
                continue
            
            # Validate required fields
            if not server_config.get("command") and not server_config.get("url"):
                errors.append(f"Server '{server_name}' must have either 'command' or 'url' field")
            
            if server_config.get("command") and server_config.get("url"):
                errors.append(f"Server '{server_name}' cannot have both 'command' and 'url' fields")
            
            # Validate field types
            if "args" in server_config and not isinstance(server_config["args"], list):
                errors.append(f"Server '{server_name}' 'args' must be a list")
            
            if "env" in server_config and not isinstance(server_config["env"], dict):
                errors.append(f"Server '{server_name}' 'env' must be a dictionary")
            
            if "disabled" in server_config and not isinstance(server_config["disabled"], bool):
                errors.append(f"Server '{server_name}' 'disabled' must be a boolean")
            
            if "autoApprove" in server_config and not isinstance(server_config["autoApprove"], list):
                errors.append(f"Server '{server_name}' 'autoApprove' must be a list")
            
            if "timeout" in server_config and not isinstance(server_config["timeout"], (int, float)):
                errors.append(f"Server '{server_name}' 'timeout' must be a number")
        
        return errors
    
    def clear_cache(self):
        """Clear the cached configurations."""
        self._cached_configs = None


# Global configuration manager instance
mcp_config_manager = MCPConfigManager()


def load_mcp_configs() -> Dict[str, MCPServerConfig]:
    """Convenience function to load MCP configurations."""
    return mcp_config_manager.load_configs()


def get_mcp_config(server_name: str) -> Optional[MCPServerConfig]:
    """Convenience function to get a specific MCP server configuration."""
    return mcp_config_manager.get_config(server_name)


def create_example_mcp_config(workspace: bool = True) -> bool:
    """
    Create an example MCP configuration file.
    
    Args:
        workspace: If True, create in workspace .kiro/settings/, 
                  otherwise in user ~/.kiro/settings/
    
    Returns:
        True if the example config was created successfully
    """
    if workspace:
        config_path = Path(".kiro/settings/mcp.json")
    else:
        config_path = Path.home() / ".kiro/settings/mcp.json"
    
    return mcp_config_manager.create_example_config(config_path)


# Export main components
__all__ = [
    "MCPConfigManager",
    "mcp_config_manager", 
    "load_mcp_configs",
    "get_mcp_config",
    "create_example_mcp_config"
]
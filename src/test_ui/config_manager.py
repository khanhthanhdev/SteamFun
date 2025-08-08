"""
Test Configuration Manager

This module provides functionality for saving, loading, and managing test configurations
for the Gradio testing interface. It supports both agent and workflow configurations
with JSON-based persistence.
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TestConfigManager:
    """Manages test configuration persistence and validation."""
    
    def __init__(self, config_dir: str = "test_configs"):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory to store configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Separate directories for different config types
        self.agent_config_dir = self.config_dir / "agents"
        self.workflow_config_dir = self.config_dir / "workflows"
        
        self.agent_config_dir.mkdir(exist_ok=True)
        self.workflow_config_dir.mkdir(exist_ok=True)
        
        logger.info(f"Configuration manager initialized with directory: {self.config_dir}")
    
    def save_agent_config(
        self,
        name: str,
        agent_name: str,
        inputs: Dict[str, Any],
        options: Dict[str, Any] = None,
        description: str = ""
    ) -> str:
        """
        Save an agent test configuration.
        
        Args:
            name: Configuration name
            agent_name: Name of the agent
            inputs: Agent input parameters
            options: Execution options
            description: Configuration description
            
        Returns:
            Configuration ID
        """
        try:
            config_id = str(uuid.uuid4())
            config_data = {
                "id": config_id,
                "name": name,
                "description": description,
                "type": "agent",
                "agent_name": agent_name,
                "inputs": inputs,
                "options": options or {},
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # Validate configuration
            self._validate_agent_config(config_data)
            
            # Save to file
            config_file = self.agent_config_dir / f"{config_id}.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved agent configuration: {name} ({config_id})")
            return config_id
            
        except Exception as e:
            logger.error(f"Failed to save agent configuration: {e}")
            raise Exception(f"Failed to save configuration: {e}")
    
    def save_workflow_config(
        self,
        name: str,
        topic: str,
        description: str,
        config_overrides: Dict[str, Any] = None,
        config_description: str = ""
    ) -> str:
        """
        Save a workflow test configuration.
        
        Args:
            name: Configuration name
            topic: Workflow topic
            description: Workflow description
            config_overrides: Configuration overrides
            config_description: Configuration description
            
        Returns:
            Configuration ID
        """
        try:
            config_id = str(uuid.uuid4())
            config_data = {
                "id": config_id,
                "name": name,
                "description": config_description,
                "type": "workflow",
                "topic": topic,
                "workflow_description": description,
                "config_overrides": config_overrides or {},
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # Validate configuration
            self._validate_workflow_config(config_data)
            
            # Save to file
            config_file = self.workflow_config_dir / f"{config_id}.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved workflow configuration: {name} ({config_id})")
            return config_id
            
        except Exception as e:
            logger.error(f"Failed to save workflow configuration: {e}")
            raise Exception(f"Failed to save configuration: {e}")
    
    def load_config(self, config_id: str) -> Dict[str, Any]:
        """
        Load a configuration by ID.
        
        Args:
            config_id: Configuration ID to load
            
        Returns:
            Configuration data dictionary
        """
        try:
            # Try agent configs first
            agent_config_file = self.agent_config_dir / f"{config_id}.json"
            if agent_config_file.exists():
                with open(agent_config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    self._validate_agent_config(config_data)
                    return config_data
            
            # Try workflow configs
            workflow_config_file = self.workflow_config_dir / f"{config_id}.json"
            if workflow_config_file.exists():
                with open(workflow_config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    self._validate_workflow_config(config_data)
                    return config_data
            
            raise FileNotFoundError(f"Configuration not found: {config_id}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration {config_id}: {e}")
            raise Exception(f"Failed to load configuration: {e}")
    
    def list_configs(self, config_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all available configurations.
        
        Args:
            config_type: Optional filter by type ('agent' or 'workflow')
            
        Returns:
            List of configuration summaries
        """
        try:
            configs = []
            
            # Load agent configs
            if config_type is None or config_type == "agent":
                for config_file in self.agent_config_dir.glob("*.json"):
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config_data = json.load(f)
                            configs.append({
                                "id": config_data["id"],
                                "name": config_data["name"],
                                "description": config_data.get("description", ""),
                                "type": "agent",
                                "agent_name": config_data.get("agent_name", ""),
                                "created_at": config_data.get("created_at", ""),
                                "updated_at": config_data.get("updated_at", "")
                            })
                    except Exception as e:
                        logger.warning(f"Failed to load config {config_file}: {e}")
            
            # Load workflow configs
            if config_type is None or config_type == "workflow":
                for config_file in self.workflow_config_dir.glob("*.json"):
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config_data = json.load(f)
                            configs.append({
                                "id": config_data["id"],
                                "name": config_data["name"],
                                "description": config_data.get("description", ""),
                                "type": "workflow",
                                "topic": config_data.get("topic", ""),
                                "created_at": config_data.get("created_at", ""),
                                "updated_at": config_data.get("updated_at", "")
                            })
                    except Exception as e:
                        logger.warning(f"Failed to load config {config_file}: {e}")
            
            # Sort by creation date (newest first)
            configs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            return configs
            
        except Exception as e:
            logger.error(f"Failed to list configurations: {e}")
            raise Exception(f"Failed to list configurations: {e}")
    
    def delete_config(self, config_id: str) -> bool:
        """
        Delete a configuration by ID.
        
        Args:
            config_id: Configuration ID to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            # Try agent configs first
            agent_config_file = self.agent_config_dir / f"{config_id}.json"
            if agent_config_file.exists():
                agent_config_file.unlink()
                logger.info(f"Deleted agent configuration: {config_id}")
                return True
            
            # Try workflow configs
            workflow_config_file = self.workflow_config_dir / f"{config_id}.json"
            if workflow_config_file.exists():
                workflow_config_file.unlink()
                logger.info(f"Deleted workflow configuration: {config_id}")
                return True
            
            raise FileNotFoundError(f"Configuration not found: {config_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete configuration {config_id}: {e}")
            raise Exception(f"Failed to delete configuration: {e}")
    
    def update_config(
        self,
        config_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update an existing configuration.
        
        Args:
            config_id: Configuration ID to update
            updates: Dictionary of updates to apply
            
        Returns:
            Updated configuration data
        """
        try:
            # Load existing configuration
            config_data = self.load_config(config_id)
            
            # Apply updates
            config_data.update(updates)
            config_data["updated_at"] = datetime.now().isoformat()
            
            # Validate updated configuration
            if config_data["type"] == "agent":
                self._validate_agent_config(config_data)
                config_file = self.agent_config_dir / f"{config_id}.json"
            else:
                self._validate_workflow_config(config_data)
                config_file = self.workflow_config_dir / f"{config_id}.json"
            
            # Save updated configuration
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Updated configuration: {config_id}")
            return config_data
            
        except Exception as e:
            logger.error(f"Failed to update configuration {config_id}: {e}")
            raise Exception(f"Failed to update configuration: {e}")
    
    def _validate_agent_config(self, config_data: Dict[str, Any]) -> None:
        """
        Validate agent configuration data.
        
        Args:
            config_data: Configuration data to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ["id", "name", "type", "agent_name", "inputs"]
        for field in required_fields:
            if field not in config_data:
                raise ValueError(f"Missing required field: {field}")
        
        if config_data["type"] != "agent":
            raise ValueError("Invalid configuration type for agent config")
        
        if not isinstance(config_data["inputs"], dict):
            raise ValueError("Agent inputs must be a dictionary")
        
        if not isinstance(config_data.get("options", {}), dict):
            raise ValueError("Agent options must be a dictionary")
    
    def _validate_workflow_config(self, config_data: Dict[str, Any]) -> None:
        """
        Validate workflow configuration data.
        
        Args:
            config_data: Configuration data to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ["id", "name", "type", "topic", "workflow_description"]
        for field in required_fields:
            if field not in config_data:
                raise ValueError(f"Missing required field: {field}")
        
        if config_data["type"] != "workflow":
            raise ValueError("Invalid configuration type for workflow config")
        
        if not isinstance(config_data.get("config_overrides", {}), dict):
            raise ValueError("Config overrides must be a dictionary")
    
    def export_config(self, config_id: str, export_path: str) -> None:
        """
        Export a configuration to a file.
        
        Args:
            config_id: Configuration ID to export
            export_path: Path to export the configuration to
        """
        try:
            config_data = self.load_config(config_id)
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported configuration {config_id} to {export_path}")
            
        except Exception as e:
            logger.error(f"Failed to export configuration {config_id}: {e}")
            raise Exception(f"Failed to export configuration: {e}")
    
    def import_config(self, import_path: str) -> str:
        """
        Import a configuration from a file.
        
        Args:
            import_path: Path to import the configuration from
            
        Returns:
            New configuration ID
        """
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Generate new ID and update timestamps
            old_id = config_data.get("id")
            config_data["id"] = str(uuid.uuid4())
            config_data["created_at"] = datetime.now().isoformat()
            config_data["updated_at"] = datetime.now().isoformat()
            
            # Validate and save
            if config_data["type"] == "agent":
                self._validate_agent_config(config_data)
                config_file = self.agent_config_dir / f"{config_data['id']}.json"
            else:
                self._validate_workflow_config(config_data)
                config_file = self.workflow_config_dir / f"{config_data['id']}.json"
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Imported configuration from {import_path} with new ID: {config_data['id']}")
            return config_data["id"]
            
        except Exception as e:
            logger.error(f"Failed to import configuration from {import_path}: {e}")
            raise Exception(f"Failed to import configuration: {e}")
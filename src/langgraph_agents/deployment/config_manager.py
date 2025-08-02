"""
Environment-specific configuration management for deployment.

This module provides configuration management for different deployment environments,
configuration validation, and environment-specific overrides.
"""

import logging
import json
import yaml
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
from copy import deepcopy

logger = logging.getLogger(__name__)


class DeploymentEnvironment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class EnvironmentConfig:
    """Configuration for a specific environment."""
    environment: DeploymentEnvironment
    config_data: Dict[str, Any]
    config_version: str
    last_updated: datetime
    validation_errors: List[str]
    is_valid: bool


class DeploymentConfigManager:
    """Manager for environment-specific configurations."""
    
    def __init__(self, config_dir: str = "config/environments"):
        """Initialize the deployment config manager.
        
        Args:
            config_dir: Directory containing environment configurations
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Environment configurations
        self.environments: Dict[DeploymentEnvironment, EnvironmentConfig] = {}
        self.current_environment: Optional[DeploymentEnvironment] = None
        
        # Configuration schema for validation
        self.config_schema = self._load_config_schema()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Configuration watchers
        self.config_watchers: List[callable] = []
        
        logger.info(f"Deployment config manager initialized with config dir: {config_dir}")
    
    def _load_config_schema(self) -> Dict[str, Any]:
        """Load configuration schema for validation."""
        
        schema_path = self.config_dir / "schema.json"
        if schema_path.exists():
            try:
                with open(schema_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading config schema: {str(e)}")
        
        # Default schema
        return {
            "type": "object",
            "properties": {
                "agents": {
                    "type": "object",
                    "properties": {
                        "concurrency": {"type": "object"},
                        "error_recovery": {"type": "object"},
                        "performance": {"type": "object"}
                    }
                },
                "llm_providers": {
                    "type": "object",
                    "properties": {
                        "openai": {"type": "object"},
                        "bedrock": {"type": "object"}
                    }
                },
                "monitoring": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "metrics_interval": {"type": "number"},
                        "health_checks": {"type": "object"}
                    }
                },
                "deployment": {
                    "type": "object",
                    "properties": {
                        "replicas": {"type": "number"},
                        "resources": {"type": "object"},
                        "scaling": {"type": "object"}
                    }
                }
            },
            "required": ["agents", "llm_providers", "monitoring"]
        }
    
    def load_environment_configs(self):
        """Load all environment configurations from files."""
        
        with self._lock:
            for env in DeploymentEnvironment:
                config_file = self.config_dir / f"{env.value}.yaml"
                if config_file.exists():
                    try:
                        config_data = self._load_config_file(config_file)
                        validation_errors = self._validate_config(config_data)
                        
                        env_config = EnvironmentConfig(
                            environment=env,
                            config_data=config_data,
                            config_version=self._get_config_version(config_data),
                            last_updated=datetime.fromtimestamp(config_file.stat().st_mtime),
                            validation_errors=validation_errors,
                            is_valid=len(validation_errors) == 0
                        )
                        
                        self.environments[env] = env_config
                        
                        if validation_errors:
                            logger.warning(f"Configuration validation errors for {env.value}: {validation_errors}")
                        else:
                            logger.info(f"Loaded valid configuration for {env.value}")
                    
                    except Exception as e:
                        logger.error(f"Error loading config for {env.value}: {str(e)}")
                else:
                    logger.warning(f"Configuration file not found for {env.value}: {config_file}")
    
    def _load_config_file(self, config_file: Path) -> Dict[str, Any]:
        """Load configuration from a file."""
        
        with open(config_file, 'r') as f:
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif config_file.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_file.suffix}")
    
    def _validate_config(self, config_data: Dict[str, Any]) -> List[str]:
        """Validate configuration against schema."""
        
        errors = []
        
        try:
            # Basic validation against schema
            required_sections = self.config_schema.get("required", [])
            for section in required_sections:
                if section not in config_data:
                    errors.append(f"Missing required section: {section}")
            
            # Validate agent configurations
            if "agents" in config_data:
                agent_errors = self._validate_agent_config(config_data["agents"])
                errors.extend(agent_errors)
            
            # Validate LLM provider configurations
            if "llm_providers" in config_data:
                llm_errors = self._validate_llm_config(config_data["llm_providers"])
                errors.extend(llm_errors)
            
            # Validate monitoring configuration
            if "monitoring" in config_data:
                monitoring_errors = self._validate_monitoring_config(config_data["monitoring"])
                errors.extend(monitoring_errors)
            
            # Validate deployment configuration
            if "deployment" in config_data:
                deployment_errors = self._validate_deployment_config(config_data["deployment"])
                errors.extend(deployment_errors)
        
        except Exception as e:
            errors.append(f"Configuration validation error: {str(e)}")
        
        return errors
    
    def _validate_agent_config(self, agent_config: Dict[str, Any]) -> List[str]:
        """Validate agent configuration."""
        
        errors = []
        
        # Validate concurrency settings
        if "concurrency" in agent_config:
            concurrency = agent_config["concurrency"]
            if "max_concurrent_tasks" in concurrency:
                if not isinstance(concurrency["max_concurrent_tasks"], int) or concurrency["max_concurrent_tasks"] <= 0:
                    errors.append("max_concurrent_tasks must be a positive integer")
            
            if "max_memory_usage_mb" in concurrency:
                if not isinstance(concurrency["max_memory_usage_mb"], (int, float)) or concurrency["max_memory_usage_mb"] <= 0:
                    errors.append("max_memory_usage_mb must be a positive number")
        
        # Validate error recovery settings
        if "error_recovery" in agent_config:
            error_recovery = agent_config["error_recovery"]
            if "max_retry_attempts" in error_recovery:
                if not isinstance(error_recovery["max_retry_attempts"], int) or error_recovery["max_retry_attempts"] < 0:
                    errors.append("max_retry_attempts must be a non-negative integer")
        
        return errors
    
    def _validate_llm_config(self, llm_config: Dict[str, Any]) -> List[str]:
        """Validate LLM provider configuration."""
        
        errors = []
        
        # Validate OpenAI configuration
        if "openai" in llm_config:
            openai_config = llm_config["openai"]
            if "api_key" not in openai_config and "OPENAI_API_KEY" not in os.environ:
                errors.append("OpenAI API key not configured")
            
            if "model" in openai_config:
                valid_models = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
                if openai_config["model"] not in valid_models:
                    errors.append(f"Invalid OpenAI model: {openai_config['model']}")
        
        # Validate Bedrock configuration
        if "bedrock" in llm_config:
            bedrock_config = llm_config["bedrock"]
            if "region" not in bedrock_config:
                errors.append("Bedrock region not configured")
            
            if "model_id" in bedrock_config:
                # Basic validation for model ID format
                if not bedrock_config["model_id"].startswith(("anthropic.", "amazon.", "ai21.")):
                    errors.append(f"Invalid Bedrock model ID format: {bedrock_config['model_id']}")
        
        return errors
    
    def _validate_monitoring_config(self, monitoring_config: Dict[str, Any]) -> List[str]:
        """Validate monitoring configuration."""
        
        errors = []
        
        if "metrics_interval" in monitoring_config:
            interval = monitoring_config["metrics_interval"]
            if not isinstance(interval, (int, float)) or interval <= 0:
                errors.append("metrics_interval must be a positive number")
        
        if "health_checks" in monitoring_config:
            health_checks = monitoring_config["health_checks"]
            if "enabled" in health_checks and not isinstance(health_checks["enabled"], bool):
                errors.append("health_checks.enabled must be a boolean")
            
            if "interval_seconds" in health_checks:
                interval = health_checks["interval_seconds"]
                if not isinstance(interval, (int, float)) or interval <= 0:
                    errors.append("health_checks.interval_seconds must be a positive number")
        
        return errors
    
    def _validate_deployment_config(self, deployment_config: Dict[str, Any]) -> List[str]:
        """Validate deployment configuration."""
        
        errors = []
        
        if "replicas" in deployment_config:
            replicas = deployment_config["replicas"]
            if not isinstance(replicas, int) or replicas <= 0:
                errors.append("replicas must be a positive integer")
        
        if "resources" in deployment_config:
            resources = deployment_config["resources"]
            
            if "cpu_limit" in resources:
                cpu_limit = resources["cpu_limit"]
                if not isinstance(cpu_limit, (int, float)) or cpu_limit <= 0:
                    errors.append("cpu_limit must be a positive number")
            
            if "memory_limit_mb" in resources:
                memory_limit = resources["memory_limit_mb"]
                if not isinstance(memory_limit, (int, float)) or memory_limit <= 0:
                    errors.append("memory_limit_mb must be a positive number")
        
        return errors
    
    def _get_config_version(self, config_data: Dict[str, Any]) -> str:
        """Get configuration version."""
        
        return config_data.get("version", "1.0.0")
    
    def set_current_environment(self, environment: DeploymentEnvironment):
        """Set the current deployment environment.
        
        Args:
            environment: Environment to set as current
        """
        
        with self._lock:
            if environment not in self.environments:
                raise ValueError(f"Environment not loaded: {environment.value}")
            
            env_config = self.environments[environment]
            if not env_config.is_valid:
                raise ValueError(f"Environment configuration is invalid: {environment.value}")
            
            self.current_environment = environment
            logger.info(f"Current environment set to: {environment.value}")
            
            # Notify watchers
            self._notify_config_watchers(environment, env_config)
    
    def get_current_config(self) -> Optional[EnvironmentConfig]:
        """Get the current environment configuration.
        
        Returns:
            Current environment configuration or None
        """
        
        with self._lock:
            if self.current_environment:
                return self.environments.get(self.current_environment)
            return None
    
    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the configuration value
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        
        current_config = self.get_current_config()
        if not current_config:
            return default
        
        keys = key_path.split('.')
        value = current_config.config_data
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def update_config_value(self, key_path: str, value: Any, environment: Optional[DeploymentEnvironment] = None):
        """Update a configuration value.
        
        Args:
            key_path: Dot-separated path to the configuration value
            value: New value to set
            environment: Environment to update (defaults to current)
        """
        
        with self._lock:
            target_env = environment or self.current_environment
            if not target_env or target_env not in self.environments:
                raise ValueError(f"Invalid environment: {target_env}")
            
            env_config = self.environments[target_env]
            keys = key_path.split('.')
            
            # Navigate to the parent of the target key
            config_data = env_config.config_data
            for key in keys[:-1]:
                if key not in config_data:
                    config_data[key] = {}
                config_data = config_data[key]
            
            # Set the value
            config_data[keys[-1]] = value
            
            # Re-validate configuration
            validation_errors = self._validate_config(env_config.config_data)
            env_config.validation_errors = validation_errors
            env_config.is_valid = len(validation_errors) == 0
            env_config.last_updated = datetime.now()
            
            logger.info(f"Updated config value {key_path} in {target_env.value}")
            
            # Notify watchers if this is the current environment
            if target_env == self.current_environment:
                self._notify_config_watchers(target_env, env_config)
    
    def save_environment_config(self, environment: DeploymentEnvironment):
        """Save environment configuration to file.
        
        Args:
            environment: Environment to save
        """
        
        with self._lock:
            if environment not in self.environments:
                raise ValueError(f"Environment not loaded: {environment.value}")
            
            env_config = self.environments[environment]
            config_file = self.config_dir / f"{environment.value}.yaml"
            
            try:
                with open(config_file, 'w') as f:
                    yaml.dump(env_config.config_data, f, default_flow_style=False, indent=2)
                
                logger.info(f"Saved configuration for {environment.value}")
                
            except Exception as e:
                logger.error(f"Error saving config for {environment.value}: {str(e)}")
                raise
    
    def create_environment_config(self, environment: DeploymentEnvironment, base_config: Dict[str, Any]):
        """Create a new environment configuration.
        
        Args:
            environment: Environment to create
            base_config: Base configuration data
        """
        
        with self._lock:
            # Validate the configuration
            validation_errors = self._validate_config(base_config)
            
            env_config = EnvironmentConfig(
                environment=environment,
                config_data=deepcopy(base_config),
                config_version=self._get_config_version(base_config),
                last_updated=datetime.now(),
                validation_errors=validation_errors,
                is_valid=len(validation_errors) == 0
            )
            
            self.environments[environment] = env_config
            
            # Save to file
            self.save_environment_config(environment)
            
            logger.info(f"Created configuration for {environment.value}")
    
    def merge_configs(self, base_env: DeploymentEnvironment, override_env: DeploymentEnvironment) -> Dict[str, Any]:
        """Merge configurations from two environments.
        
        Args:
            base_env: Base environment configuration
            override_env: Override environment configuration
            
        Returns:
            Merged configuration
        """
        
        with self._lock:
            if base_env not in self.environments or override_env not in self.environments:
                raise ValueError("One or both environments not loaded")
            
            base_config = deepcopy(self.environments[base_env].config_data)
            override_config = self.environments[override_env].config_data
            
            return self._deep_merge(base_config, override_config)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    def add_config_watcher(self, callback: callable):
        """Add a callback to be notified of configuration changes.
        
        Args:
            callback: Function to call on config changes
        """
        self.config_watchers.append(callback)
    
    def _notify_config_watchers(self, environment: DeploymentEnvironment, config: EnvironmentConfig):
        """Notify all config watchers of changes."""
        
        for callback in self.config_watchers:
            try:
                callback(environment, config)
            except Exception as e:
                logger.error(f"Error in config watcher callback: {str(e)}")
    
    def get_environment_status(self) -> Dict[str, Any]:
        """Get status of all environments."""
        
        with self._lock:
            status = {
                "current_environment": self.current_environment.value if self.current_environment else None,
                "environments": {}
            }
            
            for env, config in self.environments.items():
                status["environments"][env.value] = {
                    "is_valid": config.is_valid,
                    "validation_errors": config.validation_errors,
                    "config_version": config.config_version,
                    "last_updated": config.last_updated.isoformat(),
                    "config_size": len(str(config.config_data))
                }
            
            return status
    
    def export_config(self, environment: DeploymentEnvironment, format: str = "yaml") -> str:
        """Export environment configuration as string.
        
        Args:
            environment: Environment to export
            format: Export format ('yaml' or 'json')
            
        Returns:
            Configuration as string
        """
        
        with self._lock:
            if environment not in self.environments:
                raise ValueError(f"Environment not loaded: {environment.value}")
            
            config_data = self.environments[environment].config_data
            
            if format.lower() == "yaml":
                return yaml.dump(config_data, default_flow_style=False, indent=2)
            elif format.lower() == "json":
                return json.dumps(config_data, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
    
    def import_config(self, environment: DeploymentEnvironment, config_string: str, format: str = "yaml"):
        """Import environment configuration from string.
        
        Args:
            environment: Environment to import to
            config_string: Configuration as string
            format: Import format ('yaml' or 'json')
        """
        
        try:
            if format.lower() == "yaml":
                config_data = yaml.safe_load(config_string)
            elif format.lower() == "json":
                config_data = json.loads(config_string)
            else:
                raise ValueError(f"Unsupported import format: {format}")
            
            # Validate the imported configuration
            validation_errors = self._validate_config(config_data)
            
            with self._lock:
                env_config = EnvironmentConfig(
                    environment=environment,
                    config_data=config_data,
                    config_version=self._get_config_version(config_data),
                    last_updated=datetime.now(),
                    validation_errors=validation_errors,
                    is_valid=len(validation_errors) == 0
                )
                
                self.environments[environment] = env_config
                
                # Save to file
                self.save_environment_config(environment)
                
                logger.info(f"Imported configuration for {environment.value}")
                
                if validation_errors:
                    logger.warning(f"Imported config has validation errors: {validation_errors}")
        
        except Exception as e:
            logger.error(f"Error importing config for {environment.value}: {str(e)}")
            raise
    
    def validate_all_environments(self) -> Dict[str, List[str]]:
        """Validate all loaded environment configurations.
        
        Returns:
            Dictionary mapping environment names to validation errors
        """
        
        with self._lock:
            validation_results = {}
            
            for env, config in self.environments.items():
                # Re-validate configuration
                validation_errors = self._validate_config(config.config_data)
                config.validation_errors = validation_errors
                config.is_valid = len(validation_errors) == 0
                
                validation_results[env.value] = validation_errors
            
            return validation_results
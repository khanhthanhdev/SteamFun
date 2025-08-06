"""
Configuration validation utilities for the LangGraph video generation workflow.

This module provides comprehensive validation for workflow configurations
using JSON Schema and custom validation logic.
"""

import json
import yaml
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from jsonschema import validate, ValidationError as JsonSchemaValidationError
from pydantic import ValidationError
import logging

from ..models.config import WorkflowConfig

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""
    
    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or []


class ConfigValidator:
    """Validates workflow configurations against schema and business rules."""
    
    def __init__(self, schema_path: Optional[str] = None):
        """Initialize the validator with optional custom schema path."""
        self.schema_path = schema_path or self._get_default_schema_path()
        self.schema = self._load_schema()
    
    def _get_default_schema_path(self) -> str:
        """Get the default schema file path."""
        current_dir = Path(__file__).parent
        schema_path = current_dir.parent.parent.parent / "config" / "schema" / "workflow_config_schema.json"
        return str(schema_path)
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load the JSON schema for validation."""
        try:
            with open(self.schema_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Schema file not found at {self.schema_path}, using minimal validation")
            return self._get_minimal_schema()
        except json.JSONDecodeError as e:
            raise ConfigValidationError(f"Invalid JSON schema file: {e}")
    
    def _get_minimal_schema(self) -> Dict[str, Any]:
        """Return a minimal schema for basic validation."""
        return {
            "type": "object",
            "properties": {
                "workflow": {"type": "object"},
                "models": {"type": "object"}
            },
            "required": ["workflow", "models"]
        }
    
    def validate_config_dict(self, config_dict: Dict[str, Any]) -> List[str]:
        """
        Validate a configuration dictionary against the schema.
        
        Args:
            config_dict: Configuration dictionary to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # JSON Schema validation
        try:
            validate(instance=config_dict, schema=self.schema)
        except JsonSchemaValidationError as e:
            errors.append(f"Schema validation error: {e.message}")
            if e.path:
                errors.append(f"Error path: {' -> '.join(str(p) for p in e.path)}")
        
        # Custom business logic validation
        errors.extend(self._validate_business_rules(config_dict))
        
        return errors
    
    def _validate_business_rules(self, config_dict: Dict[str, Any]) -> List[str]:
        """Validate business-specific rules not covered by JSON schema."""
        errors = []
        
        # Validate model configurations
        if "models" in config_dict:
            errors.extend(self._validate_models(config_dict["models"]))
        
        # Validate performance settings
        if "performance" in config_dict:
            errors.extend(self._validate_performance(config_dict["performance"]))
        
        # Validate paths
        if "paths" in config_dict:
            errors.extend(self._validate_paths(config_dict["paths"]))
        
        # Validate feature compatibility
        if "features" in config_dict:
            errors.extend(self._validate_feature_compatibility(config_dict["features"]))
        
        return errors
    
    def _validate_models(self, models_config: Dict[str, Any]) -> List[str]:
        """Validate model configurations."""
        errors = []
        
        required_models = ["planner_model", "code_model"]
        for model_type in required_models:
            if model_type not in models_config:
                errors.append(f"Required model '{model_type}' is missing")
                continue
            
            model_config = models_config[model_type]
            if not isinstance(model_config, dict):
                errors.append(f"Model '{model_type}' must be an object")
                continue
            
            # Validate provider and model name combination
            provider = model_config.get("provider", "")
            model_name = model_config.get("model_name", "")
            
            if provider and model_name:
                if not self._is_valid_model_combination(provider, model_name):
                    errors.append(f"Invalid model combination: {provider}/{model_name}")
        
        return errors
    
    def _is_valid_model_combination(self, provider: str, model_name: str) -> bool:
        """Check if provider and model name combination is valid."""
        # Define known valid combinations
        valid_combinations = {
            "openai": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
            "anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
            "openrouter": ["openai/gpt-4", "anthropic/claude-3.5-sonnet"],
        }
        
        if provider in valid_combinations:
            # For openrouter, check if model_name starts with known prefixes
            if provider == "openrouter":
                return any(model_name.startswith(prefix) for prefix in ["openai/", "anthropic/", "meta-llama/"])
            else:
                return any(model in model_name for model in valid_combinations[provider])
        
        # Allow unknown providers (they might be custom or new)
        return True
    
    def _validate_performance(self, performance_config: Dict[str, Any]) -> List[str]:
        """Validate performance configuration."""
        errors = []
        
        max_concurrent_scenes = performance_config.get("max_concurrent_scenes", 5)
        max_concurrent_renders = performance_config.get("max_concurrent_renders", 4)
        
        if max_concurrent_renders > max_concurrent_scenes:
            errors.append("max_concurrent_renders cannot exceed max_concurrent_scenes")
        
        # Validate timeout settings
        timeout_seconds = performance_config.get("timeout_seconds", 300)
        if timeout_seconds < 30:
            errors.append("timeout_seconds should be at least 30 seconds for reliable operation")
        
        return errors
    
    def _validate_paths(self, paths_config: Dict[str, Any]) -> List[str]:
        """Validate path configurations."""
        errors = []
        
        for path_name, path_value in paths_config.items():
            if not isinstance(path_value, str):
                errors.append(f"Path '{path_name}' must be a string")
                continue
            
            # Check for path traversal attempts
            if "../" in path_value or "..\\" in path_value:
                errors.append(f"Path '{path_name}' contains invalid traversal sequences")
            
            # Check for potentially dangerous absolute paths
            if path_value.startswith("/") and not path_value.startswith(("/tmp/", "/var/tmp/")):
                errors.append(f"Absolute path '{path_name}' outside safe directories")
        
        return errors
    
    def _validate_feature_compatibility(self, features_config: Dict[str, Any]) -> List[str]:
        """Validate feature flag compatibility."""
        errors = []
        
        # Check if GPU acceleration is enabled but preview mode is also enabled
        if features_config.get("use_gpu_acceleration") and features_config.get("preview_mode"):
            errors.append("GPU acceleration and preview mode are mutually exclusive")
        
        # Check if visual analysis is enabled but RAG is disabled
        if features_config.get("use_visual_analysis") and not features_config.get("use_rag"):
            errors.append("Visual analysis requires RAG to be enabled")
        
        return errors
    
    def validate_config_file(self, config_path: str) -> List[str]:
        """
        Validate a configuration file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            List of validation error messages (empty if valid)
        """
        try:
            config_dict = self.load_config_file(config_path)
            return self.validate_config_dict(config_dict)
        except Exception as e:
            return [f"Failed to load configuration file: {e}"]
    
    def load_config_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file (supports JSON and YAML)."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.endswith(('.yaml', '.yml')):
                return yaml.safe_load(f)
            elif config_path.endswith('.json'):
                return json.load(f)
            else:
                # Try to detect format by content
                content = f.read()
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    try:
                        return yaml.safe_load(content)
                    except yaml.YAMLError:
                        raise ValueError("Unable to parse configuration file as JSON or YAML")
    
    def validate_pydantic_config(self, config: WorkflowConfig) -> List[str]:
        """
        Validate a Pydantic WorkflowConfig instance.
        
        Args:
            config: WorkflowConfig instance to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        try:
            # Convert to dict and validate against schema
            config_dict = config.model_dump()
            schema_errors = self.validate_config_dict(config_dict)
            errors.extend(schema_errors)
            
            # Additional Pydantic-specific validations
            errors.extend(self._validate_pydantic_specific(config))
            
        except Exception as e:
            errors.append(f"Failed to validate Pydantic config: {e}")
        
        return errors
    
    def _validate_pydantic_specific(self, config: WorkflowConfig) -> List[str]:
        """Validate Pydantic-specific configuration aspects."""
        errors = []
        
        # Validate model configurations are properly set
        try:
            planner_config = config.get_model_config("planner")
            if not planner_config.provider or not planner_config.model_name:
                errors.append("Planner model configuration is incomplete")
        except Exception as e:
            errors.append(f"Invalid planner model configuration: {e}")
        
        try:
            code_config = config.get_model_config("code")
            if not code_config.provider or not code_config.model_name:
                errors.append("Code model configuration is incomplete")
        except Exception as e:
            errors.append(f"Invalid code model configuration: {e}")
        
        # Validate feature flags
        if config.use_visual_analysis and not config.use_rag:
            errors.append("Visual analysis requires RAG to be enabled")
        
        return errors


def validate_config_from_file(config_path: str, schema_path: Optional[str] = None) -> None:
    """
    Validate a configuration file and raise exception if invalid.
    
    Args:
        config_path: Path to configuration file
        schema_path: Optional path to custom schema file
        
    Raises:
        ConfigValidationError: If configuration is invalid
    """
    validator = ConfigValidator(schema_path)
    errors = validator.validate_config_file(config_path)
    
    if errors:
        raise ConfigValidationError(
            f"Configuration validation failed for {config_path}",
            errors
        )


def validate_config_dict(config_dict: Dict[str, Any], schema_path: Optional[str] = None) -> None:
    """
    Validate a configuration dictionary and raise exception if invalid.
    
    Args:
        config_dict: Configuration dictionary to validate
        schema_path: Optional path to custom schema file
        
    Raises:
        ConfigValidationError: If configuration is invalid
    """
    validator = ConfigValidator(schema_path)
    errors = validator.validate_config_dict(config_dict)
    
    if errors:
        raise ConfigValidationError(
            "Configuration validation failed",
            errors
        )


def create_config_from_dict(config_dict: Dict[str, Any], validate: bool = True) -> WorkflowConfig:
    """
    Create a WorkflowConfig from a dictionary with optional validation.
    
    Args:
        config_dict: Configuration dictionary
        validate: Whether to validate against schema
        
    Returns:
        WorkflowConfig instance
        
    Raises:
        ConfigValidationError: If validation fails
        ValidationError: If Pydantic validation fails
    """
    if validate:
        validate_config_dict(config_dict)
    
    try:
        return WorkflowConfig(**config_dict)
    except ValidationError as e:
        raise ConfigValidationError(f"Pydantic validation failed: {e}")


def create_config_from_file(config_path: str, validate: bool = True) -> WorkflowConfig:
    """
    Create a WorkflowConfig from a file with optional validation.
    
    Args:
        config_path: Path to configuration file
        validate: Whether to validate against schema
        
    Returns:
        WorkflowConfig instance
        
    Raises:
        ConfigValidationError: If validation fails
    """
    validator = ConfigValidator()
    config_dict = validator.load_config_file(config_path)
    
    return create_config_from_dict(config_dict, validate)
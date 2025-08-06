"""
Tests for configuration validation functionality.

This module tests the configuration validation system including
schema validation, business rule validation, and error handling.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Any

from src.langgraph_agents.config.validation import (
    ConfigValidator,
    ConfigValidationError,
    validate_config_from_file,
    validate_config_dict,
    create_config_from_dict,
    create_config_from_file
)
from src.langgraph_agents.models.config import WorkflowConfig


class TestConfigValidator:
    """Test the ConfigValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create a ConfigValidator instance for testing."""
        return ConfigValidator()
    
    @pytest.fixture
    def valid_config_dict(self):
        """Return a valid configuration dictionary."""
        return {
            "workflow": {
                "name": "test-workflow",
                "version": "1.0.0",
                "environment": "development"
            },
            "models": {
                "planner_model": {
                    "provider": "openrouter",
                    "model_name": "anthropic/claude-3.5-sonnet",
                    "temperature": 0.7,
                    "max_tokens": 4000,
                    "timeout": 60
                },
                "code_model": {
                    "provider": "openrouter",
                    "model_name": "anthropic/claude-3.5-sonnet",
                    "temperature": 0.3,
                    "max_tokens": 8000,
                    "timeout": 120
                }
            },
            "features": {
                "use_rag": True,
                "use_visual_analysis": False,
                "enable_caching": True
            },
            "performance": {
                "max_retries": 3,
                "timeout_seconds": 300,
                "max_concurrent_scenes": 5,
                "max_concurrent_renders": 4
            }
        }
    
    def test_validate_valid_config(self, validator, valid_config_dict):
        """Test validation of a valid configuration."""
        errors = validator.validate_config_dict(valid_config_dict)
        assert errors == []
    
    def test_validate_missing_required_fields(self, validator):
        """Test validation with missing required fields."""
        config_dict = {
            "workflow": {
                "name": "test-workflow",
                "version": "1.0.0"
                # Missing environment
            }
            # Missing models section
        }
        
        errors = validator.validate_config_dict(config_dict)
        assert len(errors) > 0
        assert any("required" in error.lower() for error in errors)
    
    def test_validate_invalid_model_provider(self, validator, valid_config_dict):
        """Test validation with invalid model provider."""
        valid_config_dict["models"]["planner_model"]["provider"] = "invalid_provider"
        
        errors = validator.validate_config_dict(valid_config_dict)
        assert len(errors) > 0
        assert any("provider" in error.lower() for error in errors)
    
    def test_validate_invalid_temperature(self, validator, valid_config_dict):
        """Test validation with invalid temperature value."""
        valid_config_dict["models"]["planner_model"]["temperature"] = 3.0  # Too high
        
        errors = validator.validate_config_dict(valid_config_dict)
        assert len(errors) > 0
    
    def test_validate_performance_constraints(self, validator, valid_config_dict):
        """Test validation of performance constraints."""
        # Set max_concurrent_renders higher than max_concurrent_scenes
        valid_config_dict["performance"]["max_concurrent_renders"] = 10
        valid_config_dict["performance"]["max_concurrent_scenes"] = 5
        
        errors = validator.validate_config_dict(valid_config_dict)
        assert len(errors) > 0
        assert any("concurrent" in error.lower() for error in errors)
    
    def test_validate_path_traversal(self, validator, valid_config_dict):
        """Test validation prevents path traversal attacks."""
        valid_config_dict["paths"] = {
            "output_dir": "../../../etc/passwd"
        }
        
        errors = validator.validate_config_dict(valid_config_dict)
        assert len(errors) > 0
        assert any("traversal" in error.lower() for error in errors)
    
    def test_validate_feature_compatibility(self, validator, valid_config_dict):
        """Test validation of feature compatibility."""
        # Enable visual analysis but disable RAG
        valid_config_dict["features"]["use_visual_analysis"] = True
        valid_config_dict["features"]["use_rag"] = False
        
        errors = validator.validate_config_dict(valid_config_dict)
        assert len(errors) > 0
        assert any("visual analysis" in error.lower() and "rag" in error.lower() for error in errors)
    
    def test_validate_timeout_too_low(self, validator, valid_config_dict):
        """Test validation of unreasonably low timeout."""
        valid_config_dict["performance"]["timeout_seconds"] = 10  # Too low
        
        errors = validator.validate_config_dict(valid_config_dict)
        assert len(errors) > 0
        assert any("timeout" in error.lower() for error in errors)
    
    def test_load_json_config_file(self, validator, valid_config_dict):
        """Test loading JSON configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(valid_config_dict, f)
            temp_path = f.name
        
        try:
            loaded_config = validator.load_config_file(temp_path)
            assert loaded_config == valid_config_dict
        finally:
            os.unlink(temp_path)
    
    def test_load_yaml_config_file(self, validator, valid_config_dict):
        """Test loading YAML configuration file."""
        import yaml
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(valid_config_dict, f)
            temp_path = f.name
        
        try:
            loaded_config = validator.load_config_file(temp_path)
            assert loaded_config == valid_config_dict
        finally:
            os.unlink(temp_path)
    
    def test_load_nonexistent_file(self, validator):
        """Test loading nonexistent configuration file."""
        with pytest.raises(FileNotFoundError):
            validator.load_config_file("nonexistent_file.json")
    
    def test_validate_config_file(self, validator, valid_config_dict):
        """Test validating configuration from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(valid_config_dict, f)
            temp_path = f.name
        
        try:
            errors = validator.validate_config_file(temp_path)
            assert errors == []
        finally:
            os.unlink(temp_path)
    
    def test_validate_pydantic_config(self, validator):
        """Test validating Pydantic WorkflowConfig instance."""
        config = WorkflowConfig()
        errors = validator.validate_pydantic_config(config)
        # Should have minimal errors for default config
        assert len(errors) == 0 or all("incomplete" not in error.lower() for error in errors)


class TestConfigValidationFunctions:
    """Test standalone configuration validation functions."""
    
    @pytest.fixture
    def valid_config_dict(self):
        """Return a valid configuration dictionary."""
        return {
            "workflow": {
                "name": "test-workflow",
                "version": "1.0.0",
                "environment": "development"
            },
            "models": {
                "planner_model": {
                    "provider": "openrouter",
                    "model_name": "anthropic/claude-3.5-sonnet"
                },
                "code_model": {
                    "provider": "openrouter",
                    "model_name": "anthropic/claude-3.5-sonnet"
                }
            }
        }
    
    def test_validate_config_dict_success(self, valid_config_dict):
        """Test successful validation of config dictionary."""
        # Should not raise exception
        validate_config_dict(valid_config_dict)
    
    def test_validate_config_dict_failure(self):
        """Test failed validation of config dictionary."""
        invalid_config = {"invalid": "config"}
        
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dict(invalid_config)
        
        assert "validation failed" in str(exc_info.value).lower()
        assert len(exc_info.value.errors) > 0
    
    def test_validate_config_from_file_success(self, valid_config_dict):
        """Test successful validation from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(valid_config_dict, f)
            temp_path = f.name
        
        try:
            # Should not raise exception
            validate_config_from_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_validate_config_from_file_failure(self):
        """Test failed validation from file."""
        invalid_config = {"invalid": "config"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigValidationError):
                validate_config_from_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_create_config_from_dict_success(self, valid_config_dict):
        """Test successful creation of WorkflowConfig from dict."""
        config = create_config_from_dict(valid_config_dict, validate=False)
        assert isinstance(config, WorkflowConfig)
        assert config.workflow.name == "test-workflow"
    
    def test_create_config_from_dict_with_validation(self, valid_config_dict):
        """Test creation with validation enabled."""
        config = create_config_from_dict(valid_config_dict, validate=True)
        assert isinstance(config, WorkflowConfig)
    
    def test_create_config_from_file_success(self, valid_config_dict):
        """Test successful creation from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(valid_config_dict, f)
            temp_path = f.name
        
        try:
            config = create_config_from_file(temp_path, validate=False)
            assert isinstance(config, WorkflowConfig)
        finally:
            os.unlink(temp_path)


class TestConfigValidationError:
    """Test the ConfigValidationError exception."""
    
    def test_config_validation_error_creation(self):
        """Test creating ConfigValidationError."""
        errors = ["Error 1", "Error 2"]
        exc = ConfigValidationError("Test message", errors)
        
        assert str(exc) == "Test message"
        assert exc.errors == errors
    
    def test_config_validation_error_no_errors(self):
        """Test creating ConfigValidationError without error list."""
        exc = ConfigValidationError("Test message")
        
        assert str(exc) == "Test message"
        assert exc.errors == []


class TestSchemaValidation:
    """Test schema-specific validation scenarios."""
    
    def test_workflow_name_pattern(self):
        """Test workflow name pattern validation."""
        invalid_configs = [
            {"workflow": {"name": "invalid name with spaces", "version": "1.0.0", "environment": "development"}},
            {"workflow": {"name": "invalid@name", "version": "1.0.0", "environment": "development"}},
            {"workflow": {"name": "", "version": "1.0.0", "environment": "development"}},
        ]
        
        validator = ConfigValidator()
        
        for config in invalid_configs:
            config["models"] = {
                "planner_model": {"provider": "openai", "model_name": "gpt-4"},
                "code_model": {"provider": "openai", "model_name": "gpt-4"}
            }
            errors = validator.validate_config_dict(config)
            assert len(errors) > 0
    
    def test_version_pattern(self):
        """Test version pattern validation."""
        invalid_versions = ["1.0", "v1.0.0", "1.0.0-beta", "invalid"]
        
        validator = ConfigValidator()
        
        for version in invalid_versions:
            config = {
                "workflow": {
                    "name": "test-workflow",
                    "version": version,
                    "environment": "development"
                },
                "models": {
                    "planner_model": {"provider": "openai", "model_name": "gpt-4"},
                    "code_model": {"provider": "openai", "model_name": "gpt-4"}
                }
            }
            errors = validator.validate_config_dict(config)
            assert len(errors) > 0
    
    def test_environment_enum(self):
        """Test environment enum validation."""
        config = {
            "workflow": {
                "name": "test-workflow",
                "version": "1.0.0",
                "environment": "invalid_environment"
            },
            "models": {
                "planner_model": {"provider": "openai", "model_name": "gpt-4"},
                "code_model": {"provider": "openai", "model_name": "gpt-4"}
            }
        }
        
        validator = ConfigValidator()
        errors = validator.validate_config_dict(config)
        assert len(errors) > 0
        assert any("environment" in error.lower() for error in errors)


class TestBusinessRuleValidation:
    """Test business rule validation beyond schema validation."""
    
    def test_model_combination_validation(self):
        """Test validation of model provider/name combinations."""
        validator = ConfigValidator()
        
        # Test valid combinations
        assert validator._is_valid_model_combination("openai", "gpt-4")
        assert validator._is_valid_model_combination("anthropic", "claude-3-sonnet")
        assert validator._is_valid_model_combination("openrouter", "openai/gpt-4")
        
        # Test unknown provider (should be allowed)
        assert validator._is_valid_model_combination("custom_provider", "custom_model")
    
    def test_concurrent_limits_validation(self):
        """Test validation of concurrent processing limits."""
        config = {
            "workflow": {"name": "test", "version": "1.0.0", "environment": "development"},
            "models": {
                "planner_model": {"provider": "openai", "model_name": "gpt-4"},
                "code_model": {"provider": "openai", "model_name": "gpt-4"}
            },
            "performance": {
                "max_concurrent_scenes": 5,
                "max_concurrent_renders": 10  # Invalid: higher than scenes
            }
        }
        
        validator = ConfigValidator()
        errors = validator.validate_config_dict(config)
        assert len(errors) > 0
        assert any("concurrent" in error.lower() for error in errors)
    
    def test_gpu_preview_mode_conflict(self):
        """Test validation of conflicting GPU and preview mode settings."""
        config = {
            "workflow": {"name": "test", "version": "1.0.0", "environment": "development"},
            "models": {
                "planner_model": {"provider": "openai", "model_name": "gpt-4"},
                "code_model": {"provider": "openai", "model_name": "gpt-4"}
            },
            "features": {
                "use_gpu_acceleration": True,
                "preview_mode": True  # Conflict
            }
        }
        
        validator = ConfigValidator()
        errors = validator.validate_config_dict(config)
        assert len(errors) > 0
        assert any("gpu" in error.lower() and "preview" in error.lower() for error in errors)
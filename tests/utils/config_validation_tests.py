"""
Test utilities for configuration validation and error handling.

This module provides test cases and utilities specifically for testing
configuration validation, error handling, and edge cases.
"""

import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch

from src.config.models import (
    SystemConfig, AgentConfig, LLMProviderConfig, RAGConfig,
    EmbeddingConfig, VectorStoreConfig, ValidationResult
)
from src.config.service import ConfigurationService
from src.config.validation import ConfigValidationService
from .config_mocks import create_test_system_config, MockConfigurationManager


class ConfigurationValidationTestMixin:
    """Mixin class providing configuration validation test utilities."""
    
    def assert_validation_passes(self, config: SystemConfig, message: str = ""):
        """Assert that configuration validation passes.
        
        Args:
            config: Configuration to validate
            message: Optional message for assertion failure
        """
        validation_service = ConfigValidationService()
        result = validation_service.validate_system_config(config)
        
        if not result.valid:
            error_details = "\n".join(result.errors)
            pytest.fail(f"Configuration validation failed{': ' + message if message else ''}:\n{error_details}")
    
    def assert_validation_fails(self, config: SystemConfig, expected_errors: Optional[List[str]] = None, message: str = ""):
        """Assert that configuration validation fails.
        
        Args:
            config: Configuration to validate
            expected_errors: Optional list of expected error messages
            message: Optional message for assertion failure
        """
        validation_service = ConfigValidationService()
        result = validation_service.validate_system_config(config)
        
        if result.valid:
            pytest.fail(f"Expected configuration validation to fail{': ' + message if message else ''}")
        
        if expected_errors:
            for expected_error in expected_errors:
                if not any(expected_error in error for error in result.errors):
                    pytest.fail(f"Expected error '{expected_error}' not found in validation errors: {result.errors}")
    
    def assert_has_validation_warning(self, config: SystemConfig, expected_warning: str, message: str = ""):
        """Assert that configuration validation produces a specific warning.
        
        Args:
            config: Configuration to validate
            expected_warning: Expected warning message
            message: Optional message for assertion failure
        """
        validation_service = ConfigValidationService()
        result = validation_service.validate_system_config(config)
        
        if not any(expected_warning in warning for warning in result.warnings):
            pytest.fail(f"Expected warning '{expected_warning}' not found{': ' + message if message else ''}. Warnings: {result.warnings}")


def create_invalid_llm_provider_config() -> LLMProviderConfig:
    """Create an invalid LLM provider configuration for testing."""
    return LLMProviderConfig(
        provider="test_provider",
        models=[],  # Invalid: empty models list
        default_model="nonexistent_model",  # Invalid: not in models list
        enabled=True
    )


def create_invalid_agent_config() -> AgentConfig:
    """Create an invalid agent configuration for testing."""
    return AgentConfig(
        name="",  # Invalid: empty name
        timeout_seconds=-1,  # Invalid: negative timeout
        max_retries=-1,  # Invalid: negative retries
        temperature=3.0  # Invalid: temperature out of range
    )


def create_invalid_rag_config() -> RAGConfig:
    """Create an invalid RAG configuration for testing."""
    embedding_config = EmbeddingConfig(
        provider="jina",
        model_name="test_model",
        dimensions=-1,  # Invalid: negative dimensions
        batch_size=0  # Invalid: zero batch size
    )
    
    vector_store_config = VectorStoreConfig(
        provider="chroma",
        collection_name="test",
        max_results=-1  # Invalid: negative max results
    )
    
    return RAGConfig(
        enabled=True,
        embedding_config=embedding_config,
        vector_store_config=vector_store_config,
        chunk_size=100,
        chunk_overlap=200,  # Invalid: overlap >= chunk_size
        similarity_threshold=1.5  # Invalid: threshold > 1.0
    )


class TestConfigurationValidation:
    """Test cases for configuration validation."""
    
    def test_valid_system_config_passes_validation(self):
        """Test that a valid system configuration passes validation."""
        config = create_test_system_config()
        
        validation_service = ConfigValidationService()
        result = validation_service.validate_system_config(config)
        
        assert result.valid
        assert len(result.errors) == 0
    
    def test_invalid_llm_provider_fails_validation(self):
        """Test that invalid LLM provider configuration fails validation."""
        config = create_test_system_config()
        config.llm_providers["invalid"] = create_invalid_llm_provider_config()
        
        validation_service = ConfigValidationService()
        result = validation_service.validate_system_config(config)
        
        assert not result.valid
        assert len(result.errors) > 0
    
    def test_invalid_agent_config_fails_validation(self):
        """Test that invalid agent configuration fails validation."""
        config = create_test_system_config()
        config.agent_configs["invalid"] = create_invalid_agent_config()
        
        validation_service = ConfigValidationService()
        result = validation_service.validate_system_config(config)
        
        assert not result.valid
        assert len(result.errors) > 0
    
    def test_invalid_rag_config_fails_validation(self):
        """Test that invalid RAG configuration fails validation."""
        config = create_test_system_config()
        config.rag_config = create_invalid_rag_config()
        
        validation_service = ConfigValidationService()
        result = validation_service.validate_system_config(config)
        
        assert not result.valid
        assert len(result.errors) > 0
    
    def test_missing_required_agents_fails_validation(self):
        """Test that missing required agents fails validation."""
        config = create_test_system_config()
        # Remove required agent
        del config.agent_configs["planner_agent"]
        
        validation_service = ConfigValidationService()
        result = validation_service.validate_system_config(config)
        
        assert not result.valid
        assert any("planner_agent" in error for error in result.errors)
    
    def test_missing_api_keys_produces_warnings(self):
        """Test that missing API keys produce warnings."""
        config = create_test_system_config()
        # Remove API keys
        for provider in config.llm_providers.values():
            provider.api_key = None
        
        validation_service = ConfigValidationService()
        result = validation_service.validate_system_config(config)
        
        # Should still be valid but have warnings
        assert result.valid
        assert len(result.warnings) > 0
        assert any("API key" in warning for warning in result.warnings)
    
    def test_nonexistent_default_provider_fails_validation(self):
        """Test that nonexistent default provider fails validation."""
        config = create_test_system_config()
        config.default_llm_provider = "nonexistent_provider"
        
        validation_service = ConfigValidationService()
        result = validation_service.validate_system_config(config)
        
        assert not result.valid
        assert any("default_llm_provider" in error for error in result.errors)


class TestConfigurationErrorHandling:
    """Test cases for configuration error handling."""
    
    def test_configuration_service_handles_missing_env_file(self):
        """Test that ConfigurationService handles missing .env file gracefully."""
        service = ConfigurationService()
        
        # Should not raise exception for missing file
        config = service.load_env_config("nonexistent.env")
        assert config == {}
    
    def test_configuration_service_handles_invalid_env_syntax(self):
        """Test that ConfigurationService handles invalid .env syntax."""
        import tempfile
        import os
        
        # Create temporary file with invalid syntax
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("INVALID_SYNTAX_LINE_WITHOUT_EQUALS\n")
            f.write("VALID_KEY=valid_value\n")
            temp_path = f.name
        
        try:
            service = ConfigurationService()
            # Should handle invalid syntax gracefully
            config = service.load_env_config(temp_path)
            # Should still load valid lines
            assert "VALID_KEY" in config
        finally:
            os.unlink(temp_path)
    
    def test_configuration_manager_handles_validation_errors(self):
        """Test that ConfigurationManager handles validation errors gracefully."""
        with patch('src.config.factory.ConfigurationFactory.build_system_config') as mock_build:
            # Mock factory to return invalid config
            invalid_config = create_test_system_config()
            invalid_config.agent_configs["invalid"] = create_invalid_agent_config()
            mock_build.return_value = invalid_config
            
            from src.config.manager import ConfigurationManager
            manager = ConfigurationManager()
            
            # Should not raise exception, but should log errors
            config = manager.load_system_config()
            assert config is not None
    
    def test_configuration_manager_fallback_to_minimal_config(self):
        """Test that ConfigurationManager falls back to minimal config on errors."""
        with patch('src.config.factory.ConfigurationFactory.build_system_config') as mock_build:
            # Mock factory to raise exception
            mock_build.side_effect = Exception("Configuration build failed")
            
            from src.config.manager import ConfigurationManager
            manager = ConfigurationManager()
            
            # Should fall back to minimal config
            config = manager.load_system_config()
            assert config is not None
            assert config.environment == "development"
    
    def test_configuration_validation_service_handles_exceptions(self):
        """Test that ConfigValidationService handles exceptions gracefully."""
        validation_service = ConfigValidationService()
        
        # Create config that might cause validation exceptions
        config = create_test_system_config()
        
        # Mock a method to raise exception
        with patch.object(validation_service, '_validate_provider_connections', side_effect=Exception("Connection test failed")):
            result = validation_service.validate_system_config(config)
            
            # Should handle exception and report it as error
            assert not result.valid
            assert any("Connection test failed" in error for error in result.errors)


class TestConfigurationEdgeCases:
    """Test cases for configuration edge cases."""
    
    def test_empty_configuration(self):
        """Test handling of completely empty configuration."""
        from src.config.models import SystemConfig
        
        # Create minimal config with only required fields
        config = SystemConfig(
            environment="test",
            default_llm_provider="openai"
        )
        
        validation_service = ConfigValidationService()
        result = validation_service.validate_system_config(config)
        
        # Should be valid but have warnings
        assert result.valid
        assert len(result.warnings) > 0
    
    def test_configuration_with_all_providers_disabled(self):
        """Test configuration where all providers are disabled."""
        config = create_test_system_config()
        
        # Disable all providers
        for provider in config.llm_providers.values():
            provider.enabled = False
        
        validation_service = ConfigValidationService()
        result = validation_service.validate_system_config(config)
        
        # Should produce warnings about no enabled providers
        assert len(result.warnings) > 0
        assert any("no enabled providers" in warning.lower() for warning in result.warnings)
    
    def test_configuration_with_circular_dependencies(self):
        """Test configuration with potential circular dependencies."""
        config = create_test_system_config()
        
        # Create potential circular dependency in agent tools
        config.agent_configs["agent1"] = AgentConfig(
            name="agent1",
            tools=["agent2_tool"]
        )
        config.agent_configs["agent2"] = AgentConfig(
            name="agent2", 
            tools=["agent1_tool"]
        )
        
        validation_service = ConfigValidationService()
        result = validation_service.validate_system_config(config)
        
        # Should handle gracefully (circular dependencies in tools are allowed)
        assert result.valid or len(result.errors) == 0
    
    def test_configuration_with_extreme_values(self):
        """Test configuration with extreme values."""
        config = create_test_system_config()
        
        # Set extreme values
        config.workflow_config.max_scene_concurrency = 1000
        config.workflow_config.workflow_timeout_seconds = 86400  # 24 hours
        
        for agent in config.agent_configs.values():
            agent.timeout_seconds = 3600  # 1 hour
            agent.max_retries = 100
        
        validation_service = ConfigValidationService()
        result = validation_service.validate_system_config(config)
        
        # Should handle extreme values (may produce warnings)
        assert result.valid
    
    def test_configuration_with_unicode_values(self):
        """Test configuration with unicode values."""
        config = create_test_system_config()
        
        # Add unicode values
        config.agent_configs["unicode_agent"] = AgentConfig(
            name="unicode_agent_测试",
            system_prompt="System prompt with unicode: 你好世界"
        )
        
        validation_service = ConfigValidationService()
        result = validation_service.validate_system_config(config)
        
        # Should handle unicode values
        assert result.valid


def create_test_validation_scenarios() -> List[Dict[str, Any]]:
    """Create a list of test validation scenarios.
    
    Returns:
        List of dictionaries containing test scenarios with:
        - name: Test scenario name
        - config: SystemConfig to test
        - should_pass: Whether validation should pass
        - expected_errors: List of expected error messages (if should_pass is False)
        - expected_warnings: List of expected warning messages
    """
    scenarios = []
    
    # Valid configuration
    scenarios.append({
        "name": "valid_complete_config",
        "config": create_test_system_config(),
        "should_pass": True,
        "expected_errors": [],
        "expected_warnings": []
    })
    
    # Missing required agents
    config_missing_agents = create_test_system_config()
    del config_missing_agents.agent_configs["planner_agent"]
    scenarios.append({
        "name": "missing_required_agents",
        "config": config_missing_agents,
        "should_pass": False,
        "expected_errors": ["Required agent missing: planner_agent"],
        "expected_warnings": []
    })
    
    # Invalid provider configuration
    config_invalid_provider = create_test_system_config()
    config_invalid_provider.llm_providers["invalid"] = create_invalid_llm_provider_config()
    scenarios.append({
        "name": "invalid_provider_config",
        "config": config_invalid_provider,
        "should_pass": False,
        "expected_errors": ["At least one model must be specified"],
        "expected_warnings": []
    })
    
    # Missing API keys (should pass with warnings)
    config_no_keys = create_test_system_config()
    for provider in config_no_keys.llm_providers.values():
        provider.api_key = None
    scenarios.append({
        "name": "missing_api_keys",
        "config": config_no_keys,
        "should_pass": True,
        "expected_errors": [],
        "expected_warnings": ["No API key configured"]
    })
    
    return scenarios


# Pytest fixtures for configuration testing

@pytest.fixture
def valid_test_config():
    """Fixture providing a valid test configuration."""
    return create_test_system_config()


@pytest.fixture
def invalid_test_config():
    """Fixture providing an invalid test configuration."""
    config = create_test_system_config()
    config.agent_configs["invalid"] = create_invalid_agent_config()
    return config


@pytest.fixture
def config_validation_service():
    """Fixture providing a ConfigValidationService instance."""
    return ConfigValidationService()


@pytest.fixture
def mock_config_manager():
    """Fixture providing a mock ConfigurationManager."""
    return MockConfigurationManager()


@pytest.fixture(params=create_test_validation_scenarios())
def validation_scenario(request):
    """Parametrized fixture providing different validation scenarios."""
    return request.param
"""
Configuration mocking utilities for testing.

This module provides utilities to mock the configuration system for tests,
allowing tests to use controlled configuration without relying on environment
variables or external configuration files.
"""

import os
import tempfile
from typing import Dict, Any, Optional, List, Union
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from contextlib import contextmanager

from src.config.models import (
    SystemConfig, AgentConfig, LLMProviderConfig, RAGConfig, 
    EmbeddingConfig, VectorStoreConfig, MonitoringConfig, 
    LangfuseConfig, WorkflowConfig, HumanLoopConfig,
    DoclingConfig, MCPServerConfig, Context7Config, ValidationResult
)
from src.config.manager import ConfigurationManager


class MockConfigurationManager:
    """Mock configuration manager for testing."""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """Initialize mock configuration manager.
        
        Args:
            config: Optional SystemConfig to use. If None, creates default test config.
        """
        self._config = config or create_test_system_config()
        self._cache = {}
        self._cache_timestamps = {}
    
    @property
    def config(self) -> SystemConfig:
        """Get the mock system configuration."""
        return self._config
    
    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """Get mock agent configuration."""
        return self._config.agent_configs.get(agent_name)
    
    def get_llm_config(self) -> Dict[str, LLMProviderConfig]:
        """Get mock LLM configuration."""
        return self._config.llm_providers
    
    def get_rag_config(self) -> Optional[RAGConfig]:
        """Get mock RAG configuration."""
        return self._config.rag_config
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get mock monitoring configuration."""
        return self._config.monitoring_config
    
    def get_provider_config(self, provider_name: str) -> Optional[LLMProviderConfig]:
        """Get mock provider configuration."""
        return self._config.llm_providers.get(provider_name)
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get mock model configuration."""
        # Determine provider from model name
        if model_name.startswith('openai/'):
            provider_name = 'openai'
        elif model_name.startswith('openrouter/'):
            provider_name = 'openrouter'
        elif model_name.startswith('bedrock/'):
            provider_name = 'bedrock'
        else:
            provider_name = self._config.default_llm_provider
        
        provider_config = self._config.llm_providers.get(provider_name)
        if not provider_config:
            return {'model_name': model_name}
        
        return {
            'model_name': model_name,
            'provider': provider_name,
            'api_key': provider_config.api_key,
            'base_url': provider_config.base_url,
            'timeout': provider_config.timeout,
            'max_retries': provider_config.max_retries
        }
    
    def is_provider_enabled(self, provider_name: str) -> bool:
        """Check if provider is enabled."""
        provider_config = self.get_provider_config(provider_name)
        return provider_config is not None and provider_config.enabled
    
    def is_agent_enabled(self, agent_name: str) -> bool:
        """Check if agent is enabled."""
        agent_config = self.get_agent_config(agent_name)
        return agent_config is not None and agent_config.enabled
    
    def is_rag_enabled(self) -> bool:
        """Check if RAG is enabled."""
        return self._config.rag_config is not None and self._config.rag_config.enabled
    
    def is_monitoring_enabled(self) -> bool:
        """Check if monitoring is enabled."""
        return self._config.monitoring_config.enabled
    
    def get_environment(self) -> str:
        """Get environment."""
        return self._config.environment
    
    def is_development_mode(self) -> bool:
        """Check if in development mode."""
        return self._config.environment == 'development'
    
    def validate_configuration(self, config: Optional[SystemConfig] = None) -> ValidationResult:
        """Mock configuration validation."""
        return ValidationResult(valid=True)
    
    def reload_config(self, force: bool = False) -> SystemConfig:
        """Mock config reload."""
        return self._config
    
    def clear_cache(self):
        """Mock cache clearing."""
        self._cache.clear()
        self._cache_timestamps.clear()


def create_test_llm_provider_config(
    provider: str = "openai",
    api_key: str = "test_api_key",
    models: Optional[List[str]] = None,
    default_model: str = "gpt-4o"
) -> LLMProviderConfig:
    """Create a test LLM provider configuration.
    
    Args:
        provider: Provider name
        api_key: API key for testing
        models: List of available models
        default_model: Default model name
        
    Returns:
        LLMProviderConfig for testing
    """
    if models is None:
        if provider == "openai":
            models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
        elif provider == "openrouter":
            models = ["openrouter/anthropic/claude-3.5-sonnet", "openrouter/openai/gpt-4o"]
        elif provider == "bedrock":
            models = ["bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"]
        else:
            models = [default_model]
    
    return LLMProviderConfig(
        provider=provider,
        api_key=api_key,
        models=models,
        default_model=default_model,
        enabled=True,
        timeout=30,
        max_retries=3
    )


def create_test_agent_config(
    name: str = "test_agent",
    planner_model: str = "openai/gpt-4o",
    scene_model: str = "openai/gpt-4o",
    helper_model: str = "openai/gpt-4o-mini",
    tools: Optional[List[str]] = None
) -> AgentConfig:
    """Create a test agent configuration.
    
    Args:
        name: Agent name
        planner_model: Planner model name
        scene_model: Scene model name
        helper_model: Helper model name
        tools: List of available tools
        
    Returns:
        AgentConfig for testing
    """
    if tools is None:
        tools = ["test_tool"]
    
    return AgentConfig(
        name=name,
        planner_model=planner_model,
        scene_model=scene_model,
        helper_model=helper_model,
        tools=tools,
        max_retries=3,
        timeout_seconds=300,
        enable_human_loop=False,
        temperature=0.7,
        print_cost=True,
        verbose=False,
        enabled=True
    )


def create_test_rag_config(
    enabled: bool = True,
    embedding_provider: str = "jina",
    vector_store_provider: str = "chroma"
) -> RAGConfig:
    """Create a test RAG configuration.
    
    Args:
        enabled: Whether RAG is enabled
        embedding_provider: Embedding provider name
        vector_store_provider: Vector store provider name
        
    Returns:
        RAGConfig for testing
    """
    embedding_config = EmbeddingConfig(
        provider=embedding_provider,
        model_name="jina-embeddings-v3" if embedding_provider == "jina" else "text-embedding-3-small",
        api_key="test_embedding_key",
        dimensions=1024,
        batch_size=100,
        timeout=30
    )
    
    vector_store_config = VectorStoreConfig(
        provider=vector_store_provider,
        collection_name="test_collection",
        connection_params={"db_path": "test_chroma_db"} if vector_store_provider == "chroma" else {},
        max_results=50
    )
    
    return RAGConfig(
        enabled=enabled,
        embedding_config=embedding_config,
        vector_store_config=vector_store_config,
        chunk_size=1000,
        chunk_overlap=200,
        default_k_value=5,
        similarity_threshold=0.7,
        enable_caching=True,
        cache_ttl=3600
    )


def create_test_monitoring_config(
    enabled: bool = True,
    langfuse_enabled: bool = False
) -> MonitoringConfig:
    """Create a test monitoring configuration.
    
    Args:
        enabled: Whether monitoring is enabled
        langfuse_enabled: Whether Langfuse is enabled
        
    Returns:
        MonitoringConfig for testing
    """
    langfuse_config = None
    if langfuse_enabled:
        langfuse_config = LangfuseConfig(
            enabled=True,
            secret_key="test_langfuse_secret",
            public_key="test_langfuse_public",
            host="https://test-langfuse.com"
        )
    
    return MonitoringConfig(
        enabled=enabled,
        langfuse_config=langfuse_config,
        log_level="INFO",
        performance_tracking=True,
        error_tracking=True
    )


def create_test_system_config(
    environment: str = "test",
    debug: bool = True,
    include_rag: bool = True,
    include_monitoring: bool = True
) -> SystemConfig:
    """Create a complete test system configuration.
    
    Args:
        environment: Environment name
        debug: Debug mode flag
        include_rag: Whether to include RAG configuration
        include_monitoring: Whether to include monitoring configuration
        
    Returns:
        SystemConfig for testing
    """
    # Create LLM providers
    llm_providers = {
        "openai": create_test_llm_provider_config("openai"),
        "openrouter": create_test_llm_provider_config(
            "openrouter", 
            models=["openrouter/anthropic/claude-3.5-sonnet", "openrouter/openai/gpt-4o"],
            default_model="openrouter/anthropic/claude-3.5-sonnet"
        ),
        "bedrock": create_test_llm_provider_config(
            "bedrock",
            models=["bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"],
            default_model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
        )
    }
    
    # Create agent configurations
    agent_configs = {
        "planner_agent": create_test_agent_config(
            "planner_agent",
            tools=["planning_tool"]
        ),
        "code_generator_agent": create_test_agent_config(
            "code_generator_agent",
            tools=["code_generation_tool", "rag_tool"]
        ),
        "renderer_agent": create_test_agent_config(
            "renderer_agent",
            tools=["render_tool"]
        ),
        "rag_agent": create_test_agent_config(
            "rag_agent",
            tools=["rag_tool", "vector_search_tool"]
        ),
        "error_handler_agent": create_test_agent_config(
            "error_handler_agent",
            tools=["error_tool"]
        ),
        "human_loop_agent": create_test_agent_config(
            "human_loop_agent",
            tools=["human_tool"],
            enable_human_loop=True
        ),
        "monitoring_agent": create_test_agent_config(
            "monitoring_agent",
            tools=["monitoring_tool"]
        )
    }
    
    # Create optional configurations
    rag_config = create_test_rag_config() if include_rag else None
    monitoring_config = create_test_monitoring_config() if include_monitoring else MonitoringConfig(enabled=False)
    
    # Create workflow configuration
    workflow_config = WorkflowConfig(
        max_workflow_retries=3,
        workflow_timeout_seconds=3600,
        output_dir="test_output",
        max_scene_concurrency=5,
        max_topic_concurrency=1,
        max_concurrent_renders=4,
        default_quality="medium",
        use_gpu_acceleration=False,
        preview_mode=False
    )
    
    # Create human loop configuration
    human_loop_config = HumanLoopConfig(
        enabled=True,
        timeout_seconds=300
    )
    
    # Create other configurations
    docling_config = DoclingConfig(enabled=True, max_file_size_mb=50)
    context7_config = Context7Config(enabled=True)
    
    return SystemConfig(
        environment=environment,
        debug=debug,
        default_llm_provider="openai",
        llm_providers=llm_providers,
        rag_config=rag_config,
        agent_configs=agent_configs,
        monitoring_config=monitoring_config,
        workflow_config=workflow_config,
        human_loop_config=human_loop_config,
        docling_config=docling_config,
        context7_config=context7_config,
        mcp_servers={}
    )


@contextmanager
def mock_configuration_manager(config: Optional[SystemConfig] = None):
    """Context manager to mock ConfigurationManager for tests.
    
    Args:
        config: Optional SystemConfig to use. If None, creates default test config.
        
    Yields:
        MockConfigurationManager instance
    """
    mock_manager = MockConfigurationManager(config)
    
    with patch('src.config.manager.ConfigurationManager') as mock_class:
        # Make the mock class return our mock instance
        mock_class.return_value = mock_manager
        mock_class._instance = mock_manager
        
        # Also patch any direct imports
        with patch('src.langgraph_agents.base_agent.ConfigurationManager', return_value=mock_manager), \
             patch('src.langgraph_agents.config.ConfigurationManager', return_value=mock_manager), \
             patch('src.rag.vector_store_factory.ConfigurationManager', return_value=mock_manager):
            
            yield mock_manager


@contextmanager
def mock_environment_variables(env_vars: Dict[str, str]):
    """Context manager to mock environment variables for tests.
    
    Args:
        env_vars: Dictionary of environment variables to set
        
    Yields:
        None
    """
    with patch.dict(os.environ, env_vars, clear=False):
        yield


def create_test_env_file(env_vars: Dict[str, str], file_path: Optional[str] = None) -> str:
    """Create a temporary .env file for testing.
    
    Args:
        env_vars: Dictionary of environment variables to write
        file_path: Optional file path. If None, creates temporary file.
        
    Returns:
        Path to the created .env file
    """
    if file_path is None:
        # Create temporary file
        fd, file_path = tempfile.mkstemp(suffix='.env', text=True)
        os.close(fd)
    
    with open(file_path, 'w') as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
    
    return file_path


class ConfigurationTestCase:
    """Base test case class with configuration mocking utilities."""
    
    def setup_method(self):
        """Set up test method with default configuration."""
        self.test_config = create_test_system_config()
        self.mock_manager = MockConfigurationManager(self.test_config)
    
    def teardown_method(self):
        """Clean up after test method."""
        # Clear any patches or temporary files
        pass
    
    def get_test_agent_config(self, agent_name: str) -> AgentConfig:
        """Get test agent configuration by name.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            AgentConfig for the specified agent
        """
        return self.test_config.agent_configs[agent_name]
    
    def get_test_provider_config(self, provider_name: str) -> LLMProviderConfig:
        """Get test provider configuration by name.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            LLMProviderConfig for the specified provider
        """
        return self.test_config.llm_providers[provider_name]
    
    def assert_config_valid(self, config: SystemConfig):
        """Assert that a configuration is valid.
        
        Args:
            config: SystemConfig to validate
        """
        validation_result = self.mock_manager.validate_configuration(config)
        assert validation_result.valid, f"Configuration validation failed: {validation_result.errors}"


# Convenience functions for common test scenarios

def get_openai_test_config() -> Dict[str, str]:
    """Get OpenAI test environment variables."""
    return {
        'OPENAI_API_KEY': 'test_openai_key',
        'OPENAI_MODELS': 'gpt-4o,gpt-4o-mini,gpt-3.5-turbo',
        'OPENAI_DEFAULT_MODEL': 'gpt-4o',
        'DEFAULT_LLM_PROVIDER': 'openai'
    }


def get_openrouter_test_config() -> Dict[str, str]:
    """Get OpenRouter test environment variables."""
    return {
        'OPENROUTER_API_KEY': 'test_openrouter_key',
        'OPENROUTER_MODELS': 'openrouter/anthropic/claude-3.5-sonnet,openrouter/openai/gpt-4o',
        'OPENROUTER_DEFAULT_MODEL': 'openrouter/anthropic/claude-3.5-sonnet',
        'DEFAULT_LLM_PROVIDER': 'openrouter'
    }


def get_rag_test_config() -> Dict[str, str]:
    """Get RAG test environment variables."""
    return {
        'RAG_ENABLED': 'true',
        'EMBEDDING_PROVIDER': 'jina',
        'JINA_API_KEY': 'test_jina_key',
        'JINA_MODEL': 'jina-embeddings-v3',
        'EMBEDDING_DIMENSIONS': '1024',
        'VECTOR_STORE_PROVIDER': 'chroma',
        'CHROMA_DB_PATH': 'test_chroma_db'
    }


def get_monitoring_test_config() -> Dict[str, str]:
    """Get monitoring test environment variables."""
    return {
        'MONITORING_ENABLED': 'true',
        'LANGFUSE_ENABLED': 'false',
        'LOG_LEVEL': 'INFO'
    }


def get_complete_test_config() -> Dict[str, str]:
    """Get complete test environment configuration."""
    config = {}
    config.update(get_openai_test_config())
    config.update(get_rag_test_config())
    config.update(get_monitoring_test_config())
    config.update({
        'ENVIRONMENT': 'test',
        'DEBUG': 'true',
        'OUTPUT_DIR': 'test_output'
    })
    return config
"""
Configuration Package

Contains configuration management and settings for the FastAPI application.
This package provides centralized configuration management using Pydantic settings
with environment-specific overrides and validation.

Usage:
    from app.config import get_settings, Settings
    
    settings = get_settings()
    print(settings.app.app_name)
"""

from .settings import (
    Settings,
    AppSettings,
    DatabaseSettings,
    LLMSettings,
    RAGSettings,
    TTSSettings,
    MonitoringSettings,
    SecuritySettings,
    Environment,
    LogLevel,
    EmbeddingProvider,
    VectorStoreProvider,
    get_settings,
    reload_settings,
)

from .factory import (
    ConfigurationFactory,
    ConfigurationError,
    create_settings,
)

from .validation import (
    ValidationResult,
    ConfigValidator,
    validate_configuration,
    check_required_environment_variables,
)

from .environments import (
    DevelopmentConfig,
    TestingConfig,
    ProductionConfig,
    get_environment_config,
    apply_environment_overrides,
)

__all__ = [
    # Main settings classes
    "Settings",
    "AppSettings",
    "DatabaseSettings", 
    "LLMSettings",
    "RAGSettings",
    "TTSSettings",
    "MonitoringSettings",
    "SecuritySettings",
    
    # Enums
    "Environment",
    "LogLevel",
    "EmbeddingProvider",
    "VectorStoreProvider",
    
    # Factory and creation functions
    "ConfigurationFactory",
    "ConfigurationError",
    "create_settings",
    "get_settings",
    "reload_settings",
    
    # Validation
    "ValidationResult",
    "ConfigValidator",
    "validate_configuration",
    "check_required_environment_variables",
    
    # Environment configs
    "DevelopmentConfig",
    "TestingConfig", 
    "ProductionConfig",
    "get_environment_config",
    "apply_environment_overrides",
]
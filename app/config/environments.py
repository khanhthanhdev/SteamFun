"""
Environment-specific configuration overrides

This module provides environment-specific configuration overrides
for development, testing, and production environments.
"""

from typing import Dict, Any
from .settings import Environment


class DevelopmentConfig:
    """Development environment configuration overrides"""
    
    @staticmethod
    def get_overrides() -> Dict[str, Any]:
        return {
            "app": {
                "debug": True,
                "reload": True,
                "log_level": "DEBUG",
                "docs_url": "/docs",
                "redoc_url": "/redoc",
                "openapi_url": "/openapi.json",
            },
            "database": {
                "database_pool_size": 5,
                "database_max_overflow": 10,
            },
            "security": {
                "cors_origins": ["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:3000"],
                "cors_allow_credentials": True,
                "rate_limit_requests_per_minute": 120,  # More lenient for development
            },
            "rag": {
                "rag_enable_caching": True,
                "rag_cache_ttl": 1800,  # Shorter cache for development
                "rag_log_level": "DEBUG",
                "rag_enable_performance_monitoring": True,
            },
            "monitoring": {
                "langfuse_enabled": False,  # Disable external monitoring in dev
                "performance_tracking": True,
                "error_tracking": True,
            }
        }


class TestingConfig:
    """Testing environment configuration overrides"""
    
    @staticmethod
    def get_overrides() -> Dict[str, Any]:
        return {
            "app": {
                "debug": False,
                "reload": False,
                "log_level": "WARNING",
                "docs_url": None,  # Disable docs in testing
                "redoc_url": None,
                "openapi_url": None,
            },
            "database": {
                "database_name": "test_db",
                "database_pool_size": 2,
                "database_max_overflow": 5,
            },
            "security": {
                "cors_origins": ["http://localhost"],
                "rate_limit_enabled": False,  # Disable rate limiting for tests
            },
            "rag": {
                "rag_enable_caching": False,  # Disable caching for consistent tests
                "rag_enable_performance_monitoring": False,
                "rag_enable_usage_tracking": False,
                "rag_log_level": "ERROR",
                "vector_store_provider": "chroma",  # Use local vector store for tests
                "chroma_db_path": "test_data/chroma_db",
            },
            "monitoring": {
                "langfuse_enabled": False,
                "monitoring_enabled": False,
                "performance_tracking": False,
                "error_tracking": False,
                "execution_tracing": False,
            },
            "tts": {
                "kokoro_model_path": "test_models/kokoro-test.onnx",
                "kokoro_voices_path": "test_models/voices-test.bin",
            }
        }


class ProductionConfig:
    """Production environment configuration overrides"""
    
    @staticmethod
    def get_overrides() -> Dict[str, Any]:
        return {
            "app": {
                "debug": False,
                "reload": False,
                "log_level": "INFO",
                "workers": 4,  # Multiple workers for production
            },
            "database": {
                "database_pool_size": 20,
                "database_max_overflow": 40,
                "database_pool_timeout": 60,
            },
            "security": {
                "cors_origins": [],  # Must be explicitly set in production
                "cors_allow_credentials": True,
                "rate_limit_enabled": True,
                "rate_limit_requests_per_minute": 30,  # Stricter rate limiting
            },
            "rag": {
                "rag_enable_caching": True,
                "rag_cache_ttl": 7200,  # Longer cache for production
                "rag_max_cache_size": 5000,
                "rag_log_level": "INFO",
                "rag_enable_performance_monitoring": True,
                "rag_enable_usage_tracking": True,
                "rag_quality_sample_rate": 0.05,  # Lower sampling rate
            },
            "monitoring": {
                "langfuse_enabled": True,
                "monitoring_enabled": True,
                "performance_tracking": True,
                "error_tracking": True,
                "execution_tracing": True,
            }
        }


def get_environment_config(environment: Environment) -> Dict[str, Any]:
    """
    Get configuration overrides for the specified environment
    
    Args:
        environment: The target environment
        
    Returns:
        Dictionary of configuration overrides
    """
    config_map = {
        Environment.DEVELOPMENT: DevelopmentConfig.get_overrides(),
        Environment.TESTING: TestingConfig.get_overrides(),
        Environment.PRODUCTION: ProductionConfig.get_overrides(),
    }
    
    return config_map.get(environment, {})


def apply_environment_overrides(settings_dict: Dict[str, Any], environment: Environment) -> Dict[str, Any]:
    """
    Apply environment-specific overrides to settings dictionary
    
    Args:
        settings_dict: Base settings dictionary
        environment: Target environment
        
    Returns:
        Updated settings dictionary with environment overrides applied
    """
    overrides = get_environment_config(environment)
    
    def deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively update nested dictionaries"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
        return base_dict
    
    return deep_update(settings_dict.copy(), overrides)
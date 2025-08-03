"""
Configuration Factory

This module provides a factory for creating configuration instances
based on the current environment with proper validation and error handling.
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from .settings import Settings, Environment
from .environments import apply_environment_overrides


logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when there's an error in configuration setup"""
    pass


class ConfigurationFactory:
    """Factory class for creating and managing configuration instances"""
    
    _instance: Optional[Settings] = None
    _environment: Optional[Environment] = None
    
    @classmethod
    def create_settings(
        cls,
        environment: Optional[Environment] = None,
        env_file: Optional[str] = None,
        validate_required: bool = True
    ) -> Settings:
        """
        Create a Settings instance with environment-specific configuration
        
        Args:
            environment: Target environment (auto-detected if None)
            env_file: Path to environment file (defaults to .env)
            validate_required: Whether to validate required settings
            
        Returns:
            Configured Settings instance
            
        Raises:
            ConfigurationError: If configuration is invalid or incomplete
        """
        try:
            # Determine environment
            if environment is None:
                environment = cls._detect_environment()
            
            # Set environment file path
            if env_file is None:
                env_file = cls._get_env_file_path(environment)
            
            # Create base settings
            settings = Settings(_env_file=env_file)
            
            # Apply environment-specific overrides
            cls._apply_environment_overrides(settings, environment)
            
            # Validate configuration if requested
            if validate_required:
                cls._validate_configuration(settings, environment)
            
            # Cache the instance
            cls._instance = settings
            cls._environment = environment
            
            logger.info(f"Configuration loaded successfully for environment: {environment.value}")
            return settings
            
        except Exception as e:
            logger.error(f"Failed to create configuration: {str(e)}")
            raise ConfigurationError(f"Configuration creation failed: {str(e)}") from e
    
    @classmethod
    def get_settings(cls) -> Settings:
        """
        Get the cached Settings instance or create a new one
        
        Returns:
            Settings instance
        """
        if cls._instance is None:
            cls._instance = cls.create_settings()
        return cls._instance
    
    @classmethod
    def reload_settings(
        cls,
        environment: Optional[Environment] = None,
        env_file: Optional[str] = None
    ) -> Settings:
        """
        Reload configuration from environment variables and files
        
        Args:
            environment: Target environment (uses cached if None)
            env_file: Path to environment file
            
        Returns:
            Reloaded Settings instance
        """
        if environment is None:
            environment = cls._environment or cls._detect_environment()
        
        cls._instance = None  # Clear cache
        return cls.create_settings(environment=environment, env_file=env_file)
    
    @classmethod
    def _detect_environment(cls) -> Environment:
        """
        Detect the current environment from environment variables
        
        Returns:
            Detected Environment
        """
        env_value = os.getenv("ENVIRONMENT", "development").lower()
        
        try:
            return Environment(env_value)
        except ValueError:
            logger.warning(f"Invalid environment value '{env_value}', defaulting to development")
            return Environment.DEVELOPMENT
    
    @classmethod
    def _get_env_file_path(cls, environment: Environment) -> str:
        """
        Get the appropriate .env file path for the environment
        
        Args:
            environment: Target environment
            
        Returns:
            Path to environment file
        """
        base_path = Path(".")
        
        # Environment-specific .env files
        env_files = {
            Environment.DEVELOPMENT: base_path / ".env",
            Environment.TESTING: base_path / ".env.test",
            Environment.PRODUCTION: base_path / ".env.prod",
        }
        
        env_file = env_files.get(environment, base_path / ".env")
        
        # Fall back to .env if specific file doesn't exist
        if not env_file.exists() and environment != Environment.DEVELOPMENT:
            logger.warning(f"Environment file {env_file} not found, falling back to .env")
            env_file = base_path / ".env"
        
        return str(env_file)
    
    @classmethod
    def _apply_environment_overrides(cls, settings: Settings, environment: Environment) -> None:
        """
        Apply environment-specific configuration overrides
        
        Args:
            settings: Settings instance to modify
            environment: Target environment
        """
        try:
            # Convert settings to dict for easier manipulation
            settings_dict = settings.model_dump()
            
            # Apply environment overrides
            updated_dict = apply_environment_overrides(settings_dict, environment)
            
            # Update settings object with overrides
            for section_name, section_config in updated_dict.items():
                if hasattr(settings, section_name):
                    section_obj = getattr(settings, section_name)
                    for key, value in section_config.items():
                        if hasattr(section_obj, key):
                            setattr(section_obj, key, value)
            
        except Exception as e:
            logger.error(f"Failed to apply environment overrides: {str(e)}")
            raise ConfigurationError(f"Environment override application failed: {str(e)}") from e
    
    @classmethod
    def _validate_configuration(cls, settings: Settings, environment: Environment) -> None:
        """
        Validate configuration for completeness and correctness
        
        Args:
            settings: Settings instance to validate
            environment: Target environment
            
        Raises:
            ConfigurationError: If validation fails
        """
        errors = []
        
        # Validate production-specific requirements
        if environment == Environment.PRODUCTION:
            errors.extend(cls._validate_production_config(settings))
        
        # Validate LLM provider configuration
        errors.extend(cls._validate_llm_config(settings))
        
        # Validate RAG configuration if enabled
        if settings.rag.rag_enabled:
            errors.extend(cls._validate_rag_config(settings))
        
        # Validate monitoring configuration if enabled
        if settings.monitoring.monitoring_enabled:
            errors.extend(cls._validate_monitoring_config(settings))
        
        if errors:
            error_message = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            logger.error(error_message)
            raise ConfigurationError(error_message)
    
    @classmethod
    def _validate_production_config(cls, settings: Settings) -> list[str]:
        """Validate production-specific configuration requirements"""
        errors = []
        
        # Security validations
        if settings.security.secret_key == "your-secret-key-change-in-production":
            errors.append("SECRET_KEY must be changed from default value in production")
        
        if not settings.security.cors_origins or settings.security.cors_origins == ["*"]:
            errors.append("CORS_ORIGINS must be explicitly configured in production")
        
        # Database validations
        if not settings.database.database_url and not settings.database.database_password:
            errors.append("Database password must be set in production")
        
        return errors
    
    @classmethod
    def _validate_llm_config(cls, settings: Settings) -> list[str]:
        """Validate LLM provider configuration"""
        errors = []
        
        # Check if at least one LLM provider is configured
        has_openai = bool(settings.llm.openai_api_key)
        has_openrouter = bool(settings.llm.openrouter_api_key)
        has_gemini = bool(settings.llm.gemini_api_key)
        has_bedrock = bool(settings.llm.aws_access_key_id and settings.llm.aws_secret_access_key)
        has_vertex = bool(settings.llm.vertexai_project and settings.llm.google_application_credentials)
        
        if not any([has_openai, has_openrouter, has_gemini, has_bedrock, has_vertex]):
            errors.append("At least one LLM provider must be configured")
        
        return errors
    
    @classmethod
    def _validate_rag_config(cls, settings: Settings) -> list[str]:
        """Validate RAG system configuration"""
        errors = []
        
        # Validate embedding provider configuration
        if settings.rag.embedding_provider == "jina" and not settings.rag.jina_api_key:
            errors.append("JINA_API_KEY is required when using JINA embedding provider")
        
        # Validate vector store configuration
        if settings.rag.vector_store_provider == "astradb":
            if not settings.rag.astradb_api_endpoint:
                errors.append("ASTRADB_API_ENDPOINT is required when using AstraDB")
            if not settings.rag.astradb_application_token:
                errors.append("ASTRADB_APPLICATION_TOKEN is required when using AstraDB")
        
        # Validate document paths
        if not Path(settings.rag.manim_docs_path).exists():
            errors.append(f"MANIM_DOCS_PATH directory does not exist: {settings.rag.manim_docs_path}")
        
        return errors
    
    @classmethod
    def _validate_monitoring_config(cls, settings: Settings) -> list[str]:
        """Validate monitoring configuration"""
        errors = []
        
        if settings.monitoring.langfuse_enabled:
            if not settings.monitoring.langfuse_secret_key:
                errors.append("LANGFUSE_SECRET_KEY is required when LangFuse is enabled")
            if not settings.monitoring.langfuse_public_key:
                errors.append("LANGFUSE_PUBLIC_KEY is required when LangFuse is enabled")
        
        return errors


# Convenience functions for common use cases
def create_settings(environment: Optional[Environment] = None) -> Settings:
    """Create a Settings instance with environment-specific configuration"""
    return ConfigurationFactory.create_settings(environment=environment)


def get_settings() -> Settings:
    """Get the cached Settings instance or create a new one"""
    return ConfigurationFactory.get_settings()


def reload_settings() -> Settings:
    """Reload configuration from environment variables and files"""
    return ConfigurationFactory.reload_settings()
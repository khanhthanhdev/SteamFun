"""
Configuration Validation Utilities

This module provides utilities for validating configuration settings
and ensuring all required values are properly set.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from .settings import Settings, Environment


logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of configuration validation"""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
    
    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)"""
        return len(self.errors) == 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings"""
        return len(self.warnings) > 0
    
    def add_error(self, message: str) -> None:
        """Add an error message"""
        self.errors.append(message)
        logger.error(f"Config validation error: {message}")
    
    def add_warning(self, message: str) -> None:
        """Add a warning message"""
        self.warnings.append(message)
        logger.warning(f"Config validation warning: {message}")
    
    def add_info(self, message: str) -> None:
        """Add an info message"""
        self.info.append(message)
        logger.info(f"Config validation info: {message}")
    
    def get_summary(self) -> str:
        """Get a summary of validation results"""
        summary = []
        
        if self.errors:
            summary.append(f"Errors ({len(self.errors)}):")
            for error in self.errors:
                summary.append(f"  - {error}")
        
        if self.warnings:
            summary.append(f"Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                summary.append(f"  - {warning}")
        
        if self.info:
            summary.append(f"Info ({len(self.info)}):")
            for info in self.info:
                summary.append(f"  - {info}")
        
        if not summary:
            summary.append("Configuration validation passed successfully")
        
        return "\n".join(summary)


class ConfigValidator:
    """Configuration validator with comprehensive checks"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.result = ValidationResult()
    
    def validate_all(self) -> ValidationResult:
        """Run all validation checks"""
        self.validate_app_settings()
        self.validate_security_settings()
        self.validate_database_settings()
        self.validate_llm_settings()
        self.validate_rag_settings()
        self.validate_monitoring_settings()
        self.validate_file_paths()
        
        return self.result
    
    def validate_app_settings(self) -> None:
        """Validate core application settings"""
        app = self.settings.app
        
        # Check environment-specific settings
        if app.is_production:
            if app.debug:
                self.result.add_warning("Debug mode is enabled in production")
            
            if app.reload:
                self.result.add_warning("Auto-reload is enabled in production")
            
            if app.workers < 2:
                self.result.add_warning("Consider using multiple workers in production")
        
        # Validate port range
        if not (1 <= app.port <= 65535):
            self.result.add_error(f"Invalid port number: {app.port}")
        
        # Check upload directory
        upload_path = Path(app.upload_dir)
        if not upload_path.exists():
            try:
                upload_path.mkdir(parents=True, exist_ok=True)
                self.result.add_info(f"Created upload directory: {app.upload_dir}")
            except Exception as e:
                self.result.add_error(f"Cannot create upload directory {app.upload_dir}: {str(e)}")
    
    def validate_security_settings(self) -> None:
        """Validate security-related settings"""
        security = self.settings.security
        app = self.settings.app
        
        # Check secret key
        if security.secret_key == "your-secret-key-change-in-production":
            if app.is_production:
                self.result.add_error("Secret key must be changed from default in production")
            else:
                self.result.add_warning("Using default secret key (change for production)")
        
        if len(security.secret_key) < 32:
            self.result.add_warning("Secret key should be at least 32 characters long")
        
        # Check CORS settings
        if app.is_production:
            if "*" in security.cors_origins:
                self.result.add_error("CORS origins should not include '*' in production")
            
            if not security.cors_origins:
                self.result.add_error("CORS origins must be configured in production")
        
        # Check rate limiting
        if security.rate_limit_enabled and security.rate_limit_requests_per_minute <= 0:
            self.result.add_error("Rate limit requests per minute must be positive")
    
    def validate_database_settings(self) -> None:
        """Validate database configuration"""
        db = self.settings.database
        app = self.settings.app
        
        # Check database connection settings
        if not db.database_url:
            if not db.database_host:
                self.result.add_error("Database host is required when DATABASE_URL is not set")
            
            if not db.database_name:
                self.result.add_error("Database name is required")
            
            if app.is_production and not db.database_password:
                self.result.add_error("Database password is required in production")
        
        # Validate pool settings
        if db.database_pool_size <= 0:
            self.result.add_error("Database pool size must be positive")
        
        if db.database_max_overflow < 0:
            self.result.add_error("Database max overflow cannot be negative")
        
        if db.database_pool_timeout <= 0:
            self.result.add_error("Database pool timeout must be positive")
    
    def validate_llm_settings(self) -> None:
        """Validate LLM provider configuration"""
        llm = self.settings.llm
        
        # Check if at least one provider is configured
        providers_configured = []
        
        if llm.openai_api_key:
            providers_configured.append("OpenAI")
        
        if llm.openrouter_api_key:
            providers_configured.append("OpenRouter")
        
        if llm.gemini_api_key:
            providers_configured.append("Gemini")
        
        if llm.aws_access_key_id and llm.aws_secret_access_key:
            providers_configured.append("AWS Bedrock")
        
        if llm.vertexai_project and llm.google_application_credentials:
            providers_configured.append("Vertex AI")
        
        if not providers_configured:
            self.result.add_error("At least one LLM provider must be configured")
        else:
            self.result.add_info(f"LLM providers configured: {', '.join(providers_configured)}")
        
        # Validate Vertex AI credentials file
        if llm.google_application_credentials:
            creds_path = Path(llm.google_application_credentials)
            if not creds_path.exists():
                self.result.add_error(f"Google credentials file not found: {llm.google_application_credentials}")
    
    def validate_rag_settings(self) -> None:
        """Validate RAG system configuration"""
        rag = self.settings.rag
        
        if not rag.rag_enabled:
            self.result.add_info("RAG system is disabled")
            return
        
        # Validate embedding provider
        if rag.embedding_provider == "jina" and not rag.jina_api_key:
            self.result.add_error("JINA_API_KEY is required when using JINA embedding provider")
        
        # Validate vector store provider
        if rag.vector_store_provider == "astradb":
            if not rag.astradb_api_endpoint:
                self.result.add_error("ASTRADB_API_ENDPOINT is required when using AstraDB")
            if not rag.astradb_application_token:
                self.result.add_error("ASTRADB_APPLICATION_TOKEN is required when using AstraDB")
        
        # Validate chunk settings
        if rag.rag_chunk_size <= 0:
            self.result.add_error("RAG chunk size must be positive")
        
        if rag.rag_chunk_overlap >= rag.rag_chunk_size:
            self.result.add_error("RAG chunk overlap must be less than chunk size")
        
        if rag.rag_min_chunk_size <= 0:
            self.result.add_error("RAG minimum chunk size must be positive")
        
        # Validate thresholds
        if not (0.0 <= rag.rag_similarity_threshold <= 1.0):
            self.result.add_error("RAG similarity threshold must be between 0.0 and 1.0")
        
        if not (0.0 <= rag.rag_quality_threshold <= 1.0):
            self.result.add_error("RAG quality threshold must be between 0.0 and 1.0")
    
    def validate_monitoring_settings(self) -> None:
        """Validate monitoring configuration"""
        monitoring = self.settings.monitoring
        
        if not monitoring.monitoring_enabled:
            self.result.add_info("Monitoring is disabled")
            return
        
        if monitoring.langfuse_enabled:
            if not monitoring.langfuse_secret_key:
                self.result.add_error("LANGFUSE_SECRET_KEY is required when LangFuse is enabled")
            
            if not monitoring.langfuse_public_key:
                self.result.add_error("LANGFUSE_PUBLIC_KEY is required when LangFuse is enabled")
            
            if not monitoring.langfuse_host:
                self.result.add_error("LANGFUSE_HOST is required when LangFuse is enabled")
    
    def validate_file_paths(self) -> None:
        """Validate file and directory paths"""
        rag = self.settings.rag
        tts = self.settings.tts
        
        # Check RAG document paths
        if rag.rag_enabled:
            docs_path = Path(rag.manim_docs_path)
            if not docs_path.exists():
                self.result.add_warning(f"Manim docs path does not exist: {rag.manim_docs_path}")
            
            context_path = Path(rag.context_learning_path)
            if not context_path.exists():
                self.result.add_warning(f"Context learning path does not exist: {rag.context_learning_path}")
            
            # Check ChromaDB path if using Chroma
            if rag.vector_store_provider == "chroma":
                chroma_path = Path(rag.chroma_db_path)
                if not chroma_path.parent.exists():
                    try:
                        chroma_path.parent.mkdir(parents=True, exist_ok=True)
                        self.result.add_info(f"Created ChromaDB directory: {chroma_path.parent}")
                    except Exception as e:
                        self.result.add_error(f"Cannot create ChromaDB directory: {str(e)}")
        
        # Check TTS model paths
        kokoro_model = Path(tts.kokoro_model_path)
        if not kokoro_model.exists():
            self.result.add_warning(f"Kokoro model file not found: {tts.kokoro_model_path}")
        
        kokoro_voices = Path(tts.kokoro_voices_path)
        if not kokoro_voices.exists():
            self.result.add_warning(f"Kokoro voices file not found: {tts.kokoro_voices_path}")


def validate_configuration(settings: Settings) -> ValidationResult:
    """
    Validate configuration settings
    
    Args:
        settings: Settings instance to validate
        
    Returns:
        ValidationResult with errors, warnings, and info messages
    """
    validator = ConfigValidator(settings)
    return validator.validate_all()


def check_required_environment_variables(environment: Environment) -> ValidationResult:
    """
    Check if required environment variables are set for the given environment
    
    Args:
        environment: Target environment
        
    Returns:
        ValidationResult with missing environment variables
    """
    result = ValidationResult()
    
    # Common required variables
    common_required = []
    
    # Environment-specific required variables
    env_required = {
        Environment.PRODUCTION: [
            "SECRET_KEY",
            "DATABASE_URL",
            "CORS_ORIGINS",
        ],
        Environment.TESTING: [],
        Environment.DEVELOPMENT: [],
    }
    
    required_vars = common_required + env_required.get(environment, [])
    
    for var in required_vars:
        if not os.getenv(var):
            result.add_error(f"Required environment variable not set: {var}")
    
    return result
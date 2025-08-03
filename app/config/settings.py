"""
Centralized Configuration Management using Pydantic Settings

This module provides centralized configuration management for the FastAPI application
using Pydantic settings with environment variable support and validation.
"""

from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class Environment(str, Enum):
    """Application environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EmbeddingProvider(str, Enum):
    """Embedding provider options"""
    JINA = "jina"
    LOCAL = "local"
    OPENAI = "openai"


class VectorStoreProvider(str, Enum):
    """Vector store provider options"""
    ASTRADB = "astradb"
    CHROMA = "chroma"
    PINECONE = "pinecone"


class DatabaseSettings(BaseSettings):
    """Database configuration settings"""
    
    # Database connection
    database_url: Optional[str] = Field(None, env="DATABASE_URL")
    database_host: str = Field("localhost", env="DATABASE_HOST")
    database_port: int = Field(5432, env="DATABASE_PORT")
    database_name: str = Field("app_db", env="DATABASE_NAME")
    database_user: str = Field("postgres", env="DATABASE_USER")
    database_password: str = Field("", env="DATABASE_PASSWORD")
    
    # Connection pool settings
    database_pool_size: int = Field(10, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(20, env="DATABASE_MAX_OVERFLOW")
    database_pool_timeout: int = Field(30, env="DATABASE_POOL_TIMEOUT")
    
    model_config = SettingsConfigDict(env_prefix="DB_", extra="ignore")


class LLMSettings(BaseSettings):
    """LLM provider configuration settings"""
    
    # OpenAI
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    openai_base_url: Optional[str] = Field(None, env="OPENAI_BASE_URL")
    openai_default_model: str = Field("gpt-4o", env="OPENAI_DEFAULT_MODEL")
    
    # OpenRouter
    openrouter_api_key: Optional[str] = Field(None, env="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field("https://openrouter.ai/api/v1", env="OPENROUTER_BASE_URL")
    
    # Google Gemini
    gemini_api_key: Optional[str] = Field(None, env="GEMINI_API_KEY")
    
    # Vertex AI
    vertexai_project: Optional[str] = Field(None, env="VERTEXAI_PROJECT")
    vertexai_location: Optional[str] = Field(None, env="VERTEXAI_LOCATION")
    google_application_credentials: Optional[str] = Field(None, env="GOOGLE_APPLICATION_CREDENTIALS")
    
    # AWS Bedrock
    aws_access_key_id: Optional[str] = Field(None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(None, env="AWS_SECRET_ACCESS_KEY")
    aws_session_token: Optional[str] = Field(None, env="AWS_SESSION_TOKEN")
    aws_region: str = Field("us-east-1", env="AWS_REGION")
    aws_bedrock_region: str = Field("us-east-1", env="AWS_BEDROCK_REGION")
    aws_bedrock_model: str = Field("bedrock/amazon.nova-pro-v1:0", env="AWS_BEDROCK_MODEL")
    aws_bearer_token_bedrock: Optional[str] = Field(None, env="AWS_BEARER_TOKEN_BEDROCK")
    
    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")


class RAGSettings(BaseSettings):
    """RAG system configuration settings"""
    
    # Core RAG settings
    rag_enabled: bool = Field(True, env="RAG_ENABLED")
    rag_use_enhanced_components: bool = Field(True, env="RAG_USE_ENHANCED_COMPONENTS")
    rag_enable_caching: bool = Field(True, env="RAG_ENABLE_CACHING")
    rag_enable_quality_monitoring: bool = Field(True, env="RAG_ENABLE_QUALITY_MONITORING")
    rag_enable_error_handling: bool = Field(True, env="RAG_ENABLE_ERROR_HANDLING")
    
    # Performance settings
    rag_cache_ttl: int = Field(3600, env="RAG_CACHE_TTL")
    rag_max_cache_size: int = Field(1000, env="RAG_MAX_CACHE_SIZE")
    rag_performance_threshold: float = Field(2.0, env="RAG_PERFORMANCE_THRESHOLD")
    rag_quality_threshold: float = Field(0.7, env="RAG_QUALITY_THRESHOLD")
    rag_default_k_value: int = Field(5, env="RAG_DEFAULT_K_VALUE")
    rag_max_retries: int = Field(5, env="RAG_MAX_RETRIES")
    
    # Embedding provider settings
    embedding_provider: EmbeddingProvider = Field(EmbeddingProvider.JINA, env="EMBEDDING_PROVIDER")
    
    # JINA settings
    jina_api_key: Optional[str] = Field(None, env="JINA_API_KEY")
    jina_embedding_model: str = Field("jina-embeddings-v3", env="JINA_EMBEDDING_MODEL")
    jina_api_url: str = Field("https://api.jina.ai/v1/embeddings", env="JINA_API_URL")
    embedding_dimension: int = Field(1024, env="EMBEDDING_DIMENSION")
    embedding_batch_size: int = Field(100, env="EMBEDDING_BATCH_SIZE")
    embedding_timeout: int = Field(30, env="EMBEDDING_TIMEOUT")
    jina_max_retries: int = Field(3, env="JINA_MAX_RETRIES")
    
    # Local embeddings settings
    local_embedding_model: str = Field("hf:ibm-granite/granite-embedding-30m-english", env="LOCAL_EMBEDDING_MODEL")
    local_embedding_device: str = Field("cpu", env="LOCAL_EMBEDDING_DEVICE")
    local_embedding_cache_dir: str = Field("models/embeddings", env="LOCAL_EMBEDDING_CACHE_DIR")
    
    # OpenAI embeddings settings
    openai_embedding_model: str = Field("text-embedding-3-large", env="OPENAI_EMBEDDING_MODEL")
    openai_embedding_dimension: int = Field(3072, env="OPENAI_EMBEDDING_DIMENSION")
    
    # Vector store settings
    vector_store_provider: VectorStoreProvider = Field(VectorStoreProvider.ASTRADB, env="VECTOR_STORE_PROVIDER")
    vector_store_collection: str = Field("manim_docs_jina_1024", env="VECTOR_STORE_COLLECTION")
    vector_store_distance_metric: str = Field("cosine", env="VECTOR_STORE_DISTANCE_METRIC")
    vector_store_max_results: int = Field(50, env="VECTOR_STORE_MAX_RESULTS")
    
    # AstraDB settings
    astradb_api_endpoint: Optional[str] = Field(None, env="ASTRADB_API_ENDPOINT")
    astradb_application_token: Optional[str] = Field(None, env="ASTRADB_APPLICATION_TOKEN")
    astradb_keyspace: str = Field("default_keyspace", env="ASTRADB_KEYSPACE")
    astradb_region: str = Field("us-east-2", env="ASTRADB_REGION")
    astradb_timeout: int = Field(30, env="ASTRADB_TIMEOUT")
    astradb_max_retries: int = Field(3, env="ASTRADB_MAX_RETRIES")
    
    # ChromaDB settings
    chroma_db_path: str = Field("data/rag/chroma_db", env="CHROMA_DB_PATH")
    chroma_collection_name: str = Field("manim_docs", env="CHROMA_COLLECTION_NAME")
    chroma_persist_directory: str = Field("data/rag/chroma_persist", env="CHROMA_PERSIST_DIRECTORY")
    
    # Document processing settings
    manim_docs_path: str = Field("data/rag/manim_docs", env="MANIM_DOCS_PATH")
    context_learning_path: str = Field("data/context_learning", env="CONTEXT_LEARNING_PATH")
    rag_docs_extensions: str = Field(".md,.txt,.py,.rst", env="RAG_DOCS_EXTENSIONS")
    rag_chunk_size: int = Field(1000, env="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(200, env="RAG_CHUNK_OVERLAP")
    rag_min_chunk_size: int = Field(100, env="RAG_MIN_CHUNK_SIZE")
    
    # Document processing flags
    rag_enable_preprocessing: bool = Field(True, env="RAG_ENABLE_PREPROCESSING")
    rag_remove_code_comments: bool = Field(False, env="RAG_REMOVE_CODE_COMMENTS")
    rag_extract_code_blocks: bool = Field(True, env="RAG_EXTRACT_CODE_BLOCKS")
    rag_normalize_whitespace: bool = Field(True, env="RAG_NORMALIZE_WHITESPACE")
    
    # Query processing settings
    rag_enable_query_expansion: bool = Field(True, env="RAG_ENABLE_QUERY_EXPANSION")
    rag_enable_semantic_search: bool = Field(True, env="RAG_ENABLE_SEMANTIC_SEARCH")
    rag_enable_hybrid_search: bool = Field(False, env="RAG_ENABLE_HYBRID_SEARCH")
    rag_query_expansion_models: int = Field(3, env="RAG_QUERY_EXPANSION_MODELS")
    rag_similarity_threshold: float = Field(0.7, env="RAG_SIMILARITY_THRESHOLD")
    
    # Context learning settings
    rag_use_context_learning: bool = Field(True, env="RAG_USE_CONTEXT_LEARNING")
    rag_context_window_size: int = Field(5, env="RAG_CONTEXT_WINDOW_SIZE")
    rag_context_overlap: int = Field(1, env="RAG_CONTEXT_OVERLAP")
    
    # Plugin detection settings
    rag_enable_plugin_detection: bool = Field(True, env="RAG_ENABLE_PLUGIN_DETECTION")
    rag_plugin_confidence_threshold: float = Field(0.8, env="RAG_PLUGIN_CONFIDENCE_THRESHOLD")
    rag_max_plugins_per_query: int = Field(5, env="RAG_MAX_PLUGINS_PER_QUERY")
    rag_plugin_cache_duration: int = Field(1800, env="RAG_PLUGIN_CACHE_DURATION")
    
    # Monitoring settings
    rag_enable_performance_monitoring: bool = Field(True, env="RAG_ENABLE_PERFORMANCE_MONITORING")
    rag_enable_usage_tracking: bool = Field(True, env="RAG_ENABLE_USAGE_TRACKING")
    rag_log_level: LogLevel = Field(LogLevel.INFO, env="RAG_LOG_LEVEL")
    rag_metrics_collection_interval: int = Field(300, env="RAG_METRICS_COLLECTION_INTERVAL")
    
    # Quality monitoring
    rag_enable_relevance_scoring: bool = Field(True, env="RAG_ENABLE_RELEVANCE_SCORING")
    rag_enable_feedback_collection: bool = Field(True, env="RAG_ENABLE_FEEDBACK_COLLECTION")
    rag_quality_sample_rate: float = Field(0.1, env="RAG_QUALITY_SAMPLE_RATE")
    
    @property
    def docs_extensions_list(self) -> List[str]:
        """Convert comma-separated extensions string to list"""
        return [ext.strip() for ext in self.rag_docs_extensions.split(",")]
    
    model_config = SettingsConfigDict(env_prefix="RAG_", extra="ignore")


class TTSSettings(BaseSettings):
    """Text-to-Speech configuration settings"""
    
    # Kokoro TTS settings
    kokoro_model_path: str = Field("models/kokoro-v0_19.onnx", env="KOKORO_MODEL_PATH")
    kokoro_voices_path: str = Field("models/voices.bin", env="KOKORO_VOICES_PATH")
    kokoro_default_voice: str = Field("af", env="KOKORO_DEFAULT_VOICE")
    kokoro_default_speed: str = Field("1.0", env="KOKORO_DEFAULT_SPEED")
    kokoro_default_lang: str = Field("en-us", env="KOKORO_DEFAULT_LANG")
    
    model_config = SettingsConfigDict(env_prefix="TTS_", extra="ignore")


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration settings"""
    
    # LangFuse settings
    langfuse_secret_key: Optional[str] = Field(None, env="LANGFUSE_SECRET_KEY")
    langfuse_public_key: Optional[str] = Field(None, env="LANGFUSE_PUBLIC_KEY")
    langfuse_host: str = Field("https://cloud.langfuse.com", env="LANGFUSE_HOST")
    langfuse_enabled: bool = Field(True, env="LANGFUSE_ENABLED")
    
    # General monitoring settings
    monitoring_enabled: bool = Field(True, env="MONITORING_ENABLED")
    performance_tracking: bool = Field(True, env="PERFORMANCE_TRACKING")
    error_tracking: bool = Field(True, env="ERROR_TRACKING")
    execution_tracing: bool = Field(True, env="EXECUTION_TRACING")
    
    model_config = SettingsConfigDict(env_prefix="MONITORING_", extra="ignore")


class SecuritySettings(BaseSettings):
    """Security configuration settings"""
    
    # JWT settings
    secret_key: str = Field("your-secret-key-change-in-production", env="SECRET_KEY")
    algorithm: str = Field("HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # CORS settings
    cors_origins: List[str] = Field(["*"], env="CORS_ORIGINS")
    cors_allow_credentials: bool = Field(True, env="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: List[str] = Field(["*"], env="CORS_ALLOW_METHODS")
    cors_allow_headers: List[str] = Field(["*"], env="CORS_ALLOW_HEADERS")
    
    # Rate limiting
    rate_limit_enabled: bool = Field(True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests_per_minute: int = Field(60, env="RATE_LIMIT_REQUESTS_PER_MINUTE")
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator('cors_allow_methods', pre=True)
    def parse_cors_methods(cls, v):
        if isinstance(v, str):
            return [method.strip() for method in v.split(",")]
        return v
    
    @validator('cors_allow_headers', pre=True)
    def parse_cors_headers(cls, v):
        if isinstance(v, str):
            return [header.strip() for header in v.split(",")]
        return v
    
    model_config = SettingsConfigDict(env_prefix="SECURITY_", extra="ignore")


class AppSettings(BaseSettings):
    """Main application configuration settings"""
    
    # Basic app settings
    app_name: str = Field("FastAPI Video Generation App", env="APP_NAME")
    app_version: str = Field("1.0.0", env="APP_VERSION")
    app_description: str = Field("FastAPI application for video generation with RAG and LangGraph", env="APP_DESCRIPTION")
    
    # Environment and debugging
    environment: Environment = Field(Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(True, env="DEBUG")
    log_level: LogLevel = Field(LogLevel.INFO, env="LOG_LEVEL")
    
    # Server settings
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    reload: bool = Field(True, env="RELOAD")
    workers: int = Field(1, env="WORKERS")
    
    # API settings
    api_v1_prefix: str = Field("/api/v1", env="API_V1_PREFIX")
    docs_url: Optional[str] = Field("/docs", env="DOCS_URL")
    redoc_url: Optional[str] = Field("/redoc", env="REDOC_URL")
    openapi_url: Optional[str] = Field("/openapi.json", env="OPENAPI_URL")
    
    # File upload settings
    max_upload_size: int = Field(100 * 1024 * 1024, env="MAX_UPLOAD_SIZE")  # 100MB
    upload_dir: str = Field("uploads", env="UPLOAD_DIR")
    
    # Workflow settings
    max_workflow_retries: int = Field(3, env="MAX_WORKFLOW_RETRIES")
    workflow_timeout_seconds: int = Field(3600, env="WORKFLOW_TIMEOUT_SECONDS")
    enable_checkpoints: bool = Field(True, env="ENABLE_CHECKPOINTS")
    checkpoint_interval: int = Field(300, env="CHECKPOINT_INTERVAL")
    
    # Human loop settings
    human_loop_enabled: bool = Field(True, env="HUMAN_LOOP_ENABLED")
    human_loop_timeout_seconds: int = Field(300, env="HUMAN_LOOP_TIMEOUT_SECONDS")
    auto_approve_low_risk: bool = Field(False, env="AUTO_APPROVE_LOW_RISK")
    
    @validator('environment', pre=True)
    def validate_environment(cls, v):
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    @validator('log_level', pre=True)
    def validate_log_level(cls, v):
        if isinstance(v, str):
            return LogLevel(v.upper())
        return v
    
    @property
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT
    
    @property
    def is_testing(self) -> bool:
        return self.environment == Environment.TESTING
    
    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="APP_"
    )


class Settings(BaseModel):
    """
    Main settings class that combines all configuration sections
    """
    
    # Core app settings
    app: AppSettings
    
    # Database settings
    database: DatabaseSettings
    
    # LLM provider settings
    llm: LLMSettings
    
    # RAG system settings
    rag: RAGSettings
    
    # TTS settings
    tts: TTSSettings
    
    # Monitoring settings
    monitoring: MonitoringSettings
    
    # Security settings
    security: SecuritySettings
    
    def __init__(self, **kwargs):
        # Initialize each section with its own settings
        app_settings = AppSettings()
        database_settings = DatabaseSettings()
        llm_settings = LLMSettings()
        rag_settings = RAGSettings()
        tts_settings = TTSSettings()
        monitoring_settings = MonitoringSettings()
        security_settings = SecuritySettings()
        
        super().__init__(
            app=app_settings,
            database=database_settings,
            llm=llm_settings,
            rag=rag_settings,
            tts=tts_settings,
            monitoring=monitoring_settings,
            security=security_settings,
            **kwargs
        )


# Global settings instance - will be initialized lazily
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Dependency function to get settings instance.
    This can be used with FastAPI's dependency injection system.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """
    Reload settings from environment variables and files.
    Useful for configuration hot-reloading.
    """
    global _settings
    _settings = Settings()
    return _settings
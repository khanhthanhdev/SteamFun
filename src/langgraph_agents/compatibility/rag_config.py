"""
Centralized RAG configuration manager that reads all settings from .env file.
This makes it easy for admins to configure the entire RAG system from one place.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class RAGSystemConfig:
    """Complete RAG system configuration loaded from .env file."""
    
    # Core RAG settings
    enabled: bool = True
    use_enhanced_components: bool = True
    enable_caching: bool = True
    enable_quality_monitoring: bool = True
    enable_error_handling: bool = True
    
    # Performance settings
    cache_ttl: int = 3600
    max_cache_size: int = 1000
    performance_threshold: float = 2.0
    quality_threshold: float = 0.7
    default_k_value: int = 5
    max_retries: int = 3
    
    # Embedding configuration
    embedding_provider: str = "jina"
    embedding_dimension: int = 1024
    embedding_batch_size: int = 100
    embedding_timeout: int = 30
    
    # JINA API settings
    jina_api_key: str = ""
    jina_model: str = "jina-embeddings-v3"
    jina_api_url: str = "https://api.jina.ai/v1/embeddings"
    jina_max_retries: int = 3
    
    # Local embedding settings
    local_model: str = "hf:ibm-granite/granite-embedding-30m-english"
    local_device: str = "cpu"
    local_cache_dir: str = "models/embeddings"
    
    # OpenAI embedding settings
    openai_model: str = "text-embedding-3-large"
    openai_dimension: int = 3072
    
    # Vector store configuration
    vector_store_provider: str = "astradb"
    vector_store_collection: str = "manim_docs_jina_1024"
    vector_store_distance_metric: str = "cosine"
    vector_store_max_results: int = 50
    
    # AstraDB settings
    astradb_api_endpoint: str = ""
    astradb_application_token: str = ""
    astradb_keyspace: str = "default_keyspace"
    astradb_region: str = "us-east-2"
    astradb_timeout: int = 30
    astradb_max_retries: int = 3
    
    # ChromaDB settings
    chroma_db_path: str = "data/rag/chroma_db"
    chroma_collection_name: str = "manim_docs"
    chroma_persist_directory: str = "data/rag/chroma_persist"
    
    # Document processing
    manim_docs_path: str = "data/rag/manim_docs"
    context_learning_path: str = "data/context_learning"
    docs_extensions: List[str] = field(default_factory=lambda: [".md", ".txt", ".py", ".rst"])
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    enable_preprocessing: bool = True
    remove_code_comments: bool = False
    extract_code_blocks: bool = True
    normalize_whitespace: bool = True
    
    # Query processing
    enable_query_expansion: bool = True
    enable_semantic_search: bool = True
    enable_hybrid_search: bool = False
    query_expansion_models: int = 3
    similarity_threshold: float = 0.7
    use_context_learning: bool = True
    context_window_size: int = 5
    context_overlap: int = 1
    
    # Plugin detection
    enable_plugin_detection: bool = True
    plugin_confidence_threshold: float = 0.8
    max_plugins_per_query: int = 5
    plugin_cache_duration: int = 1800
    
    # Monitoring and logging
    enable_performance_monitoring: bool = True
    enable_usage_tracking: bool = True
    log_level: str = "INFO"
    metrics_collection_interval: int = 300
    enable_relevance_scoring: bool = True
    enable_feedback_collection: bool = True
    quality_sample_rate: float = 0.1


class RAGConfigManager:
    """Manages RAG configuration loaded from .env file."""
    
    def __init__(self):
        """Initialize RAG configuration manager."""
        self._config = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load RAG configuration from environment variables."""
        try:
            # Parse document extensions
            docs_extensions = os.getenv('RAG_DOCS_EXTENSIONS', '.md,.txt,.py,.rst')
            extensions_list = [ext.strip() for ext in docs_extensions.split(',')]
            
            self._config = RAGSystemConfig(
                # Core RAG settings
                enabled=self._get_bool_env('RAG_ENABLED', True),
                use_enhanced_components=self._get_bool_env('RAG_USE_ENHANCED_COMPONENTS', True),
                enable_caching=self._get_bool_env('RAG_ENABLE_CACHING', True),
                enable_quality_monitoring=self._get_bool_env('RAG_ENABLE_QUALITY_MONITORING', True),
                enable_error_handling=self._get_bool_env('RAG_ENABLE_ERROR_HANDLING', True),
                
                # Performance settings
                cache_ttl=self._get_int_env('RAG_CACHE_TTL', 3600),
                max_cache_size=self._get_int_env('RAG_MAX_CACHE_SIZE', 1000),
                performance_threshold=self._get_float_env('RAG_PERFORMANCE_THRESHOLD', 2.0),
                quality_threshold=self._get_float_env('RAG_QUALITY_THRESHOLD', 0.7),
                default_k_value=self._get_int_env('RAG_DEFAULT_K_VALUE', 5),
                max_retries=self._get_int_env('RAG_MAX_RETRIES', 3),
                
                # Embedding configuration
                embedding_provider=os.getenv('EMBEDDING_PROVIDER', 'jina'),
                embedding_dimension=self._get_int_env('EMBEDDING_DIMENSION', 1024),
                embedding_batch_size=self._get_int_env('EMBEDDING_BATCH_SIZE', 100),
                embedding_timeout=self._get_int_env('EMBEDDING_TIMEOUT', 30),
                
                # JINA API settings
                jina_api_key=os.getenv('JINA_API_KEY', ''),
                jina_model=os.getenv('JINA_EMBEDDING_MODEL', 'jina-embeddings-v3'),
                jina_api_url=os.getenv('JINA_API_URL', 'https://api.jina.ai/v1/embeddings'),
                jina_max_retries=self._get_int_env('JINA_MAX_RETRIES', 3),
                
                # Local embedding settings
                local_model=os.getenv('LOCAL_EMBEDDING_MODEL', 'hf:ibm-granite/granite-embedding-30m-english'),
                local_device=os.getenv('LOCAL_EMBEDDING_DEVICE', 'cpu'),
                local_cache_dir=os.getenv('LOCAL_EMBEDDING_CACHE_DIR', 'models/embeddings'),
                
                # OpenAI embedding settings
                openai_model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-large'),
                openai_dimension=self._get_int_env('OPENAI_EMBEDDING_DIMENSION', 3072),
                
                # Vector store configuration
                vector_store_provider=os.getenv('VECTOR_STORE_PROVIDER', 'astradb'),
                vector_store_collection=os.getenv('VECTOR_STORE_COLLECTION', 'manim_docs_jina_1024'),
                vector_store_distance_metric=os.getenv('VECTOR_STORE_DISTANCE_METRIC', 'cosine'),
                vector_store_max_results=self._get_int_env('VECTOR_STORE_MAX_RESULTS', 50),
                
                # AstraDB settings
                astradb_api_endpoint=os.getenv('ASTRADB_API_ENDPOINT', ''),
                astradb_application_token=os.getenv('ASTRADB_APPLICATION_TOKEN', ''),
                astradb_keyspace=os.getenv('ASTRADB_KEYSPACE', 'default_keyspace'),
                astradb_region=os.getenv('ASTRADB_REGION', 'us-east-2'),
                astradb_timeout=self._get_int_env('ASTRADB_TIMEOUT', 30),
                astradb_max_retries=self._get_int_env('ASTRADB_MAX_RETRIES', 3),
                
                # ChromaDB settings
                chroma_db_path=os.getenv('CHROMA_DB_PATH', 'data/rag/chroma_db'),
                chroma_collection_name=os.getenv('CHROMA_COLLECTION_NAME', 'manim_docs'),
                chroma_persist_directory=os.getenv('CHROMA_PERSIST_DIRECTORY', 'data/rag/chroma_persist'),
                
                # Document processing
                manim_docs_path=os.getenv('MANIM_DOCS_PATH', 'data/rag/manim_docs'),
                context_learning_path=os.getenv('CONTEXT_LEARNING_PATH', 'data/context_learning'),
                docs_extensions=extensions_list,
                chunk_size=self._get_int_env('RAG_CHUNK_SIZE', 1000),
                chunk_overlap=self._get_int_env('RAG_CHUNK_OVERLAP', 200),
                min_chunk_size=self._get_int_env('RAG_MIN_CHUNK_SIZE', 100),
                enable_preprocessing=self._get_bool_env('RAG_ENABLE_PREPROCESSING', True),
                remove_code_comments=self._get_bool_env('RAG_REMOVE_CODE_COMMENTS', False),
                extract_code_blocks=self._get_bool_env('RAG_EXTRACT_CODE_BLOCKS', True),
                normalize_whitespace=self._get_bool_env('RAG_NORMALIZE_WHITESPACE', True),
                
                # Query processing
                enable_query_expansion=self._get_bool_env('RAG_ENABLE_QUERY_EXPANSION', True),
                enable_semantic_search=self._get_bool_env('RAG_ENABLE_SEMANTIC_SEARCH', True),
                enable_hybrid_search=self._get_bool_env('RAG_ENABLE_HYBRID_SEARCH', False),
                query_expansion_models=self._get_int_env('RAG_QUERY_EXPANSION_MODELS', 3),
                similarity_threshold=self._get_float_env('RAG_SIMILARITY_THRESHOLD', 0.7),
                use_context_learning=self._get_bool_env('RAG_USE_CONTEXT_LEARNING', True),
                context_window_size=self._get_int_env('RAG_CONTEXT_WINDOW_SIZE', 5),
                context_overlap=self._get_int_env('RAG_CONTEXT_OVERLAP', 1),
                
                # Plugin detection
                enable_plugin_detection=self._get_bool_env('RAG_ENABLE_PLUGIN_DETECTION', True),
                plugin_confidence_threshold=self._get_float_env('RAG_PLUGIN_CONFIDENCE_THRESHOLD', 0.8),
                max_plugins_per_query=self._get_int_env('RAG_MAX_PLUGINS_PER_QUERY', 5),
                plugin_cache_duration=self._get_int_env('RAG_PLUGIN_CACHE_DURATION', 1800),
                
                # Monitoring and logging
                enable_performance_monitoring=self._get_bool_env('RAG_ENABLE_PERFORMANCE_MONITORING', True),
                enable_usage_tracking=self._get_bool_env('RAG_ENABLE_USAGE_TRACKING', True),
                log_level=os.getenv('RAG_LOG_LEVEL', 'INFO'),
                metrics_collection_interval=self._get_int_env('RAG_METRICS_COLLECTION_INTERVAL', 300),
                enable_relevance_scoring=self._get_bool_env('RAG_ENABLE_RELEVANCE_SCORING', True),
                enable_feedback_collection=self._get_bool_env('RAG_ENABLE_FEEDBACK_COLLECTION', True),
                quality_sample_rate=self._get_float_env('RAG_QUALITY_SAMPLE_RATE', 0.1)
            )
            
            logger.info("RAG configuration loaded successfully from .env file")
            
        except Exception as e:
            logger.error(f"Failed to load RAG configuration: {e}")
            # Use default configuration
            self._config = RAGSystemConfig()
    
    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Get boolean environment variable."""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on', 'enabled')
    
    def _get_int_env(self, key: str, default: int) -> int:
        """Get integer environment variable."""
        try:
            return int(os.getenv(key, str(default)))
        except ValueError:
            logger.warning(f"Invalid integer value for {key}, using default: {default}")
            return default
    
    def _get_float_env(self, key: str, default: float) -> float:
        """Get float environment variable."""
        try:
            return float(os.getenv(key, str(default)))
        except ValueError:
            logger.warning(f"Invalid float value for {key}, using default: {default}")
            return default
    
    @property
    def config(self) -> RAGSystemConfig:
        """Get the current RAG configuration."""
        return self._config
    
    def reload_config(self) -> None:
        """Reload configuration from .env file."""
        logger.info("Reloading RAG configuration from .env file")
        self._load_config()
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding provider configuration."""
        config = self._config
        
        if config.embedding_provider == 'jina':
            return {
                'provider': 'jina',
                'api_key': config.jina_api_key,
                'model': config.jina_model,
                'api_url': config.jina_api_url,
                'dimension': config.embedding_dimension,
                'batch_size': config.embedding_batch_size,
                'timeout': config.embedding_timeout,
                'max_retries': config.jina_max_retries
            }
        elif config.embedding_provider == 'openai':
            return {
                'provider': 'openai',
                'api_key': os.getenv('OPENAI_API_KEY', ''),
                'model': config.openai_model,
                'dimension': config.openai_dimension,
                'batch_size': config.embedding_batch_size,
                'timeout': config.embedding_timeout,
                'max_retries': config.max_retries
            }
        else:  # local
            return {
                'provider': 'local',
                'model': config.local_model,
                'device': config.local_device,
                'cache_dir': config.local_cache_dir,
                'dimension': config.embedding_dimension,
                'batch_size': config.embedding_batch_size
            }
    
    def get_vector_store_config(self) -> Dict[str, Any]:
        """Get vector store configuration."""
        config = self._config
        
        if config.vector_store_provider == 'astradb':
            return {
                'provider': 'astradb',
                'api_endpoint': config.astradb_api_endpoint,
                'application_token': config.astradb_application_token,
                'keyspace': config.astradb_keyspace,
                'collection': config.vector_store_collection,
                'region': config.astradb_region,
                'dimension': config.embedding_dimension,
                'distance_metric': config.vector_store_distance_metric,
                'timeout': config.astradb_timeout,
                'max_retries': config.astradb_max_retries,
                'max_results': config.vector_store_max_results
            }
        else:  # chroma
            return {
                'provider': 'chroma',
                'db_path': config.chroma_db_path,
                'collection_name': config.chroma_collection_name,
                'persist_directory': config.chroma_persist_directory,
                'dimension': config.embedding_dimension,
                'distance_metric': config.vector_store_distance_metric,
                'max_results': config.vector_store_max_results
            }
    
    def get_document_processing_config(self) -> Dict[str, Any]:
        """Get document processing configuration."""
        config = self._config
        return {
            'manim_docs_path': config.manim_docs_path,
            'context_learning_path': config.context_learning_path,
            'docs_extensions': config.docs_extensions,
            'chunk_size': config.chunk_size,
            'chunk_overlap': config.chunk_overlap,
            'min_chunk_size': config.min_chunk_size,
            'enable_preprocessing': config.enable_preprocessing,
            'remove_code_comments': config.remove_code_comments,
            'extract_code_blocks': config.extract_code_blocks,
            'normalize_whitespace': config.normalize_whitespace
        }
    
    def get_query_processing_config(self) -> Dict[str, Any]:
        """Get query processing configuration."""
        config = self._config
        return {
            'enable_query_expansion': config.enable_query_expansion,
            'enable_semantic_search': config.enable_semantic_search,
            'enable_hybrid_search': config.enable_hybrid_search,
            'query_expansion_models': config.query_expansion_models,
            'similarity_threshold': config.similarity_threshold,
            'use_context_learning': config.use_context_learning,
            'context_window_size': config.context_window_size,
            'context_overlap': config.context_overlap,
            'default_k_value': config.default_k_value
        }
    
    def get_plugin_detection_config(self) -> Dict[str, Any]:
        """Get plugin detection configuration."""
        config = self._config
        return {
            'enable_plugin_detection': config.enable_plugin_detection,
            'confidence_threshold': config.plugin_confidence_threshold,
            'max_plugins_per_query': config.max_plugins_per_query,
            'cache_duration': config.plugin_cache_duration
        }
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring and logging configuration."""
        config = self._config
        return {
            'enable_performance_monitoring': config.enable_performance_monitoring,
            'enable_usage_tracking': config.enable_usage_tracking,
            'log_level': config.log_level,
            'metrics_collection_interval': config.metrics_collection_interval,
            'enable_relevance_scoring': config.enable_relevance_scoring,
            'enable_feedback_collection': config.enable_feedback_collection,
            'quality_sample_rate': config.quality_sample_rate
        }
    
    def is_rag_enabled(self) -> bool:
        """Check if RAG system is enabled."""
        return self._config.enabled
    
    def validate_configuration(self) -> List[str]:
        """Validate the current configuration and return any issues."""
        issues = []
        config = self._config
        
        # Validate embedding provider configuration
        if config.embedding_provider == 'jina':
            if not config.jina_api_key:
                issues.append("JINA API key is required when using JINA embedding provider")
        elif config.embedding_provider == 'openai':
            if not os.getenv('OPENAI_API_KEY'):
                issues.append("OpenAI API key is required when using OpenAI embedding provider")
        
        # Validate vector store configuration
        if config.vector_store_provider == 'astradb':
            if not config.astradb_api_endpoint:
                issues.append("AstraDB API endpoint is required when using AstraDB vector store")
            if not config.astradb_application_token:
                issues.append("AstraDB application token is required when using AstraDB vector store")
        
        # Validate paths
        if not Path(config.manim_docs_path).exists():
            issues.append(f"Manim docs path does not exist: {config.manim_docs_path}")
        if config.use_context_learning and not Path(config.context_learning_path).exists():
            issues.append(f"Context learning path does not exist: {config.context_learning_path}")
        
        # Validate numeric ranges
        if config.embedding_dimension <= 0:
            issues.append("Embedding dimension must be positive")
        if config.chunk_size <= 0:
            issues.append("Chunk size must be positive")
        if config.chunk_overlap >= config.chunk_size:
            issues.append("Chunk overlap must be less than chunk size")
        
        return issues
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration."""
        config = self._config
        return {
            'rag_enabled': config.enabled,
            'embedding_provider': config.embedding_provider,
            'vector_store_provider': config.vector_store_provider,
            'enhanced_components': config.use_enhanced_components,
            'caching_enabled': config.enable_caching,
            'quality_monitoring': config.enable_quality_monitoring,
            'plugin_detection': config.enable_plugin_detection,
            'context_learning': config.use_context_learning,
            'embedding_dimension': config.embedding_dimension,
            'chunk_size': config.chunk_size,
            'default_k_value': config.default_k_value,
            'configuration_issues': len(self.validate_configuration())
        }


# Global configuration manager instance
_rag_config_manager = None


def get_rag_config_manager() -> RAGConfigManager:
    """Get the global RAG configuration manager instance."""
    global _rag_config_manager
    if _rag_config_manager is None:
        _rag_config_manager = RAGConfigManager()
    return _rag_config_manager


def reload_rag_config() -> None:
    """Reload RAG configuration from .env file."""
    global _rag_config_manager
    if _rag_config_manager is not None:
        _rag_config_manager.reload_config()
    else:
        _rag_config_manager = RAGConfigManager()
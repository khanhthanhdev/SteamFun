"""
Backward-compatible RAG wrapper that uses the main RAG integration from src/rag/.
This wrapper provides compatibility with existing API while using centralized .env configuration.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

# Import the main RAG integration from src/rag/
try:
    from src.rag.rag_integration import RAGIntegration, RAGConfig
    from src.rag.vector_store import RAGVectorStore
    from src.rag.plugin_detection import ContextAwarePluginDetector
    RAG_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"RAG components not available: {e}")
    RAGIntegration = None
    RAGConfig = None
    RAGVectorStore = None
    ContextAwarePluginDetector = None
    RAG_COMPONENTS_AVAILABLE = False


logger = logging.getLogger(__name__)


class BackwardCompatibleRAGWrapper:
    """Backward-compatible RAG wrapper that uses the main RAG integration with .env configuration."""
    
    def __init__(self, 
                 helper_model: Any,
                 output_dir: str = "output",
                 session_id: Optional[str] = None,
                 use_langfuse: bool = True,
                 rag_params: Optional[Dict[str, Any]] = None):
        """Initialize backward-compatible RAG wrapper.
        
        Args:
            helper_model: Model for RAG queries
            output_dir: Output directory
            session_id: Session identifier
            use_langfuse: Whether to use Langfuse
            rag_params: RAG configuration parameters from .env
        """
        from .rag_config import get_rag_config_manager
        
        self.helper_model = helper_model
        self.output_dir = output_dir
        self.session_id = session_id
        self.use_langfuse = use_langfuse
        
        # Get centralized RAG configuration
        self.rag_config_manager = get_rag_config_manager()
        self.config = self.rag_config_manager.config
        self.rag_params = rag_params or {}
        
        # Initialize main RAG integration if available and enabled
        self.rag_integration = None
        
        if RAG_COMPONENTS_AVAILABLE and self.config.enabled and self.rag_params.get('use_rag', False):
            self._initialize_main_rag_integration()
        
        logger.info(f"BackwardCompatibleRAGWrapper initialized with priority: {self.rag_params.get('rag_priority', 'none')}")
    
    def _initialize_main_rag_integration(self):
        """Initialize the main RAG integration using centralized .env configuration."""
        try:
            # Create RAG configuration from .env settings
            rag_config = RAGConfig(
                use_enhanced_components=self.config.use_enhanced_components,
                enable_caching=self.config.enable_caching,
                enable_quality_monitoring=self.config.enable_quality_monitoring,
                enable_error_handling=self.config.enable_error_handling,
                cache_ttl=self.config.cache_ttl,
                max_cache_size=self.config.max_cache_size,
                performance_threshold=self.config.performance_threshold,
                quality_threshold=self.config.quality_threshold
            )
            
            # Get embedding model string based on .env configuration
            embedding_model = self._get_embedding_model_string()
            
            # Initialize the main RAG integration with centralized configuration
            from src.config.manager import ConfigurationManager
            config_manager = ConfigurationManager()
            
            self.rag_integration = RAGIntegration(
                helper_model=self.helper_model,
                output_dir=self.output_dir,
                chroma_db_path=self.config.chroma_db_path,
                manim_docs_path=self.config.manim_docs_path,
                embedding_model=embedding_model,
                use_langfuse=self.use_langfuse,
                session_id=self.session_id,
                config=rag_config,
                # Pass provider instances if using new provider system
                embedding_provider=self._create_embedding_provider(),
                vector_store_provider=self._create_vector_store_provider(),
                config_manager=config_manager
            )
            
            logger.info(f"Main RAG integration initialized successfully from .env configuration")
            logger.info(f"Embedding provider: {self.config.embedding_provider}")
            logger.info(f"Vector store provider: {self.config.vector_store_provider}")
            
        except Exception as e:
            logger.error(f"Failed to initialize main RAG integration: {e}")
            self.rag_integration = None
    
    def _get_embedding_model_string(self) -> str:
        """Get embedding model string based on .env configuration."""
        if self.config.embedding_provider == 'jina':
            return f"jina:{self.config.jina_model}"
        elif self.config.embedding_provider == 'openai':
            return f"openai:{self.config.openai_model}"
        else:
            return self.config.local_model
    
    def _create_embedding_provider(self):
        """Create embedding provider based on .env configuration."""
        try:
            # Try to use new provider system if available
            from src.rag.embedding_providers import EmbeddingProviderFactory, EmbeddingConfig
            
            if self.config.embedding_provider == 'jina':
                embedding_config = EmbeddingConfig(
                    provider='jina',
                    model_name=self.config.jina_model,
                    api_key=self.config.jina_api_key,
                    api_url=self.config.jina_api_url,
                    dimensions=self.config.embedding_dimension,
                    batch_size=self.config.embedding_batch_size,
                    timeout=self.config.embedding_timeout,
                    max_retries=self.config.jina_max_retries
                )
            elif self.config.embedding_provider == 'openai':
                embedding_config = EmbeddingConfig(
                    provider='openai',
                    model_name=self.config.openai_model,
                    api_key=os.getenv('OPENAI_API_KEY', ''),
                    dimensions=self.config.openai_dimension,
                    batch_size=self.config.embedding_batch_size,
                    timeout=self.config.embedding_timeout,
                    max_retries=self.config.max_retries
                )
            else:  # local
                embedding_config = EmbeddingConfig(
                    provider='local',
                    model_name=self.config.local_model,
                    device=self.config.local_device,
                    cache_dir=self.config.local_cache_dir,
                    dimensions=self.config.embedding_dimension,
                    batch_size=self.config.embedding_batch_size
                )
            
            return EmbeddingProviderFactory.create_provider(embedding_config)
            
        except ImportError:
            # New provider system not available, return None to use legacy
            logger.info("New provider system not available, using legacy embedding")
            return None
        except Exception as e:
            logger.warning(f"Failed to create embedding provider: {e}")
            return None
    
    def _create_vector_store_provider(self):
        """Create vector store provider based on .env configuration."""
        try:
            # Try to use new provider system if available
            from src.rag.vector_store_providers import VectorStoreFactory, VectorStoreConfig
            
            if self.config.vector_store_provider == 'astradb':
                vector_config = VectorStoreConfig(
                    provider='astradb',
                    api_endpoint=self.config.astradb_api_endpoint,
                    application_token=self.config.astradb_application_token,
                    keyspace=self.config.astradb_keyspace,
                    collection_name=self.config.vector_store_collection,
                    embedding_dimension=self.config.embedding_dimension,
                    distance_metric=self.config.vector_store_distance_metric,
                    timeout=self.config.astradb_timeout,
                    max_retries=self.config.astradb_max_retries,
                    max_results=self.config.vector_store_max_results
                )
            else:  # chroma
                vector_config = VectorStoreConfig(
                    provider='chroma',
                    db_path=self.config.chroma_db_path,
                    collection_name=self.config.chroma_collection_name,
                    persist_directory=self.config.chroma_persist_directory,
                    embedding_dimension=self.config.embedding_dimension,
                    distance_metric=self.config.vector_store_distance_metric,
                    max_results=self.config.vector_store_max_results
                )
            
            # Create embedding provider first
            embedding_provider = self._create_embedding_provider()
            return VectorStoreFactory.create_provider(vector_config, embedding_provider)
            
        except ImportError:
            # New provider system not available, return None to use legacy
            logger.info("New provider system not available, using legacy vector store")
            return None
        except Exception as e:
            logger.warning(f"Failed to create vector store provider: {e}")
            return None
    
    def query_rag(self, query: str, k: int = None, **kwargs) -> List[Dict[str, Any]]:
        """Query RAG system with backward compatibility using .env configuration.
        
        Args:
            query: Query string
            k: Number of results to return (uses .env default if not specified)
            **kwargs: Additional query parameters
            
        Returns:
            List[Dict]: RAG query results
        """
        if not self.rag_integration:
            logger.warning("RAG integration not available")
            return []
        
        try:
            # Use k from .env configuration if not specified
            effective_k = k if k is not None else self.config.default_k_value
            
            # Apply query processing configuration from .env
            query_config = self.rag_config_manager.get_query_processing_config()
            query_params = {
                'k': min(effective_k, self.config.vector_store_max_results),
                'similarity_threshold': query_config['similarity_threshold'],
                'enable_query_expansion': query_config['enable_query_expansion'],
                'enable_semantic_search': query_config['enable_semantic_search'],
                **kwargs
            }
            
            # Use the main RAG integration's enhanced retrieval if available
            if hasattr(self.rag_integration, 'get_enhanced_retrieval_results'):
                results = self.rag_integration.get_enhanced_retrieval_results(
                    queries=[query],
                    context=query_params
                )
            else:
                # Fallback to basic retrieval
                results = self.rag_integration.get_relevant_docs(
                    rag_queries=[query],
                    scene_trace_id=self.session_id,
                    topic=kwargs.get('topic', 'unknown'),
                    scene_number=kwargs.get('scene_number', 1)
                )
                # Convert to expected format
                results = [{'content': doc, 'score': 1.0, 'metadata': {}} for doc in results]
            
            return results
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return []
    
    def detect_relevant_plugins(self, topic: str, description: str) -> List[str]:
        """Detect relevant Manim plugins for the given topic using .env configuration.
        
        Args:
            topic: Video topic
            description: Video description
            
        Returns:
            List[str]: List of relevant plugin names
        """
        if not self.config.enable_plugin_detection:
            logger.info("Plugin detection disabled in .env configuration")
            return []
        
        if not self.rag_integration:
            logger.warning("RAG integration not available for plugin detection")
            return []
        
        try:
            # Use the main RAG integration's plugin detection
            plugins = self.rag_integration.detect_relevant_plugins(topic, description)
            
            # Apply .env configuration limits
            max_plugins = self.config.max_plugins_per_query
            filtered_plugins = plugins[:max_plugins] if plugins else []
            
            logger.info(f"Detected {len(filtered_plugins)} relevant plugins for topic: {topic}")
            return filtered_plugins
            
        except Exception as e:
            logger.error(f"Plugin detection failed: {e}")
            return []
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get RAG system status including .env configuration details.
        
        Returns:
            Dict: System status information
        """
        # Get configuration validation results
        validation_issues = self.rag_config_manager.validate_configuration()
        
        status = {
            'rag_enabled': self.config.enabled,
            'rag_integration_available': self.rag_integration is not None,
            'configuration_source': 'env_file',
            'configuration_valid': len(validation_issues) == 0,
            'configuration_issues': validation_issues,
            
            # Provider information from .env
            'embedding_provider': self.config.embedding_provider,
            'vector_store_provider': self.config.vector_store_provider,
            'rag_priority': self.rag_params.get('rag_priority', 'none'),
            
            # Feature status from .env
            'enhanced_components': self.config.use_enhanced_components,
            'caching_enabled': self.config.enable_caching,
            'quality_monitoring': self.config.enable_quality_monitoring,
            'plugin_detection': self.config.enable_plugin_detection,
            'context_learning': self.config.use_context_learning,
            'performance_monitoring': self.config.enable_performance_monitoring,
            
            # Configuration summary
            'config_summary': self.rag_config_manager.get_configuration_summary()
        }
        
        # Add detailed status from main RAG integration if available
        if self.rag_integration:
            try:
                integration_status = self.rag_integration.get_system_status()
                status.update({
                    'integration_status': integration_status,
                    'performance_metrics': self.rag_integration.get_performance_metrics()
                })
            except Exception as e:
                status['integration_error'] = str(e)
        
        return status
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get provider-specific status information from .env configuration.
        
        Returns:
            Dict: Provider status information
        """
        provider_status = {
            'active_embedding_provider': self.config.embedding_provider,
            'active_vector_store_provider': self.config.vector_store_provider,
            
            'jina_api': {
                'configured': bool(self.config.jina_api_key),
                'model': self.config.jina_model,
                'api_url': self.config.jina_api_url,
                'dimension': self.config.embedding_dimension,
                'batch_size': self.config.embedding_batch_size,
                'timeout': self.config.embedding_timeout,
                'max_retries': self.config.jina_max_retries
            },
            
            'openai_embedding': {
                'configured': bool(os.getenv('OPENAI_API_KEY')),
                'model': self.config.openai_model,
                'dimension': self.config.openai_dimension
            },
            
            'local_embedding': {
                'model': self.config.local_model,
                'device': self.config.local_device,
                'cache_dir': self.config.local_cache_dir
            },
            
            'astradb': {
                'configured': bool(self.config.astradb_api_endpoint and self.config.astradb_application_token),
                'endpoint': self.config.astradb_api_endpoint,
                'keyspace': self.config.astradb_keyspace,
                'collection': self.config.vector_store_collection,
                'region': self.config.astradb_region,
                'timeout': self.config.astradb_timeout,
                'max_retries': self.config.astradb_max_retries
            },
            
            'chroma': {
                'db_path': self.config.chroma_db_path,
                'collection_name': self.config.chroma_collection_name,
                'persist_directory': self.config.chroma_persist_directory
            }
        }
        
        # Add provider status from main RAG integration if available
        if self.rag_integration:
            try:
                main_provider_status = self.rag_integration.get_provider_status()
                provider_status['main_integration_providers'] = main_provider_status
            except Exception as e:
                provider_status['main_integration_error'] = str(e)
        
        return provider_status
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from RAG system.
        
        Returns:
            Dict: Performance metrics
        """
        if not self.rag_integration:
            return {'error': 'RAG integration not available'}
        
        try:
            return self.rag_integration.get_performance_metrics()
        except Exception as e:
            return {'error': str(e)}
    
    def switch_embedding_provider(self, provider_name: str) -> bool:
        """Switch embedding provider.
        
        Args:
            provider_name: Provider to switch to ('jina', 'local', 'openai')
            
        Returns:
            bool: True if switch was successful
        """
        if not self.rag_integration:
            logger.warning("RAG integration not available for provider switching")
            return False
        
        try:
            # Use main RAG integration's provider switching if available
            if hasattr(self.rag_integration, 'switch_embedding_provider'):
                return self.rag_integration.switch_embedding_provider(provider_name)
            else:
                logger.warning("Provider switching not supported by main RAG integration")
                return False
                
        except Exception as e:
            logger.error(f"Failed to switch embedding provider: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if RAG integration is available and functional.
        
        Returns:
            bool: True if RAG is available
        """
        return self.rag_integration is not None
    
    def reload_configuration(self) -> bool:
        """Reload configuration from .env file and reinitialize if needed.
        
        Returns:
            bool: True if reload was successful
        """
        try:
            # Reload configuration
            self.rag_config_manager.reload_config()
            self.config = self.rag_config_manager.config
            
            # Reinitialize RAG integration if enabled
            if self.config.enabled and self.rag_params.get('use_rag', False):
                self._initialize_main_rag_integration()
                logger.info("RAG configuration reloaded and integration reinitialized")
                return True
            else:
                self.rag_integration = None
                logger.info("RAG disabled after configuration reload")
                return True
                
        except Exception as e:
            logger.error(f"Failed to reload RAG configuration: {e}")
            return False
    
    # Delegate other methods to main RAG integration
    def __getattr__(self, name):
        """Delegate unknown method calls to the main RAG integration."""
        if self.rag_integration and hasattr(self.rag_integration, name):
            return getattr(self.rag_integration, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


def create_rag_integration(helper_model: Any, 
                          output_dir: str = "output",
                          session_id: Optional[str] = None,
                          use_langfuse: bool = True,
                          **rag_params) -> Optional[BackwardCompatibleRAGWrapper]:
    """Factory function to create RAG wrapper with JINA API and AstraDB priority.
    
    Args:
        helper_model: Model for RAG queries
        output_dir: Output directory
        session_id: Session identifier
        use_langfuse: Whether to use Langfuse
        **rag_params: RAG configuration parameters
        
    Returns:
        Optional[BackwardCompatibleRAGWrapper]: RAG wrapper instance or None
    """
    if not rag_params.get('use_rag', False):
        return None
    
    try:
        return BackwardCompatibleRAGWrapper(
            helper_model=helper_model,
            output_dir=output_dir,
            session_id=session_id,
            use_langfuse=use_langfuse,
            rag_params=rag_params
        )
    except Exception as e:
        logger.error(f"Failed to create RAG wrapper: {e}")
        return None
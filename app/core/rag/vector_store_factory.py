"""
Vector Store Factory

This module provides factory functionality for creating vector store providers
based on configuration. It handles provider selection, configuration validation,
and automatic fallback when preferred providers are unavailable.
"""

import os
import logging
from typing import Optional, List, Dict, Any, Type
from .vector_store_providers import (
    VectorStoreProvider, 
    VectorStoreConfig, 
    VectorStoreConfigurationError,
    VectorStoreConnectionError
)
from .embedding_providers import EmbeddingProvider
try:
    from src.config.manager import ConfigurationManager as CentralizedConfigManager
except ImportError:
    # Fallback when config manager is not available
    CentralizedConfigManager = None


class VectorStoreFactory:
    """Factory for creating vector store providers based on configuration."""
    
    # Registry of available vector store provider classes
    _provider_registry: Dict[str, Type[VectorStoreProvider]] = {}
    
    @classmethod
    def register_provider(cls, provider_name: str, provider_class: Type[VectorStoreProvider]):
        """Register a vector store provider class.
        
        Args:
            provider_name: Name of the provider (e.g., "chromadb", "astradb")
            provider_class: Provider class that implements VectorStoreProvider
        """
        cls._provider_registry[provider_name.lower()] = provider_class
        logging.info(f"Registered vector store provider: {provider_name}")
    
    @classmethod
    def create_provider(cls, 
                       config: Optional[VectorStoreConfig] = None,
                       embedding_provider: Optional[EmbeddingProvider] = None,
                       use_centralized_config: bool = True) -> VectorStoreProvider:
        """Create a vector store provider based on configuration.
        
        Args:
            config: VectorStoreConfig object. If None, loads from centralized config or environment
            embedding_provider: EmbeddingProvider instance. If None, creates default
            use_centralized_config: Whether to use centralized configuration manager
            
        Returns:
            VectorStoreProvider instance
            
        Raises:
            VectorStoreConfigurationError: If configuration is invalid
            VectorStoreConnectionError: If provider creation fails
        """
        if config is None:
            if use_centralized_config:
                try:
                    # Try to get configuration from centralized config manager
                    centralized_config = CentralizedConfigManager()
                    rag_config = centralized_config.get_rag_config()
                    
                    if rag_config and rag_config.enabled:
                        # Map provider names correctly
                        provider_name = rag_config.vector_store_config.provider
                        if provider_name == "chroma":
                            provider_name = "chromadb"
                        
                        # Create connection params based on provider
                        connection_params = rag_config.vector_store_config.connection_params.copy()
                        if provider_name == "chromadb" and 'path' not in connection_params:
                            connection_params['path'] = 'data/rag/chroma_db'
                        
                        config = VectorStoreConfig(
                            provider=provider_name,
                            collection_name=rag_config.vector_store_config.collection_name,
                            connection_params=connection_params,
                            distance_metric=rag_config.vector_store_config.distance_metric,
                            embedding_dimension=rag_config.embedding_config.dimensions
                        )
                        logging.info(f"Using centralized configuration for vector store provider: {config.provider}")
                    else:
                        logging.info("RAG disabled in centralized config, falling back to environment config")
                        config = cls._load_config_from_env()
                except Exception as e:
                    logging.warning(f"Failed to load centralized config, falling back to environment: {e}")
                    config = cls._load_config_from_env()
            else:
                config = cls._load_config_from_env()
        
        if embedding_provider is None:
            from .provider_factory import create_embedding_provider
            embedding_provider = create_embedding_provider(use_centralized_config=use_centralized_config)
        
        # Validate embedding dimension compatibility
        embedding_dim = embedding_provider.get_embedding_dimension()
        if config.embedding_dimension != embedding_dim:
            logging.warning(
                f"Embedding dimension mismatch: config specifies {config.embedding_dimension}, "
                f"but embedding provider produces {embedding_dim}. Updating config."
            )
            config.embedding_dimension = embedding_dim
        
        # Try to create the requested provider
        try:
            provider = cls._create_provider_instance(config, embedding_provider)
            
            # Test the provider availability
            if not provider.is_available():
                raise VectorStoreConnectionError(f"Provider {config.provider} is not available")
            
            # Initialize the provider
            provider.initialize()
            
            logging.info(f"Successfully created vector store provider: {config.provider}")
            return provider
            
        except Exception as e:
            logging.error(f"Failed to create vector store provider {config.provider}: {e}")
            
            # Try fallback if the primary provider failed
            if config.provider != "chromadb":
                logging.info("Attempting fallback to ChromaDB")
                fallback_config = cls._create_fallback_config(config)
                return cls.create_provider(fallback_config, embedding_provider, use_centralized_config=False)
            else:
                raise VectorStoreConnectionError(
                    f"Failed to create vector store provider and fallback failed: {e}"
                )
    
    @classmethod
    def _create_provider_instance(cls, 
                                 config: VectorStoreConfig, 
                                 embedding_provider: EmbeddingProvider) -> VectorStoreProvider:
        """Create a provider instance from configuration.
        
        Args:
            config: VectorStoreConfig object
            embedding_provider: EmbeddingProvider instance
            
        Returns:
            VectorStoreProvider instance
            
        Raises:
            VectorStoreConfigurationError: If provider is not registered
        """
        provider_name = config.provider.lower()
        
        if provider_name not in cls._provider_registry:
            available_providers = list(cls._provider_registry.keys())
            raise VectorStoreConfigurationError(
                f"Unknown vector store provider: {provider_name}. "
                f"Available providers: {available_providers}"
            )
        
        provider_class = cls._provider_registry[provider_name]
        return provider_class(config=config, embedding_provider=embedding_provider)
    
    @classmethod
    def _load_config_from_env(self) -> VectorStoreConfig:
        """Load vector store configuration from environment variables.
        
        Environment variables:
        - VECTOR_STORE_PROVIDER: Provider type ("chromadb" or "astradb")
        - VECTOR_STORE_COLLECTION: Collection name
        - VECTOR_STORE_DISTANCE_METRIC: Distance metric ("cosine", "euclidean", "dot_product")
        
        ChromaDB specific:
        - CHROMADB_PATH: Path to ChromaDB database
        - CHROMADB_HOST: ChromaDB server host (for client mode)
        - CHROMADB_PORT: ChromaDB server port (for client mode)
        
        AstraDB specific:
        - ASTRADB_API_ENDPOINT: AstraDB API endpoint
        - ASTRADB_APPLICATION_TOKEN: AstraDB application token
        - ASTRADB_KEYSPACE: AstraDB keyspace (optional)
        
        Returns:
            VectorStoreConfig object with loaded configuration
            
        Raises:
            VectorStoreConfigurationError: If configuration is invalid
        """
        provider = os.getenv('VECTOR_STORE_PROVIDER', 'chromadb').lower()
        collection_name = os.getenv('VECTOR_STORE_COLLECTION', 'default_collection')
        distance_metric = os.getenv('VECTOR_STORE_DISTANCE_METRIC', 'cosine').lower()
        
        # Load provider-specific connection parameters
        if provider == "chromadb":
            connection_params = self._load_chromadb_config()
        elif provider == "astradb":
            connection_params = self._load_astradb_config()
        else:
            logging.warning(
                f"Unknown vector store provider: {provider}. "
                f"Supported providers: chromadb, astradb. Falling back to chromadb."
            )
            provider = "chromadb"
            connection_params = self._load_chromadb_config()
        
        # Get embedding dimension from embedding provider
        from .provider_factory import create_embedding_provider
        embedding_provider = create_embedding_provider()
        embedding_dimension = embedding_provider.get_embedding_dimension()
        
        # Check for embedding dimension override
        dimension_override = os.getenv('VECTOR_STORE_EMBEDDING_DIMENSIONS')
        if dimension_override:
            try:
                embedding_dimension = int(dimension_override)
                logging.info(f"Using embedding dimension override: {embedding_dimension} (provider default: {embedding_provider.get_embedding_dimension()})")
                
                # Validate dimension override
                if embedding_dimension <= 0:
                    raise ValueError("Dimensions must be positive")
                elif embedding_dimension > 4096:
                    logging.warning(f"Very high dimension count: {embedding_dimension}. This may impact performance.")
                
            except ValueError as e:
                logging.error(f"Invalid VECTOR_STORE_EMBEDDING_DIMENSIONS value '{dimension_override}': {e}. Using provider default.")
                embedding_dimension = embedding_provider.get_embedding_dimension()
        else:
            embedding_dimension = embedding_provider.get_embedding_dimension()
        
        config = VectorStoreConfig(
            provider=provider,
            connection_params=connection_params,
            collection_name=collection_name,
            embedding_dimension=embedding_dimension,
            distance_metric=distance_metric
        )
        
        # Validate the configuration
        self._validate_environment_config(config)
        return config
    
    @classmethod
    def _load_chromadb_config(cls) -> Dict[str, Any]:
        """Load ChromaDB-specific configuration."""
        config = {}
        
        # Check if we should use client mode or persistent mode
        host = os.getenv('CHROMADB_HOST')
        port = os.getenv('CHROMADB_PORT')
        
        if host and port:
            # Client mode
            config['mode'] = 'client'
            config['host'] = host
            config['port'] = int(port)
            
            # Optional SSL configuration
            config['ssl'] = os.getenv('CHROMADB_SSL', 'false').lower() == 'true'
            
            # Optional authentication
            config['headers'] = {}
            auth_token = os.getenv('CHROMADB_AUTH_TOKEN')
            if auth_token:
                config['headers']['Authorization'] = f"Bearer {auth_token}"
        else:
            # Persistent mode (default)
            config['mode'] = 'persistent'
            config['path'] = os.getenv('CHROMADB_PATH', './chroma_db')
        
        # Common ChromaDB settings
        config['anonymized_telemetry'] = os.getenv('CHROMADB_TELEMETRY', 'false').lower() == 'true'
        
        return config
    
    @classmethod
    def _load_astradb_config(cls) -> Dict[str, Any]:
        """Load AstraDB-specific configuration.
        
        Raises:
            VectorStoreConfigurationError: If required AstraDB configuration is missing
        """
        api_endpoint = os.getenv('ASTRADB_API_ENDPOINT')
        application_token = os.getenv('ASTRADB_APPLICATION_TOKEN')
        
        if not api_endpoint:
            raise VectorStoreConfigurationError(
                "ASTRADB_API_ENDPOINT is required for AstraDB provider. "
                "Get your endpoint from the AstraDB console."
            )
        
        if not application_token:
            raise VectorStoreConfigurationError(
                "ASTRADB_APPLICATION_TOKEN is required for AstraDB provider. "
                "Generate a token in the AstraDB console."
            )
        
        config = {
            'api_endpoint': api_endpoint,
            'application_token': application_token
        }
        
        # Optional keyspace
        keyspace = os.getenv('ASTRADB_KEYSPACE')
        if keyspace:
            config['keyspace'] = keyspace
        
        # Optional region
        region = os.getenv('ASTRADB_REGION')
        if region:
            config['region'] = region
        
        return config
    
    @classmethod
    def _validate_environment_config(cls, config: VectorStoreConfig) -> None:
        """Validate configuration loaded from environment.
        
        Args:
            config: VectorStoreConfig to validate
            
        Raises:
            VectorStoreConfigurationError: If configuration is invalid
        """
        # Validate distance metric
        valid_metrics = ["cosine", "euclidean", "dot_product"]
        if config.distance_metric not in valid_metrics:
            raise VectorStoreConfigurationError(
                f"Invalid distance metric: {config.distance_metric}. "
                f"Valid options: {valid_metrics}"
            )
        
        # Provider-specific validation
        if config.provider == "chromadb":
            cls._validate_chromadb_config(config.connection_params)
        elif config.provider == "astradb":
            cls._validate_astradb_config(config.connection_params)
    
    @classmethod
    def _validate_chromadb_config(cls, connection_params: Dict[str, Any]) -> None:
        """Validate ChromaDB configuration."""
        mode = connection_params.get('mode', 'persistent')
        
        if mode == 'client':
            if 'host' not in connection_params or 'port' not in connection_params:
                raise VectorStoreConfigurationError(
                    "ChromaDB client mode requires both CHROMADB_HOST and CHROMADB_PORT"
                )
            
            try:
                port = int(connection_params['port'])
                if port <= 0 or port > 65535:
                    raise ValueError()
            except (ValueError, TypeError):
                raise VectorStoreConfigurationError(
                    f"Invalid ChromaDB port: {connection_params.get('port')}"
                )
        
        elif mode == 'persistent':
            path = connection_params.get('path')
            if not path:
                raise VectorStoreConfigurationError(
                    "ChromaDB persistent mode requires CHROMADB_PATH"
                )
    
    @classmethod
    def _validate_astradb_config(cls, connection_params: Dict[str, Any]) -> None:
        """Validate AstraDB configuration."""
        required_params = ['api_endpoint', 'application_token']
        
        for param in required_params:
            if param not in connection_params or not connection_params[param]:
                env_var = f"ASTRADB_{param.upper()}"
                raise VectorStoreConfigurationError(
                    f"AstraDB requires {env_var} to be set"
                )
        
        # Validate API endpoint format
        api_endpoint = connection_params['api_endpoint']
        if not api_endpoint.startswith('https://'):
            raise VectorStoreConfigurationError(
                f"AstraDB API endpoint must start with https://, got: {api_endpoint}"
            )
    
    @classmethod
    def _create_fallback_config(cls, original_config: VectorStoreConfig) -> VectorStoreConfig:
        """Create fallback configuration when primary provider fails.
        
        Args:
            original_config: The original configuration that failed
            
        Returns:
            Fallback VectorStoreConfig (typically ChromaDB)
        """
        if original_config.provider != "chromadb":
            logging.warning(
                f"Falling back from {original_config.provider} to ChromaDB"
            )
            
            fallback_connection_params = cls._load_chromadb_config()
            
            return VectorStoreConfig(
                provider="chromadb",
                connection_params=fallback_connection_params,
                collection_name=original_config.collection_name,
                embedding_dimension=original_config.embedding_dimension,
                distance_metric=original_config.distance_metric
            )
        else:
            # If ChromaDB also fails, we have a serious problem
            raise VectorStoreConfigurationError(
                "ChromaDB vector store is not available. "
                "Please check your ChromaDB configuration."
            )
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available vector store providers.
        
        Returns:
            List of available provider names
        """
        available = []
        
        for provider_name in cls._provider_registry.keys():
            try:
                # Try to create a test configuration
                if provider_name == "chromadb":
                    # ChromaDB is usually available if the package is installed
                    available.append(provider_name)
                elif provider_name == "astradb":
                    # AstraDB requires credentials
                    if (os.getenv('ASTRADB_API_ENDPOINT') and 
                        os.getenv('ASTRADB_APPLICATION_TOKEN')):
                        available.append(provider_name)
            except Exception:
                # Provider not available
                continue
        
        return available
    
    @classmethod
    def test_provider(cls, provider_name: str) -> Dict[str, Any]:
        """Test a specific vector store provider.
        
        Args:
            provider_name: Name of the provider to test
            
        Returns:
            Dictionary with test results
        """
        test_result = {
            "provider": provider_name,
            "available": False,
            "error": None,
            "response_time": None,
            "features": []
        }
        
        try:
            import time
            start_time = time.time()
            
            # Create test configuration
            if provider_name == "chromadb":
                connection_params = cls._load_chromadb_config()
            elif provider_name == "astradb":
                connection_params = cls._load_astradb_config()
            else:
                raise VectorStoreConfigurationError(f"Unknown provider: {provider_name}")
            
            config = VectorStoreConfig(
                provider=provider_name,
                connection_params=connection_params,
                collection_name="test_collection",
                embedding_dimension=384,  # Default dimension
                distance_metric="cosine"
            )
            
            # Create embedding provider for testing
            from .provider_factory import create_embedding_provider
            embedding_provider = create_embedding_provider()
            
            # Try to create the provider
            provider = cls._create_provider_instance(config, embedding_provider)
            
            # Test availability
            if provider.is_available():
                test_result["available"] = True
                test_result["response_time"] = (time.time() - start_time) * 1000  # ms
                
                # Test health check
                health = provider.health_check()
                test_result["health"] = health
                
                # Determine supported features
                features = ["similarity_search"]
                try:
                    # Test if hybrid search is supported
                    provider.hybrid_search("test", k=1)
                    features.append("hybrid_search")
                except (NotImplementedError, AttributeError):
                    pass
                
                test_result["features"] = features
            
        except Exception as e:
            test_result["error"] = str(e)
        
        return test_result
    
    @classmethod
    def test_all_providers(cls) -> Dict[str, Dict[str, Any]]:
        """Test all registered vector store providers.
        
        Returns:
            Dictionary mapping provider names to test results
        """
        results = {}
        
        for provider_name in cls._provider_registry.keys():
            results[provider_name] = cls.test_provider(provider_name)
        
        return results
    
    @classmethod
    def get_provider_info(cls) -> Dict[str, Any]:
        """Get information about the factory and registered providers.
        
        Returns:
            Dictionary with factory information
        """
        return {
            "registered_providers": list(cls._provider_registry.keys()),
            "available_providers": cls.get_available_providers(),
            "default_provider": "chromadb",
            "supported_distance_metrics": ["cosine", "euclidean", "dot_product"],
            "environment_variables": {
                "VECTOR_STORE_PROVIDER": "Provider type (chromadb, astradb)",
                "VECTOR_STORE_COLLECTION": "Collection name",
                "VECTOR_STORE_DISTANCE_METRIC": "Distance metric",
                "CHROMADB_PATH": "ChromaDB database path",
                "CHROMADB_HOST": "ChromaDB server host",
                "CHROMADB_PORT": "ChromaDB server port",
                "ASTRADB_API_ENDPOINT": "AstraDB API endpoint",
                "ASTRADB_APPLICATION_TOKEN": "AstraDB application token"
            }
        }
    
    @classmethod
    def validate_environment_variables(cls) -> Dict[str, str]:
        """Validate all vector store related environment variables.
        
        Returns:
            Dictionary with validation results and recommendations
        """
        validation_results = {}
        
        # Check provider setting
        provider = os.getenv('VECTOR_STORE_PROVIDER', 'chromadb').lower()
        if provider not in ['chromadb', 'astradb']:
            validation_results['VECTOR_STORE_PROVIDER'] = (
                f"Invalid provider '{provider}'. Use 'chromadb' or 'astradb'"
            )
        
        # Check distance metric
        distance_metric = os.getenv('VECTOR_STORE_DISTANCE_METRIC', 'cosine').lower()
        valid_metrics = ['cosine', 'euclidean', 'dot_product']
        if distance_metric not in valid_metrics:
            validation_results['VECTOR_STORE_DISTANCE_METRIC'] = (
                f"Invalid distance metric '{distance_metric}'. Use: {valid_metrics}"
            )
        
        # Provider-specific validation
        if provider == 'astradb':
            if not os.getenv('ASTRADB_API_ENDPOINT'):
                validation_results['ASTRADB_API_ENDPOINT'] = (
                    "Required for AstraDB provider. Get from AstraDB console."
                )
            
            if not os.getenv('ASTRADB_APPLICATION_TOKEN'):
                validation_results['ASTRADB_APPLICATION_TOKEN'] = (
                    "Required for AstraDB provider. Generate in AstraDB console."
                )
        
        elif provider == 'chromadb':
            # Check ChromaDB configuration
            host = os.getenv('CHROMADB_HOST')
            port = os.getenv('CHROMADB_PORT')
            
            if host and not port:
                validation_results['CHROMADB_PORT'] = (
                    "Required when CHROMADB_HOST is specified for client mode"
                )
            elif port and not host:
                validation_results['CHROMADB_HOST'] = (
                    "Required when CHROMADB_PORT is specified for client mode"
                )
            elif port:
                try:
                    port_int = int(port)
                    if port_int <= 0 or port_int > 65535:
                        raise ValueError()
                except ValueError:
                    validation_results['CHROMADB_PORT'] = (
                        f"Must be valid port number (1-65535), got: {port}"
                    )
        
        return validation_results


# Convenience functions for common operations
def create_vector_store(config: Optional[VectorStoreConfig] = None,
                       embedding_provider: Optional[EmbeddingProvider] = None,
                       use_centralized_config: bool = True) -> VectorStoreProvider:
    """Create a vector store provider using the factory.
    
    Args:
        config: Optional VectorStoreConfig. If None, loads from centralized config or environment
        embedding_provider: Optional EmbeddingProvider. If None, creates default
        use_centralized_config: Whether to use centralized configuration manager
        
    Returns:
        VectorStoreProvider instance
    """
    return VectorStoreFactory.create_provider(config, embedding_provider, use_centralized_config)


def get_default_vector_store(embedding_provider: Optional[EmbeddingProvider] = None,
                            use_centralized_config: bool = True) -> VectorStoreProvider:
    """Get the default vector store provider based on centralized or environment configuration.
    
    Args:
        embedding_provider: Optional EmbeddingProvider. If None, creates default
        use_centralized_config: Whether to use centralized configuration manager
        
    Returns:
        VectorStoreProvider instance
    """
    return VectorStoreFactory.create_provider(None, embedding_provider, use_centralized_config)


def test_vector_store_providers() -> Dict[str, Dict[str, Any]]:
    """Test all available vector store providers.
    
    Returns:
        Dictionary mapping provider names to test results
    """
    return VectorStoreFactory.test_all_providers()

#Register available vector store providers
def _register_default_providers():
    """Register default vector store providers."""
    try:
        # Register ChromaDB provider (if available)
        from .vector_store import EnhancedRAGVectorStore
        VectorStoreFactory.register_provider("chromadb", EnhancedRAGVectorStore)
    except ImportError:
        logging.warning("ChromaDB provider not available")
    
    try:
        # Register AstraDB provider (if available)
        from .astradb_vector_store import AstraDBWithFallback
        VectorStoreFactory.register_provider("astradb", AstraDBWithFallback)
    except ImportError:
        logging.warning("AstraDB provider not available")


# Register providers when module is imported
_register_default_providers()
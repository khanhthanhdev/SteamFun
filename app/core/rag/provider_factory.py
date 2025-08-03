"""
Embedding Provider Factory

This module provides factory functions to create embedding providers
based on configuration, with automatic fallback handling and centralized configuration support.
"""

import logging
import time
from typing import Optional

from .embedding_providers import (
    EmbeddingProvider,
    EmbeddingConfig,
    ConfigurationManager,
    EmbeddingGenerationError,
    ProviderConfigurationError
)
from .jina_embedding_provider import JinaEmbeddingProvider
try:
    from src.config.manager import ConfigurationManager as CentralizedConfigManager
except ImportError:
    # Fallback when config manager is not available
    CentralizedConfigManager = None


class EmbeddingProviderFactory:
    """Factory for creating embedding providers with fallback support."""
    
    @staticmethod
    def create_provider(config: Optional[EmbeddingConfig] = None, 
                       enable_fallback: bool = True,
                       use_centralized_config: bool = True) -> EmbeddingProvider:
        """Create an embedding provider based on configuration with optional fallback support.
        
        Args:
            config: Optional EmbeddingConfig. If None, loads from centralized config or environment.
            enable_fallback: Whether to enable automatic fallback for JINA provider
            use_centralized_config: Whether to use centralized configuration manager
            
        Returns:
            EmbeddingProvider instance
            
        Raises:
            ProviderConfigurationError: If no valid provider can be created
        """
        logger = logging.getLogger(__name__)
        
        # Load config from centralized configuration or environment if not provided
        if config is None:
            if use_centralized_config:
                try:
                    # Try to get configuration from centralized config manager
                    centralized_config = CentralizedConfigManager()
                    rag_config = centralized_config.get_rag_config()
                    
                    if rag_config and rag_config.enabled:
                        config = EmbeddingConfig(
                            provider=rag_config.embedding_config.provider,
                            model_name=rag_config.embedding_config.model_name,
                            api_key=rag_config.embedding_config.api_key,
                            dimensions=rag_config.embedding_config.dimensions,
                            batch_size=rag_config.embedding_config.batch_size,
                            timeout=rag_config.embedding_config.timeout
                        )
                        logger.info(f"Using centralized configuration for embedding provider: {config.provider}")
                    else:
                        logger.info("RAG disabled in centralized config, falling back to environment config")
                        config = ConfigurationManager.load_config_from_env()
                except Exception as e:
                    logger.warning(f"Failed to load centralized config, falling back to environment: {e}")
                    config = ConfigurationManager.load_config_from_env()
            else:
                config = ConfigurationManager.load_config_from_env()
        
        # Try to create the requested provider
        try:
            if config.provider == "jina":
                # Create fallback provider if enabled
                fallback_provider = None
                if enable_fallback:
                    try:
                        fallback_config = ConfigurationManager._load_local_config()
                        from .local_embedding_provider import LocalEmbeddingProvider
                        fallback_provider = LocalEmbeddingProvider(fallback_config)
                        logger.info("Created local fallback provider for JINA")
                    except Exception as e:
                        logger.warning(f"Could not create fallback provider: {e}")
                
                provider = JinaEmbeddingProvider(config, fallback_provider)
                logger.info(f"Created JINA embedding provider with model: {config.model_name}")
                if fallback_provider:
                    logger.info("JINA provider configured with local fallback")
                return provider
                
            elif config.provider == "local":
                # Import local provider only when needed to avoid circular imports
                from .local_embedding_provider import LocalEmbeddingProvider
                provider = LocalEmbeddingProvider(config)
                logger.info(f"Created local embedding provider with model: {config.model_name}")
                return provider
            elif config.provider == "openai":
                # Import OpenAI provider if available
                try:
                    from .openai_embedding_provider import OpenAIEmbeddingProvider
                    provider = OpenAIEmbeddingProvider(config)
                    logger.info(f"Created OpenAI embedding provider with model: {config.model_name}")
                    return provider
                except ImportError:
                    logger.warning("OpenAI embedding provider not available, falling back")
                    raise ProviderConfigurationError("OpenAI embedding provider not available")
            elif config.provider == "gemini":
                # Import Gemini provider if available
                try:
                    from .gemini_embedding_provider import GeminiEmbeddingProvider
                    provider = GeminiEmbeddingProvider(config)
                    logger.info(f"Created Gemini embedding provider with model: {config.model_name}")
                    return provider
                except ImportError:
                    logger.warning("Gemini embedding provider not available, falling back")
                    raise ProviderConfigurationError("Gemini embedding provider not available")
            else:
                raise ProviderConfigurationError(
                    f"Unknown provider: {config.provider}. Supported: jina, local, openai, gemini"
                )
                
        except (EmbeddingGenerationError, ProviderConfigurationError) as e:
            logger.warning(f"Failed to create {config.provider} provider: {e}")
            
            # Try fallback to local provider if original wasn't local and fallback is enabled
            if config.provider != "local" and enable_fallback:
                try:
                    fallback_config = ConfigurationManager.create_fallback_config(config)
                    from .local_embedding_provider import LocalEmbeddingProvider
                    provider = LocalEmbeddingProvider(fallback_config)
                    logger.info(f"Successfully created fallback local provider")
                    return provider
                except Exception as fallback_error:
                    logger.error(f"Fallback provider creation failed: {fallback_error}")
            
            # If we get here, both primary and fallback failed
            raise ProviderConfigurationError(
                f"Failed to create embedding provider. Primary error: {e}"
            )
    
    @staticmethod
    def create_jina_provider(api_key: str, model_name: str = "jina-embeddings-v3", 
                           enable_fallback: bool = True) -> JinaEmbeddingProvider:
        """Create a JINA embedding provider with specific parameters and optional fallback.
        
        Args:
            api_key: JINA API key
            model_name: JINA model name
            enable_fallback: Whether to enable fallback to local embeddings
            
        Returns:
            JinaEmbeddingProvider instance
        """
        logger = logging.getLogger(__name__)
        
        # Determine dimensions based on model
        model_dimensions = {
            "jina-embeddings-v3": 1024,
            "jina-embeddings-v4": 2048,
            "jina-clip-v2": 1024
        }
        dimensions = model_dimensions.get(model_name, 1024)
        
        config = EmbeddingConfig(
            provider="jina",
            model_name=model_name,
            dimensions=dimensions,
            api_key=api_key,
            batch_size=100,
            timeout=30
        )
        
        # Create fallback provider if enabled
        fallback_provider = None
        if enable_fallback:
            try:
                fallback_config = ConfigurationManager._load_local_config()
                from .local_embedding_provider import LocalEmbeddingProvider
                fallback_provider = LocalEmbeddingProvider(fallback_config)
                logger.info("Created local fallback provider for JINA")
            except Exception as e:
                logger.warning(f"Could not create fallback provider: {e}")
        
        return JinaEmbeddingProvider(config, fallback_provider)
    
    @staticmethod
    def get_available_providers() -> dict:
        """Get information about available providers.
        
        Returns:
            Dictionary with provider availability and info
        """
        providers = {}
        
        # Check JINA availability
        try:
            jina_config = ConfigurationManager._load_jina_config()
            ConfigurationManager.validate_config(jina_config)
            providers["jina"] = {
                "available": True,
                "model": jina_config.model_name,
                "dimensions": jina_config.dimensions,
                "requires_api_key": True
            }
        except ProviderConfigurationError:
            providers["jina"] = {
                "available": False,
                "reason": "API key not configured or invalid",
                "requires_api_key": True
            }
        
        # Check local availability
        try:
            local_config = ConfigurationManager._load_local_config()
            ConfigurationManager.validate_config(local_config)
            providers["local"] = {
                "available": True,
                "model": local_config.model_name,
                "dimensions": local_config.dimensions,
                "requires_api_key": False
            }
        except ProviderConfigurationError as e:
            providers["local"] = {
                "available": False,
                "reason": str(e),
                "requires_api_key": False
            }
        
        return providers
    
    @staticmethod
    def test_provider(provider_name: str) -> dict:
        """Test a specific provider's functionality.
        
        Args:
            provider_name: Name of provider to test ("jina" or "local")
            
        Returns:
            Dictionary with test results
        """
        logger = logging.getLogger(__name__)
        
        try:
            # Create config for the specific provider
            if provider_name == "jina":
                config = ConfigurationManager._load_jina_config()
            elif provider_name == "local":
                config = ConfigurationManager._load_local_config()
            else:
                return {
                    "success": False,
                    "error": f"Unknown provider: {provider_name}",
                    "provider": provider_name
                }
            
            # Create provider
            provider = EmbeddingProviderFactory.create_provider(config)
            
            # Test basic functionality
            test_texts = ["Hello world", "This is a test"]
            start_time = time.time()
            embeddings = provider.generate_embeddings(test_texts)
            end_time = time.time()
            
            # Validate results
            if len(embeddings) != len(test_texts):
                return {
                    "success": False,
                    "error": f"Expected {len(test_texts)} embeddings, got {len(embeddings)}",
                    "provider": provider_name
                }
            
            expected_dim = provider.get_embedding_dimension()
            for i, embedding in enumerate(embeddings):
                if len(embedding) != expected_dim:
                    return {
                        "success": False,
                        "error": f"Embedding {i} has dimension {len(embedding)}, expected {expected_dim}",
                        "provider": provider_name
                    }
            
            return {
                "success": True,
                "provider": provider_name,
                "model": config.model_name,
                "dimensions": expected_dim,
                "test_duration": end_time - start_time,
                "embeddings_generated": len(embeddings),
                "provider_info": provider.get_provider_info()
            }
            
        except Exception as e:
            logger.error(f"Provider test failed for {provider_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "provider": provider_name
            }


# Convenience functions
def create_embedding_provider(config: Optional[EmbeddingConfig] = None, 
                            enable_fallback: bool = True,
                            use_centralized_config: bool = True) -> EmbeddingProvider:
    """Convenience function to create an embedding provider with optional fallback."""
    return EmbeddingProviderFactory.create_provider(config, enable_fallback, use_centralized_config)


def get_default_provider(enable_fallback: bool = True) -> EmbeddingProvider:
    """Get the default embedding provider based on environment configuration."""
    return EmbeddingProviderFactory.create_provider(enable_fallback=enable_fallback)


def test_all_providers() -> dict:
    """Test all available providers."""
    results = {}
    for provider_name in ["jina", "local"]:
        results[provider_name] = EmbeddingProviderFactory.test_provider(provider_name)
    return results
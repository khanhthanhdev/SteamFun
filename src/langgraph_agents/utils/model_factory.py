"""
Model factory for creating LLM instances across different providers.
Supports OpenAI, AWS Bedrock, OpenRouter, and other providers using litellm.
"""

import os
import logging
from typing import Dict, Any, Optional, Union
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatLiteLLM

try:
    from .bedrock_utils import create_bedrock_llm, get_bedrock_model_config, test_bedrock_connection
    BEDROCK_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Bedrock utilities not available: {e}")
    BEDROCK_AVAILABLE = False
    create_bedrock_llm = None
    get_bedrock_model_config = None
    test_bedrock_connection = None

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating LLM instances across different providers."""
    
    def __init__(self):
        """Initialize the model factory."""
        self.supported_providers = {
            'openai': self._create_openai_llm,
            'bedrock': self._create_bedrock_llm,
            'openrouter': self._create_openrouter_llm
        }
    
    def create_llm(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        **kwargs
    ) -> Union[ChatOpenAI, ChatLiteLLM]:
        """Create LLM instance based on model name with fallback support.
        
        Args:
            model_name: Full model name (e.g., 'openai/gpt-4o', 'bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0')
            temperature: Model temperature
            max_tokens: Maximum tokens
            **kwargs: Additional model parameters
            
        Returns:
            LLM instance
            
        Raises:
            ValueError: If provider is not supported or configuration is invalid
        """
        provider = self._get_provider_from_model_name(model_name)
        
        if provider not in self.supported_providers:
            raise ValueError(f"Unsupported provider: {provider}")
        
        try:
            return self.supported_providers[provider](
                model_name, temperature, max_tokens, **kwargs
            )
        except Exception as e:
            logger.error(f"Failed to create LLM for {model_name}: {e}")
            
            # Try fallback logic
            fallback_llm = self._try_fallback_provider(model_name, temperature, max_tokens, **kwargs)
            if fallback_llm:
                return fallback_llm
            
            raise
    
    def _try_fallback_provider(
        self,
        original_model: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> Optional[Union[ChatOpenAI, ChatLiteLLM]]:
        """Try to create a fallback LLM when the primary model fails.
        
        Args:
            original_model: Original model that failed
            temperature: Model temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters
            
        Returns:
            Fallback LLM instance or None if all fallbacks fail
        """
        from src.config.manager import ConfigurationManager
        
        try:
            config_manager = ConfigurationManager()
            
            # Get enabled providers in order of preference
            enabled_providers = [
                name for name, config in config_manager.get_llm_config().items()
                if config.enabled
            ]
            
            # Try each enabled provider as fallback
            for provider_name in enabled_providers:
                provider_config = config_manager.get_provider_config(provider_name)
                if not provider_config or not provider_config.enabled:
                    continue
                
                fallback_model = f"{provider_name}/{provider_config.default_model}"
                
                # Skip if this is the same as the original model
                if fallback_model == original_model:
                    continue
                
                try:
                    logger.info(f"Trying fallback provider: {fallback_model}")
                    
                    if provider_name in self.supported_providers:
                        return self.supported_providers[provider_name](
                            fallback_model, temperature, max_tokens, **kwargs
                        )
                        
                except Exception as e:
                    logger.warning(f"Fallback provider {provider_name} also failed: {e}")
                    continue
            
            logger.error("All fallback providers failed")
            return None
            
        except Exception as e:
            logger.error(f"Error in fallback logic: {e}")
            return None
    
    def _get_provider_from_model_name(self, model_name: str) -> str:
        """Extract provider from model name.
        
        Args:
            model_name: Full model name
            
        Returns:
            str: Provider name
        """
        if model_name.startswith('openai/'):
            return 'openai'
        elif model_name.startswith('bedrock/'):
            return 'bedrock'
        elif model_name.startswith('openrouter/'):
            return 'openrouter'
        else:
            # Default to OpenAI for backward compatibility
            return 'openai'
    
    def _create_openai_llm(
        self,
        model_name: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> ChatOpenAI:
        """Create OpenAI LLM instance.
        
        Args:
            model_name: OpenAI model name
            temperature: Model temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters
            
        Returns:
            ChatOpenAI: OpenAI LLM instance
        """
        # Get configuration from centralized manager
        from src.config.manager import ConfigurationManager
        config_manager = ConfigurationManager()
        
        # Get provider configuration
        provider_config = config_manager.get_provider_config('openai')
        
        # Extract model ID
        if model_name.startswith('openai/'):
            model_id = model_name[7:]  # Remove 'openai/' prefix
        else:
            model_id = model_name
        
        # Use API key from provider configuration or environment
        api_key = (provider_config.api_key if provider_config else None) or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in configuration or environment variables")
        
        # Use base URL from provider configuration if available
        base_url = (provider_config.base_url if provider_config else None)
        
        # Get timeout and retry settings from provider configuration
        timeout = (provider_config.timeout if provider_config else None) or 30
        max_retries = (provider_config.max_retries if provider_config else None) or 3
        
        return ChatOpenAI(
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
            streaming=kwargs.get('streaming', False),
            timeout=timeout,
            max_retries=max_retries,
            **{k: v for k, v in kwargs.items() if k not in ['streaming', 'timeout', 'max_retries']}
        )
    
    def _create_bedrock_llm(
        self,
        model_name: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> ChatLiteLLM:
        """Create AWS Bedrock LLM instance using litellm.
        
        Args:
            model_name: Bedrock model name
            temperature: Model temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters
            
        Returns:
            ChatLiteLLM: Bedrock LLM instance via litellm
        """
        if not BEDROCK_AVAILABLE or not create_bedrock_llm:
            raise ValueError("Bedrock utilities are not available")
        
        region = kwargs.pop('region', None)
        return create_bedrock_llm(
            model_name=model_name,
            region=region,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    def _create_openrouter_llm(
        self,
        model_name: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> ChatLiteLLM:
        """Create OpenRouter LLM instance using litellm.
        
        Args:
            model_name: OpenRouter model name (e.g., 'openrouter/openai/gpt-4o')
            temperature: Model temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters
            
        Returns:
            ChatLiteLLM: OpenRouter LLM instance via litellm
        """
        # Get configuration from centralized manager
        from src.config.manager import ConfigurationManager
        config_manager = ConfigurationManager()
        
        # Get provider configuration
        provider_config = config_manager.get_provider_config('openrouter')
        
        # Use API key from provider configuration or environment
        api_key = (provider_config.api_key if provider_config else None) or os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in configuration or environment variables")
        
        # Set environment variable for litellm
        os.environ['OPENROUTER_API_KEY'] = api_key
        
        # Get timeout and retry settings from provider configuration
        timeout = (provider_config.timeout if provider_config else None) or 30
        max_retries = (provider_config.max_retries if provider_config else None) or 3
        
        return ChatLiteLLM(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=kwargs.get('streaming', False),
            timeout=timeout,
            max_retries=max_retries,
            **{k: v for k, v in kwargs.items() if k not in ['streaming', 'timeout', 'max_retries']}
        )
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model.
        
        Args:
            model_name: Full model name
            
        Returns:
            Dict: Model information
        """
        provider = self._get_provider_from_model_name(model_name)
        
        if provider == 'bedrock':
            if BEDROCK_AVAILABLE and get_bedrock_model_config:
                return get_bedrock_model_config(model_name)
            else:
                return {"name": model_name, "provider": "bedrock", "available": False}
        elif provider == 'openai':
            return self._get_openai_model_info(model_name)
        elif provider == 'openrouter':
            return self._get_openrouter_model_info(model_name)
        else:
            return {"name": model_name, "provider": provider}
    
    def _get_openai_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get OpenAI model information.
        
        Args:
            model_name: OpenAI model name
            
        Returns:
            Dict: Model information
        """
        model_id = model_name[7:] if model_name.startswith('openai/') else model_name
        
        openai_models = {
            "gpt-4o": {
                "name": "GPT-4o",
                "provider": "OpenAI",
                "context_length": 128000,
                "supports_streaming": True,
                "recommended_for": ["complex reasoning", "code generation", "analysis"]
            },
            "gpt-4o-mini": {
                "name": "GPT-4o Mini",
                "provider": "OpenAI",
                "context_length": 128000,
                "supports_streaming": True,
                "recommended_for": ["fast responses", "cost-effective", "general tasks"]
            },
            "gpt-4": {
                "name": "GPT-4",
                "provider": "OpenAI",
                "context_length": 8192,
                "supports_streaming": True,
                "recommended_for": ["complex tasks", "reasoning", "analysis"]
            },
            "gpt-3.5-turbo": {
                "name": "GPT-3.5 Turbo",
                "provider": "OpenAI",
                "context_length": 16385,
                "supports_streaming": True,
                "recommended_for": ["general tasks", "cost-effective", "fast responses"]
            }
        }
        
        return openai_models.get(model_id, {
            "name": model_id,
            "provider": "OpenAI",
            "context_length": 4000,
            "supports_streaming": True,
            "recommended_for": ["general tasks"]
        })
    
    def _get_openrouter_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get OpenRouter model information.
        
        Args:
            model_name: OpenRouter model name
            
        Returns:
            Dict: Model information
        """
        model_id = model_name[11:] if model_name.startswith('openrouter/') else model_name
        
        return {
            "name": model_id,
            "provider": "OpenRouter",
            "context_length": 4000,  # Default, varies by model
            "supports_streaming": True,
            "recommended_for": ["general tasks", "cost-effective access"]
        }
    
    def validate_model_config(self, model_name: str) -> bool:
        """Validate that a model can be created with current configuration.
        
        Args:
            model_name: Full model name
            
        Returns:
            bool: True if model can be created
        """
        try:
            # Get configuration from centralized manager
            from src.config.manager import ConfigurationManager
            config_manager = ConfigurationManager()
            
            provider = self._get_provider_from_model_name(model_name)
            provider_config = config_manager.get_provider_config(provider)
            
            if provider == 'openai':
                return bool((provider_config.api_key if provider_config else None) or os.getenv('OPENAI_API_KEY'))
            elif provider == 'bedrock':
                return bool(os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY'))
            elif provider == 'openrouter':
                return bool((provider_config.api_key if provider_config else None) or os.getenv('OPENROUTER_API_KEY'))
            
            return False
            
        except Exception:
            return False


# Global model factory instance
model_factory = ModelFactory()
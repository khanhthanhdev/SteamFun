"""
AWS Bedrock utilities for LangGraph agents.
Provides functions to create and configure AWS Bedrock LLM instances using litellm.
"""

import os
import logging
from typing import Dict, Any, Optional
from litellm import completion
from langchain_community.chat_models import ChatLiteLLM

logger = logging.getLogger(__name__)


def create_bedrock_llm(
    model_name: str,
    region: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4000,
    **kwargs
) -> ChatLiteLLM:
    """Create AWS Bedrock LLM instance using litellm.
    
    Args:
        model_name: Bedrock model name (e.g., 'bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0')
        region: AWS region (defaults to AWS_BEDROCK_REGION env var or 'us-east-1')
        temperature: Model temperature
        max_tokens: Maximum tokens
        **kwargs: Additional model parameters
        
    Returns:
        LiteLLM: Configured Bedrock LLM instance
        
    Raises:
        ValueError: If AWS credentials are not configured
    """
    # Get region from parameter, environment, or default
    if not region:
        region = os.getenv('AWS_BEDROCK_REGION', os.getenv('AWS_REGION', 'us-east-1'))
    
    # Validate and set AWS credentials
    _setup_aws_credentials(region)
    
    try:
        llm = ChatLiteLLM(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=kwargs.get('streaming', False),
            **{k: v for k, v in kwargs.items() if k != 'streaming'}
        )
        
        logger.info(f"Created Bedrock LLM via LiteLLM: {model_name} in region {region}")
        return llm
        
    except Exception as e:
        logger.error(f"Failed to create Bedrock LLM {model_name}: {e}")
        raise


def _setup_aws_credentials(region: str):
    """Setup AWS credentials for litellm.
    
    Args:
        region: AWS region
        
    Raises:
        ValueError: If AWS credentials are not configured
    """
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    session_token = os.getenv('AWS_SESSION_TOKEN')
    
    if not access_key or not secret_key:
        raise ValueError(
            "AWS credentials not configured. Please set AWS_ACCESS_KEY_ID, "
            "AWS_SECRET_ACCESS_KEY, and optionally AWS_SESSION_TOKEN environment variables."
        )
    
    # Set environment variables for litellm
    os.environ["AWS_ACCESS_KEY_ID"] = access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key
    os.environ["AWS_REGION_NAME"] = region
    
    if session_token:
        os.environ["AWS_SESSION_TOKEN"] = session_token
    
    logger.info(f"AWS credentials configured for region {region}")


def test_bedrock_connection(model_name: str, region: Optional[str] = None) -> bool:
    """Test AWS Bedrock connection with a simple completion.
    
    Args:
        model_name: Bedrock model name
        region: AWS region
        
    Returns:
        bool: True if connection successful
    """
    try:
        if not region:
            region = os.getenv('AWS_BEDROCK_REGION', os.getenv('AWS_REGION', 'us-east-1'))
        
        _setup_aws_credentials(region)
        
        response = completion(
            model=model_name,
            messages=[{"content": "Hello, how are you?", "role": "user"}],
            max_tokens=10
        )
        
        logger.info(f"Successfully tested Bedrock connection for {model_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to test Bedrock connection for {model_name}: {e}")
        return False


def get_bedrock_completion(
    model_name: str,
    messages: list,
    temperature: float = 0.7,
    max_tokens: int = 4000,
    region: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Get completion from Bedrock model using litellm.
    
    Args:
        model_name: Bedrock model name
        messages: List of messages
        temperature: Model temperature
        max_tokens: Maximum tokens
        region: AWS region
        **kwargs: Additional parameters
        
    Returns:
        Dict: Completion response
    """
    if not region:
        region = os.getenv('AWS_BEDROCK_REGION', os.getenv('AWS_REGION', 'us-east-1'))
    
    _setup_aws_credentials(region)
    
    try:
        response = completion(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        logger.info(f"Successfully got completion from {model_name}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to get completion from {model_name}: {e}")
        raise


def get_available_bedrock_models() -> Dict[str, Dict[str, Any]]:
    """Get available Bedrock models with their configurations.
    
    Returns:
        Dict: Available models with metadata
    """
    return {
        "anthropic.claude-3-5-sonnet-20241022-v2:0": {
            "name": "Claude 3.5 Sonnet (Latest)",
            "provider": "Anthropic",
            "context_length": 200000,
            "supports_streaming": True,
            "recommended_for": ["complex reasoning", "code generation", "analysis"]
        },
        "anthropic.claude-3-5-sonnet-20240620-v1:0": {
            "name": "Claude 3.5 Sonnet",
            "provider": "Anthropic", 
            "context_length": 200000,
            "supports_streaming": True,
            "recommended_for": ["general tasks", "writing", "analysis"]
        },
        "anthropic.claude-3-5-haiku-20241022-v1:0": {
            "name": "Claude 3.5 Haiku (Latest)",
            "provider": "Anthropic",
            "context_length": 200000,
            "supports_streaming": True,
            "recommended_for": ["fast responses", "simple tasks", "cost-effective"]
        },
        "anthropic.claude-3-haiku-20240307-v1:0": {
            "name": "Claude 3 Haiku",
            "provider": "Anthropic",
            "context_length": 200000,
            "supports_streaming": True,
            "recommended_for": ["quick tasks", "summarization", "basic analysis"]
        },
        "us.anthropic.claude-3-7-sonnet-20250219-v1:0": {
            "name": "Claude 3.7 Sonnet",
            "provider": "Anthropic",
            "context_length": 200000,
            "supports_streaming": True,
            "recommended_for": ["latest features", "advanced reasoning", "complex tasks"]
        },
        "amazon.titan-text-premier-v1:0": {
            "name": "Titan Text Premier",
            "provider": "Amazon",
            "context_length": 32000,
            "supports_streaming": True,
            "recommended_for": ["general text generation", "AWS integration"]
        },
        "amazon.titan-text-express-v1": {
            "name": "Titan Text Express",
            "provider": "Amazon",
            "context_length": 8000,
            "supports_streaming": True,
            "recommended_for": ["fast responses", "simple tasks", "cost-effective"]
        }
    }


def get_bedrock_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific Bedrock model.
    
    Args:
        model_name: Full model name (e.g., 'bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0')
        
    Returns:
        Dict: Model configuration
    """
    # Extract model ID
    if model_name.startswith('bedrock/'):
        model_id = model_name[8:]
    else:
        model_id = model_name
    
    models = get_available_bedrock_models()
    return models.get(model_id, {
        "name": model_id,
        "provider": "Unknown",
        "context_length": 4000,
        "supports_streaming": False,
        "recommended_for": ["general tasks"]
    })
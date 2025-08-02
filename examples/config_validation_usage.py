#!/usr/bin/env python3
"""
Example usage of the Configuration Validation Service.

This example demonstrates how to use the ConfigValidationService to validate
configuration values, API keys, and provider compatibility.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import (
    ConfigValidationService, 
    ConfigurationManager,
    LLMProviderConfig,
    EmbeddingConfig,
    VectorStoreConfig
)


async def main():
    """Demonstrate configuration validation usage."""
    print("Configuration Validation Service Usage Example")
    print("=" * 50)
    
    # Initialize the validation service
    validator = ConfigValidationService(timeout=10.0)
    
    # Get the current system configuration
    config_manager = ConfigurationManager()
    system_config = config_manager.config
    
    print("\n1. Validating System Configuration")
    print("-" * 30)
    
    # Validate the complete system configuration
    validation_result = validator.validate_system_config(system_config)
    
    if validation_result.valid:
        print("✅ System configuration is valid!")
    else:
        print("❌ System configuration has issues:")
        for error in validation_result.errors:
            print(f"  Error: {error}")
    
    if validation_result.warnings:
        print("Warnings:")
        for warning in validation_result.warnings:
            print(f"  Warning: {warning}")
    
    print("\n2. Validating Provider-Model Compatibility")
    print("-" * 40)
    
    # Test various provider-model combinations
    test_combinations = [
        ('openai', 'gpt-4o'),
        ('openai', 'gpt-3.5-turbo'),
        ('anthropic', 'claude-3-5-sonnet-20241022'),
        ('gemini', 'gemini-1.5-pro'),
        ('openai', 'invalid-model')  # This should fail
    ]
    
    for provider, model in test_combinations:
        compatibility = validator.validate_provider_model_compatibility(provider, model)
        status = "✅" if compatibility.compatible else "❌"
        reason = f" ({compatibility.reason})" if compatibility.reason else ""
        print(f"  {status} {provider}/{model}{reason}")
    
    print("\n3. Validating Embedding-Vector Store Compatibility")
    print("-" * 45)
    
    # Create test configurations
    embedding_configs = [
        EmbeddingConfig(
            provider="jina",
            model_name="jina-embeddings-v3",
            dimensions=1024,
            api_key="test-key"
        ),
        EmbeddingConfig(
            provider="openai",
            model_name="text-embedding-3-large",
            dimensions=3072,
            api_key="test-key"
        ),
        EmbeddingConfig(
            provider="jina",
            model_name="jina-embeddings-v3",
            dimensions=512,  # Wrong dimensions - should fail
            api_key="test-key"
        )
    ]
    
    vector_configs = [
        VectorStoreConfig(
            provider="chroma",
            collection_name="test_collection",
            connection_params={"persist_directory": "./chroma_db"}
        ),
        VectorStoreConfig(
            provider="astradb",
            collection_name="test_collection",
            connection_params={
                "api_endpoint": "https://test.apps.astra.datastax.com",
                "application_token": "test-token"
            }
        )
    ]
    
    for i, embedding_config in enumerate(embedding_configs):
        for j, vector_config in enumerate(vector_configs):
            compatibility_result = validator.validate_embedding_vector_store_compatibility(
                embedding_config, vector_config
            )
            
            status = "✅" if compatibility_result.valid else "❌"
            print(f"  {status} {embedding_config.provider}/{embedding_config.model_name} + {vector_config.provider}")
            
            if compatibility_result.errors:
                for error in compatibility_result.errors:
                    print(f"    Error: {error}")
            
            if compatibility_result.warnings:
                for warning in compatibility_result.warnings:
                    print(f"    Warning: {warning}")
    
    print("\n4. API Key Validation (if available)")
    print("-" * 35)
    
    # Test API key validation for available keys
    api_key_tests = [
        ('openai', os.getenv('OPENAI_API_KEY')),
        ('anthropic', os.getenv('ANTHROPIC_API_KEY')),
        ('gemini', os.getenv('GEMINI_API_KEY')),
        ('jina', os.getenv('JINA_API_KEY'))
    ]
    
    for provider, api_key in api_key_tests:
        if api_key:
            print(f"  Testing {provider} API key...")
            try:
                validation_result = await validator.validate_api_key(provider, api_key)
                status = "✅" if validation_result.valid else "❌"
                print(f"    {status} {provider}: {'Valid' if validation_result.valid else 'Invalid'}")
                
                if validation_result.errors:
                    for error in validation_result.errors:
                        print(f"      Error: {error}")
                        
                if validation_result.warnings:
                    for warning in validation_result.warnings:
                        print(f"      Info: {warning}")
                        
            except Exception as e:
                print(f"    ❌ {provider}: Validation failed - {e}")
        else:
            print(f"  ⏭️  {provider}: No API key available (set {provider.upper()}_API_KEY to test)")
    
    print("\n5. Connection Testing")
    print("-" * 20)
    
    # Test provider connections
    print("  Testing LLM provider connections:")
    for provider_name, provider_config in system_config.llm_providers.items():
        if provider_config.enabled and provider_config.api_key:
            connection_result = await validator.test_provider_connection(provider_config)
            status = "✅" if connection_result.success else "❌"
            print(f"    {status} {provider_name}: {'Connected' if connection_result.success else 'Failed'}")
            
            if connection_result.error_message:
                print(f"      Error: {connection_result.error_message}")
            
            if connection_result.response_time_ms:
                print(f"      Response time: {connection_result.response_time_ms:.1f}ms")
        else:
            print(f"    ⏭️  {provider_name}: Disabled or no API key")
    
    # Test vector store connections
    print("\n  Testing vector store connections:")
    if system_config.rag_config and system_config.rag_config.enabled:
        vector_config = system_config.rag_config.vector_store_config
        connection_result = await validator.test_vector_store_connection(vector_config)
        status = "✅" if connection_result.success else "❌"
        print(f"    {status} {vector_config.provider}: {'Connected' if connection_result.success else 'Failed'}")
        
        if connection_result.error_message:
            print(f"      Error: {connection_result.error_message}")
        
        if connection_result.response_time_ms:
            print(f"      Response time: {connection_result.response_time_ms:.1f}ms")
    else:
        print("    ⏭️  RAG is disabled")
    
    print("\n6. Provider Information")
    print("-" * 20)
    
    # Show supported providers
    supported_providers = validator.get_supported_providers()
    print(f"  Supported providers: {', '.join(supported_providers)}")
    
    # Show known models for each provider
    for provider in ['openai', 'anthropic', 'gemini', 'jina']:
        models = validator.get_provider_models(provider)
        if models:
            model_list = sorted(list(models))
            display_models = model_list[:3]
            if len(model_list) > 3:
                display_models.append(f"... and {len(model_list) - 3} more")
            print(f"  {provider} models: {', '.join(display_models)}")
    
    # Show embedding dimensions
    print("\n  Embedding model dimensions:")
    embedding_models = [
        'jina-embeddings-v3',
        'text-embedding-3-large',
        'text-embedding-3-small',
        'text-embedding-ada-002'
    ]
    
    for model in embedding_models:
        dimensions = validator.get_embedding_dimensions(model)
        if dimensions:
            print(f"    {model}: {dimensions} dimensions")
    
    print("\n" + "=" * 50)
    print("Configuration validation example completed!")


def sync_example():
    """Demonstrate synchronous validation methods."""
    print("\nSynchronous Validation Example")
    print("-" * 30)
    
    validator = ConfigValidationService(timeout=5.0)
    
    # Test sync API key validation
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print("Testing synchronous API key validation...")
        result = validator.validate_api_key_sync('openai', openai_key)
        status = "✅" if result.valid else "❌"
        print(f"  {status} OpenAI API key: {'Valid' if result.valid else 'Invalid'}")
        
        if result.errors:
            for error in result.errors:
                print(f"    Error: {error}")
    else:
        print("  ⏭️  No OpenAI API key available for sync test")
    
    # Test sync connection testing
    print("\nTesting synchronous connection testing...")
    provider_config = LLMProviderConfig(
        provider="openai",
        api_key=openai_key or "test-key",
        models=["gpt-4o"],
        default_model="gpt-4o"
    )
    
    result = validator.test_provider_connection_sync(provider_config)
    status = "✅" if result.success else "❌"
    print(f"  {status} OpenAI connection: {'Success' if result.success else 'Failed'}")
    
    if result.error_message:
        print(f"    Error: {result.error_message}")


if __name__ == "__main__":
    # Run async example
    asyncio.run(main())
    
    # Run sync example
    sync_example()
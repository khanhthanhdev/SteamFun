#!/usr/bin/env python3
"""
Example usage of the enhanced ConfigurationManager.

This example demonstrates how to use the centralized configuration system
with caching, component-specific access, and hot-reloading capabilities.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config.manager import ConfigurationManager


def main():
    """Demonstrate ConfigurationManager usage."""
    print("ConfigurationManager Usage Example")
    print("=" * 40)
    
    # Get the singleton instance
    config_manager = ConfigurationManager()
    
    # 1. Basic configuration access
    print("\n1. Basic Configuration Access:")
    print(f"Environment: {config_manager.get_environment()}")
    print(f"Development mode: {config_manager.is_development_mode()}")
    print(f"Default LLM provider: {config_manager.get_default_provider()}")
    
    # 2. Component-specific configuration access
    print("\n2. Component-Specific Configuration:")
    
    # LLM providers
    llm_config = config_manager.get_llm_config()
    print(f"Available LLM providers: {list(llm_config.keys())}")
    
    for provider_name in config_manager.get_all_provider_names():
        provider_config = config_manager.get_provider_config(provider_name)
        print(f"  {provider_name}: {len(provider_config.models)} models, enabled={provider_config.enabled}")
    
    # RAG configuration
    rag_config = config_manager.get_rag_config()
    if rag_config:
        print(f"RAG enabled: {rag_config.enabled}")
        print(f"  Embedding provider: {rag_config.embedding_config.provider}")
        print(f"  Vector store: {rag_config.vector_store_config.provider}")
    
    # Agent configurations
    print(f"Available agents: {config_manager.get_all_agent_names()}")
    for agent_name in config_manager.get_all_agent_names()[:3]:  # Show first 3
        agent_config = config_manager.get_agent_config(agent_name)
        print(f"  {agent_name}: timeout={agent_config.timeout_seconds}s, enabled={agent_config.enabled}")
    
    # 3. Caching demonstration
    print("\n3. Caching Performance:")
    
    # Clear cache first
    config_manager.clear_cache()
    
    # Time first access
    start_time = time.time()
    llm_config1 = config_manager.get_llm_config()
    first_access_time = time.time() - start_time
    
    # Time cached access
    start_time = time.time()
    llm_config2 = config_manager.get_llm_config()
    cached_access_time = time.time() - start_time
    
    print(f"First access: {first_access_time:.4f}s")
    print(f"Cached access: {cached_access_time:.4f}s")
    print(f"Speed improvement: {first_access_time / cached_access_time:.1f}x faster")
    
    # Show cache stats
    cache_stats = config_manager.get_cache_stats()
    print(f"Cache size: {cache_stats['cache_size']} items")
    print(f"Cache TTL: {cache_stats['cache_ttl']} seconds")
    
    # 4. Configuration validation
    print("\n4. Configuration Validation:")
    validation_result = config_manager.validate_configuration()
    print(f"Configuration valid: {validation_result.valid}")
    if validation_result.errors:
        print("Errors:")
        for error in validation_result.errors:
            print(f"  - {error}")
    if validation_result.warnings:
        print("Warnings:")
        for warning in validation_result.warnings:
            print(f"  - {warning}")
    
    # 5. Configuration summary
    print("\n5. Configuration Summary:")
    summary = config_manager.get_config_summary()
    for key, value in summary.items():
        if isinstance(value, list):
            print(f"{key}: {len(value)} items - {value}")
        else:
            print(f"{key}: {value}")
    
    # 6. Hot-reload demonstration (development mode only)
    print("\n6. Hot-Reload Capabilities:")
    if config_manager.is_development_mode():
        print("Hot-reload is available in development mode")
        print("You can call config_manager.reload_config() to reload configuration")
        
        # Demonstrate cache management
        print(f"Cache TTL: {config_manager._cache_ttl} seconds")
        print("You can adjust cache TTL with config_manager.set_cache_ttl(seconds)")
    else:
        print("Hot-reload is disabled in production mode")
    
    # 7. Model configuration access (legacy compatibility)
    print("\n7. Model Configuration Access:")
    model_configs = [
        "openai/gpt-4o",
        "gemini/gemini-2.5-flash",
        "bedrock/amazon.nova-pro-v1:0"
    ]
    
    for model_name in model_configs:
        try:
            model_config = config_manager.get_model_config(model_name)
            print(f"{model_name}: provider={model_config.get('provider', 'unknown')}")
        except Exception as e:
            print(f"{model_name}: Error - {e}")
    
    print("\n" + "=" * 40)
    print("ConfigurationManager example completed successfully!")


if __name__ == '__main__':
    main()
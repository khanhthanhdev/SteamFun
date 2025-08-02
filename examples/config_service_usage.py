#!/usr/bin/env python3
"""
Example usage of ConfigurationService for .env file handling.

This example demonstrates how to use the ConfigurationService to load,
parse, and validate configuration from .env files with support for
nested variables and multiple environment files.
"""

import os
from pathlib import Path
from src.config.service import ConfigurationService
from src.config.models import SystemConfig
from src.config.factory import ConfigurationFactory


def example_basic_usage():
    """Example of basic ConfigurationService usage."""
    print("=== Basic ConfigurationService Usage ===")
    
    # Initialize the service
    service = ConfigurationService()
    
    # Load configuration from default .env file
    try:
        config = service.load_env_config()
        print(f"Loaded {len(config)} configuration values from .env")
        
        # Show some example values
        if 'environment' in config:
            print(f"Environment: {config['environment']}")
        if 'debug' in config:
            print(f"Debug mode: {config['debug']}")
        if 'default_llm_provider' in config:
            print(f"Default LLM provider: {config['default_llm_provider']}")
            
    except FileNotFoundError:
        print("No .env file found - this is normal if you haven't created one yet")
    except Exception as e:
        print(f"Error loading configuration: {e}")


def example_environment_specific_loading():
    """Example of environment-specific configuration loading."""
    print("\n=== Environment-Specific Configuration Loading ===")
    
    service = ConfigurationService()
    
    # Load configuration for different environments
    environments = ['development', 'staging', 'production']
    
    for env in environments:
        try:
            config = service.load_environment_specific_config(env)
            print(f"\n{env.title()} configuration:")
            print(f"  Environment: {config.get('environment', 'not set')}")
            print(f"  Debug: {config.get('debug', 'not set')}")
            print(f"  Config keys: {len(config)}")
            
        except Exception as e:
            print(f"Error loading {env} configuration: {e}")


def example_nested_configuration():
    """Example of nested configuration parsing."""
    print("\n=== Nested Configuration Parsing ===")
    
    service = ConfigurationService()
    
    # Example nested configuration
    nested_config = {
        'LLM_PROVIDERS__OPENAI__API_KEY': 'sk-test123',
        'LLM_PROVIDERS__OPENAI__MODELS': 'gpt-4o,gpt-4o-mini',
        'LLM_PROVIDERS__OPENAI__DEFAULT_MODEL': 'gpt-4o',
        'RAG_CONFIG__ENABLED': 'true',
        'RAG_CONFIG__EMBEDDING_CONFIG__PROVIDER': 'jina',
        'RAG_CONFIG__EMBEDDING_CONFIG__DIMENSIONS': '1024',
        'MONITORING_CONFIG__LANGFUSE_CONFIG__ENABLED': 'true'
    }
    
    # Parse nested configuration
    parsed = service.parse_nested_config(nested_config)
    
    print("Parsed nested configuration:")
    print(f"  LLM Providers: {list(parsed.get('llm_providers', {}).keys())}")
    
    if 'llm_providers' in parsed and 'openai' in parsed['llm_providers']:
        openai_config = parsed['llm_providers']['openai']
        print(f"  OpenAI API Key: {openai_config.get('api_key', 'not set')}")
        print(f"  OpenAI Models: {openai_config.get('models', 'not set')}")
    
    if 'rag_config' in parsed:
        rag_config = parsed['rag_config']
        print(f"  RAG Enabled: {rag_config.get('enabled', 'not set')}")
        if 'embedding_config' in rag_config:
            embedding = rag_config['embedding_config']
            print(f"  Embedding Provider: {embedding.get('provider', 'not set')}")
            print(f"  Embedding Dimensions: {embedding.get('dimensions', 'not set')}")


def example_type_conversion():
    """Example of automatic type conversion."""
    print("\n=== Automatic Type Conversion ===")
    
    service = ConfigurationService()
    
    # Example values with different types
    test_values = {
        'BOOL_TRUE': 'true',
        'BOOL_FALSE': 'false',
        'INTEGER': '42',
        'FLOAT': '3.14159',
        'LIST': 'item1,item2,item3',
        'STRING': 'hello world'
    }
    
    print("Type conversion examples:")
    for key, value in test_values.items():
        converted = service._convert_env_value(value)
        print(f"  {key}: '{value}' -> {converted} ({type(converted).__name__})")


def example_validation():
    """Example of configuration validation."""
    print("\n=== Configuration Validation ===")
    
    service = ConfigurationService()
    
    # Example configuration for validation
    test_config = {
        'environment': 'development',
        'debug': True,
        'default_llm_provider': 'openai',
        'openai_api_key': 'sk-test123',
        'openai_models': ['gpt-4o', 'gpt-4o-mini'],
        'openai_default_model': 'gpt-4o',
        'rag_enabled': True,
        'embedding_provider': 'jina',
        'jina_api_key': 'jina_test123'
    }
    
    # Validate configuration
    result = service.validate_env_config(test_config)
    
    print(f"Validation result: {'✓ Valid' if result.valid else '✗ Invalid'}")
    
    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")
    
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")


def example_get_env_value():
    """Example of get_env_value method with type casting."""
    print("\n=== Environment Variable Access with Type Casting ===")
    
    service = ConfigurationService()
    
    # Set some test environment variables
    test_vars = {
        'TEST_STRING': 'hello',
        'TEST_INT': '42',
        'TEST_FLOAT': '3.14',
        'TEST_BOOL': 'true',
        'TEST_LIST': 'a,b,c'
    }
    
    # Set environment variables
    for key, value in test_vars.items():
        os.environ[key] = value
    
    try:
        print("Environment variable access with type casting:")
        
        # String (default)
        string_val = service.get_env_value('TEST_STRING')
        print(f"  String: {string_val} ({type(string_val).__name__})")
        
        # Integer
        int_val = service.get_env_value('TEST_INT', cast_type=int)
        print(f"  Integer: {int_val} ({type(int_val).__name__})")
        
        # Float
        float_val = service.get_env_value('TEST_FLOAT', cast_type=float)
        print(f"  Float: {float_val} ({type(float_val).__name__})")
        
        # Boolean
        bool_val = service.get_env_value('TEST_BOOL', cast_type=bool)
        print(f"  Boolean: {bool_val} ({type(bool_val).__name__})")
        
        # List
        list_val = service.get_env_value('TEST_LIST', cast_type=list)
        print(f"  List: {list_val} ({type(list_val).__name__})")
        
        # Default value
        default_val = service.get_env_value('NONEXISTENT_KEY', default='default_value')
        print(f"  Default: {default_val} ({type(default_val).__name__})")
        
    finally:
        # Clean up test environment variables
        for key in test_vars.keys():
            os.environ.pop(key, None)


def example_integration_with_system():
    """Example of integration with existing system configuration."""
    print("\n=== Integration with System Configuration ===")
    
    service = ConfigurationService()
    
    try:
        # Load configuration using service
        config_dict = service.load_env_config()
        
        if config_dict:
            print(f"Loaded configuration with {len(config_dict)} keys")
            
            # Validate the configuration
            validation_result = service.validate_env_config(config_dict)
            print(f"Configuration validation: {'✓ Valid' if validation_result.valid else '✗ Invalid'}")
            
            # Try to build SystemConfig using the factory
            try:
                system_config = ConfigurationFactory.build_system_config()
                print(f"Successfully built SystemConfig:")
                print(f"  Environment: {system_config.environment}")
                print(f"  Debug: {system_config.debug}")
                print(f"  Default LLM Provider: {system_config.default_llm_provider}")
                print(f"  LLM Providers: {list(system_config.llm_providers.keys())}")
                
                if system_config.rag_config:
                    print(f"  RAG Enabled: {system_config.rag_config.enabled}")
                    print(f"  Embedding Provider: {system_config.rag_config.embedding_config.provider}")
                
            except Exception as e:
                print(f"Error building SystemConfig: {e}")
                
        else:
            print("No configuration loaded - create a .env file to see this in action")
            
    except Exception as e:
        print(f"Error in integration example: {e}")


def main():
    """Run all examples."""
    print("ConfigurationService Usage Examples")
    print("=" * 50)
    
    example_basic_usage()
    example_environment_specific_loading()
    example_nested_configuration()
    example_type_conversion()
    example_validation()
    example_get_env_value()
    example_integration_with_system()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo get started:")
    print("1. Create a .env file in your project root")
    print("2. Add configuration variables (see .env.example)")
    print("3. Use ConfigurationService to load and validate your configuration")


if __name__ == '__main__':
    main()
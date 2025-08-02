#!/usr/bin/env python3
"""
Comprehensive Configuration Validation Usage Example.

This example demonstrates all the new configuration validation features:
- Startup configuration validation with detailed error messages
- Configuration compatibility checks between components
- Configuration health check endpoint for monitoring
- Configuration migration utilities for version updates
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import (
    ConfigurationManager,
    ConfigValidationService,
    ConfigurationHealthChecker,
    ConfigurationMigrator,
    StartupValidationService,
    SystemConfig
)


async def demonstrate_startup_validation():
    """Demonstrate comprehensive startup validation."""
    print("=" * 60)
    print("STARTUP CONFIGURATION VALIDATION DEMO")
    print("=" * 60)
    
    # Initialize startup validation service
    startup_validator = StartupValidationService()
    
    # Perform comprehensive startup validation
    print("\n1. Running comprehensive startup validation...")
    result = await startup_validator.validate_startup_configuration(strict_mode=False)
    
    print(f"\nValidation Result:")
    print(f"  Status: {'✅ Valid' if result.valid else '❌ Invalid'}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Warnings: {len(result.warnings)}")
    
    if result.errors:
        print("\nErrors found:")
        for i, error in enumerate(result.errors[:5], 1):
            print(f"  {i}. {error}")
        if len(result.errors) > 5:
            print(f"  ... and {len(result.errors) - 5} more errors")
    
    if result.warnings:
        print("\nWarnings found:")
        for i, warning in enumerate(result.warnings[:5], 1):
            print(f"  {i}. {warning}")
        if len(result.warnings) > 5:
            print(f"  ... and {len(result.warnings) - 5} more warnings")
    
    # Demonstrate strict mode
    print("\n2. Running validation in strict mode...")
    strict_result = await startup_validator.validate_startup_configuration(strict_mode=True)
    
    print(f"\nStrict Mode Result:")
    print(f"  Status: {'✅ Valid' if strict_result.valid else '❌ Invalid'}")
    print(f"  Errors: {len(strict_result.errors)}")
    print(f"  Warnings: {len(strict_result.warnings)}")
    
    return result


async def demonstrate_health_checking():
    """Demonstrate configuration health checking."""
    print("\n" + "=" * 60)
    print("CONFIGURATION HEALTH CHECKING DEMO")
    print("=" * 60)
    
    # Initialize health checker
    health_checker = ConfigurationHealthChecker()
    
    # Quick health status
    print("\n1. Quick health status...")
    quick_status = health_checker.get_quick_status()
    print(f"  Status: {quick_status['status']}")
    print(f"  Environment: {quick_status.get('environment', 'unknown')}")
    print(f"  Providers: {quick_status.get('providers', {}).get('enabled', 0)}/{quick_status.get('providers', {}).get('total', 0)} enabled")
    print(f"  Agents: {quick_status.get('agents', {}).get('enabled', 0)}/{quick_status.get('agents', {}).get('total', 0)} enabled")
    print(f"  RAG Enabled: {quick_status.get('rag_enabled', False)}")
    
    # Comprehensive health check
    print("\n2. Comprehensive health check...")
    health_result = await health_checker.perform_health_check(include_connectivity=True)
    
    print(f"\nHealth Check Result:")
    print(f"  Overall Status: {health_result.status}")
    print(f"  Response Time: {health_result.response_time_ms:.1f}ms")
    print(f"  Errors: {len(health_result.errors)}")
    print(f"  Warnings: {len(health_result.warnings)}")
    
    # Show individual component health
    print(f"\nComponent Health:")
    for component, check_result in health_result.checks.items():
        if isinstance(check_result, dict):
            status = check_result.get('status', 'unknown')
            response_time = check_result.get('response_time_ms', 0)
            print(f"  {component}: {status} ({response_time:.1f}ms)")
    
    # Show health check as JSON (for monitoring endpoints)
    print(f"\n3. Health check JSON (for monitoring):")
    health_json = health_result.to_dict()
    print(json.dumps({
        'status': health_json['status'],
        'timestamp': health_json['timestamp'],
        'response_time_ms': health_json['response_time_ms'],
        'summary': {
            'errors': len(health_json['errors']),
            'warnings': len(health_json['warnings']),
            'components_checked': len(health_json['checks'])
        }
    }, indent=2))
    
    return health_result


def demonstrate_configuration_migration():
    """Demonstrate configuration migration utilities."""
    print("\n" + "=" * 60)
    print("CONFIGURATION MIGRATION DEMO")
    print("=" * 60)
    
    # Initialize migrator
    migrator = ConfigurationMigrator()
    
    print(f"Current schema version: {migrator.CURRENT_VERSION}")
    print(f"Available migration steps: {len(migrator.migration_steps)}")
    
    # Create a sample legacy configuration
    legacy_config = {
        'environment': 'development',
        'debug': True,
        'openai_api_key': 'sk-test-key',
        'anthropic_api_key': 'sk-ant-test-key',
        'use_rag': True,
        'embedding_model': 'jina-embeddings-v3',
        'chroma_db_path': 'data/rag/chroma_db',
        'output_dir': 'output',
        'max_scene_concurrency': 5,
        'max_retries': 3,
        'use_langfuse': True,
        'langfuse_secret_key': 'sk-lf-test',
        'langfuse_public_key': 'pk-lf-test'
    }
    
    print(f"\n1. Detecting configuration version...")
    detected_version = migrator.detect_config_version(legacy_config)
    print(f"  Detected version: {detected_version}")
    
    print(f"\n2. Checking if migration is needed...")
    needs_migration = migrator.needs_migration(legacy_config)
    print(f"  Needs migration: {needs_migration}")
    
    if needs_migration:
        print(f"\n3. Performing migration...")
        migration_result = migrator.migrate_configuration(legacy_config)
        
        print(f"  Migration Result:")
        print(f"    Success: {migration_result.success}")
        print(f"    From: {migration_result.from_version}")
        print(f"    To: {migration_result.to_version}")
        print(f"    Changes: {len(migration_result.changes_made)}")
        print(f"    Warnings: {len(migration_result.warnings)}")
        print(f"    Errors: {len(migration_result.errors)}")
        
        if migration_result.backup_path:
            print(f"    Backup: {migration_result.backup_path}")
        
        if migration_result.changes_made:
            print(f"\n  Changes made:")
            for i, change in enumerate(migration_result.changes_made[:3], 1):
                print(f"    {i}. {change}")
            if len(migration_result.changes_made) > 3:
                print(f"    ... and {len(migration_result.changes_made) - 3} more changes")
        
        # Save migration record
        migrator.save_migration_record(migration_result)
        print(f"  Migration record saved to history")
    
    # Show migration history
    print(f"\n4. Migration history...")
    history = migrator.get_migration_history()
    print(f"  Total migrations: {len(history)}")
    
    if history:
        latest = history[-1]
        print(f"  Latest migration:")
        print(f"    {latest['from_version']} -> {latest['to_version']}")
        print(f"    Success: {latest['success']}")
        print(f"    Timestamp: {latest.get('timestamp', 'unknown')}")
    
    return migration_result if needs_migration else None


async def demonstrate_component_compatibility():
    """Demonstrate component compatibility validation."""
    print("\n" + "=" * 60)
    print("COMPONENT COMPATIBILITY VALIDATION DEMO")
    print("=" * 60)
    
    # Initialize validator
    validator = ConfigValidationService()
    config_manager = ConfigurationManager()
    config = config_manager.config
    
    print("\n1. Provider-Model Compatibility...")
    test_combinations = [
        ('openai', 'gpt-4o'),
        ('openai', 'gpt-3.5-turbo'),
        ('anthropic', 'claude-3-5-sonnet-20241022'),
        ('gemini', 'gemini-1.5-pro'),
        ('openai', 'invalid-model'),  # Should fail
        ('unknown-provider', 'some-model')  # Should fail
    ]
    
    for provider, model in test_combinations:
        compatibility = validator.validate_provider_model_compatibility(provider, model)
        status = "✅" if compatibility.compatible else "❌"
        reason = f" ({compatibility.reason})" if compatibility.reason else ""
        print(f"  {status} {provider}/{model}{reason}")
    
    print("\n2. Embedding-Vector Store Compatibility...")
    if config.rag_config and config.rag_config.enabled:
        embedding_config = config.rag_config.embedding_config
        vector_config = config.rag_config.vector_store_config
        
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
    else:
        print("  RAG system is disabled - skipping compatibility check")
    
    print("\n3. Agent-Provider Compatibility...")
    for agent_name, agent_config in config.agent_configs.items():
        if not agent_config.enabled:
            continue
        
        print(f"  Agent: {agent_name}")
        
        # Check model references
        model_fields = ['planner_model', 'scene_model', 'helper_model']
        for field in model_fields:
            model_name = getattr(agent_config, field, None)
            if model_name and '/' in model_name:
                provider, model = model_name.split('/', 1)
                
                if provider in config.llm_providers:
                    provider_config = config.llm_providers[provider]
                    if provider_config.enabled:
                        if model in provider_config.models:
                            print(f"    ✅ {field}: {model_name}")
                        else:
                            print(f"    ⚠️  {field}: {model_name} (model not in provider's list)")
                    else:
                        print(f"    ❌ {field}: {model_name} (provider disabled)")
                else:
                    print(f"    ❌ {field}: {model_name} (unknown provider)")


async def demonstrate_api_key_validation():
    """Demonstrate API key validation."""
    print("\n" + "=" * 60)
    print("API KEY VALIDATION DEMO")
    print("=" * 60)
    
    validator = ConfigValidationService()
    config_manager = ConfigurationManager()
    config = config_manager.config
    
    print("\n1. Validating configured API keys...")
    
    for provider_name, provider_config in config.llm_providers.items():
        if provider_config.enabled and provider_config.api_key:
            print(f"  Testing {provider_name} API key...")
            try:
                result = await validator.validate_api_key(provider_name, provider_config.api_key)
                status = "✅" if result.valid else "❌"
                print(f"    {status} {provider_name}: {'Valid' if result.valid else 'Invalid'}")
                
                if result.errors:
                    for error in result.errors:
                        print(f"      Error: {error}")
                
                if result.warnings:
                    for warning in result.warnings:
                        print(f"      Info: {warning}")
                        
            except Exception as e:
                print(f"    ❌ {provider_name}: Validation failed - {e}")
        else:
            status = "disabled" if not provider_config.enabled else "no API key"
            print(f"  ⏭️  {provider_name}: Skipped ({status})")
    
    # Test RAG embedding provider
    if config.rag_config and config.rag_config.enabled:
        embedding_config = config.rag_config.embedding_config
        if embedding_config.provider != 'local' and embedding_config.api_key:
            print(f"  Testing {embedding_config.provider} embedding API key...")
            try:
                result = await validator.validate_api_key(embedding_config.provider, embedding_config.api_key)
                status = "✅" if result.valid else "❌"
                print(f"    {status} {embedding_config.provider}: {'Valid' if result.valid else 'Invalid'}")
                
                if result.errors:
                    for error in result.errors:
                        print(f"      Error: {error}")
                        
            except Exception as e:
                print(f"    ❌ {embedding_config.provider}: Validation failed - {e}")


async def main():
    """Run all validation demonstrations."""
    print("Configuration Validation System Demonstration")
    print("=" * 60)
    
    try:
        # 1. Startup validation
        startup_result = await demonstrate_startup_validation()
        
        # 2. Health checking
        health_result = await demonstrate_health_checking()
        
        # 3. Configuration migration
        migration_result = demonstrate_configuration_migration()
        
        # 4. Component compatibility
        await demonstrate_component_compatibility()
        
        # 5. API key validation
        await demonstrate_api_key_validation()
        
        # Summary
        print("\n" + "=" * 60)
        print("DEMONSTRATION SUMMARY")
        print("=" * 60)
        
        print(f"Startup Validation: {'✅ Passed' if startup_result.valid else '❌ Failed'}")
        print(f"Health Check: {health_result.status}")
        print(f"Migration: {'✅ Available' if migration_result else 'Not needed'}")
        print(f"API Key Tests: Completed")
        print(f"Compatibility Tests: Completed")
        
        print(f"\nAll validation features demonstrated successfully!")
        
    except Exception as e:
        print(f"\nDemonstration failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
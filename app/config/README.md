# Configuration Management System

This directory contains the centralized configuration management system for the FastAPI application, built using Pydantic Settings.

## Overview

The configuration system provides:
- **Centralized Configuration**: All settings organized in logical sections
- **Environment Variable Support**: Automatic loading from environment variables and .env files
- **Environment-Specific Overrides**: Different configurations for dev, test, and production
- **Type Safety**: Full Pydantic validation with proper types and enums
- **Configuration Validation**: Comprehensive validation with error reporting
- **Hot Reloading**: Support for configuration reloading without restart

## Quick Start

```python
from app.config import get_settings

# Get configuration instance
settings = get_settings()

# Access configuration sections
print(f"App: {settings.app.app_name}")
print(f"Database: {settings.database.database_host}")
print(f"RAG Enabled: {settings.rag.rag_enabled}")
```

## Configuration Sections

### AppSettings (`settings.app`)
Core application settings including server configuration, debugging, and workflow settings.

**Environment Variables**: `APP_*`
- `APP_NAME` - Application name
- `ENVIRONMENT` - Environment (development/testing/production)
- `DEBUG` - Debug mode
- `HOST`, `PORT` - Server settings

### DatabaseSettings (`settings.database`)
Database connection and pool configuration.

**Environment Variables**: `DB_*`
- `DATABASE_URL` - Full database URL
- `DATABASE_HOST`, `DATABASE_PORT` - Connection details
- `DATABASE_POOL_SIZE` - Connection pool size

### LLMSettings (`settings.llm`)
LLM provider configurations (OpenAI, AWS Bedrock, etc.).

**Environment Variables**: `LLM_*`
- `OPENAI_API_KEY` - OpenAI API key
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` - AWS credentials
- `GEMINI_API_KEY` - Google Gemini API key

### RAGSettings (`settings.rag`)
RAG system configuration including embeddings and vector stores.

**Environment Variables**: `RAG_*`
- `RAG_ENABLED` - Enable/disable RAG system
- `EMBEDDING_PROVIDER` - Embedding provider (jina/local/openai)
- `VECTOR_STORE_PROVIDER` - Vector store (astradb/chroma/pinecone)

### TTSSettings (`settings.tts`)
Text-to-Speech configuration.

**Environment Variables**: `TTS_*`
- `KOKORO_MODEL_PATH` - Path to Kokoro model
- `KOKORO_DEFAULT_VOICE` - Default voice

### MonitoringSettings (`settings.monitoring`)
Monitoring and observability settings.

**Environment Variables**: `MONITORING_*`
- `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY` - LangFuse credentials
- `MONITORING_ENABLED` - Enable monitoring

### SecuritySettings (`settings.security`)
Security, CORS, and authentication settings.

**Environment Variables**: `SECURITY_*`
- `SECRET_KEY` - JWT secret key
- `CORS_ORIGINS` - Allowed CORS origins

## Environment Files

### `.env` (Development)
Main environment file for development settings.

### `.env.test` (Testing)
Testing environment overrides:
- Disables external services
- Uses local/in-memory stores
- Minimal logging

### `.env.prod` (Production)
Production environment overrides:
- Enhanced security settings
- Multiple workers
- Full monitoring enabled

### `.env.example`
Template file showing all available configuration options.

## Usage Examples

### Basic Usage
```python
from app.config import get_settings

settings = get_settings()
print(f"Running {settings.app.app_name} in {settings.app.environment} mode")
```

### FastAPI Integration
```python
from fastapi import FastAPI, Depends
from app.config import get_settings, Settings

app = FastAPI()

@app.get("/health")
def health_check(settings: Settings = Depends(get_settings)):
    return {
        "status": "healthy",
        "environment": settings.app.environment,
        "debug": settings.app.debug
    }
```

### Configuration Validation
```python
from app.config import get_settings, validate_configuration

settings = get_settings()
validation_result = validate_configuration(settings)

if not validation_result.is_valid:
    print("Configuration errors:")
    for error in validation_result.errors:
        print(f"  - {error}")
```

### Environment-Specific Logic
```python
settings = get_settings()

if settings.app.is_production:
    # Production-specific setup
    setup_monitoring()
    enable_security_features()
elif settings.app.is_development:
    # Development-specific setup
    enable_debug_logging()
    setup_hot_reload()
```

## Configuration Factory

For advanced use cases, use the configuration factory:

```python
from app.config.factory import ConfigurationFactory
from app.config.settings import Environment

# Create settings for specific environment
settings = ConfigurationFactory.create_settings(Environment.PRODUCTION)

# Reload configuration
settings = ConfigurationFactory.reload_settings()
```

## CLI Tools

The configuration system includes CLI tools for management:

```bash
# Validate configuration
python -m app.config.cli validate --environment production

# Show configuration
python -m app.config.cli show --section app

# Export configuration to JSON
python -m app.config.cli export --output config.json

# Create environment template
python -m app.config.cli template production --output .env.prod
```

## File Structure

```
app/config/
├── __init__.py          # Package exports
├── settings.py          # Main configuration classes
├── factory.py           # Configuration factory
├── environments.py      # Environment-specific overrides
├── validation.py        # Configuration validation
├── cli.py              # CLI management tools
├── example.py          # Usage examples
└── README.md           # This file
```

## Best Practices

1. **Use Environment Variables**: Store sensitive data in environment variables, not in code
2. **Validate Configuration**: Always validate configuration in production
3. **Environment-Specific Files**: Use separate .env files for different environments
4. **Type Safety**: Leverage Pydantic's type validation
5. **Documentation**: Document all configuration options
6. **Security**: Never commit .env files with real secrets to version control

## Troubleshooting

### Common Issues

1. **"Extra inputs are not permitted"**: This error occurs when environment variables don't match expected field names. Check your environment variable names and prefixes.

2. **"Field required"**: Required fields are missing. Check that all necessary environment variables are set.

3. **"Invalid environment value"**: Environment variable values don't match expected types. Check data types and enum values.

### Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check loaded configuration
settings = get_settings()
print(settings.model_dump())

# Validate configuration
from app.config import validate_configuration
result = validate_configuration(settings)
print(result.get_summary())
```

## Contributing

When adding new configuration options:

1. Add fields to the appropriate settings class
2. Update environment variable documentation
3. Add validation rules if needed
4. Update the .env.example file
5. Test with different environments
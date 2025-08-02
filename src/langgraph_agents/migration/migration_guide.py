"""
Migration guide and documentation for configuration changes.
Provides comprehensive documentation for migrating to LangGraph multi-agent system.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime


logger = logging.getLogger(__name__)


class MigrationGuide:
    """Provides migration guidance and documentation for configuration changes."""
    
    def __init__(self):
        """Initialize migration guide."""
        self.version = "1.0.0"
        self.migration_date = datetime.now().strftime("%Y-%m-%d")
        logger.info("MigrationGuide initialized")
    
    def generate_complete_migration_guide(self, output_path: str = "MIGRATION_GUIDE.md") -> str:
        """Generate complete migration guide documentation.
        
        Args:
            output_path: Path to save the migration guide
            
        Returns:
            str: Path to generated guide
        """
        guide_content = self._create_complete_guide_content()
        
        output_file = Path(output_path)
        with open(output_file, 'w') as f:
            f.write(guide_content)
        
        logger.info(f"Complete migration guide generated: {output_file}")
        return str(output_file)
    
    def generate_quick_start_guide(self, output_path: str = "QUICK_START_MIGRATION.md") -> str:
        """Generate quick start migration guide.
        
        Args:
            output_path: Path to save the quick start guide
            
        Returns:
            str: Path to generated guide
        """
        guide_content = self._create_quick_start_content()
        
        output_file = Path(output_path)
        with open(output_file, 'w') as f:
            f.write(guide_content)
        
        logger.info(f"Quick start migration guide generated: {output_file}")
        return str(output_file)
    
    def generate_parameter_mapping_reference(self, output_path: str = "PARAMETER_MAPPING.md") -> str:
        """Generate parameter mapping reference documentation.
        
        Args:
            output_path: Path to save the parameter mapping reference
            
        Returns:
            str: Path to generated reference
        """
        reference_content = self._create_parameter_mapping_content()
        
        output_file = Path(output_path)
        with open(output_file, 'w') as f:
            f.write(reference_content)
        
        logger.info(f"Parameter mapping reference generated: {output_file}")
        return str(output_file)
    
    def generate_troubleshooting_guide(self, output_path: str = "MIGRATION_TROUBLESHOOTING.md") -> str:
        """Generate troubleshooting guide for common migration issues.
        
        Args:
            output_path: Path to save the troubleshooting guide
            
        Returns:
            str: Path to generated guide
        """
        guide_content = self._create_troubleshooting_content()
        
        output_file = Path(output_path)
        with open(output_file, 'w') as f:
            f.write(guide_content)
        
        logger.info(f"Migration troubleshooting guide generated: {output_file}")
        return str(output_file) 
   
    def _create_complete_guide_content(self) -> str:
        """Create complete migration guide content."""
        return f"""# LangGraph Multi-Agent System Migration Guide

**Version:** {self.version}  
**Date:** {self.migration_date}

## Overview

This guide helps you migrate from the existing sequential video generation pipeline to the new LangGraph multi-agent system. The migration provides better error handling, parallelization, and maintainability while preserving the same external API.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Migration Process](#migration-process)
3. [Configuration Changes](#configuration-changes)
4. [Parameter Mapping](#parameter-mapping)
5. [Testing Migration](#testing-migration)
6. [Rollback Procedures](#rollback-procedures)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- Python 3.8+
- LangGraph 0.2.0+
- LangChain 0.3.0+
- Existing video generation system

### Environment Setup
Ensure you have the following environment variables configured:

```bash
# Required for basic functionality
OPENAI_API_KEY="your-openai-api-key"

# Optional but recommended for enhanced features
AWS_BEDROCK_REGION="us-east-1"
JINA_API_KEY="your-jina-api-key"
ASTRADB_APPLICATION_TOKEN="your-astradb-token"
ASTRADB_API_ENDPOINT="your-astradb-endpoint"
LANGFUSE_SECRET_KEY="your-langfuse-secret"
LANGFUSE_PUBLIC_KEY="your-langfuse-public"
```

## Migration Process

### Step 1: Backup Existing Configuration

Before starting migration, backup your existing configuration:

```python
from src.langgraph_agents.migration import ConfigurationMigrator

migrator = ConfigurationMigrator()
# Backup is automatically created during migration
```

### Step 2: Run Automated Migration

#### From .env File
```python
from src.langgraph_agents.migration import ConfigurationMigrator

migrator = ConfigurationMigrator()
success, messages = migrator.migrate_from_env_file(".env")

if success:
    print("Migration completed successfully!")
    for message in messages:
        print(f"✅ {{message}}")
else:
    print("Migration failed:")
    for message in messages:
        print(f"❌ {{message}}")
```

### Step 3: Validate Migration

```python
from src.langgraph_agents.migration import ConfigurationValidator
from src.langgraph_agents.config import ConfigurationManager

config_manager = ConfigurationManager()
validator = ConfigurationValidator()

# Load migrated configuration
system_config = config_manager.load_system_config()

# Validate configuration
is_valid, errors = validator.validate_system_config(system_config)

if is_valid:
    print("✅ Configuration is valid!")
else:
    print("❌ Configuration has errors:")
    for error in errors:
        print(f"  - {{error.component}}.{{error.parameter}}: {{error.message}}")
```

## Key Changes Summary

- **Configuration**: Now in `config/system_config.json`
- **Agents**: 8 specialized agents instead of 3 classes
- **Models**: Use `provider/model` format (e.g., `openai/gpt-4o`)
- **Parameters**: Some renamed (e.g., `use_rag` → `enable_rag`)
- **Features**: Added human-loop, monitoring, error handling

## Common Parameter Mappings

### CodeGenerator → code_generator_agent
- `use_rag` → `enable_rag`
- `print_response` → `verbose`
- `output_dir` → `output_directory`
- `chroma_db_path` → `vector_store_path`

### Model Names
- `"gpt-4"` → `"openai/gpt-4"`
- `"gpt-3.5-turbo"` → `"openai/gpt-3.5-turbo"`

---
*For detailed migration guide, parameter mappings, and troubleshooting, use the specific guide generation methods.*
"""

    def _create_quick_start_content(self) -> str:
        """Create quick start migration guide content."""
        return f"""# Quick Start Migration Guide

**Version:** {self.version}  
**Date:** {self.migration_date}

## 5-Minute Migration

### Step 1: Install Dependencies
```bash
pip install langgraph langchain langchain-community
```

### Step 2: Run Migration
```python
from src.langgraph_agents.migration import ConfigurationMigrator

migrator = ConfigurationMigrator()
success, messages = migrator.migrate_from_env_file(".env")

if success:
    print("✅ Migration completed!")
else:
    print("❌ Migration failed - check messages")
```

### Step 3: Test
```python
from src.langgraph_agents.workflow import VideoGenerationWorkflow

workflow = VideoGenerationWorkflow()
result = workflow.run({{
    "topic": "Test",
    "description": "Quick test"
}})
```

## Key Changes Summary

- **Configuration**: Now in `config/system_config.json`
- **Agents**: 8 specialized agents instead of 3 classes
- **Models**: Use `provider/model` format (e.g., `openai/gpt-4o`)
- **Parameters**: Some renamed (e.g., `use_rag` → `enable_rag`)
- **Features**: Added human-loop, monitoring, error handling

## Common Fixes

### Model Names
```python
# Old
"scene_model": "gpt-4"

# New  
"scene_model": "openai/gpt-4"
```

### Parameter Names
```python
# Old
"use_rag": True
"print_response": False

# New
"enable_rag": True
"verbose": False
```

### Environment Variables
Ensure these are set in `.env`:
```bash
OPENAI_API_KEY="your-key"
JINA_API_KEY="your-jina-key"  # Optional but recommended
LANGFUSE_SECRET_KEY="your-langfuse-key"  # Optional
```

---
*For detailed migration guide, see MIGRATION_GUIDE.md*
"""    
    def _create_parameter_mapping_content(self) -> str:
        """Create parameter mapping reference content."""
        return f"""# Parameter Mapping Reference

**Version:** {self.version}  
**Date:** {self.migration_date}

## CodeGenerator → code_generator_agent

### Direct Mappings
| Old Parameter | New Parameter | Type | Default | Notes |
|---------------|---------------|------|---------|-------|
| `scene_model` | `scene_model` | str | required | Must use provider/model format |
| `helper_model` | `helper_model` | str | required | Must use provider/model format |
| `temperature` | `temperature` | float | 0.7 | Model temperature (0.0-2.0) |
| `max_retries` | `max_retries` | int | 3 | Maximum retry attempts |

### Renamed Parameters
| Old Parameter | New Parameter | Type | Default | Notes |
|---------------|---------------|------|---------|-------|
| `use_rag` | `enable_rag` | bool | true | Enable RAG functionality |
| `use_context_learning` | `enable_context_learning` | bool | true | Enable context learning |
| `use_visual_fix_code` | `enable_visual_fix_code` | bool | false | Enable visual code fixing |
| `print_response` | `verbose` | bool | false | Enable verbose output |
| `output_dir` | `output_directory` | str | "output" | Output directory path |
| `chroma_db_path` | `vector_store_path` | str | "data/rag/chroma_db" | Vector store path |

### Deprecated Parameters
| Old Parameter | Replacement | Notes |
|---------------|-------------|-------|
| `rag_queries_cache` | *(internal)* | Now handled automatically by RAG agent |
| `banned_reasonings` | *(error_handler_agent)* | Now handled by error handler agent |
| `visual_self_reflection_func` | *(visual_analysis_agent)* | Now handled by visual analysis agent |

## Model Name Format Changes

### Old Format
```python
"scene_model": "gpt-4"
"helper_model": "gpt-3.5-turbo"
```

### New Format
```python
"scene_model": "openai/gpt-4"
"helper_model": "openai/gpt-3.5-turbo"

# AWS Bedrock models
"scene_model": "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"

# OpenRouter models  
"scene_model": "openrouter/openai/gpt-4o"
```

## Environment Variables

### LLM Provider Configuration
| Old Variable | New Usage | Notes |
|--------------|-----------|-------|
| `OPENAI_API_KEY` | `llm_providers.openai.api_key_env` | Referenced in provider config |
| `AWS_BEDROCK_REGION` | `llm_providers.aws_bedrock.region` | Direct mapping to provider config |
| `JINA_API_KEY` | `rag_config.jina_config.api_key_env` | JINA API configuration |
| `LANGFUSE_SECRET_KEY` | `monitoring_config.langfuse_config.secret_key_env` | LangFuse secret key |

---
*This reference is automatically generated from the parameter conversion rules.*
"""

    def _create_troubleshooting_content(self) -> str:
        """Create troubleshooting guide content."""
        return f"""# Migration Troubleshooting Guide

**Version:** {self.version}  
**Date:** {self.migration_date}

## Common Migration Issues

### 1. Model Name Format Errors

#### Error Messages
```
❌ Invalid model name format: 'gpt-4'
❌ Model name must include provider prefix
```

#### Solutions
```python
# ❌ Old format
"scene_model": "gpt-4"

# ✅ New format
"scene_model": "openai/gpt-4"
"scene_model": "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
"scene_model": "openrouter/openai/gpt-4o"
```

### 2. Missing Environment Variables

#### Error Messages
```
❌ Environment variable not set: OPENAI_API_KEY
❌ OpenAI provider requires api_key_env
```

#### Solutions
Create or update your `.env` file:
```bash
# Required
OPENAI_API_KEY="sk-..."

# Optional but recommended
JINA_API_KEY="jina_..."
ASTRADB_APPLICATION_TOKEN="AstraCS:..."
LANGFUSE_SECRET_KEY="sk-lf-..."
```

### 3. Path Configuration Issues

#### Error Messages
```
❌ Path does not exist: data/rag/chroma_db
❌ Cannot create output directory: /invalid/path
```

#### Solutions
```python
from pathlib import Path

# Create required directories
required_dirs = [
    "data/rag/chroma_db",
    "data/rag/manim_docs", 
    "data/context_learning",
    "output"
]

for dir_path in required_dirs:
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    print(f"✅ Created directory: {{dir_path}}")
```

### 4. Agent Configuration Errors

#### Error Messages
```
❌ Required agent missing: planner_agent
❌ Agent code_generator_agent missing scene_model
```

#### Solutions
```python
from src.langgraph_agents.config import ConfigurationManager

config_manager = ConfigurationManager()

# Update agent configuration
config_manager.update_agent_config("code_generator_agent", {{
    "scene_model": "openai/gpt-4o",
    "helper_model": "openai/gpt-4o-mini",
    "max_retries": 3,
    "timeout_seconds": 600
}})
```

## Diagnostic Tools

### Configuration Validator
```python
from src.langgraph_agents.migration import ConfigurationValidator

validator = ConfigurationValidator()
config = load_system_config()
is_valid, errors = validator.validate_system_config(config)

# Generate detailed report
report = validator.generate_validation_report(errors)
print(report)
```

### Environment Check
```python
import os

def check_environment():
    env_vars = ["OPENAI_API_KEY", "JINA_API_KEY", "LANGFUSE_SECRET_KEY"]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {{var}} set")
        else:
            print(f"⚠️ {{var}} not set")

check_environment()
```

---
*For additional help, refer to the main migration guide or project documentation.*
"""

    def get_migration_checklist(self) -> List[str]:
        """Get migration checklist for users.
        
        Returns:
            List[str]: Migration checklist items
        """
        return [
            "✅ Backup existing configuration files",
            "✅ Verify environment variables are set (OPENAI_API_KEY, etc.)",
            "✅ Install required dependencies (langgraph, langchain)",
            "✅ Run configuration migration tool",
            "✅ Validate migrated configuration",
            "✅ Test with simple workflow",
            "✅ Update deployment scripts",
            "✅ Monitor performance after migration",
            "✅ Update documentation with any custom changes",
            "✅ Train team on new configuration format"
        ]
    
    def get_rollback_instructions(self) -> str:
        """Get rollback instructions.
        
        Returns:
            str: Rollback instructions
        """
        return """# Rollback Instructions

If migration causes issues, follow these steps to rollback:

## Automatic Rollback
```python
from pathlib import Path
import shutil

# Find latest backup
backup_dir = Path("config/backup")
if backup_dir.exists():
    backups = list(backup_dir.glob("backup_*"))
    if backups:
        latest_backup = max(backups, key=lambda p: p.stat().st_mtime)
        
        # Restore system config
        shutil.copy(
            latest_backup / "system_config.json",
            "config/system_config.json"
        )
        
        print(f"✅ Restored from {latest_backup}")
    else:
        print("❌ No backups found")
```

## Manual Rollback
1. Stop the application
2. Restore configuration files from backup
3. Revert environment variable changes
4. Restart with original system
5. Verify functionality
"""
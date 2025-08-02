"""
Configuration migration utilities for converting existing configurations
to LangGraph multi-agent system format.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import asdict
from datetime import datetime

from ..config import ConfigurationManager, SystemConfig, AgentConfig
from ..compatibility.parameter_mapping import BackwardCompatibleParameterMapper
from .validation_utils import ConfigurationValidator


logger = logging.getLogger(__name__)


class ConfigurationMigrator:
    """Migrates existing configurations to LangGraph multi-agent format."""
    
    def __init__(self, 
                 source_config_dir: str = "config",
                 target_config_dir: str = "config",
                 backup_dir: str = "config/backup"):
        """Initialize configuration migrator.
        
        Args:
            source_config_dir: Directory containing existing configurations
            target_config_dir: Directory for migrated configurations
            backup_dir: Directory for configuration backups
        """
        self.source_config_dir = Path(source_config_dir)
        self.target_config_dir = Path(target_config_dir)
        self.backup_dir = Path(backup_dir)
        
        # Create directories if they don't exist
        self.target_config_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_manager = ConfigurationManager(str(self.target_config_dir))
        self.parameter_mapper = BackwardCompatibleParameterMapper()
        self.validator = ConfigurationValidator()
        
        logger.info(f"ConfigurationMigrator initialized: {source_config_dir} -> {target_config_dir}")
    
    def migrate_from_env_file(self, env_file_path: str = ".env") -> Tuple[bool, List[str]]:
        """Migrate configuration from .env file to LangGraph format.
        
        Args:
            env_file_path: Path to .env file
            
        Returns:
            Tuple[bool, List[str]]: (success, list_of_messages)
        """
        messages = []
        
        try:
            # Backup existing configuration
            backup_success = self._backup_existing_config()
            if backup_success:
                messages.append("Existing configuration backed up successfully")
            
            # Parse .env file
            env_config = self._parse_env_file(env_file_path)
            if not env_config:
                return False, ["Failed to parse .env file"]
            
            messages.append(f"Parsed {len(env_config)} environment variables")
            
            # Convert to LangGraph system configuration
            system_config = self._convert_env_to_system_config(env_config)
            messages.append("Converted environment configuration to system configuration")
            
            # Validate migrated configuration
            is_valid, validation_errors = self.validator.validate_system_config(system_config)
            if not is_valid:
                messages.extend([f"Validation error: {error}" for error in validation_errors])
                return False, messages
            
            messages.append("Configuration validation passed")
            
            # Save migrated configuration
            save_success = self.config_manager.save_system_config(system_config)
            if not save_success:
                return False, messages + ["Failed to save migrated configuration"]
            
            messages.append("Migrated configuration saved successfully")
            
            # Generate migration report
            report_path = self._generate_migration_report(env_config, system_config)
            messages.append(f"Migration report generated: {report_path}")
            
            return True, messages
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False, messages + [f"Migration failed: {str(e)}"]
    
    def migrate_from_legacy_config(self, legacy_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Migrate from legacy configuration dictionary.
        
        Args:
            legacy_config: Legacy configuration dictionary
            
        Returns:
            Tuple[bool, List[str]]: (success, list_of_messages)
        """
        messages = []
        
        try:
            # Backup existing configuration
            backup_success = self._backup_existing_config()
            if backup_success:
                messages.append("Existing configuration backed up successfully")
            
            # Convert legacy configuration
            system_config = self._convert_legacy_to_system_config(legacy_config)
            messages.append("Converted legacy configuration to system configuration")
            
            # Validate migrated configuration
            is_valid, validation_errors = self.validator.validate_system_config(system_config)
            if not is_valid:
                messages.extend([f"Validation error: {error}" for error in validation_errors])
                return False, messages
            
            messages.append("Configuration validation passed")
            
            # Save migrated configuration
            save_success = self.config_manager.save_system_config(system_config)
            if not save_success:
                return False, messages + ["Failed to save migrated configuration"]
            
            messages.append("Migrated configuration saved successfully")
            
            # Generate migration report
            report_path = self._generate_migration_report(legacy_config, system_config)
            messages.append(f"Migration report generated: {report_path}")
            
            return True, messages
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False, messages + [f"Migration failed: {str(e)}"]
    
    def migrate_component_config(self, 
                                component_type: str, 
                                component_config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], List[str]]:
        """Migrate configuration for a specific component.
        
        Args:
            component_type: Type of component (code_generator, video_planner, video_renderer)
            component_config: Component configuration
            
        Returns:
            Tuple[bool, Dict[str, Any], List[str]]: (success, migrated_config, messages)
        """
        messages = []
        
        try:
            if component_type == "code_generator":
                migrated_config = self.parameter_mapper.map_code_generator_params(**component_config)
            elif component_type == "video_planner":
                migrated_config = self.parameter_mapper.map_video_planner_params(**component_config)
            elif component_type == "video_renderer":
                migrated_config = self.parameter_mapper.map_video_renderer_params(**component_config)
            else:
                return False, {}, [f"Unknown component type: {component_type}"]
            
            # Validate migrated configuration
            is_valid, validation_errors = self.parameter_mapper.validate_configuration(migrated_config)
            if not is_valid:
                messages.extend([f"Validation warning: {error}" for error in validation_errors])
            
            messages.append(f"Successfully migrated {component_type} configuration")
            return True, migrated_config, messages
            
        except Exception as e:
            logger.error(f"Component migration failed: {e}")
            return False, {}, [f"Component migration failed: {str(e)}"]
    
    def _parse_env_file(self, env_file_path: str) -> Optional[Dict[str, str]]:
        """Parse .env file into dictionary.
        
        Args:
            env_file_path: Path to .env file
            
        Returns:
            Dict[str, str]: Environment variables or None if failed
        """
        try:
            env_vars = {}
            env_path = Path(env_file_path)
            
            if not env_path.exists():
                logger.error(f".env file not found: {env_file_path}")
                return None
            
            with open(env_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse key=value pairs
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"\'')  # Remove quotes
                        env_vars[key] = value
                    else:
                        logger.warning(f"Invalid line in .env file (line {line_num}): {line}")
            
            logger.info(f"Parsed {len(env_vars)} environment variables from {env_file_path}")
            return env_vars
            
        except Exception as e:
            logger.error(f"Failed to parse .env file: {e}")
            return None
    
    def _convert_env_to_system_config(self, env_vars: Dict[str, str]) -> SystemConfig:
        """Convert environment variables to SystemConfig.
        
        Args:
            env_vars: Environment variables dictionary
            
        Returns:
            SystemConfig: Converted system configuration
        """
        # Start with default configuration
        system_config = self.config_manager.create_default_system_config()
        
        # Update LLM provider configurations
        self._update_llm_providers_from_env(system_config, env_vars)
        
        # Update agent configurations
        self._update_agents_from_env(system_config, env_vars)
        
        # Update external service configurations
        self._update_external_services_from_env(system_config, env_vars)
        
        # Update workflow settings
        self._update_workflow_settings_from_env(system_config, env_vars)
        
        return system_config
    
    def _update_llm_providers_from_env(self, system_config: SystemConfig, env_vars: Dict[str, str]):
        """Update LLM provider configurations from environment variables."""
        
        # OpenAI configuration
        if 'OPENAI_API_KEY' in env_vars:
            system_config.llm_providers['openai']['api_key_env'] = 'OPENAI_API_KEY'
        
        # AWS Bedrock configuration
        if 'AWS_BEDROCK_REGION' in env_vars:
            system_config.llm_providers['aws_bedrock']['region'] = env_vars['AWS_BEDROCK_REGION']
        elif 'AWS_REGION' in env_vars:
            system_config.llm_providers['aws_bedrock']['region'] = env_vars['AWS_REGION']
        
        if 'AWS_BEDROCK_MODEL' in env_vars:
            model = env_vars['AWS_BEDROCK_MODEL']
            if model not in system_config.llm_providers['aws_bedrock']['models']:
                system_config.llm_providers['aws_bedrock']['models'].append(model)
            system_config.llm_providers['aws_bedrock']['default_model'] = model
        
        # OpenRouter configuration
        if 'OPENROUTER_API_KEY' in env_vars:
            system_config.llm_providers['openrouter']['api_key_env'] = 'OPENROUTER_API_KEY'
    
    def _update_agents_from_env(self, system_config: SystemConfig, env_vars: Dict[str, str]):
        """Update agent configurations from environment variables."""
        
        # Update model assignments based on environment variables
        model_mappings = {
            'planner_agent': env_vars.get('PLANNER_MODEL', 'openai/gpt-4o'),
            'code_generator_agent': env_vars.get('SCENE_MODEL', 'openai/gpt-4o'),
            'renderer_agent': env_vars.get('RENDERER_MODEL', 'openai/gpt-4o'),
            'visual_analysis_agent': env_vars.get('VISUAL_MODEL', 'openai/gpt-4o'),
            'rag_agent': env_vars.get('RAG_MODEL', 'openai/gpt-4o-mini'),
            'error_handler_agent': env_vars.get('ERROR_MODEL', 'openai/gpt-4o-mini'),
            'monitoring_agent': env_vars.get('MONITORING_MODEL', 'openai/gpt-4o-mini'),
            'human_loop_agent': env_vars.get('HUMAN_LOOP_MODEL', 'openai/gpt-4o-mini')
        }
        
        for agent_name, model in model_mappings.items():
            if agent_name in system_config.agents:
                # Update primary model based on agent type
                if agent_name == 'planner_agent':
                    system_config.agents[agent_name].planner_model = model
                elif agent_name == 'code_generator_agent':
                    system_config.agents[agent_name].scene_model = model
                
                # Update helper model
                helper_model = env_vars.get('HELPER_MODEL', 'openai/gpt-4o-mini')
                system_config.agents[agent_name].helper_model = helper_model
        
        # Update global agent settings
        if 'AGENT_TEMPERATURE' in env_vars:
            temperature = float(env_vars['AGENT_TEMPERATURE'])
            for agent in system_config.agents.values():
                agent.temperature = temperature
                agent.model_config['temperature'] = temperature
        
        if 'AGENT_MAX_TOKENS' in env_vars:
            max_tokens = int(env_vars['AGENT_MAX_TOKENS'])
            for agent in system_config.agents.values():
                agent.model_config['max_tokens'] = max_tokens
        
        if 'AGENT_TIMEOUT' in env_vars:
            timeout = int(env_vars['AGENT_TIMEOUT'])
            for agent in system_config.agents.values():
                agent.timeout_seconds = timeout
                agent.model_config['timeout'] = timeout
    
    def _update_external_services_from_env(self, system_config: SystemConfig, env_vars: Dict[str, str]):
        """Update external service configurations from environment variables."""
        
        # RAG configuration
        rag_config = {}
        if 'RAG_ENABLED' in env_vars:
            rag_config['enabled'] = env_vars['RAG_ENABLED'].lower() == 'true'
        
        if 'EMBEDDING_PROVIDER' in env_vars:
            rag_config['embedding_provider'] = env_vars['EMBEDDING_PROVIDER']
        
        if 'VECTOR_STORE_PROVIDER' in env_vars:
            rag_config['vector_store_provider'] = env_vars['VECTOR_STORE_PROVIDER']
        
        # JINA API configuration
        if 'JINA_API_KEY' in env_vars:
            rag_config['jina_config'] = {
                'api_key_env': 'JINA_API_KEY',
                'model': env_vars.get('JINA_EMBEDDING_MODEL', 'jina-embeddings-v3'),
                'api_url': env_vars.get('JINA_API_URL', 'https://api.jina.ai/v1/embeddings'),
                'dimension': int(env_vars.get('EMBEDDING_DIMENSION', '1024')),
                'batch_size': int(env_vars.get('EMBEDDING_BATCH_SIZE', '100')),
                'timeout': int(env_vars.get('EMBEDDING_TIMEOUT', '30'))
            }
        
        # AstraDB configuration
        if 'ASTRADB_API_ENDPOINT' in env_vars:
            rag_config['astradb_config'] = {
                'api_endpoint': env_vars['ASTRADB_API_ENDPOINT'],
                'application_token_env': 'ASTRADB_APPLICATION_TOKEN',
                'keyspace': env_vars.get('ASTRADB_KEYSPACE', 'default_keyspace'),
                'collection': env_vars.get('VECTOR_STORE_COLLECTION', 'manim_docs_jina_1024'),
                'region': env_vars.get('ASTRADB_REGION', 'us-east-2'),
                'timeout': int(env_vars.get('ASTRADB_TIMEOUT', '30'))
            }
        
        # ChromaDB configuration (fallback)
        if 'CHROMA_DB_PATH' in env_vars:
            rag_config['chroma_config'] = {
                'db_path': env_vars['CHROMA_DB_PATH'],
                'collection_name': env_vars.get('CHROMA_COLLECTION_NAME', 'manim_docs'),
                'persist_directory': env_vars.get('CHROMA_PERSIST_DIRECTORY', 'data/rag/chroma_persist')
            }
        
        # Document processing configuration
        if 'MANIM_DOCS_PATH' in env_vars:
            rag_config['docs_config'] = {
                'manim_docs_path': env_vars['MANIM_DOCS_PATH'],
                'context_learning_path': env_vars.get('CONTEXT_LEARNING_PATH', 'data/context_learning'),
                'chunk_size': int(env_vars.get('RAG_CHUNK_SIZE', '1000')),
                'chunk_overlap': int(env_vars.get('RAG_CHUNK_OVERLAP', '200')),
                'extensions': env_vars.get('RAG_DOCS_EXTENSIONS', '.md,.txt,.py,.rst').split(',')
            }
        
        system_config.rag_config = rag_config
        
        # LangFuse configuration
        if 'LANGFUSE_SECRET_KEY' in env_vars:
            system_config.monitoring_config['langfuse_config'] = {
                'enabled': True,
                'secret_key_env': 'LANGFUSE_SECRET_KEY',
                'public_key_env': 'LANGFUSE_PUBLIC_KEY',
                'host': env_vars.get('LANGFUSE_HOST', 'https://cloud.langfuse.com'),
                'debug': env_vars.get('LANGFUSE_DEBUG', 'false').lower() == 'true'
            }
        
        # MCP servers configuration
        if 'MCP_CONTEXT7_ENABLED' in env_vars and env_vars['MCP_CONTEXT7_ENABLED'].lower() == 'true':
            system_config.mcp_servers['context7']['disabled'] = False
        
        if 'MCP_DOCLING_ENABLED' in env_vars and env_vars['MCP_DOCLING_ENABLED'].lower() == 'true':
            system_config.mcp_servers['docling']['disabled'] = False
    
    def _update_workflow_settings_from_env(self, system_config: SystemConfig, env_vars: Dict[str, str]):
        """Update workflow settings from environment variables."""
        
        if 'MAX_WORKFLOW_RETRIES' in env_vars:
            system_config.max_workflow_retries = int(env_vars['MAX_WORKFLOW_RETRIES'])
        
        if 'WORKFLOW_TIMEOUT_SECONDS' in env_vars:
            system_config.workflow_timeout_seconds = int(env_vars['WORKFLOW_TIMEOUT_SECONDS'])
        
        if 'ENABLE_CHECKPOINTS' in env_vars:
            system_config.enable_checkpoints = env_vars['ENABLE_CHECKPOINTS'].lower() == 'true'
        
        if 'CHECKPOINT_INTERVAL' in env_vars:
            system_config.checkpoint_interval = int(env_vars['CHECKPOINT_INTERVAL'])
        
        if 'HUMAN_LOOP_ENABLED' in env_vars:
            system_config.human_loop_config['enabled'] = env_vars['HUMAN_LOOP_ENABLED'].lower() == 'true'
        
        if 'HUMAN_LOOP_TIMEOUT' in env_vars:
            system_config.human_loop_config['timeout_seconds'] = int(env_vars['HUMAN_LOOP_TIMEOUT'])
    
    def _convert_legacy_to_system_config(self, legacy_config: Dict[str, Any]) -> SystemConfig:
        """Convert legacy configuration dictionary to SystemConfig.
        
        Args:
            legacy_config: Legacy configuration dictionary
            
        Returns:
            SystemConfig: Converted system configuration
        """
        # Start with default configuration
        system_config = self.config_manager.create_default_system_config()
        
        # Map legacy parameters using parameter mapper
        if 'code_generator' in legacy_config:
            cg_config = self.parameter_mapper.map_code_generator_params(**legacy_config['code_generator'])
            self._update_agent_from_mapped_config(system_config, 'code_generator_agent', cg_config)
        
        if 'video_planner' in legacy_config:
            vp_config = self.parameter_mapper.map_video_planner_params(**legacy_config['video_planner'])
            self._update_agent_from_mapped_config(system_config, 'planner_agent', vp_config)
        
        if 'video_renderer' in legacy_config:
            vr_config = self.parameter_mapper.map_video_renderer_params(**legacy_config['video_renderer'])
            self._update_agent_from_mapped_config(system_config, 'renderer_agent', vr_config)
        
        # Map global settings
        if 'global_settings' in legacy_config:
            self._update_global_settings(system_config, legacy_config['global_settings'])
        
        return system_config
    
    def _update_agent_from_mapped_config(self, system_config: SystemConfig, agent_name: str, mapped_config: Dict[str, Any]):
        """Update agent configuration from mapped parameters."""
        if agent_name in system_config.agents:
            agent = system_config.agents[agent_name]
            
            # Update model configurations
            if 'scene_model' in mapped_config:
                agent.scene_model = mapped_config['scene_model']
            if 'planner_model' in mapped_config:
                agent.planner_model = mapped_config['planner_model']
            if 'helper_model' in mapped_config:
                agent.helper_model = mapped_config['helper_model']
            
            # Update other parameters
            if 'temperature' in mapped_config:
                agent.temperature = mapped_config['temperature']
                agent.model_config['temperature'] = mapped_config['temperature']
            
            if 'max_retries' in mapped_config:
                agent.max_retries = mapped_config['max_retries']
            
            if 'timeout_seconds' in mapped_config:
                agent.timeout_seconds = mapped_config['timeout_seconds']
                agent.model_config['timeout'] = mapped_config['timeout_seconds']
    
    def _update_global_settings(self, system_config: SystemConfig, global_settings: Dict[str, Any]):
        """Update global system settings."""
        if 'max_workflow_retries' in global_settings:
            system_config.max_workflow_retries = global_settings['max_workflow_retries']
        
        if 'workflow_timeout_seconds' in global_settings:
            system_config.workflow_timeout_seconds = global_settings['workflow_timeout_seconds']
        
        if 'enable_checkpoints' in global_settings:
            system_config.enable_checkpoints = global_settings['enable_checkpoints']
    
    def _backup_existing_config(self) -> bool:
        """Backup existing configuration files.
        
        Returns:
            bool: True if backup successful
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_subdir = self.backup_dir / f"backup_{timestamp}"
            backup_subdir.mkdir(parents=True, exist_ok=True)
            
            # Backup system config if it exists
            system_config_path = self.target_config_dir / "system_config.json"
            if system_config_path.exists():
                backup_path = backup_subdir / "system_config.json"
                backup_path.write_text(system_config_path.read_text())
                logger.info(f"Backed up system_config.json to {backup_path}")
            
            # Backup .env file if it exists
            env_path = Path(".env")
            if env_path.exists():
                backup_env_path = backup_subdir / ".env"
                backup_env_path.write_text(env_path.read_text())
                logger.info(f"Backed up .env to {backup_env_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to backup existing configuration: {e}")
            return False
    
    def _generate_migration_report(self, 
                                  source_config: Dict[str, Any], 
                                  target_config: SystemConfig) -> str:
        """Generate migration report.
        
        Args:
            source_config: Source configuration
            target_config: Target configuration
            
        Returns:
            str: Path to generated report
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.target_config_dir / f"migration_report_{timestamp}.md"
            
            report_content = self._create_migration_report_content(source_config, target_config)
            
            with open(report_path, 'w') as f:
                f.write(report_content)
            
            logger.info(f"Migration report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Failed to generate migration report: {e}")
            return ""
    
    def _create_migration_report_content(self, 
                                       source_config: Dict[str, Any], 
                                       target_config: SystemConfig) -> str:
        """Create migration report content.
        
        Args:
            source_config: Source configuration
            target_config: Target configuration
            
        Returns:
            str: Report content in Markdown format
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# Configuration Migration Report

**Migration Date:** {timestamp}
**Source Configuration:** {len(source_config)} parameters
**Target Configuration:** LangGraph Multi-Agent System

## Migration Summary

### Agents Configured
"""
        
        for agent_name, agent_config in target_config.agents.items():
            report += f"- **{agent_name}**: {agent_config.name}\n"
            if hasattr(agent_config, 'scene_model') and agent_config.scene_model:
                report += f"  - Scene Model: {agent_config.scene_model}\n"
            if hasattr(agent_config, 'planner_model') and agent_config.planner_model:
                report += f"  - Planner Model: {agent_config.planner_model}\n"
            if hasattr(agent_config, 'helper_model') and agent_config.helper_model:
                report += f"  - Helper Model: {agent_config.helper_model}\n"
            report += f"  - Max Retries: {agent_config.max_retries}\n"
            report += f"  - Timeout: {agent_config.timeout_seconds}s\n"
            report += f"  - Human Loop: {agent_config.enable_human_loop}\n\n"
        
        report += f"""
### LLM Providers Configured
"""
        
        for provider_name, provider_config in target_config.llm_providers.items():
            report += f"- **{provider_name}**\n"
            if 'default_model' in provider_config:
                report += f"  - Default Model: {provider_config['default_model']}\n"
            if 'models' in provider_config:
                report += f"  - Available Models: {len(provider_config['models'])}\n"
            report += "\n"
        
        report += f"""
### External Services
- **Docling**: {'Enabled' if target_config.docling_config.get('enabled', False) else 'Disabled'}
- **MCP Servers**: {len([s for s in target_config.mcp_servers.values() if not s.get('disabled', False)])} enabled
- **Monitoring**: {'Enabled' if target_config.monitoring_config.get('enabled', False) else 'Disabled'}
- **Human Loop**: {'Enabled' if target_config.human_loop_config.get('enabled', False) else 'Disabled'}

### Workflow Settings
- **Max Retries**: {target_config.max_workflow_retries}
- **Timeout**: {target_config.workflow_timeout_seconds}s
- **Checkpoints**: {'Enabled' if target_config.enable_checkpoints else 'Disabled'}
- **Checkpoint Interval**: {target_config.checkpoint_interval}s

## Configuration Files Generated
- `system_config.json`: Main system configuration
- `migration_report_{timestamp.replace(' ', '_').replace(':', '').replace('-', '')}.md`: This report

## Next Steps
1. Review the generated configuration files
2. Test the migrated configuration with a simple workflow
3. Update any custom settings as needed
4. Remove or archive old configuration files

## Validation Results
Configuration validation: ✅ Passed

---
*Generated by LangGraph Configuration Migrator*
"""
        
        return report
"""
ConfigAdapter for migrating between old and new configuration formats.

This adapter handles the migration of configuration files and provides
utilities for converting between different configuration schemas.
"""

from typing import Dict, Any, Optional, List, Union
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime

from ..models.config import WorkflowConfig, ModelConfig, RAGConfig
from ..state import AgentConfig, SystemConfig

logger = logging.getLogger(__name__)


class ConfigAdapter:
    """
    Adapter for converting between old and new configuration formats.
    
    Handles migration of configuration files, validation of migrated data,
    and provides utilities for backward compatibility.
    """
    
    @staticmethod
    def migrate_system_config(old_config: Dict[str, Any]) -> WorkflowConfig:
        """
        Migrate old system configuration to new WorkflowConfig format.
        
        Args:
            old_config: Old configuration dictionary
            
        Returns:
            WorkflowConfig: New configuration format
        """
        try:
            # Extract model configurations
            planner_model = ConfigAdapter._extract_model_config(
                old_config, 'planner_model', 'openrouter', 'anthropic/claude-3.5-sonnet'
            )
            code_model = ConfigAdapter._extract_model_config(
                old_config, 'code_model', 'openrouter', 'anthropic/claude-3.5-sonnet'
            )
            helper_model = ConfigAdapter._extract_model_config(
                old_config, 'helper_model', 'openrouter', 'anthropic/claude-3.5-sonnet'
            )
            
            # Extract RAG configuration
            rag_config = RAGConfig(
                enabled=old_config.get('use_rag', True),
                vector_store_path=old_config.get('chroma_db_path', 'data/rag/chroma_db'),
                embedding_model=old_config.get('embedding_model', 'hf:ibm-granite/granite-embedding-30m-english'),
                chunk_size=old_config.get('chunk_size', 1000),
                chunk_overlap=old_config.get('chunk_overlap', 200),
                similarity_threshold=old_config.get('rag_quality_threshold', 0.7),
                max_results=old_config.get('max_rag_results', 5),
                enable_caching=old_config.get('enable_rag_caching', True),
                cache_ttl=old_config.get('rag_cache_ttl', 3600)
            )
            
            # Create new workflow configuration
            workflow_config = WorkflowConfig(
                # Model configurations
                planner_model=planner_model,
                code_model=code_model,
                helper_model=helper_model,
                
                # RAG configuration
                rag_config=rag_config,
                
                # Feature flags
                use_rag=old_config.get('use_rag', True),
                use_visual_analysis=old_config.get('use_visual_fix_code', False),
                enable_caching=old_config.get('enable_caching', True),
                use_context_learning=old_config.get('use_context_learning', True),
                
                # Performance settings
                max_retries=old_config.get('max_retries', 3),
                timeout_seconds=old_config.get('timeout_seconds', 300),
                max_concurrent_scenes=old_config.get('max_scene_concurrency', 5),
                max_concurrent_renders=old_config.get('max_concurrent_renders', 4),
                
                # Quality settings
                default_quality=old_config.get('default_quality', 'medium'),
                use_gpu_acceleration=old_config.get('use_gpu_acceleration', False),
                preview_mode=old_config.get('preview_mode', False),
                
                # Directory settings
                output_dir=old_config.get('output_dir', 'output'),
                context_learning_path=old_config.get('context_learning_path', 'data/context_learning'),
                manim_docs_path=old_config.get('manim_docs_path', 'data/rag/manim_docs'),
                chroma_db_path=old_config.get('chroma_db_path', 'data/rag/chroma_db'),
                embedding_model=old_config.get('embedding_model', 'hf:ibm-granite/granite-embedding-30m-english'),
                
                # Monitoring settings
                enable_monitoring=old_config.get('enable_monitoring', True),
                use_langfuse=old_config.get('use_langfuse', True),
                enable_langfuse=old_config.get('use_langfuse', True),
                print_cost=old_config.get('print_cost', True),
                verbose=old_config.get('verbose', False),
                
                # Enhanced RAG settings
                use_enhanced_rag=old_config.get('use_enhanced_rag', True),
                max_scene_concurrency=old_config.get('max_scene_concurrency', 5)
            )
            
            logger.info("Successfully migrated system configuration to new format")
            return workflow_config
            
        except Exception as e:
            logger.error(f"Failed to migrate system configuration: {e}")
            raise ValueError(f"Configuration migration failed: {e}") from e
    
    @staticmethod
    def migrate_agent_config(old_agent_config: AgentConfig) -> Dict[str, Any]:
        """
        Migrate old AgentConfig to new format.
        
        Args:
            old_agent_config: Old agent configuration
            
        Returns:
            Dict[str, Any]: New configuration format
        """
        try:
            # Extract model configurations
            model_configs = {}
            if old_agent_config.planner_model:
                model_configs['planner_model'] = ConfigAdapter._parse_model_string(
                    old_agent_config.planner_model
                )
            if old_agent_config.scene_model:
                model_configs['code_model'] = ConfigAdapter._parse_model_string(
                    old_agent_config.scene_model
                )
            if old_agent_config.helper_model:
                model_configs['helper_model'] = ConfigAdapter._parse_model_string(
                    old_agent_config.helper_model
                )
            
            # Create new configuration
            new_config = {
                'name': old_agent_config.name,
                'max_retries': old_agent_config.max_retries,
                'timeout_seconds': old_agent_config.timeout_seconds,
                'enable_human_loop': old_agent_config.enable_human_loop,
                'temperature': old_agent_config.temperature,
                'print_cost': old_agent_config.print_cost,
                'verbose': old_agent_config.verbose,
                'tools': old_agent_config.tools,
                'model_configs': model_configs
            }
            
            logger.info(f"Successfully migrated agent configuration for {old_agent_config.name}")
            return new_config
            
        except Exception as e:
            logger.error(f"Failed to migrate agent configuration: {e}")
            raise ValueError(f"Agent configuration migration failed: {e}") from e
    
    @staticmethod
    def migrate_config_file(file_path: str, output_path: Optional[str] = None) -> str:
        """
        Migrate a configuration file from old to new format.
        
        Args:
            file_path: Path to old configuration file
            output_path: Optional path for new configuration file
            
        Returns:
            str: Path to migrated configuration file
        """
        try:
            file_path = Path(file_path)
            
            # Read old configuration
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r') as f:
                    old_config = json.load(f)
            elif file_path.suffix.lower() in ['.yml', '.yaml']:
                with open(file_path, 'r') as f:
                    old_config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")
            
            # Migrate configuration
            new_config = ConfigAdapter.migrate_system_config(old_config)
            
            # Determine output path
            if output_path is None:
                output_path = file_path.parent / f"{file_path.stem}_migrated{file_path.suffix}"
            else:
                output_path = Path(output_path)
            
            # Write new configuration
            config_dict = new_config.model_dump()
            
            if output_path.suffix.lower() == '.json':
                with open(output_path, 'w') as f:
                    json.dump(config_dict, f, indent=2, default=str)
            elif output_path.suffix.lower() in ['.yml', '.yaml']:
                with open(output_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False)
            else:
                # Default to JSON
                output_path = output_path.with_suffix('.json')
                with open(output_path, 'w') as f:
                    json.dump(config_dict, f, indent=2, default=str)
            
            logger.info(f"Successfully migrated configuration file: {file_path} -> {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to migrate configuration file {file_path}: {e}")
            raise ValueError(f"Configuration file migration failed: {e}") from e
    
    @staticmethod
    def validate_migrated_config(old_config: Dict[str, Any], new_config: WorkflowConfig) -> bool:
        """
        Validate that migrated configuration preserves essential settings.
        
        Args:
            old_config: Original configuration
            new_config: Migrated configuration
            
        Returns:
            bool: True if migration is valid
        """
        try:
            # Check essential settings
            if old_config.get('use_rag', True) != new_config.use_rag:
                logger.error("RAG setting mismatch in configuration migration")
                return False
            
            if old_config.get('max_retries', 3) != new_config.max_retries:
                logger.error("Max retries mismatch in configuration migration")
                return False
            
            if old_config.get('output_dir', 'output') != new_config.output_dir:
                logger.error("Output directory mismatch in configuration migration")
                return False
            
            if old_config.get('enable_caching', True) != new_config.enable_caching:
                logger.error("Caching setting mismatch in configuration migration")
                return False
            
            logger.info("Configuration migration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration migration validation failed: {e}")
            return False
    
    @staticmethod
    def create_migration_report(old_config: Dict[str, Any], new_config: WorkflowConfig) -> Dict[str, Any]:
        """
        Create a detailed migration report.
        
        Args:
            old_config: Original configuration
            new_config: Migrated configuration
            
        Returns:
            Dict[str, Any]: Migration report
        """
        report = {
            'migration_timestamp': datetime.now().isoformat(),
            'migration_status': 'success',
            'changes': [],
            'warnings': [],
            'errors': []
        }
        
        try:
            # Check for changes in key settings
            changes = []
            
            # Model configuration changes
            old_planner = old_config.get('planner_model', 'openrouter/anthropic/claude-3.5-sonnet')
            new_planner = f"{new_config.planner_model.provider}/{new_config.planner_model.model_name.split('/', 1)[1]}"
            if old_planner != new_planner:
                changes.append(f"Planner model: {old_planner} -> {new_planner}")
            
            # Feature flag changes
            if old_config.get('use_rag', True) != new_config.use_rag:
                changes.append(f"RAG usage: {old_config.get('use_rag', True)} -> {new_config.use_rag}")
            
            if old_config.get('use_visual_fix_code', False) != new_config.use_visual_analysis:
                changes.append(f"Visual analysis: {old_config.get('use_visual_fix_code', False)} -> {new_config.use_visual_analysis}")
            
            # Performance setting changes
            if old_config.get('max_retries', 3) != new_config.max_retries:
                changes.append(f"Max retries: {old_config.get('max_retries', 3)} -> {new_config.max_retries}")
            
            if old_config.get('max_scene_concurrency', 5) != new_config.max_concurrent_scenes:
                changes.append(f"Max concurrent scenes: {old_config.get('max_scene_concurrency', 5)} -> {new_config.max_concurrent_scenes}")
            
            report['changes'] = changes
            
            # Check for potential issues
            warnings = []
            
            # Check for deprecated settings
            deprecated_settings = [
                'max_topic_concurrency',
                'enable_quality_monitoring',
                'rag_performance_threshold'
            ]
            
            for setting in deprecated_settings:
                if setting in old_config:
                    warnings.append(f"Deprecated setting '{setting}' was present in old configuration")
            
            report['warnings'] = warnings
            
            logger.info("Successfully created migration report")
            
        except Exception as e:
            logger.error(f"Failed to create migration report: {e}")
            report['migration_status'] = 'error'
            report['errors'].append(str(e))
        
        return report
    
    @staticmethod
    def _extract_model_config(config: Dict[str, Any], key: str, 
                             default_provider: str, default_model: str) -> ModelConfig:
        """Extract model configuration from old config format."""
        model_value = config.get(key)
        
        if isinstance(model_value, dict):
            # Already a structured config
            return ModelConfig(
                provider=model_value.get('provider', default_provider),
                model_name=model_value.get('model_name', f"{default_provider}/{default_model}"),
                temperature=model_value.get('temperature', 0.7),
                max_tokens=model_value.get('max_tokens', 4000),
                timeout=model_value.get('timeout', 30)
            )
        elif isinstance(model_value, str):
            # Parse string format
            if '/' in model_value:
                provider, model = model_value.split('/', 1)
            else:
                provider = default_provider
                model = model_value
            
            return ModelConfig(
                provider=provider,
                model_name=f"{provider}/{model}",
                temperature=config.get('temperature', 0.7),
                max_tokens=config.get('max_tokens', 4000),
                timeout=config.get('timeout', 30)
            )
        else:
            # Use defaults
            return ModelConfig(
                provider=default_provider,
                model_name=f"{default_provider}/{default_model}",
                temperature=config.get('temperature', 0.7),
                max_tokens=config.get('max_tokens', 4000),
                timeout=config.get('timeout', 30)
            )
    
    @staticmethod
    def _parse_model_string(model_string: str) -> Dict[str, Any]:
        """Parse model string into configuration dictionary."""
        if '/' in model_string:
            provider, model = model_string.split('/', 1)
        else:
            provider = 'openrouter'
            model = model_string
        
        return {
            'provider': provider,
            'model_name': f"{provider}/{model}",
            'temperature': 0.7,
            'max_tokens': 4000,
            'timeout': 30
        }
    
    @staticmethod
    def backup_config_file(file_path: str) -> str:
        """
        Create a backup of the original configuration file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            str: Path to backup file
        """
        try:
            file_path = Path(file_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = file_path.parent / f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
            
            # Copy original file to backup
            import shutil
            shutil.copy2(file_path, backup_path)
            
            logger.info(f"Created backup of configuration file: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Failed to create backup of configuration file: {e}")
            raise ValueError(f"Configuration backup failed: {e}") from e
    
    @staticmethod
    def restore_config_from_backup(backup_path: str, target_path: str) -> bool:
        """
        Restore configuration from backup file.
        
        Args:
            backup_path: Path to backup file
            target_path: Path to restore to
            
        Returns:
            bool: True if restoration was successful
        """
        try:
            import shutil
            shutil.copy2(backup_path, target_path)
            
            logger.info(f"Successfully restored configuration from backup: {backup_path} -> {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore configuration from backup: {e}")
            return False
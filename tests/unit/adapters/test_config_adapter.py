"""
Unit tests for ConfigAdapter.

Tests the migration between old and new configuration formats.
"""

import pytest
import json
import yaml
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from src.langgraph_agents.adapters.config_adapter import ConfigAdapter
from src.langgraph_agents.models.config import WorkflowConfig, ModelConfig, RAGConfig
from src.langgraph_agents.state import AgentConfig


class TestConfigAdapter:
    """Test cases for ConfigAdapter functionality."""
    
    @pytest.fixture
    def sample_old_config(self):
        """Create a sample old configuration for testing."""
        return {
            # Model configurations
            'planner_model': 'openrouter/anthropic/claude-3.5-sonnet',
            'code_model': 'openrouter/anthropic/claude-3.5-sonnet',
            'helper_model': 'openrouter/anthropic/claude-3.5-sonnet',
            
            # Feature flags
            'use_rag': True,
            'use_visual_fix_code': False,
            'enable_caching': True,
            'use_context_learning': True,
            
            # Performance settings
            'max_retries': 3,
            'timeout_seconds': 300,
            'max_scene_concurrency': 5,
            'max_concurrent_renders': 4,
            
            # Quality settings
            'default_quality': 'medium',
            'use_gpu_acceleration': False,
            'preview_mode': False,
            
            # Directory settings
            'output_dir': 'output',
            'context_learning_path': 'data/context_learning',
            'manim_docs_path': 'data/rag/manim_docs',
            'chroma_db_path': 'data/rag/chroma_db',
            'embedding_model': 'hf:ibm-granite/granite-embedding-30m-english',
            
            # Monitoring settings
            'enable_monitoring': True,
            'use_langfuse': True,
            'print_cost': True,
            'verbose': False,
            
            # Enhanced RAG settings
            'use_enhanced_rag': True,
            'enable_rag_caching': True,
            'rag_cache_ttl': 3600,
            'rag_quality_threshold': 0.7,
            'chunk_size': 1000,
            'chunk_overlap': 200
        }
    
    @pytest.fixture
    def sample_agent_config(self):
        """Create a sample agent configuration for testing."""
        return AgentConfig(
            name='test_agent',
            model_config={'provider': 'openrouter', 'model': 'claude-3.5-sonnet'},
            tools=['test_tool'],
            max_retries=3,
            timeout_seconds=300,
            enable_human_loop=False,
            planner_model='openrouter/anthropic/claude-3.5-sonnet',
            scene_model='openrouter/anthropic/claude-3.5-sonnet',
            helper_model='openrouter/anthropic/claude-3.5-sonnet',
            temperature=0.7,
            print_cost=True,
            verbose=False
        )
    
    def test_migrate_system_config_success(self, sample_old_config):
        """Test successful migration of system configuration."""
        new_config = ConfigAdapter.migrate_system_config(sample_old_config)
        
        # Verify it's a WorkflowConfig instance
        assert isinstance(new_config, WorkflowConfig)
        
        # Verify model configurations
        assert new_config.planner_model.provider == 'openrouter'
        assert 'anthropic/claude-3.5-sonnet' in new_config.planner_model.model_name
        assert new_config.code_model.provider == 'openrouter'
        assert new_config.helper_model.provider == 'openrouter'
        
        # Verify feature flags
        assert new_config.use_rag == sample_old_config['use_rag']
        assert new_config.use_visual_analysis == sample_old_config['use_visual_fix_code']
        assert new_config.enable_caching == sample_old_config['enable_caching']
        assert new_config.use_context_learning == sample_old_config['use_context_learning']
        
        # Verify performance settings
        assert new_config.max_retries == sample_old_config['max_retries']
        assert new_config.max_concurrent_scenes == sample_old_config['max_scene_concurrency']
        assert new_config.max_concurrent_renders == sample_old_config['max_concurrent_renders']
        
        # Verify directory settings
        assert new_config.output_dir == sample_old_config['output_dir']
        assert new_config.chroma_db_path == sample_old_config['chroma_db_path']
        assert new_config.embedding_model == sample_old_config['embedding_model']
        
        # Verify RAG configuration
        assert isinstance(new_config.rag_config, RAGConfig)
        assert new_config.rag_config.enabled == sample_old_config['use_rag']
        assert new_config.rag_config.chunk_size == sample_old_config['chunk_size']
        assert new_config.rag_config.similarity_threshold == sample_old_config['rag_quality_threshold']
    
    def test_migrate_system_config_with_defaults(self):
        """Test migration with minimal configuration using defaults."""
        minimal_config = {
            'planner_model': 'openrouter/claude-3.5-sonnet'
        }
        
        new_config = ConfigAdapter.migrate_system_config(minimal_config)
        
        # Should use defaults for missing values
        assert new_config.use_rag is True  # Default
        assert new_config.max_retries == 3  # Default
        assert new_config.output_dir == 'output'  # Default
        assert new_config.enable_caching is True  # Default
    
    def test_migrate_system_config_error_handling(self):
        """Test error handling in system configuration migration."""
        invalid_config = None
        
        with pytest.raises(ValueError, match="Configuration migration failed"):
            ConfigAdapter.migrate_system_config(invalid_config)
    
    def test_migrate_agent_config_success(self, sample_agent_config):
        """Test successful migration of agent configuration."""
        new_config = ConfigAdapter.migrate_agent_config(sample_agent_config)
        
        # Verify basic fields
        assert new_config['name'] == sample_agent_config.name
        assert new_config['max_retries'] == sample_agent_config.max_retries
        assert new_config['timeout_seconds'] == sample_agent_config.timeout_seconds
        assert new_config['enable_human_loop'] == sample_agent_config.enable_human_loop
        assert new_config['temperature'] == sample_agent_config.temperature
        assert new_config['print_cost'] == sample_agent_config.print_cost
        assert new_config['verbose'] == sample_agent_config.verbose
        assert new_config['tools'] == sample_agent_config.tools
        
        # Verify model configurations
        assert 'model_configs' in new_config
        model_configs = new_config['model_configs']
        
        if sample_agent_config.planner_model:
            assert 'planner_model' in model_configs
            planner_config = model_configs['planner_model']
            assert planner_config['provider'] == 'openrouter'
            assert 'anthropic/claude-3.5-sonnet' in planner_config['model_name']
    
    def test_migrate_agent_config_error_handling(self):
        """Test error handling in agent configuration migration."""
        invalid_config = None
        
        with pytest.raises(ValueError, match="Agent configuration migration failed"):
            ConfigAdapter.migrate_agent_config(invalid_config)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('json.dump')
    def test_migrate_config_file_json(self, mock_json_dump, mock_json_load, mock_file, sample_old_config):
        """Test migration of JSON configuration file."""
        mock_json_load.return_value = sample_old_config
        
        # Test JSON file migration
        result_path = ConfigAdapter.migrate_config_file('config.json')
        
        # Verify file operations
        mock_json_load.assert_called_once()
        mock_json_dump.assert_called_once()
        
        # Verify result path
        assert result_path.endswith('_migrated.json')
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.safe_load')
    @patch('yaml.dump')
    def test_migrate_config_file_yaml(self, mock_yaml_dump, mock_yaml_load, mock_file, sample_old_config):
        """Test migration of YAML configuration file."""
        mock_yaml_load.return_value = sample_old_config
        
        # Test YAML file migration
        result_path = ConfigAdapter.migrate_config_file('config.yml')
        
        # Verify file operations
        mock_yaml_load.assert_called_once()
        mock_yaml_dump.assert_called_once()
        
        # Verify result path
        assert result_path.endswith('_migrated.yml')
    
    def test_migrate_config_file_unsupported_format(self):
        """Test migration with unsupported file format."""
        with pytest.raises(ValueError, match="Unsupported configuration file format"):
            ConfigAdapter.migrate_config_file('config.txt')
    
    def test_validate_migrated_config_success(self, sample_old_config):
        """Test successful validation of migrated configuration."""
        new_config = ConfigAdapter.migrate_system_config(sample_old_config)
        
        # Should validate successfully
        assert ConfigAdapter.validate_migrated_config(sample_old_config, new_config)
    
    def test_validate_migrated_config_failure(self, sample_old_config):
        """Test validation failure with mismatched configuration."""
        new_config = ConfigAdapter.migrate_system_config(sample_old_config)
        
        # Modify old config to create mismatch
        mismatched_old_config = sample_old_config.copy()
        mismatched_old_config['use_rag'] = not sample_old_config['use_rag']
        
        # Should fail validation
        assert not ConfigAdapter.validate_migrated_config(mismatched_old_config, new_config)
    
    def test_create_migration_report_success(self, sample_old_config):
        """Test creation of migration report."""
        new_config = ConfigAdapter.migrate_system_config(sample_old_config)
        
        report = ConfigAdapter.create_migration_report(sample_old_config, new_config)
        
        # Verify report structure
        assert 'migration_timestamp' in report
        assert 'migration_status' in report
        assert 'changes' in report
        assert 'warnings' in report
        assert 'errors' in report
        
        # Should be successful
        assert report['migration_status'] == 'success'
        assert isinstance(report['changes'], list)
        assert isinstance(report['warnings'], list)
        assert isinstance(report['errors'], list)
    
    def test_create_migration_report_with_changes(self):
        """Test migration report with configuration changes."""
        old_config = {
            'planner_model': 'openrouter/gpt-4',
            'use_rag': False,
            'max_retries': 5
        }
        
        new_config = ConfigAdapter.migrate_system_config(old_config)
        
        report = ConfigAdapter.create_migration_report(old_config, new_config)
        
        # Should detect changes
        assert len(report['changes']) > 0
        
        # Check for specific changes
        changes_text = ' '.join(report['changes'])
        assert 'Max retries' in changes_text or 'max_retries' in changes_text.lower()
    
    def test_create_migration_report_with_deprecated_settings(self):
        """Test migration report with deprecated settings."""
        old_config_with_deprecated = {
            'planner_model': 'openrouter/claude-3.5-sonnet',
            'max_topic_concurrency': 2,  # Deprecated
            'enable_quality_monitoring': True,  # Deprecated
            'rag_performance_threshold': 1.5  # Deprecated
        }
        
        new_config = ConfigAdapter.migrate_system_config(old_config_with_deprecated)
        
        report = ConfigAdapter.create_migration_report(old_config_with_deprecated, new_config)
        
        # Should detect deprecated settings
        assert len(report['warnings']) > 0
        
        # Check for specific warnings
        warnings_text = ' '.join(report['warnings'])
        assert 'max_topic_concurrency' in warnings_text
        assert 'enable_quality_monitoring' in warnings_text
        assert 'rag_performance_threshold' in warnings_text
    
    def test_extract_model_config_string_format(self):
        """Test model configuration extraction from string format."""
        config = {
            'planner_model': 'openrouter/anthropic/claude-3.5-sonnet',
            'temperature': 0.8,
            'max_tokens': 8000
        }
        
        model_config = ConfigAdapter._extract_model_config(
            config, 'planner_model', 'openrouter', 'claude-3.5-sonnet'
        )
        
        assert isinstance(model_config, ModelConfig)
        assert model_config.provider == 'openrouter'
        assert 'anthropic/claude-3.5-sonnet' in model_config.model_name
        assert model_config.temperature == 0.7  # Uses default, not from config
        assert model_config.max_tokens == 4000  # Uses default, not from config
    
    def test_extract_model_config_dict_format(self):
        """Test model configuration extraction from dictionary format."""
        config = {
            'planner_model': {
                'provider': 'openai',
                'model_name': 'gpt-4',
                'temperature': 0.5,
                'max_tokens': 2000,
                'timeout': 60
            }
        }
        
        model_config = ConfigAdapter._extract_model_config(
            config, 'planner_model', 'openrouter', 'claude-3.5-sonnet'
        )
        
        assert isinstance(model_config, ModelConfig)
        assert model_config.provider == 'openai'
        assert model_config.model_name == 'gpt-4'
        assert model_config.temperature == 0.5
        assert model_config.max_tokens == 2000
        assert model_config.timeout == 60
    
    def test_extract_model_config_defaults(self):
        """Test model configuration extraction with defaults."""
        config = {}
        
        model_config = ConfigAdapter._extract_model_config(
            config, 'missing_model', 'openrouter', 'claude-3.5-sonnet'
        )
        
        assert isinstance(model_config, ModelConfig)
        assert model_config.provider == 'openrouter'
        assert model_config.model_name == 'openrouter/claude-3.5-sonnet'
        assert model_config.temperature == 0.7
        assert model_config.max_tokens == 4000
        assert model_config.timeout == 30
    
    def test_parse_model_string_with_provider(self):
        """Test parsing model string with provider."""
        result = ConfigAdapter._parse_model_string('openrouter/anthropic/claude-3.5-sonnet')
        
        assert result['provider'] == 'openrouter'
        assert result['model_name'] == 'openrouter/anthropic/claude-3.5-sonnet'
        assert result['temperature'] == 0.7
        assert result['max_tokens'] == 4000
        assert result['timeout'] == 30
    
    def test_parse_model_string_without_provider(self):
        """Test parsing model string without provider."""
        result = ConfigAdapter._parse_model_string('gpt-4')
        
        assert result['provider'] == 'openrouter'  # Default
        assert result['model_name'] == 'openrouter/gpt-4'
        assert result['temperature'] == 0.7
        assert result['max_tokens'] == 4000
        assert result['timeout'] == 30
    
    @patch('shutil.copy2')
    def test_backup_config_file_success(self, mock_copy):
        """Test successful configuration file backup."""
        backup_path = ConfigAdapter.backup_config_file('config.json')
        
        # Should call copy2
        mock_copy.assert_called_once()
        
        # Should return backup path with timestamp
        assert 'config_backup_' in backup_path
        assert backup_path.endswith('.json')
    
    @patch('shutil.copy2', side_effect=Exception("Copy failed"))
    def test_backup_config_file_error(self, mock_copy):
        """Test configuration file backup error handling."""
        with pytest.raises(ValueError, match="Configuration backup failed"):
            ConfigAdapter.backup_config_file('config.json')
    
    @patch('shutil.copy2')
    def test_restore_config_from_backup_success(self, mock_copy):
        """Test successful configuration restoration from backup."""
        result = ConfigAdapter.restore_config_from_backup('backup.json', 'config.json')
        
        # Should call copy2
        mock_copy.assert_called_once_with('backup.json', 'config.json')
        
        # Should return True for success
        assert result is True
    
    @patch('shutil.copy2', side_effect=Exception("Restore failed"))
    def test_restore_config_from_backup_error(self, mock_copy):
        """Test configuration restoration error handling."""
        result = ConfigAdapter.restore_config_from_backup('backup.json', 'config.json')
        
        # Should return False for failure
        assert result is False
    
    def test_migration_preserves_essential_settings(self, sample_old_config):
        """Test that migration preserves all essential settings."""
        new_config = ConfigAdapter.migrate_system_config(sample_old_config)
        
        # Essential settings that must be preserved
        essential_mappings = [
            ('use_rag', 'use_rag'),
            ('enable_caching', 'enable_caching'),
            ('max_retries', 'max_retries'),
            ('output_dir', 'output_dir'),
            ('use_langfuse', 'use_langfuse'),
            ('default_quality', 'default_quality'),
            ('max_scene_concurrency', 'max_concurrent_scenes'),
            ('max_concurrent_renders', 'max_concurrent_renders')
        ]
        
        for old_key, new_key in essential_mappings:
            if old_key in sample_old_config:
                old_value = sample_old_config[old_key]
                new_value = getattr(new_config, new_key)
                assert old_value == new_value, f"Mismatch for {old_key} -> {new_key}: {old_value} != {new_value}"
    
    def test_migration_handles_complex_model_configs(self):
        """Test migration with complex model configurations."""
        complex_config = {
            'planner_model': {
                'provider': 'azure',
                'model_name': 'azure/gpt-4',
                'temperature': 0.3,
                'max_tokens': 6000,
                'timeout': 45
            },
            'code_model': 'openai/gpt-4-turbo',
            'helper_model': 'gemini/gemini-pro'
        }
        
        new_config = ConfigAdapter.migrate_system_config(complex_config)
        
        # Verify complex model config preservation
        assert new_config.planner_model.provider == 'azure'
        assert new_config.planner_model.model_name == 'azure/gpt-4'
        assert new_config.planner_model.temperature == 0.3
        assert new_config.planner_model.max_tokens == 6000
        assert new_config.planner_model.timeout == 45
        
        # Verify string model parsing
        assert new_config.code_model.provider == 'openai'
        assert 'gpt-4-turbo' in new_config.code_model.model_name
        
        assert new_config.helper_model.provider == 'gemini'
        assert 'gemini-pro' in new_config.helper_model.model_name
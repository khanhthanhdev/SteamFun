"""
Unit tests for checkpoint manager functionality.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock

from src.langgraph_agents.checkpointing.checkpoint_manager import (
    CheckpointManager,
    CheckpointConfig,
    CheckpointBackend,
    create_checkpoint_manager
)
from src.langgraph_agents.models.config import WorkflowConfig


class TestCheckpointConfig:
    """Test checkpoint configuration."""
    
    def test_checkpoint_config_creation(self):
        """Test creating checkpoint configuration."""
        config = CheckpointConfig(
            backend=CheckpointBackend.MEMORY,
            memory_max_size=500,
            enable_compression=False
        )
        
        assert config.backend == CheckpointBackend.MEMORY
        assert config.memory_max_size == 500
        assert config.enable_compression is False
        assert config.checkpoint_ttl == 86400  # Default value
    
    def test_checkpoint_config_from_environment(self):
        """Test creating configuration from environment variables."""
        with patch.dict(os.environ, {
            'CHECKPOINT_BACKEND': 'postgres',
            'POSTGRES_CONNECTION_STRING': 'postgresql://test:test@localhost/test',
            'POSTGRES_POOL_SIZE': '15',
            'MEMORY_CHECKPOINT_MAX_SIZE': '2000',
            'CHECKPOINT_COMPRESSION': 'false',
            'CHECKPOINT_TTL': '7200'
        }):
            config = CheckpointConfig.from_environment()
            
            assert config.backend == CheckpointBackend.POSTGRES
            assert config.postgres_connection_string == 'postgresql://test:test@localhost/test'
            assert config.postgres_pool_size == 15
            assert config.memory_max_size == 2000
            assert config.enable_compression is False
            assert config.checkpoint_ttl == 7200
    
    def test_checkpoint_config_from_environment_defaults(self):
        """Test configuration from environment with defaults."""
        # Clear relevant environment variables
        env_vars_to_clear = [
            'CHECKPOINT_BACKEND', 'POSTGRES_CONNECTION_STRING',
            'POSTGRES_POOL_SIZE', 'MEMORY_CHECKPOINT_MAX_SIZE',
            'CHECKPOINT_COMPRESSION', 'CHECKPOINT_TTL'
        ]
        
        with patch.dict(os.environ, {}, clear=True):
            config = CheckpointConfig.from_environment()
            
            assert config.backend == CheckpointBackend.AUTO
            assert config.postgres_connection_string is None
            assert config.postgres_pool_size == 10
            assert config.memory_max_size == 1000
            assert config.enable_compression is True
            assert config.checkpoint_ttl == 86400


class TestCheckpointManager:
    """Test checkpoint manager functionality."""
    
    @pytest.fixture
    def workflow_config(self):
        """Create a test workflow configuration."""
        return WorkflowConfig()
    
    @pytest.fixture
    def memory_config(self):
        """Create memory checkpoint configuration."""
        return CheckpointConfig(
            backend=CheckpointBackend.MEMORY,
            memory_max_size=100,
            enable_compression=True
        )
    
    @pytest.fixture
    def postgres_config(self):
        """Create PostgreSQL checkpoint configuration."""
        return CheckpointConfig(
            backend=CheckpointBackend.POSTGRES,
            postgres_connection_string="postgresql://test:test@localhost/test",
            postgres_pool_size=5
        )
    
    def test_memory_checkpoint_manager_creation(self, memory_config):
        """Test creating checkpoint manager with memory backend."""
        manager = CheckpointManager(memory_config)
        
        assert manager is not None
        assert manager.backend_type == CheckpointBackend.MEMORY
        assert not manager.is_persistent()
        assert manager.checkpointer is not None
    
    @patch('src.langgraph_agents.checkpointing.checkpoint_manager.POSTGRES_AVAILABLE', False)
    def test_postgres_unavailable_fallback(self, postgres_config):
        """Test fallback to memory when PostgreSQL is unavailable."""
        manager = CheckpointManager(postgres_config)
        
        # Should fall back to memory
        assert manager.backend_type == CheckpointBackend.MEMORY
        assert not manager.is_persistent()
    
    @patch('src.langgraph_agents.checkpointing.checkpoint_manager.POSTGRES_AVAILABLE', True)
    @patch('src.langgraph_agents.checkpointing.checkpoint_manager.PostgresSaver')
    def test_postgres_checkpoint_manager_creation(self, mock_postgres_saver, postgres_config):
        """Test creating checkpoint manager with PostgreSQL backend."""
        # Mock PostgresSaver
        mock_checkpointer = Mock()
        mock_postgres_saver.from_conn_string.return_value = mock_checkpointer
        mock_checkpointer.setup.return_value = None
        
        manager = CheckpointManager(postgres_config)
        
        assert manager.backend_type == CheckpointBackend.POSTGRES
        assert manager.is_persistent()
        assert manager.checkpointer == mock_checkpointer
        
        # Verify PostgresSaver was called correctly
        mock_postgres_saver.from_conn_string.assert_called_once_with(
            postgres_config.postgres_connection_string,
            pool_size=postgres_config.postgres_pool_size,
            max_overflow=postgres_config.postgres_max_overflow
        )
    
    @patch('src.langgraph_agents.checkpointing.checkpoint_manager.POSTGRES_AVAILABLE', True)
    @patch('src.langgraph_agents.checkpointing.checkpoint_manager.PostgresSaver')
    def test_postgres_connection_failure_fallback(self, mock_postgres_saver, postgres_config):
        """Test fallback to memory when PostgreSQL connection fails."""
        # Mock PostgresSaver to raise exception
        mock_postgres_saver.from_conn_string.side_effect = Exception("Connection failed")
        
        manager = CheckpointManager(postgres_config)
        
        # Should fall back to memory
        assert manager.backend_type == CheckpointBackend.MEMORY
        assert not manager.is_persistent()
    
    def test_auto_backend_selection_memory(self):
        """Test automatic backend selection falling back to memory."""
        config = CheckpointConfig(backend=CheckpointBackend.AUTO)
        
        with patch('src.langgraph_agents.checkpointing.checkpoint_manager.POSTGRES_AVAILABLE', False):
            manager = CheckpointManager(config)
            
            assert manager.backend_type == CheckpointBackend.MEMORY
            assert not manager.is_persistent()
    
    @patch('src.langgraph_agents.checkpointing.checkpoint_manager.POSTGRES_AVAILABLE', True)
    @patch('src.langgraph_agents.checkpointing.checkpoint_manager.PostgresSaver')
    def test_auto_backend_selection_postgres(self, mock_postgres_saver):
        """Test automatic backend selection choosing PostgreSQL."""
        config = CheckpointConfig(
            backend=CheckpointBackend.AUTO,
            postgres_connection_string="postgresql://test:test@localhost/test"
        )
        
        # Mock successful PostgreSQL setup
        mock_checkpointer = Mock()
        mock_postgres_saver.from_conn_string.return_value = mock_checkpointer
        mock_checkpointer.setup.return_value = None
        
        manager = CheckpointManager(config)
        
        assert manager.backend_type == CheckpointBackend.POSTGRES
        assert manager.is_persistent()
    
    def test_checkpoint_info(self, memory_config):
        """Test getting checkpoint information."""
        manager = CheckpointManager(memory_config)
        
        info = manager.get_checkpoint_info()
        
        assert info["backend"] == "memory"
        assert info["persistent"] is False
        assert "postgres_available" in info
        assert "connection_configured" in info
        assert info["compression_enabled"] == memory_config.enable_compression
        assert info["ttl_seconds"] == memory_config.checkpoint_ttl
    
    @pytest.mark.asyncio
    async def test_cleanup_old_checkpoints_memory(self, memory_config):
        """Test cleanup with memory backend."""
        manager = CheckpointManager(memory_config)
        
        # Memory backend doesn't need cleanup
        count = await manager.cleanup_old_checkpoints(3600)
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_get_checkpoint_stats_memory(self, memory_config):
        """Test getting stats with memory backend."""
        manager = CheckpointManager(memory_config)
        
        stats = await manager.get_checkpoint_stats()
        
        assert "backend" in stats
        assert stats["backend"] == "memory"
        assert "total_checkpoints" in stats
    
    def test_invalid_backend_raises_error(self):
        """Test that invalid backend raises error."""
        with pytest.raises(ValueError, match="Unknown checkpoint backend"):
            # Create config with invalid backend by bypassing enum validation
            config = CheckpointConfig.__new__(CheckpointConfig)
            config.backend = "invalid_backend"
            CheckpointManager(config)


class TestCheckpointManagerFactory:
    """Test checkpoint manager factory function."""
    
    @pytest.fixture
    def workflow_config(self):
        """Create a test workflow configuration."""
        return WorkflowConfig()
    
    def test_create_checkpoint_manager_with_config(self, workflow_config):
        """Test creating checkpoint manager with explicit config."""
        checkpoint_config = CheckpointConfig(
            backend=CheckpointBackend.MEMORY,
            memory_max_size=200
        )
        
        manager = create_checkpoint_manager(workflow_config, checkpoint_config)
        
        assert manager is not None
        assert manager.backend_type == CheckpointBackend.MEMORY
    
    def test_create_checkpoint_manager_from_environment(self, workflow_config):
        """Test creating checkpoint manager from environment."""
        with patch('src.langgraph_agents.checkpointing.checkpoint_manager.CheckpointConfig.from_environment') as mock_from_env:
            mock_config = CheckpointConfig(backend=CheckpointBackend.MEMORY)
            mock_from_env.return_value = mock_config
            
            manager = create_checkpoint_manager(workflow_config)
            
            assert manager is not None
            mock_from_env.assert_called_once()
    
    def test_create_checkpoint_manager_with_workflow_overrides(self, workflow_config):
        """Test creating checkpoint manager with workflow config overrides."""
        # Add checkpoint-specific attributes to workflow config
        workflow_config.checkpoint_backend = "postgres"
        workflow_config.postgres_connection_string = "postgresql://test:test@localhost/test"
        
        with patch('src.langgraph_agents.checkpointing.checkpoint_manager.CheckpointConfig.from_environment') as mock_from_env:
            mock_config = CheckpointConfig(backend=CheckpointBackend.MEMORY)
            mock_from_env.return_value = mock_config
            
            manager = create_checkpoint_manager(workflow_config)
            
            # Should use workflow config overrides
            assert mock_config.backend == CheckpointBackend.POSTGRES
            assert mock_config.postgres_connection_string == "postgresql://test:test@localhost/test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
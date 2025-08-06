"""
Checkpoint manager for handling different checkpointing strategies.

This module provides a unified interface for managing checkpoints across
different storage backends (memory, PostgreSQL) with automatic fallback
and configuration-based selection.
"""

import logging
import os
from typing import Optional, Dict, Any, Union
from enum import Enum

from langgraph.checkpoint.memory import MemorySaver

# PostgreSQL checkpointer is optional and may not be available
try:
    from langgraph.checkpoint.postgres import PostgresSaver
    POSTGRES_AVAILABLE = True
except ImportError:
    PostgresSaver = None
    POSTGRES_AVAILABLE = False

from ..models.config import WorkflowConfig

logger = logging.getLogger(__name__)


class CheckpointBackend(Enum):
    """Supported checkpoint backends."""
    MEMORY = "memory"
    POSTGRES = "postgres"
    AUTO = "auto"


class CheckpointConfig:
    """Configuration for checkpointing system."""
    
    def __init__(
        self,
        backend: CheckpointBackend = CheckpointBackend.AUTO,
        postgres_connection_string: Optional[str] = None,
        postgres_pool_size: int = 10,
        postgres_max_overflow: int = 20,
        memory_max_size: int = 1000,
        enable_compression: bool = True,
        checkpoint_ttl: int = 86400  # 24 hours
    ):
        self.backend = backend
        self.postgres_connection_string = postgres_connection_string
        self.postgres_pool_size = postgres_pool_size
        self.postgres_max_overflow = postgres_max_overflow
        self.memory_max_size = memory_max_size
        self.enable_compression = enable_compression
        self.checkpoint_ttl = checkpoint_ttl
    
    @classmethod
    def from_environment(cls) -> 'CheckpointConfig':
        """Create checkpoint configuration from environment variables."""
        backend_str = os.getenv('CHECKPOINT_BACKEND', 'auto').lower()
        backend = CheckpointBackend(backend_str) if backend_str in [b.value for b in CheckpointBackend] else CheckpointBackend.AUTO
        
        return cls(
            backend=backend,
            postgres_connection_string=os.getenv('POSTGRES_CONNECTION_STRING'),
            postgres_pool_size=int(os.getenv('POSTGRES_POOL_SIZE', '10')),
            postgres_max_overflow=int(os.getenv('POSTGRES_MAX_OVERFLOW', '20')),
            memory_max_size=int(os.getenv('MEMORY_CHECKPOINT_MAX_SIZE', '1000')),
            enable_compression=os.getenv('CHECKPOINT_COMPRESSION', 'true').lower() == 'true',
            checkpoint_ttl=int(os.getenv('CHECKPOINT_TTL', '86400'))
        )


class CheckpointManager:
    """
    Manager for handling different checkpointing strategies.
    
    This class provides a unified interface for checkpointing that can
    automatically select the appropriate backend based on configuration
    and availability.
    """
    
    def __init__(self, config: CheckpointConfig):
        """
        Initialize checkpoint manager.
        
        Args:
            config: Checkpoint configuration
        """
        self.config = config
        self._checkpointer = None
        self._backend_type = None
        self._initialize_checkpointer()
    
    def _initialize_checkpointer(self) -> None:
        """Initialize the appropriate checkpointer based on configuration."""
        if self.config.backend == CheckpointBackend.POSTGRES:
            self._checkpointer = self._create_postgres_checkpointer()
            self._backend_type = CheckpointBackend.POSTGRES
        elif self.config.backend == CheckpointBackend.MEMORY:
            self._checkpointer = self._create_memory_checkpointer()
            self._backend_type = CheckpointBackend.MEMORY
        elif self.config.backend == CheckpointBackend.AUTO:
            # Try PostgreSQL first, fallback to memory
            postgres_checkpointer = self._create_postgres_checkpointer()
            if postgres_checkpointer:
                self._checkpointer = postgres_checkpointer
                self._backend_type = CheckpointBackend.POSTGRES
                logger.info("Using PostgreSQL checkpointer (auto-selected)")
            else:
                self._checkpointer = self._create_memory_checkpointer()
                self._backend_type = CheckpointBackend.MEMORY
                logger.info("Using memory checkpointer (auto-selected)")
        else:
            raise ValueError(f"Unknown checkpoint backend: {self.config.backend}")
    
    def _create_postgres_checkpointer(self) -> Optional[Any]:
        """Create PostgreSQL checkpointer if available and configured."""
        if not POSTGRES_AVAILABLE:
            logger.warning("PostgreSQL checkpointer requested but langgraph[postgres] not installed")
            return None
        
        if not self.config.postgres_connection_string:
            logger.warning("PostgreSQL checkpointer requested but no connection string provided")
            return None
        
        try:
            logger.info("Creating PostgreSQL checkpointer")
            checkpointer = PostgresSaver.from_conn_string(
                self.config.postgres_connection_string,
                pool_size=self.config.postgres_pool_size,
                max_overflow=self.config.postgres_max_overflow
            )
            
            # Test the connection
            checkpointer.setup()
            logger.info("PostgreSQL checkpointer created successfully")
            return checkpointer
            
        except Exception as e:
            logger.error(f"Failed to create PostgreSQL checkpointer: {e}")
            return None
    
    def _create_memory_checkpointer(self) -> MemorySaver:
        """Create memory checkpointer."""
        logger.info("Creating memory checkpointer")
        return MemorySaver()
    
    @property
    def checkpointer(self) -> Union[MemorySaver, Any]:
        """Get the active checkpointer instance."""
        return self._checkpointer
    
    @property
    def backend_type(self) -> CheckpointBackend:
        """Get the active backend type."""
        return self._backend_type
    
    def is_persistent(self) -> bool:
        """Check if the current checkpointer provides persistent storage."""
        return self._backend_type == CheckpointBackend.POSTGRES
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about the current checkpointing configuration."""
        return {
            "backend": self._backend_type.value,
            "persistent": self.is_persistent(),
            "postgres_available": POSTGRES_AVAILABLE,
            "connection_configured": bool(self.config.postgres_connection_string),
            "compression_enabled": self.config.enable_compression,
            "ttl_seconds": self.config.checkpoint_ttl
        }
    
    async def cleanup_old_checkpoints(self, max_age_seconds: Optional[int] = None) -> int:
        """
        Clean up old checkpoints based on TTL.
        
        Args:
            max_age_seconds: Maximum age for checkpoints (uses config TTL if not provided)
            
        Returns:
            Number of checkpoints cleaned up
        """
        if max_age_seconds is None:
            max_age_seconds = self.config.checkpoint_ttl
        
        if self._backend_type == CheckpointBackend.POSTGRES and hasattr(self._checkpointer, 'cleanup'):
            try:
                count = await self._checkpointer.cleanup(max_age_seconds)
                logger.info(f"Cleaned up {count} old checkpoints from PostgreSQL")
                return count
            except Exception as e:
                logger.error(f"Failed to cleanup PostgreSQL checkpoints: {e}")
                return 0
        elif self._backend_type == CheckpointBackend.MEMORY:
            # Memory checkpointer doesn't need cleanup (handled by GC)
            logger.debug("Memory checkpointer doesn't require explicit cleanup")
            return 0
        
        return 0
    
    async def get_checkpoint_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored checkpoints.
        
        Returns:
            Dictionary with checkpoint statistics
        """
        stats = {
            "backend": self._backend_type.value,
            "total_checkpoints": 0,
            "total_size_bytes": 0,
            "oldest_checkpoint": None,
            "newest_checkpoint": None
        }
        
        if self._backend_type == CheckpointBackend.POSTGRES and hasattr(self._checkpointer, 'get_stats'):
            try:
                postgres_stats = await self._checkpointer.get_stats()
                stats.update(postgres_stats)
            except Exception as e:
                logger.error(f"Failed to get PostgreSQL checkpoint stats: {e}")
        elif self._backend_type == CheckpointBackend.MEMORY:
            # For memory checkpointer, we can get basic stats
            if hasattr(self._checkpointer, 'storage'):
                stats["total_checkpoints"] = len(self._checkpointer.storage)
        
        return stats


def create_checkpoint_manager(
    workflow_config: WorkflowConfig,
    checkpoint_config: Optional[CheckpointConfig] = None
) -> CheckpointManager:
    """
    Create a checkpoint manager with appropriate configuration.
    
    Args:
        workflow_config: Workflow configuration
        checkpoint_config: Optional checkpoint configuration (uses environment if not provided)
        
    Returns:
        Configured CheckpointManager instance
    """
    if checkpoint_config is None:
        checkpoint_config = CheckpointConfig.from_environment()
    
    # Override with workflow-specific settings if available
    if hasattr(workflow_config, 'checkpoint_backend'):
        checkpoint_config.backend = CheckpointBackend(workflow_config.checkpoint_backend)
    
    if hasattr(workflow_config, 'postgres_connection_string'):
        checkpoint_config.postgres_connection_string = workflow_config.postgres_connection_string
    
    return CheckpointManager(checkpoint_config)
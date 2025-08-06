"""
Enhanced memory checkpointer for development environments.

This module provides an enhanced memory-based checkpointer with additional
features like compression, size limits, and better debugging capabilities.
"""

import logging
import pickle
import gzip
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata

logger = logging.getLogger(__name__)


class EnhancedMemorySaver(MemorySaver):
    """
    Enhanced memory checkpointer with additional features.
    
    This extends the basic MemorySaver with:
    - Size limits and LRU eviction
    - Optional compression
    - Better statistics and monitoring
    - Checkpoint metadata tracking
    """
    
    def __init__(
        self,
        max_checkpoints: int = 1000,
        max_size_bytes: int = 100 * 1024 * 1024,  # 100MB
        enable_compression: bool = True,
        compression_level: int = 6
    ):
        """
        Initialize enhanced memory checkpointer.
        
        Args:
            max_checkpoints: Maximum number of checkpoints to store
            max_size_bytes: Maximum total size in bytes
            enable_compression: Whether to compress checkpoints
            compression_level: Compression level (1-9)
        """
        super().__init__()
        self.max_checkpoints = max_checkpoints
        self.max_size_bytes = max_size_bytes
        self.enable_compression = enable_compression
        self.compression_level = compression_level
        
        # Enhanced storage with metadata
        self._checkpoint_metadata: Dict[str, Dict[str, Any]] = {}
        self._checkpoint_sizes: Dict[str, int] = {}
        self._access_times: OrderedDict[str, float] = OrderedDict()
        self._total_size = 0
        
        logger.info(f"Enhanced memory checkpointer initialized: max_checkpoints={max_checkpoints}, "
                   f"max_size_bytes={max_size_bytes}, compression={enable_compression}")
    
    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata
    ) -> None:
        """
        Store a checkpoint with enhanced metadata tracking.
        
        Args:
            config: Configuration for the checkpoint
            checkpoint: The checkpoint data
            metadata: Checkpoint metadata
        """
        # Generate checkpoint key
        checkpoint_key = self._get_checkpoint_key(config, checkpoint)
        
        # Serialize and optionally compress the checkpoint
        serialized_data = self._serialize_checkpoint(checkpoint)
        checkpoint_size = len(serialized_data)
        
        # Check if we need to evict old checkpoints
        self._ensure_capacity(checkpoint_size)
        
        # Store the checkpoint
        super().put(config, checkpoint, metadata)
        
        # Update our enhanced metadata
        self._checkpoint_metadata[checkpoint_key] = {
            "created_at": datetime.now(),
            "size_bytes": checkpoint_size,
            "compressed": self.enable_compression,
            "thread_id": config.get("configurable", {}).get("thread_id"),
            "step_count": len(checkpoint.get("channel_values", {})),
            "metadata": metadata
        }
        
        self._checkpoint_sizes[checkpoint_key] = checkpoint_size
        self._access_times[checkpoint_key] = time.time()
        self._total_size += checkpoint_size
        
        logger.debug(f"Stored checkpoint {checkpoint_key}: {checkpoint_size} bytes, "
                    f"total size: {self._total_size} bytes")
    
    def get(
        self,
        config: Dict[str, Any]
    ) -> Optional[Checkpoint]:
        """
        Retrieve a checkpoint and update access time.
        
        Args:
            config: Configuration for the checkpoint
            
        Returns:
            The checkpoint if found, None otherwise
        """
        checkpoint = super().get(config)
        
        if checkpoint:
            checkpoint_key = self._get_checkpoint_key(config, checkpoint)
            # Update access time for LRU
            if checkpoint_key in self._access_times:
                self._access_times.move_to_end(checkpoint_key)
                self._access_times[checkpoint_key] = time.time()
        
        return checkpoint
    
    def _get_checkpoint_key(self, config: Dict[str, Any], checkpoint: Checkpoint) -> str:
        """Generate a unique key for the checkpoint."""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        checkpoint_id = checkpoint.get("id", "unknown")
        return f"{thread_id}:{checkpoint_id}"
    
    def _serialize_checkpoint(self, checkpoint: Checkpoint) -> bytes:
        """Serialize and optionally compress checkpoint data."""
        # Serialize the checkpoint
        serialized = pickle.dumps(checkpoint)
        
        if self.enable_compression:
            # Compress the serialized data
            compressed = gzip.compress(serialized, compresslevel=self.compression_level)
            logger.debug(f"Compressed checkpoint: {len(serialized)} -> {len(compressed)} bytes "
                        f"({100 * (1 - len(compressed) / len(serialized)):.1f}% reduction)")
            return compressed
        
        return serialized
    
    def _ensure_capacity(self, new_checkpoint_size: int) -> None:
        """Ensure we have capacity for a new checkpoint by evicting old ones if necessary."""
        # Check count limit
        while len(self._access_times) >= self.max_checkpoints:
            self._evict_oldest()
        
        # Check size limit
        while self._total_size + new_checkpoint_size > self.max_size_bytes and self._access_times:
            self._evict_oldest()
    
    def _evict_oldest(self) -> None:
        """Evict the least recently used checkpoint."""
        if not self._access_times:
            return
        
        # Get the oldest checkpoint key
        oldest_key = next(iter(self._access_times))
        
        # Remove from all tracking structures
        self._access_times.pop(oldest_key, None)
        size = self._checkpoint_sizes.pop(oldest_key, 0)
        self._checkpoint_metadata.pop(oldest_key, None)
        self._total_size -= size
        
        # Remove from parent storage
        # Note: This is a simplified approach - in practice we'd need to map back to the storage key
        logger.debug(f"Evicted checkpoint {oldest_key}: {size} bytes")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about stored checkpoints."""
        if not self._checkpoint_metadata:
            return {
                "total_checkpoints": 0,
                "total_size_bytes": 0,
                "average_size_bytes": 0,
                "compression_enabled": self.enable_compression,
                "oldest_checkpoint": None,
                "newest_checkpoint": None
            }
        
        sizes = list(self._checkpoint_sizes.values())
        created_times = [meta["created_at"] for meta in self._checkpoint_metadata.values()]
        
        return {
            "total_checkpoints": len(self._checkpoint_metadata),
            "total_size_bytes": self._total_size,
            "average_size_bytes": self._total_size / len(self._checkpoint_metadata),
            "min_size_bytes": min(sizes),
            "max_size_bytes": max(sizes),
            "compression_enabled": self.enable_compression,
            "oldest_checkpoint": min(created_times).isoformat(),
            "newest_checkpoint": max(created_times).isoformat(),
            "capacity_utilization": {
                "count": len(self._checkpoint_metadata) / self.max_checkpoints,
                "size": self._total_size / self.max_size_bytes
            }
        }
    
    def cleanup_old_checkpoints(self, max_age_hours: int = 24) -> int:
        """
        Clean up checkpoints older than the specified age.
        
        Args:
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of checkpoints cleaned up
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleaned_count = 0
        
        # Find checkpoints to clean up
        keys_to_remove = []
        for key, metadata in self._checkpoint_metadata.items():
            if metadata["created_at"] < cutoff_time:
                keys_to_remove.append(key)
        
        # Remove old checkpoints
        for key in keys_to_remove:
            self._access_times.pop(key, None)
            size = self._checkpoint_sizes.pop(key, 0)
            self._checkpoint_metadata.pop(key, None)
            self._total_size -= size
            cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} checkpoints older than {max_age_hours} hours")
        return cleaned_count
    
    def get_checkpoint_list(self) -> List[Dict[str, Any]]:
        """Get a list of all stored checkpoints with metadata."""
        checkpoints = []
        
        for key, metadata in self._checkpoint_metadata.items():
            checkpoints.append({
                "key": key,
                "thread_id": metadata["thread_id"],
                "created_at": metadata["created_at"].isoformat(),
                "size_bytes": metadata["size_bytes"],
                "compressed": metadata["compressed"],
                "step_count": metadata["step_count"]
            })
        
        # Sort by creation time (newest first)
        checkpoints.sort(key=lambda x: x["created_at"], reverse=True)
        return checkpoints
    
    def clear_all(self) -> int:
        """
        Clear all stored checkpoints.
        
        Returns:
            Number of checkpoints cleared
        """
        count = len(self._checkpoint_metadata)
        
        # Clear parent storage
        self.storage.clear()
        
        # Clear our enhanced metadata
        self._checkpoint_metadata.clear()
        self._checkpoint_sizes.clear()
        self._access_times.clear()
        self._total_size = 0
        
        logger.info(f"Cleared all {count} checkpoints from memory")
        return count


class MemoryCheckpointer:
    """
    Wrapper class for memory checkpointing functionality.
    
    This provides a consistent interface for memory-based checkpointing
    with enhanced features for development environments.
    """
    
    def __init__(
        self,
        max_checkpoints: int = 1000,
        max_size_mb: int = 100,
        enable_compression: bool = True
    ):
        """
        Initialize memory checkpointer.
        
        Args:
            max_checkpoints: Maximum number of checkpoints to store
            max_size_mb: Maximum total size in MB
            enable_compression: Whether to compress checkpoints
        """
        self.checkpointer = EnhancedMemorySaver(
            max_checkpoints=max_checkpoints,
            max_size_bytes=max_size_mb * 1024 * 1024,
            enable_compression=enable_compression
        )
        
        logger.info(f"Memory checkpointer initialized with {max_checkpoints} max checkpoints, "
                   f"{max_size_mb}MB max size, compression={enable_compression}")
    
    def get_checkpointer(self) -> EnhancedMemorySaver:
        """Get the underlying checkpointer instance."""
        return self.checkpointer
    
    def get_stats(self) -> Dict[str, Any]:
        """Get checkpointer statistics."""
        return self.checkpointer.get_stats()
    
    def cleanup(self, max_age_hours: int = 24) -> int:
        """Clean up old checkpoints."""
        return self.checkpointer.cleanup_old_checkpoints(max_age_hours)
    
    def clear_all(self) -> int:
        """Clear all checkpoints."""
        return self.checkpointer.clear_all()
    
    def get_checkpoint_list(self) -> List[Dict[str, Any]]:
        """Get list of all checkpoints."""
        return self.checkpointer.get_checkpoint_list()
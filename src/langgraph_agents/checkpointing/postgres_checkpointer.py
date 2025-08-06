"""
PostgreSQL checkpointer for production environments.

This module provides PostgreSQL-based checkpointing with enhanced features
for production use including connection pooling, retry logic, and monitoring.
"""

import logging
import os
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

# PostgreSQL checkpointer is optional and may not be available
try:
    from langgraph.checkpoint.postgres import PostgresSaver
    import asyncpg
    POSTGRES_AVAILABLE = True
except ImportError:
    PostgresSaver = None
    asyncpg = None
    POSTGRES_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnhancedPostgresSaver:
    """
    Enhanced PostgreSQL checkpointer with additional production features.
    
    This wraps the basic PostgresSaver with:
    - Connection health monitoring
    - Automatic retry logic
    - Enhanced statistics and monitoring
    - Cleanup and maintenance operations
    """
    
    def __init__(
        self,
        connection_string: str,
        pool_size: int = 10,
        max_overflow: int = 20,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        health_check_interval: int = 300  # 5 minutes
    ):
        """
        Initialize enhanced PostgreSQL checkpointer.
        
        Args:
            connection_string: PostgreSQL connection string
            pool_size: Connection pool size
            max_overflow: Maximum pool overflow
            retry_attempts: Number of retry attempts for failed operations
            retry_delay: Delay between retry attempts in seconds
            health_check_interval: Health check interval in seconds
        """
        if not POSTGRES_AVAILABLE:
            raise ImportError("PostgreSQL checkpointer requires langgraph[postgres] and asyncpg")
        
        self.connection_string = connection_string
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.health_check_interval = health_check_interval
        
        self._checkpointer = None
        self._connection_pool = None
        self._last_health_check = None
        self._is_healthy = False
        
        logger.info(f"Enhanced PostgreSQL checkpointer initialized: pool_size={pool_size}, "
                   f"max_overflow={max_overflow}, retry_attempts={retry_attempts}")
    
    async def initialize(self) -> None:
        """Initialize the PostgreSQL checkpointer and connection pool."""
        try:
            logger.info("Initializing PostgreSQL checkpointer")
            
            # Create the basic checkpointer
            self._checkpointer = PostgresSaver.from_conn_string(
                self.connection_string,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow
            )
            
            # Set up the database schema
            await self._checkpointer.setup()
            
            # Create connection pool for direct database operations
            self._connection_pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=2,
                max_size=self.pool_size
            )
            
            # Perform initial health check
            await self._health_check()
            
            logger.info("PostgreSQL checkpointer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL checkpointer: {e}")
            raise
    
    async def close(self) -> None:
        """Close the checkpointer and connection pool."""
        if self._connection_pool:
            await self._connection_pool.close()
            self._connection_pool = None
        
        if self._checkpointer:
            # Close the checkpointer if it has a close method
            if hasattr(self._checkpointer, 'close'):
                await self._checkpointer.close()
            self._checkpointer = None
        
        logger.info("PostgreSQL checkpointer closed")
    
    async def _health_check(self) -> bool:
        """Perform a health check on the PostgreSQL connection."""
        try:
            if not self._connection_pool:
                self._is_healthy = False
                return False
            
            async with self._connection_pool.acquire() as conn:
                # Simple query to test connection
                await conn.fetchval("SELECT 1")
                
            self._is_healthy = True
            self._last_health_check = datetime.now()
            logger.debug("PostgreSQL health check passed")
            return True
            
        except Exception as e:
            self._is_healthy = False
            logger.error(f"PostgreSQL health check failed: {e}")
            return False
    
    async def _ensure_healthy(self) -> None:
        """Ensure the connection is healthy, performing health check if needed."""
        now = datetime.now()
        
        if (self._last_health_check is None or 
            (now - self._last_health_check).total_seconds() > self.health_check_interval):
            await self._health_check()
        
        if not self._is_healthy:
            raise ConnectionError("PostgreSQL connection is not healthy")
    
    async def _retry_operation(self, operation, *args, **kwargs):
        """Execute an operation with retry logic."""
        last_exception = None
        
        for attempt in range(self.retry_attempts):
            try:
                await self._ensure_healthy()
                return await operation(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Operation failed (attempt {attempt + 1}/{self.retry_attempts}): {e}")
                
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"Operation failed after {self.retry_attempts} attempts")
        
        raise last_exception
    
    def get_checkpointer(self):
        """Get the underlying PostgresSaver instance."""
        if not self._checkpointer:
            raise RuntimeError("PostgreSQL checkpointer not initialized")
        return self._checkpointer
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about stored checkpoints."""
        try:
            await self._ensure_healthy()
            
            async with self._connection_pool.acquire() as conn:
                # Get checkpoint count and size statistics
                stats_query = """
                SELECT 
                    COUNT(*) as total_checkpoints,
                    COALESCE(SUM(pg_column_size(checkpoint)), 0) as total_size_bytes,
                    COALESCE(AVG(pg_column_size(checkpoint)), 0) as average_size_bytes,
                    MIN(created_at) as oldest_checkpoint,
                    MAX(created_at) as newest_checkpoint
                FROM checkpoints
                """
                
                row = await conn.fetchrow(stats_query)
                
                # Get thread statistics
                thread_stats_query = """
                SELECT 
                    COUNT(DISTINCT thread_id) as unique_threads,
                    thread_id,
                    COUNT(*) as checkpoint_count
                FROM checkpoints 
                GROUP BY thread_id 
                ORDER BY checkpoint_count DESC 
                LIMIT 10
                """
                
                thread_rows = await conn.fetch(thread_stats_query)
                
                return {
                    "total_checkpoints": row["total_checkpoints"],
                    "total_size_bytes": int(row["total_size_bytes"]),
                    "average_size_bytes": float(row["average_size_bytes"]),
                    "oldest_checkpoint": row["oldest_checkpoint"].isoformat() if row["oldest_checkpoint"] else None,
                    "newest_checkpoint": row["newest_checkpoint"].isoformat() if row["newest_checkpoint"] else None,
                    "unique_threads": thread_rows[0]["unique_threads"] if thread_rows else 0,
                    "top_threads": [
                        {"thread_id": row["thread_id"], "checkpoint_count": row["checkpoint_count"]}
                        for row in thread_rows
                    ],
                    "connection_healthy": self._is_healthy,
                    "last_health_check": self._last_health_check.isoformat() if self._last_health_check else None
                }
                
        except Exception as e:
            logger.error(f"Failed to get PostgreSQL checkpoint stats: {e}")
            return {
                "error": str(e),
                "connection_healthy": False
            }
    
    async def cleanup_old_checkpoints(self, max_age_hours: int = 24) -> int:
        """
        Clean up checkpoints older than the specified age.
        
        Args:
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of checkpoints cleaned up
        """
        try:
            await self._ensure_healthy()
            
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            async with self._connection_pool.acquire() as conn:
                # Delete old checkpoints
                delete_query = """
                DELETE FROM checkpoints 
                WHERE created_at < $1
                """
                
                result = await conn.execute(delete_query, cutoff_time)
                
                # Extract the number of deleted rows from the result
                deleted_count = int(result.split()[-1]) if result.startswith("DELETE") else 0
                
                logger.info(f"Cleaned up {deleted_count} checkpoints older than {max_age_hours} hours")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup old checkpoints: {e}")
            return 0
    
    async def get_checkpoint_list(self, limit: int = 100, thread_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get a list of stored checkpoints with metadata.
        
        Args:
            limit: Maximum number of checkpoints to return
            thread_id: Optional thread ID to filter by
            
        Returns:
            List of checkpoint metadata
        """
        try:
            await self._ensure_healthy()
            
            async with self._connection_pool.acquire() as conn:
                if thread_id:
                    query = """
                    SELECT thread_id, checkpoint_id, created_at, 
                           pg_column_size(checkpoint) as size_bytes
                    FROM checkpoints 
                    WHERE thread_id = $1
                    ORDER BY created_at DESC 
                    LIMIT $2
                    """
                    rows = await conn.fetch(query, thread_id, limit)
                else:
                    query = """
                    SELECT thread_id, checkpoint_id, created_at,
                           pg_column_size(checkpoint) as size_bytes
                    FROM checkpoints 
                    ORDER BY created_at DESC 
                    LIMIT $1
                    """
                    rows = await conn.fetch(query, limit)
                
                return [
                    {
                        "thread_id": row["thread_id"],
                        "checkpoint_id": row["checkpoint_id"],
                        "created_at": row["created_at"].isoformat(),
                        "size_bytes": row["size_bytes"]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Failed to get checkpoint list: {e}")
            return []
    
    async def vacuum_checkpoints_table(self) -> bool:
        """
        Perform VACUUM on the checkpoints table to reclaim space.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            await self._ensure_healthy()
            
            async with self._connection_pool.acquire() as conn:
                await conn.execute("VACUUM ANALYZE checkpoints")
                logger.info("Successfully vacuumed checkpoints table")
                return True
                
        except Exception as e:
            logger.error(f"Failed to vacuum checkpoints table: {e}")
            return False


class PostgresCheckpointer:
    """
    Wrapper class for PostgreSQL checkpointing functionality.
    
    This provides a consistent interface for PostgreSQL-based checkpointing
    with enhanced features for production environments.
    """
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        pool_size: int = 10,
        max_overflow: int = 20
    ):
        """
        Initialize PostgreSQL checkpointer.
        
        Args:
            connection_string: PostgreSQL connection string (uses env var if not provided)
            pool_size: Connection pool size
            max_overflow: Maximum pool overflow
        """
        if not POSTGRES_AVAILABLE:
            raise ImportError("PostgreSQL checkpointer requires langgraph[postgres] and asyncpg")
        
        self.connection_string = connection_string or os.getenv('POSTGRES_CONNECTION_STRING')
        if not self.connection_string:
            raise ValueError("PostgreSQL connection string is required")
        
        self.checkpointer = EnhancedPostgresSaver(
            connection_string=self.connection_string,
            pool_size=pool_size,
            max_overflow=max_overflow
        )
        
        self._initialized = False
        
        logger.info(f"PostgreSQL checkpointer created with pool_size={pool_size}, "
                   f"max_overflow={max_overflow}")
    
    async def initialize(self) -> None:
        """Initialize the checkpointer."""
        if not self._initialized:
            await self.checkpointer.initialize()
            self._initialized = True
    
    async def close(self) -> None:
        """Close the checkpointer."""
        if self._initialized:
            await self.checkpointer.close()
            self._initialized = False
    
    def get_checkpointer(self):
        """Get the underlying checkpointer instance."""
        if not self._initialized:
            raise RuntimeError("PostgreSQL checkpointer not initialized")
        return self.checkpointer.get_checkpointer()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get checkpointer statistics."""
        if not self._initialized:
            await self.initialize()
        return await self.checkpointer.get_stats()
    
    async def cleanup(self, max_age_hours: int = 24) -> int:
        """Clean up old checkpoints."""
        if not self._initialized:
            await self.initialize()
        return await self.checkpointer.cleanup_old_checkpoints(max_age_hours)
    
    async def get_checkpoint_list(self, limit: int = 100, thread_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of checkpoints."""
        if not self._initialized:
            await self.initialize()
        return await self.checkpointer.get_checkpoint_list(limit, thread_id)
    
    async def vacuum(self) -> bool:
        """Vacuum the checkpoints table."""
        if not self._initialized:
            await self.initialize()
        return await self.checkpointer.vacuum_checkpoints_table()
    
    @asynccontextmanager
    async def managed_checkpointer(self):
        """Context manager for automatic initialization and cleanup."""
        try:
            await self.initialize()
            yield self
        finally:
            await self.close()
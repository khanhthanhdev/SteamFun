"""
Performance and Caching Layer for RAG System Enhancement

This module implements intelligent caching, connection pooling, and performance monitoring
for the enhanced RAG system to ensure sub-2-second response times.
"""

import os
import json
import time
import hashlib
import threading
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import pickle
import sqlite3
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheType(Enum):
    """Types of cached data."""
    QUERY_RESULTS = "query_results"
    EMBEDDINGS = "embeddings"
    VECTOR_STORE = "vector_store"
    METADATA = "metadata"


class InvalidationTrigger(Enum):
    """Cache invalidation triggers."""
    DOCUMENT_UPDATE = "document_update"
    MANUAL = "manual"
    TTL_EXPIRED = "ttl_expired"
    MEMORY_PRESSURE = "memory_pressure"


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    data: Any
    cache_type: CacheType
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: int
    size_bytes: int
    metadata: Dict[str, Any]
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds <= 0:
            return False
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class CacheConfig:
    """Configuration for the performance cache."""
    max_memory_mb: int = 512
    default_ttl_seconds: int = 3600
    query_result_ttl: int = 1800  # 30 minutes
    embedding_ttl: int = 86400    # 24 hours
    cleanup_interval_seconds: int = 300  # 5 minutes
    max_entries_per_type: int = 1000
    enable_disk_cache: bool = True
    disk_cache_path: str = ".cache/rag_performance"
    enable_compression: bool = True


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    cache_hits: int = 0
    cache_misses: int = 0
    total_queries: int = 0
    avg_response_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    disk_usage_mb: float = 0.0
    last_cleanup: Optional[datetime] = None
    
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class PerformanceOptimizedCache:
    """
    Intelligent caching system for RAG operations with TTL, memory management,
    and performance monitoring.
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.cache_lock = threading.RLock()
        self.metrics = PerformanceMetrics()
        self.response_times = deque(maxlen=1000)  # Keep last 1000 response times
        
        # Initialize disk cache
        if self.config.enable_disk_cache:
            self.disk_cache_path = Path(self.config.disk_cache_path)
            self.disk_cache_path.mkdir(parents=True, exist_ok=True)
            self._init_disk_cache()
        
        # Start background cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        
        logger.info(f"PerformanceOptimizedCache initialized with config: {self.config}")
    
    def _init_disk_cache(self):
        """Initialize SQLite database for disk caching."""
        self.db_path = self.disk_cache_path / "cache.db"
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    cache_type TEXT,
                    data BLOB,
                    created_at TEXT,
                    last_accessed TEXT,
                    access_count INTEGER,
                    ttl_seconds INTEGER,
                    size_bytes INTEGER,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_type ON cache_entries(cache_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON cache_entries(created_at)
            """)
    
    def get_cached_results(self, query_hash: str, cache_type: CacheType = CacheType.QUERY_RESULTS) -> Optional[Any]:
        """
        Retrieve cached results with intelligent invalidation.
        
        Args:
            query_hash: Hash of the query
            cache_type: Type of cache to search
            
        Returns:
            Cached data if found and valid, None otherwise
        """
        start_time = time.time()
        
        try:
            with self.cache_lock:
                # Check memory cache first
                cache_key = f"{cache_type.value}:{query_hash}"
                
                if cache_key in self.memory_cache:
                    entry = self.memory_cache[cache_key]
                    
                    if entry.is_expired():
                        del self.memory_cache[cache_key]
                        self.metrics.cache_misses += 1
                        return None
                    
                    entry.update_access()
                    self.metrics.cache_hits += 1
                    logger.debug(f"Memory cache hit for key: {cache_key}")
                    return entry.data
                
                # Check disk cache if enabled
                if self.config.enable_disk_cache:
                    disk_data = self._get_from_disk_cache(cache_key)
                    if disk_data is not None:
                        # Promote to memory cache
                        self._store_in_memory(cache_key, disk_data, cache_type)
                        self.metrics.cache_hits += 1
                        logger.debug(f"Disk cache hit for key: {cache_key}")
                        return disk_data
                
                self.metrics.cache_misses += 1
                return None
                
        finally:
            response_time = (time.time() - start_time) * 1000
            self.response_times.append(response_time)
            self._update_avg_response_time()
    
    def cache_results(self, query_hash: str, results: Any, cache_type: CacheType = CacheType.QUERY_RESULTS, 
                     ttl_override: Optional[int] = None, metadata: Optional[Dict] = None):
        """
        Cache results with appropriate TTL and invalidation rules.
        
        Args:
            query_hash: Hash of the query
            results: Data to cache
            cache_type: Type of cache entry
            ttl_override: Override default TTL
            metadata: Additional metadata for the cache entry
        """
        start_time = time.time()
        
        try:
            with self.cache_lock:
                cache_key = f"{cache_type.value}:{query_hash}"
                
                # Determine TTL
                ttl = ttl_override or self._get_default_ttl(cache_type)
                
                # Calculate size
                size_bytes = self._calculate_size(results)
                
                # Check memory pressure
                if self._should_use_disk_cache(size_bytes):
                    if self.config.enable_disk_cache:
                        self._store_in_disk_cache(cache_key, results, cache_type, ttl, metadata or {})
                        logger.debug(f"Stored large entry in disk cache: {cache_key}")
                else:
                    self._store_in_memory(cache_key, results, cache_type, ttl, metadata or {})
                    logger.debug(f"Stored entry in memory cache: {cache_key}")
                
                # Update metrics
                self.metrics.total_queries += 1
                
        finally:
            response_time = (time.time() - start_time) * 1000
            self.response_times.append(response_time)
            self._update_avg_response_time()
    
    def _store_in_memory(self, cache_key: str, data: Any, cache_type: CacheType, 
                        ttl: int = None, metadata: Dict = None):
        """Store data in memory cache."""
        ttl = ttl or self._get_default_ttl(cache_type)
        size_bytes = self._calculate_size(data)
        
        entry = CacheEntry(
            key=cache_key,
            data=data,
            cache_type=cache_type,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=0,
            ttl_seconds=ttl,
            size_bytes=size_bytes,
            metadata=metadata or {}
        )
        
        self.memory_cache[cache_key] = entry
        
        # Check if we need to evict entries
        self._maybe_evict_memory_entries()
    
    def _store_in_disk_cache(self, cache_key: str, data: Any, cache_type: CacheType, 
                           ttl: int, metadata: Dict):
        """Store data in disk cache."""
        if not self.config.enable_disk_cache:
            return
        
        try:
            # Serialize data
            serialized_data = pickle.dumps(data)
            if self.config.enable_compression:
                import gzip
                serialized_data = gzip.compress(serialized_data)
            
            now = datetime.now().isoformat()
            
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries 
                    (key, cache_type, data, created_at, last_accessed, access_count, ttl_seconds, size_bytes, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    cache_key,
                    cache_type.value,
                    serialized_data,
                    now,
                    now,
                    0,
                    ttl,
                    len(serialized_data),
                    json.dumps(metadata)
                ))
                
        except Exception as e:
            logger.error(f"Error storing in disk cache: {e}")
    
    def _get_from_disk_cache(self, cache_key: str) -> Optional[Any]:
        """Retrieve data from disk cache."""
        if not self.config.enable_disk_cache:
            return None
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("""
                    SELECT data, created_at, ttl_seconds, access_count 
                    FROM cache_entries 
                    WHERE key = ?
                """, (cache_key,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                data_blob, created_at_str, ttl_seconds, access_count = row
                
                # Check if expired
                created_at = datetime.fromisoformat(created_at_str)
                if ttl_seconds > 0 and datetime.now() > created_at + timedelta(seconds=ttl_seconds):
                    # Delete expired entry
                    conn.execute("DELETE FROM cache_entries WHERE key = ?", (cache_key,))
                    return None
                
                # Update access count
                conn.execute("""
                    UPDATE cache_entries 
                    SET last_accessed = ?, access_count = access_count + 1 
                    WHERE key = ?
                """, (datetime.now().isoformat(), cache_key))
                
                # Deserialize data
                if self.config.enable_compression:
                    import gzip
                    data_blob = gzip.decompress(data_blob)
                
                return pickle.loads(data_blob)
                
        except Exception as e:
            logger.error(f"Error retrieving from disk cache: {e}")
            return None
    
    def invalidate_cache(self, invalidation_trigger: InvalidationTrigger, 
                        cache_type: Optional[CacheType] = None, pattern: Optional[str] = None):
        """
        Intelligently invalidate cache when documentation updates.
        
        Args:
            invalidation_trigger: Reason for invalidation
            cache_type: Specific cache type to invalidate (None for all)
            pattern: Pattern to match cache keys (None for all)
        """
        with self.cache_lock:
            keys_to_remove = []
            
            for key, entry in self.memory_cache.items():
                should_remove = False
                
                if cache_type is None or entry.cache_type == cache_type:
                    if pattern is None or pattern in key:
                        should_remove = True
                
                if should_remove:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.memory_cache[key]
            
            # Also invalidate disk cache
            if self.config.enable_disk_cache:
                self._invalidate_disk_cache(cache_type, pattern)
            
            logger.info(f"Invalidated {len(keys_to_remove)} cache entries due to {invalidation_trigger.value}")
    
    def _invalidate_disk_cache(self, cache_type: Optional[CacheType], pattern: Optional[str]):
        """Invalidate disk cache entries."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                if cache_type and pattern:
                    conn.execute("""
                        DELETE FROM cache_entries 
                        WHERE cache_type = ? AND key LIKE ?
                    """, (cache_type.value, f"%{pattern}%"))
                elif cache_type:
                    conn.execute("""
                        DELETE FROM cache_entries 
                        WHERE cache_type = ?
                    """, (cache_type.value,))
                elif pattern:
                    conn.execute("""
                        DELETE FROM cache_entries 
                        WHERE key LIKE ?
                    """, (f"%{pattern}%",))
                else:
                    conn.execute("DELETE FROM cache_entries")
                    
        except Exception as e:
            logger.error(f"Error invalidating disk cache: {e}")
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        with self.cache_lock:
            # Update memory usage
            total_memory = sum(entry.size_bytes for entry in self.memory_cache.values())
            self.metrics.memory_usage_mb = total_memory / (1024 * 1024)
            
            # Update disk usage if enabled
            if self.config.enable_disk_cache:
                try:
                    disk_size = sum(f.stat().st_size for f in self.disk_cache_path.rglob('*') if f.is_file())
                    self.metrics.disk_usage_mb = disk_size / (1024 * 1024)
                except Exception:
                    pass
            
            return self.metrics
    
    def _get_default_ttl(self, cache_type: CacheType) -> int:
        """Get default TTL for cache type."""
        ttl_map = {
            CacheType.QUERY_RESULTS: self.config.query_result_ttl,
            CacheType.EMBEDDINGS: self.config.embedding_ttl,
            CacheType.VECTOR_STORE: self.config.default_ttl_seconds,
            CacheType.METADATA: self.config.default_ttl_seconds
        }
        return ttl_map.get(cache_type, self.config.default_ttl_seconds)
    
    def _calculate_size(self, data: Any) -> int:
        """Calculate approximate size of data in bytes."""
        try:
            return len(pickle.dumps(data))
        except Exception:
            # Fallback estimation
            if isinstance(data, str):
                return len(data.encode('utf-8'))
            elif isinstance(data, (list, tuple)):
                return sum(self._calculate_size(item) for item in data)
            elif isinstance(data, dict):
                return sum(self._calculate_size(k) + self._calculate_size(v) for k, v in data.items())
            else:
                return 1024  # Default estimate
    
    def _should_use_disk_cache(self, size_bytes: int) -> bool:
        """Determine if entry should go to disk cache based on size and memory pressure."""
        if not self.config.enable_disk_cache:
            return False
        
        # Large entries go to disk
        if size_bytes > 1024 * 1024:  # 1MB
            return True
        
        # Check memory pressure
        current_memory = sum(entry.size_bytes for entry in self.memory_cache.values())
        max_memory_bytes = self.config.max_memory_mb * 1024 * 1024
        
        return (current_memory + size_bytes) > max_memory_bytes * 0.8  # 80% threshold
    
    def _maybe_evict_memory_entries(self):
        """Evict memory entries if needed based on size and count limits."""
        current_memory = sum(entry.size_bytes for entry in self.memory_cache.values())
        max_memory_bytes = self.config.max_memory_mb * 1024 * 1024
        
        # Check memory limit
        if current_memory > max_memory_bytes:
            self._evict_lru_entries(target_size=max_memory_bytes * 0.7)  # Target 70%
        
        # Check count limit per type
        type_counts = defaultdict(int)
        for entry in self.memory_cache.values():
            type_counts[entry.cache_type] += 1
        
        for cache_type, count in type_counts.items():
            if count > self.config.max_entries_per_type:
                self._evict_lru_entries_by_type(cache_type, self.config.max_entries_per_type * 0.8)
    
    def _evict_lru_entries(self, target_size: int):
        """Evict least recently used entries to reach target size."""
        # Sort by last accessed time
        entries_by_access = sorted(
            self.memory_cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        current_size = sum(entry.size_bytes for _, entry in entries_by_access)
        
        for key, entry in entries_by_access:
            if current_size <= target_size:
                break
            
            # Move to disk cache if possible
            if self.config.enable_disk_cache:
                self._store_in_disk_cache(key, entry.data, entry.cache_type, 
                                        entry.ttl_seconds, entry.metadata)
            
            del self.memory_cache[key]
            current_size -= entry.size_bytes
            logger.debug(f"Evicted entry from memory cache: {key}")
    
    def _evict_lru_entries_by_type(self, cache_type: CacheType, target_count: int):
        """Evict LRU entries of specific type to reach target count."""
        type_entries = [
            (key, entry) for key, entry in self.memory_cache.items()
            if entry.cache_type == cache_type
        ]
        
        if len(type_entries) <= target_count:
            return
        
        # Sort by last accessed time
        type_entries.sort(key=lambda x: x[1].last_accessed)
        
        entries_to_remove = len(type_entries) - int(target_count)
        
        for i in range(entries_to_remove):
            key, entry = type_entries[i]
            
            # Move to disk cache if possible
            if self.config.enable_disk_cache:
                self._store_in_disk_cache(key, entry.data, entry.cache_type, 
                                        entry.ttl_seconds, entry.metadata)
            
            del self.memory_cache[key]
            logger.debug(f"Evicted {cache_type.value} entry from memory cache: {key}")
    
    def _update_avg_response_time(self):
        """Update average response time metric."""
        if self.response_times:
            self.metrics.avg_response_time_ms = sum(self.response_times) / len(self.response_times)
    
    def _cleanup_worker(self):
        """Background worker for cache cleanup."""
        while True:
            try:
                time.sleep(self.config.cleanup_interval_seconds)
                self._cleanup_expired_entries()
                self.metrics.last_cleanup = datetime.now()
                
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
    
    def _cleanup_expired_entries(self):
        """Clean up expired cache entries."""
        with self.cache_lock:
            expired_keys = []
            
            for key, entry in self.memory_cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.memory_cache[key]
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired memory cache entries")
            
            # Clean up disk cache
            if self.config.enable_disk_cache:
                self._cleanup_expired_disk_entries()
    
    def _cleanup_expired_disk_entries(self):
        """Clean up expired disk cache entries."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Delete expired entries
                conn.execute("""
                    DELETE FROM cache_entries 
                    WHERE ttl_seconds > 0 
                    AND datetime(created_at, '+' || ttl_seconds || ' seconds') < datetime('now')
                """)
                
                deleted_count = conn.total_changes
                if deleted_count > 0:
                    logger.debug(f"Cleaned up {deleted_count} expired disk cache entries")
                    
        except Exception as e:
            logger.error(f"Error cleaning up disk cache: {e}")
    
    def clear_cache(self, cache_type: Optional[CacheType] = None):
        """Clear all cache entries or entries of specific type."""
        with self.cache_lock:
            if cache_type is None:
                self.memory_cache.clear()
                if self.config.enable_disk_cache:
                    try:
                        with sqlite3.connect(str(self.db_path)) as conn:
                            conn.execute("DELETE FROM cache_entries")
                    except Exception as e:
                        logger.error(f"Error clearing disk cache: {e}")
            else:
                # Clear specific type
                keys_to_remove = [
                    key for key, entry in self.memory_cache.items()
                    if entry.cache_type == cache_type
                ]
                for key in keys_to_remove:
                    del self.memory_cache[key]
                
                if self.config.enable_disk_cache:
                    try:
                        with sqlite3.connect(str(self.db_path)) as conn:
                            conn.execute("DELETE FROM cache_entries WHERE cache_type = ?", 
                                       (cache_type.value,))
                    except Exception as e:
                        logger.error(f"Error clearing disk cache for type {cache_type}: {e}")
            
            logger.info(f"Cleared cache for type: {cache_type or 'all'}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics."""
        with self.cache_lock:
            stats = {
                "memory_cache_entries": len(self.memory_cache),
                "memory_usage_mb": sum(entry.size_bytes for entry in self.memory_cache.values()) / (1024 * 1024),
                "cache_hit_rate": self.metrics.hit_rate(),
                "total_queries": self.metrics.total_queries,
                "avg_response_time_ms": self.metrics.avg_response_time_ms,
                "entries_by_type": defaultdict(int)
            }
            
            for entry in self.memory_cache.values():
                stats["entries_by_type"][entry.cache_type.value] += 1
            
            # Add disk cache stats if enabled
            if self.config.enable_disk_cache:
                try:
                    with sqlite3.connect(str(self.db_path)) as conn:
                        cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
                        stats["disk_cache_entries"] = cursor.fetchone()[0]
                        
                        cursor = conn.execute("""
                            SELECT cache_type, COUNT(*) 
                            FROM cache_entries 
                            GROUP BY cache_type
                        """)
                        stats["disk_entries_by_type"] = dict(cursor.fetchall())
                        
                except Exception as e:
                    logger.error(f"Error getting disk cache stats: {e}")
            
            return dict(stats)


class VectorStoreConnectionPool:
    """
    Connection pooling and lazy loading for efficient vector store access.
    Supports batch query optimization and parallel processing.
    """
    
    def __init__(self, max_connections: int = 10, connection_timeout: int = 30):
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.connections: Dict[str, Any] = {}
        self.connection_usage: Dict[str, datetime] = {}
        self.connection_lock = threading.RLock()
        self.lazy_loader = LazyVectorStoreLoader()
        self.executor = ThreadPoolExecutor(max_workers=max_connections)
        
        logger.info(f"VectorStoreConnectionPool initialized with {max_connections} max connections")
    
    def get_connection(self, store_name: str, store_config: Dict[str, Any] = None) -> Any:
        """
        Get optimized connection with lazy loading.
        
        Args:
            store_name: Name/identifier of the vector store
            store_config: Configuration for the vector store
            
        Returns:
            Vector store connection
        """
        with self.connection_lock:
            # Check if connection exists and is still valid
            if store_name in self.connections:
                last_used = self.connection_usage.get(store_name)
                if last_used and (datetime.now() - last_used).seconds < self.connection_timeout:
                    self.connection_usage[store_name] = datetime.now()
                    logger.debug(f"Reusing existing connection for store: {store_name}")
                    return self.connections[store_name]
                else:
                    # Connection expired, remove it
                    del self.connections[store_name]
                    if store_name in self.connection_usage:
                        del self.connection_usage[store_name]
            
            # Create new connection using lazy loader
            connection = self.lazy_loader.load_vector_store(store_name, store_config or {})
            
            # Manage connection pool size
            if len(self.connections) >= self.max_connections:
                self._evict_oldest_connection()
            
            self.connections[store_name] = connection
            self.connection_usage[store_name] = datetime.now()
            
            logger.debug(f"Created new connection for store: {store_name}")
            return connection
    
    def _evict_oldest_connection(self):
        """Evict the oldest unused connection."""
        if not self.connection_usage:
            return
        
        oldest_store = min(self.connection_usage.items(), key=lambda x: x[1])[0]
        
        if oldest_store in self.connections:
            del self.connections[oldest_store]
        del self.connection_usage[oldest_store]
        
        logger.debug(f"Evicted oldest connection: {oldest_store}")
    
    def optimize_batch_queries(self, queries: List[Dict[str, Any]]) -> 'BatchQueryPlan':
        """
        Optimize multiple queries for parallel processing.
        
        Args:
            queries: List of query dictionaries with store_name and query parameters
            
        Returns:
            BatchQueryPlan for optimized execution
        """
        # Group queries by store
        queries_by_store = defaultdict(list)
        for i, query in enumerate(queries):
            store_name = query.get('store_name', 'default')
            queries_by_store[store_name].append((i, query))
        
        # Create batch plan
        batch_plan = BatchQueryPlan(
            total_queries=len(queries),
            stores_involved=list(queries_by_store.keys()),
            parallel_groups=[]
        )
        
        # Create parallel execution groups
        for store_name, store_queries in queries_by_store.items():
            group = BatchQueryGroup(
                store_name=store_name,
                queries=store_queries,
                estimated_time=len(store_queries) * 0.1  # Rough estimate
            )
            batch_plan.parallel_groups.append(group)
        
        logger.debug(f"Created batch plan for {len(queries)} queries across {len(queries_by_store)} stores")
        return batch_plan
    
    def execute_batch_queries(self, batch_plan: 'BatchQueryPlan') -> List[Any]:
        """
        Execute batch queries in parallel.
        
        Args:
            batch_plan: Optimized batch query plan
            
        Returns:
            List of results in original query order
        """
        results = [None] * batch_plan.total_queries
        
        # Submit parallel tasks
        future_to_group = {}
        for group in batch_plan.parallel_groups:
            future = self.executor.submit(self._execute_query_group, group)
            future_to_group[future] = group
        
        # Collect results
        for future in as_completed(future_to_group):
            group = future_to_group[future]
            try:
                group_results = future.result()
                # Place results in correct positions
                for (original_index, _), result in zip(group.queries, group_results):
                    results[original_index] = result
            except Exception as e:
                logger.error(f"Error executing query group for store {group.store_name}: {e}")
                # Fill with None for failed queries
                for original_index, _ in group.queries:
                    results[original_index] = None
        
        return results
    
    def _execute_query_group(self, group: 'BatchQueryGroup') -> List[Any]:
        """Execute queries for a single store."""
        connection = self.get_connection(group.store_name)
        results = []
        
        for _, query in group.queries:
            try:
                # Execute query based on type
                if hasattr(connection, 'similarity_search'):
                    result = connection.similarity_search(
                        query.get('query_text', ''),
                        k=query.get('k', 5)
                    )
                else:
                    # Fallback for different connection types
                    result = connection.search(query)
                
                results.append(result)
            except Exception as e:
                logger.error(f"Error executing individual query: {e}")
                results.append(None)
        
        return results
    
    def close_all_connections(self):
        """Close all connections and cleanup resources."""
        with self.connection_lock:
            for store_name, connection in self.connections.items():
                try:
                    if hasattr(connection, 'close'):
                        connection.close()
                except Exception as e:
                    logger.error(f"Error closing connection {store_name}: {e}")
            
            self.connections.clear()
            self.connection_usage.clear()
        
        self.executor.shutdown(wait=True)
        logger.info("Closed all vector store connections")


@dataclass
class BatchQueryGroup:
    """Group of queries for the same vector store."""
    store_name: str
    queries: List[Tuple[int, Dict[str, Any]]]  # (original_index, query)
    estimated_time: float


@dataclass
class BatchQueryPlan:
    """Plan for optimized batch query execution."""
    total_queries: int
    stores_involved: List[str]
    parallel_groups: List[BatchQueryGroup]


class LazyVectorStoreLoader:
    """Lazy loading of vector stores on demand."""
    
    def __init__(self):
        self.loaded_stores: Dict[str, Any] = {}
        self.store_configs: Dict[str, Dict] = {}
        self.load_lock = threading.RLock()
    
    def register_store_config(self, store_name: str, config: Dict[str, Any]):
        """Register configuration for a vector store."""
        self.store_configs[store_name] = config
        logger.debug(f"Registered config for store: {store_name}")
    
    def load_vector_store(self, store_name: str, config: Dict[str, Any] = None) -> Any:
        """
        Load vector store on demand.
        
        Args:
            store_name: Name of the vector store
            config: Configuration for the store
            
        Returns:
            Loaded vector store instance
        """
        with self.load_lock:
            if store_name in self.loaded_stores:
                return self.loaded_stores[store_name]
            
            # Use provided config or registered config
            store_config = config or self.store_configs.get(store_name, {})
            
            # Load the appropriate vector store based on config
            store_instance = self._create_vector_store(store_name, store_config)
            
            self.loaded_stores[store_name] = store_instance
            logger.info(f"Lazy loaded vector store: {store_name}")
            
            return store_instance
    
    def _create_vector_store(self, store_name: str, config: Dict[str, Any]) -> Any:
        """Create vector store instance based on configuration."""
        store_type = config.get('type', 'chroma')
        
        if store_type == 'chroma':
            return self._create_chroma_store(config)
        elif store_type == 'enhanced_rag':
            return self._create_enhanced_rag_store(config)
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")
    
    def _create_chroma_store(self, config: Dict[str, Any]) -> Any:
        """Create Chroma vector store."""
        try:
            from langchain_community.vectorstores import Chroma
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
            embeddings = HuggingFaceEmbeddings(
                model_name=config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
            )
            
            return Chroma(
                persist_directory=config.get('persist_directory'),
                embedding_function=embeddings,
                collection_name=config.get('collection_name', 'default')
            )
        except ImportError as e:
            logger.error(f"Failed to import Chroma dependencies: {e}")
            raise
    
    def _create_enhanced_rag_store(self, config: Dict[str, Any]) -> Any:
        """Create Enhanced RAG vector store."""
        try:
            from .vector_store import EnhancedRAGVectorStore
            
            return EnhancedRAGVectorStore(
                chroma_db_path=config.get('chroma_db_path'),
                manim_docs_path=config.get('manim_docs_path'),
                embedding_model=config.get('embedding_model'),
                session_id=config.get('session_id'),
                use_langfuse=config.get('use_langfuse', True),
                helper_model=config.get('helper_model')
            )
        except ImportError as e:
            logger.error(f"Failed to import Enhanced RAG store: {e}")
            raise
    
    def unload_store(self, store_name: str):
        """Unload a vector store to free memory."""
        with self.load_lock:
            if store_name in self.loaded_stores:
                store = self.loaded_stores[store_name]
                if hasattr(store, 'close'):
                    store.close()
                del self.loaded_stores[store_name]
                logger.debug(f"Unloaded vector store: {store_name}")


class PerformanceMonitor:
    """
    Performance monitoring and optimization for RAG operations.
    Tracks response times, resource usage, and provides alerts.
    """
    
    def __init__(self, alert_threshold_ms: float = 2000.0):
        self.alert_threshold_ms = alert_threshold_ms
        self.response_times: deque = deque(maxlen=10000)  # Keep last 10k response times
        self.resource_usage: deque = deque(maxlen=1000)   # Keep last 1k resource measurements
        self.alerts: List[Dict[str, Any]] = []
        self.monitoring_active = True
        self.monitor_lock = threading.RLock()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"PerformanceMonitor initialized with {alert_threshold_ms}ms alert threshold")
    
    def record_response_time(self, operation: str, response_time_ms: float, metadata: Dict[str, Any] = None):
        """
        Record response time for an operation.
        
        Args:
            operation: Name of the operation
            response_time_ms: Response time in milliseconds
            metadata: Additional metadata about the operation
        """
        with self.monitor_lock:
            record = {
                'timestamp': datetime.now(),
                'operation': operation,
                'response_time_ms': response_time_ms,
                'metadata': metadata or {}
            }
            
            self.response_times.append(record)
            
            # Check for performance alerts
            if response_time_ms > self.alert_threshold_ms:
                self._create_alert('slow_response', {
                    'operation': operation,
                    'response_time_ms': response_time_ms,
                    'threshold_ms': self.alert_threshold_ms,
                    'metadata': metadata
                })
    
    def get_performance_stats(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """
        Get performance statistics for a time window.
        
        Args:
            time_window_minutes: Time window in minutes
            
        Returns:
            Performance statistics dictionary
        """
        with self.monitor_lock:
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
            
            # Filter records within time window
            recent_records = [
                record for record in self.response_times
                if record['timestamp'] > cutoff_time
            ]
            
            if not recent_records:
                return {
                    'total_operations': 0,
                    'avg_response_time_ms': 0.0,
                    'median_response_time_ms': 0.0,
                    'p95_response_time_ms': 0.0,
                    'p99_response_time_ms': 0.0,
                    'slow_operations': 0,
                    'operations_by_type': {}
                }
            
            response_times = [record['response_time_ms'] for record in recent_records]
            operations_by_type = defaultdict(list)
            
            for record in recent_records:
                operations_by_type[record['operation']].append(record['response_time_ms'])
            
            # Calculate statistics
            response_times.sort()
            stats = {
                'total_operations': len(recent_records),
                'avg_response_time_ms': sum(response_times) / len(response_times),
                'median_response_time_ms': response_times[len(response_times) // 2],
                'p95_response_time_ms': response_times[int(len(response_times) * 0.95)],
                'p99_response_time_ms': response_times[int(len(response_times) * 0.99)],
                'slow_operations': sum(1 for rt in response_times if rt > self.alert_threshold_ms),
                'operations_by_type': {
                    op_type: {
                        'count': len(times),
                        'avg_ms': sum(times) / len(times),
                        'max_ms': max(times)
                    }
                    for op_type, times in operations_by_type.items()
                }
            }
            
            return stats
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage statistics."""
        if not PSUTIL_AVAILABLE:
            # Return mock data when psutil is not available
            usage = {
                'timestamp': datetime.now(),
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'memory_available_mb': 1024.0,
                'disk_percent': 0.0,
                'disk_free_gb': 10.0,
                'psutil_available': False
            }
            
            with self.monitor_lock:
                self.resource_usage.append(usage)
            
            return usage
        
        try:
            # Get system resource usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            usage = {
                'timestamp': datetime.now(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_mb': memory.available / (1024 * 1024),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024 * 1024 * 1024),
                'psutil_available': True
            }
            
            with self.monitor_lock:
                self.resource_usage.append(usage)
            
            return usage
            
        except Exception as e:
            logger.error(f"Error getting resource usage: {e}")
            return {}
    
    def _create_alert(self, alert_type: str, details: Dict[str, Any]):
        """Create a performance alert."""
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'details': details,
            'severity': self._determine_severity(alert_type, details)
        }
        
        self.alerts.append(alert)
        
        # Keep only recent alerts (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.alerts = [alert for alert in self.alerts if alert['timestamp'] > cutoff_time]
        
        logger.warning(f"Performance alert: {alert_type} - {details}")
    
    def _determine_severity(self, alert_type: str, details: Dict[str, Any]) -> str:
        """Determine alert severity based on type and details."""
        if alert_type == 'slow_response':
            response_time = details.get('response_time_ms', 0)
            if response_time > self.alert_threshold_ms * 3:
                return 'critical'
            elif response_time > self.alert_threshold_ms * 2:
                return 'high'
            else:
                return 'medium'
        
        return 'medium'
    
    def get_alerts(self, severity: Optional[str] = None, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get recent alerts.
        
        Args:
            severity: Filter by severity level
            hours: Number of hours to look back
            
        Returns:
            List of alert dictionaries
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_alerts = [
            alert for alert in self.alerts
            if alert['timestamp'] > cutoff_time
        ]
        
        if severity:
            filtered_alerts = [
                alert for alert in filtered_alerts
                if alert['severity'] == severity
            ]
        
        return sorted(filtered_alerts, key=lambda x: x['timestamp'], reverse=True)
    
    def _monitoring_worker(self):
        """Background worker for continuous monitoring."""
        while self.monitoring_active:
            try:
                # Record resource usage
                self.get_resource_usage()
                
                # Check for resource-based alerts
                if self.resource_usage:
                    latest_usage = self.resource_usage[-1]
                    
                    if latest_usage.get('memory_percent', 0) > 90:
                        self._create_alert('high_memory_usage', {
                            'memory_percent': latest_usage['memory_percent']
                        })
                    
                    if latest_usage.get('cpu_percent', 0) > 90:
                        self._create_alert('high_cpu_usage', {
                            'cpu_percent': latest_usage['cpu_percent']
                        })
                
                time.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Error in monitoring worker: {e}")
                time.sleep(60)
    
    def stop_monitoring(self):
        """Stop the monitoring worker."""
        self.monitoring_active = False
        logger.info("Performance monitoring stopped")


def create_query_hash(query: str, context: Dict[str, Any] = None) -> str:
    """
    Create a hash for caching query results.
    
    Args:
        query: Query string
        context: Additional context for the query
        
    Returns:
        Hash string for the query
    """
    hash_input = query
    if context:
        # Sort context keys for consistent hashing
        context_str = json.dumps(context, sort_keys=True)
        hash_input += context_str
    
    return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()[:16]
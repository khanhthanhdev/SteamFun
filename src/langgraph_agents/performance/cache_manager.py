"""
Advanced caching strategies for improved performance.

This module provides intelligent caching with TTL, LRU eviction,
cache warming, and performance-aware cache management.
"""

import asyncio
import logging
import hashlib
import pickle
import json
from typing import Dict, Any, List, Optional, TypeVar, Generic, Callable, Awaitable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import weakref
from collections import OrderedDict
import os
from pathlib import Path

logger = logging.getLogger(__name__)

T = TypeVar('T')
K = TypeVar('K')


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    FIFO = "fifo"  # First In, First Out
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with metadata."""
    key: str
    value: T
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: Optional[int]
    size_bytes: int
    tags: List[str] = field(default_factory=list)
    
    @property
    def is_expired(self) -> bool:
        """Check if the entry has expired."""
        if self.ttl_seconds is None:
            return False
        
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Get the age of the entry in seconds."""
        return (datetime.now() - self.created_at).total_seconds()
    
    @property
    def idle_seconds(self) -> float:
        """Get the idle time since last access in seconds."""
        return (datetime.now() - self.last_accessed).total_seconds()


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    hit_rate: float = 0.0
    average_access_time_ms: float = 0.0
    memory_efficiency: float = 0.0


class CacheManager(Generic[K, T]):
    """Advanced cache manager with multiple strategies and performance optimization."""
    
    def __init__(
        self,
        name: str,
        max_size: int = 1000,
        max_memory_mb: int = 100,
        default_ttl_seconds: Optional[int] = 3600,
        strategy: CacheStrategy = CacheStrategy.LRU,
        enable_persistence: bool = False,
        persistence_path: Optional[str] = None,
        enable_compression: bool = False,
        enable_metrics: bool = True,
        warmup_enabled: bool = False
    ):
        """Initialize the cache manager.
        
        Args:
            name: Name of the cache
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl_seconds: Default TTL for entries
            strategy: Cache eviction strategy
            enable_persistence: Enable persistent storage
            persistence_path: Path for persistent storage
            enable_compression: Enable value compression
            enable_metrics: Enable performance metrics
            warmup_enabled: Enable cache warming
        """
        self.name = name
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl_seconds = default_ttl_seconds
        self.strategy = strategy
        self.enable_persistence = enable_persistence
        self.enable_compression = enable_compression
        self.enable_metrics = enable_metrics
        self.warmup_enabled = warmup_enabled
        
        # Storage
        self._cache: Dict[str, CacheEntry[T]] = {}
        self._access_order: OrderedDict = OrderedDict()  # For LRU
        self._access_frequency: Dict[str, int] = {}  # For LFU
        
        # Persistence
        self.persistence_path = Path(persistence_path or f"cache_{name}.pkl")
        
        # Metrics
        self.stats = CacheStats()
        self._access_times: List[float] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        self._warmup_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        # Cache warming
        self._warmup_functions: List[Callable[[], Awaitable[Dict[str, T]]]] = []
        
        logger.info(f"Cache manager initialized: {name}, strategy: {strategy.value}, "
                   f"max_size: {max_size}, max_memory: {max_memory_mb}MB")
    
    async def start(self):
        """Start the cache manager and background tasks."""
        
        with self._lock:
            if self._cleanup_task is not None:
                return  # Already started
            
            # Load from persistence if enabled
            if self.enable_persistence:
                await self._load_from_persistence()
            
            # Start background tasks
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            if self.enable_metrics:
                self._metrics_task = asyncio.create_task(self._metrics_loop())
            
            if self.warmup_enabled:
                self._warmup_task = asyncio.create_task(self._warmup_loop())
            
            logger.info(f"Cache manager started: {self.name}")
    
    async def stop(self):
        """Stop the cache manager and save to persistence."""
        
        with self._lock:
            self._shutdown = True
            
            # Cancel background tasks
            for task in [self._cleanup_task, self._metrics_task, self._warmup_task]:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Save to persistence if enabled
            if self.enable_persistence:
                await self._save_to_persistence()
            
            logger.info(f"Cache manager stopped: {self.name}")
    
    async def get(self, key: K, default: Optional[T] = None) -> Optional[T]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        
        start_time = time.time()
        key_str = self._serialize_key(key)
        
        with self._lock:
            entry = self._cache.get(key_str)
            
            if entry is None:
                self.stats.misses += 1
                return default
            
            # Check if expired
            if entry.is_expired:
                del self._cache[key_str]
                self._access_order.pop(key_str, None)
                self._access_frequency.pop(key_str, None)
                self.stats.misses += 1
                self.stats.evictions += 1
                return default
            
            # Update access metadata
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            
            # Update strategy-specific data
            if self.strategy == CacheStrategy.LRU:
                self._access_order.move_to_end(key_str)
            elif self.strategy == CacheStrategy.LFU:
                self._access_frequency[key_str] = entry.access_count
            
            self.stats.hits += 1
            
            # Record access time
            if self.enable_metrics:
                access_time = (time.time() - start_time) * 1000  # ms
                self._access_times.append(access_time)
                if len(self._access_times) > 1000:
                    self._access_times.pop(0)
            
            return entry.value
    
    async def set(
        self,
        key: K,
        value: T,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: TTL for this entry (overrides default)
            tags: Tags for cache invalidation
            
        Returns:
            True if successfully cached
        """
        
        key_str = self._serialize_key(key)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        tags = tags or []
        
        # Calculate size
        size_bytes = self._calculate_size(value)
        
        with self._lock:
            # Check if we need to evict entries
            await self._ensure_capacity(size_bytes)
            
            # Create cache entry
            entry = CacheEntry(
                key=key_str,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=0,
                ttl_seconds=ttl,
                size_bytes=size_bytes,
                tags=tags
            )
            
            # Remove existing entry if present
            if key_str in self._cache:
                old_entry = self._cache[key_str]
                self.stats.size_bytes -= old_entry.size_bytes
            
            # Add new entry
            self._cache[key_str] = entry
            self.stats.size_bytes += size_bytes
            self.stats.entry_count = len(self._cache)
            
            # Update strategy-specific data
            if self.strategy == CacheStrategy.LRU:
                self._access_order[key_str] = True
            elif self.strategy == CacheStrategy.LFU:
                self._access_frequency[key_str] = 0
            
            return True
    
    async def delete(self, key: K) -> bool:
        """Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key was found and deleted
        """
        
        key_str = self._serialize_key(key)
        
        with self._lock:
            entry = self._cache.pop(key_str, None)
            if entry:
                self.stats.size_bytes -= entry.size_bytes
                self.stats.entry_count = len(self._cache)
                self._access_order.pop(key_str, None)
                self._access_frequency.pop(key_str, None)
                return True
            
            return False
    
    async def clear(self, tags: Optional[List[str]] = None):
        """Clear the cache or entries with specific tags.
        
        Args:
            tags: If provided, only clear entries with these tags
        """
        
        with self._lock:
            if tags is None:
                # Clear all
                self._cache.clear()
                self._access_order.clear()
                self._access_frequency.clear()
                self.stats.size_bytes = 0
                self.stats.entry_count = 0
            else:
                # Clear entries with specific tags
                keys_to_remove = []
                for key_str, entry in self._cache.items():
                    if any(tag in entry.tags for tag in tags):
                        keys_to_remove.append(key_str)
                
                for key_str in keys_to_remove:
                    entry = self._cache.pop(key_str)
                    self.stats.size_bytes -= entry.size_bytes
                    self._access_order.pop(key_str, None)
                    self._access_frequency.pop(key_str, None)
                
                self.stats.entry_count = len(self._cache)
    
    async def get_or_set(
        self,
        key: K,
        factory: Callable[[], Awaitable[T]],
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> T:
        """Get a value from cache or compute and cache it.
        
        Args:
            key: Cache key
            factory: Async function to compute the value
            ttl_seconds: TTL for the entry
            tags: Tags for cache invalidation
            
        Returns:
            Cached or computed value
        """
        
        # Try to get from cache first
        value = await self.get(key)
        if value is not None:
            return value
        
        # Compute the value
        value = await factory()
        
        # Cache the computed value
        await self.set(key, value, ttl_seconds, tags)
        
        return value
    
    async def _ensure_capacity(self, new_entry_size: int):
        """Ensure cache has capacity for a new entry."""
        
        # Check size limit
        while (len(self._cache) >= self.max_size or 
               self.stats.size_bytes + new_entry_size > self.max_memory_bytes):
            
            if not self._cache:
                break
            
            # Evict based on strategy
            evicted = await self._evict_entry()
            if not evicted:
                break  # No more entries to evict
    
    async def _evict_entry(self) -> bool:
        """Evict an entry based on the cache strategy.
        
        Returns:
            True if an entry was evicted
        """
        
        if not self._cache:
            return False
        
        key_to_evict = None
        
        if self.strategy == CacheStrategy.LRU:
            # Evict least recently used
            key_to_evict = next(iter(self._access_order))
        
        elif self.strategy == CacheStrategy.LFU:
            # Evict least frequently used
            if self._access_frequency:
                key_to_evict = min(self._access_frequency, key=self._access_frequency.get)
        
        elif self.strategy == CacheStrategy.TTL:
            # Evict expired entries first, then oldest
            now = datetime.now()
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            
            if expired_keys:
                key_to_evict = expired_keys[0]
            else:
                # Evict oldest entry
                key_to_evict = min(self._cache, key=lambda k: self._cache[k].created_at)
        
        elif self.strategy == CacheStrategy.FIFO:
            # Evict first in (oldest)
            key_to_evict = min(self._cache, key=lambda k: self._cache[k].created_at)
        
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Adaptive strategy based on access patterns
            key_to_evict = await self._adaptive_eviction()
        
        else:
            # Default to LRU
            key_to_evict = next(iter(self._access_order)) if self._access_order else next(iter(self._cache))
        
        if key_to_evict:
            entry = self._cache.pop(key_to_evict)
            self.stats.size_bytes -= entry.size_bytes
            self.stats.entry_count = len(self._cache)
            self.stats.evictions += 1
            
            self._access_order.pop(key_to_evict, None)
            self._access_frequency.pop(key_to_evict, None)
            
            return True
        
        return False
    
    async def _adaptive_eviction(self) -> Optional[str]:
        """Adaptive eviction based on access patterns."""
        
        if not self._cache:
            return None
        
        # Score entries based on multiple factors
        scores = {}
        now = datetime.now()
        
        for key, entry in self._cache.items():
            # Factors: recency, frequency, age, size
            recency_score = 1.0 / max(entry.idle_seconds, 1)
            frequency_score = entry.access_count / max(entry.age_seconds, 1)
            age_penalty = entry.age_seconds / 3600  # Penalty for old entries
            size_penalty = entry.size_bytes / (1024 * 1024)  # Penalty for large entries
            
            # Combined score (higher is better, so we evict lowest)
            score = (recency_score * 0.4 + frequency_score * 0.4 - 
                    age_penalty * 0.1 - size_penalty * 0.1)
            
            scores[key] = score
        
        # Return key with lowest score
        return min(scores, key=scores.get)
    
    def _serialize_key(self, key: K) -> str:
        """Serialize a key to string."""
        
        if isinstance(key, str):
            return key
        elif isinstance(key, (int, float, bool)):
            return str(key)
        else:
            # Use hash for complex objects
            key_str = str(key)
            return hashlib.md5(key_str.encode()).hexdigest()
    
    def _calculate_size(self, value: T) -> int:
        """Calculate the size of a value in bytes."""
        
        try:
            if self.enable_compression:
                # Estimate compressed size
                serialized = pickle.dumps(value)
                return len(serialized) // 2  # Rough compression estimate
            else:
                # Estimate uncompressed size
                if isinstance(value, str):
                    return len(value.encode('utf-8'))
                elif isinstance(value, (int, float, bool)):
                    return 8  # Rough estimate
                elif isinstance(value, (list, tuple)):
                    return sum(self._calculate_size(item) for item in value)
                elif isinstance(value, dict):
                    return sum(self._calculate_size(k) + self._calculate_size(v) 
                             for k, v in value.items())
                else:
                    # Use pickle size as estimate
                    return len(pickle.dumps(value))
        except Exception:
            return 1024  # Default estimate
    
    async def _cleanup_loop(self):
        """Background task for cleaning up expired entries."""
        
        try:
            while not self._shutdown:
                await asyncio.sleep(60)  # Check every minute
                
                if self._shutdown:
                    break
                
                with self._lock:
                    expired_keys = [
                        key for key, entry in self._cache.items()
                        if entry.is_expired
                    ]
                    
                    for key in expired_keys:
                        entry = self._cache.pop(key)
                        self.stats.size_bytes -= entry.size_bytes
                        self.stats.evictions += 1
                        self._access_order.pop(key, None)
                        self._access_frequency.pop(key, None)
                    
                    self.stats.entry_count = len(self._cache)
                    
                    if expired_keys:
                        logger.debug(f"Cleaned up {len(expired_keys)} expired entries from cache {self.name}")
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Cleanup loop error in cache {self.name}: {str(e)}")
    
    async def _metrics_loop(self):
        """Background task for updating metrics."""
        
        try:
            while not self._shutdown:
                await asyncio.sleep(30)  # Update every 30 seconds
                
                if self._shutdown:
                    break
                
                with self._lock:
                    # Update hit rate
                    total_requests = self.stats.hits + self.stats.misses
                    self.stats.hit_rate = self.stats.hits / max(total_requests, 1)
                    
                    # Update average access time
                    if self._access_times:
                        self.stats.average_access_time_ms = sum(self._access_times) / len(self._access_times)
                    
                    # Update memory efficiency
                    if self.stats.entry_count > 0:
                        avg_entry_size = self.stats.size_bytes / self.stats.entry_count
                        self.stats.memory_efficiency = min(1.0, 1024 / max(avg_entry_size, 1))
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Metrics loop error in cache {self.name}: {str(e)}")
    
    async def _warmup_loop(self):
        """Background task for cache warming."""
        
        try:
            while not self._shutdown:
                await asyncio.sleep(3600)  # Warmup every hour
                
                if self._shutdown:
                    break
                
                await self._perform_warmup()
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Warmup loop error in cache {self.name}: {str(e)}")
    
    async def _perform_warmup(self):
        """Perform cache warming using registered functions."""
        
        for warmup_func in self._warmup_functions:
            try:
                warmup_data = await warmup_func()
                
                for key, value in warmup_data.items():
                    await self.set(key, value, tags=['warmup'])
                
                logger.debug(f"Cache warmup completed for {self.name}: {len(warmup_data)} entries")
                
            except Exception as e:
                logger.error(f"Cache warmup failed for {self.name}: {str(e)}")
    
    def add_warmup_function(self, func: Callable[[], Awaitable[Dict[str, T]]]):
        """Add a function for cache warming.
        
        Args:
            func: Async function that returns dict of key-value pairs to cache
        """
        self._warmup_functions.append(func)
    
    async def _save_to_persistence(self):
        """Save cache to persistent storage."""
        
        if not self.enable_persistence:
            return
        
        try:
            # Prepare data for serialization
            cache_data = {
                'entries': {},
                'metadata': {
                    'name': self.name,
                    'strategy': self.strategy.value,
                    'saved_at': datetime.now().isoformat()
                }
            }
            
            for key, entry in self._cache.items():
                if not entry.is_expired:  # Only save non-expired entries
                    cache_data['entries'][key] = {
                        'value': entry.value,
                        'created_at': entry.created_at.isoformat(),
                        'last_accessed': entry.last_accessed.isoformat(),
                        'access_count': entry.access_count,
                        'ttl_seconds': entry.ttl_seconds,
                        'tags': entry.tags
                    }
            
            # Save to file
            with open(self.persistence_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.debug(f"Cache {self.name} saved to persistence: {len(cache_data['entries'])} entries")
            
        except Exception as e:
            logger.error(f"Failed to save cache {self.name} to persistence: {str(e)}")
    
    async def _load_from_persistence(self):
        """Load cache from persistent storage."""
        
        if not self.enable_persistence or not self.persistence_path.exists():
            return
        
        try:
            with open(self.persistence_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            entries_data = cache_data.get('entries', {})
            loaded_count = 0
            
            for key, entry_data in entries_data.items():
                # Reconstruct entry
                created_at = datetime.fromisoformat(entry_data['created_at'])
                last_accessed = datetime.fromisoformat(entry_data['last_accessed'])
                
                entry = CacheEntry(
                    key=key,
                    value=entry_data['value'],
                    created_at=created_at,
                    last_accessed=last_accessed,
                    access_count=entry_data['access_count'],
                    ttl_seconds=entry_data['ttl_seconds'],
                    size_bytes=self._calculate_size(entry_data['value']),
                    tags=entry_data.get('tags', [])
                )
                
                # Check if still valid
                if not entry.is_expired:
                    self._cache[key] = entry
                    self.stats.size_bytes += entry.size_bytes
                    
                    # Update strategy-specific data
                    if self.strategy == CacheStrategy.LRU:
                        self._access_order[key] = True
                    elif self.strategy == CacheStrategy.LFU:
                        self._access_frequency[key] = entry.access_count
                    
                    loaded_count += 1
            
            self.stats.entry_count = len(self._cache)
            
            logger.info(f"Cache {self.name} loaded from persistence: {loaded_count} entries")
            
        except Exception as e:
            logger.error(f"Failed to load cache {self.name} from persistence: {str(e)}")
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        
        with self._lock:
            return CacheStats(
                hits=self.stats.hits,
                misses=self.stats.misses,
                evictions=self.stats.evictions,
                size_bytes=self.stats.size_bytes,
                entry_count=self.stats.entry_count,
                hit_rate=self.stats.hit_rate,
                average_access_time_ms=self.stats.average_access_time_ms,
                memory_efficiency=self.stats.memory_efficiency
            )
    
    def get_info(self) -> Dict[str, Any]:
        """Get cache information and configuration."""
        
        return {
            'name': self.name,
            'strategy': self.strategy.value,
            'max_size': self.max_size,
            'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
            'default_ttl_seconds': self.default_ttl_seconds,
            'enable_persistence': self.enable_persistence,
            'enable_compression': self.enable_compression,
            'enable_metrics': self.enable_metrics,
            'warmup_enabled': self.warmup_enabled,
            'warmup_functions': len(self._warmup_functions),
            'is_running': self._cleanup_task is not None and not self._cleanup_task.done()
        }
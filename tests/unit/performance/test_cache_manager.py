"""
Unit tests for CacheManager class.

Tests cover Redis and local caching strategies, TTL management,
cache invalidation logic, and performance optimization features.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from src.langgraph_agents.performance.cache_manager import (
    CacheManager,
    CacheStrategy,
    CacheEntry,
    CacheStats,
    REDIS_AVAILABLE
)


class TestCacheManager:
    """Test cases for CacheManager functionality."""
    
    @pytest.fixture
    async def cache_manager(self):
        """Create a cache manager for testing."""
        manager = CacheManager(
            name="test_cache",
            max_size=100,
            max_memory_mb=10,
            default_ttl_seconds=3600,
            strategy=CacheStrategy.LRU,
            enable_metrics=True
        )
        await manager.start()
        yield manager
        await manager.stop()
    
    @pytest.fixture
    async def redis_cache_manager(self):
        """Create a Redis-enabled cache manager for testing."""
        if not REDIS_AVAILABLE:
            pytest.skip("Redis not available")
        
        manager = CacheManager(
            name="test_redis_cache",
            max_size=100,
            max_memory_mb=10,
            default_ttl_seconds=3600,
            strategy=CacheStrategy.LRU,
            enable_metrics=True,
            use_redis=True,
            redis_url="redis://localhost:6379"
        )
        
        # Mock Redis client for testing
        with patch('src.langgraph_agents.performance.cache_manager.redis') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.from_url.return_value = mock_client
            mock_client.ping.return_value = True
            mock_client.get.return_value = None
            mock_client.set.return_value = True
            mock_client.delete.return_value = 1
            mock_client.keys.return_value = []
            
            await manager.start()
            yield manager, mock_client
            await manager.stop()
    
    async def test_cache_manager_initialization(self):
        """Test cache manager initialization."""
        manager = CacheManager(
            name="test_init",
            max_size=50,
            max_memory_mb=5,
            default_ttl_seconds=1800,
            strategy=CacheStrategy.LFU
        )
        
        assert manager.name == "test_init"
        assert manager.max_size == 50
        assert manager.max_memory_bytes == 5 * 1024 * 1024
        assert manager.default_ttl_seconds == 1800
        assert manager.strategy == CacheStrategy.LFU
        assert not manager.use_redis
    
    async def test_cache_set_and_get(self, cache_manager):
        """Test basic cache set and get operations."""
        # Set a value
        result = await cache_manager.set("test_key", "test_value", ttl_seconds=60)
        assert result is True
        
        # Get the value
        value = await cache_manager.get("test_key")
        assert value == "test_value"
        
        # Get non-existent key
        value = await cache_manager.get("non_existent", default="default")
        assert value == "default"
    
    async def test_cache_ttl_expiration(self, cache_manager):
        """Test TTL expiration functionality."""
        # Set a value with short TTL
        await cache_manager.set("expire_key", "expire_value", ttl_seconds=1)
        
        # Should be available immediately
        value = await cache_manager.get("expire_key")
        assert value == "expire_value"
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Should be expired
        value = await cache_manager.get("expire_key")
        assert value is None
    
    async def test_cache_delete(self, cache_manager):
        """Test cache deletion."""
        # Set a value
        await cache_manager.set("delete_key", "delete_value")
        
        # Verify it exists
        value = await cache_manager.get("delete_key")
        assert value == "delete_value"
        
        # Delete it
        result = await cache_manager.delete("delete_key")
        assert result is True
        
        # Verify it's gone
        value = await cache_manager.get("delete_key")
        assert value is None
        
        # Delete non-existent key
        result = await cache_manager.delete("non_existent")
        assert result is False
    
    async def test_cache_clear(self, cache_manager):
        """Test cache clearing functionality."""
        # Set multiple values with tags
        await cache_manager.set("key1", "value1", tags=["tag1"])
        await cache_manager.set("key2", "value2", tags=["tag2"])
        await cache_manager.set("key3", "value3", tags=["tag1", "tag2"])
        
        # Clear by tag
        await cache_manager.clear(tags=["tag1"])
        
        # Check results
        assert await cache_manager.get("key1") is None  # Should be cleared
        assert await cache_manager.get("key2") == "value2"  # Should remain
        assert await cache_manager.get("key3") is None  # Should be cleared
        
        # Clear all
        await cache_manager.clear()
        assert await cache_manager.get("key2") is None
    
    async def test_cache_get_or_set(self, cache_manager):
        """Test get_or_set functionality."""
        call_count = 0
        
        async def factory():
            nonlocal call_count
            call_count += 1
            return f"computed_value_{call_count}"
        
        # First call should compute
        value1 = await cache_manager.get_or_set("compute_key", factory)
        assert value1 == "computed_value_1"
        assert call_count == 1
        
        # Second call should use cache
        value2 = await cache_manager.get_or_set("compute_key", factory)
        assert value2 == "computed_value_1"
        assert call_count == 1  # Factory not called again
    
    async def test_lru_eviction_strategy(self):
        """Test LRU eviction strategy."""
        manager = CacheManager(
            name="lru_test",
            max_size=3,
            strategy=CacheStrategy.LRU
        )
        await manager.start()
        
        try:
            # Fill cache
            await manager.set("key1", "value1")
            await manager.set("key2", "value2")
            await manager.set("key3", "value3")
            
            # Access key1 to make it recently used
            await manager.get("key1")
            
            # Add another key (should evict key2, the least recently used)
            await manager.set("key4", "value4")
            
            # Check eviction
            assert await manager.get("key1") == "value1"  # Should remain
            assert await manager.get("key2") is None  # Should be evicted
            assert await manager.get("key3") == "value3"  # Should remain
            assert await manager.get("key4") == "value4"  # Should remain
            
        finally:
            await manager.stop()
    
    async def test_lfu_eviction_strategy(self):
        """Test LFU eviction strategy."""
        manager = CacheManager(
            name="lfu_test",
            max_size=3,
            strategy=CacheStrategy.LFU
        )
        await manager.start()
        
        try:
            # Fill cache
            await manager.set("key1", "value1")
            await manager.set("key2", "value2")
            await manager.set("key3", "value3")
            
            # Access key1 multiple times
            await manager.get("key1")
            await manager.get("key1")
            await manager.get("key2")
            
            # Add another key (should evict key3, the least frequently used)
            await manager.set("key4", "value4")
            
            # Check eviction
            assert await manager.get("key1") == "value1"  # Should remain (most frequent)
            assert await manager.get("key2") == "value2"  # Should remain
            assert await manager.get("key3") is None  # Should be evicted
            assert await manager.get("key4") == "value4"  # Should remain
            
        finally:
            await manager.stop()
    
    async def test_cache_stats(self, cache_manager):
        """Test cache statistics collection."""
        # Perform some operations
        await cache_manager.set("stats_key1", "value1")
        await cache_manager.set("stats_key2", "value2")
        
        # Hit
        await cache_manager.get("stats_key1")
        
        # Miss
        await cache_manager.get("non_existent")
        
        # Get stats
        stats = cache_manager.get_stats()
        
        assert stats.hits >= 1
        assert stats.misses >= 1
        assert stats.entry_count >= 2
        assert stats.hit_rate > 0
    
    async def test_cache_key_generation(self, cache_manager):
        """Test cache key generation."""
        # Test simple arguments
        key1 = cache_manager.generate_cache_key("arg1", "arg2")
        key2 = cache_manager.generate_cache_key("arg1", "arg2")
        assert key1 == key2  # Should be deterministic
        
        # Test with keyword arguments
        key3 = cache_manager.generate_cache_key("arg1", param1="value1", param2="value2")
        key4 = cache_manager.generate_cache_key("arg1", param2="value2", param1="value1")
        assert key3 == key4  # Should be order-independent
        
        # Test different arguments produce different keys
        key5 = cache_manager.generate_cache_key("arg1", "arg3")
        assert key1 != key5
    
    async def test_pattern_invalidation(self, cache_manager):
        """Test pattern-based cache invalidation."""
        # Set multiple keys with patterns
        await cache_manager.set("user:123:profile", "profile_data")
        await cache_manager.set("user:123:settings", "settings_data")
        await cache_manager.set("user:456:profile", "other_profile")
        await cache_manager.set("product:789", "product_data")
        
        # Invalidate user:123:* pattern
        await cache_manager.invalidate_pattern("user:123:*")
        
        # Check results
        assert await cache_manager.get("user:123:profile") is None
        assert await cache_manager.get("user:123:settings") is None
        assert await cache_manager.get("user:456:profile") == "other_profile"
        assert await cache_manager.get("product:789") == "product_data"
    
    async def test_cache_size_info(self, cache_manager):
        """Test cache size information."""
        # Add some entries
        await cache_manager.set("size_key1", "value1")
        await cache_manager.set("size_key2", "value2")
        
        size_info = await cache_manager.get_cache_size()
        
        assert size_info["local_entries"] >= 2
        assert size_info["total_entries"] >= 2
        assert size_info["memory_bytes"] > 0
    
    async def test_warmup_functionality(self, cache_manager):
        """Test cache warming functionality."""
        warmup_called = False
        
        async def warmup_function():
            nonlocal warmup_called
            warmup_called = True
            return {
                "warm_key1": "warm_value1",
                "warm_key2": "warm_value2"
            }
        
        # Add warmup function
        cache_manager.add_warmup_function(warmup_function)
        
        # Trigger warmup
        await cache_manager._perform_warmup()
        
        # Check results
        assert warmup_called
        assert await cache_manager.get("warm_key1") == "warm_value1"
        assert await cache_manager.get("warm_key2") == "warm_value2"
    
    @pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")
    async def test_redis_cache_operations(self, redis_cache_manager):
        """Test Redis cache operations."""
        manager, mock_client = redis_cache_manager
        
        # Mock Redis responses
        mock_client.get.return_value = None  # Cache miss
        mock_client.set.return_value = True
        
        # Test set operation
        result = await manager.set("redis_key", "redis_value")
        assert result is True
        mock_client.set.assert_called()
        
        # Test get operation (cache miss)
        value = await manager.get("redis_key")
        assert value is None  # Because mock returns None
        mock_client.get.assert_called()
    
    @pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")
    async def test_redis_cache_hit(self, redis_cache_manager):
        """Test Redis cache hit scenario."""
        manager, mock_client = redis_cache_manager
        
        # Mock Redis cache hit
        import pickle
        entry_data = {
            'value': 'cached_value',
            'created_at': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'access_count': 0,
            'ttl_seconds': 3600,
            'tags': []
        }
        mock_client.get.return_value = pickle.dumps(entry_data)
        mock_client.set.return_value = True
        
        # Test get operation (cache hit)
        value = await manager.get("redis_key")
        assert value == "cached_value"
        
        # Verify Redis operations
        mock_client.get.assert_called()
        mock_client.set.assert_called()  # For updating access metadata
    
    async def test_memory_limit_eviction(self):
        """Test memory limit-based eviction."""
        manager = CacheManager(
            name="memory_test",
            max_size=1000,  # High size limit
            max_memory_mb=1,  # Low memory limit (1MB)
            strategy=CacheStrategy.LRU
        )
        await manager.start()
        
        try:
            # Add large values that should trigger memory-based eviction
            large_value = "x" * (512 * 1024)  # 512KB
            
            await manager.set("large1", large_value)
            await manager.set("large2", large_value)
            
            # Adding a third large value should trigger eviction
            await manager.set("large3", large_value)
            
            # Check that some entries were evicted
            size_info = await manager.get_cache_size()
            assert size_info["local_entries"] < 3
            
        finally:
            await manager.stop()
    
    async def test_adaptive_eviction_strategy(self):
        """Test adaptive eviction strategy."""
        manager = CacheManager(
            name="adaptive_test",
            max_size=3,
            strategy=CacheStrategy.ADAPTIVE
        )
        await manager.start()
        
        try:
            # Fill cache
            await manager.set("key1", "small_value")
            await manager.set("key2", "x" * 1000)  # Larger value
            await manager.set("key3", "small_value")
            
            # Access key1 frequently
            for _ in range(5):
                await manager.get("key1")
            
            # Add another key (should evict based on adaptive strategy)
            await manager.set("key4", "new_value")
            
            # key1 should likely remain due to high access frequency
            assert await manager.get("key1") == "small_value"
            
        finally:
            await manager.stop()
    
    async def test_compression_estimation(self):
        """Test compression size estimation."""
        manager = CacheManager(
            name="compression_test",
            enable_compression=True
        )
        
        # Test size calculation with compression
        test_value = "test" * 100
        size_with_compression = manager._calculate_size(test_value)
        
        manager.enable_compression = False
        size_without_compression = manager._calculate_size(test_value)
        
        # Compressed size should be smaller (rough estimate)
        assert size_with_compression <= size_without_compression
    
    async def test_cache_info(self, cache_manager):
        """Test cache information retrieval."""
        info = cache_manager.get_info()
        
        assert info["name"] == "test_cache"
        assert info["strategy"] == "lru"
        assert info["max_size"] == 100
        assert info["max_memory_mb"] == 10
        assert info["default_ttl_seconds"] == 3600
        assert info["enable_metrics"] is True
        assert "is_running" in info
    
    async def test_concurrent_access(self, cache_manager):
        """Test concurrent cache access."""
        async def worker(worker_id: int):
            for i in range(10):
                key = f"worker_{worker_id}_key_{i}"
                value = f"worker_{worker_id}_value_{i}"
                await cache_manager.set(key, value)
                retrieved = await cache_manager.get(key)
                assert retrieved == value
        
        # Run multiple workers concurrently
        tasks = [worker(i) for i in range(5)]
        await asyncio.gather(*tasks)
        
        # Verify cache state
        size_info = await cache_manager.get_cache_size()
        assert size_info["local_entries"] == 50  # 5 workers * 10 keys each
    
    async def test_error_handling_in_background_tasks(self):
        """Test error handling in background tasks."""
        manager = CacheManager(
            name="error_test",
            enable_metrics=True
        )
        
        # Mock an error in the metrics loop
        with patch.object(manager, '_metrics_loop', side_effect=Exception("Test error")):
            await manager.start()
            
            # Should still work despite background task error
            await manager.set("error_key", "error_value")
            value = await manager.get("error_key")
            assert value == "error_value"
            
            await manager.stop()


class TestCacheEntry:
    """Test cases for CacheEntry functionality."""
    
    def test_cache_entry_creation(self):
        """Test cache entry creation."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=0,
            ttl_seconds=3600,
            size_bytes=100,
            tags=["tag1", "tag2"]
        )
        
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.access_count == 0
        assert entry.ttl_seconds == 3600
        assert entry.size_bytes == 100
        assert entry.tags == ["tag1", "tag2"]
    
    def test_cache_entry_expiration(self):
        """Test cache entry expiration logic."""
        # Non-expiring entry
        entry1 = CacheEntry(
            key="key1",
            value="value1",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=0,
            ttl_seconds=None,
            size_bytes=100
        )
        assert not entry1.is_expired
        
        # Expired entry
        entry2 = CacheEntry(
            key="key2",
            value="value2",
            created_at=datetime.now() - timedelta(seconds=3700),  # Over 1 hour ago
            last_accessed=datetime.now(),
            access_count=0,
            ttl_seconds=3600,  # 1 hour TTL
            size_bytes=100
        )
        assert entry2.is_expired
        
        # Non-expired entry
        entry3 = CacheEntry(
            key="key3",
            value="value3",
            created_at=datetime.now() - timedelta(seconds=1800),  # 30 minutes ago
            last_accessed=datetime.now(),
            access_count=0,
            ttl_seconds=3600,  # 1 hour TTL
            size_bytes=100
        )
        assert not entry3.is_expired
    
    def test_cache_entry_age_and_idle(self):
        """Test cache entry age and idle time calculations."""
        now = datetime.now()
        created_time = now - timedelta(seconds=3600)  # 1 hour ago
        accessed_time = now - timedelta(seconds=1800)  # 30 minutes ago
        
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=created_time,
            last_accessed=accessed_time,
            access_count=5,
            ttl_seconds=7200,
            size_bytes=100
        )
        
        # Age should be approximately 1 hour
        assert 3590 <= entry.age_seconds <= 3610
        
        # Idle time should be approximately 30 minutes
        assert 1790 <= entry.idle_seconds <= 1810


class TestCacheStats:
    """Test cases for CacheStats functionality."""
    
    def test_cache_stats_creation(self):
        """Test cache stats creation."""
        stats = CacheStats(
            hits=100,
            misses=20,
            evictions=5,
            size_bytes=1024000,
            entry_count=50,
            hit_rate=0.83,
            average_access_time_ms=2.5,
            memory_efficiency=0.75
        )
        
        assert stats.hits == 100
        assert stats.misses == 20
        assert stats.evictions == 5
        assert stats.size_bytes == 1024000
        assert stats.entry_count == 50
        assert stats.hit_rate == 0.83
        assert stats.average_access_time_ms == 2.5
        assert stats.memory_efficiency == 0.75
    
    def test_cache_stats_defaults(self):
        """Test cache stats default values."""
        stats = CacheStats()
        
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.size_bytes == 0
        assert stats.entry_count == 0
        assert stats.hit_rate == 0.0
        assert stats.average_access_time_ms == 0.0
        assert stats.memory_efficiency == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
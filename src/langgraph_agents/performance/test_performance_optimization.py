"""
Test script for performance optimization features.

This script tests the performance optimization components to ensure they work correctly.
"""

import asyncio
import logging
import time
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_concurrency_manager():
    """Test the concurrency manager."""
    
    logger.info("Testing ConcurrencyManager...")
    
    from concurrency_manager import ConcurrencyManager, TaskPriority
    
    config = {
        'max_concurrent_tasks': 5,
        'max_concurrent_per_agent': 2,
        'max_concurrent_per_scene': 1,
        'max_memory_usage_mb': 1024,
        'max_cpu_usage_percent': 80.0,
        'adaptive_scaling': True
    }
    
    manager = ConcurrencyManager(config)
    
    # Test task execution
    async def sample_task():
        await asyncio.sleep(0.1)
        return "Task completed"
    
    # Execute multiple tasks
    tasks = []
    for i in range(10):
        task = manager.execute_task(
            task_id=f"test_task_{i}",
            agent_name="test_agent",
            scene_id=f"scene_{i % 3}",
            task_func=sample_task,
            priority=TaskPriority.NORMAL
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    # Check results
    assert len(results) == 10
    assert all(result == "Task completed" for result in results)
    
    # Test performance metrics
    metrics = manager.get_performance_metrics()
    assert metrics['total_tasks_executed'] == 10
    assert metrics['successful_tasks'] == 10
    
    # Test planner optimization
    await manager.optimize_for_planner_speed()
    
    # Test batch execution
    batch_tasks = [
        {
            'task_id': f'batch_task_{i}',
            'agent_name': 'batch_agent',
            'scene_id': f'batch_scene_{i}',
            'task_func': sample_task
        }
        for i in range(5)
    ]
    
    batch_results = await manager.batch_execute_tasks(batch_tasks)
    assert len(batch_results) == 5
    
    await manager.shutdown()
    
    logger.info("ConcurrencyManager test passed!")


async def test_cache_manager():
    """Test the cache manager."""
    
    logger.info("Testing CacheManager...")
    
    from cache_manager import CacheManager, CacheStrategy
    
    cache = CacheManager(
        name="test_cache",
        max_size=100,
        max_memory_mb=10,
        default_ttl_seconds=60,
        strategy=CacheStrategy.LRU,
        enable_persistence=False,
        enable_metrics=True
    )
    
    await cache.start()
    
    # Test basic operations
    await cache.set("key1", "value1")
    value = await cache.get("key1")
    assert value == "value1"
    
    # Test cache miss
    value = await cache.get("nonexistent", "default")
    assert value == "default"
    
    # Test get_or_set
    async def factory():
        return "computed_value"
    
    value = await cache.get_or_set("key2", factory)
    assert value == "computed_value"
    
    # Test cache stats
    stats = cache.get_stats()
    assert stats.hits > 0
    assert stats.misses > 0
    
    # Test cache clearing
    await cache.clear()
    value = await cache.get("key1")
    assert value is None
    
    await cache.stop()
    
    logger.info("CacheManager test passed!")


async def test_memory_manager():
    """Test the memory manager."""
    
    logger.info("Testing MemoryManager...")
    
    from memory_manager import MemoryManager, GarbageCollectionStrategy
    
    config = {
        'gc_strategy': GarbageCollectionStrategy.BALANCED.value,
        'warning_percent': 80.0,
        'critical_percent': 90.0,
        'enable_tracemalloc': False,  # Disable for testing
        'enable_leak_detection': False
    }
    
    manager = MemoryManager(config)
    await manager.start()
    
    # Test metrics collection
    metrics = await manager.get_current_metrics()
    assert metrics.memory_percent >= 0
    assert metrics.process_memory_mb > 0
    
    # Test memory stats
    stats = manager.get_memory_stats()
    assert 'current' in stats
    assert 'configuration' in stats
    
    # Test force cleanup
    await manager.force_cleanup()
    
    await manager.stop()
    
    logger.info("MemoryManager test passed!")


async def test_resource_pool():
    """Test the resource pool manager."""
    
    logger.info("Testing ResourcePoolManager...")
    
    from resource_pool import ResourcePoolManager, ResourceType
    
    config = {
        'default_pool_config': {
            'min_size': 1,
            'max_size': 5,
            'max_idle_time': 60,
            'health_check_interval': 30
        }
    }
    
    manager = ResourcePoolManager(config)
    
    # Create a test resource factory
    async def create_test_resource():
        return {'id': f'resource_{time.time()}', 'status': 'active'}
    
    async def destroy_test_resource(resource):
        resource['status'] = 'destroyed'
    
    async def check_resource_health(resource):
        return resource.get('status') == 'active'
    
    # Create a resource pool
    pool = await manager.create_pool(
        pool_name='test_pool',
        resource_type=ResourceType.COMPUTE_RESOURCE,
        factory=create_test_resource,
        destroyer=destroy_test_resource,
        health_checker=check_resource_health
    )
    
    # Test resource acquisition and release
    resource = await pool.acquire(timeout=5.0)
    assert resource is not None
    assert resource['status'] == 'active'
    
    await pool.release(resource)
    
    # Test pool stats
    stats = pool.get_stats()
    assert stats['total_acquisitions'] >= 1
    assert stats['total_releases'] >= 1
    
    # Test manager summary
    summary = manager.get_summary()
    assert summary['total_pools'] == 1
    
    await manager.shutdown()
    
    logger.info("ResourcePoolManager test passed!")


async def test_planner_optimizer():
    """Test the planner optimizer."""
    
    logger.info("Testing PlannerOptimizer...")
    
    from planner_optimizer import PlannerOptimizer
    
    config = {
        'max_parallel_scenes': 4,
        'max_concurrent_implementations': 3,
        'rag_batch_size': 5,
        'enable_scene_caching': True,
        'enable_implementation_caching': True,
        'enable_plugin_caching': True,
        'max_worker_threads': 2
    }
    
    optimizer = PlannerOptimizer(config)
    await optimizer.start()
    
    # Test scene outline optimization
    async def mock_planner_func(topic, description, session_id):
        await asyncio.sleep(0.1)  # Simulate work
        return f"Scene outline for {topic}"
    
    outline = await optimizer.optimize_scene_outline_generation(
        planner_func=mock_planner_func,
        topic="Test Topic",
        description="Test Description",
        session_id="test_session"
    )
    
    assert outline == "Scene outline for Test Topic"
    
    # Test caching (second call should be faster)
    start_time = time.time()
    outline2 = await optimizer.optimize_scene_outline_generation(
        planner_func=mock_planner_func,
        topic="Test Topic",
        description="Test Description",
        session_id="test_session"
    )
    cache_time = time.time() - start_time
    
    assert outline2 == outline
    assert cache_time < 0.05  # Should be much faster due to caching
    
    # Test plugin detection optimization
    async def mock_plugin_detection(topic, description):
        await asyncio.sleep(0.05)
        return ["plugin1", "plugin2"]
    
    plugins = await optimizer.optimize_plugin_detection(
        detection_func=mock_plugin_detection,
        topic="Test Topic",
        description="Test Description"
    )
    
    assert plugins == ["plugin1", "plugin2"]
    
    # Test metrics
    metrics = optimizer.get_optimization_metrics()
    assert 'optimization_summary' in metrics
    assert 'cache_statistics' in metrics
    
    # Test cache clearing
    optimizer.clear_caches(['scene'])
    
    await optimizer.stop()
    
    logger.info("PlannerOptimizer test passed!")


async def test_performance_optimizer():
    """Test the main performance optimizer."""
    
    logger.info("Testing PerformanceOptimizer...")
    
    from performance_optimizer import PerformanceOptimizer, OptimizationConfig, OptimizationStrategy
    
    config = OptimizationConfig(
        strategy=OptimizationStrategy.BALANCED,
        max_concurrent_tasks=5,
        max_concurrent_per_agent=2,
        max_scene_concurrency=4,
        enable_aggressive_caching=True,
        max_cache_memory_mb=50,  # Small for testing
        enable_memory_optimization=True
    )
    
    optimizer = PerformanceOptimizer(config)
    await optimizer.start()
    
    # Test planner optimization
    planner_result = await optimizer.optimize_planner_performance(None)
    assert planner_result['success'] == True
    assert len(planner_result['optimizations_applied']) > 0
    
    # Test code generation optimization
    code_result = await optimizer.optimize_code_generation_performance()
    assert code_result['success'] == True
    assert len(code_result['optimizations_applied']) > 0
    
    # Test rendering optimization
    render_result = await optimizer.optimize_rendering_performance()
    assert render_result['success'] == True
    assert len(render_result['optimizations_applied']) > 0
    
    # Test optimization status
    status = optimizer.get_optimization_status()
    assert status['strategy'] == OptimizationStrategy.BALANCED.value
    assert len(status['active_optimizations']) > 0
    
    # Test performance report
    report = optimizer.get_performance_report()
    assert 'performance_summary' in report
    assert 'component_performance' in report
    
    await optimizer.stop()
    
    logger.info("PerformanceOptimizer test passed!")


async def test_performance_integration():
    """Test the performance integration."""
    
    logger.info("Testing PerformanceIntegration...")
    
    from integration import PerformanceIntegration
    
    system_config = {
        'enable_performance_optimization': True,
        'optimization_strategy': 'balanced',
        'max_concurrent_tasks': 5,
        'max_concurrent_per_agent': 2,
        'max_scene_concurrency': 4,
        'enable_planner_optimization': True,
        'enable_memory_optimization': True,
        'max_cache_memory_mb': 50
    }
    
    integration = PerformanceIntegration(system_config)
    await integration.initialize()
    
    # Test workflow optimization
    workflow_config = {
        'optimize_planner': True,
        'optimize_code_generation': True,
        'optimize_rendering': True
    }
    
    result = await integration.optimize_workflow_performance(workflow_config)
    assert result['success'] == True
    assert len(result['optimizations_applied']) > 0
    
    # Test performance status
    status = integration.get_performance_status()
    assert status['integration_status']['initialized'] == True
    assert status['integration_status']['optimization_enabled'] == True
    
    # Test performance report
    report = integration.get_performance_report()
    assert 'integration_info' in report
    assert 'performance_optimizer' in report
    
    # Test cache clearing
    await integration.clear_all_caches()
    
    # Test memory cleanup
    await integration.force_memory_cleanup()
    
    await integration.shutdown()
    
    logger.info("PerformanceIntegration test passed!")


async def run_all_tests():
    """Run all performance optimization tests."""
    
    logger.info("Starting performance optimization tests...")
    
    try:
        await test_concurrency_manager()
        await test_cache_manager()
        await test_memory_manager()
        await test_resource_pool()
        await test_planner_optimizer()
        await test_performance_optimizer()
        await test_performance_integration()
        
        logger.info("All performance optimization tests passed!")
        
    except Exception as e:
        logger.error(f"Performance optimization test failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(run_all_tests())
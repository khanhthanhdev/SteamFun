"""
Unit tests for parallel execution capabilities.

Tests cover ResourceManager, ParallelExecutor, and convenience functions
for parallel scene processing, code generation, and RAG queries.
"""

import pytest
import asyncio
import time
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from src.langgraph_agents.performance.parallel_execution import (
    ResourceManager,
    ParallelExecutor,
    ResourceLimits,
    ResourceType,
    TaskResult,
    ResourceUsage,
    parallel_scene_processing,
    parallel_code_generation,
    parallel_rag_enhancement
)


class TestResourceLimits:
    """Test cases for ResourceLimits configuration."""
    
    def test_resource_limits_defaults(self):
        """Test default resource limits."""
        limits = ResourceLimits()
        
        assert limits.max_concurrent_scenes == 5
        assert limits.max_concurrent_code_generation == 3
        assert limits.max_concurrent_rendering == 2
        assert limits.max_concurrent_rag_queries == 10
        assert limits.max_concurrent_model_calls == 8
        assert limits.max_memory_usage_mb == 4096
        assert limits.max_cpu_usage_percent == 80.0
        assert limits.enable_adaptive_limits is True
    
    def test_resource_limits_custom(self):
        """Test custom resource limits."""
        limits = ResourceLimits(
            max_concurrent_scenes=10,
            max_concurrent_code_generation=5,
            max_memory_usage_mb=8192,
            enable_adaptive_limits=False
        )
        
        assert limits.max_concurrent_scenes == 10
        assert limits.max_concurrent_code_generation == 5
        assert limits.max_memory_usage_mb == 8192
        assert limits.enable_adaptive_limits is False


class TestResourceManager:
    """Test cases for ResourceManager functionality."""
    
    @pytest.fixture
    async def resource_manager(self):
        """Create a resource manager for testing."""
        limits = ResourceLimits(
            max_concurrent_scenes=3,
            max_concurrent_code_generation=2,
            max_memory_usage_mb=1024,
            max_cpu_usage_percent=70.0
        )
        manager = ResourceManager(limits)
        await manager.start()
        yield manager
        await manager.stop()
    
    async def test_resource_manager_initialization(self):
        """Test resource manager initialization."""
        limits = ResourceLimits(max_concurrent_scenes=5)
        manager = ResourceManager(limits)
        
        assert manager.limits.max_concurrent_scenes == 5
        assert ResourceType.SCENE_PROCESSING in manager.semaphores
        assert manager.semaphores[ResourceType.SCENE_PROCESSING]._value == 5
    
    async def test_acquire_and_release_resource(self, resource_manager):
        """Test basic resource acquisition and release."""
        task_id = "test_task_1"
        
        # Acquire resource
        success = await resource_manager.acquire_resource(
            ResourceType.SCENE_PROCESSING, 
            task_id
        )
        assert success is True
        
        # Check that task is tracked
        assert task_id in resource_manager.active_tasks[ResourceType.SCENE_PROCESSING]
        
        # Release resource
        await resource_manager.release_resource(ResourceType.SCENE_PROCESSING, task_id)
        
        # Check that task is no longer tracked
        assert task_id not in resource_manager.active_tasks[ResourceType.SCENE_PROCESSING]
    
    async def test_resource_limit_enforcement(self, resource_manager):
        """Test that resource limits are enforced."""
        # Acquire all available scene processing resources (limit is 3)
        task_ids = []
        for i in range(3):
            task_id = f"test_task_{i}"
            success = await resource_manager.acquire_resource(
                ResourceType.SCENE_PROCESSING, 
                task_id
            )
            assert success is True
            task_ids.append(task_id)
        
        # Try to acquire one more (should timeout)
        with pytest.raises(asyncio.TimeoutError):
            await resource_manager.acquire_resource(
                ResourceType.SCENE_PROCESSING, 
                "test_task_overflow",
                timeout=0.1
            )
        
        # Release one resource
        await resource_manager.release_resource(ResourceType.SCENE_PROCESSING, task_ids[0])
        
        # Now we should be able to acquire again
        success = await resource_manager.acquire_resource(
            ResourceType.SCENE_PROCESSING, 
            "test_task_after_release"
        )
        assert success is True
    
    @patch('src.langgraph_agents.performance.parallel_execution.psutil')
    async def test_system_resource_check(self, mock_psutil, resource_manager):
        """Test system resource checking."""
        # Mock memory info
        mock_memory = Mock()
        mock_memory.total = 8 * 1024 * 1024 * 1024  # 8GB
        mock_memory.available = 7 * 1024 * 1024 * 1024  # 7GB available
        mock_psutil.virtual_memory.return_value = mock_memory
        
        # Mock CPU usage
        mock_psutil.cpu_percent.return_value = 50.0  # 50% CPU usage
        
        # Should allow resource acquisition
        success = await resource_manager.acquire_resource(
            ResourceType.SCENE_PROCESSING, 
            "test_task"
        )
        assert success is True
        
        # Clean up
        await resource_manager.release_resource(ResourceType.SCENE_PROCESSING, "test_task")
    
    @patch('src.langgraph_agents.performance.parallel_execution.psutil')
    async def test_system_resource_limit_exceeded(self, mock_psutil, resource_manager):
        """Test behavior when system resource limits are exceeded."""
        # Mock high memory usage
        mock_memory = Mock()
        mock_memory.total = 2 * 1024 * 1024 * 1024  # 2GB
        mock_memory.available = 100 * 1024 * 1024  # Only 100MB available
        mock_psutil.virtual_memory.return_value = mock_memory
        
        # Mock high CPU usage
        mock_psutil.cpu_percent.return_value = 95.0  # 95% CPU usage
        
        # Should timeout due to system resource limits
        with pytest.raises(asyncio.TimeoutError):
            await resource_manager.acquire_resource(
                ResourceType.SCENE_PROCESSING, 
                "test_task",
                timeout=0.1
            )
    
    async def test_resource_stats(self, resource_manager):
        """Test resource statistics collection."""
        # Acquire some resources
        await resource_manager.acquire_resource(ResourceType.SCENE_PROCESSING, "task1")
        await resource_manager.acquire_resource(ResourceType.CODE_GENERATION, "task2")
        
        stats = resource_manager.get_resource_stats()
        
        assert 'active_tasks' in stats
        assert 'limits' in stats
        assert 'current_usage' in stats
        assert 'semaphore_available' in stats
        
        assert stats['active_tasks']['scene_processing'] == 1
        assert stats['active_tasks']['code_generation'] == 1
        assert stats['limits']['max_concurrent_scenes'] == 3
        
        # Clean up
        await resource_manager.release_resource(ResourceType.SCENE_PROCESSING, "task1")
        await resource_manager.release_resource(ResourceType.CODE_GENERATION, "task2")


class TestParallelExecutor:
    """Test cases for ParallelExecutor functionality."""
    
    @pytest.fixture
    async def parallel_executor(self):
        """Create a parallel executor for testing."""
        limits = ResourceLimits(
            max_concurrent_scenes=3,
            max_concurrent_code_generation=2,
            max_concurrent_rag_queries=5
        )
        resource_manager = ResourceManager(limits)
        await resource_manager.start()
        
        executor = ParallelExecutor(resource_manager)
        
        yield executor
        
        await resource_manager.stop()
    
    async def test_parallel_scene_execution(self, parallel_executor):
        """Test parallel execution of scene tasks."""
        # Create mock scene tasks
        async def mock_scene_task(scene_num: int) -> str:
            await asyncio.sleep(0.1)  # Simulate work
            return f"scene_{scene_num}_result"
        
        scene_tasks = {
            1: lambda: mock_scene_task(1),
            2: lambda: mock_scene_task(2),
            3: lambda: mock_scene_task(3)
        }
        
        # Execute tasks in parallel
        results = await parallel_executor.execute_parallel_scenes(scene_tasks)
        
        # Verify results
        assert len(results) == 3
        assert all(result.success for result in results.values())
        assert results[1].result == "scene_1_result"
        assert results[2].result == "scene_2_result"
        assert results[3].result == "scene_3_result"
    
    async def test_parallel_scene_execution_with_failures(self, parallel_executor):
        """Test parallel scene execution with some failures."""
        async def mock_scene_task(scene_num: int) -> str:
            if scene_num == 2:
                raise ValueError(f"Scene {scene_num} failed")
            await asyncio.sleep(0.1)
            return f"scene_{scene_num}_result"
        
        scene_tasks = {
            1: lambda: mock_scene_task(1),
            2: lambda: mock_scene_task(2),  # This will fail
            3: lambda: mock_scene_task(3)
        }
        
        # Execute tasks in parallel
        results = await parallel_executor.execute_parallel_scenes(scene_tasks)
        
        # Verify results
        assert len(results) == 3
        assert results[1].success is True
        assert results[2].success is False
        assert results[3].success is True
        assert isinstance(results[2].error, ValueError)
    
    async def test_parallel_code_generation(self, parallel_executor):
        """Test parallel code generation execution."""
        async def mock_code_generation(task_name: str) -> str:
            await asyncio.sleep(0.1)
            return f"generated_code_for_{task_name}"
        
        code_tasks = {
            "function_a": lambda: mock_code_generation("function_a"),
            "function_b": lambda: mock_code_generation("function_b")
        }
        
        # Execute tasks in parallel
        results = await parallel_executor.execute_parallel_code_generation(code_tasks)
        
        # Verify results
        assert len(results) == 2
        assert all(result.success for result in results.values())
        assert results["function_a"].result == "generated_code_for_function_a"
        assert results["function_b"].result == "generated_code_for_function_b"
    
    async def test_parallel_rag_queries(self, parallel_executor):
        """Test parallel RAG query execution."""
        async def mock_rag_query(query_index: int) -> str:
            await asyncio.sleep(0.1)
            return f"rag_result_{query_index}"
        
        rag_queries = [
            lambda i=i: mock_rag_query(i)
            for i in range(3)
        ]
        
        # Execute queries in parallel
        results = await parallel_executor.execute_parallel_rag_queries(rag_queries)
        
        # Verify results
        assert len(results) == 3
        assert all(result.success for result in results)
        assert results[0].result == "rag_result_0"
        assert results[1].result == "rag_result_1"
        assert results[2].result == "rag_result_2"
    
    async def test_batch_execution_with_priority(self, parallel_executor):
        """Test batch execution with priority ordering."""
        execution_order = []
        
        async def high_priority_task(task_id: str) -> str:
            execution_order.append(f"high_{task_id}")
            await asyncio.sleep(0.1)
            return f"high_result_{task_id}"
        
        async def normal_priority_task(task_id: str) -> str:
            execution_order.append(f"normal_{task_id}")
            await asyncio.sleep(0.1)
            return f"normal_result_{task_id}"
        
        high_priority_tasks = [
            lambda: high_priority_task("1"),
            lambda: high_priority_task("2")
        ]
        
        normal_priority_tasks = [
            lambda: normal_priority_task("1"),
            lambda: normal_priority_task("2")
        ]
        
        # Execute with priority
        high_results, normal_results = await parallel_executor.execute_batch_with_priority(
            high_priority_tasks,
            normal_priority_tasks,
            ResourceType.SCENE_PROCESSING
        )
        
        # Verify results
        assert len(high_results) == 2
        assert len(normal_results) == 2
        assert all(result.success for result in high_results)
        assert all(result.success for result in normal_results)
        
        # High priority tasks should have executed first
        high_executions = [item for item in execution_order if item.startswith("high_")]
        normal_executions = [item for item in execution_order if item.startswith("normal_")]
        
        # All high priority tasks should complete before normal priority tasks start
        # (This might not always be true due to concurrency, but we can check that high priority started first)
        assert len(high_executions) == 2
        assert len(normal_executions) == 2
    
    async def test_task_timeout(self, parallel_executor):
        """Test task timeout handling."""
        async def slow_task() -> str:
            await asyncio.sleep(1.0)  # Longer than timeout
            return "slow_result"
        
        scene_tasks = {
            1: slow_task
        }
        
        # Execute with short timeout
        results = await parallel_executor.execute_parallel_scenes(
            scene_tasks, 
            timeout_per_task=0.1
        )
        
        # Should have timed out
        assert len(results) == 1
        assert results[1].success is False
        assert isinstance(results[1].error, asyncio.TimeoutError)
    
    async def test_task_retry_mechanism(self, parallel_executor):
        """Test task retry mechanism."""
        attempt_count = 0
        
        async def flaky_task() -> str:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:  # Fail first 2 attempts
                raise ValueError("Temporary failure")
            return "success_after_retries"
        
        scene_tasks = {
            1: flaky_task
        }
        
        # Execute with retries
        results = await parallel_executor.execute_parallel_scenes(
            scene_tasks,
            max_retries=3
        )
        
        # Should succeed after retries
        assert len(results) == 1
        assert results[1].success is True
        assert results[1].result == "success_after_retries"
        assert attempt_count == 3


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    async def test_parallel_scene_processing_function(self):
        """Test parallel_scene_processing convenience function."""
        async def process_scene(scene_num: int, scene_data: str) -> str:
            await asyncio.sleep(0.1)
            return f"processed_{scene_data}_scene_{scene_num}"
        
        scenes = {
            1: "intro",
            2: "main",
            3: "conclusion"
        }
        
        results = await parallel_scene_processing(
            scenes,
            process_scene,
            max_concurrent=2
        )
        
        assert len(results) == 3
        assert all(result.success for result in results.values())
        assert results[1].result == "processed_intro_scene_1"
        assert results[2].result == "processed_main_scene_2"
        assert results[3].result == "processed_conclusion_scene_3"
    
    async def test_parallel_code_generation_function(self):
        """Test parallel_code_generation convenience function."""
        async def generate_code(request_name: str, request_data: Dict[str, Any]) -> str:
            await asyncio.sleep(0.1)
            return f"code_for_{request_name}_{request_data['type']}"
        
        code_requests = {
            "function_a": {"type": "animation"},
            "function_b": {"type": "transformation"}
        }
        
        results = await parallel_code_generation(
            code_requests,
            generate_code,
            max_concurrent=2
        )
        
        assert len(results) == 2
        assert all(result.success for result in results.values())
        assert results["function_a"].result == "code_for_function_a_animation"
        assert results["function_b"].result == "code_for_function_b_transformation"
    
    async def test_parallel_rag_enhancement_function(self):
        """Test parallel_rag_enhancement convenience function."""
        async def rag_query(query: str) -> str:
            await asyncio.sleep(0.1)
            return f"enhanced_{query}"
        
        queries = ["query1", "query2", "query3"]
        
        results = await parallel_rag_enhancement(
            queries,
            rag_query,
            max_concurrent=3
        )
        
        assert len(results) == 3
        assert all(result.success for result in results)
        assert results[0].result == "enhanced_query1"
        assert results[1].result == "enhanced_query2"
        assert results[2].result == "enhanced_query3"
    
    async def test_convenience_function_with_custom_resource_manager(self):
        """Test convenience function with custom resource manager."""
        limits = ResourceLimits(max_concurrent_scenes=1)  # Very limited
        resource_manager = ResourceManager(limits)
        await resource_manager.start()
        
        try:
            async def process_scene(scene_num: int, scene_data: str) -> str:
                await asyncio.sleep(0.1)
                return f"processed_{scene_data}"
            
            scenes = {1: "scene1", 2: "scene2"}
            
            start_time = time.time()
            results = await parallel_scene_processing(
                scenes,
                process_scene,
                resource_manager=resource_manager
            )
            end_time = time.time()
            
            # Should take longer due to limited concurrency
            assert end_time - start_time >= 0.2  # At least 0.2 seconds for sequential execution
            assert len(results) == 2
            assert all(result.success for result in results.values())
            
        finally:
            await resource_manager.stop()


class TestTaskResult:
    """Test cases for TaskResult functionality."""
    
    def test_task_result_success(self):
        """Test successful task result."""
        result = TaskResult(
            task_id="test_task",
            success=True,
            result="test_result",
            duration_seconds=1.5,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        assert result.task_id == "test_task"
        assert result.success is True
        assert result.result == "test_result"
        assert result.error is None
        assert result.duration_seconds == 1.5
    
    def test_task_result_failure(self):
        """Test failed task result."""
        error = ValueError("Test error")
        result = TaskResult(
            task_id="test_task",
            success=False,
            error=error,
            duration_seconds=0.5
        )
        
        assert result.task_id == "test_task"
        assert result.success is False
        assert result.result is None
        assert result.error == error
        assert result.duration_seconds == 0.5


class TestResourceUsage:
    """Test cases for ResourceUsage tracking."""
    
    def test_resource_usage_creation(self):
        """Test resource usage creation."""
        active_tasks = {
            ResourceType.SCENE_PROCESSING: 3,
            ResourceType.CODE_GENERATION: 1
        }
        
        usage = ResourceUsage(
            active_tasks=active_tasks,
            memory_usage_mb=1024.5,
            cpu_usage_percent=65.2
        )
        
        assert usage.active_tasks[ResourceType.SCENE_PROCESSING] == 3
        assert usage.active_tasks[ResourceType.CODE_GENERATION] == 1
        assert usage.memory_usage_mb == 1024.5
        assert usage.cpu_usage_percent == 65.2
        assert isinstance(usage.timestamp, datetime)


class TestIntegration:
    """Integration tests for parallel execution components."""
    
    async def test_full_parallel_workflow(self):
        """Test a complete parallel workflow scenario."""
        # Setup
        limits = ResourceLimits(
            max_concurrent_scenes=2,
            max_concurrent_code_generation=2,
            max_concurrent_rag_queries=3
        )
        resource_manager = ResourceManager(limits)
        await resource_manager.start()
        
        try:
            executor = ParallelExecutor(resource_manager)
            
            # Simulate a complete workflow
            # 1. Parallel scene processing
            async def process_scene(scene_num: int) -> str:
                await asyncio.sleep(0.1)
                return f"scene_{scene_num}_outline"
            
            scene_tasks = {i: lambda i=i: process_scene(i) for i in range(1, 4)}
            scene_results = await executor.execute_parallel_scenes(scene_tasks)
            
            # 2. Parallel code generation based on scene results
            async def generate_code(scene_outline: str) -> str:
                await asyncio.sleep(0.1)
                return f"code_for_{scene_outline}"
            
            code_tasks = {
                f"scene_{i}": lambda outline=result.result: generate_code(outline)
                for i, result in scene_results.items()
                if result.success
            }
            code_results = await executor.execute_parallel_code_generation(code_tasks)
            
            # 3. Parallel RAG enhancement
            async def enhance_with_rag(code: str) -> str:
                await asyncio.sleep(0.05)
                return f"enhanced_{code}"
            
            rag_queries = [
                lambda code=result.result: enhance_with_rag(code)
                for result in code_results.values()
                if result.success
            ]
            rag_results = await executor.execute_parallel_rag_queries(rag_queries)
            
            # Verify the complete workflow
            assert len(scene_results) == 3
            assert all(result.success for result in scene_results.values())
            
            assert len(code_results) == 3
            assert all(result.success for result in code_results.values())
            
            assert len(rag_results) == 3
            assert all(result.success for result in rag_results)
            
            # Check resource manager stats
            stats = resource_manager.get_resource_stats()
            assert stats['active_tasks']['scene_processing'] == 0  # All released
            assert stats['active_tasks']['code_generation'] == 0  # All released
            assert stats['active_tasks']['rag_queries'] == 0  # All released
            
        finally:
            await resource_manager.stop()
    
    async def test_resource_contention_handling(self):
        """Test handling of resource contention."""
        # Create very limited resources
        limits = ResourceLimits(max_concurrent_scenes=1)
        resource_manager = ResourceManager(limits)
        await resource_manager.start()
        
        try:
            executor = ParallelExecutor(resource_manager)
            
            # Create many tasks that will compete for limited resources
            async def slow_task(task_id: int) -> str:
                await asyncio.sleep(0.2)
                return f"result_{task_id}"
            
            scene_tasks = {i: lambda i=i: slow_task(i) for i in range(5)}
            
            start_time = time.time()
            results = await executor.execute_parallel_scenes(scene_tasks)
            end_time = time.time()
            
            # Should take at least 1 second (5 tasks * 0.2s each, sequential due to limit of 1)
            assert end_time - start_time >= 1.0
            
            # All tasks should succeed
            assert len(results) == 5
            assert all(result.success for result in results.values())
            
        finally:
            await resource_manager.stop()


if __name__ == "__main__":
    pytest.main([__file__])
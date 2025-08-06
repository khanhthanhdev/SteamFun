"""
Parallel execution capabilities for video generation workflow.

This module provides parallel scene processing functions, ResourceManager
for concurrent operation limits, and semaphore-based resource control.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable, TypeVar, Generic
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import psutil
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class ResourceType(Enum):
    """Types of resources that can be managed."""
    SCENE_PROCESSING = "scene_processing"
    CODE_GENERATION = "code_generation"
    RENDERING = "rendering"
    RAG_QUERIES = "rag_queries"
    MODEL_CALLS = "model_calls"
    FILE_IO = "file_io"
    MEMORY = "memory"
    CPU = "cpu"


@dataclass
class ResourceLimits:
    """Configuration for resource limits."""
    max_concurrent_scenes: int = 5
    max_concurrent_code_generation: int = 3
    max_concurrent_rendering: int = 2
    max_concurrent_rag_queries: int = 10
    max_concurrent_model_calls: int = 8
    max_memory_usage_mb: int = 4096
    max_cpu_usage_percent: float = 80.0
    enable_adaptive_limits: bool = True


@dataclass
class TaskResult(Generic[T]):
    """Result of a parallel task execution."""
    task_id: str
    success: bool
    result: Optional[T] = None
    error: Optional[Exception] = None
    duration_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class ResourceUsage:
    """Current resource usage information."""
    active_tasks: Dict[ResourceType, int] = field(default_factory=dict)
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class ResourceManager:
    """Manager for concurrent operation limits with semaphore-based resource control."""
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        """Initialize the resource manager.
        
        Args:
            limits: Resource limits configuration
        """
        self.limits = limits or ResourceLimits()
        
        # Semaphores for different resource types
        self.semaphores: Dict[ResourceType, asyncio.Semaphore] = {
            ResourceType.SCENE_PROCESSING: asyncio.Semaphore(self.limits.max_concurrent_scenes),
            ResourceType.CODE_GENERATION: asyncio.Semaphore(self.limits.max_concurrent_code_generation),
            ResourceType.RENDERING: asyncio.Semaphore(self.limits.max_concurrent_rendering),
            ResourceType.RAG_QUERIES: asyncio.Semaphore(self.limits.max_concurrent_rag_queries),
            ResourceType.MODEL_CALLS: asyncio.Semaphore(self.limits.max_concurrent_model_calls),
        }
        
        # Resource tracking
        self.active_tasks: Dict[ResourceType, Dict[str, datetime]] = defaultdict(dict)
        self.resource_usage_history: deque = deque(maxlen=100)
        
        # Performance monitoring
        self.last_resource_check = datetime.now()
        self.resource_check_interval = 30  # seconds
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background monitoring task
        self._monitor_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        logger.info(f"ResourceManager initialized with limits: {self.limits}")
    
    async def start(self):
        """Start the resource manager and monitoring."""
        if self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._monitor_resources())
            logger.info("ResourceManager started")
    
    async def stop(self):
        """Stop the resource manager."""
        self._shutdown = True
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        logger.info("ResourceManager stopped")
    
    async def acquire_resource(self, resource_type: ResourceType, task_id: str, timeout: Optional[float] = None) -> bool:
        """Acquire a resource for a task.
        
        Args:
            resource_type: Type of resource to acquire
            task_id: Unique identifier for the task
            timeout: Maximum time to wait for resource
            
        Returns:
            True if resource was acquired
            
        Raises:
            asyncio.TimeoutError: If timeout is reached
        """
        semaphore = self.semaphores.get(resource_type)
        if not semaphore:
            return True  # No limit for this resource type
        
        # Check system resource limits before acquiring
        if not await self._check_system_resources():
            if timeout is not None and timeout > 0:
                # Wait a bit and try again
                await asyncio.sleep(min(1.0, timeout / 10))
                if not await self._check_system_resources():
                    raise asyncio.TimeoutError("System resource limits exceeded")
            else:
                raise asyncio.TimeoutError("System resource limits exceeded")
        
        # Acquire semaphore
        try:
            if timeout is not None:
                await asyncio.wait_for(semaphore.acquire(), timeout=timeout)
            else:
                await semaphore.acquire()
            
            # Track the task
            with self._lock:
                self.active_tasks[resource_type][task_id] = datetime.now()
            
            logger.debug(f"Resource acquired: {resource_type.value} for task {task_id}")
            return True
            
        except asyncio.TimeoutError:
            logger.warning(f"Resource acquisition timeout: {resource_type.value} for task {task_id}")
            raise
    
    async def release_resource(self, resource_type: ResourceType, task_id: str):
        """Release a resource from a task.
        
        Args:
            resource_type: Type of resource to release
            task_id: Unique identifier for the task
        """
        semaphore = self.semaphores.get(resource_type)
        if not semaphore:
            return  # No limit for this resource type
        
        # Remove from tracking
        with self._lock:
            self.active_tasks[resource_type].pop(task_id, None)
        
        # Release semaphore
        semaphore.release()
        
        logger.debug(f"Resource released: {resource_type.value} for task {task_id}")
    
    async def _check_system_resources(self) -> bool:
        """Check if system resources are within limits."""
        try:
            # Check memory usage
            memory_info = psutil.virtual_memory()
            memory_usage_mb = (memory_info.total - memory_info.available) / (1024 * 1024)
            
            if memory_usage_mb > self.limits.max_memory_usage_mb:
                logger.warning(f"Memory usage ({memory_usage_mb:.1f} MB) exceeds limit ({self.limits.max_memory_usage_mb} MB)")
                return False
            
            # Check CPU usage
            cpu_usage = psutil.cpu_percent(interval=0.1)
            if cpu_usage > self.limits.max_cpu_usage_percent:
                logger.warning(f"CPU usage ({cpu_usage:.1f}%) exceeds limit ({self.limits.max_cpu_usage_percent}%)")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking system resources: {str(e)}")
            return True  # Assume OK if we can't check
    
    async def _monitor_resources(self):
        """Background task to monitor resource usage."""
        try:
            while not self._shutdown:
                await asyncio.sleep(self.resource_check_interval)
                
                if self._shutdown:
                    break
                
                # Collect resource usage
                try:
                    memory_info = psutil.virtual_memory()
                    memory_usage_mb = (memory_info.total - memory_info.available) / (1024 * 1024)
                    cpu_usage = psutil.cpu_percent(interval=1.0)
                    
                    with self._lock:
                        active_counts = {
                            resource_type: len(tasks)
                            for resource_type, tasks in self.active_tasks.items()
                        }
                    
                    usage = ResourceUsage(
                        active_tasks=active_counts,
                        memory_usage_mb=memory_usage_mb,
                        cpu_usage_percent=cpu_usage
                    )
                    
                    self.resource_usage_history.append(usage)
                    
                    # Adaptive limit adjustment if enabled
                    if self.limits.enable_adaptive_limits:
                        await self._adjust_limits_adaptively(usage)
                    
                except Exception as e:
                    logger.error(f"Error monitoring resources: {str(e)}")
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Resource monitoring error: {str(e)}")
    
    async def _adjust_limits_adaptively(self, usage: ResourceUsage):
        """Adjust resource limits based on current usage patterns."""
        # Simple adaptive logic - can be enhanced
        
        # If memory usage is low and CPU usage is low, we can increase limits
        if (usage.memory_usage_mb < self.limits.max_memory_usage_mb * 0.6 and
            usage.cpu_usage_percent < self.limits.max_cpu_usage_percent * 0.6):
            
            # Increase scene processing limit if it's being heavily used
            scene_usage = usage.active_tasks.get(ResourceType.SCENE_PROCESSING, 0)
            if scene_usage >= self.limits.max_concurrent_scenes * 0.8:
                old_limit = self.limits.max_concurrent_scenes
                self.limits.max_concurrent_scenes = min(old_limit + 1, old_limit * 2)
                
                # Add permits to semaphore
                for _ in range(self.limits.max_concurrent_scenes - old_limit):
                    self.semaphores[ResourceType.SCENE_PROCESSING].release()
                
                logger.info(f"Increased scene processing limit: {old_limit} -> {self.limits.max_concurrent_scenes}")
        
        # If resource usage is high, decrease limits
        elif (usage.memory_usage_mb > self.limits.max_memory_usage_mb * 0.8 or
              usage.cpu_usage_percent > self.limits.max_cpu_usage_percent * 0.8):
            
            # Decrease limits by acquiring permits (reducing available capacity)
            for resource_type in [ResourceType.SCENE_PROCESSING, ResourceType.CODE_GENERATION]:
                try:
                    # Try to acquire a permit to reduce capacity
                    semaphore = self.semaphores[resource_type]
                    acquired = semaphore.acquire_nowait()
                    if acquired:
                        logger.info(f"Temporarily reduced {resource_type.value} capacity due to high resource usage")
                except:
                    pass  # Ignore if we can't acquire
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get current resource usage statistics."""
        with self._lock:
            active_counts = {
                resource_type.value: len(tasks)
                for resource_type, tasks in self.active_tasks.items()
            }
        
        # Get current system stats
        try:
            memory_info = psutil.virtual_memory()
            memory_usage_mb = (memory_info.total - memory_info.available) / (1024 * 1024)
            cpu_usage = psutil.cpu_percent(interval=0.1)
        except:
            memory_usage_mb = 0.0
            cpu_usage = 0.0
        
        return {
            'active_tasks': active_counts,
            'limits': {
                'max_concurrent_scenes': self.limits.max_concurrent_scenes,
                'max_concurrent_code_generation': self.limits.max_concurrent_code_generation,
                'max_concurrent_rendering': self.limits.max_concurrent_rendering,
                'max_concurrent_rag_queries': self.limits.max_concurrent_rag_queries,
                'max_concurrent_model_calls': self.limits.max_concurrent_model_calls,
                'max_memory_usage_mb': self.limits.max_memory_usage_mb,
                'max_cpu_usage_percent': self.limits.max_cpu_usage_percent
            },
            'current_usage': {
                'memory_usage_mb': memory_usage_mb,
                'cpu_usage_percent': cpu_usage
            },
            'semaphore_available': {
                resource_type.value: semaphore._value
                for resource_type, semaphore in self.semaphores.items()
            }
        }


class ParallelExecutor:
    """Executor for parallel task processing with resource management."""
    
    def __init__(self, resource_manager: ResourceManager):
        """Initialize the parallel executor.
        
        Args:
            resource_manager: Resource manager for controlling concurrency
        """
        self.resource_manager = resource_manager
        self.task_counter = 0
        self._lock = threading.Lock()
    
    def _generate_task_id(self, prefix: str = "task") -> str:
        """Generate a unique task ID."""
        with self._lock:
            self.task_counter += 1
            return f"{prefix}_{self.task_counter}_{int(time.time())}"
    
    async def execute_parallel_scenes(
        self,
        scene_tasks: Dict[int, Callable[[], Awaitable[T]]],
        timeout_per_task: Optional[float] = None,
        max_retries: int = 2
    ) -> Dict[int, TaskResult[T]]:
        """Execute multiple scene processing tasks in parallel.
        
        Args:
            scene_tasks: Dictionary mapping scene numbers to async task functions
            timeout_per_task: Timeout for each individual task
            max_retries: Maximum number of retries for failed tasks
            
        Returns:
            Dictionary mapping scene numbers to task results
        """
        if not scene_tasks:
            return {}
        
        logger.info(f"Starting parallel execution of {len(scene_tasks)} scene tasks")
        
        # Create tasks with resource management
        async def execute_scene_task(scene_num: int, task_func: Callable[[], Awaitable[T]]) -> TaskResult[T]:
            task_id = self._generate_task_id(f"scene_{scene_num}")
            start_time = datetime.now()
            
            for attempt in range(max_retries + 1):
                try:
                    # Acquire resource
                    await self.resource_manager.acquire_resource(
                        ResourceType.SCENE_PROCESSING, 
                        task_id, 
                        timeout=timeout_per_task
                    )
                    
                    try:
                        # Execute the task
                        if timeout_per_task:
                            result = await asyncio.wait_for(task_func(), timeout=timeout_per_task)
                        else:
                            result = await task_func()
                        
                        end_time = datetime.now()
                        duration = (end_time - start_time).total_seconds()
                        
                        return TaskResult(
                            task_id=task_id,
                            success=True,
                            result=result,
                            duration_seconds=duration,
                            start_time=start_time,
                            end_time=end_time
                        )
                        
                    finally:
                        # Always release resource
                        await self.resource_manager.release_resource(ResourceType.SCENE_PROCESSING, task_id)
                
                except Exception as e:
                    if attempt < max_retries:
                        logger.warning(f"Scene task {scene_num} failed (attempt {attempt + 1}), retrying: {str(e)}")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        end_time = datetime.now()
                        duration = (end_time - start_time).total_seconds()
                        
                        return TaskResult(
                            task_id=task_id,
                            success=False,
                            error=e,
                            duration_seconds=duration,
                            start_time=start_time,
                            end_time=end_time
                        )
            
            # This should never be reached, but just in case
            return TaskResult(task_id=task_id, success=False, error=Exception("Unexpected error"))
        
        # Execute all scene tasks in parallel
        tasks = [
            execute_scene_task(scene_num, task_func)
            for scene_num, task_func in scene_tasks.items()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        scene_results = {}
        for i, (scene_num, result) in enumerate(zip(scene_tasks.keys(), results)):
            if isinstance(result, Exception):
                scene_results[scene_num] = TaskResult(
                    task_id=f"scene_{scene_num}_error",
                    success=False,
                    error=result
                )
            else:
                scene_results[scene_num] = result
        
        successful_count = sum(1 for r in scene_results.values() if r.success)
        logger.info(f"Parallel scene execution completed: {successful_count}/{len(scene_tasks)} successful")
        
        return scene_results
    
    async def execute_parallel_code_generation(
        self,
        code_tasks: Dict[str, Callable[[], Awaitable[T]]],
        timeout_per_task: Optional[float] = None
    ) -> Dict[str, TaskResult[T]]:
        """Execute multiple code generation tasks in parallel.
        
        Args:
            code_tasks: Dictionary mapping task names to async task functions
            timeout_per_task: Timeout for each individual task
            
        Returns:
            Dictionary mapping task names to task results
        """
        if not code_tasks:
            return {}
        
        logger.info(f"Starting parallel execution of {len(code_tasks)} code generation tasks")
        
        async def execute_code_task(task_name: str, task_func: Callable[[], Awaitable[T]]) -> TaskResult[T]:
            task_id = self._generate_task_id(f"code_{task_name}")
            start_time = datetime.now()
            
            try:
                # Acquire resource
                await self.resource_manager.acquire_resource(
                    ResourceType.CODE_GENERATION, 
                    task_id, 
                    timeout=timeout_per_task
                )
                
                try:
                    # Execute the task
                    if timeout_per_task:
                        result = await asyncio.wait_for(task_func(), timeout=timeout_per_task)
                    else:
                        result = await task_func()
                    
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    
                    return TaskResult(
                        task_id=task_id,
                        success=True,
                        result=result,
                        duration_seconds=duration,
                        start_time=start_time,
                        end_time=end_time
                    )
                    
                finally:
                    # Always release resource
                    await self.resource_manager.release_resource(ResourceType.CODE_GENERATION, task_id)
            
            except Exception as e:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                return TaskResult(
                    task_id=task_id,
                    success=False,
                    error=e,
                    duration_seconds=duration,
                    start_time=start_time,
                    end_time=end_time
                )
        
        # Execute all code generation tasks in parallel
        tasks = [
            execute_code_task(task_name, task_func)
            for task_name, task_func in code_tasks.items()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        code_results = {}
        for i, (task_name, result) in enumerate(zip(code_tasks.keys(), results)):
            if isinstance(result, Exception):
                code_results[task_name] = TaskResult(
                    task_id=f"code_{task_name}_error",
                    success=False,
                    error=result
                )
            else:
                code_results[task_name] = result
        
        successful_count = sum(1 for r in code_results.values() if r.success)
        logger.info(f"Parallel code generation completed: {successful_count}/{len(code_tasks)} successful")
        
        return code_results
    
    async def execute_parallel_rag_queries(
        self,
        rag_queries: List[Callable[[], Awaitable[T]]],
        timeout_per_query: Optional[float] = None
    ) -> List[TaskResult[T]]:
        """Execute multiple RAG queries in parallel.
        
        Args:
            rag_queries: List of async RAG query functions
            timeout_per_query: Timeout for each individual query
            
        Returns:
            List of task results in the same order as input queries
        """
        if not rag_queries:
            return []
        
        logger.info(f"Starting parallel execution of {len(rag_queries)} RAG queries")
        
        async def execute_rag_query(query_index: int, query_func: Callable[[], Awaitable[T]]) -> TaskResult[T]:
            task_id = self._generate_task_id(f"rag_query_{query_index}")
            start_time = datetime.now()
            
            try:
                # Acquire resource
                await self.resource_manager.acquire_resource(
                    ResourceType.RAG_QUERIES, 
                    task_id, 
                    timeout=timeout_per_query
                )
                
                try:
                    # Execute the query
                    if timeout_per_query:
                        result = await asyncio.wait_for(query_func(), timeout=timeout_per_query)
                    else:
                        result = await query_func()
                    
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    
                    return TaskResult(
                        task_id=task_id,
                        success=True,
                        result=result,
                        duration_seconds=duration,
                        start_time=start_time,
                        end_time=end_time
                    )
                    
                finally:
                    # Always release resource
                    await self.resource_manager.release_resource(ResourceType.RAG_QUERIES, task_id)
            
            except Exception as e:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                return TaskResult(
                    task_id=task_id,
                    success=False,
                    error=e,
                    duration_seconds=duration,
                    start_time=start_time,
                    end_time=end_time
                )
        
        # Execute all RAG queries in parallel
        tasks = [
            execute_rag_query(i, query_func)
            for i, query_func in enumerate(rag_queries)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        rag_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                rag_results.append(TaskResult(
                    task_id=f"rag_query_{i}_error",
                    success=False,
                    error=result
                ))
            else:
                rag_results.append(result)
        
        successful_count = sum(1 for r in rag_results if r.success)
        logger.info(f"Parallel RAG queries completed: {successful_count}/{len(rag_queries)} successful")
        
        return rag_results
    
    async def execute_batch_with_priority(
        self,
        high_priority_tasks: List[Callable[[], Awaitable[T]]],
        normal_priority_tasks: List[Callable[[], Awaitable[T]]],
        resource_type: ResourceType,
        timeout_per_task: Optional[float] = None
    ) -> tuple[List[TaskResult[T]], List[TaskResult[T]]]:
        """Execute tasks in batches with priority ordering.
        
        Args:
            high_priority_tasks: High priority tasks to execute first
            normal_priority_tasks: Normal priority tasks to execute after
            resource_type: Type of resource to use for limiting
            timeout_per_task: Timeout for each individual task
            
        Returns:
            Tuple of (high_priority_results, normal_priority_results)
        """
        logger.info(f"Executing batch with {len(high_priority_tasks)} high priority and {len(normal_priority_tasks)} normal priority tasks")
        
        async def execute_task_with_resource(task_func: Callable[[], Awaitable[T]], task_id: str) -> TaskResult[T]:
            start_time = datetime.now()
            
            try:
                # Acquire resource
                await self.resource_manager.acquire_resource(resource_type, task_id, timeout=timeout_per_task)
                
                try:
                    # Execute the task
                    if timeout_per_task:
                        result = await asyncio.wait_for(task_func(), timeout=timeout_per_task)
                    else:
                        result = await task_func()
                    
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    
                    return TaskResult(
                        task_id=task_id,
                        success=True,
                        result=result,
                        duration_seconds=duration,
                        start_time=start_time,
                        end_time=end_time
                    )
                    
                finally:
                    # Always release resource
                    await self.resource_manager.release_resource(resource_type, task_id)
            
            except Exception as e:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                return TaskResult(
                    task_id=task_id,
                    success=False,
                    error=e,
                    duration_seconds=duration,
                    start_time=start_time,
                    end_time=end_time
                )
        
        # Execute high priority tasks first
        high_priority_results = []
        if high_priority_tasks:
            high_tasks = [
                execute_task_with_resource(task_func, self._generate_task_id("high_priority"))
                for task_func in high_priority_tasks
            ]
            high_results = await asyncio.gather(*high_tasks, return_exceptions=True)
            
            for result in high_results:
                if isinstance(result, Exception):
                    high_priority_results.append(TaskResult(
                        task_id="high_priority_error",
                        success=False,
                        error=result
                    ))
                else:
                    high_priority_results.append(result)
        
        # Execute normal priority tasks
        normal_priority_results = []
        if normal_priority_tasks:
            normal_tasks = [
                execute_task_with_resource(task_func, self._generate_task_id("normal_priority"))
                for task_func in normal_priority_tasks
            ]
            normal_results = await asyncio.gather(*normal_tasks, return_exceptions=True)
            
            for result in normal_results:
                if isinstance(result, Exception):
                    normal_priority_results.append(TaskResult(
                        task_id="normal_priority_error",
                        success=False,
                        error=result
                    ))
                else:
                    normal_priority_results.append(result)
        
        logger.info(f"Batch execution completed: {len([r for r in high_priority_results if r.success])}/{len(high_priority_tasks)} high priority successful, "
                   f"{len([r for r in normal_priority_results if r.success])}/{len(normal_priority_tasks)} normal priority successful")
        
        return high_priority_results, normal_priority_results


# Convenience functions for common parallel operations

async def parallel_scene_processing(
    scenes: Dict[int, str],
    processing_func: Callable[[int, str], Awaitable[T]],
    resource_manager: Optional[ResourceManager] = None,
    max_concurrent: int = 5,
    timeout_per_scene: Optional[float] = None
) -> Dict[int, TaskResult[T]]:
    """Process multiple scenes in parallel.
    
    Args:
        scenes: Dictionary mapping scene numbers to scene data
        processing_func: Function to process each scene
        resource_manager: Optional resource manager for limiting concurrency
        max_concurrent: Maximum concurrent scenes if no resource manager
        timeout_per_scene: Timeout for each scene processing
        
    Returns:
        Dictionary mapping scene numbers to processing results
    """
    if not resource_manager:
        # Create a temporary resource manager
        limits = ResourceLimits(max_concurrent_scenes=max_concurrent)
        resource_manager = ResourceManager(limits)
        await resource_manager.start()
        should_stop = True
    else:
        should_stop = False
    
    try:
        executor = ParallelExecutor(resource_manager)
        
        # Create task functions
        scene_tasks = {
            scene_num: lambda sn=scene_num, sd=scene_data: processing_func(sn, sd)
            for scene_num, scene_data in scenes.items()
        }
        
        return await executor.execute_parallel_scenes(scene_tasks, timeout_per_scene)
    
    finally:
        if should_stop:
            await resource_manager.stop()


async def parallel_code_generation(
    code_requests: Dict[str, Dict[str, Any]],
    generation_func: Callable[[str, Dict[str, Any]], Awaitable[T]],
    resource_manager: Optional[ResourceManager] = None,
    max_concurrent: int = 3,
    timeout_per_request: Optional[float] = None
) -> Dict[str, TaskResult[T]]:
    """Generate code for multiple requests in parallel.
    
    Args:
        code_requests: Dictionary mapping request names to request data
        generation_func: Function to generate code for each request
        resource_manager: Optional resource manager for limiting concurrency
        max_concurrent: Maximum concurrent requests if no resource manager
        timeout_per_request: Timeout for each code generation
        
    Returns:
        Dictionary mapping request names to generation results
    """
    if not resource_manager:
        # Create a temporary resource manager
        limits = ResourceLimits(max_concurrent_code_generation=max_concurrent)
        resource_manager = ResourceManager(limits)
        await resource_manager.start()
        should_stop = True
    else:
        should_stop = False
    
    try:
        executor = ParallelExecutor(resource_manager)
        
        # Create task functions
        code_tasks = {
            request_name: lambda rn=request_name, rd=request_data: generation_func(rn, rd)
            for request_name, request_data in code_requests.items()
        }
        
        return await executor.execute_parallel_code_generation(code_tasks, timeout_per_request)
    
    finally:
        if should_stop:
            await resource_manager.stop()


async def parallel_rag_enhancement(
    queries: List[str],
    rag_func: Callable[[str], Awaitable[T]],
    resource_manager: Optional[ResourceManager] = None,
    max_concurrent: int = 10,
    timeout_per_query: Optional[float] = None
) -> List[TaskResult[T]]:
    """Execute multiple RAG queries in parallel for content enhancement.
    
    Args:
        queries: List of RAG queries to execute
        rag_func: Function to execute each RAG query
        resource_manager: Optional resource manager for limiting concurrency
        max_concurrent: Maximum concurrent queries if no resource manager
        timeout_per_query: Timeout for each RAG query
        
    Returns:
        List of RAG query results in the same order as input
    """
    if not resource_manager:
        # Create a temporary resource manager
        limits = ResourceLimits(max_concurrent_rag_queries=max_concurrent)
        resource_manager = ResourceManager(limits)
        await resource_manager.start()
        should_stop = True
    else:
        should_stop = False
    
    try:
        executor = ParallelExecutor(resource_manager)
        
        # Create query functions
        query_functions = [
            lambda q=query: rag_func(q)
            for query in queries
        ]
        
        return await executor.execute_parallel_rag_queries(query_functions, timeout_per_query)
    
    finally:
        if should_stop:
            await resource_manager.stop()
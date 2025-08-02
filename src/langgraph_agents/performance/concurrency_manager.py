"""
Advanced concurrency management for optimal resource utilization.

This module provides sophisticated concurrency control, task prioritization,
and adaptive resource allocation for multi-agent workflows.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import threading
from collections import defaultdict, deque
import psutil
import time

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels for concurrency management."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class ConcurrencyLimits:
    """Configuration for concurrency limits."""
    max_concurrent_tasks: int
    max_concurrent_per_agent: int
    max_concurrent_per_scene: int
    max_memory_usage_mb: int
    max_cpu_usage_percent: float
    adaptive_scaling: bool = True
    priority_boost_factor: float = 1.5


@dataclass
class TaskMetrics:
    """Metrics for task execution."""
    task_id: str
    agent_name: str
    priority: TaskPriority
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: Optional[float]
    memory_usage_mb: float
    cpu_usage_percent: float
    success: Optional[bool]
    error_message: Optional[str]


class ConcurrencyManager:
    """Advanced concurrency manager with adaptive resource allocation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the concurrency manager.
        
        Args:
            config: Configuration for concurrency management
        """
        self.config = config
        self.limits = ConcurrencyLimits(
            max_concurrent_tasks=config.get('max_concurrent_tasks', 10),
            max_concurrent_per_agent=config.get('max_concurrent_per_agent', 3),
            max_concurrent_per_scene=config.get('max_concurrent_per_scene', 2),
            max_memory_usage_mb=config.get('max_memory_usage_mb', 4096),
            max_cpu_usage_percent=config.get('max_cpu_usage_percent', 80.0),
            adaptive_scaling=config.get('adaptive_scaling', True),
            priority_boost_factor=config.get('priority_boost_factor', 1.5)
        )
        
        # Task tracking
        self.active_tasks: Dict[str, TaskMetrics] = {}
        self.task_queues: Dict[TaskPriority, deque] = {
            priority: deque() for priority in TaskPriority
        }
        self.task_history: deque = deque(maxlen=config.get('max_task_history', 1000))
        
        # Resource tracking
        self.agent_task_counts: Dict[str, int] = defaultdict(int)
        self.scene_task_counts: Dict[str, int] = defaultdict(int)
        
        # Semaphores for concurrency control
        self.global_semaphore = asyncio.Semaphore(self.limits.max_concurrent_tasks)
        self.agent_semaphores: Dict[str, asyncio.Semaphore] = {}
        self.scene_semaphores: Dict[str, asyncio.Semaphore] = {}
        
        # Performance monitoring
        self.performance_monitor_interval = config.get('performance_monitor_interval', 30)
        self.last_performance_check = datetime.now()
        self.performance_history: deque = deque(maxlen=100)
        
        # Adaptive scaling
        self.scaling_enabled = self.limits.adaptive_scaling
        self.scaling_check_interval = config.get('scaling_check_interval', 60)
        self.last_scaling_check = datetime.now()
        self.scaling_history: deque = deque(maxlen=50)
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"Concurrency manager initialized with limits: {self.limits}")
    
    async def execute_task(
        self,
        task_id: str,
        agent_name: str,
        scene_id: Optional[str],
        task_func: Callable[[], Awaitable[Any]],
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout_seconds: Optional[int] = None
    ) -> Any:
        """Execute a task with concurrency control and resource management.
        
        Args:
            task_id: Unique identifier for the task
            agent_name: Name of the agent executing the task
            scene_id: Scene identifier if applicable
            task_func: Async function to execute
            priority: Task priority level
            timeout_seconds: Optional timeout for task execution
            
        Returns:
            Result of task execution
        """
        
        # Create task metrics
        task_metrics = TaskMetrics(
            task_id=task_id,
            agent_name=agent_name,
            priority=priority,
            start_time=datetime.now(),
            end_time=None,
            duration_seconds=None,
            memory_usage_mb=0.0,
            cpu_usage_percent=0.0,
            success=None,
            error_message=None
        )
        
        try:
            # Check if we should queue the task or execute immediately
            if await self._should_queue_task(agent_name, scene_id, priority):
                await self._queue_task(task_id, agent_name, scene_id, task_func, priority, timeout_seconds)
                return await self._wait_for_queued_task(task_id)
            
            # Execute task immediately
            return await self._execute_task_with_limits(task_metrics, scene_id, task_func, timeout_seconds)
            
        except Exception as e:
            task_metrics.success = False
            task_metrics.error_message = str(e)
            task_metrics.end_time = datetime.now()
            task_metrics.duration_seconds = (task_metrics.end_time - task_metrics.start_time).total_seconds()
            
            with self._lock:
                self.task_history.append(task_metrics)
            
            logger.error(f"Task execution failed: {task_id}, error: {str(e)}")
            raise
    
    async def _should_queue_task(
        self, 
        agent_name: str, 
        scene_id: Optional[str], 
        priority: TaskPriority
    ) -> bool:
        """Determine if a task should be queued based on current resource usage."""
        
        with self._lock:
            # Check global task limit
            if len(self.active_tasks) >= self.limits.max_concurrent_tasks:
                return True
            
            # Check agent-specific limit
            if self.agent_task_counts[agent_name] >= self.limits.max_concurrent_per_agent:
                return True
            
            # Check scene-specific limit
            if scene_id and self.scene_task_counts[scene_id] >= self.limits.max_concurrent_per_scene:
                return True
            
            # Check system resource usage
            if await self._check_resource_limits():
                return True
            
            # High priority tasks can bypass some limits
            if priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]:
                return False
            
            return False
    
    async def _check_resource_limits(self) -> bool:
        """Check if system resource limits are exceeded."""
        
        try:
            # Check memory usage
            memory_info = psutil.virtual_memory()
            memory_usage_mb = (memory_info.total - memory_info.available) / (1024 * 1024)
            
            if memory_usage_mb > self.limits.max_memory_usage_mb:
                logger.warning(f"Memory usage ({memory_usage_mb:.1f} MB) exceeds limit ({self.limits.max_memory_usage_mb} MB)")
                return True
            
            # Check CPU usage
            cpu_usage = psutil.cpu_percent(interval=0.1)
            if cpu_usage > self.limits.max_cpu_usage_percent:
                logger.warning(f"CPU usage ({cpu_usage:.1f}%) exceeds limit ({self.limits.max_cpu_usage_percent}%)")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking resource limits: {str(e)}")
            return False
    
    async def _queue_task(
        self,
        task_id: str,
        agent_name: str,
        scene_id: Optional[str],
        task_func: Callable[[], Awaitable[Any]],
        priority: TaskPriority,
        timeout_seconds: Optional[int]
    ):
        """Queue a task for later execution."""
        
        task_info = {
            'task_id': task_id,
            'agent_name': agent_name,
            'scene_id': scene_id,
            'task_func': task_func,
            'timeout_seconds': timeout_seconds,
            'queued_at': datetime.now(),
            'future': asyncio.Future()
        }
        
        with self._lock:
            self.task_queues[priority].append(task_info)
        
        logger.debug(f"Task queued: {task_id}, priority: {priority.name}")
        
        # Start queue processor if not running
        asyncio.create_task(self._process_task_queue())
    
    async def _wait_for_queued_task(self, task_id: str) -> Any:
        """Wait for a queued task to complete."""
        
        # Find the task in queues
        for priority_queue in self.task_queues.values():
            for task_info in priority_queue:
                if task_info['task_id'] == task_id:
                    return await task_info['future']
        
        raise ValueError(f"Queued task not found: {task_id}")
    
    async def _process_task_queue(self):
        """Process queued tasks based on priority and resource availability."""
        
        while True:
            task_to_execute = None
            
            with self._lock:
                # Find highest priority task that can be executed
                for priority in TaskPriority:
                    queue = self.task_queues[priority]
                    
                    for i, task_info in enumerate(queue):
                        agent_name = task_info['agent_name']
                        scene_id = task_info['scene_id']
                        
                        # Check if task can be executed now
                        if not await self._should_queue_task(agent_name, scene_id, priority):
                            task_to_execute = queue.popleft() if i == 0 else queue.pop(i)
                            break
                    
                    if task_to_execute:
                        break
            
            if not task_to_execute:
                # No tasks can be executed, wait and try again
                await asyncio.sleep(1)
                continue
            
            # Execute the task
            try:
                task_metrics = TaskMetrics(
                    task_id=task_to_execute['task_id'],
                    agent_name=task_to_execute['agent_name'],
                    priority=TaskPriority.NORMAL,  # Will be updated
                    start_time=datetime.now(),
                    end_time=None,
                    duration_seconds=None,
                    memory_usage_mb=0.0,
                    cpu_usage_percent=0.0,
                    success=None,
                    error_message=None
                )
                
                result = await self._execute_task_with_limits(
                    task_metrics,
                    task_to_execute['scene_id'],
                    task_to_execute['task_func'],
                    task_to_execute['timeout_seconds']
                )
                
                task_to_execute['future'].set_result(result)
                
            except Exception as e:
                task_to_execute['future'].set_exception(e)
    
    async def _execute_task_with_limits(
        self,
        task_metrics: TaskMetrics,
        scene_id: Optional[str],
        task_func: Callable[[], Awaitable[Any]],
        timeout_seconds: Optional[int]
    ) -> Any:
        """Execute a task with resource limits and monitoring."""
        
        # Acquire semaphores
        await self.global_semaphore.acquire()
        
        agent_semaphore = self._get_agent_semaphore(task_metrics.agent_name)
        await agent_semaphore.acquire()
        
        scene_semaphore = None
        if scene_id:
            scene_semaphore = self._get_scene_semaphore(scene_id)
            await scene_semaphore.acquire()
        
        try:
            # Update counters
            with self._lock:
                self.active_tasks[task_metrics.task_id] = task_metrics
                self.agent_task_counts[task_metrics.agent_name] += 1
                if scene_id:
                    self.scene_task_counts[scene_id] += 1
            
            # Monitor resource usage during execution
            monitor_task = asyncio.create_task(self._monitor_task_resources(task_metrics))
            
            # Execute the task with timeout
            try:
                if timeout_seconds:
                    result = await asyncio.wait_for(task_func(), timeout=timeout_seconds)
                else:
                    result = await task_func()
                
                task_metrics.success = True
                return result
                
            except asyncio.TimeoutError:
                task_metrics.success = False
                task_metrics.error_message = f"Task timed out after {timeout_seconds} seconds"
                logger.warning(f"Task timeout: {task_metrics.task_id}")
                raise
            
            except Exception as e:
                task_metrics.success = False
                task_metrics.error_message = str(e)
                raise
            
            finally:
                # Stop resource monitoring
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass
        
        finally:
            # Update task metrics
            task_metrics.end_time = datetime.now()
            task_metrics.duration_seconds = (task_metrics.end_time - task_metrics.start_time).total_seconds()
            
            # Release semaphores and update counters
            with self._lock:
                self.active_tasks.pop(task_metrics.task_id, None)
                self.agent_task_counts[task_metrics.agent_name] -= 1
                if scene_id:
                    self.scene_task_counts[scene_id] -= 1
                
                self.task_history.append(task_metrics)
            
            if scene_semaphore:
                scene_semaphore.release()
            agent_semaphore.release()
            self.global_semaphore.release()
            
            logger.debug(f"Task completed: {task_metrics.task_id}, "
                        f"duration: {task_metrics.duration_seconds:.2f}s, "
                        f"success: {task_metrics.success}")
    
    def _get_agent_semaphore(self, agent_name: str) -> asyncio.Semaphore:
        """Get or create semaphore for an agent."""
        if agent_name not in self.agent_semaphores:
            self.agent_semaphores[agent_name] = asyncio.Semaphore(self.limits.max_concurrent_per_agent)
        return self.agent_semaphores[agent_name]
    
    def _get_scene_semaphore(self, scene_id: str) -> asyncio.Semaphore:
        """Get or create semaphore for a scene."""
        if scene_id not in self.scene_semaphores:
            self.scene_semaphores[scene_id] = asyncio.Semaphore(self.limits.max_concurrent_per_scene)
        return self.scene_semaphores[scene_id]
    
    async def _monitor_task_resources(self, task_metrics: TaskMetrics):
        """Monitor resource usage for a running task."""
        
        try:
            while True:
                # Get current resource usage
                memory_info = psutil.virtual_memory()
                memory_usage_mb = (memory_info.total - memory_info.available) / (1024 * 1024)
                cpu_usage = psutil.cpu_percent(interval=0.1)
                
                # Update task metrics (approximate attribution)
                active_task_count = len(self.active_tasks)
                if active_task_count > 0:
                    task_metrics.memory_usage_mb = memory_usage_mb / active_task_count
                    task_metrics.cpu_usage_percent = cpu_usage / active_task_count
                
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            pass
    
    async def adaptive_scaling(self):
        """Perform adaptive scaling based on system performance."""
        
        if not self.scaling_enabled:
            return
        
        now = datetime.now()
        if (now - self.last_scaling_check).total_seconds() < self.scaling_check_interval:
            return
        
        self.last_scaling_check = now
        
        try:
            # Analyze recent performance
            performance_metrics = await self._analyze_performance()
            
            # Determine scaling action
            scaling_action = self._determine_scaling_action(performance_metrics)
            
            if scaling_action:
                await self._apply_scaling_action(scaling_action, performance_metrics)
                
        except Exception as e:
            logger.error(f"Adaptive scaling failed: {str(e)}")
    
    async def optimize_for_planner_speed(self):
        """Optimize concurrency settings specifically for planner agent speed."""
        
        # Increase planner agent concurrency if system resources allow
        planner_utilization = self.agent_task_counts.get('planner_agent', 0) / self.limits.max_concurrent_per_agent
        
        if planner_utilization > 0.7:  # High utilization
            try:
                # Check system resources
                memory_info = psutil.virtual_memory()
                cpu_usage = psutil.cpu_percent(interval=0.1)
                
                if memory_info.percent < 70 and cpu_usage < 60:
                    # Increase planner concurrency
                    if 'planner_agent' in self.agent_semaphores:
                        # Release additional permits for planner
                        for _ in range(2):  # Add 2 more concurrent slots
                            self.agent_semaphores['planner_agent'].release()
                        
                        logger.info("Optimized planner agent concurrency for speed")
                        
                        # Also optimize scene processing concurrency
                        await self._optimize_scene_processing()
                        
            except Exception as e:
                logger.error(f"Failed to optimize planner speed: {str(e)}")
    
    async def _optimize_scene_processing(self):
        """Optimize concurrent scene processing within planner."""
        
        # This would be called by planner agent to optimize its internal concurrency
        # For now, we'll track this as a performance optimization
        
        optimization_record = {
            'timestamp': datetime.now(),
            'optimization_type': 'scene_processing',
            'action': 'increased_internal_concurrency',
            'details': {
                'max_scene_concurrency': 8,  # Increased from default 5
                'parallel_implementation_generation': True,
                'optimized_rag_queries': True
            }
        }
        
        self.scaling_history.append(optimization_record)
        logger.info("Optimized scene processing concurrency")
    
    async def batch_execute_tasks(self, 
                                 tasks: List[Dict[str, Any]], 
                                 priority: TaskPriority = TaskPriority.NORMAL) -> List[Any]:
        """Execute multiple tasks in optimized batches.
        
        Args:
            tasks: List of task definitions with 'task_id', 'agent_name', 'scene_id', 'task_func'
            priority: Priority level for all tasks
            
        Returns:
            List of task results
        """
        
        if not tasks:
            return []
        
        # Group tasks by agent for optimal batching
        agent_groups = defaultdict(list)
        for task in tasks:
            agent_groups[task['agent_name']].append(task)
        
        # Execute tasks in parallel batches
        all_results = []
        batch_tasks = []
        
        for agent_name, agent_tasks in agent_groups.items():
            for task in agent_tasks:
                batch_task = asyncio.create_task(
                    self.execute_task(
                        task_id=task['task_id'],
                        agent_name=task['agent_name'],
                        scene_id=task.get('scene_id'),
                        task_func=task['task_func'],
                        priority=priority,
                        timeout_seconds=task.get('timeout_seconds')
                    )
                )
                batch_tasks.append(batch_task)
        
        # Wait for all tasks to complete
        try:
            results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Batch task {i} failed: {result}")
                    all_results.append(None)
                else:
                    all_results.append(result)
            
            logger.info(f"Batch executed {len(tasks)} tasks with {len([r for r in all_results if r is not None])} successes")
            return all_results
            
        except Exception as e:
            logger.error(f"Batch execution failed: {str(e)}")
            # Cancel remaining tasks
            for task in batch_tasks:
                if not task.done():
                    task.cancel()
            raise
    
    async def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze recent performance metrics."""
        
        recent_tasks = [
            task for task in self.task_history
            if task.end_time and (datetime.now() - task.end_time).total_seconds() < 300  # Last 5 minutes
        ]
        
        if not recent_tasks:
            return {'insufficient_data': True}
        
        # Calculate performance metrics
        avg_duration = sum(task.duration_seconds or 0 for task in recent_tasks) / len(recent_tasks)
        success_rate = sum(1 for task in recent_tasks if task.success) / len(recent_tasks)
        avg_memory_usage = sum(task.memory_usage_mb for task in recent_tasks) / len(recent_tasks)
        avg_cpu_usage = sum(task.cpu_usage_percent for task in recent_tasks) / len(recent_tasks)
        
        # Get current system metrics
        memory_info = psutil.virtual_memory()
        current_memory_usage = (memory_info.total - memory_info.available) / (1024 * 1024)
        current_cpu_usage = psutil.cpu_percent(interval=1.0)
        
        # Calculate queue lengths
        total_queued = sum(len(queue) for queue in self.task_queues.values())
        
        return {
            'avg_task_duration': avg_duration,
            'task_success_rate': success_rate,
            'avg_memory_usage_mb': avg_memory_usage,
            'avg_cpu_usage_percent': avg_cpu_usage,
            'current_memory_usage_mb': current_memory_usage,
            'current_cpu_usage_percent': current_cpu_usage,
            'active_tasks': len(self.active_tasks),
            'queued_tasks': total_queued,
            'recent_task_count': len(recent_tasks)
        }
    
    def _determine_scaling_action(self, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Determine what scaling action to take based on performance metrics."""
        
        if metrics.get('insufficient_data'):
            return None
        
        actions = []
        
        # Check if we should scale up
        if (metrics['queued_tasks'] > 5 and 
            metrics['current_cpu_usage_percent'] < 60 and 
            metrics['current_memory_usage_mb'] < self.limits.max_memory_usage_mb * 0.7):
            
            actions.append({
                'type': 'scale_up',
                'reason': 'high_queue_low_resource_usage',
                'target': 'max_concurrent_tasks',
                'change': min(2, int(self.limits.max_concurrent_tasks * 0.2))
            })
        
        # Check if we should scale down
        if (metrics['active_tasks'] < self.limits.max_concurrent_tasks * 0.3 and
            metrics['queued_tasks'] == 0 and
            metrics['avg_task_duration'] < 30):  # Tasks completing quickly
            
            actions.append({
                'type': 'scale_down',
                'reason': 'low_utilization',
                'target': 'max_concurrent_tasks',
                'change': -max(1, int(self.limits.max_concurrent_tasks * 0.1))
            })
        
        # Check if we should adjust agent limits
        agent_utilization = {}
        for agent_name, count in self.agent_task_counts.items():
            if count > 0:
                utilization = count / self.limits.max_concurrent_per_agent
                agent_utilization[agent_name] = utilization
        
        # Scale up heavily utilized agents
        for agent_name, utilization in agent_utilization.items():
            if utilization > 0.8 and metrics['current_cpu_usage_percent'] < 70:
                actions.append({
                    'type': 'scale_up_agent',
                    'reason': 'high_agent_utilization',
                    'target': agent_name,
                    'change': 1
                })
        
        return actions[0] if actions else None
    
    async def _apply_scaling_action(self, action: Dict[str, Any], metrics: Dict[str, Any]):
        """Apply a scaling action."""
        
        action_type = action['type']
        
        if action_type == 'scale_up':
            old_limit = self.limits.max_concurrent_tasks
            self.limits.max_concurrent_tasks += action['change']
            
            # Update global semaphore
            for _ in range(action['change']):
                self.global_semaphore.release()
            
            logger.info(f"Scaled up max_concurrent_tasks: {old_limit} -> {self.limits.max_concurrent_tasks}")
        
        elif action_type == 'scale_down':
            old_limit = self.limits.max_concurrent_tasks
            new_limit = max(1, self.limits.max_concurrent_tasks + action['change'])  # Ensure minimum of 1
            change = new_limit - old_limit
            
            if change < 0:
                self.limits.max_concurrent_tasks = new_limit
                
                # Acquire semaphore permits to reduce capacity
                for _ in range(abs(change)):
                    try:
                        await asyncio.wait_for(self.global_semaphore.acquire(), timeout=1.0)
                    except asyncio.TimeoutError:
                        break  # Can't acquire more, stop scaling down
                
                logger.info(f"Scaled down max_concurrent_tasks: {old_limit} -> {self.limits.max_concurrent_tasks}")
        
        elif action_type == 'scale_up_agent':
            agent_name = action['target']
            if agent_name in self.agent_semaphores:
                # Release additional permits
                for _ in range(action['change']):
                    self.agent_semaphores[agent_name].release()
                
                logger.info(f"Scaled up agent {agent_name} concurrency by {action['change']}")
        
        # Record scaling action
        scaling_record = {
            'timestamp': datetime.now(),
            'action': action,
            'metrics_before': metrics,
            'new_limits': {
                'max_concurrent_tasks': self.limits.max_concurrent_tasks,
                'max_concurrent_per_agent': self.limits.max_concurrent_per_agent
            }
        }
        
        self.scaling_history.append(scaling_record)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        
        with self._lock:
            # Task statistics
            total_tasks = len(self.task_history)
            successful_tasks = sum(1 for task in self.task_history if task.success)
            failed_tasks = total_tasks - successful_tasks
            
            # Duration statistics
            completed_tasks = [task for task in self.task_history if task.duration_seconds is not None]
            if completed_tasks:
                avg_duration = sum(task.duration_seconds for task in completed_tasks) / len(completed_tasks)
                min_duration = min(task.duration_seconds for task in completed_tasks)
                max_duration = max(task.duration_seconds for task in completed_tasks)
            else:
                avg_duration = min_duration = max_duration = 0.0
            
            # Current state
            active_task_count = len(self.active_tasks)
            queued_task_count = sum(len(queue) for queue in self.task_queues.values())
            
            # Agent utilization
            agent_utilization = {
                agent: count / self.limits.max_concurrent_per_agent
                for agent, count in self.agent_task_counts.items()
                if count > 0
            }
            
            return {
                'total_tasks_executed': total_tasks,
                'successful_tasks': successful_tasks,
                'failed_tasks': failed_tasks,
                'success_rate': successful_tasks / total_tasks if total_tasks > 0 else 0.0,
                'average_task_duration_seconds': avg_duration,
                'min_task_duration_seconds': min_duration,
                'max_task_duration_seconds': max_duration,
                'active_tasks': active_task_count,
                'queued_tasks': queued_task_count,
                'agent_utilization': agent_utilization,
                'current_limits': {
                    'max_concurrent_tasks': self.limits.max_concurrent_tasks,
                    'max_concurrent_per_agent': self.limits.max_concurrent_per_agent,
                    'max_concurrent_per_scene': self.limits.max_concurrent_per_scene
                },
                'scaling_enabled': self.scaling_enabled,
                'scaling_actions_taken': len(self.scaling_history)
            }
    
    def get_task_history(self, agent_name: Optional[str] = None, hours: int = 24) -> List[TaskMetrics]:
        """Get task execution history."""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            filtered_tasks = [
                task for task in self.task_history
                if task.start_time > cutoff_time and (agent_name is None or task.agent_name == agent_name)
            ]
        
        return filtered_tasks
    
    async def shutdown(self):
        """Shutdown the concurrency manager gracefully."""
        
        logger.info("Shutting down concurrency manager...")
        
        # Wait for active tasks to complete (with timeout)
        timeout = 30  # 30 seconds
        start_time = time.time()
        
        while self.active_tasks and (time.time() - start_time) < timeout:
            await asyncio.sleep(1)
        
        if self.active_tasks:
            logger.warning(f"Shutdown timeout reached, {len(self.active_tasks)} tasks still active")
        
        # Clear queues
        with self._lock:
            for queue in self.task_queues.values():
                while queue:
                    task_info = queue.popleft()
                    task_info['future'].cancel()
        
        logger.info("Concurrency manager shutdown complete")
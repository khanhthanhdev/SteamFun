"""
Resource pooling and connection management for optimal resource utilization.

This module provides connection pooling, resource lifecycle management,
and efficient resource allocation for external services and dependencies.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, TypeVar, Generic, Callable, Awaitable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import threading
from collections import deque
import time
import weakref

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ResourceType(Enum):
    """Types of resources that can be pooled."""
    HTTP_CONNECTION = "http_connection"
    DATABASE_CONNECTION = "database_connection"
    LLM_CLIENT = "llm_client"
    VECTOR_STORE_CONNECTION = "vector_store_connection"
    FILE_HANDLE = "file_handle"
    COMPUTE_RESOURCE = "compute_resource"


@dataclass
class ResourceMetrics:
    """Metrics for resource usage."""
    resource_id: str
    resource_type: ResourceType
    created_at: datetime
    last_used: datetime
    usage_count: int
    total_active_time: float
    is_healthy: bool
    error_count: int
    last_error: Optional[str]


class ResourcePool(Generic[T]):
    """Generic resource pool with lifecycle management."""
    
    def __init__(
        self,
        resource_type: ResourceType,
        factory: Callable[[], Awaitable[T]],
        destroyer: Optional[Callable[[T], Awaitable[None]]] = None,
        health_checker: Optional[Callable[[T], Awaitable[bool]]] = None,
        min_size: int = 1,
        max_size: int = 10,
        max_idle_time: int = 300,  # 5 minutes
        health_check_interval: int = 60,  # 1 minute
        max_lifetime: int = 3600  # 1 hour
    ):
        """Initialize the resource pool.
        
        Args:
            resource_type: Type of resource being pooled
            factory: Async function to create new resources
            destroyer: Optional async function to destroy resources
            health_checker: Optional async function to check resource health
            min_size: Minimum number of resources to maintain
            max_size: Maximum number of resources in the pool
            max_idle_time: Maximum time a resource can be idle before removal
            health_check_interval: Interval for health checks in seconds
            max_lifetime: Maximum lifetime of a resource in seconds
        """
        self.resource_type = resource_type
        self.factory = factory
        self.destroyer = destroyer
        self.health_checker = health_checker
        self.min_size = min_size
        self.max_size = max_size
        self.max_idle_time = max_idle_time
        self.health_check_interval = health_check_interval
        self.max_lifetime = max_lifetime
        
        # Resource tracking
        self.available_resources: deque = deque()
        self.in_use_resources: Dict[str, T] = {}
        self.resource_metrics: Dict[str, ResourceMetrics] = {}
        
        # Synchronization
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)
        
        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        # Statistics
        self.total_created = 0
        self.total_destroyed = 0
        self.total_acquisitions = 0
        self.total_releases = 0
        self.total_health_checks = 0
        self.total_health_failures = 0
        
        logger.info(f"Resource pool created: {resource_type.value}, "
                   f"min_size: {min_size}, max_size: {max_size}")
    
    async def start(self):
        """Start the resource pool and background tasks."""
        
        async with self._lock:
            if self._health_check_task is not None:
                return  # Already started
            
            # Create initial resources
            await self._ensure_min_resources()
            
            # Start background tasks
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info(f"Resource pool started: {self.resource_type.value}")
    
    async def stop(self):
        """Stop the resource pool and cleanup resources."""
        
        async with self._lock:
            self._shutdown = True
            
            # Cancel background tasks
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Destroy all resources
            while self.available_resources:
                resource_id, resource = self.available_resources.popleft()
                await self._destroy_resource(resource_id, resource)
            
            # Note: in_use_resources will be cleaned up when released
            
            logger.info(f"Resource pool stopped: {self.resource_type.value}")
    
    async def acquire(self, timeout: Optional[float] = None) -> T:
        """Acquire a resource from the pool.
        
        Args:
            timeout: Maximum time to wait for a resource
            
        Returns:
            A resource from the pool
            
        Raises:
            asyncio.TimeoutError: If timeout is reached
            RuntimeError: If pool is shutdown
        """
        
        start_time = time.time()
        
        async with self._condition:
            while True:
                if self._shutdown:
                    raise RuntimeError("Resource pool is shutdown")
                
                # Check timeout
                if timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        raise asyncio.TimeoutError("Resource acquisition timeout")
                
                # Try to get an available resource
                if self.available_resources:
                    resource_id, resource = self.available_resources.popleft()
                    
                    # Check if resource is still healthy
                    if await self._is_resource_healthy(resource_id, resource):
                        self.in_use_resources[resource_id] = resource
                        self.total_acquisitions += 1
                        
                        # Update metrics
                        metrics = self.resource_metrics[resource_id]
                        metrics.last_used = datetime.now()
                        metrics.usage_count += 1
                        
                        logger.debug(f"Resource acquired: {resource_id}")
                        return resource
                    else:
                        # Resource is unhealthy, destroy it
                        await self._destroy_resource(resource_id, resource)
                        continue
                
                # Try to create a new resource if under max size
                total_resources = len(self.available_resources) + len(self.in_use_resources)
                if total_resources < self.max_size:
                    try:
                        resource_id, resource = await self._create_resource()
                        self.in_use_resources[resource_id] = resource
                        self.total_acquisitions += 1
                        
                        # Update metrics
                        metrics = self.resource_metrics[resource_id]
                        metrics.last_used = datetime.now()
                        metrics.usage_count += 1
                        
                        logger.debug(f"New resource created and acquired: {resource_id}")
                        return resource
                    except Exception as e:
                        logger.error(f"Failed to create resource: {str(e)}")
                        # Continue to wait for available resource
                
                # Wait for a resource to become available
                remaining_timeout = None
                if timeout is not None:
                    remaining_timeout = timeout - (time.time() - start_time)
                    if remaining_timeout <= 0:
                        raise asyncio.TimeoutError("Resource acquisition timeout")
                
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=remaining_timeout)
                except asyncio.TimeoutError:
                    raise asyncio.TimeoutError("Resource acquisition timeout")
    
    async def release(self, resource: T):
        """Release a resource back to the pool.
        
        Args:
            resource: The resource to release
        """
        
        async with self._condition:
            # Find the resource ID
            resource_id = None
            for rid, res in self.in_use_resources.items():
                if res is resource:
                    resource_id = rid
                    break
            
            if resource_id is None:
                logger.warning("Attempted to release unknown resource")
                return
            
            # Remove from in-use
            del self.in_use_resources[resource_id]
            self.total_releases += 1
            
            # Check if resource is still healthy and not expired
            if (await self._is_resource_healthy(resource_id, resource) and
                not self._is_resource_expired(resource_id)):
                
                # Return to available pool
                self.available_resources.append((resource_id, resource))
                logger.debug(f"Resource released: {resource_id}")
            else:
                # Destroy unhealthy or expired resource
                await self._destroy_resource(resource_id, resource)
                logger.debug(f"Resource destroyed on release: {resource_id}")
            
            # Notify waiting acquirers
            self._condition.notify()
    
    async def _create_resource(self) -> tuple[str, T]:
        """Create a new resource."""
        
        resource_id = f"{self.resource_type.value}_{self.total_created}_{int(time.time())}"
        
        try:
            resource = await self.factory()
            self.total_created += 1
            
            # Create metrics
            metrics = ResourceMetrics(
                resource_id=resource_id,
                resource_type=self.resource_type,
                created_at=datetime.now(),
                last_used=datetime.now(),
                usage_count=0,
                total_active_time=0.0,
                is_healthy=True,
                error_count=0,
                last_error=None
            )
            
            self.resource_metrics[resource_id] = metrics
            
            logger.debug(f"Resource created: {resource_id}")
            return resource_id, resource
            
        except Exception as e:
            logger.error(f"Failed to create resource: {str(e)}")
            raise
    
    async def _destroy_resource(self, resource_id: str, resource: T):
        """Destroy a resource."""
        
        try:
            if self.destroyer:
                await self.destroyer(resource)
            
            self.total_destroyed += 1
            
            # Remove metrics
            self.resource_metrics.pop(resource_id, None)
            
            logger.debug(f"Resource destroyed: {resource_id}")
            
        except Exception as e:
            logger.error(f"Failed to destroy resource {resource_id}: {str(e)}")
    
    async def _is_resource_healthy(self, resource_id: str, resource: T) -> bool:
        """Check if a resource is healthy."""
        
        if not self.health_checker:
            return True
        
        try:
            self.total_health_checks += 1
            is_healthy = await self.health_checker(resource)
            
            # Update metrics
            metrics = self.resource_metrics.get(resource_id)
            if metrics:
                metrics.is_healthy = is_healthy
                if not is_healthy:
                    metrics.error_count += 1
                    self.total_health_failures += 1
            
            return is_healthy
            
        except Exception as e:
            logger.error(f"Health check failed for resource {resource_id}: {str(e)}")
            
            # Update metrics
            metrics = self.resource_metrics.get(resource_id)
            if metrics:
                metrics.is_healthy = False
                metrics.error_count += 1
                metrics.last_error = str(e)
                self.total_health_failures += 1
            
            return False
    
    def _is_resource_expired(self, resource_id: str) -> bool:
        """Check if a resource has exceeded its maximum lifetime."""
        
        metrics = self.resource_metrics.get(resource_id)
        if not metrics:
            return True
        
        age = (datetime.now() - metrics.created_at).total_seconds()
        return age > self.max_lifetime
    
    def _is_resource_idle(self, resource_id: str) -> bool:
        """Check if a resource has been idle too long."""
        
        metrics = self.resource_metrics.get(resource_id)
        if not metrics:
            return True
        
        idle_time = (datetime.now() - metrics.last_used).total_seconds()
        return idle_time > self.max_idle_time
    
    async def _ensure_min_resources(self):
        """Ensure minimum number of resources are available."""
        
        total_resources = len(self.available_resources) + len(self.in_use_resources)
        
        while total_resources < self.min_size:
            try:
                resource_id, resource = await self._create_resource()
                self.available_resources.append((resource_id, resource))
                total_resources += 1
            except Exception as e:
                logger.error(f"Failed to create minimum resource: {str(e)}")
                break
    
    async def _health_check_loop(self):
        """Background task for periodic health checks."""
        
        try:
            while not self._shutdown:
                await asyncio.sleep(self.health_check_interval)
                
                if self._shutdown:
                    break
                
                async with self._lock:
                    # Check available resources
                    healthy_resources = deque()
                    
                    while self.available_resources:
                        resource_id, resource = self.available_resources.popleft()
                        
                        if await self._is_resource_healthy(resource_id, resource):
                            healthy_resources.append((resource_id, resource))
                        else:
                            await self._destroy_resource(resource_id, resource)
                    
                    self.available_resources = healthy_resources
                    
                    # Ensure minimum resources
                    await self._ensure_min_resources()
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Health check loop error: {str(e)}")
    
    async def _cleanup_loop(self):
        """Background task for cleaning up idle and expired resources."""
        
        try:
            while not self._shutdown:
                await asyncio.sleep(60)  # Check every minute
                
                if self._shutdown:
                    break
                
                async with self._lock:
                    # Clean up idle and expired resources
                    active_resources = deque()
                    
                    while self.available_resources:
                        resource_id, resource = self.available_resources.popleft()
                        
                        if (self._is_resource_idle(resource_id) or 
                            self._is_resource_expired(resource_id)):
                            
                            # Only remove if we have more than minimum
                            total_resources = len(active_resources) + len(self.in_use_resources) + 1
                            if total_resources > self.min_size:
                                await self._destroy_resource(resource_id, resource)
                                continue
                        
                        active_resources.append((resource_id, resource))
                    
                    self.available_resources = active_resources
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Cleanup loop error: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        
        return {
            'resource_type': self.resource_type.value,
            'available_resources': len(self.available_resources),
            'in_use_resources': len(self.in_use_resources),
            'total_resources': len(self.available_resources) + len(self.in_use_resources),
            'min_size': self.min_size,
            'max_size': self.max_size,
            'total_created': self.total_created,
            'total_destroyed': self.total_destroyed,
            'total_acquisitions': self.total_acquisitions,
            'total_releases': self.total_releases,
            'total_health_checks': self.total_health_checks,
            'total_health_failures': self.total_health_failures,
            'health_failure_rate': (self.total_health_failures / max(self.total_health_checks, 1)),
            'is_running': self._health_check_task is not None and not self._health_check_task.done()
        }


class ResourcePoolManager:
    """Manager for multiple resource pools."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the resource pool manager.
        
        Args:
            config: Configuration for resource pools
        """
        self.config = config
        self.pools: Dict[str, ResourcePool] = {}
        self._shutdown = False
        
        logger.info("Resource pool manager initialized")
    
    async def create_pool(
        self,
        pool_name: str,
        resource_type: ResourceType,
        factory: Callable[[], Awaitable[T]],
        destroyer: Optional[Callable[[T], Awaitable[None]]] = None,
        health_checker: Optional[Callable[[T], Awaitable[bool]]] = None,
        **pool_config
    ) -> ResourcePool[T]:
        """Create a new resource pool.
        
        Args:
            pool_name: Unique name for the pool
            resource_type: Type of resource being pooled
            factory: Async function to create new resources
            destroyer: Optional async function to destroy resources
            health_checker: Optional async function to check resource health
            **pool_config: Additional pool configuration
            
        Returns:
            The created resource pool
        """
        
        if pool_name in self.pools:
            raise ValueError(f"Pool already exists: {pool_name}")
        
        # Merge with default config
        default_config = self.config.get('default_pool_config', {})
        pool_config = {**default_config, **pool_config}
        
        pool = ResourcePool(
            resource_type=resource_type,
            factory=factory,
            destroyer=destroyer,
            health_checker=health_checker,
            **pool_config
        )
        
        await pool.start()
        self.pools[pool_name] = pool
        
        logger.info(f"Resource pool created: {pool_name}")
        return pool
    
    def get_pool(self, pool_name: str) -> Optional[ResourcePool]:
        """Get a resource pool by name.
        
        Args:
            pool_name: Name of the pool
            
        Returns:
            The resource pool or None if not found
        """
        return self.pools.get(pool_name)
    
    async def remove_pool(self, pool_name: str):
        """Remove and stop a resource pool.
        
        Args:
            pool_name: Name of the pool to remove
        """
        
        pool = self.pools.pop(pool_name, None)
        if pool:
            await pool.stop()
            logger.info(f"Resource pool removed: {pool_name}")
    
    async def shutdown(self):
        """Shutdown all resource pools."""
        
        if self._shutdown:
            return
        
        self._shutdown = True
        
        logger.info("Shutting down resource pool manager...")
        
        # Stop all pools
        for pool_name, pool in list(self.pools.items()):
            try:
                await pool.stop()
            except Exception as e:
                logger.error(f"Error stopping pool {pool_name}: {str(e)}")
        
        self.pools.clear()
        
        logger.info("Resource pool manager shutdown complete")
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all pools."""
        
        return {
            pool_name: pool.get_stats()
            for pool_name, pool in self.pools.items()
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the pool manager."""
        
        total_pools = len(self.pools)
        total_resources = sum(
            pool.get_stats()['total_resources']
            for pool in self.pools.values()
        )
        total_acquisitions = sum(
            pool.get_stats()['total_acquisitions']
            for pool in self.pools.values()
        )
        
        return {
            'total_pools': total_pools,
            'total_resources': total_resources,
            'total_acquisitions': total_acquisitions,
            'is_shutdown': self._shutdown,
            'pool_names': list(self.pools.keys())
        }
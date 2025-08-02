"""
Service discovery and registry for distributed deployment.

This module provides service registration, discovery, and health monitoring
for distributed multi-agent systems.
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import socket
import uuid
import aiohttp
from pathlib import Path

logger = logging.getLogger(__name__)


class ServiceHealth(Enum):
    """Service health status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    STARTING = "starting"
    STOPPING = "stopping"


@dataclass
class ServiceInfo:
    """Information about a registered service."""
    service_id: str
    service_name: str
    service_type: str
    host: str
    port: int
    metadata: Dict[str, Any]
    health_check_url: Optional[str]
    registered_at: datetime
    last_heartbeat: datetime
    health_status: ServiceHealth
    tags: List[str]


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    service_id: str
    status: ServiceHealth
    response_time_ms: float
    error_message: Optional[str]
    timestamp: datetime
    details: Dict[str, Any]


class ServiceRegistry:
    """Registry for managing service information."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the service registry.
        
        Args:
            config: Configuration for the service registry
        """
        self.config = config
        self.services: Dict[str, ServiceInfo] = {}
        self.health_history: Dict[str, List[HealthCheckResult]] = {}
        
        # Configuration
        self.heartbeat_interval = config.get('heartbeat_interval_seconds', 30)
        self.service_timeout = config.get('service_timeout_seconds', 90)
        self.health_check_interval = config.get('health_check_interval_seconds', 60)
        self.max_health_history = config.get('max_health_history', 100)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Event callbacks
        self.service_registered_callbacks: List[Callable[[ServiceInfo], None]] = []
        self.service_deregistered_callbacks: List[Callable[[ServiceInfo], None]] = []
        self.service_health_changed_callbacks: List[Callable[[ServiceInfo, ServiceHealth], None]] = []
        
        logger.info("Service registry initialized")
    
    def register_service(
        self,
        service_name: str,
        service_type: str,
        host: str,
        port: int,
        metadata: Optional[Dict[str, Any]] = None,
        health_check_url: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Register a service.
        
        Args:
            service_name: Name of the service
            service_type: Type of service (e.g., 'agent', 'api', 'worker')
            host: Service host
            port: Service port
            metadata: Additional service metadata
            health_check_url: URL for health checks
            tags: Service tags for filtering
            
        Returns:
            Service ID
        """
        
        service_id = f"{service_name}_{uuid.uuid4().hex[:8]}"
        
        service_info = ServiceInfo(
            service_id=service_id,
            service_name=service_name,
            service_type=service_type,
            host=host,
            port=port,
            metadata=metadata or {},
            health_check_url=health_check_url,
            registered_at=datetime.now(),
            last_heartbeat=datetime.now(),
            health_status=ServiceHealth.STARTING,
            tags=tags or []
        )
        
        with self._lock:
            self.services[service_id] = service_info
            self.health_history[service_id] = []
        
        logger.info(f"Service registered: {service_name} ({service_id}) at {host}:{port}")
        
        # Notify callbacks
        for callback in self.service_registered_callbacks:
            try:
                callback(service_info)
            except Exception as e:
                logger.error(f"Error in service registered callback: {str(e)}")
        
        return service_id
    
    def deregister_service(self, service_id: str) -> bool:
        """Deregister a service.
        
        Args:
            service_id: ID of the service to deregister
            
        Returns:
            True if service was found and deregistered
        """
        
        with self._lock:
            service_info = self.services.pop(service_id, None)
            if service_info:
                self.health_history.pop(service_id, None)
                
                logger.info(f"Service deregistered: {service_info.service_name} ({service_id})")
                
                # Notify callbacks
                for callback in self.service_deregistered_callbacks:
                    try:
                        callback(service_info)
                    except Exception as e:
                        logger.error(f"Error in service deregistered callback: {str(e)}")
                
                return True
        
        return False
    
    def update_heartbeat(self, service_id: str) -> bool:
        """Update service heartbeat.
        
        Args:
            service_id: ID of the service
            
        Returns:
            True if service was found and updated
        """
        
        with self._lock:
            service_info = self.services.get(service_id)
            if service_info:
                service_info.last_heartbeat = datetime.now()
                
                # Update health status if it was starting
                if service_info.health_status == ServiceHealth.STARTING:
                    self.update_service_health(service_id, ServiceHealth.HEALTHY)
                
                return True
        
        return False
    
    def update_service_health(self, service_id: str, health_status: ServiceHealth, details: Optional[Dict[str, Any]] = None):
        """Update service health status.
        
        Args:
            service_id: ID of the service
            health_status: New health status
            details: Additional health details
        """
        
        with self._lock:
            service_info = self.services.get(service_id)
            if service_info:
                old_status = service_info.health_status
                service_info.health_status = health_status
                
                # Record health check result
                health_result = HealthCheckResult(
                    service_id=service_id,
                    status=health_status,
                    response_time_ms=0.0,  # Will be updated by actual health checks
                    error_message=None,
                    timestamp=datetime.now(),
                    details=details or {}
                )
                
                history = self.health_history.get(service_id, [])
                history.append(health_result)
                
                # Limit history size
                if len(history) > self.max_health_history:
                    history.pop(0)
                
                self.health_history[service_id] = history
                
                # Notify callbacks if status changed
                if old_status != health_status:
                    logger.info(f"Service health changed: {service_info.service_name} ({service_id}) "
                              f"{old_status.value} -> {health_status.value}")
                    
                    for callback in self.service_health_changed_callbacks:
                        try:
                            callback(service_info, health_status)
                        except Exception as e:
                            logger.error(f"Error in service health changed callback: {str(e)}")
    
    def get_service(self, service_id: str) -> Optional[ServiceInfo]:
        """Get service information by ID.
        
        Args:
            service_id: Service ID
            
        Returns:
            Service information or None
        """
        
        with self._lock:
            return self.services.get(service_id)
    
    def get_services_by_name(self, service_name: str) -> List[ServiceInfo]:
        """Get all services with a specific name.
        
        Args:
            service_name: Service name
            
        Returns:
            List of matching services
        """
        
        with self._lock:
            return [
                service for service in self.services.values()
                if service.service_name == service_name
            ]
    
    def get_services_by_type(self, service_type: str) -> List[ServiceInfo]:
        """Get all services of a specific type.
        
        Args:
            service_type: Service type
            
        Returns:
            List of matching services
        """
        
        with self._lock:
            return [
                service for service in self.services.values()
                if service.service_type == service_type
            ]
    
    def get_healthy_services(self, service_name: Optional[str] = None, service_type: Optional[str] = None) -> List[ServiceInfo]:
        """Get all healthy services, optionally filtered by name or type.
        
        Args:
            service_name: Optional service name filter
            service_type: Optional service type filter
            
        Returns:
            List of healthy services
        """
        
        with self._lock:
            services = [
                service for service in self.services.values()
                if service.health_status == ServiceHealth.HEALTHY
            ]
            
            if service_name:
                services = [s for s in services if s.service_name == service_name]
            
            if service_type:
                services = [s for s in services if s.service_type == service_type]
            
            return services
    
    def get_services_by_tags(self, tags: List[str], match_all: bool = False) -> List[ServiceInfo]:
        """Get services by tags.
        
        Args:
            tags: Tags to match
            match_all: If True, service must have all tags; if False, any tag
            
        Returns:
            List of matching services
        """
        
        with self._lock:
            services = []
            
            for service in self.services.values():
                if match_all:
                    if all(tag in service.tags for tag in tags):
                        services.append(service)
                else:
                    if any(tag in service.tags for tag in tags):
                        services.append(service)
            
            return services
    
    def cleanup_stale_services(self):
        """Remove services that haven't sent heartbeats recently."""
        
        cutoff_time = datetime.now() - timedelta(seconds=self.service_timeout)
        stale_services = []
        
        with self._lock:
            for service_id, service_info in list(self.services.items()):
                if service_info.last_heartbeat < cutoff_time:
                    stale_services.append(service_id)
            
            for service_id in stale_services:
                service_info = self.services[service_id]
                logger.warning(f"Removing stale service: {service_info.service_name} ({service_id})")
                self.deregister_service(service_id)
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        
        with self._lock:
            stats = {
                'total_services': len(self.services),
                'services_by_type': {},
                'services_by_health': {},
                'services_by_name': {}
            }
            
            for service in self.services.values():
                # Count by type
                stats['services_by_type'][service.service_type] = \
                    stats['services_by_type'].get(service.service_type, 0) + 1
                
                # Count by health
                stats['services_by_health'][service.health_status.value] = \
                    stats['services_by_health'].get(service.health_status.value, 0) + 1
                
                # Count by name
                stats['services_by_name'][service.service_name] = \
                    stats['services_by_name'].get(service.service_name, 0) + 1
            
            return stats
    
    def add_service_registered_callback(self, callback: Callable[[ServiceInfo], None]):
        """Add callback for service registration events."""
        self.service_registered_callbacks.append(callback)
    
    def add_service_deregistered_callback(self, callback: Callable[[ServiceInfo], None]):
        """Add callback for service deregistration events."""
        self.service_deregistered_callbacks.append(callback)
    
    def add_service_health_changed_callback(self, callback: Callable[[ServiceInfo, ServiceHealth], None]):
        """Add callback for service health change events."""
        self.service_health_changed_callbacks.append(callback)


class ServiceDiscovery:
    """Service discovery client and health monitoring."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the service discovery system.
        
        Args:
            config: Configuration for service discovery
        """
        self.config = config
        self.registry = ServiceRegistry(config.get('registry', {}))
        
        # Health checking
        self.health_check_enabled = config.get('health_check_enabled', True)
        self.health_check_timeout = config.get('health_check_timeout_seconds', 10)
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        # HTTP session for health checks
        self._http_session: Optional[aiohttp.ClientSession] = None
        
        logger.info("Service discovery initialized")
    
    async def start(self):
        """Start the service discovery system."""
        
        if self._cleanup_task is not None:
            return  # Already started
        
        # Create HTTP session for health checks
        timeout = aiohttp.ClientTimeout(total=self.health_check_timeout)
        self._http_session = aiohttp.ClientSession(timeout=timeout)
        
        # Start background tasks
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        if self.health_check_enabled:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info("Service discovery started")
    
    async def stop(self):
        """Stop the service discovery system."""
        
        self._shutdown = True
        
        # Cancel background tasks
        for task in [self._cleanup_task, self._health_check_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close HTTP session
        if self._http_session:
            await self._http_session.close()
        
        logger.info("Service discovery stopped")
    
    async def register_service(
        self,
        service_name: str,
        service_type: str,
        port: int,
        metadata: Optional[Dict[str, Any]] = None,
        health_check_path: str = "/health",
        tags: Optional[List[str]] = None
    ) -> str:
        """Register this service instance.
        
        Args:
            service_name: Name of the service
            service_type: Type of service
            port: Service port
            metadata: Additional metadata
            health_check_path: Path for health checks
            tags: Service tags
            
        Returns:
            Service ID
        """
        
        # Get local IP address
        host = self._get_local_ip()
        
        # Construct health check URL
        health_check_url = f"http://{host}:{port}{health_check_path}"
        
        service_id = self.registry.register_service(
            service_name=service_name,
            service_type=service_type,
            host=host,
            port=port,
            metadata=metadata,
            health_check_url=health_check_url,
            tags=tags
        )
        
        return service_id
    
    def _get_local_ip(self) -> str:
        """Get the local IP address."""
        
        try:
            # Connect to a remote address to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
    
    async def discover_services(
        self,
        service_name: Optional[str] = None,
        service_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        healthy_only: bool = True
    ) -> List[ServiceInfo]:
        """Discover services based on criteria.
        
        Args:
            service_name: Optional service name filter
            service_type: Optional service type filter
            tags: Optional tags filter
            healthy_only: Only return healthy services
            
        Returns:
            List of matching services
        """
        
        if healthy_only:
            services = self.registry.get_healthy_services(service_name, service_type)
        else:
            services = list(self.registry.services.values())
            
            if service_name:
                services = [s for s in services if s.service_name == service_name]
            
            if service_type:
                services = [s for s in services if s.service_type == service_type]
        
        if tags:
            services = [s for s in services if any(tag in s.tags for tag in tags)]
        
        return services
    
    async def get_service_endpoint(self, service_name: str, load_balance: bool = True) -> Optional[str]:
        """Get an endpoint for a service.
        
        Args:
            service_name: Name of the service
            load_balance: Whether to load balance between instances
            
        Returns:
            Service endpoint URL or None
        """
        
        services = await self.discover_services(service_name=service_name, healthy_only=True)
        
        if not services:
            return None
        
        if load_balance:
            # Simple round-robin load balancing
            import random
            service = random.choice(services)
        else:
            service = services[0]
        
        return f"http://{service.host}:{service.port}"
    
    async def _cleanup_loop(self):
        """Background task for cleaning up stale services."""
        
        try:
            while not self._shutdown:
                await asyncio.sleep(self.registry.heartbeat_interval)
                
                if self._shutdown:
                    break
                
                self.registry.cleanup_stale_services()
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Cleanup loop error: {str(e)}")
    
    async def _health_check_loop(self):
        """Background task for health checking services."""
        
        try:
            while not self._shutdown:
                await asyncio.sleep(self.registry.health_check_interval)
                
                if self._shutdown:
                    break
                
                await self._perform_health_checks()
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Health check loop error: {str(e)}")
    
    async def _perform_health_checks(self):
        """Perform health checks on all registered services."""
        
        services = list(self.registry.services.values())
        
        # Perform health checks concurrently
        tasks = []
        for service in services:
            if service.health_check_url:
                task = asyncio.create_task(self._check_service_health(service))
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_service_health(self, service: ServiceInfo):
        """Check health of a single service.
        
        Args:
            service: Service to check
        """
        
        if not self._http_session or not service.health_check_url:
            return
        
        start_time = time.time()
        
        try:
            async with self._http_session.get(service.health_check_url) as response:
                response_time_ms = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    # Try to parse response for additional details
                    try:
                        details = await response.json()
                    except Exception:
                        details = {}
                    
                    health_result = HealthCheckResult(
                        service_id=service.service_id,
                        status=ServiceHealth.HEALTHY,
                        response_time_ms=response_time_ms,
                        error_message=None,
                        timestamp=datetime.now(),
                        details=details
                    )
                    
                    self.registry.update_service_health(
                        service.service_id,
                        ServiceHealth.HEALTHY,
                        details
                    )
                
                else:
                    health_result = HealthCheckResult(
                        service_id=service.service_id,
                        status=ServiceHealth.UNHEALTHY,
                        response_time_ms=response_time_ms,
                        error_message=f"HTTP {response.status}",
                        timestamp=datetime.now(),
                        details={}
                    )
                    
                    self.registry.update_service_health(
                        service.service_id,
                        ServiceHealth.UNHEALTHY,
                        {"http_status": response.status}
                    )
        
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            
            health_result = HealthCheckResult(
                service_id=service.service_id,
                status=ServiceHealth.UNHEALTHY,
                response_time_ms=response_time_ms,
                error_message=str(e),
                timestamp=datetime.now(),
                details={}
            )
            
            self.registry.update_service_health(
                service.service_id,
                ServiceHealth.UNHEALTHY,
                {"error": str(e)}
            )
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get service discovery statistics."""
        
        registry_stats = self.registry.get_registry_stats()
        
        return {
            'registry': registry_stats,
            'health_check_enabled': self.health_check_enabled,
            'is_running': self._cleanup_task is not None and not self._cleanup_task.done(),
            'total_health_checks': sum(
                len(history) for history in self.registry.health_history.values()
            )
        }
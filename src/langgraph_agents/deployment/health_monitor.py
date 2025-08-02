"""
Health monitoring and alerting for production deployment.

This module provides comprehensive health monitoring, alerting,
and system health assessment for multi-agent systems.
"""

import asyncio
import logging
import psutil
import time
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Configuration for a health check."""
    name: str
    check_function: Callable[[], Any]
    interval_seconds: int
    timeout_seconds: int
    warning_threshold: Optional[float]
    critical_threshold: Optional[float]
    enabled: bool = True
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class HealthCheckResult:
    """Result of a health check execution."""
    check_name: str
    status: HealthStatus
    value: Any
    message: str
    timestamp: datetime
    execution_time_ms: float
    error: Optional[str] = None
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class SystemHealth:
    """Overall system health assessment."""
    overall_status: HealthStatus
    timestamp: datetime
    check_results: List[HealthCheckResult]
    summary: Dict[str, Any]
    alerts: List[Dict[str, Any]]


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the health monitor.
        
        Args:
            config: Configuration for health monitoring
        """
        self.config = config
        self.health_checks: Dict[str, HealthCheck] = {}
        self.check_results: Dict[str, List[HealthCheckResult]] = {}
        self.max_result_history = config.get('max_result_history', 1000)
        
        # Monitoring settings
        self.monitoring_enabled = config.get('enabled', True)
        self.default_interval = config.get('default_interval_seconds', 60)
        self.default_timeout = config.get('default_timeout_seconds', 30)
        
        # Alerting settings
        self.alerting_enabled = config.get('alerting_enabled', True)
        self.alert_cooldown_seconds = config.get('alert_cooldown_seconds', 300)
        self.last_alerts: Dict[str, datetime] = {}
        
        # Background tasks
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
        self._shutdown = False
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Callbacks
        self.health_change_callbacks: List[Callable[[str, HealthCheckResult], None]] = []
        self.alert_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Initialize default health checks
        self._initialize_default_checks()
        
        logger.info("Health monitor initialized")
    
    def _initialize_default_checks(self):
        """Initialize default system health checks."""
        
        # CPU usage check
        self.add_health_check(
            name="cpu_usage",
            check_function=self._check_cpu_usage,
            interval_seconds=30,
            warning_threshold=70.0,
            critical_threshold=90.0,
            tags=["system", "performance"]
        )
        
        # Memory usage check
        self.add_health_check(
            name="memory_usage",
            check_function=self._check_memory_usage,
            interval_seconds=30,
            warning_threshold=80.0,
            critical_threshold=95.0,
            tags=["system", "performance"]
        )
        
        # Disk usage check
        self.add_health_check(
            name="disk_usage",
            check_function=self._check_disk_usage,
            interval_seconds=60,
            warning_threshold=80.0,
            critical_threshold=95.0,
            tags=["system", "storage"]
        )
        
        # Process health check
        self.add_health_check(
            name="process_health",
            check_function=self._check_process_health,
            interval_seconds=30,
            tags=["system", "process"]
        )
        
        # Network connectivity check
        self.add_health_check(
            name="network_connectivity",
            check_function=self._check_network_connectivity,
            interval_seconds=60,
            tags=["system", "network"]
        )
    
    def add_health_check(
        self,
        name: str,
        check_function: Callable[[], Any],
        interval_seconds: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        warning_threshold: Optional[float] = None,
        critical_threshold: Optional[float] = None,
        enabled: bool = True,
        tags: Optional[List[str]] = None
    ):
        """Add a health check.
        
        Args:
            name: Unique name for the health check
            check_function: Function to execute for the check
            interval_seconds: Check interval (defaults to system default)
            timeout_seconds: Check timeout (defaults to system default)
            warning_threshold: Warning threshold value
            critical_threshold: Critical threshold value
            enabled: Whether the check is enabled
            tags: Tags for categorizing the check
        """
        
        health_check = HealthCheck(
            name=name,
            check_function=check_function,
            interval_seconds=interval_seconds or self.default_interval,
            timeout_seconds=timeout_seconds or self.default_timeout,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
            enabled=enabled,
            tags=tags or []
        )
        
        with self._lock:
            self.health_checks[name] = health_check
            self.check_results[name] = []
        
        logger.info(f"Health check added: {name}")
    
    def remove_health_check(self, name: str) -> bool:
        """Remove a health check.
        
        Args:
            name: Name of the health check to remove
            
        Returns:
            True if check was found and removed
        """
        
        with self._lock:
            health_check = self.health_checks.pop(name, None)
            if health_check:
                self.check_results.pop(name, None)
                
                # Cancel monitoring task if running
                task = self._monitoring_tasks.pop(name, None)
                if task and not task.done():
                    task.cancel()
                
                logger.info(f"Health check removed: {name}")
                return True
        
        return False
    
    def enable_health_check(self, name: str, enabled: bool = True):
        """Enable or disable a health check.
        
        Args:
            name: Name of the health check
            enabled: Whether to enable the check
        """
        
        with self._lock:
            health_check = self.health_checks.get(name)
            if health_check:
                health_check.enabled = enabled
                logger.info(f"Health check {'enabled' if enabled else 'disabled'}: {name}")
    
    async def start_monitoring(self):
        """Start health monitoring for all enabled checks."""
        
        if not self.monitoring_enabled:
            logger.info("Health monitoring is disabled")
            return
        
        with self._lock:
            for name, health_check in self.health_checks.items():
                if health_check.enabled and name not in self._monitoring_tasks:
                    task = asyncio.create_task(self._monitor_health_check(health_check))
                    self._monitoring_tasks[name] = task
        
        logger.info(f"Started monitoring {len(self._monitoring_tasks)} health checks")
    
    async def stop_monitoring(self):
        """Stop all health monitoring."""
        
        self._shutdown = True
        
        with self._lock:
            # Cancel all monitoring tasks
            for task in self._monitoring_tasks.values():
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self._monitoring_tasks:
                await asyncio.gather(*self._monitoring_tasks.values(), return_exceptions=True)
            
            self._monitoring_tasks.clear()
        
        logger.info("Health monitoring stopped")
    
    async def _monitor_health_check(self, health_check: HealthCheck):
        """Monitor a single health check.
        
        Args:
            health_check: Health check to monitor
        """
        
        try:
            while not self._shutdown and health_check.enabled:
                # Execute the health check
                result = await self._execute_health_check(health_check)
                
                # Store result
                with self._lock:
                    results = self.check_results.get(health_check.name, [])
                    results.append(result)
                    
                    # Limit history size
                    if len(results) > self.max_result_history:
                        results.pop(0)
                    
                    self.check_results[health_check.name] = results
                
                # Notify callbacks
                for callback in self.health_change_callbacks:
                    try:
                        callback(health_check.name, result)
                    except Exception as e:
                        logger.error(f"Error in health change callback: {str(e)}")
                
                # Check for alerts
                if self.alerting_enabled:
                    await self._check_for_alerts(health_check, result)
                
                # Wait for next check
                await asyncio.sleep(health_check.interval_seconds)
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in health check monitoring for {health_check.name}: {str(e)}")
    
    async def _execute_health_check(self, health_check: HealthCheck) -> HealthCheckResult:
        """Execute a single health check.
        
        Args:
            health_check: Health check to execute
            
        Returns:
            Health check result
        """
        
        start_time = time.time()
        
        try:
            # Execute check with timeout
            value = await asyncio.wait_for(
                asyncio.create_task(self._run_check_function(health_check.check_function)),
                timeout=health_check.timeout_seconds
            )
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Determine status based on thresholds
            status = self._determine_status(value, health_check)
            
            # Create result
            result = HealthCheckResult(
                check_name=health_check.name,
                status=status,
                value=value,
                message=self._create_status_message(health_check.name, status, value),
                timestamp=datetime.now(),
                execution_time_ms=execution_time_ms
            )
            
            return result
        
        except asyncio.TimeoutError:
            execution_time_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                check_name=health_check.name,
                status=HealthStatus.CRITICAL,
                value=None,
                message=f"Health check timed out after {health_check.timeout_seconds}s",
                timestamp=datetime.now(),
                execution_time_ms=execution_time_ms,
                error="timeout"
            )
        
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                check_name=health_check.name,
                status=HealthStatus.CRITICAL,
                value=None,
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.now(),
                execution_time_ms=execution_time_ms,
                error=str(e)
            )
    
    async def _run_check_function(self, check_function: Callable) -> Any:
        """Run a check function, handling both sync and async functions."""
        
        if asyncio.iscoroutinefunction(check_function):
            return await check_function()
        else:
            # Run sync function in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, check_function)
    
    def _determine_status(self, value: Any, health_check: HealthCheck) -> HealthStatus:
        """Determine health status based on value and thresholds."""
        
        if value is None:
            return HealthStatus.UNKNOWN
        
        # For boolean values
        if isinstance(value, bool):
            return HealthStatus.HEALTHY if value else HealthStatus.CRITICAL
        
        # For numeric values with thresholds
        if isinstance(value, (int, float)) and (health_check.warning_threshold or health_check.critical_threshold):
            if health_check.critical_threshold and value >= health_check.critical_threshold:
                return HealthStatus.CRITICAL
            elif health_check.warning_threshold and value >= health_check.warning_threshold:
                return HealthStatus.WARNING
            else:
                return HealthStatus.HEALTHY
        
        # Default to healthy for other types
        return HealthStatus.HEALTHY
    
    def _create_status_message(self, check_name: str, status: HealthStatus, value: Any) -> str:
        """Create a human-readable status message."""
        
        if status == HealthStatus.HEALTHY:
            return f"{check_name}: OK ({value})"
        elif status == HealthStatus.WARNING:
            return f"{check_name}: WARNING ({value})"
        elif status == HealthStatus.CRITICAL:
            return f"{check_name}: CRITICAL ({value})"
        else:
            return f"{check_name}: UNKNOWN"
    
    async def _check_for_alerts(self, health_check: HealthCheck, result: HealthCheckResult):
        """Check if an alert should be triggered."""
        
        if result.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
            # Check cooldown
            last_alert = self.last_alerts.get(health_check.name)
            if last_alert:
                time_since_last = (datetime.now() - last_alert).total_seconds()
                if time_since_last < self.alert_cooldown_seconds:
                    return  # Still in cooldown
            
            # Create alert
            alert = {
                'check_name': health_check.name,
                'status': result.status.value,
                'value': result.value,
                'message': result.message,
                'timestamp': result.timestamp.isoformat(),
                'tags': health_check.tags,
                'details': result.details
            }
            
            # Update last alert time
            self.last_alerts[health_check.name] = datetime.now()
            
            # Notify alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {str(e)}")
            
            logger.warning(f"Health alert: {result.message}")
    
    async def get_system_health(self) -> SystemHealth:
        """Get overall system health assessment.
        
        Returns:
            System health summary
        """
        
        with self._lock:
            all_results = []
            status_counts = {status: 0 for status in HealthStatus}
            
            # Get latest result for each check
            for check_name, results in self.check_results.items():
                if results:
                    latest_result = results[-1]
                    all_results.append(latest_result)
                    status_counts[latest_result.status] += 1
            
            # Determine overall status
            if status_counts[HealthStatus.CRITICAL] > 0:
                overall_status = HealthStatus.CRITICAL
            elif status_counts[HealthStatus.WARNING] > 0:
                overall_status = HealthStatus.WARNING
            elif status_counts[HealthStatus.UNKNOWN] > 0:
                overall_status = HealthStatus.UNKNOWN
            else:
                overall_status = HealthStatus.HEALTHY
            
            # Create summary
            summary = {
                'total_checks': len(all_results),
                'healthy_checks': status_counts[HealthStatus.HEALTHY],
                'warning_checks': status_counts[HealthStatus.WARNING],
                'critical_checks': status_counts[HealthStatus.CRITICAL],
                'unknown_checks': status_counts[HealthStatus.UNKNOWN],
                'enabled_checks': sum(1 for hc in self.health_checks.values() if hc.enabled),
                'monitoring_active': len(self._monitoring_tasks)
            }
            
            # Get recent alerts
            recent_alerts = []
            cutoff_time = datetime.now() - timedelta(hours=1)
            
            for check_name, results in self.check_results.items():
                for result in reversed(results):
                    if result.timestamp < cutoff_time:
                        break
                    
                    if result.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                        recent_alerts.append({
                            'check_name': result.check_name,
                            'status': result.status.value,
                            'message': result.message,
                            'timestamp': result.timestamp.isoformat()
                        })
            
            return SystemHealth(
                overall_status=overall_status,
                timestamp=datetime.now(),
                check_results=all_results,
                summary=summary,
                alerts=recent_alerts
            )
    
    def get_check_history(self, check_name: str, hours: int = 24) -> List[HealthCheckResult]:
        """Get history for a specific health check.
        
        Args:
            check_name: Name of the health check
            hours: Number of hours of history to return
            
        Returns:
            List of health check results
        """
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            results = self.check_results.get(check_name, [])
            return [r for r in results if r.timestamp > cutoff_time]
    
    def add_health_change_callback(self, callback: Callable[[str, HealthCheckResult], None]):
        """Add callback for health status changes."""
        self.health_change_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for health alerts."""
        self.alert_callbacks.append(callback)
    
    # Default health check implementations
    
    def _check_cpu_usage(self) -> float:
        """Check CPU usage percentage."""
        return psutil.cpu_percent(interval=1)
    
    def _check_memory_usage(self) -> float:
        """Check memory usage percentage."""
        return psutil.virtual_memory().percent
    
    def _check_disk_usage(self) -> float:
        """Check disk usage percentage."""
        return psutil.disk_usage('/').percent
    
    def _check_process_health(self) -> bool:
        """Check if the current process is healthy."""
        try:
            process = psutil.Process()
            # Check if process is running and responsive
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except Exception:
            return False
    
    def _check_network_connectivity(self) -> bool:
        """Check basic network connectivity."""
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            return True
        except Exception:
            return False
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get health monitoring statistics."""
        
        with self._lock:
            return {
                'monitoring_enabled': self.monitoring_enabled,
                'alerting_enabled': self.alerting_enabled,
                'total_checks': len(self.health_checks),
                'enabled_checks': sum(1 for hc in self.health_checks.values() if hc.enabled),
                'active_monitoring_tasks': len(self._monitoring_tasks),
                'total_check_results': sum(len(results) for results in self.check_results.values()),
                'recent_alerts': len([
                    alert_time for alert_time in self.last_alerts.values()
                    if (datetime.now() - alert_time).total_seconds() < 3600
                ]),
                'check_names': list(self.health_checks.keys())
            }
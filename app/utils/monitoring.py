"""
Performance monitoring and metrics utilities.
"""

import time
import threading
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import functools
import asyncio

from .logging import get_logger

# Optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MetricsCollector:
    """Collects and stores performance metrics."""
    
    def __init__(self, max_metrics: int = 10000):
        self.metrics: deque = deque(maxlen=max_metrics)
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        self.logger = get_logger(__name__)
    
    def record_metric(self, metric: PerformanceMetric) -> None:
        """Record a performance metric."""
        with self._lock:
            self.metrics.append(metric)
            self.logger.debug(f"Recorded metric: {metric.name} = {metric.value} {metric.unit}")
    
    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None) -> None:
        """Increment a counter metric."""
        with self._lock:
            self.counters[name] += value
            
        metric = PerformanceMetric(
            name=name,
            value=self.counters[name],
            unit="count",
            tags=tags or {}
        )
        self.record_metric(metric)
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Set a gauge metric."""
        with self._lock:
            self.gauges[name] = value
            
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit="gauge",
            tags=tags or {}
        )
        self.record_metric(metric)
    
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Record a histogram value."""
        with self._lock:
            self.histograms[name].append(value)
            # Keep only last 1000 values
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-1000:]
        
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit="histogram",
            tags=tags or {}
        )
        self.record_metric(metric)
    
    def get_metrics(self, since: Optional[datetime] = None) -> List[PerformanceMetric]:
        """Get metrics since a specific time."""
        with self._lock:
            if since is None:
                return list(self.metrics)
            
            return [m for m in self.metrics if m.timestamp >= since]
    
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a histogram."""
        with self._lock:
            values = self.histograms.get(name, [])
            
        if not values:
            return {}
        
        sorted_values = sorted(values)
        count = len(sorted_values)
        
        return {
            "count": count,
            "min": min(sorted_values),
            "max": max(sorted_values),
            "mean": sum(sorted_values) / count,
            "median": sorted_values[count // 2],
            "p95": sorted_values[int(count * 0.95)] if count > 0 else 0,
            "p99": sorted_values[int(count * 0.99)] if count > 0 else 0,
        }
    
    def clear_metrics(self) -> None:
        """Clear all stored metrics."""
        with self._lock:
            self.metrics.clear()
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()


# Global metrics collector instance
metrics_collector = MetricsCollector()


def get_system_metrics() -> SystemMetrics:
    """
    Get current system performance metrics.
    
    Raises:
        ImportError: If psutil is not available
    """
    if not PSUTIL_AVAILABLE:
        raise ImportError("psutil is required for system metrics. Install with: pip install psutil")
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Memory usage
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    memory_used_mb = memory.used / (1024 * 1024)
    memory_available_mb = memory.available / (1024 * 1024)
    
    # Disk usage
    disk = psutil.disk_usage('/')
    disk_usage_percent = (disk.used / disk.total) * 100
    disk_free_gb = disk.free / (1024 * 1024 * 1024)
    
    return SystemMetrics(
        cpu_percent=cpu_percent,
        memory_percent=memory_percent,
        memory_used_mb=memory_used_mb,
        memory_available_mb=memory_available_mb,
        disk_usage_percent=disk_usage_percent,
        disk_free_gb=disk_free_gb
    )


def monitor_performance(metric_name: str = None, tags: Dict[str, str] = None):
    """
    Decorator to monitor function performance.
    
    Args:
        metric_name: Custom metric name, defaults to function name
        tags: Additional tags for the metric
    """
    def decorator(func: Callable) -> Callable:
        name = metric_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                duration = time.time() - start_time
                
                # Record performance metrics
                metric_tags = (tags or {}).copy()
                metric_tags.update({
                    "function": func.__name__,
                    "success": str(success)
                })
                
                if error:
                    metric_tags["error"] = error
                
                metrics_collector.record_histogram(
                    f"{name}.duration",
                    duration,
                    metric_tags
                )
                
                metrics_collector.increment_counter(
                    f"{name}.calls",
                    tags=metric_tags
                )
            
            return result
        
        return wrapper
    return decorator


def monitor_async_performance(metric_name: str = None, tags: Dict[str, str] = None):
    """
    Decorator to monitor async function performance.
    
    Args:
        metric_name: Custom metric name, defaults to function name
        tags: Additional tags for the metric
    """
    def decorator(func: Callable) -> Callable:
        name = metric_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                duration = time.time() - start_time
                
                # Record performance metrics
                metric_tags = (tags or {}).copy()
                metric_tags.update({
                    "function": func.__name__,
                    "success": str(success)
                })
                
                if error:
                    metric_tags["error"] = error
                
                metrics_collector.record_histogram(
                    f"{name}.duration",
                    duration,
                    metric_tags
                )
                
                metrics_collector.increment_counter(
                    f"{name}.calls",
                    tags=metric_tags
                )
            
            return result
        
        return wrapper
    return decorator


class PerformanceMonitor:
    """Context manager for monitoring code blocks."""
    
    def __init__(self, name: str, tags: Dict[str, str] = None):
        self.name = name
        self.tags = tags or {}
        self.start_time = None
        self.logger = get_logger(__name__)
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug(f"Starting performance monitoring: {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            
            success = exc_type is None
            metric_tags = self.tags.copy()
            metric_tags.update({
                "success": str(success)
            })
            
            if exc_type:
                metric_tags["error"] = str(exc_val)
            
            metrics_collector.record_histogram(
                f"{self.name}.duration",
                duration,
                metric_tags
            )
            
            self.logger.debug(
                f"Completed performance monitoring: {self.name} in {duration:.3f}s"
            )


class AsyncPerformanceMonitor:
    """Async context manager for monitoring code blocks."""
    
    def __init__(self, name: str, tags: Dict[str, str] = None):
        self.name = name
        self.tags = tags or {}
        self.start_time = None
        self.logger = get_logger(__name__)
    
    async def __aenter__(self):
        self.start_time = time.time()
        self.logger.debug(f"Starting async performance monitoring: {self.name}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            
            success = exc_type is None
            metric_tags = self.tags.copy()
            metric_tags.update({
                "success": str(success)
            })
            
            if exc_type:
                metric_tags["error"] = str(exc_val)
            
            metrics_collector.record_histogram(
                f"{self.name}.duration",
                duration,
                metric_tags
            )
            
            self.logger.debug(
                f"Completed async performance monitoring: {self.name} in {duration:.3f}s"
            )


def start_system_monitoring(interval: int = 60) -> None:
    """
    Start background system monitoring.
    
    Args:
        interval: Monitoring interval in seconds
        
    Raises:
        ImportError: If psutil is not available
    """
    if not PSUTIL_AVAILABLE:
        raise ImportError("psutil is required for system monitoring. Install with: pip install psutil")
    
    def monitor_system():
        logger = get_logger(__name__)
        
        while True:
            try:
                system_metrics = get_system_metrics()
                
                # Record system metrics
                metrics_collector.set_gauge("system.cpu_percent", system_metrics.cpu_percent)
                metrics_collector.set_gauge("system.memory_percent", system_metrics.memory_percent)
                metrics_collector.set_gauge("system.memory_used_mb", system_metrics.memory_used_mb)
                metrics_collector.set_gauge("system.disk_usage_percent", system_metrics.disk_usage_percent)
                
                logger.debug(f"Recorded system metrics: CPU {system_metrics.cpu_percent}%, Memory {system_metrics.memory_percent}%")
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
            
            time.sleep(interval)
    
    # Start monitoring in background thread
    monitor_thread = threading.Thread(target=monitor_system, daemon=True)
    monitor_thread.start()


def get_performance_summary(hours: int = 1) -> Dict[str, Any]:
    """
    Get performance summary for the last N hours.
    
    Args:
        hours: Number of hours to look back
        
    Returns:
        Performance summary dictionary
    """
    since = datetime.utcnow() - timedelta(hours=hours)
    metrics = metrics_collector.get_metrics(since)
    
    summary = {
        "period_hours": hours,
        "total_metrics": len(metrics),
        "counters": dict(metrics_collector.counters),
        "gauges": dict(metrics_collector.gauges),
        "histograms": {}
    }
    
    # Add histogram statistics
    for name in metrics_collector.histograms:
        summary["histograms"][name] = metrics_collector.get_histogram_stats(name)
    
    return summary
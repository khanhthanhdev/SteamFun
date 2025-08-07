"""
Performance metrics collection system for agent testing.

This module provides comprehensive performance metrics collection and analysis
for individual agents and overall workflow execution.
"""

import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    """A snapshot of performance metrics at a specific point in time."""
    
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    thread_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_mb': self.memory_used_mb,
            'disk_io_read_mb': self.disk_io_read_mb,
            'disk_io_write_mb': self.disk_io_write_mb,
            'network_sent_mb': self.network_sent_mb,
            'network_recv_mb': self.network_recv_mb,
            'thread_count': self.thread_count
        }


@dataclass
class AgentPerformanceMetrics:
    """Performance metrics for a specific agent execution."""
    
    agent_name: str
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    execution_time: Optional[float] = None
    
    # Resource usage snapshots
    snapshots: List[PerformanceSnapshot] = field(default_factory=list)
    
    # Agent-specific metrics
    api_calls_made: int = 0
    tokens_used: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    retry_count: int = 0
    error_count: int = 0
    
    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def add_snapshot(self, snapshot: PerformanceSnapshot) -> None:
        """Add a performance snapshot."""
        self.snapshots.append(snapshot)
    
    def add_custom_metric(self, key: str, value: Any) -> None:
        """Add a custom metric."""
        self.custom_metrics[key] = value
    
    def increment_counter(self, counter_name: str, amount: int = 1) -> None:
        """Increment a counter metric."""
        if hasattr(self, counter_name):
            current_value = getattr(self, counter_name)
            setattr(self, counter_name, current_value + amount)
    
    def finalize(self) -> None:
        """Finalize metrics collection."""
        if self.end_time is None:
            self.end_time = datetime.now()
        
        if self.start_time and self.end_time:
            self.execution_time = (self.end_time - self.start_time).total_seconds()
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get summary of resource usage."""
        if not self.snapshots:
            return {}
        
        cpu_values = [s.cpu_percent for s in self.snapshots]
        memory_values = [s.memory_percent for s in self.snapshots]
        memory_mb_values = [s.memory_used_mb for s in self.snapshots]
        
        return {
            'cpu_usage': {
                'min': min(cpu_values),
                'max': max(cpu_values),
                'avg': sum(cpu_values) / len(cpu_values),
                'samples': len(cpu_values)
            },
            'memory_usage': {
                'min_percent': min(memory_values),
                'max_percent': max(memory_values),
                'avg_percent': sum(memory_values) / len(memory_values),
                'min_mb': min(memory_mb_values),
                'max_mb': max(memory_mb_values),
                'avg_mb': sum(memory_mb_values) / len(memory_mb_values),
                'samples': len(memory_values)
            },
            'monitoring_duration': (self.snapshots[-1].timestamp - self.snapshots[0].timestamp).total_seconds() if len(self.snapshots) > 1 else 0
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'agent_name': self.agent_name,
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'execution_time': self.execution_time,
            'api_calls_made': self.api_calls_made,
            'tokens_used': self.tokens_used,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'retry_count': self.retry_count,
            'error_count': self.error_count,
            'custom_metrics': self.custom_metrics,
            'resource_summary': self.get_resource_summary(),
            'snapshot_count': len(self.snapshots)
        }


class PerformanceMonitor:
    """
    Real-time performance monitoring for agent execution.
    
    Collects system resource usage and agent-specific metrics during execution.
    """
    
    def __init__(self, sampling_interval: float = 1.0):
        """
        Initialize performance monitor.
        
        Args:
            sampling_interval: Interval in seconds between resource snapshots
        """
        self.sampling_interval = sampling_interval
        self.active_sessions: Dict[str, AgentPerformanceMetrics] = {}
        self.monitoring_threads: Dict[str, threading.Thread] = {}
        self.stop_events: Dict[str, threading.Event] = {}
        
        # Initialize psutil for system monitoring
        self.process = psutil.Process()
        
        # Baseline measurements
        self.baseline_disk_io = psutil.disk_io_counters()
        self.baseline_network_io = psutil.net_io_counters()
    
    def start_monitoring(self, session_id: str, agent_name: str) -> AgentPerformanceMetrics:
        """Start monitoring performance for a session."""
        if session_id in self.active_sessions:
            logger.warning(f"Performance monitoring already active for session {session_id}")
            return self.active_sessions[session_id]
        
        # Create metrics container
        metrics = AgentPerformanceMetrics(
            agent_name=agent_name,
            session_id=session_id,
            start_time=datetime.now()
        )
        
        self.active_sessions[session_id] = metrics
        
        # Start monitoring thread
        stop_event = threading.Event()
        self.stop_events[session_id] = stop_event
        
        monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(session_id, stop_event),
            daemon=True
        )
        monitoring_thread.start()
        self.monitoring_threads[session_id] = monitoring_thread
        
        logger.info(f"Started performance monitoring for session {session_id}")
        return metrics
    
    def stop_monitoring(self, session_id: str) -> Optional[AgentPerformanceMetrics]:
        """Stop monitoring performance for a session."""
        if session_id not in self.active_sessions:
            logger.warning(f"No active performance monitoring for session {session_id}")
            return None
        
        # Stop monitoring thread
        if session_id in self.stop_events:
            self.stop_events[session_id].set()
        
        # Wait for thread to finish
        if session_id in self.monitoring_threads:
            self.monitoring_threads[session_id].join(timeout=2.0)
            del self.monitoring_threads[session_id]
        
        # Finalize metrics
        metrics = self.active_sessions[session_id]
        metrics.finalize()
        
        # Cleanup
        del self.active_sessions[session_id]
        if session_id in self.stop_events:
            del self.stop_events[session_id]
        
        logger.info(f"Stopped performance monitoring for session {session_id}")
        return metrics
    
    def add_metric(self, session_id: str, metric_name: str, value: Any) -> None:
        """Add a custom metric to the session."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].add_custom_metric(metric_name, value)
    
    def increment_counter(self, session_id: str, counter_name: str, amount: int = 1) -> None:
        """Increment a counter metric for the session."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].increment_counter(counter_name, amount)
    
    def _monitoring_loop(self, session_id: str, stop_event: threading.Event) -> None:
        """Main monitoring loop for collecting performance snapshots."""
        try:
            while not stop_event.is_set():
                try:
                    snapshot = self._take_snapshot()
                    if session_id in self.active_sessions:
                        self.active_sessions[session_id].add_snapshot(snapshot)
                except Exception as e:
                    logger.error(f"Error taking performance snapshot for session {session_id}: {e}")
                
                # Wait for next sampling interval
                stop_event.wait(self.sampling_interval)
                
        except Exception as e:
            logger.error(f"Performance monitoring loop failed for session {session_id}: {e}")
    
    def _take_snapshot(self) -> PerformanceSnapshot:
        """Take a snapshot of current system performance."""
        try:
            # CPU and memory
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            memory_used_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
            
            # Disk I/O
            current_disk_io = psutil.disk_io_counters()
            disk_read_mb = (current_disk_io.read_bytes - self.baseline_disk_io.read_bytes) / (1024 * 1024)
            disk_write_mb = (current_disk_io.write_bytes - self.baseline_disk_io.write_bytes) / (1024 * 1024)
            
            # Network I/O
            current_network_io = psutil.net_io_counters()
            network_sent_mb = (current_network_io.bytes_sent - self.baseline_network_io.bytes_sent) / (1024 * 1024)
            network_recv_mb = (current_network_io.bytes_recv - self.baseline_network_io.bytes_recv) / (1024 * 1024)
            
            # Thread count
            thread_count = self.process.num_threads()
            
            return PerformanceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                disk_io_read_mb=disk_read_mb,
                disk_io_write_mb=disk_write_mb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                thread_count=thread_count
            )
            
        except Exception as e:
            logger.error(f"Error taking performance snapshot: {e}")
            # Return a minimal snapshot on error
            return PerformanceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                disk_io_read_mb=0.0,
                disk_io_write_mb=0.0,
                network_sent_mb=0.0,
                network_recv_mb=0.0,
                thread_count=0
            )
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active monitoring sessions."""
        return list(self.active_sessions.keys())
    
    def is_monitoring(self, session_id: str) -> bool:
        """Check if monitoring is active for a session."""
        return session_id in self.active_sessions


class PerformanceAnalyzer:
    """
    Analyzes performance metrics and provides insights.
    
    Compares performance across different agent executions and identifies
    potential performance issues or improvements.
    """
    
    def __init__(self):
        """Initialize performance analyzer."""
        self.historical_metrics: Dict[str, List[AgentPerformanceMetrics]] = defaultdict(list)
    
    def add_metrics(self, metrics: AgentPerformanceMetrics) -> None:
        """Add metrics to historical data."""
        self.historical_metrics[metrics.agent_name].append(metrics)
    
    def analyze_agent_performance(self, agent_name: str) -> Dict[str, Any]:
        """Analyze performance for a specific agent."""
        if agent_name not in self.historical_metrics:
            return {'error': f'No historical data for agent {agent_name}'}
        
        metrics_list = self.historical_metrics[agent_name]
        if not metrics_list:
            return {'error': f'No metrics available for agent {agent_name}'}
        
        # Execution time analysis
        execution_times = [m.execution_time for m in metrics_list if m.execution_time is not None]
        
        analysis = {
            'agent_name': agent_name,
            'total_executions': len(metrics_list),
            'execution_time_stats': self._analyze_values(execution_times) if execution_times else None,
            'api_usage_stats': self._analyze_values([m.api_calls_made for m in metrics_list]),
            'token_usage_stats': self._analyze_values([m.tokens_used for m in metrics_list]),
            'cache_performance': self._analyze_cache_performance(metrics_list),
            'error_analysis': self._analyze_errors(metrics_list),
            'resource_usage_trends': self._analyze_resource_trends(metrics_list)
        }
        
        return analysis
    
    def compare_agents(self, agent_names: List[str]) -> Dict[str, Any]:
        """Compare performance across multiple agents."""
        comparison = {}
        
        for agent_name in agent_names:
            if agent_name in self.historical_metrics:
                metrics_list = self.historical_metrics[agent_name]
                execution_times = [m.execution_time for m in metrics_list if m.execution_time is not None]
                
                comparison[agent_name] = {
                    'avg_execution_time': sum(execution_times) / len(execution_times) if execution_times else 0,
                    'total_executions': len(metrics_list),
                    'avg_api_calls': sum(m.api_calls_made for m in metrics_list) / len(metrics_list),
                    'avg_tokens_used': sum(m.tokens_used for m in metrics_list) / len(metrics_list),
                    'avg_error_rate': sum(m.error_count for m in metrics_list) / len(metrics_list)
                }
        
        return comparison
    
    def identify_performance_issues(self, metrics: AgentPerformanceMetrics) -> List[Dict[str, Any]]:
        """Identify potential performance issues in metrics."""
        issues = []
        
        # Check execution time
        if metrics.execution_time and metrics.execution_time > 300:  # 5 minutes
            issues.append({
                'type': 'slow_execution',
                'severity': 'high',
                'message': f'Execution time ({metrics.execution_time:.2f}s) exceeds recommended threshold',
                'recommendation': 'Consider optimizing agent logic or increasing timeout limits'
            })
        
        # Check error rate
        if metrics.error_count > 0:
            issues.append({
                'type': 'errors_detected',
                'severity': 'medium',
                'message': f'{metrics.error_count} errors detected during execution',
                'recommendation': 'Review error logs and improve error handling'
            })
        
        # Check cache performance
        total_cache_requests = metrics.cache_hits + metrics.cache_misses
        if total_cache_requests > 0:
            cache_hit_rate = metrics.cache_hits / total_cache_requests
            if cache_hit_rate < 0.5:
                issues.append({
                    'type': 'low_cache_hit_rate',
                    'severity': 'low',
                    'message': f'Cache hit rate ({cache_hit_rate:.2%}) is below optimal threshold',
                    'recommendation': 'Review caching strategy and cache key generation'
                })
        
        # Check resource usage
        resource_summary = metrics.get_resource_summary()
        if resource_summary and 'cpu_usage' in resource_summary:
            avg_cpu = resource_summary['cpu_usage']['avg']
            if avg_cpu > 80:
                issues.append({
                    'type': 'high_cpu_usage',
                    'severity': 'medium',
                    'message': f'Average CPU usage ({avg_cpu:.1f}%) is high',
                    'recommendation': 'Consider optimizing CPU-intensive operations or scaling resources'
                })
        
        return issues
    
    def _analyze_values(self, values: List[float]) -> Dict[str, float]:
        """Analyze a list of numeric values."""
        if not values:
            return {}
        
        sorted_values = sorted(values)
        n = len(values)
        
        return {
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / n,
            'median': sorted_values[n // 2] if n % 2 == 1 else (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2,
            'p95': sorted_values[int(0.95 * n)] if n > 0 else 0,
            'count': n
        }
    
    def _analyze_cache_performance(self, metrics_list: List[AgentPerformanceMetrics]) -> Dict[str, Any]:
        """Analyze cache performance across metrics."""
        total_hits = sum(m.cache_hits for m in metrics_list)
        total_misses = sum(m.cache_misses for m in metrics_list)
        total_requests = total_hits + total_misses
        
        if total_requests == 0:
            return {'hit_rate': 0, 'total_requests': 0}
        
        return {
            'hit_rate': total_hits / total_requests,
            'total_requests': total_requests,
            'total_hits': total_hits,
            'total_misses': total_misses
        }
    
    def _analyze_errors(self, metrics_list: List[AgentPerformanceMetrics]) -> Dict[str, Any]:
        """Analyze error patterns across metrics."""
        total_errors = sum(m.error_count for m in metrics_list)
        executions_with_errors = sum(1 for m in metrics_list if m.error_count > 0)
        
        return {
            'total_errors': total_errors,
            'executions_with_errors': executions_with_errors,
            'error_rate': executions_with_errors / len(metrics_list) if metrics_list else 0,
            'avg_errors_per_execution': total_errors / len(metrics_list) if metrics_list else 0
        }
    
    def _analyze_resource_trends(self, metrics_list: List[AgentPerformanceMetrics]) -> Dict[str, Any]:
        """Analyze resource usage trends across metrics."""
        cpu_averages = []
        memory_averages = []
        
        for metrics in metrics_list:
            resource_summary = metrics.get_resource_summary()
            if resource_summary and 'cpu_usage' in resource_summary:
                cpu_averages.append(resource_summary['cpu_usage']['avg'])
            if resource_summary and 'memory_usage' in resource_summary:
                memory_averages.append(resource_summary['memory_usage']['avg_percent'])
        
        return {
            'cpu_trends': self._analyze_values(cpu_averages) if cpu_averages else {},
            'memory_trends': self._analyze_values(memory_averages) if memory_averages else {}
        }


# Global instances for easy access
_global_performance_monitor = PerformanceMonitor()
_global_performance_analyzer = PerformanceAnalyzer()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _global_performance_monitor


def get_performance_analyzer() -> PerformanceAnalyzer:
    """Get the global performance analyzer instance."""
    return _global_performance_analyzer
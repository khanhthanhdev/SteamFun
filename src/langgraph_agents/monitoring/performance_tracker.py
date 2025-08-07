"""
Agent performance tracking and monitoring system.

This module provides detailed performance tracking for agent execution,
including processing times, resource usage, and performance alerts.
"""

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PerformanceAlertLevel(Enum):
    """Performance alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PerformanceAlert:
    """Represents a performance alert."""
    
    alert_id: str
    timestamp: datetime
    agent_name: str
    session_id: str
    alert_type: str
    level: PerformanceAlertLevel
    message: str
    metrics: Dict[str, Any]
    threshold_value: Optional[float] = None
    actual_value: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'agent_name': self.agent_name,
            'session_id': self.session_id,
            'alert_type': self.alert_type,
            'level': self.level.value,
            'message': self.message,
            'metrics': self.metrics,
            'threshold_value': self.threshold_value,
            'actual_value': self.actual_value
        }


@dataclass
class AgentPerformanceData:
    """Performance data for a single agent execution."""
    
    agent_name: str
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Timing metrics
    total_execution_time: float = 0.0
    processing_time: float = 0.0
    waiting_time: float = 0.0
    step_times: Dict[str, float] = field(default_factory=dict)
    
    # Resource usage
    peak_memory_mb: float = 0.0
    average_cpu_percent: float = 0.0
    peak_cpu_percent: float = 0.0
    disk_read_mb: float = 0.0
    disk_write_mb: float = 0.0
    network_sent_mb: float = 0.0
    network_received_mb: float = 0.0
    
    # Agent-specific metrics
    api_calls_count: int = 0
    tokens_used: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    retry_count: int = 0
    error_count: int = 0
    
    # Performance snapshots
    resource_snapshots: List[Dict[str, Any]] = field(default_factory=list)
    
    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def add_resource_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Add a resource usage snapshot."""
        self.resource_snapshots.append(snapshot)
        
        # Update peak values
        if 'memory_mb' in snapshot:
            self.peak_memory_mb = max(self.peak_memory_mb, snapshot['memory_mb'])
        if 'cpu_percent' in snapshot:
            self.peak_cpu_percent = max(self.peak_cpu_percent, snapshot['cpu_percent'])
    
    def finalize_metrics(self) -> None:
        """Finalize performance metrics calculation."""
        if self.end_time is None:
            self.end_time = datetime.now()
        
        self.total_execution_time = (self.end_time - self.start_time).total_seconds()
        
        # Calculate average CPU usage
        if self.resource_snapshots:
            cpu_values = [s.get('cpu_percent', 0) for s in self.resource_snapshots]
            self.average_cpu_percent = sum(cpu_values) / len(cpu_values) if cpu_values else 0
    
    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.cache_hits + self.cache_misses
        return self.cache_hits / total_requests if total_requests > 0 else 0.0
    
    def get_error_rate(self) -> float:
        """Calculate error rate based on API calls."""
        return self.error_count / self.api_calls_count if self.api_calls_count > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'agent_name': self.agent_name,
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_execution_time': self.total_execution_time,
            'processing_time': self.processing_time,
            'waiting_time': self.waiting_time,
            'step_times': self.step_times,
            'peak_memory_mb': self.peak_memory_mb,
            'average_cpu_percent': self.average_cpu_percent,
            'peak_cpu_percent': self.peak_cpu_percent,
            'disk_read_mb': self.disk_read_mb,
            'disk_write_mb': self.disk_write_mb,
            'network_sent_mb': self.network_sent_mb,
            'network_received_mb': self.network_received_mb,
            'api_calls_count': self.api_calls_count,
            'tokens_used': self.tokens_used,
            'cache_hit_rate': self.get_cache_hit_rate(),
            'error_rate': self.get_error_rate(),
            'retry_count': self.retry_count,
            'error_count': self.error_count,
            'custom_metrics': self.custom_metrics,
            'snapshot_count': len(self.resource_snapshots)
        }


class PerformanceTracker:
    """
    Comprehensive performance tracking system for agent execution.
    
    Monitors resource usage, timing metrics, and generates performance alerts
    when thresholds are exceeded.
    """
    
    def __init__(self, sampling_interval: float = 1.0):
        """
        Initialize performance tracker.
        
        Args:
            sampling_interval: Interval in seconds between resource samples
        """
        self.sampling_interval = sampling_interval
        
        # Active tracking
        self.active_sessions: Dict[str, AgentPerformanceData] = {}
        self.monitoring_threads: Dict[str, threading.Thread] = {}
        self.stop_events: Dict[str, threading.Event] = {}
        
        # Historical data
        self.performance_history: Dict[str, List[AgentPerformanceData]] = defaultdict(list)
        self.alerts_history: deque = deque(maxlen=1000)
        
        # Performance thresholds
        self.thresholds = {
            'max_execution_time': 600.0,      # 10 minutes
            'max_processing_time': 300.0,     # 5 minutes
            'max_memory_mb': 2048.0,          # 2GB
            'max_cpu_percent': 80.0,          # 80%
            'max_error_rate': 0.1,            # 10%
            'min_cache_hit_rate': 0.5,        # 50%
            'max_retry_count': 5
        }
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        
        # System baseline
        self.system_baseline = self._get_system_baseline()
        
        logger.info("PerformanceTracker initialized")
    
    def start_tracking(self, agent_name: str, session_id: str) -> AgentPerformanceData:
        """Start performance tracking for an agent session."""
        if session_id in self.active_sessions:
            logger.warning(f"Performance tracking already active for session {session_id}")
            return self.active_sessions[session_id]
        
        # Create performance data container
        perf_data = AgentPerformanceData(
            agent_name=agent_name,
            session_id=session_id,
            start_time=datetime.now()
        )
        
        self.active_sessions[session_id] = perf_data
        
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
        
        logger.info(f"Started performance tracking for {agent_name} (session: {session_id})")
        return perf_data
    
    def stop_tracking(self, session_id: str) -> Optional[AgentPerformanceData]:
        """Stop performance tracking for a session."""
        if session_id not in self.active_sessions:
            logger.warning(f"No active performance tracking for session {session_id}")
            return None
        
        # Stop monitoring thread
        if session_id in self.stop_events:
            self.stop_events[session_id].set()
        
        # Wait for thread to finish
        if session_id in self.monitoring_threads:
            self.monitoring_threads[session_id].join(timeout=2.0)
            del self.monitoring_threads[session_id]
        
        # Finalize performance data
        perf_data = self.active_sessions[session_id]
        perf_data.finalize_metrics()
        
        # Add to history
        self.performance_history[perf_data.agent_name].append(perf_data)
        
        # Check final thresholds
        self._check_final_thresholds(perf_data)
        
        # Cleanup
        del self.active_sessions[session_id]
        if session_id in self.stop_events:
            del self.stop_events[session_id]
        
        logger.info(f"Stopped performance tracking for session {session_id}")
        return perf_data
    
    def record_step_time(self, session_id: str, step_name: str, duration: float) -> None:
        """Record timing for a specific processing step."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].step_times[step_name] = duration
            self.active_sessions[session_id].processing_time += duration
            
            # Check step time threshold
            if duration > 60.0:  # 1 minute threshold for individual steps
                self._create_alert(
                    session_id,
                    "slow_step",
                    PerformanceAlertLevel.WARNING,
                    f"Step '{step_name}' took {duration:.2f}s (threshold: 60s)",
                    {'step_name': step_name, 'duration': duration},
                    threshold_value=60.0,
                    actual_value=duration
                )
    
    def record_api_call(self, session_id: str, tokens_used: int = 0) -> None:
        """Record an API call and token usage."""
        if session_id in self.active_sessions:
            perf_data = self.active_sessions[session_id]
            perf_data.api_calls_count += 1
            perf_data.tokens_used += tokens_used
    
    def record_cache_hit(self, session_id: str) -> None:
        """Record a cache hit."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].cache_hits += 1
    
    def record_cache_miss(self, session_id: str) -> None:
        """Record a cache miss."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].cache_misses += 1
    
    def record_retry(self, session_id: str) -> None:
        """Record a retry attempt."""
        if session_id in self.active_sessions:
            perf_data = self.active_sessions[session_id]
            perf_data.retry_count += 1
            
            # Check retry threshold
            if perf_data.retry_count > self.thresholds['max_retry_count']:
                self._create_alert(
                    session_id,
                    "excessive_retries",
                    PerformanceAlertLevel.CRITICAL,
                    f"Retry count ({perf_data.retry_count}) exceeds threshold",
                    {'retry_count': perf_data.retry_count},
                    threshold_value=self.thresholds['max_retry_count'],
                    actual_value=perf_data.retry_count
                )
    
    def record_error(self, session_id: str) -> None:
        """Record an error occurrence."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].error_count += 1
    
    def add_custom_metric(self, session_id: str, metric_name: str, value: Any) -> None:
        """Add a custom performance metric."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].custom_metrics[metric_name] = value
    
    def get_current_performance(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current performance data for a session."""
        if session_id not in self.active_sessions:
            return None
        
        perf_data = self.active_sessions[session_id]
        current_time = datetime.now()
        current_duration = (current_time - perf_data.start_time).total_seconds()
        
        return {
            'agent_name': perf_data.agent_name,
            'session_id': session_id,
            'current_duration': current_duration,
            'processing_time': perf_data.processing_time,
            'api_calls_count': perf_data.api_calls_count,
            'tokens_used': perf_data.tokens_used,
            'cache_hit_rate': perf_data.get_cache_hit_rate(),
            'error_count': perf_data.error_count,
            'retry_count': perf_data.retry_count,
            'peak_memory_mb': perf_data.peak_memory_mb,
            'peak_cpu_percent': perf_data.peak_cpu_percent,
            'step_count': len(perf_data.step_times),
            'custom_metrics': perf_data.custom_metrics
        }
    
    def get_agent_performance_history(self, agent_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get performance history for a specific agent."""
        if agent_name not in self.performance_history:
            return []
        
        history = self.performance_history[agent_name]
        recent_history = history[-limit:] if len(history) > limit else history
        
        return [perf_data.to_dict() for perf_data in recent_history]
    
    def get_performance_summary(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary across all agents or for a specific agent."""
        if agent_name:
            if agent_name not in self.performance_history:
                return {'error': f'No performance data for agent {agent_name}'}
            
            history = self.performance_history[agent_name]
            if not history:
                return {'agent_name': agent_name, 'executions': 0}
            
            execution_times = [p.total_execution_time for p in history]
            memory_usage = [p.peak_memory_mb for p in history]
            cpu_usage = [p.average_cpu_percent for p in history]
            
            return {
                'agent_name': agent_name,
                'executions': len(history),
                'avg_execution_time': sum(execution_times) / len(execution_times),
                'min_execution_time': min(execution_times),
                'max_execution_time': max(execution_times),
                'avg_memory_mb': sum(memory_usage) / len(memory_usage),
                'peak_memory_mb': max(memory_usage),
                'avg_cpu_percent': sum(cpu_usage) / len(cpu_usage),
                'peak_cpu_percent': max(cpu_usage),
                'total_api_calls': sum(p.api_calls_count for p in history),
                'total_tokens': sum(p.tokens_used for p in history),
                'total_errors': sum(p.error_count for p in history),
                'avg_cache_hit_rate': sum(p.get_cache_hit_rate() for p in history) / len(history)
            }
        else:
            # Summary across all agents
            all_history = []
            for agent_history in self.performance_history.values():
                all_history.extend(agent_history)
            
            if not all_history:
                return {'total_executions': 0}
            
            return {
                'total_executions': len(all_history),
                'agents_tracked': len(self.performance_history),
                'active_sessions': len(self.active_sessions),
                'total_alerts': len(self.alerts_history),
                'avg_execution_time': sum(p.total_execution_time for p in all_history) / len(all_history),
                'total_api_calls': sum(p.api_calls_count for p in all_history),
                'total_tokens': sum(p.tokens_used for p in all_history),
                'total_errors': sum(p.error_count for p in all_history)
            }
    
    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent performance alerts."""
        alerts = list(self.alerts_history)
        recent_alerts = alerts[-limit:] if len(alerts) > limit else alerts
        return [alert.to_dict() for alert in recent_alerts]
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """Add a callback for performance alerts."""
        self.alert_callbacks.append(callback)
    
    def update_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """Update performance thresholds."""
        self.thresholds.update(new_thresholds)
        logger.info(f"Updated performance thresholds: {new_thresholds}")
    
    def _monitoring_loop(self, session_id: str, stop_event: threading.Event) -> None:
        """Main monitoring loop for collecting resource usage."""
        try:
            while not stop_event.is_set():
                try:
                    if session_id in self.active_sessions:
                        snapshot = self._take_resource_snapshot()
                        self.active_sessions[session_id].add_resource_snapshot(snapshot)
                        
                        # Check real-time thresholds
                        self._check_realtime_thresholds(session_id, snapshot)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop for session {session_id}: {e}")
                
                # Wait for next sampling interval
                stop_event.wait(self.sampling_interval)
                
        except Exception as e:
            logger.error(f"Monitoring loop failed for session {session_id}: {e}")
    
    def _take_resource_snapshot(self) -> Dict[str, Any]:
        """Take a snapshot of current resource usage."""
        try:
            process = psutil.Process()
            
            # Memory usage
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            # CPU usage
            cpu_percent = process.cpu_percent()
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            
            # Network I/O
            network_io = psutil.net_io_counters()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'memory_mb': memory_mb,
                'cpu_percent': cpu_percent,
                'disk_read_mb': disk_io.read_bytes / (1024 * 1024) if disk_io else 0,
                'disk_write_mb': disk_io.write_bytes / (1024 * 1024) if disk_io else 0,
                'network_sent_mb': network_io.bytes_sent / (1024 * 1024) if network_io else 0,
                'network_recv_mb': network_io.bytes_recv / (1024 * 1024) if network_io else 0,
                'thread_count': process.num_threads()
            }
            
        except Exception as e:
            logger.error(f"Error taking resource snapshot: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'memory_mb': 0,
                'cpu_percent': 0,
                'disk_read_mb': 0,
                'disk_write_mb': 0,
                'network_sent_mb': 0,
                'network_recv_mb': 0,
                'thread_count': 0
            }
    
    def _check_realtime_thresholds(self, session_id: str, snapshot: Dict[str, Any]) -> None:
        """Check real-time performance thresholds."""
        # Memory threshold
        if snapshot['memory_mb'] > self.thresholds['max_memory_mb']:
            self._create_alert(
                session_id,
                "high_memory_usage",
                PerformanceAlertLevel.WARNING,
                f"Memory usage ({snapshot['memory_mb']:.1f}MB) exceeds threshold",
                snapshot,
                threshold_value=self.thresholds['max_memory_mb'],
                actual_value=snapshot['memory_mb']
            )
        
        # CPU threshold
        if snapshot['cpu_percent'] > self.thresholds['max_cpu_percent']:
            self._create_alert(
                session_id,
                "high_cpu_usage",
                PerformanceAlertLevel.WARNING,
                f"CPU usage ({snapshot['cpu_percent']:.1f}%) exceeds threshold",
                snapshot,
                threshold_value=self.thresholds['max_cpu_percent'],
                actual_value=snapshot['cpu_percent']
            )
    
    def _check_final_thresholds(self, perf_data: AgentPerformanceData) -> None:
        """Check performance thresholds at the end of execution."""
        # Execution time threshold
        if perf_data.total_execution_time > self.thresholds['max_execution_time']:
            self._create_alert(
                perf_data.session_id,
                "long_execution",
                PerformanceAlertLevel.CRITICAL,
                f"Execution time ({perf_data.total_execution_time:.1f}s) exceeds threshold",
                {'total_execution_time': perf_data.total_execution_time},
                threshold_value=self.thresholds['max_execution_time'],
                actual_value=perf_data.total_execution_time
            )
        
        # Error rate threshold
        error_rate = perf_data.get_error_rate()
        if error_rate > self.thresholds['max_error_rate']:
            self._create_alert(
                perf_data.session_id,
                "high_error_rate",
                PerformanceAlertLevel.CRITICAL,
                f"Error rate ({error_rate:.2%}) exceeds threshold",
                {'error_rate': error_rate, 'error_count': perf_data.error_count},
                threshold_value=self.thresholds['max_error_rate'],
                actual_value=error_rate
            )
        
        # Cache hit rate threshold
        cache_hit_rate = perf_data.get_cache_hit_rate()
        if cache_hit_rate < self.thresholds['min_cache_hit_rate'] and (perf_data.cache_hits + perf_data.cache_misses) > 0:
            self._create_alert(
                perf_data.session_id,
                "low_cache_hit_rate",
                PerformanceAlertLevel.WARNING,
                f"Cache hit rate ({cache_hit_rate:.2%}) below threshold",
                {'cache_hit_rate': cache_hit_rate},
                threshold_value=self.thresholds['min_cache_hit_rate'],
                actual_value=cache_hit_rate
            )
    
    def _create_alert(self, session_id: str, alert_type: str, level: PerformanceAlertLevel,
                     message: str, metrics: Dict[str, Any],
                     threshold_value: Optional[float] = None,
                     actual_value: Optional[float] = None) -> None:
        """Create and process a performance alert."""
        if session_id not in self.active_sessions:
            return
        
        perf_data = self.active_sessions[session_id]
        
        alert = PerformanceAlert(
            alert_id=f"{alert_type}_{session_id}_{int(time.time())}",
            timestamp=datetime.now(),
            agent_name=perf_data.agent_name,
            session_id=session_id,
            alert_type=alert_type,
            level=level,
            message=message,
            metrics=metrics,
            threshold_value=threshold_value,
            actual_value=actual_value
        )
        
        self.alerts_history.append(alert)
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        logger.warning(f"Performance alert: {alert.message}")
    
    def _get_system_baseline(self) -> Dict[str, Any]:
        """Get system baseline metrics."""
        try:
            return {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'disk_total_gb': psutil.disk_usage('/').total / (1024**3) if psutil.disk_usage('/') else 0
            }
        except Exception as e:
            logger.error(f"Error getting system baseline: {e}")
            return {}


# Global instance
_global_performance_tracker = PerformanceTracker()


def get_performance_tracker() -> PerformanceTracker:
    """Get the global performance tracker instance."""
    return _global_performance_tracker
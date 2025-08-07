"""
Real-time agent execution monitoring system.

This module provides comprehensive monitoring of agent execution status,
processing times, and state transitions with real-time visualization support.
"""

import asyncio
import logging
import time
import threading
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import uuid

from ..models.state import VideoGenerationState
from ..models.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Agent execution status enumeration."""
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    WAITING = "waiting"
    PROCESSING = "processing"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ExecutionEvent:
    """Represents a single execution event."""
    
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    agent_name: str = ""
    session_id: str = ""
    event_type: str = ""
    status: ExecutionStatus = ExecutionStatus.IDLE
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'agent_name': self.agent_name,
            'session_id': self.session_id,
            'event_type': self.event_type,
            'status': self.status.value,
            'message': self.message,
            'data': self.data,
            'duration_ms': self.duration_ms,
            'error': self.error
        }


@dataclass
class AgentExecutionTracker:
    """Tracks execution state for a single agent session."""
    
    agent_name: str
    session_id: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    current_status: ExecutionStatus = ExecutionStatus.IDLE
    
    # Event tracking
    events: List[ExecutionEvent] = field(default_factory=list)
    status_history: List[tuple[ExecutionStatus, datetime]] = field(default_factory=list)
    
    # Performance tracking
    processing_start_time: Optional[datetime] = None
    total_processing_time: float = 0.0
    step_times: Dict[str, float] = field(default_factory=dict)
    
    # State tracking
    current_step: str = ""
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    intermediate_states: List[Dict[str, Any]] = field(default_factory=list)
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    retry_count: int = 0
    
    def add_event(self, event: ExecutionEvent) -> None:
        """Add an execution event."""
        self.events.append(event)
        
        # Update status if provided
        if event.status != ExecutionStatus.IDLE:
            self.update_status(event.status)
    
    def update_status(self, status: ExecutionStatus) -> None:
        """Update the current execution status."""
        if self.current_status != status:
            self.status_history.append((self.current_status, datetime.now()))
            self.current_status = status
            
            # Handle processing time tracking
            if status == ExecutionStatus.PROCESSING:
                self.processing_start_time = datetime.now()
            elif status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED]:
                if self.processing_start_time:
                    processing_duration = (datetime.now() - self.processing_start_time).total_seconds()
                    self.total_processing_time += processing_duration
                    self.processing_start_time = None
                
                if self.end_time is None:
                    self.end_time = datetime.now()
    
    def add_step_time(self, step_name: str, duration: float) -> None:
        """Add timing for a specific step."""
        self.step_times[step_name] = duration
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        logger.error(f"Agent {self.agent_name} session {self.session_id}: {error}")
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
        logger.warning(f"Agent {self.agent_name} session {self.session_id}: {warning}")
    
    def increment_retry_count(self) -> None:
        """Increment the retry counter."""
        self.retry_count += 1
    
    def get_execution_duration(self) -> Optional[float]:
        """Get total execution duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()
    
    def get_status_duration(self, status: ExecutionStatus) -> float:
        """Get total time spent in a specific status."""
        total_duration = 0.0
        current_status = None
        current_start = self.start_time
        
        for status_change, timestamp in self.status_history:
            if current_status == status:
                total_duration += (timestamp - current_start).total_seconds()
            current_status = status_change
            current_start = timestamp
        
        # Check if currently in the requested status
        if self.current_status == status:
            total_duration += (datetime.now() - current_start).total_seconds()
        
        return total_duration
    
    def is_active(self) -> bool:
        """Check if the agent execution is currently active."""
        return self.current_status in [
            ExecutionStatus.STARTING,
            ExecutionStatus.RUNNING,
            ExecutionStatus.WAITING,
            ExecutionStatus.PROCESSING,
            ExecutionStatus.COMPLETING
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tracker to dictionary for serialization."""
        return {
            'agent_name': self.agent_name,
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'current_status': self.current_status.value,
            'execution_duration': self.get_execution_duration(),
            'total_processing_time': self.total_processing_time,
            'current_step': self.current_step,
            'step_times': self.step_times,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'retry_count': self.retry_count,
            'event_count': len(self.events),
            'is_active': self.is_active()
        }


class ExecutionMonitor:
    """
    Real-time agent execution monitoring system.
    
    Provides comprehensive monitoring of agent execution with real-time status
    tracking, performance monitoring, and event logging.
    """
    
    def __init__(self, max_history_size: int = 1000):
        """
        Initialize execution monitor.
        
        Args:
            max_history_size: Maximum number of events to keep in history
        """
        self.max_history_size = max_history_size
        
        # Active tracking
        self.active_trackers: Dict[str, AgentExecutionTracker] = {}
        self.session_to_tracker: Dict[str, str] = {}
        
        # Event history
        self.event_history: deque = deque(maxlen=max_history_size)
        self.agent_events: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Monitoring callbacks
        self.status_callbacks: List[Callable[[AgentExecutionTracker], None]] = []
        self.event_callbacks: List[Callable[[ExecutionEvent], None]] = []
        self.error_callbacks: List[Callable[[str, str, str], None]] = []
        
        # Performance thresholds
        self.performance_thresholds = {
            'max_execution_time': 600.0,  # 10 minutes
            'max_processing_time': 300.0,  # 5 minutes
            'max_step_time': 60.0,         # 1 minute
            'max_retry_count': 3
        }
        
        # Background monitoring
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        logger.info("ExecutionMonitor initialized")
    
    def start_monitoring(self) -> None:
        """Start background monitoring thread."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._stop_event.clear()
        
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        
        logger.info("Background monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring thread."""
        if not self._monitoring_active:
            return
        
        self._monitoring_active = False
        self._stop_event.set()
        
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
            self._monitoring_thread = None
        
        logger.info("Background monitoring stopped")
    
    def start_agent_execution(self, agent_name: str, session_id: str, 
                            input_data: Optional[Dict[str, Any]] = None) -> AgentExecutionTracker:
        """Start monitoring an agent execution."""
        tracker_id = f"{agent_name}_{session_id}"
        
        # Create new tracker
        tracker = AgentExecutionTracker(
            agent_name=agent_name,
            session_id=session_id,
            input_data=input_data or {}
        )
        
        self.active_trackers[tracker_id] = tracker
        self.session_to_tracker[session_id] = tracker_id
        
        # Create start event
        start_event = ExecutionEvent(
            agent_name=agent_name,
            session_id=session_id,
            event_type="execution_started",
            status=ExecutionStatus.STARTING,
            message=f"Started monitoring execution for {agent_name}",
            data=input_data or {}
        )
        
        self._add_event(start_event)
        tracker.add_event(start_event)
        tracker.update_status(ExecutionStatus.STARTING)
        
        # Notify callbacks
        self._notify_status_callbacks(tracker)
        
        logger.info(f"Started monitoring agent execution: {agent_name} (session: {session_id})")
        return tracker
    
    def stop_agent_execution(self, session_id: str, status: ExecutionStatus = ExecutionStatus.COMPLETED,
                           output_data: Optional[Dict[str, Any]] = None) -> Optional[AgentExecutionTracker]:
        """Stop monitoring an agent execution."""
        tracker_id = self.session_to_tracker.get(session_id)
        if not tracker_id or tracker_id not in self.active_trackers:
            logger.warning(f"No active tracker found for session {session_id}")
            return None
        
        tracker = self.active_trackers[tracker_id]
        tracker.output_data = output_data or {}
        tracker.update_status(status)
        
        # Create stop event
        stop_event = ExecutionEvent(
            agent_name=tracker.agent_name,
            session_id=session_id,
            event_type="execution_stopped",
            status=status,
            message=f"Stopped monitoring execution for {tracker.agent_name}",
            data=output_data or {},
            duration_ms=tracker.get_execution_duration() * 1000 if tracker.get_execution_duration() else None
        )
        
        self._add_event(stop_event)
        tracker.add_event(stop_event)
        
        # Notify callbacks
        self._notify_status_callbacks(tracker)
        
        # Remove from active tracking
        del self.active_trackers[tracker_id]
        del self.session_to_tracker[session_id]
        
        logger.info(f"Stopped monitoring agent execution: {tracker.agent_name} (session: {session_id})")
        return tracker
    
    def update_agent_status(self, session_id: str, status: ExecutionStatus, 
                          message: str = "", data: Optional[Dict[str, Any]] = None) -> None:
        """Update agent execution status."""
        tracker_id = self.session_to_tracker.get(session_id)
        if not tracker_id or tracker_id not in self.active_trackers:
            logger.warning(f"No active tracker found for session {session_id}")
            return
        
        tracker = self.active_trackers[tracker_id]
        
        # Create status update event
        status_event = ExecutionEvent(
            agent_name=tracker.agent_name,
            session_id=session_id,
            event_type="status_update",
            status=status,
            message=message,
            data=data or {}
        )
        
        self._add_event(status_event)
        tracker.add_event(status_event)
        tracker.update_status(status)
        
        # Notify callbacks
        self._notify_status_callbacks(tracker)
    
    def update_agent_step(self, session_id: str, step_name: str, 
                         data: Optional[Dict[str, Any]] = None) -> None:
        """Update current agent processing step."""
        tracker_id = self.session_to_tracker.get(session_id)
        if not tracker_id or tracker_id not in self.active_trackers:
            return
        
        tracker = self.active_trackers[tracker_id]
        tracker.current_step = step_name
        
        # Add intermediate state
        if data:
            state_snapshot = {
                'step': step_name,
                'timestamp': datetime.now().isoformat(),
                'data': data
            }
            tracker.intermediate_states.append(state_snapshot)
        
        # Create step event
        step_event = ExecutionEvent(
            agent_name=tracker.agent_name,
            session_id=session_id,
            event_type="step_update",
            status=tracker.current_status,
            message=f"Processing step: {step_name}",
            data=data or {}
        )
        
        self._add_event(step_event)
        tracker.add_event(step_event)
    
    def record_step_completion(self, session_id: str, step_name: str, 
                             duration: float, success: bool = True,
                             data: Optional[Dict[str, Any]] = None) -> None:
        """Record completion of a processing step."""
        tracker_id = self.session_to_tracker.get(session_id)
        if not tracker_id or tracker_id not in self.active_trackers:
            return
        
        tracker = self.active_trackers[tracker_id]
        tracker.add_step_time(step_name, duration)
        
        # Create completion event
        completion_event = ExecutionEvent(
            agent_name=tracker.agent_name,
            session_id=session_id,
            event_type="step_completed",
            status=tracker.current_status,
            message=f"Completed step: {step_name} ({'success' if success else 'failed'})",
            data=data or {},
            duration_ms=duration * 1000
        )
        
        self._add_event(completion_event)
        tracker.add_event(completion_event)
        
        # Check performance thresholds
        if duration > self.performance_thresholds['max_step_time']:
            self._trigger_performance_alert(tracker, 'slow_step', {
                'step_name': step_name,
                'duration': duration,
                'threshold': self.performance_thresholds['max_step_time']
            })
    
    def record_agent_error(self, session_id: str, error: str, 
                         error_type: str = "general", data: Optional[Dict[str, Any]] = None) -> None:
        """Record an agent execution error."""
        tracker_id = self.session_to_tracker.get(session_id)
        if not tracker_id or tracker_id not in self.active_trackers:
            return
        
        tracker = self.active_trackers[tracker_id]
        tracker.add_error(error)
        
        # Create error event
        error_event = ExecutionEvent(
            agent_name=tracker.agent_name,
            session_id=session_id,
            event_type="error",
            status=tracker.current_status,
            message=f"Error: {error}",
            data=data or {},
            error=error
        )
        
        self._add_event(error_event)
        tracker.add_event(error_event)
        
        # Notify error callbacks
        for callback in self.error_callbacks:
            try:
                callback(tracker.agent_name, session_id, error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
    
    def record_agent_retry(self, session_id: str, reason: str = "") -> None:
        """Record an agent retry attempt."""
        tracker_id = self.session_to_tracker.get(session_id)
        if not tracker_id or tracker_id not in self.active_trackers:
            return
        
        tracker = self.active_trackers[tracker_id]
        tracker.increment_retry_count()
        
        # Create retry event
        retry_event = ExecutionEvent(
            agent_name=tracker.agent_name,
            session_id=session_id,
            event_type="retry",
            status=tracker.current_status,
            message=f"Retry attempt #{tracker.retry_count}: {reason}",
            data={'retry_count': tracker.retry_count, 'reason': reason}
        )
        
        self._add_event(retry_event)
        tracker.add_event(retry_event)
        
        # Check retry threshold
        if tracker.retry_count > self.performance_thresholds['max_retry_count']:
            self._trigger_performance_alert(tracker, 'excessive_retries', {
                'retry_count': tracker.retry_count,
                'threshold': self.performance_thresholds['max_retry_count']
            })
    
    def get_active_executions(self) -> List[AgentExecutionTracker]:
        """Get all currently active agent executions."""
        return list(self.active_trackers.values())
    
    def get_execution_tracker(self, session_id: str) -> Optional[AgentExecutionTracker]:
        """Get execution tracker for a specific session."""
        tracker_id = self.session_to_tracker.get(session_id)
        if tracker_id and tracker_id in self.active_trackers:
            return self.active_trackers[tracker_id]
        return None
    
    def get_agent_events(self, agent_name: str, limit: int = 50) -> List[ExecutionEvent]:
        """Get recent events for a specific agent."""
        if agent_name not in self.agent_events:
            return []
        
        events = list(self.agent_events[agent_name])
        return events[-limit:] if len(events) > limit else events
    
    def get_recent_events(self, limit: int = 100) -> List[ExecutionEvent]:
        """Get recent events across all agents."""
        events = list(self.event_history)
        return events[-limit:] if len(events) > limit else events
    
    def add_status_callback(self, callback: Callable[[AgentExecutionTracker], None]) -> None:
        """Add a callback for status updates."""
        self.status_callbacks.append(callback)
    
    def add_event_callback(self, callback: Callable[[ExecutionEvent], None]) -> None:
        """Add a callback for events."""
        self.event_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[str, str, str], None]) -> None:
        """Add a callback for errors (agent_name, session_id, error)."""
        self.error_callbacks.append(callback)
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get a summary of current monitoring status."""
        active_agents = {}
        for tracker in self.active_trackers.values():
            agent_name = tracker.agent_name
            if agent_name not in active_agents:
                active_agents[agent_name] = []
            active_agents[agent_name].append({
                'session_id': tracker.session_id,
                'status': tracker.current_status.value,
                'duration': tracker.get_execution_duration(),
                'current_step': tracker.current_step
            })
        
        return {
            'active_executions': len(self.active_trackers),
            'active_agents': active_agents,
            'total_events': len(self.event_history),
            'monitoring_active': self._monitoring_active,
            'performance_thresholds': self.performance_thresholds
        }
    
    def _add_event(self, event: ExecutionEvent) -> None:
        """Add event to history and notify callbacks."""
        self.event_history.append(event)
        self.agent_events[event.agent_name].append(event)
        
        # Notify event callbacks
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")
    
    def _notify_status_callbacks(self, tracker: AgentExecutionTracker) -> None:
        """Notify all status callbacks."""
        for callback in self.status_callbacks:
            try:
                callback(tracker)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")
    
    def _trigger_performance_alert(self, tracker: AgentExecutionTracker, 
                                 alert_type: str, data: Dict[str, Any]) -> None:
        """Trigger a performance alert."""
        alert_event = ExecutionEvent(
            agent_name=tracker.agent_name,
            session_id=tracker.session_id,
            event_type="performance_alert",
            status=tracker.current_status,
            message=f"Performance alert: {alert_type}",
            data=data
        )
        
        self._add_event(alert_event)
        tracker.add_event(alert_event)
        
        logger.warning(f"Performance alert for {tracker.agent_name}: {alert_type} - {data}")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop for checking thresholds and cleanup."""
        while not self._stop_event.is_set():
            try:
                current_time = datetime.now()
                
                # Check for long-running executions
                for tracker in list(self.active_trackers.values()):
                    execution_duration = tracker.get_execution_duration()
                    
                    if execution_duration and execution_duration > self.performance_thresholds['max_execution_time']:
                        self._trigger_performance_alert(tracker, 'long_execution', {
                            'duration': execution_duration,
                            'threshold': self.performance_thresholds['max_execution_time']
                        })
                    
                    # Check for stuck executions (no events in last 5 minutes)
                    if tracker.events:
                        last_event_time = tracker.events[-1].timestamp
                        if (current_time - last_event_time).total_seconds() > 300:
                            self._trigger_performance_alert(tracker, 'stuck_execution', {
                                'last_event_time': last_event_time.isoformat(),
                                'minutes_since_last_event': (current_time - last_event_time).total_seconds() / 60
                            })
                
                # Sleep for monitoring interval
                self._stop_event.wait(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                self._stop_event.wait(30)


# Global instance
_global_execution_monitor = ExecutionMonitor()


def get_execution_monitor() -> ExecutionMonitor:
    """Get the global execution monitor instance."""
    return _global_execution_monitor
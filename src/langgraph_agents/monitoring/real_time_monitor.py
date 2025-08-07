"""
Real-time agent execution monitoring and visualization system.

This module provides a comprehensive real-time monitoring interface that
integrates execution monitoring, performance tracking, error detection,
and execution history for complete agent oversight.
"""

import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import websockets
from websockets.server import WebSocketServerProtocol

from .execution_monitor import get_execution_monitor, ExecutionStatus, AgentExecutionTracker
from .performance_tracker import get_performance_tracker, PerformanceAlert
from .error_detector import get_error_detector, ErrorAlert
from .execution_history import get_execution_history

logger = logging.getLogger(__name__)


@dataclass
class MonitoringDashboard:
    """Real-time monitoring dashboard data."""
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Active executions
    active_executions: List[Dict[str, Any]] = field(default_factory=list)
    execution_count_by_status: Dict[str, int] = field(default_factory=dict)
    execution_count_by_agent: Dict[str, int] = field(default_factory=dict)
    
    # Performance metrics
    avg_execution_time: float = 0.0
    avg_processing_time: float = 0.0
    peak_memory_usage: float = 0.0
    avg_cpu_usage: float = 0.0
    
    # Error statistics
    error_count_last_hour: int = 0
    error_rate: float = 0.0
    critical_alerts: int = 0
    
    # System health
    system_health_score: float = 100.0
    health_indicators: Dict[str, str] = field(default_factory=dict)
    
    # Recent events
    recent_events: List[Dict[str, Any]] = field(default_factory=list)
    recent_alerts: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dashboard to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'active_executions': self.active_executions,
            'execution_count_by_status': self.execution_count_by_status,
            'execution_count_by_agent': self.execution_count_by_agent,
            'avg_execution_time': self.avg_execution_time,
            'avg_processing_time': self.avg_processing_time,
            'peak_memory_usage': self.peak_memory_usage,
            'avg_cpu_usage': self.avg_cpu_usage,
            'error_count_last_hour': self.error_count_last_hour,
            'error_rate': self.error_rate,
            'critical_alerts': self.critical_alerts,
            'system_health_score': self.system_health_score,
            'health_indicators': self.health_indicators,
            'recent_events': self.recent_events,
            'recent_alerts': self.recent_alerts
        }


class RealTimeMonitor:
    """
    Comprehensive real-time monitoring system for agent execution.
    
    Provides real-time dashboard updates, WebSocket streaming, alerting,
    and visualization capabilities for complete agent oversight.
    """
    
    def __init__(self, websocket_port: int = 8765, update_interval: float = 1.0):
        """
        Initialize real-time monitor.
        
        Args:
            websocket_port: Port for WebSocket server
            update_interval: Dashboard update interval in seconds
        """
        self.websocket_port = websocket_port
        self.update_interval = update_interval
        
        # Component references
        self.execution_monitor = get_execution_monitor()
        self.performance_tracker = get_performance_tracker()
        self.error_detector = get_error_detector()
        self.execution_history = get_execution_history()
        
        # WebSocket connections
        self.websocket_clients: Set[WebSocketServerProtocol] = set()
        self.websocket_server = None
        
        # Dashboard state
        self.current_dashboard = MonitoringDashboard()
        self.dashboard_history: deque = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        
        # Monitoring state
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Alert thresholds
        self.alert_thresholds = {
            'max_active_executions': 50,
            'max_avg_execution_time': 300.0,  # 5 minutes
            'max_error_rate': 0.1,  # 10%
            'min_health_score': 80.0
        }
        
        # Event subscribers
        self.event_subscribers: List[Callable[[Dict[str, Any]], None]] = []
        
        # Setup callbacks
        self._setup_callbacks()
        
        logger.info(f"RealTimeMonitor initialized on port {websocket_port}")
    
    async def start_monitoring(self) -> None:
        """Start real-time monitoring with WebSocket server."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._stop_event.clear()
        
        # Start background monitoring thread
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        
        # Start WebSocket server
        await self._start_websocket_server()
        
        # Start component monitoring
        self.execution_monitor.start_monitoring()
        
        logger.info("Real-time monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        if not self._monitoring_active:
            return
        
        self._monitoring_active = False
        self._stop_event.set()
        
        # Stop monitoring thread
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
            self._monitoring_thread = None
        
        # Stop WebSocket server
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
            self.websocket_server = None
        
        # Stop component monitoring
        self.execution_monitor.stop_monitoring()
        
        logger.info("Real-time monitoring stopped")
    
    def get_current_dashboard(self) -> Dict[str, Any]:
        """Get current dashboard state."""
        return self.current_dashboard.to_dict()
    
    def get_dashboard_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get dashboard history for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        history = []
        for dashboard in self.dashboard_history:
            if dashboard.timestamp >= cutoff_time:
                history.append(dashboard.to_dict())
        
        return sorted(history, key=lambda d: d['timestamp'])
    
    def get_agent_status(self, agent_name: str) -> Dict[str, Any]:
        """Get detailed status for a specific agent."""
        # Get active executions for agent
        active_executions = [
            tracker for tracker in self.execution_monitor.get_active_executions()
            if tracker.agent_name == agent_name
        ]
        
        # Get performance history
        performance_history = self.performance_tracker.get_agent_performance_history(agent_name, limit=10)
        
        # Get error statistics
        error_stats = self.error_detector.get_error_statistics(agent_name, time_window_hours=24)
        
        # Get execution history
        execution_records = self.execution_history.get_agent_records(agent_name, limit=20)
        
        return {
            'agent_name': agent_name,
            'active_executions': len(active_executions),
            'active_sessions': [
                {
                    'session_id': tracker.session_id,
                    'status': tracker.current_status.value,
                    'duration': tracker.get_execution_duration(),
                    'current_step': tracker.current_step
                }
                for tracker in active_executions
            ],
            'performance_history': performance_history,
            'error_statistics': error_stats,
            'recent_executions': [record.to_dict() for record in execution_records[:5]]
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        # Calculate health indicators
        health_indicators = {}
        health_score = 100.0
        
        # Check active executions
        active_count = len(self.execution_monitor.get_active_executions())
        if active_count > self.alert_thresholds['max_active_executions']:
            health_indicators['executions'] = 'warning'
            health_score -= 10
        else:
            health_indicators['executions'] = 'healthy'
        
        # Check average execution time
        performance_summary = self.performance_tracker.get_performance_summary()
        avg_exec_time = performance_summary.get('avg_execution_time', 0)
        if avg_exec_time > self.alert_thresholds['max_avg_execution_time']:
            health_indicators['performance'] = 'warning'
            health_score -= 15
        else:
            health_indicators['performance'] = 'healthy'
        
        # Check error rate
        error_stats = self.error_detector.get_error_statistics(time_window_hours=1)
        error_rate = error_stats.get('error_rate_per_hour', 0) / 100  # Convert to rate
        if error_rate > self.alert_thresholds['max_error_rate']:
            health_indicators['errors'] = 'critical'
            health_score -= 25
        elif error_rate > self.alert_thresholds['max_error_rate'] / 2:
            health_indicators['errors'] = 'warning'
            health_score -= 10
        else:
            health_indicators['errors'] = 'healthy'
        
        # Check recent alerts
        recent_alerts = self.error_detector.get_recent_alerts(limit=10)
        critical_alerts = len([a for a in recent_alerts if a['severity'] == 'critical'])
        if critical_alerts > 0:
            health_indicators['alerts'] = 'critical'
            health_score -= 20
        else:
            health_indicators['alerts'] = 'healthy'
        
        return {
            'health_score': max(0, health_score),
            'health_indicators': health_indicators,
            'active_executions': active_count,
            'avg_execution_time': avg_exec_time,
            'error_rate': error_rate,
            'critical_alerts': critical_alerts,
            'timestamp': datetime.now().isoformat()
        }
    
    def add_event_subscriber(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add a callback for real-time events."""
        self.event_subscribers.append(callback)
    
    def broadcast_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcast an event to all subscribers and WebSocket clients."""
        event = {
            'type': event_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        # Notify subscribers
        for callback in self.event_subscribers:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event subscriber: {e}")
        
        # Broadcast to WebSocket clients
        asyncio.create_task(self._broadcast_to_websockets(event))
    
    def _setup_callbacks(self) -> None:
        """Setup callbacks for monitoring components."""
        # Execution monitor callbacks
        self.execution_monitor.add_status_callback(self._on_execution_status_change)
        self.execution_monitor.add_event_callback(self._on_execution_event)
        self.execution_monitor.add_error_callback(self._on_execution_error)
        
        # Performance tracker callbacks
        self.performance_tracker.add_alert_callback(self._on_performance_alert)
        
        # Error detector callbacks
        self.error_detector.add_alert_callback(self._on_error_alert)
    
    def _on_execution_status_change(self, tracker: AgentExecutionTracker) -> None:
        """Handle execution status changes."""
        self.broadcast_event('execution_status_change', {
            'agent_name': tracker.agent_name,
            'session_id': tracker.session_id,
            'status': tracker.current_status.value,
            'duration': tracker.get_execution_duration(),
            'current_step': tracker.current_step
        })
    
    def _on_execution_event(self, event) -> None:
        """Handle execution events."""
        self.broadcast_event('execution_event', event.to_dict())
    
    def _on_execution_error(self, agent_name: str, session_id: str, error: str) -> None:
        """Handle execution errors."""
        self.broadcast_event('execution_error', {
            'agent_name': agent_name,
            'session_id': session_id,
            'error': error
        })
    
    def _on_performance_alert(self, alert: PerformanceAlert) -> None:
        """Handle performance alerts."""
        self.broadcast_event('performance_alert', alert.to_dict())
    
    def _on_error_alert(self, alert: ErrorAlert) -> None:
        """Handle error alerts."""
        self.broadcast_event('error_alert', alert.to_dict())
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop for dashboard updates."""
        while not self._stop_event.is_set():
            try:
                # Update dashboard
                self._update_dashboard()
                
                # Broadcast dashboard update
                self.broadcast_event('dashboard_update', self.current_dashboard.to_dict())
                
                # Store in history
                self.dashboard_history.append(self.current_dashboard)
                
                # Wait for next update
                self._stop_event.wait(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                self._stop_event.wait(self.update_interval)
    
    def _update_dashboard(self) -> None:
        """Update the current dashboard with latest data."""
        # Get active executions
        active_executions = self.execution_monitor.get_active_executions()
        
        # Count by status and agent
        status_counts = defaultdict(int)
        agent_counts = defaultdict(int)
        
        for tracker in active_executions:
            status_counts[tracker.current_status.value] += 1
            agent_counts[tracker.agent_name] += 1
        
        # Get performance metrics
        performance_summary = self.performance_tracker.get_performance_summary()
        
        # Get error statistics
        error_stats = self.error_detector.get_error_statistics(time_window_hours=1)
        
        # Get system health
        health_status = self.get_system_health()
        
        # Get recent events and alerts
        recent_events = self.execution_monitor.get_recent_events(limit=10)
        recent_alerts = self.error_detector.get_recent_alerts(limit=10)
        
        # Update dashboard
        self.current_dashboard = MonitoringDashboard(
            timestamp=datetime.now(),
            active_executions=[tracker.to_dict() for tracker in active_executions],
            execution_count_by_status=dict(status_counts),
            execution_count_by_agent=dict(agent_counts),
            avg_execution_time=performance_summary.get('avg_execution_time', 0),
            avg_processing_time=performance_summary.get('avg_execution_time', 0),  # Placeholder
            peak_memory_usage=performance_summary.get('peak_memory_mb', 0),
            avg_cpu_usage=performance_summary.get('avg_cpu_percent', 0),
            error_count_last_hour=error_stats.get('total_errors', 0),
            error_rate=error_stats.get('error_rate_per_hour', 0),
            critical_alerts=len([a for a in recent_alerts if a['severity'] == 'critical']),
            system_health_score=health_status['health_score'],
            health_indicators=health_status['health_indicators'],
            recent_events=[event.to_dict() for event in recent_events],
            recent_alerts=recent_alerts
        )
    
    async def _start_websocket_server(self) -> None:
        """Start WebSocket server for real-time updates."""
        try:
            self.websocket_server = await websockets.serve(
                self._handle_websocket_connection,
                "localhost",
                self.websocket_port
            )
            logger.info(f"WebSocket server started on port {self.websocket_port}")
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
    
    async def _handle_websocket_connection(self, websocket: WebSocketServerProtocol, path: str) -> None:
        """Handle new WebSocket connection."""
        self.websocket_clients.add(websocket)
        logger.info(f"New WebSocket client connected: {websocket.remote_address}")
        
        try:
            # Send initial dashboard state
            await websocket.send(json.dumps({
                'type': 'initial_dashboard',
                'data': self.current_dashboard.to_dict()
            }))
            
            # Keep connection alive
            async for message in websocket:
                # Handle client messages if needed
                try:
                    data = json.loads(message)
                    await self._handle_websocket_message(websocket, data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from client: {message}")
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket client disconnected: {websocket.remote_address}")
        except Exception as e:
            logger.error(f"Error in WebSocket connection: {e}")
        finally:
            self.websocket_clients.discard(websocket)
    
    async def _handle_websocket_message(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]) -> None:
        """Handle messages from WebSocket clients."""
        message_type = data.get('type')
        
        if message_type == 'get_agent_status':
            agent_name = data.get('agent_name')
            if agent_name:
                status = self.get_agent_status(agent_name)
                await websocket.send(json.dumps({
                    'type': 'agent_status',
                    'data': status
                }))
        
        elif message_type == 'get_system_health':
            health = self.get_system_health()
            await websocket.send(json.dumps({
                'type': 'system_health',
                'data': health
            }))
        
        elif message_type == 'get_dashboard_history':
            hours = data.get('hours', 1)
            history = self.get_dashboard_history(hours)
            await websocket.send(json.dumps({
                'type': 'dashboard_history',
                'data': history
            }))
    
    async def _broadcast_to_websockets(self, event: Dict[str, Any]) -> None:
        """Broadcast event to all WebSocket clients."""
        if not self.websocket_clients:
            return
        
        message = json.dumps(event)
        disconnected_clients = set()
        
        for client in self.websocket_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.websocket_clients -= disconnected_clients


# Global instance
_global_real_time_monitor = RealTimeMonitor()


def get_real_time_monitor() -> RealTimeMonitor:
    """Get the global real-time monitor instance."""
    return _global_real_time_monitor
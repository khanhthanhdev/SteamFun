"""
MonitoringAgent for system observability and performance tracking.
Provides comprehensive monitoring, diagnostics, and analytics for the multi-agent system.
"""

import asyncio
import json
import psutil
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from langgraph.types import Command
import logging

from ..base_agent import BaseAgent
from ..state import VideoGenerationState, AgentConfig
from ..services.langfuse_service import get_langfuse_service, trace_agent_method


logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    timestamp: str


@dataclass
class AgentPerformanceMetrics:
    """Performance metrics for individual agents."""
    agent_name: str
    total_executions: int
    successful_executions: int
    failed_executions: int
    average_execution_time: float
    total_execution_time: float
    success_rate: float
    last_execution_time: Optional[float]
    error_patterns: Dict[str, int]
    timestamp: str


@dataclass
class WorkflowMetrics:
    """Overall workflow performance metrics."""
    session_id: str
    total_agents_involved: int
    total_execution_time: float
    total_steps: int
    success_rate: float
    error_count: int
    retry_count: int
    human_interventions: int
    bottleneck_agent: Optional[str]
    timestamp: str


class MonitoringAgent(BaseAgent):
    """Agent responsible for system observability and performance monitoring.
    
    Collects performance metrics, tracks resource usage, generates diagnostics,
    and provides system health insights for the multi-agent workflow.
    """
    
    def __init__(self, config: AgentConfig, system_config: Dict[str, Any]):
        """Initialize MonitoringAgent.
        
        Args:
            config: Agent configuration
            system_config: System-wide configuration
        """
        super().__init__(config, system_config)
        
        # Monitoring configuration
        self.monitoring_config = system_config.monitoring_config
        self.performance_tracking = self.monitoring_config.performance_tracking
        self.error_tracking = self.monitoring_config.error_tracking
        self.execution_tracing = self.monitoring_config.execution_tracing
        
        # Metrics storage
        self.system_metrics_history: List[SystemMetrics] = []
        self.agent_metrics_history: Dict[str, List[AgentPerformanceMetrics]] = {}
        self.workflow_metrics_history: List[WorkflowMetrics] = []
        
        # Performance thresholds
        self.cpu_threshold = self.monitoring_config.cpu_threshold
        self.memory_threshold = self.monitoring_config.memory_threshold
        self.execution_time_threshold = self.monitoring_config.execution_time_threshold
        
        # Monitoring intervals
        self.metrics_collection_interval = self.monitoring_config.metrics_collection_interval
        self.history_retention_hours = self.monitoring_config.history_retention_hours
        
        # Initialize background monitoring task (will be started when needed)
        self._monitoring_task = None
        self._monitoring_started = False
        
        logger.info(f"MonitoringAgent initialized with config: {self.monitoring_config}")
    
    def _start_background_monitoring(self):
        """Start background system monitoring task."""
        if self.performance_tracking and not self._monitoring_started:
            try:
                # Only start if there's a running event loop
                loop = asyncio.get_running_loop()
                self._monitoring_task = loop.create_task(self._background_monitoring_loop())
                self._monitoring_started = True
                logger.info("Background monitoring started")
            except RuntimeError:
                # No event loop running, will start later when execute is called
                logger.debug("No event loop running, background monitoring will start during execution")
    
    async def _background_monitoring_loop(self):
        """Background loop for continuous system monitoring."""
        try:
            while True:
                await self._collect_system_metrics()
                await asyncio.sleep(self.metrics_collection_interval)
        except asyncio.CancelledError:
            logger.info("Background monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in background monitoring loop: {e}")
    
    async def _collect_system_metrics(self):
        """Collect current system resource metrics."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=disk.percent,
                disk_free_gb=disk.free / (1024 * 1024 * 1024),
                timestamp=datetime.now().isoformat()
            )
            
            # Store metrics
            self.system_metrics_history.append(metrics)
            
            # Cleanup old metrics
            cutoff_time = datetime.now() - timedelta(hours=self.history_retention_hours)
            self.system_metrics_history = [
                m for m in self.system_metrics_history 
                if datetime.fromisoformat(m.timestamp) > cutoff_time
            ]
            
            # Check for resource alerts
            await self._check_resource_alerts(metrics)
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    async def _check_resource_alerts(self, metrics: SystemMetrics):
        """Check for resource usage alerts."""
        alerts = []
        
        if metrics.cpu_percent > self.cpu_threshold:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.memory_threshold:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.disk_usage_percent > 90:
            alerts.append(f"High disk usage: {metrics.disk_usage_percent:.1f}%")
        
        if alerts:
            logger.warning(f"Resource alerts: {', '.join(alerts)}")
            
            # Send alerts to LangFuse if enabled
            langfuse_service = get_langfuse_service()
            if langfuse_service and langfuse_service.is_enabled():
                await langfuse_service.track_performance_metrics(
                    agent_name=self.name,
                    metrics={
                        'alert_type': 'resource_usage',
                        'alerts': alerts,
                        'system_metrics': asdict(metrics)
                    },
                    state={}
                )
    
    @trace_agent_method("execute")
    async def execute(self, state: VideoGenerationState) -> Command:
        """Execute monitoring and diagnostics collection.
        
        Args:
            state: Current workflow state
            
        Returns:
            Command: Next action command
        """
        try:
            # Start background monitoring if not already started
            if not self._monitoring_started:
                self._start_background_monitoring()
            
            self.log_agent_action("starting_monitoring", {
                'session_id': state.get('session_id'),
                'current_agent': state.get('current_agent')
            })
            
            # Collect comprehensive metrics
            monitoring_data = await self._collect_comprehensive_metrics(state)
            
            # Generate diagnostics
            diagnostics = await self._generate_diagnostics(state, monitoring_data)
            
            # Update performance metrics in state
            updated_performance_metrics = {
                **state.get('performance_metrics', {}),
                'monitoring_data': monitoring_data,
                'diagnostics': diagnostics,
                'last_monitoring_update': datetime.now().isoformat()
            }
            
            # Check if monitoring detected any issues requiring attention
            next_agent = await self._determine_next_action(state, diagnostics)
            
            self.log_agent_action("monitoring_complete", {
                'metrics_collected': len(monitoring_data),
                'diagnostics_generated': len(diagnostics),
                'next_agent': next_agent
            })
            
            return Command(
                goto=next_agent,
                update={
                    'performance_metrics': updated_performance_metrics,
                    'current_agent': next_agent
                }
            )
            
        except Exception as e:
            logger.error(f"Error in MonitoringAgent execution: {e}")
            return await self.handle_error(e, state)
    
    async def _collect_comprehensive_metrics(self, state: VideoGenerationState) -> Dict[str, Any]:
        """Collect comprehensive system and workflow metrics.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dict: Comprehensive metrics data
        """
        metrics_data = {
            'collection_timestamp': datetime.now().isoformat(),
            'session_id': state.get('session_id'),
            'system_metrics': {},
            'agent_metrics': {},
            'workflow_metrics': {},
            'execution_trace_analysis': {},
            'error_analysis': {}
        }
        
        # Collect current system metrics
        if self.performance_tracking:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                metrics_data['system_metrics'] = {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / (1024 ** 3),
                    'memory_available_gb': memory.available / (1024 ** 3),
                    'disk_usage_percent': disk.percent,
                    'disk_free_gb': disk.free / (1024 ** 3),
                    'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                }
            except Exception as e:
                logger.error(f"Failed to collect system metrics: {e}")
                metrics_data['system_metrics'] = {'error': str(e)}
        
        # Analyze agent performance from state
        performance_metrics = state.get('performance_metrics', {})
        for agent_name, agent_perf in performance_metrics.items():
            if isinstance(agent_perf, dict) and agent_name != 'monitoring_data':
                metrics_data['agent_metrics'][agent_name] = {
                    'last_execution_time': agent_perf.get('last_execution_time', 0),
                    'average_execution_time': agent_perf.get('average_execution_time', 0),
                    'success_rate': agent_perf.get('success_rate', 0),
                    'performance_trend': self._calculate_performance_trend(agent_name)
                }
        
        # Analyze workflow execution
        execution_trace = state.get('execution_trace', [])
        if execution_trace:
            metrics_data['workflow_metrics'] = await self._analyze_workflow_execution(execution_trace, state)
        
        # Analyze execution trace patterns
        if self.execution_tracing and execution_trace:
            metrics_data['execution_trace_analysis'] = self._analyze_execution_trace(execution_trace)
        
        # Analyze error patterns
        if self.error_tracking:
            metrics_data['error_analysis'] = self._analyze_error_patterns(state)
        
        return metrics_data
    
    def _calculate_performance_trend(self, agent_name: str) -> str:
        """Calculate performance trend for an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            str: Performance trend (improving, stable, degrading)
        """
        if agent_name not in self.agent_metrics_history:
            return "insufficient_data"
        
        history = self.agent_metrics_history[agent_name]
        if len(history) < 3:
            return "insufficient_data"
        
        # Compare recent performance with historical average
        recent_metrics = history[-3:]
        historical_metrics = history[:-3] if len(history) > 3 else history
        
        recent_avg_time = sum(m.average_execution_time for m in recent_metrics) / len(recent_metrics)
        historical_avg_time = sum(m.average_execution_time for m in historical_metrics) / len(historical_metrics)
        
        if recent_avg_time < historical_avg_time * 0.9:
            return "improving"
        elif recent_avg_time > historical_avg_time * 1.1:
            return "degrading"
        else:
            return "stable"
    
    async def _analyze_workflow_execution(self, execution_trace: List[Dict[str, Any]], state: VideoGenerationState) -> Dict[str, Any]:
        """Analyze overall workflow execution metrics.
        
        Args:
            execution_trace: Execution trace data
            state: Current workflow state
            
        Returns:
            Dict: Workflow analysis results
        """
        if not execution_trace:
            return {'error': 'No execution trace available'}
        
        # Calculate workflow timing
        start_time = None
        end_time = None
        total_execution_time = 0
        
        for step in execution_trace:
            timestamp = step.get('timestamp')
            if timestamp:
                step_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                if start_time is None or step_time < start_time:
                    start_time = step_time
                if end_time is None or step_time > end_time:
                    end_time = step_time
            
            exec_time = step.get('execution_time', 0)
            if isinstance(exec_time, (int, float)):
                total_execution_time += exec_time
        
        # Analyze agent involvement
        agents_involved = set()
        agent_execution_times = {}
        agent_step_counts = {}
        error_count = 0
        
        for step in execution_trace:
            agent = step.get('agent')
            if agent:
                agents_involved.add(agent)
                
                # Track execution times
                exec_time = step.get('execution_time', 0)
                if isinstance(exec_time, (int, float)):
                    agent_execution_times[agent] = agent_execution_times.get(agent, 0) + exec_time
                
                # Track step counts
                agent_step_counts[agent] = agent_step_counts.get(agent, 0) + 1
                
                # Count errors
                if 'error' in step or step.get('action') == 'error_occurred':
                    error_count += 1
        
        # Identify bottleneck agent
        bottleneck_agent = None
        if agent_execution_times:
            bottleneck_agent = max(agent_execution_times.items(), key=lambda x: x[1])[0]
        
        # Calculate success rate
        total_steps = len(execution_trace)
        success_rate = (total_steps - error_count) / total_steps if total_steps > 0 else 0
        
        workflow_duration = (end_time - start_time).total_seconds() if start_time and end_time else 0
        
        return {
            'total_steps': total_steps,
            'total_agents_involved': len(agents_involved),
            'agents_involved': list(agents_involved),
            'total_execution_time': total_execution_time,
            'workflow_duration': workflow_duration,
            'success_rate': success_rate,
            'error_count': error_count,
            'bottleneck_agent': bottleneck_agent,
            'agent_execution_times': agent_execution_times,
            'agent_step_counts': agent_step_counts,
            'retry_count': state.get('retry_count', {}),
            'human_interventions': len([s for s in execution_trace if 'human' in s.get('action', '').lower()])
        }
    
    def _analyze_execution_trace(self, execution_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze execution trace for patterns and insights.
        
        Args:
            execution_trace: Execution trace data
            
        Returns:
            Dict: Execution trace analysis
        """
        analysis = {
            'total_steps': len(execution_trace),
            'agent_transitions': [],
            'execution_patterns': {},
            'timing_analysis': {},
            'error_patterns': []
        }
        
        # Analyze agent transitions
        for i in range(len(execution_trace) - 1):
            current_agent = execution_trace[i].get('agent')
            next_agent = execution_trace[i + 1].get('agent')
            
            if current_agent and next_agent and current_agent != next_agent:
                analysis['agent_transitions'].append({
                    'from': current_agent,
                    'to': next_agent,
                    'step': i + 1
                })
        
        # Analyze execution patterns
        agent_actions = {}
        for step in execution_trace:
            agent = step.get('agent')
            action = step.get('action')
            
            if agent and action:
                if agent not in agent_actions:
                    agent_actions[agent] = {}
                agent_actions[agent][action] = agent_actions[agent].get(action, 0) + 1
        
        analysis['execution_patterns'] = agent_actions
        
        # Analyze timing patterns
        execution_times = [step.get('execution_time', 0) for step in execution_trace if step.get('execution_time')]
        if execution_times:
            analysis['timing_analysis'] = {
                'min_execution_time': min(execution_times),
                'max_execution_time': max(execution_times),
                'avg_execution_time': sum(execution_times) / len(execution_times),
                'total_execution_time': sum(execution_times)
            }
        
        # Analyze error patterns
        for step in execution_trace:
            if 'error' in step or step.get('action') == 'error_occurred':
                analysis['error_patterns'].append({
                    'agent': step.get('agent'),
                    'error': step.get('error'),
                    'timestamp': step.get('timestamp'),
                    'step_number': execution_trace.index(step) + 1
                })
        
        return analysis
    
    def _analyze_error_patterns(self, state: VideoGenerationState) -> Dict[str, Any]:
        """Analyze error patterns and trends.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dict: Error analysis results
        """
        analysis = {
            'total_errors': state.get('error_count', 0),
            'escalated_errors': len(state.get('escalated_errors', [])),
            'retry_patterns': state.get('retry_count', {}),
            'error_by_agent': {},
            'error_types': {},
            'recovery_success_rate': 0
        }
        
        # Analyze escalated errors
        escalated_errors = state.get('escalated_errors', [])
        for error in escalated_errors:
            agent = error.get('agent', 'unknown')
            error_type = error.get('error_type', 'unknown')
            
            # Count errors by agent
            analysis['error_by_agent'][agent] = analysis['error_by_agent'].get(agent, 0) + 1
            
            # Count error types
            analysis['error_types'][error_type] = analysis['error_types'].get(error_type, 0) + 1
        
        # Calculate recovery success rate
        total_retries = sum(state.get('retry_count', {}).values())
        if total_retries > 0:
            successful_recoveries = total_retries - len(escalated_errors)
            analysis['recovery_success_rate'] = successful_recoveries / total_retries
        
        return analysis
    
    async def _generate_diagnostics(self, state: VideoGenerationState, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate diagnostic information and recommendations.
        
        Args:
            state: Current workflow state
            metrics_data: Collected metrics data
            
        Returns:
            Dict: Diagnostic information
        """
        diagnostics = {
            'timestamp': datetime.now().isoformat(),
            'session_id': state.get('session_id'),
            'system_health': 'healthy',
            'performance_issues': [],
            'recommendations': [],
            'alerts': [],
            'resource_status': 'normal'
        }
        
        # Analyze system health
        system_metrics = metrics_data.get('system_metrics', {})
        if system_metrics and not system_metrics.get('error'):
            cpu_percent = system_metrics.get('cpu_percent', 0)
            memory_percent = system_metrics.get('memory_percent', 0)
            disk_percent = system_metrics.get('disk_usage_percent', 0)
            
            # Check resource thresholds
            if cpu_percent > self.cpu_threshold:
                diagnostics['performance_issues'].append(f"High CPU usage: {cpu_percent:.1f}%")
                diagnostics['recommendations'].append("Consider reducing concurrent operations")
                diagnostics['system_health'] = 'degraded'
            
            if memory_percent > self.memory_threshold:
                diagnostics['performance_issues'].append(f"High memory usage: {memory_percent:.1f}%")
                diagnostics['recommendations'].append("Monitor memory leaks and optimize caching")
                diagnostics['system_health'] = 'degraded'
            
            if disk_percent > 90:
                diagnostics['performance_issues'].append(f"High disk usage: {disk_percent:.1f}%")
                diagnostics['recommendations'].append("Clean up temporary files and old outputs")
                diagnostics['alerts'].append("Disk space critically low")
        
        # Analyze workflow performance
        workflow_metrics = metrics_data.get('workflow_metrics', {})
        if workflow_metrics and not workflow_metrics.get('error'):
            success_rate = workflow_metrics.get('success_rate', 1.0)
            error_count = workflow_metrics.get('error_count', 0)
            bottleneck_agent = workflow_metrics.get('bottleneck_agent')
            
            if success_rate < 0.8:
                diagnostics['performance_issues'].append(f"Low workflow success rate: {success_rate:.1%}")
                diagnostics['recommendations'].append("Review error handling and retry strategies")
            
            if error_count > 5:
                diagnostics['performance_issues'].append(f"High error count: {error_count}")
                diagnostics['recommendations'].append("Investigate recurring error patterns")
            
            if bottleneck_agent:
                agent_time = workflow_metrics.get('agent_execution_times', {}).get(bottleneck_agent, 0)
                if agent_time > self.execution_time_threshold:
                    diagnostics['performance_issues'].append(f"Bottleneck detected in {bottleneck_agent}: {agent_time:.1f}s")
                    diagnostics['recommendations'].append(f"Optimize {bottleneck_agent} performance or increase timeout")
        
        # Analyze agent performance trends
        agent_metrics = metrics_data.get('agent_metrics', {})
        for agent_name, agent_perf in agent_metrics.items():
            trend = agent_perf.get('performance_trend', 'stable')
            if trend == 'degrading':
                diagnostics['performance_issues'].append(f"Performance degradation in {agent_name}")
                diagnostics['recommendations'].append(f"Review {agent_name} configuration and resource allocation")
        
        # Set overall health status
        if diagnostics['performance_issues']:
            if len(diagnostics['performance_issues']) > 3 or diagnostics['alerts']:
                diagnostics['system_health'] = 'critical'
            else:
                diagnostics['system_health'] = 'degraded'
        
        return diagnostics
    
    async def _determine_next_action(self, state: VideoGenerationState, diagnostics: Dict[str, Any]) -> str:
        """Determine next action based on monitoring results.
        
        Args:
            state: Current workflow state
            diagnostics: Generated diagnostics
            
        Returns:
            str: Next agent to execute
        """
        # Check if monitoring detected critical issues
        if diagnostics.get('system_health') == 'critical':
            if diagnostics.get('alerts'):
                # Critical system issues require human intervention
                return "human_loop_agent"
        
        # Check if workflow is complete
        if state.get('workflow_complete', False):
            return "END"
        
        # Check if there are pending errors that need handling
        if state.get('error_count', 0) > 0 and state.get('escalated_errors'):
            return "error_handler_agent"
        
        # Continue with normal workflow
        next_agent = state.get('next_agent')
        if next_agent:
            return next_agent
        
        # Default to continuing workflow
        return "planner_agent"
    
    async def get_performance_report(self, state: VideoGenerationState) -> Dict[str, Any]:
        """Generate comprehensive performance report.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dict: Performance report
        """
        try:
            # Collect current metrics
            metrics_data = await self._collect_comprehensive_metrics(state)
            
            # Generate diagnostics
            diagnostics = await self._generate_diagnostics(state, metrics_data)
            
            # Create comprehensive report
            report = {
                'report_timestamp': datetime.now().isoformat(),
                'session_id': state.get('session_id'),
                'system_metrics': metrics_data.get('system_metrics', {}),
                'workflow_metrics': metrics_data.get('workflow_metrics', {}),
                'agent_metrics': metrics_data.get('agent_metrics', {}),
                'execution_trace_analysis': metrics_data.get('execution_trace_analysis', {}),
                'error_analysis': metrics_data.get('error_analysis', {}),
                'diagnostics': diagnostics,
                'historical_trends': self._get_historical_trends(),
                'recommendations': diagnostics.get('recommendations', [])
            }
            
            # Track comprehensive monitoring data in LangFuse
            langfuse_service = get_langfuse_service()
            if langfuse_service and langfuse_service.is_enabled():
                await langfuse_service.track_monitoring_metrics(
                    monitoring_data=metrics_data,
                    diagnostics=diagnostics,
                    state=state
                )
                
                # Create performance dashboard data
                dashboard_data = await langfuse_service.create_performance_dashboard_data(
                    session_id=state.get('session_id', ''),
                    monitoring_data=metrics_data
                )
                
                # Add dashboard data to report
                report['dashboard_data'] = dashboard_data
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _get_historical_trends(self) -> Dict[str, Any]:
        """Get historical performance trends.
        
        Returns:
            Dict: Historical trends data
        """
        trends = {
            'system_metrics_trend': 'stable',
            'agent_performance_trends': {},
            'error_rate_trend': 'stable',
            'data_points': len(self.system_metrics_history)
        }
        
        # Analyze system metrics trend
        if len(self.system_metrics_history) >= 5:
            recent_cpu = [m.cpu_percent for m in self.system_metrics_history[-5:]]
            historical_cpu = [m.cpu_percent for m in self.system_metrics_history[:-5]]
            
            if historical_cpu:
                recent_avg = sum(recent_cpu) / len(recent_cpu)
                historical_avg = sum(historical_cpu) / len(historical_cpu)
                
                if recent_avg > historical_avg * 1.2:
                    trends['system_metrics_trend'] = 'degrading'
                elif recent_avg < historical_avg * 0.8:
                    trends['system_metrics_trend'] = 'improving'
        
        # Analyze agent performance trends
        for agent_name, history in self.agent_metrics_history.items():
            trends['agent_performance_trends'][agent_name] = self._calculate_performance_trend(agent_name)
        
        return trends
    
    async def cleanup(self):
        """Cleanup monitoring resources."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("MonitoringAgent cleanup complete")
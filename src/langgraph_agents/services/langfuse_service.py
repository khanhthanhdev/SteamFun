"""
LangFuse integration service for tracing, monitoring, and analytics.
Provides comprehensive observability for the multi-agent system.
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import logging

try:
    from langfuse import Langfuse
    from langfuse.decorators import observe, langfuse_context
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    # Create mock decorators for when LangFuse is not available
    def observe(func=None, *, name=None, as_type=None):
        def decorator(f):
            return f
        return decorator(func) if func else decorator
    
    class MockLangfuseContext:
        def update_current_trace(self, **kwargs): pass
        def update_current_span(self, **kwargs): pass
        def get_current_trace_id(self): return None
        def get_current_span_id(self): return None
    
    langfuse_context = MockLangfuseContext()

from ..state import VideoGenerationState, AgentError


logger = logging.getLogger(__name__)


class LangFuseService:
    """Service for LangFuse tracing and monitoring integration.
    
    Provides comprehensive observability for agent interactions,
    performance monitoring, and error tracking.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LangFuse service.
        
        Args:
            config: LangFuse configuration
        """
        self.config = config
        self.enabled = config.get('enabled', True) and LANGFUSE_AVAILABLE
        self.client = None
        
        if self.enabled:
            try:
                # Initialize LangFuse client
                self.client = Langfuse(
                    secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
                    public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
                    host=os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com'),
                    debug=config.get('debug', False)
                )
                
                # Test connection
                self.client.auth_check()
                logger.info("LangFuse service initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize LangFuse: {e}")
                self.enabled = False
                self.client = None
        else:
            if not LANGFUSE_AVAILABLE:
                logger.warning("LangFuse not available - install with: pip install langfuse")
            else:
                logger.info("LangFuse service disabled in configuration")
    
    def is_enabled(self) -> bool:
        """Check if LangFuse service is enabled and available.
        
        Returns:
            bool: True if service is enabled
        """
        return self.enabled and self.client is not None
    
    async def trace_workflow_execution(self, 
                                     session_id: str,
                                     topic: str,
                                     description: str,
                                     state: VideoGenerationState) -> str:
        """Start tracing a complete workflow execution.
        
        Args:
            session_id: Unique session identifier
            topic: Video topic
            description: Video description
            state: Current workflow state
            
        Returns:
            str: Trace ID
        """
        if not self.is_enabled():
            return session_id
        
        try:
            # Create a trace using the client directly
            trace = self.client.trace(
                name="video_generation_workflow",
                session_id=session_id,
                user_id=state.get('user_id', 'anonymous'),
                metadata={
                    'topic': topic,
                    'description': description,
                    'workflow_type': 'multi_agent_video_generation',
                    'agents_enabled': list(state.get('performance_metrics', {}).keys()),
                    'configuration': {
                        'use_rag': state.get('use_rag', False),
                        'use_visual_fix_code': state.get('use_visual_fix_code', False),
                        'max_scene_concurrency': state.get('max_scene_concurrency', 1),
                        'default_quality': state.get('default_quality', 'medium')
                    }
                },
                tags=['video_generation', 'multi_agent', 'langgraph']
            )
            
            logger.info(f"Started workflow trace: {trace.id}")
            return trace.id or session_id
            
        except Exception as e:
            logger.error(f"Failed to start workflow trace: {e}")
            return session_id
    
    async def trace_agent_execution(self,
                                  agent_name: str,
                                  action: str,
                                  state: VideoGenerationState,
                                  input_data: Dict[str, Any] = None,
                                  output_data: Dict[str, Any] = None) -> Optional[str]:
        """Trace individual agent execution.
        
        Args:
            agent_name: Name of the executing agent
            action: Action being performed
            state: Current workflow state
            input_data: Input data for the agent
            output_data: Output data from the agent
            
        Returns:
            Optional[str]: Span ID
        """
        if not self.is_enabled():
            return None
        
        try:
            # Create a span using the client directly
            span = self.client.span(
                name=f"{agent_name}_{action}",
                metadata={
                    'agent_name': agent_name,
                    'action': action,
                    'session_id': state.get('session_id'),
                    'topic': state.get('topic'),
                    'error_count': state.get('error_count', 0),
                    'retry_count': state.get('retry_count', {}).get(agent_name, 0),
                    'input_size': len(str(input_data)) if input_data else 0,
                    'output_size': len(str(output_data)) if output_data else 0
                },
                input=input_data,
                output=output_data,
                tags=[agent_name, action, 'agent_execution']
            )
            
            logger.debug(f"Traced agent execution: {agent_name}_{action} - {span.id}")
            return span.id
            
        except Exception as e:
            logger.error(f"Failed to trace agent execution: {e}")
            return None
    
    async def trace_llm_call(self,
                           agent_name: str,
                           model_name: str,
                           prompt: str,
                           response: str,
                           metadata: Dict[str, Any] = None) -> Optional[str]:
        """Trace LLM generation calls.
        
        Args:
            agent_name: Name of the calling agent
            model_name: Name of the LLM model
            prompt: Input prompt
            response: Model response
            metadata: Additional metadata
            
        Returns:
            Optional[str]: Generation ID
        """
        if not self.is_enabled():
            return None
        
        try:
            # Create generation trace
            generation = self.client.generation(
                name=f"{agent_name}_llm_call",
                model=model_name,
                input=prompt,
                output=response,
                metadata={
                    'agent_name': agent_name,
                    'model_name': model_name,
                    'prompt_length': len(prompt),
                    'response_length': len(response),
                    **(metadata or {})
                },
                tags=[agent_name, model_name, 'llm_generation']
            )
            
            logger.debug(f"Traced LLM call: {agent_name} -> {model_name}")
            return generation.id
            
        except Exception as e:
            logger.error(f"Failed to trace LLM call: {e}")
            return None
    
    async def track_performance_metrics(self,
                                      agent_name: str,
                                      metrics: Dict[str, Any],
                                      state: VideoGenerationState):
        """Track performance metrics for agents.
        
        Args:
            agent_name: Name of the agent
            metrics: Performance metrics
            state: Current workflow state
        """
        if not self.is_enabled():
            return
        
        try:
            # Create custom event for performance metrics
            self.client.event(
                name=f"{agent_name}_performance_metrics",
                metadata={
                    'agent_name': agent_name,
                    'session_id': state.get('session_id'),
                    'topic': state.get('topic'),
                    'metrics': metrics,
                    'timestamp': datetime.now().isoformat()
                },
                tags=[agent_name, 'performance', 'metrics']
            )
            
            logger.debug(f"Tracked performance metrics for {agent_name}")
            
        except Exception as e:
            logger.error(f"Failed to track performance metrics: {e}")
    
    async def track_error(self,
                         agent_error: AgentError,
                         state: VideoGenerationState,
                         recovery_action: str = None):
        """Track errors and recovery actions.
        
        Args:
            agent_error: Error information
            state: Current workflow state
            recovery_action: Recovery action taken
        """
        if not self.is_enabled():
            return
        
        try:
            # Create error event
            self.client.event(
                name=f"{agent_error.agent_name}_error",
                metadata={
                    'agent_name': agent_error.agent_name,
                    'error_type': agent_error.error_type,
                    'error_message': agent_error.error_message,
                    'context': agent_error.context,
                    'retry_count': agent_error.retry_count,
                    'recovery_action': recovery_action,
                    'session_id': state.get('session_id'),
                    'topic': state.get('topic'),
                    'timestamp': agent_error.timestamp.isoformat(),
                    'stack_trace': agent_error.stack_trace
                },
                tags=[agent_error.agent_name, 'error', agent_error.error_type]
            )
            
            logger.debug(f"Tracked error for {agent_error.agent_name}: {agent_error.error_type}")
            
        except Exception as e:
            logger.error(f"Failed to track error: {e}")
    
    async def track_human_intervention(self,
                                     agent_name: str,
                                     intervention_type: str,
                                     context: str,
                                     user_response: str = None,
                                     state: VideoGenerationState = None):
        """Track human-in-the-loop interventions.
        
        Args:
            agent_name: Name of the requesting agent
            intervention_type: Type of intervention
            context: Context for the intervention
            user_response: User's response
            state: Current workflow state
        """
        if not self.is_enabled():
            return
        
        try:
            # Create human intervention event
            self.client.event(
                name=f"{agent_name}_human_intervention",
                metadata={
                    'agent_name': agent_name,
                    'intervention_type': intervention_type,
                    'context': context,
                    'user_response': user_response,
                    'session_id': state.get('session_id') if state else None,
                    'topic': state.get('topic') if state else None,
                    'timestamp': datetime.now().isoformat()
                },
                tags=[agent_name, 'human_intervention', intervention_type]
            )
            
            logger.debug(f"Tracked human intervention for {agent_name}: {intervention_type}")
            
        except Exception as e:
            logger.error(f"Failed to track human intervention: {e}")
    
    async def create_execution_flow_visualization(self, 
                                                session_id: str,
                                                execution_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create execution flow visualization data.
        
        Args:
            session_id: Session identifier
            execution_trace: Execution trace data
            
        Returns:
            Dict: Visualization data
        """
        if not self.is_enabled():
            return {'error': 'LangFuse not available'}
        
        try:
            # Process execution trace into flow visualization
            flow_data = {
                'session_id': session_id,
                'total_steps': len(execution_trace),
                'agents_involved': list(set(step.get('agent', 'unknown') for step in execution_trace)),
                'execution_timeline': [],
                'agent_transitions': [],
                'performance_summary': {},
                'bottlenecks': [],
                'error_hotspots': []
            }
            
            # Build timeline
            for i, step in enumerate(execution_trace):
                flow_data['execution_timeline'].append({
                    'step': i + 1,
                    'agent': step.get('agent'),
                    'action': step.get('action'),
                    'timestamp': step.get('timestamp'),
                    'execution_time': step.get('execution_time', 0),
                    'status': 'success' if 'error' not in step else 'error'
                })
            
            # Build agent transitions
            for i in range(len(execution_trace) - 1):
                current_agent = execution_trace[i].get('agent')
                next_agent = execution_trace[i + 1].get('agent')
                
                if current_agent != next_agent:
                    flow_data['agent_transitions'].append({
                        'from': current_agent,
                        'to': next_agent,
                        'step': i + 1
                    })
            
            # Calculate performance summary and identify bottlenecks
            agent_performance = {}
            for step in execution_trace:
                agent = step.get('agent', 'unknown')
                exec_time = step.get('execution_time', 0)
                
                if agent not in agent_performance:
                    agent_performance[agent] = {
                        'total_time': 0,
                        'call_count': 0,
                        'error_count': 0,
                        'max_execution_time': 0,
                        'min_execution_time': float('inf')
                    }
                
                agent_performance[agent]['total_time'] += exec_time
                agent_performance[agent]['call_count'] += 1
                agent_performance[agent]['max_execution_time'] = max(
                    agent_performance[agent]['max_execution_time'], exec_time
                )
                agent_performance[agent]['min_execution_time'] = min(
                    agent_performance[agent]['min_execution_time'], exec_time
                )
                
                if 'error' in step:
                    agent_performance[agent]['error_count'] += 1
            
            # Calculate averages and identify bottlenecks
            for agent, perf in agent_performance.items():
                perf['average_time'] = perf['total_time'] / perf['call_count'] if perf['call_count'] > 0 else 0
                perf['success_rate'] = (perf['call_count'] - perf['error_count']) / perf['call_count'] if perf['call_count'] > 0 else 0
                
                # Identify bottlenecks (agents with high execution times)
                if perf['average_time'] > 10.0:  # More than 10 seconds average
                    flow_data['bottlenecks'].append({
                        'agent': agent,
                        'average_time': perf['average_time'],
                        'total_time': perf['total_time'],
                        'severity': 'high' if perf['average_time'] > 30.0 else 'medium'
                    })
                
                # Identify error hotspots
                if perf['error_count'] > 0 and perf['success_rate'] < 0.8:
                    flow_data['error_hotspots'].append({
                        'agent': agent,
                        'error_count': perf['error_count'],
                        'success_rate': perf['success_rate'],
                        'severity': 'high' if perf['success_rate'] < 0.5 else 'medium'
                    })
            
            flow_data['performance_summary'] = agent_performance
            
            # Store visualization data in LangFuse
            self.client.event(
                name="execution_flow_visualization",
                metadata={
                    'session_id': session_id,
                    'flow_data': flow_data,
                    'timestamp': datetime.now().isoformat(),
                    'bottlenecks_detected': len(flow_data['bottlenecks']),
                    'error_hotspots_detected': len(flow_data['error_hotspots'])
                },
                tags=['visualization', 'execution_flow', 'analytics', 'monitoring']
            )
            
            return flow_data
            
        except Exception as e:
            logger.error(f"Failed to create execution flow visualization: {e}")
            return {'error': str(e)}
    
    async def get_session_analytics(self, 
                                  session_id: str,
                                  time_range: timedelta = None) -> Dict[str, Any]:
        """Get analytics for a specific session.
        
        Args:
            session_id: Session identifier
            time_range: Time range for analytics
            
        Returns:
            Dict: Session analytics
        """
        if not self.is_enabled():
            return {'error': 'LangFuse not available'}
        
        try:
            # This would typically query LangFuse API for session data
            # For now, return a placeholder structure with enhanced analytics
            analytics = {
                'session_id': session_id,
                'total_traces': 0,
                'total_generations': 0,
                'total_errors': 0,
                'average_execution_time': 0,
                'agent_performance': {},
                'error_breakdown': {},
                'cost_analysis': {
                    'total_tokens': 0,
                    'estimated_cost': 0.0,
                    'cost_by_model': {},
                    'token_efficiency': 0.0
                },
                'time_range': {
                    'start': (datetime.now() - (time_range or timedelta(hours=1))).isoformat(),
                    'end': datetime.now().isoformat()
                },
                'performance_insights': {
                    'fastest_agent': None,
                    'slowest_agent': None,
                    'most_reliable_agent': None,
                    'error_prone_agents': []
                },
                'workflow_efficiency': {
                    'total_workflow_time': 0,
                    'agent_utilization': {},
                    'parallel_execution_opportunities': []
                }
            }
            
            logger.info(f"Generated enhanced analytics for session: {session_id}")
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get session analytics: {e}")
            return {'error': str(e)}
    
    async def flush_traces(self):
        """Flush pending traces to LangFuse."""
        if not self.is_enabled():
            return
        
        try:
            self.client.flush()
            logger.debug("Flushed traces to LangFuse")
        except Exception as e:
            logger.error(f"Failed to flush traces: {e}")
    
    async def track_monitoring_metrics(self,
                                     monitoring_data: Dict[str, Any],
                                     diagnostics: Dict[str, Any],
                                     state: VideoGenerationState):
        """Track comprehensive monitoring metrics from MonitoringAgent.
        
        Args:
            monitoring_data: Collected monitoring data
            diagnostics: Generated diagnostics
            state: Current workflow state
        """
        if not self.is_enabled():
            return
        
        try:
            # Create comprehensive monitoring event
            self.client.event(
                name="system_monitoring_report",
                metadata={
                    'session_id': state.get('session_id'),
                    'topic': state.get('topic'),
                    'monitoring_data': monitoring_data,
                    'diagnostics': diagnostics,
                    'system_health': diagnostics.get('system_health', 'unknown'),
                    'performance_issues_count': len(diagnostics.get('performance_issues', [])),
                    'recommendations_count': len(diagnostics.get('recommendations', [])),
                    'alerts_count': len(diagnostics.get('alerts', [])),
                    'timestamp': datetime.now().isoformat()
                },
                tags=['monitoring', 'system_health', 'diagnostics', 'performance']
            )
            
            # Track system resource metrics separately for better analysis
            system_metrics = monitoring_data.get('system_metrics', {})
            if system_metrics and not system_metrics.get('error'):
                self.client.event(
                    name="system_resource_metrics",
                    metadata={
                        'session_id': state.get('session_id'),
                        'cpu_percent': system_metrics.get('cpu_percent', 0),
                        'memory_percent': system_metrics.get('memory_percent', 0),
                        'memory_used_gb': system_metrics.get('memory_used_gb', 0),
                        'disk_usage_percent': system_metrics.get('disk_usage_percent', 0),
                        'timestamp': datetime.now().isoformat()
                    },
                    tags=['system_resources', 'performance', 'monitoring']
                )
            
            # Track workflow performance metrics
            workflow_metrics = monitoring_data.get('workflow_metrics', {})
            if workflow_metrics and not workflow_metrics.get('error'):
                self.client.event(
                    name="workflow_performance_metrics",
                    metadata={
                        'session_id': state.get('session_id'),
                        'total_steps': workflow_metrics.get('total_steps', 0),
                        'success_rate': workflow_metrics.get('success_rate', 0),
                        'error_count': workflow_metrics.get('error_count', 0),
                        'bottleneck_agent': workflow_metrics.get('bottleneck_agent'),
                        'total_execution_time': workflow_metrics.get('total_execution_time', 0),
                        'agents_involved': workflow_metrics.get('agents_involved', []),
                        'timestamp': datetime.now().isoformat()
                    },
                    tags=['workflow', 'performance', 'monitoring', 'analytics']
                )
            
            logger.debug("Monitoring metrics tracked in LangFuse")
            
        except Exception as e:
            logger.error(f"Failed to track monitoring metrics: {e}")
    
    async def create_performance_dashboard_data(self, 
                                              session_id: str,
                                              monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create performance dashboard data for visualization.
        
        Args:
            session_id: Session identifier
            monitoring_data: Monitoring data from MonitoringAgent
            
        Returns:
            Dict: Dashboard data structure
        """
        if not self.is_enabled():
            return {'error': 'LangFuse not available'}
        
        try:
            dashboard_data = {
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'system_overview': {
                    'health_status': 'healthy',
                    'active_agents': 0,
                    'total_executions': 0,
                    'error_rate': 0.0
                },
                'resource_utilization': {
                    'cpu_usage': 0.0,
                    'memory_usage': 0.0,
                    'disk_usage': 0.0,
                    'trend': 'stable'
                },
                'agent_performance': {},
                'workflow_insights': {
                    'bottlenecks': [],
                    'optimization_opportunities': [],
                    'success_patterns': []
                },
                'alerts': []
            }
            
            # Process system metrics
            system_metrics = monitoring_data.get('system_metrics', {})
            if system_metrics and not system_metrics.get('error'):
                dashboard_data['resource_utilization'].update({
                    'cpu_usage': system_metrics.get('cpu_percent', 0),
                    'memory_usage': system_metrics.get('memory_percent', 0),
                    'disk_usage': system_metrics.get('disk_usage_percent', 0)
                })
            
            # Process agent metrics
            agent_metrics = monitoring_data.get('agent_metrics', {})
            for agent_name, metrics in agent_metrics.items():
                dashboard_data['agent_performance'][agent_name] = {
                    'execution_time': metrics.get('last_execution_time', 0),
                    'success_rate': metrics.get('success_rate', 0),
                    'trend': metrics.get('performance_trend', 'stable'),
                    'status': 'healthy' if metrics.get('success_rate', 0) > 0.8 else 'degraded'
                }
            
            # Process workflow insights
            workflow_metrics = monitoring_data.get('workflow_metrics', {})
            if workflow_metrics and not workflow_metrics.get('error'):
                bottleneck_agent = workflow_metrics.get('bottleneck_agent')
                if bottleneck_agent:
                    dashboard_data['workflow_insights']['bottlenecks'].append({
                        'agent': bottleneck_agent,
                        'impact': 'high',
                        'recommendation': f'Optimize {bottleneck_agent} performance'
                    })
            
            # Store dashboard data in LangFuse
            self.client.event(
                name="performance_dashboard_data",
                metadata={
                    'session_id': session_id,
                    'dashboard_data': dashboard_data,
                    'timestamp': datetime.now().isoformat()
                },
                tags=['dashboard', 'performance', 'visualization', 'monitoring']
            )
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Failed to create performance dashboard data: {e}")
            return {'error': str(e)}
    
    async def shutdown(self):
        """Shutdown LangFuse service and flush remaining traces."""
        if not self.is_enabled():
            return
        
        try:
            await self.flush_traces()
            logger.info("LangFuse service shutdown complete")
        except Exception as e:
            logger.error(f"Error during LangFuse shutdown: {e}")


# Global LangFuse service instance
_langfuse_service: Optional[LangFuseService] = None


def get_langfuse_service() -> Optional[LangFuseService]:
    """Get the global LangFuse service instance.
    
    Returns:
        Optional[LangFuseService]: LangFuse service instance
    """
    return _langfuse_service


def initialize_langfuse_service(config: Dict[str, Any]) -> LangFuseService:
    """Initialize the global LangFuse service.
    
    Args:
        config: LangFuse configuration
        
    Returns:
        LangFuseService: Initialized service
    """
    global _langfuse_service
    _langfuse_service = LangFuseService(config)
    return _langfuse_service


# Decorator for automatic agent tracing
def trace_agent_method(method_name: str = None):
    """Decorator for automatic agent method tracing.
    
    Args:
        method_name: Optional method name override
    """
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            # Get LangFuse service
            service = get_langfuse_service()
            
            if service and service.is_enabled():
                # Extract state if available
                state = None
                for arg in args:
                    if isinstance(arg, dict) and 'session_id' in arg:
                        state = arg
                        break
                
                # Trace the method execution
                await service.trace_agent_execution(
                    agent_name=getattr(self, 'name', self.__class__.__name__),
                    action=method_name or func.__name__,
                    state=state or {},
                    input_data={'args_count': len(args), 'kwargs_keys': list(kwargs.keys())}
                )
            
            # Execute the original method
            result = await func(self, *args, **kwargs)
            
            if service and service.is_enabled() and state:
                # Update trace with output
                await service.trace_agent_execution(
                    agent_name=getattr(self, 'name', self.__class__.__name__),
                    action=f"{method_name or func.__name__}_complete",
                    state=state,
                    output_data={'result_type': type(result).__name__}
                )
            
            return result
        
        return wrapper
    return decorator
"""
MonitoringAgent for system monitoring and metrics collection.
Tracks performance and system health throughout the workflow.
"""

from typing import Dict, Any, List, Optional
from langgraph.types import Command
import logging
from datetime import datetime

from ..base_agent import BaseAgent
from ..state import VideoGenerationState

logger = logging.getLogger(__name__)


class MonitoringAgent(BaseAgent):
    """Agent responsible for monitoring system performance and collecting metrics."""
    
    def __init__(self, config, system_config):
        """Initialize MonitoringAgent.
        
        Args:
            config: Agent configuration
            system_config: System configuration
        """
        super().__init__(config, system_config)
        logger.info(f"MonitoringAgent initialized: {config.name}")
    
    async def execute(self, state: VideoGenerationState) -> Command:
        """Execute monitoring and metrics collection.
        
        Args:
            state: Current workflow state
            
        Returns:
            Command: LangGraph command for next action (typically END)
        """
        try:
            self.log_agent_action("start_monitoring", {
                'session_id': state.get('session_id', ''),
                'workflow_complete': state.get('workflow_complete', False)
            })
            
            # Collect performance metrics
            performance_metrics = await self._collect_performance_metrics(state)
            
            # Assess system health
            system_health = await self._assess_system_health(state)
            
            # Generate workflow summary
            workflow_summary = await self._generate_workflow_summary(state)
            
            self.log_agent_action("monitoring_complete", {
                'metrics_collected': len(performance_metrics),
                'overall_health': system_health.get('overall_status', 'unknown')
            })
            
            # Mark workflow as complete
            return Command(
                goto="END",
                update={
                    "performance_metrics": performance_metrics,
                    "system_health": system_health,
                    "workflow_summary": workflow_summary,
                    "workflow_complete": True,
                    "current_agent": self.name
                }
            )
            
        except Exception as e:
            logger.error(f"Monitoring failed: {e}")
            return await self.handle_error(e, state)
    
    async def _collect_performance_metrics(self, state: VideoGenerationState) -> Dict[str, Any]:
        """Collect performance metrics from the workflow execution.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dict[str, Any]: Collected performance metrics
        """
        execution_trace = state.get('execution_trace', [])
        
        # Calculate timing metrics
        start_time = None
        end_time = datetime.now()
        
        if execution_trace:
            start_time = execution_trace[0].get('timestamp')
            if start_time:
                start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        
        total_execution_time = 0
        if start_time:
            total_execution_time = (end_time - start_time).total_seconds()
        
        # Agent-specific metrics
        agent_metrics = {}
        for trace_entry in execution_trace:
            agent_name = trace_entry.get('agent', 'unknown')
            if agent_name not in agent_metrics:
                agent_metrics[agent_name] = {
                    'executions': 0,
                    'total_time': 0,
                    'errors': 0
                }
            
            agent_metrics[agent_name]['executions'] += 1
            
            if 'execution_time' in trace_entry:
                agent_metrics[agent_name]['total_time'] += trace_entry['execution_time']
            
            if trace_entry.get('action') == 'error_occurred':
                agent_metrics[agent_name]['errors'] += 1
        
        return {
            'total_execution_time': total_execution_time,
            'start_time': start_time.isoformat() if start_time else None,
            'end_time': end_time.isoformat(),
            'agent_metrics': agent_metrics,
            'scenes_processed': len(state.get('generated_code', {})),
            'videos_rendered': len(state.get('rendered_videos', {})),
            'errors_encountered': state.get('error_count', 0),
            'recovery_attempts': len([e for e in execution_trace if e.get('action') == 'error_recovery'])
        }
    
    async def _assess_system_health(self, state: VideoGenerationState) -> Dict[str, Any]:
        """Assess overall system health based on workflow execution.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dict[str, Any]: System health assessment
        """
        error_count = state.get('error_count', 0)
        escalated_errors = state.get('escalated_errors', [])
        generated_code = state.get('generated_code', {})
        rendered_videos = state.get('rendered_videos', {})
        
        # Determine overall status
        if error_count == 0 and rendered_videos:
            overall_status = 'healthy'
        elif error_count <= 2 and rendered_videos:
            overall_status = 'warning'
        elif len(escalated_errors) > 0:
            overall_status = 'critical'
        else:
            overall_status = 'degraded'
        
        # Agent health status
        agent_statuses = {}
        execution_trace = state.get('execution_trace', [])
        
        for trace_entry in execution_trace:
            agent_name = trace_entry.get('agent', 'unknown')
            if agent_name not in agent_statuses:
                agent_statuses[agent_name] = 'healthy'
            
            if trace_entry.get('action') == 'error_occurred':
                agent_statuses[agent_name] = 'error'
        
        return {
            'overall_status': overall_status,
            'agent_statuses': agent_statuses,
            'error_rate': error_count / max(1, len(execution_trace)),
            'success_rate': len(rendered_videos) / max(1, len(generated_code)),
            'resource_usage': 'normal',  # Placeholder
            'timestamp': datetime.now().isoformat()
        }
    
    async def _generate_workflow_summary(self, state: VideoGenerationState) -> Dict[str, Any]:
        """Generate a summary of the workflow execution.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dict[str, Any]: Workflow execution summary
        """
        return {
            'session_id': state.get('session_id', ''),
            'topic': state.get('topic', ''),
            'description': state.get('description', ''),
            'scenes_planned': len(state.get('scene_implementations', {})),
            'code_generated': len(state.get('generated_code', {})),
            'videos_rendered': len(state.get('rendered_videos', {})),
            'final_video': state.get('combined_video_path'),
            'errors_encountered': state.get('error_count', 0),
            'workflow_complete': True,
            'success': len(state.get('rendered_videos', {})) > 0
        }
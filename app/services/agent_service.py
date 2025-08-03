"""
Agent service for LangGraph workflow orchestration.
Provides service layer interface for agent operations.
"""

import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from app.core.agents.workflow import WorkflowOrchestrator
from app.core.agents.state import SystemConfig, AgentConfig, create_initial_state
from app.models.schemas.agent import (
    WorkflowExecutionRequest,
    WorkflowExecutionResponse,
    WorkflowStatusResponse,
    WorkflowListResponse,
    AgentExecutionRequest,
    AgentExecutionResponse,
    AgentConfigRequest,
    AgentConfigResponse,
    AgentStateRequest,
    AgentStateResponse,
    SystemConfigResponse,
    AgentListResponse,
    WorkflowCancelRequest,
    WorkflowCancelResponse,
    AgentHealthResponse,
    SystemHealthResponse,
    WorkflowTemplateRequest,
    WorkflowTemplateResponse,
    WorkflowTemplateListResponse
)
from app.models.enums import AgentStatus, AgentType
from app.utils.exceptions import ServiceError, ValidationError
from app.utils.logging import get_logger

logger = get_logger(__name__)


class AgentService:
    """Service layer for LangGraph agent operations."""
    
    def __init__(self, system_config: Optional[SystemConfig] = None):
        """Initialize the agent service.
        
        Args:
            system_config: System configuration for agents
        """
        self.system_config = system_config or self._create_default_config()
        try:
            self.orchestrator = WorkflowOrchestrator(self.system_config)
        except Exception as e:
            logger.error(f"Failed to initialize workflow orchestrator: {str(e)}")
            self.orchestrator = None
            
        self._active_workflows: Dict[str, Dict[str, Any]] = {}
        self._workflow_templates: Dict[str, WorkflowTemplateResponse] = {}
        self._agent_states: Dict[str, Dict[str, Any]] = {}
        self._execution_history: List[Dict[str, Any]] = []
        self._system_start_time = datetime.utcnow()
    
    def _create_default_config(self) -> SystemConfig:
        """Create default system configuration.
        
        Returns:
            SystemConfig: Default configuration
        """
        # Create default agent configurations
        default_agents = {
            'planner_agent': AgentConfig(
                name='planner_agent',
                model_config={'model_name': 'gpt-4o-mini'},
                tools=[],
                max_retries=3,
                timeout_seconds=300,
                enable_human_loop=False
            ),
            'rag_agent': AgentConfig(
                name='rag_agent',
                model_config={'model_name': 'gpt-4o-mini'},
                tools=[],
                max_retries=3,
                timeout_seconds=300,
                enable_human_loop=False
            ),
            'code_generator_agent': AgentConfig(
                name='code_generator_agent',
                model_config={'model_name': 'gpt-4o-mini'},
                tools=[],
                max_retries=3,
                timeout_seconds=300,
                enable_human_loop=False
            ),
            'renderer_agent': AgentConfig(
                name='renderer_agent',
                model_config={'model_name': 'gpt-4o-mini'},
                tools=[],
                max_retries=3,
                timeout_seconds=300,
                enable_human_loop=False
            ),
            'visual_analysis_agent': AgentConfig(
                name='visual_analysis_agent',
                model_config={'model_name': 'gpt-4o-mini'},
                tools=[],
                max_retries=2,
                timeout_seconds=300,
                enable_human_loop=False
            ),
            'error_handler_agent': AgentConfig(
                name='error_handler_agent',
                model_config={'model_name': 'gpt-4o-mini'},
                tools=[],
                max_retries=1,
                timeout_seconds=300,
                enable_human_loop=True
            ),
            'human_loop_agent': AgentConfig(
                name='human_loop_agent',
                model_config={'model_name': 'gpt-4o-mini'},
                tools=[],
                max_retries=1,
                timeout_seconds=600,
                enable_human_loop=True
            ),
            'monitoring_agent': AgentConfig(
                name='monitoring_agent',
                model_config={'model_name': 'gpt-4o-mini'},
                tools=[],
                max_retries=1,
                timeout_seconds=300,
                enable_human_loop=False
            )
        }
        
        return SystemConfig(
            agents=default_agents,
            llm_providers={
                'openai': {
                    'api_key': 'your-api-key',
                    'base_url': 'https://api.openai.com/v1'
                }
            },
            docling_config={},
            mcp_servers={},
            monitoring_config={},
            human_loop_config={},
            max_workflow_retries=3,
            workflow_timeout_seconds=3600,
            enable_checkpoints=True,
            checkpoint_interval=300
        )
    
    async def execute_workflow(self, request: WorkflowExecutionRequest) -> WorkflowExecutionResponse:
        """Execute a workflow based on the request.
        
        Args:
            request: Workflow execution request
            
        Returns:
            WorkflowExecutionResponse: Workflow execution result
        """
        try:
            if not self.orchestrator:
                raise ServiceError("Workflow orchestrator not available")
            
            session_id = request.session_id or f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
            started_at = datetime.utcnow()
            
            logger.info(f"Starting {request.workflow_type} workflow: {session_id}")
            
            # Store workflow info
            workflow_info = {
                'session_id': session_id,
                'workflow_type': request.workflow_type,
                'topic': request.topic,
                'description': request.description,
                'status': AgentStatus.PROCESSING,
                'started_at': started_at,
                'config_overrides': request.config_overrides,
                'agents_to_use': request.agents_to_use,
                'priority': request.priority,
                'timeout_seconds': request.timeout_seconds
            }
            
            self._active_workflows[session_id] = workflow_info
            
            # Execute the workflow
            if request.workflow_type == "video_generation":
                result = await self._execute_video_generation_workflow(request, session_id)
            else:
                result = await self._execute_generic_workflow(request, session_id)
            
            # Update workflow status
            completed_at = datetime.utcnow()
            workflow_info.update({
                'status': AgentStatus.COMPLETED if result.get('success') else AgentStatus.FAILED,
                'completed_at': completed_at,
                'result': result,
                'error': result.get('error') if not result.get('success') else None
            })
            
            return WorkflowExecutionResponse(
                session_id=session_id,
                workflow_type=request.workflow_type,
                status=workflow_info['status'],
                topic=request.topic,
                description=request.description,
                result=result,
                error=workflow_info.get('error'),
                started_at=started_at,
                completed_at=completed_at
            )
            
        except Exception as e:
            logger.error(f"Failed to execute workflow: {str(e)}")
            
            # Update workflow status to failed
            if 'session_id' in locals():
                if session_id in self._active_workflows:
                    self._active_workflows[session_id].update({
                        'status': AgentStatus.FAILED,
                        'error': str(e),
                        'completed_at': datetime.utcnow()
                    })
            
            raise ServiceError(f"Workflow execution failed: {str(e)}")
    
    async def _execute_video_generation_workflow(self, request: WorkflowExecutionRequest, session_id: str) -> Dict[str, Any]:
        """Execute video generation workflow."""
        try:
            result = await self.orchestrator.execute_workflow(
                topic=request.topic,
                description=request.description,
                session_id=session_id,
                initial_config=request.config_overrides
            )
            
            return {
                'success': result.get('workflow_complete', False),
                'workflow_result': result,
                'session_id': session_id
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'session_id': session_id
            }
    
    async def _execute_generic_workflow(self, request: WorkflowExecutionRequest, session_id: str) -> Dict[str, Any]:
        """Execute generic workflow."""
        # For now, return a placeholder result
        return {
            'success': True,
            'message': f"Generic workflow '{request.workflow_type}' executed successfully",
            'session_id': session_id
        }
    
    async def get_workflow_status(self, session_id: str) -> WorkflowStatusResponse:
        """Get the status of a workflow.
        
        Args:
            session_id: Session identifier
            
        Returns:
            WorkflowStatusResponse: Workflow status information
        """
        try:
            if session_id not in self._active_workflows:
                raise ServiceError(f"Workflow not found: {session_id}")
            
            workflow_info = self._active_workflows[session_id]
            
            return WorkflowStatusResponse(
                session_id=session_id,
                status=workflow_info['status'],
                workflow_type=workflow_info['workflow_type'],
                topic=workflow_info['topic'],
                current_agent=workflow_info.get('current_agent'),
                result=workflow_info.get('result'),
                error=workflow_info.get('error'),
                started_at=workflow_info['started_at'],
                updated_at=workflow_info.get('completed_at', workflow_info['started_at']),
                estimated_completion=workflow_info.get('estimated_completion'),
                metadata=workflow_info.get('metadata')
            )
            
        except Exception as e:
            logger.error(f"Failed to get workflow status: {str(e)}")
            raise ServiceError(f"Failed to get workflow status: {str(e)}")
    
    async def list_workflows(self) -> WorkflowListResponse:
        """List all workflows.
        
        Returns:
            WorkflowListResponse: List of workflow information
        """
        try:
            workflows = []
            active_count = 0
            completed_count = 0
            failed_count = 0
            
            for session_id, info in self._active_workflows.items():
                status = info['status']
                
                if status == AgentStatus.PROCESSING:
                    active_count += 1
                elif status == AgentStatus.COMPLETED:
                    completed_count += 1
                elif status == AgentStatus.FAILED:
                    failed_count += 1
                
                workflows.append(WorkflowStatusResponse(
                    session_id=session_id,
                    status=status,
                    workflow_type=info['workflow_type'],
                    topic=info['topic'],
                    current_agent=info.get('current_agent'),
                    result=info.get('result'),
                    error=info.get('error'),
                    started_at=info['started_at'],
                    updated_at=info.get('completed_at', info['started_at']),
                    estimated_completion=info.get('estimated_completion'),
                    metadata=info.get('metadata')
                ))
            
            return WorkflowListResponse(
                workflows=workflows,
                total_count=len(workflows),
                active_count=active_count,
                completed_count=completed_count,
                failed_count=failed_count
            )
            
        except Exception as e:
            logger.error(f"Failed to list workflows: {str(e)}")
            raise ServiceError(f"Failed to list workflows: {str(e)}")
    
    async def list_agents(self) -> AgentListResponse:
        """Get list of available agents.
        
        Returns:
            AgentListResponse: List of available agents
        """
        try:
            agents = []
            active_count = 0
            
            for agent_name, agent_config in self.system_config.agents.items():
                # Determine agent status (simplified)
                status = AgentStatus.IDLE  # Default status
                if self.orchestrator:
                    # In a real implementation, you would check actual agent status
                    active_count += 1
                
                agents.append(AgentConfigResponse(
                    name=agent_config.name,
                    model_configuration=agent_config.model_config,
                    tools=agent_config.tools,
                    max_retries=agent_config.max_retries,
                    timeout_seconds=agent_config.timeout_seconds,
                    enable_human_loop=agent_config.enable_human_loop,
                    status=status,
                    last_updated=datetime.utcnow()
                ))
            
            # Get available agent types from enum
            available_types = list(AgentType)
            
            return AgentListResponse(
                agents=agents,
                total_count=len(agents),
                active_count=active_count,
                available_types=available_types
            )
            
        except Exception as e:
            logger.error(f"Failed to list agents: {str(e)}")
            raise ServiceError(f"Failed to list agents: {str(e)}")
    
    async def get_agent_info(self, agent_name: str) -> AgentConfigResponse:
        """Get information about a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            AgentConfigResponse: Agent information
        """
        try:
            if agent_name not in self.system_config.agents:
                raise ServiceError(f"Agent not found: {agent_name}")
            
            agent_config = self.system_config.agents[agent_name]
            
            return AgentConfigResponse(
                name=agent_config.name,
                model_configuration=agent_config.model_config,
                tools=agent_config.tools,
                max_retries=agent_config.max_retries,
                timeout_seconds=agent_config.timeout_seconds,
                enable_human_loop=agent_config.enable_human_loop,
                status=AgentStatus.IDLE,  # Would need to check actual status
                last_updated=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to get agent info: {str(e)}")
            raise ServiceError(f"Failed to get agent info: {str(e)}")
    
    def update_agent_config(
        self, 
        agent_name: str, 
        config_updates: Dict[str, Any]
    ) -> bool:
        """Update configuration for a specific agent.
        
        Args:
            agent_name: Name of the agent
            config_updates: Configuration updates to apply
            
        Returns:
            bool: True if update was successful
        """
        try:
            if agent_name not in self.system_config.agents:
                return False
            
            agent_config = self.system_config.agents[agent_name]
            
            # Update configuration fields
            for key, value in config_updates.items():
                if hasattr(agent_config, key):
                    setattr(agent_config, key, value)
            
            logger.info(f"Updated configuration for agent: {agent_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update agent config: {e}")
            return False
    
    async def cancel_workflow(self, request: WorkflowCancelRequest) -> WorkflowCancelResponse:
        """Cancel a running workflow.
        
        Args:
            request: Workflow cancellation request
            
        Returns:
            WorkflowCancelResponse: Cancellation result
        """
        try:
            if request.session_id not in self._active_workflows:
                raise ServiceError(f"Workflow not found: {request.session_id}")
            
            workflow_info = self._active_workflows[request.session_id]
            
            if workflow_info['status'] not in [AgentStatus.PROCESSING, AgentStatus.PENDING]:
                raise ServiceError(f"Workflow cannot be cancelled in status: {workflow_info['status']}")
            
            # Update workflow status
            workflow_info.update({
                'status': AgentStatus.CANCELLED,
                'completed_at': datetime.utcnow(),
                'cancellation_reason': request.reason
            })
            
            cancelled_at = datetime.utcnow()
            
            return WorkflowCancelResponse(
                session_id=request.session_id,
                status="cancelled",
                message=f"Workflow cancelled successfully. Reason: {request.reason or 'User requested'}",
                cancelled_at=cancelled_at
            )
            
        except Exception as e:
            logger.error(f"Failed to cancel workflow: {str(e)}")
            raise ServiceError(f"Failed to cancel workflow: {str(e)}")
    
    async def execute_agent(self, request: AgentExecutionRequest) -> AgentExecutionResponse:
        """Execute a single agent.
        
        Args:
            request: Agent execution request
            
        Returns:
            AgentExecutionResponse: Agent execution result
        """
        try:
            execution_id = str(uuid.uuid4())
            started_at = datetime.utcnow()
            
            # Simulate agent execution
            # In a real implementation, this would execute the actual agent
            result = {
                'agent_type': request.agent_type.value,
                'input_data': request.input_data,
                'output': f"Processed by {request.agent_type.value}",
                'success': True
            }
            
            completed_at = datetime.utcnow()
            execution_time = (completed_at - started_at).total_seconds()
            
            # Store execution history
            self._execution_history.append({
                'execution_id': execution_id,
                'agent_type': request.agent_type,
                'started_at': started_at,
                'completed_at': completed_at,
                'execution_time': execution_time,
                'success': True
            })
            
            return AgentExecutionResponse(
                agent_type=request.agent_type,
                execution_id=execution_id,
                session_id=request.session_id,
                status=AgentStatus.COMPLETED,
                result=result,
                execution_time=execution_time,
                started_at=started_at,
                completed_at=completed_at
            )
            
        except Exception as e:
            logger.error(f"Failed to execute agent: {str(e)}")
            
            completed_at = datetime.utcnow()
            execution_time = (completed_at - started_at).total_seconds()
            
            return AgentExecutionResponse(
                agent_type=request.agent_type,
                execution_id=execution_id,
                session_id=request.session_id,
                status=AgentStatus.FAILED,
                error=str(e),
                execution_time=execution_time,
                started_at=started_at,
                completed_at=completed_at
            )
    
    async def get_system_config(self) -> SystemConfigResponse:
        """Get current system configuration.
        
        Returns:
            SystemConfigResponse: System configuration
        """
        try:
            agents = {}
            for name, config in self.system_config.agents.items():
                agents[name] = AgentConfigResponse(
                    name=config.name,
                    model_configuration=config.model_config,
                    tools=config.tools,
                    max_retries=config.max_retries,
                    timeout_seconds=config.timeout_seconds,
                    enable_human_loop=config.enable_human_loop,
                    status=AgentStatus.IDLE,
                    last_updated=datetime.utcnow()
                )
            
            return SystemConfigResponse(
                agents=agents,
                workflow_config={
                    'max_workflow_retries': self.system_config.max_workflow_retries,
                    'workflow_timeout_seconds': self.system_config.workflow_timeout_seconds,
                    'enable_checkpoints': self.system_config.enable_checkpoints,
                    'checkpoint_interval': self.system_config.checkpoint_interval
                },
                llm_providers=self.system_config.llm_providers,
                monitoring_config=self.system_config.monitoring_config,
                human_loop_config=self.system_config.human_loop_config,
                system_status="active" if self.orchestrator else "error",
                last_updated=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to get system config: {str(e)}")
            raise ServiceError(f"Failed to get system config: {str(e)}")
    
    async def get_system_health(self) -> SystemHealthResponse:
        """Get system health information.
        
        Returns:
            SystemHealthResponse: System health status
        """
        try:
            # Calculate system metrics
            total_executions = len(self._execution_history)
            failed_executions = len([e for e in self._execution_history if not e.get('success', True)])
            error_rate = failed_executions / total_executions if total_executions > 0 else 0.0
            
            avg_response_time = 0.0
            if self._execution_history:
                total_time = sum(e.get('execution_time', 0) for e in self._execution_history)
                avg_response_time = total_time / len(self._execution_history)
            
            system_uptime = (datetime.utcnow() - self._system_start_time).total_seconds()
            
            # Get agent health
            agent_health = []
            for agent_name, agent_config in self.system_config.agents.items():
                # Calculate agent-specific metrics
                agent_executions = [e for e in self._execution_history if e.get('agent_type') == agent_name]
                agent_errors = len([e for e in agent_executions if not e.get('success', True)])
                
                last_execution = None
                if agent_executions:
                    last_execution = max(e.get('completed_at') for e in agent_executions)
                
                avg_exec_time = None
                if agent_executions:
                    total_time = sum(e.get('execution_time', 0) for e in agent_executions)
                    avg_exec_time = total_time / len(agent_executions)
                
                # Calculate health score (simplified)
                health_score = 1.0
                if agent_executions:
                    health_score = 1.0 - (agent_errors / len(agent_executions))
                
                agent_health.append(AgentHealthResponse(
                    agent_type=AgentType.PLANNER,  # Would need proper mapping
                    status=AgentStatus.IDLE,
                    last_execution=last_execution,
                    error_count=agent_errors,
                    average_execution_time=avg_exec_time,
                    health_score=health_score
                ))
            
            overall_status = "healthy" if error_rate < 0.1 else "degraded" if error_rate < 0.3 else "unhealthy"
            
            return SystemHealthResponse(
                overall_status=overall_status,
                agents=agent_health,
                active_workflows=len([w for w in self._active_workflows.values() if w['status'] == AgentStatus.PROCESSING]),
                total_executions=total_executions,
                error_rate=error_rate,
                average_response_time=avg_response_time,
                system_uptime=system_uptime,
                last_health_check=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to get system health: {str(e)}")
            raise ServiceError(f"Failed to get system health: {str(e)}")
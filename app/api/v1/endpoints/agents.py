"""
LangGraph Agents API Endpoints

Provides REST API endpoints for LangGraph agent operations including:
- Workflow execution and management
- Agent configuration and monitoring
- State management
- System health and status
"""

import uuid
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from app.models.schemas.agent import (
    WorkflowExecutionRequest,
    WorkflowExecutionResponse,
    WorkflowStatusResponse,
    WorkflowListResponse,
    WorkflowCancelRequest,
    WorkflowCancelResponse,
    AgentExecutionRequest,
    AgentExecutionResponse,
    AgentConfigRequest,
    AgentConfigResponse,
    AgentListResponse,
    AgentStateRequest,
    AgentStateResponse,
    SystemConfigResponse,
    SystemHealthResponse,
    AgentHealthResponse,
    WorkflowTemplateRequest,
    WorkflowTemplateResponse,
    WorkflowTemplateListResponse
)
from app.models.enums import AgentStatus, AgentType
from app.services.agent_service import AgentService
from app.api.dependencies import CommonDeps, get_logger
from app.utils.exceptions import AgentError

router = APIRouter(prefix="/agents", tags=["agents"])

# Initialize agent service (this would typically be dependency injected)
_agent_service: Optional[AgentService] = None


def get_agent_service() -> AgentService:
    """Get or create agent service instance."""
    global _agent_service
    if _agent_service is None:
        try:
            _agent_service = AgentService()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Agent service unavailable: {str(e)}"
            )
    return _agent_service


@router.post("/workflows/execute", response_model=WorkflowExecutionResponse)
async def execute_workflow(
    request: WorkflowExecutionRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = CommonDeps,
    logger = Depends(get_logger),
    agent_service: AgentService = Depends(get_agent_service)
) -> WorkflowExecutionResponse:
    """
    Execute a workflow using LangGraph agents.
    
    Starts a workflow execution with the specified configuration. The workflow
    runs in the background and can be monitored via the status endpoint.
    """
    try:
        logger.info(f"Starting workflow execution: {request.workflow_type}")
        
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Start workflow execution in background
        if request.workflow_type == "video_generation":
            background_tasks.add_task(
                _execute_video_workflow_background,
                agent_service,
                request.topic,
                request.description,
                session_id,
                request.config_overrides
            )
        else:
            # Handle other workflow types
            background_tasks.add_task(
                _execute_generic_workflow_background,
                agent_service,
                request,
                session_id
            )
        
        return WorkflowExecutionResponse(
            session_id=session_id,
            workflow_type=request.workflow_type,
            status=AgentStatus.RUNNING,
            topic=request.topic,
            description=request.description,
            started_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to execute workflow: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute workflow: {str(e)}"
        )


async def _execute_video_workflow_background(
    agent_service: AgentService,
    topic: str,
    description: str,
    session_id: str,
    config_overrides: Optional[Dict[str, Any]]
):
    """Background task for video workflow execution."""
    try:
        result = await agent_service.start_video_generation_workflow(
            topic=topic,
            description=description,
            session_id=session_id,
            config_overrides=config_overrides
        )
        print(f"Video workflow completed for session {session_id}: {result.get('status')}")
    except Exception as e:
        print(f"Video workflow failed for session {session_id}: {str(e)}")


async def _execute_generic_workflow_background(
    agent_service: AgentService,
    request: WorkflowExecutionRequest,
    session_id: str
):
    """Background task for generic workflow execution."""
    try:
        # This would be implemented based on the specific workflow type
        print(f"Generic workflow {request.workflow_type} started for session {session_id}")
        # Mock completion after some processing
        await asyncio.sleep(1)
        print(f"Generic workflow completed for session {session_id}")
    except Exception as e:
        print(f"Generic workflow failed for session {session_id}: {str(e)}")


@router.get("/workflows/{session_id}/status", response_model=WorkflowStatusResponse)
async def get_workflow_status(
    session_id: str,
    current_user: dict = CommonDeps,
    logger = Depends(get_logger),
    agent_service: AgentService = Depends(get_agent_service)
) -> WorkflowStatusResponse:
    """
    Get workflow execution status.
    
    Returns the current status of a workflow execution including progress,
    current agent, and any results or errors.
    """
    try:
        logger.info(f"Getting workflow status for session: {session_id}")
        
        status_info = agent_service.get_workflow_status(session_id)
        
        if status_info.get('status') == 'not_found':
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow with session ID {session_id} not found"
            )
        
        # Convert service response to API response
        workflow_status = AgentStatus.COMPLETED if status_info.get('status') == 'completed' else AgentStatus.RUNNING
        if status_info.get('status') == 'failed':
            workflow_status = AgentStatus.FAILED
        
        return WorkflowStatusResponse(
            session_id=session_id,
            status=workflow_status,
            workflow_type="video_generation",  # Would be stored in service
            topic="Unknown",  # Would be retrieved from service
            result=status_info.get('result'),
            error=status_info.get('error'),
            started_at=datetime.fromisoformat(status_info.get('started_at', datetime.utcnow().isoformat())),
            updated_at=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow status: {str(e)}"
        )


@router.get("/workflows", response_model=WorkflowListResponse)
async def list_workflows(
    status_filter: Optional[AgentStatus] = None,
    workflow_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: dict = CommonDeps,
    logger = Depends(get_logger),
    agent_service: AgentService = Depends(get_agent_service)
) -> WorkflowListResponse:
    """
    List workflows with optional filtering.
    
    Returns a paginated list of workflows with optional filtering by status
    and workflow type.
    """
    try:
        logger.info(f"Listing workflows with status: {status_filter}, type: {workflow_type}")
        
        active_workflows = agent_service.list_active_workflows()
        
        # Convert to API response format
        workflows = []
        for workflow in active_workflows:
            workflow_status = AgentStatus.RUNNING
            if workflow.get('status') == 'completed':
                workflow_status = AgentStatus.COMPLETED
            elif workflow.get('status') == 'failed':
                workflow_status = AgentStatus.FAILED
            
            workflows.append(WorkflowStatusResponse(
                session_id=workflow['session_id'],
                status=workflow_status,
                workflow_type="video_generation",  # Would be stored
                topic="Unknown",  # Would be retrieved
                started_at=datetime.fromisoformat(workflow['started_at']),
                updated_at=datetime.utcnow()
            ))
        
        # Apply filters
        if status_filter:
            workflows = [w for w in workflows if w.status == status_filter]
        if workflow_type:
            workflows = [w for w in workflows if w.workflow_type == workflow_type]
        
        # Apply pagination
        total_count = len(workflows)
        workflows = workflows[offset:offset + limit]
        
        # Count by status
        active_count = sum(1 for w in workflows if w.status == AgentStatus.RUNNING)
        completed_count = sum(1 for w in workflows if w.status == AgentStatus.COMPLETED)
        failed_count = sum(1 for w in workflows if w.status == AgentStatus.FAILED)
        
        return WorkflowListResponse(
            workflows=workflows,
            total_count=total_count,
            active_count=active_count,
            completed_count=completed_count,
            failed_count=failed_count
        )
        
    except Exception as e:
        logger.error(f"Failed to list workflows: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list workflows: {str(e)}"
        )


@router.post("/workflows/{session_id}/cancel", response_model=WorkflowCancelResponse)
async def cancel_workflow(
    session_id: str,
    request: WorkflowCancelRequest,
    current_user: dict = CommonDeps,
    logger = Depends(get_logger)
) -> WorkflowCancelResponse:
    """
    Cancel a running workflow.
    
    Attempts to gracefully cancel a running workflow. If force is specified,
    the workflow will be terminated immediately.
    """
    try:
        logger.info(f"Cancelling workflow: {session_id}")
        
        # This would implement actual workflow cancellation
        # For now, return a mock response
        
        return WorkflowCancelResponse(
            session_id=session_id,
            status="cancelled",
            message=f"Workflow {session_id} has been cancelled",
            cancelled_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to cancel workflow: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel workflow: {str(e)}"
        )


@router.post("/execute", response_model=AgentExecutionResponse)
async def execute_agent(
    request: AgentExecutionRequest,
    current_user: dict = CommonDeps,
    logger = Depends(get_logger)
) -> AgentExecutionResponse:
    """
    Execute a single agent.
    
    Executes a specific agent with the provided input data and configuration.
    This is useful for testing individual agents or running standalone operations.
    """
    try:
        logger.info(f"Executing agent: {request.agent_type}")
        
        execution_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        # Mock agent execution
        # In a real implementation, this would execute the actual agent
        import asyncio
        await asyncio.sleep(0.1)  # Simulate processing time
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        return AgentExecutionResponse(
            agent_type=request.agent_type,
            execution_id=execution_id,
            session_id=request.session_id,
            status=AgentStatus.COMPLETED,
            result={
                "message": f"Agent {request.agent_type} executed successfully",
                "input_data": request.input_data,
                "processed_at": datetime.utcnow().isoformat()
            },
            execution_time=execution_time,
            started_at=start_time,
            completed_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to execute agent: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute agent: {str(e)}"
        )


@router.get("/", response_model=AgentListResponse)
async def list_agents(
    current_user: dict = CommonDeps,
    logger = Depends(get_logger),
    agent_service: AgentService = Depends(get_agent_service)
) -> AgentListResponse:
    """
    List all available agents.
    
    Returns information about all available agents including their
    configuration and current status.
    """
    try:
        logger.info("Listing available agents")
        
        available_agents = agent_service.get_available_agents()
        
        agents = []
        for agent_name in available_agents:
            agent_info = agent_service.get_agent_info(agent_name)
            if agent_info:
                agents.append(AgentConfigResponse(
                    name=agent_name,
                    model_config=agent_info.get('model_config', {}),
                    tools=agent_info.get('tools', []),
                    max_retries=agent_info.get('max_retries', 3),
                    timeout_seconds=agent_info.get('timeout_seconds', 300),
                    enable_human_loop=agent_info.get('enable_human_loop', False),
                    status=AgentStatus.IDLE,
                    last_updated=datetime.utcnow()
                ))
        
        return AgentListResponse(
            agents=agents,
            total_count=len(agents),
            active_count=0,  # Would be calculated from actual status
            available_types=list(AgentType)
        )
        
    except Exception as e:
        logger.error(f"Failed to list agents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list agents: {str(e)}"
        )


@router.get("/{agent_name}/config", response_model=AgentConfigResponse)
async def get_agent_config(
    agent_name: str,
    current_user: dict = CommonDeps,
    logger = Depends(get_logger),
    agent_service: AgentService = Depends(get_agent_service)
) -> AgentConfigResponse:
    """
    Get configuration for a specific agent.
    
    Returns the current configuration settings for the specified agent.
    """
    try:
        logger.info(f"Getting config for agent: {agent_name}")
        
        agent_info = agent_service.get_agent_info(agent_name)
        
        if not agent_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_name} not found"
            )
        
        return AgentConfigResponse(
            name=agent_name,
            model_config=agent_info.get('model_config', {}),
            tools=agent_info.get('tools', []),
            max_retries=agent_info.get('max_retries', 3),
            timeout_seconds=agent_info.get('timeout_seconds', 300),
            enable_human_loop=agent_info.get('enable_human_loop', False),
            status=AgentStatus.IDLE,
            last_updated=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent config: {str(e)}"
        )


@router.put("/{agent_name}/config", response_model=AgentConfigResponse)
async def update_agent_config(
    agent_name: str,
    request: AgentConfigRequest,
    current_user: dict = CommonDeps,
    logger = Depends(get_logger),
    agent_service: AgentService = Depends(get_agent_service)
) -> AgentConfigResponse:
    """
    Update configuration for a specific agent.
    
    Updates the configuration settings for the specified agent. Changes
    take effect immediately for new executions.
    """
    try:
        logger.info(f"Updating config for agent: {agent_name}")
        
        config_updates = {
            'model_config': request.model_config,
            'tools': request.tools,
            'max_retries': request.max_retries,
            'timeout_seconds': request.timeout_seconds,
            'enable_human_loop': request.enable_human_loop
        }
        
        success = agent_service.update_agent_config(agent_name, config_updates)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_name} not found or update failed"
            )
        
        return AgentConfigResponse(
            name=agent_name,
            model_config=request.model_config,
            tools=request.tools,
            max_retries=request.max_retries,
            timeout_seconds=request.timeout_seconds,
            enable_human_loop=request.enable_human_loop,
            custom_settings=request.custom_settings,
            status=AgentStatus.IDLE,
            last_updated=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update agent config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update agent config: {str(e)}"
        )


@router.get("/system/config", response_model=SystemConfigResponse)
async def get_system_config(
    current_user: dict = CommonDeps,
    logger = Depends(get_logger),
    agent_service: AgentService = Depends(get_agent_service)
) -> SystemConfigResponse:
    """
    Get system configuration.
    
    Returns the complete system configuration including all agents,
    workflow settings, and provider configurations.
    """
    try:
        logger.info("Getting system configuration")
        
        system_config = agent_service.get_system_config()
        
        # Convert agent configs to response format
        agents = {}
        for name, config in system_config.get('agents', {}).items():
            agents[name] = AgentConfigResponse(
                name=name,
                model_config=config.get('model_config', {}),
                tools=config.get('tools', []),
                max_retries=config.get('max_retries', 3),
                timeout_seconds=config.get('timeout_seconds', 300),
                enable_human_loop=config.get('enable_human_loop', False),
                status=AgentStatus.IDLE,
                last_updated=datetime.utcnow()
            )
        
        return SystemConfigResponse(
            agents=agents,
            workflow_config=system_config.get('workflow_config', {}),
            llm_providers={},  # Would be populated from actual config
            monitoring_config={},  # Would be populated from actual config
            human_loop_config={},  # Would be populated from actual config
            system_status="active",
            last_updated=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to get system config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system config: {str(e)}"
        )


@router.get("/system/health", response_model=SystemHealthResponse)
async def get_system_health(
    current_user: dict = CommonDeps,
    logger = Depends(get_logger)
) -> SystemHealthResponse:
    """
    Get system health status.
    
    Returns comprehensive health information about the agent system
    including individual agent health and overall system metrics.
    """
    try:
        logger.info("Getting system health")
        
        # Mock health data
        agent_health = [
            AgentHealthResponse(
                agent_type=AgentType.PLANNER,
                status=AgentStatus.IDLE,
                last_execution=datetime.utcnow(),
                error_count=0,
                average_execution_time=2.5,
                health_score=0.95
            ),
            AgentHealthResponse(
                agent_type=AgentType.RAG,
                status=AgentStatus.IDLE,
                last_execution=datetime.utcnow(),
                error_count=1,
                average_execution_time=1.8,
                health_score=0.88
            )
        ]
        
        return SystemHealthResponse(
            overall_status="healthy",
            agents=agent_health,
            active_workflows=0,
            total_executions=150,
            error_rate=0.02,
            average_response_time=2.1,
            system_uptime=86400.0,  # 24 hours
            last_health_check=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to get system health: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system health: {str(e)}"
        )
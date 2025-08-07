"""
Testing API Endpoints

Provides REST API endpoints for testing LangGraph agents including:
- Agent discovery and listing for testing
- Individual agent testing execution
- Complete workflow testing
- Session management for test tracking
"""

import uuid
import asyncio
import json
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.models.enums import AgentStatus, AgentType
from app.services.agent_service import AgentService
from app.api.dependencies import CommonDeps, get_logger
from app.utils.exceptions import AgentError

router = APIRouter(prefix="/test", tags=["testing"])

# Test-specific models
class AgentTestInfo(BaseModel):
    """Information about an agent available for testing"""
    name: str
    type: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    test_examples: List[Dict[str, Any]] = []

class AgentTestRequest(BaseModel):
    """Request model for individual agent testing"""
    inputs: Dict[str, Any]
    session_id: Optional[str] = None
    options: Dict[str, Any] = {}

class AgentTestResponse(BaseModel):
    """Response model for individual agent testing"""
    session_id: str
    agent_name: str
    status: str
    results: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    logs: List[str] = []
    errors: List[str] = []
    started_at: datetime
    completed_at: Optional[datetime] = None

class WorkflowTestRequest(BaseModel):
    """Request model for complete workflow testing"""
    topic: str
    description: str
    session_id: Optional[str] = None
    config_overrides: Dict[str, Any] = {}

class WorkflowTestResponse(BaseModel):
    """Response model for complete workflow testing"""
    session_id: str
    status: str
    current_step: Optional[str] = None
    progress: float = 0.0
    results: Optional[Dict[str, Any]] = None
    video_output: Optional[str] = None
    execution_time: Optional[float] = None
    logs: List[str] = []
    errors: List[str] = []
    started_at: datetime
    completed_at: Optional[datetime] = None

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

# In-memory storage for test sessions (in production, use proper storage)
_test_sessions: Dict[str, Dict[str, Any]] = {}

# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for real-time log streaming."""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept a WebSocket connection and add it to the session."""
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, session_id: str):
        """Remove a WebSocket connection from the session."""
        if session_id in self.active_connections:
            if websocket in self.active_connections[session_id]:
                self.active_connections[session_id].remove(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
    
    async def send_log(self, session_id: str, log_entry: Dict[str, Any]):
        """Send a log entry to all connected clients for a session."""
        if session_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_text(json.dumps(log_entry))
                except:
                    disconnected.append(connection)
            
            # Remove disconnected connections
            for connection in disconnected:
                self.disconnect(connection, session_id)
    
    async def send_status_update(self, session_id: str, status_update: Dict[str, Any]):
        """Send a status update to all connected clients for a session."""
        if session_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_text(json.dumps({
                        'type': 'status_update',
                        'data': status_update
                    }))
                except:
                    disconnected.append(connection)
            
            # Remove disconnected connections
            for connection in disconnected:
                self.disconnect(connection, session_id)

# Global connection manager instance
connection_manager = ConnectionManager()

@router.get("/agents", response_model=List[AgentTestInfo])
async def list_available_agents(
    current_user: dict = CommonDeps,
    logger = Depends(get_logger),
    agent_service: AgentService = Depends(get_agent_service)
) -> List[AgentTestInfo]:
    """
    List all available agents for testing.
    
    Returns detailed information about each agent including input/output schemas
    and test examples to help with testing setup.
    """
    try:
        logger.info("Listing available agents for testing")
        
        available_agents = agent_service.get_available_agents()
        agent_test_info = []
        
        for agent_name in available_agents:
            agent_info = agent_service.get_agent_info(agent_name)
            if agent_info:
                # Create test info with schema and examples
                test_info = AgentTestInfo(
                    name=agent_name,
                    type=agent_info.get('type', 'unknown'),
                    description=agent_info.get('description', f'{agent_name} agent'),
                    input_schema=_get_agent_input_schema(agent_name),
                    output_schema=_get_agent_output_schema(agent_name),
                    test_examples=_get_agent_test_examples(agent_name)
                )
                agent_test_info.append(test_info)
        
        return agent_test_info
        
    except Exception as e:
        logger.error(f"Failed to list available agents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list available agents: {str(e)}"
        )

@router.post("/agent/{agent_name}", response_model=AgentTestResponse)
async def test_individual_agent(
    agent_name: str,
    request: AgentTestRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = CommonDeps,
    logger = Depends(get_logger),
    agent_service: AgentService = Depends(get_agent_service)
) -> AgentTestResponse:
    """
    Test an individual agent with provided inputs.
    
    Executes a specific agent in isolation with the provided test data.
    This is useful for debugging specific agent behaviors and validating
    agent functionality independently.
    """
    try:
        logger.info(f"Testing individual agent: {agent_name}")
        
        # Validate agent exists
        available_agents = agent_service.get_available_agents()
        if agent_name not in available_agents:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent '{agent_name}' not found"
            )
        
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Initialize test session
        _test_sessions[session_id] = {
            'type': 'agent_test',
            'agent_name': agent_name,
            'status': 'running',
            'started_at': datetime.utcnow(),
            'logs': [],
            'errors': []
        }
        
        # Start agent execution in background
        background_tasks.add_task(
            _execute_agent_test_background,
            agent_service,
            agent_name,
            request.inputs,
            session_id,
            request.options
        )
        
        return AgentTestResponse(
            session_id=session_id,
            agent_name=agent_name,
            status='running',
            started_at=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to test agent {agent_name}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to test agent {agent_name}: {str(e)}"
        )

@router.post("/workflow", response_model=WorkflowTestResponse)
async def test_complete_workflow(
    request: WorkflowTestRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = CommonDeps,
    logger = Depends(get_logger),
    agent_service: AgentService = Depends(get_agent_service)
) -> WorkflowTestResponse:
    """
    Test the complete workflow from input to video output.
    
    Executes the entire video generation pipeline including planning,
    code generation, and rendering agents. Provides step-by-step progress
    tracking and final video output.
    """
    try:
        logger.info(f"Testing complete workflow: {request.topic}")
        
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Initialize test session
        _test_sessions[session_id] = {
            'type': 'workflow_test',
            'topic': request.topic,
            'description': request.description,
            'status': 'running',
            'current_step': 'initializing',
            'progress': 0.0,
            'started_at': datetime.utcnow(),
            'logs': [],
            'errors': []
        }
        
        # Start workflow execution in background
        background_tasks.add_task(
            _execute_workflow_test_background,
            agent_service,
            request.topic,
            request.description,
            session_id,
            request.config_overrides
        )
        
        return WorkflowTestResponse(
            session_id=session_id,
            status='running',
            current_step='initializing',
            progress=0.0,
            started_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to test workflow: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to test workflow: {str(e)}"
        )

async def _execute_agent_test_background(
    agent_service: AgentService,
    agent_name: str,
    inputs: Dict[str, Any],
    session_id: str,
    options: Dict[str, Any]
):
    """Background task for individual agent testing."""
    try:
        session = _test_sessions.get(session_id, {})
        
        # Send initial log
        log_message = f"Starting {agent_name} agent test"
        session['logs'].append(log_message)
        await connection_manager.send_log(session_id, {
            'type': 'log',
            'level': 'info',
            'message': log_message,
            'timestamp': datetime.utcnow().isoformat(),
            'session_id': session_id,
            'component': agent_name
        })
        
        # Send status update
        await connection_manager.send_status_update(session_id, {
            'session_id': session_id,
            'status': 'running',
            'agent_name': agent_name,
            'progress': 0.1
        })
        
        # Mock agent execution with progress updates
        for i, step in enumerate(['initializing', 'processing', 'finalizing'], 1):
            log_message = f"Agent {agent_name}: {step}"
            session['logs'].append(log_message)
            await connection_manager.send_log(session_id, {
                'type': 'log',
                'level': 'info',
                'message': log_message,
                'timestamp': datetime.utcnow().isoformat(),
                'session_id': session_id,
                'component': agent_name
            })
            
            await connection_manager.send_status_update(session_id, {
                'session_id': session_id,
                'status': 'running',
                'agent_name': agent_name,
                'progress': 0.1 + (i * 0.3)
            })
            
            await asyncio.sleep(0.7)  # Simulate processing time
        
        # Mock successful result
        result = {
            'agent': agent_name,
            'inputs': inputs,
            'outputs': {
                'message': f'Agent {agent_name} executed successfully',
                'processed_data': inputs,
                'timestamp': datetime.utcnow().isoformat()
            },
            'metadata': {
                'execution_time': 2.0,
                'options_used': options
            }
        }
        
        # Update session
        session.update({
            'status': 'completed',
            'results': result,
            'execution_time': 2.0,
            'completed_at': datetime.utcnow()
        })
        
        # Send completion log and status
        log_message = f"Agent {agent_name} test completed successfully"
        session['logs'].append(log_message)
        await connection_manager.send_log(session_id, {
            'type': 'log',
            'level': 'success',
            'message': log_message,
            'timestamp': datetime.utcnow().isoformat(),
            'session_id': session_id,
            'component': agent_name
        })
        
        await connection_manager.send_status_update(session_id, {
            'session_id': session_id,
            'status': 'completed',
            'agent_name': agent_name,
            'progress': 1.0,
            'results': result,
            'execution_time': 2.0
        })
        
    except Exception as e:
        session = _test_sessions.get(session_id, {})
        session.update({
            'status': 'failed',
            'completed_at': datetime.utcnow()
        })
        
        error_message = f"Agent test failed: {str(e)}"
        session['errors'].append(error_message)
        session['logs'].append(f"Agent {agent_name} test failed: {str(e)}")
        
        # Send error log and status
        await connection_manager.send_log(session_id, {
            'type': 'log',
            'level': 'error',
            'message': error_message,
            'timestamp': datetime.utcnow().isoformat(),
            'session_id': session_id,
            'component': agent_name
        })
        
        await connection_manager.send_status_update(session_id, {
            'session_id': session_id,
            'status': 'failed',
            'agent_name': agent_name,
            'error': error_message
        })

async def _execute_workflow_test_background(
    agent_service: AgentService,
    topic: str,
    description: str,
    session_id: str,
    config_overrides: Dict[str, Any]
):
    """Background task for complete workflow testing."""
    try:
        session = _test_sessions.get(session_id, {})
        
        # Step 1: Planning
        session.update({
            'current_step': 'planning',
            'progress': 0.2
        })
        log_message = "Starting planning phase"
        session['logs'].append(log_message)
        await connection_manager.send_log(session_id, {
            'type': 'log',
            'level': 'info',
            'message': log_message,
            'timestamp': datetime.utcnow().isoformat(),
            'session_id': session_id,
            'component': 'planning_agent'
        })
        await connection_manager.send_status_update(session_id, {
            'session_id': session_id,
            'status': 'running',
            'current_step': 'planning',
            'progress': 0.2
        })
        await asyncio.sleep(1)
        
        # Step 2: Code Generation
        session.update({
            'current_step': 'code_generation',
            'progress': 0.5
        })
        log_message = "Starting code generation phase"
        session['logs'].append(log_message)
        await connection_manager.send_log(session_id, {
            'type': 'log',
            'level': 'info',
            'message': log_message,
            'timestamp': datetime.utcnow().isoformat(),
            'session_id': session_id,
            'component': 'code_generation_agent'
        })
        await connection_manager.send_status_update(session_id, {
            'session_id': session_id,
            'status': 'running',
            'current_step': 'code_generation',
            'progress': 0.5
        })
        await asyncio.sleep(2)
        
        # Step 3: Rendering
        session.update({
            'current_step': 'rendering',
            'progress': 0.8
        })
        log_message = "Starting rendering phase"
        session['logs'].append(log_message)
        await connection_manager.send_log(session_id, {
            'type': 'log',
            'level': 'info',
            'message': log_message,
            'timestamp': datetime.utcnow().isoformat(),
            'session_id': session_id,
            'component': 'rendering_agent'
        })
        await connection_manager.send_status_update(session_id, {
            'session_id': session_id,
            'status': 'running',
            'current_step': 'rendering',
            'progress': 0.8
        })
        await asyncio.sleep(3)
        
        # Complete
        result = {
            'topic': topic,
            'description': description,
            'video_output': f'/test_output/{session_id}/video.mp4',
            'thumbnail': f'/test_output/{session_id}/thumbnail.png',
            'metadata': {
                'total_execution_time': 6.0,
                'config_overrides': config_overrides,
                'steps_completed': ['planning', 'code_generation', 'rendering']
            }
        }
        
        session.update({
            'status': 'completed',
            'current_step': 'completed',
            'progress': 1.0,
            'results': result,
            'video_output': result['video_output'],
            'execution_time': 6.0,
            'completed_at': datetime.utcnow()
        })
        
        log_message = "Workflow test completed successfully"
        session['logs'].append(log_message)
        await connection_manager.send_log(session_id, {
            'type': 'log',
            'level': 'success',
            'message': log_message,
            'timestamp': datetime.utcnow().isoformat(),
            'session_id': session_id,
            'component': 'workflow'
        })
        await connection_manager.send_status_update(session_id, {
            'session_id': session_id,
            'status': 'completed',
            'current_step': 'completed',
            'progress': 1.0,
            'results': result,
            'video_output': result['video_output'],
            'execution_time': 6.0
        })
        
    except Exception as e:
        session = _test_sessions.get(session_id, {})
        session.update({
            'status': 'failed',
            'completed_at': datetime.utcnow()
        })
        
        error_message = f"Workflow test failed: {str(e)}"
        session['errors'].append(error_message)
        session['logs'].append(f"Workflow test failed: {str(e)}")
        
        await connection_manager.send_log(session_id, {
            'type': 'log',
            'level': 'error',
            'message': error_message,
            'timestamp': datetime.utcnow().isoformat(),
            'session_id': session_id,
            'component': 'workflow'
        })
        await connection_manager.send_status_update(session_id, {
            'session_id': session_id,
            'status': 'failed',
            'error': error_message
        })

def _get_agent_input_schema(agent_name: str) -> Dict[str, Any]:
    """Get input schema for a specific agent."""
    schemas = {
        'planning_agent': {
            'topic': {'type': 'string', 'required': True, 'description': 'Topic for the video'},
            'description': {'type': 'string', 'required': True, 'description': 'Detailed description'},
            'complexity': {'type': 'string', 'enum': ['basic', 'intermediate', 'advanced'], 'default': 'intermediate'}
        },
        'code_generation_agent': {
            'scene_plan': {'type': 'object', 'required': True, 'description': 'Scene plan from planning agent'},
            'style': {'type': 'string', 'enum': ['minimal', 'detailed', 'animated'], 'default': 'detailed'}
        },
        'rendering_agent': {
            'code': {'type': 'string', 'required': True, 'description': 'Manim code to render'},
            'quality': {'type': 'string', 'enum': ['low', 'medium', 'high'], 'default': 'medium'}
        }
    }
    return schemas.get(agent_name, {
        'input': {'type': 'object', 'description': 'Agent input data'}
    })

def _get_agent_output_schema(agent_name: str) -> Dict[str, Any]:
    """Get output schema for a specific agent."""
    schemas = {
        'planning_agent': {
            'scene_plan': {'type': 'object', 'description': 'Generated scene plan'},
            'metadata': {'type': 'object', 'description': 'Planning metadata'}
        },
        'code_generation_agent': {
            'code': {'type': 'string', 'description': 'Generated Manim code'},
            'imports': {'type': 'array', 'description': 'Required imports'},
            'metadata': {'type': 'object', 'description': 'Generation metadata'}
        },
        'rendering_agent': {
            'video_path': {'type': 'string', 'description': 'Path to rendered video'},
            'thumbnail_path': {'type': 'string', 'description': 'Path to video thumbnail'},
            'metadata': {'type': 'object', 'description': 'Rendering metadata'}
        }
    }
    return schemas.get(agent_name, {
        'result': {'type': 'object', 'description': 'Agent output data'}
    })

@router.websocket("/ws/logs/{session_id}")
async def websocket_logs(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time log streaming.
    
    Provides real-time streaming of logs and status updates for a specific
    test session. Clients can connect to receive live updates during
    agent execution or workflow testing.
    """
    await connection_manager.connect(websocket, session_id)
    try:
        # Send initial session status if available
        if session_id in _test_sessions:
            session = _test_sessions[session_id]
            await connection_manager.send_status_update(session_id, {
                'session_id': session_id,
                'status': session.get('status', 'unknown'),
                'current_step': session.get('current_step'),
                'progress': session.get('progress', 0.0),
                'started_at': session.get('started_at', datetime.utcnow()).isoformat()
            })
            
            # Send existing logs
            for log_entry in session.get('logs', []):
                await connection_manager.send_log(session_id, {
                    'type': 'log',
                    'level': 'info',
                    'message': log_entry,
                    'timestamp': datetime.utcnow().isoformat(),
                    'session_id': session_id
                })
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                # Handle any client messages if needed (e.g., log level filtering)
                message = json.loads(data)
                if message.get('type') == 'ping':
                    await websocket.send_text(json.dumps({'type': 'pong'}))
            except WebSocketDisconnect:
                break
            except Exception as e:
                # Log error but don't break connection
                await connection_manager.send_log(session_id, {
                    'type': 'log',
                    'level': 'error',
                    'message': f'WebSocket error: {str(e)}',
                    'timestamp': datetime.utcnow().isoformat(),
                    'session_id': session_id
                })
    except WebSocketDisconnect:
        pass
    finally:
        connection_manager.disconnect(websocket, session_id)

@router.get("/logs/{session_id}")
async def get_session_logs(
    session_id: str,
    current_user: dict = CommonDeps,
    logger = Depends(get_logger)
) -> Dict[str, Any]:
    """
    Get logs for a specific test session.
    
    Returns all logs and status information for a test session.
    This is useful for retrieving historical logs or getting the
    current state of a session.
    """
    try:
        logger.info(f"Getting logs for session: {session_id}")
        
        if session_id not in _test_sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found"
            )
        
        session = _test_sessions[session_id]
        
        return {
            'session_id': session_id,
            'type': session.get('type', 'unknown'),
            'status': session.get('status', 'unknown'),
            'current_step': session.get('current_step'),
            'progress': session.get('progress', 0.0),
            'logs': [
                {
                    'level': 'info',
                    'message': log_entry,
                    'timestamp': datetime.utcnow().isoformat(),
                    'component': session.get('agent_name', 'system')
                }
                for log_entry in session.get('logs', [])
            ],
            'errors': [
                {
                    'level': 'error',
                    'message': error,
                    'timestamp': datetime.utcnow().isoformat(),
                    'component': session.get('agent_name', 'system')
                }
                for error in session.get('errors', [])
            ],
            'started_at': session.get('started_at', datetime.utcnow()).isoformat(),
            'completed_at': session.get('completed_at', {}).isoformat() if session.get('completed_at') else None,
            'execution_time': session.get('execution_time'),
            'results': session.get('results')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session logs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session logs: {str(e)}"
        )

@router.get("/sessions")
async def list_test_sessions(
    status_filter: Optional[str] = None,
    session_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: dict = CommonDeps,
    logger = Depends(get_logger)
) -> Dict[str, Any]:
    """
    List all test sessions with optional filtering.
    
    Returns a paginated list of test sessions with optional filtering
    by status and session type. Useful for session management and
    monitoring active tests.
    """
    try:
        logger.info(f"Listing test sessions with filters: status={status_filter}, type={session_type}")
        
        sessions = []
        for session_id, session_data in _test_sessions.items():
            # Apply filters
            if status_filter and session_data.get('status') != status_filter:
                continue
            if session_type and session_data.get('type') != session_type:
                continue
            
            sessions.append({
                'session_id': session_id,
                'type': session_data.get('type', 'unknown'),
                'status': session_data.get('status', 'unknown'),
                'agent_name': session_data.get('agent_name'),
                'topic': session_data.get('topic'),
                'current_step': session_data.get('current_step'),
                'progress': session_data.get('progress', 0.0),
                'started_at': session_data.get('started_at', datetime.utcnow()).isoformat(),
                'completed_at': session_data.get('completed_at', {}).isoformat() if session_data.get('completed_at') else None,
                'execution_time': session_data.get('execution_time'),
                'log_count': len(session_data.get('logs', [])),
                'error_count': len(session_data.get('errors', []))
            })
        
        # Sort by started_at (most recent first)
        sessions.sort(key=lambda x: x['started_at'], reverse=True)
        
        # Apply pagination
        total_count = len(sessions)
        sessions = sessions[offset:offset + limit]
        
        # Calculate status counts
        status_counts = {}
        for session_data in _test_sessions.values():
            status = session_data.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'sessions': sessions,
            'total_count': total_count,
            'status_counts': status_counts,
            'pagination': {
                'limit': limit,
                'offset': offset,
                'has_more': offset + limit < total_count
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to list test sessions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list test sessions: {str(e)}"
        )

def _get_agent_test_examples(agent_name: str) -> List[Dict[str, Any]]:
    """Get test examples for a specific agent."""
    examples = {
        'planning_agent': [
            {
                'name': 'Basic Math Concept',
                'inputs': {
                    'topic': 'Pythagorean Theorem',
                    'description': 'Explain the Pythagorean theorem with visual proof',
                    'complexity': 'basic'
                }
            },
            {
                'name': 'Advanced Physics',
                'inputs': {
                    'topic': 'Wave Interference',
                    'description': 'Demonstrate constructive and destructive wave interference',
                    'complexity': 'advanced'
                }
            }
        ],
        'code_generation_agent': [
            {
                'name': 'Simple Animation',
                'inputs': {
                    'scene_plan': {
                        'title': 'Circle Animation',
                        'elements': ['circle', 'text'],
                        'animations': ['create', 'transform']
                    },
                    'style': 'minimal'
                }
            }
        ],
        'rendering_agent': [
            {
                'name': 'Basic Scene',
                'inputs': {
                    'code': 'from manim import *\n\nclass TestScene(Scene):\n    def construct(self):\n        circle = Circle()\n        self.play(Create(circle))',
                    'quality': 'medium'
                }
            }
        ]
    }
    return examples.get(agent_name, [])
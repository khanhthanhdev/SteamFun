"""
Studio server integration for LangGraph Studio compatibility.

This module provides the server integration layer that connects the workflow
graph with LangGraph Studio's server infrastructure, enabling real-time
monitoring, debugging, and execution control.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..models.state import VideoGenerationState
from ..models.config import WorkflowConfig
from .studio_integration import studio_monitor, studio_registry
from .studio_config import get_studio_config
from .studio_workflow_config import (
    create_studio_workflow_config,
    get_studio_workflow_info
)
from .studio_workflow_visualization import get_studio_visualizer, get_studio_inspector
from .test_scenarios import get_test_scenario_manager

logger = logging.getLogger(__name__)


class WorkflowExecutionRequest(BaseModel):
    """Request model for workflow execution."""
    topic: str = Field(..., min_length=3, max_length=200, description="Video topic")
    description: str = Field(..., min_length=10, max_length=1000, description="Video description")
    session_id: Optional[str] = Field(None, description="Session identifier")
    preview_mode: bool = Field(True, description="Enable preview mode")
    max_scenes: int = Field(5, ge=1, le=10, description="Maximum number of scenes")


class AgentTestRequest(BaseModel):
    """Request model for agent testing."""
    agent_name: str = Field(..., description="Name of the agent to test")
    scenario_name: str = Field(..., description="Test scenario name")
    test_input: Dict[str, Any] = Field(..., description="Test input data")


class StudioServerIntegration:
    """Server integration for LangGraph Studio."""
    
    def __init__(self):
        self.app = FastAPI(
            title="LangGraph Studio - Video Generation Workflow",
            description="Studio server integration for video generation workflow testing",
            version="2.0.0"
        )
        self.studio_config = get_studio_config()
        self.workflow_config = create_studio_workflow_config()
        self.visualizer = get_studio_visualizer()
        self.inspector = get_studio_inspector()
        self.test_manager = get_test_scenario_manager()
        self.active_connections: List[WebSocket] = []
        self.setup_middleware()
        self.setup_routes()
    
    def setup_middleware(self):
        """Set up CORS and other middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                "http://localhost:8123",
                "http://127.0.0.1:8123"
            ],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Set up API routes for Studio integration."""
        
        # Health and info endpoints
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "studio_enabled": self.studio_config.studio_enabled,
                "workflow_ready": True
            }
        
        @self.app.get("/info")
        async def workflow_info():
            """Get comprehensive workflow information."""
            return get_studio_workflow_info()
        
        # Graph schema endpoints
        @self.app.get("/api/schemas/workflow")
        async def get_workflow_schema():
            """Get workflow input/output schema."""
            return self.workflow_config.get_workflow_schema()
        
        @self.app.get("/api/schemas/nodes")
        async def get_node_schemas():
            """Get node schemas."""
            return self.workflow_config._get_node_schemas()
        
        @self.app.get("/api/schemas/state")
        async def get_state_schema():
            """Get state schema."""
            return {
                "state_type": "VideoGenerationState",
                "properties": {
                    "topic": {"type": "string"},
                    "description": {"type": "string"},
                    "session_id": {"type": "string"},
                    "current_step": {"type": "string"},
                    "workflow_complete": {"type": "boolean"},
                    "scene_outline": {"type": "string"},
                    "scene_implementations": {"type": "object"},
                    "generated_code": {"type": "object"},
                    "rendered_videos": {"type": "object"},
                    "errors": {"type": "array"}
                }
            }
        
        # Workflow execution endpoints
        @self.app.post("/api/workflow/execute")
        async def execute_workflow(request: WorkflowExecutionRequest):
            """Execute the complete workflow."""
            try:
                # Create workflow state
                state = VideoGenerationState(
                    topic=request.topic,
                    description=request.description,
                    session_id=request.session_id or f"studio_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config=self.workflow_config.workflow_config
                )
                
                # Create workflow instance
                workflow = self.workflow_config.create_studio_compatible_workflow()
                
                # Execute workflow
                result_state = await workflow.invoke(state)
                
                # Broadcast execution completion
                await self.broadcast_to_connections({
                    "type": "workflow_completed",
                    "session_id": result_state.session_id,
                    "completion_percentage": result_state.get_completion_percentage() if hasattr(result_state, 'get_completion_percentage') else 100
                })
                
                return {
                    "status": "completed",
                    "session_id": result_state.session_id,
                    "result": {
                        "workflow_complete": result_state.workflow_complete,
                        "current_step": result_state.current_step,
                        "scene_count": len(result_state.scene_implementations) if result_state.scene_implementations else 0,
                        "video_count": len(result_state.rendered_videos) if result_state.rendered_videos else 0,
                        "error_count": len(result_state.errors),
                        "completion_percentage": result_state.get_completion_percentage() if hasattr(result_state, 'get_completion_percentage') else 0
                    }
                }
                
            except Exception as e:
                logger.error(f"Workflow execution failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/workflow/stream")
        async def stream_workflow(request: WorkflowExecutionRequest):
            """Stream workflow execution."""
            try:
                # Create workflow state
                state = VideoGenerationState(
                    topic=request.topic,
                    description=request.description,
                    session_id=request.session_id or f"studio_stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config=self.workflow_config.workflow_config
                )
                
                # Create workflow instance
                workflow = self.workflow_config.create_studio_compatible_workflow()
                
                # Stream workflow execution
                results = []
                async for chunk in workflow.stream(state):
                    results.append(chunk)
                    # Broadcast chunk to WebSocket connections
                    await self.broadcast_to_connections({
                        "type": "workflow_chunk",
                        "session_id": state.session_id,
                        "chunk": chunk
                    })
                
                return {
                    "status": "completed",
                    "session_id": state.session_id,
                    "chunks": len(results)
                }
                
            except Exception as e:
                logger.error(f"Workflow streaming failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Agent testing endpoints
        @self.app.post("/api/agents/test")
        async def test_agent(request: AgentTestRequest):
            """Test an individual agent."""
            try:
                from .studio_integration import create_studio_tester
                
                tester = create_studio_tester(self.workflow_config.workflow_config)
                result = await tester.test_agent(
                    request.agent_name,
                    request.test_input,
                    request.scenario_name
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Agent testing failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/agents/list")
        async def list_agents():
            """List available agents."""
            return {
                "agents": studio_registry.list_agents(),
                "agent_metadata": {
                    name: studio_registry.get_agent_metadata(name)
                    for name in studio_registry.list_agents()
                }
            }
        
        # Monitoring endpoints
        @self.app.get("/api/monitoring/metrics")
        async def get_metrics():
            """Get performance metrics."""
            return studio_monitor.get_performance_metrics()
        
        @self.app.get("/api/monitoring/sessions")
        async def get_active_sessions():
            """Get active sessions."""
            return {
                "active_sessions": studio_monitor.active_sessions,
                "session_count": len(studio_monitor.active_sessions)
            }
        
        @self.app.get("/api/monitoring/history")
        async def get_execution_history():
            """Get execution history."""
            return {
                "execution_history": studio_monitor.get_execution_history(),
                "total_executions": len(studio_monitor.execution_history)
            }
        
        # Debugging endpoints
        @self.app.get("/api/debug/inspect/{session_id}")
        async def inspect_session(session_id: str):
            """Inspect a specific session."""
            return self.inspector.get_inspection_summary(session_id)
        
        @self.app.get("/api/debug/state-snapshots/{session_id}")
        async def get_state_snapshots(session_id: str):
            """Get state snapshots for a session."""
            session_snapshots = [
                snapshot for snapshot in self.inspector.state_snapshots.values()
                if snapshot["session_id"] == session_id
            ]
            return {
                "session_id": session_id,
                "snapshots": session_snapshots
            }
        
        @self.app.get("/api/debug/state-diff/{snapshot_id1}/{snapshot_id2}")
        async def get_state_diff(snapshot_id1: str, snapshot_id2: str):
            """Get difference between two state snapshots."""
            return self.inspector.get_state_diff(snapshot_id1, snapshot_id2)
        
        # Test scenario endpoints
        @self.app.get("/api/test/scenarios")
        async def get_test_scenarios():
            """Get available test scenarios."""
            return {
                "scenarios": self.test_manager.list_scenarios(),
                "scenario_count": sum(len(scenarios) for scenarios in self.test_manager.scenarios.values())
            }
        
        @self.app.get("/api/test/scenarios/{agent_name}")
        async def get_agent_scenarios(agent_name: str):
            """Get scenarios for a specific agent."""
            return {
                "agent_name": agent_name,
                "scenarios": self.test_manager.list_scenarios(agent_name)
            }
        
        @self.app.post("/api/test/run/{agent_name}/{scenario_name}")
        async def run_test_scenario(agent_name: str, scenario_name: str):
            """Run a specific test scenario."""
            try:
                scenario = self.test_manager.get_scenario(agent_name, scenario_name)
                if not scenario:
                    raise HTTPException(status_code=404, detail="Scenario not found")
                
                from .studio_integration import create_studio_tester
                tester = create_studio_tester(self.workflow_config.workflow_config)
                
                result = await tester.test_agent(
                    agent_name,
                    scenario["input"],
                    scenario_name
                )
                
                # Validate result against scenario expectations
                validation = self.test_manager.validate_scenario_output(
                    agent_name,
                    scenario_name,
                    result.get("output", {})
                )
                
                return {
                    "test_result": result,
                    "validation": validation
                }
                
            except Exception as e:
                logger.error(f"Test scenario execution failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Visualization endpoints
        @self.app.get("/api/visualization/graph")
        async def get_graph_visualization():
            """Get graph visualization data."""
            return {
                "mermaid_diagram": self.visualizer.generate_mermaid_diagram(),
                "graph_metadata": self.visualizer.create_studio_graph_metadata()
            }
        
        @self.app.get("/api/visualization/nodes/{node_name}")
        async def get_node_configuration(node_name: str):
            """Get configuration for a specific node."""
            config = self.visualizer.get_node_configuration(node_name)
            if not config:
                raise HTTPException(status_code=404, detail="Node not found")
            return config
        
        @self.app.get("/api/visualization/execution-path/{session_id}")
        async def get_execution_path(session_id: str):
            """Get execution path for a session."""
            # This would need to be implemented with actual state data
            return {
                "session_id": session_id,
                "execution_path": [],
                "note": "Implementation requires session state data"
            }
        
        # WebSocket endpoint for real-time monitoring
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time monitoring."""
            await websocket.accept()
            self.active_connections.append(websocket)
            
            try:
                while True:
                    # Keep connection alive and handle incoming messages
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Handle different message types
                    if message.get("type") == "subscribe":
                        await websocket.send_text(json.dumps({
                            "type": "subscription_confirmed",
                            "subscribed_to": message.get("topics", [])
                        }))
                    
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
                logger.info("WebSocket connection closed")
    
    async def broadcast_to_connections(self, message: Dict[str, Any]):
        """Broadcast message to all active WebSocket connections."""
        if not self.active_connections:
            return
        
        message_str = json.dumps(message)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except:
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.active_connections.remove(connection)
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application."""
        return self.app


def create_studio_server() -> StudioServerIntegration:
    """Create a Studio server integration instance."""
    return StudioServerIntegration()


def run_studio_server(host: str = "0.0.0.0", port: int = 8123, debug: bool = True):
    """Run the Studio server."""
    import uvicorn
    
    server = create_studio_server()
    app = server.get_app()
    
    logger.info(f"Starting Studio server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        debug=debug,
        reload=debug,
        log_level="info"
    )


if __name__ == "__main__":
    # Run the Studio server
    run_studio_server()
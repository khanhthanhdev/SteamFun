"""
LangGraph Studio integration for agent testing.

This module provides Studio-compatible interfaces for testing individual agents
and monitoring their execution in the LangGraph Studio environment.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Union, Type
from datetime import datetime
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from ..models.state import VideoGenerationState
from ..models.config import WorkflowConfig
from ..workflow_graph import VideoGenerationWorkflow
from ..nodes.planning_node import planning_node
from ..nodes.code_generation_node import code_generation_node
from ..nodes.rendering_node import rendering_node
from ..nodes.error_handler_node import error_handler_node

logger = logging.getLogger(__name__)


class StudioAgentRegistry:
    """Registry for managing agent nodes in LangGraph Studio."""
    
    def __init__(self):
        self._agents = {}
        self._agent_schemas = {}
        self._agent_metadata = {}
        self._register_default_agents()
    
    def _register_default_agents(self):
        """Register default agents from the workflow."""
        self.register_agent(
            name="planning_agent",
            node_function=planning_node,
            description="Generates scene outline and implementations for video topics",
            input_schema={
                "topic": {"type": "string", "required": True, "description": "Video topic"},
                "description": {"type": "string", "required": True, "description": "Video description"},
                "session_id": {"type": "string", "required": False, "description": "Session identifier"}
            },
            output_schema={
                "scene_outline": {"type": "string", "description": "Generated scene outline"},
                "scene_implementations": {"type": "object", "description": "Scene implementations by number"},
                "detected_plugins": {"type": "array", "description": "Detected Manim plugins"}
            },
            tags=["planning", "content-generation"]
        )
        
        self.register_agent(
            name="code_generation_agent",
            node_function=code_generation_node,
            description="Generates Manim code for video scenes",
            input_schema={
                "topic": {"type": "string", "required": True},
                "description": {"type": "string", "required": True},
                "scene_outline": {"type": "string", "required": True},
                "scene_implementations": {"type": "object", "required": True}
            },
            output_schema={
                "generated_code": {"type": "object", "description": "Generated code by scene number"},
                "code_errors": {"type": "object", "description": "Code generation errors"}
            },
            tags=["code-generation", "manim"]
        )
        
        self.register_agent(
            name="rendering_agent",
            node_function=rendering_node,
            description="Renders Manim code to video files",
            input_schema={
                "generated_code": {"type": "object", "required": True},
                "topic": {"type": "string", "required": True},
                "session_id": {"type": "string", "required": True}
            },
            output_schema={
                "rendered_videos": {"type": "object", "description": "Rendered video paths by scene"},
                "combined_video_path": {"type": "string", "description": "Final combined video path"},
                "rendering_errors": {"type": "object", "description": "Rendering errors"}
            },
            tags=["rendering", "video-processing"]
        )
        
        self.register_agent(
            name="error_handler_agent",
            node_function=error_handler_node,
            description="Handles and recovers from workflow errors",
            input_schema={
                "errors": {"type": "array", "required": True},
                "current_step": {"type": "string", "required": True}
            },
            output_schema={
                "recovery_action": {"type": "string", "description": "Recovery action taken"},
                "escalated_errors": {"type": "array", "description": "Errors escalated to human"}
            },
            tags=["error-handling", "recovery"]
        )
    
    def register_agent(
        self,
        name: str,
        node_function: callable,
        description: str,
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
        tags: List[str] = None
    ):
        """Register an agent for Studio testing."""
        self._agents[name] = node_function
        self._agent_schemas[name] = {
            "input": input_schema,
            "output": output_schema
        }
        self._agent_metadata[name] = {
            "description": description,
            "tags": tags or [],
            "registered_at": datetime.now().isoformat()
        }
        logger.info(f"Registered agent: {name}")
    
    def get_agent(self, name: str) -> Optional[callable]:
        """Get agent node function by name."""
        return self._agents.get(name)
    
    def get_agent_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """Get agent input/output schema."""
        return self._agent_schemas.get(name)
    
    def get_agent_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get agent metadata."""
        return self._agent_metadata.get(name)
    
    def list_agents(self) -> List[str]:
        """List all registered agent names."""
        return list(self._agents.keys())
    
    def get_agents_by_tag(self, tag: str) -> List[str]:
        """Get agents filtered by tag."""
        return [
            name for name, metadata in self._agent_metadata.items()
            if tag in metadata.get("tags", [])
        ]


class StudioAgentTester:
    """Test runner for individual agents in Studio environment."""
    
    def __init__(self, registry: StudioAgentRegistry, config: WorkflowConfig = None):
        self.registry = registry
        self.config = config or WorkflowConfig()
        self.test_results = {}
        self.execution_logs = []
    
    async def test_agent(
        self,
        agent_name: str,
        test_input: Dict[str, Any],
        test_scenario: str = "default"
    ) -> Dict[str, Any]:
        """Test an individual agent with given input."""
        logger.info(f"Testing agent: {agent_name} with scenario: {test_scenario}")
        
        # Get agent function
        agent_function = self.registry.get_agent(agent_name)
        if not agent_function:
            raise ValueError(f"Agent not found: {agent_name}")
        
        # Create test state
        test_state = self._create_test_state(test_input)
        
        # Record test start
        test_id = f"{agent_name}_{test_scenario}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        try:
            # Execute agent
            result_state = await agent_function(test_state)
            
            # Record successful execution
            execution_time = (datetime.now() - start_time).total_seconds()
            
            test_result = {
                "test_id": test_id,
                "agent_name": agent_name,
                "scenario": test_scenario,
                "status": "success",
                "execution_time": execution_time,
                "input": test_input,
                "output": self._extract_agent_output(result_state, agent_name),
                "state_changes": self._compare_states(test_state, result_state),
                "errors": result_state.errors if hasattr(result_state, 'errors') else [],
                "execution_trace": result_state.execution_trace if hasattr(result_state, 'execution_trace') else [],
                "timestamp": start_time.isoformat()
            }
            
            self.test_results[test_id] = test_result
            self.execution_logs.append({
                "test_id": test_id,
                "agent": agent_name,
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"Agent test completed successfully: {test_id}")
            return test_result
            
        except Exception as e:
            # Record failed execution
            execution_time = (datetime.now() - start_time).total_seconds()
            
            test_result = {
                "test_id": test_id,
                "agent_name": agent_name,
                "scenario": test_scenario,
                "status": "failed",
                "execution_time": execution_time,
                "input": test_input,
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": start_time.isoformat()
            }
            
            self.test_results[test_id] = test_result
            self.execution_logs.append({
                "test_id": test_id,
                "agent": agent_name,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            
            logger.error(f"Agent test failed: {test_id} - {str(e)}")
            return test_result
    
    def _create_test_state(self, test_input: Dict[str, Any]) -> VideoGenerationState:
        """Create a test state from input parameters."""
        # Set default values
        state_data = {
            "topic": test_input.get("topic", "Test Topic"),
            "description": test_input.get("description", "Test Description"),
            "session_id": test_input.get("session_id", f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            "config": self.config
        }
        
        # Add any additional input fields
        for key, value in test_input.items():
            if key not in state_data:
                state_data[key] = value
        
        return VideoGenerationState(**state_data)
    
    def _extract_agent_output(self, state: VideoGenerationState, agent_name: str) -> Dict[str, Any]:
        """Extract relevant output from state based on agent type."""
        output = {}
        
        if agent_name == "planning_agent":
            output = {
                "scene_outline": state.scene_outline,
                "scene_implementations": state.scene_implementations,
                "detected_plugins": state.detected_plugins
            }
        elif agent_name == "code_generation_agent":
            output = {
                "generated_code": state.generated_code,
                "code_errors": state.code_errors
            }
        elif agent_name == "rendering_agent":
            output = {
                "rendered_videos": state.rendered_videos,
                "combined_video_path": state.combined_video_path,
                "rendering_errors": state.rendering_errors
            }
        elif agent_name == "error_handler_agent":
            output = {
                "escalated_errors": state.escalated_errors,
                "retry_counts": state.retry_counts
            }
        
        # Always include common fields
        output.update({
            "current_step": state.current_step,
            "workflow_complete": state.workflow_complete,
            "has_errors": state.has_errors(),
            "error_count": len(state.errors)
        })
        
        return output
    
    def _compare_states(self, before: VideoGenerationState, after: VideoGenerationState) -> Dict[str, Any]:
        """Compare states to identify changes."""
        changes = {}
        
        # Compare key fields
        fields_to_compare = [
            "current_step", "workflow_complete", "scene_outline",
            "scene_implementations", "generated_code", "rendered_videos"
        ]
        
        for field in fields_to_compare:
            before_value = getattr(before, field, None)
            after_value = getattr(after, field, None)
            
            if before_value != after_value:
                changes[field] = {
                    "before": before_value,
                    "after": after_value
                }
        
        return changes
    
    def get_test_results(self, agent_name: str = None) -> Dict[str, Any]:
        """Get test results, optionally filtered by agent."""
        if agent_name:
            return {
                test_id: result for test_id, result in self.test_results.items()
                if result["agent_name"] == agent_name
            }
        return self.test_results
    
    def get_execution_logs(self) -> List[Dict[str, Any]]:
        """Get execution logs."""
        return self.execution_logs
    
    def clear_results(self):
        """Clear test results and logs."""
        self.test_results.clear()
        self.execution_logs.clear()


class StudioWorkflowBuilder:
    """Builder for creating Studio-compatible workflow graphs."""
    
    def __init__(self, registry: StudioAgentRegistry):
        self.registry = registry
    
    def create_single_agent_graph(self, agent_name: str) -> StateGraph:
        """Create a graph for testing a single agent."""
        agent_function = self.registry.get_agent(agent_name)
        if not agent_function:
            raise ValueError(f"Agent not found: {agent_name}")
        
        # Create simple graph with just the agent
        graph = StateGraph(VideoGenerationState)
        graph.add_node(agent_name, agent_function)
        graph.add_edge(START, agent_name)
        graph.add_edge(agent_name, END)
        
        return graph
    
    def create_agent_chain_graph(self, agent_names: List[str]) -> StateGraph:
        """Create a graph for testing a chain of agents."""
        graph = StateGraph(VideoGenerationState)
        
        # Add all agents
        for agent_name in agent_names:
            agent_function = self.registry.get_agent(agent_name)
            if not agent_function:
                raise ValueError(f"Agent not found: {agent_name}")
            graph.add_node(agent_name, agent_function)
        
        # Chain agents together
        graph.add_edge(START, agent_names[0])
        for i in range(len(agent_names) - 1):
            graph.add_edge(agent_names[i], agent_names[i + 1])
        graph.add_edge(agent_names[-1], END)
        
        return graph
    
    def create_full_workflow_graph(self) -> StateGraph:
        """Create the full workflow graph for Studio."""
        # Use the existing workflow but make it Studio-compatible
        config = WorkflowConfig()
        workflow = VideoGenerationWorkflow(config)
        return workflow.graph


class StudioMonitor:
    """Monitor for tracking agent execution in Studio."""
    
    def __init__(self):
        self.execution_history = []
        self.performance_metrics = {}
        self.active_sessions = {}
    
    def start_session(self, session_id: str, agent_name: str) -> None:
        """Start monitoring a session."""
        self.active_sessions[session_id] = {
            "agent": agent_name,
            "start_time": datetime.now(),
            "status": "running"
        }
    
    def end_session(self, session_id: str, status: str = "completed") -> None:
        """End monitoring a session."""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session["end_time"] = datetime.now()
            session["status"] = status
            session["duration"] = (session["end_time"] - session["start_time"]).total_seconds()
            
            # Move to history
            self.execution_history.append(session)
            del self.active_sessions[session_id]
    
    def record_performance_metric(self, agent_name: str, metric_name: str, value: Any) -> None:
        """Record a performance metric."""
        if agent_name not in self.performance_metrics:
            self.performance_metrics[agent_name] = {}
        
        if metric_name not in self.performance_metrics[agent_name]:
            self.performance_metrics[agent_name][metric_name] = []
        
        self.performance_metrics[agent_name][metric_name].append({
            "value": value,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an active session."""
        return self.active_sessions.get(session_id)
    
    def get_execution_history(self, agent_name: str = None) -> List[Dict[str, Any]]:
        """Get execution history, optionally filtered by agent."""
        if agent_name:
            return [h for h in self.execution_history if h["agent"] == agent_name]
        return self.execution_history
    
    def get_performance_metrics(self, agent_name: str = None) -> Dict[str, Any]:
        """Get performance metrics."""
        if agent_name:
            return self.performance_metrics.get(agent_name, {})
        return self.performance_metrics


# Global instances for Studio integration
studio_registry = StudioAgentRegistry()
studio_monitor = StudioMonitor()


def create_studio_tester(config: WorkflowConfig = None) -> StudioAgentTester:
    """Create a Studio agent tester instance."""
    return StudioAgentTester(studio_registry, config)


def create_studio_workflow_builder() -> StudioWorkflowBuilder:
    """Create a Studio workflow builder instance."""
    return StudioWorkflowBuilder(studio_registry)
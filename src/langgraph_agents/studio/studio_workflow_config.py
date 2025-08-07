"""
Studio-compatible workflow graph configuration.

This module provides the configuration and setup for making the workflow graph
visible and executable in LangGraph Studio with proper input/output schemas,
state inspection, and debugging capabilities.
"""

import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
# CompiledGraph type is not available in this version of langgraph

from ..models.state import VideoGenerationState
from ..models.config import WorkflowConfig
from ..workflow_graph import VideoGenerationWorkflow, create_workflow
from .studio_config import get_studio_config
from .studio_integration import studio_registry, studio_monitor
from .studio_workflow_visualization import get_studio_visualizer, get_studio_inspector

logger = logging.getLogger(__name__)


class StudioWorkflowConfig:
    """Configuration manager for Studio-compatible workflow graphs."""
    
    def __init__(self):
        self.studio_config = get_studio_config()
        self.workflow_config = self.studio_config.get_workflow_config()
        self.graph_schemas = {}
        self.debugging_enabled = True
        self.state_inspection_enabled = True
        self.server_integration_config = None
        self._initialize_schemas()
        
    def create_studio_compatible_workflow(self) -> VideoGenerationWorkflow:
        """Create a Studio-compatible workflow with enhanced monitoring."""
        logger.info("Creating Studio-compatible workflow")
        
        # Create workflow with memory checkpointer for Studio
        workflow = create_workflow(
            config=self.workflow_config,
            use_checkpointing=True,
            checkpoint_config=None  # Will use memory checkpointer
        )
        
        # Enhance with Studio-specific features
        self._add_studio_enhancements(workflow)
        
        return workflow
    
    def _add_studio_enhancements(self, workflow: VideoGenerationWorkflow) -> None:
        """Add Studio-specific enhancements to the workflow."""
        # Store original graph for reference
        original_graph = workflow.graph
        
        # Create enhanced graph with Studio features
        enhanced_graph = self._create_enhanced_graph(workflow)
        
        # Replace the workflow's graph
        workflow.graph = enhanced_graph
        
        logger.info("Studio enhancements added to workflow")
    
    def _initialize_schemas(self):
        """Initialize input/output schemas for Studio visualization."""
        self.graph_schemas = {
            "workflow": {
                "input_schema": {
                    "type": "object",
                    "title": "Video Generation Workflow Input",
                    "description": "Input parameters for the complete video generation workflow",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "title": "Video Topic",
                            "description": "The main topic or subject for the video",
                            "examples": ["Linear Equations", "Pythagorean Theorem", "Quantum Mechanics"],
                            "minLength": 3,
                            "maxLength": 200
                        },
                        "description": {
                            "type": "string", 
                            "title": "Video Description",
                            "description": "Detailed description of what the video should cover",
                            "examples": [
                                "Explain how to solve linear equations step by step with examples",
                                "Demonstrate the Pythagorean theorem with geometric proofs"
                            ],
                            "minLength": 10,
                            "maxLength": 1000
                        },
                        "session_id": {
                            "type": "string",
                            "title": "Session ID",
                            "description": "Unique identifier for this workflow session",
                            "pattern": "^[a-zA-Z0-9_-]+$",
                            "default": "auto-generated"
                        },
                        "preview_mode": {
                            "type": "boolean",
                            "title": "Preview Mode",
                            "description": "Enable preview mode for faster, lower quality output",
                            "default": True
                        },
                        "max_scenes": {
                            "type": "integer",
                            "title": "Maximum Scenes",
                            "description": "Maximum number of scenes to generate",
                            "minimum": 1,
                            "maximum": 10,
                            "default": 5
                        }
                    },
                    "required": ["topic", "description"],
                    "additionalProperties": False
                },
                "output_schema": {
                    "type": "object",
                    "title": "Video Generation Workflow Output",
                    "description": "Complete output from the video generation workflow",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session identifier for this workflow execution"
                        },
                        "workflow_complete": {
                            "type": "boolean",
                            "description": "Whether the workflow completed successfully"
                        },
                        "current_step": {
                            "type": "string",
                            "description": "Current or final step in the workflow",
                            "enum": ["planning", "code_generation", "rendering", "complete", "error_handler", "human_loop"]
                        },
                        "scene_outline": {
                            "type": "string",
                            "description": "Generated outline of video scenes"
                        },
                        "scene_implementations": {
                            "type": "object",
                            "description": "Detailed implementations for each scene",
                            "patternProperties": {
                                "^[0-9]+$": {
                                    "type": "string",
                                    "description": "Implementation details for scene number"
                                }
                            }
                        },
                        "generated_code": {
                            "type": "object",
                            "description": "Generated Manim code for each scene",
                            "patternProperties": {
                                "^[0-9]+$": {
                                    "type": "string",
                                    "description": "Manim code for scene number"
                                }
                            }
                        },
                        "rendered_videos": {
                            "type": "object",
                            "description": "Paths to rendered video files for each scene",
                            "patternProperties": {
                                "^[0-9]+$": {
                                    "type": "string",
                                    "description": "File path to rendered video for scene number"
                                }
                            }
                        },
                        "combined_video_path": {
                            "type": "string",
                            "description": "Path to the final combined video file"
                        },
                        "completion_percentage": {
                            "type": "number",
                            "description": "Workflow completion percentage (0-100)",
                            "minimum": 0,
                            "maximum": 100
                        },
                        "errors": {
                            "type": "array",
                            "description": "List of errors encountered during execution",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "step": {"type": "string"},
                                    "error_type": {"type": "string"},
                                    "message": {"type": "string"},
                                    "severity": {"type": "string"}
                                }
                            }
                        },
                        "execution_trace": {
                            "type": "array",
                            "description": "Detailed execution trace for debugging",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "step": {"type": "string"},
                                    "timestamp": {"type": "string"},
                                    "data": {"type": "object"}
                                }
                            }
                        },
                        "performance_metrics": {
                            "type": "object",
                            "description": "Performance metrics for the workflow execution",
                            "properties": {
                                "total_execution_time": {"type": "number"},
                                "planning_time": {"type": "number"},
                                "code_generation_time": {"type": "number"},
                                "rendering_time": {"type": "number"}
                            }
                        }
                    },
                    "additionalProperties": True
                }
            },
            "nodes": self._get_node_schemas()
        }
    
    def _create_enhanced_graph(self, workflow: VideoGenerationWorkflow):
        """Create an enhanced graph with Studio monitoring and debugging."""
        # Create new StateGraph with enhanced nodes
        graph_builder = StateGraph(VideoGenerationState)
        
        # Add enhanced nodes with monitoring
        self._add_enhanced_nodes(graph_builder, workflow)
        
        # Add conditional edges (same as original)
        self._add_conditional_edges(graph_builder, workflow)
        
        # Set entry point
        graph_builder.add_edge(START, "planning")
        
        # Compile with memory checkpointer for Studio
        checkpointer = MemorySaver()
        enhanced_graph = graph_builder.compile(checkpointer=checkpointer)
        
        return enhanced_graph
    
    def _add_enhanced_nodes(self, graph_builder: StateGraph, workflow: VideoGenerationWorkflow) -> None:
        """Add enhanced nodes with Studio monitoring to the graph."""
        # Get original node functions
        from ..nodes.planning_node import planning_node
        from ..nodes.code_generation_node import code_generation_node
        from ..nodes.rendering_node import rendering_node
        from ..nodes.error_handler_node import error_handler_node
        
        # Create enhanced versions of each node
        enhanced_planning = self._create_enhanced_node("planning", planning_node)
        enhanced_code_gen = self._create_enhanced_node("code_generation", code_generation_node)
        enhanced_rendering = self._create_enhanced_node("rendering", rendering_node)
        enhanced_error_handler = self._create_enhanced_node("error_handler", error_handler_node)
        
        # Add enhanced nodes to graph
        graph_builder.add_node("planning", enhanced_planning)
        graph_builder.add_node("code_generation", enhanced_code_gen)
        graph_builder.add_node("rendering", enhanced_rendering)
        graph_builder.add_node("error_handler", enhanced_error_handler)
        
        # Add placeholder nodes for future implementation
        graph_builder.add_node("rag_enhancement", self._create_enhanced_node("rag_enhancement", workflow._rag_enhancement_node))
        graph_builder.add_node("visual_analysis", self._create_enhanced_node("visual_analysis", workflow._visual_analysis_node))
        graph_builder.add_node("human_loop", self._create_enhanced_node("human_loop", workflow._human_loop_node))
        graph_builder.add_node("complete", self._create_enhanced_node("complete", workflow._complete_node))
    
    def _create_enhanced_node(self, node_name: str, original_function: callable) -> callable:
        """Create an enhanced node function with Studio monitoring."""
        async def enhanced_node_function(state: VideoGenerationState) -> VideoGenerationState:
            session_id = state.session_id
            
            # Start node monitoring
            studio_monitor.start_session(f"{session_id}_{node_name}", node_name)
            
            # Record node start metrics
            studio_monitor.record_performance_metric(
                node_name,
                "node_started",
                {
                    "session_id": session_id,
                    "current_step": state.current_step,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Add state inspection point (before execution)
            if self.state_inspection_enabled:
                self._record_state_inspection(node_name, "before", state)
                # Capture detailed state snapshot for Studio
                get_studio_inspector().capture_state_snapshot(state, node_name, "before")
            
            try:
                # Execute original node function
                start_time = datetime.now()
                result_state = await original_function(state)
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # Record successful execution metrics
                studio_monitor.record_performance_metric(
                    node_name,
                    "execution_time",
                    execution_time
                )
                
                studio_monitor.record_performance_metric(
                    node_name,
                    "execution_success",
                    True
                )
                
                # Add state inspection point (after execution)
                if self.state_inspection_enabled:
                    self._record_state_inspection(node_name, "after", result_state)
                    # Capture detailed state snapshot for Studio
                    get_studio_inspector().capture_state_snapshot(result_state, node_name, "after")
                
                # Record node completion
                studio_monitor.record_performance_metric(
                    node_name,
                    "node_completed",
                    {
                        "session_id": session_id,
                        "execution_time": execution_time,
                        "has_errors": result_state.has_errors(),
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                # End node monitoring with success
                studio_monitor.end_session(f"{session_id}_{node_name}", "completed")
                
                return result_state
                
            except Exception as e:
                # Record error metrics
                studio_monitor.record_performance_metric(
                    node_name,
                    "execution_success",
                    False
                )
                
                studio_monitor.record_performance_metric(
                    node_name,
                    "error_details",
                    {
                        "error_message": str(e),
                        "error_type": type(e).__name__,
                        "session_id": session_id,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                # End node monitoring with failure
                studio_monitor.end_session(f"{session_id}_{node_name}", "failed")
                
                # Re-raise the exception
                raise
        
        return enhanced_node_function
    
    def _add_conditional_edges(self, graph_builder: StateGraph, workflow: VideoGenerationWorkflow) -> None:
        """Add conditional edges to the enhanced graph (same as original workflow)."""
        # Import routing functions
        from ..routing import (
            route_from_planning,
            route_from_code_generation,
            route_from_rendering,
            route_from_error_handler
        )
        
        # Planning node routing
        graph_builder.add_conditional_edges(
            "planning",
            route_from_planning,
            {
                "code_generation": "code_generation",
                "error_handler": "error_handler",
                "human_loop": "human_loop"
            }
        )
        
        # Code generation node routing
        graph_builder.add_conditional_edges(
            "code_generation",
            route_from_code_generation,
            {
                "rendering": "rendering",
                "rag_enhancement": "rag_enhancement",
                "error_handler": "error_handler",
                "human_loop": "human_loop"
            }
        )
        
        # RAG enhancement routing
        graph_builder.add_conditional_edges(
            "rag_enhancement",
            workflow._route_from_rag_enhancement,
            {
                "code_generation": "code_generation",
                "error_handler": "error_handler",
                "human_loop": "human_loop"
            }
        )
        
        # Rendering node routing
        graph_builder.add_conditional_edges(
            "rendering",
            route_from_rendering,
            {
                "visual_analysis": "visual_analysis",
                "complete": "complete",
                "error_handler": "error_handler",
                "human_loop": "human_loop"
            }
        )
        
        # Visual analysis routing
        graph_builder.add_conditional_edges(
            "visual_analysis",
            workflow._route_from_visual_analysis,
            {
                "complete": "complete",
                "error_handler": "error_handler",
                "human_loop": "human_loop"
            }
        )
        
        # Error handler routing
        graph_builder.add_conditional_edges(
            "error_handler",
            route_from_error_handler,
            {
                "planning": "planning",
                "code_generation": "code_generation",
                "rendering": "rendering",
                "human_loop": "human_loop",
                "complete": "complete"
            }
        )
        
        # Human loop routing
        graph_builder.add_conditional_edges(
            "human_loop",
            workflow._route_from_human_loop,
            {
                "planning": "planning",
                "code_generation": "code_generation",
                "rendering": "rendering",
                "complete": "complete"
            }
        )
        
        # Complete node is terminal
        graph_builder.add_edge("complete", END)
    
    def _record_state_inspection(self, node_name: str, phase: str, state: VideoGenerationState) -> None:
        """Record state inspection data for debugging."""
        inspection_data = {
            "node": node_name,
            "phase": phase,
            "session_id": state.session_id,
            "current_step": state.current_step,
            "workflow_complete": state.workflow_complete,
            "workflow_interrupted": getattr(state, 'workflow_interrupted', False),
            "has_errors": state.has_errors(),
            "error_count": len(state.errors),
            "scene_count": len(state.scene_implementations) if state.scene_implementations else 0,
            "code_count": len(state.generated_code) if state.generated_code else 0,
            "video_count": len(state.rendered_videos) if state.rendered_videos else 0,
            "completion_percentage": state.get_completion_percentage() if hasattr(state, 'get_completion_percentage') else 0,
            "memory_usage": self._get_memory_usage(),
            "execution_time": self._get_execution_time(state),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add detailed state snapshot for debugging
        if self.debugging_enabled:
            inspection_data["state_snapshot"] = {
                "topic": state.topic,
                "description": state.description[:100] + "..." if len(state.description) > 100 else state.description,
                "scene_outline_length": len(state.scene_outline) if state.scene_outline else 0,
                "detected_plugins": state.detected_plugins if hasattr(state, 'detected_plugins') else [],
                "retry_counts": getattr(state, 'retry_counts', {}),
                "escalated_errors": len(getattr(state, 'escalated_errors', [])),
                "pending_human_input": bool(getattr(state, 'pending_human_input', None))
            }
        
        studio_monitor.record_performance_metric(
            node_name,
            f"state_inspection_{phase}",
            inspection_data
        )
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage for monitoring."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss": memory_info.rss,
                "vms": memory_info.vms,
                "percent": process.memory_percent()
            }
        except ImportError:
            return {"error": "psutil not available"}
        except Exception as e:
            return {"error": str(e)}
    
    def _get_execution_time(self, state: VideoGenerationState) -> float:
        """Get execution time for the current state."""
        if hasattr(state, 'execution_trace') and state.execution_trace:
            # Find the workflow start time
            start_trace = next((trace for trace in state.execution_trace if trace.get('step') == 'workflow_start'), None)
            if start_trace:
                try:
                    from datetime import datetime
                    start_time = datetime.fromisoformat(start_trace['timestamp'])
                    return (datetime.now() - start_time).total_seconds()
                except:
                    pass
        return 0.0
    
    def create_workflow_state_inspector(self) -> Dict[str, Any]:
        """Create workflow state inspection capabilities for Studio."""
        return {
            "inspection_enabled": self.state_inspection_enabled,
            "debugging_enabled": self.debugging_enabled,
            "inspection_points": [
                "node_entry",
                "node_exit", 
                "error_occurrence",
                "state_transition",
                "checkpoint_creation"
            ],
            "captured_metrics": [
                "execution_time",
                "memory_usage",
                "error_count",
                "completion_percentage",
                "state_changes"
            ],
            "debugging_features": [
                "state_snapshots",
                "execution_trace",
                "performance_metrics",
                "error_analysis",
                "checkpoint_inspection"
            ]
        }
    
    def get_workflow_schema(self) -> Dict[str, Any]:
        """Get the workflow input/output schema for Studio."""
        return {
            "input_schema": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic for the video to be generated",
                        "required": True
                    },
                    "description": {
                        "type": "string", 
                        "description": "Detailed description of the video content",
                        "required": True
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Unique session identifier",
                        "required": False,
                        "default": "auto-generated"
                    },
                    "preview_mode": {
                        "type": "boolean",
                        "description": "Whether to run in preview mode (faster, lower quality)",
                        "required": False,
                        "default": True
                    }
                },
                "required": ["topic", "description"]
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session identifier"
                    },
                    "workflow_complete": {
                        "type": "boolean",
                        "description": "Whether the workflow completed successfully"
                    },
                    "scene_outline": {
                        "type": "string",
                        "description": "Generated scene outline"
                    },
                    "scene_implementations": {
                        "type": "object",
                        "description": "Scene implementations by number"
                    },
                    "generated_code": {
                        "type": "object",
                        "description": "Generated Manim code by scene"
                    },
                    "rendered_videos": {
                        "type": "object",
                        "description": "Rendered video file paths by scene"
                    },
                    "combined_video_path": {
                        "type": "string",
                        "description": "Path to the final combined video"
                    },
                    "completion_percentage": {
                        "type": "number",
                        "description": "Workflow completion percentage"
                    },
                    "errors": {
                        "type": "array",
                        "description": "List of errors encountered"
                    },
                    "execution_trace": {
                        "type": "array",
                        "description": "Execution trace for debugging"
                    }
                }
            },
            "node_schemas": self._get_node_schemas()
        }
    
    def _get_node_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get input/output schemas for individual nodes."""
        return {
            "planning": {
                "input": {
                    "topic": {"type": "string", "required": True},
                    "description": {"type": "string", "required": True}
                },
                "output": {
                    "scene_outline": {"type": "string"},
                    "scene_implementations": {"type": "object"},
                    "detected_plugins": {"type": "array"}
                }
            },
            "code_generation": {
                "input": {
                    "scene_outline": {"type": "string", "required": True},
                    "scene_implementations": {"type": "object", "required": True}
                },
                "output": {
                    "generated_code": {"type": "object"},
                    "code_errors": {"type": "object"}
                }
            },
            "rendering": {
                "input": {
                    "generated_code": {"type": "object", "required": True},
                    "session_id": {"type": "string", "required": True}
                },
                "output": {
                    "rendered_videos": {"type": "object"},
                    "combined_video_path": {"type": "string"},
                    "rendering_errors": {"type": "object"}
                }
            },
            "error_handler": {
                "input": {
                    "errors": {"type": "array", "required": True},
                    "current_step": {"type": "string", "required": True}
                },
                "output": {
                    "recovery_action": {"type": "string"},
                    "escalated_errors": {"type": "array"}
                }
            }
        }
    
    def get_debugging_info(self) -> Dict[str, Any]:
        """Get debugging information for Studio."""
        return {
            "debugging_enabled": self.debugging_enabled,
            "state_inspection_enabled": self.state_inspection_enabled,
            "checkpoint_info": {
                "checkpointer_type": "MemorySaver",
                "persistent": False,
                "studio_compatible": True
            },
            "monitoring_info": {
                "active_sessions": len(studio_monitor.active_sessions),
                "execution_history_count": len(studio_monitor.execution_history),
                "performance_metrics_count": len(studio_monitor.performance_metrics)
            },
            "workflow_config": {
                "max_retries": self.workflow_config.max_retries,
                "timeout_seconds": self.workflow_config.timeout_seconds,
                "preview_mode": self.workflow_config.preview_mode,
                "output_dir": self.workflow_config.output_dir
            }
        }
    
    def create_studio_server_integration(self) -> Dict[str, Any]:
        """Create Studio server integration configuration."""
        if not self.server_integration_config:
            self.server_integration_config = {
                "server_config": {
                    "host": "0.0.0.0",
                    "port": 8123,
                    "cors_enabled": True,
                    "debug_mode": True,
                    "auto_reload": True,
                    "log_level": "INFO"
                },
                "graph_endpoints": {
                    "main_workflow": "/graphs/workflow",
                    "planning_agent": "/graphs/planning",
                    "code_generation_agent": "/graphs/code_generation", 
                    "rendering_agent": "/graphs/rendering",
                    "error_handler_agent": "/graphs/error_handler",
                    "agent_chains": "/graphs/chains",
                    "monitored_agents": "/graphs/monitored"
                },
                "monitoring_endpoints": {
                    "metrics": "/api/monitoring/metrics",
                    "sessions": "/api/monitoring/sessions",
                    "history": "/api/monitoring/history",
                    "state": "/api/monitoring/state",
                    "performance": "/api/monitoring/performance",
                    "health": "/api/monitoring/health"
                },
                "debugging_endpoints": {
                    "inspect": "/api/debug/inspect",
                    "trace": "/api/debug/trace",
                    "checkpoints": "/api/debug/checkpoints",
                    "state_history": "/api/debug/state-history",
                    "execution_logs": "/api/debug/execution-logs"
                },
                "schema_endpoints": {
                    "workflow_schema": "/api/schemas/workflow",
                    "node_schemas": "/api/schemas/nodes",
                    "state_schema": "/api/schemas/state"
                },
                "test_endpoints": {
                    "scenarios": "/api/test/scenarios",
                    "fixtures": "/api/test/fixtures",
                    "validate": "/api/test/validate",
                    "run_scenario": "/api/test/run"
                }
            }
        
        return self.server_integration_config
    
    def setup_studio_server_integration(self) -> Dict[str, Any]:
        """Set up the Studio server integration with the existing backend."""
        logger.info("Setting up Studio server integration")
        
        integration_config = self.create_studio_server_integration()
        
        # Configure CORS for Studio
        cors_config = {
            "allow_origins": [
                "http://localhost:3000",  # Studio frontend
                "http://127.0.0.1:3000",
                "http://localhost:8123",  # Studio server
                "http://127.0.0.1:8123"
            ],
            "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
            "allow_headers": [
                "Content-Type",
                "Authorization", 
                "X-Session-ID",
                "X-Trace-ID",
                "X-Test-Scenario",
                "X-Debug-Mode"
            ],
            "allow_credentials": True
        }
        
        # Configure WebSocket support for real-time monitoring
        websocket_config = {
            "enabled": True,
            "path": "/ws",
            "heartbeat_interval": 30,
            "max_connections": 100
        }
        
        # Configure authentication for Studio access
        auth_config = {
            "enabled": False,  # Disabled for local testing
            "type": "bearer_token",
            "token_header": "Authorization",
            "allowed_tokens": []
        }
        
        setup_result = {
            "status": "configured",
            "integration_config": integration_config,
            "cors_config": cors_config,
            "websocket_config": websocket_config,
            "auth_config": auth_config,
            "graph_count": len(integration_config["graph_endpoints"]),
            "endpoint_count": (
                len(integration_config["monitoring_endpoints"]) +
                len(integration_config["debugging_endpoints"]) +
                len(integration_config["schema_endpoints"]) +
                len(integration_config["test_endpoints"])
            )
        }
        
        logger.info(f"Studio server integration configured with {setup_result['endpoint_count']} endpoints")
        return setup_result


def create_studio_workflow_config() -> StudioWorkflowConfig:
    """Create a Studio workflow configuration instance."""
    return StudioWorkflowConfig()


def create_studio_compatible_graph():
    """Create a Studio-compatible workflow graph."""
    config = create_studio_workflow_config()
    workflow = config.create_studio_compatible_workflow()
    return workflow.graph


# Create the main Studio-compatible graph
studio_workflow_config = create_studio_workflow_config()
studio_compatible_workflow = studio_workflow_config.create_studio_compatible_workflow()

# Export the graph for Studio
graph = studio_compatible_workflow.graph


def get_studio_workflow_info() -> Dict[str, Any]:
    """Get comprehensive information about the Studio workflow."""
    visualizer = get_studio_visualizer()
    
    return {
        "workflow_type": "StudioCompatibleVideoGenerationWorkflow",
        "schema": studio_workflow_config.get_workflow_schema(),
        "debugging": studio_workflow_config.get_debugging_info(),
        "server_integration": studio_workflow_config.create_studio_server_integration(),
        "graph_visualization": studio_compatible_workflow.get_graph_visualization(),
        "checkpoint_info": studio_compatible_workflow.get_checkpoint_info(),
        "studio_visualization": {
            "mermaid_diagram": visualizer.generate_mermaid_diagram(),
            "graph_metadata": visualizer.create_studio_graph_metadata(),
            "node_configurations": {
                node: visualizer.get_node_configuration(node)
                for node in ["planning", "code_generation", "rendering", "error_handler", 
                           "rag_enhancement", "visual_analysis", "human_loop", "complete"]
            }
        },
        "state_inspection": studio_workflow_config.create_workflow_state_inspector()
    }


def test_studio_workflow_setup() -> Dict[str, Any]:
    """Test the Studio workflow setup."""
    try:
        # Test workflow creation
        workflow = studio_workflow_config.create_studio_compatible_workflow()
        
        # Test schema generation
        schema = studio_workflow_config.get_workflow_schema()
        
        # Test debugging info
        debug_info = studio_workflow_config.get_debugging_info()
        
        return {
            "status": "success",
            "workflow_created": True,
            "graph_compiled": workflow.graph is not None,
            "schema_generated": bool(schema),
            "debugging_enabled": debug_info["debugging_enabled"],
            "monitoring_active": debug_info["monitoring_info"]["active_sessions"] >= 0,
            "checkpointer_type": debug_info["checkpoint_info"]["checkpointer_type"]
        }
        
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "error_type": type(e).__name__
        }


if __name__ == "__main__":
    # Test the Studio workflow configuration
    print("Testing Studio workflow configuration...")
    
    test_result = test_studio_workflow_setup()
    
    if test_result["status"] == "success":
        print("✅ Studio workflow configuration successful")
        print(f"   - Workflow created: {test_result['workflow_created']}")
        print(f"   - Graph compiled: {test_result['graph_compiled']}")
        print(f"   - Schema generated: {test_result['schema_generated']}")
        print(f"   - Debugging enabled: {test_result['debugging_enabled']}")
        print(f"   - Checkpointer: {test_result['checkpointer_type']}")
        
        # Get and display workflow info
        workflow_info = get_studio_workflow_info()
        print(f"\n📊 Workflow Information:")
        print(f"   - Type: {workflow_info['workflow_type']}")
        print(f"   - Input schema properties: {len(workflow_info['schema']['input_schema']['properties'])}")
        print(f"   - Output schema properties: {len(workflow_info['schema']['output_schema']['properties'])}")
        print(f"   - Node schemas: {len(workflow_info['schema']['node_schemas'])}")
        
    else:
        print(f"❌ Studio workflow configuration failed: {test_result['error']}")
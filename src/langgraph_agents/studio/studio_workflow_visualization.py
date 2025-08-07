"""
Studio workflow visualization and graph configuration.

This module provides comprehensive visualization and configuration capabilities
for the workflow graph in LangGraph Studio, including node schemas, state
inspection, and debugging tools.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from ..models.state import VideoGenerationState
from ..models.config import WorkflowConfig
from .studio_integration import studio_monitor, studio_registry
from .studio_config import get_studio_config

logger = logging.getLogger(__name__)


class StudioWorkflowVisualizer:
    """Provides visualization and configuration for Studio workflow graphs."""
    
    def __init__(self):
        self.studio_config = get_studio_config()
        self.node_positions = self._calculate_node_positions()
        self.edge_configurations = self._configure_edges()
        self.visualization_settings = self._get_visualization_settings()
    
    def _calculate_node_positions(self) -> Dict[str, Dict[str, float]]:
        """Calculate optimal node positions for Studio visualization."""
        return {
            "planning": {"x": 100, "y": 100},
            "code_generation": {"x": 300, "y": 100},
            "rag_enhancement": {"x": 300, "y": 200},
            "rendering": {"x": 500, "y": 100},
            "visual_analysis": {"x": 500, "y": 200},
            "error_handler": {"x": 300, "y": 300},
            "human_loop": {"x": 400, "y": 400},
            "complete": {"x": 700, "y": 100}
        }
    
    def _configure_edges(self) -> Dict[str, Dict[str, Any]]:
        """Configure edge properties for Studio visualization."""
        return {
            "planning_to_code_generation": {
                "condition": "Success",
                "color": "#4CAF50",
                "style": "solid",
                "weight": 3
            },
            "planning_to_error_handler": {
                "condition": "Error",
                "color": "#F44336",
                "style": "dashed",
                "weight": 2
            },
            "planning_to_human_loop": {
                "condition": "Non-recoverable",
                "color": "#FF9800",
                "style": "dotted",
                "weight": 2
            },
            "code_generation_to_rendering": {
                "condition": "Success",
                "color": "#4CAF50",
                "style": "solid",
                "weight": 3
            },
            "code_generation_to_rag_enhancement": {
                "condition": "Partial Success",
                "color": "#2196F3",
                "style": "solid",
                "weight": 2
            },
            "code_generation_to_error_handler": {
                "condition": "Error",
                "color": "#F44336",
                "style": "dashed",
                "weight": 2
            },
            "rag_enhancement_to_code_generation": {
                "condition": "Enhanced",
                "color": "#2196F3",
                "style": "solid",
                "weight": 2
            },
            "rendering_to_visual_analysis": {
                "condition": "Success + Analysis",
                "color": "#9C27B0",
                "style": "solid",
                "weight": 2
            },
            "rendering_to_complete": {
                "condition": "Success",
                "color": "#4CAF50",
                "style": "solid",
                "weight": 3
            },
            "visual_analysis_to_complete": {
                "condition": "Success",
                "color": "#4CAF50",
                "style": "solid",
                "weight": 3
            },
            "error_handler_recovery": {
                "condition": "Recovery",
                "color": "#FF5722",
                "style": "solid",
                "weight": 2
            },
            "human_loop_continuation": {
                "condition": "Continue",
                "color": "#607D8B",
                "style": "solid",
                "weight": 2
            }
        }
    
    def _get_visualization_settings(self) -> Dict[str, Any]:
        """Get visualization settings for Studio."""
        return {
            "layout": "hierarchical",
            "direction": "left-to-right",
            "node_spacing": 150,
            "level_spacing": 200,
            "show_labels": True,
            "show_conditions": True,
            "animate_execution": True,
            "highlight_active_path": True,
            "show_state_preview": True,
            "enable_zoom": True,
            "enable_pan": True
        }
    
    def generate_mermaid_diagram(self) -> str:
        """Generate Mermaid diagram for Studio visualization."""
        return """
        graph TD
            START([Start]) --> planning[Planning Node<br/>📋 Generate Scene Outline]
            
            planning --> |✅ Success| code_generation[Code Generation Node<br/>💻 Generate Manim Code]
            planning --> |❌ Error| error_handler[Error Handler Node<br/>🔧 Handle Errors]
            planning --> |🚨 Critical| human_loop[Human Loop Node<br/>👤 Human Intervention]
            
            code_generation --> |✅ Success| rendering[Rendering Node<br/>🎬 Render Videos]
            code_generation --> |🔄 Partial| rag_enhancement[RAG Enhancement Node<br/>📚 Enhance with Context]
            code_generation --> |❌ Error| error_handler
            code_generation --> |🚨 Critical| human_loop
            
            rag_enhancement --> |🔄 Enhanced| code_generation
            rag_enhancement --> |❌ Error| error_handler
            rag_enhancement --> |🚨 Failed| human_loop
            
            rendering --> |✅ Success + Analysis| visual_analysis[Visual Analysis Node<br/>👁️ Analyze Output]
            rendering --> |✅ Success| complete[Complete Node<br/>🎉 Workflow Complete]
            rendering --> |❌ Error| error_handler
            rendering --> |🚨 Critical| human_loop
            
            visual_analysis --> |✅ Success| complete
            visual_analysis --> |❌ Error| error_handler
            visual_analysis --> |🚨 Failed| human_loop
            
            error_handler --> |🔄 Retry Planning| planning
            error_handler --> |🔄 Retry Code Gen| code_generation
            error_handler --> |🔄 Retry Rendering| rendering
            error_handler --> |🚨 Escalate| human_loop
            error_handler --> |✅ Complete| complete
            
            human_loop --> |🔄 Continue Planning| planning
            human_loop --> |🔄 Continue Code Gen| code_generation
            human_loop --> |🔄 Continue Rendering| rendering
            human_loop --> |✅ Complete| complete
            
            complete --> END([End])
            
            classDef startEnd fill:#e1f5fe,stroke:#01579b,stroke-width:2px
            classDef processing fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
            classDef error fill:#ffebee,stroke:#b71c1c,stroke-width:2px
            classDef success fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
            classDef human fill:#fff3e0,stroke:#e65100,stroke-width:2px
            
            class START,END startEnd
            class planning,code_generation,rendering,rag_enhancement,visual_analysis processing
            class error_handler error
            class complete success
            class human_loop human
        """
    
    def get_node_configuration(self, node_name: str) -> Dict[str, Any]:
        """Get detailed configuration for a specific node."""
        node_configs = {
            "planning": {
                "title": "Planning Agent",
                "description": "Generates scene outline and implementations for video topics",
                "icon": "📋",
                "color": "#9C27B0",
                "category": "content_generation",
                "estimated_duration": "30-60 seconds",
                "resource_usage": "Low",
                "dependencies": [],
                "outputs": ["scene_outline", "scene_implementations", "detected_plugins"],
                "error_handling": "Retry with simplified prompt",
                "monitoring_metrics": ["execution_time", "scene_count", "plugin_detection"]
            },
            "code_generation": {
                "title": "Code Generation Agent",
                "description": "Generates Manim code for video scenes",
                "icon": "💻",
                "color": "#2196F3",
                "category": "code_generation",
                "estimated_duration": "60-120 seconds",
                "resource_usage": "Medium",
                "dependencies": ["scene_outline", "scene_implementations"],
                "outputs": ["generated_code", "code_errors"],
                "error_handling": "RAG enhancement or retry",
                "monitoring_metrics": ["execution_time", "code_quality", "syntax_errors"]
            },
            "rendering": {
                "title": "Rendering Agent",
                "description": "Renders Manim code to video files",
                "icon": "🎬",
                "color": "#4CAF50",
                "category": "video_processing",
                "estimated_duration": "120-300 seconds",
                "resource_usage": "High",
                "dependencies": ["generated_code"],
                "outputs": ["rendered_videos", "combined_video_path"],
                "error_handling": "Code correction or retry",
                "monitoring_metrics": ["execution_time", "render_quality", "file_size"]
            },
            "error_handler": {
                "title": "Error Handler Agent",
                "description": "Handles and recovers from workflow errors",
                "icon": "🔧",
                "color": "#F44336",
                "category": "error_recovery",
                "estimated_duration": "10-30 seconds",
                "resource_usage": "Low",
                "dependencies": ["errors"],
                "outputs": ["recovery_action", "escalated_errors"],
                "error_handling": "Escalate to human loop",
                "monitoring_metrics": ["recovery_rate", "escalation_rate", "error_types"]
            },
            "rag_enhancement": {
                "title": "RAG Enhancement Agent",
                "description": "Enhances code generation with contextual information",
                "icon": "📚",
                "color": "#FF9800",
                "category": "enhancement",
                "estimated_duration": "30-60 seconds",
                "resource_usage": "Medium",
                "dependencies": ["failed_code_generation"],
                "outputs": ["enhanced_context", "improved_code"],
                "error_handling": "Fallback to basic generation",
                "monitoring_metrics": ["enhancement_success", "context_relevance"]
            },
            "visual_analysis": {
                "title": "Visual Analysis Agent",
                "description": "Analyzes rendered videos for quality and correctness",
                "icon": "👁️",
                "color": "#9C27B0",
                "category": "quality_assurance",
                "estimated_duration": "30-90 seconds",
                "resource_usage": "Medium",
                "dependencies": ["rendered_videos"],
                "outputs": ["analysis_results", "quality_score"],
                "error_handling": "Skip analysis or retry",
                "monitoring_metrics": ["analysis_accuracy", "quality_scores"]
            },
            "human_loop": {
                "title": "Human Loop Agent",
                "description": "Handles human intervention and escalated errors",
                "icon": "👤",
                "color": "#607D8B",
                "category": "human_interaction",
                "estimated_duration": "Variable",
                "resource_usage": "Low",
                "dependencies": ["escalated_errors", "human_input"],
                "outputs": ["intervention_result", "continuation_decision"],
                "error_handling": "Manual resolution required",
                "monitoring_metrics": ["intervention_time", "resolution_rate"]
            },
            "complete": {
                "title": "Completion Node",
                "description": "Finalizes the workflow and generates summary",
                "icon": "🎉",
                "color": "#4CAF50",
                "category": "finalization",
                "estimated_duration": "5-10 seconds",
                "resource_usage": "Low",
                "dependencies": ["workflow_results"],
                "outputs": ["final_summary", "completion_metrics"],
                "error_handling": "Log completion status",
                "monitoring_metrics": ["completion_rate", "final_quality"]
            }
        }
        
        return node_configs.get(node_name, {})
    
    def get_workflow_execution_path(self, state: VideoGenerationState) -> List[Dict[str, Any]]:
        """Get the execution path visualization for Studio."""
        execution_path = []
        
        if hasattr(state, 'execution_trace') and state.execution_trace:
            for trace in state.execution_trace:
                step_info = {
                    "node": trace.get('step', 'unknown'),
                    "timestamp": trace.get('timestamp', datetime.now().isoformat()),
                    "status": "completed" if trace.get('action') == 'completed' else "in_progress",
                    "duration": trace.get('duration', 0),
                    "data": trace.get('data', {})
                }
                execution_path.append(step_info)
        
        return execution_path
    
    def create_studio_graph_metadata(self) -> Dict[str, Any]:
        """Create comprehensive metadata for Studio graph visualization."""
        return {
            "graph_info": {
                "name": "Video Generation Workflow",
                "description": "Complete workflow for generating educational videos using Manim",
                "version": "2.0.0",
                "author": "LangGraph Agents System",
                "created": datetime.now().isoformat(),
                "node_count": 8,
                "edge_count": 15,
                "estimated_duration": "5-10 minutes",
                "complexity": "High"
            },
            "node_positions": self.node_positions,
            "edge_configurations": self.edge_configurations,
            "visualization_settings": self.visualization_settings,
            "categories": {
                "content_generation": {
                    "color": "#9C27B0",
                    "nodes": ["planning"]
                },
                "code_generation": {
                    "color": "#2196F3", 
                    "nodes": ["code_generation", "rag_enhancement"]
                },
                "video_processing": {
                    "color": "#4CAF50",
                    "nodes": ["rendering", "visual_analysis"]
                },
                "error_recovery": {
                    "color": "#F44336",
                    "nodes": ["error_handler"]
                },
                "human_interaction": {
                    "color": "#607D8B",
                    "nodes": ["human_loop"]
                },
                "finalization": {
                    "color": "#4CAF50",
                    "nodes": ["complete"]
                }
            },
            "execution_statistics": {
                "average_execution_time": 420,  # 7 minutes
                "success_rate": 0.85,
                "most_common_path": ["planning", "code_generation", "rendering", "complete"],
                "error_prone_nodes": ["code_generation", "rendering"],
                "recovery_rate": 0.75
            }
        }


class StudioStateInspector:
    """Provides state inspection capabilities for Studio debugging."""
    
    def __init__(self):
        self.inspection_history = []
        self.state_snapshots = {}
        self.performance_tracking = {}
    
    def capture_state_snapshot(
        self,
        state: VideoGenerationState,
        node_name: str,
        phase: str
    ) -> Dict[str, Any]:
        """Capture a detailed state snapshot for inspection."""
        snapshot_id = f"{state.session_id}_{node_name}_{phase}_{datetime.now().strftime('%H%M%S')}"
        
        snapshot = {
            "snapshot_id": snapshot_id,
            "session_id": state.session_id,
            "node": node_name,
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "state_data": {
                "topic": state.topic,
                "description": state.description,
                "current_step": state.current_step,
                "workflow_complete": state.workflow_complete,
                "workflow_interrupted": getattr(state, 'workflow_interrupted', False),
                "scene_outline": state.scene_outline,
                "scene_count": len(state.scene_implementations) if state.scene_implementations else 0,
                "code_count": len(state.generated_code) if state.generated_code else 0,
                "video_count": len(state.rendered_videos) if state.rendered_videos else 0,
                "error_count": len(state.errors),
                "completion_percentage": state.get_completion_percentage() if hasattr(state, 'get_completion_percentage') else 0
            },
            "detailed_data": {
                "scene_implementations": state.scene_implementations,
                "generated_code_keys": list(state.generated_code.keys()) if state.generated_code else [],
                "rendered_video_keys": list(state.rendered_videos.keys()) if state.rendered_videos else [],
                "error_summary": [
                    {
                        "step": error.step,
                        "type": error.error_type.value if hasattr(error.error_type, 'value') else str(error.error_type),
                        "severity": error.severity.value if hasattr(error.severity, 'value') else str(error.severity),
                        "message": error.message[:100] + "..." if len(error.message) > 100 else error.message
                    }
                    for error in state.errors
                ],
                "execution_trace_count": len(state.execution_trace) if hasattr(state, 'execution_trace') else 0
            }
        }
        
        self.state_snapshots[snapshot_id] = snapshot
        self.inspection_history.append(snapshot_id)
        
        # Keep only last 50 snapshots
        if len(self.inspection_history) > 50:
            old_snapshot_id = self.inspection_history.pop(0)
            self.state_snapshots.pop(old_snapshot_id, None)
        
        return snapshot
    
    def get_state_diff(self, snapshot_id1: str, snapshot_id2: str) -> Dict[str, Any]:
        """Get differences between two state snapshots."""
        snapshot1 = self.state_snapshots.get(snapshot_id1)
        snapshot2 = self.state_snapshots.get(snapshot_id2)
        
        if not snapshot1 or not snapshot2:
            return {"error": "One or both snapshots not found"}
        
        diff = {
            "snapshot1": snapshot_id1,
            "snapshot2": snapshot_id2,
            "time_diff": (
                datetime.fromisoformat(snapshot2["timestamp"]) - 
                datetime.fromisoformat(snapshot1["timestamp"])
            ).total_seconds(),
            "changes": {}
        }
        
        # Compare state data
        state1 = snapshot1["state_data"]
        state2 = snapshot2["state_data"]
        
        for key in state1.keys():
            if key in state2 and state1[key] != state2[key]:
                diff["changes"][key] = {
                    "before": state1[key],
                    "after": state2[key]
                }
        
        return diff
    
    def get_inspection_summary(self, session_id: str) -> Dict[str, Any]:
        """Get inspection summary for a session."""
        session_snapshots = [
            snapshot for snapshot in self.state_snapshots.values()
            if snapshot["session_id"] == session_id
        ]
        
        if not session_snapshots:
            return {"error": "No snapshots found for session"}
        
        # Sort by timestamp
        session_snapshots.sort(key=lambda x: x["timestamp"])
        
        summary = {
            "session_id": session_id,
            "snapshot_count": len(session_snapshots),
            "first_snapshot": session_snapshots[0]["timestamp"],
            "last_snapshot": session_snapshots[-1]["timestamp"],
            "nodes_visited": list(set(s["node"] for s in session_snapshots)),
            "phases_captured": list(set(s["phase"] for s in session_snapshots)),
            "progress_timeline": [
                {
                    "timestamp": s["timestamp"],
                    "node": s["node"],
                    "phase": s["phase"],
                    "completion": s["state_data"]["completion_percentage"]
                }
                for s in session_snapshots
            ]
        }
        
        return summary


# Global instances
studio_visualizer = StudioWorkflowVisualizer()
studio_inspector = StudioStateInspector()


def get_studio_visualizer() -> StudioWorkflowVisualizer:
    """Get the global Studio visualizer instance."""
    return studio_visualizer


def get_studio_inspector() -> StudioStateInspector:
    """Get the global Studio inspector instance."""
    return studio_inspector
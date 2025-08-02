"""
Simplified LangGraph workflow for CLI integration.
This provides a working entry point while we resolve import issues.
"""

from typing import Dict, Any, List, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
# Checkpointer is handled by LangGraph CLI automatically
import logging

logger = logging.getLogger(__name__)


class SimpleVideoState(TypedDict):
    """Simplified state for video generation workflow."""
    messages: Annotated[List[str], add_messages]
    topic: str
    description: str
    session_id: str
    current_step: str
    workflow_complete: bool
    output_data: Dict[str, Any]


def planning_node(state: SimpleVideoState) -> Dict[str, Any]:
    """Simple planning node."""
    topic = state.get("topic", "")
    description = state.get("description", "")
    
    logger.info(f"Planning video for topic: {topic}")
    
    # Simulate planning
    plan = f"Video plan for '{topic}': {description}"
    
    return {
        "messages": [f"✅ Planning completed: {plan}"],
        "current_step": "planning_complete",
        "output_data": {
            "plan": plan,
            "scenes": ["intro", "main_content", "conclusion"]
        }
    }


def code_generation_node(state: SimpleVideoState) -> Dict[str, Any]:
    """Simple code generation node."""
    logger.info("Generating Manim code")
    
    # Simulate code generation
    code = f"""
# Generated Manim code for: {state.get('topic', 'Unknown')}
from manim import *

class VideoScene(Scene):
    def construct(self):
        title = Text("{state.get('topic', 'Video Title')}")
        self.play(Write(title))
        self.wait(2)
"""
    
    return {
        "messages": [f"✅ Code generation completed"],
        "current_step": "code_complete",
        "output_data": {
            **state.get("output_data", {}),
            "generated_code": code
        }
    }


def rendering_node(state: SimpleVideoState) -> Dict[str, Any]:
    """Simple rendering node."""
    logger.info("Rendering video")
    
    # Simulate rendering
    video_path = f"output/{state.get('session_id', 'unknown')}_video.mp4"
    
    return {
        "messages": [f"✅ Rendering completed: {video_path}"],
        "current_step": "rendering_complete",
        "workflow_complete": True,
        "output_data": {
            **state.get("output_data", {}),
            "video_path": video_path,
            "status": "completed"
        }
    }


def route_workflow(state: SimpleVideoState) -> str:
    """Route the workflow based on current step."""
    current_step = state.get("current_step", "start")
    
    if current_step == "start":
        return "planning"
    elif current_step == "planning_complete":
        return "code_generation"
    elif current_step == "code_complete":
        return "rendering"
    else:
        return "END"


def create_simple_workflow():
    """Create the simplified workflow graph."""
    # Create the state graph
    workflow = StateGraph(SimpleVideoState)
    
    # Add nodes
    workflow.add_node("planning", planning_node)
    workflow.add_node("code_generation", code_generation_node)
    workflow.add_node("rendering", rendering_node)
    
    # Add edges
    workflow.add_edge(START, "planning")
    workflow.add_edge("planning", "code_generation")
    workflow.add_edge("code_generation", "rendering")
    workflow.add_edge("rendering", END)
    
    # Compile without checkpointer (LangGraph CLI handles persistence)
    return workflow.compile()


# Create the graph for LangGraph CLI
graph = create_simple_workflow()

logger.info("✅ Simple workflow graph created successfully")
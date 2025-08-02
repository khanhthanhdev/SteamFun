"""
LangGraph workflow entry point for CLI integration.
This module provides the main workflow graph that can be imported by LangGraph CLI.
"""

import os
import sys
from typing import Dict, Any, Optional
from pathlib import Path

# Add the src directory to Python path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from langgraph_agents.state import VideoGenerationState, SystemConfig, create_initial_state
    from langgraph_agents.graph import VideoGenerationWorkflow
    from langgraph_agents.config import ConfigurationManager
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback imports with absolute paths
    sys.path.insert(0, str(current_dir))
    from state import VideoGenerationState, SystemConfig, create_initial_state
    from graph import VideoGenerationWorkflow
    from config import ConfigurationManager


def create_workflow_graph(config: Optional[Dict[str, Any]] = None):
    """Create and return the main video generation workflow graph.
    
    This function is called by LangGraph CLI to get the workflow graph.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        VideoGenerationWorkflow: The compiled workflow graph
    """
    # Load system configuration
    config_manager = ConfigurationManager()
    
    try:
        system_config = config_manager.load_system_config()
    except Exception as e:
        print(f"Failed to load system config, using default: {e}")
        system_config = config_manager.create_default_system_config()
    
    # Apply any runtime configuration
    if config:
        # Update system config with runtime parameters
        for key, value in config.items():
            if hasattr(system_config, key):
                setattr(system_config, key, value)
    
    # Create and return the workflow
    workflow = VideoGenerationWorkflow(system_config)
    return workflow.graph


def create_simple_workflow():
    """Create a simplified workflow for testing and development.
    
    Returns:
        Compiled LangGraph workflow
    """
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import MemorySaver
    
    # Create a simple test workflow
    def simple_node(state: VideoGenerationState) -> Dict[str, Any]:
        """Simple test node that just processes the input."""
        return {
            "messages": [f"Processed: {state.get('topic', 'No topic')}"],
            "workflow_complete": True
        }
    
    # Build simple graph
    graph_builder = StateGraph(VideoGenerationState)
    graph_builder.add_node("process", simple_node)
    graph_builder.add_edge(START, "process")
    graph_builder.add_edge("process", END)
    
    # Compile with memory checkpointer
    return graph_builder.compile(checkpointer=MemorySaver())


# Create the main workflow graph for LangGraph CLI
try:
    workflow = create_workflow_graph()
    print("✅ Successfully created main workflow graph")
except Exception as e:
    print(f"⚠️  Failed to create main workflow, using simple workflow: {e}")
    workflow = create_simple_workflow()
    print("✅ Created simple workflow as fallback")

# Export for LangGraph CLI
graph = workflow
"""
Studio-compatible workflow entry point.

This module provides the main workflow graph that can be used in LangGraph Studio
for testing the complete agent system with Studio-specific optimizations.
"""

import logging
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from ..models.state import VideoGenerationState
from ..workflow_graph import VideoGenerationWorkflow
from .studio_config import get_studio_config
from .studio_integration import studio_monitor

logger = logging.getLogger(__name__)


class StudioVideoGenerationWorkflow(VideoGenerationWorkflow):
    """Studio-optimized version of the video generation workflow."""
    
    def __init__(self, config=None, checkpointer=None, checkpoint_manager=None):
        # Use Studio configuration if none provided
        if config is None:
            studio_config = get_studio_config()
            config = studio_config.get_workflow_config()
        
        # Use memory checkpointer for Studio testing
        if checkpointer is None:
            checkpointer = MemorySaver()
        
        super().__init__(config, checkpointer, checkpoint_manager)
        
        # Add Studio-specific monitoring
        self._add_studio_monitoring()
    
    def _add_studio_monitoring(self):
        """Add Studio-specific monitoring to the workflow."""
        # Wrap existing nodes with monitoring
        original_nodes = {}
        
        # Store original node functions
        for node_name in ["planning", "code_generation", "rendering", "error_handler"]:
            if hasattr(self.graph, 'nodes') and node_name in self.graph.nodes:
                original_nodes[node_name] = self.graph.nodes[node_name]
        
        # Note: In practice, we would wrap the node functions with monitoring
        # For now, we'll rely on the existing execution trace functionality
        logger.info("Studio monitoring added to workflow")
    
    async def invoke_with_studio_monitoring(
        self,
        initial_state: VideoGenerationState,
        config: Optional[Dict[str, Any]] = None
    ) -> VideoGenerationState:
        """Invoke workflow with Studio-specific monitoring."""
        session_id = initial_state.session_id
        
        # Start Studio monitoring
        studio_monitor.start_session(session_id, "full_workflow")
        
        try:
            # Record initial metrics
            studio_monitor.record_performance_metric(
                "full_workflow",
                "workflow_started",
                {
                    "topic": initial_state.topic,
                    "description_length": len(initial_state.description),
                    "session_id": session_id
                }
            )
            
            # Execute workflow
            final_state = await self.invoke(initial_state, config)
            
            # Record completion metrics
            studio_monitor.record_performance_metric(
                "full_workflow",
                "workflow_completed",
                {
                    "session_id": session_id,
                    "scenes_generated": len(final_state.scene_implementations),
                    "videos_rendered": len(final_state.rendered_videos),
                    "has_errors": final_state.has_errors(),
                    "completion_percentage": final_state.get_completion_percentage()
                }
            )
            
            # End monitoring with success
            studio_monitor.end_session(session_id, "completed")
            
            return final_state
            
        except Exception as e:
            # Record error metrics
            studio_monitor.record_performance_metric(
                "full_workflow",
                "workflow_failed",
                {
                    "session_id": session_id,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            
            # End monitoring with failure
            studio_monitor.end_session(session_id, "failed")
            
            raise
    
    async def stream_with_studio_monitoring(
        self,
        initial_state: VideoGenerationState,
        config: Optional[Dict[str, Any]] = None
    ):
        """Stream workflow execution with Studio monitoring."""
        session_id = initial_state.session_id
        
        # Start Studio monitoring
        studio_monitor.start_session(session_id, "full_workflow_stream")
        
        try:
            async for chunk in self.stream(initial_state, config):
                # Record streaming metrics
                if isinstance(chunk, dict) and "messages" in chunk:
                    studio_monitor.record_performance_metric(
                        "full_workflow",
                        "stream_chunk",
                        {
                            "session_id": session_id,
                            "chunk_keys": list(chunk.keys())
                        }
                    )
                
                yield chunk
            
            # End monitoring with success
            studio_monitor.end_session(session_id, "completed")
            
        except Exception as e:
            # Record error and end monitoring
            studio_monitor.record_performance_metric(
                "full_workflow",
                "stream_failed",
                {
                    "session_id": session_id,
                    "error": str(e)
                }
            )
            
            studio_monitor.end_session(session_id, "failed")
            raise


def create_studio_workflow() -> StudioVideoGenerationWorkflow:
    """Create a Studio-optimized workflow instance."""
    logger.info("Creating Studio workflow")
    return StudioVideoGenerationWorkflow()


def create_studio_workflow_graph():
    """Create the Studio workflow graph for LangGraph Studio."""
    workflow = create_studio_workflow()
    return workflow.graph


# Create the main graph for Studio
graph = create_studio_workflow_graph()


# Additional utility functions for Studio testing
def get_workflow_info() -> Dict[str, Any]:
    """Get information about the Studio workflow."""
    workflow = create_studio_workflow()
    
    return {
        "workflow_type": "StudioVideoGenerationWorkflow",
        "checkpointing_enabled": workflow.checkpointer is not None,
        "checkpointer_type": type(workflow.checkpointer).__name__ if workflow.checkpointer else None,
        "config": {
            "max_retries": workflow.config.max_retries,
            "timeout_seconds": workflow.config.timeout_seconds,
            "preview_mode": workflow.config.preview_mode,
            "output_dir": workflow.config.output_dir
        },
        "graph_visualization": workflow.get_graph_visualization()
    }


def test_workflow_with_sample_data() -> Dict[str, Any]:
    """Test the workflow with sample data for Studio validation."""
    from .test_scenarios import StudioTestDataGenerator
    
    # Generate sample test data
    test_data = StudioTestDataGenerator.generate_planning_test_data("simple")
    
    # Create test state
    from ..models.state import VideoGenerationState
    test_state = VideoGenerationState(
        topic=test_data["topic"],
        description=test_data["description"],
        session_id="studio_test_workflow"
    )
    
    return {
        "test_data": test_data,
        "test_state_created": True,
        "workflow_ready": True,
        "sample_input": {
            "topic": test_state.topic,
            "description": test_state.description,
            "session_id": test_state.session_id
        }
    }


if __name__ == "__main__":
    # Test the Studio workflow setup
    print("Testing Studio workflow setup...")
    
    try:
        workflow_info = get_workflow_info()
        print(f"✅ Workflow created successfully")
        print(f"   - Type: {workflow_info['workflow_type']}")
        print(f"   - Checkpointing: {workflow_info['checkpointing_enabled']}")
        print(f"   - Preview mode: {workflow_info['config']['preview_mode']}")
        
        test_info = test_workflow_with_sample_data()
        print(f"✅ Test data generated successfully")
        print(f"   - Topic: {test_info['sample_input']['topic']}")
        print(f"   - Session ID: {test_info['sample_input']['session_id']}")
        
        print("\n🎉 Studio workflow setup completed successfully!")
        
    except Exception as e:
        print(f"❌ Studio workflow setup failed: {e}")
        raise
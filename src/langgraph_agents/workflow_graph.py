"""
LangGraph StateGraph implementation for video generation workflow.

This module creates the main workflow graph using LangGraph StateGraph with
all node functions and conditional routing logic following LangGraph best practices.
"""

import logging
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# PostgreSQL checkpointer is optional and may not be available
try:
    from langgraph.checkpoint.postgres import PostgresSaver
    POSTGRES_AVAILABLE = True
except ImportError:
    PostgresSaver = None
    POSTGRES_AVAILABLE = False

from .checkpointing import CheckpointManager, CheckpointConfig, create_checkpoint_manager
from .checkpointing.checkpoint_manager import CheckpointBackend
from .checkpointing.recovery import CheckpointRecovery, create_recovery_handler

from .models.state import VideoGenerationState
from .models.config import WorkflowConfig
from .nodes.planning_node import planning_node
from .nodes.code_generation_node import code_generation_node
from .nodes.rendering_node import rendering_node
from .nodes.error_handler_node import error_handler_node
from .routing import (
    route_from_planning,
    route_from_code_generation,
    route_from_rendering,
    route_from_error_handler,
    validate_routing_decision
)

logger = logging.getLogger(__name__)


class VideoGenerationWorkflow:
    """
    Main workflow class that encapsulates the LangGraph StateGraph for video generation.
    
    This class follows LangGraph best practices by:
    - Using simple node functions instead of complex agents
    - Implementing conditional routing based on state
    - Providing checkpointing and persistence capabilities
    - Supporting both development and production configurations
    """
    
    def __init__(
        self, 
        config: WorkflowConfig, 
        checkpointer: Optional[Any] = None,
        checkpoint_manager: Optional[CheckpointManager] = None
    ):
        """
        Initialize the video generation workflow.
        
        Args:
            config: Workflow configuration
            checkpointer: Optional checkpointer for persistence (MemorySaver or PostgresSaver)
            checkpoint_manager: Optional checkpoint manager for enhanced functionality
        """
        self.config = config
        self.checkpointer = checkpointer
        self.checkpoint_manager = checkpoint_manager
        self.recovery_handler = create_recovery_handler(config)
        self.graph = None
        self._build_graph()
    
    def _build_graph(self) -> None:
        """Build the StateGraph with all nodes and conditional edges."""
        logger.info("Building video generation workflow graph")
        
        # Create StateGraph with VideoGenerationState
        graph_builder = StateGraph(VideoGenerationState)
        
        # Add all node functions to the graph
        self._add_nodes(graph_builder)
        
        # Add conditional edges for routing
        self._add_conditional_edges(graph_builder)
        
        # Set entry point
        graph_builder.add_edge(START, "planning")
        
        # Compile the graph with checkpointer if provided
        if self.checkpointer:
            self.graph = graph_builder.compile(checkpointer=self.checkpointer)
            logger.info(f"Graph compiled with checkpointing enabled: {type(self.checkpointer).__name__}")
        else:
            self.graph = graph_builder.compile()
            logger.info("Graph compiled without checkpointing")
    
    def _add_nodes(self, graph_builder: StateGraph) -> None:
        """Add all node functions to the graph."""
        logger.debug("Adding nodes to graph")
        
        # Core workflow nodes
        graph_builder.add_node("planning", planning_node)
        graph_builder.add_node("code_generation", code_generation_node)
        graph_builder.add_node("rendering", rendering_node)
        graph_builder.add_node("error_handler", error_handler_node)
        
        # Additional workflow nodes (to be implemented)
        graph_builder.add_node("rag_enhancement", self._rag_enhancement_node)
        graph_builder.add_node("visual_analysis", self._visual_analysis_node)
        graph_builder.add_node("human_loop", self._human_loop_node)
        graph_builder.add_node("complete", self._complete_node)
        
        logger.debug("All nodes added to graph")
    
    def _add_conditional_edges(self, graph_builder: StateGraph) -> None:
        """Add conditional edges for workflow routing."""
        logger.debug("Adding conditional edges to graph")
        
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
        
        # RAG enhancement routing (back to code generation)
        graph_builder.add_conditional_edges(
            "rag_enhancement",
            self._route_from_rag_enhancement,
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
            self._route_from_visual_analysis,
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
            self._route_from_human_loop,
            {
                "planning": "planning",
                "code_generation": "code_generation",
                "rendering": "rendering",
                "complete": "complete"
            }
        )
        
        # Complete node is terminal
        graph_builder.add_edge("complete", END)
        
        logger.debug("All conditional edges added to graph")
    
    async def _rag_enhancement_node(self, state: VideoGenerationState) -> VideoGenerationState:
        """
        RAG enhancement node for improving failed code generation attempts.
        
        This is a placeholder implementation that will be fully implemented
        in a future task.
        """
        logger.info(f"RAG enhancement node for session {state.session_id}")
        
        state.current_step = "code_generation"  # RAG enhancement is part of code generation
        state.add_execution_trace("rag_enhancement_node", {
            "action": "started",
            "session_id": state.session_id
        })
        
        # TODO: Implement RAG enhancement logic
        # For now, just mark as completed and return to code generation
        
        state.add_execution_trace("rag_enhancement_node", {
            "action": "completed",
            "note": "placeholder_implementation"
        })
        
        return state
    
    async def _visual_analysis_node(self, state: VideoGenerationState) -> VideoGenerationState:
        """
        Visual analysis node for analyzing rendered videos.
        
        This is a placeholder implementation that will be fully implemented
        in a future task.
        """
        logger.info(f"Visual analysis node for session {state.session_id}")
        
        state.current_step = "visual_analysis"
        state.add_execution_trace("visual_analysis_node", {
            "action": "started",
            "session_id": state.session_id
        })
        
        # TODO: Implement visual analysis logic
        # For now, just mark as completed
        
        state.add_execution_trace("visual_analysis_node", {
            "action": "completed",
            "note": "placeholder_implementation"
        })
        
        return state
    
    async def _human_loop_node(self, state: VideoGenerationState) -> VideoGenerationState:
        """
        Human loop node for handling escalated errors and user intervention.
        
        This is a placeholder implementation that will be fully implemented
        in a future task.
        """
        logger.info(f"Human loop node for session {state.session_id}")
        
        state.current_step = "human_loop"
        state.add_execution_trace("human_loop_node", {
            "action": "started",
            "session_id": state.session_id,
            "escalated_errors": len(state.escalated_errors),
            "pending_input": bool(state.pending_human_input)
        })
        
        # TODO: Implement human intervention logic
        # For now, just clear pending input and continue
        
        if state.pending_human_input:
            state.pending_human_input = None
        
        state.workflow_interrupted = False
        
        state.add_execution_trace("human_loop_node", {
            "action": "completed",
            "note": "placeholder_implementation"
        })
        
        return state
    
    async def _complete_node(self, state: VideoGenerationState) -> VideoGenerationState:
        """
        Completion node that finalizes the workflow.
        """
        logger.info(f"Completing workflow for session {state.session_id}")
        
        state.current_step = "complete"
        state.workflow_complete = True
        
        # Calculate final metrics
        completion_percentage = state.get_completion_percentage()
        
        state.add_execution_trace("complete_node", {
            "action": "workflow_completed",
            "session_id": state.session_id,
            "completion_percentage": completion_percentage,
            "total_scenes": len(state.scene_implementations),
            "rendered_scenes": len(state.rendered_videos),
            "has_combined_video": bool(state.combined_video_path),
            "total_errors": len(state.errors),
            "escalated_errors": len(state.escalated_errors)
        })
        
        logger.info(f"Workflow completed: {completion_percentage:.1f}% success rate")
        
        return state
    
    def _route_from_rag_enhancement(self, state: VideoGenerationState) -> str:
        """Route from RAG enhancement node."""
        # Simple routing logic - return to code generation
        if state.workflow_interrupted or state.pending_human_input:
            return "human_loop"
        elif state.has_errors():
            return "error_handler"
        else:
            return "code_generation"
    
    def _route_from_visual_analysis(self, state: VideoGenerationState) -> str:
        """Route from visual analysis node."""
        if state.workflow_interrupted or state.pending_human_input:
            return "human_loop"
        elif state.has_errors():
            return "error_handler"
        else:
            return "complete"
    
    def _route_from_human_loop(self, state: VideoGenerationState) -> str:
        """Route from human loop node."""
        # Determine where to continue based on workflow state
        if not state.scene_outline or not state.scene_implementations:
            return "planning"
        elif len(state.generated_code) < len(state.scene_implementations):
            return "code_generation"
        elif len(state.rendered_videos) < len(state.generated_code):
            return "rendering"
        else:
            return "complete"
    
    async def invoke(self, initial_state: VideoGenerationState, config: Optional[Dict[str, Any]] = None) -> VideoGenerationState:
        """
        Invoke the workflow with an initial state.
        
        Args:
            initial_state: Initial workflow state
            config: Optional runtime configuration
            
        Returns:
            Final workflow state
        """
        logger.info(f"Invoking workflow for session {initial_state.session_id}")
        
        try:
            # Validate initial state
            if not initial_state.topic or not initial_state.description:
                raise ValueError("Topic and description are required")
            
            # Set workflow configuration in state if not already set
            if not hasattr(initial_state, 'config') or not initial_state.config:
                initial_state.config = self.config
            
            # Add workflow start trace
            initial_state.add_execution_trace("workflow_start", {
                "session_id": initial_state.session_id,
                "topic": initial_state.topic,
                "config": {
                    "max_retries": self.config.max_retries,
                    "use_rag": self.config.use_rag,
                    "use_visual_analysis": self.config.use_visual_analysis
                }
            })
            
            # Invoke the graph
            final_state = await self.graph.ainvoke(initial_state, config=config)
            
            logger.info(f"Workflow completed for session {initial_state.session_id}")
            return final_state
            
        except Exception as e:
            logger.error(f"Workflow invocation failed: {str(e)}")
            
            # Add error to state
            from .models.errors import WorkflowError, ErrorType, ErrorSeverity
            error = WorkflowError(
                step="workflow",
                error_type=ErrorType.SYSTEM,
                message=f"Workflow invocation failed: {str(e)}",
                severity=ErrorSeverity.CRITICAL,
                context={"exception": str(e), "exception_type": type(e).__name__}
            )
            initial_state.add_error(error)
            
            # Add failure trace
            initial_state.add_execution_trace("workflow_failed", {
                "error": str(e),
                "error_type": type(e).__name__
            })
            
            return initial_state
    
    async def stream(self, initial_state: VideoGenerationState, config: Optional[Dict[str, Any]] = None):
        """
        Stream workflow execution for real-time monitoring.
        
        Args:
            initial_state: Initial workflow state
            config: Optional runtime configuration
            
        Yields:
            State updates during workflow execution
        """
        logger.info(f"Streaming workflow for session {initial_state.session_id}")
        
        try:
            # Set workflow configuration in state if not already set
            if not hasattr(initial_state, 'config') or not initial_state.config:
                initial_state.config = self.config
            
            # Stream the graph execution
            async for chunk in self.graph.astream(initial_state, config=config):
                yield chunk
                
        except Exception as e:
            logger.error(f"Workflow streaming failed: {str(e)}")
            
            # Yield error state
            from .models.errors import WorkflowError, ErrorType, ErrorSeverity
            error = WorkflowError(
                step="workflow",
                error_type=ErrorType.SYSTEM,
                message=f"Workflow streaming failed: {str(e)}",
                severity=ErrorSeverity.CRITICAL,
                context={"exception": str(e), "exception_type": type(e).__name__}
            )
            initial_state.add_error(error)
            
            yield {"error": initial_state}
    
    async def recover_from_checkpoint(
        self,
        thread_id: str,
        checkpoint_id: Optional[str] = None
    ) -> Optional[VideoGenerationState]:
        """
        Recover workflow state from a checkpoint.
        
        Args:
            thread_id: Thread ID to recover
            checkpoint_id: Specific checkpoint ID (uses latest if not provided)
            
        Returns:
            Recovered state if successful, None otherwise
        """
        if not self.checkpointer:
            logger.error("Cannot recover from checkpoint: no checkpointer configured")
            return None
        
        try:
            logger.info(f"Attempting to recover checkpoint for thread {thread_id}")
            
            # Get the checkpoint
            config = {"configurable": {"thread_id": thread_id}}
            if checkpoint_id:
                config["configurable"]["checkpoint_id"] = checkpoint_id
            
            checkpoint = self.checkpointer.get(config)
            if not checkpoint:
                logger.warning(f"No checkpoint found for thread {thread_id}")
                return None
            
            # Extract state from checkpoint
            state_data = checkpoint.get("channel_values", {})
            if not state_data:
                logger.error("Checkpoint contains no state data")
                return None
            
            # Reconstruct the state
            state = VideoGenerationState(**state_data)
            
            # Analyze the checkpoint state
            analysis = self.recovery_handler.analyze_checkpoint_state(state)
            logger.info(f"Checkpoint analysis: health_score={analysis['health_score']:.1f}")
            
            # Determine recovery strategy
            recovery_decision = self.recovery_handler.determine_recovery_strategy(state, analysis)
            logger.info(f"Recovery strategy: {recovery_decision.strategy.value}")
            
            # Apply recovery modifications
            recovered_state = self.recovery_handler.apply_recovery_modifications(state, recovery_decision)
            
            # Validate the recovered state
            is_valid, issues = self.recovery_handler.validate_recovery_state(recovered_state)
            if not is_valid:
                logger.error(f"Recovered state validation failed: {issues}")
                return None
            
            logger.info(f"Successfully recovered checkpoint for thread {thread_id}")
            return recovered_state
            
        except Exception as e:
            logger.error(f"Failed to recover from checkpoint: {e}")
            return None
    
    async def resume_workflow(
        self,
        thread_id: str,
        checkpoint_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> VideoGenerationState:
        """
        Resume a workflow from a checkpoint.
        
        Args:
            thread_id: Thread ID to resume
            checkpoint_id: Specific checkpoint ID (uses latest if not provided)
            config: Optional runtime configuration
            
        Returns:
            Final workflow state
        """
        # Recover the state
        recovered_state = await self.recover_from_checkpoint(thread_id, checkpoint_id)
        if not recovered_state:
            raise ValueError(f"Cannot resume workflow: failed to recover checkpoint for thread {thread_id}")
        
        # Resume execution from the recovered state
        logger.info(f"Resuming workflow from step: {recovered_state.current_step}")
        
        # Add resume trace
        recovered_state.add_execution_trace("workflow_resumed", {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
            "resumed_from_step": recovered_state.current_step,
            "resume_timestamp": datetime.now().isoformat()
        })
        
        # Continue execution
        runtime_config = config or {}
        runtime_config.setdefault("configurable", {})["thread_id"] = thread_id
        
        return await self.invoke(recovered_state, runtime_config)
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """
        Get information about the checkpointing configuration.
        
        Returns:
            Dictionary with checkpoint information
        """
        info = {
            "checkpointing_enabled": bool(self.checkpointer),
            "checkpointer_type": type(self.checkpointer).__name__ if self.checkpointer else None,
            "persistent": False,
            "manager_available": bool(self.checkpoint_manager)
        }
        
        if self.checkpoint_manager:
            info.update(self.checkpoint_manager.get_checkpoint_info())
        
        return info
    
    async def cleanup_old_checkpoints(self, max_age_hours: int = 24) -> int:
        """
        Clean up old checkpoints.
        
        Args:
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of checkpoints cleaned up
        """
        if self.checkpoint_manager:
            return await self.checkpoint_manager.cleanup_old_checkpoints(max_age_hours * 3600)
        else:
            logger.warning("No checkpoint manager available for cleanup")
            return 0
    
    async def get_checkpoint_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored checkpoints.
        
        Returns:
            Dictionary with checkpoint statistics
        """
        if self.checkpoint_manager:
            return await self.checkpoint_manager.get_checkpoint_stats()
        else:
            return {"error": "No checkpoint manager available"}
    
    def get_graph_visualization(self) -> str:
        """
        Get a visual representation of the workflow graph.
        
        Returns:
            Mermaid diagram representation of the graph
        """
        return """
        graph TD
            START([Start]) --> planning[Planning Node]
            planning --> |Success| code_generation[Code Generation Node]
            planning --> |Error| error_handler[Error Handler Node]
            planning --> |Non-recoverable| human_loop[Human Loop Node]
            
            code_generation --> |Success| rendering[Rendering Node]
            code_generation --> |Partial Success| rag_enhancement[RAG Enhancement Node]
            code_generation --> |Error| error_handler
            code_generation --> |Non-recoverable| human_loop
            
            rag_enhancement --> |Enhanced| code_generation
            rag_enhancement --> |Error| error_handler
            rag_enhancement --> |Failed| human_loop
            
            rendering --> |Success + Visual Analysis| visual_analysis[Visual Analysis Node]
            rendering --> |Success| complete[Complete Node]
            rendering --> |Error| error_handler
            rendering --> |Non-recoverable| human_loop
            
            visual_analysis --> |Success| complete
            visual_analysis --> |Error| error_handler
            visual_analysis --> |Failed| human_loop
            
            error_handler --> |Retry Planning| planning
            error_handler --> |Retry Code Gen| code_generation
            error_handler --> |Retry Rendering| rendering
            error_handler --> |Escalate| human_loop
            error_handler --> |Complete| complete
            
            human_loop --> |Continue Planning| planning
            human_loop --> |Continue Code Gen| code_generation
            human_loop --> |Continue Rendering| rendering
            human_loop --> |Complete| complete
            
            complete --> END([End])
        """


def create_workflow(
    config: WorkflowConfig,
    use_checkpointing: bool = False,
    postgres_config: Optional[Dict[str, Any]] = None,
    checkpoint_config: Optional[CheckpointConfig] = None
) -> VideoGenerationWorkflow:
    """
    Create a video generation workflow with optional checkpointing.
    
    Args:
        config: Workflow configuration
        use_checkpointing: Whether to enable checkpointing
        postgres_config: PostgreSQL configuration for production checkpointing (deprecated)
        checkpoint_config: Checkpoint configuration (preferred over postgres_config)
        
    Returns:
        Configured VideoGenerationWorkflow instance
    """
    checkpointer = None
    checkpoint_manager = None
    
    if use_checkpointing:
        # Create checkpoint manager
        if checkpoint_config is None:
            # Create from environment or legacy postgres_config
            if postgres_config:
                checkpoint_config = CheckpointConfig(
                    backend=CheckpointBackend.POSTGRES,
                    postgres_connection_string=postgres_config.get("connection_string")
                )
            else:
                checkpoint_config = CheckpointConfig.from_environment()
        
        checkpoint_manager = create_checkpoint_manager(config, checkpoint_config)
        checkpointer = checkpoint_manager.checkpointer
        
        logger.info(f"Created workflow with {checkpoint_manager.backend_type.value} checkpointing")
    else:
        logger.info("Creating workflow without checkpointing")
    
    return VideoGenerationWorkflow(config, checkpointer, checkpoint_manager)


def validate_workflow_configuration(config: WorkflowConfig) -> bool:
    """
    Validate workflow configuration for required settings.
    
    Args:
        config: Workflow configuration to validate
        
    Returns:
        True if configuration is valid, False otherwise
    """
    required_fields = [
        "planner_model",
        "code_model",
        "helper_model",
        "output_dir",
        "max_retries"
    ]
    
    for field in required_fields:
        if not hasattr(config, field) or getattr(config, field) is None:
            logger.error(f"Missing required configuration field: {field}")
            return False
    
    # Validate model configurations
    for model_field in ["planner_model", "code_model", "helper_model"]:
        model_config = getattr(config, model_field)
        if not hasattr(model_config, "provider") or not hasattr(model_config, "model_name"):
            logger.error(f"Invalid model configuration for {model_field}")
            return False
    
    # Validate numeric settings
    if config.max_retries < 1:
        logger.error("max_retries must be at least 1")
        return False
    
    if config.timeout_seconds < 30:
        logger.error("timeout_seconds must be at least 30")
        return False
    
    logger.info("Workflow configuration is valid")
    return True
"""
LangGraph StateGraph foundation with proper state management.
Compatible with existing workflow while enabling multi-agent coordination.
"""

from typing import Dict, Any, List, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
import logging
import asyncio
import time

from .state import VideoGenerationState, SystemConfig, create_initial_state
from .base_agent import BaseAgent, AgentFactory


logger = logging.getLogger(__name__)


class VideoGenerationWorkflow:
    """LangGraph workflow for video generation with multi-agent coordination.
    
    Maintains compatibility with existing workflow patterns while enabling
    sophisticated agent coordination and error handling.
    """
    
    def __init__(self, system_config: SystemConfig):
        """Initialize the workflow with system configuration.
        
        Args:
            system_config: System-wide configuration
        """
        self.system_config = system_config
        self.agents: Dict[str, BaseAgent] = {}
        self.graph = None
        self.checkpointer = MemorySaver() if system_config.workflow_config.enable_checkpoints else None
        
        # Validate system configuration
        self._validate_system_config()
        
        # Initialize agents
        self._initialize_agents()
        
        # Build the workflow graph
        self._build_graph()
        
        logger.info("VideoGenerationWorkflow initialized with agents: %s", list(self.agents.keys()))
    
    def _validate_system_config(self):
        """Validate system configuration before initialization."""
        if not self.system_config.agent_configs:
            raise ValueError("No agents configured in system configuration")
        
        # Check for required core agents
        required_agents = ['planner_agent', 'code_generator_agent', 'renderer_agent']
        available_agents = set(self.system_config.agent_configs.keys())
        
        missing_core_agents = [agent for agent in required_agents if agent not in available_agents]
        if missing_core_agents:
            logger.warning(f"Missing core agents: {missing_core_agents}. Workflow may have limited functionality.")
        
        # Validate workflow settings
        if self.system_config.workflow_config.max_workflow_retries < 0:
            raise ValueError("max_workflow_retries must be non-negative")
        
        if self.system_config.workflow_config.workflow_timeout_seconds <= 0:
            raise ValueError("workflow_timeout_seconds must be positive")
        
        logger.info("System configuration validation passed")
    
    def _initialize_agents(self):
        """Initialize all agents with their configurations."""
        for agent_name, agent_config in self.system_config.agent_configs.items():
            try:
                agent = AgentFactory.create_agent(
                    agent_name, 
                    agent_config, 
                    self.system_config
                )
                self.agents[agent_name] = agent
                logger.info(f"Initialized agent: {agent_name}")
            except Exception as e:
                logger.error(f"Failed to initialize agent {agent_name}: {e}")
                raise
    
    def _build_graph(self):
        """Build the LangGraph StateGraph with agent nodes and routing logic."""
        # Create the state graph
        graph_builder = StateGraph(VideoGenerationState)
        
        # Add agent nodes
        for agent_name, agent in self.agents.items():
            graph_builder.add_node(agent_name, self._create_agent_node(agent))
        
        # Add conditional routing logic
        self._add_routing_logic(graph_builder)
        
        # Compile the graph
        self.graph = graph_builder.compile(
            checkpointer=self.checkpointer,
            interrupt_before=self._get_interrupt_points(),
            interrupt_after=self._get_interrupt_after_points()
        )
        
        logger.info("LangGraph workflow compiled successfully")
    
    def _create_agent_node(self, agent: BaseAgent):
        """Create a node function for an agent.
        
        Args:
            agent: Agent instance
            
        Returns:
            Callable: Node function for the agent
        """
        async def agent_node(state: VideoGenerationState) -> Dict[str, Any]:
            """Execute agent with monitoring and error handling."""
            try:
                # Update current agent in state
                state['current_agent'] = agent.name
                
                # Execute agent with monitoring
                command = await agent.execute_with_monitoring(state)
                
                # Extract updates from command
                updates = {}
                if hasattr(command, 'update') and command.update:
                    updates.update(command.update)
                
                # Set next agent
                if hasattr(command, 'goto') and command.goto:
                    updates['next_agent'] = command.goto
                
                return updates
                
            except Exception as e:
                logger.error(f"Error in agent node {agent.name}: {e}")
                # Handle error through agent's error handler
                error_command = await agent.handle_error(e, state)
                
                updates = {}
                if hasattr(error_command, 'update') and error_command.update:
                    updates.update(error_command.update)
                
                if hasattr(error_command, 'goto') and error_command.goto:
                    updates['next_agent'] = error_command.goto
                
                return updates
        
        return agent_node
    
    def _add_routing_logic(self, graph_builder: StateGraph):
        """Add conditional routing logic between agents.
        
        Args:
            graph_builder: StateGraph builder instance
        """
        # Get list of available agents
        available_agents = set(self.agents.keys())
        
        # Entry point - start with planner if available, otherwise start with first available agent
        if "planner_agent" in available_agents:
            graph_builder.add_edge(START, "planner_agent")
        elif "code_generator_agent" in available_agents:
            graph_builder.add_edge(START, "code_generator_agent")
        elif "renderer_agent" in available_agents:
            graph_builder.add_edge(START, "renderer_agent")
        else:
            # If no core agents available, end immediately
            graph_builder.add_edge(START, END)
        
        # Planner routing (if available)
        if "planner_agent" in available_agents:
            planner_routes = {"END": END}
            if "code_generator_agent" in available_agents:
                planner_routes["code_generator_agent"] = "code_generator_agent"
            if "error_handler_agent" in available_agents:
                planner_routes["error_handler_agent"] = "error_handler_agent"
            if "human_loop_agent" in available_agents:
                planner_routes["human_loop_agent"] = "human_loop_agent"
            
            graph_builder.add_conditional_edges(
                "planner_agent",
                self._route_from_planner,
                planner_routes
            )
        
        # Code generator routing (if available)
        if "code_generator_agent" in available_agents:
            code_gen_routes = {"END": END}
            if "renderer_agent" in available_agents:
                code_gen_routes["renderer_agent"] = "renderer_agent"
            if "rag_agent" in available_agents:
                code_gen_routes["rag_agent"] = "rag_agent"
            if "error_handler_agent" in available_agents:
                code_gen_routes["error_handler_agent"] = "error_handler_agent"
            if "human_loop_agent" in available_agents:
                code_gen_routes["human_loop_agent"] = "human_loop_agent"
            
            graph_builder.add_conditional_edges(
                "code_generator_agent",
                self._route_from_code_generator,
                code_gen_routes
            )
        
        # Renderer routing (if available)
        if "renderer_agent" in available_agents:
            renderer_routes = {"END": END}
            if "visual_analysis_agent" in available_agents:
                renderer_routes["visual_analysis_agent"] = "visual_analysis_agent"
            if "error_handler_agent" in available_agents:
                renderer_routes["error_handler_agent"] = "error_handler_agent"
            if "human_loop_agent" in available_agents:
                renderer_routes["human_loop_agent"] = "human_loop_agent"
            
            graph_builder.add_conditional_edges(
                "renderer_agent",
                self._route_from_renderer,
                renderer_routes
            )
        
        # Visual analysis routing (if available)
        if "visual_analysis_agent" in available_agents:
            visual_routes = {"END": END}
            if "code_generator_agent" in available_agents:
                visual_routes["code_generator_agent"] = "code_generator_agent"
            if "error_handler_agent" in available_agents:
                visual_routes["error_handler_agent"] = "error_handler_agent"
            if "human_loop_agent" in available_agents:
                visual_routes["human_loop_agent"] = "human_loop_agent"
            
            graph_builder.add_conditional_edges(
                "visual_analysis_agent",
                self._route_from_visual_analysis,
                visual_routes
            )
        
        # RAG agent routing (if available)
        if "rag_agent" in available_agents:
            rag_routes = {"END": END}
            if "code_generator_agent" in available_agents:
                rag_routes["code_generator_agent"] = "code_generator_agent"
            if "planner_agent" in available_agents:
                rag_routes["planner_agent"] = "planner_agent"
            if "error_handler_agent" in available_agents:
                rag_routes["error_handler_agent"] = "error_handler_agent"
            
            graph_builder.add_conditional_edges(
                "rag_agent",
                self._route_from_rag,
                rag_routes
            )
        
        # Error handler routing (if available)
        if "error_handler_agent" in available_agents:
            error_routes = {"END": END}
            if "planner_agent" in available_agents:
                error_routes["planner_agent"] = "planner_agent"
            if "code_generator_agent" in available_agents:
                error_routes["code_generator_agent"] = "code_generator_agent"
            if "renderer_agent" in available_agents:
                error_routes["renderer_agent"] = "renderer_agent"
            if "visual_analysis_agent" in available_agents:
                error_routes["visual_analysis_agent"] = "visual_analysis_agent"
            if "human_loop_agent" in available_agents:
                error_routes["human_loop_agent"] = "human_loop_agent"
            
            graph_builder.add_conditional_edges(
                "error_handler_agent",
                self._route_from_error_handler,
                error_routes
            )
        
        # Human loop routing (if available)
        if "human_loop_agent" in available_agents:
            human_routes = {"END": END}
            if "planner_agent" in available_agents:
                human_routes["planner_agent"] = "planner_agent"
            if "code_generator_agent" in available_agents:
                human_routes["code_generator_agent"] = "code_generator_agent"
            if "renderer_agent" in available_agents:
                human_routes["renderer_agent"] = "renderer_agent"
            if "visual_analysis_agent" in available_agents:
                human_routes["visual_analysis_agent"] = "visual_analysis_agent"
            if "error_handler_agent" in available_agents:
                human_routes["error_handler_agent"] = "error_handler_agent"
            
            graph_builder.add_conditional_edges(
                "human_loop_agent",
                self._route_from_human_loop,
                human_routes
            )
        
        # Monitoring agent can be called from anywhere (if available)
        if "monitoring_agent" in available_agents:
            monitoring_routes = {"END": END}
            if "planner_agent" in available_agents:
                monitoring_routes["planner_agent"] = "planner_agent"
            if "code_generator_agent" in available_agents:
                monitoring_routes["code_generator_agent"] = "code_generator_agent"
            if "renderer_agent" in available_agents:
                monitoring_routes["renderer_agent"] = "renderer_agent"
            
            graph_builder.add_conditional_edges(
                "monitoring_agent",
                self._route_from_monitoring,
                monitoring_routes
            )
    
    def _route_from_planner(self, state: VideoGenerationState) -> str:
        """Route from planner agent based on state."""
        # Check for workflow completion first
        if state.get('workflow_complete'):
            return "END"
        
        # Check for human intervention needs
        if state.get('pending_human_input') and "human_loop_agent" in self.agents:
            return "human_loop_agent"
        
        # Check for errors that need handling
        if state.get('error_count', 0) > 0 and "error_handler_agent" in self.agents:
            return "error_handler_agent"
        
        # Check if only planning was requested
        if state.get('only_plan', False):
            return "END"
        
        # Normal flow: planner -> code generator (if available)
        if "code_generator_agent" in self.agents:
            return "code_generator_agent"
        
        # Fallback to renderer if code generator not available
        if "renderer_agent" in self.agents:
            return "renderer_agent"
        
        return "END"
    
    def _route_from_code_generator(self, state: VideoGenerationState) -> str:
        """Route from code generator agent based on state."""
        # Check for workflow completion first
        if state.get('workflow_complete'):
            return "END"
        
        # Check for human intervention needs
        if state.get('pending_human_input') and "human_loop_agent" in self.agents:
            return "human_loop_agent"
        
        # Check for errors that need handling
        if state.get('error_count', 0) > 0 and "error_handler_agent" in self.agents:
            return "error_handler_agent"
        
        # Check if RAG context is needed and available
        if (state.get('use_rag') and 
            not state.get('rag_context') and 
            "rag_agent" in self.agents):
            return "rag_agent"
        
        # Check if code generation has errors that need visual analysis
        if (state.get('code_errors') and 
            state.get('use_visual_fix_code') and 
            "visual_analysis_agent" in self.agents):
            return "visual_analysis_agent"
        
        # Normal flow: code generator -> renderer (if available)
        if "renderer_agent" in self.agents:
            return "renderer_agent"
        
        return "END"
    
    def _route_from_renderer(self, state: VideoGenerationState) -> str:
        """Route from renderer agent based on state."""
        # Check for workflow completion first
        if state.get('workflow_complete'):
            return "END"
        
        # Check for human intervention needs
        if state.get('pending_human_input') and "human_loop_agent" in self.agents:
            return "human_loop_agent"
        
        # Check for errors that need handling
        if state.get('error_count', 0) > 0 and "error_handler_agent" in self.agents:
            return "error_handler_agent"
        
        # Check if visual analysis is needed for rendered videos
        if (state.get('use_visual_fix_code') and 
            (state.get('rendering_errors') or state.get('rendered_videos')) and 
            "visual_analysis_agent" in self.agents):
            return "visual_analysis_agent"
        
        # Check if monitoring is needed
        if (state.get('enable_monitoring', True) and 
            "monitoring_agent" in self.agents and
            not state.get('monitoring_complete', False)):
            return "monitoring_agent"
        
        # Normal completion - workflow is done
        return "END"
    
    def _route_from_visual_analysis(self, state: VideoGenerationState) -> str:
        """Route from visual analysis agent based on state."""
        if state.get('workflow_complete'):
            return "END"
        
        if state.get('pending_human_input'):
            return "human_loop_agent"
        
        if state.get('error_count', 0) > 0:
            return "error_handler_agent"
        
        # Visual analysis typically routes back to code generator for fixes
        return "code_generator_agent"
    
    def _route_from_rag(self, state: VideoGenerationState) -> str:
        """Route from RAG agent based on state."""
        if state.get('workflow_complete'):
            return "END"
        
        if state.get('error_count', 0) > 0:
            return "error_handler_agent"
        
        # RAG typically provides context back to code generator or planner
        next_agent = state.get('next_agent', 'code_generator_agent')
        return next_agent
    
    def _route_from_error_handler(self, state: VideoGenerationState) -> str:
        """Route from error handler agent based on state."""
        if state.get('workflow_complete'):
            return "END"
        
        if state.get('pending_human_input'):
            return "human_loop_agent"
        
        # Error handler determines recovery path
        next_agent = state.get('next_agent', 'END')
        return next_agent
    
    def _route_from_human_loop(self, state: VideoGenerationState) -> str:
        """Route from human loop agent based on state."""
        if state.get('workflow_complete'):
            return "END"
        
        # Human loop determines next step based on user input
        next_agent = state.get('next_agent', 'END')
        return next_agent
    
    def _route_from_monitoring(self, state: VideoGenerationState) -> str:
        """Route from monitoring agent based on state."""
        if state.get('workflow_complete'):
            return "END"
        
        # Monitoring typically returns to the previous agent
        next_agent = state.get('next_agent', 'END')
        return next_agent
    
    def _get_interrupt_points(self) -> List[str]:
        """Get list of nodes where workflow can be interrupted for human input."""
        interrupt_points = []
        
        # Add human loop interruption points
        if self.system_config.human_loop_config.enable_interrupts:
            interrupt_points.extend([
                "planner_agent",
                "code_generator_agent", 
                "renderer_agent"
            ])
        
        return interrupt_points
    
    def _get_interrupt_after_points(self) -> List[str]:
        """Get list of nodes where workflow should interrupt after execution."""
        return ["human_loop_agent"]
    
    async def invoke(self, 
                    topic: str, 
                    description: str, 
                    session_id: str,
                    config: Dict[str, Any] = None) -> VideoGenerationState:
        """Invoke the video generation workflow.
        
        Args:
            topic: Video topic
            description: Video description
            session_id: Session identifier
            config: Optional runtime configuration
            
        Returns:
            VideoGenerationState: Final workflow state
        """
        # Validate inputs
        if not topic or not topic.strip():
            raise ValueError("Topic cannot be empty")
        
        if not description or not description.strip():
            raise ValueError("Description cannot be empty")
        
        if not session_id or not session_id.strip():
            raise ValueError("Session ID cannot be empty")
        
        # Create initial state with validation
        initial_state = create_initial_state(topic, description, session_id, self.system_config)
        
        # Apply runtime configuration if provided
        if config:
            # Validate runtime configuration
            self._validate_runtime_config(config)
            initial_state.update(config)
        
        logger.info(f"Starting video generation workflow for topic: {topic} (session: {session_id})")
        
        try:
            # Set workflow start time for timeout handling
            import time
            start_time = time.time()
            
            # Execute the workflow with timeout
            final_state = await asyncio.wait_for(
                self.graph.ainvoke(
                    initial_state,
                    config={"configurable": {"thread_id": session_id}}
                ),
                timeout=self.system_config.workflow_config.workflow_timeout_seconds
            )
            
            # Calculate total execution time
            execution_time = time.time() - start_time
            
            # Mark workflow as complete
            final_state['workflow_complete'] = True
            final_state['total_execution_time'] = execution_time
            
            logger.info(f"Workflow completed for topic: {topic} in {execution_time:.2f}s")
            return final_state
            
        except asyncio.TimeoutError:
            logger.error(f"Workflow timed out for topic {topic} after {self.system_config.workflow_config.workflow_timeout_seconds}s")
            raise TimeoutError(f"Workflow execution exceeded timeout of {self.system_config.workflow_config.workflow_timeout_seconds} seconds")
            
        except Exception as e:
            logger.error(f"Workflow failed for topic {topic}: {e}")
            raise
    
    def _validate_runtime_config(self, config: Dict[str, Any]):
        """Validate runtime configuration parameters.
        
        Args:
            config: Runtime configuration to validate
        """
        # Validate boolean parameters
        boolean_params = [
            'use_rag', 'use_context_learning', 'use_visual_fix_code', 
            'use_langfuse', 'enable_caching', 'use_gpu_acceleration', 
            'preview_mode', 'only_plan'
        ]
        
        for param in boolean_params:
            if param in config and not isinstance(config[param], bool):
                raise ValueError(f"Parameter {param} must be a boolean")
        
        # Validate integer parameters
        integer_params = [
            'max_scene_concurrency', 'max_topic_concurrency', 'max_retries',
            'max_concurrent_renders', 'rag_cache_ttl', 'rag_max_cache_size'
        ]
        
        for param in integer_params:
            if param in config:
                if not isinstance(config[param], int) or config[param] < 0:
                    raise ValueError(f"Parameter {param} must be a non-negative integer")
        
        # Validate float parameters
        float_params = ['rag_performance_threshold', 'rag_quality_threshold']
        
        for param in float_params:
            if param in config:
                if not isinstance(config[param], (int, float)) or config[param] < 0:
                    raise ValueError(f"Parameter {param} must be a non-negative number")
        
        # Validate string parameters
        string_params = ['output_dir', 'default_quality', 'embedding_model']
        
        for param in string_params:
            if param in config and not isinstance(config[param], str):
                raise ValueError(f"Parameter {param} must be a string")
        
        # Validate specific scenes parameter
        if 'specific_scenes' in config:
            if not isinstance(config['specific_scenes'], list):
                raise ValueError("specific_scenes must be a list")
            
            for scene in config['specific_scenes']:
                if not isinstance(scene, int) or scene < 0:
                    raise ValueError("specific_scenes must contain non-negative integers")
        
        logger.debug("Runtime configuration validation passed")
    
    async def stream(self, 
                    topic: str, 
                    description: str, 
                    session_id: str,
                    config: Dict[str, Any] = None):
        """Stream the video generation workflow execution.
        
        Args:
            topic: Video topic
            description: Video description
            session_id: Session identifier
            config: Optional runtime configuration
            
        Yields:
            Dict: Workflow state updates
        """
        # Validate inputs
        if not topic or not topic.strip():
            raise ValueError("Topic cannot be empty")
        
        if not description or not description.strip():
            raise ValueError("Description cannot be empty")
        
        if not session_id or not session_id.strip():
            raise ValueError("Session ID cannot be empty")
        
        # Create initial state
        initial_state = create_initial_state(topic, description, session_id, self.system_config)
        
        # Apply runtime configuration if provided
        if config:
            self._validate_runtime_config(config)
            initial_state.update(config)
        
        logger.info(f"Starting streaming workflow for topic: {topic} (session: {session_id})")
        
        try:
            # Stream the workflow execution with enhanced error handling
            chunk_count = 0
            start_time = time.time()
            
            async for chunk in self.graph.astream(
                initial_state,
                config={"configurable": {"thread_id": session_id}}
            ):
                chunk_count += 1
                
                # Add metadata to chunk
                enhanced_chunk = {
                    **chunk,
                    'chunk_number': chunk_count,
                    'timestamp': time.time(),
                    'elapsed_time': time.time() - start_time,
                    'session_id': session_id
                }
                
                # Check for workflow completion
                if any(state.get('workflow_complete', False) for state in chunk.values() if isinstance(state, dict)):
                    enhanced_chunk['workflow_status'] = 'completed'
                elif any(state.get('error_count', 0) > 0 for state in chunk.values() if isinstance(state, dict)):
                    enhanced_chunk['workflow_status'] = 'error'
                else:
                    enhanced_chunk['workflow_status'] = 'running'
                
                yield enhanced_chunk
                
                # Check for timeout
                if time.time() - start_time > self.system_config.workflow_config.workflow_timeout_seconds:
                    logger.error(f"Streaming workflow timed out for topic {topic}")
                    yield {
                        'error': 'Workflow execution timed out',
                        'chunk_number': chunk_count + 1,
                        'timestamp': time.time(),
                        'elapsed_time': time.time() - start_time,
                        'session_id': session_id,
                        'workflow_status': 'timeout'
                    }
                    break
                
        except Exception as e:
            logger.error(f"Streaming workflow failed for topic {topic}: {e}")
            # Yield error information
            yield {
                'error': str(e),
                'error_type': type(e).__name__,
                'timestamp': time.time(),
                'session_id': session_id,
                'workflow_status': 'failed'
            }
            raise
    
    def get_workflow_status(self, session_id: str) -> Dict[str, Any]:
        """Get current workflow status for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict: Workflow status information
        """
        if not self.checkpointer:
            return {
                "status": "no_checkpointing_enabled",
                "session_id": session_id,
                "message": "Checkpointing is disabled, cannot retrieve workflow status"
            }
        
        try:
            # Get latest checkpoint
            config = {"configurable": {"thread_id": session_id}}
            checkpoint = self.checkpointer.get(config)
            
            if not checkpoint:
                return {
                    "status": "no_session_found",
                    "session_id": session_id,
                    "message": "No workflow session found for this ID"
                }
            
            # Extract state information from checkpoint
            state = checkpoint.get('channel_values', {})
            
            # Determine workflow status
            if state.get('workflow_complete', False):
                status = "completed"
            elif state.get('workflow_interrupted', False):
                status = "interrupted"
            elif state.get('pending_human_input'):
                status = "waiting_for_human_input"
            elif state.get('error_count', 0) > 0:
                status = "error"
            else:
                status = "running"
            
            return {
                "status": status,
                "session_id": session_id,
                "current_agent": state.get("current_agent"),
                "next_agent": state.get("next_agent"),
                "error_count": state.get("error_count", 0),
                "workflow_complete": state.get("workflow_complete", False),
                "workflow_interrupted": state.get("workflow_interrupted", False),
                "topic": state.get("topic"),
                "execution_trace": state.get("execution_trace", []),
                "performance_metrics": state.get("performance_metrics", {}),
                "pending_human_input": state.get("pending_human_input"),
                "retry_count": state.get("retry_count", {}),
                "checkpoint_timestamp": checkpoint.get('ts')
            }
            
        except Exception as e:
            logger.error(f"Failed to get workflow status for session {session_id}: {e}")
            return {
                "status": "error", 
                "session_id": session_id,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def interrupt_workflow(self, session_id: str, reason: str = "User requested") -> bool:
        """Interrupt a running workflow.
        
        Args:
            session_id: Session identifier
            reason: Reason for interruption
            
        Returns:
            bool: True if interrupted successfully
        """
        if not self.checkpointer:
            logger.warning(f"Cannot interrupt workflow {session_id}: checkpointing disabled")
            return False
        
        try:
            # Get current checkpoint
            config = {"configurable": {"thread_id": session_id}}
            checkpoint = self.checkpointer.get(config)
            
            if not checkpoint:
                logger.warning(f"Cannot interrupt workflow {session_id}: session not found")
                return False
            
            # Update state to mark as interrupted
            state = checkpoint.get('channel_values', {})
            state['workflow_interrupted'] = True
            state['interruption_reason'] = reason
            state['interruption_timestamp'] = time.time()
            
            # Save updated checkpoint
            checkpoint['channel_values'] = state
            self.checkpointer.put(config, checkpoint)
            
            logger.info(f"Workflow {session_id} interrupted: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to interrupt workflow {session_id}: {e}")
            return False
    
    def resume_workflow(self, session_id: str) -> bool:
        """Resume an interrupted workflow.
        
        Args:
            session_id: Session identifier
            
        Returns:
            bool: True if resumed successfully
        """
        if not self.checkpointer:
            logger.warning(f"Cannot resume workflow {session_id}: checkpointing disabled")
            return False
        
        try:
            # Get current checkpoint
            config = {"configurable": {"thread_id": session_id}}
            checkpoint = self.checkpointer.get(config)
            
            if not checkpoint:
                logger.warning(f"Cannot resume workflow {session_id}: session not found")
                return False
            
            # Update state to clear interruption
            state = checkpoint.get('channel_values', {})
            state['workflow_interrupted'] = False
            state.pop('interruption_reason', None)
            state.pop('interruption_timestamp', None)
            
            # Save updated checkpoint
            checkpoint['channel_values'] = state
            self.checkpointer.put(config, checkpoint)
            
            logger.info(f"Workflow {session_id} resumed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume workflow {session_id}: {e}")
            return False
"""
Workflow orchestration for LangGraph agents.
Provides centralized workflow management and coordination.
"""

from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
import logging
from datetime import datetime

from .state import VideoGenerationState, SystemConfig, create_initial_state
from .base_agent import AgentFactory

logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    """Orchestrates the execution of LangGraph agents in a workflow."""
    
    def __init__(self, system_config: SystemConfig):
        """Initialize the workflow orchestrator.
        
        Args:
            system_config: System configuration containing agent definitions
        """
        self.system_config = system_config
        self.agents = {}
        self.workflow_graph = None
        self._initialize_agents()
        self._build_workflow_graph()
    
    def _initialize_agents(self):
        """Initialize all agents from configuration."""
        for agent_name, agent_config in self.system_config.agents.items():
            try:
                agent = AgentFactory.create_agent(
                    agent_type=agent_name,
                    config=agent_config,
                    system_config=self.system_config.__dict__
                )
                self.agents[agent_name] = agent
                logger.info(f"Initialized agent: {agent_name}")
            except Exception as e:
                logger.error(f"Failed to initialize agent {agent_name}: {e}")
    
    def _build_workflow_graph(self):
        """Build the LangGraph workflow graph."""
        self.workflow_graph = StateGraph(VideoGenerationState)
        
        # Add agent nodes
        for agent_name, agent in self.agents.items():
            self.workflow_graph.add_node(agent_name, self._create_agent_node(agent))
        
        # Add workflow routing
        self._add_workflow_edges()
        
        # Compile the graph
        self.workflow_graph = self.workflow_graph.compile()
    
    def _create_agent_node(self, agent):
        """Create a workflow node for an agent.
        
        Args:
            agent: Agent instance
            
        Returns:
            Callable: Node function for the workflow graph
        """
        async def agent_node(state: VideoGenerationState) -> Dict[str, Any]:
            """Execute agent and return state updates."""
            try:
                logger.info(f"Executing agent: {agent.name}")
                
                # Execute agent with monitoring
                command = await agent.execute_with_monitoring(state)
                
                # Extract updates from command
                updates = {}
                if hasattr(command, 'update') and command.update:
                    updates.update(command.update)
                
                # Add agent tracking
                updates.update({
                    'current_agent': agent.name,
                    'next_agent': getattr(command, 'goto', None)
                })
                
                return updates
                
            except Exception as e:
                logger.error(f"Agent {agent.name} failed: {e}")
                return {
                    'current_agent': agent.name,
                    'error_count': state.get('error_count', 0) + 1,
                    'escalated_errors': state.get('escalated_errors', []) + [{
                        'agent': agent.name,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }],
                    'next_agent': 'error_handler_agent'
                }
        
        return agent_node
    
    def _add_workflow_edges(self):
        """Add edges to the workflow graph."""
        # Start with planner agent
        self.workflow_graph.add_edge(START, "planner_agent")
        
        # Add conditional routing based on agent decisions
        self.workflow_graph.add_conditional_edges(
            "planner_agent",
            self._route_from_planner,
            {
                "rag_agent": "rag_agent",
                "code_generator_agent": "code_generator_agent",
                "error_handler_agent": "error_handler_agent",
                "END": END
            }
        )
        
        self.workflow_graph.add_conditional_edges(
            "rag_agent",
            self._route_from_rag,
            {
                "code_generator_agent": "code_generator_agent",
                "error_handler_agent": "error_handler_agent",
                "END": END
            }
        )
        
        self.workflow_graph.add_conditional_edges(
            "code_generator_agent",
            self._route_from_code_generator,
            {
                "renderer_agent": "renderer_agent",
                "visual_analysis_agent": "visual_analysis_agent",
                "error_handler_agent": "error_handler_agent",
                "END": END
            }
        )
        
        self.workflow_graph.add_conditional_edges(
            "renderer_agent",
            self._route_from_renderer,
            {
                "visual_analysis_agent": "visual_analysis_agent",
                "monitoring_agent": "monitoring_agent",
                "error_handler_agent": "error_handler_agent",
                "END": END
            }
        )
        
        self.workflow_graph.add_conditional_edges(
            "visual_analysis_agent",
            self._route_from_visual_analysis,
            {
                "code_generator_agent": "code_generator_agent",
                "monitoring_agent": "monitoring_agent",
                "error_handler_agent": "error_handler_agent",
                "END": END
            }
        )
        
        self.workflow_graph.add_conditional_edges(
            "error_handler_agent",
            self._route_from_error_handler,
            {
                "planner_agent": "planner_agent",
                "code_generator_agent": "code_generator_agent",
                "renderer_agent": "renderer_agent",
                "human_loop_agent": "human_loop_agent",
                "monitoring_agent": "monitoring_agent",
                "END": END
            }
        )
        
        self.workflow_graph.add_conditional_edges(
            "human_loop_agent",
            self._route_from_human_loop,
            {
                "planner_agent": "planner_agent",
                "code_generator_agent": "code_generator_agent",
                "renderer_agent": "renderer_agent",
                "monitoring_agent": "monitoring_agent",
                "END": END
            }
        )
        
        # Monitoring agent typically ends the workflow
        self.workflow_graph.add_edge("monitoring_agent", END)
    
    def _route_from_planner(self, state: VideoGenerationState) -> str:
        """Route from planner agent."""
        if state.get('workflow_complete'):
            return "END"
        if state.get('error_count', 0) > 0:
            return "error_handler_agent"
        if state.get('use_rag', True):
            return "rag_agent"
        return "code_generator_agent"
    
    def _route_from_rag(self, state: VideoGenerationState) -> str:
        """Route from RAG agent."""
        if state.get('workflow_complete'):
            return "END"
        if state.get('error_count', 0) > 0:
            return "error_handler_agent"
        return "code_generator_agent"
    
    def _route_from_code_generator(self, state: VideoGenerationState) -> str:
        """Route from code generator agent."""
        if state.get('workflow_complete'):
            return "END"
        if state.get('error_count', 0) > 0:
            return "error_handler_agent"
        return "renderer_agent"
    
    def _route_from_renderer(self, state: VideoGenerationState) -> str:
        """Route from renderer agent."""
        if state.get('workflow_complete'):
            return "END"
        if state.get('error_count', 0) > 0:
            return "error_handler_agent"
        if state.get('use_visual_fix_code', False):
            return "visual_analysis_agent"
        return "monitoring_agent"
    
    def _route_from_visual_analysis(self, state: VideoGenerationState) -> str:
        """Route from visual analysis agent."""
        if state.get('workflow_complete'):
            return "END"
        if state.get('error_count', 0) > 0:
            return "error_handler_agent"
        
        # Check if visual issues require code regeneration
        visual_errors = state.get('visual_errors', {})
        if visual_errors:
            retry_count = state.get('retry_count', {}).get('visual_analysis_agent', 0)
            if retry_count < 2:  # Allow up to 2 retries
                return "code_generator_agent"
        
        return "monitoring_agent"
    
    def _route_from_error_handler(self, state: VideoGenerationState) -> str:
        """Route from error handler agent."""
        if state.get('workflow_complete'):
            return "END"
        
        # Check if human intervention is needed
        if state.get('pending_human_input'):
            return "human_loop_agent"
        
        # Route back to the agent that needs retry
        next_agent = state.get('next_agent')
        if next_agent and next_agent in self.agents:
            return next_agent
        
        return "monitoring_agent"
    
    def _route_from_human_loop(self, state: VideoGenerationState) -> str:
        """Route from human loop agent."""
        if state.get('workflow_complete'):
            return "END"
        
        # Route based on human decision
        human_feedback = state.get('human_feedback', {})
        decision = human_feedback.get('decision', 'continue')
        
        if decision == 'retry_from_start':
            return "planner_agent"
        elif decision == 'skip_to_monitoring':
            return "monitoring_agent"
        
        # Default: continue to monitoring
        return "monitoring_agent"
    
    async def execute_workflow(
        self,
        topic: str,
        description: str,
        session_id: str,
        initial_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute the complete workflow.
        
        Args:
            topic: Video topic
            description: Video description
            session_id: Unique session identifier
            initial_config: Additional configuration overrides
            
        Returns:
            Dict containing workflow results
        """
        try:
            # Create initial state
            initial_state = create_initial_state(
                topic=topic,
                description=description,
                session_id=session_id,
                config=self.system_config
            )
            
            # Apply any configuration overrides
            if initial_config:
                initial_state.update(initial_config)
            
            logger.info(f"Starting workflow for topic: {topic}")
            
            # Execute the workflow
            result = await self.workflow_graph.ainvoke(initial_state)
            
            logger.info("Workflow completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                'error': str(e),
                'workflow_complete': True,
                'success': False
            }
    
    def get_workflow_status(self, session_id: str) -> Dict[str, Any]:
        """Get the current status of a workflow.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict containing workflow status
        """
        # This would typically query a state store
        # For now, return a placeholder
        return {
            'session_id': session_id,
            'status': 'unknown',
            'message': 'Status tracking not implemented'
        }
    
    def list_available_agents(self) -> List[str]:
        """List all available agents.
        
        Returns:
            List of agent names
        """
        return list(self.agents.keys())
    
    def get_agent_info(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Dict containing agent information or None if not found
        """
        if agent_name not in self.agents:
            return None
        
        agent = self.agents[agent_name]
        return {
            'name': agent.name,
            'config': agent.config.__dict__,
            'execution_stats': agent.execution_stats
        }
"""
AgentAdapter for maintaining existing agent interfaces while using new node functions.

This adapter provides backward compatibility for existing agent-based code
while internally using the new simplified node function architecture.
"""

from typing import Dict, Any, Optional, List, Union, Callable, Awaitable
from datetime import datetime
import logging
import asyncio

from ..base_agent import BaseAgent
from ..models.state import VideoGenerationState as NewVideoGenerationState
from ..state import VideoGenerationState as OldVideoGenerationState, AgentConfig
from .state_adapter import StateAdapter
from langgraph.types import Command

logger = logging.getLogger(__name__)


class AgentAdapter:
    """
    Adapter that wraps new node functions to maintain old agent interfaces.
    
    This allows existing code that expects agent classes to continue working
    while internally using the new simplified node function architecture.
    """
    
    def __init__(self, node_function: Callable[[NewVideoGenerationState], Awaitable[NewVideoGenerationState]],
                 agent_name: str, config: AgentConfig, system_config: Dict[str, Any]):
        """
        Initialize agent adapter.
        
        Args:
            node_function: New node function to wrap
            agent_name: Name of the agent
            config: Agent configuration
            system_config: System configuration
        """
        self.node_function = node_function
        self.agent_name = agent_name
        self.config = config
        self.system_config = system_config
        
        # Create a mock BaseAgent for compatibility
        self._mock_agent = MockAgent(config, system_config)
        
        logger.info(f"Initialized AgentAdapter for {agent_name}")
    
    async def execute(self, state: Union[OldVideoGenerationState, Dict[str, Any]]) -> Command:
        """
        Execute the wrapped node function with state conversion.
        
        Args:
            state: State in old format
            
        Returns:
            Command: LangGraph command for next action
        """
        try:
            # Convert old state to new format
            if isinstance(state, dict):
                new_state = StateAdapter.old_to_new(state)
            else:
                new_state = StateAdapter.old_to_new(dict(state))
            
            # Execute the node function
            updated_state = await self.node_function(new_state)
            
            # Convert back to old format for compatibility
            old_state = StateAdapter.new_to_old(updated_state)
            
            # Determine next action based on state changes
            next_agent = self._determine_next_agent(updated_state)
            
            # Create command with state updates
            return Command(
                goto=next_agent,
                update=dict(old_state)
            )
            
        except Exception as e:
            logger.error(f"Error in AgentAdapter.execute for {self.agent_name}: {e}")
            return await self._mock_agent.handle_error(e, state)
    
    def _determine_next_agent(self, state: NewVideoGenerationState) -> Optional[str]:
        """Determine next agent based on current step and state."""
        current_step = state.current_step
        
        # Handle completion
        if state.workflow_complete:
            return None
        
        # Handle errors
        if state.has_errors() and current_step != 'error_handling':
            return 'error_handler_agent'
        
        # Normal workflow progression
        step_to_agent = {
            'planning': 'code_generator_agent',
            'code_generation': 'renderer_agent',
            'rendering': 'visual_analysis_agent' if state.config.use_visual_analysis else None,
            'visual_analysis': None,
            'error_handling': 'planner_agent',
            'human_loop': 'planner_agent'
        }
        
        return step_to_agent.get(current_step)
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to mock agent for compatibility."""
        return getattr(self._mock_agent, name)


class MockAgent(BaseAgent):
    """
    Mock agent that provides BaseAgent interface for compatibility.
    
    This class implements the BaseAgent interface but delegates actual
    execution to the wrapped node functions through AgentAdapter.
    """
    
    async def execute(self, state: NewVideoGenerationState) -> Command:
        """Mock execute method - should not be called directly."""
        raise NotImplementedError("MockAgent.execute should not be called directly")


class LegacyAgentWrapper:
    """
    Wrapper that makes new node functions look like old agent classes.
    
    This provides a complete agent-like interface for maximum compatibility
    with existing code that expects agent instances.
    """
    
    def __init__(self, node_function: Callable[[NewVideoGenerationState], Awaitable[NewVideoGenerationState]],
                 agent_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize legacy agent wrapper.
        
        Args:
            node_function: New node function to wrap
            agent_name: Name of the agent
            config: Optional configuration
        """
        self.node_function = node_function
        self.name = agent_name
        self.config = config or {}
        
        # Agent-like attributes for compatibility
        self.max_retries = self.config.get('max_retries', 3)
        self.timeout_seconds = self.config.get('timeout_seconds', 300)
        self.enable_human_loop = self.config.get('enable_human_loop', False)
        
        # Model configurations for compatibility
        self.model_config = self.config.get('model_config', {})
        self.planner_model = self.config.get('planner_model')
        self.scene_model = self.config.get('scene_model')
        self.helper_model = self.config.get('helper_model')
        
        # Performance tracking
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0.0,
            'total_execution_time': 0.0
        }
        
        logger.info(f"Initialized LegacyAgentWrapper for {agent_name}")
    
    async def execute(self, state: Union[OldVideoGenerationState, Dict[str, Any], NewVideoGenerationState]) -> Any:
        """
        Execute the wrapped node function with automatic state handling.
        
        Args:
            state: State in any supported format
            
        Returns:
            Updated state or Command
        """
        start_time = datetime.now()
        
        try:
            self.execution_stats['total_executions'] += 1
            
            # Handle different state formats
            if isinstance(state, NewVideoGenerationState):
                new_state = state
            elif isinstance(state, dict):
                new_state = StateAdapter.old_to_new(state)
            else:
                new_state = StateAdapter.old_to_new(dict(state))
            
            # Execute the node function
            updated_state = await self.node_function(new_state)
            
            # Update execution stats
            self.execution_stats['successful_executions'] += 1
            execution_time = (datetime.now() - start_time).total_seconds()
            self.execution_stats['total_execution_time'] += execution_time
            self.execution_stats['average_execution_time'] = (
                self.execution_stats['total_execution_time'] / 
                self.execution_stats['total_executions']
            )
            
            # Return in the same format as input
            if isinstance(state, NewVideoGenerationState):
                return updated_state
            else:
                return StateAdapter.new_to_old(updated_state)
                
        except Exception as e:
            self.execution_stats['failed_executions'] += 1
            logger.error(f"Error in LegacyAgentWrapper.execute for {self.name}: {e}")
            raise
    
    async def process(self, *args, **kwargs) -> Any:
        """Legacy process method for compatibility."""
        # Try to extract state from arguments
        state = None
        if args and len(args) > 0:
            state = args[0]
        elif 'state' in kwargs:
            state = kwargs['state']
        
        if state is None:
            raise ValueError("No state provided to process method")
        
        return await self.execute(state)
    
    def get_model_wrapper(self, model_name: str, state: Any = None):
        """Get model wrapper for compatibility."""
        # Import existing model wrappers
        from mllm_tools.litellm import LiteLLMWrapper
        from mllm_tools.openrouter import OpenRouterWrapper
        from mllm_tools.openai import OpenAIWrapper
        
        # Simple model wrapper creation for compatibility
        try:
            if model_name.startswith('openrouter/'):
                return OpenRouterWrapper(
                    model_name=model_name,
                    temperature=0.7,
                    print_cost=True,
                    verbose=False,
                    use_langfuse=True
                )
            elif 'gpt' in model_name.lower():
                return OpenAIWrapper(
                    model_name=model_name,
                    temperature=0.7,
                    print_cost=True,
                    verbose=False,
                    use_langfuse=True,
                    use_github_token=False
                )
            else:
                return LiteLLMWrapper(
                    model_name=model_name,
                    temperature=0.7,
                    print_cost=True,
                    verbose=False,
                    use_langfuse=True
                )
        except Exception as e:
            logger.error(f"Failed to create model wrapper for {model_name}: {e}")
            # Return a basic wrapper as fallback
            return LiteLLMWrapper(
                model_name="gpt-4o-mini",
                temperature=0.7,
                print_cost=True,
                verbose=False,
                use_langfuse=True
            )
    
    def should_escalate_to_human(self, state: Any) -> bool:
        """Determine if human intervention is needed."""
        if not self.enable_human_loop:
            return False
        
        # Convert state if needed
        if not isinstance(state, NewVideoGenerationState):
            try:
                if isinstance(state, dict):
                    new_state = StateAdapter.old_to_new(state)
                else:
                    new_state = StateAdapter.old_to_new(dict(state))
            except:
                return False
        else:
            new_state = state
        
        # Check error conditions
        if len(new_state.errors) >= 3:
            return True
        
        retry_count = new_state.retry_counts.get(self.name, 0)
        if retry_count >= self.max_retries:
            return True
        
        return False
    
    def log_agent_action(self, action: str, details: Dict[str, Any] = None):
        """Log agent action for compatibility."""
        log_entry = {
            'agent': self.name,
            'action': action,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        
        logger.info(f"Agent {self.name}: {action}", extra=log_entry)


class AgentFactory:
    """
    Factory for creating agent adapters and wrappers.
    
    Provides a unified interface for creating both new node-based agents
    and legacy-compatible agent wrappers.
    """
    
    @staticmethod
    def create_adapter(node_function: Callable[[NewVideoGenerationState], Awaitable[NewVideoGenerationState]],
                      agent_name: str, config: AgentConfig, system_config: Dict[str, Any]) -> AgentAdapter:
        """
        Create an AgentAdapter for a node function.
        
        Args:
            node_function: Node function to wrap
            agent_name: Name of the agent
            config: Agent configuration
            system_config: System configuration
            
        Returns:
            AgentAdapter: Configured adapter
        """
        return AgentAdapter(node_function, agent_name, config, system_config)
    
    @staticmethod
    def create_legacy_wrapper(node_function: Callable[[NewVideoGenerationState], Awaitable[NewVideoGenerationState]],
                             agent_name: str, config: Optional[Dict[str, Any]] = None) -> LegacyAgentWrapper:
        """
        Create a LegacyAgentWrapper for a node function.
        
        Args:
            node_function: Node function to wrap
            agent_name: Name of the agent
            config: Optional configuration
            
        Returns:
            LegacyAgentWrapper: Configured wrapper
        """
        return LegacyAgentWrapper(node_function, agent_name, config)
    
    @staticmethod
    def wrap_all_nodes(node_functions: Dict[str, Callable[[NewVideoGenerationState], Awaitable[NewVideoGenerationState]]],
                      configs: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, LegacyAgentWrapper]:
        """
        Wrap multiple node functions as legacy agents.
        
        Args:
            node_functions: Dictionary of node functions by name
            configs: Optional configurations by agent name
            
        Returns:
            Dict[str, LegacyAgentWrapper]: Dictionary of wrapped agents
        """
        wrapped_agents = {}
        configs = configs or {}
        
        for agent_name, node_function in node_functions.items():
            config = configs.get(agent_name, {})
            wrapped_agents[agent_name] = AgentFactory.create_legacy_wrapper(
                node_function, agent_name, config
            )
        
        return wrapped_agents
    
    @staticmethod
    def create_compatible_agent(agent_type: str, node_functions: Dict[str, Callable],
                               config: Optional[Dict[str, Any]] = None) -> LegacyAgentWrapper:
        """
        Create a compatible agent wrapper for a specific agent type.
        
        Args:
            agent_type: Type of agent to create
            node_functions: Available node functions
            config: Optional configuration
            
        Returns:
            LegacyAgentWrapper: Compatible agent wrapper
        """
        # Map agent types to node function names
        agent_to_node_mapping = {
            'planner_agent': 'planning_node',
            'code_generator_agent': 'code_generation_node',
            'renderer_agent': 'rendering_node',
            'visual_analysis_agent': 'visual_analysis_node',
            'error_handler_agent': 'error_handler_node',
            'human_loop_agent': 'human_loop_node',
            'monitoring_agent': 'monitoring_node'
        }
        
        node_name = agent_to_node_mapping.get(agent_type)
        if not node_name or node_name not in node_functions:
            raise ValueError(f"Unknown agent type or missing node function: {agent_type}")
        
        node_function = node_functions[node_name]
        return AgentFactory.create_legacy_wrapper(node_function, agent_type, config)
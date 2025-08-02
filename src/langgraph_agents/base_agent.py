"""
Base agent interface and abstract classes compatible with existing agent patterns.
Provides foundation for LangGraph multi-agent system while maintaining compatibility.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Optional, List
from langgraph.types import Command
from datetime import datetime
import logging
import traceback
import asyncio

from .state import VideoGenerationState, AgentConfig, AgentError
from .services.langfuse_service import get_langfuse_service, trace_agent_method


logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base interface for all agents in the LangGraph multi-agent system.
    
    Provides common functionality while maintaining compatibility with
    existing agent patterns from EnhancedVideoPlanner, CodeGenerator, etc.
    """
    
    def __init__(self, config: AgentConfig, system_config: Dict[str, Any]):
        """Initialize base agent with configuration.
        
        Args:
            config: Agent-specific configuration
            system_config: System-wide configuration
        """
        self.config = config
        self.system_config = system_config
        self.name = config.name
        self.max_retries = config.max_retries
        self.timeout_seconds = config.timeout_seconds
        self.enable_human_loop = config.enable_human_loop
        
        # Model configurations - compatible with existing patterns
        self.model_config = config.model_config
        self.planner_model = getattr(config, 'planner_model', None)
        self.scene_model = getattr(config, 'scene_model', None)
        self.helper_model = getattr(config, 'helper_model', None)
        
        # Performance tracking
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0.0,
            'total_execution_time': 0.0
        }
        
        logger.info(f"Initialized {self.name} with config: {config}")
    
    @abstractmethod
    async def execute(self, state: VideoGenerationState) -> Command:
        """Execute the agent's primary function.
        
        Args:
            state: Current workflow state
            
        Returns:
            Command: LangGraph command for next action
        """
        pass
    
    async def handle_error(self, error: Exception, state: VideoGenerationState) -> Command:
        """Handle errors specific to this agent.
        
        Args:
            error: Exception that occurred
            state: Current workflow state
            
        Returns:
            Command: LangGraph command for error handling
        """
        agent_error = AgentError(
            agent_name=self.name,
            error_type=type(error).__name__,
            error_message=str(error),
            context={
                'topic': state.get('topic', ''),
                'session_id': state.get('session_id', ''),
                'current_agent': state.get('current_agent', ''),
                'retry_count': state.get('retry_count', {}).get(self.name, 0)
            },
            timestamp=datetime.now(),
            retry_count=state.get('retry_count', {}).get(self.name, 0),
            stack_trace=traceback.format_exc()
        )
        
        logger.error(f"Error in {self.name}: {error}")
        
        # Update error tracking in state
        escalated_errors = state.get('escalated_errors', [])
        escalated_errors.append({
            'agent': self.name,
            'error': str(error),
            'timestamp': agent_error.timestamp.isoformat(),
            'retry_count': agent_error.retry_count
        })
        
        return Command(
            goto="error_handler_agent",
            update={
                "escalated_errors": escalated_errors,
                "error_count": state.get('error_count', 0) + 1,
                "current_agent": "error_handler_agent"
            }
        )
    
    @trace_agent_method("execute_with_monitoring")
    async def execute_with_monitoring(self, state: VideoGenerationState) -> Command:
        """Execute agent with performance monitoring and error handling.
        
        Args:
            state: Current workflow state
            
        Returns:
            Command: LangGraph command for next action
        """
        start_time = datetime.now()
        langfuse_service = get_langfuse_service()
        
        try:
            # Update execution tracking
            self.execution_stats['total_executions'] += 1
            
            # Start LangFuse tracing for agent execution
            if langfuse_service and langfuse_service.is_enabled():
                await langfuse_service.trace_agent_execution(
                    agent_name=self.name,
                    action="start_execution",
                    state=state,
                    input_data={
                        'agent_config': {
                            'max_retries': self.max_retries,
                            'timeout_seconds': self.timeout_seconds,
                            'enable_human_loop': self.enable_human_loop
                        },
                        'execution_stats': self.execution_stats
                    }
                )
            
            # Add execution trace
            execution_trace = state.get('execution_trace', [])
            execution_trace.append({
                'agent': self.name,
                'action': 'start_execution',
                'timestamp': start_time.isoformat(),
                'state_snapshot': {
                    'topic': state.get('topic', ''),
                    'session_id': state.get('session_id', ''),
                    'error_count': state.get('error_count', 0)
                }
            })
            
            # Execute with timeout
            try:
                command = await asyncio.wait_for(
                    self.execute(state),
                    timeout=self.timeout_seconds
                )
                
                # Update success stats
                self.execution_stats['successful_executions'] += 1
                
                # Calculate execution time
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                self.execution_stats['total_execution_time'] += execution_time
                self.execution_stats['average_execution_time'] = (
                    self.execution_stats['total_execution_time'] / 
                    self.execution_stats['total_executions']
                )
                
                # Track performance metrics in LangFuse
                if langfuse_service and langfuse_service.is_enabled():
                    await langfuse_service.track_performance_metrics(
                        agent_name=self.name,
                        metrics={
                            'execution_time': execution_time,
                            'success_rate': self.execution_stats['successful_executions'] / self.execution_stats['total_executions'],
                            'average_execution_time': self.execution_stats['average_execution_time'],
                            'total_executions': self.execution_stats['total_executions']
                        },
                        state=state
                    )
                
                # Add completion trace
                execution_trace.append({
                    'agent': self.name,
                    'action': 'complete_execution',
                    'timestamp': end_time.isoformat(),
                    'execution_time': execution_time,
                    'next_agent': getattr(command, 'goto', None)
                })
                
                # Trace completion in LangFuse
                if langfuse_service and langfuse_service.is_enabled():
                    await langfuse_service.trace_agent_execution(
                        agent_name=self.name,
                        action="complete_execution",
                        state=state,
                        output_data={
                            'execution_time': execution_time,
                            'next_agent': getattr(command, 'goto', None),
                            'success': True
                        }
                    )
                
                # Update state with monitoring data
                if hasattr(command, 'update') and command.update:
                    command.update.update({
                        'execution_trace': execution_trace,
                        'performance_metrics': {
                            **state.get('performance_metrics', {}),
                            self.name: {
                                'last_execution_time': execution_time,
                                'average_execution_time': self.execution_stats['average_execution_time'],
                                'success_rate': (
                                    self.execution_stats['successful_executions'] / 
                                    self.execution_stats['total_executions']
                                )
                            }
                        }
                    })
                else:
                    # Create update if it doesn't exist
                    command = Command(
                        goto=getattr(command, 'goto', None),
                        update={
                            'execution_trace': execution_trace,
                            'performance_metrics': {
                                **state.get('performance_metrics', {}),
                                self.name: {
                                    'last_execution_time': execution_time,
                                    'average_execution_time': self.execution_stats['average_execution_time'],
                                    'success_rate': (
                                        self.execution_stats['successful_executions'] / 
                                        self.execution_stats['total_executions']
                                    )
                                }
                            }
                        }
                    )
                
                return command
                
            except asyncio.TimeoutError:
                raise TimeoutError(f"Agent {self.name} execution timed out after {self.timeout_seconds} seconds")
                
        except Exception as error:
            # Update failure stats
            self.execution_stats['failed_executions'] += 1
            
            # Add error trace
            execution_trace.append({
                'agent': self.name,
                'action': 'error_occurred',
                'timestamp': datetime.now().isoformat(),
                'error': str(error),
                'error_type': type(error).__name__
            })
            
            # Track error in LangFuse
            if langfuse_service and langfuse_service.is_enabled():
                agent_error = AgentError(
                    agent_name=self.name,
                    error_type=type(error).__name__,
                    error_message=str(error),
                    context={
                        'topic': state.get('topic', ''),
                        'session_id': state.get('session_id', ''),
                        'current_agent': state.get('current_agent', ''),
                        'retry_count': state.get('retry_count', {}).get(self.name, 0)
                    },
                    timestamp=datetime.now(),
                    retry_count=state.get('retry_count', {}).get(self.name, 0),
                    stack_trace=traceback.format_exc()
                )
                
                await langfuse_service.track_error(agent_error, state)
            
            # Handle error
            return await self.handle_error(error, state)
    
    def get_model_wrapper(self, model_name: str, state: VideoGenerationState):
        """Get model wrapper compatible with existing patterns with fallback support.
        
        Args:
            model_name: Name of the model to create wrapper for
            state: Current workflow state
            
        Returns:
            Model wrapper compatible with existing CodeGenerator, etc.
        """
        # Import existing model wrappers
        from mllm_tools.litellm import LiteLLMWrapper
        from mllm_tools.openrouter import OpenRouterWrapper
        from mllm_tools.openai import OpenAIWrapper
        from mllm_tools.gemini import GeminiWrapper
        
        # Get model configuration from centralized config manager
        from src.config.manager import ConfigurationManager
        config_manager = ConfigurationManager()
        
        # Try to get the requested model first, with fallback logic
        try:
            model_config = config_manager.get_model_config(model_name)
            
            # Use specific wrapper based on model type
            if model_name.startswith('openrouter/'):
                return OpenRouterWrapper(
                    model_name=model_name,
                    temperature=self.config.temperature,
                    print_cost=self.config.print_cost,
                    verbose=state.get('print_response', False),
                    use_langfuse=state.get('use_langfuse', True),
                    api_key=model_config.get('api_key'),
                    base_url=model_config.get('base_url'),
                    timeout=model_config.get('timeout'),
                    max_retries=model_config.get('max_retries')
                )
            elif model_name.startswith('openai/') or 'gpt' in model_name.lower():
                return OpenAIWrapper(
                    model_name=model_name,
                    temperature=self.config.temperature,
                    print_cost=self.config.print_cost,
                    verbose=state.get('print_response', False),
                    use_langfuse=state.get('use_langfuse', True),
                    api_key=model_config.get('api_key'),
                    base_url=model_config.get('base_url'),
                    use_github_token=False  # Use regular OpenAI API
                )
            elif model_name.startswith('gemini/') or 'gemini' in model_name.lower():
                return GeminiWrapper(
                    model_name=model_name,
                    temperature=self.config.temperature,
                    print_cost=self.config.print_cost,
                    verbose=state.get('print_response', False),
                    use_langfuse=state.get('use_langfuse', True),
                    api_key=model_config.get('api_key')
                )
            else:
                # Use LiteLLM wrapper for other models
                return LiteLLMWrapper(
                    model_name=model_name,
                    temperature=self.config.temperature,
                    print_cost=self.config.print_cost,
                    verbose=state.get('print_response', False),
                    use_langfuse=state.get('use_langfuse', True),
                    api_key=model_config.get('api_key'),
                    base_url=model_config.get('base_url'),
                    timeout=model_config.get('timeout'),
                    max_retries=model_config.get('max_retries')
                )
                
        except Exception as e:
            logger.warning(f"Failed to create wrapper for {model_name}: {e}")
            # Fallback to default provider
            return self._get_fallback_model_wrapper(state)
    
    def _get_fallback_model_wrapper(self, state: VideoGenerationState):
        """Get fallback model wrapper when primary model fails.
        
        Args:
            state: Current workflow state
            
        Returns:
            Fallback model wrapper
        """
        from mllm_tools.litellm import LiteLLMWrapper
        from src.config.manager import ConfigurationManager
        
        config_manager = ConfigurationManager()
        
        # Try to get default provider and its fallback models
        default_provider = config_manager.get_default_provider()
        provider_config = config_manager.get_provider_config(default_provider)
        
        if provider_config and provider_config.enabled:
            fallback_model = f"{default_provider}/{provider_config.default_model}"
            logger.info(f"Using fallback model: {fallback_model}")
            
            try:
                return LiteLLMWrapper(
                    model_name=fallback_model,
                    temperature=self.config.temperature,
                    print_cost=self.config.print_cost,
                    verbose=state.get('print_response', False),
                    use_langfuse=state.get('use_langfuse', True),
                    api_key=provider_config.api_key,
                    base_url=provider_config.base_url,
                    timeout=provider_config.timeout,
                    max_retries=provider_config.max_retries
                )
            except Exception as e:
                logger.error(f"Fallback model also failed: {e}")
        
        # Last resort: try to create a basic OpenAI wrapper
        try:
            from mllm_tools.openai import OpenAIWrapper
            logger.warning("Using last resort OpenAI wrapper with environment variables")
            return OpenAIWrapper(
                model_name="gpt-4o-mini",  # Most cost-effective fallback
                temperature=self.config.temperature,
                print_cost=self.config.print_cost,
                verbose=state.get('print_response', False),
                use_langfuse=state.get('use_langfuse', True),
                use_github_token=False
            )
        except Exception as e:
            logger.error(f"All model creation attempts failed: {e}")
            raise RuntimeError(f"Unable to create any model wrapper. Please check your configuration and API keys.")
    
    def should_escalate_to_human(self, state: VideoGenerationState) -> bool:
        """Determine if current situation requires human intervention.
        
        Args:
            state: Current workflow state
            
        Returns:
            bool: True if human intervention is needed
        """
        if not self.enable_human_loop:
            return False
        
        # Check error count threshold
        error_count = state.get('error_count', 0)
        if error_count >= 3:
            return True
        
        # Check retry count for this agent
        retry_count = state.get('retry_count', {}).get(self.name, 0)
        if retry_count >= self.max_retries:
            return True
        
        # Check for specific error patterns that require human input
        escalated_errors = state.get('escalated_errors', [])
        recent_errors = [e for e in escalated_errors if e.get('agent') == self.name]
        
        if len(recent_errors) >= 2:
            return True
        
        return False
    
    def create_human_intervention_command(self, 
                                        context: str, 
                                        options: List[str],
                                        state: VideoGenerationState,
                                        priority: str = "medium",
                                        timeout_seconds: int = 300) -> Command:
        """Create command for human intervention.
        
        Args:
            context: Context for human decision
            options: Available options for human to choose from
            state: Current workflow state
            priority: Priority level (low, medium, high, critical)
            timeout_seconds: Timeout for human response
            
        Returns:
            Command: LangGraph command for human loop
        """
        from .tools.human_intervention_tools import request_human_approval
        
        return request_human_approval(
            context=context,
            options=options,
            priority=priority,
            timeout_seconds=timeout_seconds,
            default_action=options[0] if options else "continue",
            state=state
        )
    
    def get_human_intervention_interface(self):
        """Get human intervention interface for this agent.
        
        Returns:
            HumanInterventionInterface: Interface for human intervention
        """
        from .interfaces.human_intervention_interface import HumanInterventionInterface
        
        if not hasattr(self, '_human_interface'):
            self._human_interface = HumanInterventionInterface(
                config=self.system_config.get('human_loop_config', {})
            )
        
        return self._human_interface
    
    def log_agent_action(self, action: str, details: Dict[str, Any] = None):
        """Log agent action for debugging and monitoring.
        
        Args:
            action: Action being performed
            details: Additional details about the action
        """
        log_entry = {
            'agent': self.name,
            'action': action,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        
        logger.info(f"Agent {self.name}: {action}", extra=log_entry)


class AgentFactory:
    """Factory for creating agents with proper configuration.
    
    Maintains compatibility with existing component creation patterns
    while supporting LangGraph multi-agent architecture.
    """
    
    @staticmethod
    def create_agent(agent_type: str, config: AgentConfig, system_config: Dict[str, Any]) -> BaseAgent:
        """Create agent instance based on type.
        
        Args:
            agent_type: Type of agent to create
            config: Agent configuration
            system_config: System configuration
            
        Returns:
            BaseAgent: Configured agent instance
        """
        # Import agent implementations
        from .agents.planner_agent import PlannerAgent
        from .agents.code_generator_agent import CodeGeneratorAgent
        from .agents.renderer_agent import RendererAgent
        from .agents.visual_analysis_agent import VisualAnalysisAgent
        from .agents.rag_agent import RAGAgent
        from .agents.error_handler_agent import ErrorHandlerAgent
        from .agents.monitoring_agent import MonitoringAgent
        from .agents.human_loop_agent import HumanLoopAgent
        
        agent_classes = {
            'planner_agent': PlannerAgent,
            'code_generator_agent': CodeGeneratorAgent,
            'renderer_agent': RendererAgent,
            'visual_analysis_agent': VisualAnalysisAgent,
            'rag_agent': RAGAgent,
            'error_handler_agent': ErrorHandlerAgent,
            'monitoring_agent': MonitoringAgent,
            'human_loop_agent': HumanLoopAgent
        }
        
        if agent_type not in agent_classes:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        agent_class = agent_classes[agent_type]
        return agent_class(config, system_config)
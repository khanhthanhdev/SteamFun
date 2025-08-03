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
    
    async def execute_with_monitoring(self, state: VideoGenerationState) -> Command:
        """Execute agent with performance monitoring and error handling.
        
        Args:
            state: Current workflow state
            
        Returns:
            Command: LangGraph command for next action
        """
        start_time = datetime.now()
        
        try:
            # Update execution tracking
            self.execution_stats['total_executions'] += 1
            
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
                
                # Add completion trace
                execution_trace.append({
                    'agent': self.name,
                    'action': 'complete_execution',
                    'timestamp': end_time.isoformat(),
                    'execution_time': execution_time,
                    'next_agent': getattr(command, 'goto', None)
                })
                
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
            
            # Handle error
            return await self.handle_error(error, state)
    
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
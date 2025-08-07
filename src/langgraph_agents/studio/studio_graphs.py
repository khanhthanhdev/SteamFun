"""
Studio-compatible graph definitions for workflow testing.

This module provides pre-configured graphs that can be used in LangGraph Studio
for testing different aspects of the workflow system.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from ..models.state import VideoGenerationState
from ..models.config import WorkflowConfig
from ..workflow_graph import VideoGenerationWorkflow, create_workflow
from ..nodes.planning_node import planning_node
from ..nodes.code_generation_node import code_generation_node
from ..nodes.rendering_node import rendering_node
from ..nodes.error_handler_node import error_handler_node
from .studio_integration import studio_monitor, studio_registry
from .studio_workflow_config import StudioWorkflowConfig
from .studio_workflow_testing import get_studio_workflow_tester

logger = logging.getLogger(__name__)


def create_monitored_node(node_name: str, original_function: callable) -> callable:
    """Create a monitored version of a node function for Studio testing."""
    async def monitored_node_function(state: VideoGenerationState) -> VideoGenerationState:
        session_id = state.session_id
        
        # Start monitoring
        studio_monitor.start_session(f"{session_id}_{node_name}", node_name)
        
        # Record start
        studio_monitor.record_performance_metric(
            node_name,
            "node_execution_start",
            {
                'session_id': session_id,
                'current_step': state.current_step,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        try:
            # Execute original function
            start_time = datetime.now()
            result_state = await original_function(state)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Record success metrics
            studio_monitor.record_performance_metric(
                node_name,
                "execution_time",
                execution_time
            )
            
            studio_monitor.record_performance_metric(
                node_name,
                "node_execution_success",
                {
                    'session_id': session_id,
                    'execution_time': execution_time,
                    'has_errors': result_state.has_errors(),
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            # End monitoring
            studio_monitor.end_session(f"{session_id}_{node_name}", "completed")
            
            return result_state
            
        except Exception as e:
            # Record error
            studio_monitor.record_performance_metric(
                node_name,
                "node_execution_error",
                {
                    'session_id': session_id,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            # End monitoring with error
            studio_monitor.end_session(f"{session_id}_{node_name}", "failed")
            
            raise
    
    return monitored_node_function


# Individual Agent Test Graphs
def create_planning_agent_graph() -> StateGraph:
    """Create graph for testing planning agent in isolation."""
    graph = StateGraph(VideoGenerationState)
    
    # Add monitored planning node
    monitored_planning = create_monitored_node("planning", planning_node)
    graph.add_node("planning", monitored_planning)
    
    # Simple linear flow
    graph.add_edge(START, "planning")
    graph.add_edge("planning", END)
    
    return graph.compile(checkpointer=MemorySaver())


def create_code_generation_agent_graph() -> StateGraph:
    """Create graph for testing code generation agent in isolation."""
    graph = StateGraph(VideoGenerationState)
    
    # Add monitored code generation node
    monitored_code_gen = create_monitored_node("code_generation", code_generation_node)
    graph.add_node("code_generation", monitored_code_gen)
    
    # Simple linear flow
    graph.add_edge(START, "code_generation")
    graph.add_edge("code_generation", END)
    
    return graph.compile(checkpointer=MemorySaver())


def create_rendering_agent_graph() -> StateGraph:
    """Create graph for testing rendering agent in isolation."""
    graph = StateGraph(VideoGenerationState)
    
    # Add monitored rendering node
    monitored_rendering = create_monitored_node("rendering", rendering_node)
    graph.add_node("rendering", monitored_rendering)
    
    # Simple linear flow
    graph.add_edge(START, "rendering")
    graph.add_edge("rendering", END)
    
    return graph.compile(checkpointer=MemorySaver())


def create_error_handler_agent_graph() -> StateGraph:
    """Create graph for testing error handler agent in isolation."""
    graph = StateGraph(VideoGenerationState)
    
    # Add monitored error handler node
    monitored_error_handler = create_monitored_node("error_handler", error_handler_node)
    graph.add_node("error_handler", monitored_error_handler)
    
    # Simple linear flow
    graph.add_edge(START, "error_handler")
    graph.add_edge("error_handler", END)
    
    return graph.compile(checkpointer=MemorySaver())


# Agent Chain Test Graphs
def create_planning_to_code_chain_graph() -> StateGraph:
    """Create graph for testing planning -> code generation chain."""
    graph = StateGraph(VideoGenerationState)
    
    # Add monitored nodes
    monitored_planning = create_monitored_node("planning", planning_node)
    monitored_code_gen = create_monitored_node("code_generation", code_generation_node)
    
    graph.add_node("planning", monitored_planning)
    graph.add_node("code_generation", monitored_code_gen)
    
    # Chain them together
    graph.add_edge(START, "planning")
    graph.add_edge("planning", "code_generation")
    graph.add_edge("code_generation", END)
    
    return graph.compile(checkpointer=MemorySaver())


def create_code_to_rendering_chain_graph() -> StateGraph:
    """Create graph for testing code generation -> rendering chain."""
    graph = StateGraph(VideoGenerationState)
    
    # Add monitored nodes
    monitored_code_gen = create_monitored_node("code_generation", code_generation_node)
    monitored_rendering = create_monitored_node("rendering", rendering_node)
    
    graph.add_node("code_generation", monitored_code_gen)
    graph.add_node("rendering", monitored_rendering)
    
    # Chain them together
    graph.add_edge(START, "code_generation")
    graph.add_edge("code_generation", "rendering")
    graph.add_edge("rendering", END)
    
    return graph.compile(checkpointer=MemorySaver())


def create_full_agent_chain_graph() -> StateGraph:
    """Create graph for testing full agent chain without conditional routing."""
    graph = StateGraph(VideoGenerationState)
    
    # Add all monitored nodes
    monitored_planning = create_monitored_node("planning", planning_node)
    monitored_code_gen = create_monitored_node("code_generation", code_generation_node)
    monitored_rendering = create_monitored_node("rendering", rendering_node)
    
    graph.add_node("planning", monitored_planning)
    graph.add_node("code_generation", monitored_code_gen)
    graph.add_node("rendering", monitored_rendering)
    
    # Chain them together
    graph.add_edge(START, "planning")
    graph.add_edge("planning", "code_generation")
    graph.add_edge("code_generation", "rendering")
    graph.add_edge("rendering", END)
    
    return graph.compile(checkpointer=MemorySaver())


# Workflow Testing Graphs
def create_workflow_test_graph() -> StateGraph:
    """Create graph for comprehensive workflow testing."""
    # Use the Studio workflow configuration
    studio_config = StudioWorkflowConfig()
    workflow = studio_config.create_studio_compatible_workflow()
    
    return workflow.graph


def create_state_validation_graph() -> StateGraph:
    """Create graph specifically for state validation testing."""
    graph = StateGraph(VideoGenerationState)
    
    # Create state validation wrapper
    async def state_validation_wrapper(state: VideoGenerationState) -> VideoGenerationState:
        """Wrapper that validates state at each step."""
        session_id = state.session_id
        
        # Validate state consistency
        validation_results = {
            'session_id_consistent': state.session_id == session_id,
            'has_topic': bool(state.topic),
            'has_description': bool(state.description),
            'current_step_valid': state.current_step in ['planning', 'code_generation', 'rendering', 'complete', 'error_handler'],
            'workflow_complete_boolean': isinstance(state.workflow_complete, bool),
            'errors_list': isinstance(state.errors, list)
        }
        
        # Record validation results
        studio_monitor.record_performance_metric(
            "state_validation",
            "validation_results",
            validation_results
        )
        
        # Add validation trace
        state.add_execution_trace('state_validation', {
            'validation_results': validation_results,
            'all_valid': all(validation_results.values()),
            'timestamp': datetime.now().isoformat()
        })
        
        return state
    
    # Add validation nodes between each step
    monitored_planning = create_monitored_node("planning", planning_node)
    monitored_code_gen = create_monitored_node("code_generation", code_generation_node)
    monitored_rendering = create_monitored_node("rendering", rendering_node)
    
    graph.add_node("planning", monitored_planning)
    graph.add_node("planning_validation", state_validation_wrapper)
    graph.add_node("code_generation", monitored_code_gen)
    graph.add_node("code_validation", state_validation_wrapper)
    graph.add_node("rendering", monitored_rendering)
    graph.add_node("rendering_validation", state_validation_wrapper)
    
    # Chain with validation steps
    graph.add_edge(START, "planning")
    graph.add_edge("planning", "planning_validation")
    graph.add_edge("planning_validation", "code_generation")
    graph.add_edge("code_generation", "code_validation")
    graph.add_edge("code_validation", "rendering")
    graph.add_edge("rendering", "rendering_validation")
    graph.add_edge("rendering_validation", END)
    
    return graph.compile(checkpointer=MemorySaver())


def create_checkpoint_inspection_graph() -> StateGraph:
    """Create graph for checkpoint inspection testing."""
    graph = StateGraph(VideoGenerationState)
    
    # Create checkpoint capture wrapper
    async def checkpoint_capture_wrapper(node_name: str):
        async def wrapper(state: VideoGenerationState) -> VideoGenerationState:
            """Wrapper that captures checkpoint data."""
            session_id = state.session_id
            
            # Capture checkpoint data
            checkpoint_data = {
                'node': node_name,
                'session_id': session_id,
                'current_step': state.current_step,
                'workflow_complete': state.workflow_complete,
                'has_errors': state.has_errors(),
                'completion_percentage': state.get_completion_percentage() if hasattr(state, 'get_completion_percentage') else 0,
                'timestamp': datetime.now().isoformat()
            }
            
            # Record checkpoint
            studio_monitor.record_performance_metric(
                f"{node_name}_checkpoint",
                "checkpoint_captured",
                checkpoint_data
            )
            
            # Add checkpoint trace
            state.add_execution_trace(f'{node_name}_checkpoint', checkpoint_data)
            
            return state
        
        return wrapper
    
    # Add nodes with checkpoint capture
    monitored_planning = create_monitored_node("planning", planning_node)
    monitored_code_gen = create_monitored_node("code_generation", code_generation_node)
    monitored_rendering = create_monitored_node("rendering", rendering_node)
    
    graph.add_node("planning", monitored_planning)
    graph.add_node("planning_checkpoint", checkpoint_capture_wrapper("planning"))
    graph.add_node("code_generation", monitored_code_gen)
    graph.add_node("code_checkpoint", checkpoint_capture_wrapper("code_generation"))
    graph.add_node("rendering", monitored_rendering)
    graph.add_node("rendering_checkpoint", checkpoint_capture_wrapper("rendering"))
    
    # Chain with checkpoint capture
    graph.add_edge(START, "planning")
    graph.add_edge("planning", "planning_checkpoint")
    graph.add_edge("planning_checkpoint", "code_generation")
    graph.add_edge("code_generation", "code_checkpoint")
    graph.add_edge("code_checkpoint", "rendering")
    graph.add_edge("rendering", "rendering_checkpoint")
    graph.add_edge("rendering_checkpoint", END)
    
    return graph.compile(checkpointer=MemorySaver())


def create_transition_testing_graph() -> StateGraph:
    """Create graph for testing agent transitions."""
    graph = StateGraph(VideoGenerationState)
    
    # Import routing functions
    from ..routing import route_from_planning, route_from_code_generation, route_from_rendering
    
    # Create transition monitoring wrapper
    def create_transition_monitor(from_node: str, routing_function: callable):
        def monitor_transition(state: VideoGenerationState) -> str:
            """Monitor and log transitions."""
            # Get routing decision
            next_node = routing_function(state)
            
            # Record transition
            transition_data = {
                'from_node': from_node,
                'to_node': next_node,
                'session_id': state.session_id,
                'current_step': state.current_step,
                'has_errors': state.has_errors(),
                'timestamp': datetime.now().isoformat()
            }
            
            studio_monitor.record_performance_metric(
                "transition_monitoring",
                f"{from_node}_to_{next_node}",
                transition_data
            )
            
            # Add transition trace
            state.add_execution_trace('agent_transition', transition_data)
            
            return next_node
        
        return monitor_transition
    
    # Add monitored nodes
    monitored_planning = create_monitored_node("planning", planning_node)
    monitored_code_gen = create_monitored_node("code_generation", code_generation_node)
    monitored_rendering = create_monitored_node("rendering", rendering_node)
    monitored_error_handler = create_monitored_node("error_handler", error_handler_node)
    
    graph.add_node("planning", monitored_planning)
    graph.add_node("code_generation", monitored_code_gen)
    graph.add_node("rendering", monitored_rendering)
    graph.add_node("error_handler", monitored_error_handler)
    
    # Add conditional edges with monitoring
    graph.add_conditional_edges(
        "planning",
        create_transition_monitor("planning", route_from_planning),
        {
            "code_generation": "code_generation",
            "error_handler": "error_handler"
        }
    )
    
    graph.add_conditional_edges(
        "code_generation",
        create_transition_monitor("code_generation", route_from_code_generation),
        {
            "rendering": "rendering",
            "error_handler": "error_handler"
        }
    )
    
    graph.add_conditional_edges(
        "rendering",
        create_transition_monitor("rendering", route_from_rendering),
        {
            "error_handler": "error_handler"
        }
    )
    
    # Set entry point
    graph.add_edge(START, "planning")
    
    # Error handler and rendering can end
    graph.add_edge("error_handler", END)
    graph.add_edge("rendering", END)
    
    return graph.compile(checkpointer=MemorySaver())


# Graph instances for Studio
planning_agent_graph = create_planning_agent_graph()
code_generation_agent_graph = create_code_generation_agent_graph()
rendering_agent_graph = create_rendering_agent_graph()
error_handler_agent_graph = create_error_handler_agent_graph()

planning_to_code_chain_graph = create_planning_to_code_chain_graph()
code_to_rendering_chain_graph = create_code_to_rendering_chain_graph()
full_agent_chain_graph = create_full_agent_chain_graph()

workflow_test_graph = create_workflow_test_graph()
state_validation_graph = create_state_validation_graph()
checkpoint_inspection_graph = create_checkpoint_inspection_graph()
transition_testing_graph = create_transition_testing_graph()

# Monitored versions for enhanced testing
monitored_planning_graph = planning_agent_graph
monitored_code_generation_graph = code_generation_agent_graph
monitored_rendering_graph = rendering_agent_graph
monitored_error_handler_graph = error_handler_agent_graph


# Utility functions for Studio integration
def get_available_graphs() -> Dict[str, str]:
    """Get list of available graphs for Studio testing."""
    return {
        'planning_agent_test': 'Individual planning agent testing',
        'code_generation_agent_test': 'Individual code generation agent testing',
        'rendering_agent_test': 'Individual rendering agent testing',
        'error_handler_agent_test': 'Individual error handler agent testing',
        'planning_to_code_chain': 'Planning to code generation chain testing',
        'code_to_rendering_chain': 'Code generation to rendering chain testing',
        'full_agent_chain': 'Full agent chain testing (linear)',
        'workflow_test_graph': 'Complete workflow testing with conditional routing',
        'state_validation_graph': 'State validation and consistency testing',
        'checkpoint_inspection_graph': 'Checkpoint data capture and inspection',
        'transition_testing_graph': 'Agent transition monitoring and validation',
        'monitored_planning': 'Enhanced planning agent with monitoring',
        'monitored_code_generation': 'Enhanced code generation agent with monitoring',
        'monitored_rendering': 'Enhanced rendering agent with monitoring',
        'monitored_error_handler': 'Enhanced error handler agent with monitoring'
    }


def get_graph_info(graph_name: str) -> Dict[str, Any]:
    """Get information about a specific graph."""
    graph_descriptions = get_available_graphs()
    
    if graph_name not in graph_descriptions:
        return {'error': f'Unknown graph: {graph_name}'}
    
    return {
        'name': graph_name,
        'description': graph_descriptions[graph_name],
        'type': 'individual' if 'agent_test' in graph_name else 'chain' if 'chain' in graph_name else 'workflow',
        'monitoring_enabled': True,
        'checkpointing_enabled': True,
        'state_validation': 'state_validation' in graph_name,
        'checkpoint_inspection': 'checkpoint_inspection' in graph_name,
        'transition_monitoring': 'transition' in graph_name or 'monitored' in graph_name
    }


async def run_graph_test(graph_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run a test on a specific graph."""
    graphs = {
        'planning_agent_test': planning_agent_graph,
        'code_generation_agent_test': code_generation_agent_graph,
        'rendering_agent_test': rendering_agent_graph,
        'error_handler_agent_test': error_handler_agent_graph,
        'planning_to_code_chain': planning_to_code_chain_graph,
        'code_to_rendering_chain': code_to_rendering_chain_graph,
        'full_agent_chain': full_agent_chain_graph,
        'workflow_test_graph': workflow_test_graph,
        'state_validation_graph': state_validation_graph,
        'checkpoint_inspection_graph': checkpoint_inspection_graph,
        'transition_testing_graph': transition_testing_graph
    }
    
    if graph_name not in graphs:
        return {'error': f'Unknown graph: {graph_name}'}
    
    try:
        # Create initial state
        from ..models.config import WorkflowConfig
        config = WorkflowConfig()
        
        initial_state = VideoGenerationState(
            topic=input_data.get('topic', 'Test Topic'),
            description=input_data.get('description', 'Test Description'),
            session_id=input_data.get('session_id', f'test_{graph_name}_{int(datetime.now().timestamp())}'),
            config=config
        )
        
        # Add any additional input data to state
        for key, value in input_data.items():
            if key not in ['topic', 'description', 'session_id'] and hasattr(initial_state, key):
                setattr(initial_state, key, value)
        
        # Execute graph
        graph = graphs[graph_name]
        final_state = await graph.ainvoke(initial_state)
        
        return {
            'success': True,
            'graph_name': graph_name,
            'session_id': initial_state.session_id,
            'final_state': {
                'current_step': final_state.current_step,
                'workflow_complete': final_state.workflow_complete,
                'has_errors': final_state.has_errors(),
                'completion_percentage': final_state.get_completion_percentage() if hasattr(final_state, 'get_completion_percentage') else 0
            },
            'execution_trace_count': len(final_state.execution_trace) if hasattr(final_state, 'execution_trace') else 0
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__,
            'graph_name': graph_name
        }


if __name__ == "__main__":
    # Test graph creation
    print("Testing Studio graph creation...")
    
    try:
        graphs = get_available_graphs()
        print(f"✅ Created {len(graphs)} Studio graphs:")
        for name, description in graphs.items():
            print(f"   - {name}: {description}")
        
        # Test a simple graph
        test_input = {
            'topic': 'Test Topic',
            'description': 'Test Description',
            'session_id': 'studio_test'
        }
        
        import asyncio
        result = asyncio.run(run_graph_test('planning_agent_test', test_input))
        
        if result.get('success'):
            print(f"✅ Test execution successful")
            print(f"   - Session ID: {result['session_id']}")
            print(f"   - Final step: {result['final_state']['current_step']}")
        else:
            print(f"❌ Test execution failed: {result.get('error')}")
        
        print("\n🎉 Studio graphs setup completed successfully!")
        
    except Exception as e:
        print(f"❌ Studio graphs setup failed: {e}")
        raise
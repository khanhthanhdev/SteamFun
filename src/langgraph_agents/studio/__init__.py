"""
LangGraph Studio integration package.

This package provides comprehensive Studio integration for testing individual agents
and monitoring their execution in the LangGraph Studio environment.
"""

from .studio_integration import (
    StudioAgentRegistry,
    StudioAgentTester,
    StudioWorkflowBuilder,
    StudioMonitor,
    studio_registry,
    studio_monitor,
    create_studio_tester,
    create_studio_workflow_builder
)

from .studio_graphs import (
    planning_agent_graph,
    code_generation_agent_graph,
    rendering_agent_graph,
    error_handler_agent_graph,
    planning_to_code_chain_graph,
    code_to_rendering_chain_graph,
    full_agent_chain_graph,
    monitored_planning_graph,
    monitored_code_generation_graph,
    monitored_rendering_graph,
    monitored_error_handler_graph,
    get_available_graphs,
    get_graph_metadata
)

from .test_scenarios import (
    TestScenarioManager,
    StudioTestDataGenerator,
    test_scenario_manager,
    get_test_scenario_manager
)

from .studio_config import (
    StudioConfig,
    studio_config,
    get_studio_config,
    is_studio_environment,
    create_studio_langgraph_config,
    save_studio_langgraph_config,
    setup_studio_logging
)

__all__ = [
    # Core integration classes
    "StudioAgentRegistry",
    "StudioAgentTester", 
    "StudioWorkflowBuilder",
    "StudioMonitor",
    
    # Global instances
    "studio_registry",
    "studio_monitor",
    "studio_config",
    "test_scenario_manager",
    
    # Factory functions
    "create_studio_tester",
    "create_studio_workflow_builder",
    "get_test_scenario_manager",
    "get_studio_config",
    
    # Pre-compiled graphs
    "planning_agent_graph",
    "code_generation_agent_graph", 
    "rendering_agent_graph",
    "error_handler_agent_graph",
    "planning_to_code_chain_graph",
    "code_to_rendering_chain_graph",
    "full_agent_chain_graph",
    "monitored_planning_graph",
    "monitored_code_generation_graph",
    "monitored_rendering_graph",
    "monitored_error_handler_graph",
    
    # Utility functions
    "get_available_graphs",
    "get_graph_metadata",
    "is_studio_environment",
    "create_studio_langgraph_config",
    "save_studio_langgraph_config",
    "setup_studio_logging",
    
    # Test management
    "TestScenarioManager",
    "StudioTestDataGenerator",
    "StudioConfig"
]
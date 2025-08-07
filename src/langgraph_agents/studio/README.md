# LangGraph Studio Integration for Agent Testing

This directory contains the LangGraph Studio integration for testing individual agents and monitoring their execution in the Studio environment.

## Overview

The Studio integration provides:

1. **Agent Registration and Discovery** - Automatic registration of agents with their schemas and metadata
2. **Individual Agent Testing** - Test each agent in isolation with predefined scenarios
3. **Agent Chain Testing** - Test sequences of agents working together
4. **State Visualization and Monitoring** - Real-time monitoring of agent execution and state changes
5. **Test Data Management** - Comprehensive test scenarios and data fixtures

## Components

### Core Integration (`studio_integration.py`)

- **StudioAgentRegistry**: Manages agent registration and discovery
- **StudioAgentTester**: Executes individual agent tests with input validation
- **StudioWorkflowBuilder**: Creates Studio-compatible workflow graphs
- **StudioMonitor**: Tracks agent execution and performance metrics

### Graph Definitions (`studio_graphs.py`)

Pre-configured graphs for Studio testing:

- **Individual Agent Graphs**: Test single agents in isolation
  - `planning_agent_graph`
  - `code_generation_agent_graph`
  - `rendering_agent_graph`
  - `error_handler_agent_graph`

- **Agent Chain Graphs**: Test agent sequences
  - `planning_to_code_chain_graph`
  - `code_to_rendering_chain_graph`
  - `full_agent_chain_graph`

- **Monitored Graphs**: Enhanced versions with performance monitoring
  - `monitored_planning_graph`
  - `monitored_code_generation_graph`
  - `monitored_rendering_graph`
  - `monitored_error_handler_graph`

### Test Scenarios (`test_scenarios.py`)

- **TestScenarioManager**: Manages test scenarios and validation rules
- **StudioTestDataGenerator**: Generates test data for different complexity levels
- Pre-defined scenarios for each agent type with expected outputs

### Configuration (`studio_config.py`)

- **StudioConfig**: Environment detection and configuration management
- Optimized settings for Studio testing (faster models, preview mode, etc.)
- Automatic directory setup and logging configuration

### Main Workflow (`studio_workflow.py`)

- **StudioVideoGenerationWorkflow**: Studio-optimized version of the main workflow
- Enhanced monitoring and error tracking
- Streaming support for real-time updates

## Usage in LangGraph Studio

### 1. Individual Agent Testing

Test a single agent with predefined scenarios:

```python
# Available graphs in Studio:
- planning_agent_test
- code_generation_agent_test  
- rendering_agent_test
- error_handler_agent_test
```

**Example Input for Planning Agent:**
```json
{
  "topic": "Linear Equations",
  "description": "Create a video explaining how to solve linear equations step by step",
  "session_id": "test_session_001"
}
```

**Expected Output:**
```json
{
  "scene_outline": "Scene 1: Introduction\nScene 2: Process\nScene 3: Examples",
  "scene_implementations": {
    "1": "Introduce linear equations with definition",
    "2": "Show step-by-step solving process", 
    "3": "Present concrete examples"
  },
  "detected_plugins": ["manim", "numpy"]
}
```

### 2. Agent Chain Testing

Test sequences of agents working together:

```python
# Available chain graphs:
- planning_to_code_chain
- code_to_rendering_chain
- full_agent_chain
```

**Example Input for Planning to Code Chain:**
```json
{
  "topic": "Pythagorean Theorem",
  "description": "Explain the Pythagorean theorem with geometric proofs",
  "session_id": "chain_test_001"
}
```

### 3. Monitored Testing

Use monitored versions for enhanced observability:

```python
# Monitored graphs with performance tracking:
- monitored_planning
- monitored_code_generation
- monitored_rendering
- monitored_error_handler
```

### 4. Full Workflow Testing

Test the complete workflow with Studio optimizations:

```python
# Main Studio workflow:
- studio_workflow
```

## Test Scenarios

### Planning Agent Scenarios

1. **basic_math_topic**: Simple mathematics topic
2. **complex_physics_topic**: Advanced physics concepts
3. **simple_concept**: Very basic educational content

### Code Generation Agent Scenarios

1. **basic_scene_implementation**: Standard scene code generation
2. **mathematical_content**: Complex mathematical formulas and expressions

### Rendering Agent Scenarios

1. **simple_code_rendering**: Basic Manim code rendering
2. **multiple_scenes**: Multi-scene video rendering and combination

### Error Handler Agent Scenarios

1. **recoverable_error**: Test error recovery mechanisms
2. **critical_error**: Test error escalation to human intervention

## Configuration

The Studio integration automatically detects the Studio environment and applies optimized settings:

- **Faster Models**: Uses Claude-3-Haiku for quicker testing
- **Preview Mode**: Enables low-quality, fast rendering
- **Reduced Timeouts**: Shorter timeouts for responsive testing
- **Enhanced Logging**: Detailed logs for debugging
- **Memory Checkpointing**: Uses in-memory checkpointing for testing

## Environment Variables

Optional environment variables for customization:

```bash
LANGGRAPH_STUDIO=true              # Force Studio mode
STUDIO_OUTPUT_DIR=studio_output    # Custom output directory
STUDIO_TEST_DATA_DIR=test_data     # Custom test data directory
LANGSMITH_TRACING=true             # Enable LangSmith tracing
```

## File Structure

```
src/langgraph_agents/studio/
├── __init__.py                 # Package initialization and exports
├── README.md                   # This documentation
├── studio_integration.py       # Core integration classes
├── studio_graphs.py           # Pre-configured graphs for Studio
├── test_scenarios.py          # Test scenarios and data management
├── studio_config.py           # Configuration and environment setup
└── studio_workflow.py         # Main workflow entry point
```

## Getting Started

1. **Start LangGraph Studio** with the project directory
2. **Select a graph** from the available Studio graphs
3. **Provide input** according to the agent's schema requirements
4. **Execute and monitor** the agent's performance in real-time
5. **Review results** including outputs, state changes, and performance metrics

## Monitoring and Debugging

The Studio integration provides comprehensive monitoring:

- **Execution Traces**: Detailed step-by-step execution logs
- **Performance Metrics**: Timing, memory usage, and success rates
- **State Visualization**: Real-time state changes and data flow
- **Error Tracking**: Detailed error information and recovery attempts
- **Test Results**: Validation against expected outputs

## Best Practices

1. **Start Simple**: Begin with individual agent tests before testing chains
2. **Use Scenarios**: Leverage pre-defined scenarios for consistent testing
3. **Monitor Performance**: Use monitored graphs to track agent performance
4. **Validate Outputs**: Check test results against expected outputs
5. **Iterate Quickly**: Use preview mode for fast iteration during development

## Troubleshooting

### Common Issues

1. **Agent Not Found**: Ensure the agent is registered in `StudioAgentRegistry`
2. **Invalid Input**: Check input schema requirements for each agent
3. **Timeout Errors**: Increase timeout settings in Studio configuration
4. **Missing Dependencies**: Ensure all required packages are installed
5. **Path Issues**: Verify output and test data directories exist

### Debug Mode

Enable verbose logging for detailed debugging:

```python
import logging
logging.getLogger("langgraph_agents.studio").setLevel(logging.DEBUG)
```

## Contributing

When adding new agents or test scenarios:

1. Register the agent in `StudioAgentRegistry`
2. Create appropriate test scenarios in `TestScenarioManager`
3. Add graph definitions in `studio_graphs.py`
4. Update the LangGraph configuration in `langgraph.json`
5. Document the new functionality in this README
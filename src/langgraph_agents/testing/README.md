# LangGraph Agents Testing Infrastructure

This module provides comprehensive testing capabilities for individual LangGraph agents with support for LangGraph Studio integration.

## Overview

The testing infrastructure enables developers to:

- Test individual agents in isolation
- Capture and validate agent outputs
- Manage test data and scenarios
- Integrate with LangGraph Studio for visualization
- Run comprehensive test suites

## Components

### Agent Test Runners

Individual test runners for each agent type:

- **PlannerAgentTestRunner**: Tests planning functionality with topic/description inputs
- **CodeGeneratorAgentTestRunner**: Tests code generation with scene implementation inputs  
- **RendererAgentTestRunner**: Tests rendering with code execution inputs
- **ErrorHandlerAgentTestRunner**: Tests error handling with error scenario inputs
- **HumanLoopAgentTestRunner**: Tests human intervention with intervention scenario inputs

### Output Capture System

Comprehensive output capture and validation:

- **AgentOutputCapture**: Captures stdout, stderr, logs, metrics, and results
- **OutputFormatter**: Formats output for Studio visualization, JSON, or text
- **OutputValidator**: Validates output against expected patterns and criteria

### Test Data Management

Test data fixtures and scenario management:

- **TestDataManager**: Provides test data fixtures for different agent types
- **TestScenarioManager**: Manages test scenarios and execution tracking
- **StaticTestDataProvider**: Provides predefined test data sets

### Studio Integration

Main interface for LangGraph Studio:

- **StudioTestRunner**: Unified interface for running all agent tests
- Convenience functions for individual agent testing
- Comprehensive test suite execution

## Usage

### Basic Agent Testing

```python
from langgraph_agents.testing import PlannerAgentTestRunner
from langgraph_agents.models.config import WorkflowConfig

# Initialize test runner
config = WorkflowConfig()
runner = PlannerAgentTestRunner(config)

# Run test
inputs = {
    'topic': 'Linear Algebra Basics',
    'description': 'Educational video about linear algebra concepts',
    'session_id': 'test_001'
}

result = await runner.run_test(inputs)
print(f"Success: {result['success']}")
```

### Studio Integration

```python
from langgraph_agents.testing import get_studio_test_runner

# Get Studio test runner
studio_runner = get_studio_test_runner()

# Run agent test with output capture
result = await studio_runner.run_agent_test('PlannerAgent', inputs)

# Access Studio-formatted output
studio_output = result['studio_output']
validation_results = result['validation']
```

### Scenario Management

```python
from langgraph_agents.testing import get_scenario_manager
from langgraph_agents.testing.test_data_manager import TestComplexity

# Get scenario manager
scenario_manager = get_scenario_manager()

# Generate test scenarios
scenario_ids = scenario_manager.generate_scenarios_for_agent(
    'PlannerAgent', 
    TestComplexity.MEDIUM, 
    count=3
)

# Run scenario
studio_runner = get_studio_test_runner()
result = await studio_runner.run_scenario(scenario_ids[0])
```

### Comprehensive Test Suite

```python
# Run comprehensive test suite
suite_result = await studio_runner.run_comprehensive_test_suite(
    complexity=TestComplexity.SIMPLE,
    agents_to_test=['PlannerAgent', 'CodeGeneratorAgent']
)

print(f"Success rate: {suite_result['summary']['success_rate']:.2%}")
```

## Test Data Types

### PlannerAgent Test Data
- **topic**: Video topic string
- **description**: Video description string  
- **session_id**: Unique session identifier

### CodeGeneratorAgent Test Data
- **topic**: Video topic string
- **description**: Video description string
- **scene_outline**: Generated scene outline
- **scene_implementations**: Dictionary of scene implementations by number
- **session_id**: Unique session identifier

### RendererAgent Test Data
- **generated_code**: Dictionary of generated code by scene number
- **file_prefix**: File prefix for output files
- **quality**: Rendering quality ('low', 'medium', 'high', 'ultra')
- **session_id**: Unique session identifier

### ErrorHandlerAgent Test Data
- **error_scenarios**: List of error scenario dictionaries
- **session_id**: Unique session identifier

### HumanLoopAgent Test Data
- **intervention_scenarios**: List of intervention scenario dictionaries
- **session_id**: Unique session identifier

## Output Validation

The system validates outputs based on agent-specific criteria:

### Common Validations
- Execution time within reasonable limits
- No critical errors
- Low error rate in logs

### Agent-Specific Validations
- **PlannerAgent**: Valid scene outline and implementations
- **CodeGeneratorAgent**: High code generation success rate
- **RendererAgent**: Successful video rendering
- **ErrorHandlerAgent**: Good error recovery rate
- **HumanLoopAgent**: Successful intervention handling

## Studio Visualization

Output is formatted for LangGraph Studio with:

- **Session Info**: Execution time, timestamps, agent type
- **Console Output**: Captured stdout/stderr with summaries
- **Logs**: Structured log entries grouped by level
- **Metrics**: Performance and execution metrics
- **Errors**: Error summaries and details
- **Results**: Test results and outputs

## Test Complexity Levels

- **SIMPLE**: Basic functionality tests
- **MEDIUM**: Moderate complexity with multiple scenarios
- **COMPLEX**: Advanced tests with edge cases
- **EDGE_CASE**: Boundary conditions and error cases

## Error Handling

The testing infrastructure handles:

- Test execution failures
- Output capture errors
- Validation failures
- Studio integration issues

All errors are captured and included in test results for debugging.

## Example Usage

See `examples/agent_testing_usage.py` for comprehensive usage examples including:

- Individual agent testing
- Studio integration
- Scenario management
- Comprehensive test suites

## Configuration

The testing infrastructure uses the standard `WorkflowConfig` for agent configuration. Test-specific settings can be customized through the test runners and managers.

## Integration with LangGraph Studio

The testing infrastructure is designed to work seamlessly with LangGraph Studio:

1. **Node Testing**: Each agent can be tested as an individual node
2. **Output Visualization**: Rich output formatting for Studio display
3. **Scenario Execution**: Predefined scenarios can be run from Studio
4. **Performance Monitoring**: Execution metrics and validation results
5. **Error Analysis**: Detailed error capture and analysis

This enables comprehensive testing and debugging of LangGraph agents directly within the Studio environment.
# Integration Tests for Multi-Agent Workflows

This directory contains comprehensive integration tests for the LangGraph multi-agent system. The tests verify agent-to-agent communication, state management, error propagation, and tool integration across the entire workflow.

## Test Structure

### Core Integration Tests

1. **test_agent_communication.py** - Tests agent-to-agent communication patterns
   - Planner to CodeGenerator communication
   - Error propagation between agents
   - RAG agent integration
   - State consistency across agents
   - Human-in-the-loop integration
   - Monitoring agent integration

2. **test_state_management.py** - Tests state management validation
   - State immutability during agent execution
   - State validation across workflow stages
   - State persistence across agent transitions
   - Error state management
   - Concurrent state access
   - State rollback on failure
   - State schema evolution

3. **test_error_propagation.py** - Tests error handling and recovery
   - Single agent error recovery
   - Cascading error propagation
   - Error recovery strategies
   - Human escalation workflow
   - Error context preservation

4. **test_tool_integration.py** - Tests external tool integration
   - MCP server connectivity
   - Context7 integration
   - Docling document processing
   - Human intervention tools
   - Tool error handling
   - Tool coordination between agents
   - Tool performance monitoring

5. **test_comprehensive_integration.py** - Complete workflow integration tests
   - End-to-end workflow integration
   - Error recovery workflow integration
   - Human-in-the-loop workflow integration
   - Tool integration across agents
   - Monitoring integration across workflow
   - State consistency across complex workflows

## Test Categories

### Agent Communication Tests
- **Purpose**: Verify agents can communicate effectively through state updates
- **Coverage**: Data flow, command routing, agent coordination
- **Key Scenarios**: Planning → Code Generation → Rendering workflow

### State Management Tests
- **Purpose**: Ensure state consistency and integrity across agent transitions
- **Coverage**: State immutability, validation, persistence, rollback
- **Key Scenarios**: Complex state transformations, concurrent access, schema evolution

### Error Propagation Tests
- **Purpose**: Validate error handling and recovery mechanisms
- **Coverage**: Error classification, recovery strategies, escalation
- **Key Scenarios**: Cascading failures, retry mechanisms, human intervention

### Tool Integration Tests
- **Purpose**: Test external tool connectivity and coordination
- **Coverage**: MCP servers, Context7, Docling, human tools
- **Key Scenarios**: Multi-tool workflows, error handling, performance monitoring

## Running the Tests

### Prerequisites
```bash
pip install pytest pytest-asyncio
```

### Run All Integration Tests
```bash
python -m pytest tests/integration/ -v
```

### Run Specific Test Categories
```bash
# Agent communication tests
python -m pytest tests/integration/test_agent_communication.py -v

# State management tests
python -m pytest tests/integration/test_state_management.py -v

# Error propagation tests
python -m pytest tests/integration/test_error_propagation.py -v

# Tool integration tests
python -m pytest tests/integration/test_tool_integration.py -v

# Comprehensive integration tests
python -m pytest tests/integration/test_comprehensive_integration.py -v
```

### Run with Specific Markers
```bash
# Run only integration tests
python -m pytest -m integration -v

# Run only async tests
python -m pytest -m asyncio -v
```

### Simple Test Runner
For basic validation without pytest dependencies:
```bash
python test_integration_runner.py
```

## Test Implementation Details

### Mock Strategy
- **Agent Mocking**: Mock agents implement the BaseAgent interface with controlled behavior
- **State Mocking**: Use deepcopy to ensure state immutability testing
- **Tool Mocking**: Mock external tools (MCP servers, Context7, Docling) with realistic responses
- **Service Mocking**: Mock LangFuse, monitoring services, and human interfaces

### Test Data
- **Realistic State**: Tests use realistic VideoGenerationState with all required fields
- **Configuration**: Comprehensive SystemConfig with all agent types and tools
- **Error Scenarios**: Various error types (timeout, connection, validation, etc.)
- **Performance Data**: Simulated execution times and resource usage

### Assertions
- **State Consistency**: Verify state fields are preserved and updated correctly
- **Command Structure**: Validate LangGraph Command objects have correct goto and update fields
- **Error Handling**: Ensure errors are properly escalated and handled
- **Tool Integration**: Verify external tools are called with correct parameters
- **Performance**: Check execution times and resource usage are tracked

## Coverage Areas

### Requirements Coverage
- **1.2**: Agent coordination and communication ✓
- **1.3**: State management across workflow ✓
- **4.4**: Error handling and recovery ✓

### Agent Types Tested
- PlannerAgent ✓
- CodeGeneratorAgent ✓
- RendererAgent ✓
- RAGAgent ✓
- ErrorHandlerAgent ✓
- HumanLoopAgent ✓
- MonitoringAgent ✓

### Integration Patterns
- Sequential workflow execution ✓
- Parallel agent coordination ✓
- Error recovery workflows ✓
- Human intervention workflows ✓
- Tool coordination workflows ✓
- Monitoring and observability ✓

## Test Maintenance

### Adding New Tests
1. Follow the existing test structure and naming conventions
2. Use appropriate fixtures for system configuration and initial state
3. Mock external dependencies appropriately
4. Include both success and failure scenarios
5. Add appropriate markers (@pytest.mark.asyncio, @pytest.mark.integration)

### Updating Tests
1. Update tests when agent interfaces change
2. Maintain backward compatibility where possible
3. Update mock responses when external tool APIs change
4. Keep test data realistic and representative

### Performance Considerations
- Tests use minimal delays (0.01s) for timing simulation
- Mock responses are lightweight to avoid test slowdown
- Concurrent tests are limited to avoid resource contention
- Test data sizes are kept reasonable for fast execution

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed and paths are correct
2. **Async Test Issues**: Verify pytest-asyncio is installed and configured
3. **Mock Failures**: Check that mocked services match expected interfaces
4. **State Errors**: Ensure all required state fields are included in test data

### Debug Tips
1. Use `-v` flag for verbose output
2. Use `--tb=long` for detailed tracebacks
3. Add print statements in test methods for debugging
4. Use `pytest.set_trace()` for interactive debugging
5. Run individual tests to isolate issues

## Future Enhancements

### Planned Additions
- Performance benchmarking tests
- Load testing for concurrent workflows
- Integration with actual MCP servers (optional)
- Visual workflow validation tests
- Stress testing for error scenarios

### Test Infrastructure
- Automated test data generation
- Test result reporting and analytics
- Continuous integration setup
- Test coverage reporting
- Performance regression detection
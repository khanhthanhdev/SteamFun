# End-to-End Tests for LangGraph Multi-Agent Video Generation System

This directory contains comprehensive end-to-end tests for the LangGraph multi-agent video generation system. These tests validate the complete workflow from planning to final video output, including performance benchmarking, human-in-the-loop scenarios, and failure recovery mechanisms.

## Test Structure

### Core Test Files

1. **test_complete_workflow.py** - Complete video generation workflow tests
   - Full pipeline execution from planning to rendering
   - Scene outline generation only
   - Specific scene processing
   - Streaming workflow execution
   - Checkpoint management
   - Workflow interruption and resumption

2. **test_performance_benchmarking.py** - Performance and scalability tests
   - Single workflow performance across different scenarios
   - Concurrent workflow execution
   - Memory usage patterns and optimization
   - Scalability limits and breaking points
   - Performance regression detection

3. **test_human_loop_scenarios.py** - Human-in-the-loop interaction tests
   - Planning approval scenarios
   - Code quality review workflows
   - Visual error escalation
   - Error recovery decision making
   - Human intervention timeout handling
   - Multiple intervention workflows
   - Streaming updates during human interaction

4. **test_failure_recovery.py** - Failure scenarios and recovery mechanisms
   - Agent timeout recovery
   - Model API failure recovery with backoff
   - Code generation error recovery using RAG
   - Rendering failure recovery with quality adjustment
   - Visual analysis failure recovery
   - RAG service failure recovery
   - Cascading failure escalation
   - Persistent failure handling
   - Workflow-level retry mechanisms
   - Recovery performance impact analysis

5. **test_runner.py** - Comprehensive test runner
   - Orchestrates all end-to-end test categories
   - Provides detailed reporting and summaries
   - Handles test environment setup and cleanup
   - Supports parallel test execution

## Test Categories

### Complete Workflow Tests
- **Purpose**: Validate end-to-end video generation workflows
- **Coverage**: Planning → Code Generation → Rendering → Visual Analysis
- **Key Scenarios**: 
  - Full video generation pipeline
  - Planning-only mode
  - Specific scene processing
  - Streaming execution with progress updates
  - Checkpoint creation and management
  - Workflow interruption and resumption

### Performance Benchmarking Tests
- **Purpose**: Measure system performance and identify bottlenecks
- **Coverage**: Execution time, memory usage, throughput, scalability
- **Key Scenarios**:
  - Single workflow performance across video complexities
  - Concurrent workflow execution
  - Memory usage patterns and leak detection
  - Scalability limits under increasing load
  - Performance regression detection

### Human-in-the-Loop Tests
- **Purpose**: Validate human intervention capabilities
- **Coverage**: Decision points, approval workflows, timeout handling
- **Key Scenarios**:
  - Planning approval and modification
  - Code quality review and fixes
  - Visual error escalation and resolution
  - Error recovery strategy selection
  - Timeout handling with default actions
  - Multiple intervention points in single workflow
  - Real-time streaming updates during human interaction

### Failure Recovery Tests
- **Purpose**: Test system resilience and error handling
- **Coverage**: Error detection, recovery strategies, escalation
- **Key Scenarios**:
  - Individual agent failures and recovery
  - API failures with exponential backoff
  - Code generation errors with RAG-assisted recovery
  - Rendering failures with quality fallback
  - Service unavailability handling
  - Cascading failure escalation to human intervention
  - Persistent failure handling
  - Performance impact of recovery mechanisms

## Running the Tests

### Prerequisites
```bash
pip install pytest pytest-asyncio psutil
```

### Run All End-to-End Tests
```bash
python tests/e2e/test_runner.py
```

### Run Specific Test Categories
```bash
# Complete workflow tests
python -m pytest tests/e2e/test_complete_workflow.py -v

# Performance benchmarking tests
python -m pytest tests/e2e/test_performance_benchmarking.py -v

# Human-in-the-loop tests
python -m pytest tests/e2e/test_human_loop_scenarios.py -v

# Failure recovery tests
python -m pytest tests/e2e/test_failure_recovery.py -v
```

### Run with Specific Markers
```bash
# Run only end-to-end tests
python -m pytest -m e2e -v

# Run only slow tests (performance benchmarking)
python -m pytest -m slow -v

# Run specific test function
python -m pytest tests/e2e/test_complete_workflow.py::TestCompleteWorkflow::test_complete_video_generation_workflow -v
```

## Test Implementation Details

### Mock Strategy
- **Workflow Mocking**: Mock LangGraph workflow execution with realistic state transitions
- **Agent Mocking**: Mock individual agents with controlled behavior and timing
- **Service Mocking**: Mock external services (LangFuse, RAG, MCP servers) with realistic responses
- **State Mocking**: Use realistic VideoGenerationState with all required fields
- **Performance Mocking**: Simulate realistic execution times and resource usage

### Test Data
- **Realistic Scenarios**: Tests use realistic video generation scenarios with varying complexity
- **Configuration**: Comprehensive SystemConfig with all agent types and tools
- **Error Scenarios**: Various error types (timeout, connection, validation, API failures)
- **Performance Data**: Simulated execution times, memory usage, and throughput metrics
- **Human Interaction**: Realistic human intervention scenarios with different priorities

### Assertions and Validation
- **Workflow Completion**: Verify workflows complete successfully with expected outputs
- **State Consistency**: Ensure state fields are preserved and updated correctly
- **Performance Metrics**: Validate execution times, memory usage, and throughput
- **Error Handling**: Verify errors are properly detected, handled, and recovered
- **Human Interaction**: Ensure human interventions are processed correctly
- **Recovery Mechanisms**: Validate failure recovery strategies work as expected

## Coverage Areas

### Requirements Coverage
- **3.1**: External API compatibility ✓
- **3.2**: Response format consistency ✓
- **4.4**: Error handling and recovery ✓
- **6.1**: Performance monitoring and optimization ✓

### Workflow Patterns Tested
- Sequential workflow execution ✓
- Parallel scene processing ✓
- Error recovery workflows ✓
- Human intervention workflows ✓
- Streaming execution ✓
- Checkpoint management ✓
- Workflow interruption and resumption ✓

### Agent Types Tested
- PlannerAgent ✓
- CodeGeneratorAgent ✓
- RendererAgent ✓
- VisualAnalysisAgent ✓
- RAGAgent ✓
- ErrorHandlerAgent ✓
- HumanLoopAgent ✓
- MonitoringAgent ✓

## Performance Benchmarks

### Baseline Expectations
- **Execution Time**: < 10 seconds for medium complexity videos
- **Memory Usage**: < 300MB peak memory usage
- **Throughput**: > 0.2 workflows/second
- **Error Rate**: < 5% under normal conditions
- **Recovery Time**: < 30 seconds for most failure scenarios

### Scalability Targets
- **Concurrent Load**: Support at least 4 concurrent workflows
- **Memory Scaling**: Linear memory growth with concurrent load
- **Throughput Scaling**: Maintain > 50% peak throughput under load
- **Error Resilience**: < 10% error rate increase under high load

## Test Maintenance

### Adding New Tests
1. Follow existing test structure and naming conventions
2. Use appropriate fixtures for system configuration and state
3. Mock external dependencies appropriately
4. Include both success and failure scenarios
5. Add appropriate markers (@pytest.mark.asyncio, @pytest.mark.e2e)
6. Update test_runner.py to include new test categories

### Updating Tests
1. Update tests when agent interfaces change
2. Maintain backward compatibility where possible
3. Update mock responses when external APIs change
4. Keep test data realistic and representative
5. Update performance baselines based on actual measurements

### Performance Considerations
- Tests use minimal delays for timing simulation
- Mock responses are lightweight to avoid test slowdown
- Concurrent tests are limited to avoid resource contention
- Test data sizes are kept reasonable for fast execution
- Memory usage is monitored and cleaned up properly

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed and paths are correct
2. **Async Test Issues**: Verify pytest-asyncio is installed and configured
3. **Mock Failures**: Check that mocked services match expected interfaces
4. **State Errors**: Ensure all required state fields are included in test data
5. **Performance Issues**: Check system resources and reduce concurrent test load

### Debug Tips
1. Use `-v` flag for verbose output
2. Use `--tb=long` for detailed tracebacks
3. Add print statements in test methods for debugging
4. Use `pytest.set_trace()` for interactive debugging
5. Run individual tests to isolate issues
6. Check temporary directories for test artifacts
7. Monitor system resources during test execution

## Future Enhancements

### Planned Additions
- Integration with actual MCP servers (optional)
- Visual workflow validation tests
- Load testing for production scenarios
- Stress testing for extreme failure conditions
- Integration with CI/CD pipelines
- Test result analytics and trending

### Test Infrastructure Improvements
- Automated test data generation
- Test result reporting and analytics
- Continuous integration setup
- Test coverage reporting
- Performance regression detection
- Automated baseline updates

## Contributing

When contributing to end-to-end tests:

1. **Test Design**: Design tests to be realistic and representative of actual usage
2. **Performance**: Keep tests efficient while maintaining realistic scenarios
3. **Reliability**: Ensure tests are deterministic and don't have flaky behavior
4. **Documentation**: Document test scenarios and expected outcomes
5. **Maintenance**: Write tests that are easy to maintain and update
6. **Coverage**: Ensure new features have corresponding end-to-end test coverage

## Test Results Interpretation

### Success Criteria
- All workflow tests pass with expected state transitions
- Performance tests meet baseline expectations
- Human-loop tests handle all intervention scenarios correctly
- Failure recovery tests demonstrate proper error handling and recovery
- No memory leaks or resource issues detected

### Failure Analysis
- Check test logs for specific failure points
- Verify mock configurations match expected interfaces
- Ensure test data is realistic and complete
- Check for timing issues in async operations
- Validate state transitions and field updates
- Monitor resource usage for performance issues

The end-to-end test suite provides comprehensive validation of the LangGraph multi-agent video generation system, ensuring reliability, performance, and user experience across all supported scenarios.
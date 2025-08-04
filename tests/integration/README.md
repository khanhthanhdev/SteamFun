# Integration Tests

This directory contains comprehensive integration tests for the FastAPI application. The tests verify API endpoints, service integrations, database operations, and external service connectivity across the entire system.

## Test Structure

### API Integration Tests (`test_api/`)

1. **test_video_api_integration.py** - Tests video API endpoints
   - Complete video creation workflow through API
   - Video status checking and download
   - API error handling and validation
   - Concurrent request handling

2. **test_rag_api_integration.py** - Tests RAG API endpoints
   - RAG query workflow through API
   - Document management operations
   - API error handling and validation
   - Query result processing

3. **test_agents_api_integration.py** - Tests agent API endpoints
   - Agent execution workflow through API
   - Agent status monitoring
   - Execution history retrieval
   - Chained agent execution

4. **test_aws_api_integration.py** - Tests AWS API endpoints
   - S3 upload/download workflows through API
   - DynamoDB metadata operations
   - Multipart upload handling
   - AWS service error handling

### Service Integration Tests (`test_services/`)

1. **test_video_service_integration.py** - Tests VideoService integrations
   - Database integration for video operations
   - File system integration for video processing
   - External API integration (TTS, Animation)
   - AWS integration for video storage
   - Error recovery and monitoring

2. **test_rag_service_integration.py** - Tests RAGService integrations
   - Vector store integration
   - Embedding model integration
   - Document processing integration
   - Caching system integration
   - Concurrent query handling

3. **test_agent_service_integration.py** - Tests AgentService integrations
   - LangGraph workflow integration
   - State management integration
   - Database integration for agent operations
   - Monitoring and error recovery
   - Human-in-the-loop workflows

### Database Integration Tests (`test_database/`)

1. **test_video_database_integration.py** - Tests video database operations
   - Video model creation and persistence
   - Video metadata relationships
   - Status updates and queries
   - Performance with bulk operations

2. **test_agent_database_integration.py** - Tests agent database operations
   - Agent execution record persistence
   - Agent state relationship management
   - Error tracking and persistence
   - State versioning and history

3. **test_rag_database_integration.py** - Tests RAG database operations
   - RAG query and document persistence
   - Embedding relationship management
   - Document search operations
   - Performance with bulk operations

### External Service Integration Tests (`test_external/`)

1. **test_aws_integration.py** - Tests AWS service integrations
   - S3 upload/download with mocked AWS
   - DynamoDB operations with mocked AWS
   - AWS service error handling

2. **test_langgraph_integration.py** - Tests LangGraph integrations
   - Workflow execution integration
   - Streaming workflow integration
   - State management integration
   - Agent coordination and monitoring

3. **test_embedding_service_integration.py** - Tests embedding service integrations
   - OpenAI embedding service integration
   - Sentence Transformers integration
   - Embedding model selection and caching
   - Performance monitoring and validation

### Agent Communication Tests (`test_agents/`)

1. **test_agent_communication.py** - Tests agent-to-agent communication patterns
   - Planner to CodeGenerator communication
   - Error propagation between agents
   - RAG agent integration
   - State consistency across agents

2. **test_comprehensive_integration.py** - Complete workflow integration tests
   - End-to-end workflow integration
   - Error recovery workflow integration
   - Agent coordination workflows
   - State consistency across complex workflows

## Test Categories

### API Integration Tests
- **Purpose**: Verify API endpoints work correctly with underlying services
- **Coverage**: Request/response handling, validation, error handling, authentication
- **Key Scenarios**: Complete workflows through API endpoints, concurrent requests

### Service Integration Tests
- **Purpose**: Test service layer integration with external dependencies
- **Coverage**: Database operations, external APIs, file systems, caching
- **Key Scenarios**: Service orchestration, error recovery, performance monitoring

### Database Integration Tests
- **Purpose**: Ensure database models and relationships work correctly
- **Coverage**: CRUD operations, relationships, transactions, performance
- **Key Scenarios**: Complex queries, bulk operations, data consistency

### External Service Integration Tests
- **Purpose**: Test integration with external services and APIs
- **Coverage**: AWS services, LangGraph workflows, embedding services
- **Key Scenarios**: Service connectivity, error handling, fallback mechanisms

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
# API integration tests
python -m pytest tests/integration/test_api/ -v

# Service integration tests
python -m pytest tests/integration/test_services/ -v

# Database integration tests
python -m pytest tests/integration/test_database/ -v

# External service integration tests
python -m pytest tests/integration/test_external/ -v

# Agent communication tests
python -m pytest tests/integration/test_agents/ -v
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
- **5.1**: Testing infrastructure properly organized ✓
- **5.2**: Clear separation between unit, integration, and end-to-end tests ✓
- **2.1**: FastAPI-compatible directory structure ✓
- **3.1**: Business logic separated from framework code ✓
- **3.2**: Clear interfaces between different layers ✓

### Service Types Tested
- VideoService ✓
- RAGService ✓
- AgentService ✓
- AWSService ✓

### Integration Patterns
- API endpoint workflows ✓
- Service layer orchestration ✓
- Database operations and relationships ✓
- External service connectivity ✓
- Error handling and recovery ✓
- Performance and monitoring ✓

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
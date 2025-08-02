# Design Document

## Overview

This design outlines the restructuring of the current project into a clean, scalable FastAPI-based architecture. The restructure will eliminate unused files, organize code into logical modules, and establish clear separation of concerns while maintaining all existing functionality.

## Architecture

### High-Level Structure

The new project structure will follow FastAPI best practices with clear separation between:

- **API Layer**: FastAPI routers and endpoints
- **Service Layer**: Business logic and orchestration
- **Data Layer**: Models, schemas, and database interactions
- **Infrastructure Layer**: External integrations (AWS, LangGraph, etc.)
- **Configuration Layer**: Environment and application configuration

### Directory Structure

```
project_root/
├── app/                          # Main FastAPI application
│   ├── __init__.py
│   ├── main.py                   # FastAPI app initialization
│   ├── api/                      # API endpoints
│   │   ├── __init__.py
│   │   ├── v1/                   # API version 1
│   │   │   ├── __init__.py
│   │   │   ├── endpoints/        # Individual endpoint modules
│   │   │   │   ├── __init__.py
│   │   │   │   ├── video.py      # Video generation endpoints
│   │   │   │   ├── rag.py        # RAG system endpoints
│   │   │   │   ├── aws.py        # AWS integration endpoints
│   │   │   │   └── agents.py     # LangGraph agents endpoints
│   │   │   └── router.py         # Main API router
│   │   └── dependencies.py       # FastAPI dependencies
│   ├── core/                     # Core business logic
│   │   ├── __init__.py
│   │   ├── video/                # Video generation core
│   │   ├── rag/                  # RAG system core
│   │   ├── agents/               # LangGraph agents core
│   │   └── aws/                  # AWS integration core
│   ├── models/                   # Data models and schemas
│   │   ├── __init__.py
│   │   ├── database/             # Database models
│   │   ├── schemas/              # Pydantic schemas
│   │   └── enums.py              # Enumerations
│   ├── services/                 # Service layer
│   │   ├── __init__.py
│   │   ├── video_service.py
│   │   ├── rag_service.py
│   │   ├── agent_service.py
│   │   └── aws_service.py
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py
│   │   ├── logging.py
│   │   ├── exceptions.py
│   │   └── helpers.py
│   └── config/                   # Configuration management
│       ├── __init__.py
│       ├── settings.py           # Pydantic settings
│       └── database.py           # Database configuration
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   ├── e2e/                      # End-to-end tests
│   ├── fixtures/                 # Test fixtures
│   └── conftest.py               # Pytest configuration
├── scripts/                      # Utility scripts
│   ├── setup.py                  # Setup scripts
│   └── migrate.py                # Migration scripts
├── docs/                         # Documentation
├── deployment/                   # Deployment configurations
│   ├── docker/
│   └── k8s/
├── .env.example                  # Environment template
├── requirements.txt              # Dependencies
├── pyproject.toml               # Project configuration
└── README.md                    # Project documentation
```

## Components and Interfaces

### API Layer

**FastAPI Application (`app/main.py`)**
- Initialize FastAPI app with middleware, CORS, and exception handlers
- Include API routers from different modules
- Configure OpenAPI documentation

**API Routers (`app/api/v1/endpoints/`)**
- Video generation endpoints: `/api/v1/video/`
- RAG system endpoints: `/api/v1/rag/`
- AWS integration endpoints: `/api/v1/aws/`
- LangGraph agents endpoints: `/api/v1/agents/`

**Dependencies (`app/api/dependencies.py`)**
- Authentication and authorization
- Database session management
- Configuration injection

### Service Layer

**Video Service (`app/services/video_service.py`)**
- Orchestrates video generation workflow
- Integrates with Manim and TTS systems
- Manages video processing pipeline

**RAG Service (`app/services/rag_service.py`)**
- Handles document retrieval and generation
- Manages vector stores and embeddings
- Implements query processing logic

**Agent Service (`app/services/agent_service.py`)**
- Manages LangGraph agent workflows
- Handles agent state and execution
- Provides agent orchestration

**AWS Service (`app/services/aws_service.py`)**
- Manages S3 operations and metadata
- Handles DynamoDB interactions
- Provides AWS resource management

### Data Layer

**Database Models (`app/models/database/`)**
- SQLAlchemy or similar ORM models
- Database table definitions
- Relationship mappings

**Pydantic Schemas (`app/models/schemas/`)**
- Request/response models
- Data validation schemas
- API documentation models

### Configuration Management

**Settings (`app/config/settings.py`)**
- Pydantic-based configuration
- Environment variable management
- Feature flags and toggles

## Data Models

### Core Entities

```python
# Video Generation
class VideoRequest(BaseModel):
    script: str
    voice_settings: VoiceSettings
    animation_config: AnimationConfig

class VideoResponse(BaseModel):
    video_id: str
    status: VideoStatus
    download_url: Optional[str]

# RAG System
class RAGQuery(BaseModel):
    query: str
    context: Optional[str]
    filters: Optional[Dict[str, Any]]

class RAGResponse(BaseModel):
    answer: str
    sources: List[Source]
    confidence: float

# Agent System
class AgentRequest(BaseModel):
    agent_type: AgentType
    input_data: Dict[str, Any]
    config: Optional[AgentConfig]

class AgentResponse(BaseModel):
    result: Dict[str, Any]
    execution_time: float
    status: AgentStatus
```

## Error Handling

### Exception Hierarchy

```python
class AppException(Exception):
    """Base application exception"""
    pass

class ValidationError(AppException):
    """Data validation errors"""
    pass

class ServiceError(AppException):
    """Service layer errors"""
    pass

class ExternalServiceError(AppException):
    """External service integration errors"""
    pass
```

### Error Response Format

```python
class ErrorResponse(BaseModel):
    error_code: str
    message: str
    details: Optional[Dict[str, Any]]
    timestamp: datetime
```

## Testing Strategy

### Unit Tests
- Test individual functions and methods
- Mock external dependencies
- Focus on business logic validation

### Integration Tests
- Test service interactions
- Database integration testing
- External API integration testing

### End-to-End Tests
- Full workflow testing
- API endpoint testing
- User scenario validation

### Test Organization
```
tests/
├── unit/
│   ├── test_services/
│   ├── test_models/
│   └── test_utils/
├── integration/
│   ├── test_api/
│   ├── test_database/
│   └── test_external/
└── e2e/
    ├── test_video_workflow/
    ├── test_rag_workflow/
    └── test_agent_workflow/
```

## Migration Strategy

### Phase 1: File Cleanup
1. Remove unused cache files and temporary directories
2. Delete redundant configuration files
3. Clean up old documentation and logs

### Phase 2: Core Restructure
1. Create new directory structure
2. Move existing modules to appropriate locations
3. Update import statements and dependencies

### Phase 3: FastAPI Integration
1. Create FastAPI application structure
2. Implement API endpoints
3. Add middleware and configuration

### Phase 4: Testing and Validation
1. Update test suite structure
2. Ensure all functionality works
3. Performance testing and optimization

## Files to Remove

### Cache and Temporary Files
- `__pycache__/` directories
- `.pytest_cache/`
- `test_output/`
- `demo_output/`
- `example_output/`
- `.langgraph_api/` (if not needed)

### Redundant Files
- Old setup scripts that are no longer used
- Duplicate configuration files
- Unused example files
- Legacy documentation files

### Development Artifacts
- `.egg-info` directories
- Temporary log files
- Old migration files
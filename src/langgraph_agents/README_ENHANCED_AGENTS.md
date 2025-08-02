# Enhanced LangGraph Agents with AWS Integration

This document describes the enhanced LangGraph agents that integrate with AWS S3 and DynamoDB for cloud storage and metadata management.

## Overview

The enhanced agents extend the existing LangGraph agent system with AWS capabilities while maintaining full backward compatibility. They provide:

- **Automatic S3 upload** for rendered videos and generated code
- **DynamoDB metadata management** for tracking video and code versions
- **Graceful degradation** when AWS services are unavailable
- **Progress tracking** for upload operations
- **Version management** for code files

## Enhanced Agents

### 1. EnhancedRendererAgent

Extends `RendererAgent` with S3 video upload capabilities.

#### Features

- **Automatic Video Upload**: Uploads rendered videos to S3 after successful rendering
- **Progress Tracking**: Real-time upload progress with callbacks
- **Metadata Updates**: Updates DynamoDB with S3 paths and video metadata
- **Graceful Degradation**: Continues workflow even if AWS upload fails (configurable)
- **Retry Logic**: Exponential backoff for failed uploads

#### Usage

```python
from langgraph_agents.agents.enhanced_renderer_agent import EnhancedRendererAgent
from aws.config import AWSConfig

# Initialize with AWS configuration
aws_config = AWSConfig()
agent = EnhancedRendererAgent(agent_config, system_config, aws_config)

# Add progress callback
def progress_callback(video_id, bytes_uploaded, total_bytes, percentage):
    print(f"Upload progress: {percentage:.1f}%")

agent.add_upload_progress_callback(progress_callback)

# Execute with AWS upload enabled
state = VideoGenerationState(
    enable_aws_upload=True,
    # ... other state parameters
)
result = await agent.execute(state)
```

#### Configuration

```python
# State parameters for controlling AWS upload
state = VideoGenerationState(
    enable_aws_upload=True,          # Enable/disable AWS upload
    project_id="my_project",         # Project identifier for S3 organization
    video_id="unique_video_id",      # Unique video identifier
    version=1,                       # Video version number
    # ... other parameters
)
```

### 2. EnhancedCodeGeneratorAgent

Extends `CodeGeneratorAgent` with S3 code storage and version management.

#### Features

- **Code Download**: Downloads existing code for editing workflows
- **Version Management**: Automatic versioning of code files
- **S3 Storage**: Stores generated code in S3 with UTF-8 encoding
- **Metadata Tracking**: Updates DynamoDB with code paths and versions
- **Fallback Mechanisms**: Handles download failures gracefully

#### Usage

```python
from langgraph_agents.agents.enhanced_code_generator_agent import EnhancedCodeGeneratorAgent
from aws.config import AWSConfig

# Initialize with AWS configuration
aws_config = AWSConfig()
agent = EnhancedCodeGeneratorAgent(agent_config, system_config, aws_config)

# Execute with code management enabled
state = VideoGenerationState(
    enable_aws_code_management=True,
    editing_existing_video=True,     # Download existing code
    current_version=2,               # Current code version
    # ... other state parameters
)
result = await agent.execute(state)
```

#### Configuration

```python
# State parameters for controlling code management
state = VideoGenerationState(
    enable_aws_code_management=True,  # Enable/disable AWS code management
    editing_existing_video=False,     # Whether to download existing code
    enable_code_object_lock=False,    # Enable S3 Object Lock for critical versions
    project_id="my_project",          # Project identifier
    video_id="unique_video_id",       # Video identifier
    current_version=1,                # Current version (for editing)
    # ... other parameters
)
```

## AWS Integration Service

The `AWSIntegrationService` provides a unified interface for all AWS operations used by the enhanced agents.

### Features

- **Unified Interface**: Single service for S3 and DynamoDB operations
- **Health Checks**: Monitor AWS service availability
- **Integrated Workflows**: Combined upload and metadata operations
- **Error Handling**: Comprehensive error handling and retry logic

### Usage

```python
from aws.aws_integration_service import AWSIntegrationService
from aws.config import AWSConfig

# Initialize service
aws_config = AWSConfig()
service = AWSIntegrationService(aws_config)

# Health check
health = await service.health_check()
print(f"AWS services status: {health['overall_status']}")

# Upload video with metadata
upload_result = await service.upload_video_with_metadata(
    chunks=video_chunks,
    video_metadata=metadata,
    progress_callback=progress_callback
)
```

## Configuration

### AWS Configuration

The enhanced agents use the existing AWS configuration system:

```python
from aws.config import AWSConfig

# Load from environment variables
aws_config = AWSConfig()

# Required environment variables:
# AWS_REGION=us-east-1
# AWS_S3_VIDEO_BUCKET=my-video-bucket
# AWS_S3_CODE_BUCKET=my-code-bucket
# AWS_DYNAMODB_TABLE=VideoMetadata
```

### Agent Configuration

Enhanced agents accept the same configuration as base agents plus AWS config:

```python
from langgraph_agents.state import AgentConfig

agent_config = AgentConfig(
    name="enhanced_renderer",
    agent_type="enhanced_renderer",
    enabled=True,
    max_retries=3,
    timeout_seconds=300,
    enable_human_loop=False
)
```

## State Management

### Enhanced State Parameters

The enhanced agents add new state parameters:

```python
# AWS Upload Control
enable_aws_upload: bool = True
enable_aws_code_management: bool = True

# Upload Results
aws_upload_results: Dict[str, str] = {}
code_upload_results: Dict[str, str] = {}
upload_status: str = "not_started"  # not_started, completed, failed, failed_graceful
code_upload_status: str = "not_started"

# Version Management
current_version: int = 1
version: int = 1
editing_existing_video: bool = False

# Progress Tracking
upload_retry_count: int = 0
code_upload_retry_count: int = 0
```

## Error Handling

### Graceful Degradation

Enhanced agents support graceful degradation when AWS services are unavailable:

```python
# Configure graceful degradation
aws_config = AWSConfig()
aws_config.require_aws_upload = False  # Allow workflow to continue without AWS

# Agent will log warnings but continue processing
agent = EnhancedRendererAgent(config, system_config, aws_config)
```

### Retry Logic

Both agents implement exponential backoff retry logic:

- **Initial retry delay**: 1 second
- **Maximum retries**: 3 (configurable)
- **Backoff multiplier**: 2x with jitter
- **Retryable errors**: Network timeouts, service unavailable, throttling

### Error Escalation

Agents can escalate to human intervention for critical errors:

```python
# Enable human intervention for AWS errors
agent_config.enable_human_loop = True

# Agent will create intervention commands for unrecoverable errors
```

## Monitoring and Observability

### Progress Tracking

```python
# Add progress callbacks
def upload_progress(video_id, bytes_uploaded, total_bytes, percentage):
    print(f"Uploading {video_id}: {percentage:.1f}%")

agent.add_upload_progress_callback(upload_progress)

# Get current progress
progress = agent.get_upload_progress(video_id)
```

### Status Reporting

```python
# Get comprehensive status
upload_status = agent.get_upload_status(state)
code_status = agent.get_code_management_status(state)

# Status includes:
# - AWS service availability
# - Upload progress and results
# - Error information
# - Retry counts
```

### Health Monitoring

```python
# Check AWS service health
health = await aws_service.health_check()

# Returns status for:
# - S3 video bucket
# - S3 code bucket  
# - DynamoDB table
```

## Testing

### Unit Tests

```bash
# Run basic functionality tests
python src/langgraph_agents/test_enhanced_agents_simple.py

# Run comprehensive verification
python src/langgraph_agents/verify_task_4_completion.py
```

### Integration Tests

```python
# Test with actual AWS services (requires credentials)
from langgraph_agents.test_enhanced_agents import main
await main()
```

## Best Practices

### 1. Configuration Management

- Use environment variables for AWS configuration
- Enable graceful degradation for non-critical workflows
- Configure appropriate retry limits

### 2. Error Handling

- Always check upload status in state
- Implement progress callbacks for user feedback
- Handle both retryable and non-retryable errors

### 3. Performance

- Use appropriate S3 multipart thresholds
- Limit concurrent uploads to avoid throttling
- Enable caching for frequently accessed metadata

### 4. Security

- Use IAM roles instead of access keys
- Enable S3 server-side encryption
- Implement proper access controls

## Migration Guide

### From Base Agents

1. **Import Enhanced Agents**:
   ```python
   # Old
   from langgraph_agents.agents.renderer_agent import RendererAgent
   
   # New
   from langgraph_agents.agents.enhanced_renderer_agent import EnhancedRendererAgent
   ```

2. **Add AWS Configuration**:
   ```python
   # Initialize with AWS config
   aws_config = AWSConfig()
   agent = EnhancedRendererAgent(config, system_config, aws_config)
   ```

3. **Update State Parameters**:
   ```python
   # Add AWS control parameters
   state.update({
       'enable_aws_upload': True,
       'project_id': 'my_project',
       'video_id': 'unique_id'
   })
   ```

### Backward Compatibility

Enhanced agents are fully backward compatible:

- Work without AWS configuration (AWS disabled)
- Accept same state parameters as base agents
- Maintain same method signatures
- Preserve existing functionality

## Troubleshooting

### Common Issues

1. **AWS Credentials Not Found**
   ```
   Solution: Set AWS credentials via environment variables or IAM roles
   ```

2. **S3 Bucket Access Denied**
   ```
   Solution: Verify bucket permissions and IAM policies
   ```

3. **DynamoDB Table Not Found**
   ```
   Solution: Create table or verify table name in configuration
   ```

4. **Upload Timeouts**
   ```
   Solution: Increase timeout settings or reduce file sizes
   ```

### Debug Logging

```python
import logging
logging.getLogger('langgraph_agents').setLevel(logging.DEBUG)
logging.getLogger('aws').setLevel(logging.DEBUG)
```

## Support

For issues and questions:

1. Check the verification script output
2. Review AWS service health status
3. Enable debug logging
4. Check AWS credentials and permissions
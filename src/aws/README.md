# AWS Integration Foundation

This module provides the foundation for AWS SDK integration in the video generation system, including configuration management, credentials handling, logging, and error handling.

## Features

- **Configuration Management**: Environment variable-based configuration with validation
- **Credentials Management**: Secure AWS credentials handling using IAM roles and boto3 sessions
- **Comprehensive Logging**: Structured logging for AWS operations with performance monitoring
- **Error Handling**: Custom exceptions with proper error categorization and retry logic
- **Security**: Encryption support and secure credential management practices

## Quick Start

### 1. Install Dependencies

The required dependencies are already included in `requirements.txt`:
- `boto3~=1.36.9`
- `python-dotenv~=0.21.1`

### 2. Configure Environment Variables

Add AWS configuration to your `.env` file:

```bash
# Core AWS Settings
AWS_REGION="us-east-1"
AWS_PROFILE=""

# S3 Configuration
AWS_S3_VIDEO_BUCKET="your-video-bucket"
AWS_S3_CODE_BUCKET="your-code-bucket"

# Feature Flags
AWS_ENABLE_UPLOAD="true"
AWS_ENABLE_ENCRYPTION="true"

# Logging
AWS_LOG_LEVEL="INFO"
```

### 3. Basic Usage

```python
from aws.config import AWSConfig
from aws.credentials import AWSCredentialsManager
from aws.logging_config import setup_aws_logging

# Load configuration
config = AWSConfig.from_env()

# Set up logging
logger = setup_aws_logging(config)

# Initialize credentials
creds_manager = AWSCredentialsManager(config)

# Get AWS clients
s3_client = creds_manager.get_client('s3')
dynamodb = creds_manager.get_resource('dynamodb')
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_REGION` | `us-east-1` | AWS region |
| `AWS_PROFILE` | `None` | AWS profile name |
| `AWS_S3_VIDEO_BUCKET` | `""` | S3 bucket for videos |
| `AWS_S3_CODE_BUCKET` | `""` | S3 bucket for code |
| `AWS_DYNAMODB_METADATA_TABLE` | `VideoMetadata` | DynamoDB table name |
| `AWS_MULTIPART_THRESHOLD` | `104857600` | Multipart upload threshold (100MB) |
| `AWS_MAX_CONCURRENT_UPLOADS` | `3` | Max concurrent uploads |
| `AWS_CHUNK_SIZE` | `8388608` | Upload chunk size (8MB) |
| `AWS_ENABLE_ENCRYPTION` | `true` | Enable S3 encryption |
| `AWS_KMS_KEY_ID` | `""` | KMS key ID for encryption |
| `AWS_MAX_RETRIES` | `3` | Max retry attempts |
| `AWS_RETRY_BACKOFF_BASE` | `2.0` | Retry backoff multiplier |
| `AWS_ENABLE_UPLOAD` | `false` | Enable AWS upload functionality |
| `AWS_REQUIRE_UPLOAD` | `false` | Require AWS upload to succeed |
| `AWS_LOG_LEVEL` | `INFO` | Logging level |
| `AWS_ENABLE_BOTO3_LOGGING` | `false` | Enable boto3 debug logging |

### Configuration Validation

The configuration system validates:
- Required settings when AWS upload is enabled
- Numeric values are positive
- Region format is valid
- Log level is valid

## Credentials Management

### Credential Sources (in priority order)

1. Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
2. AWS credentials file (`~/.aws/credentials`)
3. AWS config file (`~/.aws/config`)
4. IAM roles for EC2 instances
5. IAM roles for ECS tasks
6. IAM roles for Lambda functions

### Best Practices

- **Production**: Use IAM roles instead of access keys
- **Development**: Use AWS profiles or environment variables
- **Security**: Never commit credentials to version control

### Testing Credentials

```python
creds_manager = AWSCredentialsManager(config)

# Test service access
s3_access = creds_manager.test_service_access('s3')
dynamodb_access = creds_manager.test_service_access('dynamodb')

# Get credential information
info = creds_manager.get_credentials_info()
print(f"Account: {info['account_id']}, ARN: {info['user_arn']}")
```

## Logging

### Setup

```python
from aws.logging_config import setup_aws_logging, aws_operation

config = AWSConfig(log_level='INFO')
logger = setup_aws_logging(config, log_file='logs/aws.log')
```

### Operation Logging

```python
# Using context manager
with aws_operation('upload_file', 's3', bucket='my-bucket', key='file.txt'):
    # AWS operation code
    s3_client.upload_file('local_file.txt', 'my-bucket', 'file.txt')
```

### Log Files

- Console output: Simplified format
- File output: Detailed format with function names and line numbers
- Rotation: 10MB max size, 5 backup files
- Performance metrics: Automatic timing for operations

## Error Handling

### Exception Hierarchy

```
AWSIntegrationError (base)
├── AWSConfigurationError
├── AWSCredentialsError
├── AWSS3Error
├── AWSDynamoDBError
├── AWSMediaConvertError
├── AWSCloudFrontError
├── AWSRetryableError
└── AWSNonRetryableError
```

### Usage

```python
from aws.exceptions import AWSConfigurationError, AWSCredentialsError

try:
    config = AWSConfig.from_env()
    creds_manager = AWSCredentialsManager(config)
    session = creds_manager.get_session()
except AWSConfigurationError as e:
    print(f"Configuration error: {e}")
except AWSCredentialsError as e:
    print(f"Credentials error: {e}")
```

## Testing

### Run Foundation Tests

```bash
python src/aws/test_foundation.py
```

### Run Usage Examples

```bash
python examples/aws_integration_usage.py
```

## Security Considerations

1. **Encryption**: Enable S3 server-side encryption
2. **IAM Roles**: Use minimal required permissions
3. **Logging**: Sensitive data is automatically filtered from logs
4. **Credentials**: Never log or expose credentials
5. **Network**: Use HTTPS for all AWS API calls

## Performance Optimization

1. **Connection Pooling**: Boto3 handles connection pooling automatically
2. **Multipart Uploads**: Configured for files > 100MB
3. **Concurrent Uploads**: Limited to 3 concurrent uploads
4. **Retry Logic**: Exponential backoff with jitter
5. **Caching**: Credential caching to avoid repeated authentication

## Troubleshooting

### Common Issues

1. **Credentials Not Found**
   - Check AWS CLI configuration: `aws configure list`
   - Verify IAM permissions
   - Check environment variables

2. **Configuration Validation Errors**
   - Ensure required buckets are specified when upload is enabled
   - Check numeric values are positive
   - Verify region format

3. **Permission Errors**
   - Review IAM policies
   - Check bucket policies
   - Verify resource ARNs

### Debug Logging

Enable detailed logging:

```bash
AWS_LOG_LEVEL="DEBUG"
AWS_ENABLE_BOTO3_LOGGING="true"
```

## Next Steps

After setting up the foundation:

1. Implement the AWS Integration Service (Task 2)
2. Add S3 upload functionality
3. Implement DynamoDB metadata management
4. Integrate with LangGraph agents

## Contributing

When adding new AWS functionality:

1. Use the existing configuration system
2. Add proper error handling with custom exceptions
3. Include comprehensive logging
4. Add tests for new functionality
5. Update documentation
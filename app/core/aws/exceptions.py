"""
AWS Integration Exceptions

Custom exception classes for AWS integration errors with proper error handling
and logging support.
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class AWSIntegrationError(Exception):
    """Base exception for AWS integration errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)
        
        # Log the error
        logger.error(f"AWS Integration Error: {message}", extra={
            'error_code': error_code,
            'details': details
        })
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class AWSConfigurationError(AWSIntegrationError):
    """Exception raised for AWS configuration errors."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="AWS_CONFIG_ERROR",
            details={'config_key': config_key} if config_key else None
        )
        self.config_key = config_key


class AWSCredentialsError(AWSIntegrationError):
    """Exception raised for AWS credentials errors."""
    
    def __init__(self, message: str, credential_source: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="AWS_CREDENTIALS_ERROR",
            details={'credential_source': credential_source} if credential_source else None
        )
        self.credential_source = credential_source


class AWSS3Error(AWSIntegrationError):
    """Exception raised for S3 operation errors."""
    
    def __init__(self, message: str, bucket: Optional[str] = None, 
                 key: Optional[str] = None, operation: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="AWS_S3_ERROR",
            details={
                'bucket': bucket,
                'key': key,
                'operation': operation
            }
        )
        self.bucket = bucket
        self.key = key
        self.operation = operation


class AWSDynamoDBError(AWSIntegrationError):
    """Exception raised for DynamoDB operation errors."""
    
    def __init__(self, message: str, table_name: Optional[str] = None, 
                 operation: Optional[str] = None, item_key: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="AWS_DYNAMODB_ERROR",
            details={
                'table_name': table_name,
                'operation': operation,
                'item_key': item_key
            }
        )
        self.table_name = table_name
        self.operation = operation
        self.item_key = item_key


class AWSMetadataError(AWSIntegrationError):
    """Exception raised for metadata management errors."""
    
    def __init__(self, message: str, video_id: Optional[str] = None, 
                 operation: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="AWS_METADATA_ERROR",
            details={
                'video_id': video_id,
                'operation': operation
            }
        )
        self.video_id = video_id
        self.operation = operation


class AWSMediaConvertError(AWSIntegrationError):
    """Exception raised for MediaConvert operation errors."""
    
    def __init__(self, message: str, job_id: Optional[str] = None, 
                 operation: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="AWS_MEDIACONVERT_ERROR",
            details={
                'job_id': job_id,
                'operation': operation
            }
        )
        self.job_id = job_id
        self.operation = operation


class AWSCloudFrontError(AWSIntegrationError):
    """Exception raised for CloudFront operation errors."""
    
    def __init__(self, message: str, distribution_id: Optional[str] = None, 
                 operation: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="AWS_CLOUDFRONT_ERROR",
            details={
                'distribution_id': distribution_id,
                'operation': operation
            }
        )
        self.distribution_id = distribution_id
        self.operation = operation


class AWSRetryableError(AWSIntegrationError):
    """Exception for errors that can be retried."""
    
    def __init__(self, message: str, retry_count: int = 0, max_retries: int = 3):
        super().__init__(
            message=message,
            error_code="AWS_RETRYABLE_ERROR",
            details={
                'retry_count': retry_count,
                'max_retries': max_retries
            }
        )
        self.retry_count = retry_count
        self.max_retries = max_retries
    
    def can_retry(self) -> bool:
        """Check if the error can be retried."""
        return self.retry_count < self.max_retries


class AWSNonRetryableError(AWSIntegrationError):
    """Exception for errors that should not be retried."""
    
    def __init__(self, message: str, reason: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="AWS_NON_RETRYABLE_ERROR",
            details={'reason': reason} if reason else None
        )
        self.reason = reason
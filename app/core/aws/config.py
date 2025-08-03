"""
AWS Configuration Management

Handles AWS service configuration with environment variable support,
validation, and secure defaults.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv
import logging

from .exceptions import AWSConfigurationError

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class AWSConfig:
    """
    AWS Configuration class with environment variable support and validation.
    
    Supports configuration through environment variables with AWS_ prefix
    and provides secure defaults for production use.
    """
    
    # Core AWS Settings
    region: str = field(default_factory=lambda: os.getenv('AWS_REGION', 'us-east-1'))
    profile: Optional[str] = field(default_factory=lambda: os.getenv('AWS_PROFILE'))
    
    # S3 Configuration
    video_bucket_name: str = field(default_factory=lambda: os.getenv('AWS_S3_VIDEO_BUCKET', ''))
    code_bucket_name: str = field(default_factory=lambda: os.getenv('AWS_S3_CODE_BUCKET', ''))
    
    # DynamoDB Configuration
    metadata_table_name: str = field(default_factory=lambda: os.getenv('AWS_DYNAMODB_METADATA_TABLE', 'VideoMetadata'))
    
    # Upload Configuration
    multipart_threshold: int = field(default_factory=lambda: int(os.getenv('AWS_MULTIPART_THRESHOLD', '104857600')))  # 100MB
    max_concurrent_uploads: int = field(default_factory=lambda: int(os.getenv('AWS_MAX_CONCURRENT_UPLOADS', '3')))
    chunk_size: int = field(default_factory=lambda: int(os.getenv('AWS_CHUNK_SIZE', '8388608')))  # 8MB
    
    # Security Configuration
    enable_encryption: bool = field(default_factory=lambda: os.getenv('AWS_ENABLE_ENCRYPTION', 'true').lower() == 'true')
    kms_key_id: Optional[str] = field(default_factory=lambda: os.getenv('AWS_KMS_KEY_ID'))
    
    # MediaConvert Configuration
    mediaconvert_role_arn: Optional[str] = field(default_factory=lambda: os.getenv('AWS_MEDIACONVERT_ROLE_ARN'))
    mediaconvert_endpoint: Optional[str] = field(default_factory=lambda: os.getenv('AWS_MEDIACONVERT_ENDPOINT'))
    
    # CloudFront Configuration
    cloudfront_distribution_id: Optional[str] = field(default_factory=lambda: os.getenv('AWS_CLOUDFRONT_DISTRIBUTION_ID'))
    cloudfront_domain: Optional[str] = field(default_factory=lambda: os.getenv('AWS_CLOUDFRONT_DOMAIN'))
    
    # Retry Configuration
    max_retries: int = field(default_factory=lambda: int(os.getenv('AWS_MAX_RETRIES', '3')))
    retry_backoff_base: float = field(default_factory=lambda: float(os.getenv('AWS_RETRY_BACKOFF_BASE', '2.0')))
    
    # Feature Flags
    enable_aws_upload: bool = field(default_factory=lambda: os.getenv('AWS_ENABLE_UPLOAD', 'false').lower() == 'true')
    require_aws_upload: bool = field(default_factory=lambda: os.getenv('AWS_REQUIRE_UPLOAD', 'false').lower() == 'true')
    enable_transcoding: bool = field(default_factory=lambda: os.getenv('AWS_ENABLE_TRANSCODING', 'false').lower() == 'true')
    enable_cloudfront: bool = field(default_factory=lambda: os.getenv('AWS_ENABLE_CLOUDFRONT', 'false').lower() == 'true')
    
    # Logging Configuration
    log_level: str = field(default_factory=lambda: os.getenv('AWS_LOG_LEVEL', 'INFO'))
    enable_boto3_logging: bool = field(default_factory=lambda: os.getenv('AWS_ENABLE_BOTO3_LOGGING', 'false').lower() == 'true')
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
        logger.info(f"AWS Configuration initialized for region: {self.region}")
    
    def validate(self) -> None:
        """
        Validate AWS configuration settings.
        
        Raises:
            AWSConfigurationError: If configuration is invalid
        """
        errors = []
        
        # Validate required settings if AWS upload is enabled
        if self.enable_aws_upload:
            if not self.video_bucket_name:
                errors.append("AWS_S3_VIDEO_BUCKET is required when AWS upload is enabled")
            
            if not self.code_bucket_name:
                errors.append("AWS_S3_CODE_BUCKET is required when AWS upload is enabled")
        
        # Validate transcoding configuration
        if self.enable_transcoding and not self.mediaconvert_role_arn:
            errors.append("AWS_MEDIACONVERT_ROLE_ARN is required when transcoding is enabled")
        
        # Validate numeric settings
        if self.multipart_threshold <= 0:
            errors.append("AWS_MULTIPART_THRESHOLD must be positive")
        
        if self.max_concurrent_uploads <= 0:
            errors.append("AWS_MAX_CONCURRENT_UPLOADS must be positive")
        
        if self.chunk_size <= 0:
            errors.append("AWS_CHUNK_SIZE must be positive")
        
        if self.max_retries < 0:
            errors.append("AWS_MAX_RETRIES must be non-negative")
        
        if self.retry_backoff_base <= 1.0:
            errors.append("AWS_RETRY_BACKOFF_BASE must be greater than 1.0")
        
        # Validate region format
        if not self.region or len(self.region.split('-')) < 3:
            errors.append(f"Invalid AWS region format: {self.region}")
        
        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level.upper() not in valid_log_levels:
            errors.append(f"Invalid log level: {self.log_level}. Must be one of {valid_log_levels}")
        
        if errors:
            raise AWSConfigurationError(f"AWS configuration validation failed: {'; '.join(errors)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'region': self.region,
            'profile': self.profile,
            'video_bucket_name': self.video_bucket_name,
            'code_bucket_name': self.code_bucket_name,
            'metadata_table_name': self.metadata_table_name,
            'multipart_threshold': self.multipart_threshold,
            'max_concurrent_uploads': self.max_concurrent_uploads,
            'chunk_size': self.chunk_size,
            'enable_encryption': self.enable_encryption,
            'kms_key_id': self.kms_key_id,
            'mediaconvert_role_arn': self.mediaconvert_role_arn,
            'mediaconvert_endpoint': self.mediaconvert_endpoint,
            'cloudfront_distribution_id': self.cloudfront_distribution_id,
            'cloudfront_domain': self.cloudfront_domain,
            'max_retries': self.max_retries,
            'retry_backoff_base': self.retry_backoff_base,
            'enable_aws_upload': self.enable_aws_upload,
            'require_aws_upload': self.require_aws_upload,
            'enable_transcoding': self.enable_transcoding,
            'enable_cloudfront': self.enable_cloudfront,
            'log_level': self.log_level,
            'enable_boto3_logging': self.enable_boto3_logging
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AWSConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_env(cls) -> 'AWSConfig':
        """Create configuration from environment variables."""
        return cls()
    
    def is_enabled(self) -> bool:
        """Check if AWS integration is enabled."""
        return self.enable_aws_upload
    
    def get_boto3_config(self) -> Dict[str, Any]:
        """Get boto3 client configuration."""
        from botocore.config import Config
        
        config = {
            'region_name': self.region,
            'config': Config(
                retries={
                    'max_attempts': self.max_retries,
                    'mode': 'adaptive'
                }
            )
        }
        
        return config
    
    def get_s3_transfer_config(self) -> Dict[str, Any]:
        """Get S3 transfer configuration for boto3."""
        return {
            'multipart_threshold': self.multipart_threshold,
            'max_concurrency': self.max_concurrent_uploads,
            'multipart_chunksize': self.chunk_size,
            'use_threads': True
        }
    
    def __repr__(self) -> str:
        """String representation of configuration (without sensitive data)."""
        return (
            f"AWSConfig(region='{self.region}', "
            f"video_bucket='{self.video_bucket_name}', "
            f"code_bucket='{self.code_bucket_name}', "
            f"metadata_table='{self.metadata_table_name}', "
            f"enabled={self.enable_aws_upload})"
        )
"""
AWS Integration Core Module

This module provides AWS SDK integration for the video generation system,
including S3 storage, DynamoDB metadata management, and related services.
"""

from .config import AWSConfig
from .credentials import AWSCredentialsManager
from .exceptions import (
    AWSIntegrationError, 
    AWSConfigurationError, 
    AWSCredentialsError, 
    AWSS3Error,
    AWSDynamoDBError,
    AWSMetadataError,
    AWSMediaConvertError
)
from .logging_config import setup_aws_logging
from .s3_video_upload import S3VideoUploadService, VideoChunk, ProgressPercentage
from .s3_code_storage import S3CodeStorageService, CodeMetadata, CodeVersion
from .multipart_upload_handler import (
    MultipartUploadHandler, 
    MultipartUploadInfo, 
    UploadPart, 
    MultipartProgressTracker
)
from .metadata_service import MetadataService
from .cloudfront_service import CloudFrontService
from .mediaconvert_service import (
    MediaConvertService, 
    TranscodingJobConfig, 
    TranscodingJobResult, 
    TranscodingStatus, 
    OutputFormat, 
    QualityLevel
)
from .aws_integration_service import AWSIntegrationService

__all__ = [
    'AWSConfig',
    'AWSCredentialsManager', 
    'AWSIntegrationError',
    'AWSConfigurationError',
    'AWSCredentialsError',
    'AWSS3Error',
    'AWSDynamoDBError',
    'AWSMetadataError',
    'AWSMediaConvertError',
    'setup_aws_logging',
    'S3VideoUploadService',
    'VideoChunk',
    'ProgressPercentage',
    'S3CodeStorageService',
    'CodeMetadata',
    'CodeVersion',
    'MultipartUploadHandler',
    'MultipartUploadInfo',
    'UploadPart',
    'MultipartProgressTracker',
    'MetadataService',
    'CloudFrontService',
    'MediaConvertService',
    'TranscodingJobConfig',
    'TranscodingJobResult',
    'TranscodingStatus',
    'OutputFormat',
    'QualityLevel',
    'AWSIntegrationService'
]
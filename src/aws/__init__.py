"""
AWS Integration Module

This module provides AWS SDK integration for the video generation system,
including S3 storage, DynamoDB metadata management, and related services.
"""

from .config import AWSConfig
from .credentials import AWSCredentialsManager
from .exceptions import AWSIntegrationError, AWSConfigurationError, AWSCredentialsError, AWSS3Error
from .logging_config import setup_aws_logging
from .s3_video_upload import S3VideoUploadService, VideoChunk, ProgressPercentage
from .multipart_upload_handler import (
    MultipartUploadHandler, 
    MultipartUploadInfo, 
    UploadPart, 
    MultipartProgressTracker
)

__all__ = [
    'AWSConfig',
    'AWSCredentialsManager', 
    'AWSIntegrationError',
    'AWSConfigurationError',
    'AWSCredentialsError',
    'AWSS3Error',
    'setup_aws_logging',
    'S3VideoUploadService',
    'VideoChunk',
    'ProgressPercentage',
    'MultipartUploadHandler',
    'MultipartUploadInfo',
    'UploadPart',
    'MultipartProgressTracker'
]
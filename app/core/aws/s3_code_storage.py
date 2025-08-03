"""
S3 Code Storage Service

Implements S3 code storage functionality with UTF-8 encoding, versioning system,
download functionality, and S3 Object Lock for critical code versions.
"""

import os
import sys
import time
import random
import asyncio
import logging
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from boto3.s3.transfer import TransferConfig

from .config import AWSConfig
from .credentials import AWSCredentialsManager
from .exceptions import AWSS3Error, AWSRetryableError, AWSNonRetryableError
from .logging_config import aws_operation

logger = logging.getLogger(__name__)


@dataclass
class CodeMetadata:
    """Represents code metadata for S3 storage."""
    video_id: str
    project_id: str
    version: int
    scene_number: Optional[int] = None
    created_at: Optional[datetime] = None
    metadata: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        """Validate code metadata."""
        if not self.video_id or not self.project_id:
            raise ValueError("video_id and project_id are required")
        
        if self.version < 1:
            raise ValueError("version must be >= 1")
        
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class CodeVersion:
    """Represents a code version with its content and metadata."""
    content: str
    metadata: CodeMetadata
    s3_path: Optional[str] = None
    file_size: Optional[int] = None
    
    def __post_init__(self):
        """Calculate file size from content."""
        if self.file_size is None:
            self.file_size = len(self.content.encode('utf-8'))


class S3CodeStorageService:
    """
    Service for storing and retrieving Manim code in S3 with versioning,
    UTF-8 encoding, and S3 Object Lock support.
    """
    
    def __init__(self, config: AWSConfig, credentials_manager: AWSCredentialsManager):
        self.config = config
        self.credentials_manager = credentials_manager
        
        # Initialize S3 client and resource
        self.s3_client = credentials_manager.get_client('s3')
        self.s3_resource = credentials_manager.get_resource('s3')
        
        logger.info(f"S3 Code Storage Service initialized - Bucket: {config.code_bucket_name}")
    
    async def upload_code(self, code: str, metadata: CodeMetadata, 
                         enable_object_lock: bool = False) -> str:
        """
        Upload Manim code to S3 with UTF-8 encoding and versioning.
        
        Args:
            code: Manim code content as string
            metadata: Code metadata
            enable_object_lock: Enable S3 Object Lock for critical versions
            
        Returns:
            S3 URL of uploaded code
            
        Raises:
            AWSS3Error: If upload fails after all retries
        """
        # Generate S3 key with versioning
        s3_key = self._generate_s3_key(metadata)
        
        # Prepare upload arguments
        extra_args = self._prepare_upload_metadata(metadata, enable_object_lock)
        
        with aws_operation('upload_code', 's3',
                          bucket=self.config.code_bucket_name,
                          video_id=metadata.video_id,
                          version=metadata.version):
            
            try:
                # Upload code with retry logic
                await self._upload_code_with_retry(
                    code=code,
                    bucket=self.config.code_bucket_name,
                    key=s3_key,
                    extra_args=extra_args
                )
                
                s3_url = f"s3://{self.config.code_bucket_name}/{s3_key}"
                logger.info(f"Successfully uploaded code: {s3_url}")
                
                return s3_url
                
            except Exception as e:
                logger.error(f"Failed to upload code for {metadata.video_id} v{metadata.version}: {e}")
                raise AWSS3Error(
                    f"Failed to upload code: {str(e)}",
                    bucket=self.config.code_bucket_name,
                    key=s3_key,
                    operation="upload_code"
                ) from e   
 
    async def download_code(self, metadata: CodeMetadata) -> str:
        """
        Download Manim code from S3.
        
        Args:
            metadata: Code metadata specifying what to download
            
        Returns:
            Code content as string
            
        Raises:
            AWSS3Error: If download fails
        """
        s3_key = self._generate_s3_key(metadata)
        
        with aws_operation('download_code', 's3',
                          bucket=self.config.code_bucket_name,
                          video_id=metadata.video_id,
                          version=metadata.version):
            
            try:
                # Download code with retry logic
                code_content = await self._download_code_with_retry(
                    bucket=self.config.code_bucket_name,
                    key=s3_key
                )
                
                logger.info(f"Successfully downloaded code: {metadata.video_id} v{metadata.version}")
                return code_content
                
            except ClientError as e:
                error_code = e.response['Error']['Code']
                
                if error_code == 'NoSuchKey':
                    raise AWSS3Error(
                        f"Code not found: {metadata.video_id} v{metadata.version}",
                        bucket=self.config.code_bucket_name,
                        key=s3_key,
                        operation="download_code"
                    ) from e
                else:
                    raise AWSS3Error(
                        f"Failed to download code: {e.response['Error']['Message']}",
                        bucket=self.config.code_bucket_name,
                        key=s3_key,
                        operation="download_code"
                    ) from e
            
            except Exception as e:
                raise AWSS3Error(
                    f"Failed to download code: {str(e)}",
                    bucket=self.config.code_bucket_name,
                    key=s3_key,
                    operation="download_code"
                ) from e
    
    async def list_code_versions(self, project_id: str, video_id: str) -> List[CodeVersion]:
        """
        List all code versions for a video.
        
        Args:
            project_id: Project identifier
            video_id: Video identifier
            
        Returns:
            List of CodeVersion objects
            
        Raises:
            AWSS3Error: If listing fails
        """
        prefix = f"code/{project_id}/{video_id}/"
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.config.code_bucket_name,
                Prefix=prefix
            )
            
            versions = []
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    key = obj['Key']
                    # Extract version from filename (e.g., video_id_v1.py -> 1)
                    if key.endswith('.py'):
                        filename = os.path.basename(key)
                        if '_v' in filename:
                            try:
                                version_str = filename.split('_v')[1].split('.')[0]
                                version = int(version_str)
                                
                                # Create CodeVersion object
                                metadata = CodeMetadata(
                                    video_id=video_id,
                                    project_id=project_id,
                                    version=version
                                )
                                
                                code_version = CodeVersion(
                                    content="",  # Content not loaded in listing
                                    metadata=metadata,
                                    s3_path=f"s3://{self.config.code_bucket_name}/{key}",
                                    file_size=obj['Size']
                                )
                                
                                versions.append(code_version)
                                
                            except (IndexError, ValueError):
                                logger.warning(f"Could not parse version from filename: {filename}")
            
            versions.sort(key=lambda v: v.metadata.version)
            logger.info(f"Found {len(versions)} code versions for {video_id}")
            
            return versions
            
        except ClientError as e:
            raise AWSS3Error(
                f"Failed to list code versions: {e.response['Error']['Message']}",
                bucket=self.config.code_bucket_name,
                operation="list_code_versions"
            ) from e
        
        except Exception as e:
            raise AWSS3Error(
                f"Failed to list code versions: {str(e)}",
                bucket=self.config.code_bucket_name,
                operation="list_code_versions"
            ) from e
    
    def _generate_s3_key(self, metadata: CodeMetadata) -> str:
        """
        Generate S3 key with versioning system.
        Format: code/{project_id}/{video_id}/{video_id}_v{version}.py
        Or for scene-specific: code/{project_id}/{video_id}/{video_id}_scene{scene_number}_v{version}.py
        """
        if metadata.scene_number is not None:
            filename = f"{metadata.video_id}_scene{metadata.scene_number}_v{metadata.version}.py"
        else:
            filename = f"{metadata.video_id}_v{metadata.version}.py"
        s3_key = f"code/{metadata.project_id}/{metadata.video_id}/{filename}"
        return s3_key
    
    def _prepare_upload_metadata(self, metadata: CodeMetadata, 
                                enable_object_lock: bool = False) -> Dict[str, Any]:
        """
        Prepare metadata and extra arguments for S3 code upload.
        
        Args:
            metadata: Code metadata
            enable_object_lock: Enable S3 Object Lock
            
        Returns:
            Dictionary of extra arguments for S3 upload
        """
        # Base metadata
        s3_metadata = {
            'video_id': metadata.video_id,
            'project_id': metadata.project_id,
            'version': str(metadata.version),
            'upload_timestamp': metadata.created_at.isoformat(),
            'content_type': 'text/x-python'
        }
        
        # Add scene number if provided
        if metadata.scene_number is not None:
            s3_metadata['scene_number'] = str(metadata.scene_number)
        
        # Add custom metadata if provided
        if metadata.metadata:
            s3_metadata.update(metadata.metadata)
        
        # Prepare extra arguments
        extra_args = {
            'Metadata': s3_metadata,
            'ContentType': 'text/x-python',
            'ContentEncoding': 'utf-8'
        }
        
        # Add server-side encryption
        if self.config.enable_encryption:
            if self.config.kms_key_id:
                # Use KMS encryption
                extra_args.update({
                    'ServerSideEncryption': 'aws:kms',
                    'SSEKMSKeyId': self.config.kms_key_id
                })
                logger.debug(f"Using KMS encryption for code with key: {self.config.kms_key_id}")
            else:
                # Use S3 managed encryption (SSE-S3)
                extra_args['ServerSideEncryption'] = 'AES256'
                logger.debug("Using S3 managed encryption (SSE-S3) for code")
        
        return extra_args
    
    async def _upload_code_with_retry(self, code: str, bucket: str, key: str,
                                     extra_args: Dict[str, Any]) -> None:
        """
        Upload code with retry logic and exponential backoff.
        
        Args:
            code: Code content as string
            bucket: S3 bucket name
            key: S3 object key
            extra_args: Extra arguments for upload
            
        Raises:
            AWSS3Error: If upload fails after all retries
        """
        max_retries = self.config.max_retries
        base_delay = self.config.retry_backoff_base
        
        # Encode code to UTF-8 bytes
        code_bytes = code.encode('utf-8')
        
        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                # Perform the upload
                self.s3_client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=code_bytes,
                    **extra_args
                )
                
                # Upload successful
                if attempt > 0:
                    logger.info(f"Code upload succeeded on attempt {attempt + 1}")
                
                return
                
            except ClientError as e:
                error_code = e.response['Error']['Code']
                error_message = e.response['Error']['Message']
                
                # Check if this is a retryable error
                if not self._is_retryable_error(error_code):
                    raise AWSNonRetryableError(
                        f"Non-retryable S3 error: {error_message}",
                        reason=error_code
                    ) from e
                
                # Check if we've exhausted retries
                if attempt >= max_retries:
                    raise AWSRetryableError(
                        f"Code upload failed after {max_retries + 1} attempts: {error_message}",
                        retry_count=attempt,
                        max_retries=max_retries
                    ) from e
                
                # Calculate exponential backoff delay with jitter
                delay = (base_delay ** attempt) + (random.random() * 0.1)
                
                logger.warning(
                    f"Code upload attempt {attempt + 1} failed ({error_code}): {error_message}. "
                    f"Retrying in {delay:.2f}s..."
                )
                
                await asyncio.sleep(delay)
                
            except Exception as e:
                # Handle non-ClientError exceptions
                if attempt >= max_retries:
                    raise AWSS3Error(
                        f"Code upload failed after {max_retries + 1} attempts: {str(e)}",
                        bucket=bucket,
                        key=key,
                        operation="upload_code_with_retry"
                    ) from e
                
                # Retry for general exceptions
                delay = (base_delay ** attempt) + (random.random() * 0.1)
                logger.warning(f"Code upload attempt {attempt + 1} failed: {str(e)}. Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)
    
    async def _download_code_with_retry(self, bucket: str, key: str) -> str:
        """
        Download code with retry logic and exponential backoff.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            
        Returns:
            Code content as string
            
        Raises:
            AWSS3Error: If download fails after all retries
        """
        max_retries = self.config.max_retries
        base_delay = self.config.retry_backoff_base
        
        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                # Perform the download
                response = self.s3_client.get_object(
                    Bucket=bucket,
                    Key=key
                )
                
                # Read and decode content
                code_bytes = response['Body'].read()
                code_content = code_bytes.decode('utf-8')
                
                # Download successful
                if attempt > 0:
                    logger.info(f"Code download succeeded on attempt {attempt + 1}")
                
                return code_content
                
            except ClientError as e:
                error_code = e.response['Error']['Code']
                error_message = e.response['Error']['Message']
                
                # Don't retry for NoSuchKey - it's a permanent error
                if error_code == 'NoSuchKey':
                    raise
                
                # Check if this is a retryable error
                if not self._is_retryable_error(error_code):
                    raise AWSNonRetryableError(
                        f"Non-retryable S3 error: {error_message}",
                        reason=error_code
                    ) from e
                
                # Check if we've exhausted retries
                if attempt >= max_retries:
                    raise AWSRetryableError(
                        f"Code download failed after {max_retries + 1} attempts: {error_message}",
                        retry_count=attempt,
                        max_retries=max_retries
                    ) from e
                
                # Calculate exponential backoff delay with jitter
                delay = (base_delay ** attempt) + (random.random() * 0.1)
                
                logger.warning(
                    f"Code download attempt {attempt + 1} failed ({error_code}): {error_message}. "
                    f"Retrying in {delay:.2f}s..."
                )
                
                await asyncio.sleep(delay)
                
            except UnicodeDecodeError as e:
                # Handle UTF-8 decoding errors
                raise AWSS3Error(
                    f"Failed to decode code as UTF-8: {str(e)}",
                    bucket=bucket,
                    key=key,
                    operation="download_code_with_retry"
                ) from e
                
            except Exception as e:
                # Handle non-ClientError exceptions
                if attempt >= max_retries:
                    raise AWSS3Error(
                        f"Code download failed after {max_retries + 1} attempts: {str(e)}",
                        bucket=bucket,
                        key=key,
                        operation="download_code_with_retry"
                    ) from e
                
                # Retry for general exceptions
                delay = (base_delay ** attempt) + (random.random() * 0.1)
                logger.warning(f"Code download attempt {attempt + 1} failed: {str(e)}. Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)
    
    def _is_retryable_error(self, error_code: str) -> bool:
        """
        Determine if an S3 error is retryable.
        
        Args:
            error_code: AWS error code
            
        Returns:
            True if error is retryable, False otherwise
        """
        # Non-retryable errors
        non_retryable_errors = {
            'NoSuchBucket',
            'AccessDenied',
            'InvalidAccessKeyId',
            'SignatureDoesNotMatch',
            'TokenRefreshRequired',
            'InvalidArgument',
            'MalformedPolicy',
            'InvalidBucketName',
            'BucketNotEmpty',
            'NoSuchKey'  # Don't retry for missing objects
        }
        
        # Retryable errors
        retryable_errors = {
            'RequestTimeout',
            'ServiceUnavailable',
            'SlowDown',
            'RequestTimeTooSkewed',
            'InternalError',
            'RequestLimitExceeded',
            'ThrottlingException',
            'ProvisionedThroughputExceededException'
        }
        
        if error_code in non_retryable_errors:
            return False
        elif error_code in retryable_errors:
            return True
        else:
            # Default to retryable for unknown errors
            logger.warning(f"Unknown error code '{error_code}', treating as retryable")
            return True
    
    def get_code_url(self, project_id: str, video_id: str, version: int = 1, 
                    scene_number: Optional[int] = None) -> str:
        """
        Generate S3 URL for code without downloading.
        
        Args:
            project_id: Project identifier
            video_id: Video identifier
            version: Code version
            scene_number: Scene number (None for main code)
            
        Returns:
            S3 URL for the code
        """
        metadata = CodeMetadata(
            video_id=video_id,
            project_id=project_id,
            version=version,
            scene_number=scene_number
        )
        s3_key = self._generate_s3_key(metadata)
        return f"s3://{self.config.code_bucket_name}/{s3_key}"
    
    def __repr__(self) -> str:
        """String representation of the code storage service."""
        return (
            f"S3CodeStorageService(bucket='{self.config.code_bucket_name}', "
            f"region='{self.config.region}', "
            f"encryption={self.config.enable_encryption})"
        )
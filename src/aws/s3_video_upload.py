"""
S3 Video Upload Service

Implements S3 video upload functionality with multipart support, progress tracking,
metadata attachment, server-side encryption, and retry logic with exponential backoff.
"""

import os
import sys
import time
import random
import asyncio
import threading
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
class VideoChunk:
    """Represents a video chunk to be uploaded to S3."""
    file_path: str
    project_id: str
    video_id: str
    scene_number: int
    version: int
    metadata: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        """Validate video chunk data."""
        if not os.path.exists(self.file_path):
            raise ValueError(f"Video file does not exist: {self.file_path}")
        
        if not self.project_id or not self.video_id:
            raise ValueError("project_id and video_id are required")
        
        if self.scene_number < 0 or self.version < 1:
            raise ValueError("scene_number must be >= 0 and version must be >= 1")


class ProgressPercentage:
    """
    Progress callback class for S3 uploads with thread-safe progress tracking.
    Based on Context7 Boto3 examples with enhanced functionality.
    """
    
    def __init__(self, filename: str, callback: Optional[Callable[[int, int, float], None]] = None):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()
        self._callback = callback
        self._start_time = time.time()
        
        logger.info(f"Starting upload progress tracking for {filename} ({self._size:,.0f} bytes)")
    
    def __call__(self, bytes_amount: int):
        """Called by boto3 during upload with bytes transferred."""
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            elapsed_time = time.time() - self._start_time
            
            # Calculate transfer rate
            if elapsed_time > 0:
                rate_mbps = (self._seen_so_far / (1024 * 1024)) / elapsed_time
            else:
                rate_mbps = 0
            
            # Update console output
            sys.stdout.write(
                f"\r{os.path.basename(self._filename)}: "
                f"{self._seen_so_far:,} / {self._size:,.0f} bytes "
                f"({percentage:.1f}%) - {rate_mbps:.2f} MB/s"
            )
            sys.stdout.flush()
            
            # Call custom callback if provided
            if self._callback:
                self._callback(int(self._seen_so_far), int(self._size), percentage)
            
            # Log progress at intervals
            if percentage > 0 and int(percentage) % 25 == 0:
                logger.debug(f"Upload progress: {percentage:.1f}% - {rate_mbps:.2f} MB/s")
    
    def complete(self):
        """Mark upload as complete."""
        elapsed_time = time.time() - self._start_time
        rate_mbps = (self._size / (1024 * 1024)) / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\n✅ Upload completed in {elapsed_time:.1f}s - Average: {rate_mbps:.2f} MB/s")
        logger.info(f"Upload completed: {self._filename} in {elapsed_time:.1f}s")


class S3VideoUploadService:
    """
    Service for uploading video files to S3 with multipart support, progress tracking,
    metadata attachment, encryption, and retry logic.
    """
    
    def __init__(self, config: AWSConfig, credentials_manager: AWSCredentialsManager):
        self.config = config
        self.credentials_manager = credentials_manager
        
        # Initialize S3 client and resource
        self.s3_client = credentials_manager.get_client('s3')
        self.s3_resource = credentials_manager.get_resource('s3')
        
        # Configure multipart upload settings based on Context7 best practices
        self.transfer_config = TransferConfig(
            multipart_threshold=config.multipart_threshold,
            max_concurrency=config.max_concurrent_uploads,
            multipart_chunksize=config.chunk_size,
            use_threads=True
        )
        
        logger.info(f"S3 Video Upload Service initialized - Bucket: {config.video_bucket_name}")
        logger.info(f"Transfer config - Threshold: {config.multipart_threshold:,} bytes, "
                   f"Concurrency: {config.max_concurrent_uploads}, "
                   f"Chunk size: {config.chunk_size:,} bytes")
    
    async def upload_video_chunks(self, chunks: List[VideoChunk], 
                                 progress_callback: Optional[Callable] = None) -> List[str]:
        """
        Upload multiple video chunks to S3 with multipart support and progress tracking.
        
        Args:
            chunks: List of VideoChunk objects to upload
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of S3 URLs for uploaded chunks (None for failed uploads)
            
        Raises:
            AWSS3Error: If upload fails after all retries
        """
        upload_results = []
        
        with aws_operation('upload_video_chunks', 's3', 
                          bucket=self.config.video_bucket_name, 
                          chunk_count=len(chunks)):
            
            for i, chunk in enumerate(chunks):
                try:
                    logger.info(f"Uploading chunk {i+1}/{len(chunks)}: {chunk.file_path}")
                    
                    s3_url = await self._upload_single_chunk(chunk, progress_callback)
                    upload_results.append(s3_url)
                    
                    logger.info(f"Successfully uploaded chunk {i+1}: {s3_url}")
                    
                except Exception as e:
                    logger.error(f"Failed to upload chunk {i+1} ({chunk.file_path}): {e}")
                    upload_results.append(None)
                    
                    # Continue with other chunks unless upload is required
                    if self.config.require_aws_upload:
                        raise AWSS3Error(
                            f"Required upload failed for chunk: {chunk.file_path}",
                            bucket=self.config.video_bucket_name,
                            operation="upload_video_chunks"
                        ) from e
        
        successful_uploads = [url for url in upload_results if url is not None]
        logger.info(f"Upload batch completed: {len(successful_uploads)}/{len(chunks)} successful")
        
        return upload_results
    
    async def _upload_single_chunk(self, chunk: VideoChunk, 
                                  progress_callback: Optional[Callable] = None) -> str:
        """
        Upload a single video chunk to S3 with retry logic.
        
        Args:
            chunk: VideoChunk to upload
            progress_callback: Optional progress callback
            
        Returns:
            S3 URL of uploaded chunk
            
        Raises:
            AWSS3Error: If upload fails after all retries
        """
        # Generate S3 key with organized naming convention
        s3_key = self._generate_s3_key(chunk)
        
        # Prepare metadata for upload
        extra_args = self._prepare_upload_metadata(chunk)
        
        # Create progress tracker
        progress_tracker = ProgressPercentage(
            chunk.file_path, 
            callback=progress_callback
        )
        
        # Upload with retry logic
        try:
            await self._upload_with_retry(
                file_path=chunk.file_path,
                bucket=self.config.video_bucket_name,
                key=s3_key,
                extra_args=extra_args,
                callback=progress_tracker
            )
            
            progress_tracker.complete()
            
            # Return S3 URL
            s3_url = f"s3://{self.config.video_bucket_name}/{s3_key}"
            return s3_url
            
        except Exception as e:
            logger.error(f"Upload failed for {chunk.file_path}: {e}")
            raise AWSS3Error(
                f"Failed to upload video chunk: {str(e)}",
                bucket=self.config.video_bucket_name,
                key=s3_key,
                operation="upload_single_chunk"
            ) from e
    
    def _generate_s3_key(self, chunk: VideoChunk) -> str:
        """
        Generate S3 key with organized naming convention.
        Format: videos/{project_id}/{video_id}/chunk_{scene_number:03d}_v{version}.mp4
        """
        if chunk.scene_number == 0:
            # Combined/full video
            filename = f"{chunk.video_id}_full_v{chunk.version}.mp4"
        else:
            # Individual scene chunk
            filename = f"chunk_{chunk.scene_number:03d}_v{chunk.version}.mp4"
        
        s3_key = f"videos/{chunk.project_id}/{chunk.video_id}/{filename}"
        return s3_key
    
    def _prepare_upload_metadata(self, chunk: VideoChunk) -> Dict[str, Any]:
        """
        Prepare metadata and extra arguments for S3 upload.
        
        Args:
            chunk: VideoChunk with metadata
            
        Returns:
            Dictionary of extra arguments for S3 upload
        """
        # Base metadata
        metadata = {
            'video_id': chunk.video_id,
            'scene_number': str(chunk.scene_number),
            'version': str(chunk.version),
            'project_id': chunk.project_id,
            'upload_timestamp': datetime.utcnow().isoformat(),
            'file_size': str(os.path.getsize(chunk.file_path))
        }
        
        # Add custom metadata if provided
        if chunk.metadata:
            metadata.update(chunk.metadata)
        
        # Prepare extra arguments
        extra_args = {
            'Metadata': metadata,
            'ContentType': 'video/mp4'
        }
        
        # Add server-side encryption
        if self.config.enable_encryption:
            if self.config.kms_key_id:
                # Use KMS encryption
                extra_args.update({
                    'ServerSideEncryption': 'aws:kms',
                    'SSEKMSKeyId': self.config.kms_key_id
                })
                logger.debug(f"Using KMS encryption with key: {self.config.kms_key_id}")
            else:
                # Use S3 managed encryption (SSE-S3)
                extra_args['ServerSideEncryption'] = 'AES256'
                logger.debug("Using S3 managed encryption (SSE-S3)")
        
        return extra_args
    
    async def _upload_with_retry(self, file_path: str, bucket: str, key: str,
                               extra_args: Dict[str, Any], callback=None) -> None:
        """
        Upload file with retry logic and exponential backoff.
        
        Args:
            file_path: Local file path to upload
            bucket: S3 bucket name
            key: S3 object key
            extra_args: Extra arguments for upload
            callback: Progress callback
            
        Raises:
            AWSS3Error: If upload fails after all retries
        """
        max_retries = self.config.max_retries
        base_delay = self.config.retry_backoff_base
        
        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                # Perform the upload
                self.s3_client.upload_file(
                    file_path, bucket, key,
                    ExtraArgs=extra_args,
                    Config=self.transfer_config,
                    Callback=callback
                )
                
                # Upload successful
                if attempt > 0:
                    logger.info(f"Upload succeeded on attempt {attempt + 1}")
                
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
                        f"Upload failed after {max_retries + 1} attempts: {error_message}",
                        retry_count=attempt,
                        max_retries=max_retries
                    ) from e
                
                # Calculate exponential backoff delay with jitter
                delay = (base_delay ** attempt) + (random.random() * 0.1)
                
                logger.warning(
                    f"Upload attempt {attempt + 1} failed ({error_code}): {error_message}. "
                    f"Retrying in {delay:.2f}s..."
                )
                
                await asyncio.sleep(delay)
                
            except Exception as e:
                # Handle non-ClientError exceptions
                if attempt >= max_retries:
                    raise AWSS3Error(
                        f"Upload failed after {max_retries + 1} attempts: {str(e)}",
                        bucket=bucket,
                        key=key,
                        operation="upload_with_retry"
                    ) from e
                
                # Retry for general exceptions
                delay = (base_delay ** attempt) + (random.random() * 0.1)
                logger.warning(f"Upload attempt {attempt + 1} failed: {str(e)}. Retrying in {delay:.2f}s...")
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
            'BucketNotEmpty'
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
    
    async def upload_combined_video(self, file_path: str, project_id: str, 
                                   video_id: str, version: int = 1,
                                   metadata: Optional[Dict[str, str]] = None,
                                   progress_callback: Optional[Callable] = None) -> str:
        """
        Upload a combined/full video file to S3.
        
        Args:
            file_path: Path to the combined video file
            project_id: Project identifier
            video_id: Video identifier
            version: Video version number
            metadata: Additional metadata
            progress_callback: Optional progress callback
            
        Returns:
            S3 URL of uploaded video
            
        Raises:
            AWSS3Error: If upload fails
        """
        # Create a VideoChunk for the combined video (scene_number = 0)
        combined_chunk = VideoChunk(
            file_path=file_path,
            project_id=project_id,
            video_id=video_id,
            scene_number=0,  # 0 indicates combined/full video
            version=version,
            metadata=metadata
        )
        
        logger.info(f"Uploading combined video: {file_path}")
        
        with aws_operation('upload_combined_video', 's3',
                          bucket=self.config.video_bucket_name,
                          video_id=video_id):
            
            s3_url = await self._upload_single_chunk(combined_chunk, progress_callback)
            logger.info(f"Successfully uploaded combined video: {s3_url}")
            
            return s3_url
    
    def get_video_url(self, project_id: str, video_id: str, scene_number: int = 0, 
                     version: int = 1) -> str:
        """
        Generate S3 URL for a video without uploading.
        
        Args:
            project_id: Project identifier
            video_id: Video identifier
            scene_number: Scene number (0 for combined video)
            version: Video version
            
        Returns:
            S3 URL for the video
        """
        # Generate S3 key directly without creating VideoChunk
        if scene_number == 0:
            # Combined/full video
            filename = f"{video_id}_full_v{version}.mp4"
        else:
            # Individual scene chunk
            filename = f"chunk_{scene_number:03d}_v{version}.mp4"
        
        s3_key = f"videos/{project_id}/{video_id}/{filename}"
        return f"s3://{self.config.video_bucket_name}/{s3_key}"
    
    def __repr__(self) -> str:
        """String representation of the upload service."""
        return (
            f"S3VideoUploadService(bucket='{self.config.video_bucket_name}', "
            f"region='{self.config.region}', "
            f"encryption={self.config.enable_encryption})"
        )
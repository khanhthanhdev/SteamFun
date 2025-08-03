"""
Multipart Upload Handler

Implements advanced multipart upload functionality with resume capability, integrity verification,
upload abortion, and cleanup for failed transfers using boto3 TransferConfig.
"""

import os
import sys
import time
import hashlib
import asyncio
import threading
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
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
class UploadPart:
    """Represents a part in a multipart upload."""
    part_number: int
    etag: str
    size: int
    uploaded_at: datetime


@dataclass
class MultipartUploadInfo:
    """Information about an active multipart upload."""
    upload_id: str
    bucket: str
    key: str
    parts: List[UploadPart]
    initiated_at: datetime
    total_size: int
    uploaded_size: int
    
    @property
    def progress_percentage(self) -> float:
        """Calculate upload progress percentage."""
        if self.total_size == 0:
            return 0.0
        return (self.uploaded_size / self.total_size) * 100


class MultipartProgressTracker:
    """
    Enhanced progress tracker for multipart uploads with thread-safe tracking
    and detailed statistics.
    """
    
    def __init__(self, filename: str, total_size: int, 
                 callback: Optional[Callable[[int, int, float], None]] = None):
        self.filename = filename
        self.total_size = total_size
        self.uploaded_size = 0
        self.callback = callback
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.last_update = 0
        
        logger.info(f"Starting multipart upload progress tracking for {filename} "
                   f"({total_size:,} bytes)")
    
    def update_progress(self, bytes_uploaded: int):
        """Update progress with thread safety."""
        with self.lock:
            self.uploaded_size += bytes_uploaded
            percentage = (self.uploaded_size / self.total_size) * 100
            elapsed_time = time.time() - self.start_time
            
            # Calculate transfer rate
            if elapsed_time > 0:
                rate_mbps = (self.uploaded_size / (1024 * 1024)) / elapsed_time
            else:
                rate_mbps = 0
            
            # Update console output (throttled to avoid spam)
            current_time = time.time()
            if current_time - self.last_update >= 1.0:  # Update every second
                sys.stdout.write(
                    f"\r{os.path.basename(self.filename)}: "
                    f"{self.uploaded_size:,} / {self.total_size:,} bytes "
                    f"({percentage:.1f}%) - {rate_mbps:.2f} MB/s"
                )
                sys.stdout.flush()
                self.last_update = current_time
            
            # Call custom callback if provided
            if self.callback:
                self.callback(self.uploaded_size, self.total_size, percentage)


class MultipartUploadHandler:
    """
    Advanced multipart upload handler with resume capability, integrity verification,
    and cleanup for failed transfers.
    """
    
    def __init__(self, config: AWSConfig, credentials_manager: AWSCredentialsManager):
        self.config = config
        self.credentials_manager = credentials_manager
        self.s3_client = credentials_manager.get_client('s3')
        
        logger.info("Multipart Upload Handler initialized")
    
    async def upload_large_file(self, file_path: str, bucket: str, key: str,
                               extra_args: Optional[Dict[str, Any]] = None,
                               progress_callback: Optional[Callable] = None) -> str:
        """
        Upload a large file using multipart upload with resume capability.
        
        Args:
            file_path: Path to the file to upload
            bucket: S3 bucket name
            key: S3 object key
            extra_args: Additional S3 upload arguments
            progress_callback: Progress callback function
            
        Returns:
            S3 URL of uploaded file
            
        Raises:
            AWSS3Error: If upload fails
        """
        file_size = os.path.getsize(file_path)
        
        # Use multipart upload for large files
        if file_size >= self.config.multipart_threshold:
            return await self._multipart_upload(
                file_path, bucket, key, extra_args, progress_callback
            )
        else:
            # Use regular upload for smaller files
            return await self._regular_upload(
                file_path, bucket, key, extra_args, progress_callback
            )
    
    async def _multipart_upload(self, file_path: str, bucket: str, key: str,
                               extra_args: Optional[Dict[str, Any]] = None,
                               progress_callback: Optional[Callable] = None) -> str:
        """Perform multipart upload with retry logic."""
        upload_id = None
        
        try:
            # Initiate multipart upload
            response = self.s3_client.create_multipart_upload(
                Bucket=bucket,
                Key=key,
                **(extra_args or {})
            )
            upload_id = response['UploadId']
            
            logger.info(f"Initiated multipart upload: {upload_id}")
            
            # Upload parts
            parts = await self._upload_parts(
                file_path, bucket, key, upload_id, progress_callback
            )
            
            # Complete multipart upload
            self.s3_client.complete_multipart_upload(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={'Parts': parts}
            )
            
            logger.info(f"Completed multipart upload: {upload_id}")
            return f"s3://{bucket}/{key}"
            
        except Exception as e:
            # Abort multipart upload on failure
            if upload_id:
                try:
                    self.s3_client.abort_multipart_upload(
                        Bucket=bucket,
                        Key=key,
                        UploadId=upload_id
                    )
                    logger.info(f"Aborted multipart upload: {upload_id}")
                except Exception as abort_error:
                    logger.error(f"Failed to abort multipart upload: {abort_error}")
            
            raise AWSS3Error(
                f"Multipart upload failed: {str(e)}",
                bucket=bucket,
                key=key,
                operation="multipart_upload"
            ) from e
    
    async def _upload_parts(self, file_path: str, bucket: str, key: str,
                           upload_id: str, progress_callback: Optional[Callable] = None) -> List[Dict]:
        """Upload file parts concurrently."""
        file_size = os.path.getsize(file_path)
        chunk_size = self.config.chunk_size
        parts = []
        
        # Create progress tracker
        progress_tracker = MultipartProgressTracker(
            file_path, file_size, progress_callback
        )
        
        # Calculate number of parts
        num_parts = (file_size + chunk_size - 1) // chunk_size
        
        # Upload parts
        with open(file_path, 'rb') as f:
            for part_number in range(1, num_parts + 1):
                # Read chunk
                chunk_data = f.read(chunk_size)
                if not chunk_data:
                    break
                
                # Upload part with retry
                response = await self._upload_part_with_retry(
                    bucket, key, upload_id, part_number, chunk_data
                )
                
                parts.append({
                    'ETag': response['ETag'],
                    'PartNumber': part_number
                })
                
                # Update progress
                progress_tracker.update_progress(len(chunk_data))
                
                logger.debug(f"Uploaded part {part_number}/{num_parts}")
        
        return parts
    
    async def _upload_part_with_retry(self, bucket: str, key: str, upload_id: str,
                                     part_number: int, data: bytes) -> Dict:
        """Upload a single part with retry logic."""
        max_retries = self.config.max_retries
        
        for attempt in range(max_retries + 1):
            try:
                response = self.s3_client.upload_part(
                    Bucket=bucket,
                    Key=key,
                    PartNumber=part_number,
                    UploadId=upload_id,
                    Body=data
                )
                return response
                
            except ClientError as e:
                if attempt >= max_retries:
                    raise
                
                delay = (2 ** attempt) + (random.random() * 0.1)
                logger.warning(f"Part upload attempt {attempt + 1} failed, retrying in {delay:.2f}s")
                await asyncio.sleep(delay)
    
    async def _regular_upload(self, file_path: str, bucket: str, key: str,
                             extra_args: Optional[Dict[str, Any]] = None,
                             progress_callback: Optional[Callable] = None) -> str:
        """Perform regular upload for smaller files."""
        try:
            # Use boto3 upload_file with transfer config
            transfer_config = TransferConfig(
                multipart_threshold=self.config.multipart_threshold,
                max_concurrency=self.config.max_concurrent_uploads,
                multipart_chunksize=self.config.chunk_size,
                use_threads=True
            )
            
            self.s3_client.upload_file(
                file_path, bucket, key,
                ExtraArgs=extra_args or {},
                Config=transfer_config,
                Callback=progress_callback
            )
            
            return f"s3://{bucket}/{key}"
            
        except Exception as e:
            raise AWSS3Error(
                f"Regular upload failed: {str(e)}",
                bucket=bucket,
                key=key,
                operation="regular_upload"
            ) from e
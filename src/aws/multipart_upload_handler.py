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
    
    def complete(self):
        """Mark upload as complete and show final statistics."""
        elapsed_time = time.time() - self.start_time
        rate_mbps = (self.total_size / (1024 * 1024)) / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\n✅ Multipart upload completed in {elapsed_time:.1f}s - "
              f"Average: {rate_mbps:.2f} MB/s")
        logger.info(f"Multipart upload completed: {self.filename} in {elapsed_time:.1f}s")


class MultipartUploadHandler:
    """
    Handles large file uploads with multipart support, resume capability,
    integrity verification, and cleanup for failed transfers.
    """
    
    def __init__(self, config: AWSConfig, credentials_manager: AWSCredentialsManager):
        self.config = config
        self.credentials_manager = credentials_manager
        
        # Initialize S3 client
        self.s3_client = credentials_manager.get_client('s3')
        
        # Configure multipart upload settings based on Context7 best practices
        self.transfer_config = TransferConfig(
            multipart_threshold=config.multipart_threshold,
            max_concurrency=config.max_concurrent_uploads,
            multipart_chunksize=config.chunk_size,
            use_threads=True
        )
        
        # Minimum part size for multipart uploads (5MB as per AWS requirement)
        self.min_part_size = 5 * 1024 * 1024  # 5MB
        
        # Active uploads tracking
        self.active_uploads: Dict[str, MultipartUploadInfo] = {}
        
        logger.info(f"Multipart Upload Handler initialized - "
                   f"Threshold: {config.multipart_threshold:,} bytes, "
                   f"Chunk size: {config.chunk_size:,} bytes, "
                   f"Max concurrency: {config.max_concurrent_uploads}")
    
    async def upload_large_file(self, file_path: str, bucket: str, key: str,
                               extra_args: Optional[Dict[str, Any]] = None,
                               progress_callback: Optional[Callable] = None) -> str:
        """
        Upload large file using multipart upload with automatic detection.
        
        Args:
            file_path: Path to the file to upload
            bucket: S3 bucket name
            key: S3 object key
            extra_args: Additional arguments for S3 upload
            progress_callback: Optional progress callback
            
        Returns:
            S3 URL of uploaded file
            
        Raises:
            AWSS3Error: If upload fails after all retries
        """
        file_size = os.path.getsize(file_path)
        
        with aws_operation('upload_large_file', 's3', 
                          bucket=bucket, key=key, file_size=file_size):
            
            # Check if file qualifies for multipart upload
            if file_size >= self.config.multipart_threshold:
                logger.info(f"Using multipart upload for large file: {file_path} "
                           f"({file_size:,} bytes)")
                return await self._multipart_upload(
                    file_path, bucket, key, extra_args, progress_callback
                )
            else:
                logger.info(f"Using standard upload for file: {file_path} "
                           f"({file_size:,} bytes)")
                return await self._standard_upload(
                    file_path, bucket, key, extra_args, progress_callback
                )
    
    async def _multipart_upload(self, file_path: str, bucket: str, key: str,
                               extra_args: Optional[Dict[str, Any]] = None,
                               progress_callback: Optional[Callable] = None) -> str:
        """
        Perform multipart upload with resume capability.
        
        Args:
            file_path: Path to the file to upload
            bucket: S3 bucket name
            key: S3 object key
            extra_args: Additional arguments for S3 upload
            progress_callback: Optional progress callback
            
        Returns:
            S3 URL of uploaded file
        """
        file_size = os.path.getsize(file_path)
        upload_key = f"{bucket}/{key}"
        
        try:
            # Check for existing incomplete upload
            upload_info = await self._find_existing_upload(bucket, key, file_size)
            
            if upload_info:
                logger.info(f"Resuming existing multipart upload: {upload_info.upload_id}")
                return await self._resume_multipart_upload(
                    file_path, upload_info, extra_args, progress_callback
                )
            else:
                logger.info(f"Starting new multipart upload for: {file_path}")
                return await self._start_new_multipart_upload(
                    file_path, bucket, key, extra_args, progress_callback
                )
                
        except Exception as e:
            # Clean up any partial upload on failure
            if upload_key in self.active_uploads:
                await self._abort_multipart_upload(self.active_uploads[upload_key])
                del self.active_uploads[upload_key]
            
            raise AWSS3Error(
                f"Multipart upload failed: {str(e)}",
                bucket=bucket,
                key=key,
                operation="multipart_upload"
            ) from e
    
    async def _find_existing_upload(self, bucket: str, key: str, 
                                   expected_size: int) -> Optional[MultipartUploadInfo]:
        """
        Find existing incomplete multipart upload for the same file.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            expected_size: Expected file size
            
        Returns:
            MultipartUploadInfo if found, None otherwise
        """
        try:
            response = self.s3_client.list_multipart_uploads(
                Bucket=bucket,
                Prefix=key
            )
            
            uploads = response.get('Uploads', [])
            
            for upload in uploads:
                if upload['Key'] == key:
                    upload_id = upload['UploadId']
                    
                    # Get parts for this upload
                    parts_response = self.s3_client.list_parts(
                        Bucket=bucket,
                        Key=key,
                        UploadId=upload_id
                    )
                    
                    parts = []
                    uploaded_size = 0
                    
                    for part in parts_response.get('Parts', []):
                        upload_part = UploadPart(
                            part_number=part['PartNumber'],
                            etag=part['ETag'],
                            size=part['Size'],
                            uploaded_at=part['LastModified']
                        )
                        parts.append(upload_part)
                        uploaded_size += part['Size']
                    
                    upload_info = MultipartUploadInfo(
                        upload_id=upload_id,
                        bucket=bucket,
                        key=key,
                        parts=parts,
                        initiated_at=upload['Initiated'],
                        total_size=expected_size,
                        uploaded_size=uploaded_size
                    )
                    
                    logger.info(f"Found existing upload: {upload_id} "
                               f"({upload_info.progress_percentage:.1f}% complete)")
                    
                    return upload_info
            
            return None
            
        except ClientError as e:
            logger.warning(f"Could not check for existing uploads: {e}")
            return None  
  
    async def _start_new_multipart_upload(self, file_path: str, bucket: str, key: str,
                                         extra_args: Optional[Dict[str, Any]] = None,
                                         progress_callback: Optional[Callable] = None) -> str:
        """
        Start a new multipart upload.
        
        Args:
            file_path: Path to the file to upload
            bucket: S3 bucket name
            key: S3 object key
            extra_args: Additional arguments for S3 upload
            progress_callback: Optional progress callback
            
        Returns:
            S3 URL of uploaded file
        """
        file_size = os.path.getsize(file_path)
        
        # Prepare create_multipart_upload arguments
        create_args = {}
        if extra_args:
            # Map upload_file extra_args to create_multipart_upload format
            if 'Metadata' in extra_args:
                create_args['Metadata'] = extra_args['Metadata']
            if 'ContentType' in extra_args:
                create_args['ContentType'] = extra_args['ContentType']
            if 'ServerSideEncryption' in extra_args:
                create_args['ServerSideEncryption'] = extra_args['ServerSideEncryption']
            if 'SSEKMSKeyId' in extra_args:
                create_args['SSEKMSKeyId'] = extra_args['SSEKMSKeyId']
        
        # Initiate multipart upload
        response = self.s3_client.create_multipart_upload(
            Bucket=bucket,
            Key=key,
            **create_args
        )
        
        upload_id = response['UploadId']
        
        # Create upload info
        upload_info = MultipartUploadInfo(
            upload_id=upload_id,
            bucket=bucket,
            key=key,
            parts=[],
            initiated_at=datetime.utcnow(),
            total_size=file_size,
            uploaded_size=0
        )
        
        # Track active upload
        upload_key = f"{bucket}/{key}"
        self.active_uploads[upload_key] = upload_info
        
        logger.info(f"Started multipart upload: {upload_id}")
        
        try:
            # Upload parts
            await self._upload_parts(file_path, upload_info, progress_callback)
            
            # Complete multipart upload
            s3_url = await self._complete_multipart_upload(upload_info)
            
            # Remove from active uploads
            del self.active_uploads[upload_key]
            
            return s3_url
            
        except Exception as e:
            # Abort upload on failure
            await self._abort_multipart_upload(upload_info)
            if upload_key in self.active_uploads:
                del self.active_uploads[upload_key]
            raise
    
    async def _resume_multipart_upload(self, file_path: str, upload_info: MultipartUploadInfo,
                                      extra_args: Optional[Dict[str, Any]] = None,
                                      progress_callback: Optional[Callable] = None) -> str:
        """
        Resume an existing multipart upload.
        
        Args:
            file_path: Path to the file to upload
            upload_info: Information about the existing upload
            extra_args: Additional arguments (not used for resume)
            progress_callback: Optional progress callback
            
        Returns:
            S3 URL of uploaded file
        """
        upload_key = f"{upload_info.bucket}/{upload_info.key}"
        self.active_uploads[upload_key] = upload_info
        
        try:
            # Verify file integrity for existing parts
            if not await self._verify_existing_parts(file_path, upload_info):
                logger.warning("File integrity check failed, starting new upload")
                await self._abort_multipart_upload(upload_info)
                return await self._start_new_multipart_upload(
                    file_path, upload_info.bucket, upload_info.key, 
                    extra_args, progress_callback
                )
            
            # Upload remaining parts
            await self._upload_parts(file_path, upload_info, progress_callback)
            
            # Complete multipart upload
            s3_url = await self._complete_multipart_upload(upload_info)
            
            # Remove from active uploads
            del self.active_uploads[upload_key]
            
            return s3_url
            
        except Exception as e:
            # Abort upload on failure
            await self._abort_multipart_upload(upload_info)
            if upload_key in self.active_uploads:
                del self.active_uploads[upload_key]
            raise
    
    async def _upload_parts(self, file_path: str, upload_info: MultipartUploadInfo,
                           progress_callback: Optional[Callable] = None):
        """
        Upload file parts for multipart upload.
        
        Args:
            file_path: Path to the file to upload
            upload_info: Upload information
            progress_callback: Optional progress callback
        """
        # Calculate part size and number of parts
        part_size = self.config.chunk_size
        total_parts = (upload_info.total_size + part_size - 1) // part_size
        
        # Create progress tracker
        progress_tracker = MultipartProgressTracker(
            file_path, upload_info.total_size, progress_callback
        )
        
        # Set initial progress for resumed uploads
        progress_tracker.uploaded_size = upload_info.uploaded_size
        
        # Get already uploaded part numbers
        uploaded_parts = {part.part_number for part in upload_info.parts}
        
        logger.info(f"Uploading {total_parts} parts, "
                   f"{len(uploaded_parts)} already completed")
        
        # Upload parts concurrently
        semaphore = asyncio.Semaphore(self.config.max_concurrent_uploads)
        tasks = []
        
        with open(file_path, 'rb') as file:
            for part_number in range(1, total_parts + 1):
                if part_number in uploaded_parts:
                    continue  # Skip already uploaded parts
                
                # Calculate part boundaries
                start_byte = (part_number - 1) * part_size
                end_byte = min(start_byte + part_size, upload_info.total_size)
                part_data_size = end_byte - start_byte
                
                # Read part data
                file.seek(start_byte)
                part_data = file.read(part_data_size)
                
                # Create upload task
                task = self._upload_single_part(
                    upload_info, part_number, part_data, 
                    progress_tracker, semaphore
                )
                tasks.append(task)
        
        # Wait for all parts to upload
        if tasks:
            await asyncio.gather(*tasks)
        
        progress_tracker.complete()
        
        logger.info(f"All parts uploaded successfully for upload: {upload_info.upload_id}")
    
    async def _upload_single_part(self, upload_info: MultipartUploadInfo, 
                                 part_number: int, part_data: bytes,
                                 progress_tracker: MultipartProgressTracker,
                                 semaphore: asyncio.Semaphore):
        """
        Upload a single part with retry logic.
        
        Args:
            upload_info: Upload information
            part_number: Part number to upload
            part_data: Part data bytes
            progress_tracker: Progress tracker
            semaphore: Concurrency semaphore
        """
        async with semaphore:
            max_retries = self.config.max_retries
            base_delay = self.config.retry_backoff_base
            
            for attempt in range(max_retries + 1):
                try:
                    # Upload part
                    response = self.s3_client.upload_part(
                        Bucket=upload_info.bucket,
                        Key=upload_info.key,
                        PartNumber=part_number,
                        UploadId=upload_info.upload_id,
                        Body=part_data
                    )
                    
                    # Create part info
                    part = UploadPart(
                        part_number=part_number,
                        etag=response['ETag'],
                        size=len(part_data),
                        uploaded_at=datetime.utcnow()
                    )
                    
                    # Add to upload info
                    upload_info.parts.append(part)
                    upload_info.uploaded_size += len(part_data)
                    
                    # Update progress
                    progress_tracker.update_progress(len(part_data))
                    
                    logger.debug(f"Uploaded part {part_number} "
                               f"({len(part_data):,} bytes) - ETag: {response['ETag']}")
                    
                    return
                    
                except ClientError as e:
                    error_code = e.response['Error']['Code']
                    
                    if attempt >= max_retries:
                        raise AWSS3Error(
                            f"Failed to upload part {part_number} after {max_retries + 1} attempts: "
                            f"{e.response['Error']['Message']}",
                            bucket=upload_info.bucket,
                            key=upload_info.key,
                            operation="upload_part"
                        ) from e
                    
                    # Calculate exponential backoff delay
                    delay = (base_delay ** attempt) + (asyncio.get_event_loop().time() % 0.1)
                    
                    logger.warning(f"Part {part_number} upload attempt {attempt + 1} failed "
                                 f"({error_code}). Retrying in {delay:.2f}s...")
                    
                    await asyncio.sleep(delay)
    
    async def _complete_multipart_upload(self, upload_info: MultipartUploadInfo) -> str:
        """
        Complete the multipart upload.
        
        Args:
            upload_info: Upload information
            
        Returns:
            S3 URL of completed upload
        """
        # Sort parts by part number
        upload_info.parts.sort(key=lambda p: p.part_number)
        
        # Prepare parts list for completion
        parts = [
            {
                'ETag': part.etag,
                'PartNumber': part.part_number
            }
            for part in upload_info.parts
        ]
        
        # Complete multipart upload
        response = self.s3_client.complete_multipart_upload(
            Bucket=upload_info.bucket,
            Key=upload_info.key,
            UploadId=upload_info.upload_id,
            MultipartUpload={'Parts': parts}
        )
        
        s3_url = f"s3://{upload_info.bucket}/{upload_info.key}"
        
        # Verify upload integrity
        if await self._verify_upload_integrity(upload_info, response.get('ETag')):
            logger.info(f"Multipart upload completed successfully: {s3_url}")
        else:
            logger.warning(f"Upload integrity verification failed for: {s3_url}")
        
        return s3_url
    
    async def _abort_multipart_upload(self, upload_info: MultipartUploadInfo):
        """
        Abort a multipart upload and clean up.
        
        Args:
            upload_info: Upload information
        """
        try:
            self.s3_client.abort_multipart_upload(
                Bucket=upload_info.bucket,
                Key=upload_info.key,
                UploadId=upload_info.upload_id
            )
            
            logger.info(f"Aborted multipart upload: {upload_info.upload_id}")
            
        except ClientError as e:
            logger.error(f"Failed to abort multipart upload {upload_info.upload_id}: {e}")
    
    async def _verify_existing_parts(self, file_path: str, 
                                    upload_info: MultipartUploadInfo) -> bool:
        """
        Verify integrity of existing uploaded parts.
        
        Args:
            file_path: Path to the local file
            upload_info: Upload information
            
        Returns:
            True if all existing parts are valid, False otherwise
        """
        if not upload_info.parts:
            return True
        
        try:
            part_size = self.config.chunk_size
            
            with open(file_path, 'rb') as file:
                for part in upload_info.parts:
                    # Calculate part boundaries
                    start_byte = (part.part_number - 1) * part_size
                    
                    # Read part data
                    file.seek(start_byte)
                    part_data = file.read(part.size)
                    
                    # Calculate MD5 hash (ETag for single part)
                    md5_hash = hashlib.md5(part_data).hexdigest()
                    expected_etag = f'"{md5_hash}"'
                    
                    if part.etag != expected_etag:
                        logger.warning(f"Part {part.part_number} integrity check failed: "
                                     f"expected {expected_etag}, got {part.etag}")
                        return False
            
            logger.info(f"Verified {len(upload_info.parts)} existing parts")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying existing parts: {e}")
            return False
    
    async def _verify_upload_integrity(self, upload_info: MultipartUploadInfo, 
                                      final_etag: Optional[str]) -> bool:
        """
        Verify the integrity of the completed upload.
        
        Args:
            upload_info: Upload information
            final_etag: ETag returned from complete_multipart_upload
            
        Returns:
            True if integrity check passes, False otherwise
        """
        if not final_etag:
            logger.warning("No ETag provided for integrity verification")
            return False
        
        try:
            # For multipart uploads, ETag is not a simple MD5
            # Instead, verify by checking object exists and size matches
            response = self.s3_client.head_object(
                Bucket=upload_info.bucket,
                Key=upload_info.key
            )
            
            object_size = response['ContentLength']
            
            if object_size == upload_info.total_size:
                logger.info(f"Upload integrity verified: size matches ({object_size:,} bytes)")
                return True
            else:
                logger.error(f"Upload integrity failed: expected {upload_info.total_size:,} bytes, "
                           f"got {object_size:,} bytes")
                return False
                
        except ClientError as e:
            logger.error(f"Error verifying upload integrity: {e}")
            return False
    
    async def _standard_upload(self, file_path: str, bucket: str, key: str,
                              extra_args: Optional[Dict[str, Any]] = None,
                              progress_callback: Optional[Callable] = None) -> str:
        """
        Perform standard (non-multipart) upload for smaller files.
        
        Args:
            file_path: Path to the file to upload
            bucket: S3 bucket name
            key: S3 object key
            extra_args: Additional arguments for S3 upload
            progress_callback: Optional progress callback
            
        Returns:
            S3 URL of uploaded file
        """
        # Use boto3's high-level upload_file method
        self.s3_client.upload_file(
            file_path, bucket, key,
            ExtraArgs=extra_args or {},
            Config=self.transfer_config,
            Callback=progress_callback
        )
        
        s3_url = f"s3://{bucket}/{key}"
        logger.info(f"Standard upload completed: {s3_url}")
        
        return s3_url
    
    async def list_active_uploads(self, bucket: Optional[str] = None) -> List[MultipartUploadInfo]:
        """
        List all active multipart uploads.
        
        Args:
            bucket: Optional bucket name to filter by
            
        Returns:
            List of active upload information
        """
        active_uploads = []
        
        for upload_info in self.active_uploads.values():
            if bucket is None or upload_info.bucket == bucket:
                active_uploads.append(upload_info)
        
        return active_uploads
    
    async def cleanup_abandoned_uploads(self, bucket: str, max_age_hours: int = 24) -> int:
        """
        Clean up abandoned multipart uploads older than specified age.
        
        Args:
            bucket: S3 bucket name
            max_age_hours: Maximum age in hours for uploads to keep
            
        Returns:
            Number of uploads cleaned up
        """
        cleanup_count = 0
        cutoff_time = datetime.utcnow().timestamp() - (max_age_hours * 3600)
        
        try:
            response = self.s3_client.list_multipart_uploads(Bucket=bucket)
            uploads = response.get('Uploads', [])
            
            for upload in uploads:
                initiated_time = upload['Initiated'].timestamp()
                
                if initiated_time < cutoff_time:
                    try:
                        self.s3_client.abort_multipart_upload(
                            Bucket=bucket,
                            Key=upload['Key'],
                            UploadId=upload['UploadId']
                        )
                        
                        cleanup_count += 1
                        logger.info(f"Cleaned up abandoned upload: {upload['UploadId']}")
                        
                    except ClientError as e:
                        logger.error(f"Failed to cleanup upload {upload['UploadId']}: {e}")
            
            if cleanup_count > 0:
                logger.info(f"Cleaned up {cleanup_count} abandoned uploads from {bucket}")
            
            return cleanup_count
            
        except ClientError as e:
            logger.error(f"Error listing multipart uploads for cleanup: {e}")
            return 0
    
    def __repr__(self) -> str:
        """String representation of the multipart upload handler."""
        return (
            f"MultipartUploadHandler(threshold={self.config.multipart_threshold:,}, "
            f"chunk_size={self.config.chunk_size:,}, "
            f"max_concurrency={self.config.max_concurrent_uploads})"
        )
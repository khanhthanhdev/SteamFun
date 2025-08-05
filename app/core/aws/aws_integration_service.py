"""
AWS Integration Service

Central service for all AWS operations combining S3 video upload, S3 code storage,
and DynamoDB metadata management for LangGraph agents.
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime

from .config import AWSConfig
from .credentials import AWSCredentialsManager
from .s3_video_upload import S3VideoUploadService, VideoChunk
from .s3_code_storage import S3CodeStorageService, CodeMetadata, CodeVersion
from .metadata_service import MetadataService
from .cloudfront_service import CloudFrontService
from .mediaconvert_service import MediaConvertService, TranscodingJobConfig, TranscodingJobResult, OutputFormat, QualityLevel
from .exceptions import AWSIntegrationError, AWSS3Error, AWSMetadataError, AWSMediaConvertError

logger = logging.getLogger(__name__)


class AWSIntegrationService:
    """
    Central service for all AWS operations used by LangGraph agents.
    
    Provides a unified interface for:
    - S3 video upload with multipart support and progress tracking
    - S3 code storage with versioning
    - DynamoDB metadata management
    - CloudFront CDN integration for global content delivery
    - Integrated error handling and retry logic
    """
    
    def __init__(self, config: AWSConfig):
        """
        Initialize AWS Integration Service.
        
        Args:
            config: AWS configuration
            
        Raises:
            AWSIntegrationError: If initialization fails
        """
        self.config = config
        
        try:
            # Initialize credentials manager
            self.credentials_manager = AWSCredentialsManager(config)
            
            # Initialize individual services
            self.video_upload_service = S3VideoUploadService(config, self.credentials_manager)
            self.code_storage_service = S3CodeStorageService(config, self.credentials_manager)
            self.metadata_service = MetadataService(config)
            
            # Initialize CloudFront service if enabled
            if config.enable_cloudfront:
                self.cloudfront_service = CloudFrontService(config, self.credentials_manager)
                logger.info("CloudFront service initialized")
            else:
                self.cloudfront_service = None
                logger.info("CloudFront service disabled")
            
            # Initialize MediaConvert service if enabled
            if config.enable_transcoding:
                self.mediaconvert_service = MediaConvertService(config, self.credentials_manager)
                logger.info("MediaConvert service initialized")
            else:
                self.mediaconvert_service = None
                logger.info("MediaConvert service disabled")
            
            logger.info(f"AWS Integration Service initialized for region: {config.region}")
            logger.info(f"Video bucket: {config.video_bucket_name}")
            logger.info(f"Code bucket: {config.code_bucket_name}")
            logger.info(f"Metadata table: {config.metadata_table_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize AWS Integration Service: {e}")
            raise AWSIntegrationError(f"Service initialization failed: {str(e)}") from e
    
    # Video Upload Methods
    
    async def upload_video_chunks(self, chunks: List[VideoChunk], 
                                 progress_callback: Optional[Callable] = None) -> List[str]:
        """
        Upload video chunks to S3 with multipart support and progress tracking.
        
        Args:
            chunks: List of VideoChunk objects to upload
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of S3 URLs for uploaded chunks (None for failed uploads)
            
        Raises:
            AWSS3Error: If upload fails after all retries
        """
        try:
            return await self.video_upload_service.upload_video_chunks(chunks, progress_callback)
        except Exception as e:
            logger.error(f"Video chunk upload failed: {e}")
            raise
    
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
        try:
            return await self.video_upload_service.upload_combined_video(
                file_path, project_id, video_id, version, metadata, progress_callback
            )
        except Exception as e:
            logger.error(f"Combined video upload failed: {e}")
            raise
    
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
        return self.video_upload_service.get_video_url(project_id, video_id, scene_number, version)
    
    # Code Storage Methods
    
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
        try:
            return await self.code_storage_service.upload_code(code, metadata, enable_object_lock)
        except Exception as e:
            logger.error(f"Code upload failed: {e}")
            raise
    
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
        try:
            return await self.code_storage_service.download_code(metadata)
        except Exception as e:
            logger.error(f"Code download failed: {e}")
            raise
    
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
        try:
            return await self.code_storage_service.list_code_versions(project_id, video_id)
        except Exception as e:
            logger.error(f"Code version listing failed: {e}")
            raise
    
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
        return self.code_storage_service.get_code_url(project_id, video_id, version, scene_number)
    
    # Metadata Management Methods
    
    async def create_video_record(self, video_id: str, project_id: str, 
                                 metadata: Dict[str, Any]) -> bool:
        """
        Create a new video record in DynamoDB.
        
        Args:
            video_id: Video identifier
            project_id: Project identifier
            metadata: Video metadata
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            AWSMetadataError: If creation fails
        """
        try:
            return await self.metadata_service.create_video_record(
                video_id, project_id, metadata
            )
        except Exception as e:
            logger.error(f"Video record creation failed: {e}")
            raise
    
    async def update_video_metadata(self, video_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update video metadata in DynamoDB.
        
        Args:
            video_id: Video identifier
            metadata: Metadata to update
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            AWSMetadataError: If update fails
        """
        try:
            return await self.metadata_service.update_video_metadata(video_id, metadata)
        except Exception as e:
            logger.error(f"Video metadata update failed: {e}")
            raise
    
    async def get_video_metadata(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Get video metadata from DynamoDB.
        
        Args:
            video_id: Video identifier
            
        Returns:
            Video metadata or None if not found
            
        Raises:
            AWSMetadataError: If retrieval fails
        """
        try:
            return await self.metadata_service.get_video_metadata(video_id)
        except Exception as e:
            logger.error(f"Video metadata retrieval failed: {e}")
            raise
    
    async def list_project_videos(self, project_id: str) -> List[Dict[str, Any]]:
        """
        List all videos for a project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            List of video metadata
            
        Raises:
            AWSMetadataError: If listing fails
        """
        try:
            return await self.metadata_service.list_project_videos(project_id)
        except Exception as e:
            logger.error(f"Project video listing failed: {e}")
            raise
    
    # Integrated Workflow Methods
    
    async def upload_video_with_metadata(self, chunks: List[VideoChunk], 
                                        video_metadata: Dict[str, Any],
                                        progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Upload videos and update metadata in a single operation.
        
        Args:
            chunks: Video chunks to upload
            video_metadata: Metadata to update
            progress_callback: Optional progress callback
            
        Returns:
            Dictionary with upload results and metadata update status
            
        Raises:
            AWSIntegrationError: If operation fails
        """
        try:
            # Upload videos
            upload_results = await self.upload_video_chunks(chunks, progress_callback)
            
            # Filter successful uploads
            successful_uploads = {
                i: url for i, url in enumerate(upload_results) if url is not None
            }
            
            if not successful_uploads:
                raise AWSIntegrationError("No videos uploaded successfully")
            
            # Update metadata with S3 paths
            video_id = chunks[0].video_id if chunks else None
            if not video_id:
                raise AWSIntegrationError("No video_id available for metadata update")
            
            # Prepare metadata update
            s3_paths = {}
            for i, chunk in enumerate(chunks):
                if i in successful_uploads:
                    if chunk.scene_number == 0:
                        s3_paths['combined'] = upload_results[i]
                    else:
                        s3_paths[chunk.scene_number] = upload_results[i]
            
            metadata_update = {
                **video_metadata,
                's3_paths': s3_paths,
                'upload_completed_at': datetime.utcnow().isoformat(),
                'status': 'uploaded'
            }
            
            # Update metadata
            metadata_success = await self.update_video_metadata(video_id, metadata_update)
            
            return {
                'upload_results': upload_results,
                'successful_uploads': len(successful_uploads),
                'total_uploads': len(upload_results),
                'metadata_updated': metadata_success,
                's3_paths': s3_paths
            }
            
        except Exception as e:
            logger.error(f"Integrated video upload with metadata failed: {e}")
            raise AWSIntegrationError(f"Integrated upload failed: {str(e)}") from e
    
    # CloudFront CDN Methods
    
    async def create_cloudfront_distribution(self, video_bucket_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create CloudFront distribution with S3 origin and OAI.
        
        Args:
            video_bucket_name: Optional S3 bucket name (uses config if not provided)
            
        Returns:
            Dictionary with distribution information
            
        Raises:
            AWSIntegrationError: If CloudFront is not enabled or creation fails
        """
        if not self.cloudfront_service:
            raise AWSIntegrationError("CloudFront service is not enabled")
        
        bucket_name = video_bucket_name or self.config.video_bucket_name
        if not bucket_name:
            raise AWSIntegrationError("No video bucket name provided or configured")
        
        try:
            return await self.cloudfront_service.create_distribution(bucket_name)
        except Exception as e:
            logger.error(f"CloudFront distribution creation failed: {e}")
            raise
    
    async def invalidate_cloudfront_cache(self, paths: Union[str, List[str]], 
                                         distribution_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Invalidate CloudFront cache for updated videos.
        
        Args:
            paths: Single path or list of paths to invalidate
            distribution_id: Optional distribution ID
            
        Returns:
            Dictionary with invalidation information
            
        Raises:
            AWSIntegrationError: If CloudFront is not enabled or invalidation fails
        """
        if not self.cloudfront_service:
            raise AWSIntegrationError("CloudFront service is not enabled")
        
        try:
            return await self.cloudfront_service.invalidate_cache(paths, distribution_id)
        except Exception as e:
            logger.error(f"CloudFront cache invalidation failed: {e}")
            raise
    
    def get_cloudfront_video_url(self, s3_key: str, distribution_id: Optional[str] = None) -> str:
        """
        Generate CloudFront URL for video access.
        
        Args:
            s3_key: S3 object key for the video
            distribution_id: Optional distribution ID
            
        Returns:
            CloudFront URL for video access
            
        Raises:
            AWSIntegrationError: If CloudFront is not enabled or URL generation fails
        """
        if not self.cloudfront_service:
            raise AWSIntegrationError("CloudFront service is not enabled")
        
        try:
            return self.cloudfront_service.generate_video_url(s3_key, distribution_id)
        except Exception as e:
            logger.error(f"CloudFront URL generation failed: {e}")
            raise
    
    async def get_cloudfront_performance_metrics(self, distribution_id: Optional[str] = None,
                                                start_time: Optional[datetime] = None,
                                                end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get CloudFront performance metrics including cache hit rate.
        
        Args:
            distribution_id: Optional distribution ID
            start_time: Start time for metrics
            end_time: End time for metrics
            
        Returns:
            Dictionary with performance metrics
            
        Raises:
            AWSIntegrationError: If CloudFront is not enabled or metrics retrieval fails
        """
        if not self.cloudfront_service:
            raise AWSIntegrationError("CloudFront service is not enabled")
        
        try:
            return await self.cloudfront_service.get_performance_metrics(
                distribution_id, start_time, end_time
            )
        except Exception as e:
            logger.error(f"CloudFront performance metrics retrieval failed: {e}")
            raise
    
    async def optimize_cloudfront_cache(self, distribution_id: Optional[str] = None,
                                       hit_rate_threshold: float = 80.0) -> Dict[str, Any]:
        """
        Analyze and suggest CloudFront cache optimizations.
        
        Args:
            distribution_id: Optional distribution ID
            hit_rate_threshold: Minimum acceptable cache hit rate percentage
            
        Returns:
            Dictionary with optimization recommendations
            
        Raises:
            AWSIntegrationError: If CloudFront is not enabled or optimization fails
        """
        if not self.cloudfront_service:
            raise AWSIntegrationError("CloudFront service is not enabled")
        
        try:
            return await self.cloudfront_service.optimize_cache_behaviors(
                distribution_id, hit_rate_threshold
            )
        except Exception as e:
            logger.error(f"CloudFront cache optimization failed: {e}")
            raise
    
    # Enhanced Workflow Methods with CloudFront Integration
    
    async def upload_video_with_cdn(self, chunks: List[VideoChunk], 
                                   video_metadata: Dict[str, Any],
                                   progress_callback: Optional[Callable] = None,
                                   invalidate_cache: bool = True) -> Dict[str, Any]:
        """
        Upload videos, update metadata, and manage CloudFront CDN in a single operation.
        
        Args:
            chunks: Video chunks to upload
            video_metadata: Metadata to update
            progress_callback: Optional progress callback
            invalidate_cache: Whether to invalidate CloudFront cache
            
        Returns:
            Dictionary with upload results, metadata update status, and CDN URLs
            
        Raises:
            AWSIntegrationError: If operation fails
        """
        try:
            # Upload videos and update metadata
            upload_result = await self.upload_video_with_metadata(
                chunks, video_metadata, progress_callback
            )
            
            # Generate CloudFront URLs if service is enabled
            cloudfront_urls = {}
            cache_invalidation = None
            
            if self.cloudfront_service and upload_result['s3_paths']:
                try:
                    # Generate CloudFront URLs for successful uploads
                    for key, s3_path in upload_result['s3_paths'].items():
                        if s3_path:
                            # Extract S3 key from S3 URL
                            s3_key = s3_path.split('/', 3)[-1] if s3_path.startswith('s3://') else s3_path
                            cloudfront_url = self.get_cloudfront_video_url(s3_key)
                            cloudfront_urls[key] = cloudfront_url
                    
                    # Invalidate cache for updated videos if requested
                    if invalidate_cache and cloudfront_urls:
                        cache_paths = [
                            url.split('/', 3)[-1] for url in cloudfront_urls.values()
                        ]
                        cache_invalidation = await self.invalidate_cloudfront_cache(cache_paths)
                        logger.info(f"Invalidated CloudFront cache for {len(cache_paths)} paths")
                    
                except Exception as e:
                    logger.warning(f"CloudFront operations failed, continuing without CDN: {e}")
            
            # Combine results
            result = {
                **upload_result,
                'cloudfront_urls': cloudfront_urls,
                'cache_invalidation': cache_invalidation,
                'cdn_enabled': self.cloudfront_service is not None
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Integrated video upload with CDN failed: {e}")
            raise AWSIntegrationError(f"Integrated CDN upload failed: {str(e)}") from e
    
    # MediaConvert Transcoding Methods
    
    async def create_transcoding_job(self, input_s3_path: str, output_s3_prefix: str,
                                   video_id: str, project_id: str,
                                   output_formats: Optional[List[OutputFormat]] = None,
                                   quality_levels: Optional[List[QualityLevel]] = None,
                                   metadata: Optional[Dict[str, Any]] = None,
                                   progress_callback: Optional[Callable] = None) -> str:
        """
        Create a transcoding job for adaptive bitrate streaming.
        
        Args:
            input_s3_path: S3 path to input video file
            output_s3_prefix: S3 prefix for output files
            video_id: Video identifier
            project_id: Project identifier
            output_formats: List of output formats (HLS, DASH, MP4)
            quality_levels: List of quality levels (1080p, 720p, 480p)
            metadata: Additional metadata
            progress_callback: Optional progress callback
            
        Returns:
            MediaConvert job ID
            
        Raises:
            AWSIntegrationError: If MediaConvert is not enabled or job creation fails
        """
        if not self.mediaconvert_service:
            raise AWSIntegrationError("MediaConvert service is not enabled")
        
        try:
            job_config = TranscodingJobConfig(
                input_s3_path=input_s3_path,
                output_s3_prefix=output_s3_prefix,
                video_id=video_id,
                project_id=project_id,
                output_formats=output_formats,
                quality_levels=quality_levels,
                metadata=metadata
            )
            
            return await self.mediaconvert_service.create_transcoding_job(job_config, progress_callback)
            
        except Exception as e:
            logger.error(f"MediaConvert job creation failed: {e}")
            raise AWSIntegrationError(f"Transcoding job creation failed: {str(e)}") from e
    
    async def get_transcoding_job_status(self, job_id: str) -> TranscodingJobResult:
        """
        Get the status of a transcoding job.
        
        Args:
            job_id: MediaConvert job ID
            
        Returns:
            TranscodingJobResult with current job status
            
        Raises:
            AWSIntegrationError: If MediaConvert is not enabled or status retrieval fails
        """
        if not self.mediaconvert_service:
            raise AWSIntegrationError("MediaConvert service is not enabled")
        
        try:
            return await self.mediaconvert_service.get_job_status(job_id)
        except Exception as e:
            logger.error(f"MediaConvert job status retrieval failed: {e}")
            raise AWSIntegrationError(f"Job status retrieval failed: {str(e)}") from e
    
    async def wait_for_transcoding_completion(self, job_id: str, timeout_minutes: int = 60) -> TranscodingJobResult:
        """
        Wait for a transcoding job to complete.
        
        Args:
            job_id: MediaConvert job ID
            timeout_minutes: Maximum time to wait in minutes
            
        Returns:
            TranscodingJobResult with final job status
            
        Raises:
            AWSIntegrationError: If MediaConvert is not enabled or job fails/times out
        """
        if not self.mediaconvert_service:
            raise AWSIntegrationError("MediaConvert service is not enabled")
        
        try:
            return await self.mediaconvert_service.wait_for_job_completion(job_id, timeout_minutes)
        except Exception as e:
            logger.error(f"MediaConvert job completion wait failed: {e}")
            raise AWSIntegrationError(f"Job completion wait failed: {str(e)}") from e
    
    async def cancel_transcoding_job(self, job_id: str) -> bool:
        """
        Cancel a transcoding job.
        
        Args:
            job_id: MediaConvert job ID
            
        Returns:
            True if job was canceled successfully
            
        Raises:
            AWSIntegrationError: If MediaConvert is not enabled or cancellation fails
        """
        if not self.mediaconvert_service:
            raise AWSIntegrationError("MediaConvert service is not enabled")
        
        try:
            return await self.mediaconvert_service.cancel_job(job_id)
        except Exception as e:
            logger.error(f"MediaConvert job cancellation failed: {e}")
            raise AWSIntegrationError(f"Job cancellation failed: {str(e)}") from e
    
    async def list_transcoding_jobs(self, status: Optional[str] = None, 
                                  max_results: int = 20) -> List[TranscodingJobResult]:
        """
        List transcoding jobs.
        
        Args:
            status: Optional status filter
            max_results: Maximum number of results
            
        Returns:
            List of TranscodingJobResult objects
            
        Raises:
            AWSIntegrationError: If MediaConvert is not enabled or listing fails
        """
        if not self.mediaconvert_service:
            raise AWSIntegrationError("MediaConvert service is not enabled")
        
        try:
            from .mediaconvert_service import TranscodingStatus
            status_enum = TranscodingStatus(status) if status else None
            return await self.mediaconvert_service.list_jobs(status_enum, max_results)
        except Exception as e:
            logger.error(f"MediaConvert job listing failed: {e}")
            raise AWSIntegrationError(f"Job listing failed: {str(e)}") from e
    
    # Enhanced Workflow Methods with Transcoding
    
    async def upload_video_with_transcoding(self, chunks: List[VideoChunk], 
                                          video_metadata: Dict[str, Any],
                                          output_formats: Optional[List[OutputFormat]] = None,
                                          quality_levels: Optional[List[QualityLevel]] = None,
                                          progress_callback: Optional[Callable] = None,
                                          wait_for_completion: bool = False) -> Dict[str, Any]:
        """
        Upload videos, update metadata, and trigger transcoding in a single operation.
        
        Args:
            chunks: Video chunks to upload
            video_metadata: Metadata to update
            output_formats: List of output formats for transcoding
            quality_levels: List of quality levels for transcoding
            progress_callback: Optional progress callback
            wait_for_completion: Whether to wait for transcoding to complete
            
        Returns:
            Dictionary with upload results, metadata update status, and transcoding job info
            
        Raises:
            AWSIntegrationError: If operation fails
        """
        try:
            # Upload videos and update metadata
            upload_result = await self.upload_video_with_metadata(
                chunks, video_metadata, progress_callback
            )
            
            transcoding_job_id = None
            transcoding_result = None
            
            # Start transcoding if service is enabled and we have successful uploads
            if self.mediaconvert_service and upload_result['s3_paths']:
                try:
                    # Use the combined video for transcoding if available, otherwise use first chunk
                    input_s3_path = None
                    if 'combined' in upload_result['s3_paths']:
                        input_s3_path = upload_result['s3_paths']['combined']
                    elif upload_result['s3_paths']:
                        input_s3_path = list(upload_result['s3_paths'].values())[0]
                    
                    if input_s3_path:
                        video_id = chunks[0].video_id if chunks else None
                        project_id = chunks[0].project_id if chunks else None
                        
                        if video_id and project_id:
                            # Create output prefix
                            output_s3_prefix = f"s3://{self.config.video_bucket_name}/transcoded/{project_id}/{video_id}"
                            
                            # Start transcoding job
                            transcoding_job_id = await self.create_transcoding_job(
                                input_s3_path=input_s3_path,
                                output_s3_prefix=output_s3_prefix,
                                video_id=video_id,
                                project_id=project_id,
                                output_formats=output_formats,
                                quality_levels=quality_levels,
                                metadata=video_metadata,
                                progress_callback=progress_callback
                            )
                            
                            logger.info(f"Started transcoding job {transcoding_job_id} for video {video_id}")
                            
                            # Wait for completion if requested
                            if wait_for_completion:
                                transcoding_result = await self.wait_for_transcoding_completion(transcoding_job_id)
                                logger.info(f"Transcoding job {transcoding_job_id} completed")
                    
                except Exception as e:
                    logger.warning(f"Transcoding failed, continuing without transcoding: {e}")
            
            # Combine results
            result = {
                **upload_result,
                'transcoding_job_id': transcoding_job_id,
                'transcoding_result': transcoding_result,
                'transcoding_enabled': self.mediaconvert_service is not None
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Integrated video upload with transcoding failed: {e}")
            raise AWSIntegrationError(f"Integrated transcoding upload failed: {str(e)}") from e
    
    # Health Check and Status Methods
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all AWS services.
        
        Returns:
            Dictionary with health status of each service
        """
        health_status = {
            'overall_status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'services': {}
        }
        
        # Check S3 video bucket
        try:
            # Simple head bucket operation
            self.video_upload_service.s3_client.head_bucket(Bucket=self.config.video_bucket_name)
            health_status['services']['s3_video'] = {'status': 'healthy', 'bucket': self.config.video_bucket_name}
        except Exception as e:
            health_status['services']['s3_video'] = {'status': 'unhealthy', 'error': str(e)}
            health_status['overall_status'] = 'degraded'
        
        # Check S3 code bucket
        try:
            self.code_storage_service.s3_client.head_bucket(Bucket=self.config.code_bucket_name)
            health_status['services']['s3_code'] = {'status': 'healthy', 'bucket': self.config.code_bucket_name}
        except Exception as e:
            health_status['services']['s3_code'] = {'status': 'unhealthy', 'error': str(e)}
            health_status['overall_status'] = 'degraded'
        
        # Check DynamoDB table
        try:
            table_status = self.metadata_service.table.table_status
            health_status['services']['dynamodb'] = {
                'status': 'healthy' if table_status == 'ACTIVE' else 'degraded',
                'table': self.config.metadata_table_name,
                'table_status': table_status
            }
        except Exception as e:
            health_status['services']['dynamodb'] = {'status': 'unhealthy', 'error': str(e)}
            health_status['overall_status'] = 'degraded'
        
        # Check CloudFront distribution if enabled
        if self.cloudfront_service:
            try:
                cloudfront_health = await self.cloudfront_service.health_check()
                health_status['services']['cloudfront'] = {
                    'status': cloudfront_health['status'],
                    'distribution_id': self.config.cloudfront_distribution_id,
                    'domain': self.config.cloudfront_domain
                }
                
                if cloudfront_health['status'] != 'healthy':
                    health_status['overall_status'] = 'degraded'
                    
            except Exception as e:
                health_status['services']['cloudfront'] = {'status': 'unhealthy', 'error': str(e)}
                health_status['overall_status'] = 'degraded'
        else:
            health_status['services']['cloudfront'] = {'status': 'disabled'}
        
        # Check MediaConvert service if enabled
        if self.mediaconvert_service:
            try:
                mediaconvert_health = await self.mediaconvert_service.health_check()
                health_status['services']['mediaconvert'] = {
                    'status': mediaconvert_health['status'],
                    'endpoint': self.config.mediaconvert_endpoint,
                    'role_arn': self.config.mediaconvert_role_arn
                }
                
                if mediaconvert_health['status'] != 'healthy':
                    health_status['overall_status'] = 'degraded'
                    
            except Exception as e:
                health_status['services']['mediaconvert'] = {'status': 'unhealthy', 'error': str(e)}
                health_status['overall_status'] = 'degraded'
        else:
            health_status['services']['mediaconvert'] = {'status': 'disabled'}
        
        return health_status
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about the AWS integration service.
        
        Returns:
            Service information dictionary
        """
        return {
            'service_name': 'AWS Integration Service',
            'version': '1.0.0',
            'region': self.config.region,
            'video_bucket': self.config.video_bucket_name,
            'code_bucket': self.config.code_bucket_name,
            'metadata_table': self.config.metadata_table_name,
            'cloudfront_distribution_id': self.config.cloudfront_distribution_id,
            'cloudfront_domain': self.config.cloudfront_domain,
            'cloudfront_enabled': self.config.enable_cloudfront,
            'transcoding_enabled': self.config.enable_transcoding,
            'mediaconvert_endpoint': self.config.mediaconvert_endpoint,
            'mediaconvert_role_arn': self.config.mediaconvert_role_arn,
            'encryption_enabled': self.config.enable_encryption,
            'multipart_threshold': self.config.multipart_threshold,
            'max_concurrent_uploads': self.config.max_concurrent_uploads,
            'max_retries': self.config.max_retries
        }
    
    def __repr__(self) -> str:
        """String representation of the integration service."""
        return (
            f"AWSIntegrationService(region='{self.config.region}', "
            f"video_bucket='{self.config.video_bucket_name}', "
            f"code_bucket='{self.config.code_bucket_name}', "
            f"metadata_table='{self.config.metadata_table_name}')"
        )
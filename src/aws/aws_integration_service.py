"""
AWS Integration Service

Central service for all AWS operations combining S3 video upload, S3 code storage,
and DynamoDB metadata management for LangGraph agents.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

from .config import AWSConfig
from .credentials import AWSCredentialsManager
from .s3_video_upload import S3VideoUploadService, VideoChunk
from .s3_code_storage import S3CodeStorageService, CodeMetadata, CodeVersion
from .metadata_service import MetadataService
from .exceptions import AWSIntegrationError, AWSS3Error, AWSMetadataError

logger = logging.getLogger(__name__)


class AWSIntegrationService:
    """
    Central service for all AWS operations used by LangGraph agents.
    
    Provides a unified interface for:
    - S3 video upload with multipart support and progress tracking
    - S3 code storage with versioning
    - DynamoDB metadata management
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
            return await self.metadata_service.operations.create_video_record(
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
            return await self.metadata_service.operations.update_video_metadata(video_id, metadata)
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
            return await self.metadata_service.operations.get_video_metadata(video_id)
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
            return await self.metadata_service.operations.list_project_videos(project_id)
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
    
    async def upload_code_with_metadata(self, code_data: Dict[str, str], 
                                       video_id: str, project_id: str, 
                                       version: int, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Upload code and update metadata in a single operation.
        
        Args:
            code_data: Dictionary of scene_number -> code content
            video_id: Video identifier
            project_id: Project identifier
            version: Code version
            metadata: Additional metadata
            
        Returns:
            Dictionary with upload results and metadata update status
            
        Raises:
            AWSIntegrationError: If operation fails
        """
        try:
            upload_results = {}
            
            # Upload each code file
            for scene_key, code_content in code_data.items():
                scene_number = scene_key if isinstance(scene_key, int) else None
                
                code_metadata = CodeMetadata(
                    video_id=video_id,
                    project_id=project_id,
                    version=version,
                    scene_number=scene_number,
                    created_at=datetime.utcnow()
                )
                
                try:
                    s3_url = await self.upload_code(code_content, code_metadata)
                    upload_results[scene_key] = s3_url
                except Exception as e:
                    logger.error(f"Failed to upload code for scene {scene_key}: {e}")
                    upload_results[scene_key] = None
            
            # Filter successful uploads
            successful_uploads = {k: v for k, v in upload_results.items() if v is not None}
            
            if not successful_uploads:
                raise AWSIntegrationError("No code files uploaded successfully")
            
            # Update metadata with code paths
            metadata_update = {
                **metadata,
                'code_s3_paths': successful_uploads,
                'current_version_id': f'v{version}',
                'code_upload_completed_at': datetime.utcnow().isoformat(),
                'status': 'code_ready'
            }
            
            # Set main code path
            if 'main' in successful_uploads:
                metadata_update['s3_path_code'] = successful_uploads['main']
            elif successful_uploads:
                metadata_update['s3_path_code'] = list(successful_uploads.values())[0]
            
            # Update metadata
            metadata_success = await self.update_video_metadata(video_id, metadata_update)
            
            return {
                'upload_results': upload_results,
                'successful_uploads': len(successful_uploads),
                'total_uploads': len(upload_results),
                'metadata_updated': metadata_success,
                'code_s3_paths': successful_uploads,
                'version': version
            }
            
        except Exception as e:
            logger.error(f"Integrated code upload with metadata failed: {e}")
            raise AWSIntegrationError(f"Integrated code upload failed: {str(e)}") from e
    
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
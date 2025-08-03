"""
AWS Service Layer

Service layer for AWS operations providing a clean interface between
the API layer and the AWS core integration.
"""

import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta

from app.core.aws import (
    AWSConfig,
    AWSIntegrationService,
    VideoChunk,
    CodeMetadata,
    CodeVersion,
    AWSIntegrationError,
    AWSS3Error,
    AWSMetadataError
)
from app.models.schemas.aws import (
    S3UploadRequest,
    S3UploadResponse,
    S3DownloadRequest,
    S3DownloadResponse,
    VideoUploadRequest,
    VideoUploadResponse,
    CodeUploadRequest,
    CodeUploadResponse,
    CodeDownloadRequest,
    CodeDownloadResponse,
    VideoProjectRequest,
    VideoProjectResponse,
    VideoMetadataUpdateRequest,
    VideoDetailsResponse,
    ProjectVideosResponse,
    CodeVersionInfo,
    CodeVersionsResponse,
    BatchUploadRequest,
    BatchUploadResponse,
    IntegratedUploadRequest,
    IntegratedUploadResponse,
    DynamoDBOperationRequest,
    DynamoDBOperationResponse,
    AWSHealthResponse,
    AWSConfigResponse,
    S3ObjectInfo,
    S3ListResponse,
    S3DeleteRequest,
    S3DeleteResponse
)
from app.models.enums import AWSServiceType
from app.utils.exceptions import ServiceError, ValidationError
from app.utils.logging import get_logger

logger = get_logger(__name__)


class AWSService:
    """
    Service layer for AWS operations providing business logic and orchestration
    for video uploads, code storage, and metadata management.
    """
    
    def __init__(self, config: Optional[AWSConfig] = None):
        """
        Initialize AWS service.
        
        Args:
            config: Optional AWS configuration, defaults to environment config
        """
        self.config = config or AWSConfig.from_env()
        try:
            self.integration_service = AWSIntegrationService(self.config)
            logger.info("AWS Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AWS integration service: {str(e)}")
            self.integration_service = None
    
    # Video Operations
    
    async def upload_s3_file(self, request: S3UploadRequest) -> S3UploadResponse:
        """
        Upload a file to S3.
        
        Args:
            request: S3 upload request
            
        Returns:
            S3UploadResponse: Upload result
        """
        try:
            if not self.integration_service:
                raise ServiceError("AWS integration service not available")
            
            # Create VideoChunk for compatibility with existing integration
            chunk = VideoChunk(
                file_path=request.file_path,
                project_id=request.project_id,
                video_id=request.video_id,
                scene_number=request.scene_number or 0,
                version=request.version,
                metadata=request.metadata
            )
            
            # Upload file
            s3_url = await self.integration_service.upload_video_chunks([chunk])
            if not s3_url or not s3_url[0]:
                raise ServiceError("Failed to upload file to S3")
            
            # Get file info
            import os
            file_size = os.path.getsize(request.file_path) if os.path.exists(request.file_path) else None
            
            return S3UploadResponse(
                s3_url=s3_url[0],
                bucket=self.config.s3_bucket,
                key=f"{request.project_id}/{request.video_id}/scene{request.scene_number or 0}_v{request.version}",
                file_size=file_size,
                content_type=request.content_type,
                upload_time=datetime.utcnow(),
                metadata=request.metadata
            )
            
        except Exception as e:
            logger.error(f"S3 upload failed: {str(e)}")
            raise ServiceError(f"S3 upload failed: {str(e)}")
    
    async def download_s3_file(self, request: S3DownloadRequest) -> S3DownloadResponse:
        """
        Get download URL for an S3 file.
        
        Args:
            request: S3 download request
            
        Returns:
            S3DownloadResponse: Download information
        """
        try:
            if not self.integration_service:
                raise ServiceError("AWS integration service not available")
            
            if request.file_type == "video":
                download_url = self.integration_service.get_video_url(
                    request.project_id, 
                    request.video_id, 
                    request.scene_number or 0, 
                    request.version
                )
            elif request.file_type == "code":
                download_url = self.integration_service.get_code_url(
                    request.project_id, 
                    request.video_id, 
                    request.version, 
                    request.scene_number
                )
            else:
                raise ServiceError(f"Unsupported file type: {request.file_type}")
            
            # Pre-signed URLs typically expire in 1 hour
            expires_at = datetime.utcnow() + timedelta(hours=1)
            
            return S3DownloadResponse(
                download_url=download_url,
                expires_at=expires_at,
                content_type="video/mp4" if request.file_type == "video" else "text/plain"
            )
            
        except Exception as e:
            logger.error(f"S3 download failed: {str(e)}")
            raise ServiceError(f"S3 download failed: {str(e)}")
    
    async def batch_upload(self, request: BatchUploadRequest) -> BatchUploadResponse:
        """
        Upload multiple files in batch.
        
        Args:
            request: Batch upload request
            
        Returns:
            BatchUploadResponse: Batch upload results
        """
        start_time = time.time()
        
        try:
            if not self.integration_service:
                raise ServiceError("AWS integration service not available")
            
            # Convert to VideoChunk objects
            chunks = []
            for file_info in request.files:
                chunk = VideoChunk(
                    file_path=file_info['file_path'],
                    project_id=request.project_id,
                    video_id=request.video_id,
                    scene_number=file_info.get('scene_number', 0),
                    version=file_info.get('version', 1),
                    metadata={**(request.metadata or {}), **(file_info.get('metadata', {}))}
                )
                chunks.append(chunk)
            
            # Upload chunks
            upload_results = await self.integration_service.upload_video_chunks(chunks)
            
            # Process results
            successful_uploads = [url for url in upload_results if url is not None]
            failed_uploads = len(upload_results) - len(successful_uploads)
            success_rate = (len(successful_uploads) / len(upload_results)) * 100 if upload_results else 0
            
            processing_time = time.time() - start_time
            
            # Create detailed results
            upload_details = []
            for i, (file_info, result) in enumerate(zip(request.files, upload_results)):
                upload_details.append({
                    'file_path': file_info['file_path'],
                    'success': result is not None,
                    's3_url': result,
                    'error': None if result else "Upload failed"
                })
            
            return BatchUploadResponse(
                total_files=len(request.files),
                successful_uploads=len(successful_uploads),
                failed_uploads=failed_uploads,
                success_rate=success_rate,
                upload_results=upload_details,
                processing_time=processing_time,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Batch upload failed: {str(e)}")
            processing_time = time.time() - start_time
            
            return BatchUploadResponse(
                total_files=len(request.files),
                successful_uploads=0,
                failed_uploads=len(request.files),
                success_rate=0.0,
                upload_results=[],
                processing_time=processing_time,
                timestamp=datetime.utcnow()
            )
    
    async def upload_video(self, request: VideoUploadRequest) -> VideoUploadResponse:
        """
        Upload a video file.
        
        Args:
            request: Video upload request
            
        Returns:
            VideoUploadResponse: Upload result
        """
        try:
            if not self.integration_service:
                raise ServiceError("AWS integration service not available")
            
            # Upload video
            s3_url = await self.integration_service.upload_combined_video(
                request.file_path, 
                request.project_id, 
                request.video_id, 
                request.version, 
                request.metadata
            )
            
            # Get file info
            import os
            file_size = os.path.getsize(request.file_path) if os.path.exists(request.file_path) else None
            
            # Create download URL
            download_url = self.integration_service.get_video_url(
                request.project_id, 
                request.video_id, 
                request.scene_number or 0, 
                request.version
            )
            
            return VideoUploadResponse(
                video_id=request.video_id,
                project_id=request.project_id,
                s3_url=s3_url,
                download_url=download_url,
                file_size=file_size,
                upload_time=datetime.utcnow(),
                status="uploaded"
            )
            
        except Exception as e:
            logger.error(f"Video upload failed: {str(e)}")
            raise ServiceError(f"Video upload failed: {str(e)}")
    
    def get_video_download_url(self, project_id: str, video_id: str, 
                              scene_number: int = 0, version: int = 1) -> str:
        """
        Get download URL for a video.
        
        Args:
            project_id: Project identifier
            video_id: Video identifier
            scene_number: Scene number (0 for combined video)
            version: Video version
            
        Returns:
            S3 URL for video download
        """
        return self.integration_service.get_video_url(project_id, video_id, scene_number, version)
    
    # Code Operations
    
    async def upload_code(self, request: CodeUploadRequest) -> CodeUploadResponse:
        """
        Upload a code file to S3.
        
        Args:
            request: Code upload request
            
        Returns:
            CodeUploadResponse: Upload result
        """
        try:
            if not self.integration_service:
                raise ServiceError("AWS integration service not available")
            
            code_metadata = CodeMetadata(
                video_id=request.video_id,
                project_id=request.project_id,
                version=request.version,
                scene_number=request.scene_number,
                created_at=datetime.utcnow(),
                metadata=request.metadata
            )
            
            s3_url = await self.integration_service.upload_code(
                request.code_content, code_metadata, request.enable_object_lock
            )
            
            # Create download URL
            download_url = self.integration_service.get_code_url(
                request.project_id, 
                request.video_id, 
                request.version, 
                request.scene_number
            )
            
            return CodeUploadResponse(
                video_id=request.video_id,
                project_id=request.project_id,
                s3_url=s3_url,
                download_url=download_url,
                version=request.version,
                file_size=len(request.code_content.encode('utf-8')),
                upload_time=datetime.utcnow(),
                language=request.language
            )
            
        except Exception as e:
            logger.error(f"Code upload failed: {str(e)}")
            raise ServiceError(f"Code upload failed: {str(e)}")
    
    async def download_code(self, request: CodeDownloadRequest) -> CodeDownloadResponse:
        """
        Download a code file from S3.
        
        Args:
            request: Code download request
            
        Returns:
            CodeDownloadResponse: Downloaded code content
        """
        try:
            if not self.integration_service:
                raise ServiceError("AWS integration service not available")
            
            code_metadata = CodeMetadata(
                video_id=request.video_id,
                project_id=request.project_id,
                version=request.version,
                scene_number=request.scene_number
            )
            
            code_content = await self.integration_service.download_code(code_metadata)
            
            return CodeDownloadResponse(
                code_content=code_content,
                project_id=request.project_id,
                video_id=request.video_id,
                version=request.version,
                file_size=len(code_content.encode('utf-8')),
                last_modified=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Code download failed: {str(e)}")
            raise ServiceError(f"Code download failed: {str(e)}")
    
    async def list_code_versions(self, project_id: str, video_id: str) -> CodeVersionsResponse:
        """
        List all code versions for a video.
        
        Args:
            project_id: Project identifier
            video_id: Video identifier
            
        Returns:
            CodeVersionsResponse: List of code versions
        """
        try:
            if not self.integration_service:
                raise ServiceError("AWS integration service not available")
            
            versions = await self.integration_service.list_code_versions(project_id, video_id)
            
            # Convert to CodeVersionInfo objects
            version_infos = []
            latest_version = 0
            
            for version in versions:
                version_info = CodeVersionInfo(
                    version=version.metadata.version,
                    scene_number=version.metadata.scene_number,
                    created_at=version.metadata.created_at,
                    s3_path=version.s3_path,
                    file_size=version.file_size,
                    metadata=version.metadata.metadata
                )
                version_infos.append(version_info)
                
                if version.metadata.version > latest_version:
                    latest_version = version.metadata.version
            
            return CodeVersionsResponse(
                project_id=project_id,
                video_id=video_id,
                versions=version_infos,
                total_count=len(version_infos),
                latest_version=latest_version
            )
            
        except Exception as e:
            logger.error(f"Code version listing failed: {str(e)}")
            raise ServiceError(f"Code version listing failed: {str(e)}")
    
    def get_code_download_url(self, project_id: str, video_id: str, version: int = 1,
                             scene_number: Optional[int] = None) -> str:
        """
        Get download URL for a code file.
        
        Args:
            project_id: Project identifier
            video_id: Video identifier
            version: Code version
            scene_number: Optional scene number
            
        Returns:
            S3 URL for code download
        """
        return self.integration_service.get_code_url(project_id, video_id, version, scene_number)
    
    # Metadata Operations
    
    async def create_video_project(self, video_id: str, project_id: str, 
                                  title: str, description: str = "",
                                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a new video project with metadata.
        
        Args:
            video_id: Video identifier
            project_id: Project identifier
            title: Video title
            description: Video description
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            video_metadata = {
                'title': title,
                'description': description,
                'status': 'created',
                'version': 1,
                **(metadata or {})
            }
            
            return await self.integration_service.create_video_record(
                video_id, project_id, video_metadata
            )
            
        except Exception as e:
            logger.error(f"Video project creation failed: {e}")
            raise
    
    async def update_video_status(self, video_id: str, status: str, 
                                 metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update video status and metadata.
        
        Args:
            video_id: Video identifier
            status: New status
            metadata: Additional metadata to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            update_data = {
                'status': status,
                'last_edited_timestamp': datetime.utcnow().isoformat(),
                **(metadata or {})
            }
            
            return await self.integration_service.update_video_metadata(video_id, update_data)
            
        except Exception as e:
            logger.error(f"Video status update failed: {e}")
            raise
    
    async def get_video_details(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed video information.
        
        Args:
            video_id: Video identifier
            
        Returns:
            Video details or None if not found
        """
        try:
            return await self.integration_service.get_video_metadata(video_id)
        except Exception as e:
            logger.error(f"Video details retrieval failed: {e}")
            raise
    
    async def list_project_videos(self, project_id: str) -> List[Dict[str, Any]]:
        """
        List all videos in a project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            List of video metadata
        """
        try:
            return await self.integration_service.list_project_videos(project_id)
        except Exception as e:
            logger.error(f"Project video listing failed: {e}")
            raise
    
    # Integrated Operations
    
    async def upload_video_with_code(self, video_file_path: str, code_content: str,
                                    project_id: str, video_id: str, title: str,
                                    description: str = "", version: int = 1,
                                    progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Upload video and code together with metadata creation.
        
        Args:
            video_file_path: Path to video file
            code_content: Code content
            project_id: Project identifier
            video_id: Video identifier
            title: Video title
            description: Video description
            version: Version number
            progress_callback: Optional progress callback
            
        Returns:
            Dictionary with upload results
        """
        try:
            # Create video project
            await self.create_video_project(video_id, project_id, title, description)
            
            # Upload video
            video_url = await self.upload_single_video(
                video_file_path, project_id, video_id, version, None, progress_callback
            )
            
            # Upload code
            code_url = await self.upload_code_file(
                code_content, project_id, video_id, version
            )
            
            # Update metadata with URLs
            await self.update_video_status(video_id, 'uploaded', {
                's3_path_full_video': video_url,
                's3_path_code': code_url,
                'upload_completed_at': datetime.utcnow().isoformat(),
                'code_upload_completed_at': datetime.utcnow().isoformat()
            })
            
            return {
                'video_id': video_id,
                'project_id': project_id,
                'video_url': video_url,
                'code_url': code_url,
                'status': 'uploaded',
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Integrated upload failed: {e}")
            raise AWSIntegrationError(f"Integrated upload failed: {str(e)}") from e
    
    # Health and Status
    
    async def create_video_project(self, request: VideoProjectRequest) -> VideoProjectResponse:
        """
        Create a new video project.
        
        Args:
            request: Video project creation request
            
        Returns:
            VideoProjectResponse: Created project information
        """
        try:
            if not self.integration_service:
                raise ServiceError("AWS integration service not available")
            
            video_metadata = {
                'title': request.title,
                'description': request.description,
                'status': 'created',
                'version': 1,
                'tags': request.tags,
                **(request.metadata or {})
            }
            
            success = await self.integration_service.create_video_record(
                request.video_id, request.project_id, video_metadata
            )
            
            if not success:
                raise ServiceError("Failed to create video project")
            
            return VideoProjectResponse(
                video_id=request.video_id,
                project_id=request.project_id,
                title=request.title,
                description=request.description,
                status='created',
                created_at=datetime.utcnow(),
                metadata=request.metadata,
                tags=request.tags
            )
            
        except Exception as e:
            logger.error(f"Video project creation failed: {str(e)}")
            raise ServiceError(f"Video project creation failed: {str(e)}")
    
    async def update_video_metadata(self, video_id: str, request: VideoMetadataUpdateRequest) -> VideoDetailsResponse:
        """
        Update video metadata.
        
        Args:
            video_id: Video identifier
            request: Metadata update request
            
        Returns:
            VideoDetailsResponse: Updated video details
        """
        try:
            if not self.integration_service:
                raise ServiceError("AWS integration service not available")
            
            update_data = {}
            if request.status:
                update_data['status'] = request.status
            if request.title:
                update_data['title'] = request.title
            if request.description:
                update_data['description'] = request.description
            if request.tags:
                update_data['tags'] = request.tags
            if request.metadata:
                update_data.update(request.metadata)
            
            update_data['last_edited_timestamp'] = datetime.utcnow().isoformat()
            
            success = await self.integration_service.update_video_metadata(video_id, update_data)
            
            if not success:
                raise ServiceError("Failed to update video metadata")
            
            # Get updated details
            return await self.get_video_details(video_id)
            
        except Exception as e:
            logger.error(f"Video metadata update failed: {str(e)}")
            raise ServiceError(f"Video metadata update failed: {str(e)}")
    
    async def get_video_details(self, video_id: str) -> VideoDetailsResponse:
        """
        Get detailed video information.
        
        Args:
            video_id: Video identifier
            
        Returns:
            VideoDetailsResponse: Video details
        """
        try:
            if not self.integration_service:
                raise ServiceError("AWS integration service not available")
            
            metadata = await self.integration_service.get_video_metadata(video_id)
            
            if not metadata:
                raise ServiceError(f"Video not found: {video_id}")
            
            return VideoDetailsResponse(
                video_id=video_id,
                project_id=metadata.get('project_id', ''),
                title=metadata.get('title', ''),
                description=metadata.get('description', ''),
                status=metadata.get('status', 'unknown'),
                version=metadata.get('version', 1),
                created_at=datetime.fromisoformat(metadata.get('created_at', datetime.utcnow().isoformat())),
                updated_at=datetime.fromisoformat(metadata.get('last_edited_timestamp')) if metadata.get('last_edited_timestamp') else None,
                s3_urls=metadata.get('s3_urls', {}),
                download_urls=metadata.get('download_urls', {}),
                file_sizes=metadata.get('file_sizes', {}),
                metadata=metadata,
                tags=metadata.get('tags', [])
            )
            
        except Exception as e:
            logger.error(f"Video details retrieval failed: {str(e)}")
            raise ServiceError(f"Video details retrieval failed: {str(e)}")
    
    async def list_project_videos(self, project_id: str) -> ProjectVideosResponse:
        """
        List all videos in a project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            ProjectVideosResponse: List of project videos
        """
        try:
            if not self.integration_service:
                raise ServiceError("AWS integration service not available")
            
            videos_metadata = await self.integration_service.list_project_videos(project_id)
            
            videos = []
            total_size = 0
            
            for metadata in videos_metadata:
                video_details = VideoDetailsResponse(
                    video_id=metadata.get('video_id', ''),
                    project_id=project_id,
                    title=metadata.get('title', ''),
                    description=metadata.get('description', ''),
                    status=metadata.get('status', 'unknown'),
                    version=metadata.get('version', 1),
                    created_at=datetime.fromisoformat(metadata.get('created_at', datetime.utcnow().isoformat())),
                    updated_at=datetime.fromisoformat(metadata.get('last_edited_timestamp')) if metadata.get('last_edited_timestamp') else None,
                    s3_urls=metadata.get('s3_urls', {}),
                    download_urls=metadata.get('download_urls', {}),
                    file_sizes=metadata.get('file_sizes', {}),
                    metadata=metadata,
                    tags=metadata.get('tags', [])
                )
                videos.append(video_details)
                
                # Sum up file sizes
                for size in metadata.get('file_sizes', {}).values():
                    if isinstance(size, int):
                        total_size += size
            
            return ProjectVideosResponse(
                project_id=project_id,
                videos=videos,
                total_count=len(videos),
                total_size=total_size
            )
            
        except Exception as e:
            logger.error(f"Project video listing failed: {str(e)}")
            raise ServiceError(f"Project video listing failed: {str(e)}")
    
    async def integrated_upload(self, request: IntegratedUploadRequest) -> IntegratedUploadResponse:
        """
        Upload video and code together with metadata creation.
        
        Args:
            request: Integrated upload request
            
        Returns:
            IntegratedUploadResponse: Upload results
        """
        try:
            if not self.integration_service:
                raise ServiceError("AWS integration service not available")
            
            # Create video project
            project_request = VideoProjectRequest(
                video_id=request.video_id,
                project_id=request.project_id,
                title=request.title,
                description=request.description,
                metadata=request.metadata
            )
            await self.create_video_project(project_request)
            
            # Upload video
            video_url = await self.integration_service.upload_combined_video(
                request.video_file_path, 
                request.project_id, 
                request.video_id, 
                request.version
            )
            
            # Upload code
            code_metadata = CodeMetadata(
                video_id=request.video_id,
                project_id=request.project_id,
                version=request.version,
                created_at=datetime.utcnow(),
                metadata=request.metadata
            )
            
            code_url = await self.integration_service.upload_code(
                request.code_content, code_metadata
            )
            
            # Create download URLs
            video_download_url = self.integration_service.get_video_url(
                request.project_id, request.video_id, 0, request.version
            )
            code_download_url = self.integration_service.get_code_url(
                request.project_id, request.video_id, request.version
            )
            
            # Update metadata with URLs
            await self.integration_service.update_video_metadata(request.video_id, {
                's3_path_full_video': video_url,
                's3_path_code': code_url,
                'upload_completed_at': datetime.utcnow().isoformat(),
                'code_upload_completed_at': datetime.utcnow().isoformat()
            })
            
            # Calculate total size
            import os
            total_size = 0
            if os.path.exists(request.video_file_path):
                total_size += os.path.getsize(request.video_file_path)
            total_size += len(request.code_content.encode('utf-8'))
            
            return IntegratedUploadResponse(
                video_id=request.video_id,
                project_id=request.project_id,
                video_url=video_url,
                code_url=code_url,
                video_download_url=video_download_url,
                code_download_url=code_download_url,
                status='uploaded',
                upload_time=datetime.utcnow(),
                total_size=total_size
            )
            
        except Exception as e:
            logger.error(f"Integrated upload failed: {str(e)}")
            raise ServiceError(f"Integrated upload failed: {str(e)}")
    
    async def get_service_health(self) -> AWSHealthResponse:
        """
        Get AWS service health status.
        
        Returns:
            AWSHealthResponse: Health status
        """
        try:
            if not self.integration_service:
                return AWSHealthResponse(
                    overall_status='unhealthy',
                    services={},
                    region=self.config.region,
                    timestamp=datetime.utcnow(),
                    error_counts={'integration_service': 1}
                )
            
            health_data = await self.integration_service.health_check()
            
            return AWSHealthResponse(
                overall_status=health_data.get('overall_status', 'unknown'),
                services=health_data.get('services', {}),
                region=self.config.region,
                timestamp=datetime.utcnow(),
                response_times=health_data.get('response_times', {}),
                error_counts=health_data.get('error_counts', {})
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return AWSHealthResponse(
                overall_status='unhealthy',
                services={'error': str(e)},
                region=self.config.region,
                timestamp=datetime.utcnow(),
                error_counts={'health_check': 1}
            )
    
    async def get_service_config(self) -> AWSConfigResponse:
        """
        Get service configuration information.
        
        Returns:
            AWSConfigResponse: Configuration information
        """
        try:
            service_info = self.integration_service.get_service_info() if self.integration_service else {}
            
            return AWSConfigResponse(
                region=self.config.region,
                s3_bucket=self.config.s3_bucket,
                dynamodb_table=self.config.dynamodb_table,
                enabled_services=[AWSServiceType.S3, AWSServiceType.DYNAMODB],  # Based on config
                configuration=service_info,
                credentials_configured=self.config.is_enabled(),
                last_updated=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to get service config: {str(e)}")
            raise ServiceError(f"Failed to get service config: {str(e)}")
    
    def is_enabled(self) -> bool:
        """
        Check if AWS integration is enabled.
        
        Returns:
            True if enabled, False otherwise
        """
        return self.config.is_enabled()
    
    def __repr__(self) -> str:
        """String representation of the AWS service."""
        return (
            f"AWSService(enabled={self.config.is_enabled()}, "
            f"region='{self.config.region}')"
        )
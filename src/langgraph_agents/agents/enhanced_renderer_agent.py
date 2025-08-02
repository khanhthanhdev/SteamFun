"""
Enhanced RendererAgent with AWS S3 Upload Integration

Extends the existing RendererAgent with AWS S3 upload capabilities, graceful degradation,
progress tracking, and DynamoDB metadata updates.
"""

import os
import re
import asyncio
import logging
import random
from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
from langgraph.types import Command
from datetime import datetime

from .renderer_agent import RendererAgent
from ..state import VideoGenerationState
from src.aws.s3_video_upload import S3VideoUploadService, VideoChunk
from src.aws.metadata_service import MetadataService
from src.aws.config import AWSConfig
from src.aws.credentials import AWSCredentialsManager
from src.aws.exceptions import AWSS3Error, AWSMetadataError, AWSRetryableError

logger = logging.getLogger(__name__)


class EnhancedRendererAgent(RendererAgent):
    """
    Enhanced RendererAgent with AWS S3 upload integration.
    
    Extends the base RendererAgent with:
    - Automatic S3 upload after successful rendering
    - Progress tracking and status reporting
    - DynamoDB metadata updates
    - Graceful degradation when AWS upload fails
    """
    
    def __init__(self, config, system_config, aws_config: Optional[AWSConfig] = None):
        """
        Initialize EnhancedRendererAgent with AWS capabilities.
        
        Args:
            config: Agent configuration
            system_config: System configuration
            aws_config: AWS configuration (optional, will use default if not provided)
        """
        super().__init__(config, system_config)
        
        # Initialize upload progress tracking first
        self.upload_progress = {}
        self.upload_callbacks = []
        
        # Initialize AWS services if configuration is provided
        self.aws_config = aws_config
        self.aws_enabled = aws_config is not None
        self.s3_upload_service = None
        self.metadata_service = None
        
        if self.aws_enabled:
            try:
                credentials_manager = AWSCredentialsManager(aws_config)
                self.s3_upload_service = S3VideoUploadService(aws_config, credentials_manager)
                self.metadata_service = MetadataService(aws_config)
                logger.info("AWS services initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize AWS services: {e}")
                self.aws_enabled = False
                if aws_config and hasattr(aws_config, 'require_aws_upload') and aws_config.require_aws_upload:
                    raise
        
        logger.info(f"EnhancedRendererAgent initialized - AWS enabled: {self.aws_enabled}")
    
    async def execute(self, state: VideoGenerationState) -> Command:
        """
        Execute video rendering with AWS upload integration.
        
        Args:
            state: Current workflow state
            
        Returns:
            Command: LangGraph command for next action
        """
        self.log_agent_action("starting_enhanced_rendering", {
            'topic': state.get('topic', ''),
            'scene_count': len(state.get('generated_code', {})),
            'aws_enabled': self.aws_enabled,
            'upload_enabled': state.get('enable_aws_upload', True)
        })
        
        # First, execute the base rendering functionality
        try:
            render_command = await super().execute(state)
            
            # Check if rendering was successful
            rendered_videos = state.get('rendered_videos', {})
            combined_video_path = state.get('combined_video_path')
            
            if not rendered_videos and not combined_video_path:
                # Rendering failed, return the original command
                return render_command
            
            # If AWS upload is enabled and we have successful renders, upload to S3
            if (self.aws_enabled and 
                state.get('enable_aws_upload', True) and 
                (rendered_videos or combined_video_path)):
                
                try:
                    upload_results = await self._upload_videos_to_aws(state)
                    
                    # Update state with upload results
                    state.update({
                        "aws_upload_results": upload_results,
                        "upload_status": "completed",
                        "upload_timestamp": datetime.utcnow().isoformat()
                    })
                    
                    # Update DynamoDB metadata
                    await self._update_video_metadata(state, upload_results)
                    
                    self.log_agent_action("aws_upload_completed", {
                        'successful_uploads': len([r for r in upload_results.values() if r]),
                        'total_videos': len(upload_results)
                    })
                    
                except Exception as upload_error:
                    logger.error(f"AWS upload failed: {upload_error}")
                    
                    # Handle upload failure based on configuration
                    if self.aws_config and self.aws_config.require_aws_upload:
                        # Upload is required, treat as error
                        state.update({
                            "upload_status": "failed",
                            "upload_error": str(upload_error),
                            "upload_timestamp": datetime.utcnow().isoformat()
                        })
                        
                        return await self._handle_upload_failure(upload_error, state)
                    else:
                        # Graceful degradation - continue workflow without upload
                        state.update({
                            "upload_status": "failed_graceful",
                            "upload_error": str(upload_error),
                            "upload_timestamp": datetime.utcnow().isoformat()
                        })
                        
                        self.log_agent_action("graceful_degradation", {
                            'error': str(upload_error),
                            'continuing_workflow': True
                        })
            
            # Update the render command with our enhanced state
            if hasattr(render_command, 'update') and render_command.update:
                render_command.update.update(state)
            
            return render_command
            
        except Exception as e:
            logger.error(f"Error in EnhancedRendererAgent execution: {e}")
            return await self.handle_error(e, state)
    
    async def _upload_videos_to_aws(self, state: VideoGenerationState) -> Dict[str, Optional[str]]:
        """
        Upload rendered videos to AWS S3 with progress tracking.
        
        Args:
            state: Current workflow state containing rendered videos
            
        Returns:
            Dictionary mapping scene numbers to S3 URLs (None for failed uploads)
            
        Raises:
            AWSS3Error: If upload fails and is required
        """
        if not self.s3_upload_service:
            raise AWSS3Error("S3 upload service not initialized", operation="upload_videos")
        
        upload_results = {}
        rendered_videos = state.get('rendered_videos', {})
        combined_video_path = state.get('combined_video_path')
        
        # Prepare video chunks for upload
        video_chunks = []
        
        # Add individual scene videos
        for scene_number, video_path in rendered_videos.items():
            if video_path and os.path.exists(video_path):
                chunk = VideoChunk(
                    file_path=video_path,
                    project_id=state.get('project_id', 'default'),
                    video_id=state.get('video_id', state.get('session_id', 'unknown')),
                    scene_number=scene_number,
                    version=state.get('version', 1),
                    metadata={
                        'topic': state.get('topic', ''),
                        'description': state.get('description', ''),
                        'render_timestamp': datetime.utcnow().isoformat(),
                        'quality': state.get('default_quality', 'medium')
                    }
                )
                video_chunks.append(chunk)
        
        # Add combined video if available
        if combined_video_path and os.path.exists(combined_video_path):
            combined_chunk = VideoChunk(
                file_path=combined_video_path,
                project_id=state.get('project_id', 'default'),
                video_id=state.get('video_id', state.get('session_id', 'unknown')),
                scene_number=0,  # 0 indicates combined video
                version=state.get('version', 1),
                metadata={
                    'topic': state.get('topic', ''),
                    'description': state.get('description', ''),
                    'render_timestamp': datetime.utcnow().isoformat(),
                    'quality': state.get('default_quality', 'medium'),
                    'video_type': 'combined'
                }
            )
            video_chunks.append(combined_chunk)
        
        if not video_chunks:
            logger.warning("No video files found for upload")
            return upload_results
        
        # Create progress callback
        progress_callback = self._create_upload_progress_callback(state)
        
        # Upload video chunks
        self.log_agent_action("uploading_videos_to_s3", {
            'chunk_count': len(video_chunks),
            'total_size_mb': sum(os.path.getsize(chunk.file_path) for chunk in video_chunks) / (1024 * 1024)
        })
        
        s3_urls = await self.s3_upload_service.upload_video_chunks(
            chunks=video_chunks,
            progress_callback=progress_callback
        )
        
        # Map results back to scene numbers
        for i, chunk in enumerate(video_chunks):
            if chunk.scene_number == 0:
                upload_results['combined'] = s3_urls[i]
            else:
                upload_results[chunk.scene_number] = s3_urls[i]
        
        return upload_results
    
    def _create_upload_progress_callback(self, state: VideoGenerationState) -> Callable:
        """
        Create a progress callback for upload tracking.
        
        Args:
            state: Current workflow state
            
        Returns:
            Progress callback function
        """
        def progress_callback(bytes_uploaded: int, total_bytes: int, percentage: float):
            """Progress callback for upload tracking."""
            video_id = state.get('video_id', state.get('session_id', 'unknown'))
            
            # Update progress tracking
            self.upload_progress[video_id] = {
                'bytes_uploaded': bytes_uploaded,
                'total_bytes': total_bytes,
                'percentage': percentage,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Call registered callbacks
            for callback in self.upload_callbacks:
                try:
                    callback(video_id, bytes_uploaded, total_bytes, percentage)
                except Exception as e:
                    logger.error(f"Upload progress callback error: {e}")
            
            # Log progress at intervals
            if int(percentage) % 10 == 0:
                self.log_agent_action("upload_progress", {
                    'video_id': video_id,
                    'percentage': percentage,
                    'bytes_uploaded': bytes_uploaded,
                    'total_bytes': total_bytes
                })
        
        return progress_callback
    
    async def _update_video_metadata(self, state: VideoGenerationState, 
                                   upload_results: Dict[str, Optional[str]]) -> None:
        """
        Update DynamoDB metadata after successful video uploads.
        
        Args:
            state: Current workflow state
            upload_results: Dictionary of upload results
        """
        if not self.metadata_service:
            logger.warning("Metadata service not available, skipping metadata update")
            return
        
        try:
            video_id = state.get('video_id', state.get('session_id', 'unknown'))
            
            # Filter successful uploads
            successful_uploads = {k: v for k, v in upload_results.items() if v is not None}
            
            if not successful_uploads:
                logger.warning("No successful uploads to record in metadata")
                return
            
            # Prepare metadata update
            metadata_update = {
                'status': 'uploaded',
                'upload_completed_at': datetime.utcnow().isoformat(),
                'last_edited_timestamp': datetime.utcnow().isoformat(),
                's3_paths': successful_uploads,
                'upload_summary': {
                    'successful_uploads': len(successful_uploads),
                    'total_attempts': len(upload_results),
                    'success_rate': len(successful_uploads) / len(upload_results) * 100
                }
            }
            
            # Set main video path (combined video takes precedence)
            if 'combined' in successful_uploads:
                metadata_update['s3_path_full_video'] = successful_uploads['combined']
            elif successful_uploads:
                # Use first successful upload as main path
                metadata_update['s3_path_main'] = list(successful_uploads.values())[0]
            
            # Add rendering metadata
            rendered_videos = state.get('rendered_videos', {})
            if rendered_videos:
                metadata_update['scene_count'] = len(rendered_videos)
                metadata_update['rendered_scenes'] = list(rendered_videos.keys())
            
            # Update metadata in DynamoDB
            success = await self.metadata_service.operations.update_video_metadata(
                video_id=video_id,
                metadata=metadata_update
            )
            
            if success:
                self.log_agent_action("metadata_updated", {
                    'video_id': video_id,
                    'successful_uploads': len(successful_uploads)
                })
            else:
                logger.error(f"Failed to update metadata for video: {video_id}")
                
        except Exception as e:
            logger.error(f"Error updating video metadata: {e}")
            # Don't raise - metadata update failure shouldn't stop the workflow
    
    async def _handle_upload_failure(self, error: Exception, 
                                   state: VideoGenerationState) -> Command:
        """
        Handle upload failures with retry strategies and error escalation.
        
        Args:
            error: Upload error that occurred
            state: Current workflow state
            
        Returns:
            Command for error handling
        """
        error_type = type(error).__name__
        
        self.log_agent_action("handling_upload_failure", {
            'error_type': error_type,
            'error_message': str(error),
            'retryable': isinstance(error, AWSRetryableError)
        })
        
        # Check if this is a retryable error
        if isinstance(error, AWSRetryableError):
            retry_count = state.get('upload_retry_count', 0)
            max_retries = self.aws_config.max_retries if self.aws_config else 3
            
            if retry_count < max_retries:
                # Retry upload with exponential backoff
                delay = (2 ** retry_count) + (random.random() * 0.1)
                
                self.log_agent_action("retrying_upload", {
                    'retry_count': retry_count + 1,
                    'max_retries': max_retries,
                    'delay_seconds': delay
                })
                
                await asyncio.sleep(delay)
                
                # Update retry count and retry
                retry_state = state.copy()
                retry_state['upload_retry_count'] = retry_count + 1
                
                return await self.execute(retry_state)
        
        # Check if we should escalate to human intervention
        if self.should_escalate_to_human(state):
            return self.create_human_intervention_command(
                context=f"AWS upload failed for video '{state.get('topic', '')}': {str(error)}",
                options=[
                    "Retry upload with different settings",
                    "Continue without upload (if allowed)",
                    "Manual intervention required",
                    "Switch to local storage only"
                ],
                state=state
            )
        
        # Handle error through base class
        return await self.handle_error(error, state)
    
    def add_upload_progress_callback(self, callback: Callable[[str, int, int, float], None]):
        """
        Add a callback for upload progress tracking.
        
        Args:
            callback: Function to call with (video_id, bytes_uploaded, total_bytes, percentage)
        """
        self.upload_callbacks.append(callback)
    
    def remove_upload_progress_callback(self, callback: Callable):
        """
        Remove an upload progress callback.
        
        Args:
            callback: Callback function to remove
        """
        if callback in self.upload_callbacks:
            self.upload_callbacks.remove(callback)
    
    def get_upload_progress(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current upload progress for a video.
        
        Args:
            video_id: Video identifier
            
        Returns:
            Upload progress information or None if not found
        """
        return self.upload_progress.get(video_id)
    
    def get_upload_status(self, state: VideoGenerationState) -> Dict[str, Any]:
        """
        Get comprehensive upload status information.
        
        Args:
            state: Current workflow state
            
        Returns:
            Upload status information
        """
        base_status = self.get_rendering_status(state)
        
        upload_status = {
            'aws_enabled': self.aws_enabled,
            'upload_service_available': self.s3_upload_service is not None,
            'metadata_service_available': self.metadata_service is not None,
            'upload_status': state.get('upload_status', 'not_started'),
            'upload_error': state.get('upload_error'),
            'upload_timestamp': state.get('upload_timestamp'),
            'aws_upload_results': state.get('aws_upload_results', {}),
            'upload_retry_count': state.get('upload_retry_count', 0)
        }
        
        # Add progress information
        video_id = state.get('video_id', state.get('session_id'))
        if video_id and video_id in self.upload_progress:
            upload_status['current_progress'] = self.upload_progress[video_id]
        
        # Combine with base rendering status
        base_status.update(upload_status)
        return base_status
    
    async def cleanup_upload_resources(self):
        """Clean up upload-related resources."""
        # Clear progress tracking
        self.upload_progress.clear()
        self.upload_callbacks.clear()
        
        # Cleanup base renderer resources
        if hasattr(super(), 'cleanup_cache'):
            super().cleanup_cache()
        
        # Cleanup AWS service resources
        if self.metadata_service and hasattr(self.metadata_service, 'executor'):
            self.metadata_service.executor.shutdown(wait=False)
    
    def __del__(self):
        """Cleanup resources when agent is destroyed."""
        try:
            # Use asyncio to run cleanup if event loop is available
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.cleanup_upload_resources())
            else:
                asyncio.run(self.cleanup_upload_resources())
        except Exception:
            # Fallback cleanup
            self.upload_progress.clear()
            self.upload_callbacks.clear()
        
        # Call parent destructor if it exists
        if hasattr(super(), '__del__'):
            super().__del__()
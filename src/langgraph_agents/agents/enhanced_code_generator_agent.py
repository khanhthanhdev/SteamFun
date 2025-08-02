"""
Enhanced CodeGeneratorAgent with S3 Code Management

Extends the existing CodeGeneratorAgent with AWS S3 code storage capabilities,
existing code download functionality, versioning, and fallback mechanisms.
"""

import os
import asyncio
import logging
import random
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from langgraph.types import Command
from datetime import datetime

from .code_generator_agent import CodeGeneratorAgent
from ..state import VideoGenerationState
from src.aws.s3_code_storage import S3CodeStorageService, CodeMetadata, CodeVersion
from src.aws.metadata_service import MetadataService
from src.aws.config import AWSConfig
from src.aws.credentials import AWSCredentialsManager
from src.aws.exceptions import AWSS3Error, AWSMetadataError, AWSRetryableError

logger = logging.getLogger(__name__)


class EnhancedCodeGeneratorAgent(CodeGeneratorAgent):
    """
    Enhanced CodeGeneratorAgent with AWS S3 code management.
    
    Extends the base CodeGeneratorAgent with:
    - Existing code download functionality for editing workflows
    - Code upload with proper versioning after generation
    - Code metadata management and S3 path tracking
    - Fallback mechanisms when code download fails
    """
    
    def __init__(self, config, system_config, aws_config: Optional[AWSConfig] = None):
        """
        Initialize EnhancedCodeGeneratorAgent with AWS capabilities.
        
        Args:
            config: Agent configuration
            system_config: System configuration
            aws_config: AWS configuration (optional, will use default if not provided)
        """
        super().__init__(config, system_config)
        
        # Initialize code version tracking first
        self.code_versions = {}
        self.download_cache = {}
        
        # Initialize AWS services if configuration is provided
        self.aws_config = aws_config
        self.aws_enabled = aws_config is not None
        self.s3_code_service = None
        self.metadata_service = None
        
        if self.aws_enabled:
            try:
                credentials_manager = AWSCredentialsManager(aws_config)
                self.s3_code_service = S3CodeStorageService(aws_config, credentials_manager)
                self.metadata_service = MetadataService(aws_config)
                logger.info("AWS code management services initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize AWS code services: {e}")
                self.aws_enabled = False
                if aws_config and hasattr(aws_config, 'require_aws_upload') and aws_config.require_aws_upload:
                    raise
        
        logger.info(f"EnhancedCodeGeneratorAgent initialized - AWS enabled: {self.aws_enabled}")
    
    async def execute(self, state: VideoGenerationState) -> Command:
        """
        Execute code generation with AWS code management integration.
        
        Args:
            state: Current workflow state
            
        Returns:
            Command: LangGraph command for next action
        """
        self.log_agent_action("starting_enhanced_code_generation", {
            'topic': state.get('topic', ''),
            'scene_count': len(state.get('scene_implementations', {})),
            'aws_enabled': self.aws_enabled,
            'editing_existing': state.get('editing_existing_video', False)
        })
        
        # Check if we need to download existing code for editing workflows
        if (self.aws_enabled and 
            state.get('editing_existing_video', False) and 
            state.get('enable_aws_code_management', True)):
            
            try:
                existing_code = await self._download_existing_code(state)
                if existing_code:
                    state.update({"existing_code": existing_code})
                    self.log_agent_action("existing_code_downloaded", {
                        'video_id': state.get('video_id'),
                        'code_versions_found': len(existing_code)
                    })
                else:
                    logger.info("No existing code found, proceeding with fresh generation")
                    
            except Exception as e:
                logger.error(f"Failed to download existing code: {e}")
                
                # Handle download failure based on configuration
                if self.aws_config and self.aws_config.require_aws_upload:
                    return await self._handle_code_download_failure(e, state)
                else:
                    # Graceful fallback - continue without existing code
                    state.update({"existing_code": None})
                    self.log_agent_action("graceful_fallback_no_existing_code", {
                        'error': str(e),
                        'continuing_without_existing_code': True
                    })
        
        # Execute the base code generation functionality
        try:
            generation_command = await super().execute(state)
            
            # Check if code generation was successful
            generated_code = state.get('generated_code', {})
            
            if not generated_code:
                # Code generation failed, return the original command
                return generation_command
            
            # If AWS code management is enabled and we have generated code, upload to S3
            if (self.aws_enabled and 
                state.get('enable_aws_code_management', True) and 
                generated_code):
                
                try:
                    upload_results = await self._upload_code_to_aws(state)
                    
                    # Update state with upload results
                    state.update({
                        "code_upload_results": upload_results,
                        "code_upload_status": "completed",
                        "code_upload_timestamp": datetime.utcnow().isoformat()
                    })
                    
                    # Update DynamoDB metadata with code paths
                    await self._update_code_metadata(state, upload_results)
                    
                    self.log_agent_action("code_upload_completed", {
                        'successful_uploads': len([r for r in upload_results.values() if r]),
                        'total_code_files': len(upload_results)
                    })
                    
                except Exception as upload_error:
                    logger.error(f"Code upload failed: {upload_error}")
                    
                    # Handle upload failure based on configuration
                    if self.aws_config and self.aws_config.require_aws_upload:
                        # Upload is required, treat as error
                        state.update({
                            "code_upload_status": "failed",
                            "code_upload_error": str(upload_error),
                            "code_upload_timestamp": datetime.utcnow().isoformat()
                        })
                        
                        return await self._handle_code_upload_failure(upload_error, state)
                    else:
                        # Graceful degradation - continue workflow without upload
                        state.update({
                            "code_upload_status": "failed_graceful",
                            "code_upload_error": str(upload_error),
                            "code_upload_timestamp": datetime.utcnow().isoformat()
                        })
                        
                        self.log_agent_action("graceful_degradation_code_upload", {
                            'error': str(upload_error),
                            'continuing_workflow': True
                        })
            
            # Update the generation command with our enhanced state
            if hasattr(generation_command, 'update') and generation_command.update:
                generation_command.update.update(state)
            
            return generation_command
            
        except Exception as e:
            logger.error(f"Error in EnhancedCodeGeneratorAgent execution: {e}")
            return await self.handle_error(e, state)
    
    async def _download_existing_code(self, state: VideoGenerationState) -> Optional[Dict[str, str]]:
        """
        Download existing Manim code from S3 for editing workflows.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dictionary of existing code by scene number, or None if not found
            
        Raises:
            AWSS3Error: If download fails and is required
        """
        if not self.s3_code_service:
            raise AWSS3Error("S3 code service not initialized", operation="download_existing_code")
        
        video_id = state.get('video_id')
        if not video_id:
            logger.warning("No video_id provided for existing code download")
            return None
        
        # Check cache first
        cache_key = f"{video_id}_{state.get('current_version', 'latest')}"
        if cache_key in self.download_cache:
            logger.info(f"Using cached code for {video_id}")
            return self.download_cache[cache_key]
        
        try:
            # Get current version from metadata or use latest
            current_version = await self._get_current_code_version(video_id, state)
            
            if not current_version:
                logger.info(f"No existing code versions found for video: {video_id}")
                return None
            
            # Check what scene code files are available
            scene_files = await self.s3_code_service.list_scene_code_files(
                video_id=video_id,
                project_id=state.get('project_id', 'default'),
                version=current_version
            )
            
            # Download code for each available scene
            existing_code = {}
            scene_implementations = state.get('scene_implementations', {})
            
            # Try to download scene-specific code first
            for scene_number in scene_implementations.keys():
                if scene_number in scene_files:
                    try:
                        code_content = await self.s3_code_service.download_code(
                            video_id=video_id,
                            project_id=state.get('project_id', 'default'),
                            version=current_version,
                            scene_number=scene_number
                        )
                        if code_content:
                            existing_code[scene_number] = code_content
                            
                            self.log_agent_action("scene_code_downloaded", {
                                'video_id': video_id,
                                'scene_number': scene_number,
                                'version': current_version,
                                'code_length': len(code_content)
                            })
                        
                    except Exception as scene_error:
                        logger.warning(f"Failed to download code for scene {scene_number}: {scene_error}")
                        # Continue with other scenes
            
            # Also try to download main/combined code
            try:
                main_code = await self.s3_code_service.download_code(
                    video_id=video_id,
                    project_id=state.get('project_id', 'default'),
                    version=current_version
                )
                if main_code:
                    existing_code['main'] = main_code
                    
            except Exception as main_error:
                logger.info(f"No main code file found: {main_error}")
            
            # Cache the results
            if existing_code:
                self.download_cache[cache_key] = existing_code
                
                self.log_agent_action("existing_code_download_completed", {
                    'video_id': video_id,
                    'version': current_version,
                    'scenes_downloaded': len(existing_code)
                })
            
            return existing_code if existing_code else None
            
        except Exception as e:
            logger.error(f"Error downloading existing code for {video_id}: {e}")
            raise
    
    async def _get_current_code_version(self, video_id: str, 
                                      state: VideoGenerationState) -> Optional[int]:
        """
        Get the current code version from metadata.
        
        Args:
            video_id: Video identifier
            state: Current workflow state
            
        Returns:
            Current version number or None if not found
        """
        if not self.metadata_service:
            # Fallback to state or default
            return state.get('current_version', 1)
        
        try:
            metadata = await self.metadata_service.operations.get_video_metadata(video_id)
            if metadata:
                # Extract version from current_version_id (e.g., "v2" -> 2)
                version_id = metadata.get('current_version_id', 'v1')
                if version_id.startswith('v'):
                    return int(version_id[1:])
                else:
                    return int(version_id)
            
        except Exception as e:
            logger.warning(f"Failed to get current version from metadata: {e}")
        
        # Fallback to state or default
        return state.get('current_version', 1)
    
    async def _upload_code_to_aws(self, state: VideoGenerationState) -> Dict[str, Optional[str]]:
        """
        Upload generated Manim code to S3 with versioning.
        
        Args:
            state: Current workflow state containing generated code
            
        Returns:
            Dictionary mapping scene numbers to S3 URLs (None for failed uploads)
            
        Raises:
            AWSS3Error: If upload fails and is required
        """
        if not self.s3_code_service:
            raise AWSS3Error("S3 code service not initialized", operation="upload_code")
        
        upload_results = {}
        generated_code = state.get('generated_code', {})
        
        if not generated_code:
            logger.warning("No generated code found for upload")
            return upload_results
        
        # Determine new version number
        new_version = await self._calculate_new_version(state)
        
        # Upload each scene's code
        for scene_number, scene_code in generated_code.items():
            try:
                # Create metadata for this code upload
                code_metadata = CodeMetadata(
                    video_id=state.get('video_id', state.get('session_id', 'unknown')),
                    project_id=state.get('project_id', 'default'),
                    version=new_version,
                    scene_number=scene_number,
                    created_at=datetime.utcnow(),
                    metadata={
                        'topic': state.get('topic', ''),
                        'description': state.get('description', ''),
                        'generation_timestamp': datetime.utcnow().isoformat(),
                        'agent_version': 'enhanced_code_generator_v1'
                    }
                )
                
                # Upload to S3
                s3_url = await self.s3_code_service.upload_code(
                    code=scene_code,
                    metadata=code_metadata,
                    enable_object_lock=state.get('enable_code_object_lock', False)
                )
                
                upload_results[scene_number] = s3_url
                
                self.log_agent_action("scene_code_uploaded", {
                    'scene_number': scene_number,
                    'version': new_version,
                    's3_url': s3_url,
                    'code_length': len(scene_code)
                })
                
            except Exception as e:
                logger.error(f"Failed to upload code for scene {scene_number}: {e}")
                upload_results[scene_number] = None
                
                # Continue with other scenes unless upload is required
                if self.aws_config and self.aws_config.require_aws_upload:
                    raise AWSS3Error(
                        f"Required code upload failed for scene: {scene_number}",
                        operation="upload_code_to_aws"
                    ) from e
        
        # Update version tracking
        video_id = state.get('video_id', state.get('session_id', 'unknown'))
        self.code_versions[video_id] = new_version
        
        # Update state with new version
        state.update({
            'current_version': new_version,
            'version': new_version
        })
        
        successful_uploads = [url for url in upload_results.values() if url is not None]
        self.log_agent_action("code_upload_batch_completed", {
            'successful_uploads': len(successful_uploads),
            'total_attempts': len(upload_results),
            'new_version': new_version
        })
        
        return upload_results
    
    async def _calculate_new_version(self, state: VideoGenerationState) -> int:
        """
        Calculate the new version number for code upload.
        
        Args:
            state: Current workflow state
            
        Returns:
            New version number
        """
        if state.get('editing_existing_video', False):
            # Increment from current version
            current_version = await self._get_current_code_version(
                state.get('video_id', state.get('session_id', 'unknown')), 
                state
            )
            return (current_version or 0) + 1
        else:
            # New video, start with version 1
            return state.get('version', 1)
    
    async def _update_code_metadata(self, state: VideoGenerationState, 
                                  upload_results: Dict[str, Optional[str]]) -> None:
        """
        Update DynamoDB metadata with code upload results.
        
        Args:
            state: Current workflow state
            upload_results: Dictionary of upload results
        """
        if not self.metadata_service:
            logger.warning("Metadata service not available, skipping code metadata update")
            return
        
        try:
            video_id = state.get('video_id', state.get('session_id', 'unknown'))
            
            # Filter successful uploads
            successful_uploads = {k: v for k, v in upload_results.items() if v is not None}
            
            if not successful_uploads:
                logger.warning("No successful code uploads to record in metadata")
                return
            
            # Prepare metadata update
            new_version = state.get('current_version', state.get('version', 1))
            
            metadata_update = {
                'current_version_id': f"v{new_version}",
                'code_s3_paths': successful_uploads,
                'last_edited_timestamp': datetime.utcnow().isoformat(),
                'code_upload_completed_at': datetime.utcnow().isoformat(),
                'status': 'code_ready',
                'code_upload_summary': {
                    'successful_uploads': len(successful_uploads),
                    'total_attempts': len(upload_results),
                    'success_rate': len(successful_uploads) / len(upload_results) * 100,
                    'version': new_version
                }
            }
            
            # Set main code path (use main if available, otherwise first successful upload)
            if 'main' in successful_uploads:
                metadata_update['s3_path_code'] = successful_uploads['main']
            elif successful_uploads:
                metadata_update['s3_path_code'] = list(successful_uploads.values())[0]
            
            # Add scene information
            if len(successful_uploads) > 1 or (len(successful_uploads) == 1 and 'main' not in successful_uploads):
                metadata_update['scene_code_count'] = len([k for k in successful_uploads.keys() if k != 'main'])
                metadata_update['scene_codes_available'] = [k for k in successful_uploads.keys() if k != 'main']
            
            # Update metadata in DynamoDB
            success = await self.metadata_service.operations.update_video_metadata(
                video_id=video_id,
                metadata=metadata_update
            )
            
            if success:
                self.log_agent_action("code_metadata_updated", {
                    'video_id': video_id,
                    'version': new_version,
                    'successful_uploads': len(successful_uploads)
                })
            else:
                logger.error(f"Failed to update code metadata for video: {video_id}")
                
        except Exception as e:
            logger.error(f"Error updating code metadata: {e}")
            # Don't raise - metadata update failure shouldn't stop the workflow
    
    async def _handle_code_download_failure(self, error: Exception, 
                                          state: VideoGenerationState) -> Command:
        """
        Handle code download failures with retry strategies.
        
        Args:
            error: Download error that occurred
            state: Current workflow state
            
        Returns:
            Command for error handling
        """
        error_type = type(error).__name__
        
        self.log_agent_action("handling_code_download_failure", {
            'error_type': error_type,
            'error_message': str(error),
            'video_id': state.get('video_id')
        })
        
        # Check if we should escalate to human intervention
        if self.should_escalate_to_human(state):
            return self.create_human_intervention_command(
                context=f"Failed to download existing code for video '{state.get('video_id', '')}': {str(error)}",
                options=[
                    "Continue without existing code",
                    "Retry download",
                    "Manual code provision",
                    "Switch to local-only mode"
                ],
                state=state
            )
        
        # Handle error through base class
        return await self.handle_error(error, state)
    
    async def _handle_code_upload_failure(self, error: Exception, 
                                        state: VideoGenerationState) -> Command:
        """
        Handle code upload failures with retry strategies.
        
        Args:
            error: Upload error that occurred
            state: Current workflow state
            
        Returns:
            Command for error handling
        """
        error_type = type(error).__name__
        
        self.log_agent_action("handling_code_upload_failure", {
            'error_type': error_type,
            'error_message': str(error),
            'retryable': isinstance(error, AWSRetryableError)
        })
        
        # Check if this is a retryable error
        if isinstance(error, AWSRetryableError):
            retry_count = state.get('code_upload_retry_count', 0)
            max_retries = self.aws_config.max_retries if self.aws_config else 3
            
            if retry_count < max_retries:
                # Retry upload with exponential backoff
                delay = (2 ** retry_count) + (random.random() * 0.1)
                
                self.log_agent_action("retrying_code_upload", {
                    'retry_count': retry_count + 1,
                    'max_retries': max_retries,
                    'delay_seconds': delay
                })
                
                await asyncio.sleep(delay)
                
                # Update retry count and retry
                retry_state = state.copy()
                retry_state['code_upload_retry_count'] = retry_count + 1
                
                return await self.execute(retry_state)
        
        # Check if we should escalate to human intervention
        if self.should_escalate_to_human(state):
            return self.create_human_intervention_command(
                context=f"Code upload failed for video '{state.get('video_id', '')}': {str(error)}",
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
    
    def get_code_management_status(self, state: VideoGenerationState) -> Dict[str, Any]:
        """
        Get comprehensive code management status information.
        
        Args:
            state: Current workflow state
            
        Returns:
            Code management status information
        """
        base_status = self.get_code_generation_status(state)
        
        code_status = {
            'aws_enabled': self.aws_enabled,
            'code_service_available': self.s3_code_service is not None,
            'metadata_service_available': self.metadata_service is not None,
            'code_upload_status': state.get('code_upload_status', 'not_started'),
            'code_upload_error': state.get('code_upload_error'),
            'code_upload_timestamp': state.get('code_upload_timestamp'),
            'code_upload_results': state.get('code_upload_results', {}),
            'code_upload_retry_count': state.get('code_upload_retry_count', 0),
            'existing_code_available': bool(state.get('existing_code')),
            'current_version': state.get('current_version', 1),
            'cached_downloads': len(self.download_cache),
            'version_tracking': dict(self.code_versions)
        }
        
        # Combine with base code generation status
        base_status.update(code_status)
        return base_status
    
    async def get_code_history(self, video_id: str) -> List[Dict[str, Any]]:
        """
        Get code version history for a video.
        
        Args:
            video_id: Video identifier
            
        Returns:
            List of code version information
        """
        if not self.metadata_service:
            return []
        
        try:
            # Get metadata which should contain version history
            metadata = await self.metadata_service.operations.get_video_metadata(video_id)
            if not metadata:
                return []
            
            # Extract version information
            history = []
            current_version = metadata.get('current_version_id', 'v1')
            
            # For now, return basic version info
            # In a full implementation, we'd query S3 for all versions
            if current_version:
                version_num = int(current_version[1:]) if current_version.startswith('v') else int(current_version)
                
                for v in range(1, version_num + 1):
                    history.append({
                        'version': v,
                        'version_id': f'v{v}',
                        'is_current': v == version_num,
                        'upload_timestamp': metadata.get('code_upload_completed_at'),
                        'status': 'available'
                    })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting code history for {video_id}: {e}")
            return []
    
    async def cleanup_code_resources(self):
        """Clean up code management related resources."""
        # Clear caches
        self.download_cache.clear()
        self.code_versions.clear()
        
        # Cleanup base code generator resources
        if hasattr(super(), 'cleanup_cache'):
            super().cleanup_cache()
        
        # Cleanup AWS service resources
        if self.metadata_service and hasattr(self.metadata_service, 'executor'):
            self.metadata_service.executor.shutdown(wait=False)
    
    def __del__(self):
        """Cleanup resources when agent is destroyed."""
        try:
            # Fallback cleanup - only clear if attributes exist
            if hasattr(self, 'download_cache'):
                self.download_cache.clear()
            if hasattr(self, 'code_versions'):
                self.code_versions.clear()
        except Exception:
            pass  # Ignore cleanup errors during destruction
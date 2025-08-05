"""
AWS MediaConvert Transcoding Service

Handles video transcoding using AWS Elemental MediaConvert for adaptive bitrate streaming
with HLS and DASH output formats and multiple quality levels.
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum

import boto3
from botocore.exceptions import ClientError

from .config import AWSConfig
from .credentials import AWSCredentialsManager
from .exceptions import AWSMediaConvertError, AWSRetryableError
# Metadata service import is optional to avoid circular dependencies
try:
    from .metadata_service import MetadataService
except ImportError:
    MetadataService = None

logger = logging.getLogger(__name__)


class TranscodingStatus(Enum):
    """Transcoding job status enumeration."""
    SUBMITTED = "SUBMITTED"
    PROGRESSING = "PROGRESSING" 
    COMPLETE = "COMPLETE"
    CANCELED = "CANCELED"
    ERROR = "ERROR"


class OutputFormat(Enum):
    """Supported output formats."""
    HLS = "HLS"
    DASH = "DASH"
    MP4 = "MP4"


class QualityLevel(Enum):
    """Video quality levels."""
    HD_1080P = "1080p"
    HD_720P = "720p"
    SD_480P = "480p"
    SD_360P = "360p"


class TranscodingJobConfig:
    """Configuration for a transcoding job."""
    
    def __init__(self, 
                 input_s3_path: str,
                 output_s3_prefix: str,
                 video_id: str,
                 project_id: str,
                 output_formats: List[OutputFormat] = None,
                 quality_levels: List[QualityLevel] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.input_s3_path = input_s3_path
        self.output_s3_prefix = output_s3_prefix
        self.video_id = video_id
        self.project_id = project_id
        self.output_formats = output_formats or [OutputFormat.HLS, OutputFormat.DASH]
        self.quality_levels = quality_levels or [QualityLevel.HD_1080P, QualityLevel.HD_720P, QualityLevel.SD_480P]
        self.metadata = metadata or {}


class TranscodingJobResult:
    """Result of a transcoding job."""
    
    def __init__(self,
                 job_id: str,
                 status: TranscodingStatus,
                 input_s3_path: str,
                 output_paths: Dict[str, List[str]] = None,
                 created_at: Optional[datetime] = None,
                 completed_at: Optional[datetime] = None,
                 error_message: Optional[str] = None,
                 progress_percentage: Optional[int] = None):
        self.job_id = job_id
        self.status = status
        self.input_s3_path = input_s3_path
        self.output_paths = output_paths or {}
        self.created_at = created_at
        self.completed_at = completed_at
        self.error_message = error_message
        self.progress_percentage = progress_percentage


class MediaConvertService:
    """
    AWS MediaConvert service for video transcoding with adaptive bitrate streaming.
    
    Provides functionality for:
    - Creating transcoding jobs with multiple quality levels
    - Supporting HLS and DASH output formats
    - Monitoring job progress and status
    - Managing transcoded file paths in metadata
    """
    
    def __init__(self, config: AWSConfig, credentials_manager: AWSCredentialsManager):
        """
        Initialize MediaConvert service.
        
        Args:
            config: AWS configuration
            credentials_manager: AWS credentials manager
        """
        self.config = config
        self.credentials_manager = credentials_manager
        
        # Initialize MediaConvert client
        try:
            self.mediaconvert_client = self.credentials_manager.get_client('mediaconvert')
            
            # Get MediaConvert endpoint if not configured
            if not self.config.mediaconvert_endpoint:
                self._discover_endpoint()
            else:
                # Create client with custom endpoint
                self.mediaconvert_client = self.credentials_manager.get_client(
                    'mediaconvert',
                    endpoint_url=self.config.mediaconvert_endpoint
                )
            
            logger.info(f"MediaConvert service initialized with endpoint: {self.config.mediaconvert_endpoint}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MediaConvert client: {e}")
            raise AWSMediaConvertError(f"MediaConvert initialization failed: {str(e)}") from e
        
        # Initialize metadata service for updating transcoding results
        self.metadata_service = MetadataService(config) if MetadataService else None
    
    def _discover_endpoint(self) -> None:
        """Discover MediaConvert endpoint for the region."""
        try:
            response = self.mediaconvert_client.describe_endpoints()
            if response['Endpoints']:
                endpoint_url = response['Endpoints'][0]['Url']
                self.config.mediaconvert_endpoint = endpoint_url
                
                # Recreate client with discovered endpoint
                self.mediaconvert_client = self.credentials_manager.get_client(
                    'mediaconvert',
                    endpoint_url=endpoint_url
                )
                
                logger.info(f"Discovered MediaConvert endpoint: {endpoint_url}")
            else:
                raise AWSMediaConvertError("No MediaConvert endpoints found")
                
        except ClientError as e:
            raise AWSMediaConvertError(f"Failed to discover MediaConvert endpoint: {str(e)}") from e
    
    async def create_transcoding_job(self, job_config: TranscodingJobConfig,
                                   progress_callback: Optional[Callable] = None) -> str:
        """
        Create a new transcoding job for adaptive bitrate streaming.
        
        Args:
            job_config: Transcoding job configuration
            progress_callback: Optional callback for progress updates
            
        Returns:
            Job ID of the created transcoding job
            
        Raises:
            AWSMediaConvertError: If job creation fails
        """
        try:
            # Build job settings
            job_settings = self._build_job_settings(job_config)
            
            # Create the job
            response = self.mediaconvert_client.create_job(**job_settings)
            job_id = response['Job']['Id']
            
            logger.info(f"Created MediaConvert job {job_id} for video {job_config.video_id}")
            
            # Update metadata with job information
            await self._update_job_metadata(job_config, job_id, TranscodingStatus.SUBMITTED)
            
            # Start monitoring if callback provided
            if progress_callback:
                asyncio.create_task(self._monitor_job_progress(job_id, progress_callback))
            
            return job_id
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            
            logger.error(f"MediaConvert job creation failed: {error_code} - {error_message}")
            raise AWSMediaConvertError(
                f"Failed to create transcoding job: {error_message}",
                operation="create_job"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error creating transcoding job: {e}")
            raise AWSMediaConvertError(f"Transcoding job creation failed: {str(e)}") from e
    
    def _build_job_settings(self, job_config: TranscodingJobConfig) -> Dict[str, Any]:
        """
        Build MediaConvert job settings for adaptive bitrate streaming.
        
        Args:
            job_config: Transcoding job configuration
            
        Returns:
            MediaConvert job settings dictionary
        """
        # Base job settings
        job_settings = {
            "Role": self.config.mediaconvert_role_arn,
            "Settings": {
                "Inputs": [{
                    "FileInput": job_config.input_s3_path,
                    "AudioSelectors": {
                        "Audio Selector 1": {
                            "DefaultSelection": "DEFAULT"
                        }
                    },
                    "VideoSelector": {
                        "ColorSpace": "FOLLOW"
                    }
                }],
                "OutputGroups": []
            },
            "UserMetadata": {
                "video_id": job_config.video_id,
                "project_id": job_config.project_id,
                "created_by": "langgraph-agents",
                **job_config.metadata
            }
        }
        
        # Add output groups based on requested formats
        for output_format in job_config.output_formats:
            if output_format == OutputFormat.HLS:
                job_settings["Settings"]["OutputGroups"].append(
                    self._build_hls_output_group(job_config)
                )
            elif output_format == OutputFormat.DASH:
                job_settings["Settings"]["OutputGroups"].append(
                    self._build_dash_output_group(job_config)
                )
            elif output_format == OutputFormat.MP4:
                job_settings["Settings"]["OutputGroups"].append(
                    self._build_mp4_output_group(job_config)
                )
        
        return job_settings
    
    def _build_hls_output_group(self, job_config: TranscodingJobConfig) -> Dict[str, Any]:
        """Build HLS output group configuration."""
        hls_group = {
            "Name": "HLS",
            "OutputGroupSettings": {
                "Type": "HLS_GROUP_SETTINGS",
                "HlsGroupSettings": {
                    "Destination": f"{job_config.output_s3_prefix}/hls/",
                    "SegmentLength": 10,
                    "MinSegmentLength": 0,
                    "DirectoryStructure": "SINGLE_DIRECTORY",
                    "ManifestDurationFormat": "INTEGER",
                    "StreamInfResolution": "INCLUDE",
                    "ClientCache": "ENABLED",
                    "CaptionLanguageSetting": "OMIT",
                    "ManifestCompression": "NONE",
                    "CodecSpecification": "RFC_4281",
                    "OutputSelection": "MANIFESTS_AND_SEGMENTS",
                    "ProgramDateTime": "EXCLUDE",
                    "SegmentControl": "SEGMENTED_FILES"
                }
            },
            "Outputs": []
        }
        
        # Add outputs for each quality level
        for quality in job_config.quality_levels:
            hls_group["Outputs"].append(self._build_video_output(quality, "HLS"))
        
        # Add audio-only output
        hls_group["Outputs"].append(self._build_audio_output("HLS"))
        
        return hls_group
    
    def _build_dash_output_group(self, job_config: TranscodingJobConfig) -> Dict[str, Any]:
        """Build DASH output group configuration."""
        dash_group = {
            "Name": "DASH",
            "OutputGroupSettings": {
                "Type": "DASH_ISO_GROUP_SETTINGS",
                "DashIsoGroupSettings": {
                    "Destination": f"{job_config.output_s3_prefix}/dash/",
                    "SegmentLength": 30,
                    "FragmentLength": 2,
                    "SegmentControl": "SEGMENTED_FILES",
                    "MpdProfile": "MAIN_PROFILE"
                }
            },
            "Outputs": []
        }
        
        # Add outputs for each quality level
        for quality in job_config.quality_levels:
            dash_group["Outputs"].append(self._build_video_output(quality, "DASH"))
        
        # Add audio-only output
        dash_group["Outputs"].append(self._build_audio_output("DASH"))
        
        return dash_group
    
    def _build_mp4_output_group(self, job_config: TranscodingJobConfig) -> Dict[str, Any]:
        """Build MP4 output group configuration."""
        mp4_group = {
            "Name": "MP4",
            "OutputGroupSettings": {
                "Type": "FILE_GROUP_SETTINGS",
                "FileGroupSettings": {
                    "Destination": f"{job_config.output_s3_prefix}/mp4/"
                }
            },
            "Outputs": []
        }
        
        # Add outputs for each quality level
        for quality in job_config.quality_levels:
            mp4_group["Outputs"].append(self._build_video_output(quality, "MP4"))
        
        return mp4_group
    
    def _build_video_output(self, quality: QualityLevel, container_type: str) -> Dict[str, Any]:
        """Build video output configuration for a specific quality level."""
        # Quality level settings
        quality_settings = {
            QualityLevel.HD_1080P: {
                "width": 1920,
                "height": 1080,
                "bitrate": 5000000,
                "name_modifier": "_1080p"
            },
            QualityLevel.HD_720P: {
                "width": 1280,
                "height": 720,
                "bitrate": 3000000,
                "name_modifier": "_720p"
            },
            QualityLevel.SD_480P: {
                "width": 854,
                "height": 480,
                "bitrate": 1500000,
                "name_modifier": "_480p"
            },
            QualityLevel.SD_360P: {
                "width": 640,
                "height": 360,
                "bitrate": 800000,
                "name_modifier": "_360p"
            }
        }
        
        settings = quality_settings[quality]
        
        output = {
            "NameModifier": settings["name_modifier"],
            "VideoDescription": {
                "Width": settings["width"],
                "Height": settings["height"],
                "ScalingBehavior": "DEFAULT",
                "TimecodeInsertion": "DISABLED",
                "AntiAlias": "ENABLED",
                "Sharpness": 50,
                "CodecSettings": {
                    "Codec": "H_264",
                    "H264Settings": {
                        "InterlaceMode": "PROGRESSIVE",
                        "NumberReferenceFrames": 3,
                        "Syntax": "DEFAULT",
                        "Softness": 0,
                        "GopClosedCadence": 1,
                        "GopSize": 90,
                        "Slices": 1,
                        "GopBReference": "DISABLED",
                        "SlowPal": "DISABLED",
                        "SpatialAdaptiveQuantization": "ENABLED",
                        "TemporalAdaptiveQuantization": "ENABLED",
                        "FlickerAdaptiveQuantization": "DISABLED",
                        "EntropyEncoding": "CABAC",
                        "Bitrate": settings["bitrate"],
                        "FramerateControl": "INITIALIZE_FROM_SOURCE",
                        "RateControlMode": "CBR",
                        "CodecProfile": "MAIN",
                        "Telecine": "NONE",
                        "MinIInterval": 0,
                        "AdaptiveQuantization": "HIGH",
                        "CodecLevel": "AUTO",
                        "FieldEncoding": "PAFF",
                        "SceneChangeDetect": "ENABLED",
                        "QualityTuningLevel": "SINGLE_PASS",
                        "FramerateConversionAlgorithm": "DUPLICATE_DROP",
                        "UnregisteredSeiTimecode": "DISABLED",
                        "GopSizeUnits": "FRAMES",
                        "ParControl": "INITIALIZE_FROM_SOURCE",
                        "NumberBFramesBetweenReferenceFrames": 2,
                        "RepeatPps": "DISABLED"
                    }
                }
            },
            "AudioDescriptions": [{
                "AudioTypeControl": "FOLLOW_INPUT",
                "AudioSourceName": "Audio Selector 1",
                "CodecSettings": {
                    "Codec": "AAC",
                    "AacSettings": {
                        "AudioDescriptionBroadcasterMix": "NORMAL",
                        "Bitrate": 96000,
                        "RateControlMode": "CBR",
                        "CodecProfile": "LC",
                        "CodingMode": "CODING_MODE_2_0",
                        "RawFormat": "NONE",
                        "SampleRate": 48000,
                        "Specification": "MPEG4"
                    }
                },
                "LanguageCodeControl": "FOLLOW_INPUT"
            }]
        }
        
        # Container-specific settings
        if container_type == "MP4":
            output["ContainerSettings"] = {
                "Container": "MP4",
                "Mp4Settings": {
                    "CslgAtom": "INCLUDE",
                    "FreeSpaceBox": "EXCLUDE",
                    "MoovPlacement": "PROGRESSIVE_DOWNLOAD"
                }
            }
        
        return output
    
    def _build_audio_output(self, container_type: str) -> Dict[str, Any]:
        """Build audio-only output configuration."""
        output = {
            "NameModifier": "_audio",
            "AudioDescriptions": [{
                "AudioTypeControl": "FOLLOW_INPUT",
                "AudioSourceName": "Audio Selector 1",
                "CodecSettings": {
                    "Codec": "AAC",
                    "AacSettings": {
                        "AudioDescriptionBroadcasterMix": "NORMAL",
                        "Bitrate": 128000,
                        "RateControlMode": "CBR",
                        "CodecProfile": "LC",
                        "CodingMode": "CODING_MODE_2_0",
                        "RawFormat": "NONE",
                        "SampleRate": 48000,
                        "Specification": "MPEG4"
                    }
                },
                "LanguageCodeControl": "FOLLOW_INPUT"
            }]
        }
        
        return output
    
    async def get_job_status(self, job_id: str) -> TranscodingJobResult:
        """
        Get the status of a transcoding job.
        
        Args:
            job_id: MediaConvert job ID
            
        Returns:
            TranscodingJobResult with current job status
            
        Raises:
            AWSMediaConvertError: If status retrieval fails
        """
        try:
            response = self.mediaconvert_client.get_job(Id=job_id)
            job = response['Job']
            
            # Parse job status
            status = TranscodingStatus(job['Status'])
            
            # Extract timestamps
            created_at = job.get('CreatedAt')
            completed_at = job.get('FinishedAt')
            
            # Extract progress
            progress_percentage = None
            if 'JobPercentComplete' in job:
                progress_percentage = job['JobPercentComplete']
            
            # Extract error message if failed
            error_message = None
            if status == TranscodingStatus.ERROR and 'ErrorMessage' in job:
                error_message = job['ErrorMessage']
            
            # Extract output paths
            output_paths = {}
            if status == TranscodingStatus.COMPLETE and 'OutputGroupDetails' in job:
                output_paths = self._extract_output_paths(job['OutputGroupDetails'])
            
            return TranscodingJobResult(
                job_id=job_id,
                status=status,
                input_s3_path=job['Settings']['Inputs'][0]['FileInput'],
                output_paths=output_paths,
                created_at=created_at,
                completed_at=completed_at,
                error_message=error_message,
                progress_percentage=progress_percentage
            )
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NotFoundException':
                raise AWSMediaConvertError(f"Transcoding job not found: {job_id}", job_id=job_id)
            else:
                raise AWSMediaConvertError(
                    f"Failed to get job status: {e.response['Error']['Message']}",
                    job_id=job_id,
                    operation="get_job_status"
                ) from e
        except Exception as e:
            raise AWSMediaConvertError(f"Failed to get job status: {str(e)}", job_id=job_id) from e
    
    def _extract_output_paths(self, output_group_details: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Extract output file paths from job details."""
        output_paths = {}
        
        for group in output_group_details:
            group_name = group.get('Type', 'unknown').lower()
            paths = []
            
            if 'OutputDetails' in group:
                for output in group['OutputDetails']:
                    if 'OutputFilePaths' in output:
                        paths.extend(output['OutputFilePaths'])
            
            if paths:
                output_paths[group_name] = paths
        
        return output_paths
    
    async def _monitor_job_progress(self, job_id: str, progress_callback: Callable) -> None:
        """
        Monitor job progress and call progress callback.
        
        Args:
            job_id: MediaConvert job ID
            progress_callback: Callback function for progress updates
        """
        try:
            while True:
                job_result = await self.get_job_status(job_id)
                
                # Call progress callback
                await progress_callback(job_result)
                
                # Check if job is complete
                if job_result.status in [TranscodingStatus.COMPLETE, TranscodingStatus.ERROR, TranscodingStatus.CANCELED]:
                    break
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
        except Exception as e:
            logger.error(f"Error monitoring job {job_id}: {e}")
    
    async def wait_for_job_completion(self, job_id: str, timeout_minutes: int = 60) -> TranscodingJobResult:
        """
        Wait for a transcoding job to complete.
        
        Args:
            job_id: MediaConvert job ID
            timeout_minutes: Maximum time to wait in minutes
            
        Returns:
            TranscodingJobResult with final job status
            
        Raises:
            AWSMediaConvertError: If job fails or times out
        """
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        try:
            while True:
                job_result = await self.get_job_status(job_id)
                
                # Check if job is complete
                if job_result.status == TranscodingStatus.COMPLETE:
                    logger.info(f"Transcoding job {job_id} completed successfully")
                    
                    # Update metadata with transcoded paths
                    await self._update_completion_metadata(job_result)
                    
                    return job_result
                
                elif job_result.status == TranscodingStatus.ERROR:
                    error_msg = job_result.error_message or "Unknown error"
                    logger.error(f"Transcoding job {job_id} failed: {error_msg}")
                    raise AWSMediaConvertError(f"Transcoding job failed: {error_msg}", job_id=job_id)
                
                elif job_result.status == TranscodingStatus.CANCELED:
                    logger.warning(f"Transcoding job {job_id} was canceled")
                    raise AWSMediaConvertError("Transcoding job was canceled", job_id=job_id)
                
                # Check timeout
                if time.time() - start_time > timeout_seconds:
                    raise AWSMediaConvertError(
                        f"Transcoding job timed out after {timeout_minutes} minutes",
                        job_id=job_id
                    )
                
                # Wait before next check
                await asyncio.sleep(30)
                
        except AWSMediaConvertError:
            raise
        except Exception as e:
            raise AWSMediaConvertError(f"Error waiting for job completion: {str(e)}", job_id=job_id) from e
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a transcoding job.
        
        Args:
            job_id: MediaConvert job ID
            
        Returns:
            True if job was canceled successfully
            
        Raises:
            AWSMediaConvertError: If cancellation fails
        """
        try:
            self.mediaconvert_client.cancel_job(Id=job_id)
            logger.info(f"Canceled transcoding job: {job_id}")
            
            # Update metadata
            await self._update_job_metadata_status(job_id, TranscodingStatus.CANCELED)
            
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ConflictException':
                # Job might already be complete or canceled
                logger.warning(f"Cannot cancel job {job_id}: {e.response['Error']['Message']}")
                return False
            else:
                raise AWSMediaConvertError(
                    f"Failed to cancel job: {e.response['Error']['Message']}",
                    job_id=job_id,
                    operation="cancel_job"
                ) from e
        except Exception as e:
            raise AWSMediaConvertError(f"Failed to cancel job: {str(e)}", job_id=job_id) from e
    
    async def list_jobs(self, status: Optional[TranscodingStatus] = None, 
                       max_results: int = 20) -> List[TranscodingJobResult]:
        """
        List transcoding jobs.
        
        Args:
            status: Optional status filter
            max_results: Maximum number of results
            
        Returns:
            List of TranscodingJobResult objects
        """
        try:
            params = {'MaxResults': max_results}
            if status:
                params['Status'] = status.value
            
            response = self.mediaconvert_client.list_jobs(**params)
            
            jobs = []
            for job in response.get('Jobs', []):
                job_result = TranscodingJobResult(
                    job_id=job['Id'],
                    status=TranscodingStatus(job['Status']),
                    input_s3_path=job['Settings']['Inputs'][0]['FileInput'],
                    created_at=job.get('CreatedAt'),
                    completed_at=job.get('FinishedAt'),
                    progress_percentage=job.get('JobPercentComplete')
                )
                jobs.append(job_result)
            
            return jobs
            
        except Exception as e:
            raise AWSMediaConvertError(f"Failed to list jobs: {str(e)}") from e
    
    async def _update_job_metadata(self, job_config: TranscodingJobConfig, 
                                 job_id: str, status: TranscodingStatus) -> None:
        """Update video metadata with transcoding job information."""
        if not self.metadata_service:
            logger.warning("Metadata service not available, skipping metadata update")
            return
            
        try:
            metadata_update = {
                'transcoding_job_id': job_id,
                'transcoding_status': status.value,
                'transcoding_started_at': datetime.utcnow().isoformat(),
                'transcoding_input_path': job_config.input_s3_path,
                'transcoding_output_prefix': job_config.output_s3_prefix,
                'transcoding_formats': [fmt.value for fmt in job_config.output_formats],
                'transcoding_quality_levels': [qual.value for qual in job_config.quality_levels]
            }
            
            await self.metadata_service.update_video_metadata(job_config.video_id, metadata_update)
            logger.debug(f"Updated metadata for video {job_config.video_id} with job {job_id}")
            
        except Exception as e:
            logger.error(f"Failed to update job metadata: {e}")
    
    async def _update_job_metadata_status(self, job_id: str, status: TranscodingStatus) -> None:
        """Update transcoding status in metadata."""
        try:
            # This is a simplified update - in practice you'd need to find the video_id from job metadata
            # For now, we'll log the status change
            logger.info(f"Transcoding job {job_id} status changed to {status.value}")
            
        except Exception as e:
            logger.error(f"Failed to update job status metadata: {e}")
    
    async def _update_completion_metadata(self, job_result: TranscodingJobResult) -> None:
        """Update metadata with transcoded file paths upon completion."""
        try:
            metadata_update = {
                'transcoding_status': job_result.status.value,
                'transcoding_completed_at': datetime.utcnow().isoformat(),
                'transcoded_output_paths': job_result.output_paths,
                'transcoding_progress': 100
            }
            
            # Extract video_id from job metadata if available
            # This would need to be implemented based on how you store the mapping
            logger.info(f"Transcoding job {job_result.job_id} completed with outputs: {job_result.output_paths}")
            
        except Exception as e:
            logger.error(f"Failed to update completion metadata: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on MediaConvert service.
        
        Returns:
            Health check results
        """
        try:
            # Test basic connectivity
            response = self.mediaconvert_client.list_jobs(MaxResults=1)
            
            return {
                'status': 'healthy',
                'endpoint': self.config.mediaconvert_endpoint,
                'region': self.config.region,
                'role_arn': self.config.mediaconvert_role_arn,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'endpoint': self.config.mediaconvert_endpoint,
                'region': self.config.region,
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def __repr__(self) -> str:
        """String representation of MediaConvert service."""
        return (
            f"MediaConvertService(region='{self.config.region}', "
            f"endpoint='{self.config.mediaconvert_endpoint}', "
            f"role_arn='{self.config.mediaconvert_role_arn}')"
        )
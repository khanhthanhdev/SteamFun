"""
Unit tests for MediaConvert service.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from app.core.aws.mediaconvert_service import (
    MediaConvertService,
    TranscodingJobConfig,
    TranscodingJobResult,
    TranscodingStatus,
    OutputFormat,
    QualityLevel
)
from app.core.aws.config import AWSConfig
from app.core.aws.credentials import AWSCredentialsManager
from app.core.aws.exceptions import AWSMediaConvertError


@pytest.fixture
def aws_config():
    """Create test AWS configuration."""
    config = AWSConfig()
    config.region = 'us-east-1'
    config.mediaconvert_role_arn = 'arn:aws:iam::123456789012:role/MediaConvertRole'
    config.mediaconvert_endpoint = 'https://test.mediaconvert.us-east-1.amazonaws.com'
    config.enable_transcoding = True
    return config


@pytest.fixture
def credentials_manager(aws_config):
    """Create mock credentials manager."""
    mock_manager = Mock(spec=AWSCredentialsManager)
    mock_client = Mock()
    mock_manager.get_client.return_value = mock_client
    return mock_manager


@pytest.fixture
def transcoding_job_config():
    """Create test transcoding job configuration."""
    return TranscodingJobConfig(
        input_s3_path="s3://test-bucket/input/video.mp4",
        output_s3_prefix="s3://test-bucket/output/transcoded",
        video_id="test-video-123",
        project_id="test-project-456",
        output_formats=[OutputFormat.HLS, OutputFormat.DASH],
        quality_levels=[QualityLevel.HD_1080P, QualityLevel.HD_720P, QualityLevel.SD_480P]
    )


class TestMediaConvertService:
    """Test cases for MediaConvert service."""
    
    def test_init_with_endpoint(self, aws_config, credentials_manager):
        """Test MediaConvert service initialization with endpoint."""
        service = MediaConvertService(aws_config, credentials_manager)
        
        assert service.config == aws_config
        assert service.credentials_manager == credentials_manager
        credentials_manager.get_client.assert_called_with(
            'mediaconvert',
            endpoint_url=aws_config.mediaconvert_endpoint
        )
    
    def test_init_without_endpoint(self, aws_config, credentials_manager):
        """Test MediaConvert service initialization without endpoint."""
        aws_config.mediaconvert_endpoint = None
        
        # Mock the describe_endpoints response
        mock_client = Mock()
        mock_client.describe_endpoints.return_value = {
            'Endpoints': [{'Url': 'https://discovered.mediaconvert.us-east-1.amazonaws.com'}]
        }
        credentials_manager.get_client.return_value = mock_client
        
        service = MediaConvertService(aws_config, credentials_manager)
        
        # Should discover endpoint and recreate client
        assert aws_config.mediaconvert_endpoint == 'https://discovered.mediaconvert.us-east-1.amazonaws.com'
        assert credentials_manager.get_client.call_count == 2  # Once for discovery, once with endpoint
    
    def test_build_job_settings(self, aws_config, credentials_manager, transcoding_job_config):
        """Test building MediaConvert job settings."""
        service = MediaConvertService(aws_config, credentials_manager)
        
        job_settings = service._build_job_settings(transcoding_job_config)
        
        # Verify basic structure
        assert job_settings['Role'] == aws_config.mediaconvert_role_arn
        assert 'Settings' in job_settings
        assert 'Inputs' in job_settings['Settings']
        assert 'OutputGroups' in job_settings['Settings']
        
        # Verify input configuration
        inputs = job_settings['Settings']['Inputs']
        assert len(inputs) == 1
        assert inputs[0]['FileInput'] == transcoding_job_config.input_s3_path
        
        # Verify output groups (HLS and DASH)
        output_groups = job_settings['Settings']['OutputGroups']
        assert len(output_groups) == 2
        
        # Find HLS and DASH groups
        hls_group = next((g for g in output_groups if g['Name'] == 'HLS'), None)
        dash_group = next((g for g in output_groups if g['Name'] == 'DASH'), None)
        
        assert hls_group is not None
        assert dash_group is not None
        
        # Verify HLS configuration
        assert hls_group['OutputGroupSettings']['Type'] == 'HLS_GROUP_SETTINGS'
        assert hls_group['OutputGroupSettings']['HlsGroupSettings']['Destination'] == f"{transcoding_job_config.output_s3_prefix}/hls/"
        
        # Verify DASH configuration
        assert dash_group['OutputGroupSettings']['Type'] == 'DASH_ISO_GROUP_SETTINGS'
        assert dash_group['OutputGroupSettings']['DashIsoGroupSettings']['Destination'] == f"{transcoding_job_config.output_s3_prefix}/dash/"
        
        # Verify outputs for quality levels (3 video + 1 audio for each format)
        assert len(hls_group['Outputs']) == 4  # 3 video qualities + 1 audio
        assert len(dash_group['Outputs']) == 4  # 3 video qualities + 1 audio
    
    def test_build_video_output_1080p(self, aws_config, credentials_manager):
        """Test building 1080p video output configuration."""
        service = MediaConvertService(aws_config, credentials_manager)
        
        output = service._build_video_output(QualityLevel.HD_1080P, "HLS")
        
        assert output['NameModifier'] == '_1080p'
        assert output['VideoDescription']['Width'] == 1920
        assert output['VideoDescription']['Height'] == 1080
        assert output['VideoDescription']['CodecSettings']['H264Settings']['Bitrate'] == 5000000
    
    def test_build_video_output_720p(self, aws_config, credentials_manager):
        """Test building 720p video output configuration."""
        service = MediaConvertService(aws_config, credentials_manager)
        
        output = service._build_video_output(QualityLevel.HD_720P, "DASH")
        
        assert output['NameModifier'] == '_720p'
        assert output['VideoDescription']['Width'] == 1280
        assert output['VideoDescription']['Height'] == 720
        assert output['VideoDescription']['CodecSettings']['H264Settings']['Bitrate'] == 3000000
    
    def test_build_video_output_480p(self, aws_config, credentials_manager):
        """Test building 480p video output configuration."""
        service = MediaConvertService(aws_config, credentials_manager)
        
        output = service._build_video_output(QualityLevel.SD_480P, "MP4")
        
        assert output['NameModifier'] == '_480p'
        assert output['VideoDescription']['Width'] == 854
        assert output['VideoDescription']['Height'] == 480
        assert output['VideoDescription']['CodecSettings']['H264Settings']['Bitrate'] == 1500000
        
        # MP4 should have container settings
        assert 'ContainerSettings' in output
        assert output['ContainerSettings']['Container'] == 'MP4'
    
    def test_build_audio_output(self, aws_config, credentials_manager):
        """Test building audio output configuration."""
        service = MediaConvertService(aws_config, credentials_manager)
        
        output = service._build_audio_output("HLS")
        
        assert output['NameModifier'] == '_audio'
        assert len(output['AudioDescriptions']) == 1
        
        audio_desc = output['AudioDescriptions'][0]
        assert audio_desc['CodecSettings']['Codec'] == 'AAC'
        assert audio_desc['CodecSettings']['AacSettings']['Bitrate'] == 128000
    
    @pytest.mark.asyncio
    async def test_create_transcoding_job_success(self, aws_config, credentials_manager, transcoding_job_config):
        """Test successful transcoding job creation."""
        mock_client = Mock()
        mock_client.create_job.return_value = {
            'Job': {'Id': 'test-job-123'}
        }
        credentials_manager.get_client.return_value = mock_client
        
        with patch('app.core.aws.mediaconvert_service.MetadataService'):
            service = MediaConvertService(aws_config, credentials_manager)
            service._update_job_metadata = AsyncMock()
            
            job_id = await service.create_transcoding_job(transcoding_job_config)
            
            assert job_id == 'test-job-123'
            mock_client.create_job.assert_called_once()
            service._update_job_metadata.assert_called_once_with(
                transcoding_job_config, 'test-job-123', TranscodingStatus.SUBMITTED
            )
    
    @pytest.mark.asyncio
    async def test_get_job_status_success(self, aws_config, credentials_manager):
        """Test successful job status retrieval."""
        mock_client = Mock()
        mock_client.get_job.return_value = {
            'Job': {
                'Id': 'test-job-123',
                'Status': 'PROGRESSING',
                'JobPercentComplete': 50,
                'CreatedAt': datetime.utcnow(),
                'Settings': {
                    'Inputs': [{'FileInput': 's3://test-bucket/input.mp4'}]
                }
            }
        }
        credentials_manager.get_client.return_value = mock_client
        
        with patch('app.core.aws.mediaconvert_service.MetadataService'):
            service = MediaConvertService(aws_config, credentials_manager)
            
            result = await service.get_job_status('test-job-123')
            
            assert result.job_id == 'test-job-123'
            assert result.status == TranscodingStatus.PROGRESSING
            assert result.progress_percentage == 50
            assert result.input_s3_path == 's3://test-bucket/input.mp4'
    
    @pytest.mark.asyncio
    async def test_get_job_status_not_found(self, aws_config, credentials_manager):
        """Test job status retrieval for non-existent job."""
        from botocore.exceptions import ClientError
        
        mock_client = Mock()
        mock_client.get_job.side_effect = ClientError(
            {'Error': {'Code': 'NotFoundException', 'Message': 'Job not found'}},
            'GetJob'
        )
        credentials_manager.get_client.return_value = mock_client
        
        with patch('app.core.aws.mediaconvert_service.MetadataService'):
            service = MediaConvertService(aws_config, credentials_manager)
            
            with pytest.raises(AWSMediaConvertError, match="Transcoding job not found"):
                await service.get_job_status('non-existent-job')
    
    @pytest.mark.asyncio
    async def test_cancel_job_success(self, aws_config, credentials_manager):
        """Test successful job cancellation."""
        mock_client = Mock()
        mock_client.cancel_job.return_value = {}
        credentials_manager.get_client.return_value = mock_client
        
        with patch('app.core.aws.mediaconvert_service.MetadataService'):
            service = MediaConvertService(aws_config, credentials_manager)
            service._update_job_metadata_status = AsyncMock()
            
            result = await service.cancel_job('test-job-123')
            
            assert result is True
            mock_client.cancel_job.assert_called_once_with(Id='test-job-123')
            service._update_job_metadata_status.assert_called_once_with(
                'test-job-123', TranscodingStatus.CANCELED
            )
    
    @pytest.mark.asyncio
    async def test_list_jobs(self, aws_config, credentials_manager):
        """Test listing transcoding jobs."""
        mock_client = Mock()
        mock_client.list_jobs.return_value = {
            'Jobs': [
                {
                    'Id': 'job-1',
                    'Status': 'COMPLETE',
                    'CreatedAt': datetime.utcnow(),
                    'Settings': {'Inputs': [{'FileInput': 's3://bucket/input1.mp4'}]}
                },
                {
                    'Id': 'job-2',
                    'Status': 'PROGRESSING',
                    'JobPercentComplete': 75,
                    'CreatedAt': datetime.utcnow(),
                    'Settings': {'Inputs': [{'FileInput': 's3://bucket/input2.mp4'}]}
                }
            ]
        }
        credentials_manager.get_client.return_value = mock_client
        
        with patch('app.core.aws.mediaconvert_service.MetadataService'):
            service = MediaConvertService(aws_config, credentials_manager)
            
            jobs = await service.list_jobs(TranscodingStatus.PROGRESSING, 10)
            
            assert len(jobs) == 2
            assert jobs[0].job_id == 'job-1'
            assert jobs[0].status == TranscodingStatus.COMPLETE
            assert jobs[1].job_id == 'job-2'
            assert jobs[1].status == TranscodingStatus.PROGRESSING
            assert jobs[1].progress_percentage == 75
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, aws_config, credentials_manager):
        """Test health check when service is healthy."""
        mock_client = Mock()
        mock_client.list_jobs.return_value = {'Jobs': []}
        credentials_manager.get_client.return_value = mock_client
        
        with patch('app.core.aws.mediaconvert_service.MetadataService'):
            service = MediaConvertService(aws_config, credentials_manager)
            
            health = await service.health_check()
            
            assert health['status'] == 'healthy'
            assert health['endpoint'] == aws_config.mediaconvert_endpoint
            assert health['region'] == aws_config.region
            assert health['role_arn'] == aws_config.mediaconvert_role_arn
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, aws_config, credentials_manager):
        """Test health check when service is unhealthy."""
        mock_client = Mock()
        mock_client.list_jobs.side_effect = Exception("Service unavailable")
        credentials_manager.get_client.return_value = mock_client
        
        with patch('app.core.aws.mediaconvert_service.MetadataService'):
            service = MediaConvertService(aws_config, credentials_manager)
            
            health = await service.health_check()
            
            assert health['status'] == 'unhealthy'
            assert 'Service unavailable' in health['error']


class TestTranscodingJobConfig:
    """Test cases for TranscodingJobConfig."""
    
    def test_init_with_defaults(self):
        """Test initialization with default values."""
        config = TranscodingJobConfig(
            input_s3_path="s3://bucket/input.mp4",
            output_s3_prefix="s3://bucket/output",
            video_id="video-123",
            project_id="project-456"
        )
        
        assert config.input_s3_path == "s3://bucket/input.mp4"
        assert config.output_s3_prefix == "s3://bucket/output"
        assert config.video_id == "video-123"
        assert config.project_id == "project-456"
        assert config.output_formats == [OutputFormat.HLS, OutputFormat.DASH]
        assert config.quality_levels == [QualityLevel.HD_1080P, QualityLevel.HD_720P, QualityLevel.SD_480P]
        assert config.metadata == {}
    
    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        metadata = {"custom": "value"}
        config = TranscodingJobConfig(
            input_s3_path="s3://bucket/input.mp4",
            output_s3_prefix="s3://bucket/output",
            video_id="video-123",
            project_id="project-456",
            output_formats=[OutputFormat.MP4],
            quality_levels=[QualityLevel.HD_720P],
            metadata=metadata
        )
        
        assert config.output_formats == [OutputFormat.MP4]
        assert config.quality_levels == [QualityLevel.HD_720P]
        assert config.metadata == metadata


class TestTranscodingJobResult:
    """Test cases for TranscodingJobResult."""
    
    def test_init(self):
        """Test initialization of TranscodingJobResult."""
        created_at = datetime.utcnow()
        completed_at = datetime.utcnow()
        output_paths = {"hls": ["path1.m3u8"], "dash": ["path2.mpd"]}
        
        result = TranscodingJobResult(
            job_id="job-123",
            status=TranscodingStatus.COMPLETE,
            input_s3_path="s3://bucket/input.mp4",
            output_paths=output_paths,
            created_at=created_at,
            completed_at=completed_at,
            error_message=None,
            progress_percentage=100
        )
        
        assert result.job_id == "job-123"
        assert result.status == TranscodingStatus.COMPLETE
        assert result.input_s3_path == "s3://bucket/input.mp4"
        assert result.output_paths == output_paths
        assert result.created_at == created_at
        assert result.completed_at == completed_at
        assert result.error_message is None
        assert result.progress_percentage == 100


class TestEnums:
    """Test cases for enum classes."""
    
    def test_transcoding_status_enum(self):
        """Test TranscodingStatus enum values."""
        assert TranscodingStatus.SUBMITTED.value == "SUBMITTED"
        assert TranscodingStatus.PROGRESSING.value == "PROGRESSING"
        assert TranscodingStatus.COMPLETE.value == "COMPLETE"
        assert TranscodingStatus.CANCELED.value == "CANCELED"
        assert TranscodingStatus.ERROR.value == "ERROR"
    
    def test_output_format_enum(self):
        """Test OutputFormat enum values."""
        assert OutputFormat.HLS.value == "HLS"
        assert OutputFormat.DASH.value == "DASH"
        assert OutputFormat.MP4.value == "MP4"
    
    def test_quality_level_enum(self):
        """Test QualityLevel enum values."""
        assert QualityLevel.HD_1080P.value == "1080p"
        assert QualityLevel.HD_720P.value == "720p"
        assert QualityLevel.SD_480P.value == "480p"
        assert QualityLevel.SD_360P.value == "360p"
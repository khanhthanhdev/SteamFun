"""
Tests for S3 Video Upload Service

Comprehensive tests for S3 video upload functionality using moto for AWS mocking.
"""

import os
import sys
import tempfile
import asyncio
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import moto for AWS mocking
try:
    from moto import mock_aws
    import boto3
    from botocore.exceptions import ClientError
    MOTO_AVAILABLE = True
except ImportError:
    print("Warning: moto not installed. Install with: pip install moto[s3]")
    mock_aws = None
    MOTO_AVAILABLE = False

from aws.config import AWSConfig
from aws.credentials import AWSCredentialsManager
from aws.s3_video_upload import S3VideoUploadService, VideoChunk, ProgressPercentage
from aws.exceptions import AWSS3Error, AWSRetryableError, AWSNonRetryableError


class TestVideoChunk:
    """Test VideoChunk data class."""
    
    def test_video_chunk_creation(self):
        """Test creating a valid VideoChunk."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
            f.write(b"fake video content")
            temp_file = f.name
        
        try:
            chunk = VideoChunk(
                file_path=temp_file,
                project_id="test_project",
                video_id="test_video",
                scene_number=1,
                version=1
            )
            
            assert chunk.file_path == temp_file
            assert chunk.project_id == "test_project"
            assert chunk.video_id == "test_video"
            assert chunk.scene_number == 1
            assert chunk.version == 1
            assert chunk.metadata is None
            
        finally:
            os.unlink(temp_file)
    
    def test_video_chunk_with_metadata(self):
        """Test VideoChunk with custom metadata."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
            f.write(b"fake video content")
            temp_file = f.name
        
        try:
            metadata = {"custom_key": "custom_value", "duration": "30s"}
            chunk = VideoChunk(
                file_path=temp_file,
                project_id="test_project",
                video_id="test_video",
                scene_number=2,
                version=1,
                metadata=metadata
            )
            
            assert chunk.metadata == metadata
            
        finally:
            os.unlink(temp_file)
    
    def test_video_chunk_validation_missing_file(self):
        """Test VideoChunk validation with missing file."""
        with pytest.raises(ValueError, match="Video file does not exist"):
            VideoChunk(
                file_path="/nonexistent/file.mp4",
                project_id="test_project",
                video_id="test_video",
                scene_number=1,
                version=1
            )
    
    def test_video_chunk_validation_missing_ids(self):
        """Test VideoChunk validation with missing IDs."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
            f.write(b"fake video content")
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="project_id and video_id are required"):
                VideoChunk(
                    file_path=temp_file,
                    project_id="",
                    video_id="test_video",
                    scene_number=1,
                    version=1
                )
                
        finally:
            os.unlink(temp_file)
    
    def test_video_chunk_validation_invalid_numbers(self):
        """Test VideoChunk validation with invalid numbers."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
            f.write(b"fake video content")
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="scene_number must be >= 0 and version must be >= 1"):
                VideoChunk(
                    file_path=temp_file,
                    project_id="test_project",
                    video_id="test_video",
                    scene_number=-1,
                    version=1
                )
                
        finally:
            os.unlink(temp_file)


class TestProgressPercentage:
    """Test ProgressPercentage callback class."""
    
    def test_progress_percentage_creation(self):
        """Test creating ProgressPercentage tracker."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content for progress tracking")
            temp_file = f.name
        
        try:
            progress = ProgressPercentage(temp_file)
            
            assert progress._filename == temp_file
            assert progress._size == len(b"test content for progress tracking")
            assert progress._seen_so_far == 0
            
        finally:
            os.unlink(temp_file)
    
    def test_progress_percentage_callback(self):
        """Test progress callback functionality."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            content = b"test content for progress tracking"
            f.write(content)
            temp_file = f.name
        
        try:
            callback_calls = []
            
            def test_callback(seen, total, percentage):
                callback_calls.append((seen, total, percentage))
            
            progress = ProgressPercentage(temp_file, callback=test_callback)
            
            # Simulate progress updates
            progress(10)  # 10 bytes
            progress(20)  # 20 more bytes (30 total)
            
            assert progress._seen_so_far == 30
            assert len(callback_calls) == 2
            assert callback_calls[0] == (10, len(content), (10 / len(content)) * 100)
            assert callback_calls[1] == (30, len(content), (30 / len(content)) * 100)
            
        finally:
            os.unlink(temp_file)


@pytest.mark.skipif(not MOTO_AVAILABLE, reason="moto not available")
class TestS3VideoUploadService:
    """Test S3VideoUploadService with mocked AWS services."""
    
    @pytest.fixture
    def aws_config(self):
        """Create test AWS configuration."""
        return AWSConfig(
            region='us-east-1',
            video_bucket_name='test-video-bucket',
            code_bucket_name='test-code-bucket',
            enable_aws_upload=True,
            enable_encryption=True,
            max_retries=2,
            multipart_threshold=1024,  # Small threshold for testing
            max_concurrent_uploads=2,
            chunk_size=512
        )
    
    @pytest.fixture
    def temp_video_file(self):
        """Create a temporary video file for testing."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
            # Create content larger than multipart threshold for testing
            content = b"fake video content " * 100  # ~2KB
            f.write(content)
            temp_file = f.name
        
        yield temp_file
        
        # Cleanup
        if os.path.exists(temp_file):
            os.unlink(temp_file)
    
    @mock_aws
    def test_upload_service_initialization(self, aws_config):
        """Test S3VideoUploadService initialization."""
        # Create mock S3 bucket
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.create_bucket(Bucket='test-video-bucket')
        
        # Create credentials manager
        creds_manager = AWSCredentialsManager(aws_config)
        
        # Initialize upload service
        upload_service = S3VideoUploadService(aws_config, creds_manager)
        
        assert upload_service.config == aws_config
        assert upload_service.credentials_manager == creds_manager
        assert upload_service.s3_client is not None
        assert upload_service.s3_resource is not None
        assert upload_service.transfer_config is not None
    
    @mock_aws
    def test_generate_s3_key(self, aws_config, temp_video_file):
        """Test S3 key generation."""
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.create_bucket(Bucket='test-video-bucket')
        
        creds_manager = AWSCredentialsManager(aws_config)
        upload_service = S3VideoUploadService(aws_config, creds_manager)
        
        # Test scene chunk key
        chunk = VideoChunk(
            file_path=temp_video_file,
            project_id="test_project",
            video_id="test_video",
            scene_number=1,
            version=2
        )
        
        key = upload_service._generate_s3_key(chunk)
        expected = "videos/test_project/test_video/chunk_001_v2.mp4"
        assert key == expected
        
        # Test combined video key
        chunk.scene_number = 0
        key = upload_service._generate_s3_key(chunk)
        expected = "videos/test_project/test_video/test_video_full_v2.mp4"
        assert key == expected
    
    @mock_aws
    def test_prepare_upload_metadata(self, aws_config, temp_video_file):
        """Test upload metadata preparation."""
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.create_bucket(Bucket='test-video-bucket')
        
        creds_manager = AWSCredentialsManager(aws_config)
        upload_service = S3VideoUploadService(aws_config, creds_manager)
        
        chunk = VideoChunk(
            file_path=temp_video_file,
            project_id="test_project",
            video_id="test_video",
            scene_number=1,
            version=1,
            metadata={"custom": "value"}
        )
        
        extra_args = upload_service._prepare_upload_metadata(chunk)
        
        # Check metadata
        assert 'Metadata' in extra_args
        metadata = extra_args['Metadata']
        assert metadata['video_id'] == 'test_video'
        assert metadata['scene_number'] == '1'
        assert metadata['version'] == '1'
        assert metadata['project_id'] == 'test_project'
        assert metadata['custom'] == 'value'
        assert 'upload_timestamp' in metadata
        assert 'file_size' in metadata
        
        # Check content type
        assert extra_args['ContentType'] == 'video/mp4'
        
        # Check encryption (SSE-S3 since no KMS key)
        assert extra_args['ServerSideEncryption'] == 'AES256'
    
    @mock_aws
    def test_prepare_upload_metadata_with_kms(self, temp_video_file):
        """Test upload metadata preparation with KMS encryption."""
        config = AWSConfig(
            region='us-east-1',
            video_bucket_name='test-video-bucket',
            enable_encryption=True,
            kms_key_id='arn:aws:kms:us-east-1:123456789012:key/test-key'
        )
        
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.create_bucket(Bucket='test-video-bucket')
        
        creds_manager = AWSCredentialsManager(config)
        upload_service = S3VideoUploadService(config, creds_manager)
        
        chunk = VideoChunk(
            file_path=temp_video_file,
            project_id="test_project",
            video_id="test_video",
            scene_number=1,
            version=1
        )
        
        extra_args = upload_service._prepare_upload_metadata(chunk)
        
        # Check KMS encryption
        assert extra_args['ServerSideEncryption'] == 'aws:kms'
        assert extra_args['SSEKMSKeyId'] == 'arn:aws:kms:us-east-1:123456789012:key/test-key'
    
    def test_is_retryable_error(self, aws_config):
        """Test error retry logic."""
        creds_manager = Mock()
        upload_service = S3VideoUploadService(aws_config, creds_manager)
        
        # Non-retryable errors
        assert not upload_service._is_retryable_error('NoSuchBucket')
        assert not upload_service._is_retryable_error('AccessDenied')
        assert not upload_service._is_retryable_error('InvalidAccessKeyId')
        
        # Retryable errors
        assert upload_service._is_retryable_error('RequestTimeout')
        assert upload_service._is_retryable_error('ServiceUnavailable')
        assert upload_service._is_retryable_error('SlowDown')
        
        # Unknown errors (default to retryable)
        assert upload_service._is_retryable_error('UnknownError')
    
    @mock_aws
    @pytest.mark.asyncio
    async def test_upload_single_chunk_success(self, aws_config, temp_video_file):
        """Test successful single chunk upload."""
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.create_bucket(Bucket='test-video-bucket')
        
        creds_manager = AWSCredentialsManager(aws_config)
        upload_service = S3VideoUploadService(aws_config, creds_manager)
        
        chunk = VideoChunk(
            file_path=temp_video_file,
            project_id="test_project",
            video_id="test_video",
            scene_number=1,
            version=1
        )
        
        # Upload the chunk
        s3_url = await upload_service._upload_single_chunk(chunk)
        
        # Verify result
        expected_url = "s3://test-video-bucket/videos/test_project/test_video/chunk_001_v1.mp4"
        assert s3_url == expected_url
        
        # Verify object exists in S3
        s3_key = "videos/test_project/test_video/chunk_001_v1.mp4"
        response = s3_client.head_object(Bucket='test-video-bucket', Key=s3_key)
        
        # Check metadata
        metadata = response['Metadata']
        assert metadata['video_id'] == 'test_video'
        assert metadata['scene_number'] == '1'
        assert metadata['version'] == '1'
        assert metadata['project_id'] == 'test_project'
    
    @mock_aws
    @pytest.mark.asyncio
    async def test_upload_video_chunks_success(self, aws_config, temp_video_file):
        """Test successful multiple chunk upload."""
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.create_bucket(Bucket='test-video-bucket')
        
        creds_manager = AWSCredentialsManager(aws_config)
        upload_service = S3VideoUploadService(aws_config, creds_manager)
        
        # Create multiple chunks
        chunks = [
            VideoChunk(
                file_path=temp_video_file,
                project_id="test_project",
                video_id="test_video",
                scene_number=i,
                version=1
            )
            for i in range(1, 4)  # 3 chunks
        ]
        
        # Upload chunks
        results = await upload_service.upload_video_chunks(chunks)
        
        # Verify results
        assert len(results) == 3
        assert all(url is not None for url in results)
        assert all(url.startswith("s3://test-video-bucket/videos/test_project/test_video/") for url in results)
        
        # Verify objects exist in S3
        for i in range(1, 4):
            s3_key = f"videos/test_project/test_video/chunk_{i:03d}_v1.mp4"
            response = s3_client.head_object(Bucket='test-video-bucket', Key=s3_key)
            assert response['ContentLength'] > 0
    
    @mock_aws
    @pytest.mark.asyncio
    async def test_upload_combined_video(self, aws_config, temp_video_file):
        """Test combined video upload."""
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.create_bucket(Bucket='test-video-bucket')
        
        creds_manager = AWSCredentialsManager(aws_config)
        upload_service = S3VideoUploadService(aws_config, creds_manager)
        
        # Upload combined video
        s3_url = await upload_service.upload_combined_video(
            file_path=temp_video_file,
            project_id="test_project",
            video_id="test_video",
            version=1,
            metadata={"type": "combined"}
        )
        
        # Verify result
        expected_url = "s3://test-video-bucket/videos/test_project/test_video/test_video_full_v1.mp4"
        assert s3_url == expected_url
        
        # Verify object exists in S3
        s3_key = "videos/test_project/test_video/test_video_full_v1.mp4"
        response = s3_client.head_object(Bucket='test-video-bucket', Key=s3_key)
        
        # Check metadata
        metadata = response['Metadata']
        assert metadata['video_id'] == 'test_video'
        assert metadata['scene_number'] == '0'  # 0 for combined
        assert metadata['type'] == 'combined'
    
    def test_get_video_url(self, aws_config):
        """Test video URL generation."""
        creds_manager = Mock()
        upload_service = S3VideoUploadService(aws_config, creds_manager)
        
        # Test scene chunk URL
        url = upload_service.get_video_url("test_project", "test_video", 1, 2)
        expected = "s3://test-video-bucket/videos/test_project/test_video/chunk_001_v2.mp4"
        assert url == expected
        
        # Test combined video URL
        url = upload_service.get_video_url("test_project", "test_video", 0, 1)
        expected = "s3://test-video-bucket/videos/test_project/test_video/test_video_full_v1.mp4"
        assert url == expected


def run_tests():
    """Run all tests."""
    print("Running S3 Video Upload Service Tests")
    print("=" * 50)
    
    # Check if moto is available
    if not MOTO_AVAILABLE:
        print("❌ moto not available. Install with: pip install moto[s3]")
        return False
    
    # Run tests using pytest
    try:
        import pytest
        
        # Run tests in this file
        test_file = __file__
        result = pytest.main(['-v', test_file])
        
        if result == 0:
            print("\n✅ All S3 video upload tests passed!")
            return True
        else:
            print("\n❌ Some tests failed")
            return False
            
    except ImportError:
        print("❌ pytest not available. Install with: pip install pytest")
        return False


if __name__ == '__main__':
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
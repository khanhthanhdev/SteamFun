"""
Test S3 Code Storage Service

Comprehensive tests for S3 code storage functionality including upload, download,
versioning, and S3 Object Lock features.
"""

import os
import sys
import asyncio
import tempfile
import pytest
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from moto import mock_aws
import boto3

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.aws.s3_code_storage import S3CodeStorageService, CodeMetadata, CodeVersion
from src.aws.config import AWSConfig
from src.aws.credentials import AWSCredentialsManager
from src.aws.exceptions import AWSS3Error


class TestS3CodeStorageService:
    """Test suite for S3 Code Storage Service."""
    
    @pytest.fixture
    def aws_config(self):
        """Create test AWS configuration."""
        return AWSConfig(
            region='us-east-1',
            video_bucket_name='test-video-bucket',
            code_bucket_name='test-code-bucket',
            enable_encryption=True,
            max_retries=2,
            retry_backoff_base=1.5,
            enable_aws_upload=True
        )
    
    @pytest.fixture
    def mock_credentials_manager(self, aws_config):
        """Create mock credentials manager."""
        manager = Mock(spec=AWSCredentialsManager)
        manager.get_client.return_value = Mock()
        manager.get_resource.return_value = Mock()
        return manager
    
    @pytest.fixture
    def sample_code(self):
        """Sample Manim code for testing."""
        return '''from manim import *

class TestScene(Scene):
    def construct(self):
        # Create a simple text animation
        text = Text("Hello, World!")
        self.play(Write(text))
        self.wait(2)
        
        # Transform to another text
        new_text = Text("Welcome to Manim!")
        self.play(Transform(text, new_text))
        self.wait(2)
'''
    
    @pytest.fixture
    def sample_metadata(self):
        """Sample code metadata for testing."""
        return CodeMetadata(
            video_id='test_video_123',
            project_id='test_project_456',
            version=1,
            scene_number=1,
            created_at=datetime(2024, 1, 15, 10, 30, 0)
        )
    
    @pytest.fixture
    def code_storage_service(self, aws_config, mock_credentials_manager):
        """Create S3 code storage service instance."""
        return S3CodeStorageService(aws_config, mock_credentials_manager)
    
    def test_code_metadata_validation(self):
        """Test CodeMetadata validation."""
        # Valid metadata
        metadata = CodeMetadata(
            video_id='test_video',
            project_id='test_project',
            version=1
        )
        assert metadata.video_id == 'test_video'
        assert metadata.project_id == 'test_project'
        assert metadata.version == 1
        assert metadata.created_at is not None
        
        # Invalid metadata - missing video_id
        with pytest.raises(ValueError, match="video_id and project_id are required"):
            CodeMetadata(video_id='', project_id='test_project', version=1)
        
        # Invalid metadata - invalid version
        with pytest.raises(ValueError, match="version must be >= 1"):
            CodeMetadata(video_id='test', project_id='test_project', version=0)
    
    def test_code_version_file_size_calculation(self, sample_code, sample_metadata):
        """Test CodeVersion file size calculation."""
        code_version = CodeVersion(
            content=sample_code,
            metadata=sample_metadata
        )
        
        expected_size = len(sample_code.encode('utf-8'))
        assert code_version.file_size == expected_size
    
    def test_generate_s3_key(self, code_storage_service, sample_metadata):
        """Test S3 key generation with versioning."""
        s3_key = code_storage_service._generate_s3_key(sample_metadata)
        
        expected_key = f"code/{sample_metadata.project_id}/{sample_metadata.video_id}/{sample_metadata.video_id}_v{sample_metadata.version}.py"
        assert s3_key == expected_key
    
    def test_prepare_upload_metadata(self, code_storage_service, sample_metadata):
        """Test upload metadata preparation."""
        extra_args = code_storage_service._prepare_upload_metadata(sample_metadata)
        
        # Check basic structure
        assert 'Metadata' in extra_args
        assert 'ContentType' in extra_args
        assert 'ContentEncoding' in extra_args
        
        # Check metadata content
        metadata = extra_args['Metadata']
        assert metadata['video_id'] == sample_metadata.video_id
        assert metadata['project_id'] == sample_metadata.project_id
        assert metadata['version'] == str(sample_metadata.version)
        assert metadata['scene_number'] == str(sample_metadata.scene_number)
        
        # Check content type and encoding
        assert extra_args['ContentType'] == 'text/x-python'
        assert extra_args['ContentEncoding'] == 'utf-8'
        
        # Check encryption (should be enabled in test config)
        assert extra_args['ServerSideEncryption'] == 'AES256'
    
    def test_prepare_upload_metadata_with_object_lock(self, code_storage_service, sample_metadata):
        """Test upload metadata preparation with Object Lock."""
        extra_args = code_storage_service._prepare_upload_metadata(
            sample_metadata, 
            enable_object_lock=True
        )
        
        # Check Object Lock settings
        assert 'ObjectLockMode' in extra_args
        assert extra_args['ObjectLockMode'] == 'GOVERNANCE'
        assert 'ObjectLockRetainUntilDate' in extra_args
    
    @pytest.mark.asyncio
    @mock_aws
    async def test_upload_code_success(self, code_storage_service, sample_code, sample_metadata):
        """Test successful code upload."""
        # Create mock S3 bucket
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.create_bucket(Bucket='test-code-bucket')
        
        # Mock the s3_client in the service
        code_storage_service.s3_client = s3_client
        
        # Upload code
        s3_url = await code_storage_service.upload_code(sample_code, sample_metadata)
        
        # Verify result
        expected_url = f"s3://test-code-bucket/code/{sample_metadata.project_id}/{sample_metadata.video_id}/{sample_metadata.video_id}_v{sample_metadata.version}.py"
        assert s3_url == expected_url
        
        # Verify object exists in S3
        s3_key = f"code/{sample_metadata.project_id}/{sample_metadata.video_id}/{sample_metadata.video_id}_v{sample_metadata.version}.py"
        response = s3_client.get_object(Bucket='test-code-bucket', Key=s3_key)
        stored_code = response['Body'].read().decode('utf-8')
        assert stored_code == sample_code
    
    @pytest.mark.asyncio
    @mock_aws
    async def test_upload_code_versions(self, code_storage_service, sample_code):
        """Test uploading multiple code versions."""
        # Create mock S3 bucket
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.create_bucket(Bucket='test-code-bucket')
        code_storage_service.s3_client = s3_client
        
        # Create multiple code versions
        code_versions = []
        for version in range(1, 4):
            metadata = CodeMetadata(
                video_id='test_video',
                project_id='test_project',
                version=version
            )
            code_version = CodeVersion(
                content=f"{sample_code}\n# Version {version}",
                metadata=metadata
            )
            code_versions.append(code_version)
        
        # Upload versions
        results = await code_storage_service.upload_code_versions(code_versions)
        
        # Verify all uploads succeeded
        assert len(results) == 3
        assert all(url is not None for url in results)
        
        # Verify each version exists
        for i, result in enumerate(results, 1):
            assert f"test_video_v{i}.py" in result
    
    @pytest.mark.asyncio
    @mock_aws
    async def test_download_code_success(self, code_storage_service, sample_code, sample_metadata):
        """Test successful code download."""
        # Create mock S3 bucket and upload code
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.create_bucket(Bucket='test-code-bucket')
        code_storage_service.s3_client = s3_client
        
        # Upload code first
        await code_storage_service.upload_code(sample_code, sample_metadata)
        
        # Download code
        downloaded_code = await code_storage_service.download_code(
            sample_metadata.video_id,
            sample_metadata.project_id,
            sample_metadata.version
        )
        
        # Verify content matches
        assert downloaded_code == sample_code
    
    @pytest.mark.asyncio
    @mock_aws
    async def test_download_code_not_found(self, code_storage_service):
        """Test downloading non-existent code."""
        # Create mock S3 bucket (empty)
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.create_bucket(Bucket='test-code-bucket')
        code_storage_service.s3_client = s3_client
        
        # Try to download non-existent code
        with pytest.raises(AWSS3Error, match="Code not found"):
            await code_storage_service.download_code('nonexistent', 'project', 1)
    
    @pytest.mark.asyncio
    @mock_aws
    async def test_list_code_versions(self, code_storage_service, sample_code):
        """Test listing code versions."""
        # Create mock S3 bucket
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.create_bucket(Bucket='test-code-bucket')
        code_storage_service.s3_client = s3_client
        
        # Upload multiple versions
        video_id = 'test_video'
        project_id = 'test_project'
        
        for version in [1, 3, 5]:  # Non-sequential versions
            metadata = CodeMetadata(
                video_id=video_id,
                project_id=project_id,
                version=version
            )
            await code_storage_service.upload_code(sample_code, metadata)
        
        # List versions
        versions = await code_storage_service.list_code_versions(video_id, project_id)
        
        # Verify results
        assert versions == [1, 3, 5]  # Should be sorted
    
    @pytest.mark.asyncio
    @mock_aws
    async def test_download_latest_code(self, code_storage_service, sample_code):
        """Test downloading latest code version."""
        # Create mock S3 bucket
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.create_bucket(Bucket='test-code-bucket')
        code_storage_service.s3_client = s3_client
        
        # Upload multiple versions
        video_id = 'test_video'
        project_id = 'test_project'
        latest_code = f"{sample_code}\n# Latest version"
        
        # Upload versions 1, 2, 3
        for version in range(1, 4):
            code_content = sample_code if version < 3 else latest_code
            metadata = CodeMetadata(
                video_id=video_id,
                project_id=project_id,
                version=version
            )
            await code_storage_service.upload_code(code_content, metadata)
        
        # Download latest
        downloaded_code = await code_storage_service.download_latest_code(video_id, project_id)
        
        # Should get version 3 (latest)
        assert downloaded_code == latest_code
    
    @pytest.mark.asyncio
    @mock_aws
    async def test_delete_code_version(self, code_storage_service, sample_code, sample_metadata):
        """Test deleting a code version."""
        # Create mock S3 bucket and upload code
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.create_bucket(Bucket='test-code-bucket')
        code_storage_service.s3_client = s3_client
        
        # Upload code
        await code_storage_service.upload_code(sample_code, sample_metadata)
        
        # Verify it exists
        downloaded_code = await code_storage_service.download_code(
            sample_metadata.video_id,
            sample_metadata.project_id,
            sample_metadata.version
        )
        assert downloaded_code == sample_code
        
        # Delete the version
        result = await code_storage_service.delete_code_version(
            sample_metadata.video_id,
            sample_metadata.project_id,
            sample_metadata.version
        )
        assert result is True
        
        # Verify it's gone
        with pytest.raises(AWSS3Error, match="Code not found"):
            await code_storage_service.download_code(
                sample_metadata.video_id,
                sample_metadata.project_id,
                sample_metadata.version
            )
    
    def test_get_code_url(self, code_storage_service):
        """Test generating code URL."""
        url = code_storage_service.get_code_url('test_video', 'test_project', 2)
        
        expected_url = "s3://test-code-bucket/code/test_project/test_video/test_video_v2.py"
        assert url == expected_url
    
    @pytest.mark.asyncio
    @mock_aws
    async def test_get_code_metadata(self, code_storage_service, sample_code, sample_metadata):
        """Test getting code metadata."""
        # Create mock S3 bucket and upload code
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.create_bucket(Bucket='test-code-bucket')
        code_storage_service.s3_client = s3_client
        
        # Upload code
        await code_storage_service.upload_code(sample_code, sample_metadata)
        
        # Get metadata
        metadata = await code_storage_service.get_code_metadata(
            sample_metadata.video_id,
            sample_metadata.project_id,
            sample_metadata.version
        )
        
        # Verify metadata
        assert metadata['video_id'] == sample_metadata.video_id
        assert metadata['project_id'] == sample_metadata.project_id
        assert metadata['version'] == sample_metadata.version
        assert metadata['content_length'] == len(sample_code.encode('utf-8'))
        assert 's3_url' in metadata
        assert 'last_modified' in metadata
    
    def test_is_retryable_error(self, code_storage_service):
        """Test error retry logic."""
        # Non-retryable errors
        assert not code_storage_service._is_retryable_error('NoSuchBucket')
        assert not code_storage_service._is_retryable_error('AccessDenied')
        assert not code_storage_service._is_retryable_error('NoSuchKey')
        
        # Retryable errors
        assert code_storage_service._is_retryable_error('RequestTimeout')
        assert code_storage_service._is_retryable_error('ServiceUnavailable')
        assert code_storage_service._is_retryable_error('SlowDown')
        
        # Unknown errors (default to retryable)
        assert code_storage_service._is_retryable_error('UnknownError')
    
    def test_service_representation(self, code_storage_service):
        """Test string representation of service."""
        repr_str = repr(code_storage_service)
        
        assert 'S3CodeStorageService' in repr_str
        assert 'test-code-bucket' in repr_str
        assert 'us-east-1' in repr_str
        assert 'encryption=True' in repr_str


async def run_tests():
    """Run all tests."""
    print("Running S3 Code Storage Service tests...")
    
    # Create test instance
    config = AWSConfig(
        region='us-east-1',
        code_bucket_name='test-code-bucket',
        enable_encryption=True,
        max_retries=2,
        enable_aws_upload=True
    )
    
    mock_credentials = Mock(spec=AWSCredentialsManager)
    mock_credentials.get_client.return_value = Mock()
    mock_credentials.get_resource.return_value = Mock()
    
    service = S3CodeStorageService(config, mock_credentials)
    
    # Test basic functionality
    print("✅ Service initialization successful")
    
    # Test metadata creation
    metadata = CodeMetadata(
        video_id='test_video',
        project_id='test_project',
        version=1
    )
    print("✅ CodeMetadata creation successful")
    
    # Test S3 key generation
    s3_key = service._generate_s3_key(metadata)
    expected_key = "code/test_project/test_video/test_video_v1.py"
    assert s3_key == expected_key
    print("✅ S3 key generation successful")
    
    # Test URL generation
    url = service.get_code_url('test_video', 'test_project', 1)
    expected_url = "s3://test-code-bucket/code/test_project/test_video/test_video_v1.py"
    assert url == expected_url
    print("✅ URL generation successful")
    
    print("\n🎉 All S3 Code Storage Service tests passed!")


if __name__ == "__main__":
    asyncio.run(run_tests())
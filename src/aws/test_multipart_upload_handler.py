"""
Test suite for MultipartUploadHandler

Tests multipart upload functionality including resume capability, integrity verification,
upload abortion, and cleanup for failed transfers.
"""

import os
import sys
import tempfile
import asyncio
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from aws.multipart_upload_handler import (
    MultipartUploadHandler, 
    MultipartUploadInfo, 
    UploadPart,
    MultipartProgressTracker
)
from aws.config import AWSConfig
from aws.credentials import AWSCredentialsManager
from aws.exceptions import AWSS3Error


class TestMultipartProgressTracker:
    """Test the multipart progress tracker."""
    
    def test_progress_tracker_initialization(self):
        """Test progress tracker initialization."""
        tracker = MultipartProgressTracker("test.mp4", 1000000)
        
        assert tracker.filename == "test.mp4"
        assert tracker.total_size == 1000000
        assert tracker.uploaded_size == 0
        assert tracker.callback is None
    
    def test_progress_tracker_with_callback(self):
        """Test progress tracker with callback."""
        callback_calls = []
        
        def test_callback(uploaded, total, percentage):
            callback_calls.append((uploaded, total, percentage))
        
        tracker = MultipartProgressTracker("test.mp4", 1000, test_callback)
        tracker.update_progress(250)
        tracker.update_progress(250)
        
        assert tracker.uploaded_size == 500
        assert len(callback_calls) >= 1  # May be throttled
        
        # Check final callback values
        final_call = callback_calls[-1]
        assert final_call[0] == 500
        assert final_call[1] == 1000
        assert final_call[2] == 50.0


class TestMultipartUploadInfo:
    """Test the multipart upload info dataclass."""
    
    def test_upload_info_creation(self):
        """Test upload info creation."""
        parts = [
            UploadPart(1, '"etag1"', 1000, datetime.utcnow()),
            UploadPart(2, '"etag2"', 1000, datetime.utcnow())
        ]
        
        info = MultipartUploadInfo(
            upload_id="test-upload-id",
            bucket="test-bucket",
            key="test/key.mp4",
            parts=parts,
            initiated_at=datetime.utcnow(),
            total_size=5000,
            uploaded_size=2000
        )
        
        assert info.upload_id == "test-upload-id"
        assert info.bucket == "test-bucket"
        assert info.key == "test/key.mp4"
        assert len(info.parts) == 2
        assert info.total_size == 5000
        assert info.uploaded_size == 2000
        assert info.progress_percentage == 40.0
    
    def test_progress_percentage_zero_size(self):
        """Test progress percentage with zero total size."""
        info = MultipartUploadInfo(
            upload_id="test",
            bucket="test",
            key="test",
            parts=[],
            initiated_at=datetime.utcnow(),
            total_size=0,
            uploaded_size=0
        )
        
        assert info.progress_percentage == 0.0


class TestMultipartUploadHandler:
    """Test the multipart upload handler."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock AWS config."""
        config = Mock(spec=AWSConfig)
        config.multipart_threshold = 100 * 1024 * 1024  # 100MB
        config.chunk_size = 8 * 1024 * 1024  # 8MB
        config.max_concurrent_uploads = 3
        config.max_retries = 3
        config.retry_backoff_base = 2
        return config
    
    @pytest.fixture
    def mock_credentials_manager(self):
        """Create mock credentials manager."""
        manager = Mock(spec=AWSCredentialsManager)
        mock_client = Mock()
        manager.get_client.return_value = mock_client
        return manager
    
    @pytest.fixture
    def handler(self, mock_config, mock_credentials_manager):
        """Create multipart upload handler."""
        return MultipartUploadHandler(mock_config, mock_credentials_manager)
    
    @pytest.fixture
    def temp_file(self):
        """Create temporary test file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            # Write 10MB of test data
            test_data = b'A' * (10 * 1024 * 1024)
            f.write(test_data)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def test_handler_initialization(self, handler, mock_config):
        """Test handler initialization."""
        assert handler.config == mock_config
        assert handler.min_part_size == 5 * 1024 * 1024
        assert len(handler.active_uploads) == 0
    
    @pytest.mark.asyncio
    async def test_upload_small_file_uses_standard_upload(self, handler, temp_file):
        """Test that small files use standard upload."""
        # Mock file size to be below threshold
        with patch('os.path.getsize', return_value=50 * 1024 * 1024):  # 50MB
            with patch.object(handler, '_standard_upload', return_value="s3://test/key") as mock_standard:
                result = await handler.upload_large_file(
                    temp_file, "test-bucket", "test/key.mp4"
                )
                
                assert result == "s3://test/key"
                mock_standard.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_upload_large_file_uses_multipart_upload(self, handler, temp_file):
        """Test that large files use multipart upload."""
        # Mock file size to be above threshold
        with patch('os.path.getsize', return_value=200 * 1024 * 1024):  # 200MB
            with patch.object(handler, '_multipart_upload', return_value="s3://test/key") as mock_multipart:
                result = await handler.upload_large_file(
                    temp_file, "test-bucket", "test/key.mp4"
                )
                
                assert result == "s3://test/key"
                mock_multipart.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_find_existing_upload_no_uploads(self, handler):
        """Test finding existing upload when none exist."""
        handler.s3_client.list_multipart_uploads.return_value = {'Uploads': []}
        
        result = await handler._find_existing_upload("test-bucket", "test/key", 1000)
        
        assert result is None
        handler.s3_client.list_multipart_uploads.assert_called_once_with(
            Bucket="test-bucket",
            Prefix="test/key"
        )
    
    @pytest.mark.asyncio
    async def test_find_existing_upload_with_matching_upload(self, handler):
        """Test finding existing upload with matching key."""
        # Mock list_multipart_uploads response
        handler.s3_client.list_multipart_uploads.return_value = {
            'Uploads': [
                {
                    'Key': 'test/key',
                    'UploadId': 'test-upload-id',
                    'Initiated': datetime.utcnow()
                }
            ]
        }
        
        # Mock list_parts response
        handler.s3_client.list_parts.return_value = {
            'Parts': [
                {
                    'PartNumber': 1,
                    'ETag': '"etag1"',
                    'Size': 1000,
                    'LastModified': datetime.utcnow()
                }
            ]
        }
        
        result = await handler._find_existing_upload("test-bucket", "test/key", 5000)
        
        assert result is not None
        assert result.upload_id == "test-upload-id"
        assert result.bucket == "test-bucket"
        assert result.key == "test/key"
        assert len(result.parts) == 1
        assert result.uploaded_size == 1000
        assert result.total_size == 5000
    
    @pytest.mark.asyncio
    async def test_abort_multipart_upload(self, handler):
        """Test aborting multipart upload."""
        upload_info = MultipartUploadInfo(
            upload_id="test-upload-id",
            bucket="test-bucket",
            key="test/key",
            parts=[],
            initiated_at=datetime.utcnow(),
            total_size=1000,
            uploaded_size=0
        )
        
        await handler._abort_multipart_upload(upload_info)
        
        handler.s3_client.abort_multipart_upload.assert_called_once_with(
            Bucket="test-bucket",
            Key="test/key",
            UploadId="test-upload-id"
        )
    
    @pytest.mark.asyncio
    async def test_verify_upload_integrity_success(self, handler):
        """Test successful upload integrity verification."""
        upload_info = MultipartUploadInfo(
            upload_id="test-upload-id",
            bucket="test-bucket",
            key="test/key",
            parts=[],
            initiated_at=datetime.utcnow(),
            total_size=1000,
            uploaded_size=1000
        )
        
        # Mock head_object response
        handler.s3_client.head_object.return_value = {
            'ContentLength': 1000
        }
        
        result = await handler._verify_upload_integrity(upload_info, '"test-etag"')
        
        assert result is True
        handler.s3_client.head_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="test/key"
        )
    
    @pytest.mark.asyncio
    async def test_verify_upload_integrity_size_mismatch(self, handler):
        """Test upload integrity verification with size mismatch."""
        upload_info = MultipartUploadInfo(
            upload_id="test-upload-id",
            bucket="test-bucket",
            key="test/key",
            parts=[],
            initiated_at=datetime.utcnow(),
            total_size=1000,
            uploaded_size=1000
        )
        
        # Mock head_object response with wrong size
        handler.s3_client.head_object.return_value = {
            'ContentLength': 500
        }
        
        result = await handler._verify_upload_integrity(upload_info, '"test-etag"')
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_standard_upload(self, handler, temp_file):
        """Test standard upload functionality."""
        extra_args = {'ContentType': 'video/mp4'}
        
        result = await handler._standard_upload(
            temp_file, "test-bucket", "test/key.mp4", extra_args
        )
        
        assert result == "s3://test-bucket/test/key.mp4"
        handler.s3_client.upload_file.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_list_active_uploads(self, handler):
        """Test listing active uploads."""
        # Add some active uploads
        upload1 = MultipartUploadInfo(
            upload_id="upload1",
            bucket="bucket1",
            key="key1",
            parts=[],
            initiated_at=datetime.utcnow(),
            total_size=1000,
            uploaded_size=0
        )
        
        upload2 = MultipartUploadInfo(
            upload_id="upload2",
            bucket="bucket2",
            key="key2",
            parts=[],
            initiated_at=datetime.utcnow(),
            total_size=2000,
            uploaded_size=0
        )
        
        handler.active_uploads["bucket1/key1"] = upload1
        handler.active_uploads["bucket2/key2"] = upload2
        
        # Test listing all uploads
        all_uploads = await handler.list_active_uploads()
        assert len(all_uploads) == 2
        
        # Test filtering by bucket
        bucket1_uploads = await handler.list_active_uploads("bucket1")
        assert len(bucket1_uploads) == 1
        assert bucket1_uploads[0].bucket == "bucket1"
    
    @pytest.mark.asyncio
    async def test_cleanup_abandoned_uploads(self, handler):
        """Test cleanup of abandoned uploads."""
        # Mock old upload
        old_time = datetime.utcnow() - timedelta(hours=25)
        
        handler.s3_client.list_multipart_uploads.return_value = {
            'Uploads': [
                {
                    'Key': 'old/upload',
                    'UploadId': 'old-upload-id',
                    'Initiated': old_time
                }
            ]
        }
        
        cleanup_count = await handler.cleanup_abandoned_uploads("test-bucket", 24)
        
        assert cleanup_count == 1
        handler.s3_client.abort_multipart_upload.assert_called_once_with(
            Bucket="test-bucket",
            Key="old/upload",
            UploadId="old-upload-id"
        )
    
    def test_handler_repr(self, handler):
        """Test handler string representation."""
        repr_str = repr(handler)
        assert "MultipartUploadHandler" in repr_str
        assert "threshold=" in repr_str
        assert "chunk_size=" in repr_str
        assert "max_concurrency=" in repr_str


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
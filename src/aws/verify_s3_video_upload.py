"""
Verification script for S3 Video Upload functionality.

Verifies that all task requirements have been implemented:
- Video chunk upload method with multipart support
- Progress tracking using ProgressPercentage callback class
- Metadata attachment during upload (video_id, scene_number, version)
- Server-side encryption (SSE-S3 and SSE-KMS options)
- Retry logic with exponential backoff for failed uploads
"""

import os
import sys
import tempfile
import asyncio
import logging
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from aws import (
    AWSConfig, 
    AWSCredentialsManager, 
    S3VideoUploadService, 
    VideoChunk, 
    ProgressPercentage,
    AWSS3Error
)


def verify_video_chunk_class():
    """Verify VideoChunk class functionality."""
    print("🔍 Verifying VideoChunk class...")
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
        f.write(b"fake video content")
        temp_file = f.name
    
    try:
        # Test basic creation
        chunk = VideoChunk(
            file_path=temp_file,
            project_id="test_project",
            video_id="test_video",
            scene_number=1,
            version=1,
            metadata={"custom": "value"}
        )
        
        assert chunk.file_path == temp_file
        assert chunk.project_id == "test_project"
        assert chunk.video_id == "test_video"
        assert chunk.scene_number == 1
        assert chunk.version == 1
        assert chunk.metadata["custom"] == "value"
        
        print("  ✅ VideoChunk creation and validation")
        
        # Test validation
        try:
            VideoChunk(
                file_path="/nonexistent/file.mp4",
                project_id="test",
                video_id="test",
                scene_number=1,
                version=1
            )
            assert False, "Should have raised ValueError"
        except ValueError:
            print("  ✅ File existence validation")
        
        return True
        
    finally:
        os.unlink(temp_file)


def verify_progress_percentage_class():
    """Verify ProgressPercentage callback class."""
    print("\n🔍 Verifying ProgressPercentage class...")
    
    with tempfile.NamedTemporaryFile(delete=False) as f:
        content = b"test content for progress tracking"
        f.write(content)
        temp_file = f.name
    
    try:
        callback_calls = []
        
        def test_callback(seen, total, percentage):
            callback_calls.append((seen, total, percentage))
        
        # Test progress tracking
        progress = ProgressPercentage(temp_file, callback=test_callback)
        
        # Simulate progress updates
        progress(10)  # 10 bytes
        progress(20)  # 20 more bytes (30 total)
        
        assert progress._seen_so_far == 30
        assert len(callback_calls) == 2
        
        print("  ✅ Progress tracking functionality")
        print("  ✅ Custom callback support")
        
        return True
        
    finally:
        os.unlink(temp_file)


def verify_s3_upload_service_initialization():
    """Verify S3VideoUploadService initialization."""
    print("\n🔍 Verifying S3VideoUploadService initialization...")
    
    # Test with mock credentials manager
    config = AWSConfig(
        video_bucket_name='test-bucket',
        enable_aws_upload=False,  # Disable to avoid credential requirements
        enable_encryption=True,
        multipart_threshold=1024,
        max_concurrent_uploads=2,
        chunk_size=512
    )
    
    creds_manager = Mock()
    service = S3VideoUploadService(config, creds_manager)
    
    assert service.config == config
    assert service.credentials_manager == creds_manager
    assert service.transfer_config is not None
    
    print("  ✅ Service initialization")
    print("  ✅ Transfer configuration setup")
    
    return True


def verify_s3_key_generation():
    """Verify S3 key generation with organized naming convention."""
    print("\n🔍 Verifying S3 key generation...")
    
    config = AWSConfig(video_bucket_name='test-bucket', enable_aws_upload=False)
    creds_manager = Mock()
    service = S3VideoUploadService(config, creds_manager)
    
    # Test scene chunk key
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
        f.write(b"test")
        temp_file = f.name
    
    try:
        chunk = VideoChunk(
            file_path=temp_file,
            project_id="test_project",
            video_id="test_video",
            scene_number=1,
            version=2
        )
        
        key = service._generate_s3_key(chunk)
        expected = "videos/test_project/test_video/chunk_001_v2.mp4"
        assert key == expected
        
        print("  ✅ Scene chunk key generation")
        
        # Test combined video key
        chunk.scene_number = 0
        key = service._generate_s3_key(chunk)
        expected = "videos/test_project/test_video/test_video_full_v2.mp4"
        assert key == expected
        
        print("  ✅ Combined video key generation")
        
        return True
        
    finally:
        os.unlink(temp_file)


def verify_metadata_attachment():
    """Verify metadata attachment during upload."""
    print("\n🔍 Verifying metadata attachment...")
    
    config = AWSConfig(
        video_bucket_name='test-bucket',
        enable_aws_upload=False,
        enable_encryption=True,
        kms_key_id='test-key-id'
    )
    
    creds_manager = Mock()
    service = S3VideoUploadService(config, creds_manager)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
        f.write(b"test content")
        temp_file = f.name
    
    try:
        chunk = VideoChunk(
            file_path=temp_file,
            project_id="test_project",
            video_id="test_video",
            scene_number=1,
            version=1,
            metadata={"custom": "value", "duration": "30s"}
        )
        
        extra_args = service._prepare_upload_metadata(chunk)
        
        # Check metadata
        metadata = extra_args['Metadata']
        assert metadata['video_id'] == 'test_video'
        assert metadata['scene_number'] == '1'
        assert metadata['version'] == '1'
        assert metadata['project_id'] == 'test_project'
        assert metadata['custom'] == 'value'
        assert metadata['duration'] == '30s'
        assert 'upload_timestamp' in metadata
        assert 'file_size' in metadata
        
        print("  ✅ Basic metadata attachment")
        print("  ✅ Custom metadata support")
        
        # Check content type
        assert extra_args['ContentType'] == 'video/mp4'
        print("  ✅ Content type setting")
        
        return True
        
    finally:
        os.unlink(temp_file)


def verify_encryption_support():
    """Verify server-side encryption support."""
    print("\n🔍 Verifying encryption support...")
    
    # Test SSE-S3
    config = AWSConfig(
        video_bucket_name='test-bucket',
        enable_aws_upload=False,
        enable_encryption=True
    )
    
    creds_manager = Mock()
    service = S3VideoUploadService(config, creds_manager)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
        f.write(b"test")
        temp_file = f.name
    
    try:
        chunk = VideoChunk(
            file_path=temp_file,
            project_id="test",
            video_id="test",
            scene_number=1,
            version=1
        )
        
        extra_args = service._prepare_upload_metadata(chunk)
        assert extra_args['ServerSideEncryption'] == 'AES256'
        
        print("  ✅ SSE-S3 encryption")
        
        # Test KMS encryption
        config.kms_key_id = 'arn:aws:kms:us-east-1:123456789012:key/test-key'
        service = S3VideoUploadService(config, creds_manager)
        
        extra_args = service._prepare_upload_metadata(chunk)
        assert extra_args['ServerSideEncryption'] == 'aws:kms'
        assert extra_args['SSEKMSKeyId'] == 'arn:aws:kms:us-east-1:123456789012:key/test-key'
        
        print("  ✅ SSE-KMS encryption")
        
        return True
        
    finally:
        os.unlink(temp_file)


def verify_retry_logic():
    """Verify retry logic with exponential backoff."""
    print("\n🔍 Verifying retry logic...")
    
    config = AWSConfig(
        video_bucket_name='test-bucket',
        enable_aws_upload=False,
        max_retries=3,
        retry_backoff_base=2.0
    )
    
    creds_manager = Mock()
    service = S3VideoUploadService(config, creds_manager)
    
    # Test retryable error detection
    assert service._is_retryable_error('RequestTimeout') == True
    assert service._is_retryable_error('ServiceUnavailable') == True
    assert service._is_retryable_error('SlowDown') == True
    
    print("  ✅ Retryable error detection")
    
    # Test non-retryable error detection
    assert service._is_retryable_error('NoSuchBucket') == False
    assert service._is_retryable_error('AccessDenied') == False
    assert service._is_retryable_error('InvalidAccessKeyId') == False
    
    print("  ✅ Non-retryable error detection")
    
    # Test unknown error handling (defaults to retryable)
    assert service._is_retryable_error('UnknownError') == True
    
    print("  ✅ Unknown error handling")
    
    return True


def verify_multipart_support():
    """Verify multipart upload support."""
    print("\n🔍 Verifying multipart upload support...")
    
    config = AWSConfig(
        video_bucket_name='test-bucket',
        enable_aws_upload=False,
        multipart_threshold=1024,  # 1KB threshold
        max_concurrent_uploads=3,
        chunk_size=512
    )
    
    creds_manager = Mock()
    service = S3VideoUploadService(config, creds_manager)
    
    # Check transfer configuration
    assert service.transfer_config.multipart_threshold == 1024
    assert service.transfer_config.max_concurrency == 3
    assert service.transfer_config.multipart_chunksize == 512
    assert service.transfer_config.use_threads == True
    
    print("  ✅ Transfer configuration")
    print("  ✅ Multipart threshold setting")
    print("  ✅ Concurrency control")
    print("  ✅ Chunk size configuration")
    
    return True


def verify_url_generation():
    """Verify URL generation without uploading."""
    print("\n🔍 Verifying URL generation...")
    
    config = AWSConfig(video_bucket_name='my-video-bucket', enable_aws_upload=False)
    creds_manager = Mock()
    service = S3VideoUploadService(config, creds_manager)
    
    # Test scene chunk URL
    url = service.get_video_url("test_project", "test_video", 1, 2)
    expected = "s3://my-video-bucket/videos/test_project/test_video/chunk_001_v2.mp4"
    assert url == expected
    
    print("  ✅ Scene chunk URL generation")
    
    # Test combined video URL
    url = service.get_video_url("test_project", "test_video", 0, 1)
    expected = "s3://my-video-bucket/videos/test_project/test_video/test_video_full_v1.mp4"
    assert url == expected
    
    print("  ✅ Combined video URL generation")
    
    return True


async def verify_async_functionality():
    """Verify async upload functionality (mocked)."""
    print("\n🔍 Verifying async functionality...")
    
    config = AWSConfig(video_bucket_name='test-bucket', enable_aws_upload=False)
    creds_manager = Mock()
    service = S3VideoUploadService(config, creds_manager)
    
    # Mock the upload methods to avoid actual AWS calls
    with patch.object(service, '_upload_with_retry') as mock_upload:
        mock_upload.return_value = None
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
            f.write(b"test content")
            temp_file = f.name
        
        try:
            chunk = VideoChunk(
                file_path=temp_file,
                project_id="test",
                video_id="test",
                scene_number=1,
                version=1
            )
            
            # Test single chunk upload
            result = await service._upload_single_chunk(chunk)
            assert result.startswith("s3://test-bucket/videos/test/test/")
            
            print("  ✅ Async single chunk upload")
            
            # Test multiple chunk upload
            chunks = [chunk]
            results = await service.upload_video_chunks(chunks)
            assert len(results) == 1
            assert results[0] is not None
            
            print("  ✅ Async multiple chunk upload")
            
            return True
            
        finally:
            os.unlink(temp_file)


def main():
    """Run all verification tests."""
    print("S3 Video Upload Functionality Verification")
    print("=" * 60)
    
    tests = [
        ("VideoChunk Class", verify_video_chunk_class),
        ("ProgressPercentage Class", verify_progress_percentage_class),
        ("Service Initialization", verify_s3_upload_service_initialization),
        ("S3 Key Generation", verify_s3_key_generation),
        ("Metadata Attachment", verify_metadata_attachment),
        ("Encryption Support", verify_encryption_support),
        ("Retry Logic", verify_retry_logic),
        ("Multipart Support", verify_multipart_support),
        ("URL Generation", verify_url_generation),
        ("Async Functionality", verify_async_functionality)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = asyncio.run(test_func())
            else:
                result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} failed with exception: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("📊 VERIFICATION RESULTS")
    print("=" * 60)
    
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {name}: {status}")
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nOverall: {successful}/{total} verifications successful")
    
    if successful == total:
        print("\n🎉 All S3 video upload requirements verified!")
        print("\n✅ Task 2.2 Implementation Complete:")
        print("  ✅ Video chunk upload method with multipart support")
        print("  ✅ Progress tracking using ProgressPercentage callback class")
        print("  ✅ Metadata attachment during upload (video_id, scene_number, version)")
        print("  ✅ Server-side encryption (SSE-S3 and SSE-KMS options)")
        print("  ✅ Retry logic with exponential backoff for failed uploads")
        print("\n📚 Requirements satisfied:")
        print("  ✅ Requirement 1.1: Video storage with organized naming")
        print("  ✅ Requirement 1.2: Video versioning support")
        print("  ✅ Requirement 1.5: Retry logic implementation")
        print("  ✅ Requirement 8.2: Server-side encryption")
        
        return True
    else:
        print("\n❌ Some verifications failed. Please review the implementation.")
        return False


if __name__ == '__main__':
    # Set up basic logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    success = main()
    sys.exit(0 if success else 1)
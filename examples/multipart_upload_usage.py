#!/usr/bin/env python3
"""
Example usage of MultipartUploadHandler

Demonstrates how to use the multipart upload handler for large file uploads
with resume capability, progress tracking, and error handling.
"""

import os
import sys
import asyncio
import tempfile
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aws.multipart_upload_handler import MultipartUploadHandler
from aws.config import AWSConfig
from aws.credentials import AWSCredentialsManager
from aws.exceptions import AWSS3Error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_large_test_file(size_mb: int) -> str:
    """Create a large test file for upload demonstration."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    
    # Write test data in chunks
    chunk_size = 1024 * 1024  # 1MB chunks
    data_chunk = b'X' * chunk_size
    
    print(f"Creating {size_mb}MB test file...")
    for i in range(size_mb):
        temp_file.write(data_chunk)
        if (i + 1) % 10 == 0:
            print(f"  Written {i + 1}MB...")
    
    temp_file.close()
    print(f"✅ Created test file: {temp_file.name}")
    return temp_file.name


async def upload_progress_callback(uploaded: int, total: int, percentage: float):
    """Custom progress callback for upload tracking."""
    mb_uploaded = uploaded / (1024 * 1024)
    mb_total = total / (1024 * 1024)
    print(f"📊 Upload Progress: {mb_uploaded:.1f}/{mb_total:.1f} MB ({percentage:.1f}%)")


async def demonstrate_multipart_upload():
    """Demonstrate multipart upload functionality."""
    print("🚀 MultipartUploadHandler Usage Example")
    print("=" * 50)
    
    try:
        # 1. Configure AWS settings
        print("\n1️⃣ Configuring AWS settings...")
        config = AWSConfig(
            region='us-east-1',
            video_bucket_name='my-video-bucket',
            code_bucket_name='my-code-bucket',
            metadata_table_name='video-metadata',
            multipart_threshold=50 * 1024 * 1024,  # 50MB threshold
            chunk_size=8 * 1024 * 1024,  # 8MB chunks
            max_concurrent_uploads=3,
            max_retries=3,
            retry_backoff_base=2,
            enable_encryption=True,
            kms_key_id='alias/my-s3-key'  # Optional KMS key
        )
        
        # 2. Initialize credentials and handler
        print("\n2️⃣ Initializing multipart upload handler...")
        credentials_manager = AWSCredentialsManager(config)
        handler = MultipartUploadHandler(config, credentials_manager)
        
        print(f"✅ Handler initialized: {handler}")
        
        # 3. Create a large test file
        print("\n3️⃣ Creating large test file...")
        large_file = create_large_test_file(100)  # 100MB file
        
        try:
            file_size = os.path.getsize(large_file)
            print(f"📁 File size: {file_size:,} bytes ({file_size / (1024*1024):.1f} MB)")
            
            # 4. Upload with multipart
            print("\n4️⃣ Starting multipart upload...")
            
            # Prepare extra arguments for upload
            extra_args = {
                'Metadata': {
                    'project_id': 'demo-project',
                    'video_id': 'demo-video-001',
                    'content_type': 'demo-content',
                    'created_by': 'multipart-demo'
                },
                'ContentType': 'video/mp4'
            }
            
            # Upload the file
            s3_url = await handler.upload_large_file(
                file_path=large_file,
                bucket='my-video-bucket',
                key='demos/large-video.mp4',
                extra_args=extra_args,
                progress_callback=upload_progress_callback
            )
            
            print(f"✅ Upload completed successfully!")
            print(f"📍 S3 URL: {s3_url}")
            
        finally:
            # Clean up test file
            if os.path.exists(large_file):
                os.unlink(large_file)
                print(f"🧹 Cleaned up test file: {large_file}")
        
        # 5. Demonstrate upload management
        print("\n5️⃣ Upload management features...")
        
        # List active uploads
        active_uploads = await handler.list_active_uploads()
        print(f"📋 Active uploads: {len(active_uploads)}")
        
        # Cleanup abandoned uploads (older than 24 hours)
        cleanup_count = await handler.cleanup_abandoned_uploads('my-video-bucket', 24)
        print(f"🧹 Cleaned up {cleanup_count} abandoned uploads")
        
        print("\n🎉 Multipart upload demonstration completed successfully!")
        
    except AWSS3Error as e:
        print(f"\n❌ S3 Error: {e}")
        logger.error(f"S3 operation failed: {e}")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.exception("Demonstration failed")


async def demonstrate_upload_resume():
    """Demonstrate upload resume functionality."""
    print("\n🔄 Upload Resume Demonstration")
    print("=" * 40)
    
    try:
        # This would typically happen when an upload is interrupted
        print("📝 In a real scenario, you would:")
        print("  1. Start a large file upload")
        print("  2. Have it interrupted (network issue, etc.)")
        print("  3. Restart the upload - it will automatically resume")
        print("  4. Only upload the remaining parts")
        
        print("\n✅ Resume functionality is built into the handler")
        print("   - Automatically detects existing incomplete uploads")
        print("   - Verifies integrity of existing parts")
        print("   - Uploads only missing parts")
        print("   - Completes the multipart upload")
        
    except Exception as e:
        print(f"❌ Resume demonstration error: {e}")


async def demonstrate_error_handling():
    """Demonstrate error handling scenarios."""
    print("\n🚨 Error Handling Demonstration")
    print("=" * 40)
    
    try:
        # Configure with invalid settings to show error handling
        config = AWSConfig(
            region='us-east-1',
            video_bucket_name='non-existent-bucket-12345',
            code_bucket_name='test-code-bucket',
            metadata_table_name='test-metadata'
        )
        
        credentials_manager = AWSCredentialsManager(config)
        handler = MultipartUploadHandler(config, credentials_manager)
        
        print("✅ Error handling features:")
        print("  - Exponential backoff retry logic")
        print("  - Automatic upload abortion on failure")
        print("  - Graceful degradation for network issues")
        print("  - Detailed error logging and reporting")
        print("  - Upload cleanup and resource management")
        
        # Show exception types
        from aws.exceptions import AWSS3Error, AWSRetryableError, AWSNonRetryableError
        
        print("\n📋 Exception types handled:")
        print("  - AWSS3Error: General S3 operation errors")
        print("  - AWSRetryableError: Temporary failures (will retry)")
        print("  - AWSNonRetryableError: Permanent failures (won't retry)")
        
    except Exception as e:
        print(f"❌ Error handling demonstration error: {e}")


def main():
    """Main demonstration function."""
    print("📚 MultipartUploadHandler Usage Examples")
    print("=" * 60)
    
    # Run async demonstrations
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Note: These demonstrations show the API usage
        # In a real environment, you would need valid AWS credentials
        # and existing S3 buckets
        
        print("⚠️  Note: This demonstration shows API usage.")
        print("   For actual uploads, ensure you have:")
        print("   - Valid AWS credentials configured")
        print("   - Existing S3 buckets with proper permissions")
        print("   - Network connectivity to AWS")
        
        # Run demonstrations
        loop.run_until_complete(demonstrate_multipart_upload())
        loop.run_until_complete(demonstrate_upload_resume())
        loop.run_until_complete(demonstrate_error_handling())
        
        print("\n" + "=" * 60)
        print("📋 USAGE SUMMARY")
        print("=" * 60)
        print("✅ MultipartUploadHandler provides:")
        print("  🔧 Automatic multipart detection based on file size")
        print("  📊 Progress tracking with custom callbacks")
        print("  🔄 Upload resume for interrupted transfers")
        print("  🛡️  Integrity verification using ETags")
        print("  🧹 Automatic cleanup of failed uploads")
        print("  ⚡ Concurrent part uploads for speed")
        print("  🔁 Exponential backoff retry logic")
        print("  📈 Active upload monitoring and management")
        
        print("\n🎯 Perfect for:")
        print("  📹 Large video file uploads")
        print("  💾 Backup and archival operations")
        print("  🔄 Resumable file transfers")
        print("  🌐 Unreliable network conditions")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        logger.exception("Main demonstration failed")
        
    finally:
        loop.close()


if __name__ == "__main__":
    main()
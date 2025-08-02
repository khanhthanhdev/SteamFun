"""
S3 Video Upload Service Usage Example

Demonstrates how to use the S3VideoUploadService for uploading video chunks
with multipart support, progress tracking, metadata attachment, encryption,
and retry logic.
"""

import os
import sys
import asyncio
import tempfile
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aws import (
    AWSConfig, 
    AWSCredentialsManager, 
    S3VideoUploadService, 
    VideoChunk, 
    ProgressPercentage,
    setup_aws_logging,
    AWSS3Error
)


def create_sample_video_files():
    """Create sample video files for testing."""
    video_files = []
    
    print("📹 Creating sample video files...")
    
    for i in range(1, 4):  # Create 3 scene files
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'_scene_{i}.mp4') as f:
            # Create fake video content (larger for multipart testing)
            content = f"Fake video content for scene {i} ".encode() * 50000  # ~1MB per file
            f.write(content)
            video_files.append(f.name)
            print(f"  ✅ Created scene {i}: {f.name} ({len(content):,} bytes)")
    
    # Create combined video file
    with tempfile.NamedTemporaryFile(delete=False, suffix='_combined.mp4') as f:
        content = b"Combined video content " * 100000  # ~2MB
        f.write(content)
        combined_file = f.name
        print(f"  ✅ Created combined video: {f.name} ({len(content):,} bytes)")
    
    return video_files, combined_file


def cleanup_files(files):
    """Clean up temporary files."""
    print("\n🧹 Cleaning up temporary files...")
    for file_path in files:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                print(f"  ✅ Deleted: {file_path}")
        except Exception as e:
            print(f"  ❌ Failed to delete {file_path}: {e}")


async def demonstrate_basic_upload():
    """Demonstrate basic video chunk upload."""
    print("\n" + "="*60)
    print("🚀 BASIC VIDEO UPLOAD DEMONSTRATION")
    print("="*60)
    
    # Create sample files
    scene_files, combined_file = create_sample_video_files()
    
    try:
        # Configure AWS (using environment variables or defaults)
        config = AWSConfig(
            region='us-east-1',
            video_bucket_name=os.getenv('AWS_S3_VIDEO_BUCKET', 'my-test-video-bucket'),
            code_bucket_name=os.getenv('AWS_S3_CODE_BUCKET', 'my-test-code-bucket'),
            enable_aws_upload=True,
            enable_encryption=True,
            max_retries=2,
            multipart_threshold=500000,  # 500KB for testing
            max_concurrent_uploads=2
        )
        
        print(f"📋 Configuration:")
        print(f"  Region: {config.region}")
        print(f"  Video Bucket: {config.video_bucket_name}")
        print(f"  Encryption: {config.enable_encryption}")
        print(f"  Multipart Threshold: {config.multipart_threshold:,} bytes")
        
        # Set up logging
        logger = setup_aws_logging(config)
        
        # Initialize credentials and upload service
        print("\n🔐 Initializing AWS credentials...")
        creds_manager = AWSCredentialsManager(config)
        
        # Test credentials
        creds_info = creds_manager.get_credentials_info()
        print(f"  Status: {creds_info.get('status', 'unknown')}")
        if creds_info.get('account_id'):
            print(f"  Account: {creds_info['account_id']}")
        
        print("\n📤 Initializing S3 upload service...")
        upload_service = S3VideoUploadService(config, creds_manager)
        
        # Create video chunks
        print("\n📦 Preparing video chunks...")
        chunks = []
        for i, file_path in enumerate(scene_files, 1):
            chunk = VideoChunk(
                file_path=file_path,
                project_id="demo_project",
                video_id="demo_video_001",
                scene_number=i,
                version=1,
                metadata={
                    "scene_name": f"Scene {i}",
                    "duration": "30s",
                    "resolution": "1920x1080"
                }
            )
            chunks.append(chunk)
            print(f"  ✅ Chunk {i}: Scene {i} ({os.path.getsize(file_path):,} bytes)")
        
        # Upload chunks with progress tracking
        print(f"\n🚀 Uploading {len(chunks)} video chunks...")
        
        def progress_callback(seen, total, percentage):
            """Custom progress callback."""
            if int(percentage) % 10 == 0:  # Log every 10%
                print(f"    Progress: {percentage:.1f}% ({seen:,}/{total:,} bytes)")
        
        try:
            upload_results = await upload_service.upload_video_chunks(chunks, progress_callback)
            
            print(f"\n✅ Upload completed!")
            print(f"  Successful uploads: {sum(1 for r in upload_results if r is not None)}/{len(upload_results)}")
            
            for i, result in enumerate(upload_results, 1):
                if result:
                    print(f"  ✅ Scene {i}: {result}")
                else:
                    print(f"  ❌ Scene {i}: Upload failed")
        
        except AWSS3Error as e:
            print(f"❌ Upload failed: {e}")
            print(f"  Error code: {e.error_code}")
            print(f"  Bucket: {e.bucket}")
            print(f"  Operation: {e.operation}")
            return False
        
        # Upload combined video
        print(f"\n🎬 Uploading combined video...")
        try:
            combined_url = await upload_service.upload_combined_video(
                file_path=combined_file,
                project_id="demo_project",
                video_id="demo_video_001",
                version=1,
                metadata={
                    "type": "combined",
                    "total_scenes": str(len(chunks)),
                    "total_duration": "90s"
                },
                progress_callback=progress_callback
            )
            
            print(f"✅ Combined video uploaded: {combined_url}")
            
        except AWSS3Error as e:
            print(f"❌ Combined video upload failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        return False
        
    finally:
        # Clean up files
        cleanup_files(scene_files + [combined_file])


async def demonstrate_error_handling():
    """Demonstrate error handling and retry logic."""
    print("\n" + "="*60)
    print("⚠️  ERROR HANDLING DEMONSTRATION")
    print("="*60)
    
    # Create a sample file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
        f.write(b"test content")
        test_file = f.name
    
    try:
        # Configure with invalid bucket to trigger errors
        config = AWSConfig(
            region='us-east-1',
            video_bucket_name='nonexistent-bucket-12345',
            code_bucket_name='nonexistent-code-bucket-12345',
            enable_aws_upload=True,
            max_retries=1  # Reduce retries for faster demo
        )
        
        print(f"📋 Testing with invalid bucket: {config.video_bucket_name}")
        
        # Set up service
        creds_manager = AWSCredentialsManager(config)
        upload_service = S3VideoUploadService(config, creds_manager)
        
        # Create test chunk
        chunk = VideoChunk(
            file_path=test_file,
            project_id="error_test",
            video_id="error_video",
            scene_number=1,
            version=1
        )
        
        print("\n🧪 Testing error handling...")
        try:
            await upload_service.upload_video_chunks([chunk])
            print("❌ Expected error but upload succeeded")
            
        except AWSS3Error as e:
            print(f"✅ Caught expected S3 error: {e.error_code}")
            print(f"  Message: {e.message}")
            print(f"  Bucket: {e.bucket}")
            
        except Exception as e:
            print(f"✅ Caught expected error: {type(e).__name__}: {e}")
        
        # Test retry logic with retryable error
        print("\n🔄 Testing retry logic...")
        print("  (This would normally retry on temporary failures)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling demo failed: {e}")
        return False
        
    finally:
        cleanup_files([test_file])


def demonstrate_url_generation():
    """Demonstrate URL generation without uploading."""
    print("\n" + "="*60)
    print("🔗 URL GENERATION DEMONSTRATION")
    print("="*60)
    
    try:
        config = AWSConfig(
            video_bucket_name='my-video-bucket',
            enable_aws_upload=False  # Don't need actual upload
        )
        
        # Mock credentials manager for URL generation
        from unittest.mock import Mock
        creds_manager = Mock()
        upload_service = S3VideoUploadService(config, creds_manager)
        
        print("📋 Generating S3 URLs for different video types:")
        
        # Scene chunk URLs
        for scene in range(1, 4):
            url = upload_service.get_video_url("my_project", "video_123", scene, 1)
            print(f"  Scene {scene}: {url}")
        
        # Combined video URL
        combined_url = upload_service.get_video_url("my_project", "video_123", 0, 1)
        print(f"  Combined: {combined_url}")
        
        # Different version
        v2_url = upload_service.get_video_url("my_project", "video_123", 1, 2)
        print(f"  Version 2: {v2_url}")
        
        return True
        
    except Exception as e:
        print(f"❌ URL generation demo failed: {e}")
        return False


async def main():
    """Run all demonstrations."""
    print("S3 Video Upload Service - Usage Examples")
    print("=" * 60)
    
    # Check environment
    print("🔍 Environment Check:")
    print(f"  Python: {sys.version}")
    print(f"  Working directory: {os.getcwd()}")
    
    # Check AWS configuration
    video_bucket = os.getenv('AWS_S3_VIDEO_BUCKET')
    if video_bucket:
        print(f"  Video bucket (from env): {video_bucket}")
    else:
        print("  ⚠️  AWS_S3_VIDEO_BUCKET not set - using default test bucket")
    
    # Run demonstrations
    demos = [
        ("URL Generation", demonstrate_url_generation),
        ("Error Handling", demonstrate_error_handling),
        ("Basic Upload", demonstrate_basic_upload)
    ]
    
    results = []
    for name, demo_func in demos:
        try:
            if asyncio.iscoroutinefunction(demo_func):
                result = await demo_func()
            else:
                result = demo_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 DEMONSTRATION SUMMARY")
    print("="*60)
    
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {name}: {status}")
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nOverall: {successful}/{total} demonstrations successful")
    
    if successful == total:
        print("\n🎉 All demonstrations completed successfully!")
        print("\n📚 Next steps:")
        print("  1. Configure your AWS credentials and S3 bucket")
        print("  2. Update your .env file with AWS settings")
        print("  3. Integrate S3VideoUploadService with your LangGraph agents")
        print("  4. Implement DynamoDB metadata management (Task 3)")
    else:
        print("\n⚠️  Some demonstrations failed. Check the errors above.")
    
    return successful == total


if __name__ == '__main__':
    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run demonstrations
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
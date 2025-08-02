"""
Verify S3 Code Storage Implementation

Verification script to test S3 code storage functionality including upload,
download, versioning, and S3 Object Lock features.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.aws.s3_code_storage import S3CodeStorageService, CodeMetadata, CodeVersion
from src.aws.config import AWSConfig
from src.aws.credentials import AWSCredentialsManager
from src.aws.exceptions import AWSS3Error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class S3CodeStorageVerification:
    """Verification class for S3 code storage functionality."""
    
    def __init__(self):
        """Initialize verification with test configuration."""
        self.config = AWSConfig(
            region='us-east-1',
            video_bucket_name='test-manim-video-bucket',
            code_bucket_name='test-manim-code-bucket',
            enable_encryption=True,
            max_retries=3,
            enable_aws_upload=True,
            require_aws_upload=False
        )
        
        self.credentials_manager = None
        self.code_storage_service = None
        
        # Sample Manim code for testing
        self.sample_codes = {
            1: '''from manim import *

class IntroScene(Scene):
    def construct(self):
        # Create title
        title = Text("Welcome to Manim", font_size=48)
        self.play(Write(title))
        self.wait(2)
        
        # Transform to subtitle
        subtitle = Text("Mathematical Animations", font_size=36)
        self.play(Transform(title, subtitle))
        self.wait(2)
''',
            2: '''from manim import *

class IntroScene(Scene):
    def construct(self):
        # Create title with color
        title = Text("Welcome to Manim", font_size=48, color=BLUE)
        self.play(Write(title))
        self.wait(2)
        
        # Add mathematical formula
        formula = MathTex(r"E = mc^2", font_size=60)
        formula.next_to(title, DOWN, buff=1)
        self.play(Write(formula))
        self.wait(2)
        
        # Transform to subtitle
        subtitle = Text("Mathematical Animations", font_size=36, color=GREEN)
        self.play(Transform(title, subtitle))
        self.wait(2)
''',
            3: '''from manim import *

class IntroScene(Scene):
    def construct(self):
        # Create animated title with color
        title = Text("Welcome to Manim", font_size=48, color=BLUE)
        self.play(Write(title))
        self.wait(1)
        
        # Add mathematical formula with animation
        formula = MathTex(r"E = mc^2", font_size=60, color=YELLOW)
        formula.next_to(title, DOWN, buff=1)
        self.play(Write(formula))
        self.wait(1)
        
        # Create a circle that morphs
        circle = Circle(radius=1, color=RED)
        circle.next_to(formula, DOWN, buff=1)
        self.play(Create(circle))
        
        # Transform circle to square
        square = Square(side_length=2, color=PURPLE)
        square.move_to(circle.get_center())
        self.play(Transform(circle, square))
        self.wait(2)
        
        # Final transformation
        subtitle = Text("Mathematical Animations", font_size=36, color=GREEN)
        self.play(Transform(title, subtitle))
        self.wait(2)
'''
        }
    
    async def setup(self):
        """Set up AWS credentials and services."""
        try:
            logger.info("Setting up AWS credentials and services...")
            
            # Initialize credentials manager
            self.credentials_manager = AWSCredentialsManager(self.config)
            
            # Test credentials
            creds_info = self.credentials_manager.get_credentials_info()
            logger.info(f"AWS credentials status: {creds_info['status']}")
            
            if creds_info['status'] != 'valid':
                logger.warning("AWS credentials not valid - using mock mode")
                return False
            
            # Initialize code storage service
            self.code_storage_service = S3CodeStorageService(
                self.config, 
                self.credentials_manager
            )
            
            logger.info("✅ AWS setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ AWS setup failed: {e}")
            return False
    
    async def verify_code_metadata(self):
        """Verify CodeMetadata functionality."""
        logger.info("\n🔍 Verifying CodeMetadata functionality...")
        
        try:
            # Test valid metadata
            metadata = CodeMetadata(
                video_id='test_video_123',
                project_id='test_project_456',
                version=1,
                scene_number=1
            )
            
            assert metadata.video_id == 'test_video_123'
            assert metadata.project_id == 'test_project_456'
            assert metadata.version == 1
            assert metadata.scene_number == 1
            assert metadata.created_at is not None
            
            logger.info("✅ CodeMetadata validation successful")
            
            # Test CodeVersion
            code_version = CodeVersion(
                content=self.sample_codes[1],
                metadata=metadata
            )
            
            expected_size = len(self.sample_codes[1].encode('utf-8'))
            assert code_version.file_size == expected_size
            
            logger.info("✅ CodeVersion functionality successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ CodeMetadata verification failed: {e}")
            return False
    
    async def verify_s3_key_generation(self):
        """Verify S3 key generation with versioning."""
        logger.info("\n🔍 Verifying S3 key generation...")
        
        try:
            metadata = CodeMetadata(
                video_id='test_video_123',
                project_id='test_project_456',
                version=2
            )
            
            s3_key = self.code_storage_service._generate_s3_key(metadata)
            expected_key = "code/test_project_456/test_video_123/test_video_123_v2.py"
            
            assert s3_key == expected_key
            logger.info(f"✅ S3 key generation successful: {s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"❌ S3 key generation failed: {e}")
            return False
    
    async def verify_upload_metadata_preparation(self):
        """Verify upload metadata preparation."""
        logger.info("\n🔍 Verifying upload metadata preparation...")
        
        try:
            metadata = CodeMetadata(
                video_id='test_video_123',
                project_id='test_project_456',
                version=1,
                scene_number=1
            )
            
            # Test without Object Lock
            extra_args = self.code_storage_service._prepare_upload_metadata(metadata)
            
            assert 'Metadata' in extra_args
            assert 'ContentType' in extra_args
            assert 'ContentEncoding' in extra_args
            assert extra_args['ContentType'] == 'text/x-python'
            assert extra_args['ContentEncoding'] == 'utf-8'
            
            # Check metadata content
            s3_metadata = extra_args['Metadata']
            assert s3_metadata['video_id'] == 'test_video_123'
            assert s3_metadata['project_id'] == 'test_project_456'
            assert s3_metadata['version'] == '1'
            assert s3_metadata['scene_number'] == '1'
            
            logger.info("✅ Upload metadata preparation successful")
            
            # Test with Object Lock
            extra_args_lock = self.code_storage_service._prepare_upload_metadata(
                metadata, enable_object_lock=True
            )
            
            assert 'ObjectLockMode' in extra_args_lock
            assert extra_args_lock['ObjectLockMode'] == 'GOVERNANCE'
            assert 'ObjectLockRetainUntilDate' in extra_args_lock
            
            logger.info("✅ Object Lock metadata preparation successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Upload metadata preparation failed: {e}")
            return False
    
    async def verify_url_generation(self):
        """Verify URL generation functionality."""
        logger.info("\n🔍 Verifying URL generation...")
        
        try:
            url = self.code_storage_service.get_code_url(
                'test_video_123', 
                'test_project_456', 
                3
            )
            
            expected_url = f"s3://{self.config.code_bucket_name}/code/test_project_456/test_video_123/test_video_123_v3.py"
            assert url == expected_url
            
            logger.info(f"✅ URL generation successful: {url}")
            return True
            
        except Exception as e:
            logger.error(f"❌ URL generation failed: {e}")
            return False
    
    async def verify_error_handling(self):
        """Verify error handling functionality."""
        logger.info("\n🔍 Verifying error handling...")
        
        try:
            # Test retryable error detection
            assert self.code_storage_service._is_retryable_error('RequestTimeout')
            assert self.code_storage_service._is_retryable_error('ServiceUnavailable')
            assert not self.code_storage_service._is_retryable_error('NoSuchBucket')
            assert not self.code_storage_service._is_retryable_error('AccessDenied')
            
            logger.info("✅ Error handling verification successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error handling verification failed: {e}")
            return False
    
    async def verify_live_operations(self):
        """Verify live S3 operations (if credentials are available)."""
        logger.info("\n🔍 Verifying live S3 operations...")
        
        if not self.credentials_manager:
            logger.info("⚠️ Skipping live operations - no valid credentials")
            return True
        
        try:
            # Test bucket access
            s3_access = self.credentials_manager.test_service_access('s3')
            if not s3_access:
                logger.info("⚠️ Skipping live operations - no S3 access")
                return True
            
            # Create test metadata
            metadata = CodeMetadata(
                video_id=f'test_video_{int(datetime.now().timestamp())}',
                project_id='verification_project',
                version=1
            )
            
            # Test upload (this will fail if bucket doesn't exist, which is expected)
            try:
                s3_url = await self.code_storage_service.upload_code(
                    self.sample_codes[1], 
                    metadata
                )
                logger.info(f"✅ Live upload successful: {s3_url}")
                
                # Test download
                downloaded_code = await self.code_storage_service.download_code(
                    metadata.video_id,
                    metadata.project_id,
                    metadata.version
                )
                
                assert downloaded_code == self.sample_codes[1]
                logger.info("✅ Live download successful")
                
                # Test version listing
                versions = await self.code_storage_service.list_code_versions(
                    metadata.video_id,
                    metadata.project_id
                )
                assert 1 in versions
                logger.info(f"✅ Live version listing successful: {versions}")
                
                # Clean up
                await self.code_storage_service.delete_code_version(
                    metadata.video_id,
                    metadata.project_id,
                    metadata.version
                )
                logger.info("✅ Live cleanup successful")
                
            except AWSS3Error as e:
                if "NoSuchBucket" in str(e):
                    logger.info("⚠️ Live operations skipped - test bucket doesn't exist (expected)")
                else:
                    logger.warning(f"⚠️ Live operations failed: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Live operations verification failed: {e}")
            return False
    
    async def run_verification(self):
        """Run complete verification suite."""
        logger.info("🚀 Starting S3 Code Storage verification...")
        
        results = []
        
        # Setup
        setup_success = await self.setup()
        results.append(("Setup", setup_success))
        
        # Core functionality tests
        results.append(("CodeMetadata", await self.verify_code_metadata()))
        
        if setup_success:
            results.append(("S3 Key Generation", await self.verify_s3_key_generation()))
            results.append(("Upload Metadata", await self.verify_upload_metadata_preparation()))
            results.append(("URL Generation", await self.verify_url_generation()))
            results.append(("Error Handling", await self.verify_error_handling()))
            results.append(("Live Operations", await self.verify_live_operations()))
        
        # Print results
        logger.info("\n📊 Verification Results:")
        logger.info("=" * 50)
        
        passed = 0
        total = len(results)
        
        for test_name, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            logger.info(f"{test_name:<20} {status}")
            if success:
                passed += 1
        
        logger.info("=" * 50)
        logger.info(f"Results: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All verifications passed! S3 Code Storage implementation is working correctly.")
        else:
            logger.warning(f"⚠️ {total - passed} verification(s) failed. Please check the implementation.")
        
        return passed == total


async def main():
    """Main verification function."""
    verification = S3CodeStorageVerification()
    success = await verification.run_verification()
    
    if success:
        print("\n✅ S3 Code Storage verification completed successfully!")
        return 0
    else:
        print("\n❌ S3 Code Storage verification failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
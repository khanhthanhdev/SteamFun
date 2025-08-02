"""
S3 Code Storage Usage Example

Demonstrates how to use the S3 Code Storage Service for uploading, downloading,
and managing Manim code with versioning and S3 Object Lock.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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


async def basic_code_upload_example():
    """Example: Basic code upload with versioning."""
    print("\n🔧 Example 1: Basic Code Upload with Versioning")
    print("=" * 60)
    
    # Configure AWS
    config = AWSConfig(
        region='us-east-1',
        code_bucket_name='my-manim-code-bucket',
        enable_encryption=True,
        enable_aws_upload=True
    )
    
    # Initialize services
    credentials_manager = AWSCredentialsManager(config)
    code_storage = S3CodeStorageService(config, credentials_manager)
    
    # Sample Manim code
    manim_code = '''from manim import *

class HelloWorld(Scene):
    def construct(self):
        text = Text("Hello, World!")
        self.play(Write(text))
        self.wait(2)
'''
    
    # Create metadata for version 1
    metadata_v1 = CodeMetadata(
        video_id='hello_world_demo',
        project_id='tutorial_project',
        version=1,
        scene_number=1
    )
    
    try:
        # Upload code
        s3_url = await code_storage.upload_code(manim_code, metadata_v1)
        print(f"✅ Code uploaded successfully: {s3_url}")
        
        # Generate URL without uploading
        url = code_storage.get_code_url('hello_world_demo', 'tutorial_project', 1)
        print(f"📍 Generated URL: {url}")
        
    except AWSS3Error as e:
        print(f"❌ Upload failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


async def code_versioning_example():
    """Example: Code versioning and editing workflow."""
    print("\n🔧 Example 2: Code Versioning and Editing Workflow")
    print("=" * 60)
    
    # Configure AWS
    config = AWSConfig(
        region='us-east-1',
        code_bucket_name='my-manim-code-bucket',
        enable_encryption=True,
        enable_aws_upload=True
    )
    
    # Initialize services
    credentials_manager = AWSCredentialsManager(config)
    code_storage = S3CodeStorageService(config, credentials_manager)
    
    # Different versions of the same scene
    code_versions = {
        1: '''from manim import *

class AnimationDemo(Scene):
    def construct(self):
        circle = Circle()
        self.play(Create(circle))
        self.wait(1)
''',
        2: '''from manim import *

class AnimationDemo(Scene):
    def construct(self):
        circle = Circle(color=BLUE)
        self.play(Create(circle))
        self.play(circle.animate.shift(RIGHT * 2))
        self.wait(1)
''',
        3: '''from manim import *

class AnimationDemo(Scene):
    def construct(self):
        circle = Circle(color=BLUE)
        square = Square(color=RED)
        square.next_to(circle, RIGHT, buff=1)
        
        self.play(Create(circle), Create(square))
        self.play(circle.animate.shift(RIGHT * 2))
        self.play(Transform(circle, square))
        self.wait(2)
'''
    }
    
    video_id = 'animation_demo'
    project_id = 'advanced_tutorial'
    
    try:
        # Upload multiple versions
        uploaded_urls = []
        for version, code in code_versions.items():
            metadata = CodeMetadata(
                video_id=video_id,
                project_id=project_id,
                version=version
            )
            
            s3_url = await code_storage.upload_code(code, metadata)
            uploaded_urls.append(s3_url)
            print(f"✅ Version {version} uploaded: {s3_url}")
        
        # List all versions
        versions = await code_storage.list_code_versions(video_id, project_id)
        print(f"📋 Available versions: {versions}")
        
        # Download latest version
        latest_code = await code_storage.download_latest_code(video_id, project_id)
        print(f"📥 Downloaded latest version ({max(versions)} lines)")
        
        # Download specific version
        v2_code = await code_storage.download_code(video_id, project_id, 2)
        print(f"📥 Downloaded version 2 ({len(v2_code.splitlines())} lines)")
        
    except AWSS3Error as e:
        print(f"❌ Operation failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


async def batch_upload_example():
    """Example: Batch upload of multiple code versions."""
    print("\n🔧 Example 3: Batch Upload of Multiple Code Versions")
    print("=" * 60)
    
    # Configure AWS
    config = AWSConfig(
        region='us-east-1',
        code_bucket_name='my-manim-code-bucket',
        enable_encryption=True,
        enable_aws_upload=True
    )
    
    # Initialize services
    credentials_manager = AWSCredentialsManager(config)
    code_storage = S3CodeStorageService(config, credentials_manager)
    
    # Create multiple code versions for batch upload
    code_versions = []
    
    base_code = '''from manim import *

class Scene{scene_num}(Scene):
    def construct(self):
        text = Text("Scene {scene_num}")
        self.play(Write(text))
        self.wait(2)
'''
    
    for scene_num in range(1, 4):
        code_content = base_code.format(scene_num=scene_num)
        metadata = CodeMetadata(
            video_id='multi_scene_video',
            project_id='batch_project',
            version=scene_num,
            scene_number=scene_num
        )
        
        code_version = CodeVersion(
            content=code_content,
            metadata=metadata
        )
        code_versions.append(code_version)
    
    try:
        # Batch upload
        results = await code_storage.upload_code_versions(code_versions)
        
        successful_uploads = [url for url in results if url is not None]
        print(f"✅ Batch upload completed: {len(successful_uploads)}/{len(code_versions)} successful")
        
        for i, url in enumerate(results, 1):
            if url:
                print(f"  Scene {i}: {url}")
            else:
                print(f"  Scene {i}: ❌ Failed")
        
    except AWSS3Error as e:
        print(f"❌ Batch upload failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


async def object_lock_example():
    """Example: Using S3 Object Lock for critical code versions."""
    print("\n🔧 Example 4: S3 Object Lock for Critical Code Versions")
    print("=" * 60)
    
    # Configure AWS
    config = AWSConfig(
        region='us-east-1',
        code_bucket_name='my-manim-code-bucket',
        enable_encryption=True,
        enable_aws_upload=True
    )
    
    # Initialize services
    credentials_manager = AWSCredentialsManager(config)
    code_storage = S3CodeStorageService(config, credentials_manager)
    
    # Critical production code
    production_code = '''from manim import *

class ProductionScene(Scene):
    """
    CRITICAL: This is the final production version.
    DO NOT MODIFY without proper approval.
    """
    def construct(self):
        # Final approved animation sequence
        title = Text("Production Ready", font_size=48, color=GOLD)
        subtitle = Text("Version 1.0", font_size=24, color=WHITE)
        subtitle.next_to(title, DOWN, buff=0.5)
        
        self.play(Write(title))
        self.play(Write(subtitle))
        self.wait(3)
        
        # Complex animation sequence
        circle = Circle(radius=2, color=BLUE)
        square = Square(side_length=3, color=RED)
        
        self.play(Create(circle))
        self.play(Transform(circle, square))
        self.wait(2)
'''
    
    # Create metadata for production version
    metadata = CodeMetadata(
        video_id='production_final',
        project_id='client_project',
        version=1,
        metadata={
            'environment': 'production',
            'approved_by': 'project_manager',
            'approval_date': datetime.now().isoformat()
        }
    )
    
    try:
        # Check if Object Lock is available on bucket
        object_lock_available = await code_storage.enable_object_lock_on_bucket()
        
        if object_lock_available:
            print("🔒 S3 Object Lock is available on bucket")
            
            # Upload with Object Lock enabled
            s3_url = await code_storage.upload_code(
                production_code, 
                metadata, 
                enable_object_lock=True
            )
            print(f"✅ Production code uploaded with Object Lock: {s3_url}")
            
            # Get metadata to verify Object Lock
            code_metadata = await code_storage.get_code_metadata(
                metadata.video_id,
                metadata.project_id,
                metadata.version
            )
            
            if 'object_lock_mode' in code_metadata:
                print(f"🔒 Object Lock Mode: {code_metadata['object_lock_mode']}")
                print(f"🔒 Retain Until: {code_metadata.get('object_lock_retain_until_date')}")
            else:
                print("⚠️ Object Lock not applied (bucket may not support it)")
        else:
            print("⚠️ S3 Object Lock not available on bucket")
            
            # Upload without Object Lock
            s3_url = await code_storage.upload_code(production_code, metadata)
            print(f"✅ Production code uploaded (without Object Lock): {s3_url}")
        
    except AWSS3Error as e:
        print(f"❌ Production upload failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


async def code_management_example():
    """Example: Complete code management workflow."""
    print("\n🔧 Example 5: Complete Code Management Workflow")
    print("=" * 60)
    
    # Configure AWS
    config = AWSConfig(
        region='us-east-1',
        code_bucket_name='my-manim-code-bucket',
        enable_encryption=True,
        enable_aws_upload=True
    )
    
    # Initialize services
    credentials_manager = AWSCredentialsManager(config)
    code_storage = S3CodeStorageService(config, credentials_manager)
    
    video_id = 'workflow_demo'
    project_id = 'management_example'
    
    try:
        # 1. Upload initial version
        initial_code = '''from manim import *

class WorkflowDemo(Scene):
    def construct(self):
        text = Text("Initial Version")
        self.play(Write(text))
        self.wait(1)
'''
        
        metadata_v1 = CodeMetadata(
            video_id=video_id,
            project_id=project_id,
            version=1
        )
        
        s3_url_v1 = await code_storage.upload_code(initial_code, metadata_v1)
        print(f"✅ Initial version uploaded: {s3_url_v1}")
        
        # 2. Upload updated version
        updated_code = '''from manim import *

class WorkflowDemo(Scene):
    def construct(self):
        text = Text("Updated Version", color=BLUE)
        self.play(Write(text))
        
        circle = Circle(color=RED)
        circle.next_to(text, DOWN, buff=1)
        self.play(Create(circle))
        self.wait(2)
'''
        
        metadata_v2 = CodeMetadata(
            video_id=video_id,
            project_id=project_id,
            version=2
        )
        
        s3_url_v2 = await code_storage.upload_code(updated_code, metadata_v2)
        print(f"✅ Updated version uploaded: {s3_url_v2}")
        
        # 3. List all versions
        versions = await code_storage.list_code_versions(video_id, project_id)
        print(f"📋 Available versions: {versions}")
        
        # 4. Get metadata for each version
        for version in versions:
            metadata = await code_storage.get_code_metadata(video_id, project_id, version)
            print(f"📊 Version {version} metadata:")
            print(f"   Size: {metadata['content_length']} bytes")
            print(f"   Modified: {metadata['last_modified']}")
            print(f"   ETag: {metadata['etag']}")
        
        # 5. Download and compare versions
        v1_code = await code_storage.download_code(video_id, project_id, 1)
        v2_code = await code_storage.download_code(video_id, project_id, 2)
        
        print(f"📥 Version 1: {len(v1_code.splitlines())} lines")
        print(f"📥 Version 2: {len(v2_code.splitlines())} lines")
        
        # 6. Download latest version
        latest_code = await code_storage.download_latest_code(video_id, project_id)
        print(f"📥 Latest version: {len(latest_code.splitlines())} lines")
        
        # 7. Clean up (optional)
        print("\n🧹 Cleanup options:")
        print(f"   To delete version 1: delete_code_version('{video_id}', '{project_id}', 1)")
        print(f"   To delete version 2: delete_code_version('{video_id}', '{project_id}', 2)")
        
    except AWSS3Error as e:
        print(f"❌ Workflow failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


async def main():
    """Run all examples."""
    print("🚀 S3 Code Storage Usage Examples")
    print("=" * 60)
    
    examples = [
        ("Basic Code Upload", basic_code_upload_example),
        ("Code Versioning", code_versioning_example),
        ("Batch Upload", batch_upload_example),
        ("Object Lock", object_lock_example),
        ("Complete Workflow", code_management_example)
    ]
    
    for name, example_func in examples:
        try:
            await example_func()
        except Exception as e:
            print(f"❌ Example '{name}' failed: {e}")
        
        print("\n" + "─" * 60)
    
    print("\n🎉 All examples completed!")
    print("\n💡 Tips:")
    print("   - Ensure your AWS credentials are configured")
    print("   - Create the S3 bucket before running examples")
    print("   - Enable S3 Object Lock at bucket creation for Object Lock examples")
    print("   - Check AWS costs for S3 storage and requests")


if __name__ == "__main__":
    asyncio.run(main())
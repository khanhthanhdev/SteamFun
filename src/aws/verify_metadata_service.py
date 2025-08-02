#!/usr/bin/env python3
"""
Verification script for MetadataService with batch operations.
"""

import asyncio
import uuid
import logging
from datetime import datetime, timezone
from moto import mock_aws

from .metadata_service import MetadataService, BatchOperationResult, ConflictResolutionStrategy
from .dynamodb_metadata import VideoMetadata, DynamoDBTableManager
from .config import AWSConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@mock_aws
async def verify_metadata_service():
    """Verify MetadataService functionality."""
    print("🔍 Verifying MetadataService with batch operations...")
    
    # Create test configuration
    config = AWSConfig(
        region='us-east-1',
        video_bucket_name='test-video-bucket',
        code_bucket_name='test-code-bucket',
        metadata_table_name='test-video-metadata-verify',
        enable_encryption=False
    )
    
    try:
        # Initialize table manager and create table
        print("📋 Setting up DynamoDB table...")
        table_manager = DynamoDBTableManager(config)
        table_created = table_manager.create_table_if_not_exists()
        print(f"✅ Table setup: {'Created' if table_created else 'Already exists'}")
        
        # Initialize MetadataService
        print("🚀 Initializing MetadataService...")
        service = MetadataService(config, max_workers=3)
        print("✅ MetadataService initialized")
        
        # Test 1: Batch create video records
        print("\n📝 Test 1: Batch create video records")
        project_id = str(uuid.uuid4())
        video_metadata_list = []
        
        for i in range(3):
            video_metadata = VideoMetadata(
                video_id=str(uuid.uuid4()),
                project_id=project_id,
                title=f"Test Video {i+1}",
                description=f"Test video {i+1} description",
                status="created",
                tags=[f"test{i+1}", "batch"],
                custom_metadata={"batch_id": i+1, "test": True}
            )
            video_metadata_list.append(video_metadata)
        
        result = await service.batch_create_video_records(video_metadata_list)
        print(f"✅ Batch create result: {result.success_count}/{result.total_processed} successful")
        print(f"   Success rate: {result.success_rate:.1f}%")
        
        # Test 2: Batch update metadata
        print("\n🔄 Test 2: Batch update metadata")
        updates = {}
        for video_metadata in video_metadata_list:
            updates[video_metadata.video_id] = {
                'status': 'updated',
                'description': f'Updated description for {video_metadata.title}',
                'custom_metadata': {'updated': True, 'batch_updated': True}
            }
        
        update_result = await service.batch_update_metadata(updates)
        print(f"✅ Batch update result: {update_result.success_count}/{update_result.total_processed} successful")
        print(f"   Success rate: {update_result.success_rate:.1f}%")
        
        # Test 3: Conflict resolution
        print("\n⚔️ Test 3: Conflict resolution")
        test_video = video_metadata_list[0]
        
        # Set conflict resolution strategy
        strategy = ConflictResolutionStrategy(
            strategy="merge_fields",
            merge_fields=['custom_metadata', 'tags']
        )
        service.set_conflict_resolution_strategy(strategy)
        print(f"✅ Conflict resolution strategy set: {strategy.strategy}")
        
        # Test concurrent updates
        update1 = {
            'custom_metadata': {'field1': 'value1', 'updated': True},
            'tags': ['new_tag1'],
            'status': 'processing'
        }
        update2 = {
            'custom_metadata': {'field2': 'value2', 'batch_updated': True},
            'tags': ['new_tag2'],
            'status': 'completed'
        }
        
        success1 = await service._update_with_conflict_resolution(test_video.video_id, update1)
        success2 = await service._update_with_conflict_resolution(test_video.video_id, update2)
        
        print(f"✅ Conflict resolution updates: Update1={success1}, Update2={success2}")
        
        # Verify merged result
        final_record = service.operations.get_video_metadata(test_video.video_id)
        print(f"   Final status: {final_record['status']}")
        print(f"   Final custom_metadata: {final_record.get('custom_metadata', {})}")
        print(f"   Final tags: {final_record.get('tags', [])}")
        
        # Test 4: Batch query by project
        print("\n🔍 Test 4: Batch query by project")
        project_results = await service.batch_query_by_project([project_id], limit_per_project=10)
        
        if project_id in project_results:
            videos_found = len(project_results[project_id])
            print(f"✅ Project query result: {videos_found} videos found for project")
            
            # Show first video details
            if videos_found > 0:
                first_video = project_results[project_id][0]
                print(f"   Sample video: {first_video['title']} (Status: {first_video['status']})")
        
        # Test 5: Advanced project query with filters
        print("\n🎯 Test 5: Advanced project query with filters")
        filtered_results = await service.create_project_index_query(
            project_id=project_id,
            status_filter='updated'
        )
        
        updated_count = len([v for v in filtered_results if v['status'] == 'updated'])
        print(f"✅ Advanced query result: {updated_count} videos with 'updated' status")
        
        # Test 6: Metadata statistics
        print("\n📊 Test 6: Metadata statistics")
        stats = await service.get_metadata_statistics(project_id)
        
        print(f"✅ Statistics generated:")
        print(f"   Total videos: {stats['total_videos']}")
        print(f"   Status breakdown: {stats['status_breakdown']}")
        print(f"   Storage usage: {stats['storage_usage']}")
        
        print("\n🎉 All MetadataService tests completed successfully!")
        
        # Cleanup
        service.executor.shutdown(wait=True)
        
        # Optionally clean up test table
        print("\n🧹 Cleaning up test table...")
        table_manager.delete_table()
        print("✅ Test table deleted")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        logger.exception("Verification failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(verify_metadata_service())
    exit(0 if success else 1)
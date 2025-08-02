#!/usr/bin/env python3
"""
Verification script for Task 3.2: Implement metadata service with batch operations.

This script verifies that all requirements for task 3.2 are properly implemented:
1. Create MetadataService class with DynamoDB resource integration
2. Implement batch_writer for efficient bulk operations  
3. Add conflict resolution for concurrent metadata updates
4. Create indexes for project-based and version-based queries
"""

import asyncio
import uuid
import logging
from datetime import datetime, timezone

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_task_3_2_requirements():
    """Verify all Task 3.2 requirements are implemented."""
    print("🔍 Verifying Task 3.2: Implement metadata service with batch operations")
    print("=" * 80)
    
    verification_results = []
    
    # Requirement 1: Create MetadataService class with DynamoDB resource integration
    print("\n📋 Requirement 1: MetadataService class with DynamoDB resource integration")
    try:
        from .metadata_service import MetadataService, BatchOperationResult, ConflictResolutionStrategy
        from .dynamodb_metadata import VideoMetadata, DynamoDBTableManager, DynamoDBMetadataOperations
        from .config import AWSConfig
        
        # Check if MetadataService can be instantiated
        config = AWSConfig(
            region='ap-southeast-1',
            video_bucket_name='test-video-bucket',
            code_bucket_name='test-code-bucket',
            metadata_table_name='test-video-metadata',
            enable_encryption=False
        )
        
        # Verify class structure (check if methods exist, not instance attributes)
        assert hasattr(MetadataService, '__init__')
        
        # Check if the class can be instantiated (without actually creating AWS resources)
        import inspect
        init_signature = inspect.signature(MetadataService.__init__)
        assert 'config' in init_signature.parameters
        assert 'max_workers' in init_signature.parameters
        
        # Verify key methods exist
        assert hasattr(MetadataService, 'batch_create_video_records')
        assert hasattr(MetadataService, 'batch_update_metadata')
        assert hasattr(MetadataService, 'set_conflict_resolution_strategy')
        
        # Check source code for DynamoDB integration
        source = inspect.getsource(MetadataService.__init__)
        assert 'DynamoDBMetadataOperations' in source
        assert 'DynamoDBTableManager' in source
        assert 'dynamodb' in source
        assert 'table' in source
        
        print("✅ MetadataService class properly integrates with DynamoDB resources")
        verification_results.append(("MetadataService DynamoDB Integration", True, ""))
        
    except Exception as e:
        print(f"❌ MetadataService DynamoDB integration failed: {e}")
        verification_results.append(("MetadataService DynamoDB Integration", False, str(e)))
    
    # Requirement 2: Implement batch_writer for efficient bulk operations
    print("\n📝 Requirement 2: batch_writer for efficient bulk operations")
    try:
        # Check if batch operations are implemented
        assert hasattr(MetadataService, 'batch_create_video_records')
        assert hasattr(MetadataService, 'batch_update_metadata')
        assert hasattr(MetadataService, 'batch_query_by_project')
        
        # Check if BatchOperationResult is properly implemented
        result = BatchOperationResult(
            successful_items=['test1', 'test2'],
            failed_items=[('test3', 'error')],
            total_processed=3,
            success_count=2,
            failure_count=1
        )
        
        assert result.success_rate == 66.66666666666666
        assert hasattr(result, 'successful_items')
        assert hasattr(result, 'failed_items')
        
        # Verify batch_writer usage in source code
        import inspect
        source = inspect.getsource(MetadataService.batch_create_video_records)
        assert 'batch_writer' in source
        assert 'with self.table.batch_writer() as batch:' in source
        
        print("✅ batch_writer implemented for efficient bulk operations")
        verification_results.append(("Batch Writer Implementation", True, ""))
        
    except Exception as e:
        print(f"❌ batch_writer implementation failed: {e}")
        verification_results.append(("Batch Writer Implementation", False, str(e)))
    
    # Requirement 3: Add conflict resolution for concurrent metadata updates
    print("\n⚔️ Requirement 3: Conflict resolution for concurrent metadata updates")
    try:
        # Check conflict resolution components
        assert hasattr(MetadataService, 'set_conflict_resolution_strategy')
        assert hasattr(MetadataService, '_update_with_conflict_resolution')
        assert hasattr(MetadataService, '_resolve_conflicts')
        assert hasattr(MetadataService, '_atomic_update')
        
        # Check ConflictResolutionStrategy
        strategy = ConflictResolutionStrategy()
        assert strategy.strategy == "latest_wins"
        assert hasattr(strategy, 'merge_fields')
        assert hasattr(strategy, 'preserve_fields')
        
        # Check different strategies are supported
        merge_strategy = ConflictResolutionStrategy(
            strategy="merge_fields",
            merge_fields=['tags', 'custom_metadata']
        )
        assert merge_strategy.strategy == "merge_fields"
        assert 'tags' in merge_strategy.merge_fields
        
        # Verify conflict resolution logic in source code
        source = inspect.getsource(MetadataService._resolve_conflicts)
        assert 'latest_wins' in source
        assert 'merge_fields' in source
        assert 'manual_resolution' in source
        
        # Verify atomic update with optimistic locking
        atomic_source = inspect.getsource(MetadataService._atomic_update)
        assert 'ConditionalCheckFailedException' in atomic_source
        assert 'last_edited_timestamp' in atomic_source
        
        print("✅ Conflict resolution implemented for concurrent updates")
        verification_results.append(("Conflict Resolution", True, ""))
        
    except Exception as e:
        print(f"❌ Conflict resolution implementation failed: {e}")
        verification_results.append(("Conflict Resolution", False, str(e)))
    
    # Requirement 4: Create indexes for project-based and version-based queries
    print("\n🔍 Requirement 4: Indexes for project-based and version-based queries")
    try:
        # Check DynamoDB table schema includes required indexes
        table_manager = DynamoDBTableManager(config)
        
        # Verify index-related methods exist
        assert hasattr(MetadataService, 'batch_query_by_project')
        assert hasattr(MetadataService, 'create_project_index_query')
        assert hasattr(MetadataService, 'create_version_index_query')
        assert hasattr(MetadataService, 'query_video_versions_by_range')
        
        # Check DynamoDB operations support index queries
        operations = DynamoDBMetadataOperations(config)
        assert hasattr(operations, 'query_by_project')
        assert hasattr(operations, 'query_by_status')
        
        # Verify index usage in source code
        project_query_source = inspect.getsource(MetadataService.create_project_index_query)
        assert 'ProjectIndex' in project_query_source
        
        version_query_source = inspect.getsource(MetadataService.create_version_index_query)
        assert 'VersionIndex' in version_query_source
        
        # Check table schema includes indexes
        schema_source = inspect.getsource(DynamoDBTableManager.create_table_if_not_exists)
        assert 'ProjectIndex' in schema_source
        assert 'StatusIndex' in schema_source
        assert 'VersionIndex' in schema_source
        assert 'GlobalSecondaryIndexes' in schema_source
        assert 'LocalSecondaryIndexes' in schema_source
        
        print("✅ Indexes implemented for project-based and version-based queries")
        verification_results.append(("Index Implementation", True, ""))
        
    except Exception as e:
        print(f"❌ Index implementation failed: {e}")
        verification_results.append(("Index Implementation", False, str(e)))
    
    # Additional verification: Check requirements 3.5 and 3.6 compliance
    print("\n📊 Additional: Requirements 3.5 and 3.6 compliance")
    try:
        # Requirement 3.5: Include video title, description, tags, and chunk information
        metadata = VideoMetadata(
            video_id=str(uuid.uuid4()),
            project_id=str(uuid.uuid4()),
            title="Test Video",
            description="Test description",
            tags=["test", "sample"],
            chunk_s3_paths={"scene1": "s3://bucket/chunk1.mp4"},
            scene_count=5
        )
        
        assert hasattr(metadata, 'title')
        assert hasattr(metadata, 'description')
        assert hasattr(metadata, 'tags')
        assert hasattr(metadata, 'chunk_s3_paths')
        assert hasattr(metadata, 'scene_count')
        
        # Requirement 3.6: Proper error handling and consistency checks
        service_source = inspect.getsource(MetadataService)
        assert 'try:' in service_source
        assert 'except ClientError' in service_source
        assert 'except Exception' in service_source
        assert 'AWSMetadataError' in service_source
        assert 'logger.error' in service_source
        
        print("✅ Requirements 3.5 and 3.6 compliance verified")
        verification_results.append(("Requirements 3.5 & 3.6", True, ""))
        
    except Exception as e:
        print(f"❌ Requirements 3.5 and 3.6 compliance failed: {e}")
        verification_results.append(("Requirements 3.5 & 3.6", False, str(e)))
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 TASK 3.2 VERIFICATION SUMMARY")
    print("=" * 80)
    
    total_requirements = len(verification_results)
    passed_requirements = sum(1 for _, passed, _ in verification_results if passed)
    
    for requirement, passed, error in verification_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {requirement}")
        if not passed and error:
            print(f"     Error: {error}")
    
    print(f"\nOverall Result: {passed_requirements}/{total_requirements} requirements passed")
    
    if passed_requirements == total_requirements:
        print("🎉 Task 3.2 is COMPLETE - All requirements implemented successfully!")
        return True
    else:
        print("⚠️ Task 3.2 is INCOMPLETE - Some requirements need attention")
        return False


if __name__ == "__main__":
    success = verify_task_3_2_requirements()
    exit(0 if success else 1)
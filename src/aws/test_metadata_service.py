"""
Tests for MetadataService with batch operations and conflict resolution.
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock
from moto import mock_aws
import boto3

from .metadata_service import MetadataService, BatchOperationResult, ConflictResolutionStrategy
from .dynamodb_metadata import VideoMetadata, DynamoDBTableManager
from .config import AWSConfig
from .exceptions import AWSMetadataError


@pytest.fixture
def aws_config():
    """Create test AWS configuration."""
    return AWSConfig(
        region='us-east-1',
        video_bucket_name='test-video-bucket',
        code_bucket_name='test-code-bucket',
        metadata_table_name='test-video-metadata',
        enable_encryption=False
    )


@pytest.fixture
def sample_video_metadata():
    """Create sample video metadata for testing."""
    return VideoMetadata(
        video_id=str(uuid.uuid4()),
        project_id=str(uuid.uuid4()),
        title="Test Video",
        description="Test video description",
        status="created",
        tags=["test", "sample"],
        custom_metadata={"test_key": "test_value"}
    )


@pytest.fixture
def multiple_video_metadata():
    """Create multiple video metadata objects for batch testing."""
    project_id = str(uuid.uuid4())
    return [
        VideoMetadata(
            video_id=str(uuid.uuid4()),
            project_id=project_id,
            title=f"Test Video {i}",
            description=f"Test video {i} description",
            status="created" if i % 2 == 0 else "uploaded",
            tags=[f"test{i}", "batch"],
            custom_metadata={"batch_id": i}
        )
        for i in range(5)
    ]


@mock_aws
class TestMetadataService:
    """Test suite for MetadataService."""
    
    @pytest.fixture(autouse=True)
    async def setup_service(self, aws_config):
        """Set up MetadataService with mocked DynamoDB."""
        # Create DynamoDB table
        table_manager = DynamoDBTableManager(aws_config)
        table_manager.create_table_if_not_exists()
        
        # Initialize service
        self.service = MetadataService(aws_config, max_workers=3)
        
        yield
        
        # Cleanup
        if hasattr(self.service, 'executor'):
            self.service.executor.shutdown(wait=True)
    
    @pytest.mark.asyncio
    async def test_batch_create_video_records_success(self, multiple_video_metadata):
        """Test successful batch creation of video records."""
        result = await self.service.batch_create_video_records(multiple_video_metadata)
        
        assert isinstance(result, BatchOperationResult)
        assert result.success_count == 5
        assert result.failure_count == 0
        assert result.total_processed == 5
        assert result.success_rate == 100.0
        assert len(result.successful_items) == 5
        assert len(result.failed_items) == 0
        
        # Verify records were created
        for video_metadata in multiple_video_metadata:
            record = self.service.operations.get_video_metadata(video_metadata.video_id)
            assert record is not None
            assert record['title'] == video_metadata.title
            assert record['project_id'] == video_metadata.project_id
    
    @pytest.mark.asyncio
    async def test_batch_create_empty_list(self):
        """Test batch creation with empty list."""
        result = await self.service.batch_create_video_records([])
        
        assert result.success_count == 0
        assert result.failure_count == 0
        assert result.total_processed == 0
        assert result.success_rate == 0.0
    
    async def test_batch_update_metadata_success(self, multiple_video_metadata):
        """Test successful batch update of metadata."""
        # First create the records
        await self.service.batch_create_video_records(multiple_video_metadata)
        
        # Prepare updates
        updates = {}
        for video_metadata in multiple_video_metadata:
            updates[video_metadata.video_id] = {
                'status': 'updated',
                'description': f'Updated description for {video_metadata.title}',
                'custom_metadata': {'updated': True}
            }
        
        # Perform batch update
        result = await self.service.batch_update_metadata(updates)
        
        assert result.success_count == 5
        assert result.failure_count == 0
        assert result.success_rate == 100.0
        
        # Verify updates were applied
        for video_metadata in multiple_video_metadata:
            record = self.service.operations.get_video_metadata(video_metadata.video_id)
            assert record['status'] == 'updated'
            assert 'Updated description' in record['description']
    
    async def test_batch_update_empty_dict(self):
        """Test batch update with empty dictionary."""
        result = await self.service.batch_update_metadata({})
        
        assert result.success_count == 0
        assert result.failure_count == 0
        assert result.total_processed == 0
    
    async def test_conflict_resolution_latest_wins(self, sample_video_metadata):
        """Test conflict resolution with latest_wins strategy."""
        # Create initial record
        self.service.operations.create_video_record(sample_video_metadata)
        
        # Set conflict resolution strategy
        strategy = ConflictResolutionStrategy(strategy="latest_wins")
        self.service.set_conflict_resolution_strategy(strategy)
        
        # Simulate concurrent updates
        update1 = {'status': 'rendering', 'description': 'First update'}
        update2 = {'status': 'uploaded', 'description': 'Second update'}
        
        # Both updates should succeed with latest_wins
        success1 = await self.service._update_with_conflict_resolution(
            sample_video_metadata.video_id, update1
        )
        success2 = await self.service._update_with_conflict_resolution(
            sample_video_metadata.video_id, update2
        )
        
        assert success1 is True
        assert success2 is True
        
        # Verify final state
        record = self.service.operations.get_video_metadata(sample_video_metadata.video_id)
        assert record['status'] == 'uploaded'  # Latest update wins
        assert record['description'] == 'Second update'
    
    async def test_conflict_resolution_merge_fields(self, sample_video_metadata):
        """Test conflict resolution with merge_fields strategy."""
        # Create initial record with custom metadata
        sample_video_metadata.custom_metadata = {'field1': 'value1', 'field2': 'value2'}
        self.service.operations.create_video_record(sample_video_metadata)
        
        # Set merge strategy
        strategy = ConflictResolutionStrategy(
            strategy="merge_fields",
            merge_fields=['custom_metadata', 'tags']
        )
        self.service.set_conflict_resolution_strategy(strategy)
        
        # Update with overlapping custom metadata
        update_data = {
            'custom_metadata': {'field2': 'updated_value2', 'field3': 'value3'},
            'tags': ['new_tag'],
            'status': 'updated'
        }
        
        success = await self.service._update_with_conflict_resolution(
            sample_video_metadata.video_id, update_data
        )
        
        assert success is True
        
        # Verify merged result
        record = self.service.operations.get_video_metadata(sample_video_metadata.video_id)
        assert record['status'] == 'updated'
        
        # Custom metadata should be merged
        expected_metadata = {
            'field1': 'value1',  # Preserved from original
            'field2': 'updated_value2',  # Updated
            'field3': 'value3'  # Added
        }
        assert record['custom_metadata'] == expected_metadata
        
        # Tags should be merged
        expected_tags = ['test', 'sample', 'new_tag']  # Original + new
        assert set(record['tags']) == set(expected_tags)
    
    async def test_batch_query_by_project(self, multiple_video_metadata):
        """Test batch querying multiple projects."""
        # Create records
        await self.service.batch_create_video_records(multiple_video_metadata)
        
        # Get unique project IDs
        project_ids = list(set(vm.project_id for vm in multiple_video_metadata))
        
        # Perform batch query
        results = await self.service.batch_query_by_project(project_ids, limit_per_project=10)
        
        assert isinstance(results, dict)
        assert len(results) == len(project_ids)
        
        # Verify each project has its videos
        for project_id in project_ids:
            assert project_id in results
            project_videos = results[project_id]
            assert len(project_videos) > 0
            
            # All videos should belong to this project
            for video in project_videos:
                assert video['project_id'] == project_id
    
    async def test_advanced_project_query_with_filters(self, multiple_video_metadata):
        """Test advanced project query with date and status filters."""
        # Create records
        await self.service.batch_create_video_records(multiple_video_metadata)
        
        project_id = multiple_video_metadata[0].project_id
        
        # Test status filtering
        results = await self.service.create_project_index_query(
            project_id=project_id,
            status_filter='created'
        )
        
        assert len(results) > 0
        for video in results:
            assert video['status'] == 'created'
            assert video['project_id'] == project_id
    
    async def test_version_index_query(self, sample_video_metadata):
        """Test version history query."""
        # Create record
        self.service.operations.create_video_record(sample_video_metadata)
        
        # Query version history
        versions = await self.service.create_version_index_query(sample_video_metadata.video_id)
        
        assert len(versions) == 1
        assert versions[0]['video_id'] == sample_video_metadata.video_id
    
    async def test_metadata_statistics(self, multiple_video_metadata):
        """Test metadata statistics generation."""
        # Create records with different statuses
        await self.service.batch_create_video_records(multiple_video_metadata)
        
        # Update some records to have file sizes
        updates = {}
        for i, vm in enumerate(multiple_video_metadata[:3]):
            updates[vm.video_id] = {
                'file_size_bytes': (i + 1) * 1000000,  # 1MB, 2MB, 3MB
                'code_s3_paths': {'main': f's3://bucket/code_{i}.py'}
            }
        
        await self.service.batch_update_metadata(updates)
        
        # Get statistics for the project
        project_id = multiple_video_metadata[0].project_id
        stats = await self.service.get_metadata_statistics(project_id)
        
        assert 'total_videos' in stats
        assert 'status_breakdown' in stats
        assert 'storage_usage' in stats
        assert 'recent_activity' in stats
        
        assert stats['total_videos'] > 0
        assert 'created' in stats['status_breakdown']
        assert stats['storage_usage']['total_video_size'] > 0
        assert stats['storage_usage']['total_code_files'] > 0
    
    async def test_atomic_update_with_version_check(self, sample_video_metadata):
        """Test atomic update with optimistic locking."""
        # Create initial record
        self.service.operations.create_video_record(sample_video_metadata)
        
        # Get current record
        current_record = self.service.operations.get_video_metadata(sample_video_metadata.video_id)
        
        # Perform atomic update
        update_data = {'status': 'updated', 'description': 'Atomically updated'}
        success = await self.service._atomic_update(
            sample_video_metadata.video_id, update_data, current_record
        )
        
        assert success is True
        
        # Verify update
        updated_record = self.service.operations.get_video_metadata(sample_video_metadata.video_id)
        assert updated_record['status'] == 'updated'
        assert updated_record['description'] == 'Atomically updated'
        assert updated_record['last_edited_timestamp'] != current_record['last_edited_timestamp']
    
    async def test_concurrent_updates_with_conflict_resolution(self, sample_video_metadata):
        """Test concurrent updates with proper conflict resolution."""
        # Create initial record
        self.service.operations.create_video_record(sample_video_metadata)
        
        # Simulate concurrent updates
        async def update_task(update_data, task_id):
            return await self.service._update_with_conflict_resolution(
                sample_video_metadata.video_id, 
                {**update_data, 'task_id': task_id}
            )
        
        # Run concurrent updates
        tasks = [
            asyncio.create_task(update_task({'status': 'rendering'}, 1)),
            asyncio.create_task(update_task({'status': 'uploaded'}, 2)),
            asyncio.create_task(update_task({'status': 'transcoded'}, 3))
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # At least one should succeed
        successful_updates = [r for r in results if r is True]
        assert len(successful_updates) > 0
        
        # Verify final state is consistent
        final_record = self.service.operations.get_video_metadata(sample_video_metadata.video_id)
        assert final_record['status'] in ['rendering', 'uploaded', 'transcoded']
    
    async def test_batch_operation_error_handling(self):
        """Test error handling in batch operations."""
        # Test with invalid video metadata (missing required fields)
        invalid_metadata = [
            VideoMetadata(
                video_id="",  # Invalid empty video_id
                project_id="test-project",
                title="Invalid Video"
            )
        ]
        
        # This should handle the error gracefully
        result = await self.service.batch_create_video_records(invalid_metadata)
        
        # Should have some failures
        assert result.failure_count > 0 or result.success_count == 0
    
    def test_conflict_resolution_strategy_configuration(self):
        """Test conflict resolution strategy configuration."""
        # Test default strategy
        assert self.service.conflict_resolution.strategy == "latest_wins"
        
        # Test setting new strategy
        new_strategy = ConflictResolutionStrategy(
            strategy="merge_fields",
            merge_fields=['tags', 'custom_metadata'],
            preserve_fields=['video_id', 'project_id', 'created_timestamp']
        )
        
        self.service.set_conflict_resolution_strategy(new_strategy)
        
        assert self.service.conflict_resolution.strategy == "merge_fields"
        assert 'tags' in self.service.conflict_resolution.merge_fields
        assert 'video_id' in self.service.conflict_resolution.preserve_fields
    
    async def test_service_cleanup(self):
        """Test proper cleanup of service resources."""
        # Service should clean up executor on deletion
        executor = self.service.executor
        assert executor is not None
        
        # Manually trigger cleanup
        del self.service
        
        # Executor should be shutdown (we can't easily test this without implementation details)
        # This test mainly ensures no exceptions are raised during cleanup


@pytest.mark.asyncio
class TestBatchOperationResult:
    """Test BatchOperationResult utility class."""
    
    def test_batch_operation_result_creation(self):
        """Test BatchOperationResult creation and properties."""
        result = BatchOperationResult(
            successful_items=['item1', 'item2', 'item3'],
            failed_items=[('item4', 'Error message')],
            total_processed=4,
            success_count=3,
            failure_count=1
        )
        
        assert result.success_rate == 75.0
        assert result.total_processed == 4
        assert len(result.successful_items) == 3
        assert len(result.failed_items) == 1
    
    def test_batch_operation_result_zero_division(self):
        """Test success rate calculation with zero total."""
        result = BatchOperationResult(
            successful_items=[],
            failed_items=[],
            total_processed=0,
            success_count=0,
            failure_count=0
        )
        
        assert result.success_rate == 0.0


@pytest.mark.asyncio
class TestConflictResolutionStrategy:
    """Test ConflictResolutionStrategy configuration."""
    
    def test_default_strategy(self):
        """Test default conflict resolution strategy."""
        strategy = ConflictResolutionStrategy()
        
        assert strategy.strategy == "latest_wins"
        assert strategy.merge_fields == []
        assert 'created_timestamp' in strategy.preserve_fields
        assert 'video_id' in strategy.preserve_fields
        assert 'project_id' in strategy.preserve_fields
    
    def test_custom_strategy(self):
        """Test custom conflict resolution strategy."""
        strategy = ConflictResolutionStrategy(
            strategy="merge_fields",
            merge_fields=['tags', 'custom_metadata'],
            preserve_fields=['video_id', 'created_timestamp']
        )
        
        assert strategy.strategy == "merge_fields"
        assert 'tags' in strategy.merge_fields
        assert 'custom_metadata' in strategy.merge_fields
        assert len(strategy.preserve_fields) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
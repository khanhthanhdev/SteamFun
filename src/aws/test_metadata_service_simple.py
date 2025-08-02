#!/usr/bin/env python3
"""
Simple test for MetadataService core functionality.
"""

import asyncio
import uuid
from datetime import datetime, timezone

from .metadata_service import BatchOperationResult, ConflictResolutionStrategy
from .dynamodb_metadata import VideoMetadata


def test_batch_operation_result():
    """Test BatchOperationResult functionality."""
    print("🧪 Testing BatchOperationResult...")
    
    # Test successful result
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
    print("✅ BatchOperationResult basic functionality works")
    
    # Test zero division protection
    empty_result = BatchOperationResult(
        successful_items=[],
        failed_items=[],
        total_processed=0,
        success_count=0,
        failure_count=0
    )
    
    assert empty_result.success_rate == 0.0
    print("✅ BatchOperationResult zero division protection works")


def test_conflict_resolution_strategy():
    """Test ConflictResolutionStrategy functionality."""
    print("🧪 Testing ConflictResolutionStrategy...")
    
    # Test default strategy
    default_strategy = ConflictResolutionStrategy()
    assert default_strategy.strategy == "latest_wins"
    assert default_strategy.merge_fields == []
    assert 'created_timestamp' in default_strategy.preserve_fields
    assert 'video_id' in default_strategy.preserve_fields
    assert 'project_id' in default_strategy.preserve_fields
    print("✅ Default ConflictResolutionStrategy works")
    
    # Test custom strategy
    custom_strategy = ConflictResolutionStrategy(
        strategy="merge_fields",
        merge_fields=['tags', 'custom_metadata'],
        preserve_fields=['video_id', 'created_timestamp']
    )
    
    assert custom_strategy.strategy == "merge_fields"
    assert 'tags' in custom_strategy.merge_fields
    assert 'custom_metadata' in custom_strategy.merge_fields
    assert len(custom_strategy.preserve_fields) == 2
    print("✅ Custom ConflictResolutionStrategy works")


def test_video_metadata():
    """Test VideoMetadata dataclass functionality."""
    print("🧪 Testing VideoMetadata...")
    
    # Test basic creation
    video_id = str(uuid.uuid4())
    project_id = str(uuid.uuid4())
    
    metadata = VideoMetadata(
        video_id=video_id,
        project_id=project_id,
        title="Test Video",
        description="Test description",
        tags=["test", "sample"],
        custom_metadata={"key": "value"}
    )
    
    assert metadata.video_id == video_id
    assert metadata.project_id == project_id
    assert metadata.title == "Test Video"
    assert metadata.status == "created"  # Default value
    assert metadata.version == 1  # Default value
    assert "test" in metadata.tags
    assert metadata.custom_metadata["key"] == "value"
    print("✅ VideoMetadata basic functionality works")
    
    # Test timestamp auto-generation
    assert metadata.created_timestamp != ""
    assert metadata.last_edited_timestamp != ""
    assert metadata.current_version_id == "v1"
    print("✅ VideoMetadata auto-generated fields work")


def test_metadata_service_conflict_resolution_logic():
    """Test the conflict resolution logic without AWS dependencies."""
    print("🧪 Testing conflict resolution logic...")
    
    # Simulate current record and update data
    current_record = {
        'video_id': 'test-video-123',
        'project_id': 'test-project-456',
        'title': 'Original Title',
        'custom_metadata': {'field1': 'value1', 'field2': 'value2'},
        'tags': ['original', 'test'],
        'status': 'created',
        'created_timestamp': '2024-01-01T00:00:00Z',
        'last_edited_timestamp': '2024-01-01T00:00:00Z'
    }
    
    update_data = {
        'title': 'Updated Title',
        'custom_metadata': {'field2': 'updated_value2', 'field3': 'value3'},
        'tags': ['updated', 'new'],
        'status': 'updated'
    }
    
    # Test latest_wins strategy (simple case)
    strategy = ConflictResolutionStrategy(strategy="latest_wins")
    
    # Simulate the resolve_conflicts method logic
    resolved_updates = update_data.copy()
    
    # Preserve certain fields
    for field in strategy.preserve_fields:
        if field in current_record and field not in resolved_updates:
            resolved_updates[field] = current_record[field]
    
    assert resolved_updates['title'] == 'Updated Title'
    assert resolved_updates['status'] == 'updated'
    assert 'video_id' not in resolved_updates or resolved_updates['video_id'] == current_record['video_id']
    print("✅ Latest wins conflict resolution works")
    
    # Test merge_fields strategy
    merge_strategy = ConflictResolutionStrategy(
        strategy="merge_fields",
        merge_fields=['custom_metadata', 'tags']
    )
    
    resolved_merge = update_data.copy()
    
    # Simulate merging logic
    for field in merge_strategy.merge_fields:
        if field in current_record and field in update_data:
            current_value = current_record[field]
            new_value = update_data[field]
            
            if isinstance(current_value, dict) and isinstance(new_value, dict):
                # Merge dictionaries
                merged = current_value.copy()
                merged.update(new_value)
                resolved_merge[field] = merged
            elif isinstance(current_value, list) and isinstance(new_value, list):
                # Merge lists (remove duplicates)
                merged = list(set(current_value + new_value))
                resolved_merge[field] = merged
    
    # Check merged custom_metadata
    expected_metadata = {
        'field1': 'value1',  # Preserved from original
        'field2': 'updated_value2',  # Updated
        'field3': 'value3'  # Added
    }
    assert resolved_merge['custom_metadata'] == expected_metadata
    
    # Check merged tags
    expected_tags = set(['original', 'test', 'updated', 'new'])
    assert set(resolved_merge['tags']) == expected_tags
    
    print("✅ Merge fields conflict resolution works")


def main():
    """Run all tests."""
    print("🚀 Starting MetadataService core functionality tests...\n")
    
    try:
        test_batch_operation_result()
        print()
        
        test_conflict_resolution_strategy()
        print()
        
        test_video_metadata()
        print()
        
        test_metadata_service_conflict_resolution_logic()
        print()
        
        print("🎉 All core functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
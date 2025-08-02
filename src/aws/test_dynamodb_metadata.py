"""
Test suite for DynamoDB metadata management functionality.

Tests table creation, CRUD operations, and query functionality.
"""

import pytest
import boto3
import uuid
import time
from datetime import datetime, timezone
from moto import mock_aws
from unittest.mock import patch, MagicMock

from .config import AWSConfig
from .dynamodb_metadata import (
    DynamoDBTableManager,
    DynamoDBMetadataOperations,
    VideoMetadata,
    generate_video_id,
    generate_project_id
)
from .exceptions import AWSMetadataError


@pytest.fixture
def aws_config():
    """Create test AWS configuration."""
    return AWSConfig(
        region='us-east-1',
        metadata_table_name='test-video-metadata',
        video_bucket_name='test-video-bucket',
        code_bucket_name='test-code-bucket',
        enable_aws_upload=True
    )


@pytest.fixture
def mock_dynamodb_setup():
    """Set up mock DynamoDB for testing."""
    with mock_aws():
        yield


@pytest.fixture
def table_manager(aws_config, mock_dynamodb_setup):
    """Create DynamoDB table manager for testing."""
    return DynamoDBTableManager(aws_config)


@pytest.fixture
def metadata_ops(aws_config, mock_dynamodb_setup, table_manager):
    """Create metadata operations instance with table created."""
    # Create table first
    table_manager.create_table_if_not_exists()
    return DynamoDBMetadataOperations(aws_config)


@pytest.fixture
def sample_video_metadata():
    """Create sample video metadata for testing."""
    video_id = generate_video_id()
    project_id = generate_project_id()
    
    return VideoMetadata(
        video_id=video_id,
        project_id=project_id,
        title="Test Video",
        description="A test video for unit testing",
        status="created",
        version=1,
        tags=["test", "unit-test"],
        custom_metadata={"test_key": "test_value"}
    )


class TestDynamoDBTableManager:
    """Test DynamoDB table management functionality."""
    
    def test_create_table_if_not_exists_success(self, table_manager):
        """Test successful table creation."""
        result = table_manager.create_table_if_not_exists()
        assert result is True
        
        # Verify table exists
        existing_tables = table_manager.client.list_tables()['TableNames']
        assert table_manager.table_name in existing_tables
    
    def test_create_table_if_not_exists_already_exists(self, table_manager):
        """Test table creation when table already exists."""
        # Create table first time
        table_manager.create_table_if_not_exists()
        
        # Try to create again
        result = table_manager.create_table_if_not_exists()
        assert result is True
    
    def test_describe_table(self, table_manager):
        """Test table description functionality."""
        table_manager.create_table_if_not_exists()
        
        description = table_manager.describe_table()
        assert description['TableName'] == table_manager.table_name
        assert description['TableStatus'] == 'ACTIVE'
        assert len(description['KeySchema']) == 1
        assert description['KeySchema'][0]['AttributeName'] == 'video_id'
    
    def test_delete_table(self, table_manager):
        """Test table deletion."""
        # Create table first
        table_manager.create_table_if_not_exists()
        
        # Delete table
        result = table_manager.delete_table()
        assert result is True
        
        # Verify table is gone
        existing_tables = table_manager.client.list_tables()['TableNames']
        assert table_manager.table_name not in existing_tables


class TestVideoMetadata:
    """Test VideoMetadata dataclass functionality."""
    
    def test_video_metadata_creation(self):
        """Test VideoMetadata creation with defaults."""
        video_id = generate_video_id()
        project_id = generate_project_id()
        
        metadata = VideoMetadata(
            video_id=video_id,
            project_id=project_id,
            title="Test Video"
        )
        
        assert metadata.video_id == video_id
        assert metadata.project_id == project_id
        assert metadata.title == "Test Video"
        assert metadata.status == "created"
        assert metadata.version == 1
        assert metadata.current_version_id == "v1"
        assert metadata.tags == []
        assert metadata.custom_metadata == {}
        assert metadata.created_timestamp != ""
        assert metadata.last_edited_timestamp != ""
    
    def test_video_metadata_with_custom_data(self):
        """Test VideoMetadata with custom data."""
        metadata = VideoMetadata(
            video_id="test-id",
            project_id="test-project",
            title="Custom Video",
            description="Custom description",
            status="uploaded",
            version=2,
            tags=["custom", "test"],
            custom_metadata={"key": "value"}
        )
        
        assert metadata.status == "uploaded"
        assert metadata.version == 2
        assert metadata.current_version_id == "v2"
        assert metadata.tags == ["custom", "test"]
        assert metadata.custom_metadata == {"key": "value"}


class TestDynamoDBMetadataOperations:
    """Test DynamoDB metadata CRUD operations."""
    
    def test_create_video_record_success(self, metadata_ops, sample_video_metadata):
        """Test successful video record creation."""
        result = metadata_ops.create_video_record(sample_video_metadata)
        assert result is True
        
        # Verify record was created
        retrieved = metadata_ops.get_video_metadata(sample_video_metadata.video_id)
        assert retrieved is not None
        assert retrieved['video_id'] == sample_video_metadata.video_id
        assert retrieved['title'] == sample_video_metadata.title
    
    def test_create_video_record_duplicate(self, metadata_ops, sample_video_metadata):
        """Test creating duplicate video record fails."""
        # Create first record
        metadata_ops.create_video_record(sample_video_metadata)
        
        # Try to create duplicate
        with pytest.raises(AWSMetadataError, match="already exists"):
            metadata_ops.create_video_record(sample_video_metadata)
    
    def test_get_video_metadata_success(self, metadata_ops, sample_video_metadata):
        """Test successful metadata retrieval."""
        metadata_ops.create_video_record(sample_video_metadata)
        
        retrieved = metadata_ops.get_video_metadata(sample_video_metadata.video_id)
        assert retrieved is not None
        assert retrieved['video_id'] == sample_video_metadata.video_id
        assert retrieved['project_id'] == sample_video_metadata.project_id
        assert retrieved['title'] == sample_video_metadata.title
        assert retrieved['status'] == sample_video_metadata.status
        assert retrieved['tags'] == sample_video_metadata.tags
        assert retrieved['custom_metadata'] == sample_video_metadata.custom_metadata
    
    def test_get_video_metadata_not_found(self, metadata_ops):
        """Test metadata retrieval for non-existent video."""
        result = metadata_ops.get_video_metadata("non-existent-id")
        assert result is None
    
    def test_update_metadata_success(self, metadata_ops, sample_video_metadata):
        """Test successful metadata update."""
        metadata_ops.create_video_record(sample_video_metadata)
        
        updates = {
            'status': 'uploaded',
            'duration_seconds': 120.5,
            's3_path_full_video': 's3://test-bucket/video.mp4',
            'chunk_s3_paths': {'scene1': 's3://test-bucket/scene1.mp4'},
            'custom_metadata': {'updated': True}
        }
        
        result = metadata_ops.update_metadata(sample_video_metadata.video_id, updates)
        assert result is True
        
        # Verify updates
        retrieved = metadata_ops.get_video_metadata(sample_video_metadata.video_id)
        assert retrieved['status'] == 'uploaded'
        assert retrieved['duration_seconds'] == 120.5
        assert retrieved['s3_path_full_video'] == 's3://test-bucket/video.mp4'
        assert retrieved['chunk_s3_paths'] == {'scene1': 's3://test-bucket/scene1.mp4'}
        assert retrieved['custom_metadata'] == {'updated': True}
        assert retrieved['last_edited_timestamp'] != sample_video_metadata.last_edited_timestamp
    
    def test_update_metadata_nonexistent_record(self, metadata_ops):
        """Test updating non-existent record fails."""
        updates = {'status': 'uploaded'}
        
        with pytest.raises(AWSMetadataError, match="does not exist"):
            metadata_ops.update_metadata("non-existent-id", updates)
    
    def test_update_metadata_empty_updates(self, metadata_ops, sample_video_metadata):
        """Test update with empty updates."""
        metadata_ops.create_video_record(sample_video_metadata)
        
        result = metadata_ops.update_metadata(sample_video_metadata.video_id, {})
        assert result is True
    
    def test_query_by_project(self, metadata_ops):
        """Test querying videos by project ID."""
        project_id = generate_project_id()
        
        # Create multiple videos for the same project
        video1 = VideoMetadata(
            video_id=generate_video_id(),
            project_id=project_id,
            title="Video 1"
        )
        video2 = VideoMetadata(
            video_id=generate_video_id(),
            project_id=project_id,
            title="Video 2"
        )
        video3 = VideoMetadata(
            video_id=generate_video_id(),
            project_id=generate_project_id(),  # Different project
            title="Video 3"
        )
        
        metadata_ops.create_video_record(video1)
        time.sleep(0.1)  # Ensure different timestamps
        metadata_ops.create_video_record(video2)
        metadata_ops.create_video_record(video3)
        
        # Query by project
        results = metadata_ops.query_by_project(project_id)
        assert len(results) == 2
        
        # Results should be sorted by created_timestamp descending
        assert results[0]['title'] == "Video 2"  # More recent
        assert results[1]['title'] == "Video 1"
    
    def test_query_by_project_with_limit(self, metadata_ops):
        """Test querying videos by project with limit."""
        project_id = generate_project_id()
        
        # Create multiple videos
        for i in range(5):
            video = VideoMetadata(
                video_id=generate_video_id(),
                project_id=project_id,
                title=f"Video {i}"
            )
            metadata_ops.create_video_record(video)
            time.sleep(0.01)
        
        # Query with limit
        results = metadata_ops.query_by_project(project_id, limit=3)
        assert len(results) == 3
    
    def test_query_by_status(self, metadata_ops):
        """Test querying videos by status."""
        # Create videos with different statuses
        video1 = VideoMetadata(
            video_id=generate_video_id(),
            project_id=generate_project_id(),
            title="Video 1",
            status="created"
        )
        video2 = VideoMetadata(
            video_id=generate_video_id(),
            project_id=generate_project_id(),
            title="Video 2",
            status="uploaded"
        )
        video3 = VideoMetadata(
            video_id=generate_video_id(),
            project_id=generate_project_id(),
            title="Video 3",
            status="uploaded"
        )
        
        metadata_ops.create_video_record(video1)
        metadata_ops.create_video_record(video2)
        metadata_ops.create_video_record(video3)
        
        # Query by status
        uploaded_videos = metadata_ops.query_by_status("uploaded")
        assert len(uploaded_videos) == 2
        
        created_videos = metadata_ops.query_by_status("created")
        assert len(created_videos) == 1
        assert created_videos[0]['title'] == "Video 1"
    
    def test_delete_video_record_success(self, metadata_ops, sample_video_metadata):
        """Test successful video record deletion."""
        metadata_ops.create_video_record(sample_video_metadata)
        
        result = metadata_ops.delete_video_record(sample_video_metadata.video_id)
        assert result is True
        
        # Verify record is gone
        retrieved = metadata_ops.get_video_metadata(sample_video_metadata.video_id)
        assert retrieved is None
    
    def test_delete_video_record_not_found(self, metadata_ops):
        """Test deleting non-existent record."""
        with pytest.raises(AWSMetadataError, match="does not exist"):
            metadata_ops.delete_video_record("non-existent-id")


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_generate_video_id(self):
        """Test video ID generation."""
        video_id1 = generate_video_id()
        video_id2 = generate_video_id()
        
        assert video_id1 != video_id2
        assert len(video_id1) == 36  # UUID4 length
        assert '-' in video_id1
    
    def test_generate_project_id(self):
        """Test project ID generation."""
        project_id1 = generate_project_id()
        project_id2 = generate_project_id()
        
        assert project_id1 != project_id2
        assert len(project_id1) == 36  # UUID4 length
        assert '-' in project_id1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
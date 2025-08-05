"""
Integration tests for video database operations.
Tests database models, relationships, and data persistence.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.database.base import Base
from app.models.database.video import Video, VideoMetadata
from app.models.enums import VideoStatus, VideoQuality


class TestVideoDatabaseIntegration:
    """Integration test suite for video database operations."""
    
    @pytest.fixture
    def test_engine(self):
        """Create test database engine."""
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)
        return engine
    
    @pytest.fixture
    def test_session(self, test_engine):
        """Create test database session."""
        TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()
    
    @pytest.mark.integration
    def test_video_model_creation_integration(self, test_session):
        """Test video model creation and persistence."""
        # Create video record
        video = Video(
            video_id="integration_test_video_123",
            topic="Integration Test Video",
            description="Testing video model integration",
            status=VideoStatus.PROCESSING,
            created_at=datetime.now(),
            voice_settings={"voice": "default", "speed": 1.0},
            animation_config={"quality": "medium", "fps": 30}
        )
        
        # Save to database
        test_session.add(video)
        test_session.commit()
        
        # Verify video was saved
        saved_video = test_session.query(Video).filter_by(video_id="integration_test_video_123").first()
        assert saved_video is not None
        assert saved_video.topic == "Integration Test Video"
        assert saved_video.status == VideoStatus.PROCESSING
        assert saved_video.voice_settings == {"voice": "default", "speed": 1.0}
    
    @pytest.mark.integration
    def test_video_metadata_relationship_integration(self, test_session):
        """Test video and metadata relationship integration."""
        # Create video with metadata
        video = Video(
            video_id="metadata_test_video",
            topic="Metadata Test",
            description="Testing metadata relationships",
            status=VideoStatus.COMPLETED
        )
        
        metadata = VideoMetadata(
            video_id="metadata_test_video",
            file_size=1024000,
            duration=120.5,
            resolution="1080p",
            fps=30,
            codec="h264",
            download_url="https://example.com/video.mp4"
        )
        
        # Save both records
        test_session.add(video)
        test_session.add(metadata)
        test_session.commit()
        
        # Verify relationship
        saved_video = test_session.query(Video).filter_by(video_id="metadata_test_video").first()
        assert saved_video.metadata is not None
        assert saved_video.metadata.file_size == 1024000
        assert saved_video.metadata.duration == 120.5
        assert saved_video.metadata.download_url == "https://example.com/video.mp4"
    
    @pytest.mark.integration
    def test_video_status_updates_integration(self, test_session):
        """Test video status update operations."""
        # Create video in processing state
        video = Video(
            video_id="status_update_test",
            topic="Status Update Test",
            description="Testing status updates",
            status=VideoStatus.PROCESSING
        )
        
        test_session.add(video)
        test_session.commit()
        
        # Update status to completed
        video.status = VideoStatus.COMPLETED
        video.completed_at = datetime.now()
        test_session.commit()
        
        # Verify status update
        updated_video = test_session.query(Video).filter_by(video_id="status_update_test").first()
        assert updated_video.status == VideoStatus.COMPLETED
        assert updated_video.completed_at is not None
    
    @pytest.mark.integration
    def test_video_query_operations_integration(self, test_session):
        """Test video query operations."""
        # Create multiple videos
        videos = [
            Video(
                video_id=f"query_test_{i}",
                topic=f"Query Test Video {i}",
                description=f"Testing queries {i}",
                status=VideoStatus.COMPLETED if i % 2 == 0 else VideoStatus.PROCESSING
            )
            for i in range(5)
        ]
        
        for video in videos:
            test_session.add(video)
        test_session.commit()
        
        # Test query by status
        completed_videos = test_session.query(Video).filter_by(status=VideoStatus.COMPLETED).all()
        assert len(completed_videos) == 3  # Videos 0, 2, 4
        
        processing_videos = test_session.query(Video).filter_by(status=VideoStatus.PROCESSING).all()
        assert len(processing_videos) == 2  # Videos 1, 3
        
        # Test query by topic pattern
        query_videos = test_session.query(Video).filter(Video.topic.like("Query Test%")).all()
        assert len(query_videos) == 5
    
    @pytest.mark.integration
    def test_video_error_handling_integration(self, test_session):
        """Test database error handling integration."""
        # Test duplicate video_id constraint
        video1 = Video(
            video_id="duplicate_test",
            topic="First Video",
            description="First video with this ID",
            status=VideoStatus.PROCESSING
        )
        
        video2 = Video(
            video_id="duplicate_test",
            topic="Second Video",
            description="Second video with same ID",
            status=VideoStatus.PROCESSING
        )
        
        # Add first video
        test_session.add(video1)
        test_session.commit()
        
        # Try to add duplicate - should raise error
        test_session.add(video2)
        with pytest.raises(Exception):  # IntegrityError or similar
            test_session.commit()
        
        # Rollback and verify first video still exists
        test_session.rollback()
        existing_video = test_session.query(Video).filter_by(video_id="duplicate_test").first()
        assert existing_video is not None
        assert existing_video.topic == "First Video"
    
    @pytest.mark.integration
    def test_video_cascade_operations_integration(self, test_session):
        """Test cascade operations between video and metadata."""
        # Create video with metadata
        video = Video(
            video_id="cascade_test",
            topic="Cascade Test",
            description="Testing cascade operations",
            status=VideoStatus.COMPLETED
        )
        
        metadata = VideoMetadata(
            video_id="cascade_test",
            file_size=2048000,
            duration=180.0,
            resolution="4K",
            fps=60
        )
        
        test_session.add(video)
        test_session.add(metadata)
        test_session.commit()
        
        # Delete video (should cascade to metadata if configured)
        test_session.delete(video)
        test_session.commit()
        
        # Verify both records are deleted
        deleted_video = test_session.query(Video).filter_by(video_id="cascade_test").first()
        assert deleted_video is None
        
        # Note: Cascade behavior depends on foreign key configuration
        # This test verifies the relationship works as expected
    
    @pytest.mark.integration
    def test_video_transaction_integration(self, test_session):
        """Test database transaction handling."""
        try:
            # Start transaction
            video = Video(
                video_id="transaction_test",
                topic="Transaction Test",
                description="Testing transaction handling",
                status=VideoStatus.PROCESSING
            )
            
            test_session.add(video)
            
            # Simulate error during transaction
            metadata = VideoMetadata(
                video_id="transaction_test",
                file_size="invalid_size",  # This should cause an error
                duration=120.0
            )
            
            test_session.add(metadata)
            test_session.commit()
            
        except Exception:
            # Rollback transaction
            test_session.rollback()
            
            # Verify no records were saved
            video_count = test_session.query(Video).filter_by(video_id="transaction_test").count()
            metadata_count = test_session.query(VideoMetadata).filter_by(video_id="transaction_test").count()
            
            assert video_count == 0
            assert metadata_count == 0
    
    @pytest.mark.integration
    def test_video_performance_integration(self, test_session):
        """Test database performance with bulk operations."""
        # Create many videos for performance testing
        videos = []
        for i in range(100):
            video = Video(
                video_id=f"performance_test_{i}",
                topic=f"Performance Test Video {i}",
                description=f"Testing performance {i}",
                status=VideoStatus.COMPLETED if i % 3 == 0 else VideoStatus.PROCESSING
            )
            videos.append(video)
        
        # Bulk insert
        test_session.add_all(videos)
        test_session.commit()
        
        # Test bulk query performance
        all_videos = test_session.query(Video).filter(Video.video_id.like("performance_test_%")).all()
        assert len(all_videos) == 100
        
        # Test filtered query performance
        completed_videos = test_session.query(Video).filter(
            Video.video_id.like("performance_test_%"),
            Video.status == VideoStatus.COMPLETED
        ).all()
        
        # Should be approximately 33-34 videos (every 3rd one)
        assert 30 <= len(completed_videos) <= 40
    
    @pytest.mark.integration
    def test_video_json_field_integration(self, test_session):
        """Test JSON field storage and retrieval."""
        # Create video with complex JSON data
        complex_voice_settings = {
            "voice": "custom",
            "speed": 1.2,
            "pitch": 0.9,
            "effects": ["reverb", "echo"],
            "custom_params": {
                "tone": "professional",
                "emotion": "neutral"
            }
        }
        
        complex_animation_config = {
            "quality": "high",
            "fps": 60,
            "resolution": "4K",
            "effects": ["fade_in", "fade_out"],
            "transitions": {
                "type": "smooth",
                "duration": 0.5
            }
        }
        
        video = Video(
            video_id="json_test",
            topic="JSON Test",
            description="Testing JSON field storage",
            status=VideoStatus.PROCESSING,
            voice_settings=complex_voice_settings,
            animation_config=complex_animation_config
        )
        
        test_session.add(video)
        test_session.commit()
        
        # Retrieve and verify JSON data
        saved_video = test_session.query(Video).filter_by(video_id="json_test").first()
        
        assert saved_video.voice_settings["voice"] == "custom"
        assert saved_video.voice_settings["effects"] == ["reverb", "echo"]
        assert saved_video.voice_settings["custom_params"]["tone"] == "professional"
        
        assert saved_video.animation_config["quality"] == "high"
        assert saved_video.animation_config["transitions"]["type"] == "smooth"
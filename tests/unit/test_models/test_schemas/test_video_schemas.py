"""
Unit tests for video-related Pydantic schemas.
Tests validation, serialization, and data transformation.
"""

import pytest
from pydantic import ValidationError
from datetime import datetime

from app.models.schemas.video import (
    VideoRequest, VideoResponse, VoiceSettings, 
    AnimationConfig, VideoStatus
)
from app.models.enums import VideoQuality, VoiceType


class TestVideoRequest:
    """Test suite for VideoRequest schema."""
    
    def test_valid_video_request(self):
        """Test creation of valid video request."""
        request = VideoRequest(
            topic="Python basics",
            description="Introduction to Python programming",
            voice_settings=VoiceSettings(
                voice=VoiceType.DEFAULT,
                speed=1.0,
                pitch=1.0
            ),
            animation_config=AnimationConfig(
                quality=VideoQuality.MEDIUM,
                fps=30,
                resolution="1080p"
            )
        )
        
        assert request.topic == "Python basics"
        assert request.description == "Introduction to Python programming"
        assert request.voice_settings.voice == VoiceType.DEFAULT
        assert request.animation_config.quality == VideoQuality.MEDIUM
    
    def test_video_request_empty_topic(self):
        """Test validation error for empty topic."""
        with pytest.raises(ValidationError) as exc_info:
            VideoRequest(
                topic="",
                description="Test description",
                voice_settings=VoiceSettings(),
                animation_config=AnimationConfig()
            )
        
        assert "topic" in str(exc_info.value)
    
    def test_video_request_missing_required_fields(self):
        """Test validation error for missing required fields."""
        with pytest.raises(ValidationError) as exc_info:
            VideoRequest()
        
        errors = exc_info.value.errors()
        required_fields = [error["loc"][0] for error in errors]
        assert "topic" in required_fields
        assert "description" in required_fields
    
    def test_video_request_default_values(self):
        """Test default values for optional fields."""
        request = VideoRequest(
            topic="Test topic",
            description="Test description"
        )
        
        # Should have default voice settings
        assert request.voice_settings is not None
        assert request.voice_settings.voice == VoiceType.DEFAULT
        assert request.voice_settings.speed == 1.0
        
        # Should have default animation config
        assert request.animation_config is not None
        assert request.animation_config.quality == VideoQuality.MEDIUM
        assert request.animation_config.fps == 30


class TestVideoResponse:
    """Test suite for VideoResponse schema."""
    
    def test_valid_video_response(self):
        """Test creation of valid video response."""
        response = VideoResponse(
            video_id="test_video_123",
            status=VideoStatus.COMPLETED,
            download_url="https://example.com/video.mp4",
            created_at=datetime.now(),
            progress=1.0
        )
        
        assert response.video_id == "test_video_123"
        assert response.status == VideoStatus.COMPLETED
        assert response.download_url == "https://example.com/video.mp4"
        assert response.progress == 1.0
    
    def test_video_response_processing_status(self):
        """Test video response with processing status."""
        response = VideoResponse(
            video_id="processing_video",
            status=VideoStatus.PROCESSING,
            progress=0.5
        )
        
        assert response.status == VideoStatus.PROCESSING
        assert response.download_url is None
        assert response.progress == 0.5
    
    def test_video_response_error_status(self):
        """Test video response with error status."""
        response = VideoResponse(
            video_id="failed_video",
            status=VideoStatus.FAILED,
            error_message="Processing failed due to invalid input"
        )
        
        assert response.status == VideoStatus.FAILED
        assert response.error_message == "Processing failed due to invalid input"
        assert response.download_url is None


class TestVoiceSettings:
    """Test suite for VoiceSettings schema."""
    
    def test_valid_voice_settings(self):
        """Test creation of valid voice settings."""
        settings = VoiceSettings(
            voice=VoiceType.FEMALE,
            speed=1.2,
            pitch=0.9,
            volume=0.8
        )
        
        assert settings.voice == VoiceType.FEMALE
        assert settings.speed == 1.2
        assert settings.pitch == 0.9
        assert settings.volume == 0.8
    
    def test_voice_settings_speed_validation(self):
        """Test speed validation constraints."""
        # Valid speed
        settings = VoiceSettings(speed=1.5)
        assert settings.speed == 1.5
        
        # Invalid speed - too low
        with pytest.raises(ValidationError):
            VoiceSettings(speed=0.1)
        
        # Invalid speed - too high
        with pytest.raises(ValidationError):
            VoiceSettings(speed=3.0)
    
    def test_voice_settings_pitch_validation(self):
        """Test pitch validation constraints."""
        # Valid pitch
        settings = VoiceSettings(pitch=1.1)
        assert settings.pitch == 1.1
        
        # Invalid pitch - too low
        with pytest.raises(ValidationError):
            VoiceSettings(pitch=0.1)
        
        # Invalid pitch - too high
        with pytest.raises(ValidationError):
            VoiceSettings(pitch=2.5)
    
    def test_voice_settings_default_values(self):
        """Test default values for voice settings."""
        settings = VoiceSettings()
        
        assert settings.voice == VoiceType.DEFAULT
        assert settings.speed == 1.0
        assert settings.pitch == 1.0
        assert settings.volume == 1.0


class TestAnimationConfig:
    """Test suite for AnimationConfig schema."""
    
    def test_valid_animation_config(self):
        """Test creation of valid animation configuration."""
        config = AnimationConfig(
            quality=VideoQuality.HIGH,
            fps=60,
            resolution="4K",
            background_color="#FFFFFF",
            text_color="#000000"
        )
        
        assert config.quality == VideoQuality.HIGH
        assert config.fps == 60
        assert config.resolution == "4K"
        assert config.background_color == "#FFFFFF"
        assert config.text_color == "#000000"
    
    def test_animation_config_fps_validation(self):
        """Test FPS validation constraints."""
        # Valid FPS
        config = AnimationConfig(fps=30)
        assert config.fps == 30
        
        # Invalid FPS - too low
        with pytest.raises(ValidationError):
            AnimationConfig(fps=5)
        
        # Invalid FPS - too high
        with pytest.raises(ValidationError):
            AnimationConfig(fps=150)
    
    def test_animation_config_color_validation(self):
        """Test color validation for hex colors."""
        # Valid hex colors
        config = AnimationConfig(
            background_color="#FF0000",
            text_color="#00FF00"
        )
        assert config.background_color == "#FF0000"
        assert config.text_color == "#00FF00"
        
        # Invalid hex color
        with pytest.raises(ValidationError):
            AnimationConfig(background_color="invalid_color")
    
    def test_animation_config_default_values(self):
        """Test default values for animation configuration."""
        config = AnimationConfig()
        
        assert config.quality == VideoQuality.MEDIUM
        assert config.fps == 30
        assert config.resolution == "1080p"
        assert config.background_color == "#FFFFFF"
        assert config.text_color == "#000000"


class TestVideoSchemaIntegration:
    """Test suite for schema integration and serialization."""
    
    def test_video_request_to_dict(self):
        """Test video request serialization to dictionary."""
        request = VideoRequest(
            topic="Test topic",
            description="Test description",
            voice_settings=VoiceSettings(voice=VoiceType.MALE),
            animation_config=AnimationConfig(quality=VideoQuality.HIGH)
        )
        
        data = request.model_dump()
        
        assert data["topic"] == "Test topic"
        assert data["description"] == "Test description"
        assert data["voice_settings"]["voice"] == "male"
        assert data["animation_config"]["quality"] == "high"
    
    def test_video_response_from_dict(self):
        """Test video response creation from dictionary."""
        data = {
            "video_id": "test_123",
            "status": "completed",
            "download_url": "https://example.com/video.mp4",
            "progress": 1.0
        }
        
        response = VideoResponse(**data)
        
        assert response.video_id == "test_123"
        assert response.status == VideoStatus.COMPLETED
        assert response.download_url == "https://example.com/video.mp4"
        assert response.progress == 1.0
    
    def test_schema_json_serialization(self):
        """Test JSON serialization of schemas."""
        request = VideoRequest(
            topic="JSON test",
            description="Testing JSON serialization"
        )
        
        json_str = request.model_dump_json()
        assert isinstance(json_str, str)
        assert "JSON test" in json_str
        assert "Testing JSON serialization" in json_str
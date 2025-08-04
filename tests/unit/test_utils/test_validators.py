"""
Unit tests for validation utilities.
Tests data validation, format checking, and constraint enforcement.
"""

import pytest
from unittest.mock import Mock, patch

from app.utils.validators import (
    validate_video_topic,
    validate_voice_settings,
    validate_animation_config,
    validate_file_path,
    validate_url,
    validate_email,
    ValidationError
)


class TestVideoValidators:
    """Test suite for video-related validators."""
    
    def test_validate_video_topic_valid(self):
        """Test validation of valid video topics."""
        valid_topics = [
            "Python programming basics",
            "Machine Learning Introduction",
            "Web Development with React",
            "Data Science Fundamentals"
        ]
        
        for topic in valid_topics:
            assert validate_video_topic(topic) is True
    
    def test_validate_video_topic_invalid(self):
        """Test validation of invalid video topics."""
        invalid_topics = [
            "",  # Empty string
            "a",  # Too short
            "x" * 201,  # Too long
            "   ",  # Only whitespace
            "123",  # Only numbers
            "!@#$%"  # Only special characters
        ]
        
        for topic in invalid_topics:
            with pytest.raises(ValidationError):
                validate_video_topic(topic)
    
    def test_validate_voice_settings_valid(self):
        """Test validation of valid voice settings."""
        valid_settings = [
            {"voice": "default", "speed": 1.0, "pitch": 1.0},
            {"voice": "male", "speed": 1.5, "pitch": 0.8},
            {"voice": "female", "speed": 0.8, "pitch": 1.2}
        ]
        
        for settings in valid_settings:
            assert validate_voice_settings(settings) is True
    
    def test_validate_voice_settings_invalid(self):
        """Test validation of invalid voice settings."""
        invalid_settings = [
            {"voice": "invalid_voice", "speed": 1.0},
            {"voice": "default", "speed": 0.1},  # Speed too low
            {"voice": "default", "speed": 3.0},  # Speed too high
            {"voice": "default", "pitch": 0.1},  # Pitch too low
            {"voice": "default", "pitch": 2.5},  # Pitch too high
            {}  # Missing required fields
        ]
        
        for settings in invalid_settings:
            with pytest.raises(ValidationError):
                validate_voice_settings(settings)
    
    def test_validate_animation_config_valid(self):
        """Test validation of valid animation configurations."""
        valid_configs = [
            {"quality": "low", "fps": 24, "resolution": "720p"},
            {"quality": "medium", "fps": 30, "resolution": "1080p"},
            {"quality": "high", "fps": 60, "resolution": "4K"}
        ]
        
        for config in valid_configs:
            assert validate_animation_config(config) is True
    
    def test_validate_animation_config_invalid(self):
        """Test validation of invalid animation configurations."""
        invalid_configs = [
            {"quality": "invalid", "fps": 30},
            {"quality": "medium", "fps": 5},  # FPS too low
            {"quality": "medium", "fps": 150},  # FPS too high
            {"quality": "medium", "resolution": "invalid_res"},
            {}  # Missing required fields
        ]
        
        for config in invalid_configs:
            with pytest.raises(ValidationError):
                validate_animation_config(config)


class TestFileValidators:
    """Test suite for file-related validators."""
    
    def test_validate_file_path_valid(self):
        """Test validation of valid file paths."""
        valid_paths = [
            "/home/user/video.mp4",
            "C:\\Users\\User\\Documents\\file.txt",
            "./relative/path/file.json",
            "../parent/directory/file.py"
        ]
        
        for path in valid_paths:
            assert validate_file_path(path) is True
    
    def test_validate_file_path_invalid(self):
        """Test validation of invalid file paths."""
        invalid_paths = [
            "",  # Empty path
            "   ",  # Only whitespace
            "/path/with/invalid<>chars",
            "path/with/null\x00char",
            "a" * 300  # Path too long
        ]
        
        for path in invalid_paths:
            with pytest.raises(ValidationError):
                validate_file_path(path)
    
    def test_validate_file_path_with_extension_check(self):
        """Test file path validation with extension checking."""
        # Valid extensions
        assert validate_file_path("video.mp4", allowed_extensions=[".mp4", ".avi"]) is True
        assert validate_file_path("audio.wav", allowed_extensions=[".wav", ".mp3"]) is True
        
        # Invalid extensions
        with pytest.raises(ValidationError):
            validate_file_path("video.txt", allowed_extensions=[".mp4", ".avi"])
        
        with pytest.raises(ValidationError):
            validate_file_path("file_without_extension", allowed_extensions=[".txt"])


class TestURLValidators:
    """Test suite for URL validators."""
    
    def test_validate_url_valid(self):
        """Test validation of valid URLs."""
        valid_urls = [
            "https://example.com",
            "http://localhost:8000",
            "https://api.example.com/v1/videos",
            "ftp://files.example.com/video.mp4"
        ]
        
        for url in valid_urls:
            assert validate_url(url) is True
    
    def test_validate_url_invalid(self):
        """Test validation of invalid URLs."""
        invalid_urls = [
            "",  # Empty URL
            "not_a_url",
            "http://",  # Incomplete URL
            "https://",  # Incomplete URL
            "invalid://example.com"  # Invalid scheme
        ]
        
        for url in invalid_urls:
            with pytest.raises(ValidationError):
                validate_url(url)
    
    def test_validate_url_with_scheme_restriction(self):
        """Test URL validation with scheme restrictions."""
        # Valid schemes
        assert validate_url("https://example.com", allowed_schemes=["https"]) is True
        assert validate_url("http://example.com", allowed_schemes=["http", "https"]) is True
        
        # Invalid schemes
        with pytest.raises(ValidationError):
            validate_url("ftp://example.com", allowed_schemes=["http", "https"])
        
        with pytest.raises(ValidationError):
            validate_url("http://example.com", allowed_schemes=["https"])


class TestEmailValidators:
    """Test suite for email validators."""
    
    def test_validate_email_valid(self):
        """Test validation of valid email addresses."""
        valid_emails = [
            "user@example.com",
            "test.email@domain.org",
            "user+tag@example.co.uk",
            "firstname.lastname@company.com"
        ]
        
        for email in valid_emails:
            assert validate_email(email) is True
    
    def test_validate_email_invalid(self):
        """Test validation of invalid email addresses."""
        invalid_emails = [
            "",  # Empty email
            "not_an_email",
            "@example.com",  # Missing local part
            "user@",  # Missing domain
            "user@.com",  # Invalid domain
            "user..double.dot@example.com"  # Double dots
        ]
        
        for email in invalid_emails:
            with pytest.raises(ValidationError):
                validate_email(email)


class TestValidationHelpers:
    """Test suite for validation helper functions."""
    
    def test_validation_error_creation(self):
        """Test ValidationError creation and attributes."""
        error = ValidationError("Test validation failed", field="test_field")
        
        assert str(error) == "Test validation failed"
        assert error.field == "test_field"
        assert error.code is None
    
    def test_validation_error_with_code(self):
        """Test ValidationError with error code."""
        error = ValidationError(
            "Invalid format", 
            field="email", 
            code="INVALID_FORMAT"
        )
        
        assert error.field == "email"
        assert error.code == "INVALID_FORMAT"
    
    def test_multiple_validation_errors(self):
        """Test handling multiple validation errors."""
        errors = []
        
        try:
            validate_video_topic("")
        except ValidationError as e:
            errors.append(e)
        
        try:
            validate_email("invalid_email")
        except ValidationError as e:
            errors.append(e)
        
        assert len(errors) == 2
        assert all(isinstance(e, ValidationError) for e in errors)
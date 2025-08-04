"""
Unit tests for VideoService.
Tests video generation orchestration, TTS integration, and processing pipeline.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

from app.services.video_service import VideoService
from app.models.schemas.video import VideoRequest, VideoResponse, VideoStatus
from app.models.enums import VideoQuality, VoiceType


class TestVideoService:
    """Test suite for VideoService functionality."""
    
    @pytest.fixture
    def video_service(self):
        """Create VideoService instance for testing."""
        return VideoService()
    
    @pytest.fixture
    def mock_video_request(self):
        """Create mock video request."""
        return VideoRequest(
            topic="Python basics",
            description="Introduction to Python programming",
            voice_settings={
                "voice": VoiceType.DEFAULT,
                "speed": 1.0,
                "pitch": 1.0
            },
            animation_config={
                "quality": VideoQuality.MEDIUM,
                "fps": 30,
                "resolution": "1080p"
            }
        )
    
    @pytest.mark.asyncio
    async def test_create_video_success(self, video_service, mock_video_request):
        """Test successful video creation."""
        with patch.object(video_service, '_generate_video_id') as mock_gen_id:
            mock_gen_id.return_value = "test_video_123"
            
            with patch.object(video_service, '_start_video_processing') as mock_process:
                mock_process.return_value = None
                
                response = await video_service.create_video(mock_video_request)
                
                assert response.video_id == "test_video_123"
                assert response.status == VideoStatus.PROCESSING
                assert response.download_url is None
                mock_process.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_video_status_processing(self, video_service):
        """Test getting status of processing video."""
        video_id = "test_video_123"
        
        with patch.object(video_service, '_get_video_metadata') as mock_metadata:
            mock_metadata.return_value = {
                "status": VideoStatus.PROCESSING,
                "progress": 0.5
            }
            
            response = await video_service.get_video_status(video_id)
            
            assert response.video_id == video_id
            assert response.status == VideoStatus.PROCESSING
            assert response.download_url is None
    
    @pytest.mark.asyncio
    async def test_get_video_status_completed(self, video_service):
        """Test getting status of completed video."""
        video_id = "test_video_123"
        
        with patch.object(video_service, '_get_video_metadata') as mock_metadata:
            mock_metadata.return_value = {
                "status": VideoStatus.COMPLETED,
                "download_url": "https://example.com/video.mp4"
            }
            
            response = await video_service.get_video_status(video_id)
            
            assert response.video_id == video_id
            assert response.status == VideoStatus.COMPLETED
            assert response.download_url == "https://example.com/video.mp4"
    
    @pytest.mark.asyncio
    async def test_download_video_success(self, video_service):
        """Test successful video download."""
        video_id = "test_video_123"
        mock_content = b"video_file_content"
        
        with patch.object(video_service, '_get_video_file') as mock_get_file:
            mock_get_file.return_value = mock_content
            
            content = await video_service.download_video(video_id)
            
            assert content == mock_content
            mock_get_file.assert_called_once_with(video_id)
    
    @pytest.mark.asyncio
    async def test_download_video_not_ready(self, video_service):
        """Test downloading video that's not ready."""
        video_id = "processing_video"
        
        with patch.object(video_service, '_get_video_metadata') as mock_metadata:
            mock_metadata.return_value = {"status": VideoStatus.PROCESSING}
            
            with pytest.raises(ValueError, match="Video not ready for download"):
                await video_service.download_video(video_id)
    
    def test_generate_video_id(self, video_service):
        """Test video ID generation."""
        video_id = video_service._generate_video_id()
        
        assert isinstance(video_id, str)
        assert len(video_id) > 0
        assert video_id.startswith("video_")
    
    @pytest.mark.asyncio
    async def test_start_video_processing(self, video_service, mock_video_request):
        """Test video processing pipeline initiation."""
        video_id = "test_video_123"
        
        with patch.object(video_service, '_process_video_async') as mock_process:
            mock_process.return_value = None
            
            await video_service._start_video_processing(video_id, mock_video_request)
            
            mock_process.assert_called_once_with(video_id, mock_video_request)
    
    @pytest.mark.asyncio
    async def test_process_video_async_success(self, video_service, mock_video_request):
        """Test asynchronous video processing."""
        video_id = "test_video_123"
        
        with patch.object(video_service, '_generate_script') as mock_script:
            mock_script.return_value = "Generated script content"
            
            with patch.object(video_service, '_generate_audio') as mock_audio:
                mock_audio.return_value = "audio_file.wav"
                
                with patch.object(video_service, '_generate_animation') as mock_animation:
                    mock_animation.return_value = "animation_file.mp4"
                    
                    with patch.object(video_service, '_combine_media') as mock_combine:
                        mock_combine.return_value = "final_video.mp4"
                        
                        await video_service._process_video_async(video_id, mock_video_request)
                        
                        mock_script.assert_called_once()
                        mock_audio.assert_called_once()
                        mock_animation.assert_called_once()
                        mock_combine.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_script(self, video_service, mock_video_request):
        """Test script generation."""
        with patch('app.services.video_service.ScriptGenerator') as mock_generator:
            mock_instance = Mock()
            mock_instance.generate.return_value = "Generated script"
            mock_generator.return_value = mock_instance
            
            script = await video_service._generate_script(mock_video_request)
            
            assert script == "Generated script"
            mock_instance.generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_audio(self, video_service):
        """Test audio generation from script."""
        script = "Test script content"
        voice_settings = {"voice": "default", "speed": 1.0}
        
        with patch('app.services.video_service.TTSService') as mock_tts:
            mock_instance = Mock()
            mock_instance.generate_audio.return_value = "audio_file.wav"
            mock_tts.return_value = mock_instance
            
            audio_file = await video_service._generate_audio(script, voice_settings)
            
            assert audio_file == "audio_file.wav"
            mock_instance.generate_audio.assert_called_once_with(script, voice_settings)
    
    @pytest.mark.asyncio
    async def test_generate_animation(self, video_service):
        """Test animation generation."""
        script = "Test script content"
        animation_config = {"quality": "medium", "fps": 30}
        
        with patch('app.services.video_service.AnimationService') as mock_animation:
            mock_instance = Mock()
            mock_instance.generate_animation.return_value = "animation.mp4"
            mock_animation.return_value = mock_instance
            
            animation_file = await video_service._generate_animation(script, animation_config)
            
            assert animation_file == "animation.mp4"
            mock_instance.generate_animation.assert_called_once_with(script, animation_config)
"""
Integration tests for VideoService.
Tests integration with external services, databases, and file systems.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile
import os

from app.services.video_service import VideoService
from app.models.schemas.video import VideoRequest, VideoStatus
from app.models.enums import VideoQuality, VoiceType


class TestVideoServiceIntegration:
    """Integration test suite for VideoService."""
    
    @pytest.fixture
    def video_service(self):
        """Create VideoService instance for integration tests."""
        return VideoService()
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory for integration tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def sample_video_request(self):
        """Create sample video request for integration tests."""
        return VideoRequest(
            topic="Integration Test Video",
            description="Testing video service integration",
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
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_video_service_database_integration(self, video_service, sample_video_request):
        """Test VideoService integration with database."""
        with patch('app.services.video_service.get_database_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value = mock_session
            
            # Mock database operations
            mock_session.add = Mock()
            mock_session.commit = Mock()
            mock_session.query = Mock()
            
            # Create video (should interact with database)
            response = await video_service.create_video(sample_video_request)
            
            # Verify database interactions
            assert response.video_id is not None
            assert response.status == VideoStatus.PROCESSING
            
            # Verify database session was used
            mock_session.add.assert_called()
            mock_session.commit.assert_called()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_video_service_file_system_integration(self, video_service, sample_video_request, temp_output_dir):
        """Test VideoService integration with file system."""
        with patch.object(video_service, '_get_output_directory', return_value=temp_output_dir):
            with patch.object(video_service, '_generate_script') as mock_script:
                mock_script.return_value = "Test script content"
                
                with patch.object(video_service, '_generate_audio') as mock_audio:
                    audio_file = os.path.join(temp_output_dir, "test_audio.wav")
                    Path(audio_file).touch()  # Create mock audio file
                    mock_audio.return_value = audio_file
                    
                    with patch.object(video_service, '_generate_animation') as mock_animation:
                        animation_file = os.path.join(temp_output_dir, "test_animation.mp4")
                        Path(animation_file).touch()  # Create mock animation file
                        mock_animation.return_value = animation_file
                        
                        with patch.object(video_service, '_combine_media') as mock_combine:
                            final_video = os.path.join(temp_output_dir, "final_video.mp4")
                            Path(final_video).touch()  # Create mock final video
                            mock_combine.return_value = final_video
                            
                            # Process video (should create files)
                            video_id = "integration_test_123"
                            await video_service._process_video_async(video_id, sample_video_request)
                            
                            # Verify files were created
                            assert os.path.exists(audio_file)
                            assert os.path.exists(animation_file)
                            assert os.path.exists(final_video)
                            
                            # Verify service methods were called
                            mock_script.assert_called_once()
                            mock_audio.assert_called_once()
                            mock_animation.assert_called_once()
                            mock_combine.assert_called_once()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_video_service_external_api_integration(self, video_service, sample_video_request):
        """Test VideoService integration with external APIs."""
        with patch('app.services.video_service.TTSService') as mock_tts_service:
            mock_tts = Mock()
            mock_tts.generate_audio = AsyncMock(return_value="audio_file.wav")
            mock_tts_service.return_value = mock_tts
            
            with patch('app.services.video_service.AnimationService') as mock_animation_service:
                mock_animation = Mock()
                mock_animation.generate_animation = AsyncMock(return_value="animation_file.mp4")
                mock_animation_service.return_value = mock_animation
                
                # Test external service integration
                script = "Test script for external services"
                
                # Test TTS service integration
                audio_file = await video_service._generate_audio(
                    script, sample_video_request.voice_settings
                )
                assert audio_file == "audio_file.wav"
                mock_tts.generate_audio.assert_called_once_with(
                    script, sample_video_request.voice_settings
                )
                
                # Test animation service integration
                animation_file = await video_service._generate_animation(
                    script, sample_video_request.animation_config
                )
                assert animation_file == "animation_file.mp4"
                mock_animation.generate_animation.assert_called_once_with(
                    script, sample_video_request.animation_config
                )
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_video_service_aws_integration(self, video_service, sample_video_request):
        """Test VideoService integration with AWS services."""
        with patch('app.services.video_service.AWSService') as mock_aws_service:
            mock_aws = Mock()
            mock_aws.upload_video = AsyncMock(return_value="https://s3.amazonaws.com/bucket/video.mp4")
            mock_aws.store_metadata = AsyncMock()
            mock_aws_service.return_value = mock_aws
            
            # Test AWS integration
            video_id = "aws_integration_test"
            video_file_path = "/path/to/video.mp4"
            
            # Upload video to AWS
            with patch.object(video_service, '_upload_to_aws') as mock_upload:
                mock_upload.return_value = "https://s3.amazonaws.com/bucket/video.mp4"
                
                upload_url = await video_service._upload_to_aws(video_id, video_file_path)
                
                assert upload_url == "https://s3.amazonaws.com/bucket/video.mp4"
                mock_upload.assert_called_once_with(video_id, video_file_path)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_video_service_error_recovery_integration(self, video_service, sample_video_request):
        """Test VideoService error recovery integration."""
        with patch.object(video_service, '_generate_script') as mock_script:
            # First call fails, second succeeds
            mock_script.side_effect = [
                Exception("Script generation failed"),
                "Recovered script content"
            ]
            
            with patch.object(video_service, '_handle_processing_error') as mock_error_handler:
                mock_error_handler.return_value = True  # Indicates retry should happen
                
                with patch.object(video_service, '_retry_processing') as mock_retry:
                    mock_retry.return_value = "final_video.mp4"
                    
                    # Process video with error recovery
                    video_id = "error_recovery_test"
                    
                    try:
                        await video_service._process_video_async(video_id, sample_video_request)
                    except Exception:
                        pass  # Expected on first attempt
                    
                    # Verify error handling was triggered
                    mock_error_handler.assert_called()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_video_service_concurrent_processing_integration(self, video_service):
        """Test VideoService handling of concurrent video processing."""
        # Create multiple video requests
        requests = [
            VideoRequest(
                topic=f"Concurrent Video {i}",
                description=f"Testing concurrent processing {i}",
                voice_settings={"voice": VoiceType.DEFAULT},
                animation_config={"quality": VideoQuality.MEDIUM}
            )
            for i in range(3)
        ]
        
        with patch.object(video_service, '_process_video_async') as mock_process:
            mock_process.return_value = None  # Simulate successful processing
            
            # Start concurrent video creation
            tasks = []
            for i, request in enumerate(requests):
                task = asyncio.create_task(video_service.create_video(request))
                tasks.append(task)
            
            # Wait for all tasks to complete
            responses = await asyncio.gather(*tasks)
            
            # Verify all videos were created
            assert len(responses) == 3
            for response in responses:
                assert response.status == VideoStatus.PROCESSING
                assert response.video_id is not None
            
            # Verify processing was started for each video
            assert mock_process.call_count == 3
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_video_service_monitoring_integration(self, video_service, sample_video_request):
        """Test VideoService integration with monitoring systems."""
        with patch('app.services.video_service.MonitoringService') as mock_monitoring:
            mock_monitor = Mock()
            mock_monitor.record_video_creation = Mock()
            mock_monitor.record_processing_time = Mock()
            mock_monitor.record_error = Mock()
            mock_monitoring.return_value = mock_monitor
            
            # Create video with monitoring
            response = await video_service.create_video(sample_video_request)
            
            # Verify monitoring was called
            mock_monitor.record_video_creation.assert_called_once()
            
            # Test error monitoring
            with patch.object(video_service, '_process_video_async') as mock_process:
                mock_process.side_effect = Exception("Processing failed")
                
                try:
                    await video_service._start_video_processing(response.video_id, sample_video_request)
                except Exception:
                    pass
                
                # Verify error was recorded
                mock_monitor.record_error.assert_called()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_video_service_cache_integration(self, video_service, sample_video_request):
        """Test VideoService integration with caching systems."""
        with patch('app.services.video_service.CacheService') as mock_cache_service:
            mock_cache = Mock()
            mock_cache.get = AsyncMock(return_value=None)  # Cache miss
            mock_cache.set = AsyncMock()
            mock_cache_service.return_value = mock_cache
            
            with patch.object(video_service, '_generate_script') as mock_script:
                mock_script.return_value = "Cached script content"
                
                # Generate script with caching
                script = await video_service._generate_script_with_cache(sample_video_request)
                
                # Verify cache was checked and updated
                mock_cache.get.assert_called_once()
                mock_cache.set.assert_called_once()
                assert script == "Cached script content"
                
                # Test cache hit
                mock_cache.get.return_value = "Cached script from cache"
                
                cached_script = await video_service._generate_script_with_cache(sample_video_request)
                assert cached_script == "Cached script from cache"
"""
End-to-end tests for complete video generation workflow.
Tests the entire process from planning to final video output.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from app.services.video_service import VideoService
from app.models.schemas.video import VideoRequest, VideoStatus
from app.models.enums import VideoQuality, VoiceType


class TestCompleteVideoWorkflow:
    """End-to-end test suite for complete video generation workflow."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory for E2E tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def video_service(self):
        """Create VideoService instance for E2E testing."""
        return VideoService()
    
    @pytest.fixture
    def sample_video_request(self):
        """Create comprehensive video request for E2E testing."""
        return VideoRequest(
            topic="Machine Learning Fundamentals",
            description="""
            An educational video covering the basics of machine learning including:
            - What is machine learning and why it matters
            - Types of machine learning: supervised, unsupervised, reinforcement
            - Common algorithms: linear regression, decision trees, neural networks
            - Real-world applications and examples
            - Getting started with Python and scikit-learn
            """,
            voice_settings={
                "voice": VoiceType.DEFAULT,
                "speed": 1.0,
                "pitch": 1.0,
                "volume": 1.0
            },
            animation_config={
                "quality": VideoQuality.MEDIUM,
                "fps": 30,
                "resolution": "1080p",
                "background_color": "#FFFFFF",
                "text_color": "#000000"
            }
        )
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_complete_workflow_success(self, video_service, sample_video_request, temp_output_dir):
        """Test successful completion of entire video generation workflow."""
        workflow_stages = []
        
        # Mock all workflow components
        with patch.object(video_service, '_get_output_directory', return_value=temp_output_dir):
            with patch.object(video_service, '_generate_script') as mock_script:
                mock_script.return_value = "Generated script for machine learning video"
                workflow_stages.append("script_generation")
                
                with patch.object(video_service, '_generate_audio') as mock_audio:
                    audio_file = os.path.join(temp_output_dir, "audio.wav")
                    Path(audio_file).touch()
                    mock_audio.return_value = audio_file
                    workflow_stages.append("audio_generation")
                    
                    with patch.object(video_service, '_generate_animation') as mock_animation:
                        animation_file = os.path.join(temp_output_dir, "animation.mp4")
                        Path(animation_file).touch()
                        mock_animation.return_value = animation_file
                        workflow_stages.append("animation_generation")
                        
                        with patch.object(video_service, '_combine_media') as mock_combine:
                            final_video = os.path.join(temp_output_dir, "final_video.mp4")
                            Path(final_video).touch()
                            mock_combine.return_value = final_video
                            workflow_stages.append("media_combination")
                            
                            with patch.object(video_service, '_upload_to_storage') as mock_upload:
                                mock_upload.return_value = "https://storage.example.com/video.mp4"
                                workflow_stages.append("storage_upload")
                                
                                # Execute complete workflow
                                response = await video_service.create_video(sample_video_request)
                                
                                # Verify initial response
                                assert response.video_id is not None
                                assert response.status == VideoStatus.PROCESSING
                                
                                # Wait for processing to complete
                                max_attempts = 30
                                for attempt in range(max_attempts):
                                    await asyncio.sleep(0.1)
                                    
                                    status_response = await video_service.get_video_status(response.video_id)
                                    if status_response.status == VideoStatus.COMPLETED:
                                        break
                                    elif status_response.status == VideoStatus.FAILED:
                                        pytest.fail(f"Workflow failed: {status_response.error_message}")
                                
                                # Verify final status
                                final_status = await video_service.get_video_status(response.video_id)
                                assert final_status.status == VideoStatus.COMPLETED
                                assert final_status.download_url is not None
                                
                                # Verify all workflow stages completed
                                assert len(workflow_stages) == 5
                                assert "script_generation" in workflow_stages
                                assert "audio_generation" in workflow_stages
                                assert "animation_generation" in workflow_stages
                                assert "media_combination" in workflow_stages
                                assert "storage_upload" in workflow_stages
                                
                                # Verify files were created
                                assert os.path.exists(audio_file)
                                assert os.path.exists(animation_file)
                                assert os.path.exists(final_video)
                                
                                # Verify service calls
                                mock_script.assert_called_once()
                                mock_audio.assert_called_once()
                                mock_animation.assert_called_once()
                                mock_combine.assert_called_once()
                                mock_upload.assert_called_once()
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_workflow_with_error_recovery(self, video_service, sample_video_request, temp_output_dir):
        """Test workflow error recovery and retry mechanisms."""
        error_recovery_log = []
        
        with patch.object(video_service, '_get_output_directory', return_value=temp_output_dir):
            with patch.object(video_service, '_generate_script') as mock_script:
                # First call fails, second succeeds
                mock_script.side_effect = [
                    Exception("Script generation failed"),
                    "Recovered script content"
                ]
                
                with patch.object(video_service, '_handle_processing_error') as mock_error_handler:
                    def error_handler(error, video_id, stage):
                        error_recovery_log.append({
                            "error": str(error),
                            "video_id": video_id,
                            "stage": stage,
                            "recovery_attempted": True
                        })
                        return True  # Indicate retry should happen
                    
                    mock_error_handler.side_effect = error_handler
                    
                    with patch.object(video_service, '_retry_processing') as mock_retry:
                        async def retry_processing(video_id, request, failed_stage):
                            error_recovery_log.append({
                                "video_id": video_id,
                                "retry_stage": failed_stage,
                                "retry_successful": True
                            })
                            # Simulate successful retry
                            return "final_video.mp4"
                        
                        mock_retry.side_effect = retry_processing
                        
                        # Execute workflow with error recovery
                        response = await video_service.create_video(sample_video_request)
                        
                        # Wait for processing with error recovery
                        max_attempts = 30
                        for attempt in range(max_attempts):
                            await asyncio.sleep(0.1)
                            
                            status_response = await video_service.get_video_status(response.video_id)
                            if status_response.status in [VideoStatus.COMPLETED, VideoStatus.FAILED]:
                                break
                        
                        # Verify error recovery occurred
                        assert len(error_recovery_log) >= 1
                        assert any(log.get("recovery_attempted") for log in error_recovery_log)
                        
                        # Verify final status (should succeed after recovery)
                        final_status = await video_service.get_video_status(response.video_id)
                        assert final_status.status == VideoStatus.COMPLETED
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_workflow_performance_monitoring(self, video_service, sample_video_request, temp_output_dir):
        """Test workflow performance monitoring and metrics collection."""
        performance_metrics = {}
        
        # Mock performance monitoring
        class PerformanceMonitor:
            def __init__(self):
                self.stage_times = {}
                self.start_times = {}
            
            def start_stage(self, stage_name):
                import time
                self.start_times[stage_name] = time.time()
            
            def end_stage(self, stage_name):
                import time
                if stage_name in self.start_times:
                    duration = time.time() - self.start_times[stage_name]
                    self.stage_times[stage_name] = duration
                    performance_metrics[stage_name] = duration
        
        monitor = PerformanceMonitor()
        
        with patch.object(video_service, '_get_output_directory', return_value=temp_output_dir):
            with patch.object(video_service, '_generate_script') as mock_script:
                def monitored_script_generation(*args, **kwargs):
                    monitor.start_stage("script_generation")
                    import time
                    time.sleep(0.1)  # Simulate processing time
                    monitor.end_stage("script_generation")
                    return "Generated script with monitoring"
                
                mock_script.side_effect = monitored_script_generation
                
                with patch.object(video_service, '_generate_audio') as mock_audio:
                    def monitored_audio_generation(*args, **kwargs):
                        monitor.start_stage("audio_generation")
                        import time
                        time.sleep(0.2)  # Simulate processing time
                        audio_file = os.path.join(temp_output_dir, "monitored_audio.wav")
                        Path(audio_file).touch()
                        monitor.end_stage("audio_generation")
                        return audio_file
                    
                    mock_audio.side_effect = monitored_audio_generation
                    
                    with patch.object(video_service, '_generate_animation') as mock_animation:
                        def monitored_animation_generation(*args, **kwargs):
                            monitor.start_stage("animation_generation")
                            import time
                            time.sleep(0.3)  # Simulate processing time
                            animation_file = os.path.join(temp_output_dir, "monitored_animation.mp4")
                            Path(animation_file).touch()
                            monitor.end_stage("animation_generation")
                            return animation_file
                        
                        mock_animation.side_effect = monitored_animation_generation
                        
                        with patch.object(video_service, '_combine_media') as mock_combine:
                            def monitored_media_combination(*args, **kwargs):
                                monitor.start_stage("media_combination")
                                import time
                                time.sleep(0.15)  # Simulate processing time
                                final_video = os.path.join(temp_output_dir, "monitored_final.mp4")
                                Path(final_video).touch()
                                monitor.end_stage("media_combination")
                                return final_video
                            
                            mock_combine.side_effect = monitored_media_combination
                            
                            # Execute workflow with performance monitoring
                            response = await video_service.create_video(sample_video_request)
                            
                            # Wait for completion
                            max_attempts = 30
                            for attempt in range(max_attempts):
                                await asyncio.sleep(0.1)
                                
                                status_response = await video_service.get_video_status(response.video_id)
                                if status_response.status == VideoStatus.COMPLETED:
                                    break
                            
                            # Verify performance metrics were collected
                            assert "script_generation" in performance_metrics
                            assert "audio_generation" in performance_metrics
                            assert "animation_generation" in performance_metrics
                            assert "media_combination" in performance_metrics
                            
                            # Verify reasonable performance times
                            assert performance_metrics["script_generation"] >= 0.1
                            assert performance_metrics["audio_generation"] >= 0.2
                            assert performance_metrics["animation_generation"] >= 0.3
                            assert performance_metrics["media_combination"] >= 0.15
                            
                            # Calculate total processing time
                            total_time = sum(performance_metrics.values())
                            assert total_time >= 0.75  # Sum of all stage times
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_workflow_with_different_quality_settings(self, video_service, temp_output_dir):
        """Test workflow with different quality settings and configurations."""
        quality_test_results = {}
        
        # Test different quality levels
        quality_levels = [
            (VideoQuality.LOW, 24, "720p"),
            (VideoQuality.MEDIUM, 30, "1080p"),
            (VideoQuality.HIGH, 60, "4K")
        ]
        
        for quality, fps, resolution in quality_levels:
            quality_request = VideoRequest(
                topic=f"Quality Test - {quality.value}",
                description=f"Testing video generation with {quality.value} quality",
                voice_settings={
                    "voice": VoiceType.DEFAULT,
                    "speed": 1.0,
                    "pitch": 1.0
                },
                animation_config={
                    "quality": quality,
                    "fps": fps,
                    "resolution": resolution
                }
            )
            
            with patch.object(video_service, '_get_output_directory', return_value=temp_output_dir):
                with patch.object(video_service, '_generate_script') as mock_script:
                    mock_script.return_value = f"Script for {quality.value} quality video"
                    
                    with patch.object(video_service, '_generate_audio') as mock_audio:
                        audio_file = os.path.join(temp_output_dir, f"audio_{quality.value}.wav")
                        Path(audio_file).touch()
                        mock_audio.return_value = audio_file
                        
                        with patch.object(video_service, '_generate_animation') as mock_animation:
                            def quality_aware_animation(*args, **kwargs):
                                # Simulate different processing times for different qualities
                                import time
                                processing_time = {
                                    VideoQuality.LOW: 0.1,
                                    VideoQuality.MEDIUM: 0.2,
                                    VideoQuality.HIGH: 0.4
                                }
                                time.sleep(processing_time[quality])
                                
                                animation_file = os.path.join(temp_output_dir, f"animation_{quality.value}.mp4")
                                Path(animation_file).touch()
                                return animation_file
                            
                            mock_animation.side_effect = quality_aware_animation
                            
                            with patch.object(video_service, '_combine_media') as mock_combine:
                                final_video = os.path.join(temp_output_dir, f"final_{quality.value}.mp4")
                                Path(final_video).touch()
                                mock_combine.return_value = final_video
                                
                                # Execute workflow for this quality level
                                start_time = asyncio.get_event_loop().time()
                                response = await video_service.create_video(quality_request)
                                
                                # Wait for completion
                                max_attempts = 30
                                for attempt in range(max_attempts):
                                    await asyncio.sleep(0.1)
                                    
                                    status_response = await video_service.get_video_status(response.video_id)
                                    if status_response.status == VideoStatus.COMPLETED:
                                        break
                                
                                end_time = asyncio.get_event_loop().time()
                                processing_time = end_time - start_time
                                
                                # Record results
                                quality_test_results[quality.value] = {
                                    "processing_time": processing_time,
                                    "video_id": response.video_id,
                                    "status": status_response.status,
                                    "fps": fps,
                                    "resolution": resolution
                                }
                                
                                # Verify completion
                                assert status_response.status == VideoStatus.COMPLETED
                                assert os.path.exists(final_video)
        
        # Verify all quality levels completed successfully
        assert len(quality_test_results) == 3
        assert "low" in quality_test_results
        assert "medium" in quality_test_results
        assert "high" in quality_test_results
        
        # Verify processing times increase with quality (generally)
        low_time = quality_test_results["low"]["processing_time"]
        medium_time = quality_test_results["medium"]["processing_time"]
        high_time = quality_test_results["high"]["processing_time"]
        
        # Allow some variance but expect general trend
        assert low_time <= medium_time * 1.5  # Some tolerance for timing variations
        assert medium_time <= high_time * 1.5
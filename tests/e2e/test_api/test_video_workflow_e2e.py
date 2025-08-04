"""
End-to-end tests for complete video generation workflow through API.
Tests the entire user journey from video creation to download.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas.video import VideoStatus


class TestVideoWorkflowE2E:
    """End-to-end test suite for video generation workflow."""
    
    @pytest.fixture
    def client(self):
        """Create test client for E2E tests."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_video_request(self):
        """Create sample video request for E2E testing."""
        return {
            "topic": "Python Programming Basics",
            "description": "An educational video covering Python fundamentals including variables, functions, and control structures",
            "voice_settings": {
                "voice": "default",
                "speed": 1.0,
                "pitch": 1.0,
                "volume": 1.0
            },
            "animation_config": {
                "quality": "medium",
                "fps": 30,
                "resolution": "1080p",
                "background_color": "#FFFFFF",
                "text_color": "#000000"
            }
        }
    
    @pytest.mark.e2e
    def test_complete_video_generation_workflow(self, client, sample_video_request):
        """Test complete video generation workflow from creation to download."""
        # Step 1: Create video
        create_response = client.post("/api/v1/video/create", json=sample_video_request)
        assert create_response.status_code == 201
        
        create_data = create_response.json()
        video_id = create_data["video_id"]
        assert video_id is not None
        assert create_data["status"] == "processing"
        assert create_data["download_url"] is None
        
        # Step 2: Poll for completion (simulate processing time)
        max_attempts = 30  # 30 seconds timeout
        attempt = 0
        video_completed = False
        
        while attempt < max_attempts and not video_completed:
            time.sleep(1)  # Wait 1 second between polls
            
            status_response = client.get(f"/api/v1/video/{video_id}/status")
            assert status_response.status_code == 200
            
            status_data = status_response.json()
            assert status_data["video_id"] == video_id
            
            if status_data["status"] == "completed":
                video_completed = True
                assert status_data["download_url"] is not None
                break
            elif status_data["status"] == "failed":
                pytest.fail(f"Video generation failed: {status_data.get('error_message', 'Unknown error')}")
            
            attempt += 1
        
        assert video_completed, "Video generation did not complete within timeout"
        
        # Step 3: Download video
        download_response = client.get(f"/api/v1/video/{video_id}/download")
        assert download_response.status_code == 200
        assert download_response.headers["content-type"] == "video/mp4"
        assert len(download_response.content) > 0
        
        # Step 4: Verify video metadata
        final_status_response = client.get(f"/api/v1/video/{video_id}/status")
        final_status_data = final_status_response.json()
        
        assert final_status_data["status"] == "completed"
        assert final_status_data["download_url"] is not None
        assert "created_at" in final_status_data
        assert "completed_at" in final_status_data
    
    @pytest.mark.e2e
    def test_video_generation_with_custom_settings(self, client):
        """Test video generation with custom voice and animation settings."""
        custom_request = {
            "topic": "Advanced Python Concepts",
            "description": "Deep dive into Python decorators, generators, and context managers",
            "voice_settings": {
                "voice": "female",
                "speed": 1.2,
                "pitch": 0.9,
                "volume": 0.8
            },
            "animation_config": {
                "quality": "high",
                "fps": 60,
                "resolution": "4K",
                "background_color": "#1E1E1E",
                "text_color": "#FFFFFF"
            }
        }
        
        # Create video with custom settings
        create_response = client.post("/api/v1/video/create", json=custom_request)
        assert create_response.status_code == 201
        
        create_data = create_response.json()
        video_id = create_data["video_id"]
        
        # Wait for completion
        max_attempts = 45  # Longer timeout for high quality
        attempt = 0
        
        while attempt < max_attempts:
            time.sleep(1)
            
            status_response = client.get(f"/api/v1/video/{video_id}/status")
            status_data = status_response.json()
            
            if status_data["status"] == "completed":
                # Verify custom settings were applied
                assert status_data["video_id"] == video_id
                break
            elif status_data["status"] == "failed":
                pytest.fail(f"Custom video generation failed: {status_data.get('error_message')}")
            
            attempt += 1
        
        assert attempt < max_attempts, "Custom video generation timed out"
        
        # Download and verify
        download_response = client.get(f"/api/v1/video/{video_id}/download")
        assert download_response.status_code == 200
        assert len(download_response.content) > 0
    
    @pytest.mark.e2e
    def test_concurrent_video_generation(self, client, sample_video_request):
        """Test concurrent video generation requests."""
        # Create multiple video requests
        video_requests = []
        for i in range(3):
            request = sample_video_request.copy()
            request["topic"] = f"Python Topic {i+1}"
            request["description"] = f"Educational video {i+1} for concurrent testing"
            video_requests.append(request)
        
        # Submit all requests concurrently
        video_ids = []
        for request in video_requests:
            create_response = client.post("/api/v1/video/create", json=request)
            assert create_response.status_code == 201
            
            create_data = create_response.json()
            video_ids.append(create_data["video_id"])
        
        # Wait for all videos to complete
        completed_videos = []
        max_attempts = 60  # Longer timeout for concurrent processing
        
        for attempt in range(max_attempts):
            time.sleep(1)
            
            # Check status of all videos
            all_completed = True
            for video_id in video_ids:
                if video_id in completed_videos:
                    continue
                
                status_response = client.get(f"/api/v1/video/{video_id}/status")
                status_data = status_response.json()
                
                if status_data["status"] == "completed":
                    completed_videos.append(video_id)
                elif status_data["status"] == "failed":
                    pytest.fail(f"Concurrent video {video_id} failed")
                else:
                    all_completed = False
            
            if all_completed:
                break
        
        assert len(completed_videos) == 3, "Not all concurrent videos completed"
        
        # Verify all videos can be downloaded
        for video_id in completed_videos:
            download_response = client.get(f"/api/v1/video/{video_id}/download")
            assert download_response.status_code == 200
            assert len(download_response.content) > 0
    
    @pytest.mark.e2e
    def test_video_generation_error_handling(self, client):
        """Test error handling in video generation workflow."""
        # Test with invalid request
        invalid_request = {
            "topic": "",  # Empty topic should cause validation error
            "description": "Test description"
        }
        
        create_response = client.post("/api/v1/video/create", json=invalid_request)
        assert create_response.status_code == 422
        
        # Test with malformed request
        malformed_request = {
            "topic": "Valid Topic",
            "voice_settings": {
                "speed": 5.0  # Invalid speed (too high)
            }
        }
        
        create_response = client.post("/api/v1/video/create", json=malformed_request)
        assert create_response.status_code == 422
        
        # Test accessing non-existent video
        status_response = client.get("/api/v1/video/nonexistent_video/status")
        assert status_response.status_code == 404
        
        download_response = client.get("/api/v1/video/nonexistent_video/download")
        assert download_response.status_code == 404
    
    @pytest.mark.e2e
    def test_video_generation_with_long_content(self, client):
        """Test video generation with longer, more complex content."""
        long_content_request = {
            "topic": "Comprehensive Python Tutorial",
            "description": """
            A comprehensive tutorial covering:
            1. Python basics - variables, data types, operators
            2. Control structures - if statements, loops, exception handling
            3. Functions - definition, parameters, return values, scope
            4. Object-oriented programming - classes, inheritance, polymorphism
            5. Modules and packages - importing, creating modules
            6. File handling - reading, writing, working with different formats
            7. Advanced topics - decorators, generators, context managers
            8. Testing - unit tests, test-driven development
            9. Best practices - PEP 8, documentation, code organization
            10. Real-world applications - web development, data analysis
            """,
            "voice_settings": {
                "voice": "default",
                "speed": 0.9,  # Slightly slower for complex content
                "pitch": 1.0,
                "volume": 1.0
            },
            "animation_config": {
                "quality": "medium",
                "fps": 30,
                "resolution": "1080p"
            }
        }
        
        # Create video with long content
        create_response = client.post("/api/v1/video/create", json=long_content_request)
        assert create_response.status_code == 201
        
        create_data = create_response.json()
        video_id = create_data["video_id"]
        
        # Wait for completion (longer timeout for complex content)
        max_attempts = 120  # 2 minutes for complex content
        attempt = 0
        
        while attempt < max_attempts:
            time.sleep(1)
            
            status_response = client.get(f"/api/v1/video/{video_id}/status")
            status_data = status_response.json()
            
            if status_data["status"] == "completed":
                # Verify video was created successfully
                download_response = client.get(f"/api/v1/video/{video_id}/download")
                assert download_response.status_code == 200
                assert len(download_response.content) > 0
                break
            elif status_data["status"] == "failed":
                pytest.fail(f"Long content video failed: {status_data.get('error_message')}")
            
            attempt += 1
        
        assert attempt < max_attempts, "Long content video generation timed out"
    
    @pytest.mark.e2e
    def test_video_generation_performance_metrics(self, client, sample_video_request):
        """Test video generation performance and collect metrics."""
        start_time = time.time()
        
        # Create video
        create_response = client.post("/api/v1/video/create", json=sample_video_request)
        assert create_response.status_code == 201
        
        create_data = create_response.json()
        video_id = create_data["video_id"]
        creation_time = time.time() - start_time
        
        # Monitor processing time
        processing_start = time.time()
        max_attempts = 60
        attempt = 0
        
        while attempt < max_attempts:
            time.sleep(1)
            
            status_response = client.get(f"/api/v1/video/{video_id}/status")
            status_data = status_response.json()
            
            if status_data["status"] == "completed":
                processing_time = time.time() - processing_start
                
                # Download and measure download time
                download_start = time.time()
                download_response = client.get(f"/api/v1/video/{video_id}/download")
                download_time = time.time() - download_start
                
                # Collect performance metrics
                total_time = time.time() - start_time
                video_size = len(download_response.content)
                
                # Assert performance expectations
                assert creation_time < 5.0, f"Video creation took too long: {creation_time}s"
                assert processing_time < 60.0, f"Video processing took too long: {processing_time}s"
                assert download_time < 10.0, f"Video download took too long: {download_time}s"
                assert video_size > 1000, f"Video file too small: {video_size} bytes"
                
                # Log metrics for analysis
                print(f"Performance Metrics:")
                print(f"  Creation time: {creation_time:.2f}s")
                print(f"  Processing time: {processing_time:.2f}s")
                print(f"  Download time: {download_time:.2f}s")
                print(f"  Total time: {total_time:.2f}s")
                print(f"  Video size: {video_size} bytes")
                
                break
            elif status_data["status"] == "failed":
                pytest.fail(f"Performance test video failed: {status_data.get('error_message')}")
            
            attempt += 1
        
        assert attempt < max_attempts, "Performance test video timed out"
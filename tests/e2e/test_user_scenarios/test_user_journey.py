"""
End-to-end tests for complete user journey scenarios.
Tests realistic user workflows and interactions.
"""

import pytest
import time
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from app.main import app


class TestUserJourney:
    """End-to-end test suite for user journey scenarios."""
    
    @pytest.fixture
    def client(self):
        """Create test client for user journey tests."""
        return TestClient(app)
    
    @pytest.mark.e2e
    def test_first_time_user_journey(self, client):
        """Test complete journey for a first-time user."""
        # Scenario: New user creates their first educational video
        
        # Step 1: User explores API documentation (simulated by checking endpoints)
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        docs_response = client.get("/docs")
        assert docs_response.status_code == 200
        
        # Step 2: User creates their first video with basic settings
        first_video_request = {
            "topic": "Introduction to Programming",
            "description": "A beginner-friendly introduction to programming concepts",
            "voice_settings": {
                "voice": "default",
                "speed": 1.0,
                "pitch": 1.0
            },
            "animation_config": {
                "quality": "medium",
                "fps": 30,
                "resolution": "1080p"
            }
        }
        
        create_response = client.post("/api/v1/video/create", json=first_video_request)
        assert create_response.status_code == 201
        
        create_data = create_response.json()
        first_video_id = create_data["video_id"]
        assert create_data["status"] == "processing"
        
        # Step 3: User checks status multiple times (typical behavior)
        status_checks = 0
        max_status_checks = 30
        
        while status_checks < max_status_checks:
            time.sleep(1)
            status_response = client.get(f"/api/v1/video/{first_video_id}/status")
            assert status_response.status_code == 200
            
            status_data = status_response.json()
            status_checks += 1
            
            if status_data["status"] == "completed":
                break
            elif status_data["status"] == "failed":
                pytest.fail("First user video failed")
        
        assert status_checks < max_status_checks, "First video didn't complete in time"
        
        # Step 4: User downloads their first video
        download_response = client.get(f"/api/v1/video/{first_video_id}/download")
        assert download_response.status_code == 200
        assert download_response.headers["content-type"] == "video/mp4"
        assert len(download_response.content) > 0
        
        # Step 5: Encouraged by success, user creates a second video with custom settings
        second_video_request = {
            "topic": "Advanced Programming Concepts",
            "description": "Diving deeper into object-oriented programming and design patterns",
            "voice_settings": {
                "voice": "female",
                "speed": 1.1,
                "pitch": 0.95
            },
            "animation_config": {
                "quality": "high",
                "fps": 60,
                "resolution": "4K",
                "background_color": "#1E1E1E",
                "text_color": "#00FF00"
            }
        }
        
        second_create_response = client.post("/api/v1/video/create", json=second_video_request)
        assert second_create_response.status_code == 201
        
        second_video_id = second_create_response.json()["video_id"]
        
        # Step 6: User waits for second video (more patient now)
        second_status_checks = 0
        max_second_checks = 45  # Longer timeout for high quality
        
        while second_status_checks < max_second_checks:
            time.sleep(1)
            status_response = client.get(f"/api/v1/video/{second_video_id}/status")
            status_data = status_response.json()
            second_status_checks += 1
            
            if status_data["status"] == "completed":
                # User successfully downloads second video
                download_response = client.get(f"/api/v1/video/{second_video_id}/download")
                assert download_response.status_code == 200
                break
            elif status_data["status"] == "failed":
                pytest.fail("Second user video failed")
        
        assert second_status_checks < max_second_checks, "Second video didn't complete"
        
        # Verify user has successfully completed their journey
        # Both videos should be accessible
        final_first_status = client.get(f"/api/v1/video/{first_video_id}/status")
        final_second_status = client.get(f"/api/v1/video/{second_video_id}/status")
        
        assert final_first_status.json()["status"] == "completed"
        assert final_second_status.json()["status"] == "completed"
    
    @pytest.mark.e2e
    def test_power_user_journey(self, client):
        """Test journey for an experienced power user."""
        # Scenario: Experienced user creates multiple videos with advanced settings
        
        # Step 1: Power user creates multiple videos simultaneously
        video_topics = [
            "Machine Learning Fundamentals",
            "Deep Learning with Neural Networks", 
            "Natural Language Processing Basics",
            "Computer Vision Applications"
        ]
        
        video_ids = []
        
        for i, topic in enumerate(video_topics):
            advanced_request = {
                "topic": topic,
                "description": f"Advanced tutorial on {topic.lower()} with practical examples and code demonstrations",
                "voice_settings": {
                    "voice": "male" if i % 2 == 0 else "female",
                    "speed": 1.0 + (i * 0.1),  # Varying speeds
                    "pitch": 1.0 - (i * 0.05),  # Varying pitches
                    "volume": 0.9 + (i * 0.025)  # Varying volumes
                },
                "animation_config": {
                    "quality": "high",
                    "fps": 60,
                    "resolution": "4K",
                    "background_color": f"#{hex(200 + i * 10)[2:]}0000",  # Varying colors
                    "text_color": "#FFFFFF"
                }
            }
            
            create_response = client.post("/api/v1/video/create", json=advanced_request)
            assert create_response.status_code == 201
            
            video_ids.append(create_response.json()["video_id"])
        
        # Step 2: Power user efficiently monitors all videos
        completed_videos = set()
        max_attempts = 60  # Longer timeout for multiple high-quality videos
        
        for attempt in range(max_attempts):
            time.sleep(1)
            
            for video_id in video_ids:
                if video_id in completed_videos:
                    continue
                
                status_response = client.get(f"/api/v1/video/{video_id}/status")
                status_data = status_response.json()
                
                if status_data["status"] == "completed":
                    completed_videos.add(video_id)
                elif status_data["status"] == "failed":
                    pytest.fail(f"Power user video {video_id} failed")
            
            # Check if all videos completed
            if len(completed_videos) == len(video_ids):
                break
        
        assert len(completed_videos) == len(video_ids), "Not all power user videos completed"
        
        # Step 3: Power user batch downloads all videos
        download_sizes = []
        
        for video_id in video_ids:
            download_response = client.get(f"/api/v1/video/{video_id}/download")
            assert download_response.status_code == 200
            assert download_response.headers["content-type"] == "video/mp4"
            
            video_size = len(download_response.content)
            assert video_size > 0
            download_sizes.append(video_size)
        
        # Verify all downloads were successful and reasonably sized
        assert len(download_sizes) == len(video_ids)
        assert all(size > 1000 for size in download_sizes)  # Minimum size check
    
    @pytest.mark.e2e
    def test_error_recovery_user_journey(self, client):
        """Test user journey when encountering and recovering from errors."""
        # Scenario: User encounters errors but successfully recovers
        
        # Step 1: User makes invalid request (learning experience)
        invalid_request = {
            "topic": "",  # Invalid empty topic
            "description": "Test description"
        }
        
        error_response = client.post("/api/v1/video/create", json=invalid_request)
        assert error_response.status_code == 422
        
        error_data = error_response.json()
        assert "detail" in error_data
        
        # Step 2: User learns from error and makes corrected request
        corrected_request = {
            "topic": "Learning from Mistakes",
            "description": "How to handle and learn from programming errors",
            "voice_settings": {
                "voice": "default",
                "speed": 1.0,
                "pitch": 1.0
            },
            "animation_config": {
                "quality": "medium",
                "fps": 30,
                "resolution": "1080p"
            }
        }
        
        corrected_response = client.post("/api/v1/video/create", json=corrected_request)
        assert corrected_response.status_code == 201
        
        video_id = corrected_response.json()["video_id"]
        
        # Step 3: User waits for video completion
        max_attempts = 30
        video_completed = False
        
        for attempt in range(max_attempts):
            time.sleep(1)
            
            status_response = client.get(f"/api/v1/video/{video_id}/status")
            status_data = status_response.json()
            
            if status_data["status"] == "completed":
                video_completed = True
                break
            elif status_data["status"] == "failed":
                pytest.fail("Corrected video failed")
        
        assert video_completed, "Corrected video didn't complete"
        
        # Step 4: User successfully downloads the corrected video
        download_response = client.get(f"/api/v1/video/{video_id}/download")
        assert download_response.status_code == 200
        assert len(download_response.content) > 0
        
        # Step 5: User tries to access non-existent video (another learning moment)
        nonexistent_response = client.get("/api/v1/video/nonexistent_video_123/status")
        assert nonexistent_response.status_code == 404
        
        # Step 6: User successfully checks their actual video again
        final_status_response = client.get(f"/api/v1/video/{video_id}/status")
        assert final_status_response.status_code == 200
        assert final_status_response.json()["status"] == "completed"
    
    @pytest.mark.e2e
    def test_educational_content_creator_journey(self, client):
        """Test journey for an educational content creator."""
        # Scenario: Teacher creating a series of educational videos
        
        # Step 1: Create a series of related educational videos
        lesson_series = [
            {
                "topic": "Python Basics - Variables and Data Types",
                "description": "Introduction to Python variables, strings, numbers, and basic data types with examples",
                "lesson_number": 1
            },
            {
                "topic": "Python Basics - Control Structures",
                "description": "Understanding if statements, loops, and conditional logic in Python programming",
                "lesson_number": 2
            },
            {
                "topic": "Python Basics - Functions and Modules",
                "description": "Creating and using functions, understanding scope, and working with modules",
                "lesson_number": 3
            }
        ]
        
        lesson_video_ids = []
        
        for lesson in lesson_series:
            educational_request = {
                "topic": lesson["topic"],
                "description": lesson["description"],
                "voice_settings": {
                    "voice": "default",
                    "speed": 0.9,  # Slightly slower for educational content
                    "pitch": 1.0,
                    "volume": 1.0
                },
                "animation_config": {
                    "quality": "high",  # High quality for educational content
                    "fps": 30,
                    "resolution": "1080p",
                    "background_color": "#F0F8FF",  # Light blue background
                    "text_color": "#000080"  # Dark blue text
                }
            }
            
            create_response = client.post("/api/v1/video/create", json=educational_request)
            assert create_response.status_code == 201
            
            lesson_video_ids.append({
                "video_id": create_response.json()["video_id"],
                "lesson_number": lesson["lesson_number"],
                "topic": lesson["topic"]
            })
        
        # Step 2: Monitor progress of all lessons
        completed_lessons = []
        max_attempts = 90  # Longer timeout for educational content
        
        for attempt in range(max_attempts):
            time.sleep(1)
            
            for lesson_info in lesson_video_ids:
                if lesson_info["video_id"] in [l["video_id"] for l in completed_lessons]:
                    continue
                
                status_response = client.get(f"/api/v1/video/{lesson_info['video_id']}/status")
                status_data = status_response.json()
                
                if status_data["status"] == "completed":
                    completed_lessons.append(lesson_info)
                elif status_data["status"] == "failed":
                    pytest.fail(f"Educational video {lesson_info['topic']} failed")
            
            if len(completed_lessons) == len(lesson_video_ids):
                break
        
        assert len(completed_lessons) == len(lesson_video_ids), "Not all educational videos completed"
        
        # Step 3: Verify lesson series is complete and in order
        completed_lessons.sort(key=lambda x: x["lesson_number"])
        
        for i, lesson in enumerate(completed_lessons):
            assert lesson["lesson_number"] == i + 1
            
            # Download each lesson to verify quality
            download_response = client.get(f"/api/v1/video/{lesson['video_id']}/download")
            assert download_response.status_code == 200
            assert len(download_response.content) > 0
        
        # Step 4: Verify all lessons maintain consistent quality
        lesson_statuses = []
        for lesson in completed_lessons:
            status_response = client.get(f"/api/v1/video/{lesson['video_id']}/status")
            lesson_statuses.append(status_response.json())
        
        # All lessons should be completed successfully
        assert all(status["status"] == "completed" for status in lesson_statuses)
        assert all("download_url" in status for status in lesson_statuses)
    
    @pytest.mark.e2e
    def test_api_rate_limiting_user_journey(self, client):
        """Test user journey with API rate limiting considerations."""
        # Scenario: User hits rate limits and learns to work within them
        
        # Step 1: User makes rapid requests (simulating rate limit testing)
        rapid_requests = []
        
        for i in range(5):  # Make several rapid requests
            request = {
                "topic": f"Rapid Request Test {i+1}",
                "description": f"Testing rapid request handling - video {i+1}",
                "voice_settings": {"voice": "default", "speed": 1.0},
                "animation_config": {"quality": "low", "fps": 24}  # Low quality for speed
            }
            
            response = client.post("/api/v1/video/create", json=request)
            rapid_requests.append({
                "request_number": i+1,
                "status_code": response.status_code,
                "response": response.json() if response.status_code == 201 else None
            })
        
        # Step 2: Analyze rate limiting behavior
        successful_requests = [r for r in rapid_requests if r["status_code"] == 201]
        rate_limited_requests = [r for r in rapid_requests if r["status_code"] == 429]
        
        # Should have some successful requests
        assert len(successful_requests) > 0
        
        # Step 3: User waits for successful videos to complete
        if successful_requests:
            video_ids = [r["response"]["video_id"] for r in successful_requests]
            
            completed_count = 0
            max_attempts = 60
            
            for attempt in range(max_attempts):
                time.sleep(1)
                
                for video_id in video_ids:
                    status_response = client.get(f"/api/v1/video/{video_id}/status")
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        if status_data["status"] == "completed":
                            completed_count += 1
                
                if completed_count >= len(video_ids):
                    break
            
            # Verify at least some videos completed successfully
            assert completed_count > 0
        
        # Step 4: User learns to space out requests appropriately
        spaced_request = {
            "topic": "Learning Rate Limits",
            "description": "Understanding how to work with API rate limits effectively",
            "voice_settings": {"voice": "default", "speed": 1.0},
            "animation_config": {"quality": "medium", "fps": 30}
        }
        
        # Wait a bit before making the spaced request
        time.sleep(2)
        
        spaced_response = client.post("/api/v1/video/create", json=spaced_request)
        assert spaced_response.status_code == 201
        
        # Verify the spaced request completes successfully
        spaced_video_id = spaced_response.json()["video_id"]
        
        max_attempts = 30
        for attempt in range(max_attempts):
            time.sleep(1)
            
            status_response = client.get(f"/api/v1/video/{spaced_video_id}/status")
            status_data = status_response.json()
            
            if status_data["status"] == "completed":
                download_response = client.get(f"/api/v1/video/{spaced_video_id}/download")
                assert download_response.status_code == 200
                break
            elif status_data["status"] == "failed":
                pytest.fail("Spaced request video failed")
        
        assert attempt < max_attempts - 1, "Spaced request video didn't complete"
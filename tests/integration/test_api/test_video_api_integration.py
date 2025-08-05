"""
Integration tests for video API endpoints.
Tests complete video generation workflow through API endpoints.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.main import app
from app.models.schemas.video import VideoGenerationRequest, VideoGenerationResponse, VideoStatusResponse
from app.models.enums import VideoStatus
from app.services.video_service import VideoService


class TestVideoAPIIntegration:
    """Integration test suite for video API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client for API integration tests."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_video_service(self):
        """Create mock video service for integration tests."""
        service = Mock(spec=VideoService)
        service.create_video = AsyncMock()
        service.get_video_status = AsyncMock()
        service.download_video = AsyncMock()
        return service
    
    @pytest.mark.integration
    def test_video_creation_workflow_integration(self, client, mock_video_service):
        """Test complete video creation workflow through API."""
        # Mock video service responses
        mock_video_service.create_video.return_value = VideoResponse(
            video_id="integration_test_123",
            status=VideoStatus.PROCESSING,
            download_url=None
        )
        
        mock_video_service.get_video_status.return_value = VideoResponse(
            video_id="integration_test_123",
            status=VideoStatus.COMPLETED,
            download_url="https://example.com/video.mp4"
        )
        
        mock_video_service.download_video.return_value = b"mock_video_content"
        
        with patch('app.api.v1.endpoints.video.get_video_service', return_value=mock_video_service):
            # Step 1: Create video
            create_request = {
                "topic": "Integration Test Video",
                "description": "Testing complete video creation workflow",
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
            
            create_response = client.post("/api/v1/video/create", json=create_request)
            assert create_response.status_code == 201
            
            create_data = create_response.json()
            video_id = create_data["video_id"]
            assert video_id == "integration_test_123"
            assert create_data["status"] == "processing"
            
            # Step 2: Check video status
            status_response = client.get(f"/api/v1/video/{video_id}/status")
            assert status_response.status_code == 200
            
            status_data = status_response.json()
            assert status_data["video_id"] == video_id
            assert status_data["status"] == "completed"
            assert status_data["download_url"] is not None
            
            # Step 3: Download video
            download_response = client.get(f"/api/v1/video/{video_id}/download")
            assert download_response.status_code == 200
            assert download_response.headers["content-type"] == "video/mp4"
            assert download_response.content == b"mock_video_content"
            
            # Verify service calls
            mock_video_service.create_video.assert_called_once()
            mock_video_service.get_video_status.assert_called_once_with(video_id)
            mock_video_service.download_video.assert_called_once_with(video_id)
    
    @pytest.mark.integration
    def test_video_api_error_handling_integration(self, client, mock_video_service):
        """Test API error handling integration."""
        # Mock service errors
        mock_video_service.create_video.side_effect = Exception("Service unavailable")
        mock_video_service.get_video_status.side_effect = ValueError("Video not found")
        
        with patch('app.api.v1.endpoints.video.get_video_service', return_value=mock_video_service):
            # Test creation error
            create_request = {
                "topic": "Error Test Video",
                "description": "Testing error handling"
            }
            
            create_response = client.post("/api/v1/video/create", json=create_request)
            assert create_response.status_code == 500
            
            # Test status error
            status_response = client.get("/api/v1/video/nonexistent_video/status")
            assert status_response.status_code == 404
    
    @pytest.mark.integration
    def test_video_api_validation_integration(self, client):
        """Test API request validation integration."""
        # Test missing required fields
        invalid_request = {
            "description": "Missing topic field"
        }
        
        response = client.post("/api/v1/video/create", json=invalid_request)
        assert response.status_code == 422
        
        error_data = response.json()
        assert "detail" in error_data
        assert any("topic" in str(error) for error in error_data["detail"])
    
    @pytest.mark.integration
    def test_video_api_concurrent_requests_integration(self, client, mock_video_service):
        """Test handling of concurrent API requests."""
        # Mock service to handle concurrent requests
        video_responses = [
            VideoResponse(video_id=f"concurrent_test_{i}", status=VideoStatus.PROCESSING)
            for i in range(3)
        ]
        mock_video_service.create_video.side_effect = video_responses
        
        with patch('app.api.v1.endpoints.video.get_video_service', return_value=mock_video_service):
            # Send concurrent requests
            requests = [
                {
                    "topic": f"Concurrent Test Video {i}",
                    "description": f"Testing concurrent request {i}"
                }
                for i in range(3)
            ]
            
            responses = []
            for request in requests:
                response = client.post("/api/v1/video/create", json=request)
                responses.append(response)
            
            # Verify all requests succeeded
            for i, response in enumerate(responses):
                assert response.status_code == 201
                data = response.json()
                assert data["video_id"] == f"concurrent_test_{i}"
            
            # Verify service was called for each request
            assert mock_video_service.create_video.call_count == 3
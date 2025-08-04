"""
Unit tests for video generation API endpoints.
Tests video creation, status checking, and download functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException

from app.api.v1.endpoints.video import router
from app.models.schemas.video import VideoRequest, VideoResponse, VideoStatus
from app.services.video_service import VideoService


class TestVideoEndpoints:
    """Test suite for video generation endpoints."""
    
    @pytest.fixture
    def mock_video_service(self):
        """Create mock video service."""
        service = Mock(spec=VideoService)
        service.create_video = AsyncMock()
        service.get_video_status = AsyncMock()
        service.download_video = AsyncMock()
        return service
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)
    
    def test_create_video_success(self, client, mock_video_service):
        """Test successful video creation."""
        request_data = {
            "topic": "Python basics",
            "description": "Introduction to Python programming",
            "voice_settings": {"voice": "default", "speed": 1.0},
            "animation_config": {"quality": "medium", "fps": 30}
        }
        
        mock_response = VideoResponse(
            video_id="test_video_123",
            status=VideoStatus.PROCESSING,
            download_url=None
        )
        mock_video_service.create_video.return_value = mock_response
        
        with patch('app.api.v1.endpoints.video.get_video_service', return_value=mock_video_service):
            response = client.post("/video/create", json=request_data)
            
            assert response.status_code == 201
            data = response.json()
            assert data["video_id"] == "test_video_123"
            assert data["status"] == "processing"
    
    def test_create_video_validation_error(self, client):
        """Test video creation with invalid data."""
        invalid_data = {
            "topic": "",  # Empty topic should fail validation
            "description": "Test description"
        }
        
        response = client.post("/video/create", json=invalid_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_get_video_status_success(self, client, mock_video_service):
        """Test successful video status retrieval."""
        video_id = "test_video_123"
        mock_response = VideoResponse(
            video_id=video_id,
            status=VideoStatus.COMPLETED,
            download_url="https://example.com/video.mp4"
        )
        mock_video_service.get_video_status.return_value = mock_response
        
        with patch('app.api.v1.endpoints.video.get_video_service', return_value=mock_video_service):
            response = client.get(f"/video/{video_id}/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["video_id"] == video_id
            assert data["status"] == "completed"
            assert data["download_url"] is not None
    
    def test_get_video_status_not_found(self, client, mock_video_service):
        """Test video status retrieval for non-existent video."""
        video_id = "nonexistent_video"
        mock_video_service.get_video_status.side_effect = HTTPException(status_code=404)
        
        with patch('app.api.v1.endpoints.video.get_video_service', return_value=mock_video_service):
            response = client.get(f"/video/{video_id}/status")
            
            assert response.status_code == 404
    
    def test_download_video_success(self, client, mock_video_service):
        """Test successful video download."""
        video_id = "test_video_123"
        mock_video_service.download_video.return_value = b"video_content"
        
        with patch('app.api.v1.endpoints.video.get_video_service', return_value=mock_video_service):
            response = client.get(f"/video/{video_id}/download")
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "video/mp4"
            assert response.content == b"video_content"
    
    def test_download_video_not_ready(self, client, mock_video_service):
        """Test video download when video is not ready."""
        video_id = "processing_video"
        mock_video_service.download_video.side_effect = HTTPException(
            status_code=400, detail="Video not ready for download"
        )
        
        with patch('app.api.v1.endpoints.video.get_video_service', return_value=mock_video_service):
            response = client.get(f"/video/{video_id}/download")
            
            assert response.status_code == 400
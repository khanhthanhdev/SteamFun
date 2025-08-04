"""
Integration tests for AWS API endpoints.
Tests complete AWS integration workflow through API endpoints.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas.aws import S3UploadRequest, S3UploadResponse, S3DownloadResponse
from app.services.aws_service import AWSService


class TestAWSAPIIntegration:
    """Integration test suite for AWS API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client for API integration tests."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_aws_service(self):
        """Create mock AWS service for integration tests."""
        service = Mock(spec=AWSService)
        service.upload_to_s3 = AsyncMock()
        service.download_from_s3 = AsyncMock()
        service.get_metadata = AsyncMock()
        service.store_metadata = AsyncMock()
        return service
    
    @pytest.mark.integration
    def test_s3_upload_workflow_integration(self, client, mock_aws_service):
        """Test complete S3 upload workflow through API."""
        # Mock AWS service responses
        mock_aws_service.upload_to_s3.return_value = S3UploadResponse(
            success=True,
            object_key="videos/integration_test.mp4",
            download_url="https://s3.amazonaws.com/bucket/videos/integration_test.mp4",
            metadata={"video_id": "test_123", "size": 1024000}
        )
        
        with patch('app.api.v1.endpoints.aws.get_aws_service', return_value=mock_aws_service):
            # Test S3 upload
            upload_request = {
                "file_path": "/path/to/test_video.mp4",
                "bucket_name": "test-bucket",
                "object_key": "videos/integration_test.mp4",
                "metadata": {
                    "video_id": "test_123",
                    "quality": "high"
                }
            }
            
            response = client.post("/api/v1/aws/s3/upload", json=upload_request)
            assert response.status_code == 201
            
            data = response.json()
            assert data["success"] is True
            assert data["object_key"] == "videos/integration_test.mp4"
            assert data["download_url"] is not None
            
            # Verify service was called correctly
            mock_aws_service.upload_to_s3.assert_called_once()
    
    @pytest.mark.integration
    def test_s3_download_workflow_integration(self, client, mock_aws_service):
        """Test complete S3 download workflow through API."""
        # Mock download response
        mock_aws_service.download_from_s3.return_value = S3DownloadResponse(
            success=True,
            content=b"mock_video_content",
            content_type="video/mp4",
            metadata={"video_id": "test_123"}
        )
        
        with patch('app.api.v1.endpoints.aws.get_aws_service', return_value=mock_aws_service):
            # Test S3 download
            response = client.get("/api/v1/aws/s3/download/test-bucket/videos/test_video.mp4")
            assert response.status_code == 200
            assert response.headers["content-type"] == "video/mp4"
            assert response.content == b"mock_video_content"
            
            # Verify service was called
            mock_aws_service.download_from_s3.assert_called_once()
    
    @pytest.mark.integration
    def test_dynamodb_metadata_integration(self, client, mock_aws_service):
        """Test DynamoDB metadata operations through API."""
        # Mock metadata operations
        mock_aws_service.store_metadata.return_value = {
            "success": True,
            "item_id": "metadata_123"
        }
        
        mock_aws_service.get_metadata.return_value = {
            "video_id": "test_123",
            "title": "Integration Test Video",
            "status": "completed",
            "file_size": 1024000
        }
        
        with patch('app.api.v1.endpoints.aws.get_aws_service', return_value=mock_aws_service):
            # Test metadata storage
            metadata_request = {
                "table_name": "video-metadata",
                "item": {
                    "video_id": "test_123",
                    "title": "Integration Test Video",
                    "status": "completed"
                }
            }
            
            store_response = client.post("/api/v1/aws/dynamodb/store", json=metadata_request)
            assert store_response.status_code == 201
            
            store_data = store_response.json()
            assert store_data["success"] is True
            
            # Test metadata retrieval
            get_response = client.get("/api/v1/aws/dynamodb/video-metadata/test_123")
            assert get_response.status_code == 200
            
            get_data = get_response.json()
            assert get_data["video_id"] == "test_123"
            assert get_data["title"] == "Integration Test Video"
            
            # Verify service calls
            mock_aws_service.store_metadata.assert_called_once()
            mock_aws_service.get_metadata.assert_called_once()
    
    @pytest.mark.integration
    def test_aws_api_error_handling_integration(self, client, mock_aws_service):
        """Test AWS API error handling integration."""
        # Mock service errors
        mock_aws_service.upload_to_s3.side_effect = Exception("S3 upload failed")
        
        with patch('app.api.v1.endpoints.aws.get_aws_service', return_value=mock_aws_service):
            upload_request = {
                "file_path": "/invalid/path.mp4",
                "bucket_name": "test-bucket",
                "object_key": "test.mp4"
            }
            
            response = client.post("/api/v1/aws/s3/upload", json=upload_request)
            assert response.status_code == 500
    
    @pytest.mark.integration
    def test_aws_api_validation_integration(self, client):
        """Test AWS API request validation integration."""
        # Test missing required fields
        invalid_request = {
            "bucket_name": "test-bucket"
            # Missing file_path and object_key
        }
        
        response = client.post("/api/v1/aws/s3/upload", json=invalid_request)
        assert response.status_code == 422
        
        error_data = response.json()
        assert "detail" in error_data
    
    @pytest.mark.integration
    def test_aws_multipart_upload_integration(self, client, mock_aws_service):
        """Test AWS multipart upload workflow."""
        # Mock multipart upload responses
        mock_aws_service.initiate_multipart_upload = AsyncMock(return_value={
            "upload_id": "multipart_123",
            "bucket": "test-bucket",
            "key": "large_video.mp4"
        })
        
        mock_aws_service.complete_multipart_upload = AsyncMock(return_value={
            "success": True,
            "download_url": "https://s3.amazonaws.com/bucket/large_video.mp4"
        })
        
        with patch('app.api.v1.endpoints.aws.get_aws_service', return_value=mock_aws_service):
            # Initiate multipart upload
            initiate_request = {
                "bucket_name": "test-bucket",
                "object_key": "large_video.mp4",
                "content_type": "video/mp4"
            }
            
            initiate_response = client.post("/api/v1/aws/s3/multipart/initiate", json=initiate_request)
            assert initiate_response.status_code == 201
            
            initiate_data = initiate_response.json()
            upload_id = initiate_data["upload_id"]
            
            # Complete multipart upload
            complete_request = {
                "upload_id": upload_id,
                "parts": [
                    {"part_number": 1, "etag": "etag1"},
                    {"part_number": 2, "etag": "etag2"}
                ]
            }
            
            complete_response = client.post(f"/api/v1/aws/s3/multipart/{upload_id}/complete", json=complete_request)
            assert complete_response.status_code == 200
            
            complete_data = complete_response.json()
            assert complete_data["success"] is True
            
            # Verify service calls
            mock_aws_service.initiate_multipart_upload.assert_called_once()
            mock_aws_service.complete_multipart_upload.assert_called_once()
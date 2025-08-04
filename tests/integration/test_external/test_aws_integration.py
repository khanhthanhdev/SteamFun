"""
Integration tests for AWS service integrations.
Tests S3, DynamoDB, and other AWS service integrations.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import boto3
from moto import mock_s3, mock_dynamodb

from app.services.aws_service import AWSService
from app.models.schemas.aws import S3UploadRequest, S3UploadResponse


class TestAWSIntegration:
    """Integration test suite for AWS service integrations."""
    
    @pytest.fixture
    def aws_service(self):
        """Create AWSService instance for integration tests."""
        return AWSService()
    
    @pytest.mark.integration
    @mock_s3
    def test_s3_upload_integration(self, aws_service):
        """Test S3 upload integration with mocked AWS."""
        # Create mock S3 bucket
        s3_client = boto3.client('s3', region_name='us-east-1')
        bucket_name = 'test-video-bucket'
        s3_client.create_bucket(Bucket=bucket_name)
        
        # Test S3 upload
        upload_request = S3UploadRequest(
            file_path="/path/to/test_video.mp4",
            bucket_name=bucket_name,
            object_key="videos/test_video.mp4",
            metadata={"video_id": "test_123", "quality": "high"}
        )
        
        with patch.object(aws_service, '_get_s3_client', return_value=s3_client):
            with patch('builtins.open', mock_open(read_data=b"mock_video_data")):
                response = aws_service.upload_to_s3(upload_request)
                
                assert response.success is True
                assert response.object_key == "videos/test_video.mp4"
                assert response.download_url is not None
                
                # Verify file was uploaded
                objects = s3_client.list_objects_v2(Bucket=bucket_name)
                assert objects['KeyCount'] == 1
                assert objects['Contents'][0]['Key'] == "videos/test_video.mp4"
    
    @pytest.mark.integration
    @mock_dynamodb
    def test_dynamodb_integration(self, aws_service):
        """Test DynamoDB integration with mocked AWS."""
        # Create mock DynamoDB table
        dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        table_name = 'video-metadata'
        
        table = dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                {'AttributeName': 'video_id', 'KeyType': 'HASH'}
            ],
            AttributeDefinitions=[
                {'AttributeName': 'video_id', 'AttributeType': 'S'}
            ],
            BillingMode='PAY_PER_REQUEST'
        )
        
        # Test DynamoDB operations
        video_metadata = {
            'video_id': 'test_video_123',
            'title': 'Integration Test Video',
            'status': 'completed',
            'file_size': 1024000,
            'duration': 120.5
        }
        
        with patch.object(aws_service, '_get_dynamodb_table', return_value=table):
            # Test put item
            aws_service.store_video_metadata(video_metadata)
            
            # Test get item
            retrieved_metadata = aws_service.get_video_metadata('test_video_123')
            
            assert retrieved_metadata['video_id'] == 'test_video_123'
            assert retrieved_metadata['title'] == 'Integration Test Video'
            assert retrieved_metadata['status'] == 'completed'
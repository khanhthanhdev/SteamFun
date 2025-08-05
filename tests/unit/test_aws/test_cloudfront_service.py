"""
Unit tests for CloudFront service.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from app.core.aws.cloudfront_service import CloudFrontService
from app.core.aws.config import AWSConfig
from app.core.aws.exceptions import AWSIntegrationError


@pytest.fixture
def aws_config():
    """Create AWS configuration for testing."""
    return AWSConfig(
        region='us-east-1',
        video_bucket_name='test-video-bucket',
        code_bucket_name='test-code-bucket',
        cloudfront_distribution_id='E1234567890ABC',
        cloudfront_domain='d1234567890abc.cloudfront.net',
        enable_cloudfront=True
    )


@pytest.fixture
def mock_cloudfront_client():
    """Create mock CloudFront client."""
    client = Mock()
    client.create_distribution = Mock()
    client.create_origin_access_identity = Mock()
    client.get_distribution = Mock()
    client.get_distribution_config = Mock()
    client.update_distribution = Mock()
    client.create_invalidation = Mock()
    client.get_invalidation = Mock()
    client.list_invalidations = Mock()
    client.create_response_headers_policy = Mock()
    return client


@pytest.fixture
def mock_cloudwatch_client():
    """Create mock CloudWatch client."""
    client = Mock()
    client.get_metric_statistics = Mock()
    return client


@pytest.fixture
def cloudfront_service(aws_config, mock_cloudfront_client, mock_cloudwatch_client):
    """Create CloudFront service with mocked clients."""
    with patch('boto3.client') as mock_boto3_client:
        def client_side_effect(service_name, **kwargs):
            if service_name == 'cloudfront':
                return mock_cloudfront_client
            elif service_name == 'cloudwatch':
                return mock_cloudwatch_client
            return Mock()
        
        mock_boto3_client.side_effect = client_side_effect
        
        service = CloudFrontService(aws_config)
        return service


class TestCloudFrontService:
    """Test cases for CloudFront service."""
    
    def test_init_success(self, aws_config):
        """Test successful CloudFront service initialization."""
        with patch('boto3.client') as mock_boto3_client:
            mock_boto3_client.return_value = Mock()
            
            service = CloudFrontService(aws_config)
            
            assert service.config == aws_config
            assert service.cloudfront_client is not None
            assert service.cloudwatch_client is not None
    
    def test_init_failure(self, aws_config):
        """Test CloudFront service initialization failure."""
        with patch('boto3.client') as mock_boto3_client:
            mock_boto3_client.side_effect = Exception("AWS credentials not found")
            
            with pytest.raises(AWSIntegrationError):
                CloudFrontService(aws_config)
    
    @pytest.mark.asyncio
    async def test_create_distribution_success(self, cloudfront_service, mock_cloudfront_client):
        """Test successful CloudFront distribution creation."""
        # Mock OAI creation
        mock_cloudfront_client.create_origin_access_identity.return_value = {
            'OriginAccessIdentity': {
                'Id': 'E1234567890ABC',
                'S3CanonicalUserId': 'canonical-user-id'
            },
            'ETag': 'etag-value'
        }
        
        # Mock distribution creation
        mock_cloudfront_client.create_distribution.return_value = {
            'Distribution': {
                'Id': 'E1234567890ABC',
                'DomainName': 'd1234567890abc.cloudfront.net',
                'Status': 'InProgress'
            },
            'ETag': 'etag-value'
        }
        
        result = await cloudfront_service.create_distribution('test-video-bucket')
        
        assert result['distribution_id'] == 'E1234567890ABC'
        assert result['domain_name'] == 'd1234567890abc.cloudfront.net'
        assert result['status'] == 'InProgress'
        assert 'origin_access_identity_id' in result
        
        # Verify OAI creation was called
        mock_cloudfront_client.create_origin_access_identity.assert_called_once()
        
        # Verify distribution creation was called
        mock_cloudfront_client.create_distribution.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_invalidate_cache_success(self, cloudfront_service, mock_cloudfront_client):
        """Test successful cache invalidation."""
        mock_cloudfront_client.create_invalidation.return_value = {
            'Invalidation': {
                'Id': 'I1234567890ABC',
                'Status': 'InProgress',
                'CreateTime': datetime.utcnow(),
                'InvalidationBatch': {
                    'CallerReference': 'test-ref',
                    'Paths': {
                        'Items': ['/videos/test.mp4']
                    }
                }
            }
        }
        
        result = await cloudfront_service.invalidate_cache('/videos/test.mp4')
        
        assert result['invalidation_id'] == 'I1234567890ABC'
        assert result['status'] == 'InProgress'
        assert result['paths'] == ['/videos/test.mp4']
        
        mock_cloudfront_client.create_invalidation.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_invalidate_cache_multiple_paths(self, cloudfront_service, mock_cloudfront_client):
        """Test cache invalidation with multiple paths."""
        paths = ['/videos/test1.mp4', '/videos/test2.mp4']
        
        mock_cloudfront_client.create_invalidation.return_value = {
            'Invalidation': {
                'Id': 'I1234567890ABC',
                'Status': 'InProgress',
                'CreateTime': datetime.utcnow(),
                'InvalidationBatch': {
                    'CallerReference': 'test-ref',
                    'Paths': {
                        'Items': paths
                    }
                }
            }
        }
        
        result = await cloudfront_service.invalidate_cache(paths)
        
        assert result['paths'] == paths
        mock_cloudfront_client.create_invalidation.assert_called_once()
    
    def test_generate_video_url_with_domain(self, cloudfront_service):
        """Test video URL generation with provided domain."""
        url = cloudfront_service.generate_video_url(
            'videos/test.mp4', 
            domain_name='custom.cloudfront.net'
        )
        
        assert url == 'https://custom.cloudfront.net/videos/test.mp4'
    
    def test_generate_video_url_with_config_domain(self, cloudfront_service):
        """Test video URL generation with config domain."""
        url = cloudfront_service.generate_video_url('videos/test.mp4')
        
        assert url == 'https://d1234567890abc.cloudfront.net/videos/test.mp4'
    
    def test_generate_video_url_no_domain(self, aws_config):
        """Test video URL generation failure when no domain available."""
        # Create config without CloudFront domain
        config = AWSConfig(
            region='us-east-1',
            video_bucket_name='test-video-bucket',
            enable_cloudfront=True
        )
        
        with patch('boto3.client') as mock_boto3_client:
            mock_boto3_client.return_value = Mock()
            service = CloudFrontService(config)
            
            with pytest.raises(AWSIntegrationError):
                service.generate_video_url('videos/test.mp4')
    
    def test_get_distribution_info_success(self, cloudfront_service, mock_cloudfront_client):
        """Test successful distribution info retrieval."""
        mock_cloudfront_client.get_distribution.return_value = {
            'Distribution': {
                'Id': 'E1234567890ABC',
                'ARN': 'arn:aws:cloudfront::123456789012:distribution/E1234567890ABC',
                'Status': 'Deployed',
                'DomainName': 'd1234567890abc.cloudfront.net',
                'LastModifiedTime': datetime.utcnow(),
                'DistributionConfig': {
                    'Enabled': True,
                    'PriceClass': 'PriceClass_100',
                    'Origins': {
                        'Items': [
                            {
                                'Id': 'test-origin',
                                'DomainName': 'test-bucket.s3.amazonaws.com'
                            }
                        ]
                    }
                }
            }
        }
        
        result = cloudfront_service.get_distribution_info()
        
        assert result['id'] == 'E1234567890ABC'
        assert result['status'] == 'Deployed'
        assert result['domain_name'] == 'd1234567890abc.cloudfront.net'
        assert result['enabled'] is True
        assert len(result['origins']) == 1
        
        mock_cloudfront_client.get_distribution.assert_called_once_with(Id='E1234567890ABC')
    
    @pytest.mark.asyncio
    async def test_get_cache_hit_rate_success(self, cloudfront_service, mock_cloudwatch_client):
        """Test successful cache hit rate retrieval."""
        mock_cloudwatch_client.get_metric_statistics.return_value = {
            'Datapoints': [
                {
                    'Timestamp': datetime.utcnow() - timedelta(hours=2),
                    'Average': 85.5,
                    'Maximum': 90.0,
                    'Minimum': 80.0
                },
                {
                    'Timestamp': datetime.utcnow() - timedelta(hours=1),
                    'Average': 87.2,
                    'Maximum': 92.0,
                    'Minimum': 82.0
                }
            ]
        }
        
        result = await cloudfront_service.get_cache_hit_rate()
        
        assert result['distribution_id'] == 'E1234567890ABC'
        assert result['average_hit_rate'] == 86.35  # Average of 85.5 and 87.2
        assert result['maximum_hit_rate'] == 92.0
        assert result['minimum_hit_rate'] == 80.0
        assert len(result['datapoints']) == 2
        
        mock_cloudwatch_client.get_metric_statistics.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_performance_metrics_success(self, cloudfront_service, mock_cloudwatch_client):
        """Test successful performance metrics retrieval."""
        # Mock different responses for different metrics
        def mock_get_metric_statistics(**kwargs):
            metric_name = kwargs['MetricName']
            if metric_name == 'Requests':
                return {
                    'Datapoints': [
                        {
                            'Timestamp': datetime.utcnow(),
                            'Sum': 1000,
                            'Average': 100,
                            'Maximum': 200
                        }
                    ]
                }
            elif metric_name == 'CacheHitRate':
                return {
                    'Datapoints': [
                        {
                            'Timestamp': datetime.utcnow(),
                            'Average': 85.0,
                            'Maximum': 90.0
                        }
                    ]
                }
            else:
                return {'Datapoints': []}
        
        mock_cloudwatch_client.get_metric_statistics.side_effect = mock_get_metric_statistics
        
        result = await cloudfront_service.get_performance_metrics()
        
        assert result['distribution_id'] == 'E1234567890ABC'
        assert 'metrics' in result
        assert 'Requests' in result['metrics']
        assert result['metrics']['Requests']['total'] == 1000
        
        # Verify multiple metric calls were made
        assert mock_cloudwatch_client.get_metric_statistics.call_count > 1
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, cloudfront_service, mock_cloudfront_client):
        """Test successful health check."""
        mock_cloudfront_client.get_distribution.return_value = {
            'Distribution': {
                'Id': 'E1234567890ABC',
                'Status': 'Deployed',
                'DomainName': 'd1234567890abc.cloudfront.net',
                'DistributionConfig': {
                    'Enabled': True
                }
            }
        }
        
        # Mock cache hit rate call
        with patch.object(cloudfront_service, 'get_cache_hit_rate') as mock_hit_rate:
            mock_hit_rate.return_value = {'average_hit_rate': 85.0}
            
            result = await cloudfront_service.health_check()
            
            assert result['service'] == 'CloudFront'
            assert result['status'] == 'healthy'
            assert 'distribution_status' in result['checks']
            assert 'cache_performance' in result['checks']
    
    def test_repr(self, cloudfront_service):
        """Test string representation of CloudFront service."""
        repr_str = repr(cloudfront_service)
        
        assert 'CloudFrontService' in repr_str
        assert 'us-east-1' in repr_str
        assert 'E1234567890ABC' in repr_str
        assert 'd1234567890abc.cloudfront.net' in repr_str
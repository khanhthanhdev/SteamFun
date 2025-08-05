"""
CloudFront CDN Service

Handles CloudFront distribution configuration, cache management, and invalidation
for video content delivery optimization.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from .config import AWSConfig
from .credentials import AWSCredentialsManager
from .exceptions import AWSIntegrationError

logger = logging.getLogger(__name__)


class CloudFrontService:
    """
    CloudFront CDN service for video content delivery optimization.
    
    Provides functionality for:
    - CloudFront distribution configuration with S3 origin and OAI
    - Cache management and invalidation
    - URL generation for video access
    - Performance monitoring and metrics collection
    """
    
    def __init__(self, config: AWSConfig, credentials_manager: Optional[AWSCredentialsManager] = None):
        """
        Initialize CloudFront service.
        
        Args:
            config: AWS configuration
            credentials_manager: Optional credentials manager
            
        Raises:
            AWSIntegrationError: If initialization fails
        """
        self.config = config
        self.credentials_manager = credentials_manager or AWSCredentialsManager(config)
        
        try:
            # Initialize CloudFront client
            boto3_config = config.get_boto3_config()
            self.cloudfront_client = boto3.client('cloudfront', **boto3_config)
            
            # Initialize CloudWatch client for metrics
            self.cloudwatch_client = boto3.client('cloudwatch', **boto3_config)
            
            logger.info(f"CloudFront service initialized for region: {config.region}")
            if config.cloudfront_distribution_id:
                logger.info(f"Using existing distribution: {config.cloudfront_distribution_id}")
            
        except (NoCredentialsError, ClientError) as e:
            logger.error(f"Failed to initialize CloudFront service: {e}")
            raise AWSIntegrationError(f"CloudFront service initialization failed: {str(e)}") from e
    
    async def create_distribution(self, video_bucket_name: str, 
                                 origin_access_identity_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create CloudFront distribution with S3 origin and OAI.
        
        Args:
            video_bucket_name: S3 bucket name for video content
            origin_access_identity_id: Optional existing OAI ID
            
        Returns:
            Dictionary with distribution information
            
        Raises:
            AWSIntegrationError: If distribution creation fails
        """
        try:
            # Create Origin Access Identity if not provided
            if not origin_access_identity_id:
                oai_response = await self._create_origin_access_identity()
                origin_access_identity_id = oai_response['Id']
                logger.info(f"Created new OAI: {origin_access_identity_id}")
            
            # Distribution configuration optimized for video content
            distribution_config = {
                'CallerReference': f"video-cdn-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
                'Comment': 'CloudFront distribution for AI-generated video content',
                'DefaultRootObject': '',
                'Origins': {
                    'Quantity': 1,
                    'Items': [
                        {
                            'Id': f"{video_bucket_name}-origin",
                            'DomainName': f"{video_bucket_name}.s3.amazonaws.com",
                            'S3OriginConfig': {
                                'OriginAccessIdentity': f"origin-access-identity/cloudfront/{origin_access_identity_id}"
                            }
                        }
                    ]
                },
                'DefaultCacheBehavior': {
                    'TargetOriginId': f"{video_bucket_name}-origin",
                    'ViewerProtocolPolicy': 'redirect-to-https',
                    'TrustedSigners': {
                        'Enabled': False,
                        'Quantity': 0
                    },
                    'ForwardedValues': {
                        'QueryString': True,
                        'Cookies': {'Forward': 'none'},
                        'Headers': {
                            'Quantity': 3,
                            'Items': ['Origin', 'Access-Control-Request-Method', 'Access-Control-Request-Headers']
                        }
                    },
                    'MinTTL': 0,
                    'DefaultTTL': 86400,  # 24 hours
                    'MaxTTL': 31536000,   # 1 year
                    'Compress': True,
                    'AllowedMethods': {
                        'Quantity': 7,
                        'Items': ['GET', 'HEAD', 'OPTIONS', 'PUT', 'POST', 'PATCH', 'DELETE'],
                        'CachedMethods': {
                            'Quantity': 2,
                            'Items': ['GET', 'HEAD']
                        }
                    }
                },
                'CacheBehaviors': {
                    'Quantity': 2,
                    'Items': [
                        # Video content caching behavior
                        {
                            'PathPattern': '*.mp4',
                            'TargetOriginId': f"{video_bucket_name}-origin",
                            'ViewerProtocolPolicy': 'redirect-to-https',
                            'TrustedSigners': {
                                'Enabled': False,
                                'Quantity': 0
                            },
                            'ForwardedValues': {
                                'QueryString': False,
                                'Cookies': {'Forward': 'none'}
                            },
                            'MinTTL': 0,
                            'DefaultTTL': 604800,  # 7 days for videos
                            'MaxTTL': 31536000,    # 1 year
                            'Compress': False      # Don't compress videos
                        },
                        # HLS/DASH streaming content
                        {
                            'PathPattern': 'transcoded/*',
                            'TargetOriginId': f"{video_bucket_name}-origin",
                            'ViewerProtocolPolicy': 'redirect-to-https',
                            'TrustedSigners': {
                                'Enabled': False,
                                'Quantity': 0
                            },
                            'ForwardedValues': {
                                'QueryString': True,
                                'Cookies': {'Forward': 'none'}
                            },
                            'MinTTL': 0,
                            'DefaultTTL': 3600,    # 1 hour for streaming segments
                            'MaxTTL': 86400,       # 24 hours
                            'Compress': False
                        }
                    ]
                },
                'Enabled': True,
                'PriceClass': 'PriceClass_100',  # Use only North America and Europe edge locations
                'ViewerCertificate': {
                    'CloudFrontDefaultCertificate': True
                },
                'CustomErrorResponses': {
                    'Quantity': 2,
                    'Items': [
                        {
                            'ErrorCode': 403,
                            'ResponsePagePath': '/error.html',
                            'ResponseCode': '404',
                            'ErrorCachingMinTTL': 300
                        },
                        {
                            'ErrorCode': 404,
                            'ResponsePagePath': '/error.html',
                            'ResponseCode': '404',
                            'ErrorCachingMinTTL': 300
                        }
                    ]
                },
                'HttpVersion': 'http2',
                'IsIPV6Enabled': True,
                'WebACLId': ''  # Can be configured for additional security
            }
            
            # Create distribution
            response = self.cloudfront_client.create_distribution(
                DistributionConfig=distribution_config
            )
            
            distribution = response['Distribution']
            distribution_id = distribution['Id']
            domain_name = distribution['DomainName']
            
            logger.info(f"Created CloudFront distribution: {distribution_id}")
            logger.info(f"Distribution domain: {domain_name}")
            
            return {
                'distribution_id': distribution_id,
                'domain_name': domain_name,
                'origin_access_identity_id': origin_access_identity_id,
                'status': distribution['Status'],
                'etag': response['ETag']
            }
            
        except ClientError as e:
            logger.error(f"Failed to create CloudFront distribution: {e}")
            raise AWSIntegrationError(f"Distribution creation failed: {str(e)}") from e
    
    async def _create_origin_access_identity(self) -> Dict[str, Any]:
        """
        Create Origin Access Identity for S3 bucket access.
        
        Returns:
            Dictionary with OAI information
            
        Raises:
            AWSIntegrationError: If OAI creation fails
        """
        try:
            oai_config = {
                'CallerReference': f"video-oai-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
                'Comment': 'Origin Access Identity for AI video content'
            }
            
            response = self.cloudfront_client.create_origin_access_identity(
                OriginAccessIdentityConfig=oai_config
            )
            
            oai = response['OriginAccessIdentity']
            logger.info(f"Created Origin Access Identity: {oai['Id']}")
            
            return {
                'Id': oai['Id'],
                'S3CanonicalUserId': oai['S3CanonicalUserId'],
                'ETag': response['ETag']
            }
            
        except ClientError as e:
            logger.error(f"Failed to create Origin Access Identity: {e}")
            raise AWSIntegrationError(f"OAI creation failed: {str(e)}") from e
    
    async def configure_security_headers(self, distribution_id: str) -> bool:
        """
        Configure security headers for CloudFront distribution.
        
        Args:
            distribution_id: CloudFront distribution ID
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            AWSIntegrationError: If configuration fails
        """
        try:
            # Get current distribution configuration
            response = self.cloudfront_client.get_distribution_config(Id=distribution_id)
            config = response['DistributionConfig']
            etag = response['ETag']
            
            # Add security headers using response headers policy
            security_headers_policy = {
                'Name': f'SecurityHeaders-{distribution_id}',
                'Comment': 'Security headers for video content',
                'SecurityHeadersConfig': {
                    'StrictTransportSecurity': {
                        'AccessControlMaxAgeSec': 31536000,  # 1 year
                        'IncludeSubdomains': True,
                        'Override': True
                    },
                    'ContentTypeOptions': {
                        'Override': True
                    },
                    'FrameOptions': {
                        'FrameOption': 'DENY',
                        'Override': True
                    },
                    'ReferrerPolicy': {
                        'ReferrerPolicy': 'strict-origin-when-cross-origin',
                        'Override': True
                    }
                },
                'CorsConfig': {
                    'AccessControlAllowOrigins': {
                        'Quantity': 1,
                        'Items': ['*']  # Configure based on your domain requirements
                    },
                    'AccessControlAllowHeaders': {
                        'Quantity': 4,
                        'Items': ['Origin', 'Content-Type', 'Accept', 'Authorization']
                    },
                    'AccessControlAllowMethods': {
                        'Quantity': 3,
                        'Items': ['GET', 'HEAD', 'OPTIONS']
                    },
                    'AccessControlMaxAgeSec': 86400,  # 24 hours
                    'AccessControlAllowCredentials': False,
                    'OriginOverride': True
                }
            }
            
            # Create response headers policy
            try:
                policy_response = self.cloudfront_client.create_response_headers_policy(
                    ResponseHeadersPolicyConfig=security_headers_policy
                )
                policy_id = policy_response['ResponseHeadersPolicy']['Id']
                logger.info(f"Created security headers policy: {policy_id}")
                
                # Update distribution to use the policy
                config['DefaultCacheBehavior']['ResponseHeadersPolicyId'] = policy_id
                
                # Update distribution
                self.cloudfront_client.update_distribution(
                    Id=distribution_id,
                    DistributionConfig=config,
                    IfMatch=etag
                )
                
                logger.info(f"Applied security headers to distribution: {distribution_id}")
                return True
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResponseHeadersPolicyAlreadyExists':
                    logger.warning("Security headers policy already exists")
                    return True
                else:
                    raise
            
        except ClientError as e:
            logger.error(f"Failed to configure security headers: {e}")
            raise AWSIntegrationError(f"Security headers configuration failed: {str(e)}") from e
    
    async def configure_geographic_restrictions(self, distribution_id: str, 
                                              restriction_type: str = 'none',
                                              locations: Optional[List[str]] = None) -> bool:
        """
        Configure geographic restrictions for CloudFront distribution.
        
        Args:
            distribution_id: CloudFront distribution ID
            restriction_type: 'whitelist', 'blacklist', or 'none'
            locations: List of country codes (ISO 3166-1 alpha-2)
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            AWSIntegrationError: If configuration fails
        """
        try:
            # Get current distribution configuration
            response = self.cloudfront_client.get_distribution_config(Id=distribution_id)
            config = response['DistributionConfig']
            etag = response['ETag']
            
            # Configure restrictions
            if restriction_type == 'none':
                config['Restrictions'] = {
                    'GeoRestriction': {
                        'RestrictionType': 'none',
                        'Quantity': 0
                    }
                }
            else:
                if not locations:
                    locations = []
                
                config['Restrictions'] = {
                    'GeoRestriction': {
                        'RestrictionType': restriction_type,
                        'Quantity': len(locations),
                        'Items': locations
                    }
                }
            
            # Update distribution
            self.cloudfront_client.update_distribution(
                Id=distribution_id,
                DistributionConfig=config,
                IfMatch=etag
            )
            
            logger.info(f"Applied geographic restrictions to distribution: {distribution_id}")
            logger.info(f"Restriction type: {restriction_type}, Locations: {locations}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to configure geographic restrictions: {e}")
            raise AWSIntegrationError(f"Geographic restrictions configuration failed: {str(e)}") from e
    
    def get_distribution_info(self, distribution_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get CloudFront distribution information.
        
        Args:
            distribution_id: Optional distribution ID (uses config if not provided)
            
        Returns:
            Dictionary with distribution information
            
        Raises:
            AWSIntegrationError: If retrieval fails
        """
        try:
            dist_id = distribution_id or self.config.cloudfront_distribution_id
            if not dist_id:
                raise AWSIntegrationError("No distribution ID provided or configured")
            
            response = self.cloudfront_client.get_distribution(Id=dist_id)
            distribution = response['Distribution']
            
            return {
                'id': distribution['Id'],
                'arn': distribution['ARN'],
                'status': distribution['Status'],
                'domain_name': distribution['DomainName'],
                'last_modified_time': distribution['LastModifiedTime'].isoformat(),
                'enabled': distribution['DistributionConfig']['Enabled'],
                'price_class': distribution['DistributionConfig']['PriceClass'],
                'origins': [
                    {
                        'id': origin['Id'],
                        'domain_name': origin['DomainName']
                    }
                    for origin in distribution['DistributionConfig']['Origins']['Items']
                ]
            }
            
        except ClientError as e:
            logger.error(f"Failed to get distribution info: {e}")
            raise AWSIntegrationError(f"Distribution info retrieval failed: {str(e)}") from e
    
    def generate_video_url(self, s3_key: str, distribution_id: Optional[str] = None,
                          domain_name: Optional[str] = None) -> str:
        """
        Generate CloudFront URL for video access.
        
        Args:
            s3_key: S3 object key for the video
            distribution_id: Optional distribution ID
            domain_name: Optional CloudFront domain name
            
        Returns:
            CloudFront URL for video access
            
        Raises:
            AWSIntegrationError: If URL generation fails
        """
        try:
            # Use provided domain or get from config
            if domain_name:
                cf_domain = domain_name
            elif self.config.cloudfront_domain:
                cf_domain = self.config.cloudfront_domain
            elif distribution_id or self.config.cloudfront_distribution_id:
                # Get domain from distribution info
                dist_info = self.get_distribution_info(distribution_id)
                cf_domain = dist_info['domain_name']
            else:
                raise AWSIntegrationError("No CloudFront domain available for URL generation")
            
            # Generate HTTPS URL
            cloudfront_url = f"https://{cf_domain}/{s3_key}"
            
            logger.debug(f"Generated CloudFront URL: {cloudfront_url}")
            return cloudfront_url
            
        except Exception as e:
            logger.error(f"Failed to generate CloudFront URL: {e}")
            raise AWSIntegrationError(f"URL generation failed: {str(e)}") from e
    
    def __repr__(self) -> str:
        """String representation of CloudFront service."""
        return (
            f"CloudFrontService(region='{self.config.region}', "
            f"distribution_id='{self.config.cloudfront_distribution_id}', "
            f"domain='{self.config.cloudfront_domain}')"
        )   
 # Cache Management and Invalidation Methods
    
    async def invalidate_cache(self, paths: Union[str, List[str]], 
                              distribution_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create cache invalidation for updated videos.
        
        Args:
            paths: Single path or list of paths to invalidate
            distribution_id: Optional distribution ID (uses config if not provided)
            
        Returns:
            Dictionary with invalidation information
            
        Raises:
            AWSIntegrationError: If invalidation fails
        """
        try:
            dist_id = distribution_id or self.config.cloudfront_distribution_id
            if not dist_id:
                raise AWSIntegrationError("No distribution ID provided or configured")
            
            # Ensure paths is a list
            if isinstance(paths, str):
                paths = [paths]
            
            # Prepare invalidation request
            invalidation_batch = {
                'Paths': {
                    'Quantity': len(paths),
                    'Items': paths
                },
                'CallerReference': f"invalidation-{datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')}"
            }
            
            # Create invalidation
            response = self.cloudfront_client.create_invalidation(
                DistributionId=dist_id,
                InvalidationBatch=invalidation_batch
            )
            
            invalidation = response['Invalidation']
            invalidation_id = invalidation['Id']
            
            logger.info(f"Created cache invalidation: {invalidation_id}")
            logger.info(f"Invalidated paths: {paths}")
            
            return {
                'invalidation_id': invalidation_id,
                'status': invalidation['Status'],
                'create_time': invalidation['CreateTime'].isoformat(),
                'paths': paths,
                'caller_reference': invalidation['InvalidationBatch']['CallerReference']
            }
            
        except ClientError as e:
            logger.error(f"Failed to create cache invalidation: {e}")
            raise AWSIntegrationError(f"Cache invalidation failed: {str(e)}") from e
    
    async def get_invalidation_status(self, invalidation_id: str, 
                                     distribution_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of cache invalidation.
        
        Args:
            invalidation_id: Invalidation ID
            distribution_id: Optional distribution ID
            
        Returns:
            Dictionary with invalidation status
            
        Raises:
            AWSIntegrationError: If status retrieval fails
        """
        try:
            dist_id = distribution_id or self.config.cloudfront_distribution_id
            if not dist_id:
                raise AWSIntegrationError("No distribution ID provided or configured")
            
            response = self.cloudfront_client.get_invalidation(
                DistributionId=dist_id,
                Id=invalidation_id
            )
            
            invalidation = response['Invalidation']
            
            return {
                'invalidation_id': invalidation['Id'],
                'status': invalidation['Status'],
                'create_time': invalidation['CreateTime'].isoformat(),
                'paths': invalidation['InvalidationBatch']['Paths']['Items']
            }
            
        except ClientError as e:
            logger.error(f"Failed to get invalidation status: {e}")
            raise AWSIntegrationError(f"Invalidation status retrieval failed: {str(e)}") from e
    
    async def list_invalidations(self, distribution_id: Optional[str] = None,
                                max_items: int = 100) -> List[Dict[str, Any]]:
        """
        List cache invalidations for distribution.
        
        Args:
            distribution_id: Optional distribution ID
            max_items: Maximum number of invalidations to return
            
        Returns:
            List of invalidation summaries
            
        Raises:
            AWSIntegrationError: If listing fails
        """
        try:
            dist_id = distribution_id or self.config.cloudfront_distribution_id
            if not dist_id:
                raise AWSIntegrationError("No distribution ID provided or configured")
            
            response = self.cloudfront_client.list_invalidations(
                DistributionId=dist_id,
                MaxItems=str(max_items)
            )
            
            invalidations = []
            for item in response['InvalidationList']['Items']:
                invalidations.append({
                    'invalidation_id': item['Id'],
                    'status': item['Status'],
                    'create_time': item['CreateTime'].isoformat()
                })
            
            return invalidations
            
        except ClientError as e:
            logger.error(f"Failed to list invalidations: {e}")
            raise AWSIntegrationError(f"Invalidation listing failed: {str(e)}") from e
    
    # Performance Monitoring and Metrics
    
    async def get_cache_hit_rate(self, distribution_id: Optional[str] = None,
                                start_time: Optional[datetime] = None,
                                end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get cache hit rate metrics for CloudFront distribution.
        
        Args:
            distribution_id: Optional distribution ID
            start_time: Start time for metrics (defaults to 24 hours ago)
            end_time: End time for metrics (defaults to now)
            
        Returns:
            Dictionary with cache hit rate metrics
            
        Raises:
            AWSIntegrationError: If metrics retrieval fails
        """
        try:
            dist_id = distribution_id or self.config.cloudfront_distribution_id
            if not dist_id:
                raise AWSIntegrationError("No distribution ID provided or configured")
            
            # Set default time range (last 24 hours)
            if not end_time:
                end_time = datetime.utcnow()
            if not start_time:
                start_time = end_time - timedelta(hours=24)
            
            # Get cache hit rate metric
            response = self.cloudwatch_client.get_metric_statistics(
                Namespace='AWS/CloudFront',
                MetricName='CacheHitRate',
                Dimensions=[
                    {
                        'Name': 'DistributionId',
                        'Value': dist_id
                    }
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,  # 1 hour periods
                Statistics=['Average', 'Maximum', 'Minimum']
            )
            
            datapoints = response['Datapoints']
            datapoints.sort(key=lambda x: x['Timestamp'])
            
            # Calculate overall metrics
            if datapoints:
                avg_hit_rate = sum(dp['Average'] for dp in datapoints) / len(datapoints)
                max_hit_rate = max(dp['Maximum'] for dp in datapoints)
                min_hit_rate = min(dp['Minimum'] for dp in datapoints)
            else:
                avg_hit_rate = max_hit_rate = min_hit_rate = 0
            
            return {
                'distribution_id': dist_id,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'average_hit_rate': round(avg_hit_rate, 2),
                'maximum_hit_rate': round(max_hit_rate, 2),
                'minimum_hit_rate': round(min_hit_rate, 2),
                'datapoints': [
                    {
                        'timestamp': dp['Timestamp'].isoformat(),
                        'average': round(dp['Average'], 2),
                        'maximum': round(dp['Maximum'], 2),
                        'minimum': round(dp['Minimum'], 2)
                    }
                    for dp in datapoints
                ]
            }
            
        except ClientError as e:
            logger.error(f"Failed to get cache hit rate metrics: {e}")
            raise AWSIntegrationError(f"Cache hit rate metrics retrieval failed: {str(e)}") from e
    
    async def get_performance_metrics(self, distribution_id: Optional[str] = None,
                                     start_time: Optional[datetime] = None,
                                     end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for CloudFront distribution.
        
        Args:
            distribution_id: Optional distribution ID
            start_time: Start time for metrics (defaults to 24 hours ago)
            end_time: End time for metrics (defaults to now)
            
        Returns:
            Dictionary with performance metrics
            
        Raises:
            AWSIntegrationError: If metrics retrieval fails
        """
        try:
            dist_id = distribution_id or self.config.cloudfront_distribution_id
            if not dist_id:
                raise AWSIntegrationError("No distribution ID provided or configured")
            
            # Set default time range (last 24 hours)
            if not end_time:
                end_time = datetime.utcnow()
            if not start_time:
                start_time = end_time - timedelta(hours=24)
            
            # Define metrics to collect
            metrics_to_collect = [
                'Requests',
                'BytesDownloaded',
                'BytesUploaded',
                'TotalErrorRate',
                '4xxErrorRate',
                '5xxErrorRate',
                'OriginLatency'
            ]
            
            metrics_data = {}
            
            # Collect each metric
            for metric_name in metrics_to_collect:
                try:
                    response = self.cloudwatch_client.get_metric_statistics(
                        Namespace='AWS/CloudFront',
                        MetricName=metric_name,
                        Dimensions=[
                            {
                                'Name': 'DistributionId',
                                'Value': dist_id
                            }
                        ],
                        StartTime=start_time,
                        EndTime=end_time,
                        Period=3600,  # 1 hour periods
                        Statistics=['Sum', 'Average', 'Maximum'] if metric_name in ['Requests', 'BytesDownloaded', 'BytesUploaded'] else ['Average', 'Maximum']
                    )
                    
                    datapoints = response['Datapoints']
                    datapoints.sort(key=lambda x: x['Timestamp'])
                    
                    if datapoints:
                        if metric_name in ['Requests', 'BytesDownloaded', 'BytesUploaded']:
                            total = sum(dp['Sum'] for dp in datapoints)
                            avg = sum(dp['Average'] for dp in datapoints) / len(datapoints)
                            max_val = max(dp['Maximum'] for dp in datapoints)
                        else:
                            total = None
                            avg = sum(dp['Average'] for dp in datapoints) / len(datapoints)
                            max_val = max(dp['Maximum'] for dp in datapoints)
                        
                        metrics_data[metric_name] = {
                            'total': total,
                            'average': round(avg, 2),
                            'maximum': round(max_val, 2),
                            'datapoints': len(datapoints)
                        }
                    else:
                        metrics_data[metric_name] = {
                            'total': None,
                            'average': 0,
                            'maximum': 0,
                            'datapoints': 0
                        }
                        
                except ClientError as e:
                    logger.warning(f"Failed to get metric {metric_name}: {e}")
                    metrics_data[metric_name] = {
                        'total': None,
                        'average': 0,
                        'maximum': 0,
                        'datapoints': 0,
                        'error': str(e)
                    }
            
            return {
                'distribution_id': dist_id,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'metrics': metrics_data
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            raise AWSIntegrationError(f"Performance metrics retrieval failed: {str(e)}") from e
    
    async def optimize_cache_behaviors(self, distribution_id: Optional[str] = None,
                                      hit_rate_threshold: float = 80.0) -> Dict[str, Any]:
        """
        Analyze and suggest cache behavior optimizations based on performance metrics.
        
        Args:
            distribution_id: Optional distribution ID
            hit_rate_threshold: Minimum acceptable cache hit rate percentage
            
        Returns:
            Dictionary with optimization recommendations
            
        Raises:
            AWSIntegrationError: If optimization analysis fails
        """
        try:
            dist_id = distribution_id or self.config.cloudfront_distribution_id
            if not dist_id:
                raise AWSIntegrationError("No distribution ID provided or configured")
            
            # Get current cache hit rate
            hit_rate_data = await self.get_cache_hit_rate(dist_id)
            current_hit_rate = hit_rate_data['average_hit_rate']
            
            # Get performance metrics
            perf_metrics = await self.get_performance_metrics(dist_id)
            
            # Analyze and generate recommendations
            recommendations = []
            
            if current_hit_rate < hit_rate_threshold:
                recommendations.append({
                    'type': 'cache_hit_rate',
                    'severity': 'high' if current_hit_rate < 60 else 'medium',
                    'message': f"Cache hit rate ({current_hit_rate}%) is below threshold ({hit_rate_threshold}%)",
                    'suggestions': [
                        "Increase TTL values for static video content",
                        "Review cache behaviors for video files (*.mp4)",
                        "Consider enabling compression for non-video assets",
                        "Optimize query string forwarding settings"
                    ]
                })
            
            # Check error rates
            error_rate = perf_metrics['metrics'].get('TotalErrorRate', {}).get('average', 0)
            if error_rate > 5.0:
                recommendations.append({
                    'type': 'error_rate',
                    'severity': 'high' if error_rate > 10 else 'medium',
                    'message': f"Total error rate ({error_rate}%) is elevated",
                    'suggestions': [
                        "Check origin server health and capacity",
                        "Review S3 bucket permissions and OAI configuration",
                        "Consider implementing custom error pages",
                        "Monitor origin latency metrics"
                    ]
                })
            
            # Check origin latency
            origin_latency = perf_metrics['metrics'].get('OriginLatency', {}).get('average', 0)
            if origin_latency > 1000:  # 1 second
                recommendations.append({
                    'type': 'origin_latency',
                    'severity': 'medium',
                    'message': f"Origin latency ({origin_latency}ms) is high",
                    'suggestions': [
                        "Consider using S3 Transfer Acceleration",
                        "Review S3 bucket region and CloudFront edge locations",
                        "Optimize video file sizes and formats",
                        "Implement video transcoding for multiple bitrates"
                    ]
                })
            
            return {
                'distribution_id': dist_id,
                'analysis_time': datetime.utcnow().isoformat(),
                'current_hit_rate': current_hit_rate,
                'hit_rate_threshold': hit_rate_threshold,
                'total_recommendations': len(recommendations),
                'recommendations': recommendations,
                'performance_summary': {
                    'cache_hit_rate': current_hit_rate,
                    'error_rate': error_rate,
                    'origin_latency': origin_latency,
                    'total_requests': perf_metrics['metrics'].get('Requests', {}).get('total', 0),
                    'bytes_downloaded': perf_metrics['metrics'].get('BytesDownloaded', {}).get('total', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize cache behaviors: {e}")
            raise AWSIntegrationError(f"Cache optimization analysis failed: {str(e)}") from e
    
    # Health Check and Status Methods
    
    async def health_check(self, distribution_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform health check on CloudFront distribution.
        
        Args:
            distribution_id: Optional distribution ID
            
        Returns:
            Dictionary with health status
        """
        health_status = {
            'service': 'CloudFront',
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {}
        }
        
        try:
            dist_id = distribution_id or self.config.cloudfront_distribution_id
            
            if not dist_id:
                health_status['status'] = 'degraded'
                health_status['checks']['distribution_config'] = {
                    'status': 'failed',
                    'message': 'No distribution ID configured'
                }
                return health_status
            
            # Check distribution status
            try:
                dist_info = self.get_distribution_info(dist_id)
                health_status['checks']['distribution_status'] = {
                    'status': 'healthy' if dist_info['status'] == 'Deployed' else 'degraded',
                    'distribution_status': dist_info['status'],
                    'enabled': dist_info['enabled']
                }
                
                if dist_info['status'] != 'Deployed':
                    health_status['status'] = 'degraded'
                    
            except Exception as e:
                health_status['checks']['distribution_status'] = {
                    'status': 'failed',
                    'error': str(e)
                }
                health_status['status'] = 'unhealthy'
            
            # Check recent performance metrics
            try:
                hit_rate_data = await self.get_cache_hit_rate(dist_id)
                health_status['checks']['cache_performance'] = {
                    'status': 'healthy' if hit_rate_data['average_hit_rate'] > 50 else 'degraded',
                    'cache_hit_rate': hit_rate_data['average_hit_rate']
                }
                
                if hit_rate_data['average_hit_rate'] < 30:
                    health_status['status'] = 'degraded'
                    
            except Exception as e:
                health_status['checks']['cache_performance'] = {
                    'status': 'unknown',
                    'error': str(e)
                }
            
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['error'] = str(e)
        
        return health_status
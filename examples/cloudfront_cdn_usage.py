#!/usr/bin/env python3
"""
CloudFront CDN Integration Usage Example

This example demonstrates how to use the CloudFront service for video content delivery
optimization, including distribution creation, cache management, and performance monitoring.
"""

import asyncio
import logging
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.aws.config import AWSConfig
from app.core.aws.cloudfront_service import CloudFrontService
from app.core.aws.aws_integration_service import AWSIntegrationService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Main example function demonstrating CloudFront CDN usage."""
    
    # Initialize AWS configuration
    config = AWSConfig(
        region='us-east-1',
        video_bucket_name='my-video-bucket',
        code_bucket_name='my-code-bucket',
        enable_cloudfront=True,
        cloudfront_distribution_id='E1234567890ABC',  # Optional: existing distribution
        cloudfront_domain='d1234567890abc.cloudfront.net'  # Optional: existing domain
    )
    
    print("=== CloudFront CDN Integration Example ===\n")
    
    # Example 1: Initialize CloudFront service
    print("1. Initializing CloudFront service...")
    try:
        cloudfront_service = CloudFrontService(config)
        print("✓ CloudFront service initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize CloudFront service: {e}")
        return
    
    # Example 2: Create CloudFront distribution (if needed)
    print("\n2. Creating CloudFront distribution...")
    try:
        # Only create if no distribution ID is configured
        if not config.cloudfront_distribution_id:
            distribution_info = await cloudfront_service.create_distribution(
                video_bucket_name=config.video_bucket_name
            )
            print(f"✓ Created distribution: {distribution_info['distribution_id']}")
            print(f"  Domain: {distribution_info['domain_name']}")
            print(f"  Status: {distribution_info['status']}")
        else:
            print(f"✓ Using existing distribution: {config.cloudfront_distribution_id}")
    except Exception as e:
        print(f"✗ Failed to create distribution: {e}")
    
    # Example 3: Generate CloudFront URLs for videos
    print("\n3. Generating CloudFront URLs...")
    try:
        video_paths = [
            'videos/project1/video1/chunk_001_v1.mp4',
            'videos/project1/video1/chunk_002_v1.mp4',
            'videos/project1/video1/combined_v1.mp4'
        ]
        
        for video_path in video_paths:
            cloudfront_url = cloudfront_service.generate_video_url(video_path)
            print(f"  {video_path} -> {cloudfront_url}")
        
        print("✓ Generated CloudFront URLs successfully")
    except Exception as e:
        print(f"✗ Failed to generate URLs: {e}")
    
    # Example 4: Cache invalidation
    print("\n4. Invalidating CloudFront cache...")
    try:
        # Invalidate cache for updated videos
        invalidation_paths = [
            '/videos/project1/video1/chunk_001_v1.mp4',
            '/videos/project1/video1/combined_v1.mp4'
        ]
        
        invalidation_result = await cloudfront_service.invalidate_cache(invalidation_paths)
        print(f"✓ Cache invalidation created: {invalidation_result['invalidation_id']}")
        print(f"  Status: {invalidation_result['status']}")
        print(f"  Paths: {len(invalidation_result['paths'])} paths invalidated")
    except Exception as e:
        print(f"✗ Failed to invalidate cache: {e}")
    
    # Example 5: Get cache hit rate metrics
    print("\n5. Retrieving cache hit rate metrics...")
    try:
        hit_rate_data = await cloudfront_service.get_cache_hit_rate()
        print(f"✓ Cache hit rate metrics retrieved:")
        print(f"  Average hit rate: {hit_rate_data['average_hit_rate']}%")
        print(f"  Maximum hit rate: {hit_rate_data['maximum_hit_rate']}%")
        print(f"  Minimum hit rate: {hit_rate_data['minimum_hit_rate']}%")
        print(f"  Data points: {len(hit_rate_data['datapoints'])}")
    except Exception as e:
        print(f"✗ Failed to get cache hit rate: {e}")
    
    # Example 6: Get comprehensive performance metrics
    print("\n6. Retrieving performance metrics...")
    try:
        perf_metrics = await cloudfront_service.get_performance_metrics()
        print(f"✓ Performance metrics retrieved:")
        
        metrics = perf_metrics['metrics']
        if 'Requests' in metrics:
            print(f"  Total requests: {metrics['Requests'].get('total', 'N/A')}")
        if 'BytesDownloaded' in metrics:
            print(f"  Bytes downloaded: {metrics['BytesDownloaded'].get('total', 'N/A')}")
        if 'TotalErrorRate' in metrics:
            print(f"  Error rate: {metrics['TotalErrorRate'].get('average', 'N/A')}%")
        if 'OriginLatency' in metrics:
            print(f"  Origin latency: {metrics['OriginLatency'].get('average', 'N/A')}ms")
    except Exception as e:
        print(f"✗ Failed to get performance metrics: {e}")
    
    # Example 7: Cache optimization analysis
    print("\n7. Analyzing cache optimization...")
    try:
        optimization_result = await cloudfront_service.optimize_cache_behaviors()
        print(f"✓ Cache optimization analysis completed:")
        print(f"  Current hit rate: {optimization_result['current_hit_rate']}%")
        print(f"  Recommendations: {optimization_result['total_recommendations']}")
        
        for rec in optimization_result['recommendations']:
            print(f"    - {rec['type']}: {rec['message']}")
            for suggestion in rec['suggestions'][:2]:  # Show first 2 suggestions
                print(f"      • {suggestion}")
    except Exception as e:
        print(f"✗ Failed to analyze cache optimization: {e}")
    
    # Example 8: Health check
    print("\n8. Performing health check...")
    try:
        health_status = await cloudfront_service.health_check()
        print(f"✓ Health check completed:")
        print(f"  Overall status: {health_status['status']}")
        
        for service, status in health_status['checks'].items():
            print(f"  {service}: {status['status']}")
    except Exception as e:
        print(f"✗ Failed to perform health check: {e}")
    
    # Example 9: Using AWS Integration Service with CloudFront
    print("\n9. Using AWS Integration Service with CloudFront...")
    try:
        aws_service = AWSIntegrationService(config)
        
        # Check if CloudFront is enabled
        if aws_service.cloudfront_service:
            print("✓ CloudFront service is enabled in AWS Integration Service")
            
            # Generate CloudFront URL through integration service
            cloudfront_url = aws_service.get_cloudfront_video_url('videos/test.mp4')
            print(f"  Generated URL: {cloudfront_url}")
            
            # Get performance metrics through integration service
            metrics = await aws_service.get_cloudfront_performance_metrics()
            print(f"  Retrieved metrics for distribution: {metrics.get('distribution_id', 'N/A')}")
        else:
            print("✗ CloudFront service is not enabled")
    except Exception as e:
        print(f"✗ Failed to use AWS Integration Service: {e}")
    
    print("\n=== CloudFront CDN Integration Example Complete ===")


async def example_video_upload_with_cdn():
    """Example of uploading videos with automatic CDN integration."""
    
    print("\n=== Video Upload with CDN Example ===")
    
    # This would be used in a real scenario with actual video files
    config = AWSConfig(
        region='us-east-1',
        video_bucket_name='my-video-bucket',
        code_bucket_name='my-code-bucket',
        enable_cloudfront=True,
        enable_aws_upload=True
    )
    
    try:
        aws_service = AWSIntegrationService(config)
        
        # Simulate video chunks (in real usage, these would be actual VideoChunk objects)
        print("1. Simulating video upload with CDN integration...")
        
        # This is a simulation - in real usage you would have actual video chunks
        video_metadata = {
            'title': 'AI Generated Video',
            'description': 'Example video with CDN delivery',
            'tags': ['ai', 'generated', 'demo']
        }
        
        print("✓ Video upload with CDN would include:")
        print("  - Upload video chunks to S3")
        print("  - Update DynamoDB metadata")
        print("  - Generate CloudFront URLs")
        print("  - Invalidate cache for updated content")
        
        # Example of what the result would look like
        example_result = {
            'upload_results': ['s3://bucket/video1.mp4', 's3://bucket/video2.mp4'],
            'successful_uploads': 2,
            'total_uploads': 2,
            'metadata_updated': True,
            'cloudfront_urls': {
                'chunk1': 'https://d123.cloudfront.net/videos/chunk1.mp4',
                'chunk2': 'https://d123.cloudfront.net/videos/chunk2.mp4'
            },
            'cache_invalidation': {
                'invalidation_id': 'I1234567890ABC',
                'status': 'InProgress'
            },
            'cdn_enabled': True
        }
        
        print(f"✓ Example result structure: {len(example_result)} keys")
        print(f"  CDN URLs generated: {len(example_result['cloudfront_urls'])}")
        print(f"  Cache invalidation: {example_result['cache_invalidation']['status']}")
        
    except Exception as e:
        print(f"✗ Failed to demonstrate video upload with CDN: {e}")


if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())
    
    # Run the video upload example
    asyncio.run(example_video_upload_with_cdn())
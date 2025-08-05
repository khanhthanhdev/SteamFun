#!/usr/bin/env python3
"""
MediaConvert Transcoding Integration Usage Example

This example demonstrates how to use the MediaConvert transcoding service
for adaptive bitrate streaming with HLS and DASH output formats.
"""

import asyncio
import os
from datetime import datetime

from app.core.aws.config import AWSConfig
from app.core.aws.credentials import AWSCredentialsManager
from app.core.aws.mediaconvert_service import (
    MediaConvertService,
    TranscodingJobConfig,
    OutputFormat,
    QualityLevel,
    TranscodingStatus
)


async def main():
    """Main example function."""
    print("MediaConvert Transcoding Integration Example")
    print("=" * 50)
    
    # Initialize AWS configuration
    config = AWSConfig()
    config.region = 'us-east-1'
    config.mediaconvert_role_arn = os.getenv('AWS_MEDIACONVERT_ROLE_ARN', 'arn:aws:iam::123456789012:role/MediaConvertRole')
    config.enable_transcoding = True
    
    print(f"Region: {config.region}")
    print(f"MediaConvert Role ARN: {config.mediaconvert_role_arn}")
    print()
    
    try:
        # Initialize credentials manager
        credentials_manager = AWSCredentialsManager(config)
        
        # Initialize MediaConvert service
        mediaconvert_service = MediaConvertService(config, credentials_manager)
        print(f"MediaConvert service initialized with endpoint: {config.mediaconvert_endpoint}")
        print()
        
        # Create transcoding job configuration
        job_config = TranscodingJobConfig(
            input_s3_path="s3://my-video-bucket/input/sample-video.mp4",
            output_s3_prefix="s3://my-video-bucket/transcoded/sample-video",
            video_id="sample-video-123",
            project_id="demo-project-456",
            output_formats=[OutputFormat.HLS, OutputFormat.DASH, OutputFormat.MP4],
            quality_levels=[QualityLevel.HD_1080P, QualityLevel.HD_720P, QualityLevel.SD_480P],
            metadata={
                "title": "Sample Video",
                "description": "Demo video for transcoding",
                "created_by": "example_script"
            }
        )
        
        print("Transcoding Job Configuration:")
        print(f"  Input: {job_config.input_s3_path}")
        print(f"  Output Prefix: {job_config.output_s3_prefix}")
        print(f"  Video ID: {job_config.video_id}")
        print(f"  Project ID: {job_config.project_id}")
        print(f"  Output Formats: {[fmt.value for fmt in job_config.output_formats]}")
        print(f"  Quality Levels: {[qual.value for qual in job_config.quality_levels]}")
        print()
        
        # Create progress callback
        async def progress_callback(job_result):
            print(f"Job {job_result.job_id} status: {job_result.status.value}")
            if job_result.progress_percentage:
                print(f"Progress: {job_result.progress_percentage}%")
            if job_result.error_message:
                print(f"Error: {job_result.error_message}")
            print()
        
        # Create transcoding job
        print("Creating transcoding job...")
        job_id = await mediaconvert_service.create_transcoding_job(
            job_config, 
            progress_callback=progress_callback
        )
        print(f"Transcoding job created with ID: {job_id}")
        print()
        
        # Get initial job status
        print("Getting job status...")
        job_status = await mediaconvert_service.get_job_status(job_id)
        print(f"Job Status: {job_status.status.value}")
        print(f"Created At: {job_status.created_at}")
        if job_status.progress_percentage:
            print(f"Progress: {job_status.progress_percentage}%")
        print()
        
        # List recent jobs
        print("Listing recent transcoding jobs...")
        recent_jobs = await mediaconvert_service.list_jobs(max_results=5)
        for i, job in enumerate(recent_jobs, 1):
            print(f"  {i}. Job ID: {job.job_id}")
            print(f"     Status: {job.status.value}")
            print(f"     Created: {job.created_at}")
            if job.progress_percentage:
                print(f"     Progress: {job.progress_percentage}%")
            print()
        
        # Demonstrate waiting for completion (commented out for demo)
        # print("Waiting for job completion...")
        # final_result = await mediaconvert_service.wait_for_job_completion(job_id, timeout_minutes=60)
        # print(f"Job completed with status: {final_result.status.value}")
        # if final_result.output_paths:
        #     print("Output paths:")
        #     for format_name, paths in final_result.output_paths.items():
        #         print(f"  {format_name}: {paths}")
        
        # Health check
        print("Performing health check...")
        health = await mediaconvert_service.health_check()
        print(f"MediaConvert Health: {health['status']}")
        print(f"Endpoint: {health.get('endpoint', 'N/A')}")
        print(f"Region: {health['region']}")
        print()
        
        print("Example completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. AWS credentials configured")
        print("2. MediaConvert role ARN set in AWS_MEDIACONVERT_ROLE_ARN environment variable")
        print("3. Proper IAM permissions for MediaConvert operations")


def demonstrate_job_settings():
    """Demonstrate building MediaConvert job settings."""
    print("\nMediaConvert Job Settings Example")
    print("=" * 40)
    
    # Create a sample configuration
    config = AWSConfig()
    config.mediaconvert_role_arn = "arn:aws:iam::123456789012:role/MediaConvertRole"
    
    # Create credentials manager (mock for demo)
    from unittest.mock import Mock
    credentials_manager = Mock()
    
    # Create service
    try:
        service = MediaConvertService(config, credentials_manager)
        
        # Create job config
        job_config = TranscodingJobConfig(
            input_s3_path="s3://example-bucket/input/video.mp4",
            output_s3_prefix="s3://example-bucket/output/transcoded",
            video_id="demo-video",
            project_id="demo-project",
            output_formats=[OutputFormat.HLS, OutputFormat.DASH],
            quality_levels=[QualityLevel.HD_1080P, QualityLevel.HD_720P]
        )
        
        # Build job settings
        job_settings = service._build_job_settings(job_config)
        
        print("Generated MediaConvert Job Settings:")
        print(f"Role: {job_settings['Role']}")
        print(f"Input File: {job_settings['Settings']['Inputs'][0]['FileInput']}")
        print(f"Number of Output Groups: {len(job_settings['Settings']['OutputGroups'])}")
        
        for i, group in enumerate(job_settings['Settings']['OutputGroups']):
            print(f"  Output Group {i+1}: {group['Name']}")
            print(f"    Type: {group['OutputGroupSettings']['Type']}")
            print(f"    Outputs: {len(group['Outputs'])}")
        
        print("\nJob settings generated successfully!")
        
    except Exception as e:
        print(f"Error generating job settings: {e}")


if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())
    
    # Demonstrate job settings generation
    demonstrate_job_settings()
"""
AWS Integration Usage Example

This example demonstrates how to use the AWS integration foundation
components in your application.
"""

import os
import sys
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aws.config import AWSConfig
from aws.credentials import AWSCredentialsManager
from aws.logging_config import setup_aws_logging, aws_operation
from aws.exceptions import AWSConfigurationError, AWSCredentialsError


def example_basic_setup():
    """Example of basic AWS integration setup."""
    print("=== Basic AWS Integration Setup ===")
    
    try:
        # 1. Load configuration from environment
        config = AWSConfig.from_env()
        print(f"Configuration loaded: {config}")
        
        # 2. Set up logging
        logger = setup_aws_logging(config)
        logger.info("AWS integration initialized")
        
        # 3. Initialize credentials manager
        creds_manager = AWSCredentialsManager(config)
        
        # 4. Get credential information (without creating session)
        creds_info = creds_manager.get_credentials_info()
        print(f"Credentials status: {creds_info['status']}")
        
        return config, creds_manager, logger
        
    except (AWSConfigurationError, AWSCredentialsError) as e:
        print(f"AWS setup error: {e}")
        return None, None, None


def example_configuration_options():
    """Example of different configuration options."""
    print("\n=== Configuration Options ===")
    
    # Configuration from environment variables
    config1 = AWSConfig.from_env()
    print(f"From environment: {config1.region}")
    
    # Configuration with explicit values
    config2 = AWSConfig(
        region='us-west-2',
        video_bucket_name='my-video-bucket',
        code_bucket_name='my-code-bucket',
        enable_aws_upload=True,
        log_level='DEBUG'
    )
    print(f"Explicit config: {config2.region}")
    
    # Configuration from dictionary
    config_dict = {
        'region': 'eu-west-1',
        'enable_aws_upload': False,
        'max_retries': 5
    }
    config3 = AWSConfig.from_dict(config_dict)
    print(f"From dict: {config3.region}")


def example_logging_usage():
    """Example of AWS logging usage."""
    print("\n=== Logging Usage ===")
    
    config = AWSConfig(log_level='INFO')
    logger = setup_aws_logging(config)
    
    # Basic logging
    logger.info("Starting AWS operation")
    logger.warning("This is a warning")
    
    # Using operation context manager
    with aws_operation('upload_file', 's3', bucket='test-bucket', key='test.txt'):
        logger.info("Uploading file to S3")
        # Simulate some work
        import time
        time.sleep(0.1)
    
    print("Check the logs directory for detailed logs")


def example_error_handling():
    """Example of error handling."""
    print("\n=== Error Handling ===")
    
    try:
        # This will fail validation
        config = AWSConfig(
            enable_aws_upload=True,
            video_bucket_name='',  # Missing required bucket
            max_retries=-1  # Invalid value
        )
    except AWSConfigurationError as e:
        print(f"Configuration error caught: {e}")
    
    try:
        # This will fail with invalid profile
        config = AWSConfig(profile='non-existent-profile')
        creds_manager = AWSCredentialsManager(config)
        # This would fail if we tried to create a session
        # session = creds_manager.get_session()
    except AWSCredentialsError as e:
        print(f"Credentials error would be: {e}")


def example_production_setup():
    """Example of production-ready setup."""
    print("\n=== Production Setup Example ===")
    
    # Set environment variables for production
    os.environ.update({
        'AWS_REGION': 'us-east-1',
        'AWS_S3_VIDEO_BUCKET': 'my-production-videos',
        'AWS_S3_CODE_BUCKET': 'my-production-code',
        'AWS_DYNAMODB_METADATA_TABLE': 'ProductionVideoMetadata',
        'AWS_ENABLE_UPLOAD': 'true',
        'AWS_ENABLE_ENCRYPTION': 'true',
        'AWS_LOG_LEVEL': 'WARNING',
        'AWS_MAX_RETRIES': '5'
    })
    
    try:
        config = AWSConfig.from_env()
        logger = setup_aws_logging(config, log_file='logs/production_aws.log')
        creds_manager = AWSCredentialsManager(config)
        
        logger.info("Production AWS integration configured")
        print(f"Production config: {config}")
        
        # Test service access (would require actual AWS credentials)
        # s3_access = creds_manager.test_service_access('s3')
        # print(f"S3 access: {s3_access}")
        
    except Exception as e:
        print(f"Production setup error: {e}")


def main():
    """Run all examples."""
    print("AWS Integration Usage Examples")
    print("=" * 50)
    
    # Run examples
    example_basic_setup()
    example_configuration_options()
    example_logging_usage()
    example_error_handling()
    example_production_setup()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nNext steps:")
    print("1. Set your AWS credentials (AWS CLI, IAM roles, or environment variables)")
    print("2. Configure your S3 buckets and DynamoDB table")
    print("3. Update your .env file with AWS settings")
    print("4. Test with actual AWS services")


if __name__ == '__main__':
    main()
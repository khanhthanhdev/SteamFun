"""
Test script for AWS SDK integration foundation.

This script tests the basic AWS configuration, credentials, and logging setup
without requiring actual AWS resources.
"""

import os
import sys
import logging
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from aws.config import AWSConfig
from aws.credentials import AWSCredentialsManager
from aws.logging_config import setup_aws_logging
from aws.exceptions import AWSConfigurationError, AWSCredentialsError


def test_aws_config():
    """Test AWS configuration loading and validation."""
    print("Testing AWS Configuration...")
    
    try:
        # Test with minimal configuration
        os.environ['AWS_REGION'] = 'us-east-1'
        os.environ['AWS_ENABLE_UPLOAD'] = 'false'  # Disable to avoid bucket validation
        
        config = AWSConfig.from_env()
        print(f"✓ Configuration loaded: {config}")
        
        # Test configuration methods
        boto3_config = config.get_boto3_config()
        print(f"✓ Boto3 config: {boto3_config}")
        
        transfer_config = config.get_s3_transfer_config()
        print(f"✓ S3 transfer config: {transfer_config}")
        
        print(f"✓ AWS integration enabled: {config.is_enabled()}")
        
        return True
        
    except AWSConfigurationError as e:
        print(f"✗ Configuration error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_aws_logging():
    """Test AWS logging configuration."""
    print("\nTesting AWS Logging...")
    
    try:
        config = AWSConfig(log_level='INFO', enable_boto3_logging=False)
        logger = setup_aws_logging(config)
        
        # Test logging
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.debug("Test debug message (should not appear)")
        
        print("✓ AWS logging configured successfully")
        return True
        
    except Exception as e:
        print(f"✗ Logging setup error: {e}")
        return False


def test_credentials_without_aws():
    """Test credentials manager without actual AWS access."""
    print("\nTesting Credentials Manager (without AWS access)...")
    
    try:
        config = AWSConfig(region='us-east-1')
        creds_manager = AWSCredentialsManager(config)
        
        # Test credential source priority
        priority = creds_manager.get_credential_source_priority()
        print(f"✓ Credential source priority: {len(priority)} sources")
        
        # Test credential info without session (should show not_initialized)
        info = creds_manager.get_credentials_info()
        print(f"✓ Credentials info: {info}")
        
        print("✓ Credentials manager initialized successfully")
        return True
        
    except Exception as e:
        print(f"✗ Credentials manager error: {e}")
        return False


def test_error_handling():
    """Test error handling and exceptions."""
    print("\nTesting Error Handling...")
    
    try:
        # Test configuration validation error
        try:
            config = AWSConfig(
                enable_aws_upload=True,
                video_bucket_name='',  # Should cause validation error
                code_bucket_name=''
            )
            print("✗ Should have raised configuration error")
            return False
        except AWSConfigurationError:
            print("✓ Configuration validation error handled correctly")
        
        # Test invalid region
        try:
            config = AWSConfig(region='invalid-region')
            print("✗ Should have raised configuration error for invalid region")
            return False
        except AWSConfigurationError:
            print("✓ Invalid region error handled correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False


def main():
    """Run all foundation tests."""
    print("AWS SDK Integration Foundation Tests")
    print("=" * 50)
    
    tests = [
        test_aws_config,
        test_aws_logging,
        test_credentials_without_aws,
        test_error_handling
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("✓ All foundation tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
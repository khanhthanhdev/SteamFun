"""
Verification script for AWS SDK integration foundation.

This script performs comprehensive verification of all foundation components
to ensure they are properly integrated and working together.
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from aws import (
    AWSConfig, 
    AWSCredentialsManager, 
    setup_aws_logging,
    AWSIntegrationError,
    AWSConfigurationError,
    AWSCredentialsError
)
from aws.logging_config import aws_operation


def verify_imports():
    """Verify all AWS integration imports work correctly."""
    print("🔍 Verifying imports...")
    
    try:
        # Test all main imports
        from aws.config import AWSConfig
        from aws.credentials import AWSCredentialsManager
        from aws.logging_config import setup_aws_logging, aws_operation
        from aws.exceptions import (
            AWSIntegrationError, AWSConfigurationError, AWSCredentialsError,
            AWSS3Error, AWSDynamoDBError, AWSRetryableError
        )
        
        print("✅ All imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def verify_configuration():
    """Verify configuration system works correctly."""
    print("\n🔍 Verifying configuration system...")
    
    try:
        # Test environment-based configuration
        os.environ.update({
            'AWS_REGION': 'us-east-1',
            'AWS_ENABLE_UPLOAD': 'false',
            'AWS_LOG_LEVEL': 'INFO'
        })
        
        config = AWSConfig.from_env()
        assert config.region == 'us-east-1'
        assert config.enable_aws_upload == False
        assert config.log_level == 'INFO'
        
        # Test explicit configuration
        config2 = AWSConfig(
            region='us-west-2',
            enable_aws_upload=False,
            max_retries=5
        )
        assert config2.region == 'us-west-2'
        assert config2.max_retries == 5
        
        # Test configuration methods
        boto3_config = config.get_boto3_config()
        assert 'region_name' in boto3_config
        assert 'retries' in boto3_config
        
        transfer_config = config.get_s3_transfer_config()
        assert 'multipart_threshold' in transfer_config
        assert 'max_concurrency' in transfer_config
        
        print("✅ Configuration system working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def verify_validation():
    """Verify configuration validation works."""
    print("\n🔍 Verifying configuration validation...")
    
    try:
        # Test validation errors
        validation_tests = [
            # Missing buckets when upload enabled
            {
                'config': {'enable_aws_upload': True, 'video_bucket_name': ''},
                'should_fail': True
            },
            # Invalid region
            {
                'config': {'region': 'invalid'},
                'should_fail': True
            },
            # Negative values
            {
                'config': {'max_retries': -1},
                'should_fail': True
            },
            # Valid configuration
            {
                'config': {'region': 'us-east-1', 'enable_aws_upload': False},
                'should_fail': False
            }
        ]
        
        for i, test in enumerate(validation_tests):
            try:
                config = AWSConfig(**test['config'])
                if test['should_fail']:
                    print(f"❌ Test {i+1}: Should have failed validation")
                    return False
            except AWSConfigurationError:
                if not test['should_fail']:
                    print(f"❌ Test {i+1}: Should not have failed validation")
                    return False
        
        print("✅ Configuration validation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False


def verify_logging():
    """Verify logging system works correctly."""
    print("\n🔍 Verifying logging system...")
    
    try:
        # Test logging setup
        config = AWSConfig(log_level='INFO', enable_boto3_logging=False)
        
        # Test with temporary log file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_file = f.name
        
        try:
            logger = setup_aws_logging(config, log_file=log_file)
            
            # Test logging levels
            logger.debug("Debug message")  # Should not appear
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            
            # Test operation logging
            with aws_operation('test_operation', 'test_service', test_param='value'):
                logger.info("Inside operation context")
            
            # Verify log file was created and has content
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    log_content = f.read()
                    if 'Info message' in log_content and 'Warning message' in log_content:
                        print("✅ Logging system working correctly")
                        return True
                    else:
                        print("❌ Log content not as expected")
                        return False
            else:
                print("❌ Log file not created")
                return False
                
        finally:
            # Clean up log file
            if os.path.exists(log_file):
                os.unlink(log_file)
        
    except Exception as e:
        print(f"❌ Logging error: {e}")
        return False


def verify_credentials_manager():
    """Verify credentials manager initialization."""
    print("\n🔍 Verifying credentials manager...")
    
    try:
        config = AWSConfig(region='us-east-1')
        creds_manager = AWSCredentialsManager(config)
        
        # Test credential info without session
        info = creds_manager.get_credentials_info()
        assert info['status'] == 'not_initialized'
        
        # Test credential source priority
        priority = creds_manager.get_credential_source_priority()
        assert len(priority) > 0
        assert isinstance(priority, list)
        
        print("✅ Credentials manager working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Credentials manager error: {e}")
        return False


def verify_error_handling():
    """Verify error handling system."""
    print("\n🔍 Verifying error handling...")
    
    try:
        # Test custom exceptions
        try:
            raise AWSConfigurationError("Test config error", config_key="test_key")
        except AWSConfigurationError as e:
            assert e.error_code == "AWS_CONFIG_ERROR"
            assert e.config_key == "test_key"
        
        try:
            raise AWSCredentialsError("Test creds error", credential_source="test")
        except AWSCredentialsError as e:
            assert e.error_code == "AWS_CREDENTIALS_ERROR"
            assert e.credential_source == "test"
        
        # Test base exception
        try:
            raise AWSIntegrationError("Test base error", error_code="TEST", details={"key": "value"})
        except AWSIntegrationError as e:
            assert e.error_code == "TEST"
            assert e.details["key"] == "value"
        
        print("✅ Error handling working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def verify_integration():
    """Verify all components work together."""
    print("\n🔍 Verifying component integration...")
    
    try:
        # Set up complete integration
        os.environ.update({
            'AWS_REGION': 'us-east-1',
            'AWS_ENABLE_UPLOAD': 'false',
            'AWS_LOG_LEVEL': 'WARNING'
        })
        
        # Initialize all components
        config = AWSConfig.from_env()
        logger = setup_aws_logging(config)
        creds_manager = AWSCredentialsManager(config)
        
        # Test they work together
        logger.info("Testing integration")
        info = creds_manager.get_credentials_info()
        
        # Test configuration serialization
        config_dict = config.to_dict()
        config_restored = AWSConfig.from_dict(config_dict)
        assert config_restored.region == config.region
        
        print("✅ Component integration working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Integration error: {e}")
        return False


def verify_requirements():
    """Verify requirements are met."""
    print("\n🔍 Verifying requirements...")
    
    try:
        # Check boto3 is available
        import boto3
        print(f"✅ boto3 version: {boto3.__version__}")
        
        # Check python-dotenv is available
        import dotenv
        print(f"✅ python-dotenv available")
        
        # Check botocore is available
        import botocore
        print(f"✅ botocore version: {botocore.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing requirement: {e}")
        return False


def main():
    """Run all verification tests."""
    print("AWS SDK Integration Foundation Verification")
    print("=" * 60)
    
    tests = [
        verify_requirements,
        verify_imports,
        verify_configuration,
        verify_validation,
        verify_logging,
        verify_credentials_manager,
        verify_error_handling,
        verify_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print(f"Verification Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All verification tests passed!")
        print("\n✅ AWS SDK integration foundation is ready!")
        print("\nNext steps:")
        print("1. Configure your AWS credentials")
        print("2. Set up your S3 buckets and DynamoDB table")
        print("3. Update your .env file with AWS settings")
        print("4. Proceed to implement the AWS Integration Service (Task 2)")
        return 0
    else:
        print("❌ Some verification tests failed")
        print("Please review the errors above and fix any issues.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
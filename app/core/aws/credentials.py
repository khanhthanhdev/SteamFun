"""
AWS Credentials Management

Handles AWS credentials using IAM roles, boto3 sessions, and secure
credential management practices.
"""

import os
import logging
from typing import Optional, Dict, Any
import boto3
from botocore.exceptions import ClientError, NoCredentialsError, ProfileNotFound
# Note: These imports are for reference only - boto3 handles credential providers automatically

from .exceptions import AWSCredentialsError
from .config import AWSConfig

logger = logging.getLogger(__name__)


class AWSCredentialsManager:
    """
    Manages AWS credentials using boto3 sessions with support for:
    - IAM roles (preferred for production)
    - AWS profiles
    - Environment variables
    - Instance metadata (EC2)
    - Container metadata (ECS/Fargate)
    """
    
    def __init__(self, config: AWSConfig):
        self.config = config
        self._session: Optional[boto3.Session] = None
        self._credentials_source: Optional[str] = None
        
        logger.info("Initializing AWS credentials manager")
    
    def get_session(self) -> boto3.Session:
        """
        Get or create a boto3 session with proper credential handling.
        
        Returns:
            boto3.Session: Configured session
            
        Raises:
            AWSCredentialsError: If credentials cannot be obtained
        """
        if self._session is None:
            self._session = self._create_session()
        
        return self._session
    
    def _create_session(self) -> boto3.Session:
        """Create a new boto3 session with credential detection."""
        try:
            # Try to create session with profile if specified
            if self.config.profile:
                logger.info(f"Attempting to use AWS profile: {self.config.profile}")
                session = boto3.Session(
                    profile_name=self.config.profile,
                    region_name=self.config.region
                )
                self._credentials_source = f"profile:{self.config.profile}"
            else:
                # Use default credential chain
                session = boto3.Session(region_name=self.config.region)
                self._credentials_source = "default_chain"
            
            # Test credentials by making a simple call
            self._validate_credentials(session)
            
            logger.info(f"Successfully initialized AWS session using: {self._credentials_source}")
            return session
            
        except ProfileNotFound as e:
            raise AWSCredentialsError(
                f"AWS profile '{self.config.profile}' not found",
                credential_source="profile"
            ) from e
        except NoCredentialsError as e:
            raise AWSCredentialsError(
                "No AWS credentials found. Please configure credentials using one of: "
                "IAM roles, AWS profiles, environment variables, or instance metadata",
                credential_source="none"
            ) from e
        except Exception as e:
            raise AWSCredentialsError(
                f"Failed to create AWS session: {str(e)}",
                credential_source="unknown"
            ) from e
    
    def _validate_credentials(self, session: boto3.Session) -> None:
        """
        Validate credentials by making a test call to STS.
        
        Args:
            session: boto3 session to validate
            
        Raises:
            AWSCredentialsError: If credentials are invalid
        """
        try:
            sts_client = session.client('sts')
            response = sts_client.get_caller_identity()
            
            user_arn = response.get('Arn', 'Unknown')
            account_id = response.get('Account', 'Unknown')
            
            logger.info(f"AWS credentials validated - Account: {account_id}, ARN: {user_arn}")
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code in ['InvalidUserID.NotFound', 'AccessDenied']:
                raise AWSCredentialsError(
                    f"Invalid AWS credentials: {e.response['Error']['Message']}",
                    credential_source=self._credentials_source
                ) from e
            else:
                raise AWSCredentialsError(
                    f"Failed to validate AWS credentials: {str(e)}",
                    credential_source=self._credentials_source
                ) from e
    
    def get_credentials_info(self) -> Dict[str, Any]:
        """
        Get information about current credentials.
        
        Returns:
            Dict containing credential information (without sensitive data)
        """
        if not self._session:
            return {'status': 'not_initialized'}
        
        try:
            sts_client = self._session.client('sts')
            response = sts_client.get_caller_identity()
            
            return {
                'status': 'valid',
                'source': self._credentials_source,
                'account_id': response.get('Account'),
                'user_arn': response.get('Arn'),
                'region': self.config.region
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'source': self._credentials_source,
                'error': str(e)
            }
    
    def refresh_credentials(self) -> None:
        """
        Refresh credentials by creating a new session.
        Useful for long-running applications with temporary credentials.
        """
        logger.info("Refreshing AWS credentials")
        self._session = None
        self._credentials_source = None
        self.get_session()  # This will create a new session
    
    def get_client(self, service_name: str, **kwargs) -> Any:
        """
        Get an AWS service client with proper configuration.
        
        Args:
            service_name: AWS service name (e.g., 's3', 'dynamodb')
            **kwargs: Additional client configuration
            
        Returns:
            Configured AWS service client
        """
        session = self.get_session()
        
        # Merge with default boto3 config
        client_config = self.config.get_boto3_config()
        client_config.update(kwargs)
        
        try:
            client = session.client(service_name, **client_config)
            logger.debug(f"Created {service_name} client for region {self.config.region}")
            return client
            
        except Exception as e:
            raise AWSCredentialsError(
                f"Failed to create {service_name} client: {str(e)}",
                credential_source=self._credentials_source
            ) from e
    
    def get_resource(self, service_name: str, **kwargs) -> Any:
        """
        Get an AWS service resource with proper configuration.
        
        Args:
            service_name: AWS service name (e.g., 's3', 'dynamodb')
            **kwargs: Additional resource configuration
            
        Returns:
            Configured AWS service resource
        """
        session = self.get_session()
        
        # Merge with default boto3 config
        resource_config = self.config.get_boto3_config()
        resource_config.update(kwargs)
        
        try:
            resource = session.resource(service_name, **resource_config)
            logger.debug(f"Created {service_name} resource for region {self.config.region}")
            return resource
            
        except Exception as e:
            raise AWSCredentialsError(
                f"Failed to create {service_name} resource: {str(e)}",
                credential_source=self._credentials_source
            ) from e
    
    def test_service_access(self, service_name: str) -> bool:
        """
        Test access to a specific AWS service.
        
        Args:
            service_name: AWS service name to test
            
        Returns:
            True if service is accessible, False otherwise
        """
        try:
            if service_name == 's3':
                client = self.get_client('s3')
                client.list_buckets()
            elif service_name == 'dynamodb':
                client = self.get_client('dynamodb')
                client.list_tables()
            elif service_name == 'mediaconvert':
                client = self.get_client('mediaconvert')
                client.list_jobs(MaxResults=1)
            elif service_name == 'cloudfront':
                client = self.get_client('cloudfront')
                client.list_distributions(MaxItems='1')
            else:
                # Generic test - just create the client
                self.get_client(service_name)
            
            logger.info(f"Successfully tested access to {service_name}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to access {service_name}: {str(e)}")
            return False
    
    def get_credential_source_priority(self) -> list:
        """
        Get the credential source priority order used by boto3.
        
        Returns:
            List of credential sources in priority order
        """
        return [
            "Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)",
            "AWS credentials file (~/.aws/credentials)",
            "AWS config file (~/.aws/config)",
            "IAM roles for Amazon EC2 instances",
            "IAM roles for Amazon ECS tasks",
            "IAM roles for AWS Lambda functions"
        ]
    
    def __repr__(self) -> str:
        """String representation of credentials manager."""
        return (
            f"AWSCredentialsManager(region='{self.config.region}', "
            f"profile='{self.config.profile}', "
            f"source='{self._credentials_source}')"
        )
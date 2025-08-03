"""
DynamoDB Metadata Management

Handles video and code metadata storage and retrieval using DynamoDB.
Implements table schema, CRUD operations, and query functionality.
"""

import boto3
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from botocore.exceptions import ClientError, BotoCoreError
from dataclasses import dataclass, asdict
from decimal import Decimal
import json

from .config import AWSConfig
from .exceptions import AWSMetadataError, AWSConfigurationError
from .credentials import AWSCredentialsManager

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Video metadata structure for DynamoDB storage."""
    video_id: str
    project_id: str
    title: str
    description: str = ""
    status: str = "created"  # created, code_ready, rendering, uploaded, transcoded
    version: int = 1
    current_version_id: str = "v1"
    
    # Timestamps
    created_timestamp: str = ""
    last_edited_timestamp: str = ""
    upload_completed_at: Optional[str] = None
    code_upload_completed_at: Optional[str] = None
    
    # S3 Paths
    s3_path_full_video: Optional[str] = None
    s3_path_code: Optional[str] = None
    chunk_s3_paths: Optional[Dict[str, str]] = None
    code_s3_paths: Optional[Dict[str, str]] = None
    transcoded_paths: Optional[Dict[str, str]] = None
    
    # Video Properties
    duration_seconds: Optional[float] = None
    resolution: Optional[str] = None
    file_size_bytes: Optional[int] = None
    scene_count: Optional[int] = None
    
    # Metadata
    tags: Optional[List[str]] = None
    custom_metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Set default timestamps if not provided."""
        if not self.created_timestamp:
            self.created_timestamp = datetime.now(timezone.utc).isoformat()
        if not self.last_edited_timestamp:
            self.last_edited_timestamp = self.created_timestamp
        if not self.current_version_id or self.current_version_id == "v1":
            self.current_version_id = f"v{self.version}"
        if self.tags is None:
            self.tags = []
        if self.custom_metadata is None:
            self.custom_metadata = {}


class DynamoDBMetadataOperations:
    """
    Core DynamoDB operations for video metadata management.
    """
    
    def __init__(self, config: AWSConfig, credentials_manager: AWSCredentialsManager):
        self.config = config
        self.credentials_manager = credentials_manager
        self.dynamodb = credentials_manager.get_resource('dynamodb')
        self.table = self.dynamodb.Table(config.metadata_table_name)
        
        logger.info(f"DynamoDB Metadata Operations initialized for table: {config.metadata_table_name}")
    
    async def create_video_record(self, video_id: str, project_id: str, 
                                 metadata: Dict[str, Any]) -> bool:
        """
        Create a new video record in DynamoDB.
        
        Args:
            video_id: Video identifier
            project_id: Project identifier
            metadata: Video metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare item for DynamoDB
            item = {
                'video_id': video_id,
                'project_id': project_id,
                'created_timestamp': datetime.now(timezone.utc).isoformat(),
                'last_edited_timestamp': datetime.now(timezone.utc).isoformat(),
                **metadata
            }
            
            # Convert any float values to Decimal for DynamoDB
            item = self._convert_floats_to_decimal(item)
            
            # Put item in table
            self.table.put_item(
                Item=item,
                ConditionExpression='attribute_not_exists(video_id)'
            )
            
            logger.info(f"Created video record: {video_id}")
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
                logger.warning(f"Video record already exists: {video_id}")
                return False
            else:
                logger.error(f"Failed to create video record: {e}")
                raise AWSMetadataError(
                    f"Failed to create video record: {str(e)}",
                    video_id=video_id,
                    operation="create_video_record"
                ) from e
        
        except Exception as e:
            logger.error(f"Failed to create video record: {e}")
            raise AWSMetadataError(
                f"Failed to create video record: {str(e)}",
                video_id=video_id,
                operation="create_video_record"
            ) from e
    
    async def update_video_metadata(self, video_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update video metadata in DynamoDB.
        
        Args:
            video_id: Video identifier
            metadata: Metadata to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert floats to Decimal
            metadata = self._convert_floats_to_decimal(metadata)
            
            # Build update expression
            update_expression = "SET last_edited_timestamp = :timestamp"
            expression_values = {':timestamp': datetime.now(timezone.utc).isoformat()}
            
            for key, value in metadata.items():
                if key not in ['video_id', 'project_id']:  # Don't update key attributes
                    update_expression += f", {key} = :{key}"
                    expression_values[f":{key}"] = value
            
            # Update item
            self.table.update_item(
                Key={'video_id': video_id},
                UpdateExpression=update_expression,
                ExpressionAttributeValues=expression_values
            )
            
            logger.info(f"Updated video metadata: {video_id}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to update video metadata: {e}")
            raise AWSMetadataError(
                f"Failed to update video metadata: {str(e)}",
                video_id=video_id,
                operation="update_video_metadata"
            ) from e
        
        except Exception as e:
            logger.error(f"Failed to update video metadata: {e}")
            raise AWSMetadataError(
                f"Failed to update video metadata: {str(e)}",
                video_id=video_id,
                operation="update_video_metadata"
            ) from e
    
    async def get_video_metadata(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Get video metadata from DynamoDB.
        
        Args:
            video_id: Video identifier
            
        Returns:
            Video metadata or None if not found
        """
        try:
            response = self.table.get_item(
                Key={'video_id': video_id}
            )
            
            if 'Item' in response:
                item = response['Item']
                # Convert Decimal back to float
                item = self._convert_decimal_to_float(item)
                logger.info(f"Retrieved video metadata: {video_id}")
                return item
            else:
                logger.info(f"Video metadata not found: {video_id}")
                return None
                
        except ClientError as e:
            logger.error(f"Failed to get video metadata: {e}")
            raise AWSMetadataError(
                f"Failed to get video metadata: {str(e)}",
                video_id=video_id,
                operation="get_video_metadata"
            ) from e
        
        except Exception as e:
            logger.error(f"Failed to get video metadata: {e}")
            raise AWSMetadataError(
                f"Failed to get video metadata: {str(e)}",
                video_id=video_id,
                operation="get_video_metadata"
            ) from e
    
    async def list_project_videos(self, project_id: str) -> List[Dict[str, Any]]:
        """
        List all videos for a project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            List of video metadata
        """
        try:
            # Use scan with filter for simplicity (in production, consider GSI)
            response = self.table.scan(
                FilterExpression='project_id = :project_id',
                ExpressionAttributeValues={':project_id': project_id}
            )
            
            items = response.get('Items', [])
            # Convert Decimal back to float
            items = [self._convert_decimal_to_float(item) for item in items]
            
            logger.info(f"Retrieved {len(items)} videos for project: {project_id}")
            return items
            
        except ClientError as e:
            logger.error(f"Failed to list project videos: {e}")
            raise AWSMetadataError(
                f"Failed to list project videos: {str(e)}",
                operation="list_project_videos"
            ) from e
        
        except Exception as e:
            logger.error(f"Failed to list project videos: {e}")
            raise AWSMetadataError(
                f"Failed to list project videos: {str(e)}",
                operation="list_project_videos"
            ) from e
    
    def _convert_floats_to_decimal(self, obj: Any) -> Any:
        """Convert float values to Decimal for DynamoDB compatibility."""
        if isinstance(obj, float):
            return Decimal(str(obj))
        elif isinstance(obj, dict):
            return {k: self._convert_floats_to_decimal(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_floats_to_decimal(item) for item in obj]
        else:
            return obj
    
    def _convert_decimal_to_float(self, obj: Any) -> Any:
        """Convert Decimal values back to float."""
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_decimal_to_float(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_decimal_to_float(item) for item in obj]
        else:
            return obj


class DynamoDBTableManager:
    """Manages DynamoDB table creation and schema operations."""
    
    def __init__(self, config: AWSConfig):
        self.config = config
        self.credentials_manager = AWSCredentialsManager(config)
        self.dynamodb = self.credentials_manager.get_resource('dynamodb')
        self.client = self.credentials_manager.get_client('dynamodb')
        self.table_name = config.metadata_table_name
        
        logger.info(f"DynamoDB Table Manager initialized for table: {self.table_name}")
    
    def create_table_if_not_exists(self) -> bool:
        """
        Create VideoMetadata table if it doesn't exist.
        
        Returns:
            True if table was created or already exists, False on error
        """
        try:
            # Check if table already exists
            existing_tables = self.client.list_tables()['TableNames']
            if self.table_name in existing_tables:
                logger.info(f"Table {self.table_name} already exists")
                return True
            
            # Create table
            table = self.dynamodb.create_table(
                TableName=self.table_name,
                KeySchema=[
                    {
                        'AttributeName': 'video_id',
                        'KeyType': 'HASH'
                    }
                ],
                AttributeDefinitions=[
                    {
                        'AttributeName': 'video_id',
                        'AttributeType': 'S'
                    }
                ],
                BillingMode='PAY_PER_REQUEST'
            )
            
            # Wait for table to be created
            table.wait_until_exists()
            logger.info(f"Created DynamoDB table: {self.table_name}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to create table: {e}")
            return False
        
        except Exception as e:
            logger.error(f"Failed to create table: {e}")
            return False
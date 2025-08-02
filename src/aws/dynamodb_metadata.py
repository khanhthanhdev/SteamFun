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
        Create VideoMetadata table with proper key schema and indexes if it doesn't exist.
        
        Returns:
            bool: True if table was created or already exists, False on error
        """
        try:
            # Check if table already exists
            existing_tables = self.client.list_tables()['TableNames']
            if self.table_name in existing_tables:
                logger.info(f"Table {self.table_name} already exists")
                return True
            
            # Create table with proper schema
            table_definition = {
                'TableName': self.table_name,
                'KeySchema': [
                    {
                        'AttributeName': 'video_id',
                        'KeyType': 'HASH'  # Partition key
                    }
                ],
                'AttributeDefinitions': [
                    {
                        'AttributeName': 'video_id',
                        'AttributeType': 'S'
                    },
                    {
                        'AttributeName': 'project_id',
                        'AttributeType': 'S'
                    },
                    {
                        'AttributeName': 'created_timestamp',
                        'AttributeType': 'S'
                    },
                    {
                        'AttributeName': 'status',
                        'AttributeType': 'S'
                    },
                    {
                        'AttributeName': 'current_version_id',
                        'AttributeType': 'S'
                    }
                ],
                'GlobalSecondaryIndexes': [
                    {
                        'IndexName': 'ProjectIndex',
                        'KeySchema': [
                            {
                                'AttributeName': 'project_id',
                                'KeyType': 'HASH'
                            },
                            {
                                'AttributeName': 'created_timestamp',
                                'KeyType': 'RANGE'
                            }
                        ],
                        'Projection': {
                            'ProjectionType': 'ALL'
                        }
                    },
                    {
                        'IndexName': 'StatusIndex',
                        'KeySchema': [
                            {
                                'AttributeName': 'status',
                                'KeyType': 'HASH'
                            },
                            {
                                'AttributeName': 'created_timestamp',
                                'KeyType': 'RANGE'
                            }
                        ],
                        'Projection': {
                            'ProjectionType': 'ALL'
                        }
                    }
                ],
                'LocalSecondaryIndexes': [
                    {
                        'IndexName': 'VersionIndex',
                        'KeySchema': [
                            {
                                'AttributeName': 'video_id',
                                'KeyType': 'HASH'
                            },
                            {
                                'AttributeName': 'current_version_id',
                                'KeyType': 'RANGE'
                            }
                        ],
                        'Projection': {
                            'ProjectionType': 'ALL'
                        }
                    }
                ],
                'BillingMode': 'PAY_PER_REQUEST'  # On-demand pricing
            }
            
            # Create the table
            response = self.client.create_table(**table_definition)
            logger.info(f"Creating table {self.table_name}...")
            
            # Wait for table to be active
            waiter = self.client.get_waiter('table_exists')
            waiter.wait(
                TableName=self.table_name,
                WaiterConfig={
                    'Delay': 5,
                    'MaxAttempts': 20
                }
            )
            
            logger.info(f"Table {self.table_name} created successfully")
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ResourceInUseException':
                logger.info(f"Table {self.table_name} already exists")
                return True
            else:
                logger.error(f"Failed to create table {self.table_name}: {e}")
                raise AWSMetadataError(f"Failed to create DynamoDB table: {e}")
        except Exception as e:
            logger.error(f"Unexpected error creating table: {e}")
            raise AWSMetadataError(f"Unexpected error creating table: {e}")
    
    def delete_table(self) -> bool:
        """
        Delete the VideoMetadata table (for testing/cleanup).
        
        Returns:
            bool: True if table was deleted successfully
        """
        try:
            self.client.delete_table(TableName=self.table_name)
            
            # Wait for table to be deleted
            waiter = self.client.get_waiter('table_not_exists')
            waiter.wait(
                TableName=self.table_name,
                WaiterConfig={
                    'Delay': 5,
                    'MaxAttempts': 20
                }
            )
            
            logger.info(f"Table {self.table_name} deleted successfully")
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ResourceNotFoundException':
                logger.info(f"Table {self.table_name} does not exist")
                return True
            else:
                logger.error(f"Failed to delete table {self.table_name}: {e}")
                raise AWSMetadataError(f"Failed to delete DynamoDB table: {e}")
    
    def describe_table(self) -> Dict[str, Any]:
        """
        Get table description and status.
        
        Returns:
            Dict containing table description
        """
        try:
            response = self.client.describe_table(TableName=self.table_name)
            return response['Table']
        except ClientError as e:
            logger.error(f"Failed to describe table {self.table_name}: {e}")
            raise AWSMetadataError(f"Failed to describe table: {e}")


class DynamoDBMetadataOperations:
    """Core DynamoDB operations for video metadata management."""
    
    def __init__(self, config: AWSConfig):
        self.config = config
        self.credentials_manager = AWSCredentialsManager(config)
        self.dynamodb = self.credentials_manager.get_resource('dynamodb')
        self.table = self.dynamodb.Table(config.metadata_table_name)
        self.table_name = config.metadata_table_name
        
        logger.info(f"DynamoDB Metadata Operations initialized for table: {self.table_name}")
    
    def create_video_record(self, video_metadata: VideoMetadata) -> bool:
        """
        Create a new video record in DynamoDB.
        
        Args:
            video_metadata: VideoMetadata object containing video information
            
        Returns:
            bool: True if record was created successfully
            
        Raises:
            AWSMetadataError: If creation fails
        """
        try:
            # Convert dataclass to dict and handle None values
            item_data = asdict(video_metadata)
            
            # Remove None values to avoid DynamoDB issues
            item_data = {k: v for k, v in item_data.items() if v is not None}
            
            # Convert lists and dicts to DynamoDB format if needed
            if 'tags' in item_data and item_data['tags']:
                item_data['tags'] = item_data['tags']
            
            if 'custom_metadata' in item_data and item_data['custom_metadata']:
                item_data['custom_metadata'] = json.dumps(item_data['custom_metadata'])
            
            # Use condition expression to prevent overwriting existing records
            response = self.table.put_item(
                Item=item_data,
                ConditionExpression='attribute_not_exists(video_id)'
            )
            
            logger.info(f"Created video record: {video_metadata.video_id}")
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ConditionalCheckFailedException':
                logger.error(f"Video record {video_metadata.video_id} already exists")
                raise AWSMetadataError(f"Video record {video_metadata.video_id} already exists")
            else:
                logger.error(f"Failed to create video record {video_metadata.video_id}: {e}")
                raise AWSMetadataError(f"Failed to create video record: {e}")
        except Exception as e:
            logger.error(f"Unexpected error creating video record: {e}")
            raise AWSMetadataError(f"Unexpected error creating video record: {e}")
    
    def update_metadata(self, video_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update video metadata with proper UpdateExpression syntax.
        
        Args:
            video_id: Video ID to update
            updates: Dictionary of fields to update
            
        Returns:
            bool: True if update was successful
            
        Raises:
            AWSMetadataError: If update fails
        """
        try:
            if not updates:
                logger.warning(f"No updates provided for video {video_id}")
                return True
            
            # Build UpdateExpression and ExpressionAttributeValues
            update_expression_parts = []
            expression_attribute_values = {}
            expression_attribute_names = {}
            
            # Always update last_edited_timestamp
            updates['last_edited_timestamp'] = datetime.now(timezone.utc).isoformat()
            
            for key, value in updates.items():
                if key == 'video_id':  # Don't update the primary key
                    continue
                
                # Handle reserved keywords by using expression attribute names
                attr_name = f"#{key}"
                attr_value = f":{key}"
                
                expression_attribute_names[attr_name] = key
                
                # Handle special data types
                if key == 'custom_metadata' and isinstance(value, dict):
                    expression_attribute_values[attr_value] = json.dumps(value)
                elif key in ['chunk_s3_paths', 'code_s3_paths', 'transcoded_paths'] and isinstance(value, dict):
                    expression_attribute_values[attr_value] = value
                elif key == 'tags' and isinstance(value, list):
                    expression_attribute_values[attr_value] = value
                elif isinstance(value, float):
                    # Convert float to Decimal for DynamoDB compatibility
                    expression_attribute_values[attr_value] = Decimal(str(value))
                else:
                    expression_attribute_values[attr_value] = value
                
                update_expression_parts.append(f"{attr_name} = {attr_value}")
            
            if not update_expression_parts:
                logger.warning(f"No valid updates for video {video_id}")
                return True
            
            update_expression = "SET " + ", ".join(update_expression_parts)
            
            # Perform the update
            response = self.table.update_item(
                Key={'video_id': video_id},
                UpdateExpression=update_expression,
                ExpressionAttributeNames=expression_attribute_names,
                ExpressionAttributeValues=expression_attribute_values,
                ConditionExpression='attribute_exists(video_id)',  # Ensure record exists
                ReturnValues='UPDATED_NEW'
            )
            
            logger.info(f"Updated metadata for video: {video_id}")
            logger.debug(f"Updated attributes: {list(updates.keys())}")
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ConditionalCheckFailedException':
                logger.error(f"Video record {video_id} does not exist")
                raise AWSMetadataError(f"Video record {video_id} does not exist")
            else:
                logger.error(f"Failed to update metadata for {video_id}: {e}")
                raise AWSMetadataError(f"Failed to update metadata: {e}")
        except Exception as e:
            logger.error(f"Unexpected error updating metadata: {e}")
            raise AWSMetadataError(f"Unexpected error updating metadata: {e}")
    
    def get_video_metadata(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve video metadata by video_id.
        
        Args:
            video_id: Video ID to retrieve
            
        Returns:
            Dict containing video metadata or None if not found
            
        Raises:
            AWSMetadataError: If retrieval fails
        """
        try:
            response = self.table.get_item(
                Key={'video_id': video_id}
            )
            
            if 'Item' not in response:
                logger.info(f"Video metadata not found: {video_id}")
                return None
            
            item = response['Item']
            
            # Parse JSON fields back to objects
            if 'custom_metadata' in item and isinstance(item['custom_metadata'], str):
                try:
                    item['custom_metadata'] = json.loads(item['custom_metadata'])
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse custom_metadata for {video_id}")
                    item['custom_metadata'] = {}
            
            logger.info(f"Retrieved metadata for video: {video_id}")
            return item
            
        except ClientError as e:
            logger.error(f"Failed to get metadata for {video_id}: {e}")
            raise AWSMetadataError(f"Failed to get metadata: {e}")
        except Exception as e:
            logger.error(f"Unexpected error getting metadata: {e}")
            raise AWSMetadataError(f"Unexpected error getting metadata: {e}")
    
    def query_by_project(self, project_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Query videos by project_id using ProjectIndex.
        
        Args:
            project_id: Project ID to query
            limit: Maximum number of items to return
            
        Returns:
            List of video metadata dictionaries
            
        Raises:
            AWSMetadataError: If query fails
        """
        try:
            query_params = {
                'IndexName': 'ProjectIndex',
                'KeyConditionExpression': 'project_id = :project_id',
                'ExpressionAttributeValues': {
                    ':project_id': project_id
                },
                'ScanIndexForward': False  # Sort by created_timestamp descending (newest first)
            }
            
            if limit:
                query_params['Limit'] = limit
            
            response = self.table.query(**query_params)
            
            items = response.get('Items', [])
            
            # Parse JSON fields for each item
            for item in items:
                if 'custom_metadata' in item and isinstance(item['custom_metadata'], str):
                    try:
                        item['custom_metadata'] = json.loads(item['custom_metadata'])
                    except json.JSONDecodeError:
                        item['custom_metadata'] = {}
            
            logger.info(f"Retrieved {len(items)} videos for project: {project_id}")
            return items
            
        except ClientError as e:
            logger.error(f"Failed to query videos for project {project_id}: {e}")
            raise AWSMetadataError(f"Failed to query videos: {e}")
        except Exception as e:
            logger.error(f"Unexpected error querying videos: {e}")
            raise AWSMetadataError(f"Unexpected error querying videos: {e}")
    
    def query_by_status(self, status: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Query videos by status using StatusIndex.
        
        Args:
            status: Status to query (created, code_ready, rendering, uploaded, transcoded)
            limit: Maximum number of items to return
            
        Returns:
            List of video metadata dictionaries
            
        Raises:
            AWSMetadataError: If query fails
        """
        try:
            query_params = {
                'IndexName': 'StatusIndex',
                'KeyConditionExpression': '#status = :status',
                'ExpressionAttributeNames': {
                    '#status': 'status'  # 'status' is a reserved keyword
                },
                'ExpressionAttributeValues': {
                    ':status': status
                },
                'ScanIndexForward': False  # Sort by created_timestamp descending
            }
            
            if limit:
                query_params['Limit'] = limit
            
            response = self.table.query(**query_params)
            
            items = response.get('Items', [])
            
            # Parse JSON fields for each item
            for item in items:
                if 'custom_metadata' in item and isinstance(item['custom_metadata'], str):
                    try:
                        item['custom_metadata'] = json.loads(item['custom_metadata'])
                    except json.JSONDecodeError:
                        item['custom_metadata'] = {}
            
            logger.info(f"Retrieved {len(items)} videos with status: {status}")
            return items
            
        except ClientError as e:
            logger.error(f"Failed to query videos with status {status}: {e}")
            raise AWSMetadataError(f"Failed to query videos: {e}")
        except Exception as e:
            logger.error(f"Unexpected error querying videos: {e}")
            raise AWSMetadataError(f"Unexpected error querying videos: {e}")
    
    def delete_video_record(self, video_id: str) -> bool:
        """
        Delete a video record from DynamoDB.
        
        Args:
            video_id: Video ID to delete
            
        Returns:
            bool: True if deletion was successful
            
        Raises:
            AWSMetadataError: If deletion fails
        """
        try:
            response = self.table.delete_item(
                Key={'video_id': video_id},
                ConditionExpression='attribute_exists(video_id)',
                ReturnValues='ALL_OLD'
            )
            
            if 'Attributes' in response:
                logger.info(f"Deleted video record: {video_id}")
                return True
            else:
                logger.warning(f"Video record {video_id} was not found for deletion")
                return False
                
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ConditionalCheckFailedException':
                logger.error(f"Video record {video_id} does not exist")
                raise AWSMetadataError(f"Video record {video_id} does not exist")
            else:
                logger.error(f"Failed to delete video record {video_id}: {e}")
                raise AWSMetadataError(f"Failed to delete video record: {e}")
        except Exception as e:
            logger.error(f"Unexpected error deleting video record: {e}")
            raise AWSMetadataError(f"Unexpected error deleting video record: {e}")


def generate_video_id() -> str:
    """Generate a unique video ID using UUID4."""
    return str(uuid.uuid4())


def generate_project_id() -> str:
    """Generate a unique project ID using UUID4."""
    return str(uuid.uuid4())
"""
Metadata Service with Batch Operations

Provides high-level metadata management with batch operations, conflict resolution,
and efficient bulk operations for video and code metadata.
"""

import boto3
import logging
import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, Tuple
from botocore.exceptions import ClientError, BotoCoreError
from dataclasses import dataclass, asdict
from decimal import Decimal
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .config import AWSConfig
from .exceptions import AWSMetadataError, AWSConfigurationError
from .credentials import AWSCredentialsManager
from .dynamodb_metadata import VideoMetadata, DynamoDBMetadataOperations, DynamoDBTableManager

logger = logging.getLogger(__name__)


@dataclass
class BatchOperationResult:
    """Result of a batch operation."""
    successful_items: List[str]
    failed_items: List[Tuple[str, str]]  # (item_id, error_message)
    total_processed: int
    success_count: int
    failure_count: int
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_processed == 0:
            return 0.0
        return (self.success_count / self.total_processed) * 100.0


class MetadataService:
    """
    High-level metadata service providing batch operations, conflict resolution,
    and efficient bulk operations for video and code metadata.
    """
    
    def __init__(self, config: AWSConfig):
        self.config = config
        self.credentials_manager = AWSCredentialsManager(config)
        
        # Initialize DynamoDB components
        self.table_manager = DynamoDBTableManager(config)
        self.operations = DynamoDBMetadataOperations(config, self.credentials_manager)
        
        # Get table reference
        self.table = self.credentials_manager.get_resource('dynamodb').Table(config.metadata_table_name)
        
        # Ensure table exists
        self.table_manager.create_table_if_not_exists()
        
        logger.info(f"Metadata Service initialized for table: {config.metadata_table_name}")
    
    async def create_video_record(self, video_id: str, project_id: str, 
                                 metadata: Dict[str, Any]) -> bool:
        """
        Create a new video record with metadata.
        
        Args:
            video_id: Video identifier
            project_id: Project identifier
            metadata: Video metadata
            
        Returns:
            True if successful, False otherwise
        """
        return await self.operations.create_video_record(video_id, project_id, metadata)
    
    async def update_video_metadata(self, video_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update video metadata.
        
        Args:
            video_id: Video identifier
            metadata: Metadata to update
            
        Returns:
            True if successful, False otherwise
        """
        return await self.operations.update_video_metadata(video_id, metadata)
    
    async def get_video_metadata(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Get video metadata.
        
        Args:
            video_id: Video identifier
            
        Returns:
            Video metadata or None if not found
        """
        return await self.operations.get_video_metadata(video_id)
    
    async def list_project_videos(self, project_id: str) -> List[Dict[str, Any]]:
        """
        List all videos for a project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            List of video metadata
        """
        return await self.operations.list_project_videos(project_id)
    
    async def batch_create_video_records(self, video_records: List[Dict[str, Any]]) -> BatchOperationResult:
        """
        Create multiple video records in batch.
        
        Args:
            video_records: List of video record dictionaries
            
        Returns:
            BatchOperationResult with success/failure details
        """
        successful_items = []
        failed_items = []
        
        for record in video_records:
            try:
                video_id = record.get('video_id')
                project_id = record.get('project_id')
                metadata = {k: v for k, v in record.items() if k not in ['video_id', 'project_id']}
                
                success = await self.create_video_record(video_id, project_id, metadata)
                if success:
                    successful_items.append(video_id)
                else:
                    failed_items.append((video_id, "Record already exists"))
                    
            except Exception as e:
                video_id = record.get('video_id', 'unknown')
                failed_items.append((video_id, str(e)))
        
        return BatchOperationResult(
            successful_items=successful_items,
            failed_items=failed_items,
            total_processed=len(video_records),
            success_count=len(successful_items),
            failure_count=len(failed_items)
        )
    
    async def batch_update_video_metadata(self, updates: List[Dict[str, Any]]) -> BatchOperationResult:
        """
        Update multiple video records in batch.
        
        Args:
            updates: List of update dictionaries with video_id and metadata
            
        Returns:
            BatchOperationResult with success/failure details
        """
        successful_items = []
        failed_items = []
        
        for update in updates:
            try:
                video_id = update.get('video_id')
                metadata = {k: v for k, v in update.items() if k != 'video_id'}
                
                success = await self.update_video_metadata(video_id, metadata)
                if success:
                    successful_items.append(video_id)
                else:
                    failed_items.append((video_id, "Update failed"))
                    
            except Exception as e:
                video_id = update.get('video_id', 'unknown')
                failed_items.append((video_id, str(e)))
        
        return BatchOperationResult(
            successful_items=successful_items,
            failed_items=failed_items,
            total_processed=len(updates),
            success_count=len(successful_items),
            failure_count=len(failed_items)
        )
    
    async def search_videos_by_status(self, status: str, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search videos by status.
        
        Args:
            status: Video status to search for
            project_id: Optional project filter
            
        Returns:
            List of matching video metadata
        """
        try:
            if project_id:
                # Filter by both project and status
                response = self.table.scan(
                    FilterExpression='#status = :status AND project_id = :project_id',
                    ExpressionAttributeNames={'#status': 'status'},
                    ExpressionAttributeValues={
                        ':status': status,
                        ':project_id': project_id
                    }
                )
            else:
                # Filter by status only
                response = self.table.scan(
                    FilterExpression='#status = :status',
                    ExpressionAttributeNames={'#status': 'status'},
                    ExpressionAttributeValues={':status': status}
                )
            
            items = response.get('Items', [])
            # Convert Decimal back to float
            items = [self.operations._convert_decimal_to_float(item) for item in items]
            
            logger.info(f"Found {len(items)} videos with status: {status}")
            return items
            
        except ClientError as e:
            logger.error(f"Failed to search videos by status: {e}")
            raise AWSMetadataError(
                f"Failed to search videos by status: {str(e)}",
                operation="search_videos_by_status"
            ) from e
    
    async def get_video_statistics(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get video statistics.
        
        Args:
            project_id: Optional project filter
            
        Returns:
            Dictionary with video statistics
        """
        try:
            if project_id:
                videos = await self.list_project_videos(project_id)
            else:
                # Get all videos (use scan for simplicity)
                response = self.table.scan()
                videos = [self.operations._convert_decimal_to_float(item) for item in response.get('Items', [])]
            
            # Calculate statistics
            total_videos = len(videos)
            status_counts = {}
            total_duration = 0
            total_size = 0
            
            for video in videos:
                status = video.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
                
                if video.get('duration_seconds'):
                    total_duration += video['duration_seconds']
                
                if video.get('file_size_bytes'):
                    total_size += video['file_size_bytes']
            
            return {
                'total_videos': total_videos,
                'status_counts': status_counts,
                'total_duration_seconds': total_duration,
                'total_size_bytes': total_size,
                'average_duration_seconds': total_duration / total_videos if total_videos > 0 else 0,
                'average_size_bytes': total_size / total_videos if total_videos > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get video statistics: {e}")
            raise AWSMetadataError(
                f"Failed to get video statistics: {str(e)}",
                operation="get_video_statistics"
            ) from e
    
    async def cleanup_old_records(self, days_old: int = 30) -> int:
        """
        Clean up old video records.
        
        Args:
            days_old: Number of days old to consider for cleanup
            
        Returns:
            Number of records cleaned up
        """
        try:
            # Calculate cutoff date
            cutoff_date = datetime.now(timezone.utc).replace(
                day=datetime.now(timezone.utc).day - days_old
            ).isoformat()
            
            # Scan for old records
            response = self.table.scan(
                FilterExpression='created_timestamp < :cutoff',
                ExpressionAttributeValues={':cutoff': cutoff_date}
            )
            
            old_records = response.get('Items', [])
            deleted_count = 0
            
            # Delete old records
            for record in old_records:
                try:
                    self.table.delete_item(
                        Key={'video_id': record['video_id']}
                    )
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete record {record['video_id']}: {e}")
            
            logger.info(f"Cleaned up {deleted_count} old video records")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old records: {e}")
            raise AWSMetadataError(
                f"Failed to cleanup old records: {str(e)}",
                operation="cleanup_old_records"
            ) from e
    
    def __repr__(self) -> str:
        """String representation of the metadata service."""
        return (
            f"MetadataService(table='{self.config.metadata_table_name}', "
            f"region='{self.config.region}')"
        )
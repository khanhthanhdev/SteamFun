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


@dataclass
class ConflictResolutionStrategy:
    """Configuration for conflict resolution strategies."""
    strategy: str = "latest_wins"  # latest_wins, merge_fields, manual_resolution
    merge_fields: Optional[List[str]] = None  # Fields to merge for merge_fields strategy
    preserve_fields: Optional[List[str]] = None  # Fields to preserve during conflicts
    
    def __post_init__(self):
        if self.merge_fields is None:
            self.merge_fields = []
        if self.preserve_fields is None:
            self.preserve_fields = ['created_timestamp', 'video_id', 'project_id']


class MetadataService:
    """
    High-level metadata service with batch operations and conflict resolution.
    
    Provides efficient bulk operations, concurrent update handling, and advanced
    querying capabilities for video and code metadata management.
    """
    
    def __init__(self, config: AWSConfig, max_workers: int = 5):
        """
        Initialize MetadataService.
        
        Args:
            config: AWS configuration
            max_workers: Maximum number of concurrent workers for batch operations
        """
        self.config = config
        self.max_workers = max_workers
        
        # Initialize underlying operations
        self.operations = DynamoDBMetadataOperations(config)
        self.table_manager = DynamoDBTableManager(config)
        
        # Initialize DynamoDB resources for batch operations
        self.credentials_manager = AWSCredentialsManager(config)
        self.dynamodb = self.credentials_manager.get_resource('dynamodb')
        self.table = self.dynamodb.Table(config.metadata_table_name)
        
        # Thread pool for concurrent operations
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Conflict resolution configuration
        self.conflict_resolution = ConflictResolutionStrategy()
        
        # Rate limiting for batch operations
        self.batch_delay = 0.1  # Delay between batch operations to avoid throttling
        self.max_retries = 3
        
        logger.info(f"MetadataService initialized with {max_workers} workers")
    
    def set_conflict_resolution_strategy(self, strategy: ConflictResolutionStrategy):
        """Set the conflict resolution strategy for concurrent updates."""
        self.conflict_resolution = strategy
        logger.info(f"Conflict resolution strategy set to: {strategy.strategy}")
    
    async def batch_create_video_records(self, video_metadata_list: List[VideoMetadata]) -> BatchOperationResult:
        """
        Create multiple video records using batch operations.
        
        Args:
            video_metadata_list: List of VideoMetadata objects to create
            
        Returns:
            BatchOperationResult with success/failure details
        """
        if not video_metadata_list:
            return BatchOperationResult([], [], 0, 0, 0)
        
        logger.info(f"Starting batch creation of {len(video_metadata_list)} video records")
        
        successful_items = []
        failed_items = []
        
        # Use DynamoDB batch_writer for efficient bulk operations
        try:
            with self.table.batch_writer() as batch:
                for video_metadata in video_metadata_list:
                    try:
                        # Convert dataclass to dict and clean up None values
                        item_data = asdict(video_metadata)
                        item_data = {k: v for k, v in item_data.items() if v is not None}
                        
                        # Handle special data types
                        if 'custom_metadata' in item_data and item_data['custom_metadata']:
                            item_data['custom_metadata'] = json.dumps(item_data['custom_metadata'])
                        
                        # Add to batch
                        batch.put_item(Item=item_data)
                        successful_items.append(video_metadata.video_id)
                        
                        # Small delay to avoid overwhelming DynamoDB
                        await asyncio.sleep(self.batch_delay)
                        
                    except Exception as e:
                        error_msg = f"Failed to prepare item {video_metadata.video_id}: {e}"
                        logger.error(error_msg)
                        failed_items.append((video_metadata.video_id, error_msg))
            
            logger.info(f"Batch creation completed: {len(successful_items)} successful, {len(failed_items)} failed")
            
        except ClientError as e:
            logger.error(f"Batch creation failed: {e}")
            # If batch fails, mark all as failed
            failed_items = [(vm.video_id, str(e)) for vm in video_metadata_list]
            successful_items = []
        
        return BatchOperationResult(
            successful_items=successful_items,
            failed_items=failed_items,
            total_processed=len(video_metadata_list),
            success_count=len(successful_items),
            failure_count=len(failed_items)
        )
    
    async def batch_update_metadata(self, updates: Dict[str, Dict[str, Any]]) -> BatchOperationResult:
        """
        Update multiple video records with conflict resolution.
        
        Args:
            updates: Dictionary mapping video_id to update fields
            
        Returns:
            BatchOperationResult with success/failure details
        """
        if not updates:
            return BatchOperationResult([], [], 0, 0, 0)
        
        logger.info(f"Starting batch update of {len(updates)} video records")
        
        successful_items = []
        failed_items = []
        
        # Process updates concurrently with conflict resolution
        tasks = []
        for video_id, update_data in updates.items():
            task = asyncio.create_task(
                self._update_with_conflict_resolution(video_id, update_data)
            )
            tasks.append((video_id, task))
        
        # Wait for all updates to complete
        for video_id, task in tasks:
            try:
                success = await task
                if success:
                    successful_items.append(video_id)
                else:
                    failed_items.append((video_id, "Update failed"))
            except Exception as e:
                error_msg = f"Update failed for {video_id}: {e}"
                logger.error(error_msg)
                failed_items.append((video_id, error_msg))
        
        logger.info(f"Batch update completed: {len(successful_items)} successful, {len(failed_items)} failed")
        
        return BatchOperationResult(
            successful_items=successful_items,
            failed_items=failed_items,
            total_processed=len(updates),
            success_count=len(successful_items),
            failure_count=len(failed_items)
        )
    
    async def _update_with_conflict_resolution(self, video_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update a single record with conflict resolution.
        
        Args:
            video_id: Video ID to update
            update_data: Fields to update
            
        Returns:
            bool: True if update was successful
        """
        max_attempts = self.max_retries
        
        for attempt in range(max_attempts):
            try:
                # Get current record for conflict detection
                current_record = self.operations.get_video_metadata(video_id)
                if not current_record:
                    logger.error(f"Video record {video_id} not found for update")
                    return False
                
                # Apply conflict resolution strategy
                resolved_updates = self._resolve_conflicts(current_record, update_data)
                
                # Perform the update with optimistic locking
                success = await self._atomic_update(video_id, resolved_updates, current_record)
                
                if success:
                    return True
                
                # If update failed due to conflict, retry with exponential backoff
                if attempt < max_attempts - 1:
                    delay = (2 ** attempt) * 0.1  # Exponential backoff
                    await asyncio.sleep(delay)
                    logger.warning(f"Retrying update for {video_id} (attempt {attempt + 2})")
                
            except Exception as e:
                logger.error(f"Error updating {video_id} on attempt {attempt + 1}: {e}")
                if attempt == max_attempts - 1:
                    raise
        
        return False
    
    def _resolve_conflicts(self, current_record: Dict[str, Any], update_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve conflicts between current record and update data.
        
        Args:
            current_record: Current record from DynamoDB
            update_data: New data to update
            
        Returns:
            Dict with resolved update data
        """
        resolved_updates = update_data.copy()
        
        if self.conflict_resolution.strategy == "latest_wins":
            # Simply use the new data (default behavior)
            pass
        
        elif self.conflict_resolution.strategy == "merge_fields":
            # Merge specific fields while preserving others
            for field in self.conflict_resolution.merge_fields:
                if field in current_record and field in update_data:
                    current_value = current_record[field]
                    new_value = update_data[field]
                    
                    # Handle different data types for merging
                    if isinstance(current_value, dict) and isinstance(new_value, dict):
                        # Merge dictionaries
                        merged = current_value.copy()
                        merged.update(new_value)
                        resolved_updates[field] = merged
                    elif isinstance(current_value, list) and isinstance(new_value, list):
                        # Merge lists (remove duplicates)
                        merged = list(set(current_value + new_value))
                        resolved_updates[field] = merged
        
        elif self.conflict_resolution.strategy == "manual_resolution":
            # For manual resolution, we would typically queue this for human review
            # For now, we'll use latest_wins as fallback
            logger.warning(f"Manual resolution required for conflicts - using latest_wins fallback")
        
        # Always preserve certain fields
        for field in self.conflict_resolution.preserve_fields:
            if field in current_record and field not in resolved_updates:
                resolved_updates[field] = current_record[field]
        
        return resolved_updates
    
    async def _atomic_update(self, video_id: str, update_data: Dict[str, Any], 
                           current_record: Dict[str, Any]) -> bool:
        """
        Perform atomic update with optimistic locking.
        
        Args:
            video_id: Video ID to update
            update_data: Data to update
            current_record: Current record for version checking
            
        Returns:
            bool: True if update was successful
        """
        try:
            # Add version check to prevent concurrent modification
            # Use last_edited_timestamp as a version field
            current_timestamp = current_record.get('last_edited_timestamp')
            
            # Build update expression with condition
            update_expression_parts = []
            expression_attribute_values = {}
            expression_attribute_names = {}
            
            # Always update last_edited_timestamp
            update_data['last_edited_timestamp'] = datetime.now(timezone.utc).isoformat()
            
            for key, value in update_data.items():
                if key == 'video_id':
                    continue
                
                attr_name = f"#{key}"
                attr_value = f":{key}"
                
                expression_attribute_names[attr_name] = key
                
                # Handle special data types
                if key == 'custom_metadata' and isinstance(value, dict):
                    expression_attribute_values[attr_value] = json.dumps(value)
                elif isinstance(value, float):
                    expression_attribute_values[attr_value] = Decimal(str(value))
                else:
                    expression_attribute_values[attr_value] = value
                
                update_expression_parts.append(f"{attr_name} = {attr_value}")
            
            update_expression = "SET " + ", ".join(update_expression_parts)
            
            # Condition to check that record hasn't been modified
            condition_expression = "attribute_exists(video_id)"
            if current_timestamp:
                condition_expression += " AND last_edited_timestamp = :current_timestamp"
                expression_attribute_values[':current_timestamp'] = current_timestamp
            
            # Perform atomic update
            self.table.update_item(
                Key={'video_id': video_id},
                UpdateExpression=update_expression,
                ExpressionAttributeNames=expression_attribute_names,
                ExpressionAttributeValues=expression_attribute_values,
                ConditionExpression=condition_expression
            )
            
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ConditionalCheckFailedException':
                logger.warning(f"Concurrent modification detected for {video_id}")
                return False
            else:
                logger.error(f"Atomic update failed for {video_id}: {e}")
                raise
    
    async def batch_query_by_project(self, project_ids: List[str], 
                                   limit_per_project: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Query multiple projects concurrently.
        
        Args:
            project_ids: List of project IDs to query
            limit_per_project: Maximum items per project
            
        Returns:
            Dictionary mapping project_id to list of video metadata
        """
        logger.info(f"Batch querying {len(project_ids)} projects")
        
        results = {}
        
        # Create concurrent tasks for each project query
        tasks = []
        for project_id in project_ids:
            task = asyncio.create_task(
                asyncio.to_thread(
                    self.operations.query_by_project, 
                    project_id, 
                    limit_per_project
                )
            )
            tasks.append((project_id, task))
        
        # Wait for all queries to complete
        for project_id, task in tasks:
            try:
                project_videos = await task
                results[project_id] = project_videos
            except Exception as e:
                logger.error(f"Failed to query project {project_id}: {e}")
                results[project_id] = []
        
        total_videos = sum(len(videos) for videos in results.values())
        logger.info(f"Batch query completed: {total_videos} total videos across {len(project_ids)} projects")
        
        return results
    
    async def create_project_index_query(self, project_id: str, 
                                       start_date: Optional[str] = None,
                                       end_date: Optional[str] = None,
                                       status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Advanced project query with date range and status filtering.
        
        Args:
            project_id: Project ID to query
            start_date: Start date filter (ISO format)
            end_date: End date filter (ISO format)
            status_filter: Status to filter by
            
        Returns:
            List of filtered video metadata
        """
        try:
            # Build query parameters
            query_params = {
                'IndexName': 'ProjectIndex',
                'KeyConditionExpression': 'project_id = :project_id',
                'ExpressionAttributeValues': {
                    ':project_id': project_id
                }
            }
            
            # Add date range filtering
            if start_date or end_date:
                if start_date and end_date:
                    query_params['KeyConditionExpression'] += ' AND created_timestamp BETWEEN :start_date AND :end_date'
                    query_params['ExpressionAttributeValues'][':start_date'] = start_date
                    query_params['ExpressionAttributeValues'][':end_date'] = end_date
                elif start_date:
                    query_params['KeyConditionExpression'] += ' AND created_timestamp >= :start_date'
                    query_params['ExpressionAttributeValues'][':start_date'] = start_date
                elif end_date:
                    query_params['KeyConditionExpression'] += ' AND created_timestamp <= :end_date'
                    query_params['ExpressionAttributeValues'][':end_date'] = end_date
            
            # Add status filtering
            if status_filter:
                query_params['FilterExpression'] = '#status = :status'
                query_params['ExpressionAttributeNames'] = {'#status': 'status'}
                query_params['ExpressionAttributeValues'][':status'] = status_filter
            
            # Execute query
            response = self.table.query(**query_params)
            items = response.get('Items', [])
            
            # Parse JSON fields
            for item in items:
                if 'custom_metadata' in item and isinstance(item['custom_metadata'], str):
                    try:
                        item['custom_metadata'] = json.loads(item['custom_metadata'])
                    except json.JSONDecodeError:
                        item['custom_metadata'] = {}
            
            logger.info(f"Advanced project query returned {len(items)} items for project {project_id}")
            return items
            
        except ClientError as e:
            logger.error(f"Advanced project query failed: {e}")
            raise AWSMetadataError(f"Advanced project query failed: {e}")
    
    async def create_version_index_query(self, video_id: str) -> List[Dict[str, Any]]:
        """
        Query version history for a specific video using VersionIndex.
        
        Args:
            video_id: Video ID to get version history for
            
        Returns:
            List of version metadata ordered by version
        """
        try:
            # Query using VersionIndex (Local Secondary Index)
            query_params = {
                'IndexName': 'VersionIndex',
                'KeyConditionExpression': 'video_id = :video_id',
                'ExpressionAttributeValues': {
                    ':video_id': video_id
                },
                'ScanIndexForward': False  # Sort by current_version_id descending (newest first)
            }
            
            response = self.table.query(**query_params)
            items = response.get('Items', [])
            
            # Parse JSON fields for each item
            for item in items:
                if 'custom_metadata' in item and isinstance(item['custom_metadata'], str):
                    try:
                        item['custom_metadata'] = json.loads(item['custom_metadata'])
                    except json.JSONDecodeError:
                        item['custom_metadata'] = {}
            
            logger.info(f"Retrieved {len(items)} version records for video: {video_id}")
            return items
            
        except ClientError as e:
            logger.error(f"Version query failed for {video_id}: {e}")
            # Fallback to getting current record only
            try:
                current_record = self.operations.get_video_metadata(video_id)
                if current_record:
                    return [current_record]
                return []
            except Exception as fallback_error:
                logger.error(f"Fallback version query also failed: {fallback_error}")
                return []
        except Exception as e:
            logger.error(f"Unexpected error in version query for {video_id}: {e}")
            return []
    
    async def query_video_versions_by_range(self, video_id: str, 
                                          start_version: Optional[str] = None,
                                          end_version: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Query version history for a specific video within a version range.
        
        Args:
            video_id: Video ID to get version history for
            start_version: Starting version (e.g., "v1")
            end_version: Ending version (e.g., "v5")
            
        Returns:
            List of version metadata within the specified range
        """
        try:
            # Build query parameters for version range
            query_params = {
                'IndexName': 'VersionIndex',
                'KeyConditionExpression': 'video_id = :video_id',
                'ExpressionAttributeValues': {
                    ':video_id': video_id
                },
                'ScanIndexForward': True  # Sort by current_version_id ascending
            }
            
            # Add version range filtering
            if start_version and end_version:
                query_params['KeyConditionExpression'] += ' AND current_version_id BETWEEN :start_version AND :end_version'
                query_params['ExpressionAttributeValues'][':start_version'] = start_version
                query_params['ExpressionAttributeValues'][':end_version'] = end_version
            elif start_version:
                query_params['KeyConditionExpression'] += ' AND current_version_id >= :start_version'
                query_params['ExpressionAttributeValues'][':start_version'] = start_version
            elif end_version:
                query_params['KeyConditionExpression'] += ' AND current_version_id <= :end_version'
                query_params['ExpressionAttributeValues'][':end_version'] = end_version
            
            response = self.table.query(**query_params)
            items = response.get('Items', [])
            
            # Parse JSON fields for each item
            for item in items:
                if 'custom_metadata' in item and isinstance(item['custom_metadata'], str):
                    try:
                        item['custom_metadata'] = json.loads(item['custom_metadata'])
                    except json.JSONDecodeError:
                        item['custom_metadata'] = {}
            
            logger.info(f"Retrieved {len(items)} version records for video {video_id} in range {start_version}-{end_version}")
            return items
            
        except ClientError as e:
            logger.error(f"Version range query failed for {video_id}: {e}")
            raise AWSMetadataError(f"Version range query failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in version range query: {e}")
            raise AWSMetadataError(f"Unexpected error in version range query: {e}")
    
    async def get_metadata_statistics(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metadata statistics for monitoring and optimization.
        
        Args:
            project_id: Optional project ID to filter statistics
            
        Returns:
            Dictionary with various statistics
        """
        try:
            stats = {
                'total_videos': 0,
                'status_breakdown': {},
                'storage_usage': {
                    'total_video_size': 0,
                    'total_code_files': 0
                },
                'recent_activity': {
                    'videos_created_today': 0,
                    'videos_updated_today': 0
                }
            }
            
            # Query videos (limited scope for performance)
            if project_id:
                videos = self.operations.query_by_project(project_id, limit=1000)
            else:
                # For global stats, we'd need to scan (expensive) or use aggregation
                # For now, return stats for recent videos only
                videos = self.operations.query_by_status('uploaded', limit=100)
                videos.extend(self.operations.query_by_status('transcoded', limit=100))
            
            stats['total_videos'] = len(videos)
            
            # Calculate status breakdown
            for video in videos:
                status = video.get('status', 'unknown')
                stats['status_breakdown'][status] = stats['status_breakdown'].get(status, 0) + 1
                
                # Calculate storage usage
                if video.get('file_size_bytes'):
                    stats['storage_usage']['total_video_size'] += video['file_size_bytes']
                
                if video.get('code_s3_paths'):
                    stats['storage_usage']['total_code_files'] += len(video['code_s3_paths'])
            
            # Calculate recent activity (simplified)
            today = datetime.now(timezone.utc).date().isoformat()
            for video in videos:
                created_date = video.get('created_timestamp', '')[:10]  # Extract date part
                updated_date = video.get('last_edited_timestamp', '')[:10]
                
                if created_date == today:
                    stats['recent_activity']['videos_created_today'] += 1
                if updated_date == today:
                    stats['recent_activity']['videos_updated_today'] += 1
            
            logger.info(f"Generated metadata statistics: {stats['total_videos']} videos analyzed")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to generate metadata statistics: {e}")
            raise AWSMetadataError(f"Failed to generate statistics: {e}")
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
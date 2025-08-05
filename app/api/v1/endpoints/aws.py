"""
AWS Integration API Endpoints

Provides REST API endpoints for AWS operations including:
- S3 file upload and download
- DynamoDB operations
- Video and code management
- Metadata operations
"""

import os
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse

from app.models.schemas.aws import (
    S3UploadRequest,
    S3UploadResponse,
    S3DownloadRequest,
    S3DownloadResponse,
    VideoUploadRequest,
    VideoUploadResponse,
    CodeUploadRequest,
    CodeUploadResponse,
    CodeDownloadRequest,
    CodeDownloadResponse,
    VideoProjectRequest,
    VideoProjectResponse,
    VideoMetadataUpdateRequest,
    VideoDetailsResponse,
    ProjectVideosResponse,
    CodeVersionsResponse,
    BatchUploadRequest,
    BatchUploadResponse,
    IntegratedUploadRequest,
    IntegratedUploadResponse,
    DynamoDBOperationRequest,
    DynamoDBOperationResponse,
    AWSHealthResponse,
    AWSConfigResponse,
    S3ListResponse,
    S3DeleteRequest,
    S3DeleteResponse,
    TranscodingJobRequest,
    TranscodingJobResponse,
    TranscodingJobStatusResponse,
    TranscodingJobListResponse,
    TranscodingJobCancelRequest,
    TranscodingJobCancelResponse,
    IntegratedTranscodingUploadRequest,
    IntegratedTranscodingUploadResponse,
    MediaConvertHealthResponse
)
from app.services.aws_service import AWSService
from app.core.aws.exceptions import AWSIntegrationError, AWSS3Error, AWSMetadataError
router= APIRouter(prefix="/aws", tags=["AWS Integration"])

# Dependency to get AWS service instance
def get_aws_service() -> AWSService:
    """Get AWS service instance."""
    return AWSService()


# S3 Upload/Download Endpoints

@router.post("/s3/upload", response_model=S3UploadResponse)
async def upload_file_to_s3(
    request: S3UploadRequest,
    aws_service: AWSService = Depends(get_aws_service)
):
    """
    Upload a file to S3.
    
    - **file_path**: Local file path to upload
    - **project_id**: Project identifier
    - **video_id**: Video identifier
    - **scene_number**: Optional scene number (0 for combined video)
    - **version**: File version
    - **metadata**: Additional metadata
    - **enable_object_lock**: Enable S3 Object Lock
    - **content_type**: File content type
    """
    try:
        if not os.path.exists(request.file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File not found: {request.file_path}"
            )
        
        # Upload video file
        s3_url = await aws_service.upload_single_video(
            file_path=request.file_path,
            project_id=request.project_id,
            video_id=request.video_id,
            version=request.version,
            metadata=request.metadata
        )
        
        # Get file info
        file_size = os.path.getsize(request.file_path)
        
        return S3UploadResponse(
            s3_url=s3_url,
            bucket=aws_service.config.video_bucket_name,
            key=f"{request.project_id}/{request.video_id}/v{request.version}/scene_{request.scene_number or 0}.mp4",
            file_size=file_size,
            content_type=request.content_type or "video/mp4",
            upload_time=datetime.utcnow(),
            metadata=request.metadata
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {request.file_path}"
        )
    except AWSIntegrationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AWS upload failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


@router.post("/s3/download", response_model=S3DownloadResponse)
async def get_s3_download_url(
    request: S3DownloadRequest,
    aws_service: AWSService = Depends(get_aws_service)
):
    """
    Get a pre-signed download URL for an S3 file.
    
    - **project_id**: Project identifier
    - **video_id**: Video identifier
    - **scene_number**: Optional scene number
    - **version**: File version
    - **file_type**: Type of file (video, code, etc.)
    """
    try:
        if request.file_type == "video":
            download_url = aws_service.get_video_download_url(
                project_id=request.project_id,
                video_id=request.video_id,
                scene_number=request.scene_number or 0,
                version=request.version
            )
        elif request.file_type == "code":
            download_url = aws_service.get_code_download_url(
                project_id=request.project_id,
                video_id=request.video_id,
                version=request.version,
                scene_number=request.scene_number
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {request.file_type}"
            )
        
        # Pre-signed URLs typically expire in 1 hour
        expires_at = datetime.utcnow() + timedelta(hours=1)
        
        return S3DownloadResponse(
            download_url=download_url,
            expires_at=expires_at,
            content_type="video/mp4" if request.file_type == "video" else "text/plain"
        )
        
    except AWSIntegrationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AWS download URL generation failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Download URL generation failed: {str(e)}"
        )


# Video Management Endpoints

@router.post("/video/upload", response_model=VideoUploadResponse)
async def upload_video(
    request: VideoUploadRequest,
    background_tasks: BackgroundTasks,
    aws_service: AWSService = Depends(get_aws_service)
):
    """
    Upload a video file with metadata creation.
    
    - **file_path**: Local video file path
    - **project_id**: Project identifier
    - **video_id**: Video identifier
    - **title**: Video title
    - **description**: Video description
    - **version**: Video version
    - **scene_number**: Optional scene number
    - **metadata**: Additional metadata
    """
    try:
        if not os.path.exists(request.file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Video file not found: {request.file_path}"
            )
        
        # Create video project first
        await aws_service.create_video_project(
            video_id=request.video_id,
            project_id=request.project_id,
            title=request.title,
            description=request.description,
            metadata=request.metadata
        )
        
        # Upload video
        s3_url = await aws_service.upload_single_video(
            file_path=request.file_path,
            project_id=request.project_id,
            video_id=request.video_id,
            version=request.version,
            metadata=request.metadata
        )
        
        # Get file info
        file_size = os.path.getsize(request.file_path)
        
        # Update metadata with upload info
        await aws_service.update_video_status(
            video_id=request.video_id,
            status="uploaded",
            metadata={
                "s3_url": s3_url,
                "file_size": file_size,
                "upload_completed_at": datetime.utcnow().isoformat()
            }
        )
        
        # Generate download URL
        download_url = aws_service.get_video_download_url(
            project_id=request.project_id,
            video_id=request.video_id,
            scene_number=request.scene_number or 0,
            version=request.version
        )
        
        return VideoUploadResponse(
            video_id=request.video_id,
            project_id=request.project_id,
            s3_url=s3_url,
            download_url=download_url,
            file_size=file_size,
            upload_time=datetime.utcnow(),
            status="uploaded"
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video file not found: {request.file_path}"
        )
    except AWSIntegrationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video upload failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video upload failed: {str(e)}"
        )


@router.post("/video/project", response_model=VideoProjectResponse)
async def create_video_project(
    request: VideoProjectRequest,
    aws_service: AWSService = Depends(get_aws_service)
):
    """
    Create a new video project with metadata.
    
    - **video_id**: Video identifier
    - **project_id**: Project identifier
    - **title**: Video title
    - **description**: Video description
    - **metadata**: Additional metadata
    - **tags**: Project tags
    """
    try:
        success = await aws_service.create_video_project(
            video_id=request.video_id,
            project_id=request.project_id,
            title=request.title,
            description=request.description,
            metadata=request.metadata
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Video project already exists: {request.video_id}"
            )
        
        return VideoProjectResponse(
            video_id=request.video_id,
            project_id=request.project_id,
            title=request.title,
            description=request.description,
            status="created",
            created_at=datetime.utcnow(),
            metadata=request.metadata,
            tags=request.tags
        )
        
    except AWSIntegrationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Project creation failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Project creation failed: {str(e)}"
        )


@router.get("/video/{video_id}", response_model=VideoDetailsResponse)
async def get_video_details(
    video_id: str,
    aws_service: AWSService = Depends(get_aws_service)
):
    """
    Get detailed video information.
    
    - **video_id**: Video identifier
    """
    try:
        video_details = await aws_service.get_video_details(video_id)
        
        if not video_details:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Video not found: {video_id}"
            )
        
        # Generate download URLs if S3 paths exist
        download_urls = {}
        if video_details.get("s3_path_full_video"):
            download_urls["video"] = aws_service.get_video_download_url(
                project_id=video_details["project_id"],
                video_id=video_id,
                version=video_details.get("version", 1)
            )
        
        if video_details.get("s3_path_code"):
            download_urls["code"] = aws_service.get_code_download_url(
                project_id=video_details["project_id"],
                video_id=video_id,
                version=video_details.get("version", 1)
            )
        
        return VideoDetailsResponse(
            video_id=video_id,
            project_id=video_details["project_id"],
            title=video_details.get("title", ""),
            description=video_details.get("description", ""),
            status=video_details.get("status", "unknown"),
            version=video_details.get("version", 1),
            created_at=datetime.fromisoformat(video_details["created_timestamp"]),
            updated_at=datetime.fromisoformat(video_details["last_edited_timestamp"]) if video_details.get("last_edited_timestamp") else None,
            s3_urls={
                "video": video_details.get("s3_path_full_video"),
                "code": video_details.get("s3_path_code")
            },
            download_urls=download_urls,
            file_sizes={
                "video": video_details.get("file_size_bytes"),
                "code": video_details.get("code_file_size")
            },
            metadata=video_details.get("metadata"),
            tags=video_details.get("tags")
        )
        
    except AWSIntegrationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get video details: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get video details: {str(e)}"
        )


@router.put("/video/{video_id}/metadata")
async def update_video_metadata(
    video_id: str,
    request: VideoMetadataUpdateRequest,
    aws_service: AWSService = Depends(get_aws_service)
):
    """
    Update video metadata.
    
    - **video_id**: Video identifier
    - **status**: Optional video status
    - **title**: Optional video title
    - **description**: Optional video description
    - **metadata**: Additional metadata to update
    - **tags**: Optional project tags
    """
    try:
        # Build update data from non-None fields
        update_data = {}
        if request.status is not None:
            update_data["status"] = request.status
        if request.title is not None:
            update_data["title"] = request.title
        if request.description is not None:
            update_data["description"] = request.description
        if request.tags is not None:
            update_data["tags"] = request.tags
        if request.metadata is not None:
            update_data.update(request.metadata)
        
        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No update data provided"
            )
        
        success = await aws_service.update_video_status(
            video_id=video_id,
            status=request.status or "updated",
            metadata=update_data
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Video not found: {video_id}"
            )
        
        return {"message": "Video metadata updated successfully", "video_id": video_id}
        
    except AWSIntegrationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metadata update failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metadata update failed: {str(e)}"
        )


@router.get("/project/{project_id}/videos", response_model=ProjectVideosResponse)
async def list_project_videos(
    project_id: str,
    aws_service: AWSService = Depends(get_aws_service)
):
    """
    List all videos in a project.
    
    - **project_id**: Project identifier
    """
    try:
        videos_data = await aws_service.list_project_videos(project_id)
        
        videos = []
        total_size = 0
        
        for video_data in videos_data:
            # Generate download URLs
            download_urls = {}
            if video_data.get("s3_path_full_video"):
                download_urls["video"] = aws_service.get_video_download_url(
                    project_id=project_id,
                    video_id=video_data["video_id"],
                    version=video_data.get("version", 1)
                )
            
            if video_data.get("s3_path_code"):
                download_urls["code"] = aws_service.get_code_download_url(
                    project_id=project_id,
                    video_id=video_data["video_id"],
                    version=video_data.get("version", 1)
                )
            
            video_size = video_data.get("file_size_bytes", 0)
            if video_size:
                total_size += video_size
            
            video_response = VideoDetailsResponse(
                video_id=video_data["video_id"],
                project_id=project_id,
                title=video_data.get("title", ""),
                description=video_data.get("description", ""),
                status=video_data.get("status", "unknown"),
                version=video_data.get("version", 1),
                created_at=datetime.fromisoformat(video_data["created_timestamp"]),
                updated_at=datetime.fromisoformat(video_data["last_edited_timestamp"]) if video_data.get("last_edited_timestamp") else None,
                s3_urls={
                    "video": video_data.get("s3_path_full_video"),
                    "code": video_data.get("s3_path_code")
                },
                download_urls=download_urls,
                file_sizes={
                    "video": video_data.get("file_size_bytes"),
                    "code": video_data.get("code_file_size")
                },
                metadata=video_data.get("metadata"),
                tags=video_data.get("tags")
            )
            videos.append(video_response)
        
        return ProjectVideosResponse(
            project_id=project_id,
            videos=videos,
            total_count=len(videos),
            total_size=total_size if total_size > 0 else None
        )
        
    except AWSIntegrationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list project videos: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list project videos: {str(e)}"
        )


# Code Management Endpoints

@router.post("/code/upload", response_model=CodeUploadResponse)
async def upload_code(
    request: CodeUploadRequest,
    aws_service: AWSService = Depends(get_aws_service)
):
    """
    Upload code content to S3.
    
    - **code_content**: Code content as string
    - **project_id**: Project identifier
    - **video_id**: Video identifier
    - **version**: Code version
    - **scene_number**: Optional scene number
    - **language**: Programming language
    - **filename**: Original filename
    - **metadata**: Additional metadata
    - **enable_object_lock**: Enable S3 Object Lock
    """
    try:
        s3_url = await aws_service.upload_code_file(
            code_content=request.code_content,
            project_id=request.project_id,
            video_id=request.video_id,
            version=request.version,
            scene_number=request.scene_number,
            metadata=request.metadata,
            enable_object_lock=request.enable_object_lock
        )
        
        # Generate download URL
        download_url = aws_service.get_code_download_url(
            project_id=request.project_id,
            video_id=request.video_id,
            version=request.version,
            scene_number=request.scene_number
        )
        
        return CodeUploadResponse(
            video_id=request.video_id,
            project_id=request.project_id,
            s3_url=s3_url,
            download_url=download_url,
            version=request.version,
            file_size=len(request.code_content.encode('utf-8')),
            upload_time=datetime.utcnow(),
            language=request.language
        )
        
    except AWSIntegrationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Code upload failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Code upload failed: {str(e)}"
        )


@router.post("/code/download", response_model=CodeDownloadResponse)
async def download_code(
    request: CodeDownloadRequest,
    aws_service: AWSService = Depends(get_aws_service)
):
    """
    Download code content from S3.
    
    - **project_id**: Project identifier
    - **video_id**: Video identifier
    - **version**: Code version
    - **scene_number**: Optional scene number
    """
    try:
        code_content = await aws_service.download_code_file(
            project_id=request.project_id,
            video_id=request.video_id,
            version=request.version,
            scene_number=request.scene_number
        )
        
        return CodeDownloadResponse(
            code_content=code_content,
            project_id=request.project_id,
            video_id=request.video_id,
            version=request.version,
            file_size=len(code_content.encode('utf-8')),
            last_modified=datetime.utcnow()
        )
        
    except AWSIntegrationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Code download failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Code download failed: {str(e)}"
        )


@router.get("/code/{project_id}/{video_id}/versions", response_model=CodeVersionsResponse)
async def list_code_versions(
    project_id: str,
    video_id: str,
    aws_service: AWSService = Depends(get_aws_service)
):
    """
    List all code versions for a video.
    
    - **project_id**: Project identifier
    - **video_id**: Video identifier
    """
    try:
        versions_data = await aws_service.list_code_versions_for_video(project_id, video_id)
        
        latest_version = max([v["version"] for v in versions_data]) if versions_data else 1
        
        return CodeVersionsResponse(
            project_id=project_id,
            video_id=video_id,
            versions=versions_data,
            total_count=len(versions_data),
            latest_version=latest_version
        )
        
    except AWSIntegrationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list code versions: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list code versions: {str(e)}"
        )


# Integrated Operations

@router.post("/integrated/upload", response_model=IntegratedUploadResponse)
async def integrated_upload(
    request: IntegratedUploadRequest,
    background_tasks: BackgroundTasks,
    aws_service: AWSService = Depends(get_aws_service)
):
    """
    Upload video and code together with metadata creation.
    
    - **video_file_path**: Local video file path
    - **code_content**: Code content
    - **project_id**: Project identifier
    - **video_id**: Video identifier
    - **title**: Video title
    - **description**: Video description
    - **version**: Version number
    - **metadata**: Additional metadata
    """
    try:
        if not os.path.exists(request.video_file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Video file not found: {request.video_file_path}"
            )
        
        result = await aws_service.upload_video_with_code(
            video_file_path=request.video_file_path,
            code_content=request.code_content,
            project_id=request.project_id,
            video_id=request.video_id,
            title=request.title,
            description=request.description,
            version=request.version
        )
        
        # Generate download URLs
        video_download_url = aws_service.get_video_download_url(
            project_id=request.project_id,
            video_id=request.video_id,
            version=request.version
        )
        
        code_download_url = aws_service.get_code_download_url(
            project_id=request.project_id,
            video_id=request.video_id,
            version=request.version
        )
        
        # Calculate total size
        video_size = os.path.getsize(request.video_file_path)
        code_size = len(request.code_content.encode('utf-8'))
        total_size = video_size + code_size
        
        return IntegratedUploadResponse(
            video_id=request.video_id,
            project_id=request.project_id,
            video_url=result["video_url"],
            code_url=result["code_url"],
            video_download_url=video_download_url,
            code_download_url=code_download_url,
            status=result["status"],
            upload_time=datetime.fromisoformat(result["timestamp"]),
            total_size=total_size
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video file not found: {request.video_file_path}"
        )
    except AWSIntegrationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Integrated upload failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Integrated upload failed: {str(e)}"
        )


# DynamoDB Operations

@router.post("/dynamodb/operation", response_model=DynamoDBOperationResponse)
async def execute_dynamodb_operation(
    request: DynamoDBOperationRequest,
    aws_service: AWSService = Depends(get_aws_service)
):
    """
    Execute a DynamoDB operation.
    
    - **table_name**: DynamoDB table name
    - **operation**: Operation type (get, put, update, delete, query, scan)
    - **key**: Primary key for get/update/delete operations
    - **item**: Item data for put operations
    - **update_expression**: Update expression for update operations
    - **condition_expression**: Condition expression
    - **expression_attribute_values**: Expression attribute values
    - **query_params**: Query parameters
    """
    try:
        start_time = datetime.utcnow()
        
        # Get DynamoDB resource through AWS service
        dynamodb = aws_service.integration_service.credentials_manager.get_resource('dynamodb')
        table = dynamodb.Table(request.table_name)
        
        result = None
        items = None
        count = None
        consumed_capacity = None
        error = None
        success = True
        
        try:
            if request.operation == "get":
                if not request.key:
                    raise ValueError("Key is required for get operation")
                response = table.get_item(Key=request.key)
                result = response.get('Item')
                consumed_capacity = response.get('ConsumedCapacity')
                
            elif request.operation == "put":
                if not request.item:
                    raise ValueError("Item is required for put operation")
                kwargs = {'Item': request.item}
                if request.condition_expression:
                    kwargs['ConditionExpression'] = request.condition_expression
                if request.expression_attribute_values:
                    kwargs['ExpressionAttributeValues'] = request.expression_attribute_values
                response = table.put_item(**kwargs)
                consumed_capacity = response.get('ConsumedCapacity')
                result = {"message": "Item created successfully"}
                
            elif request.operation == "update":
                if not request.key or not request.update_expression:
                    raise ValueError("Key and update_expression are required for update operation")
                kwargs = {
                    'Key': request.key,
                    'UpdateExpression': request.update_expression,
                    'ReturnValues': 'ALL_NEW'
                }
                if request.condition_expression:
                    kwargs['ConditionExpression'] = request.condition_expression
                if request.expression_attribute_values:
                    kwargs['ExpressionAttributeValues'] = request.expression_attribute_values
                response = table.update_item(**kwargs)
                result = response.get('Attributes')
                consumed_capacity = response.get('ConsumedCapacity')
                
            elif request.operation == "delete":
                if not request.key:
                    raise ValueError("Key is required for delete operation")
                kwargs = {'Key': request.key}
                if request.condition_expression:
                    kwargs['ConditionExpression'] = request.condition_expression
                if request.expression_attribute_values:
                    kwargs['ExpressionAttributeValues'] = request.expression_attribute_values
                response = table.delete_item(**kwargs)
                consumed_capacity = response.get('ConsumedCapacity')
                result = {"message": "Item deleted successfully"}
                
            elif request.operation == "query":
                if not request.query_params:
                    raise ValueError("Query parameters are required for query operation")
                kwargs = request.query_params.copy()
                if request.expression_attribute_values:
                    kwargs['ExpressionAttributeValues'] = request.expression_attribute_values
                response = table.query(**kwargs)
                items = response.get('Items', [])
                count = response.get('Count', 0)
                consumed_capacity = response.get('ConsumedCapacity')
                
            elif request.operation == "scan":
                kwargs = request.query_params or {}
                if request.expression_attribute_values:
                    kwargs['ExpressionAttributeValues'] = request.expression_attribute_values
                response = table.scan(**kwargs)
                items = response.get('Items', [])
                count = response.get('Count', 0)
                consumed_capacity = response.get('ConsumedCapacity')
                
            else:
                raise ValueError(f"Unsupported operation: {request.operation}")
                
        except Exception as op_error:
            success = False
            error = str(op_error)
            
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return DynamoDBOperationResponse(
            operation=request.operation,
            table_name=request.table_name,
            success=success,
            result=result,
            items=items,
            count=count,
            consumed_capacity=consumed_capacity,
            processing_time=processing_time,
            error=error
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"DynamoDB operation failed: {str(e)}"
        )


# Health and Configuration Endpoints

@router.get("/health", response_model=AWSHealthResponse)
async def get_aws_health(
    aws_service: AWSService = Depends(get_aws_service)
):
    """
    Get AWS service health status.
    """
    try:
        health_data = await aws_service.get_service_health()
        
        return AWSHealthResponse(
            overall_status=health_data["overall_status"],
            services=health_data["services"],
            region=aws_service.config.region,
            timestamp=datetime.fromisoformat(health_data["timestamp"]),
            response_times=health_data.get("response_times", {}),
            error_counts=health_data.get("error_counts", {})
        )
        
    except Exception as e:
        return AWSHealthResponse(
            overall_status="unhealthy",
            services={"error": {"status": "unhealthy", "error": str(e)}},
            region=aws_service.config.region if hasattr(aws_service, 'config') else "unknown",
            timestamp=datetime.utcnow(),
            response_times={},
            error_counts={}
        )


@router.get("/config", response_model=AWSConfigResponse)
async def get_aws_config(
    aws_service: AWSService = Depends(get_aws_service)
):
    """
    Get AWS service configuration.
    """
    try:
        config_data = aws_service.get_service_configuration()
        
        return AWSConfigResponse(
            region=aws_service.config.region,
            s3_bucket=aws_service.config.video_bucket_name,
            dynamodb_table=aws_service.config.metadata_table_name,
            enabled_services=["S3", "DynamoDB"],  # Based on available services
            configuration=config_data,
            credentials_configured=aws_service.is_enabled(),
            last_updated=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get AWS configuration: {str(e)}"
        )


# Batch Operations

@router.post("/batch/upload", response_model=BatchUploadResponse)
async def batch_upload_files(
    request: BatchUploadRequest,
    background_tasks: BackgroundTasks,
    aws_service: AWSService = Depends(get_aws_service)
):
    """
    Upload multiple files in batch.
    
    - **files**: List of files to upload
    - **project_id**: Project identifier
    - **video_id**: Video identifier
    - **metadata**: Common metadata for all files
    """
    try:
        start_time = datetime.utcnow()
        
        # Validate all files exist
        for file_info in request.files:
            file_path = file_info.get('file_path')
            if file_path and not os.path.exists(file_path):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"File not found: {file_path}"
                )
        
        # Upload files using the service
        result = await aws_service.upload_video_files(
            video_files=request.files
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return BatchUploadResponse(
            total_files=result["total_files"],
            successful_uploads=result["successful_uploads"],
            failed_uploads=result["failed_uploads"],
            success_rate=result["success_rate"],
            upload_results=result["upload_urls"],
            processing_time=processing_time,
            timestamp=datetime.fromisoformat(result["timestamp"])
        )
        
    except AWSIntegrationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch upload failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch upload failed: {str(e)}"
        )


# MediaConvert Transcoding Endpoints

@router.post("/transcoding/job", response_model=TranscodingJobResponse)
async def create_transcoding_job(
    request: TranscodingJobRequest,
    background_tasks: BackgroundTasks,
    aws_service: AWSService = Depends(get_aws_service)
):
    """
    Create a transcoding job for adaptive bitrate streaming.
    
    - **input_s3_path**: S3 path to input video file
    - **output_s3_prefix**: S3 prefix for output files
    - **video_id**: Video identifier
    - **project_id**: Project identifier
    - **output_formats**: Output formats (HLS, DASH, MP4)
    - **quality_levels**: Quality levels (1080p, 720p, 480p)
    - **metadata**: Additional metadata
    - **wait_for_completion**: Whether to wait for job completion
    - **timeout_minutes**: Timeout in minutes for completion wait
    """
    try:
        result = await aws_service.create_transcoding_job(request)
        
        return result
        
    except AWSIntegrationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcoding job creation failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcoding job creation failed: {str(e)}"
        )


@router.get("/transcoding/job/{job_id}", response_model=TranscodingJobStatusResponse)
async def get_transcoding_job_status(
    job_id: str,
    aws_service: AWSService = Depends(get_aws_service)
):
    """
    Get the status of a transcoding job.
    
    - **job_id**: MediaConvert job ID
    """
    try:
        result = await aws_service.get_transcoding_job_status(job_id)
        
        return result
        
    except AWSIntegrationError as e:
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transcoding job not found: {job_id}"
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get transcoding job status: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get transcoding job status: {str(e)}"
        )


@router.get("/transcoding/jobs", response_model=TranscodingJobListResponse)
async def list_transcoding_jobs(
    status: Optional[str] = None,
    max_results: int = 20,
    aws_service: AWSService = Depends(get_aws_service)
):
    """
    List transcoding jobs.
    
    - **status**: Optional status filter (SUBMITTED, PROGRESSING, COMPLETE, CANCELED, ERROR)
    - **max_results**: Maximum number of results (1-100)
    """
    try:
        if max_results < 1 or max_results > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="max_results must be between 1 and 100"
            )
        
        result = await aws_service.list_transcoding_jobs(status, max_results)
        
        return result
        
    except AWSIntegrationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list transcoding jobs: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list transcoding jobs: {str(e)}"
        )


@router.delete("/transcoding/job/{job_id}", response_model=TranscodingJobCancelResponse)
async def cancel_transcoding_job(
    job_id: str,
    reason: Optional[str] = None,
    aws_service: AWSService = Depends(get_aws_service)
):
    """
    Cancel a transcoding job.
    
    - **job_id**: MediaConvert job ID to cancel
    - **reason**: Optional reason for cancellation
    """
    try:
        request = TranscodingJobCancelRequest(job_id=job_id, reason=reason)
        result = await aws_service.cancel_transcoding_job(request)
        
        return result
        
    except AWSIntegrationError as e:
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transcoding job not found: {job_id}"
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel transcoding job: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel transcoding job: {str(e)}"
        )


@router.post("/transcoding/upload", response_model=IntegratedTranscodingUploadResponse)
async def upload_video_with_transcoding(
    request: IntegratedTranscodingUploadRequest,
    background_tasks: BackgroundTasks,
    aws_service: AWSService = Depends(get_aws_service)
):
    """
    Upload video and trigger transcoding in a single operation.
    
    - **video_file_path**: Local video file path
    - **code_content**: Optional code content
    - **project_id**: Project identifier
    - **video_id**: Video identifier
    - **title**: Video title
    - **description**: Video description
    - **version**: Version number
    - **output_formats**: Transcoding output formats
    - **quality_levels**: Transcoding quality levels
    - **wait_for_transcoding**: Whether to wait for transcoding completion
    - **enable_cdn**: Whether to enable CloudFront CDN
    - **metadata**: Additional metadata
    """
    try:
        if not os.path.exists(request.video_file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Video file not found: {request.video_file_path}"
            )
        
        result = await aws_service.upload_video_with_transcoding(request)
        
        return result
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video file not found: {request.video_file_path}"
        )
    except AWSIntegrationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Integrated transcoding upload failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Integrated transcoding upload failed: {str(e)}"
        )


@router.get("/transcoding/health", response_model=MediaConvertHealthResponse)
async def get_mediaconvert_health(
    aws_service: AWSService = Depends(get_aws_service)
):
    """
    Get MediaConvert service health status.
    """
    try:
        result = await aws_service.get_mediaconvert_health()
        
        return result
        
    except Exception as e:
        # Return unhealthy status instead of raising exception
        return MediaConvertHealthResponse(
            status='unhealthy',
            region=aws_service.config.region,
            enabled=False,
            timestamp=datetime.utcnow(),
            error=str(e)
        )
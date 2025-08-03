"""
Video Generation API Endpoints

Provides REST API endpoints for video generation operations including:
- Video creation and generation
- Status monitoring
- File upload and download
- Configuration management
"""

import os
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse

from app.models.schemas.video import (
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoStatusResponse,
    VideoConfigRequest,
    VideoConfigResponse,
    VideoListResponse,
    VideoDetailsResponse,
    SceneOutlineRequest,
    SceneOutlineResponse
)
from app.models.enums import VideoStatus
from app.services.video_service import VideoService
from app.api.dependencies import CommonDeps, get_logger
from app.utils.exceptions import VideoNotFoundError, VideoGenerationError

router = APIRouter(prefix="/video", tags=["video"])

# Initialize video service
video_service = VideoService()


@router.post("/outline", response_model=SceneOutlineResponse)
async def generate_scene_outline(
    request: SceneOutlineRequest,
    current_user: dict = CommonDeps,
    logger = Depends(get_logger)
) -> SceneOutlineResponse:
    """
    Generate a scene outline for video planning.
    
    This endpoint creates a detailed scene outline based on the provided topic
    and description without actually generating the video content.
    """
    try:
        logger.info(f"Generating scene outline for topic: {request.topic}")
        start_time = datetime.utcnow()
        
        # Convert config to dict if provided
        config_dict = request.config.dict() if request.config else None
        
        # Generate scene outline
        outline = await video_service.generate_scene_outline(
            topic=request.topic,
            description=request.description,
            config=config_dict
        )
        
        generation_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Parse outline to estimate scene count and duration
        scene_count = outline.count("Scene") if outline else 0
        estimated_duration = scene_count * 30.0 if scene_count > 0 else None  # Rough estimate
        
        return SceneOutlineResponse(
            topic=request.topic,
            description=request.description,
            outline=outline,
            scene_count=scene_count,
            estimated_duration=estimated_duration,
            generation_time=generation_time
        )
        
    except Exception as e:
        logger.error(f"Failed to generate scene outline: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate scene outline: {str(e)}"
        )


@router.post("/generate", response_model=VideoGenerationResponse)
async def generate_video(
    request: VideoGenerationRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = CommonDeps,
    logger = Depends(get_logger)
) -> VideoGenerationResponse:
    """
    Start video generation process.
    
    This endpoint initiates video generation based on the provided topic and description.
    The generation runs in the background and can be monitored via the status endpoint.
    """
    try:
        # Generate unique video ID if not provided
        video_id = request.video_id or str(uuid.uuid4())
        
        logger.info(f"Starting video generation for video_id: {video_id}, topic: {request.topic}")
        
        # Convert config to dict if provided
        config_dict = request.config.dict() if request.config else None
        
        # Add background task for video generation
        background_tasks.add_task(
            _generate_video_background,
            video_id=video_id,
            topic=request.topic,
            description=request.description,
            only_plan=request.only_plan,
            specific_scenes=request.specific_scenes,
            config=config_dict,
            project_id=request.project_id
        )
        
        return VideoGenerationResponse(
            video_id=video_id,
            project_id=request.project_id,
            topic=request.topic,
            description=request.description,
            status=VideoStatus.CREATED,
            only_plan=request.only_plan,
            specific_scenes=request.specific_scenes,
            message="Video generation started successfully",
            created_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to start video generation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start video generation: {str(e)}"
        )


async def _generate_video_background(
    video_id: str,
    topic: str,
    description: str,
    only_plan: bool = False,
    specific_scenes: Optional[List[int]] = None,
    config: Optional[Dict[str, Any]] = None,
    project_id: Optional[str] = None
):
    """Background task for video generation."""
    try:
        result = await video_service.generate_video(
            topic=topic,
            description=description,
            only_plan=only_plan,
            specific_scenes=specific_scenes,
            config=config
        )
        
        # Here you would typically update the video status in a database
        # For now, we'll just log the result
        if result.get('success'):
            print(f"Video generation completed for {video_id}")
        else:
            print(f"Video generation failed for {video_id}: {result.get('error')}")
            
    except Exception as e:
        print(f"Background video generation failed for {video_id}: {str(e)}")


@router.get("/{video_id}/status", response_model=VideoStatusResponse)
async def get_video_status(
    video_id: str,
    current_user: dict = CommonDeps,
    logger = Depends(get_logger)
) -> VideoStatusResponse:
    """
    Get video generation status.
    
    Returns the current status of video generation including progress information,
    file locations, and any error messages.
    """
    try:
        logger.info(f"Getting status for video_id: {video_id}")
        
        # For now, we'll use topic-based status checking
        # In a real implementation, you'd look up the video by ID in a database
        # This is a simplified implementation
        
        # Try to get status using video_id as topic (temporary solution)
        status_info = await video_service.get_video_status(topic=video_id)
        
        if not status_info.get('exists'):
            raise VideoNotFoundError(f"Video with ID {video_id} not found")
        
        # Determine status based on file existence
        video_status = VideoStatus.COMPLETED if status_info.get('has_combined_video') else VideoStatus.GENERATING
        if status_info.get('has_scene_outline') and not status_info.get('has_combined_video'):
            video_status = VideoStatus.PLANNING
        
        return VideoStatusResponse(
            video_id=video_id,
            topic=status_info.get('topic', video_id),
            status=video_status,
            exists=status_info['exists'],
            has_scene_outline=status_info.get('has_scene_outline', False),
            has_combined_video=status_info.get('has_combined_video', False),
            scene_count=status_info.get('scene_count', 0),
            output_directory=status_info.get('output_directory'),
            combined_video_path=status_info.get('combined_video_path'),
            download_url=f"/api/v1/video/{video_id}/download" if status_info.get('has_combined_video') else None,
            created_at=datetime.utcnow()  # Would come from database in real implementation
        )
        
    except VideoNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with ID {video_id} not found"
        )
    except Exception as e:
        logger.error(f"Failed to get video status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get video status: {str(e)}"
        )


@router.get("/{video_id}/download")
async def download_video(
    video_id: str,
    current_user: dict = CommonDeps,
    logger = Depends(get_logger)
):
    """
    Download generated video file.
    
    Returns the generated video file for download. The video must be in
    completed status for download to be available.
    """
    try:
        logger.info(f"Download requested for video_id: {video_id}")
        
        # Get video status to find file path
        status_info = await video_service.get_video_status(topic=video_id)
        
        if not status_info.get('exists'):
            raise VideoNotFoundError(f"Video with ID {video_id} not found")
        
        video_path = status_info.get('combined_video_path')
        if not video_path or not os.path.exists(video_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video file not found or not yet generated"
            )
        
        # Return file response
        filename = f"{video_id}.mp4"
        return FileResponse(
            path=video_path,
            filename=filename,
            media_type="video/mp4"
        )
        
    except VideoNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with ID {video_id} not found"
        )
    except Exception as e:
        logger.error(f"Failed to download video: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download video: {str(e)}"
        )


@router.post("/{video_id}/upload")
async def upload_video_file(
    video_id: str,
    file: UploadFile = File(...),
    current_user: dict = CommonDeps,
    logger = Depends(get_logger)
):
    """
    Upload a video file to the server.
    
    Allows uploading of video files associated with a video ID.
    This can be used for custom video uploads or replacing generated content.
    """
    try:
        logger.info(f"Upload requested for video_id: {video_id}, filename: {file.filename}")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('video/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be a video file"
            )
        
        # Create upload directory
        upload_dir = os.path.join("generated_videos", "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save uploaded file
        file_extension = os.path.splitext(file.filename)[1] if file.filename else '.mp4'
        file_path = os.path.join(upload_dir, f"{video_id}{file_extension}")
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        file_size = len(content)
        
        return {
            "video_id": video_id,
            "filename": file.filename,
            "file_path": file_path,
            "file_size": file_size,
            "content_type": file.content_type,
            "message": "File uploaded successfully",
            "upload_url": f"/api/v1/video/{video_id}/download"
        }
        
    except Exception as e:
        logger.error(f"Failed to upload video file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload video file: {str(e)}"
        )


@router.get("/", response_model=VideoListResponse)
async def list_videos(
    project_id: Optional[str] = None,
    status_filter: Optional[VideoStatus] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: dict = CommonDeps,
    logger = Depends(get_logger)
) -> VideoListResponse:
    """
    List videos with optional filtering.
    
    Returns a paginated list of videos with optional filtering by project ID
    and status. This is useful for dashboard and management interfaces.
    """
    try:
        logger.info(f"Listing videos with project_id: {project_id}, status: {status_filter}")
        
        # In a real implementation, this would query a database
        # For now, we'll return a mock response
        videos = []
        
        # This is a placeholder implementation
        # In reality, you'd query your database for videos matching the criteria
        
        return VideoListResponse(
            videos=videos,
            total_count=len(videos),
            project_id=project_id
        )
        
    except Exception as e:
        logger.error(f"Failed to list videos: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list videos: {str(e)}"
        )


@router.get("/{video_id}", response_model=VideoDetailsResponse)
async def get_video_details(
    video_id: str,
    current_user: dict = CommonDeps,
    logger = Depends(get_logger)
) -> VideoDetailsResponse:
    """
    Get detailed video information.
    
    Returns comprehensive information about a video including scenes,
    files, metadata, and generation statistics.
    """
    try:
        logger.info(f"Getting details for video_id: {video_id}")
        
        # Get basic status info
        status_info = await video_service.get_video_status(topic=video_id)
        
        if not status_info.get('exists'):
            raise VideoNotFoundError(f"Video with ID {video_id} not found")
        
        # Determine status
        video_status = VideoStatus.COMPLETED if status_info.get('has_combined_video') else VideoStatus.GENERATING
        if status_info.get('has_scene_outline') and not status_info.get('has_combined_video'):
            video_status = VideoStatus.PLANNING
        
        return VideoDetailsResponse(
            video_id=video_id,
            topic=status_info.get('topic', video_id),
            description=f"Generated video for topic: {status_info.get('topic', video_id)}",
            status=video_status,
            scenes=[],  # Would be populated from database/filesystem
            files=[],   # Would be populated from database/filesystem
            created_at=datetime.utcnow()  # Would come from database
        )
        
    except VideoNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with ID {video_id} not found"
        )
    except Exception as e:
        logger.error(f"Failed to get video details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get video details: {str(e)}"
        )


@router.get("/config/default", response_model=VideoConfigResponse)
async def get_default_config(
    current_user: dict = CommonDeps,
    logger = Depends(get_logger)
) -> VideoConfigResponse:
    """
    Get default video generation configuration.
    
    Returns the default configuration settings for video generation
    along with available models and options.
    """
    try:
        logger.info("Getting default video configuration")
        
        default_config = video_service.get_default_config()
        available_models = video_service.get_available_models()
        
        return VideoConfigResponse(
            config=default_config,
            available_models=available_models,
            default_settings={
                "quality": "medium",
                "resolution": "1920x1080",
                "frame_rate": 30,
                "max_duration": 300,
                "enable_gpu": False
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get default config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get default config: {str(e)}"
        )


@router.delete("/{video_id}")
async def delete_video(
    video_id: str,
    current_user: dict = CommonDeps,
    logger = Depends(get_logger)
):
    """
    Delete a video and its associated files.
    
    Removes the video record and all associated files from the system.
    This operation cannot be undone.
    """
    try:
        logger.info(f"Deleting video_id: {video_id}")
        
        # Get video status to find files
        status_info = await video_service.get_video_status(topic=video_id)
        
        if not status_info.get('exists'):
            raise VideoNotFoundError(f"Video with ID {video_id} not found")
        
        # Delete files if they exist
        output_dir = status_info.get('output_directory')
        if output_dir and os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
            logger.info(f"Deleted video directory: {output_dir}")
        
        return {
            "video_id": video_id,
            "message": "Video deleted successfully",
            "deleted_files": output_dir
        }
        
    except VideoNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with ID {video_id} not found"
        )
    except Exception as e:
        logger.error(f"Failed to delete video: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete video: {str(e)}"
        )
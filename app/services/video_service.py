"""
Video service layer.

This service orchestrates video generation operations and provides
a clean interface between the API layer and core business logic.
"""

import asyncio
import os
import re
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from app.core.video import (
    EnhancedVideoGenerator,
    VideoGenerationConfig as CoreVideoConfig
)
from app.models.schemas.video import (
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoStatusResponse,
    SceneOutlineResponse,
    VideoDetailsResponse,
    SceneInfo,
    VoiceSettings,
    AnimationConfig
)
from app.models.enums import VideoStatus
from app.utils.exceptions import ServiceError, ValidationError
from app.utils.logging import get_logger

logger = get_logger(__name__)


class VideoService:
    """Service for video generation operations."""
    
    def __init__(self):
        """Initialize the video service."""
        self._generators: Dict[str, EnhancedVideoGenerator] = {}
        self._active_generations: Dict[str, Dict[str, Any]] = {}
        self._default_output_dir = "generated_videos"
    
    def _convert_config_to_core(self, config: VideoGenerationConfig) -> CoreVideoConfig:
        """Convert API config to core config."""
        config_dict = config.dict()
        
        # Handle voice settings conversion
        if config.voice_settings:
            voice_dict = config.voice_settings.dict()
            config_dict.update({f"voice_{k}": v for k, v in voice_dict.items()})
        
        # Handle animation config conversion
        if config.animation_config:
            anim_dict = config.animation_config.dict()
            config_dict.update({f"animation_{k}": v for k, v in anim_dict.items()})
        
        return CoreVideoConfig(**config_dict)
    
    def _get_or_create_generator(self, config: VideoGenerationConfig) -> EnhancedVideoGenerator:
        """Get or create a video generator with the given configuration."""
        # Create a config key for caching generators
        config_key = f"{config.planner_model}_{config.output_dir}_{config.max_scene_concurrency}"
        
        if config_key not in self._generators:
            core_config = self._convert_config_to_core(config)
            self._generators[config_key] = EnhancedVideoGenerator(core_config)
        
        return self._generators[config_key]
    
    def _generate_video_id(self, topic: str) -> str:
        """Generate a unique video ID."""
        from app.utils.helpers import sanitize_filename, generate_short_id, format_timestamp, get_current_timestamp
        
        topic_slug = sanitize_filename(topic.lower().replace(' ', '_'))[:50]
        timestamp = format_timestamp(get_current_timestamp(), "%Y%m%d_%H%M%S")
        unique_id = generate_short_id(8)
        return f"{topic_slug}_{timestamp}_{unique_id}"
    
    def _get_topic_directory(self, topic: str, output_dir: str) -> str:
        """Get the directory path for a topic."""
        from app.utils.helpers import sanitize_filename, ensure_directory
        from pathlib import Path
        
        file_prefix = sanitize_filename(topic.lower().replace(' ', '_'))
        topic_dir = Path(output_dir) / file_prefix
        ensure_directory(topic_dir)
        return str(topic_dir)
    
    async def generate_scene_outline(self, topic: str, description: str, 
                                   config: Optional[VideoGenerationConfig] = None) -> SceneOutlineResponse:
        """Generate a scene outline for the given topic and description."""
        try:
            if config is None:
                config = VideoGenerationConfig(
                    planner_model='openai/gpt-4o-mini',
                    output_dir=self._default_output_dir,
                    verbose=True
                )
            
            start_time = datetime.utcnow()
            generator = self._get_or_create_generator(config)
            outline = await generator.generate_scene_outline(topic, description)
            generation_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Parse outline to estimate scene count and duration
            scene_count = len([line for line in outline.split('\n') if line.strip().startswith('Scene')])
            estimated_duration = scene_count * 30.0  # Rough estimate of 30 seconds per scene
            
            return SceneOutlineResponse(
                topic=topic,
                description=description,
                outline=outline,
                scene_count=scene_count,
                estimated_duration=estimated_duration,
                generation_time=generation_time
            )
            
        except Exception as e:
            logger.error(f"Failed to generate scene outline for topic '{topic}': {str(e)}")
            raise ServiceError(f"Scene outline generation failed: {str(e)}")
    
    async def generate_video(self, request: VideoGenerationRequest) -> VideoGenerationResponse:
        """Generate a complete video or just the planning."""
        try:
            # Generate unique video ID
            video_id = request.video_id or self._generate_video_id(request.topic)
            
            # Use provided config or create default
            config = request.config or VideoGenerationConfig(
                planner_model='openai/gpt-4o-mini',
                scene_model='openai/gpt-4o-mini',
                helper_model='openai/gpt-4o-mini',
                output_dir=self._default_output_dir,
                verbose=True,
                use_rag=False,
                use_langfuse=False,
                max_scene_concurrency=1,
                enable_caching=True,
                preview_mode=True
            )
            
            # Track active generation
            self._active_generations[video_id] = {
                'topic': request.topic,
                'description': request.description,
                'status': VideoStatus.PROCESSING,
                'started_at': datetime.utcnow(),
                'only_plan': request.only_plan,
                'specific_scenes': request.specific_scenes
            }
            
            generator = self._get_or_create_generator(config)
            
            # Start generation in background
            asyncio.create_task(self._run_video_generation(
                video_id, generator, request, config
            ))
            
            return VideoGenerationResponse(
                video_id=video_id,
                project_id=request.project_id,
                topic=request.topic,
                description=request.description,
                status=VideoStatus.PROCESSING,
                only_plan=request.only_plan,
                specific_scenes=request.specific_scenes,
                message='Video generation started successfully' if not request.only_plan else 'Planning started successfully'
            )
            
        except Exception as e:
            logger.error(f"Failed to start video generation for topic '{request.topic}': {str(e)}")
            raise ServiceError(f"Video generation failed to start: {str(e)}")
    
    async def _run_video_generation(self, video_id: str, generator: EnhancedVideoGenerator, 
                                  request: VideoGenerationRequest, config: VideoGenerationConfig):
        """Run video generation in background."""
        try:
            await generator.generate_video_pipeline(
                topic=request.topic,
                description=request.description,
                only_plan=request.only_plan,
                specific_scenes=request.specific_scenes
            )
            
            # Update status to completed
            if video_id in self._active_generations:
                self._active_generations[video_id]['status'] = VideoStatus.COMPLETED
                self._active_generations[video_id]['completed_at'] = datetime.utcnow()
                
                # Set download URL if video was generated
                if not request.only_plan:
                    topic_dir = self._get_topic_directory(request.topic, config.output_dir)
                    file_prefix = re.sub(r'[^a-z0-9_]+', '_', request.topic.lower())
                    combined_video_path = os.path.join(topic_dir, f"{file_prefix}_combined.mp4")
                    if os.path.exists(combined_video_path):
                        self._active_generations[video_id]['download_url'] = combined_video_path
            
        except Exception as e:
            logger.error(f"Video generation failed for video_id '{video_id}': {str(e)}")
            if video_id in self._active_generations:
                self._active_generations[video_id]['status'] = VideoStatus.FAILED
                self._active_generations[video_id]['error'] = str(e)
                self._active_generations[video_id]['failed_at'] = datetime.utcnow()
    
    async def get_video_status(self, video_id: str) -> VideoStatusResponse:
        """Get the status of video generation by video ID."""
        try:
            # Check active generations first
            if video_id in self._active_generations:
                gen_info = self._active_generations[video_id]
                return VideoStatusResponse(
                    video_id=video_id,
                    topic=gen_info['topic'],
                    status=gen_info['status'],
                    exists=True,
                    has_scene_outline=gen_info['status'] in [VideoStatus.COMPLETED, VideoStatus.PROCESSING],
                    has_combined_video=gen_info['status'] == VideoStatus.COMPLETED and not gen_info.get('only_plan', False),
                    scene_count=0,  # Would need to check filesystem for accurate count
                    download_url=gen_info.get('download_url'),
                    error_message=gen_info.get('error'),
                    created_at=gen_info['started_at'],
                    updated_at=gen_info.get('completed_at') or gen_info.get('failed_at')
                )
            
            # If not in active generations, this is a legacy lookup by topic
            # For backward compatibility, treat video_id as topic
            topic = video_id
            file_prefix = re.sub(r'[^a-z0-9_]+', '_', topic.lower())
            topic_dir = os.path.join(self._default_output_dir, file_prefix)
            
            if not os.path.exists(topic_dir):
                return VideoStatusResponse(
                    video_id=video_id,
                    topic=topic,
                    status=VideoStatus.NOT_FOUND,
                    exists=False
                )
            
            # Check for scene outline
            scene_outline_path = os.path.join(topic_dir, f"{file_prefix}_scene_outline.txt")
            has_scene_outline = os.path.exists(scene_outline_path)
            
            # Check for combined video
            combined_video_path = os.path.join(topic_dir, f"{file_prefix}_combined.mp4")
            has_combined_video = os.path.exists(combined_video_path)
            
            # Count scene directories
            scene_count = 0
            if os.path.exists(topic_dir):
                for item in os.listdir(topic_dir):
                    if os.path.isdir(os.path.join(topic_dir, item)) and item.startswith('scene'):
                        scene_count += 1
            
            # Determine status based on files
            if has_combined_video:
                status = VideoStatus.COMPLETED
            elif has_scene_outline:
                status = VideoStatus.PROCESSING
            else:
                status = VideoStatus.PENDING
            
            return VideoStatusResponse(
                video_id=video_id,
                topic=topic,
                status=status,
                exists=True,
                has_scene_outline=has_scene_outline,
                has_combined_video=has_combined_video,
                scene_count=scene_count,
                output_directory=topic_dir,
                combined_video_path=combined_video_path if has_combined_video else None,
                download_url=combined_video_path if has_combined_video else None
            )
            
        except Exception as e:
            logger.error(f"Failed to get video status for video_id '{video_id}': {str(e)}")
            raise ServiceError(f"Failed to get video status: {str(e)}")
    
    async def get_video_details(self, video_id: str) -> VideoDetailsResponse:
        """Get detailed information about a video."""
        try:
            status_response = await self.get_video_status(video_id)
            
            # Get scene information
            scenes = []
            if status_response.output_directory and os.path.exists(status_response.output_directory):
                for item in os.listdir(status_response.output_directory):
                    scene_path = os.path.join(status_response.output_directory, item)
                    if os.path.isdir(scene_path) and item.startswith('scene'):
                        scene_number = int(item.replace('scene', ''))
                        scene_video_path = os.path.join(scene_path, f"{item}.mp4")
                        
                        scenes.append(SceneInfo(
                            scene_number=scene_number,
                            file_path=scene_video_path if os.path.exists(scene_video_path) else None,
                            download_url=scene_video_path if os.path.exists(scene_video_path) else None,
                            status=VideoStatus.COMPLETED if os.path.exists(scene_video_path) else VideoStatus.PENDING
                        ))
            
            return VideoDetailsResponse(
                video_id=video_id,
                topic=status_response.topic,
                description="",  # Would need to store this separately
                status=status_response.status,
                scenes=sorted(scenes, key=lambda x: x.scene_number),
                created_at=status_response.created_at or datetime.utcnow(),
                updated_at=status_response.updated_at
            )
            
        except Exception as e:
            logger.error(f"Failed to get video details for video_id '{video_id}': {str(e)}")
            raise ServiceError(f"Failed to get video details: {str(e)}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available models for video generation."""
        # This would typically come from configuration
        return [
            'openai/gpt-4o-mini',
            'openai/gpt-4o',
            'openai/gpt-3.5-turbo',
            'openrouter/anthropic/claude-3-haiku',
            'openrouter/anthropic/claude-3-sonnet'
        ]
    
    def get_default_config(self) -> VideoGenerationConfig:
        """Get default configuration for video generation."""
        return VideoGenerationConfig(
            planner_model='openai/gpt-4o-mini',
            scene_model='openai/gpt-4o-mini',
            helper_model='openai/gpt-4o-mini',
            output_dir=self._default_output_dir,
            verbose=False,
            use_rag=False,
            use_langfuse=True,
            max_scene_concurrency=1,
            enable_caching=True,
            voice_settings=VoiceSettings(),
            animation_config=AnimationConfig()
        )
    
    async def cancel_video_generation(self, video_id: str) -> bool:
        """Cancel an active video generation."""
        try:
            if video_id in self._active_generations:
                self._active_generations[video_id]['status'] = VideoStatus.CANCELLED
                self._active_generations[video_id]['cancelled_at'] = datetime.utcnow()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to cancel video generation for video_id '{video_id}': {str(e)}")
            raise ServiceError(f"Failed to cancel video generation: {str(e)}")
    
    async def delete_video(self, video_id: str) -> bool:
        """Delete a video and its associated files."""
        try:
            # Remove from active generations
            if video_id in self._active_generations:
                del self._active_generations[video_id]
            
            # For legacy support, treat video_id as topic for file deletion
            topic = video_id
            topic_dir = self._get_topic_directory(topic, self._default_output_dir)
            
            if os.path.exists(topic_dir):
                import shutil
                shutil.rmtree(topic_dir)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete video for video_id '{video_id}': {str(e)}")
            raise ServiceError(f"Failed to delete video: {str(e)}")
    
    def cleanup_completed_generations(self, max_age_hours: int = 24):
        """Clean up old completed generations from memory."""
        try:
            cutoff_time = datetime.utcnow().timestamp() - (max_age_hours * 3600)
            
            to_remove = []
            for video_id, gen_info in self._active_generations.items():
                completed_at = gen_info.get('completed_at') or gen_info.get('failed_at')
                if completed_at and completed_at.timestamp() < cutoff_time:
                    to_remove.append(video_id)
            
            for video_id in to_remove:
                del self._active_generations[video_id]
                
            logger.info(f"Cleaned up {len(to_remove)} old video generations")
            
        except Exception as e:
            logger.error(f"Failed to cleanup completed generations: {str(e)}")
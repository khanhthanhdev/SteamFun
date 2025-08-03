"""
Video service layer.

This service orchestrates video generation operations and provides
a clean interface between the API layer and core business logic.
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from app.core.video import (
    EnhancedVideoGenerator,
    VideoGenerationConfig
)


class VideoService:
    """Service for video generation operations."""
    
    def __init__(self):
        """Initialize the video service."""
        self._generators: Dict[str, EnhancedVideoGenerator] = {}
    
    def _get_or_create_generator(self, config_dict: Dict[str, Any]) -> EnhancedVideoGenerator:
        """Get or create a video generator with the given configuration."""
        # Create a config key for caching generators
        config_key = f"{config_dict.get('planner_model', 'default')}_{config_dict.get('output_dir', 'output')}"
        
        if config_key not in self._generators:
            config = VideoGenerationConfig(**config_dict)
            self._generators[config_key] = EnhancedVideoGenerator(config)
        
        return self._generators[config_key]
    
    async def generate_scene_outline(self, topic: str, description: str, 
                                   config: Optional[Dict[str, Any]] = None) -> str:
        """Generate a scene outline for the given topic and description."""
        if config is None:
            config = {
                'planner_model': 'openai/gpt-4o-mini',
                'output_dir': 'generated_videos',
                'verbose': True
            }
        
        generator = self._get_or_create_generator(config)
        return await generator.generate_scene_outline(topic, description)
    
    async def generate_video(self, topic: str, description: str, 
                           only_plan: bool = False,
                           specific_scenes: Optional[List[int]] = None,
                           config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a complete video or just the planning."""
        if config is None:
            config = {
                'planner_model': 'openai/gpt-4o-mini',
                'scene_model': 'openai/gpt-4o-mini',
                'helper_model': 'openai/gpt-4o-mini',
                'output_dir': 'generated_videos',
                'verbose': True,
                'use_rag': False,
                'use_langfuse': False,
                'max_scene_concurrency': 1,
                'enable_caching': True,
                'preview_mode': True
            }
        
        generator = self._get_or_create_generator(config)
        
        try:
            await generator.generate_video_pipeline(
                topic=topic,
                description=description,
                only_plan=only_plan,
                specific_scenes=specific_scenes
            )
            
            return {
                'success': True,
                'topic': topic,
                'description': description,
                'only_plan': only_plan,
                'message': 'Video generation completed successfully' if not only_plan else 'Planning completed successfully'
            }
        except Exception as e:
            return {
                'success': False,
                'topic': topic,
                'description': description,
                'error': str(e),
                'message': 'Video generation failed'
            }
    
    async def get_video_status(self, topic: str, output_dir: str = 'generated_videos') -> Dict[str, Any]:
        """Get the status of video generation for a topic."""
        import os
        import re
        
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', topic.lower())
        topic_dir = os.path.join(output_dir, file_prefix)
        
        if not os.path.exists(topic_dir):
            return {
                'topic': topic,
                'exists': False,
                'message': 'No video generation found for this topic'
            }
        
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
        
        return {
            'topic': topic,
            'exists': True,
            'has_scene_outline': has_scene_outline,
            'has_combined_video': has_combined_video,
            'scene_count': scene_count,
            'output_directory': topic_dir,
            'combined_video_path': combined_video_path if has_combined_video else None
        }
    
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
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for video generation."""
        default_config = VideoGenerationConfig(
            planner_model='openai/gpt-4o-mini'
        )
        return asdict(default_config)
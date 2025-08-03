"""
Video generator core logic.

Migrated from generate_video.py to provide clean separation
of business logic from framework-specific code.
"""

import os
import json
import asyncio
import uuid
from typing import Union, List, Dict, Optional, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod
import argparse
import re
from dotenv import load_dotenv

from mllm_tools.litellm import LiteLLMWrapper
from mllm_tools.openrouter import OpenRouterWrapper
from .video_planner import EnhancedVideoPlanner
from .video_renderer import VideoRenderer
from src.core.code_generator import CodeGenerator
from src.utils.utils import extract_xml
from src.config.config import Config
from task_generator import get_banned_reasonings


@dataclass
class VideoGenerationConfig:
    """Configuration for video generation pipeline."""
    planner_model: str
    scene_model: Optional[str] = None
    helper_model: Optional[str] = None
    output_dir: str = "output"
    verbose: bool = False
    use_rag: bool = False
    use_context_learning: bool = False
    context_learning_path: str = "data/context_learning"
    chroma_db_path: str = "data/rag/chroma_db"
    manim_docs_path: str = "data/rag/manim_docs"
    embedding_model: str = "hf:ibm-granite/granite-embedding-30m-english"
    use_visual_fix_code: bool = False
    use_langfuse: bool = True
    max_scene_concurrency: int = 5
    max_topic_concurrency: int = 1
    max_retries: int = 5
    
    # Enhanced RAG Configuration
    use_enhanced_rag: bool = True
    enable_rag_caching: bool = True
    enable_quality_monitoring: bool = True
    enable_error_handling: bool = True
    rag_cache_ttl: int = 3600
    rag_max_cache_size: int = 1000
    rag_performance_threshold: float = 2.0  # seconds
    rag_quality_threshold: float = 0.7
    
    # Renderer optimizations
    enable_caching: bool = True
    default_quality: str = "medium"
    use_gpu_acceleration: bool = False
    preview_mode: bool = False
    max_concurrent_renders: int = 4


# Protocols for dependency injection (Interface Segregation Principle)
class ModelProvider(Protocol):
    """Protocol for AI model providers."""
    def __call__(self, prompt: str, **kwargs) -> str: ...


class ComponentFactory:
    """Factory for creating video generation components."""
    
    @staticmethod
    def create_model(model_name: str, config: VideoGenerationConfig) -> ModelProvider:
        """Create AI model wrapper using configuration manager."""
        # Import configuration manager
        from src.config.manager import ConfigurationManager
        
        try:
            config_manager = ConfigurationManager()
            model_config = config_manager.get_model_config(model_name)
            
            # Use OpenRouter wrapper for OpenRouter models
            if model_name.startswith('openrouter/'):
                return OpenRouterWrapper(
                    model_name=model_name,
                    temperature=0.7,
                    print_cost=True,
                    verbose=config.verbose,
                    use_langfuse=config.use_langfuse,
                    # Use configuration from centralized manager
                    api_key=model_config.get('api_key'),
                    base_url=model_config.get('base_url'),
                    timeout=model_config.get('timeout'),
                    max_retries=model_config.get('max_retries')
                )
            else:
                # Use LiteLLM wrapper for other models
                return LiteLLMWrapper(
                    model_name=model_name,
                    temperature=0.7,
                    print_cost=True,
                    verbose=config.verbose,
                    use_langfuse=config.use_langfuse,
                    # Use configuration from centralized manager
                    api_key=model_config.get('api_key'),
                    base_url=model_config.get('base_url'),
                    timeout=model_config.get('timeout'),
                    max_retries=model_config.get('max_retries')
                )
        except Exception as e:
            print(f"Warning: Could not load model configuration for {model_name}: {e}")
            # Fallback to original behavior
            if model_name.startswith('openrouter/'):
                return OpenRouterWrapper(
                    model_name=model_name,
                    temperature=0.7,
                    print_cost=True,
                    verbose=config.verbose,
                    use_langfuse=config.use_langfuse
                )
            else:
                return LiteLLMWrapper(
                    model_name=model_name,
                    temperature=0.7,
                    print_cost=True,
                    verbose=config.verbose,
                    use_langfuse=config.use_langfuse
                )   
    @staticmethod
    def create_planner(planner_model: ModelProvider, helper_model: ModelProvider, 
                      config: VideoGenerationConfig, session_id: str) -> EnhancedVideoPlanner:
        """Create video planner with enhanced capabilities."""
        return EnhancedVideoPlanner(
            planner_model=planner_model,
            helper_model=helper_model,
            output_dir=config.output_dir,
            print_response=config.verbose,
            use_context_learning=config.use_context_learning,
            context_learning_path=config.context_learning_path,
            use_rag=config.use_rag,
            session_id=session_id,
            chroma_db_path=config.chroma_db_path,
            manim_docs_path=config.manim_docs_path,
            embedding_model=config.embedding_model,
            use_langfuse=config.use_langfuse,
            max_scene_concurrency=config.max_scene_concurrency,
            max_step_concurrency=3,
            enable_caching=config.enable_caching,
            # Enhanced RAG configuration
            use_enhanced_rag=config.use_enhanced_rag,
            rag_config={
                'use_enhanced_components': config.use_enhanced_rag,
                'enable_caching': config.enable_rag_caching,
                'enable_quality_monitoring': config.enable_quality_monitoring,
                'enable_error_handling': config.enable_error_handling,
                'cache_ttl': config.rag_cache_ttl,
                'max_cache_size': config.rag_max_cache_size,
                'performance_threshold': config.rag_performance_threshold,
                'quality_threshold': config.rag_quality_threshold
            }
        )
    
    @staticmethod
    def create_code_generator(scene_model: ModelProvider, helper_model: ModelProvider,
                            config: VideoGenerationConfig, session_id: str) -> CodeGenerator:
        """Create code generator with existing implementation."""
        return CodeGenerator(  # Use existing CodeGenerator
            scene_model=scene_model,
            helper_model=helper_model,
            output_dir=config.output_dir,
            print_response=config.verbose,
            use_rag=config.use_rag,
            use_context_learning=config.use_context_learning,
            context_learning_path=config.context_learning_path,
            chroma_db_path=config.chroma_db_path,
            manim_docs_path=config.manim_docs_path,
            embedding_model=config.embedding_model,
            use_visual_fix_code=config.use_visual_fix_code,
            use_langfuse=config.use_langfuse,
            session_id=session_id
        )
    
    @staticmethod
    def create_renderer(config: VideoGenerationConfig) -> VideoRenderer:
        """Create video renderer with existing implementation."""
        return VideoRenderer(
            output_dir=config.output_dir,
            print_response=config.verbose,
            use_visual_fix_code=config.use_visual_fix_code,
            max_concurrent_renders=config.max_concurrent_renders,
            enable_caching=config.enable_caching,
            default_quality=config.default_quality,
            use_gpu_acceleration=config.use_gpu_acceleration,
            preview_mode=config.preview_mode
        )


class SessionManager:
    """Manages session IDs for video generation."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
    
    def load_or_create_session_id(self) -> str:
        """Load existing session ID or create new one."""
        session_file = os.path.join(self.output_dir, "session_id.txt")
        
        if os.path.exists(session_file):
            with open(session_file, 'r') as f:
                session_id = f.read().strip()
                print(f"📋 Loaded existing session ID: {session_id}")
                return session_id
        
        session_id = str(uuid.uuid4())
        os.makedirs(self.output_dir, exist_ok=True)
        with open(session_file, 'w') as f:
            f.write(session_id)
        print(f"🆕 Created new session ID: {session_id}")
        return session_id
    
    def save_topic_session_id(self, topic: str, session_id: str) -> None:
        """Save session ID for specific topic."""
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', topic.lower())
        topic_dir = os.path.join(self.output_dir, file_prefix)
        os.makedirs(topic_dir, exist_ok=True)
        
        session_file = os.path.join(topic_dir, "session_id.txt")
        with open(session_file, 'w') as f:
            f.write(session_id)
            
class EnhancedVideoGenerator:
    """Enhanced video generator following SOLID principles."""
    
    def __init__(self, config: VideoGenerationConfig):
        self.config = config
        self.session_manager = SessionManager(config.output_dir)
        self.banned_reasonings = get_banned_reasonings()
        
        # Initialize session
        self.session_id = self.session_manager.load_or_create_session_id()
        
        # Create AI models
        self.planner_model = ComponentFactory.create_model(config.planner_model, config)
        self.scene_model = ComponentFactory.create_model(
            config.scene_model or config.planner_model, config
        )
        self.helper_model = ComponentFactory.create_model(
            config.helper_model or config.planner_model, config
        )
        
        # Create components using dependency injection
        self.planner = ComponentFactory.create_planner(
            self.planner_model, self.helper_model, config, self.session_id
        )
        self.code_generator = ComponentFactory.create_code_generator(
            self.scene_model, self.helper_model, config, self.session_id
        )
        self.renderer = ComponentFactory.create_renderer(config)
        
        # Concurrency control
        self.scene_semaphore = asyncio.Semaphore(config.max_scene_concurrency)
        
        print(f"🚀 Enhanced VideoGenerator initialized with:")
        print(f"   Planner: {config.planner_model}")
        print(f"   Scene: {config.scene_model or config.planner_model}")
        print(f"   Helper: {config.helper_model or config.planner_model}")
        print(f"   Max Scene Concurrency: {config.max_scene_concurrency}")
        print(f"   Caching: {'✅' if config.enable_caching else '❌'}")
        print(f"   GPU Acceleration: {'✅' if config.use_gpu_acceleration else '❌'}")

    async def generate_scene_outline(self, topic: str, description: str) -> str:
        """Generate scene outline for topic."""
        print(f"📝 Generating scene outline for: {topic}")
        return await self.planner.generate_scene_outline(topic, description, self.session_id)

    async def generate_video_pipeline(self, topic: str, description: str, 
                                    only_plan: bool = False, 
                                    specific_scenes: List[int] = None) -> None:
        """Complete video generation pipeline with enhanced performance."""
        
        print(f"🎬 Starting enhanced video pipeline for: {topic}")
        self.session_manager.save_topic_session_id(topic, self.session_id)
        
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', topic.lower())
        
        # Step 1: Load or generate scene outline
        scene_outline = await self._load_or_generate_outline(topic, description, file_prefix)
        
        # Step 2: Generate implementation plans
        implementation_plans = await self._generate_implementation_plans(
            topic, description, scene_outline, file_prefix, specific_scenes
        )
        
        if only_plan:
            print(f"📋 Plan-only mode completed for: {topic}")
            return
        
        # Step 3: Render scenes with optimization
        await self._render_scenes_optimized(
            topic, description, scene_outline, implementation_plans, file_prefix
        )
        
        # Step 4: Combine videos
        await self._combine_videos_optimized(topic)
        
        print(f"✅ Enhanced video pipeline completed for: {topic}")

    async def _load_or_generate_outline(self, topic: str, description: str, file_prefix: str) -> str:
        """Load existing outline or generate new one."""
        scene_outline_path = os.path.join(self.config.output_dir, file_prefix, f"{file_prefix}_scene_outline.txt")
        
        if os.path.exists(scene_outline_path):
            with open(scene_outline_path, "r") as f:
                scene_outline = f.read()
            print(f"📄 Loaded existing scene outline for: {topic}")
        else:
            print(f"📝 Generating new scene outline for: {topic}")
            scene_outline = await self.planner.generate_scene_outline(topic, description, self.session_id)
            
            os.makedirs(os.path.join(self.config.output_dir, file_prefix), exist_ok=True)
            with open(scene_outline_path, "w") as f:
                f.write(scene_outline)
        
        return scene_outline

    async def _generate_implementation_plans(self, topic: str, description: str, 
                                           scene_outline: str, file_prefix: str,
                                           specific_scenes: List[int] = None) -> Dict[int, str]:
        """Generate missing implementation plans."""
        
        # First, ensure the topic directory exists
        topic_dir = os.path.join(self.config.output_dir, file_prefix)
        os.makedirs(topic_dir, exist_ok=True)
        
        try:
            # Use enhanced concurrent generation
            all_plans = await self.planner.generate_scene_implementation_concurrently_enhanced(
                topic, description, scene_outline, self.session_id
            )
            
            # Convert to dictionary format
            implementation_plans_dict = {}
            for i, plan in enumerate(all_plans, 1):
                implementation_plans_dict[i] = plan
            
            return implementation_plans_dict
            
        except Exception as e:
            print(f"❌ Error generating implementation plans: {e}")
            raise

    async def _render_scenes_optimized(self, topic: str, description: str, 
                                     scene_outline: str, implementation_plans: Dict[int, str], 
                                     file_prefix: str) -> None:
        """Render scenes with optimization."""
        print(f"🎬 Starting optimized scene rendering for: {topic}")
        
        # For now, just print that rendering would happen here
        print(f"📹 Would render {len(implementation_plans)} scenes")
        for scene_num in implementation_plans.keys():
            print(f"   - Scene {scene_num}: Ready for rendering")

    async def _combine_videos_optimized(self, topic: str) -> str:
        """Combine videos with optimization."""
        print(f"🎞️ Starting video combination for: {topic}")
        
        try:
            combined_path = self.renderer.combine_videos(topic)
            print(f"✅ Video combination completed: {combined_path}")
            return combined_path
        except Exception as e:
            print(f"❌ Video combination failed: {e}")
            raise
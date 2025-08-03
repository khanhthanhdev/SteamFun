"""
PlannerAgent with enhanced video planning capabilities.
Ports EnhancedVideoPlanner functionality to LangGraph agent pattern while maintaining compatibility.
"""

import os
import re
import json
import uuid
import asyncio
import time
import warnings
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from langgraph.types import Command
import logging

from ..base_agent import BaseAgent
from ..state import VideoGenerationState

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    """PlannerAgent for video planning and scene outline generation.
    
    Ports EnhancedVideoPlanner functionality to LangGraph agent pattern while
    maintaining existing method signatures and workflow logic.
    """
    
    def __init__(self, config, system_config):
        """Initialize PlannerAgent with enhanced video planning capabilities.
        
        Args:
            config: Agent configuration
            system_config: System configuration
        """
        super().__init__(config, system_config)
        
        # Initialize internal video planner (will be created on first use)
        self._video_planner = None
        self._rag_integration = None
        
        # Initialize performance optimizer for planner
        self._planner_optimizer = None
        self._optimization_enabled = getattr(config, 'enable_optimization', True)
        
        logger.info(f"PlannerAgent initialized with config: {config.name}, optimization: {self._optimization_enabled}")
    
    async def execute(self, state: VideoGenerationState) -> Command:
        """Execute video planning and scene outline generation.
        
        Args:
            state: Current workflow state
            
        Returns:
            Command: LangGraph command for next action
        """
        try:
            self.log_agent_action("start_planning", {
                'topic': state.get('topic', ''),
                'use_rag': state.get('use_rag', False)
            })
            
            topic = state.get('topic', '')
            description = state.get('description', '')
            
            if not topic:
                raise ValueError("Topic is required for video planning")
            
            # Generate scene outline
            scene_outline = await self._generate_scene_outline(topic, description, state)
            
            # Generate scene implementations
            scene_implementations = await self._generate_scene_implementations(
                scene_outline, topic, description, state
            )
            
            # Detect relevant plugins
            detected_plugins = await self._detect_plugins(scene_outline, state)
            
            self.log_agent_action("planning_complete", {
                'scenes_generated': len(scene_implementations),
                'plugins_detected': len(detected_plugins)
            })
            
            # Determine next agent based on configuration
            next_agent = "rag_agent" if state.get('use_rag', True) else "code_generator_agent"
            
            return Command(
                goto=next_agent,
                update={
                    "scene_outline": scene_outline,
                    "scene_implementations": scene_implementations,
                    "detected_plugins": detected_plugins,
                    "current_agent": self.name
                }
            )
            
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return await self.handle_error(e, state)
    
    async def _generate_scene_outline(self, topic: str, description: str, state: VideoGenerationState) -> str:
        """Generate scene outline for the video.
        
        Args:
            topic: Video topic
            description: Video description
            state: Current workflow state
            
        Returns:
            str: Generated scene outline
        """
        # Simulate scene outline generation
        # In a real implementation, this would use the model wrapper
        scene_outline = f"""
# Video Outline: {topic}

## Scene 1: Introduction
- Duration: 30 seconds
- Content: Introduce the topic "{topic}"
- Visual: Title animation with background

## Scene 2: Main Content
- Duration: 2 minutes
- Content: {description}
- Visual: Explanatory animations and diagrams

## Scene 3: Conclusion
- Duration: 30 seconds
- Content: Summary and call to action
- Visual: Closing animation
"""
        
        return scene_outline.strip()
    
    async def _generate_scene_implementations(
        self, 
        scene_outline: str, 
        topic: str, 
        description: str, 
        state: VideoGenerationState
    ) -> Dict[int, str]:
        """Generate detailed scene implementations.
        
        Args:
            scene_outline: Generated scene outline
            topic: Video topic
            description: Video description
            state: Current workflow state
            
        Returns:
            Dict[int, str]: Scene implementations by scene number
        """
        # Parse scenes from outline and generate implementations
        scene_implementations = {}
        
        # Extract scenes from outline (simplified parsing)
        scene_matches = re.findall(r'## Scene (\d+): (.+)', scene_outline)
        
        for scene_num_str, scene_title in scene_matches:
            scene_num = int(scene_num_str)
            
            # Generate implementation for each scene
            implementation = f"""
Scene {scene_num}: {scene_title}

Implementation Details:
- Topic: {topic}
- Title: {scene_title}
- Description: {description}
- Visual Elements: Animation, text, graphics
- Duration: 30-120 seconds
- Manim Components: Scene, Text, Transform, FadeIn, FadeOut
"""
            
            scene_implementations[scene_num] = implementation.strip()
        
        return scene_implementations
    
    async def _detect_plugins(self, scene_outline: str, state: VideoGenerationState) -> List[str]:
        """Detect relevant Manim plugins based on scene content.
        
        Args:
            scene_outline: Generated scene outline
            state: Current workflow state
            
        Returns:
            List[str]: List of detected plugin names
        """
        detected_plugins = ["manim"]  # Base Manim is always included
        
        # Analyze content for plugin requirements
        content_lower = scene_outline.lower()
        
        if any(keyword in content_lower for keyword in ['math', 'equation', 'formula', 'graph']):
            detected_plugins.extend(["numpy", "matplotlib"])
        
        if any(keyword in content_lower for keyword in ['3d', 'three dimensional', 'perspective']):
            detected_plugins.append("manim_3d")
        
        if any(keyword in content_lower for keyword in ['chart', 'data', 'statistics']):
            detected_plugins.extend(["pandas", "seaborn"])
        
        return list(set(detected_plugins))  # Remove duplicates
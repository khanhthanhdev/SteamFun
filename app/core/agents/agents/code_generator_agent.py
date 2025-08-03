"""
CodeGeneratorAgent for Manim code generation.
Provides code generation capabilities for video scenes.
"""

from typing import Dict, Any, List, Optional
from langgraph.types import Command
import logging

from ..base_agent import BaseAgent
from ..state import VideoGenerationState

logger = logging.getLogger(__name__)


class CodeGeneratorAgent(BaseAgent):
    """Agent responsible for generating Manim code from scene implementations."""
    
    def __init__(self, config, system_config):
        """Initialize CodeGeneratorAgent.
        
        Args:
            config: Agent configuration
            system_config: System configuration
        """
        super().__init__(config, system_config)
        logger.info(f"CodeGeneratorAgent initialized: {config.name}")
    
    async def execute(self, state: VideoGenerationState) -> Command:
        """Execute code generation for video scenes.
        
        Args:
            state: Current workflow state
            
        Returns:
            Command: LangGraph command for next action
        """
        try:
            self.log_agent_action("start_code_generation", {
                'scene_count': len(state.get('scene_implementations', {}))
            })
            
            scene_implementations = state.get('scene_implementations', {})
            if not scene_implementations:
                raise ValueError("No scene implementations available for code generation")
            
            # Generate code for each scene
            generated_code = {}
            code_errors = {}
            
            for scene_num, implementation in scene_implementations.items():
                try:
                    code = await self._generate_scene_code(scene_num, implementation, state)
                    generated_code[scene_num] = code
                except Exception as e:
                    code_errors[scene_num] = str(e)
                    logger.error(f"Code generation failed for scene {scene_num}: {e}")
            
            if not generated_code:
                raise ValueError("No code was successfully generated")
            
            self.log_agent_action("code_generation_complete", {
                'scenes_generated': len(generated_code),
                'errors': len(code_errors)
            })
            
            return Command(
                goto="renderer_agent",
                update={
                    "generated_code": generated_code,
                    "code_errors": code_errors,
                    "current_agent": self.name
                }
            )
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return await self.handle_error(e, state)
    
    async def _generate_scene_code(
        self, 
        scene_num: int, 
        implementation: str, 
        state: VideoGenerationState
    ) -> str:
        """Generate Manim code for a specific scene.
        
        Args:
            scene_num: Scene number
            implementation: Scene implementation details
            state: Current workflow state
            
        Returns:
            str: Generated Manim code
        """
        topic = state.get('topic', 'Video')
        
        # Generate basic Manim code structure
        code = f'''
from manim import *

class Scene{scene_num}(Scene):
    """
    Scene {scene_num} for video: {topic}
    
    Implementation:
    {implementation}
    """
    
    def construct(self):
        # Scene {scene_num} implementation
        
        # Title
        title = Text("{topic}", font_size=48)
        title.to_edge(UP)
        
        # Main content
        content = Text(
            "Scene {scene_num} Content",
            font_size=24,
            line_spacing=1.5
        )
        content.next_to(title, DOWN, buff=1)
        
        # Animations
        self.play(Write(title))
        self.wait(1)
        self.play(FadeIn(content))
        self.wait(2)
        
        # Cleanup
        self.play(FadeOut(title), FadeOut(content))
'''
        
        return code.strip()
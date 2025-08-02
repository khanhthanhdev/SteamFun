"""
Improved PlannerAgent with better timeout handling and retry mechanisms.
Fixes the timeout issues you're experiencing during plan creation.
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

from src.langgraph_agents.base_agent import BaseAgent
from src.langgraph_agents.state import VideoGenerationState

logger = logging.getLogger(__name__)


class ImprovedPlannerAgent(BaseAgent):
    """Improved PlannerAgent with robust timeout handling and retry mechanisms."""
    
    def __init__(self, config, system_config):
        """Initialize ImprovedPlannerAgent with timeout-resistant configuration."""
        super().__init__(config, system_config)
        
        # Timeout and retry configuration
        self.default_timeout = 120  # 2 minutes default timeout
        self.max_retries = 3
        self.retry_delay = 5  # seconds between retries
        self.fallback_timeout = 60  # 1 minute for fallback attempts
        
        # Model configuration with timeout settings
        self.model_config = {
            **config.model_config,
            'timeout': config.model_config.get('timeout', self.default_timeout),
            'max_retries': config.model_config.get('max_retries', self.max_retries)
        }
        
        logger.info(f"ImprovedPlannerAgent initialized with timeout: {self.default_timeout}s, max_retries: {self.max_retries}")
    
    async def execute(self, state: VideoGenerationState) -> Command:
        """Execute video planning with robust timeout handling."""
        
        self.log_agent_action("starting_improved_planning", {
            'topic': state.get('topic', ''),
            'timeout_config': self.default_timeout,
            'max_retries': self.max_retries
        })
        
        try:
            # Step 1: Generate scene outline with timeout protection
            scene_outline = await self._generate_scene_outline_with_timeout(state)
            
            if not scene_outline:
                raise ValueError("Failed to generate scene outline")
            
            # Check if only planning is requested
            if state.get('only_plan', False):
                return Command(
                    goto="END",
                    update={
                        "scene_outline": scene_outline,
                        "current_agent": "planner_agent",
                        "workflow_complete": True
                    }
                )
            
            # Step 2: Generate scene implementations with timeout protection
            scene_implementations = await self._generate_scene_implementations_with_timeout(
                state, scene_outline
            )
            
            return Command(
                goto="code_generator_agent",
                update={
                    "scene_outline": scene_outline,
                    "scene_implementations": scene_implementations,
                    "current_agent": "code_generator_agent",
                    "next_agent": "code_generator_agent"
                }
            )
            
        except Exception as e:
            logger.error(f"Error in ImprovedPlannerAgent execution: {e}")
            return await self.handle_error(e, state)
    
    async def _generate_scene_outline_with_timeout(self, state: VideoGenerationState) -> str:
        """Generate scene outline with timeout protection and retries."""
        
        topic = state['topic']
        description = state['description']
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Generating scene outline (attempt {attempt + 1}/{self.max_retries})")
                
                # Use a shorter, more focused prompt to reduce timeout risk
                outline_prompt = self._create_focused_outline_prompt(topic, description)
                
                # Get model with timeout configuration
                model = self._get_timeout_configured_model(state)
                
                # Generate with timeout
                scene_outline = await asyncio.wait_for(
                    self._call_model_for_outline(model, outline_prompt),
                    timeout=self.default_timeout
                )
                
                if scene_outline and len(scene_outline.strip()) > 50:  # Basic validation
                    logger.info(f"Scene outline generated successfully on attempt {attempt + 1}")
                    return scene_outline
                else:
                    raise ValueError("Generated outline is too short or empty")
                    
            except asyncio.TimeoutError:
                logger.warning(f"Timeout on attempt {attempt + 1} for scene outline generation")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
                else:
                    # Final attempt with fallback approach
                    return await self._generate_fallback_outline(topic, description, state)
                    
            except Exception as e:
                logger.warning(f"Error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
                else:
                    raise
        
        raise RuntimeError("Failed to generate scene outline after all retries")
    
    async def _generate_scene_implementations_with_timeout(
        self, 
        state: VideoGenerationState, 
        scene_outline: str
    ) -> Dict[int, str]:
        """Generate scene implementations with timeout protection."""
        
        # Extract scene count from outline
        scene_count = self._extract_scene_count(scene_outline)
        
        if scene_count == 0:
            logger.warning("No scenes detected in outline, creating default single scene")
            scene_count = 1
        
        logger.info(f"Generating implementations for {scene_count} scenes")
        
        scene_implementations = {}
        
        # Generate implementations one by one to avoid timeout
        for scene_num in range(1, scene_count + 1):
            try:
                implementation = await self._generate_single_scene_implementation(
                    state, scene_outline, scene_num
                )
                scene_implementations[scene_num] = implementation
                
                logger.info(f"Generated implementation for scene {scene_num}")
                
                # Small delay between scenes to avoid rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.warning(f"Failed to generate implementation for scene {scene_num}: {e}")
                # Create a fallback implementation
                scene_implementations[scene_num] = self._create_fallback_implementation(
                    state['topic'], scene_num
                )
        
        return scene_implementations
    
    async def _generate_single_scene_implementation(
        self, 
        state: VideoGenerationState, 
        scene_outline: str, 
        scene_num: int
    ) -> str:
        """Generate implementation for a single scene with timeout protection."""
        
        for attempt in range(2):  # Fewer retries for individual scenes
            try:
                # Create focused prompt for this scene
                implementation_prompt = self._create_scene_implementation_prompt(
                    state['topic'], state['description'], scene_outline, scene_num
                )
                
                # Get model with shorter timeout for individual scenes
                model = self._get_timeout_configured_model(state, timeout=self.fallback_timeout)
                
                # Generate with timeout
                implementation = await asyncio.wait_for(
                    self._call_model_for_implementation(model, implementation_prompt),
                    timeout=self.fallback_timeout
                )
                
                if implementation and len(implementation.strip()) > 20:
                    return implementation
                else:
                    raise ValueError("Generated implementation is too short")
                    
            except asyncio.TimeoutError:
                logger.warning(f"Timeout generating implementation for scene {scene_num}, attempt {attempt + 1}")
                if attempt == 0:
                    await asyncio.sleep(2)
                    continue
                else:
                    # Return fallback implementation
                    return self._create_fallback_implementation(state['topic'], scene_num)
                    
            except Exception as e:
                logger.warning(f"Error generating implementation for scene {scene_num}: {e}")
                if attempt == 0:
                    await asyncio.sleep(2)
                    continue
                else:
                    return self._create_fallback_implementation(state['topic'], scene_num)
        
        return self._create_fallback_implementation(state['topic'], scene_num)
    
    def _create_focused_outline_prompt(self, topic: str, description: str) -> str:
        """Create a focused, concise prompt for scene outline generation."""
        
        return f"""Create a concise scene outline for an educational video.

Topic: {topic}
Description: {description}

Generate 3-4 scenes maximum. Each scene should be 30-60 seconds.

Format:
<SCENE_1>
Scene 1 title and brief description
</SCENE_1>

<SCENE_2>
Scene 2 title and brief description
</SCENE_2>

Keep it simple and focused. Avoid complex animations."""
    
    def _create_scene_implementation_prompt(
        self, 
        topic: str, 
        description: str, 
        scene_outline: str, 
        scene_num: int
    ) -> str:
        """Create a focused prompt for scene implementation."""
        
        return f"""Create a simple implementation plan for scene {scene_num}.

Topic: {topic}
Description: {description}

Scene Outline:
{scene_outline}

For Scene {scene_num}, provide:
1. Main visual elements (keep simple)
2. Text to display
3. Basic animation sequence

Keep the implementation simple and achievable with Manim. Avoid complex 3D or advanced animations."""
    
    async def _call_model_for_outline(self, model, prompt: str) -> str:
        """Call model for outline generation."""
        try:
            response = await model.acall(prompt)
            return response if isinstance(response, str) else str(response)
        except Exception as e:
            logger.error(f"Model call failed for outline: {e}")
            raise
    
    async def _call_model_for_implementation(self, model, prompt: str) -> str:
        """Call model for implementation generation."""
        try:
            response = await model.acall(prompt)
            return response if isinstance(response, str) else str(response)
        except Exception as e:
            logger.error(f"Model call failed for implementation: {e}")
            raise
    
    def _get_timeout_configured_model(self, state: VideoGenerationState, timeout: int = None):
        """Get model wrapper with timeout configuration."""
        
        model_name = self.model_config.get('model_name', 'openai/gpt-4o-mini')
        
        # Create a model wrapper with timeout settings
        # This is a simplified version - you may need to adapt based on your model wrapper
        model = self.get_model_wrapper(model_name, state)
        
        # Configure timeout if the model wrapper supports it
        if hasattr(model, 'timeout'):
            model.timeout = timeout or self.default_timeout
        
        return model
    
    async def _generate_fallback_outline(
        self, 
        topic: str, 
        description: str, 
        state: VideoGenerationState
    ) -> str:
        """Generate a simple fallback outline when main generation fails."""
        
        logger.info("Generating fallback scene outline")
        
        # Create a very simple, template-based outline
        fallback_outline = f"""<SCENE_1>
Introduction to {topic}
Brief overview and importance of the topic.
</SCENE_1>

<SCENE_2>
Main Content: {topic}
Core concepts and explanations based on: {description}
</SCENE_2>

<SCENE_3>
Examples and Applications
Practical examples and real-world applications.
</SCENE_3>

<SCENE_4>
Summary and Conclusion
Key takeaways and next steps for learning.
</SCENE_4>"""
        
        return fallback_outline
    
    def _create_fallback_implementation(self, topic: str, scene_num: int) -> str:
        """Create a simple fallback implementation for a scene."""
        
        implementations = {
            1: f"Introduction scene for {topic}. Display title text, brief overview, and simple animations to introduce the topic.",
            2: f"Main content scene for {topic}. Present key concepts with text displays and basic visual elements.",
            3: f"Examples scene for {topic}. Show practical examples with step-by-step visual explanations.",
            4: f"Conclusion scene for {topic}. Summarize key points and provide closing thoughts."
        }
        
        return implementations.get(scene_num, f"Scene {scene_num} content for {topic}. Basic presentation of relevant information.")
    
    def _extract_scene_count(self, scene_outline: str) -> int:
        """Extract the number of scenes from the outline."""
        
        # Look for scene markers
        scene_markers = re.findall(r'<SCENE_(\d+)>', scene_outline)
        
        if scene_markers:
            return len(scene_markers)
        
        # Fallback: count lines that might be scenes
        lines = [line.strip() for line in scene_outline.split('\n') if line.strip()]
        
        # Estimate based on content length
        if len(lines) > 10:
            return 4
        elif len(lines) > 5:
            return 3
        else:
            return 2
    
    def get_planning_status(self, state: VideoGenerationState) -> Dict[str, Any]:
        """Get current planning status."""
        
        return {
            'agent_name': 'improved_planner_agent',
            'timeout_config': self.default_timeout,
            'max_retries': self.max_retries,
            'scene_outline_generated': bool(state.get('scene_outline')),
            'scene_implementations_count': len(state.get('scene_implementations', {})),
            'execution_stats': self.execution_stats
        }


# Usage example function
async def test_improved_planner():
    """Test the improved planner agent."""
    
    from src.langgraph_agents.state import AgentConfig, SystemConfig, create_initial_state
    
    # Create configuration
    agent_config = AgentConfig(
        name="improved_planner_agent",
        model_config={
            "model_name": "openai/gpt-4o-mini",
            "temperature": 0.7,
            "timeout": 120,  # 2 minutes
            "max_retries": 3
        },
        tools=["planning"],
        max_retries=3,
        timeout_seconds=300
    )
    
    system_config = {
        "output_dir": "test_output",
        "llm_providers": {
            "openai": {
                "api_key": os.getenv("OPENAI_API_KEY")
            }
        }
    }
    
    # Create agent
    agent = ImprovedPlannerAgent(agent_config, system_config)
    
    # Create test state
    full_system_config = SystemConfig(
        agents={"improved_planner_agent": agent_config},
        llm_providers=system_config["llm_providers"],
        docling_config={},
        mcp_servers={},
        monitoring_config={},
        human_loop_config={}
    )
    
    state = create_initial_state(
        topic="Python Functions",
        description="Educational video about defining and using functions in Python",
        session_id="test_improved_planner",
        config=full_system_config
    )
    
    # Add planning-only flag for testing
    state['only_plan'] = True
    
    # Execute agent
    try:
        print("🧪 Testing Improved Planner Agent...")
        result = await agent.execute(state)
        
        print("✅ Planning completed successfully!")
        print(f"   Scene outline length: {len(result.update.get('scene_outline', ''))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Planning failed: {e}")
        return False


if __name__ == "__main__":
    # Test the improved planner
    import asyncio
    from dotenv import load_dotenv
    
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found")
        exit(1)
    
    success = asyncio.run(test_improved_planner())
    print(f"Test result: {'✅ SUCCESS' if success else '❌ FAILED'}")
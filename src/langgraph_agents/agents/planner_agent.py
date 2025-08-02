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
from ..utils.warning_suppression import suppress_deprecation_warnings

# Suppress warnings before importing components that may trigger them
suppress_deprecation_warnings()

from src.core.video_planner import EnhancedVideoPlanner

# Import RAG components with warnings suppressed
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    try:
        from src.rag.rag_integration import RAGIntegration, RAGConfig
    except ImportError:
        # Handle case where RAG components are not available
        RAGIntegration = None
        RAGConfig = None


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
    
    def _get_video_planner(self, state: VideoGenerationState) -> EnhancedVideoPlanner:
        """Get or create EnhancedVideoPlanner instance with current state configuration.
        
        Args:
            state: Current workflow state
            
        Returns:
            EnhancedVideoPlanner: Configured video planner instance
        """
        if self._video_planner is None:
            # Get model wrappers compatible with existing patterns
            planner_model = self.get_model_wrapper(self.planner_model, state)
            helper_model = self.get_model_wrapper(self.helper_model, state)
            
            # Create RAG configuration if enhanced RAG is enabled
            rag_config = None
            if state.get('use_enhanced_rag', True):
                rag_config = {
                    'use_enhanced_components': True,
                    'enable_caching': state.get('enable_rag_caching', True),
                    'enable_quality_monitoring': state.get('enable_quality_monitoring', True),
                    'enable_error_handling': state.get('enable_error_handling', True),
                    'cache_ttl': state.get('rag_cache_ttl', 3600),
                    'max_cache_size': state.get('rag_max_cache_size', 1000),
                    'performance_threshold': state.get('rag_performance_threshold', 2.0),
                    'quality_threshold': state.get('rag_quality_threshold', 0.7)
                }
            
            # Initialize EnhancedVideoPlanner with state configuration
            self._video_planner = EnhancedVideoPlanner(
                planner_model=planner_model,
                helper_model=helper_model,
                output_dir=state.get('output_dir', 'output'),
                print_response=state.get('print_response', False),
                use_context_learning=state.get('use_context_learning', True),
                context_learning_path=state.get('context_learning_path', 'data/context_learning'),
                use_rag=state.get('use_rag', True),
                session_id=state.get('session_id'),
                chroma_db_path=state.get('chroma_db_path', 'data/rag/chroma_db'),
                manim_docs_path=state.get('manim_docs_path', 'data/rag/manim_docs'),
                embedding_model=state.get('embedding_model', 'hf:ibm-granite/granite-embedding-30m-english'),
                use_langfuse=state.get('use_langfuse', True),
                max_scene_concurrency=state.get('max_scene_concurrency', 5),
                max_step_concurrency=3,
                enable_caching=state.get('enable_caching', True),
                use_enhanced_rag=state.get('use_enhanced_rag', True),
                rag_config=rag_config
            )
            
            logger.info("Created EnhancedVideoPlanner instance with current state configuration")
        
        return self._video_planner
    
    async def execute(self, state: VideoGenerationState) -> Command:
        """Execute video planning with enhanced capabilities and performance optimization.
        
        Args:
            state: Current workflow state
            
        Returns:
            Command: LangGraph command for next action
        """
        self.log_agent_action("starting_video_planning", {
            'topic': state.get('topic', ''),
            'use_rag': state.get('use_rag', False),
            'use_context_learning': state.get('use_context_learning', False),
            'optimization_enabled': self._optimization_enabled
        })
        
        try:
            # Initialize performance optimizer if enabled
            if self._optimization_enabled and not self._planner_optimizer:
                await self._initialize_planner_optimizer(state)
            
            # Get video planner instance
            video_planner = self._get_video_planner(state)
            
            # Generate scene outline with optimization
            if self._optimization_enabled and self._planner_optimizer:
                scene_outline = await self._planner_optimizer.optimize_scene_outline_generation(
                    planner_func=video_planner.generate_scene_outline,
                    topic=state['topic'],
                    description=state['description'],
                    session_id=state['session_id']
                )
            else:
                scene_outline = await video_planner.generate_scene_outline(
                    topic=state['topic'],
                    description=state['description'],
                    session_id=state['session_id']
                )
            
            if not scene_outline:
                raise ValueError("Failed to generate scene outline")
            
            # Detect plugins with optimization
            if self._optimization_enabled and self._planner_optimizer:
                detected_plugins = await self._planner_optimizer.optimize_plugin_detection(
                    detection_func=self._detect_plugins_async,
                    topic=state['topic'],
                    description=state['description']
                )
            else:
                detected_plugins = await self._detect_plugins_async(
                    state['topic'], state['description'], state
                )
            
            self.log_agent_action("scene_outline_generated", {
                'outline_length': len(scene_outline),
                'detected_plugins': detected_plugins,
                'optimization_used': self._optimization_enabled
            })
            
            # Check if we should only generate plan (compatible with existing API)
            only_plan = state.get('only_plan', False)
            if only_plan:
                self.log_agent_action("plan_only_mode_complete")
                return Command(
                    goto="END",
                    update={
                        "scene_outline": scene_outline,
                        "detected_plugins": detected_plugins,
                        "current_agent": "planner_agent",
                        "workflow_complete": True
                    }
                )
            
            # Generate scene implementations with optimization
            if self._optimization_enabled and self._planner_optimizer:
                scene_implementations_list = await self._planner_optimizer.optimize_scene_implementation_generation(
                    planner_func=video_planner.generate_scene_implementation_concurrently_enhanced,
                    topic=state['topic'],
                    description=state['description'],
                    plan=scene_outline,
                    session_id=state['session_id']
                )
            else:
                scene_implementations_list = await video_planner.generate_scene_implementation_concurrently_enhanced(
                    topic=state['topic'],
                    description=state['description'],
                    plan=scene_outline,
                    session_id=state['session_id']
                )
            
            # Convert list to dictionary for state compatibility
            scene_implementations = {}
            for i, implementation in enumerate(scene_implementations_list, 1):
                scene_implementations[i] = implementation
            
            self.log_agent_action("scene_implementations_generated", {
                'scene_count': len(scene_implementations),
                'successful_scenes': len([impl for impl in scene_implementations_list if not impl.startswith("# Scene") or "Error:" not in impl]),
                'optimization_used': self._optimization_enabled
            })
            
            # Route to CodeGeneratorAgent with preserved workflow logic
            return Command(
                goto="code_generator_agent",
                update={
                    "scene_outline": scene_outline,
                    "scene_implementations": scene_implementations,
                    "detected_plugins": detected_plugins,
                    "current_agent": "code_generator_agent",
                    "next_agent": "code_generator_agent"
                }
            )
            
        except Exception as e:
            logger.error(f"Error in PlannerAgent execution: {e}")
            
            # Check if we should escalate to human
            if self.should_escalate_to_human(state):
                return self.create_human_intervention_command(
                    context=f"Video planning failed for topic '{state.get('topic', '')}': {str(e)}",
                    options=["Retry with different approach", "Skip planning", "Manual intervention"],
                    state=state
                )
            
            # Handle error through base class
            return await self.handle_error(e, state)
    
    async def generate_scene_outline(self, 
                                   topic: str, 
                                   description: str, 
                                   session_id: str,
                                   state: VideoGenerationState) -> str:
        """Generate scene outline (compatible with existing API).
        
        Args:
            topic: Video topic
            description: Video description
            session_id: Session identifier
            state: Current workflow state
            
        Returns:
            str: Generated scene outline
        """
        video_planner = self._get_video_planner(state)
        return await video_planner.generate_scene_outline(topic, description, session_id)
    
    async def generate_scene_implementation_concurrently(self,
                                                       topic: str,
                                                       description: str,
                                                       plan: str,
                                                       session_id: str,
                                                       state: VideoGenerationState) -> List[str]:
        """Generate scene implementations concurrently (compatible with existing API).
        
        Args:
            topic: Video topic
            description: Video description
            plan: Scene plan/outline
            session_id: Session identifier
            state: Current workflow state
            
        Returns:
            List[str]: Scene implementation plans
        """
        video_planner = self._get_video_planner(state)
        return await video_planner.generate_scene_implementation_concurrently_enhanced(
            topic, description, plan, session_id
        )
    
    async def _detect_plugins_async(self, 
                                  topic: str, 
                                  description: str,
                                  state: VideoGenerationState) -> List[str]:
        """Detect relevant plugins asynchronously (compatible with existing method).
        
        Args:
            topic: Video topic
            description: Video description
            state: Current workflow state
            
        Returns:
            List[str]: Detected relevant plugins
        """
        video_planner = self._get_video_planner(state)
        
        if hasattr(video_planner, 'rag_integration') and video_planner.rag_integration:
            return await video_planner._detect_plugins_async(topic, description)
        
        return []
    
    def get_planning_status(self, state: VideoGenerationState) -> Dict[str, Any]:
        """Get current planning status and metrics.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dict: Planning status information
        """
        return {
            'agent_name': self.name,
            'scene_outline_generated': bool(state.get('scene_outline')),
            'scene_implementations_count': len(state.get('scene_implementations', {})),
            'detected_plugins': state.get('detected_plugins', []),
            'execution_stats': self.execution_stats,
            'rag_enabled': state.get('use_rag', False),
            'context_learning_enabled': state.get('use_context_learning', False),
            'enhanced_rag_enabled': state.get('use_enhanced_rag', False)
        }
    
    async def handle_planning_error(self, 
                                  error: Exception, 
                                  state: VideoGenerationState,
                                  retry_with_fallback: bool = True) -> Command:
        """Handle planning-specific errors with fallback strategies.
        
        Args:
            error: Exception that occurred
            state: Current workflow state
            retry_with_fallback: Whether to try fallback strategies
            
        Returns:
            Command: LangGraph command for error handling
        """
        error_type = type(error).__name__
        
        # Planning-specific error handling
        if "scene_outline" in str(error).lower() and retry_with_fallback:
            # Try with reduced complexity
            self.log_agent_action("retrying_with_fallback", {
                'error_type': error_type,
                'fallback_strategy': 'reduced_complexity'
            })
            
            # Update state to use simpler planning approach
            fallback_state = state.copy()
            fallback_state.update({
                'use_context_learning': False,
                'max_scene_concurrency': 1
            })
            
            try:
                return await self.execute(fallback_state)
            except Exception as fallback_error:
                logger.error(f"Fallback planning also failed: {fallback_error}")
        
        # Use base error handling
        return await self.handle_error(error, state)
    
    async def _initialize_planner_optimizer(self, state: VideoGenerationState):
        """Initialize the planner performance optimizer.
        
        Args:
            state: Current workflow state
        """
        try:
            from ..performance.planner_optimizer import PlannerOptimizer
            
            # Create optimizer configuration
            optimizer_config = {
                'max_parallel_scenes': state.get('max_scene_concurrency', 8),
                'max_concurrent_implementations': min(state.get('max_scene_concurrency', 8), 6),
                'rag_batch_size': 10,
                'enable_scene_caching': True,
                'enable_implementation_caching': True,
                'enable_plugin_caching': True,
                'max_worker_threads': 4
            }
            
            self._planner_optimizer = PlannerOptimizer(optimizer_config)
            await self._planner_optimizer.start()
            
            logger.info("Planner optimizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize planner optimizer: {str(e)}")
            self._optimization_enabled = False
    
    async def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get planner optimization metrics.
        
        Returns:
            Dict: Optimization metrics
        """
        if self._planner_optimizer:
            return self._planner_optimizer.get_optimization_metrics()
        else:
            return {'error': 'Planner optimizer not initialized'}
    
    async def clear_optimization_caches(self, cache_types: Optional[List[str]] = None):
        """Clear planner optimization caches.
        
        Args:
            cache_types: List of cache types to clear
        """
        if self._planner_optimizer:
            self._planner_optimizer.clear_caches(cache_types)
            logger.info(f"Cleared planner optimization caches: {cache_types}")
    
    def __del__(self):
        """Cleanup resources when agent is destroyed."""
        if self._video_planner and hasattr(self._video_planner, 'thread_pool'):
            try:
                self._video_planner.thread_pool.shutdown(wait=False)
            except Exception:
                pass  # Ignore cleanup errors
        
        if self._planner_optimizer:
            try:
                # Note: Can't use await in __del__, so we'll just log
                logger.info("Planner optimizer cleanup needed (async)")
            except Exception:
                pass  # Ignore cleanup errors
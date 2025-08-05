"""
PlanningService - Business logic for video planning and scene outline generation.

Extracted from PlannerAgent to follow separation of concerns and single responsibility principles.
"""

import logging
import warnings
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Suppress warnings before importing components that may trigger them
from ..utils.warning_suppression import suppress_deprecation_warnings
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


class PlanningService:
    """Service class for video planning operations.
    
    Handles scene outline generation, plugin detection, and scene implementation
    generation with comprehensive error handling and logging.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize PlanningService with configuration.
        
        Args:
            config: Service configuration containing model settings, paths, etc.
        """
        self.config = config
        self._video_planner = None
        self._rag_integration = None
        
        # Extract configuration values
        self.planner_model = config.get('planner_model', 'openrouter/anthropic/claude-3.5-sonnet')
        self.helper_model = config.get('helper_model', 'openrouter/anthropic/claude-3.5-sonnet')
        self.output_dir = config.get('output_dir', 'output')
        self.use_rag = config.get('use_rag', True)
        self.use_context_learning = config.get('use_context_learning', True)
        self.context_learning_path = config.get('context_learning_path', 'data/context_learning')
        self.chroma_db_path = config.get('chroma_db_path', 'data/rag/chroma_db')
        self.manim_docs_path = config.get('manim_docs_path', 'data/rag/manim_docs')
        self.embedding_model = config.get('embedding_model', 'hf:ibm-granite/granite-embedding-30m-english')
        self.use_langfuse = config.get('use_langfuse', True)
        self.max_scene_concurrency = config.get('max_scene_concurrency', 5)
        self.enable_caching = config.get('enable_caching', True)
        self.use_enhanced_rag = config.get('use_enhanced_rag', True)
        
        logger.info(f"PlanningService initialized with config: {self.config}")
    
    def _get_video_planner(self, model_wrappers: Dict[str, Any]) -> EnhancedVideoPlanner:
        """Get or create EnhancedVideoPlanner instance.
        
        Args:
            model_wrappers: Dictionary containing planner_model and helper_model wrappers
            
        Returns:
            EnhancedVideoPlanner: Configured video planner instance
        """
        if self._video_planner is None:
            # Create RAG configuration if enhanced RAG is enabled
            rag_config = None
            if self.use_enhanced_rag:
                rag_config = {
                    'use_enhanced_components': True,
                    'enable_caching': self.config.get('enable_rag_caching', True),
                    'enable_quality_monitoring': self.config.get('enable_quality_monitoring', True),
                    'enable_error_handling': self.config.get('enable_error_handling', True),
                    'cache_ttl': self.config.get('rag_cache_ttl', 3600),
                    'max_cache_size': self.config.get('rag_max_cache_size', 1000),
                    'performance_threshold': self.config.get('rag_performance_threshold', 2.0),
                    'quality_threshold': self.config.get('rag_quality_threshold', 0.7)
                }
            
            # Initialize EnhancedVideoPlanner
            self._video_planner = EnhancedVideoPlanner(
                planner_model=model_wrappers['planner_model'],
                helper_model=model_wrappers['helper_model'],
                output_dir=self.output_dir,
                print_response=self.config.get('print_response', False),
                use_context_learning=self.use_context_learning,
                context_learning_path=self.context_learning_path,
                use_rag=self.use_rag,
                session_id=self.config.get('session_id'),
                chroma_db_path=self.chroma_db_path,
                manim_docs_path=self.manim_docs_path,
                embedding_model=self.embedding_model,
                use_langfuse=self.use_langfuse,
                max_scene_concurrency=self.max_scene_concurrency,
                max_step_concurrency=3,
                enable_caching=self.enable_caching,
                use_enhanced_rag=self.use_enhanced_rag,
                rag_config=rag_config
            )
            
            logger.info("Created EnhancedVideoPlanner instance")
        
        return self._video_planner
    
    async def generate_scene_outline(self, 
                                   topic: str, 
                                   description: str, 
                                   session_id: str,
                                   model_wrappers: Dict[str, Any]) -> str:
        """Generate scene outline for video planning.
        
        Args:
            topic: Video topic
            description: Video description
            session_id: Session identifier
            model_wrappers: Dictionary containing model wrappers
            
        Returns:
            str: Generated scene outline
            
        Raises:
            ValueError: If topic or description is empty
            Exception: If scene outline generation fails
        """
        if not topic or not topic.strip():
            raise ValueError("Topic cannot be empty")
        
        if not description or not description.strip():
            raise ValueError("Description cannot be empty")
        
        logger.info(f"Generating scene outline for topic: {topic}")
        
        try:
            video_planner = self._get_video_planner(model_wrappers)
            
            scene_outline = await video_planner.generate_scene_outline(
                topic=topic,
                description=description,
                session_id=session_id
            )
            
            if not scene_outline:
                raise ValueError("Failed to generate scene outline - empty result")
            
            logger.info(f"Successfully generated scene outline with length: {len(scene_outline)}")
            return scene_outline
            
        except Exception as e:
            logger.error(f"Error generating scene outline: {e}")
            raise
    
    async def generate_scene_implementations(self,
                                           topic: str,
                                           description: str,
                                           plan: str,
                                           session_id: str,
                                           model_wrappers: Dict[str, Any]) -> List[str]:
        """Generate scene implementations concurrently.
        
        Args:
            topic: Video topic
            description: Video description
            plan: Scene plan/outline
            session_id: Session identifier
            model_wrappers: Dictionary containing model wrappers
            
        Returns:
            List[str]: Scene implementation plans
            
        Raises:
            ValueError: If required parameters are empty
            Exception: If scene implementation generation fails
        """
        if not topic or not topic.strip():
            raise ValueError("Topic cannot be empty")
        
        if not description or not description.strip():
            raise ValueError("Description cannot be empty")
        
        if not plan or not plan.strip():
            raise ValueError("Plan cannot be empty")
        
        logger.info(f"Generating scene implementations for topic: {topic}")
        
        try:
            video_planner = self._get_video_planner(model_wrappers)
            
            scene_implementations = await video_planner.generate_scene_implementation_concurrently_enhanced(
                topic=topic,
                description=description,
                plan=plan,
                session_id=session_id
            )
            
            if not scene_implementations:
                raise ValueError("Failed to generate scene implementations - empty result")
            
            logger.info(f"Successfully generated {len(scene_implementations)} scene implementations")
            return scene_implementations
            
        except Exception as e:
            logger.error(f"Error generating scene implementations: {e}")
            raise
    
    async def detect_plugins(self, 
                           topic: str, 
                           description: str,
                           model_wrappers: Dict[str, Any]) -> List[str]:
        """Detect relevant plugins for the video topic.
        
        Args:
            topic: Video topic
            description: Video description
            model_wrappers: Dictionary containing model wrappers
            
        Returns:
            List[str]: List of detected relevant plugins
            
        Raises:
            ValueError: If topic or description is empty
        """
        if not topic or not topic.strip():
            raise ValueError("Topic cannot be empty")
        
        if not description or not description.strip():
            raise ValueError("Description cannot be empty")
        
        logger.info(f"Detecting plugins for topic: {topic}")
        
        try:
            video_planner = self._get_video_planner(model_wrappers)
            
            # Check if RAG integration is available for plugin detection
            if hasattr(video_planner, 'rag_integration') and video_planner.rag_integration:
                detected_plugins = await video_planner._detect_plugins_async(topic, description)
                logger.info(f"Detected {len(detected_plugins)} plugins: {detected_plugins}")
                return detected_plugins
            else:
                logger.warning("RAG integration not available for plugin detection")
                return []
            
        except Exception as e:
            logger.error(f"Error detecting plugins: {e}")
            # Plugin detection failure should not stop the workflow
            return []
    
    async def validate_scene_outline(self, scene_outline: str) -> Tuple[bool, List[str]]:
        """Validate the generated scene outline for completeness and structure.
        
        Args:
            scene_outline: The scene outline to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_issues)
        """
        issues = []
        
        if not scene_outline or not scene_outline.strip():
            issues.append("Scene outline is empty")
            return False, issues
        
        # Check for minimum length
        if len(scene_outline.strip()) < 50:
            issues.append("Scene outline is too short (minimum 50 characters)")
        
        # Check for scene structure indicators
        scene_indicators = ['scene', 'Scene', 'SCENE', '1.', '2.', '3.']
        has_scene_structure = any(indicator in scene_outline for indicator in scene_indicators)
        
        if not has_scene_structure:
            issues.append("Scene outline lacks clear scene structure")
        
        # Check for basic content elements
        content_elements = ['show', 'display', 'animate', 'create', 'draw', 'explain']
        has_content_elements = any(element in scene_outline.lower() for element in content_elements)
        
        if not has_content_elements:
            issues.append("Scene outline lacks actionable content elements")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info("Scene outline validation passed")
        else:
            logger.warning(f"Scene outline validation failed with issues: {issues}")
        
        return is_valid, issues
    
    async def validate_scene_implementations(self, 
                                           scene_implementations: List[str]) -> Tuple[bool, Dict[int, List[str]]]:
        """Validate the generated scene implementations.
        
        Args:
            scene_implementations: List of scene implementations to validate
            
        Returns:
            Tuple[bool, Dict[int, List[str]]]: (all_valid, issues_by_scene)
        """
        issues_by_scene = {}
        all_valid = True
        
        if not scene_implementations:
            issues_by_scene[0] = ["No scene implementations provided"]
            return False, issues_by_scene
        
        for i, implementation in enumerate(scene_implementations, 1):
            scene_issues = []
            
            if not implementation or not implementation.strip():
                scene_issues.append("Scene implementation is empty")
            elif len(implementation.strip()) < 30:
                scene_issues.append("Scene implementation is too short (minimum 30 characters)")
            
            # Check for error indicators
            error_indicators = ['Error:', 'error:', 'ERROR:', 'Failed', 'failed', 'FAILED']
            has_errors = any(indicator in implementation for indicator in error_indicators)
            
            if has_errors:
                scene_issues.append("Scene implementation contains error indicators")
            
            # Check for basic implementation elements
            implementation_elements = ['manim', 'Manim', 'animation', 'object', 'scene']
            has_implementation_elements = any(element in implementation.lower() for element in implementation_elements)
            
            if not has_implementation_elements:
                scene_issues.append("Scene implementation lacks Manim-specific elements")
            
            if scene_issues:
                issues_by_scene[i] = scene_issues
                all_valid = False
        
        if all_valid:
            logger.info(f"All {len(scene_implementations)} scene implementations validation passed")
        else:
            logger.warning(f"Scene implementations validation failed for scenes: {list(issues_by_scene.keys())}")
        
        return all_valid, issues_by_scene
    
    def get_planning_metrics(self) -> Dict[str, Any]:
        """Get planning service metrics and statistics.
        
        Returns:
            Dict[str, Any]: Planning metrics
        """
        metrics = {
            'service_name': 'PlanningService',
            'config': {
                'use_rag': self.use_rag,
                'use_context_learning': self.use_context_learning,
                'use_enhanced_rag': self.use_enhanced_rag,
                'max_scene_concurrency': self.max_scene_concurrency,
                'enable_caching': self.enable_caching
            },
            'video_planner_initialized': self._video_planner is not None
        }
        
        # Add video planner metrics if available
        if self._video_planner and hasattr(self._video_planner, 'get_metrics'):
            try:
                planner_metrics = self._video_planner.get_metrics()
                metrics['planner_metrics'] = planner_metrics
            except Exception as e:
                logger.warning(f"Could not retrieve planner metrics: {e}")
        
        return metrics
    
    async def cleanup(self):
        """Cleanup resources used by the planning service."""
        try:
            if self._video_planner and hasattr(self._video_planner, 'thread_pool'):
                self._video_planner.thread_pool.shutdown(wait=False)
                logger.info("Planning service thread pool shutdown")
            
            if self._rag_integration:
                # Cleanup RAG integration if needed
                self._rag_integration = None
                
        except Exception as e:
            logger.warning(f"Error during planning service cleanup: {e}")
    
    def __del__(self):
        """Cleanup resources when service is destroyed."""
        try:
            if self._video_planner and hasattr(self._video_planner, 'thread_pool'):
                self._video_planner.thread_pool.shutdown(wait=False)
        except Exception:
            pass  # Ignore cleanup errors in destructor
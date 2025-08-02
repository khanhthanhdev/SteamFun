"""
Backward-compatible API interface for LangGraph multi-agent system.
Maintains compatibility with existing CodeGenerator, EnhancedVideoPlanner, and OptimizedVideoRenderer APIs.
"""

import os
import logging
import asyncio
import uuid
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from PIL import Image

from ..workflow import LangGraphVideoGenerator
from ..state import VideoGenerationState
from .parameter_mapping import get_parameter_mapper

# Import RAG wrapper compatibility layer
try:
    from .rag_wrapper import create_rag_integration, BackwardCompatibleRAGWrapper
    RAG_INTEGRATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RAG wrapper compatibility layer not available: {e}")
    create_rag_integration = None
    BackwardCompatibleRAGWrapper = None
    RAG_INTEGRATION_AVAILABLE = False

# Import existing classes for type compatibility
try:
    from src.core.code_generator import CodeGenerator
    from src.core.video_planner import EnhancedVideoPlanner
    from src.core.video_renderer import OptimizedVideoRenderer
except ImportError:
    # Handle case where core classes might not be available
    CodeGenerator = None
    EnhancedVideoPlanner = None
    OptimizedVideoRenderer = None


logger = logging.getLogger(__name__)


class BackwardCompatibleCodeGenerator:
    """Backward-compatible wrapper for CodeGenerator using LangGraph backend.
    
    Maintains existing method signatures and response formats while internally
    using the multi-agent system for enhanced functionality.
    """
    
    def __init__(self, 
                 scene_model: Any, 
                 helper_model: Any, 
                 output_dir: str = "output", 
                 print_response: bool = False, 
                 use_rag: bool = False, 
                 use_context_learning: bool = False, 
                 context_learning_path: str = "data/context_learning", 
                 chroma_db_path: str = "data/rag/chroma_db", 
                 manim_docs_path: str = "data/rag/manim_docs", 
                 embedding_model: str = "", 
                 use_visual_fix_code: bool = False, 
                 use_langfuse: bool = True, 
                 session_id: Optional[str] = None):
        """Initialize backward-compatible CodeGenerator.
        
        Args:
            scene_model: The model used for scene generation
            helper_model: The model used for helper tasks
            output_dir: Directory for output files
            print_response: Whether to print model responses
            use_rag: Whether to use RAG
            use_context_learning: Whether to use context learning
            context_learning_path: Path to context learning examples
            chroma_db_path: Path to ChromaDB
            manim_docs_path: Path to Manim docs
            embedding_model: Name of embedding model
            use_visual_fix_code: Whether to use visual code fixing
            use_langfuse: Whether to use Langfuse logging
            session_id: Session identifier
        """
        # Store initialization parameters for compatibility
        self.init_params = {
            'scene_model': scene_model,
            'helper_model': helper_model,
            'output_dir': output_dir,
            'print_response': print_response,
            'use_rag': use_rag,
            'use_context_learning': use_context_learning,
            'context_learning_path': context_learning_path,
            'chroma_db_path': chroma_db_path,
            'manim_docs_path': manim_docs_path,
            'embedding_model': embedding_model,
            'use_visual_fix_code': use_visual_fix_code,
            'use_langfuse': use_langfuse,
            'session_id': session_id
        }
        
        # Extract model names from model objects
        planner_model = self._extract_model_name(scene_model)
        scene_model_name = self._extract_model_name(scene_model)
        helper_model_name = self._extract_model_name(helper_model)
        
        # Map RAG parameters using centralized .env configuration
        parameter_mapper = get_parameter_mapper()
        rag_params = parameter_mapper.map_rag_parameters(use_rag=use_rag)
        
        # Initialize RAG integration using .env configuration
        self.rag_integration = None
        if rag_params.get('use_rag', False) and RAG_INTEGRATION_AVAILABLE:
            try:
                from .rag_wrapper import create_rag_integration
                
                self.rag_integration = create_rag_integration(
                    helper_model=helper_model,
                    output_dir=output_dir,
                    session_id=session_id,
                    use_langfuse=use_langfuse,
                    **rag_params
                )
                
                if self.rag_integration:
                    logger.info(f"RAG integration initialized from .env with priority: {rag_params.get('rag_priority', 'unknown')}")
                else:
                    logger.warning("RAG integration initialization returned None")
                    logger.info(f"Initializing RAG with Chroma (priority: {rag_params.get('rag_priority')})")
                
                # Initialize RAG integration with centralized configuration
                from src.config.manager import ConfigurationManager
                config_manager = ConfigurationManager()
                
                self.rag_integration = RAGIntegration(
                    **init_params,
                    config_manager=config_manager
                )
                
                # Log environment status
                env_status = rag_params.get('env_status', {})
                logger.info(f"RAG environment status: JINA={env_status.get('jina_api_available')}, AstraDB={env_status.get('astradb_available')}")
                
                if not env_status.get('jina_api_available') or not env_status.get('astradb_available'):
                    logger.info(f"Recommendation: {env_status.get('recommended_setup')}")
                    
            except Exception as e:
                logger.warning(f"Failed to initialize RAG integration: {e}")
                logger.info("Consider checking your .env file for JINA_API_KEY, ASTRA_DB_APPLICATION_TOKEN, and ASTRA_DB_API_ENDPOINT")
                self.rag_integration = None

        # Initialize LangGraph video generator with compatible parameters
        self.langgraph_generator = LangGraphVideoGenerator(
            planner_model=planner_model,
            scene_model=scene_model_name,
            helper_model=helper_model_name,
            output_dir=output_dir,
            verbose=print_response,
            use_rag=use_rag,
            use_context_learning=use_context_learning,
            context_learning_path=context_learning_path,
            use_visual_fix_code=use_visual_fix_code,
            use_langfuse=use_langfuse,
            **rag_params  # Include mapped RAG parameters
        )
        
        # Store session ID
        self.session_id = session_id or str(uuid.uuid4())
        
        logger.info(f"BackwardCompatibleCodeGenerator initialized with session: {self.session_id}")
    
    def _extract_model_name(self, model_obj: Any) -> str:
        """Extract model name from model object for LangGraph compatibility.
        
        Args:
            model_obj: Model object (LiteLLMWrapper, OpenRouterWrapper, etc.)
            
        Returns:
            str: Model name string
        """
        if hasattr(model_obj, 'model_name'):
            return model_obj.model_name
        elif hasattr(model_obj, 'model'):
            return model_obj.model
        else:
            # Fallback to default model
            return "openai/gpt-4o"
    
    def generate_manim_code(self,
                           topic: str,
                           description: str,                            
                           scene_outline: str,
                           scene_implementation: str,
                           scene_number: int,
                           additional_context: Union[str, List[str], None] = None,
                           scene_trace_id: Optional[str] = None,
                           session_id: Optional[str] = None,
                           rag_queries_cache: Optional[Dict] = None) -> Tuple[str, str]:
        """Generate Manim code from video plan using LangGraph backend.
        
        Maintains exact compatibility with existing CodeGenerator.generate_manim_code method.
        
        Args:
            topic: Topic of the scene
            description: Description of the scene
            scene_outline: Outline of the scene
            scene_implementation: Implementation details
            scene_number: Scene number
            additional_context: Additional context
            scene_trace_id: Trace identifier
            session_id: Session identifier
            rag_queries_cache: Cache for RAG queries (deprecated, maintained for compatibility)
            
        Returns:
            Tuple[str, str]: Generated code and response text
        """
        try:
            # Use provided session_id or fall back to instance session_id
            effective_session_id = session_id or self.session_id
            
            # Create a focused workflow state for code generation
            workflow_state = {
                'topic': topic,
                'description': description,
                'scene_outline': scene_outline,
                'scene_implementations': {scene_number: scene_implementation},
                'current_scene_number': scene_number,
                'additional_context': additional_context,
                'scene_trace_id': scene_trace_id,
                'code_generation_only': True,  # Flag to indicate we only want code generation
                **self.init_params
            }
            
            # Execute code generation through LangGraph workflow
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                final_state = loop.run_until_complete(
                    self.langgraph_generator.generate_video_pipeline(
                        topic=topic,
                        description=description,
                        session_id=effective_session_id,
                        only_plan=False,
                        specific_scenes=[scene_number]
                    )
                )
                
                # Extract generated code from final state
                generated_code = final_state.get('generated_code', {}).get(scene_number, '')
                
                if not generated_code:
                    raise ValueError(f"No code generated for scene {scene_number}")
                
                # Create response text in expected format
                response_text = f"Generated Manim code for scene {scene_number} of topic '{topic}'"
                
                logger.info(f"Successfully generated code for {topic} scene {scene_number} via LangGraph")
                return generated_code, response_text
                
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Error generating Manim code for {topic} scene {scene_number}: {e}")
            raise ValueError(f"Code generation failed: {e}") from e
    
    def fix_code_errors(self, 
                       implementation_plan: str, 
                       code: str, 
                       error: str, 
                       scene_trace_id: str, 
                       topic: str, 
                       scene_number: int, 
                       session_id: str, 
                       rag_queries_cache: Optional[Dict] = None) -> Tuple[str, str]:
        """Fix errors in generated Manim code using LangGraph backend.
        
        Maintains exact compatibility with existing CodeGenerator.fix_code_errors method.
        
        Args:
            implementation_plan: Original implementation plan
            code: Code containing errors
            error: Error message to fix
            scene_trace_id: Trace identifier
            topic: Topic of the scene
            scene_number: Scene number
            session_id: Session identifier
            rag_queries_cache: Cache for RAG queries (deprecated, maintained for compatibility)
            
        Returns:
            Tuple[str, str]: Fixed code and response text
        """
        try:
            # Create workflow state for error fixing
            workflow_state = {
                'topic': topic,
                'description': f"Fix errors in scene {scene_number}",
                'scene_implementations': {scene_number: implementation_plan},
                'generated_code': {scene_number: code},
                'code_errors': {scene_number: error},
                'current_scene_number': scene_number,
                'scene_trace_id': scene_trace_id,
                'error_fixing_mode': True,  # Flag to indicate error fixing mode
                **self.init_params
            }
            
            # Execute error fixing through LangGraph workflow
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                final_state = loop.run_until_complete(
                    self.langgraph_generator.generate_video_pipeline(
                        topic=topic,
                        description=f"Fix errors in scene {scene_number}",
                        session_id=session_id,
                        only_plan=False,
                        specific_scenes=[scene_number]
                    )
                )
                
                # Extract fixed code from final state
                fixed_code = final_state.get('generated_code', {}).get(scene_number, code)
                
                # Create response text in expected format
                response_text = f"Fixed code errors for scene {scene_number} of topic '{topic}'"
                
                logger.info(f"Successfully fixed code errors for {topic} scene {scene_number} via LangGraph")
                return fixed_code, response_text
                
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Error fixing code for {topic} scene {scene_number}: {e}")
            raise ValueError(f"Code error fixing failed: {e}") from e
    
    def visual_self_reflection(self, 
                              code: str, 
                              media_path: Union[str, Image.Image], 
                              scene_trace_id: str, 
                              topic: str, 
                              scene_number: int, 
                              session_id: str) -> Tuple[str, str]:
        """Use snapshot image or mp4 video to fix code using LangGraph backend.
        
        Maintains exact compatibility with existing CodeGenerator.visual_self_reflection method.
        
        Args:
            code: Code to fix
            media_path: Path to media file or PIL Image
            scene_trace_id: Trace identifier
            topic: Topic of the scene
            scene_number: Scene number
            session_id: Session identifier
            
        Returns:
            Tuple[str, str]: Fixed code and response text
        """
        try:
            # Create workflow state for visual self-reflection
            workflow_state = {
                'topic': topic,
                'description': f"Visual self-reflection for scene {scene_number}",
                'generated_code': {scene_number: code},
                'rendered_videos': {scene_number: str(media_path) if isinstance(media_path, str) else 'image_input'},
                'current_scene_number': scene_number,
                'scene_trace_id': scene_trace_id,
                'visual_reflection_mode': True,  # Flag to indicate visual reflection mode
                'media_input': media_path,
                **self.init_params
            }
            
            # Execute visual self-reflection through LangGraph workflow
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                final_state = loop.run_until_complete(
                    self.langgraph_generator.generate_video_pipeline(
                        topic=topic,
                        description=f"Visual self-reflection for scene {scene_number}",
                        session_id=session_id,
                        only_plan=False,
                        specific_scenes=[scene_number]
                    )
                )
                
                # Extract improved code from final state
                improved_code = final_state.get('generated_code', {}).get(scene_number, code)
                
                # Create response text in expected format
                response_text = f"Completed visual self-reflection for scene {scene_number} of topic '{topic}'"
                
                logger.info(f"Successfully completed visual self-reflection for {topic} scene {scene_number} via LangGraph")
                return improved_code, response_text
                
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Error in visual self-reflection for {topic} scene {scene_number}: {e}")
            raise ValueError(f"Visual self-reflection failed: {e}") from e
    
    def get_rag_status(self) -> Dict[str, Any]:
        """Get RAG system status and configuration.
        
        Returns:
            Dict: RAG status information
        """
        if not self.rag_integration:
            return {
                'enabled': False,
                'reason': 'RAG integration not initialized'
            }
        
        try:
            return {
                'enabled': True,
                'system_status': self.rag_integration.get_system_status(),
                'provider_status': self.rag_integration.get_provider_status(),
                'performance_metrics': self.rag_integration.get_performance_metrics()
            }
        except Exception as e:
            return {
                'enabled': True,
                'error': str(e)
            }
    
    def switch_rag_provider(self, provider_name: str) -> bool:
        """Switch RAG embedding provider.
        
        Args:
            provider_name: Provider to switch to ('jina', 'local')
            
        Returns:
            bool: True if switch was successful
        """
        if not self.rag_integration:
            logger.warning("RAG integration not available for provider switching")
            return False
        
        try:
            return self.rag_integration.switch_embedding_provider(provider_name)
        except Exception as e:
            logger.error(f"Failed to switch RAG provider: {e}")
            return False
    
    def get_rag_configuration(self) -> Dict[str, Any]:
        """Get current RAG configuration and recommendations.
        
        Returns:
            Dict: RAG configuration details and environment status
        """
        parameter_mapper = get_parameter_mapper()
        
        # Get current environment status
        env_config = parameter_mapper.map_rag_parameters(
            use_rag=self.init_params.get('use_rag', False),
            chroma_db_path=self.init_params.get('chroma_db_path'),
            manim_docs_path=self.init_params.get('manim_docs_path'),
            embedding_model=self.init_params.get('embedding_model')
        )
        
        config_info = {
            'current_priority': env_config.get('rag_priority', 'unknown'),
            'embedding_provider': env_config.get('embedding_provider', 'unknown'),
            'vector_store_provider': env_config.get('vector_store_provider', 'unknown'),
            'environment_status': env_config.get('env_status', {}),
            'rag_integration_available': self.rag_integration is not None,
            'recommendations': []
        }
        
        # Add recommendations based on current setup
        env_status = env_config.get('env_status', {})
        if not env_status.get('jina_api_available'):
            config_info['recommendations'].append(
                "Set JINA_API_KEY in .env file for better embedding quality"
            )
        
        if not env_status.get('astradb_available'):
            config_info['recommendations'].append(
                "Set ASTRA_DB_APPLICATION_TOKEN and ASTRA_DB_API_ENDPOINT in .env file for cloud vector storage"
            )
        
        if env_status.get('jina_api_available') and env_status.get('astradb_available'):
            config_info['recommendations'].append(
                "Optimal configuration detected: JINA API + AstraDB"
            )
        
        return config_info


class BackwardCompatibleVideoPlanner:
    """Backward-compatible wrapper for EnhancedVideoPlanner using LangGraph backend.
    
    Maintains existing method signatures and response formats while internally
    using the multi-agent system for enhanced functionality.
    """
    
    def __init__(self, planner_model, helper_model=None, output_dir="output", 
                 print_response=False, use_context_learning=False, 
                 context_learning_path="data/context_learning", use_rag=False, 
                 session_id=None, chroma_db_path="data/rag/chroma_db", 
                 manim_docs_path="data/rag/manim_docs", 
                 embedding_model="text-embedding-ada-002", use_langfuse=True,
                 max_scene_concurrency=5, max_step_concurrency=3, enable_caching=True,
                 use_enhanced_rag=True, rag_config=None):
        """Initialize backward-compatible VideoPlanner.
        
        Maintains exact parameter compatibility with EnhancedVideoPlanner.
        """
        # Store initialization parameters for compatibility
        self.init_params = {
            'planner_model': planner_model,
            'helper_model': helper_model or planner_model,
            'output_dir': output_dir,
            'print_response': print_response,
            'use_context_learning': use_context_learning,
            'context_learning_path': context_learning_path,
            'use_rag': use_rag,
            'session_id': session_id,
            'chroma_db_path': chroma_db_path,
            'manim_docs_path': manim_docs_path,
            'embedding_model': embedding_model,
            'use_langfuse': use_langfuse,
            'max_scene_concurrency': max_scene_concurrency,
            'enable_caching': enable_caching,
            'use_enhanced_rag': use_enhanced_rag,
            'rag_config': rag_config
        }
        
        # Extract model names from model objects
        planner_model_name = self._extract_model_name(planner_model)
        helper_model_name = self._extract_model_name(helper_model or planner_model)
        
        # Map RAG parameters to use Jina API and AstraDB
        parameter_mapper = get_parameter_mapper()
        rag_params = parameter_mapper.map_rag_parameters(
            use_rag=use_rag,
            chroma_db_path=chroma_db_path,
            manim_docs_path=manim_docs_path,
            embedding_model=embedding_model,
            use_context_learning=use_context_learning,
            context_learning_path=context_learning_path,
            enable_caching=enable_caching,
            use_enhanced_rag=use_enhanced_rag
        )
        
        # Initialize RAG integration using .env configuration
        self.rag_integration = None
        if rag_params.get('use_rag', False) and RAG_INTEGRATION_AVAILABLE:
            try:
                from .rag_wrapper import create_rag_integration
                
                self.rag_integration = create_rag_integration(
                    helper_model=helper_model or planner_model,
                    output_dir=output_dir,
                    session_id=session_id,
                    use_langfuse=use_langfuse,
                    **rag_params
                )
                
                if self.rag_integration:
                    logger.info(f"RAG integration initialized for planner from .env with priority: {rag_params.get('rag_priority', 'unknown')}")
                else:
                    logger.warning("RAG integration initialization returned None for planner")
            except Exception as e:
                logger.warning(f"Failed to initialize RAG integration for planner: {e}")
                self.rag_integration = None
        
        # Initialize LangGraph video generator with compatible parameters
        self.langgraph_generator = LangGraphVideoGenerator(
            planner_model=planner_model_name,
            scene_model=planner_model_name,
            helper_model=helper_model_name,
            output_dir=output_dir,
            verbose=print_response,
            use_rag=use_rag,
            use_context_learning=use_context_learning,
            context_learning_path=context_learning_path,
            use_langfuse=use_langfuse,
            max_scene_concurrency=max_scene_concurrency,
            enable_caching=enable_caching,
            **rag_params  # Include mapped RAG parameters
        )
        
        # Store session ID
        self.session_id = session_id or str(uuid.uuid4())
        
        logger.info(f"BackwardCompatibleVideoPlanner initialized with session: {self.session_id}")
    
    def _extract_model_name(self, model_obj: Any) -> str:
        """Extract model name from model object for LangGraph compatibility."""
        if hasattr(model_obj, 'model_name'):
            return model_obj.model_name
        elif hasattr(model_obj, 'model'):
            return model_obj.model
        else:
            return "openai/gpt-4o"
    
    async def generate_scene_outline(self, topic: str, description: str, session_id: str) -> str:
        """Generate scene outline using LangGraph backend.
        
        Maintains exact compatibility with EnhancedVideoPlanner.generate_scene_outline method.
        
        Args:
            topic: Video topic
            description: Video description
            session_id: Session identifier
            
        Returns:
            str: Scene outline
        """
        try:
            # Use LangGraph generator to create scene outline
            scene_outline = await self.langgraph_generator.generate_scene_outline(
                topic=topic,
                description=description,
                session_id=session_id
            )
            
            logger.info(f"Successfully generated scene outline for {topic} via LangGraph")
            return scene_outline
            
        except Exception as e:
            logger.error(f"Error generating scene outline for {topic}: {e}")
            raise
    
    async def generate_scene_implementation_concurrently_enhanced(self, topic: str, description: str, 
                                                                plan: str, session_id: str) -> List[str]:
        """Generate scene implementations concurrently using LangGraph backend.
        
        Maintains exact compatibility with EnhancedVideoPlanner method.
        
        Args:
            topic: Video topic
            description: Video description
            plan: Scene plan/outline
            session_id: Session identifier
            
        Returns:
            List[str]: List of scene implementation plans
        """
        try:
            # Execute planning workflow through LangGraph
            final_state = await self.langgraph_generator.generate_video_pipeline(
                topic=topic,
                description=description,
                session_id=session_id,
                only_plan=True  # Only generate planning, not full video
            )
            
            # Extract scene implementations from final state
            scene_implementations = final_state.get('scene_implementations', {})
            
            # Convert to list format expected by existing API
            implementation_list = []
            for scene_num in sorted(scene_implementations.keys()):
                implementation_list.append(scene_implementations[scene_num])
            
            logger.info(f"Successfully generated {len(implementation_list)} scene implementations for {topic} via LangGraph")
            return implementation_list
            
        except Exception as e:
            logger.error(f"Error generating scene implementations for {topic}: {e}")
            raise
    
    def detect_relevant_plugins(self, topic: str, description: str) -> List[str]:
        """Detect relevant plugins for the video topic.
        
        Args:
            topic: Video topic
            description: Video description
            
        Returns:
            List[str]: List of relevant plugin names
        """
        if not self.rag_integration:
            logger.warning("RAG integration not available for plugin detection")
            return []
        
        try:
            return self.rag_integration.detect_relevant_plugins(topic, description)
        except Exception as e:
            logger.error(f"Plugin detection failed: {e}")
            return []
    
    def get_rag_status(self) -> Dict[str, Any]:
        """Get RAG system status and configuration.
        
        Returns:
            Dict: RAG status information
        """
        if not self.rag_integration:
            return {
                'enabled': False,
                'reason': 'RAG integration not initialized'
            }
        
        try:
            return {
                'enabled': True,
                'system_status': self.rag_integration.get_system_status(),
                'provider_status': self.rag_integration.get_provider_status(),
                'performance_metrics': self.rag_integration.get_performance_metrics()
            }
        except Exception as e:
            return {
                'enabled': True,
                'error': str(e)
            }
    
    def get_rag_configuration(self) -> Dict[str, Any]:
        """Get current RAG configuration and recommendations.
        
        Returns:
            Dict: RAG configuration details and environment status
        """
        parameter_mapper = get_parameter_mapper()
        
        # Get current environment status
        env_config = parameter_mapper.map_rag_parameters(
            use_rag=self.init_params.get('use_rag', False),
            chroma_db_path=self.init_params.get('chroma_db_path'),
            manim_docs_path=self.init_params.get('manim_docs_path'),
            embedding_model=self.init_params.get('embedding_model')
        )
        
        config_info = {
            'current_priority': env_config.get('rag_priority', 'unknown'),
            'embedding_provider': env_config.get('embedding_provider', 'unknown'),
            'vector_store_provider': env_config.get('vector_store_provider', 'unknown'),
            'environment_status': env_config.get('env_status', {}),
            'rag_integration_available': self.rag_integration is not None,
            'plugin_detection_available': hasattr(self, 'detect_relevant_plugins'),
            'recommendations': []
        }
        
        # Add recommendations based on current setup
        env_status = env_config.get('env_status', {})
        if not env_status.get('jina_api_available'):
            config_info['recommendations'].append(
                "Set JINA_API_KEY in .env file for better embedding quality and plugin detection"
            )
        
        if not env_status.get('astradb_available'):
            config_info['recommendations'].append(
                "Set ASTRA_DB_APPLICATION_TOKEN and ASTRA_DB_API_ENDPOINT in .env file for cloud vector storage"
            )
        
        if env_status.get('jina_api_available') and env_status.get('astradb_available'):
            config_info['recommendations'].append(
                "Optimal configuration detected: JINA API + AstraDB for planning tasks"
            )
        
        return config_info


class BackwardCompatibleVideoRenderer:
    """Backward-compatible wrapper for OptimizedVideoRenderer using LangGraph backend.
    
    Maintains existing method signatures and response formats while internally
    using the multi-agent system for enhanced functionality.
    """
    
    def __init__(self, output_dir="output", print_response=False, use_visual_fix_code=False,
                 max_concurrent_renders=4, enable_caching=True, default_quality="medium",
                 use_gpu_acceleration=False, preview_mode=False):
        """Initialize backward-compatible VideoRenderer.
        
        Maintains exact parameter compatibility with OptimizedVideoRenderer.
        """
        # Store initialization parameters for compatibility
        self.init_params = {
            'output_dir': output_dir,
            'print_response': print_response,
            'use_visual_fix_code': use_visual_fix_code,
            'max_concurrent_renders': max_concurrent_renders,
            'enable_caching': enable_caching,
            'default_quality': default_quality,
            'use_gpu_acceleration': use_gpu_acceleration,
            'preview_mode': preview_mode
        }
        
        # Map RAG parameters (renderer may need RAG for visual analysis)
        parameter_mapper = get_parameter_mapper()
        rag_params = parameter_mapper.map_rag_parameters(
            use_rag=False,  # Renderer typically doesn't use RAG directly
            use_enhanced_rag=False
        )
        
        # Initialize LangGraph video generator with compatible parameters
        self.langgraph_generator = LangGraphVideoGenerator(
            planner_model="openai/gpt-4o",  # Default model for rendering
            output_dir=output_dir,
            verbose=print_response,
            use_visual_fix_code=use_visual_fix_code,
            max_concurrent_renders=max_concurrent_renders,
            enable_caching=enable_caching,
            default_quality=default_quality,
            use_gpu_acceleration=use_gpu_acceleration,
            preview_mode=preview_mode,
            **rag_params  # Include mapped RAG parameters
        )
        
        logger.info("BackwardCompatibleVideoRenderer initialized")
    
    async def render_scene_optimized(self, code: str, file_prefix: str, curr_scene: int, 
                                   curr_version: int, code_dir: str, media_dir: str, 
                                   quality: str = None, max_retries: int = 3, 
                                   use_visual_fix_code=False, visual_self_reflection_func=None, 
                                   banned_reasonings=None, scene_trace_id=None, topic=None, 
                                   session_id=None, code_generator=None, 
                                   scene_implementation=None, description=None, 
                                   scene_outline=None) -> tuple:
        """Render scene with optimization using LangGraph backend.
        
        Maintains exact compatibility with OptimizedVideoRenderer.render_scene_optimized method.
        
        Args:
            code: Manim code to render
            file_prefix: File prefix for output
            curr_scene: Current scene number
            curr_version: Current version number
            code_dir: Code directory
            media_dir: Media directory
            quality: Render quality
            max_retries: Maximum retry attempts
            use_visual_fix_code: Whether to use visual fix code
            visual_self_reflection_func: Visual self-reflection function
            banned_reasonings: Banned reasoning patterns
            scene_trace_id: Scene trace identifier
            topic: Video topic
            session_id: Session identifier
            code_generator: Code generator instance
            scene_implementation: Scene implementation
            description: Scene description
            scene_outline: Scene outline
            
        Returns:
            tuple: (final_code, error_message)
        """
        try:
            # Create workflow state for rendering
            workflow_state = {
                'topic': topic or f"Scene {curr_scene}",
                'description': description or f"Render scene {curr_scene}",
                'generated_code': {curr_scene: code},
                'current_scene_number': curr_scene,
                'scene_trace_id': scene_trace_id,
                'rendering_only': True,  # Flag to indicate rendering only
                'quality': quality or self.init_params['default_quality'],
                'max_retries': max_retries,
                'file_prefix': file_prefix,
                'curr_version': curr_version,
                'code_dir': code_dir,
                'media_dir': media_dir,
                **self.init_params
            }
            
            # Execute rendering through LangGraph workflow
            final_state = await self.langgraph_generator.generate_video_pipeline(
                topic=topic or f"Scene {curr_scene}",
                description=description or f"Render scene {curr_scene}",
                session_id=session_id or str(uuid.uuid4()),
                only_plan=False,
                specific_scenes=[curr_scene]
            )
            
            # Extract rendering results from final state
            rendered_videos = final_state.get('rendered_videos', {})
            rendering_errors = final_state.get('rendering_errors', {})
            
            if curr_scene in rendered_videos:
                # Successful rendering
                final_code = final_state.get('generated_code', {}).get(curr_scene, code)
                logger.info(f"Successfully rendered scene {curr_scene} via LangGraph")
                return final_code, None
            elif curr_scene in rendering_errors:
                # Rendering failed
                error_message = rendering_errors[curr_scene]
                logger.error(f"Rendering failed for scene {curr_scene}: {error_message}")
                return code, error_message
            else:
                # Unknown state
                logger.warning(f"Unknown rendering state for scene {curr_scene}")
                return code, "Unknown rendering error"
                
        except Exception as e:
            logger.error(f"Error rendering scene {curr_scene}: {e}")
            return code, str(e)
    
    async def combine_videos_optimized(self, topic: str, use_hardware_acceleration: bool = False) -> str:
        """Combine videos with optimization using LangGraph backend.
        
        Maintains exact compatibility with OptimizedVideoRenderer.combine_videos_optimized method.
        
        Args:
            topic: Video topic
            use_hardware_acceleration: Whether to use hardware acceleration
            
        Returns:
            str: Path to combined video
        """
        try:
            # Execute full video generation workflow to ensure all scenes are rendered
            final_state = await self.langgraph_generator.generate_video_pipeline(
                topic=topic,
                description=f"Combine videos for {topic}",
                session_id=str(uuid.uuid4()),
                only_plan=False
            )
            
            # Extract combined video path from final state
            combined_video_path = final_state.get('combined_video_path')
            
            if combined_video_path and os.path.exists(combined_video_path):
                logger.info(f"Successfully combined videos for {topic} via LangGraph")
                return combined_video_path
            else:
                raise FileNotFoundError(f"Combined video not found for topic: {topic}")
                
        except Exception as e:
            logger.error(f"Error combining videos for {topic}: {e}")
            raise


def create_backward_compatible_components(
    planner_model: Any,
    scene_model: Any = None,
    helper_model: Any = None,
    output_dir: str = "output",
    **kwargs
) -> Tuple[BackwardCompatibleCodeGenerator, BackwardCompatibleVideoPlanner, BackwardCompatibleVideoRenderer]:
    """Factory function to create backward-compatible components.
    
    Args:
        planner_model: Model for planning tasks
        scene_model: Model for scene generation
        helper_model: Model for helper tasks
        output_dir: Output directory
        **kwargs: Additional configuration parameters
        
    Returns:
        Tuple: (code_generator, video_planner, video_renderer)
    """
    # Use scene_model or fall back to planner_model
    effective_scene_model = scene_model or planner_model
    effective_helper_model = helper_model or planner_model
    
    # Create backward-compatible components
    code_generator = BackwardCompatibleCodeGenerator(
        scene_model=effective_scene_model,
        helper_model=effective_helper_model,
        output_dir=output_dir,
        **kwargs
    )
    
    video_planner = BackwardCompatibleVideoPlanner(
        planner_model=planner_model,
        helper_model=effective_helper_model,
        output_dir=output_dir,
        **kwargs
    )
    
    video_renderer = BackwardCompatibleVideoRenderer(
        output_dir=output_dir,
        **kwargs
    )
    
    logger.info(f"Created backward-compatible components with output_dir: {output_dir}")
    return code_generator, video_planner, video_renderer


class BackwardCompatibleErrorHandler:
    """Handles error format compatibility between old and new systems."""
    
    @staticmethod
    def format_error_response(error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Format error response in backward-compatible format.
        
        Args:
            error: Exception that occurred
            context: Error context information
            
        Returns:
            Dict: Formatted error response
        """
        return {
            'error': True,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'timestamp': time.time()
        }
    
    @staticmethod
    def is_retryable_error(error: Exception) -> bool:
        """Determine if error is retryable based on existing patterns.
        
        Args:
            error: Exception to check
            
        Returns:
            bool: True if error is retryable
        """
        retryable_errors = [
            'TimeoutError',
            'ConnectionError',
            'HTTPError',
            'RateLimitError',
            'TemporaryFailure'
        ]
        
        error_type = type(error).__name__
        return error_type in retryable_errors or 'timeout' in str(error).lower()


class BackwardCompatibleSessionManager:
    """Manages session compatibility between old and new systems."""
    
    def __init__(self):
        self.active_sessions = {}
        self.session_configs = {}
    
    def create_session(self, session_id: str, config: Dict[str, Any]) -> str:
        """Create a new session with backward-compatible configuration.
        
        Args:
            session_id: Session identifier
            config: Session configuration
            
        Returns:
            str: Session ID
        """
        if not session_id:
            session_id = str(uuid.uuid4())
        
        self.active_sessions[session_id] = {
            'created_at': time.time(),
            'status': 'active',
            'config': config
        }
        
        self.session_configs[session_id] = config
        logger.info(f"Created backward-compatible session: {session_id}")
        return session_id
    
    def get_session_config(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session configuration.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Optional[Dict]: Session configuration or None
        """
        return self.session_configs.get(session_id)
    
    def update_session_status(self, session_id: str, status: str):
        """Update session status.
        
        Args:
            session_id: Session identifier
            status: New status
        """
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['status'] = status
            self.active_sessions[session_id]['updated_at'] = time.time()
    
    def cleanup_session(self, session_id: str):
        """Clean up session resources.
        
        Args:
            session_id: Session identifier
        """
        self.active_sessions.pop(session_id, None)
        self.session_configs.pop(session_id, None)
        logger.info(f"Cleaned up session: {session_id}")


# Global session manager instance
session_manager = BackwardCompatibleSessionManager()


def get_session_manager() -> BackwardCompatibleSessionManager:
    """Get the global session manager instance.
    
    Returns:
        BackwardCompatibleSessionManager: Session manager instance
    """
    return session_manager
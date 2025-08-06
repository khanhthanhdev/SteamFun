"""
StateAdapter for converting between old TypedDict and new Pydantic state formats.

This adapter ensures backward compatibility while migrating to the new
Pydantic-based state management system.
"""

from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import logging

from ..models.state import VideoGenerationState as NewVideoGenerationState
from ..models.config import WorkflowConfig
from ..models.errors import WorkflowError
from ..models.metrics import PerformanceMetrics
from ..state import VideoGenerationState as OldVideoGenerationState

logger = logging.getLogger(__name__)


class StateAdapter:
    """
    Adapter for converting between old TypedDict and new Pydantic state formats.
    
    Provides bidirectional conversion while maintaining data integrity and
    ensuring all fields are properly mapped between formats.
    """
    
    @staticmethod
    def old_to_new(old_state: Union[OldVideoGenerationState, Dict[str, Any]]) -> NewVideoGenerationState:
        """
        Convert old TypedDict state format to new Pydantic state format.
        
        Args:
            old_state: Old state in TypedDict format
            
        Returns:
            NewVideoGenerationState: Converted state in new Pydantic format
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        try:
            # Handle both TypedDict and regular dict inputs
            if isinstance(old_state, dict):
                state_dict = old_state
            else:
                state_dict = dict(old_state)
            
            # Extract core workflow data with validation
            topic = state_dict.get('topic', '')
            description = state_dict.get('description', '')
            session_id = state_dict.get('session_id', '')
            
            if not topic:
                raise ValueError("Topic is required but missing from old state")
            if not description:
                raise ValueError("Description is required but missing from old state")
            if not session_id:
                raise ValueError("Session ID is required but missing from old state")
            
            # Create workflow configuration from old state parameters
            config = WorkflowConfig(
                # Model configurations - extract from old state
                planner_model=StateAdapter._extract_model_config(
                    state_dict, 'planner_model', 'openrouter/anthropic/claude-3.5-sonnet'
                ),
                code_model=StateAdapter._extract_model_config(
                    state_dict, 'code_model', 'openrouter/anthropic/claude-3.5-sonnet'
                ),
                helper_model=StateAdapter._extract_model_config(
                    state_dict, 'helper_model', 'openrouter/anthropic/claude-3.5-sonnet'
                ),
                
                # RAG configuration
                use_rag=state_dict.get('use_rag', True),
                use_visual_analysis=state_dict.get('use_visual_fix_code', False),
                enable_caching=state_dict.get('enable_caching', True),
                use_context_learning=state_dict.get('use_context_learning', True),
                
                # Performance settings
                max_retries=state_dict.get('max_retries', 3),
                timeout_seconds=300,  # Default timeout
                max_concurrent_scenes=state_dict.get('max_scene_concurrency', 5),
                max_concurrent_renders=state_dict.get('max_concurrent_renders', 4),
                
                # Quality settings
                default_quality=state_dict.get('default_quality', 'medium'),
                use_gpu_acceleration=state_dict.get('use_gpu_acceleration', False),
                preview_mode=state_dict.get('preview_mode', False),
                
                # Directory settings
                output_dir=state_dict.get('output_dir', 'output'),
                context_learning_path=state_dict.get('context_learning_path', 'data/context_learning'),
                manim_docs_path=state_dict.get('manim_docs_path', 'data/rag/manim_docs'),
                chroma_db_path=state_dict.get('chroma_db_path', 'data/rag/chroma_db'),
                embedding_model=state_dict.get('embedding_model', 'hf:ibm-granite/granite-embedding-30m-english'),
                
                # Monitoring settings
                enable_monitoring=True,
                use_langfuse=state_dict.get('use_langfuse', True),
                enable_langfuse=state_dict.get('use_langfuse', True),
                print_cost=state_dict.get('print_cost', True),
                verbose=state_dict.get('print_response', False),
                
                # Enhanced RAG settings
                use_enhanced_rag=state_dict.get('use_enhanced_rag', True),
                max_scene_concurrency=state_dict.get('max_scene_concurrency', 5)
            )
            
            # Convert workflow errors
            errors = []
            escalated_errors = state_dict.get('escalated_errors', [])
            for error_data in escalated_errors:
                if isinstance(error_data, dict):
                    workflow_error = WorkflowError(
                        step=error_data.get('agent', 'unknown'),
                        error_type=error_data.get('error_type', 'UnknownError'),
                        message=error_data.get('error', str(error_data)),
                        timestamp=StateAdapter._parse_timestamp(error_data.get('timestamp')),
                        retry_count=error_data.get('retry_count', 0),
                        recoverable=True,
                        context=error_data
                    )
                    errors.append(workflow_error)
            
            # Create performance metrics
            performance_metrics = PerformanceMetrics(
                session_id=session_id,
                step_durations=StateAdapter._extract_step_durations(state_dict),
                total_duration=0.0,  # Will be calculated
                success_rates=StateAdapter._extract_success_rates(state_dict),
                resource_usage=state_dict.get('performance_metrics', {})
            )
            
            # Determine current step from old state
            current_step = StateAdapter._determine_current_step(state_dict)
            
            # Create new state with all mapped fields
            new_state = NewVideoGenerationState(
                # Core workflow data
                topic=topic,
                description=description,
                session_id=session_id,
                config=config,
                
                # Planning state
                scene_outline=state_dict.get('scene_outline'),
                scene_implementations=state_dict.get('scene_implementations', {}),
                detected_plugins=state_dict.get('detected_plugins', []),
                
                # Code generation state
                generated_code=state_dict.get('generated_code', {}),
                code_errors=state_dict.get('code_errors', {}),
                rag_context=state_dict.get('rag_context', {}),
                
                # Rendering state
                rendered_videos=state_dict.get('rendered_videos', {}),
                combined_video_path=state_dict.get('combined_video_path'),
                rendering_errors=state_dict.get('rendering_errors', {}),
                
                # Visual analysis state
                visual_analysis_results=state_dict.get('visual_analysis_results', {}),
                visual_errors=state_dict.get('visual_errors', {}),
                
                # Error handling
                errors=errors,
                retry_counts=state_dict.get('retry_count', {}),
                escalated_errors=escalated_errors,
                
                # Performance metrics
                metrics=performance_metrics,
                
                # Human loop state
                pending_human_input=state_dict.get('pending_human_input'),
                human_feedback=state_dict.get('human_feedback'),
                
                # Workflow control
                current_step=current_step,
                workflow_complete=state_dict.get('workflow_complete', False),
                workflow_interrupted=state_dict.get('workflow_interrupted', False),
                
                # Execution tracking
                execution_trace=state_dict.get('execution_trace', []),
                
                # Timestamps
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            logger.info(f"Successfully converted old state to new format for session {session_id}")
            return new_state
            
        except Exception as e:
            logger.error(f"Failed to convert old state to new format: {e}")
            raise ValueError(f"State conversion failed: {e}") from e
    
    @staticmethod
    def new_to_old(new_state: NewVideoGenerationState) -> OldVideoGenerationState:
        """
        Convert new Pydantic state format to old TypedDict state format.
        
        Args:
            new_state: New state in Pydantic format
            
        Returns:
            OldVideoGenerationState: Converted state in old TypedDict format
        """
        try:
            # Extract configuration values
            config = new_state.config
            
            # Create old state format
            old_state = OldVideoGenerationState(
                # Core workflow data
                messages=[],  # Old format expects messages list
                topic=new_state.topic,
                description=new_state.description,
                session_id=new_state.session_id,
                
                # Configuration parameters
                output_dir=config.output_dir,
                print_response=config.verbose,
                use_rag=config.use_rag,
                use_context_learning=config.use_context_learning,
                context_learning_path=config.context_learning_path,
                chroma_db_path=config.chroma_db_path,
                manim_docs_path=config.manim_docs_path,
                embedding_model=config.embedding_model,
                use_visual_fix_code=config.use_visual_analysis,
                use_langfuse=config.use_langfuse,
                max_scene_concurrency=config.max_concurrent_scenes,
                max_topic_concurrency=1,  # Default value
                max_retries=config.max_retries,
                
                # Enhanced RAG Configuration
                use_enhanced_rag=config.use_enhanced_rag,
                enable_rag_caching=True,  # Default value
                enable_quality_monitoring=True,  # Default value
                enable_error_handling=True,  # Default value
                rag_cache_ttl=3600,  # Default value
                rag_max_cache_size=1000,  # Default value
                rag_performance_threshold=2.0,  # Default value
                rag_quality_threshold=0.7,  # Default value
                
                # Renderer optimizations
                enable_caching=config.enable_caching,
                default_quality=config.default_quality,
                use_gpu_acceleration=config.use_gpu_acceleration,
                preview_mode=config.preview_mode,
                max_concurrent_renders=config.max_concurrent_renders,
                
                # Planning state
                scene_outline=new_state.scene_outline,
                scene_implementations=new_state.scene_implementations,
                detected_plugins=new_state.detected_plugins,
                
                # Code generation state
                generated_code=new_state.generated_code,
                code_errors=new_state.code_errors,
                rag_context=new_state.rag_context,
                
                # Rendering state
                rendered_videos=new_state.rendered_videos,
                combined_video_path=new_state.combined_video_path,
                rendering_errors=new_state.rendering_errors,
                
                # Visual analysis state
                visual_analysis_results=new_state.visual_analysis_results,
                visual_errors=new_state.visual_errors,
                
                # Error handling state
                error_count=len(new_state.errors),
                retry_count=new_state.retry_counts,
                escalated_errors=new_state.escalated_errors,
                
                # Human loop state
                pending_human_input=new_state.pending_human_input,
                human_feedback=new_state.human_feedback,
                
                # Monitoring state
                performance_metrics=new_state.metrics.model_dump() if new_state.metrics else {},
                execution_trace=new_state.execution_trace,
                
                # Current agent tracking
                current_agent=StateAdapter._map_step_to_agent(new_state.current_step),
                next_agent=StateAdapter._determine_next_agent(new_state),
                
                # Workflow control
                workflow_complete=new_state.workflow_complete,
                workflow_interrupted=new_state.workflow_interrupted
            )
            
            logger.info(f"Successfully converted new state to old format for session {new_state.session_id}")
            return old_state
            
        except Exception as e:
            logger.error(f"Failed to convert new state to old format: {e}")
            raise ValueError(f"State conversion failed: {e}") from e
    
    @staticmethod
    def _extract_model_config(state_dict: Dict[str, Any], key: str, default: str) -> Any:
        """Extract model configuration from old state format."""
        from ..models.config import ModelConfig
        
        # Try to get model name from state
        model_name = state_dict.get(key, default)
        
        # If it's already a ModelConfig object, return it
        if hasattr(model_name, 'provider'):
            return model_name
        
        # Parse model name to extract provider and model
        if isinstance(model_name, str) and '/' in model_name:
            provider, model = model_name.split('/', 1)
        else:
            provider = 'openrouter'
            model = str(model_name) if model_name else default.split('/', 1)[1]
        
        return ModelConfig(
            provider=provider,
            model_name=f"{provider}/{model}",
            temperature=0.7,
            max_tokens=4000,
            timeout=30
        )
    
    @staticmethod
    def _extract_step_durations(state_dict: Dict[str, Any]) -> Dict[str, float]:
        """Extract step durations from old performance metrics."""
        performance_metrics = state_dict.get('performance_metrics', {})
        step_durations = {}
        
        # Extract durations from agent-specific metrics
        for agent_name, metrics in performance_metrics.items():
            if isinstance(metrics, dict) and 'last_execution_time' in metrics:
                # Map agent names to step names
                step_name = StateAdapter._map_agent_to_step(agent_name)
                step_durations[step_name] = metrics['last_execution_time']
        
        return step_durations
    
    @staticmethod
    def _extract_success_rates(state_dict: Dict[str, Any]) -> Dict[str, float]:
        """Extract success rates from old performance metrics."""
        performance_metrics = state_dict.get('performance_metrics', {})
        success_rates = {}
        
        # Extract success rates from agent-specific metrics
        for agent_name, metrics in performance_metrics.items():
            if isinstance(metrics, dict) and 'success_rate' in metrics:
                # Map agent names to step names
                step_name = StateAdapter._map_agent_to_step(agent_name)
                success_rates[step_name] = metrics['success_rate']
        
        return success_rates
    
    @staticmethod
    def _determine_current_step(state_dict: Dict[str, Any]) -> str:
        """Determine current step from old state format."""
        current_agent = state_dict.get('current_agent')
        next_agent = state_dict.get('next_agent')
        
        # If workflow is complete
        if state_dict.get('workflow_complete', False):
            return 'complete'
        
        # If workflow is interrupted
        if state_dict.get('workflow_interrupted', False):
            return 'error_handling'
        
        # Map from agent names to step names
        if current_agent:
            return StateAdapter._map_agent_to_step(current_agent)
        elif next_agent:
            return StateAdapter._map_agent_to_step(next_agent)
        
        # Default to planning if no agent specified
        return 'planning'
    
    @staticmethod
    def _map_agent_to_step(agent_name: str) -> str:
        """Map old agent names to new step names."""
        agent_to_step_mapping = {
            'planner_agent': 'planning',
            'code_generator_agent': 'code_generation',
            'renderer_agent': 'rendering',
            'visual_analysis_agent': 'visual_analysis',
            'error_handler_agent': 'error_handling',
            'human_loop_agent': 'human_loop',
            'monitoring_agent': 'monitoring'
        }
        
        return agent_to_step_mapping.get(agent_name, 'planning')
    
    @staticmethod
    def _map_step_to_agent(step_name: str) -> str:
        """Map new step names to old agent names."""
        step_to_agent_mapping = {
            'planning': 'planner_agent',
            'code_generation': 'code_generator_agent',
            'rendering': 'renderer_agent',
            'visual_analysis': 'visual_analysis_agent',
            'error_handling': 'error_handler_agent',
            'human_loop': 'human_loop_agent',
            'monitoring': 'monitoring_agent',
            'complete': 'monitoring_agent'
        }
        
        return step_to_agent_mapping.get(step_name, 'planner_agent')
    
    @staticmethod
    def _determine_next_agent(new_state: NewVideoGenerationState) -> Optional[str]:
        """Determine next agent from new state format."""
        current_step = new_state.current_step
        
        # Simple workflow progression
        step_progression = {
            'planning': 'code_generator_agent',
            'code_generation': 'renderer_agent',
            'rendering': 'visual_analysis_agent' if new_state.config.use_visual_analysis else None,
            'visual_analysis': None,
            'error_handling': 'planner_agent',  # Retry from planning
            'human_loop': 'planner_agent',  # Resume from planning
            'complete': None
        }
        
        return step_progression.get(current_step)
    
    @staticmethod
    def _parse_timestamp(timestamp_str: Optional[str]) -> datetime:
        """Parse timestamp string to datetime object."""
        if not timestamp_str:
            return datetime.now()
        
        try:
            # Try ISO format first
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            # Fallback to current time if parsing fails
            return datetime.now()
    
    @staticmethod
    def validate_conversion(old_state: Dict[str, Any], new_state: NewVideoGenerationState) -> bool:
        """
        Validate that conversion preserved essential data.
        
        Args:
            old_state: Original old state
            new_state: Converted new state
            
        Returns:
            bool: True if conversion is valid
        """
        try:
            # Check core fields
            if old_state.get('topic') != new_state.topic:
                logger.error("Topic mismatch in conversion")
                return False
            
            if old_state.get('description') != new_state.description:
                logger.error("Description mismatch in conversion")
                return False
            
            if old_state.get('session_id') != new_state.session_id:
                logger.error("Session ID mismatch in conversion")
                return False
            
            # Check state preservation
            if old_state.get('scene_outline') != new_state.scene_outline:
                logger.error("Scene outline mismatch in conversion")
                return False
            
            if old_state.get('workflow_complete') != new_state.workflow_complete:
                logger.error("Workflow completion status mismatch in conversion")
                return False
            
            logger.info("State conversion validation passed")
            return True
            
        except Exception as e:
            logger.error(f"State conversion validation failed: {e}")
            return False
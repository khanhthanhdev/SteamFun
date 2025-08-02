"""
Parameter mapping and validation for backward compatibility.
Handles conversion between existing API parameters and LangGraph configuration.
"""

import logging
import os
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class JinaRAGConfig:
    """Configuration for Jina API and AstraDB RAG setup."""
    
    # Jina API configuration
    jina_api_key: Optional[str] = None
    jina_model: str = "jina-embeddings-v3"
    jina_dimensions: int = 1024
    jina_batch_size: int = 32
    jina_task: str = "retrieval.passage"
    
    # AstraDB configuration
    astradb_application_token: Optional[str] = None
    astradb_api_endpoint: Optional[str] = None
    astradb_collection_name: str = "manim_docs_embeddings"
    astradb_keyspace: Optional[str] = None
    astradb_dimension: int = 1024
    astradb_metric: str = "cosine"
    
    # Fallback to local if cloud services not available
    use_local_fallback: bool = True
    local_chroma_db_path: str = "data/rag/chroma_db"
    local_embedding_model: str = "hf:ibm-granite/granite-embedding-30m-english"
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate the configuration.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        # Check if cloud services are properly configured
        if not self.jina_api_key:
            errors.append("JINA_API_KEY environment variable is required for Jina embeddings")
        
        if not self.astradb_application_token:
            errors.append("ASTRADB_APPLICATION_TOKEN environment variable is required for AstraDB")
        
        if not self.astradb_api_endpoint:
            errors.append("ASTRADB_API_ENDPOINT environment variable is required for AstraDB")
        
        # If cloud services not configured but fallback enabled, that's okay
        if errors and not self.use_local_fallback:
            return False, errors
        
        # If fallback enabled, check local configuration
        if errors and self.use_local_fallback:
            if not Path(self.local_chroma_db_path).parent.exists():
                errors.append(f"Local ChromaDB path parent directory does not exist: {self.local_chroma_db_path}")
        
        # Validate dimensions match
        if self.jina_dimensions != self.astradb_dimension:
            errors.append(f"Jina dimensions ({self.jina_dimensions}) must match AstraDB dimensions ({self.astradb_dimension})")
        
        return len(errors) == 0, errors


@dataclass
class ParameterMappingConfig:
    """Configuration for parameter mapping between old and new systems."""
    
    # Model parameter mappings
    model_mappings: Dict[str, str] = field(default_factory=dict)
    
    # Feature flag mappings
    feature_mappings: Dict[str, str] = field(default_factory=dict)
    
    # Path mappings
    path_mappings: Dict[str, str] = field(default_factory=dict)
    
    # Default values for missing parameters
    default_values: Dict[str, Any] = field(default_factory=dict)
    
    # Deprecated parameters that should be ignored
    deprecated_params: List[str] = field(default_factory=list)
    
    # Required parameters that must be present
    required_params: List[str] = field(default_factory=list)


class BackwardCompatibleParameterMapper:
    """Maps parameters between existing API and LangGraph configuration."""
    
    def __init__(self, config: Optional[ParameterMappingConfig] = None):
        """Initialize parameter mapper with configuration.
        
        Args:
            config: Parameter mapping configuration
        """
        self.config = config or self._create_default_config()
        logger.info("BackwardCompatibleParameterMapper initialized")
    
    def _create_default_config(self) -> ParameterMappingConfig:
        """Create default parameter mapping configuration.
        
        Returns:
            ParameterMappingConfig: Default configuration
        """
        return ParameterMappingConfig(
            model_mappings={
                # Legacy model names to new format
                'gpt-4': 'openai/gpt-4',
                'gpt-4-turbo': 'openai/gpt-4-turbo',
                'gpt-3.5-turbo': 'openai/gpt-3.5-turbo',
                'claude-3-opus': 'anthropic/claude-3-opus-20240229',
                'claude-3-sonnet': 'anthropic/claude-3-sonnet-20240229',
                'claude-3-haiku': 'anthropic/claude-3-haiku-20240307',
                # AWS Bedrock models
                'bedrock/anthropic.claude-3-opus-20240229-v1:0': 'bedrock/anthropic.claude-3-opus-20240229-v1:0',
                'bedrock/anthropic.claude-3-sonnet-20240229-v1:0': 'bedrock/anthropic.claude-3-sonnet-20240229-v1:0',
                'bedrock/anthropic.claude-3-haiku-20240307-v1:0': 'bedrock/anthropic.claude-3-haiku-20240307-v1:0'
            },
            feature_mappings={
                'use_rag': 'enable_rag',
                'use_context_learning': 'enable_context_learning',
                'use_visual_fix_code': 'enable_visual_fix_code',
                'use_langfuse': 'enable_langfuse',
                'print_response': 'verbose',
                'use_enhanced_rag': 'enable_enhanced_rag'
            },
            path_mappings={
                'output_dir': 'output_directory',
                'context_learning_path': 'context_learning_directory',
                'chroma_db_path': 'vector_store_path',
                'manim_docs_path': 'documentation_path'
            },
            default_values={
                'output_dir': 'output',
                'print_response': False,
                'use_rag': True,
                'use_context_learning': True,
                'use_visual_fix_code': False,
                'use_langfuse': True,
                'max_retries': 3,
                'max_scene_concurrency': 5,
                'enable_caching': True,
                'default_quality': 'medium',
                'embedding_model': 'jina-embeddings-v3',  # Use Jina API by default
                'embedding_provider': 'jina',  # Specify Jina as provider
                'vector_store_provider': 'astradb',  # Use AstraDB instead of ChromaDB
                'context_learning_path': 'data/context_learning',
                'chroma_db_path': 'data/rag/chroma_db',  # Keep for backward compatibility
                'manim_docs_path': 'data/rag/manim_docs',
                # Jina API configuration
                'jina_api_key': os.getenv('JINA_API_KEY'),
                'jina_model': 'jina-embeddings-v3',
                'jina_dimensions': 1024,
                'jina_batch_size': 32,
                # AstraDB configuration
                'astradb_application_token': os.getenv('ASTRADB_APPLICATION_TOKEN'),
                'astradb_api_endpoint': os.getenv('ASTRADB_API_ENDPOINT'),
                'astradb_collection_name': 'manim_docs_embeddings',
                'astradb_keyspace': None,
                'astradb_dimension': 1024,
                'astradb_metric': 'cosine'
            },
            deprecated_params=[
                'rag_queries_cache',  # Now handled internally
                'banned_reasonings',  # Now handled by error handler agent
                'visual_self_reflection_func',  # Now handled by visual analysis agent
                'curr_version',  # Now handled internally
                'code_dir',  # Now handled internally
                'media_dir'  # Now handled internally
            ],
            required_params=[
                'scene_model',
                'helper_model'
            ]
        )
    
    def map_code_generator_params(self, **kwargs) -> Dict[str, Any]:
        """Map CodeGenerator initialization parameters to LangGraph format.
        
        Args:
            **kwargs: Original CodeGenerator parameters
            
        Returns:
            Dict: Mapped parameters for LangGraph system
        """
        mapped_params = {}
        
        # Map model parameters
        if 'scene_model' in kwargs:
            mapped_params['scene_model'] = self._map_model_name(kwargs['scene_model'])
        if 'helper_model' in kwargs:
            mapped_params['helper_model'] = self._map_model_name(kwargs['helper_model'])
        
        # Map feature flags
        for old_param, new_param in self.config.feature_mappings.items():
            if old_param in kwargs:
                mapped_params[new_param] = kwargs[old_param]
        
        # Map path parameters
        for old_param, new_param in self.config.path_mappings.items():
            if old_param in kwargs:
                mapped_params[new_param] = str(Path(kwargs[old_param]).resolve())
        
        # Add RAG configuration with JINA API and AstraDB
        mapped_params.update(self.map_rag_parameters(
            use_rag=kwargs.get('use_rag', False),
            chroma_db_path=kwargs.get('chroma_db_path', 'data/rag/chroma_db'),
            manim_docs_path=kwargs.get('manim_docs_path', 'data/rag/manim_docs'),
            embedding_model=kwargs.get('embedding_model', ''),
            use_context_learning=kwargs.get('use_context_learning', False),
            context_learning_path=kwargs.get('context_learning_path', 'data/context_learning')
        ))
        
        # Add default values for missing parameters
        for param, default_value in self.config.default_values.items():
            if param not in mapped_params and param not in kwargs:
                mapped_params[param] = default_value
        
        # Copy other parameters directly
        for key, value in kwargs.items():
            if (key not in self.config.feature_mappings and 
                key not in self.config.path_mappings and
                key not in self.config.deprecated_params and
                key not in mapped_params):
                mapped_params[key] = value
        
        # Validate required parameters
        self._validate_required_params(mapped_params, 'CodeGenerator')
        
        logger.debug(f"Mapped CodeGenerator parameters: {len(mapped_params)} parameters")
        return mapped_params
    
    def map_video_planner_params(self, **kwargs) -> Dict[str, Any]:
        """Map EnhancedVideoPlanner initialization parameters to LangGraph format.
        
        Args:
            **kwargs: Original EnhancedVideoPlanner parameters
            
        Returns:
            Dict: Mapped parameters for LangGraph system
        """
        mapped_params = {}
        
        # Map model parameters
        if 'planner_model' in kwargs:
            mapped_params['planner_model'] = self._map_model_name(kwargs['planner_model'])
        if 'helper_model' in kwargs:
            mapped_params['helper_model'] = self._map_model_name(kwargs['helper_model'])
        
        # Map feature flags
        for old_param, new_param in self.config.feature_mappings.items():
            if old_param in kwargs:
                mapped_params[new_param] = kwargs[old_param]
        
        # Map path parameters
        for old_param, new_param in self.config.path_mappings.items():
            if old_param in kwargs:
                mapped_params[new_param] = str(Path(kwargs[old_param]).resolve())
        
        # Map planner-specific parameters
        planner_specific_mappings = {
            'max_scene_concurrency': 'max_scene_concurrency',
            'max_step_concurrency': 'max_step_concurrency',
            'enable_caching': 'enable_caching',
            'rag_config': 'rag_config'
        }
        
        for old_param, new_param in planner_specific_mappings.items():
            if old_param in kwargs:
                mapped_params[new_param] = kwargs[old_param]
        
        # Add default values for missing parameters
        for param, default_value in self.config.default_values.items():
            if param not in mapped_params and param not in kwargs:
                mapped_params[param] = default_value
        
        # Add RAG configuration with JINA API and AstraDB
        mapped_params.update(self.map_rag_parameters(
            use_rag=kwargs.get('use_rag', False),
            chroma_db_path=kwargs.get('chroma_db_path', 'data/rag/chroma_db'),
            manim_docs_path=kwargs.get('manim_docs_path', 'data/rag/manim_docs'),
            embedding_model=kwargs.get('embedding_model', ''),
            use_context_learning=kwargs.get('use_context_learning', False),
            context_learning_path=kwargs.get('context_learning_path', 'data/context_learning'),
            enable_caching=kwargs.get('enable_caching', True),
            use_enhanced_rag=kwargs.get('use_enhanced_rag', True)
        ))
        
        # Copy other parameters directly
        for key, value in kwargs.items():
            if (key not in self.config.feature_mappings and 
                key not in self.config.path_mappings and
                key not in planner_specific_mappings and
                key not in self.config.deprecated_params and
                key not in mapped_params):
                mapped_params[key] = value
        
        logger.debug(f"Mapped VideoPlanner parameters: {len(mapped_params)} parameters")
        return mapped_params
    
    def map_video_renderer_params(self, **kwargs) -> Dict[str, Any]:
        """Map OptimizedVideoRenderer initialization parameters to LangGraph format.
        
        Args:
            **kwargs: Original OptimizedVideoRenderer parameters
            
        Returns:
            Dict: Mapped parameters for LangGraph system
        """
        mapped_params = {}
        
        # Map renderer-specific parameters
        renderer_specific_mappings = {
            'max_concurrent_renders': 'max_concurrent_renders',
            'enable_caching': 'enable_caching',
            'default_quality': 'default_quality',
            'use_gpu_acceleration': 'use_gpu_acceleration',
            'preview_mode': 'preview_mode'
        }
        
        for old_param, new_param in renderer_specific_mappings.items():
            if old_param in kwargs:
                mapped_params[new_param] = kwargs[old_param]
        
        # Map common parameters
        for old_param, new_param in self.config.feature_mappings.items():
            if old_param in kwargs:
                mapped_params[new_param] = kwargs[old_param]
        
        for old_param, new_param in self.config.path_mappings.items():
            if old_param in kwargs:
                mapped_params[new_param] = str(Path(kwargs[old_param]).resolve())
        
        # Add default values for missing parameters
        renderer_defaults = {
            'max_concurrent_renders': 4,
            'enable_caching': True,
            'default_quality': 'medium',
            'use_gpu_acceleration': False,
            'preview_mode': False
        }
        
        for param, default_value in {**self.config.default_values, **renderer_defaults}.items():
            if param not in mapped_params and param not in kwargs:
                mapped_params[param] = default_value
        
        # Add minimal RAG configuration (renderer may need RAG for visual analysis)
        mapped_params.update(self.map_rag_parameters(
            use_rag=kwargs.get('use_rag', False),
            use_enhanced_rag=kwargs.get('use_enhanced_rag', False)
        ))
        
        # Copy other parameters directly
        for key, value in kwargs.items():
            if (key not in self.config.feature_mappings and 
                key not in self.config.path_mappings and
                key not in renderer_specific_mappings and
                key not in self.config.deprecated_params and
                key not in mapped_params):
                mapped_params[key] = value
        
        logger.debug(f"Mapped VideoRenderer parameters: {len(mapped_params)} parameters")
        return mapped_params
    
    def map_method_params(self, method_name: str, **kwargs) -> Dict[str, Any]:
        """Map method parameters to LangGraph format.
        
        Args:
            method_name: Name of the method being called
            **kwargs: Method parameters
            
        Returns:
            Dict: Mapped parameters
        """
        mapped_params = {}
        
        # Method-specific parameter mappings
        method_mappings = {
            'generate_manim_code': {
                'scene_trace_id': 'trace_id',
                'rag_queries_cache': None,  # Deprecated
                'additional_context': 'context'
            },
            'fix_code_errors': {
                'implementation_plan': 'scene_implementation',
                'scene_trace_id': 'trace_id',
                'rag_queries_cache': None  # Deprecated
            },
            'visual_self_reflection': {
                'media_path': 'media_input',
                'scene_trace_id': 'trace_id'
            },
            'render_scene_optimized': {
                'curr_scene': 'scene_number',
                'curr_version': 'version',
                'scene_trace_id': 'trace_id',
                'visual_self_reflection_func': None,  # Deprecated
                'banned_reasonings': None  # Deprecated
            }
        }
        
        if method_name in method_mappings:
            for old_param, new_param in method_mappings[method_name].items():
                if old_param in kwargs:
                    if new_param is not None:
                        mapped_params[new_param] = kwargs[old_param]
                    # Skip deprecated parameters (new_param is None)
        
        # Copy other parameters directly
        for key, value in kwargs.items():
            if key not in mapped_params and key not in self.config.deprecated_params:
                mapped_params[key] = value
        
        return mapped_params
    
    def _map_model_name(self, model_obj: Any) -> str:
        """Map model object or name to standardized format.
        
        Args:
            model_obj: Model object or string name
            
        Returns:
            str: Standardized model name
        """
        # Extract model name from object if needed
        if hasattr(model_obj, 'model_name'):
            model_name = model_obj.model_name
        elif hasattr(model_obj, 'model'):
            model_name = model_obj.model
        elif isinstance(model_obj, str):
            model_name = model_obj
        else:
            logger.warning(f"Unknown model object type: {type(model_obj)}, using default")
            return "openai/gpt-4o"
        
        # Apply model name mapping
        mapped_name = self.config.model_mappings.get(model_name, model_name)
        
        # Ensure proper format (provider/model)
        if '/' not in mapped_name and not mapped_name.startswith('bedrock/'):
            # Default to OpenAI if no provider specified
            mapped_name = f"openai/{mapped_name}"
        
        return mapped_name
    
    def _validate_required_params(self, params: Dict[str, Any], component_name: str):
        """Validate that required parameters are present.
        
        Args:
            params: Parameters to validate
            component_name: Name of component for error messages
            
        Raises:
            ValueError: If required parameters are missing
        """
        missing_params = []
        for required_param in self.config.required_params:
            if required_param not in params:
                missing_params.append(required_param)
        
        if missing_params:
            raise ValueError(
                f"Missing required parameters for {component_name}: {missing_params}"
            )
    
    def validate_configuration(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration parameters.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        # Check for deprecated parameters
        for param in config:
            if param in self.config.deprecated_params:
                errors.append(f"Parameter '{param}' is deprecated and will be ignored")
        
        # Validate paths exist
        path_params = ['output_dir', 'context_learning_path', 'chroma_db_path', 'manim_docs_path']
        for param in path_params:
            if param in config:
                path = Path(config[param])
                if param == 'output_dir':
                    # Output directory can be created
                    try:
                        path.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        errors.append(f"Cannot create output directory '{path}': {e}")
                else:
                    # Other paths should exist
                    if not path.exists():
                        errors.append(f"Path '{path}' does not exist for parameter '{param}'")
        
        # Validate model names
        model_params = ['planner_model', 'scene_model', 'helper_model']
        for param in model_params:
            if param in config:
                model_name = config[param]
                if isinstance(model_name, str) and not self._is_valid_model_name(model_name):
                    errors.append(f"Invalid model name format: '{model_name}'")
        
        # Validate numeric parameters
        numeric_params = {
            'max_retries': (1, 10),
            'max_scene_concurrency': (1, 20),
            'max_concurrent_renders': (1, 10)
        }
        
        for param, (min_val, max_val) in numeric_params.items():
            if param in config:
                value = config[param]
                if not isinstance(value, int) or value < min_val or value > max_val:
                    errors.append(f"Parameter '{param}' must be an integer between {min_val} and {max_val}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def map_rag_parameters(self, **kwargs) -> Dict[str, Any]:
        """Map RAG parameters to use Jina API and AstraDB with fallback to local.
        
        Args:
            **kwargs: RAG-related parameters
            
        Returns:
            Dict: Mapped RAG parameters with priority for JINA API and AstraDB
        """
        mapped_params = {}
        
        # Priority order: JINA API + AstraDB -> Local embeddings + AstraDB -> Local fallback
        use_rag = kwargs.get('use_rag', True)
        
        if use_rag:
            # Check for JINA API and AstraDB availability (preferred combination)
            jina_api_key = os.getenv('JINA_API_KEY')
            astra_db_token = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
            astra_db_endpoint = os.getenv('ASTRA_DB_API_ENDPOINT')
            astra_db_keyspace = os.getenv('ASTRA_DB_KEYSPACE', 'default_keyspace')
            
            if jina_api_key and astra_db_token and astra_db_endpoint:
                # Best case: JINA API + AstraDB
                mapped_params.update({
                    'embedding_provider': 'jina',
                    'embedding_model': 'jina-embeddings-v2-base-en',
                    'use_enhanced_rag': True,
                    'vector_store_provider': 'astradb',
                    'astra_db_config': {
                        'token': astra_db_token,
                        'api_endpoint': astra_db_endpoint,
                        'keyspace': astra_db_keyspace,
                        'collection_name': 'manim_docs_jina',
                        'embedding_dimension': 768  # JINA v2 base dimension
                    },
                    'rag_priority': 'jina_astradb'
                })
                logger.info("Using JINA API for embeddings with AstraDB vector store (optimal configuration)")
                
            elif astra_db_token and astra_db_endpoint:
                # Second choice: Local embeddings + AstraDB
                embedding_model = kwargs.get('embedding_model', 'hf:ibm-granite/granite-embedding-30m-english')
                # Determine embedding dimension based on model
                embedding_dim = 384 if 'granite-embedding-30m' in embedding_model else 768
                
                mapped_params.update({
                    'embedding_provider': 'local',
                    'embedding_model': embedding_model,
                    'use_enhanced_rag': True,
                    'vector_store_provider': 'astradb',
                    'astra_db_config': {
                        'token': astra_db_token,
                        'api_endpoint': astra_db_endpoint,
                        'keyspace': astra_db_keyspace,
                        'collection_name': 'manim_docs_local',
                        'embedding_dimension': embedding_dim
                    },
                    'rag_priority': 'local_astradb'
                })
                logger.info("Using AstraDB vector store with local embeddings")
                
            elif jina_api_key:
                # Third choice: JINA API + local vector store (Chroma)
                mapped_params.update({
                    'embedding_provider': 'jina',
                    'embedding_model': 'jina-embeddings-v2-base-en',
                    'use_enhanced_rag': True,
                    'vector_store_provider': 'chroma',
                    'chroma_db_path': kwargs.get('chroma_db_path', 'data/rag/chroma_db'),
                    'rag_priority': 'jina_chroma'
                })
                logger.info("Using JINA API for embeddings with Chroma vector store")
                
            else:
                # Fallback: Local embeddings + local vector store
                mapped_params.update({
                    'embedding_provider': 'local',
                    'embedding_model': kwargs.get('embedding_model', 'hf:ibm-granite/granite-embedding-30m-english'),
                    'use_enhanced_rag': False,
                    'vector_store_provider': 'chroma',
                    'chroma_db_path': kwargs.get('chroma_db_path', 'data/rag/chroma_db'),
                    'rag_priority': 'local_fallback'
                })
                logger.warning("Using local RAG system - consider setting JINA_API_KEY and AstraDB credentials for better performance")
        else:
            mapped_params.update({
                'use_rag': False,
                'rag_priority': 'disabled'
            })
        
        # Common RAG parameters
        mapped_params.update({
            'manim_docs_path': kwargs.get('manim_docs_path', 'data/rag/manim_docs'),
            'use_context_learning': kwargs.get('use_context_learning', True),
            'context_learning_path': kwargs.get('context_learning_path', 'data/context_learning'),
            'enable_caching': kwargs.get('enable_caching', True),
            'enable_quality_monitoring': kwargs.get('enable_quality_monitoring', True),
            'enable_error_handling': kwargs.get('enable_error_handling', True)
        })
        
        # Add environment validation info
        mapped_params['env_status'] = {
            'jina_api_available': bool(os.getenv('JINA_API_KEY')),
            'astradb_available': bool(os.getenv('ASTRA_DB_APPLICATION_TOKEN') and os.getenv('ASTRA_DB_API_ENDPOINT')),
            'recommended_setup': 'Set JINA_API_KEY, ASTRA_DB_APPLICATION_TOKEN, and ASTRA_DB_API_ENDPOINT for optimal performance'
        }
        
        return mapped_params
    
    def map_rag_parameters(self, 
                          use_rag: bool = None,
                          **legacy_params) -> Dict[str, Any]:
        """Map RAG parameters using centralized .env configuration.
        
        Args:
            use_rag: Whether to enable RAG (overrides .env if specified)
            **legacy_params: Legacy parameters (ignored, all config from .env)
            
        Returns:
            Dict: Complete RAG configuration from .env file
        """
        from .rag_config import get_rag_config_manager
        
        # Get centralized RAG configuration
        rag_config_manager = get_rag_config_manager()
        config = rag_config_manager.config
        
        # Override RAG enabled status if explicitly provided
        rag_enabled = use_rag if use_rag is not None else config.enabled
        
        if not rag_enabled:
            return {'use_rag': False}
        
        # Build complete RAG parameters from .env configuration
        rag_params = {
            # Core RAG settings
            'use_rag': True,
            'use_enhanced_rag': config.use_enhanced_components,
            'enable_caching': config.enable_caching,
            'enable_quality_monitoring': config.enable_quality_monitoring,
            'enable_error_handling': config.enable_error_handling,
            
            # Performance settings
            'cache_ttl': config.cache_ttl,
            'max_cache_size': config.max_cache_size,
            'performance_threshold': config.performance_threshold,
            'quality_threshold': config.quality_threshold,
            'default_k_value': config.default_k_value,
            'max_retries': config.max_retries,
            
            # Embedding configuration
            'embedding_config': rag_config_manager.get_embedding_config(),
            'embedding_provider': config.embedding_provider,
            'embedding_model': self._get_embedding_model_string(config),
            'embedding_dimension': config.embedding_dimension,
            'embedding_batch_size': config.embedding_batch_size,
            'embedding_timeout': config.embedding_timeout,
            
            # Vector store configuration
            'vector_store_config': rag_config_manager.get_vector_store_config(),
            'vector_store_provider': config.vector_store_provider,
            'vector_store_collection': config.vector_store_collection,
            'vector_store_distance_metric': config.vector_store_distance_metric,
            'vector_store_max_results': config.vector_store_max_results,
            
            # Document processing configuration
            'document_config': rag_config_manager.get_document_processing_config(),
            'manim_docs_path': config.manim_docs_path,
            'context_learning_path': config.context_learning_path,
            'use_context_learning': config.use_context_learning,
            'chunk_size': config.chunk_size,
            'chunk_overlap': config.chunk_overlap,
            'min_chunk_size': config.min_chunk_size,
            
            # Query processing configuration
            'query_config': rag_config_manager.get_query_processing_config(),
            'enable_query_expansion': config.enable_query_expansion,
            'enable_semantic_search': config.enable_semantic_search,
            'similarity_threshold': config.similarity_threshold,
            'context_window_size': config.context_window_size,
            
            # Plugin detection configuration
            'plugin_config': rag_config_manager.get_plugin_detection_config(),
            'enable_plugin_detection': config.enable_plugin_detection,
            'plugin_confidence_threshold': config.plugin_confidence_threshold,
            'max_plugins_per_query': config.max_plugins_per_query,
            
            # Monitoring configuration
            'monitoring_config': rag_config_manager.get_monitoring_config(),
            'enable_performance_monitoring': config.enable_performance_monitoring,
            'enable_usage_tracking': config.enable_usage_tracking,
            'enable_relevance_scoring': config.enable_relevance_scoring,
            
            # Legacy compatibility (for fallback)
            'chroma_db_path': config.chroma_db_path,
            
            # Configuration metadata
            'rag_priority': self._determine_rag_priority(config),
            'config_source': 'env_file',
            'config_valid': len(rag_config_manager.validate_configuration()) == 0
        }
        
        # Validate configuration and adjust if needed
        validation_issues = rag_config_manager.validate_configuration()
        if validation_issues:
            logger.warning(f"RAG configuration issues detected: {validation_issues}")
            rag_params['config_issues'] = validation_issues
            rag_params = self._apply_fallback_config(rag_params, config)
        
        logger.info(f"RAG parameters mapped from .env with priority: {rag_params['rag_priority']}")
        return rag_params
    
    def _get_embedding_model_string(self, config) -> str:
        """Get embedding model string based on provider."""
        if config.embedding_provider == 'jina':
            return f"jina:{config.jina_model}"
        elif config.embedding_provider == 'openai':
            return f"openai:{config.openai_model}"
        else:
            return config.local_model
    
    def _determine_rag_priority(self, config) -> str:
        """Determine RAG system priority based on configuration."""
        if config.embedding_provider == 'jina' and config.vector_store_provider == 'astradb':
            if config.jina_api_key and config.astradb_api_endpoint and config.astradb_application_token:
                return 'jina_astradb_optimal'
            else:
                return 'jina_astradb_incomplete'
        elif config.embedding_provider == 'jina':
            return 'jina_chroma'
        elif config.vector_store_provider == 'astradb':
            return 'local_astradb'
        else:
            return 'local_chroma_fallback'
    
    def _apply_fallback_config(self, rag_params: Dict[str, Any], config) -> Dict[str, Any]:
        """Apply fallback configuration when validation fails."""
        # If JINA API key is missing, fall back to local embeddings
        if config.embedding_provider == 'jina' and not config.jina_api_key:
            logger.info("Falling back to local embeddings due to missing JINA API key")
            rag_params['embedding_provider'] = 'local'
            rag_params['embedding_model'] = config.local_model
            rag_params['embedding_config'] = {
                'provider': 'local',
                'model': config.local_model,
                'device': config.local_device,
                'cache_dir': config.local_cache_dir,
                'dimension': config.embedding_dimension,
                'batch_size': config.embedding_batch_size
            }
        
        # If AstraDB config is incomplete, fall back to ChromaDB
        if (config.vector_store_provider == 'astradb' and 
            (not config.astradb_api_endpoint or not config.astradb_application_token)):
            logger.info("Falling back to ChromaDB due to incomplete AstraDB configuration")
            rag_params['vector_store_provider'] = 'chroma'
            rag_params['vector_store_config'] = {
                'provider': 'chroma',
                'db_path': config.chroma_db_path,
                'collection_name': config.chroma_collection_name,
                'persist_directory': config.chroma_persist_directory,
                'dimension': config.embedding_dimension,
                'distance_metric': config.vector_store_distance_metric,
                'max_results': config.vector_store_max_results
            }
        
        # Update priority based on fallbacks
        rag_params['rag_priority'] = self._determine_rag_priority(config)
        rag_params['fallback_applied'] = True
        
        return rag_params
    
    def _is_valid_model_name(self, model_name: str) -> bool:
        """Check if model name is in valid format.
        
        Args:
            model_name: Model name to validate
            
        Returns:
            bool: True if valid format
        """
        # Valid formats: provider/model, bedrock/model, or known legacy names
        if '/' in model_name:
            return True
        if model_name.startswith('bedrock/'):
            return True
        if model_name in self.config.model_mappings:
            return True
        
        # Check for common model names without provider
        common_models = ['gpt-4', 'gpt-3.5-turbo', 'claude-3-opus', 'claude-3-sonnet']
        return model_name in common_models
    
    def create_jina_rag_config(self, **kwargs) -> JinaRAGConfig:
        """Create Jina RAG configuration from parameters.
        
        Args:
            **kwargs: Configuration parameters
            
        Returns:
            JinaRAGConfig: Jina RAG configuration
        """
        return JinaRAGConfig(
            # Jina API configuration
            jina_api_key=kwargs.get('jina_api_key') or os.getenv('JINA_API_KEY'),
            jina_model=kwargs.get('jina_model', 'jina-embeddings-v3'),
            jina_dimensions=kwargs.get('jina_dimensions', 1024),
            jina_batch_size=kwargs.get('jina_batch_size', 32),
            jina_task=kwargs.get('jina_task', 'retrieval.passage'),
            
            # AstraDB configuration
            astradb_application_token=kwargs.get('astradb_application_token') or os.getenv('ASTRADB_APPLICATION_TOKEN'),
            astradb_api_endpoint=kwargs.get('astradb_api_endpoint') or os.getenv('ASTRADB_API_ENDPOINT'),
            astradb_collection_name=kwargs.get('astradb_collection_name', 'manim_docs_embeddings'),
            astradb_keyspace=kwargs.get('astradb_keyspace'),
            astradb_dimension=kwargs.get('astradb_dimension', 1024),
            astradb_metric=kwargs.get('astradb_metric', 'cosine'),
            
            # Fallback configuration
            use_local_fallback=kwargs.get('use_local_fallback', True),
            local_chroma_db_path=kwargs.get('chroma_db_path', 'data/rag/chroma_db'),
            local_embedding_model=kwargs.get('embedding_model', 'hf:ibm-granite/granite-embedding-30m-english')
        )
    
    def map_rag_parameters(self, **kwargs) -> Dict[str, Any]:
        """Map RAG parameters to use Jina API and AstraDB.
        
        Args:
            **kwargs: Original RAG parameters
            
        Returns:
            Dict: Mapped RAG parameters
        """
        mapped_params = {}
        
        # Create Jina RAG configuration
        jina_rag_config = self.create_jina_rag_config(**kwargs)
        
        # Validate configuration
        is_valid, errors = jina_rag_config.validate()
        
        if is_valid:
            logger.info("Using Jina API and AstraDB for RAG")
            mapped_params.update({
                'embedding_provider': 'jina',
                'vector_store_provider': 'astradb',
                'jina_config': jina_rag_config,
                'use_cloud_rag': True
            })
        else:
            if jina_rag_config.use_local_fallback:
                logger.warning(f"Jina/AstraDB configuration issues: {errors}")
                logger.info("Falling back to local ChromaDB and local embeddings")
                mapped_params.update({
                    'embedding_provider': 'local',
                    'vector_store_provider': 'chromadb',
                    'chroma_db_path': jina_rag_config.local_chroma_db_path,
                    'embedding_model': jina_rag_config.local_embedding_model,
                    'use_cloud_rag': False
                })
            else:
                logger.error(f"RAG configuration errors: {errors}")
                raise ValueError(f"RAG configuration invalid: {errors}")
        
        # Copy other RAG-related parameters
        rag_params = [
            'use_rag', 'manim_docs_path', 'use_enhanced_rag', 
            'enable_rag_caching', 'enable_quality_monitoring'
        ]
        
        for param in rag_params:
            if param in kwargs:
                mapped_params[param] = kwargs[param]
        
        return mapped_params


# Global parameter mapper instance
parameter_mapper = BackwardCompatibleParameterMapper()


def get_parameter_mapper() -> BackwardCompatibleParameterMapper:
    """Get the global parameter mapper instance.
    
    Returns:
        BackwardCompatibleParameterMapper: Parameter mapper instance
    """
    return parameter_mapper
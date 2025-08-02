"""
Parameter conversion utilities for mapping between old and new configuration formats.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field

from ..compatibility.parameter_mapping import BackwardCompatibleParameterMapper


logger = logging.getLogger(__name__)


@dataclass
class ConversionRule:
    """Defines a parameter conversion rule."""
    source_key: str
    target_key: str
    converter: Optional[callable] = None
    default_value: Any = None
    required: bool = False
    deprecated: bool = False
    description: str = ""


@dataclass
class ConversionResult:
    """Result of parameter conversion."""
    success: bool
    converted_params: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    deprecated_params: List[str] = field(default_factory=list)


class ParameterConverter:
    """Converts parameters between different configuration formats."""
    
    def __init__(self):
        """Initialize parameter converter with conversion rules."""
        self.conversion_rules = self._create_conversion_rules()
        self.parameter_mapper = BackwardCompatibleParameterMapper()
        logger.info("ParameterConverter initialized")
    
    def _create_conversion_rules(self) -> Dict[str, List[ConversionRule]]:
        """Create parameter conversion rules for different components.
        
        Returns:
            Dict[str, List[ConversionRule]]: Conversion rules by component type
        """
        return {
            'code_generator': [
                ConversionRule(
                    source_key='scene_model',
                    target_key='scene_model',
                    converter=self._convert_model_name,
                    required=True,
                    description='Primary model for code generation'
                ),
                ConversionRule(
                    source_key='helper_model',
                    target_key='helper_model',
                    converter=self._convert_model_name,
                    required=True,
                    description='Helper model for auxiliary tasks'
                ),
                ConversionRule(
                    source_key='temperature',
                    target_key='temperature',
                    converter=float,
                    default_value=0.7,
                    description='Model temperature setting'
                ),
                ConversionRule(
                    source_key='max_retries',
                    target_key='max_retries',
                    converter=int,
                    default_value=3,
                    description='Maximum retry attempts'
                ),
                ConversionRule(
                    source_key='use_rag',
                    target_key='enable_rag',
                    converter=self._convert_boolean,
                    default_value=True,
                    description='Enable RAG functionality'
                ),
                ConversionRule(
                    source_key='use_context_learning',
                    target_key='enable_context_learning',
                    converter=self._convert_boolean,
                    default_value=True,
                    description='Enable context learning'
                ),
                ConversionRule(
                    source_key='use_visual_fix_code',
                    target_key='enable_visual_fix_code',
                    converter=self._convert_boolean,
                    default_value=False,
                    description='Enable visual code fixing'
                ),
                ConversionRule(
                    source_key='print_response',
                    target_key='verbose',
                    converter=self._convert_boolean,
                    default_value=False,
                    description='Enable verbose output'
                ),
                ConversionRule(
                    source_key='output_dir',
                    target_key='output_directory',
                    converter=self._convert_path,
                    default_value='output',
                    description='Output directory path'
                ),
                ConversionRule(
                    source_key='chroma_db_path',
                    target_key='vector_store_path',
                    converter=self._convert_path,
                    default_value='data/rag/chroma_db',
                    description='Vector store database path'
                ),
                ConversionRule(
                    source_key='manim_docs_path',
                    target_key='documentation_path',
                    converter=self._convert_path,
                    default_value='data/rag/manim_docs',
                    description='Documentation directory path'
                ),
                ConversionRule(
                    source_key='context_learning_path',
                    target_key='context_learning_directory',
                    converter=self._convert_path,
                    default_value='data/context_learning',
                    description='Context learning directory path'
                ),
                ConversionRule(
                    source_key='embedding_model',
                    target_key='embedding_model',
                    converter=str,
                    default_value='jina-embeddings-v3',
                    description='Embedding model name'
                ),
                # Deprecated parameters
                ConversionRule(
                    source_key='rag_queries_cache',
                    target_key=None,
                    deprecated=True,
                    description='RAG queries cache (now handled internally)'
                ),
                ConversionRule(
                    source_key='banned_reasonings',
                    target_key=None,
                    deprecated=True,
                    description='Banned reasonings (now handled by error handler agent)'
                ),
                ConversionRule(
                    source_key='visual_self_reflection_func',
                    target_key=None,
                    deprecated=True,
                    description='Visual self-reflection function (now handled by visual analysis agent)'
                ),
                ConversionRule(
                    source_key='curr_version',
                    target_key=None,
                    deprecated=True,
                    description='Current version (now handled internally)'
                ),
                ConversionRule(
                    source_key='code_dir',
                    target_key=None,
                    deprecated=True,
                    description='Code directory (now handled internally)'
                ),
                ConversionRule(
                    source_key='media_dir',
                    target_key=None,
                    deprecated=True,
                    description='Media directory (now handled internally)'
                )
            ],
            'video_planner': [
                ConversionRule(
                    source_key='planner_model',
                    target_key='planner_model',
                    converter=self._convert_model_name,
                    required=True,
                    description='Primary model for planning'
                ),
                ConversionRule(
                    source_key='helper_model',
                    target_key='helper_model',
                    converter=self._convert_model_name,
                    required=True,
                    description='Helper model for auxiliary tasks'
                ),
                ConversionRule(
                    source_key='temperature',
                    target_key='temperature',
                    converter=float,
                    default_value=0.7,
                    description='Model temperature setting'
                ),
                ConversionRule(
                    source_key='max_scene_concurrency',
                    target_key='max_scene_concurrency',
                    converter=int,
                    default_value=5,
                    description='Maximum concurrent scenes'
                ),
                ConversionRule(
                    source_key='max_step_concurrency',
                    target_key='max_step_concurrency',
                    converter=int,
                    default_value=3,
                    description='Maximum concurrent steps'
                ),
                ConversionRule(
                    source_key='enable_caching',
                    target_key='enable_caching',
                    converter=self._convert_boolean,
                    default_value=True,
                    description='Enable result caching'
                ),
                ConversionRule(
                    source_key='use_rag',
                    target_key='enable_rag',
                    converter=self._convert_boolean,
                    default_value=True,
                    description='Enable RAG functionality'
                ),
                ConversionRule(
                    source_key='use_enhanced_rag',
                    target_key='enable_enhanced_rag',
                    converter=self._convert_boolean,
                    default_value=True,
                    description='Enable enhanced RAG features'
                )
            ],
            'video_renderer': [
                ConversionRule(
                    source_key='max_concurrent_renders',
                    target_key='max_concurrent_renders',
                    converter=int,
                    default_value=4,
                    description='Maximum concurrent renders'
                ),
                ConversionRule(
                    source_key='enable_caching',
                    target_key='enable_caching',
                    converter=self._convert_boolean,
                    default_value=True,
                    description='Enable render caching'
                ),
                ConversionRule(
                    source_key='default_quality',
                    target_key='default_quality',
                    converter=str,
                    default_value='medium',
                    description='Default render quality'
                ),
                ConversionRule(
                    source_key='use_gpu_acceleration',
                    target_key='use_gpu_acceleration',
                    converter=self._convert_boolean,
                    default_value=False,
                    description='Enable GPU acceleration'
                ),
                ConversionRule(
                    source_key='preview_mode',
                    target_key='preview_mode',
                    converter=self._convert_boolean,
                    default_value=False,
                    description='Enable preview mode'
                )
            ],
            'environment': [
                # LLM Provider settings
                ConversionRule(
                    source_key='OPENAI_API_KEY',
                    target_key='openai_api_key',
                    converter=str,
                    description='OpenAI API key'
                ),
                ConversionRule(
                    source_key='AWS_BEDROCK_REGION',
                    target_key='aws_bedrock_region',
                    converter=str,
                    default_value='us-east-1',
                    description='AWS Bedrock region'
                ),
                ConversionRule(
                    source_key='AWS_BEDROCK_MODEL',
                    target_key='aws_bedrock_model',
                    converter=str,
                    description='AWS Bedrock model'
                ),
                ConversionRule(
                    source_key='OPENROUTER_API_KEY',
                    target_key='openrouter_api_key',
                    converter=str,
                    description='OpenRouter API key'
                ),
                
                # RAG settings
                ConversionRule(
                    source_key='RAG_ENABLED',
                    target_key='rag_enabled',
                    converter=self._convert_boolean,
                    default_value=True,
                    description='Enable RAG system'
                ),
                ConversionRule(
                    source_key='EMBEDDING_PROVIDER',
                    target_key='embedding_provider',
                    converter=str,
                    default_value='jina',
                    description='Embedding provider'
                ),
                ConversionRule(
                    source_key='VECTOR_STORE_PROVIDER',
                    target_key='vector_store_provider',
                    converter=str,
                    default_value='astradb',
                    description='Vector store provider'
                ),
                ConversionRule(
                    source_key='JINA_API_KEY',
                    target_key='jina_api_key',
                    converter=str,
                    description='JINA API key'
                ),
                ConversionRule(
                    source_key='JINA_EMBEDDING_MODEL',
                    target_key='jina_embedding_model',
                    converter=str,
                    default_value='jina-embeddings-v3',
                    description='JINA embedding model'
                ),
                ConversionRule(
                    source_key='ASTRADB_API_ENDPOINT',
                    target_key='astradb_api_endpoint',
                    converter=str,
                    description='AstraDB API endpoint'
                ),
                ConversionRule(
                    source_key='ASTRADB_APPLICATION_TOKEN',
                    target_key='astradb_application_token',
                    converter=str,
                    description='AstraDB application token'
                ),
                ConversionRule(
                    source_key='ASTRADB_KEYSPACE',
                    target_key='astradb_keyspace',
                    converter=str,
                    default_value='default_keyspace',
                    description='AstraDB keyspace'
                ),
                
                # Monitoring settings
                ConversionRule(
                    source_key='LANGFUSE_SECRET_KEY',
                    target_key='langfuse_secret_key',
                    converter=str,
                    description='LangFuse secret key'
                ),
                ConversionRule(
                    source_key='LANGFUSE_PUBLIC_KEY',
                    target_key='langfuse_public_key',
                    converter=str,
                    description='LangFuse public key'
                ),
                ConversionRule(
                    source_key='LANGFUSE_HOST',
                    target_key='langfuse_host',
                    converter=str,
                    default_value='https://cloud.langfuse.com',
                    description='LangFuse host URL'
                ),
                
                # Workflow settings
                ConversionRule(
                    source_key='MAX_WORKFLOW_RETRIES',
                    target_key='max_workflow_retries',
                    converter=int,
                    default_value=3,
                    description='Maximum workflow retries'
                ),
                ConversionRule(
                    source_key='WORKFLOW_TIMEOUT_SECONDS',
                    target_key='workflow_timeout_seconds',
                    converter=int,
                    default_value=3600,
                    description='Workflow timeout in seconds'
                ),
                ConversionRule(
                    source_key='HUMAN_LOOP_ENABLED',
                    target_key='human_loop_enabled',
                    converter=self._convert_boolean,
                    default_value=False,
                    description='Enable human-in-the-loop'
                ),
                ConversionRule(
                    source_key='HUMAN_LOOP_TIMEOUT',
                    target_key='human_loop_timeout',
                    converter=int,
                    default_value=300,
                    description='Human loop timeout in seconds'
                )
            ]
        }
    
    def convert_parameters(self, 
                          component_type: str, 
                          source_params: Dict[str, Any]) -> ConversionResult:
        """Convert parameters for a specific component type.
        
        Args:
            component_type: Type of component (code_generator, video_planner, video_renderer, environment)
            source_params: Source parameters to convert
            
        Returns:
            ConversionResult: Conversion result with converted parameters and messages
        """
        result = ConversionResult(success=True)
        
        if component_type not in self.conversion_rules:
            result.success = False
            result.errors.append(f"Unknown component type: {component_type}")
            return result
        
        rules = self.conversion_rules[component_type]
        
        # Process each conversion rule
        for rule in rules:
            try:
                self._apply_conversion_rule(rule, source_params, result)
            except Exception as e:
                result.errors.append(f"Error applying rule for {rule.source_key}: {str(e)}")
                result.success = False
        
        # Check for required parameters
        for rule in rules:
            if rule.required and rule.target_key and rule.target_key not in result.converted_params:
                result.errors.append(f"Required parameter missing: {rule.source_key}")
                result.success = False
        
        # Add any unmapped parameters with warnings
        mapped_source_keys = {rule.source_key for rule in rules}
        for key, value in source_params.items():
            if key not in mapped_source_keys:
                result.converted_params[key] = value
                result.warnings.append(f"Unmapped parameter passed through: {key}")
        
        logger.info(f"Parameter conversion for {component_type}: "
                   f"{'Success' if result.success else 'Failed'} "
                   f"({len(result.converted_params)} params, "
                   f"{len(result.warnings)} warnings, "
                   f"{len(result.errors)} errors)")
        
        return result
    
    def _apply_conversion_rule(self, 
                              rule: ConversionRule, 
                              source_params: Dict[str, Any], 
                              result: ConversionResult):
        """Apply a single conversion rule.
        
        Args:
            rule: Conversion rule to apply
            source_params: Source parameters
            result: Conversion result to update
        """
        # Handle deprecated parameters
        if rule.deprecated:
            if rule.source_key in source_params:
                result.deprecated_params.append(rule.source_key)
                result.warnings.append(f"Deprecated parameter ignored: {rule.source_key} - {rule.description}")
            return
        
        # Skip if no target key (deprecated parameter)
        if not rule.target_key:
            return
        
        # Get source value
        if rule.source_key in source_params:
            source_value = source_params[rule.source_key]
            
            # Apply converter if specified
            if rule.converter:
                try:
                    converted_value = rule.converter(source_value)
                except Exception as e:
                    result.warnings.append(f"Conversion failed for {rule.source_key}: {str(e)}, using default")
                    converted_value = rule.default_value
            else:
                converted_value = source_value
            
            result.converted_params[rule.target_key] = converted_value
            
        elif rule.default_value is not None:
            # Use default value if source parameter not present
            result.converted_params[rule.target_key] = rule.default_value
            result.warnings.append(f"Using default value for {rule.target_key}: {rule.default_value}")
    
    def _convert_model_name(self, model_obj: Any) -> str:
        """Convert model object or name to standardized format.
        
        Args:
            model_obj: Model object or string name
            
        Returns:
            str: Standardized model name
        """
        return self.parameter_mapper._map_model_name(model_obj)
    
    def _convert_boolean(self, value: Any) -> bool:
        """Convert value to boolean.
        
        Args:
            value: Value to convert
            
        Returns:
            bool: Converted boolean value
        """
        if isinstance(value, bool):
            return value
        elif isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
        elif isinstance(value, (int, float)):
            return bool(value)
        else:
            return False
    
    def _convert_path(self, value: Any) -> str:
        """Convert value to path string.
        
        Args:
            value: Value to convert
            
        Returns:
            str: Path string
        """
        return str(Path(value).resolve())
    
    def batch_convert_parameters(self, 
                                conversions: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, ConversionResult]:
        """Convert parameters for multiple components in batch.
        
        Args:
            conversions: List of (component_type, source_params) tuples
            
        Returns:
            Dict[str, ConversionResult]: Conversion results by component type
        """
        results = {}
        
        for component_type, source_params in conversions:
            results[component_type] = self.convert_parameters(component_type, source_params)
        
        # Log batch summary
        total_success = sum(1 for result in results.values() if result.success)
        total_warnings = sum(len(result.warnings) for result in results.values())
        total_errors = sum(len(result.errors) for result in results.values())
        
        logger.info(f"Batch parameter conversion completed: "
                   f"{total_success}/{len(conversions)} successful, "
                   f"{total_warnings} warnings, {total_errors} errors")
        
        return results
    
    def get_conversion_summary(self, component_type: str) -> Dict[str, Any]:
        """Get summary of available conversions for a component type.
        
        Args:
            component_type: Component type to get summary for
            
        Returns:
            Dict[str, Any]: Conversion summary
        """
        if component_type not in self.conversion_rules:
            return {'error': f'Unknown component type: {component_type}'}
        
        rules = self.conversion_rules[component_type]
        
        summary = {
            'component_type': component_type,
            'total_rules': len(rules),
            'required_parameters': [],
            'optional_parameters': [],
            'deprecated_parameters': [],
            'parameter_mappings': {}
        }
        
        for rule in rules:
            mapping_info = {
                'target_key': rule.target_key,
                'converter': rule.converter.__name__ if rule.converter else None,
                'default_value': rule.default_value,
                'description': rule.description
            }
            
            if rule.deprecated:
                summary['deprecated_parameters'].append({
                    'source_key': rule.source_key,
                    'description': rule.description
                })
            elif rule.required:
                summary['required_parameters'].append({
                    'source_key': rule.source_key,
                    **mapping_info
                })
            else:
                summary['optional_parameters'].append({
                    'source_key': rule.source_key,
                    **mapping_info
                })
            
            summary['parameter_mappings'][rule.source_key] = mapping_info
        
        return summary
    
    def validate_conversion_rules(self) -> Tuple[bool, List[str]]:
        """Validate all conversion rules for consistency.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        for component_type, rules in self.conversion_rules.items():
            # Check for duplicate source keys
            source_keys = [rule.source_key for rule in rules]
            duplicates = set([key for key in source_keys if source_keys.count(key) > 1])
            if duplicates:
                errors.append(f"Duplicate source keys in {component_type}: {duplicates}")
            
            # Check for duplicate target keys (excluding None for deprecated params)
            target_keys = [rule.target_key for rule in rules if rule.target_key]
            duplicates = set([key for key in target_keys if target_keys.count(key) > 1])
            if duplicates:
                errors.append(f"Duplicate target keys in {component_type}: {duplicates}")
            
            # Check for required parameters without converters
            for rule in rules:
                if rule.required and not rule.converter and rule.default_value is None:
                    errors.append(f"Required parameter {rule.source_key} in {component_type} "
                                f"has no converter or default value")
        
        is_valid = len(errors) == 0
        return is_valid, errors
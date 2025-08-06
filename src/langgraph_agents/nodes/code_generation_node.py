"""
Code generation node function following LangGraph patterns.

This module implements the code_generation_node function that converts CodeGeneratorAgent logic
to a simple async function using CodeGenerationService for business logic.
"""

import logging
import asyncio
from typing import Dict, Any, Tuple

from ..models.state import VideoGenerationState
from ..models.errors import WorkflowError, ErrorType, ErrorSeverity
from ..services.code_generation_service import CodeGenerationService

logger = logging.getLogger(__name__)


async def code_generation_node(state: VideoGenerationState) -> VideoGenerationState:
    """
    Code generation node function that generates Manim code for all scenes.
    
    This function follows LangGraph best practices by:
    - Taking state as input and returning updated state
    - Using CodeGenerationService for business logic separation
    - Implementing parallel processing for multiple scenes
    - Adding comprehensive error handling and logging
    
    Args:
        state: Current workflow state
        
    Returns:
        VideoGenerationState: Updated state with generated code
    """
    logger.info(f"Starting code generation node for session {state.session_id}")
    
    # Update current step
    state.current_step = "code_generation"
    state.add_execution_trace("code_generation_node", {
        "action": "started", 
        "session_id": state.session_id,
        "scene_count": len(state.scene_implementations)
    })
    
    try:
        # Validate input state
        validation_error = _validate_input_state(state)
        if validation_error:
            state.add_error(validation_error)
            return state
        
        # Initialize code generation service
        code_gen_config = _build_code_generation_config(state)
        code_gen_service = CodeGenerationService(code_gen_config)
        
        # Get model wrappers from state config
        model_wrappers = _get_model_wrappers(state)
        
        # Determine if we should use parallel processing
        max_concurrent = min(state.config.max_concurrent_scenes, len(state.scene_implementations))
        use_parallel = len(state.scene_implementations) > 1 and max_concurrent > 1
        
        if use_parallel:
            logger.info(f"Using parallel code generation with max_concurrent={max_concurrent}")
            generated_results = await _generate_code_parallel(
                state, code_gen_service, model_wrappers, max_concurrent
            )
        else:
            logger.info("Using sequential code generation")
            generated_results = await _generate_code_sequential(
                state, code_gen_service, model_wrappers
            )
        
        # Process results and update state
        successful_generations = 0
        for scene_num, result in generated_results.items():
            if isinstance(result, Exception):
                # Handle generation error
                error = WorkflowError(
                    step="code_generation",
                    error_type=ErrorType.MODEL,
                    message=f"Code generation failed for scene {scene_num}: {str(result)}",
                    severity=ErrorSeverity.HIGH,
                    scene_number=scene_num,
                    context={"exception": str(result), "exception_type": type(result).__name__}
                )
                state.add_error(error)
                state.code_errors[scene_num] = str(result)
            else:
                # Successful generation
                code, response_text = result
                
                # Validate generated code
                is_valid, issues = await code_gen_service.validate_generated_code(code, scene_num)
                if not is_valid:
                    error = WorkflowError(
                        step="code_generation",
                        error_type=ErrorType.CONTENT,
                        message=f"Generated code validation failed for scene {scene_num}: {'; '.join(issues)}",
                        severity=ErrorSeverity.MEDIUM,
                        scene_number=scene_num,
                        context={"validation_issues": issues}
                    )
                    state.add_error(error)
                    # Still store the code for potential fixing
                
                state.generated_code[scene_num] = code
                successful_generations += 1
                
                logger.info(f"Successfully generated code for scene {scene_num} (length: {len(code)})")
        
        # Update metrics if available
        if state.metrics:
            code_gen_metrics = code_gen_service.get_code_generation_metrics()
            code_gen_metrics.update({
                "scenes_processed": len(state.scene_implementations),
                "successful_generations": successful_generations,
                "failed_generations": len(state.scene_implementations) - successful_generations,
                "parallel_processing": use_parallel,
                "max_concurrent": max_concurrent if use_parallel else 1
            })
            state.metrics.add_step_metrics("code_generation", code_gen_metrics)
        
        # Add completion trace
        state.add_execution_trace("code_generation_node", {
            "action": "completed",
            "successful_generations": successful_generations,
            "failed_generations": len(state.scene_implementations) - successful_generations,
            "total_scenes": len(state.scene_implementations),
            "has_errors": state.has_errors(),
            "parallel_processing": use_parallel
        })
        
        logger.info(f"Code generation completed: {successful_generations}/{len(state.scene_implementations)} successful")
        
        # Cleanup service
        await code_gen_service.cleanup()
        
        return state
        
    except Exception as e:
        logger.error(f"Code generation node failed: {str(e)}")
        
        # Create workflow error
        error = WorkflowError(
            step="code_generation",
            error_type=ErrorType.SYSTEM,
            message=f"Code generation failed: {str(e)}",
            severity=ErrorSeverity.CRITICAL,
            context={"exception": str(e), "exception_type": type(e).__name__}
        )
        state.add_error(error)
        
        # Add failure trace
        state.add_execution_trace("code_generation_node", {
            "action": "failed",
            "error": str(e),
            "error_type": type(e).__name__
        })
        
        return state


async def _generate_code_parallel(
    state: VideoGenerationState,
    code_gen_service: CodeGenerationService,
    model_wrappers: Dict[str, Any],
    max_concurrent: int
) -> Dict[int, Tuple[str, str]]:
    """Generate code for multiple scenes in parallel."""
    try:
        results = await code_gen_service.generate_multiple_scenes_parallel(
            topic=state.topic,
            description=state.description,
            scene_outline=state.scene_outline,
            scene_implementations=state.scene_implementations,
            session_id=state.session_id,
            model_wrappers=model_wrappers,
            max_concurrent=max_concurrent
        )
        return results
        
    except Exception as e:
        logger.error(f"Parallel code generation failed: {e}")
        # Fall back to sequential processing
        logger.info("Falling back to sequential code generation")
        return await _generate_code_sequential(state, code_gen_service, model_wrappers)


async def _generate_code_sequential(
    state: VideoGenerationState,
    code_gen_service: CodeGenerationService,
    model_wrappers: Dict[str, Any]
) -> Dict[int, Tuple[str, str]]:
    """Generate code for scenes sequentially."""
    results = {}
    
    for scene_num, scene_implementation in state.scene_implementations.items():
        try:
            logger.info(f"Generating code for scene {scene_num}")
            
            code, response_text = await code_gen_service.generate_scene_code(
                topic=state.topic,
                description=state.description,
                scene_outline=state.scene_outline,
                scene_implementation=scene_implementation,
                scene_number=scene_num,
                session_id=state.session_id,
                model_wrappers=model_wrappers
            )
            
            results[scene_num] = (code, response_text)
            
        except Exception as e:
            logger.error(f"Failed to generate code for scene {scene_num}: {e}")
            results[scene_num] = e
    
    return results


def _validate_input_state(state: VideoGenerationState) -> WorkflowError:
    """Validate input state for code generation."""
    if not state.scene_outline:
        return WorkflowError(
            step="code_generation",
            error_type=ErrorType.VALIDATION,
            message="Scene outline is required for code generation",
            severity=ErrorSeverity.HIGH,
            context={"missing_field": "scene_outline"}
        )
    
    if not state.scene_implementations:
        return WorkflowError(
            step="code_generation",
            error_type=ErrorType.VALIDATION,
            message="Scene implementations are required for code generation",
            severity=ErrorSeverity.HIGH,
            context={"missing_field": "scene_implementations"}
        )
    
    # Validate individual scene implementations
    for scene_num, implementation in state.scene_implementations.items():
        if not implementation or not implementation.strip():
            return WorkflowError(
                step="code_generation",
                error_type=ErrorType.VALIDATION,
                message=f"Scene {scene_num} implementation is empty",
                severity=ErrorSeverity.HIGH,
                scene_number=scene_num,
                context={"empty_implementation": scene_num}
            )
    
    return None


def _build_code_generation_config(state: VideoGenerationState) -> Dict[str, Any]:
    """Build configuration for CodeGenerationService from state."""
    config = {
        'scene_model': f"{state.config.code_model.provider}/{state.config.code_model.model_name}",
        'helper_model': f"{state.config.helper_model.provider}/{state.config.helper_model.model_name}",
        'output_dir': state.config.output_dir,
        'use_rag': state.config.use_rag,
        'use_context_learning': state.config.use_context_learning,
        'context_learning_path': state.config.context_learning_path,
        'chroma_db_path': state.config.chroma_db_path,
        'manim_docs_path': state.config.manim_docs_path,
        'embedding_model': state.config.embedding_model,
        'use_visual_fix_code': state.config.use_visual_analysis,
        'use_langfuse': state.config.use_langfuse,
        'max_retries': state.config.max_retries,
        'enable_caching': state.config.enable_caching,
        'session_id': state.session_id,
        'print_response': False  # Keep logs clean in production
    }
    
    return config


def _get_model_wrappers(state: VideoGenerationState) -> Dict[str, Any]:
    """Get model wrappers from state configuration.
    
    Note: This is a placeholder implementation. In the actual system,
    model wrappers would be created from the model configuration.
    """
    return {
        'scene_model': {
            'provider': state.config.code_model.provider,
            'model_name': state.config.code_model.model_name,
            'temperature': state.config.code_model.temperature,
            'max_tokens': state.config.code_model.max_tokens
        },
        'helper_model': {
            'provider': state.config.helper_model.provider,
            'model_name': state.config.helper_model.model_name,
            'temperature': state.config.helper_model.temperature,
            'max_tokens': state.config.helper_model.max_tokens
        }
    }
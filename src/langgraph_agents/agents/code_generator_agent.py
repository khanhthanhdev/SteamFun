"""
CodeGeneratorAgent with error handling and RAG integration.
Ports CodeGenerator functionality to LangGraph agent pattern while maintaining compatibility.
"""

import os
import re
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from langgraph.types import Command
from PIL import Image

from ..base_agent import BaseAgent
from ..state import VideoGenerationState
from src.core.code_generator import CodeGenerator


logger = logging.getLogger(__name__)


class CodeGeneratorAgent(BaseAgent):
    """CodeGeneratorAgent for Manim code generation and error correction.
    
    Ports CodeGenerator functionality to LangGraph agent pattern while
    preserving existing method signatures and RAG integration patterns.
    """
    
    def __init__(self, config, system_config):
        """Initialize CodeGeneratorAgent with error handling capabilities.
        
        Args:
            config: Agent configuration
            system_config: System configuration
        """
        super().__init__(config, system_config)
        
        # Initialize internal code generator (will be created on first use)
        self._code_generator = None
        
        logger.info(f"CodeGeneratorAgent initialized with config: {config.name}")
    
    def _get_code_generator(self, state: VideoGenerationState) -> CodeGenerator:
        """Get or create CodeGenerator instance with current state configuration.
        
        Args:
            state: Current workflow state
            
        Returns:
            CodeGenerator: Configured code generator instance
        """
        if self._code_generator is None:
            # Get model wrappers compatible with existing patterns
            scene_model = self.get_model_wrapper(self.scene_model or self.model_config.get('scene_model', 'openrouter/anthropic/claude-3.5-sonnet'), state)
            helper_model = self.get_model_wrapper(self.helper_model or self.model_config.get('helper_model', 'openrouter/anthropic/claude-3.5-sonnet'), state)
            
            # Initialize CodeGenerator with state configuration
            self._code_generator = CodeGenerator(
                scene_model=scene_model,
                helper_model=helper_model,
                output_dir=state.get('output_dir', 'output'),
                print_response=state.get('print_response', False),
                use_rag=state.get('use_rag', True),
                use_context_learning=state.get('use_context_learning', True),
                context_learning_path=state.get('context_learning_path', 'data/context_learning'),
                chroma_db_path=state.get('chroma_db_path', 'data/rag/chroma_db'),
                manim_docs_path=state.get('manim_docs_path', 'data/rag/manim_docs'),
                embedding_model=state.get('embedding_model', 'hf:ibm-granite/granite-embedding-30m-english'),
                use_visual_fix_code=state.get('use_visual_fix_code', False),
                use_langfuse=state.get('use_langfuse', True),
                session_id=state.get('session_id')
            )
            
            logger.info("Created CodeGenerator instance with current state configuration")
        
        return self._code_generator
    
    async def execute(self, state: VideoGenerationState) -> Command:
        """Execute Manim code generation with error handling and retry mechanisms.
        
        Args:
            state: Current workflow state
            
        Returns:
            Command: LangGraph command for next action
        """
        self.log_agent_action("starting_code_generation", {
            'topic': state.get('topic', ''),
            'scene_count': len(state.get('scene_implementations', {})),
            'use_rag': state.get('use_rag', False)
        })
        
        try:
            # Get required data from state
            scene_outline = state.get('scene_outline')
            scene_implementations = state.get('scene_implementations', {})
            detected_plugins = state.get('detected_plugins', [])
            
            if not scene_outline:
                raise ValueError("No scene outline available for code generation")
            
            if not scene_implementations:
                raise ValueError("No scene implementations available for code generation")
            
            # Get code generator instance
            code_generator = self._get_code_generator(state)
            
            # Generate code for each scene with retry mechanisms
            generated_code = state.get('generated_code', {})
            code_errors = state.get('code_errors', {})
            retry_count = state.get('retry_count', {})
            
            for scene_number, implementation in scene_implementations.items():
                # Skip if already successfully generated
                if scene_number in generated_code:
                    continue
                    
                scene_retry_count = retry_count.get(f"scene_{scene_number}", 0)
                max_retries = self.max_retries
                
                for attempt in range(max_retries):
                    try:
                        self.log_agent_action("generating_scene_code", {
                            'scene_number': scene_number,
                            'attempt': attempt + 1,
                            'max_attempts': max_retries,
                            'implementation_length': len(implementation)
                        })
                        
                        # Generate Manim code using existing method signature
                        code, response_text = await self.generate_manim_code(
                            topic=state['topic'],
                            description=state['description'],
                            scene_outline=scene_outline,
                            scene_implementation=implementation,
                            scene_number=scene_number,
                            state=state,
                            additional_context=None,
                            scene_trace_id=f"scene_{scene_number}_{state['session_id']}",
                            session_id=state['session_id']
                        )
                        
                        generated_code[scene_number] = code
                        
                        # Clear any previous errors for this scene
                        if scene_number in code_errors:
                            del code_errors[scene_number]
                        
                        self.log_agent_action("scene_code_generated", {
                            'scene_number': scene_number,
                            'attempt': attempt + 1,
                            'code_length': len(code)
                        })
                        
                        break  # Success, exit retry loop
                        
                    except Exception as scene_error:
                        logger.error(f"Error generating code for scene {scene_number} (attempt {attempt + 1}): {scene_error}")
                        
                        # Try to fix the error using existing fix_code_errors method
                        if attempt < max_retries - 1:  # Don't try to fix on last attempt
                            try:
                                self.log_agent_action("attempting_code_fix", {
                                    'scene_number': scene_number,
                                    'attempt': attempt + 1,
                                    'error': str(scene_error)
                                })
                                
                                fixed_code, fix_response = await self.fix_code_errors(
                                    implementation_plan=implementation,
                                    code="# Error occurred during generation",
                                    error=str(scene_error),
                                    scene_trace_id=f"scene_{scene_number}_{state['session_id']}",
                                    topic=state['topic'],
                                    scene_number=scene_number,
                                    session_id=state['session_id'],
                                    state=state
                                )
                                
                                generated_code[scene_number] = fixed_code
                                
                                self.log_agent_action("scene_code_fixed", {
                                    'scene_number': scene_number,
                                    'attempt': attempt + 1,
                                    'fixed_code_length': len(fixed_code)
                                })
                                
                                break  # Success, exit retry loop
                                
                            except Exception as fix_error:
                                logger.error(f"Failed to fix code for scene {scene_number} (attempt {attempt + 1}): {fix_error}")
                                
                                # If this is the last attempt, record the error
                                if attempt == max_retries - 1:
                                    code_errors[scene_number] = f"Generation error: {scene_error}, Fix error: {fix_error}"
                                    retry_count[f"scene_{scene_number}"] = scene_retry_count + attempt + 1
                        else:
                            # Last attempt failed
                            code_errors[scene_number] = str(scene_error)
                            retry_count[f"scene_{scene_number}"] = scene_retry_count + attempt + 1
            
            # Check if we have any successful code generation
            if not generated_code:
                # No scenes generated successfully - route to RAGAgent for additional context
                if state.get('use_rag', False) and not state.get('rag_context_attempted', False):
                    self.log_agent_action("routing_to_rag_agent", {
                        'reason': 'no_successful_code_generation',
                        'failed_scenes': len(code_errors)
                    })
                    
                    return Command(
                        goto="rag_agent",
                        update={
                            "generated_code": generated_code,
                            "code_errors": code_errors,
                            "retry_count": retry_count,
                            "rag_context_attempted": True,
                            "current_agent": "rag_agent",
                            "next_agent": "code_generator_agent"
                        }
                    )
                else:
                    raise ValueError("Failed to generate code for any scenes after all retry attempts")
            
            # Check if we have partial failures that might benefit from RAG assistance
            if code_errors and state.get('use_rag', False) and not state.get('rag_context_attempted', False):
                failure_rate = len(code_errors) / len(scene_implementations)
                if failure_rate > 0.3:  # More than 30% failure rate
                    self.log_agent_action("routing_to_rag_agent", {
                        'reason': 'high_failure_rate',
                        'failure_rate': failure_rate,
                        'failed_scenes': len(code_errors),
                        'successful_scenes': len(generated_code)
                    })
                    
                    return Command(
                        goto="rag_agent",
                        update={
                            "generated_code": generated_code,
                            "code_errors": code_errors,
                            "retry_count": retry_count,
                            "rag_context_attempted": True,
                            "current_agent": "rag_agent",
                            "next_agent": "code_generator_agent"
                        }
                    )
            
            self.log_agent_action("code_generation_completed", {
                'successful_scenes': len(generated_code),
                'failed_scenes': len(code_errors),
                'total_scenes': len(scene_implementations),
                'success_rate': len(generated_code) / len(scene_implementations)
            })
            
            # Route to RendererAgent with preserved workflow logic
            return Command(
                goto="renderer_agent",
                update={
                    "generated_code": generated_code,
                    "code_errors": code_errors,
                    "retry_count": retry_count,
                    "current_agent": "renderer_agent",
                    "next_agent": "renderer_agent"
                }
            )
            
        except Exception as e:
            logger.error(f"Error in CodeGeneratorAgent execution: {e}")
            
            # Check if we should escalate to human
            if self.should_escalate_to_human(state):
                return self.create_human_intervention_command(
                    context=f"Code generation failed for topic '{state.get('topic', '')}': {str(e)}",
                    options=["Retry with different approach", "Skip failed scenes", "Manual code review", "Use RAG assistance"],
                    state=state
                )
            
            # Handle error through base class
            return await self.handle_error(e, state)
    
    async def generate_manim_code(self,
                                topic: str,
                                description: str,
                                scene_outline: str,
                                scene_implementation: str,
                                scene_number: int,
                                state: VideoGenerationState,
                                additional_context: Union[str, List[str], None] = None,
                                scene_trace_id: Optional[str] = None,
                                session_id: Optional[str] = None) -> Tuple[str, str]:
        """Generate Manim code (compatible with existing API).
        
        Args:
            topic: Video topic
            description: Video description
            scene_outline: Scene outline
            scene_implementation: Scene implementation details
            scene_number: Scene number
            state: Current workflow state
            additional_context: Additional context
            scene_trace_id: Scene trace identifier
            session_id: Session identifier
            
        Returns:
            Tuple[str, str]: Generated code and response text
        """
        try:
            code_generator = self._get_code_generator(state)
            
            # Add RAG context if available and not already included
            if state.get('use_rag', False) and not additional_context:
                rag_queries = await self._generate_rag_queries_code(
                    implementation=scene_implementation,
                    scene_trace_id=scene_trace_id,
                    topic=topic,
                    scene_number=scene_number,
                    session_id=session_id,
                    relevant_plugins=state.get('detected_plugins', []),
                    state=state
                )
                
                if rag_queries:
                    rag_context = await self._retrieve_rag_context(
                        rag_queries=rag_queries,
                        scene_trace_id=scene_trace_id,
                        topic=topic,
                        scene_number=scene_number,
                        state=state
                    )
                    
                    if rag_context:
                        additional_context = [rag_context] if isinstance(additional_context, type(None)) else (
                            additional_context + [rag_context] if isinstance(additional_context, list) else [additional_context, rag_context]
                        )
            
            return code_generator.generate_manim_code(
                topic=topic,
                description=description,
                scene_outline=scene_outline,
                scene_implementation=scene_implementation,
                scene_number=scene_number,
                additional_context=additional_context,
                scene_trace_id=scene_trace_id,
                session_id=session_id
            )
            
        except Exception as e:
            logger.error(f"Error in generate_manim_code for scene {scene_number}: {e}")
            raise
    
    async def fix_code_errors(self,
                            implementation_plan: str,
                            code: str,
                            error: str,
                            scene_trace_id: str,
                            topic: str,
                            scene_number: int,
                            session_id: str,
                            state: VideoGenerationState) -> Tuple[str, str]:
        """Fix errors in generated Manim code with RAG assistance (compatible with existing API).
        
        Args:
            implementation_plan: Original implementation plan
            code: Code containing errors
            error: Error message to fix
            scene_trace_id: Scene trace identifier
            topic: Video topic
            scene_number: Scene number
            session_id: Session identifier
            state: Current workflow state
            
        Returns:
            Tuple[str, str]: Fixed code and response text
        """
        try:
            code_generator = self._get_code_generator(state)
            
            # Generate RAG queries for error fixing if RAG is enabled
            if state.get('use_rag', False):
                rag_queries = code_generator._generate_rag_queries_error_fix(
                    error=error,
                    code=code,
                    scene_trace_id=scene_trace_id,
                    topic=topic,
                    scene_number=scene_number,
                    session_id=session_id,
                    relevant_plugins=state.get('detected_plugins', [])
                )
                
                if rag_queries:
                    rag_context = code_generator._retrieve_rag_context(
                        rag_queries=rag_queries,
                        scene_trace_id=scene_trace_id,
                        topic=topic,
                        scene_number=scene_number
                    )
                    
                    # Store RAG context in state for potential reuse
                    if rag_context:
                        current_rag_context = state.get('rag_context', {})
                        current_rag_context[f"error_fix_scene_{scene_number}"] = rag_context
                        state['rag_context'] = current_rag_context
            
            return code_generator.fix_code_errors(
                implementation_plan=implementation_plan,
                code=code,
                error=error,
                scene_trace_id=scene_trace_id,
                topic=topic,
                scene_number=scene_number,
                session_id=session_id
            )
            
        except Exception as e:
            logger.error(f"Error in fix_code_errors for scene {scene_number}: {e}")
            raise
    
    async def visual_self_reflection(self,
                                   code: str,
                                   media_path: Union[str, Image.Image],
                                   scene_trace_id: str,
                                   topic: str,
                                   scene_number: int,
                                   session_id: str,
                                   state: VideoGenerationState) -> Tuple[str, str]:
        """Use visual analysis to fix code (compatible with existing API).
        
        Args:
            code: Code to analyze and fix
            media_path: Path to media file or PIL Image
            scene_trace_id: Scene trace identifier
            topic: Video topic
            scene_number: Scene number
            session_id: Session identifier
            state: Current workflow state
            
        Returns:
            Tuple[str, str]: Fixed code and response text
        """
        try:
            code_generator = self._get_code_generator(state)
            
            # Store visual analysis attempt in state
            visual_analysis_results = state.get('visual_analysis_results', {})
            visual_analysis_results[scene_number] = {
                'attempted': True,
                'media_path': str(media_path) if isinstance(media_path, (str, Path)) else 'PIL_Image',
                'timestamp': time.time()
            }
            state['visual_analysis_results'] = visual_analysis_results
            
            result = code_generator.visual_self_reflection(
                code=code,
                media_path=media_path,
                scene_trace_id=scene_trace_id,
                topic=topic,
                scene_number=scene_number,
                session_id=session_id
            )
            
            # Update results with success
            visual_analysis_results[scene_number]['success'] = True
            visual_analysis_results[scene_number]['fixed_code_length'] = len(result[0])
            
            return result
            
        except Exception as e:
            logger.error(f"Error in visual_self_reflection for scene {scene_number}: {e}")
            
            # Update results with failure
            visual_analysis_results = state.get('visual_analysis_results', {})
            if scene_number in visual_analysis_results:
                visual_analysis_results[scene_number]['success'] = False
                visual_analysis_results[scene_number]['error'] = str(e)
            
            raise
    
    async def enhanced_visual_self_reflection(self,
                                            code: str,
                                            media_path: Union[str, Image.Image],
                                            scene_trace_id: str,
                                            topic: str,
                                            scene_number: int,
                                            session_id: str,
                                            state: VideoGenerationState,
                                            implementation_plan: Optional[str] = None) -> Tuple[str, str]:
        """Enhanced visual self-reflection (compatible with existing API).
        
        Args:
            code: Code to analyze and fix
            media_path: Path to media file or PIL Image
            scene_trace_id: Scene trace identifier
            topic: Video topic
            scene_number: Scene number
            session_id: Session identifier
            state: Current workflow state
            implementation_plan: Optional implementation plan
            
        Returns:
            Tuple[str, str]: Fixed code and response text
        """
        code_generator = self._get_code_generator(state)
        return code_generator.enhanced_visual_self_reflection(
            code=code,
            media_path=media_path,
            scene_trace_id=scene_trace_id,
            topic=topic,
            scene_number=scene_number,
            session_id=session_id,
            implementation_plan=implementation_plan
        )
    
    async def _generate_rag_queries_code(self,
                                       implementation: str,
                                       scene_trace_id: Optional[str],
                                       topic: Optional[str],
                                       scene_number: Optional[int],
                                       session_id: Optional[str],
                                       relevant_plugins: List[str],
                                       state: VideoGenerationState) -> List[str]:
        """Generate RAG queries for code generation (compatible with existing method).
        
        Args:
            implementation: Implementation plan text
            scene_trace_id: Scene trace identifier
            topic: Video topic
            scene_number: Scene number
            session_id: Session identifier
            relevant_plugins: List of relevant plugins
            state: Current workflow state
            
        Returns:
            List[str]: Generated RAG queries
        """
        try:
            code_generator = self._get_code_generator(state)
            queries = code_generator._generate_rag_queries_code(
                implementation=implementation,
                scene_trace_id=scene_trace_id,
                topic=topic,
                scene_number=scene_number,
                session_id=session_id,
                relevant_plugins=relevant_plugins
            )
            
            # Store queries in state for monitoring
            current_rag_context = state.get('rag_context', {})
            current_rag_context[f"queries_scene_{scene_number}"] = queries
            state['rag_context'] = current_rag_context
            
            return queries
            
        except Exception as e:
            logger.error(f"Error generating RAG queries for scene {scene_number}: {e}")
            return []
    
    async def _retrieve_rag_context(self,
                                  rag_queries: List[str],
                                  scene_trace_id: Optional[str],
                                  topic: str,
                                  scene_number: int,
                                  state: VideoGenerationState) -> Optional[str]:
        """Retrieve context from RAG vector store (compatible with existing method).
        
        Args:
            rag_queries: List of RAG queries
            scene_trace_id: Scene trace identifier
            topic: Video topic
            scene_number: Scene number
            state: Current workflow state
            
        Returns:
            Optional[str]: Retrieved RAG context
        """
        try:
            if not rag_queries:
                return None
                
            code_generator = self._get_code_generator(state)
            context = code_generator._retrieve_rag_context(
                rag_queries=rag_queries,
                scene_trace_id=scene_trace_id,
                topic=topic,
                scene_number=scene_number
            )
            
            # Store context in state for monitoring and reuse
            if context:
                current_rag_context = state.get('rag_context', {})
                current_rag_context[f"context_scene_{scene_number}"] = context
                state['rag_context'] = current_rag_context
            
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving RAG context for scene {scene_number}: {e}")
            return None
    
    def get_code_generation_status(self, state: VideoGenerationState) -> Dict[str, Any]:
        """Get current code generation status and metrics.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dict: Code generation status information
        """
        generated_code = state.get('generated_code', {})
        code_errors = state.get('code_errors', {})
        
        return {
            'agent_name': self.name,
            'scenes_with_code': len(generated_code),
            'scenes_with_errors': len(code_errors),
            'total_scenes': len(state.get('scene_implementations', {})),
            'success_rate': len(generated_code) / max(1, len(state.get('scene_implementations', {}))),
            'execution_stats': self.execution_stats,
            'rag_enabled': state.get('use_rag', False),
            'visual_fix_enabled': state.get('use_visual_fix_code', False)
        }
    
    async def _extract_code_with_retries(self,
                                       response_text: str,
                                       state: VideoGenerationState,
                                       pattern: str = r"```python(.*)```",
                                       generation_name: Optional[str] = None,
                                       trace_id: Optional[str] = None,
                                       session_id: Optional[str] = None,
                                       max_retries: int = 3) -> str:
        """Extract code from response text with retry logic (compatible with existing method).
        
        Args:
            response_text: The text containing code to extract
            state: Current workflow state
            pattern: Regex pattern for extracting code
            generation_name: Name of generation step
            trace_id: Trace identifier
            session_id: Session identifier
            max_retries: Maximum number of retries
            
        Returns:
            The extracted code
            
        Raises:
            ValueError: If code extraction fails after max retries
        """
        try:
            code_generator = self._get_code_generator(state)
            return code_generator._extract_code_with_retries(
                response_text=response_text,
                pattern=pattern,
                generation_name=generation_name,
                trace_id=trace_id,
                session_id=session_id,
                max_retries=max_retries
            )
        except Exception as e:
            logger.error(f"Error in _extract_code_with_retries: {e}")
            raise

    async def handle_code_generation_error(self,
                                         error: Exception,
                                         state: VideoGenerationState,
                                         scene_number: Optional[int] = None,
                                         retry_with_rag: bool = True) -> Command:
        """Handle code generation-specific errors with recovery strategies.
        
        Args:
            error: Exception that occurred
            state: Current workflow state
            scene_number: Scene number where error occurred
            retry_with_rag: Whether to retry with RAG assistance
            
        Returns:
            Command: LangGraph command for error handling
        """
        error_type = type(error).__name__
        
        # Code generation-specific error handling
        if "code" in str(error).lower() and retry_with_rag and state.get('use_rag', False):
            self.log_agent_action("retrying_with_rag", {
                'error_type': error_type,
                'scene_number': scene_number,
                'retry_strategy': 'rag_assisted'
            })
            
            # Route to RAGAgent for additional context
            return Command(
                goto="rag_agent",
                update={
                    "current_agent": "rag_agent",
                    "next_agent": "code_generator_agent",
                    "error_context": {
                        "agent": self.name,
                        "error": str(error),
                        "scene_number": scene_number,
                        "retry_with_rag": True
                    }
                }
            )
        
        # Use base error handling
        return await self.handle_error(error, state)
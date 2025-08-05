"""
CodeGenerationService - Business logic for Manim code generation and error correction.

Extracted from CodeGeneratorAgent to follow separation of concerns and single responsibility principles.
"""

import logging
import warnings
import re
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from PIL import Image

# Suppress warnings before importing components that may trigger them
from ..utils.warning_suppression import suppress_deprecation_warnings
suppress_deprecation_warnings()

from src.core.code_generator import CodeGenerator

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


class CodeGenerationService:
    """Service class for Manim code generation operations.
    
    Handles code generation, error fixing, and visual analysis integration
    with comprehensive error handling and logging.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize CodeGenerationService with configuration.
        
        Args:
            config: Service configuration containing model settings, paths, etc.
        """
        self.config = config
        self._code_generator = None
        
        # Extract configuration values
        self.scene_model = config.get('scene_model', 'openrouter/anthropic/claude-3.5-sonnet')
        self.helper_model = config.get('helper_model', 'openrouter/anthropic/claude-3.5-sonnet')
        self.output_dir = config.get('output_dir', 'output')
        self.use_rag = config.get('use_rag', True)
        self.use_context_learning = config.get('use_context_learning', True)
        self.context_learning_path = config.get('context_learning_path', 'data/context_learning')
        self.chroma_db_path = config.get('chroma_db_path', 'data/rag/chroma_db')
        self.manim_docs_path = config.get('manim_docs_path', 'data/rag/manim_docs')
        self.embedding_model = config.get('embedding_model', 'hf:ibm-granite/granite-embedding-30m-english')
        self.use_visual_fix_code = config.get('use_visual_fix_code', False)
        self.use_langfuse = config.get('use_langfuse', True)
        self.max_retries = config.get('max_retries', 3)
        self.enable_caching = config.get('enable_caching', True)
        
        logger.info(f"CodeGenerationService initialized with config: {self.config}")
    
    def _get_code_generator(self, model_wrappers: Dict[str, Any], session_id: str) -> CodeGenerator:
        """Get or create CodeGenerator instance.
        
        Args:
            model_wrappers: Dictionary containing scene_model and helper_model wrappers
            session_id: Session identifier
            
        Returns:
            CodeGenerator: Configured code generator instance
        """
        if self._code_generator is None:
            self._code_generator = CodeGenerator(
                scene_model=model_wrappers['scene_model'],
                helper_model=model_wrappers['helper_model'],
                output_dir=self.output_dir,
                print_response=self.config.get('print_response', False),
                use_rag=self.use_rag,
                use_context_learning=self.use_context_learning,
                context_learning_path=self.context_learning_path,
                chroma_db_path=self.chroma_db_path,
                manim_docs_path=self.manim_docs_path,
                embedding_model=self.embedding_model,
                use_visual_fix_code=self.use_visual_fix_code,
                use_langfuse=self.use_langfuse,
                session_id=session_id
            )
            
            logger.info("Created CodeGenerator instance")
        
        return self._code_generator
    
    async def generate_scene_code(self,
                                topic: str,
                                description: str,
                                scene_outline: str,
                                scene_implementation: str,
                                scene_number: int,
                                session_id: str,
                                model_wrappers: Dict[str, Any],
                                additional_context: Optional[Union[str, List[str]]] = None) -> Tuple[str, str]:
        """Generate Manim code for a single scene.
        
        Args:
            topic: Video topic
            description: Video description
            scene_outline: Scene outline
            scene_implementation: Scene implementation details
            scene_number: Scene number
            session_id: Session identifier
            model_wrappers: Dictionary containing model wrappers
            additional_context: Additional context for generation
            
        Returns:
            Tuple[str, str]: Generated code and response text
            
        Raises:
            ValueError: If required parameters are empty
            Exception: If code generation fails
        """
        if not topic or not topic.strip():
            raise ValueError("Topic cannot be empty")
        
        if not description or not description.strip():
            raise ValueError("Description cannot be empty")
        
        if not scene_outline or not scene_outline.strip():
            raise ValueError("Scene outline cannot be empty")
        
        if not scene_implementation or not scene_implementation.strip():
            raise ValueError("Scene implementation cannot be empty")
        
        if scene_number < 1:
            raise ValueError("Scene number must be positive")
        
        logger.info(f"Generating code for scene {scene_number} with topic: {topic}")
        
        try:
            code_generator = self._get_code_generator(model_wrappers, session_id)
            
            # Generate RAG context if enabled and not provided
            if self.use_rag and not additional_context:
                rag_queries = await self._generate_rag_queries_for_code(
                    implementation=scene_implementation,
                    topic=topic,
                    scene_number=scene_number,
                    session_id=session_id,
                    code_generator=code_generator
                )
                
                if rag_queries:
                    rag_context = await self._retrieve_rag_context(
                        rag_queries=rag_queries,
                        topic=topic,
                        scene_number=scene_number,
                        code_generator=code_generator
                    )
                    
                    if rag_context:
                        additional_context = [rag_context] if additional_context is None else (
                            additional_context + [rag_context] if isinstance(additional_context, list) 
                            else [additional_context, rag_context]
                        )
            
            # Generate the code
            code, response_text = code_generator.generate_manim_code(
                topic=topic,
                description=description,
                scene_outline=scene_outline,
                scene_implementation=scene_implementation,
                scene_number=scene_number,
                additional_context=additional_context,
                scene_trace_id=f"scene_{scene_number}_{session_id}",
                session_id=session_id
            )
            
            if not code:
                raise ValueError("Generated code is empty")
            
            logger.info(f"Successfully generated code for scene {scene_number} with length: {len(code)}")
            return code, response_text
            
        except Exception as e:
            logger.error(f"Error generating code for scene {scene_number}: {e}")
            raise
    
    async def fix_code_errors(self,
                            implementation_plan: str,
                            code: str,
                            error: str,
                            scene_number: int,
                            topic: str,
                            session_id: str,
                            model_wrappers: Dict[str, Any]) -> Tuple[str, str]:
        """Fix errors in generated Manim code.
        
        Args:
            implementation_plan: Original implementation plan
            code: Code containing errors
            error: Error message to fix
            scene_number: Scene number
            topic: Video topic
            session_id: Session identifier
            model_wrappers: Dictionary containing model wrappers
            
        Returns:
            Tuple[str, str]: Fixed code and response text
            
        Raises:
            ValueError: If required parameters are empty
            Exception: If error fixing fails
        """
        if not implementation_plan or not implementation_plan.strip():
            raise ValueError("Implementation plan cannot be empty")
        
        if not code or not code.strip():
            raise ValueError("Code cannot be empty")
        
        if not error or not error.strip():
            raise ValueError("Error message cannot be empty")
        
        if scene_number < 1:
            raise ValueError("Scene number must be positive")
        
        logger.info(f"Fixing code errors for scene {scene_number}: {error[:100]}...")
        
        try:
            code_generator = self._get_code_generator(model_wrappers, session_id)
            
            # Generate RAG queries for error fixing if enabled
            if self.use_rag:
                rag_queries = await self._generate_rag_queries_for_error_fix(
                    error=error,
                    code=code,
                    topic=topic,
                    scene_number=scene_number,
                    session_id=session_id,
                    code_generator=code_generator
                )
                
                if rag_queries:
                    rag_context = await self._retrieve_rag_context(
                        rag_queries=rag_queries,
                        topic=topic,
                        scene_number=scene_number,
                        code_generator=code_generator
                    )
                    
                    # Store RAG context for potential reuse
                    if rag_context:
                        logger.info(f"Retrieved RAG context for error fixing scene {scene_number}")
            
            # Fix the code errors
            fixed_code, response_text = code_generator.fix_code_errors(
                implementation_plan=implementation_plan,
                code=code,
                error=error,
                scene_trace_id=f"scene_{scene_number}_{session_id}",
                topic=topic,
                scene_number=scene_number,
                session_id=session_id
            )
            
            if not fixed_code:
                raise ValueError("Fixed code is empty")
            
            logger.info(f"Successfully fixed code errors for scene {scene_number}")
            return fixed_code, response_text
            
        except Exception as e:
            logger.error(f"Error fixing code for scene {scene_number}: {e}")
            raise
    
    async def visual_analysis_fix(self,
                                code: str,
                                media_path: Union[str, Image.Image],
                                scene_number: int,
                                topic: str,
                                session_id: str,
                                model_wrappers: Dict[str, Any],
                                implementation_plan: Optional[str] = None) -> Tuple[str, str]:
        """Use visual analysis to fix code issues.
        
        Args:
            code: Code to analyze and fix
            media_path: Path to media file or PIL Image
            scene_number: Scene number
            topic: Video topic
            session_id: Session identifier
            model_wrappers: Dictionary containing model wrappers
            implementation_plan: Optional implementation plan
            
        Returns:
            Tuple[str, str]: Fixed code and response text
            
        Raises:
            ValueError: If required parameters are empty
            Exception: If visual analysis fails
        """
        if not code or not code.strip():
            raise ValueError("Code cannot be empty")
        
        if not media_path:
            raise ValueError("Media path cannot be empty")
        
        if scene_number < 1:
            raise ValueError("Scene number must be positive")
        
        if not self.use_visual_fix_code:
            raise ValueError("Visual code fixing is not enabled")
        
        logger.info(f"Performing visual analysis for scene {scene_number}")
        
        try:
            code_generator = self._get_code_generator(model_wrappers, session_id)
            
            # Use enhanced visual self-reflection if implementation plan is provided
            if implementation_plan:
                fixed_code, response_text = code_generator.enhanced_visual_self_reflection(
                    code=code,
                    media_path=media_path,
                    scene_trace_id=f"scene_{scene_number}_{session_id}",
                    topic=topic,
                    scene_number=scene_number,
                    session_id=session_id,
                    implementation_plan=implementation_plan
                )
            else:
                fixed_code, response_text = code_generator.visual_self_reflection(
                    code=code,
                    media_path=media_path,
                    scene_trace_id=f"scene_{scene_number}_{session_id}",
                    topic=topic,
                    scene_number=scene_number,
                    session_id=session_id
                )
            
            if not fixed_code:
                raise ValueError("Visual analysis produced empty code")
            
            logger.info(f"Successfully completed visual analysis for scene {scene_number}")
            return fixed_code, response_text
            
        except Exception as e:
            logger.error(f"Error in visual analysis for scene {scene_number}: {e}")
            raise
    
    async def generate_multiple_scenes_parallel(self,
                                              topic: str,
                                              description: str,
                                              scene_outline: str,
                                              scene_implementations: Dict[int, str],
                                              session_id: str,
                                              model_wrappers: Dict[str, Any],
                                              max_concurrent: int = 3) -> Dict[int, Tuple[str, str]]:
        """Generate code for multiple scenes in parallel.
        
        Args:
            topic: Video topic
            description: Video description
            scene_outline: Scene outline
            scene_implementations: Dictionary of scene implementations by scene number
            session_id: Session identifier
            model_wrappers: Dictionary containing model wrappers
            max_concurrent: Maximum number of concurrent generations
            
        Returns:
            Dict[int, Tuple[str, str]]: Generated code and response text by scene number
            
        Raises:
            ValueError: If required parameters are empty
        """
        if not scene_implementations:
            raise ValueError("Scene implementations cannot be empty")
        
        logger.info(f"Generating code for {len(scene_implementations)} scenes in parallel")
        
        import asyncio
        
        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_single_scene(scene_num: int, implementation: str) -> Tuple[int, Tuple[str, str]]:
            """Generate code for a single scene with semaphore control."""
            async with semaphore:
                try:
                    code, response = await self.generate_scene_code(
                        topic=topic,
                        description=description,
                        scene_outline=scene_outline,
                        scene_implementation=implementation,
                        scene_number=scene_num,
                        session_id=session_id,
                        model_wrappers=model_wrappers
                    )
                    return scene_num, (code, response)
                except Exception as e:
                    logger.error(f"Failed to generate code for scene {scene_num}: {e}")
                    raise
        
        # Create tasks for all scenes
        tasks = [
            generate_single_scene(scene_num, implementation)
            for scene_num, implementation in scene_implementations.items()
        ]
        
        # Execute tasks and collect results
        results = {}
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(completed_tasks):
            if isinstance(result, Exception):
                scene_num = list(scene_implementations.keys())[i]
                logger.error(f"Scene {scene_num} generation failed: {result}")
                # Re-raise the exception to be handled by the caller
                raise result
            else:
                scene_num, (code, response) = result
                results[scene_num] = (code, response)
        
        logger.info(f"Successfully generated code for {len(results)} scenes")
        return results
    
    async def _generate_rag_queries_for_code(self,
                                           implementation: str,
                                           topic: str,
                                           scene_number: int,
                                           session_id: str,
                                           code_generator: CodeGenerator) -> List[str]:
        """Generate RAG queries for code generation.
        
        Args:
            implementation: Implementation plan text
            topic: Video topic
            scene_number: Scene number
            session_id: Session identifier
            code_generator: CodeGenerator instance
            
        Returns:
            List[str]: Generated RAG queries
        """
        try:
            if hasattr(code_generator, '_generate_rag_queries_code'):
                queries = code_generator._generate_rag_queries_code(
                    implementation=implementation,
                    scene_trace_id=f"scene_{scene_number}_{session_id}",
                    topic=topic,
                    scene_number=scene_number,
                    session_id=session_id,
                    relevant_plugins=[]  # Could be enhanced to pass detected plugins
                )
                logger.info(f"Generated {len(queries)} RAG queries for scene {scene_number}")
                return queries
            else:
                logger.warning("CodeGenerator does not support RAG query generation")
                return []
        except Exception as e:
            logger.error(f"Error generating RAG queries for scene {scene_number}: {e}")
            return []
    
    async def _generate_rag_queries_for_error_fix(self,
                                                error: str,
                                                code: str,
                                                topic: str,
                                                scene_number: int,
                                                session_id: str,
                                                code_generator: CodeGenerator) -> List[str]:
        """Generate RAG queries for error fixing.
        
        Args:
            error: Error message
            code: Code with errors
            topic: Video topic
            scene_number: Scene number
            session_id: Session identifier
            code_generator: CodeGenerator instance
            
        Returns:
            List[str]: Generated RAG queries
        """
        try:
            if hasattr(code_generator, '_generate_rag_queries_error_fix'):
                queries = code_generator._generate_rag_queries_error_fix(
                    error=error,
                    code=code,
                    scene_trace_id=f"scene_{scene_number}_{session_id}",
                    topic=topic,
                    scene_number=scene_number,
                    session_id=session_id,
                    relevant_plugins=[]  # Could be enhanced to pass detected plugins
                )
                logger.info(f"Generated {len(queries)} RAG queries for error fixing scene {scene_number}")
                return queries
            else:
                logger.warning("CodeGenerator does not support RAG query generation for error fixing")
                return []
        except Exception as e:
            logger.error(f"Error generating RAG queries for error fixing scene {scene_number}: {e}")
            return []
    
    async def _retrieve_rag_context(self,
                                  rag_queries: List[str],
                                  topic: str,
                                  scene_number: int,
                                  code_generator: CodeGenerator) -> Optional[str]:
        """Retrieve context from RAG vector store.
        
        Args:
            rag_queries: List of RAG queries
            topic: Video topic
            scene_number: Scene number
            code_generator: CodeGenerator instance
            
        Returns:
            Optional[str]: Retrieved RAG context
        """
        try:
            if not rag_queries:
                return None
            
            if hasattr(code_generator, '_retrieve_rag_context'):
                context = code_generator._retrieve_rag_context(
                    rag_queries=rag_queries,
                    scene_trace_id=f"scene_{scene_number}_{topic}",
                    topic=topic,
                    scene_number=scene_number
                )
                
                if context:
                    logger.info(f"Retrieved RAG context for scene {scene_number} (length: {len(context)})")
                    return context
                else:
                    logger.info(f"No RAG context retrieved for scene {scene_number}")
                    return None
            else:
                logger.warning("CodeGenerator does not support RAG context retrieval")
                return None
        except Exception as e:
            logger.error(f"Error retrieving RAG context for scene {scene_number}: {e}")
            return None
    
    async def validate_generated_code(self, code: str, scene_number: int) -> Tuple[bool, List[str]]:
        """Validate generated code for security and correctness.
        
        Args:
            code: Generated code to validate
            scene_number: Scene number for logging
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_issues)
        """
        issues = []
        
        if not code or not code.strip():
            issues.append("Generated code is empty")
            return False, issues
        
        # Check for minimum code length
        if len(code.strip()) < 50:
            issues.append("Generated code is too short (minimum 50 characters)")
        
        # Check for dangerous patterns
        dangerous_patterns = [
            r'import\s+os',
            r'import\s+subprocess',
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
            r'raw_input\s*\('
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append(f"Code contains potentially dangerous pattern: {pattern}")
        
        # Check for Manim-specific elements
        manim_indicators = ['from manim import', 'import manim', 'Scene', 'self.play', 'self.add']
        has_manim_elements = any(indicator in code for indicator in manim_indicators)
        
        if not has_manim_elements:
            issues.append("Code lacks Manim-specific elements")
        
        # Check for basic Python syntax (simple check)
        try:
            compile(code, f"scene_{scene_number}.py", "exec")
        except SyntaxError as e:
            issues.append(f"Code has syntax errors: {e}")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info(f"Code validation passed for scene {scene_number}")
        else:
            logger.warning(f"Code validation failed for scene {scene_number} with issues: {issues}")
        
        return is_valid, issues
    
    def get_code_generation_metrics(self) -> Dict[str, Any]:
        """Get code generation service metrics and statistics.
        
        Returns:
            Dict[str, Any]: Code generation metrics
        """
        metrics = {
            'service_name': 'CodeGenerationService',
            'config': {
                'use_rag': self.use_rag,
                'use_context_learning': self.use_context_learning,
                'use_visual_fix_code': self.use_visual_fix_code,
                'max_retries': self.max_retries,
                'enable_caching': self.enable_caching
            },
            'code_generator_initialized': self._code_generator is not None
        }
        
        # Add code generator metrics if available
        if self._code_generator and hasattr(self._code_generator, 'get_metrics'):
            try:
                generator_metrics = self._code_generator.get_metrics()
                metrics['generator_metrics'] = generator_metrics
            except Exception as e:
                logger.warning(f"Could not retrieve code generator metrics: {e}")
        
        return metrics
    
    async def cleanup(self):
        """Cleanup resources used by the code generation service."""
        try:
            if self._code_generator:
                # Cleanup code generator resources if needed
                if hasattr(self._code_generator, 'cleanup'):
                    self._code_generator.cleanup()
                self._code_generator = None
                logger.info("Code generation service cleaned up")
                
        except Exception as e:
            logger.warning(f"Error during code generation service cleanup: {e}")
    
    def __del__(self):
        """Cleanup resources when service is destroyed."""
        try:
            if self._code_generator:
                # Cleanup code generator resources if needed
                if hasattr(self._code_generator, 'cleanup'):
                    self._code_generator.cleanup()
        except Exception:
            pass  # Ignore cleanup errors in destructor
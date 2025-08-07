"""
Agent-specific test runners for LangGraph Studio integration.

This module provides test runners for each agent type that can be executed
individually in LangGraph Studio for testing and debugging purposes.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime

from ..models.state import VideoGenerationState
from ..models.config import WorkflowConfig, ModelConfig
from ..models.errors import WorkflowError, ErrorType, ErrorSeverity
from ..services.planning_service import PlanningService
from ..services.code_generation_service import CodeGenerationService
from ..services.rendering_service import RenderingService
from ..error_recovery.error_handler import ErrorHandler
from .output_capture import AgentOutputCapture
from .performance_metrics import get_performance_monitor, get_performance_analyzer

logger = logging.getLogger(__name__)


class BaseAgentTestRunner(ABC):
    """Base class for all agent test runners."""
    
    def __init__(self, config: WorkflowConfig):
        """Initialize the test runner with configuration."""
        self.config = config
        self.output_capture = AgentOutputCapture()
        self.performance_monitor = get_performance_monitor()
        self.performance_analyzer = get_performance_analyzer()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def run_test(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent test with given inputs."""
        pass
    
    @abstractmethod
    def validate_inputs(self, inputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate test inputs."""
        pass
    
    def create_test_state(self, **kwargs) -> VideoGenerationState:
        """Create a test state with default values."""
        defaults = {
            'topic': 'Test Topic',
            'description': 'Test Description',
            'config': self.config
        }
        defaults.update(kwargs)
        return VideoGenerationState(**defaults)
    
    def _create_mock_model_wrappers(self) -> Dict[str, Any]:
        """Create mock model wrappers for testing."""
        return {
            'planner_model': {
                'provider': self.config.planner_model.provider,
                'model_name': self.config.planner_model.model_name,
                'temperature': self.config.planner_model.temperature,
                'max_tokens': self.config.planner_model.max_tokens
            },
            'code_model': {
                'provider': self.config.code_model.provider,
                'model_name': self.config.code_model.model_name,
                'temperature': self.config.code_model.temperature,
                'max_tokens': self.config.code_model.max_tokens
            },
            'helper_model': {
                'provider': self.config.helper_model.provider,
                'model_name': self.config.helper_model.model_name,
                'temperature': self.config.helper_model.temperature,
                'max_tokens': self.config.helper_model.max_tokens
            }
        }


class PlannerAgentTestRunner(BaseAgentTestRunner):
    """Test runner for PlannerAgent with topic/description inputs."""
    
    def __init__(self, config: WorkflowConfig):
        """Initialize PlannerAgent test runner."""
        super().__init__(config)
        self.planning_service = None
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate inputs for planner agent testing."""
        issues = []
        
        if 'topic' not in inputs:
            issues.append("Missing required input: topic")
        elif not inputs['topic'] or not inputs['topic'].strip():
            issues.append("Topic cannot be empty")
        
        if 'description' not in inputs:
            issues.append("Missing required input: description")
        elif not inputs['description'] or not inputs['description'].strip():
            issues.append("Description cannot be empty")
        
        # Optional validation for additional inputs
        if 'session_id' in inputs and not inputs['session_id']:
            issues.append("Session ID cannot be empty if provided")
        
        return len(issues) == 0, issues
    
    async def run_test(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run PlannerAgent test with topic/description inputs."""
        start_time = time.time()
        
        # Validate inputs
        is_valid, validation_issues = self.validate_inputs(inputs)
        if not is_valid:
            return {
                'success': False,
                'error': f"Input validation failed: {'; '.join(validation_issues)}",
                'execution_time': time.time() - start_time,
                'agent': 'PlannerAgent',
                'inputs': inputs
            }
        
        try:
            # Extract inputs
            topic = inputs['topic']
            description = inputs['description']
            session_id = inputs.get('session_id', f"test_{int(time.time())}")
            
            # Create test state
            state = self.create_test_state(
                topic=topic,
                description=description,
                session_id=session_id
            )
            
            # Initialize planning service
            planning_config = self._build_planning_config(state)
            self.planning_service = PlanningService(planning_config)
            
            # Create model wrappers
            model_wrappers = self._create_mock_model_wrappers()
            
            # Start output capture and performance monitoring
            self.output_capture.start_capture(f"planner_{session_id}", "PlannerAgent")
            performance_metrics = self.performance_monitor.start_monitoring(f"planner_{session_id}", "PlannerAgent")
            
            # Test scene outline generation
            self.logger.info(f"Testing scene outline generation for topic: {topic}")
            self.output_capture.add_state_tracking(f"planner_{session_id}", "outline_generation_start", {
                'topic': topic,
                'description': description[:100] + "..." if len(description) > 100 else description
            })
            
            scene_outline = await self.planning_service.generate_scene_outline(
                topic=topic,
                description=description,
                session_id=session_id,
                model_wrappers=model_wrappers
            )
            
            # Track API call
            self.performance_monitor.increment_counter(f"planner_{session_id}", "api_calls_made")
            
            self.output_capture.add_state_tracking(f"planner_{session_id}", "outline_generation_complete", {
                'outline_length': len(scene_outline) if scene_outline else 0,
                'outline_preview': scene_outline[:200] + "..." if scene_outline and len(scene_outline) > 200 else scene_outline
            })
            
            # Validate scene outline
            outline_valid, outline_issues = await self.planning_service.validate_scene_outline(scene_outline)
            
            # Test scene implementations generation
            self.logger.info("Testing scene implementations generation")
            self.output_capture.add_state_tracking(f"planner_{session_id}", "implementations_generation_start", {
                'outline_available': bool(scene_outline)
            })
            
            scene_implementations = await self.planning_service.generate_scene_implementations(
                topic=topic,
                description=description,
                plan=scene_outline,
                session_id=session_id,
                model_wrappers=model_wrappers
            )
            
            # Track API call
            self.performance_monitor.increment_counter(f"planner_{session_id}", "api_calls_made")
            
            self.output_capture.add_state_tracking(f"planner_{session_id}", "implementations_generation_complete", {
                'scene_count': len(scene_implementations) if scene_implementations else 0,
                'scene_numbers': list(scene_implementations.keys()) if scene_implementations else []
            })
            
            # Validate scene implementations
            impl_valid, impl_issues = await self.planning_service.validate_scene_implementations(scene_implementations)
            
            # Test plugin detection
            self.logger.info("Testing plugin detection")
            self.output_capture.add_state_tracking(f"planner_{session_id}", "plugin_detection_start", {})
            
            detected_plugins = await self.planning_service.detect_plugins(
                topic=topic,
                description=description,
                model_wrappers=model_wrappers
            )
            
            # Track API call
            self.performance_monitor.increment_counter(f"planner_{session_id}", "api_calls_made")
            
            self.output_capture.add_state_tracking(f"planner_{session_id}", "plugin_detection_complete", {
                'plugin_count': len(detected_plugins) if detected_plugins else 0,
                'detected_plugins': detected_plugins if detected_plugins else []
            })
            
            # Get planning metrics
            planning_metrics = self.planning_service.get_planning_metrics()
            
            # Collect performance metrics
            performance_metrics = {
                'execution_time': time.time() - start_time,
                'scene_count': len(scene_implementations),
                'plugin_count': len(detected_plugins),
                'outline_generation_time': planning_metrics.get('outline_generation_time', 0),
                'implementation_generation_time': planning_metrics.get('implementation_generation_time', 0),
                'plugin_detection_time': planning_metrics.get('plugin_detection_time', 0),
                'total_tokens_used': planning_metrics.get('total_tokens_used', 0),
                'api_calls_made': planning_metrics.get('api_calls_made', 0)
            }
            
            # Add metrics to output capture
            self.output_capture.add_metrics(f"planner_{session_id}", performance_metrics)
            
            # Stop performance monitoring
            final_performance_metrics = self.performance_monitor.stop_monitoring(f"planner_{session_id}")
            if final_performance_metrics:
                self.performance_analyzer.add_metrics(final_performance_metrics)
                performance_issues = self.performance_analyzer.identify_performance_issues(final_performance_metrics)
                self.output_capture.add_results(f"planner_{session_id}", {
                    'performance_metrics': final_performance_metrics.to_dict(),
                    'performance_issues': performance_issues
                })
            
            # Stop output capture
            captured_output = self.output_capture.stop_capture(f"planner_{session_id}")
            
            # Build result
            result = {
                'success': True,
                'agent': 'PlannerAgent',
                'execution_time': time.time() - start_time,
                'inputs': inputs,
                'outputs': {
                    'scene_outline': scene_outline,
                    'scene_implementations': scene_implementations,
                    'detected_plugins': detected_plugins,
                    'scene_count': len(scene_implementations)
                },
                'validation': {
                    'scene_outline_valid': outline_valid,
                    'scene_outline_issues': outline_issues,
                    'scene_implementations_valid': impl_valid,
                    'scene_implementations_issues': impl_issues
                },
                'metrics': planning_metrics,
                'captured_output': captured_output
            }
            
            self.logger.info(f"PlannerAgent test completed successfully in {result['execution_time']:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"PlannerAgent test failed: {str(e)}")
            
            # Stop output capture on error
            captured_output = self.output_capture.stop_capture(f"planner_{inputs.get('session_id', 'unknown')}")
            
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'execution_time': time.time() - start_time,
                'agent': 'PlannerAgent',
                'inputs': inputs,
                'captured_output': captured_output
            }
        
        finally:
            # Cleanup
            if self.planning_service:
                await self.planning_service.cleanup()
    
    def _build_planning_config(self, state: VideoGenerationState) -> Dict[str, Any]:
        """Build configuration for PlanningService from state."""
        return {
            'planner_model': f"{state.config.planner_model.provider}/{state.config.planner_model.model_name}",
            'helper_model': f"{state.config.helper_model.provider}/{state.config.helper_model.model_name}",
            'output_dir': state.config.output_dir,
            'use_rag': state.config.use_rag,
            'use_context_learning': state.config.use_context_learning,
            'context_learning_path': state.config.context_learning_path,
            'chroma_db_path': state.config.chroma_db_path,
            'manim_docs_path': state.config.manim_docs_path,
            'embedding_model': state.config.embedding_model,
            'use_langfuse': state.config.use_langfuse,
            'max_scene_concurrency': state.config.max_concurrent_scenes,
            'enable_caching': state.config.enable_caching,
            'use_enhanced_rag': state.config.use_enhanced_rag,
            'session_id': state.session_id,
            'print_response': False
        }


class CodeGeneratorAgentTestRunner(BaseAgentTestRunner):
    """Test runner for CodeGeneratorAgent with scene implementation inputs."""
    
    def __init__(self, config: WorkflowConfig):
        """Initialize CodeGeneratorAgent test runner."""
        super().__init__(config)
        self.code_gen_service = None
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate inputs for code generator agent testing."""
        issues = []
        
        required_fields = ['topic', 'description', 'scene_outline', 'scene_implementations']
        for field in required_fields:
            if field not in inputs:
                issues.append(f"Missing required input: {field}")
            elif not inputs[field]:
                issues.append(f"{field} cannot be empty")
        
        # Validate scene implementations structure
        if 'scene_implementations' in inputs:
            scene_impls = inputs['scene_implementations']
            if not isinstance(scene_impls, dict):
                issues.append("scene_implementations must be a dictionary")
            else:
                for scene_num, impl in scene_impls.items():
                    try:
                        scene_num = int(scene_num)
                        if scene_num < 1:
                            issues.append(f"Scene number must be positive: {scene_num}")
                    except (ValueError, TypeError):
                        issues.append(f"Invalid scene number: {scene_num}")
                    
                    if not isinstance(impl, str) or not impl.strip():
                        issues.append(f"Scene implementation must be non-empty string for scene {scene_num}")
        
        return len(issues) == 0, issues
    
    async def run_test(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run CodeGeneratorAgent test with scene implementation inputs."""
        start_time = time.time()
        
        # Validate inputs
        is_valid, validation_issues = self.validate_inputs(inputs)
        if not is_valid:
            return {
                'success': False,
                'error': f"Input validation failed: {'; '.join(validation_issues)}",
                'execution_time': time.time() - start_time,
                'agent': 'CodeGeneratorAgent',
                'inputs': inputs
            }
        
        try:
            # Extract inputs
            topic = inputs['topic']
            description = inputs['description']
            scene_outline = inputs['scene_outline']
            scene_implementations = inputs['scene_implementations']
            session_id = inputs.get('session_id', f"test_{int(time.time())}")
            
            # Convert scene numbers to integers
            scene_implementations = {int(k): v for k, v in scene_implementations.items()}
            
            # Create test state
            state = self.create_test_state(
                topic=topic,
                description=description,
                scene_outline=scene_outline,
                scene_implementations=scene_implementations,
                session_id=session_id
            )
            
            # Initialize code generation service
            code_gen_config = self._build_code_generation_config(state)
            self.code_gen_service = CodeGenerationService(code_gen_config)
            
            # Create model wrappers
            model_wrappers = self._create_mock_model_wrappers()
            
            # Start output capture and performance monitoring
            self.output_capture.start_capture(f"codegen_{session_id}", "CodeGeneratorAgent")
            performance_metrics = self.performance_monitor.start_monitoring(f"codegen_{session_id}", "CodeGeneratorAgent")
            
            # Test code generation for each scene
            generated_code = {}
            generation_errors = {}
            validation_results = {}
            
            for scene_num, scene_implementation in scene_implementations.items():
                try:
                    self.logger.info(f"Testing code generation for scene {scene_num}")
                    self.output_capture.add_state_tracking(f"codegen_{session_id}", f"scene_{scene_num}_generation_start", {
                        'scene_number': scene_num,
                        'implementation_length': len(scene_implementation)
                    })
                    
                    code, response_text = await self.code_gen_service.generate_scene_code(
                        topic=topic,
                        description=description,
                        scene_outline=scene_outline,
                        scene_implementation=scene_implementation,
                        scene_number=scene_num,
                        session_id=session_id,
                        model_wrappers=model_wrappers
                    )
                    
                    generated_code[scene_num] = code
                    
                    # Track API call and tokens (if available)
                    self.performance_monitor.increment_counter(f"codegen_{session_id}", "api_calls_made")
                    if hasattr(self.code_gen_service, 'get_last_token_usage'):
                        token_usage = self.code_gen_service.get_last_token_usage()
                        if token_usage:
                            self.performance_monitor.increment_counter(f"codegen_{session_id}", "tokens_used", token_usage)
                    
                    # Validate generated code
                    is_valid, issues = await self.code_gen_service.validate_generated_code(code, scene_num)
                    validation_results[scene_num] = {
                        'valid': is_valid,
                        'issues': issues
                    }
                    
                    self.output_capture.add_state_tracking(f"codegen_{session_id}", f"scene_{scene_num}_generation_complete", {
                        'scene_number': scene_num,
                        'code_length': len(code) if code else 0,
                        'validation_valid': is_valid,
                        'validation_issues_count': len(issues) if issues else 0
                    })
                    
                except Exception as e:
                    self.logger.error(f"Code generation failed for scene {scene_num}: {str(e)}")
                    generation_errors[scene_num] = str(e)
                    self.output_capture.add_state_tracking(f"codegen_{session_id}", f"scene_{scene_num}_generation_failed", {
                        'scene_number': scene_num,
                        'error': str(e)
                    })
                    
                    # Track error
                    self.performance_monitor.increment_counter(f"codegen_{session_id}", "error_count")
            
            # Get code generation metrics
            code_gen_metrics = self.code_gen_service.get_code_generation_metrics()
            
            # Collect performance metrics
            performance_metrics = {
                'execution_time': time.time() - start_time,
                'total_scenes': len(scene_implementations),
                'successful_scenes': len(generated_code),
                'failed_scenes': len(generation_errors),
                'success_rate': len(generated_code) / len(scene_implementations) if scene_implementations else 0,
                'average_generation_time': code_gen_metrics.get('average_generation_time', 0),
                'total_tokens_used': code_gen_metrics.get('total_tokens_used', 0),
                'api_calls_made': code_gen_metrics.get('api_calls_made', 0),
                'code_validation_rate': sum(1 for v in validation_results.values() if v.get('valid', False)) / len(validation_results) if validation_results else 0
            }
            
            # Add metrics to output capture
            self.output_capture.add_metrics(f"codegen_{session_id}", performance_metrics)
            
            # Stop performance monitoring
            final_performance_metrics = self.performance_monitor.stop_monitoring(f"codegen_{session_id}")
            if final_performance_metrics:
                self.performance_analyzer.add_metrics(final_performance_metrics)
                performance_issues = self.performance_analyzer.identify_performance_issues(final_performance_metrics)
                self.output_capture.add_results(f"codegen_{session_id}", {
                    'performance_metrics': final_performance_metrics.to_dict(),
                    'performance_issues': performance_issues
                })
            
            # Stop output capture
            captured_output = self.output_capture.stop_capture(f"codegen_{session_id}")
            
            # Build result
            result = {
                'success': len(generation_errors) == 0,
                'agent': 'CodeGeneratorAgent',
                'execution_time': time.time() - start_time,
                'inputs': inputs,
                'outputs': {
                    'generated_code': generated_code,
                    'successful_scenes': list(generated_code.keys()),
                    'failed_scenes': list(generation_errors.keys()),
                    'total_scenes': len(scene_implementations)
                },
                'validation': validation_results,
                'errors': generation_errors,
                'metrics': code_gen_metrics,
                'captured_output': captured_output
            }
            
            self.logger.info(f"CodeGeneratorAgent test completed in {result['execution_time']:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"CodeGeneratorAgent test failed: {str(e)}")
            
            # Stop output capture on error
            captured_output = self.output_capture.stop_capture(f"codegen_{inputs.get('session_id', 'unknown')}")
            
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'execution_time': time.time() - start_time,
                'agent': 'CodeGeneratorAgent',
                'inputs': inputs,
                'captured_output': captured_output
            }
        
        finally:
            # Cleanup
            if self.code_gen_service:
                await self.code_gen_service.cleanup()
    
    def _build_code_generation_config(self, state: VideoGenerationState) -> Dict[str, Any]:
        """Build configuration for CodeGenerationService from state."""
        return {
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
            'print_response': False
        }


class RendererAgentTestRunner(BaseAgentTestRunner):
    """Test runner for RendererAgent with code execution inputs."""
    
    def __init__(self, config: WorkflowConfig):
        """Initialize RendererAgent test runner."""
        super().__init__(config)
        self.rendering_service = None
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate inputs for renderer agent testing."""
        issues = []
        
        if 'generated_code' not in inputs:
            issues.append("Missing required input: generated_code")
        elif not inputs['generated_code']:
            issues.append("generated_code cannot be empty")
        
        if 'file_prefix' not in inputs:
            issues.append("Missing required input: file_prefix")
        elif not inputs['file_prefix'] or not inputs['file_prefix'].strip():
            issues.append("file_prefix cannot be empty")
        
        # Validate generated code structure
        if 'generated_code' in inputs:
            generated_code = inputs['generated_code']
            if not isinstance(generated_code, dict):
                issues.append("generated_code must be a dictionary")
            else:
                for scene_num, code in generated_code.items():
                    try:
                        scene_num = int(scene_num)
                        if scene_num < 1:
                            issues.append(f"Scene number must be positive: {scene_num}")
                    except (ValueError, TypeError):
                        issues.append(f"Invalid scene number: {scene_num}")
                    
                    if not isinstance(code, str) or not code.strip():
                        issues.append(f"Code must be non-empty string for scene {scene_num}")
        
        # Validate optional quality setting
        if 'quality' in inputs:
            valid_qualities = ['low', 'medium', 'high', 'ultra', 'preview']
            if inputs['quality'] not in valid_qualities:
                issues.append(f"Invalid quality: {inputs['quality']}. Must be one of {valid_qualities}")
        
        return len(issues) == 0, issues
    
    async def run_test(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run RendererAgent test with code execution inputs."""
        start_time = time.time()
        
        # Validate inputs
        is_valid, validation_issues = self.validate_inputs(inputs)
        if not is_valid:
            return {
                'success': False,
                'error': f"Input validation failed: {'; '.join(validation_issues)}",
                'execution_time': time.time() - start_time,
                'agent': 'RendererAgent',
                'inputs': inputs
            }
        
        try:
            # Extract inputs
            generated_code = inputs['generated_code']
            file_prefix = inputs['file_prefix']
            quality = inputs.get('quality', 'medium')
            session_id = inputs.get('session_id', f"test_{int(time.time())}")
            topic = inputs.get('topic', 'Test Topic')
            description = inputs.get('description', 'Test Description')
            scene_outline = inputs.get('scene_outline', '')
            
            # Convert scene numbers to integers
            generated_code = {int(k): v for k, v in generated_code.items()}
            
            # Create test state
            state = self.create_test_state(
                topic=topic,
                description=description,
                generated_code=generated_code,
                session_id=session_id
            )
            
            # Initialize rendering service
            rendering_config = self._build_rendering_config(state)
            self.rendering_service = RenderingService(rendering_config)
            
            # Start output capture and performance monitoring
            self.output_capture.start_capture(f"renderer_{session_id}", "RendererAgent")
            performance_metrics = self.performance_monitor.start_monitoring(f"renderer_{session_id}", "RendererAgent")
            
            # Test rendering for each scene
            rendered_videos = {}
            rendering_errors = {}
            final_codes = {}
            
            for scene_num, code in generated_code.items():
                try:
                    self.logger.info(f"Testing rendering for scene {scene_num}")
                    self.output_capture.add_state_tracking(f"renderer_{session_id}", f"scene_{scene_num}_render_start", {
                        'scene_number': scene_num,
                        'code_length': len(code),
                        'quality': quality
                    })
                    
                    final_code, error = await self.rendering_service.render_scene(
                        code=code,
                        file_prefix=file_prefix,
                        scene_number=scene_num,
                        version=1,
                        quality=quality,
                        topic=topic,
                        description=description,
                        scene_outline=scene_outline,
                        session_id=session_id
                    )
                    
                    final_codes[scene_num] = final_code
                    
                    if error:
                        rendering_errors[scene_num] = error
                        self.output_capture.add_state_tracking(f"renderer_{session_id}", f"scene_{scene_num}_render_failed", {
                            'scene_number': scene_num,
                            'error': error
                        })
                    else:
                        # Try to find the rendered video file
                        try:
                            video_path = self.rendering_service.find_rendered_video(
                                file_prefix=file_prefix,
                                scene_number=scene_num,
                                version=1
                            )
                            rendered_videos[scene_num] = video_path
                            self.output_capture.add_state_tracking(f"renderer_{session_id}", f"scene_{scene_num}_render_complete", {
                                'scene_number': scene_num,
                                'video_path': video_path,
                                'final_code_length': len(final_code) if final_code else 0
                            })
                        except FileNotFoundError as e:
                            rendering_errors[scene_num] = f"Rendered video not found: {str(e)}"
                            self.output_capture.add_state_tracking(f"renderer_{session_id}", f"scene_{scene_num}_video_not_found", {
                                'scene_number': scene_num,
                                'error': str(e)
                            })
                    
                except Exception as e:
                    self.logger.error(f"Rendering failed for scene {scene_num}: {str(e)}")
                    rendering_errors[scene_num] = str(e)
                    self.output_capture.add_state_tracking(f"renderer_{session_id}", f"scene_{scene_num}_render_exception", {
                        'scene_number': scene_num,
                        'error': str(e)
                    })
            
            # Test video combination if we have multiple successful renders
            combined_video_path = None
            if len(rendered_videos) > 1:
                try:
                    self.logger.info("Testing video combination")
                    self.output_capture.add_state_tracking(f"renderer_{session_id}", "video_combination_start", {
                        'video_count': len(rendered_videos),
                        'video_scenes': list(rendered_videos.keys())
                    })
                    
                    combined_video_path = await self.rendering_service.combine_videos(
                        topic=topic,
                        rendered_videos=rendered_videos
                    )
                    
                    self.output_capture.add_state_tracking(f"renderer_{session_id}", "video_combination_complete", {
                        'combined_video_path': combined_video_path
                    })
                except Exception as e:
                    self.logger.error(f"Video combination failed: {str(e)}")
                    rendering_errors['combination'] = str(e)
                    self.output_capture.add_state_tracking(f"renderer_{session_id}", "video_combination_failed", {
                        'error': str(e)
                    })
            
            # Get rendering performance stats
            performance_stats = self.rendering_service.get_performance_stats()
            
            # Collect performance metrics
            performance_metrics = {
                'execution_time': time.time() - start_time,
                'total_scenes': len(generated_code),
                'successful_renders': len(rendered_videos),
                'failed_renders': len([k for k in rendering_errors.keys() if k != 'combination']),
                'rendering_success_rate': len(rendered_videos) / len(generated_code) if generated_code else 0,
                'average_render_time': performance_stats.get('average_render_time', 0),
                'total_render_time': performance_stats.get('total_render_time', 0),
                'video_combination_time': performance_stats.get('video_combination_time', 0),
                'has_combined_video': combined_video_path is not None,
                'quality_setting': quality
            }
            
            # Add metrics to output capture
            self.output_capture.add_metrics(f"renderer_{session_id}", performance_metrics)
            
            # Stop performance monitoring
            final_performance_metrics = self.performance_monitor.stop_monitoring(f"renderer_{session_id}")
            if final_performance_metrics:
                self.performance_analyzer.add_metrics(final_performance_metrics)
                performance_issues = self.performance_analyzer.identify_performance_issues(final_performance_metrics)
                self.output_capture.add_results(f"renderer_{session_id}", {
                    'performance_metrics': final_performance_metrics.to_dict(),
                    'performance_issues': performance_issues
                })
            
            # Stop output capture
            captured_output = self.output_capture.stop_capture(f"renderer_{session_id}")
            
            # Build result
            result = {
                'success': len(rendering_errors) == 0,
                'agent': 'RendererAgent',
                'execution_time': time.time() - start_time,
                'inputs': inputs,
                'outputs': {
                    'rendered_videos': rendered_videos,
                    'combined_video_path': combined_video_path,
                    'final_codes': final_codes,
                    'successful_scenes': list(rendered_videos.keys()),
                    'failed_scenes': [k for k in rendering_errors.keys() if k != 'combination'],
                    'total_scenes': len(generated_code)
                },
                'errors': rendering_errors,
                'performance_stats': performance_stats,
                'captured_output': captured_output
            }
            
            self.logger.info(f"RendererAgent test completed in {result['execution_time']:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"RendererAgent test failed: {str(e)}")
            
            # Stop output capture on error
            captured_output = self.output_capture.stop_capture(f"renderer_{inputs.get('session_id', 'unknown')}")
            
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'execution_time': time.time() - start_time,
                'agent': 'RendererAgent',
                'inputs': inputs,
                'captured_output': captured_output
            }
        
        finally:
            # Cleanup
            if self.rendering_service:
                await self.rendering_service.cleanup()
    
    def _build_rendering_config(self, state: VideoGenerationState) -> Dict[str, Any]:
        """Build configuration for RenderingService from state."""
        return {
            'output_dir': state.config.output_dir,
            'max_concurrent_renders': state.config.max_concurrent_renders,
            'enable_caching': state.config.enable_caching,
            'default_quality': state.config.default_quality,
            'use_gpu_acceleration': state.config.use_gpu_acceleration,
            'preview_mode': state.config.preview_mode,
            'use_visual_fix_code': state.config.use_visual_analysis,
            'max_retries': state.config.max_retries,
            'print_response': False
        }


class ErrorHandlerAgentTestRunner(BaseAgentTestRunner):
    """Test runner for ErrorHandlerAgent with error scenario inputs."""
    
    def __init__(self, config: WorkflowConfig):
        """Initialize ErrorHandlerAgent test runner."""
        super().__init__(config)
        self.error_handler = None
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate inputs for error handler agent testing."""
        issues = []
        
        if 'error_scenarios' not in inputs:
            issues.append("Missing required input: error_scenarios")
        elif not inputs['error_scenarios']:
            issues.append("error_scenarios cannot be empty")
        
        # Validate error scenarios structure
        if 'error_scenarios' in inputs:
            scenarios = inputs['error_scenarios']
            if not isinstance(scenarios, list):
                issues.append("error_scenarios must be a list")
            else:
                for i, scenario in enumerate(scenarios):
                    if not isinstance(scenario, dict):
                        issues.append(f"Error scenario {i} must be a dictionary")
                        continue
                    
                    required_fields = ['error_type', 'message', 'step']
                    for field in required_fields:
                        if field not in scenario:
                            issues.append(f"Error scenario {i} missing required field: {field}")
                    
                    # Validate error type
                    if 'error_type' in scenario:
                        valid_types = ['VALIDATION', 'MODEL', 'TIMEOUT', 'RATE_LIMIT', 'CONTENT', 'SYSTEM', 'TRANSIENT']
                        if scenario['error_type'] not in valid_types:
                            issues.append(f"Invalid error_type in scenario {i}: {scenario['error_type']}")
        
        return len(issues) == 0, issues
    
    async def run_test(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run ErrorHandlerAgent test with error scenario inputs."""
        start_time = time.time()
        
        # Validate inputs
        is_valid, validation_issues = self.validate_inputs(inputs)
        if not is_valid:
            return {
                'success': False,
                'error': f"Input validation failed: {'; '.join(validation_issues)}",
                'execution_time': time.time() - start_time,
                'agent': 'ErrorHandlerAgent',
                'inputs': inputs
            }
        
        try:
            # Extract inputs
            error_scenarios = inputs['error_scenarios']
            session_id = inputs.get('session_id', f"test_{int(time.time())}")
            topic = inputs.get('topic', 'Test Topic')
            description = inputs.get('description', 'Test Description')
            
            # Create test state
            state = self.create_test_state(
                topic=topic,
                description=description,
                session_id=session_id
            )
            
            # Initialize error handler
            self.error_handler = ErrorHandler(self.config)
            
            # Start output capture
            self.output_capture.start_capture(f"errorhandler_{session_id}", "ErrorHandlerAgent")
            
            # Test error handling for each scenario
            recovery_results = {}
            
            for i, scenario in enumerate(error_scenarios):
                try:
                    self.logger.info(f"Testing error handling for scenario {i}: {scenario['error_type']}")
                    
                    # Create WorkflowError from scenario
                    error = WorkflowError(
                        step=scenario['step'],
                        error_type=ErrorType[scenario['error_type']],
                        message=scenario['message'],
                        severity=ErrorSeverity[scenario.get('severity', 'MEDIUM')],
                        scene_number=scenario.get('scene_number'),
                        operation=scenario.get('operation'),
                        context=scenario.get('context', {})
                    )
                    
                    # Test error recovery
                    recovery_result = await self.error_handler.handle_error(error, state)
                    
                    recovery_results[i] = {
                        'scenario': scenario,
                        'recovery_result': {
                            'success': recovery_result.success,
                            'strategy_used': recovery_result.strategy_used,
                            'attempts_made': recovery_result.attempts_made,
                            'time_taken': recovery_result.time_taken,
                            'escalated': recovery_result.escalated,
                            'escalation_reason': recovery_result.escalation_reason,
                            'recovery_data': recovery_result.recovery_data
                        }
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error handling failed for scenario {i}: {str(e)}")
                    recovery_results[i] = {
                        'scenario': scenario,
                        'error': str(e),
                        'error_type': type(e).__name__
                    }
            
            # Get recovery statistics
            recovery_stats = self.error_handler.get_recovery_statistics()
            
            # Stop output capture
            captured_output = self.output_capture.stop_capture(f"errorhandler_{session_id}")
            
            # Calculate success metrics
            successful_recoveries = sum(1 for result in recovery_results.values() 
                                      if 'recovery_result' in result and result['recovery_result']['success'])
            total_scenarios = len(error_scenarios)
            
            # Build result
            result = {
                'success': True,  # Test runner success, not recovery success
                'agent': 'ErrorHandlerAgent',
                'execution_time': time.time() - start_time,
                'inputs': inputs,
                'outputs': {
                    'recovery_results': recovery_results,
                    'successful_recoveries': successful_recoveries,
                    'total_scenarios': total_scenarios,
                    'success_rate': successful_recoveries / total_scenarios if total_scenarios > 0 else 0.0
                },
                'recovery_statistics': recovery_stats,
                'captured_output': captured_output
            }
            
            self.logger.info(f"ErrorHandlerAgent test completed in {result['execution_time']:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"ErrorHandlerAgent test failed: {str(e)}")
            
            # Stop output capture on error
            captured_output = self.output_capture.stop_capture(f"errorhandler_{inputs.get('session_id', 'unknown')}")
            
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'execution_time': time.time() - start_time,
                'agent': 'ErrorHandlerAgent',
                'inputs': inputs,
                'captured_output': captured_output
            }


class HumanLoopAgentTestRunner(BaseAgentTestRunner):
    """Test runner for HumanLoopAgent with intervention scenario inputs."""
    
    def __init__(self, config: WorkflowConfig):
        """Initialize HumanLoopAgent test runner."""
        super().__init__(config)
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate inputs for human loop agent testing."""
        issues = []
        
        if 'intervention_scenarios' not in inputs:
            issues.append("Missing required input: intervention_scenarios")
        elif not inputs['intervention_scenarios']:
            issues.append("intervention_scenarios cannot be empty")
        
        # Validate intervention scenarios structure
        if 'intervention_scenarios' in inputs:
            scenarios = inputs['intervention_scenarios']
            if not isinstance(scenarios, list):
                issues.append("intervention_scenarios must be a list")
            else:
                for i, scenario in enumerate(scenarios):
                    if not isinstance(scenario, dict):
                        issues.append(f"Intervention scenario {i} must be a dictionary")
                        continue
                    
                    required_fields = ['intervention_type', 'trigger_condition', 'expected_action']
                    for field in required_fields:
                        if field not in scenario:
                            issues.append(f"Intervention scenario {i} missing required field: {field}")
                    
                    # Validate intervention type
                    if 'intervention_type' in scenario:
                        valid_types = ['error_escalation', 'quality_review', 'manual_override', 'approval_required']
                        if scenario['intervention_type'] not in valid_types:
                            issues.append(f"Invalid intervention_type in scenario {i}: {scenario['intervention_type']}")
        
        return len(issues) == 0, issues
    
    async def run_test(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run HumanLoopAgent test with intervention scenario inputs."""
        start_time = time.time()
        
        # Validate inputs
        is_valid, validation_issues = self.validate_inputs(inputs)
        if not is_valid:
            return {
                'success': False,
                'error': f"Input validation failed: {'; '.join(validation_issues)}",
                'execution_time': time.time() - start_time,
                'agent': 'HumanLoopAgent',
                'inputs': inputs
            }
        
        try:
            # Extract inputs
            intervention_scenarios = inputs['intervention_scenarios']
            session_id = inputs.get('session_id', f"test_{int(time.time())}")
            topic = inputs.get('topic', 'Test Topic')
            description = inputs.get('description', 'Test Description')
            
            # Create test state
            state = self.create_test_state(
                topic=topic,
                description=description,
                session_id=session_id
            )
            
            # Start output capture
            self.output_capture.start_capture(f"humanloop_{session_id}", "HumanLoopAgent")
            
            # Test human loop scenarios
            intervention_results = {}
            
            for i, scenario in enumerate(intervention_scenarios):
                try:
                    self.logger.info(f"Testing human loop scenario {i}: {scenario['intervention_type']}")
                    
                    # Simulate intervention trigger
                    intervention_result = await self._simulate_intervention(scenario, state)
                    
                    intervention_results[i] = {
                        'scenario': scenario,
                        'intervention_result': intervention_result
                    }
                    
                except Exception as e:
                    self.logger.error(f"Human loop test failed for scenario {i}: {str(e)}")
                    intervention_results[i] = {
                        'scenario': scenario,
                        'error': str(e),
                        'error_type': type(e).__name__
                    }
            
            # Stop output capture
            captured_output = self.output_capture.stop_capture(f"humanloop_{session_id}")
            
            # Calculate success metrics
            successful_interventions = sum(1 for result in intervention_results.values() 
                                         if 'intervention_result' in result and result['intervention_result']['success'])
            total_scenarios = len(intervention_scenarios)
            
            # Build result
            result = {
                'success': True,  # Test runner success
                'agent': 'HumanLoopAgent',
                'execution_time': time.time() - start_time,
                'inputs': inputs,
                'outputs': {
                    'intervention_results': intervention_results,
                    'successful_interventions': successful_interventions,
                    'total_scenarios': total_scenarios,
                    'success_rate': successful_interventions / total_scenarios if total_scenarios > 0 else 0.0
                },
                'captured_output': captured_output
            }
            
            self.logger.info(f"HumanLoopAgent test completed in {result['execution_time']:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"HumanLoopAgent test failed: {str(e)}")
            
            # Stop output capture on error
            captured_output = self.output_capture.stop_capture(f"humanloop_{inputs.get('session_id', 'unknown')}")
            
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'execution_time': time.time() - start_time,
                'agent': 'HumanLoopAgent',
                'inputs': inputs,
                'captured_output': captured_output
            }
    
    async def _simulate_intervention(self, scenario: Dict[str, Any], state: VideoGenerationState) -> Dict[str, Any]:
        """Simulate a human intervention scenario."""
        intervention_type = scenario['intervention_type']
        trigger_condition = scenario['trigger_condition']
        expected_action = scenario['expected_action']
        
        # Simulate different intervention types
        if intervention_type == 'error_escalation':
            # Simulate error escalation scenario
            return await self._simulate_error_escalation(scenario, state)
        
        elif intervention_type == 'quality_review':
            # Simulate quality review scenario
            return await self._simulate_quality_review(scenario, state)
        
        elif intervention_type == 'manual_override':
            # Simulate manual override scenario
            return await self._simulate_manual_override(scenario, state)
        
        elif intervention_type == 'approval_required':
            # Simulate approval required scenario
            return await self._simulate_approval_required(scenario, state)
        
        else:
            raise ValueError(f"Unknown intervention type: {intervention_type}")
    
    async def _simulate_error_escalation(self, scenario: Dict[str, Any], state: VideoGenerationState) -> Dict[str, Any]:
        """Simulate error escalation intervention."""
        # Create a mock escalated error
        escalated_error = {
            'error_type': scenario.get('error_type', 'SYSTEM'),
            'message': scenario.get('error_message', 'Simulated escalated error'),
            'step': scenario.get('step', 'code_generation'),
            'escalation_reason': scenario.get('escalation_reason', 'Max retries exceeded'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to state
        state.escalated_errors.append(escalated_error)
        
        # Simulate human intervention response
        intervention_response = {
            'action_taken': scenario['expected_action'],
            'resolution': scenario.get('resolution', 'Error acknowledged and logged'),
            'continue_workflow': scenario.get('continue_workflow', False),
            'timestamp': datetime.now().isoformat()
        }
        
        return {
            'success': True,
            'intervention_type': 'error_escalation',
            'escalated_error': escalated_error,
            'intervention_response': intervention_response,
            'processing_time': 0.1  # Simulated processing time
        }
    
    async def _simulate_quality_review(self, scenario: Dict[str, Any], state: VideoGenerationState) -> Dict[str, Any]:
        """Simulate quality review intervention."""
        # Simulate quality review request
        quality_review_request = {
            'review_type': scenario.get('review_type', 'content_quality'),
            'content_to_review': scenario.get('content_to_review', 'Generated scene outline'),
            'quality_threshold': scenario.get('quality_threshold', 0.8),
            'timestamp': datetime.now().isoformat()
        }
        
        # Simulate human review response
        review_response = {
            'approved': scenario.get('approved', True),
            'quality_score': scenario.get('quality_score', 0.85),
            'feedback': scenario.get('feedback', 'Content meets quality standards'),
            'action_taken': scenario['expected_action'],
            'timestamp': datetime.now().isoformat()
        }
        
        return {
            'success': True,
            'intervention_type': 'quality_review',
            'review_request': quality_review_request,
            'review_response': review_response,
            'processing_time': 0.2  # Simulated processing time
        }
    
    async def _simulate_manual_override(self, scenario: Dict[str, Any], state: VideoGenerationState) -> Dict[str, Any]:
        """Simulate manual override intervention."""
        # Simulate manual override request
        override_request = {
            'override_type': scenario.get('override_type', 'parameter_adjustment'),
            'original_value': scenario.get('original_value', 'auto'),
            'override_value': scenario.get('override_value', 'manual_setting'),
            'reason': scenario.get('reason', 'User preference'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Apply override to state (simulated)
        if 'pending_human_input' not in state.__dict__ or state.pending_human_input is None:
            state.pending_human_input = {}
        
        state.pending_human_input['manual_override'] = override_request
        
        # Simulate override response
        override_response = {
            'override_applied': True,
            'action_taken': scenario['expected_action'],
            'new_configuration': scenario.get('new_configuration', {}),
            'timestamp': datetime.now().isoformat()
        }
        
        return {
            'success': True,
            'intervention_type': 'manual_override',
            'override_request': override_request,
            'override_response': override_response,
            'processing_time': 0.15  # Simulated processing time
        }
    
    async def _simulate_approval_required(self, scenario: Dict[str, Any], state: VideoGenerationState) -> Dict[str, Any]:
        """Simulate approval required intervention."""
        # Simulate approval request
        approval_request = {
            'approval_type': scenario.get('approval_type', 'workflow_continuation'),
            'item_requiring_approval': scenario.get('item_requiring_approval', 'Generated content'),
            'approval_criteria': scenario.get('approval_criteria', 'Content appropriateness'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Simulate approval response
        approval_response = {
            'approved': scenario.get('approved', True),
            'approval_reason': scenario.get('approval_reason', 'Content meets criteria'),
            'action_taken': scenario['expected_action'],
            'conditions': scenario.get('conditions', []),
            'timestamp': datetime.now().isoformat()
        }
        
        return {
            'success': True,
            'intervention_type': 'approval_required',
            'approval_request': approval_request,
            'approval_response': approval_response,
            'processing_time': 0.25  # Simulated processing time
        }
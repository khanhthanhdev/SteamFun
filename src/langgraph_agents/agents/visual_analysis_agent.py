"""
VisualAnalysisAgent for error detection and visual quality analysis.
Ports existing detect_visual_errors and enhanced_visual_self_reflection methods to LangGraph agent.
"""

import re
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional, Tuple, List
from PIL import Image
from langgraph.types import Command

from ..base_agent import BaseAgent
from ..state import VideoGenerationState
from mllm_tools.gemini import GeminiWrapper
from mllm_tools.vertex_ai import VertexAIWrapper

logger = logging.getLogger(__name__)

# Cache file encoding constant
CACHE_FILE_ENCODING = 'utf-8'


class VisualAnalysisAgent(BaseAgent):
    """Agent for visual error detection and analysis.
    
    Ports existing visual analysis functionality from CodeGenerator while
    maintaining compatibility with current visual analysis workflow and
    _parse_visual_analysis logic.
    """
    
    async def execute(self, state: VideoGenerationState) -> Command:
        """Execute visual analysis on rendered videos.
        
        Args:
            state: Current workflow state
            
        Returns:
            Command: Next action based on visual analysis results
        """
        self.log_agent_action("starting_visual_analysis", {
            "rendered_videos": list(state.get('rendered_videos', {}).keys()),
            "use_visual_fix_code": state.get('use_visual_fix_code', False)
        })
        
        try:
            # Get scene model for visual analysis
            scene_model = self._get_scene_model(state)
            
            # Analyze all rendered videos
            visual_analysis_results = {}
            visual_errors = {}
            updated_code = {}
            
            rendered_videos = state.get('rendered_videos', {})
            generated_code = state.get('generated_code', {})
            
            for scene_number, video_path in rendered_videos.items():
                if not video_path or not Path(video_path).exists():
                    logger.warning(f"Video file not found for scene {scene_number}: {video_path}")
                    continue
                
                # Perform visual error detection
                analysis_result = await self._detect_visual_errors(
                    media_path=video_path,
                    scene_number=scene_number,
                    state=state,
                    scene_model=scene_model
                )
                
                visual_analysis_results[scene_number] = analysis_result
                
                # Extract visual errors for tracking
                errors = self._extract_visual_errors(analysis_result)
                if errors:
                    visual_errors[scene_number] = errors
                    
                    # If visual fix code is enabled and we have code for this scene
                    if state.get('use_visual_fix_code', False) and scene_number in generated_code:
                        try:
                            # Perform enhanced visual self-reflection to fix code
                            fixed_code, reflection_response = await self._enhanced_visual_self_reflection(
                                code=generated_code[scene_number],
                                media_path=video_path,
                                scene_number=scene_number,
                                state=state,
                                scene_model=scene_model
                            )
                            
                            # Only update if code was actually fixed
                            if fixed_code != generated_code[scene_number]:
                                updated_code[scene_number] = fixed_code
                                self.log_agent_action("visual_code_fix_applied", {
                                    "scene_number": scene_number,
                                    "errors_detected": len(errors)
                                })
                        
                        except Exception as e:
                            logger.error(f"Error in visual self-reflection for scene {scene_number}: {e}")
                            # Continue with analysis even if fix fails
            
            # Determine next action based on results
            total_errors = sum(len(errors) for errors in visual_errors.values())
            
            if total_errors > 0:
                self.log_agent_action("visual_errors_detected", {
                    "total_errors": total_errors,
                    "scenes_with_errors": list(visual_errors.keys()),
                    "fixes_applied": len(updated_code)
                })
                
                # If we have code fixes, route back to code generator for re-rendering
                if updated_code:
                    return Command(
                        goto="code_generator_agent",
                        update={
                            "visual_analysis_results": visual_analysis_results,
                            "visual_errors": visual_errors,
                            "generated_code": {**generated_code, **updated_code},
                            "current_agent": "code_generator_agent",
                            "next_agent": "renderer_agent"
                        }
                    )
                else:
                    # Request human intervention for quality review
                    from ..interfaces.human_intervention_interface import (
                        HumanInterventionInterface,
                        QualityReviewRequest,
                        InterventionPriority
                    )
                    
                    interface = self.get_human_intervention_interface()
                    
                    # Calculate quality metrics
                    total_scenes = len(state.get('rendered_videos', {}))
                    scenes_with_errors = len(visual_errors)
                    error_rate = scenes_with_errors / max(1, total_scenes)
                    
                    quality_metrics = {
                        "total_scenes": total_scenes,
                        "scenes_with_errors": scenes_with_errors,
                        "error_rate": error_rate,
                        "total_errors": total_errors,
                        "visual_quality_score": max(0, 1 - error_rate),
                        "critical_errors": sum(len(result.get('critical_errors', [])) for result in visual_analysis_results.values()),
                        "major_errors": sum(len(result.get('major_errors', [])) for result in visual_analysis_results.values()),
                        "minor_errors": sum(len(result.get('minor_errors', [])) for result in visual_analysis_results.values())
                    }
                    
                    quality_request = QualityReviewRequest(
                        content_description=f"Visual analysis detected {total_errors} issues across {scenes_with_errors} scenes",
                        quality_metrics=quality_metrics,
                        review_criteria=[
                            "Element overlap detection",
                            "Spatial boundary compliance", 
                            "Visual composition quality",
                            "Educational effectiveness"
                        ],
                        improvement_suggestions=[
                            "Accept current quality if errors are minor",
                            "Regenerate scenes with critical errors",
                            "Adjust visual parameters for better spacing",
                            "Manual review and correction of problematic scenes"
                        ],
                        timeout_seconds=600
                    )
                    
                    return await interface.request_quality_review(quality_request, state)
            else:
                # No visual errors detected, workflow can continue
                self.log_agent_action("visual_analysis_passed", {
                    "scenes_analyzed": len(rendered_videos)
                })
                
                return Command(
                    goto="END",
                    update={
                        "visual_analysis_results": visual_analysis_results,
                        "visual_errors": visual_errors,
                        "workflow_complete": True,
                        "current_agent": None
                    }
                )
                
        except Exception as e:
            logger.error(f"Error in visual analysis: {e}")
            return await self.handle_error(e, state)
    
    async def _detect_visual_errors(self,
                                  media_path: Union[str, Image.Image],
                                  scene_number: int,
                                  state: VideoGenerationState,
                                  scene_model) -> Dict[str, Any]:
        """Detect visual errors using VLM without code modification.
        
        Ports the detect_visual_errors method from CodeGenerator.
        
        Args:
            media_path: Path to media file or PIL Image
            scene_number: Scene number being analyzed
            state: Current workflow state
            scene_model: Model wrapper for visual analysis
            
        Returns:
            Dictionary containing visual error analysis results
        """
        try:
            # Validate media input
            if isinstance(media_path, str):
                media_file = Path(media_path)
                if not media_file.exists():
                    raise FileNotFoundError(f"Media file not found: {media_path}")
            
            # Create analysis prompt
            analysis_prompt = """
            You are an expert visual quality analyst. Analyze this Manim-generated frame/video for:
            
            1. **Element Overlap Detection:**
               - Text overlapping with shapes or other text
               - Mathematical expressions colliding
               - Unintentional object occlusion
            
            2. **Spatial Boundary Issues:**
               - Objects extending beyond frame boundaries
               - Violations of safe area margins (0.5 units from edges)
               - Insufficient spacing between elements (minimum 0.3 units)
            
            3. **Visual Quality Assessment:**
               - Overall composition balance
               - Readability of text elements
               - Educational effectiveness of arrangement
            
            Provide your analysis in the following format:
            
            **VISUAL ERROR ANALYSIS:**
            - Overlap Issues: [List any overlapping elements]
            - Boundary Violations: [List out-of-bounds elements]
            - Spacing Problems: [List spacing violations]
            - Quality Issues: [List other visual problems]
            
            **SEVERITY ASSESSMENT:**
            - Critical Errors: [Issues that severely impact readability]
            - Major Errors: [Issues that noticeably reduce quality]
            - Minor Errors: [Issues that slightly affect visual appeal]
            
            **OVERALL RATING:** [Excellent/Good/Fair/Poor]
            """
            
            # Determine media type and prepare input
            is_video = isinstance(media_path, str) and media_path.lower().endswith('.mp4')
            
            if is_video and isinstance(scene_model, (GeminiWrapper, VertexAIWrapper)):
                messages = [
                    {"type": "text", "content": analysis_prompt},
                    {"type": "video", "content": str(media_path)}
                ]
            else:
                if isinstance(media_path, str):
                    media = Image.open(media_path)
                else:
                    media = media_path
                messages = [
                    {"type": "text", "content": analysis_prompt},
                    {"type": "image", "content": media}
                ]
            
            # Get analysis response
            response_text = scene_model(
                messages,
                metadata={
                    "generation_name": "visual_error_detection",
                    "trace_id": state.get('session_id', ''),
                    "tags": [state.get('topic', 'unknown'), f"scene{scene_number}", "quality_analysis"],
                    "session_id": state.get('session_id', '')
                }
            )
            
            # Parse response into structured results
            analysis_results = self._parse_visual_analysis(response_text)
            
            logger.info(f"Visual error detection completed for scene {scene_number}")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in visual error detection for scene {scene_number}: {e}")
            
            # Check if human intervention is needed for critical visual failures
            if self.should_escalate_to_human(state):
                from ..interfaces.human_intervention_interface import (
                    HumanInterventionInterface,
                    ErrorResolutionRequest,
                    InterventionPriority
                )
                
                interface = self.get_human_intervention_interface()
                
                error_request = ErrorResolutionRequest(
                    error_description=f"Visual error detection failed for scene {scene_number}: {str(e)}",
                    error_type="visual_analysis_failure",
                    error_context={
                        "scene_number": scene_number,
                        "media_path": str(media_path),
                        "error_message": str(e),
                        "agent": "visual_analysis_agent",
                        "retry_count": state.get('retry_count', {}).get('visual_analysis_agent', 0)
                    },
                    recovery_options=[
                        "Skip visual analysis for this scene",
                        "Retry with different analysis parameters", 
                        "Mark scene as requiring manual review",
                        "Use fallback quality assessment"
                    ],
                    impact_assessment="Medium - Scene quality cannot be automatically validated",
                    suggested_action="Skip visual analysis for this scene",
                    priority=InterventionPriority.MEDIUM,
                    timeout_seconds=300
                )
                
                # This would raise an exception to be caught by the outer try-catch
                # and handled by the main error handling flow
                raise RuntimeError(
                    f"Visual error detection failed and human intervention requested: {e}"
                ) from e
            
            raise ValueError(f"Visual error detection failed: {e}") from e
    
    async def _enhanced_visual_self_reflection(self,
                                             code: str,
                                             media_path: Union[str, Image.Image],
                                             scene_number: int,
                                             state: VideoGenerationState,
                                             scene_model,
                                             implementation_plan: Optional[str] = None) -> Tuple[str, str]:
        """Enhanced visual self-reflection using VLM for detailed error detection.
        
        Ports the enhanced_visual_self_reflection method from CodeGenerator.
        
        Args:
            code: Code to analyze and fix
            media_path: Path to media file or PIL Image
            scene_number: Scene number
            state: Current workflow state
            scene_model: Model wrapper for visual analysis
            implementation_plan: Optional implementation plan for context
            
        Returns:
            Tuple of fixed code and response text
        """
        try:
            # Validate media input
            if isinstance(media_path, str):
                media_file = Path(media_path)
                if not media_file.exists():
                    raise FileNotFoundError(f"Media file not found: {media_path}")
            
            # Determine if we're dealing with video or image
            is_video = isinstance(media_path, str) and media_path.lower().endswith('.mp4')
            
            # Load enhanced visual analysis prompt
            enhanced_prompt_file = Path('task_generator/prompts_raw/prompt_enhanced_visual_self_reflection.txt')
            if enhanced_prompt_file.exists():
                with enhanced_prompt_file.open('r', encoding=CACHE_FILE_ENCODING) as f:
                    prompt_template = f.read()
            else:
                # Fallback to original prompt if enhanced version not found
                logger.warning("Enhanced visual self-reflection prompt not found, using fallback")
                prompt_template = self._get_fallback_visual_prompt()
            
            # Format prompt with implementation plan and code
            prompt = prompt_template.format(
                implementation=implementation_plan or "No implementation plan provided",
                code=code
            )
            
            # Prepare input based on media type and model capabilities
            if is_video and isinstance(scene_model, (GeminiWrapper, VertexAIWrapper)):
                # For video with Gemini/Vertex AI models
                messages = [
                    {"type": "text", "content": prompt},
                    {"type": "video", "content": str(media_path)}
                ]
            else:
                # For images or non-Gemini models
                if isinstance(media_path, str):
                    media = Image.open(media_path)
                else:
                    media = media_path
                messages = [
                    {"type": "text", "content": prompt},
                    {"type": "image", "content": media}
                ]
            
            # Get enhanced VLM analysis response
            response_text = scene_model(
                messages,
                metadata={
                    "generation_name": "enhanced_visual_self_reflection",
                    "trace_id": state.get('session_id', ''),
                    "tags": [state.get('topic', 'unknown'), f"scene{scene_number}", "visual_error_detection"],
                    "session_id": state.get('session_id', '')
                }
            )
            
            # Parse response for visual analysis results
            if "<LGTM>" in response_text or response_text.strip() == "<LGTM>":
                logger.info(f"Enhanced visual analysis passed for scene {scene_number}")
                return code, response_text
            
            # Extract improved code if visual issues were found
            fixed_code = self._extract_visual_fix_code(response_text, state)
            
            logger.info(f"Enhanced visual self-reflection completed with fixes for scene {scene_number}")
            return fixed_code, response_text
            
        except Exception as e:
            logger.error(f"Error in enhanced visual self-reflection for scene {scene_number}: {e}")
            # Fallback to original code if enhanced version fails
            logger.info("Enhanced visual self-reflection failed, returning original code")
            return code, f"Enhanced visual self-reflection failed: {e}"
    
    def _parse_visual_analysis(self, response_text: str) -> Dict[str, Any]:
        """Parse visual analysis response into structured data.
        
        Maintains compatibility with current _parse_visual_analysis logic.
        
        Args:
            response_text: Raw response from VLM
            
        Returns:
            Structured analysis results
        """
        results = {
            "overlap_issues": [],
            "boundary_violations": [],
            "spacing_problems": [],
            "quality_issues": [],
            "critical_errors": [],
            "major_errors": [],
            "minor_errors": [],
            "overall_rating": "Unknown",
            "raw_analysis": response_text
        }
        
        try:
            # Extract different sections using regex patterns
            overlap_match = re.search(r'Overlap Issues:\s*(.*?)(?=\n-|\n\*\*|$)', response_text, re.DOTALL)
            if overlap_match:
                results["overlap_issues"] = [item.strip() for item in overlap_match.group(1).split('\n') if item.strip()]
            
            boundary_match = re.search(r'Boundary Violations:\s*(.*?)(?=\n-|\n\*\*|$)', response_text, re.DOTALL)
            if boundary_match:
                results["boundary_violations"] = [item.strip() for item in boundary_match.group(1).split('\n') if item.strip()]
            
            spacing_match = re.search(r'Spacing Problems:\s*(.*?)(?=\n-|\n\*\*|$)', response_text, re.DOTALL)
            if spacing_match:
                results["spacing_problems"] = [item.strip() for item in spacing_match.group(1).split('\n') if item.strip()]
            
            quality_match = re.search(r'Quality Issues:\s*(.*?)(?=\n-|\n\*\*|$)', response_text, re.DOTALL)
            if quality_match:
                results["quality_issues"] = [item.strip() for item in quality_match.group(1).split('\n') if item.strip()]
            
            critical_match = re.search(r'Critical Errors:\s*(.*?)(?=\n-|\n\*\*|$)', response_text, re.DOTALL)
            if critical_match:
                results["critical_errors"] = [item.strip() for item in critical_match.group(1).split('\n') if item.strip()]
            
            major_match = re.search(r'Major Errors:\s*(.*?)(?=\n-|\n\*\*|$)', response_text, re.DOTALL)
            if major_match:
                results["major_errors"] = [item.strip() for item in major_match.group(1).split('\n') if item.strip()]
            
            minor_match = re.search(r'Minor Errors:\s*(.*?)(?=\n-|\n\*\*|$)', response_text, re.DOTALL)
            if minor_match:
                results["minor_errors"] = [item.strip() for item in minor_match.group(1).split('\n') if item.strip()]
            
            rating_match = re.search(r'OVERALL RATING.*?:\s*([A-Za-z]+)', response_text)
            if rating_match:
                results["overall_rating"] = rating_match.group(1)
            
        except Exception as e:
            logger.warning(f"Error parsing visual analysis: {e}")
        
        return results
    
    def _extract_visual_errors(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Extract visual errors from analysis result for tracking.
        
        Args:
            analysis_result: Structured analysis results
            
        Returns:
            List of visual error descriptions
        """
        errors = []
        
        # Collect all types of errors
        errors.extend(analysis_result.get("overlap_issues", []))
        errors.extend(analysis_result.get("boundary_violations", []))
        errors.extend(analysis_result.get("spacing_problems", []))
        errors.extend(analysis_result.get("quality_issues", []))
        errors.extend(analysis_result.get("critical_errors", []))
        errors.extend(analysis_result.get("major_errors", []))
        errors.extend(analysis_result.get("minor_errors", []))
        
        # Filter out empty strings and "None" entries
        errors = [error for error in errors if error and error.lower() not in ['none', 'no issues', 'no errors']]
        
        return errors
    
    def _extract_visual_fix_code(self, response_text: str, state: VideoGenerationState) -> str:
        """Extract code from enhanced visual analysis response.
        
        Args:
            response_text: The VLM response containing visual analysis
            state: Current workflow state
            
        Returns:
            The extracted and fixed code
        """
        # Try to extract code from <improved_code> tags first
        improved_code_pattern = r'<improved_code>\s*```python\s*(.*?)\s*```\s*</improved_code>'
        code_match = re.search(improved_code_pattern, response_text, re.DOTALL)
        
        if code_match:
            extracted_code = code_match.group(1).strip()
            logger.debug("Successfully extracted code from <improved_code> tags")
            return extracted_code
        
        # Fallback to standard code extraction patterns
        code_patterns = [
            r'```python\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'<code>\s*(.*?)\s*</code>'
        ]
        
        for pattern in code_patterns:
            code_match = re.search(pattern, response_text, re.DOTALL)
            if code_match:
                extracted_code = code_match.group(1).strip()
                logger.debug(f"Successfully extracted code using pattern: {pattern}")
                return extracted_code
        
        # If no code blocks found, return original response (might contain inline fixes)
        logger.warning("No code blocks found in visual fix response")
        return response_text.strip()
    
    def _get_fallback_visual_prompt(self) -> str:
        """Get fallback visual analysis prompt if enhanced version is not available.
        
        Preserves current visual fix code patterns and banned reasoning logic.
        """
        return """
        Analyze the visual output and the provided code for the following issues:
        
        1. **Element Overlap:** Check for overlapping text, shapes, or mathematical expressions
        2. **Out-of-Bounds Objects:** Identify elements outside the visible frame
        3. **Spacing Issues:** Verify minimum 0.3 unit spacing between elements
        4. **Safe Area Compliance:** Ensure 0.5 unit margins from frame edges
        5. **Educational Clarity:** Assess if arrangement supports learning objectives
        
        Implementation Plan: {implementation}
        
        Code to analyze:
        {code}
        
        If issues are found, provide fixed code. If no issues, return "<LGTM>".
        
        <improved_code>
        ```python
        [Fixed code here]
        ```
        </improved_code>
        """
    
    def _get_scene_model(self, state: VideoGenerationState):
        """Get scene model for visual analysis.
        
        Args:
            state: Current workflow state
            
        Returns:
            Model wrapper for visual analysis
        """
        # Use scene model from config if available
        if hasattr(self.config, 'scene_model') and self.config.scene_model:
            return self.get_model_wrapper(self.config.scene_model, state)
        
        # Fallback to default model configuration
        model_config = self.config.model_config
        if 'scene_model' in model_config:
            return self.get_model_wrapper(model_config['scene_model'], state)
        
        # Final fallback to any available model
        if 'default_model' in model_config:
            return self.get_model_wrapper(model_config['default_model'], state)
        
        raise ValueError("No scene model configured for visual analysis")
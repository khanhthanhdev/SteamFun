"""
Unit tests for the routing module.

Tests all routing functions with various state conditions to ensure
correct routing decisions are made based on workflow state.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from src.langgraph_agents.routing import (
    route_from_planning,
    route_from_code_generation,
    route_from_rendering,
    route_from_error_handler,
    validate_routing_decision,
    get_routing_summary
)
from src.langgraph_agents.models.state import VideoGenerationState
from src.langgraph_agents.models.config import WorkflowConfig
from src.langgraph_agents.models.errors import WorkflowError


class TestRouteFromPlanning:
    """Test routing from planning step."""
    
    @pytest.fixture
    def base_state(self):
        """Create a base state for testing."""
        return VideoGenerationState(
            topic="Test topic",
            description="Test description",
            session_id="test-session",
            config=WorkflowConfig(max_retries=3)
        )
    
    def test_successful_planning_routes_to_code_generation(self, base_state):
        """Test that successful planning routes to code generation."""
        base_state.scene_outline = "Scene 1: Introduction\nScene 2: Main content"
        base_state.scene_implementations = {
            1: "Implementation for scene 1",
            2: "Implementation for scene 2"
        }
        
        result = route_from_planning(base_state)
        
        assert result == "code_generation"
        assert len(base_state.execution_trace) > 0
        assert base_state.execution_trace[-1]["step"] == "routing_from_planning"
    
    def test_missing_scene_outline_routes_to_error_handler(self, base_state):
        """Test that missing scene outline routes to error handler."""
        base_state.scene_outline = None
        
        result = route_from_planning(base_state)
        
        assert result == "error_handler"
        assert len(base_state.errors) > 0
        assert base_state.errors[-1].error_type == "missing_output"
    
    def test_empty_scene_outline_routes_to_error_handler(self, base_state):
        """Test that empty scene outline routes to error handler."""
        base_state.scene_outline = "   "  # Whitespace only
        
        result = route_from_planning(base_state)
        
        assert result == "error_handler"
        assert len(base_state.errors) > 0
    
    def test_missing_scene_implementations_routes_to_error_handler(self, base_state):
        """Test that missing scene implementations routes to error handler."""
        base_state.scene_outline = "Scene 1: Test"
        base_state.scene_implementations = {}
        
        result = route_from_planning(base_state)
        
        assert result == "error_handler"
        assert len(base_state.errors) > 0
        assert base_state.errors[-1].error_type == "missing_output"
    
    def test_workflow_interrupted_routes_to_human_loop(self, base_state):
        """Test that interrupted workflow routes to human loop."""
        base_state.workflow_interrupted = True
        base_state.scene_outline = "Scene 1: Test"
        base_state.scene_implementations = {1: "Implementation"}
        
        result = route_from_planning(base_state)
        
        assert result == "human_loop"
    
    def test_pending_human_input_routes_to_human_loop(self, base_state):
        """Test that pending human input routes to human loop."""
        base_state.pending_human_input = {"type": "clarification", "message": "Need help"}
        base_state.scene_outline = "Scene 1: Test"
        base_state.scene_implementations = {1: "Implementation"}
        
        result = route_from_planning(base_state)
        
        assert result == "human_loop"
    
    def test_max_retries_exceeded_routes_to_human_loop(self, base_state):
        """Test that max retries exceeded routes to human loop."""
        base_state.retry_counts["planning"] = 3  # Equals max_retries
        base_state.scene_outline = "Scene 1: Test"
        base_state.scene_implementations = {1: "Implementation"}
        
        result = route_from_planning(base_state)
        
        assert result == "human_loop"
        assert len(base_state.errors) > 0
        assert base_state.errors[-1].error_type == "max_retries_exceeded"
    
    def test_recoverable_error_routes_to_error_handler(self, base_state):
        """Test that recoverable error routes to error handler."""
        error = WorkflowError(
            step="planning",
            error_type="api_timeout",
            message="API timeout occurred",
            recoverable=True,
            retry_count=0
        )
        base_state.errors.append(error)
        base_state.retry_counts["planning"] = 0
        
        result = route_from_planning(base_state)
        
        assert result == "error_handler"
    
    def test_non_recoverable_error_routes_to_human_loop(self, base_state):
        """Test that non-recoverable error routes to human loop."""
        error = WorkflowError(
            step="planning",
            error_type="invalid_api_key",
            message="Invalid API key",
            recoverable=False,
            retry_count=0
        )
        base_state.errors.append(error)
        base_state.retry_counts["planning"] = 0
        
        result = route_from_planning(base_state)
        
        assert result == "human_loop"


class TestRouteFromCodeGeneration:
    """Test routing from code generation step."""
    
    @pytest.fixture
    def base_state(self):
        """Create a base state for testing."""
        state = VideoGenerationState(
            topic="Test topic",
            description="Test description",
            session_id="test-session",
            config=WorkflowConfig(max_retries=3, use_rag=True)
        )
        state.scene_implementations = {
            1: "Implementation 1",
            2: "Implementation 2",
            3: "Implementation 3"
        }
        return state
    
    def test_complete_success_routes_to_rendering(self, base_state):
        """Test that complete success routes to rendering."""
        base_state.generated_code = {
            1: "Code for scene 1",
            2: "Code for scene 2", 
            3: "Code for scene 3"
        }
        
        result = route_from_code_generation(base_state)
        
        assert result == "rendering"
    
    def test_partial_success_with_rag_routes_to_rag_enhancement(self, base_state):
        """Test that partial success with RAG enabled routes to RAG enhancement."""
        base_state.generated_code = {1: "Code for scene 1", 2: "Code for scene 2"}
        base_state.code_errors = {3: "Failed to generate code for scene 3"}
        
        result = route_from_code_generation(base_state)
        
        assert result == "rag_enhancement"
    
    def test_partial_success_without_rag_routes_to_error_handler(self, base_state):
        """Test that partial success without RAG routes to error handler."""
        base_state.config.use_rag = False
        base_state.generated_code = {1: "Code for scene 1", 2: "Code for scene 2"}
        base_state.code_errors = {3: "Failed to generate code for scene 3"}
        
        result = route_from_code_generation(base_state)
        
        assert result == "error_handler"
    
    def test_low_success_rate_routes_to_error_handler(self, base_state):
        """Test that low success rate routes to error handler."""
        base_state.generated_code = {1: "Code for scene 1"}  # Only 1/3 success
        base_state.code_errors = {2: "Error 2", 3: "Error 3"}
        
        result = route_from_code_generation(base_state)
        
        assert result == "error_handler"
    
    def test_complete_failure_routes_to_error_handler(self, base_state):
        """Test that complete failure routes to error handler."""
        base_state.generated_code = {}
        base_state.code_errors = {1: "Error 1", 2: "Error 2", 3: "Error 3"}
        
        result = route_from_code_generation(base_state)
        
        assert result == "error_handler"
        assert len(base_state.errors) > 0
        assert base_state.errors[-1].error_type == "complete_failure"
    
    def test_max_retries_exceeded_routes_to_human_loop(self, base_state):
        """Test that max retries exceeded routes to human loop."""
        base_state.retry_counts["code_generation"] = 3
        base_state.generated_code = {1: "Code for scene 1"}
        
        result = route_from_code_generation(base_state)
        
        assert result == "human_loop"
    
    def test_workflow_interrupted_routes_to_human_loop(self, base_state):
        """Test that interrupted workflow routes to human loop."""
        base_state.workflow_interrupted = True
        base_state.generated_code = {1: "Code", 2: "Code", 3: "Code"}
        
        result = route_from_code_generation(base_state)
        
        assert result == "human_loop"
    
    def test_no_scenes_to_generate_routes_to_error_handler(self, base_state):
        """Test that no scenes to generate routes to error handler."""
        base_state.scene_implementations = {}
        
        result = route_from_code_generation(base_state)
        
        assert result == "error_handler"


class TestRouteFromRendering:
    """Test routing from rendering step."""
    
    @pytest.fixture
    def base_state(self):
        """Create a base state for testing."""
        state = VideoGenerationState(
            topic="Test topic",
            description="Test description",
            session_id="test-session",
            config=WorkflowConfig(max_retries=3, use_visual_analysis=False)
        )
        state.generated_code = {
            1: "Code for scene 1",
            2: "Code for scene 2",
            3: "Code for scene 3"
        }
        return state
    
    def test_complete_success_without_visual_analysis_routes_to_complete(self, base_state):
        """Test that complete success without visual analysis routes to complete."""
        base_state.rendered_videos = {
            1: "video1.mp4",
            2: "video2.mp4",
            3: "video3.mp4"
        }
        base_state.combined_video_path = "combined.mp4"
        
        result = route_from_rendering(base_state)
        
        assert result == "complete"
    
    def test_complete_success_with_visual_analysis_routes_to_visual_analysis(self, base_state):
        """Test that complete success with visual analysis routes to visual analysis."""
        base_state.config.use_visual_analysis = True
        base_state.rendered_videos = {
            1: "video1.mp4",
            2: "video2.mp4",
            3: "video3.mp4"
        }
        base_state.combined_video_path = "combined.mp4"
        
        result = route_from_rendering(base_state)
        
        assert result == "visual_analysis"
    
    def test_missing_combined_video_routes_to_error_handler(self, base_state):
        """Test that missing combined video routes to error handler."""
        base_state.rendered_videos = {
            1: "video1.mp4",
            2: "video2.mp4",
            3: "video3.mp4"
        }
        base_state.combined_video_path = None
        
        result = route_from_rendering(base_state)
        
        assert result == "error_handler"
    
    def test_partial_success_above_threshold_routes_to_complete(self, base_state):
        """Test that partial success above threshold routes to complete."""
        base_state.rendered_videos = {1: "video1.mp4", 2: "video2.mp4"}  # 2/3 = 67% < 70%
        base_state.rendering_errors = {3: "Rendering failed"}
        base_state.combined_video_path = "combined.mp4"
        
        # Manually set to meet 70% threshold
        base_state.rendered_videos[3] = "video3.mp4"  # Now 3/3 = 100%
        del base_state.rendering_errors[3]
        
        result = route_from_rendering(base_state)
        
        assert result == "complete"
    
    def test_low_success_rate_routes_to_error_handler(self, base_state):
        """Test that low success rate routes to error handler."""
        base_state.rendered_videos = {1: "video1.mp4"}  # Only 1/3 = 33%
        base_state.rendering_errors = {2: "Error 2", 3: "Error 3"}
        
        result = route_from_rendering(base_state)
        
        assert result == "error_handler"
        assert len(base_state.errors) > 0
        assert base_state.errors[-1].error_type == "low_success_rate"
    
    def test_complete_failure_routes_to_error_handler(self, base_state):
        """Test that complete failure routes to error handler."""
        base_state.rendered_videos = {}
        base_state.rendering_errors = {1: "Error 1", 2: "Error 2", 3: "Error 3"}
        
        result = route_from_rendering(base_state)
        
        assert result == "error_handler"
        assert len(base_state.errors) > 0
        assert base_state.errors[-1].error_type == "complete_failure"
    
    def test_max_retries_exceeded_routes_to_human_loop(self, base_state):
        """Test that max retries exceeded routes to human loop."""
        base_state.retry_counts["rendering"] = 3
        base_state.rendered_videos = {1: "video1.mp4"}
        
        result = route_from_rendering(base_state)
        
        assert result == "human_loop"
    
    def test_workflow_interrupted_routes_to_human_loop(self, base_state):
        """Test that interrupted workflow routes to human loop."""
        base_state.workflow_interrupted = True
        base_state.rendered_videos = {1: "video1.mp4", 2: "video2.mp4", 3: "video3.mp4"}
        base_state.combined_video_path = "combined.mp4"
        
        result = route_from_rendering(base_state)
        
        assert result == "human_loop"
    
    def test_no_scenes_to_render_routes_to_error_handler(self, base_state):
        """Test that no scenes to render routes to error handler."""
        base_state.generated_code = {}
        
        result = route_from_rendering(base_state)
        
        assert result == "error_handler"


class TestRouteFromErrorHandler:
    """Test routing from error handler step."""
    
    @pytest.fixture
    def base_state(self):
        """Create a base state for testing."""
        return VideoGenerationState(
            topic="Test topic",
            description="Test description",
            session_id="test-session",
            config=WorkflowConfig(max_retries=3)
        )
    
    def test_missing_planning_routes_to_planning(self, base_state):
        """Test that missing planning data routes back to planning."""
        base_state.scene_outline = None
        base_state.retry_counts["planning"] = 1
        
        result = route_from_error_handler(base_state)
        
        assert result == "planning"
    
    def test_incomplete_code_generation_routes_to_code_generation(self, base_state):
        """Test that incomplete code generation routes back to code generation."""
        base_state.scene_outline = "Scene 1: Test"
        base_state.scene_implementations = {1: "Implementation", 2: "Implementation"}
        base_state.generated_code = {1: "Code"}  # Missing scene 2
        base_state.retry_counts["code_generation"] = 1
        
        result = route_from_error_handler(base_state)
        
        assert result == "code_generation"
    
    def test_incomplete_rendering_routes_to_rendering(self, base_state):
        """Test that incomplete rendering routes back to rendering."""
        base_state.scene_outline = "Scene 1: Test"
        base_state.scene_implementations = {1: "Implementation", 2: "Implementation"}
        base_state.generated_code = {1: "Code", 2: "Code"}
        base_state.rendered_videos = {1: "video1.mp4"}  # Missing scene 2
        base_state.retry_counts["rendering"] = 1
        
        result = route_from_error_handler(base_state)
        
        assert result == "rendering"
    
    def test_acceptable_completion_routes_to_complete(self, base_state):
        """Test that acceptable completion percentage routes to complete."""
        base_state.scene_outline = "Scene 1: Test"
        base_state.scene_implementations = {1: "Impl1", 2: "Impl2", 3: "Impl3", 4: "Impl4"}
        base_state.generated_code = {1: "Code1", 2: "Code2", 3: "Code3"}  # 75% complete
        base_state.rendered_videos = {1: "video1.mp4", 2: "video2.mp4", 3: "video3.mp4"}
        base_state.combined_video_path = "combined.mp4"
        
        result = route_from_error_handler(base_state)
        
        assert result == "complete"
    
    def test_max_retries_exceeded_routes_to_human_loop(self, base_state):
        """Test that max retries exceeded routes to human loop."""
        base_state.scene_outline = None
        base_state.retry_counts["planning"] = 3  # Max retries reached
        
        result = route_from_error_handler(base_state)
        
        assert result == "human_loop"
    
    def test_escalated_errors_routes_to_human_loop(self, base_state):
        """Test that escalated errors route to human loop."""
        base_state.escalated_errors = [{"error": "Critical system error"}]
        
        result = route_from_error_handler(base_state)
        
        assert result == "human_loop"
    
    def test_workflow_interrupted_routes_to_human_loop(self, base_state):
        """Test that interrupted workflow routes to human loop."""
        base_state.workflow_interrupted = True
        
        result = route_from_error_handler(base_state)
        
        assert result == "human_loop"
    
    def test_all_retries_exhausted_routes_to_human_loop(self, base_state):
        """Test that all retries exhausted routes to human loop."""
        base_state.retry_counts = {
            "planning": 3,
            "code_generation": 3,
            "rendering": 3
        }
        
        result = route_from_error_handler(base_state)
        
        assert result == "human_loop"


class TestValidateRoutingDecision:
    """Test routing decision validation."""
    
    @pytest.fixture
    def base_state(self):
        """Create a base state for testing."""
        return VideoGenerationState(
            topic="Test topic",
            description="Test description",
            session_id="test-session"
        )
    
    def test_valid_planning_to_code_generation(self, base_state):
        """Test valid transition from planning to code generation."""
        base_state.scene_implementations = {1: "Implementation"}
        
        result = validate_routing_decision("planning", "code_generation", base_state)
        
        assert result is True
    
    def test_invalid_planning_to_rendering(self, base_state):
        """Test invalid transition from planning to rendering."""
        result = validate_routing_decision("planning", "rendering", base_state)
        
        assert result is False
    
    def test_code_generation_without_implementations_invalid(self, base_state):
        """Test that routing to code generation without implementations is invalid."""
        base_state.scene_implementations = {}
        
        result = validate_routing_decision("planning", "code_generation", base_state)
        
        assert result is False
    
    def test_rendering_without_code_invalid(self, base_state):
        """Test that routing to rendering without code is invalid."""
        base_state.generated_code = {}
        
        result = validate_routing_decision("code_generation", "rendering", base_state)
        
        assert result is False
    
    def test_visual_analysis_without_videos_invalid(self, base_state):
        """Test that routing to visual analysis without videos is invalid."""
        base_state.rendered_videos = {}
        
        result = validate_routing_decision("rendering", "visual_analysis", base_state)
        
        assert result is False
    
    def test_complete_interrupted_workflow_invalid(self, base_state):
        """Test that completing interrupted workflow is invalid."""
        base_state.workflow_interrupted = True
        
        result = validate_routing_decision("rendering", "complete", base_state)
        
        assert result is False
    
    def test_invalid_from_step(self, base_state):
        """Test invalid from step."""
        result = validate_routing_decision("invalid_step", "planning", base_state)
        
        assert result is False
    
    def test_complete_state_no_transitions(self, base_state):
        """Test that complete state has no valid transitions."""
        result = validate_routing_decision("complete", "planning", base_state)
        
        assert result is False


class TestGetRoutingSummary:
    """Test routing summary generation."""
    
    @pytest.fixture
    def base_state(self):
        """Create a base state for testing."""
        state = VideoGenerationState(
            topic="Test topic",
            description="Test description",
            session_id="test-session",
            config=WorkflowConfig(max_retries=3, use_rag=True, use_visual_analysis=True)
        )
        state.scene_implementations = {1: "Impl1", 2: "Impl2"}
        state.generated_code = {1: "Code1"}
        state.rendered_videos = {1: "video1.mp4"}
        state.combined_video_path = "combined.mp4"
        state.retry_counts = {"planning": 1, "code_generation": 2}
        state.errors = [WorkflowError(step="test", error_type="test", message="test")]
        return state
    
    def test_routing_summary_contains_all_fields(self, base_state):
        """Test that routing summary contains all expected fields."""
        summary = get_routing_summary(base_state)
        
        expected_fields = [
            "current_step", "workflow_complete", "workflow_interrupted",
            "pending_human_input", "error_count", "escalated_errors",
            "retry_counts", "completion_percentage", "scene_counts",
            "has_combined_video", "config"
        ]
        
        for field in expected_fields:
            assert field in summary
    
    def test_routing_summary_values(self, base_state):
        """Test that routing summary contains correct values."""
        summary = get_routing_summary(base_state)
        
        assert summary["current_step"] == "planning"
        assert summary["workflow_complete"] is False
        assert summary["workflow_interrupted"] is False
        assert summary["pending_human_input"] is False
        assert summary["error_count"] == 1
        assert summary["escalated_errors"] == 0
        assert summary["retry_counts"] == {"planning": 1, "code_generation": 2}
        assert summary["completion_percentage"] == 50.0  # 1/2 scenes complete
        assert summary["scene_counts"]["implementations"] == 2
        assert summary["scene_counts"]["generated_code"] == 1
        assert summary["scene_counts"]["rendered_videos"] == 1
        assert summary["has_combined_video"] is True
        assert summary["config"]["max_retries"] == 3
        assert summary["config"]["use_rag"] is True
        assert summary["config"]["use_visual_analysis"] is True


if __name__ == "__main__":
    pytest.main([__file__])
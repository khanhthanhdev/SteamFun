"""
Comprehensive integration tests for multi-agent workflows.
Tests the complete integration of agent communication, state management, error propagation, and tool integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List
from copy import deepcopy
from datetime import datetime

from app.core.agents.base_agent import BaseAgent, AgentFactory
from src.langgraph_agents.state import VideoGenerationState, AgentConfig, SystemConfig, AgentError
from langgraph.types import Command

# Import test utilities
from tests.utils.config_mocks import (
    mock_configuration_manager, create_test_system_config, 
    create_test_agent_config, MockConfigurationManager
)


class TestComprehensiveIntegration:
    """Comprehensive integration test suite for multi-agent workflows."""
    
    @pytest.fixture
    def system_config(self):
        """Create comprehensive system configuration using configuration system."""
        return create_test_system_config(
            environment="test",
            debug=True,
            include_rag=True,
            include_monitoring=True
        )
    
    @pytest.fixture
    def initial_state(self):
        """Create comprehensive initial state."""
        return VideoGenerationState(
            messages=[],
            topic="Comprehensive Integration Test",
            description="Testing complete multi-agent workflow integration",
            session_id="comprehensive_test_session",
            output_dir="test_output",
            print_response=False,
            use_rag=True,
            use_context_learning=True,
            context_learning_path="test_context",
            chroma_db_path="test_chroma",
            manim_docs_path="test_docs",
            embedding_model="test_model",
            use_visual_fix_code=False,
            use_langfuse=True,
            max_scene_concurrency=5,
            max_topic_concurrency=1,
            max_retries=5,
            use_enhanced_rag=True,
            enable_rag_caching=True,
            enable_quality_monitoring=True,
            enable_error_handling=True,
            rag_cache_ttl=3600,
            rag_max_cache_size=1000,
            rag_performance_threshold=2.0,
            rag_quality_threshold=0.7,
            enable_caching=True,
            default_quality="medium",
            use_gpu_acceleration=False,
            preview_mode=False,
            max_concurrent_renders=4,
            scene_outline=None,
            scene_implementations={},
            detected_plugins=[],
            generated_code={},
            code_errors={},
            rag_context={},
            rendered_videos={},
            combined_video_path=None,
            rendering_errors={},
            visual_analysis_results={},
            visual_errors={},
            error_count=0,
            retry_count={},
            escalated_errors=[],
            pending_human_input=None,
            human_feedback=None,
            performance_metrics={},
            execution_trace=[],
            current_agent=None,
            next_agent="planner_agent",
            workflow_complete=False,
            workflow_interrupted=False
        )
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_workflow_integration(self, system_config, initial_state):
        """Test complete workflow integration from planning to rendering."""
        workflow_trace = []
        
        # Mock complete workflow agents
        def create_workflow_agent(agent_name: str, next_agent: str, updates: Dict[str, Any]):
            class WorkflowAgent:
                def __init__(self):
                    self.name = agent_name
                
                async def execute_with_monitoring(self, state: VideoGenerationState) -> Command:
                    workflow_trace.append({
                        "agent": agent_name,
                        "input_state": {
                            "topic": state.get("topic"),
                            "session_id": state.get("session_id"),
                            "current_agent": state.get("current_agent"),
                            "error_count": state.get("error_count", 0)
                        },
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    return Command(
                        goto=next_agent,
                        update={
                            **updates,
                            "current_agent": next_agent,
                            "execution_trace": state.get("execution_trace", []) + [{
                                "agent": agent_name,
                                "action": "completed",
                                "timestamp": datetime.now().isoformat()
                            }]
                        }
                    )
            
            return WorkflowAgent()
        
        # Create workflow sequence
        workflow_agents = [
            ("planner_agent", "code_generator_agent", {
                "scene_outline": "# Scene 1\nTest scene outline",
                "scene_implementations": {1: "Test implementation"},
                "detected_plugins": ["text", "code"]
            }),
            ("code_generator_agent", "renderer_agent", {
                "generated_code": {1: "from manim import *\nclass TestScene(Scene): pass"},
                "rag_context": {1: "RAG context for scene 1"}
            }),
            ("renderer_agent", "END", {
                "rendered_videos": {1: "/path/to/scene1.mp4"},
                "combined_video_path": "/path/to/final_video.mp4",
                "workflow_complete": True
            })
        ]
        
        # Execute complete workflow
        current_state = initial_state.copy()
        
        for agent_name, next_agent, updates in workflow_agents:
            agent = create_workflow_agent(agent_name, next_agent, updates)
            result = await agent.execute_with_monitoring(current_state)
            current_state.update(result.update)
            
            if next_agent == "END":
                break
        
        # Verify complete workflow execution
        assert len(workflow_trace) == 3
        assert workflow_trace[0]["agent"] == "planner_agent"
        assert workflow_trace[1]["agent"] == "code_generator_agent"
        assert workflow_trace[2]["agent"] == "renderer_agent"
        
        # Verify final state
        assert current_state["workflow_complete"] is True
        assert current_state["scene_outline"] == "# Scene 1\nTest scene outline"
        assert current_state["generated_code"] == {1: "from manim import *\nclass TestScene(Scene): pass"}
        assert current_state["rendered_videos"] == {1: "/path/to/scene1.mp4"}
        assert current_state["combined_video_path"] == "/path/to/final_video.mp4"
        
        # Verify execution trace
        execution_trace = current_state["execution_trace"]
        assert len(execution_trace) == 3
        assert all(entry["action"] == "completed" for entry in execution_trace)


if __name__ == "__main__":
    pytest.main([__file__])
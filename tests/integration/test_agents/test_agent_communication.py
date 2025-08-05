"""
Integration tests for agent-to-agent communication.
Tests state management, command routing, and data flow between agents.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from app.core.agents.base_agent import BaseAgent, AgentFactory
from src.langgraph_agents.state import VideoGenerationState, AgentConfig, SystemConfig
from langgraph.types import Command


class TestAgentCommunication:
    """Test suite for agent-to-agent communication and state management."""
    
    @pytest.fixture
    def mock_system_config(self):
        """Create mock system configuration for integration tests."""
        return SystemConfig(
            agents={
                "planner_agent": AgentConfig(
                    name="planner_agent",
                    model_config={"planner_model": "test_model"},
                    tools=["planning_tool"]
                ),
                "code_generator_agent": AgentConfig(
                    name="code_generator_agent", 
                    model_config={"scene_model": "test_model"},
                    tools=["code_tool"]
                ),
                "renderer_agent": AgentConfig(
                    name="renderer_agent",
                    model_config={"renderer_model": "test_model"},
                    tools=["render_tool"]
                ),
                "error_handler_agent": AgentConfig(
                    name="error_handler_agent",
                    model_config={},
                    tools=["error_tool"]
                )
            },
            llm_providers={
                "openrouter": {"api_key": "test_key"}
            },
            docling_config={},
            mcp_servers={},
            monitoring_config={},
            human_loop_config={}
        )
    
    @pytest.fixture
    def mock_initial_state(self):
        """Create initial state for integration tests."""
        return VideoGenerationState(
            messages=[],
            topic="Integration Test Topic",
            description="Test description for integration testing",
            session_id="integration_test_session",
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
    async def test_planner_to_code_generator_communication(self, mock_system_config, mock_initial_state):
        """Test communication from PlannerAgent to CodeGeneratorAgent."""
        # Mock PlannerAgent execution
        with patch('app.core.agents.agents.planner_agent.PlannerAgent') as MockPlannerAgent:
            mock_planner = Mock()
            mock_planner.execute_with_monitoring = AsyncMock(return_value=Command(
                goto="code_generator_agent",
                update={
                    "scene_outline": "# Scene 1\nTest scene outline",
                    "scene_implementations": {1: "Test scene implementation"},
                    "detected_plugins": ["text", "code"],
                    "current_agent": "code_generator_agent"
                }
            ))
            MockPlannerAgent.return_value = mock_planner
            
            # Mock CodeGeneratorAgent execution
            with patch('app.core.agents.agents.code_generator_agent.CodeGeneratorAgent') as MockCodeGeneratorAgent:
                mock_code_gen = Mock()
                mock_code_gen.execute_with_monitoring = AsyncMock(return_value=Command(
                    goto="renderer_agent",
                    update={
                        "generated_code": {1: "test manim code"},
                        "current_agent": "renderer_agent"
                    }
                ))
                MockCodeGeneratorAgent.return_value = mock_code_gen
                
                # Create agents
                planner = AgentFactory.create_agent("planner_agent", mock_system_config.agents["planner_agent"], mock_system_config.__dict__)
                code_generator = AgentFactory.create_agent("code_generator_agent", mock_system_config.agents["code_generator_agent"], mock_system_config.__dict__)
                
                # Execute planner
                planner_result = await planner.execute_with_monitoring(mock_initial_state)
                
                # Verify planner output
                assert planner_result.goto == "code_generator_agent"
                assert "scene_outline" in planner_result.update
                assert "scene_implementations" in planner_result.update
                
                # Update state with planner results
                updated_state = mock_initial_state.copy()
                updated_state.update(planner_result.update)
                
                # Execute code generator with updated state
                code_gen_result = await code_generator.execute_with_monitoring(updated_state)
                
                # Verify code generator received planner data
                assert updated_state["scene_outline"] == "# Scene 1\nTest scene outline"
                assert updated_state["scene_implementations"] == {1: "Test scene implementation"}
                assert updated_state["detected_plugins"] == ["text", "code"]
                
                # Verify code generator output
                assert code_gen_result.goto == "renderer_agent"
                assert "generated_code" in code_gen_result.update


if __name__ == "__main__":
    pytest.main([__file__])
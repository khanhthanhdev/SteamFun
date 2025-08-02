"""
Integration tests for agent-to-agent communication.
Tests state management, command routing, and data flow between agents.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from src.langgraph_agents.base_agent import BaseAgent, AgentFactory
from src.langgraph_agents.state import VideoGenerationState, AgentConfig, SystemConfig
from src.langgraph_agents.graph import create_workflow_graph
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
        with patch('src.langgraph_agents.agents.planner_agent.PlannerAgent') as MockPlannerAgent:
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
            with patch('src.langgraph_agents.agents.code_generator_agent.CodeGeneratorAgent') as MockCodeGeneratorAgent:
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
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_error_propagation_between_agents(self, mock_system_config, mock_initial_state):
        """Test error propagation and handling between agents."""
        # Mock CodeGeneratorAgent with error
        with patch('src.langgraph_agents.agents.code_generator_agent.CodeGeneratorAgent') as MockCodeGeneratorAgent:
            mock_code_gen = Mock()
            mock_code_gen.execute_with_monitoring = AsyncMock(return_value=Command(
                goto="error_handler_agent",
                update={
                    "error_count": 1,
                    "escalated_errors": [{
                        "agent": "code_generator_agent",
                        "error": "Code generation failed",
                        "timestamp": "2024-01-01T00:00:00"
                    }],
                    "current_agent": "error_handler_agent"
                }
            ))
            MockCodeGeneratorAgent.return_value = mock_code_gen
            
            # Mock ErrorHandlerAgent
            with patch('src.langgraph_agents.agents.error_handler_agent.ErrorHandlerAgent') as MockErrorHandlerAgent:
                mock_error_handler = Mock()
                mock_error_handler.execute_with_monitoring = AsyncMock(return_value=Command(
                    goto="code_generator_agent",
                    update={
                        "retry_count": {"code_generator_agent": 1},
                        "current_agent": "code_generator_agent"
                    }
                ))
                MockErrorHandlerAgent.return_value = mock_error_handler
                
                # Create agents
                code_generator = AgentFactory.create_agent("code_generator_agent", mock_system_config.agents["code_generator_agent"], mock_system_config.__dict__)
                error_handler = AgentFactory.create_agent("error_handler_agent", mock_system_config.agents["error_handler_agent"], mock_system_config.__dict__)
                
                # Execute code generator (which fails)
                code_gen_result = await code_generator.execute_with_monitoring(mock_initial_state)
                
                # Verify error was escalated
                assert code_gen_result.goto == "error_handler_agent"
                assert code_gen_result.update["error_count"] == 1
                assert len(code_gen_result.update["escalated_errors"]) == 1
                
                # Update state with error
                error_state = mock_initial_state.copy()
                error_state.update(code_gen_result.update)
                
                # Execute error handler
                error_handler_result = await error_handler.execute_with_monitoring(error_state)
                
                # Verify error handler processed the error
                assert error_handler_result.goto == "code_generator_agent"
                assert "retry_count" in error_handler_result.update
                assert error_handler_result.update["retry_count"]["code_generator_agent"] == 1
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_rag_agent_integration(self, mock_system_config, mock_initial_state):
        """Test RAGAgent integration with other agents."""
        # Add RAGAgent to system config
        mock_system_config.agents["rag_agent"] = AgentConfig(
            name="rag_agent",
            model_config={"helper_model": "test_model"},
            tools=["rag_tool"]
        )
        
        # Mock CodeGeneratorAgent requesting RAG context
        with patch('src.langgraph_agents.agents.code_generator_agent.CodeGeneratorAgent') as MockCodeGeneratorAgent:
            mock_code_gen = Mock()
            mock_code_gen.execute_with_monitoring = AsyncMock(return_value=Command(
                goto="rag_agent",
                update={
                    "requesting_agent": "code_generator_agent",
                    "rag_query_context": "Need help with Manim animations",
                    "current_agent": "rag_agent"
                }
            ))
            MockCodeGeneratorAgent.return_value = mock_code_gen
            
            # Mock RAGAgent
            with patch('src.langgraph_agents.agents.rag_agent.RAGAgent') as MockRAGAgent:
                mock_rag = Mock()
                mock_rag.execute_with_monitoring = AsyncMock(return_value=Command(
                    goto="code_generator_agent",
                    update={
                        "rag_context": {
                            "query1": "Manim animation context",
                            "query2": "Scene construction help"
                        },
                        "current_agent": "code_generator_agent"
                    }
                ))
                MockRAGAgent.return_value = mock_rag
                
                # Create agents
                code_generator = AgentFactory.create_agent("code_generator_agent", mock_system_config.agents["code_generator_agent"], mock_system_config.__dict__)
                rag_agent = AgentFactory.create_agent("rag_agent", mock_system_config.agents["rag_agent"], mock_system_config.__dict__)
                
                # Execute code generator (requests RAG)
                code_gen_result = await code_generator.execute_with_monitoring(mock_initial_state)
                
                # Verify RAG request
                assert code_gen_result.goto == "rag_agent"
                assert code_gen_result.update["requesting_agent"] == "code_generator_agent"
                
                # Update state for RAG
                rag_state = mock_initial_state.copy()
                rag_state.update(code_gen_result.update)
                
                # Execute RAG agent
                rag_result = await rag_agent.execute_with_monitoring(rag_state)
                
                # Verify RAG response
                assert rag_result.goto == "code_generator_agent"
                assert "rag_context" in rag_result.update
                assert len(rag_result.update["rag_context"]) == 2
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_state_consistency_across_agents(self, mock_system_config, mock_initial_state):
        """Test that state remains consistent as it flows between agents."""
        state_updates = []
        
        # Mock multiple agents to track state changes
        def create_mock_agent(agent_name: str, next_agent: str, update_data: Dict[str, Any]):
            mock_agent = Mock()
            
            async def mock_execute(state):
                # Record state at execution time
                state_updates.append({
                    "agent": agent_name,
                    "state_snapshot": state.copy(),
                    "update": update_data
                })
                return Command(goto=next_agent, update=update_data)
            
            mock_agent.execute_with_monitoring = AsyncMock(side_effect=mock_execute)
            return mock_agent
        
        with patch('src.langgraph_agents.agents.planner_agent.PlannerAgent') as MockPlannerAgent:
            MockPlannerAgent.return_value = create_mock_agent(
                "planner_agent", 
                "code_generator_agent",
                {"scene_outline": "test outline", "planning_complete": True}
            )
            
            with patch('src.langgraph_agents.agents.code_generator_agent.CodeGeneratorAgent') as MockCodeGeneratorAgent:
                MockCodeGeneratorAgent.return_value = create_mock_agent(
                    "code_generator_agent",
                    "renderer_agent", 
                    {"generated_code": {1: "test code"}, "code_generation_complete": True}
                )
                
                with patch('src.langgraph_agents.agents.renderer_agent.RendererAgent') as MockRendererAgent:
                    MockRendererAgent.return_value = create_mock_agent(
                        "renderer_agent",
                        "END",
                        {"rendered_videos": {1: "test.mp4"}, "rendering_complete": True}
                    )
                    
                    # Create agents
                    planner = AgentFactory.create_agent("planner_agent", mock_system_config.agents["planner_agent"], mock_system_config.__dict__)
                    code_generator = AgentFactory.create_agent("code_generator_agent", mock_system_config.agents["code_generator_agent"], mock_system_config.__dict__)
                    renderer = AgentFactory.create_agent("renderer_agent", mock_system_config.agents["renderer_agent"], mock_system_config.__dict__)
                    
                    # Execute workflow sequence
                    current_state = mock_initial_state.copy()
                    
                    # Planner execution
                    planner_result = await planner.execute_with_monitoring(current_state)
                    current_state.update(planner_result.update)
                    
                    # Code generator execution
                    code_gen_result = await code_generator.execute_with_monitoring(current_state)
                    current_state.update(code_gen_result.update)
                    
                    # Renderer execution
                    renderer_result = await renderer.execute_with_monitoring(current_state)
                    current_state.update(renderer_result.update)
                    
                    # Verify state consistency
                    assert len(state_updates) == 3
                    
                    # Check that each agent received expected data
                    planner_state = state_updates[0]["state_snapshot"]
                    assert planner_state["topic"] == "Integration Test Topic"
                    
                    code_gen_state = state_updates[1]["state_snapshot"]
                    assert code_gen_state["scene_outline"] == "test outline"
                    assert code_gen_state["planning_complete"] is True
                    
                    renderer_state = state_updates[2]["state_snapshot"]
                    assert renderer_state["generated_code"] == {1: "test code"}
                    assert renderer_state["code_generation_complete"] is True
                    
                    # Verify final state has all updates
                    assert current_state["scene_outline"] == "test outline"
                    assert current_state["generated_code"] == {1: "test code"}
                    assert current_state["rendered_videos"] == {1: "test.mp4"}
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_human_loop_integration(self, mock_system_config, mock_initial_state):
        """Test human-in-the-loop integration with agent workflow."""
        # Add HumanLoopAgent to system config
        mock_system_config.agents["human_loop_agent"] = AgentConfig(
            name="human_loop_agent",
            model_config={},
            tools=["human_tool"],
            enable_human_loop=True
        )
        
        # Mock agent that escalates to human
        with patch('src.langgraph_agents.agents.code_generator_agent.CodeGeneratorAgent') as MockCodeGeneratorAgent:
            mock_code_gen = Mock()
            mock_code_gen.execute_with_monitoring = AsyncMock(return_value=Command(
                goto="human_loop_agent",
                update={
                    "pending_human_input": {
                        "context": "Code generation needs human review",
                        "options": ["approve", "modify", "reject"],
                        "requesting_agent": "code_generator_agent"
                    },
                    "current_agent": "human_loop_agent"
                }
            ))
            MockCodeGeneratorAgent.return_value = mock_code_gen
            
            # Mock HumanLoopAgent
            with patch('src.langgraph_agents.agents.human_loop_agent.HumanLoopAgent') as MockHumanLoopAgent:
                mock_human = Mock()
                mock_human.execute_with_monitoring = AsyncMock(return_value=Command(
                    goto="code_generator_agent",
                    update={
                        "human_feedback": {
                            "decision": "approve",
                            "comments": "Code looks good",
                            "timestamp": "2024-01-01T00:00:00"
                        },
                        "pending_human_input": None,
                        "current_agent": "code_generator_agent"
                    }
                ))
                MockHumanLoopAgent.return_value = mock_human
                
                # Create agents
                code_generator = AgentFactory.create_agent("code_generator_agent", mock_system_config.agents["code_generator_agent"], mock_system_config.__dict__)
                human_loop = AgentFactory.create_agent("human_loop_agent", mock_system_config.agents["human_loop_agent"], mock_system_config.__dict__)
                
                # Execute code generator (escalates to human)
                code_gen_result = await code_generator.execute_with_monitoring(mock_initial_state)
                
                # Verify human escalation
                assert code_gen_result.goto == "human_loop_agent"
                assert "pending_human_input" in code_gen_result.update
                assert code_gen_result.update["pending_human_input"]["requesting_agent"] == "code_generator_agent"
                
                # Update state for human loop
                human_state = mock_initial_state.copy()
                human_state.update(code_gen_result.update)
                
                # Execute human loop agent
                human_result = await human_loop.execute_with_monitoring(human_state)
                
                # Verify human feedback
                assert human_result.goto == "code_generator_agent"
                assert "human_feedback" in human_result.update
                assert human_result.update["human_feedback"]["decision"] == "approve"
                assert human_result.update["pending_human_input"] is None
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_monitoring_agent_integration(self, mock_system_config, mock_initial_state):
        """Test MonitoringAgent integration with workflow tracking."""
        # Add MonitoringAgent to system config
        mock_system_config.agents["monitoring_agent"] = AgentConfig(
            name="monitoring_agent",
            model_config={},
            tools=["monitoring_tool"]
        )
        
        # Mock agents with performance tracking
        execution_traces = []
        
        def create_monitored_agent(agent_name: str, execution_time: float):
            mock_agent = Mock()
            
            async def mock_execute(state):
                # Simulate execution with monitoring
                trace_entry = {
                    "agent": agent_name,
                    "action": "execute",
                    "timestamp": "2024-01-01T00:00:00",
                    "execution_time": execution_time
                }
                execution_traces.append(trace_entry)
                
                return Command(
                    goto="monitoring_agent",
                    update={
                        "execution_trace": state.get("execution_trace", []) + [trace_entry],
                        "performance_metrics": {
                            **state.get("performance_metrics", {}),
                            agent_name: {
                                "last_execution_time": execution_time,
                                "success_rate": 1.0
                            }
                        }
                    }
                )
            
            mock_agent.execute_with_monitoring = AsyncMock(side_effect=mock_execute)
            return mock_agent
        
        with patch('src.langgraph_agents.agents.planner_agent.PlannerAgent') as MockPlannerAgent:
            MockPlannerAgent.return_value = create_monitored_agent("planner_agent", 1.5)
            
            with patch('src.langgraph_agents.agents.monitoring_agent.MonitoringAgent') as MockMonitoringAgent:
                mock_monitoring = Mock()
                mock_monitoring.execute_with_monitoring = AsyncMock(return_value=Command(
                    goto="code_generator_agent",
                    update={
                        "monitoring_report": {
                            "total_agents_executed": 1,
                            "average_execution_time": 1.5,
                            "performance_summary": "All agents performing within normal parameters"
                        }
                    }
                ))
                MockMonitoringAgent.return_value = mock_monitoring
                
                # Create agents
                planner = AgentFactory.create_agent("planner_agent", mock_system_config.agents["planner_agent"], mock_system_config.__dict__)
                monitoring = AgentFactory.create_agent("monitoring_agent", mock_system_config.agents["monitoring_agent"], mock_system_config.__dict__)
                
                # Execute planner with monitoring
                planner_result = await planner.execute_with_monitoring(mock_initial_state)
                
                # Verify monitoring data was collected
                assert "execution_trace" in planner_result.update
                assert "performance_metrics" in planner_result.update
                assert len(planner_result.update["execution_trace"]) == 1
                
                # Update state for monitoring agent
                monitoring_state = mock_initial_state.copy()
                monitoring_state.update(planner_result.update)
                
                # Execute monitoring agent
                monitoring_result = await monitoring.execute_with_monitoring(monitoring_state)
                
                # Verify monitoring report
                assert "monitoring_report" in monitoring_result.update
                assert monitoring_result.update["monitoring_report"]["total_agents_executed"] == 1
                assert monitoring_result.update["monitoring_report"]["average_execution_time"] == 1.5


if __name__ == "__main__":
    pytest.main([__file__])
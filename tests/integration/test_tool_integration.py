"""
Integration tests for tool integration testing.
Tests external tool connectivity, MCP servers, and tool coordination.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from src.langgraph_agents.base_agent import BaseAgent, AgentFactory
from src.langgraph_agents.state import VideoGenerationState, AgentConfig, SystemConfig
from src.langgraph_agents.agents.rag_agent import RAGAgent
from src.langgraph_agents.tools.human_intervention_tools import request_human_approval
from langgraph.types import Command

class TestToolIntegration:
    """Test suite for external tool integration and coordination."""
    
    @pytest.fixture
    def mock_system_config(self):
        """Create mock system configuration with tool integrations."""
        return SystemConfig(
            agents={
                "rag_agent": AgentConfig(
                    name="rag_agent",
                    model_config={"helper_model": "test_model"},
                    tools=["rag_tool", "context7_tool", "docling_tool"]
                ),
                "code_generator_agent": AgentConfig(
                    name="code_generator_agent",
                    model_config={"scene_model": "test_model"},
                    tools=["code_tool", "rag_tool"]
                ),
                "human_loop_agent": AgentConfig(
                    name="human_loop_agent",
                    model_config={},
                    tools=["human_interface", "approval_tool"],
                    enable_human_loop=True
                )
            },
            llm_providers={
                "openrouter": {"api_key": "test_key"}
            },
            docling_config={
                "enabled": True,
                "max_file_size_mb": 50,
                "supported_formats": [".md", ".txt", ".pdf"]
            },
            mcp_servers={
                "context7": {
                    "command": "uvx",
                    "args": ["context7-mcp-server@latest"],
                    "disabled": False,
                    "autoApprove": ["resolve_library_id", "get_library_docs"]
                },
                "docling": {
                    "command": "uvx", 
                    "args": ["docling-mcp-server@latest"],
                    "disabled": False,
                    "autoApprove": ["docling_document_processor"]
                }
            },
            monitoring_config={},
            human_loop_config={
                "enabled": True,
                "timeout": 300,
                "interface_type": "cli"
            }
        )
    
    @pytest.fixture
    def initial_state(self, mock_system_config):
        """Create initial state for tool integration tests."""
        return VideoGenerationState(
            messages=[],
            topic="Tool Integration Test",
            description="Testing external tool integration",
            session_id="tool_test_session",
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
            scene_implementations={1: "Test scene implementation"},
            detected_plugins=["text", "code"],
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
            next_agent=None,
            workflow_complete=False,
            workflow_interrupted=False
        )
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_mcp_server_connectivity(self, mock_system_config, initial_state):
        """Test MCP server connectivity and tool loading."""
        mcp_calls = []
        
        # Mock MCP client
        class MockMCPClient:
            def __init__(self, server_name: str):
                self.server_name = server_name
                self.connected = True
            
            async def call(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
                mcp_calls.append({
                    "server": self.server_name,
                    "tool": tool_name,
                    "parameters": parameters,
                    "timestamp": "2024-01-01T00:00:00"
                })
                
                # Mock responses based on tool
                if tool_name == "resolve_library_id":
                    return "/test/library"
                elif tool_name == "get_library_docs":
                    return "Test library documentation content"
                elif tool_name == "docling_document_processor":
                    return "Processed document content"
                else:
                    return f"Mock response for {tool_name}"
            
            async def list_tools(self) -> List[str]:
                if self.server_name == "context7":
                    return ["resolve_library_id", "get_library_docs"]
                elif self.server_name == "docling":
                    return ["docling_document_processor"]
                return []
        
        # Mock MCP service
        with patch('src.langgraph_agents.services.mcp_service.MCPService') as MockMCPService:
            mock_service = Mock()
            mock_service.get_client = Mock(side_effect=lambda name: MockMCPClient(name))
            mock_service.is_server_available = Mock(return_value=True)
            mock_service.list_available_tools = AsyncMock(return_value={
                "context7": ["resolve_library_id", "get_library_docs"],
                "docling": ["docling_document_processor"]
            })
            MockMCPService.return_value = mock_service
            
            # Test RAG agent with MCP tools
            with patch('src.langgraph_agents.agents.rag_agent.RAGAgent') as MockRAGAgent:
                mock_rag = Mock()
                
                async def mock_execute(state):
                    # Simulate MCP tool usage
                    context7_client = MockMCPClient("context7")
                    docling_client = MockMCPClient("docling")
                    
                    # Call Context7 tools
                    library_id = await context7_client.call("resolve_library_id", {"libraryName": "test_library"})
                    docs = await context7_client.call("get_library_docs", {"context7CompatibleLibraryID": library_id})
                    
                    # Call Docling tools
                    processed_doc = await docling_client.call("docling_document_processor", {"document_path": "test.md"})
                    
                    return Command(
                        goto="code_generator_agent",
                        update={
                            "rag_context": {1: f"Context7: {docs}\nDocling: {processed_doc}"},
                            "mcp_tools_used": ["context7", "docling"]
                        }
                    )
                
                mock_rag.execute_with_monitoring = AsyncMock(side_effect=mock_execute)
                MockRAGAgent.return_value = mock_rag
                
                # Create and execute RAG agent
                rag_agent = AgentFactory.create_agent("rag_agent", mock_system_config.agents["rag_agent"], mock_system_config.__dict__)
                result = await rag_agent.execute_with_monitoring(initial_state)
                
                # Verify MCP tool calls
                assert len(mcp_calls) == 3
                assert any(call["tool"] == "resolve_library_id" for call in mcp_calls)
                assert any(call["tool"] == "get_library_docs" for call in mcp_calls)
                assert any(call["tool"] == "docling_document_processor" for call in mcp_calls)
                
                # Verify result
                assert result.goto == "code_generator_agent"
                assert "rag_context" in result.update
                assert "mcp_tools_used" in result.update
                assert "Context7" in result.update["rag_context"][1]
                assert "Docling" in result.update["rag_context"][1]
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_context7_integration(self, mock_system_config, initial_state):
        """Test Context7 integration for enhanced documentation."""
        context7_queries = []
        
        # Mock Context7 MCP client
        class MockContext7Client:
            async def call(self, tool_name: str, parameters: Dict[str, Any]) -> str:
                context7_queries.append({
                    "tool": tool_name,
                    "parameters": parameters
                })
                
                if tool_name == "resolve_library_id":
                    library_name = parameters.get("libraryName", "")
                    return f"/test/{library_name.replace(' ', '-')}"
                elif tool_name == "get_library_docs":
                    library_id = parameters.get("context7CompatibleLibraryID", "")
                    topic = parameters.get("topic", "")
                    return f"Documentation for {library_id} on topic: {topic}\n\nExample code:\n```python\nimport {library_id.split('/')[-1]}\n```"
                
                return "Mock Context7 response"
        
        # Test Context7 integration through RAG agent
        with patch('src.langgraph_agents.agents.rag_agent.RAGAgent') as MockRAGAgent:
            mock_rag = Mock()
            
            async def mock_execute_with_context7(state):
                # Simulate Context7 integration
                client = MockContext7Client()
                
                # Query for Python documentation
                library_id = await client.call("resolve_library_id", {"libraryName": "python matplotlib"})
                docs = await client.call("get_library_docs", {
                    "context7CompatibleLibraryID": library_id,
                    "topic": "plotting",
                    "tokens": 10000
                })
                
                return Command(
                    goto="code_generator_agent",
                    update={
                        "rag_context": {1: docs},
                        "context7_library_id": library_id,
                        "context7_queries_made": len(context7_queries)
                    }
                )
            
            mock_rag.execute_with_monitoring = AsyncMock(side_effect=mock_execute_with_context7)
            MockRAGAgent.return_value = mock_rag
            
            # Execute RAG agent with Context7
            rag_agent = AgentFactory.create_agent("rag_agent", mock_system_config.agents["rag_agent"], mock_system_config.__dict__)
            result = await rag_agent.execute_with_monitoring(initial_state)
            
            # Verify Context7 integration
            assert len(context7_queries) == 2
            assert context7_queries[0]["tool"] == "resolve_library_id"
            assert context7_queries[1]["tool"] == "get_library_docs"
            
            # Verify result contains Context7 data
            assert result.update["context7_library_id"] == "/test/python-matplotlib"
            assert "Documentation for /test/python-matplotlib" in result.update["rag_context"][1]
            assert "Example code:" in result.update["rag_context"][1]
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_docling_document_processing(self, mock_system_config, initial_state):
        """Test Docling integration for document processing."""
        processed_documents = []
        
        # Mock Docling MCP client
        class MockDoclingClient:
            async def call(self, tool_name: str, parameters: Dict[str, Any]) -> str:
                if tool_name == "docling_document_processor":
                    doc_path = parameters.get("document_path", "")
                    processed_documents.append(doc_path)
                    
                    # Simulate document processing
                    if doc_path.endswith(".md"):
                        return f"Processed Markdown document: {doc_path}\n\nExtracted content:\n- Headers: 3\n- Code blocks: 2\n- Links: 5"
                    elif doc_path.endswith(".pdf"):
                        return f"Processed PDF document: {doc_path}\n\nExtracted content:\n- Pages: 10\n- Images: 3\n- Tables: 2"
                    else:
                        return f"Processed document: {doc_path}\n\nGeneric content extraction"
                
                return "Mock Docling response"
        
        # Test Docling integration
        with patch('src.langgraph_agents.agents.rag_agent.RAGAgent') as MockRAGAgent:
            mock_rag = Mock()
            
            async def mock_execute_with_docling(state):
                client = MockDoclingClient()
                
                # Process different document types
                documents = ["README.md", "guide.pdf", "notes.txt"]
                processed_content = []
                
                for doc in documents:
                    content = await client.call("docling_document_processor", {"document_path": doc})
                    processed_content.append(content)
                
                combined_content = "\n\n".join(processed_content)
                
                return Command(
                    goto="code_generator_agent",
                    update={
                        "rag_context": {1: combined_content},
                        "processed_documents": processed_documents,
                        "docling_processing_complete": True
                    }
                )
            
            mock_rag.execute_with_monitoring = AsyncMock(side_effect=mock_execute_with_docling)
            MockRAGAgent.return_value = mock_rag
            
            # Execute RAG agent with Docling
            rag_agent = AgentFactory.create_agent("rag_agent", mock_system_config.agents["rag_agent"], mock_system_config.__dict__)
            result = await rag_agent.execute_with_monitoring(initial_state)
            
            # Verify Docling processing
            assert len(processed_documents) == 3
            assert "README.md" in processed_documents
            assert "guide.pdf" in processed_documents
            assert "notes.txt" in processed_documents
            
            # Verify processed content
            assert "Processed Markdown document" in result.update["rag_context"][1]
            assert "Processed PDF document" in result.update["rag_context"][1]
            assert result.update["docling_processing_complete"] is True
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_human_intervention_tools(self, mock_system_config, initial_state):
        """Test human intervention tools and interfaces."""
        human_interactions = []
        
        # Mock human intervention interface
        class MockHumanInterface:
            def __init__(self):
                self.pending_requests = []
            
            async def request_approval(self, context: str, options: List[str], timeout: int = 300) -> str:
                request = {
                    "context": context,
                    "options": options,
                    "timeout": timeout,
                    "timestamp": "2024-01-01T00:00:00"
                }
                human_interactions.append(request)
                self.pending_requests.append(request)
                
                # Simulate human response
                if "critical" in context.lower():
                    return "abort"
                elif "approve" in options:
                    return "approve"
                else:
                    return options[0] if options else "continue"
            
            async def collect_feedback(self, prompt: str) -> str:
                feedback_request = {
                    "type": "feedback",
                    "prompt": prompt,
                    "timestamp": "2024-01-01T00:00:00"
                }
                human_interactions.append(feedback_request)
                return "User feedback: Looks good, proceed with changes"
            
            def get_pending_requests(self) -> List[Dict[str, Any]]:
                return self.pending_requests
        
        # Test human intervention tools
        with patch('src.langgraph_agents.tools.human_intervention_tools.HumanInterventionInterface') as MockInterface:
            mock_interface = MockHumanInterface()
            MockInterface.return_value = mock_interface
            
            # Test request_human_approval tool
            approval_command = request_human_approval(
                context="Code generation requires human review",
                options=["approve", "modify", "reject"],
                priority="medium",
                timeout_seconds=300,
                default_action="approve",
                state=initial_state
            )
            
            # Verify approval command structure
            assert approval_command.goto == "human_loop_agent"
            assert "pending_human_input" in approval_command.update
            
            # Test human loop agent with tools
            with patch('src.langgraph_agents.agents.human_loop_agent.HumanLoopAgent') as MockHumanLoopAgent:
                mock_human_agent = Mock()
                
                async def mock_execute_human_loop(state):
                    pending_input = state.get("pending_human_input", {})
                    
                    if pending_input:
                        # Use human interface to get approval
                        response = await mock_interface.request_approval(
                            context=pending_input.get("context", ""),
                            options=pending_input.get("options", []),
                            timeout=pending_input.get("timeout_seconds", 300)
                        )
                        
                        # Collect additional feedback if needed
                        feedback = await mock_interface.collect_feedback(
                            "Any additional comments on the decision?"
                        )
                        
                        return Command(
                            goto="code_generator_agent",
                            update={
                                "human_feedback": {
                                    "decision": response,
                                    "feedback": feedback,
                                    "timestamp": "2024-01-01T00:00:00"
                                },
                                "pending_human_input": None
                            }
                        )
                    
                    return Command(goto="next_agent")
                
                mock_human_agent.execute_with_monitoring = AsyncMock(side_effect=mock_execute_human_loop)
                MockHumanLoopAgent.return_value = mock_human_agent
                
                # Execute human loop workflow
                human_state = initial_state.copy()
                human_state.update(approval_command.update)
                
                human_agent = AgentFactory.create_agent("human_loop_agent", mock_system_config.agents["human_loop_agent"], mock_system_config.__dict__)
                result = await human_agent.execute_with_monitoring(human_state)
                
                # Verify human intervention workflow
                assert len(human_interactions) == 2  # Approval + feedback
                assert human_interactions[0]["context"] == "Code generation requires human review"
                assert human_interactions[0]["options"] == ["approve", "modify", "reject"]
                assert human_interactions[1]["type"] == "feedback"
                
                # Verify result
                assert result.goto == "code_generator_agent"
                assert result.update["human_feedback"]["decision"] == "approve"
                assert "User feedback:" in result.update["human_feedback"]["feedback"]
                assert result.update["pending_human_input"] is None
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tool_error_handling(self, mock_system_config, initial_state):
        """Test error handling for external tool failures."""
        tool_errors = []
        
        # Mock failing MCP client
        class FailingMCPClient:
            def __init__(self, failure_type: str):
                self.failure_type = failure_type
            
            async def call(self, tool_name: str, parameters: Dict[str, Any]) -> str:
                error_info = {
                    "tool": tool_name,
                    "parameters": parameters,
                    "failure_type": self.failure_type,
                    "timestamp": "2024-01-01T00:00:00"
                }
                tool_errors.append(error_info)
                
                if self.failure_type == "timeout":
                    raise asyncio.TimeoutError(f"Tool {tool_name} timed out")
                elif self.failure_type == "connection":
                    raise ConnectionError(f"Cannot connect to {tool_name} service")
                elif self.failure_type == "invalid_response":
                    raise ValueError(f"Invalid response from {tool_name}")
                else:
                    raise Exception(f"Unknown error in {tool_name}")
        
        # Test tool error handling in RAG agent
        with patch('src.langgraph_agents.agents.rag_agent.RAGAgent') as MockRAGAgent:
            mock_rag = Mock()
            
            async def mock_execute_with_errors(state):
                # Try different failing clients
                clients = [
                    FailingMCPClient("timeout"),
                    FailingMCPClient("connection"),
                    FailingMCPClient("invalid_response")
                ]
                
                successful_calls = 0
                failed_calls = 0
                
                for client in clients:
                    try:
                        await client.call("test_tool", {"test": "parameter"})
                        successful_calls += 1
                    except Exception as e:
                        failed_calls += 1
                        # Log error but continue
                        continue
                
                # Return with error information
                return Command(
                    goto="error_handler_agent",
                    update={
                        "tool_errors": tool_errors,
                        "successful_tool_calls": successful_calls,
                        "failed_tool_calls": failed_calls,
                        "error_count": state.get("error_count", 0) + failed_calls
                    }
                )
            
            mock_rag.execute_with_monitoring = AsyncMock(side_effect=mock_execute_with_errors)
            MockRAGAgent.return_value = mock_rag
            
            # Execute RAG agent with failing tools
            rag_agent = AgentFactory.create_agent("rag_agent", mock_system_config.agents["rag_agent"], mock_system_config.__dict__)
            result = await rag_agent.execute_with_monitoring(initial_state)
            
            # Verify error handling
            assert len(tool_errors) == 3
            assert result.goto == "error_handler_agent"
            assert result.update["failed_tool_calls"] == 3
            assert result.update["successful_tool_calls"] == 0
            assert result.update["error_count"] == 3
            
            # Verify different error types were captured
            error_types = [error["failure_type"] for error in tool_errors]
            assert "timeout" in error_types
            assert "connection" in error_types
            assert "invalid_response" in error_types
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tool_coordination_between_agents(self, mock_system_config, initial_state):
        """Test tool coordination and data sharing between agents."""
        tool_coordination_log = []
        shared_tool_data = {}
        
        # Mock coordinated tool usage
        class CoordinatedToolClient:
            def __init__(self, agent_name: str):
                self.agent_name = agent_name
            
            async def call(self, tool_name: str, parameters: Dict[str, Any]) -> str:
                call_info = {
                    "agent": self.agent_name,
                    "tool": tool_name,
                    "parameters": parameters,
                    "timestamp": "2024-01-01T00:00:00"
                }
                tool_coordination_log.append(call_info)
                
                # Store data for sharing between agents
                if tool_name == "generate_context":
                    context_data = f"Context generated by {self.agent_name}"
                    shared_tool_data["context"] = context_data
                    return context_data
                elif tool_name == "process_context":
                    context = shared_tool_data.get("context", "No context available")
                    processed = f"Processed by {self.agent_name}: {context}"
                    shared_tool_data["processed_context"] = processed
                    return processed
                elif tool_name == "finalize_output":
                    processed = shared_tool_data.get("processed_context", "No processed context")
                    final = f"Finalized by {self.agent_name}: {processed}"
                    shared_tool_data["final_output"] = final
                    return final
                
                return f"Tool {tool_name} response from {self.agent_name}"
        
        # Create agents with coordinated tool usage
        def create_coordinated_agent(agent_name: str, tool_sequence: List[str], next_agent: str):
            class CoordinatedAgent:
                def __init__(self):
                    self.name = agent_name
                    self.tool_client = CoordinatedToolClient(agent_name)
                
                async def execute_with_monitoring(self, state: VideoGenerationState) -> Command:
                    results = []
                    
                    for tool in tool_sequence:
                        result = await self.tool_client.call(tool, {"state": "current"})
                        results.append(result)
                    
                    return Command(
                        goto=next_agent,
                        update={
                            f"{agent_name}_results": results,
                            f"{agent_name}_tools_used": tool_sequence,
                            "shared_tool_data": shared_tool_data.copy(),
                            "current_agent": next_agent
                        }
                    )
            
            return CoordinatedAgent()
        
        # Create agent sequence with tool coordination
        agents = [
            ("planner_agent", ["generate_context"], "code_generator_agent"),
            ("code_generator_agent", ["process_context"], "renderer_agent"),
            ("renderer_agent", ["finalize_output"], "END")
        ]
        
        current_state = initial_state.copy()
        
        # Execute coordinated workflow
        for agent_name, tools, next_agent in agents:
            agent = create_coordinated_agent(agent_name, tools, next_agent)
            result = await agent.execute_with_monitoring(current_state)
            current_state.update(result.update)
            
            if next_agent == "END":
                break
        
        # Verify tool coordination
        assert len(tool_coordination_log) == 3
        
        # Verify data flow between agents
        assert "context" in shared_tool_data
        assert "processed_context" in shared_tool_data
        assert "final_output" in shared_tool_data
        
        # Verify each agent used tools in sequence
        assert tool_coordination_log[0]["agent"] == "planner_agent"
        assert tool_coordination_log[0]["tool"] == "generate_context"
        
        assert tool_coordination_log[1]["agent"] == "code_generator_agent"
        assert tool_coordination_log[1]["tool"] == "process_context"
        
        assert tool_coordination_log[2]["agent"] == "renderer_agent"
        assert tool_coordination_log[2]["tool"] == "finalize_output"
        
        # Verify final state contains all agent results
        assert "planner_agent_results" in current_state
        assert "code_generator_agent_results" in current_state
        assert "renderer_agent_results" in current_state
        
        # Verify data transformation chain
        final_output = shared_tool_data["final_output"]
        assert "Context generated by planner_agent" in final_output
        assert "Processed by code_generator_agent" in final_output
        assert "Finalized by renderer_agent" in final_output
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tool_performance_monitoring(self, mock_system_config, initial_state):
        """Test performance monitoring for external tools."""
        performance_metrics = []
        
        # Mock performance-tracked tool client
        class PerformanceTrackedClient:
            def __init__(self):
                self.call_count = 0
            
            async def call(self, tool_name: str, parameters: Dict[str, Any]) -> str:
                import time
                start_time = time.time()
                
                # Simulate different execution times
                execution_times = {
                    "fast_tool": 0.1,
                    "medium_tool": 0.5,
                    "slow_tool": 2.0
                }
                
                await asyncio.sleep(execution_times.get(tool_name, 0.1))
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                self.call_count += 1
                
                # Record performance metrics
                metrics = {
                    "tool": tool_name,
                    "execution_time": execution_time,
                    "call_number": self.call_count,
                    "parameters_size": len(str(parameters)),
                    "timestamp": "2024-01-01T00:00:00"
                }
                performance_metrics.append(metrics)
                
                return f"Response from {tool_name} (execution time: {execution_time:.2f}s)"
        
        # Test performance monitoring
        with patch('src.langgraph_agents.agents.monitoring_agent.MonitoringAgent') as MockMonitoringAgent:
            mock_monitoring = Mock()
            
            async def mock_execute_with_monitoring(state):
                client = PerformanceTrackedClient()
                
                # Call different tools with performance tracking
                tools = ["fast_tool", "medium_tool", "slow_tool", "fast_tool"]
                
                for tool in tools:
                    await client.call(tool, {"test": "data"})
                
                # Calculate performance statistics
                total_calls = len(performance_metrics)
                avg_execution_time = sum(m["execution_time"] for m in performance_metrics) / total_calls
                slowest_tool = max(performance_metrics, key=lambda x: x["execution_time"])
                fastest_tool = min(performance_metrics, key=lambda x: x["execution_time"])
                
                return Command(
                    goto="next_agent",
                    update={
                        "tool_performance_metrics": performance_metrics,
                        "performance_summary": {
                            "total_tool_calls": total_calls,
                            "average_execution_time": avg_execution_time,
                            "slowest_tool": slowest_tool,
                            "fastest_tool": fastest_tool
                        }
                    }
                )
            
            mock_monitoring.execute_with_monitoring = AsyncMock(side_effect=mock_execute_with_monitoring)
            MockMonitoringAgent.return_value = mock_monitoring
            
            # Execute monitoring agent
            monitoring_agent = AgentFactory.create_agent("monitoring_agent", 
                                                       AgentConfig("monitoring_agent", {}, ["performance_tool"]), 
                                                       mock_system_config.__dict__)
            result = await monitoring_agent.execute_with_monitoring(initial_state)
            
            # Verify performance monitoring
            assert len(performance_metrics) == 4
            assert result.update["performance_summary"]["total_tool_calls"] == 4
            
            # Verify different tools were tracked
            tool_names = [m["tool"] for m in performance_metrics]
            assert "fast_tool" in tool_names
            assert "medium_tool" in tool_names
            assert "slow_tool" in tool_names
            
            # Verify performance analysis
            summary = result.update["performance_summary"]
            assert summary["slowest_tool"]["tool"] == "slow_tool"
            assert summary["fastest_tool"]["tool"] == "fast_tool"
            assert summary["average_execution_time"] > 0


if __name__ == "__main__":
    pytest.main([__file__])
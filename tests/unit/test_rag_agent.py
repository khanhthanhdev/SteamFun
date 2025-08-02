"""
Unit tests for RAGAgent.
Tests RAG query generation, context retrieval, and caching functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from src.langgraph_agents.agents.rag_agent import RAGAgent
from src.langgraph_agents.state import VideoGenerationState, AgentConfig
from langgraph.types import Command

# Import test utilities
from tests.utils.config_mocks import (
    mock_configuration_manager, create_test_system_config, 
    create_test_agent_config, MockConfigurationManager
)


class TestRAGAgent:
    """Test suite for RAGAgent functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock agent configuration for RAGAgent using configuration system."""
        return create_test_agent_config(
            name="rag_agent",
            helper_model="openai/gpt-4o-mini",
            tools=["rag_tool", "vector_search_tool"]
        )
    
    @pytest.fixture
    def mock_system_config(self):
        """Create mock system configuration using configuration system."""
        return create_test_system_config()
    
    @pytest.fixture
    def mock_state(self):
        """Create mock video generation state."""
        return VideoGenerationState(
            messages=[],
            topic="Python programming basics",
            description="Educational video about Python fundamentals",
            session_id="test_session_123",
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
            scene_outline="# Scene 1\nIntroduction\n# Scene 2\nBasic syntax",
            scene_implementations={
                1: "Scene 1: Show Python logo and introduction text",
                2: "Scene 2: Display code examples with syntax highlighting"
            },
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
    
    @pytest.fixture
    def rag_agent(self, mock_config, mock_system_config):
        """Create RAGAgent instance for testing."""
        with mock_configuration_manager(mock_system_config):
            return RAGAgent(mock_config, mock_system_config)
    
    @pytest.fixture
    def mock_rag_integration(self):
        """Create mock RAG integration."""
        mock_rag = Mock()
        mock_rag.generate_rag_queries = AsyncMock(return_value=[
            "How to create text animations in Manim?",
            "Python syntax highlighting in Manim",
            "Manim scene construction basics"
        ])
        mock_rag.retrieve_context = AsyncMock(return_value={
            "query1": "Context about text animations...",
            "query2": "Context about syntax highlighting...",
            "query3": "Context about scene construction..."
        })
        mock_rag.is_cache_enabled = True
        mock_rag.cache_stats = {"hits": 5, "misses": 2}
        return mock_rag
    
    def test_rag_agent_initialization(self, rag_agent, mock_config):
        """Test RAGAgent initialization."""
        assert rag_agent.name == "rag_agent"
        assert rag_agent.helper_model == "openai/gpt-4o-mini"
        assert rag_agent._rag_integration is None  # Lazy initialization
        assert rag_agent._context7_client is None
        assert rag_agent._docling_processor is None
    
    @patch('src.langgraph_agents.agents.rag_agent.RAGIntegration')
    def test_get_rag_integration_creation(self, mock_rag_class, rag_agent, mock_state):
        """Test RAG integration creation with state configuration."""
        mock_instance = Mock()
        mock_rag_class.return_value = mock_instance
        
        with patch.object(rag_agent, 'get_model_wrapper') as mock_get_wrapper:
            mock_wrapper = Mock()
            mock_get_wrapper.return_value = mock_wrapper
            
            rag_integration = rag_agent._get_rag_integration(mock_state)
            
            # Verify RAG integration was created with correct configuration
            mock_rag_class.assert_called_once()
            call_kwargs = mock_rag_class.call_args[1]
            assert call_kwargs['chroma_db_path'] == 'test_chroma'
            assert call_kwargs['embedding_model'] == 'test_model'
            assert call_kwargs['enable_caching'] is True
            
            assert rag_integration == mock_instance
            assert rag_agent._rag_integration == mock_instance
    
    @pytest.mark.asyncio
    async def test_execute_success_full_workflow(self, rag_agent, mock_state, mock_rag_integration):
        """Test successful execution of RAG workflow."""
        # Set up state to request RAG context
        mock_state["rag_context"] = {}
        mock_state["current_agent"] = "rag_agent"
        
        with patch.object(rag_agent, '_get_rag_integration', return_value=mock_rag_integration):
            command = await rag_agent.execute(mock_state)
            
            # Verify RAG queries were generated
            mock_rag_integration.generate_rag_queries.assert_called_once()
            
            # Verify context retrieval was called
            mock_rag_integration.retrieve_context.assert_called_once()
            
            # Verify command structure
            assert command.goto == "code_generator_agent"  # Default return agent
            assert len(command.update["rag_context"]) == 3
            assert command.update["current_agent"] == "code_generator_agent"
    
    @pytest.mark.asyncio
    async def test_execute_with_requesting_agent(self, rag_agent, mock_state, mock_rag_integration):
        """Test execution with specific requesting agent."""
        mock_state["requesting_agent"] = "planner_agent"
        
        with patch.object(rag_agent, '_get_rag_integration', return_value=mock_rag_integration):
            command = await rag_agent.execute(mock_state)
            
            # Should return to requesting agent
            assert command.goto == "planner_agent"
            assert command.update["current_agent"] == "planner_agent"
    
    @pytest.mark.asyncio
    async def test_execute_rag_disabled(self, rag_agent, mock_state):
        """Test execution when RAG is disabled."""
        mock_state["use_rag"] = False
        
        command = await rag_agent.execute(mock_state)
        
        # Should skip RAG and return immediately
        assert command.goto == "code_generator_agent"
        assert command.update["rag_context"] == {}
    
    @pytest.mark.asyncio
    async def test_execute_query_generation_failure(self, rag_agent, mock_state, mock_rag_integration):
        """Test handling of RAG query generation failure."""
        mock_rag_integration.generate_rag_queries.side_effect = Exception("Query generation failed")
        
        with patch.object(rag_agent, '_get_rag_integration', return_value=mock_rag_integration):
            command = await rag_agent.execute(mock_state)
            
            # Should route to error handler
            assert command.goto == "error_handler_agent"
            assert command.update["error_count"] == 1
    
    @pytest.mark.asyncio
    async def test_execute_context_retrieval_failure(self, rag_agent, mock_state, mock_rag_integration):
        """Test handling of context retrieval failure."""
        mock_rag_integration.retrieve_context.side_effect = Exception("Context retrieval failed")
        
        with patch.object(rag_agent, '_get_rag_integration', return_value=mock_rag_integration):
            command = await rag_agent.execute(mock_state)
            
            # Should route to error handler
            assert command.goto == "error_handler_agent"
            assert command.update["error_count"] == 1
    
    @pytest.mark.asyncio
    async def test_generate_rag_queries_compatibility(self, rag_agent, mock_state, mock_rag_integration):
        """Test RAG query generation method compatibility."""
        with patch.object(rag_agent, '_get_rag_integration', return_value=mock_rag_integration):
            queries = await rag_agent.generate_rag_queries(
                topic="Test topic",
                description="Test description",
                context="Test context",
                state=mock_state
            )
            
            mock_rag_integration.generate_rag_queries.assert_called_once()
            call_kwargs = mock_rag_integration.generate_rag_queries.call_args[1]
            assert call_kwargs['topic'] == "Test topic"
            assert call_kwargs['description'] == "Test description"
            
            assert queries == [
                "How to create text animations in Manim?",
                "Python syntax highlighting in Manim",
                "Manim scene construction basics"
            ]
    
    @pytest.mark.asyncio
    async def test_retrieve_context_compatibility(self, rag_agent, mock_state, mock_rag_integration):
        """Test context retrieval method compatibility."""
        test_queries = ["query1", "query2"]
        
        with patch.object(rag_agent, '_get_rag_integration', return_value=mock_rag_integration):
            context = await rag_agent.retrieve_context(
                queries=test_queries,
                state=mock_state
            )
            
            mock_rag_integration.retrieve_context.assert_called_once_with(
                queries=test_queries
            )
            
            assert len(context) == 3
    
    @pytest.mark.asyncio
    async def test_context7_integration(self, rag_agent, mock_state):
        """Test Context7 integration for enhanced documentation."""
        mock_context7 = Mock()
        mock_context7.query_library = AsyncMock(return_value="Context7 documentation content")
        
        with patch.object(rag_agent, '_get_context7_client', return_value=mock_context7):
            context = await rag_agent._query_context7(
                library_id="manim/manim",
                query="text animations",
                state=mock_state
            )
            
            mock_context7.query_library.assert_called_once_with(
                library_id="manim/manim",
                query="text animations"
            )
            
            assert context == "Context7 documentation content"
    
    @pytest.mark.asyncio
    async def test_docling_integration(self, rag_agent, mock_state):
        """Test Docling integration for document processing."""
        mock_docling = Mock()
        mock_docling.process_document = AsyncMock(return_value={
            "content": "Processed document content",
            "metadata": {"type": "pdf", "pages": 10}
        })
        
        with patch.object(rag_agent, '_get_docling_processor', return_value=mock_docling):
            result = await rag_agent._process_document(
                document_path="test_document.pdf",
                state=mock_state
            )
            
            mock_docling.process_document.assert_called_once_with("test_document.pdf")
            
            assert result["content"] == "Processed document content"
            assert result["metadata"]["type"] == "pdf"
    
    def test_get_rag_status(self, rag_agent, mock_state):
        """Test RAG status reporting."""
        mock_state.update({
            "rag_context": {"query1": "context1", "query2": "context2"},
            "use_enhanced_rag": True,
            "enable_rag_caching": True
        })
        
        status = rag_agent.get_rag_status(mock_state)
        
        assert status["agent_name"] == "rag_agent"
        assert status["rag_enabled"] is True
        assert status["enhanced_rag_enabled"] is True
        assert status["caching_enabled"] is True
        assert status["context_queries_count"] == 2
    
    def test_get_cache_statistics(self, rag_agent, mock_rag_integration):
        """Test cache statistics retrieval."""
        with patch.object(rag_agent, '_get_rag_integration', return_value=mock_rag_integration):
            stats = rag_agent.get_cache_statistics()
            
            assert stats["cache_enabled"] is True
            assert stats["cache_hits"] == 5
            assert stats["cache_misses"] == 2
            assert stats["hit_rate"] == 5 / 7  # 5 hits out of 7 total
    
    @pytest.mark.asyncio
    async def test_clear_cache(self, rag_agent, mock_rag_integration):
        """Test cache clearing functionality."""
        mock_rag_integration.clear_cache = AsyncMock()
        
        with patch.object(rag_agent, '_get_rag_integration', return_value=mock_rag_integration):
            await rag_agent.clear_cache()
            
            mock_rag_integration.clear_cache.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_vector_store(self, rag_agent, mock_state, mock_rag_integration):
        """Test vector store update functionality."""
        mock_rag_integration.update_vector_store = AsyncMock()
        
        with patch.object(rag_agent, '_get_rag_integration', return_value=mock_rag_integration):
            await rag_agent.update_vector_store(
                documents=["doc1", "doc2"],
                state=mock_state
            )
            
            mock_rag_integration.update_vector_store.assert_called_once_with(
                documents=["doc1", "doc2"]
            )
    
    def test_validate_rag_queries(self, rag_agent):
        """Test RAG query validation."""
        valid_queries = [
            "How to create animations in Manim?",
            "Python syntax examples",
            "Scene construction patterns"
        ]
        
        invalid_queries = [
            "",  # Empty query
            "a",  # Too short
            "x" * 1000  # Too long
        ]
        
        assert rag_agent._validate_rag_queries(valid_queries) is True
        assert rag_agent._validate_rag_queries(invalid_queries) is False
    
    def test_filter_relevant_context(self, rag_agent):
        """Test context relevance filtering."""
        context = {
            "query1": "Highly relevant Manim animation content with specific details",
            "query2": "Somewhat relevant content",
            "query3": "Not relevant at all",
            "query4": "Very relevant Python and Manim specific information"
        }
        
        filtered = rag_agent._filter_relevant_context(
            context, 
            topic="Manim animations",
            relevance_threshold=0.5
        )
        
        # Should keep highly relevant content
        assert "query1" in filtered
        assert "query4" in filtered
        # May filter out less relevant content based on scoring
        assert len(filtered) <= len(context)
    
    def test_merge_context_sources(self, rag_agent):
        """Test merging context from multiple sources."""
        rag_context = {"rag_query1": "RAG content 1"}
        context7_context = {"context7_query1": "Context7 content 1"}
        docling_context = {"docling_doc1": "Docling content 1"}
        
        merged = rag_agent._merge_context_sources(
            rag_context=rag_context,
            context7_context=context7_context,
            docling_context=docling_context
        )
        
        assert "rag_query1" in merged
        assert "context7_query1" in merged
        assert "docling_doc1" in merged
        assert len(merged) == 3
    
    @pytest.mark.asyncio
    async def test_handle_rag_error_with_fallback(self, rag_agent, mock_state):
        """Test RAG error handling with fallback strategy."""
        error = Exception("Vector store connection failed")
        
        with patch.object(rag_agent, '_try_fallback_retrieval') as mock_fallback:
            mock_fallback.return_value = {"fallback_query": "fallback context"}
            
            command = await rag_agent.handle_rag_error(
                error, mock_state, use_fallback=True
            )
            
            mock_fallback.assert_called_once()
            assert command.goto == "code_generator_agent"
            assert "fallback_query" in command.update["rag_context"]
    
    @pytest.mark.asyncio
    async def test_handle_rag_error_no_fallback(self, rag_agent, mock_state):
        """Test RAG error handling without fallback."""
        error = Exception("RAG system failure")
        
        with patch.object(rag_agent, 'handle_error') as mock_handle_error:
            mock_handle_error.return_value = Command(goto="error_handler_agent")
            
            command = await rag_agent.handle_rag_error(
                error, mock_state, use_fallback=False
            )
            
            mock_handle_error.assert_called_once_with(error, mock_state)
            assert command.goto == "error_handler_agent"
    
    def test_cleanup_on_destruction(self, rag_agent):
        """Test resource cleanup when agent is destroyed."""
        mock_rag = Mock()
        mock_context7 = Mock()
        mock_docling = Mock()
        
        rag_agent._rag_integration = mock_rag
        rag_agent._context7_client = mock_context7
        rag_agent._docling_processor = mock_docling
        
        # Mock cleanup methods
        mock_rag.cleanup = Mock()
        mock_context7.close = Mock()
        mock_docling.cleanup = Mock()
        
        # Trigger destructor
        rag_agent.__del__()
        
        mock_rag.cleanup.assert_called_once()
        mock_context7.close.assert_called_once()
        mock_docling.cleanup.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
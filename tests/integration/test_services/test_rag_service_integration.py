"""
Integration tests for RAGService.
Tests integration with vector stores, embedding models, and document processing.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile

from app.services.rag_service import RAGService
from app.models.schemas.rag import RAGQueryRequest, RAGQueryResponse
from app.models.enums import EmbeddingModel


class TestRAGServiceIntegration:
    """Integration test suite for RAGService."""
    
    @pytest.fixture
    def rag_service(self):
        """Create RAGService instance for integration tests."""
        return RAGService()
    
    @pytest.fixture
    def temp_docs_dir(self):
        """Create temporary documents directory for integration tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample documents
            doc1_path = Path(temp_dir) / "doc1.txt"
            doc1_path.write_text("This is a sample document about integration testing.")
            
            doc2_path = Path(temp_dir) / "doc2.md"
            doc2_path.write_text("# Integration Testing\nThis document covers integration testing concepts.")
            
            yield temp_dir
    
    @pytest.fixture
    def sample_rag_request(self):
        """Create sample RAG query request for integration tests."""
        return RAGQueryRequest(
            query="What is integration testing?",
            context="software testing",
            max_results=5,
            confidence_threshold=0.7,
            include_sources=True
        )
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_rag_service_vector_store_integration(self, rag_service, temp_docs_dir, sample_rag_request):
        """Test RAGService integration with vector store."""
        with patch('app.services.rag_service.ChromaVectorStore') as mock_vector_store:
            mock_store = Mock()
            mock_store.add_documents = AsyncMock()
            mock_store.similarity_search = AsyncMock(return_value=[
                {
                    "content": "Integration testing is a type of software testing",
                    "metadata": {"source": "doc1.txt", "score": 0.85}
                },
                {
                    "content": "Integration testing covers testing concepts",
                    "metadata": {"source": "doc2.md", "score": 0.78}
                }
            ])
            mock_vector_store.return_value = mock_store
            
            # Test document indexing
            await rag_service.index_documents(temp_docs_dir)
            
            # Verify documents were added to vector store
            mock_store.add_documents.assert_called()
            
            # Test query processing
            response = await rag_service.query_documents(sample_rag_request)
            
            # Verify vector store was queried
            mock_store.similarity_search.assert_called_once()
            
            # Verify response structure
            assert isinstance(response, RAGQueryResponse)
            assert response.sources is not None
            assert len(response.sources) > 0
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_rag_service_embedding_integration(self, rag_service, sample_rag_request):
        """Test RAGService integration with embedding models."""
        with patch('app.services.rag_service.EmbeddingService') as mock_embedding_service:
            mock_embeddings = Mock()
            mock_embeddings.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4])
            mock_embeddings.embed_documents = AsyncMock(return_value=[
                [0.1, 0.2, 0.3, 0.4],
                [0.2, 0.3, 0.4, 0.5]
            ])
            mock_embedding_service.return_value = mock_embeddings
            
            # Test query embedding
            query_embedding = await rag_service._embed_query(sample_rag_request.query)
            
            # Verify embedding service was called
            mock_embeddings.embed_query.assert_called_once_with(sample_rag_request.query)
            assert query_embedding == [0.1, 0.2, 0.3, 0.4]
            
            # Test document embedding
            documents = ["Test document 1", "Test document 2"]
            doc_embeddings = await rag_service._embed_documents(documents)
            
            mock_embeddings.embed_documents.assert_called_once_with(documents)
            assert len(doc_embeddings) == 2
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_rag_service_document_processing_integration(self, rag_service, temp_docs_dir):
        """Test RAGService integration with document processing."""
        with patch('app.services.rag_service.DocumentProcessor') as mock_processor:
            mock_doc_processor = Mock()
            mock_doc_processor.process_document = AsyncMock(return_value={
                "chunks": [
                    {"content": "Chunk 1", "metadata": {"page": 1}},
                    {"content": "Chunk 2", "metadata": {"page": 2}}
                ],
                "metadata": {"title": "Test Document", "pages": 2}
            })
            mock_processor.return_value = mock_doc_processor
            
            # Test document processing
            doc_path = Path(temp_docs_dir) / "doc1.txt"
            processed_doc = await rag_service._process_document(str(doc_path))
            
            # Verify document processor was called
            mock_doc_processor.process_document.assert_called_once_with(str(doc_path))
            
            # Verify processed document structure
            assert "chunks" in processed_doc
            assert "metadata" in processed_doc
            assert len(processed_doc["chunks"]) == 2
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_rag_service_caching_integration(self, rag_service, sample_rag_request):
        """Test RAGService integration with caching systems."""
        with patch('app.services.rag_service.CacheService') as mock_cache_service:
            mock_cache = Mock()
            mock_cache.get = AsyncMock(return_value=None)  # Cache miss
            mock_cache.set = AsyncMock()
            mock_cache_service.return_value = mock_cache
            
            with patch.object(rag_service, '_perform_rag_query') as mock_query:
                mock_response = RAGQueryResponse(
                    answer="Cached answer",
                    sources=["source1.txt"],
                    confidence=0.8,
                    context="Cached context"
                )
                mock_query.return_value = mock_response
                
                # Test query with caching
                response = await rag_service.query_documents_with_cache(sample_rag_request)
                
                # Verify cache was checked and updated
                mock_cache.get.assert_called_once()
                mock_cache.set.assert_called_once()
                assert response.answer == "Cached answer"
                
                # Test cache hit
                mock_cache.get.return_value = mock_response
                
                cached_response = await rag_service.query_documents_with_cache(sample_rag_request)
                assert cached_response.answer == "Cached answer"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_rag_service_database_integration(self, rag_service, sample_rag_request):
        """Test RAGService integration with database."""
        with patch('app.services.rag_service.get_database_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value = mock_session
            
            # Mock database operations
            mock_session.add = Mock()
            mock_session.commit = Mock()
            mock_session.query = Mock()
            
            # Test query logging to database
            response = await rag_service.query_documents(sample_rag_request)
            
            # Verify database interactions for query logging
            mock_session.add.assert_called()
            mock_session.commit.assert_called()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_rag_service_concurrent_queries_integration(self, rag_service):
        """Test RAGService handling of concurrent queries."""
        # Create multiple query requests
        requests = [
            RAGQueryRequest(
                query=f"Test query {i}",
                context="test context",
                max_results=3
            )
            for i in range(5)
        ]
        
        with patch.object(rag_service, '_perform_rag_query') as mock_query:
            mock_responses = [
                RAGQueryResponse(
                    answer=f"Answer {i}",
                    sources=[f"source{i}.txt"],
                    confidence=0.8
                )
                for i in range(5)
            ]
            mock_query.side_effect = mock_responses
            
            # Execute concurrent queries
            tasks = [
                asyncio.create_task(rag_service.query_documents(request))
                for request in requests
            ]
            
            responses = await asyncio.gather(*tasks)
            
            # Verify all queries were processed
            assert len(responses) == 5
            for i, response in enumerate(responses):
                assert response.answer == f"Answer {i}"
            
            # Verify all queries were executed
            assert mock_query.call_count == 5
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_rag_service_error_recovery_integration(self, rag_service, sample_rag_request):
        """Test RAGService error recovery integration."""
        with patch.object(rag_service, '_perform_rag_query') as mock_query:
            # First call fails, second succeeds
            mock_query.side_effect = [
                Exception("Vector store unavailable"),
                RAGQueryResponse(
                    answer="Recovered answer",
                    sources=["source.txt"],
                    confidence=0.7
                )
            ]
            
            with patch.object(rag_service, '_handle_query_error') as mock_error_handler:
                mock_error_handler.return_value = True  # Indicates retry should happen
                
                with patch.object(rag_service, '_retry_query') as mock_retry:
                    mock_retry.return_value = RAGQueryResponse(
                        answer="Recovered answer",
                        sources=["source.txt"],
                        confidence=0.7
                    )
                    
                    # Process query with error recovery
                    try:
                        response = await rag_service.query_documents_with_retry(sample_rag_request)
                        assert response.answer == "Recovered answer"
                    except Exception:
                        pass  # Expected on first attempt
                    
                    # Verify error handling was triggered
                    mock_error_handler.assert_called()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_rag_service_monitoring_integration(self, rag_service, sample_rag_request):
        """Test RAGService integration with monitoring systems."""
        with patch('app.services.rag_service.MonitoringService') as mock_monitoring:
            mock_monitor = Mock()
            mock_monitor.record_query = Mock()
            mock_monitor.record_response_time = Mock()
            mock_monitor.record_error = Mock()
            mock_monitoring.return_value = mock_monitor
            
            # Test query with monitoring
            response = await rag_service.query_documents(sample_rag_request)
            
            # Verify monitoring was called
            mock_monitor.record_query.assert_called_once()
            mock_monitor.record_response_time.assert_called_once()
            
            # Test error monitoring
            with patch.object(rag_service, '_perform_rag_query') as mock_query:
                mock_query.side_effect = Exception("Query failed")
                
                try:
                    await rag_service.query_documents(sample_rag_request)
                except Exception:
                    pass
                
                # Verify error was recorded
                mock_monitor.record_error.assert_called()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_rag_service_context_enhancement_integration(self, rag_service, sample_rag_request):
        """Test RAGService integration with context enhancement."""
        with patch('app.services.rag_service.ContextEnhancementService') as mock_context_service:
            mock_context = Mock()
            mock_context.enhance_context = AsyncMock(return_value={
                "enhanced_query": "Enhanced: What is integration testing?",
                "additional_context": "Software testing methodology",
                "keywords": ["integration", "testing", "software"]
            })
            mock_context_service.return_value = mock_context
            
            # Test context enhancement
            enhanced_request = await rag_service._enhance_query_context(sample_rag_request)
            
            # Verify context enhancement was applied
            mock_context.enhance_context.assert_called_once()
            assert "enhanced_query" in enhanced_request
            assert "additional_context" in enhanced_request
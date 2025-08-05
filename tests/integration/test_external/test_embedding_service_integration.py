"""
Integration tests for embedding service integrations.
Tests OpenAI, Sentence Transformers, and other embedding service integrations.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import numpy as np

from app.services.rag_service import RAGService
from app.models.enums import EmbeddingModel


class TestEmbeddingServiceIntegration:
    """Integration test suite for embedding service integrations."""
    
    @pytest.fixture
    def rag_service(self):
        """Create RAGService instance for integration tests."""
        return RAGService()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_openai_embedding_integration(self, rag_service):
        """Test OpenAI embedding service integration."""
        with patch('app.services.rag_service.OpenAIEmbeddings') as mock_openai:
            mock_embeddings = Mock()
            mock_embeddings.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4])
            mock_embeddings.embed_documents = AsyncMock(return_value=[
                [0.1, 0.2, 0.3, 0.4],
                [0.2, 0.3, 0.4, 0.5]
            ])
            mock_openai.return_value = mock_embeddings
            
            # Test query embedding
            query = "What is integration testing?"
            embedding = await rag_service._embed_query_openai(query)
            
            # Verify OpenAI service was called
            mock_embeddings.embed_query.assert_called_once_with(query)
            assert embedding == [0.1, 0.2, 0.3, 0.4]
            
            # Test document embedding
            documents = ["Document 1 content", "Document 2 content"]
            doc_embeddings = await rag_service._embed_documents_openai(documents)
            
            mock_embeddings.embed_documents.assert_called_once_with(documents)
            assert len(doc_embeddings) == 2
            assert doc_embeddings[0] == [0.1, 0.2, 0.3, 0.4]
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_sentence_transformers_integration(self, rag_service):
        """Test Sentence Transformers embedding service integration."""
        with patch('app.services.rag_service.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.encode = Mock(return_value=np.array([
                [0.1, 0.2, 0.3, 0.4],
                [0.2, 0.3, 0.4, 0.5]
            ]))
            mock_st.return_value = mock_model
            
            # Test sentence transformer embedding
            texts = ["Text 1", "Text 2"]
            embeddings = await rag_service._embed_texts_sentence_transformer(texts)
            
            # Verify Sentence Transformer was called
            mock_model.encode.assert_called_once_with(texts)
            assert len(embeddings) == 2
            assert embeddings[0].tolist() == [0.1, 0.2, 0.3, 0.4]
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_embedding_model_selection_integration(self, rag_service):
        """Test embedding model selection integration."""
        # Test different embedding models
        models_to_test = [
            (EmbeddingModel.OPENAI_ADA, "openai"),
            (EmbeddingModel.SENTENCE_TRANSFORMERS, "sentence_transformers"),
            (EmbeddingModel.HUGGINGFACE, "huggingface")
        ]
        
        for model_enum, model_type in models_to_test:
            with patch(f'app.services.rag_service.{model_type}_embed') as mock_embed:
                mock_embed.return_value = [0.1, 0.2, 0.3, 0.4]
                
                # Test model selection
                embedding = await rag_service._embed_with_model("test text", model_enum)
                
                # Verify correct model was used
                mock_embed.assert_called_once_with("test text")
                assert embedding == [0.1, 0.2, 0.3, 0.4]
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_embedding_caching_integration(self, rag_service):
        """Test embedding caching integration."""
        with patch('app.services.rag_service.CacheService') as mock_cache_service:
            mock_cache = Mock()
            mock_cache.get = AsyncMock(return_value=None)  # Cache miss
            mock_cache.set = AsyncMock()
            mock_cache_service.return_value = mock_cache
            
            with patch.object(rag_service, '_compute_embedding') as mock_compute:
                mock_compute.return_value = [0.1, 0.2, 0.3, 0.4]
                
                # Test embedding with caching
                text = "Test text for caching"
                embedding = await rag_service._embed_with_cache(text, EmbeddingModel.OPENAI_ADA)
                
                # Verify cache was checked and updated
                mock_cache.get.assert_called_once()
                mock_cache.set.assert_called_once()
                assert embedding == [0.1, 0.2, 0.3, 0.4]
                
                # Test cache hit
                mock_cache.get.return_value = [0.1, 0.2, 0.3, 0.4]
                
                cached_embedding = await rag_service._embed_with_cache(text, EmbeddingModel.OPENAI_ADA)
                assert cached_embedding == [0.1, 0.2, 0.3, 0.4]
                
                # Verify compute was not called again
                assert mock_compute.call_count == 1
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_embedding_batch_processing_integration(self, rag_service):
        """Test embedding batch processing integration."""
        with patch('app.services.rag_service.OpenAIEmbeddings') as mock_openai:
            mock_embeddings = Mock()
            
            # Mock batch embedding response
            batch_embeddings = [
                [0.1, 0.2, 0.3, 0.4],
                [0.2, 0.3, 0.4, 0.5],
                [0.3, 0.4, 0.5, 0.6],
                [0.4, 0.5, 0.6, 0.7],
                [0.5, 0.6, 0.7, 0.8]
            ]
            mock_embeddings.embed_documents = AsyncMock(return_value=batch_embeddings)
            mock_openai.return_value = mock_embeddings
            
            # Test batch processing
            documents = [f"Document {i}" for i in range(5)]
            embeddings = await rag_service._embed_documents_batch(documents, batch_size=3)
            
            # Verify batch processing
            assert len(embeddings) == 5
            assert embeddings[0] == [0.1, 0.2, 0.3, 0.4]
            assert embeddings[4] == [0.5, 0.6, 0.7, 0.8]
            
            # Verify batching was used (should be called with batches)
            mock_embeddings.embed_documents.assert_called()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_embedding_error_handling_integration(self, rag_service):
        """Test embedding service error handling integration."""
        with patch('app.services.rag_service.OpenAIEmbeddings') as mock_openai:
            mock_embeddings = Mock()
            mock_embeddings.embed_query.side_effect = Exception("OpenAI API error")
            mock_openai.return_value = mock_embeddings
            
            # Test error handling with fallback
            with patch.object(rag_service, '_embed_with_fallback') as mock_fallback:
                mock_fallback.return_value = [0.1, 0.2, 0.3, 0.4]
                
                embedding = await rag_service._embed_query_with_retry("test query")
                
                # Verify fallback was used
                mock_fallback.assert_called_once()
                assert embedding == [0.1, 0.2, 0.3, 0.4]
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_embedding_performance_monitoring_integration(self, rag_service):
        """Test embedding performance monitoring integration."""
        with patch('app.services.rag_service.MonitoringService') as mock_monitoring:
            mock_monitor = Mock()
            mock_monitor.record_embedding_time = Mock()
            mock_monitor.record_embedding_cost = Mock()
            mock_monitoring.return_value = mock_monitor
            
            with patch.object(rag_service, '_compute_embedding') as mock_compute:
                mock_compute.return_value = [0.1, 0.2, 0.3, 0.4]
                
                # Test embedding with monitoring
                embedding = await rag_service._embed_with_monitoring("test text", EmbeddingModel.OPENAI_ADA)
                
                # Verify monitoring was recorded
                mock_monitor.record_embedding_time.assert_called_once()
                mock_monitor.record_embedding_cost.assert_called_once()
                assert embedding == [0.1, 0.2, 0.3, 0.4]
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_embedding_concurrent_processing_integration(self, rag_service):
        """Test concurrent embedding processing integration."""
        with patch('app.services.rag_service.OpenAIEmbeddings') as mock_openai:
            mock_embeddings = Mock()
            
            # Mock concurrent embedding responses
            embedding_responses = [
                [i * 0.1, i * 0.2, i * 0.3, i * 0.4]
                for i in range(1, 6)
            ]
            mock_embeddings.embed_query.side_effect = embedding_responses
            mock_openai.return_value = mock_embeddings
            
            # Test concurrent embedding
            queries = [f"Query {i}" for i in range(5)]
            
            tasks = [
                asyncio.create_task(rag_service._embed_query_openai(query))
                for query in queries
            ]
            
            embeddings = await asyncio.gather(*tasks)
            
            # Verify concurrent processing
            assert len(embeddings) == 5
            for i, embedding in enumerate(embeddings):
                expected = [(i+1) * 0.1, (i+1) * 0.2, (i+1) * 0.3, (i+1) * 0.4]
                assert embedding == expected
            
            # Verify all queries were processed
            assert mock_embeddings.embed_query.call_count == 5
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_embedding_quality_validation_integration(self, rag_service):
        """Test embedding quality validation integration."""
        with patch('app.services.rag_service.EmbeddingValidator') as mock_validator:
            mock_val = Mock()
            mock_val.validate_embedding = Mock(return_value={
                "is_valid": True,
                "quality_score": 0.85,
                "dimension_check": True,
                "norm_check": True
            })
            mock_validator.return_value = mock_val
            
            with patch.object(rag_service, '_compute_embedding') as mock_compute:
                mock_compute.return_value = [0.1, 0.2, 0.3, 0.4]
                
                # Test embedding with validation
                result = await rag_service._embed_with_validation("test text", EmbeddingModel.OPENAI_ADA)
                
                # Verify validation was performed
                mock_val.validate_embedding.assert_called_once()
                
                # Verify result structure
                assert result["embedding"] == [0.1, 0.2, 0.3, 0.4]
                assert result["validation"]["is_valid"] is True
                assert result["validation"]["quality_score"] == 0.85
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_embedding_model_comparison_integration(self, rag_service):
        """Test embedding model comparison integration."""
        # Mock different model responses
        model_responses = {
            EmbeddingModel.OPENAI_ADA: [0.1, 0.2, 0.3, 0.4],
            EmbeddingModel.SENTENCE_TRANSFORMERS: [0.2, 0.3, 0.4, 0.5],
            EmbeddingModel.HUGGINGFACE: [0.3, 0.4, 0.5, 0.6]
        }
        
        with patch.object(rag_service, '_embed_with_model') as mock_embed:
            def mock_embed_side_effect(text, model):
                return model_responses[model]
            
            mock_embed.side_effect = mock_embed_side_effect
            
            # Test model comparison
            text = "Test text for comparison"
            models = [EmbeddingModel.OPENAI_ADA, EmbeddingModel.SENTENCE_TRANSFORMERS, EmbeddingModel.HUGGINGFACE]
            
            comparison_results = await rag_service._compare_embedding_models(text, models)
            
            # Verify all models were tested
            assert len(comparison_results) == 3
            assert mock_embed.call_count == 3
            
            # Verify results structure
            for model, result in comparison_results.items():
                assert "embedding" in result
                assert "model" in result
                assert result["embedding"] == model_responses[model]
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_embedding_dimension_consistency_integration(self, rag_service):
        """Test embedding dimension consistency integration."""
        with patch('app.services.rag_service.OpenAIEmbeddings') as mock_openai:
            mock_embeddings = Mock()
            
            # Mock embeddings with consistent dimensions
            consistent_embeddings = [
                [0.1, 0.2, 0.3, 0.4],  # 4 dimensions
                [0.2, 0.3, 0.4, 0.5],  # 4 dimensions
                [0.3, 0.4, 0.5, 0.6]   # 4 dimensions
            ]
            mock_embeddings.embed_documents.return_value = consistent_embeddings
            mock_openai.return_value = mock_embeddings
            
            # Test dimension consistency
            documents = ["Doc 1", "Doc 2", "Doc 3"]
            embeddings = await rag_service._embed_documents_with_validation(documents)
            
            # Verify dimension consistency
            dimensions = [len(emb) for emb in embeddings]
            assert all(dim == 4 for dim in dimensions)
            assert len(set(dimensions)) == 1  # All dimensions are the same
            
            # Verify embeddings were generated
            mock_embeddings.embed_documents.assert_called_once_with(documents)
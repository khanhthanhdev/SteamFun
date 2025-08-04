"""
Integration tests for RAG API endpoints.
Tests complete RAG query workflow through API endpoints.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.main import app
from app.models.schemas.rag import RAGQueryRequest, RAGQueryResponse
from app.services.rag_service import RAGService


class TestRAGAPIIntegration:
    """Integration test suite for RAG API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client for API integration tests."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_rag_service(self):
        """Create mock RAG service for integration tests."""
        service = Mock(spec=RAGService)
        service.query_documents = AsyncMock()
        service.add_document = AsyncMock()
        service.get_context = AsyncMock()
        return service
    
    @pytest.mark.integration
    def test_rag_query_workflow_integration(self, client, mock_rag_service):
        """Test complete RAG query workflow through API."""
        # Mock RAG service responses
        mock_rag_service.query_documents.return_value = RAGQueryResponse(
            answer="Integration test answer",
            sources=["source1.pdf", "source2.md"],
            confidence=0.85,
            context="Retrieved context for integration test"
        )
        
        with patch('app.api.v1.endpoints.rag.get_rag_service', return_value=mock_rag_service):
            # Test RAG query
            query_request = {
                "query": "What is integration testing?",
                "context": "software testing",
                "max_results": 5,
                "confidence_threshold": 0.7
            }
            
            response = client.post("/api/v1/rag/query", json=query_request)
            assert response.status_code == 200
            
            data = response.json()
            assert data["answer"] == "Integration test answer"
            assert data["sources"] == ["source1.pdf", "source2.md"]
            assert data["confidence"] == 0.85
            
            # Verify service was called correctly
            mock_rag_service.query_documents.assert_called_once()
    
    @pytest.mark.integration
    def test_rag_document_management_integration(self, client, mock_rag_service):
        """Test RAG document management through API."""
        # Mock document addition
        mock_rag_service.add_document.return_value = {
            "document_id": "doc_123",
            "status": "processed",
            "chunks": 5
        }
        
        with patch('app.api.v1.endpoints.rag.get_rag_service', return_value=mock_rag_service):
            # Test document upload
            document_data = {
                "content": "Integration test document content",
                "metadata": {
                    "title": "Test Document",
                    "source": "integration_test"
                }
            }
            
            response = client.post("/api/v1/rag/documents", json=document_data)
            assert response.status_code == 201
            
            data = response.json()
            assert data["document_id"] == "doc_123"
            assert data["status"] == "processed"
            
            # Verify service was called
            mock_rag_service.add_document.assert_called_once()
    
    @pytest.mark.integration
    def test_rag_api_error_handling_integration(self, client, mock_rag_service):
        """Test RAG API error handling integration."""
        # Mock service errors
        mock_rag_service.query_documents.side_effect = Exception("RAG service unavailable")
        
        with patch('app.api.v1.endpoints.rag.get_rag_service', return_value=mock_rag_service):
            query_request = {
                "query": "Test query",
                "context": "test context"
            }
            
            response = client.post("/api/v1/rag/query", json=query_request)
            assert response.status_code == 500
    
    @pytest.mark.integration
    def test_rag_api_validation_integration(self, client):
        """Test RAG API request validation integration."""
        # Test missing required fields
        invalid_request = {
            "context": "Missing query field"
        }
        
        response = client.post("/api/v1/rag/query", json=invalid_request)
        assert response.status_code == 422
        
        error_data = response.json()
        assert "detail" in error_data
        assert any("query" in str(error) for error in error_data["detail"])
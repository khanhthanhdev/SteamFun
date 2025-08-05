"""
Integration tests for RAG database operations.
Tests database models, relationships, and data persistence for RAG system.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.database.base import Base
from app.models.database.rag import RAGQuery, RAGDocument, RAGEmbedding
from app.models.enums import EmbeddingModel


class TestRAGDatabaseIntegration:
    """Integration test suite for RAG database operations."""
    
    @pytest.fixture
    def test_engine(self):
        """Create test database engine."""
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)
        return engine
    
    @pytest.fixture
    def test_session(self, test_engine):
        """Create test database session."""
        TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()
    
    @pytest.mark.integration
    def test_rag_query_creation_integration(self, test_session):
        """Test RAG query record creation and persistence."""
        # Create RAG query record
        query = RAGQuery(
            query_id="rag_query_123",
            query_text="What is integration testing?",
            context="software testing",
            response_text="Integration testing is a type of software testing...",
            confidence_score=0.85,
            sources=["doc1.txt", "doc2.md"],
            embedding_model=EmbeddingModel.OPENAI_ADA,
            created_at=datetime.now()
        )
        
        # Save to database
        test_session.add(query)
        test_session.commit()
        
        # Verify query was saved
        saved_query = test_session.query(RAGQuery).filter_by(query_id="rag_query_123").first()
        assert saved_query is not None
        assert saved_query.query_text == "What is integration testing?"
        assert saved_query.confidence_score == 0.85
        assert saved_query.sources == ["doc1.txt", "doc2.md"]
    
    @pytest.mark.integration
    def test_rag_document_creation_integration(self, test_session):
        """Test RAG document record creation and persistence."""
        # Create RAG document record
        document = RAGDocument(
            document_id="rag_doc_123",
            title="Integration Testing Guide",
            content="This document covers integration testing concepts and practices...",
            metadata={
                "author": "Test Author",
                "category": "testing",
                "tags": ["integration", "testing", "software"]
            },
            source_path="/docs/integration_testing.md",
            chunk_count=5,
            indexed_at=datetime.now()
        )
        
        # Save to database
        test_session.add(document)
        test_session.commit()
        
        # Verify document was saved
        saved_document = test_session.query(RAGDocument).filter_by(document_id="rag_doc_123").first()
        assert saved_document is not None
        assert saved_document.title == "Integration Testing Guide"
        assert saved_document.chunk_count == 5
        assert saved_document.metadata["author"] == "Test Author"
        assert saved_document.metadata["tags"] == ["integration", "testing", "software"]
    
    @pytest.mark.integration
    def test_rag_embedding_relationship_integration(self, test_session):
        """Test RAG document and embedding relationship integration."""
        # Create document with embeddings
        document = RAGDocument(
            document_id="embedding_test_doc",
            title="Test Document",
            content="Test content for embedding",
            chunk_count=2
        )
        
        embeddings = [
            RAGEmbedding(
                embedding_id="embed_1",
                document_id="embedding_test_doc",
                chunk_index=0,
                chunk_text="First chunk of test content",
                embedding_vector=[0.1, 0.2, 0.3, 0.4],
                embedding_model=EmbeddingModel.OPENAI_ADA
            ),
            RAGEmbedding(
                embedding_id="embed_2",
                document_id="embedding_test_doc",
                chunk_index=1,
                chunk_text="Second chunk of test content",
                embedding_vector=[0.2, 0.3, 0.4, 0.5],
                embedding_model=EmbeddingModel.OPENAI_ADA
            )
        ]
        
        # Save all records
        test_session.add(document)
        for embedding in embeddings:
            test_session.add(embedding)
        test_session.commit()
        
        # Verify relationship
        saved_document = test_session.query(RAGDocument).filter_by(document_id="embedding_test_doc").first()
        assert saved_document.embeddings is not None
        assert len(saved_document.embeddings) == 2
        assert saved_document.embeddings[0].chunk_text == "First chunk of test content"
        assert saved_document.embeddings[1].embedding_vector == [0.2, 0.3, 0.4, 0.5]
    
    @pytest.mark.integration
    def test_rag_query_operations_integration(self, test_session):
        """Test RAG query operations."""
        # Create multiple RAG queries
        queries = [
            RAGQuery(
                query_id=f"query_test_{i}",
                query_text=f"Test query {i}",
                response_text=f"Test response {i}",
                confidence_score=0.8 + (i * 0.02),
                embedding_model=EmbeddingModel.OPENAI_ADA if i % 2 == 0 else EmbeddingModel.SENTENCE_TRANSFORMERS,
                created_at=datetime.now()
            )
            for i in range(5)
        ]
        
        for query in queries:
            test_session.add(query)
        test_session.commit()
        
        # Test query by embedding model
        openai_queries = test_session.query(RAGQuery).filter_by(embedding_model=EmbeddingModel.OPENAI_ADA).all()
        assert len(openai_queries) == 3  # Queries 0, 2, 4
        
        st_queries = test_session.query(RAGQuery).filter_by(embedding_model=EmbeddingModel.SENTENCE_TRANSFORMERS).all()
        assert len(st_queries) == 2  # Queries 1, 3
        
        # Test query by confidence score range
        high_confidence_queries = test_session.query(RAGQuery).filter(RAGQuery.confidence_score >= 0.85).all()
        assert len(high_confidence_queries) >= 2
    
    @pytest.mark.integration
    def test_rag_document_search_integration(self, test_session):
        """Test RAG document search operations."""
        # Create documents with different metadata
        documents = [
            RAGDocument(
                document_id=f"search_doc_{i}",
                title=f"Document {i}",
                content=f"Content for document {i}",
                metadata={
                    "category": "testing" if i % 2 == 0 else "development",
                    "difficulty": "beginner" if i < 2 else "advanced",
                    "tags": ["integration", "testing"] if i % 2 == 0 else ["coding", "development"]
                },
                chunk_count=i + 1
            )
            for i in range(4)
        ]
        
        for document in documents:
            test_session.add(document)
        test_session.commit()
        
        # Test search by title pattern
        search_docs = test_session.query(RAGDocument).filter(RAGDocument.title.like("Document%")).all()
        assert len(search_docs) == 4
        
        # Test search by metadata (this would require JSON query capabilities)
        # For SQLite, we'll test basic operations
        all_docs = test_session.query(RAGDocument).all()
        testing_docs = [doc for doc in all_docs if doc.metadata.get("category") == "testing"]
        assert len(testing_docs) == 2
    
    @pytest.mark.integration
    def test_rag_embedding_vector_operations_integration(self, test_session):
        """Test RAG embedding vector operations."""
        # Create document with embeddings
        document = RAGDocument(
            document_id="vector_test_doc",
            title="Vector Test Document",
            content="Test content for vector operations",
            chunk_count=3
        )
        test_session.add(document)
        test_session.commit()
        
        # Create embeddings with different vectors
        embeddings = [
            RAGEmbedding(
                embedding_id=f"vector_embed_{i}",
                document_id="vector_test_doc",
                chunk_index=i,
                chunk_text=f"Chunk {i} content",
                embedding_vector=[i * 0.1, i * 0.2, i * 0.3, i * 0.4],
                embedding_model=EmbeddingModel.OPENAI_ADA
            )
            for i in range(3)
        ]
        
        for embedding in embeddings:
            test_session.add(embedding)
        test_session.commit()
        
        # Test vector retrieval
        saved_embeddings = test_session.query(RAGEmbedding).filter_by(document_id="vector_test_doc").all()
        assert len(saved_embeddings) == 3
        
        # Verify vector data
        for i, embedding in enumerate(saved_embeddings):
            expected_vector = [i * 0.1, i * 0.2, i * 0.3, i * 0.4]
            assert embedding.embedding_vector == expected_vector
    
    @pytest.mark.integration
    def test_rag_query_history_integration(self, test_session):
        """Test RAG query history and analytics."""
        # Create queries with timestamps
        base_time = datetime.now()
        queries = []
        
        for i in range(10):
            query = RAGQuery(
                query_id=f"history_query_{i}",
                query_text=f"Historical query {i}",
                response_text=f"Historical response {i}",
                confidence_score=0.7 + (i * 0.02),
                response_time=0.5 + (i * 0.1),
                created_at=base_time
            )
            queries.append(query)
            test_session.add(query)
        
        test_session.commit()
        
        # Test query history retrieval
        recent_queries = test_session.query(RAGQuery).order_by(RAGQuery.created_at.desc()).limit(5).all()
        assert len(recent_queries) == 5
        
        # Test analytics queries
        avg_confidence = test_session.query(RAGQuery).with_entities(
            test_session.query(RAGQuery.confidence_score).subquery().c.confidence_score
        ).all()
        
        # Verify we can retrieve confidence scores for analytics
        confidence_scores = [query.confidence_score for query in queries]
        assert len(confidence_scores) == 10
        assert min(confidence_scores) == 0.7
        assert max(confidence_scores) == 0.88
    
    @pytest.mark.integration
    def test_rag_document_versioning_integration(self, test_session):
        """Test RAG document versioning and updates."""
        # Create initial document version
        document = RAGDocument(
            document_id="versioned_doc",
            title="Versioned Document",
            content="Initial content",
            metadata={"version": 1, "last_updated": datetime.now().isoformat()},
            chunk_count=1
        )
        test_session.add(document)
        test_session.commit()
        
        # Update document content
        document.content = "Updated content with more information"
        document.metadata = {"version": 2, "last_updated": datetime.now().isoformat()}
        document.chunk_count = 2
        test_session.commit()
        
        # Verify document was updated
        updated_document = test_session.query(RAGDocument).filter_by(document_id="versioned_doc").first()
        assert updated_document.content == "Updated content with more information"
        assert updated_document.metadata["version"] == 2
        assert updated_document.chunk_count == 2
    
    @pytest.mark.integration
    def test_rag_performance_integration(self, test_session):
        """Test RAG database performance with bulk operations."""
        # Create many documents and embeddings for performance testing
        documents = []
        embeddings = []
        
        for i in range(50):
            document = RAGDocument(
                document_id=f"perf_doc_{i}",
                title=f"Performance Document {i}",
                content=f"Content for performance testing document {i}",
                chunk_count=2
            )
            documents.append(document)
            
            # Create embeddings for each document
            for j in range(2):
                embedding = RAGEmbedding(
                    embedding_id=f"perf_embed_{i}_{j}",
                    document_id=f"perf_doc_{i}",
                    chunk_index=j,
                    chunk_text=f"Chunk {j} of document {i}",
                    embedding_vector=[i * 0.01, j * 0.02, (i+j) * 0.03, (i*j) * 0.04],
                    embedding_model=EmbeddingModel.OPENAI_ADA
                )
                embeddings.append(embedding)
        
        # Bulk insert documents
        test_session.add_all(documents)
        test_session.commit()
        
        # Bulk insert embeddings
        test_session.add_all(embeddings)
        test_session.commit()
        
        # Test bulk query performance
        all_documents = test_session.query(RAGDocument).filter(RAGDocument.document_id.like("perf_doc_%")).all()
        assert len(all_documents) == 50
        
        all_embeddings = test_session.query(RAGEmbedding).filter(RAGEmbedding.embedding_id.like("perf_embed_%")).all()
        assert len(all_embeddings) == 100  # 50 documents * 2 embeddings each
        
        # Test join query performance
        docs_with_embeddings = test_session.query(RAGDocument).join(RAGEmbedding).filter(
            RAGDocument.document_id.like("perf_doc_%")
        ).all()
        assert len(docs_with_embeddings) > 0
    
    @pytest.mark.integration
    def test_rag_json_metadata_integration(self, test_session):
        """Test JSON metadata storage and retrieval for RAG documents."""
        # Create document with complex metadata
        complex_metadata = {
            "author": "Test Author",
            "category": "technical_documentation",
            "tags": ["integration", "testing", "database"],
            "processing_info": {
                "chunking_strategy": "semantic",
                "chunk_size": 512,
                "overlap": 50,
                "embedding_model": "openai-ada-002"
            },
            "quality_metrics": {
                "readability_score": 8.5,
                "technical_complexity": "medium",
                "completeness": 0.92
            },
            "relationships": {
                "related_docs": ["doc_1", "doc_2"],
                "prerequisites": ["basic_testing"],
                "follows_up": ["advanced_integration"]
            }
        }
        
        document = RAGDocument(
            document_id="complex_metadata_doc",
            title="Complex Metadata Document",
            content="Document with complex metadata structure",
            metadata=complex_metadata,
            chunk_count=3
        )
        
        test_session.add(document)
        test_session.commit()
        
        # Retrieve and verify complex metadata
        saved_document = test_session.query(RAGDocument).filter_by(document_id="complex_metadata_doc").first()
        
        assert saved_document.metadata["author"] == "Test Author"
        assert saved_document.metadata["tags"] == ["integration", "testing", "database"]
        assert saved_document.metadata["processing_info"]["chunk_size"] == 512
        assert saved_document.metadata["quality_metrics"]["readability_score"] == 8.5
        assert saved_document.metadata["relationships"]["related_docs"] == ["doc_1", "doc_2"]
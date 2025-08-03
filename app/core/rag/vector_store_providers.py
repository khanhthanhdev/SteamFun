"""
Vector Store Provider Interface and Base Classes

This module defines the abstract interface for vector store providers and common
data models used throughout the vector storage system. It provides a unified interface
for different vector storage backends (ChromaDB, AstraDB, etc.) to ensure consistent behavior.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
import os
import logging
from .embedding_providers import EmbeddingProvider


@dataclass
class VectorStoreConfig:
    """Configuration for vector store providers."""
    provider: str  # "chromadb" or "astradb"
    connection_params: Dict[str, Any]
    collection_name: str
    embedding_dimension: int
    distance_metric: str = "cosine"  # "cosine", "euclidean", "dot_product"


class VectorStoreError(Exception):
    """Base exception for vector store operations."""
    pass


class VectorStoreConnectionError(VectorStoreError):
    """Exception raised when vector store connection fails."""
    pass


class VectorStoreOperationError(VectorStoreError):
    """Exception raised when vector store operations fail."""
    pass


class VectorStoreConfigurationError(VectorStoreError):
    """Exception raised when vector store configuration is invalid."""
    pass


@dataclass
class SearchResult:
    """Result from vector store search operations."""
    document_id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    embedding: Optional[List[float]] = None


@dataclass
class DocumentInput:
    """Input document for vector store operations."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class VectorStoreProvider(ABC):
    """Abstract base class for vector store providers.
    
    This interface defines the contract that all vector store providers must implement
    to ensure consistent behavior across different vector storage backends.
    """
    
    def __init__(self, config: VectorStoreConfig, embedding_provider: EmbeddingProvider):
        """Initialize the vector store provider with configuration.
        
        Args:
            config: VectorStoreConfig object containing provider settings
            embedding_provider: EmbeddingProvider instance for generating embeddings
        """
        self.config = config
        self.embedding_provider = embedding_provider
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate the vector store configuration.
        
        Raises:
            VectorStoreConfigurationError: If configuration is invalid
        """
        if not self.config.provider:
            raise VectorStoreConfigurationError("Provider must be specified")
        
        if not self.config.collection_name:
            raise VectorStoreConfigurationError("Collection name must be specified")
        
        if self.config.embedding_dimension <= 0:
            raise VectorStoreConfigurationError("Embedding dimension must be positive")
        
        if not self.config.connection_params:
            raise VectorStoreConfigurationError("Connection parameters must be provided")
        
        # Validate distance metric
        valid_metrics = ["cosine", "euclidean", "dot_product"]
        if self.config.distance_metric not in valid_metrics:
            raise VectorStoreConfigurationError(
                f"Invalid distance metric: {self.config.distance_metric}. "
                f"Valid options: {valid_metrics}"
            )
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the vector store connection and create collection if needed.
        
        Raises:
            VectorStoreConnectionError: If connection fails
            VectorStoreOperationError: If collection creation fails
        """
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[DocumentInput]) -> None:
        """Add documents to the vector store.
        
        Args:
            documents: List of DocumentInput objects to add
            
        Raises:
            VectorStoreOperationError: If document addition fails
        """
        pass
    
    @abstractmethod
    def update_documents(self, documents: List[DocumentInput]) -> None:
        """Update existing documents in the vector store.
        
        Args:
            documents: List of DocumentInput objects to update
            
        Raises:
            VectorStoreOperationError: If document update fails
        """
        pass
    
    @abstractmethod
    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from the vector store.
        
        Args:
            document_ids: List of document IDs to delete
            
        Raises:
            VectorStoreOperationError: If document deletion fails
        """
        pass
    
    @abstractmethod
    def similarity_search(self, 
                         query: str, 
                         k: int = 5,
                         filter_metadata: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Perform similarity search using vector embeddings.
        
        Args:
            query: Query text to search for
            k: Number of results to return
            filter_metadata: Optional metadata filters to apply
            
        Returns:
            List of SearchResult objects ordered by similarity score
            
        Raises:
            VectorStoreOperationError: If search fails
        """
        pass
    
    @abstractmethod
    def hybrid_search(self, 
                     query: str, 
                     k: int = 5,
                     filter_metadata: Optional[Dict[str, Any]] = None,
                     alpha: float = 0.5) -> List[SearchResult]:
        """Perform hybrid search combining vector and lexical search.
        
        Args:
            query: Query text to search for
            k: Number of results to return
            filter_metadata: Optional metadata filters to apply
            alpha: Weight for vector vs lexical search (0.0 = pure lexical, 1.0 = pure vector)
            
        Returns:
            List of SearchResult objects ordered by combined score
            
        Raises:
            VectorStoreOperationError: If search fails
        """
        pass
    
    @abstractmethod
    def get_document(self, document_id: str) -> Optional[SearchResult]:
        """Retrieve a specific document by ID.
        
        Args:
            document_id: ID of the document to retrieve
            
        Returns:
            SearchResult object if found, None otherwise
            
        Raises:
            VectorStoreOperationError: If retrieval fails
        """
        pass
    
    @abstractmethod
    def list_documents(self, 
                      limit: Optional[int] = None,
                      offset: int = 0,
                      filter_metadata: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """List documents in the vector store.
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            filter_metadata: Optional metadata filters to apply
            
        Returns:
            List of SearchResult objects
            
        Raises:
            VectorStoreOperationError: If listing fails
        """
        pass
    
    @abstractmethod
    def delete_collection(self) -> None:
        """Delete the entire collection.
        
        Raises:
            VectorStoreOperationError: If collection deletion fails
        """
        pass
    
    @abstractmethod
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection.
        
        Returns:
            Dictionary containing collection information including:
            - provider: Provider name
            - collection_name: Collection name
            - embedding_dimension: Embedding dimensions
            - document_count: Number of documents
            - distance_metric: Distance metric used
            - Additional provider-specific information
            
        Raises:
            VectorStoreOperationError: If info retrieval fails
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the vector store is available and properly configured.
        
        Returns:
            True if vector store is available, False otherwise
        """
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the vector store.
        
        Returns:
            Dictionary with health check results including:
            - status: "healthy", "degraded", or "unhealthy"
            - response_time: Response time in milliseconds
            - error_message: Error message if unhealthy
            - Additional provider-specific health metrics
        """
        pass
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the provider for logging and debugging.
        
        Returns:
            Dictionary containing provider information
        """
        return {
            "provider": self.config.provider,
            "collection_name": self.config.collection_name,
            "embedding_dimension": self.config.embedding_dimension,
            "distance_metric": self.config.distance_metric,
            "embedding_provider": self.embedding_provider.get_provider_info(),
            "available": self.is_available()
        }
    
    def _generate_embeddings_for_documents(self, documents: List[DocumentInput]) -> List[DocumentInput]:
        """Generate embeddings for documents that don't have them.
        
        Args:
            documents: List of DocumentInput objects
            
        Returns:
            List of DocumentInput objects with embeddings
            
        Raises:
            VectorStoreOperationError: If embedding generation fails
        """
        documents_to_embed = []
        documents_with_embeddings = []
        
        for doc in documents:
            if doc.embedding is None:
                documents_to_embed.append(doc)
            else:
                documents_with_embeddings.append(doc)
        
        if documents_to_embed:
            try:
                texts = [doc.content for doc in documents_to_embed]
                embeddings = self.embedding_provider.generate_embeddings(texts)
                
                for doc, embedding in zip(documents_to_embed, embeddings):
                    documents_with_embeddings.append(
                        DocumentInput(
                            id=doc.id,
                            content=doc.content,
                            metadata=doc.metadata,
                            embedding=embedding
                        )
                    )
            except Exception as e:
                raise VectorStoreOperationError(f"Failed to generate embeddings: {e}")
        
        return documents_with_embeddings
    
    def _generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a query string.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding vector
            
        Raises:
            VectorStoreOperationError: If embedding generation fails
        """
        try:
            embeddings = self.embedding_provider.generate_embeddings([query])
            return embeddings[0]
        except Exception as e:
            raise VectorStoreOperationError(f"Failed to generate query embedding: {e}")
    
    def _validate_embedding_dimension(self, embedding: List[float]) -> None:
        """Validate that embedding dimension matches configuration.
        
        Args:
            embedding: Embedding vector to validate
            
        Raises:
            VectorStoreOperationError: If dimension mismatch
        """
        if len(embedding) != self.config.embedding_dimension:
            raise VectorStoreOperationError(
                f"Embedding dimension mismatch: expected {self.config.embedding_dimension}, "
                f"got {len(embedding)}"
            )
    
    def _validate_documents(self, documents: List[DocumentInput]) -> None:
        """Validate document inputs.
        
        Args:
            documents: List of documents to validate
            
        Raises:
            VectorStoreOperationError: If validation fails
        """
        if not documents:
            raise VectorStoreOperationError("No documents provided")
        
        for i, doc in enumerate(documents):
            if not doc.id:
                raise VectorStoreOperationError(f"Document {i} missing ID")
            
            if not doc.content:
                raise VectorStoreOperationError(f"Document {i} missing content")
            
            if doc.embedding is not None:
                self._validate_embedding_dimension(doc.embedding)
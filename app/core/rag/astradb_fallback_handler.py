"""
AstraDB Fallback Handler

This module provides fallback mechanisms when AstraDB is unavailable,
automatically switching to ChromaDB as a backup vector store.
"""

import logging
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass

from .vector_store_providers import (
    VectorStoreProvider, 
    VectorStoreConfig, 
    SearchResult, 
    DocumentInput,
    VectorStoreError,
    VectorStoreConnectionError,
    VectorStoreOperationError,
    VectorStoreConfigurationError
)
from .embedding_providers import EmbeddingProvider


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior."""
    enable_fallback: bool = True
    fallback_provider: str = "chromadb"
    max_retry_attempts: int = 3
    retry_delay: float = 1.0
    fallback_threshold: int = 3  # Number of consecutive failures before fallback


class AstraDBFallbackHandler:
    """Handles fallback from AstraDB to ChromaDB when AstraDB is unavailable."""
    
    def __init__(self, 
                 astradb_provider: VectorStoreProvider,
                 fallback_config: FallbackConfig,
                 embedding_provider: EmbeddingProvider):
        """Initialize fallback handler.
        
        Args:
            astradb_provider: Primary AstraDB vector store provider
            fallback_config: Configuration for fallback behavior
            embedding_provider: Embedding provider for fallback store
        """
        self.astradb_provider = astradb_provider
        self.fallback_config = fallback_config
        self.embedding_provider = embedding_provider
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Fallback state tracking
        self.consecutive_failures = 0
        self.is_using_fallback = False
        self.fallback_provider = None
        
        # Initialize fallback provider if needed
        if self.fallback_config.enable_fallback:
            self._initialize_fallback_provider()
    
    def _initialize_fallback_provider(self) -> None:
        """Initialize the fallback ChromaDB provider."""
        try:
            from .vector_store_factory import VectorStoreFactory
            
            # Create fallback configuration
            fallback_config = VectorStoreConfig(
                provider=self.fallback_config.fallback_provider,
                connection_params={
                    "path": "./fallback_chromadb",  # Local fallback path
                    "host": None,
                    "port": None
                },
                collection_name=f"{self.astradb_provider.config.collection_name}_fallback",
                embedding_dimension=self.astradb_provider.config.embedding_dimension,
                distance_metric=self.astradb_provider.config.distance_metric
            )
            
            # Create fallback provider
            factory = VectorStoreFactory()
            self.fallback_provider = factory.create_provider(
                fallback_config, 
                self.embedding_provider
            )
            
            # Initialize fallback provider
            self.fallback_provider.initialize()
            
            self.logger.info("Fallback ChromaDB provider initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize fallback provider: {e}")
            self.fallback_config.enable_fallback = False
    
    def _should_use_fallback(self) -> bool:
        """Determine if fallback should be used based on failure count."""
        return (self.fallback_config.enable_fallback and 
                self.consecutive_failures >= self.fallback_config.fallback_threshold)
    
    def _handle_operation_success(self) -> None:
        """Reset failure count on successful operation."""
        if self.consecutive_failures > 0:
            self.logger.info("AstraDB operation successful, resetting failure count")
            self.consecutive_failures = 0
        
        if self.is_using_fallback:
            self.logger.info("AstraDB recovered, switching back from fallback")
            self.is_using_fallback = False
    
    def _handle_operation_failure(self, operation: str, error: Exception) -> None:
        """Handle operation failure and update fallback state."""
        self.consecutive_failures += 1
        
        self.logger.warning(
            f"AstraDB {operation} failed (attempt {self.consecutive_failures}): {error}"
        )
        
        if self._should_use_fallback() and not self.is_using_fallback:
            self.logger.warning(
                f"Switching to fallback provider after {self.consecutive_failures} failures"
            )
            self.is_using_fallback = True
    
    def _execute_with_fallback(self, operation_name: str, astradb_operation, fallback_operation):
        """Execute operation with automatic fallback on failure.
        
        Args:
            operation_name: Name of the operation for logging
            astradb_operation: Function to execute on AstraDB
            fallback_operation: Function to execute on fallback provider
            
        Returns:
            Result of the operation
            
        Raises:
            VectorStoreOperationError: If both primary and fallback operations fail
        """
        # If already using fallback, try fallback first
        if self.is_using_fallback and self.fallback_provider:
            try:
                result = fallback_operation()
                self.logger.debug(f"Fallback {operation_name} successful")
                return result
            except Exception as fallback_error:
                self.logger.error(f"Fallback {operation_name} failed: {fallback_error}")
                # Try AstraDB as last resort
                pass
        
        # Try AstraDB operation
        try:
            result = astradb_operation()
            self._handle_operation_success()
            return result
            
        except Exception as astradb_error:
            self._handle_operation_failure(operation_name, astradb_error)
            
            # Try fallback if available and not already tried
            if (self.fallback_config.enable_fallback and 
                self.fallback_provider and 
                not self.is_using_fallback):
                
                try:
                    self.logger.info(f"Attempting {operation_name} with fallback provider")
                    result = fallback_operation()
                    self.is_using_fallback = True
                    return result
                    
                except Exception as fallback_error:
                    self.logger.error(f"Fallback {operation_name} also failed: {fallback_error}")
                    raise VectorStoreOperationError(
                        f"Both AstraDB and fallback {operation_name} failed. "
                        f"AstraDB error: {astradb_error}. Fallback error: {fallback_error}"
                    )
            
            # No fallback available or fallback already failed
            raise VectorStoreOperationError(f"AstraDB {operation_name} failed: {astradb_error}")
    
    def add_documents(self, documents: List[DocumentInput]) -> None:
        """Add documents with fallback support."""
        return self._execute_with_fallback(
            "add_documents",
            lambda: self.astradb_provider.add_documents(documents),
            lambda: self.fallback_provider.add_documents(documents) if self.fallback_provider else None
        )
    
    def update_documents(self, documents: List[DocumentInput]) -> None:
        """Update documents with fallback support."""
        return self._execute_with_fallback(
            "update_documents",
            lambda: self.astradb_provider.update_documents(documents),
            lambda: self.fallback_provider.update_documents(documents) if self.fallback_provider else None
        )
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents with fallback support."""
        return self._execute_with_fallback(
            "delete_documents",
            lambda: self.astradb_provider.delete_documents(document_ids),
            lambda: self.fallback_provider.delete_documents(document_ids) if self.fallback_provider else None
        )
    
    def similarity_search(self, 
                         query: str, 
                         k: int = 5,
                         filter_metadata: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Perform similarity search with fallback support."""
        return self._execute_with_fallback(
            "similarity_search",
            lambda: self.astradb_provider.similarity_search(query, k, filter_metadata),
            lambda: self.fallback_provider.similarity_search(query, k, filter_metadata) if self.fallback_provider else []
        )
    
    def hybrid_search(self, 
                     query: str, 
                     k: int = 5,
                     filter_metadata: Optional[Dict[str, Any]] = None,
                     alpha: float = 0.5) -> List[SearchResult]:
        """Perform hybrid search with fallback support."""
        return self._execute_with_fallback(
            "hybrid_search",
            lambda: self.astradb_provider.hybrid_search(query, k, filter_metadata, alpha),
            lambda: self.fallback_provider.hybrid_search(query, k, filter_metadata, alpha) if self.fallback_provider else []
        )
    
    def get_document(self, document_id: str) -> Optional[SearchResult]:
        """Get document with fallback support."""
        return self._execute_with_fallback(
            "get_document",
            lambda: self.astradb_provider.get_document(document_id),
            lambda: self.fallback_provider.get_document(document_id) if self.fallback_provider else None
        )
    
    def list_documents(self, 
                      limit: Optional[int] = None,
                      offset: int = 0,
                      filter_metadata: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """List documents with fallback support."""
        return self._execute_with_fallback(
            "list_documents",
            lambda: self.astradb_provider.list_documents(limit, offset, filter_metadata),
            lambda: self.fallback_provider.list_documents(limit, offset, filter_metadata) if self.fallback_provider else []
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of both primary and fallback providers."""
        status = {
            "primary_provider": "astradb",
            "fallback_provider": self.fallback_config.fallback_provider,
            "is_using_fallback": self.is_using_fallback,
            "consecutive_failures": self.consecutive_failures,
            "fallback_enabled": self.fallback_config.enable_fallback,
            "fallback_threshold": self.fallback_config.fallback_threshold
        }
        
        # Add primary provider status
        try:
            status["primary_available"] = self.astradb_provider.is_available()
            status["primary_health"] = self.astradb_provider.health_check()
        except Exception as e:
            status["primary_available"] = False
            status["primary_error"] = str(e)
        
        # Add fallback provider status
        if self.fallback_provider:
            try:
                status["fallback_available"] = self.fallback_provider.is_available()
                status["fallback_health"] = self.fallback_provider.health_check()
            except Exception as e:
                status["fallback_available"] = False
                status["fallback_error"] = str(e)
        else:
            status["fallback_available"] = False
            status["fallback_error"] = "Fallback provider not initialized"
        
        return status
    
    def force_fallback(self, enable: bool = True) -> None:
        """Force enable or disable fallback mode.
        
        Args:
            enable: True to force fallback mode, False to force primary mode
        """
        if enable and self.fallback_provider:
            self.logger.info("Forcing fallback mode")
            self.is_using_fallback = True
            self.consecutive_failures = self.fallback_config.fallback_threshold
        else:
            self.logger.info("Forcing primary mode")
            self.is_using_fallback = False
            self.consecutive_failures = 0
    
    def sync_to_primary(self) -> Dict[str, Any]:
        """Sync data from fallback to primary when primary recovers.
        
        Returns:
            Dictionary with sync results
        """
        if not self.is_using_fallback or not self.fallback_provider:
            return {"status": "no_sync_needed", "message": "Not using fallback"}
        
        if not self.astradb_provider.is_available():
            return {"status": "primary_unavailable", "message": "Primary provider not available"}
        
        try:
            # Get all documents from fallback
            fallback_docs = self.fallback_provider.list_documents()
            
            if not fallback_docs:
                return {"status": "no_data", "message": "No data to sync"}
            
            # Convert to DocumentInput format
            documents_to_sync = []
            for doc in fallback_docs:
                doc_input = DocumentInput(
                    id=doc.document_id,
                    content=doc.content,
                    metadata=doc.metadata,
                    embedding=doc.embedding
                )
                documents_to_sync.append(doc_input)
            
            # Add to primary provider
            self.astradb_provider.add_documents(documents_to_sync)
            
            self.logger.info(f"Successfully synced {len(documents_to_sync)} documents to primary")
            
            return {
                "status": "success",
                "synced_documents": len(documents_to_sync),
                "message": f"Synced {len(documents_to_sync)} documents to primary provider"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to sync data to primary: {e}")
            return {
                "status": "error",
                "message": f"Sync failed: {e}"
            }
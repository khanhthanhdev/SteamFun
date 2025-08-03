"""
AstraDB Vector Store Provider Implementation

This module implements the AstraDB vector store provider using the astrapy library.
It provides cloud-based vector storage with native vector search and hybrid search capabilities.
"""

import os
import time
import logging
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass

try:
    from astrapy import DataAPIClient
    from astrapy.constants import VectorMetric
    from astrapy.info import CollectionDefinition
    from astrapy.exceptions import (
        DataAPIException, 
        DataAPIResponseException,
        DataAPITimeoutException
    )
    ASTRAPY_AVAILABLE = True
except ImportError:
    ASTRAPY_AVAILABLE = False
    DataAPIClient = None
    VectorMetric = None
    CollectionDefinition = None
    DataAPIException = Exception
    DataAPIResponseException = Exception
    DataAPITimeoutException = Exception

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


class AstraDBVectorStore(VectorStoreProvider):
    """AstraDB vector store implementation using astrapy library.
    
    This provider implements cloud-based vector storage using DataStax AstraDB,
    providing native vector search capabilities, automatic scaling, and hybrid search features.
    """
    
    def __init__(self, config: VectorStoreConfig, embedding_provider: EmbeddingProvider):
        """Initialize AstraDB vector store.
        
        Args:
            config: VectorStoreConfig with AstraDB connection parameters
            embedding_provider: EmbeddingProvider for generating embeddings
            
        Raises:
            VectorStoreConfigurationError: If astrapy is not available or config is invalid
        """
        if not ASTRAPY_AVAILABLE:
            raise VectorStoreConfigurationError(
                "astrapy library is not installed. Install it with: pip install astrapy"
            )
        
        # Map distance metrics to AstraDB VectorMetric first
        self.metric_mapping = {
            "cosine": VectorMetric.COSINE,
            "euclidean": VectorMetric.EUCLIDEAN,
            "dot_product": VectorMetric.DOT_PRODUCT
        }
        
        super().__init__(config, embedding_provider)
        
        # Initialize AstraDB client and database
        self.client = None
        self.database = None
        self.collection = None
        
        # Connection parameters
        self.api_endpoint = config.connection_params.get('api_endpoint')
        self.application_token = config.connection_params.get('application_token')
        self.keyspace = config.connection_params.get('keyspace')
        self.region = config.connection_params.get('region')
        
        # Validate required parameters
        self._validate_astradb_config()
        
        self.logger.info(f"Initialized AstraDB vector store for collection: {self.config.collection_name}")
    
    def _validate_astradb_config(self) -> None:
        """Validate AstraDB-specific configuration.
        
        Raises:
            VectorStoreConfigurationError: If required parameters are missing
        """
        if not self.api_endpoint:
            raise VectorStoreConfigurationError(
                "AstraDB API endpoint is required. Get it from the AstraDB console."
            )
        
        if not self.application_token:
            raise VectorStoreConfigurationError(
                "AstraDB application token is required. Generate one in the AstraDB console."
            )
        
        if not self.api_endpoint.startswith('https://'):
            raise VectorStoreConfigurationError(
                f"AstraDB API endpoint must start with https://, got: {self.api_endpoint}"
            )
        
        if self.config.distance_metric not in self.metric_mapping:
            raise VectorStoreConfigurationError(
                f"Unsupported distance metric for AstraDB: {self.config.distance_metric}. "
                f"Supported metrics: {list(self.metric_mapping.keys())}"
            )
    
    def initialize(self) -> None:
        """Initialize AstraDB connection and create collection if needed.
        
        Raises:
            VectorStoreConnectionError: If connection fails
            VectorStoreOperationError: If collection creation fails
        """
        try:
            # Initialize DataAPI client
            self.client = DataAPIClient()
            
            # Get database connection
            self.database = self.client.get_database(
                api_endpoint=self.api_endpoint,
                token=self.application_token,
                keyspace=self.keyspace
            )
            
            # Test connection
            self._test_connection()
            
            # Get or create collection
            self.collection = self._get_or_create_collection()
            
            self.logger.info(f"Successfully initialized AstraDB connection to collection: {self.config.collection_name}")
            
        except DataAPIException as e:
            raise VectorStoreConnectionError(f"Failed to connect to AstraDB: {e}")
        except Exception as e:
            raise VectorStoreConnectionError(f"Unexpected error connecting to AstraDB: {e}")
    
    def _test_connection(self) -> None:
        """Test the AstraDB connection.
        
        Raises:
            VectorStoreConnectionError: If connection test fails
        """
        try:
            # Try to list collections to test connection
            collections = list(self.database.list_collections())
            self.logger.debug(f"Connection test successful. Found {len(collections)} collections.")
        except Exception as e:
            raise VectorStoreConnectionError(f"AstraDB connection test failed: {e}")
    
    def _get_or_create_collection(self):
        """Get existing collection or create a new one with vector support.
        
        Returns:
            AstraDB collection object
            
        Raises:
            VectorStoreOperationError: If collection operations fail
        """
        try:
            # Try to get existing collection
            try:
                collection = self.database.get_collection(self.config.collection_name)
                
                # Verify collection has correct vector dimension
                collection_info = collection.info()
                vector_config = collection_info.options.vector
                
                if vector_config and vector_config.dimension != self.config.embedding_dimension:
                    self.logger.warning(
                        f"Existing collection has dimension {vector_config.dimension}, "
                        f"but expected {self.config.embedding_dimension}. "
                        f"Consider using a different collection name or recreating the collection."
                    )
                
                self.logger.info(f"Using existing AstraDB collection: {self.config.collection_name}")
                return collection
                
            except Exception:
                # Collection doesn't exist, create it
                self.logger.info(f"Creating new AstraDB collection: {self.config.collection_name}")
                
                collection = self.database.create_collection(
                    name=self.config.collection_name,
                    definition=(
                        CollectionDefinition.builder()
                        .set_vector_dimension(self.config.embedding_dimension)
                        .set_vector_metric(self.metric_mapping[self.config.distance_metric])
                        .build()
                    )
                )
                
                self.logger.info(f"Successfully created AstraDB collection: {self.config.collection_name}")
                return collection
                
        except DataAPIException as e:
            raise VectorStoreOperationError(f"Failed to get or create AstraDB collection: {e}")
        except Exception as e:
            raise VectorStoreOperationError(f"Unexpected error with AstraDB collection: {e}")    
    def add_documents(self, documents: List[DocumentInput]) -> None:
        """Add documents to AstraDB collection.
        
        Args:
            documents: List of DocumentInput objects to add
            
        Raises:
            VectorStoreOperationError: If document addition fails
        """
        if not self.collection:
            raise VectorStoreOperationError("AstraDB collection not initialized. Call initialize() first.")
        
        self._validate_documents(documents)
        
        try:
            # Generate embeddings for documents that don't have them
            documents_with_embeddings = self._generate_embeddings_for_documents(documents)
            
            # Prepare documents for AstraDB insertion
            astradb_documents = []
            for doc in documents_with_embeddings:
                astradb_doc = {
                    "_id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "$vector": doc.embedding
                }
                astradb_documents.append(astradb_doc)
            
            # Insert documents in batches to handle large datasets
            batch_size = 100  # AstraDB recommended batch size
            for i in range(0, len(astradb_documents), batch_size):
                batch = astradb_documents[i:i + batch_size]
                
                try:
                    result = self.collection.insert_many(batch)
                    self.logger.debug(f"Inserted batch of {len(batch)} documents")
                    
                    # Check for any insertion errors
                    if hasattr(result, 'inserted_ids'):
                        inserted_count = len(result.inserted_ids)
                        if inserted_count != len(batch):
                            self.logger.warning(
                                f"Expected to insert {len(batch)} documents, "
                                f"but only {inserted_count} were inserted"
                            )
                
                except DataAPIException as e:
                    # Try individual inserts for failed batch
                    self.logger.warning(f"Batch insert failed, trying individual inserts: {e}")
                    self._insert_documents_individually(batch)
            
            self.logger.info(f"Successfully added {len(documents)} documents to AstraDB collection")
            
        except Exception as e:
            raise VectorStoreOperationError(f"Failed to add documents to AstraDB: {e}")
    
    def _insert_documents_individually(self, documents: List[Dict[str, Any]]) -> None:
        """Insert documents individually when batch insert fails.
        
        Args:
            documents: List of document dictionaries to insert
        """
        for doc in documents:
            try:
                self.collection.insert_one(doc)
            except DataAPIException as e:
                self.logger.error(f"Failed to insert document {doc.get('_id', 'unknown')}: {e}")
                # Continue with other documents
    
    def update_documents(self, documents: List[DocumentInput]) -> None:
        """Update existing documents in AstraDB collection.
        
        Args:
            documents: List of DocumentInput objects to update
            
        Raises:
            VectorStoreOperationError: If document update fails
        """
        if not self.collection:
            raise VectorStoreOperationError("AstraDB collection not initialized. Call initialize() first.")
        
        self._validate_documents(documents)
        
        try:
            # Generate embeddings for documents that don't have them
            documents_with_embeddings = self._generate_embeddings_for_documents(documents)
            
            # Update documents one by one (AstraDB doesn't have bulk update)
            for doc in documents_with_embeddings:
                update_doc = {
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "$vector": doc.embedding
                }
                
                try:
                    result = self.collection.find_one_and_replace(
                        filter={"_id": doc.id},
                        replacement=update_doc,
                        upsert=True  # Create if doesn't exist
                    )
                    
                    if result is None:
                        self.logger.debug(f"Document {doc.id} was created (upserted)")
                    else:
                        self.logger.debug(f"Document {doc.id} was updated")
                        
                except DataAPIException as e:
                    self.logger.error(f"Failed to update document {doc.id}: {e}")
                    raise VectorStoreOperationError(f"Failed to update document {doc.id}: {e}")
            
            self.logger.info(f"Successfully updated {len(documents)} documents in AstraDB collection")
            
        except Exception as e:
            raise VectorStoreOperationError(f"Failed to update documents in AstraDB: {e}")
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from AstraDB collection.
        
        Args:
            document_ids: List of document IDs to delete
            
        Raises:
            VectorStoreOperationError: If document deletion fails
        """
        if not self.collection:
            raise VectorStoreOperationError("AstraDB collection not initialized. Call initialize() first.")
        
        if not document_ids:
            return
        
        try:
            # Delete documents in batches
            batch_size = 100
            deleted_count = 0
            
            for i in range(0, len(document_ids), batch_size):
                batch_ids = document_ids[i:i + batch_size]
                
                try:
                    # Use delete_many with filter on _id
                    result = self.collection.delete_many(
                        filter={"_id": {"$in": batch_ids}}
                    )
                    
                    batch_deleted = getattr(result, 'deleted_count', len(batch_ids))
                    deleted_count += batch_deleted
                    
                    self.logger.debug(f"Deleted {batch_deleted} documents from batch")
                    
                except DataAPIException as e:
                    self.logger.error(f"Failed to delete batch of documents: {e}")
                    # Try individual deletes
                    for doc_id in batch_ids:
                        try:
                            self.collection.delete_one({"_id": doc_id})
                            deleted_count += 1
                        except DataAPIException as individual_e:
                            self.logger.error(f"Failed to delete document {doc_id}: {individual_e}")
            
            self.logger.info(f"Successfully deleted {deleted_count} documents from AstraDB collection")
            
        except Exception as e:
            raise VectorStoreOperationError(f"Failed to delete documents from AstraDB: {e}")    
            
    def similarity_search(self, 
                         query: str, 
                         k: int = 5,
                         filter_metadata: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Perform vector similarity search using AstraDB native capabilities.
        
        Args:
            query: Query text to search for
            k: Number of results to return
            filter_metadata: Optional metadata filters to apply
            
        Returns:
            List of SearchResult objects ordered by similarity score
            
        Raises:
            VectorStoreOperationError: If search fails
        """
        if not self.collection:
            raise VectorStoreOperationError("AstraDB collection not initialized. Call initialize() first.")
        
        try:
            # Generate query embedding
            query_embedding = self._generate_query_embedding(query)
            
            # Build filter
            search_filter = {}
            if filter_metadata:
                # Convert metadata filters to AstraDB format
                for key, value in filter_metadata.items():
                    search_filter[f"metadata.{key}"] = value
            
            # Perform vector similarity search
            cursor = self.collection.find(
                filter=search_filter,
                sort={"$vector": query_embedding},
                limit=k,
                include_similarity=True
            )
            
            # Convert results to SearchResult objects
            results = []
            for doc in cursor:
                # Extract similarity score
                similarity_score = doc.get("$similarity", 0.0)
                
                # Create SearchResult
                result = SearchResult(
                    document_id=doc["_id"],
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}),
                    score=similarity_score,
                    embedding=doc.get("$vector")
                )
                results.append(result)
            
            self.logger.debug(f"Vector similarity search returned {len(results)} results")
            return results
            
        except DataAPIException as e:
            raise VectorStoreOperationError(f"AstraDB similarity search failed: {e}")
        except Exception as e:
            raise VectorStoreOperationError(f"Unexpected error during similarity search: {e}")
    
    def hybrid_search(self, 
                     query: str, 
                     k: int = 5,
                     filter_metadata: Optional[Dict[str, Any]] = None,
                     alpha: float = 0.5) -> List[SearchResult]:
        """Perform hybrid search using AstraDB's native find_and_rerank capability.
        
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
        if not self.collection:
            raise VectorStoreOperationError("AstraDB collection not initialized. Call initialize() first.")
        
        try:
            # Build filter
            search_filter = {}
            if filter_metadata:
                # Convert metadata filters to AstraDB format
                for key, value in filter_metadata.items():
                    search_filter[f"metadata.{key}"] = value
            
            # Perform hybrid search using AstraDB's find_and_rerank
            # This combines vector similarity with text-based search
            results = self.collection.find_and_rerank(
                filter=search_filter,
                sort={"$hybrid": query},
                limit=k,
                include_scores=True
            )
            
            # Convert results to SearchResult objects
            search_results = []
            for result in results:
                doc = result.document
                
                # Extract hybrid score
                hybrid_score = getattr(result, 'score', 0.0)
                
                # Create SearchResult
                search_result = SearchResult(
                    document_id=doc["_id"],
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}),
                    score=hybrid_score,
                    embedding=doc.get("$vector")
                )
                search_results.append(search_result)
            
            self.logger.debug(f"Hybrid search returned {len(search_results)} results")
            return search_results
            
        except DataAPIException as e:
            # Fallback to pure vector search if hybrid search is not available
            self.logger.warning(f"Hybrid search failed, falling back to vector search: {e}")
            return self.similarity_search(query, k, filter_metadata)
        except Exception as e:
            raise VectorStoreOperationError(f"Unexpected error during hybrid search: {e}")
    
    def get_document(self, document_id: str) -> Optional[SearchResult]:
        """Retrieve a specific document by ID.
        
        Args:
            document_id: ID of the document to retrieve
            
        Returns:
            SearchResult object if found, None otherwise
            
        Raises:
            VectorStoreOperationError: If retrieval fails
        """
        if not self.collection:
            raise VectorStoreOperationError("AstraDB collection not initialized. Call initialize() first.")
        
        try:
            doc = self.collection.find_one({"_id": document_id})
            
            if doc is None:
                return None
            
            # Create SearchResult
            result = SearchResult(
                document_id=doc["_id"],
                content=doc.get("content", ""),
                metadata=doc.get("metadata", {}),
                score=1.0,  # Perfect match for direct retrieval
                embedding=doc.get("$vector")
            )
            
            return result
            
        except DataAPIException as e:
            raise VectorStoreOperationError(f"Failed to retrieve document {document_id}: {e}")
        except Exception as e:
            raise VectorStoreOperationError(f"Unexpected error retrieving document: {e}")
    
    def list_documents(self, 
                      limit: Optional[int] = None,
                      offset: int = 0,
                      filter_metadata: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """List documents in the AstraDB collection.
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            filter_metadata: Optional metadata filters to apply
            
        Returns:
            List of SearchResult objects
            
        Raises:
            VectorStoreOperationError: If listing fails
        """
        if not self.collection:
            raise VectorStoreOperationError("AstraDB collection not initialized. Call initialize() first.")
        
        try:
            # Build filter
            search_filter = {}
            if filter_metadata:
                # Convert metadata filters to AstraDB format
                for key, value in filter_metadata.items():
                    search_filter[f"metadata.{key}"] = value
            
            # Build find options
            find_options = {}
            if limit is not None:
                find_options['limit'] = limit
            if offset > 0:
                find_options['skip'] = offset
            
            # Find documents
            cursor = self.collection.find(filter=search_filter, **find_options)
            
            # Convert to SearchResult objects
            results = []
            for doc in cursor:
                result = SearchResult(
                    document_id=doc["_id"],
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}),
                    score=1.0,  # No scoring for listing
                    embedding=doc.get("$vector")
                )
                results.append(result)
            
            self.logger.debug(f"Listed {len(results)} documents from AstraDB collection")
            return results
            
        except DataAPIException as e:
            raise VectorStoreOperationError(f"Failed to list documents: {e}")
        except Exception as e:
            raise VectorStoreOperationError(f"Unexpected error listing documents: {e}")    
            
    def delete_collection(self) -> None:
        """Delete the entire AstraDB collection.
        
        Raises:
            VectorStoreOperationError: If collection deletion fails
        """
        if not self.collection:
            raise VectorStoreOperationError("AstraDB collection not initialized. Call initialize() first.")
        
        try:
            self.collection.drop()
            self.collection = None
            self.logger.info(f"Successfully deleted AstraDB collection: {self.config.collection_name}")
            
        except DataAPIException as e:
            raise VectorStoreOperationError(f"Failed to delete AstraDB collection: {e}")
        except Exception as e:
            raise VectorStoreOperationError(f"Unexpected error deleting collection: {e}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the AstraDB collection.
        
        Returns:
            Dictionary containing collection information
            
        Raises:
            VectorStoreOperationError: If info retrieval fails
        """
        if not self.collection:
            raise VectorStoreOperationError("AstraDB collection not initialized. Call initialize() first.")
        
        try:
            # Get collection info
            collection_info = self.collection.info()
            
            # Get document count
            document_count = self.collection.count_documents({})
            
            # Extract vector configuration
            vector_config = collection_info.options.vector
            vector_dimension = vector_config.dimension if vector_config else None
            vector_metric = vector_config.metric if vector_config else None
            
            info = {
                "provider": "astradb",
                "collection_name": self.config.collection_name,
                "embedding_dimension": vector_dimension or self.config.embedding_dimension,
                "document_count": document_count,
                "distance_metric": self.config.distance_metric,
                "vector_metric": str(vector_metric) if vector_metric else None,
                "api_endpoint": self.api_endpoint,
                "keyspace": self.keyspace,
                "region": self.region,
                "collection_status": "active" if self.collection else "inactive"
            }
            
            return info
            
        except DataAPIException as e:
            raise VectorStoreOperationError(f"Failed to get AstraDB collection info: {e}")
        except Exception as e:
            raise VectorStoreOperationError(f"Unexpected error getting collection info: {e}")
    
    def is_available(self) -> bool:
        """Check if AstraDB is available and properly configured.
        
        Returns:
            True if AstraDB is available, False otherwise
        """
        try:
            # Check if astrapy is available
            if not ASTRAPY_AVAILABLE:
                return False
            
            # Check if configuration is valid
            if not self.api_endpoint or not self.application_token:
                return False
            
            # Test connection if not already initialized
            if not self.database:
                try:
                    client = DataAPIClient()
                    database = client.get_database(
                        api_endpoint=self.api_endpoint,
                        token=self.application_token,
                        keyspace=self.keyspace
                    )
                    # Try to list collections to test connection
                    list(database.list_collections())
                    return True
                except Exception:
                    return False
            
            # If already initialized, test with a simple operation
            try:
                list(self.database.list_collections())
                return True
            except Exception:
                return False
                
        except Exception:
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on AstraDB.
        
        Returns:
            Dictionary with health check results
        """
        start_time = time.time()
        
        try:
            if not self.is_available():
                return {
                    "status": "unhealthy",
                    "response_time": (time.time() - start_time) * 1000,
                    "error_message": "AstraDB is not available or not properly configured",
                    "astrapy_available": ASTRAPY_AVAILABLE,
                    "configuration_valid": bool(self.api_endpoint and self.application_token)
                }
            
            # Test basic operations
            if self.collection:
                # Test with a simple count operation
                try:
                    # Use estimated_document_count() instead of count_documents() for health check
                    # as it's more efficient and doesn't require upper_bound parameter
                    count = self.collection.estimated_document_count()
                    response_time = (time.time() - start_time) * 1000
                    
                    return {
                        "status": "healthy",
                        "response_time": response_time,
                        "document_count": count,
                        "collection_active": True,
                        "astrapy_available": True,
                        "api_endpoint": self.api_endpoint,
                        "keyspace": self.keyspace
                    }
                except Exception as e:
                    return {
                        "status": "degraded",
                        "response_time": (time.time() - start_time) * 1000,
                        "error_message": f"Collection operations failing: {e}",
                        "collection_active": False,
                        "astrapy_available": True
                    }
            else:
                # Collection not initialized but connection works
                response_time = (time.time() - start_time) * 1000
                return {
                    "status": "degraded",
                    "response_time": response_time,
                    "error_message": "Collection not initialized",
                    "collection_active": False,
                    "astrapy_available": True,
                    "api_endpoint": self.api_endpoint
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "response_time": (time.time() - start_time) * 1000,
                "error_message": f"Health check failed: {e}",
                "astrapy_available": ASTRAPY_AVAILABLE
            }
class AstraDBWithFallback(VectorStoreProvider):
    """AstraDB vector store with automatic fallback to ChromaDB.
    
    This wrapper provides automatic fallback functionality when AstraDB is unavailable,
    ensuring high availability and reliability of the vector store operations.
    """
    
    def __init__(self, config: VectorStoreConfig, embedding_provider: EmbeddingProvider):
        """Initialize AstraDB with fallback capabilities.
        
        Args:
            config: VectorStoreConfig with AstraDB connection parameters
            embedding_provider: EmbeddingProvider for generating embeddings
        """
        super().__init__(config, embedding_provider)
        
        # Initialize primary AstraDB provider
        self.primary_provider = AstraDBVectorStore(config, embedding_provider)
        
        # Initialize fallback handler
        from .astradb_fallback_handler import AstraDBFallbackHandler, FallbackConfig
        
        fallback_config = FallbackConfig(
            enable_fallback=True,
            fallback_provider="chromadb",
            max_retry_attempts=3,
            retry_delay=1.0,
            fallback_threshold=2  # Switch to fallback after 2 failures
        )
        
        self.fallback_handler = AstraDBFallbackHandler(
            self.primary_provider,
            fallback_config,
            embedding_provider
        )
        
        self.logger.info("Initialized AstraDB with fallback capabilities")
    
    def initialize(self) -> None:
        """Initialize AstraDB connection with fallback support."""
        try:
            self.primary_provider.initialize()
            self.logger.info("AstraDB primary provider initialized successfully")
        except Exception as e:
            self.logger.warning(f"AstraDB primary initialization failed: {e}")
            # Fallback will be used automatically when operations are called
    
    def add_documents(self, documents: List[DocumentInput]) -> None:
        """Add documents with automatic fallback."""
        return self.fallback_handler.add_documents(documents)
    
    def update_documents(self, documents: List[DocumentInput]) -> None:
        """Update documents with automatic fallback."""
        return self.fallback_handler.update_documents(documents)
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents with automatic fallback."""
        return self.fallback_handler.delete_documents(document_ids)
    
    def similarity_search(self, 
                         query: str, 
                         k: int = 5,
                         filter_metadata: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Perform similarity search with automatic fallback."""
        return self.fallback_handler.similarity_search(query, k, filter_metadata)
    
    def hybrid_search(self, 
                     query: str, 
                     k: int = 5,
                     filter_metadata: Optional[Dict[str, Any]] = None,
                     alpha: float = 0.5) -> List[SearchResult]:
        """Perform hybrid search with automatic fallback."""
        return self.fallback_handler.hybrid_search(query, k, filter_metadata, alpha)
    
    def get_document(self, document_id: str) -> Optional[SearchResult]:
        """Get document with automatic fallback."""
        return self.fallback_handler.get_document(document_id)
    
    def list_documents(self, 
                      limit: Optional[int] = None,
                      offset: int = 0,
                      filter_metadata: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """List documents with automatic fallback."""
        return self.fallback_handler.list_documents(limit, offset, filter_metadata)
    
    def delete_collection(self) -> None:
        """Delete collection with fallback awareness."""
        try:
            self.primary_provider.delete_collection()
        except Exception as e:
            self.logger.warning(f"Failed to delete primary collection: {e}")
        
        # Also delete fallback collection if it exists
        if (self.fallback_handler.fallback_provider and 
            hasattr(self.fallback_handler.fallback_provider, 'delete_collection')):
            try:
                self.fallback_handler.fallback_provider.delete_collection()
            except Exception as e:
                self.logger.warning(f"Failed to delete fallback collection: {e}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection info with fallback status."""
        info = {
            "provider": "astradb_with_fallback",
            "collection_name": self.config.collection_name,
            "embedding_dimension": self.config.embedding_dimension,
            "distance_metric": self.config.distance_metric
        }
        
        # Add fallback status
        status = self.fallback_handler.get_status()
        info.update(status)
        
        # Try to get primary provider info
        try:
            primary_info = self.primary_provider.get_collection_info()
            info["primary_info"] = primary_info
        except Exception as e:
            info["primary_error"] = str(e)
        
        # Try to get fallback provider info
        if self.fallback_handler.fallback_provider:
            try:
                fallback_info = self.fallback_handler.fallback_provider.get_collection_info()
                info["fallback_info"] = fallback_info
            except Exception as e:
                info["fallback_error"] = str(e)
        
        return info
    
    def is_available(self) -> bool:
        """Check if either primary or fallback is available."""
        primary_available = self.primary_provider.is_available()
        
        if primary_available:
            return True
        
        # Check fallback availability
        if self.fallback_handler.fallback_provider:
            return self.fallback_handler.fallback_provider.is_available()
        
        return False
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check including fallback status."""
        primary_health = self.primary_provider.health_check()
        
        health_info = {
            "provider": "astradb_with_fallback",
            "primary_health": primary_health,
            "fallback_status": self.fallback_handler.get_status()
        }
        
        # Overall status based on availability
        if primary_health.get("status") == "healthy":
            health_info["status"] = "healthy"
            health_info["active_provider"] = "astradb"
        elif self.fallback_handler.is_using_fallback:
            if self.fallback_handler.fallback_provider:
                fallback_health = self.fallback_handler.fallback_provider.health_check()
                health_info["fallback_health"] = fallback_health
                health_info["status"] = fallback_health.get("status", "unknown")
                health_info["active_provider"] = "chromadb_fallback"
            else:
                health_info["status"] = "unhealthy"
                health_info["active_provider"] = "none"
        else:
            health_info["status"] = "degraded"
            health_info["active_provider"] = "astradb_degraded"
        
        return health_info
    
    def get_fallback_status(self) -> Dict[str, Any]:
        """Get detailed fallback status information."""
        return self.fallback_handler.get_status()
    
    def force_fallback(self, enable: bool = True) -> None:
        """Force enable or disable fallback mode."""
        self.fallback_handler.force_fallback(enable)
    
    def sync_to_primary(self) -> Dict[str, Any]:
        """Sync data from fallback to primary when primary recovers."""
        return self.fallback_handler.sync_to_primary()


def create_astradb_error_message(error: Exception, operation: str) -> str:
    """Create user-friendly error messages for AstraDB operations.
    
    Args:
        error: The exception that occurred
        operation: The operation that failed
        
    Returns:
        User-friendly error message with troubleshooting guidance
    """
    error_type = type(error).__name__
    error_message = str(error)
    
    # Common error patterns and solutions
    error_patterns = {
        "authentication": {
            "keywords": ["unauthorized", "authentication", "token", "401"],
            "message": (
                f"AstraDB authentication failed during {operation}. "
                "Please check your ASTRADB_APPLICATION_TOKEN. "
                "Generate a new token at: https://astra.datastax.com/"
            )
        },
        "connection": {
            "keywords": ["connection", "timeout", "network", "endpoint"],
            "message": (
                f"AstraDB connection failed during {operation}. "
                "Please check your ASTRADB_API_ENDPOINT and network connectivity. "
                "Verify the endpoint at: https://astra.datastax.com/"
            )
        },
        "quota": {
            "keywords": ["quota", "limit", "rate", "429"],
            "message": (
                f"AstraDB rate limit or quota exceeded during {operation}. "
                "Please wait a moment and try again, or upgrade your plan."
            )
        },
        "collection": {
            "keywords": ["collection", "not found", "404"],
            "message": (
                f"AstraDB collection issue during {operation}. "
                "The collection may not exist or may have been deleted. "
                "Check your collection configuration."
            )
        },
        "dimension": {
            "keywords": ["dimension", "vector", "embedding"],
            "message": (
                f"AstraDB vector dimension mismatch during {operation}. "
                "Ensure your embeddings match the collection's vector dimension. "
                "You may need to recreate the collection with the correct dimension."
            )
        }
    }
    
    # Check for known error patterns
    error_lower = error_message.lower()
    for pattern_name, pattern_info in error_patterns.items():
        if any(keyword in error_lower for keyword in pattern_info["keywords"]):
            return pattern_info["message"]
    
    # Generic error message
    return (
        f"AstraDB {operation} failed: {error_message}. "
        f"Error type: {error_type}. "
        "Check your AstraDB configuration and network connectivity. "
        "For more help, visit: https://docs.datastax.com/en/astra-serverless/docs/"
    )
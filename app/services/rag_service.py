"""
RAG Service Layer

This module provides the service layer for RAG (Retrieval-Augmented Generation) operations,
orchestrating document retrieval, query processing, and integration with various providers.
"""

import time
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any, Union

from app.core.rag import (
    RAGIntegration,
    RAGConfig,
    EmbeddingProvider as CoreEmbeddingProvider,
    VectorStoreProvider as CoreVectorStoreProvider,
    EmbeddingConfig,
    VectorStoreConfig
)
from app.models.schemas.rag import (
    RAGQueryRequest,
    RAGQueryResponse,
    DocumentIndexRequest,
    DocumentIndexResponse,
    DocumentDeleteRequest,
    DocumentDeleteResponse,
    DocumentSource,
    CollectionInfo,
    CollectionListResponse,
    RAGConfigRequest,
    RAGConfigResponse,
    RAGStatusResponse,
    QueryGenerationRequest,
    QueryGenerationResponse,
    PluginDetectionRequest,
    PluginDetectionResponse,
    SearchSuggestionRequest,
    SearchSuggestionResponse
)
from app.models.enums import TaskType, EmbeddingProvider, VectorStoreProvider
from app.utils.exceptions import ServiceError, ValidationError
from app.utils.logging import get_logger

logger = get_logger(__name__)





class RAGService:
    """Service layer for RAG operations."""
    
    def __init__(self, 
                 helper_model=None,
                 output_dir: str = "output",
                 config: Optional[RAGConfig] = None,
                 embedding_provider: Optional[CoreEmbeddingProvider] = None,
                 vector_store_provider: Optional[CoreVectorStoreProvider] = None,
                 session_id: Optional[str] = None):
        """Initialize RAG service.
        
        Args:
            helper_model: Model used for generating queries and processing text
            output_dir: Directory for output files
            config: RAG configuration
            embedding_provider: Optional embedding provider instance
            vector_store_provider: Optional vector store provider instance
            session_id: Optional session identifier
        """
        self.helper_model = helper_model
        self.output_dir = output_dir
        self.session_id = session_id or str(uuid.uuid4())
        self._collections_cache: Dict[str, CollectionInfo] = {}
        
        # Initialize RAG integration
        try:
            self.rag_integration = RAGIntegration(
                helper_model=helper_model,
                output_dir=output_dir,
                config=config,
                embedding_provider=embedding_provider,
                vector_store_provider=vector_store_provider,
                session_id=session_id
            )
            logger.info("RAG service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG integration: {str(e)}")
            self.rag_integration = None
    
    async def query_documents(self, request: RAGQueryRequest) -> RAGQueryResponse:
        """Query documents using RAG system.
        
        Args:
            request: RAG query request
            
        Returns:
            RAG query response with results
        """
        start_time = time.time()
        
        try:
            if not self.rag_integration:
                raise ServiceError("RAG integration not available")
            
            # Generate specialized queries if needed
            generated_queries = None
            if request.task_type != TaskType.GENERAL:
                generated_queries = await self._generate_specialized_queries(request)
            
            # Use the RAG integration to perform the query
            if hasattr(self.rag_integration, 'context_aware_retriever') and self.rag_integration.context_aware_retriever:
                # Use enhanced retrieval if available
                results = await self._enhanced_query(request)
            else:
                # Use basic vector store query
                results = await self._basic_query(request)
            
            processing_time = time.time() - start_time
            
            return RAGQueryResponse(
                query=request.query,
                results=results,
                total_results=len(results),
                processing_time=processing_time,
                task_type=request.task_type or TaskType.GENERAL,
                generated_queries=generated_queries,
                metadata={
                    "filters_applied": bool(request.filters),
                    "context_provided": bool(request.context),
                    "collection_name": request.collection_name,
                    "similarity_threshold": request.similarity_threshold
                }
            )
            
        except Exception as e:
            logger.error(f"Error querying documents: {str(e)}")
            processing_time = time.time() - start_time
            
            return RAGQueryResponse(
                query=request.query,
                results=[],
                total_results=0,
                processing_time=processing_time,
                task_type=request.task_type or TaskType.GENERAL,
                metadata={"error": str(e)}
            )
    
    async def _generate_specialized_queries(self, request: RAGQueryRequest) -> List[str]:
        """Generate specialized queries based on task type."""
        try:
            if request.task_type == TaskType.STORYBOARD:
                return self.rag_integration._generate_rag_queries_storyboard(
                    scene_plan=request.query,
                    topic=request.context,
                    session_id=self.session_id
                )
            elif request.task_type == TaskType.TECHNICAL:
                return self.rag_integration._generate_rag_queries_technical(
                    storyboard=request.query,
                    topic=request.context,
                    session_id=self.session_id
                )
            elif request.task_type == TaskType.NARRATION:
                return self.rag_integration._generate_rag_queries_narration(
                    storyboard=request.query,
                    topic=request.context,
                    session_id=self.session_id
                )
            return [request.query]
        except Exception as e:
            logger.warning(f"Failed to generate specialized queries: {str(e)}")
            return [request.query]
    
    async def _enhanced_query(self, request: RAGQueryRequest) -> List[DocumentSource]:
        """Perform enhanced query using context-aware retrieval."""
        # This would use the enhanced RAG components
        # For now, fall back to basic query
        return await self._basic_query(request)
    
    async def _basic_query(self, request: RAGQueryRequest) -> List[DocumentSource]:
        """Perform basic vector store query."""
        try:
            # Use the vector store to search for relevant documents
            if hasattr(self.rag_integration, 'vector_store') and self.rag_integration.vector_store:
                # Perform similarity search
                results = self.rag_integration.vector_store.similarity_search(
                    query=request.query,
                    k=request.max_results
                )
                
                # Convert results to DocumentSource format
                formatted_results = []
                for i, result in enumerate(results):
                    score = getattr(result, 'score', None)
                    
                    # Apply similarity threshold if specified
                    if request.similarity_threshold and score and score < request.similarity_threshold:
                        continue
                    
                    formatted_results.append(DocumentSource(
                        content=result.page_content,
                        metadata=result.metadata if request.include_metadata else {},
                        score=score,
                        source_id=result.metadata.get('source_id', f"doc_{i}"),
                        chunk_id=result.metadata.get('chunk_id', f"chunk_{i}"),
                        collection=request.collection_name or result.metadata.get('collection', 'default')
                    ))
                
                return formatted_results
            else:
                logger.warning("Vector store not available")
                return []
                
        except Exception as e:
            logger.error(f"Error in basic query: {str(e)}")
            return []
    
    async def index_documents(self, request: DocumentIndexRequest) -> DocumentIndexResponse:
        """Index documents into the vector store.
        
        Args:
            request: Document indexing request
            
        Returns:
            Document indexing response
        """
        start_time = time.time()
        
        indexed_count = 0
        failed_count = 0
        errors = []
        total_chunks = 0
        
        try:
            if not self.rag_integration:
                raise ServiceError("RAG integration not available")
            
            if hasattr(self.rag_integration, 'vector_store') and self.rag_integration.vector_store:
                for doc in request.documents:
                    try:
                        # Add document to vector store
                        # This would need to be implemented based on the specific vector store interface
                        # For now, simulate chunking and indexing
                        content = doc.get('content', '')
                        if content:
                            # Simulate chunking
                            chunk_size = request.chunk_size or 1000
                            chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
                            total_chunks += len(chunks)
                            
                            # In a real implementation, you would:
                            # 1. Split the document into chunks
                            # 2. Generate embeddings for each chunk
                            # 3. Store in vector database
                            indexed_count += 1
                        else:
                            failed_count += 1
                            errors.append("Document has no content")
                            
                    except Exception as e:
                        failed_count += 1
                        errors.append(f"Failed to index document: {str(e)}")
                        
                # Update collections cache
                if request.collection_name and indexed_count > 0:
                    self._update_collection_cache(request.collection_name, indexed_count, total_chunks)
                    
            else:
                errors.append("Vector store not available")
                failed_count = len(request.documents)
            
            processing_time = time.time() - start_time
            
            return DocumentIndexResponse(
                indexed_count=indexed_count,
                failed_count=failed_count,
                processing_time=processing_time,
                errors=errors,
                collection_name=request.collection_name,
                total_chunks=total_chunks
            )
            
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            processing_time = time.time() - start_time
            
            return DocumentIndexResponse(
                indexed_count=0,
                failed_count=len(request.documents),
                processing_time=processing_time,
                errors=[str(e)],
                collection_name=request.collection_name
            )
    
    def generate_rag_queries(self, 
                           content: str, 
                           task_type: str = "general",
                           topic: Optional[str] = None,
                           scene_number: Optional[int] = None,
                           relevant_plugins: Optional[List[str]] = None) -> List[str]:
        """Generate RAG queries for given content.
        
        Args:
            content: Content to generate queries for
            task_type: Type of task (storyboard, technical, narration, etc.)
            topic: Optional topic name
            scene_number: Optional scene number
            relevant_plugins: Optional list of relevant plugins
            
        Returns:
            List of generated queries
        """
        try:
            relevant_plugins = relevant_plugins or []
            
            if task_type == "storyboard":
                return self.rag_integration._generate_rag_queries_storyboard(
                    scene_plan=content,
                    topic=topic,
                    scene_number=scene_number,
                    session_id=self.session_id,
                    relevant_plugins=relevant_plugins
                )
            elif task_type == "technical":
                return self.rag_integration._generate_rag_queries_technical(
                    storyboard=content,
                    topic=topic,
                    scene_number=scene_number,
                    session_id=self.session_id,
                    relevant_plugins=relevant_plugins
                )
            elif task_type == "narration":
                return self.rag_integration._generate_rag_queries_narration(
                    storyboard=content,
                    topic=topic,
                    scene_number=scene_number,
                    session_id=self.session_id,
                    relevant_plugins=relevant_plugins
                )
            else:
                # Generic query generation
                return [content]
                
        except Exception as e:
            self.logger.error(f"Error generating RAG queries: {str(e)}")
            return []
    
    def detect_relevant_plugins(self, topic: str, description: str) -> List[str]:
        """Detect relevant plugins for given topic and description.
        
        Args:
            topic: Topic of the content
            description: Description of the content
            
        Returns:
            List of relevant plugin names
        """
        try:
            return self.rag_integration.detect_relevant_plugins(topic, description)
        except Exception as e:
            self.logger.error(f"Error detecting relevant plugins: {str(e)}")
            return []
    
    def set_relevant_plugins(self, plugins: List[str]) -> None:
        """Set relevant plugins for the current session.
        
        Args:
            plugins: List of plugin names to set as relevant
        """
        try:
            self.rag_integration.set_relevant_plugins(plugins)
        except Exception as e:
            self.logger.error(f"Error setting relevant plugins: {str(e)}")
    
    async def delete_documents(self, request: DocumentDeleteRequest) -> DocumentDeleteResponse:
        """Delete documents from the vector store."""
        start_time = time.time()
        
        try:
            if not self.rag_integration:
                raise ServiceError("RAG integration not available")
            
            deleted_count = 0
            
            # In a real implementation, you would delete documents based on:
            # - document_ids: specific document IDs
            # - filters: metadata filters
            # - delete_all: delete all documents in collection
            
            if request.delete_all and request.collection_name:
                # Simulate deleting all documents in collection
                if request.collection_name in self._collections_cache:
                    deleted_count = self._collections_cache[request.collection_name].document_count
                    del self._collections_cache[request.collection_name]
            elif request.document_ids:
                deleted_count = len(request.document_ids)
            elif request.filters:
                # Simulate filtered deletion
                deleted_count = 1  # Would depend on actual filter results
            
            processing_time = time.time() - start_time
            
            return DocumentDeleteResponse(
                deleted_count=deleted_count,
                processing_time=processing_time,
                collection_name=request.collection_name
            )
            
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            processing_time = time.time() - start_time
            
            return DocumentDeleteResponse(
                deleted_count=0,
                processing_time=processing_time,
                collection_name=request.collection_name
            )
    
    async def list_collections(self) -> CollectionListResponse:
        """List all available collections."""
        try:
            collections = list(self._collections_cache.values())
            
            return CollectionListResponse(
                collections=collections,
                total_count=len(collections)
            )
            
        except Exception as e:
            logger.error(f"Error listing collections: {str(e)}")
            return CollectionListResponse(collections=[], total_count=0)
    
    async def generate_queries(self, request: QueryGenerationRequest) -> QueryGenerationResponse:
        """Generate specialized queries for given content."""
        start_time = time.time()
        
        try:
            generated_queries = self.generate_rag_queries(
                content=request.content,
                task_type=request.task_type.value if request.task_type else "general",
                topic=request.topic,
                scene_number=request.scene_number,
                relevant_plugins=request.relevant_plugins
            )
            
            # Limit to max_queries
            if len(generated_queries) > request.max_queries:
                generated_queries = generated_queries[:request.max_queries]
            
            processing_time = time.time() - start_time
            
            return QueryGenerationResponse(
                original_content=request.content,
                generated_queries=generated_queries,
                task_type=request.task_type,
                processing_time=processing_time,
                metadata={
                    "topic": request.topic,
                    "scene_number": request.scene_number,
                    "relevant_plugins": request.relevant_plugins
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating queries: {str(e)}")
            processing_time = time.time() - start_time
            
            return QueryGenerationResponse(
                original_content=request.content,
                generated_queries=[request.content],  # Fallback to original content
                task_type=request.task_type,
                processing_time=processing_time,
                metadata={"error": str(e)}
            )
    
    async def detect_plugins(self, request: PluginDetectionRequest) -> PluginDetectionResponse:
        """Detect relevant plugins for given topic and description."""
        start_time = time.time()
        
        try:
            relevant_plugins = self.detect_relevant_plugins(request.topic, request.description)
            
            # Filter by available plugins if provided
            if request.available_plugins:
                relevant_plugins = [p for p in relevant_plugins if p in request.available_plugins]
            
            processing_time = time.time() - start_time
            
            return PluginDetectionResponse(
                topic=request.topic,
                description=request.description,
                relevant_plugins=relevant_plugins,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error detecting plugins: {str(e)}")
            processing_time = time.time() - start_time
            
            return PluginDetectionResponse(
                topic=request.topic,
                description=request.description,
                relevant_plugins=[],
                processing_time=processing_time
            )
    
    async def get_search_suggestions(self, request: SearchSuggestionRequest) -> SearchSuggestionResponse:
        """Get search suggestions for partial query."""
        start_time = time.time()
        
        try:
            # In a real implementation, this would:
            # 1. Use the vector store to find similar queries
            # 2. Use query completion models
            # 3. Analyze historical queries
            
            # For now, provide basic suggestions
            suggestions = [
                f"{request.partial_query} examples",
                f"{request.partial_query} tutorial",
                f"{request.partial_query} documentation",
                f"how to {request.partial_query}",
                f"{request.partial_query} best practices"
            ]
            
            # Limit to max_suggestions
            suggestions = suggestions[:request.max_suggestions]
            
            processing_time = time.time() - start_time
            
            return SearchSuggestionResponse(
                partial_query=request.partial_query,
                suggestions=suggestions,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error getting search suggestions: {str(e)}")
            processing_time = time.time() - start_time
            
            return SearchSuggestionResponse(
                partial_query=request.partial_query,
                suggestions=[],
                processing_time=processing_time
            )
    
    def get_service_status(self) -> RAGStatusResponse:
        """Get current service status and configuration."""
        try:
            vector_store_available = (
                self.rag_integration and 
                hasattr(self.rag_integration, 'vector_store') and 
                self.rag_integration.vector_store is not None
            )
            
            embedding_provider_available = (
                self.rag_integration and 
                hasattr(self.rag_integration, 'embedding_provider') and 
                self.rag_integration.embedding_provider is not None
            )
            
            enhanced_components_available = (
                self.rag_integration and 
                hasattr(self.rag_integration, 'enhanced_query_generator') and 
                self.rag_integration.enhanced_query_generator is not None
            )
            
            total_documents = sum(
                collection.document_count for collection in self._collections_cache.values()
            )
            
            return RAGStatusResponse(
                service="RAG Service",
                status="active" if self.rag_integration else "error",
                session_id=self.session_id,
                vector_store_available=vector_store_available,
                embedding_provider_available=embedding_provider_available,
                enhanced_components_available=enhanced_components_available,
                collections_count=len(self._collections_cache),
                total_documents=total_documents,
                last_updated=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error getting service status: {str(e)}")
            return RAGStatusResponse(
                service="RAG Service",
                status="error",
                session_id=self.session_id,
                vector_store_available=False,
                embedding_provider_available=False,
                enhanced_components_available=False
            )
    
    def _update_collection_cache(self, collection_name: str, doc_count: int, chunk_count: int):
        """Update the collections cache with new information."""
        if collection_name in self._collections_cache:
            collection = self._collections_cache[collection_name]
            collection.document_count += doc_count
            collection.total_chunks += chunk_count
            collection.updated_at = datetime.utcnow()
        else:
            self._collections_cache[collection_name] = CollectionInfo(
                name=collection_name,
                document_count=doc_count,
                total_chunks=chunk_count,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
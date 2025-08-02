"""
RAGAgent with Context7 and document processing integration.
Ports existing RAG functionality while adding enhanced capabilities.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from langgraph.types import Command

from ..base_agent import BaseAgent
from ..state import VideoGenerationState
from src.rag.rag_integration import RAGIntegration
from src.rag.vector_store import RAGVectorStore

logger = logging.getLogger(__name__)

# Cache file encoding constant
CACHE_FILE_ENCODING = 'utf-8'


class RAGAgent(BaseAgent):
    """Agent for RAG query generation, context retrieval, and document processing.
    
    Ports existing RAG functionality from _initialize_vector_store and _retrieve_rag_context methods
    while maintaining compatibility with current ChromaDB integration and embedding model configurations.
    Integrates Context7 for enhanced documentation retrieval and Docling for document processing.
    """
    
    def __init__(self, config, system_config):
        """Initialize RAGAgent with vector store and external tool integrations."""
        super().__init__(config, system_config)
        
        # Initialize RAG integration
        self.rag_integration = None
        self.vector_store = None
        
        # Get centralized configuration manager
        from src.config.manager import ConfigurationManager
        self.config_manager = ConfigurationManager()
        
        # Context7 and Docling configurations
        self.context7_config = system_config.context7_config
        self.docling_config = system_config.docling_config
        self.mcp_servers = system_config.mcp_servers
        
        # Cache for Context7 responses
        self.context7_cache = {}
        
        self.log_agent_action("initialized_rag_agent", {
            "context7_enabled": self.context7_config.enabled,
            "docling_enabled": self.docling_config.enabled,
            "mcp_servers": list(self.mcp_servers.keys()),
            "rag_enabled": self.config_manager.is_rag_enabled()
        })
    
    async def execute(self, state: VideoGenerationState) -> Command:
        """Execute RAG operations based on current workflow state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Command: Next action based on RAG results
        """
        self.log_agent_action("starting_rag_execution", {
            "use_rag": state.get('use_rag', False),
            "current_agent": state.get('current_agent', ''),
            "next_agent": state.get('next_agent', '')
        })
        
        try:
            # Initialize RAG components if not already done
            if not self.rag_integration:
                await self._initialize_rag_components(state)
            
            # Determine what RAG operation to perform based on state
            requesting_agent = state.get('current_agent', '')
            
            if requesting_agent == 'code_generator_agent':
                # Generate RAG queries and retrieve context for code generation
                return await self._handle_code_generation_rag(state)
            elif requesting_agent == 'planner_agent':
                # Generate RAG queries for planning
                return await self._handle_planning_rag(state)
            elif requesting_agent == 'error_handler_agent':
                # Generate RAG queries for error fixing
                return await self._handle_error_fixing_rag(state)
            else:
                # General RAG query handling
                return await self._handle_general_rag_query(state)
                
        except Exception as e:
            logger.error(f"Error in RAG agent execution: {e}")
            return await self.handle_error(e, state)
    
    async def _initialize_rag_components(self, state: VideoGenerationState):
        """Initialize RAG integration and vector store components using centralized configuration.
        
        Uses centralized configuration manager for RAG settings while maintaining compatibility.
        """
        try:
            # Check if RAG is enabled in centralized configuration
            if not self.config_manager.is_rag_enabled():
                logger.warning("RAG is disabled in centralized configuration")
                return
            
            # Get RAG configuration from centralized config
            rag_config = self.config_manager.get_rag_config()
            
            # Extract configuration from state (legacy parameters)
            session_id = state.get('session_id', '')
            use_langfuse = state.get('use_langfuse', True)
            output_dir = state.get('output_dir', 'output')
            
            # Override with centralized configuration values
            if rag_config:
                chroma_db_path = rag_config.vector_store_config.connection_params.get('path', 'data/rag/chroma_db')
                manim_docs_path = state.get('manim_docs_path', 'data/rag/manim_docs')  # Keep from state for now
                embedding_model = rag_config.embedding_config.model_name
                
                self.log_agent_action("using_centralized_rag_config", {
                    "embedding_provider": rag_config.embedding_config.provider,
                    "embedding_model": embedding_model,
                    "vector_store_provider": rag_config.vector_store_config.provider,
                    "collection_name": rag_config.vector_store_config.collection_name
                })
            else:
                # Fallback to state values
                chroma_db_path = state.get('chroma_db_path', 'data/rag/chroma_db')
                manim_docs_path = state.get('manim_docs_path', 'data/rag/manim_docs')
                embedding_model = state.get('embedding_model', 'hf:ibm-granite/granite-embedding-30m-english')
                
                self.log_agent_action("using_legacy_rag_config", {
                    "chroma_db_path": chroma_db_path,
                    "embedding_model": embedding_model
                })
            
            # Get helper model for RAG queries
            helper_model = self._get_helper_model(state)
            
            # Initialize RAG integration with centralized configuration
            self.rag_integration = RAGIntegration(
                helper_model=helper_model,
                output_dir=output_dir,
                chroma_db_path=chroma_db_path,
                manim_docs_path=manim_docs_path,
                embedding_model=embedding_model,
                session_id=session_id,
                use_langfuse=use_langfuse,
                config=self._create_rag_config(state),
                config_manager=self.config_manager
            )
            
            # Set vector store reference for compatibility
            self.vector_store = self.rag_integration.vector_store
            
            self.log_agent_action("rag_components_initialized", {
                "chroma_db_path": chroma_db_path,
                "embedding_model": embedding_model,
                "enhanced_components": hasattr(self.rag_integration, 'enhanced_query_generator'),
                "using_centralized_config": rag_config is not None
            })
            
        except Exception as e:
            logger.error(f"Error initializing RAG components: {e}")
            raise
    
    async def _handle_code_generation_rag(self, state: VideoGenerationState) -> Command:
        """Handle RAG operations for code generation.
        
        Args:
            state: Current workflow state
            
        Returns:
            Command: Return to code generator with RAG context
        """
        try:
            # Get scene implementation for RAG query generation
            scene_implementations = state.get('scene_implementations', {})
            generated_code = state.get('generated_code', {})
            
            # Process each scene that needs RAG context
            updated_rag_context = state.get('rag_context', {})
            
            for scene_number, implementation in scene_implementations.items():
                if scene_number not in generated_code:  # Only process scenes that haven't been generated yet
                    # Generate RAG queries for this scene
                    rag_queries = await self._generate_rag_queries_code(
                        implementation=implementation,
                        scene_number=scene_number,
                        state=state
                    )
                    
                    if rag_queries:
                        # Retrieve RAG context
                        rag_context = await self._retrieve_rag_context(
                            rag_queries=rag_queries,
                            scene_number=scene_number,
                            state=state
                        )
                        
                        if rag_context:
                            updated_rag_context[scene_number] = rag_context
            
            # Return to code generator with updated RAG context
            return Command(
                goto="code_generator_agent",
                update={
                    "rag_context": updated_rag_context,
                    "current_agent": "code_generator_agent"
                }
            )
            
        except Exception as e:
            logger.error(f"Error in code generation RAG handling: {e}")
            return await self.handle_error(e, state)
    
    async def _handle_planning_rag(self, state: VideoGenerationState) -> Command:
        """Handle RAG operations for planning.
        
        Args:
            state: Current workflow state
            
        Returns:
            Command: Return to planner with RAG context
        """
        try:
            topic = state.get('topic', '')
            description = state.get('description', '')
            
            # Generate planning-specific RAG queries
            planning_queries = await self._generate_planning_rag_queries(
                topic=topic,
                description=description,
                state=state
            )
            
            if planning_queries:
                # Retrieve context for planning
                planning_context = await self._retrieve_rag_context(
                    rag_queries=planning_queries,
                    scene_number=0,  # Planning is scene 0
                    state=state
                )
                
                return Command(
                    goto="planner_agent",
                    update={
                        "rag_context": {0: planning_context} if planning_context else {},
                        "current_agent": "planner_agent"
                    }
                )
            else:
                # No RAG queries generated, return to planner
                return Command(
                    goto="planner_agent",
                    update={"current_agent": "planner_agent"}
                )
                
        except Exception as e:
            logger.error(f"Error in planning RAG handling: {e}")
            return await self.handle_error(e, state)
    
    async def _handle_error_fixing_rag(self, state: VideoGenerationState) -> Command:
        """Handle RAG operations for error fixing.
        
        Args:
            state: Current workflow state
            
        Returns:
            Command: Return to error handler with RAG context
        """
        try:
            # Get error information
            escalated_errors = state.get('escalated_errors', [])
            code_errors = state.get('code_errors', {})
            
            # Generate error-fixing RAG queries
            error_context = {}
            
            for scene_number, error_info in code_errors.items():
                error_queries = await self._generate_error_fixing_rag_queries(
                    error_info=error_info,
                    scene_number=scene_number,
                    state=state
                )
                
                if error_queries:
                    error_rag_context = await self._retrieve_rag_context(
                        rag_queries=error_queries,
                        scene_number=scene_number,
                        state=state
                    )
                    
                    if error_rag_context:
                        error_context[scene_number] = error_rag_context
            
            return Command(
                goto="error_handler_agent",
                update={
                    "rag_context": {**state.get('rag_context', {}), **error_context},
                    "current_agent": "error_handler_agent"
                }
            )
            
        except Exception as e:
            logger.error(f"Error in error fixing RAG handling: {e}")
            return await self.handle_error(e, state)
    
    async def _handle_general_rag_query(self, state: VideoGenerationState) -> Command:
        """Handle general RAG queries.
        
        Args:
            state: Current workflow state
            
        Returns:
            Command: Return to requesting agent with RAG context
        """
        # For general queries, return to the next agent specified in state
        next_agent = state.get('next_agent', 'code_generator_agent')
        
        return Command(
            goto=next_agent,
            update={"current_agent": next_agent}
        )
    
    async def _generate_rag_queries_code(self,
                                       implementation: str,
                                       scene_number: int,
                                       state: VideoGenerationState) -> List[str]:
        """Generate RAG queries from implementation plan.
        
        Preserves current query caching logic from _load_cached_queries and _save_queries_to_cache.
        
        Args:
            implementation: Implementation plan text
            scene_number: Scene number
            state: Current workflow state
            
        Returns:
            List of generated RAG queries
        """
        try:
            if not self.rag_integration:
                return []
            
            topic = state.get('topic', '')
            detected_plugins = state.get('detected_plugins', [])
            session_id = state.get('session_id', '')
            
            # Use RAG integration to generate queries
            queries = self.rag_integration._generate_rag_queries_code(
                implementation_plan=implementation,
                scene_trace_id=session_id,
                topic=topic,
                scene_number=scene_number,
                relevant_plugins=detected_plugins
            )
            
            self.log_agent_action("rag_queries_generated", {
                "scene_number": scene_number,
                "query_count": len(queries),
                "plugins": detected_plugins
            })
            
            return queries
            
        except Exception as e:
            logger.error(f"Error generating RAG queries for scene {scene_number}: {e}")
            return []
    
    async def _generate_planning_rag_queries(self,
                                           topic: str,
                                           description: str,
                                           state: VideoGenerationState) -> List[str]:
        """Generate RAG queries for planning phase.
        
        Args:
            topic: Video topic
            description: Video description
            state: Current workflow state
            
        Returns:
            List of planning-specific RAG queries
        """
        try:
            # Create planning-specific queries
            planning_queries = [
                f"How to create educational animations about {topic}",
                f"Manim examples for {topic} visualization",
                f"Best practices for {topic} animation sequences",
                f"Educational video structure for {topic}",
                f"Visual elements for explaining {topic}"
            ]
            
            # Add Context7 queries if enabled
            if self.context7_config.enabled:
                context7_queries = await self._generate_context7_queries(topic, description)
                planning_queries.extend(context7_queries)
            
            self.log_agent_action("planning_queries_generated", {
                "topic": topic,
                "query_count": len(planning_queries)
            })
            
            return planning_queries
            
        except Exception as e:
            logger.error(f"Error generating planning RAG queries: {e}")
            return []
    
    async def _generate_error_fixing_rag_queries(self,
                                               error_info: str,
                                               scene_number: int,
                                               state: VideoGenerationState) -> List[str]:
        """Generate RAG queries for error fixing.
        
        Args:
            error_info: Error information
            scene_number: Scene number with error
            state: Current workflow state
            
        Returns:
            List of error-fixing RAG queries
        """
        try:
            if not self.rag_integration:
                return []
            
            topic = state.get('topic', '')
            session_id = state.get('session_id', '')
            
            # Use RAG integration to generate error-fixing queries
            queries = self.rag_integration._generate_rag_queries_error_fix(
                error_message=error_info,
                scene_trace_id=session_id,
                topic=topic,
                scene_number=scene_number
            )
            
            self.log_agent_action("error_fix_queries_generated", {
                "scene_number": scene_number,
                "query_count": len(queries)
            })
            
            return queries
            
        except Exception as e:
            logger.error(f"Error generating error-fixing RAG queries: {e}")
            return []
    
    async def _retrieve_rag_context(self,
                                  rag_queries: List[str],
                                  scene_number: int,
                                  state: VideoGenerationState) -> Optional[str]:
        """Retrieve context from RAG vector store and external sources.
        
        Maintains compatibility with current chroma_db_path structure while adding
        Context7 and Docling integration.
        
        Args:
            rag_queries: List of RAG queries
            scene_number: Scene number
            state: Current workflow state
            
        Returns:
            Retrieved context string or None
        """
        try:
            if not rag_queries:
                return None
            
            contexts = []
            
            # 1. Retrieve from traditional RAG vector store
            if self.vector_store:
                try:
                    vector_context = self.vector_store.find_relevant_docs(
                        queries=rag_queries,
                        k=5,  # Default RAG K value
                        trace_id=state.get('session_id', ''),
                        topic=state.get('topic', ''),
                        scene_number=scene_number
                    )
                    if vector_context:
                        contexts.append(f"=== Vector Store Context ===\n{vector_context}")
                except Exception as e:
                    logger.warning(f"Error retrieving vector store context: {e}")
            
            # 2. Retrieve from Context7 if enabled
            if self.context7_config.enabled:
                try:
                    context7_context = await self._retrieve_context7_context(rag_queries, state)
                    if context7_context:
                        contexts.append(f"=== Context7 Documentation ===\n{context7_context}")
                except Exception as e:
                    logger.warning(f"Error retrieving Context7 context: {e}")
            
            # 3. Process documents with Docling if enabled
            if self.docling_config.enabled:
                try:
                    docling_context = await self._retrieve_docling_context(rag_queries, state)
                    if docling_context:
                        contexts.append(f"=== Document Processing Context ===\n{docling_context}")
                except Exception as e:
                    logger.warning(f"Error retrieving Docling context: {e}")
            
            # Combine all contexts
            if contexts:
                combined_context = "\n\n".join(contexts)
                
                self.log_agent_action("rag_context_retrieved", {
                    "scene_number": scene_number,
                    "context_sources": len(contexts),
                    "context_length": len(combined_context)
                })
                
                return combined_context
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving RAG context for scene {scene_number}: {e}")
            return None
    
    async def _generate_context7_queries(self, topic: str, description: str) -> List[str]:
        """Generate Context7-specific queries.
        
        Args:
            topic: Video topic
            description: Video description
            
        Returns:
            List of Context7 queries
        """
        # Generate library-specific queries based on topic
        context7_queries = []
        
        # Map topics to potential libraries
        library_mappings = {
            'machine learning': ['scikit-learn', 'tensorflow', 'pytorch'],
            'data science': ['pandas', 'numpy', 'matplotlib'],
            'web development': ['react', 'vue', 'angular'],
            'python': ['python', 'django', 'flask'],
            'mathematics': ['numpy', 'scipy', 'sympy'],
            'visualization': ['matplotlib', 'plotly', 'seaborn']
        }
        
        # Find relevant libraries
        topic_lower = topic.lower()
        for key, libraries in library_mappings.items():
            if key in topic_lower or any(lib in topic_lower for lib in libraries):
                for lib in libraries:
                    context7_queries.append(f"{lib} documentation for {topic}")
        
        return context7_queries
    
    async def _retrieve_context7_context(self, queries: List[str], state: VideoGenerationState) -> Optional[str]:
        """Retrieve context from Context7 for enhanced documentation.
        
        Integrates Context7 for enhanced documentation retrieval as additional context source.
        
        Args:
            queries: List of queries
            state: Current workflow state
            
        Returns:
            Context7 documentation context
        """
        try:
            # Check if Context7 MCP server is available
            if 'context7' not in self.mcp_servers or self.mcp_servers['context7'].get('disabled', False):
                return None
            
            context7_results = []
            
            for query in queries[:3]:  # Limit to first 3 queries to avoid rate limits
                # Check cache first
                cache_key = f"context7_{hash(query)}"
                if cache_key in self.context7_cache:
                    context7_results.append(self.context7_cache[cache_key])
                    continue
                
                try:
                    # This would be implemented with actual MCP client calls
                    # For now, we'll simulate the structure
                    
                    # Step 1: Resolve library ID
                    # library_id = await mcp_client.call("resolve_library_id", {"libraryName": query})
                    
                    # Step 2: Get library docs
                    # docs = await mcp_client.call("get_library_docs", {
                    #     "context7CompatibleLibraryID": library_id,
                    #     "tokens": self.context7_config.default_tokens,
                    #     "topic": query
                    # })
                    
                    # Placeholder for actual implementation
                    docs = f"Context7 documentation for: {query}\n[This would contain actual documentation from Context7]"
                    
                    # Cache the result
                    self.context7_cache[cache_key] = docs
                    context7_results.append(docs)
                    
                except Exception as e:
                    logger.warning(f"Error querying Context7 for '{query}': {e}")
                    continue
            
            if context7_results:
                return "\n\n".join(context7_results)
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving Context7 context: {e}")
            return None
    
    async def _retrieve_docling_context(self, queries: List[str], state: VideoGenerationState) -> Optional[str]:
        """Retrieve context using Docling document processing.
        
        Adds Docling integration for document processing while maintaining existing RAG workflow.
        
        Args:
            queries: List of queries
            state: Current workflow state
            
        Returns:
            Processed document context
        """
        try:
            # Check if Docling MCP server is available
            if 'docling' not in self.mcp_servers or self.mcp_servers['docling'].get('disabled', False):
                return None
            
            # Look for relevant documents in the project
            doc_paths = self._find_relevant_documents(queries, state)
            
            if not doc_paths:
                return None
            
            docling_results = []
            
            for doc_path in doc_paths[:5]:  # Limit to 5 documents
                try:
                    # Check file size limit
                    max_size_mb = self.docling_config.max_file_size_mb
                    if os.path.getsize(doc_path) > max_size_mb * 1024 * 1024:
                        logger.warning(f"Document {doc_path} exceeds size limit")
                        continue
                    
                    # This would be implemented with actual MCP client calls
                    # processed_doc = await mcp_client.call("docling_document_processor", {
                    #     "document_path": doc_path
                    # })
                    
                    # Placeholder for actual implementation
                    processed_doc = f"Processed document: {doc_path}\n[This would contain actual processed content from Docling]"
                    
                    docling_results.append(processed_doc)
                    
                except Exception as e:
                    logger.warning(f"Error processing document {doc_path} with Docling: {e}")
                    continue
            
            if docling_results:
                return "\n\n".join(docling_results)
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving Docling context: {e}")
            return None
    
    def _find_relevant_documents(self, queries: List[str], state: VideoGenerationState) -> List[str]:
        """Find relevant documents for processing.
        
        Args:
            queries: List of queries to find documents for
            state: Current workflow state
            
        Returns:
            List of document paths
        """
        doc_paths = []
        
        # Common document locations
        search_paths = [
            'docs/',
            'documentation/',
            'README.md',
            'GUIDE.md',
            state.get('manim_docs_path', 'data/rag/manim_docs')
        ]
        
        # Common document extensions
        doc_extensions = ['.md', '.txt', '.rst', '.pdf', '.docx']
        
        for search_path in search_paths:
            if os.path.exists(search_path):
                if os.path.isfile(search_path):
                    doc_paths.append(search_path)
                elif os.path.isdir(search_path):
                    for root, dirs, files in os.walk(search_path):
                        for file in files:
                            if any(file.lower().endswith(ext) for ext in doc_extensions):
                                doc_paths.append(os.path.join(root, file))
        
        return doc_paths
    
    def _create_rag_config(self, state: VideoGenerationState):
        """Create RAG configuration from state.
        
        Args:
            state: Current workflow state
            
        Returns:
            RAG configuration object
        """
        # This would create a proper RAG config object
        # For now, return a simple dict
        return {
            'use_enhanced_components': state.get('use_enhanced_rag', True),
            'enable_caching': state.get('enable_rag_caching', True),
            'cache_ttl': state.get('rag_cache_ttl', 3600),
            'max_cache_size': state.get('rag_max_cache_size', 1000),
            'enable_quality_monitoring': state.get('enable_quality_monitoring', True),
            'enable_error_handling': state.get('enable_error_handling', True),
            'performance_threshold': state.get('rag_performance_threshold', 2.0),
            'quality_threshold': state.get('rag_quality_threshold', 0.7)
        }
    
    def _get_helper_model(self, state: VideoGenerationState):
        """Get helper model for RAG queries.
        
        Args:
            state: Current workflow state
            
        Returns:
            Model wrapper for RAG queries
        """
        # Use helper model from config if available
        if hasattr(self.config, 'helper_model') and self.config.helper_model:
            return self.get_model_wrapper(self.config.helper_model, state)
        
        # Fallback to default model configuration
        model_config = self.config.model_config
        if 'helper_model' in model_config:
            return self.get_model_wrapper(model_config['helper_model'], state)
        
        # Final fallback to any available model
        if 'default_model' in model_config:
            return self.get_model_wrapper(model_config['default_model'], state)
        
        raise ValueError("No helper model configured for RAG queries")
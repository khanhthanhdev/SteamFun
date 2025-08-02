#!/usr/bin/env python3
"""
Create embeddings using JINA AI and store in AstraDB.

This script handles:
1. Document chunking 
2. Embedding generation using JINA AI
3. Vector storage in AstraDB using astrapy

Prerequisites:
- JINA_API_KEY environment variable
- ASTRADB_APPLICATION_TOKEN environment variable
- ASTRA_DB_API_ENDPOINT environment variable
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime

# Third-party imports
try:
    from astrapy import DataAPIClient
    from astrapy.constants import VectorMetric
    from astrapy.exceptions import DataAPIException
    from astrapy.info import CollectionDefinition
    import requests
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install: pip install astrapy requests tqdm")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class JinaConfig:
    """Configuration for JINA AI embedding."""
    api_key: str
    model: str = "jina-embeddings-v3"
    task: str = "retrieval.passage"
    dimensions: int = 1024
    batch_size: int = 32
    base_url: str = "https://api.jina.ai/v1/embeddings"

@dataclass
class AstraConfig:
    """Configuration for AstraDB."""
    application_token: str
    api_endpoint: str
    collection_name: str = "manim_docs_embeddings"
    keyspace: Optional[str] = None
    dimension: int = 1024
    metric: str = "cosine"

@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    chunk_size: int = 1000
    overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000

@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata."""
    content: str
    source_file: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class JinaEmbeddingProvider:
    """JINA AI embedding provider."""
    
    def __init__(self, config: JinaConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {config.api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            # Process in batches
            all_embeddings = []
            
            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i:i + self.config.batch_size]
                
                payload = {
                    "model": self.config.model,
                    "task": self.config.task,
                    "dimensions": self.config.dimensions,
                    "input": batch
                }
                
                response = self.session.post(self.config.base_url, json=payload)
                response.raise_for_status()
                
                result = response.json()
                
                # Extract embeddings from response
                batch_embeddings = []
                for item in result.get('data', []):
                    batch_embeddings.append(item['embedding'])
                
                all_embeddings.extend(batch_embeddings)
                
                logger.info(f"Generated embeddings for batch {i//self.config.batch_size + 1}")
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector
        """
        embeddings = self.embed_texts([text])
        return embeddings[0] if embeddings else []

# Local imports for chunking and relationship detection
try:
    import sys
    sys.path.append('src')
    from rag.vector_store import CodeAwareTextSplitter
    from rag.chunk_relationships import ChunkRelationshipDetector, ChunkRelationship
    from langchain.schema import Document
    ADVANCED_CHUNKING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Advanced chunking not available: {e}")
    ADVANCED_CHUNKING_AVAILABLE = False
    Document = None
    ChunkRelationship = None


class DocumentChunker:
    """Document chunking using advanced CodeAwareTextSplitter or fallback to simple chunking."""
    
    def __init__(self):
        if ADVANCED_CHUNKING_AVAILABLE:
            self.splitter = CodeAwareTextSplitter()
            self.relationship_detector = ChunkRelationshipDetector()
        else:
            self.splitter = None
            self.relationship_detector = None

    def chunk_and_detect(self, text: str, source_file: str, metadata: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
        """Chunk text and detect relationships.

        Args:
            text: Text content to chunk
            source_file: Source file path
            metadata: Additional metadata

        Returns:
            List of document chunks and relationships
        """
        
        if ADVANCED_CHUNKING_AVAILABLE:
            # Use advanced chunking with relationship detection
            doc = Document(page_content=text, metadata=metadata)
            chunks, relationships = self.splitter.split_documents_with_relationships([doc])
            
            logger.info(f"Created {len(chunks)} chunks and detected {len(relationships)} relationships from {source_file}")
            
            return chunks, relationships
        else:
            # Fallback to simple chunking
            logger.warning("Using fallback simple chunking")
            chunks = self._simple_chunk(text, source_file, metadata)
            return chunks, {}
    
    def _simple_chunk(self, text: str, source_file: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Simple fallback chunbing method."""
        chunks = []
        words = text.split()
        chunk_size = 200  # Simple chunk size in words
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            content = ' '.join(chunk_words)
            
            if len(content.strip()) < 50:  # Skip very small chunks
                continue
            
            chunk_metadata = {
                **metadata,
                'chunk_method': 'simple_fallback',
                'created_at': datetime.utcnow().isoformat()
            }
            
            chunk = DocumentChunk(
                content=content,
                source_file=source_file,
                chunk_index=len(chunks),
                start_char=0,
                end_char=len(content),
                metadata=chunk_metadata
            )
            chunks.append(chunk)
        
        return chunks

class AstraDBVectorStore:
    """AstraDB vector store using astrapy."""
    
    def __init__(self, config: AstraConfig):
        self.config = config
        self.client = None
        self.database = None
        self.collection = None
        self._initialize()
    
    def _initialize(self):
        """Initialize AstraDB connection."""
        try:
            # Initialize DataAPI client (no token needed here)
            self.client = DataAPIClient()
            
            # Connect to database with token
            self.database = self.client.get_database(
                self.config.api_endpoint,
                token=self.config.application_token
            )
            
            logger.info(f"Connected to AstraDB: {self.config.api_endpoint}")
            
            # Create or get collection
            self._setup_collection()
            
        except Exception as e:
            logger.error(f"Failed to initialize AstraDB: {e}")
            raise
    
    def _setup_collection(self):
        """Setup the vector collection."""
        try:
            # Check if collection exists
            collections = self.database.list_collection_names()
            
            if self.config.collection_name in collections:
                logger.info(f"Using existing collection: {self.config.collection_name}")
                self.collection = self.database.get_collection(self.config.collection_name)
            else:
                logger.info(f"Creating new collection: {self.config.collection_name}")
                
                # Determine vector metric
                vector_metric = VectorMetric.COSINE
                if self.config.metric.lower() == "dot_product":
                    vector_metric = VectorMetric.DOT_PRODUCT
                elif self.config.metric.lower() == "euclidean":
                    vector_metric = VectorMetric.EUCLIDEAN
                
                # Create collection with vector support using CollectionDefinition
                collection_definition = (
                    CollectionDefinition.builder()
                    .set_vector_dimension(self.config.dimension)
                    .set_vector_metric(vector_metric)
                    .build()
                )
                
                self.collection = self.database.create_collection(
                    self.config.collection_name,
                    definition=collection_definition
                )
                
                logger.info(f"Created collection with {self.config.dimension}d vectors")
            
        except DataAPIException as e:
            logger.error(f"AstraDB API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error setting up collection: {e}")
            raise
    
    def store_chunks(self, chunks: List[Any], batch_size: int = 100) -> int:
        """Store document chunks in AstraDB.
        
        Args:
            chunks: List of document chunks with embeddings
            batch_size: Batch size for insertion
            
        Returns:
            Number of successfully stored chunks
        """
        if not chunks:
            return 0
        
        stored_count = 0
        
        try:
            # Process in batches
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                documents = []
                
                for chunk in batch:
                    # Handle both Document objects and DocumentChunk objects
                    if hasattr(chunk, 'embedding'):
                        embedding = chunk.embedding
                        if hasattr(chunk, 'page_content'):
                            # Langchain Document object
                            content = chunk.page_content
                            metadata = chunk.metadata
                            source_file = metadata.get('file_path', 'unknown')
                            chunk_index = metadata.get('chunk_index', 0)
                        else:
                            # DocumentChunk object
                            content = chunk.content
                            source_file = chunk.source_file
                            chunk_index = chunk.chunk_index
                            metadata = chunk.metadata
                    else:
                        logger.warning(f"Skipping chunk without embedding")
                        continue
                    
                    if not embedding:
                        logger.warning(f"Skipping chunk without embedding: {source_file}:{chunk_index}")
                        continue
                    
                    # Create document for AstraDB
                    doc = {
                        "_id": f"{Path(source_file).stem}_{chunk_index}_{hash(content[:100]) % 10000}",
                        "content": content,
                        "source_file": source_file,
                        "chunk_index": chunk_index,
                        "metadata": metadata,
                        "$vector": embedding
                    }
                    documents.append(doc)
                
                if documents:
                    # Insert batch
                    result = self.collection.insert_many(documents, ordered=False)
                    batch_stored = len(result.inserted_ids) if hasattr(result, 'inserted_ids') else len(documents)
                    stored_count += batch_stored
                    
                    logger.info(f"Stored batch {i//batch_size + 1}: {batch_stored} chunks")
            
            logger.info(f"Successfully stored {stored_count} chunks in AstraDB")
            return stored_count
            
        except Exception as e:
            logger.error(f"Error storing chunks: {e}")
            raise
    
    def store_chunks_with_embeddings(self, chunk_embeddings: List[Dict], batch_size: int = 100) -> int:
        """Store document chunks with embeddings in AstraDB.
        
        Args:
            chunk_embeddings: List of dictionaries with 'chunk' and 'embedding' keys
            batch_size: Batch size for insertion
            
        Returns:
            Number of successfully stored chunks
        """
        if not chunk_embeddings:
            return 0
        
        stored_count = 0
        
        try:
            # Process in batches
            for i in range(0, len(chunk_embeddings), batch_size):
                batch = chunk_embeddings[i:i + batch_size]
                documents = []
                
                for chunk_data in batch:
                    chunk = chunk_data['chunk']
                    embedding = chunk_data['embedding']
                    
                    if not embedding:
                        logger.warning(f"Skipping chunk without embedding")
                        continue
                    
                    # Handle both Document objects and DocumentChunk objects
                    if hasattr(chunk, 'page_content'):
                        # Langchain Document object
                        content = chunk.page_content
                        metadata = chunk.metadata
                        source_file = metadata.get('file_path', 'unknown')
                        chunk_index = metadata.get('chunk_index', 0)
                    else:
                        # DocumentChunk object
                        content = chunk.content
                        source_file = chunk.source_file
                        chunk_index = chunk.chunk_index
                        metadata = chunk.metadata
                    
                    # Create document for AstraDB
                    doc = {
                        "_id": f"{Path(source_file).stem}_{chunk_index}_{hash(content[:100]) % 10000}",
                        "content": content,
                        "source_file": source_file,
                        "chunk_index": chunk_index,
                        "metadata": metadata,
                        "$vector": embedding
                    }
                    documents.append(doc)
                
                if documents:
                    # Insert batch
                    result = self.collection.insert_many(documents, ordered=False)
                    batch_stored = len(result.inserted_ids) if hasattr(result, 'inserted_ids') else len(documents)
                    stored_count += batch_stored
                    
                    logger.info(f"Stored batch {i//batch_size + 1}: {batch_stored} chunks")
            
            logger.info(f"Successfully stored {stored_count} chunks in AstraDB")
            return stored_count
            
        except Exception as e:
            logger.error(f"Error storing chunks: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            # Get basic collection info
            stats = {
                "collection_name": self.config.collection_name,
                "dimension": self.config.dimension,
                "metric": self.config.metric
            }
            
            # Try to get document count (this might not be available in all versions)
            try:
                count_result = self.collection.count_documents({})
                stats["document_count"] = count_result
            except:
                stats["document_count"] = "unavailable"
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}

class EmbeddingPipeline:
    """Main pipeline for creating embeddings using JINA AI and AstraDB with relationship detection."""
    
    def __init__(
        self,
        jina_config: JinaConfig,
        astra_config: AstraConfig
    ):
        self.jina_config = jina_config
        self.astra_config = astra_config
        
        # Initialize components
        self.embedding_provider = JinaEmbeddingProvider(jina_config)
        self.chunker = DocumentChunker()
        self.vector_store = AstraDBVectorStore(astra_config)
    
    def process_directory(self, input_dir: str, file_patterns: List[str] = None) -> Dict[str, Any]:
        """Process all documents in a directory with relationship detection.
        
        Args:
            input_dir: Directory containing documents
            file_patterns: List of file patterns to match (e.g., ['*.txt', '*.md', '*.py'])
            
        Returns:
            Processing results summary
        """
        if file_patterns is None:
            file_patterns = ['*.txt', '*.md', '*.rst', '*.py']
        
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        # Find all matching files
        all_files = []
        for pattern in file_patterns:
            all_files.extend(input_path.glob(f"**/{pattern}"))
        
        logger.info(f"Found {len(all_files)} files to process")
        
        results = {
            "total_files": len(all_files),
            "processed_files": 0,
            "total_chunks": 0,
            "stored_chunks": 0,
            "failed_files": [],
            "processing_time": 0
        }
        
        start_time = datetime.now()
        
        try:
            for file_path in tqdm(all_files, desc="Processing files"):
                try:
                    file_results = self.process_file(str(file_path))
                    results["processed_files"] += 1
                    results["total_chunks"] += file_results["chunks_created"]
                    results["stored_chunks"] += file_results["chunks_stored"]
                    
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    results["failed_files"].append(str(file_path))
            
            results["processing_time"] = (datetime.now() - start_time).total_seconds()
            
            # Get final collection stats
            collection_stats = self.vector_store.get_collection_stats()
            results["collection_stats"] = collection_stats
            
            logger.info(f"Pipeline completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single file with chunking and relationship detection.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Processing results for the file
        """
        logger.info(f"Processing file: {file_path}")
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            raise
        
        # Create metadata
        file_metadata = {
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "file_size": len(content),
            "processing_date": datetime.utcnow().isoformat(),
            "file_type": 'python' if file_path.endswith('.py') else 'markdown'
        }
        
        # Chunk the document and detect relationships
        chunks, relationships = self.chunker.chunk_and_detect(content, file_path, file_metadata)
        
        if not chunks:
            logger.warning(f"No chunks created for {file_path}")
            return {"chunks_created": 0, "chunks_stored": 0}

        # Generate embeddings
        if ADVANCED_CHUNKING_AVAILABLE and chunks and hasattr(chunks[0], 'page_content'):
            # Langchain Document objects
            chunk_texts = [chunk.page_content for chunk in chunks]
        else:
            # DocumentChunk objects
            chunk_texts = [chunk.content for chunk in chunks]
        
        embeddings = self.embedding_provider.embed_texts(chunk_texts)
        
        # Create a mapping of chunks to embeddings instead of modifying the objects
        chunk_embeddings = {}
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if hasattr(chunk, 'page_content'):
                # LangChain Document - store embedding in metadata or separate mapping
                chunk_id = f"chunk_{i}"
                chunk_embeddings[chunk_id] = {
                    'chunk': chunk,
                    'embedding': embedding
                }
            else:
                # DocumentChunk objects can be modified directly
                chunk.embedding = embedding
                chunk_embeddings[f"chunk_{i}"] = {
                    'chunk': chunk,
                    'embedding': embedding
                }

        # Store in vector database using the embedding mapping
        chunks_with_embeddings = [chunk_embeddings[f"chunk_{i}"] for i in range(len(chunks))]
        stored_count = self.vector_store.store_chunks_with_embeddings(chunks_with_embeddings)
        
        logger.info(f"File {file_path}: {len(chunks)} chunks created, {stored_count} stored")
        
        return {
            "chunks_created": len(chunks),
            "chunks_stored": stored_count,
            "relationships": relationships
        }

def load_config_from_env() -> tuple[JinaConfig, AstraConfig, ChunkingConfig]:
    """Load configuration from environment variables."""
    
    # JINA configuration
    jina_api_key = os.getenv('JINA_API_KEY')
    if not jina_api_key:
        raise ValueError("JINA_API_KEY environment variable is required")
    
    jina_config = JinaConfig(
        api_key=jina_api_key,
        model=os.getenv('JINA_MODEL', 'jina-embeddings-v3'),
        dimensions=int(os.getenv('JINA_DIMENSIONS', '1024')),
        batch_size=int(os.getenv('JINA_BATCH_SIZE', '32'))
    )
    
    # AstraDB configuration
    astra_token = os.getenv('ASTRADB_APPLICATION_TOKEN')
    astra_endpoint = os.getenv('ASTRADB_API_ENDPOINT')
    
    logger.info(f"Environment variables - Token exists: {bool(astra_token)}, Endpoint exists: {bool(astra_endpoint)}")
    if astra_endpoint:
        logger.info(f"Endpoint value: {astra_endpoint[:50]}...")
    
    if not astra_token:
        raise ValueError("ASTRADB_APPLICATION_TOKEN environment variable is required")
    if not astra_endpoint:
        raise ValueError("ASTRADB_API_ENDPOINT environment variable is required")
    
    astra_config = AstraConfig(
        application_token=astra_token,
        api_endpoint=astra_endpoint,
        collection_name=os.getenv('ASTRA_COLLECTION_NAME', 'manim_docs'),
        keyspace=os.getenv('ASTRA_KEYSPACE'),
        dimension=int(os.getenv('ASTRA_DIMENSION', '1024')),
        metric=os.getenv('ASTRA_METRIC', 'cosine')
    )
    
    # Chunking configuration
    chunking_config = ChunkingConfig(
        chunk_size=int(os.getenv('CHUNK_SIZE', '1000')),
        overlap=int(os.getenv('CHUNK_OVERLAP', '200')),
        min_chunk_size=int(os.getenv('MIN_CHUNK_SIZE', '100')),
        max_chunk_size=int(os.getenv('MAX_CHUNK_SIZE', '2000'))
    )
    
    return jina_config, astra_config, chunking_config

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Create embeddings using JINA AI and store in AstraDB")
    parser.add_argument("input_dir", help="Directory containing documents to process")
    parser.add_argument("--patterns", nargs="+", default=['*.txt', '*.md', '*.rst', '*.py'], 
                       help="File patterns to match")
    parser.add_argument("--config", help="JSON configuration file (optional)")
    parser.add_argument("--dry-run", action="store_true", help="Process without storing in database")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        if args.config and os.path.exists(args.config):
            logger.info(f"Loading configuration from {args.config}")
            with open(args.config, 'r') as f:
                config_data = json.load(f)
            
            jina_config = JinaConfig(**config_data.get('jina', {}))
            astra_config = AstraConfig(**config_data.get('astra', {}))
            chunking_config = ChunkingConfig(**config_data.get('chunking', {}))
        else:
            logger.info("Loading configuration from environment variables")
            jina_config, astra_config, chunking_config = load_config_from_env()
        
        # Log configuration (without sensitive data)
        logger.info(f"JINA Config: model={jina_config.model}, dimensions={jina_config.dimensions}")
        logger.info(f"Astra Config: collection={astra_config.collection_name}, dimension={astra_config.dimension}")
        logger.info(f"Chunking Config: chunk_size={chunking_config.chunk_size}, overlap={chunking_config.overlap}")
        
        # Create a mock vector store for dry run if needed
        class MockVectorStore:
            def store_chunks(self, chunks, batch_size=100):
                return len(chunks)
            def store_chunks_with_embeddings(self, chunk_embeddings, batch_size=100):
                return len(chunk_embeddings)
            def get_collection_stats(self):
                return {"status": "dry_run"}
        
        # Initialize pipeline
        pipeline = EmbeddingPipeline(jina_config, astra_config)
        
        if args.dry_run:
            logger.info("DRY RUN MODE - No data will be stored")
            # Override the vector store with mock
            pipeline.vector_store = MockVectorStore()
        
        # Process directory
        results = pipeline.process_directory(args.input_dir, args.patterns)
        
        # Print results
        print("\n" + "="*50)
        print("EMBEDDING PIPELINE RESULTS")
        print("="*50)
        print(f"Total files: {results['total_files']}")
        print(f"Processed files: {results['processed_files']}")
        print(f"Total chunks: {results['total_chunks']}")
        print(f"Stored chunks: {results['stored_chunks']}")
        print(f"Processing time: {results['processing_time']:.2f} seconds")
        
        if results['failed_files']:
            print(f"Failed files: {len(results['failed_files'])}")
            for failed_file in results['failed_files']:
                print(f"  - {failed_file}")
        
        if 'collection_stats' in results:
            print(f"Collection stats: {results['collection_stats']}")
        
        logger.info("Embedding pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

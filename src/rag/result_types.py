"""
Result Types for RAG System

This module defines data structures for RAG retrieval results and related types.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ResultType(Enum):
    """Types of retrieval results"""
    API_REFERENCE = "api_reference"
    CODE_EXAMPLE = "code_example"
    TUTORIAL = "tutorial"
    CONCEPT = "concept"
    PLUGIN_DOC = "plugin_doc"


@dataclass
class ChunkData:
    """Data structure for document chunks"""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    source: str
    
    def __post_init__(self):
        """Ensure chunk_id is set"""
        if not self.chunk_id:
            self.chunk_id = f"chunk_{hash(self.content)}"


@dataclass
class ResultMetadata:
    """Metadata for retrieval results"""
    source_file: str
    content_type: str
    semantic_tags: List[str]
    plugin_namespace: Optional[str] = None
    complexity_level: int = 1
    has_examples: bool = False


@dataclass
class SearchResult:
    """Basic search result from vector store"""
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str = ""
    
    def __post_init__(self):
        """Set default source if not provided"""
        if not self.source and 'source' in self.metadata:
            self.source = self.metadata['source']


@dataclass
class RankedResult:
    """Ranked retrieval result with scoring information"""
    chunk_id: str
    chunk: ChunkData
    similarity_score: float
    context_score: float
    final_score: float
    result_type: ResultType
    metadata: ResultMetadata
    rank: int = 0
    
    def __post_init__(self):
        """Ensure chunk_id consistency"""
        if not self.chunk_id and hasattr(self.chunk, 'chunk_id'):
            self.chunk_id = self.chunk.chunk_id
        elif not hasattr(self.chunk, 'chunk_id'):
            self.chunk.chunk_id = self.chunk_id
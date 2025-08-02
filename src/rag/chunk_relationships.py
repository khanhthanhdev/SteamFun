"""
Chunk relationship detection and management system.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Set
from enum import Enum
import re
import ast
from langchain.schema import Document


class RelationshipType(Enum):
    """Enumeration for chunk relationship types."""
    PARENT = "parent"
    CHILD = "child"
    SIBLING = "sibling"
    REFERENCE = "reference"
    DEPENDENCY = "dependency"
    EXAMPLE_OF = "example_of"
    EXPLAINS = "explains"
    CONTINUATION = "continuation"  # For code blocks that span multiple chunks
    PREREQUISITE = "prerequisite"  # For chunks that should be read before this one
    RELATED_CONCEPT = "related_concept"  # For conceptually related chunks


@dataclass
class ChunkRelationship:
    """Represents a relationship between two chunks."""
    related_chunk_id: str
    relationship_type: RelationshipType
    strength: float  # 0.0 to 1.0
    context: Optional[str] = None  # Additional context about the relationship


class ChunkRelationshipDetector:
    """Detects and manages relationships between document chunks."""
    
    def __init__(self):
        self.relationships: Dict[str, List[ChunkRelationship]] = {}
        
    def detect_relationships(self, documents: List[Document]) -> Dict[str, List[ChunkRelationship]]:
        """Detect relationships between all chunks in a document set."""
        self.relationships = {}
        
        # Create chunk index for efficient lookup
        chunk_index = self._create_chunk_index(documents)
        
        for i, doc in enumerate(documents):
            chunk_id = f"chunk_{i}"
            relationships = []
            
            # Detect different types of relationships
            relationships.extend(self._detect_hierarchy_relationships(doc, documents, chunk_index))
            relationships.extend(self._detect_reference_relationships(doc, documents, chunk_index))
            relationships.extend(self._detect_dependency_relationships(doc, documents, chunk_index))
            relationships.extend(self._detect_example_relationships(doc, documents, chunk_index))
            relationships.extend(self._detect_sibling_relationships(doc, documents, chunk_index))
            relationships.extend(self._detect_continuation_relationships(doc, documents, chunk_index))
            relationships.extend(self._detect_prerequisite_relationships(doc, documents, chunk_index))
            relationships.extend(self._detect_concept_relationships(doc, documents, chunk_index))
            
            if relationships:
                self.relationships[chunk_id] = relationships
        
        return self.relationships
    
    def _create_chunk_index(self, documents: List[Document]) -> Dict[str, Dict]:
        """Create an index of chunks for efficient relationship detection."""
        index = {}
        
        for i, doc in enumerate(documents):
            chunk_id = f"chunk_{i}"
            index[chunk_id] = {
                'document': doc,
                'content': doc.page_content.lower(),
                'metadata': doc.metadata,
                'type': doc.metadata.get('type', 'unknown'),
                'name': doc.metadata.get('name', ''),
                'source': doc.metadata.get('source', ''),
                'hierarchy_path': doc.metadata.get('hierarchy_path', []),
                'semantic_tags': doc.metadata.get('semantic_tags', [])
            }
        
        return index
    
    def _detect_hierarchy_relationships(self, doc: Document, all_docs: List[Document], 
                                     chunk_index: Dict[str, Dict]) -> List[ChunkRelationship]:
        """Detect parent/child relationships based on document hierarchy."""
        relationships = []
        doc_metadata = doc.metadata
        
        # For markdown sections, detect parent/child based on header levels
        if doc_metadata.get('type') == 'markdown_section':
            current_level = doc_metadata.get('level', 0)
            hierarchy_path = doc_metadata.get('hierarchy_path', [])
            
            for chunk_id, chunk_info in chunk_index.items():
                if chunk_info['type'] == 'markdown_section':
                    other_level = chunk_info['metadata'].get('level', 0)
                    other_hierarchy = chunk_info['metadata'].get('hierarchy_path', [])
                    
                    # Parent relationship (other chunk is parent)
                    if (other_level < current_level and 
                        len(other_hierarchy) == len(hierarchy_path) - 1 and
                        hierarchy_path[:-1] == other_hierarchy):
                        relationships.append(ChunkRelationship(
                            related_chunk_id=chunk_id,
                            relationship_type=RelationshipType.PARENT,
                            strength=0.9,
                            context=f"Header hierarchy: {' > '.join(other_hierarchy)}"
                        ))
                    
                    # Child relationship (other chunk is child)
                    elif (other_level > current_level and 
                          len(other_hierarchy) == len(hierarchy_path) + 1 and
                          other_hierarchy[:-1] == hierarchy_path):
                        relationships.append(ChunkRelationship(
                            related_chunk_id=chunk_id,
                            relationship_type=RelationshipType.CHILD,
                            strength=0.9,
                            context=f"Header hierarchy: {' > '.join(other_hierarchy)}"
                        ))
        
        # For code chunks, detect class/method relationships
        elif doc_metadata.get('type') in ['class', 'function']:
            source_file = doc_metadata.get('source', '')
            
            for chunk_id, chunk_info in chunk_index.items():
                if (chunk_info['metadata'].get('source') == source_file and
                    chunk_info['type'] in ['class', 'function']):
                    
                    # Method belongs to class
                    if (doc_metadata.get('type') == 'class' and 
                        chunk_info['type'] == 'function'):
                        # Check if function is within class boundaries
                        class_start = doc_metadata.get('start_line', 0)
                        class_end = doc_metadata.get('end_line', 0)
                        func_start = chunk_info['metadata'].get('start_line', 0)
                        
                        if class_start < func_start < class_end:
                            relationships.append(ChunkRelationship(
                                related_chunk_id=chunk_id,
                                relationship_type=RelationshipType.CHILD,
                                strength=0.95,
                                context=f"Method {chunk_info['name']} in class {doc_metadata.get('name')}"
                            ))
        
        return relationships
    
    def _detect_reference_relationships(self, doc: Document, all_docs: List[Document], 
                                      chunk_index: Dict[str, Dict]) -> List[ChunkRelationship]:
        """Detect reference relationships based on content cross-references."""
        relationships = []
        content = doc.page_content.lower()
        
        for chunk_id, chunk_info in chunk_index.items():
            if chunk_id == f"chunk_{all_docs.index(doc)}":  # Skip self
                continue
            
            other_name = chunk_info['name']
            if not other_name:
                continue
            
            # Check for explicit references to other chunks
            reference_patterns = [
                rf'\b{re.escape(other_name.lower())}\b',
                rf'see\s+{re.escape(other_name.lower())}',
                rf'refer\s+to\s+{re.escape(other_name.lower())}',
                rf'{re.escape(other_name.lower())}\s+class',
                rf'{re.escape(other_name.lower())}\s+function'
            ]
            
            for pattern in reference_patterns:
                if re.search(pattern, content):
                    strength = 0.7 if 'see' in pattern or 'refer' in pattern else 0.5
                    relationships.append(ChunkRelationship(
                        related_chunk_id=chunk_id,
                        relationship_type=RelationshipType.REFERENCE,
                        strength=strength,
                        context=f"References {other_name}"
                    ))
                    break
        
        return relationships
    
    def _detect_dependency_relationships(self, doc: Document, all_docs: List[Document], 
                                       chunk_index: Dict[str, Dict]) -> List[ChunkRelationship]:
        """Detect dependency relationships based on imports and usage."""
        relationships = []
        content = doc.page_content
        
        # Extract imports and dependencies from code
        if doc.metadata.get('type') in ['class', 'function', 'code_block']:
            dependencies = self._extract_code_dependencies(content)
            
            for chunk_id, chunk_info in chunk_index.items():
                if chunk_id == f"chunk_{all_docs.index(doc)}":  # Skip self
                    continue
                
                other_name = chunk_info['name']
                if other_name in dependencies:
                    relationships.append(ChunkRelationship(
                        related_chunk_id=chunk_id,
                        relationship_type=RelationshipType.DEPENDENCY,
                        strength=0.8,
                        context=f"Depends on {other_name}"
                    ))
        
        return relationships
    
    def _detect_example_relationships(self, doc: Document, all_docs: List[Document], 
                                    chunk_index: Dict[str, Dict]) -> List[ChunkRelationship]:
        """Detect example relationships between code blocks and explanatory text."""
        relationships = []
        
        # Code blocks that explain concepts
        if doc.metadata.get('type') == 'code_block':
            associated_section = doc.metadata.get('associated_section', '')
            
            for chunk_id, chunk_info in chunk_index.items():
                if (chunk_info['type'] == 'markdown_section' and 
                    chunk_info['metadata'].get('header') == associated_section):
                    relationships.append(ChunkRelationship(
                        related_chunk_id=chunk_id,
                        relationship_type=RelationshipType.EXAMPLE_OF,
                        strength=0.9,
                        context=f"Code example for {associated_section}"
                    ))
        
        # Markdown sections that explain code
        elif doc.metadata.get('type') == 'markdown_section':
            if doc.metadata.get('has_code_blocks', False):
                for chunk_id, chunk_info in chunk_index.items():
                    if (chunk_info['type'] == 'code_block' and 
                        chunk_info['metadata'].get('associated_section') == doc.metadata.get('header')):
                        relationships.append(ChunkRelationship(
                            related_chunk_id=chunk_id,
                            relationship_type=RelationshipType.EXPLAINS,
                            strength=0.9,
                            context=f"Explains code example"
                        ))
        
        return relationships
    
    def _detect_sibling_relationships(self, doc: Document, all_docs: List[Document], 
                                    chunk_index: Dict[str, Dict]) -> List[ChunkRelationship]:
        """Detect sibling relationships between chunks at the same level."""
        relationships = []
        doc_metadata = doc.metadata
        
        # Markdown sections at the same level
        if doc_metadata.get('type') == 'markdown_section':
            current_level = doc_metadata.get('level', 0)
            parent_headers = doc_metadata.get('parent_headers', [])
            
            for chunk_id, chunk_info in chunk_index.items():
                if (chunk_info['type'] == 'markdown_section' and
                    chunk_info['metadata'].get('level') == current_level and
                    chunk_info['metadata'].get('parent_headers') == parent_headers):
                    
                    # Calculate semantic similarity for sibling strength
                    semantic_similarity = self._calculate_semantic_similarity(
                        doc_metadata.get('semantic_tags', []),
                        chunk_info['metadata'].get('semantic_tags', [])
                    )
                    
                    if semantic_similarity > 0.3:
                        relationships.append(ChunkRelationship(
                            related_chunk_id=chunk_id,
                            relationship_type=RelationshipType.SIBLING,
                            strength=semantic_similarity,
                            context=f"Sibling section at level {current_level}"
                        ))
        
        # Methods in the same class
        elif doc_metadata.get('type') == 'function':
            source_file = doc_metadata.get('source', '')
            
            for chunk_id, chunk_info in chunk_index.items():
                if (chunk_info['type'] == 'function' and
                    chunk_info['metadata'].get('source') == source_file):
                    
                    # Check if they belong to the same class
                    if self._are_methods_in_same_class(doc, chunk_info['document'], all_docs):
                        relationships.append(ChunkRelationship(
                            related_chunk_id=chunk_id,
                            relationship_type=RelationshipType.SIBLING,
                            strength=0.6,
                            context="Methods in the same class"
                        ))
        
        return relationships
    
    def _detect_continuation_relationships(self, doc: Document, all_docs: List[Document], 
                                         chunk_index: Dict[str, Dict]) -> List[ChunkRelationship]:
        """Detect continuation relationships for code blocks that span multiple chunks."""
        relationships = []
        doc_metadata = doc.metadata
        
        # Look for chunks from the same source file with adjacent line numbers
        if doc_metadata.get('type') in ['class', 'function', 'code_block']:
            source_file = doc_metadata.get('source', '')
            current_end_line = doc_metadata.get('end_line', 0)
            current_start_line = doc_metadata.get('start_line', 0)
            
            for chunk_id, chunk_info in chunk_index.items():
                if chunk_id == f"chunk_{all_docs.index(doc)}":  # Skip self
                    continue
                
                if (chunk_info['metadata'].get('source') == source_file and
                    chunk_info['type'] in ['class', 'function', 'code_block']):
                    
                    other_start_line = chunk_info['metadata'].get('start_line', 0)
                    other_end_line = chunk_info['metadata'].get('end_line', 0)
                    
                    # Check for continuation (adjacent or overlapping line ranges)
                    line_gap = abs(other_start_line - current_end_line)
                    reverse_gap = abs(current_start_line - other_end_line)
                    
                    # Consider chunks as continuations if they're within 5 lines of each other
                    if line_gap <= 5 or reverse_gap <= 5:
                        # Determine strength based on proximity
                        min_gap = min(line_gap, reverse_gap)
                        strength = max(0.3, 1.0 - (min_gap / 10.0))  # Closer = stronger
                        
                        relationships.append(ChunkRelationship(
                            related_chunk_id=chunk_id,
                            relationship_type=RelationshipType.CONTINUATION,
                            strength=strength,
                            context=f"Adjacent code blocks (gap: {min_gap} lines)"
                        ))
        
        # Also check for logical continuation in markdown sections
        elif doc_metadata.get('type') == 'markdown_section':
            hierarchy_path = doc_metadata.get('hierarchy_path', [])
            current_level = doc_metadata.get('level', 0)
            
            for chunk_id, chunk_info in chunk_index.items():
                if (chunk_info['type'] == 'markdown_section' and
                    chunk_info['metadata'].get('level') == current_level):
                    
                    other_hierarchy = chunk_info['metadata'].get('hierarchy_path', [])
                    
                    # Check if sections are logically sequential
                    if (len(hierarchy_path) == len(other_hierarchy) and
                        hierarchy_path[:-1] == other_hierarchy[:-1]):  # Same parent path
                        
                        # Check for sequential naming patterns
                        current_header = hierarchy_path[-1] if hierarchy_path else ""
                        other_header = other_hierarchy[-1] if other_hierarchy else ""
                        
                        if self._are_sequential_sections(current_header, other_header):
                            relationships.append(ChunkRelationship(
                                related_chunk_id=chunk_id,
                                relationship_type=RelationshipType.CONTINUATION,
                                strength=0.7,
                                context=f"Sequential sections: {current_header} -> {other_header}"
                            ))
        
        return relationships
    
    def _detect_prerequisite_relationships(self, doc: Document, all_docs: List[Document], 
                                         chunk_index: Dict[str, Dict]) -> List[ChunkRelationship]:
        """Detect prerequisite relationships for chunks that should be read before this one."""
        relationships = []
        content = doc.page_content.lower()
        doc_metadata = doc.metadata
        
        # Look for explicit prerequisite indicators
        prerequisite_patterns = [
            r'before\s+(?:reading|using|implementing)\s+(\w+)',
            r'first\s+(?:read|understand|implement)\s+(\w+)',
            r'requires?\s+(?:understanding\s+of\s+)?(\w+)',
            r'depends?\s+on\s+(\w+)',
            r'see\s+(\w+)\s+first',
            r'after\s+(?:reading|understanding)\s+(\w+)'
        ]
        
        for pattern in prerequisite_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                # Find chunks that match the prerequisite
                for chunk_id, chunk_info in chunk_index.items():
                    if chunk_id == f"chunk_{all_docs.index(doc)}":  # Skip self
                        continue
                    
                    chunk_name = chunk_info['name'].lower()
                    if chunk_name and match.lower() in chunk_name:
                        relationships.append(ChunkRelationship(
                            related_chunk_id=chunk_id,
                            relationship_type=RelationshipType.PREREQUISITE,
                            strength=0.8,
                            context=f"Prerequisite: {match}"
                        ))
        
        # For code chunks, detect import-based prerequisites
        if doc_metadata.get('type') in ['class', 'function', 'code_block']:
            dependencies = self._extract_code_dependencies(doc.page_content)
            
            for chunk_id, chunk_info in chunk_index.items():
                if chunk_id == f"chunk_{all_docs.index(doc)}":  # Skip self
                    continue
                
                # If this chunk imports/uses something defined in another chunk
                other_name = chunk_info['name']
                if other_name in dependencies:
                    # Check if the other chunk is a definition (class/function)
                    if chunk_info['type'] in ['class', 'function']:
                        relationships.append(ChunkRelationship(
                            related_chunk_id=chunk_id,
                            relationship_type=RelationshipType.PREREQUISITE,
                            strength=0.9,
                            context=f"Imports/uses {other_name}"
                        ))
        
        return relationships
    
    def _detect_concept_relationships(self, doc: Document, all_docs: List[Document], 
                                    chunk_index: Dict[str, Dict]) -> List[ChunkRelationship]:
        """Detect conceptually related chunks based on semantic similarity."""
        relationships = []
        doc_metadata = doc.metadata
        doc_tags = set(doc_metadata.get('semantic_tags', []))
        
        if not doc_tags:
            return relationships
        
        for chunk_id, chunk_info in chunk_index.items():
            if chunk_id == f"chunk_{all_docs.index(doc)}":  # Skip self
                continue
            
            other_tags = set(chunk_info['metadata'].get('semantic_tags', []))
            if not other_tags:
                continue
            
            # Calculate semantic similarity
            similarity = self._calculate_semantic_similarity(list(doc_tags), list(other_tags))
            
            # Only create relationships for moderately similar chunks
            # (avoid too weak or too strong relationships)
            if 0.2 <= similarity <= 0.8:
                relationships.append(ChunkRelationship(
                    related_chunk_id=chunk_id,
                    relationship_type=RelationshipType.RELATED_CONCEPT,
                    strength=similarity,
                    context=f"Shared concepts: {', '.join(doc_tags.intersection(other_tags))}"
                ))
        
        return relationships
    
    def _are_sequential_sections(self, header1: str, header2: str) -> bool:
        """Check if two section headers are sequential (e.g., Step 1, Step 2)."""
        # Look for numbered patterns
        number_patterns = [
            r'(?:step|part|section|chapter)\s*(\d+)',
            r'(\d+)\.?\s*(?:step|part|section|chapter)?',
            r'(\d+)\.\s*'
        ]
        
        for pattern in number_patterns:
            match1 = re.search(pattern, header1.lower())
            match2 = re.search(pattern, header2.lower())
            
            if match1 and match2:
                try:
                    num1 = int(match1.group(1))
                    num2 = int(match2.group(1))
                    # Sequential if numbers are consecutive
                    return abs(num1 - num2) == 1
                except ValueError:
                    continue
        
        # Look for alphabetical patterns (A, B, C)
        alpha_pattern = r'^([a-z])\.\s*'
        match1 = re.search(alpha_pattern, header1.lower())
        match2 = re.search(alpha_pattern, header2.lower())
        
        if match1 and match2:
            char1 = ord(match1.group(1))
            char2 = ord(match2.group(1))
            return abs(char1 - char2) == 1
        
        return False
    
    def _extract_code_dependencies(self, content: str) -> Set[str]:
        """Extract dependencies from code content."""
        dependencies = set()
        
        # Extract from import statements
        import_patterns = [
            r'from\s+[\w.]+\s+import\s+([\w,\s]+)',
            r'import\s+([\w.]+)',
            r'from\s+([\w.]+)\s+import'
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, str):
                    # Split comma-separated imports
                    deps = [dep.strip() for dep in match.split(',')]
                    dependencies.update(deps)
        
        # Extract class and function names used in the code
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    dependencies.add(node.id)
                elif isinstance(node, ast.Attribute):
                    dependencies.add(node.attr)
        except:
            pass  # Ignore parsing errors
        
        return dependencies
    
    def _calculate_semantic_similarity(self, tags1: List[str], tags2: List[str]) -> float:
        """Calculate semantic similarity between two sets of tags."""
        if not tags1 or not tags2:
            return 0.0
        
        set1 = set(tags1)
        set2 = set(tags2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _are_methods_in_same_class(self, doc1: Document, doc2: Document, all_docs: List[Document]) -> bool:
        """Check if two function documents are methods in the same class."""
        source1 = doc1.metadata.get('source', '')
        source2 = doc2.metadata.get('source', '')
        
        if source1 != source2:
            return False
        
        # Find class that contains both methods
        for doc in all_docs:
            if (doc.metadata.get('type') == 'class' and 
                doc.metadata.get('source') == source1):
                
                class_start = doc.metadata.get('start_line', 0)
                class_end = doc.metadata.get('end_line', 0)
                
                func1_start = doc1.metadata.get('start_line', 0)
                func2_start = doc2.metadata.get('start_line', 0)
                
                if (class_start < func1_start < class_end and 
                    class_start < func2_start < class_end):
                    return True
        
        return False
    
    def get_relationships(self, chunk_id: str) -> List[ChunkRelationship]:
        """Get all relationships for a specific chunk."""
        return self.relationships.get(chunk_id, [])
    
    def get_related_chunks(self, chunk_id: str, relationship_type: RelationshipType = None) -> List[str]:
        """Get IDs of chunks related to the given chunk."""
        relationships = self.get_relationships(chunk_id)
        
        if relationship_type:
            relationships = [r for r in relationships if r.relationship_type == relationship_type]
        
        return [r.related_chunk_id for r in relationships]
    
    def get_relationship_strength(self, chunk_id1: str, chunk_id2: str) -> float:
        """Get the strength of relationship between two chunks."""
        relationships = self.get_relationships(chunk_id1)
        
        for rel in relationships:
            if rel.related_chunk_id == chunk_id2:
                return rel.strength
        
        return 0.0
    
    def get_bidirectional_relationships(self, chunk_id: str) -> Dict[str, List[ChunkRelationship]]:
        """Get both outgoing and incoming relationships for a chunk."""
        outgoing = self.get_relationships(chunk_id)
        incoming = []
        
        # Find incoming relationships
        for other_chunk_id, other_relationships in self.relationships.items():
            if other_chunk_id != chunk_id:
                for rel in other_relationships:
                    if rel.related_chunk_id == chunk_id:
                        incoming.append(ChunkRelationship(
                            related_chunk_id=other_chunk_id,
                            relationship_type=self._get_inverse_relationship(rel.relationship_type),
                            strength=rel.strength,
                            context=rel.context
                        ))
        
        return {
            'outgoing': outgoing,
            'incoming': incoming
        }
    
    def _get_inverse_relationship(self, relationship_type: RelationshipType) -> RelationshipType:
        """Get the inverse of a relationship type."""
        inverse_map = {
            RelationshipType.PARENT: RelationshipType.CHILD,
            RelationshipType.CHILD: RelationshipType.PARENT,
            RelationshipType.SIBLING: RelationshipType.SIBLING,
            RelationshipType.REFERENCE: RelationshipType.REFERENCE,
            RelationshipType.DEPENDENCY: RelationshipType.PREREQUISITE,
            RelationshipType.PREREQUISITE: RelationshipType.DEPENDENCY,
            RelationshipType.EXAMPLE_OF: RelationshipType.EXPLAINS,
            RelationshipType.EXPLAINS: RelationshipType.EXAMPLE_OF,
            RelationshipType.CONTINUATION: RelationshipType.CONTINUATION,
            RelationshipType.RELATED_CONCEPT: RelationshipType.RELATED_CONCEPT
        }
        return inverse_map.get(relationship_type, relationship_type)
    
    def get_relationship_graph(self) -> Dict[str, Dict[str, float]]:
        """Get a graph representation of all relationships with strengths."""
        graph = {}
        
        for chunk_id, relationships in self.relationships.items():
            if chunk_id not in graph:
                graph[chunk_id] = {}
            
            for rel in relationships:
                graph[chunk_id][rel.related_chunk_id] = rel.strength
        
        return graph
    
    def find_related_chunks_by_path(self, start_chunk_id: str, max_depth: int = 2) -> Dict[str, List[str]]:
        """Find chunks related through relationship paths up to max_depth."""
        visited = set()
        paths = {start_chunk_id: []}
        queue = [(start_chunk_id, 0)]
        
        while queue:
            current_chunk, depth = queue.pop(0)
            
            if depth >= max_depth or current_chunk in visited:
                continue
            
            visited.add(current_chunk)
            
            # Get all related chunks
            relationships = self.get_relationships(current_chunk)
            for rel in relationships:
                related_id = rel.related_chunk_id
                if related_id not in visited:
                    new_path = paths[current_chunk] + [f"{rel.relationship_type.value}({rel.strength:.2f})"]
                    paths[related_id] = new_path
                    queue.append((related_id, depth + 1))
        
        # Remove the starting chunk from results
        del paths[start_chunk_id]
        return paths
    
    def get_strongest_relationships(self, chunk_id: str, top_k: int = 5) -> List[ChunkRelationship]:
        """Get the top-k strongest relationships for a chunk."""
        relationships = self.get_relationships(chunk_id)
        return sorted(relationships, key=lambda r: r.strength, reverse=True)[:top_k]
    
    def get_relationships_by_type(self, relationship_type: RelationshipType) -> Dict[str, List[ChunkRelationship]]:
        """Get all relationships of a specific type."""
        filtered_relationships = {}
        
        for chunk_id, relationships in self.relationships.items():
            type_relationships = [r for r in relationships if r.relationship_type == relationship_type]
            if type_relationships:
                filtered_relationships[chunk_id] = type_relationships
        
        return filtered_relationships
    
    def export_relationships_summary(self) -> Dict[str, any]:
        """Export a summary of all relationships for analysis."""
        summary = {
            'total_chunks_with_relationships': len(self.relationships),
            'total_relationships': sum(len(rels) for rels in self.relationships.values()),
            'relationship_type_counts': {},
            'average_relationships_per_chunk': 0,
            'strongest_relationships': [],
            'relationship_strength_distribution': {'weak': 0, 'medium': 0, 'strong': 0}
        }
        
        all_relationships = []
        for chunk_id, relationships in self.relationships.items():
            all_relationships.extend(relationships)
        
        # Count by type
        for rel in all_relationships:
            rel_type = rel.relationship_type.value
            summary['relationship_type_counts'][rel_type] = summary['relationship_type_counts'].get(rel_type, 0) + 1
        
        # Calculate averages
        if self.relationships:
            summary['average_relationships_per_chunk'] = len(all_relationships) / len(self.relationships)
        
        # Get strongest relationships
        summary['strongest_relationships'] = [
            {
                'chunk_id': chunk_id,
                'related_chunk_id': rel.related_chunk_id,
                'type': rel.relationship_type.value,
                'strength': rel.strength,
                'context': rel.context
            }
            for chunk_id, relationships in self.relationships.items()
            for rel in sorted(relationships, key=lambda r: r.strength, reverse=True)[:3]
        ][:10]  # Top 10 overall
        
        # Strength distribution
        for rel in all_relationships:
            if rel.strength < 0.4:
                summary['relationship_strength_distribution']['weak'] += 1
            elif rel.strength < 0.7:
                summary['relationship_strength_distribution']['medium'] += 1
            else:
                summary['relationship_strength_distribution']['strong'] += 1
        
        return summary
"""Semantic Understanding and Relationship Engine for Manim RAG System.

This module implements the semantic understanding capabilities including:
- ManimKnowledgeGraph for concept relationships
- SemanticRelationshipEngine for query expansion
- Dependency tracking and inheritance chain analysis
"""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union
from pathlib import Path

import networkx as nx


class ConceptType(Enum):
    """Types of Manim concepts in the knowledge graph."""
    MOBJECT = "mobject"
    ANIMATION = "animation"
    SCENE = "scene"
    CAMERA = "camera"
    RENDERER = "renderer"
    UTILITY = "utility"
    PLUGIN = "plugin"
    CONFIG = "config"


class RelationshipType(Enum):
    """Types of relationships between concepts."""
    INHERITS_FROM = "inherits_from"
    CONTAINS = "contains"
    USES = "uses"
    TRANSFORMS_TO = "transforms_to"
    SIMILAR_TO = "similar_to"
    DEPENDS_ON = "depends_on"
    PLUGIN_EXTENDS = "plugin_extends"
    EXAMPLE_OF = "example_of"


@dataclass
class ManimConcept:
    """Represents a concept in the Manim knowledge graph."""
    name: str
    concept_type: ConceptType
    description: str = ""
    module_path: str = ""
    class_hierarchy: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    parameters: List[str] = field(default_factory=list)
    usage_patterns: List[str] = field(default_factory=list)
    plugin_namespace: Optional[str] = None
    complexity_level: int = 1
    common_use_cases: List[str] = field(default_factory=list)
    related_examples: List[str] = field(default_factory=list)


@dataclass
class ConceptRelationship:
    """Represents a relationship between two concepts."""
    source_concept: str
    target_concept: str
    relationship_type: RelationshipType
    strength: float = 1.0
    context: str = ""
    bidirectional: bool = False


@dataclass
class RelatedConcept:
    """Represents a concept related to a query with relevance scoring."""
    concept: ManimConcept
    relevance_score: float
    relationship_path: List[str]
    relationship_types: List[RelationshipType]


class ManimKnowledgeGraph:
    """Knowledge graph for Manim concepts and their relationships.
    
    This class builds and maintains a comprehensive understanding of Manim's
    class hierarchy, concept relationships, and plugin namespaces.
    """
    
    def __init__(self, manim_docs_path: str):
        """Initialize the knowledge graph.
        
        Args:
            manim_docs_path: Path to the Manim documentation directory
        """
        self.manim_docs_path = Path(manim_docs_path)
        self.graph = nx.DiGraph()
        self.concepts: Dict[str, ManimConcept] = {}
        self.relationships: List[ConceptRelationship] = []
        self.plugin_namespaces: Dict[str, Dict] = {}
        self.class_hierarchy: Dict[str, List[str]] = {}
        self.concept_categories: Dict[ConceptType, Set[str]] = {
            concept_type: set() for concept_type in ConceptType
        }
        
        # Build the knowledge graph
        self._build_knowledge_graph()
    
    def _build_knowledge_graph(self) -> None:
        """Build the complete knowledge graph from Manim documentation."""
        print("Building Manim Knowledge Graph...")
        
        # Step 1: Build class hierarchy from core Manim
        self._build_core_class_hierarchy()
        
        # Step 2: Build concept relationships
        self._build_concept_relationships()
        
        # Step 3: Load plugin information
        self._build_plugin_namespaces()
        
        # Step 4: Create the NetworkX graph
        self._create_networkx_graph()
        
        print(f"Knowledge graph built with {len(self.concepts)} concepts and {len(self.relationships)} relationships")
    
    def _build_core_class_hierarchy(self) -> None:
        """Build class hierarchy mapping from Manim core source code."""
        core_source_path = self.manim_docs_path / "manim_core" / "source"
        
        if not core_source_path.exists():
            print(f"Warning: Core source path not found: {core_source_path}")
            return
        
        # Key directories to analyze
        key_directories = [
            "mobject",
            "animation", 
            "scene",
            "camera",
            "renderer",
            "utils"
        ]
        
        for directory in key_directories:
            dir_path = core_source_path / directory
            if dir_path.exists():
                self._analyze_directory_hierarchy(dir_path, directory)
    
    def _analyze_directory_hierarchy(self, directory_path: Path, category: str) -> None:
        """Analyze a directory to extract class hierarchy and concepts."""
        concept_type = self._get_concept_type_from_category(category)
        
        for python_file in directory_path.rglob("*.py"):
            if python_file.name.startswith("__"):
                continue
                
            try:
                with open(python_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse the Python file
                tree = ast.parse(content)
                module_path = str(python_file.relative_to(self.manim_docs_path))
                
                # Extract classes and their information
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        self._process_class_node(node, concept_type, module_path, content)
                        
            except Exception as e:
                print(f"Error processing {python_file}: {e}")
                continue
    
    def _process_class_node(self, node: ast.ClassDef, concept_type: ConceptType, 
                          module_path: str, content: str) -> None:
        """Process a class node to extract concept information."""
        class_name = node.name
        
        # Extract base classes (inheritance)
        base_classes = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_classes.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_classes.append(f"{base.value.id}.{base.attr}" if hasattr(base.value, 'id') else base.attr)
        
        # Extract methods
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(item.name)
        
        # Extract docstring
        docstring = ast.get_docstring(node) or ""
        
        # Create concept
        concept = ManimConcept(
            name=class_name,
            concept_type=concept_type,
            description=docstring[:200] + "..." if len(docstring) > 200 else docstring,
            module_path=module_path,
            class_hierarchy=base_classes,
            methods=methods,
            complexity_level=self._calculate_class_complexity(node),
            usage_patterns=self._extract_usage_patterns(content, class_name),
            common_use_cases=self._infer_use_cases(class_name, concept_type, docstring)
        )
        
        self.concepts[class_name] = concept
        self.concept_categories[concept_type].add(class_name)
        self.class_hierarchy[class_name] = base_classes
        
        # Create inheritance relationships
        for base_class in base_classes:
            relationship = ConceptRelationship(
                source_concept=class_name,
                target_concept=base_class,
                relationship_type=RelationshipType.INHERITS_FROM,
                strength=0.9,
                context=f"{class_name} inherits from {base_class}"
            )
            self.relationships.append(relationship)
    
    def _build_concept_relationships(self) -> None:
        """Build relationships between Manim concepts."""
        # Define common relationship patterns
        relationship_patterns = [
            # Mobject relationships
            (r".*Mobject", r".*Animation", RelationshipType.USES, 0.8),
            (r".*Scene", r".*Mobject", RelationshipType.CONTAINS, 0.9),
            (r".*Scene", r".*Animation", RelationshipType.USES, 0.8),
            
            # Animation relationships  
            (r".*Transform.*", r".*Mobject", RelationshipType.TRANSFORMS_TO, 0.9),
            (r".*Creation.*", r".*Mobject", RelationshipType.USES, 0.8),
            (r".*Fading.*", r".*Mobject", RelationshipType.USES, 0.8),
            
            # Camera and renderer relationships
            (r".*Scene", r".*Camera", RelationshipType.USES, 0.7),
            (r".*Camera", r".*Renderer", RelationshipType.USES, 0.6),
        ]
        
        # Apply pattern-based relationships
        for source_pattern, target_pattern, rel_type, strength in relationship_patterns:
            self._create_pattern_relationships(source_pattern, target_pattern, rel_type, strength)
        
        # Create similarity relationships based on naming patterns
        self._create_similarity_relationships()
        
        # Create dependency relationships
        self._create_dependency_relationships()
    
    def _create_pattern_relationships(self, source_pattern: str, target_pattern: str, 
                                    rel_type: RelationshipType, strength: float) -> None:
        """Create relationships based on naming patterns."""
        source_concepts = [name for name in self.concepts.keys() if re.match(source_pattern, name)]
        target_concepts = [name for name in self.concepts.keys() if re.match(target_pattern, name)]
        
        for source in source_concepts:
            for target in target_concepts:
                if source != target:
                    relationship = ConceptRelationship(
                        source_concept=source,
                        target_concept=target,
                        relationship_type=rel_type,
                        strength=strength,
                        context=f"Pattern-based relationship: {source} {rel_type.value} {target}"
                    )
                    self.relationships.append(relationship)
    
    def _create_similarity_relationships(self) -> None:
        """Create similarity relationships between concepts with similar names or purposes."""
        concept_names = list(self.concepts.keys())
        
        for i, concept1 in enumerate(concept_names):
            for concept2 in concept_names[i+1:]:
                similarity_score = self._calculate_name_similarity(concept1, concept2)
                
                if similarity_score > 0.6:  # Threshold for similarity
                    relationship = ConceptRelationship(
                        source_concept=concept1,
                        target_concept=concept2,
                        relationship_type=RelationshipType.SIMILAR_TO,
                        strength=similarity_score,
                        context=f"Similar naming pattern: {concept1} ~ {concept2}",
                        bidirectional=True
                    )
                    self.relationships.append(relationship)
    
    def _create_dependency_relationships(self) -> None:
        """Create dependency relationships based on import patterns and usage."""
        # This would analyze import statements and method calls
        # For now, we'll create some basic dependency relationships
        
        dependency_patterns = [
            ("Scene", "Mobject", 0.9),
            ("Scene", "Animation", 0.9),
            ("Animation", "Mobject", 0.8),
            ("Camera", "Scene", 0.7),
            ("Renderer", "Camera", 0.6),
        ]
        
        for source, target, strength in dependency_patterns:
            if source in self.concepts and target in self.concepts:
                relationship = ConceptRelationship(
                    source_concept=source,
                    target_concept=target,
                    relationship_type=RelationshipType.DEPENDS_ON,
                    strength=strength,
                    context=f"{source} depends on {target} for functionality"
                )
                self.relationships.append(relationship)
    
    def _build_plugin_namespaces(self) -> None:
        """Build plugin namespace understanding from plugin documentation."""
        plugins_json_path = self.manim_docs_path / "plugin_docs" / "plugins.json"
        
        if not plugins_json_path.exists():
            print(f"Warning: Plugin configuration not found: {plugins_json_path}")
            return
        
        try:
            with open(plugins_json_path, 'r', encoding='utf-8') as f:
                plugins_data = json.load(f)
            
            for plugin_info in plugins_data:
                plugin_name = plugin_info["name"]
                plugin_description = plugin_info["description"]
                
                # Store plugin namespace information
                self.plugin_namespaces[plugin_name] = {
                    "description": plugin_description,
                    "concepts": [],
                    "namespace": plugin_name.replace("-", "_")
                }
                
                # Create plugin concept
                plugin_concept = ManimConcept(
                    name=plugin_name,
                    concept_type=ConceptType.PLUGIN,
                    description=plugin_description,
                    plugin_namespace=plugin_name,
                    complexity_level=2,
                    common_use_cases=self._extract_plugin_use_cases(plugin_description)
                )
                
                self.concepts[plugin_name] = plugin_concept
                self.concept_categories[ConceptType.PLUGIN].add(plugin_name)
                
                # Analyze plugin source code if available
                plugin_source_path = self.manim_docs_path / "plugin_docs" / plugin_name / "source"
                if plugin_source_path.exists():
                    self._analyze_plugin_source(plugin_name, plugin_source_path)
                    
        except Exception as e:
            print(f"Error loading plugin information: {e}")
    
    def _analyze_plugin_source(self, plugin_name: str, source_path: Path) -> None:
        """Analyze plugin source code to extract concepts."""
        for python_file in source_path.rglob("*.py"):
            if python_file.name.startswith("__"):
                continue
                
            try:
                with open(python_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                module_path = str(python_file.relative_to(self.manim_docs_path))
                
                # Extract plugin-specific classes
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_name = f"{plugin_name}.{node.name}"
                        
                        concept = ManimConcept(
                            name=class_name,
                            concept_type=ConceptType.PLUGIN,
                            description=ast.get_docstring(node) or "",
                            module_path=module_path,
                            plugin_namespace=plugin_name,
                            methods=[item.name for item in node.body if isinstance(item, ast.FunctionDef)],
                            complexity_level=2
                        )
                        
                        self.concepts[class_name] = concept
                        self.plugin_namespaces[plugin_name]["concepts"].append(class_name)
                        
                        # Create plugin extension relationship
                        relationship = ConceptRelationship(
                            source_concept=class_name,
                            target_concept=plugin_name,
                            relationship_type=RelationshipType.PLUGIN_EXTENDS,
                            strength=1.0,
                            context=f"{class_name} extends {plugin_name} plugin"
                        )
                        self.relationships.append(relationship)
                        
            except Exception as e:
                print(f"Error processing plugin file {python_file}: {e}")
                continue
    
    def _create_networkx_graph(self) -> None:
        """Create NetworkX graph from concepts and relationships."""
        # Add nodes (concepts)
        for concept_name, concept in self.concepts.items():
            self.graph.add_node(
                concept_name,
                concept_type=concept.concept_type.value,
                description=concept.description,
                complexity_level=concept.complexity_level,
                plugin_namespace=concept.plugin_namespace
            )
        
        # Add edges (relationships)
        for relationship in self.relationships:
            if relationship.source_concept in self.concepts and relationship.target_concept in self.concepts:
                self.graph.add_edge(
                    relationship.source_concept,
                    relationship.target_concept,
                    relationship_type=relationship.relationship_type.value,
                    strength=relationship.strength,
                    context=relationship.context
                )
                
                # Add reverse edge if bidirectional
                if relationship.bidirectional:
                    self.graph.add_edge(
                        relationship.target_concept,
                        relationship.source_concept,
                        relationship_type=relationship.relationship_type.value,
                        strength=relationship.strength,
                        context=relationship.context
                    )
    
    def find_related_concepts(self, concepts: List[str], max_depth: int = 2) -> List[RelatedConcept]:
        """Find concepts related to the input concepts.
        
        Args:
            concepts: List of concept names to find relationships for
            max_depth: Maximum depth to search in the graph
            
        Returns:
            List of related concepts with relevance scores
        """
        related_concepts = []
        visited = set(concepts)
        
        for concept in concepts:
            if concept not in self.graph:
                continue
                
            # Use NetworkX to find related nodes
            for target_concept in self.graph.nodes():
                if target_concept in visited:
                    continue
                
                try:
                    # Find shortest path
                    path = nx.shortest_path(self.graph, concept, target_concept)
                    if len(path) <= max_depth + 1:  # +1 because path includes source
                        # Calculate relevance score based on path length and relationship strengths
                        relevance_score = self._calculate_path_relevance(path)
                        
                        # Extract relationship types along the path
                        relationship_types = []
                        for i in range(len(path) - 1):
                            edge_data = self.graph.get_edge_data(path[i], path[i+1])
                            if edge_data:
                                rel_type = RelationshipType(edge_data.get('relationship_type', 'similar_to'))
                                relationship_types.append(rel_type)
                        
                        related_concept = RelatedConcept(
                            concept=self.concepts[target_concept],
                            relevance_score=relevance_score,
                            relationship_path=path,
                            relationship_types=relationship_types
                        )
                        related_concepts.append(related_concept)
                        visited.add(target_concept)
                        
                except nx.NetworkXNoPath:
                    continue
        
        # Sort by relevance score
        related_concepts.sort(key=lambda x: x.relevance_score, reverse=True)
        return related_concepts
    
    def get_inheritance_chain(self, class_name: str) -> List[str]:
        """Get complete inheritance chain for a class.
        
        Args:
            class_name: Name of the class
            
        Returns:
            List of class names in inheritance order (from most specific to most general)
        """
        if class_name not in self.class_hierarchy:
            return [class_name]
        
        chain = [class_name]
        current_classes = self.class_hierarchy[class_name]
        visited = {class_name}
        
        while current_classes:
            next_classes = []
            for parent_class in current_classes:
                if parent_class not in visited:
                    chain.append(parent_class)
                    visited.add(parent_class)
                    if parent_class in self.class_hierarchy:
                        next_classes.extend(self.class_hierarchy[parent_class])
            current_classes = next_classes
        
        return chain
    
    def get_concept_by_type(self, concept_type: ConceptType) -> List[ManimConcept]:
        """Get all concepts of a specific type.
        
        Args:
            concept_type: Type of concepts to retrieve
            
        Returns:
            List of concepts of the specified type
        """
        concept_names = self.concept_categories.get(concept_type, set())
        return [self.concepts[name] for name in concept_names if name in self.concepts]
    
    def search_concepts(self, query: str, concept_types: Optional[List[ConceptType]] = None) -> List[ManimConcept]:
        """Search for concepts matching a query.
        
        Args:
            query: Search query
            concept_types: Optional list of concept types to filter by
            
        Returns:
            List of matching concepts
        """
        query_lower = query.lower()
        matching_concepts = []
        
        for concept_name, concept in self.concepts.items():
            # Check if concept type matches filter
            if concept_types and concept.concept_type not in concept_types:
                continue
            
            # Calculate match score
            score = 0
            if query_lower in concept_name.lower():
                score += 2
            if query_lower in concept.description.lower():
                score += 1
            if any(query_lower in tag.lower() for tag in concept.usage_patterns):
                score += 1
            
            if score > 0:
                matching_concepts.append((concept, score))
        
        # Sort by score and return concepts
        matching_concepts.sort(key=lambda x: x[1], reverse=True)
        return [concept for concept, score in matching_concepts]
    
    # Helper methods
    
    def _get_concept_type_from_category(self, category: str) -> ConceptType:
        """Map directory category to concept type."""
        mapping = {
            "mobject": ConceptType.MOBJECT,
            "animation": ConceptType.ANIMATION,
            "scene": ConceptType.SCENE,
            "camera": ConceptType.CAMERA,
            "renderer": ConceptType.RENDERER,
            "utils": ConceptType.UTILITY,
            "config": ConceptType.CONFIG
        }
        return mapping.get(category, ConceptType.UTILITY)
    
    def _calculate_class_complexity(self, node: ast.ClassDef) -> int:
        """Calculate complexity level of a class."""
        complexity = 1
        
        # Count methods
        method_count = sum(1 for item in node.body if isinstance(item, ast.FunctionDef))
        complexity += min(method_count // 5, 3)
        
        # Count inheritance
        if node.bases:
            complexity += 1
        
        # Count decorators
        if node.decorator_list:
            complexity += 1
        
        return min(complexity, 5)
    
    def _extract_usage_patterns(self, content: str, class_name: str) -> List[str]:
        """Extract usage patterns from code content."""
        patterns = []
        
        # Look for common patterns in docstrings and comments
        pattern_keywords = [
            "example", "usage", "use", "create", "animate", "transform", 
            "add", "play", "scene", "mobject"
        ]
        
        for keyword in pattern_keywords:
            if keyword in content.lower():
                patterns.append(f"commonly_used_for_{keyword}")
        
        return patterns[:5]  # Limit to 5 patterns
    
    def _infer_use_cases(self, class_name: str, concept_type: ConceptType, docstring: str) -> List[str]:
        """Infer common use cases for a concept."""
        use_cases = []
        
        # Type-based use cases
        if concept_type == ConceptType.MOBJECT:
            use_cases.extend(["visual_representation", "geometric_shapes", "text_display"])
        elif concept_type == ConceptType.ANIMATION:
            use_cases.extend(["object_transformation", "visual_effects", "transitions"])
        elif concept_type == ConceptType.SCENE:
            use_cases.extend(["animation_container", "video_creation", "presentation"])
        
        # Name-based use cases
        name_lower = class_name.lower()
        if "text" in name_lower:
            use_cases.append("text_rendering")
        if "graph" in name_lower:
            use_cases.append("data_visualization")
        if "transform" in name_lower:
            use_cases.append("object_morphing")
        
        return list(set(use_cases))[:5]  # Remove duplicates and limit
    
    def _extract_plugin_use_cases(self, description: str) -> List[str]:
        """Extract use cases from plugin description."""
        use_cases = []
        description_lower = description.lower()
        
        # Common plugin use case patterns
        if "chemistry" in description_lower:
            use_cases.extend(["molecular_visualization", "chemical_reactions", "periodic_table"])
        if "circuit" in description_lower:
            use_cases.extend(["electrical_diagrams", "circuit_analysis", "electronics"])
        if "physics" in description_lower:
            use_cases.extend(["physics_simulation", "force_visualization", "mechanics"])
        if "machine learning" in description_lower or "ml" in description_lower:
            use_cases.extend(["neural_networks", "data_science", "ai_visualization"])
        if "data structure" in description_lower or "algorithm" in description_lower:
            use_cases.extend(["algorithm_visualization", "data_structures", "computer_science"])
        
        return use_cases
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two concept names."""
        # Simple similarity based on common substrings
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        # Check for common prefixes/suffixes
        common_parts = 0
        for i in range(min(len(name1_lower), len(name2_lower))):
            if name1_lower[i] == name2_lower[i]:
                common_parts += 1
            else:
                break
        
        # Check for common words
        words1 = set(re.findall(r'\w+', name1_lower))
        words2 = set(re.findall(r'\w+', name2_lower))
        common_words = len(words1.intersection(words2))
        
        # Calculate similarity score
        max_len = max(len(name1), len(name2))
        similarity = (common_parts + common_words * 3) / max_len
        
        return min(similarity, 1.0)
    
    def _calculate_path_relevance(self, path: List[str]) -> float:
        """Calculate relevance score for a relationship path."""
        if len(path) <= 1:
            return 0.0
        
        # Base score decreases with path length
        base_score = 1.0 / len(path)
        
        # Adjust based on relationship strengths
        total_strength = 0.0
        for i in range(len(path) - 1):
            edge_data = self.graph.get_edge_data(path[i], path[i+1])
            if edge_data:
                total_strength += edge_data.get('strength', 0.5)
        
        # Average strength along the path
        avg_strength = total_strength / (len(path) - 1) if len(path) > 1 else 0.5
        
        return base_score * avg_strength

@dataclass
class DependencyInfo:
    """Information about code dependencies."""
    import_statements: List[str]
    required_modules: List[str]
    manim_classes: List[str]
    plugin_dependencies: List[str]
    external_libraries: List[str]


@dataclass
class QueryExpansion:
    """Result of semantic query expansion."""
    original_query: str
    expanded_concepts: List[str]
    related_concepts: List[RelatedConcept]
    dependency_concepts: List[str]
    sub_queries: List[str]
    expansion_reasoning: str


class ConceptExpander:
    """Expands queries to include related Manim concepts."""
    
    def __init__(self, knowledge_graph: ManimKnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        
    def expand_concepts(self, query: str, max_related: int = 10) -> List[RelatedConcept]:
        """Expand query concepts to include related Manim concepts.
        
        Args:
            query: Original query string
            max_related: Maximum number of related concepts to return
            
        Returns:
            List of related concepts with relevance scores
        """
        # Extract potential concept names from query
        extracted_concepts = self._extract_concepts_from_query(query)
        
        if not extracted_concepts:
            # If no direct concepts found, search for matching concepts
            extracted_concepts = self._search_concepts_in_query(query)
        
        # Find related concepts for each extracted concept
        all_related = []
        for concept in extracted_concepts:
            related = self.knowledge_graph.find_related_concepts([concept], max_depth=2)
            all_related.extend(related)
        
        # Remove duplicates and sort by relevance
        seen_concepts = set()
        unique_related = []
        for related_concept in all_related:
            if related_concept.concept.name not in seen_concepts:
                unique_related.append(related_concept)
                seen_concepts.add(related_concept.concept.name)
        
        # Sort by relevance and limit results
        unique_related.sort(key=lambda x: x.relevance_score, reverse=True)
        return unique_related[:max_related]
    
    def _extract_concepts_from_query(self, query: str) -> List[str]:
        """Extract known Manim concept names from query."""
        concepts = []
        query_words = re.findall(r'\b\w+\b', query)
        
        # Check each word and combination against known concepts
        for concept_name in self.knowledge_graph.concepts.keys():
            concept_words = re.findall(r'\b\w+\b', concept_name.lower())
            
            # Check if all words of the concept appear in the query
            if all(word in [w.lower() for w in query_words] for word in concept_words):
                concepts.append(concept_name)
        
        return concepts
    
    def _search_concepts_in_query(self, query: str) -> List[str]:
        """Search for concepts that match parts of the query."""
        matching_concepts = self.knowledge_graph.search_concepts(query)
        return [concept.name for concept in matching_concepts[:5]]  # Top 5 matches


class DependencyTracker:
    """Tracks and extracts dependencies from code examples."""
    
    def __init__(self, knowledge_graph: ManimKnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        
    def extract_dependencies(self, code_content: str) -> DependencyInfo:
        """Extract dependencies from code content.
        
        Args:
            code_content: Python code content to analyze
            
        Returns:
            DependencyInfo object with extracted dependencies
        """
        import_statements = []
        required_modules = []
        manim_classes = []
        plugin_dependencies = []
        external_libraries = []
        
        try:
            # Parse the code
            tree = ast.parse(code_content)
            
            # Extract import statements
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_statements.append(f"import {alias.name}")
                        self._categorize_import(alias.name, required_modules, 
                                             manim_classes, plugin_dependencies, external_libraries)
                
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        import_statements.append(f"from {module} import {alias.name}")
                        self._categorize_from_import(module, alias.name, required_modules,
                                                   manim_classes, plugin_dependencies, external_libraries)
                
                # Extract class usage (instantiation)
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        class_name = node.func.id
                        if class_name in self.knowledge_graph.concepts:
                            manim_classes.append(class_name)
        
        except SyntaxError:
            # If code can't be parsed, try regex-based extraction
            import_statements, required_modules, manim_classes, plugin_dependencies, external_libraries = \
                self._extract_dependencies_regex(code_content)
        
        return DependencyInfo(
            import_statements=list(set(import_statements)),
            required_modules=list(set(required_modules)),
            manim_classes=list(set(manim_classes)),
            plugin_dependencies=list(set(plugin_dependencies)),
            external_libraries=list(set(external_libraries))
        )
    
    def _categorize_import(self, module_name: str, required_modules: List[str], 
                          manim_classes: List[str], plugin_dependencies: List[str], 
                          external_libraries: List[str]) -> None:
        """Categorize an import statement."""
        if module_name.startswith('manim'):
            required_modules.append(module_name)
            if any(plugin in module_name for plugin in self.knowledge_graph.plugin_namespaces):
                plugin_dependencies.append(module_name)
        elif module_name in ['numpy', 'scipy', 'matplotlib', 'PIL', 'cv2']:
            external_libraries.append(module_name)
        else:
            required_modules.append(module_name)
    
    def _categorize_from_import(self, module: str, name: str, required_modules: List[str],
                               manim_classes: List[str], plugin_dependencies: List[str],
                               external_libraries: List[str]) -> None:
        """Categorize a from-import statement."""
        if module.startswith('manim'):
            required_modules.append(module)
            if name in self.knowledge_graph.concepts:
                manim_classes.append(name)
            if any(plugin in module for plugin in self.knowledge_graph.plugin_namespaces):
                plugin_dependencies.append(f"{module}.{name}")
        elif module in ['numpy', 'scipy', 'matplotlib', 'PIL', 'cv2']:
            external_libraries.append(f"{module}.{name}")
        else:
            required_modules.append(f"{module}.{name}")
    
    def _extract_dependencies_regex(self, code_content: str) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
        """Extract dependencies using regex when AST parsing fails."""
        import_statements = []
        required_modules = []
        manim_classes = []
        plugin_dependencies = []
        external_libraries = []
        
        # Extract import statements
        import_patterns = [
            r'^import\s+([^\s#]+)',
            r'^from\s+([^\s]+)\s+import\s+([^\s#,]+)'
        ]
        
        for line in code_content.split('\n'):
            line = line.strip()
            
            for pattern in import_patterns:
                match = re.match(pattern, line)
                if match:
                    import_statements.append(line)
                    if 'from' in pattern:
                        module, name = match.groups()
                        self._categorize_from_import(module, name, required_modules,
                                                   manim_classes, plugin_dependencies, external_libraries)
                    else:
                        module = match.group(1)
                        self._categorize_import(module, required_modules,
                                             manim_classes, plugin_dependencies, external_libraries)
        
        return import_statements, required_modules, manim_classes, plugin_dependencies, external_libraries


class SemanticRelationshipEngine:
    """Main engine for semantic understanding and query expansion."""
    
    def __init__(self, manim_docs_path: str):
        """Initialize the semantic relationship engine.
        
        Args:
            manim_docs_path: Path to Manim documentation
        """
        self.knowledge_graph = ManimKnowledgeGraph(manim_docs_path)
        self.concept_expander = ConceptExpander(self.knowledge_graph)
        self.dependency_tracker = DependencyTracker(self.knowledge_graph)
    
    def expand_query_concepts(self, query: str) -> QueryExpansion:
        """Expand query to include related Manim concepts.
        
        Args:
            query: Original query string
            
        Returns:
            QueryExpansion object with expanded concepts and sub-queries
        """
        # Extract and expand concepts
        related_concepts = self.concept_expander.expand_concepts(query)
        
        # Extract dependency concepts
        dependency_concepts = self._extract_dependency_concepts(query, related_concepts)
        
        # Generate sub-queries
        sub_queries = self._decompose_query(query, related_concepts)
        
        # Create expansion reasoning
        reasoning = self._create_expansion_reasoning(query, related_concepts, dependency_concepts, sub_queries)
        
        return QueryExpansion(
            original_query=query,
            expanded_concepts=[rc.concept.name for rc in related_concepts],
            related_concepts=related_concepts,
            dependency_concepts=dependency_concepts,
            sub_queries=sub_queries,
            expansion_reasoning=reasoning
        )
    
    def find_class_relationships(self, class_name: str) -> Dict[str, any]:
        """Find relationships for a specific Manim class.
        
        Args:
            class_name: Name of the class to analyze
            
        Returns:
            Dictionary with relationship information
        """
        if class_name not in self.knowledge_graph.concepts:
            return {"error": f"Class {class_name} not found in knowledge graph"}
        
        concept = self.knowledge_graph.concepts[class_name]
        
        # Get inheritance chain
        inheritance_chain = self.knowledge_graph.get_inheritance_chain(class_name)
        
        # Get related concepts
        related_concepts = self.knowledge_graph.find_related_concepts([class_name])
        
        # Get usage patterns
        usage_patterns = concept.usage_patterns
        common_use_cases = concept.common_use_cases
        
        return {
            "class_name": class_name,
            "concept_type": concept.concept_type.value,
            "inheritance_chain": inheritance_chain,
            "related_concepts": [
                {
                    "name": rc.concept.name,
                    "relevance_score": rc.relevance_score,
                    "relationship_types": [rt.value for rt in rc.relationship_types]
                }
                for rc in related_concepts[:10]
            ],
            "methods": concept.methods,
            "usage_patterns": usage_patterns,
            "common_use_cases": common_use_cases,
            "plugin_namespace": concept.plugin_namespace,
            "description": concept.description
        }
    
    def extract_dependencies(self, code_example: str) -> DependencyInfo:
        """Extract dependencies from a code example.
        
        Args:
            code_example: Code content to analyze
            
        Returns:
            DependencyInfo with extracted dependencies
        """
        return self.dependency_tracker.extract_dependencies(code_example)
    
    def decompose_complex_query(self, query: str) -> List[str]:
        """Decompose complex queries into targeted sub-queries.
        
        Args:
            query: Complex query to decompose
            
        Returns:
            List of sub-queries
        """
        return self._decompose_query(query, [])
    
    def _extract_dependency_concepts(self, query: str, related_concepts: List[RelatedConcept]) -> List[str]:
        """Extract concepts that are dependencies of the main concepts."""
        dependency_concepts = []
        
        # Look for dependency relationships in related concepts
        for related_concept in related_concepts:
            for rel_type in related_concept.relationship_types:
                if rel_type == RelationshipType.DEPENDS_ON:
                    dependency_concepts.append(related_concept.concept.name)
        
        # Add common dependencies based on concept types
        concept_types = [rc.concept.concept_type for rc in related_concepts]
        
        if ConceptType.ANIMATION in concept_types:
            dependency_concepts.extend(["Mobject", "Scene"])
        if ConceptType.MOBJECT in concept_types:
            dependency_concepts.extend(["Scene"])
        if ConceptType.SCENE in concept_types:
            dependency_concepts.extend(["Camera", "Renderer"])
        
        return list(set(dependency_concepts))
    
    def _decompose_query(self, query: str, related_concepts: List[RelatedConcept]) -> List[str]:
        """Decompose query into sub-queries targeting different aspects."""
        sub_queries = []
        query_lower = query.lower()
        
        # Basic decomposition patterns
        if "how to" in query_lower:
            # Tutorial-focused sub-queries
            sub_queries.append(f"tutorial {query}")
            sub_queries.append(f"example {query}")
            sub_queries.append(f"step by step {query}")
        
        if "create" in query_lower or "make" in query_lower:
            # Creation-focused sub-queries
            sub_queries.append(f"API reference {query}")
            sub_queries.append(f"constructor {query}")
            sub_queries.append(f"parameters {query}")
        
        if "animate" in query_lower or "animation" in query_lower:
            # Animation-focused sub-queries
            sub_queries.append(f"animation examples {query}")
            sub_queries.append(f"transform {query}")
            sub_queries.append(f"play method {query}")
        
        # Concept-specific sub-queries
        for related_concept in related_concepts[:3]:  # Top 3 related concepts
            concept_name = related_concept.concept.name
            sub_queries.append(f"{concept_name} usage")
            sub_queries.append(f"{concept_name} examples")
            
            if related_concept.concept.concept_type == ConceptType.MOBJECT:
                sub_queries.append(f"{concept_name} properties")
                sub_queries.append(f"{concept_name} methods")
            elif related_concept.concept.concept_type == ConceptType.ANIMATION:
                sub_queries.append(f"{concept_name} parameters")
                sub_queries.append(f"{concept_name} rate function")
        
        # Remove duplicates and limit
        sub_queries = list(set(sub_queries))
        return sub_queries[:8]  # Limit to 8 sub-queries
    
    def _create_expansion_reasoning(self, original_query: str, related_concepts: List[RelatedConcept],
                                  dependency_concepts: List[str], sub_queries: List[str]) -> str:
        """Create reasoning explanation for the query expansion."""
        reasoning_parts = [
            f"Original query: '{original_query}'",
            f"Found {len(related_concepts)} related concepts",
        ]
        
        if related_concepts:
            top_concepts = [rc.concept.name for rc in related_concepts[:3]]
            reasoning_parts.append(f"Top related concepts: {', '.join(top_concepts)}")
        
        if dependency_concepts:
            reasoning_parts.append(f"Dependency concepts: {', '.join(dependency_concepts[:3])}")
        
        reasoning_parts.append(f"Generated {len(sub_queries)} sub-queries for comprehensive coverage")
        
        return ". ".join(reasoning_parts)
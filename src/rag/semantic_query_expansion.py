"""
Semantic Query Expansion System for RAG Enhancement

This module implements semantic query expansion including:
- Concept expansion to include related Manim concepts
- Dependency extraction for code examples (imports, requirements)
- Query decomposition for complex queries

Requirements addressed: 4.3, 4.4, 4.6
"""

import ast
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from pathlib import Path

# Import existing components
from .query_generation import EnhancedQuery, QueryIntent, ComplexityLevel
try:
    from .semantic_understanding import (
        ManimKnowledgeGraph, RelatedConcept, DependencyInfo, 
        ConceptType, RelationshipType
    )
except ImportError:
    # Fallback if semantic understanding is not available
    ManimKnowledgeGraph = None
    RelatedConcept = None
    DependencyInfo = None


@dataclass
class ConceptExpansion:
    """Result of concept expansion for a query."""
    original_concepts: List[str]
    expanded_concepts: List[str]
    related_concepts: List[str]
    concept_relationships: Dict[str, List[str]]
    expansion_confidence: float


@dataclass
class DependencyExtraction:
    """Result of dependency extraction from code examples."""
    import_statements: List[str]
    required_modules: List[str]
    manim_classes: List[str]
    plugin_dependencies: List[str]
    external_libraries: List[str]
    dependency_graph: Dict[str, List[str]]


@dataclass
class QueryDecomposition:
    """Result of query decomposition into sub-queries."""
    original_query: str
    sub_queries: List[str]
    query_aspects: List[str]
    complexity_breakdown: Dict[str, ComplexityLevel]
    decomposition_strategy: str


@dataclass
class SemanticExpansionResult:
    """Complete result of semantic query expansion."""
    original_query: EnhancedQuery
    concept_expansion: ConceptExpansion
    dependency_extraction: Optional[DependencyExtraction]
    query_decomposition: QueryDecomposition
    expanded_queries: List[EnhancedQuery]
    expansion_metadata: Dict[str, Any]


class ConceptExpander:
    """Expands queries to include related Manim concepts."""
    
    def __init__(self, knowledge_graph: Optional[ManimKnowledgeGraph] = None):
        self.knowledge_graph = knowledge_graph
        self.manim_concept_map = self._build_concept_map()
        self.concept_synonyms = self._build_concept_synonyms()
        self.concept_hierarchies = self._build_concept_hierarchies()
    
    def expand_concepts(self, query: EnhancedQuery) -> ConceptExpansion:
        """
        Expand query concepts to include related Manim concepts.
        
        Args:
            query: Enhanced query to expand
            
        Returns:
            ConceptExpansion with related concepts and relationships
        """
        # Extract concepts from the query
        original_concepts = self._extract_concepts_from_query(query.original_query)
        
        # Expand concepts using knowledge graph if available
        if self.knowledge_graph:
            expanded_concepts, relationships = self._expand_with_knowledge_graph(
                original_concepts, query
            )
        else:
            expanded_concepts, relationships = self._expand_with_heuristics(
                original_concepts, query
            )
        
        # Find related concepts
        related_concepts = self._find_related_concepts(original_concepts, expanded_concepts)
        
        # Calculate expansion confidence
        confidence = self._calculate_expansion_confidence(
            original_concepts, expanded_concepts, related_concepts
        )
        
        return ConceptExpansion(
            original_concepts=original_concepts,
            expanded_concepts=expanded_concepts,
            related_concepts=related_concepts,
            concept_relationships=relationships,
            expansion_confidence=confidence
        )
    
    def _extract_concepts_from_query(self, query: str) -> List[str]:
        """Extract Manim concepts from query text."""
        concepts = []
        query_lower = query.lower()
        
        # Extract explicit Manim concepts
        for concept in self.manim_concept_map.keys():
            if concept.lower() in query_lower:
                concepts.append(concept)
        
        # Extract CamelCase class names
        class_names = re.findall(r'\b[A-Z][a-zA-Z]+\b', query)
        concepts.extend(class_names)
        
        # Extract function-like patterns
        function_patterns = re.findall(r'\b[a-z_][a-zA-Z0-9_]*(?=\(|\s)', query)
        concepts.extend(function_patterns)
        
        return list(set(concepts))
    
    def _expand_with_knowledge_graph(self, concepts: List[str], 
                                   query: EnhancedQuery) -> Tuple[List[str], Dict[str, List[str]]]:
        """Expand concepts using the knowledge graph."""
        expanded_concepts = concepts.copy()
        relationships = {}
        
        for concept in concepts:
            # Find related concepts in the knowledge graph
            related = self.knowledge_graph.find_related_concepts([concept], max_depth=2)
            
            concept_related = []
            for related_concept in related[:5]:  # Limit to top 5 related concepts
                related_name = related_concept.concept.name
                expanded_concepts.append(related_name)
                concept_related.append(related_name)
            
            if concept_related:
                relationships[concept] = concept_related
        
        return list(set(expanded_concepts)), relationships
    
    def _expand_with_heuristics(self, concepts: List[str], 
                              query: EnhancedQuery) -> Tuple[List[str], Dict[str, List[str]]]:
        """Expand concepts using heuristic rules when knowledge graph is not available."""
        expanded_concepts = concepts.copy()
        relationships = {}
        
        for concept in concepts:
            concept_lower = concept.lower()
            related = []
            
            # Check concept synonyms
            if concept_lower in self.concept_synonyms:
                related.extend(self.concept_synonyms[concept_lower])
            
            # Check concept hierarchies
            if concept_lower in self.concept_hierarchies:
                related.extend(self.concept_hierarchies[concept_lower])
            
            # Add intent-based expansions
            if query.intent == QueryIntent.EXAMPLE_CODE:
                related.extend(self._get_example_related_concepts(concept))
            elif query.intent == QueryIntent.API_REFERENCE:
                related.extend(self._get_api_related_concepts(concept))
            
            if related:
                relationships[concept] = related
                expanded_concepts.extend(related)
        
        return list(set(expanded_concepts)), relationships  
  
    def _find_related_concepts(self, original: List[str], expanded: List[str]) -> List[str]:
        """Find additional related concepts based on patterns."""
        related = []
        
        for concept in original:
            concept_lower = concept.lower()
            
            # Find concepts with similar patterns
            if 'mobject' in concept_lower:
                related.extend(['VMobject', 'Mobject', 'Group', 'VGroup'])
            elif 'animation' in concept_lower:
                related.extend(['Transform', 'Create', 'FadeIn', 'FadeOut'])
            elif 'scene' in concept_lower:
                related.extend(['Scene', 'MovingCameraScene', 'ThreeDScene'])
            elif 'text' in concept_lower:
                related.extend(['Text', 'MathTex', 'Tex', 'MarkupText'])
            elif 'graph' in concept_lower:
                related.extend(['Axes', 'NumberPlane', 'FunctionGraph'])
        
        return list(set(related))
    
    def _get_example_related_concepts(self, concept: str) -> List[str]:
        """Get concepts related to examples for a given concept."""
        concept_lower = concept.lower()
        related = []
        
        if 'circle' in concept_lower:
            related.extend(['Square', 'Rectangle', 'Polygon', 'RegularPolygon'])
        elif 'transform' in concept_lower:
            related.extend(['ReplacementTransform', 'TransformMatchingTex', 'AnimationGroup'])
        elif 'create' in concept_lower:
            related.extend(['Write', 'DrawBorderThenFill', 'ShowCreation'])
        
        return related
    
    def _get_api_related_concepts(self, concept: str) -> List[str]:
        """Get API-related concepts for a given concept."""
        concept_lower = concept.lower()
        related = []
        
        if 'mobject' in concept_lower:
            related.extend(['add', 'remove', 'shift', 'rotate', 'scale'])
        elif 'scene' in concept_lower:
            related.extend(['construct', 'play', 'wait', 'add', 'remove'])
        elif 'animation' in concept_lower:
            related.extend(['run_time', 'rate_func', 'lag_ratio'])
        
        return related
    
    def _calculate_expansion_confidence(self, original: List[str], 
                                      expanded: List[str], related: List[str]) -> float:
        """Calculate confidence score for concept expansion."""
        if not original:
            return 0.0
        
        expansion_ratio = len(expanded) / len(original)
        related_ratio = len(related) / len(original) if original else 0
        
        # Higher confidence for moderate expansion (not too little, not too much)
        if 2 <= expansion_ratio <= 5:
            base_confidence = 0.8
        elif 1 < expansion_ratio < 2:
            base_confidence = 0.6
        elif expansion_ratio > 5:
            base_confidence = 0.4  # Too much expansion might be noisy
        else:
            base_confidence = 0.3
        
        # Boost confidence if we have good related concepts
        if related_ratio > 0.5:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _build_concept_map(self) -> Dict[str, List[str]]:
        """Build a map of Manim concepts and their categories."""
        return {
            # Core Mobjects
            'Mobject': ['base_class', 'core'],
            'VMobject': ['vector_mobject', 'core'],
            'Group': ['container', 'core'],
            'VGroup': ['vector_group', 'core'],
            
            # Geometric shapes
            'Circle': ['geometry', 'shape'],
            'Square': ['geometry', 'shape'],
            'Rectangle': ['geometry', 'shape'],
            'Polygon': ['geometry', 'shape'],
            'Line': ['geometry', 'shape'],
            'Arrow': ['geometry', 'shape'],
            
            # Text objects
            'Text': ['text', 'display'],
            'MathTex': ['text', 'math'],
            'Tex': ['text', 'latex'],
            'MarkupText': ['text', 'markup'],
            
            # Animations
            'Animation': ['animation', 'base'],
            'Transform': ['animation', 'transform'],
            'Create': ['animation', 'creation'],
            'FadeIn': ['animation', 'fade'],
            'FadeOut': ['animation', 'fade'],
            'Write': ['animation', 'text'],
            
            # Scenes
            'Scene': ['scene', 'base'],
            'MovingCameraScene': ['scene', 'camera'],
            'ThreeDScene': ['scene', '3d'],
            
            # Graphs and plots
            'Axes': ['graph', 'coordinate'],
            'NumberPlane': ['graph', 'coordinate'],
            'FunctionGraph': ['graph', 'function'],
        }
    
    def _build_concept_synonyms(self) -> Dict[str, List[str]]:
        """Build synonyms for Manim concepts."""
        return {
            'circle': ['Circle', 'Dot', 'Annulus'],
            'square': ['Square', 'Rectangle', 'Polygon'],
            'text': ['Text', 'MathTex', 'Tex', 'MarkupText'],
            'animation': ['Transform', 'Create', 'FadeIn', 'FadeOut'],
            'scene': ['Scene', 'MovingCameraScene', 'ThreeDScene'],
            'graph': ['Axes', 'NumberPlane', 'FunctionGraph'],
            'transform': ['Transform', 'ReplacementTransform', 'TransformMatchingTex'],
            'create': ['Create', 'Write', 'DrawBorderThenFill', 'ShowCreation'],
        }
    
    def _build_concept_hierarchies(self) -> Dict[str, List[str]]:
        """Build concept hierarchies for inheritance relationships."""
        return {
            'mobject': ['VMobject', 'Group', 'VGroup'],
            'vmobject': ['Circle', 'Square', 'Text', 'Line'],
            'animation': ['Transform', 'Create', 'Fade'],
            'transform': ['ReplacementTransform', 'TransformMatchingTex'],
            'scene': ['MovingCameraScene', 'ThreeDScene'],
            'text': ['MathTex', 'Tex', 'MarkupText'],
        }

class DependencyExtractor:
    """Extracts dependencies from code examples."""
    
    def __init__(self):
        self.manim_modules = {
            'manim': ['Scene', 'Mobject', 'VMobject', 'Animation'],
            'manim.mobject': ['Circle', 'Square', 'Text', 'Group'],
            'manim.animation': ['Transform', 'Create', 'FadeIn', 'FadeOut'],
            'manim.scene': ['Scene', 'MovingCameraScene', 'ThreeDScene'],
            'manim.utils': ['config', 'color'],
        }
        self.plugin_patterns = {
            'manim_slides': r'from\s+manim_slides',
            'manim_physics': r'from\s+manim_physics',
            'manim_chemistry': r'from\s+manim_chemistry',
            'manim_ml': r'from\s+manim_ml',
        }
    
    def extract_dependencies(self, code_content: str) -> DependencyExtraction:
        """
        Extract dependencies from code examples.
        
        Args:
            code_content: Code content to analyze
            
        Returns:
            DependencyExtraction with all dependency information
        """
        # Extract import statements
        import_statements = self._extract_import_statements(code_content)
        
        # Analyze imports to categorize dependencies
        required_modules = self._extract_required_modules(import_statements)
        manim_classes = self._extract_manim_classes(import_statements, code_content)
        plugin_dependencies = self._extract_plugin_dependencies(import_statements)
        external_libraries = self._extract_external_libraries(import_statements)
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(
            import_statements, manim_classes, plugin_dependencies
        )
        
        return DependencyExtraction(
            import_statements=import_statements,
            required_modules=required_modules,
            manim_classes=manim_classes,
            plugin_dependencies=plugin_dependencies,
            external_libraries=external_libraries,
            dependency_graph=dependency_graph
        )
    
    def _extract_import_statements(self, code_content: str) -> List[str]:
        """Extract all import statements from code."""
        import_statements = []
        
        # Find import statements using regex
        import_patterns = [
            r'^import\s+[\w\.]+(?:\s+as\s+\w+)?',
            r'^from\s+[\w\.]+\s+import\s+.+',
        ]
        
        lines = code_content.split('\n')
        for line in lines:
            line = line.strip()
            for pattern in import_patterns:
                if re.match(pattern, line):
                    import_statements.append(line)
                    break
        
        return import_statements
    
    def _extract_required_modules(self, import_statements: List[str]) -> List[str]:
        """Extract required modules from import statements."""
        modules = set()
        
        for statement in import_statements:
            # Extract module name from import statement
            if statement.startswith('import '):
                module = statement.replace('import ', '').split(' as ')[0].split('.')[0]
                modules.add(module)
            elif statement.startswith('from '):
                module = statement.split(' import ')[0].replace('from ', '').split('.')[0]
                modules.add(module)
        
        return list(modules)
    
    def _extract_manim_classes(self, import_statements: List[str], code_content: str) -> List[str]:
        """Extract Manim classes used in the code."""
        manim_classes = set()
        
        # Extract from import statements
        for statement in import_statements:
            if 'manim' in statement and 'import' in statement:
                # Extract imported items
                if 'from' in statement:
                    imports_part = statement.split('import')[1].strip()
                    # Handle different import formats
                    if '(' in imports_part:
                        imports_part = imports_part.replace('(', '').replace(')', '')
                    
                    imported_items = [item.strip() for item in imports_part.split(',')]
                    for item in imported_items:
                        # Remove 'as alias' part
                        class_name = item.split(' as ')[0].strip()
                        if class_name and class_name[0].isupper():  # Likely a class
                            manim_classes.add(class_name)
        
        # Extract classes used in code (even if not explicitly imported)
        class_patterns = [
            r'\b([A-Z][a-zA-Z]*)\(',  # Class instantiation
            r'class\s+\w+\([^)]*([A-Z][a-zA-Z]*)[^)]*\)',  # Class inheritance
        ]
        
        for pattern in class_patterns:
            matches = re.findall(pattern, code_content)
            for match in matches:
                if isinstance(match, tuple):
                    manim_classes.update(match)
                else:
                    manim_classes.add(match)
        
        # Filter to known Manim classes
        known_manim_classes = set()
        for module_classes in self.manim_modules.values():
            known_manim_classes.update(module_classes)
        
        return list(manim_classes.intersection(known_manim_classes))
    
    def _extract_plugin_dependencies(self, import_statements: List[str]) -> List[str]:
        """Extract plugin dependencies from import statements."""
        plugin_dependencies = []
        
        for statement in import_statements:
            for plugin, pattern in self.plugin_patterns.items():
                if re.search(pattern, statement):
                    plugin_dependencies.append(plugin)
        
        return list(set(plugin_dependencies))
    
    def _extract_external_libraries(self, import_statements: List[str]) -> List[str]:
        """Extract external library dependencies."""
        external_libraries = set()
        manim_related = {'manim', 'numpy', 'scipy', 'matplotlib'}
        
        for statement in import_statements:
            if statement.startswith('import '):
                module = statement.replace('import ', '').split(' as ')[0].split('.')[0]
                if module not in manim_related and not module.startswith('manim_'):
                    external_libraries.add(module)
            elif statement.startswith('from '):
                module = statement.split(' import ')[0].replace('from ', '').split('.')[0]
                if module not in manim_related and not module.startswith('manim_'):
                    external_libraries.add(module)
        
        return list(external_libraries)
    
    def _build_dependency_graph(self, import_statements: List[str], 
                               manim_classes: List[str], 
                               plugin_dependencies: List[str]) -> Dict[str, List[str]]:
        """Build a dependency graph showing relationships between components."""
        graph = {}
        
        # Add Manim class dependencies
        for class_name in manim_classes:
            dependencies = []
            
            # Find module dependencies for each class
            for module, classes in self.manim_modules.items():
                if class_name in classes:
                    dependencies.append(module)
            
            if dependencies:
                graph[class_name] = dependencies
        
        # Add plugin dependencies
        for plugin in plugin_dependencies:
            graph[plugin] = ['manim']  # Plugins depend on manim
        
        return graph
class QueryDecomposer:
    """Decomposes complex queries into targeted sub-queries."""
    
    def __init__(self):
        self.decomposition_strategies = {
            'aspect_based': self._decompose_by_aspects,
            'complexity_based': self._decompose_by_complexity,
            'intent_based': self._decompose_by_intent,
            'concept_based': self._decompose_by_concepts,
        }
        
        self.query_aspects = {
            'api': ['documentation', 'reference', 'parameters', 'methods'],
            'example': ['usage', 'sample', 'implementation', 'demo'],
            'concept': ['overview', 'theory', 'fundamentals', 'explanation'],
            'tutorial': ['guide', 'walkthrough', 'steps', 'learning'],
            'error': ['troubleshooting', 'debugging', 'solution', 'fix'],
        }
    
    def decompose_query(self, query: EnhancedQuery) -> QueryDecomposition:
        """
        Decompose complex queries into targeted sub-queries.
        
        Args:
            query: Enhanced query to decompose
            
        Returns:
            QueryDecomposition with sub-queries and metadata
        """
        # Determine the best decomposition strategy
        strategy = self._select_decomposition_strategy(query)
        
        # Apply the selected strategy
        sub_queries = self.decomposition_strategies[strategy](query)
        
        # Extract query aspects
        query_aspects = self._extract_query_aspects(query)
        
        # Analyze complexity breakdown
        complexity_breakdown = self._analyze_complexity_breakdown(sub_queries)
        
        return QueryDecomposition(
            original_query=query.original_query,
            sub_queries=sub_queries,
            query_aspects=query_aspects,
            complexity_breakdown=complexity_breakdown,
            decomposition_strategy=strategy
        )
    
    def _select_decomposition_strategy(self, query: EnhancedQuery) -> str:
        """Select the best decomposition strategy for a query."""
        query_text = query.original_query.lower()
        
        # Check for multiple concepts (concept-based decomposition)
        concept_count = len(re.findall(r'\b[A-Z][a-zA-Z]+\b', query.original_query))
        if concept_count > 2:
            return 'concept_based'
        
        # Check for multiple aspects (aspect-based decomposition)
        aspect_indicators = 0
        for aspect_keywords in self.query_aspects.values():
            if any(keyword in query_text for keyword in aspect_keywords):
                aspect_indicators += 1
        
        if aspect_indicators > 1:
            return 'aspect_based'
        
        # Check for high complexity (complexity-based decomposition)
        if query.complexity_level == ComplexityLevel.BROAD:
            return 'complexity_based'
        
        # Check for mixed intents (intent-based decomposition)
        intent_keywords = {
            'api': ['api', 'documentation', 'reference'],
            'example': ['example', 'sample', 'demo'],
            'tutorial': ['tutorial', 'guide', 'how to'],
        }
        
        intent_count = 0
        for keywords in intent_keywords.values():
            if any(keyword in query_text for keyword in keywords):
                intent_count += 1
        
        if intent_count > 1:
            return 'intent_based'
        
        # Default to aspect-based decomposition
        return 'aspect_based'
    
    def _decompose_by_aspects(self, query: EnhancedQuery) -> List[str]:
        """Decompose query by different aspects (API, examples, concepts, etc.)."""
        sub_queries = []
        base_concepts = self._extract_main_concepts(query.original_query)
        
        for concept in base_concepts:
            # API aspect
            sub_queries.append(f"{concept} API documentation")
            sub_queries.append(f"{concept} method reference")
            
            # Example aspect
            sub_queries.append(f"{concept} usage example")
            sub_queries.append(f"{concept} code sample")
            
            # Conceptual aspect
            sub_queries.append(f"{concept} overview")
            sub_queries.append(f"understanding {concept}")
        
        return sub_queries
    
    def _decompose_by_complexity(self, query: EnhancedQuery) -> List[str]:
        """Decompose query by complexity levels."""
        sub_queries = []
        base_concepts = self._extract_main_concepts(query.original_query)
        
        for concept in base_concepts:
            # Specific level
            sub_queries.append(f"{concept} function signature")
            sub_queries.append(f"{concept} parameters")
            
            # Moderate level
            sub_queries.append(f"{concept} usage patterns")
            sub_queries.append(f"{concept} implementation")
            
            # Broad level
            sub_queries.append(f"{concept} fundamentals")
            sub_queries.append(f"{concept} concepts")
        
        return sub_queries
    
    def _decompose_by_intent(self, query: EnhancedQuery) -> List[str]:
        """Decompose query by different intents."""
        sub_queries = []
        base_concepts = self._extract_main_concepts(query.original_query)
        
        for concept in base_concepts:
            # Different intent-based queries
            sub_queries.append(f"{concept} API reference")  # API_REFERENCE
            sub_queries.append(f"{concept} example code")   # EXAMPLE_CODE
            sub_queries.append(f"{concept} tutorial")       # TUTORIAL
            sub_queries.append(f"{concept} explanation")    # CONCEPTUAL
        
        return sub_queries
    
    def _decompose_by_concepts(self, query: EnhancedQuery) -> List[str]:
        """Decompose query by individual concepts."""
        sub_queries = []
        concepts = self._extract_main_concepts(query.original_query)
        
        # Create individual queries for each concept
        for concept in concepts:
            sub_queries.append(f"{concept} documentation")
            sub_queries.append(f"how to use {concept}")
        
        # Create combination queries for related concepts
        if len(concepts) > 1:
            for i, concept1 in enumerate(concepts):
                for concept2 in concepts[i+1:]:
                    sub_queries.append(f"{concept1} with {concept2}")
                    sub_queries.append(f"{concept1} and {concept2} example")
        
        return sub_queries
    
    def _extract_main_concepts(self, query: str) -> List[str]:
        """Extract main concepts from a query."""
        concepts = []
        
        # Extract CamelCase words (likely class names)
        camel_case = re.findall(r'\b[A-Z][a-zA-Z]+\b', query)
        concepts.extend(camel_case)
        
        # Extract quoted terms
        quoted_terms = re.findall(r'"([^"]+)"', query)
        concepts.extend(quoted_terms)
        
        # Extract function-like patterns
        functions = re.findall(r'\b[a-z_][a-zA-Z0-9_]*(?=\()', query)
        concepts.extend(functions)
        
        return list(set(concepts))
    
    def _extract_query_aspects(self, query: EnhancedQuery) -> List[str]:
        """Extract aspects present in the query."""
        aspects = []
        query_text = query.original_query.lower()
        
        for aspect, keywords in self.query_aspects.items():
            if any(keyword in query_text for keyword in keywords):
                aspects.append(aspect)
        
        return aspects
    
    def _analyze_complexity_breakdown(self, sub_queries: List[str]) -> Dict[str, ComplexityLevel]:
        """Analyze complexity level of each sub-query."""
        complexity_breakdown = {}
        
        for sub_query in sub_queries:
            query_lower = sub_query.lower()
            
            # Determine complexity based on keywords
            if any(word in query_lower for word in ['signature', 'parameters', 'method']):
                complexity = ComplexityLevel.SPECIFIC
            elif any(word in query_lower for word in ['usage', 'implementation', 'patterns']):
                complexity = ComplexityLevel.MODERATE
            elif any(word in query_lower for word in ['overview', 'fundamentals', 'concepts']):
                complexity = ComplexityLevel.BROAD
            else:
                complexity = ComplexityLevel.MODERATE  # Default
            
            complexity_breakdown[sub_query] = complexity
        
        return complexity_breakdown

class SemanticQueryExpansionEngine:
    """
    Main engine for semantic query expansion combining all components.
    
    This class orchestrates concept expansion, dependency extraction,
    and query decomposition to create comprehensive query expansions.
    """
    
    def __init__(self, knowledge_graph: Optional[ManimKnowledgeGraph] = None):
        self.knowledge_graph = knowledge_graph
        self.concept_expander = ConceptExpander(knowledge_graph)
        self.dependency_extractor = DependencyExtractor()
        self.query_decomposer = QueryDecomposer()
    
    def expand_query_semantically(self, query: EnhancedQuery, 
                                 code_context: Optional[str] = None) -> SemanticExpansionResult:
        """
        Perform comprehensive semantic expansion of a query.
        
        Args:
            query: Enhanced query to expand
            code_context: Optional code context for dependency extraction
            
        Returns:
            SemanticExpansionResult with all expansion components
        """
        # Step 1: Expand concepts
        concept_expansion = self.concept_expander.expand_concepts(query)
        
        # Step 2: Extract dependencies if code context is provided
        dependency_extraction = None
        if code_context:
            dependency_extraction = self.dependency_extractor.extract_dependencies(code_context)
        
        # Step 3: Decompose query
        query_decomposition = self.query_decomposer.decompose_query(query)
        
        # Step 4: Create expanded queries
        expanded_queries = self._create_expanded_queries(
            query, concept_expansion, dependency_extraction, query_decomposition
        )
        
        # Step 5: Compile expansion metadata
        expansion_metadata = self._compile_expansion_metadata(
            concept_expansion, dependency_extraction, query_decomposition
        )
        
        return SemanticExpansionResult(
            original_query=query,
            concept_expansion=concept_expansion,
            dependency_extraction=dependency_extraction,
            query_decomposition=query_decomposition,
            expanded_queries=expanded_queries,
            expansion_metadata=expansion_metadata
        )
    
    def _create_expanded_queries(self, original_query: EnhancedQuery,
                               concept_expansion: ConceptExpansion,
                               dependency_extraction: Optional[DependencyExtraction],
                               query_decomposition: QueryDecomposition) -> List[EnhancedQuery]:
        """Create expanded queries from all expansion components."""
        expanded_queries = []
        
        # Create queries from concept expansion
        for concept in concept_expansion.expanded_concepts:
            expanded_query = EnhancedQuery(
                original_query=f"{concept} documentation",
                expanded_queries=[f"{concept} documentation", f"{concept} usage"],
                intent=original_query.intent,
                complexity_level=ComplexityLevel.MODERATE,
                plugins=original_query.plugins,
                confidence_score=original_query.confidence_score * concept_expansion.expansion_confidence,
                metadata={
                    **original_query.metadata,
                    'expansion_type': 'concept_expansion',
                    'source_concept': concept,
                    'expansion_confidence': concept_expansion.expansion_confidence
                }
            )
            expanded_queries.append(expanded_query)
        
        # Create queries from dependency extraction
        if dependency_extraction:
            for manim_class in dependency_extraction.manim_classes:
                expanded_query = EnhancedQuery(
                    original_query=f"{manim_class} dependency example",
                    expanded_queries=[f"{manim_class} import", f"{manim_class} requirements"],
                    intent=QueryIntent.EXAMPLE_CODE,
                    complexity_level=ComplexityLevel.SPECIFIC,
                    plugins=original_query.plugins + dependency_extraction.plugin_dependencies,
                    confidence_score=original_query.confidence_score * 0.8,
                    metadata={
                        **original_query.metadata,
                        'expansion_type': 'dependency_extraction',
                        'dependencies': dependency_extraction.required_modules,
                        'plugins': dependency_extraction.plugin_dependencies
                    }
                )
                expanded_queries.append(expanded_query)
        
        # Create queries from decomposition
        for sub_query in query_decomposition.sub_queries:
            complexity = query_decomposition.complexity_breakdown.get(
                sub_query, ComplexityLevel.MODERATE
            )
            
            expanded_query = EnhancedQuery(
                original_query=sub_query,
                expanded_queries=[sub_query],
                intent=original_query.intent,
                complexity_level=complexity,
                plugins=original_query.plugins,
                confidence_score=original_query.confidence_score * 0.9,
                metadata={
                    **original_query.metadata,
                    'expansion_type': 'query_decomposition',
                    'decomposition_strategy': query_decomposition.decomposition_strategy,
                    'parent_query': original_query.original_query
                }
            )
            expanded_queries.append(expanded_query)
        
        return expanded_queries
    
    def _compile_expansion_metadata(self, concept_expansion: ConceptExpansion,
                                  dependency_extraction: Optional[DependencyExtraction],
                                  query_decomposition: QueryDecomposition) -> Dict[str, Any]:
        """Compile metadata from all expansion components."""
        metadata = {
            'concept_expansion': {
                'original_concepts': concept_expansion.original_concepts,
                'expanded_count': len(concept_expansion.expanded_concepts),
                'related_count': len(concept_expansion.related_concepts),
                'confidence': concept_expansion.expansion_confidence,
                'relationships': concept_expansion.concept_relationships
            },
            'query_decomposition': {
                'strategy': query_decomposition.decomposition_strategy,
                'sub_query_count': len(query_decomposition.sub_queries),
                'aspects': query_decomposition.query_aspects,
                'complexity_distribution': {
                    level.name: sum(1 for l in query_decomposition.complexity_breakdown.values() if l == level)
                    for level in ComplexityLevel
                }
            }
        }
        
        if dependency_extraction:
            metadata['dependency_extraction'] = {
                'import_count': len(dependency_extraction.import_statements),
                'manim_classes': dependency_extraction.manim_classes,
                'plugin_dependencies': dependency_extraction.plugin_dependencies,
                'external_libraries': dependency_extraction.external_libraries,
                'dependency_graph_size': len(dependency_extraction.dependency_graph)
            }
        
        return metadata
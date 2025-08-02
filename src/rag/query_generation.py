"""
Intelligent Query Generation Engine for RAG System Enhancement

This module implements context-aware query generation that understands implementation plans,
error messages, and storyboards to create diverse, targeted queries for document retrieval.
"""

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Union, Any
from abc import ABC, abstractmethod

# Import the new diversification system
try:
    from .query_diversification import (
        QueryDiversificationSystem, DiversityScore, QueryType,
        QueryComplexityVariator, QueryTypeClassifier, QueryDeduplicator
    )
except ImportError:
    # Fallback if diversification module is not available
    QueryDiversificationSystem = None
    DiversityScore = None

class QueryIntent(Enum):
    """Intent classification for queries"""
    API_REFERENCE = "api_reference"
    EXAMPLE_CODE = "example_code"
    CONCEPTUAL = "conceptual"
    ERROR_SOLUTION = "error_solution"
    TUTORIAL = "tutorial"

class TaskType(Enum):
    """Type of task context"""
    IMPLEMENTATION_PLAN = "implementation_plan"
    ERROR_FIXING = "error_fixing"
    STORYBOARD_CREATION = "storyboard_creation"
    TECHNICAL_IMPLEMENTATION = "technical_implementation"
    NARRATION = "narration"

class ComplexityLevel(Enum):
    """Query complexity levels"""
    SPECIFIC = 1  # Specific function names, exact API calls
    MODERATE = 2  # Class names, method groups
    BROAD = 3     # Conceptual topics, general patterns

@dataclass
class InputContext:
    """Context information for query generation"""
    content: str
    task_type: TaskType
    topic: Optional[str] = None
    scene_number: Optional[int] = None
    error_message: Optional[str] = None
    relevant_plugins: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class EnhancedQuery:
    """Enhanced query with metadata and context"""
    original_query: str
    expanded_queries: List[str]
    intent: QueryIntent
    complexity_level: ComplexityLevel
    plugins: List[str]
    confidence_score: float
    metadata: Dict[str, Any]

@dataclass
class QueryContext:
    """Context for query processing"""
    task_type: TaskType
    plugins: List[str]
    domain_concepts: List[str]
    user_intent: QueryIntent

class ContextAnalyzer:
    """Analyzes input context to extract relevant information for query generation"""
    
    def __init__(self):
        self.manim_concepts = [
            "mobject", "vmobject", "scene", "animation", "transform", "create", "write",
            "fadeIn", "fadeOut", "circle", "square", "text", "mathTex", "axes", "graph",
            "camera", "config", "construct", "play", "wait", "add", "remove"
        ]
        self.plugin_patterns = {
            "manim_slides": ["slide", "presentation", "next_slide"],
            "manim_physics": ["physics", "gravity", "collision", "rigid_body"],
            "manim_chemistry": ["molecule", "atom", "bond", "reaction"],
            "manim_ml": ["neural_network", "layer", "activation", "gradient"]
        }
    
    def analyze(self, input_context: InputContext) -> Dict[str, Any]:
        """Analyze input context and extract relevant information"""
        analysis = {
            "task_type": input_context.task_type,
            "manim_concepts": self._extract_manim_concepts(input_context.content),
            "complexity_indicators": self._assess_complexity(input_context.content),
            "plugin_relevance": self._assess_plugin_relevance(input_context),
            "intent_signals": self._detect_intent_signals(input_context),
            "domain_entities": self._extract_domain_entities(input_context.content)
        }
        return analysis
    
    def _extract_manim_concepts(self, content: str) -> List[str]:
        """Extract Manim-specific concepts from content"""
        found_concepts = []
        content_lower = content.lower()
        
        for concept in self.manim_concepts:
            if concept.lower() in content_lower:
                found_concepts.append(concept)
        
        return found_concepts
    
    def _assess_complexity(self, content: str) -> Dict[str, Any]:
        """Assess the complexity level of the content"""
        # Count specific indicators
        function_calls = len(re.findall(r'\w+\(', content))
        class_references = len(re.findall(r'[A-Z][a-zA-Z]+', content))
        code_blocks = len(re.findall(r'```|`[^`]+`', content))
        
        complexity_score = (function_calls * 0.3 + class_references * 0.4 + code_blocks * 0.3)
        
        if complexity_score > 10:
            level = ComplexityLevel.SPECIFIC
        elif complexity_score > 5:
            level = ComplexityLevel.MODERATE
        else:
            level = ComplexityLevel.BROAD
            
        return {
            "level": level,
            "score": complexity_score,
            "function_calls": function_calls,
            "class_references": class_references,
            "code_blocks": code_blocks
        }
    
    def _assess_plugin_relevance(self, input_context: InputContext) -> Dict[str, float]:
        """Assess relevance of different plugins based on content"""
        relevance_scores = {}
        content_lower = input_context.content.lower()
        
        # Check explicit plugin mentions
        if input_context.relevant_plugins:
            for plugin in input_context.relevant_plugins:
                relevance_scores[plugin] = 1.0
        
        # Check for plugin-specific patterns
        for plugin, patterns in self.plugin_patterns.items():
            score = 0.0
            for pattern in patterns:
                if pattern in content_lower:
                    score += 0.3
            relevance_scores[plugin] = min(score, 1.0)
        
        return relevance_scores
    
    def _detect_intent_signals(self, input_context: InputContext) -> List[QueryIntent]:
        """Detect intent signals from the input context"""
        intents = []
        content_lower = input_context.content.lower()
        
        # Error fixing intent
        if input_context.error_message or "error" in content_lower or "fix" in content_lower:
            intents.append(QueryIntent.ERROR_SOLUTION)
        
        # API reference intent
        if any(word in content_lower for word in ["function", "method", "class", "api", "reference"]):
            intents.append(QueryIntent.API_REFERENCE)
        
        # Example code intent
        if any(word in content_lower for word in ["example", "sample", "demo", "how to"]):
            intents.append(QueryIntent.EXAMPLE_CODE)
        
        # Tutorial intent
        if any(word in content_lower for word in ["tutorial", "guide", "learn", "step by step"]):
            intents.append(QueryIntent.TUTORIAL)
        
        # Conceptual intent (default if no specific intent detected)
        if not intents:
            intents.append(QueryIntent.CONCEPTUAL)
        
        return intents
    
    def _extract_domain_entities(self, content: str) -> List[str]:
        """Extract domain-specific entities (class names, function names, etc.)"""
        entities = []
        
        # Extract potential class names (CamelCase)
        class_names = re.findall(r'\b[A-Z][a-zA-Z]+\b', content)
        entities.extend(class_names)
        
        # Extract potential function names (snake_case or camelCase)
        function_names = re.findall(r'\b[a-z_][a-zA-Z0-9_]*\(', content)
        entities.extend([name[:-1] for name in function_names])  # Remove the '('
        
        # Remove duplicates and common words
        common_words = {"The", "This", "That", "With", "From", "To", "In", "On", "At"}
        entities = list(set([e for e in entities if e not in common_words]))
        
        return entities

class QueryTemplateManager:
    """Manages query templates for different contexts and intents"""
    
    def __init__(self):
        self.templates = {
            QueryIntent.API_REFERENCE: [
                "{concept} API documentation",
                "{concept} class reference",
                "{concept} method parameters",
                "how to use {concept}",
                "{concept} function signature"
            ],
            QueryIntent.EXAMPLE_CODE: [
                "{concept} example code",
                "{concept} usage example",
                "sample {concept} implementation",
                "{concept} code snippet",
                "how to implement {concept}"
            ],
            QueryIntent.CONCEPTUAL: [
                "{concept} overview",
                "understanding {concept}",
                "{concept} concepts",
                "{concept} fundamentals",
                "what is {concept}"
            ],
            QueryIntent.ERROR_SOLUTION: [
                "{concept} error fix",
                "solving {concept} problems",
                "{concept} troubleshooting",
                "common {concept} errors",
                "{concept} debugging"
            ],
            QueryIntent.TUTORIAL: [
                "{concept} tutorial",
                "{concept} step by step guide",
                "learning {concept}",
                "{concept} walkthrough",
                "{concept} getting started"
            ]
        }
        
        self.plugin_templates = {
            "manim_slides": [
                "slide transitions",
                "presentation setup",
                "slide navigation"
            ],
            "manim_physics": [
                "physics simulation",
                "gravity effects",
                "collision detection"
            ],
            "manim_chemistry": [
                "molecular visualization",
                "chemical reactions",
                "atomic structures"
            ]
        }
    
    def get_templates(self, intent: QueryIntent, plugins: List[str] = None) -> List[str]:
        """Get query templates for given intent and plugins"""
        templates = self.templates.get(intent, [])
        
        if plugins:
            for plugin in plugins:
                if plugin in self.plugin_templates:
                    templates.extend(self.plugin_templates[plugin])
        
        return templates
    
    def format_template(self, template: str, concept: str, **kwargs) -> str:
        """Format a template with given concept and additional parameters"""
        try:
            return template.format(concept=concept, **kwargs)
        except KeyError:
            return template.replace("{concept}", concept)

class QueryExpander:
    """Expands base queries into diverse, targeted variations"""
    
    def __init__(self):
        self.expansion_strategies = {
            ComplexityLevel.SPECIFIC: self._expand_specific,
            ComplexityLevel.MODERATE: self._expand_moderate,
            ComplexityLevel.BROAD: self._expand_broad
        }
    
    def expand_queries(self, base_queries: List[str], complexity_level: ComplexityLevel, 
                      context: Dict[str, Any]) -> List[str]:
        """Expand base queries based on complexity level and context"""
        expanded = []
        strategy = self.expansion_strategies.get(complexity_level, self._expand_moderate)
        
        for query in base_queries:
            expanded.extend(strategy(query, context))
        
        return list(set(expanded))  # Remove duplicates
    
    def _expand_specific(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Expand query for specific, detailed information"""
        expansions = [query]
        
        # Add parameter-specific queries
        if "function" in query.lower():
            expansions.append(f"{query} parameters")
            expansions.append(f"{query} return value")
            expansions.append(f"{query} usage example")
        
        # Add class-specific queries
        if any(concept in query for concept in context.get("domain_entities", [])):
            expansions.append(f"{query} methods")
            expansions.append(f"{query} attributes")
            expansions.append(f"{query} inheritance")
        
        return expansions
    
    def _expand_moderate(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Expand query for moderate complexity information"""
        expansions = [query]
        
        # Add related concept queries
        manim_concepts = context.get("manim_concepts", [])
        for concept in manim_concepts:
            if concept.lower() not in query.lower():
                expansions.append(f"{query} {concept}")
        
        # Add usage pattern queries
        expansions.append(f"{query} best practices")
        expansions.append(f"{query} common patterns")
        
        return expansions
    
    def _expand_broad(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Expand query for broad, conceptual information"""
        expansions = [query]
        
        # Add conceptual variations
        expansions.extend([
            f"{query} overview",
            f"{query} introduction",
            f"{query} fundamentals",
            f"understanding {query}"
        ])
        
        return expansions

class IntelligentQueryGenerator:
    """
    Main class for intelligent query generation that creates context-aware,
    diverse queries for RAG document retrieval.
    """
    
    def __init__(self, llm_model=None):
        self.llm_model = llm_model
        self.context_analyzer = ContextAnalyzer()
        self.template_manager = QueryTemplateManager()
        self.query_expander = QueryExpander()
        
        # Initialize the diversification system if available
        if QueryDiversificationSystem:
            self.diversification_system = QueryDiversificationSystem()
        else:
            self.diversification_system = None
    
    def generate_queries(self, input_context: InputContext) -> List[EnhancedQuery]:
        """
        Generate diverse, contextually relevant queries from input context.
        
        Args:
            input_context: Context information including content, task type, and metadata
            
        Returns:
            List of enhanced queries with metadata and context information
        """
        # Analyze the input context
        context_analysis = self.context_analyzer.analyze(input_context)
        
        # Generate base queries based on task type
        base_queries = self._generate_base_queries(input_context, context_analysis)
        
        # Create enhanced queries with metadata
        enhanced_queries = []
        for query_data in base_queries:
            enhanced_query = self._create_enhanced_query(
                query_data, input_context, context_analysis
            )
            enhanced_queries.append(enhanced_query)
        
        # Diversify and deduplicate queries
        diversified_queries = self._diversify_queries(enhanced_queries)
        
        return diversified_queries
    
    def _generate_base_queries(self, input_context: InputContext, 
                              context_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate base queries based on task type and context"""
        if input_context.task_type == TaskType.IMPLEMENTATION_PLAN:
            return self._generate_implementation_queries(input_context, context_analysis)
        elif input_context.task_type == TaskType.ERROR_FIXING:
            return self._generate_error_queries(input_context, context_analysis)
        elif input_context.task_type == TaskType.STORYBOARD_CREATION:
            return self._generate_storyboard_queries(input_context, context_analysis)
        elif input_context.task_type == TaskType.TECHNICAL_IMPLEMENTATION:
            return self._generate_technical_queries(input_context, context_analysis)
        elif input_context.task_type == TaskType.NARRATION:
            return self._generate_narration_queries(input_context, context_analysis)
        else:
            return self._generate_generic_queries(input_context, context_analysis)
    
    def _generate_implementation_queries(self, input_context: InputContext, 
                                       context_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate queries targeting specific Manim functionality from implementation plans"""
        queries = []
        
        # Extract key concepts from the implementation plan
        manim_concepts = context_analysis["manim_concepts"]
        domain_entities = context_analysis["domain_entities"]
        
        # Generate API reference queries for specific functions/classes
        for concept in manim_concepts + domain_entities:
            queries.append({
                "query": f"{concept} API documentation",
                "intent": QueryIntent.API_REFERENCE,
                "complexity": ComplexityLevel.SPECIFIC,
                "confidence": 0.9
            })
            
            queries.append({
                "query": f"{concept} usage example",
                "intent": QueryIntent.EXAMPLE_CODE,
                "complexity": ComplexityLevel.MODERATE,
                "confidence": 0.8
            })
        
        # Generate broader conceptual queries
        if input_context.topic:
            queries.append({
                "query": f"{input_context.topic} manim implementation",
                "intent": QueryIntent.CONCEPTUAL,
                "complexity": ComplexityLevel.BROAD,
                "confidence": 0.7
            })
        
        return queries
    
    def _generate_error_queries(self, input_context: InputContext, 
                               context_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate queries focused on error patterns and solutions"""
        queries = []
        
        if input_context.error_message:
            # Extract error type and relevant concepts
            error_concepts = self._extract_error_concepts(input_context.error_message)
            
            for concept in error_concepts:
                queries.append({
                    "query": f"{concept} error solution",
                    "intent": QueryIntent.ERROR_SOLUTION,
                    "complexity": ComplexityLevel.SPECIFIC,
                    "confidence": 0.95
                })
                
                queries.append({
                    "query": f"common {concept} problems",
                    "intent": QueryIntent.ERROR_SOLUTION,
                    "complexity": ComplexityLevel.MODERATE,
                    "confidence": 0.8
                })
        
        # Add general debugging queries
        queries.append({
            "query": "manim debugging techniques",
            "intent": QueryIntent.TUTORIAL,
            "complexity": ComplexityLevel.BROAD,
            "confidence": 0.6
        })
        
        return queries
    
    def _generate_storyboard_queries(self, input_context: InputContext, 
                                   context_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate queries for visual elements and corresponding Manim objects"""
        queries = []
        
        # Extract visual elements from storyboard content
        visual_elements = self._extract_visual_elements(input_context.content)
        
        for element in visual_elements:
            queries.append({
                "query": f"{element} manim object",
                "intent": QueryIntent.API_REFERENCE,
                "complexity": ComplexityLevel.MODERATE,
                "confidence": 0.8
            })
            
            queries.append({
                "query": f"creating {element} animation",
                "intent": QueryIntent.EXAMPLE_CODE,
                "complexity": ComplexityLevel.MODERATE,
                "confidence": 0.85
            })
        
        return queries
    
    def _generate_technical_queries(self, input_context: InputContext, 
                                  context_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate queries for technical implementation details"""
        queries = []
        
        # Focus on implementation-specific concepts
        manim_concepts = context_analysis["manim_concepts"]
        
        for concept in manim_concepts:
            queries.append({
                "query": f"{concept} implementation details",
                "intent": QueryIntent.API_REFERENCE,
                "complexity": ComplexityLevel.SPECIFIC,
                "confidence": 0.9
            })
            
            queries.append({
                "query": f"{concept} best practices",
                "intent": QueryIntent.CONCEPTUAL,
                "complexity": ComplexityLevel.MODERATE,
                "confidence": 0.7
            })
        
        return queries
    
    def _generate_narration_queries(self, input_context: InputContext, 
                                  context_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate queries for narration and explanation content"""
        queries = []
        
        # Focus on educational and explanatory content
        if input_context.topic:
            queries.append({
                "query": f"{input_context.topic} explanation",
                "intent": QueryIntent.CONCEPTUAL,
                "complexity": ComplexityLevel.BROAD,
                "confidence": 0.8
            })
            
            queries.append({
                "query": f"{input_context.topic} tutorial",
                "intent": QueryIntent.TUTORIAL,
                "complexity": ComplexityLevel.BROAD,
                "confidence": 0.75
            })
        
        return queries
    
    def _generate_generic_queries(self, input_context: InputContext, 
                                context_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate generic queries when task type is not specific"""
        queries = []
        
        manim_concepts = context_analysis["manim_concepts"]
        for concept in manim_concepts:
            queries.append({
                "query": f"{concept} documentation",
                "intent": QueryIntent.API_REFERENCE,
                "complexity": ComplexityLevel.MODERATE,
                "confidence": 0.6
            })
        
        return queries
    
    def _extract_error_concepts(self, error_message: str) -> List[str]:
        """Extract relevant concepts from error messages"""
        concepts = []
        
        # Common error patterns
        error_patterns = [
            r"AttributeError.*'(\w+)'",
            r"NameError.*'(\w+)'",
            r"TypeError.*(\w+)",
            r"ImportError.*(\w+)",
            r"(\w+Error)"
        ]
        
        for pattern in error_patterns:
            matches = re.findall(pattern, error_message)
            concepts.extend(matches)
        
        return list(set(concepts))
    
    def _extract_visual_elements(self, content: str) -> List[str]:
        """Extract visual elements from storyboard content"""
        visual_keywords = [
            "circle", "square", "rectangle", "triangle", "line", "arrow", "text",
            "graph", "chart", "plot", "axis", "curve", "point", "vector",
            "animation", "transform", "fade", "move", "rotate", "scale"
        ]
        
        found_elements = []
        content_lower = content.lower()
        
        for keyword in visual_keywords:
            if keyword in content_lower:
                found_elements.append(keyword)
        
        return found_elements
    
    def _create_enhanced_query(self, query_data: Dict[str, Any], 
                              input_context: InputContext, 
                              context_analysis: Dict[str, Any]) -> EnhancedQuery:
        """Create an enhanced query with full metadata"""
        base_query = query_data["query"]
        
        # Expand the query based on complexity level
        expanded_queries = self.query_expander.expand_queries(
            [base_query], 
            query_data["complexity"], 
            context_analysis
        )
        
        return EnhancedQuery(
            original_query=base_query,
            expanded_queries=expanded_queries,
            intent=query_data["intent"],
            complexity_level=query_data["complexity"],
            plugins=input_context.relevant_plugins or [],
            confidence_score=query_data["confidence"],
            metadata={
                "task_type": input_context.task_type,
                "topic": input_context.topic,
                "scene_number": input_context.scene_number,
                "context_analysis": context_analysis
            }
        )
    
    def _diversify_queries(self, queries: List[EnhancedQuery]) -> List[EnhancedQuery]:
        """Diversify queries using the comprehensive diversification system"""
        if not queries:
            return []
        
        # Use the advanced diversification system if available
        if self.diversification_system:
            diversified_queries, diversity_score = self.diversification_system.diversify_queries(
                queries, max_queries=20, target_diversity_score=0.7
            )
            
            # Add diversity score to metadata for monitoring
            for query in diversified_queries:
                if "diversification" not in query.metadata:
                    query.metadata["diversification"] = {}
                query.metadata["diversification"]["diversity_score"] = diversity_score.overall_score
                query.metadata["diversification"]["type_diversity"] = diversity_score.type_diversity
                query.metadata["diversification"]["complexity_diversity"] = diversity_score.complexity_diversity
                query.metadata["diversification"]["semantic_diversity"] = diversity_score.semantic_diversity
                query.metadata["diversification"]["plugin_diversity"] = diversity_score.plugin_diversity
            
            return diversified_queries
        else:
            # Fallback to simple diversification if advanced system is not available
            return self._simple_diversify_queries(queries)
    
    def _simple_diversify_queries(self, queries: List[EnhancedQuery]) -> List[EnhancedQuery]:
        """Simple fallback diversification method"""
        # Group queries by intent and complexity
        grouped_queries = {}
        for query in queries:
            key = (query.intent, query.complexity_level)
            if key not in grouped_queries:
                grouped_queries[key] = []
            grouped_queries[key].append(query)
        
        # Select diverse queries from each group
        diversified = []
        for group_queries in grouped_queries.values():
            # Sort by confidence and take top queries
            group_queries.sort(key=lambda q: q.confidence_score, reverse=True)
            diversified.extend(group_queries[:2])  # Take top 2 from each group
        
        return diversified
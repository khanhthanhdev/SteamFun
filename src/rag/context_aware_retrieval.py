"""
Context-Aware Retrieval Engine for RAG Enhancement

This module implements intelligent document retrieval that considers task context,
result type prioritization, and plugin-aware filtering beyond basic similarity search.
"""

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any, Set
from collections import defaultdict, Counter
import numpy as np

from .query_generation import EnhancedQuery, QueryIntent, ComplexityLevel
from .plugin_detection import PluginRelevanceScore, PluginRelevanceLevel
from .advanced_result_processing import AdvancedResultProcessor


# Import shared result types
from .result_types import ResultType, SearchResult, RankedResult, ResultMetadata


class TaskType(Enum):
    """Types of tasks for context-aware ranking"""
    ANIMATION_CREATION = "animation_creation"
    ERROR_DEBUGGING = "error_debugging"
    CONCEPT_LEARNING = "concept_learning"
    IMPLEMENTATION = "implementation"
    PRESENTATION = "presentation"
    PHYSICS_SIMULATION = "physics_simulation"


@dataclass
class RetrievalContext:
    """Context information for retrieval ranking"""
    task_type: TaskType
    query_intent: QueryIntent
    complexity_level: ComplexityLevel
    relevant_plugins: List[str] = field(default_factory=list)
    implementation_plan: Optional[str] = None
    error_context: Optional[str] = None
    storyboard_context: Optional[str] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    visual_elements: List[str] = field(default_factory=list)
    technical_requirements: List[str] = field(default_factory=list)


# SearchResult, RankedResult, and ResultMetadata are imported from result_types


class ResultTypeClassifier:
    """Classifies search results into different types for prioritization"""
    
    def __init__(self):
        self.type_indicators = {
            ResultType.COMPLETE_EXAMPLE: {
                "patterns": [
                    r"class\s+\w+.*?def\s+construct",
                    r"from\s+manim\s+import.*?class.*?Scene",
                    r"def\s+construct\(self\):",
                    r"self\.play\(",
                    r"self\.add\(",
                    r"self\.wait\("
                ],
                "keywords": ["example", "complete", "full", "working", "runnable"],
                "metadata_indicators": ["has_examples", "semantic_complete"]
            },
            ResultType.API_REFERENCE: {
                "patterns": [
                    r"class\s+\w+\([^)]*\):",
                    r"def\s+\w+\([^)]*\):",
                    r"Parameters:",
                    r"Returns:",
                    r"Args:"
                ],
                "keywords": ["api", "reference", "documentation", "method", "function", "class"],
                "metadata_indicators": ["content_type", "api"]
            },
            ResultType.PARTIAL_SNIPPET: {
                "patterns": [
                    r"^\s*[a-zA-Z_]\w*\s*=",
                    r"^\s*\w+\.",
                    r"^\s*#"
                ],
                "keywords": ["snippet", "code", "line"],
                "metadata_indicators": ["semantic_complete"]
            },
            ResultType.TUTORIAL: {
                "patterns": [
                    r"step\s+\d+",
                    r"first.*then.*finally",
                    r"let's.*create",
                    r"we.*will.*show"
                ],
                "keywords": ["tutorial", "guide", "walkthrough", "step", "learn", "how to"],
                "metadata_indicators": ["content_type", "tutorial"]
            },
            ResultType.CONCEPT_EXPLANATION: {
                "patterns": [
                    r"what\s+is",
                    r"understanding",
                    r"concept\s+of",
                    r"principle"
                ],
                "keywords": ["concept", "explanation", "theory", "principle", "understanding"],
                "metadata_indicators": ["content_type", "concept"]
            },
            ResultType.ERROR_SOLUTION: {
                "patterns": [
                    r"error",
                    r"exception",
                    r"fix",
                    r"solve",
                    r"troubleshoot"
                ],
                "keywords": ["error", "fix", "solution", "debug", "troubleshoot", "problem"],
                "metadata_indicators": ["error", "solution"]
            }
        }
    
    def classify_result(self, result: SearchResult) -> ResultType:
        """Classify a search result into its primary type"""
        content = result.content.lower()
        metadata = result.metadata
        
        # Score each result type
        type_scores = {}
        for result_type, indicators in self.type_indicators.items():
            score = 0.0
            
            # Pattern matching
            for pattern in indicators["patterns"]:
                if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                    score += 1.0
            
            # Keyword matching
            for keyword in indicators["keywords"]:
                if keyword in content:
                    score += 0.5
            
            # Metadata indicators
            for meta_key in indicators["metadata_indicators"]:
                if meta_key in metadata:
                    if isinstance(metadata[meta_key], bool) and metadata[meta_key]:
                        score += 0.7
                    elif isinstance(metadata[meta_key], str) and metadata[meta_key]:
                        score += 0.7
            
            type_scores[result_type] = score
        
        # Return the type with the highest score
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            if type_scores[best_type] > 0:
                return best_type
        
        # Default classification based on content analysis
        if "class" in content and "def construct" in content:
            return ResultType.COMPLETE_EXAMPLE
        elif "def " in content or "class " in content:
            return ResultType.API_REFERENCE
        elif any(word in content for word in ["tutorial", "guide", "step"]):
            return ResultType.TUTORIAL
        elif any(word in content for word in ["concept", "theory", "principle"]):
            return ResultType.CONCEPT_EXPLANATION
        else:
            return ResultType.PARTIAL_SNIPPET


class ContextualResultRanker:
    """Ranks results based on context relevance, not just similarity"""
    
    def __init__(self):
        self.result_type_classifier = ResultTypeClassifier()
        
        # Priority weights for different result types by task type
        self.type_priorities = {
            TaskType.ANIMATION_CREATION: {
                ResultType.COMPLETE_EXAMPLE: 1.0,
                ResultType.API_REFERENCE: 0.8,
                ResultType.TUTORIAL: 0.7,
                ResultType.PARTIAL_SNIPPET: 0.4,
                ResultType.CONCEPT_EXPLANATION: 0.3,
                ResultType.ERROR_SOLUTION: 0.2
            },
            TaskType.ERROR_DEBUGGING: {
                ResultType.ERROR_SOLUTION: 1.0,
                ResultType.COMPLETE_EXAMPLE: 0.8,
                ResultType.API_REFERENCE: 0.7,
                ResultType.PARTIAL_SNIPPET: 0.5,
                ResultType.TUTORIAL: 0.3,
                ResultType.CONCEPT_EXPLANATION: 0.2
            },
            TaskType.CONCEPT_LEARNING: {
                ResultType.CONCEPT_EXPLANATION: 1.0,
                ResultType.TUTORIAL: 0.9,
                ResultType.COMPLETE_EXAMPLE: 0.7,
                ResultType.API_REFERENCE: 0.6,
                ResultType.PARTIAL_SNIPPET: 0.3,
                ResultType.ERROR_SOLUTION: 0.2
            },
            TaskType.IMPLEMENTATION: {
                ResultType.COMPLETE_EXAMPLE: 1.0,
                ResultType.API_REFERENCE: 0.9,
                ResultType.PARTIAL_SNIPPET: 0.6,
                ResultType.TUTORIAL: 0.5,
                ResultType.CONCEPT_EXPLANATION: 0.3,
                ResultType.ERROR_SOLUTION: 0.4
            },
            TaskType.PRESENTATION: {
                ResultType.COMPLETE_EXAMPLE: 1.0,
                ResultType.TUTORIAL: 0.8,
                ResultType.CONCEPT_EXPLANATION: 0.7,
                ResultType.API_REFERENCE: 0.6,
                ResultType.PARTIAL_SNIPPET: 0.3,
                ResultType.ERROR_SOLUTION: 0.2
            },
            TaskType.PHYSICS_SIMULATION: {
                ResultType.COMPLETE_EXAMPLE: 1.0,
                ResultType.API_REFERENCE: 0.8,
                ResultType.CONCEPT_EXPLANATION: 0.7,
                ResultType.TUTORIAL: 0.6,
                ResultType.PARTIAL_SNIPPET: 0.4,
                ResultType.ERROR_SOLUTION: 0.3
            }
        }
    
    def rank_by_context(self, results: List[SearchResult], 
                       context: RetrievalContext) -> List[RankedResult]:
        """Rank results based on context relevance"""
        ranked_results = []
        
        for result in results:
            # Classify result type
            result_type = self.result_type_classifier.classify_result(result)
            
            # Calculate context score
            context_score = self._calculate_context_score(result, context, result_type)
            
            # Calculate final score combining similarity and context
            final_score = self._combine_scores(result.similarity_score, context_score, context)
            
            # Get ranking factors for transparency
            ranking_factors = self._get_ranking_factors(result, context, result_type)
            
            # Get boost reasons
            boost_reasons = self._get_boost_reasons(result, context, result_type, ranking_factors)
            
            ranked_result = RankedResult(
                content=result.content,
                metadata=result.metadata,
                similarity_score=result.similarity_score,
                context_score=context_score,
                final_score=final_score,
                result_type=result_type,
                chunk_id=result.chunk_id,
                ranking_factors=ranking_factors,
                boost_reasons=boost_reasons
            )
            
            ranked_results.append(ranked_result)
        
        # Sort by final score
        ranked_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return ranked_results
    
    def _calculate_context_score(self, result: SearchResult, 
                                context: RetrievalContext, 
                                result_type: ResultType) -> float:
        """Calculate context relevance score for a result"""
        score = 0.0
        
        # 1. Task type alignment (30% weight)
        task_priority = self.type_priorities.get(context.task_type, {}).get(result_type, 0.5)
        score += task_priority * 0.3
        
        # 2. Plugin relevance (25% weight)
        plugin_score = self._calculate_plugin_relevance(result, context.relevant_plugins)
        score += plugin_score * 0.25
        
        # 3. Complexity alignment (20% weight)
        complexity_score = self._calculate_complexity_alignment(result, context.complexity_level)
        score += complexity_score * 0.2
        
        # 4. Content completeness (15% weight)
        completeness_score = self._calculate_completeness_score(result, result_type)
        score += completeness_score * 0.15
        
        # 5. Technical requirements match (10% weight)
        tech_score = self._calculate_technical_match(result, context.technical_requirements)
        score += tech_score * 0.1
        
        return min(score, 1.0)
    
    def _calculate_plugin_relevance(self, result: SearchResult, 
                                   relevant_plugins: List[str]) -> float:
        """Calculate plugin relevance score"""
        if not relevant_plugins:
            return 0.5  # Neutral score if no plugins specified
        
        content = result.content.lower()
        metadata = result.metadata
        
        # Check plugin namespace in metadata
        plugin_namespace = metadata.get('plugin_namespace', '')
        if plugin_namespace and plugin_namespace.lower() in [p.lower() for p in relevant_plugins]:
            return 1.0
        
        # Check plugin mentions in content
        plugin_mentions = 0
        for plugin in relevant_plugins:
            if plugin.lower() in content:
                plugin_mentions += 1
        
        if plugin_mentions > 0:
            return min(plugin_mentions / len(relevant_plugins), 1.0)
        
        return 0.0
    
    def _calculate_complexity_alignment(self, result: SearchResult, 
                                      target_complexity: ComplexityLevel) -> float:
        """Calculate complexity alignment score"""
        result_complexity = result.metadata.get('complexity_level', 2)
        target_level = target_complexity.value if hasattr(target_complexity, 'value') else 2
        
        # Perfect match gets full score
        if result_complexity == target_level:
            return 1.0
        
        # Calculate distance penalty
        distance = abs(result_complexity - target_level)
        max_distance = 4  # Assuming complexity levels 1-5
        
        return max(0.0, 1.0 - (distance / max_distance))
    
    def _calculate_completeness_score(self, result: SearchResult, 
                                     result_type: ResultType) -> float:
        """Calculate content completeness score"""
        metadata = result.metadata
        
        # Complete examples get highest score
        if result_type == ResultType.COMPLETE_EXAMPLE:
            return 1.0
        
        # Check semantic completeness
        if metadata.get('semantic_complete', False):
            return 0.8
        
        # Check for examples
        if metadata.get('has_examples', False):
            return 0.6
        
        # Check for docstrings
        if metadata.get('has_docstring', False):
            return 0.4
        
        return 0.2
    
    def _calculate_technical_match(self, result: SearchResult, 
                                  tech_requirements: List[str]) -> float:
        """Calculate technical requirements match score"""
        if not tech_requirements:
            return 0.5  # Neutral score
        
        content = result.content.lower()
        matches = 0
        
        for requirement in tech_requirements:
            if requirement.lower() in content:
                matches += 1
        
        return matches / len(tech_requirements) if tech_requirements else 0.0
    
    def _combine_scores(self, similarity_score: float, context_score: float, 
                       context: RetrievalContext) -> float:
        """Combine similarity and context scores"""
        # Weight based on query intent
        if context.query_intent == QueryIntent.API_REFERENCE:
            # For API queries, similarity is more important
            return similarity_score * 0.7 + context_score * 0.3
        elif context.query_intent == QueryIntent.EXAMPLE_CODE:
            # For example queries, context is more important
            return similarity_score * 0.4 + context_score * 0.6
        else:
            # Balanced approach for other intents
            return similarity_score * 0.5 + context_score * 0.5
    
    def _get_ranking_factors(self, result: SearchResult, context: RetrievalContext, 
                           result_type: ResultType) -> Dict[str, float]:
        """Get detailed ranking factors for transparency"""
        return {
            "similarity_score": result.similarity_score,
            "task_type_priority": self.type_priorities.get(context.task_type, {}).get(result_type, 0.5),
            "plugin_relevance": self._calculate_plugin_relevance(result, context.relevant_plugins),
            "complexity_alignment": self._calculate_complexity_alignment(result, context.complexity_level),
            "completeness_score": self._calculate_completeness_score(result, result_type),
            "technical_match": self._calculate_technical_match(result, context.technical_requirements)
        }
    
    def _get_boost_reasons(self, result: SearchResult, context: RetrievalContext, 
                          result_type: ResultType, ranking_factors: Dict[str, float]) -> List[str]:
        """Get human-readable boost reasons"""
        reasons = []
        
        if ranking_factors["task_type_priority"] > 0.8:
            reasons.append(f"High priority for {context.task_type.value} tasks")
        
        if ranking_factors["plugin_relevance"] > 0.8:
            reasons.append("Matches relevant plugins")
        
        if ranking_factors["completeness_score"] > 0.8:
            reasons.append("Complete example with full context")
        
        if ranking_factors["complexity_alignment"] > 0.8:
            reasons.append("Matches target complexity level")
        
        if ranking_factors["technical_match"] > 0.6:
            reasons.append("Matches technical requirements")
        
        if result_type == ResultType.COMPLETE_EXAMPLE:
            reasons.append("Complete runnable example")
        
        return reasons


class PluginAwareFilter:
    """Filters results based on plugin relevance"""
    
    def __init__(self):
        pass
    
    def filter_by_plugins(self, results: List[RankedResult], 
                         relevant_plugins: List[str],
                         min_relevance_threshold: float = 0.3) -> List[RankedResult]:
        """Filter results based on plugin relevance"""
        if not relevant_plugins:
            return results  # No filtering if no plugins specified
        
        filtered_results = []
        
        for result in results:
            plugin_score = self._calculate_plugin_score(result, relevant_plugins)
            
            # Keep results that meet the minimum threshold or have high overall scores
            if plugin_score >= min_relevance_threshold or result.final_score > 0.8:
                filtered_results.append(result)
        
        return filtered_results
    
    def _calculate_plugin_score(self, result: RankedResult, 
                               relevant_plugins: List[str]) -> float:
        """Calculate plugin relevance score for a result"""
        content = result.content.lower()
        metadata = result.metadata
        
        # Check plugin namespace
        plugin_namespace = metadata.get('plugin_namespace', '')
        if plugin_namespace and plugin_namespace.lower() in [p.lower() for p in relevant_plugins]:
            return 1.0
        
        # Check plugin mentions
        plugin_mentions = sum(1 for plugin in relevant_plugins if plugin.lower() in content)
        
        return min(plugin_mentions / len(relevant_plugins), 1.0) if relevant_plugins else 0.0


class ContextAwareRetriever:
    """
    Main context-aware retrieval engine that combines all components
    to provide intelligent document retrieval beyond basic similarity search.
    """
    
    def __init__(self, vector_store_manager):
        """
        Initialize the context-aware retriever.
        
        Args:
            vector_store_manager: Vector store manager for basic similarity search
        """
        self.vector_store_manager = vector_store_manager
        self.result_ranker = ContextualResultRanker()
        self.plugin_filter = PluginAwareFilter()
        self.result_type_classifier = ResultTypeClassifier()
        self.advanced_processor = AdvancedResultProcessor()
    
    def retrieve_documents(self, queries: List[EnhancedQuery], 
                          context: RetrievalContext,
                          max_results: int = 20,
                          diversity_threshold: float = 0.7) -> List[RankedResult]:
        """
        Retrieve and rank documents with context awareness.
        
        Args:
            queries: List of enhanced queries to search for
            context: Retrieval context for ranking
            max_results: Maximum number of results to return
            diversity_threshold: Minimum diversity threshold for results
            
        Returns:
            List of ranked results with context-aware scoring
        """
        # Step 1: Perform basic similarity search
        raw_results = self._multi_query_search(queries, max_results * 2)  # Get more for filtering
        
        # Step 2: Context-aware ranking
        ranked_results = self.result_ranker.rank_by_context(raw_results, context)
        
        # Step 3: Plugin-aware filtering
        filtered_results = self.plugin_filter.filter_by_plugins(
            ranked_results, context.relevant_plugins
        )
        
        # Step 4: Apply advanced result processing (boosting, deduplication, metadata enrichment)
        processed_results = self.advanced_processor.process_results(
            filtered_results, 
            preserve_diversity=True, 
            enrich_metadata=True
        )
        
        # Step 5: Ensure result diversity (additional diversity check)
        diverse_results = self._ensure_result_diversity(
            processed_results, max_results, diversity_threshold
        )
        
        return diverse_results[:max_results]
    
    def _multi_query_search(self, queries: List[EnhancedQuery], 
                           max_results_per_query: int = 10) -> List[SearchResult]:
        """Perform similarity search across multiple queries"""
        all_results = []
        seen_chunks = set()
        
        for query in queries:
            # Use the original query for search
            query_text = query.original_query
            
            # Perform similarity search (this would interface with the actual vector store)
            # For now, we'll create a placeholder that would be replaced with actual vector store calls
            query_results = self._perform_similarity_search(query_text, max_results_per_query)
            
            # Deduplicate by chunk ID
            for result in query_results:
                if result.chunk_id not in seen_chunks:
                    all_results.append(result)
                    seen_chunks.add(result.chunk_id)
        
        return all_results
    
    def _perform_similarity_search(self, query: str, k: int) -> List[SearchResult]:
        """
        Placeholder for actual similarity search.
        This would interface with the vector store manager.
        """
        # This is a placeholder - in the actual implementation, this would call
        # the vector store manager's search method
        # For now, return empty list as this will be integrated with existing vector store
        return []
    
    def _boost_animation_examples(self, results: List[RankedResult], 
                                 queries: List[EnhancedQuery]) -> List[RankedResult]:
        """Boost results that contain both animation class and usage examples"""
        animation_keywords = [
            "animation", "transform", "create", "fadeIn", "fadeOut", 
            "write", "drawBoundingBox", "showCreation", "uncreate"
        ]
        
        boosted_results = []
        
        for result in results:
            content_lower = result.content.lower()
            
            # Check for animation keywords and class definitions
            has_animation_class = any(keyword in content_lower for keyword in animation_keywords)
            has_usage_example = ("self.play(" in content_lower or 
                               "self.add(" in content_lower or
                               "def construct" in content_lower)
            
            if has_animation_class and has_usage_example:
                # Boost the score
                result.final_score = min(result.final_score * 1.2, 1.0)
                result.boost_reasons.append("Contains both animation class and usage example")
            
            boosted_results.append(result)
        
        # Re-sort after boosting
        boosted_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return boosted_results
    
    def _ensure_result_diversity(self, results: List[RankedResult], 
                                max_results: int, 
                                diversity_threshold: float) -> List[RankedResult]:
        """Ensure diversity in result types and content"""
        if len(results) <= max_results:
            return results
        
        diverse_results = []
        result_types_seen = Counter()
        content_hashes = set()
        
        for result in results:
            # Check content diversity (simple hash-based approach)
            content_hash = hash(result.content[:200])  # Hash first 200 chars
            
            # Check type diversity
            type_count = result_types_seen[result.result_type]
            max_per_type = max(1, max_results // len(ResultType))
            
            # Add result if it's diverse enough
            if (content_hash not in content_hashes and 
                type_count < max_per_type and 
                len(diverse_results) < max_results):
                
                diverse_results.append(result)
                result_types_seen[result.result_type] += 1
                content_hashes.add(content_hash)
            elif len(diverse_results) < max_results and result.final_score > 0.8:
                # Always include very high-scoring results
                diverse_results.append(result)
        
        return diverse_results
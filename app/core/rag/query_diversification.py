"""
Query Diversification System for RAG Enhancement

This module implements query complexity variation, type classification, 
deduplication, and diversity scoring to ensure comprehensive and varied 
query generation for optimal document retrieval.
"""

import re
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
import math

from .query_generation import EnhancedQuery, QueryIntent, ComplexityLevel


class QueryType(Enum):
    """Classification of query types for diversification"""
    API_REFERENCE = "api_reference"
    EXAMPLE_CODE = "example_code"
    CONCEPTUAL = "conceptual"
    TUTORIAL = "tutorial"
    ERROR_SOLUTION = "error_solution"
    BEST_PRACTICES = "best_practices"


@dataclass
class QueryComplexityVariation:
    """Represents different complexity variations of a base query"""
    base_concept: str
    specific_queries: List[str] = field(default_factory=list)
    moderate_queries: List[str] = field(default_factory=list)
    broad_queries: List[str] = field(default_factory=list)


@dataclass
class DiversityScore:
    """Scoring metrics for query diversity"""
    type_diversity: float  # Variety in query types
    complexity_diversity: float  # Variety in complexity levels
    semantic_diversity: float  # Semantic distance between queries
    plugin_diversity: float  # Coverage of different plugins
    overall_score: float  # Combined diversity score


@dataclass
class QueryCluster:
    """Cluster of similar queries for deduplication"""
    representative_query: EnhancedQuery
    similar_queries: List[EnhancedQuery] = field(default_factory=list)
    similarity_threshold: float = 0.8


class QueryComplexityVariator:
    """Creates complexity variations of queries from specific to broad concepts"""
    
    def __init__(self):
        self.complexity_patterns = {
            ComplexityLevel.SPECIFIC: {
                "function_patterns": [
                    "{concept}()",
                    "{concept}.{method}()",
                    "{concept} function parameters",
                    "{concept} method signature",
                    "{concept} return type"
                ],
                "class_patterns": [
                    "{concept} class methods",
                    "{concept} attributes",
                    "{concept} constructor",
                    "{concept} inheritance",
                    "{concept} properties"
                ]
            },
            ComplexityLevel.MODERATE: {
                "usage_patterns": [
                    "how to use {concept}",
                    "{concept} implementation",
                    "{concept} configuration",
                    "{concept} common patterns",
                    "{concept} integration"
                ],
                "relationship_patterns": [
                    "{concept} with other classes",
                    "{concept} dependencies",
                    "{concept} related concepts",
                    "{concept} workflow",
                    "{concept} ecosystem"
                ]
            },
            ComplexityLevel.BROAD: {
                "conceptual_patterns": [
                    "{concept} overview",
                    "understanding {concept}",
                    "{concept} fundamentals",
                    "{concept} theory",
                    "{concept} principles"
                ],
                "domain_patterns": [
                    "{concept} in manim",
                    "{concept} applications",
                    "{concept} use cases",
                    "{concept} domain knowledge",
                    "{concept} best practices"
                ]
            }
        }
    
    def create_complexity_variations(self, base_concept: str, 
                                   original_complexity: ComplexityLevel) -> QueryComplexityVariation:
        """Create queries at different complexity levels for a base concept"""
        variation = QueryComplexityVariation(base_concept=base_concept)
        
        # Generate specific queries
        variation.specific_queries = self._generate_queries_for_level(
            base_concept, ComplexityLevel.SPECIFIC
        )
        
        # Generate moderate queries
        variation.moderate_queries = self._generate_queries_for_level(
            base_concept, ComplexityLevel.MODERATE
        )
        
        # Generate broad queries
        variation.broad_queries = self._generate_queries_for_level(
            base_concept, ComplexityLevel.BROAD
        )
        
        return variation
    
    def _generate_queries_for_level(self, concept: str, level: ComplexityLevel) -> List[str]:
        """Generate queries for a specific complexity level"""
        queries = []
        patterns = self.complexity_patterns.get(level, {})
        
        for pattern_group in patterns.values():
            for pattern in pattern_group:
                try:
                    # Try to format with concept and common placeholders
                    query = pattern.format(concept=concept, method="method")
                    queries.append(query)
                except KeyError:
                    # If formatting fails, just replace {concept} manually
                    query = pattern.replace("{concept}", concept)
                    queries.append(query)
        
        return queries
    
    def vary_query_complexity(self, query: EnhancedQuery) -> List[EnhancedQuery]:
        """Create complexity variations of an existing query"""
        variations = []
        
        # Extract main concept from the original query
        main_concept = self._extract_main_concept(query.original_query)
        if not main_concept:
            return [query]
        
        # Create complexity variations
        complexity_variation = self.create_complexity_variations(
            main_concept, query.complexity_level
        )
        
        # Create EnhancedQuery objects for each variation
        for level, queries in [
            (ComplexityLevel.SPECIFIC, complexity_variation.specific_queries),
            (ComplexityLevel.MODERATE, complexity_variation.moderate_queries),
            (ComplexityLevel.BROAD, complexity_variation.broad_queries)
        ]:
            for varied_query in queries[:2]:  # Limit to 2 per level
                if varied_query.lower() != query.original_query.lower():
                    variation = EnhancedQuery(
                        original_query=varied_query,
                        expanded_queries=[varied_query],
                        intent=query.intent,
                        complexity_level=level,
                        plugins=query.plugins,
                        confidence_score=query.confidence_score * 0.8,  # Slightly lower confidence
                        metadata={**query.metadata, "variation_of": query.original_query}
                    )
                    variations.append(variation)
        
        return [query] + variations
    
    def _extract_main_concept(self, query: str) -> Optional[str]:
        """Extract the main concept from a query string"""
        # Remove common query words
        stop_words = {
            "how", "to", "use", "what", "is", "the", "a", "an", "and", "or", "but",
            "in", "on", "at", "by", "for", "with", "from", "about", "example",
            "documentation", "api", "reference", "tutorial", "guide", "code"
        }
        
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', query.lower())
        filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Prioritize capitalized words (likely class/concept names)
        original_words = re.findall(r'\b[A-Z][a-zA-Z0-9_]*\b', query)
        if original_words:
            return original_words[0].lower()
        
        if filtered_words:
            # Return the first significant word as the main concept
            return filtered_words[0]
        
        return None


class QueryTypeClassifier:
    """Classifies queries into different types for diversification"""
    
    def __init__(self):
        self.type_indicators = {
            QueryType.API_REFERENCE: [
                "api", "reference", "documentation", "method", "function", "class",
                "parameter", "argument", "return", "signature", "interface"
            ],
            QueryType.EXAMPLE_CODE: [
                "example", "sample", "demo", "code", "snippet", "implementation",
                "usage", "how to", "show me", "demonstrate"
            ],
            QueryType.CONCEPTUAL: [
                "concept", "theory", "principle", "understanding", "overview",
                "fundamentals", "basics", "introduction", "what is", "explain"
            ],
            QueryType.TUTORIAL: [
                "tutorial", "guide", "walkthrough", "step by step", "learn",
                "getting started", "beginner", "course", "lesson"
            ],
            QueryType.ERROR_SOLUTION: [
                "error", "fix", "solve", "problem", "issue", "bug", "debug",
                "troubleshoot", "exception", "failure"
            ],
            QueryType.BEST_PRACTICES: [
                "best practice", "recommendation", "pattern", "convention",
                "standard", "guideline", "tip", "advice", "optimization"
            ]
        }
    
    def classify_query(self, query: EnhancedQuery) -> QueryType:
        """Classify a query into its primary type"""
        query_text = query.original_query.lower()
        
        # Count indicators for each type
        type_scores = {}
        for query_type, indicators in self.type_indicators.items():
            score = sum(1 for indicator in indicators if indicator in query_text)
            type_scores[query_type] = score
        
        # Return the type with the highest score
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            if type_scores[best_type] > 0:
                return best_type
        
        # Default classification based on QueryIntent
        intent_to_type = {
            QueryIntent.API_REFERENCE: QueryType.API_REFERENCE,
            QueryIntent.EXAMPLE_CODE: QueryType.EXAMPLE_CODE,
            QueryIntent.CONCEPTUAL: QueryType.CONCEPTUAL,
            QueryIntent.TUTORIAL: QueryType.TUTORIAL,
            QueryIntent.ERROR_SOLUTION: QueryType.ERROR_SOLUTION
        }
        
        return intent_to_type.get(query.intent, QueryType.CONCEPTUAL)
    
    def ensure_type_diversity(self, queries: List[EnhancedQuery], 
                            target_distribution: Optional[Dict[QueryType, int]] = None) -> List[EnhancedQuery]:
        """Ensure queries have diverse types according to target distribution"""
        if not target_distribution:
            # Default balanced distribution
            num_types = len(QueryType)
            target_per_type = max(1, len(queries) // num_types)
            target_distribution = {qtype: target_per_type for qtype in QueryType}
        
        # Classify existing queries
        classified_queries = defaultdict(list)
        for query in queries:
            query_type = self.classify_query(query)
            classified_queries[query_type].append(query)
        
        # Balance the distribution
        balanced_queries = []
        for query_type, target_count in target_distribution.items():
            available_queries = classified_queries[query_type]
            
            if len(available_queries) >= target_count:
                # Take the best queries of this type
                sorted_queries = sorted(available_queries, 
                                      key=lambda q: q.confidence_score, reverse=True)
                balanced_queries.extend(sorted_queries[:target_count])
            else:
                # Take all available queries of this type
                balanced_queries.extend(available_queries)
        
        return balanced_queries


class QueryDeduplicator:
    """Removes duplicate and near-duplicate queries while preserving diversity"""
    
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
    
    def deduplicate_queries(self, queries: List[EnhancedQuery]) -> List[EnhancedQuery]:
        """Remove duplicate queries while preserving the most diverse set"""
        if not queries:
            return []
        
        # Create clusters of similar queries
        clusters = self._cluster_similar_queries(queries)
        
        # Select representative queries from each cluster
        deduplicated = []
        for cluster in clusters:
            representative = self._select_cluster_representative(cluster)
            deduplicated.append(representative)
        
        return deduplicated
    
    def _cluster_similar_queries(self, queries: List[EnhancedQuery]) -> List[QueryCluster]:
        """Cluster similar queries together"""
        clusters = []
        processed = set()
        
        for i, query in enumerate(queries):
            if i in processed:
                continue
            
            # Create a new cluster with this query as representative
            cluster = QueryCluster(representative_query=query)
            processed.add(i)
            
            # Find similar queries to add to this cluster
            for j, other_query in enumerate(queries[i+1:], i+1):
                if j in processed:
                    continue
                
                similarity = self._calculate_query_similarity(query, other_query)
                if similarity >= self.similarity_threshold:
                    cluster.similar_queries.append(other_query)
                    processed.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _calculate_query_similarity(self, query1: EnhancedQuery, query2: EnhancedQuery) -> float:
        """Calculate similarity between two queries"""
        # Normalize query texts
        text1 = self._normalize_query_text(query1.original_query)
        text2 = self._normalize_query_text(query2.original_query)
        
        # Calculate Jaccard similarity
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        jaccard_similarity = intersection / union if union > 0 else 0.0
        
        # Boost similarity if queries have same intent and complexity
        intent_bonus = 0.2 if query1.intent == query2.intent else 0.0
        complexity_bonus = 0.1 if query1.complexity_level == query2.complexity_level else 0.0
        
        return min(1.0, jaccard_similarity + intent_bonus + complexity_bonus)
    
    def _normalize_query_text(self, text: str) -> str:
        """Normalize query text for comparison"""
        # Convert to lowercase and remove punctuation
        normalized = re.sub(r'[^\w\s]', '', text.lower())
        
        # Remove common stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "by", "for",
            "with", "from", "to", "of", "is", "are", "was", "were", "be", "been"
        }
        
        words = [word for word in normalized.split() if word not in stop_words]
        return " ".join(words)
    
    def _select_cluster_representative(self, cluster: QueryCluster) -> EnhancedQuery:
        """Select the best representative query from a cluster"""
        all_queries = [cluster.representative_query] + cluster.similar_queries
        
        # Score queries based on confidence and metadata richness
        scored_queries = []
        for query in all_queries:
            score = query.confidence_score
            
            # Bonus for having expanded queries
            if len(query.expanded_queries) > 1:
                score += 0.1
            
            # Bonus for having plugin information
            if query.plugins:
                score += 0.05
            
            # Bonus for having rich metadata
            if query.metadata and len(query.metadata) > 2:
                score += 0.05
            
            scored_queries.append((score, query))
        
        # Return the highest scoring query
        scored_queries.sort(key=lambda x: x[0], reverse=True)
        return scored_queries[0][1]


class DiversityScorer:
    """Calculates diversity scores for query sets"""
    
    def __init__(self):
        self.type_classifier = QueryTypeClassifier()
    
    def calculate_diversity_score(self, queries: List[EnhancedQuery]) -> DiversityScore:
        """Calculate comprehensive diversity score for a set of queries"""
        if not queries:
            return DiversityScore(0.0, 0.0, 0.0, 0.0, 0.0)
        
        type_diversity = self._calculate_type_diversity(queries)
        complexity_diversity = self._calculate_complexity_diversity(queries)
        semantic_diversity = self._calculate_semantic_diversity(queries)
        plugin_diversity = self._calculate_plugin_diversity(queries)
        
        # Calculate overall score as weighted average
        overall_score = (
            type_diversity * 0.3 +
            complexity_diversity * 0.25 +
            semantic_diversity * 0.3 +
            plugin_diversity * 0.15
        )
        
        return DiversityScore(
            type_diversity=type_diversity,
            complexity_diversity=complexity_diversity,
            semantic_diversity=semantic_diversity,
            plugin_diversity=plugin_diversity,
            overall_score=overall_score
        )
    
    def _calculate_type_diversity(self, queries: List[EnhancedQuery]) -> float:
        """Calculate diversity in query types"""
        types = [self.type_classifier.classify_query(query) for query in queries]
        type_counts = Counter(types)
        
        # Calculate entropy-based diversity
        total = len(types)
        entropy = 0.0
        for count in type_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        # Normalize by maximum possible entropy
        max_entropy = math.log2(len(QueryType))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _calculate_complexity_diversity(self, queries: List[EnhancedQuery]) -> float:
        """Calculate diversity in complexity levels"""
        complexities = [query.complexity_level for query in queries]
        complexity_counts = Counter(complexities)
        
        # Calculate entropy-based diversity
        total = len(complexities)
        entropy = 0.0
        for count in complexity_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        # Normalize by maximum possible entropy
        max_entropy = math.log2(len(ComplexityLevel))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _calculate_semantic_diversity(self, queries: List[EnhancedQuery]) -> float:
        """Calculate semantic diversity between queries"""
        if len(queries) < 2:
            return 1.0
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(queries)):
            for j in range(i + 1, len(queries)):
                similarity = self._calculate_semantic_similarity(queries[i], queries[j])
                similarities.append(similarity)
        
        # Diversity is inverse of average similarity
        avg_similarity = sum(similarities) / len(similarities)
        return 1.0 - avg_similarity
    
    def _calculate_semantic_similarity(self, query1: EnhancedQuery, query2: EnhancedQuery) -> float:
        """Calculate semantic similarity between two queries"""
        # Simple word overlap similarity
        words1 = set(query1.original_query.lower().split())
        words2 = set(query2.original_query.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_plugin_diversity(self, queries: List[EnhancedQuery]) -> float:
        """Calculate diversity in plugin coverage"""
        all_plugins = set()
        for query in queries:
            all_plugins.update(query.plugins)
        
        if not all_plugins:
            return 0.0
        
        # Calculate how evenly plugins are distributed across queries
        plugin_query_counts = defaultdict(int)
        for query in queries:
            for plugin in query.plugins:
                plugin_query_counts[plugin] += 1
        
        if not plugin_query_counts:
            return 0.0
        
        # Calculate entropy of plugin distribution
        total_plugin_mentions = sum(plugin_query_counts.values())
        entropy = 0.0
        for count in plugin_query_counts.values():
            if count > 0:
                p = count / total_plugin_mentions
                entropy -= p * math.log2(p)
        
        # Normalize by maximum possible entropy
        max_entropy = math.log2(len(all_plugins))
        return entropy / max_entropy if max_entropy > 0 else 0.0


class QueryDiversificationSystem:
    """Main system for query diversification, combining all components"""
    
    def __init__(self, similarity_threshold: float = 0.7):
        self.complexity_variator = QueryComplexityVariator()
        self.type_classifier = QueryTypeClassifier()
        self.deduplicator = QueryDeduplicator(similarity_threshold)
        self.diversity_scorer = DiversityScorer()
    
    def diversify_queries(self, queries: List[EnhancedQuery], 
                         max_queries: int = 20,
                         target_diversity_score: float = 0.7) -> Tuple[List[EnhancedQuery], DiversityScore]:
        """
        Diversify a set of queries to maximize variety and coverage.
        
        Args:
            queries: Input queries to diversify
            max_queries: Maximum number of queries to return
            target_diversity_score: Target diversity score to achieve
            
        Returns:
            Tuple of (diversified_queries, diversity_score)
        """
        if not queries:
            return [], DiversityScore(0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Step 1: Create complexity variations
        varied_queries = []
        for query in queries:
            variations = self.complexity_variator.vary_query_complexity(query)
            varied_queries.extend(variations)
        
        # Step 2: Ensure type diversity
        type_diverse_queries = self.type_classifier.ensure_type_diversity(varied_queries)
        
        # Step 3: Remove duplicates
        deduplicated_queries = self.deduplicator.deduplicate_queries(type_diverse_queries)
        
        # Step 4: Select best diverse subset
        final_queries = self._select_diverse_subset(
            deduplicated_queries, max_queries, target_diversity_score
        )
        
        # Step 5: Calculate final diversity score
        diversity_score = self.diversity_scorer.calculate_diversity_score(final_queries)
        
        return final_queries, diversity_score
    
    def _select_diverse_subset(self, queries: List[EnhancedQuery], 
                              max_queries: int, 
                              target_diversity_score: float) -> List[EnhancedQuery]:
        """Select a diverse subset of queries up to max_queries"""
        if len(queries) <= max_queries:
            return queries
        
        # Use greedy selection to maximize diversity
        selected = []
        remaining = queries.copy()
        
        # Start with the highest confidence query
        remaining.sort(key=lambda q: q.confidence_score, reverse=True)
        selected.append(remaining.pop(0))
        
        # Greedily add queries that maximize diversity
        while len(selected) < max_queries and remaining:
            best_query = None
            best_score = -1.0
            best_index = -1
            
            for i, candidate in enumerate(remaining):
                test_set = selected + [candidate]
                diversity_score = self.diversity_scorer.calculate_diversity_score(test_set)
                
                if diversity_score.overall_score > best_score:
                    best_score = diversity_score.overall_score
                    best_query = candidate
                    best_index = i
            
            if best_query:
                selected.append(best_query)
                remaining.pop(best_index)
            else:
                break
        
        return selected
"""
RAG Quality Evaluation Framework

This module provides comprehensive evaluation capabilities for the RAG system,
including relevance metrics, code generation impact measurement, and query pattern analysis.
"""

import json
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum
import numpy as np
from collections import defaultdict, Counter
import re
import os

from .result_types import RankedResult, ResultType


class EvaluationMetric(Enum):
    """Types of evaluation metrics"""
    PRECISION = "precision"
    RECALL = "recall"
    NDCG = "ndcg"
    MAP = "map"  # Mean Average Precision
    MRR = "mrr"  # Mean Reciprocal Rank


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    precision: float
    recall: float
    ndcg: float
    map_score: float = 0.0
    mrr: float = 0.0
    f1_score: float = 0.0
    
    def __post_init__(self):
        """Calculate derived metrics"""
        if self.precision + self.recall > 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)


@dataclass
class CodeQualityMetrics:
    """Metrics for code generation impact"""
    generation_success_rate: float
    code_compilation_rate: float
    functionality_correctness: float
    documentation_coverage: float
    average_generation_time: float
    error_reduction_rate: float = 0.0


@dataclass
class QueryPatternMetrics:
    """Metrics for query pattern analysis"""
    query_type_distribution: Dict[str, int]
    success_rate_by_type: Dict[str, float]
    average_response_time: float
    cache_hit_rate: float
    unique_queries_count: int
    repeated_queries_count: int


@dataclass
class EvaluationResult:
    """Complete evaluation result"""
    timestamp: datetime
    retrieval_metrics: EvaluationMetrics
    code_metrics: CodeQualityMetrics
    query_metrics: QueryPatternMetrics
    sample_size: int
    evaluation_duration: float


class RAGQualityEvaluator:
    """
    Comprehensive quality evaluator for RAG system
    
    Implements standard IR metrics (precision, recall, NDCG) and measures
    code generation impact and query pattern analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the quality evaluator
        
        Args:
            config: Configuration dictionary with evaluation parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Evaluation history storage
        self.evaluation_history: List[EvaluationResult] = []
        self.query_logs: List[Dict[str, Any]] = []
        self.code_generation_logs: List[Dict[str, Any]] = []
        
        # Baseline metrics for comparison
        self.baseline_metrics: Optional[EvaluationMetrics] = None
        
        # Query pattern tracking
        self.query_patterns = defaultdict(list)
        self.success_patterns = defaultdict(list)
        
    def evaluate_retrieval_quality(
        self, 
        queries: List[str], 
        results: List[List[RankedResult]], 
        expected_results: List[List[str]],
        relevance_judgments: Optional[List[List[int]]] = None
    ) -> EvaluationMetrics:
        """
        Evaluate retrieval quality using standard IR metrics
        
        Args:
            queries: List of input queries
            results: List of ranked results for each query
            expected_results: List of expected relevant document IDs for each query
            relevance_judgments: Optional binary relevance judgments (1=relevant, 0=not relevant)
            
        Returns:
            EvaluationMetrics with precision, recall, NDCG, MAP, and MRR
        """
        if len(queries) != len(results) or len(queries) != len(expected_results):
            raise ValueError("Queries, results, and expected_results must have same length")
            
        precisions = []
        recalls = []
        ndcg_scores = []
        ap_scores = []  # Average Precision scores
        rr_scores = []  # Reciprocal Rank scores
        
        for i, (query, query_results, expected) in enumerate(zip(queries, results, expected_results)):
            # Extract retrieved document IDs
            retrieved_ids = [result.chunk_id for result in query_results]
            expected_set = set(expected)
            
            # Use relevance judgments if provided, otherwise binary relevance
            if relevance_judgments and i < len(relevance_judgments):
                relevance = relevance_judgments[i]
            else:
                relevance = [1 if doc_id in expected_set else 0 for doc_id in retrieved_ids]
            
            # Calculate metrics for this query
            precision = self._calculate_precision(retrieved_ids, expected_set)
            recall = self._calculate_recall(retrieved_ids, expected_set)
            ndcg = self._calculate_ndcg(relevance)
            ap = self._calculate_average_precision(relevance)
            rr = self._calculate_reciprocal_rank(relevance)
            
            precisions.append(precision)
            recalls.append(recall)
            ndcg_scores.append(ndcg)
            ap_scores.append(ap)
            rr_scores.append(rr)
            
        # Calculate mean metrics
        return EvaluationMetrics(
            precision=np.mean(precisions),
            recall=np.mean(recalls),
            ndcg=np.mean(ndcg_scores),
            map_score=np.mean(ap_scores),
            mrr=np.mean(rr_scores)
        )
    
    def evaluate_code_generation_impact(
        self, 
        rag_results: List[List[RankedResult]], 
        generated_codes: List[str],
        compilation_results: List[bool],
        functionality_scores: List[float],
        generation_times: List[float]
    ) -> CodeQualityMetrics:
        """
        Measure how retrieved documentation impacts code generation success
        
        Args:
            rag_results: Retrieved documentation for each generation task
            generated_codes: Generated code snippets
            compilation_results: Whether each code compiles successfully
            functionality_scores: Functionality correctness scores (0-1)
            generation_times: Time taken for each generation
            
        Returns:
            CodeQualityMetrics with various code quality measures
        """
        if not all(len(lst) == len(generated_codes) for lst in [compilation_results, functionality_scores, generation_times]):
            raise ValueError("All input lists must have same length")
            
        # Calculate success rates
        generation_success_rate = len([code for code in generated_codes if code.strip()]) / len(generated_codes)
        compilation_rate = sum(compilation_results) / len(compilation_results)
        avg_functionality = np.mean(functionality_scores)
        avg_generation_time = np.mean(generation_times)
        
        # Calculate documentation coverage (how much of retrieved docs was useful)
        doc_coverage_scores = []
        for results, code in zip(rag_results, generated_codes):
            coverage = self._calculate_documentation_coverage(results, code)
            doc_coverage_scores.append(coverage)
        
        avg_doc_coverage = np.mean(doc_coverage_scores)
        
        return CodeQualityMetrics(
            generation_success_rate=generation_success_rate,
            code_compilation_rate=compilation_rate,
            functionality_correctness=avg_functionality,
            documentation_coverage=avg_doc_coverage,
            average_generation_time=avg_generation_time
        )
    
    def track_query_patterns(
        self, 
        query: str, 
        results: List[RankedResult], 
        success_rate: float,
        response_time: float,
        cache_hit: bool = False
    ) -> None:
        """
        Log query patterns and retrieval success rates
        
        Args:
            query: The input query
            results: Retrieved results
            success_rate: Success rate for this query type
            response_time: Time taken to process query
            cache_hit: Whether result came from cache
        """
        query_type = self._classify_query_type(query)
        
        query_log = {
            'timestamp': datetime.now(),
            'query': query,
            'query_type': query_type,
            'results_count': len(results),
            'success_rate': success_rate,
            'response_time': response_time,
            'cache_hit': cache_hit,
            'result_types': [result.result_type.value for result in results]
        }
        
        self.query_logs.append(query_log)
        self.query_patterns[query_type].append(query_log)
        self.success_patterns[query_type].append(success_rate)
    
    def analyze_query_patterns(self, time_window_hours: int = 24) -> QueryPatternMetrics:
        """
        Analyze query patterns and success rates
        
        Args:
            time_window_hours: Time window for analysis in hours
            
        Returns:
            QueryPatternMetrics with pattern analysis results
        """
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_logs = [log for log in self.query_logs if log['timestamp'] > cutoff_time]
        
        if not recent_logs:
            return QueryPatternMetrics(
                query_type_distribution={},
                success_rate_by_type={},
                average_response_time=0.0,
                cache_hit_rate=0.0,
                unique_queries_count=0,
                repeated_queries_count=0
            )
        
        # Query type distribution
        type_counts = Counter(log['query_type'] for log in recent_logs)
        
        # Success rates by type
        success_by_type = {}
        for query_type in type_counts.keys():
            type_logs = [log for log in recent_logs if log['query_type'] == query_type]
            success_by_type[query_type] = np.mean([log['success_rate'] for log in type_logs])
        
        # Response time and cache metrics
        avg_response_time = np.mean([log['response_time'] for log in recent_logs])
        cache_hits = sum(1 for log in recent_logs if log['cache_hit'])
        cache_hit_rate = cache_hits / len(recent_logs)
        
        # Query uniqueness
        unique_queries = set(log['query'] for log in recent_logs)
        unique_count = len(unique_queries)
        repeated_count = len(recent_logs) - unique_count
        
        return QueryPatternMetrics(
            query_type_distribution=dict(type_counts),
            success_rate_by_type=success_by_type,
            average_response_time=avg_response_time,
            cache_hit_rate=cache_hit_rate,
            unique_queries_count=unique_count,
            repeated_queries_count=repeated_count
        )
    
    def run_comprehensive_evaluation(
        self,
        test_queries: List[str],
        test_results: List[List[RankedResult]],
        expected_results: List[List[str]],
        code_generation_data: Optional[Dict[str, List]] = None
    ) -> EvaluationResult:
        """
        Run comprehensive evaluation of the RAG system
        
        Args:
            test_queries: Test queries for evaluation
            test_results: Retrieved results for test queries
            expected_results: Expected relevant results
            code_generation_data: Optional data for code generation evaluation
            
        Returns:
            Complete EvaluationResult
        """
        start_time = time.time()
        
        # Evaluate retrieval quality
        retrieval_metrics = self.evaluate_retrieval_quality(
            test_queries, test_results, expected_results
        )
        
        # Evaluate code generation impact if data provided
        if code_generation_data:
            code_metrics = self.evaluate_code_generation_impact(
                code_generation_data.get('rag_results', []),
                code_generation_data.get('generated_codes', []),
                code_generation_data.get('compilation_results', []),
                code_generation_data.get('functionality_scores', []),
                code_generation_data.get('generation_times', [])
            )
        else:
            code_metrics = CodeQualityMetrics(0, 0, 0, 0, 0)
        
        # Analyze query patterns
        query_metrics = self.analyze_query_patterns()
        
        evaluation_duration = time.time() - start_time
        
        result = EvaluationResult(
            timestamp=datetime.now(),
            retrieval_metrics=retrieval_metrics,
            code_metrics=code_metrics,
            query_metrics=query_metrics,
            sample_size=len(test_queries),
            evaluation_duration=evaluation_duration
        )
        
        self.evaluation_history.append(result)
        return result
    
    def set_baseline_metrics(self, metrics: EvaluationMetrics) -> None:
        """Set baseline metrics for comparison"""
        self.baseline_metrics = metrics
    
    def compare_to_baseline(self, current_metrics: EvaluationMetrics) -> Dict[str, float]:
        """
        Compare current metrics to baseline
        
        Returns:
            Dictionary with metric improvements (positive) or degradations (negative)
        """
        if not self.baseline_metrics:
            return {}
        
        return {
            'precision_change': current_metrics.precision - self.baseline_metrics.precision,
            'recall_change': current_metrics.recall - self.baseline_metrics.recall,
            'ndcg_change': current_metrics.ndcg - self.baseline_metrics.ndcg,
            'f1_change': current_metrics.f1_score - self.baseline_metrics.f1_score
        }
    
    # Helper methods for metric calculations
    
    def _calculate_precision(self, retrieved: List[str], relevant: Set[str]) -> float:
        """Calculate precision at k"""
        if not retrieved:
            return 0.0
        relevant_retrieved = sum(1 for doc_id in retrieved if doc_id in relevant)
        return relevant_retrieved / len(retrieved)
    
    def _calculate_recall(self, retrieved: List[str], relevant: Set[str]) -> float:
        """Calculate recall at k"""
        if not relevant:
            return 0.0
        relevant_retrieved = sum(1 for doc_id in retrieved if doc_id in relevant)
        return relevant_retrieved / len(relevant)
    
    def _calculate_ndcg(self, relevance_scores: List[int], k: Optional[int] = None) -> float:
        """Calculate Normalized Discounted Cumulative Gain"""
        if not relevance_scores:
            return 0.0
            
        k = k or len(relevance_scores)
        relevance_scores = relevance_scores[:k]
        
        # Calculate DCG
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))
        
        # Calculate IDCG (ideal DCG)
        ideal_relevance = sorted(relevance_scores, reverse=True)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_average_precision(self, relevance_scores: List[int]) -> float:
        """Calculate Average Precision"""
        if not relevance_scores:
            return 0.0
            
        relevant_count = 0
        precision_sum = 0.0
        
        for i, rel in enumerate(relevance_scores):
            if rel > 0:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        total_relevant = sum(relevance_scores)
        return precision_sum / total_relevant if total_relevant > 0 else 0.0
    
    def _calculate_reciprocal_rank(self, relevance_scores: List[int]) -> float:
        """Calculate Reciprocal Rank"""
        for i, rel in enumerate(relevance_scores):
            if rel > 0:
                return 1.0 / (i + 1)
        return 0.0
    
    def _calculate_documentation_coverage(self, results: List[RankedResult], generated_code: str) -> float:
        """
        Calculate how much of the retrieved documentation was useful for code generation
        
        This is a heuristic based on content overlap and code structure similarity
        """
        if not results or not generated_code.strip():
            return 0.0
        
        # Extract code snippets and API references from results
        doc_code_snippets = []
        doc_api_refs = []
        
        for result in results:
            # Extract code blocks from documentation
            code_blocks = re.findall(r'```python\s*(.*?)\s*```', result.content, re.DOTALL)
            doc_code_snippets.extend(code_blocks)
            
            # Extract API references (class names, function names)
            api_refs = re.findall(r'\b[A-Z][a-zA-Z]*\b|\b[a-z_]+\([^)]*\)', result.content)
            doc_api_refs.extend(api_refs)
        
        if not doc_code_snippets and not doc_api_refs:
            return 0.0
        
        # Calculate overlap with generated code
        coverage_score = 0.0
        total_elements = len(doc_code_snippets) + len(doc_api_refs)
        
        # Check code snippet similarity
        for snippet in doc_code_snippets:
            if self._code_similarity(snippet, generated_code) > 0.3:
                coverage_score += 1.0
        
        # Check API reference usage
        for api_ref in doc_api_refs:
            if api_ref in generated_code:
                coverage_score += 0.5
        
        return min(coverage_score / total_elements, 1.0) if total_elements > 0 else 0.0
    
    def _code_similarity(self, code1: str, code2: str) -> float:
        """Calculate similarity between two code snippets"""
        # Simple token-based similarity
        tokens1 = set(re.findall(r'\b\w+\b', code1.lower()))
        tokens2 = set(re.findall(r'\b\w+\b', code2.lower()))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0.0
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query type for pattern analysis"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['error', 'fix', 'debug', 'problem']):
            return 'error_fix'
        elif any(word in query_lower for word in ['example', 'how to', 'tutorial']):
            return 'example_request'
        elif any(word in query_lower for word in ['api', 'reference', 'documentation']):
            return 'api_reference'
        elif any(word in query_lower for word in ['animation', 'scene', 'mobject']):
            return 'animation_specific'
        elif any(word in query_lower for word in ['plugin', 'extension']):
            return 'plugin_specific'
        else:
            return 'general'
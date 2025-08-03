"""
RAG Quality Evaluator using RAGAS Framework

This module implements comprehensive RAG quality evaluation using the RAGAS framework,
including relevance evaluation using standard IR metrics (precision, recall, NDCG),
code generation impact measurement, and query pattern analysis.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from collections import defaultdict

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# RAGAS imports with warning suppression
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*pydantic.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas.*")
    
    from ragas import evaluate
    from ragas.dataset_schema import SingleTurnSample
    
    # Import metrics with fallback for missing components
    try:
        from ragas.metrics import (
            ContextPrecision, 
            ContextRecall, 
            Faithfulness, 
            AnswerRelevancy,        
        )
    except ImportError as e:
        # Handle missing metrics gracefully
        print(f"Some RAGAS metrics not available: {e}")
        # Define fallback classes or set to None
        ContextPrecision = None
        ContextRecall = None
        Faithfulness = None
        AnswerRelevancy = None
    
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

# LangChain imports for LLM integration
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .result_types import RankedResult, ResultType


@dataclass
class EvaluationMetrics:
    """Standard IR evaluation metrics"""
    precision: float
    recall: float
    ndcg: float
    context_precision: float = 0.0
    context_recall: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    answer_accuracy: float = 0.0
    response_groundedness: float = 0.0


@dataclass
class CodeQualityMetrics:
    """Code generation quality metrics"""
    generation_success_rate: float
    compilation_success_rate: float
    functionality_score: float
    code_relevance_score: float
    documentation_usage_score: float


@dataclass
class QueryPatternMetrics:
    """Query pattern analysis metrics"""
    total_queries: int
    unique_patterns: int
    success_rate_by_pattern: Dict[str, float]
    average_response_time: float
    cache_hit_rate: float
    pattern_distribution: Dict[str, int]


@dataclass
class EvaluationResult:
    """Complete evaluation result"""
    timestamp: datetime
    retrieval_metrics: EvaluationMetrics
    code_metrics: CodeQualityMetrics
    query_metrics: QueryPatternMetrics
    sample_count: int
    evaluation_duration: float


@dataclass
class QueryLog:
    """Query execution log for pattern analysis"""
    query_id: str
    query: str
    timestamp: datetime
    response_time: float
    results: List[RankedResult]
    generated_code: Optional[str]
    success: bool
    error_message: Optional[str]
    cache_hit: bool
    metadata: Dict[str, Any]


class RAGQualityEvaluator:
    """
    Comprehensive RAG quality evaluator using RAGAS framework
    
    Implements relevance evaluation using standard IR metrics (precision, recall, NDCG),
    code generation impact measurement, and query pattern analysis with success rate tracking.
    """
    
    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        storage_path: Optional[str] = None
    ):
        """
        Initialize RAG quality evaluator
        
        Args:
            llm_model: LLM model for RAGAS evaluation
            embedding_model: Embedding model for RAGAS evaluation
            storage_path: Path to store evaluation results and logs
        """
        self.storage_path = Path(storage_path) if storage_path else Path("rag_evaluation")
        self.storage_path.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize RAGAS components
        self.llm = ChatOpenAI(model=llm_model)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
        # Wrap for RAGAS
        self.evaluator_llm = LangchainLLMWrapper(self.llm)
        self.evaluator_embeddings = LangchainEmbeddingsWrapper(self.embeddings)
        
        # Initialize RAGAS metrics
        self.ragas_metrics = [
            ContextPrecision(llm=self.evaluator_llm),
            ContextRecall(llm=self.evaluator_llm),
            Faithfulness(llm=self.evaluator_llm),
            AnswerRelevancy(llm=self.evaluator_llm, embeddings=self.evaluator_embeddings),
        ]
        
        # Evaluation data storage
        self.evaluation_history: List[EvaluationResult] = []
        self.query_logs: List[QueryLog] = []
        self.baseline_metrics: Optional[EvaluationMetrics] = None
        
        # Query pattern tracking
        self.query_patterns: Dict[str, List[float]] = defaultdict(list)
        self.pattern_classifier = self._create_pattern_classifier()
        
        # Load existing data
        self._load_evaluation_history()
    
    async def evaluate_retrieval_quality(
        self,
        queries: List[str],
        results: List[List[RankedResult]],
        expected_results: Optional[List[List[str]]] = None,
        reference_answers: Optional[List[str]] = None
    ) -> EvaluationMetrics:
        """
        Evaluate retrieval quality using standard IR metrics and RAGAS
        
        Args:
            queries: List of queries
            results: List of ranked results for each query
            expected_results: Expected relevant result IDs for each query (for precision/recall)
            reference_answers: Reference answers for each query (for RAGAS metrics)
            
        Returns:
            EvaluationMetrics with precision, recall, NDCG, and RAGAS scores
        """
        if len(queries) != len(results):
            raise ValueError("Queries and results must have same length")
        
        start_time = time.time()
        
        # Calculate traditional IR metrics
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        
        if expected_results:
            for i, (query, query_results, expected) in enumerate(zip(queries, results, expected_results)):
                # Calculate precision@k and recall@k
                retrieved_ids = [r.chunk_id for r in query_results]
                relevant_retrieved = set(retrieved_ids) & set(expected)
                
                precision = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0.0
                recall = len(relevant_retrieved) / len(expected) if expected else 0.0
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                
                # Calculate NDCG@k
                ndcg = self._calculate_ndcg(query_results, expected)
                ndcg_scores.append(ndcg)
        
        # Calculate RAGAS metrics if reference answers provided
        ragas_scores = {}
        if reference_answers:
            ragas_scores = await self._evaluate_with_ragas(queries, results, reference_answers)
        
        # Aggregate metrics
        metrics = EvaluationMetrics(
            precision=np.mean(precision_scores) if precision_scores else 0.0,
            recall=np.mean(recall_scores) if recall_scores else 0.0,
            ndcg=np.mean(ndcg_scores) if ndcg_scores else 0.0,
            context_precision=ragas_scores.get('context_precision', 0.0),
            context_recall=ragas_scores.get('context_recall', 0.0),
            faithfulness=ragas_scores.get('faithfulness', 0.0),
            answer_relevancy=ragas_scores.get('answer_relevancy', 0.0),
            answer_accuracy=ragas_scores.get('answer_accuracy', 0.0),
            response_groundedness=ragas_scores.get('response_groundedness', 0.0)
        )
        
        evaluation_time = time.time() - start_time
        self.logger.info(f"Retrieval quality evaluation completed in {evaluation_time:.2f}s")
        
        return metrics
    
    def evaluate_code_generation_impact(
        self,
        rag_results: List[RankedResult],
        generated_code: str,
        compilation_success: bool,
        functionality_score: float,
        query: str
    ) -> CodeQualityMetrics:
        """
        Measure how retrieved documentation impacts code generation success
        
        Args:
            rag_results: Retrieved results used for code generation
            generated_code: Generated code
            compilation_success: Whether code compiled successfully
            functionality_score: How well code meets requirements (0-1)
            query: Original query
            
        Returns:
            CodeQualityMetrics measuring generation impact
        """
        # Calculate documentation usage score
        doc_usage_score = self._calculate_documentation_usage(rag_results, generated_code)
        
        # Calculate code relevance to query
        code_relevance = self._calculate_code_relevance(query, generated_code)
        
        # Generation success (binary)
        generation_success = 1.0 if generated_code and len(generated_code.strip()) > 0 else 0.0
        
        # Compilation success (provided)
        compilation_rate = 1.0 if compilation_success else 0.0
        
        metrics = CodeQualityMetrics(
            generation_success_rate=generation_success,
            compilation_success_rate=compilation_rate,
            functionality_score=functionality_score,
            code_relevance_score=code_relevance,
            documentation_usage_score=doc_usage_score
        )
        
        return metrics
    
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
            query: Query string
            results: Retrieved results
            success_rate: Success rate for this query
            response_time: Response time in seconds
            cache_hit: Whether result was from cache
        """
        # Classify query pattern
        pattern = self._classify_query_pattern(query)
        
        # Track pattern success rate
        self.query_patterns[pattern].append(success_rate)
        
        # Create query log
        query_log = QueryLog(
            query_id=f"query_{len(self.query_logs)}_{int(time.time())}",
            query=query,
            timestamp=datetime.now(),
            response_time=response_time,
            results=results,
            generated_code=None,  # Will be updated if code is generated
            success=success_rate > 0.7,  # Consider successful if > 70%
            error_message=None,
            cache_hit=cache_hit,
            metadata={
                'pattern': pattern,
                'result_count': len(results),
                'avg_similarity': np.mean([r.similarity_score for r in results]) if results else 0.0,
                'avg_context_score': np.mean([r.context_score for r in results]) if results else 0.0
            }
        )
        
        self.query_logs.append(query_log)
        
        # Periodically save logs
        if len(self.query_logs) % 100 == 0:
            self._save_query_logs()
    
    def analyze_query_patterns(self) -> QueryPatternMetrics:
        """
        Analyze query patterns and success rates
        
        Returns:
            QueryPatternMetrics with pattern analysis
        """
        if not self.query_logs:
            return QueryPatternMetrics(
                total_queries=0,
                unique_patterns=0,
                success_rate_by_pattern={},
                average_response_time=0.0,
                cache_hit_rate=0.0,
                pattern_distribution={}
            )
        
        # Calculate pattern success rates
        success_rate_by_pattern = {}
        for pattern, success_rates in self.query_patterns.items():
            success_rate_by_pattern[pattern] = np.mean(success_rates) if success_rates else 0.0
        
        # Calculate pattern distribution
        pattern_distribution = defaultdict(int)
        for log in self.query_logs:
            pattern = log.metadata.get('pattern', 'unknown')
            pattern_distribution[pattern] += 1
        
        # Calculate average response time
        response_times = [log.response_time for log in self.query_logs if log.response_time > 0]
        avg_response_time = np.mean(response_times) if response_times else 0.0
        
        # Calculate cache hit rate
        cache_hits = sum(1 for log in self.query_logs if log.cache_hit)
        cache_hit_rate = cache_hits / len(self.query_logs) if self.query_logs else 0.0
        
        return QueryPatternMetrics(
            total_queries=len(self.query_logs),
            unique_patterns=len(self.query_patterns),
            success_rate_by_pattern=success_rate_by_pattern,
            average_response_time=avg_response_time,
            cache_hit_rate=cache_hit_rate,
            pattern_distribution=dict(pattern_distribution)
        )
    
    def set_baseline_metrics(self, metrics: EvaluationMetrics) -> None:
        """Set baseline metrics for comparison"""
        self.baseline_metrics = metrics
        self._save_baseline_metrics()
    
    def compare_with_baseline(self, current_metrics: EvaluationMetrics) -> Dict[str, float]:
        """
        Compare current metrics with baseline
        
        Args:
            current_metrics: Current evaluation metrics
            
        Returns:
            Dictionary of metric differences (positive = improvement)
        """
        if not self.baseline_metrics:
            return {}
        
        return {
            'precision_diff': current_metrics.precision - self.baseline_metrics.precision,
            'recall_diff': current_metrics.recall - self.baseline_metrics.recall,
            'ndcg_diff': current_metrics.ndcg - self.baseline_metrics.ndcg,
            'context_precision_diff': current_metrics.context_precision - self.baseline_metrics.context_precision,
            'context_recall_diff': current_metrics.context_recall - self.baseline_metrics.context_recall,
            'faithfulness_diff': current_metrics.faithfulness - self.baseline_metrics.faithfulness,
            'answer_relevancy_diff': current_metrics.answer_relevancy - self.baseline_metrics.answer_relevancy
        }
    
    async def run_comprehensive_evaluation(
        self,
        queries: List[str],
        results: List[List[RankedResult]],
        expected_results: Optional[List[List[str]]] = None,
        reference_answers: Optional[List[str]] = None,
        generated_codes: Optional[List[str]] = None,
        compilation_results: Optional[List[bool]] = None,
        functionality_scores: Optional[List[float]] = None
    ) -> EvaluationResult:
        """
        Run comprehensive evaluation including retrieval quality, code generation impact,
        and query pattern analysis
        
        Args:
            queries: List of queries
            results: List of ranked results for each query
            expected_results: Expected relevant results for precision/recall
            reference_answers: Reference answers for RAGAS evaluation
            generated_codes: Generated code for each query
            compilation_results: Compilation success for each generated code
            functionality_scores: Functionality scores for each generated code
            
        Returns:
            Complete EvaluationResult
        """
        start_time = time.time()
        
        # Evaluate retrieval quality
        retrieval_metrics = await self.evaluate_retrieval_quality(
            queries, results, expected_results, reference_answers
        )
        
        # Evaluate code generation impact
        code_metrics_list = []
        if generated_codes:
            for i, (query, query_results, code) in enumerate(zip(queries, results, generated_codes)):
                compilation_success = compilation_results[i] if compilation_results else True
                functionality_score = functionality_scores[i] if functionality_scores else 0.8
                
                code_metrics = self.evaluate_code_generation_impact(
                    query_results, code, compilation_success, functionality_score, query
                )
                code_metrics_list.append(code_metrics)
        
        # Aggregate code metrics
        if code_metrics_list:
            aggregated_code_metrics = CodeQualityMetrics(
                generation_success_rate=np.mean([m.generation_success_rate for m in code_metrics_list]),
                compilation_success_rate=np.mean([m.compilation_success_rate for m in code_metrics_list]),
                functionality_score=np.mean([m.functionality_score for m in code_metrics_list]),
                code_relevance_score=np.mean([m.code_relevance_score for m in code_metrics_list]),
                documentation_usage_score=np.mean([m.documentation_usage_score for m in code_metrics_list])
            )
        else:
            aggregated_code_metrics = CodeQualityMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Analyze query patterns
        query_metrics = self.analyze_query_patterns()
        
        # Create evaluation result
        evaluation_duration = time.time() - start_time
        result = EvaluationResult(
            timestamp=datetime.now(),
            retrieval_metrics=retrieval_metrics,
            code_metrics=aggregated_code_metrics,
            query_metrics=query_metrics,
            sample_count=len(queries),
            evaluation_duration=evaluation_duration
        )
        
        # Store result
        self.evaluation_history.append(result)
        self._save_evaluation_result(result)
        
        self.logger.info(f"Comprehensive evaluation completed in {evaluation_duration:.2f}s")
        return result
    
    # Private helper methods
    
    async def _evaluate_with_ragas(
        self,
        queries: List[str],
        results: List[List[RankedResult]],
        reference_answers: List[str]
    ) -> Dict[str, float]:
        """Evaluate using RAGAS metrics"""
        try:
            # Prepare RAGAS dataset
            samples = []
            for query, query_results, reference in zip(queries, results, reference_answers):
                # Extract contexts from results
                contexts = [result.chunk.content for result in query_results[:5]]  # Top 5 results
                
                # Create sample (we'll use reference as response for now)
                sample = SingleTurnSample(
                    user_input=query,
                    response=reference,  # Using reference as response
                    reference=reference,
                    retrieved_contexts=contexts
                )
                samples.append(sample)
            
            # Run RAGAS evaluation
            ragas_scores = {}
            for metric in self.ragas_metrics:
                metric_scores = []
                for sample in samples:
                    try:
                        score = await metric.single_turn_ascore(sample)
                        metric_scores.append(score)
                    except Exception as e:
                        self.logger.warning(f"Error evaluating {metric.__class__.__name__}: {e}")
                        metric_scores.append(0.0)
                
                ragas_scores[metric.__class__.__name__.lower()] = np.mean(metric_scores)
            
            return ragas_scores
            
        except Exception as e:
            self.logger.error(f"Error in RAGAS evaluation: {e}")
            return {}
    
    def _calculate_ndcg(self, results: List[RankedResult], expected_results: List[str]) -> float:
        """Calculate NDCG@k for ranked results"""
        if not results or not expected_results:
            return 0.0
        
        # Create relevance scores (1 if relevant, 0 if not)
        relevance_scores = []
        for result in results:
            relevance_scores.append(1.0 if result.chunk_id in expected_results else 0.0)
        
        # Calculate DCG
        dcg = 0.0
        for i, rel in enumerate(relevance_scores):
            if rel > 0:
                dcg += rel / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        ideal_relevance = sorted([1.0] * len(expected_results), reverse=True)
        idcg = 0.0
        for i, rel in enumerate(ideal_relevance):
            if i >= len(results):
                break
            idcg += rel / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_documentation_usage(self, results: List[RankedResult], generated_code: str) -> float:
        """Calculate how well the generated code uses the retrieved documentation"""
        if not results or not generated_code:
            return 0.0
        
        usage_score = 0.0
        total_weight = 0.0
        
        for result in results:
            # Weight by result ranking (higher ranked results should be used more)
            weight = 1.0 / (results.index(result) + 1)
            total_weight += weight
            
            # Simple text overlap check (could be improved with semantic similarity)
            doc_content = result.chunk.content.lower()
            code_content = generated_code.lower()
            
            # Check for function/class names, keywords, etc.
            doc_words = set(doc_content.split())
            code_words = set(code_content.split())
            
            overlap = len(doc_words & code_words) / len(doc_words) if doc_words else 0.0
            usage_score += overlap * weight
        
        return usage_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_code_relevance(self, query: str, generated_code: str) -> float:
        """Calculate how relevant the generated code is to the original query"""
        if not query or not generated_code:
            return 0.0
        
        # Simple keyword overlap (could be improved with semantic similarity)
        query_words = set(query.lower().split())
        code_words = set(generated_code.lower().split())
        
        # Remove common programming words
        common_words = {'def', 'class', 'import', 'from', 'if', 'else', 'for', 'while', 'return'}
        query_words -= common_words
        code_words -= common_words
        
        if not query_words:
            return 0.5  # Neutral score if no meaningful query words
        
        overlap = len(query_words & code_words) / len(query_words)
        return min(overlap, 1.0)
    
    def _classify_query_pattern(self, query: str) -> str:
        """Classify query into patterns for tracking"""
        query_lower = query.lower()
        
        # Define pattern keywords
        patterns = {
            'api_reference': ['api', 'function', 'method', 'class', 'parameter', 'documentation'],
            'how_to': ['how to', 'how do i', 'tutorial', 'guide', 'example'],
            'error_help': ['error', 'exception', 'bug', 'fix', 'problem', 'issue'],
            'animation': ['animate', 'animation', 'move', 'transform', 'transition'],
            'geometry': ['circle', 'square', 'line', 'polygon', 'shape', 'geometry'],
            'text': ['text', 'label', 'title', 'font', 'string'],
            'math': ['equation', 'formula', 'math', 'latex', 'symbol'],
            'scene': ['scene', 'camera', 'background', 'setup'],
            'plugin': ['plugin', 'extension', 'addon', 'community']
        }
        
        # Check for pattern matches
        for pattern, keywords in patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                return pattern
        
        return 'general'
    
    def _create_pattern_classifier(self) -> Dict[str, List[str]]:
        """Create pattern classifier with keywords"""
        return {
            'api_reference': ['api', 'function', 'method', 'class', 'parameter', 'documentation', 'reference'],
            'how_to': ['how to', 'how do i', 'tutorial', 'guide', 'example', 'walkthrough'],
            'error_help': ['error', 'exception', 'bug', 'fix', 'problem', 'issue', 'troubleshoot'],
            'animation': ['animate', 'animation', 'move', 'transform', 'transition', 'motion'],
            'geometry': ['circle', 'square', 'line', 'polygon', 'shape', 'geometry', 'mobject'],
            'text': ['text', 'label', 'title', 'font', 'string', 'typography'],
            'math': ['equation', 'formula', 'math', 'latex', 'symbol', 'mathematical'],
            'scene': ['scene', 'camera', 'background', 'setup', 'render'],
            'plugin': ['plugin', 'extension', 'addon', 'community', 'third-party']
        }
    
    def _save_evaluation_result(self, result: EvaluationResult) -> None:
        """Save evaluation result to storage"""
        filename = f"evaluation_{result.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.storage_path / filename
        
        # Convert to serializable format
        serializable_result = {
            'timestamp': result.timestamp.isoformat(),
            'retrieval_metrics': asdict(result.retrieval_metrics),
            'code_metrics': asdict(result.code_metrics),
            'query_metrics': asdict(result.query_metrics),
            'sample_count': result.sample_count,
            'evaluation_duration': result.evaluation_duration
        }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_result, f, indent=2)
    
    def _save_query_logs(self) -> None:
        """Save query logs to storage"""
        logs_file = self.storage_path / "query_logs.json"
        
        # Convert logs to serializable format
        serializable_logs = []
        for log in self.query_logs[-1000:]:  # Keep last 1000 logs
            log_dict = {
                'query_id': log.query_id,
                'query': log.query,
                'timestamp': log.timestamp.isoformat(),
                'response_time': log.response_time,
                'success': log.success,
                'cache_hit': log.cache_hit,
                'metadata': log.metadata
            }
            serializable_logs.append(log_dict)
        
        with open(logs_file, 'w') as f:
            json.dump(serializable_logs, f, indent=2)
    
    def _save_baseline_metrics(self) -> None:
        """Save baseline metrics to storage"""
        if self.baseline_metrics:
            baseline_file = self.storage_path / "baseline_metrics.json"
            with open(baseline_file, 'w') as f:
                json.dump(asdict(self.baseline_metrics), f, indent=2)
    
    def _load_evaluation_history(self) -> None:
        """Load existing evaluation history from storage"""
        try:
            # Load baseline metrics
            baseline_file = self.storage_path / "baseline_metrics.json"
            if baseline_file.exists():
                with open(baseline_file, 'r') as f:
                    baseline_data = json.load(f)
                    self.baseline_metrics = EvaluationMetrics(**baseline_data)
            
            # Load query logs
            logs_file = self.storage_path / "query_logs.json"
            if logs_file.exists():
                with open(logs_file, 'r') as f:
                    logs_data = json.load(f)
                    for log_dict in logs_data:
                        # Reconstruct QueryLog (simplified version)
                        query_log = QueryLog(
                            query_id=log_dict['query_id'],
                            query=log_dict['query'],
                            timestamp=datetime.fromisoformat(log_dict['timestamp']),
                            response_time=log_dict['response_time'],
                            results=[],  # Not stored in simplified format
                            generated_code=None,
                            success=log_dict['success'],
                            error_message=None,
                            cache_hit=log_dict['cache_hit'],
                            metadata=log_dict['metadata']
                        )
                        self.query_logs.append(query_log)
        
        except Exception as e:
            self.logger.error(f"Error loading evaluation history: {e}")
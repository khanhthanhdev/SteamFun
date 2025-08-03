"""
Evaluation Data Collection and Feedback System

This module provides user feedback collection, automated evaluation dataset creation,
and human evaluation interface for quality assessment.
"""

import json
import uuid
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable, Set, Tuple
from enum import Enum
from pathlib import Path
import sqlite3
import threading
from collections import defaultdict

from .result_types import RankedResult, ResultType
from .quality_evaluator import EvaluationMetrics


class FeedbackType(Enum):
    """Types of user feedback"""
    RELEVANCE_RATING = "relevance_rating"
    CODE_QUALITY_RATING = "code_quality_rating"
    RESULT_USEFULNESS = "result_usefulness"
    QUERY_SATISFACTION = "query_satisfaction"
    ERROR_REPORT = "error_report"
    SUGGESTION = "suggestion"


class FeedbackRating(Enum):
    """Feedback rating scale"""
    VERY_POOR = 1
    POOR = 2
    FAIR = 3
    GOOD = 4
    EXCELLENT = 5


@dataclass
class UserFeedback:
    """User feedback data structure"""
    feedback_id: str
    user_id: str
    session_id: str
    timestamp: datetime
    feedback_type: FeedbackType
    rating: Optional[FeedbackRating]
    query: str
    results: List[str]  # Result IDs
    generated_code: Optional[str]
    comments: str
    metadata: Dict[str, Any]


@dataclass
class EvaluationDataPoint:
    """Single evaluation data point for dataset"""
    query: str
    relevant_results: List[str]
    irrelevant_results: List[str]
    generated_code: Optional[str]
    code_success: bool
    user_satisfaction: float
    timestamp: datetime
    context: Dict[str, Any]


@dataclass
class HumanEvaluationTask:
    """Human evaluation task"""
    task_id: str
    evaluator_id: str
    query: str
    results: List[RankedResult]
    evaluation_criteria: List[str]
    status: str  # pending, in_progress, completed
    created_at: datetime
    completed_at: Optional[datetime]
    evaluation_results: Dict[str, Any]


class FeedbackCollector:
    """
    User feedback collection system
    
    Collects user feedback on retrieval quality, code generation success,
    and overall system satisfaction.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize feedback collector
        
        Args:
            storage_path: Path to store feedback data
        """
        self.storage_path = Path(storage_path) if storage_path else Path("rag_feedback")
        self.storage_path.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize SQLite database for feedback storage
        self.db_path = self.storage_path / "feedback.db"
        self._init_database()
        
        # Feedback processing
        self.feedback_processors: List[Callable[[UserFeedback], None]] = []
        self.feedback_cache: List[UserFeedback] = []
        
        # Thread safety
        self._lock = threading.Lock()
    
    def collect_relevance_feedback(
        self,
        user_id: str,
        session_id: str,
        query: str,
        results: List[RankedResult],
        relevance_ratings: List[FeedbackRating],
        comments: str = ""
    ) -> str:
        """
        Collect relevance feedback for search results
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            query: Original query
            results: Retrieved results
            relevance_ratings: Relevance rating for each result
            comments: Optional user comments
            
        Returns:
            Feedback ID
        """
        if len(results) != len(relevance_ratings):
            raise ValueError("Results and ratings must have same length")
        
        feedback = UserFeedback(
            feedback_id=str(uuid.uuid4()),
            user_id=user_id,
            session_id=session_id,
            timestamp=datetime.now(),
            feedback_type=FeedbackType.RELEVANCE_RATING,
            rating=None,  # Individual ratings in metadata
            query=query,
            results=[result.chunk_id for result in results],
            generated_code=None,
            comments=comments,
            metadata={
                'relevance_ratings': [rating.value for rating in relevance_ratings],
                'result_types': [result.result_type.value for result in results],
                'similarity_scores': [result.similarity_score for result in results],
                'context_scores': [result.context_score for result in results]
            }
        )
        
        self._store_feedback(feedback)
        return feedback.feedback_id
    
    def collect_code_quality_feedback(
        self,
        user_id: str,
        session_id: str,
        query: str,
        results: List[RankedResult],
        generated_code: str,
        code_rating: FeedbackRating,
        compilation_success: bool,
        functionality_rating: FeedbackRating,
        comments: str = ""
    ) -> str:
        """
        Collect feedback on generated code quality
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            query: Original query
            results: Retrieved results used for generation
            generated_code: Generated code
            code_rating: Overall code quality rating
            compilation_success: Whether code compiled successfully
            functionality_rating: How well code meets requirements
            comments: Optional user comments
            
        Returns:
            Feedback ID
        """
        feedback = UserFeedback(
            feedback_id=str(uuid.uuid4()),
            user_id=user_id,
            session_id=session_id,
            timestamp=datetime.now(),
            feedback_type=FeedbackType.CODE_QUALITY_RATING,
            rating=code_rating,
            query=query,
            results=[result.chunk_id for result in results],
            generated_code=generated_code,
            comments=comments,
            metadata={
                'compilation_success': compilation_success,
                'functionality_rating': functionality_rating.value,
                'code_length': len(generated_code),
                'result_count': len(results)
            }
        )
        
        self._store_feedback(feedback)
        return feedback.feedback_id
    
    def collect_query_satisfaction_feedback(
        self,
        user_id: str,
        session_id: str,
        query: str,
        satisfaction_rating: FeedbackRating,
        found_what_needed: bool,
        would_recommend: bool,
        comments: str = ""
    ) -> str:
        """
        Collect overall query satisfaction feedback
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            query: Original query
            satisfaction_rating: Overall satisfaction rating
            found_what_needed: Whether user found what they needed
            would_recommend: Whether user would recommend the system
            comments: Optional user comments
            
        Returns:
            Feedback ID
        """
        feedback = UserFeedback(
            feedback_id=str(uuid.uuid4()),
            user_id=user_id,
            session_id=session_id,
            timestamp=datetime.now(),
            feedback_type=FeedbackType.QUERY_SATISFACTION,
            rating=satisfaction_rating,
            query=query,
            results=[],
            generated_code=None,
            comments=comments,
            metadata={
                'found_what_needed': found_what_needed,
                'would_recommend': would_recommend
            }
        )
        
        self._store_feedback(feedback)
        return feedback.feedback_id
    
    def report_error(
        self,
        user_id: str,
        session_id: str,
        query: str,
        error_description: str,
        error_type: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Report system error or issue
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            query: Query that caused error
            error_description: Description of the error
            error_type: Type/category of error
            context: Additional context information
            
        Returns:
            Feedback ID
        """
        feedback = UserFeedback(
            feedback_id=str(uuid.uuid4()),
            user_id=user_id,
            session_id=session_id,
            timestamp=datetime.now(),
            feedback_type=FeedbackType.ERROR_REPORT,
            rating=None,
            query=query,
            results=[],
            generated_code=None,
            comments=error_description,
            metadata={
                'error_type': error_type,
                'context': context
            }
        )
        
        self._store_feedback(feedback)
        return feedback.feedback_id
    
    def get_feedback_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Get summary of feedback over specified time period
        
        Args:
            days: Number of days to include in summary
            
        Returns:
            Feedback summary statistics
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get feedback counts by type
            cursor.execute("""
                SELECT feedback_type, COUNT(*) 
                FROM feedback 
                WHERE timestamp > ? 
                GROUP BY feedback_type
            """, (cutoff_date.isoformat(),))
            
            feedback_counts = dict(cursor.fetchall())
            
            # Get average ratings by type
            cursor.execute("""
                SELECT feedback_type, AVG(CAST(rating AS FLOAT))
                FROM feedback 
                WHERE timestamp > ? AND rating IS NOT NULL
                GROUP BY feedback_type
            """, (cutoff_date.isoformat(),))
            
            avg_ratings = dict(cursor.fetchall())
            
            # Get recent feedback
            cursor.execute("""
                SELECT * FROM feedback 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC 
                LIMIT 10
            """, (cutoff_date.isoformat(),))
            
            recent_feedback = cursor.fetchall()
        
        return {
            'period_days': days,
            'total_feedback': sum(feedback_counts.values()),
            'feedback_by_type': feedback_counts,
            'average_ratings': avg_ratings,
            'recent_feedback_count': len(recent_feedback),
            'summary_generated': datetime.now().isoformat()
        }
    
    def add_feedback_processor(self, processor: Callable[[UserFeedback], None]) -> None:
        """Add feedback processor function"""
        self.feedback_processors.append(processor)
    
    def _store_feedback(self, feedback: UserFeedback) -> None:
        """Store feedback in database and process"""
        with self._lock:
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO feedback (
                        feedback_id, user_id, session_id, timestamp, feedback_type,
                        rating, query, results, generated_code, comments, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feedback.feedback_id,
                    feedback.user_id,
                    feedback.session_id,
                    feedback.timestamp.isoformat(),
                    feedback.feedback_type.value,
                    feedback.rating.value if feedback.rating else None,
                    feedback.query,
                    json.dumps(feedback.results),
                    feedback.generated_code,
                    feedback.comments,
                    json.dumps(feedback.metadata)
                ))
                conn.commit()
            
            # Add to cache
            self.feedback_cache.append(feedback)
            
            # Process feedback
            for processor in self.feedback_processors:
                try:
                    processor(feedback)
                except Exception as e:
                    self.logger.error(f"Error in feedback processor: {e}")
    
    def _init_database(self) -> None:
        """Initialize SQLite database for feedback storage"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    rating INTEGER,
                    query TEXT NOT NULL,
                    results TEXT,
                    generated_code TEXT,
                    comments TEXT,
                    metadata TEXT
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_timestamp 
                ON feedback(timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_type 
                ON feedback(feedback_type)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_user 
                ON feedback(user_id)
            """)
            
            conn.commit()


class EvaluationDatasetBuilder:
    """
    Automated evaluation dataset builder
    
    Creates evaluation datasets from successful code generations and user feedback.
    """
    
    def __init__(self, feedback_collector: FeedbackCollector, storage_path: Optional[str] = None):
        """
        Initialize dataset builder
        
        Args:
            feedback_collector: FeedbackCollector instance
            storage_path: Path to store dataset files
        """
        self.feedback_collector = feedback_collector
        self.storage_path = Path(storage_path) if storage_path else Path("rag_datasets")
        self.storage_path.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Dataset building parameters
        self.min_rating_threshold = FeedbackRating.GOOD.value
        self.min_feedback_count = 3
    
    def build_relevance_dataset(self, min_samples: int = 100) -> List[EvaluationDataPoint]:
        """
        Build relevance evaluation dataset from user feedback
        
        Args:
            min_samples: Minimum number of samples to include
            
        Returns:
            List of evaluation data points
        """
        dataset = []
        
        with sqlite3.connect(self.feedback_collector.db_path) as conn:
            cursor = conn.cursor()
            
            # Get relevance feedback (rating is stored in metadata for relevance feedback)
            cursor.execute("""
                SELECT query, results, metadata, timestamp
                FROM feedback 
                WHERE feedback_type = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (
                FeedbackType.RELEVANCE_RATING.value,
                min_samples * 2  # Get more to filter
            ))
            
            feedback_data = cursor.fetchall()
        
        for query, results_json, metadata_json, timestamp in feedback_data:
            try:
                results = json.loads(results_json)
                metadata = json.loads(metadata_json)
                relevance_ratings = metadata.get('relevance_ratings', [])
                
                if len(results) != len(relevance_ratings):
                    continue
                
                # Separate relevant and irrelevant results
                relevant_results = []
                irrelevant_results = []
                
                for result_id, rating in zip(results, relevance_ratings):
                    if rating >= self.min_rating_threshold:
                        relevant_results.append(result_id)
                    else:
                        irrelevant_results.append(result_id)
                
                if relevant_results:  # Only include if there are relevant results
                    data_point = EvaluationDataPoint(
                        query=query,
                        relevant_results=relevant_results,
                        irrelevant_results=irrelevant_results,
                        generated_code=None,
                        code_success=True,  # Inferred from high rating
                        user_satisfaction=sum(relevance_ratings) / len(relevance_ratings) / 5.0,
                        timestamp=datetime.fromisoformat(timestamp),
                        context={'source': 'user_feedback', 'metadata': metadata}
                    )
                    dataset.append(data_point)
                    
                    if len(dataset) >= min_samples:
                        break
            
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.warning(f"Error processing feedback data: {e}")
                continue
        
        self.logger.info(f"Built relevance dataset with {len(dataset)} samples")
        return dataset
    
    def build_code_generation_dataset(self, min_samples: int = 50) -> List[EvaluationDataPoint]:
        """
        Build code generation evaluation dataset
        
        Args:
            min_samples: Minimum number of samples to include
            
        Returns:
            List of evaluation data points for code generation
        """
        dataset = []
        
        with sqlite3.connect(self.feedback_collector.db_path) as conn:
            cursor = conn.cursor()
            
            # Get code quality feedback with successful generations
            cursor.execute("""
                SELECT query, results, generated_code, rating, metadata, timestamp
                FROM feedback 
                WHERE feedback_type = ? AND rating >= ? AND generated_code IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT ?
            """, (
                FeedbackType.CODE_QUALITY_RATING.value,
                self.min_rating_threshold,
                min_samples * 2
            ))
            
            feedback_data = cursor.fetchall()
        
        for query, results_json, generated_code, rating, metadata_json, timestamp in feedback_data:
            try:
                results = json.loads(results_json)
                metadata = json.loads(metadata_json)
                
                compilation_success = metadata.get('compilation_success', False)
                functionality_rating = metadata.get('functionality_rating', 3)
                
                data_point = EvaluationDataPoint(
                    query=query,
                    relevant_results=results,  # All results used for generation
                    irrelevant_results=[],
                    generated_code=generated_code,
                    code_success=compilation_success and functionality_rating >= self.min_rating_threshold,
                    user_satisfaction=rating / 5.0,
                    timestamp=datetime.fromisoformat(timestamp),
                    context={
                        'source': 'code_generation_feedback',
                        'compilation_success': compilation_success,
                        'functionality_rating': functionality_rating,
                        'metadata': metadata
                    }
                )
                dataset.append(data_point)
                
                if len(dataset) >= min_samples:
                    break
            
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.warning(f"Error processing code generation data: {e}")
                continue
        
        self.logger.info(f"Built code generation dataset with {len(dataset)} samples")
        return dataset
    
    def save_dataset(self, dataset: List[EvaluationDataPoint], filename: str) -> str:
        """
        Save dataset to file
        
        Args:
            dataset: Dataset to save
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.storage_path / filename
        
        # Convert to serializable format
        serializable_data = []
        for data_point in dataset:
            serializable_data.append({
                'query': data_point.query,
                'relevant_results': data_point.relevant_results,
                'irrelevant_results': data_point.irrelevant_results,
                'generated_code': data_point.generated_code,
                'code_success': data_point.code_success,
                'user_satisfaction': data_point.user_satisfaction,
                'timestamp': data_point.timestamp.isoformat(),
                'context': data_point.context
            })
        
        with open(output_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        self.logger.info(f"Saved dataset with {len(dataset)} samples to {output_path}")
        return str(output_path)
    
    def load_dataset(self, filename: str) -> List[EvaluationDataPoint]:
        """
        Load dataset from file
        
        Args:
            filename: Input filename
            
        Returns:
            Loaded dataset
        """
        input_path = self.storage_path / filename
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        dataset = []
        for item in data:
            data_point = EvaluationDataPoint(
                query=item['query'],
                relevant_results=item['relevant_results'],
                irrelevant_results=item['irrelevant_results'],
                generated_code=item['generated_code'],
                code_success=item['code_success'],
                user_satisfaction=item['user_satisfaction'],
                timestamp=datetime.fromisoformat(item['timestamp']),
                context=item['context']
            )
            dataset.append(data_point)
        
        self.logger.info(f"Loaded dataset with {len(dataset)} samples from {input_path}")
        return dataset


class HumanEvaluationInterface:
    """
    Human evaluation interface for quality assessment
    
    Provides interface for human evaluators to assess retrieval quality
    and code generation effectiveness.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize human evaluation interface
        
        Args:
            storage_path: Path to store evaluation tasks and results
        """
        self.storage_path = Path(storage_path) if storage_path else Path("human_evaluation")
        self.storage_path.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize database for evaluation tasks
        self.db_path = self.storage_path / "human_evaluation.db"
        self._init_database()
        
        # Evaluation criteria
        self.default_criteria = [
            "Relevance to query",
            "Completeness of information",
            "Code quality and correctness",
            "Clarity and understandability",
            "Practical usefulness"
        ]
    
    def create_evaluation_task(
        self,
        evaluator_id: str,
        query: str,
        results: List[RankedResult],
        criteria: Optional[List[str]] = None
    ) -> str:
        """
        Create human evaluation task
        
        Args:
            evaluator_id: ID of human evaluator
            query: Query to evaluate
            results: Results to evaluate
            criteria: Evaluation criteria (uses default if None)
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        criteria = criteria or self.default_criteria
        
        task = HumanEvaluationTask(
            task_id=task_id,
            evaluator_id=evaluator_id,
            query=query,
            results=results,
            evaluation_criteria=criteria,
            status="pending",
            created_at=datetime.now(),
            completed_at=None,
            evaluation_results={}
        )
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Convert results to serializable format
            serializable_results = []
            for result in task.results:
                result_dict = asdict(result)
                result_dict['result_type'] = result.result_type.value  # Convert enum to string
                serializable_results.append(result_dict)
            
            cursor.execute("""
                INSERT INTO evaluation_tasks (
                    task_id, evaluator_id, query, results, criteria, 
                    status, created_at, evaluation_results
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task.task_id,
                task.evaluator_id,
                task.query,
                json.dumps(serializable_results),
                json.dumps(task.evaluation_criteria),
                task.status,
                task.created_at.isoformat(),
                json.dumps(task.evaluation_results)
            ))
            conn.commit()
        
        self.logger.info(f"Created evaluation task {task_id} for evaluator {evaluator_id}")
        return task_id
    
    def submit_evaluation(
        self,
        task_id: str,
        evaluator_id: str,
        evaluation_results: Dict[str, Any]
    ) -> bool:
        """
        Submit evaluation results
        
        Args:
            task_id: Task ID
            evaluator_id: Evaluator ID
            evaluation_results: Evaluation results
            
        Returns:
            True if submission successful
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Verify task exists and belongs to evaluator
            cursor.execute("""
                SELECT status FROM evaluation_tasks 
                WHERE task_id = ? AND evaluator_id = ?
            """, (task_id, evaluator_id))
            
            result = cursor.fetchone()
            if not result:
                self.logger.error(f"Task {task_id} not found for evaluator {evaluator_id}")
                return False
            
            if result[0] == "completed":
                self.logger.warning(f"Task {task_id} already completed")
                return False
            
            # Update task with results
            cursor.execute("""
                UPDATE evaluation_tasks 
                SET status = ?, completed_at = ?, evaluation_results = ?
                WHERE task_id = ?
            """, (
                "completed",
                datetime.now().isoformat(),
                json.dumps(evaluation_results),
                task_id
            ))
            conn.commit()
        
        self.logger.info(f"Evaluation task {task_id} completed by {evaluator_id}")
        return True
    
    def get_pending_tasks(self, evaluator_id: str) -> List[HumanEvaluationTask]:
        """
        Get pending evaluation tasks for evaluator
        
        Args:
            evaluator_id: Evaluator ID
            
        Returns:
            List of pending tasks
        """
        tasks = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM evaluation_tasks 
                WHERE evaluator_id = ? AND status = 'pending'
                ORDER BY created_at
            """, (evaluator_id,))
            
            for row in cursor.fetchall():
                task = self._row_to_task(row)
                tasks.append(task)
        
        return tasks
    
    def get_evaluation_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Get summary of human evaluations
        
        Args:
            days: Number of days to include
            
        Returns:
            Evaluation summary
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get task counts by status
            cursor.execute("""
                SELECT status, COUNT(*) 
                FROM evaluation_tasks 
                WHERE created_at > ?
                GROUP BY status
            """, (cutoff_date.isoformat(),))
            
            status_counts = dict(cursor.fetchall())
            
            # Get completed evaluations
            cursor.execute("""
                SELECT evaluation_results 
                FROM evaluation_tasks 
                WHERE status = 'completed' AND created_at > ?
            """, (cutoff_date.isoformat(),))
            
            completed_evaluations = []
            for (results_json,) in cursor.fetchall():
                try:
                    results = json.loads(results_json)
                    completed_evaluations.append(results)
                except json.JSONDecodeError:
                    continue
        
        return {
            'period_days': days,
            'task_counts': status_counts,
            'completed_evaluations': len(completed_evaluations),
            'pending_tasks': status_counts.get('pending', 0),
            'summary_generated': datetime.now().isoformat()
        }
    
    def _init_database(self) -> None:
        """Initialize database for human evaluation tasks"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_tasks (
                    task_id TEXT PRIMARY KEY,
                    evaluator_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    results TEXT NOT NULL,
                    criteria TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    evaluation_results TEXT
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_evaluator_status 
                ON evaluation_tasks(evaluator_id, status)
            """)
            
            conn.commit()
    
    def _row_to_task(self, row: Tuple) -> HumanEvaluationTask:
        """Convert database row to HumanEvaluationTask"""
        (task_id, evaluator_id, query, results_json, criteria_json, 
         status, created_at, completed_at, evaluation_results_json) = row
        
        # Deserialize results and convert result_type back to enum
        results_data = json.loads(results_json)
        results = []
        for result_data in results_data:
            # Convert result_type string back to enum
            result_data['result_type'] = ResultType(result_data['result_type'])
            results.append(RankedResult(**result_data))
        
        criteria = json.loads(criteria_json)
        evaluation_results = json.loads(evaluation_results_json) if evaluation_results_json else {}
        
        return HumanEvaluationTask(
            task_id=task_id,
            evaluator_id=evaluator_id,
            query=query,
            results=results,
            evaluation_criteria=criteria,
            status=status,
            created_at=datetime.fromisoformat(created_at),
            completed_at=datetime.fromisoformat(completed_at) if completed_at else None,
            evaluation_results=evaluation_results
        )
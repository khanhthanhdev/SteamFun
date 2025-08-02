"""
Quality Monitoring and Alerting System

This module provides quality degradation detection, performance dashboard,
and backward compatibility monitoring for the RAG system.
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable, Set
from enum import Enum
import threading
from pathlib import Path

from .quality_evaluator import (
    RAGQualityEvaluator, 
    EvaluationMetrics, 
    EvaluationResult,
    CodeQualityMetrics,
    QueryPatternMetrics
)


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of quality alerts"""
    PRECISION_DEGRADATION = "precision_degradation"
    RECALL_DEGRADATION = "recall_degradation"
    RESPONSE_TIME_INCREASE = "response_time_increase"
    CODE_GENERATION_FAILURE = "code_generation_failure"
    CACHE_MISS_INCREASE = "cache_miss_increase"
    BACKWARD_COMPATIBILITY = "backward_compatibility"
    SYSTEM_ERROR = "system_error"


@dataclass
class QualityAlert:
    """Quality alert data structure"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: datetime
    current_value: float
    baseline_value: float
    threshold: float
    metadata: Dict[str, Any]
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None


@dataclass
class QualityThresholds:
    """Quality monitoring thresholds"""
    precision_degradation_threshold: float = 0.1  # 10% degradation
    recall_degradation_threshold: float = 0.1
    ndcg_degradation_threshold: float = 0.1
    response_time_threshold: float = 2.0  # seconds
    response_time_increase_threshold: float = 0.5  # 50% increase
    code_generation_failure_threshold: float = 0.2  # 20% failure rate
    cache_miss_threshold: float = 0.5  # 50% cache miss rate
    backward_compatibility_threshold: float = 0.05  # 5% degradation


@dataclass
class PerformanceDashboardData:
    """Data structure for performance dashboard"""
    timestamp: datetime
    retrieval_metrics: EvaluationMetrics
    code_metrics: CodeQualityMetrics
    query_metrics: QueryPatternMetrics
    system_health: Dict[str, Any]
    active_alerts: List[QualityAlert]
    historical_trends: Dict[str, List[float]]


class QualityMonitor:
    """
    Quality monitoring and alerting system for RAG
    
    Provides quality degradation detection, performance dashboard,
    and backward compatibility monitoring.
    """
    
    def __init__(
        self, 
        evaluator: RAGQualityEvaluator,
        thresholds: Optional[QualityThresholds] = None,
        storage_path: Optional[str] = None
    ):
        """
        Initialize quality monitor
        
        Args:
            evaluator: RAGQualityEvaluator instance
            thresholds: Quality thresholds for alerting
            storage_path: Path to store monitoring data
        """
        self.evaluator = evaluator
        self.thresholds = thresholds or QualityThresholds()
        self.storage_path = Path(storage_path) if storage_path else Path("rag_monitoring")
        self.storage_path.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Alert management
        self.active_alerts: List[QualityAlert] = []
        self.alert_handlers: List[Callable[[QualityAlert], None]] = []
        self.alert_counter = 0
        
        # Monitoring data
        self.baseline_metrics: Optional[EvaluationMetrics] = None
        self.monitoring_history: List[PerformanceDashboardData] = []
        self.backward_compatibility_patterns: Set[str] = set()
        
        # Background monitoring
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_interval = 300  # 5 minutes
        
        # Load existing data
        self._load_monitoring_data()
    
    def set_baseline_metrics(self, metrics: EvaluationMetrics) -> None:
        """Set baseline metrics for comparison"""
        self.baseline_metrics = metrics
        self.evaluator.set_baseline_metrics(metrics)
        self._save_baseline_metrics()
    
    def detect_quality_degradation(
        self, 
        current_metrics: EvaluationMetrics
    ) -> List[QualityAlert]:
        """
        Detect quality degradation by comparing current vs baseline metrics
        
        Args:
            current_metrics: Current evaluation metrics
            
        Returns:
            List of quality alerts if degradation detected
        """
        if not self.baseline_metrics:
            self.logger.warning("No baseline metrics set for quality degradation detection")
            return []
        
        alerts = []
        
        # Check precision degradation
        precision_diff = self.baseline_metrics.precision - current_metrics.precision
        if precision_diff > self.thresholds.precision_degradation_threshold:
            alert = self._create_alert(
                AlertType.PRECISION_DEGRADATION,
                AlertSeverity.HIGH if precision_diff > 0.2 else AlertSeverity.MEDIUM,
                f"Precision degraded by {precision_diff:.3f} ({precision_diff/self.baseline_metrics.precision*100:.1f}%)",
                current_metrics.precision,
                self.baseline_metrics.precision,
                self.thresholds.precision_degradation_threshold
            )
            alerts.append(alert)
        
        # Check recall degradation
        recall_diff = self.baseline_metrics.recall - current_metrics.recall
        if recall_diff > self.thresholds.recall_degradation_threshold:
            alert = self._create_alert(
                AlertType.RECALL_DEGRADATION,
                AlertSeverity.HIGH if recall_diff > 0.2 else AlertSeverity.MEDIUM,
                f"Recall degraded by {recall_diff:.3f} ({recall_diff/self.baseline_metrics.recall*100:.1f}%)",
                current_metrics.recall,
                self.baseline_metrics.recall,
                self.thresholds.recall_degradation_threshold
            )
            alerts.append(alert)
        
        # Check NDCG degradation
        ndcg_diff = self.baseline_metrics.ndcg - current_metrics.ndcg
        if ndcg_diff > self.thresholds.ndcg_degradation_threshold:
            alert = self._create_alert(
                AlertType.PRECISION_DEGRADATION,  # Using precision type for NDCG
                AlertSeverity.MEDIUM,
                f"NDCG degraded by {ndcg_diff:.3f} ({ndcg_diff/self.baseline_metrics.ndcg*100:.1f}%)",
                current_metrics.ndcg,
                self.baseline_metrics.ndcg,
                self.thresholds.ndcg_degradation_threshold
            )
            alerts.append(alert)
        
        # Add alerts to active list and trigger handlers
        for alert in alerts:
            self.active_alerts.append(alert)
            self._trigger_alert_handlers(alert)
        
        return alerts
    
    def monitor_response_time(self, response_time: float, query_type: str = "general") -> Optional[QualityAlert]:
        """
        Monitor response time and alert if threshold exceeded
        
        Args:
            response_time: Response time in seconds
            query_type: Type of query for context
            
        Returns:
            Alert if response time threshold exceeded
        """
        alert = None
        
        if response_time > self.thresholds.response_time_threshold:
            alert = self._create_alert(
                AlertType.RESPONSE_TIME_INCREASE,
                AlertSeverity.HIGH if response_time > 5.0 else AlertSeverity.MEDIUM,
                f"Response time {response_time:.2f}s exceeds threshold {self.thresholds.response_time_threshold:.2f}s for {query_type} query",
                response_time,
                self.thresholds.response_time_threshold,
                self.thresholds.response_time_threshold,
                {"query_type": query_type}
            )
            self.active_alerts.append(alert)
            self._trigger_alert_handlers(alert)
        
        return alert
    
    def monitor_code_generation(self, success_rate: float, compilation_rate: float) -> List[QualityAlert]:
        """
        Monitor code generation quality and alert on failures
        
        Args:
            success_rate: Code generation success rate
            compilation_rate: Code compilation success rate
            
        Returns:
            List of alerts if thresholds exceeded
        """
        alerts = []
        
        failure_rate = 1.0 - success_rate
        if failure_rate > self.thresholds.code_generation_failure_threshold:
            alert = self._create_alert(
                AlertType.CODE_GENERATION_FAILURE,
                AlertSeverity.CRITICAL if failure_rate > 0.5 else AlertSeverity.HIGH,
                f"Code generation failure rate {failure_rate:.1%} exceeds threshold {self.thresholds.code_generation_failure_threshold:.1%}",
                failure_rate,
                0.0,
                self.thresholds.code_generation_failure_threshold,
                {"success_rate": success_rate, "compilation_rate": compilation_rate}
            )
            alerts.append(alert)
        
        compilation_failure_rate = 1.0 - compilation_rate
        if compilation_failure_rate > self.thresholds.code_generation_failure_threshold:
            alert = self._create_alert(
                AlertType.CODE_GENERATION_FAILURE,
                AlertSeverity.HIGH,
                f"Code compilation failure rate {compilation_failure_rate:.1%} exceeds threshold {self.thresholds.code_generation_failure_threshold:.1%}",
                compilation_failure_rate,
                0.0,
                self.thresholds.code_generation_failure_threshold,
                {"type": "compilation", "compilation_rate": compilation_rate}
            )
            alerts.append(alert)
        
        for alert in alerts:
            self.active_alerts.append(alert)
            self._trigger_alert_handlers(alert)
        
        return alerts
    
    def monitor_backward_compatibility(self, query_patterns: Dict[str, float]) -> List[QualityAlert]:
        """
        Monitor backward compatibility for existing query patterns
        
        Args:
            query_patterns: Dictionary of query patterns and their success rates
            
        Returns:
            List of backward compatibility alerts
        """
        alerts = []
        
        for pattern, success_rate in query_patterns.items():
            if pattern in self.backward_compatibility_patterns:
                # This is a known pattern, check for degradation
                if success_rate < (1.0 - self.thresholds.backward_compatibility_threshold):
                    alert = self._create_alert(
                        AlertType.BACKWARD_COMPATIBILITY,
                        AlertSeverity.HIGH,
                        f"Backward compatibility issue: query pattern '{pattern}' success rate dropped to {success_rate:.1%}",
                        success_rate,
                        1.0,
                        1.0 - self.thresholds.backward_compatibility_threshold,
                        {"pattern": pattern}
                    )
                    alerts.append(alert)
            else:
                # New pattern, add to tracking
                self.backward_compatibility_patterns.add(pattern)
        
        for alert in alerts:
            self.active_alerts.append(alert)
            self._trigger_alert_handlers(alert)
        
        return alerts
    
    def generate_performance_dashboard(self) -> PerformanceDashboardData:
        """
        Generate comprehensive performance dashboard data
        
        Returns:
            PerformanceDashboardData with current system status
        """
        # Get current metrics from evaluator
        query_metrics = self.evaluator.analyze_query_patterns()
        
        # Create dummy metrics if no recent evaluation
        if not self.evaluator.evaluation_history:
            retrieval_metrics = EvaluationMetrics(0.0, 0.0, 0.0)
            code_metrics = CodeQualityMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
        else:
            latest_eval = self.evaluator.evaluation_history[-1]
            retrieval_metrics = latest_eval.retrieval_metrics
            code_metrics = latest_eval.code_metrics
        
        # System health indicators
        system_health = {
            "active_alerts_count": len([a for a in self.active_alerts if not a.resolved]),
            "critical_alerts_count": len([a for a in self.active_alerts if not a.resolved and a.severity == AlertSeverity.CRITICAL]),
            "monitoring_active": self.monitoring_active,
            "baseline_set": self.baseline_metrics is not None,
            "evaluation_history_size": len(self.evaluator.evaluation_history),
            "query_logs_size": len(self.evaluator.query_logs)
        }
        
        # Historical trends (last 10 evaluations)
        historical_trends = {}
        if len(self.evaluator.evaluation_history) > 1:
            recent_evals = self.evaluator.evaluation_history[-10:]
            historical_trends = {
                "precision": [e.retrieval_metrics.precision for e in recent_evals],
                "recall": [e.retrieval_metrics.recall for e in recent_evals],
                "ndcg": [e.retrieval_metrics.ndcg for e in recent_evals],
                "response_time": [e.query_metrics.average_response_time for e in recent_evals],
                "cache_hit_rate": [e.query_metrics.cache_hit_rate for e in recent_evals]
            }
        
        dashboard_data = PerformanceDashboardData(
            timestamp=datetime.now(),
            retrieval_metrics=retrieval_metrics,
            code_metrics=code_metrics,
            query_metrics=query_metrics,
            system_health=system_health,
            active_alerts=[a for a in self.active_alerts if not a.resolved],
            historical_trends=historical_trends
        )
        
        self.monitoring_history.append(dashboard_data)
        self._save_dashboard_data(dashboard_data)
        
        return dashboard_data
    
    def add_alert_handler(self, handler: Callable[[QualityAlert], None]) -> None:
        """Add alert handler function"""
        self.alert_handlers.append(handler)
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an active alert
        
        Args:
            alert_id: ID of alert to resolve
            
        Returns:
            True if alert was found and resolved
        """
        for alert in self.active_alerts:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolution_timestamp = datetime.now()
                self.logger.info(f"Alert {alert_id} resolved")
                return True
        return False
    
    def start_background_monitoring(self, interval_seconds: int = 300) -> None:
        """
        Start background monitoring thread
        
        Args:
            interval_seconds: Monitoring interval in seconds
        """
        if self.monitoring_active:
            self.logger.warning("Background monitoring already active")
            return
        
        self.monitoring_interval = interval_seconds
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._background_monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        self.logger.info(f"Started background monitoring with {interval_seconds}s interval")
    
    def stop_background_monitoring(self) -> None:
        """Stop background monitoring thread"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Stopped background monitoring")
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of current alerts"""
        active_alerts = [a for a in self.active_alerts if not a.resolved]
        
        return {
            "total_alerts": len(active_alerts),
            "critical_alerts": len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
            "high_alerts": len([a for a in active_alerts if a.severity == AlertSeverity.HIGH]),
            "medium_alerts": len([a for a in active_alerts if a.severity == AlertSeverity.MEDIUM]),
            "low_alerts": len([a for a in active_alerts if a.severity == AlertSeverity.LOW]),
            "alert_types": list(set(a.alert_type.value for a in active_alerts)),
            "oldest_alert": min((a.timestamp for a in active_alerts), default=None),
            "newest_alert": max((a.timestamp for a in active_alerts), default=None)
        }
    
    # Private helper methods
    
    def _create_alert(
        self, 
        alert_type: AlertType, 
        severity: AlertSeverity, 
        message: str,
        current_value: float,
        baseline_value: float,
        threshold: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> QualityAlert:
        """Create a new quality alert"""
        self.alert_counter += 1
        alert_id = f"alert_{self.alert_counter}_{int(time.time())}"
        
        return QualityAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            current_value=current_value,
            baseline_value=baseline_value,
            threshold=threshold,
            metadata=metadata or {}
        )
    
    def _trigger_alert_handlers(self, alert: QualityAlert) -> None:
        """Trigger all registered alert handlers"""
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {e}")
    
    def _background_monitoring_loop(self) -> None:
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Generate dashboard data (includes quality checks)
                self.generate_performance_dashboard()
                
                # Sleep for monitoring interval
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Error in background monitoring: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _save_baseline_metrics(self) -> None:
        """Save baseline metrics to storage"""
        if self.baseline_metrics:
            baseline_file = self.storage_path / "baseline_metrics.json"
            with open(baseline_file, 'w') as f:
                json.dump(asdict(self.baseline_metrics), f, indent=2)
    
    def _save_dashboard_data(self, data: PerformanceDashboardData) -> None:
        """Save dashboard data to storage"""
        dashboard_file = self.storage_path / f"dashboard_{data.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert to serializable format
        serializable_data = {
            "timestamp": data.timestamp.isoformat(),
            "retrieval_metrics": asdict(data.retrieval_metrics),
            "code_metrics": asdict(data.code_metrics),
            "query_metrics": asdict(data.query_metrics),
            "system_health": data.system_health,
            "active_alerts": [asdict(alert) for alert in data.active_alerts],
            "historical_trends": data.historical_trends
        }
        
        with open(dashboard_file, 'w') as f:
            json.dump(serializable_data, f, indent=2, default=str)
    
    def _load_monitoring_data(self) -> None:
        """Load existing monitoring data from storage"""
        try:
            # Load baseline metrics
            baseline_file = self.storage_path / "baseline_metrics.json"
            if baseline_file.exists():
                with open(baseline_file, 'r') as f:
                    baseline_data = json.load(f)
                    self.baseline_metrics = EvaluationMetrics(**baseline_data)
                    self.evaluator.set_baseline_metrics(self.baseline_metrics)
            
            # Load backward compatibility patterns
            patterns_file = self.storage_path / "compatibility_patterns.json"
            if patterns_file.exists():
                with open(patterns_file, 'r') as f:
                    patterns = json.load(f)
                    self.backward_compatibility_patterns = set(patterns)
        
        except Exception as e:
            self.logger.error(f"Error loading monitoring data: {e}")


# Default alert handlers

def console_alert_handler(alert: QualityAlert) -> None:
    """Simple console alert handler"""
    severity_colors = {
        AlertSeverity.LOW: "\033[92m",      # Green
        AlertSeverity.MEDIUM: "\033[93m",   # Yellow
        AlertSeverity.HIGH: "\033[91m",     # Red
        AlertSeverity.CRITICAL: "\033[95m"  # Magenta
    }
    reset_color = "\033[0m"
    
    color = severity_colors.get(alert.severity, "")
    print(f"{color}[{alert.severity.value.upper()}] {alert.alert_type.value}: {alert.message}{reset_color}")


def log_alert_handler(alert: QualityAlert) -> None:
    """Logging alert handler"""
    logger = logging.getLogger("rag_quality_alerts")
    
    log_levels = {
        AlertSeverity.LOW: logging.INFO,
        AlertSeverity.MEDIUM: logging.WARNING,
        AlertSeverity.HIGH: logging.ERROR,
        AlertSeverity.CRITICAL: logging.CRITICAL
    }
    
    level = log_levels.get(alert.severity, logging.INFO)
    logger.log(level, f"{alert.alert_type.value}: {alert.message} (ID: {alert.alert_id})")
"""
Agent error detection and alerting system.

This module provides comprehensive error detection, pattern analysis,
and alerting capabilities for agent execution monitoring.
"""

import re
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Pattern
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    NETWORK = "network"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    RESOURCE = "resource"
    LOGIC = "logic"
    EXTERNAL_API = "external_api"
    SYSTEM = "system"
    UNKNOWN = "unknown"


@dataclass
class ErrorPattern:
    """Defines a pattern for error detection and classification."""
    
    pattern_id: str
    name: str
    regex_pattern: Pattern[str]
    category: ErrorCategory
    severity: ErrorSeverity
    description: str
    suggested_action: str
    threshold_count: int = 1
    threshold_window_minutes: int = 5
    
    def matches(self, error_message: str) -> bool:
        """Check if error message matches this pattern."""
        return bool(self.regex_pattern.search(error_message))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary."""
        return {
            'pattern_id': self.pattern_id,
            'name': self.name,
            'pattern': self.regex_pattern.pattern,
            'category': self.category.value,
            'severity': self.severity.value,
            'description': self.description,
            'suggested_action': self.suggested_action,
            'threshold_count': self.threshold_count,
            'threshold_window_minutes': self.threshold_window_minutes
        }


@dataclass
class ErrorAlert:
    """Represents an error alert."""
    
    alert_id: str
    timestamp: datetime
    agent_name: str
    session_id: str
    error_message: str
    pattern: Optional[ErrorPattern]
    category: ErrorCategory
    severity: ErrorSeverity
    occurrence_count: int = 1
    first_occurrence: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'agent_name': self.agent_name,
            'session_id': self.session_id,
            'error_message': self.error_message,
            'pattern_id': self.pattern.pattern_id if self.pattern else None,
            'category': self.category.value,
            'severity': self.severity.value,
            'occurrence_count': self.occurrence_count,
            'first_occurrence': self.first_occurrence.isoformat() if self.first_occurrence else None,
            'context': self.context
        }


class ErrorDetector:
    """
    Comprehensive error detection and alerting system.
    
    Monitors agent execution for errors, classifies them using patterns,
    and generates alerts when thresholds are exceeded.
    """
    
    def __init__(self):
        """Initialize error detector with default patterns."""
        self.error_patterns: Dict[str, ErrorPattern] = {}
        self.error_history: deque = deque(maxlen=10000)
        self.alert_history: deque = deque(maxlen=1000)
        
        # Pattern occurrence tracking
        self.pattern_occurrences: Dict[str, List[datetime]] = defaultdict(list)
        self.session_errors: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[ErrorAlert], None]] = []
        
        # Initialize default patterns
        self._initialize_default_patterns()
        
        logger.info("ErrorDetector initialized with default patterns")
    
    def add_error_pattern(self, pattern: ErrorPattern) -> None:
        """Add a new error pattern for detection."""
        self.error_patterns[pattern.pattern_id] = pattern
        logger.info(f"Added error pattern: {pattern.name}")
    
    def remove_error_pattern(self, pattern_id: str) -> bool:
        """Remove an error pattern."""
        if pattern_id in self.error_patterns:
            del self.error_patterns[pattern_id]
            logger.info(f"Removed error pattern: {pattern_id}")
            return True
        return False
    
    def detect_error(self, agent_name: str, session_id: str, error_message: str,
                    context: Optional[Dict[str, Any]] = None) -> Optional[ErrorAlert]:
        """
        Detect and classify an error, potentially generating an alert.
        
        Args:
            agent_name: Name of the agent that encountered the error
            session_id: Session ID where the error occurred
            error_message: The error message to analyze
            context: Additional context information
            
        Returns:
            ErrorAlert if an alert should be generated, None otherwise
        """
        timestamp = datetime.now()
        context = context or {}
        
        # Record error in history
        error_record = {
            'timestamp': timestamp,
            'agent_name': agent_name,
            'session_id': session_id,
            'error_message': error_message,
            'context': context
        }
        self.error_history.append(error_record)
        self.session_errors[session_id].append(error_record)
        
        # Find matching pattern
        matched_pattern = self._find_matching_pattern(error_message)
        
        if matched_pattern:
            # Check if threshold is exceeded
            if self._check_pattern_threshold(matched_pattern, timestamp):
                alert = self._create_alert(
                    agent_name, session_id, error_message,
                    matched_pattern, context, timestamp
                )
                return alert
        else:
            # Create alert for unclassified error
            alert = self._create_unclassified_alert(
                agent_name, session_id, error_message, context, timestamp
            )
            return alert
        
        return None
    
    def get_error_statistics(self, agent_name: Optional[str] = None,
                           time_window_hours: int = 24) -> Dict[str, Any]:
        """Get error statistics for analysis."""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        # Filter errors by time window and agent
        filtered_errors = []
        for error in self.error_history:
            if error['timestamp'] >= cutoff_time:
                if agent_name is None or error['agent_name'] == agent_name:
                    filtered_errors.append(error)
        
        if not filtered_errors:
            return {
                'total_errors': 0,
                'time_window_hours': time_window_hours,
                'agent_name': agent_name
            }
        
        # Calculate statistics
        error_by_agent = defaultdict(int)
        error_by_category = defaultdict(int)
        error_by_severity = defaultdict(int)
        error_by_pattern = defaultdict(int)
        
        for error in filtered_errors:
            error_by_agent[error['agent_name']] += 1
            
            # Classify error
            pattern = self._find_matching_pattern(error['error_message'])
            if pattern:
                error_by_category[pattern.category.value] += 1
                error_by_severity[pattern.severity.value] += 1
                error_by_pattern[pattern.name] += 1
            else:
                error_by_category['unknown'] += 1
                error_by_severity['medium'] += 1
        
        return {
            'total_errors': len(filtered_errors),
            'time_window_hours': time_window_hours,
            'agent_name': agent_name,
            'errors_by_agent': dict(error_by_agent),
            'errors_by_category': dict(error_by_category),
            'errors_by_severity': dict(error_by_severity),
            'errors_by_pattern': dict(error_by_pattern),
            'error_rate_per_hour': len(filtered_errors) / time_window_hours
        }
    
    def get_session_errors(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all errors for a specific session."""
        return self.session_errors.get(session_id, [])
    
    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent error alerts."""
        alerts = list(self.alert_history)
        recent_alerts = alerts[-limit:] if len(alerts) > limit else alerts
        return [alert.to_dict() for alert in recent_alerts]
    
    def get_error_patterns(self) -> List[Dict[str, Any]]:
        """Get all configured error patterns."""
        return [pattern.to_dict() for pattern in self.error_patterns.values()]
    
    def analyze_error_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze error trends over time."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Group errors by hour
        hourly_errors = defaultdict(int)
        hourly_categories = defaultdict(lambda: defaultdict(int))
        
        for error in self.error_history:
            if error['timestamp'] >= cutoff_time:
                hour_key = error['timestamp'].strftime('%Y-%m-%d %H:00')
                hourly_errors[hour_key] += 1
                
                # Classify error
                pattern = self._find_matching_pattern(error['error_message'])
                category = pattern.category.value if pattern else 'unknown'
                hourly_categories[hour_key][category] += 1
        
        # Calculate trend
        hours_list = sorted(hourly_errors.keys())
        if len(hours_list) >= 2:
            recent_avg = sum(hourly_errors[h] for h in hours_list[-6:]) / min(6, len(hours_list))
            earlier_avg = sum(hourly_errors[h] for h in hours_list[:-6]) / max(1, len(hours_list) - 6)
            trend = "increasing" if recent_avg > earlier_avg else "decreasing" if recent_avg < earlier_avg else "stable"
        else:
            trend = "insufficient_data"
        
        return {
            'time_window_hours': hours,
            'hourly_errors': dict(hourly_errors),
            'hourly_categories': {h: dict(cats) for h, cats in hourly_categories.items()},
            'trend': trend,
            'total_hours_analyzed': len(hours_list)
        }
    
    def add_alert_callback(self, callback: Callable[[ErrorAlert], None]) -> None:
        """Add a callback for error alerts."""
        self.alert_callbacks.append(callback)
    
    def _initialize_default_patterns(self) -> None:
        """Initialize default error patterns."""
        default_patterns = [
            ErrorPattern(
                pattern_id="timeout_error",
                name="Timeout Error",
                regex_pattern=re.compile(r"timeout|timed out|connection timeout", re.IGNORECASE),
                category=ErrorCategory.TIMEOUT,
                severity=ErrorSeverity.HIGH,
                description="Request or operation timed out",
                suggested_action="Check network connectivity and increase timeout values",
                threshold_count=3,
                threshold_window_minutes=10
            ),
            ErrorPattern(
                pattern_id="network_error",
                name="Network Error",
                regex_pattern=re.compile(r"connection refused|network unreachable|dns resolution failed", re.IGNORECASE),
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.HIGH,
                description="Network connectivity issue",
                suggested_action="Check network configuration and external service availability",
                threshold_count=2,
                threshold_window_minutes=5
            ),
            ErrorPattern(
                pattern_id="auth_error",
                name="Authentication Error",
                regex_pattern=re.compile(r"unauthorized|authentication failed|invalid token|access denied", re.IGNORECASE),
                category=ErrorCategory.AUTHENTICATION,
                severity=ErrorSeverity.CRITICAL,
                description="Authentication or authorization failure",
                suggested_action="Check API keys, tokens, and permissions",
                threshold_count=1,
                threshold_window_minutes=1
            ),
            ErrorPattern(
                pattern_id="validation_error",
                name="Validation Error",
                regex_pattern=re.compile(r"validation error|invalid input|bad request|malformed", re.IGNORECASE),
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.MEDIUM,
                description="Input validation failure",
                suggested_action="Review input data format and validation rules",
                threshold_count=5,
                threshold_window_minutes=15
            ),
            ErrorPattern(
                pattern_id="resource_error",
                name="Resource Error",
                regex_pattern=re.compile(r"out of memory|disk full|resource exhausted|quota exceeded", re.IGNORECASE),
                category=ErrorCategory.RESOURCE,
                severity=ErrorSeverity.CRITICAL,
                description="System resource exhaustion",
                suggested_action="Check system resources and scale if necessary",
                threshold_count=1,
                threshold_window_minutes=1
            ),
            ErrorPattern(
                pattern_id="api_error",
                name="External API Error",
                regex_pattern=re.compile(r"api error|service unavailable|internal server error|bad gateway", re.IGNORECASE),
                category=ErrorCategory.EXTERNAL_API,
                severity=ErrorSeverity.HIGH,
                description="External API service error",
                suggested_action="Check external service status and implement retry logic",
                threshold_count=3,
                threshold_window_minutes=10
            ),
            ErrorPattern(
                pattern_id="system_error",
                name="System Error",
                regex_pattern=re.compile(r"system error|kernel panic|segmentation fault|core dumped", re.IGNORECASE),
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.CRITICAL,
                description="System-level error",
                suggested_action="Check system logs and restart services if necessary",
                threshold_count=1,
                threshold_window_minutes=1
            )
        ]
        
        for pattern in default_patterns:
            self.add_error_pattern(pattern)
    
    def _find_matching_pattern(self, error_message: str) -> Optional[ErrorPattern]:
        """Find the first matching error pattern."""
        for pattern in self.error_patterns.values():
            if pattern.matches(error_message):
                return pattern
        return None
    
    def _check_pattern_threshold(self, pattern: ErrorPattern, timestamp: datetime) -> bool:
        """Check if pattern threshold is exceeded."""
        pattern_id = pattern.pattern_id
        
        # Add current occurrence
        self.pattern_occurrences[pattern_id].append(timestamp)
        
        # Clean old occurrences outside the window
        cutoff_time = timestamp - timedelta(minutes=pattern.threshold_window_minutes)
        self.pattern_occurrences[pattern_id] = [
            t for t in self.pattern_occurrences[pattern_id] if t >= cutoff_time
        ]
        
        # Check if threshold is exceeded
        return len(self.pattern_occurrences[pattern_id]) >= pattern.threshold_count
    
    def _create_alert(self, agent_name: str, session_id: str, error_message: str,
                     pattern: ErrorPattern, context: Dict[str, Any],
                     timestamp: datetime) -> ErrorAlert:
        """Create an error alert for a matched pattern."""
        alert_id = f"error_{pattern.pattern_id}_{session_id}_{int(time.time())}"
        
        # Count occurrences in the window
        cutoff_time = timestamp - timedelta(minutes=pattern.threshold_window_minutes)
        occurrence_count = len([
            t for t in self.pattern_occurrences[pattern.pattern_id] if t >= cutoff_time
        ])
        
        # Find first occurrence
        first_occurrence = min(self.pattern_occurrences[pattern.pattern_id]) if self.pattern_occurrences[pattern.pattern_id] else timestamp
        
        alert = ErrorAlert(
            alert_id=alert_id,
            timestamp=timestamp,
            agent_name=agent_name,
            session_id=session_id,
            error_message=error_message,
            pattern=pattern,
            category=pattern.category,
            severity=pattern.severity,
            occurrence_count=occurrence_count,
            first_occurrence=first_occurrence,
            context=context
        )
        
        self._process_alert(alert)
        return alert
    
    def _create_unclassified_alert(self, agent_name: str, session_id: str,
                                 error_message: str, context: Dict[str, Any],
                                 timestamp: datetime) -> ErrorAlert:
        """Create an alert for an unclassified error."""
        alert_id = f"error_unclassified_{session_id}_{int(time.time())}"
        
        alert = ErrorAlert(
            alert_id=alert_id,
            timestamp=timestamp,
            agent_name=agent_name,
            session_id=session_id,
            error_message=error_message,
            pattern=None,
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.MEDIUM,
            occurrence_count=1,
            first_occurrence=timestamp,
            context=context
        )
        
        self._process_alert(alert)
        return alert
    
    def _process_alert(self, alert: ErrorAlert) -> None:
        """Process and store an error alert."""
        self.alert_history.append(alert)
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        # Log alert
        logger.error(f"Error alert generated: {alert.severity.value.upper()} - {alert.error_message}")


# Global instance
_global_error_detector = ErrorDetector()


def get_error_detector() -> ErrorDetector:
    """Get the global error detector instance."""
    return _global_error_detector
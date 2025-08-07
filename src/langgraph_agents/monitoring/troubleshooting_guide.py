"""
Agent execution troubleshooting guide and automated diagnosis.

This module provides automated troubleshooting capabilities and diagnostic
tools for common agent execution issues.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .execution_monitor import get_execution_monitor, ExecutionStatus
from .performance_tracker import get_performance_tracker
from .error_detector import get_error_detector, ErrorCategory, ErrorSeverity
from .execution_history import get_execution_history
from .debugging_tools import get_agent_debugger, get_flow_analyzer

logger = logging.getLogger(__name__)


class IssueSeverity(Enum):
    """Issue severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IssueCategory(Enum):
    """Categories of issues."""
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    RESOURCE = "resource"
    LOGIC = "logic"
    CONFIGURATION = "configuration"
    EXTERNAL = "external"


@dataclass
class DiagnosticResult:
    """Result of a diagnostic check."""
    
    check_name: str
    issue_found: bool
    severity: IssueSeverity
    category: IssueCategory
    description: str
    evidence: Dict[str, Any]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'check_name': self.check_name,
            'issue_found': self.issue_found,
            'severity': self.severity.value,
            'category': self.category.value,
            'description': self.description,
            'evidence': self.evidence,
            'recommendations': self.recommendations
        }


class TroubleshootingGuide:
    """
    Comprehensive troubleshooting and diagnostic system.
    
    Provides automated diagnosis of common agent execution issues
    and actionable recommendations for resolution.
    """
    
    def __init__(self):
        """Initialize troubleshooting guide."""
        self.execution_monitor = get_execution_monitor()
        self.performance_tracker = get_performance_tracker()
        self.error_detector = get_error_detector()
        self.execution_history = get_execution_history()
        self.flow_analyzer = get_flow_analyzer()
        
        # Diagnostic thresholds
        self.thresholds = {
            'max_execution_time': 300.0,  # 5 minutes
            'max_error_rate': 0.1,        # 10%
            'min_success_rate': 0.9,      # 90%
            'max_memory_usage': 2048.0,   # 2GB
            'max_retry_rate': 0.2,        # 20%
            'min_cache_hit_rate': 0.5     # 50%
        }
        
        logger.info("TroubleshootingGuide initialized")
    
    def diagnose_agent(self, agent_name: str, hours: int = 24) -> List[DiagnosticResult]:
        """Run comprehensive diagnostics for a specific agent."""
        results = []
        
        # Performance diagnostics
        results.extend(self._diagnose_performance(agent_name, hours))
        
        # Reliability diagnostics
        results.extend(self._diagnose_reliability(agent_name, hours))
        
        # Resource diagnostics
        results.extend(self._diagnose_resources(agent_name, hours))
        
        # Error pattern diagnostics
        results.extend(self._diagnose_error_patterns(agent_name, hours))
        
        # Configuration diagnostics
        results.extend(self._diagnose_configuration(agent_name))
        
        return results
    
    def diagnose_session(self, session_id: str) -> List[DiagnosticResult]:
        """Run diagnostics for a specific execution session."""
        results = []
        
        # Get execution record
        records = self.execution_history.search_records(limit=1000)
        record = None
        for r in records:
            if r.session_id == session_id:
                record = r
                break
        
        if not record:
            return [DiagnosticResult(
                check_name="session_existence",
                issue_found=True,
                severity=IssueSeverity.HIGH,
                category=IssueCategory.CONFIGURATION,
                description=f"Session {session_id} not found in execution history",
                evidence={"session_id": session_id},
                recommendations=["Verify the session ID is correct", "Check if execution completed"]
            )]
        
        # Analyze execution flow
        flow_analysis = self.flow_analyzer.analyze_execution_flow(session_id)
        
        if 'error' not in flow_analysis:
            # Check for bottlenecks
            if flow_analysis['bottlenecks']:
                results.append(DiagnosticResult(
                    check_name="execution_bottlenecks",
                    issue_found=True,
                    severity=IssueSeverity.MEDIUM,
                    category=IssueCategory.PERFORMANCE,
                    description=f"Found {len(flow_analysis['bottlenecks'])} execution bottlenecks",
                    evidence={"bottlenecks": flow_analysis['bottlenecks']},
                    recommendations=[
                        "Optimize slow steps identified in bottlenecks",
                        "Consider parallel processing for independent operations",
                        "Review resource allocation for bottleneck steps"
                    ]
                ))
            
            # Check execution time
            if record.total_duration and record.total_duration > self.thresholds['max_execution_time']:
                results.append(DiagnosticResult(
                    check_name="execution_time",
                    issue_found=True,
                    severity=IssueSeverity.HIGH,
                    category=IssueCategory.PERFORMANCE,
                    description=f"Execution time ({record.total_duration:.2f}s) exceeds threshold",
                    evidence={"duration": record.total_duration, "threshold": self.thresholds['max_execution_time']},
                    recommendations=[
                        "Profile individual steps to identify slow operations",
                        "Implement caching for repeated operations",
                        "Consider breaking down complex operations"
                    ]
                ))
            
            # Check error count
            if len(record.errors) > 0:
                severity = IssueSeverity.CRITICAL if record.final_status == ExecutionStatus.FAILED else IssueSeverity.MEDIUM
                results.append(DiagnosticResult(
                    check_name="execution_errors",
                    issue_found=True,
                    severity=severity,
                    category=IssueCategory.RELIABILITY,
                    description=f"Found {len(record.errors)} errors during execution",
                    evidence={"errors": record.errors, "final_status": record.final_status.value},
                    recommendations=[
                        "Review error messages for root cause analysis",
                        "Implement better error handling and recovery",
                        "Add input validation to prevent common errors"
                    ]
                ))
            
            # Check retry count
            if record.retry_count > 3:
                results.append(DiagnosticResult(
                    check_name="excessive_retries",
                    issue_found=True,
                    severity=IssueSeverity.MEDIUM,
                    category=IssueCategory.RELIABILITY,
                    description=f"Excessive retry count: {record.retry_count}",
                    evidence={"retry_count": record.retry_count},
                    recommendations=[
                        "Investigate root cause of failures requiring retries",
                        "Implement exponential backoff for retry logic",
                        "Consider circuit breaker pattern for external dependencies"
                    ]
                ))
        
        return results
    
    def diagnose_system_health(self) -> List[DiagnosticResult]:
        """Run system-wide health diagnostics."""
        results = []
        
        # Check active executions
        active_executions = self.execution_monitor.get_active_executions()
        if len(active_executions) > 20:  # Arbitrary threshold
            results.append(DiagnosticResult(
                check_name="active_executions",
                issue_found=True,
                severity=IssueSeverity.MEDIUM,
                category=IssueCategory.RESOURCE,
                description=f"High number of active executions: {len(active_executions)}",
                evidence={"active_count": len(active_executions)},
                recommendations=[
                    "Monitor system resources (CPU, memory)",
                    "Consider implementing execution queuing",
                    "Review concurrent execution limits"
                ]
            ))
        
        # Check overall error rate
        error_stats = self.error_detector.get_error_statistics(time_window_hours=1)
        error_rate = error_stats.get('error_rate_per_hour', 0) / 100
        if error_rate > self.thresholds['max_error_rate']:
            results.append(DiagnosticResult(
                check_name="system_error_rate",
                issue_found=True,
                severity=IssueSeverity.HIGH,
                category=IssueCategory.RELIABILITY,
                description=f"System error rate ({error_rate:.2%}) exceeds threshold",
                evidence={"error_rate": error_rate, "threshold": self.thresholds['max_error_rate']},
                recommendations=[
                    "Investigate most common error patterns",
                    "Review system logs for infrastructure issues",
                    "Consider implementing circuit breakers"
                ]
            ))
        
        # Check performance trends
        performance_summary = self.performance_tracker.get_performance_summary()
        avg_exec_time = performance_summary.get('avg_execution_time', 0)
        if avg_exec_time > self.thresholds['max_execution_time'] / 2:
            results.append(DiagnosticResult(
                check_name="system_performance",
                issue_found=True,
                severity=IssueSeverity.MEDIUM,
                category=IssueCategory.PERFORMANCE,
                description=f"Average execution time ({avg_exec_time:.2f}s) is concerning",
                evidence={"avg_execution_time": avg_exec_time},
                recommendations=[
                    "Profile agent execution to identify bottlenecks",
                    "Optimize database queries and external API calls",
                    "Consider horizontal scaling"
                ]
            ))
        
        return results
    
    def get_troubleshooting_recommendations(self, issue_category: IssueCategory) -> List[str]:
        """Get general troubleshooting recommendations for an issue category."""
        recommendations = {
            IssueCategory.PERFORMANCE: [
                "Profile agent execution to identify bottlenecks",
                "Implement caching for frequently accessed data",
                "Optimize database queries and reduce N+1 problems",
                "Consider parallel processing for independent operations",
                "Review and optimize algorithm complexity",
                "Monitor resource usage (CPU, memory, I/O)",
                "Implement connection pooling for external services"
            ],
            IssueCategory.RELIABILITY: [
                "Implement comprehensive error handling and recovery",
                "Add input validation and sanitization",
                "Use circuit breaker pattern for external dependencies",
                "Implement exponential backoff for retries",
                "Add health checks for critical dependencies",
                "Implement graceful degradation strategies",
                "Monitor and alert on error rates"
            ],
            IssueCategory.RESOURCE: [
                "Monitor memory usage and implement garbage collection",
                "Set appropriate resource limits and quotas",
                "Implement connection pooling and resource reuse",
                "Use streaming for large data processing",
                "Monitor disk space and implement cleanup policies",
                "Consider horizontal scaling for high load",
                "Implement resource-aware scheduling"
            ],
            IssueCategory.LOGIC: [
                "Add comprehensive unit and integration tests",
                "Implement state validation at key checkpoints",
                "Use type hints and static analysis tools",
                "Add logging for critical decision points",
                "Implement assertion checks for invariants",
                "Review business logic with domain experts",
                "Use formal verification for critical paths"
            ],
            IssueCategory.CONFIGURATION: [
                "Validate configuration at startup",
                "Use configuration management tools",
                "Implement configuration versioning",
                "Add configuration documentation",
                "Use environment-specific configurations",
                "Implement configuration hot-reloading",
                "Monitor configuration drift"
            ],
            IssueCategory.EXTERNAL: [
                "Implement timeout and retry policies",
                "Monitor external service health",
                "Use service discovery and load balancing",
                "Implement fallback mechanisms",
                "Cache external service responses",
                "Monitor API rate limits and quotas",
                "Implement service mesh for microservices"
            ]
        }
        
        return recommendations.get(issue_category, [])
    
    def generate_troubleshooting_report(self, agent_name: Optional[str] = None,
                                      session_id: Optional[str] = None) -> str:
        """Generate a comprehensive troubleshooting report."""
        report = "=== Agent Troubleshooting Report ===\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        if session_id:
            report += f"Session Analysis: {session_id}\n"
            results = self.diagnose_session(session_id)
        elif agent_name:
            report += f"Agent Analysis: {agent_name}\n"
            results = self.diagnose_agent(agent_name)
        else:
            report += "System Health Analysis\n"
            results = self.diagnose_system_health()
        
        report += "=" * 50 + "\n\n"
        
        # Group results by severity
        critical_issues = [r for r in results if r.severity == IssueSeverity.CRITICAL]
        high_issues = [r for r in results if r.severity == IssueSeverity.HIGH]
        medium_issues = [r for r in results if r.severity == IssueSeverity.MEDIUM]
        low_issues = [r for r in results if r.severity == IssueSeverity.LOW]
        
        # Summary
        report += f"Issues Found: {len(results)}\n"
        report += f"  Critical: {len(critical_issues)}\n"
        report += f"  High: {len(high_issues)}\n"
        report += f"  Medium: {len(medium_issues)}\n"
        report += f"  Low: {len(low_issues)}\n\n"
        
        # Critical issues first
        if critical_issues:
            report += "=== CRITICAL ISSUES ===\n"
            for issue in critical_issues:
                report += self._format_issue(issue)
            report += "\n"
        
        # High priority issues
        if high_issues:
            report += "=== HIGH PRIORITY ISSUES ===\n"
            for issue in high_issues:
                report += self._format_issue(issue)
            report += "\n"
        
        # Medium priority issues
        if medium_issues:
            report += "=== MEDIUM PRIORITY ISSUES ===\n"
            for issue in medium_issues:
                report += self._format_issue(issue)
            report += "\n"
        
        # Low priority issues
        if low_issues:
            report += "=== LOW PRIORITY ISSUES ===\n"
            for issue in low_issues:
                report += self._format_issue(issue)
            report += "\n"
        
        # General recommendations
        categories = set(r.category for r in results if r.issue_found)
        if categories:
            report += "=== GENERAL RECOMMENDATIONS ===\n"
            for category in categories:
                report += f"\n{category.value.title()} Issues:\n"
                recommendations = self.get_troubleshooting_recommendations(category)
                for rec in recommendations[:5]:  # Top 5 recommendations
                    report += f"  - {rec}\n"
        
        return report
    
    def _diagnose_performance(self, agent_name: str, hours: int) -> List[DiagnosticResult]:
        """Diagnose performance issues."""
        results = []
        
        # Get performance summary
        performance_summary = self.performance_tracker.get_agent_performance_history(agent_name, limit=100)
        
        if performance_summary:
            # Calculate average execution time
            exec_times = [p.get('total_execution_time', 0) for p in performance_summary if p.get('total_execution_time')]
            if exec_times:
                avg_time = sum(exec_times) / len(exec_times)
                if avg_time > self.thresholds['max_execution_time']:
                    results.append(DiagnosticResult(
                        check_name="average_execution_time",
                        issue_found=True,
                        severity=IssueSeverity.HIGH,
                        category=IssueCategory.PERFORMANCE,
                        description=f"Average execution time ({avg_time:.2f}s) exceeds threshold",
                        evidence={"avg_time": avg_time, "threshold": self.thresholds['max_execution_time']},
                        recommendations=[
                            "Profile agent execution to identify bottlenecks",
                            "Optimize slow operations and database queries",
                            "Consider caching frequently accessed data"
                        ]
                    ))
            
            # Check memory usage
            memory_usage = [p.get('peak_memory_mb', 0) for p in performance_summary if p.get('peak_memory_mb')]
            if memory_usage:
                avg_memory = sum(memory_usage) / len(memory_usage)
                if avg_memory > self.thresholds['max_memory_usage']:
                    results.append(DiagnosticResult(
                        check_name="memory_usage",
                        issue_found=True,
                        severity=IssueSeverity.MEDIUM,
                        category=IssueCategory.RESOURCE,
                        description=f"Average memory usage ({avg_memory:.1f}MB) is high",
                        evidence={"avg_memory": avg_memory, "threshold": self.thresholds['max_memory_usage']},
                        recommendations=[
                            "Review memory usage patterns",
                            "Implement memory-efficient data structures",
                            "Consider streaming for large data processing"
                        ]
                    ))
        
        return results
    
    def _diagnose_reliability(self, agent_name: str, hours: int) -> List[DiagnosticResult]:
        """Diagnose reliability issues."""
        results = []
        
        # Get execution statistics
        exec_stats = self.execution_history.get_execution_statistics(agent_name, days=hours//24 or 1)
        
        if exec_stats.get('total_executions', 0) > 0:
            success_rate = exec_stats.get('success_rate', 0)
            if success_rate < self.thresholds['min_success_rate']:
                results.append(DiagnosticResult(
                    check_name="success_rate",
                    issue_found=True,
                    severity=IssueSeverity.CRITICAL,
                    category=IssueCategory.RELIABILITY,
                    description=f"Success rate ({success_rate:.2%}) below threshold",
                    evidence={"success_rate": success_rate, "threshold": self.thresholds['min_success_rate']},
                    recommendations=[
                        "Investigate common failure patterns",
                        "Improve error handling and recovery",
                        "Add input validation and sanitization"
                    ]
                ))
        
        return results
    
    def _diagnose_resources(self, agent_name: str, hours: int) -> List[DiagnosticResult]:
        """Diagnose resource-related issues."""
        results = []
        
        # This would typically check system resources, database connections, etc.
        # For now, we'll check performance metrics as a proxy
        
        performance_summary = self.performance_tracker.get_agent_performance_history(agent_name, limit=50)
        
        if performance_summary:
            # Check cache hit rate
            cache_rates = [p.get('cache_hit_rate', 0) for p in performance_summary if 'cache_hit_rate' in p]
            if cache_rates:
                avg_cache_rate = sum(cache_rates) / len(cache_rates)
                if avg_cache_rate < self.thresholds['min_cache_hit_rate']:
                    results.append(DiagnosticResult(
                        check_name="cache_hit_rate",
                        issue_found=True,
                        severity=IssueSeverity.MEDIUM,
                        category=IssueCategory.PERFORMANCE,
                        description=f"Cache hit rate ({avg_cache_rate:.2%}) is low",
                        evidence={"cache_hit_rate": avg_cache_rate, "threshold": self.thresholds['min_cache_hit_rate']},
                        recommendations=[
                            "Review caching strategy and cache keys",
                            "Increase cache size or TTL if appropriate",
                            "Implement cache warming for frequently accessed data"
                        ]
                    ))
        
        return results
    
    def _diagnose_error_patterns(self, agent_name: str, hours: int) -> List[DiagnosticResult]:
        """Diagnose error patterns."""
        results = []
        
        # Get error statistics
        error_stats = self.error_detector.get_error_statistics(agent_name, hours)
        
        if error_stats.get('total_errors', 0) > 0:
            # Check for specific error categories
            error_categories = error_stats.get('errors_by_category', {})
            
            for category, count in error_categories.items():
                if count > 5:  # Arbitrary threshold
                    severity = IssueSeverity.HIGH if category in ['authentication', 'system'] else IssueSeverity.MEDIUM
                    results.append(DiagnosticResult(
                        check_name=f"error_pattern_{category}",
                        issue_found=True,
                        severity=severity,
                        category=IssueCategory.RELIABILITY,
                        description=f"High frequency of {category} errors: {count}",
                        evidence={"error_category": category, "count": count},
                        recommendations=self._get_error_category_recommendations(category)
                    ))
        
        return results
    
    def _diagnose_configuration(self, agent_name: str) -> List[DiagnosticResult]:
        """Diagnose configuration issues."""
        results = []
        
        # This would typically check configuration validity
        # For now, we'll check if the agent has any execution history
        
        records = self.execution_history.get_agent_records(agent_name, limit=1)
        if not records:
            results.append(DiagnosticResult(
                check_name="agent_activity",
                issue_found=True,
                severity=IssueSeverity.MEDIUM,
                category=IssueCategory.CONFIGURATION,
                description=f"No execution history found for agent {agent_name}",
                evidence={"agent_name": agent_name},
                recommendations=[
                    "Verify agent is properly configured and registered",
                    "Check if agent is being invoked correctly",
                    "Review agent initialization and setup"
                ]
            ))
        
        return results
    
    def _get_error_category_recommendations(self, category: str) -> List[str]:
        """Get recommendations for specific error categories."""
        recommendations = {
            'network': [
                "Implement retry logic with exponential backoff",
                "Check network connectivity and DNS resolution",
                "Monitor external service availability"
            ],
            'timeout': [
                "Increase timeout values if appropriate",
                "Implement connection pooling",
                "Optimize slow operations"
            ],
            'authentication': [
                "Verify API keys and tokens are valid",
                "Implement token refresh mechanisms",
                "Check authentication service availability"
            ],
            'validation': [
                "Add comprehensive input validation",
                "Improve error messages for validation failures",
                "Review data schemas and formats"
            ],
            'resource': [
                "Monitor system resources (CPU, memory, disk)",
                "Implement resource limits and quotas",
                "Consider scaling resources"
            ]
        }
        
        return recommendations.get(category, [
            "Review error logs for specific details",
            "Implement better error handling",
            "Monitor error patterns and trends"
        ])
    
    def _format_issue(self, issue: DiagnosticResult) -> str:
        """Format an issue for the report."""
        report = f"\n{issue.check_name.upper().replace('_', ' ')}\n"
        report += f"Severity: {issue.severity.value.upper()}\n"
        report += f"Category: {issue.category.value.title()}\n"
        report += f"Description: {issue.description}\n"
        
        if issue.recommendations:
            report += "Recommendations:\n"
            for rec in issue.recommendations:
                report += f"  - {rec}\n"
        
        return report


# Global instance
_global_troubleshooting_guide = TroubleshootingGuide()


def get_troubleshooting_guide() -> TroubleshootingGuide:
    """Get the global troubleshooting guide instance."""
    return _global_troubleshooting_guide
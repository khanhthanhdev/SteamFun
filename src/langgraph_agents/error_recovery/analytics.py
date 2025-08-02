"""
Error analytics and reporting system for advanced error recovery.

This module provides comprehensive analytics, reporting, and insights
for error patterns, recovery effectiveness, and system health.
"""

import logging
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import statistics
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

from .advanced_recovery import ErrorPattern, RecoveryStrategy, ErrorAnalysis, RecoveryExecution
from .escalation_manager import EscalationEvent, EscalationLevel

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsReport:
    """Comprehensive analytics report."""
    report_id: str
    timestamp: datetime
    time_period: str
    summary: Dict[str, Any]
    error_analysis: Dict[str, Any]
    recovery_analysis: Dict[str, Any]
    escalation_analysis: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    recommendations: List[str]
    charts: Dict[str, str]  # Chart name -> file path


@dataclass
class SystemHealthMetrics:
    """System health metrics."""
    timestamp: datetime
    overall_health_score: float  # 0-100
    error_rate: float
    recovery_success_rate: float
    average_recovery_time: float
    active_escalations: int
    circuit_breaker_count: int
    system_load: str
    trending: str  # improving, stable, degrading


class ErrorAnalyticsSystem:
    """Comprehensive error analytics and reporting system."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the error analytics system.
        
        Args:
            config: Configuration for analytics system
        """
        self.config = config
        self.reports_path = Path(config.get('reports_path', 'logs/analytics_reports'))
        self.charts_path = Path(config.get('charts_path', 'logs/analytics_charts'))
        
        # Create directories
        self.reports_path.mkdir(parents=True, exist_ok=True)
        self.charts_path.mkdir(parents=True, exist_ok=True)
        
        # Analytics settings
        self.health_check_interval = config.get('health_check_interval_seconds', 300)  # 5 minutes
        self.report_generation_interval = config.get('report_generation_interval_hours', 24)  # Daily
        self.retention_days = config.get('retention_days', 30)
        
        # Health metrics history
        self.health_history = []
        self.max_health_history = config.get('max_health_history', 1000)
        
        # Chart generation settings
        self.generate_charts = config.get('generate_charts', True)
        self.chart_format = config.get('chart_format', 'png')
        
        logger.info("Error analytics system initialized")
    
    async def generate_comprehensive_report(
        self,
        error_history: List[Dict[str, Any]],
        recovery_history: List[RecoveryExecution],
        escalation_history: List[EscalationEvent],
        time_period_hours: int = 24
    ) -> AnalyticsReport:
        """Generate a comprehensive analytics report.
        
        Args:
            error_history: Error history data
            recovery_history: Recovery execution history
            escalation_history: Escalation event history
            time_period_hours: Time period for the report in hours
            
        Returns:
            Comprehensive analytics report
        """
        report_id = f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        now = datetime.now()
        cutoff_time = now - timedelta(hours=time_period_hours)
        
        # Filter data to time period
        recent_errors = [
            entry for entry in error_history
            if entry.get('timestamp', datetime.min) > cutoff_time
        ]
        
        recent_recoveries = [
            execution for execution in recovery_history
            if execution.start_time > cutoff_time
        ]
        
        recent_escalations = [
            event for event in escalation_history
            if event.timestamp > cutoff_time
        ]
        
        # Generate report sections
        summary = await self._generate_summary(recent_errors, recent_recoveries, recent_escalations)
        error_analysis = await self._analyze_errors(recent_errors)
        recovery_analysis = await self._analyze_recoveries(recent_recoveries)
        escalation_analysis = await self._analyze_escalations(recent_escalations)
        performance_metrics = await self._calculate_performance_metrics(recent_errors, recent_recoveries)
        recommendations = await self._generate_recommendations(
            error_analysis, recovery_analysis, escalation_analysis, performance_metrics
        )
        
        # Generate charts if enabled
        charts = {}
        if self.generate_charts:
            charts = await self._generate_charts(
                report_id, recent_errors, recent_recoveries, recent_escalations
            )
        
        report = AnalyticsReport(
            report_id=report_id,
            timestamp=now,
            time_period=f"{time_period_hours} hours",
            summary=summary,
            error_analysis=error_analysis,
            recovery_analysis=recovery_analysis,
            escalation_analysis=escalation_analysis,
            performance_metrics=performance_metrics,
            recommendations=recommendations,
            charts=charts
        )
        
        # Export report
        await self._export_report(report)
        
        logger.info(f"Comprehensive analytics report generated: {report_id}")
        return report
    
    async def _generate_summary(
        self,
        errors: List[Dict[str, Any]],
        recoveries: List[RecoveryExecution],
        escalations: List[EscalationEvent]
    ) -> Dict[str, Any]:
        """Generate summary statistics."""
        
        total_errors = len(errors)
        total_recoveries = len(recoveries)
        total_escalations = len(escalations)
        
        # Success rates
        successful_recoveries = sum(1 for r in recoveries if r.success)
        recovery_success_rate = successful_recoveries / total_recoveries if total_recoveries > 0 else 0.0
        
        # Error rate calculation
        if errors:
            time_span = (max(e.get('timestamp', datetime.min) for e in errors) - 
                        min(e.get('timestamp', datetime.min) for e in errors)).total_seconds()
            error_rate = total_errors / (time_span / 60) if time_span > 0 else 0.0  # errors per minute
        else:
            error_rate = 0.0
        
        # Most common error patterns
        error_patterns = [e.get('error_analysis', {}).get('pattern_type') for e in errors]
        pattern_counts = Counter(p for p in error_patterns if p)
        most_common_pattern = pattern_counts.most_common(1)[0] if pattern_counts else ("none", 0)
        
        # Most effective recovery strategy
        strategy_success = defaultdict(lambda: {'total': 0, 'success': 0})
        for recovery in recoveries:
            strategy = recovery.strategy.value
            strategy_success[strategy]['total'] += 1
            if recovery.success:
                strategy_success[strategy]['success'] += 1
        
        best_strategy = None
        best_success_rate = 0.0
        for strategy, stats in strategy_success.items():
            if stats['total'] > 0:
                success_rate = stats['success'] / stats['total']
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_strategy = strategy
        
        return {
            'total_errors': total_errors,
            'total_recoveries': total_recoveries,
            'total_escalations': total_escalations,
            'error_rate_per_minute': round(error_rate, 2),
            'recovery_success_rate': round(recovery_success_rate, 3),
            'most_common_error_pattern': most_common_pattern[0],
            'most_common_pattern_count': most_common_pattern[1],
            'most_effective_recovery_strategy': best_strategy,
            'best_strategy_success_rate': round(best_success_rate, 3),
            'critical_escalations': sum(1 for e in escalations if e.level == EscalationLevel.CRITICAL),
            'emergency_escalations': sum(1 for e in escalations if e.level == EscalationLevel.EMERGENCY)
        }
    
    async def _analyze_errors(self, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze error patterns and trends."""
        
        if not errors:
            return {'message': 'No errors in the specified time period'}
        
        # Error distribution by agent
        agent_errors = defaultdict(int)
        for error in errors:
            agent_name = error.get('error', {}).get('agent_name', 'unknown')
            agent_errors[agent_name] += 1
        
        # Error pattern distribution
        pattern_distribution = defaultdict(int)
        for error in errors:
            pattern = error.get('error_analysis', {}).get('pattern_type', 'unknown')
            pattern_distribution[pattern] += 1
        
        # Error severity analysis
        severity_analysis = defaultdict(int)
        for error in errors:
            # Estimate severity based on retry count and escalation
            retry_count = error.get('error', {}).get('retry_count', 0)
            if retry_count == 0:
                severity = 'low'
            elif retry_count < 3:
                severity = 'medium'
            else:
                severity = 'high'
            severity_analysis[severity] += 1
        
        # Time-based analysis
        hourly_distribution = defaultdict(int)
        for error in errors:
            timestamp = error.get('timestamp', datetime.now())
            hour = timestamp.hour
            hourly_distribution[hour] += 1
        
        # Peak error hour
        peak_hour = max(hourly_distribution.items(), key=lambda x: x[1]) if hourly_distribution else (0, 0)
        
        # Error trend analysis
        if len(errors) > 1:
            errors_sorted = sorted(errors, key=lambda x: x.get('timestamp', datetime.min))
            first_half = errors_sorted[:len(errors_sorted)//2]
            second_half = errors_sorted[len(errors_sorted)//2:]
            
            first_half_rate = len(first_half) / max((first_half[-1].get('timestamp', datetime.min) - 
                                                   first_half[0].get('timestamp', datetime.min)).total_seconds() / 60, 1)
            second_half_rate = len(second_half) / max((second_half[-1].get('timestamp', datetime.min) - 
                                                     second_half[0].get('timestamp', datetime.min)).total_seconds() / 60, 1)
            
            if second_half_rate > first_half_rate * 1.2:
                trend = 'increasing'
            elif second_half_rate < first_half_rate * 0.8:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'total_errors': len(errors),
            'errors_by_agent': dict(agent_errors),
            'pattern_distribution': dict(pattern_distribution),
            'severity_distribution': dict(severity_analysis),
            'hourly_distribution': dict(hourly_distribution),
            'peak_error_hour': peak_hour[0],
            'peak_hour_count': peak_hour[1],
            'error_trend': trend,
            'most_problematic_agent': max(agent_errors.items(), key=lambda x: x[1])[0] if agent_errors else 'none',
            'dominant_pattern': max(pattern_distribution.items(), key=lambda x: x[1])[0] if pattern_distribution else 'none'
        }
    
    async def _analyze_recoveries(self, recoveries: List[RecoveryExecution]) -> Dict[str, Any]:
        """Analyze recovery effectiveness and patterns."""
        
        if not recoveries:
            return {'message': 'No recoveries in the specified time period'}
        
        # Strategy effectiveness
        strategy_stats = defaultdict(lambda: {'total': 0, 'success': 0, 'avg_time': 0.0, 'attempts': []})
        
        for recovery in recoveries:
            strategy = recovery.strategy.value
            stats = strategy_stats[strategy]
            stats['total'] += 1
            stats['attempts'].append(recovery.attempts)
            
            if recovery.success:
                stats['success'] += 1
            
            duration = recovery.metrics.get('duration_seconds', 0)
            stats['avg_time'] = (stats['avg_time'] * (stats['total'] - 1) + duration) / stats['total']
        
        # Calculate success rates and average attempts
        strategy_analysis = {}
        for strategy, stats in strategy_stats.items():
            success_rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0.0
            avg_attempts = statistics.mean(stats['attempts']) if stats['attempts'] else 0.0
            
            strategy_analysis[strategy] = {
                'total_uses': stats['total'],
                'success_rate': round(success_rate, 3),
                'average_duration_seconds': round(stats['avg_time'], 2),
                'average_attempts': round(avg_attempts, 2)
            }
        
        # Overall recovery metrics
        successful_recoveries = sum(1 for r in recoveries if r.success)
        overall_success_rate = successful_recoveries / len(recoveries)
        
        recovery_times = [r.metrics.get('duration_seconds', 0) for r in recoveries if r.success]
        avg_recovery_time = statistics.mean(recovery_times) if recovery_times else 0.0
        
        # Recovery trend analysis
        if len(recoveries) > 1:
            recoveries_sorted = sorted(recoveries, key=lambda x: x.start_time)
            first_half = recoveries_sorted[:len(recoveries_sorted)//2]
            second_half = recoveries_sorted[len(recoveries_sorted)//2:]
            
            first_half_success = sum(1 for r in first_half if r.success) / len(first_half)
            second_half_success = sum(1 for r in second_half if r.success) / len(second_half)
            
            if second_half_success > first_half_success + 0.1:
                trend = 'improving'
            elif second_half_success < first_half_success - 0.1:
                trend = 'degrading'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        # Best and worst performing strategies
        best_strategy = max(strategy_analysis.items(), 
                          key=lambda x: x[1]['success_rate'])[0] if strategy_analysis else 'none'
        worst_strategy = min(strategy_analysis.items(), 
                           key=lambda x: x[1]['success_rate'])[0] if strategy_analysis else 'none'
        
        return {
            'total_recoveries': len(recoveries),
            'overall_success_rate': round(overall_success_rate, 3),
            'average_recovery_time_seconds': round(avg_recovery_time, 2),
            'strategy_analysis': strategy_analysis,
            'recovery_trend': trend,
            'best_performing_strategy': best_strategy,
            'worst_performing_strategy': worst_strategy,
            'total_recovery_attempts': sum(r.attempts for r in recoveries),
            'average_attempts_per_recovery': round(sum(r.attempts for r in recoveries) / len(recoveries), 2)
        }
    
    async def _analyze_escalations(self, escalations: List[EscalationEvent]) -> Dict[str, Any]:
        """Analyze escalation patterns and effectiveness."""
        
        if not escalations:
            return {'message': 'No escalations in the specified time period'}
        
        # Escalation level distribution
        level_distribution = defaultdict(int)
        for escalation in escalations:
            level_distribution[escalation.level.value] += 1
        
        # Resolution analysis
        resolved_escalations = [e for e in escalations if e.resolved]
        resolution_rate = len(resolved_escalations) / len(escalations)
        
        # Resolution time analysis
        if resolved_escalations:
            resolution_times = [
                (e.resolution_timestamp - e.timestamp).total_seconds()
                for e in resolved_escalations
                if e.resolution_timestamp
            ]
            avg_resolution_time = statistics.mean(resolution_times) if resolution_times else 0.0
            median_resolution_time = statistics.median(resolution_times) if resolution_times else 0.0
        else:
            avg_resolution_time = 0.0
            median_resolution_time = 0.0
        
        # Threshold analysis
        threshold_triggers = defaultdict(int)
        for escalation in escalations:
            threshold_triggers[escalation.threshold_name] += 1
        
        most_triggered_threshold = max(threshold_triggers.items(), 
                                     key=lambda x: x[1])[0] if threshold_triggers else 'none'
        
        # Escalation trend
        if len(escalations) > 1:
            escalations_sorted = sorted(escalations, key=lambda x: x.timestamp)
            first_half = escalations_sorted[:len(escalations_sorted)//2]
            second_half = escalations_sorted[len(escalations_sorted)//2:]
            
            critical_first = sum(1 for e in first_half if e.level in [EscalationLevel.CRITICAL, EscalationLevel.EMERGENCY])
            critical_second = sum(1 for e in second_half if e.level in [EscalationLevel.CRITICAL, EscalationLevel.EMERGENCY])
            
            if critical_second > critical_first:
                trend = 'escalating'
            elif critical_second < critical_first:
                trend = 'de-escalating'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'total_escalations': len(escalations),
            'level_distribution': dict(level_distribution),
            'resolution_rate': round(resolution_rate, 3),
            'average_resolution_time_seconds': round(avg_resolution_time, 2),
            'median_resolution_time_seconds': round(median_resolution_time, 2),
            'threshold_triggers': dict(threshold_triggers),
            'most_triggered_threshold': most_triggered_threshold,
            'escalation_trend': trend,
            'active_escalations': len([e for e in escalations if not e.resolved]),
            'critical_escalations': level_distribution.get('critical', 0),
            'emergency_escalations': level_distribution.get('emergency', 0)
        }
    
    async def _calculate_performance_metrics(
        self,
        errors: List[Dict[str, Any]],
        recoveries: List[RecoveryExecution]
    ) -> Dict[str, Any]:
        """Calculate system performance metrics."""
        
        # Mean Time To Recovery (MTTR)
        if recoveries:
            recovery_times = [r.metrics.get('duration_seconds', 0) for r in recoveries if r.success]
            mttr = statistics.mean(recovery_times) if recovery_times else 0.0
        else:
            mttr = 0.0
        
        # Mean Time Between Failures (MTBF)
        if len(errors) > 1:
            errors_sorted = sorted(errors, key=lambda x: x.get('timestamp', datetime.min))
            time_diffs = [
                (errors_sorted[i+1].get('timestamp', datetime.min) - 
                 errors_sorted[i].get('timestamp', datetime.min)).total_seconds()
                for i in range(len(errors_sorted) - 1)
            ]
            mtbf = statistics.mean(time_diffs) if time_diffs else 0.0
        else:
            mtbf = 0.0
        
        # System availability (simplified calculation)
        total_time = 24 * 3600  # 24 hours in seconds
        total_downtime = sum(r.metrics.get('duration_seconds', 0) for r in recoveries if not r.success)
        availability = (total_time - total_downtime) / total_time if total_time > 0 else 1.0
        
        # Error density (errors per hour)
        error_density = len(errors) / 24.0  # errors per hour over 24 hour period
        
        # Recovery efficiency
        successful_recoveries = sum(1 for r in recoveries if r.success)
        recovery_efficiency = successful_recoveries / len(recoveries) if recoveries else 0.0
        
        return {
            'mean_time_to_recovery_seconds': round(mttr, 2),
            'mean_time_between_failures_seconds': round(mtbf, 2),
            'system_availability_percentage': round(availability * 100, 2),
            'error_density_per_hour': round(error_density, 2),
            'recovery_efficiency_percentage': round(recovery_efficiency * 100, 2),
            'total_system_downtime_seconds': round(total_downtime, 2)
        } 
   
    async def _generate_recommendations(
        self,
        error_analysis: Dict[str, Any],
        recovery_analysis: Dict[str, Any],
        escalation_analysis: Dict[str, Any],
        performance_metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        
        recommendations = []
        
        # Error-based recommendations
        if error_analysis.get('error_trend') == 'increasing':
            recommendations.append(
                "Error rate is increasing. Consider investigating the root cause of the dominant error pattern: "
                f"{error_analysis.get('dominant_pattern', 'unknown')}"
            )
        
        most_problematic_agent = error_analysis.get('most_problematic_agent')
        if most_problematic_agent and most_problematic_agent != 'none':
            recommendations.append(
                f"Agent '{most_problematic_agent}' has the highest error count. "
                "Review its configuration and dependencies."
            )
        
        # Recovery-based recommendations
        recovery_success_rate = recovery_analysis.get('overall_success_rate', 0.0)
        if recovery_success_rate < 0.7:
            recommendations.append(
                f"Recovery success rate is low ({recovery_success_rate:.1%}). "
                "Consider reviewing recovery strategies and thresholds."
            )
        
        worst_strategy = recovery_analysis.get('worst_performing_strategy')
        if worst_strategy and worst_strategy != 'none':
            strategy_stats = recovery_analysis.get('strategy_analysis', {}).get(worst_strategy, {})
            success_rate = strategy_stats.get('success_rate', 0.0)
            if success_rate < 0.5:
                recommendations.append(
                    f"Recovery strategy '{worst_strategy}' has low success rate ({success_rate:.1%}). "
                    "Consider adjusting parameters or replacing with alternative strategy."
                )
        
        # Escalation-based recommendations
        if escalation_analysis.get('escalation_trend') == 'escalating':
            recommendations.append(
                "Escalation trend is increasing. Review escalation thresholds and consider "
                "proactive measures to prevent critical issues."
            )
        
        resolution_rate = escalation_analysis.get('resolution_rate', 0.0)
        if resolution_rate < 0.8:
            recommendations.append(
                f"Escalation resolution rate is low ({resolution_rate:.1%}). "
                "Review escalation response procedures and resolution workflows."
            )
        
        # Performance-based recommendations
        mttr = performance_metrics.get('mean_time_to_recovery_seconds', 0)
        if mttr > 300:  # 5 minutes
            recommendations.append(
                f"Mean Time To Recovery is high ({mttr:.0f} seconds). "
                "Consider optimizing recovery procedures and automation."
            )
        
        availability = performance_metrics.get('system_availability_percentage', 100)
        if availability < 99.0:
            recommendations.append(
                f"System availability is below target ({availability:.2f}%). "
                "Focus on reducing downtime and improving fault tolerance."
            )
        
        error_density = performance_metrics.get('error_density_per_hour', 0)
        if error_density > 10:
            recommendations.append(
                f"High error density ({error_density:.1f} errors/hour). "
                "Implement preventive measures and improve system stability."
            )
        
        # General recommendations
        if not recommendations:
            recommendations.append(
                "System performance is within acceptable parameters. "
                "Continue monitoring and maintain current operational procedures."
            )
        
        return recommendations
    
    async def _generate_charts(
        self,
        report_id: str,
        errors: List[Dict[str, Any]],
        recoveries: List[RecoveryExecution],
        escalations: List[EscalationEvent]
    ) -> Dict[str, str]:
        """Generate visualization charts for the report."""
        
        charts = {}
        
        try:
            # Error trend chart
            if errors:
                error_chart_path = await self._create_error_trend_chart(report_id, errors)
                charts['error_trend'] = str(error_chart_path)
            
            # Recovery effectiveness chart
            if recoveries:
                recovery_chart_path = await self._create_recovery_chart(report_id, recoveries)
                charts['recovery_effectiveness'] = str(recovery_chart_path)
            
            # Escalation timeline chart
            if escalations:
                escalation_chart_path = await self._create_escalation_chart(report_id, escalations)
                charts['escalation_timeline'] = str(escalation_chart_path)
            
            # System health overview
            health_chart_path = await self._create_health_overview_chart(report_id, errors, recoveries)
            charts['system_health'] = str(health_chart_path)
            
        except Exception as e:
            logger.error(f"Error generating charts: {str(e)}")
        
        return charts
    
    async def _create_error_trend_chart(self, report_id: str, errors: List[Dict[str, Any]]) -> Path:
        """Create error trend visualization chart."""
        
        # Prepare data
        timestamps = [e.get('timestamp', datetime.now()) for e in errors]
        timestamps.sort()
        
        # Group by hour
        hourly_counts = defaultdict(int)
        for ts in timestamps:
            hour_key = ts.replace(minute=0, second=0, microsecond=0)
            hourly_counts[hour_key] += 1
        
        # Create chart
        plt.figure(figsize=(12, 6))
        hours = sorted(hourly_counts.keys())
        counts = [hourly_counts[hour] for hour in hours]
        
        plt.plot(hours, counts, marker='o', linewidth=2, markersize=4)
        plt.title('Error Trend Over Time')
        plt.xlabel('Time')
        plt.ylabel('Error Count')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save chart
        chart_path = self.charts_path / f"{report_id}_error_trend.{self.chart_format}"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    async def _create_recovery_chart(self, report_id: str, recoveries: List[RecoveryExecution]) -> Path:
        """Create recovery effectiveness chart."""
        
        # Prepare data
        strategy_stats = defaultdict(lambda: {'success': 0, 'failure': 0})
        
        for recovery in recoveries:
            strategy = recovery.strategy.value
            if recovery.success:
                strategy_stats[strategy]['success'] += 1
            else:
                strategy_stats[strategy]['failure'] += 1
        
        # Create chart
        strategies = list(strategy_stats.keys())
        success_counts = [strategy_stats[s]['success'] for s in strategies]
        failure_counts = [strategy_stats[s]['failure'] for s in strategies]
        
        plt.figure(figsize=(12, 8))
        x = range(len(strategies))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], success_counts, width, label='Success', color='green', alpha=0.7)
        plt.bar([i + width/2 for i in x], failure_counts, width, label='Failure', color='red', alpha=0.7)
        
        plt.title('Recovery Strategy Effectiveness')
        plt.xlabel('Recovery Strategy')
        plt.ylabel('Count')
        plt.xticks(x, [s.replace('_', ' ').title() for s in strategies], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save chart
        chart_path = self.charts_path / f"{report_id}_recovery_effectiveness.{self.chart_format}"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    async def _create_escalation_chart(self, report_id: str, escalations: List[EscalationEvent]) -> Path:
        """Create escalation timeline chart."""
        
        # Prepare data
        timestamps = [e.timestamp for e in escalations]
        levels = [e.level.value for e in escalations]
        
        # Create timeline chart
        plt.figure(figsize=(12, 6))
        
        level_colors = {
            'warning': 'yellow',
            'critical': 'orange',
            'emergency': 'red'
        }
        
        for i, (ts, level) in enumerate(zip(timestamps, levels)):
            color = level_colors.get(level, 'blue')
            plt.scatter(ts, i, c=color, s=100, alpha=0.7)
        
        plt.title('Escalation Timeline')
        plt.xlabel('Time')
        plt.ylabel('Escalation Events')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Add legend
        for level, color in level_colors.items():
            plt.scatter([], [], c=color, s=100, label=level.title())
        plt.legend()
        
        plt.tight_layout()
        
        # Save chart
        chart_path = self.charts_path / f"{report_id}_escalation_timeline.{self.chart_format}"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    async def _create_health_overview_chart(
        self, 
        report_id: str, 
        errors: List[Dict[str, Any]], 
        recoveries: List[RecoveryExecution]
    ) -> Path:
        """Create system health overview chart."""
        
        # Calculate health metrics
        total_operations = len(errors) + len(recoveries)
        successful_recoveries = sum(1 for r in recoveries if r.success)
        
        # Create pie chart for system health
        plt.figure(figsize=(10, 8))
        
        # Health categories
        categories = ['Successful Operations', 'Failed Operations', 'Recovered Operations']
        sizes = [
            max(0, total_operations - len(errors)),  # Successful operations
            len(errors) - successful_recoveries,      # Failed operations
            successful_recoveries                     # Recovered operations
        ]
        colors = ['green', 'red', 'orange']
        
        # Remove zero values
        non_zero_data = [(cat, size, color) for cat, size, color in zip(categories, sizes, colors) if size > 0]
        if non_zero_data:
            categories, sizes, colors = zip(*non_zero_data)
        
        plt.pie(sizes, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('System Health Overview')
        plt.axis('equal')
        
        # Save chart
        chart_path = self.charts_path / f"{report_id}_system_health.{self.chart_format}"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    async def _export_report(self, report: AnalyticsReport):
        """Export the analytics report to file."""
        
        # Convert report to dictionary
        report_data = {
            'report_id': report.report_id,
            'timestamp': report.timestamp.isoformat(),
            'time_period': report.time_period,
            'summary': report.summary,
            'error_analysis': report.error_analysis,
            'recovery_analysis': report.recovery_analysis,
            'escalation_analysis': report.escalation_analysis,
            'performance_metrics': report.performance_metrics,
            'recommendations': report.recommendations,
            'charts': report.charts
        }
        
        # Export as JSON
        json_path = self.reports_path / f"{report.report_id}.json"
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Export as human-readable text
        text_path = self.reports_path / f"{report.report_id}.txt"
        with open(text_path, 'w') as f:
            f.write(f"Error Analytics Report\n")
            f.write(f"Report ID: {report.report_id}\n")
            f.write(f"Generated: {report.timestamp}\n")
            f.write(f"Time Period: {report.time_period}\n")
            f.write(f"{'='*50}\n\n")
            
            f.write("SUMMARY\n")
            f.write("-" * 20 + "\n")
            for key, value in report.summary.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")
            
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 20 + "\n")
            for i, rec in enumerate(report.recommendations, 1):
                f.write(f"{i}. {rec}\n")
            f.write("\n")
            
            f.write("DETAILED ANALYSIS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Error Analysis: {json.dumps(report.error_analysis, indent=2)}\n\n")
            f.write(f"Recovery Analysis: {json.dumps(report.recovery_analysis, indent=2)}\n\n")
            f.write(f"Escalation Analysis: {json.dumps(report.escalation_analysis, indent=2)}\n\n")
            f.write(f"Performance Metrics: {json.dumps(report.performance_metrics, indent=2)}\n\n")
        
        logger.info(f"Analytics report exported: {json_path}")
    
    async def calculate_system_health(
        self,
        error_history: List[Dict[str, Any]],
        recovery_history: List[RecoveryExecution],
        escalation_history: List[EscalationEvent],
        circuit_breaker_states: Dict[str, str]
    ) -> SystemHealthMetrics:
        """Calculate current system health metrics."""
        
        now = datetime.now()
        recent_window = timedelta(hours=1)
        cutoff_time = now - recent_window
        
        # Filter recent data
        recent_errors = [e for e in error_history if e.get('timestamp', datetime.min) > cutoff_time]
        recent_recoveries = [r for r in recovery_history if r.start_time > cutoff_time]
        recent_escalations = [e for e in escalation_history if e.timestamp > cutoff_time]
        
        # Calculate metrics
        error_rate = len(recent_errors) / 60.0  # errors per minute
        
        if recent_recoveries:
            recovery_success_rate = sum(1 for r in recent_recoveries if r.success) / len(recent_recoveries)
            recovery_times = [r.metrics.get('duration_seconds', 0) for r in recent_recoveries if r.success]
            avg_recovery_time = statistics.mean(recovery_times) if recovery_times else 0.0
        else:
            recovery_success_rate = 1.0
            avg_recovery_time = 0.0
        
        active_escalations = len([e for e in recent_escalations if not e.resolved])
        circuit_breaker_count = sum(1 for state in circuit_breaker_states.values() if state == 'open')
        
        # Calculate overall health score (0-100)
        health_score = 100.0
        
        # Deduct points for various issues
        health_score -= min(error_rate * 5, 30)  # Max 30 points for error rate
        health_score -= (1 - recovery_success_rate) * 40  # Max 40 points for recovery failures
        health_score -= min(active_escalations * 10, 20)  # Max 20 points for escalations
        health_score -= min(circuit_breaker_count * 5, 10)  # Max 10 points for circuit breakers
        
        health_score = max(0.0, health_score)
        
        # Determine system load
        if error_rate > 5:
            system_load = "high"
        elif error_rate > 2:
            system_load = "medium"
        else:
            system_load = "low"
        
        # Determine trending
        if len(self.health_history) > 0:
            last_health = self.health_history[-1].overall_health_score
            if health_score > last_health + 5:
                trending = "improving"
            elif health_score < last_health - 5:
                trending = "degrading"
            else:
                trending = "stable"
        else:
            trending = "stable"
        
        health_metrics = SystemHealthMetrics(
            timestamp=now,
            overall_health_score=health_score,
            error_rate=error_rate,
            recovery_success_rate=recovery_success_rate,
            average_recovery_time=avg_recovery_time,
            active_escalations=active_escalations,
            circuit_breaker_count=circuit_breaker_count,
            system_load=system_load,
            trending=trending
        )
        
        # Add to history
        self.health_history.append(health_metrics)
        if len(self.health_history) > self.max_health_history:
            self.health_history.pop(0)
        
        return health_metrics
    
    def get_health_history(self, hours: int = 24) -> List[SystemHealthMetrics]:
        """Get system health history for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [h for h in self.health_history if h.timestamp > cutoff_time]
    
    async def cleanup_old_reports(self, days: int = None):
        """Clean up old analytics reports and charts."""
        if days is None:
            days = self.retention_days
        
        cutoff_time = datetime.now() - timedelta(days=days)
        
        # Clean up reports
        for report_file in self.reports_path.glob("*.json"):
            if report_file.stat().st_mtime < cutoff_time.timestamp():
                report_file.unlink()
                logger.debug(f"Deleted old report: {report_file}")
        
        # Clean up charts
        for chart_file in self.charts_path.glob(f"*.{self.chart_format}"):
            if chart_file.stat().st_mtime < cutoff_time.timestamp():
                chart_file.unlink()
                logger.debug(f"Deleted old chart: {chart_file}")
        
        logger.info(f"Cleaned up analytics files older than {days} days")
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get summary of analytics system status."""
        return {
            'reports_generated': len(list(self.reports_path.glob("*.json"))),
            'charts_generated': len(list(self.charts_path.glob(f"*.{self.chart_format}"))),
            'health_history_size': len(self.health_history),
            'reports_path': str(self.reports_path),
            'charts_path': str(self.charts_path),
            'chart_generation_enabled': self.generate_charts,
            'retention_days': self.retention_days
        }
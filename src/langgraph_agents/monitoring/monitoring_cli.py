"""
Command-line interface for agent execution monitoring.

This module provides a comprehensive CLI for monitoring agent execution,
viewing performance metrics, analyzing errors, and managing monitoring systems.
"""

import asyncio
import click
import json
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from tabulate import tabulate
import logging

from .execution_monitor import get_execution_monitor, ExecutionStatus
from .performance_tracker import get_performance_tracker
from .error_detector import get_error_detector
from .execution_history import get_execution_history
from .real_time_monitor import get_real_time_monitor

logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose: bool) -> None:
    """Agent Execution Monitoring CLI."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


@cli.group()
def monitor() -> None:
    """Real-time monitoring commands."""
    pass


@monitor.command()
@click.option('--port', '-p', default=8765, help='WebSocket server port')
@click.option('--interval', '-i', default=1.0, help='Update interval in seconds')
def start(port: int, interval: float) -> None:
    """Start real-time monitoring server."""
    async def start_monitoring():
        monitor = get_real_time_monitor()
        monitor.websocket_port = port
        monitor.update_interval = interval
        
        click.echo(f"Starting real-time monitoring on port {port}...")
        await monitor.start_monitoring()
        
        try:
            # Keep running until interrupted
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            click.echo("Stopping monitoring...")
            await monitor.stop_monitoring()
    
    asyncio.run(start_monitoring())


@monitor.command()
def status() -> None:
    """Show current monitoring status."""
    execution_monitor = get_execution_monitor()
    performance_tracker = get_performance_tracker()
    error_detector = get_error_detector()
    
    # Get active executions
    active_executions = execution_monitor.get_active_executions()
    
    # Get monitoring summary
    monitoring_summary = execution_monitor.get_monitoring_summary()
    
    # Get performance summary
    performance_summary = performance_tracker.get_performance_summary()
    
    # Get error statistics
    error_stats = error_detector.get_error_statistics(time_window_hours=1)
    
    click.echo("=== Agent Execution Monitoring Status ===\n")
    
    # Active executions
    click.echo(f"Active Executions: {len(active_executions)}")
    if active_executions:
        headers = ["Agent", "Session ID", "Status", "Duration", "Current Step"]
        rows = []
        for tracker in active_executions:
            duration = tracker.get_execution_duration()
            duration_str = f"{duration:.1f}s" if duration else "N/A"
            rows.append([
                tracker.agent_name,
                tracker.session_id[:8] + "...",
                tracker.current_status.value,
                duration_str,
                tracker.current_step or "N/A"
            ])
        click.echo(tabulate(rows, headers=headers, tablefmt="grid"))
    
    click.echo(f"\nMonitoring Active: {monitoring_summary.get('monitoring_active', False)}")
    click.echo(f"Total Events: {monitoring_summary.get('total_events', 0)}")
    
    # Performance metrics
    click.echo(f"\n=== Performance Metrics ===")
    click.echo(f"Total Executions: {performance_summary.get('total_executions', 0)}")
    click.echo(f"Average Execution Time: {performance_summary.get('avg_execution_time', 0):.2f}s")
    click.echo(f"Total API Calls: {performance_summary.get('total_api_calls', 0)}")
    click.echo(f"Total Tokens Used: {performance_summary.get('total_tokens', 0)}")
    
    # Error statistics
    click.echo(f"\n=== Error Statistics (Last Hour) ===")
    click.echo(f"Total Errors: {error_stats.get('total_errors', 0)}")
    click.echo(f"Error Rate: {error_stats.get('error_rate_per_hour', 0):.2f} errors/hour")
    
    if error_stats.get('errors_by_category'):
        click.echo("\nErrors by Category:")
        for category, count in error_stats['errors_by_category'].items():
            click.echo(f"  {category}: {count}")


@monitor.command()
@click.option('--agent', '-a', help='Filter by agent name')
@click.option('--limit', '-l', default=20, help='Number of events to show')
def events(agent: Optional[str], limit: int) -> None:
    """Show recent execution events."""
    execution_monitor = get_execution_monitor()
    
    if agent:
        events_list = execution_monitor.get_agent_events(agent, limit)
        click.echo(f"=== Recent Events for {agent} ===")
    else:
        events_list = execution_monitor.get_recent_events(limit)
        click.echo("=== Recent Execution Events ===")
    
    if not events_list:
        click.echo("No events found.")
        return
    
    headers = ["Timestamp", "Agent", "Event Type", "Status", "Message"]
    rows = []
    
    for event in events_list[-limit:]:
        timestamp = event.timestamp.strftime("%H:%M:%S")
        rows.append([
            timestamp,
            event.agent_name,
            event.event_type,
            event.status.value,
            event.message[:50] + "..." if len(event.message) > 50 else event.message
        ])
    
    click.echo(tabulate(rows, headers=headers, tablefmt="grid"))


@cli.group()
def performance() -> None:
    """Performance monitoring commands."""
    pass


@performance.command()
@click.option('--agent', '-a', help='Filter by agent name')
@click.option('--hours', '-h', default=24, help='Time window in hours')
def summary(agent: Optional[str], hours: int) -> None:
    """Show performance summary."""
    performance_tracker = get_performance_tracker()
    
    if agent:
        summary_data = performance_tracker.get_performance_summary(agent)
        click.echo(f"=== Performance Summary for {agent} (Last {hours}h) ===")
    else:
        summary_data = performance_tracker.get_performance_summary()
        click.echo(f"=== Overall Performance Summary (Last {hours}h) ===")
    
    if summary_data.get('total_executions', 0) == 0:
        click.echo("No execution data available.")
        return
    
    click.echo(f"Total Executions: {summary_data.get('executions', summary_data.get('total_executions', 0))}")
    click.echo(f"Average Execution Time: {summary_data.get('avg_execution_time', 0):.2f}s")
    click.echo(f"Min Execution Time: {summary_data.get('min_execution_time', 0):.2f}s")
    click.echo(f"Max Execution Time: {summary_data.get('max_execution_time', 0):.2f}s")
    click.echo(f"Average Memory Usage: {summary_data.get('avg_memory_mb', 0):.1f}MB")
    click.echo(f"Peak Memory Usage: {summary_data.get('peak_memory_mb', 0):.1f}MB")
    click.echo(f"Average CPU Usage: {summary_data.get('avg_cpu_percent', 0):.1f}%")
    click.echo(f"Peak CPU Usage: {summary_data.get('peak_cpu_percent', 0):.1f}%")
    click.echo(f"Total API Calls: {summary_data.get('total_api_calls', 0)}")
    click.echo(f"Total Tokens: {summary_data.get('total_tokens', 0)}")
    click.echo(f"Total Errors: {summary_data.get('total_errors', 0)}")
    click.echo(f"Average Cache Hit Rate: {summary_data.get('avg_cache_hit_rate', 0):.2%}")


@performance.command()
@click.option('--limit', '-l', default=10, help='Number of alerts to show')
def alerts(limit: int) -> None:
    """Show recent performance alerts."""
    performance_tracker = get_performance_tracker()
    alerts_list = performance_tracker.get_recent_alerts(limit)
    
    click.echo("=== Recent Performance Alerts ===")
    
    if not alerts_list:
        click.echo("No performance alerts found.")
        return
    
    headers = ["Timestamp", "Agent", "Alert Type", "Level", "Message"]
    rows = []
    
    for alert in alerts_list[-limit:]:
        timestamp = datetime.fromisoformat(alert['timestamp']).strftime("%m-%d %H:%M:%S")
        rows.append([
            timestamp,
            alert['agent_name'],
            alert['alert_type'],
            alert['level'],
            alert['message'][:60] + "..." if len(alert['message']) > 60 else alert['message']
        ])
    
    click.echo(tabulate(rows, headers=headers, tablefmt="grid"))


@cli.group()
def errors() -> None:
    """Error monitoring commands."""
    pass


@errors.command()
@click.option('--agent', '-a', help='Filter by agent name')
@click.option('--hours', '-h', default=24, help='Time window in hours')
def stats(agent: Optional[str], hours: int) -> None:
    """Show error statistics."""
    error_detector = get_error_detector()
    error_stats = error_detector.get_error_statistics(agent, hours)
    
    if agent:
        click.echo(f"=== Error Statistics for {agent} (Last {hours}h) ===")
    else:
        click.echo(f"=== Overall Error Statistics (Last {hours}h) ===")
    
    click.echo(f"Total Errors: {error_stats.get('total_errors', 0)}")
    click.echo(f"Error Rate: {error_stats.get('error_rate_per_hour', 0):.2f} errors/hour")
    
    if error_stats.get('errors_by_category'):
        click.echo("\nErrors by Category:")
        for category, count in error_stats['errors_by_category'].items():
            click.echo(f"  {category.title()}: {count}")
    
    if error_stats.get('errors_by_severity'):
        click.echo("\nErrors by Severity:")
        for severity, count in error_stats['errors_by_severity'].items():
            click.echo(f"  {severity.title()}: {count}")
    
    if error_stats.get('errors_by_pattern'):
        click.echo("\nTop Error Patterns:")
        sorted_patterns = sorted(
            error_stats['errors_by_pattern'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for pattern, count in sorted_patterns[:5]:
            click.echo(f"  {pattern}: {count}")


@errors.command()
@click.option('--limit', '-l', default=10, help='Number of alerts to show')
def alerts(limit: int) -> None:
    """Show recent error alerts."""
    error_detector = get_error_detector()
    alerts_list = error_detector.get_recent_alerts(limit)
    
    click.echo("=== Recent Error Alerts ===")
    
    if not alerts_list:
        click.echo("No error alerts found.")
        return
    
    headers = ["Timestamp", "Agent", "Severity", "Category", "Message"]
    rows = []
    
    for alert in alerts_list[-limit:]:
        timestamp = datetime.fromisoformat(alert['timestamp']).strftime("%m-%d %H:%M:%S")
        rows.append([
            timestamp,
            alert['agent_name'],
            alert['severity'],
            alert['category'],
            alert['error_message'][:50] + "..." if len(alert['error_message']) > 50 else alert['error_message']
        ])
    
    click.echo(tabulate(rows, headers=headers, tablefmt="grid"))


@errors.command()
@click.option('--hours', '-h', default=24, help='Time window in hours')
def trends(hours: int) -> None:
    """Show error trends analysis."""
    error_detector = get_error_detector()
    trends_data = error_detector.analyze_error_trends(hours)
    
    click.echo(f"=== Error Trends Analysis (Last {hours}h) ===")
    
    click.echo(f"Trend: {trends_data.get('trend', 'unknown').title()}")
    click.echo(f"Hours Analyzed: {trends_data.get('total_hours_analyzed', 0)}")
    
    if trends_data.get('hourly_errors'):
        click.echo("\nHourly Error Counts:")
        for hour, count in list(trends_data['hourly_errors'].items())[-12:]:  # Last 12 hours
            hour_display = datetime.fromisoformat(hour + ":00").strftime("%m-%d %H:%M")
            click.echo(f"  {hour_display}: {count} errors")


@cli.group()
def history() -> None:
    """Execution history commands."""
    pass


@history.command()
@click.option('--agent', '-a', help='Filter by agent name')
@click.option('--status', '-s', help='Filter by status')
@click.option('--limit', '-l', default=10, help='Number of records to show')
def list(agent: Optional[str], status: Optional[str], limit: int) -> None:
    """List execution history records."""
    execution_history = get_execution_history()
    
    # Convert status string to enum if provided
    status_filter = None
    if status:
        try:
            status_filter = ExecutionStatus(status.lower())
        except ValueError:
            click.echo(f"Invalid status: {status}")
            click.echo(f"Valid statuses: {', '.join([s.value for s in ExecutionStatus])}")
            return
    
    records = execution_history.search_records(
        agent_name=agent,
        status=status_filter,
        limit=limit
    )
    
    if agent:
        click.echo(f"=== Execution History for {agent} ===")
    else:
        click.echo("=== Execution History ===")
    
    if not records:
        click.echo("No execution records found.")
        return
    
    headers = ["Record ID", "Agent", "Status", "Start Time", "Duration", "Errors"]
    rows = []
    
    for record in records:
        start_time = record.start_time.strftime("%m-%d %H:%M:%S")
        duration = f"{record.total_duration:.1f}s" if record.total_duration else "N/A"
        rows.append([
            record.record_id[:12] + "...",
            record.agent_name,
            record.final_status.value,
            start_time,
            duration,
            len(record.errors)
        ])
    
    click.echo(tabulate(rows, headers=headers, tablefmt="grid"))


@history.command()
@click.argument('record_id')
def show(record_id: str) -> None:
    """Show detailed information for a specific execution record."""
    execution_history = get_execution_history()
    record = execution_history.get_record(record_id)
    
    if not record:
        click.echo(f"Record not found: {record_id}")
        return
    
    click.echo(f"=== Execution Record: {record.record_id} ===")
    click.echo(f"Agent: {record.agent_name}")
    click.echo(f"Session ID: {record.session_id}")
    click.echo(f"Status: {record.final_status.value}")
    click.echo(f"Start Time: {record.start_time}")
    click.echo(f"End Time: {record.end_time}")
    click.echo(f"Duration: {record.total_duration:.2f}s" if record.total_duration else "N/A")
    click.echo(f"Processing Time: {record.processing_time:.2f}s")
    click.echo(f"Events: {len(record.events)}")
    click.echo(f"Errors: {len(record.errors)}")
    click.echo(f"Warnings: {len(record.warnings)}")
    click.echo(f"Retries: {record.retry_count}")
    
    if record.step_times:
        click.echo("\nStep Times:")
        for step, duration in record.step_times.items():
            click.echo(f"  {step}: {duration:.2f}s")
    
    if record.errors:
        click.echo("\nErrors:")
        for i, error in enumerate(record.errors, 1):
            click.echo(f"  {i}. {error}")


@history.command()
@click.option('--agent', '-a', help='Filter by agent name')
@click.option('--days', '-d', default=7, help='Time window in days')
def stats(agent: Optional[str], days: int) -> None:
    """Show execution statistics."""
    execution_history = get_execution_history()
    stats_data = execution_history.get_execution_statistics(agent, days)
    
    if agent:
        click.echo(f"=== Execution Statistics for {agent} (Last {days} days) ===")
    else:
        click.echo(f"=== Overall Execution Statistics (Last {days} days) ===")
    
    if stats_data.get('total_executions', 0) == 0:
        click.echo("No execution data available.")
        return
    
    click.echo(f"Total Executions: {stats_data.get('total_executions', 0)}")
    click.echo(f"Successful Executions: {stats_data.get('successful_executions', 0)}")
    click.echo(f"Failed Executions: {stats_data.get('failed_executions', 0)}")
    click.echo(f"Success Rate: {stats_data.get('success_rate', 0):.2%}")
    click.echo(f"Average Duration: {stats_data.get('avg_duration', 0):.2f}s")
    click.echo(f"Min Duration: {stats_data.get('min_duration', 0):.2f}s")
    click.echo(f"Max Duration: {stats_data.get('max_duration', 0):.2f}s")
    click.echo(f"Average Processing Time: {stats_data.get('avg_processing_time', 0):.2f}s")
    click.echo(f"Total Errors: {stats_data.get('total_errors', 0)}")
    click.echo(f"Total Retries: {stats_data.get('total_retries', 0)}")
    click.echo(f"Agents Involved: {stats_data.get('agents_involved', 0)}")


@cli.command()
@click.option('--format', '-f', default='json', type=click.Choice(['json', 'table']), help='Output format')
def dashboard(format: str) -> None:
    """Show monitoring dashboard."""
    real_time_monitor = get_real_time_monitor()
    dashboard_data = real_time_monitor.get_current_dashboard()
    
    if format == 'json':
        click.echo(json.dumps(dashboard_data, indent=2, default=str))
    else:
        click.echo("=== Monitoring Dashboard ===")
        click.echo(f"Timestamp: {dashboard_data['timestamp']}")
        click.echo(f"Active Executions: {len(dashboard_data['active_executions'])}")
        click.echo(f"System Health Score: {dashboard_data['system_health_score']:.1f}")
        click.echo(f"Average Execution Time: {dashboard_data['avg_execution_time']:.2f}s")
        click.echo(f"Peak Memory Usage: {dashboard_data['peak_memory_usage']:.1f}MB")
        click.echo(f"Error Count (Last Hour): {dashboard_data['error_count_last_hour']}")
        click.echo(f"Critical Alerts: {dashboard_data['critical_alerts']}")
        
        if dashboard_data['execution_count_by_status']:
            click.echo("\nExecutions by Status:")
            for status, count in dashboard_data['execution_count_by_status'].items():
                click.echo(f"  {status}: {count}")
        
        if dashboard_data['execution_count_by_agent']:
            click.echo("\nExecutions by Agent:")
            for agent, count in dashboard_data['execution_count_by_agent'].items():
                click.echo(f"  {agent}: {count}")


if __name__ == '__main__':
    cli()
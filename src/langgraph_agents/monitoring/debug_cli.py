"""
Interactive debugging CLI for agent execution.

This module provides an interactive command-line interface for debugging
agent execution with breakpoints, state inspection, and flow analysis.
"""

import click
import json
import sys
from datetime import datetime
from typing import Optional, List
from tabulate import tabulate

from .debugging_tools import (
    get_agent_debugger,
    get_state_inspector,
    get_flow_analyzer,
    DebugLevel,
    InspectionType
)

@click.group()
def debug_cli() -> None:
    """Interactive Agent Debugging CLI."""
    pass


@debug_cli.group()
def breakpoint() -> None:
    """Breakpoint management commands."""
    pass


@breakpoint.command()
@click.argument('agent_name')
@click.option('--step', '-s', help='Specific step name to break on')
@click.option('--condition', '-c', help='Python condition expression')
def add(agent_name: str, step: Optional[str], condition: Optional[str]) -> None:
    """Add a debugging breakpoint."""
    debugger = get_agent_debugger()
    
    breakpoint_id = debugger.add_breakpoint(
        agent_name=agent_name,
        step_name=step,
        condition=condition
    )
    
    click.echo(f"Added breakpoint: {breakpoint_id}")
    click.echo(f"  Agent: {agent_name}")
    if step:
        click.echo(f"  Step: {step}")
    if condition:
        click.echo(f"  Condition: {condition}")


@breakpoint.command()
@click.argument('breakpoint_id')
def remove(breakpoint_id: str) -> None:
    """Remove a breakpoint."""
    debugger = get_agent_debugger()
    
    if debugger.remove_breakpoint(breakpoint_id):
        click.echo(f"Removed breakpoint: {breakpoint_id}")
    else:
        click.echo(f"Breakpoint not found: {breakpoint_id}")


@breakpoint.command()
@click.argument('breakpoint_id')
@click.option('--disable', is_flag=True, help='Disable instead of enable')
def toggle(breakpoint_id: str, disable: bool) -> None:
    """Enable or disable a breakpoint."""
    debugger = get_agent_debugger()
    
    enabled = not disable
    if debugger.enable_breakpoint(breakpoint_id, enabled):
        status = "disabled" if disable else "enabled"
        click.echo(f"Breakpoint {breakpoint_id} {status}")
    else:
        click.echo(f"Breakpoint not found: {breakpoint_id}")


@breakpoint.command()
def list() -> None:
    """List all breakpoints."""
    debugger = get_agent_debugger()
    
    if not debugger.breakpoints:
        click.echo("No breakpoints configured.")
        return
    
    headers = ["ID", "Agent", "Step", "Condition", "Enabled", "Hit Count"]
    rows = []
    
    for bp_id, bp in debugger.breakpoints.items():
        rows.append([
            bp_id,
            bp.agent_name,
            bp.step_name or "Any",
            bp.condition or "None",
            "Yes" if bp.enabled else "No",
            bp.hit_count
        ])
    
    click.echo("=== Configured Breakpoints ===")
    click.echo(tabulate(rows, headers=headers, tablefmt="grid"))


@debug_cli.group()
def state() -> None:
    """State inspection commands."""
    pass


@state.command()
@click.option('--agent', '-a', help='Filter by agent name')
@click.option('--session', '-s', help='Filter by session ID')
@click.option('--step', help='Filter by step name')
@click.option('--type', 'state_type', help='Filter by state type')
@click.option('--limit', '-l', default=20, help='Number of snapshots to show')
def list(agent: Optional[str], session: Optional[str], step: Optional[str], 
         state_type: Optional[str], limit: int) -> None:
    """List state snapshots."""
    inspector = get_state_inspector()
    
    # Convert state type string to enum
    type_filter = None
    if state_type:
        try:
            type_filter = InspectionType(state_type.lower())
        except ValueError:
            click.echo(f"Invalid state type: {state_type}")
            click.echo(f"Valid types: {', '.join([t.value for t in InspectionType])}")
            return
    
    snapshots = inspector.search_snapshots(
        agent_name=agent,
        session_id=session,
        step_name=step,
        state_type=type_filter,
        limit=limit
    )
    
    if not snapshots:
        click.echo("No state snapshots found.")
        return
    
    headers = ["Snapshot ID", "Agent", "Step", "Type", "Timestamp", "Keys"]
    rows = []
    
    for snapshot in snapshots:
        timestamp = snapshot.timestamp.strftime("%m-%d %H:%M:%S")
        rows.append([
            snapshot.snapshot_id[:12] + "...",
            snapshot.agent_name,
            snapshot.step_name,
            snapshot.state_type.value,
            timestamp,
            len(snapshot.state_data)
        ])
    
    click.echo("=== State Snapshots ===")
    click.echo(tabulate(rows, headers=headers, tablefmt="grid"))


@state.command()
@click.argument('snapshot_id')
@click.option('--format', '-f', default='summary', type=click.Choice(['summary', 'full', 'json']))
def show(snapshot_id: str, format: str) -> None:
    """Show detailed state snapshot information."""
    inspector = get_state_inspector()
    snapshot = inspector.get_snapshot(snapshot_id)
    
    if not snapshot:
        click.echo(f"Snapshot not found: {snapshot_id}")
        return
    
    if format == 'json':
        click.echo(json.dumps(snapshot.to_dict(), indent=2, default=str))
        return
    
    click.echo(f"=== State Snapshot: {snapshot.snapshot_id} ===")
    click.echo(f"Agent: {snapshot.agent_name}")
    click.echo(f"Session: {snapshot.session_id}")
    click.echo(f"Step: {snapshot.step_name}")
    click.echo(f"Type: {snapshot.state_type.value}")
    click.echo(f"Timestamp: {snapshot.timestamp}")
    
    if format == 'full':
        click.echo(f"\n=== State Data ===")
        for key, value in snapshot.state_data.items():
            value_str = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
            click.echo(f"{key}: {value_str}")
        
        if snapshot.variables:
            click.echo(f"\n=== Variables ===")
            for key, value in snapshot.variables.items():
                value_str = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                click.echo(f"{key}: {value_str}")
        
        if snapshot.call_stack:
            click.echo(f"\n=== Call Stack ===")
            for i, frame in enumerate(snapshot.call_stack):
                click.echo(f"{i}: {frame}")
    else:
        click.echo(f"Data Keys: {list(snapshot.state_data.keys())}")
        click.echo(f"Variables: {len(snapshot.variables)}")
        click.echo(f"Stack Depth: {len(snapshot.call_stack)}")
        
        if snapshot.memory_usage:
            click.echo(f"Memory Usage: {snapshot.memory_usage:.1f}MB")


@state.command()
@click.argument('snapshot_id1')
@click.argument('snapshot_id2')
def compare(snapshot_id1: str, snapshot_id2: str) -> None:
    """Compare two state snapshots."""
    inspector = get_state_inspector()
    comparison = inspector.compare_snapshots(snapshot_id1, snapshot_id2)
    
    if 'error' in comparison:
        click.echo(f"Error: {comparison['error']}")
        return
    
    click.echo("=== Snapshot Comparison ===")
    click.echo(f"Snapshot 1: {comparison['snapshot1']['snapshot_id']}")
    click.echo(f"Snapshot 2: {comparison['snapshot2']['snapshot_id']}")
    
    if comparison['added_keys']:
        click.echo(f"\nAdded Keys: {comparison['added_keys']}")
    
    if comparison['removed_keys']:
        click.echo(f"Removed Keys: {comparison['removed_keys']}")
    
    if comparison['changed_values']:
        click.echo("\nChanged Values:")
        for key, change in comparison['changed_values'].items():
            click.echo(f"  {key}:")
            click.echo(f"    Before: {change['before']}")
            click.echo(f"    After: {change['after']}")


@state.command()
@click.argument('session_id')
def evolution(session_id: str) -> None:
    """Analyze state evolution for a session."""
    inspector = get_state_inspector()
    analysis = inspector.analyze_state_evolution(session_id)
    
    if 'error' in analysis:
        click.echo(f"Error: {analysis['error']}")
        return
    
    click.echo(f"=== State Evolution for Session {session_id} ===")
    click.echo(f"Total Snapshots: {analysis['total_snapshots']}")
    click.echo(f"Time Span: {analysis['time_span']:.2f} seconds")
    
    click.echo("\n=== Steps ===")
    for step in analysis['steps']:
        click.echo(f"- {step['step_name']}: {step['data_keys']} keys, {step['variables']} variables")
    
    if analysis['memory_trend']:
        avg_memory = sum(analysis['memory_trend']) / len(analysis['memory_trend'])
        click.echo(f"\nAverage Memory Usage: {avg_memory:.1f}MB")
    
    if analysis['complexity_trend']:
        avg_complexity = sum(analysis['complexity_trend']) / len(analysis['complexity_trend'])
        click.echo(f"Average Complexity: {avg_complexity:.1f}")


@debug_cli.group()
def flow() -> None:
    """Execution flow analysis commands."""
    pass


@flow.command()
@click.argument('session_id')
@click.option('--format', '-f', default='summary', type=click.Choice(['summary', 'detailed', 'json']))
def analyze(session_id: str, format: str) -> None:
    """Analyze execution flow for a session."""
    analyzer = get_flow_analyzer()
    analysis = analyzer.analyze_execution_flow(session_id)
    
    if 'error' in analysis:
        click.echo(f"Error: {analysis['error']}")
        return
    
    if format == 'json':
        click.echo(json.dumps(analysis, indent=2, default=str))
        return
    
    click.echo(f"=== Execution Flow Analysis: {session_id} ===")
    click.echo(f"Agent: {analysis['agent_name']}")
    click.echo(f"Status: {analysis['final_status']}")
    click.echo(f"Duration: {analysis['total_duration']:.2f}s")
    
    metrics = analysis['performance_metrics']
    click.echo(f"Total Steps: {metrics['total_steps']}")
    click.echo(f"Average Step Duration: {metrics['avg_step_duration']:.2f}s")
    click.echo(f"Errors: {metrics['error_count']}")
    click.echo(f"Bottlenecks: {metrics['bottleneck_count']}")
    
    if format == 'detailed':
        if analysis['flow_steps']:
            click.echo("\n=== Flow Steps ===")
            headers = ["Step", "Duration", "Start Time"]
            rows = []
            for step in analysis['flow_steps']:
                start_time = datetime.fromisoformat(step['start_time']).strftime("%H:%M:%S")
                rows.append([
                    step['step_name'],
                    f"{step['duration']:.2f}s",
                    start_time
                ])
            click.echo(tabulate(rows, headers=headers, tablefmt="grid"))
        
        if analysis['bottlenecks']:
            click.echo("\n=== Bottlenecks ===")
            for bottleneck in analysis['bottlenecks']:
                click.echo(f"- {bottleneck['step_name']}: {bottleneck['duration']:.2f}s ({bottleneck['percentage']:.1f}%)")
        
        if analysis['error_points']:
            click.echo("\n=== Error Points ===")
            for error in analysis['error_points']:
                timestamp = datetime.fromisoformat(error['timestamp']).strftime("%H:%M:%S")
                click.echo(f"- [{timestamp}] {error['step']}: {error['error']}")


@flow.command()
@click.argument('session_ids', nargs=-1, required=True)
def compare(session_ids: List[str]) -> None:
    """Compare execution flows across multiple sessions."""
    analyzer = get_flow_analyzer()
    comparison = analyzer.compare_execution_flows(list(session_ids))
    
    if 'error' in comparison:
        click.echo(f"Error: {comparison['error']}")
        return
    
    click.echo("=== Execution Flow Comparison ===")
    click.echo(f"Sessions Compared: {comparison['sessions_compared']}")
    click.echo(f"Average Duration: {comparison['average_duration']:.2f}s")
    
    if comparison['common_steps']:
        click.echo(f"\nCommon Steps: {', '.join(comparison['common_steps'])}")
        
        click.echo("\n=== Step Duration Comparison ===")
        headers = ["Step", "Avg Duration", "Min", "Max", "Variance"]
        rows = []
        
        for step, stats in comparison['step_duration_comparison'].items():
            rows.append([
                step,
                f"{stats['avg_duration']:.2f}s",
                f"{stats['min_duration']:.2f}s",
                f"{stats['max_duration']:.2f}s",
                f"{stats['variance']:.2f}s"
            ])
        
        click.echo(tabulate(rows, headers=headers, tablefmt="grid"))
    
    if comparison['bottleneck_patterns']:
        click.echo("\n=== Bottleneck Patterns ===")
        for step, count in comparison['bottleneck_patterns'].items():
            click.echo(f"- {step}: {count} occurrences")
    
    if comparison['error_patterns']:
        click.echo("\n=== Error Patterns ===")
        for step, count in comparison['error_patterns'].items():
            click.echo(f"- {step}: {count} occurrences")


@flow.command()
@click.argument('session_id')
@click.option('--format', '-f', default='mermaid', type=click.Choice(['mermaid']))
@click.option('--output', '-o', help='Output file path')
def diagram(session_id: str, format: str, output: Optional[str]) -> None:
    """Generate execution flow diagram."""
    analyzer = get_flow_analyzer()
    diagram_text = analyzer.generate_flow_diagram(session_id, format)
    
    if output:
        with open(output, 'w') as f:
            f.write(diagram_text)
        click.echo(f"Flow diagram saved to: {output}")
    else:
        click.echo("=== Execution Flow Diagram ===")
        click.echo(diagram_text)


@debug_cli.group()
def session() -> None:
    """Debug session management commands."""
    pass


@session.command()
@click.argument('session_id')
@click.option('--level', '-l', default='detailed', 
              type=click.Choice(['basic', 'detailed', 'verbose']))
def report(session_id: str, level: str) -> None:
    """Generate comprehensive debug report for a session."""
    debugger = get_agent_debugger()
    
    debug_level = DebugLevel(level)
    report_text = debugger.generate_debug_report(session_id, debug_level)
    
    click.echo(report_text)


@session.command()
@click.argument('session_id')
def summary(session_id: str) -> None:
    """Get debug summary for a session."""
    debugger = get_agent_debugger()
    summary = debugger.get_debug_summary(session_id)
    
    click.echo(f"=== Debug Summary: {session_id} ===")
    
    flow = summary['execution_flow']
    if 'error' not in flow:
        click.echo(f"Agent: {flow['agent_name']}")
        click.echo(f"Status: {flow['final_status']}")
        click.echo(f"Duration: {flow['total_duration']:.2f}s")
        click.echo(f"Steps: {flow['performance_metrics']['total_steps']}")
        click.echo(f"Errors: {flow['performance_metrics']['error_count']}")
        click.echo(f"Bottlenecks: {flow['performance_metrics']['bottleneck_count']}")
    
    click.echo(f"State Snapshots: {len(summary['state_snapshots'])}")
    click.echo(f"Breakpoints Configured: {summary['breakpoints_configured']}")
    click.echo(f"Active Breakpoints: {summary['active_breakpoints']}")


@debug_cli.command()
def status() -> None:
    """Show overall debugging system status."""
    debugger = get_agent_debugger()
    inspector = get_state_inspector()
    
    click.echo("=== Debug System Status ===")
    click.echo(f"Total Breakpoints: {len(debugger.breakpoints)}")
    
    enabled_bp = len([bp for bp in debugger.breakpoints.values() if bp.enabled])
    click.echo(f"Enabled Breakpoints: {enabled_bp}")
    
    total_hits = sum(bp.hit_count for bp in debugger.breakpoints.values())
    click.echo(f"Total Breakpoint Hits: {total_hits}")
    
    click.echo(f"Active Debug Sessions: {len(debugger.active_sessions)}")
    click.echo(f"State Snapshots: {len(inspector.snapshots)}")
    
    if debugger.active_sessions:
        click.echo("\n=== Active Debug Sessions ===")
        for session_id, session_data in debugger.active_sessions.items():
            duration = (datetime.now() - session_data['start_time']).total_seconds()
            click.echo(f"- {session_id}: {session_data['agent_name']} ({duration:.1f}s)")


if __name__ == '__main__':
    debug_cli()
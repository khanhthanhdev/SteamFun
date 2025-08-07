"""
Agent debugging and inspection tools.

This module provides comprehensive debugging capabilities for agent execution,
including state inspection, execution flow analysis, and interactive debugging.
"""

import json
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import inspect
import copy

from .execution_monitor import get_execution_monitor, ExecutionEvent, AgentExecutionTracker
from .execution_history import get_execution_history, ExecutionRecord

logger = logging.getLogger(__name__)


class DebugLevel(Enum):
    """Debug information detail levels."""
    BASIC = "basic"
    DETAILED = "detailed"
    VERBOSE = "verbose"


class InspectionType(Enum):
    """Types of state inspection."""
    INPUT = "input"
    OUTPUT = "output"
    INTERMEDIATE = "intermediate"
    ERROR = "error"
    FULL = "full"


@dataclass
class StateSnapshot:
    """Represents a snapshot of agent state at a specific point."""
    
    snapshot_id: str
    timestamp: datetime
    agent_name: str
    session_id: str
    step_name: str
    state_type: InspectionType
    
    # State data
    state_data: Dict[str, Any]
    variables: Dict[str, Any] = field(default_factory=dict)
    local_context: Dict[str, Any] = field(default_factory=dict)
    
    # Execution context
    call_stack: List[str] = field(default_factory=list)
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary."""
        return {
            'snapshot_id': self.snapshot_id,
            'timestamp': self.timestamp.isoformat(),
            'agent_name': self.agent_name,
            'session_id': self.session_id,
            'step_name': self.step_name,
            'state_type': self.state_type.value,
            'state_data': self.state_data,
            'variables': self.variables,
            'local_context': self.local_context,
            'call_stack': self.call_stack,
            'memory_usage': self.memory_usage,
            'cpu_usage': self.cpu_usage,
            'tags': self.tags,
            'notes': self.notes
        }
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the state snapshot."""
        return {
            'snapshot_id': self.snapshot_id,
            'timestamp': self.timestamp.isoformat(),
            'step_name': self.step_name,
            'state_type': self.state_type.value,
            'data_keys': list(self.state_data.keys()),
            'variable_count': len(self.variables),
            'context_keys': list(self.local_context.keys()),
            'stack_depth': len(self.call_stack),
            'memory_usage': self.memory_usage,
            'tags': self.tags
        }


@dataclass
class ExecutionBreakpoint:
    """Represents a debugging breakpoint."""
    
    breakpoint_id: str
    agent_name: str
    step_name: Optional[str] = None
    condition: Optional[str] = None  # Python expression
    hit_count: int = 0
    enabled: bool = True
    
    # Breakpoint actions
    capture_state: bool = True
    log_message: Optional[str] = None
    callback: Optional[Callable] = None
    
    def should_trigger(self, agent_name: str, step_name: str, 
                      context: Dict[str, Any]) -> bool:
        """Check if breakpoint should trigger."""
        if not self.enabled:
            return False
        
        if self.agent_name != agent_name:
            return False
        
        if self.step_name and self.step_name != step_name:
            return False
        
        if self.condition:
            try:
                # Evaluate condition in context
                return eval(self.condition, {"__builtins__": {}}, context)
            except Exception as e:
                logger.warning(f"Breakpoint condition evaluation failed: {e}")
                return False
        
        return True
    
    def trigger(self, context: Dict[str, Any]) -> None:
        """Trigger breakpoint actions."""
        self.hit_count += 1
        
        if self.log_message:
            logger.info(f"Breakpoint hit: {self.log_message}")
        
        if self.callback:
            try:
                self.callback(self, context)
            except Exception as e:
                logger.error(f"Breakpoint callback failed: {e}")


class StateInspector:
    """
    Comprehensive state inspection system for agent debugging.
    
    Provides detailed inspection of agent state at various execution points
    with support for filtering, comparison, and analysis.
    """
    
    def __init__(self, max_snapshots: int = 1000):
        """
        Initialize state inspector.
        
        Args:
            max_snapshots: Maximum number of snapshots to keep in memory
        """
        self.max_snapshots = max_snapshots
        
        # Snapshot storage
        self.snapshots: Dict[str, StateSnapshot] = {}
        self.session_snapshots: Dict[str, List[str]] = defaultdict(list)
        self.agent_snapshots: Dict[str, List[str]] = defaultdict(list)
        
        # Inspection configuration
        self.auto_capture_steps: List[str] = []
        self.capture_filters: Dict[str, Any] = {}
        
        logger.info("StateInspector initialized")
    
    def capture_state(self, agent_name: str, session_id: str, step_name: str,
                     state_data: Dict[str, Any], state_type: InspectionType = InspectionType.INTERMEDIATE,
                     variables: Optional[Dict[str, Any]] = None,
                     local_context: Optional[Dict[str, Any]] = None) -> StateSnapshot:
        """Capture a state snapshot."""
        snapshot_id = f"{agent_name}_{session_id}_{step_name}_{int(datetime.now().timestamp())}"
        
        # Get call stack
        call_stack = []
        frame = inspect.currentframe()
        try:
            while frame:
                if frame.f_code.co_filename != __file__:
                    call_stack.append(f"{frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}")
                frame = frame.f_back
        finally:
            del frame
        
        snapshot = StateSnapshot(
            snapshot_id=snapshot_id,
            timestamp=datetime.now(),
            agent_name=agent_name,
            session_id=session_id,
            step_name=step_name,
            state_type=state_type,
            state_data=copy.deepcopy(state_data),
            variables=copy.deepcopy(variables or {}),
            local_context=copy.deepcopy(local_context or {}),
            call_stack=call_stack[:10]  # Limit stack depth
        )
        
        # Store snapshot
        self.snapshots[snapshot_id] = snapshot
        self.session_snapshots[session_id].append(snapshot_id)
        self.agent_snapshots[agent_name].append(snapshot_id)
        
        # Cleanup old snapshots if needed
        self._cleanup_old_snapshots()
        
        logger.debug(f"Captured state snapshot: {snapshot_id}")
        return snapshot
    
    def get_snapshot(self, snapshot_id: str) -> Optional[StateSnapshot]:
        """Get a specific state snapshot."""
        return self.snapshots.get(snapshot_id)
    
    def get_session_snapshots(self, session_id: str, 
                            state_type: Optional[InspectionType] = None) -> List[StateSnapshot]:
        """Get all snapshots for a session."""
        snapshot_ids = self.session_snapshots.get(session_id, [])
        snapshots = []
        
        for snapshot_id in snapshot_ids:
            snapshot = self.snapshots.get(snapshot_id)
            if snapshot and (state_type is None or snapshot.state_type == state_type):
                snapshots.append(snapshot)
        
        return sorted(snapshots, key=lambda s: s.timestamp)
    
    def get_agent_snapshots(self, agent_name: str, limit: int = 50) -> List[StateSnapshot]:
        """Get recent snapshots for an agent."""
        snapshot_ids = self.agent_snapshots.get(agent_name, [])
        snapshots = []
        
        for snapshot_id in snapshot_ids[-limit:]:
            snapshot = self.snapshots.get(snapshot_id)
            if snapshot:
                snapshots.append(snapshot)
        
        return sorted(snapshots, key=lambda s: s.timestamp, reverse=True)
    
    def compare_snapshots(self, snapshot_id1: str, snapshot_id2: str) -> Dict[str, Any]:
        """Compare two state snapshots."""
        snapshot1 = self.snapshots.get(snapshot_id1)
        snapshot2 = self.snapshots.get(snapshot_id2)
        
        if not snapshot1 or not snapshot2:
            return {"error": "One or both snapshots not found"}
        
        comparison = {
            'snapshot1': snapshot1.get_state_summary(),
            'snapshot2': snapshot2.get_state_summary(),
            'differences': {},
            'added_keys': [],
            'removed_keys': [],
            'changed_values': {}
        }
        
        # Compare state data
        keys1 = set(snapshot1.state_data.keys())
        keys2 = set(snapshot2.state_data.keys())
        
        comparison['added_keys'] = list(keys2 - keys1)
        comparison['removed_keys'] = list(keys1 - keys2)
        
        # Compare common keys
        for key in keys1 & keys2:
            val1 = snapshot1.state_data[key]
            val2 = snapshot2.state_data[key]
            
            if val1 != val2:
                comparison['changed_values'][key] = {
                    'before': val1,
                    'after': val2
                }
        
        return comparison
    
    def search_snapshots(self, agent_name: Optional[str] = None,
                        session_id: Optional[str] = None,
                        step_name: Optional[str] = None,
                        state_type: Optional[InspectionType] = None,
                        contains_key: Optional[str] = None,
                        limit: int = 100) -> List[StateSnapshot]:
        """Search snapshots with filters."""
        results = []
        
        for snapshot in self.snapshots.values():
            # Apply filters
            if agent_name and snapshot.agent_name != agent_name:
                continue
            if session_id and snapshot.session_id != session_id:
                continue
            if step_name and snapshot.step_name != step_name:
                continue
            if state_type and snapshot.state_type != state_type:
                continue
            if contains_key and contains_key not in snapshot.state_data:
                continue
            
            results.append(snapshot)
        
        # Sort by timestamp (most recent first)
        results.sort(key=lambda s: s.timestamp, reverse=True)
        return results[:limit]
    
    def analyze_state_evolution(self, session_id: str) -> Dict[str, Any]:
        """Analyze how state evolves during a session."""
        snapshots = self.get_session_snapshots(session_id)
        
        if len(snapshots) < 2:
            return {"error": "Need at least 2 snapshots for evolution analysis"}
        
        analysis = {
            'session_id': session_id,
            'total_snapshots': len(snapshots),
            'time_span': (snapshots[-1].timestamp - snapshots[0].timestamp).total_seconds(),
            'steps': [],
            'key_evolution': defaultdict(list),
            'memory_trend': [],
            'complexity_trend': []
        }
        
        for i, snapshot in enumerate(snapshots):
            step_info = {
                'step_name': snapshot.step_name,
                'timestamp': snapshot.timestamp.isoformat(),
                'data_keys': len(snapshot.state_data),
                'variables': len(snapshot.variables),
                'memory_usage': snapshot.memory_usage
            }
            analysis['steps'].append(step_info)
            
            # Track key evolution
            for key, value in snapshot.state_data.items():
                analysis['key_evolution'][key].append({
                    'step': snapshot.step_name,
                    'value': str(value)[:100],  # Truncate long values
                    'type': type(value).__name__
                })
            
            # Track memory trend
            if snapshot.memory_usage:
                analysis['memory_trend'].append(snapshot.memory_usage)
            
            # Track complexity (number of keys + variables)
            complexity = len(snapshot.state_data) + len(snapshot.variables)
            analysis['complexity_trend'].append(complexity)
        
        return analysis
    
    def export_snapshots(self, snapshot_ids: List[str], format: str = "json") -> str:
        """Export snapshots to file."""
        snapshots_data = []
        for snapshot_id in snapshot_ids:
            snapshot = self.snapshots.get(snapshot_id)
            if snapshot:
                snapshots_data.append(snapshot.to_dict())
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"state_snapshots_{timestamp}.{format}"
        
        if format == "json":
            with open(filename, 'w') as f:
                json.dump(snapshots_data, f, indent=2, default=str)
        
        logger.info(f"Exported {len(snapshots_data)} snapshots to {filename}")
        return filename
    
    def _cleanup_old_snapshots(self) -> None:
        """Clean up old snapshots if limit exceeded."""
        if len(self.snapshots) <= self.max_snapshots:
            return
        
        # Sort by timestamp and keep most recent
        sorted_snapshots = sorted(
            self.snapshots.items(),
            key=lambda x: x[1].timestamp,
            reverse=True
        )
        
        # Keep only the most recent snapshots
        to_keep = dict(sorted_snapshots[:self.max_snapshots])
        to_remove = set(self.snapshots.keys()) - set(to_keep.keys())
        
        for snapshot_id in to_remove:
            snapshot = self.snapshots[snapshot_id]
            del self.snapshots[snapshot_id]
            
            # Remove from indexes
            self.session_snapshots[snapshot.session_id].remove(snapshot_id)
            self.agent_snapshots[snapshot.agent_name].remove(snapshot_id)


class ExecutionFlowAnalyzer:
    """
    Execution flow analysis and visualization system.
    
    Analyzes agent execution patterns, identifies bottlenecks,
    and provides flow visualization capabilities.
    """
    
    def __init__(self):
        """Initialize execution flow analyzer."""
        self.execution_history = get_execution_history()
        
        logger.info("ExecutionFlowAnalyzer initialized")
    
    def analyze_execution_flow(self, session_id: str) -> Dict[str, Any]:
        """Analyze execution flow for a session."""
        # Get execution record
        records = self.execution_history.search_records(limit=1000)
        record = None
        
        for r in records:
            if r.session_id == session_id:
                record = r
                break
        
        if not record:
            return {"error": f"No execution record found for session {session_id}"}
        
        analysis = {
            'session_id': session_id,
            'agent_name': record.agent_name,
            'total_duration': record.total_duration,
            'final_status': record.final_status.value,
            'flow_steps': [],
            'bottlenecks': [],
            'error_points': [],
            'performance_metrics': {}
        }
        
        # Analyze events for flow
        current_step = None
        step_start_time = record.start_time
        
        for event in record.events:
            if event.event_type == "step_update":
                # End previous step
                if current_step:
                    step_duration = (event.timestamp - step_start_time).total_seconds()
                    analysis['flow_steps'].append({
                        'step_name': current_step,
                        'duration': step_duration,
                        'start_time': step_start_time.isoformat(),
                        'end_time': event.timestamp.isoformat()
                    })
                    
                    # Check for bottleneck (step taking > 30% of total time)
                    if record.total_duration and step_duration > (record.total_duration * 0.3):
                        analysis['bottlenecks'].append({
                            'step_name': current_step,
                            'duration': step_duration,
                            'percentage': (step_duration / record.total_duration) * 100
                        })
                
                # Start new step
                current_step = event.data.get('step', event.message.split(': ')[-1])
                step_start_time = event.timestamp
            
            elif event.event_type == "error":
                analysis['error_points'].append({
                    'timestamp': event.timestamp.isoformat(),
                    'error': event.error or event.message,
                    'step': current_step,
                    'context': event.data
                })
        
        # Performance metrics
        analysis['performance_metrics'] = {
            'total_steps': len(analysis['flow_steps']),
            'avg_step_duration': sum(s['duration'] for s in analysis['flow_steps']) / len(analysis['flow_steps']) if analysis['flow_steps'] else 0,
            'longest_step': max(analysis['flow_steps'], key=lambda s: s['duration']) if analysis['flow_steps'] else None,
            'error_count': len(analysis['error_points']),
            'bottleneck_count': len(analysis['bottlenecks'])
        }
        
        return analysis
    
    def compare_execution_flows(self, session_ids: List[str]) -> Dict[str, Any]:
        """Compare execution flows across multiple sessions."""
        flows = []
        for session_id in session_ids:
            flow = self.analyze_execution_flow(session_id)
            if 'error' not in flow:
                flows.append(flow)
        
        if not flows:
            return {"error": "No valid execution flows found"}
        
        comparison = {
            'sessions_compared': len(flows),
            'average_duration': sum(f['total_duration'] or 0 for f in flows) / len(flows),
            'common_steps': [],
            'step_duration_comparison': {},
            'bottleneck_patterns': defaultdict(int),
            'error_patterns': defaultdict(int)
        }
        
        # Find common steps
        all_steps = set()
        for flow in flows:
            all_steps.update(step['step_name'] for step in flow['flow_steps'])
        
        step_counts = defaultdict(int)
        for flow in flows:
            flow_steps = {step['step_name'] for step in flow['flow_steps']}
            for step in flow_steps:
                step_counts[step] += 1
        
        # Steps that appear in all flows
        comparison['common_steps'] = [
            step for step, count in step_counts.items() 
            if count == len(flows)
        ]
        
        # Compare step durations
        for step_name in comparison['common_steps']:
            durations = []
            for flow in flows:
                for step in flow['flow_steps']:
                    if step['step_name'] == step_name:
                        durations.append(step['duration'])
                        break
            
            if durations:
                comparison['step_duration_comparison'][step_name] = {
                    'avg_duration': sum(durations) / len(durations),
                    'min_duration': min(durations),
                    'max_duration': max(durations),
                    'variance': max(durations) - min(durations)
                }
        
        # Analyze bottleneck patterns
        for flow in flows:
            for bottleneck in flow['bottlenecks']:
                comparison['bottleneck_patterns'][bottleneck['step_name']] += 1
        
        # Analyze error patterns
        for flow in flows:
            for error in flow['error_points']:
                comparison['error_patterns'][error['step']] += 1
        
        return comparison
    
    def generate_flow_diagram(self, session_id: str, format: str = "mermaid") -> str:
        """Generate a flow diagram for execution visualization."""
        flow = self.analyze_execution_flow(session_id)
        
        if 'error' in flow:
            return flow['error']
        
        if format == "mermaid":
            diagram = "graph TD\n"
            diagram += f"    Start([Start: {flow['agent_name']}])\n"
            
            prev_node = "Start"
            for i, step in enumerate(flow['flow_steps']):
                node_id = f"Step{i}"
                step_name = step['step_name'].replace(' ', '_')
                duration = f"{step['duration']:.2f}s"
                
                # Color code based on duration
                if step['duration'] > 5.0:
                    color = "fill:#ff9999"  # Red for slow steps
                elif step['duration'] > 2.0:
                    color = "fill:#ffcc99"  # Orange for medium steps
                else:
                    color = "fill:#99ff99"  # Green for fast steps
                
                diagram += f"    {node_id}[{step_name}<br/>{duration}]\n"
                diagram += f"    {prev_node} --> {node_id}\n"
                diagram += f"    class {node_id} stepNode\n"
                
                prev_node = node_id
            
            # Add error points
            for i, error in enumerate(flow['error_points']):
                error_node = f"Error{i}"
                diagram += f"    {error_node}[Error: {error['error'][:30]}...]\n"
                diagram += f"    {error_node} --> End\n"
                diagram += f"    class {error_node} errorNode\n"
            
            diagram += f"    {prev_node} --> End([End: {flow['final_status']}])\n"
            
            # Add styling
            diagram += "\n    classDef stepNode fill:#e1f5fe,stroke:#01579b,stroke-width:2px\n"
            diagram += "    classDef errorNode fill:#ffebee,stroke:#c62828,stroke-width:2px\n"
            
            return diagram
        
        return "Unsupported format"


class AgentDebugger:
    """
    Comprehensive agent debugging system.
    
    Provides interactive debugging capabilities including breakpoints,
    state inspection, and execution flow analysis.
    """
    
    def __init__(self):
        """Initialize agent debugger."""
        self.state_inspector = StateInspector()
        self.flow_analyzer = ExecutionFlowAnalyzer()
        
        # Breakpoint management
        self.breakpoints: Dict[str, ExecutionBreakpoint] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Debug callbacks
        self.debug_callbacks: List[Callable] = []
        
        logger.info("AgentDebugger initialized")
    
    def add_breakpoint(self, agent_name: str, step_name: Optional[str] = None,
                      condition: Optional[str] = None) -> str:
        """Add a debugging breakpoint."""
        breakpoint_id = f"bp_{agent_name}_{step_name or 'any'}_{len(self.breakpoints)}"
        
        breakpoint = ExecutionBreakpoint(
            breakpoint_id=breakpoint_id,
            agent_name=agent_name,
            step_name=step_name,
            condition=condition
        )
        
        self.breakpoints[breakpoint_id] = breakpoint
        logger.info(f"Added breakpoint: {breakpoint_id}")
        return breakpoint_id
    
    def remove_breakpoint(self, breakpoint_id: str) -> bool:
        """Remove a breakpoint."""
        if breakpoint_id in self.breakpoints:
            del self.breakpoints[breakpoint_id]
            logger.info(f"Removed breakpoint: {breakpoint_id}")
            return True
        return False
    
    def enable_breakpoint(self, breakpoint_id: str, enabled: bool = True) -> bool:
        """Enable or disable a breakpoint."""
        if breakpoint_id in self.breakpoints:
            self.breakpoints[breakpoint_id].enabled = enabled
            return True
        return False
    
    def check_breakpoints(self, agent_name: str, step_name: str,
                         context: Dict[str, Any]) -> List[ExecutionBreakpoint]:
        """Check if any breakpoints should trigger."""
        triggered = []
        
        for breakpoint in self.breakpoints.values():
            if breakpoint.should_trigger(agent_name, step_name, context):
                breakpoint.trigger(context)
                triggered.append(breakpoint)
                
                # Capture state if requested
                if breakpoint.capture_state:
                    session_id = context.get('session_id', 'unknown')
                    self.state_inspector.capture_state(
                        agent_name, session_id, step_name,
                        context, InspectionType.INTERMEDIATE
                    )
        
        return triggered
    
    def start_debug_session(self, session_id: str, agent_name: str) -> None:
        """Start a debug session."""
        self.active_sessions[session_id] = {
            'agent_name': agent_name,
            'start_time': datetime.now(),
            'breakpoints_hit': [],
            'snapshots_captured': []
        }
        
        logger.info(f"Started debug session: {session_id}")
    
    def end_debug_session(self, session_id: str) -> Dict[str, Any]:
        """End a debug session and return summary."""
        if session_id not in self.active_sessions:
            return {"error": "Debug session not found"}
        
        session_data = self.active_sessions[session_id]
        end_time = datetime.now()
        duration = (end_time - session_data['start_time']).total_seconds()
        
        summary = {
            'session_id': session_id,
            'agent_name': session_data['agent_name'],
            'duration': duration,
            'breakpoints_hit': len(session_data['breakpoints_hit']),
            'snapshots_captured': len(session_data['snapshots_captured']),
            'execution_flow': self.flow_analyzer.analyze_execution_flow(session_id)
        }
        
        del self.active_sessions[session_id]
        logger.info(f"Ended debug session: {session_id}")
        return summary
    
    def get_debug_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive debugging summary for a session."""
        # Get execution flow analysis
        flow_analysis = self.flow_analyzer.analyze_execution_flow(session_id)
        
        # Get state snapshots
        snapshots = self.state_inspector.get_session_snapshots(session_id)
        
        # Get state evolution
        state_evolution = self.state_inspector.analyze_state_evolution(session_id)
        
        return {
            'session_id': session_id,
            'execution_flow': flow_analysis,
            'state_snapshots': [s.get_state_summary() for s in snapshots],
            'state_evolution': state_evolution,
            'breakpoints_configured': len(self.breakpoints),
            'active_breakpoints': len([bp for bp in self.breakpoints.values() if bp.enabled])
        }
    
    def generate_debug_report(self, session_id: str, level: DebugLevel = DebugLevel.DETAILED) -> str:
        """Generate a comprehensive debug report."""
        summary = self.get_debug_summary(session_id)
        
        report = f"=== Debug Report for Session {session_id} ===\n\n"
        
        # Execution flow
        flow = summary['execution_flow']
        if 'error' not in flow:
            report += f"Agent: {flow['agent_name']}\n"
            report += f"Status: {flow['final_status']}\n"
            report += f"Duration: {flow['total_duration']:.2f}s\n"
            report += f"Steps: {flow['performance_metrics']['total_steps']}\n\n"
            
            if level in [DebugLevel.DETAILED, DebugLevel.VERBOSE]:
                report += "=== Execution Steps ===\n"
                for step in flow['flow_steps']:
                    report += f"- {step['step_name']}: {step['duration']:.2f}s\n"
                report += "\n"
            
            if flow['bottlenecks']:
                report += "=== Bottlenecks ===\n"
                for bottleneck in flow['bottlenecks']:
                    report += f"- {bottleneck['step_name']}: {bottleneck['duration']:.2f}s ({bottleneck['percentage']:.1f}%)\n"
                report += "\n"
            
            if flow['error_points']:
                report += "=== Errors ===\n"
                for error in flow['error_points']:
                    report += f"- {error['step']}: {error['error']}\n"
                report += "\n"
        
        # State snapshots
        if level == DebugLevel.VERBOSE:
            report += "=== State Snapshots ===\n"
            for snapshot in summary['state_snapshots']:
                report += f"- {snapshot['step_name']} ({snapshot['state_type']}): {snapshot['data_keys']} keys\n"
            report += "\n"
        
        # State evolution
        if level in [DebugLevel.DETAILED, DebugLevel.VERBOSE]:
            evolution = summary['state_evolution']
            if 'error' not in evolution:
                report += "=== State Evolution ===\n"
                report += f"Total Snapshots: {evolution['total_snapshots']}\n"
                report += f"Time Span: {evolution['time_span']:.2f}s\n"
                
                if evolution['complexity_trend']:
                    avg_complexity = sum(evolution['complexity_trend']) / len(evolution['complexity_trend'])
                    report += f"Average Complexity: {avg_complexity:.1f}\n"
                
                report += "\n"
        
        return report
    
    def add_debug_callback(self, callback: Callable) -> None:
        """Add a callback for debug events."""
        self.debug_callbacks.append(callback)


# Global instance
_global_agent_debugger = AgentDebugger()


def get_agent_debugger() -> AgentDebugger:
    """Get the global agent debugger instance."""
    return _global_agent_debugger


def get_state_inspector() -> StateInspector:
    """Get the global state inspector instance."""
    return _global_agent_debugger.state_inspector


def get_flow_analyzer() -> ExecutionFlowAnalyzer:
    """Get the global execution flow analyzer instance."""
    return _global_agent_debugger.flow_analyzer
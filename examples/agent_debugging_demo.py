"""
Agent debugging and inspection demonstration.

This example shows how to use the comprehensive debugging system to inspect
agent state, set breakpoints, analyze execution flow, and troubleshoot issues.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any

from src.langgraph_agents.monitoring.debugging_tools import (
    get_agent_debugger,
    get_state_inspector,
    get_flow_analyzer,
    InspectionType,
    DebugLevel
)
from src.langgraph_agents.monitoring import (
    get_execution_monitor,
    get_execution_history,
    ExecutionStatus
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DebuggableAgent:
    """Mock agent with comprehensive debugging capabilities."""
    
    def __init__(self, name: str):
        self.name = name
        self.debugger = get_agent_debugger()
        self.state_inspector = get_state_inspector()
        self.execution_monitor = get_execution_monitor()
        self.execution_history = get_execution_history()
    
    async def execute_with_debugging(self, session_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent with full debugging capabilities."""
        # Start debug session
        self.debugger.start_debug_session(session_id, self.name)
        
        # Start execution monitoring
        tracker = self.execution_monitor.start_agent_execution(
            self.name, session_id, input_data
        )
        
        try:
            # Capture initial state
            self.state_inspector.capture_state(
                self.name, session_id, "initialization",
                {"input": input_data, "agent_state": "starting"},
                InspectionType.INPUT
            )
            
            # Execute with debugging
            result = await self._execute_with_breakpoints(session_id, input_data)
            
            # Capture final state
            self.state_inspector.capture_state(
                self.name, session_id, "completion",
                {"output": result, "agent_state": "completed"},
                InspectionType.OUTPUT
            )
            
            # Complete execution
            self.execution_monitor.stop_agent_execution(
                session_id, ExecutionStatus.COMPLETED, result
            )
            
            # Record in history
            self.execution_history.record_execution(tracker)
            
            return result
            
        except Exception as e:
            # Capture error state
            self.state_inspector.capture_state(
                self.name, session_id, "error",
                {"error": str(e), "agent_state": "failed"},
                InspectionType.ERROR
            )
            
            # Complete with error
            self.execution_monitor.stop_agent_execution(
                session_id, ExecutionStatus.FAILED
            )
            
            # Record in history
            self.execution_history.record_execution(tracker)
            
            raise
        finally:
            # End debug session
            debug_summary = self.debugger.end_debug_session(session_id)
            logger.info(f"Debug session ended: {debug_summary}")
    
    async def _execute_with_breakpoints(self, session_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent steps with breakpoint checking."""
        context = {
            'session_id': session_id,
            'input_data': input_data,
            'agent_name': self.name
        }
        
        # Step 1: Input validation
        await self._execute_step("input_validation", session_id, context)
        
        # Step 2: Data preprocessing
        context['preprocessed_data'] = {"processed": True, "timestamp": datetime.now()}
        await self._execute_step("data_preprocessing", session_id, context)
        
        # Step 3: Core processing
        context['processing_state'] = {"stage": "core", "progress": 0.5}
        await self._execute_step("core_processing", session_id, context)
        
        # Step 4: Result generation
        context['results'] = {"generated": True, "quality_score": 0.85}
        await self._execute_step("result_generation", session_id, context)
        
        # Step 5: Output formatting
        context['formatted_output'] = {"format": "json", "size": 1024}
        await self._execute_step("output_formatting", session_id, context)
        
        return {
            "result": f"Processed by {self.name}",
            "success": True,
            "processing_time": time.time(),
            "metadata": context.get('formatted_output', {})
        }
    
    async def _execute_step(self, step_name: str, session_id: str, context: Dict[str, Any]) -> None:
        """Execute a single step with debugging support."""
        # Update execution monitor
        self.execution_monitor.update_agent_step(session_id, step_name)
        self.execution_monitor.update_agent_status(
            session_id, ExecutionStatus.PROCESSING, f"Executing {step_name}"
        )
        
        # Check breakpoints
        triggered_breakpoints = self.debugger.check_breakpoints(
            self.name, step_name, context
        )
        
        if triggered_breakpoints:
            logger.info(f"Breakpoints triggered in {step_name}: {len(triggered_breakpoints)}")
            for bp in triggered_breakpoints:
                logger.info(f"  - {bp.breakpoint_id}: {bp.hit_count} hits")
        
        # Capture intermediate state
        self.state_inspector.capture_state(
            self.name, session_id, step_name,
            context.copy(), InspectionType.INTERMEDIATE,
            variables={"step_start_time": datetime.now()},
            local_context={"step_name": step_name, "agent": self.name}
        )
        
        # Simulate processing time
        processing_time = 0.5 + (hash(step_name) % 10) / 10  # 0.5-1.5 seconds
        await asyncio.sleep(processing_time)
        
        # Simulate potential error
        if "error" in context.get('input_data', {}).get('scenario', ''):
            if step_name == "core_processing":
                raise ValueError(f"Simulated error in {step_name}")
        
        # Record step completion
        self.execution_monitor.record_step_completion(
            session_id, step_name, processing_time, success=True
        )


async def demonstrate_basic_debugging():
    """Demonstrate basic debugging capabilities."""
    print("=== Basic Agent Debugging Demo ===\n")
    
    # Create debuggable agent
    agent = DebuggableAgent("DebuggableAgent")
    
    # Set up some breakpoints
    debugger = get_agent_debugger()
    
    # Breakpoint on specific step
    bp1 = debugger.add_breakpoint("DebuggableAgent", "core_processing")
    print(f"Added breakpoint: {bp1}")
    
    # Conditional breakpoint
    bp2 = debugger.add_breakpoint(
        "DebuggableAgent", 
        condition="'error' in input_data.get('scenario', '')"
    )
    print(f"Added conditional breakpoint: {bp2}")
    
    # Execute agent
    session_id = f"debug_session_{int(time.time())}"
    input_data = {"scenario": "normal", "data": "test_data"}
    
    print(f"\nExecuting agent with session: {session_id}")
    
    try:
        result = await agent.execute_with_debugging(session_id, input_data)
        print(f"Execution completed: {result['success']}")
    except Exception as e:
        print(f"Execution failed: {e}")
    
    return session_id


async def demonstrate_state_inspection(session_id: str):
    """Demonstrate state inspection capabilities."""
    print("\n=== State Inspection Demo ===")
    
    inspector = get_state_inspector()
    
    # Get all snapshots for the session
    snapshots = inspector.get_session_snapshots(session_id)
    print(f"Total snapshots captured: {len(snapshots)}")
    
    # Show snapshot summaries
    print("\nSnapshot Summaries:")
    for snapshot in snapshots:
        summary = snapshot.get_state_summary()
        print(f"- {summary['step_name']} ({summary['state_type']}): {len(summary['data_keys'])} keys")
    
    # Compare first and last snapshots
    if len(snapshots) >= 2:
        first_snapshot = snapshots[0]
        last_snapshot = snapshots[-1]
        
        comparison = inspector.compare_snapshots(
            first_snapshot.snapshot_id,
            last_snapshot.snapshot_id
        )
        
        print(f"\nState Evolution from {first_snapshot.step_name} to {last_snapshot.step_name}:")
        if comparison['added_keys']:
            print(f"  Added keys: {comparison['added_keys']}")
        if comparison['changed_values']:
            print(f"  Changed values: {list(comparison['changed_values'].keys())}")
    
    # Analyze state evolution
    evolution = inspector.analyze_state_evolution(session_id)
    if 'error' not in evolution:
        print(f"\nState Evolution Analysis:")
        print(f"  Time span: {evolution['time_span']:.2f} seconds")
        print(f"  Complexity trend: {evolution['complexity_trend']}")
        
        if evolution['key_evolution']:
            print("  Key evolution:")
            for key, changes in list(evolution['key_evolution'].items())[:3]:
                print(f"    {key}: {len(changes)} changes")


async def demonstrate_flow_analysis(session_id: str):
    """Demonstrate execution flow analysis."""
    print("\n=== Execution Flow Analysis Demo ===")
    
    analyzer = get_flow_analyzer()
    
    # Analyze execution flow
    flow_analysis = analyzer.analyze_execution_flow(session_id)
    
    if 'error' not in flow_analysis:
        print(f"Agent: {flow_analysis['agent_name']}")
        print(f"Status: {flow_analysis['final_status']}")
        print(f"Duration: {flow_analysis['total_duration']:.2f}s")
        
        metrics = flow_analysis['performance_metrics']
        print(f"Steps: {metrics['total_steps']}")
        print(f"Average step duration: {metrics['avg_step_duration']:.2f}s")
        
        if metrics['longest_step']:
            longest = metrics['longest_step']
            print(f"Longest step: {longest['step_name']} ({longest['duration']:.2f}s)")
        
        if flow_analysis['bottlenecks']:
            print("\nBottlenecks:")
            for bottleneck in flow_analysis['bottlenecks']:
                print(f"  - {bottleneck['step_name']}: {bottleneck['duration']:.2f}s ({bottleneck['percentage']:.1f}%)")
        
        # Generate flow diagram
        diagram = analyzer.generate_flow_diagram(session_id, "mermaid")
        print(f"\nFlow Diagram (Mermaid):")
        print(diagram[:500] + "..." if len(diagram) > 500 else diagram)


async def demonstrate_error_debugging():
    """Demonstrate debugging with errors."""
    print("\n=== Error Debugging Demo ===")
    
    # Create agent and set up error scenario
    agent = DebuggableAgent("ErrorAgent")
    debugger = get_agent_debugger()
    
    # Add breakpoint for error conditions
    bp_error = debugger.add_breakpoint(
        "ErrorAgent",
        "core_processing",
        condition="'error' in input_data.get('scenario', '')"
    )
    print(f"Added error breakpoint: {bp_error}")
    
    # Execute with error scenario
    session_id = f"error_session_{int(time.time())}"
    input_data = {"scenario": "error", "data": "test_data"}
    
    print(f"Executing agent with error scenario: {session_id}")
    
    try:
        result = await agent.execute_with_debugging(session_id, input_data)
        print(f"Unexpected success: {result}")
    except Exception as e:
        print(f"Expected error caught: {e}")
    
    # Analyze the error
    inspector = get_state_inspector()
    error_snapshots = inspector.get_session_snapshots(session_id, InspectionType.ERROR)
    
    if error_snapshots:
        error_snapshot = error_snapshots[0]
        print(f"\nError State Analysis:")
        print(f"  Error occurred in: {error_snapshot.step_name}")
        print(f"  Error details: {error_snapshot.state_data.get('error', 'Unknown')}")
        print(f"  Context keys: {list(error_snapshot.local_context.keys())}")
    
    # Generate debug report
    debug_report = debugger.generate_debug_report(session_id, DebugLevel.DETAILED)
    print(f"\nDebug Report Preview:")
    print(debug_report[:800] + "..." if len(debug_report) > 800 else debug_report)


async def demonstrate_comparative_analysis():
    """Demonstrate comparative analysis across multiple executions."""
    print("\n=== Comparative Analysis Demo ===")
    
    # Execute multiple sessions for comparison
    agent = DebuggableAgent("ComparisonAgent")
    session_ids = []
    
    scenarios = ["fast", "normal", "slow"]
    
    for scenario in scenarios:
        session_id = f"comparison_{scenario}_{int(time.time())}"
        input_data = {"scenario": scenario, "complexity": len(scenario)}
        
        try:
            await agent.execute_with_debugging(session_id, input_data)
            session_ids.append(session_id)
            print(f"Completed {scenario} scenario: {session_id}")
        except Exception as e:
            print(f"Failed {scenario} scenario: {e}")
        
        # Small delay between executions
        await asyncio.sleep(0.1)
    
    if len(session_ids) >= 2:
        # Compare execution flows
        analyzer = get_flow_analyzer()
        comparison = analyzer.compare_execution_flows(session_ids)
        
        print(f"\nFlow Comparison Results:")
        print(f"  Sessions compared: {comparison['sessions_compared']}")
        print(f"  Average duration: {comparison['average_duration']:.2f}s")
        print(f"  Common steps: {comparison['common_steps']}")
        
        if comparison['step_duration_comparison']:
            print("\n  Step Duration Analysis:")
            for step, stats in comparison['step_duration_comparison'].items():
                print(f"    {step}: avg={stats['avg_duration']:.2f}s, variance={stats['variance']:.2f}s")
        
        # Compare state evolution
        inspector = get_state_inspector()
        print(f"\n  State Snapshot Comparison:")
        for session_id in session_ids:
            snapshots = inspector.get_session_snapshots(session_id)
            print(f"    {session_id}: {len(snapshots)} snapshots")


async def demonstrate_interactive_debugging():
    """Demonstrate interactive debugging features."""
    print("\n=== Interactive Debugging Demo ===")
    
    debugger = get_agent_debugger()
    inspector = get_state_inspector()
    
    # Show current debugging status
    print("Current Debugging Status:")
    print(f"  Total breakpoints: {len(debugger.breakpoints)}")
    print(f"  Active sessions: {len(debugger.active_sessions)}")
    print(f"  State snapshots: {len(inspector.snapshots)}")
    
    # Show breakpoint statistics
    if debugger.breakpoints:
        print("\nBreakpoint Statistics:")
        for bp_id, bp in debugger.breakpoints.items():
            print(f"  {bp_id}: {bp.hit_count} hits, {'enabled' if bp.enabled else 'disabled'}")
    
    # Show recent snapshots
    recent_snapshots = inspector.search_snapshots(limit=5)
    if recent_snapshots:
        print(f"\nRecent State Snapshots:")
        for snapshot in recent_snapshots:
            print(f"  {snapshot.agent_name}.{snapshot.step_name}: {len(snapshot.state_data)} keys")
    
    print("\nInteractive debugging features available:")
    print("  - Set breakpoints with conditions")
    print("  - Capture and compare state snapshots")
    print("  - Analyze execution flow patterns")
    print("  - Generate comprehensive debug reports")
    print("  - Export debugging data for analysis")


async def main():
    """Run all debugging demonstrations."""
    print("Starting comprehensive agent debugging demonstration...\n")
    
    # Basic debugging
    session_id = await demonstrate_basic_debugging()
    
    # State inspection
    await demonstrate_state_inspection(session_id)
    
    # Flow analysis
    await demonstrate_flow_analysis(session_id)
    
    # Error debugging
    await demonstrate_error_debugging()
    
    # Comparative analysis
    await demonstrate_comparative_analysis()
    
    # Interactive features
    await demonstrate_interactive_debugging()
    
    print("\n=== Debugging Demonstration Complete ===")
    print("Use the debug CLI for interactive debugging:")
    print("  python -m src.langgraph_agents.monitoring.debug_cli --help")


if __name__ == "__main__":
    asyncio.run(main())
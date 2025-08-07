"""
Comprehensive agent monitoring and debugging system demonstration.

This example showcases the complete monitoring infrastructure including:
- Real-time execution monitoring
- Performance tracking and alerting
- Error detection and classification
- State inspection and debugging
- Execution flow analysis
- Troubleshooting and diagnostics
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any

from src.langgraph_agents.monitoring import (
    get_execution_monitor,
    get_performance_tracker,
    get_error_detector,
    get_execution_history,
    get_real_time_monitor,
    get_agent_debugger,
    get_state_inspector,
    get_flow_analyzer,
    ExecutionStatus,
    InspectionType,
    DebugLevel
)
from src.langgraph_agents.monitoring.troubleshooting_guide import get_troubleshooting_guide

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComprehensiveMonitoredAgent:
    """Agent with complete monitoring and debugging integration."""
    
    def __init__(self, name: str):
        self.name = name
        
        # Get all monitoring components
        self.execution_monitor = get_execution_monitor()
        self.performance_tracker = get_performance_tracker()
        self.error_detector = get_error_detector()
        self.execution_history = get_execution_history()
        self.debugger = get_agent_debugger()
        self.state_inspector = get_state_inspector()
        self.troubleshooting_guide = get_troubleshooting_guide()
        
        # Setup monitoring callbacks
        self._setup_monitoring_callbacks()
    
    def _setup_monitoring_callbacks(self):
        """Setup monitoring callbacks for comprehensive tracking."""
        
        def on_status_change(tracker):
            logger.info(f"[{self.name}] Status: {tracker.current_status.value} (Step: {tracker.current_step})")
        
        def on_performance_alert(alert):
            logger.warning(f"[{self.name}] Performance Alert: {alert.message}")
        
        def on_error_alert(alert):
            logger.error(f"[{self.name}] Error Alert: {alert.error_message}")
        
        self.execution_monitor.add_status_callback(on_status_change)
        self.performance_tracker.add_alert_callback(on_performance_alert)
        self.error_detector.add_alert_callback(on_error_alert)
    
    async def execute_with_full_monitoring(self, session_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent with comprehensive monitoring and debugging."""
        
        # Start all monitoring systems
        tracker = self.execution_monitor.start_agent_execution(self.name, session_id, input_data)
        perf_data = self.performance_tracker.start_tracking(self.name, session_id)
        self.debugger.start_debug_session(session_id, self.name)
        
        # Capture initial state
        self.state_inspector.capture_state(
            self.name, session_id, "initialization",
            {"input": input_data, "timestamp": datetime.now()},
            InspectionType.INPUT
        )
        
        try:
            # Execute with comprehensive monitoring
            result = await self._execute_monitored_workflow(session_id, input_data)
            
            # Capture final state
            self.state_inspector.capture_state(
                self.name, session_id, "completion",
                {"output": result, "timestamp": datetime.now()},
                InspectionType.OUTPUT
            )
            
            # Complete monitoring
            self.execution_monitor.stop_agent_execution(session_id, ExecutionStatus.COMPLETED, result)
            self.performance_tracker.stop_tracking(session_id)
            
            # Record execution history
            self.execution_history.record_execution(tracker)
            
            return result
            
        except Exception as e:
            # Comprehensive error handling
            error_msg = str(e)
            
            # Record error in all systems
            self.execution_monitor.record_agent_error(session_id, error_msg)
            self.error_detector.detect_error(self.name, session_id, error_msg, {"input": input_data})
            self.performance_tracker.record_error(session_id)
            
            # Capture error state
            self.state_inspector.capture_state(
                self.name, session_id, "error",
                {"error": error_msg, "timestamp": datetime.now()},
                InspectionType.ERROR
            )
            
            # Complete with error
            self.execution_monitor.stop_agent_execution(session_id, ExecutionStatus.FAILED)
            self.performance_tracker.stop_tracking(session_id)
            
            # Record execution history
            self.execution_history.record_execution(tracker)
            
            raise
        finally:
            # End debug session
            debug_summary = self.debugger.end_debug_session(session_id)
            logger.info(f"Debug session completed: {debug_summary.get('breakpoints_hit', 0)} breakpoints hit")
    
    async def _execute_monitored_workflow(self, session_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent workflow with detailed monitoring."""
        
        workflow_context = {
            'session_id': session_id,
            'input_data': input_data,
            'agent_name': self.name,
            'workflow_state': {}
        }
        
        # Step 1: Input Processing
        await self._execute_monitored_step("input_processing", session_id, workflow_context)
        workflow_context['workflow_state']['input_processed'] = True
        
        # Step 2: Data Validation
        await self._execute_monitored_step("data_validation", session_id, workflow_context)
        workflow_context['workflow_state']['data_validated'] = True
        
        # Step 3: Core Processing
        await self._execute_monitored_step("core_processing", session_id, workflow_context)
        workflow_context['workflow_state']['core_completed'] = True
        
        # Step 4: Result Generation
        await self._execute_monitored_step("result_generation", session_id, workflow_context)
        workflow_context['workflow_state']['results_generated'] = True
        
        # Step 5: Output Formatting
        await self._execute_monitored_step("output_formatting", session_id, workflow_context)
        workflow_context['workflow_state']['output_formatted'] = True
        
        return {
            "agent": self.name,
            "session_id": session_id,
            "result": "Successfully processed",
            "timestamp": datetime.now().isoformat(),
            "workflow_state": workflow_context['workflow_state']
        }
    
    async def _execute_monitored_step(self, step_name: str, session_id: str, context: Dict[str, Any]):
        """Execute a single step with comprehensive monitoring."""
        
        # Update execution monitor
        self.execution_monitor.update_agent_step(session_id, step_name)
        self.execution_monitor.update_agent_status(
            session_id, ExecutionStatus.PROCESSING, f"Executing {step_name}"
        )
        
        # Check breakpoints
        triggered_breakpoints = self.debugger.check_breakpoints(self.name, step_name, context)
        if triggered_breakpoints:
            logger.info(f"Breakpoints triggered in {step_name}: {[bp.breakpoint_id for bp in triggered_breakpoints]}")
        
        # Capture state before processing
        self.state_inspector.capture_state(
            self.name, session_id, f"{step_name}_start",
            context.copy(), InspectionType.INTERMEDIATE
        )
        
        step_start_time = time.time()
        
        # Simulate step processing with monitoring
        await self._simulate_step_processing(step_name, session_id, context)
        
        step_duration = time.time() - step_start_time
        
        # Record performance metrics
        self.performance_tracker.record_step_time(session_id, step_name, step_duration)
        self.execution_monitor.record_step_completion(session_id, step_name, step_duration)
        
        # Capture state after processing
        self.state_inspector.capture_state(
            self.name, session_id, f"{step_name}_end",
            context.copy(), InspectionType.INTERMEDIATE
        )
    
    async def _simulate_step_processing(self, step_name: str, session_id: str, context: Dict[str, Any]):
        """Simulate step processing with realistic behavior."""
        
        # Simulate API calls
        if step_name in ["data_validation", "core_processing"]:
            self.performance_tracker.record_api_call(session_id, tokens_used=100 + hash(step_name) % 200)
        
        # Simulate cache operations
        if hash(step_name + session_id) % 3 == 0:
            self.performance_tracker.record_cache_hit(session_id)
        else:
            self.performance_tracker.record_cache_miss(session_id)
        
        # Simulate processing time based on step complexity
        step_times = {
            "input_processing": 0.3,
            "data_validation": 0.5,
            "core_processing": 1.2,
            "result_generation": 0.8,
            "output_formatting": 0.4
        }
        
        processing_time = step_times.get(step_name, 0.5)
        
        # Add some randomness
        processing_time += (hash(session_id + step_name) % 100) / 1000
        
        await asyncio.sleep(processing_time)
        
        # Simulate potential errors
        error_scenarios = context.get('input_data', {}).get('error_scenarios', [])
        if step_name in error_scenarios:
            if step_name == "data_validation":
                raise ValueError(f"Validation failed in {step_name}")
            elif step_name == "core_processing":
                # Simulate retry scenario
                self.performance_tracker.record_retry(session_id)
                self.execution_monitor.record_agent_retry(session_id, f"Retry in {step_name}")
                await asyncio.sleep(0.2)  # Retry delay
                
                if hash(session_id) % 3 == 0:  # Sometimes still fail
                    raise RuntimeError(f"Processing failed in {step_name}")


async def demonstrate_comprehensive_monitoring():
    """Demonstrate the complete monitoring system."""
    
    print("=== Comprehensive Agent Monitoring System Demo ===\n")
    
    # Initialize monitoring systems
    execution_monitor = get_execution_monitor()
    real_time_monitor = get_real_time_monitor()
    debugger = get_agent_debugger()
    troubleshooting_guide = get_troubleshooting_guide()
    
    # Start monitoring
    execution_monitor.start_monitoring()
    
    # Set up debugging breakpoints
    bp1 = debugger.add_breakpoint("MonitoredAgent", "core_processing")
    bp2 = debugger.add_breakpoint("MonitoredAgent", condition="'error' in input_data.get('error_scenarios', [])")
    
    print(f"Set up breakpoints: {bp1}, {bp2}")
    
    # Create monitored agents
    agents = [
        ComprehensiveMonitoredAgent("MonitoredAgent"),
        ComprehensiveMonitoredAgent("TestAgent"),
        ComprehensiveMonitoredAgent("DebugAgent")
    ]
    
    # Execute agents with different scenarios
    scenarios = [
        {"scenario": "normal", "complexity": "low"},
        {"scenario": "complex", "complexity": "high", "error_scenarios": ["core_processing"]},
        {"scenario": "validation_error", "error_scenarios": ["data_validation"]},
        {"scenario": "performance_test", "complexity": "medium"}
    ]
    
    execution_results = []
    
    print("Starting agent executions with comprehensive monitoring...\n")
    
    for i, (agent, scenario) in enumerate(zip(agents, scenarios)):
        session_id = f"comprehensive_session_{i}_{int(time.time())}"
        
        print(f"Executing {agent.name} with scenario: {scenario['scenario']}")
        
        try:
            result = await agent.execute_with_full_monitoring(session_id, scenario)
            execution_results.append((session_id, "SUCCESS", result))
            print(f"  ✓ Completed successfully")
        except Exception as e:
            execution_results.append((session_id, "FAILED", str(e)))
            print(f"  ✗ Failed: {e}")
        
        # Small delay between executions
        await asyncio.sleep(0.5)
    
    # Wait for monitoring to process
    await asyncio.sleep(2)
    
    print("\n=== Monitoring Results Analysis ===")
    
    # Real-time dashboard
    dashboard = real_time_monitor.get_current_dashboard()
    print(f"Dashboard Summary:")
    print(f"  Active Executions: {len(dashboard['active_executions'])}")
    print(f"  System Health Score: {dashboard['system_health_score']:.1f}")
    print(f"  Total Errors (Last Hour): {dashboard['error_count_last_hour']}")
    print(f"  Critical Alerts: {dashboard['critical_alerts']}")
    
    # Performance analysis
    performance_tracker = get_performance_tracker()
    performance_summary = performance_tracker.get_performance_summary()
    print(f"\nPerformance Summary:")
    print(f"  Total Executions: {performance_summary.get('total_executions', 0)}")
    print(f"  Average Execution Time: {performance_summary.get('avg_execution_time', 0):.2f}s")
    print(f"  Total API Calls: {performance_summary.get('total_api_calls', 0)}")
    print(f"  Total Errors: {performance_summary.get('total_errors', 0)}")
    
    # Error analysis
    error_detector = get_error_detector()
    error_stats = error_detector.get_error_statistics(time_window_hours=1)
    print(f"\nError Analysis:")
    print(f"  Total Errors: {error_stats.get('total_errors', 0)}")
    print(f"  Error Rate: {error_stats.get('error_rate_per_hour', 0):.2f} errors/hour")
    
    if error_stats.get('errors_by_category'):
        print("  Errors by Category:")
        for category, count in error_stats['errors_by_category'].items():
            print(f"    {category}: {count}")
    
    # Debugging analysis
    print(f"\nDebugging Summary:")
    print(f"  Total Breakpoints: {len(debugger.breakpoints)}")
    
    total_hits = sum(bp.hit_count for bp in debugger.breakpoints.values())
    print(f"  Total Breakpoint Hits: {total_hits}")
    
    # State inspection summary
    state_inspector = get_state_inspector()
    print(f"  State Snapshots Captured: {len(state_inspector.snapshots)}")
    
    # Execution flow analysis
    flow_analyzer = get_flow_analyzer()
    print(f"\n=== Execution Flow Analysis ===")
    
    for session_id, status, result in execution_results:
        print(f"\nSession: {session_id} ({status})")
        
        flow_analysis = flow_analyzer.analyze_execution_flow(session_id)
        if 'error' not in flow_analysis:
            metrics = flow_analysis['performance_metrics']
            print(f"  Steps: {metrics['total_steps']}")
            print(f"  Average Step Duration: {metrics['avg_step_duration']:.2f}s")
            print(f"  Bottlenecks: {metrics['bottleneck_count']}")
            print(f"  Errors: {metrics['error_count']}")
            
            if flow_analysis['bottlenecks']:
                print("  Bottleneck Details:")
                for bottleneck in flow_analysis['bottlenecks']:
                    print(f"    {bottleneck['step_name']}: {bottleneck['duration']:.2f}s")
    
    # Troubleshooting analysis
    print(f"\n=== Troubleshooting Analysis ===")
    
    # System health diagnosis
    system_issues = troubleshooting_guide.diagnose_system_health()
    print(f"System Health Issues Found: {len(system_issues)}")
    
    for issue in system_issues:
        print(f"  {issue.severity.value.upper()}: {issue.description}")
    
    # Agent-specific diagnosis
    for agent in agents:
        agent_issues = troubleshooting_guide.diagnose_agent(agent.name, hours=1)
        if agent_issues:
            print(f"\n{agent.name} Issues:")
            for issue in agent_issues:
                print(f"  {issue.severity.value.upper()}: {issue.description}")
    
    # Generate comprehensive troubleshooting report
    print(f"\n=== Generating Troubleshooting Report ===")
    report = troubleshooting_guide.generate_troubleshooting_report()
    
    # Save report to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"troubleshooting_report_{timestamp}.txt"
    
    with open(report_filename, 'w') as f:
        f.write(report)
    
    print(f"Comprehensive troubleshooting report saved to: {report_filename}")
    
    # Show report preview
    print(f"\nReport Preview:")
    print(report[:1000] + "..." if len(report) > 1000 else report)
    
    # Stop monitoring
    execution_monitor.stop_monitoring()
    
    print(f"\n=== Comprehensive Monitoring Demo Complete ===")
    print(f"Monitoring Features Demonstrated:")
    print(f"  ✓ Real-time execution monitoring")
    print(f"  ✓ Performance tracking and alerting")
    print(f"  ✓ Error detection and classification")
    print(f"  ✓ State inspection and debugging")
    print(f"  ✓ Execution flow analysis")
    print(f"  ✓ Automated troubleshooting and diagnostics")
    print(f"  ✓ Comprehensive reporting")


if __name__ == "__main__":
    asyncio.run(demonstrate_comprehensive_monitoring())
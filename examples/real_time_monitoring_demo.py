"""
Real-time agent execution monitoring demonstration.

This example shows how to use the comprehensive monitoring system to track
agent execution in real-time, including performance metrics, error detection,
and execution history.
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
    ExecutionStatus
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockAgent:
    """Mock agent for demonstration purposes."""
    
    def __init__(self, name: str):
        self.name = name
    
    async def execute(self, session_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock agent execution with monitoring."""
        # Get monitoring components
        execution_monitor = get_execution_monitor()
        performance_tracker = get_performance_tracker()
        error_detector = get_error_detector()
        execution_history = get_execution_history()
        
        # Start monitoring
        tracker = execution_monitor.start_agent_execution(
            self.name, session_id, input_data
        )
        perf_data = performance_tracker.start_tracking(self.name, session_id)
        
        try:
            # Simulate agent processing steps
            await self._simulate_processing(session_id, input_data)
            
            # Complete execution
            output_data = {"result": f"Processed by {self.name}", "success": True}
            execution_monitor.stop_agent_execution(session_id, ExecutionStatus.COMPLETED, output_data)
            performance_tracker.stop_tracking(session_id)
            
            # Record in history
            execution_history.record_execution(tracker)
            
            return output_data
            
        except Exception as e:
            # Handle errors
            error_msg = str(e)
            execution_monitor.record_agent_error(session_id, error_msg)
            error_detector.detect_error(self.name, session_id, error_msg)
            
            # Complete with error
            execution_monitor.stop_agent_execution(session_id, ExecutionStatus.FAILED)
            performance_tracker.stop_tracking(session_id)
            
            # Record in history
            execution_history.record_execution(tracker)
            
            raise
    
    async def _simulate_processing(self, session_id: str, input_data: Dict[str, Any]) -> None:
        """Simulate agent processing with monitoring events."""
        execution_monitor = get_execution_monitor()
        performance_tracker = get_performance_tracker()
        error_detector = get_error_detector()
        
        # Step 1: Input validation
        execution_monitor.update_agent_step(session_id, "input_validation")
        execution_monitor.update_agent_status(session_id, ExecutionStatus.PROCESSING, "Validating input")
        
        await asyncio.sleep(0.5)  # Simulate processing time
        step_duration = 0.5
        execution_monitor.record_step_completion(session_id, "input_validation", step_duration)
        performance_tracker.record_step_time(session_id, "input_validation", step_duration)
        
        # Step 2: Data processing
        execution_monitor.update_agent_step(session_id, "data_processing")
        execution_monitor.update_agent_status(session_id, ExecutionStatus.PROCESSING, "Processing data")
        
        # Simulate API call
        performance_tracker.record_api_call(session_id, tokens_used=150)
        
        # Simulate cache hit/miss
        if hash(session_id) % 2 == 0:
            performance_tracker.record_cache_hit(session_id)
        else:
            performance_tracker.record_cache_miss(session_id)
        
        await asyncio.sleep(1.0)
        step_duration = 1.0
        execution_monitor.record_step_completion(session_id, "data_processing", step_duration)
        performance_tracker.record_step_time(session_id, "data_processing", step_duration)
        
        # Step 3: Result generation
        execution_monitor.update_agent_step(session_id, "result_generation")
        execution_monitor.update_agent_status(session_id, ExecutionStatus.PROCESSING, "Generating results")
        
        # Simulate potential error
        if "error" in input_data.get("topic", "").lower():
            # Simulate retry
            execution_monitor.record_agent_retry(session_id, "Simulated error occurred")
            performance_tracker.record_retry(session_id)
            performance_tracker.record_error(session_id)
            
            # Detect error
            error_detector.detect_error(
                self.name, session_id, 
                "Simulated processing error for demonstration",
                {"step": "result_generation", "input": input_data}
            )
            
            await asyncio.sleep(0.5)  # Retry delay
        
        await asyncio.sleep(0.8)
        step_duration = 0.8
        execution_monitor.record_step_completion(session_id, "result_generation", step_duration)
        performance_tracker.record_step_time(session_id, "result_generation", step_duration)
        
        # Final step: Output formatting
        execution_monitor.update_agent_step(session_id, "output_formatting")
        execution_monitor.update_agent_status(session_id, ExecutionStatus.COMPLETING, "Formatting output")
        
        await asyncio.sleep(0.3)
        step_duration = 0.3
        execution_monitor.record_step_completion(session_id, "output_formatting", step_duration)
        performance_tracker.record_step_time(session_id, "output_formatting", step_duration)


async def demonstrate_monitoring():
    """Demonstrate comprehensive monitoring capabilities."""
    print("=== Real-time Agent Execution Monitoring Demo ===\n")
    
    # Get monitoring components
    execution_monitor = get_execution_monitor()
    performance_tracker = get_performance_tracker()
    error_detector = get_error_detector()
    execution_history = get_execution_history()
    real_time_monitor = get_real_time_monitor()
    
    # Setup monitoring callbacks
    def on_status_change(tracker):
        print(f"[STATUS] {tracker.agent_name} ({tracker.session_id[:8]}): {tracker.current_status.value}")
    
    def on_performance_alert(alert):
        print(f"[PERF ALERT] {alert.level.value.upper()}: {alert.message}")
    
    def on_error_alert(alert):
        print(f"[ERROR ALERT] {alert.severity.value.upper()}: {alert.error_message}")
    
    execution_monitor.add_status_callback(on_status_change)
    performance_tracker.add_alert_callback(on_performance_alert)
    error_detector.add_alert_callback(on_error_alert)
    
    # Start monitoring
    execution_monitor.start_monitoring()
    
    # Create mock agents
    agents = [
        MockAgent("PlannerAgent"),
        MockAgent("CodeGeneratorAgent"),
        MockAgent("RendererAgent")
    ]
    
    print("Starting agent executions...\n")
    
    # Execute agents concurrently
    tasks = []
    for i, agent in enumerate(agents):
        session_id = f"session_{i}_{int(time.time())}"
        input_data = {
            "topic": f"test_topic_{i}",
            "description": f"Test description for agent {i}"
        }
        
        # Add error scenario for demonstration
        if i == 1:
            input_data["topic"] = "error_scenario"
        
        task = asyncio.create_task(agent.execute(session_id, input_data))
        tasks.append(task)
        
        # Stagger execution starts
        await asyncio.sleep(0.2)
    
    # Wait for all executions to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    print("\n=== Execution Results ===")
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Agent {i}: FAILED - {result}")
        else:
            print(f"Agent {i}: SUCCESS - {result}")
    
    # Wait a moment for monitoring to process
    await asyncio.sleep(1)
    
    # Display monitoring results
    print("\n=== Monitoring Summary ===")
    
    # Execution monitor summary
    monitoring_summary = execution_monitor.get_monitoring_summary()
    print(f"Total Events: {monitoring_summary.get('total_events', 0)}")
    print(f"Active Executions: {monitoring_summary.get('active_executions', 0)}")
    
    # Performance summary
    performance_summary = performance_tracker.get_performance_summary()
    print(f"Total Executions: {performance_summary.get('total_executions', 0)}")
    print(f"Average Execution Time: {performance_summary.get('avg_execution_time', 0):.2f}s")
    print(f"Total API Calls: {performance_summary.get('total_api_calls', 0)}")
    print(f"Total Tokens: {performance_summary.get('total_tokens', 0)}")
    
    # Error statistics
    error_stats = error_detector.get_error_statistics(time_window_hours=1)
    print(f"Total Errors: {error_stats.get('total_errors', 0)}")
    print(f"Error Rate: {error_stats.get('error_rate_per_hour', 0):.2f} errors/hour")
    
    # Execution history
    history_stats = execution_history.get_execution_statistics(days=1)
    print(f"Success Rate: {history_stats.get('success_rate', 0):.2%}")
    
    print("\n=== Recent Events ===")
    recent_events = execution_monitor.get_recent_events(limit=5)
    for event in recent_events:
        timestamp = event.timestamp.strftime("%H:%M:%S")
        print(f"[{timestamp}] {event.agent_name}: {event.message}")
    
    print("\n=== Performance Alerts ===")
    perf_alerts = performance_tracker.get_recent_alerts(limit=3)
    for alert in perf_alerts:
        timestamp = datetime.fromisoformat(alert['timestamp']).strftime("%H:%M:%S")
        print(f"[{timestamp}] {alert['level'].upper()}: {alert['message']}")
    
    print("\n=== Error Alerts ===")
    error_alerts = error_detector.get_recent_alerts(limit=3)
    for alert in error_alerts:
        timestamp = datetime.fromisoformat(alert['timestamp']).strftime("%H:%M:%S")
        print(f"[{timestamp}] {alert['severity'].upper()}: {alert['error_message']}")
    
    # Stop monitoring
    execution_monitor.stop_monitoring()
    
    print("\n=== Demo Complete ===")


async def demonstrate_real_time_dashboard():
    """Demonstrate real-time dashboard capabilities."""
    print("\n=== Real-time Dashboard Demo ===")
    
    real_time_monitor = get_real_time_monitor()
    
    # Get current dashboard
    dashboard = real_time_monitor.get_current_dashboard()
    
    print(f"Dashboard Timestamp: {dashboard['timestamp']}")
    print(f"Active Executions: {len(dashboard['active_executions'])}")
    print(f"System Health Score: {dashboard['system_health_score']:.1f}")
    print(f"Average Execution Time: {dashboard['avg_execution_time']:.2f}s")
    print(f"Error Count (Last Hour): {dashboard['error_count_last_hour']}")
    
    if dashboard['execution_count_by_agent']:
        print("\nExecutions by Agent:")
        for agent, count in dashboard['execution_count_by_agent'].items():
            print(f"  {agent}: {count}")
    
    # Get system health
    health = real_time_monitor.get_system_health()
    print(f"\nSystem Health: {health['health_score']:.1f}")
    print("Health Indicators:")
    for indicator, status in health['health_indicators'].items():
        print(f"  {indicator}: {status}")


async def demonstrate_agent_specific_monitoring():
    """Demonstrate agent-specific monitoring."""
    print("\n=== Agent-specific Monitoring Demo ===")
    
    real_time_monitor = get_real_time_monitor()
    
    # Get status for specific agents
    agents = ["PlannerAgent", "CodeGeneratorAgent", "RendererAgent"]
    
    for agent_name in agents:
        status = real_time_monitor.get_agent_status(agent_name)
        print(f"\n{agent_name} Status:")
        print(f"  Active Executions: {status['active_executions']}")
        print(f"  Recent Executions: {len(status['recent_executions'])}")
        print(f"  Total Errors: {status['error_statistics'].get('total_errors', 0)}")
        
        if status['active_sessions']:
            print("  Active Sessions:")
            for session in status['active_sessions']:
                print(f"    {session['session_id'][:8]}: {session['status']} ({session['duration']:.1f}s)")


if __name__ == "__main__":
    async def main():
        await demonstrate_monitoring()
        await demonstrate_real_time_dashboard()
        await demonstrate_agent_specific_monitoring()
    
    asyncio.run(main())
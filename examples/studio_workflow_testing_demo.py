#!/usr/bin/env python3
"""
Demonstration of Studio workflow testing capabilities.

This script shows how to use the comprehensive workflow testing system
for end-to-end validation, state inspection, checkpoint analysis, and
agent transition testing in LangGraph Studio.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any

# Add the src directory to the path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.langgraph_agents.studio.studio_workflow_testing import (
    get_studio_workflow_tester,
    WorkflowTestType,
    WorkflowTestScenario,
    run_end_to_end_test,
    run_state_validation_test,
    run_agent_transition_test,
    run_error_recovery_test,
    get_workflow_test_info
)
from src.langgraph_agents.studio.studio_graphs import (
    get_available_graphs,
    get_graph_info,
    run_graph_test
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demonstrate_end_to_end_workflow_testing():
    """Demonstrate end-to-end workflow testing."""
    print("\n" + "="*80)
    print("END-TO-END WORKFLOW TESTING")
    print("="*80)
    
    # Test with default scenario
    print("\n1. Running basic end-to-end workflow test...")
    result = await run_end_to_end_test()
    
    print(f"   Status: {result.status.value}")
    print(f"   Execution time: {result.execution_time:.2f}s")
    print(f"   State snapshots captured: {len(result.state_snapshots)}")
    print(f"   Checkpoint data points: {len(result.checkpoint_data)}")
    print(f"   Agent transitions logged: {len(result.transition_logs)}")
    
    if result.final_state:
        print(f"   Final workflow state:")
        print(f"     - Current step: {result.final_state.current_step}")
        print(f"     - Workflow complete: {result.final_state.workflow_complete}")
        print(f"     - Has errors: {result.final_state.has_errors()}")
        print(f"     - Completion: {result.final_state.get_completion_percentage() if hasattr(result.final_state, 'get_completion_percentage') else 0:.1f}%")
    
    # Test with custom input
    print("\n2. Running end-to-end test with custom input...")
    custom_input = {
        'topic': 'Calculus Derivatives',
        'description': 'Explain the concept of derivatives in calculus with visual examples and step-by-step calculations',
        'session_id': 'custom_e2e_test',
        'preview_mode': True
    }
    
    custom_result = await run_end_to_end_test(custom_input)
    print(f"   Custom test status: {custom_result.status.value}")
    print(f"   Custom test execution time: {custom_result.execution_time:.2f}s")
    
    # Show validation results
    if custom_result.validation_results:
        validation = custom_result.validation_results[0]
        print(f"   Validation results:")
        print(f"     - Overall status: {validation.overall_status.value}")
        print(f"     - Overall score: {validation.overall_score:.2f}")
        print(f"     - Passed comparisons: {len(validation.get_passed_comparisons())}")
        print(f"     - Failed comparisons: {len(validation.get_failed_comparisons())}")


async def demonstrate_state_validation_testing():
    """Demonstrate state validation and inspection."""
    print("\n" + "="*80)
    print("STATE VALIDATION AND INSPECTION TESTING")
    print("="*80)
    
    print("\n1. Running state validation test...")
    result = await run_state_validation_test()
    
    print(f"   Status: {result.status.value}")
    print(f"   Execution time: {result.execution_time:.2f}s")
    print(f"   State snapshots: {len(result.state_snapshots)}")
    
    # Show state snapshot details
    if result.state_snapshots:
        print(f"\n   State snapshot progression:")
        for i, snapshot in enumerate(result.state_snapshots):
            print(f"     {i+1}. {snapshot['node']} - {snapshot['completion_percentage']:.1f}% complete")
            print(f"        Errors: {snapshot['error_count']}, Step: {snapshot['current_step']}")
    
    # Show validation results
    if result.validation_results:
        validation = result.validation_results[0]
        print(f"\n   State validation results:")
        for comparison in validation.comparisons:
            status_icon = "✅" if comparison.status.value == "passed" else "❌"
            print(f"     {status_icon} {comparison.field}: {comparison.message}")


async def demonstrate_agent_transition_testing():
    """Demonstrate agent transition testing and validation."""
    print("\n" + "="*80)
    print("AGENT TRANSITION TESTING AND VALIDATION")
    print("="*80)
    
    print("\n1. Running agent transition test...")
    result = await run_agent_transition_test()
    
    print(f"   Status: {result.status.value}")
    print(f"   Execution time: {result.execution_time:.2f}s")
    print(f"   Transitions logged: {len(result.transition_logs)}")
    
    # Show transition details
    if result.transition_logs:
        print(f"\n   Agent transition log:")
        for i, transition in enumerate(result.transition_logs):
            if 'from_node' in transition and 'to_node' in transition:
                condition_status = "✅" if transition.get('condition_met', True) else "❌"
                print(f"     {i+1}. {transition['from_node']} → {transition['to_node']}")
                print(f"        Condition: {transition.get('condition', 'N/A')} {condition_status}")
    
    # Show any transition validation errors
    transition_errors = [error for error in result.errors if 'Transition' in error.get('type', '')]
    if transition_errors:
        print(f"\n   Transition validation errors:")
        for error in transition_errors:
            print(f"     ❌ {error['message']}")
    else:
        print(f"\n   ✅ All agent transitions validated successfully")


async def demonstrate_error_recovery_testing():
    """Demonstrate error recovery testing."""
    print("\n" + "="*80)
    print("ERROR RECOVERY TESTING")
    print("="*80)
    
    print("\n1. Running error recovery test...")
    result = await run_error_recovery_test()
    
    print(f"   Status: {result.status.value}")
    print(f"   Execution time: {result.execution_time:.2f}s")
    print(f"   Errors encountered: {len(result.errors)}")
    
    # Show error details
    if result.errors:
        print(f"\n   Error details:")
        for i, error in enumerate(result.errors[:3]):  # Show first 3 errors
            print(f"     {i+1}. {error.get('type', 'Unknown')}: {error.get('message', 'No message')}")
    
    # Show final state despite errors
    if result.final_state:
        print(f"\n   Final state after error handling:")
        print(f"     - Current step: {result.final_state.current_step}")
        print(f"     - Has errors: {result.final_state.has_errors()}")
        print(f"     - Workflow interrupted: {getattr(result.final_state, 'workflow_interrupted', False)}")


async def demonstrate_individual_graph_testing():
    """Demonstrate testing individual graphs."""
    print("\n" + "="*80)
    print("INDIVIDUAL GRAPH TESTING")
    print("="*80)
    
    # Get available graphs
    graphs = get_available_graphs()
    print(f"\n1. Available graphs for testing: {len(graphs)}")
    
    # Test a few key graphs
    test_graphs = [
        'planning_agent_test',
        'code_generation_agent_test',
        'planning_to_code_chain',
        'state_validation_graph'
    ]
    
    test_input = {
        'topic': 'Graph Theory Basics',
        'description': 'Introduction to graph theory with examples of vertices, edges, and common graph types',
        'session_id': 'individual_graph_test'
    }
    
    for graph_name in test_graphs:
        print(f"\n2. Testing {graph_name}...")
        
        # Get graph info
        graph_info = get_graph_info(graph_name)
        print(f"   Description: {graph_info.get('description', 'N/A')}")
        print(f"   Type: {graph_info.get('type', 'N/A')}")
        print(f"   Monitoring: {graph_info.get('monitoring_enabled', False)}")
        
        # Run test
        try:
            result = await run_graph_test(graph_name, test_input)
            
            if result.get('success'):
                print(f"   ✅ Test successful")
                print(f"      - Session ID: {result['session_id']}")
                print(f"      - Final step: {result['final_state']['current_step']}")
                print(f"      - Completion: {result['final_state']['completion_percentage']:.1f}%")
                print(f"      - Execution traces: {result['execution_trace_count']}")
            else:
                print(f"   ❌ Test failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ Test exception: {str(e)}")


async def demonstrate_custom_scenario_creation():
    """Demonstrate creating custom test scenarios."""
    print("\n" + "="*80)
    print("CUSTOM TEST SCENARIO CREATION")
    print("="*80)
    
    # Get the workflow tester
    tester = get_studio_workflow_tester()
    
    print("\n1. Creating custom performance test scenario...")
    
    # Create a custom performance test scenario
    custom_scenario = tester.create_custom_scenario(
        scenario_id='custom_performance_test',
        test_type=WorkflowTestType.PERFORMANCE,
        name='Custom Performance Test',
        description='Test workflow performance with complex mathematical topic',
        input_data={
            'topic': 'Advanced Calculus: Multivariable Functions and Partial Derivatives',
            'description': 'Comprehensive explanation of multivariable calculus including partial derivatives, gradients, and optimization with detailed mathematical proofs and visual representations',
            'session_id': 'custom_performance_test',
            'preview_mode': False  # Full quality for performance testing
        },
        expected_outcomes={
            'workflow_complete': True,
            'execution_time': {'max': 180},  # Should complete within 3 minutes
            'scene_count': {'min': 3, 'max': 8},
            'performance_acceptable': True
        },
        validation_criteria={
            'workflow_complete': {'type': 'exact_match'},
            'execution_time': {'type': 'threshold_match', 'params': {'threshold': 180, 'comparison_type': 'absolute'}},
            'scene_count': {'type': 'threshold_match', 'params': {'threshold': 2, 'comparison_type': 'absolute'}}
        },
        timeout_seconds=300,
        checkpoint_intervals=['planning', 'code_generation', 'rendering'],
        state_validation_points=['planning', 'code_generation', 'rendering', 'complete']
    )
    
    print(f"   ✅ Created custom scenario: {custom_scenario.name}")
    print(f"      - Type: {custom_scenario.test_type.value}")
    print(f"      - Timeout: {custom_scenario.timeout_seconds}s")
    print(f"      - Validation points: {len(custom_scenario.state_validation_points)}")
    
    print("\n2. Running custom scenario...")
    try:
        result = await tester.run_workflow_test('custom_performance_test')
        
        print(f"   Status: {result.status.value}")
        print(f"   Execution time: {result.execution_time:.2f}s")
        
        if result.performance_metrics:
            print(f"   Performance metrics:")
            for key, value in result.performance_metrics.items():
                print(f"     - {key}: {value}")
                
    except Exception as e:
        print(f"   ❌ Custom scenario failed: {str(e)}")


async def demonstrate_test_monitoring_and_history():
    """Demonstrate test monitoring and history tracking."""
    print("\n" + "="*80)
    print("TEST MONITORING AND HISTORY")
    print("="*80)
    
    # Get the workflow tester
    tester = get_studio_workflow_tester()
    
    print("\n1. Current test status...")
    
    # Get active tests
    active_tests = tester.get_active_tests()
    print(f"   Active tests: {len(active_tests)}")
    
    # Get test history
    history = tester.get_test_history(5)  # Last 5 tests
    print(f"   Recent test history: {len(history)} tests")
    
    if history:
        print(f"\n   Recent test results:")
        for i, test in enumerate(history[-3:]):  # Show last 3
            print(f"     {i+1}. {test['scenario_id']} - {test['status']} ({test['execution_time']:.1f}s)")
            if test.get('validation_results'):
                validation = test['validation_results'][0]
                print(f"        Validation: {validation['overall_status']} (score: {validation['overall_score']:.2f})")
    
    # Get workflow test info
    print("\n2. Workflow testing capabilities...")
    test_info = get_workflow_test_info()
    
    print(f"   Available scenarios: {len(test_info['available_scenarios'])}")
    print(f"   Test types supported: {len(test_info['test_types'])}")
    print(f"   Capabilities:")
    for capability in test_info['capabilities']:
        print(f"     - {capability.replace('_', ' ').title()}")


async def demonstrate_checkpoint_inspection():
    """Demonstrate checkpoint inspection capabilities."""
    print("\n" + "="*80)
    print("CHECKPOINT INSPECTION AND ANALYSIS")
    print("="*80)
    
    print("\n1. Running workflow with checkpoint inspection...")
    
    # Test checkpoint inspection graph
    test_input = {
        'topic': 'Linear Algebra: Matrix Operations',
        'description': 'Explain matrix multiplication, determinants, and eigenvalues with step-by-step examples',
        'session_id': 'checkpoint_inspection_test'
    }
    
    try:
        result = await run_graph_test('checkpoint_inspection_graph', test_input)
        
        if result.get('success'):
            print(f"   ✅ Checkpoint inspection test successful")
            print(f"      - Session ID: {result['session_id']}")
            print(f"      - Final step: {result['final_state']['current_step']}")
            print(f"      - Execution traces: {result['execution_trace_count']}")
        else:
            print(f"   ❌ Checkpoint inspection test failed: {result.get('error')}")
            
    except Exception as e:
        print(f"   ❌ Checkpoint inspection exception: {str(e)}")
    
    print("\n2. Checkpoint data analysis...")
    
    # Get the most recent test result that has checkpoint data
    tester = get_studio_workflow_tester()
    history = tester.get_test_history(10)
    
    checkpoint_test = None
    for test in reversed(history):
        if test.get('checkpoint_count', 0) > 0:
            checkpoint_test = test
            break
    
    if checkpoint_test:
        print(f"   Found test with checkpoint data: {checkpoint_test['scenario_id']}")
        print(f"   Checkpoint count: {checkpoint_test['checkpoint_count']}")
        print(f"   State snapshots: {checkpoint_test['state_snapshots_count']}")
    else:
        print(f"   No recent tests with checkpoint data found")


async def main():
    """Main demonstration function."""
    print("🚀 Studio Workflow Testing Demonstration")
    print("This demo shows comprehensive workflow testing capabilities for LangGraph Studio")
    
    try:
        # Run all demonstrations
        await demonstrate_end_to_end_workflow_testing()
        await demonstrate_state_validation_testing()
        await demonstrate_agent_transition_testing()
        await demonstrate_error_recovery_testing()
        await demonstrate_individual_graph_testing()
        await demonstrate_custom_scenario_creation()
        await demonstrate_test_monitoring_and_history()
        await demonstrate_checkpoint_inspection()
        
        print("\n" + "="*80)
        print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nThe Studio workflow testing system provides:")
        print("✅ End-to-end workflow testing with comprehensive validation")
        print("✅ State validation and consistency checking at each step")
        print("✅ Checkpoint data capture and inspection capabilities")
        print("✅ Agent transition monitoring and validation")
        print("✅ Error recovery testing and analysis")
        print("✅ Individual agent and graph testing")
        print("✅ Custom test scenario creation and execution")
        print("✅ Performance monitoring and metrics collection")
        print("✅ Test history tracking and analysis")
        print("✅ Real-time execution logging and result capture")
        
        print(f"\nAll testing capabilities are now ready for use in LangGraph Studio!")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {str(e)}")
        logger.exception("Demonstration error")
        raise


if __name__ == "__main__":
    asyncio.run(main())
"""
Example usage of the agent testing infrastructure.

This script demonstrates how to use the individual agent test runners
and the Studio integration for testing LangGraph agents.
"""

import asyncio
import json
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from langgraph_agents.testing import (
    PlannerAgentTestRunner,
    CodeGeneratorAgentTestRunner,
    RendererAgentTestRunner,
    ErrorHandlerAgentTestRunner,
    HumanLoopAgentTestRunner,
    get_studio_test_runner,
    get_test_data_manager,
    get_scenario_manager
)
from langgraph_agents.testing.test_data_manager import TestComplexity
from langgraph_agents.models.config import WorkflowConfig


async def test_individual_agents():
    """Test individual agents with sample data."""
    print("=== Testing Individual Agents ===\n")
    
    # Initialize configuration
    config = WorkflowConfig()
    
    # Test PlannerAgent
    print("1. Testing PlannerAgent...")
    planner_runner = PlannerAgentTestRunner(config)
    
    planner_inputs = {
        'topic': 'Linear Algebra Basics',
        'description': 'Create an educational video explaining basic linear algebra concepts including vectors, matrices, and linear transformations.',
        'session_id': 'planner_demo_001'
    }
    
    try:
        planner_result = await planner_runner.run_test(planner_inputs)
        print(f"   Success: {planner_result['success']}")
        if planner_result['success']:
            outputs = planner_result['outputs']
            print(f"   Generated {outputs['scene_count']} scenes")
            print(f"   Scene outline length: {len(outputs['scene_outline'])} characters")
            print(f"   Detected plugins: {outputs['detected_plugins']}")
        else:
            print(f"   Error: {planner_result['error']}")
    except Exception as e:
        print(f"   Exception: {str(e)}")
    
    print()
    
    # Test CodeGeneratorAgent
    print("2. Testing CodeGeneratorAgent...")
    codegen_runner = CodeGeneratorAgentTestRunner(config)
    
    codegen_inputs = {
        'topic': 'Linear Algebra Basics',
        'description': 'Create an educational video explaining basic linear algebra concepts.',
        'scene_outline': '''Scene 1: Introduction
- Show title and topic
- Brief overview of linear algebra

Scene 2: Vectors
- Define vectors
- Show vector operations

Scene 3: Matrices
- Define matrices
- Show matrix operations''',
        'scene_implementations': {
            1: 'Implement introduction scene with title and overview of linear algebra concepts',
            2: 'Implement vector demonstration with visual representations and operations',
            3: 'Implement matrix demonstration with visual representations and operations'
        },
        'session_id': 'codegen_demo_001'
    }
    
    try:
        codegen_result = await codegen_runner.run_test(codegen_inputs)
        print(f"   Success: {codegen_result['success']}")
        if codegen_result['success']:
            outputs = codegen_result['outputs']
            print(f"   Generated code for {len(outputs['successful_scenes'])} scenes")
            print(f"   Failed scenes: {len(outputs['failed_scenes'])}")
        else:
            print(f"   Error: {codegen_result['error']}")
    except Exception as e:
        print(f"   Exception: {str(e)}")
    
    print()
    
    # Test ErrorHandlerAgent
    print("3. Testing ErrorHandlerAgent...")
    error_runner = ErrorHandlerAgentTestRunner(config)
    
    error_inputs = {
        'error_scenarios': [
            {
                'error_type': 'MODEL',
                'message': 'API rate limit exceeded',
                'step': 'code_generation',
                'severity': 'MEDIUM'
            },
            {
                'error_type': 'TIMEOUT',
                'message': 'Request timeout after 30 seconds',
                'step': 'planning',
                'severity': 'MEDIUM'
            },
            {
                'error_type': 'VALIDATION',
                'message': 'Invalid input parameters',
                'step': 'planning',
                'severity': 'HIGH'
            }
        ],
        'session_id': 'error_demo_001'
    }
    
    try:
        error_result = await error_runner.run_test(error_inputs)
        print(f"   Success: {error_result['success']}")
        if error_result['success']:
            outputs = error_result['outputs']
            print(f"   Recovery success rate: {outputs['success_rate']:.2%}")
            print(f"   Processed {outputs['total_scenarios']} scenarios")
        else:
            print(f"   Error: {error_result['error']}")
    except Exception as e:
        print(f"   Exception: {str(e)}")
    
    print()


async def test_studio_integration():
    """Test Studio integration features."""
    print("=== Testing Studio Integration ===\n")
    
    # Get Studio test runner
    studio_runner = get_studio_test_runner()
    
    print("1. Studio Test Runner Info:")
    info = studio_runner.get_test_runner_info()
    print(f"   Available agents: {info['studio_test_runner']['available_agents']}")
    print(f"   Features: {len(info['studio_test_runner']['features'])} features")
    print()
    
    # Test with sample data
    print("2. Running PlannerAgent test through Studio:")
    sample_data = studio_runner.get_agent_test_data_sample('PlannerAgent', TestComplexity.SIMPLE)
    
    try:
        result = await studio_runner.run_agent_test('PlannerAgent', sample_data)
        print(f"   Success: {result['success']}")
        print(f"   Session ID: {result['session_id']}")
        
        if result['studio_output']:
            session_info = result['studio_output']['session_info']
            print(f"   Execution time: {session_info['execution_time']:.2f}s")
            
            logs_summary = result['studio_output']['logs']['summary']
            print(f"   Log entries: {logs_summary['total']}")
            
        if result['validation']:
            print(f"   Validation passed: {result['validation'].get('completed_successfully', False)}")
            
    except Exception as e:
        print(f"   Exception: {str(e)}")
    
    print()


async def test_scenario_management():
    """Test scenario management features."""
    print("=== Testing Scenario Management ===\n")
    
    # Get managers
    data_manager = get_test_data_manager()
    scenario_manager = get_scenario_manager()
    
    print("1. Generating test scenarios:")
    
    # Generate scenarios for different agents
    planner_scenarios = scenario_manager.generate_scenarios_for_agent('PlannerAgent', TestComplexity.SIMPLE, 2)
    print(f"   Generated {len(planner_scenarios)} PlannerAgent scenarios")
    
    codegen_scenarios = scenario_manager.generate_scenarios_for_agent('CodeGeneratorAgent', TestComplexity.SIMPLE, 1)
    print(f"   Generated {len(codegen_scenarios)} CodeGeneratorAgent scenarios")
    
    print()
    
    print("2. Running a scenario:")
    if planner_scenarios:
        scenario_id = planner_scenarios[0]
        scenario = scenario_manager.get_scenario(scenario_id)
        print(f"   Scenario ID: {scenario_id}")
        print(f"   Scenario name: {scenario.name}")
        
        # Get Studio runner and execute scenario
        studio_runner = get_studio_test_runner()
        
        try:
            result = await studio_runner.run_scenario(scenario_id)
            print(f"   Execution success: {result['success']}")
            print(f"   Agent type: {result['agent_type']}")
        except Exception as e:
            print(f"   Exception: {str(e)}")
    
    print()
    
    print("3. Scenario statistics:")
    all_scenarios = scenario_manager.list_scenarios()
    print(f"   Total scenarios: {len(all_scenarios)}")
    
    execution_history = scenario_manager.get_execution_history()
    print(f"   Executed scenarios: {len(execution_history)}")
    
    print()


async def test_comprehensive_suite():
    """Test comprehensive test suite."""
    print("=== Testing Comprehensive Suite ===\n")
    
    studio_runner = get_studio_test_runner()
    
    print("Running comprehensive test suite (this may take a while)...")
    
    try:
        # Run suite for selected agents to avoid long execution time
        agents_to_test = ['PlannerAgent', 'ErrorHandlerAgent']
        
        suite_result = await studio_runner.run_comprehensive_test_suite(
            complexity=TestComplexity.SIMPLE,
            agents_to_test=agents_to_test
        )
        
        summary = suite_result['summary']
        print(f"   Suite ID: {suite_result['suite_id']}")
        print(f"   Total tests: {summary['total_tests']}")
        print(f"   Successful tests: {summary['successful_tests']}")
        print(f"   Success rate: {summary['success_rate']:.2%}")
        
        print("\n   Individual results:")
        for agent_type, result in suite_result['results'].items():
            status = "✓" if result['success'] else "✗"
            print(f"     {status} {agent_type}")
            
    except Exception as e:
        print(f"   Exception: {str(e)}")
    
    print()


async def main():
    """Main function to run all tests."""
    print("Agent Testing Infrastructure Demo")
    print("=" * 50)
    print()
    
    try:
        # Test individual agents
        await test_individual_agents()
        
        # Test Studio integration
        await test_studio_integration()
        
        # Test scenario management
        await test_scenario_management()
        
        # Test comprehensive suite
        await test_comprehensive_suite()
        
        print("Demo completed successfully!")
        
    except Exception as e:
        print(f"Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
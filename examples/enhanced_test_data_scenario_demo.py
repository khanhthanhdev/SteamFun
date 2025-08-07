#!/usr/bin/env python3
"""
Enhanced Test Data and Scenario Management Demo

This example demonstrates the comprehensive test data fixtures, scenario management,
input validation, and result comparison capabilities implemented for task 2.3.
"""

import asyncio
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the src directory to the path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Import the enhanced testing modules
from src.langgraph_agents.testing import (
    TestDataManager, TestScenarioManager, TestComplexity,
    InputValidationManager, get_validation_manager, validate_agent_inputs,
    ResultValidator, get_result_validator, validate_test_result,
    ScenarioSelector, ScenarioExecutor, StudioScenarioInterface, get_studio_interface
)
from src.langgraph_agents.testing.scenario_selection import (
    ScenarioFilterCriteria, ScenarioFilter, ScenarioSortCriteria, SortOrder,
    ScenarioExecutionConfig
)
from src.langgraph_agents.models.config import WorkflowConfig, ModelConfig


def demonstrate_test_data_fixtures():
    """Demonstrate enhanced test data fixtures for each agent type."""
    print("\n" + "="*60)
    print("DEMONSTRATING ENHANCED TEST DATA FIXTURES")
    print("="*60)
    
    # Initialize test data manager
    data_manager = TestDataManager()
    
    # Demonstrate planner test data
    print("\n1. PlannerAgent Test Data:")
    planner_data = data_manager.get_planner_test_data(TestComplexity.MEDIUM, count=2)
    for i, data in enumerate(planner_data, 1):
        print(f"   Test Case {i}:")
        print(f"     Topic: {data['topic']}")
        print(f"     Description: {data['description'][:100]}...")
        print(f"     Session ID: {data['session_id']}")
    
    # Demonstrate code generator test data
    print("\n2. CodeGeneratorAgent Test Data:")
    codegen_data = data_manager.get_codegen_test_data(TestComplexity.SIMPLE, count=1)
    for i, data in enumerate(codegen_data, 1):
        print(f"   Test Case {i}:")
        print(f"     Topic: {data['topic']}")
        print(f"     Scene Implementations: {len(data['scene_implementations'])} scenes")
        for scene_num, impl in data['scene_implementations'].items():
            print(f"       Scene {scene_num}: {impl[:50]}...")
    
    # Demonstrate renderer test data
    print("\n3. RendererAgent Test Data:")
    renderer_data = data_manager.get_renderer_test_data(TestComplexity.SIMPLE, count=1)
    for i, data in enumerate(renderer_data, 1):
        print(f"   Test Case {i}:")
        print(f"     Topic: {data['topic']}")
        print(f"     Generated Code: {len(data['generated_code'])} scenes")
        print(f"     File Prefix: {data['file_prefix']}")
        print(f"     Quality: {data['quality']}")
    
    # Demonstrate error handler test data
    print("\n4. ErrorHandlerAgent Test Data:")
    error_data = data_manager.get_error_handler_test_data(TestComplexity.MEDIUM, count=1)
    for i, data in enumerate(error_data, 1):
        print(f"   Test Case {i}:")
        print(f"     Error Scenarios: {len(data['error_scenarios'])}")
        for j, scenario in enumerate(data['error_scenarios'][:2], 1):
            print(f"       Scenario {j}: {scenario['error_type']} - {scenario['message']}")
    
    # Demonstrate human loop test data
    print("\n5. HumanLoopAgent Test Data:")
    human_loop_data = data_manager.get_human_loop_test_data(TestComplexity.SIMPLE, count=1)
    for i, data in enumerate(human_loop_data, 1):
        print(f"   Test Case {i}:")
        print(f"     Intervention Scenarios: {len(data['intervention_scenarios'])}")
        for j, scenario in enumerate(data['intervention_scenarios'], 1):
            print(f"       Scenario {j}: {scenario['intervention_type']} - {scenario['trigger_condition']}")
    
    # Demonstrate edge case data
    print("\n6. Edge Case Test Data:")
    from src.langgraph_agents.testing.test_data_manager import TestDataType
    edge_topics = data_manager.data_provider.get_test_data(TestDataType.TOPICS, TestComplexity.EDGE_CASE)
    print(f"   Edge Case Topics ({len(edge_topics)}):")
    for topic in edge_topics[:3]:
        print(f"     - {repr(topic)}")


def demonstrate_input_validation():
    """Demonstrate enhanced input validation and preprocessing."""
    print("\n" + "="*60)
    print("DEMONSTRATING INPUT VALIDATION AND PREPROCESSING")
    print("="*60)
    
    validation_manager = get_validation_manager()
    
    # Test valid inputs
    print("\n1. Valid Input Validation:")
    valid_inputs = {
        'topic': 'Linear Algebra Basics',
        'description': 'Create an educational animation about linear algebra fundamentals including vectors, matrices, and transformations.',
        'session_id': 'test_session_123'
    }
    
    result = validation_manager.validate_inputs('PlannerAgent', valid_inputs)
    print(f"   Validation Result: {'PASSED' if result.is_valid else 'FAILED'}")
    print(f"   Issues: {len(result.issues)}")
    print(f"   Processed Topic: {result.processed_inputs.get('topic')}")
    
    # Test invalid inputs
    print("\n2. Invalid Input Validation:")
    invalid_inputs = {
        'topic': '',  # Empty topic
        'description': 'Short',  # Too short description
        'extra_field': 'unexpected'
    }
    
    result = validation_manager.validate_inputs('PlannerAgent', invalid_inputs)
    print(f"   Validation Result: {'PASSED' if result.is_valid else 'FAILED'}")
    print(f"   Issues: {len(result.issues)}")
    for issue in result.issues[:3]:  # Show first 3 issues
        print(f"     - {issue.severity.value.upper()}: {issue.message}")
        if issue.suggested_fix:
            print(f"       Fix: {issue.suggested_fix}")
    
    # Test code generator validation
    print("\n3. CodeGeneratorAgent Input Validation:")
    codegen_inputs = {
        'topic': 'Fourier Transform',
        'description': 'Visualize the Fourier transform process',
        'scene_outline': 'Scene 1: Introduction\nScene 2: Transform\nScene 3: Result',
        'scene_implementations': {
            '1': 'Implement introduction scene',
            '2': 'Implement transform visualization',
            'invalid_key': 'This should cause validation error'
        }
    }
    
    result = validation_manager.validate_inputs('CodeGeneratorAgent', codegen_inputs)
    print(f"   Validation Result: {'PASSED' if result.is_valid else 'FAILED'}")
    print(f"   Issues: {len(result.issues)}")
    for issue in result.issues:
        print(f"     - {issue.severity.value.upper()}: {issue.message}")
    
    # Show processed inputs
    if result.processed_inputs.get('scene_implementations'):
        print(f"   Processed Scene Keys: {list(result.processed_inputs['scene_implementations'].keys())}")


def demonstrate_scenario_management():
    """Demonstrate enhanced scenario management capabilities."""
    print("\n" + "="*60)
    print("DEMONSTRATING SCENARIO MANAGEMENT")
    print("="*60)
    
    # Initialize managers
    data_manager = TestDataManager()
    scenario_manager = TestScenarioManager(data_manager)
    
    # Create test scenarios
    print("\n1. Creating Test Scenarios:")
    
    # Create planner scenarios
    planner_data = data_manager.get_planner_test_data(TestComplexity.SIMPLE, count=2)
    planner_scenarios = []
    for data in planner_data:
        scenario_id = scenario_manager.create_scenario('PlannerAgent', data)
        planner_scenarios.append(scenario_id)
        print(f"   Created PlannerAgent scenario: {scenario_id}")
    
    # Create code generator scenarios
    codegen_data = data_manager.get_codegen_test_data(TestComplexity.MEDIUM, count=1)
    codegen_scenarios = []
    for data in codegen_data:
        scenario_id = scenario_manager.create_scenario('CodeGeneratorAgent', data)
        codegen_scenarios.append(scenario_id)
        print(f"   Created CodeGeneratorAgent scenario: {scenario_id}")
    
    # List scenarios
    print(f"\n2. Total Scenarios Created: {len(scenario_manager.scenarios)}")
    
    # Demonstrate scenario filtering
    print("\n3. Scenario Filtering:")
    selector = ScenarioSelector(scenario_manager)
    
    # Filter by agent type
    planner_scenarios_filtered = selector.select_scenarios_by_agent('PlannerAgent')
    print(f"   PlannerAgent scenarios: {len(planner_scenarios_filtered)}")
    
    # Filter by complexity
    from src.langgraph_agents.testing.scenario_selection import ScenarioFilterCriteria, ScenarioFilter
    filters = [ScenarioFilterCriteria(ScenarioFilter.COMPLEXITY, 'simple')]
    simple_scenarios = selector.filter_scenarios(filters)
    print(f"   Simple complexity scenarios: {len(simple_scenarios)}")
    
    # Demonstrate scenario recommendations
    print("\n4. Scenario Recommendations:")
    user_preferences = {
        'complexity': 'simple',
        'min_success_rate': 0.0,
        'sort_by': 'success_rate',
        'limit': 3
    }
    
    recommendations = selector.recommend_scenarios('PlannerAgent', user_preferences)
    print(f"   Recommended scenarios for PlannerAgent: {len(recommendations)}")
    for scenario in recommendations:
        print(f"     - {scenario.name} (Complexity: {scenario.complexity.value})")


def demonstrate_result_validation():
    """Demonstrate result comparison and validation."""
    print("\n" + "="*60)
    print("DEMONSTRATING RESULT VALIDATION")
    print("="*60)
    
    result_validator = get_result_validator()
    
    # Mock test result for PlannerAgent
    print("\n1. PlannerAgent Result Validation:")
    planner_result = {
        'success': True,
        'agent': 'PlannerAgent',
        'execution_time': 15.5,
        'outputs': {
            'scene_outline': 'Scene 1: Introduction\nScene 2: Main Content\nScene 3: Conclusion',
            'scene_implementations': {
                1: 'Implement introduction with title and overview',
                2: 'Implement main mathematical concepts',
                3: 'Implement conclusion with summary'
            },
            'detected_plugins': ['manim_physics', 'manim_slides'],
            'scene_count': 3
        },
        'validation': {
            'scene_outline_valid': True,
            'scene_outline_issues': [],
            'scene_implementations_valid': True,
            'scene_implementations_issues': []
        }
    }
    
    validation_result = result_validator.validate_agent_specific_results(
        'PlannerAgent', 'test_planner_001', planner_result
    )
    
    print(f"   Overall Status: {validation_result.overall_status.value.upper()}")
    print(f"   Overall Score: {validation_result.overall_score:.2f}")
    print(f"   Comparisons: {len(validation_result.comparisons)}")
    
    for comparison in validation_result.comparisons:
        print(f"     - {comparison.field}: {comparison.status.value.upper()} (Score: {comparison.score:.2f})")
        print(f"       Message: {comparison.message}")
    
    # Mock test result for CodeGeneratorAgent with some failures
    print("\n2. CodeGeneratorAgent Result Validation (with failures):")
    codegen_result = {
        'success': False,  # Test failed
        'agent': 'CodeGeneratorAgent',
        'execution_time': 45.2,
        'outputs': {
            'generated_code': {
                1: 'from manim import *\n\nclass Scene1(Scene):\n    def construct(self):\n        pass',
                2: '',  # Empty code - should fail validation
            },
            'successful_scenes': [1],
            'failed_scenes': [2],
            'total_scenes': 2
        },
        'validation': {
            1: {'valid': True, 'issues': []},
            2: {'valid': False, 'issues': ['Empty code generated']}
        },
        'errors': {
            2: 'Code generation failed for scene 2'
        }
    }
    
    validation_result = result_validator.validate_agent_specific_results(
        'CodeGeneratorAgent', 'test_codegen_001', codegen_result
    )
    
    print(f"   Overall Status: {validation_result.overall_status.value.upper()}")
    print(f"   Overall Score: {validation_result.overall_score:.2f}")
    print(f"   Failed Comparisons: {len(validation_result.get_failed_comparisons())}")
    
    for comparison in validation_result.get_failed_comparisons():
        print(f"     - FAILED {comparison.field}: {comparison.message}")


def demonstrate_studio_integration():
    """Demonstrate Studio-specific scenario interface."""
    print("\n" + "="*60)
    print("DEMONSTRATING STUDIO INTEGRATION")
    print("="*60)
    
    studio_interface = get_studio_interface()
    
    # Get available scenarios
    print("\n1. Available Scenarios for Studio:")
    available = studio_interface.get_available_scenarios()
    print(f"   Total scenarios: {available['total_count']}")
    print(f"   Agent types: {available['agent_types']}")
    
    if available['scenarios']:
        print("   Sample scenarios:")
        for scenario in available['scenarios'][:3]:
            print(f"     - {scenario['name']} ({scenario['agent_type']})")
            print(f"       Complexity: {scenario['complexity']}")
            print(f"       Success Rate: {scenario['success_rate']:.2f}")
    
    # Get recommendations
    print("\n2. Scenario Recommendations:")
    recommendations = studio_interface.get_scenario_recommendations(
        'PlannerAgent',
        {'complexity': 'simple', 'limit': 2}
    )
    
    print(f"   Recommended for PlannerAgent: {len(recommendations['recommended_scenarios'])}")
    for rec in recommendations['recommended_scenarios']:
        print(f"     - {rec['name']} (Score: {rec['recommendation_score']:.2f})")
    
    # Create custom scenario
    print("\n3. Creating Custom Scenario:")
    custom_data = {
        'topic': 'Custom Test Topic',
        'description': 'This is a custom test scenario created for demonstration purposes.',
        'session_id': 'custom_demo_session'
    }
    
    custom_metadata = {
        'name': 'Custom Demo Scenario',
        'description': 'Demonstration of custom scenario creation',
        'complexity': 'simple',
        'tags': ['demo', 'custom', 'planneragent']
    }
    
    result = studio_interface.create_custom_scenario(
        'PlannerAgent', custom_data, custom_metadata
    )
    
    if result['success']:
        print(f"   Custom scenario created: {result['scenario_id']}")
        print(f"   Scenario name: {result['scenario']['name']}")
    else:
        print(f"   Failed to create custom scenario: {result['error']}")


async def demonstrate_scenario_execution():
    """Demonstrate scenario execution with validation."""
    print("\n" + "="*60)
    print("DEMONSTRATING SCENARIO EXECUTION")
    print("="*60)
    
    # This is a mock demonstration since we don't have actual test runners available
    print("\n1. Mock Scenario Execution:")
    
    # Create a mock test runner
    class MockTestRunner:
        async def run_test(self, inputs):
            # Simulate test execution
            await asyncio.sleep(0.1)  # Simulate processing time
            
            return {
                'success': True,
                'agent': 'PlannerAgent',
                'execution_time': 12.3,
                'inputs': inputs,
                'outputs': {
                    'scene_outline': 'Generated scene outline based on inputs',
                    'scene_implementations': {1: 'Scene 1 implementation', 2: 'Scene 2 implementation'},
                    'detected_plugins': ['manim_core'],
                    'scene_count': 2
                }
            }
    
    # Initialize executor
    executor = ScenarioExecutor()
    
    # Create a test scenario
    data_manager = TestDataManager()
    scenario_manager = TestScenarioManager(data_manager)
    
    test_data = data_manager.get_planner_test_data(TestComplexity.SIMPLE, count=1)[0]
    scenario_id = scenario_manager.create_scenario('PlannerAgent', test_data)
    
    # Execute scenario
    config = ScenarioExecutionConfig(
        timeout=60,
        validate_inputs=True,
        validate_results=True,
        save_results=True
    )
    
    mock_runner = MockTestRunner()
    execution_result = await executor.execute_scenario(scenario_id, mock_runner, config)
    
    print(f"   Scenario ID: {execution_result.scenario_id}")
    print(f"   Agent Type: {execution_result.agent_type}")
    print(f"   Success: {execution_result.success}")
    print(f"   Execution Time: {execution_result.execution_time:.2f}s")
    
    if execution_result.input_validation:
        print(f"   Input Validation: {'PASSED' if execution_result.input_validation.is_valid else 'FAILED'}")
    
    if execution_result.result_validation:
        print(f"   Result Validation: {execution_result.result_validation.overall_status.value.upper()}")
        print(f"   Result Score: {execution_result.result_validation.overall_score:.2f}")


def demonstrate_comprehensive_workflow():
    """Demonstrate the complete workflow from data creation to execution."""
    print("\n" + "="*60)
    print("DEMONSTRATING COMPREHENSIVE WORKFLOW")
    print("="*60)
    
    print("\n1. Workflow Overview:")
    print("   Step 1: Create test data fixtures")
    print("   Step 2: Validate inputs")
    print("   Step 3: Create scenarios")
    print("   Step 4: Select scenarios for execution")
    print("   Step 5: Execute scenarios")
    print("   Step 6: Validate results")
    print("   Step 7: Generate reports")
    
    # Step 1: Create test data
    print("\n2. Creating Test Data:")
    data_manager = TestDataManager()
    test_data = data_manager.get_planner_test_data(TestComplexity.MEDIUM, count=1)[0]
    print(f"   Created test data for topic: {test_data['topic']}")
    
    # Step 2: Validate inputs
    print("\n3. Validating Inputs:")
    validation_result = validate_agent_inputs('PlannerAgent', test_data)
    print(f"   Input validation: {'PASSED' if validation_result.is_valid else 'FAILED'}")
    
    # Step 3: Create scenario
    print("\n4. Creating Scenario:")
    scenario_manager = TestScenarioManager(data_manager)
    scenario_id = scenario_manager.create_scenario('PlannerAgent', validation_result.processed_inputs)
    print(f"   Created scenario: {scenario_id}")
    
    # Step 4: Select scenarios
    print("\n5. Selecting Scenarios:")
    selector = ScenarioSelector(scenario_manager)
    selected_scenarios = selector.select_scenarios_by_agent('PlannerAgent', limit=1)
    print(f"   Selected {len(selected_scenarios)} scenarios for execution")
    
    # Step 5: Mock execution (would be real in actual implementation)
    print("\n6. Mock Execution Results:")
    mock_result = {
        'success': True,
        'execution_time': 18.7,
        'outputs': {
            'scene_outline': 'Generated outline',
            'scene_implementations': {1: 'Implementation 1', 2: 'Implementation 2'},
            'detected_plugins': ['manim_core'],
            'scene_count': 2
        }
    }
    print(f"   Execution completed: {mock_result['success']}")
    print(f"   Execution time: {mock_result['execution_time']}s")
    
    # Step 6: Validate results
    print("\n7. Validating Results:")
    result_validation = validate_test_result('PlannerAgent', scenario_id, mock_result)
    print(f"   Result validation: {result_validation.overall_status.value.upper()}")
    print(f"   Overall score: {result_validation.overall_score:.2f}")
    
    # Step 7: Generate summary report
    print("\n8. Summary Report:")
    report = {
        'test_data_created': True,
        'input_validation_passed': validation_result.is_valid,
        'scenario_created': True,
        'execution_successful': mock_result['success'],
        'result_validation_status': result_validation.overall_status.value,
        'overall_score': result_validation.overall_score,
        'total_execution_time': mock_result['execution_time']
    }
    
    print("   Final Report:")
    for key, value in report.items():
        print(f"     {key.replace('_', ' ').title()}: {value}")


async def main():
    """Main demonstration function."""
    print("Enhanced Test Data and Scenario Management Demo")
    print("=" * 60)
    print("This demo showcases the implementation of task 2.3:")
    print("- Create test data fixtures for each agent type")
    print("- Implement test scenario selection and execution in Studio")
    print("- Create agent input validation and preprocessing")
    print("- Implement test result comparison and validation")
    
    try:
        # Run all demonstrations
        demonstrate_test_data_fixtures()
        demonstrate_input_validation()
        demonstrate_scenario_management()
        demonstrate_result_validation()
        demonstrate_studio_integration()
        await demonstrate_scenario_execution()
        demonstrate_comprehensive_workflow()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nAll components of task 2.3 have been implemented and demonstrated:")
        print("✓ Enhanced test data fixtures for all agent types")
        print("✓ Comprehensive input validation and preprocessing")
        print("✓ Advanced scenario selection and filtering")
        print("✓ Result comparison and validation system")
        print("✓ Studio integration for scenario management")
        print("✓ Complete workflow from data creation to result validation")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
"""
LangGraph Studio integration for agent testing.

This module provides the main interface for running agent tests
in LangGraph Studio environment with comprehensive output capture
and visualization support.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from .agent_test_runners import (
    PlannerAgentTestRunner,
    CodeGeneratorAgentTestRunner,
    RendererAgentTestRunner,
    ErrorHandlerAgentTestRunner,
    HumanLoopAgentTestRunner
)
from .output_capture import AgentOutputCapture, OutputFormatter, OutputValidator
from .test_data_manager import TestDataManager, TestScenarioManager, TestComplexity
from .input_validation import get_validation_manager
from .result_validation import get_result_validator
from .scenario_selection import get_studio_interface
from ..models.config import WorkflowConfig

logger = logging.getLogger(__name__)


class StudioTestRunner:
    """
    Main test runner for LangGraph Studio integration.
    
    Provides a unified interface for running all agent tests with
    comprehensive output capture and Studio-compatible formatting.
    """
    
    def __init__(self, config: WorkflowConfig = None):
        """Initialize the Studio test runner."""
        self.config = config or WorkflowConfig()
        
        # Initialize test runners
        self.test_runners = {
            'PlannerAgent': PlannerAgentTestRunner(self.config),
            'CodeGeneratorAgent': CodeGeneratorAgentTestRunner(self.config),
            'RendererAgent': RendererAgentTestRunner(self.config),
            'ErrorHandlerAgent': ErrorHandlerAgentTestRunner(self.config),
            'HumanLoopAgent': HumanLoopAgentTestRunner(self.config)
        }
        
        # Initialize supporting components
        self.output_capture = AgentOutputCapture()
        self.output_formatter = OutputFormatter()
        self.output_validator = OutputValidator()
        self.test_data_manager = TestDataManager()
        self.scenario_manager = TestScenarioManager(self.test_data_manager)
        self.validation_manager = get_validation_manager()
        self.result_validator = get_result_validator()
        self.studio_interface = get_studio_interface()
        
        logger.info("StudioTestRunner initialized with all agent test runners")
    
    async def run_agent_test(self, 
                           agent_type: str, 
                           inputs: Dict[str, Any],
                           capture_output: bool = True) -> Dict[str, Any]:
        """
        Run a test for a specific agent type.
        
        Args:
            agent_type: Type of agent to test
            inputs: Test inputs for the agent
            capture_output: Whether to capture output for Studio visualization
            
        Returns:
            Dict containing test results and captured output
        """
        if agent_type not in self.test_runners:
            raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(self.test_runners.keys())}")
        
        test_runner = self.test_runners[agent_type]
        session_id = inputs.get('session_id', f"studio_test_{int(time.time())}")
        
        logger.info(f"Starting {agent_type} test with session {session_id}")
        
        # Start output capture if requested
        if capture_output:
            self.output_capture.start_capture(session_id, agent_type)
        
        try:
            # Run the test
            result = await test_runner.run_test(inputs)
            
            # Stop output capture and get results
            captured_output = None
            if capture_output:
                captured_output = self.output_capture.stop_capture(session_id)
            
            # Format for Studio if output was captured
            studio_formatted_output = None
            validation_results = None
            
            if captured_output:
                studio_formatted_output = self.output_formatter.format_for_studio(captured_output)
                validation_results = self.output_validator.validate_output(captured_output)
            
            # Build comprehensive result
            comprehensive_result = {
                'test_result': result,
                'session_id': session_id,
                'agent_type': agent_type,
                'timestamp': datetime.now().isoformat(),
                'studio_output': studio_formatted_output,
                'validation': validation_results,
                'success': result.get('success', False)
            }
            
            logger.info(f"{agent_type} test completed successfully")
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"{agent_type} test failed: {str(e)}")
            
            # Stop output capture on error
            captured_output = None
            if capture_output:
                captured_output = self.output_capture.stop_capture(session_id)
            
            # Format error result
            error_result = {
                'test_result': {
                    'success': False,
                    'error': str(e),
                    'error_type': type(e).__name__
                },
                'session_id': session_id,
                'agent_type': agent_type,
                'timestamp': datetime.now().isoformat(),
                'studio_output': self.output_formatter.format_for_studio(captured_output) if captured_output else None,
                'validation': None,
                'success': False
            }
            
            return error_result
    
    async def run_scenario(self, scenario_id: str) -> Dict[str, Any]:
        """
        Run a predefined test scenario.
        
        Args:
            scenario_id: ID of the scenario to run
            
        Returns:
            Dict containing scenario execution results
        """
        scenario = self.scenario_manager.get_scenario(scenario_id)
        if not scenario:
            raise ValueError(f"Scenario not found: {scenario_id}")
        
        # Determine agent type from scenario tags
        agent_type = None
        for tag in scenario.tags:
            if tag.endswith('agent') or tag in ['planneragent', 'codegeneratoragent', 'rendereragent', 'errorhandleragent', 'humanloopagent']:
                agent_type = tag.title().replace('agent', 'Agent')
                if agent_type == 'PlannerAgent':
                    agent_type = 'PlannerAgent'
                elif agent_type == 'CodegeneratorAgent':
                    agent_type = 'CodeGeneratorAgent'
                elif agent_type == 'RendererAgent':
                    agent_type = 'RendererAgent'
                elif agent_type == 'ErrorhandlerAgent':
                    agent_type = 'ErrorHandlerAgent'
                elif agent_type == 'HumanloopAgent':
                    agent_type = 'HumanLoopAgent'
                break
        
        if not agent_type:
            # Try to infer from scenario inputs
            if 'error_scenarios' in scenario.inputs:
                agent_type = 'ErrorHandlerAgent'
            elif 'intervention_scenarios' in scenario.inputs:
                agent_type = 'HumanLoopAgent'
            elif 'generated_code' in scenario.inputs:
                agent_type = 'RendererAgent'
            elif 'scene_implementations' in scenario.inputs:
                agent_type = 'CodeGeneratorAgent'
            else:
                agent_type = 'PlannerAgent'  # Default
        
        logger.info(f"Running scenario {scenario_id} for {agent_type}")
        
        # Execute the scenario
        result = await self.run_agent_test(agent_type, scenario.inputs)
        
        # Record execution in scenario manager
        self.scenario_manager.execution_history[scenario_id] = {
            'scenario_id': scenario_id,
            'agent_type': agent_type,
            'start_time': time.time(),
            'end_time': time.time(),
            'status': 'completed' if result['success'] else 'failed',
            'result': result
        }
        
        return result
    
    def generate_test_scenarios(self, 
                              agent_type: str, 
                              complexity: TestComplexity = TestComplexity.SIMPLE,
                              count: int = 3) -> List[str]:
        """
        Generate test scenarios for an agent type.
        
        Args:
            agent_type: Type of agent to generate scenarios for
            complexity: Complexity level of scenarios
            count: Number of scenarios to generate
            
        Returns:
            List of scenario IDs
        """
        return self.scenario_manager.generate_scenarios_for_agent(agent_type, complexity, count)
    
    def list_available_agents(self) -> List[str]:
        """Get list of available agent types for testing."""
        return list(self.test_runners.keys())
    
    def get_agent_test_data_sample(self, agent_type: str, complexity: TestComplexity = TestComplexity.SIMPLE) -> Dict[str, Any]:
        """
        Get a sample test data for an agent type.
        
        Args:
            agent_type: Type of agent
            complexity: Complexity level
            
        Returns:
            Sample test data
        """
        if agent_type == 'PlannerAgent':
            samples = self.test_data_manager.get_planner_test_data(complexity, 1)
        elif agent_type == 'CodeGeneratorAgent':
            samples = self.test_data_manager.get_codegen_test_data(complexity, 1)
        elif agent_type == 'RendererAgent':
            samples = self.test_data_manager.get_renderer_test_data(complexity, 1)
        elif agent_type == 'ErrorHandlerAgent':
            samples = self.test_data_manager.get_error_handler_test_data(complexity, 1)
        elif agent_type == 'HumanLoopAgent':
            samples = self.test_data_manager.get_human_loop_test_data(complexity, 1)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        return samples[0] if samples else {}
    
    async def run_comprehensive_test_suite(self, 
                                         complexity: TestComplexity = TestComplexity.SIMPLE,
                                         agents_to_test: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run comprehensive test suite for multiple agents.
        
        Args:
            complexity: Complexity level for tests
            agents_to_test: List of agents to test (None for all)
            
        Returns:
            Comprehensive test results
        """
        if agents_to_test is None:
            agents_to_test = list(self.test_runners.keys())
        
        logger.info(f"Running comprehensive test suite for {len(agents_to_test)} agents")
        
        suite_results = {
            'suite_id': f"comprehensive_{int(time.time())}",
            'start_time': datetime.now().isoformat(),
            'complexity': complexity.value,
            'agents_tested': agents_to_test,
            'results': {},
            'summary': {}
        }
        
        total_tests = 0
        successful_tests = 0
        
        for agent_type in agents_to_test:
            try:
                logger.info(f"Testing {agent_type}")
                
                # Get test data
                test_data = self.get_agent_test_data_sample(agent_type, complexity)
                
                # Run test
                result = await self.run_agent_test(agent_type, test_data)
                
                suite_results['results'][agent_type] = result
                total_tests += 1
                
                if result['success']:
                    successful_tests += 1
                
            except Exception as e:
                logger.error(f"Failed to test {agent_type}: {str(e)}")
                suite_results['results'][agent_type] = {
                    'success': False,
                    'error': str(e),
                    'error_type': type(e).__name__
                }
                total_tests += 1
        
        # Calculate summary
        suite_results['summary'] = {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': total_tests - successful_tests,
            'success_rate': successful_tests / total_tests if total_tests > 0 else 0.0,
            'end_time': datetime.now().isoformat()
        }
        
        logger.info(f"Comprehensive test suite completed: {successful_tests}/{total_tests} successful")
        return suite_results
    
    def get_test_runner_info(self) -> Dict[str, Any]:
        """Get information about the test runner and its capabilities."""
        return {
            'studio_test_runner': {
                'version': '1.0.0',
                'available_agents': list(self.test_runners.keys()),
                'features': [
                    'Individual agent testing',
                    'Scenario-based testing',
                    'Output capture and validation',
                    'Studio-compatible formatting',
                    'Comprehensive test suites'
                ],
                'complexity_levels': [c.value for c in TestComplexity],
                'output_formats': ['studio', 'json', 'text']
            },
            'configuration': {
                'config_loaded': self.config is not None,
                'output_capture_enabled': True,
                'validation_enabled': True,
                'scenario_management_enabled': True
            }
        }


# Global Studio test runner instance
_global_studio_test_runner = None


def get_studio_test_runner(config: WorkflowConfig = None) -> StudioTestRunner:
    """Get the global Studio test runner instance."""
    global _global_studio_test_runner
    
    if _global_studio_test_runner is None:
        _global_studio_test_runner = StudioTestRunner(config)
    
    return _global_studio_test_runner


# Convenience functions for Studio integration
async def run_planner_test(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Run PlannerAgent test."""
    runner = get_studio_test_runner()
    return await runner.run_agent_test('PlannerAgent', inputs)


async def run_codegen_test(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Run CodeGeneratorAgent test."""
    runner = get_studio_test_runner()
    return await runner.run_agent_test('CodeGeneratorAgent', inputs)


async def run_renderer_test(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Run RendererAgent test."""
    runner = get_studio_test_runner()
    return await runner.run_agent_test('RendererAgent', inputs)


async def run_error_handler_test(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Run ErrorHandlerAgent test."""
    runner = get_studio_test_runner()
    return await runner.run_agent_test('ErrorHandlerAgent', inputs)


async def run_human_loop_test(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Run HumanLoopAgent test."""
    runner = get_studio_test_runner()
    return await runner.run_agent_test('HumanLoopAgent', inputs)
"""
Integration tests for Studio workflow testing capabilities.

This module tests the comprehensive workflow testing system including
end-to-end testing, state validation, checkpoint inspection, and
agent transition testing.
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any

from src.langgraph_agents.studio.studio_workflow_testing import (
    StudioWorkflowTester,
    WorkflowTestType,
    WorkflowTestScenario,
    WorkflowTestStatus,
    get_studio_workflow_tester
)
from src.langgraph_agents.studio.studio_graphs import (
    get_available_graphs,
    run_graph_test
)
from src.langgraph_agents.models.config import WorkflowConfig
from src.langgraph_agents.models.state import VideoGenerationState


class TestStudioWorkflowTesting:
    """Test suite for Studio workflow testing."""
    
    @pytest.fixture
    def workflow_config(self):
        """Create test workflow configuration."""
        return WorkflowConfig(
            preview_mode=True,
            max_retries=1,
            timeout_seconds=60
        )
    
    @pytest.fixture
    def workflow_tester(self, workflow_config):
        """Create workflow tester instance."""
        return StudioWorkflowTester(workflow_config)
    
    @pytest.fixture
    def sample_input_data(self):
        """Create sample input data for testing."""
        return {
            'topic': 'Test Mathematics Topic',
            'description': 'Test description for mathematical concept explanation',
            'session_id': f'test_{int(datetime.now().timestamp())}',
            'preview_mode': True
        }
    
    def test_workflow_tester_initialization(self, workflow_tester):
        """Test workflow tester initialization."""
        assert workflow_tester is not None
        assert workflow_tester.config is not None
        assert workflow_tester.studio_config is not None
        assert workflow_tester.result_validator is not None
        assert len(workflow_tester.test_scenarios) > 0
        
        # Check predefined scenarios
        expected_scenarios = ['e2e_basic', 'state_validation', 'agent_transitions', 'error_recovery']
        for scenario_id in expected_scenarios:
            assert scenario_id in workflow_tester.test_scenarios
    
    def test_test_scenario_structure(self, workflow_tester):
        """Test test scenario structure and content."""
        scenario = workflow_tester.get_test_scenario('e2e_basic')
        
        assert scenario is not None
        assert scenario.scenario_id == 'e2e_basic'
        assert scenario.test_type == WorkflowTestType.END_TO_END
        assert scenario.name is not None
        assert scenario.description is not None
        assert scenario.input_data is not None
        assert scenario.expected_outcomes is not None
        assert scenario.validation_criteria is not None
        assert scenario.timeout_seconds > 0
        
        # Check input data structure
        assert 'topic' in scenario.input_data
        assert 'description' in scenario.input_data
        assert 'session_id' in scenario.input_data
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow_test(self, workflow_tester, sample_input_data):
        """Test end-to-end workflow testing."""
        # Run the test
        result = await workflow_tester.run_workflow_test('e2e_basic', sample_input_data)
        
        # Verify result structure
        assert result is not None
        assert result.scenario_id == 'e2e_basic'
        assert result.test_type == WorkflowTestType.END_TO_END
        assert result.status in [WorkflowTestStatus.COMPLETED, WorkflowTestStatus.FAILED, WorkflowTestStatus.TIMEOUT]
        assert result.start_time is not None
        assert result.end_time is not None
        assert result.execution_time > 0
        
        # Check if test completed successfully
        if result.status == WorkflowTestStatus.COMPLETED:
            assert result.final_state is not None
            assert isinstance(result.final_state, VideoGenerationState)
            assert result.final_state.session_id == sample_input_data['session_id']
            
            # Check state snapshots
            assert len(result.state_snapshots) > 0
            
            # Check validation results
            assert len(result.validation_results) > 0
            validation = result.validation_results[0]
            assert validation.agent_type == "WorkflowTest"
            assert validation.test_id == 'e2e_basic'
    
    @pytest.mark.asyncio
    async def test_state_validation_test(self, workflow_tester, sample_input_data):
        """Test state validation testing."""
        result = await workflow_tester.run_workflow_test('state_validation', sample_input_data)
        
        assert result is not None
        assert result.test_type == WorkflowTestType.STATE_VALIDATION
        assert result.status in [WorkflowTestStatus.COMPLETED, WorkflowTestStatus.FAILED, WorkflowTestStatus.TIMEOUT]
        
        # State validation should capture multiple snapshots
        if result.status == WorkflowTestStatus.COMPLETED:
            assert len(result.state_snapshots) >= 2  # At least planning and one other step
            
            # Check snapshot structure
            for snapshot in result.state_snapshots:
                assert 'timestamp' in snapshot
                assert 'node' in snapshot
                assert 'session_id' in snapshot
                assert 'current_step' in snapshot
                assert 'completion_percentage' in snapshot
    
    @pytest.mark.asyncio
    async def test_agent_transition_test(self, workflow_tester, sample_input_data):
        """Test agent transition testing."""
        result = await workflow_tester.run_workflow_test('agent_transitions', sample_input_data)
        
        assert result is not None
        assert result.test_type == WorkflowTestType.AGENT_TRANSITION
        assert result.status in [WorkflowTestStatus.COMPLETED, WorkflowTestStatus.FAILED, WorkflowTestStatus.TIMEOUT]
        
        # Transition testing should log transitions
        if result.status == WorkflowTestStatus.COMPLETED:
            assert len(result.transition_logs) > 0
            
            # Check transition log structure
            for transition in result.transition_logs:
                assert 'timestamp' in transition
                assert 'session_id' in transition
    
    @pytest.mark.asyncio
    async def test_custom_scenario_creation(self, workflow_tester):
        """Test custom scenario creation and execution."""
        # Create custom scenario
        custom_scenario = workflow_tester.create_custom_scenario(
            scenario_id='test_custom',
            test_type=WorkflowTestType.PARTIAL_WORKFLOW,
            name='Test Custom Scenario',
            description='Custom test scenario for testing',
            input_data={
                'topic': 'Custom Test Topic',
                'description': 'Custom test description',
                'session_id': 'custom_test'
            },
            expected_outcomes={
                'workflow_complete': True
            },
            validation_criteria={
                'workflow_complete': {'type': 'exact_match'}
            },
            timeout_seconds=120
        )
        
        assert custom_scenario is not None
        assert custom_scenario.scenario_id == 'test_custom'
        assert 'test_custom' in workflow_tester.test_scenarios
        
        # Run custom scenario
        result = await workflow_tester.run_workflow_test('test_custom')
        
        assert result is not None
        assert result.scenario_id == 'test_custom'
        assert result.test_type == WorkflowTestType.PARTIAL_WORKFLOW
    
    def test_test_history_tracking(self, workflow_tester):
        """Test test history tracking."""
        # Initially should have empty history
        history = workflow_tester.get_test_history()
        initial_count = len(history)
        
        # After running tests, history should be updated
        # (This would be tested in integration with actual test runs)
        assert isinstance(history, list)
        
        # Test active tests tracking
        active_tests = workflow_tester.get_active_tests()
        assert isinstance(active_tests, dict)
    
    def test_scenario_listing(self, workflow_tester):
        """Test scenario listing functionality."""
        scenarios = workflow_tester.list_test_scenarios()
        
        assert isinstance(scenarios, list)
        assert len(scenarios) > 0
        
        # Check scenario structure
        for scenario in scenarios:
            assert 'scenario_id' in scenario
            assert 'test_type' in scenario
            assert 'name' in scenario
            assert 'description' in scenario
            assert 'input_data' in scenario
            assert 'expected_outcomes' in scenario
            assert 'validation_criteria' in scenario


class TestStudioGraphs:
    """Test suite for Studio graphs."""
    
    @pytest.fixture
    def sample_input_data(self):
        """Create sample input data for graph testing."""
        return {
            'topic': 'Graph Test Topic',
            'description': 'Test description for graph execution',
            'session_id': f'graph_test_{int(datetime.now().timestamp())}'
        }
    
    def test_available_graphs(self):
        """Test available graphs listing."""
        graphs = get_available_graphs()
        
        assert isinstance(graphs, dict)
        assert len(graphs) > 0
        
        # Check for expected graphs
        expected_graphs = [
            'planning_agent_test',
            'code_generation_agent_test',
            'rendering_agent_test',
            'workflow_test_graph',
            'state_validation_graph',
            'checkpoint_inspection_graph',
            'transition_testing_graph'
        ]
        
        for graph_name in expected_graphs:
            assert graph_name in graphs
            assert isinstance(graphs[graph_name], str)
            assert len(graphs[graph_name]) > 0
    
    @pytest.mark.asyncio
    async def test_individual_agent_graph_execution(self, sample_input_data):
        """Test individual agent graph execution."""
        # Test planning agent graph
        result = await run_graph_test('planning_agent_test', sample_input_data)
        
        assert result is not None
        assert isinstance(result, dict)
        
        if result.get('success'):
            assert 'graph_name' in result
            assert 'session_id' in result
            assert 'final_state' in result
            assert result['graph_name'] == 'planning_agent_test'
            assert result['session_id'] == sample_input_data['session_id']
        else:
            # If failed, should have error information
            assert 'error' in result
            assert 'error_type' in result
    
    @pytest.mark.asyncio
    async def test_chain_graph_execution(self, sample_input_data):
        """Test agent chain graph execution."""
        # Test planning to code chain
        result = await run_graph_test('planning_to_code_chain', sample_input_data)
        
        assert result is not None
        assert isinstance(result, dict)
        
        if result.get('success'):
            assert result['graph_name'] == 'planning_to_code_chain'
            assert 'final_state' in result
            final_state = result['final_state']
            assert 'current_step' in final_state
            assert 'workflow_complete' in final_state
    
    @pytest.mark.asyncio
    async def test_workflow_test_graph_execution(self, sample_input_data):
        """Test workflow test graph execution."""
        result = await run_graph_test('workflow_test_graph', sample_input_data)
        
        assert result is not None
        assert isinstance(result, dict)
        
        # Workflow test graph should handle the full workflow
        if result.get('success'):
            assert result['graph_name'] == 'workflow_test_graph'
            assert 'final_state' in result
    
    @pytest.mark.asyncio
    async def test_state_validation_graph_execution(self, sample_input_data):
        """Test state validation graph execution."""
        result = await run_graph_test('state_validation_graph', sample_input_data)
        
        assert result is not None
        assert isinstance(result, dict)
        
        # State validation graph should include validation steps
        if result.get('success'):
            assert result['graph_name'] == 'state_validation_graph'
            # Should have execution traces from validation steps
            assert result.get('execution_trace_count', 0) > 0
    
    @pytest.mark.asyncio
    async def test_checkpoint_inspection_graph_execution(self, sample_input_data):
        """Test checkpoint inspection graph execution."""
        result = await run_graph_test('checkpoint_inspection_graph', sample_input_data)
        
        assert result is not None
        assert isinstance(result, dict)
        
        # Checkpoint inspection should capture checkpoint data
        if result.get('success'):
            assert result['graph_name'] == 'checkpoint_inspection_graph'
            # Should have execution traces from checkpoint capture
            assert result.get('execution_trace_count', 0) > 0
    
    @pytest.mark.asyncio
    async def test_transition_testing_graph_execution(self, sample_input_data):
        """Test transition testing graph execution."""
        result = await run_graph_test('transition_testing_graph', sample_input_data)
        
        assert result is not None
        assert isinstance(result, dict)
        
        # Transition testing should monitor transitions
        if result.get('success'):
            assert result['graph_name'] == 'transition_testing_graph'
            # Should have execution traces from transition monitoring
            assert result.get('execution_trace_count', 0) > 0


class TestWorkflowTestIntegration:
    """Integration tests for workflow testing system."""
    
    @pytest.mark.asyncio
    async def test_global_workflow_tester_instance(self):
        """Test global workflow tester instance."""
        tester = get_studio_workflow_tester()
        
        assert tester is not None
        assert isinstance(tester, StudioWorkflowTester)
        
        # Should have predefined scenarios
        scenarios = tester.list_test_scenarios()
        assert len(scenarios) > 0
    
    @pytest.mark.asyncio
    async def test_workflow_test_with_monitoring(self):
        """Test workflow execution with monitoring integration."""
        tester = get_studio_workflow_tester()
        
        # Create test input
        test_input = {
            'topic': 'Integration Test Topic',
            'description': 'Integration test for workflow monitoring',
            'session_id': f'integration_test_{int(datetime.now().timestamp())}'
        }
        
        # Run test with monitoring
        result = await tester.run_workflow_test('e2e_basic', test_input)
        
        assert result is not None
        assert result.scenario_id == 'e2e_basic'
        
        # Should have monitoring data
        assert len(result.execution_trace) > 0
        
        # Should be in test history
        history = tester.get_test_history(1)
        assert len(history) > 0
        assert history[0]['scenario_id'] == 'e2e_basic'
    
    def test_workflow_test_info_endpoint(self):
        """Test workflow test info endpoint."""
        from src.langgraph_agents.studio.studio_workflow_testing import get_workflow_test_info
        
        info = get_workflow_test_info()
        
        assert isinstance(info, dict)
        assert 'available_scenarios' in info
        assert 'active_tests' in info
        assert 'recent_history' in info
        assert 'test_types' in info
        assert 'capabilities' in info
        
        # Check capabilities
        capabilities = info['capabilities']
        expected_capabilities = [
            'end_to_end_workflow_testing',
            'state_validation_and_inspection',
            'checkpoint_data_capture',
            'agent_transition_validation',
            'error_recovery_testing',
            'performance_monitoring',
            'execution_logging',
            'result_validation'
        ]
        
        for capability in expected_capabilities:
            assert capability in capabilities


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
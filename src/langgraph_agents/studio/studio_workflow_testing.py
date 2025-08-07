"""
Studio workflow testing environment for end-to-end agent workflow validation.

This module provides comprehensive workflow testing capabilities through the
LangGraph Studio interface, including state validation, checkpoint inspection,
agent transition testing, and execution logging.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

from ..models.state import VideoGenerationState
from ..models.config import WorkflowConfig
from ..models.errors import WorkflowError, ErrorType, ErrorSeverity
from ..workflow_graph import VideoGenerationWorkflow, create_workflow
from .studio_integration import studio_monitor, studio_registry
from .studio_workflow_config import StudioWorkflowConfig
from ..testing.result_validation import ResultValidator, ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)


class WorkflowTestType(Enum):
    """Types of workflow tests."""
    END_TO_END = "end_to_end"
    PARTIAL_WORKFLOW = "partial_workflow"
    STATE_VALIDATION = "state_validation"
    CHECKPOINT_INSPECTION = "checkpoint_inspection"
    AGENT_TRANSITION = "agent_transition"
    ERROR_RECOVERY = "error_recovery"
    PERFORMANCE = "performance"


class WorkflowTestStatus(Enum):
    """Status of workflow tests."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class WorkflowTestScenario:
    """Test scenario for workflow testing."""
    
    scenario_id: str
    test_type: WorkflowTestType
    name: str
    description: str
    input_data: Dict[str, Any]
    expected_outcomes: Dict[str, Any]
    validation_criteria: Dict[str, Any]
    timeout_seconds: int = 300
    checkpoint_intervals: List[str] = field(default_factory=list)
    state_validation_points: List[str] = field(default_factory=list)
    transition_validations: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'scenario_id': self.scenario_id,
            'test_type': self.test_type.value,
            'name': self.name,
            'description': self.description,
            'input_data': self.input_data,
            'expected_outcomes': self.expected_outcomes,
            'validation_criteria': self.validation_criteria,
            'timeout_seconds': self.timeout_seconds,
            'checkpoint_intervals': self.checkpoint_intervals,
            'state_validation_points': self.state_validation_points,
            'transition_validations': self.transition_validations
        }


@dataclass
class WorkflowTestResult:
    """Result of workflow test execution."""
    
    scenario_id: str
    test_type: WorkflowTestType
    status: WorkflowTestStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    execution_time: float = 0.0
    final_state: Optional[VideoGenerationState] = None
    state_snapshots: List[Dict[str, Any]] = field(default_factory=list)
    checkpoint_data: List[Dict[str, Any]] = field(default_factory=list)
    transition_logs: List[Dict[str, Any]] = field(default_factory=list)
    validation_results: List[ValidationResult] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'scenario_id': self.scenario_id,
            'test_type': self.test_type.value,
            'status': self.status.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'execution_time': self.execution_time,
            'final_state_summary': self._get_state_summary() if self.final_state else None,
            'state_snapshots_count': len(self.state_snapshots),
            'checkpoint_count': len(self.checkpoint_data),
            'transition_count': len(self.transition_logs),
            'validation_results': [vr.to_dict() for vr in self.validation_results],
            'performance_metrics': self.performance_metrics,
            'errors': self.errors,
            'execution_trace_count': len(self.execution_trace)
        }
    
    def _get_state_summary(self) -> Dict[str, Any]:
        """Get summary of final state."""
        if not self.final_state:
            return {}
        
        return {
            'session_id': self.final_state.session_id,
            'current_step': self.final_state.current_step,
            'workflow_complete': self.final_state.workflow_complete,
            'has_errors': self.final_state.has_errors(),
            'completion_percentage': self.final_state.get_completion_percentage() if hasattr(self.final_state, 'get_completion_percentage') else 0,
            'scene_count': len(self.final_state.scene_implementations) if self.final_state.scene_implementations else 0,
            'code_count': len(self.final_state.generated_code) if self.final_state.generated_code else 0,
            'video_count': len(self.final_state.rendered_videos) if self.final_state.rendered_videos else 0
        }


class StudioWorkflowTester:
    """Main workflow testing system for Studio environment."""
    
    def __init__(self, config: WorkflowConfig = None):
        """Initialize the workflow tester."""
        self.config = config or WorkflowConfig()
        self.studio_config = StudioWorkflowConfig()
        self.result_validator = ResultValidator()
        self.active_tests: Dict[str, WorkflowTestResult] = {}
        self.test_history: List[WorkflowTestResult] = []
        self.workflow_cache: Dict[str, VideoGenerationWorkflow] = {}
        
        # Initialize test scenarios
        self.test_scenarios = self._initialize_test_scenarios()
        
        logger.info("Studio workflow tester initialized")
    
    def _initialize_test_scenarios(self) -> Dict[str, WorkflowTestScenario]:
        """Initialize predefined test scenarios."""
        scenarios = {}
        
        # End-to-end workflow test
        scenarios['e2e_basic'] = WorkflowTestScenario(
            scenario_id='e2e_basic',
            test_type=WorkflowTestType.END_TO_END,
            name='Basic End-to-End Workflow',
            description='Test complete workflow from planning to rendering',
            input_data={
                'topic': 'Linear Equations',
                'description': 'Explain how to solve linear equations step by step with visual examples',
                'session_id': 'e2e_basic_test',
                'preview_mode': True
            },
            expected_outcomes={
                'workflow_complete': True,
                'scene_count': {'min': 2, 'max': 5},
                'has_scene_outline': True,
                'has_generated_code': True,
                'completion_percentage': {'min': 80}
            },
            validation_criteria={
                'workflow_complete': {'type': 'exact_match'},
                'scene_outline': {'type': 'pattern_match', 'params': {'pattern': r'.+'}},
                'scene_implementations': {'type': 'structural_match'},
                'generated_code': {'type': 'structural_match'}
            },
            checkpoint_intervals=['planning', 'code_generation', 'rendering'],
            state_validation_points=['planning', 'code_generation', 'rendering', 'complete'],
            transition_validations=[
                {'from': 'planning', 'to': 'code_generation', 'condition': 'has_scene_outline'},
                {'from': 'code_generation', 'to': 'rendering', 'condition': 'has_generated_code'},
                {'from': 'rendering', 'to': 'complete', 'condition': 'has_rendered_videos'}
            ]
        )
        
        # State validation test
        scenarios['state_validation'] = WorkflowTestScenario(
            scenario_id='state_validation',
            test_type=WorkflowTestType.STATE_VALIDATION,
            name='Workflow State Validation',
            description='Test state consistency and validation at each step',
            input_data={
                'topic': 'Pythagorean Theorem',
                'description': 'Demonstrate the Pythagorean theorem with geometric proofs',
                'session_id': 'state_validation_test',
                'preview_mode': True
            },
            expected_outcomes={
                'state_consistency': True,
                'valid_transitions': True,
                'no_state_corruption': True
            },
            validation_criteria={
                'state_consistency': {'type': 'exact_match'},
                'session_id': {'type': 'exact_match'},
                'current_step': {'type': 'pattern_match', 'params': {'pattern': r'^(planning|code_generation|rendering|complete|error_handler)$'}}
            },
            state_validation_points=['planning', 'code_generation', 'rendering', 'complete'],
            timeout_seconds=180
        )
        
        # Agent transition test
        scenarios['agent_transitions'] = WorkflowTestScenario(
            scenario_id='agent_transitions',
            test_type=WorkflowTestType.AGENT_TRANSITION,
            name='Agent Transition Testing',
            description='Test transitions between different agents in the workflow',
            input_data={
                'topic': 'Quadratic Functions',
                'description': 'Explain quadratic functions and their graphs',
                'session_id': 'transition_test',
                'preview_mode': True
            },
            expected_outcomes={
                'valid_transitions': True,
                'proper_state_handoff': True,
                'no_transition_errors': True
            },
            validation_criteria={
                'transition_count': {'type': 'threshold_match', 'params': {'threshold': 0.2, 'comparison_type': 'relative'}},
                'transition_timing': {'type': 'threshold_match', 'params': {'threshold': 30, 'comparison_type': 'absolute'}}
            },
            transition_validations=[
                {'from': 'planning', 'to': 'code_generation', 'condition': 'has_scene_implementations'},
                {'from': 'code_generation', 'to': 'rendering', 'condition': 'has_generated_code'},
                {'from': 'rendering', 'to': 'complete', 'condition': 'workflow_complete'}
            ],
            timeout_seconds=240
        )
        
        # Error recovery test
        scenarios['error_recovery'] = WorkflowTestScenario(
            scenario_id='error_recovery',
            test_type=WorkflowTestType.ERROR_RECOVERY,
            name='Error Recovery Testing',
            description='Test error handling and recovery mechanisms',
            input_data={
                'topic': 'Invalid Topic with Special Characters @#$%',
                'description': 'This is a test description that might cause errors in processing',
                'session_id': 'error_recovery_test',
                'preview_mode': True
            },
            expected_outcomes={
                'error_handling': True,
                'recovery_attempted': True,
                'graceful_degradation': True
            },
            validation_criteria={
                'has_errors': {'type': 'exact_match'},
                'error_count': {'type': 'threshold_match', 'params': {'threshold': 10, 'comparison_type': 'absolute'}},
                'recovery_action': {'type': 'pattern_match', 'params': {'pattern': r'.+'}}
            },
            timeout_seconds=300
        )
        
        return scenarios
    
    async def run_workflow_test(
        self,
        scenario_id: str,
        custom_input: Optional[Dict[str, Any]] = None
    ) -> WorkflowTestResult:
        """Run a workflow test scenario."""
        if scenario_id not in self.test_scenarios:
            raise ValueError(f"Unknown test scenario: {scenario_id}")
        
        scenario = self.test_scenarios[scenario_id]
        
        # Create test result
        test_result = WorkflowTestResult(
            scenario_id=scenario_id,
            test_type=scenario.test_type,
            status=WorkflowTestStatus.RUNNING,
            start_time=datetime.now()
        )
        
        self.active_tests[scenario_id] = test_result
        
        try:
            logger.info(f"Starting workflow test: {scenario.name}")
            
            # Prepare input data
            input_data = scenario.input_data.copy()
            if custom_input:
                input_data.update(custom_input)
            
            # Create workflow
            workflow = await self._create_test_workflow(scenario_id)
            
            # Create initial state
            initial_state = self._create_test_state(input_data)
            
            # Add test metadata to state
            initial_state.add_execution_trace('test_started', {
                'scenario_id': scenario_id,
                'test_type': scenario.test_type.value,
                'test_name': scenario.name
            })
            
            # Execute workflow with monitoring
            final_state = await self._execute_workflow_with_monitoring(
                workflow, initial_state, scenario, test_result
            )
            
            # Update test result
            test_result.final_state = final_state
            test_result.end_time = datetime.now()
            test_result.execution_time = (test_result.end_time - test_result.start_time).total_seconds()
            test_result.status = WorkflowTestStatus.COMPLETED
            
            # Perform validation
            await self._validate_test_results(scenario, test_result)
            
            logger.info(f"Workflow test completed: {scenario.name} in {test_result.execution_time:.2f}s")
            
        except asyncio.TimeoutError:
            test_result.status = WorkflowTestStatus.TIMEOUT
            test_result.end_time = datetime.now()
            test_result.execution_time = (test_result.end_time - test_result.start_time).total_seconds()
            test_result.errors.append({
                'type': 'TimeoutError',
                'message': f'Test timed out after {scenario.timeout_seconds} seconds',
                'timestamp': datetime.now().isoformat()
            })
            logger.error(f"Workflow test timed out: {scenario.name}")
            
        except Exception as e:
            test_result.status = WorkflowTestStatus.FAILED
            test_result.end_time = datetime.now()
            test_result.execution_time = (test_result.end_time - test_result.start_time).total_seconds()
            test_result.errors.append({
                'type': type(e).__name__,
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            })
            logger.error(f"Workflow test failed: {scenario.name} - {str(e)}")
            
        finally:
            # Move to history and clean up
            if scenario_id in self.active_tests:
                del self.active_tests[scenario_id]
            self.test_history.append(test_result)
            
            # Cleanup workflow cache
            if scenario_id in self.workflow_cache:
                del self.workflow_cache[scenario_id]
        
        return test_result
    
    async def _create_test_workflow(self, scenario_id: str) -> VideoGenerationWorkflow:
        """Create a workflow instance for testing."""
        if scenario_id in self.workflow_cache:
            return self.workflow_cache[scenario_id]
        
        # Create Studio-compatible workflow
        workflow = self.studio_config.create_studio_compatible_workflow()
        
        # Cache for reuse
        self.workflow_cache[scenario_id] = workflow
        
        return workflow
    
    def _create_test_state(self, input_data: Dict[str, Any]) -> VideoGenerationState:
        """Create initial state for testing."""
        # Set default session ID if not provided
        if 'session_id' not in input_data:
            input_data['session_id'] = f"test_{int(time.time())}"
        
        # Create state with test configuration
        state = VideoGenerationState(
            topic=input_data['topic'],
            description=input_data['description'],
            session_id=input_data['session_id'],
            config=self.config
        )
        
        # Add any additional input data
        for key, value in input_data.items():
            if key not in ['topic', 'description', 'session_id'] and hasattr(state, key):
                setattr(state, key, value)
        
        return state
    
    async def _execute_workflow_with_monitoring(
        self,
        workflow: VideoGenerationWorkflow,
        initial_state: VideoGenerationState,
        scenario: WorkflowTestScenario,
        test_result: WorkflowTestResult
    ) -> VideoGenerationState:
        """Execute workflow with comprehensive monitoring."""
        session_id = initial_state.session_id
        
        # Start Studio monitoring
        studio_monitor.start_session(session_id, "workflow_test")
        
        try:
            # Set up timeout
            timeout_task = asyncio.create_task(asyncio.sleep(scenario.timeout_seconds))
            
            # Execute workflow with streaming for monitoring
            workflow_task = asyncio.create_task(self._stream_workflow_execution(
                workflow, initial_state, scenario, test_result
            ))
            
            # Wait for either completion or timeout
            done, pending = await asyncio.wait(
                [workflow_task, timeout_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Check if workflow completed
            if workflow_task in done:
                final_state = await workflow_task
                
                # Record completion metrics
                studio_monitor.record_performance_metric(
                    "workflow_test",
                    "test_completed",
                    {
                        'scenario_id': scenario.scenario_id,
                        'execution_time': test_result.execution_time,
                        'final_step': final_state.current_step,
                        'completion_percentage': final_state.get_completion_percentage() if hasattr(final_state, 'get_completion_percentage') else 0
                    }
                )
                
                return final_state
            else:
                # Timeout occurred
                raise asyncio.TimeoutError(f"Workflow test timed out after {scenario.timeout_seconds} seconds")
                
        finally:
            # End Studio monitoring
            studio_monitor.end_session(session_id, "completed" if test_result.status == WorkflowTestStatus.COMPLETED else "failed")
    
    async def _stream_workflow_execution(
        self,
        workflow: VideoGenerationWorkflow,
        initial_state: VideoGenerationState,
        scenario: WorkflowTestScenario,
        test_result: WorkflowTestResult
    ) -> VideoGenerationState:
        """Stream workflow execution with state monitoring."""
        current_state = initial_state
        
        # Configure streaming
        config = {
            "configurable": {
                "thread_id": initial_state.session_id
            }
        }
        
        # Stream workflow execution
        async for chunk in workflow.stream(initial_state, config):
            # Process chunk and update monitoring
            await self._process_workflow_chunk(chunk, scenario, test_result)
            
            # Update current state if available
            if isinstance(chunk, dict):
                for node_name, node_output in chunk.items():
                    if isinstance(node_output, VideoGenerationState):
                        current_state = node_output
                        
                        # Capture state snapshot if required
                        if node_name in scenario.state_validation_points:
                            await self._capture_state_snapshot(node_name, current_state, test_result)
                        
                        # Validate transitions if required
                        await self._validate_agent_transitions(node_name, current_state, scenario, test_result)
                        
                        # Check for checkpoint intervals
                        if node_name in scenario.checkpoint_intervals:
                            await self._capture_checkpoint_data(node_name, current_state, test_result)
        
        return current_state
    
    async def _process_workflow_chunk(
        self,
        chunk: Dict[str, Any],
        scenario: WorkflowTestScenario,
        test_result: WorkflowTestResult
    ) -> None:
        """Process a workflow execution chunk."""
        timestamp = datetime.now().isoformat()
        
        # Add to execution trace
        test_result.execution_trace.append({
            'timestamp': timestamp,
            'chunk_type': type(chunk).__name__,
            'chunk_keys': list(chunk.keys()) if isinstance(chunk, dict) else [],
            'scenario_id': scenario.scenario_id
        })
        
        # Extract performance metrics
        if isinstance(chunk, dict):
            for node_name, node_output in chunk.items():
                if isinstance(node_output, VideoGenerationState):
                    # Record node execution
                    test_result.transition_logs.append({
                        'timestamp': timestamp,
                        'node': node_name,
                        'current_step': node_output.current_step,
                        'has_errors': node_output.has_errors(),
                        'error_count': len(node_output.errors)
                    })
    
    async def _capture_state_snapshot(
        self,
        node_name: str,
        state: VideoGenerationState,
        test_result: WorkflowTestResult
    ) -> None:
        """Capture detailed state snapshot for validation."""
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'node': node_name,
            'session_id': state.session_id,
            'current_step': state.current_step,
            'workflow_complete': state.workflow_complete,
            'workflow_interrupted': getattr(state, 'workflow_interrupted', False),
            'has_errors': state.has_errors(),
            'error_count': len(state.errors),
            'scene_outline_length': len(state.scene_outline) if state.scene_outline else 0,
            'scene_implementations_count': len(state.scene_implementations) if state.scene_implementations else 0,
            'generated_code_count': len(state.generated_code) if state.generated_code else 0,
            'rendered_videos_count': len(state.rendered_videos) if state.rendered_videos else 0,
            'completion_percentage': state.get_completion_percentage() if hasattr(state, 'get_completion_percentage') else 0,
            'detected_plugins': getattr(state, 'detected_plugins', []),
            'retry_counts': getattr(state, 'retry_counts', {}),
            'escalated_errors_count': len(getattr(state, 'escalated_errors', [])),
            'pending_human_input': bool(getattr(state, 'pending_human_input', None))
        }
        
        test_result.state_snapshots.append(snapshot)
        
        logger.debug(f"Captured state snapshot for {node_name}: {snapshot['completion_percentage']:.1f}% complete")
    
    async def _validate_agent_transitions(
        self,
        current_node: str,
        state: VideoGenerationState,
        scenario: WorkflowTestScenario,
        test_result: WorkflowTestResult
    ) -> None:
        """Validate agent transitions according to scenario requirements."""
        for transition_validation in scenario.transition_validations:
            if transition_validation['to'] == current_node:
                # Check if transition condition is met
                condition = transition_validation['condition']
                is_valid = self._check_transition_condition(condition, state)
                
                transition_log = {
                    'timestamp': datetime.now().isoformat(),
                    'from_node': transition_validation['from'],
                    'to_node': current_node,
                    'condition': condition,
                    'condition_met': is_valid,
                    'session_id': state.session_id
                }
                
                test_result.transition_logs.append(transition_log)
                
                if not is_valid:
                    test_result.errors.append({
                        'type': 'TransitionValidationError',
                        'message': f"Transition condition '{condition}' not met for {transition_validation['from']} -> {current_node}",
                        'timestamp': datetime.now().isoformat(),
                        'node': current_node
                    })
    
    def _check_transition_condition(self, condition: str, state: VideoGenerationState) -> bool:
        """Check if a transition condition is met."""
        if condition == 'has_scene_outline':
            return bool(state.scene_outline and state.scene_outline.strip())
        elif condition == 'has_scene_implementations':
            return bool(state.scene_implementations and len(state.scene_implementations) > 0)
        elif condition == 'has_generated_code':
            return bool(state.generated_code and len(state.generated_code) > 0)
        elif condition == 'has_rendered_videos':
            return bool(state.rendered_videos and len(state.rendered_videos) > 0)
        elif condition == 'workflow_complete':
            return state.workflow_complete
        else:
            # Default: check if attribute exists and is truthy
            return bool(getattr(state, condition, False))
    
    async def _capture_checkpoint_data(
        self,
        node_name: str,
        state: VideoGenerationState,
        test_result: WorkflowTestResult
    ) -> None:
        """Capture checkpoint data for inspection."""
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'node': node_name,
            'session_id': state.session_id,
            'checkpoint_id': f"{state.session_id}_{node_name}_{int(time.time())}",
            'state_summary': {
                'current_step': state.current_step,
                'workflow_complete': state.workflow_complete,
                'has_errors': state.has_errors(),
                'completion_percentage': state.get_completion_percentage() if hasattr(state, 'get_completion_percentage') else 0
            },
            'execution_trace_count': len(state.execution_trace) if hasattr(state, 'execution_trace') else 0,
            'memory_usage': self._get_memory_usage()
        }
        
        test_result.checkpoint_data.append(checkpoint_data)
        
        logger.debug(f"Captured checkpoint data for {node_name}: {checkpoint_data['checkpoint_id']}")
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': process.memory_percent()
            }
        except ImportError:
            return {'error': 'psutil not available'}
        except Exception as e:
            return {'error': str(e)}
    
    async def _validate_test_results(
        self,
        scenario: WorkflowTestScenario,
        test_result: WorkflowTestResult
    ) -> None:
        """Validate test results against expected outcomes."""
        if not test_result.final_state:
            return
        
        # Extract actual outputs from final state
        actual_outputs = {
            'workflow_complete': test_result.final_state.workflow_complete,
            'current_step': test_result.final_state.current_step,
            'has_errors': test_result.final_state.has_errors(),
            'scene_outline': test_result.final_state.scene_outline,
            'scene_implementations': test_result.final_state.scene_implementations,
            'generated_code': test_result.final_state.generated_code,
            'rendered_videos': test_result.final_state.rendered_videos,
            'completion_percentage': test_result.final_state.get_completion_percentage() if hasattr(test_result.final_state, 'get_completion_percentage') else 0,
            'scene_count': len(test_result.final_state.scene_implementations) if test_result.final_state.scene_implementations else 0
        }
        
        # Validate against expected outcomes
        validation_result = self.result_validator.validate_result(
            agent_type="WorkflowTest",
            test_id=scenario.scenario_id,
            expected_outputs=scenario.expected_outcomes,
            actual_outputs=actual_outputs,
            validation_criteria=scenario.validation_criteria
        )
        
        test_result.validation_results.append(validation_result)
        
        # Add performance metrics
        test_result.performance_metrics = {
            'execution_time': test_result.execution_time,
            'state_snapshots_count': len(test_result.state_snapshots),
            'checkpoint_count': len(test_result.checkpoint_data),
            'transition_count': len(test_result.transition_logs),
            'error_count': len(test_result.errors),
            'validation_score': validation_result.overall_score,
            'validation_status': validation_result.overall_status.value
        }
        
        logger.info(f"Test validation completed: {validation_result.overall_status.value} (score: {validation_result.overall_score:.2f})")
    
    def get_test_scenario(self, scenario_id: str) -> Optional[WorkflowTestScenario]:
        """Get a test scenario by ID."""
        return self.test_scenarios.get(scenario_id)
    
    def list_test_scenarios(self) -> List[Dict[str, Any]]:
        """List all available test scenarios."""
        return [scenario.to_dict() for scenario in self.test_scenarios.values()]
    
    def get_test_result(self, scenario_id: str) -> Optional[WorkflowTestResult]:
        """Get test result by scenario ID."""
        # Check active tests first
        if scenario_id in self.active_tests:
            return self.active_tests[scenario_id]
        
        # Check test history
        for result in reversed(self.test_history):
            if result.scenario_id == scenario_id:
                return result
        
        return None
    
    def get_test_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get test execution history."""
        return [result.to_dict() for result in self.test_history[-limit:]]
    
    def get_active_tests(self) -> Dict[str, Dict[str, Any]]:
        """Get currently active tests."""
        return {
            scenario_id: result.to_dict() 
            for scenario_id, result in self.active_tests.items()
        }
    
    async def cancel_test(self, scenario_id: str) -> bool:
        """Cancel an active test."""
        if scenario_id in self.active_tests:
            test_result = self.active_tests[scenario_id]
            test_result.status = WorkflowTestStatus.CANCELLED
            test_result.end_time = datetime.now()
            test_result.execution_time = (test_result.end_time - test_result.start_time).total_seconds()
            
            # Move to history
            del self.active_tests[scenario_id]
            self.test_history.append(test_result)
            
            logger.info(f"Test cancelled: {scenario_id}")
            return True
        
        return False
    
    def create_custom_scenario(
        self,
        scenario_id: str,
        test_type: WorkflowTestType,
        name: str,
        description: str,
        input_data: Dict[str, Any],
        expected_outcomes: Dict[str, Any],
        validation_criteria: Dict[str, Any],
        **kwargs
    ) -> WorkflowTestScenario:
        """Create a custom test scenario."""
        scenario = WorkflowTestScenario(
            scenario_id=scenario_id,
            test_type=test_type,
            name=name,
            description=description,
            input_data=input_data,
            expected_outcomes=expected_outcomes,
            validation_criteria=validation_criteria,
            **kwargs
        )
        
        self.test_scenarios[scenario_id] = scenario
        
        logger.info(f"Created custom test scenario: {scenario_id}")
        return scenario


# Global instance for Studio integration
studio_workflow_tester = StudioWorkflowTester()


def get_studio_workflow_tester() -> StudioWorkflowTester:
    """Get the global Studio workflow tester instance."""
    return studio_workflow_tester


# Convenience functions for Studio integration
async def run_end_to_end_test(custom_input: Optional[Dict[str, Any]] = None) -> WorkflowTestResult:
    """Run end-to-end workflow test."""
    return await studio_workflow_tester.run_workflow_test('e2e_basic', custom_input)


async def run_state_validation_test(custom_input: Optional[Dict[str, Any]] = None) -> WorkflowTestResult:
    """Run state validation test."""
    return await studio_workflow_tester.run_workflow_test('state_validation', custom_input)


async def run_agent_transition_test(custom_input: Optional[Dict[str, Any]] = None) -> WorkflowTestResult:
    """Run agent transition test."""
    return await studio_workflow_tester.run_workflow_test('agent_transitions', custom_input)


async def run_error_recovery_test(custom_input: Optional[Dict[str, Any]] = None) -> WorkflowTestResult:
    """Run error recovery test."""
    return await studio_workflow_tester.run_workflow_test('error_recovery', custom_input)


def get_workflow_test_info() -> Dict[str, Any]:
    """Get information about workflow testing capabilities."""
    return {
        'available_scenarios': studio_workflow_tester.list_test_scenarios(),
        'active_tests': studio_workflow_tester.get_active_tests(),
        'recent_history': studio_workflow_tester.get_test_history(10),
        'test_types': [test_type.value for test_type in WorkflowTestType],
        'capabilities': [
            'end_to_end_workflow_testing',
            'state_validation_and_inspection',
            'checkpoint_data_capture',
            'agent_transition_validation',
            'error_recovery_testing',
            'performance_monitoring',
            'execution_logging',
            'result_validation'
        ]
    }
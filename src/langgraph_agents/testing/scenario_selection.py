"""
Enhanced scenario selection and execution system for LangGraph Studio.

This module provides advanced scenario selection, filtering, and execution
capabilities specifically designed for Studio integration.
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

from .test_data_manager import TestDataManager, TestScenarioManager, TestComplexity, TestScenario
from .input_validation import get_validation_manager, ValidationResult
from .result_validation import get_result_validator, ValidationResult as ResultValidationResult

logger = logging.getLogger(__name__)


class ScenarioFilter(Enum):
    """Types of scenario filters."""
    AGENT_TYPE = "agent_type"
    COMPLEXITY = "complexity"
    TAGS = "tags"
    SUCCESS_RATE = "success_rate"
    EXECUTION_TIME = "execution_time"
    LAST_EXECUTED = "last_executed"
    CUSTOM = "custom"


class SortOrder(Enum):
    """Sort order options."""
    ASC = "asc"
    DESC = "desc"


@dataclass
class ScenarioFilterCriteria:
    """Criteria for filtering scenarios."""
    
    filter_type: ScenarioFilter
    value: Any
    operator: str = "eq"  # eq, ne, gt, lt, gte, lte, in, not_in, contains
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'filter_type': self.filter_type.value,
            'value': self.value,
            'operator': self.operator
        }


@dataclass
class ScenarioSortCriteria:
    """Criteria for sorting scenarios."""
    
    field: str
    order: SortOrder = SortOrder.ASC
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'field': self.field,
            'order': self.order.value
        }


@dataclass
class ScenarioExecutionConfig:
    """Configuration for scenario execution."""
    
    timeout: int = 300  # 5 minutes default
    retry_count: int = 1
    parallel_execution: bool = False
    capture_output: bool = True
    validate_inputs: bool = True
    validate_results: bool = True
    save_results: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ScenarioExecutionResult:
    """Result of scenario execution."""
    
    scenario_id: str
    agent_type: str
    start_time: float
    end_time: float
    execution_time: float
    success: bool
    test_result: Dict[str, Any]
    input_validation: Optional[ValidationResult] = None
    result_validation: Optional[ResultValidationResult] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'scenario_id': self.scenario_id,
            'agent_type': self.agent_type,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'execution_time': self.execution_time,
            'success': self.success,
            'test_result': self.test_result,
            'input_validation': self.input_validation.to_dict() if self.input_validation else None,
            'result_validation': self.result_validation.to_dict() if self.result_validation else None,
            'error': self.error
        }


class ScenarioSelector:
    """Advanced scenario selection system."""
    
    def __init__(self, scenario_manager: TestScenarioManager = None):
        """Initialize scenario selector."""
        self.scenario_manager = scenario_manager or TestScenarioManager()
        self.logger = logging.getLogger(__name__)
    
    def filter_scenarios(self, 
                        filters: List[ScenarioFilterCriteria],
                        sort_criteria: Optional[List[ScenarioSortCriteria]] = None) -> List[TestScenario]:
        """Filter scenarios based on criteria."""
        scenarios = list(self.scenario_manager.scenarios.values())
        
        # Apply filters
        for filter_criteria in filters:
            scenarios = self._apply_filter(scenarios, filter_criteria)
        
        # Apply sorting
        if sort_criteria:
            scenarios = self._apply_sorting(scenarios, sort_criteria)
        
        return scenarios
    
    def select_scenarios_by_agent(self, 
                                agent_type: str,
                                complexity: Optional[TestComplexity] = None,
                                limit: Optional[int] = None) -> List[TestScenario]:
        """Select scenarios for specific agent type."""
        filters = [ScenarioFilterCriteria(ScenarioFilter.AGENT_TYPE, agent_type.lower())]
        
        if complexity:
            filters.append(ScenarioFilterCriteria(ScenarioFilter.COMPLEXITY, complexity.value))
        
        scenarios = self.filter_scenarios(filters)
        
        if limit:
            scenarios = scenarios[:limit]
        
        return scenarios
    
    def select_scenarios_by_success_rate(self, 
                                       min_success_rate: float = 0.0,
                                       max_success_rate: float = 1.0) -> List[TestScenario]:
        """Select scenarios by historical success rate."""
        # Get execution history
        execution_history = self.scenario_manager.execution_history
        
        # Calculate success rates
        scenario_success_rates = {}
        for scenario_id, executions in execution_history.items():
            if isinstance(executions, list):
                successful = sum(1 for e in executions if e.get('status') == 'completed' and e.get('result', {}).get('success', False))
                total = len(executions)
                success_rate = successful / total if total > 0 else 0.0
            else:
                # Single execution record
                success_rate = 1.0 if executions.get('status') == 'completed' and executions.get('result', {}).get('success', False) else 0.0
            
            scenario_success_rates[scenario_id] = success_rate
        
        # Filter scenarios
        filtered_scenarios = []
        for scenario_id, scenario in self.scenario_manager.scenarios.items():
            success_rate = scenario_success_rates.get(scenario_id, 0.0)
            if min_success_rate <= success_rate <= max_success_rate:
                filtered_scenarios.append(scenario)
        
        return filtered_scenarios
    
    def select_scenarios_for_regression_testing(self, 
                                              agent_types: Optional[List[str]] = None) -> List[TestScenario]:
        """Select scenarios suitable for regression testing."""
        # Get scenarios that have been executed successfully before
        successful_scenarios = self.select_scenarios_by_success_rate(min_success_rate=0.8)
        
        if agent_types:
            # Filter by agent types
            filtered_scenarios = []
            for scenario in successful_scenarios:
                for tag in scenario.tags:
                    if any(agent_type.lower() in tag for agent_type in agent_types):
                        filtered_scenarios.append(scenario)
                        break
            successful_scenarios = filtered_scenarios
        
        # Sort by execution time (prefer faster scenarios for regression)
        sort_criteria = [ScenarioSortCriteria('execution_time', SortOrder.ASC)]
        return self._apply_sorting(successful_scenarios, sort_criteria)
    
    def select_scenarios_for_stress_testing(self, 
                                          complexity: TestComplexity = TestComplexity.COMPLEX,
                                          count: int = 10) -> List[TestScenario]:
        """Select scenarios suitable for stress testing."""
        filters = [
            ScenarioFilterCriteria(ScenarioFilter.COMPLEXITY, complexity.value)
        ]
        
        scenarios = self.filter_scenarios(filters)
        
        # Sort by complexity and execution time (prefer more complex, longer scenarios)
        sort_criteria = [
            ScenarioSortCriteria('execution_time', SortOrder.DESC)
        ]
        scenarios = self._apply_sorting(scenarios, sort_criteria)
        
        return scenarios[:count]
    
    def recommend_scenarios(self, 
                          agent_type: str,
                          user_preferences: Optional[Dict[str, Any]] = None) -> List[TestScenario]:
        """Recommend scenarios based on agent type and user preferences."""
        # Start with scenarios for the agent type
        base_scenarios = self.select_scenarios_by_agent(agent_type)
        
        if not user_preferences:
            return base_scenarios[:5]  # Return top 5 by default
        
        # Apply user preferences
        preferred_complexity = user_preferences.get('complexity')
        if preferred_complexity:
            complexity_filter = ScenarioFilterCriteria(
                ScenarioFilter.COMPLEXITY, 
                preferred_complexity
            )
            base_scenarios = self._apply_filter(base_scenarios, complexity_filter)
        
        # Prefer scenarios with good success rates
        min_success_rate = user_preferences.get('min_success_rate', 0.5)
        if min_success_rate > 0:
            base_scenarios = [s for s in base_scenarios if self._get_scenario_success_rate(s.id) >= min_success_rate]
        
        # Sort by preference
        sort_preference = user_preferences.get('sort_by', 'success_rate')
        if sort_preference == 'success_rate':
            base_scenarios.sort(key=lambda s: self._get_scenario_success_rate(s.id), reverse=True)
        elif sort_preference == 'execution_time':
            base_scenarios.sort(key=lambda s: self._get_scenario_avg_execution_time(s.id))
        elif sort_preference == 'last_executed':
            base_scenarios.sort(key=lambda s: self._get_scenario_last_execution_time(s.id), reverse=True)
        
        limit = user_preferences.get('limit', 5)
        return base_scenarios[:limit]
    
    def _apply_filter(self, scenarios: List[TestScenario], filter_criteria: ScenarioFilterCriteria) -> List[TestScenario]:
        """Apply a single filter to scenarios."""
        filtered = []
        
        for scenario in scenarios:
            if self._scenario_matches_filter(scenario, filter_criteria):
                filtered.append(scenario)
        
        return filtered
    
    def _scenario_matches_filter(self, scenario: TestScenario, filter_criteria: ScenarioFilterCriteria) -> bool:
        """Check if scenario matches filter criteria."""
        filter_type = filter_criteria.filter_type
        value = filter_criteria.value
        operator = filter_criteria.operator
        
        if filter_type == ScenarioFilter.AGENT_TYPE:
            scenario_value = scenario.tags
            if operator == "in":
                return any(value.lower() in tag.lower() for tag in scenario_value)
            elif operator == "eq":
                return any(value.lower() == tag.lower() for tag in scenario_value)
        
        elif filter_type == ScenarioFilter.COMPLEXITY:
            scenario_value = scenario.complexity.value
            return self._compare_values(scenario_value, value, operator)
        
        elif filter_type == ScenarioFilter.TAGS:
            scenario_value = scenario.tags
            if operator == "contains":
                return value in scenario_value
            elif operator == "in":
                return any(tag in value for tag in scenario_value)
        
        elif filter_type == ScenarioFilter.SUCCESS_RATE:
            scenario_value = self._get_scenario_success_rate(scenario.id)
            return self._compare_values(scenario_value, value, operator)
        
        elif filter_type == ScenarioFilter.EXECUTION_TIME:
            scenario_value = self._get_scenario_avg_execution_time(scenario.id)
            return self._compare_values(scenario_value, value, operator)
        
        elif filter_type == ScenarioFilter.LAST_EXECUTED:
            scenario_value = self._get_scenario_last_execution_time(scenario.id)
            return self._compare_values(scenario_value, value, operator)
        
        return False
    
    def _compare_values(self, scenario_value: Any, filter_value: Any, operator: str) -> bool:
        """Compare values based on operator."""
        if operator == "eq":
            return scenario_value == filter_value
        elif operator == "ne":
            return scenario_value != filter_value
        elif operator == "gt":
            return scenario_value > filter_value
        elif operator == "lt":
            return scenario_value < filter_value
        elif operator == "gte":
            return scenario_value >= filter_value
        elif operator == "lte":
            return scenario_value <= filter_value
        elif operator == "in":
            return scenario_value in filter_value
        elif operator == "not_in":
            return scenario_value not in filter_value
        elif operator == "contains":
            return filter_value in str(scenario_value)
        
        return False
    
    def _apply_sorting(self, scenarios: List[TestScenario], sort_criteria: List[ScenarioSortCriteria]) -> List[TestScenario]:
        """Apply sorting to scenarios."""
        for sort_criterion in reversed(sort_criteria):  # Apply in reverse order for stable sort
            field = sort_criterion.field
            reverse = sort_criterion.order == SortOrder.DESC
            
            if field == "name":
                scenarios.sort(key=lambda s: s.name, reverse=reverse)
            elif field == "complexity":
                scenarios.sort(key=lambda s: s.complexity.value, reverse=reverse)
            elif field == "success_rate":
                scenarios.sort(key=lambda s: self._get_scenario_success_rate(s.id), reverse=reverse)
            elif field == "execution_time":
                scenarios.sort(key=lambda s: self._get_scenario_avg_execution_time(s.id), reverse=reverse)
            elif field == "last_executed":
                scenarios.sort(key=lambda s: self._get_scenario_last_execution_time(s.id), reverse=reverse)
        
        return scenarios
    
    def _get_scenario_success_rate(self, scenario_id: str) -> float:
        """Get success rate for scenario."""
        execution_history = self.scenario_manager.execution_history.get(scenario_id, {})
        
        if isinstance(execution_history, list):
            if not execution_history:
                return 0.0
            successful = sum(1 for e in execution_history if e.get('status') == 'completed' and e.get('result', {}).get('success', False))
            return successful / len(execution_history)
        else:
            # Single execution record
            if not execution_history:
                return 0.0
            return 1.0 if execution_history.get('status') == 'completed' and execution_history.get('result', {}).get('success', False) else 0.0
    
    def _get_scenario_avg_execution_time(self, scenario_id: str) -> float:
        """Get average execution time for scenario."""
        execution_history = self.scenario_manager.execution_history.get(scenario_id, {})
        
        if isinstance(execution_history, list):
            if not execution_history:
                return 0.0
            times = [e.get('result', {}).get('execution_time', 0) for e in execution_history if e.get('result')]
            return sum(times) / len(times) if times else 0.0
        else:
            # Single execution record
            return execution_history.get('result', {}).get('execution_time', 0.0)
    
    def _get_scenario_last_execution_time(self, scenario_id: str) -> float:
        """Get last execution time for scenario."""
        execution_history = self.scenario_manager.execution_history.get(scenario_id, {})
        
        if isinstance(execution_history, list):
            if not execution_history:
                return 0.0
            return max(e.get('end_time', 0) for e in execution_history)
        else:
            # Single execution record
            return execution_history.get('end_time', 0.0)


class ScenarioExecutor:
    """Enhanced scenario execution system."""
    
    def __init__(self, 
                 scenario_manager: TestScenarioManager = None,
                 validation_manager = None,
                 result_validator = None):
        """Initialize scenario executor."""
        self.scenario_manager = scenario_manager or TestScenarioManager()
        self.validation_manager = validation_manager or get_validation_manager()
        self.result_validator = result_validator or get_result_validator()
        self.logger = logging.getLogger(__name__)
    
    async def execute_scenario(self, 
                             scenario_id: str,
                             test_runner,
                             config: ScenarioExecutionConfig = None) -> ScenarioExecutionResult:
        """Execute a single scenario."""
        config = config or ScenarioExecutionConfig()
        start_time = time.time()
        
        # Get scenario
        scenario = self.scenario_manager.get_scenario(scenario_id)
        if not scenario:
            return ScenarioExecutionResult(
                scenario_id=scenario_id,
                agent_type="Unknown",
                start_time=start_time,
                end_time=time.time(),
                execution_time=time.time() - start_time,
                success=False,
                test_result={},
                error=f"Scenario not found: {scenario_id}"
            )
        
        # Determine agent type
        agent_type = self._determine_agent_type(scenario)
        
        try:
            # Input validation
            input_validation = None
            if config.validate_inputs:
                input_validation = self.validation_manager.validate_inputs(agent_type, scenario.inputs)
                if not input_validation.is_valid:
                    return ScenarioExecutionResult(
                        scenario_id=scenario_id,
                        agent_type=agent_type,
                        start_time=start_time,
                        end_time=time.time(),
                        execution_time=time.time() - start_time,
                        success=False,
                        test_result={},
                        input_validation=input_validation,
                        error="Input validation failed"
                    )
                
                # Use processed inputs
                inputs = input_validation.processed_inputs
            else:
                inputs = scenario.inputs
            
            # Execute test
            self.logger.info(f"Executing scenario {scenario_id} for {agent_type}")
            
            if hasattr(test_runner, 'run_test'):
                test_result = await test_runner.run_test(inputs)
            else:
                # Fallback for synchronous test runners
                test_result = test_runner.run_test(inputs)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Result validation
            result_validation = None
            if config.validate_results:
                result_validation = self.result_validator.validate_agent_specific_results(
                    agent_type, scenario_id, test_result
                )
            
            # Create execution result
            execution_result = ScenarioExecutionResult(
                scenario_id=scenario_id,
                agent_type=agent_type,
                start_time=start_time,
                end_time=end_time,
                execution_time=execution_time,
                success=test_result.get('success', False),
                test_result=test_result,
                input_validation=input_validation,
                result_validation=result_validation
            )
            
            # Save results if configured
            if config.save_results:
                self._save_execution_result(execution_result)
            
            self.logger.info(f"Scenario {scenario_id} executed successfully in {execution_time:.2f}s")
            return execution_result
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            self.logger.error(f"Scenario {scenario_id} execution failed: {str(e)}")
            
            execution_result = ScenarioExecutionResult(
                scenario_id=scenario_id,
                agent_type=agent_type,
                start_time=start_time,
                end_time=end_time,
                execution_time=execution_time,
                success=False,
                test_result={},
                error=str(e)
            )
            
            # Save results even on failure if configured
            if config.save_results:
                self._save_execution_result(execution_result)
            
            return execution_result
    
    async def execute_scenarios_batch(self, 
                                    scenario_ids: List[str],
                                    test_runner,
                                    config: ScenarioExecutionConfig = None) -> List[ScenarioExecutionResult]:
        """Execute multiple scenarios."""
        config = config or ScenarioExecutionConfig()
        results = []
        
        if config.parallel_execution:
            # Parallel execution (if supported)
            import asyncio
            tasks = [
                self.execute_scenario(scenario_id, test_runner, config)
                for scenario_id in scenario_ids
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    results[i] = ScenarioExecutionResult(
                        scenario_id=scenario_ids[i],
                        agent_type="Unknown",
                        start_time=time.time(),
                        end_time=time.time(),
                        execution_time=0.0,
                        success=False,
                        test_result={},
                        error=str(result)
                    )
        else:
            # Sequential execution
            for scenario_id in scenario_ids:
                result = await self.execute_scenario(scenario_id, test_runner, config)
                results.append(result)
        
        return results
    
    def _determine_agent_type(self, scenario: TestScenario) -> str:
        """Determine agent type from scenario."""
        # Check tags first
        for tag in scenario.tags:
            if 'planner' in tag.lower():
                return 'PlannerAgent'
            elif 'codegen' in tag.lower() or 'code' in tag.lower():
                return 'CodeGeneratorAgent'
            elif 'render' in tag.lower():
                return 'RendererAgent'
            elif 'error' in tag.lower():
                return 'ErrorHandlerAgent'
            elif 'human' in tag.lower() or 'loop' in tag.lower():
                return 'HumanLoopAgent'
        
        # Check inputs to infer agent type
        inputs = scenario.inputs
        if 'error_scenarios' in inputs:
            return 'ErrorHandlerAgent'
        elif 'intervention_scenarios' in inputs:
            return 'HumanLoopAgent'
        elif 'generated_code' in inputs:
            return 'RendererAgent'
        elif 'scene_implementations' in inputs:
            return 'CodeGeneratorAgent'
        else:
            return 'PlannerAgent'  # Default
    
    def _save_execution_result(self, result: ScenarioExecutionResult) -> None:
        """Save execution result to scenario manager."""
        scenario_id = result.scenario_id
        
        # Update execution history
        if scenario_id not in self.scenario_manager.execution_history:
            self.scenario_manager.execution_history[scenario_id] = []
        
        history = self.scenario_manager.execution_history[scenario_id]
        if not isinstance(history, list):
            # Convert single record to list
            self.scenario_manager.execution_history[scenario_id] = [history]
            history = self.scenario_manager.execution_history[scenario_id]
        
        # Add new execution record
        execution_record = {
            'start_time': result.start_time,
            'end_time': result.end_time,
            'status': 'completed' if result.success else 'failed',
            'result': result.test_result,
            'agent_type': result.agent_type,
            'execution_time': result.execution_time,
            'error': result.error
        }
        
        history.append(execution_record)
        
        # Keep only last 10 executions per scenario
        if len(history) > 10:
            self.scenario_manager.execution_history[scenario_id] = history[-10:]


class StudioScenarioInterface:
    """Studio-specific scenario interface."""
    
    def __init__(self):
        """Initialize Studio scenario interface."""
        self.scenario_manager = TestScenarioManager()
        self.selector = ScenarioSelector(self.scenario_manager)
        self.executor = ScenarioExecutor(self.scenario_manager)
        self.logger = logging.getLogger(__name__)
    
    def get_available_scenarios(self, agent_type: Optional[str] = None) -> Dict[str, Any]:
        """Get available scenarios for Studio interface."""
        if agent_type:
            scenarios = self.selector.select_scenarios_by_agent(agent_type)
        else:
            scenarios = list(self.scenario_manager.scenarios.values())
        
        return {
            'scenarios': [
                {
                    'id': s.id,
                    'name': s.name,
                    'description': s.description,
                    'complexity': s.complexity.value,
                    'tags': s.tags,
                    'agent_type': self._determine_agent_type_from_scenario(s),
                    'success_rate': self.selector._get_scenario_success_rate(s.id),
                    'avg_execution_time': self.selector._get_scenario_avg_execution_time(s.id),
                    'last_executed': self.selector._get_scenario_last_execution_time(s.id)
                }
                for s in scenarios
            ],
            'total_count': len(scenarios),
            'agent_types': list(set(self._determine_agent_type_from_scenario(s) for s in scenarios))
        }
    
    def get_scenario_recommendations(self, 
                                   agent_type: str,
                                   user_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get scenario recommendations for Studio."""
        recommended = self.selector.recommend_scenarios(agent_type, user_preferences)
        
        return {
            'recommended_scenarios': [
                {
                    'id': s.id,
                    'name': s.name,
                    'description': s.description,
                    'complexity': s.complexity.value,
                    'success_rate': self.selector._get_scenario_success_rate(s.id),
                    'avg_execution_time': self.selector._get_scenario_avg_execution_time(s.id),
                    'recommendation_score': self._calculate_recommendation_score(s)
                }
                for s in recommended
            ],
            'agent_type': agent_type,
            'preferences_applied': user_preferences or {}
        }
    
    def create_custom_scenario(self, 
                             agent_type: str,
                             scenario_data: Dict[str, Any],
                             metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create custom scenario for Studio."""
        try:
            # Validate inputs
            validation_result = get_validation_manager().validate_inputs(agent_type, scenario_data)
            if not validation_result.is_valid:
                return {
                    'success': False,
                    'error': 'Input validation failed',
                    'validation_issues': [issue.to_dict() for issue in validation_result.issues]
                }
            
            # Create scenario
            scenario_id = self.scenario_manager.create_scenario(agent_type, validation_result.processed_inputs)
            
            # Update metadata if provided
            if metadata:
                scenario = self.scenario_manager.get_scenario(scenario_id)
                if scenario:
                    scenario.name = metadata.get('name', scenario.name)
                    scenario.description = metadata.get('description', scenario.description)
                    if 'complexity' in metadata:
                        scenario.complexity = TestComplexity(metadata['complexity'])
                    if 'tags' in metadata:
                        scenario.tags = metadata['tags']
            
            return {
                'success': True,
                'scenario_id': scenario_id,
                'scenario': self.scenario_manager.get_scenario(scenario_id).to_dict()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _determine_agent_type_from_scenario(self, scenario: TestScenario) -> str:
        """Determine agent type from scenario."""
        return self.executor._determine_agent_type(scenario)
    
    def _calculate_recommendation_score(self, scenario: TestScenario) -> float:
        """Calculate recommendation score for scenario."""
        success_rate = self.selector._get_scenario_success_rate(scenario.id)
        execution_time = self.selector._get_scenario_avg_execution_time(scenario.id)
        
        # Normalize execution time (prefer faster scenarios)
        time_score = max(0.0, 1.0 - (execution_time / 300.0))  # 5 minutes max
        
        # Combine scores (70% success rate, 30% execution time)
        return (success_rate * 0.7) + (time_score * 0.3)


# Global instances for easy access
_global_scenario_selector = ScenarioSelector()
_global_scenario_executor = ScenarioExecutor()
_global_studio_interface = StudioScenarioInterface()


def get_scenario_selector() -> ScenarioSelector:
    """Get the global scenario selector instance."""
    return _global_scenario_selector


def get_scenario_executor() -> ScenarioExecutor:
    """Get the global scenario executor instance."""
    return _global_scenario_executor


def get_studio_interface() -> StudioScenarioInterface:
    """Get the global Studio interface instance."""
    return _global_studio_interface
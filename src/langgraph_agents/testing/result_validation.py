"""
Test result comparison and validation system.

This module provides comprehensive result validation, comparison, and
analysis capabilities for agent testing.
"""

import json
import logging
import difflib
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ComparisonType(Enum):
    """Types of result comparisons."""
    EXACT_MATCH = "exact_match"
    FUZZY_MATCH = "fuzzy_match"
    STRUCTURAL_MATCH = "structural_match"
    SEMANTIC_MATCH = "semantic_match"
    THRESHOLD_MATCH = "threshold_match"
    PATTERN_MATCH = "pattern_match"


class ValidationStatus(Enum):
    """Status of validation results."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class ComparisonResult:
    """Result of a comparison operation."""
    
    field: str
    comparison_type: ComparisonType
    status: ValidationStatus
    expected: Any
    actual: Any
    score: float  # 0.0 to 1.0
    message: str
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'field': self.field,
            'comparison_type': self.comparison_type.value,
            'status': self.status.value,
            'expected': self.expected,
            'actual': self.actual,
            'score': self.score,
            'message': self.message,
            'details': self.details or {}
        }


@dataclass
class ValidationResult:
    """Result of result validation."""
    
    agent_type: str
    test_id: str
    overall_status: ValidationStatus
    overall_score: float
    comparisons: List[ComparisonResult]
    summary: Dict[str, Any]
    
    def get_passed_comparisons(self) -> List[ComparisonResult]:
        """Get comparisons that passed."""
        return [c for c in self.comparisons if c.status == ValidationStatus.PASSED]
    
    def get_failed_comparisons(self) -> List[ComparisonResult]:
        """Get comparisons that failed."""
        return [c for c in self.comparisons if c.status == ValidationStatus.FAILED]
    
    def get_warning_comparisons(self) -> List[ComparisonResult]:
        """Get comparisons with warnings."""
        return [c for c in self.comparisons if c.status == ValidationStatus.WARNING]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'agent_type': self.agent_type,
            'test_id': self.test_id,
            'overall_status': self.overall_status.value,
            'overall_score': self.overall_score,
            'comparisons': [c.to_dict() for c in self.comparisons],
            'summary': self.summary,
            'passed_count': len(self.get_passed_comparisons()),
            'failed_count': len(self.get_failed_comparisons()),
            'warning_count': len(self.get_warning_comparisons())
        }


class BaseResultComparator(ABC):
    """Base class for result comparators."""
    
    @abstractmethod
    def compare(self, expected: Any, actual: Any, field: str, **kwargs) -> ComparisonResult:
        """Compare expected and actual values."""
        pass


class ExactMatchComparator(BaseResultComparator):
    """Comparator for exact matches."""
    
    def compare(self, expected: Any, actual: Any, field: str, **kwargs) -> ComparisonResult:
        """Compare for exact match."""
        matches = expected == actual
        score = 1.0 if matches else 0.0
        status = ValidationStatus.PASSED if matches else ValidationStatus.FAILED
        
        message = "Exact match" if matches else f"Values don't match: expected {expected}, got {actual}"
        
        return ComparisonResult(
            field=field,
            comparison_type=ComparisonType.EXACT_MATCH,
            status=status,
            expected=expected,
            actual=actual,
            score=score,
            message=message
        )


class FuzzyMatchComparator(BaseResultComparator):
    """Comparator for fuzzy string matches."""
    
    def compare(self, expected: Any, actual: Any, field: str, **kwargs) -> ComparisonResult:
        """Compare for fuzzy match."""
        threshold = kwargs.get('threshold', 0.8)
        
        if not isinstance(expected, str) or not isinstance(actual, str):
            return ComparisonResult(
                field=field,
                comparison_type=ComparisonType.FUZZY_MATCH,
                status=ValidationStatus.FAILED,
                expected=expected,
                actual=actual,
                score=0.0,
                message="Fuzzy match requires string values"
            )
        
        # Calculate similarity using difflib
        similarity = difflib.SequenceMatcher(None, expected, actual).ratio()
        matches = similarity >= threshold
        status = ValidationStatus.PASSED if matches else ValidationStatus.FAILED
        
        message = f"Fuzzy match (similarity: {similarity:.2f})" if matches else f"Similarity too low: {similarity:.2f} < {threshold}"
        
        return ComparisonResult(
            field=field,
            comparison_type=ComparisonType.FUZZY_MATCH,
            status=status,
            expected=expected,
            actual=actual,
            score=similarity,
            message=message,
            details={'similarity': similarity, 'threshold': threshold}
        )


class StructuralMatchComparator(BaseResultComparator):
    """Comparator for structural matches (dict/list structure)."""
    
    def compare(self, expected: Any, actual: Any, field: str, **kwargs) -> ComparisonResult:
        """Compare for structural match."""
        ignore_values = kwargs.get('ignore_values', False)
        
        try:
            if type(expected) != type(actual):
                return ComparisonResult(
                    field=field,
                    comparison_type=ComparisonType.STRUCTURAL_MATCH,
                    status=ValidationStatus.FAILED,
                    expected=expected,
                    actual=actual,
                    score=0.0,
                    message=f"Type mismatch: expected {type(expected).__name__}, got {type(actual).__name__}"
                )
            
            if isinstance(expected, dict):
                score = self._compare_dict_structure(expected, actual, ignore_values)
            elif isinstance(expected, list):
                score = self._compare_list_structure(expected, actual, ignore_values)
            else:
                # For non-structural types, fall back to exact match
                score = 1.0 if expected == actual else 0.0
            
            status = ValidationStatus.PASSED if score >= 0.8 else ValidationStatus.FAILED
            message = f"Structural match (score: {score:.2f})"
            
            return ComparisonResult(
                field=field,
                comparison_type=ComparisonType.STRUCTURAL_MATCH,
                status=status,
                expected=expected,
                actual=actual,
                score=score,
                message=message,
                details={'ignore_values': ignore_values}
            )
            
        except Exception as e:
            return ComparisonResult(
                field=field,
                comparison_type=ComparisonType.STRUCTURAL_MATCH,
                status=ValidationStatus.FAILED,
                expected=expected,
                actual=actual,
                score=0.0,
                message=f"Comparison error: {str(e)}"
            )
    
    def _compare_dict_structure(self, expected: dict, actual: dict, ignore_values: bool) -> float:
        """Compare dictionary structures."""
        if not ignore_values and expected == actual:
            return 1.0
        
        expected_keys = set(expected.keys())
        actual_keys = set(actual.keys())
        
        # Calculate key overlap
        common_keys = expected_keys & actual_keys
        all_keys = expected_keys | actual_keys
        
        if not all_keys:
            return 1.0  # Both empty
        
        key_score = len(common_keys) / len(all_keys)
        
        if ignore_values:
            return key_score
        
        # Calculate value similarity for common keys
        value_scores = []
        for key in common_keys:
            if expected[key] == actual[key]:
                value_scores.append(1.0)
            elif isinstance(expected[key], (dict, list)):
                # Recursive comparison for nested structures
                if isinstance(expected[key], dict) and isinstance(actual[key], dict):
                    value_scores.append(self._compare_dict_structure(expected[key], actual[key], ignore_values))
                elif isinstance(expected[key], list) and isinstance(actual[key], list):
                    value_scores.append(self._compare_list_structure(expected[key], actual[key], ignore_values))
                else:
                    value_scores.append(0.0)
            else:
                value_scores.append(0.0)
        
        value_score = sum(value_scores) / len(value_scores) if value_scores else 0.0
        
        # Combine key and value scores
        return (key_score + value_score) / 2
    
    def _compare_list_structure(self, expected: list, actual: list, ignore_values: bool) -> float:
        """Compare list structures."""
        if not ignore_values and expected == actual:
            return 1.0
        
        if len(expected) == 0 and len(actual) == 0:
            return 1.0
        
        if len(expected) == 0 or len(actual) == 0:
            return 0.0
        
        # Calculate length similarity
        length_score = min(len(expected), len(actual)) / max(len(expected), len(actual))
        
        if ignore_values:
            return length_score
        
        # Calculate element similarity
        element_scores = []
        min_length = min(len(expected), len(actual))
        
        for i in range(min_length):
            if expected[i] == actual[i]:
                element_scores.append(1.0)
            elif isinstance(expected[i], (dict, list)):
                # Recursive comparison for nested structures
                if isinstance(expected[i], dict) and isinstance(actual[i], dict):
                    element_scores.append(self._compare_dict_structure(expected[i], actual[i], ignore_values))
                elif isinstance(expected[i], list) and isinstance(actual[i], list):
                    element_scores.append(self._compare_list_structure(expected[i], actual[i], ignore_values))
                else:
                    element_scores.append(0.0)
            else:
                element_scores.append(0.0)
        
        element_score = sum(element_scores) / len(element_scores) if element_scores else 0.0
        
        # Combine length and element scores
        return (length_score + element_score) / 2


class ThresholdMatchComparator(BaseResultComparator):
    """Comparator for numeric threshold matches."""
    
    def compare(self, expected: Any, actual: Any, field: str, **kwargs) -> ComparisonResult:
        """Compare for threshold match."""
        threshold = kwargs.get('threshold', 0.1)  # Default 10% tolerance
        comparison_type = kwargs.get('comparison_type', 'relative')  # 'relative' or 'absolute'
        
        try:
            expected_num = float(expected)
            actual_num = float(actual)
        except (ValueError, TypeError):
            return ComparisonResult(
                field=field,
                comparison_type=ComparisonType.THRESHOLD_MATCH,
                status=ValidationStatus.FAILED,
                expected=expected,
                actual=actual,
                score=0.0,
                message="Threshold match requires numeric values"
            )
        
        if comparison_type == 'relative':
            if expected_num == 0:
                # Handle division by zero
                difference = abs(actual_num)
                matches = difference <= threshold
            else:
                difference = abs(expected_num - actual_num) / abs(expected_num)
                matches = difference <= threshold
        else:  # absolute
            difference = abs(expected_num - actual_num)
            matches = difference <= threshold
        
        # Calculate score based on how close the values are
        if expected_num == actual_num:
            score = 1.0
        elif comparison_type == 'relative' and expected_num != 0:
            score = max(0.0, 1.0 - (abs(expected_num - actual_num) / abs(expected_num)) / threshold)
        else:
            score = max(0.0, 1.0 - difference / threshold)
        
        status = ValidationStatus.PASSED if matches else ValidationStatus.FAILED
        message = f"Threshold match ({comparison_type}): difference {difference:.4f} {'<=' if matches else '>'} {threshold}"
        
        return ComparisonResult(
            field=field,
            comparison_type=ComparisonType.THRESHOLD_MATCH,
            status=status,
            expected=expected,
            actual=actual,
            score=score,
            message=message,
            details={'difference': difference, 'threshold': threshold, 'comparison_type': comparison_type}
        )


class PatternMatchComparator(BaseResultComparator):
    """Comparator for pattern matches using regex."""
    
    def compare(self, expected: Any, actual: Any, field: str, **kwargs) -> ComparisonResult:
        """Compare for pattern match."""
        import re
        
        pattern = kwargs.get('pattern', str(expected))
        flags = kwargs.get('flags', 0)
        
        if not isinstance(actual, str):
            actual_str = str(actual)
        else:
            actual_str = actual
        
        try:
            match = re.search(pattern, actual_str, flags)
            matches = match is not None
            score = 1.0 if matches else 0.0
            status = ValidationStatus.PASSED if matches else ValidationStatus.FAILED
            
            message = f"Pattern match: '{pattern}'" if matches else f"Pattern '{pattern}' not found in '{actual_str}'"
            
            details = {'pattern': pattern, 'flags': flags}
            if match:
                details['match_groups'] = match.groups()
                details['match_span'] = match.span()
            
            return ComparisonResult(
                field=field,
                comparison_type=ComparisonType.PATTERN_MATCH,
                status=status,
                expected=expected,
                actual=actual,
                score=score,
                message=message,
                details=details
            )
            
        except re.error as e:
            return ComparisonResult(
                field=field,
                comparison_type=ComparisonType.PATTERN_MATCH,
                status=ValidationStatus.FAILED,
                expected=expected,
                actual=actual,
                score=0.0,
                message=f"Invalid regex pattern: {str(e)}"
            )


class ResultValidator:
    """Main result validation system."""
    
    def __init__(self):
        """Initialize result validator."""
        self.comparators = {
            ComparisonType.EXACT_MATCH: ExactMatchComparator(),
            ComparisonType.FUZZY_MATCH: FuzzyMatchComparator(),
            ComparisonType.STRUCTURAL_MATCH: StructuralMatchComparator(),
            ComparisonType.THRESHOLD_MATCH: ThresholdMatchComparator(),
            ComparisonType.PATTERN_MATCH: PatternMatchComparator()
        }
        self.logger = logging.getLogger(__name__)
    
    def validate_result(self, 
                       agent_type: str,
                       test_id: str,
                       expected_outputs: Dict[str, Any],
                       actual_outputs: Dict[str, Any],
                       validation_criteria: Dict[str, Any]) -> ValidationResult:
        """Validate test results against expected outputs."""
        comparisons = []
        
        # Perform comparisons based on validation criteria
        for field, criteria in validation_criteria.items():
            if field not in expected_outputs:
                self.logger.warning(f"Expected output missing for field: {field}")
                continue
            
            if field not in actual_outputs:
                comparisons.append(ComparisonResult(
                    field=field,
                    comparison_type=ComparisonType.EXACT_MATCH,
                    status=ValidationStatus.FAILED,
                    expected=expected_outputs[field],
                    actual=None,
                    score=0.0,
                    message=f"Field '{field}' missing from actual outputs"
                ))
                continue
            
            # Determine comparison type and parameters
            comparison_type = ComparisonType(criteria.get('type', 'exact_match'))
            comparison_params = criteria.get('params', {})
            
            # Perform comparison
            comparator = self.comparators.get(comparison_type)
            if not comparator:
                self.logger.error(f"Unknown comparison type: {comparison_type}")
                continue
            
            comparison_result = comparator.compare(
                expected_outputs[field],
                actual_outputs[field],
                field,
                **comparison_params
            )
            comparisons.append(comparison_result)
        
        # Calculate overall score and status
        if not comparisons:
            overall_score = 0.0
            overall_status = ValidationStatus.FAILED
        else:
            overall_score = sum(c.score for c in comparisons) / len(comparisons)
            failed_count = len([c for c in comparisons if c.status == ValidationStatus.FAILED])
            
            if failed_count == 0:
                overall_status = ValidationStatus.PASSED
            elif failed_count < len(comparisons) / 2:
                overall_status = ValidationStatus.WARNING
            else:
                overall_status = ValidationStatus.FAILED
        
        # Create summary
        summary = {
            'total_comparisons': len(comparisons),
            'passed_comparisons': len([c for c in comparisons if c.status == ValidationStatus.PASSED]),
            'failed_comparisons': len([c for c in comparisons if c.status == ValidationStatus.FAILED]),
            'warning_comparisons': len([c for c in comparisons if c.status == ValidationStatus.WARNING]),
            'average_score': overall_score,
            'validation_coverage': len(comparisons) / len(expected_outputs) if expected_outputs else 0.0
        }
        
        return ValidationResult(
            agent_type=agent_type,
            test_id=test_id,
            overall_status=overall_status,
            overall_score=overall_score,
            comparisons=comparisons,
            summary=summary
        )
    
    def validate_agent_specific_results(self, 
                                      agent_type: str,
                                      test_id: str,
                                      test_result: Dict[str, Any]) -> ValidationResult:
        """Validate results with agent-specific criteria."""
        if agent_type == 'PlannerAgent':
            return self._validate_planner_results(test_id, test_result)
        elif agent_type == 'CodeGeneratorAgent':
            return self._validate_codegen_results(test_id, test_result)
        elif agent_type == 'RendererAgent':
            return self._validate_renderer_results(test_id, test_result)
        elif agent_type == 'ErrorHandlerAgent':
            return self._validate_error_handler_results(test_id, test_result)
        elif agent_type == 'HumanLoopAgent':
            return self._validate_human_loop_results(test_id, test_result)
        else:
            return ValidationResult(
                agent_type=agent_type,
                test_id=test_id,
                overall_status=ValidationStatus.FAILED,
                overall_score=0.0,
                comparisons=[],
                summary={'error': f'Unknown agent type: {agent_type}'}
            )
    
    def _validate_planner_results(self, test_id: str, test_result: Dict[str, Any]) -> ValidationResult:
        """Validate PlannerAgent results."""
        comparisons = []
        
        # Check if test was successful
        if not test_result.get('success', False):
            comparisons.append(ComparisonResult(
                field='success',
                comparison_type=ComparisonType.EXACT_MATCH,
                status=ValidationStatus.FAILED,
                expected=True,
                actual=test_result.get('success', False),
                score=0.0,
                message="Test execution failed"
            ))
        else:
            comparisons.append(ComparisonResult(
                field='success',
                comparison_type=ComparisonType.EXACT_MATCH,
                status=ValidationStatus.PASSED,
                expected=True,
                actual=True,
                score=1.0,
                message="Test execution successful"
            ))
        
        # Validate outputs structure
        outputs = test_result.get('outputs', {})
        
        # Check scene outline
        if 'scene_outline' in outputs:
            outline = outputs['scene_outline']
            if outline and len(outline.strip()) > 0:
                comparisons.append(ComparisonResult(
                    field='scene_outline',
                    comparison_type=ComparisonType.PATTERN_MATCH,
                    status=ValidationStatus.PASSED,
                    expected="Non-empty scene outline",
                    actual=outline,
                    score=1.0,
                    message="Scene outline generated successfully"
                ))
            else:
                comparisons.append(ComparisonResult(
                    field='scene_outline',
                    comparison_type=ComparisonType.EXACT_MATCH,
                    status=ValidationStatus.FAILED,
                    expected="Non-empty scene outline",
                    actual=outline,
                    score=0.0,
                    message="Scene outline is empty or missing"
                ))
        
        # Check scene implementations
        if 'scene_implementations' in outputs:
            implementations = outputs['scene_implementations']
            if isinstance(implementations, dict) and len(implementations) > 0:
                comparisons.append(ComparisonResult(
                    field='scene_implementations',
                    comparison_type=ComparisonType.STRUCTURAL_MATCH,
                    status=ValidationStatus.PASSED,
                    expected="Dictionary with scene implementations",
                    actual=implementations,
                    score=1.0,
                    message=f"Generated {len(implementations)} scene implementations"
                ))
            else:
                comparisons.append(ComparisonResult(
                    field='scene_implementations',
                    comparison_type=ComparisonType.STRUCTURAL_MATCH,
                    status=ValidationStatus.FAILED,
                    expected="Dictionary with scene implementations",
                    actual=implementations,
                    score=0.0,
                    message="Scene implementations missing or invalid"
                ))
        
        # Check detected plugins
        if 'detected_plugins' in outputs:
            plugins = outputs['detected_plugins']
            if isinstance(plugins, list):
                comparisons.append(ComparisonResult(
                    field='detected_plugins',
                    comparison_type=ComparisonType.STRUCTURAL_MATCH,
                    status=ValidationStatus.PASSED,
                    expected="List of plugins",
                    actual=plugins,
                    score=1.0,
                    message=f"Detected {len(plugins)} plugins"
                ))
            else:
                comparisons.append(ComparisonResult(
                    field='detected_plugins',
                    comparison_type=ComparisonType.STRUCTURAL_MATCH,
                    status=ValidationStatus.FAILED,
                    expected="List of plugins",
                    actual=plugins,
                    score=0.0,
                    message="Plugin detection failed"
                ))
        
        # Calculate overall score and status
        overall_score = sum(c.score for c in comparisons) / len(comparisons) if comparisons else 0.0
        failed_count = len([c for c in comparisons if c.status == ValidationStatus.FAILED])
        
        if failed_count == 0:
            overall_status = ValidationStatus.PASSED
        elif failed_count < len(comparisons) / 2:
            overall_status = ValidationStatus.WARNING
        else:
            overall_status = ValidationStatus.FAILED
        
        return ValidationResult(
            agent_type='PlannerAgent',
            test_id=test_id,
            overall_status=overall_status,
            overall_score=overall_score,
            comparisons=comparisons,
            summary={
                'total_comparisons': len(comparisons),
                'passed_comparisons': len([c for c in comparisons if c.status == ValidationStatus.PASSED]),
                'failed_comparisons': failed_count,
                'average_score': overall_score
            }
        )
    
    def _validate_codegen_results(self, test_id: str, test_result: Dict[str, Any]) -> ValidationResult:
        """Validate CodeGeneratorAgent results."""
        comparisons = []
        
        # Check if test was successful
        success = test_result.get('success', False)
        comparisons.append(ComparisonResult(
            field='success',
            comparison_type=ComparisonType.EXACT_MATCH,
            status=ValidationStatus.PASSED if success else ValidationStatus.FAILED,
            expected=True,
            actual=success,
            score=1.0 if success else 0.0,
            message="Test execution successful" if success else "Test execution failed"
        ))
        
        # Validate outputs structure
        outputs = test_result.get('outputs', {})
        
        # Check generated code
        if 'generated_code' in outputs:
            generated_code = outputs['generated_code']
            if isinstance(generated_code, dict) and len(generated_code) > 0:
                # Check if all code entries are non-empty strings
                valid_code_count = 0
                for scene_num, code in generated_code.items():
                    if isinstance(code, str) and len(code.strip()) > 0:
                        valid_code_count += 1
                
                score = valid_code_count / len(generated_code) if generated_code else 0.0
                status = ValidationStatus.PASSED if score >= 0.8 else ValidationStatus.WARNING if score >= 0.5 else ValidationStatus.FAILED
                
                comparisons.append(ComparisonResult(
                    field='generated_code',
                    comparison_type=ComparisonType.STRUCTURAL_MATCH,
                    status=status,
                    expected="Dictionary with valid code for each scene",
                    actual=generated_code,
                    score=score,
                    message=f"Generated valid code for {valid_code_count}/{len(generated_code)} scenes"
                ))
            else:
                comparisons.append(ComparisonResult(
                    field='generated_code',
                    comparison_type=ComparisonType.STRUCTURAL_MATCH,
                    status=ValidationStatus.FAILED,
                    expected="Dictionary with generated code",
                    actual=generated_code,
                    score=0.0,
                    message="Generated code missing or invalid"
                ))
        
        # Check validation results
        validation = test_result.get('validation', {})
        if validation:
            valid_scenes = sum(1 for v in validation.values() if v.get('valid', False))
            total_scenes = len(validation)
            validation_score = valid_scenes / total_scenes if total_scenes > 0 else 0.0
            
            status = ValidationStatus.PASSED if validation_score >= 0.8 else ValidationStatus.WARNING if validation_score >= 0.5 else ValidationStatus.FAILED
            
            comparisons.append(ComparisonResult(
                field='code_validation',
                comparison_type=ComparisonType.THRESHOLD_MATCH,
                status=status,
                expected="Valid code for all scenes",
                actual=f"{valid_scenes}/{total_scenes} scenes valid",
                score=validation_score,
                message=f"Code validation passed for {valid_scenes}/{total_scenes} scenes"
            ))
        
        # Calculate overall score and status
        overall_score = sum(c.score for c in comparisons) / len(comparisons) if comparisons else 0.0
        failed_count = len([c for c in comparisons if c.status == ValidationStatus.FAILED])
        
        if failed_count == 0:
            overall_status = ValidationStatus.PASSED
        elif failed_count < len(comparisons) / 2:
            overall_status = ValidationStatus.WARNING
        else:
            overall_status = ValidationStatus.FAILED
        
        return ValidationResult(
            agent_type='CodeGeneratorAgent',
            test_id=test_id,
            overall_status=overall_status,
            overall_score=overall_score,
            comparisons=comparisons,
            summary={
                'total_comparisons': len(comparisons),
                'passed_comparisons': len([c for c in comparisons if c.status == ValidationStatus.PASSED]),
                'failed_comparisons': failed_count,
                'average_score': overall_score
            }
        )
    
    def _validate_renderer_results(self, test_id: str, test_result: Dict[str, Any]) -> ValidationResult:
        """Validate RendererAgent results."""
        comparisons = []
        
        # Check if test was successful
        success = test_result.get('success', False)
        comparisons.append(ComparisonResult(
            field='success',
            comparison_type=ComparisonType.EXACT_MATCH,
            status=ValidationStatus.PASSED if success else ValidationStatus.FAILED,
            expected=True,
            actual=success,
            score=1.0 if success else 0.0,
            message="Test execution successful" if success else "Test execution failed"
        ))
        
        # Validate outputs structure
        outputs = test_result.get('outputs', {})
        
        # Check rendered videos
        if 'rendered_videos' in outputs:
            rendered_videos = outputs['rendered_videos']
            if isinstance(rendered_videos, dict) and len(rendered_videos) > 0:
                comparisons.append(ComparisonResult(
                    field='rendered_videos',
                    comparison_type=ComparisonType.STRUCTURAL_MATCH,
                    status=ValidationStatus.PASSED,
                    expected="Dictionary with rendered videos",
                    actual=rendered_videos,
                    score=1.0,
                    message=f"Successfully rendered {len(rendered_videos)} videos"
                ))
            else:
                comparisons.append(ComparisonResult(
                    field='rendered_videos',
                    comparison_type=ComparisonType.STRUCTURAL_MATCH,
                    status=ValidationStatus.FAILED,
                    expected="Dictionary with rendered videos",
                    actual=rendered_videos,
                    score=0.0,
                    message="No videos were rendered"
                ))
        
        # Check rendering errors
        errors = test_result.get('errors', {})
        if errors:
            error_count = len(errors)
            total_scenes = len(test_result.get('inputs', {}).get('generated_code', {}))
            error_rate = error_count / total_scenes if total_scenes > 0 else 1.0
            
            status = ValidationStatus.PASSED if error_rate == 0 else ValidationStatus.WARNING if error_rate < 0.5 else ValidationStatus.FAILED
            score = 1.0 - error_rate
            
            comparisons.append(ComparisonResult(
                field='rendering_errors',
                comparison_type=ComparisonType.THRESHOLD_MATCH,
                status=status,
                expected="No rendering errors",
                actual=f"{error_count} errors",
                score=score,
                message=f"Rendering errors in {error_count}/{total_scenes} scenes"
            ))
        
        # Calculate overall score and status
        overall_score = sum(c.score for c in comparisons) / len(comparisons) if comparisons else 0.0
        failed_count = len([c for c in comparisons if c.status == ValidationStatus.FAILED])
        
        if failed_count == 0:
            overall_status = ValidationStatus.PASSED
        elif failed_count < len(comparisons) / 2:
            overall_status = ValidationStatus.WARNING
        else:
            overall_status = ValidationStatus.FAILED
        
        return ValidationResult(
            agent_type='RendererAgent',
            test_id=test_id,
            overall_status=overall_status,
            overall_score=overall_score,
            comparisons=comparisons,
            summary={
                'total_comparisons': len(comparisons),
                'passed_comparisons': len([c for c in comparisons if c.status == ValidationStatus.PASSED]),
                'failed_comparisons': failed_count,
                'average_score': overall_score
            }
        )
    
    def _validate_error_handler_results(self, test_id: str, test_result: Dict[str, Any]) -> ValidationResult:
        """Validate ErrorHandlerAgent results."""
        comparisons = []
        
        # Check if test was successful
        success = test_result.get('success', False)
        comparisons.append(ComparisonResult(
            field='success',
            comparison_type=ComparisonType.EXACT_MATCH,
            status=ValidationStatus.PASSED if success else ValidationStatus.FAILED,
            expected=True,
            actual=success,
            score=1.0 if success else 0.0,
            message="Test execution successful" if success else "Test execution failed"
        ))
        
        # Validate error handling results
        outputs = test_result.get('outputs', {})
        
        if 'recovery_results' in outputs:
            recovery_results = outputs['recovery_results']
            if isinstance(recovery_results, dict):
                successful_recoveries = sum(1 for r in recovery_results.values() if r.get('recovered', False))
                total_scenarios = len(recovery_results)
                recovery_rate = successful_recoveries / total_scenarios if total_scenarios > 0 else 0.0
                
                status = ValidationStatus.PASSED if recovery_rate >= 0.7 else ValidationStatus.WARNING if recovery_rate >= 0.4 else ValidationStatus.FAILED
                
                comparisons.append(ComparisonResult(
                    field='error_recovery',
                    comparison_type=ComparisonType.THRESHOLD_MATCH,
                    status=status,
                    expected="High error recovery rate",
                    actual=f"{successful_recoveries}/{total_scenarios} recovered",
                    score=recovery_rate,
                    message=f"Error recovery rate: {recovery_rate:.2f}"
                ))
        
        # Calculate overall score and status
        overall_score = sum(c.score for c in comparisons) / len(comparisons) if comparisons else 0.0
        failed_count = len([c for c in comparisons if c.status == ValidationStatus.FAILED])
        
        if failed_count == 0:
            overall_status = ValidationStatus.PASSED
        elif failed_count < len(comparisons) / 2:
            overall_status = ValidationStatus.WARNING
        else:
            overall_status = ValidationStatus.FAILED
        
        return ValidationResult(
            agent_type='ErrorHandlerAgent',
            test_id=test_id,
            overall_status=overall_status,
            overall_score=overall_score,
            comparisons=comparisons,
            summary={
                'total_comparisons': len(comparisons),
                'passed_comparisons': len([c for c in comparisons if c.status == ValidationStatus.PASSED]),
                'failed_comparisons': failed_count,
                'average_score': overall_score
            }
        )
    
    def _validate_human_loop_results(self, test_id: str, test_result: Dict[str, Any]) -> ValidationResult:
        """Validate HumanLoopAgent results."""
        comparisons = []
        
        # Check if test was successful
        success = test_result.get('success', False)
        comparisons.append(ComparisonResult(
            field='success',
            comparison_type=ComparisonType.EXACT_MATCH,
            status=ValidationStatus.PASSED if success else ValidationStatus.FAILED,
            expected=True,
            actual=success,
            score=1.0 if success else 0.0,
            message="Test execution successful" if success else "Test execution failed"
        ))
        
        # Validate intervention results
        outputs = test_result.get('outputs', {})
        
        if 'intervention_results' in outputs:
            intervention_results = outputs['intervention_results']
            if isinstance(intervention_results, dict):
                successful_interventions = sum(1 for r in intervention_results.values() if r.get('handled', False))
                total_scenarios = len(intervention_results)
                intervention_rate = successful_interventions / total_scenarios if total_scenarios > 0 else 0.0
                
                status = ValidationStatus.PASSED if intervention_rate >= 0.8 else ValidationStatus.WARNING if intervention_rate >= 0.5 else ValidationStatus.FAILED
                
                comparisons.append(ComparisonResult(
                    field='human_intervention',
                    comparison_type=ComparisonType.THRESHOLD_MATCH,
                    status=status,
                    expected="High intervention success rate",
                    actual=f"{successful_interventions}/{total_scenarios} handled",
                    score=intervention_rate,
                    message=f"Intervention success rate: {intervention_rate:.2f}"
                ))
        
        # Calculate overall score and status
        overall_score = sum(c.score for c in comparisons) / len(comparisons) if comparisons else 0.0
        failed_count = len([c for c in comparisons if c.status == ValidationStatus.FAILED])
        
        if failed_count == 0:
            overall_status = ValidationStatus.PASSED
        elif failed_count < len(comparisons) / 2:
            overall_status = ValidationStatus.WARNING
        else:
            overall_status = ValidationStatus.FAILED
        
        return ValidationResult(
            agent_type='HumanLoopAgent',
            test_id=test_id,
            overall_status=overall_status,
            overall_score=overall_score,
            comparisons=comparisons,
            summary={
                'total_comparisons': len(comparisons),
                'passed_comparisons': len([c for c in comparisons if c.status == ValidationStatus.PASSED]),
                'failed_comparisons': failed_count,
                'average_score': overall_score
            }
        )


# Global instance for easy access
_global_result_validator = ResultValidator()


def get_result_validator() -> ResultValidator:
    """Get the global result validator instance."""
    return _global_result_validator


def validate_test_result(agent_type: str, test_id: str, test_result: Dict[str, Any]) -> ValidationResult:
    """Convenience function to validate test results."""
    return _global_result_validator.validate_agent_specific_results(agent_type, test_id, test_result)
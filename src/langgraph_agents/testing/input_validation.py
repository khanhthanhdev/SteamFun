"""
Enhanced input validation and preprocessing for agent testing.

This module provides comprehensive input validation, preprocessing, and
sanitization capabilities for agent test scenarios.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    
    field: str
    message: str
    severity: ValidationSeverity
    suggested_fix: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'field': self.field,
            'message': self.message,
            'severity': self.severity.value,
            'suggested_fix': self.suggested_fix
        }


@dataclass
class ValidationResult:
    """Result of input validation."""
    
    is_valid: bool
    issues: List[ValidationIssue]
    processed_inputs: Dict[str, Any]
    
    def get_errors(self) -> List[ValidationIssue]:
        """Get only error-level issues."""
        return [issue for issue in self.issues if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]
    
    def get_warnings(self) -> List[ValidationIssue]:
        """Get warning-level issues."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.WARNING]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_valid': self.is_valid,
            'issues': [issue.to_dict() for issue in self.issues],
            'processed_inputs': self.processed_inputs,
            'error_count': len(self.get_errors()),
            'warning_count': len(self.get_warnings())
        }


class BaseInputValidator:
    """Base class for input validators."""
    
    def __init__(self):
        """Initialize validator."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def validate(self, inputs: Dict[str, Any]) -> ValidationResult:
        """Validate inputs and return result."""
        issues = []
        processed_inputs = inputs.copy()
        
        # Perform validation
        issues.extend(self._validate_required_fields(inputs))
        issues.extend(self._validate_field_types(inputs))
        issues.extend(self._validate_field_values(inputs))
        
        # Preprocess inputs
        processed_inputs = self._preprocess_inputs(processed_inputs, issues)
        
        # Determine if validation passed
        errors = [issue for issue in issues if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            processed_inputs=processed_inputs
        )
    
    def _validate_required_fields(self, inputs: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate required fields are present."""
        issues = []
        required_fields = self._get_required_fields()
        
        for field in required_fields:
            if field not in inputs:
                issues.append(ValidationIssue(
                    field=field,
                    message=f"Required field '{field}' is missing",
                    severity=ValidationSeverity.ERROR,
                    suggested_fix=f"Add '{field}' field to inputs"
                ))
            elif inputs[field] is None:
                issues.append(ValidationIssue(
                    field=field,
                    message=f"Required field '{field}' cannot be null",
                    severity=ValidationSeverity.ERROR,
                    suggested_fix=f"Provide a valid value for '{field}'"
                ))
        
        return issues
    
    def _validate_field_types(self, inputs: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate field types."""
        issues = []
        field_types = self._get_field_types()
        
        for field, expected_type in field_types.items():
            if field in inputs and inputs[field] is not None:
                if not isinstance(inputs[field], expected_type):
                    issues.append(ValidationIssue(
                        field=field,
                        message=f"Field '{field}' must be of type {expected_type.__name__}, got {type(inputs[field]).__name__}",
                        severity=ValidationSeverity.ERROR,
                        suggested_fix=f"Convert '{field}' to {expected_type.__name__}"
                    ))
        
        return issues
    
    def _validate_field_values(self, inputs: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate field values."""
        issues = []
        # Override in subclasses for specific validation
        return issues
    
    def _preprocess_inputs(self, inputs: Dict[str, Any], issues: List[ValidationIssue]) -> Dict[str, Any]:
        """Preprocess inputs."""
        # Override in subclasses for specific preprocessing
        return inputs
    
    def _get_required_fields(self) -> List[str]:
        """Get list of required fields."""
        return []
    
    def _get_field_types(self) -> Dict[str, type]:
        """Get expected field types."""
        return {}


class PlannerInputValidator(BaseInputValidator):
    """Input validator for PlannerAgent."""
    
    def _get_required_fields(self) -> List[str]:
        """Get required fields for planner inputs."""
        return ['topic', 'description']
    
    def _get_field_types(self) -> Dict[str, type]:
        """Get expected field types."""
        return {
            'topic': str,
            'description': str,
            'session_id': str
        }
    
    def _validate_field_values(self, inputs: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate planner-specific field values."""
        issues = []
        
        # Validate topic
        if 'topic' in inputs and inputs['topic']:
            topic = inputs['topic'].strip()
            if len(topic) < 3:
                issues.append(ValidationIssue(
                    field='topic',
                    message="Topic must be at least 3 characters long",
                    severity=ValidationSeverity.ERROR,
                    suggested_fix="Provide a more descriptive topic"
                ))
            elif len(topic) > 200:
                issues.append(ValidationIssue(
                    field='topic',
                    message="Topic is too long (max 200 characters)",
                    severity=ValidationSeverity.WARNING,
                    suggested_fix="Shorten the topic to under 200 characters"
                ))
        
        # Validate description
        if 'description' in inputs and inputs['description']:
            description = inputs['description'].strip()
            if len(description) < 10:
                issues.append(ValidationIssue(
                    field='description',
                    message="Description must be at least 10 characters long",
                    severity=ValidationSeverity.ERROR,
                    suggested_fix="Provide a more detailed description"
                ))
            elif len(description) > 5000:
                issues.append(ValidationIssue(
                    field='description',
                    message="Description is too long (max 5000 characters)",
                    severity=ValidationSeverity.WARNING,
                    suggested_fix="Shorten the description to under 5000 characters"
                ))
        
        return issues
    
    def _preprocess_inputs(self, inputs: Dict[str, Any], issues: List[ValidationIssue]) -> Dict[str, Any]:
        """Preprocess planner inputs."""
        processed = inputs.copy()
        
        # Clean and normalize topic
        if 'topic' in processed and processed['topic']:
            processed['topic'] = processed['topic'].strip()
        
        # Clean and normalize description
        if 'description' in processed and processed['description']:
            processed['description'] = processed['description'].strip()
        
        # Generate session_id if not provided
        if 'session_id' not in processed or not processed['session_id']:
            import uuid
            processed['session_id'] = f"planner_{uuid.uuid4().hex[:8]}"
        
        return processed


class CodeGeneratorInputValidator(BaseInputValidator):
    """Input validator for CodeGeneratorAgent."""
    
    def _get_required_fields(self) -> List[str]:
        """Get required fields for code generator inputs."""
        return ['topic', 'description', 'scene_outline', 'scene_implementations']
    
    def _get_field_types(self) -> Dict[str, type]:
        """Get expected field types."""
        return {
            'topic': str,
            'description': str,
            'scene_outline': str,
            'scene_implementations': dict,
            'session_id': str
        }
    
    def _validate_field_values(self, inputs: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate code generator-specific field values."""
        issues = []
        
        # Validate scene_implementations structure
        if 'scene_implementations' in inputs and inputs['scene_implementations']:
            scene_impls = inputs['scene_implementations']
            
            if not isinstance(scene_impls, dict):
                issues.append(ValidationIssue(
                    field='scene_implementations',
                    message="scene_implementations must be a dictionary",
                    severity=ValidationSeverity.ERROR,
                    suggested_fix="Convert scene_implementations to dictionary format"
                ))
            else:
                for scene_key, implementation in scene_impls.items():
                    # Validate scene number
                    try:
                        scene_num = int(scene_key)
                        if scene_num < 1:
                            issues.append(ValidationIssue(
                                field='scene_implementations',
                                message=f"Scene number must be positive: {scene_num}",
                                severity=ValidationSeverity.ERROR,
                                suggested_fix=f"Use positive scene number instead of {scene_num}"
                            ))
                    except (ValueError, TypeError):
                        issues.append(ValidationIssue(
                            field='scene_implementations',
                            message=f"Invalid scene number: {scene_key}",
                            severity=ValidationSeverity.ERROR,
                            suggested_fix=f"Use integer scene number instead of '{scene_key}'"
                        ))
                    
                    # Validate implementation content
                    if not isinstance(implementation, str):
                        issues.append(ValidationIssue(
                            field='scene_implementations',
                            message=f"Scene implementation must be string for scene {scene_key}",
                            severity=ValidationSeverity.ERROR,
                            suggested_fix=f"Convert scene {scene_key} implementation to string"
                        ))
                    elif not implementation.strip():
                        issues.append(ValidationIssue(
                            field='scene_implementations',
                            message=f"Scene implementation cannot be empty for scene {scene_key}",
                            severity=ValidationSeverity.ERROR,
                            suggested_fix=f"Provide implementation content for scene {scene_key}"
                        ))
        
        return issues
    
    def _preprocess_inputs(self, inputs: Dict[str, Any], issues: List[ValidationIssue]) -> Dict[str, Any]:
        """Preprocess code generator inputs."""
        processed = inputs.copy()
        
        # Clean basic fields
        for field in ['topic', 'description', 'scene_outline']:
            if field in processed and processed[field]:
                processed[field] = processed[field].strip()
        
        # Convert and clean scene_implementations
        if 'scene_implementations' in processed and processed['scene_implementations']:
            scene_impls = processed['scene_implementations']
            cleaned_impls = {}
            
            for scene_key, implementation in scene_impls.items():
                try:
                    # Convert key to integer
                    scene_num = int(scene_key)
                    # Clean implementation
                    if isinstance(implementation, str):
                        cleaned_impls[scene_num] = implementation.strip()
                    else:
                        cleaned_impls[scene_num] = str(implementation).strip()
                except (ValueError, TypeError):
                    # Keep original key if conversion fails
                    cleaned_impls[scene_key] = implementation
            
            processed['scene_implementations'] = cleaned_impls
        
        # Generate session_id if not provided
        if 'session_id' not in processed or not processed['session_id']:
            import uuid
            processed['session_id'] = f"codegen_{uuid.uuid4().hex[:8]}"
        
        return processed


class RendererInputValidator(BaseInputValidator):
    """Input validator for RendererAgent."""
    
    def _get_required_fields(self) -> List[str]:
        """Get required fields for renderer inputs."""
        return ['generated_code', 'file_prefix']
    
    def _get_field_types(self) -> Dict[str, type]:
        """Get expected field types."""
        return {
            'generated_code': dict,
            'file_prefix': str,
            'quality': str,
            'session_id': str,
            'topic': str
        }
    
    def _validate_field_values(self, inputs: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate renderer-specific field values."""
        issues = []
        
        # Validate generated_code structure
        if 'generated_code' in inputs and inputs['generated_code']:
            generated_code = inputs['generated_code']
            
            if not isinstance(generated_code, dict):
                issues.append(ValidationIssue(
                    field='generated_code',
                    message="generated_code must be a dictionary",
                    severity=ValidationSeverity.ERROR,
                    suggested_fix="Convert generated_code to dictionary format"
                ))
            else:
                for scene_key, code in generated_code.items():
                    # Validate scene number
                    try:
                        scene_num = int(scene_key)
                        if scene_num < 1:
                            issues.append(ValidationIssue(
                                field='generated_code',
                                message=f"Scene number must be positive: {scene_num}",
                                severity=ValidationSeverity.ERROR,
                                suggested_fix=f"Use positive scene number instead of {scene_num}"
                            ))
                    except (ValueError, TypeError):
                        issues.append(ValidationIssue(
                            field='generated_code',
                            message=f"Invalid scene number: {scene_key}",
                            severity=ValidationSeverity.ERROR,
                            suggested_fix=f"Use integer scene number instead of '{scene_key}'"
                        ))
                    
                    # Validate code content
                    if not isinstance(code, str):
                        issues.append(ValidationIssue(
                            field='generated_code',
                            message=f"Code must be string for scene {scene_key}",
                            severity=ValidationSeverity.ERROR,
                            suggested_fix=f"Convert scene {scene_key} code to string"
                        ))
                    elif not code.strip():
                        issues.append(ValidationIssue(
                            field='generated_code',
                            message=f"Code cannot be empty for scene {scene_key}",
                            severity=ValidationSeverity.ERROR,
                            suggested_fix=f"Provide code content for scene {scene_key}"
                        ))
                    else:
                        # Basic Python syntax validation
                        if not self._is_valid_python_syntax(code):
                            issues.append(ValidationIssue(
                                field='generated_code',
                                message=f"Invalid Python syntax in scene {scene_key}",
                                severity=ValidationSeverity.WARNING,
                                suggested_fix=f"Check Python syntax for scene {scene_key}"
                            ))
        
        # Validate quality setting
        if 'quality' in inputs and inputs['quality']:
            valid_qualities = ['low', 'medium', 'high', 'ultra', 'preview']
            if inputs['quality'] not in valid_qualities:
                issues.append(ValidationIssue(
                    field='quality',
                    message=f"Invalid quality setting: {inputs['quality']}",
                    severity=ValidationSeverity.ERROR,
                    suggested_fix=f"Use one of: {', '.join(valid_qualities)}"
                ))
        
        # Validate file_prefix
        if 'file_prefix' in inputs and inputs['file_prefix']:
            file_prefix = inputs['file_prefix']
            if not re.match(r'^[a-zA-Z0-9_-]+$', file_prefix):
                issues.append(ValidationIssue(
                    field='file_prefix',
                    message="file_prefix can only contain letters, numbers, underscores, and hyphens",
                    severity=ValidationSeverity.ERROR,
                    suggested_fix="Use only alphanumeric characters, underscores, and hyphens"
                ))
        
        return issues
    
    def _preprocess_inputs(self, inputs: Dict[str, Any], issues: List[ValidationIssue]) -> Dict[str, Any]:
        """Preprocess renderer inputs."""
        processed = inputs.copy()
        
        # Clean basic fields
        for field in ['file_prefix', 'quality', 'topic']:
            if field in processed and processed[field]:
                processed[field] = processed[field].strip()
        
        # Set default quality if not provided
        if 'quality' not in processed or not processed['quality']:
            processed['quality'] = 'medium'
        
        # Convert and clean generated_code
        if 'generated_code' in processed and processed['generated_code']:
            generated_code = processed['generated_code']
            cleaned_code = {}
            
            for scene_key, code in generated_code.items():
                try:
                    # Convert key to integer
                    scene_num = int(scene_key)
                    # Clean code
                    if isinstance(code, str):
                        cleaned_code[scene_num] = code.strip()
                    else:
                        cleaned_code[scene_num] = str(code).strip()
                except (ValueError, TypeError):
                    # Keep original key if conversion fails
                    cleaned_code[scene_key] = code
            
            processed['generated_code'] = cleaned_code
        
        # Generate session_id if not provided
        if 'session_id' not in processed or not processed['session_id']:
            import uuid
            processed['session_id'] = f"renderer_{uuid.uuid4().hex[:8]}"
        
        return processed
    
    def _is_valid_python_syntax(self, code: str) -> bool:
        """Check if code has valid Python syntax."""
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False


class ErrorHandlerInputValidator(BaseInputValidator):
    """Input validator for ErrorHandlerAgent."""
    
    def _get_required_fields(self) -> List[str]:
        """Get required fields for error handler inputs."""
        return ['error_scenarios']
    
    def _get_field_types(self) -> Dict[str, type]:
        """Get expected field types."""
        return {
            'error_scenarios': list,
            'session_id': str
        }
    
    def _validate_field_values(self, inputs: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate error handler-specific field values."""
        issues = []
        
        # Validate error_scenarios structure
        if 'error_scenarios' in inputs and inputs['error_scenarios']:
            error_scenarios = inputs['error_scenarios']
            
            if not isinstance(error_scenarios, list):
                issues.append(ValidationIssue(
                    field='error_scenarios',
                    message="error_scenarios must be a list",
                    severity=ValidationSeverity.ERROR,
                    suggested_fix="Convert error_scenarios to list format"
                ))
            else:
                for i, scenario in enumerate(error_scenarios):
                    if not isinstance(scenario, dict):
                        issues.append(ValidationIssue(
                            field='error_scenarios',
                            message=f"Error scenario {i} must be a dictionary",
                            severity=ValidationSeverity.ERROR,
                            suggested_fix=f"Convert scenario {i} to dictionary format"
                        ))
                        continue
                    
                    # Validate required scenario fields
                    required_fields = ['error_type', 'message', 'step', 'severity']
                    for field in required_fields:
                        if field not in scenario:
                            issues.append(ValidationIssue(
                                field='error_scenarios',
                                message=f"Error scenario {i} missing required field: {field}",
                                severity=ValidationSeverity.ERROR,
                                suggested_fix=f"Add '{field}' to scenario {i}"
                            ))
                    
                    # Validate error_type
                    if 'error_type' in scenario:
                        valid_types = ['MODEL', 'TIMEOUT', 'VALIDATION', 'SYSTEM', 'CONTENT', 'TRANSIENT', 'RATE_LIMIT']
                        if scenario['error_type'] not in valid_types:
                            issues.append(ValidationIssue(
                                field='error_scenarios',
                                message=f"Invalid error_type in scenario {i}: {scenario['error_type']}",
                                severity=ValidationSeverity.WARNING,
                                suggested_fix=f"Use one of: {', '.join(valid_types)}"
                            ))
                    
                    # Validate severity
                    if 'severity' in scenario:
                        valid_severities = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
                        if scenario['severity'] not in valid_severities:
                            issues.append(ValidationIssue(
                                field='error_scenarios',
                                message=f"Invalid severity in scenario {i}: {scenario['severity']}",
                                severity=ValidationSeverity.WARNING,
                                suggested_fix=f"Use one of: {', '.join(valid_severities)}"
                            ))
        
        return issues
    
    def _preprocess_inputs(self, inputs: Dict[str, Any], issues: List[ValidationIssue]) -> Dict[str, Any]:
        """Preprocess error handler inputs."""
        processed = inputs.copy()
        
        # Clean error scenarios
        if 'error_scenarios' in processed and processed['error_scenarios']:
            cleaned_scenarios = []
            for scenario in processed['error_scenarios']:
                if isinstance(scenario, dict):
                    cleaned_scenario = {}
                    for key, value in scenario.items():
                        if isinstance(value, str):
                            cleaned_scenario[key] = value.strip()
                        else:
                            cleaned_scenario[key] = value
                    cleaned_scenarios.append(cleaned_scenario)
                else:
                    cleaned_scenarios.append(scenario)
            processed['error_scenarios'] = cleaned_scenarios
        
        # Generate session_id if not provided
        if 'session_id' not in processed or not processed['session_id']:
            import uuid
            processed['session_id'] = f"errorhandler_{uuid.uuid4().hex[:8]}"
        
        return processed


class HumanLoopInputValidator(BaseInputValidator):
    """Input validator for HumanLoopAgent."""
    
    def _get_required_fields(self) -> List[str]:
        """Get required fields for human loop inputs."""
        return ['intervention_scenarios']
    
    def _get_field_types(self) -> Dict[str, type]:
        """Get expected field types."""
        return {
            'intervention_scenarios': list,
            'session_id': str
        }
    
    def _validate_field_values(self, inputs: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate human loop-specific field values."""
        issues = []
        
        # Validate intervention_scenarios structure
        if 'intervention_scenarios' in inputs and inputs['intervention_scenarios']:
            intervention_scenarios = inputs['intervention_scenarios']
            
            if not isinstance(intervention_scenarios, list):
                issues.append(ValidationIssue(
                    field='intervention_scenarios',
                    message="intervention_scenarios must be a list",
                    severity=ValidationSeverity.ERROR,
                    suggested_fix="Convert intervention_scenarios to list format"
                ))
            else:
                for i, scenario in enumerate(intervention_scenarios):
                    if not isinstance(scenario, dict):
                        issues.append(ValidationIssue(
                            field='intervention_scenarios',
                            message=f"Intervention scenario {i} must be a dictionary",
                            severity=ValidationSeverity.ERROR,
                            suggested_fix=f"Convert scenario {i} to dictionary format"
                        ))
                        continue
                    
                    # Validate required scenario fields
                    required_fields = ['intervention_type', 'trigger_condition', 'expected_action']
                    for field in required_fields:
                        if field not in scenario:
                            issues.append(ValidationIssue(
                                field='intervention_scenarios',
                                message=f"Intervention scenario {i} missing required field: {field}",
                                severity=ValidationSeverity.ERROR,
                                suggested_fix=f"Add '{field}' to scenario {i}"
                            ))
                    
                    # Validate intervention_type
                    if 'intervention_type' in scenario:
                        valid_types = ['error_escalation', 'quality_review', 'manual_override', 'approval_required']
                        if scenario['intervention_type'] not in valid_types:
                            issues.append(ValidationIssue(
                                field='intervention_scenarios',
                                message=f"Invalid intervention_type in scenario {i}: {scenario['intervention_type']}",
                                severity=ValidationSeverity.WARNING,
                                suggested_fix=f"Use one of: {', '.join(valid_types)}"
                            ))
        
        return issues
    
    def _preprocess_inputs(self, inputs: Dict[str, Any], issues: List[ValidationIssue]) -> Dict[str, Any]:
        """Preprocess human loop inputs."""
        processed = inputs.copy()
        
        # Clean intervention scenarios
        if 'intervention_scenarios' in processed and processed['intervention_scenarios']:
            cleaned_scenarios = []
            for scenario in processed['intervention_scenarios']:
                if isinstance(scenario, dict):
                    cleaned_scenario = {}
                    for key, value in scenario.items():
                        if isinstance(value, str):
                            cleaned_scenario[key] = value.strip()
                        else:
                            cleaned_scenario[key] = value
                    cleaned_scenarios.append(cleaned_scenario)
                else:
                    cleaned_scenarios.append(scenario)
            processed['intervention_scenarios'] = cleaned_scenarios
        
        # Generate session_id if not provided
        if 'session_id' not in processed or not processed['session_id']:
            import uuid
            processed['session_id'] = f"humanloop_{uuid.uuid4().hex[:8]}"
        
        return processed


class InputValidationManager:
    """Manager for input validation across all agent types."""
    
    def __init__(self):
        """Initialize validation manager."""
        self.validators = {
            'PlannerAgent': PlannerInputValidator(),
            'CodeGeneratorAgent': CodeGeneratorInputValidator(),
            'RendererAgent': RendererInputValidator(),
            'ErrorHandlerAgent': ErrorHandlerInputValidator(),
            'HumanLoopAgent': HumanLoopInputValidator()
        }
        self.logger = logging.getLogger(__name__)
    
    def validate_inputs(self, agent_type: str, inputs: Dict[str, Any]) -> ValidationResult:
        """Validate inputs for specific agent type."""
        if agent_type not in self.validators:
            return ValidationResult(
                is_valid=False,
                issues=[ValidationIssue(
                    field='agent_type',
                    message=f"Unknown agent type: {agent_type}",
                    severity=ValidationSeverity.ERROR,
                    suggested_fix=f"Use one of: {', '.join(self.validators.keys())}"
                )],
                processed_inputs=inputs
            )
        
        validator = self.validators[agent_type]
        result = validator.validate(inputs)
        
        self.logger.info(f"Validated inputs for {agent_type}: {result.is_valid} (errors: {len(result.get_errors())}, warnings: {len(result.get_warnings())})")
        
        return result
    
    def get_supported_agents(self) -> List[str]:
        """Get list of supported agent types."""
        return list(self.validators.keys())


# Global instance for easy access
_global_validation_manager = InputValidationManager()


def get_validation_manager() -> InputValidationManager:
    """Get the global input validation manager instance."""
    return _global_validation_manager


def validate_agent_inputs(agent_type: str, inputs: Dict[str, Any]) -> ValidationResult:
    """Convenience function to validate agent inputs."""
    return _global_validation_manager.validate_inputs(agent_type, inputs)
"""
Data validation utilities for migration integrity checks.

This module provides comprehensive validation of migrated data to ensure
integrity and consistency during the transition from old to new system formats.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Import models directly to avoid circular dependencies
try:
    from ..models.state import VideoGenerationState
    from ..models.config import WorkflowConfig
except ImportError:
    # Handle case where models are not available
    VideoGenerationState = None
    WorkflowConfig = None

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Exception raised during data validation operations."""
    pass


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation issue found during data validation."""
    severity: ValidationSeverity
    category: str
    message: str
    field_path: str
    old_value: Any
    new_value: Any
    suggestion: Optional[str] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report for migrated data."""
    validation_id: str
    timestamp: datetime
    data_type: str
    total_records: int
    valid_records: int
    invalid_records: int
    issues: List[ValidationIssue]
    summary: Dict[str, Any]
    
    @property
    def success_rate(self) -> float:
        """Calculate validation success rate."""
        if self.total_records == 0:
            return 100.0
        return (self.valid_records / self.total_records) * 100.0
    
    @property
    def has_critical_issues(self) -> bool:
        """Check if there are any critical validation issues."""
        return any(issue.severity == ValidationSeverity.CRITICAL for issue in self.issues)
    
    @property
    def has_errors(self) -> bool:
        """Check if there are any error-level validation issues."""
        return any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                  for issue in self.issues)


class DataValidator:
    """
    Data validator for ensuring migration integrity.
    
    This class provides comprehensive validation of migrated data including:
    - State format validation
    - Configuration validation
    - Data consistency checks
    - Field mapping verification
    """
    
    def __init__(self):
        """Initialize data validator."""
        self.validation_rules = self._load_validation_rules()
        logger.info("Data validator initialized")
    
    def validate_state_migration(
        self,
        old_states: List[Dict[str, Any]],
        new_states: List[Any]  # Changed from VideoGenerationState to Any to avoid import issues
    ) -> ValidationReport:
        """
        Validate state migration from old to new format.
        
        Args:
            old_states: List of old state dictionaries
            new_states: List of new Pydantic state objects
            
        Returns:
            ValidationReport: Comprehensive validation report
        """
        validation_id = f"state_migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        issues = []
        valid_records = 0
        
        logger.info(f"Starting state migration validation for {len(old_states)} records")
        
        # Validate count consistency
        if len(old_states) != len(new_states):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="count_mismatch",
                message=f"Record count mismatch: {len(old_states)} old vs {len(new_states)} new",
                field_path="root",
                old_value=len(old_states),
                new_value=len(new_states),
                suggestion="Ensure all records were migrated"
            ))
        
        # Validate each state pair
        for i, (old_state, new_state) in enumerate(zip(old_states, new_states)):
            try:
                state_issues = self._validate_single_state_migration(old_state, new_state, i)
                issues.extend(state_issues)
                
                # Count as valid if no critical or error issues
                if not any(issue.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR] 
                          for issue in state_issues):
                    valid_records += 1
                    
            except Exception as e:
                logger.error(f"Failed to validate state {i}: {e}")
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="validation_error",
                    message=f"Validation failed for record {i}: {e}",
                    field_path=f"record[{i}]",
                    old_value=None,
                    new_value=None,
                    suggestion="Check data format and migration logic"
                ))
        
        # Create summary
        summary = self._create_validation_summary(issues, "state_migration")
        
        report = ValidationReport(
            validation_id=validation_id,
            timestamp=datetime.now(),
            data_type="state_migration",
            total_records=len(old_states),
            valid_records=valid_records,
            invalid_records=len(old_states) - valid_records,
            issues=issues,
            summary=summary
        )
        
        logger.info(f"State migration validation completed: {report.success_rate:.1f}% success rate")
        return report
    
    def validate_config_migration(
        self,
        old_configs: List[Dict[str, Any]],
        new_configs: List[Any]  # Changed from WorkflowConfig to Any to avoid import issues
    ) -> ValidationReport:
        """
        Validate configuration migration from old to new format.
        
        Args:
            old_configs: List of old configuration dictionaries
            new_configs: List of new WorkflowConfig objects
            
        Returns:
            ValidationReport: Comprehensive validation report
        """
        validation_id = f"config_migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        issues = []
        valid_records = 0
        
        logger.info(f"Starting config migration validation for {len(old_configs)} records")
        
        # Validate count consistency
        if len(old_configs) != len(new_configs):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="count_mismatch",
                message=f"Config count mismatch: {len(old_configs)} old vs {len(new_configs)} new",
                field_path="root",
                old_value=len(old_configs),
                new_value=len(new_configs),
                suggestion="Ensure all configurations were migrated"
            ))
        
        # Validate each config pair
        for i, (old_config, new_config) in enumerate(zip(old_configs, new_configs)):
            try:
                config_issues = self._validate_single_config_migration(old_config, new_config, i)
                issues.extend(config_issues)
                
                # Count as valid if no critical or error issues
                if not any(issue.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR] 
                          for issue in config_issues):
                    valid_records += 1
                    
            except Exception as e:
                logger.error(f"Failed to validate config {i}: {e}")
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="validation_error",
                    message=f"Config validation failed for record {i}: {e}",
                    field_path=f"config[{i}]",
                    old_value=None,
                    new_value=None,
                    suggestion="Check configuration format and migration logic"
                ))
        
        # Create summary
        summary = self._create_validation_summary(issues, "config_migration")
        
        report = ValidationReport(
            validation_id=validation_id,
            timestamp=datetime.now(),
            data_type="config_migration",
            total_records=len(old_configs),
            valid_records=valid_records,
            invalid_records=len(old_configs) - valid_records,
            issues=issues,
            summary=summary
        )
        
        logger.info(f"Config migration validation completed: {report.success_rate:.1f}% success rate")
        return report
    
    def validate_data_integrity(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> ValidationReport:
        """
        Validate data integrity for migrated data.
        
        Args:
            data: Data to validate (single dict or list of dicts)
            
        Returns:
            ValidationReport: Validation report
        """
        validation_id = f"data_integrity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        issues = []
        
        # Normalize data to list
        if isinstance(data, dict):
            data_list = [data]
        else:
            data_list = data
        
        logger.info(f"Starting data integrity validation for {len(data_list)} records")
        
        valid_records = 0
        
        for i, record in enumerate(data_list):
            try:
                record_issues = self._validate_data_integrity_single(record, i)
                issues.extend(record_issues)
                
                # Count as valid if no critical or error issues
                if not any(issue.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR] 
                          for issue in record_issues):
                    valid_records += 1
                    
            except Exception as e:
                logger.error(f"Failed to validate data integrity for record {i}: {e}")
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="validation_error",
                    message=f"Data integrity validation failed for record {i}: {e}",
                    field_path=f"record[{i}]",
                    old_value=None,
                    new_value=None,
                    suggestion="Check data format and structure"
                ))
        
        # Create summary
        summary = self._create_validation_summary(issues, "data_integrity")
        
        report = ValidationReport(
            validation_id=validation_id,
            timestamp=datetime.now(),
            data_type="data_integrity",
            total_records=len(data_list),
            valid_records=valid_records,
            invalid_records=len(data_list) - valid_records,
            issues=issues,
            summary=summary
        )
        
        logger.info(f"Data integrity validation completed: {report.success_rate:.1f}% success rate")
        return report
    
    def _validate_single_state_migration(
        self,
        old_state: Dict[str, Any],
        new_state: Any,  # Changed from VideoGenerationState to Any
        record_index: int
    ) -> List[ValidationIssue]:
        """Validate a single state migration."""
        issues = []
        
        # Core field validation
        core_fields = [
            ('topic', 'topic'),
            ('description', 'description'),
            ('session_id', 'session_id')
        ]
        
        for old_field, new_field in core_fields:
            old_value = old_state.get(old_field)
            new_value = getattr(new_state, new_field, None)
            
            if old_value != new_value:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="field_mismatch",
                    message=f"Core field {old_field} mismatch",
                    field_path=f"record[{record_index}].{old_field}",
                    old_value=old_value,
                    new_value=new_value,
                    suggestion="Check field mapping in StateAdapter"
                ))
        
        # State data validation
        state_fields = [
            ('scene_outline', 'scene_outline'),
            ('scene_implementations', 'scene_implementations'),
            ('generated_code', 'generated_code'),
            ('rendered_videos', 'rendered_videos'),
            ('workflow_complete', 'workflow_complete')
        ]
        
        for old_field, new_field in state_fields:
            old_value = old_state.get(old_field)
            new_value = getattr(new_state, new_field, None)
            
            if old_value is not None and old_value != new_value:
                severity = ValidationSeverity.WARNING if old_field in ['scene_outline'] else ValidationSeverity.ERROR
                issues.append(ValidationIssue(
                    severity=severity,
                    category="state_data_mismatch",
                    message=f"State field {old_field} mismatch",
                    field_path=f"record[{record_index}].{old_field}",
                    old_value=old_value,
                    new_value=new_value,
                    suggestion="Verify state field migration logic"
                ))
        
        # Validate new state structure
        try:
            # Test serialization/deserialization if model is available
            if hasattr(new_state, 'model_dump') and VideoGenerationState:
                serialized = new_state.model_dump()
                VideoGenerationState(**serialized)
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="serialization_error",
                message=f"New state serialization failed: {e}",
                field_path=f"record[{record_index}]",
                old_value=None,
                new_value=None,
                suggestion="Check Pydantic model definition"
            ))
        
        return issues
    
    def _validate_single_config_migration(
        self,
        old_config: Dict[str, Any],
        new_config: Any,  # Changed from WorkflowConfig to Any
        record_index: int
    ) -> List[ValidationIssue]:
        """Validate a single configuration migration."""
        issues = []
        
        # Essential configuration validation
        essential_configs = [
            ('use_rag', 'use_rag'),
            ('max_retries', 'max_retries'),
            ('output_dir', 'output_dir'),
            ('enable_caching', 'enable_caching')
        ]
        
        for old_field, new_field in essential_configs:
            old_value = old_config.get(old_field)
            new_value = getattr(new_config, new_field, None)
            
            if old_value is not None and old_value != new_value:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="config_value_change",
                    message=f"Configuration {old_field} changed during migration",
                    field_path=f"config[{record_index}].{old_field}",
                    old_value=old_value,
                    new_value=new_value,
                    suggestion="Verify this change is intentional"
                ))
        
        # Model configuration validation
        model_fields = ['planner_model', 'code_model', 'helper_model']
        for model_field in model_fields:
            old_model = old_config.get(model_field)
            new_model = getattr(new_config, model_field, None)
            
            if old_model and new_model:
                # Compare model names
                old_model_name = old_model if isinstance(old_model, str) else old_model.get('model_name', '')
                new_model_name = new_model.model_name if hasattr(new_model, 'model_name') else ''
                
                if old_model_name != new_model_name:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        category="model_config_change",
                        message=f"Model {model_field} changed during migration",
                        field_path=f"config[{record_index}].{model_field}",
                        old_value=old_model_name,
                        new_value=new_model_name,
                        suggestion="Verify model configuration is correct"
                    ))
        
        # Validate new config structure
        try:
            # Test serialization/deserialization if model is available
            if hasattr(new_config, 'model_dump') and WorkflowConfig:
                serialized = new_config.model_dump()
                WorkflowConfig(**serialized)
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="config_serialization_error",
                message=f"New config serialization failed: {e}",
                field_path=f"config[{record_index}]",
                old_value=None,
                new_value=None,
                suggestion="Check WorkflowConfig model definition"
            ))
        
        return issues
    
    def _validate_data_integrity_single(self, record: Dict[str, Any], record_index: int) -> List[ValidationIssue]:
        """Validate data integrity for a single record."""
        issues = []
        
        # Check for required fields
        required_fields = ['topic', 'description', 'session_id']
        for field in required_fields:
            if field not in record or not record[field]:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="missing_required_field",
                    message=f"Required field {field} is missing or empty",
                    field_path=f"record[{record_index}].{field}",
                    old_value=None,
                    new_value=record.get(field),
                    suggestion=f"Ensure {field} is properly set"
                ))
        
        # Check data types
        type_checks = [
            ('topic', str),
            ('description', str),
            ('session_id', str),
            ('scene_implementations', dict),
            ('generated_code', dict),
            ('rendered_videos', dict),
            ('workflow_complete', bool)
        ]
        
        for field, expected_type in type_checks:
            if field in record and record[field] is not None:
                if not isinstance(record[field], expected_type):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="type_mismatch",
                        message=f"Field {field} has incorrect type",
                        field_path=f"record[{record_index}].{field}",
                        old_value=type(record[field]).__name__,
                        new_value=expected_type.__name__,
                        suggestion=f"Convert {field} to {expected_type.__name__}"
                    ))
        
        # Check data consistency
        if 'scene_implementations' in record and 'generated_code' in record:
            scene_nums = set(record['scene_implementations'].keys()) if record['scene_implementations'] else set()
            code_nums = set(record['generated_code'].keys()) if record['generated_code'] else set()
            
            if scene_nums and code_nums and not scene_nums.issubset(code_nums.union({str(k) for k in code_nums})):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="data_consistency",
                    message="Scene implementations and generated code keys don't match",
                    field_path=f"record[{record_index}]",
                    old_value=list(scene_nums),
                    new_value=list(code_nums),
                    suggestion="Ensure scene numbers are consistent across fields"
                ))
        
        return issues
    
    def _create_validation_summary(self, issues: List[ValidationIssue], validation_type: str) -> Dict[str, Any]:
        """Create a summary of validation issues."""
        summary = {
            "validation_type": validation_type,
            "total_issues": len(issues),
            "issues_by_severity": {
                "critical": sum(1 for i in issues if i.severity == ValidationSeverity.CRITICAL),
                "error": sum(1 for i in issues if i.severity == ValidationSeverity.ERROR),
                "warning": sum(1 for i in issues if i.severity == ValidationSeverity.WARNING),
                "info": sum(1 for i in issues if i.severity == ValidationSeverity.INFO)
            },
            "issues_by_category": {}
        }
        
        # Count issues by category
        for issue in issues:
            category = issue.category
            if category not in summary["issues_by_category"]:
                summary["issues_by_category"][category] = 0
            summary["issues_by_category"][category] += 1
        
        # Add recommendations
        summary["recommendations"] = []
        
        if summary["issues_by_severity"]["critical"] > 0:
            summary["recommendations"].append("Address critical issues before proceeding with migration")
        
        if summary["issues_by_severity"]["error"] > 0:
            summary["recommendations"].append("Review and fix error-level issues")
        
        if summary["issues_by_severity"]["warning"] > 0:
            summary["recommendations"].append("Consider addressing warning-level issues")
        
        return summary
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules configuration."""
        # Default validation rules
        return {
            "required_state_fields": ["topic", "description", "session_id"],
            "required_config_fields": ["use_rag", "max_retries", "output_dir"],
            "type_mappings": {
                "topic": str,
                "description": str,
                "session_id": str,
                "use_rag": bool,
                "max_retries": int,
                "workflow_complete": bool
            },
            "consistency_checks": [
                "scene_implementations_vs_generated_code",
                "rendered_videos_vs_scene_implementations"
            ]
        }
    
    def export_validation_report(self, report: ValidationReport, output_path: str) -> None:
        """
        Export validation report to file.
        
        Args:
            report: Validation report to export
            output_path: Path to output file
        """
        try:
            report_data = {
                "validation_id": report.validation_id,
                "timestamp": report.timestamp.isoformat(),
                "data_type": report.data_type,
                "total_records": report.total_records,
                "valid_records": report.valid_records,
                "invalid_records": report.invalid_records,
                "success_rate": report.success_rate,
                "has_critical_issues": report.has_critical_issues,
                "has_errors": report.has_errors,
                "summary": report.summary,
                "issues": [
                    {
                        "severity": issue.severity.value,
                        "category": issue.category,
                        "message": issue.message,
                        "field_path": issue.field_path,
                        "old_value": issue.old_value,
                        "new_value": issue.new_value,
                        "suggestion": issue.suggestion
                    }
                    for issue in report.issues
                ]
            }
            
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            logger.info(f"Validation report exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export validation report: {e}")
            raise ValidationError(f"Report export failed: {e}") from e
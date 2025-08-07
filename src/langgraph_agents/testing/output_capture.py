"""
Agent output capture and validation system for LangGraph Studio.

This module provides comprehensive output capture, formatting, and validation
capabilities for agent testing in Studio environment.
"""

import io
import sys
import logging
import time
import json
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class OutputType(Enum):
    """Types of output that can be captured."""
    STDOUT = "stdout"
    STDERR = "stderr"
    LOGS = "logs"
    METRICS = "metrics"
    ERRORS = "errors"
    RESULTS = "results"


@dataclass
class CapturedOutput:
    """Container for captured output data."""
    
    session_id: str
    agent_name: str
    start_time: datetime
    end_time: Optional[datetime]
    stdout: List[str]
    stderr: List[str]
    logs: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    errors: List[Dict[str, Any]]
    results: Dict[str, Any]
    execution_time: Optional[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'session_id': self.session_id,
            'agent_name': self.agent_name,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'stdout': self.stdout,
            'stderr': self.stderr,
            'logs': self.logs,
            'metrics': self.metrics,
            'errors': self.errors,
            'results': self.results,
            'execution_time': self.execution_time
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the captured output."""
        return {
            'session_id': self.session_id,
            'agent_name': self.agent_name,
            'execution_time': self.execution_time,
            'stdout_lines': len(self.stdout),
            'stderr_lines': len(self.stderr),
            'log_entries': len(self.logs),
            'error_count': len(self.errors),
            'has_results': bool(self.results),
            'metrics_available': bool(self.metrics)
        }


class LogCapture(logging.Handler):
    """Custom logging handler to capture log messages."""
    
    def __init__(self):
        super().__init__()
        self.logs = []
        self.setLevel(logging.DEBUG)
        
        # Set formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.setFormatter(formatter)
    
    def emit(self, record):
        """Capture log record."""
        try:
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            
            # Add exception info if present
            if record.exc_info:
                log_entry['exception'] = self.format(record)
            
            self.logs.append(log_entry)
            
        except Exception:
            # Don't let logging errors break the capture
            pass
    
    def get_logs(self) -> List[Dict[str, Any]]:
        """Get captured logs."""
        return self.logs.copy()
    
    def clear_logs(self):
        """Clear captured logs."""
        self.logs.clear()


class AgentOutputCapture:
    """
    Comprehensive output capture system for agent testing.
    
    Captures stdout, stderr, logs, metrics, and execution results
    for visualization in LangGraph Studio.
    """
    
    def __init__(self):
        """Initialize the output capture system."""
        self.active_captures: Dict[str, CapturedOutput] = {}
        self.log_handlers: Dict[str, LogCapture] = {}
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Root logger for capturing all logs
        self.root_logger = logging.getLogger()
        self.original_log_level = self.root_logger.level
    
    def start_capture(self, session_id: str, agent_name: str = "unknown") -> None:
        """Start capturing output for a session."""
        if session_id in self.active_captures:
            logger.warning(f"Capture already active for session {session_id}")
            return
        
        # Create captured output container
        captured_output = CapturedOutput(
            session_id=session_id,
            agent_name=agent_name,
            start_time=datetime.now(),
            end_time=None,
            stdout=[],
            stderr=[],
            logs=[],
            metrics={},
            errors=[],
            results={},
            execution_time=None
        )
        
        # Create log handler
        log_handler = LogCapture()
        
        # Store references
        self.active_captures[session_id] = captured_output
        self.log_handlers[session_id] = log_handler
        
        # Add log handler to root logger
        self.root_logger.addHandler(log_handler)
        
        logger.info(f"Started output capture for session {session_id}")
    
    def stop_capture(self, session_id: str) -> Optional[CapturedOutput]:
        """Stop capturing output for a session."""
        if session_id not in self.active_captures:
            logger.warning(f"No active capture for session {session_id}")
            return None
        
        # Get captured output
        captured_output = self.active_captures[session_id]
        captured_output.end_time = datetime.now()
        
        # Calculate execution time
        if captured_output.start_time:
            execution_time = (captured_output.end_time - captured_output.start_time).total_seconds()
            captured_output.execution_time = execution_time
        
        # Get captured logs
        if session_id in self.log_handlers:
            log_handler = self.log_handlers[session_id]
            captured_output.logs = log_handler.get_logs()
            
            # Remove log handler
            self.root_logger.removeHandler(log_handler)
            del self.log_handlers[session_id]
        
        # Remove from active captures
        del self.active_captures[session_id]
        
        logger.info(f"Stopped output capture for session {session_id}")
        return captured_output
    
    @contextmanager
    def capture_streams(self, session_id: str):
        """Context manager to capture stdout and stderr."""
        if session_id not in self.active_captures:
            yield
            return
        
        captured_output = self.active_captures[session_id]
        
        # Create string buffers
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        # Redirect streams
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer
        
        try:
            yield
        finally:
            # Restore original streams
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            
            # Capture output
            stdout_content = stdout_buffer.getvalue()
            stderr_content = stderr_buffer.getvalue()
            
            if stdout_content:
                captured_output.stdout.extend(stdout_content.splitlines())
            
            if stderr_content:
                captured_output.stderr.extend(stderr_content.splitlines())
    
    def add_metrics(self, session_id: str, metrics: Dict[str, Any]) -> None:
        """Add metrics to captured output."""
        if session_id in self.active_captures:
            self.active_captures[session_id].metrics.update(metrics)
    
    def add_error(self, session_id: str, error: Dict[str, Any]) -> None:
        """Add error to captured output."""
        if session_id in self.active_captures:
            error_entry = {
                'timestamp': datetime.now().isoformat(),
                **error
            }
            self.active_captures[session_id].errors.append(error_entry)
    
    def add_results(self, session_id: str, results: Dict[str, Any]) -> None:
        """Add results to captured output."""
        if session_id in self.active_captures:
            self.active_captures[session_id].results.update(results)
    
    def add_state_tracking(self, session_id: str, step: str, state_data: Dict[str, Any]) -> None:
        """Add state tracking information for a specific execution step."""
        if session_id in self.active_captures:
            if 'state_tracking' not in self.active_captures[session_id].results:
                self.active_captures[session_id].results['state_tracking'] = []
            
            state_entry = {
                'timestamp': datetime.now().isoformat(),
                'step': step,
                'state_data': state_data
            }
            self.active_captures[session_id].results['state_tracking'].append(state_entry)
    
    def add_execution_log(self, session_id: str, level: str, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Add execution log entry with context."""
        if session_id in self.active_captures:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'level': level,
                'message': message,
                'context': context or {}
            }
            
            # Add to logs if not already captured by log handler
            if 'execution_logs' not in self.active_captures[session_id].results:
                self.active_captures[session_id].results['execution_logs'] = []
            
            self.active_captures[session_id].results['execution_logs'].append(log_entry)
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active capture sessions."""
        return list(self.active_captures.keys())
    
    def is_capturing(self, session_id: str) -> bool:
        """Check if capture is active for a session."""
        return session_id in self.active_captures


class OutputFormatter:
    """
    Formats captured output for Studio visualization.
    
    Provides different formatting options for different visualization needs.
    """
    
    def __init__(self):
        """Initialize the output formatter."""
        pass
    
    def format_for_studio(self, captured_output: CapturedOutput) -> Dict[str, Any]:
        """Format output for LangGraph Studio visualization."""
        return {
            'session_info': {
                'session_id': captured_output.session_id,
                'agent_name': captured_output.agent_name,
                'execution_time': captured_output.execution_time,
                'start_time': captured_output.start_time.isoformat() if captured_output.start_time else None,
                'end_time': captured_output.end_time.isoformat() if captured_output.end_time else None
            },
            'console_output': {
                'stdout': captured_output.stdout,
                'stderr': captured_output.stderr,
                'stdout_summary': f"{len(captured_output.stdout)} lines",
                'stderr_summary': f"{len(captured_output.stderr)} lines"
            },
            'logs': {
                'entries': captured_output.logs,
                'summary': self._summarize_logs(captured_output.logs),
                'by_level': self._group_logs_by_level(captured_output.logs)
            },
            'metrics': {
                'data': captured_output.metrics,
                'summary': self._summarize_metrics(captured_output.metrics)
            },
            'errors': {
                'entries': captured_output.errors,
                'summary': self._summarize_errors(captured_output.errors)
            },
            'results': {
                'data': captured_output.results,
                'summary': self._summarize_results(captured_output.results)
            }
        }
    
    def format_for_json(self, captured_output: CapturedOutput) -> str:
        """Format output as JSON string."""
        return json.dumps(captured_output.to_dict(), indent=2, default=str)
    
    def format_for_text(self, captured_output: CapturedOutput) -> str:
        """Format output as human-readable text."""
        lines = []
        
        # Header
        lines.append(f"=== Agent Output Capture: {captured_output.agent_name} ===")
        lines.append(f"Session ID: {captured_output.session_id}")
        lines.append(f"Execution Time: {captured_output.execution_time:.2f}s" if captured_output.execution_time else "Execution Time: N/A")
        lines.append("")
        
        # Console Output
        if captured_output.stdout:
            lines.append("--- STDOUT ---")
            lines.extend(captured_output.stdout)
            lines.append("")
        
        if captured_output.stderr:
            lines.append("--- STDERR ---")
            lines.extend(captured_output.stderr)
            lines.append("")
        
        # Logs
        if captured_output.logs:
            lines.append("--- LOGS ---")
            for log_entry in captured_output.logs[-10:]:  # Last 10 log entries
                lines.append(f"[{log_entry['level']}] {log_entry['timestamp']} - {log_entry['message']}")
            lines.append("")
        
        # Errors
        if captured_output.errors:
            lines.append("--- ERRORS ---")
            for error in captured_output.errors:
                lines.append(f"Error: {error.get('message', 'Unknown error')}")
            lines.append("")
        
        # Results Summary
        if captured_output.results:
            lines.append("--- RESULTS SUMMARY ---")
            for key, value in captured_output.results.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{key}: {type(value).__name__} with {len(value)} items")
                else:
                    lines.append(f"{key}: {str(value)[:100]}...")
            lines.append("")
        
        return "\n".join(lines)
    
    def _summarize_logs(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize log entries."""
        if not logs:
            return {'total': 0, 'by_level': {}}
        
        by_level = {}
        for log_entry in logs:
            level = log_entry.get('level', 'UNKNOWN')
            by_level[level] = by_level.get(level, 0) + 1
        
        return {
            'total': len(logs),
            'by_level': by_level,
            'time_range': {
                'start': logs[0].get('timestamp') if logs else None,
                'end': logs[-1].get('timestamp') if logs else None
            }
        }
    
    def _group_logs_by_level(self, logs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group logs by level."""
        grouped = {}
        for log_entry in logs:
            level = log_entry.get('level', 'UNKNOWN')
            if level not in grouped:
                grouped[level] = []
            grouped[level].append(log_entry)
        
        return grouped
    
    def _summarize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize metrics data."""
        if not metrics:
            return {'total_metrics': 0, 'categories': []}
        
        categories = list(metrics.keys())
        numeric_metrics = {}
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                numeric_metrics[key] = value
        
        return {
            'total_metrics': len(metrics),
            'categories': categories,
            'numeric_metrics': numeric_metrics,
            'has_performance_data': any('time' in key.lower() or 'duration' in key.lower() for key in metrics.keys())
        }
    
    def _summarize_errors(self, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize error data."""
        if not errors:
            return {'total_errors': 0, 'by_type': {}}
        
        by_type = {}
        for error in errors:
            error_type = error.get('error_type', 'Unknown')
            by_type[error_type] = by_type.get(error_type, 0) + 1
        
        return {
            'total_errors': len(errors),
            'by_type': by_type,
            'latest_error': errors[-1] if errors else None
        }
    
    def _summarize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize results data."""
        if not results:
            return {'total_results': 0, 'result_types': []}
        
        result_types = []
        for key, value in results.items():
            result_types.append({
                'key': key,
                'type': type(value).__name__,
                'size': len(value) if hasattr(value, '__len__') else 1
            })
        
        return {
            'total_results': len(results),
            'result_types': result_types,
            'has_outputs': any('output' in key.lower() for key in results.keys())
        }


class OutputValidator:
    """
    Validates captured output against expected patterns and criteria.
    
    Provides validation rules for different types of agent outputs.
    """
    
    def __init__(self):
        """Initialize the output validator."""
        self.validation_rules = {
            'PlannerAgent': self._validate_planner_output,
            'CodeGeneratorAgent': self._validate_codegen_output,
            'RendererAgent': self._validate_renderer_output,
            'ErrorHandlerAgent': self._validate_errorhandler_output,
            'HumanLoopAgent': self._validate_humanloop_output
        }
    
    def validate_output(self, captured_output: CapturedOutput) -> Dict[str, Any]:
        """Validate captured output based on agent type."""
        agent_name = captured_output.agent_name
        
        # Get appropriate validation function
        validation_func = self.validation_rules.get(agent_name, self._validate_generic_output)
        
        # Run validation
        validation_result = validation_func(captured_output)
        
        # Add common validations
        common_validations = self._validate_common_output(captured_output)
        validation_result.update(common_validations)
        
        return validation_result
    
    def _validate_common_output(self, captured_output: CapturedOutput) -> Dict[str, Any]:
        """Validate common output patterns."""
        validations = {}
        
        # Check execution time
        if captured_output.execution_time is not None:
            validations['execution_time_reasonable'] = captured_output.execution_time < 300  # 5 minutes max
            validations['execution_time_value'] = captured_output.execution_time
        else:
            validations['execution_time_reasonable'] = False
            validations['execution_time_value'] = None
        
        # Check for critical errors
        critical_errors = [
            error for error in captured_output.errors
            if error.get('level') == 'CRITICAL' or 'critical' in error.get('message', '').lower()
        ]
        validations['no_critical_errors'] = len(critical_errors) == 0
        validations['critical_error_count'] = len(critical_errors)
        
        # Check log quality
        error_logs = [
            log for log in captured_output.logs
            if log.get('level') == 'ERROR'
        ]
        validations['low_error_rate'] = len(error_logs) < len(captured_output.logs) * 0.1  # Less than 10% errors
        validations['error_log_count'] = len(error_logs)
        
        return validations
    
    def _validate_planner_output(self, captured_output: CapturedOutput) -> Dict[str, Any]:
        """Validate PlannerAgent specific output."""
        validations = {}
        results = captured_output.results
        
        # Check for required outputs
        validations['has_scene_outline'] = 'scene_outline' in results.get('outputs', {})
        validations['has_scene_implementations'] = 'scene_implementations' in results.get('outputs', {})
        validations['has_detected_plugins'] = 'detected_plugins' in results.get('outputs', {})
        
        # Check scene count
        scene_count = results.get('outputs', {}).get('scene_count', 0)
        validations['reasonable_scene_count'] = 1 <= scene_count <= 10
        validations['scene_count'] = scene_count
        
        # Check validation results
        validation_data = results.get('validation', {})
        validations['scene_outline_valid'] = validation_data.get('scene_outline_valid', False)
        validations['scene_implementations_valid'] = validation_data.get('scene_implementations_valid', False)
        
        return validations
    
    def _validate_codegen_output(self, captured_output: CapturedOutput) -> Dict[str, Any]:
        """Validate CodeGeneratorAgent specific output."""
        validations = {}
        results = captured_output.results
        
        # Check for required outputs
        outputs = results.get('outputs', {})
        validations['has_generated_code'] = 'generated_code' in outputs
        validations['has_successful_scenes'] = 'successful_scenes' in outputs
        validations['has_failed_scenes'] = 'failed_scenes' in outputs
        
        # Check success rate
        total_scenes = outputs.get('total_scenes', 0)
        successful_scenes = len(outputs.get('successful_scenes', []))
        
        if total_scenes > 0:
            success_rate = successful_scenes / total_scenes
            validations['high_success_rate'] = success_rate >= 0.8
            validations['success_rate'] = success_rate
        else:
            validations['high_success_rate'] = False
            validations['success_rate'] = 0.0
        
        # Check code validation
        validation_data = results.get('validation', {})
        valid_scenes = sum(1 for v in validation_data.values() if v.get('valid', False))
        validations['code_validation_rate'] = valid_scenes / max(total_scenes, 1)
        
        return validations
    
    def _validate_renderer_output(self, captured_output: CapturedOutput) -> Dict[str, Any]:
        """Validate RendererAgent specific output."""
        validations = {}
        results = captured_output.results
        
        # Check for required outputs
        outputs = results.get('outputs', {})
        validations['has_rendered_videos'] = 'rendered_videos' in outputs
        validations['has_final_codes'] = 'final_codes' in outputs
        
        # Check rendering success rate
        total_scenes = outputs.get('total_scenes', 0)
        successful_scenes = len(outputs.get('successful_scenes', []))
        
        if total_scenes > 0:
            success_rate = successful_scenes / total_scenes
            validations['high_rendering_success_rate'] = success_rate >= 0.7  # Lower threshold for rendering
            validations['rendering_success_rate'] = success_rate
        else:
            validations['high_rendering_success_rate'] = False
            validations['rendering_success_rate'] = 0.0
        
        # Check for combined video
        validations['has_combined_video'] = outputs.get('combined_video_path') is not None
        
        return validations
    
    def _validate_errorhandler_output(self, captured_output: CapturedOutput) -> Dict[str, Any]:
        """Validate ErrorHandlerAgent specific output."""
        validations = {}
        results = captured_output.results
        
        # Check for required outputs
        outputs = results.get('outputs', {})
        validations['has_recovery_results'] = 'recovery_results' in outputs
        validations['has_recovery_statistics'] = 'recovery_statistics' in results
        
        # Check recovery success rate
        success_rate = outputs.get('success_rate', 0.0)
        validations['good_recovery_rate'] = success_rate >= 0.6
        validations['recovery_success_rate'] = success_rate
        
        # Check that scenarios were processed
        total_scenarios = outputs.get('total_scenarios', 0)
        validations['scenarios_processed'] = total_scenarios > 0
        validations['total_scenarios'] = total_scenarios
        
        return validations
    
    def _validate_humanloop_output(self, captured_output: CapturedOutput) -> Dict[str, Any]:
        """Validate HumanLoopAgent specific output."""
        validations = {}
        results = captured_output.results
        
        # Check for required outputs
        outputs = results.get('outputs', {})
        validations['has_intervention_results'] = 'intervention_results' in outputs
        
        # Check intervention success rate
        success_rate = outputs.get('success_rate', 0.0)
        validations['good_intervention_rate'] = success_rate >= 0.8
        validations['intervention_success_rate'] = success_rate
        
        # Check that scenarios were processed
        total_scenarios = outputs.get('total_scenarios', 0)
        validations['scenarios_processed'] = total_scenarios > 0
        validations['total_scenarios'] = total_scenarios
        
        return validations
    
    def _validate_generic_output(self, captured_output: CapturedOutput) -> Dict[str, Any]:
        """Validate generic agent output."""
        validations = {}
        
        # Basic output checks
        validations['has_results'] = bool(captured_output.results)
        validations['has_metrics'] = bool(captured_output.metrics)
        validations['completed_successfully'] = captured_output.results.get('success', False)
        
        return validations


# Global output capture instance for easy access
_global_output_capture = AgentOutputCapture()
_global_output_formatter = OutputFormatter()
_global_output_validator = OutputValidator()


def get_output_capture() -> AgentOutputCapture:
    """Get the global output capture instance."""
    return _global_output_capture


def get_output_formatter() -> OutputFormatter:
    """Get the global output formatter instance."""
    return _global_output_formatter


def get_output_validator() -> OutputValidator:
    """Get the global output validator instance."""
    return _global_output_validator
"""
Input Validation and Error Handling

This module provides comprehensive input validation and error handling functionality
for the Gradio testing interface, including client-side validation and graceful
error recovery.
"""

import re
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    
    def __init__(self, message: str, field: str = None, suggestions: List[str] = None):
        super().__init__(message)
        self.field = field
        self.suggestions = suggestions or []


class InputValidator:
    """Comprehensive input validation for test configurations."""
    
    @staticmethod
    def validate_agent_name(agent_name: str) -> Tuple[bool, str]:
        """
        Validate agent name.
        
        Args:
            agent_name: Agent name to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not agent_name:
            return False, "Agent name is required"
        
        if not isinstance(agent_name, str):
            return False, "Agent name must be a string"
        
        if len(agent_name.strip()) == 0:
            return False, "Agent name cannot be empty"
        
        # Check for valid agent name pattern
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', agent_name):
            return False, "Agent name must start with a letter and contain only letters, numbers, and underscores"
        
        return True, ""
    
    @staticmethod
    def validate_session_id(session_id: str) -> Tuple[bool, str]:
        """
        Validate session ID format.
        
        Args:
            session_id: Session ID to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not session_id:
            return True, ""  # Optional field
        
        if not isinstance(session_id, str):
            return False, "Session ID must be a string"
        
        # Check for valid UUID-like format or custom format
        if not re.match(r'^[a-zA-Z0-9\-_]{8,}$', session_id):
            return False, "Session ID must be at least 8 characters and contain only letters, numbers, hyphens, and underscores"
        
        return True, ""
    
    @staticmethod
    def validate_timeout(timeout: Union[int, float]) -> Tuple[bool, str]:
        """
        Validate timeout value.
        
        Args:
            timeout: Timeout value to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            timeout_val = float(timeout)
        except (ValueError, TypeError):
            return False, "Timeout must be a number"
        
        if timeout_val <= 0:
            return False, "Timeout must be greater than 0"
        
        if timeout_val > 3600:  # 1 hour max
            return False, "Timeout cannot exceed 3600 seconds (1 hour)"
        
        return True, ""
    
    @staticmethod
    def validate_retry_count(retry_count: Union[int, str]) -> Tuple[bool, str]:
        """
        Validate retry count.
        
        Args:
            retry_count: Retry count to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            retry_val = int(retry_count)
        except (ValueError, TypeError):
            return False, "Retry count must be an integer"
        
        if retry_val < 0:
            return False, "Retry count cannot be negative"
        
        if retry_val > 10:
            return False, "Retry count cannot exceed 10"
        
        return True, ""
    
    @staticmethod
    def validate_topic(topic: str) -> Tuple[bool, str]:
        """
        Validate workflow topic.
        
        Args:
            topic: Topic to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not topic:
            return False, "Topic is required"
        
        if not isinstance(topic, str):
            return False, "Topic must be a string"
        
        topic = topic.strip()
        if len(topic) == 0:
            return False, "Topic cannot be empty"
        
        if len(topic) < 3:
            return False, "Topic must be at least 3 characters long"
        
        if len(topic) > 200:
            return False, "Topic cannot exceed 200 characters"
        
        return True, ""
    
    @staticmethod
    def validate_description(description: str, required: bool = True) -> Tuple[bool, str]:
        """
        Validate description field.
        
        Args:
            description: Description to validate
            required: Whether the field is required
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not description:
            if required:
                return False, "Description is required"
            return True, ""
        
        if not isinstance(description, str):
            return False, "Description must be a string"
        
        description = description.strip()
        if required and len(description) == 0:
            return False, "Description cannot be empty"
        
        if len(description) > 2000:
            return False, "Description cannot exceed 2000 characters"
        
        return True, ""
    
    @staticmethod
    def validate_config_name(name: str) -> Tuple[bool, str]:
        """
        Validate configuration name.
        
        Args:
            name: Configuration name to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not name:
            return False, "Configuration name is required"
        
        if not isinstance(name, str):
            return False, "Configuration name must be a string"
        
        name = name.strip()
        if len(name) == 0:
            return False, "Configuration name cannot be empty"
        
        if len(name) < 2:
            return False, "Configuration name must be at least 2 characters long"
        
        if len(name) > 100:
            return False, "Configuration name cannot exceed 100 characters"
        
        # Check for valid characters
        if not re.match(r'^[a-zA-Z0-9\s\-_\.]+$', name):
            return False, "Configuration name can only contain letters, numbers, spaces, hyphens, underscores, and periods"
        
        return True, ""
    
    @staticmethod
    def validate_json_input(json_str: str, required: bool = False) -> Tuple[bool, str, Optional[Dict]]:
        """
        Validate JSON input string.
        
        Args:
            json_str: JSON string to validate
            required: Whether the field is required
            
        Returns:
            Tuple of (is_valid, error_message, parsed_json)
        """
        if not json_str:
            if required:
                return False, "JSON input is required", None
            return True, "", {}
        
        if not isinstance(json_str, str):
            return False, "JSON input must be a string", None
        
        json_str = json_str.strip()
        if not json_str:
            return True, "", {}
        
        try:
            parsed_json = json.loads(json_str)
            if not isinstance(parsed_json, dict):
                return False, "JSON input must be an object (dictionary)", None
            return True, "", parsed_json
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON format: {str(e)}", None
    
    @staticmethod
    def validate_agent_inputs(inputs: Dict[str, Any], agent_schema: Dict[str, Any] = None) -> Tuple[bool, List[str]]:
        """
        Validate agent input parameters against schema.
        
        Args:
            inputs: Input parameters to validate
            agent_schema: Optional agent schema for validation
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not isinstance(inputs, dict):
            errors.append("Agent inputs must be a dictionary")
            return False, errors
        
        # If no schema provided, do basic validation
        if not agent_schema:
            for key, value in inputs.items():
                if not isinstance(key, str):
                    errors.append(f"Input key must be a string: {key}")
                if value is None:
                    errors.append(f"Input value cannot be None: {key}")
            
            return len(errors) == 0, errors
        
        # Validate against schema if provided
        required_fields = agent_schema.get("required", [])
        properties = agent_schema.get("properties", {})
        
        # Check required fields
        for field in required_fields:
            if field not in inputs:
                errors.append(f"Required field missing: {field}")
        
        # Validate each input
        for key, value in inputs.items():
            if key in properties:
                field_schema = properties[key]
                field_type = field_schema.get("type")
                
                # Type validation
                if field_type == "string" and not isinstance(value, str):
                    errors.append(f"Field '{key}' must be a string")
                elif field_type == "number" and not isinstance(value, (int, float)):
                    errors.append(f"Field '{key}' must be a number")
                elif field_type == "integer" and not isinstance(value, int):
                    errors.append(f"Field '{key}' must be an integer")
                elif field_type == "boolean" and not isinstance(value, bool):
                    errors.append(f"Field '{key}' must be a boolean")
                
                # String length validation
                if field_type == "string" and isinstance(value, str):
                    min_length = field_schema.get("minLength")
                    max_length = field_schema.get("maxLength")
                    
                    if min_length and len(value) < min_length:
                        errors.append(f"Field '{key}' must be at least {min_length} characters")
                    if max_length and len(value) > max_length:
                        errors.append(f"Field '{key}' cannot exceed {max_length} characters")
                
                # Number range validation
                if field_type in ["number", "integer"] and isinstance(value, (int, float)):
                    minimum = field_schema.get("minimum")
                    maximum = field_schema.get("maximum")
                    
                    if minimum is not None and value < minimum:
                        errors.append(f"Field '{key}' must be at least {minimum}")
                    if maximum is not None and value > maximum:
                        errors.append(f"Field '{key}' cannot exceed {maximum}")
        
        return len(errors) == 0, errors


class ErrorHandler:
    """Handles errors and provides user-friendly error messages and recovery suggestions."""
    
    @staticmethod
    def format_validation_error(error: ValidationError) -> str:
        """
        Format validation error for display.
        
        Args:
            error: ValidationError to format
            
        Returns:
            Formatted error message
        """
        message = f"❌ {error}"
        
        if error.field:
            message = f"❌ {error.field}: {error}"
        
        if error.suggestions:
            message += "\n\n💡 Suggestions:"
            for suggestion in error.suggestions:
                message += f"\n  • {suggestion}"
        
        return message
    
    @staticmethod
    def handle_api_error(error: Exception) -> Tuple[str, List[str]]:
        """
        Handle API communication errors and provide recovery suggestions.
        
        Args:
            error: Exception that occurred
            
        Returns:
            Tuple of (error_message, recovery_suggestions)
        """
        error_str = str(error).lower()
        
        # Connection errors
        if "connection" in error_str or "timeout" in error_str:
            return (
                "❌ Connection Error: Unable to connect to the backend server",
                [
                    "Check if the backend server is running",
                    "Verify the backend URL is correct",
                    "Check your network connection",
                    "Try refreshing the page"
                ]
            )
        
        # Authentication errors
        if "unauthorized" in error_str or "403" in error_str:
            return (
                "❌ Authentication Error: Access denied",
                [
                    "Check if authentication is required",
                    "Verify your credentials",
                    "Contact the system administrator"
                ]
            )
        
        # Server errors
        if "500" in error_str or "internal server error" in error_str:
            return (
                "❌ Server Error: The backend server encountered an error",
                [
                    "Try again in a few moments",
                    "Check the server logs for details",
                    "Contact the system administrator if the problem persists"
                ]
            )
        
        # Not found errors
        if "404" in error_str or "not found" in error_str:
            return (
                "❌ Not Found: The requested resource was not found",
                [
                    "Check if the agent or endpoint exists",
                    "Refresh the agent list",
                    "Verify the backend configuration"
                ]
            )
        
        # JSON/parsing errors
        if "json" in error_str or "parsing" in error_str:
            return (
                "❌ Data Format Error: Invalid response format",
                [
                    "Check the backend server logs",
                    "Verify the API is returning valid JSON",
                    "Try refreshing the page"
                ]
            )
        
        # Generic error
        return (
            f"❌ Error: {error}",
            [
                "Try refreshing the page",
                "Check the browser console for details",
                "Contact support if the problem persists"
            ]
        )
    
    @staticmethod
    def create_error_display(
        error_message: str, 
        suggestions: List[str] = None,
        show_details: bool = False,
        error_details: str = None
    ) -> str:
        """
        Create a formatted error display for the UI.
        
        Args:
            error_message: Main error message
            suggestions: List of recovery suggestions
            show_details: Whether to show detailed error information
            error_details: Detailed error information
            
        Returns:
            Formatted error display string
        """
        display = f"**{error_message}**\n\n"
        
        if suggestions:
            display += "**💡 What you can try:**\n"
            for i, suggestion in enumerate(suggestions, 1):
                display += f"{i}. {suggestion}\n"
            display += "\n"
        
        if show_details and error_details:
            display += f"**🔍 Technical Details:**\n```\n{error_details}\n```\n"
        
        display += f"*Error occurred at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        return display
    
    @staticmethod
    def validate_form_data(form_data: Dict[str, Any], validation_rules: Dict[str, Any]) -> Tuple[bool, Dict[str, str]]:
        """
        Validate form data against validation rules.
        
        Args:
            form_data: Form data to validate
            validation_rules: Validation rules dictionary
            
        Returns:
            Tuple of (is_valid, field_errors)
        """
        errors = {}
        
        for field, rules in validation_rules.items():
            value = form_data.get(field)
            
            # Required field check
            if rules.get("required", False) and not value:
                errors[field] = f"{field.replace('_', ' ').title()} is required"
                continue
            
            # Skip validation if field is empty and not required
            if not value and not rules.get("required", False):
                continue
            
            # Type validation
            expected_type = rules.get("type")
            if expected_type and not isinstance(value, expected_type):
                errors[field] = f"{field.replace('_', ' ').title()} must be of type {expected_type.__name__}"
                continue
            
            # String validations
            if isinstance(value, str):
                min_length = rules.get("min_length")
                max_length = rules.get("max_length")
                pattern = rules.get("pattern")
                
                if min_length and len(value) < min_length:
                    errors[field] = f"{field.replace('_', ' ').title()} must be at least {min_length} characters"
                elif max_length and len(value) > max_length:
                    errors[field] = f"{field.replace('_', ' ').title()} cannot exceed {max_length} characters"
                elif pattern and not re.match(pattern, value):
                    errors[field] = f"{field.replace('_', ' ').title()} format is invalid"
            
            # Number validations
            if isinstance(value, (int, float)):
                min_value = rules.get("min_value")
                max_value = rules.get("max_value")
                
                if min_value is not None and value < min_value:
                    errors[field] = f"{field.replace('_', ' ').title()} must be at least {min_value}"
                elif max_value is not None and value > max_value:
                    errors[field] = f"{field.replace('_', ' ').title()} cannot exceed {max_value}"
            
            # Custom validator
            custom_validator = rules.get("validator")
            if custom_validator and callable(custom_validator):
                is_valid, error_msg = custom_validator(value)
                if not is_valid:
                    errors[field] = error_msg
        
        return len(errors) == 0, errors


# Predefined validation rules for common form fields
AGENT_FORM_VALIDATION_RULES = {
    "agent_name": {
        "required": True,
        "type": str,
        "min_length": 1,
        "max_length": 100,
        "pattern": r'^[a-zA-Z][a-zA-Z0-9_]*$'
    },
    "session_id": {
        "required": False,
        "type": str,
        "min_length": 8,
        "max_length": 100,
        "pattern": r'^[a-zA-Z0-9\-_]+$'
    },
    "timeout": {
        "required": True,
        "type": (int, float),
        "min_value": 1,
        "max_value": 3600
    },
    "retry_count": {
        "required": True,
        "type": int,
        "min_value": 0,
        "max_value": 10
    }
}

WORKFLOW_FORM_VALIDATION_RULES = {
    "topic": {
        "required": True,
        "type": str,
        "min_length": 3,
        "max_length": 200
    },
    "description": {
        "required": True,
        "type": str,
        "min_length": 10,
        "max_length": 2000
    },
    "session_id": {
        "required": False,
        "type": str,
        "min_length": 8,
        "max_length": 100,
        "pattern": r'^[a-zA-Z0-9\-_]+$'
    }
}

CONFIG_FORM_VALIDATION_RULES = {
    "config_name": {
        "required": True,
        "type": str,
        "min_length": 2,
        "max_length": 100,
        "pattern": r'^[a-zA-Z0-9\s\-_\.]+$'
    },
    "config_description": {
        "required": False,
        "type": str,
        "max_length": 500
    }
}
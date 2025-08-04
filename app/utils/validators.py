"""
Validation utilities for the application.
"""

import re
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
from pathlib import Path
import mimetypes

from .exceptions import ValidationError


class ValidationResult:
    """Result of a validation operation."""
    
    def __init__(self, is_valid: bool, errors: Optional[List[str]] = None):
        self.is_valid = is_valid
        self.errors = errors or []
    
    def add_error(self, error: str) -> None:
        """Add an error to the result."""
        self.errors.append(error)
        self.is_valid = False
    
    def __bool__(self) -> bool:
        """Return validation status."""
        return self.is_valid


class Validator:
    """Base validator class."""
    
    def __init__(self, field_name: str = "field"):
        self.field_name = field_name
    
    def validate(self, value: Any) -> ValidationResult:
        """Validate a value."""
        raise NotImplementedError
    
    def __call__(self, value: Any) -> ValidationResult:
        """Make validator callable."""
        return self.validate(value)


class RequiredValidator(Validator):
    """Validator for required fields."""
    
    def validate(self, value: Any) -> ValidationResult:
        """Check if value is not None and not empty."""
        if value is None:
            return ValidationResult(False, [f"{self.field_name} is required"])
        
        if isinstance(value, (str, list, dict)) and len(value) == 0:
            return ValidationResult(False, [f"{self.field_name} cannot be empty"])
        
        return ValidationResult(True)


class StringValidator(Validator):
    """Validator for string fields."""
    
    def __init__(
        self,
        field_name: str = "field",
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        allowed_values: Optional[List[str]] = None
    ):
        super().__init__(field_name)
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = re.compile(pattern) if pattern else None
        self.allowed_values = allowed_values
    
    def validate(self, value: Any) -> ValidationResult:
        """Validate string value."""
        result = ValidationResult(True)
        
        if not isinstance(value, str):
            result.add_error(f"{self.field_name} must be a string")
            return result
        
        if self.min_length is not None and len(value) < self.min_length:
            result.add_error(f"{self.field_name} must be at least {self.min_length} characters long")
        
        if self.max_length is not None and len(value) > self.max_length:
            result.add_error(f"{self.field_name} must be at most {self.max_length} characters long")
        
        if self.pattern and not self.pattern.match(value):
            result.add_error(f"{self.field_name} format is invalid")
        
        if self.allowed_values and value not in self.allowed_values:
            result.add_error(f"{self.field_name} must be one of: {', '.join(self.allowed_values)}")
        
        return result


class NumberValidator(Validator):
    """Validator for numeric fields."""
    
    def __init__(
        self,
        field_name: str = "field",
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        integer_only: bool = False
    ):
        super().__init__(field_name)
        self.min_value = min_value
        self.max_value = max_value
        self.integer_only = integer_only
    
    def validate(self, value: Any) -> ValidationResult:
        """Validate numeric value."""
        result = ValidationResult(True)
        
        if not isinstance(value, (int, float)):
            result.add_error(f"{self.field_name} must be a number")
            return result
        
        if self.integer_only and not isinstance(value, int):
            result.add_error(f"{self.field_name} must be an integer")
        
        if self.min_value is not None and value < self.min_value:
            result.add_error(f"{self.field_name} must be at least {self.min_value}")
        
        if self.max_value is not None and value > self.max_value:
            result.add_error(f"{self.field_name} must be at most {self.max_value}")
        
        return result


class EmailValidator(Validator):
    """Validator for email addresses."""
    
    EMAIL_PATTERN = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    
    def validate(self, value: Any) -> ValidationResult:
        """Validate email address."""
        result = ValidationResult(True)
        
        if not isinstance(value, str):
            result.add_error(f"{self.field_name} must be a string")
            return result
        
        if not self.EMAIL_PATTERN.match(value):
            result.add_error(f"{self.field_name} must be a valid email address")
        
        return result


class URLValidator(Validator):
    """Validator for URLs."""
    
    URL_PATTERN = re.compile(
        r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$'
    )
    
    def validate(self, value: Any) -> ValidationResult:
        """Validate URL."""
        result = ValidationResult(True)
        
        if not isinstance(value, str):
            result.add_error(f"{self.field_name} must be a string")
            return result
        
        if not self.URL_PATTERN.match(value):
            result.add_error(f"{self.field_name} must be a valid URL")
        
        return result


class UUIDValidator(Validator):
    """Validator for UUID strings."""
    
    UUID_PATTERN = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
        re.IGNORECASE
    )
    
    def validate(self, value: Any) -> ValidationResult:
        """Validate UUID."""
        result = ValidationResult(True)
        
        if not isinstance(value, str):
            result.add_error(f"{self.field_name} must be a string")
            return result
        
        if not self.UUID_PATTERN.match(value):
            result.add_error(f"{self.field_name} must be a valid UUID")
        
        return result


class DateTimeValidator(Validator):
    """Validator for datetime objects."""
    
    def __init__(
        self,
        field_name: str = "field",
        min_date: Optional[datetime] = None,
        max_date: Optional[datetime] = None
    ):
        super().__init__(field_name)
        self.min_date = min_date
        self.max_date = max_date
    
    def validate(self, value: Any) -> ValidationResult:
        """Validate datetime value."""
        result = ValidationResult(True)
        
        if not isinstance(value, datetime):
            result.add_error(f"{self.field_name} must be a datetime object")
            return result
        
        if self.min_date and value < self.min_date:
            result.add_error(f"{self.field_name} must be after {self.min_date}")
        
        if self.max_date and value > self.max_date:
            result.add_error(f"{self.field_name} must be before {self.max_date}")
        
        return result


class FileValidator(Validator):
    """Validator for file uploads."""
    
    def __init__(
        self,
        field_name: str = "field",
        allowed_extensions: Optional[List[str]] = None,
        allowed_mime_types: Optional[List[str]] = None,
        max_size: Optional[int] = None  # in bytes
    ):
        super().__init__(field_name)
        self.allowed_extensions = [ext.lower() for ext in (allowed_extensions or [])]
        self.allowed_mime_types = allowed_mime_types or []
        self.max_size = max_size
    
    def validate(self, value: Any) -> ValidationResult:
        """Validate file."""
        result = ValidationResult(True)
        
        # Assuming value is a file-like object with name and size attributes
        if not hasattr(value, 'filename'):
            result.add_error(f"{self.field_name} must be a valid file")
            return result
        
        filename = getattr(value, 'filename', '')
        if not filename:
            result.add_error(f"{self.field_name} must have a filename")
            return result
        
        # Check file extension
        if self.allowed_extensions:
            file_ext = Path(filename).suffix.lower()
            if file_ext not in self.allowed_extensions:
                result.add_error(
                    f"{self.field_name} must have one of these extensions: {', '.join(self.allowed_extensions)}"
                )
        
        # Check MIME type
        if self.allowed_mime_types:
            mime_type, _ = mimetypes.guess_type(filename)
            if mime_type not in self.allowed_mime_types:
                result.add_error(
                    f"{self.field_name} must be one of these types: {', '.join(self.allowed_mime_types)}"
                )
        
        # Check file size
        if self.max_size and hasattr(value, 'size'):
            if value.size > self.max_size:
                result.add_error(f"{self.field_name} size must not exceed {self.max_size} bytes")
        
        return result


class ListValidator(Validator):
    """Validator for list fields."""
    
    def __init__(
        self,
        field_name: str = "field",
        min_items: Optional[int] = None,
        max_items: Optional[int] = None,
        item_validator: Optional[Validator] = None
    ):
        super().__init__(field_name)
        self.min_items = min_items
        self.max_items = max_items
        self.item_validator = item_validator
    
    def validate(self, value: Any) -> ValidationResult:
        """Validate list value."""
        result = ValidationResult(True)
        
        if not isinstance(value, list):
            result.add_error(f"{self.field_name} must be a list")
            return result
        
        if self.min_items is not None and len(value) < self.min_items:
            result.add_error(f"{self.field_name} must have at least {self.min_items} items")
        
        if self.max_items is not None and len(value) > self.max_items:
            result.add_error(f"{self.field_name} must have at most {self.max_items} items")
        
        # Validate individual items
        if self.item_validator:
            for i, item in enumerate(value):
                item_result = self.item_validator.validate(item)
                if not item_result.is_valid:
                    for error in item_result.errors:
                        result.add_error(f"{self.field_name}[{i}]: {error}")
        
        return result


class DictValidator(Validator):
    """Validator for dictionary fields."""
    
    def __init__(
        self,
        field_name: str = "field",
        required_keys: Optional[List[str]] = None,
        optional_keys: Optional[List[str]] = None,
        key_validators: Optional[Dict[str, Validator]] = None
    ):
        super().__init__(field_name)
        self.required_keys = required_keys or []
        self.optional_keys = optional_keys or []
        self.key_validators = key_validators or {}
    
    def validate(self, value: Any) -> ValidationResult:
        """Validate dictionary value."""
        result = ValidationResult(True)
        
        if not isinstance(value, dict):
            result.add_error(f"{self.field_name} must be a dictionary")
            return result
        
        # Check required keys
        for key in self.required_keys:
            if key not in value:
                result.add_error(f"{self.field_name} must contain key '{key}'")
        
        # Check for unexpected keys
        allowed_keys = set(self.required_keys + self.optional_keys)
        if allowed_keys:
            for key in value.keys():
                if key not in allowed_keys:
                    result.add_error(f"{self.field_name} contains unexpected key '{key}'")
        
        # Validate individual values
        for key, validator in self.key_validators.items():
            if key in value:
                key_result = validator.validate(value[key])
                if not key_result.is_valid:
                    for error in key_result.errors:
                        result.add_error(f"{self.field_name}.{key}: {error}")
        
        return result


def validate_data(data: Dict[str, Any], validators: Dict[str, List[Validator]]) -> ValidationResult:
    """
    Validate data against multiple validators.
    
    Args:
        data: Data to validate
        validators: Dictionary mapping field names to lists of validators
        
    Returns:
        ValidationResult with all validation errors
    """
    result = ValidationResult(True)
    
    for field_name, field_validators in validators.items():
        field_value = data.get(field_name)
        
        for validator in field_validators:
            validator.field_name = field_name  # Set field name for error messages
            field_result = validator.validate(field_value)
            
            if not field_result.is_valid:
                result.errors.extend(field_result.errors)
                result.is_valid = False
    
    return result


def validate_and_raise(data: Dict[str, Any], validators: Dict[str, List[Validator]]) -> None:
    """
    Validate data and raise ValidationError if invalid.
    
    Args:
        data: Data to validate
        validators: Dictionary mapping field names to lists of validators
        
    Raises:
        ValidationError: If validation fails
    """
    result = validate_data(data, validators)
    
    if not result.is_valid:
        raise ValidationError(
            message="Validation failed",
            details={"errors": result.errors}
        )
"""Input validation and sanitization for security."""

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ValidationError


class SecurityError(Exception):
    """Raised when security validation fails."""
    pass


class ValidationResult(BaseModel):
    """Result of input validation."""
    is_valid: bool
    sanitized_value: Optional[Any] = None
    errors: List[str] = []
    warnings: List[str] = []


class InputValidator:
    """Validates and sanitizes user inputs for security and content restrictions."""
    
    # Maximum lengths for different input types
    MAX_TOPIC_LENGTH = 200
    MAX_DESCRIPTION_LENGTH = 2000
    MAX_CODE_LENGTH = 50000
    
    # Dangerous code patterns to detect
    DANGEROUS_CODE_PATTERNS = [
        r'import\s+os',
        r'import\s+subprocess',
        r'from\s+os\s+import',
        r'from\s+subprocess\s+import',
        r'eval\s*\(',
        r'exec\s*\(',
        r'__import__\s*\(',
        r'compile\s*\(',
        r'open\s*\(',
        r'file\s*\(',
        r'input\s*\(',
        r'raw_input\s*\(',
        r'execfile\s*\(',
        r'reload\s*\(',
        r'__builtins__',
        r'globals\s*\(',
        r'locals\s*\(',
        r'vars\s*\(',
        r'dir\s*\(',
        r'hasattr\s*\(',
        r'getattr\s*\(',
        r'setattr\s*\(',
        r'delattr\s*\(',
    ]
    
    # Suspicious content patterns
    SUSPICIOUS_PATTERNS = [
        r'<script[^>]*>',
        r'javascript:',
        r'vbscript:',
        r'onload\s*=',
        r'onerror\s*=',
        r'onclick\s*=',
        r'<iframe[^>]*>',
        r'<object[^>]*>',
        r'<embed[^>]*>',
        r'<link[^>]*>',
        r'<meta[^>]*>',
    ]
    
    # Allowed Manim imports and patterns
    ALLOWED_MANIM_PATTERNS = [
        r'from\s+manim\s+import',
        r'import\s+manim',
        r'from\s+manim\.',
        r'import\s+numpy',
        r'from\s+numpy\s+import',
        r'import\s+math',
        r'from\s+math\s+import',
    ]
    
    @classmethod
    def validate_topic(cls, topic: str) -> ValidationResult:
        """
        Validate and sanitize a topic string.
        
        Args:
            topic: The topic string to validate
            
        Returns:
            ValidationResult with validation status and sanitized value
        """
        errors = []
        warnings = []
        
        # Check if topic is provided
        if not topic or not isinstance(topic, str):
            errors.append("Topic must be a non-empty string")
            return ValidationResult(is_valid=False, errors=errors)
        
        # Check length
        if len(topic.strip()) == 0:
            errors.append("Topic cannot be empty or only whitespace")
            return ValidationResult(is_valid=False, errors=errors)
        
        if len(topic) > cls.MAX_TOPIC_LENGTH:
            errors.append(f"Topic too long (max {cls.MAX_TOPIC_LENGTH} characters)")
            return ValidationResult(is_valid=False, errors=errors)
        
        # Check for suspicious patterns
        for pattern in cls.SUSPICIOUS_PATTERNS:
            if re.search(pattern, topic, re.IGNORECASE):
                errors.append(f"Topic contains potentially dangerous content: {pattern}")
                return ValidationResult(is_valid=False, errors=errors)
        
        # Sanitize the topic
        sanitized = cls._sanitize_text(topic)
        
        # Check if sanitization changed the content significantly
        if len(sanitized) < len(topic) * 0.8:
            warnings.append("Topic was heavily sanitized, please review")
        
        return ValidationResult(
            is_valid=True,
            sanitized_value=sanitized,
            warnings=warnings
        )
    
    @classmethod
    def validate_description(cls, description: str) -> ValidationResult:
        """
        Validate and sanitize a description string.
        
        Args:
            description: The description string to validate
            
        Returns:
            ValidationResult with validation status and sanitized value
        """
        errors = []
        warnings = []
        
        # Check if description is provided
        if not description or not isinstance(description, str):
            errors.append("Description must be a non-empty string")
            return ValidationResult(is_valid=False, errors=errors)
        
        # Check length
        if len(description.strip()) == 0:
            errors.append("Description cannot be empty or only whitespace")
            return ValidationResult(is_valid=False, errors=errors)
        
        if len(description) > cls.MAX_DESCRIPTION_LENGTH:
            errors.append(f"Description too long (max {cls.MAX_DESCRIPTION_LENGTH} characters)")
            return ValidationResult(is_valid=False, errors=errors)
        
        # Check for suspicious patterns
        for pattern in cls.SUSPICIOUS_PATTERNS:
            if re.search(pattern, description, re.IGNORECASE):
                errors.append(f"Description contains potentially dangerous content: {pattern}")
                return ValidationResult(is_valid=False, errors=errors)
        
        # Sanitize the description
        sanitized = cls._sanitize_text(description)
        
        # Check if sanitization changed the content significantly
        if len(sanitized) < len(description) * 0.8:
            warnings.append("Description was heavily sanitized, please review")
        
        return ValidationResult(
            is_valid=True,
            sanitized_value=sanitized,
            warnings=warnings
        )
    
    @classmethod
    def validate_code(cls, code: str, allow_manim_only: bool = True) -> ValidationResult:
        """
        Validate and sanitize code for security.
        
        Args:
            code: The code string to validate
            allow_manim_only: If True, only allow Manim-related imports
            
        Returns:
            ValidationResult with validation status and sanitized value
        """
        errors = []
        warnings = []
        
        # Check if code is provided
        if not code or not isinstance(code, str):
            errors.append("Code must be a non-empty string")
            return ValidationResult(is_valid=False, errors=errors)
        
        # Check length
        if len(code.strip()) == 0:
            errors.append("Code cannot be empty or only whitespace")
            return ValidationResult(is_valid=False, errors=errors)
        
        if len(code) > cls.MAX_CODE_LENGTH:
            errors.append(f"Code too long (max {cls.MAX_CODE_LENGTH} characters)")
            return ValidationResult(is_valid=False, errors=errors)
        
        # Check for dangerous code patterns
        for pattern in cls.DANGEROUS_CODE_PATTERNS:
            matches = re.findall(pattern, code, re.IGNORECASE)
            if matches:
                # If allowing Manim only, check if it's an allowed pattern
                if allow_manim_only:
                    is_allowed = any(
                        re.search(allowed_pattern, code, re.IGNORECASE)
                        for allowed_pattern in cls.ALLOWED_MANIM_PATTERNS
                        if pattern in allowed_pattern
                    )
                    if not is_allowed:
                        errors.append(f"Potentially dangerous code pattern detected: {pattern}")
                        return ValidationResult(is_valid=False, errors=errors)
                else:
                    errors.append(f"Potentially dangerous code pattern detected: {pattern}")
                    return ValidationResult(is_valid=False, errors=errors)
        
        # Check for suspicious web content
        for pattern in cls.SUSPICIOUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                errors.append(f"Code contains potentially dangerous web content: {pattern}")
                return ValidationResult(is_valid=False, errors=errors)
        
        # Basic Python syntax validation (simple check)
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            warnings.append(f"Code has syntax errors: {str(e)}")
        
        # Code is valid
        return ValidationResult(
            is_valid=True,
            sanitized_value=code,  # Code is not sanitized, only validated
            warnings=warnings
        )
    
    @classmethod
    def validate_session_id(cls, session_id: str) -> ValidationResult:
        """
        Validate a session ID.
        
        Args:
            session_id: The session ID to validate
            
        Returns:
            ValidationResult with validation status and sanitized value
        """
        errors = []
        
        if not session_id or not isinstance(session_id, str):
            errors.append("Session ID must be a non-empty string")
            return ValidationResult(is_valid=False, errors=errors)
        
        # Session ID should be alphanumeric with hyphens and underscores
        if not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
            errors.append("Session ID can only contain letters, numbers, hyphens, and underscores")
            return ValidationResult(is_valid=False, errors=errors)
        
        # Check length (reasonable bounds)
        if len(session_id) < 3 or len(session_id) > 100:
            errors.append("Session ID must be between 3 and 100 characters")
            return ValidationResult(is_valid=False, errors=errors)
        
        return ValidationResult(is_valid=True, sanitized_value=session_id)
    
    @classmethod
    def validate_workflow_input(cls, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate complete workflow input data.
        
        Args:
            data: Dictionary containing workflow input data
            
        Returns:
            ValidationResult with validation status and sanitized values
        """
        errors = []
        warnings = []
        sanitized_data = {}
        
        # Validate required fields
        required_fields = ['topic', 'description', 'session_id']
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return ValidationResult(is_valid=False, errors=errors)
        
        # Validate each field
        topic_result = cls.validate_topic(data['topic'])
        if not topic_result.is_valid:
            errors.extend([f"Topic: {error}" for error in topic_result.errors])
        else:
            sanitized_data['topic'] = topic_result.sanitized_value
            warnings.extend([f"Topic: {warning}" for warning in topic_result.warnings])
        
        desc_result = cls.validate_description(data['description'])
        if not desc_result.is_valid:
            errors.extend([f"Description: {error}" for error in desc_result.errors])
        else:
            sanitized_data['description'] = desc_result.sanitized_value
            warnings.extend([f"Description: {warning}" for warning in desc_result.warnings])
        
        session_result = cls.validate_session_id(data['session_id'])
        if not session_result.is_valid:
            errors.extend([f"Session ID: {error}" for error in session_result.errors])
        else:
            sanitized_data['session_id'] = session_result.sanitized_value
        
        # Copy other valid fields
        for key, value in data.items():
            if key not in required_fields and key not in sanitized_data:
                sanitized_data[key] = value
        
        if errors:
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        return ValidationResult(
            is_valid=True,
            sanitized_value=sanitized_data,
            warnings=warnings
        )
    
    @staticmethod
    def _sanitize_text(text: str) -> str:
        """
        Sanitize text by removing potentially dangerous characters.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove potentially dangerous characters
        text = re.sub(r'[<>"\'\`]', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        return text.strip()
    
    @classmethod
    def get_validation_rules(cls) -> Dict[str, Any]:
        """
        Get the current validation rules and limits.
        
        Returns:
            Dictionary containing validation rules
        """
        return {
            'max_topic_length': cls.MAX_TOPIC_LENGTH,
            'max_description_length': cls.MAX_DESCRIPTION_LENGTH,
            'max_code_length': cls.MAX_CODE_LENGTH,
            'dangerous_patterns_count': len(cls.DANGEROUS_CODE_PATTERNS),
            'suspicious_patterns_count': len(cls.SUSPICIOUS_PATTERNS),
            'allowed_manim_patterns_count': len(cls.ALLOWED_MANIM_PATTERNS),
        }
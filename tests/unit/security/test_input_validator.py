"""Unit tests for InputValidator."""

import pytest
from src.langgraph_agents.security.input_validator import (
    InputValidator,
    ValidationResult,
    SecurityError
)


class TestInputValidator:
    """Test cases for InputValidator class."""
    
    def test_validate_topic_success(self):
        """Test successful topic validation."""
        topic = "Introduction to Python Programming"
        result = InputValidator.validate_topic(topic)
        
        assert result.is_valid
        assert result.sanitized_value == topic
        assert len(result.errors) == 0
    
    def test_validate_topic_empty(self):
        """Test topic validation with empty input."""
        result = InputValidator.validate_topic("")
        
        assert not result.is_valid
        assert "non-empty string" in result.errors[0]
    
    def test_validate_topic_whitespace_only(self):
        """Test topic validation with whitespace-only input."""
        result = InputValidator.validate_topic("   ")
        
        assert not result.is_valid
        assert "cannot be empty or only whitespace" in result.errors[0]
    
    def test_validate_topic_none(self):
        """Test topic validation with None input."""
        result = InputValidator.validate_topic(None)
        
        assert not result.is_valid
        assert "non-empty string" in result.errors[0]
    
    def test_validate_topic_too_long(self):
        """Test topic validation with overly long input."""
        long_topic = "x" * (InputValidator.MAX_TOPIC_LENGTH + 1)
        result = InputValidator.validate_topic(long_topic)
        
        assert not result.is_valid
        assert "too long" in result.errors[0]
    
    def test_validate_topic_suspicious_content(self):
        """Test topic validation with suspicious content."""
        suspicious_topic = "Learn Python <script>alert('xss')</script>"
        result = InputValidator.validate_topic(suspicious_topic)
        
        assert not result.is_valid
        assert "dangerous content" in result.errors[0]
    
    def test_validate_topic_sanitization(self):
        """Test topic sanitization."""
        topic_with_quotes = 'Python "Programming" Basics'
        result = InputValidator.validate_topic(topic_with_quotes)
        
        assert result.is_valid
        assert '"' not in result.sanitized_value
        assert "Python Programming Basics" == result.sanitized_value
    
    def test_validate_description_success(self):
        """Test successful description validation."""
        description = "This is a comprehensive introduction to Python programming concepts."
        result = InputValidator.validate_description(description)
        
        assert result.is_valid
        assert result.sanitized_value == description
        assert len(result.errors) == 0
    
    def test_validate_description_empty(self):
        """Test description validation with empty input."""
        result = InputValidator.validate_description("")
        
        assert not result.is_valid
        assert "non-empty string" in result.errors[0]
    
    def test_validate_description_too_long(self):
        """Test description validation with overly long input."""
        long_description = "x" * (InputValidator.MAX_DESCRIPTION_LENGTH + 1)
        result = InputValidator.validate_description(long_description)
        
        assert not result.is_valid
        assert "too long" in result.errors[0]
    
    def test_validate_description_suspicious_content(self):
        """Test description validation with suspicious content."""
        suspicious_desc = "Learn Python <iframe src='evil.com'></iframe>"
        result = InputValidator.validate_description(suspicious_desc)
        
        assert not result.is_valid
        assert "dangerous content" in result.errors[0]
    
    def test_validate_code_success_manim(self):
        """Test successful code validation with Manim imports."""
        code = """
from manim import *
import numpy as np

class TestScene(Scene):
    def construct(self):
        circle = Circle()
        self.play(Create(circle))
        """
        result = InputValidator.validate_code(code, allow_manim_only=True)
        
        assert result.is_valid
        assert result.sanitized_value == code
    
    def test_validate_code_dangerous_import(self):
        """Test code validation with dangerous imports."""
        code = """
import os
os.system('rm -rf /')
        """
        result = InputValidator.validate_code(code, allow_manim_only=True)
        
        assert not result.is_valid
        assert "dangerous code pattern" in result.errors[0]
    
    def test_validate_code_eval_exec(self):
        """Test code validation with eval/exec."""
        code = """
user_input = "print('hello')"
eval(user_input)
        """
        result = InputValidator.validate_code(code, allow_manim_only=True)
        
        assert not result.is_valid
        assert "dangerous code pattern" in result.errors[0]
    
    def test_validate_code_empty(self):
        """Test code validation with empty input."""
        result = InputValidator.validate_code("")
        
        assert not result.is_valid
        assert "non-empty string" in result.errors[0]
    
    def test_validate_code_too_long(self):
        """Test code validation with overly long input."""
        long_code = "# " + "x" * InputValidator.MAX_CODE_LENGTH
        result = InputValidator.validate_code(long_code)
        
        assert not result.is_valid
        assert "too long" in result.errors[0]
    
    def test_validate_code_syntax_error(self):
        """Test code validation with syntax errors."""
        code = """
from manim import *
class TestScene(Scene:  # Missing closing parenthesis
    def construct(self):
        pass
        """
        result = InputValidator.validate_code(code, allow_manim_only=True)
        
        assert result.is_valid  # Still valid, but with warnings
        assert len(result.warnings) > 0
        assert "syntax errors" in result.warnings[0]
    
    def test_validate_code_web_content(self):
        """Test code validation with web content."""
        code = """
html = '<script>alert("xss")</script>'
        """
        result = InputValidator.validate_code(code, allow_manim_only=True)
        
        assert not result.is_valid
        assert "dangerous web content" in result.errors[0]
    
    def test_validate_session_id_success(self):
        """Test successful session ID validation."""
        session_id = "session_123-abc"
        result = InputValidator.validate_session_id(session_id)
        
        assert result.is_valid
        assert result.sanitized_value == session_id
    
    def test_validate_session_id_invalid_chars(self):
        """Test session ID validation with invalid characters."""
        session_id = "session@123"
        result = InputValidator.validate_session_id(session_id)
        
        assert not result.is_valid
        assert "letters, numbers, hyphens, and underscores" in result.errors[0]
    
    def test_validate_session_id_too_short(self):
        """Test session ID validation with too short input."""
        session_id = "ab"
        result = InputValidator.validate_session_id(session_id)
        
        assert not result.is_valid
        assert "between 3 and 100 characters" in result.errors[0]
    
    def test_validate_session_id_too_long(self):
        """Test session ID validation with too long input."""
        session_id = "x" * 101
        result = InputValidator.validate_session_id(session_id)
        
        assert not result.is_valid
        assert "between 3 and 100 characters" in result.errors[0]
    
    def test_validate_workflow_input_success(self):
        """Test successful workflow input validation."""
        data = {
            'topic': 'Python Basics',
            'description': 'Learn Python programming fundamentals',
            'session_id': 'session_123',
            'config': {'model': 'gpt-4'}
        }
        result = InputValidator.validate_workflow_input(data)
        
        assert result.is_valid
        assert result.sanitized_value['topic'] == 'Python Basics'
        assert result.sanitized_value['description'] == 'Learn Python programming fundamentals'
        assert result.sanitized_value['session_id'] == 'session_123'
        assert result.sanitized_value['config'] == {'model': 'gpt-4'}
    
    def test_validate_workflow_input_missing_fields(self):
        """Test workflow input validation with missing required fields."""
        data = {
            'topic': 'Python Basics'
            # Missing description and session_id
        }
        result = InputValidator.validate_workflow_input(data)
        
        assert not result.is_valid
        assert any("Missing required field: description" in error for error in result.errors)
        assert any("Missing required field: session_id" in error for error in result.errors)
    
    def test_validate_workflow_input_invalid_topic(self):
        """Test workflow input validation with invalid topic."""
        data = {
            'topic': '<script>alert("xss")</script>',
            'description': 'Valid description',
            'session_id': 'session_123'
        }
        result = InputValidator.validate_workflow_input(data)
        
        assert not result.is_valid
        assert any("Topic:" in error and "dangerous content" in error for error in result.errors)
    
    def test_validate_workflow_input_with_warnings(self):
        """Test workflow input validation that produces warnings."""
        data = {
            'topic': 'Python "Advanced" <Programming>',  # Will be sanitized
            'description': 'Valid description',
            'session_id': 'session_123'
        }
        result = InputValidator.validate_workflow_input(data)
        
        assert result.is_valid
        assert len(result.warnings) > 0
    
    def test_sanitize_text(self):
        """Test text sanitization."""
        text = 'Hello <b>World</b> "with" quotes'
        sanitized = InputValidator._sanitize_text(text)
        
        assert '<b>' not in sanitized
        assert '</b>' not in sanitized
        assert '"' not in sanitized
        assert 'Hello World with quotes' == sanitized
    
    def test_sanitize_text_whitespace(self):
        """Test text sanitization with excessive whitespace."""
        text = '  Hello    World  \n\n  Test  '
        sanitized = InputValidator._sanitize_text(text)
        
        assert sanitized == 'Hello World Test'
    
    def test_get_validation_rules(self):
        """Test getting validation rules."""
        rules = InputValidator.get_validation_rules()
        
        assert 'max_topic_length' in rules
        assert 'max_description_length' in rules
        assert 'max_code_length' in rules
        assert 'dangerous_patterns_count' in rules
        assert 'suspicious_patterns_count' in rules
        assert 'allowed_manim_patterns_count' in rules
        
        assert rules['max_topic_length'] == InputValidator.MAX_TOPIC_LENGTH
        assert rules['max_description_length'] == InputValidator.MAX_DESCRIPTION_LENGTH
        assert rules['max_code_length'] == InputValidator.MAX_CODE_LENGTH
    
    def test_validation_result_model(self):
        """Test ValidationResult model."""
        result = ValidationResult(
            is_valid=True,
            sanitized_value="test",
            errors=["error1"],
            warnings=["warning1"]
        )
        
        assert result.is_valid
        assert result.sanitized_value == "test"
        assert result.errors == ["error1"]
        assert result.warnings == ["warning1"]
    
    def test_security_error_exception(self):
        """Test SecurityError exception."""
        with pytest.raises(SecurityError):
            raise SecurityError("Test security error")


class TestInputValidatorEdgeCases:
    """Test edge cases for InputValidator."""
    
    def test_validate_topic_unicode(self):
        """Test topic validation with unicode characters."""
        topic = "Python编程基础"
        result = InputValidator.validate_topic(topic)
        
        assert result.is_valid
        assert result.sanitized_value == topic
    
    def test_validate_code_allowed_imports_mixed(self):
        """Test code validation with mixed allowed and dangerous imports."""
        code = """
from manim import *
import numpy as np
import os  # This should be caught
        """
        result = InputValidator.validate_code(code, allow_manim_only=True)
        
        assert not result.is_valid
        assert "dangerous code pattern" in result.errors[0]
    
    def test_validate_code_complex_manim(self):
        """Test code validation with complex but valid Manim code."""
        code = """
from manim import *
import numpy as np
import math

class ComplexScene(Scene):
    def construct(self):
        # Create mathematical objects
        axes = Axes(x_range=[-3, 3], y_range=[-2, 2])
        func = axes.plot(lambda x: np.sin(x), color=BLUE)
        
        # Animate
        self.play(Create(axes))
        self.play(Create(func))
        self.wait()
        """
        result = InputValidator.validate_code(code, allow_manim_only=True)
        
        assert result.is_valid
    
    def test_validate_workflow_input_extra_fields(self):
        """Test workflow input validation with extra fields."""
        data = {
            'topic': 'Python Basics',
            'description': 'Learn Python programming',
            'session_id': 'session_123',
            'extra_field': 'extra_value',
            'config': {'model': 'gpt-4'}
        }
        result = InputValidator.validate_workflow_input(data)
        
        assert result.is_valid
        assert 'extra_field' in result.sanitized_value
        assert result.sanitized_value['extra_field'] == 'extra_value'
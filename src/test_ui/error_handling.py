"""
Enhanced Error Handling for Gradio Interface

This module provides enhanced error handling capabilities for the Gradio testing
interface, including user-friendly error displays, recovery suggestions, and
graceful error recovery mechanisms.
"""

import gradio as gr
import traceback
from typing import Dict, List, Optional, Any, Tuple, Callable
from functools import wraps
import logging

from src.test_ui.validation import ErrorHandler, ValidationError

logger = logging.getLogger(__name__)


class GradioErrorHandler:
    """Enhanced error handling for Gradio interface components."""
    
    def __init__(self):
        self.error_history = []
        self.max_error_history = 100
    
    def create_error_display_component(self) -> gr.Markdown:
        """
        Create a Gradio component for displaying errors.
        
        Returns:
            Gradio Markdown component for error display
        """
        return gr.Markdown(
            value="",
            visible=False,
            elem_classes=["error-display"]
        )
    
    def create_success_display_component(self) -> gr.Markdown:
        """
        Create a Gradio component for displaying success messages.
        
        Returns:
            Gradio Markdown component for success display
        """
        return gr.Markdown(
            value="",
            visible=False,
            elem_classes=["success-display"]
        )
    
    def show_error(
        self, 
        error_message: str, 
        suggestions: List[str] = None,
        error_details: str = None
    ) -> gr.update:
        """
        Show error message in the UI.
        
        Args:
            error_message: Main error message
            suggestions: List of recovery suggestions
            error_details: Detailed error information
            
        Returns:
            Gradio update for error display component
        """
        display_text = ErrorHandler.create_error_display(
            error_message=error_message,
            suggestions=suggestions,
            show_details=bool(error_details),
            error_details=error_details
        )
        
        # Add to error history
        self._add_to_error_history(error_message, suggestions, error_details)
        
        return gr.update(
            value=display_text,
            visible=True,
            elem_classes=["error-display", "alert", "alert-danger"]
        )
    
    def show_success(self, message: str) -> gr.update:
        """
        Show success message in the UI.
        
        Args:
            message: Success message
            
        Returns:
            Gradio update for success display component
        """
        display_text = f"✅ **{message}**\n\n*{self._get_timestamp()}*"
        
        return gr.update(
            value=display_text,
            visible=True,
            elem_classes=["success-display", "alert", "alert-success"]
        )
    
    def hide_messages(self) -> Tuple[gr.update, gr.update]:
        """
        Hide both error and success messages.
        
        Returns:
            Tuple of updates for error and success display components
        """
        return (
            gr.update(visible=False),  # Error display
            gr.update(visible=False)   # Success display
        )
    
    def handle_api_error(self, error: Exception) -> gr.update:
        """
        Handle API errors and return appropriate UI update.
        
        Args:
            error: Exception that occurred
            
        Returns:
            Gradio update for error display
        """
        error_message, suggestions = ErrorHandler.handle_api_error(error)
        return self.show_error(
            error_message=error_message,
            suggestions=suggestions,
            error_details=str(error)
        )
    
    def handle_validation_error(self, error: ValidationError) -> gr.update:
        """
        Handle validation errors and return appropriate UI update.
        
        Args:
            error: ValidationError that occurred
            
        Returns:
            Gradio update for error display
        """
        error_message = ErrorHandler.format_validation_error(error)
        return self.show_error(
            error_message=error_message,
            suggestions=error.suggestions
        )
    
    def _add_to_error_history(
        self, 
        error_message: str, 
        suggestions: List[str] = None,
        error_details: str = None
    ):
        """Add error to history for debugging purposes."""
        error_entry = {
            "timestamp": self._get_timestamp(),
            "message": error_message,
            "suggestions": suggestions or [],
            "details": error_details
        }
        
        self.error_history.append(error_entry)
        
        # Keep only the most recent errors
        if len(self.error_history) > self.max_error_history:
            self.error_history = self.error_history[-self.max_error_history:]
    
    def _get_timestamp(self) -> str:
        """Get formatted timestamp."""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def get_error_history(self) -> List[Dict[str, Any]]:
        """Get error history for debugging."""
        return self.error_history.copy()


def with_error_handling(
    error_handler: GradioErrorHandler,
    error_display_component: gr.Markdown = None,
    success_display_component: gr.Markdown = None,
    show_success_message: str = None
):
    """
    Decorator to add error handling to Gradio event handler functions.
    
    Args:
        error_handler: GradioErrorHandler instance
        error_display_component: Component to display errors
        success_display_component: Component to display success messages
        show_success_message: Success message to show on successful execution
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Execute the original function
                result = func(*args, **kwargs)
                
                # Show success message if specified
                if show_success_message and success_display_component:
                    success_update = error_handler.show_success(show_success_message)
                    # If result is a tuple, append the success update
                    if isinstance(result, tuple):
                        result = result + (success_update,)
                    else:
                        result = (result, success_update)
                
                return result
                
            except ValidationError as e:
                logger.error(f"Validation error in {func.__name__}: {e}")
                error_update = error_handler.handle_validation_error(e)
                
                # Return appropriate error response
                if error_display_component:
                    return error_update
                else:
                    # Return default error values based on expected return type
                    return _get_error_return_values(func, error_update)
                    
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                logger.error(traceback.format_exc())
                
                error_update = error_handler.handle_api_error(e)
                
                # Return appropriate error response
                if error_display_component:
                    return error_update
                else:
                    # Return default error values based on expected return type
                    return _get_error_return_values(func, error_update)
        
        return wrapper
    return decorator


def _get_error_return_values(func: Callable, error_update: gr.update) -> Any:
    """
    Get appropriate error return values based on function signature.
    
    Args:
        func: Original function
        error_update: Error update for display component
        
    Returns:
        Appropriate error return values
    """
    # This is a simplified approach - in a real implementation,
    # you might want to inspect the function signature more carefully
    
    # Return a tuple with error update and default values
    # The exact structure depends on what the function normally returns
    return (
        "Error occurred",  # Default error message
        error_update,      # Error display update
        gr.update(),       # Default component update
        gr.update()        # Another default component update
    )


class ValidationMixin:
    """Mixin class to add validation capabilities to Gradio interface classes."""
    
    def __init__(self):
        self.error_handler = GradioErrorHandler()
        self.validation_enabled = True
    
    def validate_and_execute(
        self,
        validation_func: Callable,
        execution_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Validate inputs and execute function if validation passes.
        
        Args:
            validation_func: Function to validate inputs
            execution_func: Function to execute if validation passes
            *args: Arguments to pass to functions
            **kwargs: Keyword arguments to pass to functions
            
        Returns:
            Result of execution_func or error response
        """
        if not self.validation_enabled:
            return execution_func(*args, **kwargs)
        
        try:
            # Validate inputs
            is_valid, errors = validation_func(*args, **kwargs)
            
            if not is_valid:
                # Create validation error
                error_message = "Validation failed"
                if isinstance(errors, list):
                    error_message = "; ".join(errors)
                elif isinstance(errors, str):
                    error_message = errors
                
                raise ValidationError(error_message)
            
            # Execute function if validation passes
            return execution_func(*args, **kwargs)
            
        except ValidationError:
            raise  # Re-raise validation errors
        except Exception as e:
            logger.error(f"Error in validate_and_execute: {e}")
            raise
    
    def create_validated_form_handler(
        self,
        validation_rules: Dict[str, Any],
        execution_func: Callable
    ) -> Callable:
        """
        Create a form handler with built-in validation.
        
        Args:
            validation_rules: Validation rules for form fields
            execution_func: Function to execute after validation
            
        Returns:
            Form handler function with validation
        """
        def form_handler(*args, **kwargs):
            # Convert args to form data dictionary
            # This is a simplified approach - you might need to adapt based on your form structure
            form_data = {}
            if args:
                # Assume first argument is form data or convert args to dict
                if isinstance(args[0], dict):
                    form_data = args[0]
                else:
                    # Create form data from args based on validation rules keys
                    field_names = list(validation_rules.keys())
                    for i, value in enumerate(args):
                        if i < len(field_names):
                            form_data[field_names[i]] = value
            
            # Validate form data
            from src.test_ui.validation import ErrorHandler
            is_valid, field_errors = ErrorHandler.validate_form_data(form_data, validation_rules)
            
            if not is_valid:
                error_messages = []
                for field, error in field_errors.items():
                    error_messages.append(f"{field}: {error}")
                
                raise ValidationError(
                    message="Form validation failed",
                    suggestions=error_messages
                )
            
            # Execute function if validation passes
            return execution_func(*args, **kwargs)
        
        return form_handler
    
    def enable_validation(self):
        """Enable input validation."""
        self.validation_enabled = True
    
    def disable_validation(self):
        """Disable input validation (for debugging)."""
        self.validation_enabled = False


# Custom CSS for error and success displays
ERROR_HANDLING_CSS = """
.error-display {
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    border-radius: 0.375rem;
    padding: 1rem;
    margin: 0.5rem 0;
    color: #721c24;
}

.success-display {
    background-color: #d1edff;
    border: 1px solid #bee5eb;
    border-radius: 0.375rem;
    padding: 1rem;
    margin: 0.5rem 0;
    color: #0c5460;
}

.alert {
    position: relative;
    padding: 0.75rem 1.25rem;
    margin-bottom: 1rem;
    border: 1px solid transparent;
    border-radius: 0.375rem;
}

.alert-danger {
    color: #721c24;
    background-color: #f8d7da;
    border-color: #f5c6cb;
}

.alert-success {
    color: #155724;
    background-color: #d4edda;
    border-color: #c3e6cb;
}

.validation-error {
    border-color: #dc3545 !important;
    box-shadow: 0 0 0 0.2rem rgba(220, 53, 69, 0.25) !important;
}

.validation-success {
    border-color: #28a745 !important;
    box-shadow: 0 0 0 0.2rem rgba(40, 167, 69, 0.25) !important;
}
"""
"""
Warning suppression utilities for the LangGraph agents.

This module provides utilities to suppress common warnings that occur
during imports and execution of various components.
"""

import warnings
import logging
from functools import wraps


def suppress_deprecation_warnings():
    """Suppress common deprecation warnings."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # Suppress specific warnings from common libraries
    warnings.filterwarnings("ignore", message=".*deprecated.*", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*distutils.*", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*pkg_resources.*", category=DeprecationWarning)


def suppress_import_warnings():
    """Suppress warnings during imports."""
    warnings.filterwarnings("ignore", category=ImportWarning)
    warnings.filterwarnings("ignore", message=".*import.*", category=UserWarning)


def suppress_all_warnings():
    """Suppress all warnings."""
    warnings.filterwarnings("ignore")


def with_suppressed_warnings(func):
    """Decorator to suppress warnings during function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)
    return wrapper


def with_suppressed_warnings_async(func):
    """Decorator to suppress warnings during async function execution."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return await func(*args, **kwargs)
    return wrapper


# Context manager for temporary warning suppression
class SuppressWarnings:
    """Context manager to temporarily suppress warnings."""
    
    def __init__(self, categories=None):
        """Initialize with specific warning categories to suppress.
        
        Args:
            categories: List of warning categories to suppress. If None, suppress all.
        """
        self.categories = categories or [Warning]
        self.original_filters = None
    
    def __enter__(self):
        """Enter the context and suppress warnings."""
        self.original_filters = warnings.filters[:]
        for category in self.categories:
            warnings.filterwarnings("ignore", category=category)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and restore original warning filters."""
        warnings.filters[:] = self.original_filters
        return False


# Initialize warning suppression on module import
suppress_deprecation_warnings()
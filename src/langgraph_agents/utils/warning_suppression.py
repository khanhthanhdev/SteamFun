"""
Warning suppression utilities for LangGraph agents.
Handles deprecation warnings from third-party libraries.
"""

import warnings
import functools
from typing import Any, Callable


def suppress_deprecation_warnings():
    """Suppress common deprecation warnings from third-party libraries.
    
    This function suppresses warnings that are outside our control,
    particularly from libraries like ragas that use deprecated pydantic v1 imports.
    """
    # Suppress LangChain pydantic v1 deprecation warnings
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module="ragas.*",
        message=".*pydantic.*"
    )
    
    warnings.filterwarnings(
        "ignore", 
        category=DeprecationWarning,
        message=".*langchain_core.pydantic_v1.*"
    )
    
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning, 
        message=".*langchain.pydantic_v1.*"
    )
    
    # Suppress specific ragas warnings
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module="ragas.metrics.*"
    )


def with_suppressed_warnings(func: Callable) -> Callable:
    """Decorator to suppress warnings for a specific function.
    
    Args:
        func: Function to wrap with warning suppression
        
    Returns:
        Wrapped function with warnings suppressed
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            suppress_deprecation_warnings()
            return func(*args, **kwargs)
    
    return wrapper


def suppress_import_warnings():
    """Suppress warnings during import operations.
    
    This is specifically useful for imports that trigger deprecation warnings
    from third-party libraries we can't control.
    """
    with warnings.catch_warnings():
        suppress_deprecation_warnings()
        
        # Import problematic modules with warnings suppressed
        try:
            import ragas.metrics
        except ImportError:
            pass  # ragas might not be installed
        
        try:
            from src.rag.rag_integration import RAGIntegration
        except ImportError:
            pass  # RAG components might not be available
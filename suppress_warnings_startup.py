#!/usr/bin/env python3
"""
Startup script to suppress common deprecation warnings.
Run this before importing any modules that might trigger warnings.
"""

import warnings
import os

def setup_warning_suppression():
    """Set up global warning suppression for common issues."""
    
    # Set environment variable to suppress warnings globally
    os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
    
    # Suppress LangChain pydantic v1 deprecation warnings
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
    
    # Suppress ragas-specific warnings
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module="ragas.*"
    )
    
    warnings.filterwarnings(
        "ignore",
        message=".*Enhanced components not fully available.*"
    )
    
    warnings.filterwarnings(
        "ignore",
        message=".*cannot import name 'AnswerAccuracy'.*"
    )
    
    # General pydantic warnings
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=".*pydantic.*"
    )
    
    print("✅ Warning suppression configured")

if __name__ == "__main__":
    setup_warning_suppression()
else:
    # Auto-setup when imported
    setup_warning_suppression()
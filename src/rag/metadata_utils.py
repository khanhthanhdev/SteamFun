"""
Utility functions for handling metadata in vector stores.
"""

from typing import Dict, Any, List, Union
import json


def sanitize_metadata_for_chroma(metadata: Dict[str, Any]) -> Dict[str, Union[str, int, float, bool]]:
    """
    Sanitize metadata to ensure compatibility with Chroma vector store.
    
    Chroma only accepts scalar values (str, int, float, bool) in metadata.
    This function converts lists and other complex types to strings.
    
    Args:
        metadata: Dictionary containing metadata with potentially complex values
        
    Returns:
        Dictionary with all values converted to Chroma-compatible types
    """
    sanitized = {}
    
    for key, value in metadata.items():
        if value is None:
            sanitized[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, list):
            # Convert lists to comma-separated strings
            if all(isinstance(item, (str, int, float)) for item in value):
                sanitized[key] = ", ".join(str(item) for item in value)
            else:
                # For complex list items, use JSON serialization
                sanitized[key] = json.dumps(value)
        elif isinstance(value, dict):
            # Convert dictionaries to JSON strings
            sanitized[key] = json.dumps(value)
        else:
            # Convert other types to string representation
            sanitized[key] = str(value)
    
    return sanitized


def restore_metadata_from_chroma(metadata: Dict[str, Union[str, int, float, bool]]) -> Dict[str, Any]:
    """
    Restore metadata that was sanitized for Chroma storage.
    
    This function attempts to restore lists and dictionaries that were
    converted to strings for Chroma compatibility.
    
    Args:
        metadata: Dictionary with Chroma-compatible values
        
    Returns:
        Dictionary with restored complex types where possible
    """
    restored = {}
    
    # Keys that should be restored as lists (comma-separated strings)
    list_keys = {
        'methods', 'decorators', 'semantic_tags', 'hierarchy_path', 
        'parent_headers', 'parameters', 'nested_elements'
    }
    
    # Keys that should be restored as JSON (if they look like JSON)
    json_keys = {
        'relationships', 'code_structure'
    }
    
    for key, value in metadata.items():
        if key in list_keys and isinstance(value, str) and value:
            # Restore comma-separated strings to lists
            restored[key] = [item.strip() for item in value.split(',') if item.strip()]
        elif key in json_keys and isinstance(value, str) and value:
            # Try to restore JSON strings
            try:
                restored[key] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                restored[key] = value
        else:
            restored[key] = value
    
    return restored


def validate_chroma_metadata(metadata: Dict[str, Any]) -> bool:
    """
    Validate that metadata is compatible with Chroma vector store.
    
    Args:
        metadata: Dictionary to validate
        
    Returns:
        True if metadata is compatible, False otherwise
    """
    for key, value in metadata.items():
        if not isinstance(value, (str, int, float, bool, type(None))):
            return False
    
    return True


def get_metadata_summary(metadata: Dict[str, Any]) -> Dict[str, str]:
    """
    Get a summary of metadata types for debugging.
    
    Args:
        metadata: Dictionary to analyze
        
    Returns:
        Dictionary mapping keys to their value types
    """
    return {key: type(value).__name__ for key, value in metadata.items()}
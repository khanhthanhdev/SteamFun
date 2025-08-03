"""
Centralized logging configuration for the application.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

from .logging import setup_logging


def get_logging_config() -> Dict[str, Any]:
    """
    Get logging configuration based on environment.
    
    Returns:
        Logging configuration dictionary
    """
    env = os.getenv("ENVIRONMENT", "development").lower()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # Base configuration
    config = {
        "level": log_level,
        "json_format": env == "production",
        "context": {
            "environment": env,
            "service": "video-generation-api"
        }
    }
    
    # Add log file for production and staging
    if env in ["production", "staging"]:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        config["log_file"] = str(log_dir / f"app-{env}.log")
    
    return config


def configure_application_logging() -> None:
    """Configure logging for the entire application."""
    config = get_logging_config()
    setup_logging(**config)


def get_service_logger_context(service_name: str) -> Dict[str, Any]:
    """
    Get logger context for a specific service.
    
    Args:
        service_name: Name of the service
        
    Returns:
        Context dictionary for the service logger
    """
    return {
        "service": service_name,
        "environment": os.getenv("ENVIRONMENT", "development"),
        "version": os.getenv("APP_VERSION", "1.0.0")
    }


def get_request_logger_context(request_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get logger context for a specific request.
    
    Args:
        request_id: Unique request identifier
        user_id: Optional user identifier
        
    Returns:
        Context dictionary for the request logger
    """
    context = {
        "request_id": request_id,
        "environment": os.getenv("ENVIRONMENT", "development")
    }
    
    if user_id:
        context["user_id"] = user_id
    
    return context
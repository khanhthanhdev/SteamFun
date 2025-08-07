"""
Configuration and setup for LangGraph Studio integration.

This module provides configuration management and setup utilities for
running the agent system in LangGraph Studio.
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from ..models.config import WorkflowConfig, ModelConfig

logger = logging.getLogger(__name__)


class StudioConfig:
    """Configuration manager for Studio integration."""
    
    def __init__(self):
        self.studio_enabled = self._check_studio_environment()
        self.workspace_path = self._get_workspace_path()
        self.output_path = self._get_output_path()
        self.test_data_path = self._get_test_data_path()
        
    def _check_studio_environment(self) -> bool:
        """Check if running in LangGraph Studio environment."""
        # Check for Studio-specific environment variables
        studio_indicators = [
            "LANGGRAPH_STUDIO",
            "LANGSMITH_TRACING",
            "LANGCHAIN_TRACING_V2"
        ]
        
        for indicator in studio_indicators:
            if os.getenv(indicator):
                logger.info(f"Studio environment detected: {indicator}")
                return True
        
        # Check for Studio-specific paths or processes
        if os.path.exists("/.langgraph"):
            logger.info("Studio environment detected: .langgraph directory found")
            return True
        
        return False
    
    def _get_workspace_path(self) -> Path:
        """Get the workspace path for Studio."""
        workspace = os.getenv("LANGGRAPH_WORKSPACE", os.getcwd())
        return Path(workspace)
    
    def _get_output_path(self) -> Path:
        """Get the output path for generated content."""
        output_dir = os.getenv("STUDIO_OUTPUT_DIR", "studio_output")
        output_path = self.workspace_path / output_dir
        output_path.mkdir(exist_ok=True)
        return output_path
    
    def _get_test_data_path(self) -> Path:
        """Get the path for test data storage."""
        test_data_dir = os.getenv("STUDIO_TEST_DATA_DIR", "studio_test_data")
        test_data_path = self.workspace_path / test_data_dir
        test_data_path.mkdir(exist_ok=True)
        return test_data_path
    
    def get_workflow_config(self) -> WorkflowConfig:
        """Get workflow configuration optimized for Studio testing."""
        config = WorkflowConfig()
        
        # Override settings for Studio environment
        if self.studio_enabled:
            # Use faster, lighter models for testing
            config.planner_model = ModelConfig(
                provider="openrouter",
                model_name="anthropic/claude-3-haiku",
                temperature=0.7,
                max_tokens=2000,
                timeout=60
            )
            
            config.code_model = ModelConfig(
                provider="openrouter",
                model_name="anthropic/claude-3-haiku",
                temperature=0.3,
                max_tokens=4000,
                timeout=120
            )
            
            config.helper_model = ModelConfig(
                provider="openrouter",
                model_name="anthropic/claude-3-haiku",
                temperature=0.7,
                max_tokens=2000,
                timeout=60
            )
            
            # Optimize for testing
            config.max_retries = 2
            config.timeout_seconds = 180
            config.max_concurrent_scenes = 2
            config.max_concurrent_renders = 1
            config.preview_mode = True
            config.default_quality = "low"
            config.use_gpu_acceleration = False
            
            # Set paths
            config.output_dir = str(self.output_path)
            config.context_learning_path = str(self.test_data_path / "context_learning")
            config.chroma_db_path = str(self.test_data_path / "chroma_db")
            config.manim_docs_path = str(self.test_data_path / "manim_docs")
            
            # Enable monitoring for Studio
            config.enable_monitoring = True
            config.use_langfuse = True
            config.verbose = True
        
        return config
    
    def setup_studio_environment(self) -> Dict[str, Any]:
        """Set up the Studio testing environment."""
        logger.info("Setting up Studio testing environment")
        
        setup_results = {
            "studio_enabled": self.studio_enabled,
            "workspace_path": str(self.workspace_path),
            "output_path": str(self.output_path),
            "test_data_path": str(self.test_data_path),
            "directories_created": [],
            "config_applied": False
        }
        
        try:
            # Create necessary directories
            directories = [
                self.output_path,
                self.test_data_path,
                self.test_data_path / "context_learning",
                self.test_data_path / "chroma_db",
                self.test_data_path / "manim_docs",
                self.output_path / "videos",
                self.output_path / "code",
                self.output_path / "logs"
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                setup_results["directories_created"].append(str(directory))
            
            # Apply configuration
            config = self.get_workflow_config()
            setup_results["config_applied"] = True
            setup_results["config"] = {
                "planner_model": f"{config.planner_model.provider}/{config.planner_model.model_name}",
                "code_model": f"{config.code_model.provider}/{config.code_model.model_name}",
                "max_retries": config.max_retries,
                "preview_mode": config.preview_mode,
                "output_dir": config.output_dir
            }
            
            logger.info("Studio environment setup completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup Studio environment: {e}")
            setup_results["error"] = str(e)
        
        return setup_results
    
    def get_studio_info(self) -> Dict[str, Any]:
        """Get information about the Studio environment."""
        return {
            "studio_enabled": self.studio_enabled,
            "workspace_path": str(self.workspace_path),
            "output_path": str(self.output_path),
            "test_data_path": str(self.test_data_path),
            "environment_variables": {
                "LANGGRAPH_STUDIO": os.getenv("LANGGRAPH_STUDIO"),
                "LANGSMITH_TRACING": os.getenv("LANGSMITH_TRACING"),
                "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2"),
                "STUDIO_OUTPUT_DIR": os.getenv("STUDIO_OUTPUT_DIR"),
                "STUDIO_TEST_DATA_DIR": os.getenv("STUDIO_TEST_DATA_DIR")
            },
            "paths_exist": {
                "workspace": self.workspace_path.exists(),
                "output": self.output_path.exists(),
                "test_data": self.test_data_path.exists()
            }
        }


def create_studio_langgraph_config() -> Dict[str, Any]:
    """Create LangGraph configuration optimized for Studio testing."""
    return {
        "dependencies": ["."],
        "graphs": {
            # Individual agent graphs
            "planning_agent_test": "./src/langgraph_agents/studio/studio_graphs.py:planning_agent_graph",
            "code_generation_agent_test": "./src/langgraph_agents/studio/studio_graphs.py:code_generation_agent_graph",
            "rendering_agent_test": "./src/langgraph_agents/studio/studio_graphs.py:rendering_agent_graph",
            "error_handler_agent_test": "./src/langgraph_agents/studio/studio_graphs.py:error_handler_agent_graph",
            
            # Agent chain graphs
            "planning_to_code_chain": "./src/langgraph_agents/studio/studio_graphs.py:planning_to_code_chain_graph",
            "code_to_rendering_chain": "./src/langgraph_agents/studio/studio_graphs.py:code_to_rendering_chain_graph",
            "full_agent_chain": "./src/langgraph_agents/studio/studio_graphs.py:full_agent_chain_graph",
            
            # Monitored graphs
            "monitored_planning": "./src/langgraph_agents/studio/studio_graphs.py:monitored_planning_graph",
            "monitored_code_generation": "./src/langgraph_agents/studio/studio_graphs.py:monitored_code_generation_graph",
            "monitored_rendering": "./src/langgraph_agents/studio/studio_graphs.py:monitored_rendering_graph",
            
            # Original workflow for comparison
            "full_workflow": "./src/langgraph_agents/full_workflow.py:graph"
        },
        "env": ".env",
        "python_version": "3.11",
        "dockerfile_lines": [
            "RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 libxrender-dev libglib2.0-0",
            "RUN pip install manim opencv-python-headless pillow",
            "RUN mkdir -p /app/studio_output /app/studio_test_data /app/output /app/media /app/code"
        ],
        "store": {
            "index": {
                "embed": "openai:text-embedding-3-small",
                "dims": 1536,
                "fields": ["$"]
            },
            "ttl": {
                "refresh_on_read": True,
                "default_ttl": 720,  # Shorter TTL for testing
                "sweep_interval_minutes": 30
            }
        },
        "checkpointer": {
            "ttl": {
                "strategy": "delete",
                "sweep_interval_minutes": 15,  # More frequent cleanup for testing
                "default_ttl": 1440
            }
        },
        "http": {
            "cors": {
                "allow_origins": ["*"],
                "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": ["*"]
            },
            "configurable_headers": {
                "include": ["x-session-id", "x-trace-id", "x-test-scenario", "authorization"]
            }
        }
    }


def save_studio_langgraph_config(output_path: Optional[Path] = None) -> Path:
    """Save the Studio LangGraph configuration to a file."""
    if output_path is None:
        output_path = Path("langgraph.studio.json")
    
    config = create_studio_langgraph_config()
    
    with open(output_path, 'w') as f:
        import json
        json.dump(config, f, indent=2)
    
    logger.info(f"Studio LangGraph configuration saved to: {output_path}")
    return output_path


# Global Studio configuration instance
studio_config = StudioConfig()


def get_studio_config() -> StudioConfig:
    """Get the global Studio configuration instance."""
    return studio_config


def is_studio_environment() -> bool:
    """Check if currently running in Studio environment."""
    return studio_config.studio_enabled


def setup_studio_logging():
    """Set up logging configuration optimized for Studio."""
    if studio_config.studio_enabled:
        # Configure logging for Studio environment
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(studio_config.output_path / "studio.log")
            ]
        )
        
        # Set specific log levels for different components
        logging.getLogger("langgraph").setLevel(logging.INFO)
        logging.getLogger("langchain").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        
        logger.info("Studio logging configured")


# Initialize Studio environment on import
if studio_config.studio_enabled:
    setup_studio_logging()
    studio_config.setup_studio_environment()
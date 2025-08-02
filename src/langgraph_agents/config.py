"""
Agent configuration system with support for AWS Bedrock and OpenAI.
Maintains compatibility with existing model configurations.

DEPRECATED: This module is being replaced by the centralized configuration system.
Use src.config.manager.ConfigurationManager instead.
"""

import os
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import warnings

from .state import AgentConfig, SystemConfig


logger = logging.getLogger(__name__)


class ConfigurationManager:
    """Manages agent and system configuration with persistence.
    
    Maintains compatibility with existing configuration patterns while
    adding support for multi-agent system settings.
    """
    
    def __init__(self, config_dir: str = "config"):
        """Initialize configuration manager.
        
        Args:
            config_dir: Directory for configuration files
        """
        warnings.warn(
            "src.langgraph_agents.config.ConfigurationManager is deprecated. "
            "Use src.config.manager.ConfigurationManager instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Import and delegate to centralized configuration manager
        from src.config.manager import ConfigurationManager as CentralizedConfigManager
        self._centralized_manager = CentralizedConfigManager()
        
        # Keep old paths for backward compatibility
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Configuration file paths
        self.system_config_path = self.config_dir / "system_config.json"
        self.agents_config_path = self.config_dir / "agents_config.json"
        self.providers_config_path = self.config_dir / "providers_config.json"
        
        logger.warning(f"DEPRECATED: ConfigurationManager initialized with config_dir: {config_dir}. Use centralized config manager instead.")
    
    def create_default_system_config(self) -> SystemConfig:
        """Create default system configuration.
        
        Returns:
            SystemConfig: Default system configuration
        """
        # Default agent configurations compatible with existing patterns
        default_agents = {
            "planner_agent": AgentConfig(
                name="planner_agent",
                model_config={
                    "temperature": 0.7,
                    "max_tokens": 4000,
                    "timeout": 300
                },
                tools=["scene_planning", "plugin_detection"],
                max_retries=3,
                timeout_seconds=300,
                enable_human_loop=False,
                planner_model="openai/gpt-4o",
                helper_model="openai/gpt-4o-mini",
                temperature=0.7,
                print_cost=True,
                verbose=False
            ),
            "code_generator_agent": AgentConfig(
                name="code_generator_agent",
                model_config={
                    "temperature": 0.7,
                    "max_tokens": 4000,
                    "timeout": 300
                },
                tools=["code_generation", "error_fixing", "rag_query"],
                max_retries=5,
                timeout_seconds=600,
                enable_human_loop=True,
                scene_model="openai/gpt-4o",
                helper_model="openai/gpt-4o-mini",
                temperature=0.7,
                print_cost=True,
                verbose=False
            ),
            "renderer_agent": AgentConfig(
                name="renderer_agent",
                model_config={
                    "temperature": 0.3,
                    "max_tokens": 2000,
                    "timeout": 600
                },
                tools=["video_rendering", "optimization"],
                max_retries=3,
                timeout_seconds=1200,
                enable_human_loop=False,
                temperature=0.3,
                print_cost=True,
                verbose=False
            ),
            "visual_analysis_agent": AgentConfig(
                name="visual_analysis_agent",
                model_config={
                    "temperature": 0.5,
                    "max_tokens": 3000,
                    "timeout": 300
                },
                tools=["visual_analysis", "error_detection"],
                max_retries=3,
                timeout_seconds=300,
                enable_human_loop=True,
                helper_model="openai/gpt-4o",
                temperature=0.5,
                print_cost=True,
                verbose=False
            ),
            "rag_agent": AgentConfig(
                name="rag_agent",
                model_config={
                    "temperature": 0.3,
                    "max_tokens": 2000,
                    "timeout": 180
                },
                tools=["rag_query", "context_retrieval", "document_processing"],
                max_retries=2,
                timeout_seconds=180,
                enable_human_loop=False,
                helper_model="openai/gpt-4o-mini",
                temperature=0.3,
                print_cost=True,
                verbose=False
            ),
            "error_handler_agent": AgentConfig(
                name="error_handler_agent",
                model_config={
                    "temperature": 0.3,
                    "max_tokens": 2000,
                    "timeout": 120
                },
                tools=["error_classification", "recovery_routing"],
                max_retries=1,
                timeout_seconds=120,
                enable_human_loop=True,
                helper_model="openai/gpt-4o-mini",
                temperature=0.3,
                print_cost=True,
                verbose=False
            ),
            "monitoring_agent": AgentConfig(
                name="monitoring_agent",
                model_config={
                    "temperature": 0.1,
                    "max_tokens": 1000,
                    "timeout": 60
                },
                tools=["performance_monitoring", "diagnostics"],
                max_retries=1,
                timeout_seconds=60,
                enable_human_loop=False,
                temperature=0.1,
                print_cost=False,
                verbose=False
            ),
            "human_loop_agent": AgentConfig(
                name="human_loop_agent",
                model_config={
                    "temperature": 0.5,
                    "max_tokens": 1500,
                    "timeout": 30
                },
                tools=["human_interaction", "decision_presentation"],
                max_retries=1,
                timeout_seconds=30,
                enable_human_loop=True,
                temperature=0.5,
                print_cost=False,
                verbose=True
            )
        }
        
        # Default LLM provider configurations
        default_providers = {
            "openai": {
                "api_key_env": "OPENAI_API_KEY",
                "base_url": None,
                "models": [
                    "openai/gpt-4o",
                    "openai/gpt-4o-mini",
                    "openai/gpt-4",
                    "openai/gpt-3.5-turbo"
                ],
                "default_model": "openai/gpt-4o"
            },
            "aws_bedrock": {
                "region": "us-east-1",
                "models": [
                    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
                    "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
                    "bedrock/amazon.titan-text-premier-v1:0"
                ],
                "default_model": "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
            },
            "openrouter": {
                "api_key_env": "OPENROUTER_API_KEY",
                "base_url": "https://openrouter.ai/api/v1",
                "models": [
                    "openrouter/openai/gpt-4o",
                    "openrouter/openai/gpt-4o-mini",
                    "openrouter/anthropic/claude-3.5-sonnet",
                    "openrouter/anthropic/claude-3-haiku"
                ],
                "default_model": "openrouter/openai/gpt-4o"
            }
        }
        
        # External tool configurations
        default_docling_config = {
            "enabled": True,
            "max_file_size_mb": 50,
            "supported_formats": ["pdf", "docx", "txt", "md"],
            "timeout_seconds": 120
        }
        
        default_mcp_servers = {
            "context7": {
                "command": "uvx",
                "args": ["context7-mcp-server@latest"],
                "env": {"FASTMCP_LOG_LEVEL": "ERROR"},
                "disabled": False,
                "autoApprove": ["resolve_library_id", "get_library_docs"]
            },
            "docling": {
                "command": "uvx", 
                "args": ["docling-mcp-server@latest"],
                "env": {"FASTMCP_LOG_LEVEL": "ERROR"},
                "disabled": False,
                "autoApprove": ["process_document"]
            }
        }
        
        default_context7_config = {
            "enabled": True,
            "default_tokens": 10000,
            "timeout_seconds": 30,
            "cache_responses": True,
            "cache_ttl": 3600
        }
        
        default_monitoring_config = {
            "enabled": True,
            "langfuse_enabled": True,
            "performance_tracking": True,
            "error_tracking": True,
            "execution_tracing": True,
            "langfuse_config": {
                "enabled": True,
                "debug": False,
                "host": "https://cloud.langfuse.com",
                "flush_interval": 10,
                "max_retries": 3
            }
        }
        
        default_human_loop_config = {
            "enabled": True,
            "enable_interrupts": True,
            "timeout_seconds": 300,
            "auto_approve_low_risk": False
        }
        
        return SystemConfig(
            agents=default_agents,
            llm_providers=default_providers,
            docling_config=default_docling_config,
            mcp_servers=default_mcp_servers,
            monitoring_config=default_monitoring_config,
            human_loop_config=default_human_loop_config,
            max_workflow_retries=3,
            workflow_timeout_seconds=3600,
            enable_checkpoints=True,
            checkpoint_interval=300
        )
    
    def load_system_config(self) -> SystemConfig:
        """Load system configuration from file or create default.
        
        Returns:
            SystemConfig: Loaded or default system configuration
        """
        logger.warning("DEPRECATED: Use centralized ConfigurationManager.config instead")
        
        try:
            # Try to use centralized configuration first
            centralized_config = self._centralized_manager.config
            
            # Convert centralized config to old SystemConfig format for backward compatibility
            agents = {}
            for name, agent_config in centralized_config.agent_configs.items():
                # Convert new AgentConfig to old format
                agents[name] = AgentConfig(
                    name=agent_config.name,
                    model_config=agent_config.model_config or {},
                    tools=agent_config.tools or [],
                    max_retries=agent_config.max_retries,
                    timeout_seconds=agent_config.timeout_seconds,
                    enable_human_loop=agent_config.enable_human_loop,
                    planner_model=getattr(agent_config, 'planner_model', None),
                    scene_model=getattr(agent_config, 'scene_model', None),
                    helper_model=getattr(agent_config, 'helper_model', None),
                    temperature=getattr(agent_config, 'temperature', 0.7),
                    print_cost=getattr(agent_config, 'print_cost', True),
                    verbose=getattr(agent_config, 'verbose', False)
                )
            
            # Convert LLM providers to old format
            llm_providers = {}
            for name, provider_config in centralized_config.llm_providers.items():
                llm_providers[name] = {
                    'api_key_env': f"{name.upper()}_API_KEY",
                    'base_url': provider_config.base_url,
                    'models': provider_config.models,
                    'default_model': provider_config.default_model
                }
            
            config = SystemConfig(
                agents=agents,
                llm_providers=llm_providers,
                docling_config=centralized_config.docling_config.__dict__ if centralized_config.docling_config else {},
                mcp_servers=centralized_config.mcp_servers,
                monitoring_config=centralized_config.monitoring_config.__dict__ if centralized_config.monitoring_config else {},
                human_loop_config=centralized_config.human_loop_config.__dict__ if centralized_config.human_loop_config else {},
                max_workflow_retries=centralized_config.workflow_config.max_workflow_retries if centralized_config.workflow_config else 3,
                workflow_timeout_seconds=centralized_config.workflow_config.workflow_timeout_seconds if centralized_config.workflow_config else 3600,
                enable_checkpoints=True,
                checkpoint_interval=300
            )
            
            logger.info("Loaded system configuration from centralized manager")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load from centralized config: {e}")
            # Fallback to old file-based loading
            return self._load_from_file_fallback()
    
    def _load_from_file_fallback(self) -> SystemConfig:
        """Fallback method to load from file if centralized config fails."""
        try:
            if self.system_config_path.exists():
                with open(self.system_config_path, 'r') as f:
                    config_dict = json.load(f)
                
                # Convert agent configs
                agents = {}
                for name, agent_dict in config_dict.get('agents', {}).items():
                    agents[name] = AgentConfig(**agent_dict)
                
                config = SystemConfig(
                    agents=agents,
                    llm_providers=config_dict.get('llm_providers', {}),
                    docling_config=config_dict.get('docling_config', {}),
                    mcp_servers=config_dict.get('mcp_servers', {}),
                    monitoring_config=config_dict.get('monitoring_config', {}),
                    human_loop_config=config_dict.get('human_loop_config', {}),
                    max_workflow_retries=config_dict.get('max_workflow_retries', 3),
                    workflow_timeout_seconds=config_dict.get('workflow_timeout_seconds', 3600),
                    enable_checkpoints=config_dict.get('enable_checkpoints', True),
                    checkpoint_interval=config_dict.get('checkpoint_interval', 300)
                )
                
                logger.info("Loaded system configuration from file (fallback)")
                return config
                
            else:
                logger.info("No system configuration file found, creating default")
                config = self.create_default_system_config()
                self.save_system_config(config)
                return config
                
        except Exception as e:
            logger.error(f"Failed to load system configuration: {e}")
            logger.info("Falling back to default configuration")
            return self.create_default_system_config()
    
    def save_system_config(self, config: SystemConfig) -> bool:
        """Save system configuration to file.
        
        Args:
            config: System configuration to save
            
        Returns:
            bool: True if saved successfully
        """
        try:
            # Convert to serializable format
            config_dict = {
                'agents': {name: asdict(agent) for name, agent in config.agents.items()},
                'llm_providers': config.llm_providers,
                'docling_config': config.docling_config,
                'mcp_servers': config.mcp_servers,
                'monitoring_config': config.monitoring_config,
                'human_loop_config': config.human_loop_config,
                'max_workflow_retries': config.max_workflow_retries,
                'workflow_timeout_seconds': config.workflow_timeout_seconds,
                'enable_checkpoints': config.enable_checkpoints,
                'checkpoint_interval': config.checkpoint_interval
            }
            
            with open(self.system_config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info("System configuration saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save system configuration: {e}")
            return False
    
    def update_agent_config(self, agent_name: str, updates: Dict[str, Any]) -> bool:
        """Update configuration for a specific agent.
        
        Args:
            agent_name: Name of agent to update
            updates: Configuration updates
            
        Returns:
            bool: True if updated successfully
        """
        try:
            config = self.load_system_config()
            
            if agent_name not in config.agents:
                logger.error(f"Agent {agent_name} not found in configuration")
                return False
            
            # Update agent configuration
            agent_config = config.agents[agent_name]
            for key, value in updates.items():
                if hasattr(agent_config, key):
                    setattr(agent_config, key, value)
                else:
                    logger.warning(f"Unknown configuration key: {key}")
            
            # Save updated configuration
            return self.save_system_config(config)
            
        except Exception as e:
            logger.error(f"Failed to update agent configuration: {e}")
            return False
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict: Model configuration
        """
        logger.warning("DEPRECATED: Use centralized ConfigurationManager.get_model_config instead")
        
        try:
            # Delegate to centralized configuration manager
            return self._centralized_manager.get_model_config(model_name)
        except Exception as e:
            logger.error(f"Failed to get model config from centralized manager: {e}")
            # Fallback to old implementation
            return self._get_model_config_fallback(model_name)
    
    def _get_model_config_fallback(self, model_name: str) -> Dict[str, Any]:
        """Fallback method for getting model config."""
        config = self.load_system_config()
        
        # Determine provider from model name
        if model_name.startswith('openai/'):
            provider_config = config.llm_providers.get('openai', {})
        elif model_name.startswith('bedrock/'):
            provider_config = config.llm_providers.get('aws_bedrock', {})
        elif model_name.startswith('openrouter/'):
            provider_config = config.llm_providers.get('openrouter', {})
        else:
            # Default to OpenAI configuration
            provider_config = config.llm_providers.get('openai', {})
        
        return {
            'model_name': model_name,
            'provider_config': provider_config,
            'api_key': os.getenv(provider_config.get('api_key_env', 'OPENAI_API_KEY')),
            'base_url': provider_config.get('base_url'),
            'region': provider_config.get('region')
        }
    
    def validate_configuration(self, config: SystemConfig) -> List[str]:
        """Validate system configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List[str]: List of validation errors
        """
        errors = []
        
        # Validate required agents
        required_agents = ['planner_agent', 'code_generator_agent', 'renderer_agent']
        for agent_name in required_agents:
            if agent_name not in config.agents:
                errors.append(f"Required agent missing: {agent_name}")
        
        # Validate agent configurations
        for agent_name, agent_config in config.agents.items():
            if not agent_config.name:
                errors.append(f"Agent {agent_name} missing name")
            
            if not agent_config.model_config:
                errors.append(f"Agent {agent_name} missing model_config")
            
            if agent_config.max_retries < 0:
                errors.append(f"Agent {agent_name} has invalid max_retries")
            
            if agent_config.timeout_seconds <= 0:
                errors.append(f"Agent {agent_name} has invalid timeout_seconds")
        
        # Validate LLM provider configurations
        for provider_name, provider_config in config.llm_providers.items():
            if provider_name == 'openai' and 'api_key_env' not in provider_config:
                errors.append(f"OpenAI provider missing api_key_env")
            
            if provider_name == 'aws_bedrock' and 'region' not in provider_config:
                errors.append(f"AWS Bedrock provider missing region")
        
        # Validate workflow settings
        if config.max_workflow_retries < 0:
            errors.append("Invalid max_workflow_retries")
        
        if config.workflow_timeout_seconds <= 0:
            errors.append("Invalid workflow_timeout_seconds")
        
        return errors
    
    def get_compatible_initialization_params(self, state_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get initialization parameters compatible with existing system.
        
        Args:
            state_config: State configuration
            
        Returns:
            Dict: Compatible initialization parameters
        """
        return {
            'output_dir': state_config.get('output_dir', 'output'),
            'print_response': state_config.get('print_response', False),
            'use_rag': state_config.get('use_rag', True),
            'use_context_learning': state_config.get('use_context_learning', True),
            'context_learning_path': state_config.get('context_learning_path', 'data/context_learning'),
            'chroma_db_path': state_config.get('chroma_db_path', 'data/rag/chroma_db'),
            'manim_docs_path': state_config.get('manim_docs_path', 'data/rag/manim_docs'),
            'embedding_model': state_config.get('embedding_model', 'hf:ibm-granite/granite-embedding-30m-english'),
            'use_visual_fix_code': state_config.get('use_visual_fix_code', False),
            'use_langfuse': state_config.get('use_langfuse', True),
            'max_scene_concurrency': state_config.get('max_scene_concurrency', 5),
            'max_topic_concurrency': state_config.get('max_topic_concurrency', 1),
            'max_retries': state_config.get('max_retries', 5),
            'enable_caching': state_config.get('enable_caching', True),
            'default_quality': state_config.get('default_quality', 'medium'),
            'use_gpu_acceleration': state_config.get('use_gpu_acceleration', False),
            'preview_mode': state_config.get('preview_mode', False),
            'max_concurrent_renders': state_config.get('max_concurrent_renders', 4)
        }
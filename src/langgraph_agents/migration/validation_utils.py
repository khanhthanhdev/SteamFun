"""
Configuration validation utilities for ensuring migrated configurations are valid.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass

from ..state import SystemConfig, AgentConfig


logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Represents a configuration validation error."""
    severity: str  # 'error', 'warning', 'info'
    component: str
    parameter: str
    message: str
    suggestion: Optional[str] = None


class ConfigurationValidator:
    """Validates migrated configurations for correctness and completeness."""
    
    def __init__(self):
        """Initialize configuration validator."""
        self.required_agents = {
            'planner_agent',
            'code_generator_agent', 
            'renderer_agent'
        }
        
        self.optional_agents = {
            'visual_analysis_agent',
            'rag_agent',
            'error_handler_agent',
            'monitoring_agent',
            'human_loop_agent'
        }
        
        self.required_llm_providers = {
            'openai'
        }
        
        self.optional_llm_providers = {
            'aws_bedrock',
            'openrouter'
        }
        
        logger.info("ConfigurationValidator initialized")
    
    def validate_system_config(self, config: SystemConfig) -> Tuple[bool, List[ValidationError]]:
        """Validate complete system configuration.
        
        Args:
            config: System configuration to validate
            
        Returns:
            Tuple[bool, List[ValidationError]]: (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate agents
        agent_errors = self._validate_agents(config.agents)
        errors.extend(agent_errors)
        
        # Validate LLM providers
        provider_errors = self._validate_llm_providers(config.llm_providers)
        errors.extend(provider_errors)
        
        # Validate external services
        service_errors = self._validate_external_services(config)
        errors.extend(service_errors)
        
        # Validate workflow settings
        workflow_errors = self._validate_workflow_settings(config)
        errors.extend(workflow_errors)
        
        # Validate environment dependencies
        env_errors = self._validate_environment_dependencies(config)
        errors.extend(env_errors)
        
        # Check for critical errors
        critical_errors = [e for e in errors if e.severity == 'error']
        is_valid = len(critical_errors) == 0
        
        logger.info(f"System configuration validation: "
                   f"{'Valid' if is_valid else 'Invalid'} "
                   f"({len(critical_errors)} errors, "
                   f"{len([e for e in errors if e.severity == 'warning'])} warnings)")
        
        return is_valid, errors
    
    def _validate_agents(self, agents: Dict[str, AgentConfig]) -> List[ValidationError]:
        """Validate agent configurations.
        
        Args:
            agents: Agent configurations to validate
            
        Returns:
            List[ValidationError]: Validation errors
        """
        errors = []
        
        # Check for required agents
        for required_agent in self.required_agents:
            if required_agent not in agents:
                errors.append(ValidationError(
                    severity='error',
                    component='agents',
                    parameter=required_agent,
                    message=f"Required agent missing: {required_agent}",
                    suggestion=f"Add {required_agent} configuration to agents section"
                ))
        
        # Validate individual agent configurations
        for agent_name, agent_config in agents.items():
            agent_errors = self._validate_single_agent(agent_name, agent_config)
            errors.extend(agent_errors)
        
        return errors
    
    def _validate_single_agent(self, agent_name: str, agent_config: AgentConfig) -> List[ValidationError]:
        """Validate a single agent configuration.
        
        Args:
            agent_name: Name of the agent
            agent_config: Agent configuration
            
        Returns:
            List[ValidationError]: Validation errors
        """
        errors = []
        
        # Validate basic properties
        if not agent_config.name:
            errors.append(ValidationError(
                severity='error',
                component=agent_name,
                parameter='name',
                message="Agent name is required",
                suggestion="Set agent name property"
            ))
        
        if not agent_config.model_config:
            errors.append(ValidationError(
                severity='error',
                component=agent_name,
                parameter='model_config',
                message="Model configuration is required",
                suggestion="Add model_config with temperature, max_tokens, and timeout"
            ))
        else:
            # Validate model configuration
            model_errors = self._validate_model_config(agent_name, agent_config.model_config)
            errors.extend(model_errors)
        
        # Validate numeric parameters
        if agent_config.max_retries < 0:
            errors.append(ValidationError(
                severity='error',
                component=agent_name,
                parameter='max_retries',
                message="max_retries must be non-negative",
                suggestion="Set max_retries to a value >= 0"
            ))
        
        if agent_config.max_retries > 10:
            errors.append(ValidationError(
                severity='warning',
                component=agent_name,
                parameter='max_retries',
                message="max_retries is very high, may cause long delays",
                suggestion="Consider reducing max_retries to 3-5"
            ))
        
        if agent_config.timeout_seconds <= 0:
            errors.append(ValidationError(
                severity='error',
                component=agent_name,
                parameter='timeout_seconds',
                message="timeout_seconds must be positive",
                suggestion="Set timeout_seconds to a value > 0"
            ))
        
        if agent_config.timeout_seconds > 3600:
            errors.append(ValidationError(
                severity='warning',
                component=agent_name,
                parameter='timeout_seconds',
                message="timeout_seconds is very high (>1 hour)",
                suggestion="Consider reducing timeout_seconds"
            ))
        
        # Validate agent-specific models
        model_errors = self._validate_agent_models(agent_name, agent_config)
        errors.extend(model_errors)
        
        # Validate tools
        if not agent_config.tools:
            errors.append(ValidationError(
                severity='warning',
                component=agent_name,
                parameter='tools',
                message="No tools configured for agent",
                suggestion="Add appropriate tools for agent functionality"
            ))
        
        return errors
    
    def _validate_model_config(self, agent_name: str, model_config: Dict[str, Any]) -> List[ValidationError]:
        """Validate model configuration.
        
        Args:
            agent_name: Name of the agent
            model_config: Model configuration
            
        Returns:
            List[ValidationError]: Validation errors
        """
        errors = []
        
        required_keys = ['temperature', 'max_tokens', 'timeout']
        for key in required_keys:
            if key not in model_config:
                errors.append(ValidationError(
                    severity='error',
                    component=agent_name,
                    parameter=f'model_config.{key}',
                    message=f"Required model config key missing: {key}",
                    suggestion=f"Add {key} to model_config"
                ))
        
        # Validate temperature
        if 'temperature' in model_config:
            temp = model_config['temperature']
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                errors.append(ValidationError(
                    severity='error',
                    component=agent_name,
                    parameter='model_config.temperature',
                    message="Temperature must be a number between 0 and 2",
                    suggestion="Set temperature to a value between 0.0 and 2.0"
                ))
        
        # Validate max_tokens
        if 'max_tokens' in model_config:
            max_tokens = model_config['max_tokens']
            if not isinstance(max_tokens, int) or max_tokens <= 0:
                errors.append(ValidationError(
                    severity='error',
                    component=agent_name,
                    parameter='model_config.max_tokens',
                    message="max_tokens must be a positive integer",
                    suggestion="Set max_tokens to a positive integer (e.g., 4000)"
                ))
            elif max_tokens > 32000:
                errors.append(ValidationError(
                    severity='warning',
                    component=agent_name,
                    parameter='model_config.max_tokens',
                    message="max_tokens is very high, may be expensive",
                    suggestion="Consider reducing max_tokens if not needed"
                ))
        
        # Validate timeout
        if 'timeout' in model_config:
            timeout = model_config['timeout']
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                errors.append(ValidationError(
                    severity='error',
                    component=agent_name,
                    parameter='model_config.timeout',
                    message="timeout must be a positive number",
                    suggestion="Set timeout to a positive number in seconds"
                ))
        
        return errors
    
    def _validate_agent_models(self, agent_name: str, agent_config: AgentConfig) -> List[ValidationError]:
        """Validate agent-specific model assignments.
        
        Args:
            agent_name: Name of the agent
            agent_config: Agent configuration
            
        Returns:
            List[ValidationError]: Validation errors
        """
        errors = []
        
        # Check required models for specific agents
        if agent_name == 'planner_agent':
            if not hasattr(agent_config, 'planner_model') or not agent_config.planner_model:
                errors.append(ValidationError(
                    severity='error',
                    component=agent_name,
                    parameter='planner_model',
                    message="planner_model is required for planner_agent",
                    suggestion="Set planner_model to a valid model name"
                ))
        
        if agent_name == 'code_generator_agent':
            if not hasattr(agent_config, 'scene_model') or not agent_config.scene_model:
                errors.append(ValidationError(
                    severity='error',
                    component=agent_name,
                    parameter='scene_model',
                    message="scene_model is required for code_generator_agent",
                    suggestion="Set scene_model to a valid model name"
                ))
        
        # Validate model name formats
        model_attrs = ['planner_model', 'scene_model', 'helper_model']
        for attr in model_attrs:
            if hasattr(agent_config, attr):
                model_name = getattr(agent_config, attr)
                if model_name and not self._is_valid_model_name(model_name):
                    errors.append(ValidationError(
                        severity='error',
                        component=agent_name,
                        parameter=attr,
                        message=f"Invalid model name format: {model_name}",
                        suggestion="Use format 'provider/model' (e.g., 'openai/gpt-4o')"
                    ))
        
        return errors
    
    def _validate_llm_providers(self, providers: Dict[str, Dict[str, Any]]) -> List[ValidationError]:
        """Validate LLM provider configurations.
        
        Args:
            providers: LLM provider configurations
            
        Returns:
            List[ValidationError]: Validation errors
        """
        errors = []
        
        # Check for required providers
        for required_provider in self.required_llm_providers:
            if required_provider not in providers:
                errors.append(ValidationError(
                    severity='error',
                    component='llm_providers',
                    parameter=required_provider,
                    message=f"Required LLM provider missing: {required_provider}",
                    suggestion=f"Add {required_provider} provider configuration"
                ))
        
        # Validate individual provider configurations
        for provider_name, provider_config in providers.items():
            provider_errors = self._validate_single_provider(provider_name, provider_config)
            errors.extend(provider_errors)
        
        return errors
    
    def _validate_single_provider(self, provider_name: str, provider_config: Dict[str, Any]) -> List[ValidationError]:
        """Validate a single LLM provider configuration.
        
        Args:
            provider_name: Name of the provider
            provider_config: Provider configuration
            
        Returns:
            List[ValidationError]: Validation errors
        """
        errors = []
        
        if provider_name == 'openai':
            if 'api_key_env' not in provider_config:
                errors.append(ValidationError(
                    severity='error',
                    component=provider_name,
                    parameter='api_key_env',
                    message="OpenAI provider requires api_key_env",
                    suggestion="Set api_key_env to 'OPENAI_API_KEY'"
                ))
            
            if 'models' not in provider_config or not provider_config['models']:
                errors.append(ValidationError(
                    severity='warning',
                    component=provider_name,
                    parameter='models',
                    message="No models configured for OpenAI provider",
                    suggestion="Add list of available OpenAI models"
                ))
        
        elif provider_name == 'aws_bedrock':
            if 'region' not in provider_config:
                errors.append(ValidationError(
                    severity='error',
                    component=provider_name,
                    parameter='region',
                    message="AWS Bedrock provider requires region",
                    suggestion="Set region to AWS region (e.g., 'us-east-1')"
                ))
            
            if 'models' not in provider_config or not provider_config['models']:
                errors.append(ValidationError(
                    severity='warning',
                    component=provider_name,
                    parameter='models',
                    message="No models configured for AWS Bedrock provider",
                    suggestion="Add list of available Bedrock models"
                ))
        
        elif provider_name == 'openrouter':
            if 'api_key_env' not in provider_config:
                errors.append(ValidationError(
                    severity='error',
                    component=provider_name,
                    parameter='api_key_env',
                    message="OpenRouter provider requires api_key_env",
                    suggestion="Set api_key_env to 'OPENROUTER_API_KEY'"
                ))
            
            if 'base_url' not in provider_config:
                errors.append(ValidationError(
                    severity='error',
                    component=provider_name,
                    parameter='base_url',
                    message="OpenRouter provider requires base_url",
                    suggestion="Set base_url to 'https://openrouter.ai/api/v1'"
                ))
        
        # Validate default model
        if 'default_model' in provider_config:
            default_model = provider_config['default_model']
            models = provider_config.get('models', [])
            if default_model not in models:
                errors.append(ValidationError(
                    severity='warning',
                    component=provider_name,
                    parameter='default_model',
                    message=f"Default model '{default_model}' not in models list",
                    suggestion="Add default model to models list or update default_model"
                ))
        
        return errors
    
    def _validate_external_services(self, config: SystemConfig) -> List[ValidationError]:
        """Validate external service configurations.
        
        Args:
            config: System configuration
            
        Returns:
            List[ValidationError]: Validation errors
        """
        errors = []
        
        # Validate Docling configuration
        if hasattr(config, 'docling_config') and config.docling_config:
            docling_errors = self._validate_docling_config(config.docling_config)
            errors.extend(docling_errors)
        
        # Validate MCP servers
        if hasattr(config, 'mcp_servers') and config.mcp_servers:
            mcp_errors = self._validate_mcp_servers(config.mcp_servers)
            errors.extend(mcp_errors)
        
        # Validate monitoring configuration
        if hasattr(config, 'monitoring_config') and config.monitoring_config:
            monitoring_errors = self._validate_monitoring_config(config.monitoring_config)
            errors.extend(monitoring_errors)
        
        return errors
    
    def _validate_docling_config(self, docling_config: Dict[str, Any]) -> List[ValidationError]:
        """Validate Docling configuration."""
        errors = []
        
        if 'max_file_size_mb' in docling_config:
            max_size = docling_config['max_file_size_mb']
            if not isinstance(max_size, (int, float)) or max_size <= 0:
                errors.append(ValidationError(
                    severity='error',
                    component='docling_config',
                    parameter='max_file_size_mb',
                    message="max_file_size_mb must be a positive number",
                    suggestion="Set max_file_size_mb to a positive number"
                ))
        
        if 'timeout_seconds' in docling_config:
            timeout = docling_config['timeout_seconds']
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                errors.append(ValidationError(
                    severity='error',
                    component='docling_config',
                    parameter='timeout_seconds',
                    message="timeout_seconds must be a positive number",
                    suggestion="Set timeout_seconds to a positive number"
                ))
        
        return errors
    
    def _validate_mcp_servers(self, mcp_servers: Dict[str, Dict[str, Any]]) -> List[ValidationError]:
        """Validate MCP server configurations."""
        errors = []
        
        for server_name, server_config in mcp_servers.items():
            if 'command' not in server_config:
                errors.append(ValidationError(
                    severity='error',
                    component='mcp_servers',
                    parameter=f'{server_name}.command',
                    message=f"MCP server '{server_name}' missing command",
                    suggestion="Add command field to MCP server configuration"
                ))
            
            if 'args' not in server_config:
                errors.append(ValidationError(
                    severity='warning',
                    component='mcp_servers',
                    parameter=f'{server_name}.args',
                    message=f"MCP server '{server_name}' missing args",
                    suggestion="Add args field to MCP server configuration"
                ))
        
        return errors
    
    def _validate_monitoring_config(self, monitoring_config: Dict[str, Any]) -> List[ValidationError]:
        """Validate monitoring configuration."""
        errors = []
        
        if 'langfuse_config' in monitoring_config:
            langfuse_config = monitoring_config['langfuse_config']
            
            if langfuse_config.get('enabled', False):
                required_keys = ['secret_key_env', 'public_key_env', 'host']
                for key in required_keys:
                    if key not in langfuse_config:
                        errors.append(ValidationError(
                            severity='error',
                            component='monitoring_config',
                            parameter=f'langfuse_config.{key}',
                            message=f"LangFuse config missing required key: {key}",
                            suggestion=f"Add {key} to langfuse_config"
                        ))
        
        return errors
    
    def _validate_workflow_settings(self, config: SystemConfig) -> List[ValidationError]:
        """Validate workflow settings.
        
        Args:
            config: System configuration
            
        Returns:
            List[ValidationError]: Validation errors
        """
        errors = []
        
        # Validate max_workflow_retries
        if config.max_workflow_retries < 0:
            errors.append(ValidationError(
                severity='error',
                component='workflow',
                parameter='max_workflow_retries',
                message="max_workflow_retries must be non-negative",
                suggestion="Set max_workflow_retries to 0 or higher"
            ))
        
        if config.max_workflow_retries > 10:
            errors.append(ValidationError(
                severity='warning',
                component='workflow',
                parameter='max_workflow_retries',
                message="max_workflow_retries is very high",
                suggestion="Consider reducing max_workflow_retries to avoid long delays"
            ))
        
        # Validate workflow_timeout_seconds
        if config.workflow_timeout_seconds <= 0:
            errors.append(ValidationError(
                severity='error',
                component='workflow',
                parameter='workflow_timeout_seconds',
                message="workflow_timeout_seconds must be positive",
                suggestion="Set workflow_timeout_seconds to a positive value"
            ))
        
        # Validate checkpoint_interval
        if config.checkpoint_interval <= 0:
            errors.append(ValidationError(
                severity='error',
                component='workflow',
                parameter='checkpoint_interval',
                message="checkpoint_interval must be positive",
                suggestion="Set checkpoint_interval to a positive value in seconds"
            ))
        
        if config.checkpoint_interval > config.workflow_timeout_seconds:
            errors.append(ValidationError(
                severity='warning',
                component='workflow',
                parameter='checkpoint_interval',
                message="checkpoint_interval is longer than workflow timeout",
                suggestion="Set checkpoint_interval to be less than workflow_timeout_seconds"
            ))
        
        return errors
    
    def _validate_environment_dependencies(self, config: SystemConfig) -> List[ValidationError]:
        """Validate environment dependencies.
        
        Args:
            config: System configuration
            
        Returns:
            List[ValidationError]: Validation errors
        """
        errors = []
        
        # Check for required environment variables
        required_env_vars = set()
        
        # Collect required environment variables from LLM providers
        for provider_name, provider_config in config.llm_providers.items():
            if 'api_key_env' in provider_config:
                required_env_vars.add(provider_config['api_key_env'])
        
        # Check if environment variables are set
        for env_var in required_env_vars:
            if not os.getenv(env_var):
                errors.append(ValidationError(
                    severity='warning',
                    component='environment',
                    parameter=env_var,
                    message=f"Environment variable not set: {env_var}",
                    suggestion=f"Set {env_var} environment variable"
                ))
        
        # Validate paths exist
        path_configs = []
        
        # Collect paths from various configurations
        if hasattr(config, 'rag_config') and config.rag_config:
            if 'docs_config' in config.rag_config:
                docs_config = config.rag_config['docs_config']
                if 'manim_docs_path' in docs_config:
                    path_configs.append(('rag_config.docs_config.manim_docs_path', docs_config['manim_docs_path']))
                if 'context_learning_path' in docs_config:
                    path_configs.append(('rag_config.docs_config.context_learning_path', docs_config['context_learning_path']))
        
        # Check if paths exist
        for param_name, path_str in path_configs:
            path = Path(path_str)
            if not path.exists():
                errors.append(ValidationError(
                    severity='warning',
                    component='environment',
                    parameter=param_name,
                    message=f"Path does not exist: {path}",
                    suggestion=f"Create directory or update path: {path}"
                ))
        
        return errors
    
    def _is_valid_model_name(self, model_name: str) -> bool:
        """Check if model name has valid format.
        
        Args:
            model_name: Model name to validate
            
        Returns:
            bool: True if valid format
        """
        if not isinstance(model_name, str):
            return False
        
        # Check for provider/model format or bedrock/ prefix
        return ('/' in model_name and not model_name.startswith('/')) or model_name.startswith('bedrock/')
    
    def generate_validation_report(self, errors: List[ValidationError]) -> str:
        """Generate a human-readable validation report.
        
        Args:
            errors: List of validation errors
            
        Returns:
            str: Validation report in Markdown format
        """
        if not errors:
            return "# Configuration Validation Report\n\n✅ **All validations passed!**\n\nYour configuration is valid and ready to use.\n"
        
        # Group errors by severity
        error_groups = {
            'error': [e for e in errors if e.severity == 'error'],
            'warning': [e for e in errors if e.severity == 'warning'],
            'info': [e for e in errors if e.severity == 'info']
        }
        
        report = "# Configuration Validation Report\n\n"
        
        # Summary
        total_errors = len(error_groups['error'])
        total_warnings = len(error_groups['warning'])
        total_info = len(error_groups['info'])
        
        if total_errors > 0:
            report += f"❌ **{total_errors} error(s)** - Configuration is invalid\n"
        else:
            report += "✅ **No errors** - Configuration is valid\n"
        
        if total_warnings > 0:
            report += f"⚠️ **{total_warnings} warning(s)** - Consider addressing these issues\n"
        
        if total_info > 0:
            report += f"ℹ️ **{total_info} info message(s)**\n"
        
        report += "\n"
        
        # Detailed errors
        for severity, severity_errors in error_groups.items():
            if not severity_errors:
                continue
            
            severity_icon = {'error': '❌', 'warning': '⚠️', 'info': 'ℹ️'}[severity]
            report += f"## {severity_icon} {severity.title()}s\n\n"
            
            for error in severity_errors:
                report += f"### {error.component}.{error.parameter}\n"
                report += f"**Message:** {error.message}\n\n"
                if error.suggestion:
                    report += f"**Suggestion:** {error.suggestion}\n\n"
                report += "---\n\n"
        
        return report
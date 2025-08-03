"""
Configuration Management CLI

Command-line utility for managing and validating configuration settings.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional

from .factory import ConfigurationFactory, ConfigurationError
from .settings import Environment
from .validation import validate_configuration, check_required_environment_variables


def validate_config_command(args) -> int:
    """Validate configuration for the specified environment"""
    try:
        environment = Environment(args.environment) if args.environment else None
        settings = ConfigurationFactory.create_settings(
            environment=environment,
            env_file=args.env_file,
            validate_required=False  # We'll do our own validation
        )
        
        print(f"Validating configuration for environment: {settings.app.environment.value}")
        print("-" * 50)
        
        # Check required environment variables
        env_result = check_required_environment_variables(settings.app.environment)
        if not env_result.is_valid:
            print("Environment Variable Check:")
            print(env_result.get_summary())
            print()
        
        # Validate configuration
        config_result = validate_configuration(settings)
        print("Configuration Validation:")
        print(config_result.get_summary())
        
        # Return appropriate exit code
        if env_result.is_valid and config_result.is_valid:
            print("\n✅ Configuration validation passed!")
            return 0
        else:
            print("\n❌ Configuration validation failed!")
            return 1
            
    except ConfigurationError as e:
        print(f"❌ Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


def show_config_command(args) -> int:
    """Show current configuration settings"""
    try:
        environment = Environment(args.environment) if args.environment else None
        settings = ConfigurationFactory.create_settings(
            environment=environment,
            env_file=args.env_file,
            validate_required=False
        )
        
        print(f"Configuration for environment: {settings.app.environment.value}")
        print("=" * 60)
        
        if args.section:
            # Show specific section
            section_obj = getattr(settings, args.section, None)
            if section_obj is None:
                print(f"❌ Unknown configuration section: {args.section}")
                return 1
            
            print(f"\n[{args.section.upper()}]")
            section_dict = section_obj.dict()
            for key, value in section_dict.items():
                # Mask sensitive values
                if any(sensitive in key.lower() for sensitive in ['key', 'secret', 'password', 'token']):
                    value = "***MASKED***" if value else None
                print(f"{key}: {value}")
        else:
            # Show all sections
            sections = ['app', 'database', 'llm', 'rag', 'tts', 'monitoring', 'security']
            for section_name in sections:
                section_obj = getattr(settings, section_name)
                print(f"\n[{section_name.upper()}]")
                section_dict = section_obj.dict()
                for key, value in section_dict.items():
                    # Mask sensitive values
                    if any(sensitive in key.lower() for sensitive in ['key', 'secret', 'password', 'token']):
                        value = "***MASKED***" if value else None
                    print(f"{key}: {value}")
        
        return 0
        
    except ConfigurationError as e:
        print(f"❌ Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


def export_config_command(args) -> int:
    """Export configuration to JSON file"""
    try:
        environment = Environment(args.environment) if args.environment else None
        settings = ConfigurationFactory.create_settings(
            environment=environment,
            env_file=args.env_file,
            validate_required=False
        )
        
        # Convert to dictionary
        config_dict = settings.dict()
        
        # Mask sensitive values if requested
        if not args.include_secrets:
            def mask_secrets(obj, path=""):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        current_path = f"{path}.{key}" if path else key
                        if any(sensitive in key.lower() for sensitive in ['key', 'secret', 'password', 'token']):
                            obj[key] = "***MASKED***" if value else None
                        elif isinstance(value, dict):
                            mask_secrets(value, current_path)
                        elif isinstance(value, list):
                            for i, item in enumerate(value):
                                if isinstance(item, dict):
                                    mask_secrets(item, f"{current_path}[{i}]")
            
            mask_secrets(config_dict)
        
        # Write to file
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        print(f"✅ Configuration exported to: {output_path}")
        return 0
        
    except ConfigurationError as e:
        print(f"❌ Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


def create_env_template_command(args) -> int:
    """Create environment file template"""
    try:
        environment = Environment(args.environment)
        
        # Determine template file based on environment
        template_files = {
            Environment.DEVELOPMENT: ".env.example",
            Environment.TESTING: ".env.test.example", 
            Environment.PRODUCTION: ".env.prod.example",
        }
        
        template_file = template_files.get(environment, ".env.example")
        output_path = Path(args.output) if args.output else Path(template_file)
        
        # Check if .env.example exists to use as base
        example_path = Path(".env.example")
        if example_path.exists():
            # Copy and modify existing template
            with open(example_path, 'r') as f:
                content = f.read()
            
            # Add environment-specific header
            header = f"""# =============================================================================
# {environment.value.title()} Environment Configuration
# =============================================================================
# Environment-specific configuration for {environment.value}

ENVIRONMENT="{environment.value}"

"""
            content = header + content
            
            with open(output_path, 'w') as f:
                f.write(content)
        else:
            print("❌ .env.example file not found")
            return 1
        
        print(f"✅ Environment template created: {output_path}")
        return 0
        
    except ValueError as e:
        print(f"❌ Invalid environment: {args.environment}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Configuration management utility",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration')
    validate_parser.add_argument(
        '--environment', '-e',
        choices=['development', 'testing', 'production'],
        help='Target environment'
    )
    validate_parser.add_argument(
        '--env-file',
        help='Path to environment file'
    )
    validate_parser.set_defaults(func=validate_config_command)
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show configuration')
    show_parser.add_argument(
        '--environment', '-e',
        choices=['development', 'testing', 'production'],
        help='Target environment'
    )
    show_parser.add_argument(
        '--env-file',
        help='Path to environment file'
    )
    show_parser.add_argument(
        '--section', '-s',
        choices=['app', 'database', 'llm', 'rag', 'tts', 'monitoring', 'security'],
        help='Show specific configuration section'
    )
    show_parser.set_defaults(func=show_config_command)
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export configuration to JSON')
    export_parser.add_argument(
        '--environment', '-e',
        choices=['development', 'testing', 'production'],
        help='Target environment'
    )
    export_parser.add_argument(
        '--env-file',
        help='Path to environment file'
    )
    export_parser.add_argument(
        '--output', '-o',
        default='config.json',
        help='Output file path'
    )
    export_parser.add_argument(
        '--include-secrets',
        action='store_true',
        help='Include sensitive values in export'
    )
    export_parser.set_defaults(func=export_config_command)
    
    # Template command
    template_parser = subparsers.add_parser('template', help='Create environment file template')
    template_parser.add_argument(
        'environment',
        choices=['development', 'testing', 'production'],
        help='Target environment'
    )
    template_parser.add_argument(
        '--output', '-o',
        help='Output file path'
    )
    template_parser.set_defaults(func=create_env_template_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
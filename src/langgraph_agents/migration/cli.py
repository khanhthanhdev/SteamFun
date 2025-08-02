#!/usr/bin/env python3
"""
Command-line interface for configuration migration utilities.
Provides easy-to-use commands for migrating to LangGraph multi-agent system.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .config_migrator import ConfigurationMigrator
from .parameter_converter import ParameterConverter
from .validation_utils import ConfigurationValidator
from .migration_guide import MigrationGuide


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def migrate_from_env(args):
    """Migrate configuration from .env file."""
    print("🚀 Starting migration from .env file...")
    
    migrator = ConfigurationMigrator(
        source_config_dir=args.source_dir,
        target_config_dir=args.target_dir,
        backup_dir=args.backup_dir
    )
    
    success, messages = migrator.migrate_from_env_file(args.env_file)
    
    if success:
        print("✅ Migration completed successfully!")
        for message in messages:
            print(f"   {message}")
    else:
        print("❌ Migration failed:")
        for message in messages:
            print(f"   {message}")
        sys.exit(1)


def migrate_from_config(args):
    """Migrate configuration from legacy config file."""
    print("🚀 Starting migration from legacy configuration...")
    
    if not Path(args.config_file).exists():
        print(f"❌ Configuration file not found: {args.config_file}")
        sys.exit(1)
    
    # Load legacy configuration
    import json
    try:
        with open(args.config_file, 'r') as f:
            legacy_config = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load configuration file: {e}")
        sys.exit(1)
    
    migrator = ConfigurationMigrator(
        source_config_dir=args.source_dir,
        target_config_dir=args.target_dir,
        backup_dir=args.backup_dir
    )
    
    success, messages = migrator.migrate_from_legacy_config(legacy_config)
    
    if success:
        print("✅ Migration completed successfully!")
        for message in messages:
            print(f"   {message}")
    else:
        print("❌ Migration failed:")
        for message in messages:
            print(f"   {message}")
        sys.exit(1)


def validate_config(args):
    """Validate migrated configuration."""
    print("🔍 Validating configuration...")
    
    from ..config import ConfigurationManager
    
    config_manager = ConfigurationManager(args.config_dir)
    validator = ConfigurationValidator()
    
    try:
        system_config = config_manager.load_system_config()
        is_valid, errors = validator.validate_system_config(system_config)
        
        if is_valid:
            print("✅ Configuration is valid!")
        else:
            print("❌ Configuration validation failed:")
            
            # Group errors by severity
            error_count = len([e for e in errors if e.severity == 'error'])
            warning_count = len([e for e in errors if e.severity == 'warning'])
            
            print(f"   {error_count} error(s), {warning_count} warning(s)")
            
            for error in errors:
                icon = "❌" if error.severity == 'error' else "⚠️"
                print(f"   {icon} {error.component}.{error.parameter}: {error.message}")
                if error.suggestion:
                    print(f"      💡 {error.suggestion}")
        
        # Generate validation report if requested
        if args.report:
            report = validator.generate_validation_report(errors)
            report_path = Path(args.config_dir) / "validation_report.md"
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"📄 Validation report saved to: {report_path}")
        
        if not is_valid:
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Failed to validate configuration: {e}")
        sys.exit(1)


def convert_parameters(args):
    """Convert parameters for a specific component."""
    print(f"🔄 Converting parameters for {args.component_type}...")
    
    converter = ParameterConverter()
    
    # Load parameters from file or stdin
    if args.input_file:
        if not Path(args.input_file).exists():
            print(f"❌ Input file not found: {args.input_file}")
            sys.exit(1)
        
        import json
        try:
            with open(args.input_file, 'r') as f:
                source_params = json.load(f)
        except Exception as e:
            print(f"❌ Failed to load input file: {e}")
            sys.exit(1)
    else:
        print("Enter parameters as JSON (Ctrl+D to finish):")
        import json
        try:
            source_params = json.load(sys.stdin)
        except Exception as e:
            print(f"❌ Failed to parse JSON input: {e}")
            sys.exit(1)
    
    # Convert parameters
    result = converter.convert_parameters(args.component_type, source_params)
    
    if result.success:
        print("✅ Parameter conversion successful!")
        
        # Output converted parameters
        if args.output_file:
            import json
            with open(args.output_file, 'w') as f:
                json.dump(result.converted_params, f, indent=2)
            print(f"📄 Converted parameters saved to: {args.output_file}")
        else:
            import json
            print("Converted parameters:")
            print(json.dumps(result.converted_params, indent=2))
        
        # Show warnings
        if result.warnings:
            print("⚠️ Warnings:")
            for warning in result.warnings:
                print(f"   {warning}")
        
        # Show deprecated parameters
        if result.deprecated_params:
            print("🗑️ Deprecated parameters removed:")
            for param in result.deprecated_params:
                print(f"   {param}")
    else:
        print("❌ Parameter conversion failed:")
        for error in result.errors:
            print(f"   {error}")
        sys.exit(1)


def generate_docs(args):
    """Generate migration documentation."""
    print("📚 Generating migration documentation...")
    
    guide = MigrationGuide()
    
    if args.doc_type == 'complete' or args.doc_type == 'all':
        path = guide.generate_complete_migration_guide(args.output_dir + "/MIGRATION_GUIDE.md")
        print(f"✅ Complete migration guide: {path}")
    
    if args.doc_type == 'quickstart' or args.doc_type == 'all':
        path = guide.generate_quick_start_guide(args.output_dir + "/QUICK_START_MIGRATION.md")
        print(f"✅ Quick start guide: {path}")
    
    if args.doc_type == 'parameters' or args.doc_type == 'all':
        path = guide.generate_parameter_mapping_reference(args.output_dir + "/PARAMETER_MAPPING.md")
        print(f"✅ Parameter mapping reference: {path}")
    
    if args.doc_type == 'troubleshooting' or args.doc_type == 'all':
        path = guide.generate_troubleshooting_guide(args.output_dir + "/MIGRATION_TROUBLESHOOTING.md")
        print(f"✅ Troubleshooting guide: {path}")


def check_environment(args):
    """Check environment for migration readiness."""
    print("🔍 Checking environment for migration readiness...")
    
    import os
    import sys
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"❌ Python version too old: {python_version.major}.{python_version.minor}.{python_version.micro} (requires 3.8+)")
    
    # Check required packages
    required_packages = [
        "langgraph", "langchain", "langchain_community",
        "langchain_openai", "openai"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
    
    # Check environment variables
    env_vars = {
        "OPENAI_API_KEY": "Required for basic functionality",
        "JINA_API_KEY": "Optional - for enhanced RAG with JINA embeddings",
        "ASTRADB_APPLICATION_TOKEN": "Optional - for AstraDB vector store",
        "ASTRADB_API_ENDPOINT": "Optional - for AstraDB vector store",
        "LANGFUSE_SECRET_KEY": "Optional - for monitoring and tracing",
        "LANGFUSE_PUBLIC_KEY": "Optional - for monitoring and tracing"
    }
    
    print("\n🔑 Environment variables:")
    for var, description in env_vars.items():
        value = os.getenv(var)
        if value:
            # Show first 10 chars and last 4 chars for security
            masked = f"{value[:10]}...{value[-4:]}" if len(value) > 14 else "****"
            print(f"✅ {var}: {masked}")
        else:
            required = "Required" if "Required" in description else "Optional"
            print(f"⚠️ {var}: Not set ({required})")
            print(f"   {description}")
    
    # Check directories
    dirs = [
        ("config", "Configuration directory"),
        ("data/rag/chroma_db", "ChromaDB vector store"),
        ("data/rag/manim_docs", "Manim documentation"),
        ("data/context_learning", "Context learning data"),
        ("output", "Output directory")
    ]
    
    print("\n📁 Directories:")
    for dir_path, description in dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ {dir_path}: Exists")
        else:
            print(f"⚠️ {dir_path}: Missing ({description})")
    
    print("\n🎯 Migration readiness summary:")
    if not missing_packages and os.getenv("OPENAI_API_KEY"):
        print("✅ Ready for migration!")
    else:
        print("⚠️ Some requirements missing - see details above")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LangGraph Multi-Agent System Configuration Migration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate from .env file
  python -m src.langgraph_agents.migration.cli migrate-env

  # Migrate from legacy config
  python -m src.langgraph_agents.migration.cli migrate-config legacy_config.json

  # Validate migrated configuration
  python -m src.langgraph_agents.migration.cli validate --report

  # Convert parameters for code generator
  python -m src.langgraph_agents.migration.cli convert code_generator --input params.json

  # Generate all documentation
  python -m src.langgraph_agents.migration.cli docs all

  # Check environment readiness
  python -m src.langgraph_agents.migration.cli check-env
        """
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Migrate from .env command
    migrate_env_parser = subparsers.add_parser(
        'migrate-env',
        help='Migrate configuration from .env file'
    )
    migrate_env_parser.add_argument(
        '--env-file',
        default='.env',
        help='Path to .env file (default: .env)'
    )
    migrate_env_parser.add_argument(
        '--source-dir',
        default='config',
        help='Source configuration directory (default: config)'
    )
    migrate_env_parser.add_argument(
        '--target-dir',
        default='config',
        help='Target configuration directory (default: config)'
    )
    migrate_env_parser.add_argument(
        '--backup-dir',
        default='config/backup',
        help='Backup directory (default: config/backup)'
    )
    migrate_env_parser.set_defaults(func=migrate_from_env)
    
    # Migrate from config command
    migrate_config_parser = subparsers.add_parser(
        'migrate-config',
        help='Migrate configuration from legacy config file'
    )
    migrate_config_parser.add_argument(
        'config_file',
        help='Path to legacy configuration file'
    )
    migrate_config_parser.add_argument(
        '--source-dir',
        default='config',
        help='Source configuration directory (default: config)'
    )
    migrate_config_parser.add_argument(
        '--target-dir',
        default='config',
        help='Target configuration directory (default: config)'
    )
    migrate_config_parser.add_argument(
        '--backup-dir',
        default='config/backup',
        help='Backup directory (default: config/backup)'
    )
    migrate_config_parser.set_defaults(func=migrate_from_config)
    
    # Validate command
    validate_parser = subparsers.add_parser(
        'validate',
        help='Validate migrated configuration'
    )
    validate_parser.add_argument(
        '--config-dir',
        default='config',
        help='Configuration directory (default: config)'
    )
    validate_parser.add_argument(
        '--report',
        action='store_true',
        help='Generate validation report'
    )
    validate_parser.set_defaults(func=validate_config)
    
    # Convert parameters command
    convert_parser = subparsers.add_parser(
        'convert',
        help='Convert parameters for specific component'
    )
    convert_parser.add_argument(
        'component_type',
        choices=['code_generator', 'video_planner', 'video_renderer', 'environment'],
        help='Component type to convert parameters for'
    )
    convert_parser.add_argument(
        '--input-file',
        help='Input JSON file with parameters (default: read from stdin)'
    )
    convert_parser.add_argument(
        '--output-file',
        help='Output JSON file for converted parameters (default: print to stdout)'
    )
    convert_parser.set_defaults(func=convert_parameters)
    
    # Generate docs command
    docs_parser = subparsers.add_parser(
        'docs',
        help='Generate migration documentation'
    )
    docs_parser.add_argument(
        'doc_type',
        choices=['complete', 'quickstart', 'parameters', 'troubleshooting', 'all'],
        help='Type of documentation to generate'
    )
    docs_parser.add_argument(
        '--output-dir',
        default='.',
        help='Output directory for documentation (default: current directory)'
    )
    docs_parser.set_defaults(func=generate_docs)
    
    # Check environment command
    check_parser = subparsers.add_parser(
        'check-env',
        help='Check environment for migration readiness'
    )
    check_parser.set_defaults(func=check_environment)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Run command
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Migration CLI script for LangGraph agents refactor.

This script provides a command-line interface for running database and
configuration migrations during the transition to the new system architecture.
"""

import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langgraph_agents.migration.migration_manager import MigrationManager, MigrationStatus
from langgraph_agents.migration.database_migration import DatabaseMigrator, create_state_schema_migrations
from langgraph_agents.migration.config_migration import ConfigMigrator
from langgraph_agents.migration.data_validator import DataValidator


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'migration_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


async def run_database_migration(args):
    """Run database migration."""
    print("Running database migration...")
    
    connection_string = args.db_connection or os.getenv('POSTGRES_CONNECTION_STRING')
    if not connection_string:
        print("Error: Database connection string is required")
        print("Provide via --db-connection or POSTGRES_CONNECTION_STRING environment variable")
        return False
    
    try:
        migrator = DatabaseMigrator(connection_string)
        await migrator.initialize()
        
        # Register default migrations
        migrations = create_state_schema_migrations()
        for migration in migrations:
            migrator.register_migration(migration)
        
        print(f"Registered {len(migrations)} migrations")
        
        # Get pending migrations
        pending = await migrator.get_pending_migrations()
        if not pending:
            print("No pending migrations to execute")
            return True
        
        print(f"Found {len(pending)} pending migrations:")
        for migration in pending:
            print(f"  - {migration.version}: {migration.name}")
        
        if not args.auto_confirm:
            response = input("Proceed with migration? (y/N): ")
            if response.lower() != 'y':
                print("Migration cancelled")
                return False
        
        # Execute migrations
        successful, failed = await migrator.execute_all_pending()
        
        if failed == 0:
            print(f"✓ All {successful} migrations executed successfully")
            return True
        else:
            print(f"✗ Migration failed: {successful} successful, {failed} failed")
            return False
    
    except Exception as e:
        print(f"Database migration failed: {e}")
        return False
    
    finally:
        if 'migrator' in locals():
            await migrator.close()


def run_config_migration(args):
    """Run configuration migration."""
    print("Running configuration migration...")
    
    if not args.config_dirs:
        print("Error: No configuration directories specified")
        return False
    
    try:
        migrator = ConfigMigrator(args.backup_dir)
        
        all_results = []
        for config_dir in args.config_dirs:
            if not Path(config_dir).exists():
                print(f"Warning: Configuration directory does not exist: {config_dir}")
                continue
            
            print(f"Migrating configurations in: {config_dir}")
            
            # Create migration plan
            plan = migrator.create_migration_plan(config_dir)
            print(f"  Found {plan['total_files']} configuration files")
            print(f"  {plan['files_needing_migration']} files need migration")
            
            if plan['files_needing_migration'] == 0:
                print("  No files need migration")
                continue
            
            if not args.auto_confirm:
                response = input(f"Migrate {plan['files_needing_migration']} files? (y/N): ")
                if response.lower() != 'y':
                    print("  Migration cancelled")
                    continue
            
            # Execute migration
            results = migrator.migrate_config_directory(config_dir)
            all_results.extend(results)
            
            successful = sum(1 for r in results if r.success)
            failed = len(results) - successful
            
            print(f"  ✓ {successful} files migrated successfully")
            if failed > 0:
                print(f"  ✗ {failed} files failed to migrate")
        
        # Summary
        total_successful = sum(1 for r in all_results if r.success)
        total_failed = len(all_results) - total_successful
        
        print(f"\nConfiguration migration summary:")
        print(f"  Total files processed: {len(all_results)}")
        print(f"  Successful: {total_successful}")
        print(f"  Failed: {total_failed}")
        
        if args.export_report and all_results:
            summary = migrator.get_migration_summary(all_results)
            report_file = f"config_migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            import json
            with open(report_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print(f"  Report exported to: {report_file}")
        
        return total_failed == 0
    
    except Exception as e:
        print(f"Configuration migration failed: {e}")
        return False


def run_data_validation(args):
    """Run data validation."""
    print("Running data validation...")
    
    try:
        validator = DataValidator()
        
        # For demonstration, we'll validate some sample data
        # In practice, this would validate actual migrated data
        
        sample_data = [
            {
                "topic": "Python Basics",
                "description": "Introduction to Python programming",
                "session_id": "session-001",
                "workflow_complete": False
            },
            {
                "topic": "Data Structures", 
                "description": "Python data structures overview",
                "session_id": "session-002",
                "workflow_complete": True
            }
        ]
        
        print(f"Validating {len(sample_data)} data records...")
        
        report = validator.validate_data_integrity(sample_data)
        
        print(f"Validation results:")
        print(f"  Total records: {report.total_records}")
        print(f"  Valid records: {report.valid_records}")
        print(f"  Success rate: {report.success_rate:.1f}%")
        print(f"  Issues found: {len(report.issues)}")
        
        if report.has_critical_issues:
            print("  ✗ Critical issues found!")
        elif report.has_errors:
            print("  ⚠ Errors found")
        else:
            print("  ✓ Validation passed")
        
        # Show issues
        if report.issues and args.verbose:
            print("\nIssues found:")
            for issue in report.issues[:10]:  # Show first 10 issues
                print(f"  - {issue.severity.value}: {issue.message}")
                if issue.suggestion:
                    print(f"    Suggestion: {issue.suggestion}")
        
        # Export report
        if args.export_report:
            report_file = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            validator.export_validation_report(report, report_file)
            print(f"  Validation report exported to: {report_file}")
        
        return not report.has_critical_issues
    
    except Exception as e:
        print(f"Data validation failed: {e}")
        return False


async def run_complete_migration(args):
    """Run complete migration workflow."""
    print("Running complete migration workflow...")
    
    try:
        manager = MigrationManager(
            database_connection_string=args.db_connection or os.getenv('POSTGRES_CONNECTION_STRING'),
            config_backup_directory=args.backup_dir
        )
        
        await manager.initialize()
        
        try:
            # Create migration plan
            print("Creating migration plan...")
            plan = manager.create_migration_plan(
                config_directories=args.config_dirs or [],
                include_database=args.db_connection is not None,
                include_validation=True
            )
            
            print(f"Migration plan created:")
            print(f"  Plan ID: {plan.plan_id}")
            print(f"  Database migrations: {len(plan.database_migrations)}")
            print(f"  Configuration files: {len(plan.config_files)}")
            print(f"  Estimated duration: {plan.estimated_duration_minutes} minutes")
            
            if not args.auto_confirm:
                response = input("Execute migration plan? (y/N): ")
                if response.lower() != 'y':
                    print("Migration cancelled")
                    return False
            
            # Execute migration
            print("Executing migration plan...")
            result = await manager.execute_migration_plan(plan)
            
            print(f"Migration completed:")
            print(f"  Status: {result.status.value}")
            print(f"  Duration: {(result.completed_at - result.started_at).total_seconds():.1f} seconds")
            print(f"  Errors: {len(result.errors)}")
            print(f"  Warnings: {len(result.warnings)}")
            
            # Show errors and warnings
            if result.errors:
                print("Errors:")
                for error in result.errors:
                    print(f"  - {error}")
            
            if result.warnings:
                print("Warnings:")
                for warning in result.warnings:
                    print(f"  - {warning}")
            
            # Export report
            if args.export_report:
                report_file = f"migration_report_{plan.plan_id}.json"
                manager.export_migration_report(result, report_file)
                print(f"Migration report exported to: {report_file}")
            
            return result.status == MigrationStatus.COMPLETED
        
        finally:
            await manager.close()
    
    except Exception as e:
        print(f"Complete migration failed: {e}")
        return False


def run_migration_status(args):
    """Show migration status."""
    print("Migration status:")
    
    # This would typically connect to the database to show actual status
    # For now, we'll show a placeholder
    
    print("  Database migrations: Not available (requires database connection)")
    print("  Configuration migrations: Check backup directory for migration history")
    print("  Last migration: Unknown")
    
    return True


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LangGraph Agents Migration CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run database migration
  python scripts/run_migration.py database --db-connection "postgresql://user:pass@localhost/db"
  
  # Run configuration migration
  python scripts/run_migration.py config --config-dirs config/ --backup-dir backups/
  
  # Run data validation
  python scripts/run_migration.py validate --export-report
  
  # Run complete migration workflow
  python scripts/run_migration.py complete --config-dirs config/ --auto-confirm
  
  # Show migration status
  python scripts/run_migration.py status
        """
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--auto-confirm',
        action='store_true',
        help='Automatically confirm migration prompts'
    )
    
    parser.add_argument(
        '--export-report',
        action='store_true',
        help='Export migration report to file'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Migration commands')
    
    # Database migration command
    db_parser = subparsers.add_parser('database', help='Run database migration')
    db_parser.add_argument(
        '--db-connection',
        help='PostgreSQL connection string (or use POSTGRES_CONNECTION_STRING env var)'
    )
    
    # Configuration migration command
    config_parser = subparsers.add_parser('config', help='Run configuration migration')
    config_parser.add_argument(
        '--config-dirs',
        nargs='+',
        help='Configuration directories to migrate'
    )
    config_parser.add_argument(
        '--backup-dir',
        default='migration_backups',
        help='Directory for configuration backups'
    )
    
    # Data validation command
    validate_parser = subparsers.add_parser('validate', help='Run data validation')
    
    # Complete migration command
    complete_parser = subparsers.add_parser('complete', help='Run complete migration workflow')
    complete_parser.add_argument(
        '--db-connection',
        help='PostgreSQL connection string (or use POSTGRES_CONNECTION_STRING env var)'
    )
    complete_parser.add_argument(
        '--config-dirs',
        nargs='*',
        help='Configuration directories to migrate'
    )
    complete_parser.add_argument(
        '--backup-dir',
        default='migration_backups',
        help='Directory for configuration backups'
    )
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show migration status')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Run command
    try:
        if args.command == 'database':
            success = asyncio.run(run_database_migration(args))
        elif args.command == 'config':
            success = run_config_migration(args)
        elif args.command == 'validate':
            success = run_data_validation(args)
        elif args.command == 'complete':
            success = asyncio.run(run_complete_migration(args))
        elif args.command == 'status':
            success = run_migration_status(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1
        
        return 0 if success else 1
    
    except KeyboardInterrupt:
        print("\nMigration interrupted by user")
        return 1
    except Exception as e:
        print(f"Migration failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
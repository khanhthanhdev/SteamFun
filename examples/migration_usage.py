"""
Example usage of migration utilities for LangGraph agents refactor.

This script demonstrates how to use the migration utilities to migrate
database schemas, configuration files, and validate migrated data.
"""

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langgraph_agents.migration.migration_manager import MigrationManager
from langgraph_agents.migration.database_migration import DatabaseMigrator, create_state_schema_migrations
from langgraph_agents.migration.config_migration import ConfigMigrator
from langgraph_agents.migration.data_validator import DataValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demonstrate_database_migration():
    """Demonstrate database migration functionality."""
    print("\n=== Database Migration Example ===")
    
    # Note: This requires a PostgreSQL connection string
    # For demo purposes, we'll show the setup without actual database connection
    connection_string = "postgresql://user:password@localhost:5432/langgraph_db"
    
    try:
        # Initialize database migrator
        migrator = DatabaseMigrator(connection_string)
        
        # Register default state schema migrations
        default_migrations = create_state_schema_migrations()
        for migration in default_migrations:
            migrator.register_migration(migration)
            print(f"Registered migration: {migration.version} - {migration.name}")
        
        # Note: Actual execution would require database connection
        print(f"Total migrations registered: {len(default_migrations)}")
        print("Database migration setup completed (connection required for execution)")
        
    except Exception as e:
        print(f"Database migration demo failed (expected without DB connection): {e}")


def demonstrate_config_migration():
    """Demonstrate configuration file migration."""
    print("\n=== Configuration Migration Example ===")
    
    # Create sample old configuration
    old_config = {
        "planner_model": "openrouter/anthropic/claude-3.5-sonnet",
        "code_model": "openrouter/anthropic/claude-3.5-sonnet",
        "use_rag": True,
        "max_retries": 3,
        "output_dir": "output",
        "enable_caching": True,
        "max_scene_concurrency": 5,
        "use_visual_fix_code": False,
        "use_langfuse": True,
        "chroma_db_path": "data/rag/chroma_db",
        "embedding_model": "hf:ibm-granite/granite-embedding-30m-english",
        "context_learning_path": "data/context_learning",
        "manim_docs_path": "data/rag/manim_docs"
    }
    
    # Create temporary config file
    config_dir = Path("temp_migration_demo")
    config_dir.mkdir(exist_ok=True)
    
    try:
        # Write old config file
        old_config_file = config_dir / "old_config.json"
        with open(old_config_file, 'w') as f:
            json.dump(old_config, f, indent=2)
        
        print(f"Created sample old config: {old_config_file}")
        
        # Initialize config migrator
        migrator = ConfigMigrator(str(config_dir / "backups"))
        
        # Validate old config format
        validation_result = migrator.validate_config_format(str(old_config_file))
        print(f"Config validation: {validation_result}")
        
        # Migrate configuration file
        migration_result = migrator.migrate_config_file(str(old_config_file))
        
        if migration_result.success:
            print(f"Migration successful!")
            print(f"  Source: {migration_result.source_path}")
            print(f"  Target: {migration_result.target_path}")
            print(f"  Backup: {migration_result.backup_path}")
            print(f"  Warnings: {len(migration_result.warnings)}")
            
            # Show migration report
            if migration_result.migration_report:
                print(f"  Migration report: {migration_result.migration_report}")
        else:
            print(f"Migration failed: {migration_result.errors}")
        
        # Create migration plan for directory
        plan = migrator.create_migration_plan(str(config_dir))
        print(f"\nMigration plan:")
        print(f"  Total files: {plan['total_files']}")
        print(f"  Files needing migration: {plan['files_needing_migration']}")
        print(f"  Estimated duration: {plan['estimated_duration_minutes']} minutes")
        
        # Demonstrate rollback
        if migration_result.success:
            print(f"\nTesting rollback...")
            rollback_success = migrator.rollback_migration(migration_result)
            print(f"Rollback successful: {rollback_success}")
        
    except Exception as e:
        print(f"Config migration demo failed: {e}")
    
    finally:
        # Cleanup
        import shutil
        if config_dir.exists():
            shutil.rmtree(config_dir)


def demonstrate_data_validation():
    """Demonstrate data validation functionality."""
    print("\n=== Data Validation Example ===")
    
    # Sample old state data
    old_states = [
        {
            "topic": "Python Basics",
            "description": "Introduction to Python programming",
            "session_id": "session-001",
            "scene_outline": "Scene 1: Variables\nScene 2: Functions",
            "scene_implementations": {1: "Show variables", 2: "Show functions"},
            "generated_code": {1: "# Variable code", 2: "# Function code"},
            "rendered_videos": {1: "video1.mp4", 2: "video2.mp4"},
            "workflow_complete": False
        },
        {
            "topic": "Data Structures",
            "description": "Python data structures overview",
            "session_id": "session-002",
            "scene_outline": "Scene 1: Lists\nScene 2: Dictionaries",
            "scene_implementations": {1: "Show lists", 2: "Show dictionaries"},
            "generated_code": {1: "# List code", 2: "# Dict code"},
            "rendered_videos": {1: "video3.mp4", 2: "video4.mp4"},
            "workflow_complete": True
        }
    ]
    
    # Sample new state data (would be actual VideoGenerationState objects)
    from langgraph_agents.models.state import VideoGenerationState
    
    new_states = []
    for old_state in old_states:
        try:
            new_state = VideoGenerationState(
                topic=old_state["topic"],
                description=old_state["description"],
                session_id=old_state["session_id"],
                scene_outline=old_state["scene_outline"],
                scene_implementations=old_state["scene_implementations"],
                generated_code=old_state["generated_code"],
                rendered_videos=old_state["rendered_videos"],
                workflow_complete=old_state["workflow_complete"]
            )
            new_states.append(new_state)
        except Exception as e:
            print(f"Failed to create new state: {e}")
            return
    
    # Initialize data validator
    validator = DataValidator()
    
    # Validate state migration
    print("Validating state migration...")
    state_report = validator.validate_state_migration(old_states, new_states)
    
    print(f"State validation results:")
    print(f"  Total records: {state_report.total_records}")
    print(f"  Valid records: {state_report.valid_records}")
    print(f"  Success rate: {state_report.success_rate:.1f}%")
    print(f"  Issues found: {len(state_report.issues)}")
    print(f"  Has critical issues: {state_report.has_critical_issues}")
    
    if state_report.issues:
        print("  Issues:")
        for issue in state_report.issues[:3]:  # Show first 3 issues
            print(f"    - {issue.severity.value}: {issue.message}")
    
    # Validate data integrity
    print("\nValidating data integrity...")
    integrity_report = validator.validate_data_integrity(old_states)
    
    print(f"Data integrity results:")
    print(f"  Total records: {integrity_report.total_records}")
    print(f"  Valid records: {integrity_report.valid_records}")
    print(f"  Success rate: {integrity_report.success_rate:.1f}%")
    print(f"  Issues found: {len(integrity_report.issues)}")
    
    # Export validation report
    report_file = Path("validation_report_demo.json")
    try:
        validator.export_validation_report(state_report, str(report_file))
        print(f"Validation report exported to: {report_file}")
        
        # Show report summary
        with open(report_file, 'r') as f:
            report_data = json.load(f)
        
        print(f"Report summary: {report_data['summary']}")
        
    except Exception as e:
        print(f"Failed to export validation report: {e}")
    
    finally:
        # Cleanup
        if report_file.exists():
            report_file.unlink()


async def demonstrate_complete_migration():
    """Demonstrate complete migration workflow using MigrationManager."""
    print("\n=== Complete Migration Workflow Example ===")
    
    # Create sample workspace
    workspace = Path("migration_demo_workspace")
    workspace.mkdir(exist_ok=True)
    config_dir = workspace / "config"
    config_dir.mkdir(exist_ok=True)
    
    try:
        # Create sample configuration files
        sample_configs = [
            {
                "planner_model": "openrouter/anthropic/claude-3.5-sonnet",
                "use_rag": True,
                "max_retries": 3,
                "output_dir": "output",
                "enable_caching": True
            },
            {
                "code_model": "openrouter/anthropic/claude-3.5-sonnet",
                "use_rag": False,
                "max_retries": 5,
                "output_dir": "custom_output",
                "enable_caching": False
            }
        ]
        
        for i, config in enumerate(sample_configs):
            config_file = config_dir / f"config{i}.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
        
        print(f"Created {len(sample_configs)} sample configuration files")
        
        # Initialize migration manager
        manager = MigrationManager(
            database_connection_string=None,  # Skip database for demo
            config_backup_directory=str(workspace / "backups")
        )
        
        await manager.initialize()
        
        try:
            # Create migration plan
            print("Creating migration plan...")
            plan = manager.create_migration_plan(
                config_directories=[str(config_dir)],
                include_database=False,
                include_validation=True
            )
            
            print(f"Migration plan created:")
            print(f"  Plan ID: {plan.plan_id}")
            print(f"  Config files: {len(plan.config_files)}")
            print(f"  Validation required: {plan.validation_required}")
            print(f"  Estimated duration: {plan.estimated_duration_minutes} minutes")
            print(f"  Backup required: {plan.backup_required}")
            
            # Execute migration plan
            print("\nExecuting migration plan...")
            result = await manager.execute_migration_plan(plan)
            
            print(f"Migration execution completed:")
            print(f"  Status: {result.status.value}")
            print(f"  Started: {result.started_at}")
            print(f"  Completed: {result.completed_at}")
            print(f"  Errors: {len(result.errors)}")
            print(f"  Warnings: {len(result.warnings)}")
            print(f"  Rollback available: {result.rollback_available}")
            
            # Show config migration results
            if result.config_migration_results:
                print(f"\nConfiguration migration results:")
                for i, config_result in enumerate(result.config_migration_results):
                    print(f"  Config {i+1}:")
                    print(f"    Success: {config_result.success}")
                    print(f"    Source: {config_result.source_path}")
                    print(f"    Target: {config_result.target_path}")
                    print(f"    Backup: {config_result.backup_path}")
                    if config_result.errors:
                        print(f"    Errors: {config_result.errors}")
            
            # Show validation results
            if result.validation_reports:
                print(f"\nValidation results:")
                for i, report in enumerate(result.validation_reports):
                    print(f"  Report {i+1}:")
                    print(f"    Data type: {report.data_type}")
                    print(f"    Success rate: {report.success_rate:.1f}%")
                    print(f"    Issues: {len(report.issues)}")
            
            # Export migration report
            report_file = workspace / "migration_report.json"
            manager.export_migration_report(result, str(report_file))
            print(f"\nMigration report exported to: {report_file}")
            
            # Get migration status
            status = await manager.get_migration_status()
            print(f"\nMigration status:")
            print(f"  Migration history: {len(status['migration_history'])} entries")
            if status['last_migration']:
                print(f"  Last migration: {status['last_migration']}")
            
        finally:
            await manager.close()
    
    except Exception as e:
        print(f"Complete migration demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        import shutil
        if workspace.exists():
            shutil.rmtree(workspace)


def demonstrate_migration_utilities():
    """Demonstrate various migration utility features."""
    print("\n=== Migration Utilities Features ===")
    
    # Show available migration scripts
    migrations = create_state_schema_migrations()
    print(f"Available database migrations: {len(migrations)}")
    for migration in migrations:
        print(f"  {migration.version}: {migration.name}")
        print(f"    Description: {migration.description}")
        print(f"    Dependencies: {migration.dependencies}")
    
    # Show validation rules
    validator = DataValidator()
    print(f"\nValidation rules:")
    print(f"  Required state fields: {validator.validation_rules['required_state_fields']}")
    print(f"  Required config fields: {validator.validation_rules['required_config_fields']}")
    print(f"  Type mappings: {len(validator.validation_rules['type_mappings'])} types")
    print(f"  Consistency checks: {validator.validation_rules['consistency_checks']}")


async def main():
    """Run all migration examples."""
    print("LangGraph Agents Migration Utilities Demo")
    print("=" * 50)
    
    # Run demonstrations
    await demonstrate_database_migration()
    demonstrate_config_migration()
    demonstrate_data_validation()
    await demonstrate_complete_migration()
    demonstrate_migration_utilities()
    
    print("\n" + "=" * 50)
    print("Migration utilities demonstration completed!")
    print("\nTo use these utilities in your project:")
    print("1. Set up PostgreSQL connection string for database migrations")
    print("2. Prepare your configuration files for migration")
    print("3. Create a migration plan using MigrationManager")
    print("4. Execute the migration plan")
    print("5. Validate the migrated data")
    print("6. Export reports for documentation")


if __name__ == "__main__":
    asyncio.run(main())
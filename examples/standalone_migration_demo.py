"""
Standalone demonstration of migration utilities.

This script demonstrates the migration functionality by importing
only the specific modules needed, avoiding circular dependencies.
"""

import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import only the specific modules we need
from langgraph_agents.migration.database_migration import MigrationScript, create_state_schema_migrations
from langgraph_agents.migration.data_validator import DataValidator, ValidationSeverity, ValidationIssue, ValidationReport


def demonstrate_database_migrations():
    """Demonstrate database migration creation."""
    print("=== Database Migration Demonstration ===")
    
    try:
        # Create default migrations
        migrations = create_state_schema_migrations()
        
        print(f"✓ Created {len(migrations)} database migrations:")
        for migration in migrations:
            print(f"  - Version {migration.version}: {migration.name}")
            print(f"    Description: {migration.description}")
            print(f"    Dependencies: {migration.dependencies}")
            print(f"    Has UP SQL: {len(migration.up_sql)} characters")
            print(f"    Has DOWN SQL: {len(migration.down_sql)} characters")
            print()
        
        # Show sample SQL from first migration
        if migrations:
            first_migration = migrations[0]
            print(f"Sample SQL from migration {first_migration.version}:")
            print("UP SQL (first 200 chars):")
            print(f"  {first_migration.up_sql[:200]}...")
            print()
        
        return True
        
    except Exception as e:
        print(f"✗ Database migration demonstration failed: {e}")
        return False


def demonstrate_data_validation():
    """Demonstrate data validation functionality."""
    print("=== Data Validation Demonstration ===")
    
    try:
        # Create sample data for validation
        sample_data = [
            {
                "topic": "Python Basics",
                "description": "Introduction to Python programming",
                "session_id": "session-001",
                "workflow_complete": False,
                "scene_implementations": {1: "Show variables", 2: "Show functions"},
                "generated_code": {1: "# Variable code", 2: "# Function code"}
            },
            {
                "topic": "Data Structures",
                "description": "Python data structures overview", 
                "session_id": "session-002",
                "workflow_complete": True,
                "scene_implementations": {1: "Show lists"},
                "generated_code": {1: "# List code"}
            },
            {
                # Invalid data to trigger validation issues
                "topic": "",  # Empty topic should cause error
                "description": "Valid description",
                "session_id": "session-003",
                "workflow_complete": "invalid",  # Should be boolean
                "scene_implementations": "invalid",  # Should be dict
                "generated_code": {1: "# Code", 2: "# More code"}  # Inconsistent with scene_implementations
            }
        ]
        
        # Initialize validator
        validator = DataValidator()
        print("✓ Data validator initialized")
        
        # Validate data integrity
        report = validator.validate_data_integrity(sample_data)
        
        print(f"\n✓ Validation completed:")
        print(f"  Total records: {report.total_records}")
        print(f"  Valid records: {report.valid_records}")
        print(f"  Invalid records: {report.invalid_records}")
        print(f"  Success rate: {report.success_rate:.1f}%")
        print(f"  Total issues: {len(report.issues)}")
        print(f"  Has critical issues: {report.has_critical_issues}")
        print(f"  Has errors: {report.has_errors}")
        
        # Show validation issues by severity
        severity_counts = {}
        for issue in report.issues:
            severity = issue.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        if severity_counts:
            print(f"\n  Issues by severity:")
            for severity, count in severity_counts.items():
                print(f"    {severity.upper()}: {count}")
        
        # Show some issues
        if report.issues:
            print(f"\n  Sample validation issues:")
            for i, issue in enumerate(report.issues[:5]):  # Show first 5
                print(f"    {i+1}. [{issue.severity.value.upper()}] {issue.message}")
                print(f"       Field: {issue.field_path}")
                if issue.suggestion:
                    print(f"       Suggestion: {issue.suggestion}")
                print()
        
        # Export validation report
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            validator.export_validation_report(report, f.name)
            print(f"✓ Validation report exported to: {f.name}")
            
            # Verify export worked
            with open(f.name, 'r') as rf:
                report_data = json.load(rf)
            
            print(f"  Report contains {len(report_data['issues'])} issues")
            print(f"  Report ID: {report_data['validation_id']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data validation demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demonstrate_validation_rules():
    """Demonstrate validation rules and configuration."""
    print("=== Validation Rules Demonstration ===")
    
    try:
        validator = DataValidator()
        
        print("✓ Validation rules configuration:")
        rules = validator.validation_rules
        
        print(f"  Required state fields ({len(rules['required_state_fields'])}):")
        for field in rules['required_state_fields']:
            print(f"    - {field}")
        
        print(f"\n  Required config fields ({len(rules['required_config_fields'])}):")
        for field in rules['required_config_fields']:
            print(f"    - {field}")
        
        print(f"\n  Type mappings ({len(rules['type_mappings'])}):")
        for field, expected_type in rules['type_mappings'].items():
            type_name = expected_type.__name__ if hasattr(expected_type, '__name__') else str(expected_type)
            print(f"    {field}: {type_name}")
        
        print(f"\n  Consistency checks ({len(rules['consistency_checks'])}):")
        for check in rules['consistency_checks']:
            print(f"    - {check}")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation rules demonstration failed: {e}")
        return False


def demonstrate_migration_script_creation():
    """Demonstrate creating custom migration scripts."""
    print("=== Custom Migration Script Demonstration ===")
    
    try:
        # Create a custom migration script
        custom_migration = MigrationScript(
            version="999",
            name="demo_custom_migration",
            description="Demonstration of custom migration script creation",
            up_sql="""
            -- Demo migration: Add demo table
            CREATE TABLE demo_table (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            
            -- Add index
            CREATE INDEX idx_demo_table_name ON demo_table(name);
            """,
            down_sql="""
            -- Rollback demo migration
            DROP INDEX IF EXISTS idx_demo_table_name;
            DROP TABLE IF EXISTS demo_table;
            """,
            dependencies=["001", "002"],
            created_at=datetime.now()
        )
        
        print("✓ Created custom migration script:")
        print(f"  Version: {custom_migration.version}")
        print(f"  Name: {custom_migration.name}")
        print(f"  Description: {custom_migration.description}")
        print(f"  Dependencies: {custom_migration.dependencies}")
        print(f"  UP SQL length: {len(custom_migration.up_sql)} characters")
        print(f"  DOWN SQL length: {len(custom_migration.down_sql)} characters")
        print(f"  Created at: {custom_migration.created_at}")
        
        # Show the SQL content
        print(f"\n  UP SQL preview:")
        print("    " + "\n    ".join(custom_migration.up_sql.strip().split('\n')[:5]))
        print("    ...")
        
        return True
        
    except Exception as e:
        print(f"✗ Custom migration script demonstration failed: {e}")
        return False


def main():
    """Run migration utilities demonstration."""
    print("LangGraph Agents Migration Utilities - Standalone Demo")
    print("=" * 65)
    print(f"Demo started at: {datetime.now()}")
    print()
    
    try:
        # Run demonstrations
        results = []
        
        results.append(demonstrate_database_migrations())
        print()
        
        results.append(demonstrate_data_validation())
        print()
        
        results.append(demonstrate_validation_rules())
        print()
        
        results.append(demonstrate_migration_script_creation())
        print()
        
        # Summary
        print("=" * 65)
        successful = sum(results)
        total = len(results)
        
        if successful == total:
            print(f"✓ All {total} demonstrations completed successfully!")
        else:
            print(f"⚠ {successful}/{total} demonstrations completed successfully")
        
        print("\nMigration utilities features demonstrated:")
        print("  ✓ Database schema migration scripts")
        print("  ✓ Data validation with comprehensive reporting")
        print("  ✓ Configurable validation rules")
        print("  ✓ Export capabilities for reports")
        print("  ✓ Custom migration script creation")
        print("  ✓ Error detection and classification")
        
        print("\nNext steps:")
        print("  1. Set up PostgreSQL connection for database migrations")
        print("  2. Prepare configuration files for migration")
        print("  3. Use MigrationManager for coordinated migration execution")
        print("  4. Run validation on migrated data")
        print("  5. Export reports for documentation")
        
        return 0 if successful == total else 1
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
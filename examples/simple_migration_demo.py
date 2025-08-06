"""
Simple demonstration of migration utilities.

This script demonstrates the core migration functionality without
dependencies on the full LangGraph agents system.
"""

import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langgraph_agents.migration.database_migration import create_state_schema_migrations
from langgraph_agents.migration.data_validator import DataValidator, ValidationSeverity


def demonstrate_database_migrations():
    """Demonstrate database migration creation."""
    print("=== Database Migration Demonstration ===")
    
    # Create default migrations
    migrations = create_state_schema_migrations()
    
    print(f"Created {len(migrations)} database migrations:")
    for migration in migrations:
        print(f"  - Version {migration.version}: {migration.name}")
        print(f"    Description: {migration.description}")
        print(f"    Dependencies: {migration.dependencies}")
        print(f"    Has rollback SQL: {bool(migration.down_sql)}")
        print()
    
    return True


def demonstrate_data_validation():
    """Demonstrate data validation functionality."""
    print("=== Data Validation Demonstration ===")
    
    # Create sample data for validation
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
        },
        {
            # Invalid data to trigger validation issues
            "topic": "",  # Empty topic should cause error
            "description": "Valid description",
            "session_id": "session-003",
            "workflow_complete": "invalid"  # Should be boolean
        }
    ]
    
    # Initialize validator
    validator = DataValidator()
    print("Data validator initialized")
    
    # Validate data integrity
    report = validator.validate_data_integrity(sample_data)
    
    print(f"\nValidation Results:")
    print(f"  Total records: {report.total_records}")
    print(f"  Valid records: {report.valid_records}")
    print(f"  Invalid records: {report.invalid_records}")
    print(f"  Success rate: {report.success_rate:.1f}%")
    print(f"  Total issues: {len(report.issues)}")
    print(f"  Has critical issues: {report.has_critical_issues}")
    print(f"  Has errors: {report.has_errors}")
    
    # Show some issues
    if report.issues:
        print(f"\nFirst few validation issues:")
        for i, issue in enumerate(report.issues[:3]):
            print(f"  {i+1}. {issue.severity.value.upper()}: {issue.message}")
            if issue.suggestion:
                print(f"     Suggestion: {issue.suggestion}")
    
    # Export validation report
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        validator.export_validation_report(report, f.name)
        print(f"\nValidation report exported to: {f.name}")
        
        # Show report summary
        with open(f.name, 'r') as rf:
            report_data = json.load(rf)
        
        print(f"Report summary:")
        print(f"  Validation ID: {report_data['validation_id']}")
        print(f"  Data type: {report_data['data_type']}")
        print(f"  Summary: {report_data['summary']}")
    
    return True


def demonstrate_validation_rules():
    """Demonstrate validation rules and configuration."""
    print("=== Validation Rules Demonstration ===")
    
    validator = DataValidator()
    
    print("Validation rules configuration:")
    print(f"  Required state fields: {validator.validation_rules['required_state_fields']}")
    print(f"  Required config fields: {validator.validation_rules['required_config_fields']}")
    print(f"  Type mappings: {len(validator.validation_rules['type_mappings'])} types defined")
    print(f"  Consistency checks: {validator.validation_rules['consistency_checks']}")
    
    # Show type mappings
    print(f"\nType mappings:")
    for field, expected_type in validator.validation_rules['type_mappings'].items():
        print(f"  {field}: {expected_type.__name__}")
    
    return True


def main():
    """Run migration utilities demonstration."""
    print("LangGraph Agents Migration Utilities - Simple Demo")
    print("=" * 60)
    print(f"Demo started at: {datetime.now()}")
    print()
    
    try:
        # Run demonstrations
        success = True
        
        success &= demonstrate_database_migrations()
        print()
        
        success &= demonstrate_data_validation()
        print()
        
        success &= demonstrate_validation_rules()
        print()
        
        print("=" * 60)
        if success:
            print("✓ All migration utilities demonstrated successfully!")
        else:
            print("✗ Some demonstrations failed")
        
        print("\nMigration utilities are ready for use in the LangGraph agents refactor.")
        print("Key features demonstrated:")
        print("  - Database schema migration scripts")
        print("  - Data validation with comprehensive reporting")
        print("  - Configurable validation rules")
        print("  - Export capabilities for reports")
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
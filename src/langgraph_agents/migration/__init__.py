"""
Migration utilities for LangGraph agents refactor.

This module provides utilities for migrating database schemas, configuration files,
and validating migrated data integrity during the transition from old to new
system architecture.
"""

# Import only the essential classes to avoid circular dependencies
# Individual modules can be imported directly when needed

__all__ = [
    'DatabaseMigrator',
    'MigrationScript', 
    'MigrationError',
    'ConfigMigrator',
    'ConfigMigrationError',
    'DataValidator',
    'ValidationError',
    'ValidationReport',
    'MigrationManager',
    'MigrationPlan',
    'MigrationStatus'
]

# Lazy imports to avoid circular dependencies
def get_database_migrator():
    """Get DatabaseMigrator class."""
    from .database_migration import DatabaseMigrator
    return DatabaseMigrator

def get_config_migrator():
    """Get ConfigMigrator class."""
    from .config_migration import ConfigMigrator
    return ConfigMigrator

def get_data_validator():
    """Get DataValidator class."""
    from .data_validator import DataValidator
    return DataValidator

def get_migration_manager():
    """Get MigrationManager class."""
    from .migration_manager import MigrationManager
    return MigrationManager
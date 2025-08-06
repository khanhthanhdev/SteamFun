# Migration Utilities

This module provides comprehensive migration utilities for the LangGraph agents refactor, including database schema migrations, configuration file migrations, and data validation.

## Overview

The migration utilities help transition from the old TypedDict-based state management to the new Pydantic-based system while ensuring data integrity and backward compatibility.

## Components

### 1. Database Migration (`database_migration.py`)

Handles database schema changes required for the new state format:

- **DatabaseMigrator**: Main class for managing database migrations
- **MigrationScript**: Represents individual migration scripts
- **create_state_schema_migrations()**: Creates default schema migrations

#### Features:
- Migration tracking table
- Dependency management
- Rollback capabilities
- Schema validation
- Performance monitoring

#### Example Usage:
```python
from langgraph_agents.migration.database_migration import DatabaseMigrator, create_state_schema_migrations

# Initialize migrator
migrator = DatabaseMigrator("postgresql://user:pass@localhost/db")
await migrator.initialize()

# Register migrations
migrations = create_state_schema_migrations()
for migration in migrations:
    migrator.register_migration(migration)

# Execute pending migrations
successful, failed = await migrator.execute_all_pending()
```

### 2. Configuration Migration (`config_migration.py`)

Migrates configuration files from old to new formats:

- **ConfigMigrator**: Main class for configuration migration
- **ConfigMigrationResult**: Result of migration operations
- Backup and rollback functionality
- Validation of migrated configurations

#### Features:
- JSON and YAML support
- Automatic backup creation
- Migration validation
- Batch processing
- Rollback capabilities

#### Example Usage:
```python
from langgraph_agents.migration.config_migration import ConfigMigrator

# Initialize migrator
migrator = ConfigMigrator("backup_directory")

# Migrate single file
result = migrator.migrate_config_file("old_config.json")

# Migrate directory
results = migrator.migrate_config_directory("config_directory")
```

### 3. Data Validation (`data_validator.py`)

Validates migrated data for integrity and consistency:

- **DataValidator**: Main validation class
- **ValidationReport**: Comprehensive validation results
- **ValidationIssue**: Individual validation issues
- Multiple validation types and severity levels

#### Features:
- State migration validation
- Configuration migration validation
- Data integrity checks
- Detailed reporting
- Export capabilities

#### Example Usage:
```python
from langgraph_agents.migration.data_validator import DataValidator

# Initialize validator
validator = DataValidator()

# Validate state migration
report = validator.validate_state_migration(old_states, new_states)

# Export report
validator.export_validation_report(report, "validation_report.json")
```

### 4. Migration Manager (`migration_manager.py`)

Coordinates the complete migration process:

- **MigrationManager**: Central coordination class
- **MigrationPlan**: Comprehensive migration planning
- **MigrationResult**: Complete migration results
- Orchestrates all migration components

#### Features:
- Migration planning
- Coordinated execution
- Progress tracking
- Rollback management
- Comprehensive reporting

#### Example Usage:
```python
from langgraph_agents.migration.migration_manager import MigrationManager

# Initialize manager
manager = MigrationManager(
    database_connection_string="postgresql://...",
    config_backup_directory="backups"
)
await manager.initialize()

# Create and execute migration plan
plan = manager.create_migration_plan(
    config_directories=["config"],
    include_database=True,
    include_validation=True
)

result = await manager.execute_migration_plan(plan)
```

## Migration Process

### 1. Planning Phase
- Analyze existing configurations
- Identify required database changes
- Create comprehensive migration plan
- Estimate duration and resources

### 2. Backup Phase
- Create backups of configuration files
- Document current database schema
- Prepare rollback procedures

### 3. Execution Phase
- Execute database schema migrations
- Migrate configuration files
- Update state formats
- Apply data transformations

### 4. Validation Phase
- Validate migrated data integrity
- Check configuration consistency
- Verify system functionality
- Generate validation reports

### 5. Cleanup Phase
- Remove temporary files
- Update documentation
- Archive migration logs
- Prepare for production

## CLI Usage

Use the migration CLI script for command-line operations:

```bash
# Run database migration
python scripts/run_migration.py database --db-connection "postgresql://..."

# Run configuration migration
python scripts/run_migration.py config --config-dirs config/ --backup-dir backups/

# Run data validation
python scripts/run_migration.py validate --export-report

# Run complete migration workflow
python scripts/run_migration.py complete --config-dirs config/ --auto-confirm

# Show migration status
python scripts/run_migration.py status
```

## Database Schema Changes

The migration includes these database schema changes:

### 1. Checkpoint Table Updates (Migration 001)
- Add `state_version` column for tracking state format
- Add `workflow_step` column for current step tracking
- Add error and retry tracking columns
- Add performance indexes

### 2. Performance Metrics Table (Migration 002)
- Create table for storing performance metrics
- Track step durations and success rates
- Monitor resource usage

### 3. Workflow State Audit Table (Migration 003)
- Create audit trail for state changes
- Track workflow progression
- Store error information and recovery actions

## Configuration Changes

Configuration migration handles these transformations:

### Model Configuration
- Convert string model names to structured ModelConfig objects
- Extract provider and model information
- Set default parameters (temperature, max_tokens, timeout)

### Feature Flags
- Map old feature flags to new configuration structure
- Handle deprecated settings
- Set appropriate defaults

### Performance Settings
- Update concurrency limits
- Map timeout configurations
- Convert quality settings

### Directory Paths
- Validate and sanitize path configurations
- Handle relative vs absolute paths
- Ensure security compliance

## Validation Rules

Data validation includes these checks:

### Required Fields
- Ensure all required fields are present
- Validate field types and formats
- Check for empty or null values

### Data Consistency
- Verify relationships between fields
- Check scene number consistency
- Validate workflow state transitions

### Security Validation
- Sanitize input data
- Check for potentially dangerous content
- Validate file paths and permissions

### Format Compliance
- Ensure Pydantic model compliance
- Validate JSON/YAML structure
- Check configuration schema adherence

## Error Handling

The migration utilities provide comprehensive error handling:

### Error Categories
- **Critical**: Migration cannot proceed
- **Error**: Significant issues requiring attention
- **Warning**: Issues that should be reviewed
- **Info**: Informational messages

### Recovery Strategies
- Automatic retry with exponential backoff
- Rollback to previous state
- Manual intervention prompts
- Detailed error reporting

### Logging
- Comprehensive logging at all levels
- Structured log messages
- Performance metrics
- Audit trail maintenance

## Best Practices

### Before Migration
1. **Backup Everything**: Create complete backups of database and configuration files
2. **Test in Development**: Run migration in development environment first
3. **Review Migration Plan**: Carefully review the generated migration plan
4. **Check Dependencies**: Ensure all required dependencies are available

### During Migration
1. **Monitor Progress**: Watch migration logs for issues
2. **Validate Incrementally**: Check results at each phase
3. **Be Prepared to Rollback**: Have rollback procedures ready
4. **Document Issues**: Record any problems encountered

### After Migration
1. **Validate Results**: Run comprehensive validation checks
2. **Test Functionality**: Verify system works correctly
3. **Update Documentation**: Update system documentation
4. **Archive Migration Data**: Keep migration logs and reports

## Troubleshooting

### Common Issues

#### Database Connection Errors
- Verify connection string format
- Check database server availability
- Ensure proper permissions

#### Configuration Migration Failures
- Check file permissions
- Verify configuration file format
- Review validation errors

#### Validation Failures
- Check data integrity issues
- Review field mapping problems
- Verify type conversions

### Recovery Procedures

#### Rollback Database Migration
```python
# Rollback specific migration
await migrator.rollback_migration("001")

# Check migration status
status = await migrator.get_migration_status()
```

#### Restore Configuration Files
```python
# Rollback configuration migration
success = config_migrator.rollback_migration(migration_result)

# Restore from backup
config_migrator.restore_config_from_backup(backup_path, target_path)
```

## Integration Tests

Run the comprehensive integration tests:

```bash
# Run all migration tests
python -m pytest tests/integration/test_migration_process.py -v

# Run specific test categories
python -m pytest tests/integration/test_migration_process.py::TestDatabaseMigration -v
python -m pytest tests/integration/test_migration_process.py::TestConfigMigration -v
python -m pytest tests/integration/test_migration_process.py::TestDataValidation -v
```

## Examples

See the `examples/migration_usage.py` file for comprehensive examples of using all migration utilities.

## Support

For issues or questions about the migration utilities:

1. Check the integration tests for usage examples
2. Review the validation reports for specific issues
3. Consult the migration logs for detailed error information
4. Use the CLI help for command-specific guidance

## Version History

- **v1.0**: Initial migration utilities implementation
- **v1.1**: Added comprehensive validation and reporting
- **v1.2**: Enhanced error handling and recovery procedures
- **v1.3**: Added CLI interface and batch processing capabilities
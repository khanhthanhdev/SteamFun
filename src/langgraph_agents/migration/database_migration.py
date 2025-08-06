"""
Database migration scripts for state schema changes.

This module provides utilities for migrating database schemas when transitioning
from old TypedDict state format to new Pydantic state format, including
checkpoint table schema updates and data migration.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

try:
    import asyncpg
    POSTGRES_AVAILABLE = True
except ImportError:
    asyncpg = None
    POSTGRES_AVAILABLE = False

logger = logging.getLogger(__name__)


class MigrationError(Exception):
    """Exception raised during database migration operations."""
    pass


class MigrationStatus(Enum):
    """Status of a migration script."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class MigrationScript:
    """Represents a database migration script."""
    version: str
    name: str
    description: str
    up_sql: str
    down_sql: str
    dependencies: List[str]
    created_at: datetime
    
    def __post_init__(self):
        """Validate migration script after initialization."""
        if not self.version:
            raise ValueError("Migration version is required")
        if not self.name:
            raise ValueError("Migration name is required")
        if not self.up_sql:
            raise ValueError("Up SQL is required")


class DatabaseMigrator:
    """
    Database migrator for handling schema changes during LangGraph agents refactor.
    
    This class manages database schema migrations, including:
    - Creating migration tracking tables
    - Executing migration scripts in order
    - Rolling back failed migrations
    - Validating schema changes
    """
    
    def __init__(self, connection_string: str):
        """
        Initialize database migrator.
        
        Args:
            connection_string: PostgreSQL connection string
        """
        if not POSTGRES_AVAILABLE:
            raise ImportError("Database migration requires asyncpg")
        
        self.connection_string = connection_string
        self._connection_pool = None
        self._migrations = {}
        self._executed_migrations = set()
        
        logger.info("Database migrator initialized")
    
    async def initialize(self) -> None:
        """Initialize the database migrator and create migration tracking table."""
        try:
            # Create connection pool
            self._connection_pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=2,
                max_size=10
            )
            
            # Create migration tracking table
            await self._create_migration_table()
            
            # Load executed migrations
            await self._load_executed_migrations()
            
            logger.info("Database migrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database migrator: {e}")
            raise MigrationError(f"Initialization failed: {e}") from e
    
    async def close(self) -> None:
        """Close the database migrator."""
        if self._connection_pool:
            await self._connection_pool.close()
            self._connection_pool = None
        
        logger.info("Database migrator closed")
    
    async def _create_migration_table(self) -> None:
        """Create the migration tracking table if it doesn't exist."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            executed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            execution_time_ms INTEGER,
            status VARCHAR(50) DEFAULT 'completed',
            rollback_sql TEXT,
            checksum VARCHAR(64)
        );
        
        CREATE INDEX IF NOT EXISTS idx_schema_migrations_executed_at 
        ON schema_migrations(executed_at);
        
        CREATE INDEX IF NOT EXISTS idx_schema_migrations_status 
        ON schema_migrations(status);
        """
        
        async with self._connection_pool.acquire() as conn:
            await conn.execute(create_table_sql)
        
        logger.info("Migration tracking table created/verified")
    
    async def _load_executed_migrations(self) -> None:
        """Load list of executed migrations from the database."""
        try:
            async with self._connection_pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT version, status FROM schema_migrations ORDER BY executed_at"
                )
                
                for row in rows:
                    if row['status'] == 'completed':
                        self._executed_migrations.add(row['version'])
                
                logger.info(f"Loaded {len(self._executed_migrations)} executed migrations")
                
        except Exception as e:
            logger.error(f"Failed to load executed migrations: {e}")
            raise MigrationError(f"Failed to load migration history: {e}") from e
    
    def register_migration(self, migration: MigrationScript) -> None:
        """
        Register a migration script.
        
        Args:
            migration: Migration script to register
        """
        if migration.version in self._migrations:
            raise ValueError(f"Migration {migration.version} already registered")
        
        self._migrations[migration.version] = migration
        logger.info(f"Registered migration {migration.version}: {migration.name}")
    
    def register_migrations_from_directory(self, directory: str) -> None:
        """
        Register all migration scripts from a directory.
        
        Args:
            directory: Directory containing migration files
        """
        directory_path = Path(directory)
        if not directory_path.exists():
            raise ValueError(f"Migration directory does not exist: {directory}")
        
        migration_files = sorted(directory_path.glob("*.sql"))
        
        for file_path in migration_files:
            try:
                migration = self._parse_migration_file(file_path)
                self.register_migration(migration)
            except Exception as e:
                logger.error(f"Failed to parse migration file {file_path}: {e}")
                raise MigrationError(f"Failed to parse migration file: {e}") from e
    
    def _parse_migration_file(self, file_path: Path) -> MigrationScript:
        """Parse a migration file and create a MigrationScript."""
        content = file_path.read_text()
        
        # Extract metadata from comments at the top of the file
        lines = content.split('\n')
        metadata = {}
        sql_lines = []
        in_metadata = True
        
        for line in lines:
            if in_metadata and line.startswith('-- '):
                key_value = line[3:].strip()
                if ':' in key_value:
                    key, value = key_value.split(':', 1)
                    metadata[key.strip().lower()] = value.strip()
                continue
            else:
                in_metadata = False
                sql_lines.append(line)
        
        # Split up and down SQL
        sql_content = '\n'.join(sql_lines)
        if '-- DOWN' in sql_content:
            up_sql, down_sql = sql_content.split('-- DOWN', 1)
        else:
            up_sql = sql_content
            down_sql = ""
        
        # Extract version from filename
        version = file_path.stem.split('_')[0]
        
        return MigrationScript(
            version=version,
            name=metadata.get('name', file_path.stem),
            description=metadata.get('description', ''),
            up_sql=up_sql.strip(),
            down_sql=down_sql.strip(),
            dependencies=metadata.get('dependencies', '').split(',') if metadata.get('dependencies') else [],
            created_at=datetime.now()
        )
    
    async def get_pending_migrations(self) -> List[MigrationScript]:
        """Get list of pending migrations in execution order."""
        pending = []
        
        for version in sorted(self._migrations.keys()):
            if version not in self._executed_migrations:
                pending.append(self._migrations[version])
        
        # Check dependencies
        for migration in pending:
            for dep in migration.dependencies:
                if dep not in self._executed_migrations and dep not in [m.version for m in pending]:
                    raise MigrationError(f"Migration {migration.version} depends on {dep} which is not available")
        
        return pending
    
    async def execute_migration(self, migration: MigrationScript) -> bool:
        """
        Execute a single migration script.
        
        Args:
            migration: Migration script to execute
            
        Returns:
            bool: True if successful, False otherwise
        """
        if migration.version in self._executed_migrations:
            logger.info(f"Migration {migration.version} already executed, skipping")
            return True
        
        logger.info(f"Executing migration {migration.version}: {migration.name}")
        start_time = datetime.now()
        
        async with self._connection_pool.acquire() as conn:
            async with conn.transaction():
                try:
                    # Record migration start
                    await conn.execute(
                        """
                        INSERT INTO schema_migrations (version, name, description, status, rollback_sql)
                        VALUES ($1, $2, $3, 'running', $4)
                        ON CONFLICT (version) DO UPDATE SET
                            status = 'running',
                            executed_at = NOW()
                        """,
                        migration.version,
                        migration.name,
                        migration.description,
                        migration.down_sql
                    )
                    
                    # Execute the migration SQL
                    await conn.execute(migration.up_sql)
                    
                    # Calculate execution time
                    execution_time = (datetime.now() - start_time).total_seconds() * 1000
                    
                    # Record successful completion
                    await conn.execute(
                        """
                        UPDATE schema_migrations 
                        SET status = 'completed', execution_time_ms = $2
                        WHERE version = $1
                        """,
                        migration.version,
                        int(execution_time)
                    )
                    
                    self._executed_migrations.add(migration.version)
                    logger.info(f"Migration {migration.version} completed successfully in {execution_time:.2f}ms")
                    return True
                    
                except Exception as e:
                    # Record failure
                    await conn.execute(
                        """
                        UPDATE schema_migrations 
                        SET status = 'failed'
                        WHERE version = $1
                        """,
                        migration.version
                    )
                    
                    logger.error(f"Migration {migration.version} failed: {e}")
                    raise MigrationError(f"Migration {migration.version} failed: {e}") from e
    
    async def rollback_migration(self, version: str) -> bool:
        """
        Rollback a migration.
        
        Args:
            version: Version of migration to rollback
            
        Returns:
            bool: True if successful, False otherwise
        """
        if version not in self._executed_migrations:
            logger.warning(f"Migration {version} not executed, cannot rollback")
            return False
        
        if version not in self._migrations:
            logger.error(f"Migration {version} not found, cannot rollback")
            return False
        
        migration = self._migrations[version]
        if not migration.down_sql:
            logger.error(f"Migration {version} has no rollback SQL")
            return False
        
        logger.info(f"Rolling back migration {version}: {migration.name}")
        
        async with self._connection_pool.acquire() as conn:
            async with conn.transaction():
                try:
                    # Execute rollback SQL
                    await conn.execute(migration.down_sql)
                    
                    # Update migration status
                    await conn.execute(
                        """
                        UPDATE schema_migrations 
                        SET status = 'rolled_back'
                        WHERE version = $1
                        """,
                        version
                    )
                    
                    self._executed_migrations.discard(version)
                    logger.info(f"Migration {version} rolled back successfully")
                    return True
                    
                except Exception as e:
                    logger.error(f"Failed to rollback migration {version}: {e}")
                    raise MigrationError(f"Rollback failed: {e}") from e
    
    async def execute_all_pending(self) -> Tuple[int, int]:
        """
        Execute all pending migrations.
        
        Returns:
            Tuple[int, int]: (successful_count, failed_count)
        """
        pending_migrations = await self.get_pending_migrations()
        
        if not pending_migrations:
            logger.info("No pending migrations to execute")
            return 0, 0
        
        logger.info(f"Executing {len(pending_migrations)} pending migrations")
        
        successful = 0
        failed = 0
        
        for migration in pending_migrations:
            try:
                if await self.execute_migration(migration):
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Failed to execute migration {migration.version}: {e}")
                failed += 1
                # Stop on first failure to maintain consistency
                break
        
        logger.info(f"Migration execution completed: {successful} successful, {failed} failed")
        return successful, failed
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get detailed migration status information."""
        try:
            async with self._connection_pool.acquire() as conn:
                # Get migration history
                rows = await conn.fetch(
                    """
                    SELECT version, name, description, executed_at, 
                           execution_time_ms, status
                    FROM schema_migrations 
                    ORDER BY executed_at DESC
                    """
                )
                
                migration_history = [
                    {
                        "version": row["version"],
                        "name": row["name"],
                        "description": row["description"],
                        "executed_at": row["executed_at"].isoformat() if row["executed_at"] else None,
                        "execution_time_ms": row["execution_time_ms"],
                        "status": row["status"]
                    }
                    for row in rows
                ]
                
                # Get pending migrations
                pending_migrations = await self.get_pending_migrations()
                
                return {
                    "total_migrations": len(self._migrations),
                    "executed_migrations": len(self._executed_migrations),
                    "pending_migrations": len(pending_migrations),
                    "migration_history": migration_history,
                    "pending_migration_list": [
                        {
                            "version": m.version,
                            "name": m.name,
                            "description": m.description
                        }
                        for m in pending_migrations
                    ]
                }
                
        except Exception as e:
            logger.error(f"Failed to get migration status: {e}")
            return {"error": str(e)}
    
    async def validate_schema(self) -> Dict[str, Any]:
        """Validate the current database schema."""
        try:
            async with self._connection_pool.acquire() as conn:
                # Check for required tables
                tables_query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                """
                
                table_rows = await conn.fetch(tables_query)
                existing_tables = {row["table_name"] for row in table_rows}
                
                # Expected tables after migration
                expected_tables = {
                    "checkpoints",
                    "schema_migrations"
                }
                
                missing_tables = expected_tables - existing_tables
                extra_tables = existing_tables - expected_tables
                
                # Check checkpoint table schema
                checkpoint_schema_valid = await self._validate_checkpoint_schema(conn)
                
                return {
                    "schema_valid": len(missing_tables) == 0 and checkpoint_schema_valid,
                    "existing_tables": list(existing_tables),
                    "missing_tables": list(missing_tables),
                    "extra_tables": list(extra_tables),
                    "checkpoint_schema_valid": checkpoint_schema_valid
                }
                
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return {
                "schema_valid": False,
                "error": str(e)
            }
    
    async def _validate_checkpoint_schema(self, conn) -> bool:
        """Validate the checkpoint table schema."""
        try:
            # Check if checkpoints table has expected columns
            columns_query = """
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'checkpoints' AND table_schema = 'public'
            """
            
            column_rows = await conn.fetch(columns_query)
            existing_columns = {row["column_name"]: row["data_type"] for row in column_rows}
            
            # Expected columns for LangGraph checkpoints
            expected_columns = {
                "thread_id": "text",
                "checkpoint_id": "text", 
                "parent_checkpoint_id": "text",
                "checkpoint": "jsonb",
                "metadata": "jsonb",
                "created_at": "timestamp with time zone"
            }
            
            # Check if all expected columns exist with correct types
            for col_name, expected_type in expected_columns.items():
                if col_name not in existing_columns:
                    logger.error(f"Missing column in checkpoints table: {col_name}")
                    return False
                
                actual_type = existing_columns[col_name]
                if not self._types_compatible(actual_type, expected_type):
                    logger.error(f"Column {col_name} has type {actual_type}, expected {expected_type}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Checkpoint schema validation failed: {e}")
            return False
    
    def _types_compatible(self, actual_type: str, expected_type: str) -> bool:
        """Check if database types are compatible."""
        # Simple type compatibility check
        type_mappings = {
            "character varying": "text",
            "varchar": "text",
            "timestamptz": "timestamp with time zone"
        }
        
        normalized_actual = type_mappings.get(actual_type, actual_type)
        normalized_expected = type_mappings.get(expected_type, expected_type)
        
        return normalized_actual == normalized_expected


def create_state_schema_migrations() -> List[MigrationScript]:
    """Create migration scripts for state schema changes."""
    migrations = []
    
    # Migration 1: Update checkpoint table for new state format
    migration_001 = MigrationScript(
        version="001",
        name="update_checkpoint_schema_for_pydantic_state",
        description="Update checkpoint table schema to support new Pydantic state format",
        up_sql="""
        -- Add columns for enhanced state tracking
        ALTER TABLE checkpoints 
        ADD COLUMN IF NOT EXISTS state_version VARCHAR(10) DEFAULT '2.0',
        ADD COLUMN IF NOT EXISTS workflow_step VARCHAR(50),
        ADD COLUMN IF NOT EXISTS error_count INTEGER DEFAULT 0,
        ADD COLUMN IF NOT EXISTS retry_count INTEGER DEFAULT 0,
        ADD COLUMN IF NOT EXISTS last_error TEXT;
        
        -- Add indexes for better query performance
        CREATE INDEX IF NOT EXISTS idx_checkpoints_state_version 
        ON checkpoints(state_version);
        
        CREATE INDEX IF NOT EXISTS idx_checkpoints_workflow_step 
        ON checkpoints(workflow_step);
        
        CREATE INDEX IF NOT EXISTS idx_checkpoints_error_count 
        ON checkpoints(error_count);
        
        -- Add constraint to ensure state_version is valid
        ALTER TABLE checkpoints 
        ADD CONSTRAINT chk_state_version 
        CHECK (state_version IN ('1.0', '2.0'));
        """,
        down_sql="""
        -- Remove added columns and constraints
        ALTER TABLE checkpoints 
        DROP CONSTRAINT IF EXISTS chk_state_version;
        
        DROP INDEX IF EXISTS idx_checkpoints_error_count;
        DROP INDEX IF EXISTS idx_checkpoints_workflow_step;
        DROP INDEX IF EXISTS idx_checkpoints_state_version;
        
        ALTER TABLE checkpoints 
        DROP COLUMN IF EXISTS last_error,
        DROP COLUMN IF EXISTS retry_count,
        DROP COLUMN IF EXISTS error_count,
        DROP COLUMN IF EXISTS workflow_step,
        DROP COLUMN IF EXISTS state_version;
        """,
        dependencies=[],
        created_at=datetime.now()
    )
    migrations.append(migration_001)
    
    # Migration 2: Create performance metrics table
    migration_002 = MigrationScript(
        version="002", 
        name="create_performance_metrics_table",
        description="Create table for storing performance metrics",
        up_sql="""
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(255) NOT NULL,
            step_name VARCHAR(100) NOT NULL,
            duration_ms INTEGER NOT NULL,
            success BOOLEAN NOT NULL DEFAULT true,
            error_message TEXT,
            resource_usage JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        CREATE INDEX idx_performance_metrics_session_id 
        ON performance_metrics(session_id);
        
        CREATE INDEX idx_performance_metrics_step_name 
        ON performance_metrics(step_name);
        
        CREATE INDEX idx_performance_metrics_created_at 
        ON performance_metrics(created_at);
        """,
        down_sql="""
        DROP TABLE IF EXISTS performance_metrics;
        """,
        dependencies=["001"],
        created_at=datetime.now()
    )
    migrations.append(migration_002)
    
    # Migration 3: Create workflow state audit table
    migration_003 = MigrationScript(
        version="003",
        name="create_workflow_state_audit_table", 
        description="Create table for auditing workflow state changes",
        up_sql="""
        CREATE TABLE IF NOT EXISTS workflow_state_audit (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(255) NOT NULL,
            thread_id VARCHAR(255) NOT NULL,
            checkpoint_id VARCHAR(255) NOT NULL,
            previous_step VARCHAR(50),
            current_step VARCHAR(50) NOT NULL,
            state_changes JSONB,
            error_info JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        CREATE INDEX idx_workflow_state_audit_session_id 
        ON workflow_state_audit(session_id);
        
        CREATE INDEX idx_workflow_state_audit_thread_id 
        ON workflow_state_audit(thread_id);
        
        CREATE INDEX idx_workflow_state_audit_current_step 
        ON workflow_state_audit(current_step);
        
        CREATE INDEX idx_workflow_state_audit_created_at 
        ON workflow_state_audit(created_at);
        """,
        down_sql="""
        DROP TABLE IF EXISTS workflow_state_audit;
        """,
        dependencies=["001", "002"],
        created_at=datetime.now()
    )
    migrations.append(migration_003)
    
    return migrations
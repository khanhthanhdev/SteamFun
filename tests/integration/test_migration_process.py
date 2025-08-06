"""
Integration tests for the migration process.

This module provides comprehensive integration tests for the migration
utilities including database migrations, configuration migrations,
and data validation.
"""

import pytest
import asyncio
import tempfile
import json
import yaml
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

try:
    from src.langgraph_agents.migration.migration_manager import MigrationManager, MigrationStatus
    from src.langgraph_agents.migration.database_migration import DatabaseMigrator, MigrationScript, create_state_schema_migrations
    from src.langgraph_agents.migration.config_migration import ConfigMigrator
    from src.langgraph_agents.migration.data_validator import DataValidator, ValidationSeverity
    from src.langgraph_agents.models.state import VideoGenerationState
    from src.langgraph_agents.models.config import WorkflowConfig
except ImportError:
    # Handle missing imports for testing
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    
    from langgraph_agents.migration.migration_manager import MigrationManager, MigrationStatus
    from langgraph_agents.migration.database_migration import DatabaseMigrator, MigrationScript, create_state_schema_migrations
    from langgraph_agents.migration.config_migration import ConfigMigrator
    from langgraph_agents.migration.data_validator import DataValidator, ValidationSeverity
    from langgraph_agents.models.state import VideoGenerationState
    from langgraph_agents.models.config import WorkflowConfig


class TestDatabaseMigration:
    """Test database migration functionality."""
    
    @pytest.fixture
    def mock_connection_string(self):
        """Mock PostgreSQL connection string."""
        return "postgresql://test:test@localhost:5432/test_db"
    
    @pytest.fixture
    def sample_migration_script(self):
        """Sample migration script for testing."""
        return MigrationScript(
            version="001",
            name="test_migration",
            description="Test migration script",
            up_sql="CREATE TABLE test_table (id SERIAL PRIMARY KEY);",
            down_sql="DROP TABLE test_table;",
            dependencies=[],
            created_at=datetime.now()
        )
    
    @pytest.mark.asyncio
    async def test_database_migrator_initialization(self, mock_connection_string):
        """Test database migrator initialization."""
        with patch('src.langgraph_agents.migration.database_migration.asyncpg') as mock_asyncpg:
            mock_pool = AsyncMock()
            mock_asyncpg.create_pool.return_value = mock_pool
            
            migrator = DatabaseMigrator(mock_connection_string)
            await migrator.initialize()
            
            assert migrator._connection_pool == mock_pool
            mock_asyncpg.create_pool.assert_called_once()
            await migrator.close()
    
    @pytest.mark.asyncio
    async def test_migration_script_registration(self, mock_connection_string, sample_migration_script):
        """Test migration script registration."""
        with patch('src.langgraph_agents.migration.database_migration.asyncpg'):
            migrator = DatabaseMigrator(mock_connection_string)
            migrator.register_migration(sample_migration_script)
            
            assert sample_migration_script.version in migrator._migrations
            assert migrator._migrations[sample_migration_script.version] == sample_migration_script
    
    @pytest.mark.asyncio
    async def test_pending_migrations_detection(self, mock_connection_string, sample_migration_script):
        """Test detection of pending migrations."""
        with patch('src.langgraph_agents.migration.database_migration.asyncpg'):
            migrator = DatabaseMigrator(mock_connection_string)
            migrator.register_migration(sample_migration_script)
            
            pending = await migrator.get_pending_migrations()
            assert len(pending) == 1
            assert pending[0] == sample_migration_script
    
    @pytest.mark.asyncio
    async def test_migration_execution(self, mock_connection_string, sample_migration_script):
        """Test migration execution."""
        with patch('src.langgraph_agents.migration.database_migration.asyncpg') as mock_asyncpg:
            mock_pool = AsyncMock()
            mock_conn = AsyncMock()
            mock_transaction = AsyncMock()
            
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            mock_conn.transaction.return_value = mock_transaction
            mock_asyncpg.create_pool.return_value = mock_pool
            
            migrator = DatabaseMigrator(mock_connection_string)
            migrator._connection_pool = mock_pool
            migrator._executed_migrations = set()
            
            result = await migrator.execute_migration(sample_migration_script)
            
            assert result is True
            assert sample_migration_script.version in migrator._executed_migrations
            mock_conn.execute.assert_called()
    
    def test_create_state_schema_migrations(self):
        """Test creation of default state schema migrations."""
        migrations = create_state_schema_migrations()
        
        assert len(migrations) >= 3  # Should have at least 3 migrations
        assert all(isinstance(m, MigrationScript) for m in migrations)
        assert all(m.up_sql and m.down_sql for m in migrations)
        
        # Check migration versions are sequential
        versions = [m.version for m in migrations]
        assert versions == sorted(versions)


class TestConfigMigration:
    """Test configuration migration functionality."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary directory for config files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sample_old_config(self):
        """Sample old configuration format."""
        return {
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
            "embedding_model": "hf:ibm-granite/granite-embedding-30m-english"
        }
    
    def test_config_migrator_initialization(self, temp_config_dir):
        """Test config migrator initialization."""
        migrator = ConfigMigrator(str(temp_config_dir))
        
        assert migrator.backup_directory == temp_config_dir
        assert temp_config_dir.exists()
    
    def test_single_config_file_migration(self, temp_config_dir, sample_old_config):
        """Test migration of a single configuration file."""
        # Create source config file
        source_file = temp_config_dir / "config.json"
        with open(source_file, 'w') as f:
            json.dump(sample_old_config, f)
        
        migrator = ConfigMigrator(str(temp_config_dir / "backups"))
        result = migrator.migrate_config_file(str(source_file))
        
        assert result.success
        assert result.backup_path is not None
        assert Path(result.target_path).exists()
        assert len(result.errors) == 0
    
    def test_config_directory_migration(self, temp_config_dir, sample_old_config):
        """Test migration of entire configuration directory."""
        # Create multiple config files
        config_files = ["config1.json", "config2.yml", "config3.yaml"]
        
        for config_file in config_files:
            file_path = temp_config_dir / config_file
            if config_file.endswith('.json'):
                with open(file_path, 'w') as f:
                    json.dump(sample_old_config, f)
            else:
                with open(file_path, 'w') as f:
                    yaml.dump(sample_old_config, f)
        
        migrator = ConfigMigrator(str(temp_config_dir / "backups"))
        results = migrator.migrate_config_directory(str(temp_config_dir))
        
        assert len(results) == len(config_files)
        assert all(r.success for r in results)
        assert all(Path(r.target_path).exists() for r in results)
    
    def test_config_validation(self, temp_config_dir, sample_old_config):
        """Test configuration validation."""
        # Create config file
        config_file = temp_config_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(sample_old_config, f)
        
        migrator = ConfigMigrator(str(temp_config_dir / "backups"))
        validation_result = migrator.validate_config_format(str(config_file))
        
        assert validation_result["valid"]
        assert validation_result["format"] == "old"
        assert len(validation_result["errors"]) == 0
    
    def test_migration_rollback(self, temp_config_dir, sample_old_config):
        """Test migration rollback functionality."""
        # Create and migrate config file
        source_file = temp_config_dir / "config.json"
        with open(source_file, 'w') as f:
            json.dump(sample_old_config, f)
        
        migrator = ConfigMigrator(str(temp_config_dir / "backups"))
        result = migrator.migrate_config_file(str(source_file))
        
        assert result.success
        
        # Test rollback
        rollback_success = migrator.rollback_migration(result)
        assert rollback_success
    
    def test_migration_plan_creation(self, temp_config_dir, sample_old_config):
        """Test creation of migration plan."""
        # Create config files
        for i in range(3):
            config_file = temp_config_dir / f"config{i}.json"
            with open(config_file, 'w') as f:
                json.dump(sample_old_config, f)
        
        migrator = ConfigMigrator(str(temp_config_dir / "backups"))
        plan = migrator.create_migration_plan(str(temp_config_dir))
        
        assert plan["total_files"] == 3
        assert plan["files_needing_migration"] == 3
        assert plan["estimated_duration_minutes"] > 0


class TestDataValidation:
    """Test data validation functionality."""
    
    @pytest.fixture
    def sample_old_state(self):
        """Sample old state format."""
        return {
            "topic": "Python Basics",
            "description": "Introduction to Python programming",
            "session_id": "test-session-123",
            "scene_outline": "Scene 1: Variables\nScene 2: Functions",
            "scene_implementations": {1: "Show variables", 2: "Show functions"},
            "generated_code": {1: "# Variable code", 2: "# Function code"},
            "rendered_videos": {1: "video1.mp4", 2: "video2.mp4"},
            "workflow_complete": False
        }
    
    @pytest.fixture
    def sample_new_state(self, sample_old_state):
        """Sample new state format."""
        return VideoGenerationState(
            topic=sample_old_state["topic"],
            description=sample_old_state["description"],
            session_id=sample_old_state["session_id"],
            scene_outline=sample_old_state["scene_outline"],
            scene_implementations=sample_old_state["scene_implementations"],
            generated_code=sample_old_state["generated_code"],
            rendered_videos=sample_old_state["rendered_videos"],
            workflow_complete=sample_old_state["workflow_complete"]
        )
    
    def test_data_validator_initialization(self):
        """Test data validator initialization."""
        validator = DataValidator()
        assert validator.validation_rules is not None
        assert "required_state_fields" in validator.validation_rules
    
    def test_state_migration_validation(self, sample_old_state, sample_new_state):
        """Test validation of state migration."""
        validator = DataValidator()
        
        old_states = [sample_old_state]
        new_states = [sample_new_state]
        
        report = validator.validate_state_migration(old_states, new_states)
        
        assert report.total_records == 1
        assert report.success_rate >= 0
        assert not report.has_critical_issues
    
    def test_config_migration_validation(self):
        """Test validation of configuration migration."""
        validator = DataValidator()
        
        old_config = {
            "use_rag": True,
            "max_retries": 3,
            "output_dir": "output",
            "enable_caching": True
        }
        
        new_config = WorkflowConfig(
            use_rag=True,
            max_retries=3,
            output_dir="output",
            enable_caching=True
        )
        
        report = validator.validate_config_migration([old_config], [new_config])
        
        assert report.total_records == 1
        assert report.success_rate >= 0
    
    def test_data_integrity_validation(self, sample_old_state):
        """Test data integrity validation."""
        validator = DataValidator()
        
        report = validator.validate_data_integrity([sample_old_state])
        
        assert report.total_records == 1
        assert report.data_type == "data_integrity"
    
    def test_validation_with_errors(self):
        """Test validation with intentional errors."""
        validator = DataValidator()
        
        # Create invalid state data
        invalid_state = {
            "topic": "",  # Empty topic should cause error
            "description": "Valid description",
            "session_id": "test-123"
        }
        
        report = validator.validate_data_integrity([invalid_state])
        
        assert report.total_records == 1
        assert report.valid_records == 0
        assert len(report.issues) > 0
        assert any(issue.severity == ValidationSeverity.CRITICAL for issue in report.issues)
    
    def test_validation_report_export(self, temp_config_dir, sample_old_state):
        """Test export of validation report."""
        validator = DataValidator()
        
        report = validator.validate_data_integrity([sample_old_state])
        output_file = temp_config_dir / "validation_report.json"
        
        validator.export_validation_report(report, str(output_file))
        
        assert output_file.exists()
        
        # Verify report content
        with open(output_file, 'r') as f:
            exported_data = json.load(f)
        
        assert exported_data["validation_id"] == report.validation_id
        assert exported_data["total_records"] == report.total_records


class TestMigrationManager:
    """Test migration manager functionality."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary directory for config files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sample_config_files(self, temp_config_dir):
        """Create sample configuration files."""
        config_data = {
            "use_rag": True,
            "max_retries": 3,
            "output_dir": "output"
        }
        
        config_files = []
        for i in range(2):
            config_file = temp_config_dir / f"config{i}.json"
            with open(config_file, 'w') as f:
                json.dump(config_data, f)
            config_files.append(str(config_file))
        
        return config_files
    
    @pytest.mark.asyncio
    async def test_migration_manager_initialization(self, temp_config_dir):
        """Test migration manager initialization."""
        manager = MigrationManager(
            database_connection_string=None,  # Skip database for this test
            config_backup_directory=str(temp_config_dir)
        )
        
        await manager.initialize()
        
        assert manager.config_migrator is not None
        assert manager.data_validator is not None
        
        await manager.close()
    
    def test_migration_plan_creation(self, temp_config_dir, sample_config_files):
        """Test creation of comprehensive migration plan."""
        manager = MigrationManager(
            database_connection_string=None,
            config_backup_directory=str(temp_config_dir)
        )
        
        plan = manager.create_migration_plan(
            config_directories=[str(temp_config_dir)],
            include_database=False,
            include_validation=True
        )
        
        assert plan.plan_id is not None
        assert len(plan.config_files) == len(sample_config_files)
        assert plan.validation_required is True
        assert plan.estimated_duration_minutes > 0
    
    @pytest.mark.asyncio
    async def test_migration_plan_execution(self, temp_config_dir, sample_config_files):
        """Test execution of migration plan."""
        manager = MigrationManager(
            database_connection_string=None,
            config_backup_directory=str(temp_config_dir)
        )
        
        await manager.initialize()
        
        plan = manager.create_migration_plan(
            config_directories=[str(temp_config_dir)],
            include_database=False,
            include_validation=True
        )
        
        result = await manager.execute_migration_plan(plan)
        
        assert result.plan_id == plan.plan_id
        assert result.status in [MigrationStatus.COMPLETED, MigrationStatus.FAILED]
        assert result.started_at is not None
        assert result.completed_at is not None
        
        await manager.close()
    
    @pytest.mark.asyncio
    async def test_migration_status_reporting(self, temp_config_dir):
        """Test migration status reporting."""
        manager = MigrationManager(
            database_connection_string=None,
            config_backup_directory=str(temp_config_dir)
        )
        
        await manager.initialize()
        
        status = await manager.get_migration_status()
        
        assert "current_plan" in status
        assert "migration_history" in status
        assert "database_status" in status
        assert "last_migration" in status
        
        await manager.close()
    
    def test_migration_report_export(self, temp_config_dir):
        """Test export of migration report."""
        from src.langgraph_agents.migration.migration_manager import MigrationResult
        
        manager = MigrationManager(
            database_connection_string=None,
            config_backup_directory=str(temp_config_dir)
        )
        
        # Create mock migration result
        result = MigrationResult(
            plan_id="test-plan",
            status=MigrationStatus.COMPLETED,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            database_migration_results={},
            config_migration_results=[],
            validation_reports=[],
            errors=[],
            warnings=[],
            rollback_available=False
        )
        
        output_file = temp_config_dir / "migration_report.json"
        manager.export_migration_report(result, str(output_file))
        
        assert output_file.exists()
        
        # Verify report content
        with open(output_file, 'r') as f:
            exported_data = json.load(f)
        
        assert exported_data["migration_report"]["plan_id"] == "test-plan"
        assert exported_data["migration_report"]["status"] == "completed"


class TestIntegrationScenarios:
    """Test complete integration scenarios."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for integration tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create directory structure
            (workspace / "config").mkdir()
            (workspace / "backups").mkdir()
            (workspace / "reports").mkdir()
            
            yield workspace
    
    @pytest.mark.asyncio
    async def test_complete_migration_workflow(self, temp_workspace):
        """Test complete migration workflow from start to finish."""
        # Create sample configuration files
        config_dir = temp_workspace / "config"
        sample_configs = [
            {
                "use_rag": True,
                "max_retries": 3,
                "output_dir": "output",
                "planner_model": "openrouter/anthropic/claude-3.5-sonnet"
            },
            {
                "use_rag": False,
                "max_retries": 5,
                "output_dir": "custom_output",
                "code_model": "openrouter/anthropic/claude-3.5-sonnet"
            }
        ]
        
        for i, config in enumerate(sample_configs):
            config_file = config_dir / f"config{i}.json"
            with open(config_file, 'w') as f:
                json.dump(config, f)
        
        # Initialize migration manager
        manager = MigrationManager(
            database_connection_string=None,  # Skip database for integration test
            config_backup_directory=str(temp_workspace / "backups")
        )
        
        await manager.initialize()
        
        try:
            # Create migration plan
            plan = manager.create_migration_plan(
                config_directories=[str(config_dir)],
                include_database=False,
                include_validation=True
            )
            
            assert plan.plan_id is not None
            assert len(plan.config_files) == 2
            
            # Execute migration plan
            result = await manager.execute_migration_plan(plan)
            
            assert result.plan_id == plan.plan_id
            assert result.completed_at is not None
            
            # Export migration report
            report_file = temp_workspace / "reports" / "migration_report.json"
            manager.export_migration_report(result, str(report_file))
            
            assert report_file.exists()
            
            # Verify migration results
            if result.status == MigrationStatus.COMPLETED:
                # Check that migrated config files exist
                for config_result in result.config_migration_results:
                    if config_result.success:
                        assert Path(config_result.target_path).exists()
                        assert config_result.backup_path is not None
                        assert Path(config_result.backup_path).exists()
            
            # Get final status
            final_status = await manager.get_migration_status()
            assert len(final_status["migration_history"]) == 1
            
        finally:
            await manager.close()
    
    @pytest.mark.asyncio
    async def test_migration_with_validation_errors(self, temp_workspace):
        """Test migration workflow with validation errors."""
        # Create invalid configuration file
        config_dir = temp_workspace / "config"
        invalid_config = {
            "use_rag": "invalid_boolean",  # Should be boolean
            "max_retries": "invalid_number",  # Should be number
            "output_dir": ""  # Should not be empty
        }
        
        config_file = config_dir / "invalid_config.json"
        with open(config_file, 'w') as f:
            json.dump(invalid_config, f)
        
        manager = MigrationManager(
            database_connection_string=None,
            config_backup_directory=str(temp_workspace / "backups")
        )
        
        await manager.initialize()
        
        try:
            plan = manager.create_migration_plan(
                config_directories=[str(config_dir)],
                include_database=False,
                include_validation=True
            )
            
            result = await manager.execute_migration_plan(plan)
            
            # Should have errors due to invalid configuration
            assert len(result.errors) > 0 or len(result.warnings) > 0
            
            # Validation reports should contain issues
            if result.validation_reports:
                for report in result.validation_reports:
                    if report.has_errors or report.has_critical_issues:
                        assert len(report.issues) > 0
            
        finally:
            await manager.close()
    
    def test_migration_utilities_integration(self, temp_workspace):
        """Test integration between different migration utilities."""
        # Test that all migration utilities work together
        config_migrator = ConfigMigrator(str(temp_workspace / "backups"))
        data_validator = DataValidator()
        
        # Create sample data
        sample_config = {
            "use_rag": True,
            "max_retries": 3,
            "output_dir": "output"
        }
        
        config_file = temp_workspace / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(sample_config, f)
        
        # Migrate configuration
        migration_result = config_migrator.migrate_config_file(str(config_file))
        assert migration_result.success
        
        # Validate migrated data
        validation_report = data_validator.validate_data_integrity([sample_config])
        assert validation_report.total_records == 1
        
        # Export validation report
        report_file = temp_workspace / "validation_report.json"
        data_validator.export_validation_report(validation_report, str(report_file))
        assert report_file.exists()
        
        # Test rollback
        rollback_success = config_migrator.rollback_migration(migration_result)
        assert rollback_success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
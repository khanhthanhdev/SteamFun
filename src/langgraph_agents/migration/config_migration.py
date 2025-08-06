"""
Configuration file migration tools.

This module provides utilities for migrating configuration files from old
formats to new formats, including validation and backup functionality.
"""

import logging
import json
import yaml
import shutil
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

from ..adapters.config_adapter import ConfigAdapter
from ..models.config import WorkflowConfig

logger = logging.getLogger(__name__)


class ConfigMigrationError(Exception):
    """Exception raised during configuration migration operations."""
    pass


@dataclass
class ConfigMigrationResult:
    """Result of a configuration migration operation."""
    success: bool
    source_path: str
    target_path: str
    backup_path: Optional[str]
    errors: List[str]
    warnings: List[str]
    migration_report: Dict[str, Any]


class ConfigMigrator:
    """
    Configuration file migrator for LangGraph agents refactor.
    
    This class handles migration of configuration files from old formats
    to new formats, including validation, backup, and rollback functionality.
    """
    
    def __init__(self, backup_directory: Optional[str] = None):
        """
        Initialize configuration migrator.
        
        Args:
            backup_directory: Directory to store configuration backups
        """
        self.backup_directory = Path(backup_directory) if backup_directory else Path("config_backups")
        self.backup_directory.mkdir(exist_ok=True)
        
        logger.info(f"Configuration migrator initialized with backup directory: {self.backup_directory}")
    
    def migrate_config_file(
        self, 
        source_path: str, 
        target_path: Optional[str] = None,
        create_backup: bool = True,
        validate_migration: bool = True
    ) -> ConfigMigrationResult:
        """
        Migrate a single configuration file.
        
        Args:
            source_path: Path to source configuration file
            target_path: Path to target configuration file (optional)
            create_backup: Whether to create a backup of the original file
            validate_migration: Whether to validate the migrated configuration
            
        Returns:
            ConfigMigrationResult: Result of the migration operation
        """
        source_path = Path(source_path)
        errors = []
        warnings = []
        backup_path = None
        
        try:
            # Validate source file exists
            if not source_path.exists():
                raise ConfigMigrationError(f"Source configuration file does not exist: {source_path}")
            
            # Determine target path
            if target_path is None:
                target_path = source_path.parent / f"{source_path.stem}_migrated{source_path.suffix}"
            else:
                target_path = Path(target_path)
            
            # Create backup if requested
            if create_backup:
                backup_path = self._create_backup(source_path)
                logger.info(f"Created backup: {backup_path}")
            
            # Load source configuration
            source_config = self._load_config_file(source_path)
            
            # Migrate configuration using ConfigAdapter
            migrated_config = ConfigAdapter.migrate_system_config(source_config)
            
            # Save migrated configuration
            self._save_config_file(migrated_config, target_path)
            
            # Validate migration if requested
            migration_report = {}
            if validate_migration:
                validation_result = self._validate_migration(source_config, migrated_config)
                if not validation_result["valid"]:
                    errors.extend(validation_result["errors"])
                warnings.extend(validation_result["warnings"])
                migration_report = validation_result
            
            # Create migration report
            if not migration_report:
                migration_report = ConfigAdapter.create_migration_report(source_config, migrated_config)
            
            logger.info(f"Successfully migrated configuration: {source_path} -> {target_path}")
            
            return ConfigMigrationResult(
                success=len(errors) == 0,
                source_path=str(source_path),
                target_path=str(target_path),
                backup_path=str(backup_path) if backup_path else None,
                errors=errors,
                warnings=warnings,
                migration_report=migration_report
            )
            
        except Exception as e:
            logger.error(f"Failed to migrate configuration file {source_path}: {e}")
            errors.append(str(e))
            
            return ConfigMigrationResult(
                success=False,
                source_path=str(source_path),
                target_path=str(target_path) if 'target_path' in locals() else "",
                backup_path=str(backup_path) if backup_path else None,
                errors=errors,
                warnings=warnings,
                migration_report={}
            )
    
    def migrate_config_directory(
        self,
        source_directory: str,
        target_directory: Optional[str] = None,
        file_patterns: List[str] = None,
        create_backup: bool = True
    ) -> List[ConfigMigrationResult]:
        """
        Migrate all configuration files in a directory.
        
        Args:
            source_directory: Source directory containing configuration files
            target_directory: Target directory for migrated files (optional)
            file_patterns: List of file patterns to match (default: ['*.json', '*.yml', '*.yaml'])
            create_backup: Whether to create backups of original files
            
        Returns:
            List[ConfigMigrationResult]: Results of migration operations
        """
        source_dir = Path(source_directory)
        if not source_dir.exists():
            raise ConfigMigrationError(f"Source directory does not exist: {source_directory}")
        
        if target_directory is None:
            target_dir = source_dir / "migrated"
        else:
            target_dir = Path(target_directory)
        
        target_dir.mkdir(exist_ok=True)
        
        # Default file patterns
        if file_patterns is None:
            file_patterns = ['*.json', '*.yml', '*.yaml']
        
        # Find configuration files
        config_files = []
        for pattern in file_patterns:
            config_files.extend(source_dir.glob(pattern))
        
        logger.info(f"Found {len(config_files)} configuration files to migrate")
        
        # Migrate each file
        results = []
        for config_file in config_files:
            target_file = target_dir / config_file.name
            result = self.migrate_config_file(
                str(config_file),
                str(target_file),
                create_backup=create_backup
            )
            results.append(result)
        
        # Log summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        logger.info(f"Directory migration completed: {successful} successful, {failed} failed")
        
        return results
    
    def rollback_migration(self, migration_result: ConfigMigrationResult) -> bool:
        """
        Rollback a configuration migration.
        
        Args:
            migration_result: Result of the original migration
            
        Returns:
            bool: True if rollback was successful
        """
        try:
            if not migration_result.backup_path:
                logger.error("No backup path available for rollback")
                return False
            
            backup_path = Path(migration_result.backup_path)
            target_path = Path(migration_result.target_path)
            
            if not backup_path.exists():
                logger.error(f"Backup file does not exist: {backup_path}")
                return False
            
            # Restore from backup
            shutil.copy2(backup_path, target_path)
            
            logger.info(f"Successfully rolled back migration: {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback migration: {e}")
            return False
    
    def validate_config_format(self, config_path: str) -> Dict[str, Any]:
        """
        Validate a configuration file format.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dict[str, Any]: Validation result
        """
        try:
            config_path = Path(config_path)
            
            if not config_path.exists():
                return {
                    "valid": False,
                    "errors": [f"Configuration file does not exist: {config_path}"],
                    "warnings": []
                }
            
            # Load and validate configuration
            config_data = self._load_config_file(config_path)
            
            # Try to create WorkflowConfig from the data
            try:
                WorkflowConfig(**config_data)
                return {
                    "valid": True,
                    "errors": [],
                    "warnings": [],
                    "format": "new"
                }
            except Exception:
                # Try to migrate and validate
                try:
                    migrated_config = ConfigAdapter.migrate_system_config(config_data)
                    return {
                        "valid": True,
                        "errors": [],
                        "warnings": ["Configuration is in old format and needs migration"],
                        "format": "old"
                    }
                except Exception as e:
                    return {
                        "valid": False,
                        "errors": [f"Configuration validation failed: {e}"],
                        "warnings": [],
                        "format": "unknown"
                    }
                    
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Failed to validate configuration: {e}"],
                "warnings": [],
                "format": "unknown"
            }
    
    def create_migration_plan(self, source_directory: str) -> Dict[str, Any]:
        """
        Create a migration plan for a directory of configuration files.
        
        Args:
            source_directory: Directory containing configuration files
            
        Returns:
            Dict[str, Any]: Migration plan
        """
        source_dir = Path(source_directory)
        if not source_dir.exists():
            raise ConfigMigrationError(f"Source directory does not exist: {source_directory}")
        
        # Find configuration files
        config_files = []
        for pattern in ['*.json', '*.yml', '*.yaml']:
            config_files.extend(source_dir.glob(pattern))
        
        plan = {
            "source_directory": str(source_dir),
            "total_files": len(config_files),
            "files": [],
            "estimated_duration_minutes": len(config_files) * 0.5,  # Estimate 30 seconds per file
            "backup_required": True,
            "validation_required": True
        }
        
        # Analyze each file
        for config_file in config_files:
            validation_result = self.validate_config_format(str(config_file))
            
            file_info = {
                "path": str(config_file),
                "name": config_file.name,
                "size_bytes": config_file.stat().st_size,
                "format": validation_result.get("format", "unknown"),
                "needs_migration": validation_result.get("format") == "old",
                "validation_errors": validation_result.get("errors", []),
                "validation_warnings": validation_result.get("warnings", [])
            }
            
            plan["files"].append(file_info)
        
        # Calculate summary statistics
        plan["files_needing_migration"] = sum(1 for f in plan["files"] if f["needs_migration"])
        plan["files_with_errors"] = sum(1 for f in plan["files"] if f["validation_errors"])
        plan["files_with_warnings"] = sum(1 for f in plan["files"] if f["validation_warnings"])
        
        return plan
    
    def _create_backup(self, source_path: Path) -> Path:
        """Create a backup of the source configuration file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{source_path.stem}_backup_{timestamp}{source_path.suffix}"
        backup_path = self.backup_directory / backup_filename
        
        shutil.copy2(source_path, backup_path)
        return backup_path
    
    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            if config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    return json.load(f)
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                raise ConfigMigrationError(f"Unsupported configuration file format: {config_path.suffix}")
                
        except Exception as e:
            raise ConfigMigrationError(f"Failed to load configuration file {config_path}: {e}") from e
    
    def _save_config_file(self, config: WorkflowConfig, target_path: Path) -> None:
        """Save configuration to file."""
        try:
            config_dict = config.model_dump()
            
            if target_path.suffix.lower() == '.json':
                with open(target_path, 'w') as f:
                    json.dump(config_dict, f, indent=2, default=str)
            elif target_path.suffix.lower() in ['.yml', '.yaml']:
                with open(target_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False)
            else:
                # Default to JSON
                target_path = target_path.with_suffix('.json')
                with open(target_path, 'w') as f:
                    json.dump(config_dict, f, indent=2, default=str)
                    
        except Exception as e:
            raise ConfigMigrationError(f"Failed to save configuration file {target_path}: {e}") from e
    
    def _validate_migration(self, source_config: Dict[str, Any], migrated_config: WorkflowConfig) -> Dict[str, Any]:
        """Validate that migration preserved essential settings."""
        errors = []
        warnings = []
        
        try:
            # Use ConfigAdapter validation
            if not ConfigAdapter.validate_migrated_config(source_config, migrated_config):
                errors.append("Configuration migration validation failed")
            
            # Additional validation checks
            essential_settings = [
                ('use_rag', 'use_rag'),
                ('max_retries', 'max_retries'),
                ('output_dir', 'output_dir'),
                ('enable_caching', 'enable_caching')
            ]
            
            for old_key, new_key in essential_settings:
                old_value = source_config.get(old_key)
                new_value = getattr(migrated_config, new_key, None)
                
                if old_value is not None and old_value != new_value:
                    warnings.append(f"Setting {old_key} changed from {old_value} to {new_value}")
            
            # Check for deprecated settings
            deprecated_settings = [
                'max_topic_concurrency',
                'enable_quality_monitoring',
                'rag_performance_threshold'
            ]
            
            for setting in deprecated_settings:
                if setting in source_config:
                    warnings.append(f"Deprecated setting '{setting}' was present in source configuration")
            
            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings
            }
            
        except Exception as e:
            logger.error(f"Migration validation failed: {e}")
            return {
                "valid": False,
                "errors": [f"Validation error: {e}"],
                "warnings": warnings
            }
    
    def cleanup_backups(self, max_age_days: int = 30) -> int:
        """
        Clean up old backup files.
        
        Args:
            max_age_days: Maximum age of backup files in days
            
        Returns:
            int: Number of backup files cleaned up
        """
        try:
            cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 3600)
            cleaned_count = 0
            
            for backup_file in self.backup_directory.glob("*_backup_*"):
                if backup_file.stat().st_mtime < cutoff_time:
                    backup_file.unlink()
                    cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} old backup files")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup backup files: {e}")
            return 0
    
    def get_migration_summary(self, results: List[ConfigMigrationResult]) -> Dict[str, Any]:
        """
        Generate a summary of migration results.
        
        Args:
            results: List of migration results
            
        Returns:
            Dict[str, Any]: Migration summary
        """
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        all_errors = []
        all_warnings = []
        
        for result in results:
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
        
        return {
            "total_files": len(results),
            "successful_migrations": successful,
            "failed_migrations": failed,
            "success_rate": (successful / len(results)) * 100 if results else 0,
            "total_errors": len(all_errors),
            "total_warnings": len(all_warnings),
            "errors": all_errors,
            "warnings": all_warnings,
            "migration_timestamp": datetime.now().isoformat()
        }
#!/usr/bin/env python3
"""
AWS RDS Database Initialization Script

This script initializes the database schema for AWS RDS PostgreSQL,
including tables, indexes, extensions, and optimizations.

Usage:
    python scripts/init_rds_database.py [--environment prod|dev|test] [--force]
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.models.database.rds_deployment import RDSDeployment
from app.models.database.migrations import DatabaseMigration, create_migration_script
from app.config.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_environment(environment: str) -> None:
    """Setup environment variables for the specified environment.
    
    Args:
        environment: Target environment (prod, dev, test)
    """
    env_files = {
        'prod': '.env.production',
        'dev': '.env.development', 
        'test': '.env.test'
    }
    
    env_file = env_files.get(environment, '.env')
    env_path = project_root / env_file
    
    if env_path.exists():
        logger.info(f"Loading environment from {env_file}")
        # Load environment variables from file
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    else:
        logger.warning(f"Environment file {env_file} not found")


def validate_rds_connection() -> bool:
    """Validate RDS connection before proceeding.
    
    Returns:
        True if connection is valid, False otherwise
    """
    try:
        deployment = RDSDeployment()
        health = deployment.health_check()
        
        if health["connection_successful"]:
            logger.info("✅ RDS connection validated")
            return True
        else:
            logger.error("❌ RDS connection failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Connection validation error: {e}")
        return False


def create_backup(deployment: RDSDeployment) -> str:
    """Create a backup before making changes.
    
    Args:
        deployment: RDS deployment instance
        
    Returns:
        Backup file path
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"backups/rds_schema_backup_{timestamp}.sql"
    
    # Create backups directory if it doesn't exist
    os.makedirs("backups", exist_ok=True)
    
    try:
        deployment.backup_schema(backup_path)
        logger.info(f"✅ Schema backup created: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"❌ Backup failed: {e}")
        raise


def initialize_database(environment: str, force: bool = False) -> bool:
    """Initialize the database with full schema and optimizations.
    
    Args:
        environment: Target environment
        force: Force initialization even if tables exist
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"🚀 Initializing RDS database for {environment} environment")
        
        # Setup environment
        setup_environment(environment)
        
        # Validate connection
        if not validate_rds_connection():
            return False
        
        # Initialize deployment
        deployment = RDSDeployment()
        
        # Check if database already has tables
        migration = deployment.migration
        existing_tables = migration.get_table_info()
        
        if existing_tables and not force:
            logger.warning("⚠️  Database already contains tables. Use --force to override.")
            logger.info("Existing tables:")
            for table_name in existing_tables.keys():
                logger.info(f"  - {table_name}")
            return False
        
        # Create backup if tables exist
        if existing_tables:
            logger.info("📦 Creating backup before initialization...")
            create_backup(deployment)
        
        # Deploy database
        result = deployment.deploy()
        
        if result["success"]:
            logger.info("✅ Database initialization completed successfully!")
            
            # Display validation results
            validation = result["validation"]
            logger.info("📊 Database Status:")
            logger.info(f"  SSL Enabled: {'✅' if validation.get('ssl_enabled') else '❌'}")
            logger.info(f"  Extensions: {len(validation.get('extensions_available', []))}/3")
            logger.info(f"  Performance Insights: {'✅' if validation.get('performance_insights') else '❌'}")
            
            # Show recommendations
            if validation.get("recommendations"):
                logger.info("📋 Recommendations:")
                for rec in validation["recommendations"]:
                    logger.info(f"  - {rec}")
            
            return True
        else:
            logger.error(f"❌ Database initialization failed: {result['error']}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Initialization error: {e}")
        return False


def create_migration_files(environment: str) -> None:
    """Create migration files for the current schema.
    
    Args:
        environment: Target environment
    """
    try:
        setup_environment(environment)
        settings = get_settings()
        database_url = os.getenv('DATABASE_URL', 'postgresql://localhost/fastapi_app')
        
        # Create migration script
        migration_script = create_migration_script(database_url, f"initial_{environment}")
        
        # Save migration script
        migration_dir = project_root / "migrations"
        migration_dir.mkdir(exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        migration_file = migration_dir / f"{timestamp}_initial_{environment}.py"
        
        with open(migration_file, 'w') as f:
            f.write(migration_script)
        
        logger.info(f"✅ Migration file created: {migration_file}")
        
    except Exception as e:
        logger.error(f"❌ Migration file creation failed: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Initialize AWS RDS database")
    parser.add_argument(
        "--environment", 
        choices=["prod", "dev", "test"], 
        default="dev",
        help="Target environment (default: dev)"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force initialization even if tables exist"
    )
    parser.add_argument(
        "--create-migration",
        action="store_true", 
        help="Create migration files only"
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Perform health check only"
    )
    
    args = parser.parse_args()
    
    if args.health_check:
        setup_environment(args.environment)
        if validate_rds_connection():
            deployment = RDSDeployment()
            health = deployment.health_check()
            print("🏥 RDS Health Check Results:")
            for key, value in health.items():
                status = "✅" if value else "❌" if isinstance(value, bool) else "ℹ️"
                print(f"  {status} {key}: {value}")
        sys.exit(0)
    
    if args.create_migration:
        create_migration_files(args.environment)
        sys.exit(0)
    
    # Initialize database
    success = initialize_database(args.environment, args.force)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
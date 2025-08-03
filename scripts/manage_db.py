#!/usr/bin/env python3
"""
Database management CLI script.

This script provides utilities for managing the database schema and migrations.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.models.database.migrations import DatabaseMigration
from app.config.database import db_config


def create_tables(args):
    """Create all database tables."""
    print("Creating database tables...")
    try:
        db_config.create_tables()
        print("✅ Database tables created successfully")
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return 1
    return 0


def drop_tables(args):
    """Drop all database tables."""
    if not args.force:
        response = input("⚠️  This will delete all data. Are you sure? (y/N): ")
        if response.lower() != 'y':
            print("Operation cancelled")
            return 0
    
    print("Dropping database tables...")
    try:
        db_config.drop_tables()
        print("✅ Database tables dropped successfully")
    except Exception as e:
        print(f"❌ Error dropping tables: {e}")
        return 1
    return 0


def recreate_tables(args):
    """Recreate all database tables."""
    if not args.force:
        response = input("⚠️  This will delete all data and recreate tables. Are you sure? (y/N): ")
        if response.lower() != 'y':
            print("Operation cancelled")
            return 0
    
    print("Recreating database tables...")
    try:
        migration = DatabaseMigration(db_config.database_url)
        migration.recreate_all_tables()
        print("✅ Database tables recreated successfully")
    except Exception as e:
        print(f"❌ Error recreating tables: {e}")
        return 1
    return 0


def show_info(args):
    """Show database information."""
    print("📊 Database Information")
    print("=" * 50)
    
    try:
        # Database configuration
        print(f"Database URL: {db_config.database_url}")
        
        # Health check
        health = db_config.health_check()
        print(f"Status: {health['status']}")
        
        if health['status'] == 'healthy':
            print(f"Pool Size: {health.get('engine_pool_size', 'N/A')}")
            print(f"Checked Out: {health.get('engine_pool_checked_out', 'N/A')}")
        else:
            print(f"Error: {health.get('error', 'Unknown error')}")
        
        # Table information
        migration = DatabaseMigration(db_config.database_url)
        table_info = migration.get_table_info()
        
        print(f"\n📋 Tables ({len(table_info)}):")
        for table_name, info in table_info.items():
            print(f"  • {table_name}")
            print(f"    Columns: {len(info['columns'])}")
            print(f"    Primary Keys: {', '.join(info['primary_keys'])}")
            if info['foreign_keys']:
                print(f"    Foreign Keys: {len(info['foreign_keys'])}")
        
    except Exception as e:
        print(f"❌ Error getting database info: {e}")
        return 1
    
    return 0


def health_check(args):
    """Perform database health check."""
    print("🏥 Database Health Check")
    print("=" * 30)
    
    try:
        health = db_config.health_check()
        
        if health['status'] == 'healthy':
            print("✅ Database is healthy")
            print(f"   URL: {health.get('database_url', 'N/A')}")
            if 'engine_pool_size' in health:
                print(f"   Pool Size: {health['engine_pool_size']}")
            if 'engine_pool_checked_out' in health:
                print(f"   Checked Out: {health['engine_pool_checked_out']}")
        else:
            print("❌ Database is unhealthy")
            print(f"   Error: {health.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return 1
    
    return 0


def generate_migration(args):
    """Generate a migration script."""
    print("📝 Generating migration script...")
    
    try:
        from app.models.database.migrations import create_migration_script
        
        script_content = create_migration_script(db_config.database_url)
        
        # Create migrations directory if it doesn't exist
        migrations_dir = project_root / "migrations"
        migrations_dir.mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"migration_{timestamp}.py"
        filepath = migrations_dir / filename
        
        # Write migration script
        with open(filepath, 'w') as f:
            f.write(script_content)
        
        print(f"✅ Migration script generated: {filepath}")
        
    except Exception as e:
        print(f"❌ Error generating migration: {e}")
        return 1
    
    return 0


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Database management utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/manage_db.py create     # Create all tables
  python scripts/manage_db.py drop      # Drop all tables
  python scripts/manage_db.py recreate  # Recreate all tables
  python scripts/manage_db.py info      # Show database info
  python scripts/manage_db.py health    # Health check
  python scripts/manage_db.py migrate   # Generate migration script
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create tables command
    create_parser = subparsers.add_parser('create', help='Create all database tables')
    create_parser.set_defaults(func=create_tables)
    
    # Drop tables command
    drop_parser = subparsers.add_parser('drop', help='Drop all database tables')
    drop_parser.add_argument('--force', action='store_true', help='Skip confirmation prompt')
    drop_parser.set_defaults(func=drop_tables)
    
    # Recreate tables command
    recreate_parser = subparsers.add_parser('recreate', help='Recreate all database tables')
    recreate_parser.add_argument('--force', action='store_true', help='Skip confirmation prompt')
    recreate_parser.set_defaults(func=recreate_tables)
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show database information')
    info_parser.set_defaults(func=show_info)
    
    # Health check command
    health_parser = subparsers.add_parser('health', help='Perform database health check')
    health_parser.set_defaults(func=health_check)
    
    # Generate migration command
    migrate_parser = subparsers.add_parser('migrate', help='Generate migration script')
    migrate_parser.set_defaults(func=generate_migration)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
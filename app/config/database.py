"""
Database configuration and connection management.
"""

import os
from typing import Optional, Generator
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from .settings import get_settings


class DatabaseConfig:
    """Database configuration class."""
    
    def __init__(self):
        """Initialize database configuration."""
        self.settings = get_settings()
        self._engine: Optional[Engine] = None
        self._SessionLocal: Optional[sessionmaker] = None
    
    @property
    def database_url(self) -> str:
        """Get database URL from settings with RDS support.
        
        Returns:
            Database connection URL
        """
        # Check for explicit database URL first
        if hasattr(self.settings, 'database') and hasattr(self.settings.database, 'database_url') and self.settings.database.database_url:
            return self.settings.database.database_url
        
        # Check environment variable
        if os.getenv('DATABASE_URL'):
            return os.getenv('DATABASE_URL')
        
        # Build URL from components
        db_type = getattr(self.settings, 'DB_TYPE', 'postgresql')  # Default to PostgreSQL for RDS
        
        if db_type.lower() == 'sqlite':
            db_path = getattr(self.settings, 'DB_PATH', 'app.db')
            return f"sqlite:///{db_path}"
        
        elif db_type.lower() == 'postgresql':
            # Support both direct settings and database settings object
            if hasattr(self.settings, 'database'):
                user = self.settings.database.database_user
                password = self.settings.database.database_password
                host = self.settings.database.database_host
                port = self.settings.database.database_port
                name = self.settings.database.database_name
            else:
                user = getattr(self.settings, 'DB_USER', 'postgres')
                password = getattr(self.settings, 'DB_PASSWORD', '')
                host = getattr(self.settings, 'DB_HOST', 'localhost')
                port = getattr(self.settings, 'DB_PORT', 5432)
                name = getattr(self.settings, 'DB_NAME', 'fastapi_app')
            
            # Add SSL parameters for RDS
            ssl_params = ""
            if 'rds.amazonaws.com' in host or os.getenv('AWS_RDS_ENDPOINT'):
                ssl_params = "?sslmode=require"
            
            return f"postgresql://{user}:{password}@{host}:{port}/{name}{ssl_params}"
        
        elif db_type.lower() == 'mysql':
            user = getattr(self.settings, 'DB_USER', 'root')
            password = getattr(self.settings, 'DB_PASSWORD', '')
            host = getattr(self.settings, 'DB_HOST', 'localhost')
            port = getattr(self.settings, 'DB_PORT', 3306)
            name = getattr(self.settings, 'DB_NAME', 'fastapi_app')
            return f"mysql+pymysql://{user}:{password}@{host}:{port}/{name}"
        
        else:
            # Default to PostgreSQL for RDS compatibility
            return "postgresql://postgres:@localhost:5432/fastapi_app"
    
    @property
    def engine(self) -> Engine:
        """Get database engine.
        
        Returns:
            SQLAlchemy engine instance
        """
        if not self._engine:
            self._engine = self._create_engine()
        return self._engine
    
    @property
    def SessionLocal(self) -> sessionmaker:
        """Get session factory.
        
        Returns:
            SQLAlchemy session factory
        """
        if not self._SessionLocal:
            self._SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
        return self._SessionLocal
    
    def _create_engine(self) -> Engine:
        """Create database engine with appropriate configuration.
        
        Returns:
            Configured SQLAlchemy engine
        """
        url = self.database_url
        
        # Base engine arguments
        engine_args = {
            "echo": getattr(self.settings, 'DB_ECHO', False),
            "pool_pre_ping": True,
        }
        
        # Database-specific configuration
        if url.startswith("sqlite"):
            engine_args.update({
                "poolclass": StaticPool,
                "connect_args": {
                    "check_same_thread": False,
                    "timeout": 20
                }
            })
        else:
            engine_args.update({
                "pool_size": getattr(self.settings, 'DB_POOL_SIZE', 10),
                "max_overflow": getattr(self.settings, 'DB_MAX_OVERFLOW', 20),
                "pool_recycle": getattr(self.settings, 'DB_POOL_RECYCLE', 3600),
                "pool_timeout": getattr(self.settings, 'DB_POOL_TIMEOUT', 30),
            })
        
        engine = create_engine(url, **engine_args)
        
        # Add event listeners for SQLite
        if url.startswith("sqlite"):
            @event.listens_for(engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                """Set SQLite pragmas for better performance and reliability."""
                cursor = dbapi_connection.cursor()
                # Enable foreign key constraints
                cursor.execute("PRAGMA foreign_keys=ON")
                # Set journal mode to WAL for better concurrency
                cursor.execute("PRAGMA journal_mode=WAL")
                # Set synchronous mode to NORMAL for better performance
                cursor.execute("PRAGMA synchronous=NORMAL")
                # Set cache size (negative value means KB)
                cursor.execute("PRAGMA cache_size=-64000")  # 64MB
                cursor.close()
        
        return engine
    
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup.
        
        Yields:
            Database session
        """
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()
    
    def create_tables(self) -> None:
        """Create all database tables."""
        from app.models.database.base import Base
        Base.metadata.create_all(bind=self.engine)
    
    def drop_tables(self) -> None:
        """Drop all database tables."""
        from app.models.database.base import Base
        Base.metadata.drop_all(bind=self.engine)
    
    def health_check(self) -> dict:
        """Perform database health check.
        
        Returns:
            Health check results
        """
        try:
            with self.engine.connect() as connection:
                result = connection.execute("SELECT 1")
                result.fetchone()
                
                return {
                    "status": "healthy",
                    "database_url": self.database_url.split("@")[-1] if "@" in self.database_url else self.database_url,
                    "engine_pool_size": getattr(self.engine.pool, 'size', None),
                    "engine_pool_checked_out": getattr(self.engine.pool, 'checkedout', None),
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "database_url": self.database_url.split("@")[-1] if "@" in self.database_url else self.database_url,
            }


# Global database configuration instance
db_config = DatabaseConfig()

# Convenience functions
def get_db() -> Generator[Session, None, None]:
    """Get database session dependency for FastAPI.
    
    Yields:
        Database session
    """
    yield from db_config.get_session()


def get_engine() -> Engine:
    """Get database engine.
    
    Returns:
        SQLAlchemy engine
    """
    return db_config.engine


def create_tables() -> None:
    """Create all database tables."""
    db_config.create_tables()


def drop_tables() -> None:
    """Drop all database tables."""
    db_config.drop_tables()


def health_check() -> dict:
    """Perform database health check.
    
    Returns:
        Health check results
    """
    return db_config.health_check()
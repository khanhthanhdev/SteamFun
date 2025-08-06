"""
Health check implementation for the LangGraph video generation workflow.

This module provides comprehensive health checks for all system components
including models, databases, and workflow components.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
import psycopg2
import redis
from pathlib import Path

from ..models.config import WorkflowConfig
from ..config.validation import ConfigValidator

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health check status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth:
    """Represents the health status of a system component."""
    
    def __init__(
        self,
        name: str,
        status: HealthStatus,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
        response_time_ms: Optional[float] = None
    ):
        self.name = name
        self.status = status
        self.message = message
        self.details = details or {}
        self.response_time_ms = response_time_ms
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "name": self.name,
            "status": self.status.value,
            "timestamp": self.timestamp
        }
        
        if self.message:
            result["message"] = self.message
        
        if self.details:
            result["details"] = self.details
        
        if self.response_time_ms is not None:
            result["response_time_ms"] = self.response_time_ms
        
        return result


class WorkflowHealthChecker:
    """Comprehensive health checker for the workflow system."""
    
    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.validator = ConfigValidator()
    
    async def check_all(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of all system components.
        
        Returns:
            Dictionary containing overall health status and component details
        """
        start_time = datetime.now()
        
        # Run all health checks concurrently
        checks = await asyncio.gather(
            self._check_configuration(),
            self._check_database(),
            self._check_redis(),
            self._check_models(),
            self._check_vector_store(),
            self._check_file_system(),
            self._check_workflow_components(),
            return_exceptions=True
        )
        
        # Process results
        component_health = []
        overall_status = HealthStatus.HEALTHY
        
        for check in checks:
            if isinstance(check, Exception):
                component_health.append(ComponentHealth(
                    name="unknown",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {check}"
                ))
                overall_status = HealthStatus.UNHEALTHY
            else:
                component_health.append(check)
                if check.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif check.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
        
        end_time = datetime.now()
        total_time_ms = (end_time - start_time).total_seconds() * 1000
        
        return {
            "status": overall_status.value,
            "timestamp": start_time.isoformat(),
            "total_check_time_ms": total_time_ms,
            "components": [comp.to_dict() for comp in component_health],
            "summary": self._generate_summary(component_health)
        }
    
    async def _check_configuration(self) -> ComponentHealth:
        """Check configuration validity."""
        start_time = datetime.now()
        
        try:
            # Validate current configuration
            errors = self.validator.validate_pydantic_config(self.config)
            
            if not errors:
                status = HealthStatus.HEALTHY
                message = "Configuration is valid"
            elif len(errors) <= 2:
                status = HealthStatus.DEGRADED
                message = f"Configuration has minor issues: {len(errors)} warnings"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Configuration has serious issues: {len(errors)} errors"
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return ComponentHealth(
                name="configuration",
                status=status,
                message=message,
                details={
                    "errors": errors[:5],  # Limit to first 5 errors
                    "total_errors": len(errors),
                    "config_version": getattr(self.config, 'version', 'unknown')
                },
                response_time_ms=response_time
            )
            
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            return ComponentHealth(
                name="configuration",
                status=HealthStatus.UNHEALTHY,
                message=f"Configuration check failed: {e}",
                response_time_ms=response_time
            )
    
    async def _check_database(self) -> ComponentHealth:
        """Check database connectivity and health."""
        start_time = datetime.now()
        
        try:
            # Try to connect to PostgreSQL
            import os
            db_url = os.getenv("DATABASE_URL", "")
            
            if not db_url:
                return ComponentHealth(
                    name="database",
                    status=HealthStatus.DEGRADED,
                    message="Database URL not configured"
                )
            
            # Parse connection string for psycopg2
            if db_url.startswith("postgresql+asyncpg://"):
                db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
            
            conn = psycopg2.connect(db_url)
            cursor = conn.cursor()
            
            # Test basic query
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            
            # Check if checkpoints table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'checkpoints'
                )
            """)
            checkpoints_exists = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if result and result[0] == 1:
                status = HealthStatus.HEALTHY if checkpoints_exists else HealthStatus.DEGRADED
                message = "Database connection successful"
                if not checkpoints_exists:
                    message += " (checkpoints table missing)"
            else:
                status = HealthStatus.UNHEALTHY
                message = "Database query failed"
            
            return ComponentHealth(
                name="database",
                status=status,
                message=message,
                details={
                    "checkpoints_table_exists": checkpoints_exists,
                    "connection_type": "postgresql"
                },
                response_time_ms=response_time
            )
            
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {e}",
                response_time_ms=response_time
            )
    
    async def _check_redis(self) -> ComponentHealth:
        """Check Redis connectivity and health."""
        start_time = datetime.now()
        
        try:
            import os
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            
            # Connect to Redis
            r = redis.from_url(redis_url)
            
            # Test basic operations
            test_key = "health_check_test"
            r.set(test_key, "test_value", ex=10)  # Expire in 10 seconds
            value = r.get(test_key)
            r.delete(test_key)
            
            # Get Redis info
            info = r.info()
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if value == b"test_value":
                status = HealthStatus.HEALTHY
                message = "Redis connection and operations successful"
            else:
                status = HealthStatus.DEGRADED
                message = "Redis connection successful but operations failed"
            
            return ComponentHealth(
                name="redis",
                status=status,
                message=message,
                details={
                    "redis_version": info.get("redis_version", "unknown"),
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory_human": info.get("used_memory_human", "unknown")
                },
                response_time_ms=response_time
            )
            
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            return ComponentHealth(
                name="redis",
                status=HealthStatus.DEGRADED,  # Redis is optional for some features
                message=f"Redis connection failed: {e}",
                response_time_ms=response_time
            )
    
    async def _check_models(self) -> ComponentHealth:
        """Check model availability and configuration."""
        start_time = datetime.now()
        
        try:
            model_checks = []
            
            # Check planner model
            planner_config = self.config.get_model_config("planner")
            model_checks.append({
                "type": "planner",
                "provider": planner_config.provider,
                "model": planner_config.model_name,
                "configured": True
            })
            
            # Check code model
            code_config = self.config.get_model_config("code")
            model_checks.append({
                "type": "code",
                "provider": code_config.provider,
                "model": code_config.model_name,
                "configured": True
            })
            
            # Check helper model if configured
            try:
                helper_config = self.config.get_model_config("helper")
                model_checks.append({
                    "type": "helper",
                    "provider": helper_config.provider,
                    "model": helper_config.model_name,
                    "configured": True
                })
            except:
                model_checks.append({
                    "type": "helper",
                    "configured": False
                })
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            configured_models = sum(1 for check in model_checks if check["configured"])
            
            if configured_models >= 2:  # At least planner and code models
                status = HealthStatus.HEALTHY
                message = f"Models configured: {configured_models}/3"
            elif configured_models >= 1:
                status = HealthStatus.DEGRADED
                message = f"Minimal models configured: {configured_models}/3"
            else:
                status = HealthStatus.UNHEALTHY
                message = "No models configured"
            
            return ComponentHealth(
                name="models",
                status=status,
                message=message,
                details={"model_configurations": model_checks},
                response_time_ms=response_time
            )
            
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            return ComponentHealth(
                name="models",
                status=HealthStatus.UNHEALTHY,
                message=f"Model configuration check failed: {e}",
                response_time_ms=response_time
            )
    
    async def _check_vector_store(self) -> ComponentHealth:
        """Check vector store availability."""
        start_time = datetime.now()
        
        try:
            if not self.config.use_rag:
                return ComponentHealth(
                    name="vector_store",
                    status=HealthStatus.HEALTHY,
                    message="Vector store disabled (RAG not enabled)"
                )
            
            vector_store_path = Path(self.config.rag_config.vector_store_path)
            
            if vector_store_path.exists():
                # Check if it's a valid ChromaDB directory
                chroma_files = list(vector_store_path.glob("*.sqlite3"))
                
                if chroma_files:
                    status = HealthStatus.HEALTHY
                    message = "Vector store available"
                    details = {
                        "path": str(vector_store_path),
                        "database_files": len(chroma_files)
                    }
                else:
                    status = HealthStatus.DEGRADED
                    message = "Vector store directory exists but no database files found"
                    details = {"path": str(vector_store_path)}
            else:
                status = HealthStatus.DEGRADED
                message = "Vector store path does not exist"
                details = {"path": str(vector_store_path)}
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return ComponentHealth(
                name="vector_store",
                status=status,
                message=message,
                details=details,
                response_time_ms=response_time
            )
            
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            return ComponentHealth(
                name="vector_store",
                status=HealthStatus.DEGRADED,
                message=f"Vector store check failed: {e}",
                response_time_ms=response_time
            )
    
    async def _check_file_system(self) -> ComponentHealth:
        """Check file system permissions and required directories."""
        start_time = datetime.now()
        
        try:
            checks = []
            
            # Check output directory
            output_dir = Path(self.config.output_dir)
            if output_dir.exists() and output_dir.is_dir():
                if os.access(output_dir, os.W_OK):
                    checks.append({"path": "output_dir", "status": "writable"})
                else:
                    checks.append({"path": "output_dir", "status": "not_writable"})
            else:
                checks.append({"path": "output_dir", "status": "missing"})
            
            # Check context learning path
            context_path = Path(self.config.context_learning_path)
            if context_path.exists():
                checks.append({"path": "context_learning", "status": "exists"})
            else:
                checks.append({"path": "context_learning", "status": "missing"})
            
            # Check manim docs path
            docs_path = Path(self.config.manim_docs_path)
            if docs_path.exists():
                checks.append({"path": "manim_docs", "status": "exists"})
            else:
                checks.append({"path": "manim_docs", "status": "missing"})
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Determine overall status
            critical_issues = sum(1 for check in checks if check["status"] in ["not_writable", "missing"] and check["path"] == "output_dir")
            minor_issues = sum(1 for check in checks if check["status"] == "missing" and check["path"] != "output_dir")
            
            if critical_issues > 0:
                status = HealthStatus.UNHEALTHY
                message = "Critical file system issues detected"
            elif minor_issues > 0:
                status = HealthStatus.DEGRADED
                message = f"Minor file system issues: {minor_issues} missing directories"
            else:
                status = HealthStatus.HEALTHY
                message = "File system checks passed"
            
            return ComponentHealth(
                name="file_system",
                status=status,
                message=message,
                details={"path_checks": checks},
                response_time_ms=response_time
            )
            
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            return ComponentHealth(
                name="file_system",
                status=HealthStatus.UNHEALTHY,
                message=f"File system check failed: {e}",
                response_time_ms=response_time
            )
    
    async def _check_workflow_components(self) -> ComponentHealth:
        """Check workflow-specific components."""
        start_time = datetime.now()
        
        try:
            components = []
            
            # Check if workflow graph can be imported
            try:
                from ..workflow_graph import create_workflow
                components.append({"component": "workflow_graph", "status": "importable"})
            except Exception as e:
                components.append({"component": "workflow_graph", "status": f"import_error: {e}"})
            
            # Check if node functions can be imported
            try:
                from ..nodes import planning_node, code_generation_node, rendering_node
                components.append({"component": "node_functions", "status": "importable"})
            except Exception as e:
                components.append({"component": "node_functions", "status": f"import_error: {e}"})
            
            # Check if services can be imported
            try:
                from ..services import PlanningService, CodeGenerationService, RenderingService
                components.append({"component": "services", "status": "importable"})
            except Exception as e:
                components.append({"component": "services", "status": f"import_error: {e}"})
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Determine status
            import_errors = sum(1 for comp in components if "import_error" in comp["status"])
            
            if import_errors == 0:
                status = HealthStatus.HEALTHY
                message = "All workflow components available"
            elif import_errors <= 1:
                status = HealthStatus.DEGRADED
                message = f"Some workflow components have issues: {import_errors} errors"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Multiple workflow component issues: {import_errors} errors"
            
            return ComponentHealth(
                name="workflow_components",
                status=status,
                message=message,
                details={"components": components},
                response_time_ms=response_time
            )
            
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            return ComponentHealth(
                name="workflow_components",
                status=HealthStatus.UNHEALTHY,
                message=f"Workflow component check failed: {e}",
                response_time_ms=response_time
            )
    
    def _generate_summary(self, components: List[ComponentHealth]) -> Dict[str, Any]:
        """Generate a summary of health check results."""
        total = len(components)
        healthy = sum(1 for comp in components if comp.status == HealthStatus.HEALTHY)
        degraded = sum(1 for comp in components if comp.status == HealthStatus.DEGRADED)
        unhealthy = sum(1 for comp in components if comp.status == HealthStatus.UNHEALTHY)
        
        return {
            "total_components": total,
            "healthy": healthy,
            "degraded": degraded,
            "unhealthy": unhealthy,
            "health_percentage": round((healthy / total) * 100, 1) if total > 0 else 0
        }


async def perform_health_check(config: Optional[WorkflowConfig] = None) -> Dict[str, Any]:
    """
    Perform a comprehensive health check of the workflow system.
    
    Args:
        config: Optional WorkflowConfig instance. If not provided, will try to load default.
        
    Returns:
        Dictionary containing health check results
    """
    if config is None:
        # Try to load default configuration
        try:
            from ..models.config import WorkflowConfig
            config = WorkflowConfig()
        except Exception as e:
            return {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": f"Failed to load configuration: {e}",
                "components": []
            }
    
    checker = WorkflowHealthChecker(config)
    return await checker.check_all()


def create_simple_health_check() -> Dict[str, Any]:
    """
    Create a simple health check that doesn't require async operations.
    
    Returns:
        Basic health status dictionary
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "message": "Basic health check passed",
        "version": "1.0.0"
    }
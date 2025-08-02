"""
Deployment scripts and automation for production deployment.

This module provides deployment automation, rollback capabilities,
and deployment strategy management.
"""

import asyncio
import logging
import json
import yaml
import subprocess
import shutil
import os
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
import tempfile

logger = logging.getLogger(__name__)


class DeploymentStrategy(Enum):
    """Deployment strategy types."""
    BLUE_GREEN = "blue_green"
    ROLLING = "rolling"
    CANARY = "canary"
    RECREATE = "recreate"


class DeploymentStatus(Enum):
    """Deployment status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class DeploymentConfig:
    """Configuration for a deployment."""
    deployment_id: str
    strategy: DeploymentStrategy
    environment: str
    version: str
    config_changes: Dict[str, Any]
    rollback_enabled: bool
    health_check_timeout: int
    deployment_timeout: int
    pre_deployment_hooks: List[str]
    post_deployment_hooks: List[str]
    rollback_hooks: List[str]


@dataclass
class DeploymentResult:
    """Result of a deployment operation."""
    deployment_id: str
    status: DeploymentStatus
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: Optional[float]
    version_deployed: str
    previous_version: Optional[str]
    error_message: Optional[str]
    logs: List[str]
    metrics: Dict[str, Any]


class DeploymentManager:
    """Manager for deployment operations and automation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the deployment manager.
        
        Args:
            config: Configuration for deployment management
        """
        self.config = config
        self.deployment_history: List[DeploymentResult] = []
        self.active_deployments: Dict[str, DeploymentResult] = {}
        
        # Configuration
        self.deployment_dir = Path(config.get('deployment_dir', 'deployments'))
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
        
        self.scripts_dir = Path(config.get('scripts_dir', 'deployment_scripts'))
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        
        self.backup_dir = Path(config.get('backup_dir', 'deployment_backups'))
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Default settings
        self.default_timeout = config.get('default_timeout_seconds', 600)
        self.health_check_timeout = config.get('health_check_timeout_seconds', 300)
        self.max_history_size = config.get('max_history_size', 100)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Callbacks
        self.deployment_callbacks: List[Callable[[DeploymentResult], None]] = []
        
        # Initialize deployment scripts
        self._initialize_deployment_scripts()
        
        logger.info("Deployment manager initialized")
    
    def _initialize_deployment_scripts(self):
        """Initialize default deployment scripts."""
        
        # Create default deployment scripts if they don't exist
        scripts = {
            'pre_deploy.sh': self._get_pre_deploy_script(),
            'post_deploy.sh': self._get_post_deploy_script(),
            'rollback.sh': self._get_rollback_script(),
            'health_check.sh': self._get_health_check_script(),
            'backup.sh': self._get_backup_script()
        }
        
        for script_name, script_content in scripts.items():
            script_path = self.scripts_dir / script_name
            if not script_path.exists():
                with open(script_path, 'w') as f:
                    f.write(script_content)
                script_path.chmod(0o755)  # Make executable
        
        logger.info(f"Deployment scripts initialized in {self.scripts_dir}")
    
    def _get_pre_deploy_script(self) -> str:
        """Get default pre-deployment script."""
        return '''#!/bin/bash
# Pre-deployment script
set -e

echo "Starting pre-deployment tasks..."

# Create backup
echo "Creating backup..."
./backup.sh

# Validate configuration
echo "Validating configuration..."
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Run tests
echo "Running tests..."
python -m pytest tests/ -v

echo "Pre-deployment tasks completed successfully"
'''
    
    def _get_post_deploy_script(self) -> str:
        """Get default post-deployment script."""
        return '''#!/bin/bash
# Post-deployment script
set -e

echo "Starting post-deployment tasks..."

# Wait for services to start
echo "Waiting for services to start..."
sleep 10

# Run health checks
echo "Running health checks..."
./health_check.sh

# Warm up caches
echo "Warming up caches..."
curl -f http://localhost:8000/health || exit 1

# Send deployment notification
echo "Sending deployment notification..."
# Add your notification logic here

echo "Post-deployment tasks completed successfully"
'''
    
    def _get_rollback_script(self) -> str:
        """Get default rollback script."""
        return '''#!/bin/bash
# Rollback script
set -e

BACKUP_DIR=${1:-"latest"}

echo "Starting rollback to backup: $BACKUP_DIR"

# Stop current services
echo "Stopping services..."
pkill -f "python.*agent" || true

# Restore from backup
echo "Restoring from backup..."
if [ -d "deployment_backups/$BACKUP_DIR" ]; then
    cp -r deployment_backups/$BACKUP_DIR/* .
else
    echo "Backup directory not found: $BACKUP_DIR"
    exit 1
fi

# Restart services
echo "Restarting services..."
python -m src.langgraph_agents.main &

# Verify rollback
echo "Verifying rollback..."
sleep 10
./health_check.sh

echo "Rollback completed successfully"
'''
    
    def _get_health_check_script(self) -> str:
        """Get default health check script."""
        return '''#!/bin/bash
# Health check script
set -e

echo "Running health checks..."

# Check if main process is running
if ! pgrep -f "python.*agent" > /dev/null; then
    echo "ERROR: Main agent process not running"
    exit 1
fi

# Check HTTP endpoint
if ! curl -f -s http://localhost:8000/health > /dev/null; then
    echo "ERROR: Health endpoint not responding"
    exit 1
fi

# Check memory usage
MEMORY_USAGE=$(ps aux | grep "python.*agent" | awk '{sum+=$4} END {print sum}')
if (( $(echo "$MEMORY_USAGE > 80" | bc -l) )); then
    echo "WARNING: High memory usage: ${MEMORY_USAGE}%"
fi

echo "Health checks passed"
'''
    
    def _get_backup_script(self) -> str:
        """Get default backup script."""
        return '''#!/bin/bash
# Backup script
set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="deployment_backups/backup_$TIMESTAMP"

echo "Creating backup: $BACKUP_DIR"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup configuration files
cp -r config/ "$BACKUP_DIR/" 2>/dev/null || true
cp -r src/ "$BACKUP_DIR/" 2>/dev/null || true
cp *.py "$BACKUP_DIR/" 2>/dev/null || true
cp requirements.txt "$BACKUP_DIR/" 2>/dev/null || true

# Backup database/state files
cp -r data/ "$BACKUP_DIR/" 2>/dev/null || true
cp -r logs/ "$BACKUP_DIR/" 2>/dev/null || true

# Create backup manifest
echo "Backup created: $(date)" > "$BACKUP_DIR/backup_manifest.txt"
echo "Version: $(git rev-parse HEAD 2>/dev/null || echo 'unknown')" >> "$BACKUP_DIR/backup_manifest.txt"

# Clean up old backups (keep last 10)
cd deployment_backups
ls -t | tail -n +11 | xargs rm -rf 2>/dev/null || true

echo "Backup completed: $BACKUP_DIR"
'''
    
    async def deploy(
        self,
        deployment_config: DeploymentConfig,
        dry_run: bool = False
    ) -> DeploymentResult:
        """Execute a deployment.
        
        Args:
            deployment_config: Deployment configuration
            dry_run: If True, simulate deployment without making changes
            
        Returns:
            Deployment result
        """
        
        deployment_result = DeploymentResult(
            deployment_id=deployment_config.deployment_id,
            status=DeploymentStatus.PENDING,
            start_time=datetime.now(),
            end_time=None,
            duration_seconds=None,
            version_deployed=deployment_config.version,
            previous_version=None,
            error_message=None,
            logs=[],
            metrics={}
        )
        
        with self._lock:
            self.active_deployments[deployment_config.deployment_id] = deployment_result
        
        try:
            deployment_result.status = DeploymentStatus.IN_PROGRESS
            deployment_result.logs.append(f"Starting deployment: {deployment_config.deployment_id}")
            
            if dry_run:
                deployment_result.logs.append("DRY RUN: Simulating deployment")
            
            # Execute deployment based on strategy
            if deployment_config.strategy == DeploymentStrategy.BLUE_GREEN:
                await self._execute_blue_green_deployment(deployment_config, deployment_result, dry_run)
            elif deployment_config.strategy == DeploymentStrategy.ROLLING:
                await self._execute_rolling_deployment(deployment_config, deployment_result, dry_run)
            elif deployment_config.strategy == DeploymentStrategy.CANARY:
                await self._execute_canary_deployment(deployment_config, deployment_result, dry_run)
            elif deployment_config.strategy == DeploymentStrategy.RECREATE:
                await self._execute_recreate_deployment(deployment_config, deployment_result, dry_run)
            else:
                raise ValueError(f"Unsupported deployment strategy: {deployment_config.strategy}")
            
            deployment_result.status = DeploymentStatus.COMPLETED
            deployment_result.logs.append("Deployment completed successfully")
            
        except Exception as e:
            deployment_result.status = DeploymentStatus.FAILED
            deployment_result.error_message = str(e)
            deployment_result.logs.append(f"Deployment failed: {str(e)}")
            
            logger.error(f"Deployment failed: {deployment_config.deployment_id}, error: {str(e)}")
            
            # Attempt rollback if enabled
            if deployment_config.rollback_enabled and not dry_run:
                try:
                    await self._execute_rollback(deployment_config, deployment_result)
                except Exception as rollback_error:
                    deployment_result.logs.append(f"Rollback failed: {str(rollback_error)}")
                    logger.error(f"Rollback failed: {str(rollback_error)}")
        
        finally:
            deployment_result.end_time = datetime.now()
            deployment_result.duration_seconds = (
                deployment_result.end_time - deployment_result.start_time
            ).total_seconds()
            
            with self._lock:
                self.active_deployments.pop(deployment_config.deployment_id, None)
                self.deployment_history.append(deployment_result)
                
                # Limit history size
                if len(self.deployment_history) > self.max_history_size:
                    self.deployment_history.pop(0)
            
            # Notify callbacks
            for callback in self.deployment_callbacks:
                try:
                    callback(deployment_result)
                except Exception as e:
                    logger.error(f"Error in deployment callback: {str(e)}")
        
        return deployment_result
    
    async def _execute_blue_green_deployment(
        self,
        config: DeploymentConfig,
        result: DeploymentResult,
        dry_run: bool
    ):
        """Execute blue-green deployment strategy."""
        
        result.logs.append("Executing blue-green deployment")
        
        # Run pre-deployment hooks
        await self._run_hooks(config.pre_deployment_hooks, "pre-deployment", result, dry_run)
        
        # Create green environment
        result.logs.append("Creating green environment")
        if not dry_run:
            await self._create_green_environment(config, result)
        
        # Deploy to green environment
        result.logs.append("Deploying to green environment")
        if not dry_run:
            await self._deploy_to_green(config, result)
        
        # Health check green environment
        result.logs.append("Health checking green environment")
        if not dry_run:
            await self._health_check_environment(config, result, "green")
        
        # Switch traffic to green
        result.logs.append("Switching traffic to green environment")
        if not dry_run:
            await self._switch_traffic_to_green(config, result)
        
        # Run post-deployment hooks
        await self._run_hooks(config.post_deployment_hooks, "post-deployment", result, dry_run)
        
        # Clean up blue environment
        result.logs.append("Cleaning up blue environment")
        if not dry_run:
            await self._cleanup_blue_environment(config, result)
    
    async def _execute_rolling_deployment(
        self,
        config: DeploymentConfig,
        result: DeploymentResult,
        dry_run: bool
    ):
        """Execute rolling deployment strategy."""
        
        result.logs.append("Executing rolling deployment")
        
        # Run pre-deployment hooks
        await self._run_hooks(config.pre_deployment_hooks, "pre-deployment", result, dry_run)
        
        # Get list of instances to update
        instances = await self._get_deployment_instances(config)
        result.logs.append(f"Found {len(instances)} instances to update")
        
        # Update instances one by one
        for i, instance in enumerate(instances):
            result.logs.append(f"Updating instance {i+1}/{len(instances)}: {instance}")
            
            if not dry_run:
                # Stop instance
                await self._stop_instance(instance, result)
                
                # Update instance
                await self._update_instance(instance, config, result)
                
                # Start instance
                await self._start_instance(instance, result)
                
                # Health check instance
                await self._health_check_instance(instance, result)
            
            # Wait between instances
            await asyncio.sleep(5)
        
        # Run post-deployment hooks
        await self._run_hooks(config.post_deployment_hooks, "post-deployment", result, dry_run)
    
    async def _execute_canary_deployment(
        self,
        config: DeploymentConfig,
        result: DeploymentResult,
        dry_run: bool
    ):
        """Execute canary deployment strategy."""
        
        result.logs.append("Executing canary deployment")
        
        # Run pre-deployment hooks
        await self._run_hooks(config.pre_deployment_hooks, "pre-deployment", result, dry_run)
        
        # Deploy canary instance
        result.logs.append("Deploying canary instance")
        if not dry_run:
            await self._deploy_canary_instance(config, result)
        
        # Route small percentage of traffic to canary
        result.logs.append("Routing 10% traffic to canary")
        if not dry_run:
            await self._route_traffic_to_canary(config, result, 0.1)
        
        # Monitor canary performance
        result.logs.append("Monitoring canary performance")
        if not dry_run:
            canary_healthy = await self._monitor_canary(config, result)
            
            if not canary_healthy:
                raise Exception("Canary deployment failed health checks")
        
        # Gradually increase traffic to canary
        for traffic_percent in [0.25, 0.5, 0.75, 1.0]:
            result.logs.append(f"Routing {traffic_percent*100}% traffic to canary")
            if not dry_run:
                await self._route_traffic_to_canary(config, result, traffic_percent)
                await asyncio.sleep(30)  # Wait between traffic increases
        
        # Run post-deployment hooks
        await self._run_hooks(config.post_deployment_hooks, "post-deployment", result, dry_run)
    
    async def _execute_recreate_deployment(
        self,
        config: DeploymentConfig,
        result: DeploymentResult,
        dry_run: bool
    ):
        """Execute recreate deployment strategy."""
        
        result.logs.append("Executing recreate deployment")
        
        # Run pre-deployment hooks
        await self._run_hooks(config.pre_deployment_hooks, "pre-deployment", result, dry_run)
        
        # Stop all instances
        result.logs.append("Stopping all instances")
        if not dry_run:
            await self._stop_all_instances(config, result)
        
        # Deploy new version
        result.logs.append("Deploying new version")
        if not dry_run:
            await self._deploy_new_version(config, result)
        
        # Start all instances
        result.logs.append("Starting all instances")
        if not dry_run:
            await self._start_all_instances(config, result)
        
        # Health check all instances
        result.logs.append("Health checking all instances")
        if not dry_run:
            await self._health_check_all_instances(config, result)
        
        # Run post-deployment hooks
        await self._run_hooks(config.post_deployment_hooks, "post-deployment", result, dry_run)
    
    async def _run_hooks(
        self,
        hooks: List[str],
        hook_type: str,
        result: DeploymentResult,
        dry_run: bool
    ):
        """Run deployment hooks."""
        
        for hook in hooks:
            result.logs.append(f"Running {hook_type} hook: {hook}")
            
            if not dry_run:
                try:
                    # Execute hook script
                    process = await asyncio.create_subprocess_shell(
                        hook,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=self.scripts_dir
                    )
                    
                    stdout, stderr = await process.communicate()
                    
                    if process.returncode == 0:
                        result.logs.append(f"Hook completed successfully: {hook}")
                        if stdout:
                            result.logs.append(f"Hook output: {stdout.decode()}")
                    else:
                        error_msg = f"Hook failed: {hook}, stderr: {stderr.decode()}"
                        result.logs.append(error_msg)
                        raise Exception(error_msg)
                
                except Exception as e:
                    result.logs.append(f"Hook execution error: {hook}, error: {str(e)}")
                    raise
    
    async def _execute_rollback(self, config: DeploymentConfig, result: DeploymentResult):
        """Execute rollback procedure."""
        
        result.logs.append("Starting rollback procedure")
        result.status = DeploymentStatus.ROLLED_BACK
        
        try:
            # Run rollback hooks
            await self._run_hooks(config.rollback_hooks, "rollback", result, False)
            
            # Execute rollback script
            rollback_script = self.scripts_dir / "rollback.sh"
            if rollback_script.exists():
                process = await asyncio.create_subprocess_shell(
                    str(rollback_script),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    result.logs.append("Rollback completed successfully")
                    if stdout:
                        result.logs.append(f"Rollback output: {stdout.decode()}")
                else:
                    error_msg = f"Rollback script failed, stderr: {stderr.decode()}"
                    result.logs.append(error_msg)
                    raise Exception(error_msg)
            
        except Exception as e:
            result.logs.append(f"Rollback failed: {str(e)}")
            raise
    
    # Placeholder methods for deployment operations
    # These would be implemented based on specific deployment infrastructure
    
    async def _create_green_environment(self, config: DeploymentConfig, result: DeploymentResult):
        """Create green environment for blue-green deployment."""
        await asyncio.sleep(1)  # Simulate work
    
    async def _deploy_to_green(self, config: DeploymentConfig, result: DeploymentResult):
        """Deploy application to green environment."""
        await asyncio.sleep(2)  # Simulate work
    
    async def _health_check_environment(self, config: DeploymentConfig, result: DeploymentResult, env: str):
        """Health check an environment."""
        await asyncio.sleep(1)  # Simulate work
    
    async def _switch_traffic_to_green(self, config: DeploymentConfig, result: DeploymentResult):
        """Switch traffic from blue to green environment."""
        await asyncio.sleep(1)  # Simulate work
    
    async def _cleanup_blue_environment(self, config: DeploymentConfig, result: DeploymentResult):
        """Clean up blue environment after successful deployment."""
        await asyncio.sleep(1)  # Simulate work
    
    async def _get_deployment_instances(self, config: DeploymentConfig) -> List[str]:
        """Get list of instances for deployment."""
        return ["instance-1", "instance-2", "instance-3"]  # Placeholder
    
    async def _stop_instance(self, instance: str, result: DeploymentResult):
        """Stop a deployment instance."""
        await asyncio.sleep(1)  # Simulate work
    
    async def _update_instance(self, instance: str, config: DeploymentConfig, result: DeploymentResult):
        """Update a deployment instance."""
        await asyncio.sleep(2)  # Simulate work
    
    async def _start_instance(self, instance: str, result: DeploymentResult):
        """Start a deployment instance."""
        await asyncio.sleep(1)  # Simulate work
    
    async def _health_check_instance(self, instance: str, result: DeploymentResult):
        """Health check a deployment instance."""
        await asyncio.sleep(1)  # Simulate work
    
    async def _deploy_canary_instance(self, config: DeploymentConfig, result: DeploymentResult):
        """Deploy canary instance."""
        await asyncio.sleep(2)  # Simulate work
    
    async def _route_traffic_to_canary(self, config: DeploymentConfig, result: DeploymentResult, percentage: float):
        """Route traffic percentage to canary."""
        await asyncio.sleep(1)  # Simulate work
    
    async def _monitor_canary(self, config: DeploymentConfig, result: DeploymentResult) -> bool:
        """Monitor canary instance health."""
        await asyncio.sleep(5)  # Simulate monitoring
        return True  # Placeholder - always healthy
    
    async def _stop_all_instances(self, config: DeploymentConfig, result: DeploymentResult):
        """Stop all deployment instances."""
        await asyncio.sleep(2)  # Simulate work
    
    async def _deploy_new_version(self, config: DeploymentConfig, result: DeploymentResult):
        """Deploy new version of application."""
        await asyncio.sleep(3)  # Simulate work
    
    async def _start_all_instances(self, config: DeploymentConfig, result: DeploymentResult):
        """Start all deployment instances."""
        await asyncio.sleep(2)  # Simulate work
    
    async def _health_check_all_instances(self, config: DeploymentConfig, result: DeploymentResult):
        """Health check all deployment instances."""
        await asyncio.sleep(2)  # Simulate work
    
    def get_deployment_history(self, limit: int = 50) -> List[DeploymentResult]:
        """Get deployment history.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of deployment results
        """
        
        with self._lock:
            return list(reversed(self.deployment_history[-limit:]))
    
    def get_active_deployments(self) -> List[DeploymentResult]:
        """Get currently active deployments.
        
        Returns:
            List of active deployment results
        """
        
        with self._lock:
            return list(self.active_deployments.values())
    
    def get_deployment_stats(self) -> Dict[str, Any]:
        """Get deployment statistics.
        
        Returns:
            Deployment statistics
        """
        
        with self._lock:
            total_deployments = len(self.deployment_history)
            successful_deployments = sum(
                1 for d in self.deployment_history 
                if d.status == DeploymentStatus.COMPLETED
            )
            failed_deployments = sum(
                1 for d in self.deployment_history 
                if d.status == DeploymentStatus.FAILED
            )
            rolled_back_deployments = sum(
                1 for d in self.deployment_history 
                if d.status == DeploymentStatus.ROLLED_BACK
            )
            
            # Calculate average deployment time
            completed_deployments = [
                d for d in self.deployment_history 
                if d.status == DeploymentStatus.COMPLETED and d.duration_seconds
            ]
            
            avg_deployment_time = (
                sum(d.duration_seconds for d in completed_deployments) / len(completed_deployments)
                if completed_deployments else 0
            )
            
            return {
                'total_deployments': total_deployments,
                'successful_deployments': successful_deployments,
                'failed_deployments': failed_deployments,
                'rolled_back_deployments': rolled_back_deployments,
                'success_rate': successful_deployments / max(total_deployments, 1),
                'active_deployments': len(self.active_deployments),
                'average_deployment_time_seconds': avg_deployment_time,
                'deployment_strategies_used': list(set(
                    d.metrics.get('strategy', 'unknown') for d in self.deployment_history
                ))
            }
    
    def add_deployment_callback(self, callback: Callable[[DeploymentResult], None]):
        """Add callback for deployment events.
        
        Args:
            callback: Function to call on deployment events
        """
        self.deployment_callbacks.append(callback)
    
    def create_deployment_config(
        self,
        deployment_id: str,
        strategy: DeploymentStrategy,
        environment: str,
        version: str,
        **kwargs
    ) -> DeploymentConfig:
        """Create a deployment configuration.
        
        Args:
            deployment_id: Unique deployment identifier
            strategy: Deployment strategy
            environment: Target environment
            version: Version to deploy
            **kwargs: Additional configuration options
            
        Returns:
            Deployment configuration
        """
        
        return DeploymentConfig(
            deployment_id=deployment_id,
            strategy=strategy,
            environment=environment,
            version=version,
            config_changes=kwargs.get('config_changes', {}),
            rollback_enabled=kwargs.get('rollback_enabled', True),
            health_check_timeout=kwargs.get('health_check_timeout', self.health_check_timeout),
            deployment_timeout=kwargs.get('deployment_timeout', self.default_timeout),
            pre_deployment_hooks=kwargs.get('pre_deployment_hooks', ['./pre_deploy.sh']),
            post_deployment_hooks=kwargs.get('post_deployment_hooks', ['./post_deploy.sh']),
            rollback_hooks=kwargs.get('rollback_hooks', ['./rollback.sh'])
        )
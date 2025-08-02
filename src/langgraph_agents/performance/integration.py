"""
Performance optimization integration for LangGraph multi-agent workflows.

This module provides a unified interface for integrating all performance
optimizations across the multi-agent system.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .performance_optimizer import PerformanceOptimizer, OptimizationConfig, OptimizationStrategy
from .planner_optimizer import PlannerOptimizer
from .concurrency_manager import ConcurrencyManager
from .memory_manager import MemoryManager
from .cache_manager import CacheManager
from .resource_pool import ResourcePoolManager

logger = logging.getLogger(__name__)


class PerformanceIntegration:
    """Unified performance optimization integration for multi-agent workflows."""
    
    def __init__(self, system_config: Dict[str, Any]):
        """Initialize performance integration.
        
        Args:
            system_config: System configuration
        """
        self.system_config = system_config
        
        # Performance components
        self.performance_optimizer: Optional[PerformanceOptimizer] = None
        self.planner_optimizer: Optional[PlannerOptimizer] = None
        self.concurrency_manager: Optional[ConcurrencyManager] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.resource_pool_manager: Optional[ResourcePoolManager] = None
        
        # Integration state
        self.is_initialized = False
        self.optimization_enabled = system_config.get('enable_performance_optimization', True)
        self.optimization_strategy = OptimizationStrategy(
            system_config.get('optimization_strategy', 'balanced')
        )
        
        logger.info(f"Performance integration initialized with strategy: {self.optimization_strategy.value}")
    
    async def initialize(self):
        """Initialize all performance optimization components."""
        
        if self.is_initialized or not self.optimization_enabled:
            return
        
        try:
            # Create optimization configuration
            optimization_config = self._create_optimization_config()
            
            # Initialize main performance optimizer
            self.performance_optimizer = PerformanceOptimizer(optimization_config)
            await self.performance_optimizer.start()
            
            # Initialize specialized optimizers
            await self._initialize_specialized_optimizers()
            
            # Initialize core performance components
            await self._initialize_core_components()
            
            self.is_initialized = True
            logger.info("Performance integration initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize performance integration: {str(e)}")
            self.optimization_enabled = False
            raise
    
    async def shutdown(self):
        """Shutdown all performance optimization components."""
        
        if not self.is_initialized:
            return
        
        try:
            # Shutdown components in reverse order
            if self.performance_optimizer:
                await self.performance_optimizer.stop()
            
            if self.planner_optimizer:
                await self.planner_optimizer.stop()
            
            if self.memory_manager:
                await self.memory_manager.stop()
            
            if self.resource_pool_manager:
                await self.resource_pool_manager.shutdown()
            
            if self.concurrency_manager:
                await self.concurrency_manager.shutdown()
            
            self.is_initialized = False
            logger.info("Performance integration shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during performance integration shutdown: {str(e)}")
    
    def _create_optimization_config(self) -> OptimizationConfig:
        """Create optimization configuration from system config.
        
        Returns:
            OptimizationConfig: Optimization configuration
        """
        
        return OptimizationConfig(
            strategy=self.optimization_strategy,
            max_concurrent_tasks=self.system_config.get('max_concurrent_tasks', 10),
            max_concurrent_per_agent=self.system_config.get('max_concurrent_per_agent', 3),
            max_scene_concurrency=self.system_config.get('max_scene_concurrency', 8),
            enable_adaptive_scaling=self.system_config.get('enable_adaptive_scaling', True),
            enable_aggressive_caching=self.system_config.get('enable_aggressive_caching', True),
            max_cache_memory_mb=self.system_config.get('max_cache_memory_mb', 512),
            cache_ttl_seconds=self.system_config.get('cache_ttl_seconds', 3600),
            memory_warning_threshold=self.system_config.get('memory_warning_threshold', 80.0),
            memory_critical_threshold=self.system_config.get('memory_critical_threshold', 90.0),
            enable_memory_optimization=self.system_config.get('enable_memory_optimization', True),
            enable_connection_pooling=self.system_config.get('enable_connection_pooling', True),
            max_connections_per_pool=self.system_config.get('max_connections_per_pool', 20),
            connection_idle_timeout=self.system_config.get('connection_idle_timeout', 300),
            enable_planner_optimization=self.system_config.get('enable_planner_optimization', True),
            planner_parallel_scenes=self.system_config.get('planner_parallel_scenes', True),
            planner_batch_size=self.system_config.get('planner_batch_size', 5),
            planner_cache_implementations=self.system_config.get('planner_cache_implementations', True)
        )
    
    async def _initialize_specialized_optimizers(self):
        """Initialize specialized optimizers for specific agents."""
        
        # Initialize planner optimizer
        if self.system_config.get('enable_planner_optimization', True):
            planner_config = {
                'max_parallel_scenes': self.system_config.get('max_scene_concurrency', 8),
                'max_concurrent_implementations': 6,
                'rag_batch_size': 10,
                'enable_scene_caching': True,
                'enable_implementation_caching': True,
                'enable_plugin_caching': True,
                'max_worker_threads': 4
            }
            
            self.planner_optimizer = PlannerOptimizer(planner_config)
            await self.planner_optimizer.start()
            
            logger.info("Planner optimizer initialized")
    
    async def _initialize_core_components(self):
        """Initialize core performance components."""
        
        # Initialize concurrency manager
        concurrency_config = {
            'max_concurrent_tasks': self.system_config.get('max_concurrent_tasks', 10),
            'max_concurrent_per_agent': self.system_config.get('max_concurrent_per_agent', 3),
            'max_concurrent_per_scene': 2,
            'max_memory_usage_mb': 4096,
            'max_cpu_usage_percent': 80.0,
            'adaptive_scaling': self.system_config.get('enable_adaptive_scaling', True)
        }
        
        self.concurrency_manager = ConcurrencyManager(concurrency_config)
        
        # Initialize memory manager
        memory_config = {
            'gc_strategy': 'adaptive',
            'warning_percent': self.system_config.get('memory_warning_threshold', 80.0),
            'critical_percent': self.system_config.get('memory_critical_threshold', 90.0),
            'enable_tracemalloc': True,
            'enable_leak_detection': True
        }
        
        self.memory_manager = MemoryManager(memory_config)
        await self.memory_manager.start()
        
        # Initialize resource pool manager
        resource_config = {
            'default_pool_config': {
                'min_size': 2,
                'max_size': self.system_config.get('max_connections_per_pool', 20),
                'max_idle_time': self.system_config.get('connection_idle_timeout', 300),
                'health_check_interval': 60
            }
        }
        
        self.resource_pool_manager = ResourcePoolManager(resource_config)
        
        logger.info("Core performance components initialized")
    
    async def optimize_agent_performance(self, agent_name: str, agent_instance) -> Dict[str, Any]:
        """Optimize performance for a specific agent.
        
        Args:
            agent_name: Name of the agent
            agent_instance: Agent instance
            
        Returns:
            Dict: Optimization results
        """
        
        if not self.optimization_enabled or not self.performance_optimizer:
            return {'success': False, 'reason': 'Optimization not enabled'}
        
        try:
            if agent_name == 'planner_agent':
                return await self.performance_optimizer.optimize_planner_performance(agent_instance)
            elif agent_name == 'code_generator_agent':
                return await self.performance_optimizer.optimize_code_generation_performance()
            elif agent_name == 'renderer_agent':
                return await self.performance_optimizer.optimize_rendering_performance()
            else:
                logger.warning(f"No specific optimization available for agent: {agent_name}")
                return {'success': False, 'reason': f'No optimization for {agent_name}'}
                
        except Exception as e:
            logger.error(f"Failed to optimize {agent_name}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def optimize_workflow_performance(self, workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize overall workflow performance.
        
        Args:
            workflow_config: Workflow configuration
            
        Returns:
            Dict: Optimization results
        """
        
        if not self.optimization_enabled or not self.performance_optimizer:
            return {'success': False, 'reason': 'Optimization not enabled'}
        
        optimization_results = {
            'timestamp': datetime.now().isoformat(),
            'optimizations_applied': [],
            'total_improvement_percent': 0.0,
            'component_results': {}
        }
        
        try:
            # Optimize planner performance
            if workflow_config.get('optimize_planner', True):
                planner_result = await self.performance_optimizer.optimize_planner_performance(None)
                optimization_results['component_results']['planner'] = planner_result
                if planner_result.get('success'):
                    optimization_results['optimizations_applied'].extend(
                        planner_result.get('optimizations_applied', [])
                    )
            
            # Optimize code generation performance
            if workflow_config.get('optimize_code_generation', True):
                code_gen_result = await self.performance_optimizer.optimize_code_generation_performance()
                optimization_results['component_results']['code_generation'] = code_gen_result
                if code_gen_result.get('success'):
                    optimization_results['optimizations_applied'].extend(
                        code_gen_result.get('optimizations_applied', [])
                    )
            
            # Optimize rendering performance
            if workflow_config.get('optimize_rendering', True):
                rendering_result = await self.performance_optimizer.optimize_rendering_performance()
                optimization_results['component_results']['rendering'] = rendering_result
                if rendering_result.get('success'):
                    optimization_results['optimizations_applied'].extend(
                        rendering_result.get('optimizations_applied', [])
                    )
            
            # Calculate total improvement
            improvements = [
                result.get('expected_improvement_percent', 0)
                for result in optimization_results['component_results'].values()
                if result.get('success')
            ]
            
            if improvements:
                # Use weighted average (planner has highest impact)
                weights = {'planner': 0.5, 'code_generation': 0.3, 'rendering': 0.2}
                total_improvement = 0.0
                total_weight = 0.0
                
                for component, result in optimization_results['component_results'].items():
                    if result.get('success'):
                        weight = weights.get(component, 0.1)
                        improvement = result.get('expected_improvement_percent', 0)
                        total_improvement += improvement * weight
                        total_weight += weight
                
                optimization_results['total_improvement_percent'] = (
                    total_improvement / total_weight if total_weight > 0 else 0.0
                )
            
            optimization_results['success'] = len(optimization_results['optimizations_applied']) > 0
            
            logger.info(f"Workflow optimization completed: "
                       f"{len(optimization_results['optimizations_applied'])} optimizations applied, "
                       f"{optimization_results['total_improvement_percent']:.1f}% expected improvement")
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Workflow optimization failed: {str(e)}")
            optimization_results['success'] = False
            optimization_results['error'] = str(e)
            return optimization_results
    
    def get_performance_status(self) -> Dict[str, Any]:
        """Get comprehensive performance status.
        
        Returns:
            Dict: Performance status information
        """
        
        status = {
            'integration_status': {
                'initialized': self.is_initialized,
                'optimization_enabled': self.optimization_enabled,
                'strategy': self.optimization_strategy.value
            },
            'component_status': {
                'performance_optimizer': self.performance_optimizer is not None,
                'planner_optimizer': self.planner_optimizer is not None,
                'concurrency_manager': self.concurrency_manager is not None,
                'memory_manager': self.memory_manager is not None,
                'resource_pool_manager': self.resource_pool_manager is not None
            }
        }
        
        # Add component-specific status
        if self.performance_optimizer:
            status['performance_optimizer_status'] = self.performance_optimizer.get_optimization_status()
        
        if self.planner_optimizer:
            status['planner_optimizer_metrics'] = self.planner_optimizer.get_optimization_metrics()
        
        if self.concurrency_manager:
            status['concurrency_metrics'] = self.concurrency_manager.get_performance_metrics()
        
        if self.memory_manager:
            status['memory_stats'] = self.memory_manager.get_memory_stats()
        
        if self.resource_pool_manager:
            status['resource_pool_stats'] = self.resource_pool_manager.get_all_stats()
        
        return status
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report.
        
        Returns:
            Dict: Performance report
        """
        
        if not self.is_initialized:
            return {'error': 'Performance integration not initialized'}
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'integration_info': {
                'strategy': self.optimization_strategy.value,
                'components_active': sum(1 for component in [
                    self.performance_optimizer,
                    self.planner_optimizer,
                    self.concurrency_manager,
                    self.memory_manager,
                    self.resource_pool_manager
                ] if component is not None)
            }
        }
        
        # Add detailed component reports
        if self.performance_optimizer:
            report['performance_optimizer'] = self.performance_optimizer.get_performance_report()
        
        if self.planner_optimizer:
            report['planner_optimizer'] = self.planner_optimizer.get_optimization_metrics()
        
        if self.concurrency_manager:
            report['concurrency_manager'] = self.concurrency_manager.get_performance_metrics()
        
        if self.memory_manager:
            report['memory_manager'] = self.memory_manager.get_memory_stats()
        
        if self.resource_pool_manager:
            report['resource_pools'] = self.resource_pool_manager.get_summary()
        
        return report
    
    async def clear_all_caches(self):
        """Clear all performance-related caches."""
        
        if self.performance_optimizer:
            # Clear cache managers
            for cache_manager in self.performance_optimizer.cache_managers.values():
                await cache_manager.clear()
        
        if self.planner_optimizer:
            self.planner_optimizer.clear_caches()
        
        logger.info("All performance caches cleared")
    
    async def force_memory_cleanup(self):
        """Force memory cleanup across all components."""
        
        if self.memory_manager:
            await self.memory_manager.force_cleanup()
        
        if self.performance_optimizer:
            await self.performance_optimizer._optimize_memory_usage()
        
        logger.info("Forced memory cleanup completed")
    
    async def emergency_performance_recovery(self):
        """Perform emergency performance recovery actions."""
        
        logger.warning("Initiating emergency performance recovery")
        
        try:
            # Clear all caches
            await self.clear_all_caches()
            
            # Force memory cleanup
            await self.force_memory_cleanup()
            
            # Reduce concurrency limits
            if self.concurrency_manager:
                self.concurrency_manager.limits.max_concurrent_tasks = max(
                    self.concurrency_manager.limits.max_concurrent_tasks - 3,
                    2
                )
                self.concurrency_manager.limits.max_concurrent_per_agent = max(
                    self.concurrency_manager.limits.max_concurrent_per_agent - 1,
                    1
                )
            
            # Trigger emergency optimizations
            if self.performance_optimizer:
                await self.performance_optimizer._trigger_emergency_optimizations()
            
            logger.info("Emergency performance recovery completed")
            
        except Exception as e:
            logger.error(f"Emergency performance recovery failed: {str(e)}")
            raise


# Global performance integration instance
_performance_integration: Optional[PerformanceIntegration] = None


def get_performance_integration() -> Optional[PerformanceIntegration]:
    """Get the global performance integration instance.
    
    Returns:
        Optional[PerformanceIntegration]: Performance integration instance
    """
    return _performance_integration


async def initialize_performance_integration(system_config: Dict[str, Any]) -> PerformanceIntegration:
    """Initialize the global performance integration.
    
    Args:
        system_config: System configuration
        
    Returns:
        PerformanceIntegration: Initialized performance integration
    """
    global _performance_integration
    
    if _performance_integration is None:
        _performance_integration = PerformanceIntegration(system_config)
        await _performance_integration.initialize()
    
    return _performance_integration


async def shutdown_performance_integration():
    """Shutdown the global performance integration."""
    global _performance_integration
    
    if _performance_integration:
        await _performance_integration.shutdown()
        _performance_integration = None
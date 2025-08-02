"""
Memory management and garbage collection optimization.

This module provides intelligent memory management, garbage collection tuning,
and memory leak detection for optimal system performance.
"""

import asyncio
import logging
import gc
import psutil
import threading
import weakref
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import tracemalloc
import sys
import os
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class GarbageCollectionStrategy(Enum):
    """Garbage collection strategies."""
    CONSERVATIVE = "conservative"  # Minimal GC intervention
    BALANCED = "balanced"  # Balanced approach
    AGGRESSIVE = "aggressive"  # Frequent GC for low memory usage
    ADAPTIVE = "adaptive"  # Adaptive based on memory pressure


@dataclass
class MemoryMetrics:
    """Memory usage metrics."""
    timestamp: datetime
    total_memory_mb: float
    available_memory_mb: float
    used_memory_mb: float
    memory_percent: float
    process_memory_mb: float
    gc_collections: Dict[int, int]  # Generation -> count
    gc_objects: int
    memory_growth_rate_mb_per_hour: float
    largest_objects: List[Dict[str, Any]]


@dataclass
class MemoryThreshold:
    """Memory threshold configuration."""
    warning_percent: float = 80.0
    critical_percent: float = 90.0
    emergency_percent: float = 95.0
    process_limit_mb: Optional[float] = None


class MemoryManager:
    """Advanced memory manager with garbage collection optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the memory manager.
        
        Args:
            config: Configuration for memory management
        """
        self.config = config
        self.strategy = GarbageCollectionStrategy(config.get('gc_strategy', 'balanced'))
        self.thresholds = MemoryThreshold(
            warning_percent=config.get('warning_percent', 80.0),
            critical_percent=config.get('critical_percent', 90.0),
            emergency_percent=config.get('emergency_percent', 95.0),
            process_limit_mb=config.get('process_limit_mb')
        )
        
        # Monitoring settings
        self.monitoring_interval = config.get('monitoring_interval_seconds', 30)
        self.enable_tracemalloc = config.get('enable_tracemalloc', True)
        self.enable_leak_detection = config.get('enable_leak_detection', True)
        self.metrics_history_size = config.get('metrics_history_size', 1000)
        
        # Memory tracking
        self.metrics_history: deque = deque(maxlen=self.metrics_history_size)
        self.object_trackers: Dict[str, weakref.WeakSet] = defaultdict(weakref.WeakSet)
        self.memory_alerts: List[Dict[str, Any]] = []
        
        # GC tuning
        self.gc_thresholds = config.get('gc_thresholds', (700, 10, 10))
        self.gc_enabled = config.get('gc_enabled', True)
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._gc_task: Optional[asyncio.Task] = None
        self._leak_detection_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Callbacks
        self.memory_warning_callbacks: List[Callable[[MemoryMetrics], None]] = []
        self.memory_critical_callbacks: List[Callable[[MemoryMetrics], None]] = []
        
        logger.info(f"Memory manager initialized with strategy: {self.strategy.value}")
    
    async def start(self):
        """Start the memory manager and background tasks."""
        
        with self._lock:
            if self._monitoring_task is not None:
                return  # Already started
            
            # Configure garbage collection
            if self.gc_enabled:
                gc.set_threshold(*self.gc_thresholds)
                gc.enable()
            
            # Enable tracemalloc if requested
            if self.enable_tracemalloc and not tracemalloc.is_tracing():
                tracemalloc.start(25)  # Keep 25 frames
            
            # Start background tasks
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._gc_task = asyncio.create_task(self._gc_loop())
            
            if self.enable_leak_detection:
                self._leak_detection_task = asyncio.create_task(self._leak_detection_loop())
            
            logger.info("Memory manager started")
    
    async def stop(self):
        """Stop the memory manager."""
        
        with self._lock:
            self._shutdown = True
            
            # Cancel background tasks
            for task in [self._monitoring_task, self._gc_task, self._leak_detection_task]:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Stop tracemalloc
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            
            logger.info("Memory manager stopped")
    
    async def get_current_metrics(self) -> MemoryMetrics:
        """Get current memory metrics."""
        
        # System memory
        memory_info = psutil.virtual_memory()
        total_memory_mb = memory_info.total / (1024 * 1024)
        available_memory_mb = memory_info.available / (1024 * 1024)
        used_memory_mb = (memory_info.total - memory_info.available) / (1024 * 1024)
        memory_percent = memory_info.percent
        
        # Process memory
        process = psutil.Process()
        process_memory_mb = process.memory_info().rss / (1024 * 1024)
        
        # Garbage collection stats
        gc_stats = gc.get_stats()
        gc_collections = {i: stats['collections'] for i, stats in enumerate(gc_stats)}
        gc_objects = len(gc.get_objects())
        
        # Memory growth rate
        growth_rate = self._calculate_memory_growth_rate()
        
        # Largest objects
        largest_objects = self._get_largest_objects()
        
        metrics = MemoryMetrics(
            timestamp=datetime.now(),
            total_memory_mb=total_memory_mb,
            available_memory_mb=available_memory_mb,
            used_memory_mb=used_memory_mb,
            memory_percent=memory_percent,
            process_memory_mb=process_memory_mb,
            gc_collections=gc_collections,
            gc_objects=gc_objects,
            memory_growth_rate_mb_per_hour=growth_rate,
            largest_objects=largest_objects
        )
        
        return metrics
    
    def _calculate_memory_growth_rate(self) -> float:
        """Calculate memory growth rate in MB per hour."""
        
        if len(self.metrics_history) < 2:
            return 0.0
        
        # Use last hour of data or all available data
        cutoff_time = datetime.now() - timedelta(hours=1)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        if len(recent_metrics) < 2:
            recent_metrics = list(self.metrics_history)
        
        if len(recent_metrics) < 2:
            return 0.0
        
        # Calculate growth rate
        first_metric = recent_metrics[0]
        last_metric = recent_metrics[-1]
        
        memory_diff = last_metric.process_memory_mb - first_metric.process_memory_mb
        time_diff_hours = (last_metric.timestamp - first_metric.timestamp).total_seconds() / 3600
        
        if time_diff_hours > 0:
            return memory_diff / time_diff_hours
        
        return 0.0
    
    def _get_largest_objects(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get information about the largest objects in memory."""
        
        if not tracemalloc.is_tracing():
            return []
        
        try:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            largest_objects = []
            for stat in top_stats[:limit]:
                largest_objects.append({
                    'size_mb': stat.size / (1024 * 1024),
                    'count': stat.count,
                    'filename': stat.traceback.format()[0] if stat.traceback else 'unknown',
                    'line': stat.traceback[0].lineno if stat.traceback else 0
                })
            
            return largest_objects
            
        except Exception as e:
            logger.error(f"Error getting largest objects: {str(e)}")
            return []
    
    async def _monitoring_loop(self):
        """Background task for memory monitoring."""
        
        try:
            while not self._shutdown:
                await asyncio.sleep(self.monitoring_interval)
                
                if self._shutdown:
                    break
                
                # Get current metrics
                metrics = await self.get_current_metrics()
                
                with self._lock:
                    self.metrics_history.append(metrics)
                
                # Check thresholds and trigger alerts
                await self._check_memory_thresholds(metrics)
                
                # Log metrics periodically
                if len(self.metrics_history) % 10 == 0:  # Every 10 intervals
                    logger.debug(f"Memory usage: {metrics.memory_percent:.1f}%, "
                               f"Process: {metrics.process_memory_mb:.1f}MB, "
                               f"Growth rate: {metrics.memory_growth_rate_mb_per_hour:.2f}MB/h")
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Memory monitoring loop error: {str(e)}")
    
    async def _check_memory_thresholds(self, metrics: MemoryMetrics):
        """Check memory thresholds and trigger appropriate actions."""
        
        memory_percent = metrics.memory_percent
        process_memory_mb = metrics.process_memory_mb
        
        # Check system memory thresholds
        if memory_percent >= self.thresholds.emergency_percent:
            await self._handle_emergency_memory(metrics)
        elif memory_percent >= self.thresholds.critical_percent:
            await self._handle_critical_memory(metrics)
        elif memory_percent >= self.thresholds.warning_percent:
            await self._handle_warning_memory(metrics)
        
        # Check process memory limit
        if (self.thresholds.process_limit_mb and 
            process_memory_mb > self.thresholds.process_limit_mb):
            await self._handle_process_memory_limit(metrics)
    
    async def _handle_warning_memory(self, metrics: MemoryMetrics):
        """Handle memory warning threshold."""
        
        alert = {
            'level': 'warning',
            'timestamp': metrics.timestamp,
            'memory_percent': metrics.memory_percent,
            'process_memory_mb': metrics.process_memory_mb,
            'message': f"Memory usage warning: {metrics.memory_percent:.1f}%"
        }
        
        self.memory_alerts.append(alert)
        logger.warning(alert['message'])
        
        # Trigger callbacks
        for callback in self.memory_warning_callbacks:
            try:
                callback(metrics)
            except Exception as e:
                logger.error(f"Memory warning callback error: {str(e)}")
        
        # Trigger light garbage collection
        await self._trigger_gc(force=False)
    
    async def _handle_critical_memory(self, metrics: MemoryMetrics):
        """Handle memory critical threshold."""
        
        alert = {
            'level': 'critical',
            'timestamp': metrics.timestamp,
            'memory_percent': metrics.memory_percent,
            'process_memory_mb': metrics.process_memory_mb,
            'message': f"Memory usage critical: {metrics.memory_percent:.1f}%"
        }
        
        self.memory_alerts.append(alert)
        logger.error(alert['message'])
        
        # Trigger callbacks
        for callback in self.memory_critical_callbacks:
            try:
                callback(metrics)
            except Exception as e:
                logger.error(f"Memory critical callback error: {str(e)}")
        
        # Trigger aggressive garbage collection
        await self._trigger_gc(force=True)
        
        # Additional cleanup actions
        await self._emergency_cleanup()
    
    async def _handle_emergency_memory(self, metrics: MemoryMetrics):
        """Handle memory emergency threshold."""
        
        alert = {
            'level': 'emergency',
            'timestamp': metrics.timestamp,
            'memory_percent': metrics.memory_percent,
            'process_memory_mb': metrics.process_memory_mb,
            'message': f"Memory usage emergency: {metrics.memory_percent:.1f}%"
        }
        
        self.memory_alerts.append(alert)
        logger.critical(alert['message'])
        
        # Emergency actions
        await self._emergency_cleanup()
        await self._trigger_gc(force=True)
        
        # Consider more drastic measures
        logger.critical("System is in emergency memory state - consider restarting")
    
    async def _handle_process_memory_limit(self, metrics: MemoryMetrics):
        """Handle process memory limit exceeded."""
        
        alert = {
            'level': 'process_limit',
            'timestamp': metrics.timestamp,
            'process_memory_mb': metrics.process_memory_mb,
            'limit_mb': self.thresholds.process_limit_mb,
            'message': f"Process memory limit exceeded: {metrics.process_memory_mb:.1f}MB > {self.thresholds.process_limit_mb}MB"
        }
        
        self.memory_alerts.append(alert)
        logger.error(alert['message'])
        
        # Aggressive cleanup
        await self._emergency_cleanup()
        await self._trigger_gc(force=True)
    
    async def _gc_loop(self):
        """Background task for garbage collection management."""
        
        try:
            while not self._shutdown:
                # Adjust GC interval based on strategy
                if self.strategy == GarbageCollectionStrategy.CONSERVATIVE:
                    interval = 300  # 5 minutes
                elif self.strategy == GarbageCollectionStrategy.BALANCED:
                    interval = 120  # 2 minutes
                elif self.strategy == GarbageCollectionStrategy.AGGRESSIVE:
                    interval = 60   # 1 minute
                else:  # ADAPTIVE
                    interval = await self._calculate_adaptive_gc_interval()
                
                await asyncio.sleep(interval)
                
                if self._shutdown:
                    break
                
                # Perform garbage collection based on strategy
                await self._perform_strategic_gc()
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"GC loop error: {str(e)}")
    
    async def _calculate_adaptive_gc_interval(self) -> int:
        """Calculate adaptive GC interval based on memory pressure."""
        
        if not self.metrics_history:
            return 120  # Default 2 minutes
        
        latest_metrics = self.metrics_history[-1]
        memory_percent = latest_metrics.memory_percent
        growth_rate = latest_metrics.memory_growth_rate_mb_per_hour
        
        # Base interval
        base_interval = 120
        
        # Adjust based on memory pressure
        if memory_percent > 85:
            interval = 30  # Very frequent
        elif memory_percent > 75:
            interval = 60  # Frequent
        elif memory_percent > 60:
            interval = base_interval  # Normal
        else:
            interval = 300  # Less frequent
        
        # Adjust based on growth rate
        if growth_rate > 100:  # Growing fast
            interval = min(interval, 60)
        elif growth_rate < 10:  # Growing slowly
            interval = max(interval, 180)
        
        return interval
    
    async def _perform_strategic_gc(self):
        """Perform garbage collection based on current strategy."""
        
        if not self.gc_enabled:
            return
        
        if self.strategy == GarbageCollectionStrategy.CONSERVATIVE:
            # Only collect generation 0
            collected = gc.collect(0)
        
        elif self.strategy == GarbageCollectionStrategy.BALANCED:
            # Collect all generations but less frequently for higher generations
            collected = gc.collect()
        
        elif self.strategy == GarbageCollectionStrategy.AGGRESSIVE:
            # Frequent collection of all generations
            collected = gc.collect()
            # Also force collection of higher generations
            gc.collect(1)
            gc.collect(2)
        
        else:  # ADAPTIVE
            # Adaptive collection based on memory state
            if not self.metrics_history:
                collected = gc.collect()
            else:
                latest_metrics = self.metrics_history[-1]
                if latest_metrics.memory_percent > 80:
                    collected = gc.collect()
                    gc.collect(1)
                    gc.collect(2)
                elif latest_metrics.memory_percent > 60:
                    collected = gc.collect()
                else:
                    collected = gc.collect(0)
        
        if collected > 0:
            logger.debug(f"Garbage collection freed {collected} objects")
    
    async def _trigger_gc(self, force: bool = False):
        """Trigger immediate garbage collection."""
        
        if not self.gc_enabled and not force:
            return
        
        try:
            # Collect all generations
            collected = gc.collect()
            
            if force:
                # Force collection of higher generations
                gc.collect(1)
                gc.collect(2)
            
            logger.debug(f"Manual GC freed {collected} objects")
            
        except Exception as e:
            logger.error(f"Error during manual GC: {str(e)}")
    
    async def _emergency_cleanup(self):
        """Perform emergency cleanup to free memory."""
        
        try:
            # Clear internal caches
            self._clear_internal_caches()
            
            # Trigger full garbage collection
            for generation in range(3):
                gc.collect(generation)
            
            # Clear weak references
            self._cleanup_weak_references()
            
            logger.info("Emergency cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during emergency cleanup: {str(e)}")
    
    def _clear_internal_caches(self):
        """Clear internal caches to free memory."""
        
        # Clear metrics history (keep only recent entries)
        if len(self.metrics_history) > 100:
            recent_metrics = list(self.metrics_history)[-50:]
            self.metrics_history.clear()
            self.metrics_history.extend(recent_metrics)
        
        # Clear old alerts
        if len(self.memory_alerts) > 100:
            self.memory_alerts = self.memory_alerts[-50:]
        
        # Clear object trackers
        for tracker_name in list(self.object_trackers.keys()):
            # Keep only live references
            live_refs = [ref for ref in self.object_trackers[tracker_name] if ref() is not None]
            self.object_trackers[tracker_name] = weakref.WeakSet(live_refs)
    
    def _cleanup_weak_references(self):
        """Clean up dead weak references."""
        
        for tracker_name, weak_set in self.object_trackers.items():
            # WeakSet automatically removes dead references, but we can force cleanup
            try:
                # Access all references to trigger cleanup
                list(weak_set)
            except Exception:
                pass
    
    async def _leak_detection_loop(self):
        """Background task for memory leak detection."""
        
        try:
            while not self._shutdown:
                await asyncio.sleep(600)  # Check every 10 minutes
                
                if self._shutdown:
                    break
                
                await self._detect_memory_leaks()
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Leak detection loop error: {str(e)}")
    
    async def _detect_memory_leaks(self):
        """Detect potential memory leaks."""
        
        if not tracemalloc.is_tracing() or len(self.metrics_history) < 10:
            return
        
        try:
            # Check for consistent memory growth
            recent_metrics = list(self.metrics_history)[-10:]
            growth_rates = []
            
            for i in range(1, len(recent_metrics)):
                prev_metric = recent_metrics[i-1]
                curr_metric = recent_metrics[i]
                
                time_diff = (curr_metric.timestamp - prev_metric.timestamp).total_seconds() / 3600
                memory_diff = curr_metric.process_memory_mb - prev_metric.process_memory_mb
                
                if time_diff > 0:
                    growth_rates.append(memory_diff / time_diff)
            
            if growth_rates:
                avg_growth_rate = sum(growth_rates) / len(growth_rates)
                
                # Check for consistent positive growth
                if avg_growth_rate > 10:  # Growing more than 10MB/hour consistently
                    positive_growth_count = sum(1 for rate in growth_rates if rate > 5)
                    
                    if positive_growth_count >= len(growth_rates) * 0.8:  # 80% of measurements show growth
                        logger.warning(f"Potential memory leak detected: "
                                     f"Average growth rate: {avg_growth_rate:.2f} MB/hour")
                        
                        # Get detailed information about memory usage
                        await self._analyze_memory_usage()
        
        except Exception as e:
            logger.error(f"Error in leak detection: {str(e)}")
    
    async def _analyze_memory_usage(self):
        """Analyze detailed memory usage for leak detection."""
        
        if not tracemalloc.is_tracing():
            return
        
        try:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            logger.info("Top memory consumers:")
            for i, stat in enumerate(top_stats[:5]):
                logger.info(f"  {i+1}. {stat.size / (1024*1024):.1f} MB - "
                          f"{stat.count} objects - {stat.traceback.format()[0] if stat.traceback else 'unknown'}")
            
            # Check for object count growth
            current_objects = len(gc.get_objects())
            if self.metrics_history:
                prev_objects = self.metrics_history[-1].gc_objects
                object_growth = current_objects - prev_objects
                
                if object_growth > 1000:
                    logger.warning(f"High object count growth: {object_growth} new objects")
        
        except Exception as e:
            logger.error(f"Error analyzing memory usage: {str(e)}")
    
    def track_object(self, obj: Any, category: str = "default"):
        """Track an object for memory monitoring.
        
        Args:
            obj: Object to track
            category: Category for grouping objects
        """
        
        try:
            self.object_trackers[category].add(obj)
        except Exception as e:
            logger.error(f"Error tracking object: {str(e)}")
    
    def get_tracked_object_count(self, category: str = "default") -> int:
        """Get count of tracked objects in a category.
        
        Args:
            category: Category to count
            
        Returns:
            Number of live tracked objects
        """
        
        return len(self.object_trackers[category])
    
    def add_memory_warning_callback(self, callback: Callable[[MemoryMetrics], None]):
        """Add callback for memory warning events.
        
        Args:
            callback: Function to call on memory warning
        """
        self.memory_warning_callbacks.append(callback)
    
    def add_memory_critical_callback(self, callback: Callable[[MemoryMetrics], None]):
        """Add callback for memory critical events.
        
        Args:
            callback: Function to call on memory critical
        """
        self.memory_critical_callbacks.append(callback)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        
        if not self.metrics_history:
            return {'error': 'No metrics available'}
        
        latest_metrics = self.metrics_history[-1]
        
        # Calculate statistics over history
        memory_percentages = [m.memory_percent for m in self.metrics_history]
        process_memories = [m.process_memory_mb for m in self.metrics_history]
        
        return {
            'current': {
                'memory_percent': latest_metrics.memory_percent,
                'process_memory_mb': latest_metrics.process_memory_mb,
                'available_memory_mb': latest_metrics.available_memory_mb,
                'gc_objects': latest_metrics.gc_objects,
                'growth_rate_mb_per_hour': latest_metrics.memory_growth_rate_mb_per_hour
            },
            'statistics': {
                'avg_memory_percent': sum(memory_percentages) / len(memory_percentages),
                'max_memory_percent': max(memory_percentages),
                'avg_process_memory_mb': sum(process_memories) / len(process_memories),
                'max_process_memory_mb': max(process_memories)
            },
            'configuration': {
                'strategy': self.strategy.value,
                'thresholds': {
                    'warning_percent': self.thresholds.warning_percent,
                    'critical_percent': self.thresholds.critical_percent,
                    'emergency_percent': self.thresholds.emergency_percent,
                    'process_limit_mb': self.thresholds.process_limit_mb
                },
                'gc_enabled': self.gc_enabled,
                'gc_thresholds': self.gc_thresholds
            },
            'alerts': {
                'total_alerts': len(self.memory_alerts),
                'recent_alerts': len([a for a in self.memory_alerts 
                                    if (datetime.now() - a['timestamp']).total_seconds() < 3600])
            },
            'tracking': {
                'tracked_categories': list(self.object_trackers.keys()),
                'tracked_object_counts': {
                    category: len(weak_set) 
                    for category, weak_set in self.object_trackers.items()
                }
            }
        }
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent memory alerts.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent alerts
        """
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.memory_alerts if alert['timestamp'] > cutoff_time]
    
    async def force_cleanup(self):
        """Force immediate cleanup and garbage collection."""
        
        logger.info("Forcing memory cleanup...")
        
        await self._emergency_cleanup()
        await self._trigger_gc(force=True)
        
        # Get metrics after cleanup
        metrics = await self.get_current_metrics()
        logger.info(f"Memory after cleanup: {metrics.memory_percent:.1f}%, "
                   f"Process: {metrics.process_memory_mb:.1f}MB")
    
    def tune_gc_thresholds(self, gen0: int, gen1: int, gen2: int):
        """Tune garbage collection thresholds.
        
        Args:
            gen0: Threshold for generation 0
            gen1: Threshold for generation 1  
            gen2: Threshold for generation 2
        """
        
        self.gc_thresholds = (gen0, gen1, gen2)
        if self.gc_enabled:
            gc.set_threshold(*self.gc_thresholds)
        
        logger.info(f"GC thresholds updated: {self.gc_thresholds}")
    
    def set_strategy(self, strategy: GarbageCollectionStrategy):
        """Change the garbage collection strategy.
        
        Args:
            strategy: New GC strategy
        """
        
        self.strategy = strategy
        logger.info(f"GC strategy changed to: {strategy.value}")
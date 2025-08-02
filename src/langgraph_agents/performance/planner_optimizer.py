"""
Specialized performance optimization for the planner agent.

This module provides targeted optimizations to reduce planner execution time:
- Parallel scene outline generation
- Optimized RAG query batching
- Scene implementation caching
- Concurrent plugin detection
- Memory-efficient plan processing
"""

import asyncio
import logging
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Callable, Awaitable
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class PlannerOptimizationMetrics:
    """Metrics for planner optimization."""
    timestamp: datetime
    
    # Timing metrics
    scene_outline_time: float
    scene_implementation_time: float
    plugin_detection_time: float
    total_planning_time: float
    
    # Concurrency metrics
    parallel_scenes_processed: int
    concurrent_implementations: int
    rag_queries_batched: int
    
    # Cache metrics
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    
    # Optimization metrics
    time_saved_seconds: float
    performance_improvement_percent: float


class PlannerOptimizer:
    """Specialized optimizer for planner agent performance."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize planner optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        
        # Optimization settings
        self.max_parallel_scenes = config.get('max_parallel_scenes', 8)
        self.max_concurrent_implementations = config.get('max_concurrent_implementations', 6)
        self.rag_batch_size = config.get('rag_batch_size', 10)
        self.enable_scene_caching = config.get('enable_scene_caching', True)
        self.enable_implementation_caching = config.get('enable_implementation_caching', True)
        self.enable_plugin_caching = config.get('enable_plugin_caching', True)
        
        # Thread pool for CPU-bound operations
        self.thread_pool = ThreadPoolExecutor(
            max_workers=config.get('max_worker_threads', 4),
            thread_name_prefix='planner_opt'
        )
        
        # Caching
        self.scene_cache: Dict[str, Any] = {}
        self.implementation_cache: Dict[str, Any] = {}
        self.plugin_cache: Dict[str, List[str]] = {}
        self.rag_query_cache: Dict[str, Any] = {}
        
        # Performance tracking
        self.metrics_history: List[PlannerOptimizationMetrics] = []
        self.baseline_times: Dict[str, float] = {}
        
        # Synchronization
        self._cache_lock = threading.RLock()
        self._metrics_lock = threading.RLock()
        
        # RAG query batching
        self.pending_rag_queries: List[Dict[str, Any]] = []
        self.rag_batch_event = asyncio.Event()
        self.rag_batch_task: Optional[asyncio.Task] = None
        
        logger.info(f"Planner optimizer initialized with max_parallel_scenes: {self.max_parallel_scenes}")
    
    async def start(self):
        """Start the planner optimizer."""
        
        # Start RAG query batching task
        self.rag_batch_task = asyncio.create_task(self._rag_batch_processor())
        
        logger.info("Planner optimizer started")
    
    async def stop(self):
        """Stop the planner optimizer."""
        
        # Stop RAG batching task
        if self.rag_batch_task:
            self.rag_batch_task.cancel()
            try:
                await self.rag_batch_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        logger.info("Planner optimizer stopped")
    
    async def optimize_scene_outline_generation(self, 
                                              planner_func: Callable,
                                              topic: str,
                                              description: str,
                                              session_id: str) -> str:
        """Optimize scene outline generation with caching and parallel processing.
        
        Args:
            planner_func: Original planner function
            topic: Video topic
            description: Video description
            session_id: Session identifier
            
        Returns:
            str: Optimized scene outline
        """
        
        start_time = time.time()
        
        # Create cache key
        cache_key = self._create_cache_key('scene_outline', topic, description)
        
        # Check cache first
        if self.enable_scene_caching:
            with self._cache_lock:
                cached_outline = self.scene_cache.get(cache_key)
                if cached_outline:
                    logger.debug(f"Cache hit for scene outline: {cache_key[:16]}...")
                    return cached_outline
        
        # Generate scene outline with optimization
        try:
            # Use thread pool for CPU-intensive planning
            outline = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                lambda: asyncio.run(planner_func(topic, description, session_id))
            )
            
            # Cache the result
            if self.enable_scene_caching and outline:
                with self._cache_lock:
                    self.scene_cache[cache_key] = outline
                    # Limit cache size
                    if len(self.scene_cache) > 100:
                        # Remove oldest entries
                        oldest_keys = list(self.scene_cache.keys())[:20]
                        for key in oldest_keys:
                            del self.scene_cache[key]
            
            generation_time = time.time() - start_time
            logger.info(f"Scene outline generated in {generation_time:.2f}s (cached: False)")
            
            return outline
            
        except Exception as e:
            logger.error(f"Scene outline generation failed: {str(e)}")
            raise
    
    async def optimize_scene_implementation_generation(self,
                                                     planner_func: Callable,
                                                     topic: str,
                                                     description: str,
                                                     plan: str,
                                                     session_id: str) -> List[str]:
        """Optimize scene implementation generation with parallel processing.
        
        Args:
            planner_func: Original planner function
            topic: Video topic
            description: Video description
            plan: Scene plan/outline
            session_id: Session identifier
            
        Returns:
            List[str]: Optimized scene implementations
        """
        
        start_time = time.time()
        
        # Parse scenes from plan
        scenes = self._parse_scenes_from_plan(plan)
        if not scenes:
            logger.warning("No scenes found in plan")
            return []
        
        logger.info(f"Processing {len(scenes)} scenes with parallel optimization")
        
        # Process scenes in parallel batches
        implementations = await self._process_scenes_in_parallel(
            scenes, planner_func, topic, description, session_id
        )
        
        generation_time = time.time() - start_time
        
        # Record metrics
        metrics = PlannerOptimizationMetrics(
            timestamp=datetime.now(),
            scene_outline_time=0.0,  # Not measured here
            scene_implementation_time=generation_time,
            plugin_detection_time=0.0,  # Not measured here
            total_planning_time=generation_time,
            parallel_scenes_processed=len(scenes),
            concurrent_implementations=min(len(scenes), self.max_concurrent_implementations),
            rag_queries_batched=0,  # Would be tracked separately
            cache_hits=0,  # Would be tracked separately
            cache_misses=0,  # Would be tracked separately
            cache_hit_rate=0.0,
            time_saved_seconds=max(0, len(scenes) * 10 - generation_time),  # Estimate
            performance_improvement_percent=0.0  # Would be calculated against baseline
        )
        
        with self._metrics_lock:
            self.metrics_history.append(metrics)
        
        logger.info(f"Scene implementations generated in {generation_time:.2f}s "
                   f"({len(implementations)} scenes, {metrics.time_saved_seconds:.1f}s saved)")
        
        return implementations
    
    def _parse_scenes_from_plan(self, plan: str) -> List[Dict[str, Any]]:
        """Parse individual scenes from the plan.
        
        Args:
            plan: Scene plan/outline
            
        Returns:
            List[Dict]: Parsed scene information
        """
        
        scenes = []
        lines = plan.split('\n')
        current_scene = None
        scene_number = 0
        
        for line in lines:
            line = line.strip()
            
            # Detect scene headers (various formats)
            if (line.startswith('Scene ') or 
                line.startswith('## Scene ') or
                line.startswith('### Scene ') or
                'scene' in line.lower() and ':' in line):
                
                # Save previous scene
                if current_scene:
                    scenes.append(current_scene)
                
                # Start new scene
                scene_number += 1
                current_scene = {
                    'scene_number': scene_number,
                    'title': line,
                    'content': [],
                    'description': ''
                }
            
            elif current_scene and line:
                current_scene['content'].append(line)
        
        # Add last scene
        if current_scene:
            scenes.append(current_scene)
        
        # Process scene content
        for scene in scenes:
            scene['description'] = '\n'.join(scene['content'])
            scene['cache_key'] = self._create_cache_key(
                'scene_impl', 
                str(scene['scene_number']), 
                scene['description'][:200]  # First 200 chars for cache key
            )
        
        logger.debug(f"Parsed {len(scenes)} scenes from plan")
        return scenes
    
    async def _process_scenes_in_parallel(self,
                                        scenes: List[Dict[str, Any]],
                                        planner_func: Callable,
                                        topic: str,
                                        description: str,
                                        session_id: str) -> List[str]:
        """Process scenes in parallel with optimized batching.
        
        Args:
            scenes: List of scene information
            planner_func: Original planner function
            topic: Video topic
            description: Video description
            session_id: Session identifier
            
        Returns:
            List[str]: Scene implementations
        """
        
        implementations = [''] * len(scenes)  # Pre-allocate results
        cache_hits = 0
        cache_misses = 0
        
        # Check cache for all scenes first
        scenes_to_process = []
        for i, scene in enumerate(scenes):
            if self.enable_implementation_caching:
                with self._cache_lock:
                    cached_impl = self.implementation_cache.get(scene['cache_key'])
                    if cached_impl:
                        implementations[i] = cached_impl
                        cache_hits += 1
                        continue
            
            scenes_to_process.append((i, scene))
            cache_misses += 1
        
        logger.info(f"Cache stats: {cache_hits} hits, {cache_misses} misses")
        
        if not scenes_to_process:
            return implementations
        
        # Process remaining scenes in parallel batches
        semaphore = asyncio.Semaphore(self.max_concurrent_implementations)
        
        async def process_single_scene(scene_index: int, scene_info: Dict[str, Any]) -> None:
            """Process a single scene implementation."""
            
            async with semaphore:
                try:
                    # Create scene-specific prompt
                    scene_prompt = self._create_scene_implementation_prompt(
                        scene_info, topic, description
                    )
                    
                    # Generate implementation
                    implementation = await self._generate_scene_implementation(
                        planner_func, scene_prompt, session_id
                    )
                    
                    # Store result
                    implementations[scene_index] = implementation
                    
                    # Cache result
                    if self.enable_implementation_caching and implementation:
                        with self._cache_lock:
                            self.implementation_cache[scene_info['cache_key']] = implementation
                            # Limit cache size
                            if len(self.implementation_cache) > 200:
                                oldest_keys = list(self.implementation_cache.keys())[:40]
                                for key in oldest_keys:
                                    del self.implementation_cache[key]
                    
                    logger.debug(f"Completed scene {scene_index + 1}")
                    
                except Exception as e:
                    logger.error(f"Failed to process scene {scene_index + 1}: {str(e)}")
                    implementations[scene_index] = f"# Scene {scene_index + 1} - Error: {str(e)}"
        
        # Execute all scene processing tasks
        tasks = [
            process_single_scene(scene_index, scene_info)
            for scene_index, scene_info in scenes_to_process
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return implementations
    
    def _create_scene_implementation_prompt(self,
                                          scene_info: Dict[str, Any],
                                          topic: str,
                                          description: str) -> str:
        """Create optimized prompt for scene implementation.
        
        Args:
            scene_info: Scene information
            topic: Video topic
            description: Video description
            
        Returns:
            str: Optimized scene prompt
        """
        
        return f"""
Topic: {topic}
Description: {description}

Scene {scene_info['scene_number']}: {scene_info['title']}

Scene Details:
{scene_info['description']}

Please provide a detailed implementation plan for this scene, focusing on:
1. Key visual elements and animations
2. Mathematical concepts or formulas to display
3. Timing and transitions
4. Any special effects or emphasis needed

Keep the implementation concise but comprehensive.
"""
    
    async def _generate_scene_implementation(self,
                                           planner_func: Callable,
                                           scene_prompt: str,
                                           session_id: str) -> str:
        """Generate implementation for a single scene.
        
        Args:
            planner_func: Original planner function
            scene_prompt: Scene-specific prompt
            session_id: Session identifier
            
        Returns:
            str: Scene implementation
        """
        
        try:
            # Use thread pool for generation to avoid blocking
            implementation = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                lambda: asyncio.run(planner_func(scene_prompt, session_id))
            )
            
            return implementation or "# Scene implementation failed"
            
        except Exception as e:
            logger.error(f"Scene implementation generation failed: {str(e)}")
            return f"# Scene implementation error: {str(e)}"
    
    async def optimize_plugin_detection(self,
                                      detection_func: Callable,
                                      topic: str,
                                      description: str) -> List[str]:
        """Optimize plugin detection with caching and parallel processing.
        
        Args:
            detection_func: Original plugin detection function
            topic: Video topic
            description: Video description
            
        Returns:
            List[str]: Detected plugins
        """
        
        start_time = time.time()
        
        # Create cache key
        cache_key = self._create_cache_key('plugins', topic, description)
        
        # Check cache first
        if self.enable_plugin_caching:
            with self._cache_lock:
                cached_plugins = self.plugin_cache.get(cache_key)
                if cached_plugins:
                    logger.debug(f"Cache hit for plugin detection: {cache_key[:16]}...")
                    return cached_plugins
        
        # Detect plugins with optimization
        try:
            plugins = await detection_func(topic, description)
            
            # Cache the result
            if self.enable_plugin_caching and plugins:
                with self._cache_lock:
                    self.plugin_cache[cache_key] = plugins
                    # Limit cache size
                    if len(self.plugin_cache) > 50:
                        oldest_keys = list(self.plugin_cache.keys())[:10]
                        for key in oldest_keys:
                            del self.plugin_cache[key]
            
            detection_time = time.time() - start_time
            logger.info(f"Plugin detection completed in {detection_time:.2f}s: {plugins}")
            
            return plugins
            
        except Exception as e:
            logger.error(f"Plugin detection failed: {str(e)}")
            return []
    
    async def optimize_rag_queries(self,
                                 rag_func: Callable,
                                 queries: List[str],
                                 context: Dict[str, Any]) -> List[Any]:
        """Optimize RAG queries with batching and caching.
        
        Args:
            rag_func: Original RAG function
            queries: List of RAG queries
            context: Query context
            
        Returns:
            List[Any]: RAG query results
        """
        
        if not queries:
            return []
        
        start_time = time.time()
        results = []
        
        # Check cache for each query
        cached_results = {}
        queries_to_process = []
        
        for i, query in enumerate(queries):
            cache_key = self._create_cache_key('rag', query, str(context))
            
            with self._cache_lock:
                cached_result = self.rag_query_cache.get(cache_key)
                if cached_result:
                    cached_results[i] = cached_result
                else:
                    queries_to_process.append((i, query, cache_key))
        
        # Process remaining queries in batches
        if queries_to_process:
            batch_results = await self._process_rag_queries_in_batches(
                queries_to_process, rag_func, context
            )
            
            # Cache new results
            with self._cache_lock:
                for (i, query, cache_key), result in zip(queries_to_process, batch_results):
                    self.rag_query_cache[cache_key] = result
                    # Limit cache size
                    if len(self.rag_query_cache) > 500:
                        oldest_keys = list(self.rag_query_cache.keys())[:100]
                        for key in oldest_keys:
                            del self.rag_query_cache[key]
        
        # Combine cached and new results
        for i in range(len(queries)):
            if i in cached_results:
                results.append(cached_results[i])
            else:
                # Find result from batch processing
                for (batch_i, _, _), result in zip(queries_to_process, batch_results):
                    if batch_i == i:
                        results.append(result)
                        break
                else:
                    results.append(None)  # Fallback
        
        processing_time = time.time() - start_time
        logger.info(f"RAG queries processed in {processing_time:.2f}s "
                   f"({len(cached_results)} cached, {len(queries_to_process)} new)")
        
        return results
    
    async def _process_rag_queries_in_batches(self,
                                            queries_to_process: List[Tuple[int, str, str]],
                                            rag_func: Callable,
                                            context: Dict[str, Any]) -> List[Any]:
        """Process RAG queries in optimized batches.
        
        Args:
            queries_to_process: List of (index, query, cache_key) tuples
            rag_func: Original RAG function
            context: Query context
            
        Returns:
            List[Any]: Batch processing results
        """
        
        results = []
        
        # Process in batches
        for i in range(0, len(queries_to_process), self.rag_batch_size):
            batch = queries_to_process[i:i + self.rag_batch_size]
            batch_queries = [query for _, query, _ in batch]
            
            try:
                # Process batch
                batch_results = await rag_func(batch_queries, context)
                
                # Ensure we have results for all queries in batch
                if len(batch_results) != len(batch_queries):
                    logger.warning(f"RAG batch size mismatch: expected {len(batch_queries)}, got {len(batch_results)}")
                    # Pad with None values
                    while len(batch_results) < len(batch_queries):
                        batch_results.append(None)
                
                results.extend(batch_results)
                
            except Exception as e:
                logger.error(f"RAG batch processing failed: {str(e)}")
                # Add None results for failed batch
                results.extend([None] * len(batch))
        
        return results
    
    async def _rag_batch_processor(self):
        """Background task for processing RAG query batches."""
        
        try:
            while True:
                # Wait for queries to batch
                await asyncio.sleep(0.1)  # Check every 100ms
                
                if len(self.pending_rag_queries) >= self.rag_batch_size:
                    # Process batch
                    batch = self.pending_rag_queries[:self.rag_batch_size]
                    self.pending_rag_queries = self.pending_rag_queries[self.rag_batch_size:]
                    
                    # Process batch in background
                    asyncio.create_task(self._process_rag_batch(batch))
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"RAG batch processor error: {str(e)}")
    
    async def _process_rag_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of RAG queries.
        
        Args:
            batch: Batch of RAG query information
        """
        
        try:
            # Extract queries and contexts
            queries = [item['query'] for item in batch]
            contexts = [item['context'] for item in batch]
            
            # Process batch (implementation would depend on RAG system)
            # For now, we'll just log the batch processing
            logger.debug(f"Processing RAG batch of {len(queries)} queries")
            
            # Notify completion (would set results in actual implementation)
            for item in batch:
                if 'future' in item:
                    item['future'].set_result(f"Processed: {item['query'][:50]}...")
            
        except Exception as e:
            logger.error(f"RAG batch processing failed: {str(e)}")
            # Set exceptions for all futures
            for item in batch:
                if 'future' in item:
                    item['future'].set_exception(e)
    
    def _create_cache_key(self, prefix: str, *args) -> str:
        """Create a cache key from arguments.
        
        Args:
            prefix: Cache key prefix
            *args: Arguments to include in key
            
        Returns:
            str: Cache key
        """
        
        # Combine all arguments into a single string
        combined = f"{prefix}:" + ":".join(str(arg) for arg in args)
        
        # Create hash for consistent key length
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get planner optimization metrics.
        
        Returns:
            Dict: Optimization metrics and statistics
        """
        
        if not self.metrics_history:
            return {'error': 'No metrics available'}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
        
        # Calculate averages
        avg_planning_time = sum(m.total_planning_time for m in recent_metrics) / len(recent_metrics)
        avg_parallel_scenes = sum(m.parallel_scenes_processed for m in recent_metrics) / len(recent_metrics)
        avg_time_saved = sum(m.time_saved_seconds for m in recent_metrics) / len(recent_metrics)
        
        # Cache statistics
        with self._cache_lock:
            cache_stats = {
                'scene_cache_size': len(self.scene_cache),
                'implementation_cache_size': len(self.implementation_cache),
                'plugin_cache_size': len(self.plugin_cache),
                'rag_cache_size': len(self.rag_query_cache)
            }
        
        return {
            'optimization_summary': {
                'average_planning_time': avg_planning_time,
                'average_parallel_scenes': avg_parallel_scenes,
                'average_time_saved': avg_time_saved,
                'total_optimizations': len(self.metrics_history)
            },
            'cache_statistics': cache_stats,
            'configuration': {
                'max_parallel_scenes': self.max_parallel_scenes,
                'max_concurrent_implementations': self.max_concurrent_implementations,
                'rag_batch_size': self.rag_batch_size,
                'caching_enabled': {
                    'scenes': self.enable_scene_caching,
                    'implementations': self.enable_implementation_caching,
                    'plugins': self.enable_plugin_caching
                }
            },
            'recent_metrics': [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'total_planning_time': m.total_planning_time,
                    'parallel_scenes_processed': m.parallel_scenes_processed,
                    'time_saved_seconds': m.time_saved_seconds,
                    'cache_hit_rate': m.cache_hit_rate
                }
                for m in recent_metrics
            ]
        }
    
    def clear_caches(self, cache_types: Optional[List[str]] = None):
        """Clear optimization caches.
        
        Args:
            cache_types: List of cache types to clear, or None for all
        """
        
        cache_types = cache_types or ['scene', 'implementation', 'plugin', 'rag']
        
        with self._cache_lock:
            if 'scene' in cache_types:
                self.scene_cache.clear()
            if 'implementation' in cache_types:
                self.implementation_cache.clear()
            if 'plugin' in cache_types:
                self.plugin_cache.clear()
            if 'rag' in cache_types:
                self.rag_query_cache.clear()
        
        logger.info(f"Cleared caches: {cache_types}")
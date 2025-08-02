"""
Performance benchmarking tests for LangGraph multi-agent video generation system.
Tests system performance, resource usage, and scalability.
"""

import pytest
import asyncio
import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List
from datetime import datetime, timedelta
import statistics
import json

from src.langgraph_agents.workflow import LangGraphVideoGenerator
from src.langgraph_agents.state import VideoGenerationState


class TestPerformanceBenchmarking:
    """Performance benchmarking tests for multi-agent workflows."""
    
    @pytest.fixture
    def performance_generator(self, temp_output_dir):
        """Create generator optimized for performance testing."""
        with patch('src.langgraph_agents.workflow.VideoGenerationWorkflow'), \
             patch('src.langgraph_agents.workflow.initialize_langfuse_service'), \
             patch('src.core.video_planner.EnhancedVideoPlanner'), \
             patch('src.core.code_generator.CodeGenerator'), \
             patch('src.core.video_renderer.OptimizedVideoRenderer'):
            
            generator = LangGraphVideoGenerator(
                planner_model="openai/gpt-4o-mini",
                scene_model="openai/gpt-4o-mini", 
                helper_model="openai/gpt-4o-mini",
                output_dir=temp_output_dir,
                verbose=False,  # Reduce logging overhead
                use_rag=True,
                use_langfuse=False,  # Disable for performance testing
                enable_human_loop=False,
                enable_monitoring=True,
                max_scene_concurrency=10,  # Higher concurrency for performance
                max_concurrent_renders=8,
                max_retries=2  # Fewer retries for faster testing
            )
            
            yield generator
    
    @pytest.fixture
    def benchmark_scenarios(self):
        """Different scenarios for performance benchmarking."""
        return {
            "small_video": {
                "topic": "Quick Python Tutorial",
                "description": "Short tutorial on Python basics",
                "expected_scenes": 2,
                "expected_duration": 60
            },
            "medium_video": {
                "topic": "Python Data Structures and Algorithms",
                "description": "Comprehensive guide to Python data structures and common algorithms",
                "expected_scenes": 5,
                "expected_duration": 300
            },
            "large_video": {
                "topic": "Complete Python Programming Course",
                "description": "Full Python programming course covering basics to advanced topics",
                "expected_scenes": 10,
                "expected_duration": 600
            },
            "complex_video": {
                "topic": "Advanced Python: Concurrency, Decorators, and Metaclasses",
                "description": "Advanced Python concepts with detailed examples and visualizations",
                "expected_scenes": 8,
                "expected_duration": 480
            }
        }
    
    def create_mock_state_for_scenario(self, scenario: Dict[str, Any], session_id: str, output_dir: str) -> VideoGenerationState:
        """Create mock state for performance testing scenario."""
        num_scenes = scenario["expected_scenes"]
        
        # Generate scene implementations
        scene_implementations = {}
        generated_code = {}
        rendered_videos = {}
        visual_analysis_results = {}
        rag_context = {}
        
        for i in range(1, num_scenes + 1):
            scene_implementations[i] = f"Scene {i}: Implementation for {scenario['topic']}"
            generated_code[i] = f"from manim import *\nclass Scene{i}(Scene):\n    def construct(self):\n        pass"
            rendered_videos[i] = f"{output_dir}/scene_{i}.mp4"
            visual_analysis_results[i] = {"quality_score": 0.85 + (i * 0.01), "visual_errors": []}
            rag_context[i] = f"RAG context for scene {i}"
        
        # Simulate realistic execution times based on scenario complexity
        base_time = 1.0
        planner_time = base_time * (num_scenes * 0.3)
        code_gen_time = base_time * (num_scenes * 1.2)
        render_time = base_time * (num_scenes * 2.5)
        
        return VideoGenerationState(
            messages=[],
            topic=scenario["topic"],
            description=scenario["description"],
            session_id=session_id,
            output_dir=output_dir,
            print_response=False,
            use_rag=True,
            use_context_learning=True,
            context_learning_path="data/context_learning",
            chroma_db_path="data/rag/chroma_db",
            manim_docs_path="data/rag/manim_docs",
            embedding_model="hf:ibm-granite/granite-embedding-30m-english",
            use_visual_fix_code=False,
            use_langfuse=False,
            max_scene_concurrency=10,
            max_topic_concurrency=1,
            max_retries=2,
            use_enhanced_rag=True,
            enable_rag_caching=True,
            enable_quality_monitoring=True,
            enable_error_handling=True,
            rag_cache_ttl=3600,
            rag_max_cache_size=1000,
            rag_performance_threshold=2.0,
            rag_quality_threshold=0.7,
            enable_caching=True,
            default_quality="medium",
            use_gpu_acceleration=False,
            preview_mode=False,
            max_concurrent_renders=8,
            scene_outline=f"# Scenes for {scenario['topic']}\n" + "\n".join([f"# Scene {i}: Content {i}" for i in range(1, num_scenes + 1)]),
            scene_implementations=scene_implementations,
            detected_plugins=["text", "code", "math"],
            generated_code=generated_code,
            code_errors={},
            rag_context=rag_context,
            rendered_videos=rendered_videos,
            combined_video_path=f"{output_dir}/final_video.mp4",
            rendering_errors={},
            visual_analysis_results=visual_analysis_results,
            visual_errors={},
            error_count=0,
            retry_count={},
            escalated_errors=[],
            pending_human_input=None,
            human_feedback=None,
            performance_metrics={
                "planner_agent": {
                    "last_execution_time": planner_time,
                    "average_execution_time": planner_time,
                    "success_rate": 1.0
                },
                "code_generator_agent": {
                    "last_execution_time": code_gen_time,
                    "average_execution_time": code_gen_time,
                    "success_rate": 1.0
                },
                "renderer_agent": {
                    "last_execution_time": render_time,
                    "average_execution_time": render_time,
                    "success_rate": 1.0
                }
            },
            execution_trace=[
                {
                    "agent": "planner_agent",
                    "action": "complete_execution",
                    "timestamp": datetime.now().isoformat(),
                    "execution_time": planner_time
                },
                {
                    "agent": "code_generator_agent",
                    "action": "complete_execution", 
                    "timestamp": (datetime.now() + timedelta(seconds=planner_time)).isoformat(),
                    "execution_time": code_gen_time
                },
                {
                    "agent": "renderer_agent",
                    "action": "complete_execution",
                    "timestamp": (datetime.now() + timedelta(seconds=planner_time + code_gen_time)).isoformat(),
                    "execution_time": render_time
                }
            ],
            current_agent=None,
            next_agent=None,
            workflow_complete=True,
            workflow_interrupted=False
        )
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_single_workflow_performance(self, performance_generator, benchmark_scenarios):
        """Test performance of single workflow execution across different scenarios."""
        generator = performance_generator
        performance_results = {}
        
        for scenario_name, scenario in benchmark_scenarios.items():
            session_id = f"perf_test_{scenario_name}"
            
            # Create mock state for scenario
            mock_state = self.create_mock_state_for_scenario(
                scenario, session_id, generator.init_params["output_dir"]
            )
            
            # Mock workflow execution with realistic timing
            async def mock_invoke_with_delay(*args, **kwargs):
                # Simulate realistic processing time based on scenario complexity
                delay = scenario["expected_scenes"] * 0.1  # 0.1s per scene for testing
                await asyncio.sleep(delay)
                return mock_state
            
            generator.workflow.invoke = mock_invoke_with_delay
            
            # Measure performance
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            final_state = await generator.generate_video_pipeline(
                topic=scenario["topic"],
                description=scenario["description"],
                session_id=session_id
            )
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Calculate metrics
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            scenes_per_second = scenario["expected_scenes"] / execution_time
            
            performance_results[scenario_name] = {
                "execution_time": execution_time,
                "memory_usage": memory_usage,
                "scenes_per_second": scenes_per_second,
                "num_scenes": scenario["expected_scenes"],
                "workflow_complete": final_state["workflow_complete"],
                "error_count": final_state["error_count"]
            }
            
            # Verify successful completion
            assert final_state["workflow_complete"] is True
            assert final_state["error_count"] == 0
            
            print(f"✓ {scenario_name}: {execution_time:.2f}s, {memory_usage:.1f}MB, {scenes_per_second:.2f} scenes/s")
        
        # Analyze performance trends
        execution_times = [r["execution_time"] for r in performance_results.values()]
        memory_usages = [r["memory_usage"] for r in performance_results.values()]
        
        print(f"\nPerformance Summary:")
        print(f"Average execution time: {statistics.mean(execution_times):.2f}s")
        print(f"Average memory usage: {statistics.mean(memory_usages):.1f}MB")
        print(f"Max execution time: {max(execution_times):.2f}s")
        print(f"Max memory usage: {max(memory_usages):.1f}MB")
        
        # Performance assertions
        assert all(r["execution_time"] < 30.0 for r in performance_results.values()), "Execution time too high"
        assert all(r["memory_usage"] < 500.0 for r in performance_results.values()), "Memory usage too high"
        assert all(r["scenes_per_second"] > 0.1 for r in performance_results.values()), "Processing rate too low"
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_concurrent_workflow_performance(self, performance_generator, benchmark_scenarios):
        """Test performance of concurrent workflow executions."""
        generator = performance_generator
        num_concurrent = 3
        scenario = benchmark_scenarios["medium_video"]
        
        # Create mock states for concurrent executions
        mock_states = []
        for i in range(num_concurrent):
            session_id = f"concurrent_test_{i}"
            mock_state = self.create_mock_state_for_scenario(
                scenario, session_id, generator.init_params["output_dir"]
            )
            mock_states.append(mock_state)
        
        # Mock workflow execution with realistic timing
        execution_count = 0
        async def mock_invoke_with_delay(*args, **kwargs):
            nonlocal execution_count
            # Simulate realistic processing time with some variation
            delay = (scenario["expected_scenes"] * 0.1) + (execution_count * 0.05)
            await asyncio.sleep(delay)
            state = mock_states[execution_count % len(mock_states)]
            execution_count += 1
            return state
        
        generator.workflow.invoke = mock_invoke_with_delay
        
        # Create concurrent tasks
        async def run_workflow(session_id: str):
            return await generator.generate_video_pipeline(
                topic=scenario["topic"],
                description=scenario["description"],
                session_id=session_id
            )
        
        tasks = [
            run_workflow(f"concurrent_test_{i}")
            for i in range(num_concurrent)
        ]
        
        # Measure concurrent performance
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Calculate metrics
        total_execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        throughput = num_concurrent / total_execution_time
        
        # Verify all workflows completed successfully
        for i, result in enumerate(results):
            assert result["workflow_complete"] is True, f"Workflow {i} did not complete"
            assert result["error_count"] == 0, f"Workflow {i} had errors"
        
        print(f"✓ Concurrent execution: {num_concurrent} workflows in {total_execution_time:.2f}s")
        print(f"  Throughput: {throughput:.2f} workflows/s")
        print(f"  Memory usage: {memory_usage:.1f}MB")
        
        # Performance assertions for concurrent execution
        assert total_execution_time < (num_concurrent * 10.0), "Concurrent execution too slow"
        assert memory_usage < 1000.0, "Concurrent memory usage too high"
        assert throughput > 0.1, "Concurrent throughput too low"
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_memory_usage_patterns(self, performance_generator, benchmark_scenarios):
        """Test memory usage patterns during workflow execution."""
        generator = performance_generator
        scenario = benchmark_scenarios["large_video"]
        session_id = "memory_test"
        
        # Create mock state
        mock_state = self.create_mock_state_for_scenario(
            scenario, session_id, generator.init_params["output_dir"]
        )
        
        # Track memory usage during execution
        memory_samples = []
        monitoring_active = True
        
        def memory_monitor():
            while monitoring_active:
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                memory_samples.append({
                    "timestamp": time.time(),
                    "memory_mb": memory_mb
                })
                time.sleep(0.1)  # Sample every 100ms
        
        # Mock workflow execution with memory simulation
        async def mock_invoke_with_memory_pattern(*args, **kwargs):
            # Simulate memory usage pattern during workflow
            phases = [
                ("planning", 0.5, 50),    # Planning phase: 0.5s, +50MB
                ("code_gen", 2.0, 150),  # Code generation: 2.0s, +150MB
                ("rendering", 3.0, 200), # Rendering: 3.0s, +200MB
                ("cleanup", 0.5, -100)   # Cleanup: 0.5s, -100MB
            ]
            
            for phase_name, duration, memory_delta in phases:
                # Simulate memory allocation/deallocation
                if memory_delta > 0:
                    # Simulate memory allocation
                    dummy_data = [0] * (memory_delta * 1000)  # Rough memory allocation
                await asyncio.sleep(duration)
                if memory_delta < 0:
                    # Simulate memory cleanup
                    dummy_data = None
            
            return mock_state
        
        generator.workflow.invoke = mock_invoke_with_memory_pattern
        
        # Start memory monitoring
        monitor_thread = threading.Thread(target=memory_monitor)
        monitor_thread.start()
        
        try:
            # Execute workflow
            start_time = time.time()
            final_state = await generator.generate_video_pipeline(
                topic=scenario["topic"],
                description=scenario["description"],
                session_id=session_id
            )
            end_time = time.time()
            
        finally:
            # Stop memory monitoring
            monitoring_active = False
            monitor_thread.join()
        
        # Analyze memory usage patterns
        if memory_samples:
            start_memory = memory_samples[0]["memory_mb"]
            peak_memory = max(sample["memory_mb"] for sample in memory_samples)
            end_memory = memory_samples[-1]["memory_mb"]
            
            memory_growth = peak_memory - start_memory
            memory_cleanup = peak_memory - end_memory
            
            print(f"✓ Memory usage analysis:")
            print(f"  Start memory: {start_memory:.1f}MB")
            print(f"  Peak memory: {peak_memory:.1f}MB")
            print(f"  End memory: {end_memory:.1f}MB")
            print(f"  Memory growth: {memory_growth:.1f}MB")
            print(f"  Memory cleanup: {memory_cleanup:.1f}MB")
            
            # Memory usage assertions
            assert memory_growth < 1000.0, "Memory growth too high"
            assert memory_cleanup > (memory_growth * 0.5), "Insufficient memory cleanup"
            assert final_state["workflow_complete"] is True
        
    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_scalability_limits(self, performance_generator, benchmark_scenarios):
        """Test system scalability limits and breaking points."""
        generator = performance_generator
        scenario = benchmark_scenarios["small_video"]
        
        # Test increasing concurrent load
        concurrent_loads = [1, 2, 4, 8]
        scalability_results = {}
        
        for load in concurrent_loads:
            # Create mock states for concurrent executions
            mock_states = []
            for i in range(load):
                session_id = f"scale_test_{load}_{i}"
                mock_state = self.create_mock_state_for_scenario(
                    scenario, session_id, generator.init_params["output_dir"]
                )
                mock_states.append(mock_state)
            
            # Mock workflow execution
            execution_count = 0
            async def mock_invoke_for_scale(*args, **kwargs):
                nonlocal execution_count
                # Simulate increasing latency with load
                base_delay = scenario["expected_scenes"] * 0.05
                load_penalty = (load - 1) * 0.02  # Penalty for higher load
                delay = base_delay + load_penalty
                await asyncio.sleep(delay)
                
                state = mock_states[execution_count % len(mock_states)]
                execution_count += 1
                return state
            
            generator.workflow.invoke = mock_invoke_for_scale
            
            # Create concurrent tasks
            tasks = [
                generator.generate_video_pipeline(
                    topic=scenario["topic"],
                    description=scenario["description"],
                    session_id=f"scale_test_{load}_{i}"
                )
                for i in range(load)
            ]
            
            # Measure performance under load
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks),
                    timeout=30.0  # 30 second timeout
                )
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                # Calculate metrics
                total_time = end_time - start_time
                memory_usage = end_memory - start_memory
                throughput = load / total_time
                avg_latency = total_time / load
                
                # Verify all completed successfully
                success_count = sum(1 for r in results if r["workflow_complete"])
                success_rate = success_count / load
                
                scalability_results[load] = {
                    "total_time": total_time,
                    "memory_usage": memory_usage,
                    "throughput": throughput,
                    "avg_latency": avg_latency,
                    "success_rate": success_rate,
                    "completed": True
                }
                
                print(f"✓ Load {load}: {total_time:.2f}s, {throughput:.2f} workflows/s, {success_rate:.1%} success")
                
            except asyncio.TimeoutError:
                scalability_results[load] = {
                    "completed": False,
                    "reason": "timeout"
                }
                print(f"✗ Load {load}: Timed out after 30s")
                break
            
            except Exception as e:
                scalability_results[load] = {
                    "completed": False,
                    "reason": str(e)
                }
                print(f"✗ Load {load}: Failed with error: {e}")
                break
        
        # Analyze scalability results
        successful_loads = [load for load, result in scalability_results.items() if result.get("completed")]
        
        if successful_loads:
            max_successful_load = max(successful_loads)
            throughputs = [scalability_results[load]["throughput"] for load in successful_loads]
            
            print(f"\nScalability Analysis:")
            print(f"Maximum successful concurrent load: {max_successful_load}")
            print(f"Peak throughput: {max(throughputs):.2f} workflows/s")
            print(f"Throughput at max load: {scalability_results[max_successful_load]['throughput']:.2f} workflows/s")
            
            # Scalability assertions
            assert max_successful_load >= 2, "System should handle at least 2 concurrent workflows"
            assert max(throughputs) > 0.5, "Peak throughput should be reasonable"
        
    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_performance_regression_detection(self, performance_generator, benchmark_scenarios):
        """Test for performance regression detection."""
        generator = performance_generator
        scenario = benchmark_scenarios["medium_video"]
        
        # Baseline performance expectations (these would be updated based on actual measurements)
        baseline_expectations = {
            "max_execution_time": 10.0,  # seconds
            "max_memory_usage": 300.0,   # MB
            "min_throughput": 0.2,       # workflows/s
            "max_error_rate": 0.05       # 5%
        }
        
        # Run multiple iterations to get stable measurements
        num_iterations = 3
        performance_measurements = []
        
        for iteration in range(num_iterations):
            session_id = f"regression_test_{iteration}"
            
            # Create mock state
            mock_state = self.create_mock_state_for_scenario(
                scenario, session_id, generator.init_params["output_dir"]
            )
            
            # Mock workflow execution
            async def mock_invoke_for_regression(*args, **kwargs):
                # Simulate realistic processing time with some variation
                base_delay = scenario["expected_scenes"] * 0.1
                variation = (iteration * 0.02)  # Small variation per iteration
                delay = base_delay + variation
                await asyncio.sleep(delay)
                return mock_state
            
            generator.workflow.invoke = mock_invoke_for_regression
            
            # Measure performance
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            final_state = await generator.generate_video_pipeline(
                topic=scenario["topic"],
                description=scenario["description"],
                session_id=session_id
            )
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Record measurements
            measurement = {
                "iteration": iteration,
                "execution_time": end_time - start_time,
                "memory_usage": end_memory - start_memory,
                "success": final_state["workflow_complete"],
                "error_count": final_state["error_count"]
            }
            performance_measurements.append(measurement)
        
        # Analyze measurements
        execution_times = [m["execution_time"] for m in performance_measurements]
        memory_usages = [m["memory_usage"] for m in performance_measurements]
        success_count = sum(1 for m in performance_measurements if m["success"])
        
        avg_execution_time = statistics.mean(execution_times)
        avg_memory_usage = statistics.mean(memory_usages)
        success_rate = success_count / num_iterations
        error_rate = 1 - success_rate
        throughput = 1 / avg_execution_time
        
        print(f"Performance Regression Analysis:")
        print(f"Average execution time: {avg_execution_time:.2f}s (baseline: <{baseline_expectations['max_execution_time']}s)")
        print(f"Average memory usage: {avg_memory_usage:.1f}MB (baseline: <{baseline_expectations['max_memory_usage']}MB)")
        print(f"Throughput: {throughput:.2f} workflows/s (baseline: >{baseline_expectations['min_throughput']} workflows/s)")
        print(f"Error rate: {error_rate:.1%} (baseline: <{baseline_expectations['max_error_rate']:.1%})")
        
        # Regression detection assertions
        assert avg_execution_time <= baseline_expectations["max_execution_time"], f"Performance regression: execution time {avg_execution_time:.2f}s exceeds baseline {baseline_expectations['max_execution_time']}s"
        assert avg_memory_usage <= baseline_expectations["max_memory_usage"], f"Performance regression: memory usage {avg_memory_usage:.1f}MB exceeds baseline {baseline_expectations['max_memory_usage']}MB"
        assert throughput >= baseline_expectations["min_throughput"], f"Performance regression: throughput {throughput:.2f} below baseline {baseline_expectations['min_throughput']}"
        assert error_rate <= baseline_expectations["max_error_rate"], f"Performance regression: error rate {error_rate:.1%} exceeds baseline {baseline_expectations['max_error_rate']:.1%}"
        
        print("✓ No performance regression detected")
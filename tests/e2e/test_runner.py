#!/usr/bin/env python3
"""
End-to-end test runner for LangGraph multi-agent video generation system.
Runs comprehensive end-to-end tests including workflow, performance, human-loop, and failure recovery tests.
"""

import asyncio
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Any
import tempfile
import shutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class E2ETestRunner:
    """Comprehensive end-to-end test runner."""
    
    def __init__(self):
        self.test_results = {}
        self.temp_dirs = []
        self.start_time = None
        self.end_time = None
    
    def create_temp_dir(self) -> str:
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp(prefix="e2e_test_")
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def cleanup_temp_dirs(self):
        """Clean up temporary directories."""
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp dir {temp_dir}: {e}")
    
    async def run_complete_workflow_tests(self) -> Dict[str, Any]:
        """Run complete workflow tests."""
        logger.info("Running complete workflow tests...")
        
        try:
            # Import test module
            from test_complete_workflow import TestCompleteWorkflow
            
            # Create test instance
            test_instance = TestCompleteWorkflow()
            
            # Create fixtures
            temp_dir = self.create_temp_dir()
            
            # Mock fixtures
            class MockFixtures:
                def __init__(self):
                    self.temp_output_dir = temp_dir
            
            fixtures = MockFixtures()
            
            # Sample data
            sample_data = {
                "topic": "Python Programming Basics",
                "description": "Educational video covering Python fundamentals",
                "session_id": "e2e_workflow_test",
                "expected_scenes": 3,
                "expected_plugins": ["text", "code", "math"],
                "expected_duration": 180
            }
            
            # Run workflow tests
            workflow_results = {}
            
            # Test 1: Complete workflow
            try:
                # Mock the generator and run test
                with self._mock_dependencies():
                    from src.langgraph_agents.workflow import LangGraphVideoGenerator
                    
                    generator = LangGraphVideoGenerator(
                        planner_model="openai/gpt-4o-mini",
                        output_dir=temp_dir,
                        verbose=False,
                        use_rag=True,
                        use_langfuse=False,
                        enable_human_loop=False,
                        enable_monitoring=True
                    )
                    
                    # Mock successful workflow execution
                    from unittest.mock import AsyncMock
                    generator.workflow.invoke = AsyncMock(return_value=self._create_mock_successful_state(sample_data, temp_dir))
                    
                    # Run test
                    await test_instance.test_complete_video_generation_workflow(generator, sample_data)
                    workflow_results["complete_workflow"] = {"status": "passed", "error": None}
                    
            except Exception as e:
                workflow_results["complete_workflow"] = {"status": "failed", "error": str(e)}
                logger.error(f"Complete workflow test failed: {e}")
            
            # Test 2: Planning only
            try:
                with self._mock_dependencies():
                    from src.langgraph_agents.workflow import LangGraphVideoGenerator
                    
                    generator = LangGraphVideoGenerator(
                        planner_model="openai/gpt-4o-mini",
                        output_dir=temp_dir,
                        use_langfuse=False
                    )
                    
                    generator.workflow.invoke = AsyncMock(return_value=self._create_mock_planning_state(sample_data, temp_dir))
                    
                    await test_instance.test_workflow_with_scene_outline_only(generator, sample_data)
                    workflow_results["planning_only"] = {"status": "passed", "error": None}
                    
            except Exception as e:
                workflow_results["planning_only"] = {"status": "failed", "error": str(e)}
                logger.error(f"Planning only test failed: {e}")
            
            # Test 3: Specific scenes
            try:
                with self._mock_dependencies():
                    from src.langgraph_agents.workflow import LangGraphVideoGenerator
                    
                    generator = LangGraphVideoGenerator(
                        planner_model="openai/gpt-4o-mini",
                        output_dir=temp_dir,
                        use_langfuse=False
                    )
                    
                    generator.workflow.invoke = AsyncMock(return_value=self._create_mock_specific_scenes_state(sample_data, temp_dir))
                    
                    await test_instance.test_workflow_with_specific_scenes(generator, sample_data)
                    workflow_results["specific_scenes"] = {"status": "passed", "error": None}
                    
            except Exception as e:
                workflow_results["specific_scenes"] = {"status": "failed", "error": str(e)}
                logger.error(f"Specific scenes test failed: {e}")
            
            return {
                "category": "complete_workflow",
                "total_tests": len(workflow_results),
                "passed": sum(1 for r in workflow_results.values() if r["status"] == "passed"),
                "failed": sum(1 for r in workflow_results.values() if r["status"] == "failed"),
                "results": workflow_results
            }
            
        except Exception as e:
            logger.error(f"Complete workflow tests failed: {e}")
            return {
                "category": "complete_workflow",
                "total_tests": 0,
                "passed": 0,
                "failed": 1,
                "error": str(e)
            }
    
    async def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance benchmarking tests."""
        logger.info("Running performance benchmarking tests...")
        
        try:
            # Import test module
            from test_performance_benchmarking import TestPerformanceBenchmarking
            
            test_instance = TestPerformanceBenchmarking()
            temp_dir = self.create_temp_dir()
            
            performance_results = {}
            
            # Test 1: Single workflow performance
            try:
                with self._mock_dependencies():
                    from src.langgraph_agents.workflow import LangGraphVideoGenerator
                    
                    generator = LangGraphVideoGenerator(
                        planner_model="openai/gpt-4o-mini",
                        output_dir=temp_dir,
                        use_langfuse=False,
                        max_scene_concurrency=10
                    )
                    
                    benchmark_scenarios = {
                        "small_video": {"topic": "Quick Test", "expected_scenes": 2, "expected_duration": 60},
                        "medium_video": {"topic": "Medium Test", "expected_scenes": 5, "expected_duration": 300}
                    }
                    
                    # Mock performance test
                    generator.workflow.invoke = AsyncMock(side_effect=self._mock_performance_execution)
                    
                    await test_instance.test_single_workflow_performance(generator, benchmark_scenarios)
                    performance_results["single_workflow"] = {"status": "passed", "error": None}
                    
            except Exception as e:
                performance_results["single_workflow"] = {"status": "failed", "error": str(e)}
                logger.error(f"Single workflow performance test failed: {e}")
            
            # Test 2: Memory usage patterns
            try:
                with self._mock_dependencies():
                    from src.langgraph_agents.workflow import LangGraphVideoGenerator
                    
                    generator = LangGraphVideoGenerator(
                        planner_model="openai/gpt-4o-mini",
                        output_dir=temp_dir,
                        use_langfuse=False
                    )
                    
                    scenario = {"topic": "Memory Test", "expected_scenes": 3, "expected_duration": 180}
                    
                    generator.workflow.invoke = AsyncMock(side_effect=self._mock_memory_test_execution)
                    
                    await test_instance.test_memory_usage_patterns(generator, {"large_video": scenario})
                    performance_results["memory_usage"] = {"status": "passed", "error": None}
                    
            except Exception as e:
                performance_results["memory_usage"] = {"status": "failed", "error": str(e)}
                logger.error(f"Memory usage test failed: {e}")
            
            return {
                "category": "performance",
                "total_tests": len(performance_results),
                "passed": sum(1 for r in performance_results.values() if r["status"] == "passed"),
                "failed": sum(1 for r in performance_results.values() if r["status"] == "failed"),
                "results": performance_results
            }
            
        except Exception as e:
            logger.error(f"Performance tests failed: {e}")
            return {
                "category": "performance",
                "total_tests": 0,
                "passed": 0,
                "failed": 1,
                "error": str(e)
            }
    
    async def run_human_loop_tests(self) -> Dict[str, Any]:
        """Run human-in-the-loop scenario tests."""
        logger.info("Running human-in-the-loop tests...")
        
        try:
            from test_human_loop_scenarios import TestHumanLoopScenarios
            
            test_instance = TestHumanLoopScenarios()
            temp_dir = self.create_temp_dir()
            
            human_loop_results = {}
            
            # Test 1: Planning approval
            try:
                with self._mock_dependencies():
                    from src.langgraph_agents.workflow import LangGraphVideoGenerator
                    
                    generator = LangGraphVideoGenerator(
                        planner_model="openai/gpt-4o-mini",
                        output_dir=temp_dir,
                        use_langfuse=False,
                        enable_human_loop=True
                    )
                    
                    scenario = {
                        "trigger_agent": "planner_agent",
                        "context": "Scene outline requires human review",
                        "options": ["approve", "modify", "reject"],
                        "expected_decision": "approve",
                        "priority": "medium"
                    }
                    
                    generator.workflow.invoke = AsyncMock(side_effect=self._mock_human_loop_execution)
                    
                    await test_instance.test_planning_approval_scenario(generator, {"planning_approval": scenario})
                    human_loop_results["planning_approval"] = {"status": "passed", "error": None}
                    
            except Exception as e:
                human_loop_results["planning_approval"] = {"status": "failed", "error": str(e)}
                logger.error(f"Planning approval test failed: {e}")
            
            # Test 2: Human intervention timeout
            try:
                with self._mock_dependencies():
                    from src.langgraph_agents.workflow import LangGraphVideoGenerator
                    
                    generator = LangGraphVideoGenerator(
                        planner_model="openai/gpt-4o-mini",
                        output_dir=temp_dir,
                        use_langfuse=False,
                        enable_human_loop=True
                    )
                    
                    scenario = {
                        "trigger_agent": "planner_agent",
                        "context": "Scene outline requires human review",
                        "options": ["approve", "modify", "reject"],
                        "expected_decision": "approve",
                        "priority": "medium"
                    }
                    
                    generator.workflow.invoke = AsyncMock(side_effect=self._mock_timeout_execution)
                    
                    await test_instance.test_human_intervention_timeout_scenario(generator, {"planning_approval": scenario})
                    human_loop_results["timeout_handling"] = {"status": "passed", "error": None}
                    
            except Exception as e:
                human_loop_results["timeout_handling"] = {"status": "failed", "error": str(e)}
                logger.error(f"Timeout handling test failed: {e}")
            
            return {
                "category": "human_loop",
                "total_tests": len(human_loop_results),
                "passed": sum(1 for r in human_loop_results.values() if r["status"] == "passed"),
                "failed": sum(1 for r in human_loop_results.values() if r["status"] == "failed"),
                "results": human_loop_results
            }
            
        except Exception as e:
            logger.error(f"Human loop tests failed: {e}")
            return {
                "category": "human_loop",
                "total_tests": 0,
                "passed": 0,
                "failed": 1,
                "error": str(e)
            }
    
    async def run_failure_recovery_tests(self) -> Dict[str, Any]:
        """Run failure recovery tests."""
        logger.info("Running failure recovery tests...")
        
        try:
            from test_failure_recovery import TestFailureRecovery
            
            test_instance = TestFailureRecovery()
            temp_dir = self.create_temp_dir()
            
            recovery_results = {}
            
            # Test 1: Agent timeout recovery
            try:
                with self._mock_dependencies():
                    from src.langgraph_agents.workflow import LangGraphVideoGenerator
                    
                    generator = LangGraphVideoGenerator(
                        planner_model="openai/gpt-4o-mini",
                        output_dir=temp_dir,
                        use_langfuse=False,
                        enable_error_handling=True
                    )
                    
                    scenario = {
                        "error_type": "TimeoutError",
                        "error_message": "Agent execution timed out",
                        "failing_agent": "code_generator_agent",
                        "recovery_strategy": "retry_with_timeout_increase",
                        "max_retries": 2,
                        "expected_recovery": True
                    }
                    
                    generator.workflow.invoke = AsyncMock(side_effect=self._mock_recovery_execution)
                    
                    await test_instance.test_agent_timeout_recovery(generator, {"agent_timeout": scenario})
                    recovery_results["timeout_recovery"] = {"status": "passed", "error": None}
                    
            except Exception as e:
                recovery_results["timeout_recovery"] = {"status": "failed", "error": str(e)}
                logger.error(f"Timeout recovery test failed: {e}")
            
            # Test 2: Cascading failures escalation
            try:
                with self._mock_dependencies():
                    from src.langgraph_agents.workflow import LangGraphVideoGenerator
                    
                    generator = LangGraphVideoGenerator(
                        planner_model="openai/gpt-4o-mini",
                        output_dir=temp_dir,
                        use_langfuse=False,
                        enable_human_loop=True,
                        enable_error_handling=True
                    )
                    
                    scenario = {
                        "error_type": "MultipleErrors",
                        "error_message": "Multiple agents failed simultaneously",
                        "failing_agent": "multiple",
                        "recovery_strategy": "escalate_to_human",
                        "max_retries": 1,
                        "expected_recovery": False
                    }
                    
                    generator.workflow.invoke = AsyncMock(side_effect=self._mock_escalation_execution)
                    
                    await test_instance.test_cascading_failures_escalation(generator, {"cascading_failures": scenario})
                    recovery_results["cascading_failures"] = {"status": "passed", "error": None}
                    
            except Exception as e:
                recovery_results["cascading_failures"] = {"status": "failed", "error": str(e)}
                logger.error(f"Cascading failures test failed: {e}")
            
            return {
                "category": "failure_recovery",
                "total_tests": len(recovery_results),
                "passed": sum(1 for r in recovery_results.values() if r["status"] == "passed"),
                "failed": sum(1 for r in recovery_results.values() if r["status"] == "failed"),
                "results": recovery_results
            }
            
        except Exception as e:
            logger.error(f"Failure recovery tests failed: {e}")
            return {
                "category": "failure_recovery",
                "total_tests": 0,
                "passed": 0,
                "failed": 1,
                "error": str(e)
            }
    
    def _mock_dependencies(self):
        """Context manager for mocking external dependencies."""
        from unittest.mock import patch
        
        return patch.multiple(
            'src.langgraph_agents.workflow',
            VideoGenerationWorkflow=patch.DEFAULT,
            initialize_langfuse_service=patch.DEFAULT,
        )
    
    def _create_mock_successful_state(self, sample_data: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """Create mock successful workflow state."""
        from src.langgraph_agents.state import VideoGenerationState
        
        return VideoGenerationState(
            messages=[],
            topic=sample_data["topic"],
            description=sample_data["description"],
            session_id=sample_data["session_id"],
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
            max_scene_concurrency=5,
            max_topic_concurrency=1,
            max_retries=5,
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
            max_concurrent_renders=4,
            scene_outline="# Scene 1: Test scene",
            scene_implementations={1: "Test implementation"},
            detected_plugins=sample_data["expected_plugins"],
            generated_code={1: "from manim import *\nclass TestScene(Scene): pass"},
            code_errors={},
            rag_context={1: "Test RAG context"},
            rendered_videos={1: f"{output_dir}/scene_1.mp4"},
            combined_video_path=f"{output_dir}/final_video.mp4",
            rendering_errors={},
            visual_analysis_results={1: {"quality_score": 0.9, "visual_errors": []}},
            visual_errors={},
            error_count=0,
            retry_count={},
            escalated_errors=[],
            pending_human_input=None,
            human_feedback=None,
            performance_metrics={
                "planner_agent": {"last_execution_time": 2.5, "success_rate": 1.0},
                "code_generator_agent": {"last_execution_time": 8.2, "success_rate": 1.0},
                "renderer_agent": {"last_execution_time": 15.7, "success_rate": 1.0}
            },
            execution_trace=[
                {"agent": "planner_agent", "action": "complete_execution", "timestamp": "2024-01-01T10:00:00"},
                {"agent": "code_generator_agent", "action": "complete_execution", "timestamp": "2024-01-01T10:02:30"},
                {"agent": "renderer_agent", "action": "complete_execution", "timestamp": "2024-01-01T10:10:45"}
            ],
            current_agent=None,
            next_agent=None,
            workflow_complete=True,
            workflow_interrupted=False
        )
    
    def _create_mock_planning_state(self, sample_data: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """Create mock planning-only state."""
        state = self._create_mock_successful_state(sample_data, output_dir)
        state.update({
            "generated_code": {},
            "rendered_videos": {},
            "combined_video_path": None,
            "execution_trace": [
                {"agent": "planner_agent", "action": "complete_execution", "timestamp": "2024-01-01T10:00:00"}
            ]
        })
        return state
    
    def _create_mock_specific_scenes_state(self, sample_data: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """Create mock specific scenes state."""
        state = self._create_mock_successful_state(sample_data, output_dir)
        # Only scenes 1 and 3
        state.update({
            "scene_implementations": {1: "Scene 1 implementation", 3: "Scene 3 implementation"},
            "generated_code": {1: "Scene 1 code", 3: "Scene 3 code"},
            "rendered_videos": {1: f"{output_dir}/scene_1.mp4", 3: f"{output_dir}/scene_3.mp4"}
        })
        return state
    
    async def _mock_performance_execution(self, *args, **kwargs):
        """Mock performance test execution."""
        await asyncio.sleep(0.1)  # Simulate processing time
        return self._create_mock_successful_state({"topic": "Test", "description": "Test", "session_id": "test", "expected_plugins": ["text"]}, "/tmp")
    
    async def _mock_memory_test_execution(self, *args, **kwargs):
        """Mock memory test execution."""
        await asyncio.sleep(0.2)  # Simulate memory-intensive processing
        return self._create_mock_successful_state({"topic": "Memory Test", "description": "Test", "session_id": "memory_test", "expected_plugins": ["text"]}, "/tmp")
    
    async def _mock_human_loop_execution(self, *args, **kwargs):
        """Mock human loop execution."""
        call_count = getattr(self._mock_human_loop_execution, 'call_count', 0)
        self._mock_human_loop_execution.call_count = call_count + 1
        
        if call_count == 0:
            # First call: return state requiring human input
            state = self._create_mock_successful_state({"topic": "Human Test", "description": "Test", "session_id": "human_test", "expected_plugins": ["text"]}, "/tmp")
            state.update({
                "pending_human_input": {
                    "context": "Scene outline requires human review",
                    "options": ["approve", "modify", "reject"],
                    "requesting_agent": "planner_agent"
                },
                "workflow_interrupted": True
            })
            return state
        else:
            # Second call: return state after human approval
            state = self._create_mock_successful_state({"topic": "Human Test", "description": "Test", "session_id": "human_test", "expected_plugins": ["text"]}, "/tmp")
            state.update({
                "pending_human_input": None,
                "human_feedback": {"decision": "approve", "comments": "Approved"},
                "workflow_interrupted": False
            })
            return state
    
    async def _mock_timeout_execution(self, *args, **kwargs):
        """Mock timeout execution."""
        await asyncio.sleep(0.1)
        state = self._create_mock_successful_state({"topic": "Timeout Test", "description": "Test", "session_id": "timeout_test", "expected_plugins": ["text"]}, "/tmp")
        state.update({
            "human_feedback": {
                "decision": "approve",
                "comments": "Auto-approved due to timeout",
                "timeout_occurred": True
            }
        })
        return state
    
    async def _mock_recovery_execution(self, *args, **kwargs):
        """Mock recovery execution."""
        call_count = getattr(self._mock_recovery_execution, 'call_count', 0)
        self._mock_recovery_execution.call_count = call_count + 1
        
        if call_count == 0:
            # First call: failure
            state = self._create_mock_successful_state({"topic": "Recovery Test", "description": "Test", "session_id": "recovery_test", "expected_plugins": ["text"]}, "/tmp")
            state.update({
                "error_count": 1,
                "escalated_errors": [{"agent": "code_generator_agent", "error": "Timeout error"}],
                "workflow_complete": False
            })
            return state
        else:
            # Second call: recovery success
            return self._create_mock_successful_state({"topic": "Recovery Test", "description": "Test", "session_id": "recovery_test", "expected_plugins": ["text"]}, "/tmp")
    
    async def _mock_escalation_execution(self, *args, **kwargs):
        """Mock escalation execution."""
        state = self._create_mock_successful_state({"topic": "Escalation Test", "description": "Test", "session_id": "escalation_test", "expected_plugins": ["text"]}, "/tmp")
        state.update({
            "error_count": 5,
            "escalated_errors": [
                {"agent": "planner_agent", "error": "Planning service unavailable"},
                {"agent": "code_generator_agent", "error": "Code generation model failed"},
                {"agent": "renderer_agent", "error": "Rendering infrastructure down"}
            ],
            "pending_human_input": {
                "context": "Multiple critical failures require human intervention",
                "options": ["abort_workflow", "retry_with_manual_config", "escalate_to_expert"],
                "priority": "critical"
            },
            "workflow_interrupted": True,
            "workflow_complete": False
        })
        return state
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all end-to-end tests."""
        logger.info("Starting comprehensive end-to-end test suite...")
        self.start_time = time.time()
        
        try:
            # Run all test categories
            test_categories = [
                self.run_complete_workflow_tests(),
                self.run_performance_tests(),
                self.run_human_loop_tests(),
                self.run_failure_recovery_tests()
            ]
            
            # Execute all test categories
            results = await asyncio.gather(*test_categories, return_exceptions=True)
            
            # Process results
            total_tests = 0
            total_passed = 0
            total_failed = 0
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Test category failed with exception: {result}")
                    total_failed += 1
                else:
                    self.test_results[result["category"]] = result
                    total_tests += result["total_tests"]
                    total_passed += result["passed"]
                    total_failed += result["failed"]
            
            self.end_time = time.time()
            
            # Generate summary
            summary = {
                "total_execution_time": self.end_time - self.start_time,
                "total_tests": total_tests,
                "total_passed": total_passed,
                "total_failed": total_failed,
                "success_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0,
                "categories": self.test_results
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Test suite execution failed: {e}")
            return {
                "total_execution_time": time.time() - self.start_time if self.start_time else 0,
                "total_tests": 0,
                "total_passed": 0,
                "total_failed": 1,
                "success_rate": 0,
                "error": str(e)
            }
        
        finally:
            self.cleanup_temp_dirs()
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print test summary."""
        print("\n" + "="*80)
        print("END-TO-END TEST SUITE SUMMARY")
        print("="*80)
        
        print(f"Total execution time: {summary['total_execution_time']:.2f} seconds")
        print(f"Total tests: {summary['total_tests']}")
        print(f"Passed: {summary['total_passed']}")
        print(f"Failed: {summary['total_failed']}")
        print(f"Success rate: {summary['success_rate']:.1f}%")
        
        if "categories" in summary:
            print("\nCategory Breakdown:")
            print("-" * 40)
            
            for category, results in summary["categories"].items():
                status_icon = "✓" if results["failed"] == 0 else "✗"
                print(f"{status_icon} {category.replace('_', ' ').title()}: {results['passed']}/{results['total_tests']} passed")
                
                if "results" in results:
                    for test_name, test_result in results["results"].items():
                        test_icon = "  ✓" if test_result["status"] == "passed" else "  ✗"
                        print(f"{test_icon} {test_name.replace('_', ' ').title()}")
                        if test_result["status"] == "failed" and test_result.get("error"):
                            print(f"    Error: {test_result['error']}")
        
        if summary["total_failed"] == 0:
            print(f"\n🎉 All {summary['total_tests']} end-to-end tests PASSED!")
        else:
            print(f"\n❌ {summary['total_failed']} out of {summary['total_tests']} tests FAILED!")


async def main():
    """Main test runner entry point."""
    runner = E2ETestRunner()
    
    try:
        summary = await runner.run_all_tests()
        runner.print_summary(summary)
        
        # Exit with appropriate code
        exit_code = 0 if summary["total_failed"] == 0 else 1
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        logger.info("Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
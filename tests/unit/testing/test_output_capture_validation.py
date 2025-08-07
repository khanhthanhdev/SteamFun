"""
Tests for agent output capture and validation system.

This module tests the comprehensive output capture, formatting, and validation
capabilities for agent testing in Studio environment.
"""

import pytest
import asyncio
import time
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from src.langgraph_agents.testing.output_capture import (
    AgentOutputCapture,
    OutputFormatter,
    OutputValidator,
    CapturedOutput,
    LogCapture
)
from src.langgraph_agents.testing.performance_metrics import (
    PerformanceMonitor,
    PerformanceAnalyzer,
    AgentPerformanceMetrics
)
from src.langgraph_agents.models.config import WorkflowConfig, ModelConfig


class TestAgentOutputCapture:
    """Test the AgentOutputCapture system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.output_capture = AgentOutputCapture()
        self.session_id = "test_session_123"
        self.agent_name = "TestAgent"
    
    def test_start_capture(self):
        """Test starting output capture."""
        self.output_capture.start_capture(self.session_id, self.agent_name)
        
        assert self.session_id in self.output_capture.active_captures
        assert self.output_capture.is_capturing(self.session_id)
        
        captured = self.output_capture.active_captures[self.session_id]
        assert captured.session_id == self.session_id
        assert captured.agent_name == self.agent_name
        assert captured.start_time is not None
        assert captured.end_time is None
    
    def test_stop_capture(self):
        """Test stopping output capture."""
        self.output_capture.start_capture(self.session_id, self.agent_name)
        
        # Add some test data
        self.output_capture.add_metrics(self.session_id, {"test_metric": 42})
        self.output_capture.add_results(self.session_id, {"test_result": "success"})
        
        captured_output = self.output_capture.stop_capture(self.session_id)
        
        assert captured_output is not None
        assert captured_output.session_id == self.session_id
        assert captured_output.agent_name == self.agent_name
        assert captured_output.end_time is not None
        assert captured_output.execution_time is not None
        assert captured_output.metrics["test_metric"] == 42
        assert captured_output.results["test_result"] == "success"
        
        # Should no longer be capturing
        assert not self.output_capture.is_capturing(self.session_id)
    
    def test_add_state_tracking(self):
        """Test adding state tracking information."""
        self.output_capture.start_capture(self.session_id, self.agent_name)
        
        state_data = {"step": "initialization", "status": "complete"}
        self.output_capture.add_state_tracking(self.session_id, "init_step", state_data)
        
        captured_output = self.output_capture.stop_capture(self.session_id)
        
        assert "state_tracking" in captured_output.results
        state_tracking = captured_output.results["state_tracking"]
        assert len(state_tracking) == 1
        assert state_tracking[0]["step"] == "init_step"
        assert state_tracking[0]["state_data"] == state_data
        assert "timestamp" in state_tracking[0]
    
    def test_add_execution_log(self):
        """Test adding execution log entries."""
        self.output_capture.start_capture(self.session_id, self.agent_name)
        
        context = {"operation": "test", "duration": 1.5}
        self.output_capture.add_execution_log(self.session_id, "INFO", "Test operation completed", context)
        
        captured_output = self.output_capture.stop_capture(self.session_id)
        
        assert "execution_logs" in captured_output.results
        execution_logs = captured_output.results["execution_logs"]
        assert len(execution_logs) == 1
        assert execution_logs[0]["level"] == "INFO"
        assert execution_logs[0]["message"] == "Test operation completed"
        assert execution_logs[0]["context"] == context
        assert "timestamp" in execution_logs[0]
    
    def test_capture_streams_context_manager(self):
        """Test the capture_streams context manager."""
        self.output_capture.start_capture(self.session_id, self.agent_name)
        
        with self.output_capture.capture_streams(self.session_id):
            print("Test stdout message")
            import sys
            print("Test stderr message", file=sys.stderr)
        
        captured_output = self.output_capture.stop_capture(self.session_id)
        
        assert "Test stdout message" in captured_output.stdout
        assert "Test stderr message" in captured_output.stderr


class TestOutputFormatter:
    """Test the OutputFormatter system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = OutputFormatter()
        self.captured_output = CapturedOutput(
            session_id="test_session",
            agent_name="TestAgent",
            start_time=datetime.now(),
            end_time=None,
            stdout=["Test stdout line 1", "Test stdout line 2"],
            stderr=["Test stderr line 1"],
            logs=[
                {
                    "timestamp": datetime.now().isoformat(),
                    "level": "INFO",
                    "logger": "test.logger",
                    "message": "Test log message"
                }
            ],
            metrics={"execution_time": 1.5, "api_calls": 3},
            errors=[{"error_type": "TestError", "message": "Test error"}],
            results={"output": "test_result", "success": True},
            execution_time=1.5
        )
    
    def test_format_for_studio(self):
        """Test formatting output for Studio visualization."""
        formatted = self.formatter.format_for_studio(self.captured_output)
        
        assert "session_info" in formatted
        assert "console_output" in formatted
        assert "logs" in formatted
        assert "metrics" in formatted
        assert "errors" in formatted
        assert "results" in formatted
        
        # Check session info
        session_info = formatted["session_info"]
        assert session_info["session_id"] == "test_session"
        assert session_info["agent_name"] == "TestAgent"
        assert session_info["execution_time"] == 1.5
        
        # Check console output
        console_output = formatted["console_output"]
        assert console_output["stdout"] == ["Test stdout line 1", "Test stdout line 2"]
        assert console_output["stderr"] == ["Test stderr line 1"]
        assert console_output["stdout_summary"] == "2 lines"
        assert console_output["stderr_summary"] == "1 lines"
    
    def test_format_for_json(self):
        """Test formatting output as JSON."""
        json_output = self.formatter.format_for_json(self.captured_output)
        
        assert isinstance(json_output, str)
        assert "test_session" in json_output
        assert "TestAgent" in json_output
        assert "execution_time" in json_output
    
    def test_format_for_text(self):
        """Test formatting output as human-readable text."""
        text_output = self.formatter.format_for_text(self.captured_output)
        
        assert isinstance(text_output, str)
        assert "Agent Output Capture: TestAgent" in text_output
        assert "Session ID: test_session" in text_output
        assert "Execution Time: 1.50s" in text_output
        assert "Test stdout line 1" in text_output
        assert "Test stderr line 1" in text_output


class TestOutputValidator:
    """Test the OutputValidator system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = OutputValidator()
    
    def create_test_captured_output(self, agent_name: str, results: dict = None, execution_time: float = 1.0):
        """Create a test CapturedOutput instance."""
        return CapturedOutput(
            session_id="test_session",
            agent_name=agent_name,
            start_time=datetime.now(),
            end_time=datetime.now(),
            stdout=[],
            stderr=[],
            logs=[],
            metrics={},
            errors=[],
            results=results or {},
            execution_time=execution_time
        )
    
    def test_validate_planner_output(self):
        """Test validation of PlannerAgent output."""
        results = {
            'outputs': {
                'scene_outline': 'Test outline',
                'scene_implementations': {'1': 'Scene 1 implementation'},
                'detected_plugins': ['plugin1'],
                'scene_count': 1
            },
            'validation': {
                'scene_outline_valid': True,
                'scene_implementations_valid': True
            }
        }
        
        captured_output = self.create_test_captured_output("PlannerAgent", results)
        validation_result = self.validator.validate_output(captured_output)
        
        assert validation_result['has_scene_outline'] is True
        assert validation_result['has_scene_implementations'] is True
        assert validation_result['has_detected_plugins'] is True
        assert validation_result['reasonable_scene_count'] is True
        assert validation_result['scene_count'] == 1
        assert validation_result['scene_outline_valid'] is True
        assert validation_result['scene_implementations_valid'] is True
    
    def test_validate_codegen_output(self):
        """Test validation of CodeGeneratorAgent output."""
        results = {
            'outputs': {
                'generated_code': {'1': 'test code'},
                'successful_scenes': [1],
                'failed_scenes': [],
                'total_scenes': 1
            },
            'validation': {
                '1': {'valid': True, 'issues': []}
            }
        }
        
        captured_output = self.create_test_captured_output("CodeGeneratorAgent", results)
        validation_result = self.validator.validate_output(captured_output)
        
        assert validation_result['has_generated_code'] is True
        assert validation_result['has_successful_scenes'] is True
        assert validation_result['has_failed_scenes'] is True
        assert validation_result['high_success_rate'] is True
        assert validation_result['success_rate'] == 1.0
        assert validation_result['code_validation_rate'] == 1.0
    
    def test_validate_renderer_output(self):
        """Test validation of RendererAgent output."""
        results = {
            'outputs': {
                'rendered_videos': {'1': '/path/to/video1.mp4'},
                'final_codes': {'1': 'final code'},
                'successful_scenes': [1],
                'total_scenes': 1,
                'combined_video_path': '/path/to/combined.mp4'
            }
        }
        
        captured_output = self.create_test_captured_output("RendererAgent", results)
        validation_result = self.validator.validate_output(captured_output)
        
        assert validation_result['has_rendered_videos'] is True
        assert validation_result['has_final_codes'] is True
        assert validation_result['high_rendering_success_rate'] is True
        assert validation_result['rendering_success_rate'] == 1.0
        assert validation_result['has_combined_video'] is True
    
    def test_validate_common_output(self):
        """Test common output validation."""
        # Test with reasonable execution time
        captured_output = self.create_test_captured_output("TestAgent", {}, 10.0)
        validation_result = self.validator.validate_output(captured_output)
        
        assert validation_result['execution_time_reasonable'] is True
        assert validation_result['execution_time_value'] == 10.0
        assert validation_result['no_critical_errors'] is True
        assert validation_result['critical_error_count'] == 0
        
        # Test with excessive execution time
        captured_output = self.create_test_captured_output("TestAgent", {}, 400.0)
        validation_result = self.validator.validate_output(captured_output)
        
        assert validation_result['execution_time_reasonable'] is False
        assert validation_result['execution_time_value'] == 400.0


class TestPerformanceMonitor:
    """Test the PerformanceMonitor system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = PerformanceMonitor(sampling_interval=0.1)  # Fast sampling for tests
        self.session_id = "test_session"
        self.agent_name = "TestAgent"
    
    def test_start_monitoring(self):
        """Test starting performance monitoring."""
        metrics = self.monitor.start_monitoring(self.session_id, self.agent_name)
        
        assert isinstance(metrics, AgentPerformanceMetrics)
        assert metrics.session_id == self.session_id
        assert metrics.agent_name == self.agent_name
        assert metrics.start_time is not None
        assert self.monitor.is_monitoring(self.session_id)
        
        # Cleanup
        self.monitor.stop_monitoring(self.session_id)
    
    def test_stop_monitoring(self):
        """Test stopping performance monitoring."""
        self.monitor.start_monitoring(self.session_id, self.agent_name)
        
        # Let it collect some samples
        time.sleep(0.2)
        
        metrics = self.monitor.stop_monitoring(self.session_id)
        
        assert metrics is not None
        assert metrics.end_time is not None
        assert metrics.execution_time is not None
        assert len(metrics.snapshots) > 0
        assert not self.monitor.is_monitoring(self.session_id)
    
    def test_add_custom_metrics(self):
        """Test adding custom metrics."""
        self.monitor.start_monitoring(self.session_id, self.agent_name)
        
        self.monitor.add_metric(self.session_id, "test_metric", 42)
        self.monitor.increment_counter(self.session_id, "api_calls_made", 3)
        
        metrics = self.monitor.stop_monitoring(self.session_id)
        
        assert metrics.custom_metrics["test_metric"] == 42
        assert metrics.api_calls_made == 3


class TestPerformanceAnalyzer:
    """Test the PerformanceAnalyzer system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = PerformanceAnalyzer()
    
    def create_test_metrics(self, agent_name: str, execution_time: float, api_calls: int = 1):
        """Create test performance metrics."""
        metrics = AgentPerformanceMetrics(
            agent_name=agent_name,
            session_id=f"test_{int(time.time())}",
            start_time=datetime.now(),
            execution_time=execution_time,
            api_calls_made=api_calls
        )
        metrics.finalize()
        return metrics
    
    def test_add_and_analyze_metrics(self):
        """Test adding metrics and analyzing performance."""
        # Add some test metrics
        metrics1 = self.create_test_metrics("TestAgent", 10.0, 5)
        metrics2 = self.create_test_metrics("TestAgent", 15.0, 7)
        metrics3 = self.create_test_metrics("TestAgent", 12.0, 6)
        
        self.analyzer.add_metrics(metrics1)
        self.analyzer.add_metrics(metrics2)
        self.analyzer.add_metrics(metrics3)
        
        analysis = self.analyzer.analyze_agent_performance("TestAgent")
        
        assert analysis['agent_name'] == "TestAgent"
        assert analysis['total_executions'] == 3
        assert 'execution_time_stats' in analysis
        assert analysis['execution_time_stats']['avg'] == 12.333333333333334  # (10+15+12)/3
        assert analysis['execution_time_stats']['min'] == 10.0
        assert analysis['execution_time_stats']['max'] == 15.0
    
    def test_compare_agents(self):
        """Test comparing performance across agents."""
        # Add metrics for different agents
        agent1_metrics = self.create_test_metrics("Agent1", 10.0, 5)
        agent2_metrics = self.create_test_metrics("Agent2", 20.0, 10)
        
        self.analyzer.add_metrics(agent1_metrics)
        self.analyzer.add_metrics(agent2_metrics)
        
        comparison = self.analyzer.compare_agents(["Agent1", "Agent2"])
        
        assert "Agent1" in comparison
        assert "Agent2" in comparison
        assert comparison["Agent1"]["avg_execution_time"] == 10.0
        assert comparison["Agent2"]["avg_execution_time"] == 20.0
        assert comparison["Agent1"]["avg_api_calls"] == 5
        assert comparison["Agent2"]["avg_api_calls"] == 10
    
    def test_identify_performance_issues(self):
        """Test identifying performance issues."""
        # Create metrics with issues
        slow_metrics = self.create_test_metrics("SlowAgent", 400.0, 1)  # Too slow
        slow_metrics.error_count = 5  # Too many errors
        slow_metrics.cache_hits = 1
        slow_metrics.cache_misses = 9  # Poor cache performance
        
        issues = self.analyzer.identify_performance_issues(slow_metrics)
        
        issue_types = [issue['type'] for issue in issues]
        assert 'slow_execution' in issue_types
        assert 'errors_detected' in issue_types
        assert 'low_cache_hit_rate' in issue_types


@pytest.mark.asyncio
async def test_integration_output_capture_with_agent_runner():
    """Integration test for output capture with agent test runner."""
    from src.langgraph_agents.testing.agent_test_runners import PlannerAgentTestRunner
    
    # Create mock config
    config = Mock(spec=WorkflowConfig)
    config.planner_model = Mock(spec=ModelConfig)
    config.planner_model.provider = "openai"
    config.planner_model.model_name = "gpt-4"
    config.planner_model.temperature = 0.7
    config.planner_model.max_tokens = 2000
    
    config.code_model = Mock(spec=ModelConfig)
    config.code_model.provider = "openai"
    config.code_model.model_name = "gpt-4"
    config.code_model.temperature = 0.3
    config.code_model.max_tokens = 4000
    
    config.helper_model = Mock(spec=ModelConfig)
    config.helper_model.provider = "openai"
    config.helper_model.model_name = "gpt-3.5-turbo"
    config.helper_model.temperature = 0.5
    config.helper_model.max_tokens = 1000
    
    config.output_dir = "/tmp/test"
    config.use_rag = False
    config.use_context_learning = False
    config.context_learning_path = ""
    config.chroma_db_path = ""
    config.manim_docs_path = ""
    config.embedding_model = "text-embedding-ada-002"
    config.use_langfuse = False
    config.max_concurrent_scenes = 3
    config.enable_caching = True
    config.use_enhanced_rag = False
    
    # Create test runner
    runner = PlannerAgentTestRunner(config)
    
    # Mock the planning service
    with patch.object(runner, 'planning_service') as mock_service:
        mock_service.generate_scene_outline = AsyncMock(return_value="Test outline")
        mock_service.validate_scene_outline = AsyncMock(return_value=(True, []))
        mock_service.generate_scene_implementations = AsyncMock(return_value={1: "Scene 1 implementation"})
        mock_service.validate_scene_implementations = AsyncMock(return_value=(True, []))
        mock_service.detect_plugins = AsyncMock(return_value=["plugin1"])
        mock_service.get_planning_metrics = Mock(return_value={
            'outline_generation_time': 2.0,
            'implementation_generation_time': 3.0,
            'plugin_detection_time': 1.0,
            'total_tokens_used': 1500,
            'api_calls_made': 3
        })
        mock_service.cleanup = AsyncMock()
        
        # Run test
        inputs = {
            'topic': 'Test Topic',
            'description': 'Test Description',
            'session_id': 'integration_test'
        }
        
        result = await runner.run_test(inputs)
        
        # Verify result structure
        assert result['success'] is True
        assert result['agent'] == 'PlannerAgent'
        assert 'execution_time' in result
        assert 'outputs' in result
        assert 'validation' in result
        assert 'metrics' in result
        assert 'captured_output' in result
        
        # Verify captured output structure
        captured_output = result['captured_output']
        assert captured_output.session_id == 'planner_integration_test'
        assert captured_output.agent_name == 'PlannerAgent'
        assert captured_output.execution_time is not None
        
        # Verify state tracking was captured
        assert 'state_tracking' in captured_output.results
        state_tracking = captured_output.results['state_tracking']
        assert len(state_tracking) >= 6  # Should have start/complete for each phase
        
        # Verify performance metrics were captured
        assert 'performance_metrics' in captured_output.results
        performance_metrics = captured_output.results['performance_metrics']
        assert performance_metrics['agent_name'] == 'PlannerAgent'
        assert 'execution_time' in performance_metrics


if __name__ == "__main__":
    pytest.main([__file__])
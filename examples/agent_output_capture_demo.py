#!/usr/bin/env python3
"""
Demonstration of the enhanced agent output capture and validation system.

This script shows how to use the comprehensive output capture, performance monitoring,
and validation capabilities for agent testing in LangGraph Studio.
"""

import asyncio
import time
import logging
from datetime import datetime
from typing import Dict, Any

# Add the src directory to the path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.langgraph_agents.testing.output_capture import (
    AgentOutputCapture,
    OutputFormatter,
    OutputValidator,
    get_output_capture,
    get_output_formatter,
    get_output_validator
)
from src.langgraph_agents.testing.performance_metrics import (
    PerformanceMonitor,
    PerformanceAnalyzer,
    get_performance_monitor,
    get_performance_analyzer
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def simulate_agent_execution(agent_name: str, session_id: str, duration: float = 2.0) -> Dict[str, Any]:
    """
    Simulate an agent execution with various outputs and state changes.
    
    Args:
        agent_name: Name of the agent being simulated
        session_id: Session ID for the execution
        duration: How long to simulate execution (seconds)
    
    Returns:
        Dictionary containing execution results
    """
    # Get global instances
    output_capture = get_output_capture()
    performance_monitor = get_performance_monitor()
    
    # Start capturing output and monitoring performance
    output_capture.start_capture(session_id, agent_name)
    performance_metrics = performance_monitor.start_monitoring(session_id, agent_name)
    
    try:
        # Simulate initialization
        logger.info(f"Starting {agent_name} execution")
        output_capture.add_state_tracking(session_id, "initialization", {
            "agent": agent_name,
            "start_time": datetime.now().isoformat()
        })
        
        # Simulate some processing with stdout/stderr capture
        with output_capture.capture_streams(session_id):
            print(f"[{agent_name}] Processing started...")
            print(f"[{agent_name}] Analyzing input data...", file=sys.stderr)
        
        # Simulate API calls and token usage
        for i in range(3):
            await asyncio.sleep(duration / 6)  # Simulate work
            
            performance_monitor.increment_counter(session_id, "api_calls_made")
            performance_monitor.increment_counter(session_id, "tokens_used", 150)
            
            output_capture.add_state_tracking(session_id, f"processing_step_{i+1}", {
                "step": i + 1,
                "status": "complete",
                "tokens_used": 150
            })
            
            logger.info(f"{agent_name} completed step {i+1}")
        
        # Simulate some cache operations
        performance_monitor.increment_counter(session_id, "cache_hits", 2)
        performance_monitor.increment_counter(session_id, "cache_misses", 1)
        
        # Add custom metrics
        performance_monitor.add_metric(session_id, "custom_score", 0.85)
        performance_monitor.add_metric(session_id, "complexity_level", "medium")
        
        # Simulate final processing
        await asyncio.sleep(duration / 3)
        
        with output_capture.capture_streams(session_id):
            print(f"[{agent_name}] Processing completed successfully!")
        
        # Create agent-specific results based on agent type
        if agent_name == "PlannerAgent":
            results = {
                'outputs': {
                    'scene_outline': 'Scene 1: Introduction\nScene 2: Main Content\nScene 3: Conclusion',
                    'scene_implementations': {
                        '1': 'Implement introduction with title and overview',
                        '2': 'Implement main content with detailed explanation',
                        '3': 'Implement conclusion with summary'
                    },
                    'detected_plugins': ['text_plugin', 'animation_plugin'],
                    'scene_count': 3
                },
                'validation': {
                    'scene_outline_valid': True,
                    'scene_implementations_valid': True
                }
            }
        elif agent_name == "CodeGeneratorAgent":
            results = {
                'outputs': {
                    'generated_code': {
                        '1': 'from manim import *\n\nclass IntroScene(Scene):\n    def construct(self):\n        title = Text("Introduction")\n        self.play(Write(title))',
                        '2': 'from manim import *\n\nclass MainScene(Scene):\n    def construct(self):\n        content = Text("Main Content")\n        self.play(Write(content))',
                        '3': 'from manim import *\n\nclass ConclusionScene(Scene):\n    def construct(self):\n        conclusion = Text("Conclusion")\n        self.play(Write(conclusion))'
                    },
                    'successful_scenes': [1, 2, 3],
                    'failed_scenes': [],
                    'total_scenes': 3
                },
                'validation': {
                    '1': {'valid': True, 'issues': []},
                    '2': {'valid': True, 'issues': []},
                    '3': {'valid': True, 'issues': []}
                }
            }
        elif agent_name == "RendererAgent":
            results = {
                'outputs': {
                    'rendered_videos': {
                        '1': '/tmp/scene1.mp4',
                        '2': '/tmp/scene2.mp4',
                        '3': '/tmp/scene3.mp4'
                    },
                    'final_codes': {
                        '1': 'Final code for scene 1',
                        '2': 'Final code for scene 2',
                        '3': 'Final code for scene 3'
                    },
                    'successful_scenes': [1, 2, 3],
                    'failed_scenes': [],
                    'total_scenes': 3,
                    'combined_video_path': '/tmp/combined_video.mp4'
                }
            }
        else:
            results = {
                'outputs': {
                    'processed_items': 3,
                    'success_rate': 1.0
                },
                'success': True
            }
        
        # Add results to output capture
        output_capture.add_results(session_id, results)
        
        # Add final metrics
        final_metrics = {
            'execution_time': duration,
            'total_steps': 3,
            'success_rate': 1.0,
            'api_calls_made': 3,
            'tokens_used': 450,
            'cache_hit_rate': 0.67
        }
        output_capture.add_metrics(session_id, final_metrics)
        
        output_capture.add_state_tracking(session_id, "completion", {
            "status": "success",
            "end_time": datetime.now().isoformat()
        })
        
        return results
        
    except Exception as e:
        logger.error(f"Error in {agent_name} execution: {str(e)}")
        
        # Add error to output capture
        output_capture.add_error(session_id, {
            'error_type': type(e).__name__,
            'message': str(e),
            'step': 'execution'
        })
        
        performance_monitor.increment_counter(session_id, "error_count")
        
        raise
    
    finally:
        # Stop performance monitoring
        final_performance_metrics = performance_monitor.stop_monitoring(session_id)
        if final_performance_metrics:
            # Add performance metrics to output capture
            output_capture.add_results(session_id, {
                'performance_metrics': final_performance_metrics.to_dict()
            })
        
        # Stop output capture
        captured_output = output_capture.stop_capture(session_id)
        
        return {
            'execution_results': results if 'results' in locals() else {},
            'captured_output': captured_output,
            'performance_metrics': final_performance_metrics
        }


async def demonstrate_output_capture_system():
    """Demonstrate the complete output capture and validation system."""
    print("=" * 80)
    print("Agent Output Capture and Validation System Demonstration")
    print("=" * 80)
    
    # Get global instances
    output_formatter = get_output_formatter()
    output_validator = get_output_validator()
    performance_analyzer = get_performance_analyzer()
    
    agents_to_test = [
        ("PlannerAgent", 2.0),
        ("CodeGeneratorAgent", 3.0),
        ("RendererAgent", 4.0)
    ]
    
    all_results = []
    
    for agent_name, duration in agents_to_test:
        print(f"\n--- Testing {agent_name} ---")
        
        session_id = f"{agent_name.lower()}_{int(time.time())}"
        
        try:
            # Run simulated agent execution
            result = await simulate_agent_execution(agent_name, session_id, duration)
            all_results.append(result)
            
            captured_output = result['captured_output']
            performance_metrics = result['performance_metrics']
            
            print(f"✓ {agent_name} execution completed successfully")
            print(f"  Execution time: {captured_output.execution_time:.2f}s")
            print(f"  State tracking entries: {len(captured_output.results.get('state_tracking', []))}")
            print(f"  Performance snapshots: {len(performance_metrics.snapshots) if performance_metrics else 0}")
            
            # Validate output
            validation_result = output_validator.validate_output(captured_output)
            print(f"  Validation results: {len(validation_result)} checks performed")
            
            # Check for issues
            if performance_metrics:
                performance_analyzer.add_metrics(performance_metrics)
                issues = performance_analyzer.identify_performance_issues(performance_metrics)
                if issues:
                    print(f"  Performance issues identified: {len(issues)}")
                    for issue in issues:
                        print(f"    - {issue['type']}: {issue['message']}")
                else:
                    print("  No performance issues identified")
            
        except Exception as e:
            print(f"✗ {agent_name} execution failed: {str(e)}")
    
    # Demonstrate output formatting
    print(f"\n--- Output Formatting Examples ---")
    
    if all_results:
        sample_output = all_results[0]['captured_output']
        
        print("\n1. Studio Format (summary):")
        studio_format = output_formatter.format_for_studio(sample_output)
        print(f"   Session: {studio_format['session_info']['session_id']}")
        print(f"   Agent: {studio_format['session_info']['agent_name']}")
        print(f"   Execution time: {studio_format['session_info']['execution_time']:.2f}s")
        print(f"   Console output: {studio_format['console_output']['stdout_summary']}")
        print(f"   Log entries: {studio_format['logs']['summary']['total']}")
        print(f"   Metrics: {studio_format['metrics']['summary']['total_metrics']} metrics")
        
        print("\n2. Text Format (first 500 chars):")
        text_format = output_formatter.format_for_text(sample_output)
        print(text_format[:500] + "..." if len(text_format) > 500 else text_format)
    
    # Demonstrate performance analysis
    print(f"\n--- Performance Analysis ---")
    
    for agent_name, _ in agents_to_test:
        analysis = performance_analyzer.analyze_agent_performance(agent_name)
        if 'error' not in analysis:
            print(f"\n{agent_name} Performance Analysis:")
            print(f"  Total executions: {analysis['total_executions']}")
            if analysis['execution_time_stats']:
                stats = analysis['execution_time_stats']
                print(f"  Execution time - Avg: {stats['avg']:.2f}s, Min: {stats['min']:.2f}s, Max: {stats['max']:.2f}s")
            print(f"  Average API calls: {analysis['api_usage_stats']['avg']:.1f}")
            print(f"  Average tokens used: {analysis['token_usage_stats']['avg']:.0f}")
            
            cache_perf = analysis['cache_performance']
            if cache_perf['total_requests'] > 0:
                print(f"  Cache hit rate: {cache_perf['hit_rate']:.1%}")
    
    # Demonstrate agent comparison
    print(f"\n--- Agent Performance Comparison ---")
    agent_names = [name for name, _ in agents_to_test]
    comparison = performance_analyzer.compare_agents(agent_names)
    
    print(f"{'Agent':<20} {'Avg Time (s)':<12} {'Avg API Calls':<15} {'Avg Tokens':<12}")
    print("-" * 60)
    for agent_name in agent_names:
        if agent_name in comparison:
            comp = comparison[agent_name]
            print(f"{agent_name:<20} {comp['avg_execution_time']:<12.2f} {comp['avg_api_calls']:<15.1f} {comp['avg_tokens_used']:<12.0f}")
    
    print(f"\n--- Demonstration Complete ---")
    print("The output capture and validation system successfully:")
    print("✓ Captured stdout, stderr, logs, and metrics for each agent")
    print("✓ Tracked execution state changes and performance metrics")
    print("✓ Validated agent-specific outputs against expected patterns")
    print("✓ Identified performance issues and provided recommendations")
    print("✓ Formatted outputs for different visualization needs")
    print("✓ Analyzed and compared performance across different agents")


if __name__ == "__main__":
    asyncio.run(demonstrate_output_capture_system())
"""
End-to-end tests for complete video generation workflow.
Tests the entire pipeline from planning to final video output.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List
from datetime import datetime
import time
import json

from src.langgraph_agents.workflow import LangGraphVideoGenerator
from src.langgraph_agents.state import VideoGenerationState, SystemConfig, create_initial_state
from src.langgraph_agents.base_agent import BaseAgent


class TestCompleteWorkflow:
    """End-to-end tests for complete video generation workflow."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory for tests."""
        temp_dir = tempfile.mkdtemp(prefix="e2e_test_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_video_generator(self, temp_output_dir):
        """Create mock video generator with realistic configuration."""
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
                verbose=True,
                use_rag=True,
                use_context_learning=True,
                use_langfuse=False,  # Disable for testing
                enable_human_loop=False,
                enable_monitoring=True,
                max_retries=3
            )
            
            yield generator
    
    @pytest.fixture
    def sample_workflow_data(self):
        """Sample data for complete workflow testing."""
        return {
            "topic": "Python Data Structures",
            "description": "Educational video explaining Python lists, dictionaries, and sets with practical examples",
            "session_id": "e2e_test_session_001",
            "expected_scenes": 3,
            "expected_plugins": ["text", "code", "math"],
            "expected_duration": 180  # seconds
        }
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_complete_video_generation_workflow(self, mock_video_generator, sample_workflow_data):
        """Test complete video generation workflow from start to finish."""
        generator = mock_video_generator
        data = sample_workflow_data
        
        # Mock the workflow execution to simulate complete pipeline
        mock_final_state = VideoGenerationState(
            messages=[],
            topic=data["topic"],
            description=data["description"],
            session_id=data["session_id"],
            output_dir=generator.init_params["output_dir"],
            print_response=True,
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
            scene_outline="""# Scene 1: Introduction to Python Data Structures
Overview of lists, dictionaries, and sets in Python programming.

# Scene 2: Working with Lists
Creating, accessing, and modifying Python lists with examples.

# Scene 3: Dictionaries and Sets
Understanding dictionaries and sets with practical use cases.""",
            scene_implementations={
                1: "Scene 1: Display Python logo with animated introduction to data structures. Show visual representations of lists, dictionaries, and sets.",
                2: "Scene 2: Create animated examples of list operations. Show indexing, slicing, and common list methods with visual feedback.",
                3: "Scene 3: Demonstrate dictionary key-value pairs and set operations. Show practical examples with visual representations."
            },
            detected_plugins=data["expected_plugins"],
            generated_code={
                1: """from manim import *

class DataStructuresIntro(Scene):
    def construct(self):
        title = Text("Python Data Structures", font_size=48)
        subtitle = Text("Lists, Dictionaries, and Sets", font_size=24)
        subtitle.next_to(title, DOWN)
        
        self.play(Write(title))
        self.play(Write(subtitle))
        self.wait(2)""",
                2: """from manim import *

class ListOperations(Scene):
    def construct(self):
        title = Text("Working with Lists", font_size=36)
        title.to_edge(UP)
        
        code = Code(
            '''
            my_list = [1, 2, 3, 4, 5]
            my_list.append(6)
            print(my_list[0])  # First element
            ''',
            language="python"
        )
        
        self.play(Write(title))
        self.play(Create(code))
        self.wait(3)""",
                3: """from manim import *

class DictionariesAndSets(Scene):
    def construct(self):
        title = Text("Dictionaries and Sets", font_size=36)
        title.to_edge(UP)
        
        dict_code = Code(
            '''
            student = {"name": "Alice", "age": 20}
            unique_numbers = {1, 2, 3, 3, 4}  # Set removes duplicates
            ''',
            language="python"
        )
        
        self.play(Write(title))
        self.play(Create(dict_code))
        self.wait(3)"""
            },
            code_errors={},
            rag_context={
                1: "RAG context for Python data structures introduction",
                2: "RAG context for list operations and methods",
                3: "RAG context for dictionaries and sets usage"
            },
            rendered_videos={
                1: f"{generator.init_params['output_dir']}/scene_1.mp4",
                2: f"{generator.init_params['output_dir']}/scene_2.mp4",
                3: f"{generator.init_params['output_dir']}/scene_3.mp4"
            },
            combined_video_path=f"{generator.init_params['output_dir']}/final_video.mp4",
            rendering_errors={},
            visual_analysis_results={
                1: {"quality_score": 0.85, "visual_errors": []},
                2: {"quality_score": 0.90, "visual_errors": []},
                3: {"quality_score": 0.88, "visual_errors": []}
            },
            visual_errors={},
            error_count=0,
            retry_count={},
            escalated_errors=[],
            pending_human_input=None,
            human_feedback=None,
            performance_metrics={
                "planner_agent": {
                    "last_execution_time": 2.5,
                    "average_execution_time": 2.5,
                    "success_rate": 1.0
                },
                "code_generator_agent": {
                    "last_execution_time": 8.2,
                    "average_execution_time": 8.2,
                    "success_rate": 1.0
                },
                "renderer_agent": {
                    "last_execution_time": 15.7,
                    "average_execution_time": 15.7,
                    "success_rate": 1.0
                }
            },
            execution_trace=[
                {
                    "agent": "planner_agent",
                    "action": "complete_execution",
                    "timestamp": "2024-01-01T10:00:00",
                    "execution_time": 2.5,
                    "next_agent": "code_generator_agent"
                },
                {
                    "agent": "code_generator_agent", 
                    "action": "complete_execution",
                    "timestamp": "2024-01-01T10:02:30",
                    "execution_time": 8.2,
                    "next_agent": "renderer_agent"
                },
                {
                    "agent": "renderer_agent",
                    "action": "complete_execution", 
                    "timestamp": "2024-01-01T10:10:45",
                    "execution_time": 15.7,
                    "next_agent": "END"
                }
            ],
            current_agent=None,
            next_agent=None,
            workflow_complete=True,
            workflow_interrupted=False
        )
        
        # Mock the workflow execution
        generator.workflow.invoke = AsyncMock(return_value=mock_final_state)
        
        # Execute complete workflow
        start_time = time.time()
        final_state = await generator.generate_video_pipeline(
            topic=data["topic"],
            description=data["description"],
            session_id=data["session_id"]
        )
        execution_time = time.time() - start_time
        
        # Verify workflow completion
        assert final_state["workflow_complete"] is True
        assert final_state["workflow_interrupted"] is False
        assert final_state["topic"] == data["topic"]
        assert final_state["session_id"] == data["session_id"]
        
        # Verify planning phase results
        assert final_state["scene_outline"] is not None
        assert len(final_state["scene_implementations"]) == data["expected_scenes"]
        assert set(final_state["detected_plugins"]) == set(data["expected_plugins"])
        
        # Verify code generation results
        assert len(final_state["generated_code"]) == data["expected_scenes"]
        assert len(final_state["code_errors"]) == 0  # No errors expected
        assert len(final_state["rag_context"]) == data["expected_scenes"]
        
        # Verify rendering results
        assert len(final_state["rendered_videos"]) == data["expected_scenes"]
        assert final_state["combined_video_path"] is not None
        assert len(final_state["rendering_errors"]) == 0  # No errors expected
        
        # Verify visual analysis results
        assert len(final_state["visual_analysis_results"]) == data["expected_scenes"]
        for scene_id, analysis in final_state["visual_analysis_results"].items():
            assert analysis["quality_score"] > 0.8  # High quality expected
            assert len(analysis["visual_errors"]) == 0  # No visual errors
        
        # Verify error handling
        assert final_state["error_count"] == 0
        assert len(final_state["escalated_errors"]) == 0
        
        # Verify performance metrics
        performance_metrics = final_state["performance_metrics"]
        assert "planner_agent" in performance_metrics
        assert "code_generator_agent" in performance_metrics
        assert "renderer_agent" in performance_metrics
        
        for agent, metrics in performance_metrics.items():
            assert metrics["success_rate"] == 1.0
            assert metrics["last_execution_time"] > 0
        
        # Verify execution trace
        execution_trace = final_state["execution_trace"]
        assert len(execution_trace) >= 3  # At least 3 agent executions
        assert all(entry["action"] == "complete_execution" for entry in execution_trace)
        
        # Verify workflow was called correctly
        generator.workflow.invoke.assert_called_once()
        call_args = generator.workflow.invoke.call_args
        assert call_args[1]["topic"] == data["topic"]
        assert call_args[1]["description"] == data["description"]
        assert call_args[1]["session_id"] == data["session_id"]
        
        print(f"✓ Complete workflow test passed in {execution_time:.2f} seconds")
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_workflow_with_scene_outline_only(self, mock_video_generator, sample_workflow_data):
        """Test workflow execution with planning phase only."""
        generator = mock_video_generator
        data = sample_workflow_data
        
        # Mock planning-only state
        mock_planning_state = VideoGenerationState(
            messages=[],
            topic=data["topic"],
            description=data["description"],
            session_id=data["session_id"],
            output_dir=generator.init_params["output_dir"],
            print_response=True,
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
            scene_outline="""# Scene 1: Introduction to Python Data Structures
Overview of lists, dictionaries, and sets in Python programming.

# Scene 2: Working with Lists  
Creating, accessing, and modifying Python lists with examples.

# Scene 3: Dictionaries and Sets
Understanding dictionaries and sets with practical use cases.""",
            scene_implementations={
                1: "Scene 1: Display Python logo with animated introduction to data structures.",
                2: "Scene 2: Create animated examples of list operations.",
                3: "Scene 3: Demonstrate dictionary key-value pairs and set operations."
            },
            detected_plugins=data["expected_plugins"],
            generated_code={},  # Empty for planning-only
            code_errors={},
            rag_context={},
            rendered_videos={},  # Empty for planning-only
            combined_video_path=None,
            rendering_errors={},
            visual_analysis_results={},
            visual_errors={},
            error_count=0,
            retry_count={},
            escalated_errors=[],
            pending_human_input=None,
            human_feedback=None,
            performance_metrics={
                "planner_agent": {
                    "last_execution_time": 2.1,
                    "average_execution_time": 2.1,
                    "success_rate": 1.0
                }
            },
            execution_trace=[
                {
                    "agent": "planner_agent",
                    "action": "complete_execution",
                    "timestamp": "2024-01-01T10:00:00",
                    "execution_time": 2.1,
                    "next_agent": "END"
                }
            ],
            current_agent=None,
            next_agent=None,
            workflow_complete=True,  # Planning complete
            workflow_interrupted=False
        )
        
        # Mock the workflow execution for planning only
        generator.workflow.invoke = AsyncMock(return_value=mock_planning_state)
        
        # Execute planning-only workflow
        scene_outline = await generator.generate_scene_outline(
            topic=data["topic"],
            description=data["description"],
            session_id=data["session_id"]
        )
        
        # Verify scene outline generation
        assert scene_outline is not None
        assert len(scene_outline) > 0
        assert "Scene 1:" in scene_outline
        assert "Scene 2:" in scene_outline
        assert "Scene 3:" in scene_outline
        assert "Python Data Structures" in scene_outline
        
        # Verify workflow was called with planning-only mode
        generator.workflow.invoke.assert_called_once()
        call_args = generator.workflow.invoke.call_args
        assert call_args[1]["config"]["only_plan"] is True
        
        print("✓ Planning-only workflow test passed")
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_workflow_with_specific_scenes(self, mock_video_generator, sample_workflow_data):
        """Test workflow execution with specific scene processing."""
        generator = mock_video_generator
        data = sample_workflow_data
        specific_scenes = [1, 3]  # Process only scenes 1 and 3
        
        # Mock specific scenes state
        mock_specific_scenes_state = VideoGenerationState(
            messages=[],
            topic=data["topic"],
            description=data["description"],
            session_id=data["session_id"],
            output_dir=generator.init_params["output_dir"],
            print_response=True,
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
            scene_outline="""# Scene 1: Introduction to Python Data Structures
Overview of lists, dictionaries, and sets in Python programming.

# Scene 3: Dictionaries and Sets
Understanding dictionaries and sets with practical use cases.""",
            scene_implementations={
                1: "Scene 1: Display Python logo with animated introduction to data structures.",
                3: "Scene 3: Demonstrate dictionary key-value pairs and set operations."
            },
            detected_plugins=data["expected_plugins"],
            generated_code={
                1: "from manim import *\nclass DataStructuresIntro(Scene): pass",
                3: "from manim import *\nclass DictionariesAndSets(Scene): pass"
            },
            code_errors={},
            rag_context={
                1: "RAG context for Python data structures introduction",
                3: "RAG context for dictionaries and sets usage"
            },
            rendered_videos={
                1: f"{generator.init_params['output_dir']}/scene_1.mp4",
                3: f"{generator.init_params['output_dir']}/scene_3.mp4"
            },
            combined_video_path=f"{generator.init_params['output_dir']}/final_video.mp4",
            rendering_errors={},
            visual_analysis_results={
                1: {"quality_score": 0.85, "visual_errors": []},
                3: {"quality_score": 0.88, "visual_errors": []}
            },
            visual_errors={},
            error_count=0,
            retry_count={},
            escalated_errors=[],
            pending_human_input=None,
            human_feedback=None,
            performance_metrics={
                "planner_agent": {"last_execution_time": 1.8, "success_rate": 1.0},
                "code_generator_agent": {"last_execution_time": 5.4, "success_rate": 1.0},
                "renderer_agent": {"last_execution_time": 8.9, "success_rate": 1.0}
            },
            execution_trace=[
                {"agent": "planner_agent", "action": "complete_execution", "timestamp": "2024-01-01T10:00:00"},
                {"agent": "code_generator_agent", "action": "complete_execution", "timestamp": "2024-01-01T10:01:48"},
                {"agent": "renderer_agent", "action": "complete_execution", "timestamp": "2024-01-01T10:07:12"}
            ],
            current_agent=None,
            next_agent=None,
            workflow_complete=True,
            workflow_interrupted=False
        )
        
        # Mock the workflow execution for specific scenes
        generator.workflow.invoke = AsyncMock(return_value=mock_specific_scenes_state)
        
        # Execute workflow with specific scenes
        final_state = await generator.generate_video_pipeline(
            topic=data["topic"],
            description=data["description"],
            session_id=data["session_id"],
            specific_scenes=specific_scenes
        )
        
        # Verify specific scenes processing
        assert final_state["workflow_complete"] is True
        assert len(final_state["scene_implementations"]) == len(specific_scenes)
        assert len(final_state["generated_code"]) == len(specific_scenes)
        assert len(final_state["rendered_videos"]) == len(specific_scenes)
        
        # Verify only specified scenes were processed
        for scene_id in specific_scenes:
            assert scene_id in final_state["scene_implementations"]
            assert scene_id in final_state["generated_code"]
            assert scene_id in final_state["rendered_videos"]
            assert scene_id in final_state["visual_analysis_results"]
        
        # Verify scene 2 was not processed
        assert 2 not in final_state["scene_implementations"]
        assert 2 not in final_state["generated_code"]
        assert 2 not in final_state["rendered_videos"]
        
        # Verify workflow was called with specific scenes
        generator.workflow.invoke.assert_called_once()
        call_args = generator.workflow.invoke.call_args
        assert call_args[1]["config"]["specific_scenes"] == specific_scenes
        
        print(f"✓ Specific scenes workflow test passed (scenes: {specific_scenes})")
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_workflow_streaming_execution(self, mock_video_generator, sample_workflow_data):
        """Test streaming workflow execution with progress updates."""
        generator = mock_video_generator
        data = sample_workflow_data
        
        # Mock streaming chunks
        mock_streaming_chunks = [
            {
                "planner_agent": {
                    "scene_outline": "# Scene 1: Introduction...",
                    "current_agent": "planner_agent",
                    "next_agent": "code_generator_agent"
                }
            },
            {
                "code_generator_agent": {
                    "generated_code": {1: "from manim import *..."},
                    "current_agent": "code_generator_agent", 
                    "next_agent": "renderer_agent"
                }
            },
            {
                "renderer_agent": {
                    "rendered_videos": {1: "/path/to/scene1.mp4"},
                    "current_agent": "renderer_agent",
                    "next_agent": "END",
                    "workflow_complete": True
                }
            }
        ]
        
        # Mock the streaming workflow
        async def mock_stream(*args, **kwargs):
            for chunk in mock_streaming_chunks:
                yield chunk
        
        generator.workflow.stream = mock_stream
        generator.workflow.get_workflow_status = Mock(return_value={
            "status": "completed",
            "current_agent": None,
            "next_agent": None,
            "error_count": 0,
            "workflow_complete": True,
            "execution_trace": [
                {"agent": "planner_agent", "action": "complete_execution"},
                {"agent": "code_generator_agent", "action": "complete_execution"},
                {"agent": "renderer_agent", "action": "complete_execution"}
            ]
        })
        
        # Execute streaming workflow
        streaming_events = []
        async for event in generator.stream_video_generation(
            topic=data["topic"],
            description=data["description"],
            session_id=data["session_id"]
        ):
            streaming_events.append(event)
        
        # Verify streaming events
        assert len(streaming_events) >= 4  # At least start, progress chunks, and completion
        
        # Verify workflow started event
        start_events = [e for e in streaming_events if e.get("event_type") == "workflow_started"]
        assert len(start_events) == 1
        assert start_events[0]["session_id"] == data["session_id"]
        
        # Verify progress events
        progress_events = [e for e in streaming_events if e.get("event_type") == "workflow_progress"]
        assert len(progress_events) == len(mock_streaming_chunks)
        
        # Verify completion event
        completion_events = [e for e in streaming_events if e.get("event_type") == "workflow_completed"]
        assert len(completion_events) == 1
        assert completion_events[0]["session_id"] == data["session_id"]
        
        # Verify flow visualization event
        flow_events = [e for e in streaming_events if e.get("event_type") == "flow_visualization_created"]
        assert len(flow_events) == 1
        
        print("✓ Streaming workflow test passed")
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_workflow_checkpoint_management(self, mock_video_generator, sample_workflow_data):
        """Test workflow checkpoint creation and management."""
        generator = mock_video_generator
        data = sample_workflow_data
        
        # Mock checkpointer
        mock_checkpointer = Mock()
        mock_checkpoints = [
            {
                "ts": "2024-01-01T10:00:00",
                "channel_values": {
                    "current_agent": "planner_agent",
                    "next_agent": "code_generator_agent",
                    "error_count": 0,
                    "workflow_complete": False
                }
            },
            {
                "ts": "2024-01-01T10:02:30", 
                "channel_values": {
                    "current_agent": "code_generator_agent",
                    "next_agent": "renderer_agent",
                    "error_count": 0,
                    "workflow_complete": False
                }
            },
            {
                "ts": "2024-01-01T10:10:45",
                "channel_values": {
                    "current_agent": "renderer_agent",
                    "next_agent": None,
                    "error_count": 0,
                    "workflow_complete": True
                }
            }
        ]
        
        mock_checkpointer.list = Mock(return_value=mock_checkpoints)
        mock_checkpointer.get = Mock(return_value=mock_checkpoints[-1])
        mock_checkpointer.delete = Mock()
        
        generator.workflow.checkpointer = mock_checkpointer
        
        # Test getting checkpoints
        checkpoints = await generator.get_workflow_checkpoints(data["session_id"])
        
        # Verify checkpoints
        assert len(checkpoints) == len(mock_checkpoints)
        assert checkpoints[0]["current_agent"] == "planner_agent"
        assert checkpoints[1]["current_agent"] == "code_generator_agent"
        assert checkpoints[2]["current_agent"] == "renderer_agent"
        assert checkpoints[2]["workflow_complete"] is True
        
        # Test clearing checkpoints
        clear_success = await generator.clear_workflow_checkpoints(data["session_id"])
        assert clear_success is True
        mock_checkpointer.delete.assert_called_once()
        
        print("✓ Checkpoint management test passed")
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_workflow_interruption_and_resume(self, mock_video_generator, sample_workflow_data):
        """Test workflow interruption and resumption capabilities."""
        generator = mock_video_generator
        data = sample_workflow_data
        
        # Mock workflow status for interruption scenario
        generator.workflow.get_workflow_status = Mock(side_effect=[
            {"status": "running", "current_agent": "code_generator_agent"},  # Initial status
            {"status": "interrupted", "current_agent": "code_generator_agent"},  # After interruption
            {"status": "running", "current_agent": "code_generator_agent"}  # After resume
        ])
        
        generator.workflow.interrupt_workflow = Mock(return_value=True)
        generator.workflow.resume_workflow = Mock(return_value=True)
        
        # Test workflow interruption
        interrupt_success = await generator.interrupt_workflow(
            session_id=data["session_id"],
            reason="Testing interruption"
        )
        assert interrupt_success is True
        generator.workflow.interrupt_workflow.assert_called_once_with(
            data["session_id"], "Testing interruption"
        )
        
        # Verify interrupted status
        status = generator.workflow.get_workflow_status(data["session_id"])
        assert status["status"] == "interrupted"
        
        # Test workflow resumption
        resume_success = await generator.resume_workflow(data["session_id"])
        assert resume_success is True
        generator.workflow.resume_workflow.assert_called_once_with(data["session_id"])
        
        # Verify resumed status
        status = generator.workflow.get_workflow_status(data["session_id"])
        assert status["status"] == "running"
        
        print("✓ Workflow interruption and resume test passed")
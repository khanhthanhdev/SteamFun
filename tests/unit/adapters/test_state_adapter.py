"""
Unit tests for StateAdapter.

Tests the conversion between old TypedDict and new Pydantic state formats.
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from src.langgraph_agents.adapters.state_adapter import StateAdapter
from src.langgraph_agents.models.state import VideoGenerationState as NewVideoGenerationState
from src.langgraph_agents.models.config import WorkflowConfig
from src.langgraph_agents.models.errors import WorkflowError
from src.langgraph_agents.models.metrics import PerformanceMetrics


class TestStateAdapter:
    """Test cases for StateAdapter functionality."""
    
    @pytest.fixture
    def sample_old_state(self) -> Dict[str, Any]:
        """Create a sample old state for testing."""
        return {
            # Core workflow data
            'messages': [],
            'topic': 'Python basics',
            'description': 'Introduction to Python programming',
            'session_id': 'test-session-123',
            
            # Configuration parameters
            'output_dir': 'output',
            'print_response': False,
            'use_rag': True,
            'use_context_learning': True,
            'context_learning_path': 'data/context_learning',
            'chroma_db_path': 'data/rag/chroma_db',
            'manim_docs_path': 'data/rag/manim_docs',
            'embedding_model': 'hf:ibm-granite/granite-embedding-30m-english',
            'use_visual_fix_code': False,
            'use_langfuse': True,
            'max_scene_concurrency': 5,
            'max_topic_concurrency': 1,
            'max_retries': 3,
            
            # Enhanced RAG Configuration
            'use_enhanced_rag': True,
            'enable_rag_caching': True,
            'enable_quality_monitoring': True,
            'enable_error_handling': True,
            'rag_cache_ttl': 3600,
            'rag_max_cache_size': 1000,
            'rag_performance_threshold': 2.0,
            'rag_quality_threshold': 0.7,
            
            # Renderer optimizations
            'enable_caching': True,
            'default_quality': 'medium',
            'use_gpu_acceleration': False,
            'preview_mode': False,
            'max_concurrent_renders': 4,
            
            # Planning state
            'scene_outline': 'Scene 1: Introduction\nScene 2: Variables',
            'scene_implementations': {1: 'Show title', 2: 'Explain variables'},
            'detected_plugins': ['manim_slides'],
            
            # Code generation state
            'generated_code': {1: 'class Scene1(Scene): pass', 2: 'class Scene2(Scene): pass'},
            'code_errors': {},
            'rag_context': {'query': 'python basics', 'results': []},
            
            # Rendering state
            'rendered_videos': {1: 'scene1.mp4', 2: 'scene2.mp4'},
            'combined_video_path': 'final_video.mp4',
            'rendering_errors': {},
            
            # Visual analysis state
            'visual_analysis_results': {},
            'visual_errors': {},
            
            # Error handling state
            'error_count': 0,
            'retry_count': {},
            'escalated_errors': [],
            
            # Human loop state
            'pending_human_input': None,
            'human_feedback': None,
            
            # Monitoring state
            'performance_metrics': {
                'planner_agent': {
                    'last_execution_time': 5.2,
                    'success_rate': 1.0
                }
            },
            'execution_trace': [
                {
                    'agent': 'planner_agent',
                    'action': 'start_execution',
                    'timestamp': '2024-01-01T10:00:00'
                }
            ],
            
            # Current agent tracking
            'current_agent': 'planner_agent',
            'next_agent': 'code_generator_agent',
            
            # Workflow control
            'workflow_complete': False,
            'workflow_interrupted': False
        }
    
    @pytest.fixture
    def sample_new_state(self) -> NewVideoGenerationState:
        """Create a sample new state for testing."""
        config = WorkflowConfig(
            use_rag=True,
            use_visual_analysis=False,
            enable_caching=True,
            max_retries=3,
            max_concurrent_scenes=5,
            output_dir='output'
        )
        
        return NewVideoGenerationState(
            topic='Python basics',
            description='Introduction to Python programming',
            session_id='test-session-123',
            config=config,
            scene_outline='Scene 1: Introduction\nScene 2: Variables',
            scene_implementations={1: 'Show title', 2: 'Explain variables'},
            detected_plugins=['manim_slides'],
            generated_code={1: 'class Scene1(Scene): pass', 2: 'class Scene2(Scene): pass'},
            rendered_videos={1: 'scene1.mp4', 2: 'scene2.mp4'},
            combined_video_path='final_video.mp4',
            current_step='planning',
            workflow_complete=False
        )
    
    def test_old_to_new_conversion_success(self, sample_old_state):
        """Test successful conversion from old to new state format."""
        # Convert old state to new format
        new_state = StateAdapter.old_to_new(sample_old_state)
        
        # Verify core fields
        assert new_state.topic == sample_old_state['topic']
        assert new_state.description == sample_old_state['description']
        assert new_state.session_id == sample_old_state['session_id']
        
        # Verify configuration mapping
        assert new_state.config.use_rag == sample_old_state['use_rag']
        assert new_state.config.enable_caching == sample_old_state['enable_caching']
        assert new_state.config.max_retries == sample_old_state['max_retries']
        assert new_state.config.output_dir == sample_old_state['output_dir']
        
        # Verify state data
        assert new_state.scene_outline == sample_old_state['scene_outline']
        assert new_state.scene_implementations == sample_old_state['scene_implementations']
        assert new_state.detected_plugins == sample_old_state['detected_plugins']
        assert new_state.generated_code == sample_old_state['generated_code']
        assert new_state.rendered_videos == sample_old_state['rendered_videos']
        
        # Verify workflow control
        assert new_state.workflow_complete == sample_old_state['workflow_complete']
        assert new_state.workflow_interrupted == sample_old_state['workflow_interrupted']
    
    def test_old_to_new_missing_required_fields(self):
        """Test conversion fails with missing required fields."""
        incomplete_state = {
            'topic': 'Test topic'
            # Missing description and session_id
        }
        
        with pytest.raises(ValueError, match="Description is required"):
            StateAdapter.old_to_new(incomplete_state)
    
    def test_old_to_new_empty_topic(self):
        """Test conversion fails with empty topic."""
        invalid_state = {
            'topic': '',
            'description': 'Test description',
            'session_id': 'test-123'
        }
        
        with pytest.raises(ValueError, match="Topic is required"):
            StateAdapter.old_to_new(invalid_state)
    
    def test_new_to_old_conversion_success(self, sample_new_state):
        """Test successful conversion from new to old state format."""
        # Convert new state to old format
        old_state = StateAdapter.new_to_old(sample_new_state)
        
        # Verify core fields
        assert old_state['topic'] == sample_new_state.topic
        assert old_state['description'] == sample_new_state.description
        assert old_state['session_id'] == sample_new_state.session_id
        
        # Verify configuration mapping
        assert old_state['use_rag'] == sample_new_state.config.use_rag
        assert old_state['enable_caching'] == sample_new_state.config.enable_caching
        assert old_state['max_retries'] == sample_new_state.config.max_retries
        assert old_state['output_dir'] == sample_new_state.config.output_dir
        
        # Verify state data
        assert old_state['scene_outline'] == sample_new_state.scene_outline
        assert old_state['scene_implementations'] == sample_new_state.scene_implementations
        assert old_state['detected_plugins'] == sample_new_state.detected_plugins
        assert old_state['generated_code'] == sample_new_state.generated_code
        assert old_state['rendered_videos'] == sample_new_state.rendered_videos
        
        # Verify workflow control
        assert old_state['workflow_complete'] == sample_new_state.workflow_complete
        assert old_state['workflow_interrupted'] == sample_new_state.workflow_interrupted
        
        # Verify required old format fields
        assert 'messages' in old_state
        assert isinstance(old_state['messages'], list)
    
    def test_bidirectional_conversion_consistency(self, sample_old_state):
        """Test that converting old->new->old preserves data."""
        # Convert old to new
        new_state = StateAdapter.old_to_new(sample_old_state)
        
        # Convert back to old
        restored_old_state = StateAdapter.new_to_old(new_state)
        
        # Verify key fields are preserved
        assert restored_old_state['topic'] == sample_old_state['topic']
        assert restored_old_state['description'] == sample_old_state['description']
        assert restored_old_state['session_id'] == sample_old_state['session_id']
        assert restored_old_state['use_rag'] == sample_old_state['use_rag']
        assert restored_old_state['scene_outline'] == sample_old_state['scene_outline']
        assert restored_old_state['workflow_complete'] == sample_old_state['workflow_complete']
    
    def test_error_conversion(self):
        """Test conversion of error data."""
        old_state_with_errors = {
            'topic': 'Test topic',
            'description': 'Test description',
            'session_id': 'test-123',
            'escalated_errors': [
                {
                    'agent': 'planner_agent',
                    'error': 'Test error message',
                    'error_type': 'TestError',
                    'timestamp': '2024-01-01T10:00:00',
                    'retry_count': 1
                }
            ],
            'retry_count': {'planner_agent': 1}
        }
        
        new_state = StateAdapter.old_to_new(old_state_with_errors)
        
        # Verify error conversion
        assert len(new_state.errors) == 1
        assert new_state.errors[0].step == 'planner_agent'
        assert new_state.errors[0].message == 'Test error message'
        assert new_state.errors[0].retry_count == 1
        assert new_state.retry_counts == {'planner_agent': 1}
    
    def test_performance_metrics_conversion(self):
        """Test conversion of performance metrics."""
        old_state_with_metrics = {
            'topic': 'Test topic',
            'description': 'Test description',
            'session_id': 'test-123',
            'performance_metrics': {
                'planner_agent': {
                    'last_execution_time': 5.2,
                    'success_rate': 0.95,
                    'average_execution_time': 4.8
                },
                'code_generator_agent': {
                    'last_execution_time': 12.1,
                    'success_rate': 0.88,
                    'average_execution_time': 11.5
                }
            }
        }
        
        new_state = StateAdapter.old_to_new(old_state_with_metrics)
        
        # Verify metrics conversion
        assert new_state.metrics is not None
        assert new_state.metrics.session_id == 'test-123'
        assert 'planning' in new_state.metrics.step_durations
        assert 'code_generation' in new_state.metrics.step_durations
        assert new_state.metrics.step_durations['planning'] == 5.2
        assert new_state.metrics.step_durations['code_generation'] == 12.1
    
    def test_current_step_mapping(self):
        """Test mapping of current agent to current step."""
        test_cases = [
            ('planner_agent', 'planning'),
            ('code_generator_agent', 'code_generation'),
            ('renderer_agent', 'rendering'),
            ('visual_analysis_agent', 'visual_analysis'),
            ('error_handler_agent', 'error_handling'),
            ('human_loop_agent', 'human_loop')
        ]
        
        for agent_name, expected_step in test_cases:
            old_state = {
                'topic': 'Test topic',
                'description': 'Test description',
                'session_id': 'test-123',
                'current_agent': agent_name
            }
            
            new_state = StateAdapter.old_to_new(old_state)
            assert new_state.current_step == expected_step
    
    def test_validation_conversion_success(self, sample_old_state, sample_new_state):
        """Test validation of successful conversion."""
        # Test old to new validation
        new_state = StateAdapter.old_to_new(sample_old_state)
        assert StateAdapter.validate_conversion(sample_old_state, new_state)
        
        # Test with matching data
        matching_old_state = {
            'topic': sample_new_state.topic,
            'description': sample_new_state.description,
            'session_id': sample_new_state.session_id,
            'scene_outline': sample_new_state.scene_outline,
            'workflow_complete': sample_new_state.workflow_complete
        }
        
        assert StateAdapter.validate_conversion(matching_old_state, sample_new_state)
    
    def test_validation_conversion_failure(self, sample_new_state):
        """Test validation of failed conversion."""
        # Create mismatched old state
        mismatched_old_state = {
            'topic': 'Different topic',  # Mismatch
            'description': sample_new_state.description,
            'session_id': sample_new_state.session_id,
            'scene_outline': sample_new_state.scene_outline,
            'workflow_complete': sample_new_state.workflow_complete
        }
        
        assert not StateAdapter.validate_conversion(mismatched_old_state, sample_new_state)
    
    def test_extract_model_config(self):
        """Test model configuration extraction."""
        state_dict = {
            'planner_model': 'openrouter/anthropic/claude-3.5-sonnet',
            'temperature': 0.8
        }
        
        model_config = StateAdapter._extract_model_config(
            state_dict, 'planner_model', 'openrouter/anthropic/claude-3.5-sonnet'
        )
        
        assert model_config.provider == 'openrouter'
        assert 'anthropic/claude-3.5-sonnet' in model_config.model_name
        assert model_config.temperature == 0.7  # Default value
    
    def test_step_duration_extraction(self):
        """Test step duration extraction from performance metrics."""
        state_dict = {
            'performance_metrics': {
                'planner_agent': {'last_execution_time': 5.2},
                'code_generator_agent': {'last_execution_time': 12.1}
            }
        }
        
        step_durations = StateAdapter._extract_step_durations(state_dict)
        
        assert step_durations['planning'] == 5.2
        assert step_durations['code_generation'] == 12.1
    
    def test_success_rate_extraction(self):
        """Test success rate extraction from performance metrics."""
        state_dict = {
            'performance_metrics': {
                'planner_agent': {'success_rate': 0.95},
                'renderer_agent': {'success_rate': 0.88}
            }
        }
        
        success_rates = StateAdapter._extract_success_rates(state_dict)
        
        assert success_rates['planning'] == 0.95
        assert success_rates['rendering'] == 0.88
    
    def test_timestamp_parsing(self):
        """Test timestamp parsing functionality."""
        # Test ISO format
        iso_timestamp = '2024-01-01T10:00:00'
        parsed = StateAdapter._parse_timestamp(iso_timestamp)
        assert isinstance(parsed, datetime)
        
        # Test None input
        parsed_none = StateAdapter._parse_timestamp(None)
        assert isinstance(parsed_none, datetime)
        
        # Test invalid format
        parsed_invalid = StateAdapter._parse_timestamp('invalid-timestamp')
        assert isinstance(parsed_invalid, datetime)
    
    def test_agent_step_mapping(self):
        """Test agent to step mapping functions."""
        # Test agent to step mapping
        assert StateAdapter._map_agent_to_step('planner_agent') == 'planning'
        assert StateAdapter._map_agent_to_step('code_generator_agent') == 'code_generation'
        assert StateAdapter._map_agent_to_step('unknown_agent') == 'planning'  # Default
        
        # Test step to agent mapping
        assert StateAdapter._map_step_to_agent('planning') == 'planner_agent'
        assert StateAdapter._map_step_to_agent('code_generation') == 'code_generator_agent'
        assert StateAdapter._map_step_to_agent('unknown_step') == 'planner_agent'  # Default
    
    def test_conversion_with_complex_data(self):
        """Test conversion with complex nested data structures."""
        complex_old_state = {
            'topic': 'Complex topic',
            'description': 'Complex description',
            'session_id': 'complex-123',
            'scene_implementations': {
                1: 'Complex scene 1 with nested data',
                2: 'Complex scene 2 with special characters: <>&"\'',
                3: 'Scene with unicode: 🎬📹'
            },
            'generated_code': {
                1: 'class Scene1(Scene):\n    def construct(self):\n        pass',
                2: 'class Scene2(Scene):\n    def construct(self):\n        # Complex code\n        pass'
            },
            'rag_context': {
                'queries': ['query1', 'query2'],
                'results': [
                    {'content': 'result1', 'score': 0.95},
                    {'content': 'result2', 'score': 0.87}
                ],
                'metadata': {'total_results': 2, 'search_time': 0.15}
            }
        }
        
        # Should convert successfully
        new_state = StateAdapter.old_to_new(complex_old_state)
        
        # Verify complex data preservation
        assert new_state.scene_implementations[1] == complex_old_state['scene_implementations'][1]
        assert new_state.scene_implementations[3] == complex_old_state['scene_implementations'][3]
        assert new_state.generated_code[1] == complex_old_state['generated_code'][1]
        assert new_state.rag_context == complex_old_state['rag_context']
        
        # Convert back and verify
        restored_old_state = StateAdapter.new_to_old(new_state)
        assert restored_old_state['scene_implementations'] == complex_old_state['scene_implementations']
        assert restored_old_state['generated_code'] == complex_old_state['generated_code']
        assert restored_old_state['rag_context'] == complex_old_state['rag_context']
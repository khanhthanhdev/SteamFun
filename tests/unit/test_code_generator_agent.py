"""
Unit tests for CodeGeneratorAgent.
Tests Manim code generation, error handling, and RAG integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from src.langgraph_agents.agents.code_generator_agent import CodeGeneratorAgent
from src.langgraph_agents.state import VideoGenerationState, AgentConfig
from langgraph.types import Command

# Import test utilities
from tests.utils.config_mocks import (
    mock_configuration_manager, create_test_system_config, 
    create_test_agent_config, MockConfigurationManager
)


class TestCodeGeneratorAgent:
    """Test suite for CodeGeneratorAgent functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock agent configuration for CodeGeneratorAgent using configuration system."""
        return create_test_agent_config(
            name="code_generator_agent",
            scene_model="openai/gpt-4o",
            helper_model="openai/gpt-4o-mini",
            tools=["code_generation_tool", "rag_tool"]
        )
    
    @pytest.fixture
    def mock_system_config(self):
        """Create mock system configuration using configuration system."""
        return create_test_system_config()
    
    @pytest.fixture
    def mock_state(self):
        """Create mock video generation state with planning data."""
        return VideoGenerationState(
            messages=[],
            topic="Python programming basics",
            description="Educational video about Python fundamentals",
            session_id="test_session_123",
            output_dir="test_output",
            print_response=False,
            use_rag=True,
            use_context_learning=True,
            context_learning_path="test_context",
            chroma_db_path="test_chroma",
            manim_docs_path="test_docs",
            embedding_model="test_model",
            use_visual_fix_code=False,
            use_langfuse=True,
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
            scene_outline="# Scene 1\nIntroduction\n# Scene 2\nBasic syntax",
            scene_implementations={
                1: "Scene 1: Show Python logo and introduction text",
                2: "Scene 2: Display code examples with syntax highlighting"
            },
            detected_plugins=["text", "code"],
            generated_code={},
            code_errors={},
            rag_context={},
            rendered_videos={},
            combined_video_path=None,
            rendering_errors={},
            visual_analysis_results={},
            visual_errors={},
            error_count=0,
            retry_count={},
            escalated_errors=[],
            pending_human_input=None,
            human_feedback=None,
            performance_metrics={},
            execution_trace=[],
            current_agent=None,
            next_agent=None,
            workflow_complete=False,
            workflow_interrupted=False
        )
    
    @pytest.fixture
    def code_generator_agent(self, mock_config, mock_system_config):
        """Create CodeGeneratorAgent instance for testing."""
        with mock_configuration_manager(mock_system_config):
            return CodeGeneratorAgent(mock_config, mock_system_config)
    
    @pytest.fixture
    def mock_code_generator(self):
        """Create mock CodeGenerator."""
        mock_generator = Mock()
        mock_generator.generate_manim_code = AsyncMock(return_value={
            1: "from manim import *\n\nclass Scene1(Scene):\n    def construct(self):\n        text = Text('Python')\n        self.play(Write(text))",
            2: "from manim import *\n\nclass Scene2(Scene):\n    def construct(self):\n        code = Code('print(\"Hello World\")')\n        self.play(Create(code))"
        })
        mock_generator.fix_code_errors = AsyncMock(return_value={})
        mock_generator.visual_self_reflection = AsyncMock(return_value={})
        return mock_generator
    
    def test_code_generator_agent_initialization(self, code_generator_agent, mock_config):
        """Test CodeGeneratorAgent initialization."""
        assert code_generator_agent.name == "code_generator_agent"
        assert code_generator_agent.scene_model == "openai/gpt-4o"
        assert code_generator_agent.helper_model == "openai/gpt-4o-mini"
        assert code_generator_agent._code_generator is None  # Lazy initialization
    
    @patch('src.langgraph_agents.agents.code_generator_agent.CodeGenerator')
    def test_get_code_generator_creation(self, mock_generator_class, code_generator_agent, mock_state):
        """Test code generator creation with state configuration."""
        mock_instance = Mock()
        mock_generator_class.return_value = mock_instance
        
        with patch.object(code_generator_agent, 'get_model_wrapper') as mock_get_wrapper:
            mock_wrapper = Mock()
            mock_get_wrapper.return_value = mock_wrapper
            
            generator = code_generator_agent._get_code_generator(mock_state)
            
            # Verify generator was created with correct configuration
            mock_generator_class.assert_called_once()
            call_kwargs = mock_generator_class.call_args[1]
            assert call_kwargs['output_dir'] == 'test_output'
            assert call_kwargs['use_rag'] is True
            assert call_kwargs['session_id'] == 'test_session_123'
            assert call_kwargs['use_visual_fix_code'] is False
            
            assert generator == mock_instance
            assert code_generator_agent._code_generator == mock_instance
    
    @patch('src.langgraph_agents.agents.code_generator_agent.CodeGenerator')
    def test_get_code_generator_reuse(self, mock_generator_class, code_generator_agent, mock_state):
        """Test code generator instance reuse."""
        mock_instance = Mock()
        code_generator_agent._code_generator = mock_instance
        
        generator = code_generator_agent._get_code_generator(mock_state)
        
        # Should not create new instance
        mock_generator_class.assert_not_called()
        assert generator == mock_instance
    
    @pytest.mark.asyncio
    async def test_execute_success_full_workflow(self, code_generator_agent, mock_state, mock_code_generator):
        """Test successful execution of code generation workflow."""
        with patch.object(code_generator_agent, '_get_code_generator', return_value=mock_code_generator):
            command = await code_generator_agent.execute(mock_state)
            
            # Verify code generation was called
            mock_code_generator.generate_manim_code.assert_called_once()
            call_args = mock_code_generator.generate_manim_code.call_args[1]
            assert call_args['topic'] == "Python programming basics"
            assert call_args['session_id'] == "test_session_123"
            
            # Verify command structure
            assert command.goto == "renderer_agent"
            assert len(command.update["generated_code"]) == 2
            assert "from manim import *" in command.update["generated_code"][1]
            assert command.update["current_agent"] == "renderer_agent"
    
    @pytest.mark.asyncio
    async def test_execute_missing_scene_data(self, code_generator_agent, mock_state):
        """Test execution with missing scene outline or implementations."""
        mock_state["scene_outline"] = None
        mock_state["scene_implementations"] = {}
        
        command = await code_generator_agent.execute(mock_state)
        
        # Should route to error handler
        assert command.goto == "error_handler_agent"
        assert command.update["error_count"] == 1
    
    @pytest.mark.asyncio
    async def test_execute_code_generation_failure(self, code_generator_agent, mock_state, mock_code_generator):
        """Test handling of code generation failure."""
        mock_code_generator.generate_manim_code.side_effect = Exception("Code generation failed")
        
        with patch.object(code_generator_agent, '_get_code_generator', return_value=mock_code_generator):
            command = await code_generator_agent.execute(mock_state)
            
            # Should route to error handler
            assert command.goto == "error_handler_agent"
            assert command.update["error_count"] == 1
    
    @pytest.mark.asyncio
    async def test_execute_with_code_errors(self, code_generator_agent, mock_state, mock_code_generator):
        """Test execution when code generation produces errors."""
        # Mock code generation with errors
        mock_code_generator.generate_manim_code.return_value = {
            1: "from manim import *\n\nclass Scene1(Scene):\n    def construct(self):\n        text = Text('Python')\n        self.play(Write(text))",
            2: "# Error in scene 2"
        }
        
        # Mock error detection
        mock_code_generator.fix_code_errors.return_value = {
            2: "from manim import *\n\nclass Scene2(Scene):\n    def construct(self):\n        code = Code('print(\"Hello World\")')\n        self.play(Create(code))"
        }
        
        with patch.object(code_generator_agent, '_get_code_generator', return_value=mock_code_generator):
            command = await code_generator_agent.execute(mock_state)
            
            # Should attempt error fixing
            mock_code_generator.fix_code_errors.assert_called_once()
            
            # Should still proceed to renderer
            assert command.goto == "renderer_agent"
    
    @pytest.mark.asyncio
    async def test_execute_with_visual_fix_enabled(self, code_generator_agent, mock_state, mock_code_generator):
        """Test execution with visual fix code enabled."""
        mock_state["use_visual_fix_code"] = True
        
        with patch.object(code_generator_agent, '_get_code_generator', return_value=mock_code_generator):
            command = await code_generator_agent.execute(mock_state)
            
            # Should call visual self-reflection
            mock_code_generator.visual_self_reflection.assert_called_once()
            
            assert command.goto == "renderer_agent"
    
    @pytest.mark.asyncio
    async def test_execute_human_escalation(self, mock_config, mock_system_config, mock_state, mock_code_generator):
        """Test human escalation when enabled and error threshold reached."""
        mock_config.enable_human_loop = True
        mock_state["error_count"] = 5  # Above threshold
        
        agent = CodeGeneratorAgent(mock_config, mock_system_config)
        mock_code_generator.generate_manim_code.side_effect = Exception("Code generation failed")
        
        with patch.object(agent, '_get_code_generator', return_value=mock_code_generator):
            with patch.object(agent, 'create_human_intervention_command') as mock_human_cmd:
                mock_human_cmd.return_value = Command(goto="human_loop_agent")
                
                command = await agent.execute(mock_state)
                
                mock_human_cmd.assert_called_once()
                assert command.goto == "human_loop_agent"
    
    @pytest.mark.asyncio
    async def test_generate_manim_code_compatibility(self, code_generator_agent, mock_state, mock_code_generator):
        """Test Manim code generation method compatibility."""
        with patch.object(code_generator_agent, '_get_code_generator', return_value=mock_code_generator):
            code = await code_generator_agent.generate_manim_code(
                topic="Test topic",
                description="Test description",
                scene_outline="Test outline",
                scene_implementations={1: "Scene 1"},
                session_id="test_session",
                state=mock_state
            )
            
            mock_code_generator.generate_manim_code.assert_called_once()
            call_kwargs = mock_code_generator.generate_manim_code.call_args[1]
            assert call_kwargs['topic'] == "Test topic"
            assert call_kwargs['session_id'] == "test_session"
            
            assert len(code) == 2  # Should return generated code
    
    @pytest.mark.asyncio
    async def test_fix_code_errors_compatibility(self, code_generator_agent, mock_state, mock_code_generator):
        """Test code error fixing method compatibility."""
        test_code = {1: "invalid code", 2: "valid code"}
        
        with patch.object(code_generator_agent, '_get_code_generator', return_value=mock_code_generator):
            fixed_code = await code_generator_agent.fix_code_errors(
                generated_code=test_code,
                topic="Test topic",
                session_id="test_session",
                state=mock_state
            )
            
            mock_code_generator.fix_code_errors.assert_called_once()
            call_kwargs = mock_code_generator.fix_code_errors.call_args[1]
            assert call_kwargs['generated_code'] == test_code
            assert call_kwargs['topic'] == "Test topic"
    
    @pytest.mark.asyncio
    async def test_visual_self_reflection_compatibility(self, code_generator_agent, mock_state, mock_code_generator):
        """Test visual self-reflection method compatibility."""
        test_code = {1: "scene code"}
        
        with patch.object(code_generator_agent, '_get_code_generator', return_value=mock_code_generator):
            reflection_result = await code_generator_agent.visual_self_reflection(
                generated_code=test_code,
                topic="Test topic",
                session_id="test_session",
                state=mock_state
            )
            
            mock_code_generator.visual_self_reflection.assert_called_once()
            call_kwargs = mock_code_generator.visual_self_reflection.call_args[1]
            assert call_kwargs['generated_code'] == test_code
            assert call_kwargs['topic'] == "Test topic"
    
    def test_get_code_generation_status(self, code_generator_agent, mock_state):
        """Test code generation status reporting."""
        mock_state.update({
            "generated_code": {1: "code1", 2: "code2"},
            "code_errors": {1: "error1"},
            "rag_context": {"query1": "context1"}
        })
        
        status = code_generator_agent.get_code_generation_status(mock_state)
        
        assert status["agent_name"] == "code_generator_agent"
        assert status["generated_scenes_count"] == 2
        assert status["scenes_with_errors"] == 1
        assert status["rag_queries_count"] == 1
        assert status["visual_fix_enabled"] is False
    
    @pytest.mark.asyncio
    async def test_handle_code_generation_error_with_rag(self, code_generator_agent, mock_state):
        """Test code generation error handling with RAG fallback."""
        error = Exception("syntax error in generated code")
        
        with patch.object(code_generator_agent, '_retry_with_rag_context') as mock_retry:
            mock_retry.return_value = Command(goto="renderer_agent")
            
            command = await code_generator_agent.handle_code_generation_error(
                error, mock_state, retry_with_rag=True
            )
            
            mock_retry.assert_called_once()
            assert command.goto == "renderer_agent"
    
    @pytest.mark.asyncio
    async def test_handle_code_generation_error_no_rag(self, code_generator_agent, mock_state):
        """Test code generation error handling without RAG fallback."""
        error = Exception("general code generation error")
        
        with patch.object(code_generator_agent, 'handle_error') as mock_handle_error:
            mock_handle_error.return_value = Command(goto="error_handler_agent")
            
            command = await code_generator_agent.handle_code_generation_error(
                error, mock_state, retry_with_rag=False
            )
            
            mock_handle_error.assert_called_once_with(error, mock_state)
            assert command.goto == "error_handler_agent"
    
    @pytest.mark.asyncio
    async def test_retry_with_rag_context(self, code_generator_agent, mock_state, mock_code_generator):
        """Test retry with RAG context."""
        with patch.object(code_generator_agent, '_get_code_generator', return_value=mock_code_generator):
            with patch.object(code_generator_agent, 'execute') as mock_execute:
                mock_execute.return_value = Command(goto="renderer_agent")
                
                command = await code_generator_agent._retry_with_rag_context(mock_state)
                
                # Should retry execution with enhanced RAG
                mock_execute.assert_called_once()
                retry_state = mock_execute.call_args[0][0]
                assert retry_state["use_enhanced_rag"] is True
                assert retry_state["enable_rag_caching"] is True
    
    def test_detect_code_errors(self, code_generator_agent):
        """Test code error detection."""
        test_code = {
            1: "from manim import *\n\nclass Scene1(Scene):\n    def construct(self):\n        pass",
            2: "# Error: Invalid syntax\nfrom manim import\nclass Scene2",
            3: "from manim import *\n\nclass Scene3(Scene):\n    def construct(self):\n        text = Text('Hello')\n        self.play(Write(text))"
        }
        
        errors = code_generator_agent._detect_code_errors(test_code)
        
        # Should detect error in scene 2
        assert 2 in errors
        assert "syntax" in errors[2].lower() or "error" in errors[2].lower()
        
        # Should not detect errors in scenes 1 and 3
        assert 1 not in errors
        assert 3 not in errors
    
    def test_validate_generated_code(self, code_generator_agent):
        """Test generated code validation."""
        valid_code = {
            1: "from manim import *\n\nclass Scene1(Scene):\n    def construct(self):\n        text = Text('Hello')\n        self.play(Write(text))"
        }
        
        invalid_code = {
            1: "# This is not valid Manim code\nprint('hello')"
        }
        
        assert code_generator_agent._validate_generated_code(valid_code) is True
        assert code_generator_agent._validate_generated_code(invalid_code) is False
    
    def test_extract_scene_classes(self, code_generator_agent):
        """Test scene class extraction from generated code."""
        test_code = {
            1: "from manim import *\n\nclass IntroScene(Scene):\n    def construct(self):\n        pass",
            2: "from manim import *\n\nclass MainScene(Scene):\n    def construct(self):\n        pass"
        }
        
        classes = code_generator_agent._extract_scene_classes(test_code)
        
        assert classes == {1: "IntroScene", 2: "MainScene"}
    
    def test_cleanup_on_destruction(self, code_generator_agent):
        """Test resource cleanup when agent is destroyed."""
        mock_generator = Mock()
        mock_thread_pool = Mock()
        mock_generator.thread_pool = mock_thread_pool
        code_generator_agent._code_generator = mock_generator
        
        # Trigger destructor
        code_generator_agent.__del__()
        
        mock_thread_pool.shutdown.assert_called_once_with(wait=False)


if __name__ == "__main__":
    pytest.main([__file__])
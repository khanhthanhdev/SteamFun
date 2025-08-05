"""
Unit tests for CodeGenerationService.

Tests code generation, error fixing, visual analysis integration, and validation methods.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Tuple
from PIL import Image

from src.langgraph_agents.services.code_generation_service import CodeGenerationService

# Mark all async tests in this module
pytestmark = pytest.mark.asyncio


class TestCodeGenerationService:
    """Test suite for CodeGenerationService functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock service configuration."""
        return {
            'scene_model': 'openrouter/anthropic/claude-3.5-sonnet',
            'helper_model': 'openrouter/anthropic/claude-3.5-sonnet',
            'output_dir': 'test_output',
            'use_rag': True,
            'use_context_learning': True,
            'context_learning_path': 'test_data/context_learning',
            'chroma_db_path': 'test_data/rag/chroma_db',
            'manim_docs_path': 'test_data/rag/manim_docs',
            'embedding_model': 'hf:ibm-granite/granite-embedding-30m-english',
            'use_visual_fix_code': True,
            'use_langfuse': False,  # Disable for testing
            'max_retries': 3,
            'enable_caching': True,
            'print_response': False
        }
    
    @pytest.fixture
    def mock_model_wrappers(self):
        """Create mock model wrappers."""
        scene_model = Mock()
        scene_model.generate = AsyncMock(return_value="Generated code response")
        
        helper_model = Mock()
        helper_model.generate = AsyncMock(return_value="Helper response")
        
        return {
            'scene_model': scene_model,
            'helper_model': helper_model
        }
    
    @pytest.fixture
    def mock_code_generator(self):
        """Create mock CodeGenerator instance."""
        code_generator = Mock()
        
        # Mock code generation
        code_generator.generate_manim_code = Mock(return_value=(
            "from manim import *\n\nclass TestScene(Scene):\n    def construct(self):\n        text = Text('Hello')\n        self.play(Write(text))",
            "Generated successfully"
        ))
        
        # Mock error fixing
        code_generator.fix_code_errors = Mock(return_value=(
            "from manim import *\n\nclass TestScene(Scene):\n    def construct(self):\n        text = Text('Hello Fixed')\n        self.play(Write(text))",
            "Fixed successfully"
        ))
        
        # Mock visual analysis
        code_generator.visual_self_reflection = Mock(return_value=(
            "from manim import *\n\nclass TestScene(Scene):\n    def construct(self):\n        text = Text('Hello Visual')\n        self.play(Write(text))",
            "Visual analysis complete"
        ))
        
        code_generator.enhanced_visual_self_reflection = Mock(return_value=(
            "from manim import *\n\nclass TestScene(Scene):\n    def construct(self):\n        text = Text('Hello Enhanced')\n        self.play(Write(text))",
            "Enhanced visual analysis complete"
        ))
        
        # Mock RAG methods
        code_generator._generate_rag_queries_code = Mock(return_value=[
            "How to create text in Manim?",
            "How to animate text in Manim?"
        ])
        
        code_generator._generate_rag_queries_error_fix = Mock(return_value=[
            "How to fix Manim import errors?",
            "Common Manim syntax issues"
        ])
        
        code_generator._retrieve_rag_context = Mock(return_value=
            "Manim context: Use Text() to create text objects and Write() to animate them."
        )
        
        return code_generator
    
    @pytest.fixture
    def code_generation_service(self, mock_config):
        """Create CodeGenerationService instance for testing."""
        return CodeGenerationService(mock_config)
    
    def test_code_generation_service_initialization(self, code_generation_service, mock_config):
        """Test CodeGenerationService initialization."""
        assert code_generation_service.scene_model == mock_config['scene_model']
        assert code_generation_service.helper_model == mock_config['helper_model']
        assert code_generation_service.output_dir == mock_config['output_dir']
        assert code_generation_service.use_rag == mock_config['use_rag']
        assert code_generation_service.use_context_learning == mock_config['use_context_learning']
        assert code_generation_service.use_visual_fix_code == mock_config['use_visual_fix_code']
        assert code_generation_service.max_retries == mock_config['max_retries']
        assert code_generation_service.enable_caching == mock_config['enable_caching']
        assert code_generation_service._code_generator is None
    
    @patch('src.langgraph_agents.services.code_generation_service.CodeGenerator')
    async def test_generate_scene_code_success(self, mock_code_generator_class, 
                                             code_generation_service, mock_model_wrappers, 
                                             mock_code_generator):
        """Test successful scene code generation."""
        mock_code_generator_class.return_value = mock_code_generator
        
        code, response = await code_generation_service.generate_scene_code(
            topic="Python basics",
            description="Introduction to Python programming",
            scene_outline="Scene 1: Hello World",
            scene_implementation="Show a simple hello world text",
            scene_number=1,
            session_id="test-session-123",
            model_wrappers=mock_model_wrappers
        )
        
        assert code is not None
        assert "from manim import" in code
        assert "class TestScene" in code
        assert response == "Generated successfully"
        
        # Verify CodeGenerator was called correctly
        mock_code_generator.generate_manim_code.assert_called_once()
        call_args = mock_code_generator.generate_manim_code.call_args
        assert call_args[1]['topic'] == "Python basics"
        assert call_args[1]['scene_number'] == 1
        assert call_args[1]['session_id'] == "test-session-123"
    
    async def test_generate_scene_code_validation_errors(self, code_generation_service, mock_model_wrappers):
        """Test scene code generation with validation errors."""
        # Test empty topic
        with pytest.raises(ValueError, match="Topic cannot be empty"):
            await code_generation_service.generate_scene_code(
                topic="",
                description="Test description",
                scene_outline="Test outline",
                scene_implementation="Test implementation",
                scene_number=1,
                session_id="test-session",
                model_wrappers=mock_model_wrappers
            )
        
        # Test empty description
        with pytest.raises(ValueError, match="Description cannot be empty"):
            await code_generation_service.generate_scene_code(
                topic="Test topic",
                description="",
                scene_outline="Test outline",
                scene_implementation="Test implementation",
                scene_number=1,
                session_id="test-session",
                model_wrappers=mock_model_wrappers
            )
        
        # Test invalid scene number
        with pytest.raises(ValueError, match="Scene number must be positive"):
            await code_generation_service.generate_scene_code(
                topic="Test topic",
                description="Test description",
                scene_outline="Test outline",
                scene_implementation="Test implementation",
                scene_number=0,
                session_id="test-session",
                model_wrappers=mock_model_wrappers
            )
    
    @patch('src.langgraph_agents.services.code_generation_service.CodeGenerator')
    async def test_generate_scene_code_with_rag(self, mock_code_generator_class,
                                              code_generation_service, mock_model_wrappers,
                                              mock_code_generator):
        """Test scene code generation with RAG integration."""
        mock_code_generator_class.return_value = mock_code_generator
        
        code, response = await code_generation_service.generate_scene_code(
            topic="Manim animations",
            description="Learn Manim basics",
            scene_outline="Scene 1: Text animation",
            scene_implementation="Animate text appearing",
            scene_number=1,
            session_id="test-session-123",
            model_wrappers=mock_model_wrappers
        )
        
        # Verify RAG queries were generated
        mock_code_generator._generate_rag_queries_code.assert_called_once()
        
        # Verify RAG context was retrieved
        mock_code_generator._retrieve_rag_context.assert_called_once()
        
        # Verify code generation was called with additional context
        mock_code_generator.generate_manim_code.assert_called_once()
        call_args = mock_code_generator.generate_manim_code.call_args
        assert call_args[1]['additional_context'] is not None
    
    @patch('src.langgraph_agents.services.code_generation_service.CodeGenerator')
    async def test_fix_code_errors_success(self, mock_code_generator_class,
                                         code_generation_service, mock_model_wrappers,
                                         mock_code_generator):
        """Test successful code error fixing."""
        mock_code_generator_class.return_value = mock_code_generator
        
        fixed_code, response = await code_generation_service.fix_code_errors(
            implementation_plan="Show hello world text",
            code="from manim import *\nclass TestScene(Scene):\n    def construct(self):\n        text = Tex('Hello')",
            error="NameError: name 'Tex' is not defined",
            scene_number=1,
            topic="Python basics",
            session_id="test-session-123",
            model_wrappers=mock_model_wrappers
        )
        
        assert fixed_code is not None
        assert "Hello Fixed" in fixed_code
        assert response == "Fixed successfully"
        
        # Verify error fixing was called correctly
        mock_code_generator.fix_code_errors.assert_called_once()
        call_args = mock_code_generator.fix_code_errors.call_args
        assert call_args[1]['error'] == "NameError: name 'Tex' is not defined"
        assert call_args[1]['scene_number'] == 1
    
    async def test_fix_code_errors_validation_errors(self, code_generation_service, mock_model_wrappers):
        """Test code error fixing with validation errors."""
        # Test empty implementation plan
        with pytest.raises(ValueError, match="Implementation plan cannot be empty"):
            await code_generation_service.fix_code_errors(
                implementation_plan="",
                code="test code",
                error="test error",
                scene_number=1,
                topic="test topic",
                session_id="test-session",
                model_wrappers=mock_model_wrappers
            )
        
        # Test empty code
        with pytest.raises(ValueError, match="Code cannot be empty"):
            await code_generation_service.fix_code_errors(
                implementation_plan="test plan",
                code="",
                error="test error",
                scene_number=1,
                topic="test topic",
                session_id="test-session",
                model_wrappers=mock_model_wrappers
            )
        
        # Test empty error message
        with pytest.raises(ValueError, match="Error message cannot be empty"):
            await code_generation_service.fix_code_errors(
                implementation_plan="test plan",
                code="test code",
                error="",
                scene_number=1,
                topic="test topic",
                session_id="test-session",
                model_wrappers=mock_model_wrappers
            )
    
    @patch('src.langgraph_agents.services.code_generation_service.CodeGenerator')
    async def test_visual_analysis_fix_success(self, mock_code_generator_class,
                                             code_generation_service, mock_model_wrappers,
                                             mock_code_generator):
        """Test successful visual analysis fix."""
        mock_code_generator_class.return_value = mock_code_generator
        
        # Test with string media path
        fixed_code, response = await code_generation_service.visual_analysis_fix(
            code="from manim import *\nclass TestScene(Scene):\n    def construct(self):\n        pass",
            media_path="test_image.png",
            scene_number=1,
            topic="Visual test",
            session_id="test-session-123",
            model_wrappers=mock_model_wrappers
        )
        
        assert fixed_code is not None
        assert "Hello Visual" in fixed_code
        assert response == "Visual analysis complete"
        
        # Verify visual analysis was called
        mock_code_generator.visual_self_reflection.assert_called_once()
    
    @patch('src.langgraph_agents.services.code_generation_service.CodeGenerator')
    async def test_visual_analysis_fix_with_implementation_plan(self, mock_code_generator_class,
                                                              code_generation_service, mock_model_wrappers,
                                                              mock_code_generator):
        """Test visual analysis fix with implementation plan."""
        mock_code_generator_class.return_value = mock_code_generator
        
        fixed_code, response = await code_generation_service.visual_analysis_fix(
            code="test code",
            media_path="test_image.png",
            scene_number=1,
            topic="Visual test",
            session_id="test-session-123",
            model_wrappers=mock_model_wrappers,
            implementation_plan="Show animated text"
        )
        
        assert fixed_code is not None
        assert "Hello Enhanced" in fixed_code
        assert response == "Enhanced visual analysis complete"
        
        # Verify enhanced visual analysis was called
        mock_code_generator.enhanced_visual_self_reflection.assert_called_once()
    
    async def test_visual_analysis_fix_validation_errors(self, code_generation_service, mock_model_wrappers):
        """Test visual analysis fix with validation errors."""
        # Test empty code
        with pytest.raises(ValueError, match="Code cannot be empty"):
            await code_generation_service.visual_analysis_fix(
                code="",
                media_path="test_image.png",
                scene_number=1,
                topic="test topic",
                session_id="test-session",
                model_wrappers=mock_model_wrappers
            )
        
        # Test empty media path
        with pytest.raises(ValueError, match="Media path cannot be empty"):
            await code_generation_service.visual_analysis_fix(
                code="test code",
                media_path="",
                scene_number=1,
                topic="test topic",
                session_id="test-session",
                model_wrappers=mock_model_wrappers
            )
    
    async def test_visual_analysis_fix_disabled(self, mock_config, mock_model_wrappers):
        """Test visual analysis fix when feature is disabled."""
        mock_config['use_visual_fix_code'] = False
        service = CodeGenerationService(mock_config)
        
        with pytest.raises(ValueError, match="Visual code fixing is not enabled"):
            await service.visual_analysis_fix(
                code="test code",
                media_path="test_image.png",
                scene_number=1,
                topic="test topic",
                session_id="test-session",
                model_wrappers=mock_model_wrappers
            )
    
    @patch('src.langgraph_agents.services.code_generation_service.CodeGenerator')
    async def test_generate_multiple_scenes_parallel_success(self, mock_code_generator_class,
                                                           code_generation_service, mock_model_wrappers,
                                                           mock_code_generator):
        """Test successful parallel scene generation."""
        mock_code_generator_class.return_value = mock_code_generator
        
        scene_implementations = {
            1: "Show hello world",
            2: "Show goodbye world",
            3: "Show animation"
        }
        
        results = await code_generation_service.generate_multiple_scenes_parallel(
            topic="Test topic",
            description="Test description",
            scene_outline="Test outline",
            scene_implementations=scene_implementations,
            session_id="test-session-123",
            model_wrappers=mock_model_wrappers,
            max_concurrent=2
        )
        
        assert len(results) == 3
        assert all(scene_num in results for scene_num in [1, 2, 3])
        
        for scene_num, (code, response) in results.items():
            assert code is not None
            assert "from manim import" in code
            assert response == "Generated successfully"
        
        # Verify code generation was called for each scene
        assert mock_code_generator.generate_manim_code.call_count == 3
    
    async def test_generate_multiple_scenes_parallel_empty_input(self, code_generation_service, mock_model_wrappers):
        """Test parallel scene generation with empty input."""
        with pytest.raises(ValueError, match="Scene implementations cannot be empty"):
            await code_generation_service.generate_multiple_scenes_parallel(
                topic="Test topic",
                description="Test description",
                scene_outline="Test outline",
                scene_implementations={},
                session_id="test-session",
                model_wrappers=mock_model_wrappers
            )
    
    async def test_validate_generated_code_success(self, code_generation_service):
        """Test successful code validation."""
        valid_code = """
from manim import *

class TestScene(Scene):
    def construct(self):
        text = Text('Hello World')
        self.play(Write(text))
        self.wait()
"""
        
        is_valid, issues = await code_generation_service.validate_generated_code(valid_code, 1)
        
        assert is_valid is True
        assert len(issues) == 0
    
    async def test_validate_generated_code_failures(self, code_generation_service):
        """Test code validation failures."""
        # Test empty code
        is_valid, issues = await code_generation_service.validate_generated_code("", 1)
        assert is_valid is False
        assert "Generated code is empty" in issues
        
        # Test too short code
        is_valid, issues = await code_generation_service.validate_generated_code("short", 1)
        assert is_valid is False
        assert "Generated code is too short" in issues[0]
        
        # Test dangerous patterns
        dangerous_code = """
import os
from manim import *

class TestScene(Scene):
    def construct(self):
        os.system('rm -rf /')
"""
        is_valid, issues = await code_generation_service.validate_generated_code(dangerous_code, 1)
        assert is_valid is False
        assert any("dangerous pattern" in issue for issue in issues)
        
        # Test missing Manim elements
        non_manim_code = """
def hello():
    print("Hello World")
    return "done"
"""
        is_valid, issues = await code_generation_service.validate_generated_code(non_manim_code, 1)
        assert is_valid is False
        assert "Code lacks Manim-specific elements" in issues
        
        # Test syntax errors
        syntax_error_code = """
from manim import *

class TestScene(Scene):
    def construct(self)
        text = Text('Hello')  # Missing colon
        self.play(Write(text))
"""
        is_valid, issues = await code_generation_service.validate_generated_code(syntax_error_code, 1)
        assert is_valid is False
        assert any("syntax errors" in issue for issue in issues)
    
    def test_get_code_generation_metrics(self, code_generation_service):
        """Test getting code generation metrics."""
        metrics = code_generation_service.get_code_generation_metrics()
        
        assert metrics['service_name'] == 'CodeGenerationService'
        assert 'config' in metrics
        assert metrics['config']['use_rag'] is True
        assert metrics['config']['use_context_learning'] is True
        assert metrics['config']['use_visual_fix_code'] is True
        assert metrics['config']['max_retries'] == 3
        assert metrics['config']['enable_caching'] is True
        assert metrics['code_generator_initialized'] is False
    
    @patch('src.langgraph_agents.services.code_generation_service.CodeGenerator')
    async def test_cleanup(self, mock_code_generator_class, code_generation_service, mock_model_wrappers):
        """Test service cleanup."""
        mock_code_generator = Mock()
        mock_code_generator.cleanup = Mock()
        mock_code_generator_class.return_value = mock_code_generator
        
        # Initialize the code generator
        code_generation_service._get_code_generator(mock_model_wrappers, "test-session")
        
        # Test cleanup
        await code_generation_service.cleanup()
        
        # Verify cleanup was called
        mock_code_generator.cleanup.assert_called_once()
        assert code_generation_service._code_generator is None
    
    @patch('src.langgraph_agents.services.code_generation_service.CodeGenerator')
    async def test_rag_query_generation_error_handling(self, mock_code_generator_class,
                                                      code_generation_service, mock_model_wrappers,
                                                      mock_code_generator):
        """Test RAG query generation error handling."""
        # Mock RAG query generation to raise an exception
        mock_code_generator._generate_rag_queries_code.side_effect = Exception("RAG query generation failed")
        mock_code_generator_class.return_value = mock_code_generator
        
        # Should still succeed even if RAG query generation fails
        code, response = await code_generation_service.generate_scene_code(
            topic="Test topic",
            description="Test description",
            scene_outline="Test outline",
            scene_implementation="Test implementation",
            scene_number=1,
            session_id="test-session-123",
            model_wrappers=mock_model_wrappers
        )
        
        assert code is not None
        assert response == "Generated successfully"
    
    @patch('src.langgraph_agents.services.code_generation_service.CodeGenerator')
    async def test_rag_context_retrieval_error_handling(self, mock_code_generator_class,
                                                       code_generation_service, mock_model_wrappers,
                                                       mock_code_generator):
        """Test RAG context retrieval error handling."""
        # Mock RAG context retrieval to raise an exception
        mock_code_generator._retrieve_rag_context.side_effect = Exception("RAG context retrieval failed")
        mock_code_generator_class.return_value = mock_code_generator
        
        # Should still succeed even if RAG context retrieval fails
        code, response = await code_generation_service.generate_scene_code(
            topic="Test topic",
            description="Test description",
            scene_outline="Test outline",
            scene_implementation="Test implementation",
            scene_number=1,
            session_id="test-session-123",
            model_wrappers=mock_model_wrappers
        )
        
        assert code is not None
        assert response == "Generated successfully"
    
    @patch('src.langgraph_agents.services.code_generation_service.CodeGenerator')
    async def test_code_generator_without_rag_support(self, mock_code_generator_class,
                                                    code_generation_service, mock_model_wrappers):
        """Test code generation with CodeGenerator that doesn't support RAG."""
        # Create mock without RAG methods
        mock_code_generator = Mock()
        mock_code_generator.generate_manim_code = Mock(return_value=(
            "from manim import *\nclass TestScene(Scene):\n    def construct(self):\n        pass",
            "Generated without RAG"
        ))
        # Remove RAG methods to simulate older CodeGenerator
        delattr(mock_code_generator, '_generate_rag_queries_code')
        delattr(mock_code_generator, '_retrieve_rag_context')
        
        mock_code_generator_class.return_value = mock_code_generator
        
        code, response = await code_generation_service.generate_scene_code(
            topic="Test topic",
            description="Test description",
            scene_outline="Test outline",
            scene_implementation="Test implementation",
            scene_number=1,
            session_id="test-session-123",
            model_wrappers=mock_model_wrappers
        )
        
        assert code is not None
        assert response == "Generated without RAG"
        
        # Verify code generation was still called
        mock_code_generator.generate_manim_code.assert_called_once()


# Integration test fixtures and helpers
@pytest.fixture
def integration_test_config():
    """Configuration for integration tests."""
    return {
        'scene_model': 'openrouter/anthropic/claude-3.5-sonnet',
        'helper_model': 'openrouter/anthropic/claude-3.5-sonnet',
        'output_dir': 'test_output',
        'use_rag': False,  # Disable RAG for integration tests
        'use_context_learning': False,
        'use_visual_fix_code': False,
        'use_langfuse': False,
        'max_retries': 1,
        'enable_caching': False,
        'print_response': False
    }


class TestCodeGenerationServiceIntegration:
    """Integration tests for CodeGenerationService."""
    
    @pytest.mark.integration
    async def test_service_initialization_integration(self, integration_test_config):
        """Test service initialization in integration environment."""
        service = CodeGenerationService(integration_test_config)
        
        assert service is not None
        assert service.use_rag is False
        assert service.use_context_learning is False
        assert service.use_visual_fix_code is False
        
        # Test cleanup
        await service.cleanup()
    
    @pytest.mark.integration
    async def test_code_validation_integration(self, integration_test_config):
        """Test code validation with real code samples."""
        service = CodeGenerationService(integration_test_config)
        
        # Test with real Manim code
        real_manim_code = """
from manim import *

class IntroScene(Scene):
    def construct(self):
        title = Text("Welcome to Manim", font_size=48)
        subtitle = Text("Mathematical Animation Engine", font_size=24)
        subtitle.next_to(title, DOWN, buff=0.5)
        
        self.play(Write(title))
        self.wait(1)
        self.play(Write(subtitle))
        self.wait(2)
        
        self.play(FadeOut(title), FadeOut(subtitle))
"""
        
        is_valid, issues = await service.validate_generated_code(real_manim_code, 1)
        
        assert is_valid is True
        assert len(issues) == 0
        
        await service.cleanup()
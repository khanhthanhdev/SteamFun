"""
Unit tests for CodeGenerationService.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List
from PIL import Image

from src.langgraph_agents.services.code_generation_service import CodeGenerationService


class TestCodeGenerationService:
    """Test suite for CodeGenerationService functionality."""
    
    @pytest.fixture
    def service_config(self):
        """Create test configuration for CodeGenerationService."""
        return {
            'scene_model': 'test-scene-model',
            'helper_model': 'test-helper-model',
            'output_dir': 'test_output',
            'use_rag': True,
            'use_context_learning': True,
            'context_learning_path': 'test_context',
            'chroma_db_path': 'test_chroma',
            'manim_docs_path': 'test_docs',
            'embedding_model': 'test-embedding',
            'use_visual_fix_code': True,
            'use_langfuse': False,
            'max_retries': 3,
            'session_id': 'test-session-123',
            'detected_plugins': ['numpy', 'matplotlib']
        }
    
    @pytest.fixture
    def code_generation_service(self, service_config):
        """Create CodeGenerationService instance for testing."""
        return CodeGenerationService(service_config)
    
    @pytest.fixture
    def mock_model_wrappers(self):
        """Create mock model wrappers."""
        return {
            'scene_model': Mock(),
            'helper_model': Mock()
        }
    
    def test_init(self, service_config):
        """Test CodeGenerationService initialization."""
        service = CodeGenerationService(service_config)
        
        assert service.config == service_config
        assert service.scene_model == 'test-scene-model'
        assert service.helper_model == 'test-helper-model'
        assert service.use_rag is True
        assert service.use_visual_fix_code is True
        assert service.max_retries == 3
        assert service._code_generator is None
    
    @patch('src.langgraph_agents.services.code_generation_service.CodeGenerator')
    def test_get_code_generator_creation(self, mock_generator_class, code_generation_service, mock_model_wrappers):
        """Test code generator creation."""
        mock_generator_instance = Mock()
        mock_generator_class.return_value = mock_generator_instance
        
        result = code_generation_service._get_code_generator(mock_model_wrappers)
        
        assert result == mock_generator_instance
        assert code_generation_service._code_generator == mock_generator_instance
        
        # Verify generator was created with correct arguments
        mock_generator_class.assert_called_once()
        call_args = mock_generator_class.call_args
        assert call_args[1]['scene_model'] == mock_model_wrappers['scene_model']
        assert call_args[1]['helper_model'] == mock_model_wrappers['helper_model']
        assert call_args[1]['use_rag'] is True
        assert call_args[1]['use_visual_fix_code'] is True
    
    @patch('src.langgraph_agents.services.code_generation_service.CodeGenerator')
    def test_get_code_generator_reuse(self, mock_generator_class, code_generation_service, mock_model_wrappers):
        """Test code generator instance reuse."""
        mock_generator_instance = Mock()
        mock_generator_class.return_value = mock_generator_instance
        
        # First call creates the generator
        result1 = code_generation_service._get_code_generator(mock_model_wrappers)
        # Second call should reuse the same instance
        result2 = code_generation_service._get_code_generator(mock_model_wrappers)
        
        assert result1 == result2 == mock_generator_instance
        # Should only be called once
        mock_generator_class.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_manim_code_success(self, code_generation_service, mock_model_wrappers):
        """Test successful Manim code generation."""
        mock_generator = Mock()
        mock_generator.generate_manim_code.return_value = (
            "from manim import *\n\nclass TestScene(Scene):\n    def construct(self):\n        pass",
            "Generated successfully"
        )
        
        with patch.object(code_generation_service, '_get_code_generator', return_value=mock_generator):
            with patch.object(code_generation_service, '_generate_rag_queries_code', return_value=[]):
                code, response = await code_generation_service.generate_manim_code(
                    topic="Python basics",
                    description="Introduction to Python programming",
                    scene_outline="Scene 1: Introduction",
                    scene_implementation="Show title and basic concepts",
                    scene_number=1,
                    model_wrappers=mock_model_wrappers
                )
        
        assert "from manim import" in code
        assert "class TestScene(Scene)" in code
        assert response == "Generated successfully"
        mock_generator.generate_manim_code.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_manim_code_empty_topic(self, code_generation_service, mock_model_wrappers):
        """Test Manim code generation with empty topic."""
        with pytest.raises(ValueError, match="Topic cannot be empty"):
            await code_generation_service.generate_manim_code(
                topic="",
                description="Test description",
                scene_outline="Test outline",
                scene_implementation="Test implementation",
                scene_number=1,
                model_wrappers=mock_model_wrappers
            )
    
    @pytest.mark.asyncio
    async def test_generate_manim_code_invalid_scene_number(self, code_generation_service, mock_model_wrappers):
        """Test Manim code generation with invalid scene number."""
        with pytest.raises(ValueError, match="Scene number must be positive"):
            await code_generation_service.generate_manim_code(
                topic="Test topic",
                description="Test description",
                scene_outline="Test outline",
                scene_implementation="Test implementation",
                scene_number=0,
                model_wrappers=mock_model_wrappers
            )
    
    @pytest.mark.asyncio
    async def test_generate_manim_code_with_rag(self, code_generation_service, mock_model_wrappers):
        """Test Manim code generation with RAG context."""
        mock_generator = Mock()
        mock_generator.generate_manim_code.return_value = (
            "from manim import *\n\nclass TestScene(Scene):\n    def construct(self):\n        pass",
            "Generated with RAG context"
        )
        
        mock_rag_queries = ["How to create text in Manim", "Manim animation basics"]
        mock_rag_context = "RAG context about Manim text creation"
        
        with patch.object(code_generation_service, '_get_code_generator', return_value=mock_generator):
            with patch.object(code_generation_service, '_generate_rag_queries_code', return_value=mock_rag_queries):
                with patch.object(code_generation_service, '_retrieve_rag_context', return_value=mock_rag_context):
                    code, response = await code_generation_service.generate_manim_code(
                        topic="Python basics",
                        description="Introduction to Python programming",
                        scene_outline="Scene 1: Introduction",
                        scene_implementation="Show title and basic concepts",
                        scene_number=1,
                        model_wrappers=mock_model_wrappers
                    )
        
        assert "from manim import" in code
        # Verify RAG context was added to additional_context
        call_args = mock_generator.generate_manim_code.call_args
        assert call_args[1]['additional_context'] == [mock_rag_context]
    
    @pytest.mark.asyncio
    async def test_fix_code_errors_success(self, code_generation_service, mock_model_wrappers):
        """Test successful code error fixing."""
        mock_generator = Mock()
        mock_generator.fix_code_errors.return_value = (
            "from manim import *\n\nclass FixedScene(Scene):\n    def construct(self):\n        # Fixed code",
            "Fixed successfully"
        )
        mock_generator._generate_rag_queries_error_fix.return_value = ["Error fixing query"]
        mock_generator._retrieve_rag_context.return_value = "Error fixing context"
        
        with patch.object(code_generation_service, '_get_code_generator', return_value=mock_generator):
            fixed_code, response = await code_generation_service.fix_code_errors(
                implementation_plan="Original implementation plan",
                code="broken code",
                error="NameError: name 'Text' is not defined",
                scene_trace_id="scene_1_test",
                topic="Python basics",
                scene_number=1,
                session_id="test-session",
                model_wrappers=mock_model_wrappers
            )
        
        assert "Fixed code" in fixed_code
        assert response == "Fixed successfully"
        mock_generator.fix_code_errors.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fix_code_errors_empty_code(self, code_generation_service, mock_model_wrappers):
        """Test code error fixing with empty code."""
        with pytest.raises(ValueError, match="Code cannot be empty"):
            await code_generation_service.fix_code_errors(
                implementation_plan="Test plan",
                code="",
                error="Test error",
                scene_trace_id="scene_1_test",
                topic="Test topic",
                scene_number=1,
                session_id="test-session",
                model_wrappers=mock_model_wrappers
            )
    
    @pytest.mark.asyncio
    async def test_visual_self_reflection_success(self, code_generation_service, mock_model_wrappers):
        """Test successful visual self-reflection."""
        mock_generator = Mock()
        mock_generator.visual_self_reflection.return_value = (
            "from manim import *\n\nclass ImprovedScene(Scene):\n    def construct(self):\n        # Visually improved",
            "Improved based on visual analysis"
        )
        
        with patch.object(code_generation_service, '_get_code_generator', return_value=mock_generator):
            fixed_code, response = await code_generation_service.visual_self_reflection(
                code="original code",
                media_path="/path/to/image.png",
                scene_trace_id="scene_1_test",
                topic="Python basics",
                scene_number=1,
                session_id="test-session",
                model_wrappers=mock_model_wrappers
            )
        
        assert "Visually improved" in fixed_code
        assert response == "Improved based on visual analysis"
        mock_generator.visual_self_reflection.assert_called_once()
        
        # Check that visual analysis results were stored
        visual_results = code_generation_service.config.get('visual_analysis_results', {})
        assert 1 in visual_results
        assert visual_results[1]['attempted'] is True
        assert visual_results[1]['success'] is True
    
    @pytest.mark.asyncio
    async def test_visual_self_reflection_with_pil_image(self, code_generation_service, mock_model_wrappers):
        """Test visual self-reflection with PIL Image."""
        mock_generator = Mock()
        mock_generator.visual_self_reflection.return_value = ("fixed code", "response")
        
        # Create a mock PIL Image
        mock_image = Mock(spec=Image.Image)
        
        with patch.object(code_generation_service, '_get_code_generator', return_value=mock_generator):
            await code_generation_service.visual_self_reflection(
                code="original code",
                media_path=mock_image,
                scene_trace_id="scene_1_test",
                topic="Python basics",
                scene_number=1,
                session_id="test-session",
                model_wrappers=mock_model_wrappers
            )
        
        # Check that PIL Image was handled correctly
        visual_results = code_generation_service.config.get('visual_analysis_results', {})
        assert visual_results[1]['media_path'] == 'PIL_Image'
    
    @pytest.mark.asyncio
    async def test_visual_self_reflection_error_handling(self, code_generation_service, mock_model_wrappers):
        """Test visual self-reflection error handling."""
        mock_generator = Mock()
        mock_generator.visual_self_reflection.side_effect = Exception("Visual analysis failed")
        
        with patch.object(code_generation_service, '_get_code_generator', return_value=mock_generator):
            with pytest.raises(Exception, match="Visual analysis failed"):
                await code_generation_service.visual_self_reflection(
                    code="original code",
                    media_path="/path/to/image.png",
                    scene_trace_id="scene_1_test",
                    topic="Python basics",
                    scene_number=1,
                    session_id="test-session",
                    model_wrappers=mock_model_wrappers
                )
        
        # Check that error was recorded
        visual_results = code_generation_service.config.get('visual_analysis_results', {})
        assert visual_results[1]['success'] is False
        assert visual_results[1]['error'] == "Visual analysis failed"
    
    @pytest.mark.asyncio
    async def test_enhanced_visual_self_reflection(self, code_generation_service, mock_model_wrappers):
        """Test enhanced visual self-reflection."""
        mock_generator = Mock()
        mock_generator.enhanced_visual_self_reflection.return_value = ("enhanced code", "enhanced response")
        
        with patch.object(code_generation_service, '_get_code_generator', return_value=mock_generator):
            fixed_code, response = await code_generation_service.enhanced_visual_self_reflection(
                code="original code",
                media_path="/path/to/image.png",
                scene_trace_id="scene_1_test",
                topic="Python basics",
                scene_number=1,
                session_id="test-session",
                model_wrappers=mock_model_wrappers,
                implementation_plan="Test implementation plan"
            )
        
        assert fixed_code == "enhanced code"
        assert response == "enhanced response"
        mock_generator.enhanced_visual_self_reflection.assert_called_once_with(
            code="original code",
            media_path="/path/to/image.png",
            scene_trace_id="scene_1_test",
            topic="Python basics",
            scene_number=1,
            session_id="test-session",
            implementation_plan="Test implementation plan"
        )
    
    @pytest.mark.asyncio
    async def test_generate_rag_queries_code(self, code_generation_service, mock_model_wrappers):
        """Test RAG queries generation for code."""
        mock_generator = Mock()
        mock_queries = ["How to create text in Manim", "Manim animation basics"]
        mock_generator._generate_rag_queries_code.return_value = mock_queries
        
        with patch.object(code_generation_service, '_get_code_generator', return_value=mock_generator):
            queries = await code_generation_service._generate_rag_queries_code(
                implementation="Show text animation",
                scene_trace_id="scene_1_test",
                topic="Python basics",
                scene_number=1,
                session_id="test-session",
                relevant_plugins=["numpy"],
                model_wrappers=mock_model_wrappers
            )
        
        assert queries == mock_queries
        # Check that queries were stored
        rag_context = code_generation_service.config.get('rag_context', {})
        assert 'queries_scene_1' in rag_context
        assert rag_context['queries_scene_1'] == mock_queries
    
    @pytest.mark.asyncio
    async def test_retrieve_rag_context(self, code_generation_service, mock_model_wrappers):
        """Test RAG context retrieval."""
        mock_generator = Mock()
        mock_context = "Retrieved context about Manim text creation"
        mock_generator._retrieve_rag_context.return_value = mock_context
        
        with patch.object(code_generation_service, '_get_code_generator', return_value=mock_generator):
            context = await code_generation_service._retrieve_rag_context(
                rag_queries=["How to create text in Manim"],
                scene_trace_id="scene_1_test",
                topic="Python basics",
                scene_number=1,
                model_wrappers=mock_model_wrappers
            )
        
        assert context == mock_context
        # Check that context was stored
        rag_context = code_generation_service.config.get('rag_context', {})
        assert 'context_scene_1' in rag_context
        assert rag_context['context_scene_1'] == mock_context
    
    @pytest.mark.asyncio
    async def test_retrieve_rag_context_empty_queries(self, code_generation_service, mock_model_wrappers):
        """Test RAG context retrieval with empty queries."""
        context = await code_generation_service._retrieve_rag_context(
            rag_queries=[],
            scene_trace_id="scene_1_test",
            topic="Python basics",
            scene_number=1,
            model_wrappers=mock_model_wrappers
        )
        
        assert context is None
    
    @pytest.mark.asyncio
    async def test_validate_generated_code_valid(self, code_generation_service):
        """Test code validation with valid Manim code."""
        valid_code = """
from manim import *

class TestScene(Scene):
    def construct(self):
        text = Text("Hello, World!")
        self.play(Write(text))
        """
        
        is_valid, issues = await code_generation_service.validate_generated_code(valid_code, 1)
        
        assert is_valid is True
        assert len(issues) == 0
    
    @pytest.mark.asyncio
    async def test_validate_generated_code_empty(self, code_generation_service):
        """Test code validation with empty code."""
        is_valid, issues = await code_generation_service.validate_generated_code("", 1)
        
        assert is_valid is False
        assert "Generated code is empty" in issues
    
    @pytest.mark.asyncio
    async def test_validate_generated_code_no_manim_import(self, code_generation_service):
        """Test code validation without Manim import."""
        code_without_import = """
class TestScene(Scene):
    def construct(self):
        pass
        """
        
        is_valid, issues = await code_generation_service.validate_generated_code(code_without_import, 1)
        
        assert is_valid is False
        assert any("Manim import" in issue for issue in issues)
    
    @pytest.mark.asyncio
    async def test_validate_generated_code_no_scene_class(self, code_generation_service):
        """Test code validation without Scene class."""
        code_without_scene = """
from manim import *

def some_function():
    pass
        """
        
        is_valid, issues = await code_generation_service.validate_generated_code(code_without_scene, 1)
        
        assert is_valid is False
        assert any("Scene class" in issue for issue in issues)
    
    @pytest.mark.asyncio
    async def test_validate_generated_code_syntax_error(self, code_generation_service):
        """Test code validation with syntax error."""
        code_with_syntax_error = """
from manim import *

class TestScene(Scene):
    def construct(self):
        text = Text("Hello, World!"  # Missing closing parenthesis
        """
        
        is_valid, issues = await code_generation_service.validate_generated_code(code_with_syntax_error, 1)
        
        assert is_valid is False
        assert any("Syntax error" in issue for issue in issues)
    
    @pytest.mark.asyncio
    async def test_extract_code_with_retries_success(self, code_generation_service):
        """Test successful code extraction."""
        response_text = """
Here is the generated code:

```python
from manim import *

class TestScene(Scene):
    def construct(self):
        text = Text("Hello, World!")
        self.play(Write(text))
```

This code creates a simple text animation.
        """
        
        extracted_code = await code_generation_service.extract_code_with_retries(response_text)
        
        assert "from manim import *" in extracted_code
        assert "class TestScene(Scene)" in extracted_code
        assert "def construct(self)" in extracted_code
    
    @pytest.mark.asyncio
    async def test_extract_code_with_retries_alternative_pattern(self, code_generation_service):
        """Test code extraction with alternative pattern."""
        response_text = """
Here is the code:

```
from manim import *

class TestScene(Scene):
    def construct(self):
        pass
```
        """
        
        extracted_code = await code_generation_service.extract_code_with_retries(response_text)
        
        assert "from manim import *" in extracted_code
        assert "class TestScene(Scene)" in extracted_code
    
    @pytest.mark.asyncio
    async def test_extract_code_with_retries_failure(self, code_generation_service):
        """Test code extraction failure."""
        response_text = "No code blocks found in this response."
        
        with pytest.raises(ValueError, match="Failed to extract code after .* attempts"):
            await code_generation_service.extract_code_with_retries(response_text, max_retries=2)
    
    def test_get_code_generation_metrics(self, code_generation_service):
        """Test code generation metrics retrieval."""
        # Add some test data
        code_generation_service.config['visual_analysis_results'] = {1: {'success': True}}
        code_generation_service.config['rag_context'] = {'context_scene_1': 'test context'}
        
        metrics = code_generation_service.get_code_generation_metrics()
        
        assert metrics['service_name'] == 'CodeGenerationService'
        assert 'config' in metrics
        assert metrics['config']['use_rag'] is True
        assert metrics['config']['use_visual_fix_code'] is True
        assert metrics['code_generator_initialized'] is False
        assert len(metrics['visual_analysis_results']) == 1
        assert metrics['rag_context_cache'] == 1
    
    @pytest.mark.asyncio
    async def test_cleanup(self, code_generation_service):
        """Test service cleanup."""
        # Add some test data to clean up
        code_generation_service.config['rag_context'] = {'test': 'data'}
        code_generation_service.config['visual_analysis_results'] = {'test': 'data'}
        
        await code_generation_service.cleanup()
        
        assert len(code_generation_service.config.get('rag_context', {})) == 0
        assert len(code_generation_service.config.get('visual_analysis_results', {})) == 0
    
    def test_destructor(self, code_generation_service):
        """Test service destructor."""
        # Add some test data
        code_generation_service.config['rag_context'] = {'test': 'data'}
        code_generation_service.config['visual_analysis_results'] = {'test': 'data'}
        
        # Call destructor
        code_generation_service.__del__()
        
        # Should not raise exception and should clean up data
        assert 'rag_context' not in code_generation_service.config
        assert 'visual_analysis_results' not in code_generation_service.config
"""
Test Enhanced LangGraph Agents with AWS Integration

Test script to verify the enhanced RendererAgent and CodeGeneratorAgent
work correctly with AWS S3 upload and code management capabilities.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from langgraph_agents.agents.enhanced_renderer_agent import EnhancedRendererAgent
from langgraph_agents.agents.enhanced_code_generator_agent import EnhancedCodeGeneratorAgent
from langgraph_agents.state import VideoGenerationState, AgentConfig
from aws.config import AWSConfig
from aws.aws_integration_service import AWSIntegrationService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockSystemConfig:
    """Mock system configuration for testing."""
    
    def __init__(self):
        self.output_dir = "test_output"
        self.enable_logging = True
        self.log_level = "INFO"


async def test_enhanced_renderer_agent():
    """Test the EnhancedRendererAgent with AWS integration."""
    logger.info("Testing EnhancedRendererAgent...")
    
    # Create mock configurations
    agent_config = AgentConfig(
        name="enhanced_renderer_test",
        agent_type="enhanced_renderer",
        enabled=True
    )
    
    system_config = MockSystemConfig()
    
    # Create AWS config (will be disabled if credentials not available)
    aws_config = None
    try:
        aws_config = AWSConfig()
        logger.info("AWS configuration loaded successfully")
    except Exception as e:
        logger.warning(f"AWS configuration not available: {e}")
    
    # Initialize enhanced renderer agent
    try:
        agent = EnhancedRendererAgent(agent_config, system_config, aws_config)
        logger.info(f"EnhancedRendererAgent initialized - AWS enabled: {agent.aws_enabled}")
        
        # Create test state
        test_state = VideoGenerationState(
            topic="Test Video",
            description="A test video for enhanced renderer",
            session_id="test_session_123",
            video_id="test_video_123",
            project_id="test_project",
            generated_code={
                1: """
from manim import *

class TestScene(Scene):
    def construct(self):
        text = Text("Hello, World!")
        self.play(Write(text))
        self.wait(2)
""",
                2: """
from manim import *

class TestScene2(Scene):
    def construct(self):
        circle = Circle()
        self.play(Create(circle))
        self.wait(2)
"""
            },
            scene_implementations={
                1: "Create a simple text animation",
                2: "Create a circle animation"
            },
            output_dir="test_output",
            enable_aws_upload=True,
            default_quality="low",  # Use low quality for faster testing
            max_concurrent_renders=1
        )
        
        # Test status methods
        status = agent.get_upload_status(test_state)
        logger.info(f"Initial upload status: {status}")
        
        # Test progress callback
        def progress_callback(video_id, bytes_uploaded, total_bytes, percentage):
            logger.info(f"Upload progress for {video_id}: {percentage:.1f}%")
        
        agent.add_upload_progress_callback(progress_callback)
        
        logger.info("EnhancedRendererAgent test completed successfully")
        
    except Exception as e:
        logger.error(f"EnhancedRendererAgent test failed: {e}")
        raise


async def test_enhanced_code_generator_agent():
    """Test the EnhancedCodeGeneratorAgent with AWS integration."""
    logger.info("Testing EnhancedCodeGeneratorAgent...")
    
    # Create mock configurations
    agent_config = AgentConfig(
        name="enhanced_code_generator_test",
        agent_type="enhanced_code_generator",
        enabled=True
    )
    
    system_config = MockSystemConfig()
    
    # Create AWS config (will be disabled if credentials not available)
    aws_config = None
    try:
        aws_config = AWSConfig()
        logger.info("AWS configuration loaded successfully")
    except Exception as e:
        logger.warning(f"AWS configuration not available: {e}")
    
    # Initialize enhanced code generator agent
    try:
        agent = EnhancedCodeGeneratorAgent(agent_config, system_config, aws_config)
        logger.info(f"EnhancedCodeGeneratorAgent initialized - AWS enabled: {agent.aws_enabled}")
        
        # Create test state
        test_state = VideoGenerationState(
            topic="Test Code Generation",
            description="A test for enhanced code generation",
            session_id="test_session_456",
            video_id="test_video_456",
            project_id="test_project",
            scene_outline="Scene 1: Introduction\nScene 2: Main content",
            scene_implementations={
                1: "Create an introduction with text animation",
                2: "Show mathematical concepts with equations"
            },
            enable_aws_code_management=True,
            editing_existing_video=False,
            use_rag=False  # Disable RAG for testing
        )
        
        # Test status methods
        status = agent.get_code_management_status(test_state)
        logger.info(f"Initial code management status: {status}")
        
        # Test code history (should be empty for new video)
        history = await agent.get_code_history("test_video_456")
        logger.info(f"Code history: {history}")
        
        logger.info("EnhancedCodeGeneratorAgent test completed successfully")
        
    except Exception as e:
        logger.error(f"EnhancedCodeGeneratorAgent test failed: {e}")
        raise


async def test_aws_integration_service():
    """Test the AWS Integration Service."""
    logger.info("Testing AWS Integration Service...")
    
    try:
        # Create AWS config
        aws_config = AWSConfig()
        
        # Initialize integration service
        integration_service = AWSIntegrationService(aws_config)
        logger.info("AWS Integration Service initialized successfully")
        
        # Test service info
        service_info = integration_service.get_service_info()
        logger.info(f"Service info: {service_info}")
        
        # Test health check
        health_status = await integration_service.health_check()
        logger.info(f"Health status: {health_status}")
        
        logger.info("AWS Integration Service test completed successfully")
        
    except Exception as e:
        logger.warning(f"AWS Integration Service test failed (expected if no AWS credentials): {e}")


async def test_agent_integration():
    """Test integration between enhanced agents."""
    logger.info("Testing agent integration...")
    
    try:
        # Create shared AWS config
        aws_config = None
        try:
            aws_config = AWSConfig()
        except Exception:
            logger.warning("AWS config not available, testing without AWS integration")
        
        # Create configurations
        system_config = MockSystemConfig()
        
        code_agent_config = AgentConfig(
            name="code_generator",
            agent_type="enhanced_code_generator",
            enabled=True
        )
        
        render_agent_config = AgentConfig(
            name="renderer",
            agent_type="enhanced_renderer",
            enabled=True
        )
        
        # Initialize agents
        code_agent = EnhancedCodeGeneratorAgent(code_agent_config, system_config, aws_config)
        render_agent = EnhancedRendererAgent(render_agent_config, system_config, aws_config)
        
        # Create shared state
        shared_state = VideoGenerationState(
            topic="Integration Test",
            description="Testing agent integration",
            session_id="integration_test_789",
            video_id="integration_video_789",
            project_id="integration_project",
            scene_outline="Scene 1: Test scene",
            scene_implementations={
                1: "Create a simple test animation"
            },
            enable_aws_upload=True,
            enable_aws_code_management=True
        )
        
        # Test that both agents can access the same state
        code_status = code_agent.get_code_management_status(shared_state)
        render_status = render_agent.get_upload_status(shared_state)
        
        logger.info(f"Code agent status: AWS enabled = {code_status['aws_enabled']}")
        logger.info(f"Render agent status: AWS enabled = {render_status['aws_enabled']}")
        
        # Verify both agents have consistent AWS configuration
        assert code_agent.aws_enabled == render_agent.aws_enabled, "AWS enablement should be consistent"
        
        logger.info("Agent integration test completed successfully")
        
    except Exception as e:
        logger.error(f"Agent integration test failed: {e}")
        raise


async def main():
    """Run all tests."""
    logger.info("Starting Enhanced LangGraph Agents Tests")
    
    try:
        # Test individual agents
        await test_enhanced_renderer_agent()
        await test_enhanced_code_generator_agent()
        
        # Test AWS integration service
        await test_aws_integration_service()
        
        # Test agent integration
        await test_agent_integration()
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Tests failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
"""
Simple Test for Enhanced LangGraph Agents

Basic test to verify the enhanced agents can be imported and initialized.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that enhanced agents can be imported."""
    logger.info("Testing imports...")
    
    try:
        from langgraph_agents.agents.enhanced_renderer_agent import EnhancedRendererAgent
        logger.info("✅ EnhancedRendererAgent imported successfully")
        
        from langgraph_agents.agents.enhanced_code_generator_agent import EnhancedCodeGeneratorAgent
        logger.info("✅ EnhancedCodeGeneratorAgent imported successfully")
        
        from aws.aws_integration_service import AWSIntegrationService
        logger.info("✅ AWSIntegrationService imported successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False


def test_basic_initialization():
    """Test basic initialization without AWS."""
    logger.info("Testing basic initialization...")
    
    try:
        # Mock configurations
        class MockConfig:
            def __init__(self):
                self.name = "test_agent"
                self.agent_type = "test"
                self.enabled = True
                self.max_retries = 3
                self.timeout_seconds = 300
                self.enable_human_loop = False
                self.model_config = {}
                self.planner_model = None
                self.scene_model = None
                self.helper_model = None
        
        class MockSystemConfig:
            def __init__(self):
                self.output_dir = "test_output"
        
        config = MockConfig()
        system_config = MockSystemConfig()
        
        # Test EnhancedRendererAgent without AWS
        from langgraph_agents.agents.enhanced_renderer_agent import EnhancedRendererAgent
        renderer_agent = EnhancedRendererAgent(config, system_config, aws_config=None)
        logger.info(f"✅ EnhancedRendererAgent initialized - AWS enabled: {renderer_agent.aws_enabled}")
        
        # Test EnhancedCodeGeneratorAgent without AWS
        from langgraph_agents.agents.enhanced_code_generator_agent import EnhancedCodeGeneratorAgent
        code_agent = EnhancedCodeGeneratorAgent(config, system_config, aws_config=None)
        logger.info(f"✅ EnhancedCodeGeneratorAgent initialized - AWS enabled: {code_agent.aws_enabled}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return False


def test_aws_config_loading():
    """Test AWS configuration loading."""
    logger.info("Testing AWS configuration loading...")
    
    try:
        from aws.config import AWSConfig
        
        # Try to load AWS config
        try:
            aws_config = AWSConfig()
            logger.info("✅ AWS configuration loaded successfully")
            logger.info(f"   Region: {aws_config.region}")
            logger.info(f"   Video bucket: {aws_config.video_bucket_name}")
            logger.info(f"   Code bucket: {aws_config.code_bucket_name}")
            return True
        except Exception as e:
            logger.warning(f"⚠️  AWS configuration not available: {e}")
            return False
            
    except ImportError as e:
        logger.error(f"❌ AWS config import failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("Starting Enhanced LangGraph Agents Simple Tests")
    
    success_count = 0
    total_tests = 3
    
    # Test imports
    if test_imports():
        success_count += 1
    
    # Test basic initialization
    if test_basic_initialization():
        success_count += 1
    
    # Test AWS config loading
    if test_aws_config_loading():
        success_count += 1
    
    logger.info(f"Tests completed: {success_count}/{total_tests} passed")
    
    if success_count == total_tests:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.warning(f"⚠️  {total_tests - success_count} tests failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
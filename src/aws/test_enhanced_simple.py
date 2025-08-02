#!/usr/bin/env python3
"""
Simple test for Enhanced CodeGeneratorAgent imports and basic functionality.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from langgraph_agents.agents.enhanced_code_generator_agent import EnhancedCodeGeneratorAgent
        print("✓ EnhancedCodeGeneratorAgent imported successfully")
    except Exception as e:
        print(f"✗ Failed to import EnhancedCodeGeneratorAgent: {e}")
        return False
    
    try:
        from aws.s3_code_storage import S3CodeStorageService, CodeMetadata
        print("✓ S3CodeStorageService imported successfully")
    except Exception as e:
        print(f"✗ Failed to import S3CodeStorageService: {e}")
        return False
    
    try:
        from aws.config import AWSConfig
        print("✓ AWSConfig imported successfully")
    except Exception as e:
        print(f"✗ Failed to import AWSConfig: {e}")
        return False
    
    return True

def test_basic_initialization():
    """Test basic initialization without AWS."""
    print("\nTesting basic initialization...")
    
    try:
        from langgraph_agents.agents.enhanced_code_generator_agent import EnhancedCodeGeneratorAgent
        
        # Create basic config with all required attributes
        agent_config = type('Config', (), {
            'name': 'test_agent',
            'max_retries': 2,
            'timeout_seconds': 30,
            'enable_human_loop': False,
            'model_config': {}
        })()
        
        system_config = {
            'output_dir': 'test_output',
            'use_rag': False
        }
        
        # Initialize without AWS
        agent = EnhancedCodeGeneratorAgent(
            config=agent_config,
            system_config=system_config,
            aws_config=None
        )
        
        print(f"✓ Agent initialized - AWS enabled: {agent.aws_enabled}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to initialize agent: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_code_metadata():
    """Test CodeMetadata creation."""
    print("\nTesting CodeMetadata...")
    
    try:
        from aws.s3_code_storage import CodeMetadata
        from datetime import datetime
        
        # Test basic metadata
        metadata = CodeMetadata(
            video_id='test_video',
            project_id='test_project',
            version=1
        )
        print(f"✓ Basic metadata created: {metadata.video_id} v{metadata.version}")
        
        # Test with scene number
        metadata_scene = CodeMetadata(
            video_id='test_video',
            project_id='test_project',
            version=1,
            scene_number=1
        )
        print(f"✓ Scene metadata created: {metadata_scene.video_id} v{metadata_scene.version} scene{metadata_scene.scene_number}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to create CodeMetadata: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run simple tests."""
    print("Enhanced CodeGeneratorAgent - Simple Tests")
    print("=" * 50)
    
    test1 = test_imports()
    test2 = test_basic_initialization()
    test3 = test_code_metadata()
    
    print("\n" + "=" * 50)
    print("RESULTS:")
    print(f"Imports: {'✓ PASSED' if test1 else '✗ FAILED'}")
    print(f"Basic Init: {'✓ PASSED' if test2 else '✗ FAILED'}")
    print(f"CodeMetadata: {'✓ PASSED' if test3 else '✗ FAILED'}")
    
    overall = test1 and test2 and test3
    print(f"Overall: {'✓ ALL PASSED' if overall else '✗ SOME FAILED'}")
    
    return overall

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
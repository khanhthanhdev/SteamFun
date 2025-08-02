#!/usr/bin/env python3
"""
Minimal test for Enhanced CodeGeneratorAgent.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    print("Testing minimal import...")
    
    try:
        # Test just the import
        from langgraph_agents.agents.enhanced_code_generator_agent import EnhancedCodeGeneratorAgent
        print("✓ Import successful")
        
        # Test CodeMetadata
        from aws.s3_code_storage import CodeMetadata
        metadata = CodeMetadata(
            video_id='test',
            project_id='test',
            version=1
        )
        print(f"✓ CodeMetadata created: {metadata.video_id}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print(f"Result: {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
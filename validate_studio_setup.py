"""
Simple validation script for Studio integration.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def validate_studio_integration():
    """Validate Studio integration setup."""
    print("Validating Studio Integration Setup...")
    print("=" * 50)
    
    try:
        # Test 1: Import modules
        print("1. Testing module imports...")
        from src.langgraph_agents.studio.studio_workflow_config import (
            create_studio_workflow_config,
            get_studio_workflow_info
        )
        print("   ✅ Studio workflow config imported successfully")
        
        from src.langgraph_agents.studio.studio_workflow_visualization import (
            get_studio_visualizer,
            get_studio_inspector
        )
        print("   ✅ Studio visualization modules imported successfully")
        
        from src.langgraph_agents.studio.studio_server_integration import (
            create_studio_server
        )
        print("   ✅ Studio server integration imported successfully")
        
        # Test 2: Create configuration
        print("\n2. Testing configuration creation...")
        config = create_studio_workflow_config()
        print("   ✅ Studio workflow config created")
        
        # Test 3: Create workflow
        print("\n3. Testing workflow creation...")
        workflow = config.create_studio_compatible_workflow()
        print("   ✅ Studio-compatible workflow created")
        print(f"   - Graph compiled: {workflow.graph is not None}")
        print(f"   - Checkpointing enabled: {workflow.checkpointer is not None}")
        
        # Test 4: Test schemas
        print("\n4. Testing schema generation...")
        schema = config.get_workflow_schema()
        input_props = len(schema.get("input_schema", {}).get("properties", {}))
        output_props = len(schema.get("output_schema", {}).get("properties", {}))
        node_schemas = len(schema.get("node_schemas", {}))
        print(f"   ✅ Workflow schema generated")
        print(f"   - Input properties: {input_props}")
        print(f"   - Output properties: {output_props}")
        print(f"   - Node schemas: {node_schemas}")
        
        # Test 5: Test visualization
        print("\n5. Testing visualization...")
        visualizer = get_studio_visualizer()
        mermaid = visualizer.generate_mermaid_diagram()
        metadata = visualizer.create_studio_graph_metadata()
        print("   ✅ Visualization components created")
        print(f"   - Mermaid diagram length: {len(mermaid)} characters")
        print(f"   - Graph metadata nodes: {metadata.get('graph_info', {}).get('node_count', 0)}")
        
        # Test 6: Test server integration
        print("\n6. Testing server integration...")
        server_config = config.setup_studio_server_integration()
        print("   ✅ Server integration configured")
        print(f"   - Status: {server_config.get('status')}")
        print(f"   - Endpoint count: {server_config.get('endpoint_count')}")
        
        # Test 7: Test comprehensive info
        print("\n7. Testing comprehensive workflow info...")
        workflow_info = get_studio_workflow_info()
        print("   ✅ Comprehensive workflow info generated")
        print(f"   - Workflow type: {workflow_info.get('workflow_type')}")
        print(f"   - Has visualization: {'studio_visualization' in workflow_info}")
        print(f"   - Has state inspection: {'state_inspection' in workflow_info}")
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED - Studio integration is ready!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = validate_studio_integration()
    sys.exit(0 if success else 1)
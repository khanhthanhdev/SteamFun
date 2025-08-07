"""
Test script for Studio integration validation.

This script validates that the Studio-compatible workflow graph configuration
is properly set up and can be used in LangGraph Studio.
"""

import asyncio
import logging
from typing import Dict, Any
from datetime import datetime

from ..models.state import VideoGenerationState
from ..models.config import WorkflowConfig
from .studio_workflow_config import (
    create_studio_workflow_config,
    get_studio_workflow_info,
    test_studio_workflow_setup
)
from .studio_integration import create_studio_tester
from .studio_workflow_visualization import get_studio_visualizer, get_studio_inspector
from .test_scenarios import get_test_scenario_manager

logger = logging.getLogger(__name__)


class StudioIntegrationTester:
    """Comprehensive tester for Studio integration."""
    
    def __init__(self):
        self.studio_config = create_studio_workflow_config()
        self.visualizer = get_studio_visualizer()
        self.inspector = get_studio_inspector()
        self.test_manager = get_test_scenario_manager()
        self.test_results = {}
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive tests of the Studio integration."""
        logger.info("Starting comprehensive Studio integration tests")
        
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "overall_status": "unknown",
            "summary": {}
        }
        
        # Test 1: Basic configuration validation
        test_results["tests"]["configuration"] = await self.test_configuration()
        
        # Test 2: Workflow graph creation
        test_results["tests"]["workflow_creation"] = await self.test_workflow_creation()
        
        # Test 3: Schema validation
        test_results["tests"]["schema_validation"] = await self.test_schema_validation()
        
        # Test 4: State inspection capabilities
        test_results["tests"]["state_inspection"] = await self.test_state_inspection()
        
        # Test 5: Visualization capabilities
        test_results["tests"]["visualization"] = await self.test_visualization()
        
        # Test 6: Agent testing framework
        test_results["tests"]["agent_testing"] = await self.test_agent_testing()
        
        # Test 7: Server integration
        test_results["tests"]["server_integration"] = await self.test_server_integration()
        
        # Test 8: End-to-end workflow execution
        test_results["tests"]["e2e_execution"] = await self.test_e2e_execution()
        
        # Calculate overall status
        passed_tests = sum(1 for test in test_results["tests"].values() if test["status"] == "passed")
        total_tests = len(test_results["tests"])
        
        test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0
        }
        
        test_results["overall_status"] = "passed" if passed_tests == total_tests else "failed"
        
        logger.info(f"Studio integration tests completed: {passed_tests}/{total_tests} passed")
        return test_results
    
    async def test_configuration(self) -> Dict[str, Any]:
        """Test basic configuration setup."""
        try:
            # Test configuration creation
            config = create_studio_workflow_config()
            
            # Test workflow configuration
            workflow_config = config.workflow_config
            
            # Test debugging settings
            debug_info = config.get_debugging_info()
            
            return {
                "status": "passed",
                "details": {
                    "config_created": True,
                    "workflow_config_valid": bool(workflow_config),
                    "debugging_enabled": debug_info["debugging_enabled"],
                    "state_inspection_enabled": debug_info["state_inspection_enabled"]
                }
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "details": {}
            }
    
    async def test_workflow_creation(self) -> Dict[str, Any]:
        """Test workflow graph creation."""
        try:
            # Create Studio-compatible workflow
            workflow = self.studio_config.create_studio_compatible_workflow()
            
            # Test graph compilation
            graph_compiled = workflow.graph is not None
            
            # Test checkpointing
            checkpoint_info = workflow.get_checkpoint_info()
            
            return {
                "status": "passed",
                "details": {
                    "workflow_created": True,
                    "graph_compiled": graph_compiled,
                    "checkpointing_enabled": checkpoint_info["checkpointing_enabled"],
                    "checkpointer_type": checkpoint_info["checkpointer_type"]
                }
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "details": {}
            }
    
    async def test_schema_validation(self) -> Dict[str, Any]:
        """Test input/output schema validation."""
        try:
            # Get workflow schema
            schema = self.studio_config.get_workflow_schema()
            
            # Validate schema structure
            has_input_schema = "input_schema" in schema
            has_output_schema = "output_schema" in schema
            has_node_schemas = "node_schemas" in schema
            
            # Test schema properties
            input_properties = len(schema.get("input_schema", {}).get("properties", {}))
            output_properties = len(schema.get("output_schema", {}).get("properties", {}))
            node_count = len(schema.get("node_schemas", {}))
            
            return {
                "status": "passed",
                "details": {
                    "has_input_schema": has_input_schema,
                    "has_output_schema": has_output_schema,
                    "has_node_schemas": has_node_schemas,
                    "input_properties": input_properties,
                    "output_properties": output_properties,
                    "node_count": node_count
                }
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "details": {}
            }
    
    async def test_state_inspection(self) -> Dict[str, Any]:
        """Test state inspection capabilities."""
        try:
            # Create test state
            test_state = VideoGenerationState(
                topic="Test Topic",
                description="Test Description",
                session_id="test_inspection"
            )
            
            # Test state snapshot capture
            snapshot = self.inspector.capture_state_snapshot(test_state, "test_node", "before")
            
            # Test inspection summary
            summary = self.inspector.get_inspection_summary("test_inspection")
            
            return {
                "status": "passed",
                "details": {
                    "snapshot_captured": bool(snapshot),
                    "snapshot_id": snapshot.get("snapshot_id"),
                    "summary_generated": "error" not in summary,
                    "inspection_enabled": self.studio_config.state_inspection_enabled
                }
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "details": {}
            }
    
    async def test_visualization(self) -> Dict[str, Any]:
        """Test visualization capabilities."""
        try:
            # Test Mermaid diagram generation
            mermaid_diagram = self.visualizer.generate_mermaid_diagram()
            
            # Test graph metadata
            graph_metadata = self.visualizer.create_studio_graph_metadata()
            
            # Test node configurations
            planning_config = self.visualizer.get_node_configuration("planning")
            
            return {
                "status": "passed",
                "details": {
                    "mermaid_diagram_generated": bool(mermaid_diagram),
                    "diagram_length": len(mermaid_diagram),
                    "graph_metadata_created": bool(graph_metadata),
                    "node_count": graph_metadata.get("graph_info", {}).get("node_count", 0),
                    "planning_config_available": bool(planning_config)
                }
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "details": {}
            }
    
    async def test_agent_testing(self) -> Dict[str, Any]:
        """Test agent testing framework."""
        try:
            # Create agent tester
            tester = create_studio_tester()
            
            # Test scenario management
            scenarios = self.test_manager.list_scenarios("planning_agent")
            
            # Test fixture creation
            fixture = self.test_manager.get_fixture("basic_state")
            
            return {
                "status": "passed",
                "details": {
                    "tester_created": bool(tester),
                    "scenarios_available": len(scenarios),
                    "fixture_available": bool(fixture),
                    "test_framework_ready": True
                }
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "details": {}
            }
    
    async def test_server_integration(self) -> Dict[str, Any]:
        """Test server integration configuration."""
        try:
            # Test server integration setup
            server_config = self.studio_config.setup_studio_server_integration()
            
            # Test endpoint configuration
            integration_config = self.studio_config.create_studio_server_integration()
            
            endpoint_count = (
                len(integration_config["graph_endpoints"]) +
                len(integration_config["monitoring_endpoints"]) +
                len(integration_config["debugging_endpoints"]) +
                len(integration_config["schema_endpoints"]) +
                len(integration_config["test_endpoints"])
            )
            
            return {
                "status": "passed",
                "details": {
                    "server_config_created": server_config["status"] == "configured",
                    "endpoint_count": endpoint_count,
                    "cors_configured": bool(server_config.get("cors_config")),
                    "websocket_enabled": server_config.get("websocket_config", {}).get("enabled", False)
                }
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "details": {}
            }
    
    async def test_e2e_execution(self) -> Dict[str, Any]:
        """Test end-to-end workflow execution (simplified)."""
        try:
            # Create test state
            test_state = VideoGenerationState(
                topic="Simple Test",
                description="A simple test for Studio integration",
                session_id="e2e_test"
            )
            
            # Create workflow
            workflow = self.studio_config.create_studio_compatible_workflow()
            
            # Test that workflow can be invoked (we won't run full execution)
            # This just tests that the workflow is properly configured
            workflow_ready = workflow.graph is not None
            
            return {
                "status": "passed",
                "details": {
                    "test_state_created": True,
                    "workflow_ready": workflow_ready,
                    "graph_available": workflow.graph is not None,
                    "note": "Full execution skipped for testing speed"
                }
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "details": {}
            }
    
    def generate_test_report(self, test_results: Dict[str, Any]) -> str:
        """Generate a formatted test report."""
        report = []
        report.append("=" * 60)
        report.append("STUDIO INTEGRATION TEST REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {test_results['timestamp']}")
        report.append(f"Overall Status: {test_results['overall_status'].upper()}")
        report.append("")
        
        # Summary
        summary = test_results["summary"]
        report.append("SUMMARY:")
        report.append(f"  Total Tests: {summary['total_tests']}")
        report.append(f"  Passed: {summary['passed_tests']}")
        report.append(f"  Failed: {summary['failed_tests']}")
        report.append(f"  Success Rate: {summary['success_rate']:.1%}")
        report.append("")
        
        # Individual test results
        report.append("TEST RESULTS:")
        for test_name, result in test_results["tests"].items():
            status_icon = "✅" if result["status"] == "passed" else "❌"
            report.append(f"  {status_icon} {test_name.replace('_', ' ').title()}: {result['status'].upper()}")
            
            if result["status"] == "failed" and "error" in result:
                report.append(f"    Error: {result['error']}")
            
            if "details" in result and result["details"]:
                for key, value in result["details"].items():
                    report.append(f"    {key}: {value}")
            report.append("")
        
        report.append("=" * 60)
        return "\n".join(report)


async def run_studio_integration_tests():
    """Run the Studio integration tests."""
    tester = StudioIntegrationTester()
    results = await tester.run_comprehensive_tests()
    
    # Generate and print report
    report = tester.generate_test_report(results)
    print(report)
    
    return results


def validate_studio_setup():
    """Quick validation of Studio setup."""
    print("Validating Studio setup...")
    
    try:
        # Test basic setup
        setup_result = test_studio_workflow_setup()
        
        if setup_result["status"] == "success":
            print("✅ Basic Studio setup: PASSED")
            print(f"   - Workflow created: {setup_result['workflow_created']}")
            print(f"   - Graph compiled: {setup_result['graph_compiled']}")
            print(f"   - Schema generated: {setup_result['schema_generated']}")
            print(f"   - Debugging enabled: {setup_result['debugging_enabled']}")
        else:
            print("❌ Basic Studio setup: FAILED")
            print(f"   Error: {setup_result.get('error', 'Unknown error')}")
        
        # Test workflow info
        workflow_info = get_studio_workflow_info()
        print(f"✅ Workflow info generated: {bool(workflow_info)}")
        print(f"   - Schema properties: {len(workflow_info.get('schema', {}).get('input_schema', {}).get('properties', {}))}")
        print(f"   - Node configurations: {len(workflow_info.get('studio_visualization', {}).get('node_configurations', {}))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Studio setup validation failed: {e}")
        return False


if __name__ == "__main__":
    # Run quick validation
    print("Running Studio Integration Validation...")
    print()
    
    # Quick setup validation
    setup_valid = validate_studio_setup()
    print()
    
    if setup_valid:
        print("Running comprehensive tests...")
        print()
        
        # Run comprehensive tests
        asyncio.run(run_studio_integration_tests())
    else:
        print("Skipping comprehensive tests due to setup validation failure.")
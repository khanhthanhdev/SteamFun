#!/usr/bin/env python3
"""
Deployment test script for LangGraph video generation workflow.

This script tests the deployment by checking various endpoints and
validating that the system is working correctly.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
import requests
import argparse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langgraph_agents.monitoring.health_check import perform_health_check
from langgraph_agents.config.validation import validate_config_from_file
from langgraph_agents.models.config import WorkflowConfig


class DeploymentTester:
    """Test deployment functionality."""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.timeout = timeout
    
    def test_basic_connectivity(self) -> Dict[str, Any]:
        """Test basic API connectivity."""
        print("🔗 Testing basic connectivity...")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                return {
                    "status": "success",
                    "message": "Basic connectivity successful",
                    "response_time_ms": response.elapsed.total_seconds() * 1000,
                    "data": response.json()
                }
            else:
                return {
                    "status": "error",
                    "message": f"HTTP {response.status_code}: {response.text}",
                    "response_time_ms": response.elapsed.total_seconds() * 1000
                }
                
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"Connection failed: {e}"
            }
    
    def test_workflow_health(self) -> Dict[str, Any]:
        """Test workflow-specific health endpoint."""
        print("🔧 Testing workflow health...")
        
        try:
            response = self.session.get(f"{self.base_url}/health/workflow")
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "success",
                    "message": "Workflow health check successful",
                    "response_time_ms": response.elapsed.total_seconds() * 1000,
                    "data": data
                }
            else:
                return {
                    "status": "error",
                    "message": f"HTTP {response.status_code}: {response.text}",
                    "response_time_ms": response.elapsed.total_seconds() * 1000
                }
                
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"Workflow health check failed: {e}"
            }
    
    def test_api_documentation(self) -> Dict[str, Any]:
        """Test API documentation endpoint."""
        print("📚 Testing API documentation...")
        
        try:
            response = self.session.get(f"{self.base_url}/docs")
            
            if response.status_code == 200:
                return {
                    "status": "success",
                    "message": "API documentation accessible",
                    "response_time_ms": response.elapsed.total_seconds() * 1000
                }
            else:
                return {
                    "status": "error",
                    "message": f"HTTP {response.status_code}: {response.text}",
                    "response_time_ms": response.elapsed.total_seconds() * 1000
                }
                
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"Documentation access failed: {e}"
            }
    
    def test_openapi_schema(self) -> Dict[str, Any]:
        """Test OpenAPI schema endpoint."""
        print("🔍 Testing OpenAPI schema...")
        
        try:
            response = self.session.get(f"{self.base_url}/openapi.json")
            
            if response.status_code == 200:
                schema = response.json()
                return {
                    "status": "success",
                    "message": "OpenAPI schema accessible",
                    "response_time_ms": response.elapsed.total_seconds() * 1000,
                    "data": {
                        "title": schema.get("info", {}).get("title", "Unknown"),
                        "version": schema.get("info", {}).get("version", "Unknown"),
                        "paths_count": len(schema.get("paths", {}))
                    }
                }
            else:
                return {
                    "status": "error",
                    "message": f"HTTP {response.status_code}: {response.text}",
                    "response_time_ms": response.elapsed.total_seconds() * 1000
                }
                
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"OpenAPI schema access failed: {e}"
            }
    
    def test_configuration_validation(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Test configuration validation."""
        print("⚙️ Testing configuration validation...")
        
        try:
            if config_path is None:
                # Try common configuration paths
                possible_paths = [
                    "config/runtime/workflow.yaml",
                    "config/templates/development.yaml",
                    "config/templates/production.yaml"
                ]
                
                config_path = None
                for path in possible_paths:
                    if Path(path).exists():
                        config_path = path
                        break
                
                if config_path is None:
                    return {
                        "status": "error",
                        "message": "No configuration file found"
                    }
            
            validate_config_from_file(config_path)
            
            return {
                "status": "success",
                "message": f"Configuration validation successful: {config_path}"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Configuration validation failed: {e}"
            }
    
    async def test_comprehensive_health(self) -> Dict[str, Any]:
        """Test comprehensive health check."""
        print("🏥 Testing comprehensive health check...")
        
        try:
            # Load configuration
            config = WorkflowConfig()
            
            # Perform health check
            health_result = await perform_health_check(config)
            
            return {
                "status": "success",
                "message": "Comprehensive health check completed",
                "data": health_result
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Comprehensive health check failed: {e}"
            }
    
    def test_database_connectivity(self) -> Dict[str, Any]:
        """Test database connectivity."""
        print("🗄️ Testing database connectivity...")
        
        try:
            import psycopg2
            import os
            
            db_url = os.getenv("DATABASE_URL", "")
            if not db_url:
                return {
                    "status": "warning",
                    "message": "DATABASE_URL not set, skipping database test"
                }
            
            # Parse connection string for psycopg2
            if db_url.startswith("postgresql+asyncpg://"):
                db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
            
            conn = psycopg2.connect(db_url)
            cursor = conn.cursor()
            cursor.execute("SELECT version()")
            version = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            return {
                "status": "success",
                "message": "Database connectivity successful",
                "data": {"version": version}
            }
            
        except ImportError:
            return {
                "status": "warning",
                "message": "psycopg2 not available, skipping database test"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Database connectivity failed: {e}"
            }
    
    def test_redis_connectivity(self) -> Dict[str, Any]:
        """Test Redis connectivity."""
        print("🔄 Testing Redis connectivity...")
        
        try:
            import redis
            import os
            
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            
            r = redis.from_url(redis_url)
            r.ping()
            
            info = r.info()
            
            return {
                "status": "success",
                "message": "Redis connectivity successful",
                "data": {
                    "version": info.get("redis_version", "unknown"),
                    "connected_clients": info.get("connected_clients", 0)
                }
            }
            
        except ImportError:
            return {
                "status": "warning",
                "message": "redis not available, skipping Redis test"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Redis connectivity failed: {e}"
            }
    
    def run_all_tests(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Run all deployment tests."""
        print("🚀 Starting deployment tests...\n")
        
        tests = [
            ("Basic Connectivity", self.test_basic_connectivity),
            ("Workflow Health", self.test_workflow_health),
            ("API Documentation", self.test_api_documentation),
            ("OpenAPI Schema", self.test_openapi_schema),
            ("Configuration Validation", lambda: self.test_configuration_validation(config_path)),
            ("Database Connectivity", self.test_database_connectivity),
            ("Redis Connectivity", self.test_redis_connectivity),
        ]
        
        results = {}
        success_count = 0
        warning_count = 0
        error_count = 0
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                results[test_name] = result
                
                if result["status"] == "success":
                    print(f"✅ {test_name}: {result['message']}")
                    success_count += 1
                elif result["status"] == "warning":
                    print(f"⚠️  {test_name}: {result['message']}")
                    warning_count += 1
                else:
                    print(f"❌ {test_name}: {result['message']}")
                    error_count += 1
                    
            except Exception as e:
                result = {
                    "status": "error",
                    "message": f"Test execution failed: {e}"
                }
                results[test_name] = result
                print(f"❌ {test_name}: {result['message']}")
                error_count += 1
        
        # Run async comprehensive health check
        try:
            print("\n🏥 Running comprehensive health check...")
            health_result = asyncio.run(self.test_comprehensive_health())
            results["Comprehensive Health"] = health_result
            
            if health_result["status"] == "success":
                print(f"✅ Comprehensive Health: {health_result['message']}")
                success_count += 1
            else:
                print(f"❌ Comprehensive Health: {health_result['message']}")
                error_count += 1
                
        except Exception as e:
            result = {
                "status": "error",
                "message": f"Comprehensive health check failed: {e}"
            }
            results["Comprehensive Health"] = result
            print(f"❌ Comprehensive Health: {result['message']}")
            error_count += 1
        
        # Summary
        total_tests = len(results)
        print(f"\n📊 Test Summary:")
        print(f"  Total Tests: {total_tests}")
        print(f"  ✅ Successful: {success_count}")
        print(f"  ⚠️  Warnings: {warning_count}")
        print(f"  ❌ Errors: {error_count}")
        
        overall_status = "success"
        if error_count > 0:
            overall_status = "error"
        elif warning_count > 0:
            overall_status = "warning"
        
        return {
            "overall_status": overall_status,
            "summary": {
                "total": total_tests,
                "success": success_count,
                "warning": warning_count,
                "error": error_count
            },
            "results": results
        }


def main():
    """Main function to run deployment tests."""
    parser = argparse.ArgumentParser(description="Test LangGraph workflow deployment")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL for the API (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--config",
        help="Path to configuration file to validate"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds (default: 30)"
    )
    parser.add_argument(
        "--output",
        help="Output file for test results (JSON format)"
    )
    
    args = parser.parse_args()
    
    # Create tester
    tester = DeploymentTester(base_url=args.url, timeout=args.timeout)
    
    # Run tests
    results = tester.run_all_tests(config_path=args.config)
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")
    
    # Exit with appropriate code
    if results["overall_status"] == "error":
        print("\n❌ Deployment tests failed!")
        sys.exit(1)
    elif results["overall_status"] == "warning":
        print("\n⚠️  Deployment tests completed with warnings")
        sys.exit(0)
    else:
        print("\n✅ All deployment tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
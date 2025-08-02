"""
Test runner for unit tests with coverage reporting and result analysis.
"""

import pytest
import sys
import os
from pathlib import Path
import subprocess
import json
from typing import Dict, List, Any


class UnitTestRunner:
    """Runner for unit tests with comprehensive reporting."""
    
    def __init__(self, test_dir: str = "tests/unit"):
        self.test_dir = Path(test_dir)
        self.results = {}
        
    def run_all_tests(self, verbose: bool = True, coverage: bool = True) -> Dict[str, Any]:
        """Run all unit tests with optional coverage reporting.
        
        Args:
            verbose: Enable verbose output
            coverage: Enable coverage reporting
            
        Returns:
            Dict containing test results and statistics
        """
        print("🧪 Running LangGraph Multi-Agent System Unit Tests")
        print("=" * 60)
        
        # Build pytest command
        cmd = ["python", "-m", "pytest", str(self.test_dir)]
        
        if verbose:
            cmd.append("-v")
            
        if coverage:
            cmd.extend([
                "--cov=src/langgraph_agents",
                "--cov-report=term-missing",
                "--cov-report=json:tests/coverage.json"
            ])
        
        # Add markers for unit tests only
        cmd.extend(["-m", "unit"])
        
        # Run tests
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )
            
            self.results = {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
            
            # Parse coverage if enabled
            if coverage and Path("tests/coverage.json").exists():
                with open("tests/coverage.json", "r") as f:
                    coverage_data = json.load(f)
                    self.results["coverage"] = coverage_data
            
            return self.results
            
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return {"success": False, "error": str(e)}
    
    def run_specific_agent_tests(self, agent_name: str) -> Dict[str, Any]:
        """Run tests for a specific agent.
        
        Args:
            agent_name: Name of the agent to test (e.g., 'planner', 'code_generator')
            
        Returns:
            Dict containing test results
        """
        test_file = self.test_dir / f"test_{agent_name}_agent.py"
        
        if not test_file.exists():
            return {"success": False, "error": f"Test file not found: {test_file}"}
        
        print(f"🧪 Running {agent_name.title()}Agent Tests")
        print("=" * 40)
        
        cmd = [
            "python", "-m", "pytest", 
            str(test_file),
            "-v",
            "--tb=short"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )
            
            return {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
                "agent": agent_name
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "agent": agent_name}
    
    def run_base_agent_tests(self) -> Dict[str, Any]:
        """Run base agent tests specifically."""
        return self.run_specific_agent_tests("base")
    
    def generate_test_report(self) -> str:
        """Generate a comprehensive test report.
        
        Returns:
            Formatted test report string
        """
        if not self.results:
            return "No test results available. Run tests first."
        
        report = []
        report.append("📊 Unit Test Report")
        report.append("=" * 50)
        
        if self.results.get("success"):
            report.append("✅ All tests passed!")
        else:
            report.append("❌ Some tests failed!")
        
        report.append(f"Exit Code: {self.results.get('exit_code', 'N/A')}")
        report.append("")
        
        # Coverage information
        if "coverage" in self.results:
            coverage = self.results["coverage"]
            total_coverage = coverage.get("totals", {}).get("percent_covered", 0)
            report.append(f"📈 Code Coverage: {total_coverage:.1f}%")
            report.append("")
            
            # Per-file coverage
            report.append("📁 File Coverage:")
            files = coverage.get("files", {})
            for file_path, file_data in files.items():
                if "langgraph_agents" in file_path:
                    coverage_pct = file_data.get("summary", {}).get("percent_covered", 0)
                    report.append(f"  {Path(file_path).name}: {coverage_pct:.1f}%")
            report.append("")
        
        # Test output
        if self.results.get("stdout"):
            report.append("📝 Test Output:")
            report.append(self.results["stdout"])
        
        if self.results.get("stderr"):
            report.append("⚠️  Errors/Warnings:")
            report.append(self.results["stderr"])
        
        return "\n".join(report)
    
    def validate_test_structure(self) -> Dict[str, Any]:
        """Validate that all expected test files exist and are properly structured.
        
        Returns:
            Dict containing validation results
        """
        expected_test_files = [
            "test_base_agent.py",
            "test_planner_agent.py", 
            "test_code_generator_agent.py",
            "test_error_handler_agent.py",
            "test_rag_agent.py"
        ]
        
        validation_results = {
            "missing_files": [],
            "existing_files": [],
            "total_expected": len(expected_test_files),
            "total_found": 0
        }
        
        for test_file in expected_test_files:
            file_path = self.test_dir / test_file
            if file_path.exists():
                validation_results["existing_files"].append(test_file)
                validation_results["total_found"] += 1
            else:
                validation_results["missing_files"].append(test_file)
        
        validation_results["complete"] = len(validation_results["missing_files"]) == 0
        
        return validation_results
    
    def run_test_validation(self) -> None:
        """Run test structure validation and print results."""
        print("🔍 Validating Test Structure")
        print("=" * 40)
        
        validation = self.validate_test_structure()
        
        print(f"Expected test files: {validation['total_expected']}")
        print(f"Found test files: {validation['total_found']}")
        
        if validation["complete"]:
            print("✅ All expected test files found!")
        else:
            print("❌ Missing test files:")
            for missing_file in validation["missing_files"]:
                print(f"  - {missing_file}")
        
        print("\n📁 Existing test files:")
        for existing_file in validation["existing_files"]:
            print(f"  ✅ {existing_file}")


def main():
    """Main function to run unit tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LangGraph multi-agent system unit tests")
    parser.add_argument("--agent", help="Run tests for specific agent (e.g., planner, code_generator)")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage reporting")
    parser.add_argument("--validate", action="store_true", help="Validate test structure only")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    runner = UnitTestRunner()
    
    if args.validate:
        runner.run_test_validation()
        return
    
    if args.agent:
        # Run specific agent tests
        results = runner.run_specific_agent_tests(args.agent)
        if results["success"]:
            print(f"✅ {args.agent.title()}Agent tests passed!")
        else:
            print(f"❌ {args.agent.title()}Agent tests failed!")
            if not args.quiet:
                print(results.get("stdout", ""))
                print(results.get("stderr", ""))
    else:
        # Run all unit tests
        results = runner.run_all_tests(
            verbose=not args.quiet,
            coverage=not args.no_coverage
        )
        
        if not args.quiet:
            print(runner.generate_test_report())
        else:
            if results["success"]:
                print("✅ All unit tests passed!")
            else:
                print("❌ Some unit tests failed!")


if __name__ == "__main__":
    main()
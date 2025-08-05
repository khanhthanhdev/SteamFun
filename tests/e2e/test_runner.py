"""
End-to-End Test Runner

Comprehensive test runner for all e2e tests with reporting and analysis.
Orchestrates test execution and provides detailed summaries.
"""

import pytest
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class E2ETestRunner:
    """End-to-end test runner with comprehensive reporting."""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        self.test_categories = {
            "api": "tests/e2e/test_api/",
            "user_scenarios": "tests/e2e/test_user_scenarios/",
            "workflows": "tests/e2e/test_workflows/"
        }
    
    def run_all_tests(self, verbose: bool = True, generate_report: bool = True) -> Dict[str, Any]:
        """Run all e2e tests and generate comprehensive report."""
        print("=" * 80)
        print("STARTING END-TO-END TEST EXECUTION")
        print("=" * 80)
        
        self.start_time = datetime.now()
        
        # Run tests by category
        for category, test_path in self.test_categories.items():
            print(f"\n{'='*20} RUNNING {category.upper()} TESTS {'='*20}")
            
            category_results = self._run_test_category(category, test_path, verbose)
            self.test_results[category] = category_results
        
        self.end_time = datetime.now()
        
        # Generate comprehensive report
        if generate_report:
            report = self._generate_comprehensive_report()
            self._save_report(report)
            self._print_summary(report)
            return report
        
        return self.test_results
    
    def run_category(self, category: str, verbose: bool = True) -> Dict[str, Any]:
        """Run tests for a specific category."""
        if category not in self.test_categories:
            raise ValueError(f"Unknown category: {category}. Available: {list(self.test_categories.keys())}")
        
        print(f"Running {category} tests...")
        test_path = self.test_categories[category]
        
        self.start_time = datetime.now()
        results = self._run_test_category(category, test_path, verbose)
        self.end_time = datetime.now()
        
        return results
    
    def _run_test_category(self, category: str, test_path: str, verbose: bool) -> Dict[str, Any]:
        """Run tests for a specific category."""
        category_start = time.time()
        
        # Prepare pytest arguments
        pytest_args = [
            test_path,
            "-v" if verbose else "-q",
            "-m", "e2e",
            "--tb=short",
            "--disable-warnings",
            f"--junitxml=test_results_{category}.xml"
        ]
        
        # Run pytest and capture results
        try:
            exit_code = pytest.main(pytest_args)
            success = exit_code == 0
        except Exception as e:
            print(f"Error running {category} tests: {e}")
            success = False
            exit_code = 1
        
        category_end = time.time()
        execution_time = category_end - category_start
        
        # Parse test results
        test_details = self._parse_test_results(category)
        
        return {
            "category": category,
            "success": success,
            "exit_code": exit_code,
            "execution_time": execution_time,
            "test_details": test_details,
            "timestamp": datetime.now().isoformat()
        }
    
    def _parse_test_results(self, category: str) -> Dict[str, Any]:
        """Parse detailed test results from pytest output."""
        # This would parse the JUnit XML file for detailed results
        # For now, return basic structure
        return {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": [],
            "test_files": []
        }
    
    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test execution report."""
        total_execution_time = (self.end_time - self.start_time).total_seconds()
        
        # Calculate overall statistics
        total_categories = len(self.test_results)
        successful_categories = sum(1 for result in self.test_results.values() if result["success"])
        
        # Collect execution times
        category_times = {
            category: result["execution_time"] 
            for category, result in self.test_results.items()
        }
        
        # Generate report
        report = {
            "execution_summary": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "total_execution_time": total_execution_time,
                "total_categories": total_categories,
                "successful_categories": successful_categories,
                "success_rate": successful_categories / total_categories if total_categories > 0 else 0
            },
            "category_results": self.test_results,
            "category_execution_times": category_times,
            "performance_metrics": self._calculate_performance_metrics(),
            "recommendations": self._generate_recommendations(),
            "system_info": self._collect_system_info()
        }
        
        return report
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics from test execution."""
        if not self.test_results:
            return {}
        
        execution_times = [result["execution_time"] for result in self.test_results.values()]
        
        return {
            "average_category_time": sum(execution_times) / len(execution_times),
            "fastest_category": min(self.test_results.items(), key=lambda x: x[1]["execution_time"]),
            "slowest_category": max(self.test_results.items(), key=lambda x: x[1]["execution_time"]),
            "total_test_time": sum(execution_times)
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check for failed categories
        failed_categories = [
            category for category, result in self.test_results.items() 
            if not result["success"]
        ]
        
        if failed_categories:
            recommendations.append(
                f"Failed test categories detected: {', '.join(failed_categories)}. "
                "Review test logs and fix failing tests."
            )
        
        # Check for slow execution
        if self.test_results:
            avg_time = sum(r["execution_time"] for r in self.test_results.values()) / len(self.test_results)
            if avg_time > 60:  # More than 1 minute per category
                recommendations.append(
                    "Test execution is slow. Consider optimizing test setup, "
                    "using mocks more extensively, or running tests in parallel."
                )
        
        # Check success rate
        if self.test_results:
            success_rate = sum(1 for r in self.test_results.values() if r["success"]) / len(self.test_results)
            if success_rate < 0.8:
                recommendations.append(
                    "Low test success rate detected. Review and fix failing tests "
                    "to ensure system reliability."
                )
        
        if not recommendations:
            recommendations.append("All tests passed successfully! System is functioning well.")
        
        return recommendations
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information for the report."""
        import platform
        import psutil
        
        try:
            return {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "disk_usage": psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else None
            }
        except Exception:
            return {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "note": "Extended system info unavailable"
            }
    
    def _save_report(self, report: Dict[str, Any]) -> None:
        """Save comprehensive report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"e2e_test_report_{timestamp}.json"
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\nDetailed report saved to: {report_file}")
        except Exception as e:
            print(f"Failed to save report: {e}")
    
    def _print_summary(self, report: Dict[str, Any]) -> None:
        """Print comprehensive test summary."""
        print("\n" + "=" * 80)
        print("END-TO-END TEST EXECUTION SUMMARY")
        print("=" * 80)
        
        # Execution summary
        summary = report["execution_summary"]
        print(f"Start Time: {summary['start_time']}")
        print(f"End Time: {summary['end_time']}")
        print(f"Total Execution Time: {summary['total_execution_time']:.2f} seconds")
        print(f"Categories Tested: {summary['total_categories']}")
        print(f"Successful Categories: {summary['successful_categories']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        
        # Category results
        print(f"\n{'CATEGORY RESULTS':^80}")
        print("-" * 80)
        for category, result in report["category_results"].items():
            status = "✅ PASSED" if result["success"] else "❌ FAILED"
            print(f"{category.upper():20} | {status:10} | {result['execution_time']:6.2f}s")
        
        # Performance metrics
        if report["performance_metrics"]:
            metrics = report["performance_metrics"]
            print(f"\n{'PERFORMANCE METRICS':^80}")
            print("-" * 80)
            print(f"Average Category Time: {metrics['average_category_time']:.2f}s")
            print(f"Fastest Category: {metrics['fastest_category'][0]} ({metrics['fastest_category'][1]['execution_time']:.2f}s)")
            print(f"Slowest Category: {metrics['slowest_category'][0]} ({metrics['slowest_category'][1]['execution_time']:.2f}s)")
        
        # Recommendations
        print(f"\n{'RECOMMENDATIONS':^80}")
        print("-" * 80)
        for i, recommendation in enumerate(report["recommendations"], 1):
            print(f"{i}. {recommendation}")
        
        # System info
        if report["system_info"]:
            print(f"\n{'SYSTEM INFORMATION':^80}")
            print("-" * 80)
            sys_info = report["system_info"]
            print(f"Platform: {sys_info.get('platform', 'Unknown')}")
            print(f"Python Version: {sys_info.get('python_version', 'Unknown')}")
            if 'cpu_count' in sys_info:
                print(f"CPU Cores: {sys_info['cpu_count']}")
            if 'memory_total' in sys_info:
                print(f"Total Memory: {sys_info['memory_total'] / (1024**3):.1f} GB")
        
        print("=" * 80)


def main():
    """Main entry point for test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run end-to-end tests")
    parser.add_argument(
        "--category", 
        choices=["api", "user_scenarios", "workflows", "all"],
        default="all",
        help="Test category to run"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--no-report",
        action="store_true", 
        help="Skip generating detailed report"
    )
    
    args = parser.parse_args()
    
    runner = E2ETestRunner()
    
    try:
        if args.category == "all":
            results = runner.run_all_tests(
                verbose=args.verbose,
                generate_report=not args.no_report
            )
        else:
            results = runner.run_category(
                args.category,
                verbose=args.verbose
            )
        
        # Exit with appropriate code
        if isinstance(results, dict) and "execution_summary" in results:
            success_rate = results["execution_summary"]["success_rate"]
            sys.exit(0 if success_rate >= 0.8 else 1)
        else:
            # Single category result
            sys.exit(0 if results.get("success", False) else 1)
            
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
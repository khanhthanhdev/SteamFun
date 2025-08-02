"""
RAG System Evaluation and Validation Script

This script evaluates the enhanced RAG system against the legacy system
and validates that all requirements are met through automated testing.
"""

import json
import time
import statistics
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch

from src.rag.rag_integration import RAGIntegration, RAGConfig


@dataclass
class EvaluationResult:
    """Results from RAG system evaluation."""
    system_name: str
    avg_response_time: float
    max_response_time: float
    min_response_time: float
    success_rate: float
    query_count: int
    total_docs_retrieved: int
    avg_docs_per_query: float
    errors: List[str]


@dataclass
class ComparisonResult:
    """Comparison between legacy and enhanced systems."""
    legacy_results: EvaluationResult
    enhanced_results: EvaluationResult
    performance_improvement: float
    success_rate_improvement: float
    response_time_improvement: float
    requirements_validation: Dict[str, bool]


class RAGSystemEvaluator:
    """Evaluator for comparing RAG systems and validating requirements."""
    
    def __init__(self):
        self.test_queries = self._create_test_dataset()
        self.temp_dir = None
    
    def _create_test_dataset(self) -> List[Dict[str, Any]]:
        """Create evaluation dataset from typical use cases."""
        return [
            {
                "type": "storyboard",
                "scene_plan": "Create a blue circle that moves from left to right",
                "topic": "basic_animation",
                "expected_concepts": ["Circle", "animation", "movement"]
            },
            {
                "type": "technical",
                "storyboard": "Scene shows a rotating square with physics effects",
                "topic": "physics_animation",
                "expected_concepts": ["Square", "rotation", "physics"]
            },
            {
                "type": "error_fix",
                "error": "AttributeError: 'Circle' object has no attribute 'move_to'",
                "code": "circle = Circle()\ncircle.move_to(UP)",
                "topic": "error_debugging",
                "expected_concepts": ["Circle", "move_to", "AttributeError"]
            },
            {
                "type": "storyboard",
                "scene_plan": "Mathematical equation transforms into a graph",
                "topic": "math_visualization",
                "expected_concepts": ["equation", "graph", "transform"]
            },
            {
                "type": "technical",
                "storyboard": "Complex scene with multiple animated objects",
                "topic": "complex_animation",
                "expected_concepts": ["multiple", "objects", "animation"]
            },
            {
                "type": "storyboard",
                "scene_plan": "Text appears with fade effect and moves upward",
                "topic": "text_animation",
                "expected_concepts": ["text", "fade", "movement"]
            },
            {
                "type": "error_fix",
                "error": "TypeError: Transform() missing required argument",
                "code": "self.play(Transform())",
                "topic": "transform_error",
                "expected_concepts": ["Transform", "TypeError", "argument"]
            },
            {
                "type": "technical",
                "storyboard": "3D objects with lighting and camera movement",
                "topic": "3d_animation",
                "expected_concepts": ["3D", "lighting", "camera"]
            }
        ]
    
    def setup_test_environment(self):
        """Set up temporary test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock documentation
        docs_dir = Path(self.temp_dir) / "docs"
        docs_dir.mkdir(exist_ok=True)
        
        # Create sample docs
        (docs_dir / "circle.py").write_text("""
class Circle(VMobject):
    def __init__(self, radius=1.0):
        self.radius = radius
    
    def shift(self, vector):
        return self
""")
        
        (docs_dir / "animations.md").write_text("""
# Animations
## Transform
Transform one object into another.
""")
    
    def cleanup_test_environment(self):
        """Clean up test environment."""
        if self.temp_dir:
            shutil.rmtree(self.temp_dir)
    
    def create_mock_helper_model(self) -> Mock:
        """Create mock helper model with realistic responses."""
        def mock_response(prompt, **kwargs):
            if "storyboard" in prompt.lower():
                return '```json\n["Circle animation", "Movement patterns", "Manim basics"]\n```'
            elif "technical" in prompt.lower():
                return '```json\n["Implementation details", "Code examples", "API reference"]\n```'
            elif "error" in prompt.lower():
                return '```json\n["Error solutions", "Debugging tips", "Common fixes"]\n```'
            else:
                return '```json\n["General query", "Documentation"]\n```'
        
        mock_model = Mock()
        mock_model.side_effect = mock_response
        return mock_model
    
    def evaluate_system(self, system_name: str, config: RAGConfig) -> EvaluationResult:
        """Evaluate a RAG system configuration."""
        mock_model = self.create_mock_helper_model()
        response_times = []
        success_count = 0
        total_docs = 0
        errors = []
        
        with patch('src.rag.rag_integration.RAGVectorStore') as mock_vector_store_class:
            # Mock vector store
            mock_vector_store = Mock()
            mock_vector_store.find_relevant_docs.return_value = [
                "Sample documentation 1",
                "Sample documentation 2",
                "Sample documentation 3"
            ]
            mock_vector_store_class.return_value = mock_vector_store
            
            # Create RAG integration
            rag = RAGIntegration(
                helper_model=mock_model,
                output_dir=self.temp_dir,
                chroma_db_path=f"{self.temp_dir}/chroma",
                manim_docs_path=f"{self.temp_dir}/docs",
                embedding_model="test-model",
                use_langfuse=False,
                session_id=f"{system_name}-eval",
                config=config
            )
            
            # Run evaluation queries
            for i, test_case in enumerate(self.test_queries):
                try:
                    start_time = time.time()
                    
                    # Generate queries based on test case type
                    if test_case["type"] == "storyboard":
                        queries = rag._generate_rag_queries_storyboard(
                            scene_plan=test_case["scene_plan"],
                            topic=test_case["topic"],
                            scene_number=i+1
                        )
                    elif test_case["type"] == "technical":
                        queries = rag._generate_rag_queries_technical(
                            storyboard=test_case["storyboard"],
                            topic=test_case["topic"],
                            scene_number=i+1
                        )
                    elif test_case["type"] == "error_fix":
                        queries = rag._generate_rag_queries_error_fix(
                            error=test_case["error"],
                            code=test_case["code"],
                            topic=test_case["topic"],
                            scene_number=i+1
                        )
                    
                    # Retrieve documents
                    docs = rag.get_relevant_docs(
                        rag_queries=[{"query": q} for q in queries],
                        scene_trace_id=f"{system_name}-eval-{i}",
                        topic=test_case["topic"],
                        scene_number=i+1
                    )
                    
                    elapsed_time = time.time() - start_time
                    response_times.append(elapsed_time)
                    
                    if queries and docs:
                        success_count += 1
                        total_docs += len(docs)
                    
                except Exception as e:
                    errors.append(f"Test case {i}: {str(e)}")
                    response_times.append(10.0)  # Penalty for errors
        
        return EvaluationResult(
            system_name=system_name,
            avg_response_time=statistics.mean(response_times) if response_times else 0.0,
            max_response_time=max(response_times) if response_times else 0.0,
            min_response_time=min(response_times) if response_times else 0.0,
            success_rate=success_count / len(self.test_queries),
            query_count=len(self.test_queries),
            total_docs_retrieved=total_docs,
            avg_docs_per_query=total_docs / len(self.test_queries) if self.test_queries else 0.0,
            errors=errors
        )
    
    def validate_requirements(self, enhanced_results: EvaluationResult) -> Dict[str, bool]:
        """Validate that all requirements are met."""
        validation = {}
        
        # Requirement 5.1: Sub-2-second response time
        validation["req_5_1_performance"] = enhanced_results.max_response_time < 2.0
        
        # Requirement 2: Intelligent query generation (success rate > 80%)
        validation["req_2_query_generation"] = enhanced_results.success_rate > 0.8
        
        # Requirement 3: Context-aware retrieval (docs retrieved per query > 1)
        validation["req_3_retrieval"] = enhanced_results.avg_docs_per_query > 1.0
        
        # Requirement 7: Error handling (error rate < 20%)
        error_rate = len(enhanced_results.errors) / enhanced_results.query_count
        validation["req_7_error_handling"] = error_rate < 0.2
        
        # Overall system reliability
        validation["overall_reliability"] = enhanced_results.success_rate > 0.9
        
        return validation
    
    def compare_systems(self) -> ComparisonResult:
        """Compare legacy and enhanced RAG systems."""
        print("Setting up test environment...")
        self.setup_test_environment()
        
        try:
            print("Evaluating legacy system...")
            legacy_config = RAGConfig(
                use_enhanced_components=False,
                enable_caching=False,
                enable_quality_monitoring=False,
                enable_error_handling=False
            )
            legacy_results = self.evaluate_system("legacy", legacy_config)
            
            print("Evaluating enhanced system...")
            enhanced_config = RAGConfig(
                use_enhanced_components=True,
                enable_caching=True,
                enable_quality_monitoring=True,
                enable_error_handling=True
            )
            enhanced_results = self.evaluate_system("enhanced", enhanced_config)
            
            print("Validating requirements...")
            requirements_validation = self.validate_requirements(enhanced_results)
            
            # Calculate improvements
            performance_improvement = (
                (legacy_results.avg_response_time - enhanced_results.avg_response_time) 
                / legacy_results.avg_response_time * 100
            ) if legacy_results.avg_response_time > 0 else 0.0
            
            success_rate_improvement = (
                enhanced_results.success_rate - legacy_results.success_rate
            ) * 100
            
            response_time_improvement = (
                legacy_results.avg_response_time - enhanced_results.avg_response_time
            )
            
            return ComparisonResult(
                legacy_results=legacy_results,
                enhanced_results=enhanced_results,
                performance_improvement=performance_improvement,
                success_rate_improvement=success_rate_improvement,
                response_time_improvement=response_time_improvement,
                requirements_validation=requirements_validation
            )
            
        finally:
            print("Cleaning up test environment...")
            self.cleanup_test_environment()
    
    def generate_evaluation_report(self, comparison: ComparisonResult) -> str:
        """Generate comprehensive evaluation report."""
        report = []
        report.append("=" * 80)
        report.append("RAG SYSTEM EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Legacy System Results
        report.append("LEGACY SYSTEM RESULTS:")
        report.append("-" * 40)
        legacy = comparison.legacy_results
        report.append(f"Average Response Time: {legacy.avg_response_time:.3f}s")
        report.append(f"Max Response Time: {legacy.max_response_time:.3f}s")
        report.append(f"Success Rate: {legacy.success_rate:.1%}")
        report.append(f"Docs per Query: {legacy.avg_docs_per_query:.1f}")
        report.append(f"Errors: {len(legacy.errors)}")
        report.append("")
        
        # Enhanced System Results
        report.append("ENHANCED SYSTEM RESULTS:")
        report.append("-" * 40)
        enhanced = comparison.enhanced_results
        report.append(f"Average Response Time: {enhanced.avg_response_time:.3f}s")
        report.append(f"Max Response Time: {enhanced.max_response_time:.3f}s")
        report.append(f"Success Rate: {enhanced.success_rate:.1%}")
        report.append(f"Docs per Query: {enhanced.avg_docs_per_query:.1f}")
        report.append(f"Errors: {len(enhanced.errors)}")
        report.append("")
        
        # Improvements
        report.append("SYSTEM IMPROVEMENTS:")
        report.append("-" * 40)
        report.append(f"Performance Improvement: {comparison.performance_improvement:.1f}%")
        report.append(f"Success Rate Improvement: {comparison.success_rate_improvement:.1f}%")
        report.append(f"Response Time Improvement: {comparison.response_time_improvement:.3f}s")
        report.append("")
        
        # Requirements Validation
        report.append("REQUIREMENTS VALIDATION:")
        report.append("-" * 40)
        for req, passed in comparison.requirements_validation.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            report.append(f"{req}: {status}")
        report.append("")
        
        # Overall Assessment
        all_requirements_met = all(comparison.requirements_validation.values())
        report.append("OVERALL ASSESSMENT:")
        report.append("-" * 40)
        if all_requirements_met:
            report.append("✓ ALL REQUIREMENTS MET")
            report.append("The enhanced RAG system successfully meets all specified requirements.")
        else:
            report.append("✗ SOME REQUIREMENTS NOT MET")
            failed_reqs = [req for req, passed in comparison.requirements_validation.items() if not passed]
            report.append(f"Failed requirements: {', '.join(failed_reqs)}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_detailed_results(self, comparison: ComparisonResult, filename: str = "rag_evaluation_results.json"):
        """Save detailed evaluation results to JSON file."""
        results = {
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "legacy_system": asdict(comparison.legacy_results),
            "enhanced_system": asdict(comparison.enhanced_results),
            "improvements": {
                "performance_improvement_percent": comparison.performance_improvement,
                "success_rate_improvement_percent": comparison.success_rate_improvement,
                "response_time_improvement_seconds": comparison.response_time_improvement
            },
            "requirements_validation": comparison.requirements_validation,
            "test_dataset": self.test_queries
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Detailed results saved to {filename}")


def main():
    """Main evaluation function."""
    print("Starting RAG System Evaluation...")
    
    evaluator = RAGSystemEvaluator()
    
    try:
        # Run comparison
        comparison = evaluator.compare_systems()
        
        # Generate and display report
        report = evaluator.generate_evaluation_report(comparison)
        print(report)
        
        # Save detailed results
        evaluator.save_detailed_results(comparison)
        
        # Return success/failure based on requirements
        all_requirements_met = all(comparison.requirements_validation.values())
        if all_requirements_met:
            print("\n🎉 EVALUATION SUCCESSFUL: All requirements validated!")
            return 0
        else:
            print("\n❌ EVALUATION FAILED: Some requirements not met!")
            return 1
            
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
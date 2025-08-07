"""
Test scenarios and data fixtures for LangGraph Studio agent testing.

This module provides pre-defined test scenarios and data fixtures for testing
individual agents and workflows in the Studio environment.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from ..models.state import VideoGenerationState
from ..models.config import WorkflowConfig
from ..models.errors import WorkflowError, ErrorType, ErrorSeverity

logger = logging.getLogger(__name__)


class TestScenarioManager:
    """Manager for test scenarios and data fixtures."""
    
    def __init__(self):
        self.scenarios = {}
        self.fixtures = {}
        self._initialize_default_scenarios()
        self._initialize_default_fixtures()
    
    def _initialize_default_scenarios(self):
        """Initialize default test scenarios for each agent."""
        
        # Planning Agent Scenarios
        self.scenarios["planning_agent"] = {
            "basic_math_topic": {
                "name": "Basic Math Topic",
                "description": "Test planning with a basic mathematics topic",
                "input": {
                    "topic": "Linear Equations",
                    "description": "Create a video explaining how to solve linear equations step by step, including graphing and real-world applications."
                },
                "expected_outputs": ["scene_outline", "scene_implementations", "detected_plugins"],
                "validation_rules": [
                    "scene_outline should contain multiple scenes",
                    "scene_implementations should be a dictionary with integer keys",
                    "detected_plugins should include 'manim'"
                ]
            },
            "complex_physics_topic": {
                "name": "Complex Physics Topic",
                "description": "Test planning with a complex physics topic",
                "input": {
                    "topic": "Quantum Mechanics Fundamentals",
                    "description": "Explain wave-particle duality, uncertainty principle, and quantum superposition with visual demonstrations and mathematical formulations."
                },
                "expected_outputs": ["scene_outline", "scene_implementations", "detected_plugins"],
                "validation_rules": [
                    "scene_outline should contain at least 3 scenes",
                    "scene_implementations should include mathematical content",
                    "detected_plugins should include physics-related plugins"
                ]
            },
            "simple_concept": {
                "name": "Simple Concept",
                "description": "Test planning with a very simple concept",
                "input": {
                    "topic": "Colors",
                    "description": "Introduce primary colors and how they mix to create secondary colors."
                },
                "expected_outputs": ["scene_outline", "scene_implementations"],
                "validation_rules": [
                    "scene_outline should be concise",
                    "scene_implementations should be simple"
                ]
            }
        }
        
        # Code Generation Agent Scenarios
        self.scenarios["code_generation_agent"] = {
            "basic_scene_implementation": {
                "name": "Basic Scene Implementation",
                "description": "Test code generation with basic scene implementations",
                "input": {
                    "topic": "Linear Equations",
                    "description": "Solve linear equations step by step",
                    "scene_outline": "Scene 1: Introduction\nScene 2: Solving Process\nScene 3: Examples",
                    "scene_implementations": {
                        1: "Introduce linear equations with title and basic definition",
                        2: "Show step-by-step solving process with algebraic manipulation",
                        3: "Present 2-3 concrete examples with solutions"
                    }
                },
                "expected_outputs": ["generated_code"],
                "validation_rules": [
                    "generated_code should contain valid Python syntax",
                    "generated_code should include Manim imports",
                    "generated_code should have Scene classes"
                ]
            },
            "mathematical_content": {
                "name": "Mathematical Content",
                "description": "Test code generation with mathematical formulas",
                "input": {
                    "topic": "Quadratic Formula",
                    "description": "Derive and apply the quadratic formula",
                    "scene_outline": "Scene 1: Formula Introduction\nScene 2: Derivation\nScene 3: Application",
                    "scene_implementations": {
                        1: "Present the quadratic formula with mathematical notation",
                        2: "Show step-by-step derivation from completing the square",
                        3: "Apply formula to solve example problems"
                    }
                },
                "expected_outputs": ["generated_code"],
                "validation_rules": [
                    "generated_code should include MathTex objects",
                    "generated_code should handle mathematical expressions",
                    "generated_code should have proper animations"
                ]
            }
        }
        
        # Rendering Agent Scenarios
        self.scenarios["rendering_agent"] = {
            "simple_code_rendering": {
                "name": "Simple Code Rendering",
                "description": "Test rendering with simple Manim code",
                "input": {
                    "topic": "Test Video",
                    "session_id": "test_session_001",
                    "generated_code": {
                        1: '''
from manim import *

class TestScene(Scene):
    def construct(self):
        title = Text("Test Video", font_size=48)
        self.play(Write(title))
        self.wait(2)
                        '''
                    }
                },
                "expected_outputs": ["rendered_videos"],
                "validation_rules": [
                    "rendered_videos should contain video file paths",
                    "video files should exist after rendering"
                ]
            },
            "multiple_scenes": {
                "name": "Multiple Scenes",
                "description": "Test rendering with multiple scenes",
                "input": {
                    "topic": "Multi Scene Test",
                    "session_id": "test_session_002",
                    "generated_code": {
                        1: '''
from manim import *

class Scene1(Scene):
    def construct(self):
        title = Text("Scene 1", font_size=48)
        self.play(Write(title))
        self.wait(1)
                        ''',
                        2: '''
from manim import *

class Scene2(Scene):
    def construct(self):
        title = Text("Scene 2", font_size=48)
        self.play(Write(title))
        self.wait(1)
                        '''
                    }
                },
                "expected_outputs": ["rendered_videos", "combined_video_path"],
                "validation_rules": [
                    "rendered_videos should contain multiple scenes",
                    "combined_video_path should be set"
                ]
            }
        }
        
        # Error Handler Agent Scenarios
        self.scenarios["error_handler_agent"] = {
            "recoverable_error": {
                "name": "Recoverable Error",
                "description": "Test error handling with recoverable errors",
                "input": {
                    "current_step": "code_generation",
                    "errors": [
                        WorkflowError(
                            step="code_generation",
                            error_type=ErrorType.MODEL,
                            message="Temporary API error",
                            severity=ErrorSeverity.MEDIUM
                        )
                    ]
                },
                "expected_outputs": ["retry_counts"],
                "validation_rules": [
                    "should attempt recovery",
                    "retry_counts should be updated"
                ]
            },
            "critical_error": {
                "name": "Critical Error",
                "description": "Test error handling with critical errors",
                "input": {
                    "current_step": "rendering",
                    "errors": [
                        WorkflowError(
                            step="rendering",
                            error_type=ErrorType.SYSTEM,
                            message="System resource exhausted",
                            severity=ErrorSeverity.CRITICAL
                        )
                    ]
                },
                "expected_outputs": ["escalated_errors"],
                "validation_rules": [
                    "should escalate critical errors",
                    "escalated_errors should contain the error"
                ]
            }
        }
    
    def _initialize_default_fixtures(self):
        """Initialize default test data fixtures."""
        
        # Basic state fixtures
        self.fixtures["basic_state"] = {
            "topic": "Test Topic",
            "description": "Test Description",
            "session_id": "test_session",
            "config": WorkflowConfig()
        }
        
        self.fixtures["planning_complete_state"] = {
            "topic": "Linear Equations",
            "description": "Solve linear equations step by step",
            "session_id": "planning_test",
            "scene_outline": "Scene 1: Introduction\nScene 2: Process\nScene 3: Examples",
            "scene_implementations": {
                1: "Introduce linear equations",
                2: "Show solving process",
                3: "Present examples"
            },
            "detected_plugins": ["manim", "numpy"],
            "config": WorkflowConfig()
        }
        
        self.fixtures["code_generation_complete_state"] = {
            "topic": "Linear Equations",
            "description": "Solve linear equations step by step",
            "session_id": "code_gen_test",
            "scene_outline": "Scene 1: Introduction\nScene 2: Process\nScene 3: Examples",
            "scene_implementations": {
                1: "Introduce linear equations",
                2: "Show solving process",
                3: "Present examples"
            },
            "generated_code": {
                1: "from manim import *\n\nclass Scene1(Scene):\n    def construct(self):\n        pass",
                2: "from manim import *\n\nclass Scene2(Scene):\n    def construct(self):\n        pass"
            },
            "config": WorkflowConfig()
        }
        
        self.fixtures["error_state"] = {
            "topic": "Test Topic",
            "description": "Test Description",
            "session_id": "error_test",
            "current_step": "code_generation",
            "errors": [
                WorkflowError(
                    step="code_generation",
                    error_type=ErrorType.MODEL,
                    message="Test error",
                    severity=ErrorSeverity.MEDIUM
                )
            ],
            "config": WorkflowConfig()
        }
    
    def get_scenario(self, agent_name: str, scenario_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific test scenario."""
        agent_scenarios = self.scenarios.get(agent_name, {})
        return agent_scenarios.get(scenario_name)
    
    def list_scenarios(self, agent_name: str = None) -> Dict[str, Any]:
        """List available scenarios, optionally filtered by agent."""
        if agent_name:
            return self.scenarios.get(agent_name, {})
        return self.scenarios
    
    def get_fixture(self, fixture_name: str) -> Optional[Dict[str, Any]]:
        """Get a test data fixture."""
        return self.fixtures.get(fixture_name)
    
    def list_fixtures(self) -> List[str]:
        """List available fixtures."""
        return list(self.fixtures.keys())
    
    def create_test_state(self, fixture_name: str, overrides: Dict[str, Any] = None) -> VideoGenerationState:
        """Create a test state from a fixture with optional overrides."""
        fixture_data = self.get_fixture(fixture_name)
        if not fixture_data:
            raise ValueError(f"Fixture not found: {fixture_name}")
        
        # Apply overrides
        if overrides:
            fixture_data = {**fixture_data, **overrides}
        
        return VideoGenerationState(**fixture_data)
    
    def add_scenario(
        self,
        agent_name: str,
        scenario_name: str,
        scenario_data: Dict[str, Any]
    ):
        """Add a custom test scenario."""
        if agent_name not in self.scenarios:
            self.scenarios[agent_name] = {}
        
        self.scenarios[agent_name][scenario_name] = scenario_data
        logger.info(f"Added scenario: {agent_name}.{scenario_name}")
    
    def add_fixture(self, fixture_name: str, fixture_data: Dict[str, Any]):
        """Add a custom test fixture."""
        self.fixtures[fixture_name] = fixture_data
        logger.info(f"Added fixture: {fixture_name}")
    
    def validate_scenario_output(
        self,
        agent_name: str,
        scenario_name: str,
        output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate scenario output against expected results."""
        scenario = self.get_scenario(agent_name, scenario_name)
        if not scenario:
            return {"valid": False, "error": "Scenario not found"}
        
        validation_results = {
            "valid": True,
            "passed_checks": [],
            "failed_checks": [],
            "warnings": []
        }
        
        # Check expected outputs
        expected_outputs = scenario.get("expected_outputs", [])
        for expected_output in expected_outputs:
            if expected_output in output and output[expected_output] is not None:
                validation_results["passed_checks"].append(f"Has {expected_output}")
            else:
                validation_results["failed_checks"].append(f"Missing {expected_output}")
                validation_results["valid"] = False
        
        # Apply validation rules
        validation_rules = scenario.get("validation_rules", [])
        for rule in validation_rules:
            # Simple rule validation (in practice, this would be more sophisticated)
            if "should contain" in rule:
                # Extract field and requirement from rule
                parts = rule.split(" should contain ")
                if len(parts) == 2:
                    field = parts[0].strip()
                    requirement = parts[1].strip()
                    
                    if field in output:
                        validation_results["passed_checks"].append(f"Rule: {rule}")
                    else:
                        validation_results["failed_checks"].append(f"Rule failed: {rule}")
                        validation_results["valid"] = False
            else:
                # Generic rule check
                validation_results["warnings"].append(f"Could not validate rule: {rule}")
        
        return validation_results


class StudioTestDataGenerator:
    """Generator for creating test data for Studio testing."""
    
    @staticmethod
    def generate_planning_test_data(complexity: str = "medium") -> Dict[str, Any]:
        """Generate test data for planning agent testing."""
        if complexity == "simple":
            return {
                "topic": "Basic Addition",
                "description": "Teach children how to add single-digit numbers with visual aids."
            }
        elif complexity == "medium":
            return {
                "topic": "Pythagorean Theorem",
                "description": "Explain the Pythagorean theorem with geometric proofs and practical applications in construction and navigation."
            }
        elif complexity == "complex":
            return {
                "topic": "Fourier Transform",
                "description": "Demonstrate the mathematical foundations of Fourier transforms, their properties, and applications in signal processing, image analysis, and quantum mechanics."
            }
        else:
            raise ValueError("Complexity must be 'simple', 'medium', or 'complex'")
    
    @staticmethod
    def generate_code_generation_test_data(scene_count: int = 3) -> Dict[str, Any]:
        """Generate test data for code generation agent testing."""
        scene_implementations = {}
        for i in range(1, scene_count + 1):
            scene_implementations[i] = f"Scene {i}: Implementation details for scene number {i}"
        
        return {
            "topic": "Test Topic for Code Generation",
            "description": "Generate Manim code for multiple scenes",
            "scene_outline": f"Outline with {scene_count} scenes",
            "scene_implementations": scene_implementations
        }
    
    @staticmethod
    def generate_rendering_test_data(scene_count: int = 2) -> Dict[str, Any]:
        """Generate test data for rendering agent testing."""
        generated_code = {}
        for i in range(1, scene_count + 1):
            generated_code[i] = f'''
from manim import *

class TestScene{i}(Scene):
    def construct(self):
        title = Text("Test Scene {i}", font_size=48)
        self.play(Write(title))
        self.wait(2)
        self.play(FadeOut(title))
            '''
        
        return {
            "topic": "Rendering Test",
            "session_id": f"render_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "generated_code": generated_code
        }
    
    @staticmethod
    def generate_error_test_data(error_type: str = "recoverable") -> Dict[str, Any]:
        """Generate test data for error handler agent testing."""
        if error_type == "recoverable":
            errors = [
                WorkflowError(
                    step="code_generation",
                    error_type=ErrorType.MODEL,
                    message="Temporary API rate limit exceeded",
                    severity=ErrorSeverity.MEDIUM
                )
            ]
        elif error_type == "critical":
            errors = [
                WorkflowError(
                    step="rendering",
                    error_type=ErrorType.SYSTEM,
                    message="Out of memory during video rendering",
                    severity=ErrorSeverity.CRITICAL
                )
            ]
        elif error_type == "multiple":
            errors = [
                WorkflowError(
                    step="planning",
                    error_type=ErrorType.VALIDATION,
                    message="Invalid topic format",
                    severity=ErrorSeverity.LOW
                ),
                WorkflowError(
                    step="code_generation",
                    error_type=ErrorType.MODEL,
                    message="Model timeout",
                    severity=ErrorSeverity.MEDIUM
                )
            ]
        else:
            raise ValueError("Error type must be 'recoverable', 'critical', or 'multiple'")
        
        return {
            "current_step": "error_handling",
            "errors": errors
        }


# Global instance
test_scenario_manager = TestScenarioManager()


def get_test_scenario_manager() -> TestScenarioManager:
    """Get the global test scenario manager instance."""
    return test_scenario_manager
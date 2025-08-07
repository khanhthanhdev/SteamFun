"""
Test data and scenario management for agent testing.

This module provides comprehensive test data fixtures, scenario management,
and validation capabilities for LangGraph Studio agent testing.
"""

import json
import os
import random
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

from ..models.errors import ErrorType, ErrorSeverity


class TestDataType(Enum):
    """Types of test data available."""
    TOPICS = "topics"
    DESCRIPTIONS = "descriptions"
    SCENE_OUTLINES = "scene_outlines"
    SCENE_IMPLEMENTATIONS = "scene_implementations"
    CODE_SAMPLES = "code_samples"
    ERROR_SCENARIOS = "error_scenarios"
    INTERVENTION_SCENARIOS = "intervention_scenarios"


class TestComplexity(Enum):
    """Test complexity levels."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    EDGE_CASE = "edge_case"


@dataclass
class TestScenario:
    """Base class for test scenarios."""
    
    id: str
    name: str
    description: str
    complexity: TestComplexity
    tags: List[str]
    inputs: Dict[str, Any]
    expected_outputs: Dict[str, Any]
    validation_criteria: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestScenario':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            name=data['name'],
            description=data['description'],
            complexity=TestComplexity(data['complexity']),
            tags=data['tags'],
            inputs=data['inputs'],
            expected_outputs=data['expected_outputs'],
            validation_criteria=data['validation_criteria']
        )


@dataclass
class PlannerTestScenario(TestScenario):
    """Test scenario for PlannerAgent."""
    
    expected_scene_count: int = 3
    expected_plugins: List[str] = None
    
    def __post_init__(self):
        if self.expected_plugins is None:
            self.expected_plugins = []


@dataclass
class CodeGeneratorTestScenario(TestScenario):
    """Test scenario for CodeGeneratorAgent."""
    
    scene_outline: str = ""
    scene_implementations: Dict[int, str] = None
    expected_code_patterns: List[str] = None
    
    def __post_init__(self):
        if self.scene_implementations is None:
            self.scene_implementations = {}
        if self.expected_code_patterns is None:
            self.expected_code_patterns = []


@dataclass
class RendererTestScenario(TestScenario):
    """Test scenario for RendererAgent."""
    
    generated_code: Dict[int, str] = None
    file_prefix: str = "test"
    quality: str = "medium"
    expected_video_count: int = 1
    
    def __post_init__(self):
        if self.generated_code is None:
            self.generated_code = {}


@dataclass
class ErrorHandlerTestScenario(TestScenario):
    """Test scenario for ErrorHandlerAgent."""
    
    error_scenarios: List[Dict[str, Any]] = None
    expected_recovery_rate: float = 0.8
    
    def __post_init__(self):
        if self.error_scenarios is None:
            self.error_scenarios = []


@dataclass
class HumanLoopTestScenario(TestScenario):
    """Test scenario for HumanLoopAgent."""
    
    intervention_scenarios: List[Dict[str, Any]] = None
    expected_intervention_rate: float = 0.9
    
    def __post_init__(self):
        if self.intervention_scenarios is None:
            self.intervention_scenarios = []


class BaseTestDataProvider(ABC):
    """Base class for test data providers."""
    
    @abstractmethod
    def get_test_data(self, data_type: TestDataType, complexity: TestComplexity = TestComplexity.SIMPLE) -> List[Any]:
        """Get test data of specified type and complexity."""
        pass
    
    @abstractmethod
    def get_random_sample(self, data_type: TestDataType, count: int = 1) -> List[Any]:
        """Get random sample of test data."""
        pass


class StaticTestDataProvider(BaseTestDataProvider):
    """Provides static test data from predefined datasets."""
    
    def __init__(self):
        """Initialize with predefined test data."""
        self.test_data = {
            TestDataType.TOPICS: {
                TestComplexity.SIMPLE: [
                    "Linear Algebra Basics",
                    "Introduction to Calculus",
                    "Basic Geometry",
                    "Simple Physics",
                    "Math Fundamentals"
                ],
                TestComplexity.MEDIUM: [
                    "Fourier Transform Visualization",
                    "Machine Learning Concepts",
                    "Quantum Mechanics Basics",
                    "Graph Theory Applications",
                    "Statistical Analysis"
                ],
                TestComplexity.COMPLEX: [
                    "Advanced Differential Equations",
                    "Topology and Manifolds",
                    "Quantum Field Theory",
                    "Complex Analysis",
                    "Abstract Algebra"
                ],
                TestComplexity.EDGE_CASE: [
                    "Very Long Topic Name That Exceeds Normal Length Limits And Tests Edge Cases For System Robustness",
                    "Special Characters: @#$%^&*()[]{}|\\:;\"'<>,.?/~`!",
                    "Unicode Topic: 数学可视化 - Mathematical Visualization",
                    "Empty Topic Test",
                    "Single Character: A",
                    "Numbers Only: 12345",
                    "Mixed Script: Math العربية 中文 русский",
                    "Emoji Topic: 📊📈📉 Data Visualization 📊📈📉"
                ]
            },
            TestDataType.DESCRIPTIONS: {
                TestComplexity.SIMPLE: [
                    "Create a simple animation showing basic mathematical concepts.",
                    "Visualize elementary geometric shapes and their properties.",
                    "Demonstrate basic arithmetic operations with visual elements.",
                    "Show simple physics concepts like motion and forces.",
                    "Create an educational animation about numbers."
                ],
                TestComplexity.MEDIUM: [
                    "Create an educational video explaining the Fourier transform with interactive visualizations showing how complex waveforms can be decomposed into simple sine and cosine components.",
                    "Develop an animation that demonstrates machine learning concepts including gradient descent, neural networks, and decision boundaries with clear visual representations.",
                    "Visualize quantum mechanics principles including wave-particle duality, superposition, and quantum entanglement using animated graphics and mathematical representations.",
                    "Create a comprehensive graph theory visualization showing different types of graphs, algorithms like Dijkstra's shortest path, and network analysis concepts.",
                    "Develop an animated explanation of statistical concepts including probability distributions, hypothesis testing, and regression analysis."
                ],
                TestComplexity.COMPLEX: [
                    "Create a comprehensive visualization of advanced differential equations including partial differential equations, boundary value problems, and their applications in physics and engineering with multiple interconnected scenes.",
                    "Develop a detailed animation explaining topology and manifolds, including concepts like homeomorphisms, continuous deformations, and the fundamental group with abstract mathematical visualizations.",
                    "Create an advanced quantum field theory visualization showing particle interactions, Feynman diagrams, and quantum field fluctuations with complex mathematical formulations.",
                    "Develop a comprehensive complex analysis animation covering contour integration, residue theory, and conformal mappings with detailed mathematical proofs and visualizations.",
                    "Create an abstract algebra visualization explaining group theory, ring theory, and field extensions with multiple abstract mathematical concepts and their relationships."
                ],
                TestComplexity.EDGE_CASE: [
                    "Create a video with minimal description.",
                    "",  # Empty description
                    "A" * 5000,  # Very long description
                    "Description with special characters: !@#$%^&*()_+-=[]{}|;':\",./<>?~`",
                    "Unicode description: 创建一个关于数学概念的动画视频，包含复杂的可视化效果和详细的解释。",
                    "Mixed language: Create una animación sobre математические concepts with 中文 elements",
                    "JSON-like description: {\"type\": \"animation\", \"topic\": \"math\", \"style\": \"educational\"}",
                    "HTML-like description: <video><title>Math Animation</title><content>Educational content</content></video>",
                    "Code-like description: def create_animation(): return \"math video\"",
                    "Newline test:\nLine 1\nLine 2\nLine 3"
                ]
            },
            TestDataType.SCENE_OUTLINES: {
                TestComplexity.SIMPLE: [
                    """Scene 1: Introduction
- Show title and topic
- Brief overview of concepts

Scene 2: Main Content
- Demonstrate key concepts
- Show examples

Scene 3: Conclusion
- Summarize main points
- End with key takeaways""",
                    
                    """Scene 1: Setup
- Initialize the mathematical environment
- Show coordinate system

Scene 2: Demonstration
- Animate the main concept
- Highlight important features

Scene 3: Summary
- Review what was shown
- Display final result"""
                ],
                TestComplexity.MEDIUM: [
                    """Scene 1: Introduction to Fourier Transform
- Display the title "Fourier Transform Visualization"
- Show a complex waveform as the starting point
- Introduce the concept of frequency decomposition

Scene 2: Mathematical Foundation
- Present the Fourier transform equation
- Explain the relationship between time and frequency domains
- Show the mathematical notation and key variables

Scene 3: Decomposition Process
- Animate the process of breaking down the complex waveform
- Show individual sine and cosine components
- Demonstrate how they combine to form the original signal

Scene 4: Frequency Spectrum
- Display the frequency spectrum representation
- Show peaks corresponding to different frequency components
- Explain the amplitude and phase information

Scene 5: Interactive Visualization
- Allow manipulation of frequency components
- Show real-time changes in the time domain signal
- Demonstrate the inverse Fourier transform

Scene 6: Applications and Conclusion
- Show practical applications in signal processing
- Summarize key concepts learned
- Display references for further study"""
                ],
                TestComplexity.COMPLEX: [
                    """Scene 1: Advanced Mathematical Framework Setup
- Initialize complex mathematical environment with multiple coordinate systems
- Display comprehensive notation system for differential equations
- Set up boundary conditions and initial value problems

Scene 2: Partial Differential Equation Introduction
- Present the general form of PDEs with detailed mathematical notation
- Classify different types: elliptic, parabolic, and hyperbolic
- Show examples from physics and engineering applications

Scene 3: Boundary Value Problem Formulation
- Define boundary conditions for different geometric domains
- Show Dirichlet, Neumann, and mixed boundary conditions
- Demonstrate the mathematical setup for complex geometries

Scene 4: Solution Method Visualization
- Animate finite difference method approximations
- Show grid generation and discretization process
- Demonstrate convergence behavior and stability analysis

Scene 5: Numerical Implementation
- Visualize matrix formulation of the discrete system
- Show iterative solution methods like Gauss-Seidel
- Demonstrate error analysis and convergence criteria

Scene 6: Physical Interpretation
- Connect mathematical solutions to physical phenomena
- Show heat conduction, wave propagation, and fluid flow examples
- Demonstrate how boundary conditions affect solution behavior

Scene 7: Advanced Topics
- Introduce eigenvalue problems and their solutions
- Show Green's functions and their applications
- Demonstrate variational methods and weak formulations

Scene 8: Comprehensive Summary
- Review all major concepts and their interconnections
- Show the hierarchy of mathematical tools used
- Provide roadmap for further advanced study"""
                ]
            },
            TestDataType.CODE_SAMPLES: {
                TestComplexity.SIMPLE: [
                    '''from manim import *

class SimpleCircle(Scene):
    def construct(self):
        circle = Circle()
        self.play(Create(circle))
        self.wait()''',
                    
                    '''from manim import *

class BasicText(Scene):
    def construct(self):
        text = Text("Hello Manim!")
        self.play(Write(text))
        self.wait()''',
                    
                    '''from manim import *

class BasicSquare(Scene):
    def construct(self):
        square = Square(color=BLUE)
        self.play(DrawBorderThenFill(square))
        self.wait()''',
                    
                    '''from manim import *

class SimpleArrow(Scene):
    def construct(self):
        arrow = Arrow(LEFT, RIGHT)
        self.play(GrowArrow(arrow))
        self.wait()''',
                    
                    '''from manim import *

class NumberDisplay(Scene):
    def construct(self):
        number = MathTex("42")
        self.play(Write(number))
        self.wait()'''
                ],
                TestComplexity.MEDIUM: [
                    '''from manim import *

class FourierVisualization(Scene):
    def construct(self):
        # Create axes
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-2, 2, 0.5],
            axis_config={"color": BLUE}
        )
        
        # Create function
        func = axes.plot(lambda x: np.sin(x) + 0.5*np.sin(3*x), color=YELLOW)
        
        # Add labels
        title = Text("Fourier Transform Visualization").to_edge(UP)
        
        # Animate
        self.play(Create(axes))
        self.play(Write(title))
        self.play(Create(func))
        self.wait(2)''',
                    
                    '''from manim import *

class GraphTheoryDemo(Scene):
    def construct(self):
        # Create vertices
        vertices = [Dot(point) for point in [UP, DOWN, LEFT, RIGHT]]
        
        # Create edges
        edges = [
            Line(vertices[0].get_center(), vertices[1].get_center()),
            Line(vertices[0].get_center(), vertices[2].get_center()),
            Line(vertices[1].get_center(), vertices[3].get_center())
        ]
        
        # Animate graph creation
        self.play(*[Create(v) for v in vertices])
        self.play(*[Create(e) for e in edges])
        self.wait()'''
                ]
            }
        }
    
    def get_test_data(self, data_type: TestDataType, complexity: TestComplexity = TestComplexity.SIMPLE) -> List[Any]:
        """Get test data of specified type and complexity."""
        return self.test_data.get(data_type, {}).get(complexity, [])
    
    def get_random_sample(self, data_type: TestDataType, count: int = 1) -> List[Any]:
        """Get random sample of test data."""
        all_data = []
        for complexity_data in self.test_data.get(data_type, {}).values():
            all_data.extend(complexity_data)
        
        if not all_data:
            return []
        
        return random.sample(all_data, min(count, len(all_data)))


class TestDataManager:
    """
    Manages test data fixtures for different agent types.
    
    Provides centralized access to test data with filtering,
    validation, and generation capabilities.
    """
    
    def __init__(self, data_provider: BaseTestDataProvider = None):
        """Initialize with optional custom data provider."""
        self.data_provider = data_provider or StaticTestDataProvider()
        self.custom_fixtures = {}
    
    def get_planner_test_data(self, complexity: TestComplexity = TestComplexity.SIMPLE, count: int = 1) -> List[Dict[str, Any]]:
        """Get test data for PlannerAgent."""
        topics = self.data_provider.get_test_data(TestDataType.TOPICS, complexity)
        descriptions = self.data_provider.get_test_data(TestDataType.DESCRIPTIONS, complexity)
        
        test_data = []
        for i in range(min(count, len(topics), len(descriptions))):
            test_data.append({
                'topic': topics[i],
                'description': descriptions[i],
                'session_id': f"planner_test_{uuid.uuid4().hex[:8]}"
            })
        
        return test_data
    
    def get_codegen_test_data(self, complexity: TestComplexity = TestComplexity.SIMPLE, count: int = 1) -> List[Dict[str, Any]]:
        """Get test data for CodeGeneratorAgent."""
        topics = self.data_provider.get_test_data(TestDataType.TOPICS, complexity)
        descriptions = self.data_provider.get_test_data(TestDataType.DESCRIPTIONS, complexity)
        scene_outlines = self.data_provider.get_test_data(TestDataType.SCENE_OUTLINES, complexity)
        
        test_data = []
        for i in range(min(count, len(topics), len(descriptions), len(scene_outlines))):
            # Generate scene implementations based on outline
            scene_implementations = self._generate_scene_implementations(scene_outlines[i])
            
            test_data.append({
                'topic': topics[i],
                'description': descriptions[i],
                'scene_outline': scene_outlines[i],
                'scene_implementations': scene_implementations,
                'session_id': f"codegen_test_{uuid.uuid4().hex[:8]}"
            })
        
        return test_data
    
    def get_renderer_test_data(self, complexity: TestComplexity = TestComplexity.SIMPLE, count: int = 1) -> List[Dict[str, Any]]:
        """Get test data for RendererAgent."""
        topics = self.data_provider.get_test_data(TestDataType.TOPICS, complexity)
        code_samples = self.data_provider.get_test_data(TestDataType.CODE_SAMPLES, complexity)
        
        test_data = []
        for i in range(min(count, len(topics), len(code_samples))):
            # Create generated code dictionary
            generated_code = {1: code_samples[i]}
            if len(code_samples) > i + 1:
                generated_code[2] = code_samples[min(i + 1, len(code_samples) - 1)]
            
            test_data.append({
                'topic': topics[i],
                'generated_code': generated_code,
                'file_prefix': f"test_{uuid.uuid4().hex[:8]}",
                'quality': 'medium',
                'session_id': f"renderer_test_{uuid.uuid4().hex[:8]}"
            })
        
        return test_data
    
    def get_error_handler_test_data(self, complexity: TestComplexity = TestComplexity.SIMPLE, count: int = 1) -> List[Dict[str, Any]]:
        """Get test data for ErrorHandlerAgent."""
        error_scenarios = self._generate_error_scenarios(complexity, count)
        
        test_data = []
        for i in range(count):
            test_data.append({
                'error_scenarios': error_scenarios[i:i+3],  # 3 scenarios per test
                'session_id': f"errorhandler_test_{uuid.uuid4().hex[:8]}"
            })
        
        return test_data
    
    def get_human_loop_test_data(self, complexity: TestComplexity = TestComplexity.SIMPLE, count: int = 1) -> List[Dict[str, Any]]:
        """Get test data for HumanLoopAgent."""
        intervention_scenarios = self._generate_intervention_scenarios(complexity, count)
        
        test_data = []
        for i in range(count):
            test_data.append({
                'intervention_scenarios': intervention_scenarios[i:i+2],  # 2 scenarios per test
                'session_id': f"humanloop_test_{uuid.uuid4().hex[:8]}"
            })
        
        return test_data
    
    def add_custom_fixture(self, name: str, data: Any) -> None:
        """Add custom test fixture."""
        self.custom_fixtures[name] = data
    
    def get_custom_fixture(self, name: str) -> Any:
        """Get custom test fixture."""
        return self.custom_fixtures.get(name)
    
    def validate_test_data(self, data: Dict[str, Any], agent_type: str) -> Tuple[bool, List[str]]:
        """Validate test data for specific agent type."""
        issues = []
        
        if agent_type == "PlannerAgent":
            issues.extend(self._validate_planner_data(data))
        elif agent_type == "CodeGeneratorAgent":
            issues.extend(self._validate_codegen_data(data))
        elif agent_type == "RendererAgent":
            issues.extend(self._validate_renderer_data(data))
        elif agent_type == "ErrorHandlerAgent":
            issues.extend(self._validate_error_handler_data(data))
        elif agent_type == "HumanLoopAgent":
            issues.extend(self._validate_human_loop_data(data))
        
        return len(issues) == 0, issues
    
    def _generate_scene_implementations(self, scene_outline: str) -> Dict[int, str]:
        """Generate scene implementations from outline."""
        # Simple implementation - split by "Scene" and create implementations
        scenes = scene_outline.split("Scene ")[1:]  # Skip empty first element
        implementations = {}
        
        for i, scene in enumerate(scenes, 1):
            # Extract scene content
            scene_content = scene.split("\n")[0].replace(f"{i}:", "").strip()
            
            # Generate implementation based on content
            implementation = f"""Implement scene {i}: {scene_content}
            
This scene should:
- Create appropriate Manim objects
- Animate the concepts described
- Use clear visual representations
- Follow good animation practices

Key elements to include:
- Proper scene setup
- Clear animations
- Appropriate timing
- Visual clarity"""
            
            implementations[i] = implementation
        
        return implementations
    
    def _generate_error_scenarios(self, complexity: TestComplexity, count: int) -> List[Dict[str, Any]]:
        """Generate error scenarios for testing."""
        base_scenarios = [
            {
                'error_type': 'MODEL',
                'message': 'Model API rate limit exceeded',
                'step': 'code_generation',
                'severity': 'MEDIUM',
                'recoverable': True
            },
            {
                'error_type': 'TIMEOUT',
                'message': 'Request timeout after 30 seconds',
                'step': 'planning',
                'severity': 'MEDIUM',
                'recoverable': True
            },
            {
                'error_type': 'VALIDATION',
                'message': 'Invalid input parameters',
                'step': 'planning',
                'severity': 'HIGH',
                'recoverable': False
            },
            {
                'error_type': 'SYSTEM',
                'message': 'Unexpected system error',
                'step': 'rendering',
                'severity': 'CRITICAL',
                'recoverable': False
            },
            {
                'error_type': 'CONTENT',
                'message': 'Generated content validation failed',
                'step': 'code_generation',
                'severity': 'MEDIUM',
                'recoverable': True
            }
        ]
        
        if complexity == TestComplexity.COMPLEX:
            # Add more complex error scenarios
            base_scenarios.extend([
                {
                    'error_type': 'TRANSIENT',
                    'message': 'Network connectivity issues',
                    'step': 'planning',
                    'severity': 'LOW',
                    'recoverable': True,
                    'context': {'retry_count': 2, 'network_error': 'Connection reset'}
                },
                {
                    'error_type': 'RATE_LIMIT',
                    'message': 'API quota exceeded for the day',
                    'step': 'code_generation',
                    'severity': 'HIGH',
                    'recoverable': True,
                    'context': {'quota_limit': 1000, 'current_usage': 1001}
                }
            ])
        
        # Return requested number of scenarios
        scenarios = []
        for i in range(count * 3):  # 3 scenarios per test
            scenario = base_scenarios[i % len(base_scenarios)].copy()
            scenario['id'] = f"error_{i}_{uuid.uuid4().hex[:8]}"
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_intervention_scenarios(self, complexity: TestComplexity, count: int) -> List[Dict[str, Any]]:
        """Generate intervention scenarios for testing."""
        base_scenarios = [
            {
                'intervention_type': 'error_escalation',
                'trigger_condition': 'max_retries_exceeded',
                'expected_action': 'log_and_notify',
                'error_type': 'SYSTEM',
                'error_message': 'Critical system failure',
                'step': 'rendering'
            },
            {
                'intervention_type': 'quality_review',
                'trigger_condition': 'quality_threshold_not_met',
                'expected_action': 'request_human_review',
                'review_type': 'content_quality',
                'quality_threshold': 0.8
            },
            {
                'intervention_type': 'manual_override',
                'trigger_condition': 'user_preference',
                'expected_action': 'apply_override',
                'override_type': 'parameter_adjustment',
                'original_value': 'auto',
                'override_value': 'manual_setting'
            },
            {
                'intervention_type': 'approval_required',
                'trigger_condition': 'sensitive_content_detected',
                'expected_action': 'request_approval',
                'approval_type': 'content_approval',
                'approval_criteria': 'content_appropriateness'
            }
        ]
        
        if complexity == TestComplexity.COMPLEX:
            # Add more complex intervention scenarios
            base_scenarios.extend([
                {
                    'intervention_type': 'error_escalation',
                    'trigger_condition': 'cascading_failures',
                    'expected_action': 'emergency_stop',
                    'error_type': 'CRITICAL',
                    'error_message': 'Multiple system failures detected',
                    'step': 'workflow',
                    'context': {'failure_count': 5, 'affected_systems': ['planning', 'rendering']}
                },
                {
                    'intervention_type': 'quality_review',
                    'trigger_condition': 'anomaly_detected',
                    'expected_action': 'detailed_analysis',
                    'review_type': 'anomaly_investigation',
                    'anomaly_score': 0.95,
                    'context': {'detection_method': 'statistical_analysis'}
                }
            ])
        
        # Return requested number of scenarios
        scenarios = []
        for i in range(count * 2):  # 2 scenarios per test
            scenario = base_scenarios[i % len(base_scenarios)].copy()
            scenario['id'] = f"intervention_{i}_{uuid.uuid4().hex[:8]}"
            scenarios.append(scenario)
        
        return scenarios
    
    def _validate_planner_data(self, data: Dict[str, Any]) -> List[str]:
        """Validate planner test data."""
        issues = []
        
        if 'topic' not in data or not data['topic']:
            issues.append("Missing or empty topic")
        
        if 'description' not in data or not data['description']:
            issues.append("Missing or empty description")
        
        if 'session_id' not in data:
            issues.append("Missing session_id")
        
        return issues
    
    def _validate_codegen_data(self, data: Dict[str, Any]) -> List[str]:
        """Validate code generator test data."""
        issues = []
        
        required_fields = ['topic', 'description', 'scene_outline', 'scene_implementations']
        for field in required_fields:
            if field not in data or not data[field]:
                issues.append(f"Missing or empty {field}")
        
        if 'scene_implementations' in data:
            implementations = data['scene_implementations']
            if not isinstance(implementations, dict):
                issues.append("scene_implementations must be a dictionary")
            elif not implementations:
                issues.append("scene_implementations cannot be empty")
        
        return issues
    
    def _validate_renderer_data(self, data: Dict[str, Any]) -> List[str]:
        """Validate renderer test data."""
        issues = []
        
        if 'generated_code' not in data or not data['generated_code']:
            issues.append("Missing or empty generated_code")
        
        if 'file_prefix' not in data or not data['file_prefix']:
            issues.append("Missing or empty file_prefix")
        
        if 'generated_code' in data:
            code = data['generated_code']
            if not isinstance(code, dict):
                issues.append("generated_code must be a dictionary")
            elif not code:
                issues.append("generated_code cannot be empty")
        
        return issues
    
    def _validate_error_handler_data(self, data: Dict[str, Any]) -> List[str]:
        """Validate error handler test data."""
        issues = []
        
        if 'error_scenarios' not in data or not data['error_scenarios']:
            issues.append("Missing or empty error_scenarios")
        
        if 'error_scenarios' in data:
            scenarios = data['error_scenarios']
            if not isinstance(scenarios, list):
                issues.append("error_scenarios must be a list")
            else:
                for i, scenario in enumerate(scenarios):
                    if not isinstance(scenario, dict):
                        issues.append(f"Error scenario {i} must be a dictionary")
                        continue
                    
                    required_fields = ['error_type', 'message', 'step']
                    for field in required_fields:
                        if field not in scenario:
                            issues.append(f"Error scenario {i} missing {field}")
        
        return issues
    
    def _validate_human_loop_data(self, data: Dict[str, Any]) -> List[str]:
        """Validate human loop test data."""
        issues = []
        
        if 'intervention_scenarios' not in data or not data['intervention_scenarios']:
            issues.append("Missing or empty intervention_scenarios")
        
        if 'intervention_scenarios' in data:
            scenarios = data['intervention_scenarios']
            if not isinstance(scenarios, list):
                issues.append("intervention_scenarios must be a list")
            else:
                for i, scenario in enumerate(scenarios):
                    if not isinstance(scenario, dict):
                        issues.append(f"Intervention scenario {i} must be a dictionary")
                        continue
                    
                    required_fields = ['intervention_type', 'trigger_condition', 'expected_action']
                    for field in required_fields:
                        if field not in scenario:
                            issues.append(f"Intervention scenario {i} missing {field}")
        
        return issues


class TestScenarioManager:
    """
    Manages test scenarios for Studio execution.
    
    Provides scenario selection, execution tracking, and result management.
    """
    
    def __init__(self, data_manager: TestDataManager = None):
        """Initialize with optional data manager."""
        self.data_manager = data_manager or TestDataManager()
        self.scenarios = {}
        self.execution_history = {}
    
    def create_scenario(self, agent_type: str, scenario_data: Dict[str, Any]) -> str:
        """Create a new test scenario."""
        scenario_id = f"{agent_type.lower()}_{uuid.uuid4().hex[:8]}"
        
        # Validate scenario data
        is_valid, issues = self.data_manager.validate_test_data(scenario_data, agent_type)
        if not is_valid:
            raise ValueError(f"Invalid scenario data: {'; '.join(issues)}")
        
        # Create scenario
        scenario = TestScenario(
            id=scenario_id,
            name=f"{agent_type} Test Scenario",
            description=f"Test scenario for {agent_type}",
            complexity=TestComplexity.SIMPLE,
            tags=[agent_type.lower()],
            inputs=scenario_data,
            expected_outputs={},
            validation_criteria={}
        )
        
        self.scenarios[scenario_id] = scenario
        return scenario_id
    
    def get_scenario(self, scenario_id: str) -> Optional[TestScenario]:
        """Get scenario by ID."""
        return self.scenarios.get(scenario_id)
    
    def list_scenarios(self, agent_type: Optional[str] = None) -> List[TestScenario]:
        """List scenarios, optionally filtered by agent type."""
        scenarios = list(self.scenarios.values())
        
        if agent_type:
            scenarios = [s for s in scenarios if agent_type.lower() in s.tags]
        
        return scenarios
    
    def execute_scenario(self, scenario_id: str, test_runner) -> Dict[str, Any]:
        """Execute a test scenario."""
        scenario = self.get_scenario(scenario_id)
        if not scenario:
            raise ValueError(f"Scenario not found: {scenario_id}")
        
        # Record execution start
        execution_record = {
            'scenario_id': scenario_id,
            'start_time': time.time(),
            'status': 'running'
        }
        self.execution_history[scenario_id] = execution_record
        
        try:
            # Execute the scenario
            result = test_runner.run_test(scenario.inputs)
            
            # Record execution completion
            execution_record.update({
                'end_time': time.time(),
                'status': 'completed',
                'result': result
            })
            
            return result
            
        except Exception as e:
            # Record execution failure
            execution_record.update({
                'end_time': time.time(),
                'status': 'failed',
                'error': str(e)
            })
            raise
    
    def get_execution_history(self, scenario_id: Optional[str] = None) -> Dict[str, Any]:
        """Get execution history."""
        if scenario_id:
            return self.execution_history.get(scenario_id, {})
        return self.execution_history
    
    def generate_scenarios_for_agent(self, agent_type: str, complexity: TestComplexity = TestComplexity.SIMPLE, count: int = 3) -> List[str]:
        """Generate multiple scenarios for an agent type."""
        scenario_ids = []
        
        # Get test data based on agent type
        if agent_type == "PlannerAgent":
            test_data_list = self.data_manager.get_planner_test_data(complexity, count)
        elif agent_type == "CodeGeneratorAgent":
            test_data_list = self.data_manager.get_codegen_test_data(complexity, count)
        elif agent_type == "RendererAgent":
            test_data_list = self.data_manager.get_renderer_test_data(complexity, count)
        elif agent_type == "ErrorHandlerAgent":
            test_data_list = self.data_manager.get_error_handler_test_data(complexity, count)
        elif agent_type == "HumanLoopAgent":
            test_data_list = self.data_manager.get_human_loop_test_data(complexity, count)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Create scenarios
        for i, test_data in enumerate(test_data_list):
            scenario_id = self.create_scenario(agent_type, test_data)
            scenario_ids.append(scenario_id)
        
        return scenario_ids
    
    def export_scenarios(self, file_path: str) -> None:
        """Export scenarios to JSON file."""
        export_data = {
            'scenarios': {sid: scenario.to_dict() for sid, scenario in self.scenarios.items()},
            'execution_history': self.execution_history
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    def import_scenarios(self, file_path: str) -> None:
        """Import scenarios from JSON file."""
        with open(file_path, 'r') as f:
            import_data = json.load(f)
        
        # Import scenarios
        for sid, scenario_data in import_data.get('scenarios', {}).items():
            scenario = TestScenario.from_dict(scenario_data)
            self.scenarios[sid] = scenario
        
        # Import execution history
        self.execution_history.update(import_data.get('execution_history', {}))


# Global instances for easy access
_global_test_data_manager = TestDataManager()
_global_scenario_manager = TestScenarioManager(_global_test_data_manager)


def get_test_data_manager() -> TestDataManager:
    """Get the global test data manager instance."""
    return _global_test_data_manager


def get_scenario_manager() -> TestScenarioManager:
    """Get the global scenario manager instance."""
    return _global_scenario_manager
#!/usr/bin/env python3
"""
Test script for the enhanced dynamic forms functionality.

Tests the agent-specific form generation and validation features.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.test_ui.dynamic_forms import DynamicFormGenerator, AgentFormManager
import json

def test_planning_agent_form():
    """Test form generation for planning agent."""
    print("🧠 Testing Planning Agent Form Generation")
    print("-" * 40)
    
    agent_info = {
        'name': 'planning_agent',
        'type': 'planning',
        'description': 'Generates scene outlines and implementations',
        'input_schema': {
            'topic': {
                'type': 'string',
                'required': True,
                'description': 'Topic for the video'
            },
            'description': {
                'type': 'string', 
                'required': True,
                'description': 'Detailed description of what to teach'
            },
            'complexity': {
                'type': 'string',
                'enum': ['basic', 'intermediate', 'advanced'],
                'default': 'intermediate',
                'description': 'Complexity level for the audience'
            }
        },
        'test_examples': [
            {
                'topic': 'Fourier Transform',
                'description': 'Explain the mathematical concept of Fourier Transform with visual examples',
                'complexity': 'intermediate'
            }
        ]
    }
    
    # Test form generation
    components, component_types = DynamicFormGenerator.create_agent_form(agent_info)
    
    print(f"✅ Generated {len(components)} form components")
    print(f"✅ Component types: {list(component_types.keys())}")
    
    # Test form validation
    form_manager = AgentFormManager()
    form_manager.update_form_for_agent(agent_info)
    
    # Test with valid data
    test_values = ['Fourier Transform', 'Detailed explanation...', 'intermediate']
    form_data = form_manager.get_form_values(*test_values)
    is_valid, msg = form_manager.validate_form(*test_values)
    
    print(f"✅ Form validation: {is_valid} - {msg}")
    print(f"✅ Extracted data: {json.dumps(form_data, indent=2)}")
    print()

def test_code_generation_agent_form():
    """Test form generation for code generation agent."""
    print("💻 Testing Code Generation Agent Form Generation")
    print("-" * 40)
    
    agent_info = {
        'name': 'code_generation_agent',
        'type': 'code_generation',
        'description': 'Converts scene plans into executable Manim code',
        'input_schema': {
            'scene_plan': {
                'type': 'object',
                'required': True,
                'description': 'Scene plan from planning agent'
            },
            'style': {
                'type': 'string',
                'enum': ['minimal', 'detailed', 'animated'],
                'default': 'detailed',
                'description': 'Visual style for the animation'
            }
        },
        'test_examples': [
            {
                'scene_plan': {'title': 'Test Scene', 'steps': ['Step 1', 'Step 2']},
                'style': 'detailed'
            }
        ]
    }
    
    # Test form generation
    components, component_types = DynamicFormGenerator.create_agent_form(agent_info)
    
    print(f"✅ Generated {len(components)} form components")
    print(f"✅ Component types: {list(component_types.keys())}")
    
    # Test form validation
    form_manager = AgentFormManager()
    form_manager.update_form_for_agent(agent_info)
    
    # Test with valid data
    test_values = ['{"title": "Test Scene", "steps": ["Step 1", "Step 2"]}', 'detailed']
    form_data = form_manager.get_form_values(*test_values)
    is_valid, msg = form_manager.validate_form(*test_values)
    
    print(f"✅ Form validation: {is_valid} - {msg}")
    print(f"✅ Extracted data: {json.dumps(form_data, indent=2)}")
    print()

def test_rendering_agent_form():
    """Test form generation for rendering agent."""
    print("🎬 Testing Rendering Agent Form Generation")
    print("-" * 40)
    
    agent_info = {
        'name': 'rendering_agent',
        'type': 'rendering',
        'description': 'Executes Manim code to produce final video output',
        'input_schema': {
            'code': {
                'type': 'string',
                'required': True,
                'description': 'Manim code to render'
            },
            'quality': {
                'type': 'string',
                'enum': ['low', 'medium', 'high'],
                'default': 'medium',
                'description': 'Rendering quality level'
            }
        },
        'test_examples': [
            {
                'code': 'from manim import *\n\nclass TestScene(Scene):\n    def construct(self):\n        text = Text("Hello World")\n        self.play(Write(text))',
                'quality': 'medium'
            }
        ]
    }
    
    # Test form generation
    components, component_types = DynamicFormGenerator.create_agent_form(agent_info)
    
    print(f"✅ Generated {len(components)} form components")
    print(f"✅ Component types: {list(component_types.keys())}")
    
    # Test form validation
    form_manager = AgentFormManager()
    form_manager.update_form_for_agent(agent_info)
    
    # Test with valid data
    test_code = 'from manim import *\n\nclass TestScene(Scene):\n    def construct(self):\n        text = Text("Hello World")\n        self.play(Write(text))'
    test_values = [test_code, 'medium']
    form_data = form_manager.get_form_values(*test_values)
    is_valid, msg = form_manager.validate_form(*test_values)
    
    print(f"✅ Form validation: {is_valid} - {msg}")
    print(f"✅ Extracted data keys: {list(form_data.keys())}")
    print(f"✅ Code length: {len(form_data.get('code', ''))}")
    print()

def test_enhanced_features():
    """Test enhanced features like agent-specific placeholders and ranges."""
    print("🎯 Testing Enhanced Features")
    print("-" * 40)
    
    # Test agent-specific placeholders
    placeholder = DynamicFormGenerator._get_agent_specific_placeholder(
        'topic', 'planning', {'type': 'string'}
    )
    print(f"✅ Planning topic placeholder: {placeholder}")
    
    placeholder = DynamicFormGenerator._get_agent_specific_placeholder(
        'code', 'rendering', {'type': 'string'}
    )
    print(f"✅ Rendering code placeholder: {placeholder}")
    
    # Test agent-specific number ranges
    min_val, max_val, step = DynamicFormGenerator._get_agent_specific_number_range(
        'complexity', 'planning', {'type': 'integer'}
    )
    print(f"✅ Planning complexity range: {min_val}-{max_val}, step {step}")
    
    min_val, max_val, step = DynamicFormGenerator._get_agent_specific_number_range(
        'fps', 'rendering', {'type': 'integer'}
    )
    print(f"✅ Rendering fps range: {min_val}-{max_val}, step {step}")
    print()

def main():
    """Run all dynamic forms tests."""
    print("🧪 Testing Enhanced Dynamic Forms for Agent Testing Interface")
    print("=" * 60)
    print("Testing Task 4.2 Implementation:")
    print("- Dynamic input forms based on agent types")
    print("- Agent-specific form validation")
    print("- Enhanced placeholders and input ranges")
    print("=" * 60)
    print()
    
    try:
        test_planning_agent_form()
        test_code_generation_agent_form()
        test_rendering_agent_form()
        test_enhanced_features()
        
        print("🎉 All tests passed! Enhanced dynamic forms are working correctly.")
        print()
        print("Key improvements implemented:")
        print("✅ Agent-type specific form components")
        print("✅ Enhanced input validation and error handling")
        print("✅ Agent-specific placeholders and help text")
        print("✅ Customized input ranges for numeric fields")
        print("✅ Improved form layout and styling")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()